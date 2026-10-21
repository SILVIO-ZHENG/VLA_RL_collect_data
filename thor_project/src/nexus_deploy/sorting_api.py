"""Batch sorting API. Hardware actions remain owned by ControlPlane.

One control process owns its existing durable state file. Request IDs and
callback outbox records live on the deployment, so retries never create a
second robot session. Restart recovery deliberately requires intervention.
"""
from __future__ import annotations

import copy
import hmac
import json
import os
import threading
import time
import uuid
from pathlib import Path
from urllib import request
from urllib.parse import urlsplit

from nexus_deploy.control.errors import TaskError, ValidationError


FINAL = {"completed", "simulated", "failed", "cancelled", "recovery_required"}


class SortingError(Exception):
    def __init__(self, message: str, status: int = 400, reason_code: str = "INVALID_REQUEST"):
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SortingController:
    def __init__(self, plane, settings: dict, *, recover: bool = True):
        self.plane = plane
        self.settings = copy.deepcopy(settings)
        self.profiles = self.settings.get("profiles", {})
        if not isinstance(self.profiles, dict) or not self.profiles:
            raise ValueError("sorting profiles must be a nonempty object")
        self.token = os.environ.get(settings.get("token_env", "G2A_SORTING_TOKEN"), "")
        if len(self.token) < 32 or not self.token.isascii():
            raise ValueError("sorting API token must contain at least 32 ASCII characters")
        self.plane.tasks.register_profiles(self.profiles)
        self.callback_url = settings.get("callback_url") or None
        self.callback_token = os.environ.get(settings.get("callback_token_env", "G2A_CALLBACK_TOKEN"), "")
        if self.callback_url:
            u = urlsplit(self.callback_url)
            if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.fragment:
                raise ValueError("callback_url must be an operator-configured HTTP(S) URL")
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
        self._stop = threading.Event()
        self._thread = None
        self.last_error = None
        if recover:
            with plane._lock:
                plane.tasks.cancel_reset_admission("CONTROL_RESTARTED")
                changed = False
                for job in plane._jobs.values():
                    if job.batch and job.batch.get("status") not in FINAL:
                        job.batch.update(status="recovery_required", start_requested=True,
                                         reason="Control service restarted; inspect and close the old task before resubmission")
                        changed = True
                    for operation in job.task_session.get("agent_reset_commands", {}).values():
                        if operation.get("status") in {"accepted", "stopping", "resetting"}:
                            target = plane._jobs.get(operation.get("target_job_id"))
                            result = target.robot_reset if target else {}
                            if (result.get("command_id") == operation.get("command_id")
                                    and result.get("status") in {"succeeded", "failed"}):
                                operation.update(status=result["status"], finished_at=result.get("finished_at"))
                            else:
                                operation.update(status="unknown", finished_at=time.time(),
                                                 error="Control service restarted; inspect the robot before retrying")
                                plane._mark_robot_reset_required(job, "unknown", "复位期间服务重启；结果需要核实")
                            changed = True
                if changed:
                    plane._persist_state()

    @classmethod
    def from_file(cls, plane, path: Path):
        return cls(plane, json.loads(path.read_text(encoding="utf-8")))

    def authorize(self, authorization: str | None, origin: str | None):
        if origin or not authorization or not hmac.compare_digest(
            authorization.encode(), ("Bearer " + self.token).encode()
        ):
            raise SortingError("A valid service Bearer token and no browser Origin are required", 401)

    def submit(self, payload: dict) -> tuple[dict, bool]:
        try:
            return self.plane.tasks.submit(payload, source="scheduler", callback_url=self.callback_url)
        except TaskError as exc:
            raise SortingError(str(exc), exc.status, exc.reason_code) from exc

    def view(self, job_id: str) -> dict:
        view = self.plane.tasks.view(job_id)
        if view["source"] != "scheduler" or view["request_id"] is None:
            raise SortingError("Sorting task not found", 404)
        return view

    def model(self) -> dict:
        """Read the selected deployment, then verify its model without starting it."""
        p = self.plane
        with p._lock:
            deployment_id = p.tasks.current_deployment_id()
            job = p._jobs.get(deployment_id) if deployment_id else None
            if job is None:
                return dict(state="absent", profile=None, deployment_id=None,
                            model_id=None, model_version=None, reason="尚未选择当前部署",
                            deployment_status=None, runtime_configured=False,
                            completion_gate_ready=False, checked_at=time.time(),
                            hardware_actions_enabled=p.enable_actions)
            config = copy.deepcopy(job.config)
            profiles = sorted(name for name in self.profiles
                              if self._matches_profile(name, config))
        live_healthy = True
        if p.enable_actions:
            model = p.deployment.models.get(str(config.get("model_id", "")))
            live_healthy = bool(model and p._safe_status(p.deployment, model)["healthy"])
        with p._lock:
            current = p._jobs.get(deployment_id)
            if (p.tasks.current_deployment_id() != deployment_id or current is None
                    or current.config != config):
                return dict(state="unhealthy", profile=None, deployment_id=None,
                            model_id=None, model_version=None, reason="当前方案已变化，请重新查询",
                            deployment_status=None, runtime_configured=False,
                            completion_gate_ready=False, checked_at=time.time(),
                            hardware_actions_enabled=p.enable_actions)
            service = copy.deepcopy(current.model_service)
            status = current.status
            configured = current.task_session.get("runtime_configured") is not False
            if (status in {"queued", "validating", "preflight", "deploying"}
                    and not service.get("healthy")):
                state, reason = "loading", "模型仍在准备"
            elif (not service.get("candidate_only") and service.get("status") != "stopped" and
                  (bool(service.get("healthy")) if p.enable_actions else
                   service.get("status") in {"ready", "simulated"}) and live_healthy):
                state, reason = "ready", None
            else:
                state, reason = "unhealthy", "当前部署或模型未就绪"
            if state != "ready":
                p.tasks.observe_model_health(str(config.get("model_id", "")), {"healthy": False})
        return dict(state=state, profile=profiles[0] if len(profiles) == 1 else None,
                    deployment_id=deployment_id, model_id=config.get("model_id"),
                    model_version=config.get("model_version"), reason=reason,
                    deployment_status=status, runtime_configured=configured,
                    completion_gate_ready=bool(config.get("presence_gate", {}).get("enabled")),
                    checked_at=time.time(), hardware_actions_enabled=p.enable_actions)

    def robot(self) -> dict:
        """Report observed robot state; a stale/failed probe never means idle."""
        p = self.plane
        model = self.model()
        with p._lock:
            deployment_id = p.tasks.current_deployment_id()
            current = p._jobs.get(deployment_id) if deployment_id else None
            batches = [job for job in p._jobs.values() if job.batch]
            latest = batches[-1] if batches else None
            active = next((job for job in reversed(list(p._jobs.values()))
                           if p.tasks.busy(job) and
                           (not p.tasks.reset_pending(job) or job.status == "recovery_required")), None)
            target = current or latest
            config = copy.deepcopy(target.config) if target else None
        probe = None
        if p.enable_actions and config and active is None:
            try:
                probe = p.probe_robot({"robot_id": config["robot_id"],
                                       "robot_host": config["robot_host"],
                                       "robot_ssh_port": config["robot_ssh_port"]}, p.csrf_token)
            except Exception:
                # A read failure must fail closed, never report an idle robot.
                probe = None
        with p._lock:
            if p.tasks.current_deployment_id() != deployment_id:
                probe = None
            batches = [job for job in p._jobs.values() if job.batch]
            latest = batches[-1] if batches else None
            active = next((job for job in reversed(list(p._jobs.values()))
                           if p.tasks.busy(job) and
                           (not p.tasks.reset_pending(job) or job.status == "recovery_required")), None)
            gate = p.tasks.reset_gate()
            reset_state = ("ready" if not gate["required"] else
                           "resetting" if gate["status"] == "resetting" else
                           "unknown" if gate["status"] in {"failed", "unknown"} else "not_reset")
            if active and (active.status == "recovery_required" or
                           active.batch.get("status") == "recovery_required"):
                state, reason = "fault", "任务需要人工核实"
            elif active:
                state, reason = "busy", "任务或复位正在进行"
            elif gate["required"] and gate["status"] in {"failed", "unknown"}:
                state, reason = "fault", gate["reason"]
            elif gate["required"] and gate["status"] == "resetting":
                state, reason = "busy", gate["reason"]
            elif not config:
                state, reason = "unknown", "尚未选择机器人"
            elif not p.enable_actions:
                state, reason = "idle", None
            elif probe is None or not probe.get("reachable") or not probe.get("ssh_authenticated"):
                state, reason = "unknown", "机器人状态无法确认"
            elif probe.get("identity", {}).get("match") is False:
                state, reason = "fault", "机器人身份不匹配"
            elif probe.get("client", {}).get("running"):
                state, reason = "busy", "机器人客户端正在运行"
            elif (probe.get("client", {}).get("status") == "idle" and
                  probe.get("client", {}).get("valid") and
                  probe.get("observation", {}).get("ready")):
                state, reason = "idle", None
            else:
                state, reason = "unknown", "机器人客户端或观测状态未就绪"
            profile_ready = False
            profile_error = None
            if model["profile"]:
                try:
                    p.tasks.ready_template(model["profile"])
                    profile_ready = True
                except TaskError as error:
                    profile_error = str(error)
            ready = bool(state == "idle" and reset_state == "ready" and
                         model["state"] == "ready" and model["profile"] and
                         model["runtime_configured"] and model["completion_gate_ready"] and profile_ready and
                         p._workstation["mode"] == "automatic" and p._workstation["accepting"] and
                         not p._workstation.get("persistence_error") and
                         not p._workstation.get("switching_deployment_id") and
                         not any(p.tasks.busy(job) for job in p._jobs.values()) and
                         p.tasks.current_deployment_id() == model["deployment_id"])
            blockers = []
            if state != "idle":
                blockers.append(dict(reason_code="ROBOT_BUSY" if state == "busy" else "ROBOT_STATE_UNKNOWN",
                                     message=reason or "机器人状态尚未确认"))
            if model["state"] != "ready":
                blockers.append(dict(reason_code="MODEL_NOT_READY", message=model.get("reason") or "模型未就绪"))
            if not model["runtime_configured"] or not profile_ready or not model["completion_gate_ready"]:
                blockers.append(dict(reason_code="NOT_READY", message=profile_error or "分拣配置尚未就绪"))
            if p._workstation["mode"] != "automatic":
                blockers.append(dict(reason_code="MODE_CONFLICT", message="Agent 分拣需要自动模式"))
            if not p._workstation["accepting"]:
                blockers.append(dict(reason_code="NOT_ACCEPTING", message=p._workstation.get("reason") or "已暂停接单"))
            if gate["required"]:
                blockers.append(dict(reason_code="RESET_REQUIRED", message=gate["reason"]))
            if p._workstation.get("persistence_error"):
                blockers.append(dict(reason_code="PERSISTENCE_FAILED", message="状态存储故障"))
            if p._workstation.get("switching_deployment_id"):
                blockers.append(dict(reason_code="SCHEME_SWITCHING", message="配置正在切换"))
            return dict(robot_id=config.get("robot_id") if config else None,
                        state=state, reset_state=reset_state,
                        reset_message=gate["reason"] if gate["required"] else
                                      (current.robot_reset.get("message") if current else None),
                        reset_command_id=(current.robot_reset.get("command_id") if current else None),
                        current_task_id=active.id if active and active.batch else None,
                        last_task_id=latest.id if latest else None,
                        ready_for_next_task=ready, reason=reason or (blockers[0]["message"] if blockers else None),
                        blocking_reasons=blockers, mode=p._workstation["mode"], accepting=p._workstation["accepting"],
                        checked_at=probe.get("checked_at") if probe else time.time(),
                        observation_source="robot_probe" if probe else
                                           ("simulation" if not p.enable_actions else "control_state"),
                        reset_operation=self.reset_status(),
                        hardware_actions_enabled=p.enable_actions)

    def _matches_profile(self, name: str, config: dict) -> bool:
        # Preserve the existing read-model shortcut for a literal fixed preset.
        # Admission/handoff still call ready_template and strictly revalidate it.
        # Changed UI trajectory values always use the single backend resolver.
        if self.profiles.get(name) == config:
            return True
        try:
            resolved = self.plane.tasks.validated_profile_config(name)
        except (TaskError, KeyError, TypeError, ValueError):
            return False
        return resolved == config

    def stop(self, job_id: str, payload: dict) -> tuple[dict, bool]:
        try:
            return self.plane.tasks.stop_scheduler(job_id, payload)
        except TaskError as exc:
            raise SortingError(str(exc), exc.status, exc.reason_code) from exc

    def reset(self, payload: dict) -> dict:
        """Reset a loaded deployment, stopping its current task when necessary.

        One reentrant control lock covers admission through the actual reset
        reservation. Never wait for the asynchronous reset worker under it.
        """
        import re
        if not isinstance(payload, dict) or set(payload) != {"command_id"}:
            raise SortingError("Only command_id is accepted", 400)
        command_id = payload["command_id"]
        if not isinstance(command_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", command_id):
            raise SortingError("Invalid command_id", 400)
        p = self.plane
        with p._lock:
            for item in p._jobs.values():
                for operation in item.task_session.get("agent_reset_commands", {}).values():
                    self._reset_view(operation)
            previous = self._find_reset_command(command_id)
            if previous is not None:
                return self._reset_view(previous)
            deployment_id = p.tasks.current_deployment_id()
            job = p._jobs.get(deployment_id) if deployment_id else None
            if job is None:
                raise SortingError("Model is not loaded; load a model first", 503, "MODEL_NOT_LOADED")
            service = job.model_service
            if (job.status == "model_stopped" or service.get("status") == "stopped"):
                raise SortingError("Model is not loaded; load a model first", 503, "MODEL_NOT_LOADED")
            if (job.status in {"queued", "validating", "preflight", "deploying"}
                    or service.get("candidate_only") or not p.tasks.model_available(job)):
                raise SortingError("Loaded model is not ready", 503, "MODEL_NOT_READY")
            if not p.enable_actions or not p.enable_reset:
                raise SortingError("Robot reset is disabled on this control service", 403, "RESET_DISABLED")
            model = p.deployment.models.get(str(job.config.get("model_id", "")))
            model_status = p._safe_status(p.deployment, model) if model is not None else {}
            if not model_status.get("healthy"):
                raise SortingError("Loaded model failed its live health check", 503, "MODEL_NOT_READY")
            if p._workstation.get("persistence_error"):
                raise SortingError("State storage is unavailable", 503, "PERSISTENCE_FAILED")
            if p._workstation.get("switching_deployment_id"):
                raise SortingError("Deployment configuration is changing; retry later", 409, "SCHEME_SWITCHING")
            if job.config.get("hardware_mode") != "execute":
                raise SortingError("Select the robot execution configuration before resetting", 409, "RESET_NOT_CONFIGURED")
            raw_robot = p.deployment.raw.get("robots", {}).get(str(job.config.get("robot_id")), {})
            if (raw_robot.get("reset_enabled") is not True or job.config.get("effector_profile") != "dahuan"
                    or not job.config.get("safety", {}).get("safety_profile_id")):
                raise SortingError("Robot reset or its calibrated safety configuration is missing", 409,
                                   "RESET_NOT_CONFIGURED")
            # A reset already in progress is never interrupted or queued.
            occupied = [item for item in p._jobs.values() if p.tasks.busy(item)]
            if len(occupied) > 1:
                raise SortingError("Multiple robot operations need inspection", 409, "ROBOT_BUSY")
            target = occupied[0] if occupied else job
            if (target.model_service.get("deployment_id", target.id) != deployment_id
                    or any(target.config.get(key) != job.config.get(key) for key in
                           ("robot_id", "robot_host", "robot_ssh_port", "effector_profile"))
                    or target.config.get("safety", {}).get("safety_profile_id")
                    != job.config.get("safety", {}).get("safety_profile_id")):
                raise SortingError("The occupied robot belongs to a different deployment", 409, "ROBOT_BUSY")
            if target.robot_reset.get("status") in {"preparing", "running", "unknown"}:
                raise SortingError("Robot reset is in progress or its result is unknown", 409, "ROBOT_BUSY")
            if target.status in {"task_starting", "starting", "pausing", "stopping", "cancelling", "resetting"}:
                raise SortingError("Robot task is changing state; retry the same command_id later", 409, "ROBOT_BUSY")
            if (target.status == "recovery_required" and target.task_session.get("remote_started")):
                raise SortingError("Robot session state needs independent inspection", 409, "ROBOT_STATE_UNKNOWN")
            if target.status in {"running", "paused"} and not target.task_session.get("remote_started"):
                raise SortingError("Robot session ownership is unconfirmed", 409, "ROBOT_STATE_UNKNOWN")
            operation = dict(command_id=command_id, deployment_id=deployment_id, target_job_id=target.id,
                             robot_id=job.config.get("robot_id"), status="accepted", accepted_at=time.time(),
                             finished_at=None, error=None)
            job.task_session.setdefault("agent_reset_commands", {})[command_id] = operation
            p.tasks.pause_for_agent_reset(operation, job, model_status)
            p._mark_robot_reset_required(target, "pending", "Agent 已请求复位；成功前禁止下一任务")
            # Persist before any pause, stop or motion; a storage failure must
            # not authorize an actuator command or allow a later replay.
            try:
                p.tasks.persist()
                if target.status in {"running", "paused"}:
                    operation["status"] = "stopping"
                    p.tasks.persist()
                    p.pause_deployment(target.id, p.csrf_token)
                    if (target.status != "paused" or not target.task_session.get("remote_started")
                            or not target.task_session.get("safe_hold_available")):
                        raise SortingError("Robot safe hold was not confirmed", 409, "ROBOT_STATE_UNKNOWN")
                    p.stop_deployment(target.id, p.csrf_token, manual_completion=False)
                    if target.task_session.get("remote_started") or target.status not in {
                        "succeeded", "cancelled", "failed", "rejected", "ready_to_start"
                    }:
                        raise SortingError("Robot stop was not confirmed", 409, "ROBOT_STATE_UNKNOWN")
                p.reset_robot_body(target.id, p.csrf_token, command_id=command_id)
                operation["status"] = "resetting"
                p.tasks.persist()
            except Exception as exc:
                p.tasks.cancel_reset_admission("RESET_DISPATCH_FAILED")
                # A failed/unknown operation remains in command history. A
                # caller can inspect it, but retrying its ID never repeats it.
                result = target.robot_reset
                dispatched = result.get("command_id") == command_id
                operation.update(status=("unknown" if dispatched and result.get("status") in
                                         {"preparing", "running"} else
                                         "unknown" if target.status == "recovery_required" and not dispatched
                                         else "failed"),
                                 error=str(exc), reason_code=getattr(exc, "reason_code", "RESET_FAILED"),
                                 finished_at=time.time())
                if operation["status"] == "unknown":
                    p._mark_robot_reset_required(target, "unknown", "复位请求结果不确定；需要核实")
                try:
                    p.tasks.persist()
                except TaskError:
                    pass
                if isinstance(exc, (TaskError, ValidationError, SortingError)):
                    raise SortingError(str(exc), exc.status,
                                       getattr(exc, "reason_code", "RESET_FAILED")) from exc
                raise SortingError("Robot reset dispatch failed; query this command_id", 503, "RESET_FAILED") from exc
            return self._reset_view(operation)

    def _find_reset_command(self, command_id):
        for job in self.plane._jobs.values():
            operation = job.task_session.get("agent_reset_commands", {}).get(command_id)
            if operation is not None:
                return operation
        # Preserve retries for commands accepted by the previous API version.
        for job in self.plane._jobs.values():
            if job.robot_reset.get("command_id") == command_id:
                result = job.robot_reset
                status = result.get("status")
                return dict(command_id=command_id, deployment_id=job.model_service.get("deployment_id", job.id),
                            target_job_id=job.id, robot_id=job.config.get("robot_id"),
                            status=status if status in {"succeeded", "failed", "unknown"} else "resetting",
                            accepted_at=result.get("started_at", job.updated_at),
                            finished_at=result.get("finished_at"), error=None)
        return None

    def _reset_view(self, operation):
        """Read the exact operation without robot probes or additional motion."""
        p = self.plane
        target = p._jobs.get(operation.get("target_job_id"))
        result = target.robot_reset if target else {}
        if (operation["status"] in {"accepted", "stopping", "resetting"}
                and result.get("command_id") == operation["command_id"]):
            state = result.get("status")
            if state in {"succeeded", "failed", "unknown"}:
                operation.update(status=state, finished_at=result.get("finished_at"),
                                 error=result.get("message") if state != "succeeded" else None)
                p.tasks.persist()
        view = copy.deepcopy(operation)
        view["reset_command_id"] = operation["command_id"]
        view["reset_state"] = ("ready" if operation["status"] == "succeeded" else
                               "resetting" if operation["status"] in {"accepted", "stopping", "resetting"}
                               else "unknown")
        view["hardware_actions_enabled"] = p.enable_actions
        view["accepting"] = p._workstation["accepting"]
        return view

    def reset_status(self, command_id=None):
        """Poll a known command or the latest reset; no model or robot access."""
        with self.plane._lock:
            if command_id is not None:
                operation = self._find_reset_command(command_id)
                if operation is None:
                    raise SortingError("Reset command not found", 404, "NOT_FOUND")
            else:
                operations = [item for job in self.plane._jobs.values()
                              for item in job.task_session.get("agent_reset_commands", {}).values()]
                operation = max(operations, key=lambda item: item["accepted_at"]) if operations else None
            return self._reset_view(operation) if operation is not None else dict(
                command_id=None, status="idle", reset_state="not_reset", error=None,
                accepting=self.plane._workstation["accepting"])

    def tick(self):
        """Single worker: never retry an ambiguous start; callbacks are at least once."""
        with self.plane._lock:
            ids = [j.id for j in self.plane._jobs.values()
                   if j.batch.get("request_id") and
                   (j.batch.get("source") or j.task_session.get("source")) == "scheduler"]
        for job_id in ids:
            with self.plane._lock:
                job = self.plane._jobs[job_id]
                current = self.view(job_id)
                if current["status"] in FINAL:
                    changed = job.batch["status"] != current["status"]
                    job.batch["status"] = current["status"]
                    if job.batch.get("callback_url") and not job.batch.get("callback"):
                        changed = True
                        event_id = uuid.uuid4().hex
                        job.batch["callback"] = {
                            "event_id": event_id, "status": "pending", "attempts": 0, "next_at": 0,
                            "payload": {"event_id": event_id, "event_type": "sorting." + current["status"],
                                        "task": current},
                        }
                    if changed:
                        self.plane._persist_state()
            self._deliver(job_id)

    def _deliver(self, job_id):
        with self.plane._lock:
            job = self.plane._jobs[job_id]
            callback = job.batch.get("callback")
            if not callback or callback["status"] == "delivered" or callback["next_at"] > time.time():
                return
            callback["attempts"] += 1
            callback["next_at"] = time.time() + min(300, 2 ** min(callback["attempts"], 9))
            self.plane._persist_state()
            data = json.dumps(callback["payload"], ensure_ascii=False).encode()
            headers = {"Content-Type": "application/json", "Idempotency-Key": callback["event_id"]}
            if self.callback_token:
                headers["Authorization"] = "Bearer " + self.callback_token
            req = request.Request(job.batch["callback_url"], data=data, headers=headers, method="POST")
        try:
            with self.opener.open(req, timeout=5) as response:
                if not 200 <= response.status < 300:
                    raise OSError(f"Callback HTTP status {response.status}")
        except Exception as error:
            with self.plane._lock:
                callback["last_error"] = type(error).__name__
                self.plane._persist_state()
            return
        with self.plane._lock:
            callback.update(status="delivered", delivered_at=time.time(), last_error=None)
            self.plane._persist_state()

    def start(self):
        def run():
            while not self._stop.is_set():
                try:
                    self.tick()
                    self.last_error = None
                except Exception as error:
                    self.last_error = type(error).__name__
                self._stop.wait(0.5)
        self._thread = threading.Thread(target=run, name="SortingBatchWorker", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
