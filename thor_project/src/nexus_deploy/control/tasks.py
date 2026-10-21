# 中文：
# 工作站任务状态机：管理操作员/调度准入、单任务所有权、幂等命令和
# 复位屏障。本模块不计算插值，但它决定动作线程是否能被启动或再次恢复。
# 异常回执用 unknown/recovery 状态保守处理，而不是重复提交物理开始命令。
# English:
# Workstation task state machine controls admission, single-task ownership, idempotency,
# and reset barriers. It does no interpolation, but determines whether execution can
# start/resume. Ambiguous physical acknowledgements become unknown/recovery, not retries.
"""Single-process task admission and read model. There is no business queue.

The ControlPlane lock serializes admission, workstation changes and legacy
commands. Deployment preparation is separate from accepted task execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from nexus_deploy.control.errors import TaskError
from nexus_deploy.control.models import DeploymentJob

ACTIVE = {"starting", "task_starting", "running", "pausing", "paused", "resetting",
          "stopping", "cancelling", "recovery_required"}
FINAL = {"completed", "simulated", "cancelled", "failed"}


APPLIED_TRAJECTORY_FIELDS = (
    "inference_mode", "rtc_inference_mode", "execution_fps", "servo_fps", "action_interpolation", "chunk_smoothing_mode",
    "arm_tracking_mode", "arm_tracking_omega", "rtc_prefix_mode",
    "prediction_horizon", "quality_horizon", "request_lead_margin_steps",
    "min_replan_after_steps", "latency_window_size", "request_timeout_s",
    "min_rtc_blend_steps", "warmup_requests", "execution_preflight_requests",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _config_mismatches(deployment, profile, prefix=""):
    """Describe exact configuration differences without changing either input."""
    differences = []
    for key in sorted(deployment.keys() | profile.keys()):
        field = f"{prefix}.{key}" if prefix else key
        deployment_value, profile_value = deployment.get(key), profile.get(key)
        if isinstance(deployment_value, dict) and isinstance(profile_value, dict):
            differences.extend(_config_mismatches(deployment_value, profile_value, field))
        elif key not in deployment or key not in profile or deployment_value != profile_value:
            differences.append(dict(field=field, deployment_value=copy.deepcopy(deployment_value),
                                    profile_value=copy.deepcopy(profile_value)))
    return differences


class TaskService:
    def __init__(self, plane):
        self.plane = plane
        self.profiles = {}
        # A reset may restore only admission it paused itself. This capability
        # is deliberately process-local, so a service restart cannot replay it.
        self._agent_reset_restore = None

    def cancel_reset_admission(self, reason):
        with self.plane._lock:
            guard = self._agent_reset_restore
            self._agent_reset_restore = None
            if guard is not None:
                guard["operation"].update(admission_restored=False, admission_restore_reason=reason)

    def observe_model_health(self, model_id, status):
        with self.plane._lock:
            guard = self._agent_reset_restore
            if guard is None or guard["model_id"] != model_id:
                return
            if not status.get("healthy"):
                self.cancel_reset_admission("MODEL_NOT_READY")
            elif guard["model_pid"] is not None and status.get("pid") != guard["model_pid"]:
                self.cancel_reset_admission("MODEL_CHANGED")

    def pause_for_agent_reset(self, operation, deployment, model_status):
        p = self.plane
        was_accepting = p._workstation["accepting"] is True
        operation.update(restore_admission_requested=was_accepting, admission_restored=False,
                         admission_restore_reason="RESET_IN_PROGRESS" if was_accepting else "ADMISSION_WAS_CLOSED")
        self._agent_reset_restore = (dict(
            operation=operation, command_id=operation["command_id"], target_job_id=operation["target_job_id"],
            deployment_id=deployment.id, config=copy.deepcopy(deployment.config),
            mode=p._workstation["mode"], model_id=str(deployment.config["model_id"]),
            model_pid=model_status.get("pid"),
        ) if was_accepting else None)
        p._workstation.update(accepting=False, reason="Agent 已请求机器人复位；新任务已暂停")

    def finish_agent_reset_admission(self, reset_job, operation):
        """Consume the reset's admission capability once, after physical success."""
        p = self.plane
        with p._lock:
            guard = self._agent_reset_restore
            if (guard is None or guard["command_id"] != operation["command_id"]
                    or guard["target_job_id"] != reset_job.id):
                return
            self._agent_reset_restore = None
            if operation["status"] != "succeeded":
                operation.update(admission_restored=False, admission_restore_reason="RESET_FAILED")
                return
            try:
                w = p._workstation
                if w.get("persistence_error"):
                    raise TaskError("PERSISTENCE_FAILED", "状态存储故障；保持停止接单", 503)
                if w.get("switching_deployment_id"):
                    raise TaskError("SCHEME_SWITCHING", "配置正在切换；保持停止接单", 409)
                if self.current_deployment_id() != guard["deployment_id"] or w["mode"] != guard["mode"]:
                    raise TaskError("CONFIG_CHANGED", "部署或模式已变化；保持停止接单", 409)
                current = p._jobs[guard["deployment_id"]]
                if (current.config != guard["config"] or current.model_service.get("candidate_only")
                        or current.task_session.get("runtime_configured") is False):
                    raise TaskError("CONFIG_CHANGED", "运行配置已变化；保持停止接单", 409)
                if not self.model_available(current) or current.model_service.get("status") == "stopped":
                    raise TaskError("MODEL_NOT_READY", "模型尚未就绪；保持停止接单", 503)
                self.check_idle()
                model = p.deployment.models[str(current.config["model_id"])]
                live = p._safe_status(p.deployment, model)
                if not live.get("healthy"):
                    raise TaskError("MODEL_NOT_READY", "模型实时健康检查未通过；保持停止接单", 503)
                if guard["model_pid"] is not None and live.get("pid") != guard["model_pid"]:
                    raise TaskError("MODEL_CHANGED", "模型进程已变化；保持停止接单", 409)
                if w["mode"] == "automatic":
                    self.require_ready_profile()
                operation.update(admission_restored=True, admission_restore_reason="RESTORED")
                w.update(accepting=True, reason="")
                # The reset worker persists this together with completion,
                # while still holding the same lock, before the next request.
            except Exception as error:
                operation.update(admission_restored=False,
                                 admission_restore_reason=getattr(error, "reason_code", "MODEL_NOT_READY"))
                p._workstation.update(accepting=False, reason=f"复位已成功，暂不接单：{error}")

    def state(self):
        with self.plane._lock:
            p = self.plane
            deployments = self.console_deployments()
            current = self.current_deployment_id()
            active_tasks = [j.id for j in p._jobs.values()
                            if self.busy(j) and (j.batch or j.task_session.get("accepted"))]
            return dict(p._workstation, hardware_actions_enabled=p.enable_actions,
                        active_deployment_id=current, active_tasks=active_tasks,
                        current_task_id=active_tasks[0] if active_tasks else None,
                        robot_gate=copy.deepcopy(self.reset_gate()),
                        profiles=self.profile_views(), console_deployments=deployments,
                        landing_url=self.landing_url(deployments))

    def current_deployment_id(self):
        """Existing installations with exactly one ready deployment stay usable."""
        selected = self.plane._workstation.get("active_deployment_id")
        if selected:
            return selected if selected in self.plane._jobs else None
        ready = [j.id for j in self.plane._jobs.values()
                 if not j.batch and not self.busy(j) and
                 j.status in {"ready_to_start", "succeeded"} and
                 j.task_session.get("runtime_configured") is not False and
                 not j.model_service.get("candidate_only") and self.model_available(j)]
        return ready[0] if len(ready) == 1 else None

    def model_available(self, job):
        return bool(job.model_service.get("healthy") or
                    (not self.plane.enable_actions and
                     job.model_service.get("status") in {"simulated", "ready"}))

    def _unavailable_deployment_reason(self, job):
        if job.status == "model_stopped" or job.model_service.get("status") == "stopped":
            return "当前部署的模型已关闭；请重新加载模型并保存应用配置"
        if not self.model_available(job):
            return "当前部署的模型服务未就绪；请核实模型后重新保存应用配置"
        return "当前部署尚未处于可用的待运行或已结束状态"

    def console_deployments(self):
        """Current runtime and unfinished configuration only; never a task queue."""
        result = []
        current_id = self.current_deployment_id()
        for job in reversed(list(self.plane._jobs.values())):
            if (current_id and job.id != current_id and not self.busy(job)
                    and not job.model_service.get("candidate_only")):
                continue
            if self.busy(job):
                phase = "active" if job.batch or job.task_session.get("accepted") else "recovery"
            elif job.batch:
                continue
            elif job.status in {"queued", "validating", "preflight", "deploying"}:
                phase = "preparing"
            elif job.status == "ready_to_start" and job.model_service.get("candidate_only"):
                phase = "candidate"
            elif job.status in {"ready_to_start", "waiting_approval", "succeeded"}:
                if self.model_available(job):
                    phase = "standby"
                elif job.id == current_id:
                    phase = "unavailable"
                else:
                    continue
            elif job.id == current_id:
                # Keep the applied deployment visible for explanation/recovery;
                # unavailable never grants standby or scheduler admission.
                phase = "unavailable"
            else:
                continue
            result.append(dict(id=job.id, phase=phase, status=job.status,
                               model_status=job.model_service.get("status"),
                               reason=self._unavailable_deployment_reason(job) if phase == "unavailable" else "",
                               is_current=job.id == current_id,
                               runtime_configured=job.task_session.get("runtime_configured") is not False,
                               robot_id=job.config.get("robot_id"),
                               model_id=job.config.get("model_id"),
                               model_version=job.config.get("model_version"),
                               effector_profile=job.config.get("effector_profile"),
                               runtime_summary={key: job.config.get(key) for key in
                                                ("task", "inference_mode", "hardware_mode", "duration_s")}
                               | {"safety_profile_id": job.config.get("safety", {}).get("safety_profile_id"),
                                  "workspace_safety_enabled": job.config.get("safety", {}).get(
                                      "workspace_safety_enabled")},
                               source=job.batch.get("source", job.task_session.get("source", "operator")),
                               deployment_id=job.model_service.get("deployment_id", job.id)))
        return result

    def landing_url(self, deployments):
        active = [item for item in deployments if item["phase"] in {"active", "recovery"}]
        if len(active) == 1:
            return f"/monitor?id={active[0]['id']}"
        current = self.current_deployment_id()
        if not active and current:
            return f"/monitor?id={current}"
        return "/monitor"

    @staticmethod
    def busy(job):
        return (job.status in ACTIVE or job.task_session.get("status") in ACTIVE
                or bool(job.task_session.get("remote_started"))
                or job.robot_reset.get("status") in {"preparing", "running", "unknown"}
                or job.batch.get("status") == "recovery_required")

    def reset_gate(self):
        return self.plane._workstation["reset_gate"]

    # 中文：
    # 下一次物理任务前要求前一次复位得到确认；不能只根据业务完成跳过。
    # English:
    # Require confirmed reset before the next physical task; business completion is insufficient.
    def require_reset_complete(self):
        gate = self.reset_gate()
        if gate["required"]:
            raise TaskError("RESET_REQUIRED", "机器人尚未成功复位；手动和调度任务均不能启动", 409)

    @staticmethod
    def reset_pending(job):
        """A stopped physical batch retains the task slot until reset succeeds."""
        return (bool(job.batch.get("reset_required"))
                and job.robot_reset.get("status") != "succeeded")

    @staticmethod
    def reset_state(job):
        status = job.robot_reset.get("status")
        if status in {"preparing", "running"}:
            return "resetting"
        if status == "succeeded":
            return "ready"
        if job.task_session.get("reset_resolved_by"):
            return "ready"
        if TaskService.reset_pending(job):
            return "unknown" if status in {"unknown", "failed"} else "not_reset"
        if status in {"unknown", "failed"} or job.status == "recovery_required":
            return "unknown"
        return "ready" if job.config.get("hardware_mode") != "execute" else "unknown"

    # 中文：
    # 统一检查当前占用和复位门禁，避免操作员与调度同时取得机器人。
    # English:
    # Check occupancy/reset gates uniformly so operator and scheduler cannot both acquire the robot.
    def check_idle(self, robot_id=None, exclude=None, *, require_reset=True):
        p = self.plane
        if any(self.busy(j) and j.id != exclude and
               (robot_id is None or j.config.get("robot_id") == robot_id) for j in p._jobs.values()):
            raise TaskError("ROBOT_BUSY", "机器人已有任务占用或需要人工核实；请求不会排队")
        if any(owner != exclude for rid, owner in p._reserved_robots.items()
               if robot_id is None or rid == robot_id):
            raise TaskError("ROBOT_BUSY", "机器人执行名额已占用；请求不会排队")
        if require_reset:
            self.require_reset_complete()

    def check_entry(self, source):
        w = self.plane._workstation
        if w.get("switching_deployment_id"):
            raise TaskError("SCHEME_SWITCHING", "当前正在应用配置；新任务不会排队", 503)
        if w.get("persistence_error"):
            raise TaskError("PERSISTENCE_FAILED", "状态持久化失败，必须核实后重启服务", 503)
        if w["mode"] != ("automatic" if source == "scheduler" else "debug"):
            raise TaskError("MODE_CONFLICT", "当前操作模式不允许此入口启动任务")
        if not w["accepting"]:
            raise TaskError("NOT_ACCEPTING", w["reason"] or "工作站已停止接单", 503)

    def configure(self, payload, token):
        p = self.plane
        p._require_csrf(token)
        if not isinstance(payload, dict):
            raise TaskError("INVALID_REQUEST", "请求体必须是对象", 400)
        if set(payload) - {"mode", "accepting"}:
            raise TaskError("INVALID_REQUEST", "只接受 mode 和 accepting", 400)
        if "mode" in payload and payload["mode"] not in {"debug", "automatic"}:
            raise TaskError("INVALID_REQUEST", "mode 必须为 debug 或 automatic", 400)
        if "accepting" in payload and not isinstance(payload["accepting"], bool):
            raise TaskError("INVALID_REQUEST", "accepting 必须为布尔值", 400)
        with p._lock:
            if p._workstation.get("persistence_error"):
                raise TaskError("PERSISTENCE_FAILED", "状态存储故障，需要人工核实", 503)
            if p._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "配置应用尚未结束", 409)
            target_mode = payload.get("mode", p._workstation["mode"])
            changing_mode = target_mode != p._workstation["mode"]
            reopening = payload.get("accepting") is True and not p._workstation["accepting"]
            will_accept = payload.get("accepting", p._workstation["accepting"])
            ready_jobs = []
            if target_mode == "automatic" and (changing_mode or reopening):
                ready_jobs = self.require_ready_profile()
            if target_mode == "debug" and (reopening or (changing_mode and will_accept)):
                current = p._jobs.get(self.current_deployment_id())
                if current is None or not self.model_available(current):
                    raise TaskError("NOT_READY", "请先应用配置并确认模型就绪", 503)
            if target_mode == "automatic":
                execute_on_start = bool(ready_jobs) and all(
                    item.config.get("hardware_mode") == "execute" for item in ready_jobs
                )
            else:
                current = p._jobs.get(self.current_deployment_id())
                execute_on_start = bool(current and current.config.get("hardware_mode") == "execute")
            defer_reset = bool(p.enable_actions and execute_on_start)
            if changing_mode:
                self.check_idle(require_reset=will_accept and not defer_reset)
            if p.enable_actions and (changing_mode or reopening):
                # A recent read-only probe is required to hand over control or
                # reopen admission. Neither operation contacts robot hardware.
                targets = ready_jobs if target_mode == "automatic" else [
                    p._jobs[item["id"]] for item in self.console_deployments()
                    if item["phase"] == "standby"
                ]
                if not targets:
                    raise TaskError("NOT_READY", "请先应用配置并确认模型就绪", 503)
                for key in {(str(j.config["robot_id"]), str(j.config["robot_host"]),
                             int(j.config["robot_ssh_port"])) for j in targets}:
                    probe = p._probe_leases.get(key, {})
                    if not (probe.get("reachable") and probe.get("ssh_authenticated") and
                            probe.get("client", {}).get("status") == "idle" and
                            time.time() - probe.get("checked_at", 0) <= 30):
                        raise TaskError("ROBOT_STATE_UNKNOWN", "请先检测机器人并确认空闲")
            if reopening:
                self.check_idle(require_reset=not defer_reset)
            if "accepting" in payload or "mode" in payload:
                # Even an explicit false while already paused revokes the
                # automatic restore; an operator's later intent wins.
                self.cancel_reset_admission("OPERATOR_CHANGED_ADMISSION")
            p._workstation.update(payload)
            if "accepting" in payload:
                p._workstation["reason"] = "" if payload["accepting"] else "操作员已暂停新任务"
            self.persist()
            return self.state()

    def persist(self):
        try:
            self.plane._persist_state()
        except OSError as exc:
            self.cancel_reset_admission("PERSISTENCE_FAILED")
            self.plane._workstation.update(accepting=False, persistence_error=True,
                                           reason="状态持久化失败，需要人工核实")
            raise TaskError("PERSISTENCE_FAILED", "状态持久化失败；未授权新的动作", 503) from exc

    def register_profiles(self, profiles):
        self.profiles = copy.deepcopy(profiles)

    def activate(self, job_id, token, *, priority_reserved=False):
        """Activate one saved scheme; model replacement never starts a robot task."""
        p = self.plane
        p._require_csrf(token)
        with p._lock:
            if p._workstation.get("persistence_error"):
                raise TaskError("PERSISTENCE_FAILED", "状态存储故障，需要人工核实", 503)
            switching = p._workstation.get("switching_deployment_id")
            if switching and not (priority_reserved and switching == job_id and
                                  p._workstation.get("priority_apply", {}).get("target_id") == job_id and
                                  p._workstation.get("priority_apply", {}).get("status") == "activating"):
                raise TaskError("SCHEME_SWITCHING", "配置应用尚未结束", 409)
            job = p._jobs.get(job_id)
            if job is None or job.batch or job.status not in {"ready_to_start", "succeeded"}:
                raise TaskError("NOT_READY", "请先保存配置并完成预检", 503)
            if job.task_session.get("runtime_configured") is False:
                raise TaskError("NOT_READY", "请先保存运行参数并完成预检", 503)
            self.check_idle()
            current_id = self.current_deployment_id()
            if current_id == job_id:
                raise TaskError("ALREADY_ACTIVE", "此配置已经生效", 409)
            if current_id and not job.model_service.get("candidate_only"):
                raise TaskError("SCHEME_NOT_ACTIVE", "旧任务配置不可重新应用；请从模型准备页建立新配置", 409)
            previous = p._jobs.get(current_id) if current_id else None
            if previous and previous.id != job.id:
                hardware = ("robot_id", "robot_host", "robot_ssh_port", "effector_profile")
                if any(previous.config.get(key) != job.config.get(key) for key in hardware):
                    raise TaskError("HARDWARE_CONFLICT", "当前机器人或夹爪硬件不同；请走现场维护流程", 409)
            validation = p.validate(p._saved_config_validation_payload(job.config))
            if not validation["valid"] or validation["normalized"] != job.config:
                raise TaskError("NOT_READY", "运行参数尚未通过严格校验", 503)
            if p.enable_actions:
                # Read-only probe at the switch boundary; a prior UI probe
                # can be stale even while its lease remains valid.
                probe = p.probe_robot({"robot_id": job.config["robot_id"],
                                       "robot_host": job.config["robot_host"],
                                       "robot_ssh_port": job.config["robot_ssh_port"]}, p.csrf_token)
                clients = probe.get("observation", {}).get("external_gdk_clients", [])
                if job.config.get("hardware_mode") == "execute" and clients:
                    pids = "、".join(str(item.get("pid", "?")) for item in clients)
                    raise TaskError("GDK_IN_USE",
                                    f"真机执行要求独占 GDK；占用进程 PID {pids}。"
                                    "请先由对应程序释放 GDK 会话并重新检测；本次真机请求未执行", 409)
                if (not probe.get("reachable") or not probe.get("ssh_authenticated") or
                        probe.get("client", {}).get("status") != "idle" or
                        not probe.get("observation", {}).get("ready")):
                    raise TaskError("ROBOT_STATE_UNKNOWN", "当前机器人实例未确认空闲或只读观测未就绪", 409)
            self.cancel_reset_admission("CONFIG_CHANGED")
            p._workstation.update(accepting=False, reason="配置应用中；新任务已暂停",
                                  switching_deployment_id=job_id, switch_error="")
            self.persist()
            threading.Thread(target=self._activate_worker, args=(job_id, current_id),
                             name="SchemeActivation", daemon=True).start()
            return self.state()

    def _activate_worker(self, job_id, previous_id):
        p = self.plane
        try:
            with p._lock:
                job = p._jobs[job_id]
                config = copy.deepcopy(job.config)
                previous = p._jobs.get(previous_id) if previous_id else None
            target = p.deployment.models[str(config["model_id"])]
            if p.enable_actions:
                variant = target.version(config["effector_profile"], config["model_version"])
                for arg in variant.real_args:
                    if arg.startswith(("--checkpoint=", "--tokenizer-path=")):
                        asset = Path(arg.split("=", 1)[1])
                        if not (target.repo / asset).exists():
                            raise TaskError("NOT_READY", f"候选模型资产不存在：{asset}", 503)
                # There is one task slot. Replace model services only after
                # admission has been durably paused and the robot is idle.
                to_stop = {str(previous.config["model_id"])} if previous and previous.id != job.id else set()
                if p.deployment.running_pid(target) is not None:
                    running_version = p.deployment.running_model_version(target)
                    running_effector = p.deployment.running_model_effector_profile(target)
                    running_mock = p.deployment.running_model_mock(target)
                    running_rtc = p.deployment.running_model_rtc_inference_mode(target)
                    if (running_version != config["model_version"] or
                            running_effector != config["effector_profile"] or
                            running_rtc != config["rtc_inference_mode"] or
                            (config["server_mode"] in {"mock", "real"} and
                             running_mock != (config["server_mode"] == "mock")) or
                            not p._safe_status(p.deployment, target)["healthy"]):
                        to_stop.add(str(config["model_id"]))
                    else:
                        to_stop.discard(str(config["model_id"]))
                for model_id in to_stop:
                    model = p.deployment.models[model_id]
                    p.deployment.stop_model(model)
                    with p._lock:
                        for item in p._jobs.values():
                            if item.config.get("model_id") == model_id:
                                item.model_service.update(status="stopped", healthy=False,
                                                          owned_by_job=False, pgid=None)
                        p._model_resources.pop(model_id, None)
                        p._reserved_models.pop(model_id, None)
                        p._reserved_gpus = {gpu: owner for gpu, owner in p._reserved_gpus.items()
                                            if p._jobs.get(owner, job).config.get("model_id") != model_id}
                        self.persist()
                running = p.deployment.running_pid(target)
                healthy = p._safe_status(p.deployment, target)["healthy"] if running else False
                same_model = (running is not None and healthy and
                              p.deployment.running_model_version(target) == config["model_version"] and
                              p.deployment.running_model_effector_profile(target) == config["effector_profile"] and
                              p.deployment.running_model_rtc_inference_mode(target) == config["rtc_inference_mode"] and
                              (config["server_mode"] == "reuse" or
                               p.deployment.running_model_mock(target) == (config["server_mode"] == "mock")))
                if same_model:
                    with p._lock:
                        job.model_service.update(status="ready", healthy=True, candidate_only=False,
                                                 phase="healthy", owned_by_job=False)
                        p._mark_ready_to_start(job)
                else:
                    p._prepare_model(job_id)
                    with p._lock:
                        job.model_service["candidate_only"] = False
                        p._reserved_models[str(config["model_id"])] = job_id
                        p._mark_ready_to_start(job)
            else:
                with p._lock:
                    job.model_service.update(status="simulated", healthy=False, candidate_only=False)
                    p._mark_ready_to_start(job)
            with p._lock:
                p._workstation.update(active_deployment_id=job_id, switching_deployment_id=None,
                                      accepting=False, reason="配置已应用；请核实后恢复新任务", switch_error="")
                if p._workstation.get("priority_apply", {}).get("target_id") == job_id:
                    p._workstation["priority_apply"].update(status="applied", finished_at=time.time())
                self.persist()
        except Exception as error:
            with p._lock:
                candidate = p._jobs.get(job_id)
                if candidate is not None:
                    candidate.model_service.update(status="candidate", healthy=False,
                                                   candidate_only=True, phase="candidate",
                                                   phase_label="应用失败，待重新核实")
                    self.plane._event(candidate, "ready_to_start", "配置应用失败；模型状态待核实", 86)
                p._workstation.update(switching_deployment_id=None, accepting=False,
                                      reason="配置应用失败；请核实模型后再恢复新任务",
                                      switch_error=str(error)[:400])
                if p._workstation.get("priority_apply", {}).get("target_id") == job_id:
                    p._workstation["priority_apply"].update(
                        status="failed", error=str(error)[:400], finished_at=time.time())
                try:
                    self.persist()
                except TaskError:
                    pass

    def require_ready_profile(self):
        ready = []
        failures = []
        for name in self.profiles:
            try:
                job = self.ready_template(name)
            except TaskError as exc:
                failures.append(f"方案 {name}：{exc}")
                continue
            if not job.config.get("presence_gate", {}).get("enabled"):
                failures.append(f"方案 {name}：未启用包裹检测完成条件")
                continue
            if self.plane.enable_actions:
                model = self.plane.deployment.models[str(job.config["model_id"])]
                status = self.plane._safe_status(self.plane.deployment, model)
                if not status["healthy"]:
                    reason = "实时健康检查未通过" if status.get("running") else "当前未确认运行"
                    failures.append(f"方案 {name}：模型 {job.config['model_id']} {reason}")
                    continue
            ready.append(job)
        if not ready:
            detail = "；".join(failures) if failures else "尚未配置固定方案"
            raise TaskError("NOT_READY", f"自动模式尚未就绪：{detail}", 503)
        return ready

    def validated_profile_config(self, profile):
        """One source of truth for UI handoff, Agent readiness and admission.

        Resolve only an operator's applied, saved deployment. Revalidate the
        resulting configuration and never modify the preset or deployment.
        Robot/model/task identity and safety fields still match exactly. The
        operator's applied hardware access mode is revalidated; real/reuse
        lifecycle choices match only for a confirmed real model service.
        """
        if profile not in self.profiles:
            raise TaskError("NOT_READY", "方案不存在", 503)
        p = self.plane
        with p._lock:
            payload = p._saved_config_validation_payload(self.profiles[profile])
            baseline = p.validate(payload)
            if not baseline["valid"]:
                raise TaskError("NOT_READY", "固定方案尚未通过严格校验", 503)
            selected = p._jobs.get(self.current_deployment_id())
            identity_fields = ("robot_id", "robot_host", "robot_ssh_port", "model_id",
                               "effector_profile", "task")
            if (selected is not None and not selected.batch
                    and not selected.model_service.get("candidate_only")
                    and selected.task_session.get("runtime_configured") is not False
                    and all(selected.config.get(key) == baseline["normalized"].get(key)
                            for key in identity_fields)):
                # The applied model version, like the inference mode and timing,
                # is selected in the web deployment. Scheduler requests cannot
                # change it; the final exact-match check still uses this job.
                payload["model_version"] = selected.config["model_version"]
                for key in APPLIED_TRAJECTORY_FIELDS:
                    if key in selected.config:
                        payload[key] = copy.deepcopy(selected.config[key])
                # The operator selects observation versus physical execution
                # when applying the deployment. Scheduler requests cannot set
                # this mode, and final validation still enforces execute gates.
                payload["hardware_mode"] = selected.config.get("hardware_mode")
                # Loading and reusing a confirmed real service are deployment
                # lifecycle choices; scheduler tasks keep the applied choice.
                # Mock and unknown live service types never gain equivalence.
                modes = {selected.config.get("server_mode"), baseline["normalized"].get("server_mode")}
                if modes == {"real", "reuse"}:
                    model = p.deployment.models.get(str(selected.config["model_id"]))
                    if not p.enable_actions or (model is not None and
                            p.deployment.running_model_mock(model) is False):
                        payload["server_mode"] = selected.config["server_mode"]
            result = p.validate(payload)
            if not result["valid"]:
                raise TaskError("NOT_READY", "已应用运行参数尚未通过严格校验", 503)
            return result["normalized"]


    def ready_template(self, profile):
        p = self.plane
        config = self.validated_profile_config(profile)
        current = self.current_deployment_id()
        for j in reversed(list(p._jobs.values())):
            if (j.id == current and not j.batch and (j.status == "succeeded" or
                                 (j.status == "ready_to_start" and not j.task_session.get("accepted"))) and
                    not self.busy(j) and j.config == config and
                    j.task_session.get("runtime_configured") is not False and
                    not j.model_service.get("candidate_only") and
                    self.model_available(j)):
                return j
        # 中文：仅解释既有精确匹配为什么失败；不修改配置、不探测现场，也不改变准入条件。
        # English: Explain an existing exact-match failure without edits, live probes, or admission changes.
        job = p._jobs.get(current)
        mismatches = []
        if job is None:
            reason = "尚未选择可用的当前部署；请先应用配置并确认模型就绪"
        elif job.batch:
            reason = "当前部署是任务记录，不能作为固定方案的部署模板"
        elif self.busy(job):
            reason = "当前部署仍被任务占用或等待核实；任务结束后才能交还调度"
        elif job.task_session.get("runtime_configured") is False or job.model_service.get("candidate_only"):
            reason = "当前配置尚未保存并应用；候选配置不能交还调度"
        elif job.status == "model_stopped" or not self.model_available(job):
            reason = self._unavailable_deployment_reason(job)
        elif not (job.status == "succeeded" or
                  (job.status == "ready_to_start" and not job.task_session.get("accepted"))):
            reason = "当前部署尚未处于可用的待运行或已结束状态"
        else:
            mismatches = _config_mismatches(job.config, config)
            reason = "当前部署与固定方案的运行配置不一致：" + "、".join(
                item["field"] for item in mismatches)
        error = TaskError("NOT_READY", reason, 503)
        error.config_mismatches = mismatches
        raise error

    def profile_views(self):
        result = []
        for name in self.profiles:
            try:
                job = self.ready_template(name)
                if not job.config.get("presence_gate", {}).get("enabled"):
                    raise TaskError("NOT_READY", "自动方案需要包裹检测完成条件", 503)
                result.append(dict(profile=name, ready=True, deployment_id=job.id,
                                   revision=digest(job.config), reason_code=None,
                                   reason="", config_mismatches=[]))
            except TaskError as exc:
                result.append(dict(profile=name, ready=False, reason_code=exc.reason_code,
                                   reason=str(exc), config_mismatches=getattr(exc, "config_mismatches", [])))
        return result

    def admit_existing(self, job, source, *, defer_reset_gate=False):
        """Caller holds the lock through reserve, durable intent and dispatch."""
        self.check_entry(source)
        current = self.current_deployment_id()
        deployment_id = job.model_service.get("deployment_id", job.id)
        if current is None or deployment_id != current:
            raise TaskError("SCHEME_NOT_ACTIVE", "请先将该方案设为当前方案", 409)
        if job.task_session.get("runtime_configured") is False:
            raise TaskError("NOT_READY", "运行参数尚未保存并通过预检", 503)
        # A physical start may own a supervised pre-start pose check/reset. The
        # reset gate is then enforced again immediately before robot startup.
        # Every other occupancy check remains mandatory at admission.
        self.check_idle(job.config["robot_id"], exclude=job.id,
                        require_reset=not defer_reset_gate)
        if self.busy(job):
            raise TaskError("ROBOT_BUSY", "当前任务仍占用执行名额")
        if not (job.model_service.get("healthy") or
                (not self.plane.enable_actions and job.model_service.get("status") in {"simulated", "ready"})):
            raise TaskError("NOT_READY", "模型部署未就绪", 503)
        if self.plane.enable_actions:
            model = self.plane.deployment.models[str(job.config["model_id"])]
            if self.plane.deployment.running_model_rtc_inference_mode(model) != job.config.get("rtc_inference_mode"):
                raise TaskError("NOT_READY", "模型服务 RTC 模式与已应用配置不同，请重新应用配置", 503)
            if not self.plane._safe_status(self.plane.deployment, model)["healthy"]:
                raise TaskError("NOT_READY", "模型服务健康检查未通过", 503)
        job.task_session.update(accepted=True, source=source,
                                profile_revision=digest(job.config), config_snapshot=copy.deepcopy(job.config))

    # 中文：
    # 将业务请求绑定到已准备模型/冻结 profile，并做单任务准入；不直接产生动作。
    # English:
    # Bind a business request to prepared model/frozen profile and admit one task; generate no actions.
    def submit(self, payload, *, source="operator", callback_url=None, identity="scheduler"):
        p = self.plane
        if not isinstance(payload, dict):
            raise TaskError("INVALID_REQUEST", "请求体必须是对象", 400)
        if set(payload) != {"request_id", "profile"}:
            raise TaskError("INVALID_REQUEST", "只接受 request_id 和 profile", 400)
        import re
        rid, profile = payload["request_id"], payload["profile"]
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", rid):
            raise TaskError("INVALID_REQUEST", "request_id 格式无效", 400)
        if not isinstance(profile, str):
            raise TaskError("INVALID_REQUEST", "profile 格式无效", 400)
        fingerprint = digest(payload)
        with p._lock:
            for job in p._jobs.values():
                b = job.batch
                if b.get("request_id") == rid and b.get("identity", "scheduler") == identity:
                    if b["fingerprint"] != fingerprint:
                        raise TaskError("IDEMPOTENCY_CONFLICT", "request_id 参数冲突")
                    return self.view(job.id), False
            self.check_entry(source)
            # The shared physical-start boundary owns the pre-start pose check
            # and any required supervised reset. Do not mistake a reset gate
            # for another running client; occupancy is still checked here.
            self.check_idle(require_reset=False)
            template = self.ready_template(profile)
            if source == "scheduler" and not template.config.get("presence_gate", {}).get("enabled"):
                raise TaskError("NOT_READY", "自动任务需要启用包裹检测作为完成触发条件", 503)
            physical_start = p.enable_actions and template.config.get("hardware_mode") == "execute"
            self.check_idle(template.config["robot_id"], require_reset=not physical_start)
            job = DeploymentJob(uuid.uuid4().hex, copy.deepcopy(template.config), status="ready_to_start")
            job.model_service = copy.deepcopy(template.model_service)
            job.model_service.update(owned_by_job=False, deployment_id=template.id)
            job.task_session = dict(status="ready", generation=0, remote_started=False)
            job.batch = dict(request_id=rid, profile=profile, fingerprint=fingerprint, identity=identity,
                             source=source, status="starting", start_requested=True,
                             completion_rule="continuous_work_area_empty_then_reset_success",
                             callback_url=callback_url if source == "scheduler" else None, callback=None)
            p._jobs[job.id] = job
            resource = p._model_resources.get(job.config["model_id"])
            if resource is not None:
                resource.setdefault("consumers", set()).add(job.id)
            try:
                p.start_deployment(job.id, p.csrf_token, sorting_authorized=source == "scheduler")
            except Exception as exc:
                if not job.task_session.get("accepted"):
                    # Validation/readiness failed before durable admission.
                    # A rejected request must leave no future executable task.
                    p._jobs.pop(job.id, None)
                    if resource is not None:
                        resource.setdefault("consumers", set()).discard(job.id)
                    raise
                job.status = "recovery_required"
                job.batch.update(status="recovery_required", reason=str(exc))
                # Fail closed: accepted/ambiguous intent remains visible and occupied.
                raise
            return self.view(job.id), True

    def view(self, job_id):
        p = self.plane
        with p._lock:
            job = p._jobs.get(job_id)
            if job is None:
                raise TaskError("NOT_FOUND", "任务不存在", 404)
            b, s = job.batch, job.task_session
            status = b.get("status")
            if status not in FINAL | {"recovery_required"}:
                if job.status == "succeeded" and s.get("completion_evidence"):
                    status = "completed" if s["completion_evidence"]["physical_execution"] else "simulated"
                else:
                    status = {"task_starting": "starting", "succeeded": "cancelled",
                              "model_stopped": "cancelled"}.get(job.status, job.status)
                if s.get("status") == "resetting":
                    status = "resetting"
            allowed, reasons = [], {}
            action_states = {"start": {"ready_to_start", "succeeded"}, "pause": {"running"},
                             "resume": {"paused"},
                             "stop": {"running", "paused", "task_starting", "ready_to_start"}}
            for action, states in action_states.items():
                reason = None
                if job.status not in states or s.get("status") == "resetting":
                    reason = "STATE_CONFLICT"
                if action == "start":
                    try:
                        self.check_entry("operator")
                        physical_start = (p.enable_actions and
                                          job.config.get("hardware_mode") == "execute")
                        self.check_idle(job.config.get("robot_id"), exclude=job.id,
                                        require_reset=not physical_start)
                        if (self.current_deployment_id() !=
                                job.model_service.get("deployment_id", job.id)):
                            raise TaskError("SCHEME_NOT_ACTIVE", "请先将该方案设为当前方案")
                    except TaskError as exc:
                        reason = exc.reason_code
                    # A debug deployment can intentionally start a fresh
                    # generation after its previous manual session ended.
                    # An accepted sorting batch is always single use.
                    if b or (s.get("accepted") and job.status != "succeeded"):
                        reason = "TASK_ALREADY_ACCEPTED"
                if action == "stop" and job.status == "ready_to_start" and not s.get("accepted"):
                    reason = "STATE_CONFLICT"
                if (not self.busy(job) and self.current_deployment_id() !=
                        job.model_service.get("deployment_id", job.id)):
                    reason = "SCHEME_NOT_ACTIVE"
                if action == "resume" and job.presence_gate.get("auto_paused"):
                    reason = "VISION_PAUSED"
                if action == "resume" and p._workstation.get("switching_deployment_id"):
                    reason = "SCHEME_SWITCHING"
                if reason:
                    reasons[action] = reason
                else:
                    allowed.append(action)
            callback = b.get("callback") or {}
            reset_state = self.reset_state(job)
            if status == "completed":
                lifecycle_state = "ended_reset" if reset_state == "ready" else "ended_unreset"
            elif status == "simulated":
                lifecycle_state = "simulated"
            elif status == "cancelled":
                lifecycle_state = ("ended_reset" if job.robot_reset.get("status") == "succeeded"
                                   else "ended_unreset")
            elif status in {"failed", "recovery_required"}:
                lifecycle_state = "abnormal"
            else:
                lifecycle_state = "executing"
            return dict(task_id=job.id, source=b.get("source", s.get("source", "manual_session")),
                        request_id=b.get("request_id"), profile=b.get("profile"),
                        profile_revision=s.get("profile_revision"), status=status,
                        lifecycle_state=lifecycle_state, reset_state=reset_state,
                        deployment_status=job.status, revision=job.revision,
                        allowed_actions=allowed, unavailable_reasons=reasons,
                        completion_rule=b.get("completion_rule") or (
                            "continuous_work_area_empty_then_reset_success"
                            if s.get("completion_evidence") else None),
                        completion_evidence=copy.deepcopy(b.get("completion_evidence") or
                                                          s.get("completion_evidence")),
                        created_at=job.created_at, updated_at=job.updated_at,
                        error=b.get("reason") or job.error,
                        services=dict(model=job.model_service.get("status"),
                                      vision=job.presence_gate.get("transport")),
                        callback=dict(enabled=bool(b.get("callback_url")), event_id=callback.get("event_id"),
                                      status=callback.get("status", "not_pending"),
                                      attempts=callback.get("attempts", 0)))

    def list_tasks(self):
        with self.plane._lock:
            return [self.view(j.id) for j in reversed(list(self.plane._jobs.values()))
                    if j.batch or j.task_session.get("accepted")]

    def stop_scheduler(self, job_id, payload):
        """Use the operator's durable stop command path after scheduler auth."""
        if not isinstance(payload, dict) or set(payload) != {"command_id"}:
            raise TaskError("INVALID_REQUEST", "只接受 command_id", 400)
        cid = payload["command_id"]
        import re
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", cid):
            raise TaskError("INVALID_REQUEST", "command_id 格式无效", 400)
        with self.plane._lock:
            job = self.plane._jobs.get(job_id)
            if job is None or job.batch.get("source") != "scheduler":
                raise TaskError("NOT_FOUND", "调度任务不存在", 404)
            previous = job.task_session.get("commands", {}).get(cid)
            if previous is not None:
                if previous["action"] != "stop":
                    raise TaskError("IDEMPOTENCY_CONFLICT", "command_id 参数冲突")
                return self.view(job_id), False
            view = self.view(job_id)
            if "stop" not in view["allowed_actions"]:
                raise TaskError("STATE_CONFLICT", "当前任务不可终止", 409)
            result = self.command(job_id, {"action": "stop", "command_id": cid,
                                           "expected_revision": job.revision}, self.plane.csrf_token,
                                  scheduler_cancel=True)
            return result, True

    # 中文：
    # 命令 ID 防重复，expected_revision 防旧网页，allowed_actions 限制状态转换。
    # 物理执行不确定时持久化恢复状态，不自动重放同一运动。
    # English:
    # Command IDs prevent duplicates, revisions reject stale UI, and allowed_actions enforce transitions.
    # Persist recovery on uncertain physical outcomes instead of replaying motion.
    def command(self, job_id, payload, token, *, scheduler_cancel=False):
        p = self.plane
        p._require_csrf(token)
        if not isinstance(payload, dict):
            raise TaskError("INVALID_REQUEST", "请求体必须是对象", 400)
        with p._lock:
            view = self.view(job_id)
            job = p._jobs[job_id]
            cid, action = payload.get("command_id"), payload.get("action")
            if not isinstance(cid, str) or not 1 <= len(cid) <= 128:
                raise TaskError("INVALID_REQUEST", "command_id 必填", 400)
            commands = job.task_session.setdefault("commands", {})
            if cid in commands:
                if commands[cid]["action"] != action:
                    raise TaskError("IDEMPOTENCY_CONFLICT", "command_id 参数冲突")
                return self.view(job_id)
            if payload.get("expected_revision") != job.revision:
                raise TaskError("REVISION_CONFLICT", "页面状态已过期，请刷新")
            if action not in view["allowed_actions"]:
                raise TaskError(view["unavailable_reasons"].get(action, "INVALID_ACTION"), "当前状态不允许该操作")
            commands[cid] = dict(action=action, generation=job.task_session.get("generation"), status="accepted")
            self.persist()
            try:
                if action == "stop":
                    p.stop_deployment(job_id, token, manual_completion=not scheduler_cancel)
                else:
                    getattr(p, action + "_deployment")(job_id, token)
                commands[cid]["status"] = "succeeded"
            except TaskError as exc:
                if exc.reason_code in {"SCHEME_SWITCHING", "NOT_ACCEPTING", "ROBOT_BUSY"}:
                    # These are admission refusals before a remote command.
                    # They do not make the previous physical outcome unknown.
                    commands.pop(cid, None)
                    self.persist()
                    raise
                commands[cid]["status"] = "unknown"
                job.status = "recovery_required"
                job.task_session["status"] = "recovery_required"
                self.persist()
                raise
            except Exception:
                commands[cid]["status"] = "unknown"
                job.status = "recovery_required"
                job.task_session["status"] = "recovery_required"
                self.persist()
                raise
            self.persist()
            return self.view(job_id)
