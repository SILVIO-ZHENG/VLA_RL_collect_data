"""Local, safety-first Web control plane for deployment configuration.

The control plane deliberately exposes no arbitrary command or argument API.
By default it validates and simulates a deployment job; site operators must
explicitly opt in to the existing, allow-listed deployment actions.
"""

from __future__ import annotations

import copy
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shlex
import socket
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from nexus_deploy.deployment import (
    POLICY_PROTOCOL_NAME,
    DeployError,
    Deployment,
)
from nexus_deploy.presence_gate import (
    GateAction,
    PresenceTaskGate,
    PresenceTaskGateConfig,
    TaskState,
)
from nexus_deploy.safety_profiles import AlignmentSafetyProfileRepository
from nexus_deploy.control.tasks import TaskService
from nexus_deploy.control.tasks import digest as config_digest
from nexus_deploy.control.tablet_start import TabletStartBridge, TabletStartTarget
from nexus_deploy.control.errors import ValidationError, TaskError
from nexus_deploy.control.defaults import TASK_DEFAULTS
from nexus_deploy.rtc_modes import rtc_inference_mode_for_args, validate_rtc_inference_mode
from nexus_deploy.control.models import (
    DeploymentJob,
    PresenceRuntimeSettings,
    PresenceSidecar,
)

TERMINAL_JOB_STATES = {"succeeded", "failed", "rejected", "cancelled", "model_stopped"}
CONTROLLABLE_JOB_STATES = {"running", "paused"}
STARTABLE_JOB_STATES = {"ready_to_start", "succeeded"}
STOPPABLE_JOB_STATES = CONTROLLABLE_JOB_STATES | {"ready_to_start", "task_starting"}
logger = logging.getLogger(__name__)
RESETTABLE_JOB_STATES = {
    "ready_to_start",
    "succeeded",
    "cancelled",
    "failed",
    "rejected",
    "model_stopped",
    "recovery_required",
}
PROBE_TTL_S = 30.0
G2_VISUAL_JOINT_NAMES = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx31_gripper_l_inner_joint1",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx71_gripper_r_inner_joint1",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
)
SIMULATOR_BRIDGE_HZ = 8.0
PRESENCE_GATE_STATES = {"PACKAGE_PRESENT", "NO_PACKAGE", "UNKNOWN"}
PRESENCE_GATE_ACTIONS = {"none", "pause", "resume", "reset"}
PRESENCE_GATE_MAX_WINDOW_STATES = 4
PRESENCE_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
PRESENCE_CAMERA_CONFIRM_FRAMES = 3
PRESENCE_GATE_MAX_REASON_LENGTH = 240


def canonical_effector_profile(value: Any) -> str:
    """Normalize the one historical profile id accepted by persisted jobs."""

    profile = str(value)
    return "dahuan" if profile == "dh" else profile


class ControlPlane:
    """Read deployment inventory, validate forms, and supervise local jobs."""

    DEFAULTS: dict[str, Any] = TASK_DEFAULTS

    def __init__(
        self,
        config_path: Path | str,
        *,
        enable_actions: bool = False,
        enable_execute: bool = False,
        enable_reset: bool = False,
        robot_telemetry_url: str | None = None,
        simulator_control_url: str | None = None,
        simulator_viewer_port: int = 8767,
        alignment_url: str | None = None,
        alignment_surfaces_directory: Path | str | None = None,
        presence_backend: str | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.deployment = Deployment(self.config_path)
        self.DEFAULTS = copy.deepcopy(type(self).DEFAULTS)
        site_defaults = self.deployment.raw.get("task_defaults", {})
        if set(site_defaults) - set(self.DEFAULTS):
            raise ValueError("Unknown task_defaults field")
        for key, value in site_defaults.items():
            if key == "safety":
                if not isinstance(value, Mapping) or set(value) - set(self.DEFAULTS["safety"]):
                    raise ValueError("Unknown task_defaults.safety field")
                self.DEFAULTS["safety"].update(value)
            elif key == "presence_gate":
                raise ValueError("Gate timings are service-owned and cannot be changed by task_defaults")
            else:
                self.DEFAULTS[key] = value
        self._presence_runtime = self._load_presence_runtime_settings()
        self.enable_actions = bool(enable_actions)
        if enable_execute and not self.enable_actions:
            raise ValueError("enable_execute requires enable_actions")
        self.enable_execute = bool(enable_execute)
        if enable_reset and not self.enable_actions:
            raise ValueError("enable_reset requires enable_actions")
        self.enable_reset = bool(enable_reset)
        self.robot_telemetry_url = (
            robot_telemetry_url.rstrip("/") if robot_telemetry_url else None
        )
        self.simulator_control_url = (
            simulator_control_url.rstrip("/") if simulator_control_url else None
        )
        self.simulator_viewer_port = int(simulator_viewer_port)
        self.alignment_url = str(alignment_url or "").strip() or None
        self.alignment_surfaces_directory = (
            Path(alignment_surfaces_directory).expanduser().resolve()
            if alignment_surfaces_directory is not None
            else (self.config_path.parent / "runtime" / "alignment-surfaces").resolve()
        )
        self.safety_profile_repository = AlignmentSafetyProfileRepository(
            self.alignment_surfaces_directory.parent / "safety-profiles"
        )
        self.deployment.alignment_surfaces_root = self.alignment_surfaces_directory
        self.deployment.generated_safety_profiles_root = (
            self.safety_profile_repository.root
        )
        self._simulator_view_initialized = False
        self.csrf_token = secrets.token_urlsafe(32)
        self._jobs: dict[str, DeploymentJob] = {}
        self._lock = threading.RLock()
        self._simulator_bridge_lock = threading.Lock()
        self._simulator_bridge_event = threading.Event()
        self._simulator_pending_pose: tuple[str, int, list[float]] | None = None
        self._simulator_bridge_thread: threading.Thread | None = None
        self._reserved_models: dict[str, str] = {}
        self._reserved_robots: dict[str, str] = {}
        self._reserved_gpus: dict[str, str] = {}
        self._model_resources: dict[str, dict[str, Any]] = {}
        self._probe_leases: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._camera_cache: dict[tuple[str, str], tuple[str, bytes, str]] = {}
        self._telemetry_tokens: dict[str, str] = {}
        self._presence_tokens: dict[str, str] = {}
        self._presence_task_gates: dict[str, PresenceTaskGate] = {}
        self._presence_sidecars: dict[str, PresenceSidecar] = {}
        self._cleanup_challenges: dict[str, dict[str, Any]] = {}
        self._state_path = self.deployment.runtime_dir / "control-plane-state.json"
        self._workstation = dict(mode="debug", accepting=True, reason="", persistence_error=False,
                                 active_deployment_id=None, switching_deployment_id=None, switch_error="",
                                 reset_gate=dict(required=False, status="ready", reason="",
                                                 source_job_id=None, updated_at=None))
        self.tasks = TaskService(self)
        from nexus_deploy.control.vision import SidecarPresenceProvider, MockPresenceProvider, ReplayPresenceProvider
        backend = presence_backend or ("sidecar" if self.enable_actions else "mock")
        if backend not in {"sidecar", "mock", "replay"}:
            raise ValueError("presence_backend must be sidecar, mock or replay")
        if self.enable_actions and backend != "sidecar":
            raise ValueError("offline presence backends require hardware actions disabled")
        self.presence_provider = {
            "sidecar": SidecarPresenceProvider,
            "mock": MockPresenceProvider,
            "replay": ReplayPresenceProvider,
        }[backend](self)
        from nexus_deploy.control.robot import RobotPort
        self.robot_port = RobotPort(self)
        recovered_running = self._load_state()
        self._health_stop = threading.Event()
        self._health_thread = threading.Thread(target=self._health_loop, name="PresenceHealth", daemon=True)
        self._health_thread.start()
        if self.simulator_control_url is not None:
            self._simulator_bridge_thread = threading.Thread(
                target=self._simulator_bridge_loop,
                name="LatestSimulatorBridge",
                daemon=True,
            )
            self._simulator_bridge_thread.start()
        if self.enable_actions:
            for job_id in recovered_running:
                threading.Thread(target=self._telemetry_loop, args=(job_id,), daemon=True).start()
        self._tablet_start_bridge: TabletStartBridge | None = None
        if self.enable_execute:
            self._tablet_start_bridge = TabletStartBridge(
                select_target=self._tablet_start_candidate,
                report=self._tablet_start_report,
                confirm=self._tablet_start_confirm,
            )
            self._tablet_start_bridge.start()

    def _tablet_start_candidate(self) -> TabletStartTarget | None:
        """Expose only a saved, operator-startable HITL deployment to the G2 recorder watcher."""

        with self._lock:
            if not self.enable_execute:
                return None
            job_id = self.tasks.current_deployment_id()
            job = self._jobs.get(job_id) if job_id is not None else None
            if (job is None or job.batch or job.status not in STARTABLE_JOB_STATES
                    or job.config.get("human_in_loop") is not True
                    or job.config.get("hardware_mode") != "execute"
                    or job.task_session.get("runtime_configured") is not True
                    or job.model_service.get("candidate_only") is True
                    or "start" not in self.tasks.view(job.id)["allowed_actions"]):
                return None
            try:
                robot = self._robot_for_config(job.config)
            except (DeployError, KeyError, ValueError):
                return None
            remote = shlex.quote(robot.remote_directory)
            command = (
                f"cd {remote} && exec ./robot_client/debug_python.sh "
                "-m robot_client.hitl.g2_runtime.tablet_start"
            )
            return TabletStartTarget(
                job.id,
                config_digest(job.config),
                tuple([*self.deployment.ssh_base(robot), command]),
            )

    def _tablet_start_report(self, target: TabletStartTarget, status: str) -> None:
        with self._lock:
            job = self._jobs.get(target.job_id)
            if job is None or config_digest(job.config) != target.config_revision:
                return
            old = job.task_session.get("tablet_start")
            if isinstance(old, dict) and old.get("status") == status:
                return
            job.task_session["tablet_start"] = {"status": status, "updated_at": time.time()}
            job.updated_at = time.time()
            job.revision += 1
            try:
                self._persist_state()
            except OSError:
                logger.exception("Could not persist G2 tablet start status")

    def _tablet_start_confirm(self, target: TabletStartTarget, record_name: str) -> None:
        if self._tablet_start_bridge is not None and self._tablet_start_bridge.stop_event.is_set():
            return
        if self._tablet_start_candidate() != target:
            return
        try:
            self.start_deployment(
                target.job_id, self.csrf_token,
                start_source="tablet", tablet_record_name=record_name,
            )
        except (TaskError, ValidationError, DeployError) as error:
            self._tablet_start_report(target, "rejected")
            logger.warning("G2 tablet start admission rejected: %s", error)

    def _load_presence_runtime_settings(self) -> PresenceRuntimeSettings | None:
        """Read immutable sidecar settings from ``[presence_runtime]``.

        A missing table simply leaves the feature unavailable.  A malformed
        table fails control-service startup rather than silently redirecting
        JPEGs or a privileged per-job token to an unexpected host.
        """

        raw = self.deployment.raw.get("presence_runtime")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("[presence_runtime] must be a TOML table")

        def configured_path(name: str) -> Path:
            value = raw.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"presence_runtime.{name} must be a non-empty path")
            path = Path(value).expanduser()
            if name == "python_executable":
                candidate = path if path.is_absolute() else self.deployment.project_root / path
                return Path(os.path.normpath(candidate.absolute()))
            return path.resolve() if path.is_absolute() else (self.deployment.project_root / path).resolve()

        project_directory = configured_path("project_directory")
        python_executable = configured_path("python_executable")
        weights = configured_path("weights")
        rules = configured_path("rules")
        bind_host = str(raw.get("bind_host", "")).strip()
        try:
            address = ipaddress.ip_address(bind_host)
        except ValueError as error:
            raise ValueError("presence_runtime.bind_host must be a literal IP address") from error
        if not (address.is_private or address.is_loopback):
            raise ValueError("presence_runtime.bind_host must be a private-LAN or loopback IP")
        ingress_port_raw = raw.get("ingress_port")
        if isinstance(ingress_port_raw, bool):
            raise ValueError("presence_runtime.ingress_port must be an integer")
        try:
            ingress_port = int(ingress_port_raw)
        except (TypeError, ValueError) as error:
            raise ValueError("presence_runtime.ingress_port must be an integer") from error
        if not 1 <= ingress_port <= 65535:
            raise ValueError("presence_runtime.ingress_port must be in 1..65535")
        control_api_url = str(raw.get("control_api_url", "")).strip().rstrip("/")
        parsed_control = urlparse(control_api_url)
        if (
            parsed_control.scheme not in {"http", "https"}
            or parsed_control.hostname not in {bind_host, "192.168.110.14"}
            or parsed_control.port is None
            or parsed_control.path not in {"", "/"}
            or parsed_control.username
            or parsed_control.password
            or parsed_control.query
            or parsed_control.fragment
        ):
            raise ValueError(
                "presence_runtime.control_api_url must be the local private-LAN HTTP(S) control URL"
            )
        device = str(raw.get("device", "0")).strip()
        if not device or len(device) > 64 or any(character.isspace() for character in device):
            raise ValueError("presence_runtime.device is invalid")
        try:
            detect_hz = float(raw.get("detect_hz", 10.0))
            startup_timeout_s = float(raw.get("startup_timeout_s", 60.0))
        except (TypeError, ValueError) as error:
            raise ValueError("presence_runtime timing values must be numeric") from error
        if not math.isfinite(detect_hz) or not 0 < detect_hz <= 15:
            raise ValueError("presence_runtime.detect_hz must be in (0, 15]")
        if not math.isfinite(startup_timeout_s) or not 5 <= startup_timeout_s <= 180:
            raise ValueError("presence_runtime.startup_timeout_s must be in [5, 180]")
        return PresenceRuntimeSettings(
            project_directory=project_directory,
            python_executable=python_executable,
            weights=weights,
            rules=rules,
            device=device,
            bind_host=bind_host,
            ingress_port=ingress_port,
            control_api_url=control_api_url,
            detect_hz=detect_hz,
            startup_timeout_s=startup_timeout_s,
        )

    def _load_state(self) -> list[str]:
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        self._workstation.update(value.get("workstation", {}))
        self._workstation.update(accepting=False, reason="控制服务已重启，请核实后恢复接单")
        if self._workstation.get("switching_deployment_id"):
            self._workstation.update(switching_deployment_id=None,
                                     switch_error="控制服务在应用配置期间重启；请核实模型状态")
        raw_tokens = value.get("telemetry_tokens", {}) if isinstance(value, Mapping) else {}
        if isinstance(raw_tokens, Mapping):
            self._telemetry_tokens.update(
                {
                    str(job_id): str(token)
                    for job_id, token in raw_tokens.items()
                    if isinstance(job_id, str)
                    and isinstance(token, str)
                    and len(token) >= 32
                }
            )
        recovered_running: list[str] = []
        reconciled = False
        live_model_pids: dict[str, int | None] = {}
        for raw_job in value.get("jobs", ()) if isinstance(value, Mapping) else ():
            try:
                job = DeploymentJob.from_dict(raw_job)
            except (KeyError, TypeError, ValueError):
                continue
            profile_name = canonical_effector_profile(
                job.config.get("effector_profile", "")
            )
            job.config["effector_profile"] = profile_name
            if job.model_service:
                job.model_service["effector_profile"] = profile_name
            model = self.deployment.models.get(str(job.config.get("model_id", "")))
            if model is not None and profile_name in model.effector_profiles:
                version_id = str(job.config.get("model_version", "")).strip()
                # Missing registration must not relabel a historical checkpoint.
                if not version_id:
                    version_id = model.variant(profile_name).default_version
                job.config["model_version"] = version_id
                job.model_service["model_version"] = version_id
                # Prior saved jobs had a version-owned server route only.
                if not job.config.get("rtc_inference_mode"):
                    if version_id in model.variant(profile_name).versions:
                        version = model.version(profile_name, version_id)
                        job.config["rtc_inference_mode"] = rtc_inference_mode_for_args(
                            (*version.base_args, *version.real_args)
                        ) or "guided"
                    elif job.model_service.get("rtc_inference_mode") in {"trained", "guided"}:
                        job.config["rtc_inference_mode"] = job.model_service["rtc_inference_mode"]
                if job.config.get("rtc_inference_mode"):
                    job.model_service.setdefault("rtc_inference_mode", job.config["rtc_inference_mode"])
            # Timings are service-owned. Update only the idle current template;
            # historical task snapshots and completion evidence remain intact.
            if (job.id == self._workstation.get("active_deployment_id")
                    and not job.batch and not job.task_session.get("accepted")
                    and job.status in {"ready_to_start", "succeeded"}):
                current_gate = job.config.get("presence_gate", {})
                if isinstance(current_gate, Mapping):
                    updated_gate = copy.deepcopy(self.DEFAULTS["presence_gate"])
                    updated_gate["enabled"] = current_gate.get("enabled") is True
                    if current_gate != updated_gate:
                        job.config["presence_gate"] = updated_gate
                        job.revision += 1
                        job.updated_at = time.time()
                        reconciled = True
            if job.status in {"queued", "validating", "preflight", "deploying", "task_starting"}:
                job.status = "recovery_required"
                job.message = "控制服务重启中断了过渡步骤；请检查目标后结束此作业并重试"
                job.error = "控制服务在非稳定状态重启"
                job.task_session["status"] = "recovery_required"
            elif job.status in {"running", "paused", "pausing", "stopping", "cancelling"} or job.robot_reset.get("status") in {"preparing", "running", "unknown"}:
                job.status = "recovery_required"
                job.task_session["status"] = "recovery_required"
                if job.batch:
                    job.batch.update(status="recovery_required", reason="控制服务重启，必须人工核实")
            if job.status in {"running", "paused"} and self._presence_enabled(job):
                # Per-job ingress/control tokens and the pure gate live only
                # in memory.  A control-plane restart must never silently
                # recreate them or let a historical sidecar resume automatic
                # control of an already-running robot task.
                presence_status = self._ensure_presence_gate_status(job)
                presence_status.update(
                    {
                        "mode": "automatic",
                        "transport": "recovery_required",
                        "task_state": "recovery_required",
                        "sequence": None,
                        "received_at": None,
                        "age_s": None,
                        "state": "UNKNOWN",
                        "window_states": [],
                        "source_fresh": False,
                        "no_package_elapsed_s": None,
                        "auto_paused": False,
                        "awaiting_reset_ready": False,
                        "pending_action": None,
                        "proposed_action": "none",
                        "action_status": "disabled_after_restart",
                        "reason": "control-plane restart cleared the in-memory package gate session",
                        "message": "控制服务重启后已安全停用本任务的自动包裹门控；请结束当前任务后重新创建任务。",
                        "last_error": "控制服务重启；不会自动暂停、复位或恢复机器人。",
                    }
                )
            self._jobs[job.id] = job
            model_id = str(job.config.get("model_id", ""))
            if job.model_service.get("owned_by_job"):
                if model_id not in live_model_pids:
                    model = self.deployment.models.get(model_id)
                    try:
                        live_model_pids[model_id] = (
                            self.deployment.running_pid(model) if model is not None else None
                        )
                    except (DeployError, OSError):
                        live_model_pids[model_id] = None
                live_pid = live_model_pids[model_id]
                recorded_pid = job.model_service.get("pgid")
                if live_pid is None:
                    job.model_service.update(
                        {
                            "status": "stopped",
                            "healthy": False,
                            "owned_by_job": False,
                            "pgid": None,
                            "recent_log": "恢复检查发现原模型进程已不存在",
                        }
                    )
                    if job.status not in TERMINAL_JOB_STATES:
                        job.status = "failed"
                        job.message = "原模型服务已退出；历史占用已自动释放，请新建部署"
                        job.error = "恢复检查发现作业拥有的模型进程已不存在"
                        job.task_session["status"] = "failed"
                    job.updated_at = time.time()
                    job.events.append(
                        {
                            "time": job.updated_at,
                            "status": job.status,
                            "message": "控制服务恢复时已释放失效的模型占用",
                            "progress": job.progress,
                        }
                    )
                    reconciled = True
                    continue
                try:
                    pid_mismatch = recorded_pid is not None and int(recorded_pid) != live_pid
                except (TypeError, ValueError):
                    pid_mismatch = True
                if pid_mismatch:
                    job.model_service.update(
                        {
                            "owned_by_job": False,
                            "pgid": None,
                            "recent_log": "当前模型进程并非由该历史作业拥有；已转为复用",
                        }
                    )
                    job.updated_at = time.time()
                    reconciled = True
                    if job.status in {"running", "paused"}:
                        self._reserved_robots[str(job.config.get("robot_id", ""))] = job.id
                    continue
                resource = self._model_resources.setdefault(
                    model_id,
                    {"owner_job_id": job.id, "pgid": job.model_service.get("pgid"), "consumers": set()},
                )
                if job.status not in TERMINAL_JOB_STATES:
                    resource["consumers"].add(job.id)
                if job.model_service.get("status") != "stopped":
                    self._reserved_models[model_id] = job.id
                selected_gpu = str(job.model_service.get("gpu", ""))
                if selected_gpu and selected_gpu != "auto" and job.model_service.get("status") != "stopped":
                    self._reserved_gpus[selected_gpu] = job.id
            if job.status in {"running", "paused"}:
                self._reserved_robots[str(job.config.get("robot_id", ""))] = job.id
        # Older state files only tracked reset on individual tasks. Migrate
        # conservatively: an executed task or failed reset without success
        # must not become admissible merely because the control service restarts.
        if "reset_gate" not in value.get("workstation", {}):
            pending = [job for job in self._jobs.values()
                       if ((job.config.get("hardware_mode") == "execute" and
                            job.task_session.get("accepted") and
                            job.robot_reset.get("status") != "succeeded") or
                           job.robot_reset.get("status") in {"failed", "unknown", "preparing", "running"})]
            if pending:
                latest = max(pending, key=lambda job: job.updated_at)
                self._mark_robot_reset_required(latest, "failed" if latest.robot_reset.get("status") == "failed"
                                                else "pending", "历史真机操作未记录成功复位")
            reconciled = True
        # A later verified reset may have resolved an old task even when its
        # deployment was already failed or ready again. Do not resurrect that
        # historical recovery as a live task after a service restart.
        if not self._workstation["reset_gate"]["required"]:
            for job in self._jobs.values():
                reset_by = self._jobs.get(job.task_session.get("reset_resolved_by"))
                if (job.batch.get("status") == "recovery_required"
                        and not job.task_session.get("remote_started")
                        and reset_by is not None
                        and reset_by.robot_reset.get("status") == "succeeded"):
                    job.batch.update(status="failed", reset_required=False,
                                     reason="历史任务已由后续机器人复位核实；任务未完成")
                    reconciled = True
        if reconciled:
            self._persist_state()
        return recovered_running

    def _persist_state(self) -> None:
        try:
            self._write_state()
        except OSError:
            self.tasks.cancel_reset_admission("PERSISTENCE_FAILED")
            self._workstation.update(accepting=False, persistence_error=True,
                                     reason="状态持久化失败，需要人工核实")
            raise

    def _write_state(self) -> None:
        self.deployment.runtime_dir.mkdir(parents=True, exist_ok=True)
        # Telemetry tokens must survive a control-plane restart while a robot
        # session is still publishing. They stay in this private server-side
        # state file and are never included in DeploymentJob.to_dict() or an
        # HTTP response.
        payload = {
            "version": 3,
            "workstation": dict(self._workstation),
            "jobs": [job.to_dict() for job in self._jobs.values()],
            "telemetry_tokens": dict(self._telemetry_tokens),
        }
        temporary = self._state_path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.chmod(0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(self._state_path)
        directory_fd = os.open(str(self._state_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _model_modes(model_id: str, raw: Mapping[str, Any]) -> tuple[str, ...]:
        configured = tuple(str(item) for item in raw.get("supported_modes", ()))
        if configured:
            return configured
        lowered = model_id.lower()
        if "pi05" in lowered or "pi0" in lowered:
            return ("sync", "async", "rtc")
        if "lingbot" in lowered:
            return ("sync", "async")
        return ("sync", "async")

    def _safe_status(self, deployment: Deployment, model: Any) -> dict[str, Any]:
        try:
            pid = deployment.running_pid(model)
            health = deployment.health(model, timeout_s=0.15) if pid is not None else None
        except (DeployError, OSError):
            pid = None
            health = None
        status = {"running": pid is not None, "healthy": health is not None, "pid": pid}
        self.tasks.observe_model_health(getattr(model, "name", None), status)
        return status

    def _generated_safety_profiles(self, robot: Any) -> list[Path]:
        template = robot.source_client / "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json"
        if not template.is_file():
            return []
        try:
            hosts = {
                str(item["host"])
                for item in self.deployment.robot_network_endpoints(robot.name)
            }
            return self.safety_profile_repository.sync(
                measurements_root=self.alignment_surfaces_directory,
                robot_id=robot.name,
                template_path=template,
                expected_robot_aid=robot.expected_aid,
                allowed_robot_hosts=hosts,
            )
        except (OSError, TypeError, ValueError):
            return []

    def _safety_profiles(self, robot: Any) -> list[dict[str, Any]]:
        source = robot.source_client
        try:
            manifest_entries = {
                line.strip()
                for line in robot.manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except OSError:
            manifest_entries = set()
        generated_candidates = self._generated_safety_profiles(robot)
        candidates = [
            *generated_candidates,
            source / "SAFETY_CONFIG_TEMPLATE.json",
            source / "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json",
        ]
        if source.is_dir():
            candidates.extend(sorted(source.glob("*safety*.json")))
        profiles: list[dict[str, Any]] = []
        seen: set[Path] = set()
        generated_resolved = {item.resolve() for item in generated_candidates}
        for path in candidates:
            try:
                resolved = path.resolve()
                if resolved in seen or not resolved.is_file():
                    continue
                seen.add(resolved)
                value = json.loads(resolved.read_text(encoding="utf-8"))
                calibrated = value.get("calibrated") is True
                read_only_ready = calibrated or value.get("read_only_ready") is True
                execute_approved = calibrated and value.get("execute_approved", True) is True
                generated = resolved in generated_resolved
                generated_deployable = False
                if generated:
                    try:
                        self.deployment.workspace_safety_profile(robot, resolved.name)
                    except DeployError:
                        generated_deployable = False
                    else:
                        generated_deployable = True
                surface_calibration = value.get("surface_calibration", {})
                surfaces = value.get("surfaces", [])
                surface = surfaces[0] if isinstance(surfaces, list) and surfaces else {}
                observed_at = (
                    surface_calibration.get("observed_at")
                    if isinstance(surface_calibration, Mapping)
                    else None
                )
                age_s = (
                    max(0.0, time.time() - float(observed_at))
                    if isinstance(observed_at, (int, float))
                    else None
                )
                profiles.append(
                    {
                        "id": resolved.name,
                        "label": str(
                            value.get("display_name")
                            or resolved.stem.replace("_", " ")
                        ),
                        "calibrated": calibrated,
                        "read_only_ready": read_only_ready,
                        "execute_approved": execute_approved,
                        "deployable": generated_deployable or resolved.name in manifest_entries,
                        "generated": generated,
                        "height_m": (
                            surface.get("height") if isinstance(surface, Mapping) else None
                        ),
                        "measurement_id": (
                            surface_calibration.get("measurement_id")
                            if isinstance(surface_calibration, Mapping)
                            else None
                        ),
                        "observed_at": observed_at,
                        "age_s": round(age_s, 1) if age_s is not None else None,
                        "fresh": age_s is not None and age_s <= 300.0,
                        "version": value.get("version"),
                        "inter_arm_collision": (
                            {
                                "enabled": True,
                                "geometry": value["inter_arm_collision"].get("geometry"),
                                "hard_margin_m": value["inter_arm_collision"].get("hard_margin"),
                                "warning_margin_m": value["inter_arm_collision"].get("warning_margin"),
                                "validation_status": value["inter_arm_collision"].get(
                                    "validation_status", "unvalidated"
                                ),
                            }
                            if isinstance(value.get("inter_arm_collision"), Mapping)
                            and value["inter_arm_collision"].get("enabled") is True
                            else {"enabled": False}
                        ),
                    }
                )
            except (OSError, ValueError, TypeError):
                continue
        return profiles

    def gpu_inventory(self) -> list[dict[str, Any]]:
        """Return a live, redacted NVIDIA GPU snapshot for scheduling and UI."""

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        processes: dict[str, list[dict[str, Any]]] = {}
        try:
            process_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in process_result.stdout.splitlines():
                columns = [item.strip() for item in line.split(",", 3)]
                if len(columns) != 4:
                    continue
                gpu_uuid, pid, process_name, used_memory = columns
                processes.setdefault(gpu_uuid, []).append(
                    {
                        "pid": int(pid),
                        "name": Path(process_name).name,
                        "memory_used_mb": int(used_memory),
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            processes = {}

        with self._lock:
            reservations = dict(self._reserved_gpus)
        inventory: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            columns = [item.strip() for item in line.split(",", 7)]
            if len(columns) != 8:
                continue
            index, gpu_uuid, name, total, used, free, utilization, temperature = columns
            try:
                process_list = processes.get(gpu_uuid, [])
                inventory.append(
                    {
                        "id": index,
                        "uuid": gpu_uuid,
                        "name": name,
                        "memory_total_mb": int(total),
                        "memory_used_mb": int(used),
                        "memory_free_mb": int(free),
                        "utilization_percent": int(utilization),
                        "temperature_c": int(temperature),
                        "process_count": len(process_list),
                        "processes": process_list,
                        "reserved_by_job": reservations.get(index),
                    }
                )
            except ValueError:
                continue
        return inventory

    def _gpu_recommendation(
        self,
        model_id: str,
        inventory: list[dict[str, Any]],
        *,
        job_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Recommend an eligible physical GPU without reserving it.

        A completely idle GPU always wins. When every eligible card already
        has a lightweight compute process, prefer the card with the most free
        memory and lowest utilization instead of incorrectly reporting that no
        GPU exists. The real launch repeats this decision while holding the
        reservation lock.
        """

        if model_id not in self.deployment.models:
            return None
        minimum_free = int(
            self.deployment.raw["models"][model_id].get("gpu_min_free_memory_mb", 0)
        )
        with self._lock:
            reservations = dict(self._reserved_gpus)
        eligible = [
            gpu
            for gpu in inventory
            if int(gpu["memory_free_mb"]) >= minimum_free
            and reservations.get(str(gpu["id"])) in {None, job_id}
        ]
        if not eligible:
            return None
        selected = min(
            eligible,
            key=lambda item: (
                int(item["process_count"]) > 0,
                -int(item["memory_free_mb"]),
                int(item["utilization_percent"]),
                int(item["id"]),
            ),
        )
        completely_idle = int(selected["process_count"]) == 0
        reason = (
            f"无计算进程，空闲 {selected['memory_free_mb']} MiB"
            if completely_idle
            else (
                "当前没有完全空闲的可用卡；"
                f"该卡空闲 {selected['memory_free_mb']} MiB、"
                f"利用率 {selected['utilization_percent']}%"
            )
        )
        return {
            "gpu_id": str(selected["id"]),
            "model_id": model_id,
            "minimum_free_memory_mb": minimum_free,
            "completely_idle": completely_idle,
            "reason": reason,
        }

    def gpu_status(self) -> dict[str, Any]:
        """Return inventory plus current model-specific recommendations."""

        inventory = self.gpu_inventory()
        recommendations = {
            model_id: recommendation
            for model_id in self.deployment.models
            if (
                recommendation := self._gpu_recommendation(model_id, inventory)
            )
            is not None
        }
        return {"gpus": inventory, "recommendations": recommendations}

    def _reserve_gpu(self, job: DeploymentJob) -> dict[str, Any]:
        model_id = str(job.config["model_id"])
        raw_model = self.deployment.raw["models"][model_id]
        minimum_free = int(raw_model.get("gpu_min_free_memory_mb", 0))
        requested = str(job.config.get("gpu_id", "auto"))
        inventory = self.gpu_inventory()
        if not inventory:
            fallback = str(self.deployment.models[model_id].cuda_visible_devices)
            if requested != "auto" and requested != fallback:
                raise DeployError("GPU 状态不可用，无法安全使用手动指定的显卡")
            selected: dict[str, Any] = {
                "id": fallback,
                "name": "配置回退 GPU",
                "memory_free_mb": None,
                "memory_total_mb": None,
                "memory_used_mb": None,
                "utilization_percent": None,
                "process_count": None,
                "processes": [],
            }
        else:
            # Selection and reservation are one critical section. Two browser
            # tabs starting different models cannot both choose the same card.
            with self._lock:
                candidates = []
                for gpu in inventory:
                    owner = self._reserved_gpus.get(str(gpu["id"]))
                    if owner is not None and owner != job.id:
                        continue
                    if int(gpu["memory_free_mb"]) < minimum_free:
                        continue
                    candidates.append(gpu)
                if requested == "auto":
                    if not candidates:
                        free_text = ", ".join(
                            f"GPU {item['id']}={item['memory_free_mb']} MiB"
                            for item in inventory
                        )
                        raise DeployError(
                            f"没有满足模型最低空闲显存 {minimum_free} MiB 的 GPU；{free_text}"
                        )
                    selected = min(
                        candidates,
                        key=lambda item: (
                            int(item["process_count"]) > 0,
                            -int(item["memory_free_mb"]),
                            int(item["utilization_percent"]),
                            int(item["id"]),
                        ),
                    )
                else:
                    selected = next(
                        (item for item in candidates if str(item["id"]) == requested), None
                    )
                    if selected is None:
                        raise DeployError(
                            f"GPU {requested} 不可用、已被平台预留，"
                            f"或空闲显存低于 {minimum_free} MiB"
                        )
                selected_id = str(selected["id"])
                self._reserved_gpus[selected_id] = job.id
            return copy.deepcopy(selected)
        selected_id = str(selected["id"])
        with self._lock:
            owner = self._reserved_gpus.get(selected_id)
            if owner is not None and owner != job.id:
                raise DeployError(f"GPU {selected_id} 已被部署任务 {owner} 占用")
            self._reserved_gpus[selected_id] = job.id
        return copy.deepcopy(selected)

    def active_alignment_surface(self, profile_id: str = "standard_table") -> dict[str, Any]:
        """Read the latest accepted alignment result without mutating it."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", profile_id):
            return {"available": False, "reason": "invalid_profile"}
        path = self.alignment_surfaces_directory / profile_id / "active.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"available": False, "reason": "not_calibrated", "profile_id": profile_id}
        except (OSError, ValueError, TypeError):
            return {"available": False, "reason": "invalid_calibration", "profile_id": profile_id}
        surface = value.get("surface") if isinstance(value, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != 1
            or value.get("profile_id") != profile_id
            or value.get("base_frame") != "base_link"
            or not isinstance(surface, Mapping)
            or surface.get("eligible") is not True
        ):
            return {"available": False, "reason": "ineligible_calibration", "profile_id": profile_id}
        try:
            height = float(surface["height_m"])
            created_at = float(value["created_at"])
        except (KeyError, TypeError, ValueError):
            return {"available": False, "reason": "invalid_calibration", "profile_id": profile_id}
        if not math.isfinite(height) or not math.isfinite(created_at):
            return {"available": False, "reason": "invalid_calibration", "profile_id": profile_id}
        age_s = max(0.0, time.time() - created_at)
        return {
            "available": True,
            "fresh": age_s <= 300.0,
            "age_s": round(age_s, 1),
            "measurement_id": value.get("measurement_id"),
            "profile_id": profile_id,
            "height_m": height,
            "tilt_deg": surface.get("tilt_deg"),
            "spread": copy.deepcopy(surface.get("spread", {})),
            "robot": copy.deepcopy(value.get("robot", {})),
            "robot_model": copy.deepcopy(value.get("robot_model", {})),
            "camera_calibration_version": value.get("camera_calibration_version"),
            "created_at": created_at,
        }

    def bootstrap(self) -> dict[str, Any]:
        gpu_status = self.gpu_status()
        gpus = gpu_status["gpus"]
        models = []
        for model_id, model in self.deployment.models.items():
            raw = self.deployment.raw["models"][model_id]
            modes = list(self._model_modes(model_id, raw))
            resource = self._model_resources.get(model_id)
            owner_job = self._jobs.get(str(resource.get("owner_job_id"))) if resource else None
            active_gpu = (
                str(owner_job.model_service.get("gpu"))
                if owner_job is not None and owner_job.model_service.get("gpu") is not None
                else model.cuda_visible_devices
            )
            model_status = self._safe_status(self.deployment, model)
            running_effector_profile = (
                self.deployment.running_model_effector_profile(model)
                if model_status.get("running")
                else None
            )
            running_model_version = (
                self.deployment.running_model_version(model)
                if model_status.get("running")
                else None
            )
            models.append(
                {
                    "id": model_id,
                    "label": str(raw.get("display_name", model_id)),
                    "description": str(raw.get("description", "已配置的 VLA 推理服务")),
                    "port": model.port,
                    "gpu": active_gpu,
                    "configured_gpu": model.cuda_visible_devices,
                    "gpu_min_free_memory_mb": int(raw.get("gpu_min_free_memory_mb", 0)),
                    "startup_timeout_s": model.startup_timeout_s,
                    "status": model_status,
                    "supported_modes": modes,
                    "inference_modes": modes,
                    "rtc_execution_horizon": raw.get("rtc_execution_horizon"),
                    "rtc_execute_enabled": raw.get("rtc_execute_enabled", True) is True,
                    "tags": [str(item) for item in raw.get("tags", ())],
                    "effector_profile": model.effector_profile,
                    "effector_profile_label": self.deployment.effector_profiles[
                        model.effector_profile
                    ].display_name,
                    "effector_profiles": list(model.effector_profiles),
                    "running_effector_profile": running_effector_profile,
                    "running_model_version": running_model_version,
                    "running_rtc_inference_mode": (
                        self.deployment.running_model_rtc_inference_mode(model)
                        if self.enable_actions else None
                    ),
                    "effector_variants": {
                        profile_name: {
                            "profile": profile_name,
                            "label": self.deployment.effector_profiles[
                                profile_name
                            ].display_name,
                            "normalization": {
                                "source": variant.normalization_source,
                                "label": variant.normalization_label,
                                "locked": True,
                            },
                            "default_version": variant.default_version,
                            "versions": [
                                {
                                    "id": version.id,
                                    "label": version.display_name,
                                    "description": version.description,
                                    "rtc_inference_mode": rtc_inference_mode_for_args(
                                        (*version.base_args, *version.real_args)
                                    ),
                                    **self.deployment.model_rtc_metadata(
                                        model, version.effector_profile, version.id
                                    ),
                                    "normalization": {
                                        "source": version.normalization_source,
                                        "label": version.normalization_label,
                                        "locked": True,
                                    },
                                }
                                for version in variant.versions.values()
                            ],
                        }
                        for profile_name, variant in model.effector_variants.items()
                    },
                    "normalization": {
                        "source": model.normalization_source,
                        "label": model.normalization_label,
                        "locked": True,
                    },
                    "ui_only": False,
                    "selectable": True,
                    "loadable": True,
                    "integration_status": "available",
                }
            )
        # UI-only models deliberately stay outside ``deployment.models``.  They
        # can be shown as roadmap entries without acquiring a GPU, exposing a
        # fake health state, or becoming valid deployment payloads.
        for model_id, raw_value in self.deployment.raw.get("ui_models", {}).items():
            raw = raw_value if isinstance(raw_value, Mapping) else {}
            if raw.get("ui_hidden") is True:
                continue
            modes = [str(item) for item in raw.get("supported_modes", ())]
            effector_profiles = [
                canonical_effector_profile(item)
                for item in raw.get("effector_profiles", ())
            ]
            raw_variants = raw.get("effector_variants", {})
            if not isinstance(raw_variants, Mapping):
                raw_variants = {}
            effector_variants = {}
            for profile_name in effector_profiles:
                raw_variant = raw_variants.get(
                    profile_name,
                    raw_variants.get("dh", {}) if profile_name == "dahuan" else {},
                )
                if not isinstance(raw_variant, Mapping):
                    raw_variant = {}
                profile = self.deployment.effector_profiles.get(profile_name)
                effector_variants[profile_name] = {
                    "profile": profile_name,
                    "label": (
                        profile.display_name if profile is not None else profile_name
                    ),
                    "normalization": {
                        "source": str(
                            raw_variant.get(
                                "normalization_source", "checkpoint_embedded"
                            )
                        ),
                        "label": str(
                            raw_variant.get(
                                "normalization_label", "Checkpoint 内置归一化"
                            )
                        ),
                        "locked": True,
                    },
                    "default_version": "default",
                    "versions": [
                        {
                            "id": "default",
                            "label": "默认配置",
                            "description": "该模型接入后由部署清单提供版本白名单。",
                            "normalization": {
                                "source": str(
                                    raw_variant.get(
                                        "normalization_source", "checkpoint_embedded"
                                    )
                                ),
                                "label": str(
                                    raw_variant.get(
                                        "normalization_label", "Checkpoint 内置归一化"
                                    )
                                ),
                                "locked": True,
                            },
                        }
                    ],
                }
            default_effector_profile = canonical_effector_profile(
                raw.get(
                    "default_effector_profile",
                    effector_profiles[0] if effector_profiles else "",
                )
            )
            default_variant = effector_variants.get(default_effector_profile, {})
            default_normalization = default_variant.get("normalization", {})
            models.append(
                {
                    "id": str(model_id),
                    "label": str(raw.get("display_name", model_id)),
                    "description": str(raw.get("description", "界面预配置策略模型")),
                    "port": raw.get("port"),
                    "gpu": None,
                    "configured_gpu": str(raw.get("cuda_visible_devices", "auto")),
                    "gpu_min_free_memory_mb": int(
                        raw.get("gpu_min_free_memory_mb", 0)
                    ),
                    "startup_timeout_s": int(raw.get("startup_timeout_s", 0)),
                    "status": {
                        "running": False,
                        "healthy": False,
                        "available": False,
                    },
                    "supported_modes": modes,
                    "inference_modes": modes,
                    "rtc_execution_horizon": raw.get("rtc_execution_horizon"),
                    "rtc_execute_enabled": raw.get("rtc_execute_enabled", True) is True,
                    "tags": [str(item) for item in raw.get("tags", ())],
                    "effector_profile": default_effector_profile or None,
                    "effector_profile_label": (
                        str(default_variant.get("label"))
                        if default_variant
                        else "自动匹配"
                    ),
                    "effector_profiles": effector_profiles,
                    "running_effector_profile": None,
                    "running_model_version": None,
                    "effector_variants": effector_variants,
                    "normalization": {
                        "source": str(
                            default_normalization.get(
                                "source", raw.get("normalization_source", "checkpoint_embedded")
                            )
                        ),
                        "label": str(
                            default_normalization.get(
                                "label", raw.get("normalization_label", "Checkpoint 内置归一化")
                            )
                        ),
                        "locked": True,
                    },
                    "ui_only": True,
                    "selectable": True,
                    "loadable": False,
                    "compatible_robots": [
                        str(item) for item in raw.get("compatible_robots", ())
                    ],
                    "integration_status": "available",
                }
            )
        robots = []
        for robot_id, robot in self.deployment.robots.items():
            raw = self.deployment.raw.get("robots", {}).get(robot_id, {})
            if raw.get("ui_hidden") is True:
                continue
            network_endpoints = [
                {
                    "host": str(item["host"]),
                    "ssh_port": int(item["ssh_port"]),
                    "label": str(item["label"]),
                    "transport": str(item["transport"]),
                    "priority": int(item["priority"]),
                }
                for item in self.deployment.robot_network_endpoints(robot_id)
            ]
            site_execute_enabled = raw.get("execute_enabled") is True
            site_reset_enabled = raw.get("reset_enabled") is True
            effectors_mode = robot.effectors_mode
            effector_profile = self.deployment.effector_profiles[robot.effector_profile]
            robots.append(
                {
                    "id": robot_id,
                    "label": str(raw.get("display_name", robot_id)),
                    "description": str(raw.get("description", "已配置的机器人部署目标")),
                    "online": None,
                    "configured": True,
                    "execution_supported": self.enable_execute
                    and site_execute_enabled
                    and effectors_mode == "required",
                    "execute_configured": site_execute_enabled,
                    "reset_supported": self.enable_reset and site_reset_enabled,
                    "reset_configured": site_reset_enabled,
                    "effectors": {
                        "mode": effectors_mode,
                        "present": effectors_mode == "required",
                        "profile": effector_profile.name,
                        "label": effector_profile.display_name,
                        "available_profiles": list(robot.effector_profiles),
                        "unit": effector_profile.unit,
                        "hardware_state_range": list(effector_profile.hardware_state_range),
                        "model_action_range": list(effector_profile.model_action_range),
                        "state_semantics": effector_profile.state_semantics,
                        "action_semantics": effector_profile.action_semantics,
                        "action_mapping": effector_profile.action_mapping,
                        "limits": {
                            "max_velocity": effector_profile.default_max_velocity,
                            "max_acceleration": effector_profile.default_max_acceleration,
                            "tracking_error": effector_profile.default_tracking_error,
                            "relative_target_warning": (
                                effector_profile.default_relative_target_warning
                            ),
                        },
                        "collision_geometry": effector_profile.collision_geometry,
                        "collision_geometry_status": effector_profile.collision_geometry_status,
                        "configuration_locked": robot.effectors_mode != "required"
                        or len(robot.effector_profiles) <= 1,
                        "observation_source": (
                            "gdk" if effectors_mode == "required" else "fixed_training_reference"
                        ),
                        "fixed_state": (
                            list(robot.fixed_effector_state)
                            if effectors_mode == "absent_fixed"
                            else None
                        ),
                    },
                    "compatible_models": [
                        str(item) for item in raw.get("compatible_models", tuple(self.deployment.models))
                    ],
                    "safety_profiles": self._safety_profiles(robot),
                    "ui_only": False,
                    "adapter_status": "available",
                    "tags": [str(item) for item in raw.get("tags", ())],
                    "specifications": {},
                    "connection": {
                        "host": str(raw.get("control_host", "")),
                        "ssh_port": int(raw.get("ssh_port", 22)),
                        "preferred_transport": str(
                            raw.get("control_host_transport", "wired")
                        ),
                        "endpoints": network_endpoints,
                        "runtime_status": "not_checked",
                    },
                }
            )
        for robot_id, raw_value in self.deployment.raw.get("ui_robots", {}).items():
            raw = raw_value if isinstance(raw_value, Mapping) else {}
            if raw.get("ui_hidden") is True:
                continue
            robots.append(
                {
                    "id": str(robot_id),
                    "label": str(raw.get("display_name", robot_id)),
                    "manufacturer": str(raw.get("manufacturer", "")),
                    "description": str(raw.get("description", "界面预配置机器人")),
                    "online": None,
                    "configured": True,
                    "execution_supported": False,
                    "compatible_models": [],
                    "safety_profiles": [],
                    "ui_only": True,
                    "adapter_status": str(raw.get("adapter_status", "planned")),
                    "urdf_status": str(raw.get("urdf_status", "unavailable")),
                    "official_url": str(raw.get("official_url", "")),
                    "tags": [str(item) for item in raw.get("tags", ())],
                    "specifications": {
                        "height_m": raw.get("height_m"),
                        "arm_span_m": raw.get("arm_span_m"),
                        "payload_per_arm_kg": raw.get("payload_per_arm_kg"),
                        "endurance_hours": list(raw.get("endurance_hours", ())),
                        "arm_dof": raw.get("arm_dof"),
                    },
                    "connection": {
                        "host": str(raw.get("control_host", "")),
                        "ssh_port": int(raw.get("ssh_port", 22)),
                        "runtime_status": "not_checked",
                    },
                }
            )
        defaults = copy.deepcopy(self.DEFAULTS)
        if robots:
            defaults["robot_id"] = robots[0]["id"]
            defaults["robot_host"] = robots[0]["connection"]["host"]
            defaults["robot_ssh_port"] = robots[0]["connection"]["ssh_port"]
            configured_robot = self.deployment.robots.get(str(robots[0]["id"]))
            if configured_robot is not None:
                defaults["effector_profile"] = configured_robot.effector_profile
                defaults["effector_control_mode"] = "device-position" if configured_robot.effector_profile == "dahuan" else "software-trajectory"
                profile = self.deployment.effector_profiles[
                    configured_robot.effector_profile
                ]
                defaults["task"] = profile.default_task
                defaults["max_effector_velocity"] = profile.default_max_velocity
                defaults["max_effector_acceleration"] = profile.default_max_acceleration
                defaults["max_effector_tracking_error"] = profile.default_tracking_error
        selectable_models = [
            item
            for item in models
            if item.get("selectable", True) and item.get("loadable", True)
        ]
        if selectable_models:
            # Prefer a healthy reusable service. Otherwise choose a stopped
            # model that can be loaded, rather than an unhealthy live process.
            selected_model = next(
                (
                    item
                    for item in selectable_models
                    if item["status"].get("running") and item["status"].get("healthy")
                ),
                None,
            ) or next(
                (item for item in selectable_models if not item["status"].get("running")),
                selectable_models[0],
            )
            defaults["model_id"] = selected_model["id"]
            selected_variant = selected_model.get("effector_variants", {}).get(
                defaults["effector_profile"], {}
            )
            defaults["model_version"] = str(
                selected_variant.get("default_version", "")
            )
            selected_version = next(
                (item for item in selected_variant.get("versions", [])
                 if item["id"] == defaults["model_version"]), {}
            )
            defaults["rtc_inference_mode"] = selected_version.get("rtc_inference_mode_default", "guided")
            defaults["server_mode"] = (
                "reuse"
                if selected_model["status"].get("running")
                and selected_model["status"].get("healthy")
                and selected_model.get("running_effector_profile")
                == defaults["effector_profile"]
                and selected_model.get("running_model_version")
                == defaults["model_version"]
                and selected_model.get("running_rtc_inference_mode") == defaults["rtc_inference_mode"]
                else "real"
            )
        return {
            "csrf_token": self.csrf_token,
            "project": {
                "name": "Nexus",
                "company": "成都安努智能技术有限公司",
                "protocol": POLICY_PROTOCOL_NAME,
                "site_id": self.deployment.raw.get("host", {}).get("site_id", ""),
            },
            "tools": {
                "alignment_url": self.alignment_url,
                "alignment_surface": self.active_alignment_surface(),
            },
            "simulator": {
                "enabled": self.simulator_control_url is not None,
                "viewer_port": self.simulator_viewer_port,
                "control_port": (
                    urlparse(self.simulator_control_url).port
                    if self.simulator_control_url is not None
                    else None
                ),
                "renderer": "缓存 GLB / 浏览器 FK / MeshCat THREE",
                "update_hz": 8,
            },
            "models": models,
            "effector_profiles": [
                {
                    "id": profile.name,
                    "label": profile.display_name,
                    "default_task": profile.default_task,
                    "unit": profile.unit,
                    "hardware_state_range": list(profile.hardware_state_range),
                    "model_action_range": list(profile.model_action_range),
                    "state_semantics": profile.state_semantics,
                    "action_semantics": profile.action_semantics,
                    "action_mapping": profile.action_mapping,
                    "limits": {
                        "max_velocity": profile.default_max_velocity,
                        "max_acceleration": profile.default_max_acceleration,
                        "tracking_error": profile.default_tracking_error,
                        "relative_target_warning": profile.default_relative_target_warning,
                    },
                    "collision_geometry": profile.collision_geometry,
                    "collision_geometry_status": profile.collision_geometry_status,
                }
                for profile in self.deployment.effector_profiles.values()
                if not profile.ui_hidden
            ],
            "gpus": gpus,
            "gpu_recommendations": gpu_status["recommendations"],
            "robots": robots,
            "defaults": defaults,
            "constraints": {
                "physical_execute_available": self.enable_execute
                and any(item["execution_supported"] for item in robots),
                "actions_enabled": self.enable_actions,
                "execute_process_enabled": self.enable_execute,
                "execute_confirmation_format": "double_click",
                "robot_reset_available": self.enable_reset
                and any(item["reset_supported"] for item in robots),
                "robot_reset_confirmation_format": "double_click",
                "task_max_length": 4096,
                "human_in_loop_scope": "g2_execute_rollback_collection",
            },
        }

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate configuration without weakening the execution boundary.

        Model preparation submits a read-only baseline. Completion and safety
        controls are saved after loading and checked again before task start.
        """
        if not isinstance(payload, Mapping):
            raise ValidationError("请求必须是 JSON 对象")
        supplied = dict(payload)
        allowed = set(self.DEFAULTS)
        errors: list[dict[str, str]] = []
        warnings: list[str] = []
        for key in supplied:
            if key not in allowed:
                errors.append(
                    {
                        "field": key,
                        "message": "不支持的参数；控制面不接受原始命令或 shell 参数",
                    }
                )
        normalized = copy.deepcopy(self.DEFAULTS)
        for key in allowed - {"safety", "presence_gate"}:
            if key in supplied:
                normalized[key] = supplied[key]
        supplied_safety = supplied.get("safety", {})
        if supplied_safety is not None and not isinstance(supplied_safety, Mapping):
            errors.append({"field": "safety", "message": "安全配置必须是对象"})
            supplied_safety = {}
        safety_allowed = set(normalized["safety"])
        for key in supplied_safety:
            if key not in safety_allowed:
                errors.append({"field": f"safety.{key}", "message": "不支持的安全参数"})
        normalized["safety"].update({key: supplied_safety[key] for key in safety_allowed if key in supplied_safety})

        # Package-presence gating is intentionally a narrow, declarative
        # option. No caller can inject a URL, command, timing threshold or
        # auto-resume behavior. Nexus owns the live adapter and uses only its
        # existing fenced pause/reset APIs after its own ownership checks.
        supplied_presence_gate = supplied.get("presence_gate", {})
        if supplied_presence_gate is not None and not isinstance(supplied_presence_gate, Mapping):
            errors.append({"field": "presence_gate.enabled", "message": "包裹检测配置必须是对象"})
            supplied_presence_gate = {}
        presence_gate_allowed = {"enabled"}
        for key in supplied_presence_gate:
            if key not in presence_gate_allowed:
                errors.append(
                    {
                        "field": f"presence_gate.{key}",
                        "message": "该包裹检测参数由服务端固定，不能由浏览器修改",
                    }
                )
        if "enabled" in supplied_presence_gate:
            normalized["presence_gate"]["enabled"] = supplied_presence_gate["enabled"]
        if not isinstance(normalized["presence_gate"].get("enabled"), bool):
            errors.append({"field": "presence_gate.enabled", "message": "包裹检测开关必须是布尔值"})
            normalized["presence_gate"]["enabled"] = False

        # Approval timing belonged to the removed HITL gate. Keep the legacy
        # field stable for persisted/API compatibility; tablet-start admission
        # now follows the same operator task gate as the web Start action.
        normalized["approval_timeout_s"] = self.DEFAULTS["approval_timeout_s"]

        if not isinstance(normalized["human_in_loop"], bool):
            errors.append({"field": "human_in_loop", "message": "必须是布尔值"})
        for field_name in safety_allowed - {"safety_profile_id"}:
            if not isinstance(normalized["safety"].get(field_name), bool):
                errors.append({"field": f"safety.{field_name}", "message": "必须是布尔值"})

        robot_id = normalized["robot_id"]
        model_id = normalized["model_id"]
        effector_profile_was_supplied = "effector_profile" in supplied
        effector_profile_value = normalized.get("effector_profile", "")
        model_version_value = normalized.get("model_version", "")
        if not isinstance(robot_id, str):
            errors.append({"field": "robot_id", "message": "机器人标识必须是字符串"})
            robot_id = ""
        if not isinstance(model_id, str):
            errors.append({"field": "model_id", "message": "模型标识必须是字符串"})
            model_id = ""
        robot_id = robot_id.strip()
        model_id = model_id.strip()
        normalized["robot_id"] = robot_id
        normalized["model_id"] = model_id
        if not isinstance(effector_profile_value, str):
            errors.append({"field": "effector_profile", "message": "请选择夹爪类型"})
            effector_profile_value = ""
        effector_profile_name = canonical_effector_profile(effector_profile_value.strip())
        if not isinstance(model_version_value, str):
            errors.append({"field": "model_version", "message": "模型版本标识必须是字符串"})
            model_version_value = ""
        model_version_id = model_version_value.strip()
        normalized["model_version"] = model_version_id
        gpu_id_value = normalized.get("gpu_id", "auto")
        if not isinstance(gpu_id_value, str):
            errors.append({"field": "gpu_id", "message": "GPU 选择必须是字符串"})
            gpu_id_value = "auto"
        gpu_id = gpu_id_value.strip().lower()
        normalized["gpu_id"] = gpu_id
        ui_robots = self.deployment.raw.get("ui_robots", {})
        is_ui_robot = robot_id in ui_robots
        if robot_id not in self.deployment.robots and not is_ui_robot:
            errors.append({"field": "robot_id", "message": "请选择已配置的机器人"})
        if model_id not in self.deployment.models:
            errors.append({"field": "model_id", "message": "请选择已配置的模型"})

        # Older API callers did not send a gripper field. Resolve that case from
        # the selected robot/model pair, while keeping an explicit UI selection
        # strict. This preserves old saved jobs without allowing checkpoint and
        # hardware profiles to drift apart.
        if (
            not effector_profile_was_supplied
            and robot_id in self.deployment.robots
            and model_id in self.deployment.models
        ):
            selected_robot = self.deployment.robots[robot_id]
            selected_model = self.deployment.models[model_id]
            compatible_profiles = tuple(
                profile
                for profile in selected_robot.effector_profiles
                if profile in selected_model.effector_profiles
            )
            if selected_robot.effector_profile in compatible_profiles:
                effector_profile_name = selected_robot.effector_profile
            elif selected_model.effector_profile in compatible_profiles:
                effector_profile_name = selected_model.effector_profile
            elif compatible_profiles:
                effector_profile_name = compatible_profiles[0]

        normalized["effector_profile"] = effector_profile_name
        effector_profile = self.deployment.effector_profiles.get(effector_profile_name)
        if effector_profile is None or effector_profile.ui_hidden:
            errors.append({"field": "effector_profile", "message": "请选择可用的夹爪类型"})
        if normalized.get("server_mode", "reuse") == "real" and gpu_id != "auto":
            gpu_inventory = self.gpu_inventory()
            gpu_ids = {str(item["id"]) for item in gpu_inventory}
            if gpu_id not in gpu_ids:
                errors.append({"field": "gpu_id", "message": "请选择当前主机上可用的 GPU"})

        endpoint_error = self._normalize_robot_endpoint(normalized)
        if endpoint_error is not None:
            errors.append(endpoint_error)

        task_value = normalized["task"]
        if not isinstance(task_value, str):
            errors.append({"field": "task", "message": "任务描述必须是字符串"})
            task_value = ""
        task = task_value.strip()
        normalized["task"] = task
        if not task:
            errors.append({"field": "task", "message": "任务描述不能为空"})
        elif len(task) > 4096:
            errors.append({"field": "task", "message": "任务描述不能超过 4096 个字符"})

        if "effector_control_mode" not in supplied and normalized.get("effector_profile") != "dahuan":
            normalized["effector_control_mode"] = "software-trajectory"
        for field_name in ("inference_mode", "hardware_mode", "server_mode", "effector_control_mode",
                           "arm_limit_profile", "chunk_smoothing_mode", "action_interpolation",
                           "arm_tracking_mode", "rtc_prefix_mode"):
            value = normalized[field_name]
            if not isinstance(value, str):
                errors.append({"field": field_name, "message": "模式必须是字符串"})
                normalized[field_name] = ""
            else:
                normalized[field_name] = value.strip()

        if robot_id in self.deployment.robots and model_id in self.deployment.models:
            robot_raw = self.deployment.raw.get("robots", {}).get(robot_id, {})
            compatible = tuple(str(item) for item in robot_raw.get("compatible_models", ()))
            if compatible and model_id not in compatible:
                errors.append({"field": "model_id", "message": "所选机器人与模型未声明兼容"})
            selected_robot = self.deployment.robots[robot_id]
            selected_model = self.deployment.models[model_id]
            if effector_profile_name not in selected_robot.effector_profiles:
                errors.append(
                    {
                        "field": "effector_profile",
                        "message": "所选机器人不支持该夹爪配置",
                    }
                )
            if effector_profile_name not in selected_model.effector_profiles:
                errors.append(
                    {
                        "field": "model_id",
                        "message": "所选模型没有该夹爪对应的 checkpoint 与归一化配置",
                    }
                )
            else:
                variant = selected_model.variant(effector_profile_name)
                if not model_version_id:
                    model_version_id = variant.default_version
                    normalized["model_version"] = model_version_id
                elif model_version_id not in variant.versions:
                    errors.append(
                        {
                            "field": "model_version",
                            "message": "所选模型没有该夹爪对应的版本；只能使用部署清单中的版本",
                        }
                    )
            modes = self._model_modes(model_id, self.deployment.raw["models"][model_id])
            if normalized["inference_mode"] not in modes:
                errors.append(
                    {
                        "field": "inference_mode",
                        "message": f"模型不支持该推理模式；可用模式：{', '.join(modes)}",
                    }
                )
            warnings.append("state / action / 相机 schema 将在 WebSocket 握手时再次校验")
        elif is_ui_robot and model_id in self.deployment.models:
            warnings.append("运行前仍需通过机器人连通性和 WebSocket schema 握手校验")
            if self.enable_actions:
                errors.append({"field": "robot_id", "message": "该机器人尚未接入底层适配，不能执行部署动作"})

        if normalized["inference_mode"] not in {"sync", "async", "rtc"}:
            errors.append({"field": "inference_mode", "message": "推理模式无效"})
        if normalized["hardware_mode"] not in {"read-only", "execute"}:
            errors.append({"field": "hardware_mode", "message": "硬件模式无效"})
        if (
            normalized["hardware_mode"] == "execute"
            and normalized["inference_mode"] == "rtc"
            and model_id in self.deployment.models
            and self.deployment.raw["models"][model_id].get("rtc_execute_enabled", True)
            is not True
        ):
            errors.append(
                {
                    "field": "hardware_mode",
                    "message": "该模型的 RTC 尚未完成 read-only 验收，禁止真机执行",
                }
            )
        if normalized["hardware_mode"] == "execute":
            robot_raw = self.deployment.raw.get("robots", {}).get(robot_id, {})
            selected_robot = self.deployment.robots.get(robot_id)
            if selected_robot is not None and selected_robot.effectors_mode == "absent_fixed":
                errors.append(
                    {
                        "field": "hardware_mode",
                        "message": "该机器人未安装夹爪，末端状态为只读合成值，禁止真机执行",
                    }
                )
            if not self.enable_execute:
                errors.append(
                    {
                        "field": "hardware_mode",
                        "message": "控制服务未使用 --enable-execute 启动，真机执行保持锁定",
                    }
                )
            elif robot_raw.get("execute_enabled") is not True:
                errors.append(
                    {
                        "field": "hardware_mode",
                        "message": "该机器人未在部署清单中显式启用真机执行",
                    }
                )
            if normalized["server_mode"] == "mock":
                errors.append(
                    {"field": "server_mode", "message": "Mock 测试服务不能用于真机动作执行"}
                )
            try:
                duration = float(normalized.get("duration_s"))
            except (TypeError, ValueError):
                duration = math.nan
            if not math.isfinite(duration) or not 0 <= duration <= 300:
                errors.append(
                    {
                        "field": "duration_s",
                        "message": "真机执行时长必须为 0（一直执行）或 1 到 300 秒",
                    }
                )
            try:
                preflight_requests = float(normalized.get("execution_preflight_requests"))
            except (TypeError, ValueError):
                preflight_requests = math.nan
            if not math.isfinite(preflight_requests) or preflight_requests < 1:
                errors.append(
                    {
                        "field": "execution_preflight_requests",
                        "message": "真机执行至少需要 1 次动作预检请求",
                    }
                )
            if normalized["safety"].get("workspace_safety_enabled") is not True:
                warnings.append(
                    "真机执行未启用碰撞危险拦截；基础限位与看门狗仍启用，急停必须可用"
                )
        if normalized.get("arm_limit_profile") not in {
            "uniform", "dataset-v30", "dataset-v30-p99-margin",
            "urdf-velocity-p99-acceleration", "urdf-velocity-uniform-acceleration",
            "urdf-velocity-only",
        }:
            errors.append({"field": "arm_limit_profile", "message": "手臂动态限制配置无效"})
        elif normalized.get("arm_limit_profile") in {
            "dataset-v30-p99-margin", "urdf-velocity-p99-acceleration",
        }:
            warnings.append(
                "当前实验配置的P99加速度不是厂商认证上限；逐关节值仍受界面全局硬上限约束"
            )
        elif normalized.get("arm_limit_profile") == "urdf-velocity-only":
            warnings.append(
                "实验配置：手臂按URDF限速，但正常跟踪不限制软件加速度；"
                "可能产生较大的100 Hz目标突变。下方加速度值仅保留为安全停车计算的有限假设，"
                "不能证明硬件能够达到该减速度"
            )
        if normalized.get("chunk_smoothing_mode") not in {
            "adaptive-fixed-time", "savgol",
        }:
            errors.append({"field": "chunk_smoothing_mode", "message": "轨迹平滑方案无效"})
        if normalized.get("action_interpolation") not in {"linear", "monotone-cubic"}:
            errors.append({"field": "action_interpolation", "message": "100 Hz 插值方式无效"})
        elif normalized.get("action_interpolation") == "monotone-cubic":
            warnings.append(
                "实验插值：保持30 Hz节点与夹爪目标不变，使手臂参考速度连续；"
                "既有004500离线对比未改善跟踪误差，且加速度超限比例可能上升；"
                "真机测试须保留碰撞拦截、预检和急停准备"
            )

        if normalized.get("arm_tracking_mode") not in {"legacy", "damped"}:
            errors.append({"field": "arm_tracking_mode", "message": "手臂跟踪模式无效"})
        if normalized.get("rtc_prefix_mode") not in {"model", "command-shadow", "command"}:
            errors.append({"field": "rtc_prefix_mode", "message": "RTC 前缀方案无效"})
        elif normalized.get("rtc_prefix_mode") != "model":
            if normalized.get("inference_mode") != "rtc":
                errors.append({"field": "rtc_prefix_mode", "message": "执行前缀测试需要 RTC 模式"})
            if normalized.get("rtc_prefix_mode") == "command" and normalized.get("hardware_mode") == "execute":
                errors.append({"field": "rtc_prefix_mode", "message": "执行前缀推理目前仅支持只读测试；真机可选择前缀记录"})
        if normalized["server_mode"] not in {"reuse", "mock", "real"}:
            errors.append({"field": "server_mode", "message": "模型服务策略无效"})
        if normalized.get("effector_control_mode") not in {"software-trajectory", "device-position"}:
            errors.append({"field": "effector_control_mode", "message": "夹爪控制方式无效"})
        elif normalized.get("effector_control_mode") == "device-position" and normalized.get("effector_profile") != "dahuan":
            errors.append({"field": "effector_control_mode", "message": "设备位置模式仅支持大寰夹爪，请选择软件轨迹模式"})

        positive = (
            "arm_tracking_omega",
            "execution_fps",
            "servo_fps",
            "watchdog_timeout_s",
            "request_timeout_s",
            "min_replan_after_steps",
            "latency_window_size",
            "action_low_pass_cutoff_hz",
            "effector_action_low_pass_cutoff_hz",
            "max_arm_velocity",
            "max_arm_acceleration",
            "max_effector_velocity",
            "max_effector_acceleration",
            "max_head_waist_velocity",
            "max_head_waist_acceleration",
            "max_arm_tracking_error",
            "max_effector_tracking_error",
            "max_head_waist_tracking_error",
        )
        nonnegative = (
            "duration_s",
            "request_lead_margin_steps",
            "min_rtc_blend_steps",
            "warmup_requests",
            "execution_preflight_requests",
        )
        integers = {
            "prediction_horizon",
            "quality_horizon",
            "request_lead_margin_steps",
            "min_replan_after_steps",
            "latency_window_size",
            "min_rtc_blend_steps",
            "warmup_requests",
            "execution_preflight_requests",
        }
        for field_name in (*positive, *nonnegative, "prediction_horizon", "quality_horizon"):
            try:
                raw = normalized[field_name]
                if isinstance(raw, bool):
                    raise ValueError
                numeric = float(raw)
                if field_name in integers:
                    if not numeric.is_integer():
                        raise ValueError
                    value: int | float = int(numeric)
                else:
                    value = numeric
                if not math.isfinite(float(value)):
                    raise ValueError
                normalized[field_name] = value
                strictly_positive = field_name in positive or field_name in {
                    "prediction_horizon",
                    "quality_horizon",
                }
                minimum_ok = value > 0 if strictly_positive else value >= 0
                if not minimum_ok:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                comparator = (
                    "大于 0"
                    if field_name in positive
                    or field_name in {"prediction_horizon", "quality_horizon"}
                    else "不小于 0"
                )
                errors.append({"field": field_name, "message": f"必须是{comparator}的有限数值"})
        omega, servo_rate = normalized.get("arm_tracking_omega"), normalized.get("servo_fps")
        if (normalized.get("arm_tracking_mode") == "damped"
                and isinstance(omega, (int, float)) and isinstance(servo_rate, (int, float))
                and servo_rate > 0 and omega / servo_rate > .25):
            errors.append({"field": "arm_tracking_omega", "message": "阻尼参数除以伺服频率必须不超过 0.25"})
        arm_velocity = normalized.get("max_arm_velocity")
        arm_acceleration = normalized.get("max_arm_acceleration")
        if (
            normalized.get("arm_limit_profile") == "urdf-velocity-uniform-acceleration"
            and isinstance(arm_velocity, (int, float))
            and isinstance(arm_acceleration, (int, float))
            and arm_acceleration >= 8.0
        ):
            warnings.append(
                "手臂速度逐关节读取当前安全URDF；URDF未声明加速度，"
                "统一8 rad/s²是部署设定值而非厂商认证上限"
            )
        if (
            isinstance(normalized["quality_horizon"], int)
            and isinstance(normalized["prediction_horizon"], int)
            and normalized["quality_horizon"] > normalized["prediction_horizon"]
        ):
            errors.append({"field": "quality_horizon", "message": "可信步数不能超过预测步数"})
        selected_rtc_metadata = dict(rtc_inference_mode_default="guided", trained_rtc_supported=False)
        if model_id in self.deployment.models:
            rtc_model = self.deployment.models[model_id]
            if (effector_profile_name in rtc_model.effector_profiles and
                    model_version_id in rtc_model.variant(effector_profile_name).versions):
                selected_rtc_metadata = self.deployment.model_rtc_metadata(
                    rtc_model, effector_profile_name, model_version_id
                )
        selected_rtc_mode = normalized.get("rtc_inference_mode")
        if selected_rtc_mode == "":
            selected_rtc_mode = selected_rtc_metadata["rtc_inference_mode_default"]
        normalized["rtc_inference_mode"] = selected_rtc_mode
        try:
            validate_rtc_inference_mode(selected_rtc_mode, selected_rtc_metadata)
        except ValueError as error:
            errors.append({"field": "rtc_inference_mode", "message": str(error)})

        if model_id in self.deployment.models and normalized["inference_mode"] == "rtc":
            model_raw = self.deployment.raw["models"][model_id]
            variant_raw = model_raw.get("effector_variants", {}).get(effector_profile_name, {})
            version_raw = variant_raw.get("versions", {}).get(model_version_id, {})
            rtc_horizon = version_raw.get(
                "rtc_execution_horizon",
                variant_raw.get("rtc_execution_horizon", model_raw.get("rtc_execution_horizon")),
            )
            if (
                rtc_horizon is not None
                and isinstance(normalized["quality_horizon"], int)
                and normalized["quality_horizon"] < int(rtc_horizon)
            ):
                errors.append(
                    {
                        "field": "quality_horizon",
                        "message": f"RTC 可信步数不能低于服务端执行窗口 {rtc_horizon}",
                    }
                )
        servo_fps = normalized.get("servo_fps")
        if isinstance(servo_fps, (int, float)) and math.isfinite(float(servo_fps)):
            for field_name in ("action_low_pass_cutoff_hz", "effector_action_low_pass_cutoff_hz"):
                cutoff = normalized.get(field_name)
                if isinstance(cutoff, (int, float)) and cutoff >= servo_fps / 2:
                    errors.append(
                        {
                            "field": field_name,
                            "message": "低通截止频率必须低于 Servo Nyquist 频率",
                        }
                    )
            if (
                normalized["safety"].get("safety_profile_id")
                and not math.isclose(float(servo_fps), 100.0, rel_tol=0.0, abs_tol=1e-9)
            ):
                errors.append(
                    {
                        "field": "servo_fps",
                        "message": "桌面安全控制要求 Servo 保持 100 Hz；碰撞检测配置为 50 Hz",
                    }
                )

        for required in ("watchdog_enabled", "kinematic_limits_enabled", "tracking_error_enabled"):
            if normalized["safety"].get(required) is not True:
                errors.append({"field": f"safety.{required}", "message": "基础安全限制必须开启"})
        if effector_profile is not None:
            # Gripper-dependent tuning is atomic with the selected profile.
            # Browser-supplied values are intentionally ignored so checkpoint,
            # units, motion limits and tracking thresholds cannot be mixed.
            normalized["max_effector_velocity"] = effector_profile.default_max_velocity
            normalized["max_effector_acceleration"] = effector_profile.default_max_acceleration
            normalized["max_effector_tracking_error"] = effector_profile.default_tracking_error
        safety_enabled = bool(normalized["safety"].get("workspace_safety_enabled"))
        profiles = []
        if robot_id in self.deployment.robots:
            profiles = self._safety_profiles(self.deployment.robots[robot_id])
        selected_value = normalized["safety"].get("safety_profile_id", "")
        if selected_value is None:
            selected = ""
        elif isinstance(selected_value, str):
            selected = selected_value.strip()
        else:
            errors.append(
                {"field": "safety.safety_profile_id", "message": "安全配置标识必须是字符串"}
            )
            selected = ""
        normalized["safety"]["safety_profile_id"] = selected
        profile = next((item for item in profiles if item["id"] == selected), None)
        if safety_enabled and profile is None:
            errors.append({"field": "safety.safety_profile_id", "message": "请选择工作空间安全配置"})
        elif selected:
            if profile is None:
                errors.append({"field": "safety.safety_profile_id", "message": "请选择工作空间安全配置"})
            elif normalized["hardware_mode"] == "read-only" and not profile.get(
                "read_only_ready"
            ):
                errors.append(
                    {
                        "field": "safety.safety_profile_id",
                        "message": "安全配置尚未完成只读验证，不能启用",
                    }
                )
            elif normalized["hardware_mode"] == "execute" and not profile.get(
                "execute_approved"
            ):
                errors.append(
                    {
                        "field": "safety.safety_profile_id",
                        "message": "安全配置尚未完成真机标定与验收，不能用于真机执行",
                    }
                )
            elif (self.enable_actions or normalized["hardware_mode"] == "execute") and not profile.get(
                "deployable"
            ):
                errors.append(
                    {
                        "field": "safety.safety_profile_id",
                        "message": "安全配置尚未加入机器人发布清单，当前不能用于运行",
                    }
                )
            elif not safety_enabled:
                warnings.append("碰撞安全层为仅监测模式：风险会告警，但不会修改候选动作")
        else:
            # The preparation page cannot configure workspace geometry. The
            # runtime page presents its own availability state and controls.
            pass
        if normalized["presence_gate"]["enabled"]:
            # A read-only task uses the same authenticated camera stream,
            # four-frame gate and client pause/resume protocol as a physical
            # task.  The client itself is started with --read-only and cannot
            # dispatch GDK servo commands.  Its ten-second reset is recorded
            # as a read-only verification only; the approved physical reset
            # path below remains exclusive to hardware_mode=execute.
            if normalized["hardware_mode"] == "read-only":
                # Offline Mock/Replay explicitly simulates the robot result.
                # It never claims a physical reset or completed batch.
                pass
            else:
                # A physical autonomous-control contract needs the complete,
                # pre-approved pause/reset/restart path.
                if not (self.enable_actions and self.enable_execute and self.enable_reset):
                    errors.append(
                        {
                            "field": "presence_gate.enabled",
                            "message": "控制服务未同时启用真机执行与机器人复位能力",
                        }
                    )
                raw_robot = self.deployment.raw.get("robots", {}).get(str(robot_id), {})
                if not isinstance(raw_robot, Mapping) or raw_robot.get("reset_enabled") is not True:
                    errors.append(
                        {
                            "field": "presence_gate.enabled",
                            "message": "所选机器人未配置经批准的自动复位能力",
                        }
                    )
                if str(normalized.get("effector_profile", "")) != "dahuan":
                    errors.append(
                        {
                            "field": "presence_gate.enabled",
                            "message": "自动包裹检测当前仅支持大寰夹爪复位序列",
                        }
                    )
                if profile is None or not profile.get("execute_approved") or not selected:
                    errors.append(
                        {
                            "field": "presence_gate.enabled",
                            "message": "自动包裹检测需要已批准且已选择的工作空间安全配置",
                        }
                    )
            if normalized["server_mode"] == "mock":
                errors.append(
                    {
                        "field": "presence_gate.enabled",
                        "message": "自动包裹检测不能与 Mock 模型服务一起使用",
                    }
                )
            runtime = self._presence_runtime
            if not self.enable_actions:
                runtime = "offline"
            if runtime is None:
                errors.append(
                    {
                        "field": "presence_gate.enabled",
                        "message": "Thor 尚未配置包裹检测 sidecar 运行时",
                    }
                )
            elif runtime != "offline":
                expected_paths = {
                    "视觉项目目录": runtime.project_directory,
                    "视觉 Python": runtime.python_executable,
                    "包裹视觉权重": runtime.weights,
                    "头部相机规则": runtime.rules,
                    "sidecar 启动脚本": runtime.project_directory / "run_nexus_presence_sidecar.py",
                }
                unavailable = [label for label, path in expected_paths.items() if not path.exists()]
                if unavailable:
                    errors.append(
                        {
                            "field": "presence_gate.enabled",
                            "message": "包裹检测运行时文件不可用：" + "、".join(unavailable),
                        }
                    )
        if normalized["human_in_loop"]:
            warnings.append(
                "人在环路将在 G2 真机运行时启用 VR 接管、回退和录制封存；"
                "执行前需要已批准的工作区安全配置和 100 Hz 伺服"
            )
        if normalized["presence_gate"]["enabled"]:
            if normalized["hardware_mode"] == "read-only":
                warnings.append(
                    "只读观测将启动同一套头部相机、PI0.5、四帧包裹门控与客户端暂停／恢复链路；客户端不会发送机械臂命令。连续无包裹 5 秒仅记录只读复位验证，不调用物理复位。"
                )
            else:
                warnings.append(
                    "包裹检测启用后：连续四帧无包裹超过 1.5 秒会软暂停；同一连续无包裹状态到 5 秒会复用既有复位流程；连续四帧有包裹只恢复同一条由门控暂停的任务。"
                )
        if normalized["server_mode"] == "real":
            warnings.append("真实模型服务会在启动前重新检查 GPU 空闲显存和占用进程")
        elif normalized["server_mode"] == "mock":
            warnings.append(
                "Mock 不加载真实权重，仅用于服务启动、协议握手和只读数据链路测试，"
                "不能代表模型推理效果"
            )

        model_label = model_id
        robot_label = robot_id
        effector_label = effector_profile_name
        version_label = model_version_id
        if model_id in self.deployment.models:
            model_label = str(self.deployment.raw["models"][model_id].get("display_name", model_id))
            selected_model = self.deployment.models[model_id]
            if (
                effector_profile_name in selected_model.effector_profiles
                and model_version_id in selected_model.variant(effector_profile_name).versions
            ):
                version_label = selected_model.version(
                    effector_profile_name, model_version_id
                ).display_name
        if effector_profile is not None:
            effector_label = effector_profile.display_name
        if robot_id in self.deployment.robots:
            robot_label = str(self.deployment.raw.get("robots", {}).get(robot_id, {}).get("display_name", robot_id))
        elif is_ui_robot:
            robot_label = str(ui_robots[robot_id].get("display_name", robot_id))
        summary = {
            "route": f"{robot_label} → {effector_label} → {model_label} → {version_label}",
            "mode": str(normalized["inference_mode"]),
            "hardware": str(normalized["hardware_mode"]),
            "task": task,
            "human_in_loop": bool(normalized["human_in_loop"]),
            "包裹检测": (
                "已启用（只读门控验证；不发送机械臂命令）"
                if normalized["presence_gate"]["enabled"]
                and normalized["hardware_mode"] == "read-only"
                else "已启用（自动暂停／复位；仅恢复同一任务）"
                if normalized["presence_gate"]["enabled"]
                else "未启用"
            ),
            "safety_enabled": normalized["safety"].get("workspace_safety_enabled", False),
            "endpoint": f"{normalized['robot_host']}:{normalized['robot_ssh_port']}",
            "gpu": "自动选择" if gpu_id == "auto" else f"GPU {gpu_id}",
        }
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "normalized": normalized,
            "summary": summary,
        }

    @staticmethod
    def _saved_config_validation_payload(config: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare a persisted server-owned job configuration for revalidation.

        ``validate`` deliberately treats its argument as browser input, so it
        rejects service-owned package-gate timing and reset policy fields. A
        saved job necessarily contains the normalized form of those fields.
        At the start boundary that saved configuration is trusted, but it must
        still be revalidated without re-submitting immutable policy values.
        """

        payload = copy.deepcopy(dict(config))
        gate = payload.get("presence_gate")
        payload["presence_gate"] = {
            "enabled": gate.get("enabled", False) if isinstance(gate, Mapping) else False,
        }
        return payload

    @staticmethod
    def _normalize_robot_endpoint(normalized: dict[str, Any]) -> dict[str, str] | None:
        host = normalized.get("robot_host")
        if not isinstance(host, str):
            return {"field": "robot_host", "message": "机器人 IP 必须是字符串"}
        host = host.strip()
        normalized["robot_host"] = host
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return {"field": "robot_host", "message": "请输入有效的 IPv4 或 IPv6 地址"}
        if not (address.is_private or address.is_loopback):
            return {"field": "robot_host", "message": "机器人 IP 必须位于私有网络"}
        port = normalized.get("robot_ssh_port")
        try:
            if isinstance(port, bool):
                raise ValueError
            numeric = float(port)
            if not numeric.is_integer() or not 1 <= numeric <= 65535:
                raise ValueError
            normalized["robot_ssh_port"] = int(numeric)
        except (TypeError, ValueError, OverflowError):
            return {"field": "robot_ssh_port", "message": "SSH 端口必须是 1–65535 的整数"}
        return None

    def _robot_for_config(self, config: Mapping[str, Any]) -> Any:
        return self.deployment.select_robot_endpoint(
            str(config["robot_id"]),
            str(config["robot_host"]),
            int(config["robot_ssh_port"]),
        )

    def probe_robot(self, payload: Mapping[str, Any], csrf_token: str | None) -> dict[str, Any]:
        """Probe network reachability and, where configured, the deployed client state."""

        self._require_csrf(csrf_token)
        if not isinstance(payload, Mapping) or set(payload) - {"robot_id", "robot_host", "robot_ssh_port"}:
            raise ValidationError("机器人检测参数无效")
        robot_id = str(payload.get("robot_id", "")).strip()
        known = robot_id in self.deployment.robots or robot_id in self.deployment.raw.get("ui_robots", {})
        if not known:
            raise ValidationError("机器人不存在")
        endpoint = {
            "robot_host": payload.get("robot_host"),
            "robot_ssh_port": payload.get("robot_ssh_port"),
        }
        error = self._normalize_robot_endpoint(endpoint)
        if error is not None:
            raise ValidationError(error["message"])
        started = time.monotonic()
        reachable = False
        try:
            with socket.create_connection(
                (endpoint["robot_host"], endpoint["robot_ssh_port"]), timeout=0.8
            ):
                reachable = True
        except OSError:
            pass
        result: dict[str, Any] = {
            "robot_id": robot_id,
            "host": endpoint["robot_host"],
            "port": endpoint["robot_ssh_port"],
            "reachable": reachable,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "checked_at": time.time(),
            "scope": "network_and_client",
            "ssh_authenticated": False if reachable else None,
            "client": {
                "status": "not_checked",
                "installed": None,
                "running": None,
                "processes": [],
                "protocol": None,
            },
            "runtime_session": "not_checked",
            "identity": {"expected_aid": None, "observed_aid": None, "match": None},
            "observation": {
                "ready": False,
                "status": "not_checked",
                "ptp_ready": None,
                "services_ready": None,
                "reason": "尚未执行机器人运行环境检查",
            },
        }
        if not reachable or robot_id not in self.deployment.robots:
            return result

        configured_endpoints = {
            (str(item["host"]), int(item["ssh_port"]))
            for item in self.deployment.robot_network_endpoints(robot_id)
        }
        selected_endpoint = (endpoint["robot_host"], endpoint["robot_ssh_port"])
        if selected_endpoint not in configured_endpoints:
            result["credential_match"] = False
            result["client"]["reason"] = "该地址尚未绑定平台 SSH 凭据，未执行客户端检查"
            return result

        result["credential_match"] = True
        selected_robot = self.deployment.select_robot_endpoint(
            robot_id, endpoint["robot_host"], endpoint["robot_ssh_port"]
        )
        result["identity"]["expected_aid"] = selected_robot.expected_aid or None
        try:
            client = self.deployment.probe_robot_client(selected_robot)
        except DeployError as error:
            result["client"]["reason"] = str(error)
            return result
        result["ssh_authenticated"] = True
        result["client"] = client
        result["runtime_session"] = "running" if client.get("running") else "idle"
        try:
            runtime = self.deployment.robot_runtime_preflight(selected_robot)
            observed_aid = str(runtime.get("asset_id", "")).strip()
            expected_aid = selected_robot.expected_aid.strip()
            identity_matches = bool(observed_aid and (not expected_aid or observed_aid == expected_aid))
            result["identity"] = {
                "expected_aid": expected_aid or None,
                "observed_aid": observed_aid or None,
                "match": identity_matches,
            }
            if expected_aid and not identity_matches:
                result["client"]["status"] = "wrong_robot"
                result["client"]["reason"] = (
                    f"机器人硬件身份不匹配：期望 {expected_aid}，"
                    f"实际 {observed_aid or '无法读取'}"
                )
                result["observation"] = {
                    "ready": False,
                    "status": "blocked",
                    "ptp_ready": None,
                    "services_ready": None,
                    "reason": result["client"]["reason"] + "；已阻止部署、观测和真机执行",
                }
                lease_key = (
                    robot_id,
                    str(endpoint["robot_host"]),
                    int(endpoint["robot_ssh_port"]),
                )
                with self._lock:
                    self._probe_leases[lease_key] = copy.deepcopy(result)
                return result
            ptp_ready = any(
                isinstance(item, Mapping) and item.get("readable") and item.get("writable")
                for item in runtime.get("ptp_devices", ())
            )
            process_text = "\n".join(str(item) for item in runtime.get("gdk_processes", ()))
            missing_services = [
                label
                for label, marker in (
                    ("GDK runtime", "gdk_service"),
                    ("GDK HTTP", "gdk_http_server"),
                    ("camera_service", "camera_service"),
                    ("Fast DDS discovery", "fastdds discovery"),
                )
                if marker not in process_text
            ]
            services_ready = not missing_services
            # Some supported GDK builds log an unusable PTP device but still
            # initialize and publish valid robot observations.  Static device
            # permissions are therefore advisory; the launched client remains
            # responsible for validating joints, effectors, and camera frames.
            observation_ready = bool(services_ready)
            reasons = []
            if not ptp_ready:
                reasons.append("PTP 设备权限不可用（仅警告，将在 GDK 运行时验证观测）")
            if missing_services:
                reasons.append("缺少 " + "、".join(missing_services))
            result["observation"] = {
                "ready": observation_ready,
                "status": (
                    "ready_with_warning"
                    if observation_ready and not ptp_ready
                    else "ready"
                    if observation_ready
                    else "blocked"
                ),
                "ptp_ready": bool(ptp_ready),
                "services_ready": services_ready,
                "external_gdk_clients": list(runtime.get("external_gdk_clients", ())),
                "reason": (
                    "；".join(reasons)
                    if reasons
                    else "GDK 基础服务已就绪，关节与相机将在运行时验证"
                ),
            }
            external_clients = result["observation"]["external_gdk_clients"]
            if external_clients:
                pids = "、".join(str(item.get("pid", "?")) for item in external_clients)
                warning = f"检测到其他 GDK 客户端进程（PID {pids}）；只读可继续，真机执行需独占"
                current_reason = result["observation"]["reason"]
                result["observation"]["reason"] = (
                    f"{current_reason}；{warning}" if current_reason else warning
                )
                if observation_ready:
                    result["observation"]["status"] = "ready_with_warning"
        except (DeployError, OSError, subprocess.SubprocessError) as error:
            result["observation"].update({"status": "error", "reason": str(error)})
        lease_key = (robot_id, str(endpoint["robot_host"]), int(endpoint["robot_ssh_port"]))
        with self._lock:
            self._probe_leases[lease_key] = copy.deepcopy(result)
        return result

    @staticmethod
    def _cleanup_target_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"robot_id", "robot_host", "robot_ssh_port", "session_id", "pid"}
        if not isinstance(payload, Mapping) or set(payload) - allowed:
            raise ValidationError("会话清理参数无效")
        session_id = str(payload.get("session_id", "")).strip()
        if not re.fullmatch(r"[0-9a-f]{1,64}", session_id):
            raise ValidationError("会话 ID 无效")
        try:
            pid = int(payload.get("pid"))
        except (TypeError, ValueError) as error:
            raise ValidationError("会话 PID 无效") from error
        if pid <= 1:
            raise ValidationError("会话 PID 无效")
        return {
            "robot_id": str(payload.get("robot_id", "")).strip(),
            "robot_host": str(payload.get("robot_host", "")).strip(),
            "robot_ssh_port": payload.get("robot_ssh_port"),
            "session_id": session_id,
            "pid": pid,
        }

    def _verify_cleanup_target(
        self, target: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        probe = self.probe_robot(
            {
                "robot_id": target["robot_id"],
                "robot_host": target["robot_host"],
                "robot_ssh_port": target["robot_ssh_port"],
            },
            self.csrf_token,
        )
        client = probe.get("client", {})
        if (
            not probe.get("reachable")
            or not probe.get("ssh_authenticated")
            or probe.get("identity", {}).get("match") is False
            or not isinstance(client, Mapping)
            or client.get("status") != "running"
            or not client.get("valid")
        ):
            raise ValidationError(
                "机器人身份、客户端完整性或当前运行状态已变化，请重新检测",
                status=HTTPStatus.CONFLICT,
            )
        matches = [
            item
            for item in client.get("processes", ())
            if isinstance(item, Mapping)
            and str(item.get("session_id") or "") == target["session_id"]
            and item.get("pid") == target["pid"]
            and item.get("managed") is True
        ]
        if len(matches) != 1:
            raise ValidationError(
                "待关闭会话的 PID 或 session 已变化，请重新检测",
                status=HTTPStatus.CONFLICT,
            )
        return probe, dict(matches[0])

    def prepare_robot_session_cleanup(
        self, payload: Mapping[str, Any], csrf_token: str | None
    ) -> dict[str, Any]:
        """First confirmation: re-check and bind one exact occupied session."""

        self._require_csrf(csrf_token)
        if not self.enable_actions:
            raise ValidationError("控制服务未启用机器人操作", status=HTTPStatus.FORBIDDEN)
        target = self._cleanup_target_payload(payload)
        probe, process = self._verify_cleanup_target(target)
        mode = str(process.get("hardware_mode") or "unknown")
        phrase = (
            f"强制关闭真机 {target['session_id'][:8]}"
            if mode == "execute"
            else f"强制关闭会话 {target['session_id'][:8]}"
        )
        challenge = secrets.token_urlsafe(32)
        expires_at = time.time() + 60.0
        with self._lock:
            now = time.time()
            self._cleanup_challenges = {
                token: item
                for token, item in self._cleanup_challenges.items()
                if float(item.get("expires_at", 0)) > now
            }
            self._cleanup_challenges[challenge] = {
                **target,
                "hardware_mode": mode,
                "confirmation_phrase": phrase,
                "expires_at": expires_at,
            }
        return {
            "challenge_token": challenge,
            "expires_at": expires_at,
            "confirmation_phrase": phrase,
            "target": {
                "robot_id": target["robot_id"],
                "host": probe["host"],
                "port": probe["port"],
                "session_id": target["session_id"],
                "pid": target["pid"],
                "hardware_mode": mode,
            },
        }

    def cleanup_robot_session(
        self, payload: Mapping[str, Any], csrf_token: str | None
    ) -> dict[str, Any]:
        """Second confirmation: safely stop the previously bound Nexus session."""

        self._require_csrf(csrf_token)
        if not self.enable_actions:
            raise ValidationError("控制服务未启用机器人操作", status=HTTPStatus.FORBIDDEN)
        allowed = {
            "robot_id",
            "robot_host",
            "robot_ssh_port",
            "session_id",
            "pid",
            "challenge_token",
            "confirmation_phrase",
        }
        if not isinstance(payload, Mapping) or set(payload) - allowed:
            raise ValidationError("会话清理参数无效")
        target = self._cleanup_target_payload(
            {
                key: payload.get(key)
                for key in allowed
                if key not in {"challenge_token", "confirmation_phrase"}
            }
        )
        challenge_token = str(payload.get("challenge_token", ""))
        with self._lock:
            challenge = self._cleanup_challenges.pop(challenge_token, None)
        if not challenge or float(challenge.get("expires_at", 0)) <= time.time():
            raise ValidationError("二次确认已过期，请重新开始", status=HTTPStatus.CONFLICT)
        bound_fields = ("robot_id", "robot_host", "robot_ssh_port", "session_id", "pid")
        if any(target[field] != challenge.get(field) for field in bound_fields):
            raise ValidationError("会话目标已变化，请重新开始", status=HTTPStatus.CONFLICT)
        supplied_phrase = str(payload.get("confirmation_phrase", ""))
        expected_phrase = str(challenge["confirmation_phrase"])
        if not secrets.compare_digest(
            supplied_phrase.encode("utf-8"), expected_phrase.encode("utf-8")
        ):
            raise ValidationError("二次确认文字不匹配")

        probe, process = self._verify_cleanup_target(target)
        mode = str(process.get("hardware_mode") or "unknown")
        if mode != challenge.get("hardware_mode"):
            raise ValidationError("会话运行模式已变化，请重新检测", status=HTTPStatus.CONFLICT)
        robot = self.deployment.select_robot_endpoint(
            target["robot_id"], target["robot_host"], int(target["robot_ssh_port"])
        )
        with self._lock:
            tracked_job = self._jobs.get(target["session_id"])
            already_paused = tracked_job is not None and tracked_job.status == "paused"
        if mode == "execute" and not already_paused:
            try:
                self.deployment.set_robot_session_paused(
                    robot, target["session_id"], True, timeout_s=12.0
                )
            except (DeployError, OSError, subprocess.SubprocessError) as error:
                raise ValidationError(
                    f"真机会话未确认进入安全暂停，已拒绝强制关闭：{error}",
                    status=HTTPStatus.CONFLICT,
                ) from error
        try:
            self.deployment.force_stop_robot_session(
                robot, target["session_id"], target["pid"], timeout_s=5.0
            )
        except (DeployError, OSError, subprocess.SubprocessError) as error:
            raise ValidationError(
                f"强制关闭会话失败：{error}", status=HTTPStatus.CONFLICT
            ) from error

        with self._lock:
            tracked_job = self._jobs.get(target["session_id"])
            if tracked_job is not None:
                tracked_job.task_session["status"] = "stopped"
                tracked_job.task_session["remote_started"] = False
                tracked_job.task_session["cancel_requested"] = True
                self._mark_telemetry_inactive(
                    tracked_job, "ended", "会话已被操作员强制关闭；画面为结束前最后一帧"
                )
                if tracked_job.status not in TERMINAL_JOB_STATES:
                    self._event(tracked_job, "succeeded", "会话已由操作员强制关闭", 100)
                self._release_robot(tracked_job)
                self._release_model(tracked_job)
            self._probe_leases.pop(
                (target["robot_id"], target["robot_host"], int(target["robot_ssh_port"])),
                None,
            )
        refreshed = self.probe_robot(
            {
                "robot_id": target["robot_id"],
                "robot_host": target["robot_host"],
                "robot_ssh_port": target["robot_ssh_port"],
            },
            self.csrf_token,
        )
        return {
            "stopped": True,
            "forced": True,
            "session_id": target["session_id"],
            "pid": target["pid"],
            "hardware_mode": mode,
            "probe": refreshed,
        }

    def _require_robot_probe(
        self, config: Mapping[str, Any], *, require_observation: bool, refresh: bool = False
    ) -> dict[str, Any]:
        key = (
            str(config["robot_id"]),
            str(config["robot_host"]),
            int(config["robot_ssh_port"]),
        )
        with self._lock:
            probe = copy.deepcopy(self._probe_leases.get(key))
        expired = not probe or time.time() - float(probe.get("checked_at", 0)) > PROBE_TTL_S
        if expired and refresh:
            probe = self.probe_robot(
                {"robot_id": key[0], "robot_host": key[1], "robot_ssh_port": key[2]},
                self.csrf_token,
            )
            expired = False
        if expired:
            raise ValidationError("机器人实例检测已过期，请重新检测", status=HTTPStatus.CONFLICT)
        client = probe.get("client", {})
        if (
            not probe.get("reachable")
            or not probe.get("ssh_authenticated")
            or not isinstance(client, Mapping)
            or client.get("status") != "idle"
        ):
            raise ValidationError("机器人客户端必须通过完整性检查且处于空闲状态", status=HTTPStatus.CONFLICT)
        if require_observation and not probe.get("observation", {}).get("ready"):
            reason = str(probe.get("observation", {}).get("reason") or "运行环境未就绪")
            raise ValidationError(f"只读观测预检失败：{reason}", status=HTTPStatus.CONFLICT)
        if (
            str(config.get("hardware_mode")) == "execute"
            and probe.get("observation", {}).get("external_gdk_clients")
        ):
            clients = probe["observation"]["external_gdk_clients"]
            pids = "、".join(str(item.get("pid", "?")) for item in clients)
            raise TaskError(
                "GDK_IN_USE",
                f"真机执行要求独占 GDK；占用进程 PID {pids}。"
                "请先由对应程序释放 GDK 会话并重新检测；本次真机请求未执行",
                HTTPStatus.CONFLICT,
            )
        return probe

    def _require_csrf(self, token: str | None) -> None:
        if not token or not secrets.compare_digest(token, self.csrf_token):
            raise ValidationError("CSRF token 无效", status=HTTPStatus.FORBIDDEN)

    def create_deployment(
        self, payload: Mapping[str, Any], csrf_token: str | None,
        *, batch: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_csrf(csrf_token)
        # Model preparation uses a neutral read-only baseline. The runtime
        # page saves completion and safety settings after the model is ready.
        validation = self.validate(payload)
        if not validation["valid"]:
            raise ValidationError(json.dumps(validation["errors"], ensure_ascii=False))
        if self.enable_actions:
            self._require_robot_probe(
                validation["normalized"], require_observation=False, refresh=False
            )
        job = DeploymentJob(id=uuid.uuid4().hex, config=validation["normalized"],
                            batch=copy.deepcopy(dict(batch or {})))
        model = self.deployment.models[job.config["model_id"]]
        job.model_service = {
            "model_id": model.name,
            "effector_profile": job.config["effector_profile"],
            "model_version": job.config["model_version"],
            "rtc_inference_mode": job.config["rtc_inference_mode"],
            "strategy": job.config["server_mode"],
            "status": "pending",
            "phase": "checking",
            "phase_label": "检查模型服务",
            "port": model.port,
            "gpu": (
                model.cuda_visible_devices
                if job.config["server_mode"] == "reuse"
                else job.config.get("gpu_id", "auto")
            ),
            "gpu_selection": job.config.get("gpu_id", "auto"),
            "gpu_snapshot": None,
            "timeout_s": model.startup_timeout_s,
            "started_at": None,
            "finished_at": None,
            "healthy": False,
            "recent_log": "等待模型服务检查",
            "shard_current": None,
            "shard_total": None,
            "owned_by_job": False,
            "pgid": None,
        }
        job.task_session = {
            "status": "not_started",
            "scope": "control_plane",
            "robot_runtime_connected": False,
            "safe_hold_available": False,
            "remote_started": False,
            "remote_pid": None,
            "generation": 0,
            "cancel_requested": False,
        }
        job.presence_gate = self._initial_presence_gate_status(job.config)
        job.robot_reset = {
            "status": "idle",
            "effectors_commanded": False,
            "effector_sequence": ["open", "close", "open"],
            "started_at": None,
            "finished_at": None,
            "message": "尚未执行机器人复位",
            "output_tail": "",
        }
        job.telemetry = {
            "available": False,
            "transport": "not_connected",
            "message": "任务尚未启动；尚未采集图像，也尚未发送推理请求",
            "observation_status": "not_received",
            "inference_status": "not_started",
            "last_observation_at": None,
            "last_response_at": None,
            "last_frame_at": None,
            "response_age_s": None,
            "frame_age_s": None,
            "request_count": 0,
            "cameras": [
                {"id": "cam_head", "label": "头部相机", "status": "waiting", "image_url": None},
                {"id": "cam_left_wrist", "label": "左腕相机", "status": "waiting", "image_url": None},
                {"id": "cam_right_wrist", "label": "右腕相机", "status": "waiting", "image_url": None},
            ],
            "request": {
                "request_id": None,
                "session_id": None,
                "rtt_ms": None,
                "inference_ms": None,
                "queue_depth": None,
                "chunk_id": None,
                "rtc_blend_steps": None,
            },
            "control": {
                "available": False,
                "sequence": None,
                "received_at": None,
                "captured_at": None,
                "hardware_mode": job.config["hardware_mode"],
                "measured_state": None,
                "model_candidate": None,
                "controller_candidate": None,
                "adopted_action": None,
                "safety": {
                    "enabled": bool(
                        job.config.get("safety", {}).get("workspace_safety_enabled")
                    ),
                    "monitoring_enabled": bool(
                        job.config.get("safety", {}).get("safety_profile_id")
                    ),
                    "enforcement_enabled": bool(
                        job.config.get("safety", {}).get("workspace_safety_enabled")
                    ),
                    "decision_available": False,
                    "collision_detected": False,
                    "intervention_active": False,
                },
            },
        }
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "优先配置接管期间不能创建新部署", 409)
            current_id = self.tasks.current_deployment_id()
            current = self._jobs.get(current_id) if current_id else None
            if current is not None:
                hardware = ("robot_id", "robot_host", "robot_ssh_port", "effector_profile")
                if any(current.config.get(key) != job.config.get(key) for key in hardware):
                    raise TaskError("HARDWARE_CONFLICT", "当前机器人或夹爪硬件不同；请走现场维护流程", 409)
            candidate_only = current is not None
            if self.enable_actions and job.config["server_mode"] in {"mock", "real"}:
                model_id = str(job.config["model_id"])
                owner = self._reserved_models.get(model_id)
                candidate_only = candidate_only or owner is not None or self.deployment.running_pid(model) is not None
                if not candidate_only:
                    self._reserved_models[model_id] = job.id
            job.model_service["candidate_only"] = candidate_only
            job.task_session["runtime_configured"] = False
            self._jobs[job.id] = job
            self._event(job, "queued", "模型配置已保存；等待运行参数预检" if candidate_only
                        else "部署配置已冻结；准备验证配置计划", 5)
            if job.batch:
                # Do not launch anything until the idempotency record is durable.
                try:
                    self._persist_state()
                except OSError:
                    job.batch.update(status="recovery_required", start_requested=True,
                                     reason="Unable to persist sorting request; no task was started")
                    raise
        threading.Thread(target=self._run_job, args=(job.id,), daemon=True).start()
        return job.to_dict()

    def _event(self, job: DeploymentJob, status: str, message: str, progress: int) -> None:
        job.status = status
        job.message = message
        job.progress = progress
        job.updated_at = time.time()
        job.revision += 1
        job.events.append({"time": job.updated_at, "revision": job.revision, "status": status, "message": message})
        try:
            self._persist_state()
        except OSError:
            # Runtime persistence must not turn a safe control transition into
            # an unsafe half-completed operation; the event remains in memory.
            pass

    def _mark_robot_reset_required(self, job: DeploymentJob, status: str, reason: str) -> None:
        self._workstation["reset_gate"] = dict(required=True, status=status,
                                                reason=reason, source_job_id=job.id,
                                                updated_at=time.time())

    def _mark_robot_reset_succeeded(self, job: DeploymentJob) -> None:
        now = time.time()
        for previous in self._jobs.values():
            if (previous.id != job.id and previous.config.get("hardware_mode") == "execute"
                    and (previous.task_session.get("accepted") or previous.batch.get("reset_required")
                         or previous.robot_reset.get("status") == "failed")
                    and previous.robot_reset.get("status") != "succeeded"):
                previous.task_session["reset_resolved_by"] = job.id
                previous.batch["reset_required"] = False
                previous.updated_at = now
                previous.events.append({"time": now, "status": "reset_resolved",
                                        "message": f"机器人已通过复位操作 {job.id} 完成核验"})
                if previous.status == "recovery_required" and not previous.task_session.get("remote_started"):
                    previous.status = "failed"
                    previous.task_session["status"] = "stopped"
                if (previous.batch.get("status") == "recovery_required"
                        and not previous.task_session.get("remote_started")):
                    previous.batch.update(status="failed",
                                          reason="历史任务已由后续机器人复位核实；任务未完成")
        self._workstation["reset_gate"] = dict(required=False, status="ready",
                                                reason="机器人复位已成功并核验",
                                                source_job_id=job.id, updated_at=now)

    def _run_job(self, job_id: str) -> None:
        try:
            steps = (
                ("validating", "正在验证参数、目标声明与安全边界", 20),
                ("preflight", "正在冻结模型与任务配置；连接能力待运行时握手", 35),
            )
            for status, message, progress in steps:
                with self._lock:
                    job = self._jobs[job_id]
                    self._event(job, status, message, progress)
                time.sleep(0.04)
            if not job.model_service.get("candidate_only"):
                self._prepare_model(job_id)
            else:
                with self._lock:
                    job.model_service.update(status="candidate", healthy=False,
                                             phase="candidate", phase_label="应用时加载模型",
                                             recent_log="新配置不会提前替换当前模型；应用时才加载")
            with self._lock:
                job = self._jobs[job_id]
                # Model readiness transitions to the operator start boundary;
                # HITL may subsequently start from a fresh tablet recording.
                job.approval_deadline = None
                self._mark_ready_to_start(job)
        except Exception as error:  # defensive boundary for background work
            with self._lock:
                job = self._jobs[job_id]
                job.error = str(error)
                self._event(job, "failed", "部署任务失败", job.progress)
                selected_gpu = str(job.model_service.get("gpu", ""))
                if self._reserved_gpus.get(selected_gpu) == job.id:
                    self._reserved_gpus.pop(selected_gpu, None)
                self._release_model(job)

    def _prepare_model(self, job_id: str) -> None:
        # The UI has no arbitrary command path. Opted-in actions are still
        # limited to model start and keep hardware execution unavailable.
        with self._lock:
            job = self._jobs[job_id]
            config = copy.deepcopy(job.config)
            job.model_service.update(
                {
                    "status": "starting",
                    "phase": "environment",
                    "phase_label": "准备模型环境",
                    "started_at": time.time(),
                    "recent_log": "正在准备受控模型启动命令",
                }
            )
            self._event(job, "deploying", "正在准备模型服务", 45)
        model = self.deployment.models[config["model_id"]]
        if config["server_mode"] == "reuse":
            if not self.enable_actions:
                with self._lock:
                    job = self._jobs[job_id]
                    job.model_service.update(
                        {
                            "status": "simulated",
                            "phase": "simulated",
                            "phase_label": "模拟复用检查",
                            "healthy": False,
                            "finished_at": time.time(),
                            "recent_log": "安全模拟模式未连接模型健康端点",
                        }
                    )
                    self._event(job, "model_ready", "模型复用计划已模拟完成", 78)
                return
            managed_pgid = self.deployment.running_pid(model)
            health = self.deployment.health(model, timeout_s=1.0) if managed_pgid is not None else None
            if health is None:
                raise DeployError(f"{model.name} 没有可复用的受管健康服务")
            running_profile = self.deployment.running_model_effector_profile(model)
            if running_profile != config["effector_profile"]:
                raise DeployError(
                    f"{model.name} 已运行服务的夹爪配置为 "
                    f"{running_profile or '未声明'}，当前选择为 {config['effector_profile']}；"
                    "请关闭模型后按当前夹爪配置重新加载"
                )
            running_version = self.deployment.running_model_version(model)
            if running_version != config["model_version"]:
                raise DeployError(
                    f"{model.name} 已运行服务的模型版本为 "
                    f"{running_version or '无法识别'}，当前选择为 {config['model_version']}；"
                    "请关闭模型后按当前版本重新加载"
                )
            running_rtc_mode = self.deployment.running_model_rtc_inference_mode(model)
            if running_rtc_mode != config["rtc_inference_mode"]:
                raise DeployError(
                    f"{model.name} 已运行服务 RTC 模式为 {running_rtc_mode or '无法识别'}，"
                    f"当前选择为 {config['rtc_inference_mode']}；请按当前模式重新加载模型"
                )
            with self._lock:
                job = self._jobs[job_id]
                resource = self._model_resources.setdefault(
                    model.name,
                    {"owner_job_id": None, "pgid": managed_pgid, "consumers": set()},
                )
                resource["pgid"] = managed_pgid
                resource["consumers"].add(job.id)
                job.model_service.update(
                    {
                        "status": "ready",
                        "phase": "healthy",
                        "phase_label": "健康检查通过",
                        "healthy": True,
                        "finished_at": time.time(),
                        "recent_log": "受管模型进程与健康端点检查通过；协议能力待任务握手确认",
                    }
                )
                self._event(job, "model_ready", "已运行的模型服务通过健康检查", 78)
            return
        started_pgid: int | None = None
        if self.enable_actions and config["server_mode"] in {"mock", "real"}:
            selected_gpu = (
                self._reserve_gpu(job)
                if config["server_mode"] == "real"
                else {
                    "id": "cpu",
                    "name": "Mock / CPU",
                    "memory_free_mb": None,
                    "memory_total_mb": None,
                    "memory_used_mb": None,
                    "utilization_percent": None,
                    "process_count": 0,
                    "processes": [],
                }
            )
            with self._lock:
                job = self._jobs[job_id]
                is_mock = config["server_mode"] == "mock"
                job.model_service.update(
                    {
                        "gpu": str(selected_gpu["id"]),
                        "gpu_snapshot": selected_gpu,
                        "recent_log": (
                            "已选择 CPU · Mock 测试不加载真实模型权重"
                            if is_mock
                            else f"已选择 GPU {selected_gpu['id']} · "
                            f"空闲 {selected_gpu.get('memory_free_mb', '未知')} MiB"
                        ),
                    }
                )
                self._event(
                    job,
                    "deploying",
                    (
                        "正在使用 CPU 启动 Mock 测试服务"
                        if is_mock
                        else f"已选择 GPU {selected_gpu['id']}，正在启动模型服务"
                    ),
                    48,
                )

            def report_startup(message: str) -> None:
                with self._lock:
                    active_job = self._jobs[job_id]
                    self._update_model_progress(active_job, message)

            started_pgid = self.deployment.start_model(
                model,
                mock=config["server_mode"] == "mock",
                wait_timeout_s=None,
                progress_callback=report_startup,
                cuda_visible_devices=(
                    "" if config["server_mode"] == "mock" else str(selected_gpu["id"])
                ),
                effector_profile=str(config["effector_profile"]),
                model_version=str(config["model_version"]),
                rtc_inference_mode=str(config["rtc_inference_mode"]),
            )
        with self._lock:
            job = self._jobs[job_id]
            simulated = not self.enable_actions and config["server_mode"] in {"mock", "real"}
            if started_pgid is not None:
                self._model_resources[model.name] = {
                    "owner_job_id": job.id,
                    "pgid": started_pgid,
                    "consumers": {job.id},
                }
            job.model_service.update(
                {
                    "status": "ready",
                    "phase": "healthy",
                    "phase_label": "健康检查通过" if not simulated else "模拟准备完成",
                    "healthy": not simulated,
                    "finished_at": time.time(),
                    "recent_log": (
                        "模型服务已启动并通过健康检查"
                        if not simulated
                        else "安全模拟模式未实际启动模型服务"
                    ),
                    "owned_by_job": started_pgid is not None,
                    "pgid": started_pgid,
                }
            )
            message = "模型服务已就绪" if not simulated else "模型启动计划已模拟完成"
            self._event(job, "model_ready", message, 78)

    def _update_model_progress(self, job: DeploymentJob, message: str) -> None:
        lowered = message.lower()
        phase = "weights"
        label = "加载模型权重"
        progress = 58
        if "initializing model" in lowered or "config" in lowered:
            phase, label, progress = "config", "加载模型配置", 52
        if "shard" in lowered:
            phase, label = "weights", "加载模型权重"
            match = re.search(r"shard\s+(\d+)\s*/\s*(\d+)", message, flags=re.IGNORECASE)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                job.model_service["shard_current"] = current
                job.model_service["shard_total"] = total
                progress = 54 + round(14 * current / max(total, 1))
        if "model initialized" in lowered or "all keys loaded" in lowered:
            phase, label, progress = "gpu", "搬入 GPU", 69
        if "warmup" in lowered:
            phase, label, progress = "warmup", "模型预热", 72
        if "server listening" in lowered or "health" in lowered:
            phase, label, progress = "health", "健康检查", 76
        job.model_service.update(
            {
                "status": "starting",
                "phase": phase,
                "phase_label": label,
                "recent_log": message,
            }
        )
        self._event(job, "deploying", message, progress)

    def _mark_ready_to_start(self, job: DeploymentJob) -> None:
        job.task_session["status"] = "ready"
        message = ("配置已保存；运行参数预检后可应用到工作站" if
                   job.model_service.get("candidate_only") else
                   "模型服务已就绪；请确认后开始任务")
        self._event(job, "ready_to_start", message, 86)

    def configure_deployment(
        self, job_id: str, payload: Mapping[str, Any], csrf_token: str | None
    ) -> dict[str, Any]:
        """Update post-load task settings while keeping model startup immutable."""

        self._require_csrf(csrf_token)
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "优先配置接管期间不能修改部署", 409)
            job = self._job_in_state(
                job_id,
                {"ready_to_start", "waiting_approval", "succeeded"},
                "只有模型就绪且任务未开始时才能修改运行配置",
            )
            current = self.tasks.current_deployment_id()
            if current and job.id != current and not job.model_service.get("candidate_only"):
                raise TaskError("SCHEME_NOT_ACTIVE", "旧任务配置不可重新应用；请从模型准备页建立新配置", 409)
            if job.batch:
                raise ValidationError("调度批次配置已冻结，请使用新的 request_id", status=HTTPStatus.CONFLICT)
            merged = self._saved_config_validation_payload(job.config)
            for key, value in payload.items():
                if key in {
                    "robot_id",
                    "robot_host",
                    "robot_ssh_port",
                    "model_id",
                    "model_version",
                    "effector_profile",
                    "server_mode",
                } and value != job.config[key]:
                    raise ValidationError(f"{key} 是模型加载前配置，当前作业中不可修改")
                if key == "safety" and isinstance(value, Mapping):
                    merged["safety"].update(value)
                else:
                    merged[key] = value
        validation = self.validate(merged)
        if not validation["valid"]:
            raise ValidationError(json.dumps(validation["errors"], ensure_ascii=False))
        if self.enable_actions:
            self._require_robot_probe(
                validation["normalized"], require_observation=True, refresh=True
            )
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "优先配置接管期间不能修改部署", 409)
            job = self._job_in_state(job_id, {"ready_to_start", "waiting_approval", "succeeded"}, "配置已被任务占用")
            if self.tasks.busy(job):
                raise TaskError("ROBOT_BUSY", "任务配置已冻结")
            self.tasks.check_idle()
            current = self.tasks.current_deployment_id()
            if current and job.id != current and not job.model_service.get("candidate_only"):
                raise TaskError("SCHEME_NOT_ACTIVE", "旧任务配置不可重新应用；请从模型准备页建立新配置", 409)
            rtc_route_changed = validation["normalized"]["rtc_inference_mode"] != job.config.get("rtc_inference_mode")
            if rtc_route_changed and validation["normalized"]["server_mode"] == "reuse":
                validation["normalized"]["server_mode"] = "real"
            if (job.id == current and validation["normalized"] != job.config) or rtc_route_changed:
                candidate = DeploymentJob(uuid.uuid4().hex, copy.deepcopy(validation["normalized"]),
                                          status="ready_to_start")
                candidate.model_service = copy.deepcopy(job.model_service)
                candidate.model_service["rtc_inference_mode"] = candidate.config["rtc_inference_mode"]
                candidate.model_service.update(candidate_only=True, owned_by_job=False, pgid=None,
                                               deployment_id=candidate.id, status="candidate",
                                               healthy=False, phase="candidate",
                                               phase_label="应用时加载模型")
                candidate.task_session = dict(status="ready", generation=0, remote_started=False,
                                              runtime_configured=True)
                candidate.presence_gate = self._initial_presence_gate_status(candidate.config)
                self._jobs[candidate.id] = candidate
                self._event(candidate, "ready_to_start", "运行参数已保存，等待应用到工作站", 86)
                return candidate.to_dict()
            job.config = validation["normalized"]
            job.task_session["runtime_configured"] = True
            # A post-load save must pass strict validation. A finished manual
            # session stays finished so its ready model can serve as a template.
            finished_manual_session = job.status == "succeeded"
            job.presence_gate = self._initial_presence_gate_status(job.config)
            job.approval_deadline = None
            if finished_manual_session:
                self._event(job, "succeeded", "运行配置已更新；模型仍就绪，可再次启动", 100)
            else:
                self._mark_ready_to_start(job)
            if not self._workstation.get("active_deployment_id") and self.tasks.model_available(job):
                self.tasks.cancel_reset_admission("CONFIG_CHANGED")
                self._workstation.update(active_deployment_id=job.id, accepting=False,
                                         reason="配置已应用；请核实后恢复新任务")
                self.tasks.persist()
            return job.to_dict()

    def priority_apply_deployment(
        self, job_id: str, payload: Mapping[str, Any], csrf_token: str | None
    ) -> dict[str, Any]:
        """Reserve an exclusive physical handoff before touching the old session.

        This is one confirmed UI operation.  Admission is durably closed before
        pause/stop/reset, and any ambiguous acknowledgement leaves it closed.
        """
        self._require_csrf(csrf_token)
        if not (self.enable_actions and self.enable_execute and self.enable_reset):
            raise TaskError("NOT_READY", "真机执行或机器人复位未启用", 503)
        if not isinstance(payload, Mapping):
            raise ValidationError("运行参数无效")
        with self._lock:
            source = self._job_in_state(
                job_id, {"ready_to_start", "waiting_approval", "succeeded"},
                "请先等待模型就绪，再应用真机配置",
            )
            if source.batch or self.tasks.busy(source):
                raise TaskError("ROBOT_BUSY", "此配置已被任务占用，请从可编辑的候选配置应用", 409)
            if self._workstation.get("persistence_error"):
                raise TaskError("PERSISTENCE_FAILED", "状态存储故障，必须人工核实", 503)
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "配置应用尚未结束", 409)
            current_id = self.tasks.current_deployment_id()
            current = self._jobs.get(current_id) if current_id else None
            if current and source.id != current_id and not source.model_service.get("candidate_only"):
                raise TaskError("SCHEME_NOT_ACTIVE", "历史配置不可重新应用", 409)
            source_config = copy.deepcopy(source.config)
            merged = self._saved_config_validation_payload(source.config)
            for key, value in payload.items():
                if key in {"robot_id", "robot_host", "robot_ssh_port", "model_id",
                           "model_version", "effector_profile", "server_mode"} and value != source.config[key]:
                    raise ValidationError(f"{key} 是模型加载前配置，当前不可修改")
                if key == "safety" and isinstance(value, Mapping):
                    merged["safety"].update(value)
                else:
                    merged[key] = value
        validation = self.validate(merged)
        if not validation["valid"]:
            raise ValidationError(json.dumps(validation["errors"], ensure_ascii=False))
        config = validation["normalized"]
        if (config["rtc_inference_mode"] != source_config.get("rtc_inference_mode")
                and config["server_mode"] == "reuse"):
            config["server_mode"] = "real"
        if config.get("hardware_mode") != "execute":
            raise TaskError("INVALID_REQUEST", "优先接管只适用于真机执行", 400)
        raw_robot = self.deployment.raw.get("robots", {}).get(str(config["robot_id"]), {})
        if raw_robot.get("reset_enabled") is not True or config.get("effector_profile") != "dahuan":
            raise TaskError("NOT_READY", "该机器人没有已配置的大寰安全复位路径", 503)
        # An operator's priority applies only to the task owned by this
        # control plane. It never takes over another GDK client/process.
        probe = self.probe_robot({"robot_id": config["robot_id"],
                                  "robot_host": config["robot_host"],
                                  "robot_ssh_port": config["robot_ssh_port"]}, self.csrf_token)
        external = probe.get("observation", {}).get("external_gdk_clients", ())
        if external:
            raise TaskError("GDK_IN_USE", "检测到平台外部 GDK 客户端；不能自动接管或终止这些进程", 409)
        if (not probe.get("reachable") or not probe.get("ssh_authenticated") or
                probe.get("identity", {}).get("match") is not True or
                probe.get("client", {}).get("valid") is not True or
                probe.get("client", {}).get("status") not in {"idle", "running"} or
                probe.get("observation", {}).get("ready") is not True):
            raise TaskError("ROBOT_STATE_UNKNOWN", "机器人身份、客户端或只读状态未确认，不能自动接管", 409)
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "配置应用尚未结束", 409)
            if self._workstation.get("persistence_error"):
                raise TaskError("PERSISTENCE_FAILED", "状态存储故障，必须人工核实", 503)
            source = self._job_in_state(
                job_id, {"ready_to_start", "waiting_approval", "succeeded"}, "配置状态已变化"
            )
            if source.config != source_config or source.batch or self.tasks.busy(source):
                raise TaskError("CONFIG_CHANGED", "原配置或任务状态已变化，请重新确认", 409)
            if self.tasks.current_deployment_id() != current_id:
                raise TaskError("CONFIG_CHANGED", "当前部署已变化，请重新确认", 409)
            current_id = self.tasks.current_deployment_id()
            current = self._jobs.get(current_id) if current_id else None
            if current:
                hardware = ("robot_id", "robot_host", "robot_ssh_port", "effector_profile")
                if any(current.config.get(key) != config.get(key) for key in hardware):
                    raise TaskError("HARDWARE_CONFLICT", "当前机器人或夹爪硬件不同；请走现场维护流程", 409)
                if (current.config.get("safety", {}).get("safety_profile_id")
                        != config.get("safety", {}).get("safety_profile_id")):
                    raise TaskError("SAFETY_PROFILE_CONFLICT",
                                    "旧任务与新配置的台面安全配置不同，不能自动复位接管", 409)
            occupied = [j for j in self._jobs.values() if self.tasks.busy(j)]
            if len(occupied) > 1 or any(j.config.get("robot_id") != config["robot_id"] for j in occupied):
                raise TaskError("ROBOT_BUSY", "多个任务或其他机器人状态未明确，不能自动接管", 409)
            if occupied and occupied[0].status not in {"running", "paused"}:
                raise TaskError("ROBOT_STATE_UNKNOWN", "旧任务状态未明确，不能自动暂停或复位", 409)
            if occupied and (occupied[0].config.get("safety", {}).get("safety_profile_id")
                             != config.get("safety", {}).get("safety_profile_id")):
                raise TaskError("SAFETY_PROFILE_CONFLICT",
                                "旧任务与新配置的台面安全配置不同，不能自动复位接管", 409)
            if occupied and (not occupied[0].task_session.get("remote_started") or
                             not occupied[0].task_session.get("accepted")):
                raise TaskError("ROBOT_STATE_UNKNOWN", "旧任务控制会话未确认，不能自动接管", 409)
            if occupied:
                gate = self._presence_task_gates.get(occupied[0].id)
                if gate is not None and gate.pending_action is not None:
                    raise TaskError("ROBOT_STATE_UNKNOWN", "旧任务正在进行包裹检测控制转换，不能自动接管", 409)
            if (probe["client"]["status"] == "running") != bool(occupied):
                raise TaskError("ROBOT_STATE_UNKNOWN", "平台任务记录与远端客户端占用状态不一致", 409)
            candidate = DeploymentJob(uuid.uuid4().hex, copy.deepcopy(config), status="ready_to_start")
            candidate.model_service = copy.deepcopy(source.model_service)
            candidate.model_service["rtc_inference_mode"] = config["rtc_inference_mode"]
            candidate.model_service.update(candidate_only=True, owned_by_job=False, pgid=None,
                                           deployment_id=candidate.id, status="candidate", healthy=False,
                                           phase="candidate", phase_label="等待安全接管")
            candidate.task_session = dict(status="ready", generation=0, remote_started=False,
                                          runtime_configured=True)
            candidate.presence_gate = self._initial_presence_gate_status(config)
            self._jobs[candidate.id] = candidate
            self._event(candidate, "ready_to_start", "真机配置已确认，正在安全接管旧任务", 86)
            prior_job_id = occupied[0].id if occupied else None
            self.tasks.cancel_reset_admission("CONFIG_CHANGED")
            self._workstation.update(
                accepting=False, reason="真机配置优先接管中；新任务已暂停",
                switching_deployment_id=candidate.id, switch_error="",
                priority_apply=dict(status="reserved", target_id=candidate.id,
                                    prior_job_id=prior_job_id, reset_job_id=None,
                                    started_at=time.time()),
            )
            self.tasks.persist()
            threading.Thread(
                target=self._priority_apply_worker,
                args=(candidate.id, current_id, prior_job_id),
                name="PriorityPhysicalApply", daemon=True,
            ).start()
            return self.tasks.state()

    def _priority_apply_worker(
        self, target_id: str, current_id: str | None, prior_job_id: str | None
    ) -> None:
        try:
            if prior_job_id:
                with self._lock:
                    prior = self._jobs[prior_job_id]
                    if self._workstation.get("switching_deployment_id") != target_id:
                        raise TaskError("SCHEME_SWITCHING", "优先接管的独占权已丢失", 409)
                    self._workstation["priority_apply"]["status"] = "pausing"
                    self.tasks.persist()
                # Also re-confirm an already-paused session. A persisted pause
                # alone is not a fresh safe-hold acknowledgement.
                self.pause_deployment(prior_job_id, self.csrf_token)
                with self._lock:
                    prior = self._jobs[prior_job_id]
                    if (prior.status != "paused" or not prior.task_session.get("remote_started") or
                            not prior.task_session.get("safe_hold_available")):
                        raise TaskError("ROBOT_STATE_UNKNOWN", "安全保持未确认，旧会话保持占用", 409)
                    self._workstation["priority_apply"]["status"] = "stopping"
                    self.tasks.persist()
                self.stop_deployment(prior_job_id, self.csrf_token, manual_completion=False)
                with self._lock:
                    prior = self._jobs[prior_job_id]
                    if prior.task_session.get("remote_started"):
                        raise TaskError("ROBOT_STATE_UNKNOWN", "旧会话结束未确认", 409)
                    prior.task_session["preempted_by"] = target_id
                    prior.events.append({"time": time.time(), "status": "preempted",
                                         "message": f"已由真机配置 {target_id} 接管"})
                    self.tasks.persist()
            reset_id = prior_job_id or current_id or target_id
            with self._lock:
                self._workstation["priority_apply"].update(status="resetting", reset_job_id=reset_id)
                self.tasks.persist()
            self.reset_robot_body(reset_id, self.csrf_token,
                                  priority_target=(reset_id == target_id),
                                  priority_reserved_id=target_id)
            deadline = time.monotonic() + 130.0
            while time.monotonic() < deadline:
                with self._lock:
                    reset_status = self._jobs[reset_id].robot_reset.get("status")
                if reset_status == "succeeded":
                    break
                if reset_status in {"failed", "unknown"}:
                    raise TaskError("RESET_REQUIRED", "自动复位未成功；禁止应用真机配置", 409)
                time.sleep(0.2)
            else:
                raise TaskError("ROBOT_STATE_UNKNOWN", "自动复位超时，结果需人工核实", 409)
            with self._lock:
                self._workstation["priority_apply"]["status"] = "activating"
                self.tasks.persist()
            self.tasks.activate(target_id, self.csrf_token, priority_reserved=True)
        except Exception as error:
            with self._lock:
                if self._workstation.get("switching_deployment_id") == target_id:
                    self._workstation.update(
                        switching_deployment_id=None, accepting=False,
                        reason="真机配置接管失败；请核实机器人、复位和模型状态",
                        switch_error=str(error)[:400],
                    )
                self._workstation["priority_apply"] = dict(
                    status="failed", target_id=target_id, error=str(error)[:400],
                    finished_at=time.time(),
                )
                try:
                    self.tasks.persist()
                except TaskError:
                    pass

    def start_deployment(
        self,
        job_id: str,
        csrf_token: str | None,
        *, sorting_authorized: bool = False,
        start_source: str = "web",
        tablet_record_name: str | None = None,
    ) -> dict[str, Any]:
        if start_source not in {"web", "tablet"}:
            raise ValidationError("任务启动来源无效")
        if start_source == "tablet":
            from nexus_deploy.control.tablet_start import RECORD_NAME

            if not isinstance(tablet_record_name, str) or not RECORD_NAME.fullmatch(tablet_record_name):
                raise ValidationError("平板录制名称无效")
        elif tablet_record_name is not None:
            raise ValidationError("网页启动不能指定平板录制")
        self._require_csrf(csrf_token)
        # Configuration is editable after model loading. Revalidate the frozen
        # snapshot at the actual start boundary so a stale browser or direct API
        # caller cannot bypass runtime and safety gates.
        with self._lock:
            self.tasks.check_entry("scheduler" if sorting_authorized else "operator")
            job = self._job_in_state(job_id, STARTABLE_JOB_STATES, "任务当前不可开始")
            if start_source == "tablet" and (
                job.config.get("human_in_loop") is not True
                or job.config.get("hardware_mode") != "execute"
                or job.batch
                or self.tasks.current_deployment_id() != job_id
            ):
                raise ValidationError("平板只能启动当前已获准的人工人在环路任务")
            if job.batch and (job.batch.get("status") != "starting" or
                              (job.batch.get("source", "scheduler") == "scheduler" and not sorting_authorized)):
                raise ValidationError("调度批次只能由调度接口启动一次", status=HTTPStatus.CONFLICT)
            submitted_config = copy.deepcopy(job.config)
        validation = self.validate(self._saved_config_validation_payload(submitted_config))
        if not validation["valid"]:
            raise ValidationError(json.dumps(validation["errors"], ensure_ascii=False))
        config = validation["normalized"]
        with self._lock:
            job = self._job_in_state(job_id, STARTABLE_JOB_STATES, "任务当前不可开始")
            if job.config != submitted_config:
                raise ValidationError("运行配置刚刚发生变化，请重新确认后再开始")
            physical_start = self.enable_actions and config.get("hardware_mode") == "execute"
            prior_reset_required = bool(self.tasks.reset_gate()["required"])
            self.tasks.admit_existing(
                job, "scheduler" if sorting_authorized else "operator",
                defer_reset_gate=physical_start,
            )
            job.task_session["start_source"] = "scheduler" if sorting_authorized else start_source
            if tablet_record_name is not None:
                job.task_session["tablet_record_name"] = tablet_record_name
            else:
                job.task_session.pop("tablet_record_name", None)
            job.config = copy.deepcopy(config)
            robot_id = str(config["robot_id"])
            owner = self._reserved_robots.get(robot_id)
            if owner is not None and owner != job.id:
                raise ValidationError(
                    f"机器人已有任务占用：{owner}", status=HTTPStatus.CONFLICT
                )
            self._reserved_robots[robot_id] = job.id
            job.task_session["generation"] = int(job.task_session.get("generation", 0)) + 1
            job.task_session.pop("completion_evidence", None)
            job.task_session.pop("completion_presence", None)
            generation = int(job.task_session["generation"])
            job.task_session["cancel_requested"] = False
            job.task_session["status"] = "starting"
            if physical_start:
                job.task_session["prestart_reset_required"] = prior_reset_required
                job.task_session["prestart_pose_status"] = "pending"
                self._mark_robot_reset_required(job, "pending", "真机任务已获准启动；下次任务前必须成功复位")
            self._clear_telemetry_for_new_generation(job)
            self._set_presence_gate_task_state(
                job,
                "starting",
                "任务正在启动；等待包裹检测服务连接头部相机帧",
            )
            self._event(job, "task_starting", "正在创建任务控制会话", 90)
            try:
                self.tasks.persist()
            except TaskError:
                job.status = "recovery_required"
                job.task_session["status"] = "recovery_required"
                raise
        try:
            if self.enable_actions:
                self._require_robot_probe(config, require_observation=True, refresh=True)
        except Exception:
            with self._lock:
                job = self._jobs[job_id]
                if job.task_session.get("cancel_requested"):
                    # An operator can end while the read-only probe is in
                    # flight.  No startup worker exists yet to consume that
                    # cancellation, so finish it here rather than restoring
                    # ready_to_start or leaving the robot slot occupied.
                    job.task_session["status"] = "cancelled"
                    if job.batch:
                        job.batch["status"] = "cancelled"
                    self._set_presence_gate_task_state(job, "cancelled", "启动前预检期间任务已由操作员结束")
                    self._event(job, "cancelled", "启动前预检期间任务已由操作员结束", 100)
                    self._release_robot(job)
                    self._release_model(job)
                    self.tasks.persist()
                else:
                    if self._reserved_robots.get(robot_id) == job_id:
                        self._reserved_robots.pop(robot_id, None)
                    job.task_session["status"] = "ready"
                    self._event(job, "ready_to_start", "机器人预检未通过；任务未启动", 86)
            raise
        threading.Thread(
            target=self._start_task_safely, args=(job_id, generation), daemon=True
        ).start()
        return self.get_job(job_id)

    def _start_task_safely(self, job_id: str, generation: int) -> None:
        robot = None
        remote_started = False
        sidecar_started = False
        try:
            with self._lock:
                job = self._jobs[job_id]
                config = copy.deepcopy(job.config)
                if job.task_session.get("cancel_requested"):
                    raise DeployError("Robot session startup was cancelled")
                self._inject_robot_telemetry_config_locked(job, config)
                if job.task_session.get("start_source") == "tablet":
                    config["_tablet_record_name"] = job.task_session["tablet_record_name"]
                presence_enabled = self._presence_enabled(job)
            if self.enable_actions and config.get("hardware_mode") == "execute":
                robot = self._robot_for_config(config)
                self._ensure_start_pose(job_id, generation, robot)
                if self._startup_cancelled(job_id, generation):
                    raise DeployError("Robot session startup was cancelled")
            if presence_enabled:
                config.update(self._start_presence_sidecar(job_id, generation))
                sidecar_started = True
            if self.enable_actions:
                robot = self._robot_for_config(config)
                model = self.deployment.models[str(config["model_id"])]
                with self._lock:
                    # A manual end can cancel startup at any point. Fence the
                    # final check, remote launch, and ownership record together
                    # so it cannot return before a new robot client is launched.
                    if self._startup_cancelled(job_id, generation):
                        raise DeployError("Robot session startup was cancelled")
                    pid = self.robot_port.start_session(robot, model, job_id, config)
                    remote_started = True
                    job = self._jobs[job_id]
                    job.task_session.update(
                        {
                            "status": "connecting",
                            "remote_started": True,
                            "remote_pid": pid,
                        }
                    )
                    job.telemetry.update(
                        {
                            "transport": "connecting",
                            "message": "机器人客户端已启动，正在等待 GDK、相机与模型握手",
                            "inference_status": "waiting_first_request",
                        }
                    )
                    self._event(job, "task_starting", "机器人客户端已启动，正在执行运行前握手", 92)
                snapshot = self.deployment.wait_robot_session_ready(
                    robot,
                    job_id,
                    cancel_check=lambda: self._startup_cancelled(job_id, generation),
                )
            else:
                time.sleep(0.04)
                snapshot = {}
            with self._lock:
                job = self._jobs[job_id]
                if self._startup_cancelled(job_id, generation):
                    raise DeployError("Robot session startup was cancelled")
                job.task_session["status"] = "running"
                job.task_session["remote_pid"] = snapshot.get("pid")
                job.task_session["robot_runtime_connected"] = bool(self.enable_actions)
                job.task_session["safe_hold_available"] = bool(self.enable_actions)
                job.task_session["soft_pause"] = True
                self._set_presence_gate_task_state(
                    job,
                    "running",
                    "任务已由操作员启动；包裹检测正在独立确认头部相机状态",
                )
                hardware = job.config["hardware_mode"]
                if self.enable_actions and hardware == "read-only":
                    description = "G2 只读客户端运行中；正在读取真实状态与相机，绝不下发动作"
                elif self.enable_actions and hardware == "execute":
                    description = "G2 真机任务运行中；安全层、看门狗与跟踪误差保护已启用"
                else:
                    description = "安全预览状态已推进；未启动机器人客户端，也未进行只读观测"
                self._event(job, "running", description, 96)
            if self.enable_actions:
                threading.Thread(
                    target=self._telemetry_loop, args=(job_id, generation), daemon=True
                ).start()
        except Exception as error:
            cleanup_error = None
            # A confirmed operator stop already released the client.  Check
            # shared ownership under the control lock before startup cleanup
            # so both paths cannot send stop_session concurrently or twice.
            with self._lock:
                prior_stop_unknown = any(
                    command.get("action") == "stop"
                    and command.get("generation") == generation
                    and command.get("status") == "unknown"
                    for command in self._jobs[job_id].task_session.get("robot_commands", [])
                )
                owned_remote = bool(
                    self.enable_actions and robot is not None and remote_started
                    and self._jobs[job_id].task_session.get("remote_started")
                    and not prior_stop_unknown
                )
                if owned_remote:
                    try:
                        self.robot_port.stop_session(robot, job_id)
                    except Exception as stop_error:  # unknown physical outcome
                        cleanup_error = str(stop_error)
                    else:
                        # Publish confirmed release before leaving this lock.
                        # An operator end arriving immediately afterwards must
                        # not send a duplicate stop_session command.
                        self._jobs[job_id].task_session.update(
                            remote_started=False, remote_pid=None,
                            robot_runtime_connected=False, safe_hold_available=False,
                        )
            if sidecar_started:
                self._stop_presence_sidecar(job_id, expected_generation=generation)
            with self._lock:
                job = self._jobs[job_id]
                cancelled = bool(job.task_session.get("cancel_requested"))
                job.error = None if cancelled else str(error)
                if cleanup_error:
                    job.error = (job.error + "；" if job.error else "") + f"后台客户端清理失败：{cleanup_error}"
                # start_session can launch the remote process and lose its
                # acknowledgement. RobotPort records that command as unknown;
                # remote_started is still false because no PID was received.
                # Keep robot ownership for manual reconciliation in that case.
                start_outcome_unknown = any(
                    command.get("action") == "start"
                    and command.get("generation") == generation
                    and command.get("status") == "unknown"
                    for command in job.task_session.get("robot_commands", [])
                )
                uncertain = bool(cleanup_error or job.task_session.get("remote_started")
                                 or start_outcome_unknown
                                 or job.task_session.get("prestart_pose_conflict")
                                 or job.robot_reset.get("status") in {"failed", "unknown"})
                job.task_session["status"] = "recovery_required" if uncertain else ("cancelled" if cancelled else "failed")
                if job.batch:
                    job.batch.update(status="recovery_required" if uncertain else
                                     ("cancelled" if cancelled else "failed"),
                                     reason=job.error)
                job.task_session["remote_started"] = bool(job.task_session.get("remote_started"))
                self._set_presence_gate_task_state(
                    job,
                    "recovery_required" if uncertain else ("cancelled" if cancelled else "failed"),
                    "任务未启动完成；包裹检测通道已结束",
                )
                job.telemetry.update(
                    {
                        "available": False,
                        "transport": "failed",
                        "message": job.error or "任务启动已取消；机器人客户端已停止",
                        "observation_status": "not_received",
                        "inference_status": "not_started",
                    }
                )
                self._event(
                    job,
                    "recovery_required" if uncertain else ("cancelled" if cancelled else "failed"),
                    "任务启动已取消" if cancelled else f"机器人任务启动失败：{job.error}",
                    job.progress,
                )
                if not uncertain:
                    self._release_robot(job)
                    self._release_model(job)

    def _ensure_start_pose(self, job_id: str, generation: int, robot: Any) -> None:
        """Check live 24D feedback, reset only when needed, and verify again."""

        if self._startup_cancelled(job_id, generation):
            raise DeployError("Robot session startup was cancelled")
        first = self.deployment.check_robot_reset_pose(robot)
        with self._lock:
            job = self._jobs[job_id]
            job.task_session["prestart_pose_status"] = first["status"]
            job.task_session["prestart_pose_check"] = first
            self.tasks.persist()
            prior_reset_required = bool(job.task_session.get("prestart_reset_required"))
        if first["status"] == "unknown":
            raise DeployError("机器人复位位姿读取结果不确定；真机任务未启动")
        if prior_reset_required or first["status"] == "not_home":
            if self._startup_cancelled(job_id, generation):
                raise DeployError("Robot session startup was cancelled")
            with self._lock:
                job = self._jobs[job_id]
                job.task_session["prestart_pose_status"] = "resetting"
                self.tasks.persist()
            self.reset_robot_body(job_id, self.csrf_token, prestart_for_task=True)
            deadline = time.monotonic() + 130.0
            while time.monotonic() < deadline:
                with self._lock:
                    reset_status = self._jobs[job_id].robot_reset.get("status")
                if reset_status == "succeeded":
                    break
                if reset_status in {"failed", "unknown", "cancelled"}:
                    raise DeployError("启动前自动复位失败；真机任务未启动")
                time.sleep(0.2)
            else:
                with self._lock:
                    job = self._jobs[job_id]
                    job.task_session["prestart_pose_conflict"] = True
                    self._mark_robot_reset_required(job, "unknown", "启动前自动复位超时；结果需人工核实")
                    self.tasks.persist()
                raise DeployError("启动前自动复位超时；结果需人工核实")
            if self._startup_cancelled(job_id, generation):
                raise DeployError("Robot session startup was cancelled")
            try:
                final = self.deployment.check_robot_reset_pose(robot)
            except Exception:
                with self._lock:
                    job = self._jobs[job_id]
                    job.task_session["prestart_pose_status"] = "unknown"
                    job.task_session["prestart_pose_conflict"] = True
                    self._mark_robot_reset_required(
                        job, "unknown", "复位后的实际位姿核验失败；禁止真机任务"
                    )
                    self.tasks.persist()
                raise
            with self._lock:
                job = self._jobs[job_id]
                job.task_session["prestart_pose_check_after_reset"] = final
                job.task_session["prestart_pose_status"] = final["status"]
                if final["status"] != "at_home":
                    job.task_session["prestart_pose_conflict"] = True
                    self._mark_robot_reset_required(job, "unknown", "复位后位姿未获确认；禁止真机任务")
                self.tasks.persist()
            if final["status"] != "at_home":
                raise DeployError("复位后关节位姿未获确认；真机任务未启动")
        with self._lock:
            if self._startup_cancelled(job_id, generation):
                raise DeployError("Robot session startup was cancelled")
            job = self._jobs[job_id]
            robot_id = str(job.config["robot_id"])
            if self._reserved_robots.get(robot_id) != job.id:
                raise DeployError("启动前机器人独占权丢失；真机任务未启动")
            job.task_session["prestart_pose_status"] = "verified"
            # A successful pre-start reset clears the *previous* task's gate.
            # Starting this task immediately creates a new next-task reset duty.
            self._mark_robot_reset_required(job, "pending", "真机任务已获准启动；下次任务前必须成功复位")
            self.tasks.persist()

    def _startup_cancelled(self, job_id: str, generation: int) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(
                job is None
                or job.task_session.get("cancel_requested")
                or int(job.task_session.get("generation", 0)) != generation
            )

    def _telemetry_loop(self, job_id: str, generation: int | None = None) -> None:
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if (
                    job is None
                    or job.status != "running"
                    or generation is not None
                    and int(job.task_session.get("generation", 0)) != generation
                ):
                    return
                config = copy.deepcopy(job.config)
            robot = self._robot_for_config(config)
            try:
                snapshot = self.deployment.robot_session_snapshot(robot, job_id)
                # Cache all three bounded JPEGs before publishing their frame id.
                self._cache_camera_frames(job_id, robot, snapshot)
                with self._lock:
                    job = self._jobs.get(job_id)
                    if (
                        job is None
                        or job.status != "running"
                        or generation is not None
                        and int(job.task_session.get("generation", 0)) != generation
                    ):
                        return
                    self._apply_robot_telemetry(job, snapshot)
                    control = (
                        snapshot.get("control", {})
                        if isinstance(snapshot.get("control"), Mapping)
                        else {}
                    )
                    if (
                        snapshot.get("running")
                        and control.get("state") == "paused"
                        and (
                            control.get("message") == "配置运行时长已到，客户端自动软暂停"
                            or control.get("pause_reason") == "collision_risk"
                        )
                    ):
                        job.task_session["status"] = "paused"
                        job.task_session["remote_started"] = True
                        job.task_session["robot_runtime_connected"] = True
                        job.task_session["safe_hold_available"] = True
                        collision_pause = control.get("pause_reason") == "collision_risk"
                        if collision_pause:
                            job.presence_gate["auto_paused"] = False
                            job.task_session["pause_reason"] = "collision_risk"
                            gate = self._presence_task_gates.get(job_id)
                            if gate is not None:
                                if gate.pending_action is not None:
                                    gate.acknowledge(gate.pending_action, applied=False)
                                gate.mark_operator_paused()
                        self._mark_telemetry_inactive(
                            job,
                            "paused",
                            (
                                "前瞻碰撞风险；机器人已安全保持，请检查工作区后手动继续或结束任务"
                                if collision_pause else
                                "配置运行时长已到；任务已自动软暂停，客户端和模型连接保持"
                            ),
                        )
                        self._event(
                            job,
                            "paused",
                            (
                                "前瞻碰撞保护已暂停任务；旧轨迹已丢弃，需人工确认后继续或结束"
                                if collision_pause else
                                "任务已按配置时长自动软暂停；可继续或结束任务"
                            ),
                            96,
                        )
                        return
                    if not snapshot.get("running"):
                        console = self.deployment.tail_robot_session(robot, job_id, lines=40)
                        job.task_session["remote_started"] = False
                        job.task_session["robot_runtime_connected"] = False
                        job.task_session["safe_hold_available"] = False
                        # A configured duration is handled above as an acknowledged
                        # soft pause. Reaching this branch means the remote process
                        # disappeared while the platform still expected it to run.
                        # The client's final lifecycle log is emitted on both clean
                        # and exceptional teardown, so it must not be treated as a
                        # successful duration completion marker.
                        job.error = console.strip()[-4000:] or "机器人客户端意外退出"
                        job.task_session["status"] = "failed"
                        self._event(job, "failed", "机器人客户端意外退出", job.progress)
                        self._release_model(job)
                        self._release_robot(job)
                        return
            except Exception as error:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None:
                        job.telemetry["message"] = f"遥测读取失败：{error}"
            time.sleep(0.05)

    def _archive_exited_robot_session(self, robot: Any, job_id: str) -> None:
        """Persist an already-exited session without changing job lifecycle."""

        try:
            archive = self.deployment.archive_robot_session(robot, job_id)
        except (DeployError, OSError, subprocess.SubprocessError) as error:
            archive = {"status": "failed", "session_id": job_id, "error": str(error)}
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.task_session["log_archive"] = copy.deepcopy(dict(archive))
            try:
                self._persist_state()
            except OSError:
                pass


    @staticmethod
    def _telemetry_vector(payload: Mapping[str, Any], field_name: str) -> list[float]:
        value = payload.get(field_name)
        if not isinstance(value, list) or len(value) != 24:
            raise ValidationError(f"{field_name} 必须是 24 维数组")
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{field_name} 包含非数值元素") from error
        if not all(math.isfinite(item) for item in result):
            raise ValidationError(f"{field_name} 包含非有限数值")
        return result

    @staticmethod
    def _initial_presence_gate_status(config: Mapping[str, Any]) -> dict[str, Any]:
        """Return UI state for a server-owned, human-authorised package gate.

        Vision only supplies a bounded observation.  It never has a route to
        create a task or issue a robot command; the control plane validates the
        observation and, if still safe, invokes an existing guarded operation.
        """

        configured = config.get("presence_gate", {})
        enabled = bool(
            isinstance(configured, Mapping) and configured.get("enabled") is True
        )
        safety = config.get("safety", {})
        awaiting_runtime_configuration = (
            enabled
            and config.get("hardware_mode") == "execute"
            and (
                not isinstance(safety, Mapping)
                or safety.get("workspace_safety_enabled") is not True
                or not isinstance(safety.get("safety_profile_id"), str)
                or not safety.get("safety_profile_id").strip()
            )
        )
        return {
            "enabled": enabled,
            "mode": "automatic" if enabled else "disabled",
            "transport": (
                "awaiting_runtime_config"
                if awaiting_runtime_configuration
                else "awaiting_task" if enabled else "disabled"
            ),
            "task_generation": None,
            "task_state": "not_started",
            "sequence": None,
            "received_at": None,
            "age_s": None,
            "observed_at_ns": None,
            "state": "UNKNOWN",
            "window_states": [],
            "source_fresh": False,
            "no_package_elapsed_s": None,
            "auto_paused": False,
            "awaiting_reset_ready": False,
            "pending_action": None,
            "proposed_action": "none",
            "last_action": "none",
            "action_status": "idle" if enabled else "disabled",
            "source_status": "not_started",
            "reason": (
                "等待保存真机执行与已验收工作空间安全配置"
                if awaiting_runtime_configuration
                else "尚未收到包裹检测结果"
            ),
            "message": (
                "已记录包裹检测选项。模型就绪后，请保存真机执行与已验收工作空间安全配置；在此之前视觉不会控制机器人。"
                if awaiting_runtime_configuration
                else "等待操作员开始只读任务后连接包裹检测服务；只读客户端不会发送机械臂命令"
                if enabled and config.get("hardware_mode") == "read-only"
                else "等待操作员开始任务后连接包裹检测服务"
                if enabled
                else "本任务未启用包裹检测"
            ),
            "last_error": None,
        }

    def _ensure_presence_gate_status(self, job: DeploymentJob) -> dict[str, Any]:
        """Migrate persisted jobs created before the presence-gate field."""

        initial = self._initial_presence_gate_status(job.config)
        if not isinstance(job.presence_gate, Mapping):
            job.presence_gate = initial
            return job.presence_gate
        merged = initial
        merged.update(copy.deepcopy(dict(job.presence_gate)))
        # The browser cannot alter this policy after the job is created, and
        # persisted state must not override the frozen deployment configuration.
        merged["enabled"] = initial["enabled"]
        merged["mode"] = "automatic" if initial["enabled"] else "disabled"
        if not initial["enabled"]:
            merged["transport"] = "disabled"
            merged["message"] = "本任务未启用包裹检测"
        job.presence_gate = merged
        return job.presence_gate

    def _set_presence_gate_task_state(
        self, job: DeploymentJob, task_state: str, message: str
    ) -> None:
        """Reflect task lifecycle changes without interpreting vision as control."""

        status = self._ensure_presence_gate_status(job)
        status["task_state"] = str(task_state)
        if not status.get("enabled"):
            return
        if task_state in {"ended", "stopped", "cancelled", "failed"}:
            status["transport"] = "ended"
        elif task_state in {"starting", "running", "paused", "pausing", "resuming"}:
            if status.get("received_at") is None:
                status["transport"] = "awaiting_observation"
        status["message"] = str(message)[:PRESENCE_GATE_MAX_REASON_LENGTH]

    def _health_loop(self):
        while not self._health_stop.wait(0.1):
            try:
                self.health_tick()
            except Exception:
                # Health failure cannot grant admission or cause an actuator action.
                with self._lock:
                    self.tasks.cancel_reset_admission("HEALTH_CHECK_FAILED")
                    self._workstation.update(accepting=False, reason="健康检查失败，需要核实")

    def close(self):
        if self._tablet_start_bridge is not None:
            self._tablet_start_bridge.close()
        self._health_stop.set()
        self._health_thread.join(timeout=1)

    def health_tick(self):
        with self._lock:
            for job in self._jobs.values():
                if job.id in self._presence_task_gates:
                    self._update_presence_gate_age(job)

    def _invalidate_presence(self, job, reason):
        if job.task_session.get("presence_expected_stop"):
            return
        gate = self._presence_task_gates.get(job.id)
        if gate is not None:
            gate.invalidate_evidence()
        status = self._ensure_presence_gate_status(job)
        if job.task_session.get("status") == "resetting":
            job.task_session["completion_conflict"] = reason
        status.update(source_fresh=False, state="UNKNOWN", window_states=[],
                      no_package_elapsed_s=None, reason=reason)

    def _update_presence_gate_age(self, job: DeploymentJob) -> None:
        """Expose source freshness; a stale or failed source never advances control."""

        status = self._ensure_presence_gate_status(job)
        if not status.get("enabled"):
            status["age_s"] = None
            return
        if job.task_session.get("presence_expected_stop"):
            status.update(transport="restarting", source_fresh=False, age_s=None)
            return
        sidecar = getattr(self, "_presence_sidecars", {}).get(job.id)
        if sidecar is not None and sidecar.process.poll() is not None:
            self._invalidate_presence(job, "vision process exited")
            status.update(
                {
                    "transport": "failed",
                    "source_fresh": False,
                    "action_status": "disabled",
                    "last_error": self._tail_text(sidecar.log_file)[:PRESENCE_GATE_MAX_REASON_LENGTH]
                    or "包裹检测服务已退出",
                    "message": "包裹检测服务已退出；自动门控已安全停用，不会发送任何机器人命令。",
                }
            )
            return
        received_at = status.get("received_at")
        if not isinstance(received_at, (int, float)):
            status["age_s"] = None
            return
        age_s = round(max(0.0, time.time() - float(received_at)), 1)
        status["age_s"] = age_s
        if status.get("task_state") in {"ended", "stopped", "cancelled", "failed"}:
            status["transport"] = "ended"
        elif age_s > 0.75:
            self._invalidate_presence(job, "vision observation gap")
            # A vision outage must never advance an empty-area timer or turn
            # into a pause/reset on its own.
            status["transport"] = "stale"
            status["message"] = "包裹检测数据已超过 0.75 秒未更新；无包裹计时已清除，不会发送任何机器人命令。"

    @staticmethod
    def _presence_enabled(job: DeploymentJob) -> bool:
        configured = job.config.get("presence_gate", {})
        return bool(isinstance(configured, Mapping) and configured.get("enabled") is True)

    @staticmethod
    def _gate_task_state(job: DeploymentJob) -> TaskState:
        """Map Nexus lifecycle state to the pure gate's deliberately small API."""

        if job.status == "running" and job.task_session.get("status") == "running":
            return TaskState.RUNNING
        if job.status in {"paused", "pausing", "task_starting"}:
            return TaskState.PAUSED
        return TaskState.ENDED

    def _new_presence_task_gate(self) -> PresenceTaskGate:
        configured = self.DEFAULTS["presence_gate"]
        gate = PresenceTaskGate(
            PresenceTaskGateConfig(
                enabled=True,
                pause_after_s=float(configured["pause_after_s"]),
                reset_after_s=float(configured["reset_after_s"]),
                max_state_age_s=0.75,
                auto_resume=bool(configured["auto_resume"]),
            )
        )
        return gate

    @staticmethod
    def _write_private_token(path: Path, token: str) -> None:
        """Create one 0600 token file without leaking it through argv or logs."""

        encoded = token.encode("utf-8")
        if not 24 <= len(encoded) <= 512:
            raise ValueError("presence token length is invalid")
        if not path.parent.is_dir():
            raise ValueError("presence token directory was not created")
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token)
                handle.write("\n")
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _tail_text(path: Path, *, limit: int = 2400) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return content[-limit:].strip()

    def _inject_robot_telemetry_config_locked(
        self, job: DeploymentJob, config: dict[str, Any]
    ) -> None:
        """Add a fresh opaque telemetry credential to a non-persisted launch copy."""

        if self.robot_telemetry_url is None:
            return
        telemetry_token = secrets.token_urlsafe(32)
        self._telemetry_tokens[job.id] = telemetry_token
        config["_platform_telemetry_endpoint"] = (
            f"{self.robot_telemetry_url}/api/deployments/{job.id}/robot-telemetry"
        )
        config["_platform_telemetry_token"] = telemetry_token

    @staticmethod
    def _clear_telemetry_for_new_generation(job: DeploymentJob) -> None:
        """Do not compare a restarted client's sequence zero with old telemetry."""

        control = job.telemetry.setdefault("control", {})
        control.update(
            {
                "available": False,
                "sequence": None,
                "received_at": None,
                "captured_at": None,
                "measured_state": None,
                "model_candidate": None,
                "controller_candidate": None,
                "adopted_action": None,
            }
        )
        job.telemetry.update(
            {
                "available": False,
                "transport": "restarting",
                "observation_status": "not_received",
                "inference_status": "restarting",
                "last_observation_at": None,
                "last_response_at": None,
                "last_frame_at": None,
                "response_age_s": None,
                "frame_age_s": None,
                "request_count": 0,
            }
        )
        for camera in job.telemetry.get("cameras", ()):
            camera.update({"status": "waiting", "image_url": None, "frame_id": None})

    def _start_presence_sidecar(self, job_id, generation, *, reuse_existing_gate=False):
        return self.presence_provider.start(job_id, generation, reuse_existing_gate=reuse_existing_gate)

    def _stop_presence_sidecar(self, job_id, *, expected_generation=None, clear_gate=True):
        return self.presence_provider.stop(job_id, expected_generation=expected_generation, clear_gate=clear_gate)

    @staticmethod
    def _parse_presence_observation(
        payload: Mapping[str, Any],
        *,
        require_three_cameras: bool = True,
    ) -> tuple[int, int, str, int, list[str], dict[str, list[str]], str]:
        """Validate the sidecar's bounded observation, never a command."""

        allowed = {"generation", "sequence", "status", "presence", "frame", "stage"}
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValidationError("包裹检测上报包含不支持的字段")
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValidationError("generation 必须是正整数")
        sequence = payload.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValidationError("sequence 必须是非负整数")
        source_status = payload.get("status", "ok")
        if not isinstance(source_status, str) or len(source_status) > 64:
            raise ValidationError("包裹检测状态字段无效")
        presence = payload.get("presence")
        if not isinstance(presence, Mapping):
            raise ValidationError("presence 必须是对象")
        state = presence.get("state")
        if not isinstance(state, str) or state not in PRESENCE_GATE_STATES:
            raise ValidationError("presence.state 无效")
        observed_at_ns = presence.get("observed_at_ns")
        if (
            isinstance(observed_at_ns, bool)
            or not isinstance(observed_at_ns, int)
            or observed_at_ns <= 0
        ):
            raise ValidationError("包裹检测结果缺少有效 observed_at_ns")
        window_states = presence.get("window_states", [])
        if (
            not isinstance(window_states, list)
            or len(window_states) > PRESENCE_GATE_MAX_WINDOW_STATES
            or any(
                not isinstance(item, str) or item not in PRESENCE_GATE_STATES
                for item in window_states
            )
        ):
            raise ValidationError("presence.window_states 无效")
        raw_cameras = presence.get("camera_windows", {})
        if not isinstance(raw_cameras, Mapping) or (
            raw_cameras and set(raw_cameras) != set(PRESENCE_CAMERA_NAMES)
        ):
            raise ValidationError("presence.camera_windows 必须包含三路相机")
        camera_windows: dict[str, list[str]] = {}
        for camera, samples in raw_cameras.items():
            if (
                not isinstance(samples, list)
                or len(samples) > PRESENCE_CAMERA_CONFIRM_FRAMES
                or any(not isinstance(item, str) or item not in PRESENCE_GATE_STATES for item in samples)
            ):
                raise ValidationError(f"presence.camera_windows.{camera} 无效")
            camera_windows[camera] = list(samples)
        if state in {"PACKAGE_PRESENT", "NO_PACKAGE"}:
            if not require_three_cameras and not camera_windows:
                if len(window_states) != PRESENCE_GATE_MAX_WINDOW_STATES or any(item != state for item in window_states):
                    raise ValidationError("模拟包裹状态必须来自原有四帧窗口")
            else:
                if len(window_states) != PRESENCE_CAMERA_CONFIRM_FRAMES or any(item != state for item in window_states):
                    raise ValidationError("确认的包裹状态必须来自三帧一致的窗口")
                confirmed = {
                    camera: len(samples) == PRESENCE_CAMERA_CONFIRM_FRAMES and all(item == state for item in samples)
                    for camera, samples in camera_windows.items()
                }
                if state == "NO_PACKAGE" and not all(confirmed.get(camera, False) for camera in PRESENCE_CAMERA_NAMES):
                    raise ValidationError("三路相机各三帧明确无包裹之前不得上报无包裹")
                if state == "PACKAGE_PRESENT" and not any(confirmed.values()):
                    raise ValidationError("有包裹状态缺少三帧相机证据")
        return generation, sequence, state, observed_at_ns, list(window_states), camera_windows, source_status

    @staticmethod
    def _apply_gate_decision_to_status(
        status: dict[str, Any],
        *,
        generation: int,
        sequence: int,
        state: str,
        observed_at_ns: int,
        window_states: list[str],
        camera_windows: dict[str, list[str]],
        source_status: str,
        decision: Any,
        task_state: TaskState,
    ) -> None:
        now = time.time()
        status.update(
            {
                "mode": "automatic",
                "transport": "online" if decision.source_fresh else "stale",
                "task_generation": generation,
                "task_state": task_state.value,
                "sequence": sequence,
                "received_at": now,
                "age_s": 0.0,
                "observed_at_ns": observed_at_ns,
                "state": state,
                "window_states": window_states,
                "camera_windows": camera_windows,
                "source_status": source_status,
                "source_fresh": bool(decision.source_fresh),
                "no_package_elapsed_s": decision.no_package_elapsed_s,
                "auto_paused": bool(decision.auto_paused),
                "awaiting_reset_ready": bool(decision.awaiting_reset_ready),
                "pending_action": (
                    decision.pending_action.value
                    if decision.pending_action is not None
                    else None
                ),
                "proposed_action": decision.action.value,
                "reason": str(decision.reason)[:PRESENCE_GATE_MAX_REASON_LENGTH],
                "message": (
                    "包裹检测正在等待控制服务完成安全动作"
                    if decision.pending_action is not None
                    else "包裹检测已收到三路相机判定；控制服务已完成本次状态评估"
                ),
                "last_error": None,
            }
        )

    def ingest_presence_gate(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        token: str | None,
    ) -> dict[str, Any]:
        """Accept a sidecar observation and independently schedule a safe proposal.

        This function deliberately receives no action name.  The visual process
        can communicate only a bounded three-camera presence result; the control
        plane decides whether it is still safe and meaningful to pause, reset,
        or resume the one human-authorised task.
        """

        action: GateAction | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            expected = self._presence_tokens.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            if expected is None or token is None or not secrets.compare_digest(token, expected):
                raise ValidationError("包裹检测令牌无效", status=HTTPStatus.FORBIDDEN)
            status = self._ensure_presence_gate_status(job)
            if not status.get("enabled"):
                raise ValidationError("本任务未启用包裹检测", status=HTTPStatus.CONFLICT)
            if job.status not in {"task_starting", "running", "paused", "pausing"}:
                raise ValidationError("部署任务当前不接收包裹检测结果", status=HTTPStatus.CONFLICT)
            if self._workstation.get("switching_deployment_id"):
                return {"accepted": False, "reason": "scheme_switching", "mode": "automatic"}
            generation, sequence, state, observed_at_ns, window_states, camera_windows, source_status = (
                self._parse_presence_observation(
                    payload, require_three_cameras=bool(getattr(self, "enable_actions", True))
                )
            )
            # An authenticated sidecar still does not get to turn a decode or
            # inference failure into a no-package decision.  Anything except
            # its normal success status is conservatively treated as UNKNOWN.
            if source_status != "ok":
                state = "UNKNOWN"
                window_states = []
                camera_windows = {}
            active_generation = int(job.task_session.get("generation", 0))
            if generation != active_generation:
                raise ValidationError("包裹检测结果属于过期任务代次", status=HTTPStatus.CONFLICT)
            self._update_presence_gate_age(job)
            status = self._ensure_presence_gate_status(job)
            previous_sequence = status.get("sequence")
            if isinstance(previous_sequence, int) and sequence <= previous_sequence:
                return {
                    "accepted": False,
                    "sequence": previous_sequence,
                    "reason": "stale",
                    "mode": "automatic",
                }
            now_ns = time.time_ns()
            if observed_at_ns > now_ns:
                raise ValidationError("包裹检测时间戳不能晚于控制服务时钟")
            gate = self._presence_task_gates.get(job_id)
            if gate is None:
                raise ValidationError(
                    "包裹检测会话在控制服务重启后需要由操作员重新开始任务",
                    status=HTTPStatus.CONFLICT,
                )
            if state != "NO_PACKAGE" and job.task_session.get("status") == "resetting":
                job.task_session["completion_conflict"] = "vision changed during reset"
            task_state = self._gate_task_state(job)
            decision = gate.observe(
                state,
                observed_at_ns=observed_at_ns,
                now_ns=now_ns,
                task_state=task_state,
            )
            self._apply_gate_decision_to_status(
                status,
                generation=generation,
                sequence=sequence,
                state=state,
                observed_at_ns=observed_at_ns,
                window_states=window_states,
                camera_windows=camera_windows,
                source_status=source_status,
                decision=decision,
                task_state=task_state,
            )
            job.updated_at = time.time()
            if decision.action is not GateAction.NONE:
                action = decision.action
                status["action_status"] = "queued"
                status["message"] = "包裹检测满足条件；正在由控制服务执行既有安全动作"
        if action is not None:
            threading.Thread(
                target=self._apply_presence_gate_action,
                args=(job_id, generation, action),
                name=f"PresenceGate-{action.value}-{job_id[:8]}",
                daemon=True,
            ).start()
        return {
            "accepted": True,
            "sequence": sequence,
            "mode": "automatic",
            "action": action.value if action is not None else "none",
        }

    def _gate_action_current_locked(
        self, job_id: str, generation: int, action: GateAction
    ) -> tuple[DeploymentJob, PresenceTaskGate] | None:
        job = self._jobs.get(job_id)
        gate = self._presence_task_gates.get(job_id)
        if (
            job is None
            or gate is None
            or job.task_session.get("cancel_requested")
            or int(job.task_session.get("generation", 0)) != generation
            or gate.pending_action is not action
        ):
            return None
        return job, gate

    def _mark_presence_action_failed(
        self,
        job_id: str,
        generation: int,
        action: GateAction,
        error: Exception,
    ) -> None:
        with self._lock:
            current = self._gate_action_current_locked(job_id, generation, action)
            if current is None:
                return
            job, gate = current
            gate.acknowledge(action, applied=False)
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "pending_action": None,
                    "proposed_action": "none",
                    "last_action": action.value,
                    "action_status": "failed",
                    "last_error": str(error)[:PRESENCE_GATE_MAX_REASON_LENGTH],
                    "message": "包裹检测自动动作未完成；不会重试或自动恢复机器人",
                }
            )
            # A transport exception cannot prove whether the command reached
            # the robot. Keep the execution slot occupied until inspected.
            job.error = str(error)
            job.task_session["status"] = "recovery_required"
            if job.batch:
                job.batch.update(status="recovery_required", reason=str(error))
            self._event(job, "recovery_required", "包裹检测动作结果未知；需要人工核实", 96)

    def _apply_presence_gate_action(
        self, job_id: str, generation: int, action: GateAction
    ) -> None:
        if not self.enable_actions:
            self.robot_port.apply_simulated_presence(job_id, generation, action)
            return
        try:
            if action is GateAction.PAUSE:
                self._apply_presence_pause(job_id, generation)
            elif action is GateAction.RESUME:
                self._apply_presence_resume(job_id, generation)
            elif action is GateAction.RESET:
                self._apply_presence_reset(job_id, generation)
        except Exception as error:  # each path remains fail-closed on any transport error
            if action is GateAction.RESET:
                self._mark_presence_reset_failed(job_id, generation, error)
            else:
                self._mark_presence_action_failed(job_id, generation, action, error)

    def _mark_presence_reset_failed(
        self, job_id: str, generation: int, error: Exception
    ) -> None:
        """Fail closed after a reset sequence has stopped the old client.

        The reset proposal is acknowledged before the physical reset begins, so
        this path intentionally does not require a still-pending gate action.
        A partially rebuilt client is stopped rather than left running without
        an authenticated sidecar/gate session.
        """

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.task_session.get("cancel_requested"):
                return
            current_generation = int(job.task_session.get("generation", 0))
            if current_generation < generation:
                return
            config = copy.deepcopy(job.config)
        try:
            self.robot_port.stop_session(self._robot_for_config(config), job_id)
        except Exception:
            pass
        self._stop_presence_sidecar(job_id, clear_gate=True)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.task_session.get("cancel_requested"):
                return
            job.error = str(error)
            job.task_session.update(
                {
                    "status": "failed",
                    "remote_started": False,
                    "remote_pid": None,
                    "robot_runtime_connected": False,
                    "safe_hold_available": False,
                }
            )
            job.robot_reset.update(
                {
                    "status": "failed",
                    "finished_at": time.time(),
                    "message": f"包裹检测自动复位失败：{error}",
                }
            )
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "transport": "failed",
                    "pending_action": None,
                    "proposed_action": "none",
                    "last_action": "reset",
                    "action_status": "failed",
                    "last_error": str(error)[:PRESENCE_GATE_MAX_REASON_LENGTH],
                    "message": "自动复位未完成；客户端已停止，不会自动恢复任务",
                }
            )
            job.task_session["status"] = "recovery_required"
            if job.batch:
                job.batch.update(status="recovery_required", reason=str(error))
            self._event(job, "recovery_required", "包裹检测自动复位结果需人工核实", job.progress)

    def _apply_presence_pause(self, job_id: str, generation: int) -> None:
        # Robot control.json has one command slot.  Serialize the gate pause
        # with operator pause/stop and gate resume through the reentrant lock.
        with self._lock:
            self._apply_presence_pause_serialized(job_id, generation)

    def _apply_presence_pause_serialized(self, job_id: str, generation: int) -> None:
        with self._lock:
            current = self._gate_action_current_locked(job_id, generation, GateAction.PAUSE)
            if current is None:
                return
            job, _gate = current
            if job.status != "running" or not job.task_session.get("remote_started"):
                raise DeployError("自动软暂停前任务不再处于可控制运行状态")
            config = copy.deepcopy(job.config)
            read_only = str(config.get("hardware_mode", "")) == "read-only"
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "action_status": "applying",
                    "message": (
                        "连续无包裹已确认；正在请求只读客户端软暂停，不会发送机械臂命令"
                        if read_only
                        else "连续无包裹已确认；正在请求客户端软暂停"
                    ),
                }
            )
            self._event(
                job,
                "pausing",
                (
                    "包裹检测确认连续无包裹；正在软暂停只读推理会话，不会发送机械臂命令"
                    if read_only
                    else "包裹检测确认连续无包裹；正在软暂停机器人任务"
                ),
                96,
            )
        self.robot_port.pause_session(self._robot_for_config(config), job_id, True)
        with self._lock:
            current = self._gate_action_current_locked(job_id, generation, GateAction.PAUSE)
            if current is None:
                return
            job, gate = current
            gate.acknowledge(GateAction.PAUSE, applied=True)
            job.task_session.update(
                {
                    "status": "paused",
                    "remote_started": True,
                    "robot_runtime_connected": True,
                    "safe_hold_available": True,
                }
            )
            self._mark_telemetry_inactive(
                job,
                "paused",
                (
                    "包裹检测已确认连续无包裹；只读客户端已暂停，不会发送机械臂命令"
                    if read_only
                    else "包裹检测已确认连续无包裹；客户端保持安全软暂停"
                ),
            )
            self._set_presence_gate_task_state(
                job,
                "paused",
                (
                    "连续四帧无包裹超过 1.5 秒；只读任务已由包裹检测软暂停，不会发送机械臂命令"
                    if read_only
                    else "连续四帧无包裹超过 1.5 秒；任务已由包裹检测软暂停"
                ),
            )
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "auto_paused": True,
                    "pending_action": None,
                    "proposed_action": "none",
                    "last_action": "pause",
                    "action_status": "applied",
                    "message": (
                        "只读任务已由包裹检测软暂停；持续无包裹 5 秒后进行只读复位验证，不会发送机械臂命令"
                        if read_only
                        else "任务已由包裹检测软暂停；持续无包裹 5 秒后才会申请复位"
                    ),
                }
            )
            self._event(
                job,
                "paused",
                (
                    "包裹检测只读软暂停已确认；机器人未接收动作命令"
                    if read_only
                    else "包裹检测软暂停已确认；机器人保持当前位置"
                ),
                96,
            )

    def _apply_presence_resume(self, job_id: str, generation: int) -> None:
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise DeployError("配置接管期间禁止包裹检测自动恢复旧任务")
            current = self._gate_action_current_locked(job_id, generation, GateAction.RESUME)
            if current is None:
                return
            job, _gate = current
            if job.status != "paused" or not job.task_session.get("remote_started"):
                raise DeployError("自动恢复前任务不再处于包裹检测软暂停状态")
            config = copy.deepcopy(job.config)
            read_only = str(config.get("hardware_mode", "")) == "read-only"
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "action_status": "applying",
                    "message": (
                        "四帧有包裹已确认；正在恢复同一只读推理任务，不会发送机械臂命令"
                        if read_only
                        else "四帧有包裹已确认；正在恢复同一任务"
                    ),
                }
            )
            # Serialize the remote unpause and its state transition with the
            # priority reservation. No new handoff can enter between them.
            self.robot_port.pause_session(self._robot_for_config(config), job_id, False)
            current = self._gate_action_current_locked(job_id, generation, GateAction.RESUME)
            if current is None:
                return
            job, gate = current
            gate.acknowledge(GateAction.RESUME, applied=True)
            job.task_session.update(
                {
                    "status": "running",
                    "remote_started": True,
                    "robot_runtime_connected": True,
                    "safe_hold_available": True,
                }
            )
            self._set_presence_gate_task_state(
                job,
                "running",
                (
                    "连续四帧检测到包裹；已恢复同一名操作员启动的只读任务，不会发送机械臂命令"
                    if read_only
                    else "连续四帧检测到包裹；已恢复同一名操作员启动的任务"
                ),
            )
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "auto_paused": False,
                    "pending_action": None,
                    "proposed_action": "none",
                    "last_action": "resume",
                    "action_status": "applied",
                    "message": (
                        "包裹已确认；同一只读任务已自动恢复，不会发送机械臂命令"
                        if read_only
                        else "包裹已确认；同一任务已自动恢复"
                    ),
                }
            )
            self._event(
                job,
                "running",
                (
                    "包裹检测确认有包裹；已恢复同一只读推理任务，不会发送机械臂命令"
                    if read_only
                    else "包裹检测确认有包裹；已恢复同一机器人任务"
                ),
                96,
            )
        threading.Thread(target=self._telemetry_loop, args=(job_id, generation), daemon=True).start()

    def _apply_presence_reset(self, job_id: str, generation: int) -> None:
        # Read-only is an end-to-end gate verification, not a physical reset.
        # The client has been launched with --read-only, so pause/resume still
        # exercise the existing client-control protocol while no GDK command
        # can be dispatched.  At the reset threshold we keep that same client
        # safely paused and advance only the gate's audited lifecycle state.
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                return
            current = self._gate_action_current_locked(job_id, generation, GateAction.RESET)
            if current is None:
                return
            job, _gate = current
            read_only = str(job.config.get("hardware_mode", "")) == "read-only"
        if read_only:
            self._apply_read_only_presence_reset(job_id, generation)
            return

        with self._lock:
            current = self._gate_action_current_locked(job_id, generation, GateAction.RESET)
            if current is None:
                return
            job, _gate = current
            if (
                job.status != "paused"
                or job.task_session.get("status") != "paused"
                or not job.task_session.get("remote_started")
            ):
                raise DeployError("自动复位前任务不再处于包裹检测软暂停状态")
            config = copy.deepcopy(job.config)
            status = self._ensure_presence_gate_status(job)
            job.task_session["completion_presence"] = {
                "generation": generation,
                "empty_seconds": status.get("no_package_elapsed_s"),
                "observed_state": status.get("state"),
                "vision_sequence": status.get("sequence"),
                "vision_received_at": status.get("received_at"),
            }
            status.update({"action_status": "applying", "message": "持续无包裹 5 秒；正在停止客户端并准备复位"})
            job.task_session["status"] = "resetting"
            # The deliberate camera shutdown begins with stop_session(), not
            # with the later sidecar stop. A health tick in between must not
            # turn the expected observation gap into a completion conflict.
            job.task_session["presence_expected_stop"] = True
            self._event(job, "task_starting", "包裹检测确认持续无包裹；正在执行既有机器人复位序列", 96)
        robot = self._robot_for_config(config)
        self.robot_port.stop_session(robot, job_id)
        with self._lock:
            current = self._gate_action_current_locked(job_id, generation, GateAction.RESET)
            if current is None:
                return
            job, gate = current
            if job.task_session.get("cancel_requested"):
                return
            gate.acknowledge(GateAction.RESET, applied=True)
            job.task_session.update(
                {
                    "remote_started": False,
                    "remote_pid": None,
                    "robot_runtime_connected": False,
                    "safe_hold_available": False,
                }
            )
            job.robot_reset.update(
                {
                    "status": "running",
                    "effectors_commanded": True,
                    "effector_sequence": ["open", "close", "open"],
                    "started_at": time.time(),
                    "finished_at": None,
                    "message": "包裹检测触发复位：正在复用已批准的大寰夹爪和身体复位序列",
                    "output_tail": "",
                }
            )
            status = self._ensure_presence_gate_status(job)
            # The old camera stream is deliberately stopped for the physical
            # reset. Its final verified observation is frozen above; the
            # ensuing gap must not invalidate that already accepted evidence.
            status.update(
                {
                    "auto_paused": True,
                    "awaiting_reset_ready": True,
                    "pending_action": None,
                    "proposed_action": "none",
                    "action_status": "resetting",
                    "transport": "restarting",
                }
            )
        self._stop_presence_sidecar(
            job_id,
            expected_generation=generation,
            clear_gate=False,
        )
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.task_session.get("cancel_requested"):
                return
        safety_profile_id = str(config.get("safety", {}).get("safety_profile_id", ""))
        if not safety_profile_id:
            raise DeployError("自动复位缺少已批准的桌面安全配置")
        # This is the same vetted operation behind the existing UI reset
        # button. Its implementation independently verifies idle client state,
        # GDK exclusivity, robot identity and the approved Dahuan reset script.
        self.deployment.deploy_generated_safety_profile(robot, safety_profile_id, job_id)
        reset_result = self.robot_port.reset_body(
            robot, safety_profile_id, job_id, timeout_s=70.0
        )
        if reset_result.get("completed") is not True:
            raise DeployError("机器人复位未返回已核验成功的结果")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.task_session.get("cancel_requested"):
                return
            job.task_session.update(
                {
                    "status": "resetting",
                    "remote_started": False,
                    "remote_pid": None,
                    "robot_runtime_connected": False,
                    "safe_hold_available": False,
                }
            )
            job.robot_reset.update(
                {
                    "status": "succeeded",
                    "finished_at": time.time(),
                    "message": "机器人复位完成；正在结束本次任务",
                    "output_tail": str(reset_result.get("output_tail", ""))[-2000:],
                }
            )
        self._finish_presence_task(job_id, simulated=False)

    def _finish_presence_task(self, job_id: str, *, simulated: bool) -> None:
        """Finish either entry path after the gate-owned reset succeeds."""
        with self._lock:
            job = self._jobs[job_id]
            if job.task_session.get("cancel_requested"):
                return
            expected = "read_only" if simulated else "succeeded"
            if job.robot_reset.get("status") != expected:
                raise DeployError("不能在复位成功前完成任务")
            if not simulated and (not self.enable_actions or job.config.get("hardware_mode") != "execute"):
                raise DeployError("非真机任务不能上报物理完成")
            if job.task_session.get("completion_conflict"):
                if job.batch:
                    job.batch.update(status="recovery_required", reason=job.task_session["completion_conflict"])
                self._event(job, "recovery_required", "复位期间证据失效，需要人工核实", 96)
                return
            gate_status = self._ensure_presence_gate_status(job)
            if not simulated:
                self._mark_robot_reset_succeeded(job)
            presence = job.task_session.get("completion_presence") or {}
            evidence = {
                "rule": "continuous_work_area_empty_then_reset_success",
                "profile_revision": job.task_session.get("profile_revision"),
                "generation": presence.get("generation", job.task_session.get("generation")),
                "commands": copy.deepcopy(job.task_session.get("robot_commands", [])),
                "empty_seconds": presence.get("empty_seconds", gate_status.get("no_package_elapsed_s")),
                "required_empty_seconds": float(job.config.get("presence_gate", {}).get(
                    "reset_after_s", self.DEFAULTS["presence_gate"]["reset_after_s"])),
                "observed_state": presence.get("observed_state", gate_status.get("state")),
                "vision_sequence": presence.get("vision_sequence", gate_status.get("sequence")),
                "vision_received_at": presence.get("vision_received_at", gate_status.get("received_at")),
                "reset_status": expected,
                "reset_finished_at": job.robot_reset.get("finished_at"),
                "physical_execution": not simulated,
                "conveyor_delivery_independently_verified": False,
            }
            if job.batch:
                job.batch.update(status="simulated" if simulated else "completed", completion_evidence=evidence)
            else:
                job.task_session["completion_evidence"] = evidence
            job.task_session.update(status="stopped", remote_started=False, remote_pid=None,
                                    presence_expected_stop=False,
                                    robot_runtime_connected=False, safe_hold_available=False)
            self._mark_telemetry_inactive(job, "stopped", "包裹检测任务已完成")
            gate_message = "只读流程验证完成；未执行真机复位" if simulated else "本次任务已完成；机器人已复位"
            self._set_presence_gate_task_state(job, "ended", gate_message)
            gate_status.update(action_status="applied", awaiting_reset_ready=False, pending_action=None,
                               proposed_action="none", last_action="reset", transport="stopped",
                               auto_paused=False, message=gate_message)
            self.tasks.persist()
            self._release_robot(job)
            self._release_model(job)
            message = ("只读包裹检测验证完成；未发送复位命令" if simulated else
                       "包裹检测确认完成，机器人已复位")
            self._event(job, "succeeded", message, 100)

    def _apply_read_only_presence_reset(self, job_id: str, generation: int) -> None:
        """End a read-only task at the reset threshold without robot motion."""

        with self._lock:
            current = self._gate_action_current_locked(job_id, generation, GateAction.RESET)
            if current is None:
                return
            job, gate = current
            if (
                str(job.config.get("hardware_mode", "")) != "read-only"
                or job.status != "paused"
                or job.task_session.get("status") != "paused"
                or not job.task_session.get("remote_started")
            ):
                raise DeployError("只读复位验证前任务不再处于包裹检测软暂停状态")

            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "action_status": "applying",
                    "message": "持续无包裹 5 秒；正在完成只读复位验证，不会调用机器人复位",
                }
            )
            self._event(
                job,
                "task_starting",
                "包裹检测确认持续无包裹；正在完成只读复位验证，不发送机器人命令",
                96,
            )

            gate.acknowledge(GateAction.RESET, applied=True)
            gate.mark_reset_complete()
            job.robot_reset.update(
                {
                    "status": "read_only",
                    "effectors_commanded": False,
                    "effector_sequence": [],
                    "started_at": time.time(),
                    "finished_at": time.time(),
                    "message": "只读观测验证已完成：未调用夹爪、关节或身体复位；正在结束客户端。",
                    "output_tail": "",
                }
            )
            if self.enable_actions and job.task_session.get("remote_started"):
                self.robot_port.stop_session(self._robot_for_config(job.config), job_id)
            self._finish_presence_task(job_id, simulated=True)

    def ingest_presence_gate_shadow(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        token: str | None,
    ) -> dict[str, Any]:
        """Accept a bounded display-only update from the local vision sidecar.

        The endpoint deliberately reuses the existing session-scoped telemetry
        token, which is not included in browser-visible job data.  Receiving a
        ``pause``/``reset``/``resume`` proposal here has no side effect: the
        value is merely displayed and audited as shadow evidence.
        """

        with self._lock:
            job = self._jobs.get(job_id)
            expected = self._telemetry_tokens.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            if expected is None or token is None or not secrets.compare_digest(token, expected):
                raise ValidationError("包裹检测遥测令牌无效", status=HTTPStatus.FORBIDDEN)
            status = self._ensure_presence_gate_status(job)
            if not status.get("enabled"):
                raise ValidationError("本任务未启用包裹检测", status=HTTPStatus.CONFLICT)
            if job.status not in {"task_starting", "running", "paused", "pausing"}:
                raise ValidationError("部署任务当前不接收包裹检测结果", status=HTTPStatus.CONFLICT)

            generation = payload.get("generation")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
                raise ValidationError("generation 必须是正整数")
            active_generation = int(job.task_session.get("generation", 0))
            if generation != active_generation:
                raise ValidationError("包裹检测结果属于过期任务代次", status=HTTPStatus.CONFLICT)

            sequence = payload.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise ValidationError("sequence 必须是非负整数")
            previous_sequence = status.get("sequence")
            if isinstance(previous_sequence, int) and sequence <= previous_sequence:
                return {
                    "accepted": False,
                    "sequence": previous_sequence,
                    "reason": "stale",
                    "mode": "shadow",
                }

            presence = payload.get("presence")
            if not isinstance(presence, Mapping):
                raise ValidationError("presence 必须是对象")
            state = presence.get("state")
            if not isinstance(state, str) or state not in PRESENCE_GATE_STATES:
                raise ValidationError("presence.state 无效")
            observed_at_ns = presence.get(
                "observed_at_ns",
                payload.get("output_timestamp_ns", payload.get("capture_timestamp_ns")),
            )
            if (
                isinstance(observed_at_ns, bool)
                or not isinstance(observed_at_ns, int)
                or observed_at_ns <= 0
            ):
                raise ValidationError("包裹检测结果缺少有效 observed_at_ns")
            window_states = presence.get("window_states", [])
            if (
                not isinstance(window_states, list)
                or len(window_states) > PRESENCE_GATE_MAX_WINDOW_STATES
                or any(
                    not isinstance(item, str) or item not in PRESENCE_GATE_STATES
                    for item in window_states
                )
            ):
                raise ValidationError("presence.window_states 无效")

            shadow = payload.get("presence_gate_shadow")
            decision = (
                shadow.get("decision") if isinstance(shadow, Mapping) else payload.get("decision")
            )
            if not isinstance(decision, Mapping):
                raise ValidationError("缺少 presence_gate_shadow.decision")
            action = decision.get("action")
            if not isinstance(action, str) or action not in PRESENCE_GATE_ACTIONS:
                raise ValidationError("包裹检测建议动作无效")
            pending_action = decision.get("pending_action")
            if pending_action is not None and (
                not isinstance(pending_action, str) or pending_action not in PRESENCE_GATE_ACTIONS
            ):
                raise ValidationError("包裹检测待确认动作无效")
            source_fresh = decision.get("source_fresh")
            if not isinstance(source_fresh, bool):
                raise ValidationError("包裹检测 source_fresh 必须是布尔值")
            no_package_elapsed_s = decision.get("no_package_elapsed_s")
            if no_package_elapsed_s is not None:
                try:
                    no_package_elapsed_s = float(no_package_elapsed_s)
                except (TypeError, ValueError) as error:
                    raise ValidationError("包裹检测 no_package_elapsed_s 必须是数值") from error
                if not math.isfinite(no_package_elapsed_s) or no_package_elapsed_s < 0:
                    raise ValidationError("包裹检测 no_package_elapsed_s 无效")
            decision_reason = decision.get("reason")
            if not isinstance(decision_reason, str) or not decision_reason.strip():
                raise ValidationError("包裹检测建议缺少原因")

            now = time.time()
            status.update(
                {
                    "transport": "online",
                    "task_generation": generation,
                    "task_state": str(job.task_session.get("status", "unknown")),
                    "sequence": sequence,
                    "received_at": now,
                    "age_s": 0.0,
                    "observed_at_ns": observed_at_ns,
                    "state": state,
                    "window_states": list(window_states),
                    "source_fresh": source_fresh,
                    "no_package_elapsed_s": no_package_elapsed_s,
                    "auto_paused": bool(decision.get("auto_paused")),
                    "awaiting_reset_ready": bool(decision.get("awaiting_reset_ready")),
                    "pending_action": pending_action,
                    "proposed_action": action,
                    "reason": decision_reason.strip()[:PRESENCE_GATE_MAX_REASON_LENGTH],
                    "message": "已收到包裹检测影子结果；未发送任何机器人命令",
                    "last_error": None,
                }
            )
            job.updated_at = now
            return {
                "accepted": True,
                "sequence": sequence,
                "mode": "shadow",
                "action": action,
            }

    def ingest_robot_telemetry(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        token: str | None,
    ) -> dict[str, Any]:
        """Accept one bounded latest-value snapshot from the robot publisher."""
        with self._lock:
            job = self._jobs.get(job_id)
            expected = self._telemetry_tokens.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            if expected is None or token is None or not secrets.compare_digest(token, expected):
                raise ValidationError("机器人遥测令牌无效", status=HTTPStatus.FORBIDDEN)
            if job.status not in {"task_starting", "running", "paused", "pausing"}:
                raise ValidationError("部署任务当前不接收机器人遥测", status=HTTPStatus.CONFLICT)

            sequence = payload.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise ValidationError("sequence 必须是非负整数")
            current = job.telemetry.setdefault("control", {})
            previous = current.get("sequence")
            if isinstance(previous, int) and sequence <= previous:
                return {"accepted": False, "sequence": previous, "reason": "stale"}

            safety_raw = payload.get("safety")
            if not isinstance(safety_raw, Mapping):
                raise ValidationError("safety 必须是对象")
            safety: dict[str, Any] = {
                "enabled": bool(safety_raw.get("enabled")),
                "monitoring_enabled": bool(safety_raw.get("monitoring_enabled")),
                "enforcement_enabled": bool(safety_raw.get("enforcement_enabled")),
                "decision_available": bool(safety_raw.get("decision_available")),
                "collision_detected": bool(safety_raw.get("collision_detected")),
                "proximity_warning": bool(safety_raw.get("proximity_warning")),
                "intervention_active": bool(safety_raw.get("intervention_active")),
                # Physical-pose alarm and rejected candidate are deliberately
                # distinct.  A tele-operated recovery must clear the former
                # even while an old model candidate remains held/rejected.
                "candidate_collision_detected": bool(
                    safety_raw.get("candidate_collision_detected")
                ),
                "candidate_proximity_warning": bool(
                    safety_raw.get("candidate_proximity_warning")
                ),
                "guard_evaluated": safety_raw.get("guard_evaluated") is not False,
            }
            for field_name in (
                "reason", "response", "limiting_surface", "limiting_point",
                "worker_mode", "geometry_mode", "candidate_reason",
            ):
                value = safety_raw.get(field_name)
                safety[field_name] = None if value is None else str(value)[:160]
            for field_name in (
                "action_scale",
                "current_z_min",
                "distance_to_surface",
                "candidate_distance_to_surface",
                "surface_height",
                "hard_margin",
                "warning_margin",
                "predicted_z_min",
                "safe_predicted_z_min",
                "guard_latency_ms",
                "guard_budget_ms",
                "check_fps",
                "certificate_age_ms",
            ):
                value = safety_raw.get(field_name)
                if value is None:
                    safety[field_name] = None
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise ValidationError(f"safety.{field_name} 必须是数值") from error
                if not math.isfinite(numeric):
                    raise ValidationError(f"safety.{field_name} 必须是有限数值")
                safety[field_name] = numeric
            for section_name, numeric_fields, string_fields in (
                (
                    "work_surface",
                    (
                        "distance_m", "clearance_m", "hard_margin_m",
                        "warning_margin_m",
                    ),
                    ("name", "limiting_point"),
                ),
                (
                    "inter_arm",
                    (
                        "minimum_separation_m", "clearance_m", "hard_margin_m",
                        "warning_margin_m",
                    ),
                    ("left_geometry", "right_geometry"),
                ),
                (
                    "candidate_inter_arm",
                    (
                        "minimum_separation_m", "clearance_m", "hard_margin_m",
                        "warning_margin_m", "checked_waypoints",
                    ),
                    ("left_geometry", "right_geometry"),
                ),
            ):
                raw_section = safety_raw.get(section_name)
                if raw_section is None:
                    safety[section_name] = None
                    continue
                if not isinstance(raw_section, Mapping):
                    raise ValidationError(f"safety.{section_name} 必须是对象")
                section: dict[str, Any] = {
                    "collision": bool(raw_section.get("collision")),
                    "warning": bool(raw_section.get("warning")),
                }
                for field_name in numeric_fields:
                    value = raw_section.get(field_name)
                    if value is None:
                        section[field_name] = None
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as error:
                        raise ValidationError(
                            f"safety.{section_name}.{field_name} 必须是数值"
                        ) from error
                    if not math.isfinite(numeric):
                        raise ValidationError(
                            f"safety.{section_name}.{field_name} 必须是有限数值"
                        )
                    section[field_name] = numeric
                for field_name in string_fields:
                    value = raw_section.get(field_name)
                    section[field_name] = None if value is None else str(value)[:160]
                safety[section_name] = section
            misses = safety_raw.get("deadline_misses", 0)
            if isinstance(misses, bool) or not isinstance(misses, int) or misses < 0:
                raise ValidationError("safety.deadline_misses 必须是非负整数")
            safety["deadline_misses"] = misses

            captured_at = payload.get("captured_at")
            if captured_at is not None:
                try:
                    captured_at = float(captured_at)
                except (TypeError, ValueError) as error:
                    raise ValidationError("captured_at 必须是时间戳") from error
                if not math.isfinite(captured_at):
                    raise ValidationError("captured_at 必须是有限时间戳")
            now = time.time()
            measured_state = self._telemetry_vector(payload, "measured_state")
            current.update(
                {
                    "available": True,
                    "sequence": sequence,
                    "received_at": now,
                    "captured_at": captured_at,
                    "hardware_mode": str(payload.get("hardware_mode") or "")[:32],
                    "measured_state": measured_state,
                    "model_candidate": self._telemetry_vector(payload, "model_candidate"),
                    "controller_candidate": self._telemetry_vector(
                        payload, "controller_candidate"
                    ),
                    "adopted_action": self._telemetry_vector(payload, "adopted_action"),
                    "safety": safety,
                }
            )
            if (
                self.simulator_control_url is not None
                and str(job.config.get("robot_id")) == "g2a"
            ):
                self._queue_simulator_pose(job_id, sequence, measured_state)
            return {"accepted": True, "sequence": sequence}

    @staticmethod
    def _apply_robot_telemetry(job: DeploymentJob, snapshot: Mapping[str, Any]) -> bool:
        timing = snapshot.get("timing", {}) if isinstance(snapshot.get("timing"), Mapping) else {}
        trace = snapshot.get("trace", {}) if isinstance(snapshot.get("trace"), Mapping) else {}
        frame = snapshot.get("frame", {}) if isinstance(snapshot.get("frame"), Mapping) else {}
        run_metadata = (
            snapshot.get("run_metadata", {})
            if isinstance(snapshot.get("run_metadata"), Mapping)
            else {}
        )
        server_timing = (
            timing.get("server_timing", {})
            if isinstance(timing.get("server_timing"), Mapping)
            else {}
        )
        rtc = trace.get("rtc", {}) if isinstance(trace.get("rtc"), Mapping) else {}
        anchor = trace.get("anchor", {}) if isinstance(trace.get("anchor"), Mapping) else {}
        image_cameras = {str(item) for item in snapshot.get("image_cameras", ())}
        request_id = timing.get("request_id") or trace.get("request_id")
        frame_id = frame.get("request_id") or (request_id if image_cameras else None)
        previous_request_id = job.telemetry.get("request", {}).get("request_id")
        is_new_request = bool(request_id and request_id != previous_request_id)
        previous_frame_ids = {
            str(camera.get("frame_id"))
            for camera in job.telemetry.get("cameras", ())
            if camera.get("frame_id")
        }
        is_new_frame = bool(frame_id and image_cameras and str(frame_id) not in previous_frame_ids)
        now = time.time()
        observation_status = (
            "synthetic"
            if request_id and job.config.get("hardware_mode") == "dry-run"
            else "received"
            if request_id
            else "waiting"
        )
        job.telemetry.update(
            {
                "available": bool(request_id),
                "transport": "online" if snapshot.get("running") else "stopped",
                "observation_status": observation_status,
                "inference_status": "running" if request_id else "waiting_first_request",
                "last_observation_at": (
                    float(snapshot.get("trace_updated_at") or now) if request_id else None
                ),
                "last_response_at": (
                    float(snapshot.get("timing_updated_at") or now) if request_id else None
                ),
                "last_frame_at": (
                    float(
                        snapshot.get("frame_updated_at")
                        or snapshot.get("trace_updated_at")
                        or now
                    )
                    if request_id and image_cameras
                    else job.telemetry.get("last_frame_at")
                ),
                "request_count": int(job.telemetry.get("request_count", 0))
                + (1 if is_new_request else 0),
                "message": (
                    "已收到机器人客户端的实际请求遥测"
                    if request_id
                    else "机器人客户端在线，等待第一条推理请求"
                ),
                "request": {
                    "request_id": request_id,
                    "session_id": run_metadata.get("session_id"),
                    "rtt_ms": timing.get("round_trip_ms") or trace.get("round_trip_ms"),
                    "inference_ms": server_timing.get("infer_ms"),
                    "queue_depth": anchor.get("remaining_actions"),
                    "chunk_id": trace.get("chunk_id"),
                    "rtc_blend_steps": rtc.get("actual_blend_steps"),
                },
            }
        )
        for camera in job.telemetry.get("cameras", []):
            camera_id = str(camera.get("id"))
            available = camera_id in image_cameras and bool(frame_id)
            camera["status"] = "online" if available else "waiting"
            camera["image_url"] = (
                f"/api/deployments/{job.id}/cameras/{camera_id}"
                f"?frame={frame_id}"
                if available
                else None
            )
            camera["frame_id"] = str(frame_id) if available else camera.get("frame_id")
            camera["last_frame_at"] = (
                float(
                    snapshot.get("frame_updated_at")
                    or snapshot.get("trace_updated_at")
                    or now
                )
                if available
                else camera.get("last_frame_at")
            )
        return is_new_frame

    def _cache_camera_frames(
        self, job_id: str, robot: Any, snapshot: Mapping[str, Any]
    ) -> None:
        timing = snapshot.get("timing", {}) if isinstance(snapshot.get("timing"), Mapping) else {}
        trace = snapshot.get("trace", {}) if isinstance(snapshot.get("trace"), Mapping) else {}
        frame = snapshot.get("frame", {}) if isinstance(snapshot.get("frame"), Mapping) else {}
        frame_id = str(
            frame.get("request_id")
            or timing.get("request_id")
            or trace.get("request_id")
            or ""
        )
        for camera_name in snapshot.get("image_cameras", ()):
            if camera_name not in {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
                continue
            try:
                mime, data = self.deployment.read_robot_camera(
                    robot,
                    job_id,
                    str(camera_name),
                    frame_id=frame_id,
                )
            except (DeployError, OSError, subprocess.SubprocessError):
                continue
            with self._lock:
                self._camera_cache[(job_id, str(camera_name))] = (mime, data, frame_id)

    @staticmethod
    def _mark_telemetry_inactive(job: DeploymentJob, status: str, message: str) -> None:
        job.telemetry["available"] = False
        job.telemetry["transport"] = status
        job.telemetry["inference_status"] = status
        job.telemetry["message"] = message
        for camera in job.telemetry.get("cameras", ()):
            if camera.get("frame_id"):
                camera["status"] = "last_frame"
            else:
                camera["status"] = "waiting"

    @staticmethod
    def _update_telemetry_ages(job: DeploymentJob) -> None:
        now = time.time()
        response_at = job.telemetry.get("last_response_at")
        frame_at = job.telemetry.get("last_frame_at")
        job.telemetry["response_age_s"] = (
            round(max(0.0, now - float(response_at)), 1) if response_at else None
        )
        job.telemetry["frame_age_s"] = (
            round(max(0.0, now - float(frame_at)), 1) if frame_at else None
        )
        stale_after = max(3.0, float(job.config.get("watchdog_timeout_s", 3.0)) * 2)
        if (
            job.status == "running"
            and response_at
            and now - float(response_at) > stale_after
        ):
            job.telemetry["available"] = False
            job.telemetry["transport"] = "stalled"
            job.telemetry["inference_status"] = "stalled"
            job.telemetry["message"] = (
                f"推理遥测已超过 {stale_after:g} 秒未更新；任务状态需要检查"
            )
            for camera in job.telemetry.get("cameras", ()):
                if camera.get("frame_id"):
                    camera["status"] = "last_frame"

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as error:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND) from error
            # Migrate jobs persisted by the former approval-gate workflow.
            if job.status == "waiting_approval":
                job.approval_deadline = None
                self._mark_ready_to_start(job)
            if (
                job.status in TERMINAL_JOB_STATES
                and job.task_session.get("status") in {"stopped", "cancelled", "failed"}
            ):
                # Repair state persisted by older control-plane versions that
                # left the remote_started flag set after the process exited.
                job.task_session["remote_started"] = False
                job.task_session["robot_runtime_connected"] = False
                job.task_session["safe_hold_available"] = False
            self._update_telemetry_ages(job)
            self._update_presence_gate_age(job)
            return dict(job.to_dict(), task_view=self.tasks.view(job_id), workstation=self.tasks.state())

    def pause_deployment(self, job_id: str, csrf_token: str | None) -> dict[str, Any]:
        """Soft-pause inference while retaining the robot runtime connection."""

        # A gate command in flight finishes first.  The operator's pause is
        # then the last command ACKed by the robot's one-slot control channel.
        with self._lock:
            return self._pause_deployment_serialized(job_id, csrf_token)

    def _pause_deployment_serialized(self, job_id: str, csrf_token: str | None) -> dict[str, Any]:

        self._require_csrf(csrf_token)
        with self._lock:
            job = self._job_in_state(job_id, {"running", "paused"}, "部署任务当前不可暂停")
            gate = self._presence_task_gates.get(job_id)
            if job.task_session.get("status") == "resetting" or job.robot_reset.get("status") == "running":
                raise TaskError("ROBOT_BUSY", "机器人复位正在执行；不能并发软暂停", 409)
            if gate is not None and gate.pending_action is not None:
                # RESET can run outside this lock.  Revoke only its still
                # queued proposal; never claim to interrupt reset motion.
                gate_status = self._ensure_presence_gate_status(job)
                if gate.pending_action is GateAction.RESET and gate_status.get("action_status") != "queued":
                    raise TaskError("ROBOT_BUSY", "机器人复位正在执行；不能并发软暂停", 409)
                cancelled = gate.pending_action
                gate.acknowledge(cancelled, applied=False)
                gate_status.update(pending_action=None, proposed_action="none",
                                   action_status="superseded_by_operator",
                                   message="人工暂停已接管；自动动作提案已撤销")
            config = copy.deepcopy(job.config)
            remote_started = bool(job.task_session.get("remote_started"))
            hardware = str(job.config.get("hardware_mode"))
            self._event(job, "pausing", "正在清空动作计划并等待机器人保持确认", 96)
        try:
            if self.enable_actions and remote_started:
                self.robot_port.pause_session(
                    self._robot_for_config(config), job_id, True
                )
        except Exception:
            with self._lock:
                job = self._jobs[job_id]
                job.task_session["status"] = "recovery_required"
                self._event(job, "recovery_required", "暂停结果未知；客户端状态需要人工确认", 96)
            raise
        with self._lock:
            job = self._jobs[job_id]
            job.task_session["status"] = "paused"
            job.task_session["pause_reason"] = "operator"
            if gate is not None:
                gate.mark_operator_paused()
            job.task_session["remote_started"] = remote_started
            job.task_session["robot_runtime_connected"] = remote_started
            job.task_session["safe_hold_available"] = bool(self.enable_actions and remote_started)
            self._set_presence_gate_task_state(
                job,
                "paused",
                "任务已由操作员暂停；包裹检测不会对人工暂停申请自动复位",
            )
            self._mark_telemetry_inactive(
                job, "paused", "任务已暂停；画面为暂停前最后一帧"
            )
            description = (
                "真机软暂停已确认；动作计划已清空，机器人持续保持当前位置"
                if hardware == "execute"
                else "只读软暂停已确认；推理已停止，GDK 与模型连接保持"
            )
            self._event(job, "paused", description, 96)
            return job.to_dict()

    def resume_deployment(self, job_id: str, csrf_token: str | None) -> dict[str, Any]:
        self._require_csrf(csrf_token)
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "配置优先接管期间禁止恢复旧任务", 409)
            job = self._job_in_state(job_id, {"paused"}, "部署任务当前不可继续")
            config = copy.deepcopy(job.config)
            robot_id = str(config["robot_id"])
            owner = self._reserved_robots.get(robot_id)
            if owner is not None and owner != job.id:
                raise ValidationError(f"机器人已有任务占用：{owner}", status=HTTPStatus.CONFLICT)
            gate = self._presence_task_gates.get(job_id)
            if gate is not None and gate.pending_action is not None:
                raise ValidationError("包裹检测正在执行安全动作；请等待结果", status=HTTPStatus.CONFLICT)
            if gate is not None and gate.auto_paused:
                raise ValidationError(
                    "任务由包裹检测自动暂停；请等待连续四帧确认有包裹后自动恢复，或结束任务",
                    status=HTTPStatus.CONFLICT,
                )
            generation = int(job.task_session.get("generation", 0))
            job.task_session["cancel_requested"] = False
            remote_started = bool(job.task_session.get("remote_started"))
            job.task_session["status"] = "resuming"
            self._event(job, "task_starting", "正在读取当前姿态并等待新的推理计划", 90)
            # Keep the control lock through the acknowledged remote resume.
            # Otherwise priority apply could reserve the robot between the
            # state check and this command, then mistake the old hold as stable.
            try:
                if self.enable_actions:
                    if not remote_started:
                        raise DeployError("软暂停客户端已退出，不能原地继续；请结束任务后重新创建")
                    snapshot = self.deployment.robot_session_snapshot(
                        self._robot_for_config(config), job_id
                    )
                    if not snapshot.get("running"):
                        raise DeployError("软暂停客户端已退出，不能原地继续")
                    self.robot_port.pause_session(
                        self._robot_for_config(config), job_id, False
                    )
            except Exception as error:
                job.task_session["status"] = "paused"
                job.error = str(error)
                self._event(job, "paused", "软继续未通过；任务保持暂停", 96)
                raise
            gate = self._presence_task_gates.get(job_id)
            if gate is not None:
                gate.mark_externally_resumed()
            job.error = None
            job.task_session["status"] = "running"
            self._set_presence_gate_task_state(
                job,
                "running",
                "任务已由操作员恢复；包裹检测继续独立确认头部相机状态",
            )
            self._event(job, "running", "当前姿态已重新锚定；新的推理计划已安装", 96)
        if self.enable_actions:
            threading.Thread(
                target=self._telemetry_loop, args=(job_id, generation), daemon=True
            ).start()
        return self.get_job(job_id)

    def stop_deployment(
        self, job_id: str, csrf_token: str | None, *, manual_completion: bool = True
    ) -> dict[str, Any]:
        # Serialize a confirmed stop after any in-flight gate PAUSE/RESUME.
        # Robot control.json has one command slot and cannot accept both.
        with self._lock:
            return self._stop_deployment_serialized(
                job_id, csrf_token, manual_completion=manual_completion
            )

    def _stop_deployment_serialized(
        self, job_id: str, csrf_token: str | None, *, manual_completion: bool = True
    ) -> dict[str, Any]:
        self._require_csrf(csrf_token)
        with self._lock:
            job = self._job_in_state(job_id, STOPPABLE_JOB_STATES | {"recovery_required"}, "部署任务当前不可结束")
            if (job.status == "recovery_required"
                    and any(
                        command.get("action") in ({"stop"} if job.task_session.get("remote_started") else {"start"})
                        and command.get("generation") == job.task_session.get("generation")
                        and command.get("status") == "unknown"
                        for command in job.task_session.get("robot_commands", [])
                    )):
                # A lost start ACK is not proof that no client was launched;
                # a lost stop ACK is not permission to send another stop.
                # Independent state evidence is required before releasing
                # the reservation or issuing another physical command.
                raise TaskError(
                    "ROBOT_STATE_UNKNOWN",
                    "机器人客户端启动结果未知；必须先独立核实机器人空闲状态",
                    409,
                )
            if (job.status == "task_starting" and
                    (job.task_session.get("prestart_pose_status") == "resetting" or
                     job.task_session.get("prestart_reset_dispatching"))):
                # The start-pose reset may still be preparing its safety
                # profile, or may already have dispatched physical motion.
                # Its worker consumes cancellation before dispatch if possible
                # and waits for the final result if motion is in flight.
                job.task_session["cancel_requested"] = True
                self._event(job, "cancelling", "人工结束已请求；等待启动前复位安全结束", job.progress)
                return job.to_dict()
            if job.task_session.get("status") == "resetting" or job.robot_reset.get("status") == "running":
                raise TaskError("ROBOT_BUSY", "机器人复位正在执行；结束任务需等待复位结果", 409)
            gate = self._presence_task_gates.get(job_id)
            if gate is not None and gate.pending_action is not None:
                gate_status = self._ensure_presence_gate_status(job)
                if gate.pending_action is GateAction.RESET and gate_status.get("action_status") != "queued":
                    raise TaskError("ROBOT_BUSY", "机器人复位正在执行；结束任务需等待复位结果", 409)
                cancelled = gate.pending_action
                gate.acknowledge(cancelled, applied=False)
                gate_status.update(pending_action=None, proposed_action="none",
                                   action_status="superseded_by_operator",
                                   message="人工结束已接管；自动动作提案已撤销")
            was_starting = job.status == "task_starting"
            config = copy.deepcopy(job.config)
            remote_started = bool(job.task_session.get("remote_started"))
            operator_completed = bool(
                manual_completion
                and job.status in {"running", "paused"}
                and remote_started
                and job.task_session.get("accepted")
                and job.batch.get("source") == "scheduler"
                and self.enable_actions
                and config.get("hardware_mode") == "execute"
            )
            job.task_session["cancel_requested"] = True
            if job.batch and self.enable_actions and config.get("hardware_mode") == "execute":
                # Stop is not reset. Keep physical scheduler work blocked until
                # the separately supervised reset has a durable success record.
                job.batch["reset_required"] = True
            if self.enable_actions and config.get("hardware_mode") == "execute":
                self._mark_robot_reset_required(job, "pending", "真机任务已停止；下次任务前必须成功复位")
            self._event(
                job,
                "cancelling" if was_starting else "stopping",
                "正在取消任务启动" if was_starting else "正在结束机器人会话",
                job.progress,
            )
        # A not-yet-launched client is cancelled by the startup worker.  Once
        # launched, the operator waits for stop_session ACK before returning.
        if was_starting and self.enable_actions and not remote_started:
            return self.get_job(job_id)
        archive = None
        if self.enable_actions and remote_started:
            try:
                value = self.robot_port.stop_session(self._robot_for_config(config), job_id)
                archive = dict(value) if isinstance(value, Mapping) else None
            except Exception:
                with self._lock:
                    job.task_session["status"] = "recovery_required"
                    if job.batch:
                        job.batch.update(status="recovery_required", reason="停止结果未知，必须人工核实")
                    self._event(job, "recovery_required", "停止结果未知，必须人工核实", job.progress)
                raise
        with self._lock:
            job = self._jobs[job_id]
            if archive is not None:
                job.task_session["log_archive"] = archive
            job.task_session["status"] = "cancelled" if was_starting else "stopped"
            job.task_session.update(
                remote_started=False,
                remote_pid=None,
                robot_runtime_connected=False,
                safe_hold_available=False,
                presence_expected_stop=False,
            )
            if job.batch:
                if operator_completed:
                    # The operator ended the admitted physical session. This is
                    # a business completion, not evidence of an empty work area
                    # or of a successful robot reset.
                    job.batch.update(
                        status="completed",
                        completion_rule="operator_end",
                        completion_evidence={
                            "rule": "operator_end",
                            "physical_execution": True,
                            "reset_status": "pending",
                            "conveyor_delivery_independently_verified": False,
                        },
                    )
                else:
                    job.batch["status"] = "cancelled"
            self._set_presence_gate_task_state(
                job,
                "cancelled" if was_starting else "stopped",
                "任务已由操作员结束；包裹检测通道已关闭",
            )
            self._mark_telemetry_inactive(
                job,
                "cancelled" if was_starting else "ended",
                "任务启动已取消" if was_starting else "任务已结束；画面为结束前最后一帧",
            )
            self._event(
                job,
                "cancelled" if was_starting else "succeeded",
                "任务启动已由操作员取消" if was_starting else "会话已由操作员结束",
                100,
            )
            self._release_robot(job)
            self._release_model(job)
            return job.to_dict()

    def reconcile_stale_robot_session(
        self, job_id: str, csrf_token: str | None
    ) -> dict[str, Any]:
        """Release only a proven-dead robot client from a recovery task.

        A failed pause has an unknown physical outcome.  This operation sends
        no robot command, does not clear the recovery state or reset gate, and
        requires a fresh independent robot probe before correcting stale local
        ownership.  The normal supervised reset remains mandatory.
        """

        self._require_csrf(csrf_token)
        if not self.enable_actions or not self.enable_reset:
            raise ValidationError("控制服务未开放机器人恢复", status=HTTPStatus.FORBIDDEN)
        with self._lock:
            job = self._job_in_state(
                job_id, {"recovery_required"}, "只有需要人工核实的任务可协调失效会话"
            )
            if self.tasks.current_deployment_id() != job.id:
                raise TaskError("SCHEME_NOT_ACTIVE", "只能协调当前方案的机器人会话", 409)
            if job.config.get("hardware_mode") != "execute" or not job.task_session.get("accepted"):
                raise ValidationError("该任务不是已启动的真机执行任务", status=HTTPStatus.CONFLICT)
            if not job.task_session.get("remote_started"):
                raise ValidationError("机器人客户端未标记为运行中", status=HTTPStatus.CONFLICT)
            if not self._workstation.get("reset_gate", {}).get("required"):
                raise ValidationError("复位门禁状态不一致", status=HTTPStatus.CONFLICT)
            try:
                recorded_pid = int(job.task_session["remote_pid"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValidationError("缺少可核对的旧客户端 PID", status=HTTPStatus.CONFLICT) from error
            if recorded_pid <= 1:
                raise ValidationError("旧客户端 PID 无效", status=HTTPStatus.CONFLICT)
            robot_id = str(job.config["robot_id"])
            if self._reserved_robots.get(robot_id) not in (None, job.id):
                raise ValidationError("机器人占用者已变化", status=HTTPStatus.CONFLICT)
            self.tasks.check_idle(exclude=job.id, require_reset=False)
            config = copy.deepcopy(job.config)
            generation = job.task_session.get("generation")

        robot = self._robot_for_config(config)
        probe = self.probe_robot(
            {
                "robot_id": robot_id,
                "robot_host": config["robot_host"],
                "robot_ssh_port": config["robot_ssh_port"],
            },
            self.csrf_token,
        )
        client = probe.get("client")
        observation = probe.get("observation")
        if (
            probe.get("reachable") is not True
            or probe.get("ssh_authenticated") is not True
            or probe.get("identity", {}).get("match") is not True
            or not isinstance(client, Mapping)
            or client.get("valid") is not True
            or client.get("status") != "idle"
            or client.get("running") is not False
            or client.get("processes") != []
            or probe.get("runtime_session") != "idle"
            or not isinstance(observation, Mapping)
            or observation.get("services_ready") is not True
            or observation.get("external_gdk_clients") != []
        ):
            raise ValidationError(
                "机器人身份、客户端空闲状态或 GDK 独占状态未通过核对",
                status=HTTPStatus.CONFLICT,
            )
        snapshot = self.deployment.robot_session_snapshot(robot, job_id)
        if snapshot.get("running") is not False or snapshot.get("pid") not in (0, recorded_pid):
            raise ValidationError("旧会话状态仍不明确", status=HTTPStatus.CONFLICT)
        try:
            # The exact recorded PID must be absent, even if its pid file was
            # already removed.  A reused PID is treated as ambiguous.
            self.deployment.run_ssh(
                f"test ! -e /proc/{recorded_pid}", robot=robot, capture=True
            )
        except (DeployError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            raise ValidationError("旧客户端 PID 仍存在或无法核对", status=HTTPStatus.CONFLICT) from error

        with self._lock:
            job = self._jobs[job_id]
            if (
                job.status != "recovery_required"
                or job.task_session.get("remote_started") is not True
                or job.task_session.get("remote_pid") != recorded_pid
                or job.task_session.get("generation") != generation
                or self.tasks.current_deployment_id() != job.id
                or self._reserved_robots.get(robot_id) not in (None, job.id)
            ):
                raise ValidationError("任务或机器人占用状态已变化，请重新核对", status=HTTPStatus.CONFLICT)
            self.tasks.check_idle(exclude=job.id, require_reset=False)
            job.task_session.update(
                remote_started=False,
                remote_pid=None,
                robot_runtime_connected=False,
                safe_hold_available=False,
                cancel_requested=True,
                status="recovery_required",
                stale_client_reconciled_at=time.time(),
            )
            if self._reserved_robots.get(robot_id) == job.id:
                self._reserved_robots.pop(robot_id, None)
            self._mark_robot_reset_required(
                job, "pending", "旧机器人客户端已核实退出；仍需成功复位"
            )
            self._event(
                job, "recovery_required",
                "已核实旧机器人客户端退出；任务仍需人工复位",
                job.progress,
            )
            return self.get_job(job_id)

    def reset_robot_body(
        self, job_id: str, csrf_token: str | None, *, command_id: str | None = None,
        priority_target: bool = False,
        priority_reserved_id: str | None = None,
        prestart_for_task: bool = False,
    ) -> dict[str, Any]:
        """Start an asynchronous Dahuan gripper cycle followed by the body reset."""

        self._require_csrf(csrf_token)
        if not self.enable_actions or not self.enable_reset:
            raise ValidationError("控制服务未开放机器人复位", status=HTTPStatus.FORBIDDEN)
        reservation = f"reset:{job_id}"
        with self._lock:
            switching = self._workstation.get("switching_deployment_id")
            if switching and not (
                priority_reserved_id == switching
                and self._workstation.get("priority_apply", {}).get("target_id") == switching
                and self._workstation.get("priority_apply", {}).get("reset_job_id") == job_id
                and self._workstation.get("priority_apply", {}).get("status") == "resetting"
            ):
                raise TaskError("SCHEME_SWITCHING", "配置接管期间禁止单独复位机器人", 409)
            allowed_states = RESETTABLE_JOB_STATES | ({"task_starting"} if prestart_for_task else set())
            job = self._job_in_state(job_id, allowed_states, "请先结束当前任务，再执行机器人复位")
            prestart_owned = bool(
                prestart_for_task
                and job.status == "task_starting"
                and job.task_session.get("accepted")
                and job.task_session.get("prestart_pose_status") == "resetting"
                and not job.task_session.get("cancel_requested")
                and not job.task_session.get("remote_started")
                and job.config.get("hardware_mode") == "execute"
            )
            if prestart_for_task and not prestart_owned:
                raise TaskError("ROBOT_STATE_UNKNOWN", "启动前复位的任务状态已变化", 409)
            target_reserved = (
                priority_target
                and self.tasks.current_deployment_id() is None
                and self._workstation.get("switching_deployment_id") == job.id
                and job.model_service.get("candidate_only") is True
            )
            if (not target_reserved and self.tasks.current_deployment_id()
                    != job.model_service.get("deployment_id", job.id)):
                raise TaskError("SCHEME_NOT_ACTIVE", "只能复位当前方案对应的机器人", 409)
            config = copy.deepcopy(job.config)
            robot_id = str(config.get("robot_id", ""))
            if (self._reserved_robots.get(robot_id) == reservation
                    and job.robot_reset.get("status") == "failed"):
                # The previous reset process ended, but its conservative
                # in-memory reservation remains. A fresh probe below must
                # prove the robot is idle before retrying.
                self._reserved_robots.pop(robot_id, None)
            self.tasks.check_idle(exclude=job.id, require_reset=False)
            raw_robot = self.deployment.raw.get("robots", {}).get(robot_id, {})
            if raw_robot.get("reset_enabled") is not True:
                raise ValidationError("该机器人未配置复位能力", status=HTTPStatus.CONFLICT)
            if str(config.get("effector_profile", "")) != "dahuan":
                raise ValidationError(
                    "当前复位序列仅适用于大寰夹爪",
                    status=HTTPStatus.CONFLICT,
                )
            reset_status = str(job.robot_reset.get("status", "idle"))
            if reset_status in {"preparing", "running"}:
                raise ValidationError("机器人复位已经在进行中", status=HTTPStatus.CONFLICT)
            if job.task_session.get("remote_started"):
                raise ValidationError(
                    "机器人客户端仍在运行，请先结束任务",
                    status=HTTPStatus.CONFLICT,
                )
            owner = self._reserved_robots.get(robot_id)
            # An automatic reset keeps this task's reservation on failure.
            # Transfer only that stopped recovery task to the supervised reset;
            # the fresh identity/idle/GDK probe below still controls admission.
            retrying_own_failed_reset = (
                owner == job.id
                and job.status == "recovery_required"
                and job.task_session.get("status") == "recovery_required"
                and reset_status == "failed"
                and job.task_session.get("remote_started") is False
                and job.task_session.get("remote_pid") is None
            )
            if owner is not None and not retrying_own_failed_reset and not (prestart_owned and owner == job.id):
                raise ValidationError(f"机器人已有任务占用：{owner}", status=HTTPStatus.CONFLICT)
            self._reserved_robots[robot_id] = reservation
            self._mark_robot_reset_required(job, "resetting", "机器人复位正在执行；成功前禁止下一任务")
            now = time.time()
            job.robot_reset = {
                "status": "preparing",
                "command_id": command_id,
                "effectors_commanded": True,
                "effector_sequence": ["open", "close", "open"],
                "started_at": now,
                "finished_at": None,
                "message": "正在重新核验机器人身份、客户端空闲状态和 GDK 独占权",
                "output_tail": "",
            }
            job.updated_at = now
            job.events.append(
                {
                    "time": now,
                    "status": "robot_reset_preparing",
                    "message": "机器人复位已确认；大寰夹爪将先松开、夹紧并最终松开",
                }
            )
            self._persist_state()
        try:
            probe = self.probe_robot(
                {
                    "robot_id": robot_id,
                    "robot_host": config["robot_host"],
                    "robot_ssh_port": config["robot_ssh_port"],
                },
                self.csrf_token,
            )
            client = probe.get("client", {})
            if (
                not probe.get("reachable")
                or not probe.get("ssh_authenticated")
                or probe.get("identity", {}).get("match") is False
                or not isinstance(client, Mapping)
                or client.get("status") != "idle"
                or not client.get("valid")
            ):
                raise ValidationError(
                    "机器人身份、客户端完整性或空闲状态未通过复位预检",
                    status=HTTPStatus.CONFLICT,
                )
            external = probe.get("observation", {}).get("external_gdk_clients", ())
            if external:
                raise ValidationError(
                    "检测到其他 GDK 客户端；复位要求独占机器人",
                    status=HTTPStatus.CONFLICT,
                )
        except Exception as error:
            with self._lock:
                if self._reserved_robots.get(robot_id) == reservation:
                    self._reserved_robots.pop(robot_id, None)
                job = self._jobs[job_id]
                job.robot_reset.update(
                    {
                        "status": "failed",
                        "finished_at": time.time(),
                        "message": str(error),
                    }
                )
                job.updated_at = time.time()
                self._mark_robot_reset_required(job, "failed", "机器人复位预检失败；必须成功复位")
                self._persist_state()
            raise
        with self._lock:
            job = self._jobs[job_id]
            if prestart_owned and job.task_session.get("cancel_requested"):
                # The manual end won while the read-only probe was running.
                if self._reserved_robots.get(robot_id) == reservation:
                    self._reserved_robots[robot_id] = job_id
                job.robot_reset.update(
                    status="cancelled", effectors_commanded=False,
                    finished_at=time.time(), message="Startup cancelled before reset motion",
                )
                self._mark_robot_reset_required(
                    job, "pending", "启动已由操作员取消；复位未执行"
                )
                self._persist_state()
                raise DeployError("Robot session startup was cancelled")
            job.robot_reset.update(
                {
                    "status": "running",
                    "message": "正在执行大寰夹爪松开-夹紧-松开，并复位身体关节",
                }
            )
            job.updated_at = time.time()
            self._persist_state()
        threading.Thread(
            target=self._run_robot_body_reset,
            args=(job_id, reservation, config),
            kwargs={"restore_task_owner": prestart_owned},
            daemon=True,
        ).start()
        return self.get_job(job_id)

    def _run_robot_body_reset(
        self, job_id: str, reservation: str, config: Mapping[str, Any],
        *, restore_task_owner: bool = False,
    ) -> None:
        robot_id = str(config.get("robot_id", ""))
        cancelled_before_motion = False
        try:
            robot = self._robot_for_config(config)
            safety_profile_id = str(
                config.get("safety", {}).get("safety_profile_id", "")
            )
            if not safety_profile_id:
                raise ValidationError("机器人复位缺少已标定的桌面安全配置")
            self.deployment.deploy_generated_safety_profile(
                robot, safety_profile_id, job_id
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if restore_task_owner and (
                    job is None or job.task_session.get("cancel_requested")
                ):
                    cancelled_before_motion = True
                    raise DeployError("Robot session startup was cancelled before reset motion")
                if restore_task_owner:
                    # From this point an in-flight reset must finish safely;
                    # manual end stays pending until the worker observes it.
                    job.task_session["prestart_reset_dispatching"] = True
                    self.tasks.persist()
            result = self.robot_port.reset_body(
                robot, safety_profile_id, job_id, timeout_s=70.0
            )
            if result.get("completed") is not True:
                raise DeployError("机器人复位未返回已核验成功的结果")
        except Exception as error:
            status = "cancelled" if cancelled_before_motion else "failed"
            message = ("启动已由操作员取消；复位未执行" if cancelled_before_motion
                       else f"机器人复位失败：{error}")
            output_tail = ""
        else:
            status = "succeeded"
            message = "机器人复位完成；身体关节已到位，大寰夹爪最终保持松开"
            output_tail = str(result.get("output_tail", ""))
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                now = time.time()
                job.robot_reset.update(
                    {
                        "status": status,
                        # Once the remote reset starts, a failure may occur
                        # after any gripper phase; report conservatively.
                        "effectors_commanded": not cancelled_before_motion,
                        "effector_sequence": [] if cancelled_before_motion else ["open", "close", "open"],
                        "finished_at": now,
                        "message": message,
                        "output_tail": output_tail,
                    }
                )
                job.updated_at = now
                if job.batch.get("completion_rule") == "operator_end":
                    evidence = job.batch.get("completion_evidence") or {}
                    evidence["reset_status"] = status
                    evidence["reset_finished_at"] = now
                    job.batch["completion_evidence"] = evidence
                job.events.append(
                    {
                        "time": now,
                        "status": f"robot_reset_{status}",
                        "message": message,
                    }
                )
            if status in {"succeeded", "cancelled"} and self._reserved_robots.get(robot_id) == reservation:
                if restore_task_owner:
                    self._reserved_robots[robot_id] = job_id
                else:
                    self._reserved_robots.pop(robot_id, None)
            elif status == "failed" and job is not None:
                job.status = "recovery_required"
                job.task_session["status"] = "recovery_required"
                if job.batch.get("completion_rule") == "operator_end":
                    job.batch.update(status="recovery_required", reason=message)
            if job is not None:
                job.task_session.pop("prestart_reset_dispatching", None)
                agent_operations = []
                # Keep each Agent command's result after a later UI/startup
                # reset replaces the per-job robot_reset snapshot.
                for owner in self._jobs.values():
                    operation = owner.task_session.get("agent_reset_commands", {}).get(
                        job.robot_reset.get("command_id")
                    )
                    if operation is not None and operation.get("target_job_id") == job_id:
                        operation.update(status=status if status != "cancelled" else "failed",
                                         finished_at=now, error=None if status == "succeeded" else message)
                        agent_operations.append(operation)
                if status == "succeeded":
                    self._mark_robot_reset_succeeded(job)
                    if job.status == "recovery_required" and not job.task_session.get("accepted"):
                        job.status = "ready_to_start"
                        job.task_session["status"] = "ready"
                    elif job.status == "recovery_required":
                        job.status = "failed"
                        job.task_session["status"] = "stopped"
                        if job.batch.get("status") == "recovery_required":
                            job.batch["status"] = "failed"
                elif status == "failed":
                    self._mark_robot_reset_required(job, "failed", "机器人复位失败；必须排障后重试")
                else:
                    self._mark_robot_reset_required(job, "pending", "启动已由操作员取消；复位未执行")
                for operation in agent_operations:
                    self.tasks.finish_agent_reset_admission(job, operation)
            try:
                self.tasks.persist()
            except TaskError:
                # The persistence wrapper has already closed admission. Keep
                # the command response consistent with that outcome as well.
                if job is not None:
                    for operation in agent_operations:
                        if operation.get("admission_restored"):
                            operation.update(admission_restored=False,
                                             admission_restore_reason="PERSISTENCE_FAILED")
                logger.exception("Could not persist robot reset completion; admission remains closed")

    def read_camera(self, job_id: str, camera_name: str) -> tuple[str, bytes]:
        with self._lock:
            try:
                self._jobs[job_id]
            except KeyError as error:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND) from error
            cached = self._camera_cache.get((job_id, camera_name))
            if cached is not None:
                return cached[0], cached[1]
        raise ValidationError("相机遥测暂不可用", status=HTTPStatus.NOT_FOUND)

    def close_model(self, job_id: str, csrf_token: str | None) -> dict[str, Any]:
        """Stop the selected managed model, independent of which job started it."""

        self._require_csrf(csrf_token)
        with self._lock:
            if self._workstation.get("switching_deployment_id"):
                raise TaskError("SCHEME_SWITCHING", "优先配置接管期间不能关闭模型", 409)
            job = self._job_in_state(
                job_id,
                {"ready_to_start", "succeeded", "failed", "rejected"},
                "运行或暂停任务时不能关闭模型；请先结束任务",
            )
            current = self._jobs.get(self.tasks.current_deployment_id())
            if job.model_service.get("candidate_only") or (current is not None and
                    current.id != job.id and current.config.get("model_id") == job.config.get("model_id")):
                raise TaskError("SCHEME_NOT_ACTIVE", "候选或历史方案不能关闭当前模型服务", 409)
            model = self.deployment.models[str(job.config["model_id"])]
            active_consumers = [
                candidate.id
                for candidate in self._jobs.values()
                if candidate.id != job.id
                and str(candidate.config.get("model_id")) == model.name
                and candidate.status in {"task_starting", "running", "paused", "stopping"}
            ]
            if active_consumers:
                raise ValidationError(
                    f"模型仍被运行中的任务使用：{active_consumers[0]}；请先结束任务",
                    status=HTTPStatus.CONFLICT,
                )
            self.tasks.check_idle(require_reset=False)
            # Close the new-task gate durably before stopping the shared model.
            # Otherwise a scheduler request could enter after this check and
            # start using the model while it is being shut down.
            self.tasks.cancel_reset_admission("MODEL_CLOSED")
            self._workstation.update(accepting=False, reason="正在关闭模型服务；新任务已暂停")
            self.tasks.persist()
        if self.enable_actions:
            self.deployment.stop_model(model)
        with self._lock:
            job = self._jobs[job_id]
            stopped_service = {
                "status": "stopped",
                "phase": "stopped",
                "phase_label": "模型服务已关闭",
                "healthy": False,
                "owned_by_job": False,
                "pgid": None,
                "recent_log": "模型服务已由操作员关闭并释放受管资源",
            }
            affected_jobs = [
                candidate
                for candidate in self._jobs.values()
                if str(candidate.config.get("model_id")) == model.name
            ]
            for candidate in affected_jobs:
                candidate.model_service.update(stopped_service)
                self._release_model(candidate)
            self._event(job, "model_stopped", "模型服务已关闭", 100)
            self._model_resources.pop(model.name, None)
            self._reserved_models.pop(model.name, None)
            self._workstation["reason"] = "模型已关闭；重新部署并核实后允许新任务"
            self.tasks.persist()
            return job.to_dict()

    def _job_in_state(
        self, job_id: str, allowed: set[str], message: str
    ) -> DeploymentJob:
        try:
            job = self._jobs[job_id]
        except KeyError as error:
            raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND) from error
        if job.status not in allowed:
            raise ValidationError(message, status=HTTPStatus.CONFLICT)
        return job

    def _release_model(self, job: DeploymentJob) -> None:
        model_id = str(job.config.get("model_id", ""))
        resource = self._model_resources.get(model_id)
        if resource is not None:
            resource.setdefault("consumers", set()).discard(job.id)
            if not resource.get("consumers") and resource.get("owner_job_id") is None:
                self._model_resources.pop(model_id, None)
                resource = None
        keeps_owned_service = bool(
            resource is not None
            and resource.get("owner_job_id") == job.id
            and job.model_service.get("status") != "stopped"
        )
        if self._reserved_models.get(model_id) == job.id and not keeps_owned_service:
            self._reserved_models.pop(model_id, None)
        gpu_id = str(job.model_service.get("gpu", ""))
        if self._reserved_gpus.get(gpu_id) == job.id and not keeps_owned_service:
            self._reserved_gpus.pop(gpu_id, None)

    def _release_robot(self, job: DeploymentJob) -> None:
        robot_id = str(job.config.get("robot_id", ""))
        if self._reserved_robots.get(robot_id) == job.id:
            self._reserved_robots.pop(robot_id, None)
        self._telemetry_tokens.pop(job.id, None)
        # The vision process owns no robot resources, but its credentials must
        # die with the human task.  This also makes a late image POST harmless.
        self._stop_presence_sidecar(job.id, clear_gate=True)

    def wait_for_terminal(self, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            if (
                job["status"] in TERMINAL_JOB_STATES
                or job["status"] == "waiting_approval"
                or job["status"] == "ready_to_start"
                or job["status"] in CONTROLLABLE_JOB_STATES
            ):
                return job
            time.sleep(0.01)
        return self.get_job(job_id)

    def get_control_telemetry(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            value = copy.deepcopy(job.telemetry.get("control", {}))
        received_at = value.get("received_at")
        value["age_s"] = (
            round(max(0.0, time.time() - float(received_at)), 3)
            if isinstance(received_at, (int, float))
            else None
        )
        return value

    def _queue_simulator_pose(
        self,
        job_id: str,
        sequence: int,
        measured_state: list[float],
    ) -> None:
        """Replace the display mailbox without blocking telemetry ingestion.

        This mailbox is intentionally size one.  A slow renderer drops stale
        visual frames instead of creating backpressure toward the robot or its
        100 Hz safety/control path.
        """

        with self._simulator_bridge_lock:
            self._simulator_pending_pose = (job_id, sequence, list(measured_state))
            self._simulator_bridge_event.set()

    def _simulator_bridge_loop(self) -> None:
        next_render_at = 0.0
        while True:
            self._simulator_bridge_event.wait()
            remaining = next_render_at - time.monotonic()
            if remaining > 0:
                # This is a display-only daemon. While it waits, producers keep
                # replacing the one-slot mailbox, so no historical pose queue
                # can form and nothing blocks robot telemetry ingestion.
                time.sleep(remaining)
            with self._simulator_bridge_lock:
                pending = self._simulator_pending_pose
                self._simulator_pending_pose = None
                self._simulator_bridge_event.clear()
            if pending is None:
                continue
            job_id, sequence, measured_state = pending
            next_render_at = time.monotonic() + 1.0 / SIMULATOR_BRIDGE_HZ
            try:
                with self._lock:
                    job = self._jobs.get(job_id)
                    robot = (
                        self.deployment.robots.get(str(job.config.get("robot_id")))
                        if job is not None
                        else None
                    )
                profile_name = (
                    str(job.config.get("effector_profile", robot.effector_profile))
                    if job is not None and robot is not None
                    else "omnipicker_legacy"
                )
                data = self._push_simulator_joints(
                    measured_state, effector_profile=profile_name
                )
            except (ValidationError, TaskError) as error:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None:
                        control = job.telemetry.setdefault("control", {})
                        control["visualization"] = {
                            "rendered_sequence": None,
                            "rendered_at": time.time(),
                            "error": str(error),
                        }
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    control = job.telemetry.setdefault("control", {})
                    control["visualization"] = {
                        "rendered_sequence": sequence,
                        "rendered_at": time.time(),
                        "revision": data.get("revision"),
                        "changed_links": data.get("changed_links"),
                        "error": None,
                    }

    def _push_simulator_joints(
        self,
        measured_state: list[float],
        *,
        effector_profile: str = "omnipicker_legacy",
    ) -> Mapping[str, Any]:
        if self.simulator_control_url is None:
            raise ValidationError(
                "完整三维渲染服务未配置", status=HTTPStatus.SERVICE_UNAVAILABLE
            )
        values = list(measured_state)
        if len(values) != len(G2_VISUAL_JOINT_NAMES) or not all(
            math.isfinite(item) for item in values
        ):
            raise ValidationError("G2 实测关节必须是完整有限的 24 维数组")
        # The bundled G2_dahuan visual URDF uses 0=open and positive prismatic
        # travel=closed.  Robot telemetry uses 0=closed and 0.025=open for DH,
        # while the optional legacy profile uses -pi/4=closed and 0=open.
        # This display-only conversion cannot feed safety or robot control.
        for index in (7, 15):
            aperture = values[index]
            if effector_profile == "dahuan":
                values[index] = max(0.0, min(0.025, 0.025 - aperture))
            else:
                legacy = max(-math.pi / 4, min(0.0, aperture))
                values[index] = -legacy * 0.025 / (math.pi / 4)
        body = json.dumps(
            {"joints": dict(zip(G2_VISUAL_JOINT_NAMES, values, strict=True))}
        ).encode("utf-8")
        request = urllib_request.Request(
            f"{self.simulator_control_url}/api/joints",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=0.5) as response:  # noqa: S310
                result = json.loads(response.read(128 * 1024))
        except (OSError, ValueError, urllib_error.URLError) as error:
            raise ValidationError(
                f"完整三维渲染服务不可用：{error}",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            ) from error
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise ValidationError("完整三维渲染服务拒绝关节更新")
        data = result.get("data", {})
        return data if isinstance(data, Mapping) else {}

    def sync_simulator(self, job_id: str, csrf_token: str | None) -> dict[str, Any]:
        """Push the latest measured G2 pose to the isolated visualizer.

        The browser requests this best-effort, low-rate visualization bridge. It is
        deliberately outside both the policy WebSocket and the robot's 100 Hz safety loop.
        """

        self._require_csrf(csrf_token)
        if self.simulator_control_url is None:
            raise ValidationError("完整三维渲染服务未配置", status=HTTPStatus.SERVICE_UNAVAILABLE)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            if str(job.config.get("robot_id")) != "g2a":
                raise ValidationError("当前三维资产只支持 G2", status=HTTPStatus.CONFLICT)
            control = copy.deepcopy(job.telemetry.get("control", {}))
        measured = control.get("measured_state")
        if not isinstance(measured, list) or len(measured) != len(G2_VISUAL_JOINT_NAMES):
            raise ValidationError("尚未收到完整的 G2 实测关节", status=HTTPStatus.CONFLICT)
        try:
            values = [float(item) for item in measured]
        except (TypeError, ValueError) as error:
            raise ValidationError("G2 实测关节包含非数值元素") from error
        data = self._push_simulator_joints(
            values,
            effector_profile=str(job.config.get("effector_profile", "omnipicker_legacy")),
        )
        return {
            "ok": True,
            "sequence": control.get("sequence"),
            "revision": data.get("revision"),
        }

    def set_simulator_view(
        self, job_id: str, view: Any, csrf_token: str | None
    ) -> dict[str, Any]:
        self._require_csrf(csrf_token)
        if view not in {"near", "overview"}:
            raise ValidationError("三维视角必须是 near 或 overview")
        if self.simulator_control_url is None:
            raise ValidationError("完整三维渲染服务未配置", status=HTTPStatus.SERVICE_UNAVAILABLE)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            if str(job.config.get("robot_id")) != "g2a":
                raise ValidationError("当前三维资产只支持 G2", status=HTTPStatus.CONFLICT)
        request = urllib_request.Request(
            f"{self.simulator_control_url}/api/camera/view",
            data=json.dumps({"view": view}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=0.8) as response:  # noqa: S310
                result = json.loads(response.read(32 * 1024))
        except (OSError, ValueError, urllib_error.URLError) as error:
            raise ValidationError(
                f"切换三维视角失败：{error}", status=HTTPStatus.SERVICE_UNAVAILABLE
            ) from error
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise ValidationError("完整三维渲染服务拒绝切换视角")
        self._simulator_view_initialized = True
        data = result.get("data", {})
        return {
            "ok": True,
            "view": view,
            "preset": data.get("preset") if isinstance(data, Mapping) else None,
        }

    def save_simulator_view(
        self, job_id: str, view: Any, camera: Any, csrf_token: str | None
    ) -> dict[str, Any]:
        self._require_csrf(csrf_token)
        if view not in {"near", "overview"}:
            raise ValidationError("三维视角必须是 near 或 overview")
        if not isinstance(camera, Mapping):
            raise ValidationError("三维相机参数缺失")
        if self.simulator_control_url is None:
            raise ValidationError("完整三维渲染服务未配置", status=HTTPStatus.SERVICE_UNAVAILABLE)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValidationError("部署任务不存在", status=HTTPStatus.NOT_FOUND)
            if str(job.config.get("robot_id")) != "g2a":
                raise ValidationError("当前三维资产只支持 G2", status=HTTPStatus.CONFLICT)
        body = json.dumps({"view": view, "camera": dict(camera)}).encode("utf-8")
        request = urllib_request.Request(
            f"{self.simulator_control_url}/api/camera/preset",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=0.8) as response:  # noqa: S310
                result = json.loads(response.read(32 * 1024))
        except (OSError, ValueError, urllib_error.URLError) as error:
            raise ValidationError(
                f"保存三维视角失败：{error}", status=HTTPStatus.SERVICE_UNAVAILABLE
            ) from error
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise ValidationError("完整三维渲染服务拒绝保存视角")
        data = result.get("data", {})
        return {
            "ok": True,
            "view": view,
            "preset": data.get("preset") if isinstance(data, Mapping) else None,
        }


from nexus_deploy.control.http_api import (
    MAX_REQUEST_BYTES,
    STATIC_DIRECTORY,
    NexusHTTPServer,
    _handler_factory,
)
from nexus_deploy.control.telemetry import (
    TelemetryTCPServer,
    _telemetry_stream_handler,
)


from nexus_deploy.control.cli import build_parser, main


if __name__ == "__main__":
    main()
