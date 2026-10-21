"""Visual adapters emit facts only. No adapter owns robot command capability."""
from __future__ import annotations
import json
import os
import secrets
import subprocess
import time
from typing import Any, Mapping, Protocol
from nexus_deploy.deployment import DeployError
from nexus_deploy.control.models import PresenceSidecar


class PresenceProvider(Protocol):
    def start(self, job_id: str, generation: int, *, reuse_existing_gate: bool = False) -> dict: ...
    def stop(self, job_id: str, *, expected_generation: int | None = None, clear_gate: bool = True) -> None: ...
    def health(self, job_id: str) -> str: ...
    def receive(self, job_id: str, observation: Mapping, token: str) -> dict: ...


class SidecarPresenceProvider:
    """Compatibility adapter for the verified sidecar process and private ingress."""
    def __init__(self, plane):
        self.plane = plane

    def __getattr__(self, name):
        # Existing lifecycle implementation uses the control-owned session store.
        return getattr(self.plane, name)

    def receive(self, job_id, observation, token):
        return self.plane.ingest_presence_gate(job_id, observation, token)

    def health(self, job_id):
        sidecar = self.plane._presence_sidecars.get(job_id)
        return "running" if sidecar and sidecar.process.poll() is None else "stopped"

    def start(
        self,
        job_id: str,
        generation: int,
        *,
        reuse_existing_gate: bool = False,
    ) -> dict[str, Any]:
        """Start an image-only sidecar before starting the robot client.

        The sidecar receives a per-generation frame token and a separate
        control-plane token.  It never receives SSH credentials, GDK access,
        a browser CSRF token, or an actuator endpoint.
        """

        runtime = self._presence_runtime
        if runtime is None:
            raise DeployError("Package-presence sidecar runtime is not configured")
        if not (
            runtime.project_directory.is_dir()
            and runtime.python_executable.is_file()
            and runtime.weights.is_file()
            and runtime.rules.is_file()
            and (runtime.project_directory / "run_nexus_presence_sidecar.py").is_file()
        ):
            raise DeployError("Package-presence sidecar runtime files are unavailable")
        if isinstance(generation, bool) or generation < 1:
            raise DeployError("Package-presence generation is invalid")

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DeployError("Deployment job no longer exists")
            if not self._presence_enabled(job):
                return {}
            if int(job.task_session.get("generation", 0)) != generation:
                raise DeployError("Package-presence sidecar belongs to an old task generation")
            if job.task_session.get("cancel_requested"):
                raise DeployError("Package-presence sidecar start was cancelled")
            existing = self._presence_sidecars.get(job_id)
            if existing is not None:
                raise DeployError("Package-presence sidecar is already active for this job")
            if reuse_existing_gate:
                gate = self._presence_task_gates.get(job_id)
                if gate is None:
                    raise DeployError("Package-presence gate session is unavailable during reset")
            else:
                gate = self._new_presence_task_gate()
                gate.start_session(job_id)
                self._presence_task_gates[job_id] = gate
            status = self._ensure_presence_gate_status(job)
            status.update(
                {
                    "transport": "starting",
                    "task_generation": generation,
                    "task_state": str(job.task_session.get("status", "starting")),
                    "sequence": None,
                    "received_at": None,
                    "age_s": None,
                    "source_fresh": False,
                    "no_package_elapsed_s": None,
                    "auto_paused": gate.auto_paused,
                    "awaiting_reset_ready": gate.awaiting_reset_ready,
                    "pending_action": (
                        gate.pending_action.value if gate.pending_action is not None else None
                    ),
                    "proposed_action": "none",
                    "reason": "waiting for authenticated head-camera frames",
                    "message": "正在启动包裹检测；仅控制服务可解释视觉状态并执行既有安全动作",
                    "last_error": None,
                }
            )

        runtime_directory = (
            self.deployment.runtime_dir
            / "presence-gate"
            / job_id
            / f"generation-{generation}"
        )
        ingress_token = secrets.token_urlsafe(32)
        control_token = secrets.token_urlsafe(32)
        ingress_token_file = runtime_directory / "ingress.token"
        control_token_file = runtime_directory / "control.token"
        ready_file = runtime_directory / "ready.json"
        log_file = runtime_directory / "sidecar.log"
        process: subprocess.Popen[Any] | None = None
        log_handle: Any = None
        registered = False
        try:
            runtime_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(runtime_directory, 0o700)
            self._write_private_token(ingress_token_file, ingress_token)
            self._write_private_token(control_token_file, control_token)
            log_handle = log_file.open("ab", buffering=0)
            command = [
                str(runtime.python_executable),
                "-E",
                str(runtime.project_directory / "run_nexus_presence_sidecar.py"),
                "--bind-host",
                runtime.bind_host,
                "--port",
                str(runtime.ingress_port),
                "--job-id",
                job_id,
                "--generation",
                str(generation),
                "--ingress-token-file",
                str(ingress_token_file),
                "--control-url",
                f"{runtime.control_api_url}/api/deployments/{job_id}/presence-gate",
                "--control-token-file",
                str(control_token_file),
                "--weights",
                str(runtime.weights),
                "--rules",
                str(runtime.rules),
                "--device",
                runtime.device,
                "--detect-hz",
                str(runtime.detect_hz),
                "--camera-fps",
                "30.0",
                "--ready-file",
                str(ready_file),
                "--log-file",
                str(log_file),
            ]
            sidecar_env = os.environ.copy()
            for variable in (
                "PYTHONPATH",
                "PYTHONHOME",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "CONDA_PROMPT_MODIFIER",
            ):
                sidecar_env.pop(variable, None)
            sidecar_env["VIRTUAL_ENV"] = str(runtime.python_executable.parent.parent)
            process = subprocess.Popen(
                command,
                cwd=str(runtime.project_directory),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=sidecar_env,
            )
            sidecar = PresenceSidecar(
                job_id=job_id,
                generation=generation,
                process=process,
                runtime_directory=runtime_directory,
                ingress_token_file=ingress_token_file,
                control_token_file=control_token_file,
                ready_file=ready_file,
                log_file=log_file,
                log_handle=log_handle,
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if (
                    job is None
                    or job.task_session.get("cancel_requested")
                    or int(job.task_session.get("generation", 0)) != generation
                ):
                    raise DeployError("Package-presence sidecar start was cancelled")
                self._presence_sidecars[job_id] = sidecar
                self._presence_tokens[job_id] = control_token
                registered = True

            deadline = time.monotonic() + runtime.startup_timeout_s
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    tail = self._tail_text(log_file)
                    raise DeployError(
                        "Package-presence sidecar exited before becoming ready"
                        + (f": {tail}" if tail else "")
                    )
                try:
                    ready = json.loads(ready_file.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    ready = None
                if isinstance(ready, Mapping):
                    if (
                        ready.get("status") == "ready"
                        and ready.get("job_id") == job_id
                        and ready.get("generation") == generation
                        and ready.get("bind_host") == runtime.bind_host
                        and ready.get("port") == runtime.ingress_port
                    ):
                        with self._lock:
                            job = self._jobs.get(job_id)
                            if (
                                job is None
                                or job.task_session.get("cancel_requested")
                                or int(job.task_session.get("generation", 0)) != generation
                            ):
                                raise DeployError("Package-presence sidecar start was cancelled")
                            status = self._ensure_presence_gate_status(job)
                            status.update(
                                {
                                    "transport": "awaiting_frames",
                                    "task_generation": generation,
                                    "message": "包裹检测已就绪，等待机器人现有头部相机帧",
                                    "last_error": None,
                                }
                            )
                        return {
                            "_platform_presence_frame_endpoint": (
                                f"tcp://{runtime.bind_host}:{runtime.ingress_port}"
                            ),
                            "_platform_presence_frame_token": ingress_token,
                            "_platform_presence_job_id": job_id,
                            "_platform_presence_generation": generation,
                        }
                time.sleep(0.1)
            raise DeployError("Package-presence sidecar readiness timed out")
        except Exception:
            if registered:
                self._stop_presence_sidecar(
                    job_id,
                    expected_generation=generation,
                    clear_gate=not reuse_existing_gate,
                )
            else:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                if log_handle is not None:
                    log_handle.close()
                for path in (ingress_token_file, control_token_file):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if not reuse_existing_gate:
                    with self._lock:
                        self._presence_task_gates.pop(job_id, None)
            raise

    def stop(
        self,
        job_id: str,
        *,
        expected_generation: int | None = None,
        clear_gate: bool = True,
    ) -> None:
        """Stop one sidecar without ever touching a robot or GDK process."""

        with self._lock:
            sidecar = self._presence_sidecars.get(job_id)
            if sidecar is None:
                if clear_gate:
                    self._presence_task_gates.pop(job_id, None)
                return
            if expected_generation is not None and sidecar.generation != expected_generation:
                return
            self._presence_sidecars.pop(job_id, None)
            self._presence_tokens.pop(job_id, None)
            if clear_gate:
                self._presence_task_gates.pop(job_id, None)
        if sidecar.process.poll() is None:
            sidecar.process.terminate()
            try:
                sidecar.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                sidecar.process.kill()
                try:
                    sidecar.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    pass
        try:
            sidecar.log_handle.close()
        except OSError:
            pass
        for path in (sidecar.ingress_token_file, sidecar.control_token_file):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class MockPresenceProvider:
    """Explicit offline facts; never creates a camera, model or subprocess."""
    def __init__(self, plane):
        self.plane = plane
        self.sessions = {}

    def start(self, job_id, generation, *, reuse_existing_gate=False):
        p = self.plane
        if p.enable_actions:
            raise ValueError("Offline presence requires hardware actions disabled")
        with p._lock:
            job = p._jobs[job_id]
            gate = p._new_presence_task_gate()
            gate.start_session(job_id)
            p._presence_task_gates[job_id] = gate
            p._presence_tokens[job_id] = secrets.token_urlsafe(32)
            self.sessions[job_id] = generation
            job.presence_gate = p._initial_presence_gate_status(job.config)
            job.presence_gate.update(transport="mock", task_generation=generation)
        return {}

    def stop(self, job_id, *, expected_generation=None, clear_gate=True):
        p = self.plane
        with p._lock:
            if expected_generation is not None and self.sessions.get(job_id) != expected_generation:
                return
            self.sessions.pop(job_id, None)
            p._presence_tokens.pop(job_id, None)
            if clear_gate:
                p._presence_task_gates.pop(job_id, None)

    def health(self, job_id):
        return "mock" if job_id in self.sessions else "stopped"

    def receive(self, job_id, observation, token=None):
        return self.plane.ingest_presence_gate(job_id, observation,
                                              token or self.plane._presence_tokens.get(job_id))

    def emit(self, job_id, state, sequence, *, observed_at_ns=None, generation=None):
        return self.receive(job_id, {
            "generation": generation or self.sessions[job_id], "sequence": sequence, "status": "ok",
            "presence": {"state": state, "observed_at_ns": observed_at_ns or time.time_ns(),
                         "window_states": [state] * 4 if state != "UNKNOWN" else []}})


class ReplayPresenceProvider(MockPresenceProvider):
    """Video frames use the exact production estimator; commands remain simulated.

    Imports and model construction belong to the explicit offline runner, never
    to a web request or startup. Timestamps are local receipt times, replay is
    paced by the video's sampling interval and does not claim physical completion.
    """
    def play(self, job_id, video_path, estimator, *, sample_interval_s=0.1):
        import cv2
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        estimator.reset()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Cannot open replay video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise ValueError("Replay video has no valid frame rate")
        index, sequence, next_sample = 0, 0, 0.0
        previous = None
        try:
            while job_id in self.sessions and self.plane._jobs[job_id].status in {
                "task_starting", "running", "paused", "pausing"
            }:
                ok, frame = cap.read()
                if not ok:
                    break
                media_time = index / fps
                index += 1
                if media_time < next_sample:
                    continue
                next_sample = media_time + sample_interval_s
                if previous is not None:
                    time.sleep(sample_interval_s)
                if previous is not None and time.monotonic() - previous > 0.75:
                    estimator.reset()
                result = estimator.process_frame_with_presence(frame, color_space="BGR",
                            source_id="video_replay", frame_index=index, capture_timestamp_ns=time.time_ns())
                previous = time.monotonic()
                sequence += 1
                observation = {"generation": self.sessions[job_id], "sequence": sequence, "status": "ok",
                               "presence": dict(result.presence.to_dict(), observed_at_ns=time.time_ns())}
                self.receive(job_id, observation)
        finally:
            cap.release()
            if job_id in self.sessions and self.plane._jobs[job_id].status in {
                "task_starting", "running", "paused", "pausing"
            }:
                self.emit(job_id, "UNKNOWN", sequence + 1)
