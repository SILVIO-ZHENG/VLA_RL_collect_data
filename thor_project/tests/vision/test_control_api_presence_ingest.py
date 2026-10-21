"""Control-plane contract tests without loading GDK, a model, or a robot.

These tests import the staged control module with the unrelated deployment
dependencies replaced by inert stand-ins.  They exercise the narrow sidecar
ingress boundary: only bounded observations reach the pure gate, and a
sidecar error can never be treated as evidence that the work area is empty.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "nexus_deploy"
CONTROL_PATH = PACKAGE_ROOT / "control_api.py"


def load_control_module():
    """Load the staged API with only its unused deployment imports stubbed."""

    package = types.ModuleType("nexus_deploy")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules.setdefault("nexus_deploy", package)

    deployctl = types.ModuleType("nexus_deploy.deployment")

    class DeployError(Exception):
        pass

    class Deployment:  # pragma: no cover - only satisfies an import annotation
        pass

    deployctl.POLICY_PROTOCOL_NAME = "nexus-policy-websocket"
    deployctl.DeployError = DeployError
    deployctl.Deployment = Deployment
    sys.modules["nexus_deploy.deployment"] = deployctl

    safety_profiles = types.ModuleType("nexus_deploy.safety_profiles")

    class AlignmentSafetyProfileRepository:  # pragma: no cover - import stub
        pass

    safety_profiles.AlignmentSafetyProfileRepository = AlignmentSafetyProfileRepository
    sys.modules["nexus_deploy.safety_profiles"] = safety_profiles

    simulator = types.ModuleType("nexus_deploy.viewer")

    class SimulatorProcess:  # pragma: no cover - import stub
        pass

    simulator.SimulatorProcess = SimulatorProcess
    sys.modules["nexus_deploy.viewer"] = simulator

    spec = importlib.util.spec_from_file_location("test_runtime_control_api", CONTROL_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {CONTROL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


API = load_control_module()


class RecordedThread:
    """Capture a queued action without running it against a robot stand-in."""

    started: list[tuple[object, tuple[object, ...]]] = []

    def __init__(self, *, target, args=(), **_kwargs):
        self.target = target
        self.args = tuple(args)

    def start(self) -> None:
        self.__class__.started.append((self.target, self.args))


class ControlApiPresenceIngressTests(unittest.TestCase):
    job_id = "a" * 32
    token = "t" * 32

    def setUp(self) -> None:
        RecordedThread.started.clear()
        self.plane = object.__new__(API.ControlPlane)
        self.plane._lock = threading.RLock()
        job = API.DeploymentJob(
            id=self.job_id,
            config={"presence_gate": {"enabled": True}},
            status="running",
            task_session={
                "status": "running",
                "generation": 1,
                "cancel_requested": False,
                "remote_started": True,
            },
        )
        job.presence_gate = API.ControlPlane._initial_presence_gate_status(job.config)
        self.plane._jobs = {self.job_id: job}
        self.plane._presence_tokens = {self.job_id: self.token}
        gate = API.PresenceTaskGate(
            API.PresenceTaskGateConfig(
                enabled=True,
                pause_after_s=1.5,
                reset_after_s=10.0,
                max_state_age_s=0.75,
                auto_resume=True,
            )
        )
        gate.start_session(self.job_id)
        self.plane._presence_task_gates = {self.job_id: gate}

    @staticmethod
    def payload(*, sequence: int, state: str, observed_at_ns: int, status: str = "ok") -> dict:
        window = [state] * 3 if state != "UNKNOWN" else []
        return {
            "generation": 1,
            "sequence": sequence,
            "status": status,
            "presence": {
                "state": state,
                "observed_at_ns": observed_at_ns,
                "window_states": window,
                "camera_windows": {
                    camera: window.copy()
                    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                },
            },
        }

    def ingest(self, *, sequence: int, state: str, observed_at_ns: int, status: str = "ok") -> dict:
        with mock.patch.object(API.threading, "Thread", RecordedThread):
            with mock.patch.object(API.time, "time_ns", return_value=observed_at_ns):
                return self.plane.ingest_presence_gate(
                    self.job_id,
                    self.payload(
                        sequence=sequence,
                        state=state,
                        observed_at_ns=observed_at_ns,
                        status=status,
                    ),
                    self.token,
                )

    def test_sidecar_error_is_conservatively_converted_to_unknown(self) -> None:
        base = time.time_ns()
        self.assertEqual(
            self.ingest(sequence=1, state="NO_PACKAGE", observed_at_ns=base)["action"],
            "none",
        )
        result = self.ingest(
            sequence=2,
            state="NO_PACKAGE",
            observed_at_ns=base + 2_000_000_000,
            status="inference_error",
        )
        self.assertEqual(result["action"], "none")
        status = self.plane._jobs[self.job_id].presence_gate
        self.assertEqual(status["state"], "UNKNOWN")
        self.assertEqual(status["source_status"], "inference_error")
        self.assertIsNone(status["no_package_elapsed_s"])
        self.assertEqual(RecordedThread.started, [])

    def test_control_plane_queues_pause_only_after_three_camera_confirmation(self) -> None:
        base = time.time_ns()
        self.ingest(sequence=1, state="NO_PACKAGE", observed_at_ns=base)
        result = None
        for sequence in range(2, 6):
            result = self.ingest(
                sequence=sequence, state="NO_PACKAGE",
                observed_at_ns=base + (sequence - 1) * 400_000_000,
            )
        self.assertEqual(result["action"], "pause")
        self.assertEqual(len(RecordedThread.started), 1)
        _target, args = RecordedThread.started[0]
        self.assertEqual(args, (self.job_id, 1, API.GateAction.PAUSE))
        status = self.plane._jobs[self.job_id].presence_gate
        self.assertEqual(status["pending_action"], "pause")
        self.assertEqual(status["action_status"], "queued")

    def test_missing_wrist_evidence_cannot_report_no_package(self) -> None:
        payload = self.payload(sequence=1, state="NO_PACKAGE", observed_at_ns=time.time_ns())
        del payload["presence"]["camera_windows"]["cam_right_wrist"]
        with self.assertRaises(API.ValidationError):
            self.plane._parse_presence_observation(payload)

    def test_unknown_camera_result_does_not_queue_pause(self) -> None:
        payload = self.payload(sequence=1, state="UNKNOWN", observed_at_ns=time.time_ns())
        payload["presence"]["camera_windows"]["cam_right_wrist"] = ["UNKNOWN"]
        with mock.patch.object(API.threading, "Thread", RecordedThread):
            with mock.patch.object(API.time, "time_ns", return_value=payload["presence"]["observed_at_ns"]):
                response = self.plane.ingest_presence_gate(self.job_id, payload, self.token)
        self.assertEqual(response["action"], "none")
        self.assertEqual(RecordedThread.started, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
