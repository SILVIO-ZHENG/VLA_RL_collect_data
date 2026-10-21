"""Pure contract tests for the Nexus-owned package-presence decision gate."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = PROJECT_ROOT / "src" / "nexus_deploy" / "presence_gate.py"


def load_gate_module():
    # Use a private import name so unittest discovery does not mistake this
    # dynamically loaded staged module for this test module itself.
    spec = importlib.util.spec_from_file_location("staged_presence_gate", GATE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_gate_module()


def ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


class ControlPlanePresenceGateTests(unittest.TestCase):
    def test_service_default_pauses_after_three_and_resets_after_six_seconds(self):
        gate = GATE.PresenceTaskGate(GATE.PresenceTaskGateConfig(enabled=True))
        gate.start_session("manual-task")
        for step in range(13):
            at_s = 1.0 + step * 0.5
            task_state = GATE.TaskState.RUNNING if step <= 6 else GATE.TaskState.PAUSED
            decision = self.observe(gate, GATE.NO_PACKAGE, at_s, task_state)
            if decision.action is GATE.GateAction.PAUSE:
                self.assertEqual(at_s, 4.0)
                gate.acknowledge(GATE.GateAction.PAUSE, applied=True)
            elif at_s < 7.0:
                self.assertNotEqual(decision.action, GATE.GateAction.RESET)
        self.assertEqual(decision.action, GATE.GateAction.RESET)

    def make_gate(self):
        gate = GATE.PresenceTaskGate(
            GATE.PresenceTaskGateConfig(
                enabled=True,
                pause_after_s=1.5,
                reset_after_s=10.0,
                max_state_age_s=0.75,
                auto_resume=True,
            )
        )
        gate.start_session("human-authorized-job:g1")
        return gate

    def observe(self, gate, state, at_s, task_state, now_s=None):
        return gate.observe(
            state,
            observed_at_ns=ns(at_s),
            now_ns=ns(at_s if now_s is None else now_s),
            task_state=task_state,
        )

    def test_continuous_empty_pauses_then_resets_then_present_resumes_same_session(self):
        gate = self.make_gate()
        self.assertEqual(
            self.observe(gate, GATE.NO_PACKAGE, 1.0, GATE.TaskState.RUNNING).action,
            GATE.GateAction.NONE,
        )
        for at_s in (1.5, 2.0):
            self.observe(gate, GATE.NO_PACKAGE, at_s, GATE.TaskState.RUNNING)
        pause = self.observe(gate, GATE.NO_PACKAGE, 2.5, GATE.TaskState.RUNNING)
        self.assertEqual(pause.action, GATE.GateAction.PAUSE)
        gate.acknowledge(pause.action, applied=True)

        for half_second in range(6, 23):
            reset = self.observe(gate, GATE.NO_PACKAGE, half_second / 2, GATE.TaskState.PAUSED)
        self.assertEqual(reset.action, GATE.GateAction.RESET)
        gate.acknowledge(reset.action, applied=True)
        self.assertTrue(gate.awaiting_reset_ready)
        self.assertEqual(
            self.observe(gate, GATE.PACKAGE_PRESENT, 11.2, GATE.TaskState.PAUSED).action,
            GATE.GateAction.NONE,
        )
        gate.mark_reset_complete()
        resume = self.observe(gate, GATE.PACKAGE_PRESENT, 11.3, GATE.TaskState.PAUSED)
        self.assertEqual(resume.action, GATE.GateAction.RESUME)
        self.assertEqual(resume.session_id, "human-authorized-job:g1")

    def test_unknown_breaks_continuous_empty_and_never_requests_control(self):
        gate = self.make_gate()
        self.observe(gate, GATE.NO_PACKAGE, 1.0, GATE.TaskState.RUNNING)
        unknown = self.observe(gate, GATE.UNKNOWN, 2.0, GATE.TaskState.RUNNING)
        self.assertEqual(unknown.action, GATE.GateAction.NONE)
        self.assertIsNone(unknown.no_package_elapsed_s)
        self.assertEqual(
            self.observe(gate, GATE.NO_PACKAGE, 3.0, GATE.TaskState.RUNNING).action,
            GATE.GateAction.NONE,
        )

    def test_stale_or_ended_observation_cannot_create_a_robot_action(self):
        gate = self.make_gate()
        stale = self.observe(gate, GATE.NO_PACKAGE, 1.0, GATE.TaskState.RUNNING, now_s=1.8)
        self.assertFalse(stale.source_fresh)
        self.assertEqual(stale.action, GATE.GateAction.NONE)
        ended = self.observe(gate, GATE.NO_PACKAGE, 2.0, GATE.TaskState.ENDED)
        self.assertEqual(ended.action, GATE.GateAction.NONE)


if __name__ == "__main__":
    unittest.main()
