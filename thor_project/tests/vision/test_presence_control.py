"""Pure tests for the advisory parcel-presence task gate.

No test here imports GDK, opens a camera, starts a model, or sends a robot
command.  They exercise only deterministic timing and acknowledgement rules.
"""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from nexus_deploy.vision.parcel.runtime.presence import (  # noqa: E402
    NO_PACKAGE,
    PACKAGE_PRESENT,
    UNKNOWN,
)
from nexus_deploy.vision.parcel.runtime.presence_control import (  # noqa: E402
    GateAction,
    PresenceTaskGate,
    PresenceTaskGateConfig,
    TaskState,
)
from nexus_deploy.vision.parcel.run_presence_shadow import build_parser, replay  # noqa: E402


def ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


class PresenceTaskGateTests(unittest.TestCase):
    def make_gate(self, **overrides) -> PresenceTaskGate:
        values = {
            "enabled": True,
            "pause_after_s": 1.5,
            "reset_after_s": 10.0,
            "max_state_age_s": 0.75,
        }
        values.update(overrides)
        gate = PresenceTaskGate(PresenceTaskGateConfig(**values))
        gate.start_session("task-123")
        return gate

    def observe(
        self,
        gate: PresenceTaskGate,
        state: str,
        seconds: float,
        task: TaskState,
        *,
        now_seconds: float | None = None,
    ):
        # Supply the intermediate video samples needed for continuous evidence.
        previous = getattr(gate, "_test_last_observation", None)
        if (state == NO_PACKAGE and previous and previous[0] == NO_PACKAGE
                and previous[2] and now_seconds is None):
            step = previous[1] + 0.5
            while step < seconds - 1e-6:
                gate.observe(NO_PACKAGE, observed_at_ns=ns(step), now_ns=ns(step), task_state=task)
                step += 0.5
        observed = ns(seconds)
        now = observed if now_seconds is None else ns(now_seconds)
        result = gate.observe(state, observed_at_ns=observed, now_ns=now, task_state=task)
        gate._test_last_observation = (state, seconds, result.source_fresh)
        return result

    def test_continuous_no_package_proposes_pause_then_reset(self):
        gate = self.make_gate()

        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING).action,
            GateAction.NONE,
        )
        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 2.49, TaskState.RUNNING).action,
            GateAction.NONE,
        )
        pause = self.observe(gate, NO_PACKAGE, 2.50, TaskState.RUNNING)
        self.assertEqual(pause.action, GateAction.PAUSE)
        self.assertAlmostEqual(pause.no_package_elapsed_s or 0.0, 1.5)

        # In shadow mode an adapter may simulate this acknowledgement.  A real
        # integration must call it only after its existing pause endpoint
        # confirms success.
        gate.acknowledge(GateAction.PAUSE, applied=True)
        self.assertTrue(gate.auto_paused)
        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 10.99, TaskState.PAUSED).action,
            GateAction.NONE,
        )
        reset = self.observe(gate, NO_PACKAGE, 11.0, TaskState.PAUSED)
        self.assertEqual(reset.action, GateAction.RESET)
        self.assertAlmostEqual(reset.no_package_elapsed_s or 0.0, 10.0)

    def test_unknown_breaks_continuous_empty_timer_and_never_requests_action(self):
        gate = self.make_gate()
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        unknown = self.observe(gate, UNKNOWN, 2.0, TaskState.RUNNING)
        self.assertEqual(unknown.action, GateAction.NONE)
        self.assertIsNone(unknown.no_package_elapsed_s)

        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 3.0, TaskState.RUNNING).action,
            GateAction.NONE,
        )
        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 4.49, TaskState.RUNNING).action,
            GateAction.NONE,
        )

    def test_package_present_clears_timer_but_auto_resume_is_off_by_default(self):
        gate = self.make_gate()
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        pause = self.observe(gate, NO_PACKAGE, 2.5, TaskState.RUNNING)
        self.assertEqual(pause.action, GateAction.PAUSE)
        gate.acknowledge(GateAction.PAUSE, applied=True)

        present = self.observe(gate, PACKAGE_PRESENT, 3.0, TaskState.PAUSED)
        self.assertEqual(present.action, GateAction.NONE)
        self.assertIsNone(present.no_package_elapsed_s)
        self.assertTrue(gate.auto_paused)

    def test_auto_resume_can_only_resume_same_acknowledged_gate_pause(self):
        gate = self.make_gate(auto_resume=True)
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        pause = self.observe(gate, NO_PACKAGE, 2.5, TaskState.RUNNING)
        self.assertEqual(pause.action, GateAction.PAUSE)
        gate.acknowledge(GateAction.PAUSE, applied=True)

        resume = self.observe(gate, PACKAGE_PRESENT, 3.0, TaskState.PAUSED)
        self.assertEqual(resume.action, GateAction.RESUME)
        gate.acknowledge(GateAction.RESUME, applied=True)
        self.assertFalse(gate.auto_paused)

    def test_reset_waits_for_completion_then_can_resume_the_same_session(self):
        gate = self.make_gate(auto_resume=True)
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        pause = self.observe(gate, NO_PACKAGE, 2.5, TaskState.RUNNING)
        self.assertEqual(pause.action, GateAction.PAUSE)
        gate.acknowledge(GateAction.PAUSE, applied=True)

        reset = self.observe(gate, NO_PACKAGE, 11.0, TaskState.PAUSED)
        self.assertEqual(reset.action, GateAction.RESET)
        gate.acknowledge(GateAction.RESET, applied=True)
        self.assertTrue(gate.awaiting_reset_ready)

        # A package cannot resume motion while physical reset remains pending.
        blocked = self.observe(gate, PACKAGE_PRESENT, 11.1, TaskState.PAUSED)
        self.assertEqual(blocked.action, GateAction.NONE)
        self.assertTrue(blocked.awaiting_reset_ready)

        # The future adapter must fence this call behind its reset-completion,
        # client-ownership, emergency-stop and session-health checks.
        gate.mark_reset_complete()
        self.assertFalse(gate.awaiting_reset_ready)
        resume = self.observe(gate, PACKAGE_PRESENT, 11.2, TaskState.PAUSED)
        self.assertEqual(resume.action, GateAction.RESUME)
        gate.acknowledge(GateAction.RESUME, applied=True)

    def test_one_empty_period_cannot_repeatedly_reset_after_completion(self):
        gate = self.make_gate()
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        pause = self.observe(gate, NO_PACKAGE, 2.5, TaskState.RUNNING)
        gate.acknowledge(pause.action, applied=True)
        reset = self.observe(gate, NO_PACKAGE, 11.0, TaskState.PAUSED)
        gate.acknowledge(reset.action, applied=True)
        gate.mark_reset_complete()

        held = self.observe(gate, NO_PACKAGE, 25.0, TaskState.PAUSED)
        self.assertEqual(held.action, GateAction.NONE)
        self.assertIn("already completed a reset", held.reason)

    def test_manual_pause_never_becomes_an_automatic_reset(self):
        gate = self.make_gate()
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.PAUSED)
        decision = self.observe(gate, NO_PACKAGE, 11.0, TaskState.PAUSED)
        self.assertEqual(decision.action, GateAction.NONE)
        self.assertIn("manually paused", decision.reason)

    def test_running_after_gate_pause_requires_explicit_reconciliation(self):
        gate = self.make_gate()
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        pause = self.observe(gate, NO_PACKAGE, 2.5, TaskState.RUNNING)
        gate.acknowledge(pause.action, applied=True)

        mismatch = self.observe(gate, NO_PACKAGE, 3.0, TaskState.RUNNING)
        self.assertEqual(mismatch.action, GateAction.NONE)
        self.assertIn("reconciliation", mismatch.reason)

        gate.mark_externally_resumed()
        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 4.0, TaskState.RUNNING).action,
            GateAction.NONE,
        )

    def test_stale_visual_result_cannot_advance_or_preserve_empty_timer(self):
        gate = self.make_gate(max_state_age_s=0.5)
        stale = self.observe(
            gate,
            NO_PACKAGE,
            1.0,
            TaskState.RUNNING,
            now_seconds=1.6,
        )
        self.assertFalse(stale.source_fresh)
        self.assertEqual(stale.action, GateAction.NONE)
        self.assertIsNone(stale.no_package_elapsed_s)

        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 3.0, TaskState.RUNNING).action,
            GateAction.NONE,
        )

    def test_disabled_gate_never_proposes_an_action(self):
        gate = PresenceTaskGate(PresenceTaskGateConfig(enabled=False))
        gate.start_session("task-123")
        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING).action,
            GateAction.NONE,
        )
        self.assertEqual(
            self.observe(gate, NO_PACKAGE, 20.0, TaskState.RUNNING).action,
            GateAction.NONE,
        )

    def test_duplicate_or_out_of_order_visual_result_is_ignored(self):
        gate = self.make_gate()
        self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        duplicate = self.observe(gate, NO_PACKAGE, 1.0, TaskState.RUNNING)
        self.assertFalse(duplicate.source_fresh)
        self.assertEqual(duplicate.action, GateAction.NONE)

    def test_offline_shadow_replay_simulates_but_never_executes_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "presence.jsonl"
            output_path = root / "shadow.jsonl"
            rows = [
                {"video_timestamp_seconds": 0.1 + index * 0.5, "presence": {"state": NO_PACKAGE}}
                for index in range(22)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = replay(
                input_path,
                output_path,
                config=PresenceTaskGateConfig(enabled=True),
                session_id="offline-test",
                simulate_ack=True,
                overwrite=False,
            )
            self.assertEqual(result["records"], 22)
            self.assertEqual(result["pause"], 1)
            self.assertEqual(result["reset"], 1)
            output_rows = [
                json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            actions = [row["presence_gate_shadow"]["decision"]["action"] for row in output_rows]
            self.assertEqual(actions.count("reset"), 1)
            self.assertEqual(actions[-1], "none")
            self.assertEqual(
                output_rows[-1]["presence_gate_shadow"]["task_state_after"],
                "ended",
            )

    def test_shadow_cli_defaults_to_same_session_auto_resume(self):
        parser = build_parser()
        default = parser.parse_args(
            ["--input-jsonl", "input.jsonl", "--output-jsonl", "output.jsonl"]
        )
        self.assertTrue(default.auto_resume)
        manual = parser.parse_args(
            [
                "--input-jsonl",
                "input.jsonl",
                "--output-jsonl",
                "output.jsonl",
                "--no-auto-resume",
            ]
        )
        self.assertFalse(manual.auto_resume)


if __name__ == "__main__":
    unittest.main()
