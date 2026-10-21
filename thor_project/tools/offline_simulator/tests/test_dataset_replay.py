from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT_VALUE = os.environ.get("NEXUS_SIM_DATASET_ROOT")
DATASET_ROOT = Path(DATASET_ROOT_VALUE) if DATASET_ROOT_VALUE else None
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus_simulator.dataset import G2_URDF_JOINTS, LeRobotDataset
from nexus_simulator.safety_reference import SAFETY_SOURCE, SafetyReference


@unittest.skipUnless(
    DATASET_ROOT is not None and DATASET_ROOT.is_dir(),
    "set NEXUS_SIM_DATASET_ROOT to run dataset integration tests",
)
class DatasetReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert DATASET_ROOT is not None
        cls.episode = LeRobotDataset(DATASET_ROOT).load_episode(0)

    def test_episode_zero_metadata_and_joint_mapping(self) -> None:
        self.assertEqual(self.episode.frame_count, 1232)
        self.assertEqual(self.episode.fps, 30)
        self.assertAlmostEqual(self.episode.duration_s, 41.0666666667)
        self.assertIn("QR code facing up", self.episode.task)
        self.assertEqual(tuple(self.episode.joint_values(0, "observation")), G2_URDF_JOINTS)
        self.assertEqual(self.episode.metadata()["display_fps"], 30)

    @unittest.skipUnless(SAFETY_SOURCE.is_file(), "production safety reference unavailable")
    def test_raised_surface_triggers_pinned_safety_reference(self) -> None:
        reference = SafetyReference(PROJECT_ROOT / "assets/g2/G2_omnipicker")
        low = reference.evaluate(
            self.episode.states[0], self.episode.states[1], height=0.75, hard_margin=0.03
        )
        raised = reference.evaluate(
            self.episode.states[0], self.episode.states[1], height=1.0, hard_margin=0.03
        )
        self.assertTrue(low["safe"])
        self.assertGreater(low["clearance"], 0)
        self.assertFalse(raised["safe"])
        self.assertEqual(raised["reason"], "zero_motion_candidate_unsafe")
        self.assertEqual(raised["action_scale"], 0.0)

    @unittest.skipUnless(SAFETY_SOURCE.is_file(), "production safety reference unavailable")
    def test_collision_response_rechecks_and_holds_previous_safe_action(self) -> None:
        reference = SafetyReference(PROJECT_ROOT / "assets/g2/G2_omnipicker")
        measured = self.episode.actions[823]
        dangerous_candidate = self.episode.actions[824]
        result = reference.collision_response(
            measured,
            dangerous_candidate,
            measured,
            height=0.77,
            hard_margin=0.03,
        )
        self.assertTrue(result["candidate_rejected"])
        self.assertTrue(result["previous_recheck_safe"])
        self.assertEqual(result["collision_response"], "hold_previous_successful_action")
        self.assertEqual(result["reason"], "collision_risk_hold_previous_safe_action")
        self.assertEqual(result["adopted_action"], measured.tolist())
        self.assertNotEqual(result["adopted_action"], dangerous_candidate.tolist())

    @unittest.skipUnless(SAFETY_SOURCE.is_file(), "production safety reference unavailable")
    def test_runtime_switch_keeps_detection_but_bypasses_or_applies_limit(self) -> None:
        reference = SafetyReference(PROJECT_ROOT / "assets/g2/G2_omnipicker")
        measured = self.episode.actions[823]
        candidate = self.episode.actions[824]
        enabled, limited_action = reference.evaluate_action(
            measured,
            candidate,
            measured,
            height=0.77,
            hard_margin=0.03,
            enforcement_enabled=True,
        )
        disabled, unrestricted_action = reference.evaluate_action(
            measured,
            candidate,
            measured,
            height=0.77,
            hard_margin=0.03,
            enforcement_enabled=False,
        )
        self.assertTrue(enabled["collision_detected"])
        self.assertTrue(enabled["enforcement_applied"])
        self.assertTrue(enabled["enforcement_effective"])
        self.assertTrue(enabled["adopted_pose_safe"])
        self.assertEqual(enabled["action_disposition"], "hold_previous_successful_action")
        self.assertEqual(limited_action.tolist(), measured.tolist())
        self.assertTrue(disabled["collision_detected"])
        self.assertFalse(disabled["enforcement_applied"])
        self.assertFalse(disabled["enforcement_effective"])
        self.assertEqual(disabled["action_disposition"], "monitor_only_allow_unsafe_candidate")
        self.assertEqual(unrestricted_action.tolist(), candidate.tolist())


if __name__ == "__main__":
    unittest.main()
