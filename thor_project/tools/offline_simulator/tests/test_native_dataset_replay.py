from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT_VALUE = os.environ.get("NEXUS_SIM_DH_DATASET_ROOT")
DATASET_ROOT = Path(DATASET_ROOT_VALUE) if DATASET_ROOT_VALUE else None
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus_simulator.dataset import (
    EFFECTOR_INDICES,
    NATIVE_DH_EFFECTOR_ENCODING,
    NATIVE_DH_EFFECTOR_UPPER_M,
    EpisodeData,
    LeRobotDataset,
)


class NativeEffectorProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        zeros = np.zeros((1, 24), dtype=np.float64)
        self.episode = EpisodeData(
            index=0,
            fps=30,
            task="fixture",
            states=zeros,
            actions=zeros,
            timestamps=np.zeros(1),
            video_path=Path("fixture.mp4"),
            video_from_s=0,
            video_to_s=1 / 30,
            effector_encoding=NATIVE_DH_EFFECTOR_ENCODING,
        )

    def test_observation_is_inverted_for_safety_and_round_trips(self) -> None:
        vector = np.zeros(24, dtype=np.float64)
        vector[7], vector[15] = 0.005, 0.020
        safety = self.episode.to_safety_vector(vector, "observation")
        self.assertAlmostEqual(safety[7], 0.020)
        self.assertAlmostEqual(safety[15], 0.005)
        np.testing.assert_allclose(
            self.episode.from_safety_vector(safety, "observation"), vector
        )

    def test_action_keeps_native_safety_command_polarity(self) -> None:
        vector = np.zeros(24, dtype=np.float64)
        vector[list(EFFECTOR_INDICES)] = NATIVE_DH_EFFECTOR_UPPER_M
        np.testing.assert_allclose(self.episode.to_safety_vector(vector, "action"), vector)
        np.testing.assert_allclose(self.episode.from_safety_vector(vector, "action"), vector)


@unittest.skipUnless(
    DATASET_ROOT is not None and DATASET_ROOT.is_dir(),
    "set NEXUS_SIM_DH_DATASET_ROOT to run native dataset integration tests",
)
class NativeDatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert DATASET_ROOT is not None
        cls.episode = LeRobotDataset(DATASET_ROOT).load_episode(0)

    def test_episode_zero_is_detected_as_native_dh(self) -> None:
        self.assertEqual(self.episode.effector_encoding, NATIVE_DH_EFFECTOR_ENCODING)
        self.assertEqual(self.episode.frame_count, 5123)
        self.assertEqual(self.episode.fps, 30)
        effectors = np.concatenate(
            [
                self.episode.states[:, list(EFFECTOR_INDICES)].ravel(),
                self.episode.actions[:, list(EFFECTOR_INDICES)].ravel(),
            ]
        )
        self.assertGreaterEqual(effectors.min(), 0.0)
        self.assertLessEqual(effectors.max(), NATIVE_DH_EFFECTOR_UPPER_M + 1e-6)


if __name__ == "__main__":
    unittest.main()
