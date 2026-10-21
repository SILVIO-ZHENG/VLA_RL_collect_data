"""Three-camera voting without a model process or robot connection."""

from __future__ import annotations

import unittest

import numpy as np

from nexus_deploy.vision.parcel.run_nexus_presence_sidecar import ThreeCameraPresenceAggregator
from nexus_deploy.vision.parcel.runtime.core_bridge import Instance
from nexus_deploy.vision.parcel.runtime.presence import (
    NO_PACKAGE, PACKAGE_PRESENT, UNKNOWN, ParcelPresenceGate, PresenceGateConfig,
)


CONFIG = PresenceGateConfig(
    include_roi_names=("frame",), exclude_roi_names=(),
    polygons={"frame": ((0, 0), (1, 0), (1, 1), (0, 1))},
    min_mask_pixels=1, window_frames=3,
)


def frame(present: bool) -> np.ndarray:
    value = 1 if present else 0
    return np.full((10, 10, 3), value, dtype=np.uint8)


def detect(image: np.ndarray):
    if not image[0, 0, 0]:
        return []
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[3:6, 3:6] = True
    return [Instance("package", 0.9, mask)]


class FakeDetector:
    predict = staticmethod(detect)


class ThreeCameraPresenceTests(unittest.TestCase):
    def setUp(self):
        self.head = ParcelPresenceGate(CONFIG)
        self.aggregate = ThreeCameraPresenceAggregator(self.head)
        self.detector = FakeDetector()

    def observe(self, head=False, left=False, right=False):
        head_frame = frame(head)
        head_result = self.head.update(detect(head_frame), head_frame.shape[:2])
        return self.aggregate.update(
            head_result,
            {"cam_left_wrist": frame(left), "cam_right_wrist": frame(right)},
            self.detector,
        )

    def test_all_three_must_be_empty_for_three_samples(self):
        self.assertEqual(self.observe()["state"], UNKNOWN)
        self.assertEqual(self.observe()["state"], UNKNOWN)
        result = self.observe()
        self.assertEqual(result["state"], NO_PACKAGE)
        self.assertTrue(all(window == [NO_PACKAGE] * 3 for window in result["camera_windows"].values()))

    def test_visible_wrist_parcel_blocks_empty_decision(self):
        self.observe()
        self.observe()
        result = self.observe(right=True)
        self.assertEqual(result["state"], UNKNOWN)
        self.assertNotEqual(result["state"], NO_PACKAGE)
        for _ in range(2):
            result = self.observe(right=True)
        self.assertEqual(result["state"], PACKAGE_PRESENT)
        self.assertEqual(result["raw_state"], PACKAGE_PRESENT)
        self.assertGreater(result["qualifying_package_count"], 0)

    def test_reset_discards_old_evidence(self):
        self.observe()
        self.observe()
        self.head.reset()
        self.aggregate.reset()
        self.assertEqual(self.observe()["state"], UNKNOWN)


if __name__ == "__main__":
    unittest.main()
