"""Tests for the four-sample aggregate parcel-presence gate.

These tests contain no GDK, CUDA, or robot-control calls.  They exercise only
the business gate derived from synthetic YOLO instance masks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARCEL_ROOT = PROJECT_ROOT / "src" / "nexus_deploy" / "vision" / "parcel"

from nexus_deploy.vision.parcel.runtime import (  # noqa: E402
    ParcelStateEstimator,
    RateLimitedParcelStateService,
    RuntimeConfig,
)
from nexus_deploy.vision.parcel.runtime.core_bridge import Instance  # noqa: E402
from nexus_deploy.vision.parcel.runtime.presence import (  # noqa: E402
    NO_PACKAGE,
    PACKAGE_PRESENT,
    UNKNOWN,
    ParcelPresenceGate,
    PresenceGateConfig,
)


SHAPE = (40, 40)


def rectangle(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def package(mask: np.ndarray) -> Instance:
    return Instance(class_name="package", confidence=0.90, mask=mask)


def left_gripper(mask: np.ndarray) -> Instance:
    return Instance(class_name="left_gripper", confidence=0.95, mask=mask)


def make_gate() -> ParcelPresenceGate:
    config = PresenceGateConfig(
        include_roi_names=("work",),
        exclude_roi_names=("conveyor",),
        polygons={
            "work": ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)),
            "conveyor": ((0.75, 0.0), (1.0, 0.0), (1.0, 1.0), (0.75, 1.0)),
        },
        roi_membership_min=0.55,
        min_mask_pixels=4,
        gripper_contact_distance_ratio=0.025,
        window_frames=4,
    )
    config.validate()
    return ParcelPresenceGate(config)


class SequenceDetector:
    """Counts calls and returns a controlled sequence of instance lists."""

    model_sha256 = "presence-test-detector"

    def __init__(self, sequence: list[list[Instance]]):
        self.sequence = sequence
        self.calls = 0
        self.closed = False

    def predict(self, frame_bgr: np.ndarray):
        del frame_bgr
        result = self.sequence[min(self.calls, len(self.sequence) - 1)]
        self.calls += 1
        return result

    def warmup(self, image_shape):
        del image_shape

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value


class PresenceGateTests(unittest.TestCase):
    def test_four_present_samples_emit_package_present(self):
        gate = make_gate()
        inside_work_area = [package(rectangle(5, 10, 13, 18))]

        results = [gate.update(inside_work_area, SHAPE) for _ in range(4)]

        self.assertEqual([item.state for item in results[:3]], [UNKNOWN] * 3)
        self.assertEqual(results[-1].state, PACKAGE_PRESENT)
        self.assertEqual(
            results[-1].window_states,
            (PACKAGE_PRESENT, PACKAGE_PRESENT, PACKAGE_PRESENT, PACKAGE_PRESENT),
        )

    def test_four_absent_samples_emit_no_package(self):
        gate = make_gate()

        results = [gate.update([], SHAPE) for _ in range(4)]

        self.assertEqual([item.state for item in results[:3]], [UNKNOWN] * 3)
        self.assertEqual(results[-1].state, NO_PACKAGE)
        self.assertEqual(
            results[-1].window_states,
            (NO_PACKAGE, NO_PACKAGE, NO_PACKAGE, NO_PACKAGE),
        )

    def test_gripper_without_parcel_does_not_block_empty_confirmation(self):
        gate = make_gate()
        gate.update([], SHAPE)
        gate.update([], SHAPE)
        covering_gripper = [left_gripper(rectangle(5, 10, 13, 18))]

        observed = gate.update(covering_gripper, SHAPE)
        self.assertEqual(observed.raw_state, NO_PACKAGE)
        self.assertEqual(observed.state, UNKNOWN)
        self.assertEqual(gate.update([], SHAPE).state, NO_PACKAGE)

    def test_gripper_outside_work_area_does_not_block_empty_confirmation(self):
        gate = make_gate()
        outside_gripper = [left_gripper(rectangle(25, 10, 33, 18))]
        results = [gate.update(outside_gripper, SHAPE) for _ in range(4)]
        self.assertEqual(results[-1].state, NO_PACKAGE)

    def test_mixed_four_sample_window_is_unknown(self):
        gate = make_gate()
        inside_work_area = [package(rectangle(5, 10, 13, 18))]
        for _ in range(4):
            gate.update(inside_work_area, SHAPE)

        mixed = gate.update([], SHAPE)

        self.assertEqual(mixed.state, UNKNOWN)
        self.assertEqual(
            mixed.window_states,
            (PACKAGE_PRESENT, PACKAGE_PRESENT, PACKAGE_PRESENT, NO_PACKAGE),
        )
        self.assertIn("both PACKAGE_PRESENT and NO_PACKAGE", mixed.reason)

    def test_package_outside_work_area_is_ignored(self):
        gate = make_gate()
        on_conveyor = [package(rectangle(32, 10, 38, 18))]

        results = [gate.update(on_conveyor, SHAPE) for _ in range(4)]

        self.assertEqual(results[-1].state, NO_PACKAGE)
        self.assertEqual(results[-1].ignored_package_count, 1)

    def test_conveyor_exclusion_wins_when_it_overlaps_a_table_roi(self):
        # A broad table ROI can overlap the visible conveyor in a real camera
        # view.  The calibrated conveyor exclusion must take priority.
        config = PresenceGateConfig(
            include_roi_names=("table",),
            exclude_roi_names=("conveyor",),
            polygons={
                "table": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                "conveyor": ((0.75, 0.0), (1.0, 0.0), (1.0, 1.0), (0.75, 1.0)),
            },
            roi_membership_min=0.55,
            min_mask_pixels=4,
            window_frames=4,
        )
        gate = ParcelPresenceGate(config)
        on_conveyor = [package(rectangle(32, 10, 38, 18))]

        results = [gate.update(on_conveyor, SHAPE) for _ in range(4)]

        self.assertEqual(results[-1].state, NO_PACKAGE)
        self.assertEqual(results[-1].ignored_package_count, 1)

    def test_package_touching_gripper_counts_even_outside_work_area(self):
        gate = make_gate()
        outside_work_area = package(rectangle(24, 10, 31, 18))
        touching_gripper = left_gripper(rectangle(29, 10, 35, 18))

        results = [
            gate.update([outside_work_area, touching_gripper], SHAPE)
            for _ in range(4)
        ]

        self.assertEqual(results[-1].state, PACKAGE_PRESENT)
        self.assertEqual(results[-1].qualifying_package_count, 1)

    def test_camera_or_model_fault_clears_visual_window(self):
        gate = make_gate()
        inside_work_area = [package(rectangle(5, 10, 13, 18))]
        gate.update(inside_work_area, SHAPE)
        gate.update(inside_work_area, SHAPE)

        fault = gate.mark_unknown("camera frame is stale")
        restarted = [gate.update(inside_work_area, SHAPE) for _ in range(4)]

        self.assertEqual(fault.state, UNKNOWN)
        self.assertEqual([item.state for item in restarted[:3]], [UNKNOWN] * 3)
        self.assertEqual(restarted[-1].state, PACKAGE_PRESENT)

    def test_estimator_exposes_presence_from_the_same_yolo_call(self):
        # Coordinates are inside the deployment's calibrated table_left ROI.
        mask = np.zeros((400, 640), dtype=bool)
        mask[200:250, 300:350] = True
        detector = SequenceDetector([[Instance("package", 0.90, mask)]])
        estimator = ParcelStateEstimator(
            RuntimeConfig(
                rules=PARCEL_ROOT / "configs" / "g2_head.yaml",
                weights=PROJECT_ROOT / "models" / "vision" / "parcel" / "yolo26m-seg-final.pt",
            ),
            detector=detector,
        )
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        try:
            results = [
                estimator.process_frame_with_presence(
                    frame,
                    color_space="RGB",
                    frame_index=index * 3,
                    capture_timestamp_ns=index + 1,
                )
                for index in range(4)
            ]
        finally:
            estimator.close()

        self.assertEqual(detector.calls, 4)
        self.assertEqual(results[-1].presence.state, PACKAGE_PRESENT)
        self.assertEqual(results[-1].stage.state, "ON_TABLE_LEFT")

    def test_rate_limited_presence_api_does_not_run_a_second_inference(self):
        mask = np.zeros((400, 640), dtype=bool)
        mask[200:250, 300:350] = True
        detector = SequenceDetector([[Instance("package", 0.90, mask)]])
        estimator = ParcelStateEstimator(
            RuntimeConfig(
                rules=PARCEL_ROOT / "configs" / "g2_head.yaml",
                weights=PROJECT_ROOT / "models" / "vision" / "parcel" / "yolo26m-seg-final.pt",
            ),
            detector=detector,
        )
        clock = FakeClock()
        service = RateLimitedParcelStateService(estimator, clock_ns=clock)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        try:
            results = []
            for index in range(4):
                clock.value = index * 100_000_000
                results.append(
                    service.process_with_presence_if_due(
                        frame,
                        color_space="RGB",
                        frame_index=index * 3,
                        capture_timestamp_ns=index + 1,
                    )
                )
            clock.value = 350_000_000
            skipped = service.process_with_presence_if_due(
                frame,
                color_space="RGB",
                frame_index=12,
                capture_timestamp_ns=5,
            )
        finally:
            estimator.close()

        self.assertTrue(all(item is not None for item in results))
        self.assertEqual(detector.calls, 4)
        self.assertEqual(results[-1].presence.state, PACKAGE_PRESENT)
        self.assertIsNone(skipped)
        self.assertEqual(service.latest_presence.state, PACKAGE_PRESENT)


if __name__ == "__main__":
    unittest.main()
