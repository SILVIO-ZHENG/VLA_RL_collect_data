from __future__ import annotations

import unittest
from threading import RLock

import numpy as np

from nexus_simulator.meshcat_scene import (
    NEAR_CAMERA_POSITION,
    NEAR_CAMERA_TARGET,
    OVERVIEW_CAMERA_POSITION,
    OVERVIEW_SCENE_Z_OFFSET,
    WORK_SURFACE_VISUAL_GAP,
    MeshcatScene,
    changed_link_transforms,
    work_surface_layout,
)


class WorkSurfaceLayoutTest(unittest.TestCase):
    def test_workspace_camera_is_close_and_orbits_robot_upper_body(self) -> None:
        self.assertGreater(NEAR_CAMERA_POSITION[0], 1.0)
        self.assertLess(NEAR_CAMERA_POSITION[0], 2.0)
        self.assertEqual(NEAR_CAMERA_TARGET[1], 0.0)
        self.assertGreater(NEAR_CAMERA_TARGET[2], 0.8)
        self.assertGreater(OVERVIEW_CAMERA_POSITION[0], NEAR_CAMERA_POSITION[0])
        self.assertLess(OVERVIEW_SCENE_Z_OFFSET, 0.0)

    def test_surface_is_centered_in_front_of_g2(self) -> None:
        table_size, table_center, margin_size, margin_center = work_surface_layout(
            height=0.75, hard_margin=0.03
        )
        self.assertEqual(table_size, (0.7, 1.6, 0.04))
        self.assertEqual(margin_size, (0.7, 1.6, 0.03))
        self.assertGreater(table_center[0], 0.0)
        self.assertEqual(table_center[1], 0.0)
        self.assertAlmostEqual(table_center[0] - table_size[0] * 0.5, 0.3)
        self.assertEqual(margin_center[:2], table_center[:2])
        self.assertAlmostEqual(table_center[2] + table_size[2] * 0.5, 0.75)
        self.assertAlmostEqual(
            margin_center[2] - margin_size[2] * 0.5,
            0.75 + WORK_SURFACE_VISUAL_GAP,
        )

    def test_only_changed_link_transforms_are_broadcast(self) -> None:
        identity = np.eye(4)
        moved = identity.copy()
        moved[0, 3] = 0.1
        changed = changed_link_transforms(
            {"same": identity.copy(), "moved": moved, "new": identity.copy()},
            {"same": identity.copy(), "moved": identity.copy()},
        )
        self.assertEqual(set(changed), {"moved", "new"})

    def test_robot_updates_persist_authoritative_link_transforms(self) -> None:
        identity = np.eye(4)
        moved = identity.copy()
        moved[0, 3] = 0.1
        calls: list[tuple[str, np.ndarray]] = []

        class FakeNode:
            def __init__(self, path: str) -> None:
                self.path = path

            def set_transform(self, transform: np.ndarray) -> None:
                calls.append((self.path, transform.copy()))

        class FakeViewer:
            def __getitem__(self, path: str) -> FakeNode:
                return FakeNode(path)

        class FakeModel:
            def link_transforms(self, _names: set[str]) -> dict[str, np.ndarray]:
                return {"same": identity.copy(), "moved": moved.copy()}

        scene = MeshcatScene.__new__(MeshcatScene)
        scene._lock = RLock()
        scene.model = FakeModel()
        scene.viewer = FakeViewer()
        scene._link_paths = {"same": "robot/same", "moved": "robot/moved"}
        scene._last_link_transforms = {
            "same": identity.copy(),
            "moved": identity.copy(),
        }

        self.assertEqual(scene.update_robot(), 1)
        self.assertEqual([path for path, _ in calls], ["robot/moved"])
        np.testing.assert_array_equal(calls[0][1], moved)
        np.testing.assert_array_equal(scene._last_link_transforms["moved"], moved)


if __name__ == "__main__":
    unittest.main()
