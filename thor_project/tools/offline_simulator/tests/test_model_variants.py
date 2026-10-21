from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus_simulator.dataset import NATIVE_DH_EFFECTOR_ENCODING
from nexus_simulator.model import G2Model
from nexus_simulator.model_variants import (
    G2_DAHUAN_MODEL_ID,
    LEGACY_GRIPPER_LOWER_RAD,
    LEGACY_MODEL_ID,
    PRISMATIC_GRIPPER_UPPER_M,
    ModelVariantError,
    model_variant_spec,
    resolve_model_variant,
)


class ModelVariantMappingTest(unittest.TestCase):
    def test_legacy_mapping_is_unchanged(self) -> None:
        values = {"idx31_gripper_l_inner_joint1": -0.5}
        mapped = model_variant_spec(LEGACY_MODEL_ID).map_replay_joint_values(values)
        self.assertEqual(mapped, values)
        self.assertIsNot(mapped, values)

    def test_new_gripper_maps_closed_and_open_positions_symmetrically(self) -> None:
        spec = model_variant_spec(G2_DAHUAN_MODEL_ID)
        closed = spec.map_replay_joint_values(
            {
                "idx31_gripper_l_inner_joint1": LEGACY_GRIPPER_LOWER_RAD,
                "idx71_gripper_r_inner_joint1": LEGACY_GRIPPER_LOWER_RAD,
            }
        )
        opened = spec.map_replay_joint_values(
            {
                "idx31_gripper_l_inner_joint1": 0.0,
                "idx71_gripper_r_inner_joint1": 0.0,
            }
        )
        for inner, outer in (
            ("idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"),
            ("idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"),
        ):
            self.assertAlmostEqual(closed[inner], PRISMATIC_GRIPPER_UPPER_M)
            self.assertAlmostEqual(opened[inner], 0.0)
            self.assertNotIn(outer, closed)
            self.assertNotIn(outer, opened)

    def test_new_gripper_rejects_values_outside_legacy_protocol_range(self) -> None:
        spec = model_variant_spec(G2_DAHUAN_MODEL_ID)
        with self.assertRaisesRegex(ModelVariantError, "越出旧协议范围"):
            spec.map_replay_joint_values(
                {
                    "idx31_gripper_l_inner_joint1": 0.1,
                    "idx71_gripper_r_inner_joint1": 0.0,
                }
            )

    def test_native_dh_observation_and_action_have_opposite_polarity(self) -> None:
        spec = model_variant_spec(G2_DAHUAN_MODEL_ID)
        values = {
            "idx31_gripper_l_inner_joint1": 0.005,
            "idx71_gripper_r_inner_joint1": 0.020,
        }
        observation = spec.map_replay_joint_values(
            values,
            effector_encoding=NATIVE_DH_EFFECTOR_ENCODING,
            source="observation",
        )
        action = spec.map_replay_joint_values(
            values,
            effector_encoding=NATIVE_DH_EFFECTOR_ENCODING,
            source="action",
        )
        self.assertAlmostEqual(observation["idx31_gripper_l_inner_joint1"], 0.020)
        self.assertAlmostEqual(action["idx31_gripper_l_inner_joint1"], 0.005)
        self.assertNotIn("idx41_gripper_l_outer_joint1", observation)
        self.assertNotIn("idx41_gripper_l_outer_joint1", action)
        self.assertAlmostEqual(observation["idx71_gripper_r_inner_joint1"], 0.005)
        self.assertAlmostEqual(action["idx71_gripper_r_inner_joint1"], 0.020)


@unittest.skipUnless(
    (
        PROJECT_ROOT.parents[1]
        / "src/nexus_deploy/viewer/assets/g2/G2_dahuan/G2_dahuan.urdf"
    ).is_file(),
    "new G2 model snapshot not imported",
)
class ImportedModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = resolve_model_variant(PROJECT_ROOT, G2_DAHUAN_MODEL_ID)
        cls.model = G2Model(
            cls.spec.urdf_path(PROJECT_ROOT),
            asset_root=cls.spec.asset_root(PROJECT_ROOT),
        )

    def test_imported_model_loads_with_expected_geometry(self) -> None:
        summary = self.model.summary()
        self.assertEqual(summary["name"], "G2_dahuan")
        self.assertEqual(summary["base_link"], "base_link")
        self.assertEqual(summary["controllable_joint_count"], 32)
        self.assertEqual(summary["visual_mesh_count"], 37)
        np.testing.assert_allclose(
            self.model.link_transforms({"base_link"})["base_link"][:3, 3],
            np.zeros(3),
            atol=1e-12,
        )

    def test_prismatic_gripper_limits_and_units_are_exposed(self) -> None:
        joints = {joint.name: joint for joint in self.model.joint_infos()}
        for name in (
            "idx31_gripper_l_inner_joint1",
            "idx71_gripper_r_inner_joint1",
        ):
            self.assertEqual(joints[name].unit, "m")
            self.assertAlmostEqual(joints[name].lower, 0.0)
            self.assertAlmostEqual(joints[name].upper, PRISMATIC_GRIPPER_UPPER_M)
        self.assertNotIn("idx41_gripper_l_outer_joint1", joints)
        self.assertNotIn("idx81_gripper_r_outer_joint1", joints)

    def test_positive_prismatic_motion_visually_closes_fingers(self) -> None:
        records = {
            link: (vertices, local)
            for _, link, vertices, _, local, _ in self.model.linked_geometry_records()
            if link in {"gripper_l_inner_link1", "gripper_l_outer_link1"}
        }

        def visual_center_gap(position: float) -> float:
            self.model.update_joints(
                {
                    "idx31_gripper_l_inner_joint1": position,
                }
            )
            links = set(records)
            transforms = self.model.link_transforms(links)
            centers = []
            for link, (vertices, local) in records.items():
                transform = transforms[link] @ local
                center = np.mean(vertices, axis=0) @ transform[:3, :3].T + transform[:3, 3]
                centers.append(center)
            return float(np.linalg.norm(centers[0] - centers[1]))

        open_gap = visual_center_gap(0.0)
        closed_gap = visual_center_gap(0.025)
        self.assertLess(closed_gap, open_gap)
        self.model.reset()


if __name__ == "__main__":
    unittest.main()
