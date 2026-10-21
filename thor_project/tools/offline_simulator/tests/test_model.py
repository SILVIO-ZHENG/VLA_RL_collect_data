from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus_simulator.model import G2Model, ModelError, joint_group


@unittest.skipUnless(
    (PROJECT_ROOT / "assets/g2/G2_omnipicker/G2_omnipicker_visual.urdf").is_file(),
    "G2 asset snapshot not synced",
)
class G2ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = G2Model(
            PROJECT_ROOT / "assets/g2/G2_omnipicker/G2_omnipicker_visual.urdf"
        )

    def test_model_summary_and_joint_groups(self) -> None:
        summary = self.model.summary()
        self.assertEqual(summary["name"], "G2_omnipicker_visual")
        self.assertEqual(summary["actuated_joint_count"], 32)
        self.assertEqual(summary["controllable_joint_count"], 28)
        self.assertEqual(summary["geometry_count"], 94)
        self.assertEqual(summary["visual_mesh_count"], 51)
        self.assertGreater(summary["face_count"], 500_000)
        self.assertEqual(joint_group("idx21_arm_l_joint1"), "左臂")

    def test_joint_update_changes_fk_and_enforces_limits(self) -> None:
        before = self.model.geometry_transforms()
        self.model.update_joints({"idx22_arm_l_joint2": 0.5})
        after = self.model.geometry_transforms()
        self.assertTrue(any((before[name] != after[name]).any() for name in before))
        with self.assertRaisesRegex(ModelError, "越界"):
            self.model.update_joints({"idx22_arm_l_joint2": 99.0})
        self.model.reset()

    def test_embedded_dae_material_colors_are_preserved(self) -> None:
        colors = {record[4] for record in self.model.geometry_records()}
        self.assertGreaterEqual(len(colors), 7)
        self.assertIn(0xFA6E00, colors)

    def test_geometry_is_grouped_under_links_for_efficient_playback(self) -> None:
        records = self.model.linked_geometry_records()
        links = {record[1] for record in records}
        self.assertEqual(len(records), 94)
        self.assertEqual(len(links), 51)
        self.assertEqual(set(self.model.link_transforms(links)), links)


if __name__ == "__main__":
    unittest.main()
