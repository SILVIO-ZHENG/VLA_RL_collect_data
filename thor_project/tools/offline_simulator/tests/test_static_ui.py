from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StaticSafetyUiTest(unittest.TestCase):
    def test_successful_enforcement_has_a_viewer_level_notice(self) -> None:
        app = (PROJECT_ROOT / "src/nexus_simulator/static/app.js").read_text(encoding="utf-8")
        styles = (PROJECT_ROOT / "src/nexus_simulator/static/styles.css").read_text(
            encoding="utf-8"
        )
        self.assertIn("安全限制已生效 · 危险候选已拦截", app)
        self.assertIn('"has-enforced-collision"', app)
        self.assertIn(".viewer-panel.has-enforced-collision .viewer-wrap", styles)
        self.assertIn(".collision-alert.is-enforced", styles)

    def test_embeddable_viewer_bridges_camera_state_to_platform(self) -> None:
        viewer = (PROJECT_ROOT / "src/nexus_simulator/static/viewer.html").read_text(
            encoding="utf-8"
        )
        bridge = (PROJECT_ROOT / "src/nexus_simulator/static/viewer.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="meshcat-pane"', viewer)
        self.assertIn("nexus-camera-capture", bridge)
        self.assertIn("nexus-camera-apply", bridge)
        self.assertIn("viewer.controls.target", bridge)
        self.assertIn("window.parent.postMessage", bridge)


if __name__ == "__main__":
    unittest.main()
