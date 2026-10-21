from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus_simulator.config import (
    SimulationConfigError,
    load_simulation_config,
    parse_simulation_config,
)


class SimulationConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(
            (PROJECT_ROOT / "configs/g2_work_surface.simulation.json").read_text(encoding="utf-8")
        )

    def test_default_config_is_fail_closed(self) -> None:
        config = load_simulation_config(
            PROJECT_ROOT / "configs/g2_work_surface.simulation.json"
        )
        self.assertTrue(config.simulation_only)
        self.assertFalse(config.calibrated)
        self.assertFalse(config.deployable)
        self.assertEqual(config.work_surface.frame, "base")
        self.assertTrue(config.guard.enabled)
        self.assertEqual(config.guard.collision_response, "hold_previous_safe_action")

    def test_rejects_deployable_simulation_config(self) -> None:
        self.payload["deployable"] = True
        with self.assertRaisesRegex(SimulationConfigError, "deployable 必须为 false"):
            parse_simulation_config(self.payload)

    def test_rejects_calibrated_simulation_config(self) -> None:
        self.payload["calibrated"] = True
        with self.assertRaisesRegex(SimulationConfigError, "calibrated 必须为 false"):
            parse_simulation_config(self.payload)

    def test_rejects_invalid_surface_bounds(self) -> None:
        self.payload["work_surface"]["xy_bounds"] = [1.0, 0.0, -1.0, 1.0]
        with self.assertRaisesRegex(SimulationConfigError, "xmin<xmax"):
            parse_simulation_config(self.payload)

    def test_rejects_non_boolean_guard_enabled(self) -> None:
        self.payload["guard"]["enabled"] = "yes"
        with self.assertRaisesRegex(SimulationConfigError, "guard.enabled 必须是布尔值"):
            parse_simulation_config(self.payload)


if __name__ == "__main__":
    unittest.main()
