from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nexus_simulator.server import SimulationState


class _Episode:
    timestamps = np.asarray([0.0, 1 / 30], dtype=np.float64)
    effector_encoding = "legacy_revolute"

    def vector(self, frame_index: int, source: str) -> np.ndarray:
        if source not in {"observation", "action"}:
            raise ValueError(source)
        return np.full(24, float(frame_index), dtype=np.float64)

    def to_safety_vector(self, vector: np.ndarray, _source: str) -> np.ndarray:
        return vector.copy()

    def from_safety_vector(self, vector: np.ndarray, _source: str) -> np.ndarray:
        return vector.copy()


class _SafetyReference:
    def evaluate_action(
        self,
        measured: np.ndarray,
        candidate: np.ndarray,
        last_successful: np.ndarray,
        *,
        height: float,
        hard_margin: float,
        enforcement_enabled: bool,
    ) -> tuple[dict[str, object], np.ndarray]:
        del measured, height, hard_margin
        collision = bool(candidate[0] == 1.0)
        adopted = last_successful.copy() if collision and enforcement_enabled else candidate.copy()
        return {
            "collision_detected": collision,
            "enforcement_enabled": enforcement_enabled,
            "enforcement_applied": collision and enforcement_enabled,
            "enforcement_effective": not collision or enforcement_enabled,
            "adopted_pose_safe": not collision or enforcement_enabled,
            "action_disposition": (
                "hold_previous_successful_action"
                if collision and enforcement_enabled
                else "monitor_only_allow_unsafe_candidate"
            ),
            "reason": "fixture_collision" if collision else "safe",
        }, adopted


class _Model:
    def __init__(self) -> None:
        self.last_values: dict[str, float] = {}

    def update_joints(self, values: dict[str, float]) -> list[object]:
        self.last_values = values
        return []


class _Scene:
    def update_robot(self) -> None:
        return

    def set_camera_view(self, _view: str) -> None:
        return


class SimulationStateSafetyTest(unittest.TestCase):
    def test_observation_replay_is_limited_when_enabled_and_passes_when_disabled(self) -> None:
        config = SimpleNamespace(
            work_surface=SimpleNamespace(height=0.75, hard_margin=0.03),
            guard=SimpleNamespace(enabled=True),
        )
        model = _Model()
        with tempfile.TemporaryDirectory() as directory:
            state = SimulationState(
                project_root=Path(directory),
                config=config,
                model=model,
                scene=_Scene(),
                meshcat_port=8767,
                asset_manifest={},
                episode=_Episode(),
                safety_reference=_SafetyReference(),
            )
            state.replay_frame({"frame_index": 0, "source": "observation"})
            limited = state.replay_frame({"frame_index": 1, "source": "observation"})
            self.assertTrue(limited["safety"]["enforcement_applied"])
            self.assertTrue(all(value == 0.0 for value in model.last_values.values()))

            state.update_safety_limit({"enabled": False})
            unrestricted = state.replay_frame({"frame_index": 1, "source": "observation"})
            self.assertFalse(unrestricted["safety"]["enforcement_applied"])
            self.assertTrue(all(value == 1.0 for value in model.last_values.values()))

    def test_camera_presets_are_validated_and_persisted(self) -> None:
        config = SimpleNamespace(
            work_surface=SimpleNamespace(height=0.75, hard_margin=0.03),
            guard=SimpleNamespace(enabled=True),
        )
        camera = {
            "position": [2.1, 0.4, 0.8],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
            "target": [0.2, 0.0, 1.0],
            "fov": 54.0,
            "zoom": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = SimulationState(
                project_root=root,
                config=config,
                model=_Model(),
                scene=_Scene(),
                meshcat_port=8767,
                asset_manifest={},
                episode=_Episode(),
                safety_reference=_SafetyReference(),
            )
            saved = state.save_camera_preset({"view": "overview", "camera": camera})
            self.assertEqual(saved["preset"], camera)
            self.assertTrue((root / "configs/camera_presets.json").is_file())

            restored = SimulationState(
                project_root=root,
                config=config,
                model=_Model(),
                scene=_Scene(),
                meshcat_port=8767,
                asset_manifest={},
                episode=_Episode(),
                safety_reference=_SafetyReference(),
            )
            selected = restored.set_camera_view({"view": "overview"})
            self.assertEqual(selected["preset"], camera)


if __name__ == "__main__":
    unittest.main()
