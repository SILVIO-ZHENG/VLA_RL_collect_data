"""In-memory robot adapter used for end-to-end tests and integration examples."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from ..contracts import Observation, TensorSpec
from ..plugins import RobotAdapter


class MockRobotAdapter(RobotAdapter):
    def __init__(self, config: Mapping[str, Any]) -> None:
        names = tuple(str(item) for item in config.get("joint_names", ("joint_1", "joint_2")))
        self._state_spec = TensorSpec(names)
        self._action_spec = TensorSpec(names)
        self._camera_names = tuple(str(item) for item in config.get("camera_names", ("front",)))
        self._state = np.zeros(len(names), dtype=np.float32)
        self._connected = False

    @property
    def state_spec(self) -> TensorSpec:
        return self._state_spec

    @property
    def action_spec(self) -> TensorSpec:
        return self._action_spec

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._camera_names

    def connect(self) -> None:
        self._connected = True

    def observe(self, task: str) -> Observation:
        if not self._connected:
            raise RuntimeError("robot is not connected")
        pixel = b"\x00\x00\x00"
        images = {
            name: {
                "encoding": "raw",
                "color_format": "rgb",
                "width": 1,
                "height": 1,
                "data": pixel,
            }
            for name in self.camera_names
        }
        return Observation(self._state.copy(), images, task, time.time_ns())

    def execute(self, action: Any) -> None:
        if not self._connected:
            raise RuntimeError("robot is not connected")
        self._state = self.action_spec.validate(action).copy()

    def close(self) -> None:
        self._connected = False
