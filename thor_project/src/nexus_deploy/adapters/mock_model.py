"""Deterministic model adapter used for protocol and deployment smoke tests."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..contracts import ActionChunk, ImageSpec, InferenceRequest, PolicyCapabilities, TensorSpec
from ..plugins import ModelAdapter


class MockModelAdapter(ModelAdapter):
    def __init__(self, config: Mapping[str, Any]) -> None:
        names = tuple(str(item) for item in config.get("joint_names", ("joint_1", "joint_2")))
        self._capabilities = PolicyCapabilities(
            model_id=str(config.get("model_id", "mock-vla")),
            state=TensorSpec(names),
            action=TensorSpec(names),
            images=tuple(
                ImageSpec(str(name), required=True) for name in config.get("camera_names", ("front",))
            ),
            chunk_size=int(config.get("chunk_size", 8)),
            model_fps=float(config.get("model_fps", 10)),
            modes=("sync", "async"),
            extra={"mock": True},
        )

    @property
    def capabilities(self) -> PolicyCapabilities:
        return self._capabilities

    def infer(self, request: InferenceRequest) -> ActionChunk:
        # A hold trajectory is intentionally safe and makes schema/order bugs obvious.
        actions = np.repeat(
            request.observation.state[None, :], self.capabilities.chunk_size, axis=0
        ).astype(np.float32)
        return ActionChunk(actions, f"{request.request_id}-chunk", {"mock": True})
