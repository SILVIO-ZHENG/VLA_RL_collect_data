"""Stateful command shaping for endpoint-driven grippers."""

from __future__ import annotations

import math

import numpy as np


class EffectorEndpointLatch:
    """Convert continuous opening targets into stable open/closed commands.

    Values at or below ``closed_max`` request the fully closed endpoint; values
    at or above ``open_min`` request the fully open endpoint. Values in between
    retain the last command. This dead band prevents a noisy model target from
    repeatedly reversing the physical gripper.
    """

    def __init__(
        self,
        *,
        dimensions: int,
        closed_endpoint: float,
        open_endpoint: float,
        closed_max: float,
        open_min: float,
    ) -> None:
        values = (closed_endpoint, open_endpoint, closed_max, open_min)
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("effector endpoint latch values must be finite")
        if not closed_endpoint <= closed_max < open_min <= open_endpoint:
            raise ValueError(
                "expected closed_endpoint <= closed_max < open_min <= open_endpoint"
            )
        self.dimensions = int(dimensions)
        self.closed_endpoint = float(closed_endpoint)
        self.open_endpoint = float(open_endpoint)
        self.closed_max = float(closed_max)
        self.open_min = float(open_min)
        self._value: np.ndarray | None = None

    def reset(self, reference: np.ndarray) -> None:
        """Re-anchor without moving a gripper that may currently be mid-travel."""
        self._value = self._validate(reference).copy()

    def update(self, target: np.ndarray) -> np.ndarray:
        requested = self._validate(target)
        if self._value is None:
            self.reset(requested)
        assert self._value is not None
        close = requested <= self.closed_max
        opened = requested >= self.open_min
        self._value[close] = self.closed_endpoint
        self._value[opened] = self.open_endpoint
        return self._value.copy()

    def fork(self) -> EffectorEndpointLatch:
        clone = EffectorEndpointLatch(
            dimensions=self.dimensions,
            closed_endpoint=self.closed_endpoint,
            open_endpoint=self.open_endpoint,
            closed_max=self.closed_max,
            open_min=self.open_min,
        )
        if self._value is not None:
            clone._value = self._value.copy()
        return clone

    def _validate(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.dimensions,) or not np.isfinite(array).all():
            raise ValueError(
                f"expected one finite {self.dimensions}D effector target, got {array.shape}"
            )
        if np.any(array < self.closed_endpoint) or np.any(array > self.open_endpoint):
            raise ValueError(
                "effector endpoint latch input is outside the configured physical range"
            )
        return array
