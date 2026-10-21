from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Frame:
    data: bytes
    mime_type: str
    width: int
    height: int
    captured_at: float
    sequence: int = 0


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    base_from_camera: np.ndarray
    version: str

    def __post_init__(self) -> None:
        if np.asarray(self.camera_matrix).shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        if np.asarray(self.base_from_camera).shape != (4, 4):
            raise ValueError("base_from_camera must be 4x4")


@dataclass(frozen=True)
class Detection:
    tag_id: int
    camera_from_tag: np.ndarray
    corners: tuple[tuple[float, float], ...]
    quality: float
    reprojection_error_px: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
