"""Small, dependency-light rigid transform helpers used by alignment."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def pose_matrix(values: Iterable[float], *, degrees: bool = True) -> np.ndarray:
    """Build ``T_parent_child`` from x/y/z and roll/pitch/yaw."""
    items = tuple(float(item) for item in values)
    if len(items) != 6 or not all(math.isfinite(item) for item in items):
        raise ValueError("pose must contain six finite values")
    x, y, z, roll, pitch, yaw = items
    if degrees:
        roll, pitch, yaw = (math.radians(value) for value in (roll, pitch, yaw))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = (x, y, z)
    return result


def matrix_pose(matrix: np.ndarray, *, degrees: bool = True) -> tuple[float, ...]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    pitch = math.asin(float(np.clip(-value[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(value[2, 1], value[2, 2])
        yaw = math.atan2(value[1, 0], value[0, 0])
    else:
        roll = math.atan2(-value[1, 2], value[1, 1])
        yaw = 0.0
    angles = (roll, pitch, yaw)
    if degrees:
        angles = tuple(math.degrees(item) for item in angles)
    return (*map(float, value[:3, 3]), *angles)


def invert(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ value[:3, 3])
    return result


def wrapped_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def tag_corners(size_m: float) -> np.ndarray:
    half = float(size_m) / 2.0
    # OpenCV IPPE_SQUARE order: top-left, top-right, bottom-right, bottom-left.
    return np.asarray(
        [(-half, half, 0.0), (half, half, 0.0), (half, -half, 0.0), (-half, -half, 0.0)],
        dtype=np.float64,
    )


def project_points(
    points: np.ndarray,
    camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | None = None,
) -> list[list[float]] | None:
    object_points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(camera_from_object, dtype=np.float64)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    homogeneous = np.column_stack((object_points, np.ones(len(object_points))))
    camera_points = (transform @ homogeneous.T).T[:, :3]
    if np.any(camera_points[:, 2] <= 1e-6):
        return None
    coefficients = np.asarray(
        () if distortion is None else distortion, dtype=np.float64
    ).reshape(-1)
    if coefficients.size and np.any(np.abs(coefficients) > 1e-15):
        # Match solvePnP/projectPoints exactly for calibrated lenses. Importing
        # here keeps the transform-only helpers usable without OpenCV.
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to project points with lens distortion"
            ) from error
        rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
        pixels, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            transform[:3, 3],
            matrix,
            coefficients,
        )
        pixels = pixels.reshape(-1, 2)
    else:
        pixels = (matrix @ camera_points.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pixels]
