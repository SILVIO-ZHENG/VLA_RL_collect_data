"""Multi-pose calibration of a head-mounted camera against one fixed AprilTag."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .geometry import invert, tag_corners


@dataclass(frozen=True)
class ExtrinsicObservation:
    label: str
    role: str
    base_from_head: np.ndarray
    corners_px: np.ndarray
    camera_from_tag_pnp: np.ndarray

    def __post_init__(self) -> None:
        if self.role not in {"fit", "holdout"}:
            raise ValueError("observation role must be fit or holdout")
        if np.asarray(self.base_from_head).shape != (4, 4):
            raise ValueError("base_from_head must be 4x4")
        if np.asarray(self.corners_px).shape != (4, 2):
            raise ValueError("corners_px must be 4x2")
        if np.asarray(self.camera_from_tag_pnp).shape != (4, 4):
            raise ValueError("camera_from_tag_pnp must be 4x4")


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Install head-camera calibration dependencies with: "
            "uv sync --extra alignment"
        ) from error
    return cv2


def rotation_vector(transform: np.ndarray) -> np.ndarray:
    vector, _ = _cv2().Rodrigues(np.asarray(transform, dtype=np.float64)[:3, :3])
    return vector.reshape(3)


def transform_from_vector(vector: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(vector), dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("transform vector must contain six finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3], _ = _cv2().Rodrigues(values[:3])
    result[:3, 3] = values[3:]
    return result


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one transform is required")
    values = [np.asarray(value, dtype=np.float64) for value in transforms]
    if any(value.shape != (4, 4) or not np.isfinite(value).all() for value in values):
        raise ValueError("all transforms must be finite 4x4 matrices")
    rotation_sum = sum((value[:3, :3] for value in values), np.zeros((3, 3)))
    left, _singular, right = np.linalg.svd(rotation_sum)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.median(
        np.stack([value[:3, 3] for value in values]), axis=0
    )
    return result


def solve_pnp_square(
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> tuple[np.ndarray, float]:
    cv2 = _cv2()
    success, rotation_vector_value, translation_vector = cv2.solvePnP(
        tag_corners(tag_size_m),
        np.asarray(corners_px, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("solvePnP failed for captured AprilTag corners")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3], _ = cv2.Rodrigues(rotation_vector_value)
    result[:3, 3] = translation_vector.reshape(3)
    pixels, _ = cv2.projectPoints(
        tag_corners(tag_size_m),
        rotation_vector_value,
        translation_vector,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    residual = pixels.reshape(4, 2) - np.asarray(corners_px, dtype=np.float64)
    return result, float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def project_tag(
    base_from_head: np.ndarray,
    head_from_camera: np.ndarray,
    base_from_tag: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> np.ndarray:
    cv2 = _cv2()
    camera_from_tag = invert(
        np.asarray(base_from_head, dtype=np.float64)
        @ np.asarray(head_from_camera, dtype=np.float64)
    ) @ np.asarray(base_from_tag, dtype=np.float64)
    if camera_from_tag[2, 3] <= 1e-6:
        return np.full((4, 2), 1e6, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(camera_from_tag[:3, :3])
    pixels, _ = cv2.projectPoints(
        tag_corners(tag_size_m),
        rvec,
        camera_from_tag[:3, 3],
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    return pixels.reshape(4, 2)


def motion_excitation(observations: Sequence[ExtrinsicObservation]) -> dict[str, Any]:
    if len(observations) < 2:
        return {"singular_values_deg": [0.0, 0.0, 0.0], "sufficient": False}
    reference = np.asarray(observations[0].base_from_head, dtype=np.float64)
    vectors = np.stack(
        [rotation_vector(invert(reference) @ item.base_from_head) for item in observations[1:]]
    )
    singular = np.linalg.svd(vectors, compute_uv=False)
    singular = np.pad(singular, (0, 3 - len(singular)))[:3]
    singular_degrees = [float(math.degrees(value)) for value in singular]
    return {
        "singular_values_deg": singular_degrees,
        # A fixed camera/tag pair cannot separate all parameters when every
        # head pose lies on effectively one rotation axis.
        "sufficient": bool(len(observations) >= 6 and singular_degrees[1] >= 3.0),
    }


def _pose_metrics(
    observations: Sequence[ExtrinsicObservation],
    head_from_camera: np.ndarray,
    base_from_tag: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in observations:
        predicted = project_tag(
            item.base_from_head,
            head_from_camera,
            base_from_tag,
            camera_matrix,
            distortion,
            tag_size_m,
        )
        error = predicted - item.corners_px
        results.append(
            {
                "label": item.label,
                "role": item.role,
                "reprojection_rmse_px": float(
                    np.sqrt(np.mean(np.sum(error * error, axis=1)))
                ),
                "maximum_corner_error_px": float(
                    np.max(np.linalg.norm(error, axis=1))
                ),
            }
        )
    return results


def _tag_consistency(
    observations: Sequence[ExtrinsicObservation],
    head_from_camera: np.ndarray,
) -> dict[str, float]:
    estimates = [
        item.base_from_head @ head_from_camera @ item.camera_from_tag_pnp
        for item in observations
    ]
    center = average_transforms(estimates)
    translation_errors = np.asarray(
        [np.linalg.norm(value[:3, 3] - center[:3, 3]) for value in estimates]
    )
    normal = center[:3, 2]
    angles = np.asarray(
        [
            math.degrees(
                math.acos(float(np.clip(np.dot(value[:3, 2], normal), -1.0, 1.0)))
            )
            for value in estimates
        ]
    )
    return {
        "translation_rmse_mm": float(np.sqrt(np.mean(translation_errors**2)) * 1000),
        "translation_max_mm": float(np.max(translation_errors) * 1000),
        "normal_rmse_deg": float(np.sqrt(np.mean(angles**2))),
        "normal_max_deg": float(np.max(angles)),
    }


def solve_head_camera_extrinsic(
    observations: Sequence[ExtrinsicObservation],
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
    factory_head_from_camera: np.ndarray,
) -> dict[str, Any]:
    """Jointly solve ``head_from_camera`` and one fixed ``base_from_tag``."""
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError(
            "Install head-camera calibration dependencies with: "
            "uv sync --extra alignment"
        ) from error

    fit = [item for item in observations if item.role == "fit"]
    holdout = [item for item in observations if item.role == "holdout"]
    if len(fit) < 6:
        raise ValueError("at least six fit poses are required")
    if len(holdout) < 2:
        raise ValueError("at least two independent holdout poses are required")
    excitation = motion_excitation(fit)
    if not excitation["sufficient"]:
        raise ValueError(
            "head-pose excitation is insufficient; vary both head yaw and pitch"
        )

    factory = np.asarray(factory_head_from_camera, dtype=np.float64)
    initial_tag = average_transforms(
        [item.base_from_head @ factory @ item.camera_from_tag_pnp for item in fit]
    )

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            factory @ transform_from_vector(parameters[:6]),
            initial_tag @ transform_from_vector(parameters[6:]),
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        head_from_camera, base_from_tag = unpack(parameters)
        return np.concatenate(
            [
                (
                    project_tag(
                        item.base_from_head,
                        head_from_camera,
                        base_from_tag,
                        camera_matrix,
                        distortion,
                        tag_size_m,
                    )
                    - item.corners_px
                ).reshape(-1)
                for item in fit
            ]
        )

    lower = np.asarray([-0.35] * 3 + [-0.15] * 3 + [-0.8] * 3 + [-0.5] * 3)
    upper = -lower
    solved = least_squares(
        residual,
        np.zeros(12, dtype=np.float64),
        bounds=(lower, upper),
        loss="huber",
        f_scale=0.5,
        x_scale="jac",
        max_nfev=3000,
    )
    if not solved.success:
        raise RuntimeError(f"extrinsic optimization failed: {solved.message}")
    calibrated, base_from_tag = unpack(solved.x)
    metrics = _pose_metrics(
        observations,
        calibrated,
        base_from_tag,
        camera_matrix,
        distortion,
        tag_size_m,
    )
    factory_metrics = _pose_metrics(
        observations,
        factory,
        initial_tag,
        camera_matrix,
        distortion,
        tag_size_m,
    )

    rotation_delta = rotation_vector(invert(factory) @ calibrated)
    translation_delta = (invert(factory) @ calibrated)[:3, 3]
    fit_errors = [item["reprojection_rmse_px"] for item in metrics if item["role"] == "fit"]
    holdout_errors = [
        item["reprojection_rmse_px"] for item in metrics if item["role"] == "holdout"
    ]
    return {
        "success": True,
        "optimizer": {
            "method": "scipy.least_squares",
            "loss": "huber",
            "f_scale_px": 0.5,
            "function_evaluations": int(solved.nfev),
            "cost": float(solved.cost),
            "optimality": float(solved.optimality),
        },
        "motion_excitation": excitation,
        "factory_head_link3_from_camera": factory.tolist(),
        "calibrated_head_link3_from_camera": calibrated.tolist(),
        "base_from_tag": base_from_tag.tolist(),
        "change_from_factory": {
            "rotation_vector_deg": [float(math.degrees(value)) for value in rotation_delta],
            "rotation_magnitude_deg": float(math.degrees(np.linalg.norm(rotation_delta))),
            "translation_mm": [float(value * 1000) for value in translation_delta],
            "translation_magnitude_mm": float(np.linalg.norm(translation_delta) * 1000),
        },
        "reprojection": {
            "fit_rmse_px": float(np.sqrt(np.mean(np.square(fit_errors)))),
            "holdout_rmse_px": float(np.sqrt(np.mean(np.square(holdout_errors)))),
            "per_pose": metrics,
            "factory_per_pose": factory_metrics,
        },
        "tag_pose_consistency": {
            "factory": _tag_consistency(observations, factory),
            "calibrated": _tag_consistency(observations, calibrated),
        },
        "activation": {
            "active_extrinsic_replaced": False,
            "requires_holdout_review": True,
            "requires_contact_plane_validation": True,
        },
    }
