"""AprilTag detector plugins."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from .config import AlignmentProfile
from .geometry import invert, matrix_pose, pose_matrix, project_points, tag_corners
from .types import CameraCalibration, Detection, Frame


class AprilTagDetector(ABC):
    @abstractmethod
    def detect(
        self, frame: Frame, calibration: CameraCalibration, profile: AlignmentProfile
    ) -> tuple[Detection, ...]: ...


class OpenCVAprilTagDetector(AprilTagDetector):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Install alignment dependencies: pip install -e '.[alignment]'") from error
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module is required for AprilTag detection")
        self.cv2 = cv2

    def _parameters(self) -> Any:
        cv2 = self.cv2
        parameters = cv2.aruco.DetectorParameters()
        refinement = str(self.config.get("corner_refinement", "subpix")).strip().lower()
        if refinement == "subpix":
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            parameters.cornerRefinementWinSize = max(
                1, min(20, int(self.config.get("corner_refinement_window_px", 5)))
            )
            parameters.cornerRefinementMaxIterations = max(
                1, min(200, int(self.config.get("corner_refinement_max_iterations", 30)))
            )
            parameters.cornerRefinementMinAccuracy = max(
                1e-6, float(self.config.get("corner_refinement_min_accuracy", 0.01))
            )
        elif refinement not in {"none", "off"}:
            raise RuntimeError(f"unsupported AprilTag corner refinement: {refinement}")
        return parameters

    def detect(
        self, frame: Frame, calibration: CameraCalibration, profile: AlignmentProfile
    ) -> tuple[Detection, ...]:
        cv2 = self.cv2
        image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot decode {frame.mime_type} camera frame")
        family_names = {
            "tag16h5": "DICT_APRILTAG_16h5",
            "tag25h9": "DICT_APRILTAG_25h9",
            "tag36h10": "DICT_APRILTAG_36h10",
            "tag36h11": "DICT_APRILTAG_36h11",
        }
        constant_name = family_names.get(profile.tag_family.lower())
        if constant_name is None or not hasattr(cv2.aruco, constant_name):
            raise RuntimeError(f"unsupported AprilTag family: {profile.tag_family}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, constant_name))
        parameters = self._parameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        detected_corners, ids, _rejected = detector.detectMarkers(image)
        marker_values: list[tuple[np.ndarray, int, dict[str, Any]]] = []
        if ids is not None:
            marker_values.extend(
                (
                    np.asarray(corners, dtype=np.float64).reshape(4, 2),
                    int(tag_id),
                    {
                        "fallback": False,
                        "corner_refinement": (
                            "subpixel"
                            if parameters.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX
                            else "none"
                        ),
                    },
                )
                for corners, tag_id in zip(detected_corners, ids.reshape(-1), strict=True)
            )
        if not any(tag_id in profile.tag_ids for _corners, tag_id, _diagnostics in marker_values):
            marker_values.extend(self._fallback_known_tags(image, dictionary, profile.tag_ids))
        if not marker_values:
            return ()
        object_points = tag_corners(profile.tag_size_m)
        results: list[Detection] = []
        for corners, tag_id, diagnostics in marker_values:
            if tag_id not in profile.tag_ids:
                continue
            diagnostics = dict(diagnostics)
            if diagnostics.get("fallback") and bool(
                self.config.get("fallback_corner_refinement", True)
            ):
                corners = self._refine_fallback_corners(image, corners)
                diagnostics["corner_refinement"] = "subpixel"
            else:
                diagnostics.setdefault("corner_refinement", "none")
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points,
                corners,
                calibration.camera_matrix,
                calibration.distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue
            rotation, _ = cv2.Rodrigues(rotation_vector)
            camera_from_tag = np.eye(4, dtype=np.float64)
            camera_from_tag[:3, :3] = rotation
            camera_from_tag[:3, 3] = translation_vector.reshape(3)
            projected, _ = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation_vector,
                calibration.camera_matrix,
                calibration.distortion,
            )
            error_px = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
            area = float(abs(cv2.contourArea(corners.astype(np.float32))))
            area_score = min(1.0, area / float(self.config.get("good_area_px", 6000.0)))
            error_score = math.exp(-error_px / float(self.config.get("error_scale_px", 3.0)))
            results.append(
                Detection(
                    tag_id=tag_id,
                    camera_from_tag=camera_from_tag,
                    corners=tuple((float(x), float(y)) for x, y in corners),
                    quality=max(0.0, min(1.0, area_score * error_score)),
                    reprojection_error_px=error_px,
                    diagnostics={"area_px": area, **diagnostics},
                )
            )
        return tuple(results)

    def _refine_fallback_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Improve fallback contour corners, without making them safety eligible."""
        cv2 = self.cv2
        window = max(1, min(10, int(self.config.get("corner_refinement_window_px", 5))))
        refined = np.asarray(corners, dtype=np.float32).reshape(4, 1, 2).copy()
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            max(1, min(200, int(self.config.get("corner_refinement_max_iterations", 30)))),
            max(1e-6, float(self.config.get("corner_refinement_min_accuracy", 0.01))),
        )
        cv2.cornerSubPix(image, refined, (window, window), (-1, -1), criteria)
        return refined.reshape(4, 2).astype(np.float64)

    def _fallback_known_tags(
        self,
        image: np.ndarray,
        dictionary: Any,
        tag_ids: tuple[int, ...],
    ) -> list[tuple[np.ndarray, int, dict[str, Any]]]:
        """Decode small known tags from dark quadrilaterals missed by ArUco.

        The G2 head stream is only 640x400. At the installed table distance a
        10 cm tag is roughly 30 pixels wide, where OpenCV sometimes rejects the
        quad before dictionary decoding. Restricting this fallback to configured
        IDs keeps it conservative while allowing the dictionary's error budget.
        """
        cv2 = self.cv2
        minimum_area = float(self.config.get("fallback_minimum_area_px", 120.0))
        maximum_area = image.shape[0] * image.shape[1] * float(
            self.config.get("fallback_maximum_image_ratio", 0.25)
        )
        max_hamming = int(self.config.get("fallback_max_hamming", 3))
        thresholds = tuple(self.config.get("fallback_dark_thresholds", (50, 65, 80, 95)))
        expected: dict[int, np.ndarray] = {}
        for tag_id in tag_ids:
            marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), 160)
            expected[int(tag_id)] = cv2.resize(marker, (8, 8), interpolation=cv2.INTER_AREA) >= 128

        best: dict[tuple[int, int, int], tuple[np.ndarray, int, dict[str, Any]]] = {}
        destination = np.float32(((0, 0), (159, 0), (159, 159), (0, 159)))
        for threshold_value in thresholds:
            _ok, dark = cv2.threshold(image, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
            contours, _hierarchy = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                hull = cv2.convexHull(contour)
                area = float(abs(cv2.contourArea(hull)))
                if area < minimum_area or area > maximum_area:
                    continue
                perimeter = float(cv2.arcLength(hull, True))
                quad = cv2.approxPolyDP(hull, 0.04 * perimeter, True)
                if len(quad) != 4 or not cv2.isContourConvex(quad):
                    continue
                corners = _ordered_quad(np.asarray(quad, dtype=np.float32).reshape(4, 2))
                if len(np.unique(corners, axis=0)) != 4:
                    continue
                lengths = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
                if float(np.min(lengths)) < 8 or float(np.min(lengths) / np.max(lengths)) < 0.35:
                    continue
                transform = cv2.getPerspectiveTransform(corners, destination)
                rectified = cv2.warpPerspective(image, transform, (160, 160))
                cells = cv2.resize(rectified, (8, 8), interpolation=cv2.INTER_AREA)
                split = (float(np.percentile(cells, 15)) + float(np.percentile(cells, 85))) / 2
                actual = cells >= split
                border = np.concatenate((actual[0], actual[-1], actual[1:-1, 0], actual[1:-1, -1]))
                if int(np.count_nonzero(border)) > 1:
                    continue
                for tag_id, expected_bits in expected.items():
                    for rotation in range(4):
                        hamming = int(np.count_nonzero(actual != np.rot90(expected_bits, rotation)))
                        if hamming > max_hamming:
                            continue
                        oriented = np.roll(corners.astype(np.float64), rotation, axis=0)
                        center = np.mean(oriented, axis=0)
                        key = (tag_id, int(round(center[0] / 4)), int(round(center[1] / 4)))
                        previous = best.get(key)
                        if previous is None or hamming < int(previous[2]["fallback_hamming"]):
                            best[key] = (
                                oriented,
                                tag_id,
                                {
                                    "fallback": True,
                                    "fallback_hamming": hamming,
                                    "fallback_threshold": int(threshold_value),
                                },
                            )
        return list(best.values())


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral points in screen top-left clockwise order."""
    ordered = np.empty((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = points[:, 1] - points[:, 0]
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


class MockAprilTagDetector(AprilTagDetector):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.started_at = time.monotonic()

    def detect(
        self, frame: Frame, calibration: CameraCalibration, profile: AlignmentProfile
    ) -> tuple[Detection, ...]:
        elapsed = time.monotonic() - self.started_at
        base_pose = list(self.config.get("actual_table_pose", (0.84, -0.012, 1.0, 0, 0, 2.8)))
        converge_s = float(self.config.get("converge_s", 0.0))
        if converge_s > 0:
            expected = list(matrix_pose(profile.base_from_expected_table))
            remaining = max(0.0, 1.0 - elapsed / converge_s)
            base_pose = [
                target + (actual - target) * remaining
                for actual, target in zip(base_pose, expected, strict=True)
            ]
        if bool(self.config.get("animate", True)):
            base_pose[0] += 0.003 * math.sin(elapsed * 0.8)
            base_pose[1] += 0.002 * math.sin(elapsed * 1.1)
            base_pose[5] += 0.15 * math.sin(elapsed * 0.7)
        base_from_table = pose_matrix(base_pose)
        camera_from_tag = invert(calibration.base_from_camera) @ base_from_table @ invert(profile.tag_from_table)
        corners = project_points(
            tag_corners(profile.tag_size_m),
            camera_from_tag,
            calibration.camera_matrix,
            calibration.distortion,
        )
        if corners is None:
            return ()
        return (
            Detection(
                tag_id=profile.tag_ids[0],
                camera_from_tag=camera_from_tag,
                corners=tuple((float(x), float(y)) for x, y in corners),
                quality=0.96,
                reprojection_error_px=0.18,
                diagnostics={"simulated": True},
            ),
        )
