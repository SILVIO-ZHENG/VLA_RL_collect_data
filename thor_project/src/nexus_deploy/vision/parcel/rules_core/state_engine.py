"""Geometry rules for turning YOLO instance masks into parcel states.

The segmentation model detects physical objects only.  This module deliberately
contains no Ultralytics dependency so that the downstream state rules can be
tested with synthetic masks and with the test-set ground-truth polygons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


STATE_NAMES = (
    "NO_PACKAGE",
    "ON_TABLE_LEFT",
    "IN_LEFT_GRIPPER",
    "IN_RIGHT_GRIPPER",
    "IN_BIN",
    "UNKNOWN",
)

PACKAGE = "package"
LEFT_GRIPPER = "left_gripper"
RIGHT_GRIPPER = "right_gripper"


@dataclass(frozen=True)
class Instance:
    """One visible object instance in original-image coordinates."""

    class_name: str
    confidence: float
    mask: np.ndarray
    box_xyxy: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"instance mask must be 2-D, got shape {mask.shape}")
        object.__setattr__(self, "mask", mask)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("instance confidence must be in [0, 1]")


@dataclass(frozen=True)
class RuleConfig:
    """Validated state-rule configuration."""

    bin_roi: tuple[tuple[float, float], ...]
    table_roi: tuple[tuple[float, float], ...]
    roi_membership_min: float = 0.55
    gripper_contact_distance_ratio: float = 0.025
    active_continuity_distance_ratio: float = 0.18
    max_track_frame_gap: int = 180
    bin_requires_tracking: bool = True
    min_tracking_continuity: float = 0.05
    active_score_ambiguity_margin: float = 0.15
    track_ambiguity_hold_frames: int = 3
    no_package_confirm_frames: int = 2
    expected_aspect_ratio: float = 1.60
    aspect_ratio_tolerance: float = 0.03
    min_report_confidence: float = 0.65
    no_package_confidence: float = 0.50
    min_mask_pixels: int = 20
    active_contact_weight: float = 4.0
    active_table_weight: float = 2.0
    active_outside_bin_weight: float = 1.0
    active_continuity_weight: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuleConfig":
        rois = payload.get("rois")
        rules = payload.get("rules", {})
        if not isinstance(rois, Mapping):
            raise ValueError("configuration must contain a 'rois' mapping")
        if not isinstance(rules, Mapping):
            raise ValueError("configuration 'rules' must be a mapping")

        def polygon(name: str) -> tuple[tuple[float, float], ...]:
            value = rois.get(name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"rois.{name} must be a list of normalized [x, y] points")
            points: list[tuple[float, float]] = []
            for point in value:
                if not isinstance(point, Sequence) or len(point) != 2:
                    raise ValueError(f"every rois.{name} point must contain x and y")
                x, y = float(point[0]), float(point[1])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError(f"rois.{name} coordinates must be in [0, 1]")
                points.append((x, y))
            if len(points) < 3:
                raise ValueError(f"rois.{name} must contain at least three points")
            return tuple(points)

        values = {
            field_name: rules.get(field_name, field_def.default)
            for field_name, field_def in cls.__dataclass_fields__.items()
            if field_name not in {"bin_roi", "table_roi"}
        }
        config = cls(
            bin_roi=polygon("bin"),
            table_roi=polygon("table_left"),
            **values,
        )
        config.validate()
        return config

    def validate(self) -> None:
        unit_interval_fields = (
            "roi_membership_min",
            "gripper_contact_distance_ratio",
            "active_continuity_distance_ratio",
            "min_tracking_continuity",
            "aspect_ratio_tolerance",
            "min_report_confidence",
            "no_package_confidence",
        )
        for name in unit_interval_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rules.{name} must be in [0, 1]")
        if self.gripper_contact_distance_ratio <= 0:
            raise ValueError("rules.gripper_contact_distance_ratio must be positive")
        if self.active_continuity_distance_ratio <= 0:
            raise ValueError("rules.active_continuity_distance_ratio must be positive")
        if int(self.max_track_frame_gap) < 0:
            raise ValueError("rules.max_track_frame_gap cannot be negative")
        if int(self.min_mask_pixels) < 1:
            raise ValueError("rules.min_mask_pixels must be positive")
        if int(self.no_package_confirm_frames) < 1:
            raise ValueError("rules.no_package_confirm_frames must be positive")
        if int(self.track_ambiguity_hold_frames) < 0:
            raise ValueError("rules.track_ambiguity_hold_frames cannot be negative")
        if float(self.active_score_ambiguity_margin) < 0:
            raise ValueError("rules.active_score_ambiguity_margin cannot be negative")
        if float(self.expected_aspect_ratio) <= 0:
            raise ValueError("rules.expected_aspect_ratio must be positive")
        if not isinstance(self.bin_requires_tracking, bool):
            raise ValueError("rules.bin_requires_tracking must be true or false")
        for name in (
            "active_contact_weight",
            "active_table_weight",
            "active_outside_bin_weight",
            "active_continuity_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"rules.{name} cannot be negative")


@dataclass(frozen=True)
class StateDecision:
    """A state plus the evidence used to produce it."""

    state: str
    confidence: float
    reason: str
    active_package_index: int | None
    active_centroid: tuple[float, float] | None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in STATE_NAMES:
            raise ValueError(f"unknown state: {self.state}")


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return an inclusive-exclusive mask box, or None for an empty mask."""

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def polygon_mask(
    shape: tuple[int, int], points: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Rasterize normalized polygon points for an image height/width."""

    height, width = shape
    pixels = np.asarray(
        [
            (
                int(round(x * max(width - 1, 0))),
                int(round(y * max(height - 1, 0))),
            )
            for x, y in points
        ],
        dtype=np.int32,
    )
    result = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(result, [pixels], 1)
    return result.astype(bool)


def overlap_fraction(mask: np.ndarray, roi: np.ndarray) -> float:
    pixels = int(mask.sum())
    if pixels == 0:
        return 0.0
    return float(np.count_nonzero(mask & roi) / pixels)


def minimum_mask_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Approximate the shortest visible-mask gap in pixels."""

    if not first.any() or not second.any():
        return float("inf")
    if np.any(first & second):
        return 0.0
    # OpenCV returns, for every non-zero pixel, the distance to the nearest
    # zero pixel.  The second mask is therefore encoded as zero.
    distance_input = (~second).astype(np.uint8)
    distances = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    return float(distances[first].min())


def _best_gripper_gap(
    package_mask: np.ndarray, grippers: Sequence[tuple[int, Instance]]
) -> tuple[float, float, int | None]:
    best_gap = float("inf")
    best_confidence = 0.0
    best_index: int | None = None
    for index, gripper in grippers:
        gap = minimum_mask_gap(package_mask, gripper.mask)
        if gap < best_gap:
            best_gap = gap
            best_confidence = float(gripper.confidence)
            best_index = index
    return best_gap, best_confidence, best_index


def _contact_strength(gap: float, contact_distance: float) -> float:
    if not isfinite(gap):
        return 0.0
    return float(np.clip(1.0 - gap / max(contact_distance, 1e-6), 0.0, 1.0))


class StateEngine:
    """Select the active package and apply conservative spatial state rules."""

    def __init__(self, config: RuleConfig):
        self.config = config
        self._source_id: str | None = None
        self._last_frame_index: int | None = None
        self._last_centroid: tuple[float, float] | None = None
        self._last_mask: np.ndarray | None = None
        self._missing_active_frames = 0
        self._ambiguous_track_frames = 0

    def reset(self) -> None:
        self._source_id = None
        self._last_frame_index = None
        self._last_centroid = None
        self._last_mask = None
        self._missing_active_frames = 0
        self._ambiguous_track_frames = 0

    def analyze(
        self,
        instances: Sequence[Instance],
        image_shape: tuple[int, int],
        *,
        source_id: str,
        frame_index: int | None,
    ) -> StateDecision:
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape must contain positive height and width")
        aspect_ratio = width / height
        relative_aspect_error = abs(aspect_ratio - self.config.expected_aspect_ratio) / self.config.expected_aspect_ratio
        if relative_aspect_error > self.config.aspect_ratio_tolerance:
            raise ValueError(
                f"image aspect ratio {aspect_ratio:.4f} differs from configured "
                f"{self.config.expected_aspect_ratio:.4f}; recalibrate normalized ROIs"
            )
        for instance in instances:
            if instance.mask.shape != image_shape:
                raise ValueError(
                    f"mask shape {instance.mask.shape} does not match image {image_shape}"
                )

        frame_gap: int | None = None
        can_continue = self._source_id == source_id
        if can_continue and frame_index is not None and self._last_frame_index is not None:
            frame_gap = frame_index - self._last_frame_index
            can_continue = 0 <= frame_gap <= self.config.max_track_frame_gap
        elif can_continue:
            can_continue = False
        previous = self._last_centroid if can_continue else None

        prior_centroid = self._last_centroid
        prior_mask = self._last_mask
        previous_mask = prior_mask if can_continue else None
        decision = self._analyze_frame(
            instances, image_shape, previous, previous_mask, frame_gap
        )
        if decision.state == "NO_PACKAGE" and can_continue and prior_centroid is not None:
            self._missing_active_frames += 1
            if self._missing_active_frames < self.config.no_package_confirm_frames:
                decision = StateDecision(
                    state="UNKNOWN",
                    confidence=0.0,
                    reason="short package-mask dropout while an active track is retained",
                    active_package_index=None,
                    active_centroid=prior_centroid,
                    metrics={
                        **decision.metrics,
                        "missing_active_frames": self._missing_active_frames,
                    },
                )
        else:
            self._missing_active_frames = 0
        retain_prior_track = bool(
            decision.state == "UNKNOWN"
            and decision.metrics.get("retain_prior_track")
            and can_continue
            and prior_centroid is not None
        )
        if retain_prior_track:
            self._ambiguous_track_frames += 1
        else:
            self._ambiguous_track_frames = 0

        self._source_id = source_id
        self._last_frame_index = frame_index
        if (
            retain_prior_track
            and self._ambiguous_track_frames <= self.config.track_ambiguity_hold_frames
        ):
            decision = StateDecision(
                state=decision.state,
                confidence=decision.confidence,
                reason=decision.reason,
                active_package_index=None,
                active_centroid=prior_centroid,
                metrics={
                    **decision.metrics,
                    "track_retained": True,
                    "track_ambiguity_frames": self._ambiguous_track_frames,
                },
            )
            self._last_centroid = prior_centroid
            self._last_mask = prior_mask
        elif decision.active_package_index is not None:
            self._last_centroid = decision.active_centroid
            self._last_mask = instances[decision.active_package_index].mask.copy()
        elif decision.reason.startswith("short package-mask dropout"):
            self._last_centroid = decision.active_centroid
            self._last_mask = prior_mask
        else:
            self._last_centroid = decision.active_centroid
            self._last_mask = None
        return decision

    def _analyze_frame(
        self,
        instances: Sequence[Instance],
        image_shape: tuple[int, int],
        previous_centroid: tuple[float, float] | None,
        previous_mask: np.ndarray | None,
        frame_gap: int | None,
    ) -> StateDecision:
        height, width = image_shape
        diagonal = hypot(width, height)
        bin_roi = polygon_mask(image_shape, self.config.bin_roi)
        table_roi = polygon_mask(image_shape, self.config.table_roi) & ~bin_roi
        indexed = list(enumerate(instances))
        packages = [
            item
            for item in indexed
            if item[1].class_name == PACKAGE
            and int(item[1].mask.sum()) >= self.config.min_mask_pixels
        ]
        left_grippers = [item for item in indexed if item[1].class_name == LEFT_GRIPPER]
        right_grippers = [item for item in indexed if item[1].class_name == RIGHT_GRIPPER]

        base_metrics: dict[str, Any] = {
            "package_count": len(packages),
            "left_gripper_count": len(left_grippers),
            "right_gripper_count": len(right_grippers),
            "frame_gap": frame_gap,
        }
        if not packages:
            return StateDecision(
                state="NO_PACKAGE",
                confidence=float(self.config.no_package_confidence),
                reason="no package mask above the minimum size",
                active_package_index=None,
                active_centroid=None,
                metrics=base_metrics,
            )

        contact_distance = diagonal * self.config.gripper_contact_distance_ratio
        continuity_distance = diagonal * self.config.active_continuity_distance_ratio
        candidates: list[dict[str, Any]] = []
        for index, package in packages:
            centroid = mask_centroid(package.mask)
            if centroid is None:
                continue
            bin_overlap = overlap_fraction(package.mask, bin_roi)
            table_overlap = overlap_fraction(package.mask, table_roi)
            left_gap, left_conf, left_index = _best_gripper_gap(package.mask, left_grippers)
            right_gap, right_conf, right_index = _best_gripper_gap(package.mask, right_grippers)
            left_contact = _contact_strength(left_gap, contact_distance)
            right_contact = _contact_strength(right_gap, contact_distance)
            contact = max(left_contact, right_contact)
            continuity = 0.0
            mask_iou = 0.0
            if previous_centroid is not None:
                distance = hypot(
                    centroid[0] - previous_centroid[0],
                    centroid[1] - previous_centroid[1],
                )
                continuity = float(
                    np.clip(1.0 - distance / max(continuity_distance, 1e-6), 0.0, 1.0)
                )
            if previous_mask is not None and previous_mask.shape == package.mask.shape:
                intersection = int(np.count_nonzero(previous_mask & package.mask))
                union = int(np.count_nonzero(previous_mask | package.mask))
                mask_iou = intersection / union if union else 0.0
                continuity = max(continuity, mask_iou)
            activity_score = (
                self.config.active_contact_weight * contact
                + self.config.active_table_weight * table_overlap
                + self.config.active_outside_bin_weight * (1.0 - bin_overlap)
                + self.config.active_continuity_weight * continuity
                + 0.10 * float(package.confidence)
            )
            candidates.append(
                {
                    "index": index,
                    "instance": package,
                    "centroid": centroid,
                    "bin_overlap": bin_overlap,
                    "table_overlap": table_overlap,
                    "left_gap_px": left_gap,
                    "right_gap_px": right_gap,
                    "left_gripper_confidence": left_conf,
                    "right_gripper_confidence": right_conf,
                    "left_gripper_index": left_index,
                    "right_gripper_index": right_index,
                    "left_contact": left_contact,
                    "right_contact": right_contact,
                    "continuity": continuity,
                    "mask_iou": mask_iou,
                    "activity_score": activity_score,
                    "left_near": left_gap <= contact_distance,
                    "right_near": right_gap <= contact_distance,
                }
            )

        if not candidates:
            return StateDecision(
                state="NO_PACKAGE",
                confidence=float(self.config.no_package_confidence),
                reason="all package masks were empty",
                active_package_index=None,
                active_centroid=None,
                metrics=base_metrics,
            )

        # A destination/source bin can contain parcels from other task cycles.
        # Without a prior track, a bin-only resident is not evidence that it is
        # the currently manipulated parcel.  Initialize on gripper contact, the
        # configured table surface, or a parcel outside the bin instead.
        if previous_centroid is None and self.config.bin_requires_tracking:
            initializable = [
                item
                for item in candidates
                if item["left_near"]
                or item["right_near"]
                or item["table_overlap"] >= self.config.roi_membership_min
                or item["bin_overlap"] < self.config.roi_membership_min
            ]
            if not initializable:
                return StateDecision(
                    state="UNKNOWN",
                    confidence=0.0,
                    reason="only untracked packages inside the bin ROI are visible",
                    active_package_index=None,
                    active_centroid=None,
                    metrics={
                        **base_metrics,
                        "untracked_bin_only": True,
                        "contact_threshold_px": float(contact_distance),
                    },
                )
            candidates = initializable

        # A deterministic geometric tie-break avoids detection/annotation order
        # changing the next frame's tracked centroid. Near ties still abstain.
        candidates.sort(
            key=lambda item: (
                float(item["activity_score"]),
                max(float(item["left_contact"]), float(item["right_contact"])),
                float(item["table_overlap"]),
                1.0 - float(item["bin_overlap"]),
                float(item["continuity"]),
                float(item["instance"].confidence),
                int(item["instance"].mask.sum()),
                float(item["centroid"][0]),
                float(item["centroid"][1]),
            ),
            reverse=True,
        )
        active = candidates[0]
        runner_up = candidates[1]["activity_score"] if len(candidates) > 1 else None
        score_margin = (
            float(active["activity_score"] - runner_up) if runner_up is not None else None
        )

        # If each gripper is touching a different parcel, a single frame cannot
        # tell which parcel belongs to the current task.  A reliable track from
        # prior video frames may disambiguate this; otherwise abstain instead of
        # selecting whichever detection happened to score slightly higher.
        left_only_contact = any(
            item["left_near"] and not item["right_near"] for item in candidates
        )
        right_only_contact = any(
            item["right_near"] and not item["left_near"] for item in candidates
        )
        has_reliable_track = (
            previous_centroid is not None
            and float(active["continuity"]) >= self.config.min_tracking_continuity
        )
        if left_only_contact and right_only_contact and not has_reliable_track:
            return StateDecision(
                state="UNKNOWN",
                confidence=0.0,
                reason=(
                    "opposite grippers contact different packages without a "
                    "reliable active track"
                ),
                active_package_index=None,
                active_centroid=None,
                metrics={
                    **base_metrics,
                    "active_score": float(active["activity_score"]),
                    "active_score_margin": score_margin,
                    "contact_threshold_px": float(contact_distance),
                    "retain_prior_track": previous_centroid is not None,
                },
            )
        if (
            score_margin is not None
            and score_margin <= self.config.active_score_ambiguity_margin
        ):
            return StateDecision(
                state="UNKNOWN",
                confidence=0.0,
                reason="multiple active-package candidates have similar evidence",
                active_package_index=None,
                active_centroid=None,
                metrics={
                    **base_metrics,
                    "active_score": float(active["activity_score"]),
                    "active_score_margin": score_margin,
                    "contact_threshold_px": float(contact_distance),
                    "retain_prior_track": previous_centroid is not None,
                },
            )

        package: Instance = active["instance"]
        left_gap = float(active["left_gap_px"])
        right_gap = float(active["right_gap_px"])
        left_near = left_gap <= contact_distance
        right_near = right_gap <= contact_distance

        metrics = {
            **base_metrics,
            "active_package_confidence": float(package.confidence),
            "active_centroid_x": float(active["centroid"][0] / width),
            "active_centroid_y": float(active["centroid"][1] / height),
            "bin_overlap": float(active["bin_overlap"]),
            "table_overlap": float(active["table_overlap"]),
            "left_gap_px": left_gap if isfinite(left_gap) else None,
            "right_gap_px": right_gap if isfinite(right_gap) else None,
            "left_contact": float(active["left_contact"]),
            "right_contact": float(active["right_contact"]),
            "active_score": float(active["activity_score"]),
            "active_score_margin": score_margin,
            "continuity": float(active["continuity"]),
            "track_mask_iou": float(active["mask_iou"]),
            "contact_threshold_px": float(contact_distance),
        }

        state: str
        confidence: float
        reason: str
        if left_near and right_near:
            state = "UNKNOWN"
            confidence = 0.0
            reason = "both grippers are close to the active package"
        elif left_near and (not right_near or left_gap < right_gap):
            state = "IN_LEFT_GRIPPER"
            proximity = float(active["left_contact"])
            confidence = float(
                np.clip(
                    package.confidence
                    * active["left_gripper_confidence"]
                    * (0.65 + 0.35 * proximity),
                    0.0,
                    1.0,
                )
            )
            reason = "active package is nearest to the physical left gripper"
        elif right_near:
            state = "IN_RIGHT_GRIPPER"
            proximity = float(active["right_contact"])
            confidence = float(
                np.clip(
                    package.confidence
                    * active["right_gripper_confidence"]
                    * (0.65 + 0.35 * proximity),
                    0.0,
                    1.0,
                )
            )
            reason = "active package is nearest to the physical right gripper"
        elif active["bin_overlap"] >= self.config.roi_membership_min:
            if (
                self.config.bin_requires_tracking
                and active["continuity"] < self.config.min_tracking_continuity
            ):
                state = "UNKNOWN"
                confidence = 0.0
                reason = "bin package cannot be associated with the previous active track"
            else:
                state = "IN_BIN"
                confidence = float(
                    np.clip(
                        package.confidence
                        * active["bin_overlap"]
                        / self.config.roi_membership_min,
                        0.0,
                        1.0,
                    )
                )
                reason = "tracked active package mask is predominantly inside the bin ROI"
        elif active["table_overlap"] >= self.config.roi_membership_min:
            state = "ON_TABLE_LEFT"
            confidence = float(
                np.clip(
                    package.confidence
                    * active["table_overlap"]
                    / self.config.roi_membership_min,
                    0.0,
                    1.0,
                )
            )
            reason = "active package mask is predominantly inside the table ROI"
        else:
            state = "UNKNOWN"
            confidence = 0.0
            reason = "active package is not close to one gripper or inside a configured ROI"

        return StateDecision(
            state=state,
            confidence=confidence,
            reason=reason,
            active_package_index=int(active["index"]),
            active_centroid=active["centroid"],
            metrics=metrics,
        )


def normalized_polygon_pixels(
    points: Sequence[tuple[float, float]], image_shape: tuple[int, int]
) -> np.ndarray:
    """Expose polygon vertices for visualization without duplicating scaling."""

    height, width = image_shape
    return np.asarray(
        [
            (
                int(round(x * max(width - 1, 0))),
                int(round(y * max(height - 1, 0))),
            )
            for x, y in points
        ],
        dtype=np.int32,
    )
