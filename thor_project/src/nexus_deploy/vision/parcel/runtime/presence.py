"""Aggregate parcel-presence gate built from the existing YOLO instance masks.

This module deliberately does not control a robot.  It turns *all* ``package``
instances from one already-computed YOLO inference into a small business-state
stream for a later, separately reviewed pause/resume integration. Per-camera
configuration decides whether gripper contact may qualify outside the work area.

The existing stage engine answers "where is one active parcel?".  This gate
answers the independent question "is any parcel still in the allowed work
area or held by a gripper?"  Multiple visible parcels are therefore evidence
for ``PACKAGE_PRESENT``, not ambiguity.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from math import hypot
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .core_bridge import Instance
from .schema import StageResult


PACKAGE_PRESENT = "PACKAGE_PRESENT"
NO_PACKAGE = "NO_PACKAGE"
UNKNOWN = "UNKNOWN"
PRESENCE_STATES = (PACKAGE_PRESENT, NO_PACKAGE, UNKNOWN)

PACKAGE = "package"
LEFT_GRIPPER = "left_gripper"
RIGHT_GRIPPER = "right_gripper"

Polygon = tuple[tuple[float, float], ...]


def _normalized_polygon(value: Any, field_name: str) -> Polygon:
    """Validate a normalized [0, 1] polygon from the YAML configuration."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of normalized [x, y] points")
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError(f"every {field_name} point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{field_name} coordinates must be in [0, 1]")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError(f"{field_name} must contain at least three points")
    return tuple(points)


def _polygon_mask(shape: tuple[int, int], polygon: Polygon) -> np.ndarray:
    """Rasterize a normalized polygon in an HxW image coordinate system."""

    height, width = shape
    pixels = np.asarray(
        [
            (
                int(round(x * max(width - 1, 0))),
                int(round(y * max(height - 1, 0))),
            )
            for x, y in polygon
        ],
        dtype=np.int32,
    )
    result = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(result, [pixels], 1)
    return result.astype(bool)


def _minimum_mask_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Return the minimum visible-mask gap in pixels, including overlap as zero."""

    if not first.any() or not second.any():
        return float("inf")
    if np.any(first & second):
        return 0.0
    distances = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distances[first].min())


@dataclass(frozen=True)
class PresenceGateConfig:
    """Validated, geometry-only settings for the aggregate parcel-presence gate."""

    include_roi_names: tuple[str, ...]
    exclude_roi_names: tuple[str, ...]
    polygons: Mapping[str, Polygon]
    roi_membership_min: float = 0.55
    min_mask_pixels: int = 20
    gripper_contact_distance_ratio: float = 0.025
    # One rolling window contains only actual detector samples, never duplicated
    # frames.  The deployment config deliberately uses four samples.
    window_frames: int = 4
    no_package_confidence: float = 0.50
    allow_gripper_outside_roi: bool = True
    ambiguous_roi_is_unknown: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PresenceGateConfig":
        """Read ``rois`` and the optional ``presence_gate`` section from YAML."""

        if not isinstance(payload, Mapping):
            raise ValueError("configuration must contain a mapping")
        raw_rois = payload.get("rois")
        if not isinstance(raw_rois, Mapping):
            raise ValueError("configuration must contain a 'rois' mapping")
        rules = payload.get("rules", {})
        if not isinstance(rules, Mapping):
            raise ValueError("configuration 'rules' must be a mapping")
        settings = payload.get("presence_gate", {})
        if settings is None:
            settings = {}
        if not isinstance(settings, Mapping):
            raise ValueError("configuration 'presence_gate' must be a mapping")

        def roi_names(name: str, default: Sequence[str]) -> tuple[str, ...]:
            value = settings.get(name, default)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"presence_gate.{name} must be a list of ROI names")
            names = tuple(str(item) for item in value)
            if not names or any(not item for item in names):
                raise ValueError(f"presence_gate.{name} cannot be empty")
            if len(set(names)) != len(names):
                raise ValueError(f"presence_gate.{name} cannot contain duplicates")
            return names

        include_names = roi_names("include_rois", ("bin", "table_left"))
        exclude_value = settings.get("exclude_rois", ())
        if not isinstance(exclude_value, Sequence) or isinstance(exclude_value, (str, bytes)):
            raise ValueError("presence_gate.exclude_rois must be a list of ROI names")
        exclude_names = tuple(str(item) for item in exclude_value)
        if any(not item for item in exclude_names) or len(set(exclude_names)) != len(exclude_names):
            raise ValueError("presence_gate.exclude_rois cannot contain empty or duplicate names")

        needed_names = (*include_names, *exclude_names)
        polygons: dict[str, Polygon] = {}
        for name in needed_names:
            if name not in raw_rois:
                raise ValueError(f"presence_gate references unknown ROI: {name}")
            polygons[name] = _normalized_polygon(raw_rois[name], f"rois.{name}")

        config = cls(
            include_roi_names=include_names,
            exclude_roi_names=exclude_names,
            polygons=polygons,
            roi_membership_min=float(
                settings.get("roi_membership_min", rules.get("roi_membership_min", 0.55))
            ),
            min_mask_pixels=int(settings.get("min_mask_pixels", rules.get("min_mask_pixels", 20))),
            gripper_contact_distance_ratio=float(
                settings.get(
                    "gripper_contact_distance_ratio",
                    rules.get("gripper_contact_distance_ratio", 0.025),
                )
            ),
            window_frames=int(settings.get("window_frames", 4)),
            no_package_confidence=float(
                settings.get("no_package_confidence", rules.get("no_package_confidence", 0.50))
            ),
            allow_gripper_outside_roi=settings.get("allow_gripper_outside_roi", True),
            ambiguous_roi_is_unknown=settings.get("ambiguous_roi_is_unknown", False),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.include_roi_names:
            raise ValueError("presence_gate.include_rois cannot be empty")
        for name in (*self.include_roi_names, *self.exclude_roi_names):
            if name not in self.polygons:
                raise ValueError(f"presence_gate has no polygon for ROI: {name}")
        if not 0.0 < self.roi_membership_min <= 1.0:
            raise ValueError("presence_gate.roi_membership_min must be in (0, 1]")
        if self.min_mask_pixels < 1:
            raise ValueError("presence_gate.min_mask_pixels must be positive")
        if self.gripper_contact_distance_ratio <= 0.0:
            raise ValueError("presence_gate.gripper_contact_distance_ratio must be positive")
        if self.window_frames < 1:
            raise ValueError("presence_gate.window_frames must be positive")
        if not 0.0 <= self.no_package_confidence <= 1.0:
            raise ValueError("presence_gate.no_package_confidence must be in [0, 1]")
        for name in ("allow_gripper_outside_roi", "ambiguous_roi_is_unknown"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"presence_gate.{name} must be a boolean")


@dataclass(frozen=True)
class PresenceResult:
    """One temporally confirmed aggregate parcel-presence result.

    ``state`` is intentionally not a robot-motion permission.  A future caller
    may use it only to advise or request soft pause/resume of an already active
    task after its own safety checks.
    """

    state: str
    confidence: float
    changed: bool
    raw_state: str
    raw_confidence: float
    candidate_state: str | None
    candidate_frames: int
    # Latest valid detector samples in the rolling window, oldest first.
    window_states: tuple[str, ...]
    package_count: int
    qualifying_package_count: int
    ignored_package_count: int
    reason: str
    raw_reason: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.state not in PRESENCE_STATES or self.raw_state not in PRESENCE_STATES:
            raise ValueError("presence states must use the published enum")
        if self.candidate_state is not None and self.candidate_state not in PRESENCE_STATES:
            raise ValueError("candidate_state must use the published enum")
        if any(state not in PRESENCE_STATES for state in self.window_states):
            raise ValueError("window_states must use the published enum")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("presence confidence must be in [0, 1]")
        if not 0.0 <= float(self.raw_confidence) <= 1.0:
            raise ValueError("raw presence confidence must be in [0, 1]")
        for value in (
            self.candidate_frames,
            self.package_count,
            self.qualifying_package_count,
            self.ignored_package_count,
        ):
            if int(value) < 0:
                raise ValueError("presence counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class FrameInferenceResult:
    """Existing single-active-parcel stage and aggregate presence from one inference."""

    stage: StageResult
    presence: PresenceResult

    def to_dict(self) -> dict[str, Any]:
        payload = self.stage.to_dict()
        payload["presence"] = self.presence.to_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class _RawPresence:
    state: str
    confidence: float
    package_count: int
    qualifying_package_count: int
    ignored_package_count: int
    reason: str


class ParcelPresenceGate:
    """Classify all detected packages with an exact rolling-window rule.

    A window of all ``PACKAGE_PRESENT`` samples produces ``PACKAGE_PRESENT``.
    A window of all ``NO_PACKAGE`` samples produces ``NO_PACKAGE``.  Every
    partial or mixed window produces ``UNKNOWN``.  Consumers must treat
    ``UNKNOWN`` as *no control action*.
    """

    def __init__(self, config: PresenceGateConfig):
        self.config = config
        self._window: deque[_RawPresence] = deque(maxlen=config.window_frames)
        self._last_emitted_state = UNKNOWN

    def reset(self) -> None:
        """Clear temporal evidence after a new task, camera reconnect, or FOV change."""

        self._window.clear()
        self._last_emitted_state = UNKNOWN

    def _roi_masks(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        include = np.zeros(shape, dtype=bool)
        for name in self.config.include_roi_names:
            include |= _polygon_mask(shape, self.config.polygons[name])
        exclude = np.zeros(shape, dtype=bool)
        for name in self.config.exclude_roi_names:
            exclude |= _polygon_mask(shape, self.config.polygons[name])
        return include, exclude

    def _raw_presence(
        self, instances: Sequence[Instance], image_shape: tuple[int, int]
    ) -> _RawPresence:
        include_roi, exclude_roi = self._roi_masks(image_shape)
        indexed = list(enumerate(instances))
        for _, instance in indexed:
            if instance.mask.shape != image_shape:
                raise ValueError(
                    f"instance mask shape {instance.mask.shape} does not match image {image_shape}"
                )
        packages = [
            item
            for item in indexed
            if item[1].class_name == PACKAGE and int(item[1].mask.sum()) >= self.config.min_mask_pixels
        ]
        grippers = [
            item
            for item in indexed
            if item[1].class_name in {LEFT_GRIPPER, RIGHT_GRIPPER}
        ]
        diagonal = hypot(image_shape[0], image_shape[1])
        contact_distance = diagonal * self.config.gripper_contact_distance_ratio

        qualifying = 0
        ignored = 0
        ambiguous = 0
        confidence = 0.0
        for _, package in packages:
            package_pixels = int(package.mask.sum())
            excluded_fraction = float(
                np.count_nonzero(package.mask & exclude_roi) / package_pixels
            )
            # Excluded regions have priority over included work-area regions.
            # This prevents a conveyor parcel that slightly overlaps a broad
            # table polygon from being treated as a work-area parcel.
            work_fraction = float(
                np.count_nonzero(package.mask & include_roi & ~exclude_roi)
                / package_pixels
            )
            held_by_gripper = self.config.allow_gripper_outside_roi and any(
                gripper.mask.shape == image_shape
                and _minimum_mask_gap(package.mask, gripper.mask) <= contact_distance
                for _, gripper in grippers
            )
            if held_by_gripper or (
                excluded_fraction < self.config.roi_membership_min
                and work_fraction >= self.config.roi_membership_min
            ):
                qualifying += 1
                confidence = max(confidence, float(package.confidence))
            elif (self.config.ambiguous_roi_is_unknown and work_fraction > 0.0
                  and excluded_fraction < self.config.roi_membership_min):
                # Partly crossing the work-area border is not evidence that
                # the area is empty. Keep it separate from wholly outside masks.
                ambiguous += 1
            else:
                ignored += 1

        if qualifying:
            return _RawPresence(
                state=PACKAGE_PRESENT,
                confidence=confidence,
                package_count=len(packages),
                qualifying_package_count=qualifying,
                ignored_package_count=ignored,
                reason=(
                    "one or more package masks are inside the configured work area "
                    "or in contact with a gripper"
                    if self.config.allow_gripper_outside_roi else
                    "one or more package masks are inside the configured work area"
                ),
            )
        if ambiguous:
            return _RawPresence(
                state=UNKNOWN,
                confidence=0.0,
                package_count=len(packages),
                qualifying_package_count=0,
                ignored_package_count=ignored,
                reason="package mask crosses the work-area boundary without confirmed ROI membership",
            )
        return _RawPresence(
            state=NO_PACKAGE,
            confidence=self.config.no_package_confidence,
            package_count=len(packages),
            qualifying_package_count=0,
            ignored_package_count=ignored,
            reason=(
                "no valid package mask is inside the configured work area or in "
                "contact with a gripper"
                if self.config.allow_gripper_outside_roi else
                "no valid package mask is inside the configured work area; outside packages are ignored"
            ),
        )

    def _result(
        self,
        *,
        state: str,
        confidence: float,
        raw: _RawPresence,
        candidate_state: str | None,
        candidate_frames: int,
        window_states: tuple[str, ...],
        reason: str,
    ) -> PresenceResult:
        changed = state != self._last_emitted_state
        self._last_emitted_state = state
        return PresenceResult(
            state=state,
            confidence=float(confidence),
            changed=changed,
            raw_state=raw.state,
            raw_confidence=float(raw.confidence),
            candidate_state=candidate_state,
            candidate_frames=int(candidate_frames),
            window_states=window_states,
            package_count=raw.package_count,
            qualifying_package_count=raw.qualifying_package_count,
            ignored_package_count=raw.ignored_package_count,
            reason=reason,
            raw_reason=raw.reason,
        )

    def update(
        self, instances: Sequence[Instance], image_shape: tuple[int, int]
    ) -> PresenceResult:
        """Consume one real inference result using the exact rolling-window rule."""

        if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
            raise ValueError("image_shape must contain positive height and width")
        raw = self._raw_presence(instances, image_shape)
        self._window.append(raw)
        window = tuple(item.state for item in self._window)
        if len(self._window) < self.config.window_frames:
            return self._result(
                state=UNKNOWN,
                confidence=0.0,
                raw=raw,
                candidate_state=raw.state if len(set(window)) == 1 else None,
                candidate_frames=len(window) if len(set(window)) == 1 else 0,
                window_states=window,
                reason=(
                    f"waiting for a complete {self.config.window_frames}-sample "
                    "presence window"
                ),
            )
        if len(set(window)) == 1:
            state = window[0]
            confidence = min(item.confidence for item in self._window)
            return self._result(
                state=state,
                confidence=confidence,
                raw=raw,
                candidate_state=None,
                candidate_frames=0,
                window_states=window,
                reason=(
                    f"all {self.config.window_frames} sampled frames agree on {state}"
                ),
            )
        return self._result(
            state=UNKNOWN,
            confidence=0.0,
            raw=raw,
            candidate_state=None,
            candidate_frames=0,
            window_states=window,
            reason=(
                f"the latest {self.config.window_frames}-sample presence window "
                + ("contains UNKNOWN evidence" if UNKNOWN in window else
                   "contains both PACKAGE_PRESENT and NO_PACKAGE")
            ),
        )

    def mark_unknown(self, reason: str) -> PresenceResult:
        """Report a camera/model fault without converting it into a pause/resume signal."""

        message = str(reason).strip()
        if not message:
            raise ValueError("unknown-presence reason cannot be empty")
        # A camera/model fault invalidates prior visual evidence.  Require a
        # completely new four-sample window when valid inference resumes.
        self._window.clear()
        raw = _RawPresence(
            state=UNKNOWN,
            confidence=0.0,
            package_count=0,
            qualifying_package_count=0,
            ignored_package_count=0,
            reason=message,
        )
        return self._result(
            state=UNKNOWN,
            confidence=0.0,
            raw=raw,
            candidate_state=None,
            candidate_frames=0,
            window_states=(),
            reason=message,
        )
