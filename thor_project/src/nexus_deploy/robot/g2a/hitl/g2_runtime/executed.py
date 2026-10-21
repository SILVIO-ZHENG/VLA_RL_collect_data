"""Paper-original FlowPRO execution history and rollback-anchor selection.

This module is control-transport independent: it records already-observed
execution evidence and selects data anchors, but never sends GDK/DDS commands.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from ...effector_profiles import get_effector_profile
from ..common.sidecar import CAMERA_SOURCES

EXECUTED_TRAJECTORY_PROTOCOL = "flowpro.executed_trajectory/v2"
# Explicit finite retention remains available for tests/resource-constrained
# deployments. Production FlowPRO keeps the active episode history and clears
# it at the next tablet-session bind instead of imposing a time horizon.
DEFAULT_HISTORY_NS = 10_000_000_000
DEFAULT_MAX_ROLLBACK_NS = 2_000_000_000
DEFAULT_MAX_ANCHOR_SNAP_NS = 200_000_000
DEFAULT_MAX_EXECUTION_GAP_NS = 100_000_000
# Observation capture necessarily precedes the first executed action by the
# non-RTC inference/transport latency.  This is not an execution underrun: the
# robot remains at the common anchor state until a candidate is installed.
# Keep this bound separate from the strict 100 ms in-trajectory gap check.
DEFAULT_MAX_ANCHOR_TO_FIRST_EXECUTION_NS = 1_000_000_000
MAX_CAMERA_SKEW_NS = 40_000_000
EXECUTION_KIND_POLICY_ACTION = "policy_action"
EXECUTION_KIND_CHUNK_TRANSITION = "chunk_transition_blend"
EXECUTION_KIND_QUEUE_UNDERRUN = "queue_underrun_hold"
EXECUTION_KIND_OPERATOR_HOLD = "operator_hold"
EXECUTION_KINDS = {
    EXECUTION_KIND_POLICY_ACTION,
    EXECUTION_KIND_CHUNK_TRANSITION,
    EXECUTION_KIND_QUEUE_UNDERRUN,
    EXECUTION_KIND_OPERATOR_HOLD,
}
EXECUTION_KIND_CODES = {
    EXECUTION_KIND_POLICY_ACTION: 0,
    EXECUTION_KIND_CHUNK_TRANSITION: 1,
    EXECUTION_KIND_QUEUE_UNDERRUN: 2,
    EXECUTION_KIND_OPERATOR_HOLD: 3,
}
EXECUTION_GAP_ANCHOR_TO_FIRST = "anchor_to_first_execution"
EXECUTION_GAP_BETWEEN_SAMPLES = "between_execution_samples"
EXECUTION_GAP_LAST_TO_FAILURE = "last_execution_to_failure"
EXECUTION_GAP_KINDS = {
    EXECUTION_GAP_ANCHOR_TO_FIRST,
    EXECUTION_GAP_BETWEEN_SAMPLES,
    EXECUTION_GAP_LAST_TO_FAILURE,
}


def _uuid(value: str, label: str) -> None:
    try:
        UUID(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a UUID") from error


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite [{size}] vector")
    result = array.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class InferenceAnchor:
    observation_id: str
    generation_id: str
    timestamp_ns: int
    state_24d: np.ndarray
    camera_timestamps_ns: dict[str, int]
    exact_observation_complete: bool = True

    def __post_init__(self) -> None:
        _uuid(self.observation_id, "observation_id")
        _uuid(self.generation_id, "generation_id")
        if isinstance(self.timestamp_ns, bool) or self.timestamp_ns <= 0:
            raise ValueError("Inference anchor timestamp_ns must be positive")
        object.__setattr__(self, "state_24d", _vector(self.state_24d, 24, "anchor state_24d"))
        if set(self.camera_timestamps_ns) != set(CAMERA_SOURCES):
            raise ValueError("Inference anchor must reference the exact three cameras")
        camera_timestamps = {key: int(self.camera_timestamps_ns[key]) for key in CAMERA_SOURCES}
        if any(value <= 0 for value in camera_timestamps.values()):
            raise ValueError("Inference anchor camera timestamps must be positive")
        object.__setattr__(self, "camera_timestamps_ns", camera_timestamps)

    @property
    def camera_skew_ns(self) -> int:
        timestamps = tuple(self.camera_timestamps_ns.values())
        return max(abs(timestamp - self.timestamp_ns) for timestamp in timestamps)


@dataclass(frozen=True)
class PolicyExecutionSample:
    timestamp_ns: int
    measured_timestamp_ns: int
    observation_id: str
    candidate_id: str
    generation_id: str
    action_id: str
    chunk_id: str
    action_index: int
    policy_native_action: np.ndarray
    safety_target_24d: np.ndarray
    gdk_command_24d: np.ndarray
    measured_state_24d: np.ndarray
    execution_kind: str = EXECUTION_KIND_POLICY_ACTION
    control_owner: str = "policy"
    command_accepted: bool = True
    safety_modified: bool = False
    request_started_monotonic_ns: int | None = None
    response_received_monotonic_ns: int | None = None
    chunk_installed_monotonic_ns: int | None = None
    latency_stale_steps: int | None = None
    installed_first_action_index: int | None = None
    first_executed_timestamp_ns: int | None = None
    first_executed_action_index: int | None = None

    def __post_init__(self) -> None:
        for label in ("observation_id", "candidate_id", "generation_id", "action_id"):
            _uuid(str(getattr(self, label)), label)
        if not self.chunk_id:
            raise ValueError("chunk_id must be non-empty")
        if (
            isinstance(self.timestamp_ns, bool)
            or isinstance(self.measured_timestamp_ns, bool)
            or self.timestamp_ns <= 0
            or self.measured_timestamp_ns <= 0
        ):
            raise ValueError("Execution timestamps must be positive")
        if isinstance(self.action_index, bool) or self.action_index < 0:
            raise ValueError("action_index must be non-negative")
        timing_values = (
            self.request_started_monotonic_ns,
            self.response_received_monotonic_ns,
            self.chunk_installed_monotonic_ns,
            self.first_executed_timestamp_ns,
        )
        if any(
            value is not None and (isinstance(value, bool) or value <= 0)
            for value in timing_values
        ):
            raise ValueError("Execution timing values must be positive integers when present")
        index_values = (
            self.latency_stale_steps,
            self.installed_first_action_index,
            self.first_executed_action_index,
        )
        if any(
            value is not None and (isinstance(value, bool) or value < 0)
            for value in index_values
        ):
            raise ValueError("Execution timing indices must be non-negative integers when present")
        monotonic_values = (
            self.request_started_monotonic_ns,
            self.response_received_monotonic_ns,
            self.chunk_installed_monotonic_ns,
        )
        if all(value is not None for value in monotonic_values) and not (
            monotonic_values[0] <= monotonic_values[1] <= monotonic_values[2]
        ):
            raise ValueError("Request/response/install monotonic timestamps are inconsistent")
        if (
            self.installed_first_action_index is not None
            and self.latency_stale_steps is not None
            and self.installed_first_action_index < self.latency_stale_steps
        ):
            raise ValueError("Installed action index cannot precede the latency-stale prefix")
        if (
            self.first_executed_action_index is not None
            and self.installed_first_action_index is not None
            and self.first_executed_action_index < self.installed_first_action_index
        ):
            raise ValueError("First executed action index cannot precede the installed prefix")
        if (
            self.first_executed_timestamp_ns is not None
            and self.first_executed_timestamp_ns > self.timestamp_ns
        ):
            raise ValueError("First chunk execution timestamp cannot follow this execution sample")
        if self.execution_kind not in EXECUTION_KINDS:
            raise ValueError(f"Unknown Policy execution_kind: {self.execution_kind!r}")
        native = np.asarray(self.policy_native_action, dtype=np.float32)
        if native.shape not in {(16,), (24,)} or not np.isfinite(native).all():
            raise ValueError("policy_native_action must be a finite [16] or [24] vector")
        native = native.copy()
        native.flags.writeable = False
        object.__setattr__(self, "policy_native_action", native)
        for field in ("safety_target_24d", "gdk_command_24d", "measured_state_24d"):
            object.__setattr__(self, field, _vector(getattr(self, field), 24, field))


@dataclass(frozen=True)
class RollbackAnchorSelection:
    anchor: InferenceAnchor
    failure_timestamp_ns: int
    requested_horizon_ns: int
    requested_target_timestamp_ns: int
    actual_anchor_timestamp_ns: int
    snap_offset_ns: int


@dataclass(frozen=True)
class ExecutionHistoryGap:
    """A known interval for which no accepted Policy sample was recorded.

    Gaps are evidence, not synthetic trajectory samples.  Samples on both
    sides remain available for diagnostics, but the physical planner and
    paper-strict preference training must both reject the gapped interval.
    """

    kind: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    strict_limit_ns: int

    def __post_init__(self) -> None:
        if self.kind not in EXECUTION_GAP_KINDS:
            raise ValueError(f"Unknown execution history gap kind: {self.kind!r}")
        values = (self.start_timestamp_ns, self.end_timestamp_ns, self.strict_limit_ns)
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("Execution history gap timestamps/limit must be positive")
        if self.end_timestamp_ns <= self.start_timestamp_ns:
            raise ValueError("Execution history gap must have positive duration")
        if self.duration_ns <= self.strict_limit_ns:
            raise ValueError("Execution history gap must exceed its strict limit")

    @property
    def duration_ns(self) -> int:
        return self.end_timestamp_ns - self.start_timestamp_ns


@dataclass(frozen=True)
class ExecutedRejectedSegment:
    anchor: InferenceAnchor
    failure_timestamp_ns: int
    timestamps_ns: np.ndarray
    measured_model_actions: np.ndarray
    samples: tuple[PolicyExecutionSample, ...]
    generation_ids: tuple[str, ...]
    safety_modified_count: int
    rollback_anchors: tuple[InferenceAnchor, ...] = ()
    execution_gaps: tuple[ExecutionHistoryGap, ...] = ()

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64).copy()
        actions = np.asarray(self.measured_model_actions, dtype=np.float32).copy()
        if timestamps.ndim != 1 or actions.ndim != 2 or len(timestamps) != len(actions):
            raise ValueError("Executed segment timestamps/actions must have matching leading dimensions")
        if len(timestamps) and np.any(np.diff(timestamps) <= 0):
            raise ValueError("Executed segment timestamps must be strictly increasing")
        if not np.isfinite(actions).all():
            raise ValueError("Executed segment actions must be finite")
        timestamps.flags.writeable = False
        actions.flags.writeable = False
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "measured_model_actions", actions)
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "generation_ids", tuple(self.generation_ids))
        object.__setattr__(self, "rollback_anchors", tuple(self.rollback_anchors))
        object.__setattr__(self, "execution_gaps", tuple(self.execution_gaps))

    @property
    def execution_kind_counts(self) -> dict[str, int]:
        return {
            kind: sum(sample.execution_kind == kind for sample in self.samples)
            for kind in sorted(EXECUTION_KINDS)
        }

    @property
    def queue_underrun_count(self) -> int:
        return self.execution_kind_counts[EXECUTION_KIND_QUEUE_UNDERRUN]

    @property
    def execution_gap_count(self) -> int:
        return len(self.execution_gaps)

    @property
    def max_execution_gap_ns(self) -> int:
        return max((gap.duration_ns for gap in self.execution_gaps), default=0)

    @property
    def physical_history_complete(self) -> bool:
        """Whether every interval needed for rewind has measured evidence.

        Explicit ``queue_underrun_hold`` samples are real measured evidence and
        therefore do not make physical history incomplete.  A bare timing gap
        does: callers must not execute a rollback plan across one.
        """

        return not self.execution_gaps

    @property
    def strict_preference_eligible(self) -> bool:
        """Whether this executed negative is eligible for paper-strict training.

        An unexpected underrun hold remains in frozen physical history so
        rollback can retrace the actual path, but it invalidates the pair. An
        explicit operator HOLD is also retained physically, then removed from
        the rejected training branch by the postprocessor; it is not an
        inference failure. A bare evidence gap remains strict-ineligible.
        """

        return self.queue_underrun_count == 0 and not self.execution_gaps


def select_rollback_anchor(
    anchors: list[InferenceAnchor] | tuple[InferenceAnchor, ...],
    *,
    failure_timestamp_ns: int,
    requested_horizon_ns: int,
    max_anchor_snap_ns: int = DEFAULT_MAX_ANCHOR_SNAP_NS,
    max_rollback_ns: int = DEFAULT_MAX_ROLLBACK_NS,
) -> RollbackAnchorSelection:
    """Snap a requested rollback target backward to the nearest valid inference anchor."""

    if failure_timestamp_ns <= 0 or not 0 < requested_horizon_ns <= max_rollback_ns:
        raise ValueError("Rollback failure time/horizon is outside the configured window")
    requested_target = failure_timestamp_ns - requested_horizon_ns
    eligible = [
        anchor
        for anchor in anchors
        if anchor.timestamp_ns <= requested_target
        and anchor.exact_observation_complete
        and anchor.camera_skew_ns <= MAX_CAMERA_SKEW_NS
    ]
    if not eligible:
        raise ValueError("No complete inference anchor exists before the requested rollback target")
    anchor = max(eligible, key=lambda item: item.timestamp_ns)
    snap_offset = requested_target - anchor.timestamp_ns
    if snap_offset > max_anchor_snap_ns:
        raise ValueError("Nearest valid inference anchor exceeds the maximum snap offset")
    if failure_timestamp_ns - anchor.timestamp_ns > max_rollback_ns:
        raise ValueError("Selected inference anchor exceeds the maximum rollback history")
    return RollbackAnchorSelection(
        anchor=anchor,
        failure_timestamp_ns=failure_timestamp_ns,
        requested_horizon_ns=requested_horizon_ns,
        requested_target_timestamp_ns=requested_target,
        actual_anchor_timestamp_ns=anchor.timestamp_ns,
        snap_offset_ns=snap_offset,
    )


def select_oldest_rollback_anchor(
    anchors: list[InferenceAnchor] | tuple[InferenceAnchor, ...],
    *,
    failure_timestamp_ns: int,
    max_rollback_ns: int = DEFAULT_MAX_ROLLBACK_NS,
) -> RollbackAnchorSelection:
    """Select the oldest complete inference anchor inside the safe window.

    Continuous X-hold rollback plans the maximum allowed history once.  The
    operator may stop at a newer anchor, but the immutable plan/snapshot must
    extend to the oldest usable anchor available when X was first pressed.
    """

    if failure_timestamp_ns <= 0 or max_rollback_ns <= 0:
        raise ValueError("Rollback failure time/window must be positive")
    requested_target = failure_timestamp_ns - max_rollback_ns
    eligible = [
        anchor
        for anchor in anchors
        if requested_target <= anchor.timestamp_ns < failure_timestamp_ns
        and anchor.exact_observation_complete
        and anchor.camera_skew_ns <= MAX_CAMERA_SKEW_NS
    ]
    if not eligible:
        raise ValueError("No complete inference anchor exists inside the rollback window")
    anchor = min(eligible, key=lambda item: item.timestamp_ns)
    return RollbackAnchorSelection(
        anchor=anchor,
        failure_timestamp_ns=failure_timestamp_ns,
        requested_horizon_ns=max_rollback_ns,
        requested_target_timestamp_ns=requested_target,
        actual_anchor_timestamp_ns=anchor.timestamp_ns,
        snap_offset_ns=anchor.timestamp_ns - requested_target,
    )


class FlowPROExecutionHistory:
    """Active-episode evidence used for rollback planning and v2 pair cutting."""

    def __init__(self, *, retention_ns: int | None = None) -> None:
        if retention_ns is not None and retention_ns <= 0:
            raise ValueError("retention_ns must be positive")
        self.retention_ns = retention_ns
        self._anchors: deque[InferenceAnchor] = deque()
        self._samples: deque[PolicyExecutionSample] = deque()
        self._latest_anchor_timestamp_ns: int | None = None
        self._latest_sample_timestamp_ns: int | None = None
        self._latest_observed_timestamp_ns: int | None = None
        self._lock = threading.RLock()

    def clear(self) -> None:
        """Drop all evidence at an explicit tablet episode boundary."""

        with self._lock:
            self._anchors.clear()
            self._samples.clear()
            self._latest_anchor_timestamp_ns = None
            self._latest_sample_timestamp_ns = None
            self._latest_observed_timestamp_ns = None

    @property
    def anchors(self) -> tuple[InferenceAnchor, ...]:
        with self._lock:
            return tuple(self._anchors)

    @property
    def samples(self) -> tuple[PolicyExecutionSample, ...]:
        with self._lock:
            return tuple(self._samples)

    @property
    def latest_sample(self) -> PolicyExecutionSample | None:
        """Return the immutable newest sample without copying episode history."""

        with self._lock:
            return None if not self._samples else self._samples[-1]

    def append_anchor(self, anchor: InferenceAnchor) -> None:
        with self._lock:
            if (
                self._latest_anchor_timestamp_ns is not None
                and anchor.timestamp_ns <= self._latest_anchor_timestamp_ns
            ):
                raise ValueError("FlowPRO inference-anchor timestamps must be strictly increasing")
            self._latest_anchor_timestamp_ns = anchor.timestamp_ns
            self._latest_observed_timestamp_ns = max(
                anchor.timestamp_ns, self._latest_observed_timestamp_ns or anchor.timestamp_ns
            )
            self._anchors.append(anchor)
            self._evict()

    def append_execution(self, sample: PolicyExecutionSample) -> None:
        with self._lock:
            if (
                self._latest_sample_timestamp_ns is not None
                and sample.timestamp_ns <= self._latest_sample_timestamp_ns
            ):
                raise ValueError("FlowPRO execution-sample timestamps must be strictly increasing")
            self._latest_sample_timestamp_ns = sample.timestamp_ns
            self._latest_observed_timestamp_ns = max(
                sample.timestamp_ns, self._latest_observed_timestamp_ns or sample.timestamp_ns
            )
            self._samples.append(sample)
            self._evict()

    def select_anchor(
        self,
        *,
        failure_timestamp_ns: int,
        requested_horizon_ns: int,
        max_anchor_snap_ns: int = DEFAULT_MAX_ANCHOR_SNAP_NS,
        max_rollback_ns: int = DEFAULT_MAX_ROLLBACK_NS,
    ) -> RollbackAnchorSelection:
        with self._lock:
            return select_rollback_anchor(
                tuple(self._anchors),
                failure_timestamp_ns=failure_timestamp_ns,
                requested_horizon_ns=requested_horizon_ns,
                max_anchor_snap_ns=max_anchor_snap_ns,
                max_rollback_ns=max_rollback_ns,
            )

    def select_oldest_anchor(
        self,
        *,
        failure_timestamp_ns: int,
        max_rollback_ns: int = DEFAULT_MAX_ROLLBACK_NS,
    ) -> RollbackAnchorSelection:
        with self._lock:
            return select_oldest_rollback_anchor(
                tuple(self._anchors),
                failure_timestamp_ns=failure_timestamp_ns,
                max_rollback_ns=max_rollback_ns,
            )

    def executed_rejected_segment(
        self,
        selection: RollbackAnchorSelection,
        *,
        native_action_dim: int,
        effector_profile: str,
        max_execution_gap_ns: int = DEFAULT_MAX_EXECUTION_GAP_NS,
        max_anchor_to_first_execution_ns: int = DEFAULT_MAX_ANCHOR_TO_FIRST_EXECUTION_NS,
    ) -> ExecutedRejectedSegment:
        """Build a paper-strict executed negative segment.

        This compatibility API retains the existing behavior: an interval
        with a missing execution span raises instead of producing a training
        segment.  Physical rewind must use :meth:`physical_rollback_segment`,
        which preserves the real samples and reports gaps as metadata.
        """

        segment = self.physical_rollback_segment(
            selection,
            native_action_dim=native_action_dim,
            effector_profile=effector_profile,
            max_execution_gap_ns=max_execution_gap_ns,
            max_anchor_to_first_execution_ns=max_anchor_to_first_execution_ns,
        )
        gap_kinds = {gap.kind for gap in segment.execution_gaps}
        if EXECUTION_GAP_BETWEEN_SAMPLES in gap_kinds:
            raise ValueError("Executed negative trajectory has non-monotonic timestamps or a history gap")
        if EXECUTION_GAP_ANCHOR_TO_FIRST in gap_kinds:
            raise ValueError("First Policy execution is too late after its inference anchor")
        if EXECUTION_GAP_LAST_TO_FAILURE in gap_kinds:
            raise ValueError("Executed negative trajectory does not cover the anchor-to-failure interval")
        return segment

    def physical_rollback_segment(
        self,
        selection: RollbackAnchorSelection,
        *,
        native_action_dim: int,
        effector_profile: str,
        max_execution_gap_ns: int = DEFAULT_MAX_EXECUTION_GAP_NS,
        max_anchor_to_first_execution_ns: int = DEFAULT_MAX_ANCHOR_TO_FIRST_EXECUTION_NS,
    ) -> ExecutedRejectedSegment:
        """Build the real sampled path used for physical rewind.

        Unlike :meth:`executed_rejected_segment`, this API preserves an older
        anchor and its real samples for diagnosis when accepted Policy evidence
        has a timing gap.  It attaches every strictness-breaking interval as
        ``execution_gaps`` and consequently marks the segment ineligible for
        paper-strict preference training.
        This method constructs evidence; it does not declare a gapped path safe
        to execute.  Production rollback must require
        ``segment.physical_history_complete``.  Dense, explicitly recorded
        ``queue_underrun_hold`` samples keep physical history complete while
        still invalidating strict preference training.
        """

        if native_action_dim not in {16, 24}:
            raise ValueError("native_action_dim must be 16 or 24")
        if max_execution_gap_ns <= 0 or max_anchor_to_first_execution_ns <= 0:
            raise ValueError("Execution-gap limits must be positive")
        with self._lock:
            samples = tuple(
                sample
                for sample in self._samples
                if selection.actual_anchor_timestamp_ns
                <= sample.timestamp_ns
                <= selection.failure_timestamp_ns
            )
            anchors = tuple(self._anchors)
        if not samples:
            raise ValueError("Rollback interval contains no execution samples")
        anchors_by_id = {anchor.observation_id: anchor for anchor in anchors}
        for sample in samples:
            if sample.control_owner != "policy" or not sample.command_accepted:
                raise ValueError("Executed negative trajectory contains non-Policy or unaccepted commands")
            source_anchor = anchors_by_id.get(sample.observation_id)
            if source_anchor is None:
                raise ValueError("Execution sample references an unknown inference anchor")
            if (
                source_anchor.generation_id != sample.generation_id
                or source_anchor.timestamp_ns > sample.timestamp_ns
            ):
                raise ValueError("Execution sample generation/observation lineage is inconsistent")
            if sample.policy_native_action.shape != (native_action_dim,):
                raise ValueError("Execution sample native action dimension differs from the contract")
            if abs(sample.measured_timestamp_ns - sample.timestamp_ns) > MAX_CAMERA_SKEW_NS:
                raise ValueError("Execution sample measured state is too far from its command timestamp")
        timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
        timestamp_diffs = np.diff(timestamps)
        if np.any(timestamp_diffs <= 0):
            raise ValueError("Executed negative trajectory has non-monotonic timestamps")
        execution_gaps: list[ExecutionHistoryGap] = []
        first_timestamp_ns = int(timestamps[0])
        last_timestamp_ns = int(timestamps[-1])
        if first_timestamp_ns - selection.actual_anchor_timestamp_ns > max_anchor_to_first_execution_ns:
            execution_gaps.append(
                ExecutionHistoryGap(
                    kind=EXECUTION_GAP_ANCHOR_TO_FIRST,
                    start_timestamp_ns=selection.actual_anchor_timestamp_ns,
                    end_timestamp_ns=first_timestamp_ns,
                    strict_limit_ns=max_anchor_to_first_execution_ns,
                )
            )
        for index, duration_ns in enumerate(timestamp_diffs):
            if int(duration_ns) > max_execution_gap_ns:
                execution_gaps.append(
                    ExecutionHistoryGap(
                        kind=EXECUTION_GAP_BETWEEN_SAMPLES,
                        start_timestamp_ns=int(timestamps[index]),
                        end_timestamp_ns=int(timestamps[index + 1]),
                        strict_limit_ns=max_execution_gap_ns,
                    )
                )
        if selection.failure_timestamp_ns - last_timestamp_ns > max_execution_gap_ns:
            execution_gaps.append(
                ExecutionHistoryGap(
                    kind=EXECUTION_GAP_LAST_TO_FAILURE,
                    start_timestamp_ns=last_timestamp_ns,
                    end_timestamp_ns=selection.failure_timestamp_ns,
                    strict_limit_ns=max_execution_gap_ns,
                )
            )
        measured24 = np.stack([sample.measured_state_24d for sample in samples])
        measured_model24 = get_effector_profile(effector_profile).hardware_targets_to_model_actions(
            measured24
        )
        generation_ids = tuple(dict.fromkeys(sample.generation_id for sample in samples))
        return ExecutedRejectedSegment(
            anchor=selection.anchor,
            failure_timestamp_ns=selection.failure_timestamp_ns,
            timestamps_ns=timestamps,
            measured_model_actions=measured_model24[:, :native_action_dim],
            samples=samples,
            generation_ids=generation_ids,
            safety_modified_count=sum(sample.safety_modified for sample in samples),
            execution_gaps=tuple(execution_gaps),
        )

    def _evict(self) -> None:
        assert self._latest_observed_timestamp_ns is not None
        if self.retention_ns is None:
            return
        cutoff = self._latest_observed_timestamp_ns - self.retention_ns
        while self._anchors and self._anchors[0].timestamp_ns < cutoff:
            self._anchors.popleft()
        while self._samples and self._samples[0].timestamp_ns < cutoff:
            self._samples.popleft()
