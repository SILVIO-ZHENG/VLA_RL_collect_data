"""Control-independent rollback planning and deadman execution for FlowPRO.

The planner reverses measured robot states, never historical commands.  The
executor is deliberately dependency-injected so offline tests can exercise
the complete ownership and safety contract without acquiring a GDK writer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..common.compat import StrEnum
from .executed import ExecutedRejectedSegment

ROLLBACK_MEDIAN_RADIUS = 2
ROLLBACK_MAX_KNOT_INTERVAL_NS = 500_000_000


def _vector24(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (24,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite [24] vector")
    return array.copy()


def _interpolate_timestamp_ns(start_timestamp_ns: int, end_timestamp_ns: int, phase: float) -> int:
    """Interpolate epoch nanoseconds without converting the large origin to float."""

    # Current wall-clock timestamps are around 1e18 ns, where float64 has a
    # granularity of hundreds of nanoseconds.  Adding the origin in floating
    # point can therefore round the first reverse waypoint *after* its start
    # and trip the fail-closed monotonicity gate.  Episode-scale deltas remain
    # exactly representable enough for interpolation, so keep the origin int.
    delta_ns = int(end_timestamp_ns) - int(start_timestamp_ns)
    return int(start_timestamp_ns) + int(round(delta_ns * phase))


def _stationary_connector(
    *,
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    target_position: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    servo_fps: float,
) -> tuple[np.ndarray, int]:
    """Plan a command-space connector with an explicit initial velocity.

    A preplanned path is checked by the final actuator limiter against its
    *last accepted command*, not measured feedback.  First brake that command
    velocity to zero with bounded discrete acceleration, then use a cubic
    zero-speed connector with conservative headroom.  The duplicate terminal
    waypoint makes the discrete final velocity exactly zero before the
    original rollback suffix is rejoined.
    """

    position = _vector24(start_position, "connector command position")
    velocity = _vector24(start_velocity, "connector command velocity")
    initial_position = position.copy()
    initial_velocity = velocity.copy()
    target = _vector24(target_position, "connector target position")
    velocity_limit = _vector24(max_velocity, "connector velocity limit")
    acceleration_limit = _vector24(max_acceleration, "connector acceleration limit")
    if not math.isfinite(servo_fps) or servo_fps <= 0:
        raise ValueError("Connector servo_fps must be positive and finite")
    if np.any(velocity_limit <= 0) or np.any(acceleration_limit <= 0):
        raise ValueError("Connector kinematic limits must be positive")
    if np.any(np.abs(velocity.astype(np.float64)) > velocity_limit.astype(np.float64) + 1e-5):
        raise ValueError("Connector initial command velocity exceeds its bound")

    dt = 1.0 / float(servo_fps)
    # Use the same headroom as the interpolation phase; braking exactly on a
    # float32 limit can round a 0.500000 command-space acceleration to
    # 0.500017 at 100 Hz.
    acceleration_step = np.asarray(acceleration_limit * dt * 0.8, dtype=np.float32)
    waypoints = [position.copy()]
    braking_steps = 0
    maximum_braking_steps = int(math.ceil(float(np.max(np.abs(velocity) / acceleration_step)))) + 2
    while np.any(velocity != 0.0):
        if braking_steps >= maximum_braking_steps:
            raise RuntimeError("Command-space connector could not brake to zero")
        velocity = np.asarray(
            velocity + np.clip(-velocity, -acceleration_step, acceleration_step),
            dtype=np.float32,
        )
        position = np.asarray(position + velocity * dt, dtype=np.float32)
        waypoints.append(position.copy())
        braking_steps += 1

    # Reserve 20% kinematic headroom for float32/GDK boundary conversions.
    margin = 0.8
    delta = np.abs(target.astype(np.float64) - position.astype(np.float64))
    duration_velocity = float(np.max(1.5 * delta / (velocity_limit * margin)))
    duration_acceleration = float(np.max(np.sqrt(6.0 * delta / (acceleration_limit * margin))))
    duration = max(dt, duration_velocity, duration_acceleration)
    interpolation_steps = max(1, int(math.ceil(duration * servo_fps)))
    interpolation_start = position.copy()
    for step in range(1, interpolation_steps + 1):
        phase = step / interpolation_steps
        blend = 3.0 * phase**2 - 2.0 * phase**3
        waypoints.append(
            np.asarray(
                interpolation_start + (target - interpolation_start) * blend,
                dtype=np.float32,
            )
        )
    # A sampled cubic has a small non-zero last interval even though its
    # analytic endpoint velocity is zero.  One repeated endpoint represents
    # the required discrete stop and is included in the acceleration audit.
    waypoints.append(target.copy())
    connector = np.ascontiguousarray(np.stack(waypoints), dtype=np.float32)

    previous_position = np.asarray(
        initial_position - initial_velocity * dt,
        dtype=np.float32,
    )
    with_boundary = np.concatenate((previous_position[None], connector), axis=0)
    observed_velocity = np.abs(np.diff(with_boundary.astype(np.float64), axis=0)) * servo_fps
    observed_acceleration = np.abs(np.diff(with_boundary.astype(np.float64), n=2, axis=0)) * servo_fps**2
    if len(observed_velocity) and np.any(observed_velocity > velocity_limit + 1e-5):
        raise RuntimeError("Internal command-space connector exceeded a velocity limit")
    if len(observed_acceleration) and np.any(observed_acceleration > acceleration_limit + 1e-5):
        raise RuntimeError("Internal command-space connector exceeded an acceleration limit")
    return connector, braking_steps


def _execution_knots(
    segment: ExecutedRejectedSegment,
    *,
    controlled_state: Callable[[object], np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build robust, bounded-gap knots from dense measured feedback.

    The raw 100 Hz measured trajectory remains untouched in the rejected-data
    artifact.  Physical rewind uses a short median filter to reject isolated
    feedback glitches, then keeps a path knot at least every 500 ms.  This
    avoids treating every noisy servo sample as a zero-velocity stop while
    retaining a bounded geometric approximation of the executed path.
    """

    samples = sorted(segment.samples, key=lambda sample: sample.measured_timestamp_ns)
    if not samples:
        return []
    raw = np.stack([controlled_state(sample.measured_state_24d) for sample in samples])
    filtered = np.empty_like(raw)
    for index in range(len(raw)):
        start = max(0, index - ROLLBACK_MEDIAN_RADIUS)
        stop = min(len(raw), index + ROLLBACK_MEDIAN_RADIUS + 1)
        filtered[index] = np.median(raw[start:stop], axis=0)

    dense = [(int(sample.measured_timestamp_ns), filtered[index]) for index, sample in enumerate(samples)]
    selected: list[tuple[int, np.ndarray]] = []
    newest_selected_ns: int | None = None
    for timestamp_ns, state in reversed(dense):
        if newest_selected_ns is None or newest_selected_ns - timestamp_ns >= ROLLBACK_MAX_KNOT_INTERVAL_NS:
            selected.append((timestamp_ns, state))
            newest_selected_ns = timestamp_ns
    return selected


@dataclass(frozen=True)
class RollbackPlanAnchor:
    """One inference anchor embedded in the immutable rollback plan."""

    observation_id: str
    source_timestamp_ns: int
    waypoint_index: int


@dataclass(frozen=True)
class RollbackTrajectory:
    intervention_id: str
    observation_id: str
    waypoints_24d: np.ndarray
    servo_fps: float
    source_sample_count: int
    duration_s: float
    collision_safety_enabled: bool
    safety_config_sha256: str | None
    kinematics_calibrated: bool
    waypoint_source_timestamps_ns: np.ndarray | None = None
    max_velocity_24d: np.ndarray | None = None
    max_acceleration_24d: np.ndarray | None = None
    controlled_action_dim: int = 24
    selectable_anchors: tuple[RollbackPlanAnchor, ...] = ()

    def __post_init__(self) -> None:
        waypoints = np.asarray(self.waypoints_24d, dtype=np.float32).copy()
        if waypoints.ndim != 2 or waypoints.shape[1:] != (24,) or len(waypoints) == 0:
            raise ValueError("Rollback waypoints must have shape [T, 24]")
        if not np.isfinite(waypoints).all():
            raise ValueError("Rollback waypoints must be finite")
        waypoints.flags.writeable = False
        object.__setattr__(self, "waypoints_24d", waypoints)
        if self.collision_safety_enabled:
            if not self.kinematics_calibrated:
                raise ValueError("Collision-checked rollback requires calibrated kinematics")
            if not isinstance(self.safety_config_sha256, str) or len(self.safety_config_sha256) != 64:
                raise ValueError("Collision-checked rollback requires a bound safety_config_sha256")
        elif self.safety_config_sha256 is not None:
            raise ValueError("Collision-disabled rollback must not claim a safety configuration")
        if self.controlled_action_dim not in {16, 24}:
            raise ValueError("Rollback controlled action dimension must be 16 or 24")
        if self.waypoint_source_timestamps_ns is not None:
            timestamps = np.asarray(self.waypoint_source_timestamps_ns, dtype=np.int64).copy()
            if timestamps.shape != (len(waypoints),):
                raise ValueError("Rollback waypoint source timestamps must match the plan length")
            if np.any(np.diff(timestamps) > 0):
                raise ValueError("Rollback waypoint source timestamps must be non-increasing")
            timestamps.flags.writeable = False
            object.__setattr__(self, "waypoint_source_timestamps_ns", timestamps)
        anchors = tuple(self.selectable_anchors)
        previous_index = -1
        seen_observation_ids: set[str] = set()
        for anchor in anchors:
            if not isinstance(anchor, RollbackPlanAnchor):
                raise TypeError("Rollback selectable anchors must be RollbackPlanAnchor values")
            if not anchor.observation_id or anchor.observation_id in seen_observation_ids:
                raise ValueError("Rollback selectable anchor observation IDs must be non-empty and unique")
            if not 0 <= anchor.waypoint_index < len(waypoints):
                raise ValueError("Rollback selectable anchor index is outside the plan")
            if anchor.waypoint_index <= previous_index:
                raise ValueError("Rollback selectable anchor indices must be strictly increasing")
            if self.waypoint_source_timestamps_ns is not None and (
                int(self.waypoint_source_timestamps_ns[anchor.waypoint_index]) != anchor.source_timestamp_ns
            ):
                raise ValueError("Rollback selectable anchor timestamp differs from its waypoint")
            previous_index = anchor.waypoint_index
            seen_observation_ids.add(anchor.observation_id)
        if anchors and (
            anchors[-1].observation_id != self.observation_id
            or anchors[-1].waypoint_index != len(waypoints) - 1
        ):
            raise ValueError("Rollback endpoint must be the final selectable inference anchor")
        object.__setattr__(self, "selectable_anchors", anchors)
        for field_name in ("max_velocity_24d", "max_acceleration_24d"):
            raw = getattr(self, field_name)
            if raw is None:
                continue
            limits = _vector24(raw, field_name)
            if np.any(limits <= 0):
                raise ValueError(f"{field_name} must be positive")
            limits.flags.writeable = False
            object.__setattr__(self, field_name, limits)


@dataclass(frozen=True)
class RollbackHealth:
    motion_healthy: bool
    measured_state_fresh: bool
    network_fresh: bool
    emergency_stop_active: bool = False
    collision_detected: bool = False
    human_takeover_active: bool = False
    # Receipt/evaluation time in the local monotonic clock domain.  Production
    # stamps every health snapshot immediately before handing it to the runtime
    # so a blocking prepare cannot turn old evidence into a new motion grant.
    evaluated_monotonic_ns: int | None = None

    @property
    def safe(self) -> bool:
        return (
            self.motion_healthy
            and self.measured_state_fresh
            and self.network_fresh
            and not self.emergency_stop_active
            and not self.collision_detected
            and not self.human_takeover_active
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return stable, field-specific diagnostics for a blocked rollback."""

        blockers: list[str] = []
        if not self.motion_healthy:
            blockers.append("motion_unhealthy")
        if not self.measured_state_fresh:
            blockers.append("measured_state_stale")
        if not self.network_fresh:
            blockers.append("network_stale")
        if self.emergency_stop_active:
            blockers.append("emergency_stop_active")
        if self.collision_detected:
            blockers.append("collision_detected")
        if self.human_takeover_active:
            blockers.append("human_takeover_active")
        return tuple(blockers)


class RollbackExecutorState(StrEnum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    HOLD = "hold"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAULT = "fault"


class RollbackDispatchPending(Exception):
    """A checked window is still pending; no rollback point was dispatched."""


@dataclass(frozen=True)
class RollbackStepResult:
    state: RollbackExecutorState
    waypoint_index: int
    command: np.ndarray
    reason: str
    measured_state: np.ndarray | None = None
    source_timestamp_ns: int | None = None
    anchor_observation_id: str | None = None


def plan_reverse_measured_trajectory(
    segment: ExecutedRejectedSegment,
    *,
    intervention_id: str,
    current_state_24d: object,
    servo_fps: float,
    max_velocity: object,
    max_acceleration: object,
    fixed_head_waist_8d: object | None = None,
    collision_safety_enabled: bool | None = None,
    safety_config_sha256: str | None = None,
    kinematics_calibrated: bool = False,
    path_guard: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> RollbackTrajectory:
    """Time-scale a reversed measured path with zero-velocity knots.

    Smoothstep interpolation is conservative: every historical knot is
    reached with zero velocity.  This can make rollback slower than forward
    execution, which is intentional for the first physical implementation.
    """

    # Never interpolate across an interval for which no accepted measured
    # evidence exists.  ``physical_rollback_segment`` may expose such a gap for
    # diagnostics, but only a complete path may reach the motion planner.
    if not segment.physical_history_complete:
        raise ValueError("Physical rollback history is incomplete due to an execution gap")
    if segment.generation_ids != (segment.anchor.generation_id,):
        raise ValueError("Physical rollback history crosses a Policy-run generation boundary")

    collision_checking = (
        safety_config_sha256 is not None
        if collision_safety_enabled is None
        else bool(collision_safety_enabled)
    )
    if collision_checking:
        if not kinematics_calibrated:
            raise ValueError("Collision-checked rollback requires calibrated kinematics/collision geometry")
        if not isinstance(safety_config_sha256, str) or len(safety_config_sha256) != 64:
            raise ValueError("Collision-checked rollback requires a bound safety_config_sha256")
    elif safety_config_sha256 is not None:
        raise ValueError("Collision-disabled rollback must not claim a safety_config_sha256")
    if not math.isfinite(servo_fps) or servo_fps <= 0:
        raise ValueError("servo_fps must be positive and finite")
    velocity = _vector24(max_velocity, "max_velocity")
    acceleration = _vector24(max_acceleration, "max_acceleration")
    if np.any(velocity <= 0) or np.any(acceleration <= 0):
        raise ValueError("Rollback velocity and acceleration limits must be positive")
    current = _vector24(current_state_24d, "current_state_24d")
    native_action_dim = int(segment.measured_model_actions.shape[1])
    if native_action_dim not in {16, 24}:
        raise ValueError("Rollback segment native action dimension must be 16 or 24")
    fixed_head_waist = None
    if native_action_dim == 16:
        fixed_head_waist = np.asarray(
            current[16:] if fixed_head_waist_8d is None else fixed_head_waist_8d,
            dtype=np.float32,
        )
        if fixed_head_waist.shape != (8,) or not np.isfinite(fixed_head_waist).all():
            raise ValueError("fixed_head_waist_8d must be a finite [8] vector")
        current[16:] = fixed_head_waist
    elif fixed_head_waist_8d is not None:
        raise ValueError("fixed_head_waist_8d is valid only for a 16D rollback segment")

    def controlled_state(state: object) -> np.ndarray:
        value = _vector24(state, "historical rollback state")
        if fixed_head_waist is not None:
            value[16:] = fixed_head_waist
        return value

    # Insert inference anchors as exact knots. This makes every selectable
    # release point physically reachable rather than merely close to the
    # historical measured samples around that timestamp.
    historical_knots = _execution_knots(segment, controlled_state=controlled_state)
    historical_knots.extend(
        (anchor.timestamp_ns, controlled_state(anchor.state_24d)) for anchor in segment.rollback_anchors
    )
    historical_knots.append((segment.anchor.timestamp_ns, controlled_state(segment.anchor.state_24d)))
    by_timestamp = {int(timestamp): state for timestamp, state in historical_knots}
    ordered_history = sorted(by_timestamp.items(), reverse=True)
    knots = [current, *(state for _timestamp, state in ordered_history)]
    knot_timestamps = [
        segment.failure_timestamp_ns,
        *(timestamp for timestamp, _state in ordered_history),
    ]
    anchor_by_timestamp = {
        int(anchor.timestamp_ns): anchor for anchor in (*segment.rollback_anchors, segment.anchor)
    }
    output = [knots[0].copy()]
    output_timestamps = [int(knot_timestamps[0])]
    selectable_anchors: list[RollbackPlanAnchor] = []
    for (start, start_timestamp), (end, end_timestamp) in zip(
        zip(knots, knot_timestamps, strict=True),
        zip(knots[1:], knot_timestamps[1:], strict=True),
        strict=False,
    ):
        delta = np.abs(end.astype(np.float64) - start.astype(np.float64))
        duration_velocity = float(np.max(1.5 * delta / velocity))
        duration_acceleration = float(np.max(np.sqrt(6.0 * delta / acceleration)))
        duration = max(1.0 / servo_fps, duration_velocity, duration_acceleration)
        steps = max(1, int(math.ceil(duration * servo_fps)))
        for step in range(1, steps + 1):
            phase = step / steps
            blend = 3.0 * phase**2 - 2.0 * phase**3
            waypoint = np.asarray(start + (end - start) * blend, dtype=np.float32)
            if path_guard is not None and not path_guard(output[-1], waypoint):
                raise ValueError("Rollback path guard rejected a planned segment")
            output.append(waypoint)
            output_timestamps.append(_interpolate_timestamp_ns(start_timestamp, end_timestamp, phase))
        anchor = anchor_by_timestamp.get(int(end_timestamp))
        if anchor is not None:
            # One repeated endpoint makes this inference anchor a discrete
            # zero-command-velocity boundary in the immutable main plan.  X-up
            # can therefore finish to the anchor without inventing a second
            # connector trajectory, swapping plans, or replaying a prefix.
            output.append(output[-1].copy())
            output_timestamps.append(int(end_timestamp))
            selectable_anchors.append(
                RollbackPlanAnchor(
                    observation_id=anchor.observation_id,
                    source_timestamp_ns=int(anchor.timestamp_ns),
                    waypoint_index=len(output) - 1,
                )
            )
    waypoints = np.ascontiguousarray(np.stack(output), dtype=np.float32)
    source_timestamps = np.ascontiguousarray(output_timestamps, dtype=np.int64)
    if np.any(np.diff(source_timestamps) > 0):
        raise RuntimeError("Internal rollback planner produced forward-moving source timestamps")
    velocity_observed = np.abs(np.diff(waypoints.astype(np.float64), axis=0)) * servo_fps
    if len(velocity_observed) and np.any(velocity_observed > velocity + 1e-5):
        raise RuntimeError("Internal rollback planner exceeded a velocity limit")
    acceleration_observed = np.abs(np.diff(waypoints.astype(np.float64), n=2, axis=0)) * servo_fps**2
    if len(acceleration_observed) and np.any(acceleration_observed > acceleration + 1e-4):
        raise RuntimeError("Internal rollback planner exceeded an acceleration limit")
    return RollbackTrajectory(
        intervention_id=intervention_id,
        observation_id=segment.anchor.observation_id,
        waypoints_24d=waypoints,
        servo_fps=float(servo_fps),
        source_sample_count=len(segment.samples),
        duration_s=(len(waypoints) - 1) / servo_fps,
        collision_safety_enabled=collision_checking,
        safety_config_sha256=safety_config_sha256,
        kinematics_calibrated=bool(kinematics_calibrated),
        waypoint_source_timestamps_ns=source_timestamps,
        max_velocity_24d=velocity,
        max_acceleration_24d=acceleration,
        controlled_action_dim=native_action_dim,
        selectable_anchors=tuple(selectable_anchors),
    )


class RollbackExecutor:
    """Deadman-gated consumer that must share the robot's sole command writer."""

    # Commands pass through several float32-backed GDK/runtime boundaries.  At
    # joint magnitudes near one radian, one float32 ULP is already larger than
    # 1e-7.  Match the adapter's preplanned-path tolerance: this is about
    # 0.00057 degrees, while GDK ACK, hard limits and kinematic limits remain
    # independently enforced.
    _COMMAND_NUMERIC_ATOL = 1e-5

    def __init__(
        self,
        *,
        read_measured: Callable[[], np.ndarray],
        read_command_motion: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
        safety_project: Callable[[np.ndarray, np.ndarray], np.ndarray],
        send_command: Callable[[np.ndarray], np.ndarray],
        send_hold_command: Callable[[np.ndarray], np.ndarray] | None = None,
        endpoint_tolerance: float = 0.01,
        resume_rejoin_max_error: float = 0.05,
        stationary_settle_timeout_s: float = 2.0,
    ) -> None:
        self.read_measured = read_measured
        self.read_command_motion = read_command_motion
        self.safety_project = safety_project
        self.send_command = send_command
        self.send_hold_command = send_command if send_hold_command is None else send_hold_command
        self.endpoint_tolerance = float(endpoint_tolerance)
        self.resume_rejoin_max_error = float(resume_rejoin_max_error)
        self.stationary_settle_timeout_s = float(stationary_settle_timeout_s)
        if not math.isfinite(self.resume_rejoin_max_error) or self.resume_rejoin_max_error <= 0:
            raise ValueError("resume_rejoin_max_error must be positive and finite")
        if self.resume_rejoin_max_error < self.endpoint_tolerance:
            raise ValueError("resume_rejoin_max_error must not be smaller than endpoint_tolerance")
        if not math.isfinite(self.stationary_settle_timeout_s) or self.stationary_settle_timeout_s <= 0:
            raise ValueError("stationary_settle_timeout_s must be positive and finite")
        self.state = RollbackExecutorState.IDLE
        self.plan: RollbackTrajectory | None = None
        self.lease_id: str | None = None
        self.index = 0
        self._paused_state: np.ndarray | None = None
        self._resume_bridge_waypoints: np.ndarray | None = None
        self._resume_bridge_timestamps_ns: np.ndarray | None = None
        self._resume_bridge_index = 0
        self._resume_join_index: int | None = None
        self._fixed_passive_state: np.ndarray | None = None
        self._stationary_hold_target: np.ndarray | None = None
        self._finish_to_anchor: RollbackPlanAnchor | None = None
        # Highest inference anchor at which this immutable plan has already
        # returned an operator-visible HOLD.  It survives a paused rebind so a
        # second X release cannot select the same anchor again.
        self._last_paused_anchor_index = -1
        self._continue_after_stationary_hold = False
        self._stationary_settle_ticks = 0

    def _clear_resume_bridge(self) -> None:
        self._resume_bridge_waypoints = None
        self._resume_bridge_timestamps_ns = None
        self._resume_bridge_index = 0
        self._resume_join_index = None

    def _target_with_passive_hold(self, target: np.ndarray) -> np.ndarray:
        value = np.asarray(target, dtype=np.float32).copy()
        if self.plan is not None and self.plan.controlled_action_dim == 16:
            assert self._fixed_passive_state is not None
            value[16:] = self._fixed_passive_state
        return value

    def _effective_progress_index(self) -> int:
        """Return action-thread-owned progress, including an active rejoin."""

        if self._resume_join_index is not None:
            return max(self.index, self._resume_join_index)
        return self.index

    def _select_finish_anchor(self) -> RollbackPlanAnchor:
        """Select the nearest reachable anchor not already used as a pause.

        On a first run, X may be released exactly when the cursor reaches an
        anchor and that anchor remains a valid stopping point.  After the
        executor has actually returned HOLD at that anchor, however, a fresh
        lease must skip it; otherwise a second X press only traverses its
        resume bridge and immediately stops at the old endpoint.
        """

        assert self.plan is not None
        progress = self._effective_progress_index()
        for anchor in self.plan.selectable_anchors:
            if (
                anchor.waypoint_index >= progress
                and anchor.waypoint_index > self._last_paused_anchor_index
            ):
                return anchor
        # Compatibility for explicitly constructed/offline plans that predate
        # selectable anchor metadata. Production planner output always carries
        # the exact observation/timestamp/index tuple.
        if not self.plan.selectable_anchors and progress <= len(self.plan.waypoints_24d) - 1:
            timestamp = (
                0
                if self.plan.waypoint_source_timestamps_ns is None
                else int(self.plan.waypoint_source_timestamps_ns[-1])
            )
            return RollbackPlanAnchor(
                observation_id=self.plan.observation_id,
                source_timestamp_ns=timestamp,
                waypoint_index=len(self.plan.waypoints_24d) - 1,
            )
        raise RuntimeError("Rollback has no selectable inference anchor in its untouched suffix")

    def _anchor_at_index(self, index: int) -> RollbackPlanAnchor | None:
        assert self.plan is not None
        return next(
            (anchor for anchor in self.plan.selectable_anchors if anchor.waypoint_index == index),
            None,
        )

    def _begin_stationary_hold(self, target: object) -> None:
        self._stationary_hold_target = _vector24(target, "rollback stationary target")
        self._stationary_settle_ticks = 0

    @classmethod
    def _command_matches(
        cls,
        sent: np.ndarray,
        requested: np.ndarray,
        controlled_dim: int,
    ) -> bool:
        return bool(
            np.allclose(
                sent[:controlled_dim],
                requested[:controlled_dim],
                rtol=0.0,
                atol=cls._COMMAND_NUMERIC_ATOL,
            )
        )

    @classmethod
    def _command_mismatch_reason(
        cls,
        prefix: str,
        sent: np.ndarray,
        requested: np.ndarray,
        controlled_dim: int,
    ) -> str:
        error = np.abs(
            sent[:controlled_dim].astype(np.float64) - requested[:controlled_dim].astype(np.float64)
        )
        index = int(np.argmax(error))
        return (
            f"{prefix}:max_error={float(error[index]):.9g}:index={index}:"
            f"requested={float(requested[index]):.9g}:sent={float(sent[index]):.9g}:"
            f"tolerance={cls._COMMAND_NUMERIC_ATOL:.9g}"
        )

    def _current_source_timestamp_ns(self) -> int | None:
        if self._resume_bridge_timestamps_ns is not None and self._resume_bridge_waypoints is not None:
            return int(self._resume_bridge_timestamps_ns[self._resume_bridge_index])
        if self.plan is None or self.plan.waypoint_source_timestamps_ns is None:
            return None
        return int(self.plan.waypoint_source_timestamps_ns[self.index])

    def _command_motion(
        self,
        measured: np.ndarray,
        *,
        command_position_24d: object | None = None,
        command_velocity_24d: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if command_position_24d is None and command_velocity_24d is None:
            if self.read_command_motion is None:
                return measured.copy(), np.zeros(24, dtype=np.float32)
            command_position_24d, command_velocity_24d = self.read_command_motion()
        elif command_position_24d is None or command_velocity_24d is None:
            raise ValueError("Command position and velocity must be supplied together")
        return (
            _vector24(command_position_24d, "rollback command position"),
            _vector24(command_velocity_24d, "rollback command velocity"),
        )

    def _build_resume_bridge(
        self,
        measured: np.ndarray,
        *,
        command_position_24d: object | None = None,
        command_velocity_24d: object | None = None,
        required_join_index: int | None = None,
    ) -> dict[str, float | int]:
        """Join the exact actuator command boundary to the untouched suffix.

        Measured feedback remains the physical drift/safety signal, but it is
        not the finite-difference origin used by ``adopt_preplanned``.
        """

        assert self.plan is not None
        if self.plan.max_velocity_24d is None or self.plan.max_acceleration_24d is None:
            raise ValueError("Paused rollback resume requires bound velocity and acceleration limits")
        waypoints = self.plan.waypoints_24d
        timestamps = self.plan.waypoint_source_timestamps_ns
        velocity = self.plan.max_velocity_24d.astype(np.float64)
        acceleration = self.plan.max_acceleration_24d.astype(np.float64)
        fps = float(self.plan.servo_fps)
        controlled_dim = self.plan.controlled_action_dim
        drift = np.abs(measured[:controlled_dim] - self._paused_state[:controlled_dim])
        max_drift = float(np.max(drift))
        if max_drift > self.resume_rejoin_max_error:
            raise ValueError(
                "Robot moved too far while rollback was paused: "
                f"max_error={max_drift:.5f} tolerance={self.resume_rejoin_max_error:.5f}"
            )

        command_position, command_velocity = self._command_motion(
            measured,
            command_position_24d=command_position_24d,
            command_velocity_24d=command_velocity_24d,
        )
        command_feedback_gap = float(
            np.max(np.abs(command_position[:controlled_dim] - measured[:controlled_dim]))
        )
        if command_feedback_gap > self.resume_rejoin_max_error:
            raise ValueError(
                "Rollback command and feedback differ too far for a safe rejoin: "
                f"max_error={command_feedback_gap:.5f} "
                f"tolerance={self.resume_rejoin_max_error:.5f}"
            )
        command_speed = float(np.max(np.abs(command_velocity)))
        # Do not require an unrealistically exact zero from the command-motion
        # snapshot.  `_stationary_connector` consumes the exact position and
        # velocity boundary, inserts bounded braking waypoints, and the final
        # GDK limiter validates every 100 Hz step.  This is especially important
        # for 16D operation where passive head/waist are only being held.
        if controlled_dim == 16:
            self._fixed_passive_state = command_position[16:].copy()

        # If a previous rejoin bridge was interrupted, continue toward that
        # same suffix point. Otherwise find the first future zero-speed knot.
        # The original planner creates such a knot at least every 500 ms and
        # at every inference anchor.
        first_candidate = self._resume_join_index if self._resume_join_index is not None else self.index
        first_candidate = max(self.index, int(first_candidate))
        if required_join_index is not None:
            required_join_index = int(required_join_index)
            if not self.index <= required_join_index < len(waypoints):
                raise ValueError("Required rollback rejoin index is outside the untouched suffix")
            first_candidate = required_join_index
            last_candidate = required_join_index + 1
        else:
            last_candidate = len(waypoints)
        selected: tuple[int, np.ndarray, np.ndarray | None] | None = None
        selected_braking_steps = 0
        for join_index in range(first_candidate, last_candidate):
            target = self._target_with_passive_hold(waypoints[join_index])
            bridge, braking_steps = _stationary_connector(
                start_position=command_position,
                start_velocity=command_velocity,
                target_position=target,
                max_velocity=velocity,
                max_acceleration=acceleration,
                servo_fps=fps,
            )
            # Validate the bridge and its join to the original suffix. A
            # smoothstep ends at zero velocity, so an arbitrary mid-curve
            # join can violate acceleration even if both paths are safe alone.
            tail = np.asarray(
                [self._target_with_passive_hold(item) for item in waypoints[join_index + 1 : join_index + 3]],
                dtype=np.float32,
            )
            previous_position = np.asarray(
                command_position - command_velocity / fps,
                dtype=np.float32,
            )
            combined_parts = [previous_position[None], bridge]
            if len(tail):
                combined_parts.append(tail)
            combined = np.concatenate(combined_parts, axis=0)
            observed_velocity = np.abs(np.diff(combined.astype(np.float64), axis=0)) * fps
            observed_acceleration = np.abs(np.diff(combined.astype(np.float64), n=2, axis=0)) * fps**2
            if len(observed_velocity) and np.any(observed_velocity > velocity + 1e-5):
                continue
            if len(observed_acceleration) and np.any(observed_acceleration > acceleration + 1e-5):
                continue
            bridge_timestamps = None
            if timestamps is not None:
                start_timestamp = self._current_source_timestamp_ns()
                assert start_timestamp is not None
                end_timestamp = int(timestamps[join_index])
                steps = len(bridge) - 1
                bridge_timestamps = np.asarray(
                    [
                        _interpolate_timestamp_ns(start_timestamp, end_timestamp, step / steps)
                        for step in range(steps + 1)
                    ],
                    dtype=np.int64,
                )
                if np.any(np.diff(bridge_timestamps) > 0):
                    raise RuntimeError("Rollback resume bridge moved forward in history time")
            selected = join_index, bridge, bridge_timestamps
            selected_braking_steps = braking_steps
            break
        if selected is None:
            raise ValueError("No acceleration-safe suffix rejoin exists for paused rollback")

        join_index, bridge, bridge_timestamps = selected
        self._resume_bridge_waypoints = bridge
        self._resume_bridge_timestamps_ns = bridge_timestamps
        self._resume_bridge_index = 0
        self._resume_join_index = join_index
        return {
            "resume_from_cursor": int(self.index),
            "resume_join_cursor": int(join_index),
            "resume_bridge_waypoint_count": int(len(bridge)),
            "resume_bridge_duration_s": float((len(bridge) - 1) / fps),
            "resume_measured_drift_max_abs": max_drift,
            "resume_command_feedback_gap_max_abs": command_feedback_gap,
            "resume_initial_command_speed_max_abs": command_speed,
            "resume_braking_waypoint_count": int(selected_braking_steps),
        }

    def arm(
        self,
        plan: RollbackTrajectory,
        *,
        lease_id: str,
        measured_state_24d: object | None = None,
        command_position_24d: object | None = None,
        command_velocity_24d: object | None = None,
    ) -> dict[str, float | int] | None:
        if self.state not in {RollbackExecutorState.IDLE, RollbackExecutorState.COMPLETE}:
            raise RuntimeError("Rollback executor is already armed")
        if not lease_id:
            raise ValueError("Rollback exclusive lease is required")
        if plan.collision_safety_enabled and not plan.kinematics_calibrated:
            raise ValueError("Collision-checked rollback requires calibrated kinematics")
        measured = _vector24(
            self.read_measured() if measured_state_24d is None else measured_state_24d,
            "measured rollback start",
        )
        start_error = np.abs(
            measured[: plan.controlled_action_dim] - plan.waypoints_24d[0, : plan.controlled_action_dim]
        )
        if np.max(start_error) > self.resume_rejoin_max_error:
            index = int(np.argmax(start_error))
            raise ValueError(
                "Current measured state differs from the rollback plan start: "
                f"max_error={float(start_error[index]):.5f} index={index} "
                f"tolerance={self.resume_rejoin_max_error:.5f}"
            )
        self.plan = plan
        self.lease_id = lease_id
        self.index = 0
        self._paused_state = measured.copy()
        self._clear_resume_bridge()
        command_position, command_velocity = self._command_motion(
            measured,
            command_position_24d=command_position_24d,
            command_velocity_24d=command_velocity_24d,
        )
        self._fixed_passive_state = command_position[16:].copy() if plan.controlled_action_dim == 16 else None
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._last_paused_anchor_index = -1
        self._continue_after_stationary_hold = False
        # The planner ran off-thread, so always join the *live* final-limiter
        # boundary back to plan[0] before consuming historical suffix. A
        # command/plan error smaller than the generic 1e-5 acceptance tolerance
        # can still exceed one low-acceleration gripper tick at 100 Hz; deciding
        # conditionally would therefore recreate the original first-frame bug.
        # Requiring index 0 also avoids silently skipping newest reverse motion.
        resume_detail = self._build_resume_bridge(
            measured,
            command_position_24d=command_position,
            command_velocity_24d=command_velocity,
            required_join_index=0,
        )
        self.state = RollbackExecutorState.READY
        return resume_detail

    def rebind_paused(
        self,
        *,
        lease_id: str,
        measured_state_24d: object | None = None,
        command_position_24d: object | None = None,
        command_velocity_24d: object | None = None,
    ) -> dict[str, float | int] | None:
        """Grant a fresh execution lease while preserving the rollback cursor."""

        if self.state != RollbackExecutorState.HOLD or self.plan is None or self._paused_state is None:
            raise RuntimeError("Only a paused rollback may be resumed")
        if not lease_id:
            raise ValueError("A fresh rollback lease is required")
        if lease_id == self.lease_id:
            raise ValueError("Paused rollback must resume under a fresh lease")
        measured = _vector24(
            self.read_measured() if measured_state_24d is None else measured_state_24d,
            "measured rollback resume",
        )
        # Rejoin from the exact command-space HOLD. Even sub-tolerance
        # command/feedback error can exceed a single 100 Hz acceleration step;
        # measured feedback is therefore used only for the drift gate. The
        # bridge stays inside the untouched suffix and never revisits the
        # already rewound prefix.
        resume_detail = self._build_resume_bridge(
            measured,
            command_position_24d=command_position_24d,
            command_velocity_24d=command_velocity_24d,
            required_join_index=self.index,
        )
        self.lease_id = lease_id
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._continue_after_stationary_hold = False
        self.state = RollbackExecutorState.READY
        return resume_detail

    def discard_paused(self) -> None:
        """Discard an already-held plan without issuing another command."""

        if self.state not in {
            RollbackExecutorState.READY,
            RollbackExecutorState.HOLD,
            RollbackExecutorState.CANCELLED,
            RollbackExecutorState.FAULT,
        }:
            raise RuntimeError("Running rollback must be stopped by the action thread before discard")
        self.plan = None
        self.lease_id = None
        self.index = 0
        self._paused_state = None
        self._clear_resume_bridge()
        self._fixed_passive_state = None
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._last_paused_anchor_index = -1
        self._continue_after_stationary_hold = False
        self.state = RollbackExecutorState.IDLE

    def abandon_for_human(self) -> None:
        """Drop any rollback state after vendor Human authority is confirmed.

        Unlike ``cancel()``, this deliberately sends no local HOLD command:
        vendor arbitration already owns motion, and the action thread is being
        suspended. Keeping a cancellation mailbox for that suspended thread
        would strand the old lease and block the later measured-HOLD handback.
        """

        self.plan = None
        self.lease_id = None
        self.index = 0
        self._paused_state = None
        self._clear_resume_bridge()
        self._fixed_passive_state = None
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._last_paused_anchor_index = -1
        self._continue_after_stationary_hold = False
        self.state = RollbackExecutorState.IDLE

    def disarm(self) -> None:
        """Release a completed lease so a later intervention can arm safely."""

        if self.state != RollbackExecutorState.COMPLETE:
            raise RuntimeError("Only a completed rollback may be disarmed")
        self.plan = None
        self.lease_id = None
        self.index = 0
        self._paused_state = None
        self._clear_resume_bridge()
        self._fixed_passive_state = None
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._last_paused_anchor_index = -1
        self._continue_after_stationary_hold = False
        self.state = RollbackExecutorState.IDLE

    def step(
        self,
        *,
        deadman_pressed: bool,
        finish_to_anchor_authorized: bool = False,
        health: RollbackHealth,
        lease_id: str,
    ) -> RollbackStepResult:
        if self.plan is None or self.lease_id is None:
            raise RuntimeError("Rollback executor is not armed")
        if self.state == RollbackExecutorState.HOLD:
            raise RuntimeError("Paused rollback requires a fresh lease before execution")
        measured = _vector24(self.read_measured(), "rollback measured state")
        if lease_id != self.lease_id:
            return self._fault_hold(measured, "rollback_lease_lost")
        if not health.safe:
            blockers = tuple(getattr(health, "blockers", ())) or ("unspecified",)
            return self._fault_hold(
                measured,
                "rollback_health_gate_failed:" + ",".join(blockers),
            )

        # A re-press while the automatic suffix is still moving simply returns
        # control to the held-X mode on the same plan, cursor, and lease.  If
        # the anchor is already in its final limiter braking phase, finish the
        # stationary boundary first and then continue automatically; this
        # avoids needing a third button press.
        if deadman_pressed:
            if self._stationary_hold_target is not None and self._finish_to_anchor is not None:
                self._continue_after_stationary_hold = True
            elif self._stationary_hold_target is None:
                self._finish_to_anchor = None
        elif finish_to_anchor_authorized:
            self._continue_after_stationary_hold = False
            if self._finish_to_anchor is None:
                try:
                    self._finish_to_anchor = self._select_finish_anchor()
                except RuntimeError:
                    return self._fault_hold(measured, "rollback_no_older_inference_anchor")
        else:
            # This path is only for an explicit authorization revocation. A
            # normal X-up always sets finish_to_anchor_authorized and follows
            # the immutable historical suffix to an inference anchor.
            self._continue_after_stationary_hold = False
            if self._stationary_hold_target is None:
                self._begin_stationary_hold(measured)
                self._paused_state = measured.copy()

        if self._stationary_hold_target is not None:
            return self._settle_stationary_target(measured)
        if self._resume_bridge_waypoints is not None:
            target = self._resume_bridge_waypoints[self._resume_bridge_index + 1]
            safe = _vector24(self.safety_project(measured, target), "rollback resume safety target")
            if not np.allclose(safe, target, rtol=0.0, atol=1e-7):
                return self._fault_hold(measured, "rollback_resume_safety_projection_modified_path")
            try:
                sent = _vector24(self.send_command(safe), "rollback resume sent command")
            except RollbackDispatchPending:
                return self._pending_dispatch_result(measured)
            controlled_dim = self.plan.controlled_action_dim
            if not self._command_matches(sent, safe, controlled_dim):
                return self._fault_hold(
                    measured,
                    self._command_mismatch_reason(
                        "rollback_resume_command_not_accepted_exactly",
                        sent,
                        safe,
                        controlled_dim,
                    ),
                )
            self._resume_bridge_index += 1
            source_timestamp_ns = self._current_source_timestamp_ns()
            if self._resume_bridge_index >= len(self._resume_bridge_waypoints) - 1:
                assert self._resume_join_index is not None
                self.index = self._resume_join_index
                self._clear_resume_bridge()
            self.state = RollbackExecutorState.RUNNING
            return self._result(
                self.state,
                sent,
                "rollback_resume_bridge_waypoint_sent",
                measured,
                source_timestamp_ns=source_timestamp_ns,
            )
        if self.index >= len(self.plan.waypoints_24d) - 1:
            endpoint = self._target_with_passive_hold(self.plan.waypoints_24d[-1])
            if self._finish_to_anchor is None:
                try:
                    self._finish_to_anchor = self._select_finish_anchor()
                except RuntimeError:
                    self._finish_to_anchor = RollbackPlanAnchor(
                        observation_id=self.plan.observation_id,
                        source_timestamp_ns=(
                            0
                            if self.plan.waypoint_source_timestamps_ns is None
                            else int(self.plan.waypoint_source_timestamps_ns[-1])
                        ),
                        waypoint_index=len(self.plan.waypoints_24d) - 1,
                    )
            self._begin_stationary_hold(endpoint)
            return self._settle_stationary_target(measured)
        if self._finish_to_anchor is not None and self.index >= self._finish_to_anchor.waypoint_index:
            self._begin_stationary_hold(
                self._target_with_passive_hold(self.plan.waypoints_24d[self._finish_to_anchor.waypoint_index])
            )
            return self._settle_stationary_target(measured)
        target = self._target_with_passive_hold(self.plan.waypoints_24d[self.index + 1])
        safe = _vector24(self.safety_project(measured, target), "rollback safety target")
        if not np.allclose(safe, target, rtol=0.0, atol=1e-7):
            return self._fault_hold(measured, "rollback_safety_projection_modified_path")
        try:
            sent = _vector24(self.send_command(safe), "rollback sent command")
        except RollbackDispatchPending:
            return self._pending_dispatch_result(measured)
        controlled_dim = self.plan.controlled_action_dim
        if not self._command_matches(sent, safe, controlled_dim):
            return self._fault_hold(
                measured,
                self._command_mismatch_reason(
                    "rollback_command_not_accepted_exactly",
                    sent,
                    safe,
                    controlled_dim,
                ),
            )
        measured_source_index = self.index
        self.index += 1
        self.state = RollbackExecutorState.RUNNING
        return self._result(
            self.state,
            sent,
            "rollback_waypoint_sent",
            measured,
            source_index=measured_source_index,
        )

    def _settle_stationary_target(self, measured: np.ndarray) -> RollbackStepResult:
        """Keep the 100 Hz heartbeat until the selected plan anchor is still."""

        assert self.plan is not None and self._stationary_hold_target is not None
        target = self._stationary_hold_target
        source_timestamp_ns = self._current_source_timestamp_ns()
        try:
            sent = _vector24(self.send_hold_command(target), "rollback stationary command")
        except RollbackDispatchPending:
            return self._pending_dispatch_result(measured)
        command_position, command_velocity = self._command_motion(measured)
        controlled_dim = self.plan.controlled_action_dim
        stopped = bool(
            np.max(np.abs(command_velocity)) <= 1e-7
            and np.allclose(
                command_position,
                sent,
                rtol=0.0,
                atol=self._COMMAND_NUMERIC_ATOL,
            )
            and np.max(np.abs(measured[:controlled_dim] - target[:controlled_dim])) <= self.endpoint_tolerance
        )
        anchor = self._finish_to_anchor
        if anchor is not None and not self._command_matches(sent, target, controlled_dim):
            return self._fault_hold(
                measured,
                self._command_mismatch_reason(
                    "rollback_anchor_hold_not_accepted_exactly",
                    sent,
                    target,
                    controlled_dim,
                ),
            )
        if not stopped:
            self._stationary_settle_ticks += 1
            if self._stationary_settle_ticks > math.ceil(
                self.stationary_settle_timeout_s * self.plan.servo_fps
            ):
                return self._fault_hold(measured, "rollback_anchor_settle_timeout")
            self.state = RollbackExecutorState.RUNNING
            return self._result(
                self.state,
                sent,
                "rollback_anchor_settling" if anchor is not None else "rollback_authorization_braking",
                measured,
                source_timestamp_ns=source_timestamp_ns,
                anchor_observation_id=None if anchor is None else anchor.observation_id,
            )

        self._paused_state = measured.copy()
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        at_endpoint = self.index >= len(self.plan.waypoints_24d) - 1
        if self._continue_after_stationary_hold and not at_endpoint:
            self._continue_after_stationary_hold = False
            self._finish_to_anchor = None
            self.state = RollbackExecutorState.RUNNING
            return self._result(
                self.state,
                sent,
                "rollback_anchor_settled_continue_requested",
                measured,
                source_timestamp_ns=source_timestamp_ns,
            )

        self._continue_after_stationary_hold = False
        self.state = RollbackExecutorState.COMPLETE if at_endpoint else RollbackExecutorState.HOLD
        if anchor is not None and self.state == RollbackExecutorState.HOLD:
            self._last_paused_anchor_index = max(
                self._last_paused_anchor_index,
                anchor.waypoint_index,
            )
        reason = "rollback_complete" if at_endpoint else "rollback_anchor_reached_stationary"
        result = self._result(
            self.state,
            sent,
            reason,
            measured,
            source_timestamp_ns=source_timestamp_ns,
            anchor_observation_id=None if anchor is None else anchor.observation_id,
        )
        self._finish_to_anchor = None
        return result

    def cancel(self, reason: str) -> RollbackStepResult:
        """Action-thread cancellation: command measured HOLD exactly once."""

        if self.plan is None or not reason:
            raise RuntimeError("Rollback cancellation requires an armed plan and reason")
        measured = _vector24(self.read_measured(), "rollback cancel measured state")
        sent = _vector24(self.send_hold_command(measured), "rollback cancel hold")
        self._paused_state = measured.copy()
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._continue_after_stationary_hold = False
        self.state = RollbackExecutorState.CANCELLED
        return self._result(self.state, sent, reason, measured)

    def maintain_hold(
        self,
        *,
        health: RollbackHealth,
        lease_id: str,
    ) -> RollbackStepResult:
        """Send the stationary anchor heartbeat while its HOLD ACK is pending."""

        if self.state != RollbackExecutorState.HOLD or self.plan is None or self.lease_id is None:
            raise RuntimeError("Rollback stationary heartbeat requires an armed HOLD")
        measured = _vector24(self.read_measured(), "rollback hold measured state")
        if lease_id != self.lease_id:
            return self._fault_hold(measured, "rollback_lease_lost_during_hold")
        if not health.safe:
            blockers = tuple(getattr(health, "blockers", ())) or ("unspecified",)
            return self._fault_hold(
                measured,
                "rollback_health_gate_failed_during_hold:" + ",".join(blockers),
            )
        target = self._target_with_passive_hold(self.plan.waypoints_24d[self.index])
        controlled_dim = self.plan.controlled_action_dim
        if np.max(np.abs(measured[:controlled_dim] - target[:controlled_dim])) > self.endpoint_tolerance:
            return self._fault_hold(measured, "rollback_anchor_drift_during_hold")
        safe = _vector24(self.safety_project(measured, target), "rollback hold safety target")
        if not np.allclose(safe, target, rtol=0.0, atol=1e-7):
            return self._fault_hold(measured, "rollback_hold_safety_projection_modified_target")
        sent = _vector24(self.send_hold_command(safe), "rollback hold heartbeat")
        if not self._command_matches(sent, safe, controlled_dim):
            return self._fault_hold(
                measured,
                self._command_mismatch_reason(
                    "rollback_hold_command_not_accepted_exactly",
                    sent,
                    safe,
                    controlled_dim,
                ),
            )
        anchor = self._anchor_at_index(self.index)
        return self._result(
            self.state,
            sent,
            "rollback_anchor_hold_heartbeat",
            measured,
            anchor_observation_id=None if anchor is None else anchor.observation_id,
        )

    def _result(
        self,
        state: RollbackExecutorState,
        command: np.ndarray,
        reason: str,
        measured: np.ndarray,
        *,
        source_index: int | None = None,
        source_timestamp_ns: int | None = None,
        anchor_observation_id: str | None = None,
    ) -> RollbackStepResult:
        if (
            source_timestamp_ns is None
            and self.plan is not None
            and self.plan.waypoint_source_timestamps_ns is not None
        ):
            index = self.index if source_index is None else source_index
            source_timestamp_ns = int(self.plan.waypoint_source_timestamps_ns[index])
        return RollbackStepResult(
            state,
            self.index,
            command,
            reason,
            measured_state=measured.copy(),
            source_timestamp_ns=source_timestamp_ns,
            anchor_observation_id=anchor_observation_id,
        )

    def _fault_hold(self, measured: np.ndarray, reason: str) -> RollbackStepResult:
        sent = _vector24(self.send_hold_command(measured), "rollback fault hold")
        self.state = RollbackExecutorState.FAULT
        self._paused_state = measured.copy()
        self._stationary_hold_target = None
        self._stationary_settle_ticks = 0
        self._finish_to_anchor = None
        self._continue_after_stationary_hold = False
        return self._result(self.state, sent, reason, measured)

    def _pending_dispatch_result(self, measured: np.ndarray) -> RollbackStepResult:
        """Wait for the selected queue point without inventing progress."""
        command, _velocity = self._command_motion(measured)
        self.state = RollbackExecutorState.RUNNING
        return self._result(
            self.state,
            command,
            "rollback_dispatch_pending",
            measured,
            source_timestamp_ns=self._current_source_timestamp_ns(),
        )
