"""State-based arm/effector coordination for device-position grippers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _SideState:
    closed: bool
    phase: str = "idle"
    arm_target: np.ndarray | None = None
    release_from: np.ndarray | None = None
    effector_target: float = 0.0
    previous_effector_target: float = 0.0
    started_at: float = 0.0
    last_feedback: float | None = None
    last_feedback_time: float | None = None
    initial_feedback: float | None = None
    stable_since: float | None = None
    timeout_reported: bool = False


class GraspEventCoordinator:
    """Keep arm motion at a grasp/release pose until its gripper responds.

    The model clock and inference continue.  Only the affected arm's emitted
    target is held, so a slow physical gripper cannot be outrun by later lift
    or retreat samples.
    """

    def __init__(self, *, action_dim: int, left_arm: tuple[int, ...], right_arm: tuple[int, ...],
                 effector_indices: tuple[int, int], state_range: tuple[float, float],
                 arm_tolerance: float = 0.05, feedback_tolerance: float = 0.002,
                 contact_stable_s: float = 0.1, contact_delta: float = 0.0002,
                 contact_min_travel: float = 0.001, timeout_s: float = 0.9,
                 release_s: float = 0.15) -> None:
        if action_dim <= 0 or contact_stable_s <= 0 or timeout_s <= 0 or release_s <= 0:
            raise ValueError("invalid grasp coordinator configuration")
        self.action_dim = action_dim
        self.arms = (tuple(left_arm), tuple(right_arm))
        self.effectors = tuple(effector_indices)
        self.minimum, self.maximum = map(float, state_range)
        if not self.maximum > self.minimum:
            raise ValueError("state_range must be increasing")
        self.midpoint = (self.minimum + self.maximum) * 0.5
        # Full-state hysteresis keeps intermediate/noisy model values from
        # manufacturing semantic open/close events. For the Dahuan 0..25 mm
        # profile these are exactly 5 mm (closed) and 20 mm (open).
        span = self.maximum - self.minimum
        self.close_threshold = self.minimum + span * 0.2
        self.open_threshold = self.maximum - span * 0.2
        self.arm_tolerance = float(arm_tolerance)
        self.feedback_tolerance = float(feedback_tolerance)
        self.contact_stable_s = float(contact_stable_s)
        self.contact_delta = float(contact_delta)
        self.contact_min_travel = float(contact_min_travel)
        self.timeout_s = float(timeout_s)
        self.release_s = float(release_s)
        self._sides: list[_SideState] | None = None

    def reset(self, action: np.ndarray) -> None:
        value = self._validate(action)
        self._sides = [
            _SideState(closed=self._initial_closed(float(value[e])),
                       effector_target=float(value[e]), previous_effector_target=float(value[e]))
            for e in self.effectors
        ]

    @property
    def blocked(self) -> bool:
        return bool(self._sides and any(side.phase != "idle" for side in self._sides))

    def update(self, candidate: np.ndarray, feedback: np.ndarray | None, *, now: float,
               feedback_time: float | None = None):
        value = self._validate(candidate).copy()
        measured = None if feedback is None else self._validate(feedback)
        if self._sides is None:
            self.reset(measured if measured is not None else value)
        events: list[dict[str, object]] = []
        assert self._sides is not None
        for side_index, state in enumerate(self._sides):
            arm = list(self.arms[side_index])
            eff = self.effectors[side_index]
            requested_closed = self._requested_state(float(value[eff]))
            if (requested_closed is not None and requested_closed != state.closed
                    and state.phase == "idle"):
                state.phase = "wait_arm"
                state.arm_target = value[arm].copy()
                state.effector_target = float(value[eff])
                state.previous_effector_target = (
                    float(measured[eff]) if measured is not None
                    else self.minimum if state.closed else self.maximum
                )
                state.started_at = now
                state.last_feedback = None
                state.last_feedback_time = None
                state.initial_feedback = None
                state.stable_since = None
                state.timeout_reported = False
                events.append({
                    "side": side_index, "event": "started", "closed": requested_closed,
                    "target": state.effector_target,
                })

            if state.phase == "wait_arm":
                # Freeze only this arm at the semantic event pose. The other
                # arm and the policy clock remain untouched.
                value[arm] = state.arm_target
                value[eff] = state.previous_effector_target
                arm_ready = (
                    measured is not None
                    and float(np.max(np.abs(measured[arm] - state.arm_target))) <= self.arm_tolerance
                )
                if arm_ready:
                    state.phase = "wait_gripper"
                    state.started_at = now
                    state.initial_feedback = float(measured[eff])
                    state.last_feedback = state.initial_feedback
                    state.last_feedback_time = feedback_time
                    state.stable_since = None
                    events.append({
                        "side": side_index, "event": "arm_ready",
                        "closed": state.effector_target <= self.close_threshold,
                    })
                elif now - state.started_at >= self.timeout_s and not state.timeout_reported:
                    state.timeout_reported = True
                    events.append({
                        "side": side_index, "event": "timeout", "phase": "wait_arm",
                        "closed": state.effector_target <= self.close_threshold,
                        "elapsed_s": now - state.started_at,
                        "feedback": None if measured is None else float(measured[eff]),
                    })

            if state.phase == "wait_gripper":
                value[arm] = state.arm_target
                value[eff] = state.effector_target
                confirmed = (
                    measured is not None
                    and abs(float(measured[eff]) - state.effector_target) <= self.feedback_tolerance
                )
                if measured is not None and not confirmed:
                    current = float(measured[eff])
                    fresh = feedback_time is None or feedback_time != state.last_feedback_time
                    if fresh:
                        sample_time = now if feedback_time is None else float(feedback_time)
                        if (state.last_feedback is not None
                                and abs(current - state.last_feedback) <= self.contact_delta):
                            if state.stable_since is None:
                                state.stable_since = (
                                    state.last_feedback_time
                                    if state.last_feedback_time is not None else sample_time
                                )
                        else:
                            state.stable_since = None
                        state.last_feedback = current
                        state.last_feedback_time = sample_time
                    # A closing gripper may stop on an object before reaching
                    # the empty-jaw target; stable contact is successful grasp.
                    if state.effector_target <= self.close_threshold:
                        travelled = (
                            state.initial_feedback is not None
                            and state.initial_feedback - current >= self.contact_min_travel
                        )
                        stable_for = (
                            0.0 if state.stable_since is None or state.last_feedback_time is None
                            else state.last_feedback_time - state.stable_since
                        )
                        confirmed = travelled and stable_for >= self.contact_stable_s
                if confirmed:
                    state.closed = state.effector_target <= self.close_threshold
                    state.phase = "release"
                    state.release_from = state.arm_target.copy()
                    state.started_at = now
                    events.append({
                        "side": side_index, "event": "confirmed", "closed": state.closed,
                        "feedback": None if measured is None else float(measured[eff]),
                    })
                elif now - state.started_at >= self.timeout_s and not state.timeout_reported:
                    state.timeout_reported = True
                    events.append({
                        "side": side_index, "event": "timeout",
                        "phase": "wait_gripper",
                        "closed": state.effector_target <= self.close_threshold,
                        "elapsed_s": now - state.started_at,
                        "feedback": None if measured is None else float(measured[eff]),
                    })

            if state.phase == "release":
                # Rejoin the still-running plan without a one-tick jump after
                # the arm may have been held for half a second or more.
                value[eff] = state.effector_target
                alpha = float(np.clip((now - state.started_at) / self.release_s, 0.0, 1.0))
                value[arm] = state.release_from + (value[arm] - state.release_from) * alpha
                if alpha >= 1.0:
                    state.phase = "idle"
                    state.arm_target = None
                    state.release_from = None
                    events.append({
                        "side": side_index, "event": "released", "closed": state.closed,
                    })
        return value, events

    def condition_preview(
        self, candidates: np.ndarray, *, now: float, period: float
    ) -> np.ndarray:
        """Apply the current per-side hold to a safety lookahead without mutation.

        Feedback-dependent phases are conservatively assumed to remain active
        for the whole preview. A time-based release can be projected exactly.
        This keeps asynchronous collision checks aligned with emitted targets.
        """
        values = np.asarray(candidates, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != self.action_dim
                or not np.isfinite(values).all() or not np.isfinite(period) or period <= 0):
            raise ValueError(f"Expected finite Nx{self.action_dim} preview and positive period")
        result = values.copy()
        if self._sides is None:
            return result
        for side_index, state in enumerate(self._sides):
            if state.phase == "idle":
                continue
            arm = list(self.arms[side_index])
            eff = self.effectors[side_index]
            if state.phase == "wait_arm":
                result[:, arm] = state.arm_target
                result[:, eff] = state.previous_effector_target
            elif state.phase == "wait_gripper":
                result[:, arm] = state.arm_target
                result[:, eff] = state.effector_target
            elif state.phase == "release":
                for index, point in enumerate(result):
                    alpha = float(np.clip(
                        (now + index * period - state.started_at) / self.release_s, 0.0, 1.0
                    ))
                    point[arm] = state.release_from + (point[arm] - state.release_from) * alpha
                    point[eff] = state.effector_target
        return result

    def _initial_closed(self, value: float) -> bool:
        if value <= self.close_threshold:
            return True
        if value >= self.open_threshold:
            return False
        return value <= self.midpoint

    def _requested_state(self, value: float) -> bool | None:
        if value <= self.close_threshold:
            return True
        if value >= self.open_threshold:
            return False
        return None

    def _validate(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"Expected one finite {self.action_dim}D action, got {array.shape}")
        return array
