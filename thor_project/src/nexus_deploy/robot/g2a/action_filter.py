"""Small NumPy-only action conditioners for the standalone robot client."""

from __future__ import annotations

import math
import copy

import numpy as np


class FixedHorizonChunkProjector:
    """Project a complete policy chunk onto fixed-time kinematic bounds.

    The number of actions and their timestamps never change.  Only selected
    trajectory channels are adjusted; device-position channels such as the DH
    grippers therefore keep their original event indices.  Unlike the live
    tracking limiter, every new RTC window is anchored to the last command and
    becomes the reference itself, so an unreachable model target cannot build
    an invisible error that a later servo tick tries to chase.

    This is a deterministic forward projection, not variable-duration C2
    retiming.  A candidate that cannot reach its original endpoint inside the
    fixed horizon ends at the closest acceleration/velocity-bounded position.
    """

    def __init__(
        self,
        *,
        policy_hz: float,
        action_dim: int,
        projected_indices: tuple[int, ...],
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
    ) -> None:
        if not math.isfinite(policy_hz) or policy_hz <= 0:
            raise ValueError("policy_hz must be positive and finite")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        indices = tuple(projected_indices)
        if len(set(indices)) != len(indices) or any(
            not isinstance(i, int) or i < 0 or i >= action_dim for i in indices
        ):
            raise ValueError("invalid fixed-horizon projection indices")
        velocity = np.asarray(max_velocity, dtype=np.float32)
        acceleration = np.asarray(max_acceleration, dtype=np.float32)
        if velocity.shape != (action_dim,) or acceleration.shape != (action_dim,):
            raise ValueError("fixed-horizon limits must match action_dim")
        if (
            not np.isfinite(velocity).all()
            or not np.isfinite(acceleration).all()
            or np.any(velocity <= 0)
            or np.any(acceleration <= 0)
        ):
            raise ValueError("fixed-horizon limits must be finite and positive")
        self.dt = 1.0 / float(policy_hz)
        self.action_dim = action_dim
        self.projected_indices = indices
        self.max_velocity = velocity
        self.max_acceleration = acceleration

    def project(
        self,
        actions: np.ndarray,
        *,
        start_position: np.ndarray,
        start_velocity: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        raw = self._chunk(actions)
        position = self._vector(start_position, "start_position").copy()
        velocity = self._vector(start_velocity, "start_velocity").copy()
        result = raw.copy()
        indices = list(self.projected_indices)
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        velocity[[i for i in range(self.action_dim) if i not in self.projected_indices]] = 0.0
        delta = self.max_acceleration * self.dt

        for step, target in enumerate(raw):
            desired = (target - position) / self.dt
            next_velocity = np.clip(desired, velocity - delta, velocity + delta)
            next_velocity = np.clip(next_velocity, -self.max_velocity, self.max_velocity)
            next_position = position + next_velocity * self.dt
            result[step, indices] = next_position[indices]
            position[indices] = next_position[indices]
            velocity[indices] = next_velocity[indices]

        difference = np.abs(result[:, indices] - raw[:, indices])
        return np.ascontiguousarray(result, dtype=np.float32), {
            "steps": int(len(result)),
            "changed_values": int(np.count_nonzero(difference > 1e-7)),
            "max_deviation_rad": float(difference.max(initial=0.0)),
            "rms_deviation_rad": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        }

    def _vector(self, value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be one finite {self.action_dim}D action")
        return array

    def _chunk(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.action_dim or not np.isfinite(array).all():
            raise ValueError(f"actions must be a finite Nx{self.action_dim} chunk")
        return array


class ChunkTrajectoryShaper:
    """Zero-phase arm smoothing for an already available policy chunk.

    A symmetric five-point Savitzky-Golay kernel removes sampling noise without
    the phase delay of a causal servo-rate filter.  Device-position channels
    and locked head/waist channels are copied verbatim.
    """

    def __init__(self, *, action_dim: int, shaped_indices: tuple[int, ...],
                 window: int = 5, max_deviation: float = 0.02,
                 event_indices: tuple[int, ...] = (), event_midpoint: float | None = None,
                 min_event_run: int = 3) -> None:
        if action_dim <= 0 or window != 1 and (window < 5 or window > 15 or window % 2 == 0):
            raise ValueError("chunk smoothing window must be 1 or an odd value from 5 through 15")
        if not math.isfinite(max_deviation) or max_deviation < 0:
            raise ValueError("chunk smoothing max deviation must be finite and non-negative")
        self.action_dim = action_dim
        self.shaped_indices = tuple(shaped_indices)
        if any(i < 0 or i >= action_dim for i in self.shaped_indices):
            raise ValueError("invalid chunk smoothing indices")
        self.window = window
        self.max_deviation = float(max_deviation)
        self.event_indices = tuple(event_indices)
        self.event_midpoint = event_midpoint
        self.min_event_run = int(min_event_run)
        self._savgol_weights: np.ndarray | None = None
        if window != 1 and self.shaped_indices:
            radius = window // 2
            x = np.arange(-radius, radius + 1, dtype=np.float32)
            self._savgol_weights = np.linalg.pinv(
                np.stack((np.ones_like(x), x, x * x), axis=1)
            )[0].astype(np.float32)

    def shape(self, actions: np.ndarray) -> np.ndarray:
        value = np.asarray(actions, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != self.action_dim or not np.isfinite(value).all():
            raise ValueError(f"Expected a finite Nx{self.action_dim} action chunk, got {value.shape}")
        result = value.copy()
        if self.window != 1 and len(value) >= self.window and self.shaped_indices:
            indices = list(self.shaped_indices)
            source = value[:, indices]
            radius = self.window // 2
            weights = self._savgol_weights
            assert weights is not None
            padded = np.pad(source, ((radius, radius), (0, 0)), mode="edge")
            smooth = sum(
                weights[i] * padded[i:i + len(value)] for i in range(self.window)
            )
            if self.max_deviation:
                smooth = source + np.clip(
                    smooth - source, -self.max_deviation, self.max_deviation
                )
            result[:, indices] = smooth
        if self.event_indices and self.event_midpoint is not None:
            for index in self.event_indices:
                binary = result[:, index] <= self.event_midpoint
                start = 0
                while start < len(binary):
                    end = start + 1
                    while end < len(binary) and binary[end] == binary[start]:
                        end += 1
                    if end - start < self.min_event_run:
                        replacement = result[start - 1, index] if start else (
                            result[end, index] if end < len(binary) else result[start, index]
                        )
                        result[start:end, index] = replacement
                    start = end
        return np.ascontiguousarray(result, dtype=np.float32)


class AdaptiveFixedTimeChunkSmoother:
    """Bounded arm-only smoothing on the policy chunk's fixed time axis.

    The convex objective penalizes acceleration and jerk more strongly where
    the source joint is moving slowly.  Fast intentional motion therefore
    stays close to the policy, while small reversals receive more damping.
    The first/last samples, timestamps, and every non-arm channel are exact.
    A NumPy-only projected FISTA solver keeps the standalone GDK deployment
    independent of SciPy.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        shaped_indices: tuple[int, ...],
        policy_hz: float,
        max_deviation: float = 0.005,
        acceleration_weight: float = 20.0,
        jerk_weight: float = 5.0,
        velocity_reference: float = 0.25,
        max_iterations: int = 40,
        tolerance: float = 1e-8,
    ) -> None:
        values = (
            policy_hz,
            max_deviation,
            acceleration_weight,
            jerk_weight,
            velocity_reference,
            tolerance,
        )
        if action_dim <= 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("adaptive chunk smoothing parameters must be finite")
        if policy_hz <= 0 or max_deviation < 0 or acceleration_weight < 0 or jerk_weight < 0:
            raise ValueError("adaptive chunk smoothing rates/weights must be non-negative")
        if velocity_reference <= 0 or max_iterations <= 0 or tolerance <= 0:
            raise ValueError("adaptive velocity reference/iterations/tolerance must be positive")
        indices = tuple(shaped_indices)
        if len(set(indices)) != len(indices) or any(
            not isinstance(index, int) or index < 0 or index >= action_dim for index in indices
        ):
            raise ValueError("invalid adaptive chunk smoothing indices")
        self.action_dim = int(action_dim)
        self.shaped_indices = indices
        self.policy_hz = float(policy_hz)
        self.max_deviation = float(max_deviation)
        self.acceleration_weight = float(acceleration_weight)
        self.jerk_weight = float(jerk_weight)
        self.velocity_reference = float(velocity_reference)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def shape(self, actions: np.ndarray) -> np.ndarray:
        value = np.asarray(actions, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != self.action_dim or not np.isfinite(value).all():
            raise ValueError(f"Expected a finite Nx{self.action_dim} action chunk, got {value.shape}")
        result = value.copy()
        if len(value) < 4 or not self.shaped_indices or self.max_deviation == 0:
            return np.ascontiguousarray(result)
        indices = list(self.shaped_indices)
        source = value[:, indices].astype(np.float64, copy=False)
        count, joint_count = source.shape
        interior = count - 2
        velocity = np.abs(np.gradient(source, axis=0)) * self.policy_hz

        second = np.zeros((count - 2, interior), dtype=np.float64)
        second_rhs = np.zeros((count - 2, joint_count), dtype=np.float64)
        for row in range(count - 2):
            for sample, coefficient in zip((row, row + 1, row + 2), (1.0, -2.0, 1.0)):
                if sample == 0:
                    second_rhs[row] -= coefficient * source[0, :]
                elif sample == count - 1:
                    second_rhs[row] -= coefficient * source[-1, :]
                else:
                    second[row, sample - 1] += coefficient
        second_weight = 1.0 / (
            1.0 + (velocity[1:-1, :] / self.velocity_reference) ** 4
        )

        third = np.zeros((count - 3, interior), dtype=np.float64)
        third_rhs = np.zeros((count - 3, joint_count), dtype=np.float64)
        for row in range(count - 3):
            for sample, coefficient in zip(
                (row, row + 1, row + 2, row + 3), (-1.0, 3.0, -3.0, 1.0)
            ):
                if sample == 0:
                    third_rhs[row] -= coefficient * source[0, :]
                elif sample == count - 1:
                    third_rhs[row] -= coefficient * source[-1, :]
                else:
                    third[row, sample - 1] += coefficient
        third_velocity = np.maximum(velocity[1:-2, :], velocity[2:-1, :])
        third_weight = 1.0 / (
            1.0 + (third_velocity / self.velocity_reference) ** 4
        )

        hessian = np.broadcast_to(
            np.eye(interior, dtype=np.float64),
            (joint_count, interior, interior),
        ).copy()
        rhs = source[1:-1, :].T.copy()
        if self.acceleration_weight:
            hessian += self.acceleration_weight * np.einsum(
                "ra,rj,rb->jab", second, second_weight, second, optimize=True
            )
            rhs += self.acceleration_weight * np.einsum(
                "ra,rj->ja", second, second_weight * second_rhs, optimize=True
            )
        if self.jerk_weight:
            hessian += self.jerk_weight * np.einsum(
                "ra,rj,rb->jab", third, third_weight, third, optimize=True
            )
            rhs += self.jerk_weight * np.einsum(
                "ra,rj->ja", third, third_weight * third_rhs, optimize=True
            )

        lower = source[1:-1, :].T - self.max_deviation
        upper = source[1:-1, :].T + self.max_deviation
        lipschitz = np.max(np.sum(np.abs(hessian), axis=2), axis=1)[:, None]
        current = source[1:-1, :].T.copy()
        momentum = current.copy()
        scale = 1.0
        for _ in range(self.max_iterations):
            gradient = np.einsum(
                "jab,jb->ja", hessian, momentum, optimize=True
            ) - rhs
            updated = np.clip(
                momentum - gradient / lipschitz,
                lower,
                upper,
            )
            if np.max(np.abs(updated - current), initial=0.0) <= self.tolerance:
                current = updated
                break
            next_scale = (1.0 + math.sqrt(1.0 + 4.0 * scale * scale)) * 0.5
            momentum = updated + ((scale - 1.0) / next_scale) * (updated - current)
            current = updated
            scale = next_scale
        result[1:-1, indices] = current.T.astype(np.float32)
        return np.ascontiguousarray(result, dtype=np.float32)


class CausalLowPassFilter:
    """First-order low-pass filter with a stateful, causal update."""

    def __init__(self, *, cutoff_hz: float, sample_hz: float, action_dim: int) -> None:
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive and finite")
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("sample_hz must be positive and finite")
        if cutoff_hz >= sample_hz * 0.5:
            raise ValueError("cutoff_hz must be below the Nyquist frequency")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        dt = 1.0 / sample_hz
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        self.alpha = dt / (rc + dt)
        self.action_dim = action_dim
        self._value: np.ndarray | None = None

    def reset(self, value: np.ndarray) -> None:
        self._value = self._validate(value).copy()

    def fork(self):
        result = copy.copy(self)
        result._value = None if self._value is None else self._value.copy()
        return result

    def update(self, target: np.ndarray) -> np.ndarray:
        target = self._validate(target)
        if self._value is None:
            self._value = target.copy()
        else:
            self._value = np.asarray(
                self._value + self.alpha * (target - self._value),
                dtype=np.float32,
            )
        return self._value.copy()

    def _validate(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"Expected one finite {self.action_dim}D action, got {array.shape}")
        return array


class JointKinematicLimiter:
    """Deterministic per-joint velocity and acceleration limiter.

    The limiter advances by one configured servo period on every successful
    call. It deliberately does not use wall-clock deltas. Explicit device
    position-target channels bypass trajectory shaping; the safety layer checks
    the measured-to-target opening path separately.
    """

    def __init__(
        self,
        *,
        sample_hz: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        position_target_indices: tuple[int, ...] = (),
        unlimited_acceleration_indices: tuple[int, ...] = (),
        tracking_damping_indices: tuple[int, ...] = (),
        tracking_damping_omega: float = 12.0,
    ) -> None:
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("sample_hz must be positive and finite")
        velocity = np.asarray(max_velocity, dtype=np.float32)
        acceleration = np.asarray(max_acceleration, dtype=np.float32)
        if velocity.ndim != 1 or acceleration.shape != velocity.shape or len(velocity) == 0:
            raise ValueError("velocity and acceleration limits must be equal non-empty 1D arrays")
        if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
            raise ValueError("kinematic limits must be finite")
        if np.any(velocity <= 0) or np.any(acceleration <= 0):
            raise ValueError("kinematic limits must be positive")
        self.dt = 1.0 / float(sample_hz)
        self.max_velocity = velocity
        self.max_acceleration = acceleration
        self.action_dim = len(velocity)
        self.position_target_indices = tuple(position_target_indices)
        if len(set(self.position_target_indices)) != len(self.position_target_indices) or any(
            not isinstance(i, int) or i < 0 or i >= self.action_dim for i in self.position_target_indices
        ):
            raise ValueError("Invalid position-target indices")
        self.unlimited_acceleration_indices = tuple(unlimited_acceleration_indices)
        if (len(set(self.unlimited_acceleration_indices)) != len(self.unlimited_acceleration_indices)
                or any(not isinstance(i, int) or i < 0 or i >= self.action_dim
                       or i in self.position_target_indices
                       for i in self.unlimited_acceleration_indices)):
            raise ValueError("Invalid unlimited-acceleration indices")
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(self.action_dim, dtype=np.float32)
        self.tracking_damping_indices = tuple(tracking_damping_indices)
        if len(set(self.tracking_damping_indices)) != len(self.tracking_damping_indices) or any(
            not isinstance(i, int) or i < 0 or i >= self.action_dim
            or i in self.position_target_indices for i in self.tracking_damping_indices
        ):
            raise ValueError("Invalid tracking-damping indices")
        if not math.isfinite(tracking_damping_omega) or tracking_damping_omega <= 0:
            raise ValueError("tracking_damping_omega must be positive and finite")
        if self.tracking_damping_indices and tracking_damping_omega * self.dt > 0.25:
            raise ValueError("tracking damping requires omega * servo period <= 0.25")
        self.tracking_damping_omega = float(tracking_damping_omega)

    def reset(self, position: np.ndarray) -> None:
        self._position = self._validate(position).copy()
        self._velocity.fill(0.0)

    def fork(self) -> JointKinematicLimiter:
        """Preview future 100 Hz steps without advancing the live limiter."""
        result = copy.copy(self)
        result._position = None if self._position is None else self._position.copy()
        result._velocity = self._velocity.copy()
        return result

    def reconcile_position_clip(self, clipped_position: np.ndarray) -> None:
        """Reanchor only clipped coordinates; preserve unrelated joint motion.

        This is for absolute hardware limits, not collision/pause holds, which
        still reset the complete trajectory. Exact comparison is deliberate:
        even a floating-point boundary violation is corrected, never admitted.
        """
        clipped = self._validate(clipped_position)
        if self._position is None:
            self.reset(clipped)
            return
        changed = clipped != self._position
        self._position[changed] = clipped[changed]
        self._velocity[changed] = 0.0

    def update(self, target: np.ndarray) -> np.ndarray:
        target = self._validate(target)
        if self._position is None:
            self.reset(target)
            return target.copy()

        error = target - self._position
        direction = np.sign(error)
        # Decelerate early enough to reach a stationary target without a
        # large final snap.  The discrete overshoot guard below remains as a
        # numerical safety net for very small residual errors.
        # A continuous-time ``sqrt(2*a*x)`` bound can still overshoot the
        # target by one discrete sample.  At a hard joint boundary that tiny
        # overshoot is position-clipped and appears as an acceleration spike.
        # Reserve one full acceleration step in the braking-speed bound so a
        # fixed-rate controller reaches the target without relying on the
        # downstream hard clamp.
        acceleration_step = self.max_acceleration * self.dt
        braking_velocity = np.maximum(
            0.0,
            np.sqrt(
                2.0 * self.max_acceleration * np.abs(error)
                + acceleration_step**2
            )
            - acceleration_step,
        )
        desired_velocity = direction * np.minimum(self.max_velocity, braking_velocity)
        max_velocity_change = acceleration_step
        if self.unlimited_acceleration_indices:
            indices = list(self.unlimited_acceleration_indices)
            desired_velocity[indices] = np.clip(
                error[indices] / self.dt,
                -self.max_velocity[indices], self.max_velocity[indices],
            )
            max_velocity_change = max_velocity_change.copy()
            max_velocity_change[indices] = np.inf
        velocity = self._velocity + np.clip(
            desired_velocity - self._velocity,
            -max_velocity_change,
            max_velocity_change,
        )
        step = velocity * self.dt
        # Snap exactly onto the target only when that final discrete velocity
        # is reachable from the previous velocity *and* can return to zero on
        # the next sample without violating acceleration.  Otherwise a tiny,
        # bounded overshoot is safer than an unbounded final deceleration.
        snap_velocity = error / self.dt
        safe_snap = (
            (np.abs(snap_velocity) <= self.max_velocity)
            & (np.abs(snap_velocity - self._velocity) <= max_velocity_change)
            & (np.abs(snap_velocity) <= max_velocity_change)
        )
        step[safe_snap] = error[safe_snap]
        velocity[safe_snap] = snap_velocity[safe_snap]

        self._position = np.asarray(self._position + step, dtype=np.float32)
        self._velocity = np.asarray(velocity, dtype=np.float32)
        self._apply_position_targets(target)
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def adopt_preplanned(self, target: np.ndarray) -> np.ndarray:
        """Validate and adopt an exact externally planned servo waypoint.

        A rollback path already has its own time scaling. Re-running a
        tracking controller would change historical points, so reject any
        waypoint that violates the live limiter instead of projecting it.
        """
        target = self._validate(target)
        if self._position is None:
            raise ValueError("Preplanned trajectory requires a measured command anchor")
        velocity = np.asarray((target - self._position) / self.dt, dtype=np.float32)
        active = np.ones(self.action_dim, dtype=bool)
        if self.position_target_indices:
            active[list(self.position_target_indices)] = False
        acceleration_limited = active.copy()
        if self.unlimited_acceleration_indices:
            acceleration_limited[list(self.unlimited_acceleration_indices)] = False
        if np.any(np.abs(velocity[active]) > self.max_velocity[active] + 1e-5):
            raise ValueError("Preplanned waypoint exceeds final velocity limits")
        delta = np.abs(velocity - self._velocity)
        if np.any(delta[acceleration_limited] > self.max_acceleration[acceleration_limited] * self.dt + 1e-5):
            raise ValueError("Preplanned waypoint exceeds final acceleration limits")
        self._position = target.copy()
        self._velocity = velocity
        if self.position_target_indices:
            self._velocity[list(self.position_target_indices)] = 0.0
        return self._position.copy()

    def update_tracking(self, target: np.ndarray, target_velocity: np.ndarray) -> np.ndarray:
        """Track a moving reference, preserving velocity across checked segments.

        Feed-forward plus bounded position correction avoids treating every
        preview sample as a stationary stopping point. Trajectory channels keep
        velocity/acceleration limits; device targets require envelope checking.
        """
        target = self._validate(target)
        reference_velocity = self._validate(target_velocity)
        if self._position is None:
            self.reset(target)
            return target.copy()
        desired = np.clip(reference_velocity + 8.0 * (target - self._position),
                          -self.max_velocity, self.max_velocity)
        if self.tracking_damping_indices:
            # Experimental, explicitly enabled arm channels only. A damped
            # acceleration request avoids demanding the full position-error
            # correction velocity immediately after a reference change.
            # This changes tracking dynamics, NOT timestamps, hard bounds,
            # gripper registers, checked-queue authority or collision holds.
            indices = list(self.tracking_damping_indices)
            omega = self.tracking_damping_omega
            acceleration = (omega**2 * (target[indices] - self._position[indices])
                            + 2.0 * omega * (reference_velocity[indices] - self._velocity[indices]))
            desired[indices] = np.clip(
                self._velocity[indices] + self.dt * acceleration,
                -self.max_velocity[indices], self.max_velocity[indices],
            )
        delta = self.max_acceleration * self.dt
        if self.unlimited_acceleration_indices:
            delta = delta.copy()
            delta[list(self.unlimited_acceleration_indices)] = np.inf
        self._velocity = np.asarray(self._velocity + np.clip(
            desired - self._velocity, -delta, delta), dtype=np.float32)
        self._position = np.asarray(self._position + self._velocity * self.dt, dtype=np.float32)
        self._apply_position_targets(target)
        return self._position.copy()

    def _apply_position_targets(self, target: np.ndarray) -> None:
        # Device-position channels are target registers, not a claim about
        # physical velocity. The safety layer checks the measured-to-target
        # opening path using the actual target opening.
        if self.position_target_indices:
            indices = list(self.position_target_indices)
            self._position[indices] = target[indices]
            self._velocity[indices] = 0.0

    def _validate(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"Expected one finite {self.action_dim}D action, got {array.shape}")
        return array
