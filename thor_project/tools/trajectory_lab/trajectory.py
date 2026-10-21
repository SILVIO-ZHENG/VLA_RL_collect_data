"""Shared-clock C2 trajectories with conservative polynomial derivative bounds.

No GDK, network, worker process, or production-client imports. A trajectory is
NOT a collision certificate and this module must not be used to send commands.
"""

from dataclasses import dataclass
from math import comb, factorial
from uuid import uuid4

import numpy as np


class PlanningError(ValueError):
    pass


def frozen(value):
    # Bytes-backed storage cannot be made writable again by a caller.
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def vector(value, dimension, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise PlanningError(f"{name}: expected finite {dimension}D vector")
    return array


@dataclass(frozen=True)
class Limits:
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        n = np.asarray(self.velocity).size
        if not n:
            raise PlanningError("Empty limits")
        for name in ("velocity", "acceleration", "jerk", "lower", "upper"):
            object.__setattr__(self, name, frozen(vector(getattr(self, name), n, name)))
        if any(np.any(getattr(self, name) <= 0) for name in ("velocity", "acceleration", "jerk")):
            raise PlanningError("Derivative limits must be positive")
        if np.any(self.lower >= self.upper):
            raise PlanningError("Invalid position bounds")


@dataclass(frozen=True)
class Sample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    phase: float
    time: float

    def __post_init__(self):
        for name in ("position", "velocity", "acceleration", "jerk"):
            object.__setattr__(self, name, frozen(getattr(self, name)))


def _hull_matrix(degree):
    """Power coefficients -> Bernstein coefficients on four subintervals."""
    matrices = np.zeros((4, degree + 1, degree + 1))
    for sub in range(4):
        lo, width = sub / 4, 0.25
        for k in range(degree + 1):
            for i in range(degree + 1):
                matrices[sub, k, i] = sum(
                    comb(k, r) / comb(degree, r) * comb(i, r) * lo ** (i - r) * width**r for r in range(min(k, i) + 1)
                )
    return matrices


_HULLS = {degree: _hull_matrix(degree) for degree in (2, 3, 4, 5)}


def polynomial_bounds(coefficients, durations, order=0):
    """Conservative bounds for every point, not only 100 Hz sample endpoints."""
    degree = 5 - order
    derivative = coefficients[:, order:, :].copy()
    derivative *= np.array([factorial(i) / factorial(i - order) for i in range(order, 6)])[None, :, None]
    derivative /= durations[:, None, None] ** order
    hull = np.einsum("ski,nij->nskj", _HULLS[degree], derivative)
    return hull.min(axis=(1, 2)), hull.max(axis=(1, 2))


def _coefficients(points, durations, velocities, accelerations):
    c = np.zeros((len(durations), 6, points.shape[1]))
    h = durations[:, None]
    c[:, 0] = points[:-1]
    c[:, 1] = velocities[:-1] * h
    c[:, 2] = accelerations[:-1] * h * h / 2
    residual = points[1:] - c[:, 0] - c[:, 1] - c[:, 2]
    v = velocities[1:] * h - c[:, 1] - 2 * c[:, 2]
    a = accelerations[1:] * h * h - 2 * c[:, 2]
    c[:, 3] = 10 * residual - 4 * v + a / 2
    c[:, 4] = -15 * residual + 7 * v - a
    c[:, 5] = 6 * residual - 3 * v + a / 2
    return c


def _cubic_derivatives(points, durations, start_velocity):
    """Clamped C2 cubic guide, solved with O(N D) tridiagonal elimination.

    Independent harmonic tangents can introduce large quintic jerk even on
    smooth input. A single guide couples interior derivatives. The final
    quintic still imposes the supplied start acceleration and a rest end.
    """
    slopes = np.diff(points, axis=0) / durations[:, None]
    diagonal = np.r_[2 * durations[0], 2 * (durations[:-1] + durations[1:]), 2 * durations[-1]]
    rhs = 6 * np.vstack((slopes[0] - start_velocity, np.diff(slopes, axis=0), -slopes[-1]))
    for i in range(1, len(points)):
        ratio = durations[i - 1] / diagonal[i - 1]
        diagonal[i] -= ratio * durations[i - 1]
        rhs[i] -= ratio * rhs[i - 1]
    acceleration = np.empty_like(points)
    acceleration[-1] = rhs[-1] / diagonal[-1]
    for i in range(len(points) - 2, -1, -1):
        acceleration[i] = (rhs[i] - durations[i] * acceleration[i + 1]) / diagonal[i]
    velocity = np.zeros_like(points)
    velocity[:-1] = slopes - durations[:, None] * (2 * acceleration[:-1] + acceleration[1:]) / 6
    return velocity, acceleration


class Trajectory:
    def __init__(
        self,
        points,
        durations,
        coefficients,
        phases,
        device_indices,
        limits,
        iterations,
        original_points=None,
        original_phases=None,
        path_refinements=(),
    ):
        self.source_id = uuid4().hex
        self.points = frozen(points)
        self.durations = frozen(durations)
        self.coefficients = frozen(coefficients)
        self.times = frozen(np.r_[0.0, np.cumsum(durations)])
        self.phases = frozen(phases)
        self.original_points = frozen(points if original_points is None else original_points)
        self.original_phases = frozen(phases if original_phases is None else original_phases)
        self.device_indices = tuple(device_indices)
        self.limits = limits
        self.iterations = iterations
        self.path_refinements = tuple(path_refinements)

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, time):
        if not np.isfinite(time) or time < 0:
            raise PlanningError("Sample time must be finite and non-negative")
        t = min(float(time), self.duration)
        if t >= self.duration:
            z = np.zeros(self.points.shape[1])
            return Sample(self.points[-1], z, z, z, float(self.phases[-1]), t)
        k = min(int(np.searchsorted(self.times, t, side="right") - 1), len(self.durations) - 1)
        u = (t - self.times[k]) / self.durations[k]
        values = []
        for order in range(4):
            factors = np.array([factorial(i) / factorial(i - order) * u ** (i - order) for i in range(order, 6)])
            values.append(factors @ self.coefficients[k, order:] / self.durations[k] ** order)
        indices = list(self.device_indices)
        # Device events occur at their source waypoint, not at a future body's
        # goal. All dimensions share the same retimed knot timestamps.
        phase = self.phases[k] + u * (self.phases[k + 1] - self.phases[k])
        event = max(0, int(np.searchsorted(self.original_phases, phase, side="right") - 1))
        values[0][indices] = self.original_points[event, indices]
        for value in values[1:]:
            value[indices] = 0  # register targets, NOT a physical derivative claim
        return Sample(*values, float(phase), t)

    def progress(self, time):
        sample = self.sample(time)
        return {
            "source_phase": sample.phase,
            "trajectory_time_s": sample.time,
            "remaining_time_s": max(0.0, self.duration - sample.time),
        }

    def bounds(self):
        body = [i for i in range(self.points.shape[1]) if i not in self.device_indices]
        result = {}
        for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            lo, hi = polynomial_bounds(self.coefficients, self.durations, order)
            peak = np.maximum(abs(lo), abs(hi))
            result[name] = float(np.max(peak[:, body] / getattr(self.limits, name)[body]))
        return result


def simplify(points, phases, body, tolerance):
    """Keep phase-aligned body knots; device events remain a separate full stream."""
    selected = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first <= 1:
            continue
        u = (phases[first + 1 : last] - phases[first]) / (phases[last] - phases[first])
        reference = points[first, body] + u[:, None] * (points[last, body] - points[first, body])
        error = np.max(abs(points[first + 1 : last][:, body] - reference), axis=1)
        at = int(np.argmax(error)) + first + 1
        if error[at - first - 1] > tolerance:
            selected.add(at)
            stack.extend(((first, at), (at, last)))
    return np.array(sorted(selected))


def path_error_bound(trajectory):
    """Bound spline deviation from original piecewise-linear body path by phase."""
    return path_error_details(trajectory)["bound_rad"]


def path_error_details(trajectory):
    """Include the source interval/joint responsible for the largest bound."""
    phase, source_intervals, body, bounds = _path_error_intervals(trajectory)
    interval, joint = np.unravel_index(np.argmax(bounds), bounds.shape)
    return {
        "bound_rad": float(bounds[interval, joint]),
        "source_interval": int(source_intervals[interval]),
        "joint_index": body[joint],
        "phase_start": float(phase[interval]),
        "phase_end": float(phase[interval + 1]),
    }


def _path_error_intervals(trajectory):
    """Split at BOTH source and spline knots before bounding each polynomial.

    Internal refinement knots are not new source actions or gripper events.
    Comparing to the original unsplit source stream is essential: otherwise
    subdivision could accidentally weaken the path-fidelity check.
    """
    source = trajectory.original_phases
    phase = np.union1d(source, trajectory.phases)
    k = np.searchsorted(trajectory.phases, phase[:-1], side="right") - 1
    source_intervals = np.searchsorted(source, phase[:-1], side="right") - 1
    span = trajectory.phases[k + 1] - trajectory.phases[k]
    lo = (phase[:-1] - trajectory.phases[k]) / span
    width = np.diff(phase) / span
    c = trajectory.coefficients[k]
    local = np.zeros_like(c)
    for r in range(6):
        for i in range(r, 6):
            local[:, r] += c[:, i] * (comb(i, r) * lo ** (i - r) * width**r)[:, None]
    source_span = np.diff(source)[source_intervals]
    source_delta = np.diff(trajectory.original_points, axis=0)[source_intervals]
    reference = (
        trajectory.original_points[source_intervals]
        + source_delta * ((phase[:-1] - source[source_intervals]) / source_span)[:, None]
    )
    local[:, 0] -= reference
    local[:, 1] -= source_delta * (np.diff(phase) / source_span)[:, None]
    lower, upper = polynomial_bounds(local, np.ones(len(local)))
    body = [i for i in range(c.shape[2]) if i not in trajectory.device_indices]
    bounds = np.maximum(abs(lower[:, body]), abs(upper[:, body]))
    return phase, source_intervals, body, bounds


def _refine_path(trajectory, budget):
    """Insert reference midpoints only where the complete bound is violated.

    Keep existing fitted knots. The next solve starts from the nominal clock
    and must recertify all derivatives and the original path again.
    """
    phase, _, _, bounds = _path_error_intervals(trajectory)
    bad = np.max(bounds, axis=1) > budget
    midpoints = (phase[:-1][bad] + phase[1:][bad]) / 2
    refined = np.union1d(trajectory.phases, midpoints)
    if len(refined) == len(trajectory.phases):
        raise PlanningError("Path refinement made no progress")
    source = trajectory.original_phases
    q = np.stack(
        [np.interp(refined, source, trajectory.original_points[:, j]) for j in range(trajectory.points.shape[1])],
        axis=1,
    )
    q[np.searchsorted(refined, trajectory.phases)] = trajectory.points
    return q, refined, tuple(float(p) for p in midpoints)


def plan(
    points,
    *,
    policy_fps,
    limits,
    device_indices=(),
    phases=None,
    start_velocity=None,
    start_acceleration=None,
    servo_period=0.01,
    max_iterations=80,
    max_stretch=30.0,
    knot_tolerance=0.0,
    max_path_deviation=0.02,
    smoothing_window=1,
    retiming_damping=0.5,
    tangent_method="harmonic",
    max_path_refinements=0,
    start_blend_phase=0.0,
):
    """Build an offline trajectory, extending shared intervals when necessary.

    Points include the current adopted anchor. Internal knots share nonzero
    velocities for same-direction motion; acceleration is continuous. A finite
    suffix ends at rest. Replacing it must start at the frozen queue tail's q/v/a.
    """
    q = np.asarray(points, dtype=np.float64)
    if tangent_method not in ("cubic", "harmonic"):
        raise PlanningError("Unknown tangent method")
    if q.ndim != 2 or len(q) < 2 or not q.shape[1] or not np.isfinite(q).all():
        raise PlanningError("Expected at least two finite waypoints")
    n, d = q.shape
    if not isinstance(limits, Limits) or limits.velocity.shape != (d,):
        raise PlanningError("Limits dimension mismatch")
    if (
        not np.isfinite(policy_fps)
        or policy_fps <= 0
        or not np.isfinite(servo_period)
        or servo_period <= 0
        or not np.isfinite(max_stretch)
        or max_stretch < 1
        or not isinstance(max_iterations, int)
        or max_iterations < 1
        or not np.isfinite(retiming_damping)
        or not 0 < retiming_damping <= 1
        or type(max_path_refinements) is not int
        or not 0 <= max_path_refinements <= 8
        or not np.isfinite(start_blend_phase)
        or not 0 <= start_blend_phase <= 12
    ):
        raise PlanningError("Invalid planner timing/budget")
    indices = tuple(device_indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < d for i in indices):
        raise PlanningError("Invalid device indices")
    body = [i for i in range(d) if i not in indices]
    if not body:
        raise PlanningError("At least one trajectory joint is required")
    if (
        not np.isfinite(knot_tolerance)
        or knot_tolerance < 0
        or not np.isfinite(max_path_deviation)
        or max_path_deviation <= 0
    ):
        raise PlanningError("Invalid path approximation budget")
    if np.any(q < limits.lower) or np.any(q > limits.upper):
        raise PlanningError("Waypoint outside position limits; never silently clip")
    phase = np.arange(n, dtype=float) if phases is None else np.asarray(phases, dtype=float)
    if phase.shape != (n,) or not np.isfinite(phase).all() or np.any(np.diff(phase) <= 0):
        raise PlanningError("Source phases must increase strictly")
    if np.any(np.diff(phase) / policy_fps < servo_period - 1e-12):
        raise PlanningError("Source events would exceed the consumer tick rate")
    v0 = vector(np.zeros(d) if start_velocity is None else start_velocity, d, "start velocity")
    a0 = vector(np.zeros(d) if start_acceleration is None else start_acceleration, d, "start acceleration")
    if np.any(abs(v0) > limits.velocity) or np.any(abs(a0) > limits.acceleration):
        raise PlanningError("Initial derivatives exceed limits")
    if np.any(v0[list(indices)]) or np.any(a0[list(indices)]):
        raise PlanningError("Device channels do not describe physical derivatives")
    original_points, original_phases = q.copy(), phase.copy()
    if type(smoothing_window) is not int or smoothing_window < 1 or smoothing_window % 2 != 1:
        raise PlanningError("Smoothing window must be a positive odd integer")
    if smoothing_window > 1:
        if smoothing_window < 5 or smoothing_window > 15:
            raise PlanningError("Offline smoothing window must be 5 through 15")
        if not np.allclose(np.diff(phase), np.diff(phase)[0], rtol=1e-7, atol=1e-10):
            raise PlanningError("Symmetric prefit requires uniform input source phases")
        # Noncausal local quadratic fit uses the available future chunk. Unlike
        # a causal low-pass it does not add a fixed body-only phase delay. The
        # COMPLETE final spline is still bounded against the ORIGINAL points.
        radius = smoothing_window // 2
        x = np.arange(-radius, radius + 1, dtype=float)
        weights = np.linalg.pinv(np.stack((np.ones_like(x), x, x * x), axis=1))[0]
        q = q.copy()
        for joint in body:
            q[:, joint] = np.convolve(np.pad(q[:, joint], radius, mode="edge"), weights, mode="valid")
        # Bound even the prefit displacement; isolated outliers must not pull
        # neighboring reference points arbitrarily far from their source phase.
        budget = max_path_deviation / 4
        q[:, body] = original_points[:, body] + np.clip(q[:, body] - original_points[:, body], -budget, budget)
        q[0], q[-1] = original_points[0], original_points[-1]
    if knot_tolerance > 0:
        selected = simplify(q, phase, body, knot_tolerance)
        q, phase = q[selected], phase[selected]
        n = len(q)
    if start_blend_phase > 0:
        # Let the moving q/v/a boundary join a farther body knot rather than
        # forcing tiny near-boundary reversals. This does NOT remove original
        # path constraints or any device events: both retain the full stream.
        keep = phase >= phase[0] + start_blend_phase
        keep[0] = keep[-1] = True
        q, phase = q[keep], phase[keep]
        n = len(q)
    nominal = np.diff(phase) / policy_fps
    # Do not round every 33.3 ms interval up to 40 ms: that would silently turn
    # an otherwise feasible 30 Hz path into 25 Hz. The consumer samples at 100 Hz.
    durations = nominal.copy()
    refinements = []
    since_refinement = 0
    for iteration in range(max_iterations * (max_path_refinements + 1)):
        since_refinement += 1
        if tangent_method == "cubic":
            velocity, acceleration = _cubic_derivatives(q, durations, v0)
        else:
            slopes = np.diff(q, axis=0) / durations[:, None]
            velocity = np.zeros_like(q)
            left, right = slopes[:-1], slopes[1:]
            same = left * right > 0
            denominator = np.where(same, left + right, 1.0)
            velocity[1:-1] = np.where(same, 2 * left * right / denominator, 0.0)
            velocity = np.clip(velocity, -limits.velocity, limits.velocity)
            acceleration = np.zeros_like(q)
            acceleration[1:-1] = 2 * (right - left) / (durations[:-1] + durations[1:])[:, None]
            acceleration = np.clip(acceleration, -limits.acceleration, limits.acceleration)
        velocity[0] = v0
        acceleration[0] = a0
        acceleration[-1] = 0
        velocity[:, list(indices)] = 0
        acceleration[:, list(indices)] = 0
        c = _coefficients(q, durations, velocity, acceleration)
        factor = np.ones(n - 1)
        for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            lo, hi = polynomial_bounds(c, durations, order)
            ratio = np.max(np.maximum(abs(lo[:, body]), abs(hi[:, body])) / getattr(limits, name)[body], axis=1)
            factor = np.maximum(factor, ratio ** (1 / order))
        if np.max(factor) <= 1 + 1e-8:
            lo, hi = polynomial_bounds(c, durations)
            if np.any(lo[:, body] < limits.lower[body] - 1e-10) or np.any(hi[:, body] > limits.upper[body] + 1e-10):
                raise PlanningError("Spline may exceed position bounds; replan required")
            trajectory = Trajectory(
                q, durations, c, phase, indices, limits, iteration + 1, original_points, original_phases, refinements
            )
            error_details = path_error_details(trajectory)
            deviation = error_details["bound_rad"]
            if deviation > max_path_deviation:
                if len(refinements) < max_path_refinements:
                    q, phase, inserted = _refine_path(trajectory, max_path_deviation)
                    refinements.append(inserted)
                    nominal = np.diff(phase) / policy_fps
                    # The old dilations solved a different spline. Reusing
                    # them with a grow-only solver accumulates avoidable delay
                    # at repeatedly split startup segments. Re-solve timing
                    # from the same nominal source clock, with unchanged caps.
                    durations = nominal.copy()
                    n = len(q)
                    since_refinement = 0
                    continue
                raise PlanningError(
                    f"Spline exceeds body path deviation budget: {deviation:.6g} > {max_path_deviation:.6g}; "
                    f"source_interval={error_details['source_interval']}, joint_index={error_details['joint_index']}"
                )
            return trajectory
        grow = factor > 1 + 1e-8
        # Recomputed knot derivatives couple neighboring intervals. A full
        # single-step dilation assumes fixed derivatives and can substantially
        # over-stretch the next iteration. Damping avoids that overshoot; the
        # release criterion above still checks the COMPLETE polynomial bounds.
        durations[grow] *= factor[grow] ** retiming_damping * 1.005
        if np.any(durations > nominal * max_stretch):
            raise PlanningError("Retiming exceeds offline stretch budget")
        if since_refinement >= max_iterations:
            raise PlanningError("Retiming did not converge; no trajectory released")
    raise PlanningError("Retiming did not converge; no trajectory released")
