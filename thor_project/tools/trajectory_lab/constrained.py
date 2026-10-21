"""Small-window, fixed-time convex trajectory fit. OFFLINE ONLY (SciPy/HiGHS).

Optimize body knot q/v/a together under continuous Bernstein bounds. Reference
positions are allowed to move only inside the EXISTING path budget; device
events and the fixed starting q/v/a are never optimization variables.
"""

import time
from math import comb

import numpy as np

from .trajectory import _HULLS, Limits, PlanningError, Trajectory, _coefficients, path_error_bound, polynomial_bounds


def fit_window(
    points,
    *,
    phases,
    policy_fps,
    limits,
    start_velocity,
    start_acceleration,
    device_indices=(7, 15),
    time_scale=1.0,
    interval_durations=None,
    max_path_deviation=0.02,
    budget_s=1.0,
    optimize_jerk=True,
    hull_subdivisions=1,
    max_stretch=30.0,
    servo_period=0.01,
):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, vstack

    started = time.perf_counter()
    q = np.asarray(points, dtype=float)
    phase = np.asarray(phases, dtype=float)
    v0, a0 = np.asarray(start_velocity, dtype=float), np.asarray(start_acceleration, dtype=float)
    if q.ndim != 2 or not 2 <= len(q) <= 65 or not np.isfinite(q).all():
        raise PlanningError("Convex fitter requires 2..65 finite source points")
    n, d = q.shape
    if type(hull_subdivisions) is not int or hull_subdivisions not in (1, 4):
        raise PlanningError("Hull subdivisions must be 1 or 4")
    if type(optimize_jerk) is not bool:
        raise PlanningError("Optimize jerk must be boolean")
    if not isinstance(limits, Limits) or limits.velocity.shape != (d,):
        raise PlanningError("Convex fitter limits dimension mismatch")
    if phase.shape != (n,) or not np.isfinite(phase).all() or np.any(np.diff(phase) <= 0):
        raise PlanningError("Convex fitter requires increasing source phases")
    if v0.shape != (d,) or a0.shape != (d,) or not np.isfinite(v0).all() or not np.isfinite(a0).all():
        raise PlanningError("Invalid moving boundary")
    if (
        not np.isfinite([time_scale, policy_fps, max_path_deviation, budget_s, max_stretch, servo_period]).all()
        or not 1 <= time_scale <= 3
        or policy_fps <= 0
        or max_path_deviation <= 0
        or not 0 < budget_s <= 5
        or max_stretch < 1
        or servo_period <= 0
        or time_scale > max_stretch
    ):
        raise PlanningError("Invalid convex fitting budget")
    indices = tuple(device_indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < d for i in indices):
        raise PlanningError("Invalid convex device indices")
    body = [j for j in range(d) if j not in indices]
    if not body or np.any(v0[list(indices)]) or np.any(a0[list(indices)]):
        raise PlanningError("Device derivatives must be zero; body required")
    if (
        np.any(q < limits.lower)
        or np.any(q > limits.upper)
        or np.any(abs(v0) > limits.velocity)
        or np.any(abs(a0) > limits.acceleration)
    ):
        raise PlanningError("Convex source or boundary exceeds limits")
    durations = np.diff(phase) / policy_fps * time_scale
    if interval_durations is not None:
        durations = np.asarray(interval_durations, dtype=float)
        nominal = np.diff(phase) / policy_fps
        if (
            durations.shape != (n - 1,)
            or not np.isfinite(durations).all()
            or np.any(durations < nominal - 1e-10)
            or np.any(durations > nominal * max_stretch)
        ):
            raise PlanningError("Convex interval durations exceed existing timing bounds")
    if np.any(np.diff(phase) / policy_fps < servo_period - 1e-10):
        raise PlanningError("Original source events exceed servo frequency")
    # Linear mapping from per-knot [q, v, a] to quintic power coefficients.
    basis = [np.zeros((n, 3 * n)) for _ in range(3)]
    for order in range(3):
        basis[order][np.arange(n), 3 * np.arange(n) + order] = 1
    coefficients = _coefficients(basis[0], durations, basis[1], basis[2])
    matrices = []
    for order in range(4):
        c = coefficients[:, order:].copy()
        for k in range(order, 6):
            multiplier = 1
            for offset in range(order):
                multiplier *= k - offset
            c[:, k - order] *= multiplier
        c /= durations[:, None, None] ** order
        degree = 5 - order
        hull = (
            _HULLS[degree]
            if hull_subdivisions == 4
            else np.array(
                [
                    [
                        [comb(k, i) / comb(degree, i) if i <= k else 0 for i in range(degree + 1)]
                        for k in range(degree + 1)
                    ]
                ]
            )
        )
        matrices.append(csr_matrix(np.einsum("ski,nij->nskj", hull, c).reshape(-1, 3 * n)))
    # Body path reference at the same Bernstein abscissae, for each source interval.
    u = ((np.arange(hull_subdivisions)[:, None] + np.arange(6)[None] / 5) / hull_subdivisions).reshape(-1)
    reference = (q[:-1, None] + (q[1:] - q[:-1])[:, None] * u[None, :, None]).reshape(-1, d)
    base = vstack([hstack((m, csr_matrix((m.shape[0], n + 1)))) for m in matrices[:3]], format="csr")
    deviation = np.zeros((n, 3 * n))
    deviation[np.arange(n), 3 * np.arange(n)] = 1
    deviation = csr_matrix(deviation)
    slack = csr_matrix(np.eye(n))
    cost = np.zeros(4 * n + 1)
    cost[3 * n : 4 * n] = 0.05 / (n * max_path_deviation)
    cost[-1] = 1.0  # minimize peak normalized jerk, not only position error
    if not optimize_jerk:
        cost[:] = 0  # bounded feasibility pass; not a minimum-jerk claim
    fitted, velocities, accelerations = q.copy(), np.zeros_like(q), np.zeros_like(q)
    # Fail on the most demanding initial joint first instead of solving easy joints first.
    body.sort(key=lambda j: max(abs(v0[j]) / limits.velocity[j], abs(a0[j]) / limits.acceleration[j]), reverse=True)
    for joint in body:
        if np.max(abs(q[:, joint] - q[0, joint])) < 1e-12 and max(abs(v0[joint]), abs(a0[joint])) < 1e-12:
            fitted[:, joint] = q[0, joint]
            velocities[0, joint], accelerations[0, joint] = v0[joint], a0[joint]
            continue
        remaining = budget_s - (time.perf_counter() - started)
        if remaining <= 0:
            raise PlanningError("Convex fitting time budget exhausted")
        jerk = hstack(
            (
                matrices[3],
                csr_matrix((matrices[3].shape[0], n)),
                csr_matrix(np.full((matrices[3].shape[0], 1), -limits.jerk[joint])),
            ),
            format="csr",
        )
        negative_jerk = hstack((-matrices[3], jerk[:, 3 * n :]), format="csr")
        aa = vstack(
            (
                base,
                -base,
                jerk,
                negative_jerk,
                hstack((deviation, -slack, csr_matrix((n, 1)))),
                hstack((-deviation, -slack, csr_matrix((n, 1)))),
            ),
            format="csr",
        )
        margin = 1 - 1e-7
        upper = np.r_[
            np.minimum(reference[:, joint] + max_path_deviation * margin, limits.upper[joint]),
            np.full(matrices[1].shape[0], limits.velocity[joint]),
            np.full(matrices[2].shape[0], limits.acceleration[joint]),
        ]
        lower = np.r_[
            np.maximum(reference[:, joint] - max_path_deviation * margin, limits.lower[joint]),
            np.full(matrices[1].shape[0], -limits.velocity[joint]),
            np.full(matrices[2].shape[0], -limits.acceleration[joint]),
        ]
        rhs = np.r_[upper, -lower, np.zeros(2 * matrices[3].shape[0]), q[:, joint], -q[:, joint]]
        bounds = [(None, None)] * (3 * n) + [(0, None)] * n + [(0, margin)]
        for index, value in [
            (0, q[0, joint]),
            (1, v0[joint]),
            (2, a0[joint]),
            (3 * (n - 1), q[-1, joint]),
            (3 * (n - 1) + 1, 0),
            (3 * (n - 1) + 2, 0),
        ]:
            bounds[index] = (value, value)
        result = linprog(
            cost,
            A_ub=aa,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={
                "time_limit": min(remaining, 0.2),
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            raise PlanningError(f"Convex fitting rejected joint {joint}: {result.message}")
        fitted[:, joint], velocities[:, joint], accelerations[:, joint] = result.x[: 3 * n].reshape(n, 3).T
    c = _coefficients(fitted, durations, velocities, accelerations)
    if time.perf_counter() - started > budget_s:
        raise PlanningError("Convex fitting time budget exhausted")
    trajectory = Trajectory(fitted, durations, c, phase, indices, limits, 1, q, phase)
    for name, expected in (("position", q[0]), ("velocity", v0), ("acceleration", a0)):
        if not np.allclose(getattr(trajectory.sample(0), name), expected, rtol=0, atol=1e-9):
            raise PlanningError("Convex result changed fixed starting boundary")
    # Inspect polynomial end directly: sample(duration) intentionally returns a
    # rest endpoint and must not conceal a malformed final polynomial.
    for order, expected in ((0, q[-1, body]), (1, np.zeros(len(body))), (2, np.zeros(len(body)))):
        weights = np.array([np.prod(range(k - order + 1, k + 1)) for k in range(order, 6)])
        endpoint = weights @ c[-1, order:, :][:, body] / durations[-1] ** order
        if not np.allclose(endpoint, expected, rtol=0, atol=1e-8):
            raise PlanningError("Convex result changed fixed ending boundary")
    if max(trajectory.bounds().values()) > 1 + 1e-8 or path_error_bound(trajectory) > max_path_deviation:
        raise PlanningError("Convex result failed independent continuous verification")
    lo, hi = polynomial_bounds(c, durations)
    if np.any(lo[:, body] < limits.lower[body] - 1e-10) or np.any(hi[:, body] > limits.upper[body] + 1e-10):
        raise PlanningError("Convex result failed position verification")
    if time.perf_counter() - started > budget_s:
        raise PlanningError("Convex fitting time budget exhausted")
    return trajectory
