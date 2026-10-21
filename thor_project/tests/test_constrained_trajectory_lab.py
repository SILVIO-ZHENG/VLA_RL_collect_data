"""Offline convex handoffs: fixed boundaries, original events, fail closed."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy.optimize")

from tools.trajectory_lab.constrained import fit_window
from tools.trajectory_lab.replay import plan_recorded
from tools.trajectory_lab.trajectory import Limits, PlanningError, path_error_bound


def arguments():
    return dict(
        points=np.array([[0.0, 0.025], [0.01, 0.0], [0.02, 0.015], [0.02, 0.025]]),
        phases=np.arange(4) + 10.0,
        policy_fps=2,
        limits=Limits(np.ones(2), np.full(2, 2), np.full(2, 10), -np.ones(2), np.ones(2)),
        start_velocity=np.array([0.02, 0]),
        start_acceleration=np.zeros(2),
        device_indices=(1,),
        budget_s=3,
    )


def verify(p, original, v0, a0):
    np.testing.assert_array_equal(p.original_points, original)
    np.testing.assert_allclose(p.sample(0).position, original[0], atol=1e-9)
    np.testing.assert_allclose(p.sample(0).velocity, v0, atol=1e-9)
    np.testing.assert_allclose(p.sample(0).acceleration, a0, atol=1e-9)
    assert max(p.bounds().values()) <= 1 + 1e-8
    assert path_error_bound(p) <= 0.02
    body = [j for j in range(original.shape[1]) if j not in p.device_indices]
    for t in p.times[1:-1]:
        left, right = p.sample(t - 1e-9), p.sample(t + 1e-9)
        for name in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(left, name)[body], getattr(right, name)[body], atol=1e-6)
    for index, phase in enumerate(p.original_phases):
        t = np.interp(phase, p.phases, p.times)
        sample = p.sample(min(t + 1e-9, p.duration))
        np.testing.assert_array_equal(sample.position[list(p.device_indices)], original[index, list(p.device_indices)])
    for t in np.linspace(0, p.duration, 501):
        sample = p.sample(t)
        ref = [np.interp(sample.phase, p.original_phases, original[:, j]) for j in body]
        assert np.max(abs(sample.position[body] - ref)) <= 0.02 + 1e-10
        for name in ("velocity", "acceleration", "jerk"):
            assert np.all(abs(getattr(sample, name)[body]) <= getattr(p.limits, name)[body] * (1 + 1e-8))


@pytest.mark.parametrize("subdivisions", [1, 4])
def test_convex_fit_is_c2_bounded_and_preserves_gripper_events(subdivisions):
    args = arguments()
    p = fit_window(**args, hull_subdivisions=subdivisions)
    verify(p, args["points"], args["start_velocity"], args["start_acceleration"])


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_s": 0},
        {"budget_s": float("nan")},
        {"policy_fps": 0},
        {"hull_subdivisions": True},
        {"optimize_jerk": "yes"},
        {"phases": [10, 11, 11, 12]},
        {"start_velocity": [0.02, 0.01]},
        {"start_acceleration": [float("nan"), 0]},
        {"device_indices": (1, 1)},
        {"interval_durations": [0.4, 0.5, 0.5]},
        {"max_path_deviation": -0.1},
        {"max_stretch": 1, "time_scale": 1.1},
        {"max_stretch": 1, "interval_durations": [0.51, 0.5, 0.5]},
        {"servo_period": 0.6},
    ],
)
def test_invalid_convex_inputs_reject(changes):
    with pytest.raises(PlanningError):
        fit_window(**(arguments() | changes))


def test_solver_failure_never_returns_unverified_candidate(monkeypatch):
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k: SimpleNamespace(success=False, message="time limit"))
    with pytest.raises(PlanningError, match="time limit"):
        fit_window(**arguments())


def test_solver_success_alone_is_not_a_certificate(monkeypatch):
    # Malformed solution has the right dimension, but violates the fixed start.
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k: SimpleNamespace(success=True, x=np.zeros(17)))
    with pytest.raises(PlanningError, match="starting boundary"):
        fit_window(**arguments())


def test_static_body_fast_path_still_preserves_all_device_events():
    args = arguments()
    args["points"][:, 0] = 0.01
    args["start_velocity"][:] = 0
    p = fit_window(**args)
    verify(p, args["points"], args["start_velocity"], args["start_acceleration"])


def test_verification_cost_is_included_in_budget(monkeypatch):
    clock = iter([0, 0, 0, 4])
    monkeypatch.setattr("tools.trajectory_lab.constrained.time.perf_counter", lambda: next(clock))
    with pytest.raises(PlanningError, match="time budget"):
        fit_window(**arguments())


def test_end_rest_sample_does_not_hide_invalid_polynomial(monkeypatch):
    from scipy.optimize import linprog

    def malformed(*args, **kwargs):
        result = linprog(*args, **kwargs)
        assert result.success
        result.x[10] = 0.001  # final body velocity: sample(duration) reports rest
        return result

    monkeypatch.setattr("scipy.optimize.linprog", malformed)
    with pytest.raises(PlanningError, match="ending boundary"):
        fit_window(**arguments())


def test_fallback_runs_after_fast_attempts_and_shares_one_budget(monkeypatch):
    calls, convex_calls = [], []
    elapsed = [0.0]

    def fast(points, **kwargs):
        if kwargs.get("start_velocity") is None:
            return SimpleNamespace(phases=np.arange(4) + 10, times=np.arange(4) * 0.5)
        calls.append(kwargs["tangent_method"])
        raise PlanningError("unsolved moving boundary")

    def expensive(*args, **kwargs):
        assert len(calls) == 5
        convex_calls.append(kwargs["budget_s"])
        assert kwargs["max_stretch"] == 2
        assert kwargs["servo_period"] == 0.02
        elapsed[0] += 0.31
        raise PlanningError("unsolved convex window")

    monkeypatch.setattr("tools.trajectory_lab.replay.plan", fast)
    monkeypatch.setattr("tools.trajectory_lab.constrained.fit_window", expensive)
    monkeypatch.setattr("tools.trajectory_lab.replay.time.perf_counter", lambda: elapsed[0])
    args = arguments()
    args.pop("budget_s")
    with pytest.raises(PlanningError, match="Shared convex fallback budget exhausted"):
        plan_recorded(**args, constrained_fallback=True, max_stretch=2, servo_period=0.02)
    assert convex_calls == [0.35]  # no second independent 350 ms budget


@pytest.mark.parametrize("case_index", [0, 1])
def test_saved_moving_failures_solve_without_relaxing_original_constraints(case_index):
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "tests/fixtures/trajectory_moving_boundaries.json").read_text())["cases"][case_index]
    profile = json.loads((root / "tools/trajectory_lab/profiles/dh_30hz_20260914.offline.json").read_text())
    vectors = []
    for name in ("velocity", "acceleration", "jerk"):
        value = np.ones(16)
        value[profile["joint_indices"]] = profile["limits"][name]
        vectors.append(value)
    lower, upper = np.full(16, -4.0), np.full(16, 4.0)
    lower[[7, 15]], upper[[7, 15]] = 0, 0.025
    options = dict(policy_fps=30, limits=Limits(*vectors, lower, upper), device_indices=(7, 15), smoothing_window=9)
    points, phases = np.asarray(saved["points"]), np.asarray(saved["phases"])
    # These exact moving boundaries failed every former geometric planner.
    with pytest.raises(PlanningError, match="All offline candidates rejected"):
        plan_recorded(
            points,
            phases=phases,
            start_velocity=saved["start_velocity"],
            start_acceleration=saved["start_acceleration"],
            **options,
        )
    seed, _ = plan_recorded(points, phases=phases, **options)
    nominal = np.diff(phases) / 30
    seed_h = np.diff(np.interp(phases, seed.phases, seed.times))
    h = nominal + np.maximum(0, seed_h - nominal) * np.minimum(1, np.arange(len(nominal)) / 24)
    p = fit_window(
        points,
        phases=phases,
        start_velocity=saved["start_velocity"],
        start_acceleration=saved["start_acceleration"],
        interval_durations=h,
        budget_s=3,
        **{k: v for k, v in options.items() if k != "smoothing_window"},
    )
    verify(p, points, saved["start_velocity"], saved["start_acceleration"])


def test_optional_solver_not_used_when_existing_planner_succeeds(monkeypatch):
    def unexpected(*a, **k):
        pytest.fail("Convex fallback must not replace a successful existing plan")

    monkeypatch.setattr("tools.trajectory_lab.constrained.fit_window", unexpected)
    args = arguments()
    args.pop("budget_s")
    p, attempts = plan_recorded(**args, constrained_fallback=True)
    assert attempts[-1]["status"] == "bounded"
    assert all(a["method"] != "convex_jerk" for a in attempts)
