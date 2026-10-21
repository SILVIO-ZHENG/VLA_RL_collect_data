"""Pure offline checks; never connects to G2 or imports its SDK."""

import numpy as np
import pytest

from tools.trajectory_lab.executor import OfflineExecutor
from tools.trajectory_lab.trajectory import Limits, PlanningError, path_error_bound, plan, polynomial_bounds


def limits(n=3, velocity=0.6, acceleration=2.0, jerk=10.0):
    return Limits(np.full(n, velocity), np.full(n, acceleration), np.full(n, jerk), np.full(n, -3.0), np.full(n, 3.0))


def build(points, **kwargs):
    return plan(np.array(points), policy_fps=30, limits=limits(np.shape(points)[1]), **kwargs)


def assert_bounded(trajectory):
    assert all(value <= 1 + 1e-8 for value in trajectory.bounds().values())
    body = [i for i in range(trajectory.points.shape[1]) if i not in trajectory.device_indices]
    for t in np.linspace(0, trajectory.duration, 1001):
        s = trajectory.sample(t)
        for field in ("velocity", "acceleration", "jerk"):
            assert np.all(abs(getattr(s, field)[body]) <= getattr(trajectory.limits, field)[body] + 1e-8)


def test_static_path_keeps_30hz_clock_not_25hz():
    p = build(np.zeros((31, 3)))
    assert p.duration == pytest.approx(1.0)
    assert p.sample(0.5).phase == pytest.approx(15.0)
    assert_bounded(p)


def test_shared_clock_gripper_events_and_derivative_bounds():
    q = np.array([[0, 0, 0.025], [0.02, -0.01, 0.025], [0.06, -0.02, 0], [0.08, -0.03, 0]])
    p = build(q, device_indices=(2,))
    assert p.duration > 0.1
    assert_bounded(p)
    for k, t in enumerate(p.times):
        s = p.sample(t)
        np.testing.assert_allclose(s.position, q[k], atol=1e-10)
        assert s.phase == pytest.approx(k)
    assert p.sample(p.times[2] - 1e-8).position[2] == 0.025
    assert p.sample(p.times[2]).position[2] == 0
    assert p.progress(0.1)["source_phase"] < 0.1 * 30
    # Hardware register jumps are not advertised as bounded motor derivatives.
    assert np.max(abs(np.diff(q[:, 2]))) == 0.025


def test_c2_knots_and_nonzero_speed_through_same_direction_points():
    p = build([[0.0, 0.0], [0.01, 0.005], [0.02, 0.01], [0.03, 0.015], [0.04, 0.02]])
    for t in p.times[1:-1]:
        left, right = p.sample(t - 1e-8), p.sample(t + 1e-8)
        for field in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(left, field), getattr(right, field), atol=1e-6)
        assert np.linalg.norm(p.sample(t).velocity) > 0.001
    assert_bounded(p)


def test_suffix_preserves_moving_boundary_q_v_a():
    first = build([[0.0, 0.0], [0.04, 0.01], [0.08, 0.02], [0.12, 0.03]])
    tail = first.sample(first.duration * 0.45)
    q = np.stack([tail.position, tail.position + [0.02, 0.01], tail.position + [0.06, 0.02]])
    second = build(q, start_velocity=tail.velocity, start_acceleration=tail.acceleration)
    for field in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(second.sample(0), field), getattr(tail, field), atol=1e-10)
    assert_bounded(second)


@pytest.mark.parametrize("seed", range(8))
def test_random_reversals_are_bounded(seed):
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, 0.008, (12, 3)), axis=0)
    p = build(q)
    assert_bounded(p)


def test_polynomial_hulls_cover_between_samples():
    rng = np.random.default_rng(22)
    coefficients = rng.normal(size=(4, 6, 3))
    durations = np.array([0.1, 0.2, 0.5, 1.0])
    for order in range(4):
        lo, hi = polynomial_bounds(coefficients, durations, order)
        for k in range(4):
            for j in range(3):
                c = np.polynomial.polynomial.polyder(coefficients[k, :, j], order) / durations[k] ** order
                values = np.polynomial.polynomial.polyval(np.linspace(0, 1, 500), c)
                assert values.min() >= lo[k, j] - 1e-9
                assert values.max() <= hi[k, j] + 1e-9


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_values_fail_without_outputs(bad):
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [bad, 0.0]])
    p = build([[0.0, 0.0], [0.01, 0.0]])
    with pytest.raises(PlanningError):
        p.sample(bad)


def test_limits_and_input_validation():
    with pytest.raises(PlanningError):
        limits(jerk=0)
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [4.0, 0.0]])
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [0.01, 0.0]], phases=[1, 1])
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [0.01, 0.0]], start_velocity=[0.7, 0.0])
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [0.01, 0.0]], device_indices=(1, 1))
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [1.0, 0.0]], max_stretch=1.0)


def test_input_and_samples_cannot_mutate_planned_points():
    q = np.array([[0.0, 0.0], [0.1, 0.0]])
    p = build(q)
    q[:] = 1.0
    assert p.points[0, 0] == 0
    with pytest.raises(ValueError):
        p.sample(0.1).position.setflags(write=True)
    with pytest.raises(ValueError):
        p.coefficients.setflags(write=True)


def test_checked_prefix_is_immutable_and_suffix_preserves_boundary():
    e = OfflineExecutor([0.0, 0.0])
    p = build([[0.0, 0.0], [0.02, 0.0], [0.04, 0.0]])
    e.propose(p, e.request_anchor())
    checked = []

    def checker(a, b):
        checked.append(b.copy())
        return True

    e.prepare(checker)
    prefix = e.committed
    request = e.request_anchor()
    tail = request.frozen_sample
    suffix = build(
        [tail.position, tail.position + [0.04, 0.02], tail.position + [0.08, 0.02]],
        start_velocity=tail.velocity,
        start_acceleration=tail.acceleration,
    )
    e.propose(suffix, request)
    assert len(e.committed) == 8
    for a, b in zip(prefix, e.committed, strict=True):
        np.testing.assert_array_equal(a.position, b.position)
    for item in prefix:
        np.testing.assert_array_equal(e.tick(checker).position, item.position)

    # Every adopted target was checked exactly; checker cannot mutate the sample.
    def malicious_checker(a, b):
        b[:] = 100
        return True

    e.prepare(malicious_checker)
    assert all(np.max(s.position) < 1 for s in e.committed)


def test_source_progress_advances_on_adoption_not_preparation():
    e = OfflineExecutor([0.0, 0.0])
    p = build([[0.0, 0.0], [0.1, 0.0]])
    e.propose(p, e.request_anchor())
    e.prepare(lambda a, b: True)
    anchor = e.request_anchor()
    assert anchor.executed_phase == 0
    assert anchor.frozen_sample.phase > 0
    e.tick(lambda a, b: True)
    assert e.last.phase > 0


def test_collision_holds_warns_and_new_safe_suffix_recovers(caplog):
    e = OfflineExecutor([0.0, 0.0])
    e.propose(build([[0.0, 0.0], [0.1, 0.0]]), e.request_anchor())
    stale = e.request_anchor()
    e.tick(lambda a, b: False)
    assert e.status == "blocked"
    assert len(e.committed) == 0
    assert "holding last adopted target" in caplog.text
    for _ in range(10):
        np.testing.assert_array_equal(e.tick(lambda a, b: True).position, [0, 0])
    assert e.last.phase == 0
    with pytest.raises(PlanningError, match="Stale"):
        e.propose(build([[0.0, 0.0], [-0.1, 0.0]]), stale)
    e.propose(build([[0.0, 0.0], [-0.1, 0.0]]), e.request_anchor())
    assert e.tick(lambda a, b: True).position[0] < 0


def test_pause_resume_reanchors_and_invalidates_old_response():
    e = OfflineExecutor([0.0, 0.0])
    p = build([[0.0, 0.0], [0.1, 0.0]])
    old = e.request_anchor()
    e.propose(p, old)
    e.tick(lambda a, b: True)
    e.pause()
    assert not e.committed
    with pytest.raises(PlanningError):
        e.propose(p, old)
    e.resume([0.05, 0.0])
    p = build([[0.05, 0.0], [0.1, 0.0]])
    with pytest.raises(PlanningError, match="Stale"):
        e.propose(p, old)
    e.propose(p, e.request_anchor())
    assert e.tick(lambda a, b: True).position[0] >= 0.05
    e.end()
    with pytest.raises(PlanningError):
        e.propose(p, e.request_anchor())


def test_checker_exception_never_releases_old_queued_motion():
    e = OfflineExecutor([0.0, 0.0])
    e.propose(build([[0.0, 0.0], [0.1, 0.0]]), e.request_anchor())

    def broken(a, b):
        raise RuntimeError("geometry unavailable")

    with pytest.raises(RuntimeError):
        e.tick(broken)
    assert e.status == "checker_error" and not e.committed
    np.testing.assert_array_equal(e.tick(lambda a, b: True).position, [0, 0])


def test_checker_must_not_return_truthy_object():
    e = OfflineExecutor([0.0, 0.0])
    e.propose(build([[0.0, 0.0], [0.1, 0.0]]), e.request_anchor())
    with pytest.raises(PlanningError, match="boolean"):
        e.tick(lambda a, b: "safe")


def test_runtime_has_no_lab_imports():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in (root / "src" / "nexus_deploy").rglob("*.py"):
        assert "trajectory_lab" not in path.read_text()


def test_simplification_preserves_every_device_event():
    q = np.zeros((31, 3))
    q[:, 2] = np.where(np.arange(31) % 3 == 0, 0.025, 0.0)
    p = build(q, device_indices=(2,), knot_tolerance=0.003, smoothing_window=9)
    assert len(p.points) == 2
    assert p.duration == pytest.approx(1.0)
    for t in np.arange(0.0, p.duration, 0.01):
        s = p.sample(t)
        assert s.position[2] == q[int(np.floor(s.phase)), 2]
    assert p.sample(p.duration).position[2] == q[-1, 2]


def test_prefit_bounds_original_path_and_keeps_gripper_unsmoothed():
    x = np.linspace(0, 2 * np.pi, 101)
    q = np.stack((0.02 * np.sin(x), 0.01 * np.sin(x), np.where(np.arange(101) > 50, 0.0, 0.025)), axis=1)
    p = build(q, device_indices=(2,), knot_tolerance=0.001, smoothing_window=9)
    assert path_error_bound(p) <= 0.02
    assert_bounded(p)
    for phase in (49.9, 50.0, 50.9, 51.0):
        t = np.interp(phase, p.phases, p.times)
        assert p.sample(t).position[2] == (0.025 if phase < 51 else 0.0)


def test_path_deviation_limit_rejects_instead_of_silently_relaxing():
    with pytest.raises(PlanningError, match="path deviation"):
        build([[0.0, 0.0], [0.1, 0.0]], max_path_deviation=0.00001)


def test_old_request_cannot_replace_newer_pending_response():
    e = OfflineExecutor([0.0, 0.0])
    request = e.request_anchor()
    p = build([[0.0, 0.0], [0.01, 0.0]])
    e.propose(p, request)
    with pytest.raises(PlanningError, match="Stale"):
        e.propose(p, request)


def test_rtc_snapshot_distinguishes_executed_and_frozen_chunk():
    e = OfflineExecutor([0.0, 0.0])
    p = build([[0.0, 0.0], [0.01, 0.0]])
    e.propose(p, e.request_anchor())
    e.prepare(lambda a, b: True)
    snapshot = e.request_anchor()
    assert snapshot.executed_source_id is None
    assert snapshot.frozen_source_id == p.source_id
    e.tick(lambda a, b: True)
    assert e.request_anchor().executed_source_id == p.source_id


@pytest.mark.parametrize("window", [0, 2, 3, 17])
def test_invalid_prefit_window_rejected(window):
    with pytest.raises(PlanningError):
        build([[0.0, 0.0], [0.01, 0.0]], smoothing_window=window)
