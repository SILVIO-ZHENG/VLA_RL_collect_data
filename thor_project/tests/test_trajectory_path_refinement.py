"""Local refinement preserves original references and device events, offline only."""

import numpy as np
import pytest

from tools.trajectory_lab.executor import OfflineExecutor
from tools.trajectory_lab.trajectory import (
    Limits,
    PlanningError,
    Trajectory,
    path_error_bound,
    path_error_details,
    plan,
)


def parameters():
    return {
        "policy_fps": 30,
        "limits": Limits(np.ones(3), np.full(3, 5), np.full(3, 80), np.full(3, -2), np.full(3, 2)),
        "device_indices": (2,),
        "tangent_method": "cubic",
    }


def test_large_startup_gap_refined_without_relaxing_bounds():
    points = np.array([[0, 0, 0.025], [0.2, -0.1, 0.004]])
    with pytest.raises(PlanningError, match="path deviation"):
        plan(points, **parameters())
    p = plan(points, **parameters(), max_path_refinements=6)
    assert len(p.path_refinements) == 2
    assert len(p.points) > len(points)
    np.testing.assert_array_equal(p.original_points, points)
    np.testing.assert_array_equal(p.original_phases, [0, 1])
    np.testing.assert_array_equal(p.sample(0).position, points[0])
    np.testing.assert_array_equal(p.sample(p.duration).position, points[-1])
    assert path_error_bound(p) <= 0.02
    assert all(ratio <= 1 + 1e-8 for ratio in p.bounds().values())
    assert np.all(p.durations / (np.diff(p.phases) / 30) <= 30)
    # Added polynomial knots never create extra register events.
    for t in p.times[:-1]:
        assert p.sample(t).position[2] == 0.025
    assert p.sample(p.duration).position[2] == 0.004
    for t in p.times[1:-1]:
        for field in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(
                getattr(p.sample(t - 1e-9), field)[:2], getattr(p.sample(t + 1e-9), field)[:2], atol=1e-6
            )


def test_generalized_bound_covers_dense_sampling_with_nonuniform_shifted_phases():
    points = np.array([[0, 0, 0.025], [0.2, -0.1, 0.004]])
    p = plan(points, **parameters(), phases=[10.0, 11.5], max_path_refinements=6)
    maximum = 0
    phases = []
    for t in np.linspace(0, p.duration, 3001):
        sample = p.sample(t)
        fraction = (sample.phase - 10) / 1.5
        reference = points[0] + fraction * (points[-1] - points[0])
        maximum = max(maximum, float(np.max(abs(sample.position[:2] - reference[:2]))))
        phases.append(sample.phase)
    assert np.all(np.diff(phases) > 0)
    assert maximum <= path_error_bound(p) + 1e-10
    details = path_error_details(p)
    assert details["source_interval"] == 0
    assert 10 <= details["phase_start"] < details["phase_end"] <= 11.5


def test_original_reference_not_replaced_by_refinement_knots():
    # Deliberately invalid intermediate knot: checking against fitted knots
    # would claim zero error; the original body reference is identically zero.
    q = np.array([[0.0, 0, 0], [0.1, 0, 1], [0, 0, 0]])
    coefficients = np.zeros((2, 6, 3))
    coefficients[:, 0] = q[:-1]
    coefficients[:, 1] = np.diff(q, axis=0)
    p = Trajectory(
        q,
        [0.5, 0.5],
        coefficients,
        [0, 0.5, 1],
        (2,),
        parameters()["limits"],
        1,
        original_points=np.zeros((2, 3)),
        original_phases=[0, 1],
    )
    assert path_error_bound(p) == pytest.approx(0.1)
    assert path_error_details(p)["joint_index"] == 0


@pytest.mark.parametrize("rounds", [-1, 9, 1.5, True, None])
def test_invalid_refinement_budget_rejected(rounds):
    with pytest.raises(PlanningError):
        plan([[0, 0, 0], [0.01, 0, 0]], **parameters(), max_path_refinements=rounds)


def test_refinement_budget_exhaustion_rejects_not_releases():
    with pytest.raises(PlanningError, match="path deviation"):
        plan([[0, 0, 0], [0.2, -0.1, 0]], **parameters(), max_path_refinements=1)


def test_refinement_does_not_disable_stretch_budget():
    with pytest.raises(PlanningError, match="stretch budget"):
        plan([[0, 0, 0], [0.3, -0.15, 0]], **parameters(), max_path_refinements=6)


def test_refinement_disabled_or_unneeded_preserves_result():
    points = np.array([[0, 0, 0.025], [0.01, -0.005, 0]])
    original = plan(points, **parameters())
    refined = plan(points, **parameters(), max_path_refinements=6)
    np.testing.assert_array_equal(original.coefficients, refined.coefficients)
    np.testing.assert_array_equal(original.durations, refined.durations)
    assert refined.path_refinements == ()


def test_refinement_keeps_moving_start_derivatives():
    v, a = [0.01, 0, 0], [0.01, 0, 0]
    p = plan(
        [[0, 0, 0.025], [0.15, -0.075, 0]],
        **parameters(),
        start_velocity=v,
        start_acceleration=a,
        max_path_refinements=6,
    )
    assert p.path_refinements
    np.testing.assert_allclose(p.sample(0).velocity, v, atol=1e-12)
    np.testing.assert_allclose(p.sample(0).acceleration, a, atol=1e-12)
    assert path_error_bound(p) <= 0.02
    assert all(ratio <= 1 + 1e-8 for ratio in p.bounds().values())


def test_refined_points_pass_through_checked_queue_at_100hz():
    points = np.array([[0, 0, 0.025], [0.2, -0.1, 0.004]])
    p = plan(points, **parameters(), max_path_refinements=6)
    executor = OfflineExecutor(points[0])
    executor.propose(p, executor.request_anchor())
    checked = []

    def checker(previous, candidate):
        checked.append(candidate.copy())
        return True

    received = []
    ticks = int(np.ceil(p.duration / 0.01))
    for _ in range(ticks):
        received.append(executor.tick(checker).position.copy())
    np.testing.assert_array_equal(received, checked)
    np.testing.assert_array_equal(received[-1], points[-1])
    assert len(checked) == ticks
    assert all(q[2] == 0.025 for q in received[:-1])
    assert received[-1][2] == 0.004
