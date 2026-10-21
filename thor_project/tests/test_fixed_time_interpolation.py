import numpy as np

from tools.compare_fixed_time_interpolation import (
    device_event_times,
    evaluate_reference,
    sample_times,
)


def test_c1_reference_passes_all_knots_without_retiming():
    points = np.array([[0.0, 0.0], [0.2, -0.1], [0.3, 0.4], [0.8, 0.2]])
    knots = np.arange(len(points)) / 30
    actual = evaluate_reference(points, knots, policy_fps=30, method="pchip_c1")
    np.testing.assert_allclose(actual, points, atol=1e-14)
    times = sample_times(len(points), 30, 100)
    assert times[-1] == (len(points) - 1) / 30


def test_servo_grid_never_extrapolates_past_unaligned_policy_horizon():
    times = sample_times(50, 30, 100)
    duration = 49 / 30
    assert times[-1] <= duration
    assert duration - times[-1] < 0.01


def test_shape_preserving_c1_does_not_overshoot_monotone_scalar_segments():
    points = np.array([[0.0], [0.1], [0.15], [0.5]])
    times = sample_times(len(points), 30, 1000)
    actual = evaluate_reference(points, times, policy_fps=30, method="pchip_c1")[:, 0]
    assert np.all(np.diff(actual) >= -1e-14)
    assert actual.min() >= points.min()
    assert actual.max() <= points.max()


def test_c1_velocity_is_continuous_at_internal_knots():
    points = np.array([[0.0], [0.2], [-0.1], [0.4]])
    knots = np.arange(len(points)) / 30
    epsilon = 1e-9
    for knot in knots[1:-1]:
        left = evaluate_reference(points, [knot - epsilon, knot], policy_fps=30, method="pchip_c1")
        right = evaluate_reference(points, [knot, knot + epsilon], policy_fps=30, method="pchip_c1")
        np.testing.assert_allclose((left[1] - left[0]) / epsilon, (right[1] - right[0]) / epsilon, atol=2e-5)


def test_device_event_times_are_source_knot_times():
    values = np.array([[0.025, 0.0], [0.025, 0.025], [0.0, 0.025], [0.0, 0.0]])
    left, right = device_event_times(values, policy_fps=30)
    np.testing.assert_allclose(left, [2 / 30])
    np.testing.assert_allclose(right, [1 / 30, 3 / 30])
