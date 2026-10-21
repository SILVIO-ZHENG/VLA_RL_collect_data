import numpy as np

from tools.analyze_deployment_limit_compatibility import (
    candidate_profile,
    contiguous_run_lengths,
    derivative_blocks,
    interpolate_reference,
    percentile_rows,
    replay_points,
    summarize_derivatives,
)


def test_derivatives_never_cross_episode_boundary():
    first = np.arange(20, dtype=float)[:, None] * np.ones((1, 14))
    second = (1000 + np.arange(20, dtype=float))[:, None] * np.ones((1, 14))
    points = np.vstack((first, second))
    ids = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    result = derivative_blocks(points, ids, fps=1, window=9)
    assert len(result["raw"]) == len(result["fitted"]) == 2
    for block in result["raw"]:
        np.testing.assert_allclose(block["velocity"], 1)


def test_percentiles_and_runs_are_deterministic():
    values = np.arange(1, 101, dtype=float)[:, None]
    stats = percentile_rows(values)
    assert stats["p80"][0] == 80.2
    assert stats["p99"][0] == 99.01
    assert contiguous_run_lengths([False, True, True, False, True]) == [2, 1]


def test_candidate_uses_larger_action_or_state_and_floor():
    channels = {}
    for channel, velocity, acceleration in (
        ("action", 0.4, 3.0),
        ("observation.state", 1.0, 1.0),
    ):
        channels[channel] = {
            "fitted": {
                "velocity": {"task_active": {"p99": [velocity] * 14}},
                "acceleration": {"task_active": {"p99": [acceleration] * 14}},
            }
        }
    result = candidate_profile(
        channels, "p99", margin=1.1, velocity_floor=0.6, acceleration_floor=2.0
    )
    np.testing.assert_allclose(result["velocity"], 1.1)
    np.testing.assert_allclose(result["acceleration"], 3.3)


def test_reference_interpolation_preserves_nominal_duration():
    points = np.array([[0.0], [1.0], [2.0]])
    reference = interpolate_reference(points, policy_fps=10, servo_fps=100)
    assert len(reference) == 21
    assert reference[0, 0] == 0
    assert reference[10, 0] == 1
    assert reference[-1, 0] == 2


def test_fitted_derivatives_share_one_centered_activity_mask():
    t = np.arange(30, dtype=float)
    block = {name: np.tile(t[:, None], (1, 14)) for name in ("velocity", "acceleration", "jerk")}
    limits = {"velocity": np.full(14, 100), "acceleration": np.full(14, 100)}
    result = summarize_derivatives([block], active_threshold=5, limits=limits, fps=30)
    assert result["velocity"]["task_active_samples"] == 25
    assert result["acceleration"]["task_active_samples"] == 25
    assert result["jerk"]["task_active_samples"] == 25


class _TrackingLimiter:
    def __init__(self, *, sample_hz, max_velocity, max_acceleration):
        self.dt = 1 / sample_hz
        self.velocity_limit = np.asarray(max_velocity)

    def reset(self, value):
        self.value = np.asarray(value).copy()

    def update_tracking(self, target, target_velocity):
        velocity = np.clip(target_velocity, -self.velocity_limit, self.velocity_limit)
        self.value = self.value + velocity * self.dt
        return self.value.copy()


def test_replay_reports_reference_demand_and_meaningful_error_thresholds():
    result = replay_points(
        np.array([[0.0], [1.0], [1.0]]),
        policy_fps=10,
        servo_fps=100,
        velocity=np.array([1.0]),
        acceleration=np.array([2.0]),
        limiter_class=_TrackingLimiter,
    )
    assert result["reference_demand_exceedance_ratio"]["velocity_any_joint"] > 0
    assert result["reference_demand_exceedance_ratio"]["acceleration_any_joint"] > 0
    assert result["tracking_error_exceedance_ratio"]["0.05_rad"] > 0
