import numpy as np

from nexus_deploy.robot.g2a.action_filter import FixedHorizonChunkProjector
from nexus_deploy.robot.g2a.action_plan import ActionPlan
from nexus_deploy.robot.g2a.gdk_adapter import (
    DATASET_V30_P99_ARM_ACCELERATION,
    DATASET_V30_P99_ARM_VELOCITY,
)


def projector():
    return FixedHorizonChunkProjector(
        policy_hz=30,
        action_dim=4,
        projected_indices=(0, 2),
        max_velocity=np.array([0.6, 99, 0.4, 99]),
        max_acceleration=np.array([2.0, 99, 1.0, 99]),
    )


def test_experimental_dataset_profile_has_expected_shape_and_stays_below_vendor_velocity():
    assert DATASET_V30_P99_ARM_VELOCITY.shape == (14,)
    assert DATASET_V30_P99_ARM_ACCELERATION.shape == (14,)
    assert np.all(DATASET_V30_P99_ARM_VELOCITY > 0)
    assert np.all(DATASET_V30_P99_ARM_ACCELERATION > 0)
    # The lowest arm velocity declared by the installed G2 URDF is 3.6 rad/s.
    assert np.max(DATASET_V30_P99_ARM_VELOCITY) < 3.6


def test_fixed_horizon_projection_keeps_clock_events_and_discrete_bounds():
    raw = np.zeros((20, 4), dtype=np.float32)
    raw[:, 0] = np.linspace(0.2, 1.0, len(raw))
    raw[:, 1] = np.where(np.arange(len(raw)) < 7, 0.025, 0.0)  # gripper event
    raw[:, 2] = np.r_[np.full(10, 0.5), np.full(10, -0.5)]
    raw[:, 3] = np.arange(len(raw))  # another non-projected register

    projected, stats = projector().project(
        raw,
        start_position=np.zeros(4),
        start_velocity=np.zeros(4),
    )

    np.testing.assert_array_equal(projected[:, [1, 3]], raw[:, [1, 3]])
    q = np.vstack((np.zeros(4), projected))
    velocity = np.diff(q[:, [0, 2]], axis=0) * 30
    acceleration = np.diff(np.vstack((np.zeros((1, 2)), velocity)), axis=0) * 30
    assert np.all(np.max(np.abs(velocity), axis=0) <= np.array([0.6, 0.4]) + 1e-5)
    assert np.all(np.max(np.abs(acceleration), axis=0) <= np.array([2.0, 1.0]) + 1e-4)
    assert stats["changed_values"] > 0
    assert stats["max_deviation_rad"] > 0


def test_action_plan_projects_only_installable_suffix_and_reports_deviation():
    plan = ActionPlan(4, 30, position_target_indices=(1,), chunk_projector=projector())
    start = np.array([0.0, 0.025, 0.0, 7.0], dtype=np.float32)
    plan.set_hold_action(start)
    raw = np.tile(start, (8, 1))
    raw[:, 0] = 1.0
    raw[:, 1] = np.linspace(0.025, 0.0, 8)

    result = plan.install("chunk", raw, plan.snapshot(), stale_steps=3, now=10.0)
    assert result.installed
    assert result.fixed_horizon_projection is not None
    assert result.fixed_horizon_projection["steps"] == 5
    sample = plan.reserve(now=10.0 + 1 / 30)
    assert sample is not None
    assert sample.action[0] == pytest.approx(2 / 900, abs=1e-6)
    assert sample.action[1] == raw[4, 1]  # unchanged register event on the existing plan clock


def test_projection_is_disabled_by_default():
    plan = ActionPlan(4, 30)
    start = np.zeros(4, dtype=np.float32)
    plan.set_hold_action(start)
    raw = np.ones((4, 4), dtype=np.float32)
    result = plan.install("raw", raw, plan.snapshot(), now=0.0)
    assert result.fixed_horizon_projection is None
    sample = plan.reserve(now=1 / 30)
    assert sample is not None
    np.testing.assert_array_equal(sample.action, raw[0])


import pytest
