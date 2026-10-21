"""Offline join invariants. These tests never import GDK or open a connection."""

import numpy as np
import pytest

from tools.trajectory_lab.chunk_join import ChunkJoin, HandoffOffset


def make(duration=0.2, n=50, first=0):
    actions = np.zeros((n, 4), dtype=np.float32)
    actions[:, 0] = np.linspace(0.02, 0.2, n)
    actions[:, 2] = -actions[:, 0]
    actions[:, 1] = np.where(np.arange(n) < 8, 0.025, 0.0)
    actions[:, 3] = np.where(np.arange(n) < 13, 0.0, 0.025)
    return ChunkJoin(
        actions,
        np.array([0.0, 0.025, 0.0, 0.0]),
        np.array([0.1, 0.0, -0.1, 0.0]),
        fps=30,
        duration=duration,
        first=first,
        arm_indices=(0, 2),
        device_indices=(1, 3),
    )


def test_cubic_starts_at_anchor_with_initial_velocity():
    sampler = make()
    q, v = sampler.sample(0.0)
    np.testing.assert_allclose(q, sampler.start, atol=1e-7)
    np.testing.assert_allclose(v, sampler.velocity, atol=1e-7)


def test_cubic_rejoins_original_clock_and_velocity():
    sampler = make()
    q, v = sampler.sample(sampler.duration - 1e-7)
    q_end, v_end = sampler.linear(sampler.duration)
    np.testing.assert_allclose(q, q_end, atol=1e-6)
    np.testing.assert_allclose(v, v_end, atol=1e-5)
    for t in (0.2, 0.21, 0.333, 1.0, 2.0):
        for actual, expected in zip(sampler.sample(t), sampler.linear(t)):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("duration", [0.0, 0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("first", [0, 9])
def test_both_gripper_event_times_and_values_are_bitwise_unchanged(duration, first):
    original, candidate = make(0.0, first=first), make(duration, first=first)
    for t in np.linspace(0.0, 2.0, 601):
        np.testing.assert_array_equal(original.sample(t)[0][[1, 3]], candidate.sample(t)[0][[1, 3]])


def test_handoff_offset_starts_at_actual_queue_tail_not_installation():
    original = make(0.0)
    q, v = np.array([0.04, 99, -0.03, 99]), np.array([0.2, 99, -0.1, 99])
    offset = HandoffOffset(original, elapsed=0.12, position=q, velocity=v, duration=0.2)
    actual_q, actual_v = offset.sample(0.12)
    np.testing.assert_allclose(actual_q[[0, 2]], q[[0, 2]], atol=1e-7)
    np.testing.assert_allclose(actual_v[[0, 2]], v[[0, 2]], atol=1e-7)
    for t in (0.0, 0.1, 0.32, 0.4, 2.0):
        for actual, expected in zip(offset.sample(t), original.sample(t)):
            np.testing.assert_array_equal(actual, expected)
    for t in np.linspace(0, 2, 301):
        np.testing.assert_array_equal(offset.sample(t)[0][[1, 3]], original.sample(t)[0][[1, 3]])


def test_handoff_offset_returns_to_reference_without_velocity_jump():
    original = make(0.0)
    offset = HandoffOffset(original, elapsed=0.123, position=np.ones(4), velocity=np.zeros(4), duration=0.2)
    q, v = offset.sample(0.323 - 1e-7)
    expected_q, expected_v = original.sample(0.323)
    np.testing.assert_allclose(q, expected_q, atol=1e-6)
    np.testing.assert_allclose(v, expected_v, atol=2e-5)


def test_short_chunk_never_extrapolates_or_extends_time():
    sampler = make(0.4, n=2)
    assert sampler.duration == pytest.approx(2 / 30)
    q, v = sampler.sample(2 / 30)
    np.testing.assert_array_equal(q, sampler.actions[-1])
    np.testing.assert_array_equal(v, np.zeros(4))


@pytest.mark.parametrize("fps,duration", [(0, 0.1), (30, -0.1), (float("nan"), 0.1), (30, float("inf"))])
def test_invalid_parameters_rejected(fps, duration):
    with pytest.raises(ValueError):
        ChunkJoin(
            np.zeros((10, 4)),
            np.zeros(4),
            np.zeros(4),
            fps=fps,
            duration=duration,
            arm_indices=(0, 2),
            device_indices=(1, 3),
        )


def test_input_not_modified_and_overlap_rejected():
    points = np.ones((10, 4))
    original = points.copy()
    candidate = ChunkJoin(
        points, np.zeros(4), np.zeros(4), fps=30, duration=0.1, arm_indices=(0, 2), device_indices=(1, 3)
    )
    candidate.sample(0.03)
    np.testing.assert_array_equal(points, original)
    with pytest.raises(ValueError):
        ChunkJoin(points, np.zeros(4), np.zeros(4), fps=30, arm_indices=(0, 1), device_indices=(1, 3))
