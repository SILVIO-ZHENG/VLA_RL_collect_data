"""Fixed sampling semantics, including intentional missed-point risk; no GDK."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('stride_helpers', Path(__file__).with_name('test_segmented_rolling_queue.py'))
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
from nexus_deploy.robot.g2a.async_safety import check_sampled_segment_window, stride_sample_indices


@pytest.mark.parametrize('length,expected', [(0, []), (1, [0]), (5, [0, 2, 4]), (6, [0, 2, 4, 5]), (10, [0, 2, 4, 6, 8, 9])])
def test_fixed_indices_and_boundary_retention(length, expected):
    assert stride_sample_indices(length) == expected


@pytest.mark.parametrize('bad_index,blocked', [(0, True), (1, False), (2, True), (3, False), (4, True), (5, True)])
def test_only_selected_actions_are_checked_even_near_table(bad_index, blocked):
    layer = h._helpers.h.make_layer()
    q = np.array([.1, 0., 0.])
    targets = [q.copy() for _ in range(6)]
    targets[bad_index][0] = 0.  # Definitely below the hard surface margin.
    task = dict(anchor=q, fallback_anchor=q, targets=targets, raw=q, committed=[])
    visited = []
    original = layer._evaluate
    def evaluate(pose):
        visited.append(pose.copy())
        return original(pose)
    layer._evaluate = evaluate
    old_samples = layer.path_samples
    result = check_sampled_segment_window(layer, q, task)
    assert result['decision'].hard_limit_triggered is blocked
    assert any(p[0] == 0 for p in visited) is blocked
    assert result['target_holds'][1] is None
    assert result['target_holds'][3] is None
    assert layer.path_samples == old_samples
    # Baseline must detect the deliberately skipped collision. This test
    # documents reduced coverage, NOT equivalence to full-path checking.
    assert check_sampled_segment_window(layer, q, task, stride=1)['decision'].hard_limit_triggered


def test_no_hidden_midpoint_interpolation():
    safety = h._helpers.h.safety
    layer = h._helpers.h.make_layer(points=(safety.CollisionPoint('finger', 'tool', (0., .1, 0.)),))
    q = np.array([.05, 0., 0.])
    target = np.array([.05, -np.pi, 0.])
    task = dict(anchor=q, fallback_anchor=q, targets=[q, target, target], raw=target, committed=[])
    assert not check_sampled_segment_window(layer, q, task)['decision'].hard_limit_triggered
    assert check_sampled_segment_window(layer, q, task, stride=1)['decision'].hard_limit_triggered


def test_skipped_nonfinite_command_is_still_rejected():
    layer = h._helpers.h.make_layer()
    q = np.array([.1, 0., 0.])
    bad = q.copy(); bad[0] = np.nan
    task = dict(anchor=q, fallback_anchor=q, targets=[q, bad, q], raw=q, committed=[])
    with pytest.raises(ValueError, match='including skipped'):
        check_sampled_segment_window(layer, q, task)


def test_inter_arm_geometry_uses_same_unique_stride2_waypoints():
    layer = h._helpers.h.make_layer()

    class CountingInterArm:
        hard_margin = 0.008
        warning_margin = 0.020

        def __init__(self):
            self.counts = {}

        def evaluate(self, configuration, _fk):
            key = np.asarray(configuration, dtype=np.float64).tobytes()
            self.counts[key] = self.counts.get(key, 0) + 1
            return SimpleNamespace(
                clearance=0.1,
                separation=0.108,
                left_name="left_test",
                right_name="right_test",
            )

    counter = CountingInterArm()
    layer.inter_arm_collision = counter
    q = np.array([0.1, 0.0, 0.0])
    targets = []
    for index in range(6):
        target = q.copy()
        target[1] = index * 0.01
        targets.append(target)
    task = dict(anchor=q, fallback_anchor=q, targets=targets, raw=targets[-1], committed=[])

    result = check_sampled_segment_window(layer, q, task)

    assert result["inter_arm"]["checked_waypoints"] == len(result["sampled_target_indices"])
    assert len(counter.counts) == len(result["sampled_target_indices"])
    assert max(counter.counts.values()) == 1


def test_inter_arm_collision_obeys_explicit_stride2_sampling():
    """The configured 50 Hz geometry cadence intentionally skips odd targets."""
    layer = h._helpers.h.make_layer()

    class OddWaypointCollision:
        hard_margin = 0.008
        warning_margin = 0.020

        def evaluate(self, configuration, _fk):
            colliding = np.isclose(float(configuration[1]), 0.03)
            separation = 0.004 if colliding else 0.050
            return SimpleNamespace(
                clearance=separation - self.hard_margin,
                separation=separation,
                left_name="left_gripper_inner",
                right_name="right_gripper_inner",
            )

    layer.inter_arm_collision = OddWaypointCollision()
    q = np.array([0.1, 0.0, 0.0])
    targets = []
    for index in range(6):
        target = q.copy()
        target[1] = index * 0.01
        targets.append(target)
    task = dict(anchor=q, fallback_anchor=q, targets=targets, raw=targets[-1], committed=[])

    result = check_sampled_segment_window(layer, q, task)

    assert 3 not in result["sampled_target_indices"]
    assert result["inter_arm"]["checked_waypoints"] == len(result["sampled_target_indices"])
    assert result["inter_arm"]["collision"] is False
    assert result["decision"].hard_limit_triggered is False
    assert check_sampled_segment_window(layer, q, task, stride=1)["decision"].hard_limit_triggered


def test_unchecked_target_is_not_relabelled_as_checked_hold(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    try:
        h._prime(adapter, clock, worker, feedback, target)
        h._tick(adapter, clock, worker, feedback, target)
        assert adapter._async_hold['step'] == 1
        assert adapter._async_hold['hold_proof'] is None
        held = adapter._select_async_hold(clock[0])
        assert held['historical_repeat'] is True
        assert held['hold_proof'] is None  # No fictitious new geometry proof.
        # Normal motion is not interrupted; all 100 Hz targets are consumed.
        h._tick(adapter, clock, worker, feedback, target)
        assert adapter._async_hold['step'] == 2
        assert adapter._async_hold['hold_proof'] is not None
        assert adapter.last_async_response == 'async_checked_motion'
    finally:
        adapter.disconnect()
