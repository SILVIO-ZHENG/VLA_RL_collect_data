"""Sampled egress does not confuse safe endpoints with safe sampled progress."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('escape_helpers', Path(__file__).with_name('test_safety_layer.py'))
h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)
from nexus_deploy.robot.g2a.async_safety import check_sampled_segment_window


@pytest.mark.parametrize('heights,allowed', [
    ([.011, .012, .013], True),  # Need not leave the margin in a single short segment.
    ([.025, .026, .027], True),
    ([.011, .012, .009], False),
    ([.009, .015, .030], False),  # Safe endpoint cannot excuse a worsening earlier sample.
    ([.010, .010, .010], False),
])
def test_unsafe_start_can_only_make_sampled_progress(heights, allowed):
    layer = h.make_layer(); q = np.array([.010, 0., 0.])
    targets = [np.array([z, 0., 0.]) for z in heights]
    task = dict(anchor=q, fallback_anchor=q, targets=targets, raw=targets[-1], committed=[], allow_sampled_escape=True)
    result = check_sampled_segment_window(layer, q, task)
    assert result['escape_approved'] is allowed
    assert result['decision'].hard_limit_triggered is not allowed
    assert result['sampled_target_indices'] == [0, 2]


def test_initial_unsafe_pose_has_no_automatic_escape_authority():
    layer = h.make_layer(); q = np.array([.010, 0., 0.]); target = q.copy(); target[0] = .03
    task = dict(anchor=q, fallback_anchor=q, targets=[target], raw=target, committed=[])
    result = check_sampled_segment_window(layer, q, task)
    assert not result['escape_approved']
    assert result['decision'].hard_limit_triggered


def test_improving_worst_point_cannot_create_a_second_collision():
    class FK:
        def frame_pose(self, q, frame):
            pose = np.eye(4); pose[2,3] = q[0 if frame == 'left' else 1]; return pose
    layer = h.safety.ActionSafetyLayer(
        fk=FK(), geometry=h.safety.GripperGeometry(tuple(
            h.safety.CollisionPoint(name, name, (0.,0.,0.)) for name in ('left','right'))),
        surface=h.safety.TableConstraint(height=0., hard_margin=.02),
        motion_indices=(0,1), passthrough_indices=(2,))
    q = np.array([.01,.03,0.]); target = np.array([.015,.015,0.])
    result = check_sampled_segment_window(layer, q, dict(anchor=q, fallback_anchor=q,
        committed=[], targets=[target], raw=target, allow_sampled_escape=True))
    assert not result['escape_approved']


def test_safe_current_pose_cannot_use_escape_to_enter_danger():
    layer = h.make_layer(); q = np.array([.03, 0., 0.]); target = q.copy(); target[0] = .01
    task = dict(anchor=q, fallback_anchor=q, targets=[target], raw=target, committed=[], allow_sampled_escape=True)
    result = check_sampled_segment_window(layer, q, task)
    assert not result['escape_approved']
    assert result['decision'].hard_limit_triggered


@pytest.mark.parametrize('arm_lift,allowed', [(0., False), (.001, True)])
def test_escape_cannot_depend_only_on_untracked_gripper_closure(arm_lift, allowed):
    class FK:
        def frame_pose(self, q, frame):
            pose = np.eye(4); pose[2, 3] = q[0] - q[15]
            return pose
    layer = h.safety.ActionSafetyLayer(
        fk=FK(), geometry=h.safety.GripperGeometry((h.safety.CollisionPoint('finger', 'tool', (0., 0., 0.)),)),
        surface=h.safety.TableConstraint(height=0., hard_margin=.002),
        motion_indices=tuple(i for i in range(24) if i not in (7, 15)), passthrough_indices=(7, 15))
    measured = np.zeros(24); measured[0] = .02; measured[15] = .0195
    anchor = measured.copy(); anchor[15] = .001
    target = anchor.copy(); target[0] += arm_lift
    result = check_sampled_segment_window(layer, measured, dict(
        anchor=anchor, fallback_anchor=anchor, targets=[target], raw=target, committed=[], allow_sampled_escape=True))
    assert result['escape_approved'] is allowed


def test_escape_result_is_actually_admitted_by_servo_queue(tmp_path, monkeypatch):
    helper_spec = importlib.util.spec_from_file_location('escape_queue_helpers', Path(__file__).with_name('test_segmented_rolling_queue.py'))
    helpers = importlib.util.module_from_spec(helper_spec); sys.modules[helper_spec.name] = helpers; helper_spec.loader.exec_module(helpers)
    adapter, clock, worker, feedback, target = helpers._setup(tmp_path, monkeypatch)
    try:
        helpers._prime(adapter, clock, worker, feedback, target)
        # Empty the old queue as a collision does, then recover from rest.
        adapter._invalidate_async_safety(clear_hold=False)
        adapter._kinematic_limiter.reset(adapter._last_state)
        class FK:
            def frame_pose(self, q, frame):
                pose = np.eye(4); pose[2, 3] = q[0]; return pose
        layer = h.safety.ActionSafetyLayer(
            fk=FK(), geometry=h.safety.GripperGeometry((h.safety.CollisionPoint('finger', 'tool', (0., 0., 0.)),)),
            surface=h.safety.TableConstraint(height=0., hard_margin=.002),
            motion_indices=tuple(i for i in range(24) if i not in (7,15)), passthrough_indices=(7,15))
        adapter._safety_controller.guard_layer = layer
        feedback = adapter._last_state.copy()
        helpers._tick(adapter, clock, worker, feedback, target)
        worker.finish(started=clock[0], measured=feedback)
        assert worker.result['escape_approved']
        expected = worker.result['task']['targets'][0].copy()
        output = helpers._tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(output, expected)
        assert adapter.last_async_response == 'async_checked_motion'
    finally:
        adapter.disconnect()
