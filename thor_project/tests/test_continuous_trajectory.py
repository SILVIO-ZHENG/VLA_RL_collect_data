"""Continuous references and checked output identity; no robot access."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "continuous_helpers", Path(__file__).with_name("test_segmented_rolling_queue.py"))
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)

from nexus_deploy.robot.g2a.action_filter import CausalLowPassFilter, JointKinematicLimiter
from nexus_deploy.robot.g2a.action_plan import ActionPlan
from nexus_deploy.robot.g2a.client import RobotClientRunner


def test_monotone_cubic_keeps_policy_knots_and_device_targets() -> None:
    points = np.array([[.1, .025], [.2, 0.0], [.25, 0.0], [.15, .025]], dtype=np.float32)
    plans = [
        ActionPlan(2, policy_fps=30, position_target_indices=(1,), interpolation_mode=mode)
        for mode in ("linear", "monotone-cubic")
    ]
    for plan in plans:
        plan.set_hold_action(np.array([0.0, .025], dtype=np.float32))
        assert plan.install("chunk", points, plan.snapshot(), now=0).installed

    def sample(plan, when):
        reservation = plan.reserve(now=when)
        value = reservation.action.copy()
        plan.cancel(reservation)
        return value

    for k, point in enumerate(points):
        np.testing.assert_allclose(sample(plans[1], (k + 1) / 30)[0], point[0], atol=1e-6)
    for when in np.arange(0, 4 / 30, .01):
        assert sample(plans[0], when)[1] == sample(plans[1], when)[1]
    around_knot = 2 / 30
    epsilon = 1e-4
    left_velocity = (sample(plans[1], around_knot)[0]
                     - sample(plans[1], around_knot - epsilon)[0]) / epsilon
    right_velocity = (sample(plans[1], around_knot + epsilon)[0]
                      - sample(plans[1], around_knot)[0]) / epsilon
    assert abs(left_velocity - right_velocity) < .05
    samples = [sample(plans[1], when)[0] for when in np.arange(1 / 30, 2 / 30, .001)]
    assert min(samples) >= .1 - 1e-6
    assert max(samples) <= .2 + 1e-6


def test_monotone_cubic_does_not_reshape_software_gripper_channel() -> None:
    points = np.array([[.1, .02], [.2, .00], [.25, .02]], dtype=np.float32)
    plans = [ActionPlan(2, policy_fps=30, interpolation_mode=mode,
                        cubic_indices=(0,)) for mode in ("linear", "monotone-cubic")]
    for plan in plans:
        plan.set_hold_action(np.array([0.0, .025], dtype=np.float32))
        assert plan.install("chunk", points, plan.snapshot(), now=0).installed
    for when in np.arange(0, .10, .01):
        values = []
        for plan in plans:
            reservation = plan.reserve(now=when)
            values.append(reservation.action.copy())
            plan.cancel(reservation)
        assert values[0][1] == values[1][1]


def test_preview_does_not_consume_plan_and_stops_at_chunk_end():
    plan = ActionPlan(1, policy_fps=10)
    plan.set_hold_action(np.zeros(1))
    plan.install("a", np.array([[.1], [.2]]), plan.snapshot(), now=0)
    reservation = plan.reserve(now=.05)
    before = plan.snapshot()
    preview = plan.preview_reserved(reservation, now=.05, period=.01, count=30)
    assert plan.snapshot() == before
    np.testing.assert_allclose(preview[:3, 0], [.05, .06, .07])
    np.testing.assert_allclose(preview[15:, 0], .2)
    plan.clear()
    with pytest.raises(RuntimeError):
        plan.preview_reserved(reservation, now=.05, period=.01)
    plan.cancel(reservation)


def test_preview_filter_fork_does_not_change_live_state():
    f = CausalLowPassFilter(cutoff_hz=2, sample_hz=100, action_dim=1)
    f.reset(np.zeros(1))
    fork = f.fork()
    first = fork.update(np.ones(1))
    for _ in range(14):
        fork.update(np.ones(1))
    np.testing.assert_array_equal(f._value, [0])
    np.testing.assert_array_equal(f.update(np.ones(1)), first)


def test_client_preview_matches_live_filter_and_does_not_advance_it():
    runner = object.__new__(RobotClientRunner)
    runner._held_head_waist_values = None
    runner._held_effector_values = None
    runner._action_filter = CausalLowPassFilter(cutoff_hz=2, sample_hz=100, action_dim=24)
    runner._effector_action_filter = None
    runner._action_filter.reset(np.zeros(24))
    runner._control_lock = threading.Lock()
    captured = {}
    def send(target, safety_context):
        captured.update(safety_context)
        return target.copy()
    runner.adapter = SimpleNamespace(_safety_layer=True, send_action=send, last_action_intervened=False)
    raw = np.stack([np.ones(24) * (i + 1) * .001 for i in range(15)]).astype(np.float32)
    sent = runner._send_action(raw[0], trajectory_preview=raw)
    np.testing.assert_array_equal(captured["trajectory_preview"][0], sent)
    np.testing.assert_array_equal(runner._action_filter._value[:7], sent[:7])
    # No hidden arm filtering of the independent gripper channel.
    np.testing.assert_array_equal(captured["trajectory_preview"][:, 7], raw[:, 7])


def test_moving_reference_removes_five_point_velocity_ripple():
    old = JointKinematicLimiter(sample_hz=100, max_velocity=np.array([.6]), max_acceleration=np.array([2.]))
    old.reset(np.zeros(1))
    new = old.fork()
    old_v, new_v = [], []
    for k in range(200):
        frozen = np.array([k * 5 * .01 * .3])
        for i in range(5):
            target = np.array([(k * 5 + i) * .01 * .3])
            old.update(frozen)
            new.update_tracking(target, np.array([.3]))
            old_v.append(old.velocity[0])
            new_v.append(new.velocity[0])
    assert np.std(new_v[200:]) < np.std(old_v[200:]) * .05
    assert np.max(np.abs(np.diff(new_v))) <= .020001
    assert max(np.abs(new_v)) <= .600001
    # Stationary endpoint and reversal retain the acceleration bound.
    for target in [np.array([2.]), np.array([-2.])]:
        for _ in range(300):
            previous = new.velocity.copy()
            new.update_tracking(target, np.zeros(1))
            assert np.max(abs(new.velocity - previous)) <= .020001


def test_preview_outputs_are_exactly_the_checked_points(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    try:
        # Eight queued points plus seven checked refill points need the same
        # twenty-point lookahead supplied by the production action thread.
        preview = np.stack([feedback.copy() for _ in range(20)])
        preview[:, 0] += np.arange(20) * .001
        adapter.send_action(preview[0], safety_context={"trajectory_preview": preview})
        task = worker.tasks[0]
        assert "trajectory_preview" not in task["context"]
        expected = task["targets"][0].copy()
        worker.finish(started=clock[0], measured=feedback)
        adopted = h._tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(adopted, expected)
        prefix = [x["action"].copy() for x in adapter._async_steps]
        clock[0] += .01
        worker.feedback = feedback.copy(), clock[0]
        adapter.send_action(preview[0], safety_context={"trajectory_preview": preview})
        pending = adapter._async_pending
        limiter = adapter._async_steps[-1]["limiter"].fork()
        expected = limiter.update_tracking(preview[9], (preview[10] - preview[9]) / .01)
        np.testing.assert_array_equal(pending["targets"][0], expected)
        for actual, expected in zip(adapter._async_steps, prefix[1:]):
            np.testing.assert_array_equal(actual["action"], expected)
    finally:
        adapter.disconnect()


@pytest.mark.parametrize("boundary,overshoot", [(0.0, -4.4e-11), (.025, .02500001), (0.0, -.002)])
def test_clip_preserves_other_joint_velocities(boundary, overshoot):
    limiter = JointKinematicLimiter(sample_hz=100, max_velocity=np.full(24, .6),
                                    max_acceleration=np.full(24, 2.))
    limiter.reset(np.zeros(24))
    limiter._position[7] = overshoot
    limiter._velocity[:] = np.linspace(.01, .24, 24)
    before = limiter.velocity
    clipped = limiter._position.copy()
    clipped[7] = boundary
    limiter.reconcile_position_clip(clipped)
    assert limiter._position[7] == np.float32(boundary)
    assert limiter.velocity[7] == 0
    indices = [i for i in range(24) if i != 7]
    np.testing.assert_array_equal(limiter.velocity[indices], before[indices])
    frozen = limiter.fork()
    limiter.reconcile_position_clip(clipped)
    np.testing.assert_array_equal(limiter.velocity, frozen.velocity)


def test_gripper_cycles_do_not_retime_arm_motion():
    mixed = JointKinematicLimiter(sample_hz=100, max_velocity=np.array([.6, .05]),
                                  max_acceleration=np.array([2., .2]))
    mixed.reset(np.zeros(2))
    baseline = mixed.fork()
    clips = 0
    for k in range(1000):
        # Open, close, then hold a numerically negative closed reference.
        grip = .025 if (k // 100) % 2 == 0 else -4.4e-11
        out = mixed.update_tracking(np.array([k * .003, grip]), np.array([.3, 0.]))
        expected = baseline.update_tracking(np.array([k * .003, 0.]), np.array([.3, 0.]))
        checked = out.copy()
        checked[1] = np.clip(checked[1], 0., .025)
        clips += int(checked[1] != out[1])
        mixed.reconcile_position_clip(checked)
        assert 0 <= mixed._position[1] <= np.float32(.025)
        np.testing.assert_array_equal(mixed._position[0], expected[0])
        np.testing.assert_array_equal(mixed.velocity[0], baseline.velocity[0])
    assert clips > 0  # This regression must actually exercise the fault.


def test_async_gripper_clip_keeps_arm_velocity_in_checked_segment(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    try:
        adapter._last_state[7] = 0.
        adapter._kinematic_limiter.reset(adapter._last_state)
        adapter._kinematic_limiter._velocity[0] = .2
        preview = np.stack([adapter._last_state.copy() for _ in range(15)])
        preview[:, 0] += np.arange(15) * .002
        preview[:, 7] = -4.4e-11
        adapter.send_action(preview[0], safety_context={"trajectory_preview": preview})
        task = worker.tasks[0]
        assert all(q[7] == 0 for q in task["targets"])
        assert all(l.velocity[0] > 0 for l in task["limiters"])
        assert task["limiters"][-1].velocity[0] > task["limiters"][0].velocity[0]
        assert all(l.velocity[7] == 0 for l in task["limiters"])
        worker.finish(started=clock[0], measured=feedback)
        actual = h._tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(actual, task["targets"][0])
    finally:
        adapter.disconnect()
