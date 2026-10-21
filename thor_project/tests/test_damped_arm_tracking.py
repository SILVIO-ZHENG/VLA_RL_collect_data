"""Opt-in tracking regressions with numeric arrays and dry adapters only."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexus_deploy.robot.g2a.action_filter import JointKinematicLimiter
from nexus_deploy.robot.g2a.gdk_adapter import AdapterConfig, G2AAdapter


def limiter(indices=(), omega=12.0, sample_hz=100):
    return JointKinematicLimiter(
        sample_hz=sample_hz,
        max_velocity=np.full(4, 0.8),
        max_acceleration=np.full(4, 2.0),
        position_target_indices=(1, 3),
        tracking_damping_indices=indices,
        tracking_damping_omega=omega,
    )


def test_legacy_remains_default_and_bitwise_equal_to_old_formula():
    controller = limiter()
    controller.reset(np.zeros(4))
    q, v = np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
    rng = np.random.default_rng(123)
    for _ in range(1000):
        target, reference_velocity = rng.normal(size=(2, 4)).astype(np.float32)
        desired = np.clip(reference_velocity + 8.0 * (target - q), -0.8, 0.8)
        v = np.asarray(v + np.clip(desired - v, -0.02, 0.02), dtype=np.float32)
        q = np.asarray(q + v * 0.01, dtype=np.float32)
        q[[1, 3]], v[[1, 3]] = target[[1, 3]], 0.0
        np.testing.assert_array_equal(controller.update_tracking(target, reference_velocity), q)


def test_damped_formula_limits_and_device_registers():
    controller = limiter((0, 2))
    controller.reset(np.zeros(4))
    rng = np.random.default_rng(456)
    for _ in range(2000):
        target, reference_velocity = rng.normal(size=(2, 4)).astype(np.float32)
        before = controller._position.copy()
        v_before = controller.velocity
        desired = np.clip(
            v_before + 0.01 * (144 * (target - before) + 24 * (reference_velocity - v_before)), -0.8, 0.8
        )
        expected_v = v_before + np.clip(desired - v_before, -0.02, 0.02)
        after = controller.update_tracking(target, reference_velocity)
        np.testing.assert_allclose(controller.velocity[[0, 2]], expected_v[[0, 2]], atol=1e-7)
        assert np.max(abs(controller.velocity[[0, 2]])) <= 0.8 + 1e-6
        assert np.max(abs((controller.velocity - v_before)[[0, 2]])) <= 0.02 + 1e-6
        np.testing.assert_array_equal(after[[1, 3]], target[[1, 3]])


def test_only_selected_arm_channels_change_and_gripper_event_indices_stay_identical():
    old, new = limiter(), limiter((0,))
    old.reset(np.zeros(4))
    new.reset(np.zeros(4))
    for i in range(400):
        target = np.array([0.3, 0.025 if i < 100 else 0, -0.2, 0.025 if i >= 150 else 0], dtype=np.float32)
        a = old.update_tracking(target, np.zeros(4))
        b = new.update_tracking(target, np.zeros(4))
        np.testing.assert_array_equal(a[[1, 2, 3]], b[[1, 2, 3]])


def test_preview_fork_isolated_and_reset_restarts_from_zero_velocity():
    controller = limiter((0, 2))
    controller.reset(np.zeros(4))
    target = np.ones(4, dtype=np.float32)
    controller.update_tracking(target, target)
    fork = controller.fork()
    old_q, old_v = controller._position.copy(), controller.velocity
    fork.update_tracking(target, target)
    np.testing.assert_array_equal(controller._position, old_q)
    np.testing.assert_array_equal(controller.velocity, old_v)
    np.testing.assert_array_equal(controller.update_tracking(target, target), fork._position)
    fork.reset(old_q)
    np.testing.assert_array_equal(fork.velocity, np.zeros(4))
    stationary = limiter((0, 2))
    stationary.reset(old_q)
    np.testing.assert_array_equal(fork.update_tracking(target, target), stationary.update_tracking(target, target))


@pytest.mark.parametrize(
    "indices,omega,hz",
    [
        ((0, 0), 12, 100),
        ((1,), 12, 100),
        ((-1,), 12, 100),
        ((4,), 12, 100),
        ((0,), 0, 100),
        ((0,), float("nan"), 100),
        ((0,), 26, 100),
        ((0,), 12, 40),
    ],
)
def test_invalid_tracking_parameters_rejected(indices, omega, hz):
    with pytest.raises(ValueError):
        limiter(indices, omega, hz)


def test_dry_adapter_enables_only_arm_indices_and_default_is_unchanged():
    legacy = G2AAdapter(AdapterConfig(hardware_mode="dry-run", servo_control_period_s=0.01))
    damped = G2AAdapter(
        AdapterConfig(hardware_mode="dry-run", servo_control_period_s=0.01, arm_tracking_mode="damped")
    )
    assert legacy._kinematic_limiter.tracking_damping_indices == ()
    assert damped._kinematic_limiter.tracking_damping_indices == tuple(range(7)) + tuple(range(8, 15))
    np.testing.assert_array_equal(legacy._kinematic_limiter.max_velocity, damped._kinematic_limiter.max_velocity)
    np.testing.assert_array_equal(
        legacy._kinematic_limiter.max_acceleration, damped._kinematic_limiter.max_acceleration
    )


def test_invalid_adapter_mode_does_not_start_a_worker():
    with pytest.raises(ValueError, match="arm_tracking_mode"):
        G2AAdapter(AdapterConfig(hardware_mode="dry-run", arm_tracking_mode="unknown"))


def test_cli_defaults_to_legacy_and_requires_explicit_experiment_selection(monkeypatch):
    from nexus_deploy.robot.g2a.client import parse_args

    arguments = ["client", "--host", "offline.invalid", "--task", "Test only"]
    monkeypatch.setattr(sys, "argv", arguments)
    args = parse_args()
    assert args.arm_tracking_mode == "legacy"
    assert args.arm_tracking_omega == 12.0
    monkeypatch.setattr(sys, "argv", arguments + ["--arm-tracking-mode", "damped", "--arm-tracking-omega", "12"])
    assert parse_args().arm_tracking_mode == "damped"


@pytest.mark.parametrize("hardware", ["dry-run", "execute"])
@pytest.mark.parametrize("collision_part", ["decision", "live_decision"])
@pytest.mark.parametrize("omega", [12.0, 16.0])
def test_damped_checked_queue_still_holds_and_recovers_with_mock_sdk(
    tmp_path, monkeypatch, hardware, collision_part, omega
):
    spec = importlib.util.spec_from_file_location(
        "_damped_collision_checks", Path(__file__).with_name("test_collision_filter_continue.py")
    )
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    original_setup = checks.h._setup

    def damped_setup(*args):
        adapter, clock, worker, feedback, target = original_setup(*args)
        adapter._kinematic_limiter.tracking_damping_indices = tuple(range(7)) + tuple(range(8, 15))
        adapter._kinematic_limiter.tracking_damping_omega = omega
        original_send = adapter.send_action

        def send_with_preview(action, **kwargs):
            context = dict(kwargs.pop("safety_context", {}) or {})
            context["trajectory_preview"] = np.tile(action, (20, 1))
            return original_send(action, safety_context=context, **kwargs)

        monkeypatch.setattr(adapter, "send_action", send_with_preview)
        return adapter, clock, worker, feedback, target

    monkeypatch.setattr(checks.h, "_setup", damped_setup)
    # This helper replaces every SDK boundary before changing hardware_mode.
    # Real geometry decisions, queues and three-second holds are still tested.
    checks.test_continuous_collision_hold_then_smooth_resume(tmp_path, monkeypatch, hardware, collision_part)
