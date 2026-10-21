"""Command conditioning must not mutate a live queue or authorize motion."""

import numpy as np
import pytest

from nexus_deploy.robot.g2a.action_filter import JointKinematicLimiter
from nexus_deploy.robot.g2a.action_plan import ActionPlan
from nexus_deploy.robot.g2a.command_prefix import build_command_prefix, validate_command_prefix


def limiter():
    return JointKinematicLimiter(
        sample_hz=100, max_velocity=np.ones(24), max_acceleration=np.full(24, 8), position_target_indices=(7, 15)
    )


def test_preview_preserves_committed_path_and_private_limiter_state():
    controller = limiter()
    start = np.zeros(24, dtype=np.float32)
    start[[7, 15]] = (0.025, 0.0)
    controller.reset(start)
    refs = np.repeat(start[None, :], 55, axis=0)
    refs[:, 0] += np.linspace(0, 0.2, len(refs))
    frozen = controller.fork()
    committed = []
    for index in range(1, 8):
        q = frozen.update_tracking(refs[index], (refs[index + 1] - refs[index]) / 0.01)
        committed.append(dict(action=q.copy(), limiter=frozen.fork()))
    before = [(s["action"].copy(), s["limiter"].velocity) for s in committed]
    packet = build_command_prefix(controller, start, committed, refs, policy_hz=30, delay_steps=15)
    assert packet["committed_steps"] == 3
    assert packet["predicted_steps"] == 12
    assert packet["not_motion_authority"] is True
    assert packet["actions"].shape == (15, 16)
    np.testing.assert_array_equal(packet["actions"][:, 7], 0.0)
    np.testing.assert_array_equal(packet["actions"][:, 15], np.float32(0.025))
    np.testing.assert_array_equal(controller._position, start)
    np.testing.assert_array_equal(controller.velocity, 0.0)
    for original, old in zip(committed, before, strict=True):
        np.testing.assert_array_equal(original["action"], old[0])
        np.testing.assert_array_equal(original["limiter"].velocity, old[1])
    hardware = packet["hardware_actions"]
    expected = committed[2]["action"] + (1 / 3) * (committed[3]["action"] - committed[2]["action"])
    np.testing.assert_allclose(hardware[1], expected, atol=1e-7)


def test_missing_preview_invalid_packet_or_authority_claim_is_rejected():
    controller = limiter()
    controller.reset(np.zeros(24))
    with pytest.raises(ValueError):
        build_command_prefix(controller, np.zeros(24), [], np.zeros((15, 24)), policy_hz=30, delay_steps=15)
    packet = dict(
        source="robot_command_preview", actions=np.zeros((15, 16)), committed_steps=3, not_motion_authority=True
    )
    assert validate_command_prefix(packet, delay_steps=15, native_dim=16)[1] == 3
    for update in [
        dict(not_motion_authority=False),
        dict(committed_steps=16),
        dict(actions=np.full((15, 16), np.nan)),
        dict(actions=np.zeros((14, 16))),
    ]:
        with pytest.raises(ValueError):
            validate_command_prefix({**packet, **update}, delay_steps=15, native_dim=16)


def test_clock_preview_is_independent_and_never_extends_trusted_horizon():
    plan = ActionPlan(24, policy_fps=30)
    plan.set_hold_action(np.zeros(24))
    assert plan.install("test", np.ones((50, 24)), plan.snapshot(), now=10.0).installed
    anchor = plan.snapshot()
    preview = plan.preview_at(10.1, period=0.01, count=40)
    assert preview.shape == (40, 24)
    assert plan.snapshot() == anchor and plan._reservation is None
    preview[0] = 100.0
    assert np.max(plan.preview_at(10.1, period=0.01, count=40)) <= 1.0
    assert plan.preview_at(11.5, period=0.01, count=40) is None
