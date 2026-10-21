"""Reset grippers only when needed, while retaining feedback confirmation."""

from types import SimpleNamespace

import numpy as np

from nexus_deploy.robot.g2a import reset_to_dataset_start as reset


def _args():
    return SimpleNamespace(
        effector_tolerance=.0015,
        effector_timeout_s=8.0,
        effector_reissue_s=.75,
        effector_max_commands=3,
        status_period_s=.5,
    )


def _state(position):
    value = np.zeros(24, dtype=np.float32)
    value[[7, 15]] = position
    return value, {}


def test_already_open_grippers_are_verified_without_another_command(monkeypatch):
    samples = iter([_state(.025)] * 4)
    monkeypatch.setattr(reset, "read_canonical_state", lambda robot: next(samples))
    monkeypatch.setattr(reset, "send_dahuan_effector_target",
                        lambda *args: (_ for _ in ()).throw(AssertionError("redundant command")))
    monkeypatch.setattr(reset.time, "sleep", lambda seconds: None)

    result = reset.move_and_verify_dahuan_effectors(None, None, .025, _args(), "open_final_after_body")

    np.testing.assert_allclose(result[[7, 15]], [.025, .025])


def test_closed_grippers_still_receive_command_and_need_three_good_samples(monkeypatch):
    samples = iter([_state(0), _state(0), _state(.025), _state(.025), _state(.025)])
    issued = []
    monkeypatch.setattr(reset, "read_canonical_state", lambda robot: next(samples))
    monkeypatch.setattr(reset, "send_dahuan_effector_target",
                        lambda gdk, robot, position: issued.append(position))
    monkeypatch.setattr(reset.time, "sleep", lambda seconds: None)

    result = reset.move_and_verify_dahuan_effectors(None, None, .025, _args(), "open_before_reset")

    assert issued == [.025]
    np.testing.assert_allclose(result[[7, 15]], [.025, .025])
