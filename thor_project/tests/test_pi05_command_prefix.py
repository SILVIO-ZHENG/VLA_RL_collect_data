"""No weights/GPU: validate trained prefix normalization and provenance."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nexus_deploy.vla.pi05.policy import G2APi05Policy, _SessionState
from nexus_deploy.vla.pi05.protocol import ProtocolError


def fixture():
    policy = object.__new__(G2APi05Policy)
    policy.device = "cpu"
    policy.rtc_inference_mode = "trained"
    policy.rtc_training_max_delay = 21
    policy.native_action_dim = 16
    policy.effector_profile = "dahuan"
    policy._relative_step = None
    policy._normalizer_step = None
    policy.rtc_config = SimpleNamespace(execution_horizon=29)
    session = _SessionState(
        chunk_id="previous", raw_actions=torch.zeros((50, 16)), processed_native_actions=torch.zeros((50, 16))
    )
    packet = dict(
        source="robot_command_preview",
        not_motion_authority=True,
        committed_steps=3,
        actions=np.full((15, 16), 0.01, dtype=np.float32),
        time_origin=123.0,
        active_chunk_id="previous",
        next_action_index=10,
    )
    control = dict(
        enable_rtc=True,
        active_chunk_id="previous",
        next_action_index=10,
        estimated_delay_steps=15,
        min_rtc_blend_steps=8,
        command_prefix=packet,
    )
    return policy, session, control


def test_actual_command_prefix_is_not_translated_to_feedback_or_old_model():
    p, s, c = fixture()
    prefix, delay, used, source, horizon, blend, reason = p._rtc_prefix(s, c, torch.ones(16))
    assert used and delay == 15 and horizon == 29 and blend == 14 and reason is None
    np.testing.assert_array_equal(prefix.numpy(), c["command_prefix"]["actions"])
    assert s.rtc_reanchor_mode == "robot_command_preview"
    assert s.rtc_prefix_committed_steps == 3


@pytest.mark.parametrize(
    "update",
    [
        {"active_chunk_id": "stale"},
        {"time_origin": float("nan")},
        {"not_motion_authority": False},
        {"actions": np.zeros((14, 16))},
    ],
)
def test_invalid_external_prefix_fails_closed(update):
    p, s, c = fixture()
    c["command_prefix"].update(update)
    with pytest.raises(ProtocolError):
        p._rtc_prefix(s, c, torch.ones(16))


def test_default_model_tail_path_is_preserved():
    p, s, c = fixture()
    c.pop("command_prefix")
    result = p._rtc_prefix(s, c, torch.zeros(16))
    assert result[2] and result[4] == 29 and s.rtc_prefix_source == "model_tail"
