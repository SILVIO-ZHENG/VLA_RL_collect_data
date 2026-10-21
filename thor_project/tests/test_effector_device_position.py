"""Offline regressions for held Dahuan targets and the full-opening envelope."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nexus_deploy.robot.g2a.action_filter import JointKinematicLimiter
from nexus_deploy.robot.g2a.action_plan import ActionPlan
from nexus_deploy.robot.g2a.gdk_adapter import AdapterConfig, G2AAdapter
from nexus_deploy.robot.g2a.safety_layer import (
    CollisionPoint, GripperGeometry, SafetyConfigurationError, URDFForwardKinematics,
    load_safety_layer, prismatic_opening_envelope,
)

TEMPLATE = ROOT / "src/nexus_deploy/robot/g2a/G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json"


@pytest.mark.parametrize("tracking", [False, True])
def test_device_targets_repeat_exactly_without_changing_body_limit(tracking):
    limits = dict(sample_hz=100, max_velocity=np.full(24, .05), max_acceleration=np.full(24, .2))
    old = JointKinematicLimiter(**limits)
    new = JointKinematicLimiter(**limits, position_target_indices=(7, 15))
    q = np.zeros(24)
    old.reset(q); new.reset(q)
    target = np.full(24, .025)
    for _ in range(30):
        a = old.update_tracking(target, q) if tracking else old.update(target)
        b = new.update_tracking(target, q) if tracking else new.update(target)
        np.testing.assert_array_equal(np.delete(a, [7, 15]), np.delete(b, [7, 15]))
        np.testing.assert_array_equal(b[[7, 15]], np.asarray(target, dtype=np.float32)[[7, 15]])
        np.testing.assert_array_equal(new.velocity[[7, 15]], [0, 0])
    preview = new.fork()
    reverse = target.copy(); reverse[[7, 15]] = 0
    assert preview.update(reverse)[7] == 0
    assert new._position[7] > 0  # preview cannot advance live state
    new.reset(np.zeros(24))  # pause/collision hold reanchors both channels
    np.testing.assert_array_equal(new.velocity, np.zeros(24))


def test_policy_samples_hold_gripper_goal_and_keep_body_interpolation():
    old, new = ActionPlan(24, 30), ActionPlan(24, 30, position_target_indices=(7, 15))
    actions = np.zeros((2, 24), dtype=np.float32)
    actions[0] = .025
    for plan in (old, new):
        plan._actions = actions
        plan._trajectory_start_time = 10.0
        plan._trajectory_start_action = np.zeros(24, dtype=np.float32)
        plan._valid_end_index = 2
    for t in [10, 10.01, 10.02]:
        a, next_a, index_a = old._sample_locked(t)
        b, next_b, index_b = new._sample_locked(t)
        np.testing.assert_array_equal(np.delete(a, [7, 15]), np.delete(b, [7, 15]))
        np.testing.assert_array_equal(b[[7, 15]], actions[0, [7, 15]])
        assert (next_a, index_a) == (next_b, index_b)
    assert new._sample_locked(10 + 1/30)[0][7] == 0


def test_full_opening_envelope_contains_both_independent_finger_ranges():
    raw = load_safety_layer(TEMPLATE, allow_read_only_ready=True)
    envelope = load_safety_layer(TEMPLATE, allow_read_only_ready=True,
                                effector_envelope_ranges={7: (0, .025), 15: (0, .025)})
    rng = np.random.default_rng(14)
    for _ in range(35):
        q = rng.uniform(-.8, .8, 24)
        q[[7, 15]] = rng.uniform(0, .025, 2)
        actual = np.concatenate([g[2] for g in raw.geometry.world_points(q, raw.fk)])
        swept = np.concatenate([g[2] for g in envelope.geometry.world_points(q, envelope.fk)])
        assert np.all(swept.min(axis=0) <= actual.min(axis=0) + 1e-10)
        assert np.all(swept.max(axis=0) >= actual.max(axis=0) - 1e-10)
        before = envelope.minimum_z(q).z
        q[[7, 15]] = rng.uniform(0, .025, 2)
        assert envelope.minimum_z(q).z == pytest.approx(before, abs=1e-12)
    assert len(envelope.geometry._groups) == 2
    assert envelope.realtime_guard().geometry is envelope.geometry


def test_rotating_finger_cannot_silently_use_endpoint_only_envelope(tmp_path):
    path = tmp_path / "rotating.urdf"
    path.write_text('<robot name="test"><link name="base"/><link name="finger"/>'
                    '<joint name="grip" type="revolute"><parent link="base"/>'
                    '<child link="finger"/><axis xyz="0 1 0"/></joint></robot>')
    fk = URDFForwardKinematics(path, root_link="base", required_frames=["finger"],
                              joint_map={"grip": {"action_index": 7}})
    geometry = GripperGeometry([CollisionPoint("tip", "finger", (1, 0, 0), 0)])
    with pytest.raises(SafetyConfigurationError, match="prismatic"):
        prismatic_opening_envelope(geometry, fk, {7: (0, .025)})


def test_adapter_readonly_uses_same_envelope_and_no_hardware_connect(tmp_path):
    adapter = G2AAdapter(AdapterConfig(hardware_mode="read-only", effector_control_mode="device-position",
                         servo_control_period_s=.01, safety_config_path=TEMPLATE, safety_log_path=tmp_path / "safety.jsonl"))
    try:
        assert not adapter.is_connected
        assert adapter._kinematic_limiter.position_target_indices == (7, 15)
        assert len(adapter._safety_controller.guard_layer.geometry._groups) == 6
    finally:
        adapter.disconnect()


def test_worker_compiles_same_envelope(monkeypatch):
    from nexus_deploy.robot.g2a import async_safety
    loaded = []
    original = load_safety_layer
    def load(path, **kwargs):
        loaded.append(kwargs)
        return original(path, allow_read_only_ready=True, **kwargs)
    class Connection:
        messages = []
        def send(self, value): self.messages.append(value)
        def recv(self): return None
        def close(self): pass
    monkeypatch.setattr(async_safety, "load_safety_layer", load)
    conn = Connection()
    ranges = {7: (0, .025), 15: (0, .025)}
    async_safety._geometry_process(conn, TEMPLATE, 2, ranges)
    assert loaded == [{"effector_envelope_ranges": ranges}]
    assert conn.messages == [{"ready": True}]


def test_non_dahuan_cannot_enable_device_position():
    with pytest.raises(ValueError, match="Dahuan"):
        G2AAdapter(AdapterConfig(effector_profile="omnipicker_legacy", effector_control_mode="device-position"))


def test_runner_device_position_has_no_hidden_lowpass_and_pause_reanchors():
    from nexus_deploy.robot.g2a.client import RobotClientRunner
    measured = np.zeros(24, dtype=np.float32)
    measured[[7, 15]] = .01
    calls = []
    adapter = SimpleNamespace(config=SimpleNamespace(effector_control_mode="device-position"),
                              read_state=lambda: measured.copy(),
                              reanchor_measured_hold=lambda q: calls.append(q.copy()),
                              reset_action_limiter=lambda q: None,
                              send_action=lambda q: q.copy())
    runner = RobotClientRunner(adapter, SimpleNamespace(metadata={}), task="test", inference_mode="rtc",
        hardware_mode="read-only", fps=30, servo_fps=100, prediction_horizon=50, quality_horizon=50,
        min_replan_after_steps=1, request_lead_margin_steps=8, min_rtc_blend_steps=8, latency_window_size=20,
        warmup_requests=1, execution_preflight_requests=1, watchdog_timeout_s=3,
        initial_head_waist_warning_threshold=.01, max_arm_tracking_error=.4, max_effector_tracking_error=.01,
        max_head_waist_tracking_error=.2, tracking_error_consecutive_limit=3,
        action_low_pass_cutoff_hz=2, effector_action_low_pass_cutoff_hz=2)
    assert runner._action_filter is not None
    assert runner._effector_action_filter is None
    assert runner.plan.position_target_indices == (7, 15)
    chunk = np.zeros((9, 24), dtype=np.float32)
    chunk[:, 7] = .025
    chunk[:, 15] = .025
    chunk[4, [7, 15]] = 0.0
    shaped = runner._chunk_shaper.shape(chunk)
    np.testing.assert_array_equal(shaped[:, [7, 15]], chunk[:, [7, 15]])
    goal = measured.copy(); goal[[7, 15]] = .025
    assert runner._send_action(goal)[7] == goal[7]
    runner._enter_soft_pause(confirm_command=True)
    assert runner._pause_active_event.is_set()
    assert not runner._execution_armed_event.is_set()
    np.testing.assert_array_equal(runner._last_action, measured)
    assert runner.plan.snapshot().chunk_id is None
    assert len(calls) == 1
    # A new accepted goal must replace the held target without a filter tail.
    resumed = measured.copy(); resumed[[7, 15]] = 0
    np.testing.assert_array_equal(runner._send_action(resumed)[[7, 15]], [0, 0])


def _tracking_runner():
    from nexus_deploy.robot.g2a.client import RobotClientRunner
    runner = object.__new__(RobotClientRunner)
    runner.hardware_mode = "execute"
    runner.adapter = SimpleNamespace(config=SimpleNamespace(effector_control_mode="device-position"))
    runner._execution_armed_event = threading.Event()
    runner._execution_armed_event.set()
    runner.shutdown_event = threading.Event()
    runner._last_action = np.zeros(24, dtype=np.float32)
    runner.max_arm_tracking_error = .4
    runner.max_effector_tracking_error = .01
    runner.max_head_waist_tracking_error = .2
    runner.tracking_error_consecutive_limit = 3
    runner._tracking_error_consecutive = 0
    runner._effector_tracking_consecutive = [0, 0]
    runner._effector_tracking_seen_revision = [0, 0]
    runner._effector_command_revision = [0, 0]
    runner._effector_last_command = None
    runner._fatal_error = None
    return runner


def test_separate_dahuan_openings_do_not_accumulate_a_false_stall():
    runner = _tracking_runner()
    closed = np.zeros(24, dtype=np.float32)
    closed[15] = .025
    opened = closed.copy()
    opened[7] = .025
    runner._record_effector_command(closed)
    # Recorded run: the left gripper reached 25 mm between these sparse
    # camera observations, while the policy requested close/open each time.
    for feedback in (.002150, .004550, .011925):
        runner._record_effector_command(opened)
        measured = opened.copy()
        measured[7] = feedback
        runner._last_action = opened
        runner._check_tracking_error(
            measured, command=opened,
            effector_revisions=tuple(runner._effector_command_revision),
        )
        assert runner._effector_tracking_consecutive == [1, 0]
        runner._record_effector_command(closed)
    assert not runner.shutdown_event.is_set()


def test_dahuan_opening_still_stops_after_three_errors_in_one_motion():
    runner = _tracking_runner()
    opened = np.zeros(24, dtype=np.float32)
    opened[[7, 15]] = .025
    runner._record_effector_command(opened)
    runner._last_action = opened
    measured = opened.copy()
    measured[7] = 0
    for _ in range(2):
        runner._check_tracking_error(measured, command=opened)
    with pytest.raises(RuntimeError, match="Persistent joint tracking error"):
        runner._check_tracking_error(measured, command=opened)
    assert runner.shutdown_event.is_set()


def test_arm_tracking_error_still_accumulates_across_gripper_commands():
    runner = _tracking_runner()
    closed = np.zeros(24, dtype=np.float32)
    opened = closed.copy()
    opened[7] = .025
    runner._record_effector_command(closed)
    for cycle in range(3):
        runner._record_effector_command(opened)
        runner._last_action = opened
        measured = opened.copy()
        measured[0] = .5
        if cycle == 2:
            with pytest.raises(RuntimeError, match="Persistent joint tracking error"):
                runner._check_tracking_error(measured, command=opened)
        else:
            runner._check_tracking_error(measured, command=opened)
        runner._record_effector_command(closed)
    assert runner.shutdown_event.is_set()
