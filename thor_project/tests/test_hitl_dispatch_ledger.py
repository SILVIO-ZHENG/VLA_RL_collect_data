"""Rollback accounting must follow checked dispatches, including queue lag."""

import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from nexus_deploy.robot.g2a.action_filter import JointKinematicLimiter
from nexus_deploy.robot.g2a.action_plan import ActionPlan
from nexus_deploy.robot.g2a.client import RobotClientRunner
from nexus_deploy.robot.g2a.hitl.g2_runtime.dispatch_ledger import RollbackDispatchLedger
from nexus_deploy.robot.g2a.hitl.g2_runtime.executed import FlowPROExecutionHistory
from nexus_deploy.robot.g2a.hitl.g2_runtime.rollback import (
    RollbackDispatchPending,
    RollbackExecutor,
    RollbackExecutorState,
    RollbackHealth,
    RollbackTrajectory,
)


def _ledger():
    path = np.zeros((5, 24), dtype=np.float32)
    path[:, 0] = np.arange(5, dtype=np.float32) * 0.01
    return RollbackDispatchLedger(lease_id="rollback-1", waypoints=path), path


def _receipt(path, *, sequence, base, offset=0, physical=True):
    return {
        "sequence": sequence,
        "physical_accepted": physical,
        "action": path[base + offset].copy(),
        "provenance": {
            "chunk_id": "rollback-1",
            "candidate_action_index": base,
            "reference_preview_index": offset,
            "holding_previous_target": False,
        },
    }


def test_pending_submission_does_not_advance_and_older_window_source_does():
    ledger, path = _ledger()
    assert ledger.observe(None) is None
    assert ledger.cursor == 0
    # Current caller may request point 3; receipt still identifies point 1
    # from an earlier certified preview window.
    first = ledger.observe(_receipt(path, sequence=11, base=1))
    assert first.waypoint_index == 1
    second = ledger.observe(_receipt(path, sequence=12, base=1, offset=1))
    assert second.waypoint_index == 2
    assert ledger.cursor == 2


def test_x_release_anchor_rejects_queued_overshoot():
    ledger, path = _ledger()
    ledger.observe(_receipt(path, sequence=1, base=1))
    ledger.select_endpoint(2)
    ledger.observe(_receipt(path, sequence=2, base=1, offset=1))
    assert ledger.complete
    with pytest.raises(RuntimeError, match="crossed its selected anchor"):
        ledger.observe(_receipt(path, sequence=3, base=2, offset=1))
    assert ledger.cursor == 2


def test_x_repress_can_restore_untouched_suffix_after_retirement():
    ledger, path = _ledger()
    ledger.observe(_receipt(path, sequence=1, base=1))
    ledger.select_endpoint(2)
    ledger.select_endpoint(4)
    assert ledger.endpoint == 4
    assert ledger.observe(_receipt(path, sequence=2, base=2)).waypoint_index == 2


@pytest.mark.parametrize("change", ["wrong_lease", "not_accepted", "changed_action", "skipped_point"])
def test_unattributable_or_modified_dispatch_fails_closed(change):
    ledger, path = _ledger()
    receipt = _receipt(path, sequence=1, base=1)
    if change == "wrong_lease":
        receipt["provenance"]["chunk_id"] = "other"
    elif change == "not_accepted":
        receipt["physical_accepted"] = False
    elif change == "changed_action":
        receipt["action"][0] += 0.005
    else:
        receipt["provenance"]["candidate_action_index"] = 2
        receipt["action"] = path[2].copy()
    with pytest.raises(RuntimeError):
        ledger.observe(receipt)
    assert ledger.cursor == 0


def test_preplanned_limiter_rejects_unreachable_point_without_advancing():
    limiter = JointKinematicLimiter(
        sample_hz=100.0,
        max_velocity=np.ones(24, dtype=np.float32),
        max_acceleration=np.ones(24, dtype=np.float32),
    )
    limiter.reset(np.zeros(24, dtype=np.float32))
    too_fast = np.zeros(24, dtype=np.float32)
    too_fast[0] = 0.02
    with pytest.raises(ValueError, match="velocity"):
        limiter.adopt_preplanned(too_fast)
    too_accelerated = np.zeros(24, dtype=np.float32)
    too_accelerated[0] = 0.0002
    with pytest.raises(ValueError, match="acceleration"):
        limiter.adopt_preplanned(too_accelerated)
    np.testing.assert_array_equal(limiter._position, np.zeros(24, dtype=np.float32))
    reachable = np.zeros(24, dtype=np.float32)
    reachable[0] = 0.00005
    np.testing.assert_array_equal(limiter.adopt_preplanned(reachable), reachable)


def test_legacy_executor_waits_for_checked_dispatch_without_advancing_cursor():
    path = np.zeros((3, 24), dtype=np.float32)
    path[:, 0] = [0.0, 0.00005, 0.0001]
    attempts = []

    def send(target):
        attempts.append(target.copy())
        if len(attempts) == 1:
            raise RollbackDispatchPending
        return target.copy()

    executor = RollbackExecutor(
        read_measured=lambda: path[0].copy(),
        read_command_motion=lambda: (path[0].copy(), np.zeros(24, dtype=np.float32)),
        safety_project=lambda _measured, target: target,
        send_command=send,
    )
    executor.plan = RollbackTrajectory(
        intervention_id="i",
        observation_id="o",
        waypoints_24d=path,
        servo_fps=100.0,
        source_sample_count=3,
        duration_s=0.02,
        collision_safety_enabled=False,
        safety_config_sha256=None,
        kinematics_calibrated=False,
    )
    executor.lease_id = "lease"
    executor.state = RollbackExecutorState.READY
    health = RollbackHealth(True, True, True)
    pending = executor.step(deadman_pressed=True, health=health, lease_id="lease")
    assert pending.reason == "rollback_dispatch_pending"
    assert pending.waypoint_index == executor.index == 0
    delivered = executor.step(deadman_pressed=True, health=health, lease_id="lease")
    assert delivered.waypoint_index == executor.index == 1
    np.testing.assert_array_equal(attempts[0], attempts[1])


def test_lineage_is_published_before_action_chunk_becomes_visible():
    plan = ActionPlan(24)
    plan.set_hold_action(np.zeros(24, dtype=np.float32))
    anchor = plan.snapshot()
    lineage = {}

    def publish(result):
        assert result.installed
        lineage["chunk"] = result.dropped_steps

    result = plan.install(
        "chunk", np.zeros((3, 24), dtype=np.float32), anchor,
        before_publish=publish,
    )
    assert result.installed
    assert lineage[plan.snapshot().chunk_id] == 0

    old_snapshot = plan.snapshot()

    def failed_publish(_result):
        raise RuntimeError("lineage unavailable")

    with pytest.raises(RuntimeError, match="lineage unavailable"):
        plan.install(
            "new-chunk", np.zeros((3, 24), dtype=np.float32), old_snapshot,
            before_publish=failed_publish,
        )
    assert plan.snapshot().chunk_id == "chunk"


def test_preview_keeps_policy_indices_at_a_different_servo_frequency():
    plan = ActionPlan(24, policy_fps=30.0)
    plan.set_hold_action(np.zeros(24, dtype=np.float32))
    actions = np.zeros((4, 24), dtype=np.float32)
    actions[:, 0] = [0.01, 0.02, 0.03, 0.04]
    assert plan.install("chunk", actions, plan.snapshot(), now=10.0).installed
    reservation = plan.reserve(now=10.0)
    preview, indices = plan.preview_reserved_with_indices(
        reservation, now=10.0, period=0.01, count=8
    )
    assert preview.shape == (8, 24)
    assert indices[:4] == (0, 0, 0, 0)
    assert indices[4:] == (1, 1, 1, 2)
    plan.cancel(reservation)


def test_policy_history_uses_dispatched_old_chunk_provenance():
    old_chunk, newer_chunk = "old-chunk", "new-chunk"
    generation = str(uuid4())
    observation_id, candidate_id = str(uuid4()), str(uuid4())
    measured = np.zeros(24, dtype=np.float32)
    command = measured.copy()
    command[0] = 0.001
    native = np.zeros((3, 24), dtype=np.float32)
    native[1, 0] = 0.001
    receipt = {
        "sequence": 2, "physical_accepted": True,
        "measured_state": measured, "measured_age_ms": 5.0,
        "measured_timestamp_ns": 1_000_000_000,
        "action": command,
        "provenance": {
            "chunk_id": old_chunk,
            "candidate_action_index": 0,
            "policy_action_index": 1,
            "holding_previous_target": False,
            "safety_target_24d": command.copy(),
        },
    }
    runner = object.__new__(RobotClientRunner)
    runner.hardware_mode = "execute"
    history = FlowPROExecutionHistory()
    runner.flowpro_capture = SimpleNamespace(
        execution_history=history, active_generation_id=generation
    )
    runner.adapter = SimpleNamespace(dispatch_receipt=lambda: receipt)
    runner._last_flowpro_sample_ns = None
    runner._flowpro_arbitration_monitor = None
    runner._flowpro_chunk_lineage = {
        old_chunk: {
            "observation_id": observation_id, "candidate_id": candidate_id,
            "generation_id": generation, "native_actions": native,
            "request_started_monotonic_ns": 100,
            "response_received_monotonic_ns": 200,
            "chunk_installed_monotonic_ns": 300,
            "latency_stale_steps": 0,
            "installed_first_action_index": 0,
            "first_executed_timestamp_ns": None,
            "first_executed_action_index": None,
        },
        newer_chunk: {"irrelevant": True},
    }
    runner._record_flowpro_policy_dispatch()
    sample = history.samples[0]
    assert sample.chunk_id == old_chunk
    assert sample.action_index == 1
    np.testing.assert_array_equal(sample.policy_native_action, native[1])
    np.testing.assert_array_equal(sample.gdk_command_24d, command)

    receipt["provenance"] = {
        "chunk_id": newer_chunk,
        "holding_previous_target": True,
    }
    receipt["measured_timestamp_ns"] += 10_000_000
    runner._record_flowpro_policy_dispatch()
    assert len(history.samples) == 2
    assert history.samples[1].execution_kind == "queue_underrun_hold"


def test_flowpro_action_slot_prioritizes_reanchor_and_rollback_over_policy():
    runner = object.__new__(RobotClientRunner)
    runner.flowpro_capture = object()
    runner._flowpro_hold_reanchor_requested_event = threading.Event()
    runner._flowpro_rollback_lock = threading.Lock()
    runner._flowpro_rollback_lease_id = "lease"
    runner._flowpro_human_control_active_event = threading.Event()
    runner.adapter = SimpleNamespace(dispatch_receipt=lambda: None)
    calls = []
    runner._execute_flowpro_human_hold_reanchor = lambda: calls.append("reanchor")
    runner._step_flowpro_rollback = lambda: calls.append("rollback")

    runner._flowpro_hold_reanchor_requested_event.set()
    assert runner._flowpro_action_tick()
    assert calls == ["reanchor"]

    runner._flowpro_hold_reanchor_requested_event.clear()
    assert runner._flowpro_action_tick()
    assert calls == ["reanchor", "rollback"]

    accepted = np.ones(24, dtype=np.float32) * 0.001
    runner.adapter.dispatch_receipt = lambda: {
        "physical_accepted": True,
        "action": accepted,
    }
    runner._servo_ticks = 0
    assert runner._flowpro_action_tick()
    assert calls == ["reanchor", "rollback", "rollback"]
    np.testing.assert_array_equal(runner._last_action, accepted)
    assert runner._servo_ticks == 1

    runner._flowpro_rollback_lease_id = None
    runner._flowpro_human_control_active_event.set()
    assert runner._flowpro_action_tick()
    assert calls == ["reanchor", "rollback", "rollback"]

    runner._flowpro_rollback_lease_id = "stale-lease"
    assert runner._flowpro_action_tick()
    assert calls == ["reanchor", "rollback", "rollback"]

    runner._flowpro_human_control_active_event.clear()
    runner._flowpro_rollback_lease_id = None
    assert not runner._flowpro_action_tick()
