"""Deterministic segmented-queue regressions; no GDK, processes, or wall waits."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest


_spec = importlib.util.spec_from_file_location(
    "segmented_rolling_helpers", Path(__file__).with_name("test_legacy_rolling_geometry.py")
)
_helpers = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _helpers
_spec.loader.exec_module(_helpers)

from nexus_deploy.robot.g2a.async_safety import (  # noqa: E402
    check_legacy_rolling_window,
    feedback_ready_for_check,
)


def _setup(tmp_path, monkeypatch):
    adapter, clock, worker = _helpers.setup(tmp_path, monkeypatch)
    feedback = adapter._dry_state.copy()
    target = feedback.copy()
    target[0] += 0.2
    return adapter, clock, worker, feedback, target


def _tick(adapter, clock, worker, feedback, target):
    clock[0] += 0.01
    worker.feedback = feedback.copy(), clock[0]
    return adapter.send_action(target)


def _prime(adapter, clock, worker, feedback, target):
    adapter.send_action(target)
    worker.finish(started=clock[0], measured=feedback)
    _tick(adapter, clock, worker, feedback, target)
    assert adapter.last_async_response == "async_checked_motion"


def test_feedback_published_during_poll_does_not_discard_approved_result(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0], measured=feedback)
        first_checked = worker.tasks[0]["targets"][0].copy()
        original_poll = worker.poll

        def poll_with_new_feedback():
            # The send loop has already captured `now`. A concurrent feedback
            # publisher moves time forward before the returned result is used.
            clock[0] += 0.002
            worker.feedback = feedback.copy(), clock[0]
            return original_poll()

        monkeypatch.setattr(worker, "poll", poll_with_new_feedback)
        adopted = _tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(adopted, first_checked)
        assert adapter.last_async_response == "async_checked_motion"
        assert not adapter.last_command_pending
        assert len(adapter._async_steps) == 9
    finally:
        adapter.disconnect()


def test_segment_appends_at_exact_tail_and_preserves_every_approved_point(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        assert len(worker.tasks[0]["targets"]) == 10
        assert len(adapter._async_steps) == 9
        assert adapter._async_pending is None  # Refill only at eight or fewer.

        opposite = target.copy()
        opposite[0] -= 0.4
        _tick(adapter, clock, worker, feedback, opposite)
        prefix = list(adapter._async_steps)
        pending = adapter._async_pending
        assert len(prefix) == 8
        assert len(pending["targets"]) == 7
        np.testing.assert_array_equal(pending["anchor"], prefix[-1]["action"])
        np.testing.assert_array_equal(pending["anchor_velocity"], prefix[-1]["limiter"].velocity)
        assert pending["anchor_identity"] == (
            prefix[-1]["result"]["task"]["generation"],
            prefix[-1]["result"]["task"]["request_id"],
            prefix[-1]["step"],
        )

        worker.finish(started=clock[0], measured=feedback)
        adopted = _tick(adapter, clock, worker, feedback, opposite)
        np.testing.assert_array_equal(adopted, prefix[0]["action"])
        expected = [step["action"] for step in prefix[1:]] + pending["targets"]
        assert len(adapter._async_steps) == 14
        for actual, expected_action in zip(adapter._async_steps, expected, strict=True):
            np.testing.assert_array_equal(actual["action"], expected_action)
        for actual, original in zip(adapter._async_steps[:7], prefix[1:], strict=True):
            assert actual["step"] == original["step"]
            assert actual["result"]["task"]["request_id"] == original["result"]["task"]["request_id"]
    finally:
        adapter.disconnect()


@pytest.mark.parametrize("delay_ticks", [2, 3, 4, 5])
def test_delayed_segment_results_cover_100hz_without_post_startup_holds(
    tmp_path, monkeypatch, delay_ticks
):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        submitted = {}
        completed = set()
        for tick in range(200):
            clock[0] += 0.01
            worker.feedback = feedback.copy(), clock[0]
            pending = adapter._async_pending
            if pending is not None:
                request_id = pending["request_id"]
                submitted.setdefault(request_id, (tick - 1, clock[0] - 0.01))
                submitted_tick, submitted_at = submitted[request_id]
                if tick - submitted_tick >= delay_ticks:
                    worker.finish(
                        started=submitted_at,
                        check_ms=delay_ticks * 10.0,
                        measured=feedback,
                    )
                    completed.add(request_id)
            adapter.send_action(target)
            assert adapter.last_async_response == "async_checked_motion", (
                delay_ticks, tick, adapter.last_async_response
            )
            assert not adapter.last_action_intervened
            assert len(adapter._async_steps) <= 14  # At most 15 before pop.
        assert len(completed) >= (200 - 10) // 7  # 70 ms suffixes cover the 2 s run.
        assert all(len(task["targets"]) == 7 for task in worker.tasks[1:])
    finally:
        adapter.disconnect()


@pytest.mark.parametrize("expired", ["feedback", "geometry", "future_feedback"])
def test_invalid_feedback_or_geometry_cannot_renew_approved_segment(tmp_path, monkeypatch, expired):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0], measured=feedback)
        result = worker.result
        if expired == "feedback":
            clock[0] += 0.041
        elif expired == "geometry":
            clock[0] += 0.201
            worker.feedback = feedback.copy(), clock[0]
        else:
            worker.feedback = feedback.copy(), clock[0] + 0.01
        assert adapter._renew_rolling_result(result, clock[0]) is None
    finally:
        adapter.disconnect()


def test_occasional_sixty_ms_check_is_absorbed_without_holding(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        submitted = {}
        completed = set()
        slow_results = 0
        for tick in range(300):
            clock[0] += 0.01
            worker.feedback = feedback.copy(), clock[0]
            pending = adapter._async_pending
            if pending is not None:
                request_id = pending["request_id"]
                submitted.setdefault(request_id, (tick - 1, clock[0] - 0.01))
                submitted_tick, submitted_at = submitted[request_id]
                delay_ticks = 6 if request_id % 7 == 0 else 3
                if tick - submitted_tick >= delay_ticks:
                    worker.finish(
                        started=submitted_at, check_ms=delay_ticks * 10.0,
                        measured=feedback,
                    )
                    completed.add(request_id)
                    slow_results += delay_ticks == 6
            adapter.send_action(target)
            assert adapter.last_async_response == "async_checked_motion", tick
            assert not adapter.last_action_intervened
            assert len(adapter._async_steps) <= 12
        assert len(completed) >= 40
        assert slow_results >= 5
    finally:
        adapter.disconnect()


def test_drained_queue_holds_and_never_revives_late_pre_hold_velocity(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        _tick(adapter, clock, worker, feedback, target)
        delayed_task = adapter._async_pending
        delayed_started = clock[0]
        assert len(adapter._async_steps) == 8
        expected_hold = adapter._async_steps[-1]["action"].copy()
        assert np.any(delayed_task["anchor_velocity"])

        # Do not return any new result. Consume exactly the already checked
        # prefix before holding; no extrapolation through an unapproved point.
        for _ in range(8):
            _tick(adapter, clock, worker, feedback, target)
            assert adapter.last_async_response == "async_checked_motion"
        assert not adapter._async_steps
        for _ in range(6):
            adopted = _tick(adapter, clock, worker, feedback, target)
            np.testing.assert_array_equal(adopted, expected_hold)
            np.testing.assert_array_equal(adapter._kinematic_limiter.velocity, np.zeros(24))
            assert adapter.last_async_response == "async_checked_hold"
            assert adapter.last_action_intervened
            assert not adapter._async_steps
        assert delayed_task["motion_cancelled"]

        # The old geometry can still be safe, but its nonzero-velocity anchor
        # predates the real hold. It must NOT restart motion from that velocity.
        worker.finish(started=delayed_started, check_ms=140.0, measured=feedback)
        assert not worker.result["decision"].hard_limit_triggered
        adopted = _tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(adopted, expected_hold)
        assert adapter.last_async_response == "async_checked_hold"
        assert not adapter._async_steps
        recovery = adapter._async_pending
        assert recovery["request_id"] != delayed_task["request_id"]
        np.testing.assert_array_equal(recovery["anchor"], expected_hold)
        np.testing.assert_array_equal(recovery["anchor_velocity"], np.zeros(24))

        # Only a newly checked rest-to-motion segment is permitted to resume.
        worker.finish(started=clock[0], measured=feedback)
        expected_restart = recovery["targets"][0].copy()
        adopted = _tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(adopted, expected_restart)
        assert not np.array_equal(adopted, delayed_task["targets"][0])
        assert adapter.last_async_response == "async_checked_motion"
        assert not adapter.last_action_intervened
    finally:
        adapter.disconnect()


def test_moving_feedback_does_not_invalidate_safe_segment(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        # Deliberately make the historical whole-body bound unusable. Geometry
        # still checks every committed and new waypoint against moving feedback.
        adapter._async_hold_coefficients[:] = 10000
        for tick in range(150):
            moving = adapter._last_state.copy()
            moving[0] += .005 * np.sin(tick * .2)
            if adapter._async_pending is not None and tick % 3 == 0:
                worker.finish(started=clock[0], measured=moving)
            _tick(adapter, clock, worker, moving, target)
            assert adapter.last_async_response == "async_checked_motion"
        assert any(task["committed"] for task in worker.tasks)
        result = adapter._async_hold["result"]
        assert result["started"] == result["geometry_feedback_at"]
        assert result["started"] < clock[0]
    finally:
        adapter.disconnect()


def test_latest_committed_path_collision_invalidates_older_approval(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        _tick(adapter, clock, worker, feedback, target)
        worker.finish(started=clock[0], measured=feedback)
        worker.result["live_decision"] = replace(worker.result["live_decision"], hard_limit_triggered=True)
        previous = adapter._last_state.copy()
        _tick(adapter, clock, worker, feedback, target)
        assert adapter.last_async_response == "async_collision_hold"
        assert not adapter._async_steps
        np.testing.assert_array_equal(adapter._last_state, previous)
    finally:
        adapter.disconnect()


def test_sampled_snapshot_cannot_be_extended_by_fresh_feedback(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        clock[0] += .201
        worker.feedback = feedback, clock[0]
        with pytest.raises(_helpers.h.gdk_adapter.SafetyDispatchExpiredError, match="worker unavailable"):
            adapter._select_async_hold(clock[0])
    finally:
        adapter.disconnect()


def test_late_worker_result_bridges_only_identical_hold_with_pending_recheck(
    tmp_path, monkeypatch
):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        clock[0] += .201
        worker.feedback = feedback, clock[0]
        adapter._async_pending = {
            "anchor": adapter._last_state.copy(),
            "fallback_anchor": adapter._last_state.copy(),
            "request_id": 999,
        }
        selected = adapter._select_async_hold(clock[0])
        assert selected["historical_repeat"]
        assert selected["recheck_bridge"]
        np.testing.assert_array_equal(selected["action"], adapter._last_state)
        assert selected["dispatch_authority_at"] == clock[0]

        clock[0] += .300
        worker.feedback = feedback, clock[0]
        with pytest.raises(_helpers.h.gdk_adapter.SafetyDispatchExpiredError, match="worker unavailable"):
            adapter._select_async_hold(clock[0])
    finally:
        adapter.disconnect()


def test_audit_timestamp_is_captured_before_background_writer_runs(tmp_path, monkeypatch):
    safety = _helpers.h.safety
    original_run = safety.SafetyJSONLLogger._run
    release_writer = threading.Event()
    clock = {"wall_ns": 123_000_000, "monotonic": 10.0}

    def delayed_writer(logger):
        assert release_writer.wait(timeout=2.0)
        original_run(logger)

    monkeypatch.setattr(safety.SafetyJSONLLogger, "_run", delayed_writer)
    monkeypatch.setattr(safety.time, "time_ns", lambda: clock["wall_ns"])
    monkeypatch.setattr(safety.time, "perf_counter", lambda: clock["monotonic"])
    logger = safety.SafetyJSONLLogger(tmp_path / "production-time.jsonl")
    context = {"record_type": "safety_lifecycle", "event": "request_submitted"}
    try:
        measured = np.array([0.1, 0.0, 0.0])
        decision = _helpers.h.make_layer().guard(measured, measured)
        logger.write_event(context)
        logger.write(decision, {"record_type": "servo_dispatch"})
        clock.update(wall_ns=900_000_000, monotonic=90.0)
    finally:
        release_writer.set()
        logger.close()
    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["timestamp_ns"] == 123_000_000 for record in records)
    assert all(record["event_monotonic"] == 10.0 for record in records)
    assert "timestamp_ns" not in context  # Caller context remains unchanged.


def _fake_execution_boundary(adapter, clock, feedback, monkeypatch):
    """Replace SDK boundaries, not readiness, certificate, or dispatch logic."""
    sent = []
    configured = []
    monkeypatch.setattr(adapter, "_wait_servo_slot", lambda **kwargs: None)
    monkeypatch.setattr(adapter, "_configure_current_pose", lambda: configured.append(True))
    monkeypatch.setattr(adapter, "read_state", lambda: feedback.copy())

    def fake_send_combined_action(action):
        assert adapter._execution_prepared
        assert adapter._command_safety_started is not None
        np.testing.assert_array_equal(action, adapter._async_dispatch_action)
        adapter._pace_servo_dispatch()
        adapter._validate_dispatch_age(clock[0])
        sent.append(action.copy())

    monkeypatch.setattr(adapter, "_send_combined_action", fake_send_combined_action)
    return sent, configured


def test_segmented_readonly_and_fake_execute_select_identical_points_and_queues(tmp_path, monkeypatch):
    outcomes = {}
    for mode in ("read-only", "execute"):
        mode_path = tmp_path / mode
        mode_path.mkdir()
        with monkeypatch.context() as mode_patch:
            adapter, clock, worker, feedback, target = _setup(mode_path, mode_patch)
            sent, configured = _fake_execution_boundary(adapter, clock, feedback, mode_patch)
            adapter.config.hardware_mode = mode
            try:
                if mode == "execute":
                    assert not adapter._execution_prepared
                    # Exercise the actual explicit readiness transition while
                    # its SDK-facing operations are test doubles only.
                    adapter.prepare_execution()
                    assert configured == [True]
                    assert adapter._execution_prepared
                else:
                    adapter.reset_command_reference(feedback)
                    assert not adapter._execution_prepared
                assert not sent
                adapter.send_action(target)
                assert adapter.last_command_pending
                assert not sent  # Readiness alone does not certify a target.
                worker.finish(started=clock[0], measured=feedback)

                submitted = {}
                selected, queues = [], []
                for tick in range(150):
                    clock[0] += 0.01
                    worker.feedback = feedback.copy(), clock[0]
                    pending = adapter._async_pending
                    if pending is not None and worker.result is None:
                        request_id = pending["request_id"]
                        submitted.setdefault(request_id, (tick - 1, clock[0] - 0.01))
                        submitted_tick, submitted_at = submitted[request_id]
                        delay_ticks = 5 if request_id % 4 == 0 else 3
                        if tick - submitted_tick >= delay_ticks:
                            worker.finish(
                                started=submitted_at, check_ms=delay_ticks * 10.0,
                                measured=feedback,
                            )
                    next_target = target.copy()
                    next_target[0] += 0.05 * np.sin(tick * 0.1)
                    selected.append(adapter.send_action(next_target).copy())
                    queues.append([point["action"].copy() for point in adapter._async_steps])
                    assert adapter.last_async_response == "async_checked_motion"
                    assert not adapter.last_action_intervened
                    assert len(adapter._async_steps) <= 12
                assert not adapter._gdk_initialized
                if mode == "execute":
                    np.testing.assert_array_equal(sent, selected)
                else:
                    assert not sent
                outcomes[mode] = selected, queues
            finally:
                adapter.config.hardware_mode = "dry-run"
                adapter.disconnect()

    np.testing.assert_array_equal(outcomes["read-only"][0], outcomes["execute"][0])
    for readonly_queue, execute_queue in zip(outcomes["read-only"][1], outcomes["execute"][1], strict=True):
        np.testing.assert_array_equal(readonly_queue, execute_queue)


def test_execution_without_preparation_cannot_dispatch_even_approved_queue(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        _prime(adapter, clock, worker, feedback, target)
        assert adapter._async_steps
        approved_count = len(adapter._async_steps)
        sent, configured = _fake_execution_boundary(adapter, clock, feedback, monkeypatch)
        adapter.config.hardware_mode = "execute"
        assert not adapter._execution_prepared
        with pytest.raises(RuntimeError, match="Physical execution is not prepared"):
            _tick(adapter, clock, worker, feedback, target)
        assert not sent
        assert not configured
        assert len(adapter._async_steps) == approved_count
    finally:
        adapter.config.hardware_mode = "dry-run"
        adapter.disconnect()


@pytest.mark.parametrize("expired_result", [False, True])
def test_prepared_execution_without_valid_certificate_never_reaches_fake_sdk(
    tmp_path, monkeypatch, expired_result
):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        sent, configured = _fake_execution_boundary(adapter, clock, feedback, monkeypatch)
        adapter.config.hardware_mode = "execute"
        adapter.prepare_execution()
        assert configured == [True]
        adapter.send_action(target)
        assert adapter.last_command_pending
        assert not sent
        if expired_result:
            worker.finish(started=clock[0], measured=feedback)
            clock[0] += 0.041
            # Neither fake feedback nor the completed certificate is fresh.
            adapter.send_action(target)
        else:
            for _ in range(3):
                _tick(adapter, clock, worker, feedback, target)
        if not expired_result:
            assert adapter.last_command_pending
        assert not adapter._async_steps
        assert not sent
        assert not adapter._gdk_initialized
    finally:
        adapter.config.hardware_mode = "dry-run"
        adapter.disconnect()


def test_stride_five_segment_checks_middle_waypoint_not_just_endpoints():
    layer = _helpers.h.make_layer()
    measured = np.zeros(3, dtype=np.float64)
    measured[0] = 0.1
    targets = [measured.copy() for _ in range(5)]
    targets[2][0] = 0.0  # Both segment endpoints are safe; its middle is not.
    checked = check_legacy_rolling_window(layer, measured, measured, targets, targets[-1], measured)
    assert not checked["hold"].hard_limit_triggered
    assert not checked["target_holds"][0].hard_limit_triggered
    assert not checked["target_holds"][-1].hard_limit_triggered
    assert checked["decision"].hard_limit_triggered
    np.testing.assert_array_equal(checked["decision"].safe_action, measured)


def test_segment_checks_interpolation_between_safe_waypoints():
    safety = _helpers.h.safety
    layer = _helpers.h.make_layer(points=(
        safety.CollisionPoint("rotating_finger", "tool", (0.0, 0.1, 0.0)),
    ))
    measured = np.array([0.05, 0.0, 0.0])
    targets = [measured.copy() for _ in range(5)]
    for target in targets[2:]:
        target[1] = -np.pi
    # Every commanded waypoint is above the surface, but rotation between
    # points 1 and 2 dips the fingertip below it. Sampling only endpoints fails.
    assert all(layer.minimum_z(q).z > 0.02 for q in [measured, *targets])
    checked = check_legacy_rolling_window(layer, measured, measured, targets, targets[-1], measured)
    assert checked["decision"].hard_limit_triggered


def test_long_rolling_check_is_not_mislabeled_as_over_budget_hold(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0], check_ms=35.0, measured=feedback)
        _tick(adapter, clock, worker, feedback, target)
        assert adapter.last_async_response == "async_checked_motion"
        assert adapter._async_hold["result"].get("hold_reason") is None
        # Withhold the NEXT segment until this approved queue genuinely drains.
        for _ in range(10):
            _tick(adapter, clock, worker, feedback, target)
        assert adapter.last_async_response == "async_checked_hold"
    finally:
        adapter.disconnect()
    records = [json.loads(line) for line in (tmp_path / "paired.jsonl").read_text().splitlines()]
    records = [record for record in records if record.get("record_type") == "servo_dispatch"]
    assert records[-1]["hold_reason"] == "awaiting_fresh_window"
    assert all(record.get("hold_reason") != "check_over_budget" for record in records)


def test_actual_collision_remains_an_explicit_hold_reason(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = _setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0], measured=feedback)
        worker.result["decision"] = replace(
            worker.result["decision"], hard_limit_triggered=True,
            action_scale=0.0, safe_action=feedback.copy(),
        )
        adopted = _tick(adapter, clock, worker, feedback, target)
        np.testing.assert_array_equal(adopted, feedback)
        assert adapter.last_async_response == "async_collision_hold"
    finally:
        adapter.disconnect()
    records = [json.loads(line) for line in (tmp_path / "paired.jsonl").read_text().splitlines()]
    records = [record for record in records if record.get("record_type") == "servo_dispatch"]
    assert records[-1]["hold_reason"] == "collision"


@pytest.mark.parametrize(
    "rolling,age,expected",
    [(True, 0.006, True), (True, 0.0199, True), (True, 0.0201, False),
     (False, 0.004, True), (False, 0.006, False)],
)
def test_rolling_feedback_cadence_uses_one_sample_period_not_five_ms(rolling, age, expected):
    now = 100.0
    feedback = np.zeros(24), now - age
    task = {"rolling": rolling, "submitted_at": now, "urgent": False}
    assert feedback_ready_for_check(feedback, task, now - 0.1, now) is expected


def test_rolling_feedback_cadence_does_not_recheck_the_same_sample():
    now = 100.0
    feedback = np.zeros(24), now - 0.001
    task = {"rolling": True, "submitted_at": now, "urgent": False}
    assert not feedback_ready_for_check(feedback, task, feedback[1], now)
