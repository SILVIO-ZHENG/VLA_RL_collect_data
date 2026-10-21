"""Production clock/provenance regressions using only mocks, never GDK."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nexus_deploy.robot.g2a.action_plan import ActionPlan  # noqa: E402
from nexus_deploy.robot.g2a.client import RobotClientRunner  # noqa: E402


def prepare():
    plan = ActionPlan(2, 30)
    plan.set_hold_action(np.zeros(2))
    plan.install("old", np.ones((50, 2)), plan.snapshot(), now=0)
    anchor = plan.snapshot()
    reservation = plan.reserve(now=0.01)
    return plan, anchor, reservation


def test_install_clock_starts_after_old_sdk_call_commits(monkeypatch):
    plan, anchor, reserved = prepare()
    clock = [10.0]
    monkeypatch.setattr("nexus_deploy.robot.g2a.action_plan.time.perf_counter", lambda: clock[0])

    def finish_old_call():
        clock[0] += 0.12
        plan.commit(reserved, np.array([0.2, 0.4]))

    monkeypatch.setattr(plan._condition, "wait", finish_old_call)
    result = plan.install("new", np.tile([1.0, 2.0], (50, 1)), anchor)
    assert result.installed
    assert result.wait_ms == pytest.approx(120)
    assert result.installed_at == pytest.approx(10.12)
    first = plan.reserve(now=clock[0])
    np.testing.assert_allclose(first.action, [0.2, 0.4])
    assert first.advanced_policy_steps == 0  # no immediate 120 ms jump
    plan.commit(first)
    second = plan.reserve(now=clock[0] + 0.01)
    np.testing.assert_allclose(second.action, [0.44, 0.88], atol=1e-6)


def test_blend_limit_rechecked_atomically_after_progress_changes(monkeypatch):
    plan, anchor, reserved = prepare()
    plan.cancel(reserved)
    reserved = plan.reserve(now=0.2)  # six source steps in flight
    monkeypatch.setattr(plan._condition, "wait", lambda: plan.commit(reserved))
    result = plan.install("new", np.zeros((50, 2)), anchor, max_dropped_steps=5, now=1)
    assert not result.installed and result.dropped_steps == 6
    assert "blend budget" in result.reason
    assert plan.snapshot().chunk_id == "old"


def test_pause_during_install_wait_still_retires_response(monkeypatch):
    plan, anchor, reserved = prepare()

    def pause():
        plan.clear()
        plan.commit(reserved)

    monkeypatch.setattr(plan._condition, "wait", pause)
    result = plan.install("new", np.zeros((50, 2)), anchor)
    assert not result.installed
    assert plan.snapshot().chunk_id is None


def test_numerical_reply_diagnostics_keep_both_gripper_conventions_and_exclude_images():
    rows = []
    runner = object.__new__(RobotClientRunner)
    runner.control_diagnostics = SimpleNamespace(write=rows.append)
    model = np.zeros((50, 24), dtype=np.float32)
    physical = model.copy()
    physical[:, [7, 15]] = 0.025
    runner._record_inference_response_diagnostics(
        request_id="r",
        request_session="s",
        response={"chunk_id": "c", "rtc": {"prefix_used": True}},
        policy_actions=model,
        response_actions=physical,
        observation={"state": np.zeros(24), "images": {"data": b"not to be logged"}, "observation_monotonic_time": 1},
        anchor=SimpleNamespace(chunk_id="old", next_action_index=3, generation=4),
        result=SimpleNamespace(installed=True, dropped_steps=9, reason="", installed_at=1.3, wait_ms=5),
        response_received_at=1.29,
        stale_steps=9,
    )
    assert len(rows) == 1 and rows[0]["record_type"] == "inference_response"
    assert rows[0]["installed_monotonic"] == 1.3
    assert "images" not in rows[0] and "observation" not in rows[0]
    np.testing.assert_array_equal(rows[0]["control_targets"][:, 7], physical[:, 7])
    np.testing.assert_array_equal(rows[0]["model_actions"][:, 7], model[:, 7])


def test_dispatch_identity_comes_from_checked_queue_not_new_candidate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "timing_queue_helpers", Path(__file__).with_name("test_segmented_rolling_queue.py")
    )
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    adapter, clock, worker, feedback, target = helpers._setup(tmp_path, monkeypatch)
    try:
        preview = np.tile(target, (15, 1))
        old = dict(chunk_id="old", action_index=7, source_sample_monotonic=clock[0], trajectory_preview=preview)
        adapter.send_action(target, safety_context=old)
        assert adapter._last_dispatch_provenance is None
        worker.finish(started=clock[0], measured=feedback)
        clock[0] += 0.01
        worker.feedback = feedback.copy(), clock[0]
        adapter.send_action(target, safety_context={**old, "chunk_id": "new", "action_index": 0})
        provenance = adapter._last_dispatch_provenance
        assert provenance["chunk_id"] == "old"
        assert provenance["candidate_action_index"] == 7
        assert provenance["trajectory_step"] == 0
        assert provenance["reference_preview_index"] == 0
        assert provenance["reference_sample_monotonic"] == old["source_sample_monotonic"]
        assert not provenance["holding_previous_target"]
        adapter.send_action(target, safety_context={**old, "chunk_id": "new"})
        assert adapter._last_dispatch_provenance["trajectory_step"] == 1
        assert adapter._last_dispatch_provenance["reference_sample_monotonic"] == pytest.approx(
            old["source_sample_monotonic"] + 0.01
        )
    finally:
        adapter.disconnect()
