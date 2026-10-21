"""Lazy lookahead changes computation frequency, never checked numerical data."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexus_deploy.robot.g2a.action_filter import CausalLowPassFilter  # noqa: E402
from nexus_deploy.robot.g2a.client import RobotClientRunner  # noqa: E402


def runner(consume, previews):
    obj = object.__new__(RobotClientRunner)
    obj._held_head_waist_values = None
    obj._held_effector_values = None
    obj._action_filter = CausalLowPassFilter(cutoff_hz=2, sample_hz=100, action_dim=24)
    obj._action_filter.reset(np.zeros(24))
    obj._effector_action_filter = None
    obj._control_lock = threading.Lock()

    def send(target, safety_context):
        value = safety_context["trajectory_preview"]
        if consume:
            if callable(value):
                first = value().copy()
                np.testing.assert_array_equal(value(), first)  # second use must not filter twice
                previews.append(first)
            else:
                previews.append(value.copy())
        return target.copy()

    obj.adapter = SimpleNamespace(_safety_layer=True, send_action=send, last_action_intervened=False)
    return obj


def test_lazy_matches_eager_bitwise_and_preserves_live_filter():
    raw = np.arange(15 * 24, dtype=np.float32).reshape(15, 24) * 0.00001
    eager, lazy = [], []
    a, b = runner(True, eager), runner(True, lazy)
    calls = []

    def source():
        calls.append(1)
        return raw

    sent_a = a._send_action(raw[0], trajectory_preview=raw)
    sent_b = b._send_action(raw[0], trajectory_preview=source)
    assert len(calls) == 1
    np.testing.assert_array_equal(sent_a, sent_b)
    np.testing.assert_array_equal(eager[0], lazy[0])
    np.testing.assert_array_equal(eager[0][0], sent_a)
    np.testing.assert_array_equal(a._action_filter._value, b._action_filter._value)
    np.testing.assert_array_equal(lazy[0][:, [7, 15]], raw[:, [7, 15]])


def test_unused_preview_is_not_computed():
    def unused():
        pytest.fail("Full queue must not request a trajectory preview")

    obj = runner(False, [])
    result = obj._send_action(np.ones(24) * 0.001, trajectory_preview=unused)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("busy", ["pending", "full", "submit"])
def test_adapter_evaluates_only_for_submission_and_never_sends_callable_to_worker(tmp_path, monkeypatch, busy):
    spec = importlib.util.spec_from_file_location(
        "lazy_queue_helpers", Path(__file__).with_name("test_segmented_rolling_queue.py")
    )
    h = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = h
    spec.loader.exec_module(h)
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    calls = []

    def preview():
        calls.append(1)
        return np.tile(target, (15, 1))

    try:
        if busy == "pending":
            adapter._async_pending = {}
        elif busy == "full":
            adapter._async_steps = [None] * 10
        adapter._submit_async_window(target, target, {"trajectory_preview": preview})
        assert len(calls) == (1 if busy == "submit" else 0)
        if busy == "submit":
            assert "trajectory_preview" not in worker.tasks[-1]["context"]
            assert len(worker.tasks[-1]["targets"]) == 10
    finally:
        adapter._async_steps = []
        adapter.disconnect()
