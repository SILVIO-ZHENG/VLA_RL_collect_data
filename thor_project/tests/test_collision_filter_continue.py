"""Collision is a filter outcome, not session termination. No real GDK calls."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('continue_helpers', Path(__file__).with_name('test_segmented_rolling_queue.py'))
h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)


@pytest.mark.parametrize('hardware', ['dry-run', 'execute'])
@pytest.mark.parametrize('collision_part', ['decision', 'live_decision'])
def test_continuous_collision_hold_then_smooth_resume(tmp_path, monkeypatch, hardware, collision_part):
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    try:
        if hardware == 'execute':
            h._fake_execution_boundary(adapter, clock, feedback, monkeypatch)
            adapter.config.hardware_mode = hardware
            adapter.prepare_execution()
        h._prime(adapter, clock, worker, feedback, target)
        h._tick(adapter, clock, worker, feedback, target)
        held = adapter._last_state.copy()  # Includes stride-skipped accepted point.
        original_identity = adapter._async_last_waypoint
        # 3 seconds of blocked candidates must not expire the historical target.
        for i in range(300):
            if i % 3 == 0:
                worker.finish(started=clock[0], measured=feedback)
                worker.result[collision_part] = replace(worker.result[collision_part], hard_limit_triggered=True)
            output = h._tick(adapter, clock, worker, feedback, target)
            np.testing.assert_array_equal(output, held)
            assert adapter.last_async_response == 'async_collision_hold'
            assert adapter._async_last_waypoint == original_identity
            np.testing.assert_array_equal(adapter._kinematic_limiter.velocity, np.zeros(24))
        pending = adapter._async_pending
        np.testing.assert_array_equal(pending['anchor'], held)
        np.testing.assert_array_equal(pending['anchor_velocity'], np.zeros(24))
        worker.finish(started=clock[0], measured=feedback)
        output = h._tick(adapter, clock, worker, feedback, target)
        assert adapter.last_async_response == 'async_checked_motion'
        assert not adapter._async_collision_active
        np.testing.assert_array_equal(output, pending['targets'][0])
        assert np.max(abs(output-held)) < .001
    finally:
        adapter.config.hardware_mode = 'dry-run'
        adapter.disconnect()


def test_worker_reports_geometric_collision_without_throwing_on_old_hold(tmp_path, monkeypatch):
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    try:
        h._prime(adapter, clock, worker, feedback, target)
        h._tick(adapter, clock, worker, feedback, target)
        layer = adapter._safety_controller.guard_layer
        layer.surface = replace(layer.surface, height=10.)
        layer.surfaces = h._helpers.h.safety.SurfaceConstraints((layer.surface,))
        worker.finish(started=clock[0], measured=feedback)
        assert worker.result['live_decision'].hard_limit_triggered
        previous = adapter._last_state.copy()
        np.testing.assert_array_equal(h._tick(adapter, clock, worker, feedback, target), previous)
        assert adapter.last_async_response == 'async_collision_hold'
        # A stopped worker or disconnected feedback is still a separate fault.
        clock[0] += .101
        with pytest.raises(h._helpers.h.gdk_adapter.SafetyDispatchExpiredError, match='Feedback'):
            adapter._select_async_hold(clock[0])
    finally:
        adapter.disconnect()


def test_collision_hold_telemetry_uses_each_latest_live_pose_distance(tmp_path, monkeypatch):
    """A held control certificate must not freeze the measured collision readout."""
    adapter, clock, worker, feedback, target = h._setup(tmp_path, monkeypatch)
    try:
        h._prime(adapter, clock, worker, feedback, target)
        h._tick(adapter, clock, worker, feedback, target)

        worker.finish(started=clock[0], measured=feedback)
        worker.result["live_decision"] = replace(
            worker.result["live_decision"],
            hard_limit_triggered=True,
            soft_limit_triggered=True,
            limiting_surface="work_surface",
            distance_to_surface=0.003,
            work_surface_distance=0.003,
            work_surface_clearance=0.0,
        )
        adapter._next_control_telemetry_monotonic = 0.0
        h._tick(adapter, clock, worker, feedback, target)
        assert adapter._latest_control_telemetry["safety"]["work_surface"]["distance_m"] == pytest.approx(0.003)

        worker.finish(started=clock[0], measured=feedback)
        worker.result["live_decision"] = replace(
            worker.result["live_decision"],
            hard_limit_triggered=True,
            soft_limit_triggered=True,
            limiting_surface="work_surface",
            distance_to_surface=0.001,
            work_surface_distance=0.001,
            work_surface_clearance=-0.002,
        )
        adapter._next_control_telemetry_monotonic = 0.0
        h._tick(adapter, clock, worker, feedback, target)
        safety = adapter._latest_control_telemetry["safety"]
        assert safety["collision_detected"] is True
        assert safety["work_surface"]["collision"] is True
        assert safety["work_surface"]["distance_m"] == pytest.approx(0.001)
    finally:
        adapter.disconnect()
