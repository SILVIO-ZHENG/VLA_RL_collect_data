"""Candidate braking tests: fake SDK only. Never connect to a robot."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("braking_test_helpers", Path(__file__).with_name("test_safety_layer.py"))
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
from nexus_deploy.robot.g2a.async_safety import (
    braking_sample, braking_clearance, hold_displacement_coefficients,
)
from nexus_deploy.robot.g2a.action_trace import ControlDiagnostics


def moving_adapter(tmp_path, monkeypatch):
    adapter, clock, worker = helper._manual_async_adapter(tmp_path, monkeypatch)
    adapter.config.certified_braking_enabled = True
    adapter._async_hold_coefficients = hold_displacement_coefficients(adapter._safety_controller.guard_layer)
    target = adapter._dry_state.copy()
    target[0] += .4
    completed = set()
    for _ in range(30):
        if adapter._async_pending is not None:
            request = adapter._async_pending["request_id"]
            if request not in completed:
                worker.finish(started=clock[0])
                completed.add(request)
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
    assert adapter._kinematic_limiter.velocity[0] > .1
    return adapter, clock, worker, target


def test_waiting_brakes_continuously_then_recovers_from_rest(tmp_path, monkeypatch):
    adapter, clock, worker, target = moving_adapter(tmp_path, monkeypatch)
    points = [adapter._last_state.copy()]
    previous_velocity = adapter._kinematic_limiter.velocity.copy()
    brake_seen = False
    try:
        # Stop completing geometry requests while feedback remains fresh.
        # Opposite new model goals must not alter the already certified stop.
        opposite = target.copy()
        opposite[0] = -.4
        for _ in range(80):
            clock[0] += .01
            worker.feedback = adapter._dry_state.copy(), clock[0]
            adopted = adapter.send_action(opposite)
            velocity = (adopted - points[-1]) / .01
            assert np.max(np.abs(velocity - previous_velocity)) <= .0201
            previous_velocity = velocity
            points.append(adopted.copy())
            brake_seen |= adapter._async_braking_active
            if brake_seen and not adapter._async_braking_active:
                break
        assert brake_seen
        assert not np.any(adapter._kinematic_limiter.velocity)
        assert np.all(np.diff(np.array(points)[:, 0]) >= -1e-7)
        held = adapter._last_state.copy()
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        np.testing.assert_array_equal(adapter.send_action(opposite), held)
        worker.finish(started=clock[0])
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        resumed = adapter.send_action(opposite)
        assert 0 < held[0] - resumed[0] <= .000201
    finally:
        adapter.disconnect()
    rows = [json.loads(line) for line in (tmp_path / 'paired.jsonl').read_text().splitlines()]
    brakes = [r for r in rows if r.get('collision_response') == 'async_checked_brake']
    assert brakes and all(r['gdk_command_accepted'] is None for r in brakes)
    assert all(r['certificate_age_ms'] < 40 for r in brakes)


@pytest.mark.parametrize("fault", ["stale_feedback", "unsafe_feedback", "anchor", "no_reserve"])
def test_braking_never_sends_without_valid_proof(tmp_path, monkeypatch, fault):
    adapter, clock, worker, target = moving_adapter(tmp_path, monkeypatch)
    try:
        adapter._async_steps = []
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        if fault == 'stale_feedback':
            worker.feedback = worker.feedback[0], clock[0] - .05
        elif fault == 'unsafe_feedback':
            worker.feedback = np.full(24, np.nan), clock[0]
        elif fault == 'anchor':
            adapter._last_state[0] += .001
        else:
            adapter._async_brake = None
        monkeypatch.setattr(adapter, '_dispatch_checked_action', lambda action: pytest.fail('Uncertified send'))
        with pytest.raises(helper.gdk_adapter.SafetyDispatchExpiredError):
            adapter._send_reserved_brake(target, {}, clock[0])
    finally:
        adapter.disconnect()


def test_braking_envelope_bounds_continuous_feedback_to_stop_paths(tmp_path, monkeypatch):
    adapter, clock, worker, target = moving_adapter(tmp_path, monkeypatch)
    try:
        reserve = adapter._async_brake['reserve']
        layer = adapter._safety_controller.guard_layer
        rng = np.random.default_rng(72)
        for _ in range(100):
            measured = reserve['anchor'].copy()
            measured[:7] += rng.uniform(-.003, .003, 7)
            clearance = braking_clearance(reserve, measured, adapter._async_hold_coefficients)
            assert clearance > 0
            k = int(rng.integers(0, reserve['steps'] + 1))
            point, _ = braking_sample(reserve, k)
            alpha = rng.random()
            interpolated = measured * (1 - alpha) + point * alpha
            z = layer._evaluate(interpolated).minimum_z.z
            assert z >= reserve['plane_limit'] + clearance - 1e-6
    finally:
        adapter.disconnect()


def test_candidate_cannot_be_enabled_for_physical_execution(tmp_path):
    with pytest.raises(ValueError, match='read-only/dry-run only'):
        helper.gdk_adapter.G2AAdapter(helper.gdk_adapter.AdapterConfig(
            hardware_mode='execute', async_safety_enabled=True, certified_braking_enabled=True,
            servo_control_period_s=.01, safety_check_fps=50,
            safety_config_path=tmp_path/'unused.json'))


def test_diagnostics_copies_values_and_finishes(tmp_path):
    recorder = ControlDiagnostics(tmp_path/'control.jsonl')
    value = np.array([1., 2.])
    recorder.write(dict(value=value))
    value[:] = 9
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path/'control.jsonl').read_text().splitlines()]
    assert rows[0]['value'] == [1., 2.]
    assert rows[-1] == dict(record_type='trace_end', dropped=0)
    assert not recorder.thread.is_alive()


def test_diagnostics_refuses_overwriting_existing_trace(tmp_path):
    path = tmp_path/'control.jsonl'
    path.write_text('preserve\n')
    recorder = ControlDiagnostics(path)
    recorder.close()
    assert recorder.error
    assert path.read_text() == 'preserve\n'


def test_diagnostics_file_size_limit_does_not_raise_in_control(tmp_path):
    recorder = ControlDiagnostics(tmp_path/'bounded.jsonl', max_bytes=64)
    recorder.write(dict(too_large='x' * 100))
    recorder.close()
    row = json.loads((tmp_path/'bounded.jsonl').read_text())
    assert row['truncated'] and row['reason'] == 'diagnostic_file_size_limit'
    assert recorder.error and not recorder.thread.is_alive()


def test_normal_async_motion_keeps_filter_history_but_hold_reanchors(tmp_path, monkeypatch):
    adapter, clock, worker = helper._manual_async_adapter(tmp_path, monkeypatch)
    try:
        target = adapter._dry_state.copy()
        target[0] += .4
        adapter.send_action(target)
        assert adapter.last_action_intervened  # Pending startup, not motion.
        worker.finish()
        for _ in range(2):
            clock[0] += .01
            adapter.send_action(target)
            assert not adapter.last_action_intervened
        clock[0] += .01
        adapter.send_action(target)
        assert adapter.last_action_intervened  # No fresh motion window.
    finally:
        adapter.disconnect()


@pytest.mark.parametrize('field', ['coefficients', 'anchor_z_min', 'plane_limit', 'anchor', 'end'])
def test_nonfinite_brake_proof_is_rejected(tmp_path, monkeypatch, field):
    adapter, clock, worker, target = moving_adapter(tmp_path, monkeypatch)
    try:
        reserve = dict(adapter._async_brake['reserve'])
        coefficients = adapter._async_hold_coefficients.copy()
        if field == 'coefficients':
            coefficients[0] = np.nan
        elif field in ('anchor', 'end'):
            reserve[field] = np.full(24, np.nan)
        else:
            reserve[field] = np.nan
        assert braking_clearance(reserve, adapter._dry_state, coefficients) == -float('inf')
    finally:
        adapter.disconnect()


@pytest.mark.parametrize('fault', ['near_plane', 'no_reserves', 'over_budget'])
def test_motion_needs_accepted_within_budget_stop_reserve(tmp_path, monkeypatch, fault):
    adapter, clock, worker = helper._manual_async_adapter(tmp_path, monkeypatch)
    adapter.config.certified_braking_enabled = True
    adapter._async_hold_coefficients = hold_displacement_coefficients(adapter._safety_controller.guard_layer)
    try:
        target = adapter._dry_state.copy()
        target[0] += .4
        initial = adapter.send_action(target)
        worker.finish(check_ms=21 if fault == 'over_budget' else 5)
        if fault == 'near_plane':
            from nexus_deploy.robot.g2a.async_safety import make_braking_reserve
            task = worker.tasks[-1]
            specification = task['braking']
            layer = adapter._safety_controller.guard_layer
            # Force the lower bound below the tabletop: no candidate movement.
            monkeypatch.setattr(layer, 'global_surface_clearance', lambda q: -100.)
            assert make_braking_reserve(layer, task['targets'][0], specification['target_velocities'][0],
                                       specification, initial, adapter._async_hold_coefficients) is None
            worker.result['target_brakes'] = [None, None]
        elif fault == 'no_reserves':
            worker.result['target_brakes'] = None
        clock[0] += .01
        worker.feedback = initial.copy(), clock[0]
        np.testing.assert_array_equal(adapter.send_action(target), initial)
        assert adapter._async_brake is None
        assert not np.any(adapter._kinematic_limiter.velocity)
    finally:
        adapter.disconnect()
