"""Rolling controller tests use fake time/SDK, never robot hardware."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('rolling_helpers', Path(__file__).with_name('test_safety_layer.py'))
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
from nexus_deploy.robot.g2a.async_safety import hold_displacement_coefficients, trajectory_clearance


@pytest.mark.parametrize('rolling,expected', [(False,100.008),(True,100.004)])
def test_preparation_lead_does_not_change_dispatch_period(rolling, expected):
    from nexus_deploy.robot.g2a.async_safety import servo_preparation_time
    assert servo_preparation_time(100., .01, rolling=rolling) == pytest.approx(expected)
    # Preparation can start earlier; the actual dispatch slot remains 100.010.
    assert expected < 100.01


def setup(tmp_path, monkeypatch):
    adapter, clock, worker = h._manual_async_adapter(tmp_path, monkeypatch)
    adapter.config.certified_braking_enabled = True
    adapter.config.rolling_safety_enabled = True
    adapter._async_hold_coefficients = hold_displacement_coefficients(adapter._safety_controller.guard_layer)
    worker.feedback = adapter._dry_state.copy(), clock[0]
    target = adapter._dry_state.copy()
    target[0] += .4
    return adapter, clock, worker, target


def test_rolling_covers_100hz_without_holds_under_50hz_checks(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    points, speeds, buffers = [], [], []
    completed = set()
    try:
        for tick in range(180):
            clock[0] += .01
            worker.feedback = adapter._dry_state.copy(), clock[0]
            if tick % 2 == 0 and adapter._async_pending is not None:
                request = adapter._async_pending['request_id']
                if request not in completed:
                    worker.finish(started=clock[0], check_ms=12)
                    completed.add(request)
            adopted = adapter.send_action(target)
            if tick > 3:
                assert not adapter.last_action_intervened
                assert not adapter._async_braking_active
            points.append(adopted.copy())
            speeds.append(adapter._kinematic_limiter.velocity.copy())
            buffers.append(len(adapter._async_steps))
        assert len(completed) >= 85
        assert min(buffers[4:]) >= 5
        assert max(buffers) <= 10
        # Each sent increment is the certified limiter velocity, including splices.
        np.testing.assert_allclose(np.diff(points, axis=0)/.01, speeds[1:], atol=5e-6)
        assert np.max(np.abs(np.diff(speeds, axis=0))) <= .02001
    finally:
        adapter.disconnect()


def test_one_late_check_uses_buffer_then_persistent_delay_brakes(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        for tick in range(7):
            clock[0] += .01
            worker.feedback = adapter._dry_state.copy(), clock[0]
            adapter.send_action(target)
            assert not adapter.last_action_intervened
        # Geometry has not completed for 70 ms, but every point is renewed
        # against fresh measured feedback and the immutable full-path bound.
        assert len(adapter._async_steps) == 3
        brake_seen = False
        for _ in range(40):
            clock[0] += .01
            worker.feedback = adapter._dry_state.copy(), clock[0]
            adapter.send_action(target)
            brake_seen |= adapter._async_braking_active
        assert brake_seen
        assert not np.any(adapter._kinematic_limiter.velocity)
    finally:
        adapter.disconnect()


def test_certificate_bounds_intermediate_joint_space_points(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        proof = worker.result['trajectory_box']
        coefficients = adapter._async_hold_coefficients
        clearance = trajectory_clearance(proof, adapter._dry_state, coefficients)
        assert clearance > 0
        rng = np.random.default_rng(42)
        for _ in range(100):
            point = proof['anchor'] + rng.uniform(-1, 1, 24)*proof['radius']
            z = adapter._safety_controller.guard_layer._evaluate(point).minimum_z.z
            assert z >= proof['plane_limit']+clearance-1e-6
        assert trajectory_clearance(dict(proof, radius=np.full(24, np.nan)), adapter._dry_state, coefficients) == -float('inf')
    finally:
        adapter.disconnect()


def test_stale_feedback_cannot_renew_trajectory(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        result = worker.result
        clock[0] += .05
        assert adapter._renew_rolling_result(result, clock[0]) is None
    finally:
        adapter.disconnect()


def test_hold_cache_does_not_pair_new_feedback_with_old_clearance(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
        clock[0] += .01
        measured = adapter._dry_state.copy()
        measured[0] += .03
        worker.feedback = measured, clock[0]
        adapter.send_action(target)
        held = adapter._async_hold
        result = held['result']
        assert held['hold_proof'].path_z_min <= (result['trajectory_box']['plane_limit']
                                                + result['trajectory_bound_clearance'])
        np.testing.assert_array_equal(result['measured'], measured)
    finally:
        adapter.disconnect()


def test_rolling_requires_braking():
    with pytest.raises(ValueError, match='requires certified braking'):
        h.gdk_adapter.G2AAdapter(h.gdk_adapter.AdapterConfig(rolling_safety_enabled=True))


@pytest.mark.parametrize('fault', ['generation', 'anchor_identity', 'velocity', 'collision'])
def test_bad_splice_keeps_existing_certified_trajectory(tmp_path, monkeypatch, fault):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
        worker.finish(started=clock[0])
        result = worker.result
        original = list(adapter._async_steps)
        if fault == 'generation':
            result['task'] = dict(result['task'], generation=-1)
        elif fault == 'anchor_identity':
            result['task'] = dict(result['task'], anchor_identity=('obsolete',))
        elif fault == 'velocity':
            result['task'] = dict(result['task'], anchor_velocity=np.ones(24))
        else:
            from dataclasses import replace
            result['decision'] = replace(result['decision'], hard_limit_triggered=True)
        adapter._splice_rolling_result(result, clock[0])
        assert len(adapter._async_steps) == len(original)
        assert all(a is b for a,b in zip(adapter._async_steps, original))
    finally:
        adapter.disconnect()


def test_slow_check_is_usable_only_with_surviving_anchor_and_fresh_bound(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
        worker.finish(started=clock[0], check_ms=32)
        result = worker.result
        request = result['task']['request_id']
        for _ in range(3):
            clock[0] += .01
            worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter._splice_rolling_result(result, clock[0])
        assert adapter._async_steps[-1]['result']['task']['request_id'] == request
        assert adapter._async_steps[-1]['result']['started'] == worker.feedback[1]
    finally:
        adapter.disconnect()


def test_result_at_current_anchor_replaces_suffix_without_legacy_deferral(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
        pending = adapter._async_pending
        request = pending['request_id']
        # Simulate a late result, exactly as its certified handoff is consumed.
        for _ in range(6):
            clock[0] += .01
            worker.feedback = adapter._dry_state.copy(), clock[0]
            adapter.send_action(target)
        assert adapter._async_steps
        np.testing.assert_array_equal(adapter._last_state, pending['anchor'])
        worker.finish(started=clock[0]-.06, check_ms=55)
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
        assert not adapter.last_action_intervened
        assert adapter._async_ready_result is None
        assert adapter._async_last_waypoint[1] == request
    finally:
        adapter.disconnect()


def test_candidate_cannot_escape_box_even_if_queue_is_corrupted(tmp_path, monkeypatch):
    adapter, clock, worker, target = setup(tmp_path, monkeypatch)
    try:
        adapter.send_action(target)
        worker.finish(started=clock[0])
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        adapter.send_action(target)
        adapter._async_steps[0]['action'] = np.ones(24)
        clock[0] += .01
        worker.feedback = adapter._dry_state.copy(), clock[0]
        monkeypatch.setattr(adapter, '_dispatch_checked_action', lambda q: pytest.fail('unchecked dispatch'))
        with pytest.raises(h.gdk_adapter.SafetyDispatchExpiredError, match='escaped certified'):
            adapter.send_action(target)
    finally:
        adapter.disconnect()
