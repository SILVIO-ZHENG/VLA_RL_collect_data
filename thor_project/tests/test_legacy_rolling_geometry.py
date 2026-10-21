"""No hardware: restored sampled geometry and feedback-bound reuse."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('legacy_rolling_helpers',Path(__file__).with_name('test_safety_layer.py'))
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
from nexus_deploy.robot.g2a.async_safety import (
    check_window,check_legacy_rolling_window,hold_displacement_coefficients,sampled_feedback_displacement)
from nexus_deploy.robot.g2a.safety_layer import SurfaceConstraints


def setup(tmp_path,monkeypatch):
    adapter,clock,worker=h._manual_async_adapter(tmp_path,monkeypatch)
    adapter.config.rolling_safety_enabled=True
    adapter.config.rolling_legacy_geometry=True
    adapter.config.certified_braking_enabled=False
    adapter._async_hold_coefficients=hold_displacement_coefficients(adapter._safety_controller.guard_layer)
    worker.feedback=adapter._dry_state.copy(),clock[0]
    return adapter,clock,worker


@pytest.mark.parametrize('height',[-2.,.5,2.])
def test_decisions_match_legacy_sampling(tmp_path,monkeypatch,height):
    adapter,clock,worker=setup(tmp_path,monkeypatch)
    try:
        layer=adapter._safety_controller.guard_layer
        surface=replace(layer.surface,height=height)
        layer.surfaces=SurfaceConstraints((surface,));layer.surface=surface
        q=adapter._dry_state.copy();targets=[q.copy(),q.copy()];targets[-1][0]+=.01
        try:
            old=check_window(layer,q,q,targets,targets[-1],include_proofs=True,fallback_anchor=q)
        except RuntimeError:
            with pytest.raises(RuntimeError):check_legacy_rolling_window(layer,q,q,targets,targets[-1],q)
        else:
            new=check_legacy_rolling_window(layer,q,q,targets,targets[-1],q)
            assert new['decision'].hard_limit_triggered==old[1].hard_limit_triggered
            np.testing.assert_array_equal(new['decision'].safe_action,old[1].safe_action)
            assert new['target_brakes'] is None
    finally:adapter.disconnect()


def test_low_pose_outside_finite_table_is_not_blocked_by_infinite_plane(tmp_path,monkeypatch):
    adapter,clock,worker=setup(tmp_path,monkeypatch)
    try:
        layer=adapter._safety_controller.guard_layer
        surface=replace(layer.surface,height=10.,xy_bounds=(100.,101.,100.,101.))
        layer.surfaces=SurfaceConstraints((surface,));layer.surface=surface
        q=adapter._dry_state.copy();target=q.copy();target[0]+=.01
        result=check_legacy_rolling_window(layer,q,q,[target],target,q)
        assert not result['decision'].hard_limit_triggered
        assert sampled_feedback_displacement(result['sampled_feedback'],q,adapter._async_hold_coefficients)==0
    finally:adapter.disconnect()


def test_finite_table_margin_accounts_for_crossing_xy_edge(tmp_path,monkeypatch):
    adapter,clock,worker=setup(tmp_path,monkeypatch)
    try:
        proof=dict(reference=adapter._dry_state.copy(),margin=.001)
        q=proof['reference'].copy();q[0]+=.1
        assert not np.isfinite(sampled_feedback_displacement(proof,q,adapter._async_hold_coefficients))
        assert not np.isfinite(sampled_feedback_displacement(proof,np.full(24,np.nan),adapter._async_hold_coefficients))
    finally:adapter.disconnect()


def test_legacy_predicate_runs_through_rolling_buffer_without_stop_gate(tmp_path,monkeypatch):
    adapter,clock,worker=setup(tmp_path,monkeypatch)
    target=adapter._dry_state.copy();target[0]+=.2
    try:
        completed=set()
        for tick in range(100):
            clock[0]+=.01;worker.feedback=adapter._dry_state.copy(),clock[0]
            if tick%2==0 and adapter._async_pending is not None:
                request=adapter._async_pending['request_id']
                if request not in completed:worker.finish(started=clock[0]);completed.add(request)
            adapter.send_action(target)
            if tick>4:assert not adapter.last_action_intervened
        assert all(task['rolling_legacy_geometry'] and 'braking' not in task for task in worker.tasks)
        assert adapter._async_brake is None
    finally:adapter.disconnect()


def test_segmented_execution_still_requires_a_valid_safety_profile(tmp_path):
    # The legacy candidate-only prohibition is retired after read-only
    # acceptance. Loading/validating the safety profile remains mandatory;
    # constructing an adapter never acquires GDK or prepares execution.
    with pytest.raises(FileNotFoundError):
        h.gdk_adapter.G2AAdapter(h.gdk_adapter.AdapterConfig(
            rolling_safety_enabled=True,rolling_legacy_geometry=True,hardware_mode='execute',
            async_safety_enabled=True,safety_check_fps=50,servo_control_period_s=.01,
            safety_config_path=tmp_path/'unused.json'))


def test_hold_uses_its_own_sampled_certificate_not_rejected_motion(tmp_path,monkeypatch):
    adapter,clock,worker=setup(tmp_path,monkeypatch)
    try:
        target=adapter._dry_state.copy();target[0]+=.1
        adapter.send_action(target);worker.finish(started=clock[0])
        clock[0]+=.01;worker.feedback=adapter._dry_state.copy(),clock[0]
        adapter.send_action(target)
        held=adapter._async_hold
        checked_at=held['result']['started']
        # A below-plane global Z alone must not override a valid finite-table
        # certificate (e.g. an inactive collision frame outside the tabletop).
        held['hold_proof']=replace(held['hold_proof'],path_z_min=-100.)
        clock[0]+=.05;worker.feedback=adapter._dry_state.copy(),clock[0]
        selected=adapter._select_async_hold(clock[0])
        assert 'hold_checked_at' not in selected
        assert selected['result']['started']==checked_at
        held['hold_proof']=replace(held['hold_proof'],hard_limit_triggered=True)
        assert adapter._select_async_hold(clock[0])['historical_repeat']
    finally:adapter.disconnect()
