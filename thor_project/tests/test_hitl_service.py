"""FlowPRO startup must fence DDS until an accepted G2 HOLD is available."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus_deploy.robot.g2a.hitl.g2_runtime.service import (
    FlowPROServiceComposition,
    build_parser,
    require_approved_safety_calibration,
)
from nexus_deploy.robot.g2a.hitl.g2_runtime.supervisor import SupervisorMode


def test_service_preserves_selected_inference_mode():
    args = build_parser().parse_args(
        ["--host=thor", "--task=sort", "--execute", "--inference-mode=rtc", "--execution-fps=25"]
    )
    assert args.inference_mode == "rtc"
    assert args.fps == 25


def test_dds_starts_only_after_accepted_startup_hold():
    starts = []
    pause = SimpleNamespace(acknowledged=False)
    pause.wait = lambda _timeout: pause.acknowledged
    adapter = SimpleNamespace(last_delivery_evidence=None)
    composition = object.__new__(FlowPROServiceComposition)
    composition.runner = SimpleNamespace(_pause_active_event=pause, adapter=adapter)
    composition.args = SimpleNamespace(hardware_mode="execute")
    composition.dds_thread = SimpleNamespace(start=lambda: starts.append("dds"))
    composition.sealer_thread = SimpleNamespace(start=lambda: starts.append("sealer"))

    with pytest.raises(TimeoutError, match="startup measured HOLD"):
        composition.start()
    assert starts == []

    pause.acknowledged = True
    with pytest.raises(RuntimeError, match="accepted GDK command"):
        composition.start()
    assert starts == []

    adapter.last_delivery_evidence = SimpleNamespace(command_accepted=True)
    composition.start()
    assert starts == ["dds", "sealer"]


def test_tablet_episode_is_bound_in_hold_before_dds_starts():
    starts = []
    composition = object.__new__(FlowPROServiceComposition)
    composition.runner = SimpleNamespace(
        _pause_active_event=SimpleNamespace(wait=lambda _timeout: True),
        adapter=SimpleNamespace(last_delivery_evidence=SimpleNamespace(command_accepted=True)),
    )
    composition.args = SimpleNamespace(
        hardware_mode="execute", initial_vendor_record_name="L-9.2_20261016120000"
    )
    composition.application = SimpleNamespace(
        bind_initial_recording_in_hold=lambda name: starts.append(("hold", name))
    )
    composition.dds_thread = SimpleNamespace(start=lambda: starts.append("dds"))
    composition.sealer_thread = SimpleNamespace(start=lambda: starts.append("sealer"))

    composition.start()
    assert starts == [("hold", "L-9.2_20261016120000"), "dds", "sealer"]


def test_later_tablet_recording_binds_in_hold_without_new_deployment():
    composition = object.__new__(FlowPROServiceComposition)
    bound = []
    app = SimpleNamespace(
        coordinator=SimpleNamespace(
            writer=None,
            list_active_recordings=lambda _root: [Path("/data/record/second")],
        ),
        session=SimpleNamespace(supervisor=SimpleNamespace(mode=SupervisorMode.DISABLED)),
        vendor_record_root=Path("/data/record"),
        bind_initial_recording_in_hold=lambda name: bound.append(name),
    )
    composition.application = app
    composition._bind_next_tablet_recording_in_hold()
    assert bound == ["second"]
    app.coordinator.writer = object()
    composition._bind_next_tablet_recording_in_hold()
    assert bound == ["second"]


def test_rollback_preflight_rejects_unapproved_and_wrong_effector(tmp_path: Path):
    profile = tmp_path / "safety.json"
    profile.write_text('{"calibrated": false, "execute_approved": true}')
    with pytest.raises(ValueError, match="approved calibrated"):
        require_approved_safety_calibration(
            profile, expected_robot_id="g2a", expected_effector_profile="dahuan"
        )
    profile.write_text(
        '{"calibrated": true, "execute_approved": true, '
        '"execution_approval": {"status": "approved"}, '
        '"robot_model": {"effector_profile": "omnipicker_legacy"}}'
    )
    with pytest.raises(ValueError, match="wrong effector geometry"):
        require_approved_safety_calibration(
            profile, expected_robot_id="g2a", expected_effector_profile="dahuan"
        )
