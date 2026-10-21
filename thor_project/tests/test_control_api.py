from __future__ import annotations

import base64
import json
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from nexus_deploy.deployment import DeployError
from nexus_deploy.control.errors import TaskError
from nexus_deploy.control_api import (
    ControlPlane,
    DeploymentJob,
    NexusHTTPServer,
    TelemetryTCPServer,
    ValidationError,
    _handler_factory,
    _telemetry_stream_handler,
    build_parser,
)


def select_ready_scheme(plane: ControlPlane, job_id: str) -> None:
    """Legacy command tests begin after the operator has saved a scheme."""
    plane._jobs[job_id].task_session["runtime_configured"] = True
    plane._workstation["active_deployment_id"] = job_id


def test_http_server_silences_disconnects_but_reports_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reported: list[BaseException | None] = []
    monkeypatch.setattr(
        ThreadingHTTPServer,
        "handle_error",
        lambda _self, _request, _address: reported.append(sys.exc_info()[1]),
    )
    server = object.__new__(NexusHTTPServer)

    for disconnect in (ConnectionResetError("peer reset"), BrokenPipeError("peer closed")):
        try:
            raise disconnect
        except type(disconnect):
            server.handle_error(None, ("127.0.0.1", 1))
    assert reported == []

    try:
        raise RuntimeError("real server failure")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert len(reported) == 1
    assert isinstance(reported[0], RuntimeError)


@pytest.fixture()
def control_plane(tmp_path: Path) -> ControlPlane:
    g2a_source = tmp_path / "robot_sources" / "g2a"
    legacy_source = tmp_path / "robot_sources" / "legacy"
    g2a_source.mkdir(parents=True)
    legacy_source.mkdir(parents=True)

    (g2a_source / "SAFETY_CONFIG_TEMPLATE.json").write_text(
        json.dumps({"version": 2, "calibrated": False}), encoding="utf-8"
    )
    (g2a_source / "workspace_safety.calibrated.json").write_text(
        json.dumps({"version": 2, "calibrated": True}), encoding="utf-8"
    )
    (g2a_source / "workspace_safety.read_only.json").write_text(
        json.dumps(
            {
                "version": 2,
                "display_name": "Read-only table safety",
                "calibrated": False,
                "read_only_ready": True,
                "execute_approved": False,
            }
        ),
        encoding="utf-8",
    )
    (legacy_source / "SAFETY_CONFIG_TEMPLATE.json").write_text(
        json.dumps({"version": 2, "calibrated": False}), encoding="utf-8"
    )

    config_path = tmp_path / "deploy.toml"
    config_path.write_text(
        f"""
[runtime]
directory = "{tmp_path / 'runtime'}"
conda = "/opt/private/miniforge/bin/conda"
require_unique_model_gpus = true

[server]
bind_host = "0.0.0.0"
advertise_host = "10.42.0.99"

[models.pi05]
display_name = "PI0.5"
description = "RTC policy"
repo = "/srv/private/models/pi05"
conda_env = "private-lerobot"
module = "private.pi05.server"
port = 5557
cuda_visible_devices = "0"
startup_timeout_s = 30
supported_modes = ["sync", "async", "rtc"]
base_args = ["--host={{bind_host}}", "--port={{port}}"]
real_args = ["--checkpoint=/srv/private/checkpoints/pi05"]
mock_args = ["--mock"]

[models.lingbot]
display_name = "LingBot"
repo = "/srv/private/models/lingbot"
conda_env = "private-lingbot"
module = "private.lingbot.server"
port = 5558
cuda_visible_devices = "1"
startup_timeout_s = 30
supported_modes = ["sync", "async", "rtc"]
rtc_execution_horizon = 30
rtc_execute_enabled = false
base_args = ["--host={{bind_host}}", "--port={{port}}"]
real_args = ["--model-path=/srv/private/checkpoints/lingbot"]
mock_args = ["--mock"]

[ui_models.fastwam]
display_name = "FastWAM"
description = "16D policy model"
supported_modes = ["sync", "async", "rtc"]
tags = ["16D", "Absolute", "30 FPS"]
port = 5559
cuda_visible_devices = "auto"
gpu_min_free_memory_mb = 16000
startup_timeout_s = 600
default_effector_profile = "dahuan"
effector_profiles = ["dahuan", "omnipicker_legacy"]
compatible_robots = ["g2a"]

[ui_models.fastwam.effector_variants.dahuan]
normalization_source = "checkpoint_embedded"
normalization_label = "大寰 checkpoint quantiles"

[ui_models.fastwam.effector_variants.omnipicker_legacy]
normalization_source = "checkpoint_embedded"
normalization_label = "Legacy checkpoint quantiles"

[robots.g2a]
display_name = "G2A"
source_client = "{g2a_source}"
source_check = "manifest"
bundle_directory = "{tmp_path / 'bundles' / 'g2a'}"
ssh_target = "agi@10.42.0.55"
ssh_key = "/home/test/.ssh/id_ed25519_private"
host_key_alias = "private-g2a"
remote_directory = "/home/agi/deploy/g2a_private_client"
policy_source_host = "10.42.1.102"
compatible_models = ["pi05", "lingbot"]

[robots.legacy]
display_name = "Legacy Robot"
effectors_mode = "absent_fixed"
fixed_effector_state = [-0.7, -0.71]
source_client = "{legacy_source}"
source_check = "manifest"
bundle_directory = "{tmp_path / 'bundles' / 'legacy'}"
ssh_target = "legacy@10.42.0.56"
ssh_key = "/home/test/.ssh/id_ed25519_legacy"
host_key_alias = "private-legacy"
remote_directory = "/home/legacy/deploy/legacy_private_client"
compatible_models = ["pi05"]

[ui_robots.astribot_s1]
display_name = "Astribot S1"
description = "UI-only robot"
ui_only = true
adapter_status = "planned"
urdf_status = "vendor_asset_pending"
height_m = 1.7
arm_span_m = 1.94
payload_per_arm_kg = 5.0
endurance_hours = [4, 6]
arm_dof = 7
tags = ["Dual arm", "UI only"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return ControlPlane(config_path=config_path, enable_actions=False)


def valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "robot_id": "g2a",
        "robot_host": "192.168.10.55",
        "robot_ssh_port": 22,
        "model_id": "pi05",
        "task": "Pick up the package",
        "inference_mode": "async",
        "hardware_mode": "read-only",
        "human_in_loop": False,
        "execution_fps": 30,
        "servo_fps": 100,
        "chunk_smoothing_mode": "adaptive-fixed-time",
        "prediction_horizon": 50,
        "quality_horizon": 50,
        "safety": {
            "workspace_safety_enabled": False,
            "safety_profile_id": "",
        },
    }
    payload.update(overrides)
    return payload


def test_alignment_entry_is_disabled_by_default_and_opt_in(control_plane: ControlPlane) -> None:
    assert control_plane.bootstrap()["tools"]["alignment_url"] is None
    enabled = ControlPlane(
        config_path=control_plane.config_path,
        alignment_url="http://127.0.0.1:8775/alignment",
    )
    assert enabled.bootstrap()["tools"]["alignment_url"] == "http://127.0.0.1:8775/alignment"


def test_platform_loads_only_eligible_base_link_surface_calibration(
    control_plane: ControlPlane, tmp_path: Path
) -> None:
    root = tmp_path / "alignment-surfaces"
    profile = root / "standard_table"
    profile.mkdir(parents=True)
    created_at = time.time()
    (profile / "active.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurement_id": "standard_table-test",
                "profile_id": "standard_table",
                "created_at": created_at,
                "base_frame": "base_link",
                "surface": {
                    "eligible": True,
                    "height_m": 0.712,
                    "tilt_deg": 0.4,
                    "spread": {"height_mm": 0.8},
                },
                "robot": {"id": "g2-102", "host": "192.168.10.55"},
                "robot_model": {"id": "G2_dahuan_safety", "sha256": "abc"},
                "camera_calibration_version": "camera-v1",
            }
        ),
        encoding="utf-8",
    )
    plane = ControlPlane(
        config_path=control_plane.config_path,
        alignment_surfaces_directory=root,
    )

    result = plane.bootstrap()["tools"]["alignment_surface"]

    assert plane.deployment.alignment_surfaces_root == root.resolve()
    assert result["available"] is True
    assert result["fresh"] is True
    assert result["height_m"] == pytest.approx(0.712)
    assert result["measurement_id"] == "standard_table-test"


def test_alignment_history_generates_selectable_dated_robot_safety_profile(
    control_plane: ControlPlane,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = control_plane.deployment.robots["g2a"]
    template_name = "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json"
    (robot.source_client / template_name).write_text(
        json.dumps(
            {
                "version": 2,
                "calibrated": False,
                "read_only_ready": True,
                "execute_approved": False,
                "base_frame": "base_link",
                "robot_model": {
                    "type": "urdf",
                    "path": "assets/G2_dahuan_safety.urdf",
                    "sha256": "test-urdf-sha",
                },
                "surfaces": [
                    {
                        "name": "work_surface",
                        "frame": "base_link",
                        "height": 0.75,
                        "hard_margin": 0.02,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    robot.manifest.write_text(f"{template_name}\n", encoding="utf-8")
    measurement_path = (
        control_plane.alignment_surfaces_directory
        / "standard_table/versions/20260910T103522+0800.json"
    )
    measurement_path.parent.mkdir(parents=True)
    measurement_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurement_id": "standard_table-20260910T103522+0800",
                "profile_id": "standard_table",
                "created_at": 1789007722.0,
                "base_frame": "base_link",
                "surface": {
                    "eligible": True,
                    "height_m": 0.78342,
                    "pose": {"x_m": 0.81, "y_m": -0.22, "z_m": 0.78342},
                    "normal": [0.004, -0.003, 0.9999875],
                },
                "robot": {"id": "", "host": "10.42.0.55"},
                "camera_calibration_version": "camera-v1",
                "robot_model": {
                    "id": "G2_dahuan_safety",
                    "sha256": "test-urdf-sha",
                },
            }
        ),
        encoding="utf-8",
    )

    profiles = control_plane._safety_profiles(robot)
    generated = next(item for item in profiles if item["generated"])

    assert generated["id"].startswith("G2_dahuan_standard_table_20260910T")
    assert generated["height_m"] == pytest.approx(0.78342)
    assert generated["measurement_id"] == "standard_table-20260910T103522+0800"
    assert generated["read_only_ready"] is True
    assert generated["execute_approved"] is True
    assert generated["deployable"] is True

    payload = valid_payload(
        safety={
            "workspace_safety_enabled": True,
            "safety_profile_id": generated["id"],
        }
    )
    config = control_plane.validate(payload)["normalized"]
    command, _session_directory = control_plane.deployment.robot_session_command(
        robot,
        control_plane.deployment.models["pi05"],
        "f" * 32,
        config,
    )
    assert (
        f"--safety-config=/home/agi/deploy/g2a_private_client/robot_client/{generated['id']}"
        in command
    )
    remote_commands: list[str] = []
    monkeypatch.setattr(
        control_plane.deployment,
        "run_ssh",
        lambda command, **_kwargs: remote_commands.append(command),
    )
    control_plane.deployment.deploy_generated_safety_profile(
        robot, generated["id"], "f" * 32
    )
    assert len(remote_commands) == 1
    assert "base64 -d" in remote_commands[0]
    assert f"robot_client/{generated['id']}" in remote_commands[0]
    assert "chmod 0600" in remote_commands[0]

    generated_path = (
        control_plane.deployment.generated_safety_profiles_root
        / robot.name
        / generated["id"]
    )
    tampered = json.loads(generated_path.read_text(encoding="utf-8"))
    tampered["surfaces"][0]["height"] += 0.05
    generated_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DeployError, match="immutable source"):
        control_plane.deployment.workspace_safety_profile(robot, generated["id"])


def grant_ready_robot_probe(control_plane: ControlPlane) -> None:
    control_plane._probe_leases[("g2a", "192.168.10.55", 22)] = {
        "robot_id": "g2a",
        "host": "192.168.10.55",
        "port": 22,
        "reachable": True,
        "ssh_authenticated": True,
        "checked_at": time.time(),
        "client": {"status": "idle", "valid": True},
        "observation": {"ready": True, "status": "ready"},
    }


def error_fields(result: dict[str, Any]) -> set[str]:
    return {str(item["field"]) for item in result["errors"]}


def walk(value: Any) -> Iterator[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from walk(item)


def control_telemetry_payload(sequence: int = 1) -> dict[str, Any]:
    vector = [float(index) / 100 for index in range(24)]
    return {
        "sequence": sequence,
        "captured_at": time.time(),
        "hardware_mode": "read-only",
        "measured_state": vector,
        "model_candidate": vector,
        "controller_candidate": vector,
        "adopted_action": vector,
        "safety": {
            "enabled": False,
            "monitoring_enabled": True,
            "enforcement_enabled": False,
            "decision_available": True,
            "collision_detected": True,
            "proximity_warning": False,
            "candidate_collision_detected": False,
            "candidate_proximity_warning": True,
            "intervention_active": False,
            "reason": "realtime_guard_hold_pending_projection",
            "response": "monitor_only_allow_candidate",
            "action_scale": 0.0,
            "distance_to_surface": 0.0005,
            "surface_height": 0.72759,
            "hard_margin": 0.001,
            "warning_margin": 0.003,
            "work_surface": {
                "name": "table",
                "distance_m": 0.0005,
                "clearance_m": -0.0005,
                "hard_margin_m": 0.001,
                "warning_margin_m": 0.003,
                "collision": True,
                "warning": True,
                "limiting_point": "left_gripper:vertex_7",
            },
            "inter_arm": {
                "minimum_separation_m": 0.021,
                "clearance_m": 0.013,
                "hard_margin_m": 0.008,
                "warning_margin_m": 0.02,
                "collision": False,
                "warning": False,
                "left_geometry": "left_gripper_inner",
                "right_geometry": "right_gripper_inner",
            },
            "candidate_inter_arm": {
                "minimum_separation_m": 0.006,
                "hard_margin_m": 0.008,
                "warning_margin_m": 0.02,
                "checked_waypoints": 10,
                "collision": True,
                "warning": True,
                "left_geometry": "left_gripper_base",
                "right_geometry": "right_gripper_base",
            },
            "guard_latency_ms": 2.4,
            "guard_budget_ms": 8.0,
            "deadline_misses": 0,
        },
    }


def occupied_robot_probe(*, mode: str = "read-only", running: bool = True) -> dict[str, Any]:
    session_id = "a" * 32
    return {
        "robot_id": "g2a",
        "host": "192.168.10.55",
        "port": 22,
        "reachable": True,
        "ssh_authenticated": True,
        "checked_at": time.time(),
        "identity": {"match": True},
        "client": {
            "status": "running" if running else "idle",
            "valid": True,
            "running": running,
            "processes": (
                [
                    {
                        "pid": 4321,
                        "session_id": session_id,
                        "hardware_mode": mode,
                        "managed": True,
                    }
                ]
                if running
                else []
            ),
        },
        "observation": {"ready": True, "status": "ready"},
    }


def test_occupied_session_cleanup_requires_two_bound_confirmations(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    live = {"running": True}
    stopped: list[tuple[str, int]] = []

    monkeypatch.setattr(
        control_plane,
        "probe_robot",
        lambda _payload, _token: occupied_robot_probe(running=live["running"]),
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "select_robot_endpoint",
        lambda *_args: object(),
    )

    def force_stop(_robot: object, session_id: str, pid: int, **_kwargs: Any) -> None:
        stopped.append((session_id, pid))
        live["running"] = False

    monkeypatch.setattr(control_plane.deployment, "force_stop_robot_session", force_stop)
    target = {
        "robot_id": "g2a",
        "robot_host": "192.168.10.55",
        "robot_ssh_port": 22,
        "session_id": "a" * 32,
        "pid": 4321,
    }

    prepared = control_plane.prepare_robot_session_cleanup(target, control_plane.csrf_token)
    assert prepared["confirmation_phrase"] == "强制关闭会话 aaaaaaaa"
    assert prepared["target"]["hardware_mode"] == "read-only"

    result = control_plane.cleanup_robot_session(
        {
            **target,
            "challenge_token": prepared["challenge_token"],
            "confirmation_phrase": prepared["confirmation_phrase"],
        },
        control_plane.csrf_token,
    )

    assert stopped == [("a" * 32, 4321)]
    assert result["stopped"] is True
    assert result["probe"]["client"]["status"] == "idle"


def test_occupied_session_cleanup_rejects_bad_phrase_and_replay(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    monkeypatch.setattr(
        control_plane,
        "probe_robot",
        lambda _payload, _token: occupied_robot_probe(),
    )
    target = {
        "robot_id": "g2a",
        "robot_host": "192.168.10.55",
        "robot_ssh_port": 22,
        "session_id": "a" * 32,
        "pid": 4321,
    }
    prepared = control_plane.prepare_robot_session_cleanup(target, control_plane.csrf_token)
    payload = {
        **target,
        "challenge_token": prepared["challenge_token"],
        "confirmation_phrase": "关闭",
    }

    with pytest.raises(ValidationError, match="二次确认文字不匹配"):
        control_plane.cleanup_robot_session(payload, control_plane.csrf_token)
    payload["confirmation_phrase"] = prepared["confirmation_phrase"]
    with pytest.raises(ValidationError, match="二次确认已过期"):
        control_plane.cleanup_robot_session(payload, control_plane.csrf_token)


def test_execute_session_cleanup_requires_acknowledged_safe_pause(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    monkeypatch.setattr(
        control_plane,
        "probe_robot",
        lambda _payload, _token: occupied_robot_probe(mode="execute"),
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "select_robot_endpoint",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "set_robot_session_paused",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DeployError("no acknowledgement")),
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "force_stop_robot_session",
        lambda *_args, **_kwargs: pytest.fail("must not kill an unpaused execute session"),
    )
    target = {
        "robot_id": "g2a",
        "robot_host": "192.168.10.55",
        "robot_ssh_port": 22,
        "session_id": "a" * 32,
        "pid": 4321,
    }
    prepared = control_plane.prepare_robot_session_cleanup(target, control_plane.csrf_token)

    with pytest.raises(ValidationError, match="未确认进入安全暂停"):
        control_plane.cleanup_robot_session(
            {
                **target,
                "challenge_token": prepared["challenge_token"],
                "confirmation_phrase": prepared["confirmation_phrase"],
            },
            control_plane.csrf_token,
        )


def test_robot_reset_is_async_and_commands_validated_dahuan_cycle(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    control_plane.enable_reset = True
    control_plane.deployment.raw["robots"]["g2a"]["reset_enabled"] = True
    job = DeploymentJob(
        id="b" * 32,
        config=valid_payload(
            effector_profile="dahuan",
            safety={
                "workspace_safety_enabled": True,
                "safety_profile_id": "workspace_safety.calibrated.json",
            },
        ),
        status="succeeded",
    )
    job.task_session = {"remote_started": False}
    job.robot_reset = {"status": "idle"}
    control_plane._jobs[job.id] = job
    control_plane._workstation["active_deployment_id"] = job.id
    monkeypatch.setattr(
        control_plane,
        "probe_robot",
        lambda _payload, _token: {
            "reachable": True,
            "ssh_authenticated": True,
            "identity": {"match": True},
            "client": {"status": "idle", "valid": True},
            "observation": {"external_gdk_clients": []},
        },
    )
    monkeypatch.setattr(control_plane, "_robot_for_config", lambda _config: object())
    monkeypatch.setattr(
        control_plane.deployment,
        "deploy_generated_safety_profile",
        lambda *_args, **_kwargs: None,
    )
    calls: list[tuple[str, float]] = []

    def reset_with_dahuan_cycle(
        _robot: object, safety_profile_id: str, timeout_s: float
    ) -> dict[str, Any]:
        calls.append((safety_profile_id, timeout_s))
        return {
            "completed": True,
            "effectors_commanded": True,
            "effector_sequence": ["open", "close", "open"],
            "output_tail": "Reset completed",
        }

    monkeypatch.setattr(
        control_plane.deployment,
        "reset_robot_with_dahuan_cycle",
        reset_with_dahuan_cycle,
    )

    started = control_plane.reset_robot_body(
        job.id,
        control_plane.csrf_token,
    )
    assert started["robot_reset"]["status"] in {"running", "succeeded"}
    deadline = time.monotonic() + 1
    while control_plane.get_job(job.id)["robot_reset"]["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    completed = control_plane.get_job(job.id)
    assert completed["robot_reset"]["status"] == "succeeded"
    assert completed["robot_reset"]["effectors_commanded"] is True
    assert completed["robot_reset"]["effector_sequence"] == ["open", "close", "open"]
    assert calls == [("workspace_safety.calibrated.json", 70.0)]


def test_robot_body_reset_requires_stopped_task(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    control_plane.enable_reset = True
    control_plane.deployment.raw["robots"]["g2a"]["reset_enabled"] = True
    job = DeploymentJob(
        id="c" * 32,
        config=valid_payload(effector_profile="dahuan"),
        status="paused",
    )
    job.task_session = {"remote_started": True}
    job.robot_reset = {"status": "idle"}
    control_plane._jobs[job.id] = job
    monkeypatch.setattr(
        control_plane.deployment,
        "reset_robot_with_dahuan_cycle",
        lambda *_args, **_kwargs: pytest.fail("reset must not start"),
    )

    with pytest.raises(ValidationError, match="请先结束当前任务"):
        control_plane.reset_robot_body(job.id, control_plane.csrf_token)


def test_failed_reset_can_be_retried_without_clearing_the_safety_gate(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    control_plane.enable_reset = True
    control_plane.deployment.raw["robots"]["g2a"]["reset_enabled"] = True
    job = DeploymentJob(id="d" * 32, config=valid_payload(
        effector_profile="dahuan",
        safety={"workspace_safety_enabled": True,
                "safety_profile_id": "workspace_safety.calibrated.json"}),
                        status="ready_to_start")
    job.task_session = {"status": "ready", "remote_started": False}
    control_plane._jobs[job.id] = job
    control_plane._workstation["active_deployment_id"] = job.id
    monkeypatch.setattr(control_plane, "probe_robot", lambda *_args: {
        "reachable": True, "ssh_authenticated": True, "identity": {"match": True},
        "client": {"status": "idle", "valid": True},
        "observation": {"external_gdk_clients": []},
    })
    monkeypatch.setattr(control_plane, "_robot_for_config", lambda *_args: object())
    monkeypatch.setattr(control_plane.deployment, "deploy_generated_safety_profile",
                        lambda *_args: None)
    outcomes = [DeployError("simulated reset failure"), {"completed": True, "output_tail": "verified"}]
    def reset(*_args, **_kwargs):
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle", reset)
    def await_reset(expected):
        deadline = time.monotonic() + 2
        while control_plane.get_job(job.id)["robot_reset"]["status"] != expected:
            assert time.monotonic() < deadline
            time.sleep(0.01)

    control_plane.reset_robot_body(job.id, control_plane.csrf_token)
    await_reset("failed")
    assert control_plane.tasks.reset_gate()["required"] is True
    assert control_plane.get_job(job.id)["status"] == "recovery_required"
    control_plane.reset_robot_body(job.id, control_plane.csrf_token)
    await_reset("succeeded")
    assert control_plane.tasks.reset_gate()["required"] is False
    assert control_plane.get_job(job.id)["status"] == "ready_to_start"


def test_robot_control_telemetry_uses_job_token_and_latest_sequence(
    control_plane: ControlPlane,
) -> None:
    job_id = "a" * 32
    job = DeploymentJob(id=job_id, config=valid_payload(), status="running")
    job.telemetry = {"control": {"available": False, "sequence": None}}
    control_plane._jobs[job_id] = job
    control_plane._telemetry_tokens[job_id] = "token-value"

    accepted = control_plane.ingest_robot_telemetry(
        job_id, control_telemetry_payload(7), "token-value"
    )

    assert accepted == {"accepted": True, "sequence": 7}
    control = control_plane.get_control_telemetry(job_id)
    assert control["available"] is True
    assert control["sequence"] == 7
    assert control["safety"]["collision_detected"] is True
    assert control["safety"]["proximity_warning"] is False
    assert control["safety"]["candidate_collision_detected"] is False
    assert control["safety"]["candidate_proximity_warning"] is True
    assert control["safety"]["distance_to_surface"] == pytest.approx(0.0005)
    assert control["safety"]["hard_margin"] == pytest.approx(0.001)
    assert control["safety"]["warning_margin"] == pytest.approx(0.003)
    assert control["safety"]["work_surface"]["distance_m"] == pytest.approx(0.0005)
    assert control["safety"]["work_surface"]["collision"] is True
    assert control["safety"]["inter_arm"]["minimum_separation_m"] == pytest.approx(0.021)
    assert control["safety"]["inter_arm"]["collision"] is False
    assert control["safety"]["candidate_inter_arm"]["minimum_separation_m"] == pytest.approx(0.006)
    assert control["safety"]["candidate_inter_arm"]["collision"] is True
    assert control["safety"]["candidate_inter_arm"]["checked_waypoints"] == pytest.approx(10)
    assert control["safety"]["enforcement_enabled"] is False
    assert len(control["measured_state"]) == 24
    assert control["age_s"] >= 0
    stale = control_plane.ingest_robot_telemetry(
        job_id, control_telemetry_payload(6), "token-value"
    )
    assert stale == {"accepted": False, "sequence": 7, "reason": "stale"}


def test_post_load_candidate_publishes_inference_camera_urls(
    control_plane: ControlPlane,
) -> None:
    prepared = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(prepared["id"], timeout=2)
    select_ready_scheme(control_plane, prepared["id"])
    candidate = control_plane.configure_deployment(
        prepared["id"], {"task": "Pick another package"}, control_plane.csrf_token
    )
    assert candidate["id"] != prepared["id"]
    assert {camera["id"] for camera in candidate["telemetry"]["cameras"]} == {
        "cam_head", "cam_left_wrist", "cam_right_wrist"
    }

    job = control_plane._jobs[candidate["id"]]
    snapshot = {
        "running": True,
        "timing": {"request_id": "test-frame"},
        "frame": {"request_id": "test-frame"},
        "image_cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
    }
    assert ControlPlane._apply_robot_telemetry(job, snapshot) is True
    assert all(camera["image_url"] for camera in job.telemetry["cameras"])

    restored = DeploymentJob.from_dict({
        "id": "e" * 32,
        "config": {"hardware_mode": "execute"},
        "telemetry": {"request_count": 7},
    })
    assert restored.telemetry["request_count"] == 7
    assert len(restored.telemetry["cameras"]) == 3


def test_inference_frame_cache_drives_all_camera_urls_in_execute_mode() -> None:
    job = DeploymentJob(
        id="f" * 32,
        config={"hardware_mode": "execute"},
        status="running",
    )
    job.telemetry = {
        "request_count": 0,
        "request": {"request_id": None},
        "cameras": [
            {"id": "cam_head", "status": "waiting", "image_url": None},
            {"id": "cam_left_wrist", "status": "waiting", "image_url": None},
            {"id": "cam_right_wrist", "status": "waiting", "image_url": None},
        ],
    }
    snapshot = {
        "running": True,
        "timing": {"request_id": "request-12", "round_trip_ms": 42.0},
        "frame": {
            "request_id": "request-12",
            "images": {
                "cam_head": {"path": "request-12-cam_head.jpg"},
                "cam_left_wrist": {"path": "request-12-cam_left_wrist.jpg"},
                "cam_right_wrist": {"path": "request-12-cam_right_wrist.jpg"},
            },
        },
        "image_cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
        "timing_updated_at": 100.0,
        "frame_updated_at": 101.0,
    }

    assert ControlPlane._apply_robot_telemetry(job, snapshot) is True
    assert job.telemetry["last_frame_at"] == 101.0
    assert all(camera["status"] == "online" for camera in job.telemetry["cameras"])
    assert all(camera["frame_id"] == "request-12" for camera in job.telemetry["cameras"])
    assert all(
        camera["image_url"].endswith("?frame=request-12")
        for camera in job.telemetry["cameras"]
    )
    assert ControlPlane._apply_robot_telemetry(job, snapshot) is False


def test_robot_control_telemetry_rejects_wrong_token_and_nonfinite_vector(
    control_plane: ControlPlane,
) -> None:
    job_id = "b" * 32
    job = DeploymentJob(id=job_id, config=valid_payload(), status="running")
    job.telemetry = {"control": {"available": False, "sequence": None}}
    control_plane._jobs[job_id] = job
    control_plane._telemetry_tokens[job_id] = "expected-token"

    with pytest.raises(ValidationError, match="令牌无效"):
        control_plane.ingest_robot_telemetry(
            job_id, control_telemetry_payload(), "wrong-token"
        )
    payload = control_telemetry_payload()
    payload["model_candidate"][3] = float("nan")
    with pytest.raises(ValidationError, match="非有限"):
        control_plane.ingest_robot_telemetry(job_id, payload, "expected-token")


def test_robot_telemetry_queues_only_measured_pose_for_async_simulator_bridge(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = "d" * 32
    job = DeploymentJob(id=job_id, config=valid_payload(), status="running")
    job.telemetry = {"control": {"available": False, "sequence": None}}
    control_plane._jobs[job_id] = job
    control_plane._telemetry_tokens[job_id] = "bridge-token"
    control_plane.simulator_control_url = "http://127.0.0.1:8766"
    queued: list[tuple[str, int, list[float]]] = []
    monkeypatch.setattr(
        control_plane,
        "_queue_simulator_pose",
        lambda target_job, sequence, measured: queued.append(
            (target_job, sequence, list(measured))
        ),
    )
    payload = control_telemetry_payload(9)
    payload["adopted_action"] = [9.0] * 24

    control_plane.ingest_robot_telemetry(job_id, payload, "bridge-token")

    assert queued == [(job_id, 9, payload["measured_state"])]
    assert queued[0][2] != payload["adopted_action"]


def test_one_way_telemetry_stream_ingests_frames_without_command_response(
    control_plane: ControlPlane,
) -> None:
    job_id = "e" * 32
    token = "stream-token-value-long-enough"
    job = DeploymentJob(id=job_id, config=valid_payload(), status="running")
    job.telemetry = {"control": {"available": False, "sequence": None}}
    control_plane._jobs[job_id] = job
    control_plane._telemetry_tokens[job_id] = token
    server = TelemetryTCPServer(
        ("127.0.0.1", 0), _telemetry_stream_handler(control_plane)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload_json = control_telemetry_payload(12)
    payload = json.dumps(payload_json).encode()
    token_bytes = token.encode()
    frame = job_id.encode() + struct.pack("!H", len(token_bytes)) + token_bytes + payload
    try:
        with socket.create_connection(server.server_address, timeout=2) as stream:
            stream.sendall(struct.pack("!I", len(frame)) + frame)
        deadline = time.monotonic() + 1
        while control_plane.get_control_telemetry(job_id).get("sequence") != 12:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert control_plane.get_control_telemetry(job_id)["measured_state"] == payload_json[
            "measured_state"
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_read_only_can_monitor_collision_without_enabling_intervention(
    control_plane: ControlPlane,
) -> None:
    payload = valid_payload()
    payload["safety"] = {
        "workspace_safety_enabled": False,
        "safety_profile_id": "workspace_safety.calibrated.json",
    }

    result = control_plane.validate(payload)

    assert result["valid"] is True
    assert result["normalized"]["safety"]["workspace_safety_enabled"] is False
    assert result["normalized"]["safety"]["safety_profile_id"].endswith(".json")
    assert any("仅监测模式" in warning for warning in result["warnings"])


def test_bootstrap_is_recursive_redacted_and_maps_model_modes(
    control_plane: ControlPlane,
) -> None:
    bootstrap = control_plane.bootstrap()
    forbidden_keys = {
        "repo",
        "conda",
        "conda_env",
        "ssh_key",
        "ssh_target",
        "source_client",
        "bundle_directory",
        "remote_directory",
        "legacy_remote_directory",
        "base_args",
        "real_args",
        "mock_args",
    }
    for key, value in walk(bootstrap):
        if key is not None:
            assert key.lower() not in forbidden_keys
        if isinstance(value, str):
            assert not value.startswith("/")
            assert "10.42." not in value
            assert "@10.42." not in value

    serialized = json.dumps(bootstrap, ensure_ascii=False)
    for secret in (
        "/opt/private/miniforge/bin/conda",
        "/srv/private/models/pi05",
        "/srv/private/models/lingbot",
        "/home/test/.ssh/id_ed25519_private",
        "agi@10.42.0.55",
        "private.pi05.server",
    ):
        assert secret not in serialized

    models = {item["id"]: item for item in bootstrap["models"]}
    assert bootstrap["project"]["protocol"] == "nexus-policy-websocket"
    assert models["pi05"]["supported_modes"] == ["sync", "async", "rtc"]
    assert models["lingbot"]["supported_modes"] == ["sync", "async", "rtc"]
    assert models["lingbot"]["rtc_execute_enabled"] is False
    assert models["pi05"]["startup_timeout_s"] == 30
    assert models["fastwam"]["ui_only"] is True
    assert models["fastwam"]["selectable"] is True
    assert models["fastwam"]["loadable"] is False
    assert models["fastwam"]["integration_status"] == "available"
    assert models["fastwam"]["supported_modes"] == ["sync", "async", "rtc"]
    assert models["fastwam"]["port"] == 5559
    assert models["fastwam"]["effector_profiles"] == ["dahuan", "omnipicker_legacy"]
    assert models["fastwam"]["normalization"]["label"] == "大寰 checkpoint quantiles"
    assert bootstrap["defaults"]["gpu_id"] == "auto"
    assert bootstrap["defaults"]["action_low_pass_cutoff_hz"] == 4.0
    assert bootstrap["defaults"]["task"].startswith(
        "Pick up the package and place it on the conveyor belt"
    )
    profile_tasks = {
        item["id"]: item["default_task"] for item in bootstrap["effector_profiles"]
    }
    assert profile_tasks["dahuan"].startswith("Pick a package from the box")
    assert profile_tasks["omnipicker_legacy"].startswith("Pick up the package")
    assert isinstance(bootstrap["gpus"], list)
    robots = {item["id"]: item for item in bootstrap["robots"]}
    assert robots["astribot_s1"]["ui_only"] is True
    assert robots["astribot_s1"]["specifications"]["arm_dof"] == 7
    assert robots["g2a"]["compatible_models"] == ["pi05", "lingbot"]
    assert {item["id"] for item in robots["g2a"]["safety_profiles"]} == {
        "SAFETY_CONFIG_TEMPLATE.json",
        "workspace_safety.calibrated.json",
        "workspace_safety.read_only.json",
    }
    read_only_profile = next(
        item
        for item in robots["g2a"]["safety_profiles"]
        if item["id"] == "workspace_safety.read_only.json"
    )
    assert read_only_profile["read_only_ready"] is True
    assert read_only_profile["execute_approved"] is False


def test_bootstrap_defaults_to_real_when_models_are_stopped(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control_plane, "gpu_inventory", lambda: [])
    monkeypatch.setattr(
        control_plane,
        "_safe_status",
        lambda _deployment, _model: {"running": False, "healthy": False},
    )

    bootstrap = control_plane.bootstrap()

    assert bootstrap["defaults"]["model_id"] == "pi05"
    assert bootstrap["defaults"]["server_mode"] == "real"


def test_bootstrap_prefers_healthy_model_but_requires_rtc_match_for_reuse(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control_plane, "gpu_inventory", lambda: [])
    monkeypatch.setattr(
        control_plane,
        "_safe_status",
        lambda _deployment, model: {
            "running": model.name == "lingbot",
            "healthy": model.name == "lingbot",
        },
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "running_model_effector_profile",
        lambda model: model.effector_profile if model.name == "lingbot" else None,
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "running_model_version",
        lambda model: model.variant().default_version if model.name == "lingbot" else None,
    )

    bootstrap = control_plane.bootstrap()

    assert bootstrap["defaults"]["model_id"] == "lingbot"
    # This fixture has actions disabled, so the running RTC mode is unknown.
    assert bootstrap["defaults"]["server_mode"] == "real"


def test_bootstrap_exposes_body_only_robot_as_read_only(
    control_plane: ControlPlane,
) -> None:
    robot = next(item for item in control_plane.bootstrap()["robots"] if item["id"] == "legacy")

    assert robot["effectors"]["mode"] == "absent_fixed"
    assert robot["effectors"]["present"] is False
    assert robot["effectors"]["observation_source"] == "fixed_training_reference"
    assert robot["effectors"]["fixed_state"] == [-0.7, -0.71]
    assert robot["effectors"]["profile"] == "omnipicker_legacy"
    assert robot["effectors"]["configuration_locked"] is True
    assert robot["execution_supported"] is False


def test_bootstrap_hides_robot_marked_ui_hidden(control_plane: ControlPlane) -> None:
    control_plane.deployment.raw["robots"]["legacy"]["ui_hidden"] = True

    robot_ids = {item["id"] for item in control_plane.bootstrap()["robots"]}

    assert "legacy" not in robot_ids
    assert "g2a" in robot_ids


def test_body_only_robot_rejects_execute_even_when_process_gate_is_open(
    control_plane: ControlPlane,
) -> None:
    control_plane.enable_execute = True
    control_plane.deployment.raw["robots"]["legacy"]["execute_enabled"] = True

    result = control_plane.validate(
        execute_payload(
            robot_id="legacy",
            robot_host="10.42.0.56",
        )
    )

    assert any(
        error["field"] == "hardware_mode" and "未安装夹爪" in error["message"]
        for error in result["errors"]
    )


def test_gpu_inventory_parses_status_and_redacts_process_path(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = iter(
        [
            "0, GPU-a, NVIDIA RTX, 32768, 1024, 31744, 3, 51\n",
            "GPU-a, 123, /private/bin/python, 900\n",
        ]
    )

    def run(*_args: Any, **_kwargs: Any) -> Any:
        return type("Result", (), {"stdout": next(outputs)})()

    monkeypatch.setattr("nexus_deploy.control_api.subprocess.run", run)
    result = control_plane.gpu_inventory()

    assert result[0]["id"] == "0"
    assert result[0]["memory_free_mb"] == 31744
    assert result[0]["processes"] == [
        {"pid": 123, "name": "python", "memory_used_mb": 900}
    ]


def test_auto_gpu_prefers_no_compute_process_then_lower_memory(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control_plane,
        "gpu_inventory",
        lambda: [
            {"id": "0", "memory_free_mb": 30000, "memory_used_mb": 1000, "utilization_percent": 0, "process_count": 1},
            {"id": "1", "memory_free_mb": 28000, "memory_used_mb": 2000, "utilization_percent": 2, "process_count": 0},
        ],
    )
    job = DeploymentJob(
        id="job-gpu", config={"model_id": "pi05", "gpu_id": "auto"}
    )
    selected = control_plane._reserve_gpu(job)

    assert selected["id"] == "1"
    assert control_plane._reserved_gpus["1"] == "job-gpu"


def test_auto_gpu_prefers_most_free_memory_when_every_card_has_a_process(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control_plane,
        "gpu_inventory",
        lambda: [
            {"id": "0", "memory_free_mb": 30000, "memory_used_mb": 2000, "utilization_percent": 4, "process_count": 1},
            {"id": "1", "memory_free_mb": 29000, "memory_used_mb": 3000, "utilization_percent": 0, "process_count": 1},
        ],
    )
    job = DeploymentJob(id="job-gpu-busy", config={"model_id": "pi05", "gpu_id": "auto"})

    selected = control_plane._reserve_gpu(job)

    assert selected["id"] == "0"
    assert control_plane._reserved_gpus["0"] == "job-gpu-busy"


def test_gpu_status_exposes_model_specific_recommendation(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = [
        {
            "id": "0", "name": "GPU A", "memory_total_mb": 32000,
            "memory_free_mb": 30000, "memory_used_mb": 2000,
            "utilization_percent": 3, "temperature_c": 40,
            "process_count": 1, "processes": [], "reserved_by_job": None,
        },
        {
            "id": "1", "name": "GPU B", "memory_total_mb": 32000,
            "memory_free_mb": 31000, "memory_used_mb": 1000,
            "utilization_percent": 1, "temperature_c": 39,
            "process_count": 0, "processes": [], "reserved_by_job": None,
        },
    ]
    monkeypatch.setattr(control_plane, "gpu_inventory", lambda: inventory)

    status = control_plane.gpu_status()

    assert status["gpus"] == inventory
    assert status["recommendations"]["pi05"]["gpu_id"] == "1"
    assert status["recommendations"]["pi05"]["completely_idle"] is True
    assert "空闲" in status["recommendations"]["pi05"]["reason"]


def test_manual_gpu_rejects_insufficient_free_memory(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.deployment.raw["models"]["pi05"]["gpu_min_free_memory_mb"] = 12000
    monkeypatch.setattr(
        control_plane,
        "gpu_inventory",
        lambda: [
            {"id": "0", "memory_free_mb": 8000, "memory_used_mb": 24000, "utilization_percent": 0, "process_count": 0},
        ],
    )
    job = DeploymentJob(id="job-gpu", config={"model_id": "pi05", "gpu_id": "0"})

    with pytest.raises(Exception, match="空闲显存低于 12000"):
        control_plane._reserve_gpu(job)


def test_real_model_start_receives_selected_physical_gpu(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)
    monkeypatch.setattr(
        control_plane,
        "gpu_inventory",
        lambda: [
            {
                "id": "0",
                "name": "GPU A",
                "memory_total_mb": 32000,
                "memory_free_mb": 30000,
                "memory_used_mb": 2000,
                "utilization_percent": 1,
                "temperature_c": 40,
                "process_count": 0,
                "processes": [],
                "reserved_by_job": None,
            },
            {
                "id": "1",
                "name": "GPU B",
                "memory_total_mb": 32000,
                "memory_free_mb": 31000,
                "memory_used_mb": 1000,
                "utilization_percent": 0,
                "temperature_c": 41,
                "process_count": 0,
                "processes": [],
                "reserved_by_job": None,
            },
        ],
    )
    captured: dict[str, Any] = {}

    def start_model(*_args: Any, **kwargs: Any) -> int:
        captured.update(kwargs)
        return 4321

    monkeypatch.setattr(control_plane.deployment, "start_model", start_model)
    created = control_plane.create_deployment(
        valid_payload(server_mode="real", gpu_id="1"), control_plane.csrf_token
    )
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert ready["status"] == "ready_to_start"
    assert ready["model_service"]["gpu"] == "1"
    assert captured["cuda_visible_devices"] == "1"


def test_mock_model_start_uses_cpu_and_never_reserves_gpu(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)
    captured: dict[str, Any] = {}

    def start_model(*_args: Any, **kwargs: Any) -> int:
        captured.update(kwargs)
        return 4321

    monkeypatch.setattr(control_plane.deployment, "start_model", start_model)
    monkeypatch.setattr(
        control_plane,
        "_reserve_gpu",
        lambda _job: pytest.fail("Mock must not reserve a GPU"),
    )

    validation = control_plane.validate(valid_payload(server_mode="mock"))
    created = control_plane.create_deployment(
        valid_payload(server_mode="mock"), control_plane.csrf_token
    )
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert any("不能代表模型推理效果" in warning for warning in validation["warnings"])
    assert captured["mock"] is True
    assert captured["cuda_visible_devices"] == ""
    assert ready["model_service"]["gpu"] == "cpu"
    assert ready["status"] == "ready_to_start"


def test_validate_returns_public_shape_and_normalized_summary(
    control_plane: ControlPlane,
) -> None:
    result = control_plane.validate(
        valid_payload(task="  Pick and place  ", execution_fps="15", quality_horizon="40")
    )

    assert set(result) == {"valid", "errors", "warnings", "normalized", "summary"}
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["normalized"]["task"] == "Pick and place"
    assert result["normalized"]["execution_fps"] == 15.0
    assert result["normalized"]["quality_horizon"] == 40
    assert result["summary"] == {
            "route": "G2A → 智元原装 Omnipicker 夹爪 → PI0.5 → default",
        "mode": "async",
        "hardware": "read-only",
        "task": "Pick and place",
        "human_in_loop": False,
        "safety_enabled": False,
        "包裹检测": "未启用",
            "endpoint": "192.168.10.55:22",
            "gpu": "自动选择",
        }


def test_validate_defaults_and_rejects_non_allow_listed_model_version(
    control_plane: ControlPlane,
) -> None:
    defaulted = control_plane.validate(valid_payload())
    rejected = control_plane.validate(valid_payload(model_version="../../checkpoint"))

    assert defaulted["valid"] is True
    assert defaulted["normalized"]["model_version"] == "default"
    assert rejected["valid"] is False
    assert any(error["field"] == "model_version" for error in rejected["errors"])


def test_validate_accepts_legacy_dh_alias_but_returns_canonical_profile(
    control_plane: ControlPlane,
) -> None:
    model = control_plane.deployment.models["pi05"]
    legacy_variant = model.effector_variants["omnipicker_legacy"]
    dahuan_versions = {
        version_id: replace(version, effector_profile="dahuan")
        for version_id, version in legacy_variant.versions.items()
    }
    control_plane.deployment.models["pi05"] = replace(
        model,
        effector_profiles=("dahuan", "omnipicker_legacy"),
        effector_variants={
            "dahuan": replace(
                legacy_variant,
                effector_profile="dahuan",
                versions=dahuan_versions,
            ),
            "omnipicker_legacy": legacy_variant,
        },
    )
    control_plane.deployment.robots["g2a"] = replace(
        control_plane.deployment.robots["g2a"],
        effector_profiles=("dahuan", "omnipicker_legacy"),
    )

    result = control_plane.validate(valid_payload(effector_profile="dh"))

    assert result["valid"] is True
    assert result["normalized"]["effector_profile"] == "dahuan"
    assert result["warnings"]


@pytest.mark.parametrize("mode", ["device-position", "software-trajectory"])
def test_validate_dahuan_effector_control_mode(control_plane: ControlPlane, mode: str) -> None:
    control_plane.deployment.robots["g2a"] = replace(
        control_plane.deployment.robots["g2a"], effector_profiles=("dahuan", "omnipicker_legacy"))
    model = control_plane.deployment.models["pi05"]
    legacy_variant = model.effector_variants["omnipicker_legacy"]
    dahuan_versions = {
        version_id: replace(version, effector_profile="dahuan")
        for version_id, version in legacy_variant.versions.items()
    }
    dahuan_variant = replace(
        legacy_variant,
        effector_profile="dahuan",
        versions=dahuan_versions,
    )
    control_plane.deployment.models["pi05"] = replace(
        model,
        effector_profiles=("dahuan", "omnipicker_legacy"),
        effector_variants={
            "dahuan": dahuan_variant,
            "omnipicker_legacy": legacy_variant,
        },
    )
    result = control_plane.validate(valid_payload(effector_profile="dahuan", effector_control_mode=mode))
    assert result["valid"], result["errors"]
    assert result["normalized"]["effector_control_mode"] == mode


@pytest.mark.parametrize("mode", ["unknown", [], None])
def test_validate_bad_effector_control_mode(control_plane: ControlPlane, mode) -> None:
    result = control_plane.validate(valid_payload(effector_control_mode=mode))
    assert not result["valid"]
    assert any(error["field"] == "effector_control_mode" for error in result["errors"])


@pytest.mark.parametrize("task", ["", "   ", "x" * 4097])
def test_validate_requires_task_with_protocol_length_limit(
    control_plane: ControlPlane, task: str
) -> None:
    result = control_plane.validate(valid_payload(task=task))

    assert result["valid"] is False
    assert "task" in error_fields(result)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"robot_id": "missing"}, "robot_id"),
        ({"model_id": "missing"}, "model_id"),
        ({"model_id": "fastwam"}, "model_id"),
        ({"robot_id": "legacy", "model_id": "lingbot"}, "model_id"),
    ],
)
def test_validate_requires_known_and_compatible_targets(
    control_plane: ControlPlane, overrides: dict[str, Any], field: str
) -> None:
    result = control_plane.validate(valid_payload(**overrides))

    assert result["valid"] is False
    assert field in error_fields(result)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"robot_host": ""}, "robot_host"),
        ({"robot_host": "robot.example"}, "robot_host"),
        ({"robot_host": "8.8.8.8"}, "robot_host"),
        ({"robot_ssh_port": 0}, "robot_ssh_port"),
        ({"robot_ssh_port": 65536}, "robot_ssh_port"),
        ({"robot_ssh_port": 22.5}, "robot_ssh_port"),
    ],
)
def test_validate_requires_private_robot_ip_and_valid_ssh_port(
    control_plane: ControlPlane, overrides: dict[str, Any], field: str
) -> None:
    result = control_plane.validate(valid_payload(**overrides))

    assert result["valid"] is False
    assert field in error_fields(result)


def test_probe_robot_reports_tcp_reachability(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Connection:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    observed: list[tuple[tuple[str, int], float]] = []

    def connect(endpoint: tuple[str, int], timeout: float) -> Connection:
        observed.append((endpoint, timeout))
        return Connection()

    monkeypatch.setattr("nexus_deploy.control_api.socket.create_connection", connect)
    result = control_plane.probe_robot(
        {"robot_id": "g2a", "robot_host": "192.168.10.55", "robot_ssh_port": 22},
        control_plane.csrf_token,
    )

    assert result["reachable"] is True
    assert result["scope"] == "network_and_client"
    assert result["ssh_authenticated"] is False
    assert result["client"]["status"] == "not_checked"
    assert result["runtime_session"] == "not_checked"
    assert observed == [(('192.168.10.55', 22), 0.8)]


def test_probe_robot_reports_deployed_idle_client(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Connection:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        "nexus_deploy.control_api.socket.create_connection",
        lambda *_args, **_kwargs: Connection(),
    )
    control_plane.deployment.raw["robots"]["g2a"]["control_host"] = "192.168.10.55"
    monkeypatch.setattr(
        control_plane.deployment,
        "probe_robot_client",
        lambda _robot: {
            "installed": True,
            "running": False,
            "status": "idle",
            "processes": [],
            "protocol": {"name": "g2a-policy-websocket", "version": 2},
        },
    )

    result = control_plane.probe_robot(
        {"robot_id": "g2a", "robot_host": "192.168.10.55", "robot_ssh_port": 22},
        control_plane.csrf_token,
    )

    assert result["ssh_authenticated"] is True
    assert result["client"]["status"] == "idle"
    assert result["runtime_session"] == "idle"


def test_probe_robot_allows_ptp_warning_when_gdk_services_are_ready(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Connection:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        "nexus_deploy.control_api.socket.create_connection",
        lambda *_args, **_kwargs: Connection(),
    )
    control_plane.deployment.raw["robots"]["g2a"]["control_host"] = "192.168.10.55"
    monkeypatch.setattr(
        control_plane.deployment,
        "probe_robot_client",
        lambda _robot: {
            "installed": True,
            "running": False,
            "status": "idle",
            "processes": [],
            "protocol": {"name": "nexus-policy-websocket", "version": 2},
        },
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_runtime_preflight",
        lambda _robot: {
            "ptp_devices": [
                {"path": "/dev/ptp_xgi0", "readable": False, "writable": False}
            ],
            "gdk_processes": [
                "gdk_service",
                "gdk_http_server",
                "camera_service",
                "fastdds discovery",
            ],
        },
    )

    result = control_plane.probe_robot(
        {"robot_id": "g2a", "robot_host": "192.168.10.55", "robot_ssh_port": 22},
        control_plane.csrf_token,
    )

    assert result["observation"]["ready"] is True
    assert result["observation"]["status"] == "ready_with_warning"
    assert result["observation"]["ptp_ready"] is False
    assert "仅警告" in result["observation"]["reason"]


def test_probe_robot_blocks_a_network_address_bound_to_another_physical_robot(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Connection:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        "nexus_deploy.control_api.socket.create_connection",
        lambda *_args, **_kwargs: Connection(),
    )
    control_plane.deployment.raw["robots"]["g2a"]["control_host"] = "192.168.10.55"
    control_plane.deployment.robots["g2a"] = replace(
        control_plane.deployment.robots["g2a"], expected_aid="G2A-EXPECTED"
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "probe_robot_client",
        lambda _robot: {"installed": True, "running": False, "status": "idle"},
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_runtime_preflight",
        lambda _robot: {"asset_id": "G2A-OTHER"},
    )

    result = control_plane.probe_robot(
        {"robot_id": "g2a", "robot_host": "192.168.10.55", "robot_ssh_port": 22},
        control_plane.csrf_token,
    )

    assert result["identity"] == {
        "expected_aid": "G2A-EXPECTED",
        "observed_aid": "G2A-OTHER",
        "match": False,
    }
    assert result["client"]["status"] == "wrong_robot"
    assert result["observation"]["ready"] is False
    assert "已阻止部署、观测和真机执行" in result["observation"]["reason"]


def test_probe_robot_requires_csrf_and_rejects_public_ip(control_plane: ControlPlane) -> None:
    with pytest.raises(ValidationError, match="CSRF"):
        control_plane.probe_robot(
            {"robot_id": "g2a", "robot_host": "192.168.10.55", "robot_ssh_port": 22},
            "wrong",
        )
    with pytest.raises(ValidationError, match="私有网络"):
        control_plane.probe_robot(
            {"robot_id": "g2a", "robot_host": "8.8.8.8", "robot_ssh_port": 22},
            control_plane.csrf_token,
        )


def test_validate_rtc_is_capability_gated(control_plane: ControlPlane) -> None:
    pi05 = control_plane.validate(valid_payload(model_id="pi05", inference_mode="rtc"))
    lingbot_read_only = control_plane.validate(
        valid_payload(model_id="lingbot", inference_mode="rtc")
    )
    lingbot_execute = control_plane.validate(
        valid_payload(
            model_id="lingbot",
            inference_mode="rtc",
            hardware_mode="execute",
        )
    )

    assert pi05["valid"] is True
    assert lingbot_read_only["valid"] is True
    assert lingbot_execute["valid"] is False
    assert "hardware_mode" in error_fields(lingbot_execute)
    assert any("RTC 尚未完成 read-only 验收" in item["message"] for item in lingbot_execute["errors"])


def test_validate_requires_quality_horizon_not_exceed_prediction(
    control_plane: ControlPlane,
) -> None:
    result = control_plane.validate(valid_payload(prediction_horizon=30, quality_horizon=31))

    assert result["valid"] is False
    assert "quality_horizon" in error_fields(result)


@pytest.mark.parametrize(
    "profile", ["dataset-v30-p99-margin", "urdf-velocity-p99-acceleration"]
)
def test_experimental_arm_limit_profile_is_valid_but_warned(
    control_plane: ControlPlane, profile: str,
) -> None:
    result = control_plane.validate(valid_payload(arm_limit_profile=profile))
    assert result["valid"] is True
    assert result["normalized"]["arm_limit_profile"] == profile
    assert any("不是厂商认证上限" in warning for warning in result["warnings"])


def test_default_urdf_velocity_acceleration_8_profile_is_valid_but_acceleration_is_disclosed(
    control_plane: ControlPlane,
) -> None:
    result = control_plane.validate(valid_payload())

    assert result["valid"] is True
    assert result["normalized"]["arm_limit_profile"] == "urdf-velocity-uniform-acceleration"
    assert result["normalized"]["max_arm_velocity"] == pytest.approx(3.76)
    assert result["normalized"]["max_arm_acceleration"] == pytest.approx(8.0)
    assert any("未声明加速度" in warning for warning in result["warnings"])


def test_new_runtime_defaults_propose_rtc_30hz_execute() -> None:
    assert ControlPlane.DEFAULTS["inference_mode"] == "rtc"
    assert ControlPlane.DEFAULTS["execution_fps"] == pytest.approx(30.0)
    assert ControlPlane.DEFAULTS["hardware_mode"] == "execute"


def test_unknown_arm_limit_profile_is_rejected(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(arm_limit_profile="unsafe"))
    assert result["valid"] is False
    assert "arm_limit_profile" in error_fields(result)


def test_urdf_velocity_only_is_explicit_and_warned(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(arm_limit_profile="urdf-velocity-only"))
    assert result["valid"] is True
    assert result["normalized"]["arm_limit_profile"] == "urdf-velocity-only"
    assert any("不限制软件加速度" in warning for warning in result["warnings"])


@pytest.mark.parametrize("mode", ["linear", "monotone-cubic"])
def test_action_interpolation_modes_are_valid(control_plane: ControlPlane, mode: str) -> None:
    result = control_plane.validate(valid_payload(action_interpolation=mode))
    assert result["valid"] is True
    assert result["normalized"]["action_interpolation"] == mode
    if mode == "monotone-cubic":
        assert any("加速度超限比例可能上升" in warning for warning in result["warnings"])


def test_monotone_cubic_can_be_selected_for_physical_execution(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(
        hardware_mode="execute", action_interpolation="monotone-cubic"
    ))
    assert "action_interpolation" not in error_fields(result)
    assert any("真机测试" in warning for warning in result["warnings"])


def test_unknown_action_interpolation_is_rejected(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(action_interpolation="unsafe"))
    assert result["valid"] is False
    assert "action_interpolation" in error_fields(result)


@pytest.mark.parametrize("mode", ["adaptive-fixed-time", "savgol"])
def test_chunk_smoothing_modes_are_valid(control_plane: ControlPlane, mode: str) -> None:
    result = control_plane.validate(valid_payload(chunk_smoothing_mode=mode))

    assert result["valid"] is True
    assert result["normalized"]["chunk_smoothing_mode"] == mode


def test_unknown_chunk_smoothing_mode_is_rejected(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(chunk_smoothing_mode="stack-both"))

    assert result["valid"] is False
    assert "chunk_smoothing_mode" in error_fields(result)


def test_tracking_parameters_and_prefix_execute_gate(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(arm_tracking_mode="damped", arm_tracking_omega=16))
    assert result["valid"] and result["normalized"]["arm_tracking_omega"] == 16
    for update, field in [({"arm_tracking_mode": "bad"}, "arm_tracking_mode"),
                          ({"arm_tracking_mode": "damped", "arm_tracking_omega": 26}, "arm_tracking_omega"),
                          ({"rtc_prefix_mode": "command", "hardware_mode": "execute"}, "rtc_prefix_mode")]:
        assert field in error_fields(control_plane.validate(valid_payload(**update)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_fps", 0),
        ("servo_fps", float("inf")),
        ("duration_s", -0.1),
        ("execution_fps", True),
        ("prediction_horizon", 0),
        ("prediction_horizon", 50.5),
        ("quality_horizon", 0),
        ("request_lead_margin_steps", -1),
        ("min_rtc_blend_steps", -1),
        ("warmup_requests", -1),
        ("watchdog_timeout_s", float("nan")),
        ("max_arm_velocity", -0.1),
    ],
)
def test_validate_rejects_nonfinite_or_out_of_range_numbers(
    control_plane: ControlPlane, field: str, value: Any
) -> None:
    result = control_plane.validate(valid_payload(**{field: value}))

    assert result["valid"] is False
    assert field in error_fields(result)


def test_hitl_enables_collection_and_ignores_legacy_approval_timeout(
    control_plane: ControlPlane,
) -> None:
    result = control_plane.validate(
        valid_payload(human_in_loop=True, approval_timeout_s="not-used")
    )

    assert result["valid"] is True
    assert result["normalized"]["human_in_loop"] is True
    assert result["normalized"]["approval_timeout_s"] == 300.0
    assert any("VR 接管、回退和录制封存" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    ("field", "cutoff"),
    [
        ("action_low_pass_cutoff_hz", 50),
        ("effector_action_low_pass_cutoff_hz", 80),
    ],
)
def test_validate_requires_low_pass_below_servo_nyquist(
    control_plane: ControlPlane, field: str, cutoff: float
) -> None:
    result = control_plane.validate(valid_payload(servo_fps=100, **{field: cutoff}))

    assert result["valid"] is False
    assert field in error_fields(result)


@pytest.mark.parametrize("workspace_safety_enabled", [False, True])
def test_selected_collision_profile_requires_100hz_servo(
    control_plane: ControlPlane,
    workspace_safety_enabled: bool,
) -> None:
    result = control_plane.validate(
        valid_payload(
            servo_fps=30,
            safety={
                "workspace_safety_enabled": workspace_safety_enabled,
                "safety_profile_id": "workspace_safety.calibrated.json",
            },
        )
    )
    assert result["valid"] is False
    assert "servo_fps" in error_fields(result)


def test_no_collision_profile_keeps_servo_rate_configurable(
    control_plane: ControlPlane,
) -> None:
    result = control_plane.validate(valid_payload(servo_fps=30))
    assert result["valid"] is True


@pytest.mark.parametrize(
    ("profile_id", "valid"),
    [
        ("", False),
        ("SAFETY_CONFIG_TEMPLATE.json", False),
        ("workspace_safety.calibrated.json", True),
        ("workspace_safety.read_only.json", True),
    ],
)
def test_read_only_workspace_safety_requires_a_read_only_ready_profile(
    control_plane: ControlPlane, profile_id: str, valid: bool
) -> None:
    result = control_plane.validate(
        valid_payload(
            safety={
                "workspace_safety_enabled": True,
                "safety_profile_id": profile_id,
            }
        )
    )

    assert result["valid"] is valid
    if not valid:
        assert "safety.safety_profile_id" in error_fields(result)


def test_read_only_safety_profile_is_never_approved_for_execute(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)
    control_plane.deployment.robots["g2a"].manifest.write_text(
        "workspace_safety.calibrated.json\nworkspace_safety.read_only.json\n",
        encoding="utf-8",
    )
    payload = execute_payload()
    payload["safety"] = {
        **payload["safety"],
        "safety_profile_id": "workspace_safety.read_only.json",
    }

    result = control_plane.validate(payload)

    assert result["valid"] is False
    assert any("真机标定与验收" in item["message"] for item in result["errors"])


def test_empty_workspace_profile_null_is_normalized_to_unselected(
    control_plane: ControlPlane,
) -> None:
    result = control_plane.validate(
        valid_payload(
            safety={
                "workspace_safety_enabled": False,
                "safety_profile_id": None,
            }
        )
    )

    assert result["valid"] is True
    assert result["normalized"]["safety"]["safety_profile_id"] == ""


def test_hardware_execute_is_never_exposed_by_default(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(hardware_mode="execute"))

    assert control_plane.bootstrap()["constraints"]["physical_execute_available"] is False
    assert result["valid"] is False
    assert "hardware_mode" in error_fields(result)


def unlock_test_execute(control_plane: ControlPlane) -> None:
    """Enable every explicit execute gate in the isolated test fixture."""

    control_plane.enable_execute = True
    control_plane.deployment.raw["robots"]["g2a"]["execute_enabled"] = True
    control_plane.deployment.robots["g2a"].manifest.write_text(
        "workspace_safety.calibrated.json\n", encoding="utf-8"
    )


def execute_payload(**overrides: Any) -> dict[str, Any]:
    payload = valid_payload(
        hardware_mode="execute",
        server_mode="real",
        human_in_loop=True,
        duration_s=60,
        safety={
            "watchdog_enabled": True,
            "kinematic_limits_enabled": True,
            "tracking_error_enabled": True,
            "low_pass_filter_enabled": True,
            "effector_low_pass_filter_enabled": True,
            "hold_effectors": False,
            "workspace_safety_enabled": True,
            "safety_profile_id": "workspace_safety.calibrated.json",
        },
    )
    payload.update(overrides)
    return payload


def test_execute_requires_process_site_safety_and_operator_gates(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)

    valid = control_plane.validate(execute_payload())
    assert valid["valid"] is True
    assert control_plane.bootstrap()["constraints"]["physical_execute_available"] is True
    assert next(
        robot for robot in control_plane.bootstrap()["robots"] if robot["id"] == "g2a"
    )["execution_supported"] is True

    assert control_plane.validate(execute_payload(human_in_loop=False))["valid"] is True
    zero_duration = control_plane.validate(execute_payload(duration_s=0))
    assert zero_duration["valid"] is True
    assert zero_duration["normalized"]["duration_s"] == 0
    assert control_plane.validate(execute_payload(server_mode="mock"))["valid"] is False


def test_execute_allows_collision_interception_to_remain_disabled(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)
    payload = execute_payload()
    payload["safety"] = {
        **payload["safety"],
        "workspace_safety_enabled": False,
        "safety_profile_id": "",
    }

    result = control_plane.validate(payload)

    assert result["valid"] is True
    assert result["normalized"]["safety"]["workspace_safety_enabled"] is False
    assert any("未启用碰撞危险拦截" in warning for warning in result["warnings"])


def test_execute_start_uses_ui_double_click_and_supports_soft_pause(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)
    created = control_plane.create_deployment(execute_payload(), control_plane.csrf_token)
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert ready["status"] == "ready_to_start"
    assert ready["config"]["human_in_loop"] is True
    select_ready_scheme(control_plane, created["id"])

    started = control_plane.start_deployment(created["id"], control_plane.csrf_token)
    assert started["status"] == "task_starting"
    running = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert running["status"] == "running"
    paused = control_plane.pause_deployment(created["id"], control_plane.csrf_token)
    assert paused["status"] == "paused"
    assert paused["task_session"]["soft_pause"] is True
    resumed = control_plane.resume_deployment(created["id"], control_plane.csrf_token)
    assert resumed["status"] == "running"
    assert control_plane.stop_deployment(created["id"], control_plane.csrf_token)["status"] == "succeeded"


def test_web_dry_run_mode_has_been_removed(control_plane: ControlPlane) -> None:
    result = control_plane.validate(valid_payload(hardware_mode="dry-run"))

    assert result["valid"] is False
    assert "hardware_mode" in error_fields(result)


def test_control_plane_has_no_raw_shell_or_argument_injection_channel(
    control_plane: ControlPlane,
) -> None:
    raw_channels = {
        "command": "touch /tmp/should-not-run",
        "args": ["--execute"],
        "base_args": ["$(id)"],
        "real_args": ["; shutdown -h now"],
        "mock_args": ["--arbitrary"],
        "shell": "/bin/sh",
        "env": {"LD_PRELOAD": "/tmp/evil.so"},
        "ssh_target": "root@host",
        "repo": "/tmp/untrusted",
    }
    payload = valid_payload(**raw_channels)
    payload["safety"]["raw_command"] = "echo unsafe"

    result = control_plane.validate(payload)

    assert result["valid"] is False
    assert set(raw_channels) <= error_fields(result)
    assert "safety.raw_command" in error_fields(result)
    with pytest.raises(ValidationError):
        control_plane.create_deployment(payload, control_plane.csrf_token)


def test_create_deployment_requires_csrf(control_plane: ControlPlane) -> None:
    with pytest.raises(ValidationError):
        control_plane.create_deployment(valid_payload(), "wrong-token")


def test_action_create_requires_a_fresh_server_side_robot_probe(
    control_plane: ControlPlane,
) -> None:
    control_plane.enable_actions = True
    with pytest.raises(ValidationError, match="检测已过期"):
        control_plane.create_deployment(valid_payload(), control_plane.csrf_token)

    grant_ready_robot_probe(control_plane)
    control_plane._probe_leases[("g2a", "192.168.10.55", 22)]["checked_at"] -= 31
    with pytest.raises(ValidationError, match="检测已过期"):
        control_plane.create_deployment(valid_payload(), control_plane.csrf_token)


def test_non_hitl_job_runs_and_can_be_paused_resumed_and_stopped(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_action(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("enable_actions=False must never start a model process")

    monkeypatch.setattr(control_plane.deployment, "start_model", forbidden_action)
    created = control_plane.create_deployment(
        valid_payload(human_in_loop=False), control_plane.csrf_token
    )
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert ready["status"] == "ready_to_start"
    assert ready["model_service"]["status"] == "simulated"
    select_ready_scheme(control_plane, created["id"])
    starting = control_plane.start_deployment(created["id"], control_plane.csrf_token)
    assert starting["status"] == "task_starting"
    running = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert running["status"] == "running"
    assert "安全预览" in running["message"]
    paused = control_plane.pause_deployment(created["id"], control_plane.csrf_token)
    assert paused["status"] == "paused"
    resumed = control_plane.resume_deployment(created["id"], control_plane.csrf_token)
    assert resumed["status"] == "running"
    with control_plane._lock:
        control_plane._jobs[created["id"]].telemetry["control"].update(
            available=True,
            sequence=8949,
            received_at=time.time(),
            measured_state=[0.0] * 24,
        )
    stopped = control_plane.stop_deployment(created["id"], control_plane.csrf_token)
    assert stopped["status"] == "succeeded"
    assert stopped["progress"] == 100

    restarted = control_plane.start_deployment(created["id"], control_plane.csrf_token)
    assert restarted["status"] == "task_starting"
    restarted_control = control_plane.get_control_telemetry(created["id"])
    assert restarted_control["available"] is False
    assert restarted_control["sequence"] is None
    assert restarted_control["received_at"] is None
    assert restarted_control["measured_state"] is None
    running_again = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert running_again["status"] == "running"
    assert control_plane.stop_deployment(created["id"], control_plane.csrf_token)["status"] == "succeeded"


def test_configured_duration_pause_keeps_remote_session_resumable(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = control_plane.validate(valid_payload(duration_s=5))["normalized"]
    job = DeploymentJob(id="a" * 32, config=config, status="running")
    job.task_session.update(
        {"generation": 1, "status": "running", "remote_started": True}
    )
    control_plane._jobs[job.id] = job
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_session_snapshot",
        lambda *_args: {
            "running": True,
            "control": {
                "state": "paused",
                "message": "配置运行时长已到，客户端自动软暂停",
            },
        },
    )

    control_plane._telemetry_loop(job.id, generation=1)

    assert job.status == "paused"
    assert job.task_session["remote_started"] is True
    assert job.task_session["safe_hold_available"] is True
    assert "可继续或结束任务" in job.message


def test_unexpected_robot_exit_is_not_misreported_as_duration_success(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = control_plane.validate(valid_payload(duration_s=0))["normalized"]
    job = DeploymentJob(id="b" * 32, config=config, status="running", progress=91)
    job.task_session.update(
        {"generation": 1, "status": "running", "remote_started": True}
    )
    control_plane._jobs[job.id] = job
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_session_snapshot",
        lambda *_args: {"running": False},
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "tail_robot_session",
        lambda *_args, **_kwargs: (
            "Robot client terminated due to an unhandled error\n"
            "RuntimeError: Safety guard exceeded its realtime deadline\n"
            "Robot client stopped after 0 policy steps and 3 servo ticks"
        ),
    )
    monkeypatch.setattr(control_plane, "_release_model", lambda *_args: None)
    monkeypatch.setattr(control_plane, "_release_robot", lambda *_args: None)

    control_plane._telemetry_loop(job.id, generation=1)

    assert job.status == "failed"
    assert job.task_session["status"] == "failed"
    assert job.message == "机器人客户端意外退出"
    assert "realtime deadline" in (job.error or "")


def test_physical_start_persists_reset_requirement_before_robot_worker(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    control_plane.enable_execute = True
    config = control_plane.validate(valid_payload(
        hardware_mode="execute", effector_profile="dahuan",
        safety={"workspace_safety_enabled": True,
                "safety_profile_id": "workspace_safety.calibrated.json"},
    ))["normalized"]
    job = DeploymentJob(id="f" * 32, config=config, status="ready_to_start")
    job.model_service = {"status": "ready", "healthy": True}
    job.task_session = {"status": "ready", "runtime_configured": True}
    control_plane._jobs[job.id] = job
    control_plane._workstation["active_deployment_id"] = job.id
    monkeypatch.setattr(control_plane, "validate",
                        lambda *_args: {"valid": True, "normalized": config})
    monkeypatch.setattr(control_plane, "_safe_status", lambda *_args: {"healthy": True})
    monkeypatch.setattr(control_plane, "_require_robot_probe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(control_plane, "_start_task_safely", lambda *_args: None)

    control_plane.start_deployment(job.id, control_plane.csrf_token)

    assert control_plane.tasks.reset_gate()["required"] is True
    assert control_plane.tasks.reset_gate()["source_job_id"] == job.id
    persisted = json.loads(control_plane._state_path.read_text(encoding="utf-8"))
    assert persisted["workstation"]["reset_gate"]["required"] is True


def test_task_startup_can_be_cancelled(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(created["id"], timeout=2)
    select_ready_scheme(control_plane, created["id"])

    starting = control_plane.start_deployment(created["id"], control_plane.csrf_token)
    assert starting["status"] == "task_starting"
    cancelled = control_plane.stop_deployment(created["id"], control_plane.csrf_token)

    assert cancelled["status"] == "cancelled"
    assert cancelled["task_session"]["cancel_requested"] is True


def test_robot_resource_is_exclusive_while_a_task_is_running(
    control_plane: ControlPlane,
) -> None:
    first = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    second = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(first["id"], timeout=2)
    control_plane.wait_for_terminal(second["id"], timeout=2)
    select_ready_scheme(control_plane, first["id"])
    control_plane.start_deployment(first["id"], control_plane.csrf_token)
    control_plane.wait_for_terminal(first["id"], timeout=2)

    with pytest.raises(TaskError, match="当前方案"):
        control_plane.start_deployment(second["id"], control_plane.csrf_token)

    control_plane.stop_deployment(first["id"], control_plane.csrf_token)
    with pytest.raises(TaskError, match="当前方案"):
        control_plane.start_deployment(second["id"], control_plane.csrf_token)


def test_hitl_job_is_ready_without_approval(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(
        valid_payload(human_in_loop=True), control_plane.csrf_token
    )
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert ready["status"] == "ready_to_start"
    assert ready["progress"] == 86
    assert ready["config"]["human_in_loop"] is True
    select_ready_scheme(control_plane, created["id"])
    control_plane.start_deployment(created["id"], control_plane.csrf_token)
    running = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert running["status"] == "running"


def test_hitl_false_and_true_follow_the_same_state_flow(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(
        valid_payload(human_in_loop=True), control_plane.csrf_token
    )
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert ready["status"] == "ready_to_start"
    assert "approved" not in ready["task_session"]


def test_post_load_configuration_is_editable_but_model_startup_is_immutable(
    control_plane: ControlPlane,
) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert ready["status"] == "ready_to_start"

    updated = control_plane.configure_deployment(
        created["id"],
        {
            "task": "Move the blue box to the left bin",
            "inference_mode": "rtc",
            "quality_horizon": 40,
            "arm_limit_profile": "dataset-v30",
            "max_arm_velocity": 1.3,
            "max_arm_acceleration": 3.5,
        },
        control_plane.csrf_token,
    )
    assert updated["status"] == "ready_to_start"
    assert updated["config"]["task"] == "Move the blue box to the left bin"
    assert updated["config"]["inference_mode"] == "rtc"
    assert updated["config"]["quality_horizon"] == 40
    assert updated["config"]["arm_limit_profile"] == "dataset-v30"
    assert updated["config"]["max_arm_velocity"] == 1.3
    assert updated["config"]["max_arm_acceleration"] == 3.5

    with pytest.raises(ValidationError, match="model_id"):
        control_plane.configure_deployment(
            created["id"], {"model_id": "lingbot"}, control_plane.csrf_token
        )
    with pytest.raises(ValidationError, match="server_mode"):
        control_plane.configure_deployment(
            created["id"], {"server_mode": "real"}, control_plane.csrf_token
        )
    with pytest.raises(ValidationError, match="robot_host"):
        control_plane.configure_deployment(
            created["id"], {"robot_host": "192.168.10.99"}, control_plane.csrf_token
        )


def test_package_completion_is_selected_after_model_preparation(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    ready = control_plane.wait_for_terminal(created["id"], timeout=2)
    assert ready["config"]["presence_gate"]["enabled"] is False
    assert ready["task_session"]["runtime_configured"] is False

    configured = control_plane.configure_deployment(
        created["id"], {"presence_gate": {"enabled": True}}, control_plane.csrf_token
    )
    assert configured["id"] == created["id"]
    assert configured["config"]["presence_gate"]["enabled"] is True
    assert configured["task_session"]["runtime_configured"] is True
    control_plane.tasks.register_profiles({"sorting": configured["config"]})
    assert control_plane.tasks.configure({"mode": "automatic"}, control_plane.csrf_token)["mode"] == "automatic"


def test_enabling_hitl_after_model_load_only_updates_metadata(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(created["id"], timeout=2)

    updated = control_plane.configure_deployment(
        created["id"], {"human_in_loop": True}, control_plane.csrf_token
    )

    assert updated["status"] == "ready_to_start"
    assert updated["config"]["human_in_loop"] is True
    assert "approved" not in updated["task_session"]


def test_job_exposes_explicit_unavailable_telemetry(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    job = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert job["telemetry"]["available"] is False
    assert job["telemetry"]["transport"] == "not_connected"
    assert {camera["id"] for camera in job["telemetry"]["cameras"]} == {
        "cam_head",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert job["telemetry"]["request"]["request_id"] is None


def test_telemetry_marks_historical_response_and_frame_as_stale(
    control_plane: ControlPlane,
) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(created["id"], timeout=2)
    with control_plane._lock:
        job = control_plane._jobs[created["id"]]
        job.status = "running"
        job.telemetry.update(
            {
                "available": True,
                "last_response_at": time.time() - 10,
                "last_frame_at": time.time() - 8,
            }
        )
        job.telemetry["cameras"][0].update({"frame_id": "request-1", "status": "online"})

    value = control_plane.get_job(created["id"])

    assert value["telemetry"]["available"] is False
    assert value["telemetry"]["transport"] == "stalled"
    assert value["telemetry"]["inference_status"] == "stalled"
    assert value["telemetry"]["cameras"][0]["status"] == "last_frame"
    assert value["telemetry"]["response_age_s"] >= 10


def test_job_snapshot_survives_control_plane_restart(control_plane: ControlPlane) -> None:
    created = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    expected = control_plane.wait_for_terminal(created["id"], timeout=2)

    restored_plane = ControlPlane(control_plane.config_path, enable_actions=False)
    restored = restored_plane.get_job(created["id"])

    assert restored["id"] == expected["id"]
    assert restored["status"] == expected["status"]
    assert restored["config"]["task"] == expected["config"]["task"]


def test_any_completed_job_can_close_its_managed_model_regardless_of_ownership(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    reused = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(reused["id"], timeout=2)
    reused_closed = control_plane.close_model(reused["id"], control_plane.csrf_token)
    assert reused_closed["status"] == "model_stopped"
    assert reused_closed["model_service"]["status"] == "stopped"
    assert control_plane._workstation["accepting"] is False

    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)
    monkeypatch.setattr(control_plane.deployment, "start_model", lambda *_args, **_kwargs: 1234)
    stopped: list[str] = []
    def stop_model(model):
        assert control_plane._workstation["accepting"] is False
        stopped.append(model.name)
    monkeypatch.setattr(control_plane.deployment, "stop_model", stop_model)
    owned = control_plane.create_deployment(
        valid_payload(server_mode="mock"), control_plane.csrf_token
    )
    control_plane.wait_for_terminal(owned["id"], timeout=2)

    closed = control_plane.close_model(owned["id"], control_plane.csrf_token)

    assert stopped == ["pi05"]
    assert closed["status"] == "model_stopped"
    assert closed["model_service"]["status"] == "stopped"


def test_close_model_still_rejects_when_another_task_is_running(
    control_plane: ControlPlane,
) -> None:
    completed = control_plane.create_deployment(valid_payload(), control_plane.csrf_token)
    control_plane.wait_for_terminal(completed["id"], timeout=2)
    active = control_plane.create_deployment(
        valid_payload(model_id="pi05"), control_plane.csrf_token
    )
    control_plane.wait_for_terminal(active["id"], timeout=2)
    control_plane._jobs[active["id"]].status = "running"

    with pytest.raises(ValidationError, match="运行中的任务"):
        control_plane.close_model(completed["id"], control_plane.csrf_token)


def test_read_only_robot_command_is_allowlisted_and_records_camera_trace(
    control_plane: ControlPlane,
) -> None:
    config = control_plane.validate(valid_payload(hardware_mode="read-only"))["normalized"]
    command, session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        "a" * 32,
        config,
    )

    assert "--read-only" in command
    assert "--execute" not in command
    assert any(item.startswith("--action-trace-dir=") for item in command)
    assert any(item.startswith("--control-file=") for item in command)
    assert any(item.startswith("--control-status-file=") for item in command)
    assert f"--host={control_plane.deployment.robots['g2a'].policy_host}" in command
    assert "--source-address=10.42.1.102" in command
    assert session_directory == (
        f"/home/agi/.cache/nexus-platform-sessions/{'a' * 32}"
    )


def test_read_only_robot_command_accepts_one_way_tcp_platform_telemetry(
    control_plane: ControlPlane,
) -> None:
    job_id = "a" * 32
    config = control_plane.validate(valid_payload(hardware_mode="read-only"))["normalized"]
    endpoint = (
        f"tcp://192.168.110.140:8768/api/deployments/{job_id}/robot-telemetry"
    )
    config["_platform_telemetry_endpoint"] = endpoint
    config["_platform_telemetry_token"] = "token-value-long-enough-for-auth"

    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        job_id,
        config,
    )

    assert f"--platform-telemetry-endpoint={endpoint}" in command
    assert "--platform-telemetry-fps=20" in command


def test_body_only_robot_command_passes_fixed_effector_observation_mode(
    control_plane: ControlPlane,
) -> None:
    config = control_plane.validate(
        valid_payload(
            robot_id="legacy",
            robot_host="10.42.0.56",
            hardware_mode="read-only",
        )
    )["normalized"]
    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["legacy"],
        control_plane.deployment.models["pi05"],
        "c" * 32,
        config,
    )

    assert "--effectors-mode=absent_fixed" in command
    assert "--effector-profile=omnipicker_legacy" in command
    assert "--fixed-left-effector-state=-0.7" in command
    assert "--fixed-right-effector-state=-0.71" in command


def test_execute_robot_command_requires_and_passes_calibrated_manifest_profile(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)
    config = control_plane.validate(execute_payload())["normalized"]
    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        "b" * 32,
        config,
    )

    assert "--execute" in command
    assert not any(item.startswith("--action-trace-dir=") for item in command)
    assert (
        "--safety-config=/home/agi/deploy/g2a_private_client/robot_client/"
        "workspace_safety.calibrated.json"
    ) in command


@pytest.mark.parametrize("inference_mode", ["sync", "async", "rtc"])
def test_hitl_tablet_start_uses_selected_inference_mode_and_initial_recording(
    control_plane: ControlPlane, inference_mode: str
) -> None:
    unlock_test_execute(control_plane)
    config = control_plane.validate(execute_payload(inference_mode=inference_mode))["normalized"]
    config["_tablet_record_name"] = "L-9.2_20261016120000"

    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        "f" * 32,
        config,
    )

    assert command[1] == "--human-in-loop"
    assert f"--inference-mode={inference_mode}" in command
    assert "--initial-vendor-record-name=L-9.2_20261016120000" in command
    assert "--enable-rollback" in command
    assert "--enable-vendor-ee-handoff" in command


def test_read_only_robot_command_passes_read_only_ready_safety_profile(
    control_plane: ControlPlane,
) -> None:
    control_plane.deployment.robots["g2a"].manifest.write_text(
        "workspace_safety.read_only.json\n", encoding="utf-8"
    )
    config = control_plane.validate(
        valid_payload(
            safety={
                "workspace_safety_enabled": True,
                "safety_profile_id": "workspace_safety.read_only.json",
            }
        )
    )["normalized"]

    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        "e" * 32,
        config,
    )

    assert (
        "--safety-config=/home/agi/deploy/g2a_private_client/robot_client/"
        "workspace_safety.read_only.json"
    ) in command
    assert "--safety-monitor-only" not in command


def test_execute_robot_command_allows_collision_layer_to_be_omitted(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)
    payload = execute_payload(human_in_loop=False)
    payload["safety"] = {
        **payload["safety"],
        "workspace_safety_enabled": False,
        "safety_profile_id": "",
    }
    config = control_plane.validate(payload)["normalized"]

    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        "d" * 32,
        config,
    )

    assert "--execute" in command
    assert not any(item.startswith("--safety-config=") for item in command)
    assert "--safety-monitor-only" not in command


def test_execute_robot_command_supports_collision_monitoring_without_interception(
    control_plane: ControlPlane,
) -> None:
    unlock_test_execute(control_plane)
    payload = execute_payload(human_in_loop=False)
    payload["safety"] = {
        **payload["safety"],
        "workspace_safety_enabled": False,
    }
    config = control_plane.validate(payload)["normalized"]

    command, _session_directory = control_plane.deployment.robot_session_command(
        control_plane.deployment.robots["g2a"],
        control_plane.deployment.models["pi05"],
        "e" * 32,
        config,
    )

    assert any(item.startswith("--safety-config=") for item in command)
    assert "--safety-monitor-only" in command


def test_action_enabled_start_waits_for_real_robot_client(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)
    monkeypatch.setattr(control_plane.deployment, "start_model", lambda *_args, **_kwargs: 1234)
    created = control_plane.create_deployment(
        valid_payload(server_mode="mock"), control_plane.csrf_token
    )
    control_plane.wait_for_terminal(created["id"], timeout=2)
    select_ready_scheme(control_plane, created["id"])
    monkeypatch.setattr(control_plane, "_safe_status", lambda *_: {"healthy": True})
    started: list[str] = []
    monkeypatch.setattr(
        control_plane.deployment,
        "start_robot_session",
        lambda _robot, _model, session_id, _config: started.append(session_id) or 4321,
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "wait_robot_session_ready",
        lambda _robot, _session_id, **_kwargs: {"pid": 4321, "running": True},
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_session_snapshot",
        lambda _robot, _session_id: {"pid": 4321, "running": True},
    )

    control_plane.start_deployment(created["id"], control_plane.csrf_token)
    running = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert running["status"] == "running"
    assert running["task_session"]["remote_started"] is True
    assert running["task_session"]["remote_pid"] == 4321
    assert started == [created["id"]]


def test_action_enabled_soft_pause_keeps_same_robot_client_and_uses_acknowledged_control(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)
    monkeypatch.setattr(control_plane.deployment, "start_model", lambda *_args, **_kwargs: 1234)
    created = control_plane.create_deployment(
        valid_payload(server_mode="mock"), control_plane.csrf_token
    )
    control_plane.wait_for_terminal(created["id"], timeout=2)
    select_ready_scheme(control_plane, created["id"])
    monkeypatch.setattr(control_plane, "_safe_status", lambda *_: {"healthy": True})
    monkeypatch.setattr(
        control_plane.deployment,
        "start_robot_session",
        lambda _robot, _model, _session_id, _config: 4321,
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "wait_robot_session_ready",
        lambda _robot, _session_id, **_kwargs: {"pid": 4321, "running": True},
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_session_snapshot",
        lambda _robot, _session_id: {"pid": 4321, "running": True},
    )
    transitions: list[bool] = []
    monkeypatch.setattr(
        control_plane.deployment,
        "set_robot_session_paused",
        lambda _robot, _session_id, paused: transitions.append(paused)
        or {"state": "paused" if paused else "running"},
    )
    stopped: list[str] = []
    monkeypatch.setattr(
        control_plane.deployment,
        "stop_robot_session",
        lambda _robot, session_id: stopped.append(session_id)
        or {"status": "ready", "session_id": session_id, "path": "/archive/session"},
    )

    control_plane.start_deployment(created["id"], control_plane.csrf_token)
    assert control_plane.wait_for_terminal(created["id"], timeout=2)["status"] == "running"
    paused = control_plane.pause_deployment(created["id"], control_plane.csrf_token)
    resumed = control_plane.resume_deployment(created["id"], control_plane.csrf_token)

    assert paused["status"] == "paused"
    assert paused["task_session"]["remote_pid"] == 4321
    assert paused["task_session"]["remote_started"] is True
    assert resumed["status"] == "running"
    assert resumed["task_session"]["remote_pid"] == 4321
    assert transitions == [True, False]
    assert stopped == []

    ended = control_plane.stop_deployment(created["id"], control_plane.csrf_token)
    assert stopped == [created["id"]]
    assert ended["task_session"]["log_archive"]["status"] == "ready"
    assert ended["task_session"]["log_archive"]["path"] == "/archive/session"


def test_read_only_preflight_treats_ptp_permission_as_runtime_warning(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_runtime_preflight",
        lambda _robot: {
            "user": "agi",
            "ptp_devices": [
                {"path": "/dev/ptp_xgi0", "readable": False, "writable": False}
            ],
            "gdk_processes": [
                "gdk_service",
                "gdk_http_server",
                "camera_service",
                "fastdds discovery",
            ],
        },
    )

    result = control_plane.deployment.require_robot_read_only_ready(
        control_plane.deployment.robots["g2a"]
    )

    assert result["ptp_ready"] is False
    assert "仅记录为警告" in result["warnings"][0]


def test_read_only_preflight_warns_but_does_not_block_other_gdk_client(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_runtime_preflight",
        lambda _robot: {
            "user": "agi",
            "ptp_devices": [{"path": "/dev/ptp3", "readable": True, "writable": True}],
            "gdk_processes": [
                "gdk_service",
                "gdk_http_server",
                "camera_service",
                "fastdds discovery",
            ],
            "external_gdk_clients": [{"pid": 1234, "command": "python observer.py"}],
        },
    )

    result = control_plane.deployment.require_robot_read_only_ready(
        control_plane.deployment.robots["g2a"]
    )

    assert result["ptp_ready"] is True
    assert "只读观测可继续" in result["warnings"][0]


def test_read_only_preflight_requires_gdk_runtime_service(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control_plane.deployment,
        "robot_runtime_preflight",
        lambda _robot: {
            "user": "agi",
            "ptp_devices": [
                {"path": "/dev/ptp_xgi0", "readable": False, "writable": False}
            ],
            "gdk_processes": ["gdk_http_server", "camera_service", "fastdds discovery"],
        },
    )

    with pytest.raises(RuntimeError, match="GDK runtime"):
        control_plane.deployment.require_robot_read_only_ready(
            control_plane.deployment.robots["g2a"]
        )


def test_failed_robot_start_cleans_up_remote_client(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)
    monkeypatch.setattr(control_plane.deployment, "start_model", lambda *_args, **_kwargs: 1234)
    created = control_plane.create_deployment(
        valid_payload(server_mode="mock"), control_plane.csrf_token
    )
    control_plane.wait_for_terminal(created["id"], timeout=2)
    select_ready_scheme(control_plane, created["id"])
    monkeypatch.setattr(control_plane, "_safe_status", lambda *_: {"healthy": True})
    monkeypatch.setattr(
        control_plane.deployment,
        "start_robot_session",
        lambda _robot, _model, _session_id, _config: 4321,
    )
    monkeypatch.setattr(
        control_plane.deployment,
        "wait_robot_session_ready",
        lambda _robot, _session_id, **_kwargs: (_ for _ in ()).throw(RuntimeError("GDK failed")),
    )
    stopped: list[str] = []
    monkeypatch.setattr(
        control_plane.deployment,
        "stop_robot_session",
        lambda _robot, session_id: stopped.append(session_id),
    )

    control_plane.start_deployment(created["id"], control_plane.csrf_token)
    failed = control_plane.wait_for_terminal(created["id"], timeout=2)

    assert failed["status"] == "failed"
    assert failed["telemetry"]["inference_status"] == "not_started"
    assert failed["task_session"]["remote_started"] is False
    assert stopped == [created["id"]]


def test_action_jobs_reserve_a_model_until_terminal(
    control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_plane.enable_actions = True
    grant_ready_robot_probe(control_plane)

    def start_model(*_args: Any, **kwargs: Any) -> int:
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback("lingbot is loading: Loaded checkpoint shard 3/6")
        return 1234

    monkeypatch.setattr(control_plane.deployment, "start_model", start_model)
    first = control_plane.create_deployment(
        valid_payload(human_in_loop=True, server_mode="mock"),
        control_plane.csrf_token,
    )
    control_plane.wait_for_terminal(first["id"], timeout=2)
    progress = control_plane.get_job(first["id"])["model_service"]
    assert progress["shard_current"] == 3
    assert progress["shard_total"] == 6

    candidate = control_plane.create_deployment(
        valid_payload(human_in_loop=True, server_mode="real"),
        control_plane.csrf_token,
    )
    candidate = control_plane.wait_for_terminal(candidate["id"], timeout=2)
    assert candidate["model_service"]["candidate_only"]
    assert candidate["model_service"]["status"] == "candidate"

    control_plane.stop_deployment(first["id"], control_plane.csrf_token)
    another_candidate = control_plane.create_deployment(
        valid_payload(human_in_loop=True, server_mode="mock"),
        control_plane.csrf_token,
    )
    assert another_candidate["model_service"]["candidate_only"]
    control_plane.close_model(first["id"], control_plane.csrf_token)
    second = control_plane.create_deployment(
        valid_payload(human_in_loop=True, server_mode="mock"),
        control_plane.csrf_token,
    )
    assert second["id"] != first["id"]


def test_http_rejects_untrusted_host_and_framing(control_plane: ControlPlane) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(control_plane))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/bootstrap",
            headers={"Host": "attacker.example"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad, timeout=2)
        assert error.value.code == 403

        good = urllib.request.Request(
            f"http://127.0.0.1:{port}/",
            headers={"Host": "localhost"},
        )
        with urllib.request.urlopen(good, timeout=2) as response:
            assert response.headers["X-Frame-Options"] == "DENY"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]

        session_page = urllib.request.Request(
            f"http://127.0.0.1:{port}/session.html?id=test",
            headers={"Host": "localhost"},
        )
        with urllib.request.urlopen(session_page, timeout=2) as response:
            assert "运行设置" in response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_private_lan_handler_requires_basic_auth_and_exact_host(
    control_plane: ControlPlane,
) -> None:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler_factory(
            control_plane,
            allowed_hosts=("192.168.110.140",),
            basic_credentials=("nexus", "secret"),
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/api/bootstrap"
        unauthenticated = urllib.request.Request(
            url, headers={"Host": "192.168.110.140"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(unauthenticated, timeout=2)
        assert error.value.code == 401
        assert error.value.headers["WWW-Authenticate"].startswith("Basic ")

        authenticated = urllib.request.Request(
            url,
            headers={
                "Host": "192.168.110.140",
                "Authorization": (
                    "Basic " + base64.b64encode(b"nexus:secret").decode("ascii")
                ),
            },
        )
        with urllib.request.urlopen(authenticated, timeout=2) as response:
            assert json.load(response)["project"]["name"] == "Nexus"

        wrong_host = urllib.request.Request(
            url,
            headers={
                "Host": "attacker.example",
                "Authorization": "Basic YXd2ZDpzZWNyZXQ=",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(wrong_host, timeout=2)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_private_lan_unauthenticated_access_requires_explicit_cli_flag() -> None:
    args = build_parser().parse_args(
        [
            "--host",
            "192.168.110.140",
            "--allow-unauthenticated-private-lan",
        ]
    )

    assert args.host == "192.168.110.140"
    assert args.allow_unauthenticated_private_lan is True


def test_robot_telemetry_can_use_a_different_private_interface_from_ui() -> None:
    args = build_parser().parse_args(
        [
            "--host",
            "192.168.110.140",
            "--robot-telemetry-url",
            "tcp://10.42.1.200:8768",
            "--robot-telemetry-bind-host",
            "10.42.1.200",
        ]
    )

    assert args.robot_telemetry_url == "tcp://10.42.1.200:8768"
    assert args.robot_telemetry_bind_host == "10.42.1.200"


def test_restart_migrates_unreset_physical_failure_to_robot_gate(
    control_plane: ControlPlane,
) -> None:
    failed = DeploymentJob(
        id="e" * 32, status="failed",
        config=valid_payload(hardware_mode="execute"),
        task_session={"accepted": True, "status": "failed", "remote_started": False},
        batch={"source": "scheduler", "status": "failed"},
    )
    control_plane._jobs[failed.id] = failed
    control_plane._persist_state()
    old_state = json.loads(control_plane._state_path.read_text(encoding="utf-8"))
    old_state["workstation"].pop("reset_gate")
    control_plane._state_path.write_text(json.dumps(old_state), encoding="utf-8")

    recovered = ControlPlane(control_plane.config_path, enable_actions=False)
    assert recovered.tasks.reset_gate()["required"] is True
    assert recovered.tasks.state()["accepting"] is False
    with pytest.raises(TaskError) as denied:
        recovered.tasks.check_idle()
    assert denied.value.reason_code == "RESET_REQUIRED"
    recovered.close()


def test_restart_does_not_resurrect_tasks_resolved_by_later_verified_reset(
    control_plane: ControlPlane,
) -> None:
    reset = DeploymentJob(
        id="a" * 32, status="ready_to_start",
        config=valid_payload(hardware_mode="execute"),
        task_session={"status": "ready", "remote_started": False},
        robot_reset={"status": "succeeded"},
    )
    for index, status in enumerate(("failed", "ready_to_start"), start=1):
        old = DeploymentJob(
            id=str(index) * 32, status=status,
            config=valid_payload(hardware_mode="execute"),
            task_session={"accepted": True, "status": "failed", "remote_started": False,
                          "reset_resolved_by": reset.id},
            batch={"status": "recovery_required", "reset_required": False},
        )
        control_plane._jobs[old.id] = old
    control_plane._jobs[reset.id] = reset
    control_plane._workstation["reset_gate"] = {
        "required": False, "status": "ready", "source_job_id": reset.id,
    }
    control_plane._persist_state()

    recovered = ControlPlane(control_plane.config_path, enable_actions=False)
    try:
        assert recovered.tasks.state()["active_tasks"] == []
        assert recovered.tasks.state()["accepting"] is False
        for index in (1, 2):
            assert recovered._jobs[str(index) * 32].batch["status"] == "failed"
        assert all(view["status"] == "failed" for view in recovered.tasks.list_tasks())
    finally:
        recovered.close()


def test_restart_releases_ghost_model_reservation_when_owned_process_is_gone(
    control_plane: ControlPlane,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghost = DeploymentJob(
        id="ghost-model-job",
        status="ready_to_start",
        message="模型服务已就绪",
        config=valid_payload(server_mode="real"),
        model_service={
            "status": "ready",
            "healthy": True,
            "owned_by_job": True,
            "pgid": 424242,
            "gpu": "0",
        },
        task_session={"status": "ready"},
    )
    control_plane._jobs[ghost.id] = ghost
    control_plane._persist_state()
    monkeypatch.setattr(type(control_plane.deployment), "running_pid", lambda *_args: None)

    recovered = ControlPlane(control_plane.config_path, enable_actions=True)
    job = recovered.get_job(ghost.id)

    assert job["status"] == "failed"
    assert job["model_service"]["owned_by_job"] is False
    assert job["model_service"]["status"] == "stopped"
    assert "pi05" not in recovered._reserved_models
    assert "释放失效的模型占用" in job["events"][-1]["message"]


def test_restart_preserves_private_robot_telemetry_token(
    control_plane: ControlPlane,
) -> None:
    job = DeploymentJob(
        id="running-readonly-job",
        status="running",
        config=valid_payload(hardware_mode="read-only"),
        task_session={"status": "running"},
    )
    token = "private-telemetry-token-that-is-long-enough"
    control_plane._jobs[job.id] = job
    control_plane._telemetry_tokens[job.id] = token
    control_plane._persist_state()

    raw_state = json.loads(control_plane._state_path.read_text(encoding="utf-8"))
    assert raw_state["telemetry_tokens"] == {job.id: token}
    assert token not in json.dumps(job.to_dict())
    assert control_plane._state_path.stat().st_mode & 0o777 == 0o600

    recovered = ControlPlane(control_plane.config_path, enable_actions=False)
    assert recovered._telemetry_tokens[job.id] == token
