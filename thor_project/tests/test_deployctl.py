from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from nexus_deploy import deployment as deployctl


@pytest.fixture()
def deployment() -> deployctl.Deployment:
    return deployctl.Deployment(PROJECT_ROOT / "configs" / "deploy.test.toml")


@pytest.fixture()
def safety_session_config(deployment, tmp_path):
    from nexus_deploy.control_api import ControlPlane

    profile_id = "test-safety.calibrated.json"
    (tmp_path / profile_id).write_text(json.dumps(dict(calibrated=True, execute_approved=True)))
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(profile_id + "\n")
    robot = replace(deployment.robot, source_client=tmp_path, manifest=manifest)
    config = json.loads(json.dumps(ControlPlane.DEFAULTS))
    config["robot_host"] = deployment.robot_network_endpoints(robot.name)[0]["host"]
    config["robot_ssh_port"] = 22
    config["safety"].update(workspace_safety_enabled=True, safety_profile_id=profile_id)
    return robot, config


@pytest.mark.parametrize("hardware_mode", ["read-only", "execute"])
def test_enabled_collision_uses_segmented_legacy_queue_for_both_modes(
    deployment,
    safety_session_config,
    hardware_mode,
):
    robot, config = safety_session_config
    config["hardware_mode"] = hardware_mode
    command, _ = deployment.robot_session_command(robot, deployment.models["pi05"], "a" * 32, config)
    assert f"--{hardware_mode}" in command
    assert command.count("--rolling-safety") == 1
    assert command.count("--rolling-geometry=legacy") == 1
    assert "--async-safety" in command
    assert "--safety-check-fps=50" in command
    assert "--safety-monitor-only" not in command
    assert "--certified-braking" not in command
    assert command.count("--arm-limit-profile=urdf-velocity-uniform-acceleration") == 1
    assert "--max-arm-velocity=3.76" in command
    assert "--max-arm-acceleration=8.0" in command
    assert command.count("--chunk-smoothing-mode=savgol") == 1
    assert command.count("--action-interpolation=linear") == 1
    assert command.count("--chunk-smoothing-window=7") == 1
    assert command.count("--chunk-smoothing-max-deviation=0.005") == 1
    assert command.count("--adaptive-smoothing-acceleration-weight=20.0") == 1
    assert command.count("--adaptive-smoothing-jerk-weight=5.0") == 1
    assert command.count("--adaptive-smoothing-velocity-reference=0.25") == 1
    assert command.count("--effector-command-mode=endpoint-latch") == 1
    assert command.count("--effector-closed-max=0.0075") == 1
    assert command.count("--effector-open-min=0.0175") == 1
    assert "--fixed-horizon-projection" not in command
    assert "--coordinate-grasp-events" not in command


def test_operator_can_select_seven_point_savgol_chunk_smoothing(
    deployment,
    safety_session_config,
):
    robot, config = safety_session_config
    config["chunk_smoothing_mode"] = "savgol"

    command, _ = deployment.robot_session_command(
        robot, deployment.models["pi05"], "a" * 32, config
    )

    assert command.count("--chunk-smoothing-mode=savgol") == 1
    assert command.count("--chunk-smoothing-window=7") == 1


def test_operator_can_select_velocity_only_and_monotone_interpolation(
    deployment, safety_session_config,
):
    robot, config = safety_session_config
    config["arm_limit_profile"] = "urdf-velocity-only"
    config["action_interpolation"] = "monotone-cubic"
    command, _ = deployment.robot_session_command(
        robot, deployment.models["pi05"], "a" * 32, config
    )
    assert command.count("--arm-limit-profile=urdf-velocity-only") == 1
    assert command.count("--action-interpolation=monotone-cubic") == 1
    assert command.count("--max-arm-acceleration=8.0") == 1


def test_execute_experimental_arm_profile_is_explicit_and_scalar_capped(
    deployment,
    safety_session_config,
):
    robot, config = safety_session_config
    config.update(
        hardware_mode="execute",
        arm_limit_profile="dataset-v30-p99-margin",
        max_arm_velocity=2.2,
        max_arm_acceleration=6.5,
    )
    command, _ = deployment.robot_session_command(
        robot, deployment.models["pi05"], "a" * 32, config
    )
    assert "--arm-limit-profile=dataset-v30-p99-margin" in command
    assert "--max-arm-velocity=2.2" in command
    assert "--max-arm-acceleration=6.5" in command


@pytest.mark.parametrize("hardware_mode", ["read-only", "execute"])
@pytest.mark.parametrize("retain_profile", [False, True])
def test_disabled_collision_does_not_activate_segmented_queue(
    deployment,
    safety_session_config,
    hardware_mode,
    retain_profile,
):
    robot, config = safety_session_config
    config["hardware_mode"] = hardware_mode
    config["safety"]["workspace_safety_enabled"] = False
    if not retain_profile:
        config["safety"]["safety_profile_id"] = ""
    command, _ = deployment.robot_session_command(robot, deployment.models["pi05"], "a" * 32, config)
    assert "--rolling-safety" not in command
    assert "--rolling-geometry=legacy" not in command
    assert ("--safety-monitor-only" in command) is retain_profile
    assert "--async-safety" in command
    assert "--safety-check-fps=50" in command


@pytest.mark.parametrize("profile", [dict(calibrated=False), dict(calibrated=True, execute_approved=False)])
def test_segmented_execute_still_requires_approved_calibration(
    deployment,
    safety_session_config,
    profile,
):
    robot, config = safety_session_config
    config["hardware_mode"] = "execute"
    (robot.source_client / config["safety"]["safety_profile_id"]).write_text(json.dumps(profile))
    with pytest.raises(deployctl.DeployError, match="not approved for physical execution"):
        deployment.robot_session_command(robot, deployment.models["pi05"], "a" * 32, config)


def test_segmented_collision_requires_profile(deployment, safety_session_config):
    robot, config = safety_session_config
    config["safety"]["safety_profile_id"] = ""
    with pytest.raises(deployctl.DeployError, match="requires a calibrated profile"):
        deployment.robot_session_command(robot, deployment.models["pi05"], "a" * 32, config)


def test_restarting_session_archives_each_run_without_deleting_logs(deployment, tmp_path, monkeypatch):
    robot = replace(deployment.robot, remote_directory=str(tmp_path))
    session_dir = tmp_path / "sessions" / "same-task"
    session_dir.mkdir(parents=True)
    old_content = "first physical run: preserve\n"
    (session_dir / "client.log").write_text(old_content)
    monkeypatch.setattr(deployment, "ensure_robot_client_idle", lambda robot: None)
    monkeypatch.setattr(deployment, "robot_session_command", lambda *args: (["true"], str(session_dir)))
    commands = []

    def local_shell(command, **kwargs):
        commands.append(command)
        # Test only the filesystem launch wrapper in a pytest temporary tree.
        # The launched command is `true`: no SSH, client, model, or robot.
        return subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True)

    monkeypatch.setattr(deployment, "run_ssh", local_shell)
    for content in ("second run\n", "third run\n"):
        assert (
            deployment.start_robot_session(robot, deployment.models["pi05"], "same-task", {"hardware_mode": "dry-run"})
            > 1
        )
        (session_dir / "client.log").write_text(content)
    archives = list(session_dir.with_name("same-task.history").iterdir())
    assert len(archives) == 2
    assert {p.joinpath("client.log").read_text() for p in archives} == {old_content, "second run\n"}
    assert (session_dir / "client.log").read_text() == "third run\n"
    assert all("rm -rf" not in command for command in commands)


def test_robot_session_uses_reboot_persistent_cache(deployment):
    session_id = "a" * 32
    assert deployment.robot_session_directory(session_id) == (
        f"/home/agi/.cache/nexus-platform-sessions/{session_id}"
    )


def test_managed_alignment_worker_yields_gdk_before_robot_task(deployment, monkeypatch):
    commands = []

    def completed(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            ["ssh"], 0, stdout='{"released": true, "gdk_state": "idle"}\n', stderr=""
        )

    monkeypatch.setattr(deployment, "run_ssh", completed)

    assert deployment.release_managed_alignment_worker(deployment.robot, hold_s=15) is True
    assert len(commands) == 1
    assert "/api/alignment/release?hold_s=15" in commands[0][0]
    assert "-X POST" in commands[0][0]
    assert commands[0][1]["timeout_s"] == pytest.approx(8.0)


def test_robot_session_prelaunch_cleanup_prunes_only_old_archived_generations(
    deployment, tmp_path
):
    root = tmp_path / "home" / "agi" / ".cache" / "nexus-platform-sessions"
    root.mkdir(parents=True)
    deployment.robot_session_remote_root = str(root)
    deployment.robot_session_remote_retention = 1
    current = "a" * 32
    newest = root / ("b" * 32)
    oldest = root / ("c" * 32)
    unarchived = root / ("d" * 32)
    history = root / (("e" * 32) + ".history") / "generation"
    for path in (newest, oldest, unarchived, history):
        path.mkdir(parents=True)
    for path in (newest, oldest, history):
        (path / ".archived-to-server").touch()
    os.utime(oldest, (1, 1))
    os.utime(history, (2, 2))
    os.utime(newest, (3, 3))

    subprocess.run(
        ["bash", "-c", deployment._robot_session_prelaunch_cleanup(current)],
        check=True,
    )

    assert newest.is_dir()
    assert unarchived.is_dir()
    assert not oldest.exists()
    assert not history.exists()


def test_robot_session_archive_keeps_numerical_logs_and_excludes_images(
    deployment, tmp_path, monkeypatch
):
    session_id = "b" * 32
    deployment.robot_session_archive_root = tmp_path / "archives"
    remote_fixture = tmp_path / "remote"
    remote_fixture.mkdir()
    (remote_fixture / "client.log").write_text("client stopped cleanly\n")
    (remote_fixture / f"control-{session_id}.jsonl").write_text('{"record_type":"trace_end"}\n')
    (remote_fixture / "timing.jsonl").write_text('{"round_trip_ms":200}\n')
    (remote_fixture / "telemetry.token").write_text("secret")
    frames = remote_fixture / "inference-frames"
    frames.mkdir()
    (frames / "latest.json").write_text('{"request_id":"r1"}\n')
    (frames / "r1-cam_head.jpg").write_bytes(b"jpeg")
    remote_commands = []

    def fake_ssh(command, **_kwargs):
        remote_commands.append(command)
        return subprocess.CompletedProcess([], 0, "", "")

    def fake_run(command, **_kwargs):
        assert command[0] == "rsync"
        target = Path(command[-1])
        for source in remote_fixture.rglob("*"):
            relative = source.relative_to(remote_fixture)
            if source.is_dir():
                target.joinpath(relative).mkdir(parents=True, exist_ok=True)
            elif source.name not in {"telemetry.token", "client.pid"} and source.suffix.lower() not in {
                ".jpg", ".jpeg", ".png"
            }:
                target.joinpath(relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / relative)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(deployment, "run_ssh", fake_ssh)
    monkeypatch.setattr(subprocess, "run", fake_run)
    archived = deployment.archive_robot_session(deployment.robot, session_id)

    destination = Path(archived["path"])
    assert archived["status"] == "ready"
    assert archived["images_included"] is False
    assert (destination / "client.log").is_file()
    assert (destination / f"control-{session_id}.jsonl").is_file()
    assert (destination / "timing.jsonl").is_file()
    assert (destination / "inference-frames/latest.json").is_file()
    assert not (destination / "telemetry.token").exists()
    assert not list(destination.rglob("*.jpg"))
    assert any(".archived-to-server" in command for command in remote_commands)


def test_alignment_restart_allows_slow_start_with_bounded_health_requests(deployment, tmp_path, monkeypatch):
    (tmp_path / "alignment_agent.py").write_text("# test payload\n")
    calls = []
    monkeypatch.setattr(deployment, "run_ssh", lambda command, **kwargs: calls.append((command, kwargs)))
    deployment._restart_robot_background_services(deployment.robot, tmp_path)
    assert len(calls) == 1
    command, options = calls[0]
    assert command.count("--connect-timeout 1 --max-time 1") == 2
    assert command.count("18 19 20; do") == 2
    assert options["timeout_s"] == 45.0
    assert "gdk_service" not in command
    assert "teleop_main_node" not in command


def test_dual_model_commands_use_separate_env_gpu_and_port(deployment: deployctl.Deployment) -> None:
    pi05 = deployment.models["pi05"]
    lingbot = deployment.models["lingbot"]
    assert pi05.conda_env == "lerobot"
    assert lingbot.conda_env == "lingbotvla"
    assert pi05.cuda_visible_devices != lingbot.cuda_visible_devices
    assert pi05.port != lingbot.port
    assert pi05.module in deployment.model_command(pi05, mock=False)
    assert lingbot.module in deployment.model_command(lingbot, mock=False)


def test_lingbot_command_uses_pi05_compatible_rtc_parameters(
    deployment: deployctl.Deployment,
) -> None:
    command = deployment.model_command(deployment.models["lingbot"], mock=False)
    assert "--execution-horizon=30" in command
    assert "--prefix-attention-schedule=EXP" in command
    assert "--max-guidance-weight=10.0" in command


def test_model_commands_use_configured_environment_python_directly(
    deployment: deployctl.Deployment,
) -> None:
    for model in deployment.models.values():
        command = deployment.model_command(model, mock=True)
        assert model.python_executable is not None
        assert command[:3] == [str(model.python_executable), "-m", model.module]
        assert "conda" not in Path(command[0]).name


def test_production_models_use_single_startup_warmup(deployment: deployctl.Deployment) -> None:
    for model in deployment.models.values():
        dahuan_command = deployment.model_command(model, mock=False, effector_profile="dahuan")
        original_command = deployment.model_command(model, mock=False, effector_profile="omnipicker_legacy")
        assert "--warmup-requests=1" in dahuan_command
        assert dahuan_command.count("--effector-profile=dahuan") == 1
        assert any(argument.startswith("--warmup-task=Pick a package from the box") for argument in dahuan_command)
        assert any(argument.startswith("--warmup-task=Pick up the package") for argument in original_command)


def test_pi05_dahuan_versions_are_allow_listed_and_select_exact_checkpoint(
    deployment: deployctl.Deployment,
) -> None:
    model = deployment.models["pi05"]
    variant = model.variant("dahuan")

    assert variant.default_version == "large_qr_clean_v2_seed1000_007500"
    assert tuple(variant.versions) == (
        "large_qr_clean_v2_seed1000_007500",
        "large_qr_clean_v2_seed1000_005000",
        "per_package_001500",
        "per_package_004500",
        "per_sample_20260820_001500",
        "per_sample_20260820_007500",
        "per_sample_20260824_005000",
        "per_sample_20260824_007500",
    )
    command = deployment.model_command(
        model,
        mock=False,
        effector_profile="dahuan",
        model_version="per_package_004500",
    )
    assert "--checkpoint=models/pi05/dahuan/per_package_20260819_20260915/checkpoint-004500" in command
    assert any(argument.startswith("--tokenizer-path=") for argument in command)
    default_command = deployment.model_command(model, mock=False, effector_profile="dahuan")
    assert "--checkpoint=models/pi05/dahuan/large_qr_clean_v2_seed1000/checkpoint-007500" in default_command
    old_command = deployment.model_command(model, mock=False, effector_profile="dahuan", model_version="large_qr_clean_v2_seed1000_005000")
    assert "--checkpoint=models/pi05/dahuan/large_qr_clean_v2_seed1000/checkpoint-005000" in old_command

    with pytest.raises(deployctl.DeployError, match="no model version arbitrary-checkpoint"):
        deployment.model_command(
            model,
            mock=False,
            effector_profile="dahuan",
            model_version="arbitrary-checkpoint",
        )


def test_legacy_dh_input_is_canonicalized_at_the_boundary(
    deployment: deployctl.Deployment,
) -> None:
    command = deployment.model_command(
        deployment.models["pi05"], mock=True, effector_profile="dh"
    )

    assert "--effector-profile=dahuan" in command
    assert "--effector-profile=dh" not in command


def test_model_environment_allows_per_start_gpu_override(
    deployment: deployctl.Deployment,
) -> None:
    model = deployment.models["pi05"]
    assert deployment.model_environment(model)["CUDA_VISIBLE_DEVICES"] == "0"
    assert deployment.model_environment(model, cuda_visible_devices="1")["CUDA_VISIBLE_DEVICES"] == "1"


def test_models_and_robots_are_discovered_from_configuration(
    deployment: deployctl.Deployment,
) -> None:
    assert tuple(deployment.models) == ("pi05", "lingbot")
    assert tuple(deployment.robots) == ("g2a",)
    assert deployment.select_robot("g2a") is deployment.robot
    g2a = deployment.select_robot("g2a")
    assert g2a.ssh_target == "agi@10.42.1.102"
    assert g2a.policy_host == "10.42.1.200"
    assert g2a.policy_source_host == "10.42.1.102"
    assert g2a.expected_aid == "G2A0004C201260"
    assert g2a.reset_script == "/home/agi/deploy/nexus_deploy/robot_client/reset_all_joints.sh"
    assert deployment.robot_network_endpoints("g2a") == (
        {
            "host": "10.42.1.102",
            "ssh_port": 22,
            "label": "Nexus 有线管理",
            "transport": "wired",
            "policy_host": "10.42.1.200",
            "policy_source_host": "10.42.1.102",
            "priority": 1,
        },
        {
            "host": "192.168.110.237",
            "ssh_port": 22,
            "label": "无线管理回退",
            "transport": "wireless",
            "policy_host": "192.168.110.140",
            "policy_source_host": "192.168.110.237",
            "priority": 2,
        },
    )
    wireless = deployment.select_robot_endpoint("g2a", "192.168.110.237", 22)
    assert wireless.ssh_target == "agi@192.168.110.237"
    assert wireless.policy_host == "192.168.110.140"
    assert wireless.policy_source_host == "192.168.110.237"
    assert deployment.robots["g2a"].effectors_mode == "required"
    assert deployment.robots["g2a"].effector_profile == "dahuan"
    assert deployment.models["pi05"].effector_profile == "dahuan"
    assert deployment.models["lingbot"].effector_profile == "dahuan"
    assert deployment.raw["models"]["lingbot"]["rtc_execute_enabled"] is True
    assert deployment.models["pi05"].normalization_source == "checkpoint_embedded"
    assert deployment.models["lingbot"].normalization_source == "external_training_asset"
    assert deployment.models["pi05"].effector_profiles == ("dahuan", "omnipicker_legacy")
    assert deployment.models["lingbot"].effector_profiles == ("dahuan", "omnipicker_legacy")
    assert deployment.effector_profiles["dahuan"].display_name == "大寰夹爪"
    assert deployment.effector_profiles["dahuan"].default_task.startswith(
        "Pick a package from the box with the left gripper"
    )
    assert deployment.effector_profiles["omnipicker_legacy"].display_name == "智元原装 Omnipicker 夹爪"
    assert deployment.effector_profiles["omnipicker_legacy"].default_task.startswith(
        "Pick up the package and place it on the conveyor belt"
    )
    assert deployment.effector_profiles["dahuan"].hardware_state_range == (0.0, 0.025)
    assert deployment.effector_profiles["dahuan"].action_mapping == "invert_about_endpoints"
    assert deployment.effector_profiles["dahuan"].collision_geometry_status == "provisional"
    assert deployment.effector_profiles["dahuan"].collision_geometry == "g2_dahuan"
    assert deployment.effector_profiles["dahuan"].default_max_velocity == pytest.approx(0.05)
    assert deployment.effector_profiles["dahuan"].default_tracking_error == pytest.approx(0.01)


def test_robot_reset_uses_only_the_explicit_dahuan_cycle_script(
    deployment: deployctl.Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = deployment.robots["g2a"]
    monkeypatch.setattr(deployment, "ensure_robot_client_idle", lambda _robot: None)
    monkeypatch.setattr(
        deployment,
        "robot_runtime_preflight",
        lambda _robot: {"asset_id": robot.expected_aid, "external_gdk_clients": []},
    )
    profile_id = "G2_dahuan_standard_table_test.json"
    monkeypatch.setattr(
        deployment,
        "workspace_safety_profile",
        lambda _robot, _profile_id: (
            {"calibrated": True, "execute_approved": True},
            Path("/tmp/profile.json"),
            f"{robot.remote_directory}/robot_client/{profile_id}",
            True,
        ),
    )
    captured: dict[str, str] = {}

    def run_ssh(command: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess([], 0, stdout="Reset completed", stderr="")

    monkeypatch.setattr(deployment, "run_ssh", run_ssh)

    result = deployment.reset_robot_with_dahuan_cycle(robot, profile_id)

    assert robot.reset_script in captured["command"]
    assert f"{robot.legacy_remote_directory}/robot_client/reset_all_joints.sh" not in captured["command"]
    assert result["completed"] is True
    assert "--cycle-effectors" in captured["command"]
    assert "--effector-profile=dahuan" in captured["command"]
    assert "--safety-config" in captured["command"]
    assert profile_id in captured["command"]
    assert result["effectors_commanded"] is True
    assert result["effector_sequence"] == ["open", "close", "open"]


def test_robot_identity_check_fails_closed_for_wrong_physical_robot(
    deployment: deployctl.Deployment,
) -> None:
    robot = deployment.robots["g2a"]
    deployment.require_robot_identity(robot, {"asset_id": robot.expected_aid})

    with pytest.raises(deployctl.DeployError, match="硬件身份不匹配"):
        deployment.require_robot_identity(robot, {"asset_id": "G2A0004C201319"})
    with pytest.raises(deployctl.DeployError, match="无法读取机器人硬件身份"):
        deployment.require_robot_identity(robot, {})


def test_mock_commands_do_not_contain_real_checkpoints(deployment: deployctl.Deployment) -> None:
    for model in deployment.models.values():
        command = deployment.model_command(model, mock=True)
        assert "--mock" in command
        assert not any(argument.startswith("--checkpoint=") for argument in command)
        assert not any(argument.startswith("--model-path=") for argument in command)


def test_mock_process_environment_hides_all_gpus(deployment: deployctl.Deployment) -> None:
    model = deployment.models["pi05"]

    assert deployment.model_environment(model, cuda_visible_devices="")["CUDA_VISIBLE_DEVICES"] == ""


def test_real_commands_include_expected_checkpoints(deployment: deployctl.Deployment) -> None:
    pi05 = deployment.model_command(deployment.models["pi05"], mock=False)
    lingbot = deployment.model_command(deployment.models["lingbot"], mock=False)
    assert any(argument.startswith("--checkpoint=") for argument in pi05)
    assert any(argument.startswith("--model-path=") for argument in lingbot)
    assert any(argument.startswith("--qwen3vl-path=") for argument in lingbot)


def test_robot_source_passes_hardware_free_dry_run_check(deployment: deployctl.Deployment) -> None:
    deployment.verify_dry_run_source(deployment.robot.source_client)


def test_bundle_is_complete_and_declares_generic_protocol(deployment: deployctl.Deployment, tmp_path: Path) -> None:
    object.__setattr__(deployment.robot, "bundle_directory", tmp_path / "bundle")
    bundle = deployment.build_robot_bundle()
    metadata = json.loads((bundle / "BUNDLE_INFO.json").read_text(encoding="utf-8"))
    assert metadata["protocol"] == {"name": "nexus-policy-websocket", "version": 2}
    assert {"name": "g2a-policy-websocket", "version": 2} in metadata["protocol_aliases"]
    assert metadata["required_cli_options"] == [
        "--source-address",
        "--effectors-mode",
        "--effector-profile",
        "--effector-control-mode",
        "--effector-command-mode",
        "--effector-closed-max",
        "--effector-open-min",
        "--fixed-left-effector-state",
        "--fixed-right-effector-state",
        "--control-file",
        "--control-status-file",
        "--platform-telemetry-endpoint",
        "--platform-telemetry-token-file",
        "--platform-frame-dir",
        "--platform-presence-frame-endpoint",
        "--platform-presence-frame-token-file",
        "--platform-presence-job-id",
        "--platform-presence-generation",
        "--platform-presence-frame-fps",
        "--start-paused",
        "--stationary-hold-feedback-max-age-ms",
        "--safety-monitor-only",
        "--async-safety",
            "--rolling-safety",
            "--rolling-geometry",
            "--chunk-smoothing-mode",
            "--chunk-smoothing-window",
            "--chunk-smoothing-max-deviation",
            "--adaptive-smoothing-acceleration-weight",
            "--adaptive-smoothing-jerk-weight",
            "--adaptive-smoothing-velocity-reference",
            "--fixed-horizon-projection",
            "--arm-limit-profile",
            "--arm-tracking-mode",
            "--arm-tracking-omega",
            "--rtc-prefix-mode",
        ]
    assert metadata["capabilities"]["effector_control"] == {
        "version": 2,
        "modes": ["software-trajectory", "device-position"],
            "device_position_profile": "dahuan",
        "opening_geometry": "measured_feedback_opening",
        "policy_sampling": "held_goal_per_policy_interval",
        "command_hz": 100,
        "dahuan_command_shaping": "endpoint_latch_7p5mm_17p5mm",
    }
    assert metadata["capabilities"]["safety_runtime"] == {
                "version": 15,
        "configured_servo_hz": 100,
        "collision_check_hz": 50,
        "collision_check_rate_semantics": "upper_limit",
        "checked_segment_steps": 5,
        "startup_checked_steps": 10,
        "refill_low_water_steps": 8,
        "max_buffer_steps": 13,
        "queue_refill": "append_only",
        "geometry_mode": "fixed_stride2_sampled_segment",
        "surface_geometry": "calibrated_plane_origin_and_normal",
        "inter_arm_geometry": "cross_chain_capsules_v2_broad_narrow",
        "kinematic_authority": "canonical_crsb_v4",
        "inter_arm_geometry_source": "canonical_collision_overlay",
        "inter_arm_waypoint_coverage": "fixed_stride2_50hz_targets",
        "inter_arm_validation_gate": "profile_scoped",
        "collision_action_stride": 2,
        "geometry_reference_max_age_ms": 200,
        "motion_feedback_max_age_ms": 40,
        "stationary_hold_feedback_max_age_ms": 100,
        "dispatch_deadline_ms": 200,
        "safety_worker": "process",
        "worker_check_budget_ms": 20,
        "worker_check_budget_policy": "diagnostic_not_admission",
        "hold_revalidation": "historical_target_with_worker_liveness",
        "danger_response": "hold_last_accepted_and_continue",
        "device_position_effector_tracking": "opening_failure_fatal_closing_contact_warning",
        "escape_policy": "sampled_per_vertex_nonworsening_actual_gripper",
        "projection_adoption": False,
        "soft_pause": "measured_reanchor",
        "session_generation_fencing": True,
    }
    assert metadata["capabilities"]["body_reset"] == {
        "available": True,
        "entrypoint": "robot_client/reset_all_joints.sh",
        "profile_id": "g2_sorting_home_20260817_v1",
        "commanded_body_joints": 22,
        "effectors_read": True,
        "effectors_commanded": True,
        "effector_profile": "dahuan",
        "effector_backend": "move_ee_pos(dual_tool,dahuan,bounded_reissue)",
        "effector_sequence": ["open", "close", "open"],
        "effector_cycle_pause_s": 0.15,
        "effector_reissue_s": 0.75,
        "effector_max_commands": 3,
        "phase_feedback_required": True,
        "failure_final_open_recovery": True,
        "max_stage_joint_travel_rad": 1.0,
        "max_total_joint_travel_rad": 4.0,
        "body_path": "workspace_checked_piecewise_linear_joint_space",
        "workspace_path_samples": 64,
        "workspace_surface": "calibrated_plane_origin_and_normal",
    }
    assert metadata["start_state"] == {
        "profile_id": "g2_sorting_home_20260817_v1",
        "symbol": "APPROVED_START_STATE_24D",
        "canonical_dim": 24,
        "legacy_aliases": ["DATASET_START_STATE_24D", "BODY_RESET_TARGET_24D"],
    }
    assert metadata["background_services"]["alignment_agent"]["service"] == ("nexus-g2-alignment-agent.service")
    assert metadata["background_services"]["alignment_agent"]["idle_release_s"] == 60
    assert (bundle / "alignment_agent.py").is_file()
    assert (bundle / "alignment-camera.json").is_file()
    reset_source = (bundle / "robot_client" / "reset_to_dataset_start.py").read_text(encoding="utf-8")
    client_source = (bundle / "robot_client" / "client.py").read_text(encoding="utf-8")
    telemetry_source = (bundle / "robot_client" / "telemetry.py").read_text(encoding="utf-8")
    reset_constants = (bundle / "robot_client" / "constants.py").read_text(encoding="utf-8")
    assert "Configured active duration" in client_source
    assert "配置运行时长已到，客户端自动软暂停" in client_source
    assert "duration_auto_paused = True" in client_source
    assert "--platform-frame-dir" in client_source
    assert "LatestInferenceFrameWriter" in client_source
    assert "class LatestInferenceFrameWriter" in telemetry_source
    assert "BODY_RESET_TARGET_24D" in reset_source
    assert 'APPROVED_START_STATE_PROFILE_ID = "g2_sorting_home_20260817_v1"' in reset_constants
    assert "BODY_RESET_PROFILE_ID = APPROVED_START_STATE_PROFILE_ID" in reset_constants
    assert "DATASET_START_STATE_24D = APPROVED_START_STATE_24D" in reset_constants
    assert (bundle / "robot_client" / "run.sh").stat().st_mode & 0o111
    assert (bundle / "robot_client" / "grasp_coordinator.py").is_file()
    assert (bundle / "robot_client" / "wheels").is_dir()
    assert (bundle / "robot_client" / "assets/agibot_g2/G2_omnipicker_fixed_dual.urdf").is_file()
    assert (bundle / "robot_client" / "assets/agibot_g2/SOURCE.json").is_file()


def test_bundle_rejects_manifest_missing_imported_module(deployment: deployctl.Deployment, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        deployment.robot.manifest.read_text(encoding="utf-8").replace("grasp_coordinator.py\n", ""),
        encoding="utf-8",
    )
    robot = replace(deployment.robot, manifest=manifest, bundle_directory=tmp_path / "bundle")
    with pytest.raises(deployctl.DeployError, match="grasp_coordinator"):
        deployment.build_robot_bundle(robot)


@pytest.mark.parametrize(("include_safety_runtime", "valid"), [(False, False), (True, True), ("old_v4", False)])
def test_robot_probe_requires_current_fail_closed_safety_runtime(
    deployment: deployctl.Deployment,
    monkeypatch: pytest.MonkeyPatch,
    include_safety_runtime: bool | str,
    valid: bool,
) -> None:
    capabilities: dict[str, object] = {
        "body_reset": {
            "available": True,
            "entrypoint": "robot_client/reset_all_joints.sh",
            "profile_id": "g2_sorting_home_20260817_v1",
            "commanded_body_joints": 22,
            "effectors_read": True,
            "effectors_commanded": True,
            "effector_profile": "dahuan",
            "effector_sequence": ["open", "close", "open"],
            "max_stage_joint_travel_rad": 1.0,
            "max_total_joint_travel_rad": 4.0,
            "body_path": "workspace_checked_piecewise_linear_joint_space",
            "workspace_path_samples": 64,
            "workspace_surface": "calibrated_plane_origin_and_normal",
        }
    }
    if include_safety_runtime:
        capabilities["safety_runtime"] = {
            "version": 15,
            "configured_servo_hz": 100,
            "collision_check_hz": 50,
            "collision_check_rate_semantics": "upper_limit",
            "checked_segment_steps": 5,
            "startup_checked_steps": 10,
            "refill_low_water_steps": 8,
            "max_buffer_steps": 13,
            "queue_refill": "append_only",
            "geometry_mode": "fixed_stride2_sampled_segment",
            "surface_geometry": "calibrated_plane_origin_and_normal",
            "inter_arm_geometry": "cross_chain_capsules_v2_broad_narrow",
            "kinematic_authority": "canonical_crsb_v4",
            "inter_arm_geometry_source": "canonical_collision_overlay",
            "inter_arm_waypoint_coverage": "fixed_stride2_50hz_targets",
            "inter_arm_validation_gate": "profile_scoped",
            "collision_action_stride": 2,
            "geometry_reference_max_age_ms": 200,
            "motion_feedback_max_age_ms": 40,
            "stationary_hold_feedback_max_age_ms": 100,
            "dispatch_deadline_ms": 200,
            "safety_worker": "process",
            "worker_check_budget_ms": 20,
            "worker_check_budget_policy": "diagnostic_not_admission",
            "hold_revalidation": "historical_target_with_worker_liveness",
            "danger_response": "hold_last_accepted_and_continue",
            "device_position_effector_tracking": "opening_failure_fatal_closing_contact_warning",
            "escape_policy": "sampled_per_vertex_nonworsening_actual_gripper",
            "projection_adoption": False,
            "soft_pause": "measured_reanchor",
            "session_generation_fencing": True,
        }
        if include_safety_runtime == "old_v4":
            capabilities["safety_runtime"]["version"] = 4
            capabilities["safety_runtime"]["checked_segment_steps"] = 2
    response = {
        "installed": True,
        "manifest_complete": True,
        "missing_cli_options": [],
        "run_sh_executable": True,
        "processes": [],
        "protocol": {"name": "nexus-policy-websocket", "version": 2},
        "start_state": {
            "profile_id": "g2_sorting_home_20260817_v1",
            "symbol": "APPROVED_START_STATE_24D",
            "canonical_dim": 24,
            "legacy_aliases": ["DATASET_START_STATE_24D", "BODY_RESET_TARGET_24D"],
        },
        "capabilities": capabilities,
        "bundle_robot": "g2a",
    }
    monkeypatch.setattr(
        deployctl.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=json.dumps(response), stderr=""),
    )

    result = deployment.probe_robot_client()

    assert result["valid"] is valid
    assert result["safety_runtime_valid"] is valid


@pytest.mark.parametrize(
    ("protocol", "supported"),
    (
        ({"name": "nexus-policy-websocket", "version": 2}, True),
        ({"name": "g2a-policy-websocket", "version": 2}, True),
        ({"name": "g2a-pi05-websocket", "version": 1}, True),
        ({"name": "unknown-policy-websocket", "version": 2}, False),
        ({"name": "nexus-policy-websocket", "version": 99}, False),
    ),
)
def test_policy_protocol_migration_allowlist(protocol: dict[str, object], supported: bool) -> None:
    assert deployctl.is_supported_policy_protocol(protocol) is supported


@pytest.mark.parametrize(
    "path",
    ("/", "/home", "/home/agi", "/home/agi/deploy", "/tmp/client", "relative/client"),
)
def test_rejects_broad_or_non_robot_remote_paths(path: str) -> None:
    with pytest.raises(deployctl.DeployError):
        deployctl.Deployment._validate_remote_directory(path)


def test_accepts_specific_robot_remote_path() -> None:
    assert (
        deployctl.Deployment._validate_remote_directory("/home/agi/deploy/g2a_robot_client")
        == "/home/agi/deploy/g2a_robot_client"
    )


def test_robot_backup_pruning_keeps_two_newest_exact_targets() -> None:
    targets = deployctl.Deployment._validated_backup_prune_targets(
        "/home/agi/deploy/g2a_robot_client",
        [
            "g2a_robot_client.backup_20260910_095810",
            "g2a_robot_client.backup_20260909_165651",
            "g2a_robot_client.backup_20260911_100000",
        ],
        keep=2,
    )

    assert targets == ["/home/agi/deploy/g2a_robot_client.backup_20260909_165651"]


def test_robot_backup_pruning_enumerates_only_timestamp_backups(
    deployment: deployctl.Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def run_ssh(command: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(deployment, "run_ssh", run_ssh)
    assert deployment._prune_robot_backups(deployment.robot, keep=2) == []
    assert "backup_*" not in calls[0]
    assert calls[0].count("[0-9]") == 14


@pytest.mark.parametrize(
    "name",
    (
        "g2a_robot_client",
        "g2a_robot_client.backup_latest",
        "../g2a_robot_client.backup_20260909_165651",
        "g2a_robot_client.backup_20260909_165651/child",
    ),
)
def test_robot_backup_pruning_rejects_unexpected_names(name: str) -> None:
    with pytest.raises(deployctl.DeployError, match="Unsafe robot backup"):
        deployctl.Deployment._validated_backup_prune_targets(
            "/home/agi/deploy/g2a_robot_client",
            [name],
            keep=2,
        )


def test_dry_run_duration_is_bounded_before_network(
    deployment: deployctl.Deployment,
) -> None:
    with pytest.raises(deployctl.DeployError, match="duration"):
        deployment.test_robot_dry_run(deployment.models["pi05"], duration_s=31)


def test_running_pid_tracks_python_child_after_conda_leader_exits(
    deployment: deployctl.Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = deployment.models["pi05"]
    monkeypatch.setattr(deployment, "read_pid", lambda _model: 4100)
    monkeypatch.setattr(
        deployment,
        "process_group_members",
        lambda _pgid: [(4101, f"python -m {model.module}", "S")],
    )

    assert deployment.running_pid(model) == 4100


def test_process_group_stop_escalates_and_removes_pid_file(
    deployment: deployctl.Deployment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = deployment.models["pi05"]
    object.__setattr__(deployment, "runtime_dir", tmp_path)
    deployment.pid_file(model).write_text("4100\n", encoding="utf-8")
    monkeypatch.setattr(
        deployment,
        "process_group_members",
        lambda _pgid: [(4101, f"python -m {model.module}", "S")],
    )
    remaining = iter((True, False))
    monkeypatch.setattr(deployment, "process_group_exists", lambda _pgid: next(remaining))
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(deployctl.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    deployment._terminate_model_group(model, 4100, term_timeout_s=0, kill_timeout_s=0)

    assert sent == [(4100, signal.SIGTERM), (4100, signal.SIGKILL)]
    assert not deployment.pid_file(model).exists()


def test_start_timeout_cleans_the_entire_model_group(
    deployment: deployctl.Deployment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = deployment.models["pi05"]
    object.__setattr__(deployment, "runtime_dir", tmp_path)

    class FakeProcess:
        pid = 4200

    monkeypatch.setattr(deployctl.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        deployment,
        "wait_healthy",
        lambda *args, **kwargs: (_ for _ in ()).throw(deployctl.DeployError("timeout")),
    )
    cleaned: list[int] = []
    monkeypatch.setattr(
        deployment,
        "_terminate_model_group",
        lambda _model, pgid: cleaned.append(pgid),
    )

    with pytest.raises(deployctl.DeployError, match="timeout"):
        deployment.start_model(model, mock=True, wait_timeout_s=0.01)

    assert cleaned == [4200]
