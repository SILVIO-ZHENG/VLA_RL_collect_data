"""Host-independent configuration and dispatch boundary regressions (no GDK)."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nexus_deploy.control_api import ControlPlane
from nexus_deploy.deployment import Deployment
from nexus_deploy.robot.g2a import gdk_adapter as adapter_module
from nexus_deploy.robot.g2a.client import RobotClientRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def do_not_load_live_control_state(monkeypatch):
    # These tests inspect site configuration only. Never read or reconcile
    # the running site's durable state while the test suite is executing.
    monkeypatch.setattr(ControlPlane, "_load_state", lambda self: [])


@pytest.mark.parametrize("site,age", [("5090", 100), ("thor", 40)])
def test_both_sites_use_the_same_source_and_explicit_hold_policy(site, age):
    deployment = Deployment(ROOT / "configs" / f"{site}.toml")
    assert deployment.robot.source_client == ROOT / "src/nexus_deploy/robot/g2a"
    assert deployment.robot.manifest == ROOT / "robot_manifest.txt"
    assert deployment.models["pi05"].repo == ROOT
    assert deployment.models["pi05"].module == "nexus_deploy.vla.pi05.server"
    assert deployment.stationary_hold_feedback_max_age_ms == age
    assert "presence_stream.py" in deployment.manifest(deployment.robot)
    assert deployment.model_environment(deployment.models["pi05"])["PYTHONPATH"].split(":")[0] == str(ROOT)


def test_5090_defaults_to_published_multicapsule_collision_guard():
    plane = ControlPlane(ROOT / "configs/5090.toml")
    profile_id = plane.DEFAULTS["safety"]["safety_profile_id"]
    profile = next(
        item for item in plane._safety_profiles(plane.deployment.robot)
        if item["id"] == profile_id
    )
    assert plane.DEFAULTS["safety"]["workspace_safety_enabled"] is True
    assert profile["deployable"] and profile["read_only_ready"] and profile["execute_approved"]
    assert profile["inter_arm_collision"]["geometry"] == "cross-chain-capsules-v2"

    payload = {"robot_id": "g2a", "model_id": "pi05", "robot_host": "10.42.1.102"}
    defaulted = plane.validate(payload)
    assert defaulted["valid"] is True
    assert defaulted["normalized"]["safety"]["workspace_safety_enabled"] is True
    assert defaulted["normalized"]["safety"]["safety_profile_id"] == profile_id

    staging = plane.validate({
        **payload,
        "safety": {"workspace_safety_enabled": False, "safety_profile_id": ""},
    })
    assert staging["valid"] is True
    assert staging["normalized"]["safety"]["workspace_safety_enabled"] is False


@pytest.mark.parametrize("site", ["5090", "thor"])
def test_project_virtualenvs_are_not_resolved_to_base_python(site):
    plane = ControlPlane(ROOT / "configs" / f"{site}.toml")
    assert plane.bootstrap()["project"]["site_id"] == site
    assert plane.deployment.raw["host"]["web_python"] == "../.venv/bin/python"
    assert plane._presence_runtime.python_executable == ROOT / ".venv/bin/python"
    assert plane._presence_runtime.weights == ROOT / "models/vision/parcel/yolo26m-seg-final.pt"
    assert plane.deployment.models["pi05"].python_executable == ROOT / ".pi05-venv/bin/python"
    assert plane.deployment.models["pi05"].python_paths == ()


@pytest.mark.parametrize("site", ["5090", "thor"])
def test_pi05_assets_are_resolved_inside_this_project(site):
    model = Deployment(ROOT / "configs" / f"{site}.toml").models["pi05"]
    expected_checkpoints = {
        "dahuan": "models/pi05/dahuan/large_qr_clean_v2_seed1000/checkpoint-007500",
        "omnipicker_legacy": "models/pi05/omnipicker_legacy/full_30k/checkpoint-030000",
    }

    for name, variant in model.effector_variants.items():
        assert f"--checkpoint={expected_checkpoints[name]}" in variant.real_args
        asset_arguments = tuple(
            argument.split("=", 1)[1]
            for argument in variant.real_args
            if argument.startswith(("--checkpoint=", "--tokenizer-path="))
        )
        assert asset_arguments
        for value in asset_arguments:
            path = Path(value)
            assert not path.is_absolute()
            assert (ROOT / path).resolve().is_relative_to(ROOT.resolve())


@pytest.mark.parametrize("hold_limit", [0.04, 0.1])
@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("age", [0.02, 0.06, 0.12])
def test_motion_remains_40ms_and_only_hold_uses_site_policy(monkeypatch, hold_limit, hold, age):
    monkeypatch.setattr(adapter_module.time, "perf_counter", lambda: 100.0)
    adapter = object.__new__(adapter_module.G2AAdapter)
    adapter.config = SimpleNamespace(rolling_safety_enabled=True, rolling_legacy_geometry=True,
                                     safety_enforcement_enabled=True,
                                     stationary_hold_feedback_max_age_s=hold_limit)
    adapter._async_dispatch_action = np.zeros(24)
    adapter._async_dispatch_is_hold = hold
    adapter._async_worker = SimpleNamespace(feedback=(np.zeros(24), 100.0 - age))
    adapter._command_safety_started = 99.99
    limit = hold_limit if hold else 0.04
    if age < limit:
        adapter._validate_dispatch_age(100.0)
    else:
        with pytest.raises(adapter_module.SafetyDispatchExpiredError):
            adapter._validate_dispatch_age(100.0)


def test_expired_target_is_discarded_and_requests_existing_pause():
    runner = object.__new__(RobotClientRunner)
    calls = []
    runner.plan = SimpleNamespace(clear=lambda: calls.append("clear"))
    runner._resume_ready_event = threading.Event()
    runner._resume_ready_event.set()
    runner._pause_requested_event = threading.Event()
    runner._write_control_status = lambda *args: calls.append(args)
    runner._request_safety_expiry_pause(adapter_module.SafetyDispatchExpiredError("expired"))
    assert calls[0] == "clear"
    assert runner._pause_requested_event.is_set()
    assert not runner._resume_ready_event.is_set()
    assert calls[1][1] == "pausing"
