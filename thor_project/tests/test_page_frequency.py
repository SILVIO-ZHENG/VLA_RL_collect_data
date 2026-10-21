"""Verify saved page rates reach manual/Agent commands without robot motion."""
import copy
import uuid

import pytest

from test_control_api import control_plane, valid_payload
from nexus_deploy.control.errors import TaskError
from nexus_deploy.control.models import DeploymentJob


def selected_template(plane, fps):
    plane._presence_runtime = "offline"
    baseline = plane.validate(valid_payload(presence_gate={"enabled": True}))["normalized"]
    plane.tasks.register_profiles({"sort": copy.deepcopy(baseline)})
    config = copy.deepcopy(baseline)
    config["execution_fps"] = float(fps)
    job = DeploymentJob(uuid.uuid4().hex, config, status="ready_to_start")
    job.model_service = {"status": "ready", "healthy": False}
    job.task_session = {"status": "ready", "runtime_configured": True}
    plane._jobs[job.id] = job
    plane._workstation.update(active_deployment_id=job.id, mode="automatic", accepting=True)
    return job, baseline


@pytest.mark.parametrize("fps", [15, 25, 30, 20])
def test_agent_uses_saved_page_frequency(control_plane, fps, monkeypatch):
    p = control_plane
    selected, baseline = selected_template(p, fps)
    assert p.tasks.ready_template("sort") is selected
    assert p.tasks.profiles["sort"] == baseline  # no mutation of the fixed scheme
    monkeypatch.setattr(p, "start_deployment", lambda *args, **kwargs: None)
    view, created = p.tasks.submit({"request_id": "rate-test", "profile": "sort"}, source="scheduler")
    assert created
    admitted = p._jobs[view["task_id"]]
    assert admitted.config["execution_fps"] == fps
    selected.config["execution_fps"] = 12.0
    assert admitted.config["execution_fps"] == fps  # immutable session snapshot


def test_rate_inheritance_does_not_ignore_other_profile_fields(control_plane):
    job, _ = selected_template(control_plane, 15)
    job.config["task"] = "A different task"
    with pytest.raises(TaskError) as denied:
        control_plane.tasks.ready_template("sort")
    assert denied.value.reason_code == "NOT_READY"
    assert any(item["field"] == "task" for item in denied.value.config_mismatches)


def test_unsaved_runtime_rates_are_not_inherited(control_plane):
    job, _ = selected_template(control_plane, 15)
    job.task_session["runtime_configured"] = False
    with pytest.raises(TaskError):
        control_plane.tasks.ready_template("sort")


@pytest.mark.parametrize("fps", [15, 25, 30, 20])
def test_robot_command_uses_saved_rate(control_plane, fps):
    p = control_plane
    job, _ = selected_template(p, fps)
    robot = p.deployment.select_robot(job.config["robot_id"])
    command, _ = p.deployment.robot_session_command(robot, p.deployment.models[job.config["model_id"]], job.id, job.config)
    assert f"--execution-fps={float(fps)}" in command
    assert "--servo-fps=100.0" in command
