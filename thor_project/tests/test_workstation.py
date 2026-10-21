"""Entry routing and mode readiness, with isolated state and no hardware."""
import copy
import http.client
import json
import threading
import time
import uuid
from http.server import ThreadingHTTPServer

import pytest
from test_control_api import grant_ready_robot_probe, valid_payload

from nexus_deploy.control.errors import TaskError
from nexus_deploy.control.models import DeploymentJob
from nexus_deploy.control_api import _handler_factory

pytest_plugins = ("test_control_api",)


@pytest.fixture
def ready_job(control_plane):
    # Offline vision is sufficient for these isolated mode tests, including
    # tests that flip the hardware-action flag without contacting a robot.
    control_plane._presence_runtime = "offline"
    config = control_plane.validate(valid_payload(presence_gate={"enabled": True}))["normalized"]
    job = DeploymentJob(uuid.uuid4().hex, config, status="ready_to_start")
    job.model_service = {"status": "ready", "healthy": False}
    job.task_session = {"status": "ready"}
    control_plane._jobs[job.id] = job
    return job


def test_entry_prioritizes_active_and_recovery_without_selecting_arbitrary_models(control_plane, ready_job):
    p = control_plane
    assert p.tasks.state()["landing_url"] == f"/monitor?id={ready_job.id}"
    other = copy.deepcopy(ready_job)
    other.id = uuid.uuid4().hex
    p._jobs[other.id] = other
    assert p.tasks.state()["landing_url"] == "/monitor"
    ready_job.status = "running"
    assert p.tasks.state()["landing_url"] == f"/monitor?id={ready_job.id}"
    ready_job.status = "recovery_required"
    ready_job.model_service = {"status": "stopped"}
    assert p.tasks.state()["landing_url"] == f"/monitor?id={ready_job.id}"
    assert p.tasks.state()["console_deployments"][1]["phase"] == "recovery"


def test_preparing_and_kept_warm_models_have_a_console(control_plane, ready_job):
    ready_job.status = "deploying"
    assert control_plane.tasks.state()["landing_url"] == "/monitor"
    ready_job.status = "succeeded"
    assert control_plane.tasks.state()["landing_url"] == f"/monitor?id={ready_job.id}"
    ready_job.model_service = {"status": "stopped"}
    assert control_plane.tasks.state()["landing_url"] == "/monitor"
    control_plane._workstation.update(mode="automatic", accepting=False)
    assert control_plane.tasks.state()["landing_url"] == "/monitor"


def test_idle_scheme_has_no_task_to_stop(control_plane, ready_job):
    assert "stop" not in control_plane.tasks.view(ready_job.id)["allowed_actions"]


def test_automatic_needs_a_matching_fixed_profile_and_preserves_restart_pause(control_plane, ready_job):
    p = control_plane
    p._workstation.update(accepting=False, reason="restart requires inspection")
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    assert denied.value.reason_code == "NOT_READY"
    assert p._workstation["mode"] == "debug"
    p.tasks.register_profiles({"sort": dict(ready_job.config, task="different task")})
    with pytest.raises(TaskError):
        p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    p.tasks.register_profiles({"sort": ready_job.config})
    result = p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    assert not result["accepting"]
    assert result["reason"] == "restart requires inspection"
    assert not p.tasks.list_tasks()
    ready_job.model_service = {"status": "stopped"}
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"accepting": True}, p.csrf_token)
    assert denied.value.reason_code == "NOT_READY"
    assert not p._workstation["accepting"]


def test_automatic_requires_package_completion_profile(control_plane, ready_job):
    p = control_plane
    ready_job.config = p.validate(valid_payload(presence_gate={"enabled": False}))["normalized"]
    p.tasks.register_profiles({"sort": ready_job.config})

    assert p.tasks.profile_views()[0]["ready"] is False
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    assert denied.value.reason_code == "NOT_READY"
    assert p._workstation["mode"] == "debug"


def test_automatic_checks_live_model_health_before_switch(control_plane, ready_job, monkeypatch):
    p = control_plane
    p.enable_actions = True
    ready_job.model_service["healthy"] = True
    p.tasks.register_profiles({"sort": ready_job.config})
    grant_ready_robot_probe(p)
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": False})
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    assert denied.value.reason_code == "NOT_READY"
    assert p._workstation["mode"] == "debug"
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": True})
    assert p.tasks.configure({"mode": "automatic"}, p.csrf_token)["mode"] == "automatic"
    assert not p.tasks.list_tasks()


def test_mode_switch_checks_the_deployed_instance_not_another_host(control_plane, ready_job, monkeypatch):
    p = control_plane
    p.enable_actions = True
    ready_job.model_service["healthy"] = True
    p.tasks.register_profiles({"sort": ready_job.config})
    grant_ready_robot_probe(p)
    lease = p._probe_leases.pop(("g2a", "192.168.10.55", 22))
    p._probe_leases[("g2a", "192.168.10.99", 22)] = lease
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": True})
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    assert denied.value.reason_code == "ROBOT_STATE_UNKNOWN"


def test_finished_manual_check_can_become_an_automatic_template(control_plane, ready_job):
    p = control_plane
    p.tasks.register_profiles({"sort": ready_job.config})
    p.start_deployment(ready_job.id, p.csrf_token)
    assert p.wait_for_terminal(ready_job.id, timeout=2)["status"] == "running"
    p.stop_deployment(ready_job.id, p.csrf_token)
    assert ready_job.status == "succeeded"
    assert p.tasks.profile_views()[0]["ready"]
    assert p.tasks.configure({"mode": "automatic"}, p.csrf_token)["mode"] == "automatic"


def test_editing_finished_manual_settings_creates_candidate_without_changing_current(control_plane, ready_job):
    p = control_plane
    p.start_deployment(ready_job.id, p.csrf_token)
    assert p.wait_for_terminal(ready_job.id, timeout=2)["status"] == "running"
    p.stop_deployment(ready_job.id, p.csrf_token)
    saved = p.configure_deployment(ready_job.id, {"task": "Updated fixed task"}, p.csrf_token)
    assert saved["status"] == "ready_to_start"
    assert saved["id"] != ready_job.id
    assert saved["model_service"]["candidate_only"]
    assert p.tasks.current_deployment_id() == ready_job.id
    assert "start" in p.tasks.view(ready_job.id)["allowed_actions"]
    p.tasks.register_profiles({"updated": saved["config"]})
    assert not p.tasks.profile_views()[0]["ready"]


def test_first_applied_configuration_pauses_new_tasks(control_plane, ready_job):
    p = control_plane
    ready_job.task_session["runtime_configured"] = False
    p._workstation.update(active_deployment_id=None, accepting=True)

    saved = p.configure_deployment(ready_job.id, {"task": "first task"}, p.csrf_token)

    assert saved["id"] == ready_job.id
    assert p._workstation["active_deployment_id"] == ready_job.id
    assert p._workstation["accepting"] is False
    assert p.tasks.list_tasks() == []


def test_runtime_configuration_rejects_busy_robot_without_saving_candidate(control_plane, ready_job):
    p = control_plane
    p._workstation["active_deployment_id"] = ready_job.id
    active = copy.deepcopy(ready_job)
    active.id = uuid.uuid4().hex
    active.status = "running"
    active.task_session = {"status": "running", "accepted": True}
    p._jobs[active.id] = active
    original = copy.deepcopy(ready_job.config)

    with pytest.raises(TaskError) as denied:
        p.configure_deployment(ready_job.id, {"task": "new task"}, p.csrf_token)

    assert denied.value.reason_code == "ROBOT_BUSY"
    assert ready_job.config == original
    assert set(p._jobs) == {ready_job.id, active.id}
    assert p._workstation["active_deployment_id"] == ready_job.id


def test_old_task_config_cannot_be_reapplied_by_stale_url_or_api(control_plane, ready_job):
    p = control_plane
    p._workstation["active_deployment_id"] = ready_job.id
    old = copy.deepcopy(ready_job)
    old.id = uuid.uuid4().hex
    old.config = dict(ready_job.config, task="old task")
    old.model_service["candidate_only"] = False
    p._jobs[old.id] = old

    assert old.id not in {item["id"] for item in p.tasks.state()["console_deployments"]}
    with pytest.raises(TaskError) as denied:
        p.configure_deployment(old.id, {"task": "reapply old task"}, p.csrf_token)
    assert denied.value.reason_code == "SCHEME_NOT_ACTIVE"
    with pytest.raises(TaskError) as denied:
        p.tasks.activate(old.id, p.csrf_token)
    assert denied.value.reason_code == "SCHEME_NOT_ACTIVE"
    assert old.config["task"] == "old task"
    assert p.tasks.current_deployment_id() == ready_job.id


def test_candidate_switch_is_explicit_and_never_queues_when_busy(control_plane, ready_job):
    p = control_plane
    p._workstation["active_deployment_id"] = ready_job.id
    candidate = copy.deepcopy(ready_job)
    candidate.id = uuid.uuid4().hex
    candidate.config = dict(ready_job.config, task="candidate task")
    candidate.model_service = {"status": "candidate", "candidate_only": True,
                               "deployment_id": candidate.id}
    candidate.task_session = {"status": "ready", "runtime_configured": True}
    p._jobs[candidate.id] = candidate
    p.start_deployment(ready_job.id, p.csrf_token)
    p.wait_for_terminal(ready_job.id, timeout=2)
    with pytest.raises(TaskError) as denied:
        p.tasks.activate(candidate.id, p.csrf_token)
    assert denied.value.reason_code == "ROBOT_BUSY"
    assert not p._workstation.get("switching_deployment_id")
    p.stop_deployment(ready_job.id, p.csrf_token)
    assert p.tasks.current_deployment_id() == ready_job.id
    p.tasks.activate(candidate.id, p.csrf_token)
    for _ in range(100):
        if not p._workstation.get("switching_deployment_id"):
            break
        time.sleep(0.01)
    assert p.tasks.current_deployment_id() == candidate.id
    assert p._workstation["accepting"] is False
    state = p.tasks.state()
    assert state["landing_url"] == f"/monitor?id={candidate.id}"
    assert [item["id"] for item in state["console_deployments"] if item["is_current"]] == [candidate.id]
    assert ready_job.id not in {item["id"] for item in state["console_deployments"]}
    assert ready_job.id in p._jobs  # Task evidence remains available through its ID.
    assert not p.tasks.list_tasks()[-1].get("request_id")


def test_unsaved_candidate_cannot_be_activated(control_plane, ready_job):
    candidate = copy.deepcopy(ready_job)
    candidate.id = uuid.uuid4().hex
    candidate.model_service = {"status": "candidate", "candidate_only": True}
    candidate.task_session = {"status": "ready", "runtime_configured": False}
    control_plane._jobs[candidate.id] = candidate
    with pytest.raises(TaskError) as denied:
        control_plane.tasks.activate(candidate.id, control_plane.csrf_token)
    assert denied.value.reason_code == "NOT_READY"
    assert control_plane.tasks.current_deployment_id() == ready_job.id


def test_candidate_cannot_change_fixed_robot_endpoint(control_plane, ready_job):
    control_plane._workstation["active_deployment_id"] = ready_job.id
    with pytest.raises(TaskError) as denied:
        control_plane.create_deployment(valid_payload(robot_host="192.168.10.66"),
                                        control_plane.csrf_token)
    assert denied.value.reason_code == "HARDWARE_CONFLICT"


def test_failed_switch_keeps_old_pointer_and_pauses_admission(control_plane, ready_job, monkeypatch):
    p = control_plane
    p._workstation["active_deployment_id"] = ready_job.id
    candidate = copy.deepcopy(ready_job)
    candidate.id = uuid.uuid4().hex
    candidate.config = dict(ready_job.config, task="new task")
    candidate.model_service = {"status": "candidate", "candidate_only": True}
    candidate.task_session = {"status": "ready", "runtime_configured": True}
    p._jobs[candidate.id] = candidate
    monkeypatch.setattr(
        p, "_mark_ready_to_start",
        lambda *_: (_ for _ in ()).throw(RuntimeError("mock model load failed")),
    )
    p.tasks.activate(candidate.id, p.csrf_token)
    for _ in range(100):
        if not p._workstation.get("switching_deployment_id"):
            break
        time.sleep(0.01)
    assert p.tasks.current_deployment_id() == ready_job.id
    assert p._workstation["accepting"] is False
    assert "mock model load failed" in p._workstation["switch_error"]


def test_switch_rejects_fresh_robot_probe_failure_even_with_old_lease(control_plane, ready_job, monkeypatch):
    p = control_plane
    p.enable_actions = True
    p._workstation["active_deployment_id"] = ready_job.id
    candidate = copy.deepcopy(ready_job)
    candidate.id = uuid.uuid4().hex
    candidate.config = dict(ready_job.config, task="new task")
    candidate.model_service = {"status": "candidate", "candidate_only": True}
    candidate.task_session = {"status": "ready", "runtime_configured": True}
    p._jobs[candidate.id] = candidate
    grant_ready_robot_probe(p)
    monkeypatch.setattr(p, "probe_robot", lambda *_args: {"reachable": False})
    with pytest.raises(TaskError) as denied:
        p.tasks.activate(candidate.id, p.csrf_token)
    assert denied.value.reason_code == "ROBOT_STATE_UNKNOWN"
    assert p.tasks.current_deployment_id() == ready_job.id
    assert not p._workstation.get("switching_deployment_id")


def test_candidate_cannot_close_shared_model_or_reset_robot(control_plane, ready_job, monkeypatch):
    p = control_plane
    p._workstation["active_deployment_id"] = ready_job.id
    candidate = copy.deepcopy(ready_job)
    candidate.id = uuid.uuid4().hex
    candidate.model_service = {"status": "candidate", "candidate_only": True}
    candidate.task_session = {"status": "ready", "runtime_configured": True}
    p._jobs[candidate.id] = candidate
    monkeypatch.setattr(p.deployment, "stop_model", lambda *_: pytest.fail("candidate stopped current model"))
    with pytest.raises(TaskError) as denied:
        p.close_model(candidate.id, p.csrf_token)
    assert denied.value.reason_code == "SCHEME_NOT_ACTIVE"
    p.enable_actions = True
    p.enable_reset = True
    with pytest.raises(TaskError) as denied:
        p.reset_robot_body(candidate.id, p.csrf_token)
    assert denied.value.reason_code == "SCHEME_NOT_ACTIVE"
    assert p.tasks.view(candidate.id)["allowed_actions"] == []


def test_activate_http_requires_csrf_and_reports_async_switch(control_plane, ready_job):
    candidate = copy.deepcopy(ready_job)
    candidate.id = uuid.uuid4().hex
    candidate.config = dict(ready_job.config, task="new task")
    candidate.model_service = {"status": "candidate", "candidate_only": True}
    candidate.task_session = {"status": "ready", "runtime_configured": True}
    control_plane._workstation["active_deployment_id"] = ready_job.id
    control_plane._jobs[candidate.id] = candidate
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(control_plane))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection(*server.server_address, timeout=3)
    body = json.dumps({"deployment_id": candidate.id})
    try:
        client.request("POST", "/api/workstation/activate", body,
                       {"Host": "localhost", "Content-Type": "application/json"})
        denied = client.getresponse()
        assert denied.status == 403
        denied.read()
        client.request("POST", "/api/workstation/activate", body,
                       {"Host": "localhost", "Content-Type": "application/json",
                        "X-CSRF-Token": control_plane.csrf_token})
        accepted = client.getresponse()
        assert accepted.status == 202
        assert json.loads(accepted.read())["accepting"] is False
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def test_http_home_resolves_current_state_but_explicit_setup_stays_accessible(control_plane, ready_job):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(control_plane))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        for method in ("GET", "HEAD"):
            client.request(method, "/", headers={"Host": "localhost"})
            response = client.getresponse()
            assert response.status == 302
            assert response.getheader("Location") == f"/monitor?id={ready_job.id}"
            assert response.getheader("Cache-Control") == "no-store"
            response.read()
        client.request("GET", "/index.html", headers={"Host": "localhost"})
        response = client.getresponse()
        assert response.status == 200
        setup = response.read()
        assert b"workstation-mode" not in setup
        assert b"scheme-manager" not in setup
        client.request("GET", "/history.html", headers={"Host": "localhost"})
        response = client.getresponse()
        assert response.status == 200
        assert b"records-tasks" in response.read()
        for method in ("GET", "HEAD"):
            client.request(method, "/monitor", headers={"Host": "localhost"})
            response = client.getresponse()
            assert response.status == 200
            if method == "GET":
                assert "机器人实时三维状态" in response.read().decode()
            else:
                response.read()
        control_plane._jobs.clear()
        client.request("GET", "/", headers={"Host": "localhost"})
        response = client.getresponse()
        assert response.status == 302
        assert response.getheader("Location") == "/monitor"
        response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
