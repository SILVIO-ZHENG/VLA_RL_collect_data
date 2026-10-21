from __future__ import annotations

import copy
import json
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

import pytest

from nexus_deploy.deployment import DeployError
from nexus_deploy.control_api import DeploymentJob, ValidationError, _handler_factory
from nexus_deploy.presence_gate import GateAction, PresenceTaskGate, PresenceTaskGateConfig
from nexus_deploy.sorting_api import SortingController, SortingError
from nexus_deploy.control.errors import TaskError
from nexus_deploy.control.vision import ReplayPresenceProvider
from test_control_api import control_plane, valid_payload  # noqa: F401


@pytest.fixture
def scheduler(control_plane, monkeypatch):
    monkeypatch.setenv("G2A_SORTING_TOKEN", "test-only-" * 8)
    config = control_plane.validate(valid_payload(presence_gate={"enabled": True}))["normalized"]
    assert config
    template = DeploymentJob(uuid.uuid4().hex, copy.deepcopy(config), status="ready_to_start")
    template.task_session = {"status": "ready", "generation": 0, "remote_started": False}
    template.model_service = {"status": "ready", "healthy": False}
    control_plane._jobs[template.id] = template
    settings = {"profiles": {"default": copy.deepcopy(config), "other": dict(config, task="other task")}}
    controller = SortingController(control_plane, settings)
    control_plane.tasks.configure({"mode": "automatic", "accepting": True}, control_plane.csrf_token)
    return controller


def submit(scheduler, request_id="batch-1"):
    result, created = scheduler.submit({"request_id": request_id, "profile": "default"})
    assert created
    return scheduler.plane._jobs[result["task_id"]]


def wait_for(scheduler, job_id, status, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if scheduler.view(job_id)["status"] == status:
            return scheduler.view(job_id)
        time.sleep(0.01)
    raise AssertionError(f"Task did not reach {status}: {scheduler.view(job_id)}")


def test_mode_and_readiness_boundaries(scheduler):
    p = scheduler.plane
    p.tasks.configure({"mode": "debug"}, p.csrf_token)
    with pytest.raises(SortingError) as denied:
        submit(scheduler)
    assert denied.value.reason_code == "MODE_CONFLICT"
    assert len(p.tasks.list_tasks()) == 0  # deployment is not a task
    p.tasks.configure({"mode": "automatic"}, p.csrf_token)
    template = next(iter(p._jobs.values()))
    with pytest.raises(TaskError) as denied:
        p.tasks.submit({"request_id": "operator-1", "profile": "default"},
                       source="operator", identity="operator")
    assert denied.value.reason_code == "MODE_CONFLICT"
    with pytest.raises(TaskError) as denied:
        p.start_deployment(template.id, p.csrf_token)
    assert denied.value.reason_code == "MODE_CONFLICT"
    p._jobs.clear()
    with pytest.raises(SortingError) as denied:
        submit(scheduler)
    assert denied.value.reason_code == "NOT_READY"
    assert not p._jobs


def test_control_handoff_is_atomic_and_does_not_start_a_task(scheduler, monkeypatch):
    p = scheduler.plane
    monkeypatch.setattr(p.robot_port, "start_session", lambda *_: pytest.fail("handoff started robot"))
    p.tasks.configure({"mode": "debug", "accepting": False}, p.csrf_token)
    assert p.tasks.state()["mode"] == "debug"
    assert p.tasks.state()["accepting"] is False
    assert p.tasks.list_tasks() == []

    gate = p.tasks.reset_gate()
    gate.update(required=True, status="failed", reason="physical reset failed")
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"mode": "automatic", "accepting": True}, p.csrf_token)
    assert denied.value.reason_code == "RESET_REQUIRED"
    assert p.tasks.state()["mode"] == "debug"
    assert p.tasks.state()["accepting"] is False
    assert p.tasks.list_tasks() == []

    gate.update(required=False, status="ready", reason="")
    p.tasks.configure({"mode": "automatic", "accepting": True}, p.csrf_token)
    assert p.tasks.state()["mode"] == "automatic"
    assert p.tasks.state()["accepting"] is True
    assert p.tasks.list_tasks() == []


def test_reopening_real_admission_requires_recent_idle_probe(scheduler):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    template.model_service["healthy"] = True
    p.tasks.configure({"mode": "debug", "accepting": False}, p.csrf_token)
    p.enable_actions = True
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"accepting": True}, p.csrf_token)
    assert denied.value.reason_code == "ROBOT_STATE_UNKNOWN"
    assert p.tasks.state()["accepting"] is False
    key = (str(template.config["robot_id"]), str(template.config["robot_host"]),
           int(template.config["robot_ssh_port"]))
    p._probe_leases[key] = dict(reachable=True, ssh_authenticated=True,
                                client={"status": "idle"}, checked_at=time.time() - 31)
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"accepting": True}, p.csrf_token)
    assert denied.value.reason_code == "ROBOT_STATE_UNKNOWN"
    p._probe_leases[key]["checked_at"] = time.time()
    p.tasks.configure({"accepting": True}, p.csrf_token)
    assert p.tasks.state()["accepting"] is True


def test_concurrent_requests_accept_one_without_queue(scheduler):
    def send(i):
        try:
            return scheduler.submit({"request_id": f"parallel-{i}", "profile": "default"})
        except SortingError as exc:
            return exc.reason_code
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(send, range(8)))
    accepted = [r for r in results if isinstance(r, tuple)]
    assert len(accepted) == 1
    assert results.count("ROBOT_BUSY") == 7
    job = scheduler.plane._jobs[accepted[0][0]["task_id"]]
    wait_for(scheduler, job.id, "running")
    scheduler.plane.stop_deployment(job.id, scheduler.plane.csrf_token)
    assert scheduler.view(job.id)["status"] == "cancelled"
    assert len(scheduler.plane._jobs) == 2  # rejected requests never appeared later


def test_busy_mode_switch_and_manual_pause_do_not_admit_or_auto_resume(scheduler):
    p = scheduler.plane
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    with pytest.raises(TaskError) as denied:
        p.tasks.configure({"mode": "debug"}, p.csrf_token)
    assert denied.value.reason_code == "ROBOT_BUSY"
    p.pause_deployment(job.id, p.csrf_token)
    assert scheduler.view(job.id)["status"] == "paused"
    for sequence in range(1, 6):
        p.presence_provider.emit(job.id, "PACKAGE_PRESENT", sequence)
    assert scheduler.view(job.id)["status"] == "paused"
    with pytest.raises(SortingError) as denied:
        submit(scheduler, "while-paused")
    assert denied.value.reason_code == "ROBOT_BUSY"
    p.tasks.configure({"accepting": False}, p.csrf_token)
    assert scheduler.view(job.id)["status"] == "paused"


def test_duplicate_and_conflict_are_checked_before_current_mode(scheduler):
    job = submit(scheduler)
    scheduler.plane.tasks.configure({"accepting": False}, scheduler.plane.csrf_token)
    same, created = scheduler.submit({"request_id": "batch-1", "profile": "default"})
    assert not created and same["task_id"] == job.id
    with pytest.raises(SortingError) as conflict:
        scheduler.submit({"request_id": "batch-1", "profile": "other"})
    assert conflict.value.reason_code == "IDEMPOTENCY_CONFLICT"
    with pytest.raises(SortingError) as denied:
        submit(scheduler, "batch-2")
    assert denied.value.reason_code == "NOT_ACCEPTING"
    assert len(scheduler.plane._jobs) == 2


@pytest.mark.parametrize("payload", [
    {"request_id": "x", "profile": "default", "callback_url": "http://arbitrary"},
    {"request_id": "x", "profile": "default", "command": "anything"},
    {"request_id": "../x", "profile": "default"},
    {"request_id": "x", "profile": "absent"},
])
def test_rejects_unbounded_inputs(scheduler, payload):
    with pytest.raises(SortingError):
        scheduler.submit(payload)


def test_restart_requires_manual_recovery_and_preserves_idempotence(scheduler):
    job = submit(scheduler)
    # A new control object reads the same isolated state file.
    from nexus_deploy.control_api import ControlPlane
    after_plane = ControlPlane(scheduler.plane.config_path, enable_actions=False)
    after = SortingController(after_plane, scheduler.settings)
    assert after_plane.tasks.state()["accepting"] is False
    assert after.view(job.id)["status"] == "recovery_required"
    same, created = after.submit({"request_id": "batch-1", "profile": "default"})
    assert same["task_id"] == job.id and not created
    with pytest.raises(SortingError):
        submit(after, "batch-2")
    after_plane.close()


def test_restart_updates_service_owned_gate_timing_on_idle_current_template(scheduler):
    plane = scheduler.plane
    template = next(iter(plane._jobs.values()))
    plane._workstation["active_deployment_id"] = template.id
    template.config["presence_gate"].update(pause_after_s=1.5, reset_after_s=5.0)
    plane._persist_state()

    from nexus_deploy.control_api import ControlPlane
    recovered = ControlPlane(plane.config_path, enable_actions=False)
    try:
        gate = recovered._jobs[template.id].config["presence_gate"]
        assert gate["enabled"] is True
        assert gate["pause_after_s"] == 3.0
        assert gate["reset_after_s"] == 6.0
        assert recovered.tasks.state()["accepting"] is False
    finally:
        recovered.close()


def test_worker_ignores_legacy_non_scheduler_batch_records(scheduler):
    plane = scheduler.plane
    legacy = DeploymentJob("c" * 32, {}, status="failed")
    legacy.batch = {"status": "failed", "reason": "historical robot recovery"}
    legacy.task_session = {"source": "operator"}
    plane._jobs[legacy.id] = legacy

    scheduler.tick()

    assert legacy.batch == {"status": "failed", "reason": "historical robot recovery"}


def test_persistence_failure_cannot_start_robot(scheduler, monkeypatch):
    p = scheduler.plane
    monkeypatch.setattr(p, "_persist_state", lambda: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(SortingError) as denied:
        submit(scheduler)
    assert denied.value.reason_code == "PERSISTENCE_FAILED"
    assert p._workstation["accepting"] is False
    assert all(j.status != "running" for j in p._jobs.values())


def test_stale_model_health_rejects_without_creating_a_task(scheduler, monkeypatch):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    template.model_service.update(status="ready", healthy=True)
    p.enable_actions = True
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": False})
    with pytest.raises(SortingError) as denied:
        submit(scheduler)
    assert denied.value.reason_code == "NOT_READY"
    assert len(p._jobs) == 1
    assert not p.tasks.list_tasks()


def test_mock_vision_to_simulated_completion_and_callback(scheduler, monkeypatch):
    p = scheduler.plane
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    assert p.presence_provider.health(job.id) == "mock"
    base = time.time_ns() - 1_000_000
    clock = [base]
    monkeypatch.setattr("nexus_deploy.control_api.time.time_ns", lambda: clock[0])
    for sequence in range(1, 23):
        if scheduler.view(job.id)["status"] == "simulated":
            break
        clock[0] = base + sequence * 500_000_000
        p.presence_provider.emit(job.id, "NO_PACKAGE", sequence, observed_at_ns=clock[0])
        if sequence == 7:
            wait_for(scheduler, job.id, "paused")
    result = wait_for(scheduler, job.id, "simulated")
    assert result["completion_evidence"]["physical_execution"] is False
    assert result["completion_evidence"]["conveyor_delivery_independently_verified"] is False
    assert p._jobs[job.id].robot_reset["status"] == "read_only"
    assert [c["action"] for c in p._jobs[job.id].task_session["robot_commands"]].count("reset") == 1


def test_replay_uses_frame_estimator_and_ends_with_unknown(scheduler, monkeypatch):
    p = scheduler.plane
    p.presence_provider = ReplayPresenceProvider(p)
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    frames = [object() for _ in range(4)]

    class Capture:
        def __init__(self, path):
            assert path == "offline.mp4"
            self.index = 0
        def isOpened(self):
            return True
        def get(self, key):
            return 30.0
        def read(self):
            if self.index == len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame
        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=5))
    class Estimator:
        def __init__(self):
            self.calls = []
            self.resets = 0
        def reset(self):
            self.resets += 1
        def process_frame_with_presence(self, frame, **kwargs):
            self.calls.append((frame, kwargs))
            presence = types.SimpleNamespace(to_dict=lambda: {"state": "UNKNOWN", "window_states": []})
            return types.SimpleNamespace(presence=presence)

    estimator = Estimator()
    p.presence_provider.play(job.id, "offline.mp4", estimator, sample_interval_s=0.001)
    assert estimator.resets == 1
    assert [call[0] for call in estimator.calls] == frames
    assert all(call[1]["color_space"] == "BGR" for call in estimator.calls)
    assert job.presence_gate["state"] == "UNKNOWN"
    assert job.presence_gate["sequence"] == 5  # end-of-stream invalidation
    assert job.batch["status"] != "completed"


def test_callback_retries_stable_payload(scheduler):
    events = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            events.append((self.headers["Idempotency-Key"], json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(503 if len(events) == 1 else 200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scheduler.callback_url = f"http://127.0.0.1:{server.server_port}/callback"
        job = submit(scheduler)
        job.batch["status"] = "simulated"
        scheduler.tick()
        assert scheduler.view(job.id)["callback"]["status"] == "pending"
        job.batch["callback"]["next_at"] = 0
        scheduler.tick()
        assert scheduler.view(job.id)["callback"]["status"] == "delivered"
        assert len(events) == 2 and events[0] == events[1]
        assert events[0][1]["event_type"] == "sorting.simulated"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_auth_submit_and_query(scheduler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(scheduler.plane, sorting=scheduler,
                                                                       allowed_hosts=("127.0.0.1",)))
    server.RequestHandlerClass = _handler_factory(scheduler.plane, sorting=scheduler, port=server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with pytest.raises(error.HTTPError) as denied:
            opener.open(base + "/api/sorting/health")
        assert denied.value.code == 401
        headers = {"Authorization": "Bearer " + scheduler.token, "Content-Type": "application/json"}
        with opener.open(request.Request(base + "/api/sorting/model", headers=headers)) as r:
            assert json.load(r)["state"] == "ready"
        with opener.open(request.Request(base + "/api/sorting/robot", headers=headers)) as r:
            assert json.load(r)["ready_for_next_task"] is True
        body = json.dumps({"request_id": "http-1", "profile": "default"}).encode()
        with opener.open(request.Request(base + "/api/sorting/tasks", data=body, headers=headers)) as r:
            assert r.status == 202
            task = json.load(r)
        with opener.open(request.Request(base + "/api/sorting/tasks", data=body, headers=headers)) as r:
            assert r.status == 200 and json.load(r)["task_id"] == task["task_id"]
        with opener.open(request.Request(base + "/api/sorting/tasks/" + task["task_id"], headers=headers)) as r:
            assert json.load(r)["request_id"] == "http-1"
        with opener.open(request.Request(base + "/api/sorting/robot", headers=headers)) as r:
            assert json.load(r)["state"] == "busy"
        stop_url = base + "/api/sorting/tasks/" + task["task_id"] + "/stop"
        stop_body = json.dumps({"command_id": "stop-http-1"}).encode()
        with opener.open(request.Request(stop_url, data=stop_body, headers=headers)) as r:
            assert r.status == 202
            assert json.load(r)["status"] == "cancelled"
        with opener.open(request.Request(stop_url, data=stop_body, headers=headers)) as r:
            assert r.status == 200
            assert json.load(r)["task_id"] == task["task_id"]
        with pytest.raises(error.HTTPError) as denied:
            opener.open(request.Request(stop_url, data=stop_body,
                                        headers={"Content-Type": "application/json"}))
        assert denied.value.code == 401
        with pytest.raises(error.HTTPError) as denied:
            opener.open(request.Request(base + "/api/sorting/health", headers=dict(headers, Origin="http://browser")))
        assert denied.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_scheduler_stop_keeps_physical_batch_blocked_until_reset(scheduler, monkeypatch):
    p = scheduler.plane
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    job.config["hardware_mode"] = "execute"
    job.task_session["remote_started"] = True
    p.enable_actions = True
    template = next(item for item in p._jobs.values() if not item.batch)
    template.model_service["healthy"] = True
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": True})
    monkeypatch.setattr(p, "_robot_for_config", lambda *_: object())
    stopped = []
    monkeypatch.setattr(p.robot_port, "stop_session", lambda *_: stopped.append(job.id))
    monkeypatch.setattr(p, "probe_robot", lambda *_args: {
        "checked_at": time.time(), "reachable": True, "ssh_authenticated": True,
        "identity": {"match": True}, "client": {"status": "idle", "valid": True, "running": False},
        "observation": {"ready": True},
    })

    result, created = scheduler.stop(job.id, {"command_id": "stop-physical-1"})
    assert created and result["status"] == "cancelled"
    assert result["lifecycle_state"] == "ended_unreset"
    assert result["reset_state"] == "not_reset"
    assert stopped == [job.id]
    same, created = scheduler.stop(job.id, {"command_id": "stop-physical-1"})
    assert not created and same["task_id"] == job.id and stopped == [job.id]
    robot = scheduler.robot()
    assert robot["state"] == "idle" and robot["reset_state"] == "not_reset"
    assert robot["ready_for_next_task"] is False
    with pytest.raises(SortingError) as denied:
        submit(scheduler, "after-stop-before-reset")
    assert denied.value.reason_code == "RESET_REQUIRED"

    job.robot_reset["status"] = "succeeded"  # simulated reset acknowledgement
    p._mark_robot_reset_succeeded(job)
    assert scheduler.robot()["ready_for_next_task"] is True, (scheduler.robot(), scheduler.model())
    assert scheduler.view(job.id)["lifecycle_state"] == "ended_reset"


def test_operator_end_of_scheduler_task_reports_completed_but_requires_reset(scheduler, monkeypatch):
    p = scheduler.plane
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    job.config["hardware_mode"] = "execute"
    job.config["safety"]["safety_profile_id"] = "approved"
    job.task_session["remote_started"] = True
    p.enable_actions = True
    stopped = []
    monkeypatch.setattr(p, "_robot_for_config", lambda *_: object())
    monkeypatch.setattr(p.robot_port, "stop_session", lambda *_: stopped.append(job.id))
    monkeypatch.setattr(p.deployment, "deploy_generated_safety_profile", lambda *_: None)
    monkeypatch.setattr(p.robot_port, "reset_body", lambda *_args, **_kwargs: {
        "completed": True, "output_tail": "verified reset"
    })

    p.stop_deployment(job.id, p.csrf_token)  # The web operator action.
    before = scheduler.view(job.id)
    assert stopped == [job.id]
    assert before["status"] == "completed"
    assert before["lifecycle_state"] == "ended_unreset"
    assert before["reset_state"] == "not_reset"
    assert before["completion_rule"] == "operator_end"
    assert before["completion_evidence"]["reset_status"] == "pending"
    assert p.tasks.reset_gate()["required"] is True
    with pytest.raises(SortingError) as denied:
        submit(scheduler, "after-operator-end-before-reset")
    assert denied.value.reason_code == "RESET_REQUIRED"

    reservation = f"reset:{job.id}"
    p._reserved_robots[job.config["robot_id"]] = reservation
    p._run_robot_body_reset(job.id, reservation, job.config)
    after = scheduler.view(job.id)
    assert after["status"] == "completed"
    assert after["lifecycle_state"] == "ended_reset"
    assert after["reset_state"] == "ready"
    assert after["completion_evidence"]["reset_status"] == "succeeded"
    assert p.tasks.reset_gate()["required"] is False


def test_operator_completed_task_becomes_recovery_required_if_reset_fails(scheduler, monkeypatch):
    p = scheduler.plane
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    job.config["hardware_mode"] = "execute"
    job.config["safety"]["safety_profile_id"] = "approved"
    job.task_session["remote_started"] = True
    p.enable_actions = True
    monkeypatch.setattr(p, "_robot_for_config", lambda *_: object())
    monkeypatch.setattr(p.robot_port, "stop_session", lambda *_: None)
    monkeypatch.setattr(p.deployment, "deploy_generated_safety_profile", lambda *_: None)
    monkeypatch.setattr(p.robot_port, "reset_body", lambda *_args, **_kwargs: {
        "completed": False
    })

    p.stop_deployment(job.id, p.csrf_token)
    assert scheduler.view(job.id)["status"] == "completed"
    reservation = f"reset:{job.id}"
    p._reserved_robots[job.config["robot_id"]] = reservation
    p._run_robot_body_reset(job.id, reservation, job.config)
    after = scheduler.view(job.id)
    assert after["status"] == "recovery_required"
    assert after["reset_state"] == "unknown"
    assert p.tasks.reset_gate()["required"] is True


def test_failed_physical_task_blocks_query_and_submission_until_verified_reset(scheduler, monkeypatch):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    template.config["hardware_mode"] = "execute"
    scheduler.profiles["default"] = copy.deepcopy(template.config)
    p.tasks.register_profiles(scheduler.profiles)
    failed = DeploymentJob(uuid.uuid4().hex, copy.deepcopy(template.config), status="failed")
    failed.task_session = {"accepted": True, "status": "failed", "remote_started": False}
    failed.batch = {"source": "scheduler", "status": "failed", "request_id": "old-failure"}
    p._jobs[failed.id] = failed
    p._mark_robot_reset_required(failed, "failed", "客户端意外退出；必须复位")
    monkeypatch.setattr(p, "probe_robot", lambda *_args: {
        "checked_at": time.time(), "reachable": True, "ssh_authenticated": True,
        "identity": {"match": True},
        "client": {"status": "idle", "valid": True, "running": False},
        "observation": {"ready": True},
    })
    assert p.tasks.state()["active_tasks"] == []
    robot = scheduler.robot()
    assert robot["reset_state"] == "unknown"
    assert robot["ready_for_next_task"] is False
    with pytest.raises(SortingError) as denied:
        submit(scheduler, "new-after-crash")
    assert denied.value.reason_code == "RESET_REQUIRED"
    p.tasks.configure({"accepting": False}, p.csrf_token)
    with pytest.raises(TaskError):
        p.tasks.configure({"accepting": True}, p.csrf_token)
    assert p.tasks.state()["accepting"] is False


def test_recovery_only_configuration_is_not_a_current_task(scheduler):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    template.status = "recovery_required"
    template.task_session["status"] = "recovery_required"
    p._mark_robot_reset_required(template, "failed", "复位失败")
    state = p.tasks.state()
    assert state["active_tasks"] == []
    assert state["current_task_id"] is None
    assert state["robot_gate"]["required"] is True


def test_scheduler_reset_requires_token_and_command_id(scheduler, monkeypatch):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    template.config["hardware_mode"] = "execute"
    p._workstation["active_deployment_id"] = template.id
    p._mark_robot_reset_required(template, "pending", "上一任务待复位")
    calls = []
    def fake_reset(job_id, _csrf, *, command_id=None):
        calls.append((job_id, command_id))
        template.robot_reset["command_id"] = command_id
        template.robot_reset["status"] = "running"
        return template.to_dict()
    monkeypatch.setattr(p, "reset_robot_body", fake_reset)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(p, sorting=scheduler,
                                                                       allowed_hosts=("127.0.0.1",)))
    server.RequestHandlerClass = _handler_factory(p, sorting=scheduler, port=server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/sorting/robot/reset"
    opener = request.build_opener(request.ProxyHandler({}))
    body = json.dumps({"command_id": "reset-1"}).encode()
    try:
        with pytest.raises(error.HTTPError) as denied:
            opener.open(request.Request(url, data=body, headers={"Content-Type": "application/json"}))
        assert denied.value.code == 401
        headers = {"Authorization": "Bearer " + scheduler.token, "Content-Type": "application/json"}
        with opener.open(request.Request(url, data=body, headers=headers)) as response:
            assert response.status == 202
            assert json.load(response)["reset_state"] == "not_reset"
        with opener.open(request.Request(url, data=body, headers=headers)) as response:
            assert response.status == 202
        assert calls == [(template.id, "reset-1")]
        assert p._workstation["accepting"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_robot_read_failure_and_model_health_fail_closed(scheduler, monkeypatch):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    template.model_service["healthy"] = True
    p.enable_actions = True
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": True})
    monkeypatch.setattr(p, "probe_robot", lambda *_args: (_ for _ in ()).throw(OSError("offline")))
    assert scheduler.robot()["state"] == "unknown"
    assert scheduler.robot()["ready_for_next_task"] is False
    monkeypatch.setattr(p, "_safe_status", lambda *_: {"healthy": False})
    assert scheduler.model()["state"] == "unhealthy"
    assert scheduler.robot()["ready_for_next_task"] is False


def test_scheduler_requires_package_detection_to_finish(scheduler):
    p = scheduler.plane
    template = next(iter(p._jobs.values()))
    disabled = p.validate(valid_payload(presence_gate={"enabled": False}))["normalized"]
    template.config = disabled
    scheduler.profiles["default"] = copy.deepcopy(disabled)
    p.tasks.register_profiles(scheduler.profiles)
    assert scheduler.model()["completion_gate_ready"] is False
    assert scheduler.robot()["ready_for_next_task"] is False
    with pytest.raises(SortingError) as denied:
        submit(scheduler)
    assert denied.value.reason_code == "NOT_READY"
    assert len(p.tasks.list_tasks()) == 0


def test_cancel_before_start_worker_runs_reaches_terminal_without_robot(scheduler):
    p = scheduler.plane
    p.enable_actions = True
    job = DeploymentJob(uuid.uuid4().hex, valid_payload(hardware_mode="execute"),
                        status="task_starting")
    job.batch = {"request_id": "early-cancel", "source": "scheduler", "status": "starting"}
    job.task_session = {"status": "starting", "cancel_requested": True,
                        "generation": 1, "remote_started": False}
    p._jobs[job.id] = job
    p._reserved_robots["g2a"] = job.id
    p._start_task_safely(job.id, 1)
    assert scheduler.view(job.id)["status"] == "cancelled"
    assert job.batch["status"] == "cancelled"
    assert "g2a" not in p._reserved_robots


def test_scheduler_stop_rejects_other_tasks_and_invalid_commands(scheduler):
    job = submit(scheduler)
    with pytest.raises(SortingError) as invalid:
        scheduler.stop(job.id, {"command_id": "../stop"})
    assert invalid.value.status == 400
    with pytest.raises(SortingError) as unknown:
        scheduler.stop("f" * 32, {"command_id": "stop-1"})
    assert unknown.value.status == 404
    scheduler.stop(job.id, {"command_id": "stop-1"})
    with pytest.raises(SortingError) as conflict:
        scheduler.stop(job.id, {"command_id": "stop-2"})
    assert conflict.value.reason_code == "STATE_CONFLICT"


def test_confirmed_stop_remains_cancelled_after_restart(scheduler):
    job = submit(scheduler)
    wait_for(scheduler, job.id, "running")
    scheduler.stop(job.id, {"command_id": "stop-before-restart"})
    from nexus_deploy.control_api import ControlPlane
    after_plane = ControlPlane(scheduler.plane.config_path, enable_actions=False)
    try:
        after = SortingController(after_plane, scheduler.settings)
        assert after.view(job.id)["status"] == "cancelled"
        same, created = after.stop(job.id, {"command_id": "stop-before-restart"})
        assert not created and same["status"] == "cancelled"
    finally:
        after_plane.close()

def ready_reset(plane, monkeypatch, *, hardware="execute"):
    plane.enable_actions = True
    plane.enable_execute = True
    plane.enable_reset = True
    job = DeploymentJob("a" * 32, valid_payload(hardware_mode=hardware, presence_gate={"enabled": True},
                                             safety={"safety_profile_id": "approved"}), status="paused")
    job.batch = {"status": "running", "request_id": "reset-1"}
    job.task_session = {"status": "paused", "generation": 1, "remote_started": True}
    job.presence_gate = {"state": "NO_PACKAGE", "no_package_elapsed_s": 10.0, "sequence": 100}
    plane._jobs[job.id] = job
    plane._reserved_robots["g2a"] = job.id
    gate = PresenceTaskGate(PresenceTaskGateConfig(enabled=True))
    gate.start_session(job.id)
    for i in range(101):
        decision = gate.observe("NO_PACKAGE", observed_at_ns=1_000_000_000 + i * 100_000_000,
                                task_state="running" if i <= 30 else "paused")
        if decision.action == GateAction.PAUSE:
            gate.acknowledge(GateAction.PAUSE, applied=True)
    assert gate.pending_action == GateAction.RESET
    plane._presence_task_gates[job.id] = gate
    monkeypatch.setattr(plane.deployment, "stop_robot_session", lambda *a, **k: None)
    monkeypatch.setattr(plane.deployment, "deploy_generated_safety_profile", lambda *a, **k: None)
    monkeypatch.setattr(plane.deployment, "start_robot_session", lambda *a, **k: pytest.fail("batch must not restart"))
    return job


def test_physical_batch_finishes_only_after_reset_without_restarting(control_plane, monkeypatch):
    job = ready_reset(control_plane, monkeypatch)
    def stop(*args, **kwargs):
        # The camera stops with the robot client, before the sidecar itself is
        # stopped. A health tick in this interval used to poison completion.
        job.presence_gate.update(enabled=True, received_at=time.time() - 3)
        control_plane.health_tick()
        assert not job.task_session.get("completion_conflict")
    monkeypatch.setattr(control_plane.deployment, "stop_robot_session", stop)
    def reset(*args, **kwargs):
        assert job.batch["status"] != "completed"
        # The physical reset takes longer than a frame freshness interval.
        # Stopping the sidecar deliberately must preserve the prior evidence.
        job.presence_gate.update(enabled=True, received_at=time.time() - 3)
        control_plane.health_tick()
        assert not job.task_session.get("completion_conflict")
        return {"completed": True, "output_tail": "ok"}
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle", reset)
    control_plane._apply_presence_reset(job.id, 1)
    assert job.batch["status"] == "completed"
    assert job.robot_reset["status"] == "succeeded"
    assert not job.task_session["remote_started"]
    assert job.id not in control_plane._presence_task_gates
    assert "g2a" not in control_plane._reserved_robots
    assert job.batch["completion_evidence"]["physical_execution"] is True
    assert job.batch["completion_evidence"]["generation"] == 1
    assert job.batch["completion_evidence"]["vision_sequence"] == 100
    assert job.batch["completion_evidence"]["required_empty_seconds"] == 6.0
    assert control_plane.tasks.view(job.id)["status"] == "completed"
    assert control_plane.tasks.view(job.id)["reset_state"] == "ready"
    with pytest.raises(ValidationError):
        control_plane.start_deployment(job.id, control_plane.csrf_token)


def test_manual_presence_task_finishes_after_physical_reset(control_plane, monkeypatch):
    job = ready_reset(control_plane, monkeypatch)
    job.batch = {}
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle",
                        lambda *args, **kwargs: {"completed": True, "output_tail": "ok"})

    control_plane._apply_presence_reset(job.id, 1)

    assert job.status == "succeeded"
    assert job.task_session["status"] == "stopped"
    assert job.task_session["completion_evidence"]["physical_execution"] is True
    assert control_plane.tasks.view(job.id)["status"] == "completed"
    assert job.robot_reset["status"] == "succeeded"
    assert job.id not in control_plane._presence_task_gates
    assert "g2a" not in control_plane._reserved_robots


def test_readonly_never_reports_physical_completion(control_plane, monkeypatch):
    job = ready_reset(control_plane, monkeypatch, hardware="read-only")
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle", lambda *a, **k: pytest.fail("no physical reset"))
    control_plane._apply_presence_reset(job.id, 1)
    assert job.batch["status"] == "simulated"
    assert job.robot_reset["status"] == "read_only"
    assert job.batch["completion_evidence"]["physical_execution"] is False


def test_manual_readonly_presence_finishes_without_physical_reset(control_plane, monkeypatch):
    job = ready_reset(control_plane, monkeypatch, hardware="read-only")
    job.batch = {}
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle",
                        lambda *args, **kwargs: pytest.fail("no physical reset"))

    control_plane._apply_presence_reset(job.id, 1)

    assert job.status == "succeeded"
    assert job.task_session["status"] == "stopped"
    assert job.task_session["completion_evidence"]["physical_execution"] is False
    assert control_plane.tasks.view(job.id)["status"] == "simulated"
    assert job.robot_reset["status"] == "read_only"
    assert "g2a" not in control_plane._reserved_robots


def test_failed_reset_does_not_complete(control_plane, monkeypatch):
    job = ready_reset(control_plane, monkeypatch)
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle",
                        lambda *a, **k: (_ for _ in ()).throw(DeployError("reset failed")))
    control_plane._apply_presence_gate_action(job.id, 1, GateAction.RESET)
    assert job.status == "recovery_required"
    assert job.robot_reset["status"] == "failed"
    assert job.batch["status"] != "completed"


def test_unknown_vision_before_sidecar_stop_blocks_physical_completion(control_plane, monkeypatch):
    job = ready_reset(control_plane, monkeypatch)
    control_plane._presence_tokens[job.id] = "test-token"
    def stop(*args, **kwargs):
        control_plane.ingest_presence_gate(job.id, {
            "generation": 1,
            "sequence": 101,
            "status": "ok",
            "presence": {
                "state": "UNKNOWN",
                "observed_at_ns": time.time_ns(),
                "window_states": [],
            },
        }, "test-token")
    monkeypatch.setattr(control_plane.deployment, "stop_robot_session", stop)
    monkeypatch.setattr(control_plane.deployment, "reset_robot_with_dahuan_cycle",
                        lambda *a, **k: {"completed": True, "output_tail": "ok"})
    control_plane._apply_presence_reset(job.id, 1)
    assert job.status == "recovery_required"
    assert job.batch["status"] == "recovery_required"
    assert "g2a" in control_plane._reserved_robots


def test_batch_metadata_is_persistent():
    job = DeploymentJob("a" * 32, {}, batch={"request_id": "persist", "status": "completed"})
    assert DeploymentJob.from_dict(job.to_dict()).batch == job.batch
