"""A fresh tablet episode may admit an approved job but cannot start Policy."""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from nexus_deploy.control.tablet_start import RECORD_NAME, TabletStartBridge, TabletStartTarget
from nexus_deploy.control_api import ControlPlane
from nexus_deploy.robot.g2a.hitl.g2_runtime.app import FlowPROG2Application
from nexus_deploy.robot.g2a.hitl.g2_runtime.supervisor import (
    EpisodeCoordinator,
    RawVRFrame,
    SupervisorMode,
    VRButtonParser,
)
from nexus_deploy.robot.g2a.hitl.g2_runtime.tablet_start import NewTabletRecordingDetector


def test_watcher_requires_empty_boundary_and_unique_new_recording() -> None:
    detector = NewTabletRecordingDetector()
    old = Path("/data/record/old")
    new = Path("/data/record/new")
    assert detector.observe([old]) == ("awaiting_clear", None)
    assert detector.observe([old]) is None
    assert detector.observe([]) == ("ready", None)
    assert detector.observe([old, new]) == ("ambiguous", None)
    assert detector.observe([new]) == ("detected", "new")


def test_watcher_accepts_new_recording_after_initial_empty_scan() -> None:
    detector = NewTabletRecordingDetector()
    assert detector.observe([]) == ("ready", None)
    assert detector.observe([Path("/data/record/new")]) == ("detected", "new")


def test_vendor_chinese_recording_name_is_safe_to_bind() -> None:
    assert RECORD_NAME.fullmatch("触板翻面L-10.19_20261019102823")
    for invalid in ("", ".", "..", "../other", "other/name", "other\\name", "name\nother"):
        assert RECORD_NAME.fullmatch(invalid) is None


def test_bridge_consumes_detection_even_if_watcher_exits_immediately() -> None:
    received = []
    confirmed = threading.Event()
    target = TabletStartTarget(
        "job", "revision",
        (
            sys.executable, "-u", "-c",
            'print(\'NEXUS_TABLET_START:{"event":"detected","record_name":"触板翻面L-10.19_20261019102823"}\')',
        ),
    )
    bridge = TabletStartBridge(
        select_target=lambda: None if confirmed.is_set() else target,
        report=lambda _target, status: received.append(status),
        confirm=lambda _target, name: (received.append(name), confirmed.set()),
    )
    bridge.start()
    try:
        assert confirmed.wait(3)
        assert received[-2:] == ["detected", "触板翻面L-10.19_20261019102823"]
    finally:
        bridge.close()


def test_recorder_discovery_requires_recent_active_output(tmp_path: Path) -> None:
    episode = tmp_path / "L-9.2_20261016120000"
    (episode / "record").mkdir(parents=True)
    (episode / "meta_info.json").write_text(
        json.dumps({"data_validate": None, "integrity": None}), encoding="utf-8"
    )
    (episode / "record" / "aligned_joints.h5").write_bytes(b"recording")
    assert EpisodeCoordinator.list_active_recordings(tmp_path) == [episode.resolve()]
    assert EpisodeCoordinator.discover_active_recording(tmp_path) == episode.resolve()


def test_control_plane_exposes_only_saved_manual_hitl_start() -> None:
    plane = object.__new__(ControlPlane)
    plane._lock = threading.RLock()
    plane.enable_execute = True
    config = {"human_in_loop": True, "hardware_mode": "execute", "robot_id": "g2a"}
    job = SimpleNamespace(
        id="saved-job", status="ready_to_start", config=config, batch={},
        task_session={"runtime_configured": True}, model_service={"candidate_only": False},
    )
    plane._jobs = {job.id: job}
    allowed = ["start"]
    plane.tasks = SimpleNamespace(
        current_deployment_id=lambda: job.id,
        view=lambda _job_id: {"allowed_actions": allowed},
    )
    plane._robot_for_config = lambda _config: SimpleNamespace(remote_directory="/home/agi/deploy/hil")
    plane.deployment = SimpleNamespace(ssh_base=lambda _robot: ["ssh", "g2"])

    target = plane._tablet_start_candidate()
    assert target is not None
    assert target.job_id == job.id
    assert target.command[-1].endswith("-m robot_client.hitl.g2_runtime.tablet_start")
    allowed.clear()
    assert plane._tablet_start_candidate() is None
    allowed.append("start")
    job.batch = {"source": "scheduler"}
    assert plane._tablet_start_candidate() is None


def test_robot_runtime_requires_neutral_frame_before_first_y() -> None:
    parser = VRButtonParser(require_initial_neutral=True)
    pressed = RawVRFrame(1_000_000_000, 1, True, y=True)
    neutral = RawVRFrame(1_010_000_000, 2, True)
    fresh_press = RawVRFrame(1_020_000_000, 3, True, y=True)

    assert "y_press" not in {event.name for event in parser.feed(pressed)}
    assert "y_press" not in {event.name for event in parser.feed(neutral)}
    assert "y_press" in {event.name for event in parser.feed(fresh_press)}


def test_initial_bind_refuses_changed_episode_before_y() -> None:
    app = object.__new__(FlowPROG2Application)
    app._input_lock = threading.RLock()
    app.session = SimpleNamespace(supervisor=SimpleNamespace(mode=SupervisorMode.DISABLED))
    app.vendor_record_root = Path("/data/record")
    app.coordinator = SimpleNamespace(
        writer=None, discover_active_recording=lambda _root: Path("/data/record/other")
    )
    try:
        app.bind_initial_recording_in_hold("expected")
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("A different tablet episode must not be bound")


def test_initial_bind_uses_hold_transition_without_policy_input() -> None:
    app = object.__new__(FlowPROG2Application)
    app._input_lock = threading.RLock()
    app.vendor_record_root = Path("/data/record")
    app.coordinator = SimpleNamespace(
        writer=None, discover_active_recording=lambda _root: Path("/data/record/expected")
    )
    app.session = SimpleNamespace(supervisor=SimpleNamespace(mode=SupervisorMode.DISABLED))
    inputs = []

    def handle(logical_input):
        inputs.append(logical_input)
        app.session.supervisor.mode = SupervisorMode.HOLD
        return object()

    app._handle_input_locked = handle
    app.bind_initial_recording_in_hold("expected")
    assert [item.name for item in inputs] == ["flowpro_mode_toggle"]
    assert inputs[0].detail["expected_record_name"] == "expected"


def test_concurrent_bind_attempts_attach_only_once() -> None:
    app = object.__new__(FlowPROG2Application)
    app._input_lock = threading.RLock()
    app.vendor_record_root = Path("/data/record")
    app.coordinator = SimpleNamespace(
        writer=None, discover_active_recording=lambda _root: Path("/data/record/episode")
    )
    app.session = SimpleNamespace(supervisor=SimpleNamespace(mode=SupervisorMode.DISABLED))
    bindings = []

    def handle(_logical_input):
        time.sleep(0.02)
        bindings.append("episode")
        app.session.supervisor.mode = SupervisorMode.HOLD
        return object()

    app._handle_input_locked = handle
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: _attempt_bind(app), range(2)))
    assert bindings == ["episode"]
    assert sorted(results) == ["bound", "rejected"]


def _attempt_bind(app: FlowPROG2Application) -> str:
    try:
        app.bind_initial_recording_in_hold("episode")
    except RuntimeError:
        return "rejected"
    return "bound"
