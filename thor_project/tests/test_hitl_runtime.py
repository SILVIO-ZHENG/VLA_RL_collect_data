#!/usr/bin/env python

import hashlib
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from nexus_deploy.robot.g2a.hitl.common.sidecar import CAMERA_SOURCES
from nexus_deploy.robot.g2a.hitl.g2_runtime.executed import FlowPROExecutionHistory
from nexus_deploy.robot.g2a.hitl.g2_runtime.intervention import InterventionState, SceneRestorationSnapshot
from nexus_deploy.robot.g2a.hitl.g2_runtime.paper_runtime import FlowPROPaperRuntime
from nexus_deploy.robot.g2a.hitl.g2_runtime.rollback import RollbackExecutorState
from nexus_deploy.robot.g2a.hitl.g2_runtime.runtime import (
    FlowPROInferenceCapture,
    FlowPROSessionController,
    validate_native_candidate_response,
)
from nexus_deploy.robot.g2a.hitl.g2_runtime.supervisor import (
    HumanHandbackEvidence,
    LogicalInput,
    RolloutSupervisor,
    SupervisorMode,
    SupervisorTransition,
)


class FakeCoordinator:
    def __init__(self) -> None:
        self.observations = []
        self.candidates = []
        self.logical_inputs = []
        self.finished = []
        self.fail_finish = False

    def record_observation(self, **kwargs):
        self.observations.append(kwargs)
        return kwargs

    def record_policy_candidate(self, values, **kwargs):
        self.candidates.append((np.asarray(values), kwargs))
        return kwargs

    def attach_active_recording(self, _path, *, flowpro_session_id=None):
        return flowpro_session_id or "session-1"

    def append_logical_input(self, logical_input, *, control_owner):
        self.logical_inputs.append((logical_input, control_owner))

    def finish_control(self, *, outcome, outcome_detail=None):
        if self.fail_finish:
            raise OSError("simulated durable write failure")
        self.finished.append((outcome, outcome_detail or {}))
        return {"outcome": outcome, "outcome_detail": outcome_detail or {}}


def _request() -> dict:
    anchor = 1_788_253_877_000_000_000
    return {
        "request_id": "request-1",
        "observation": {
            "state": np.arange(24, dtype=np.float32),
            "task": "sort the parcel",
            "images": {
                key: {
                    "encoding": "jpeg",
                    "width": 640,
                    "height": 400,
                    "data": b"\xff\xd8exact-request-jpeg\xff\xd9",
                }
                for key in CAMERA_SOURCES
            },
            "state_capture_wall_time_ns": anchor,
            "camera_frame_wall_time_ns": dict.fromkeys(CAMERA_SOURCES, anchor),
        },
    }


def _response() -> dict:
    actions = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
    payload = np.ascontiguousarray(actions.astype("<f4", copy=False)).tobytes(order="C")
    return {
        "request_id": "request-1",
        "native_actions": actions,
        "native_action_dim": 16,
        "native_action_space": "absolute_unnormalized",
        "native_actions_sha256": hashlib.sha256(payload).hexdigest(),
        "sampling": {
            "algorithm": "pi05_rectified_flow",
            "seed": 123,
            "fixed": False,
            "noise_dtype": "float32",
            "noise_shape": [1, 50, 32],
            "noise_sha256": "a" * 64,
        },
    }


def test_runtime_captures_exact_request_and_native_response_under_one_generation() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    assert not capture.active
    requested_generation_id = str(uuid4())
    generation_id = capture.begin_generation(requested_generation_id)
    assert generation_id == requested_generation_id
    assert capture.active

    first = capture.capture_inference(request=_request(), response=_response(), generation_id=generation_id)
    second_request = _request()
    second_request["request_id"] = "request-2"
    second_response = _response()
    second_response["request_id"] = "request-2"
    second = capture.capture_inference(
        request=second_request,
        response=second_response,
        generation_id=generation_id,
    )

    assert first is not None
    assert second is not None
    assert first.generation_id == second.generation_id == generation_id
    assert coordinator.observations[0]["task"] == "sort the parcel"
    np.testing.assert_array_equal(coordinator.observations[0]["state_24d"], np.arange(24))
    assert set(coordinator.observations[0]["images"]) == set(CAMERA_SOURCES)
    np.testing.assert_array_equal(coordinator.candidates[0][0], _response()["native_actions"])
    assert coordinator.candidates[0][1]["sampling"]["seed"] == 123

    capture.invalidate_generation()
    assert not capture.active
    assert (
        capture.capture_inference(
            request=_request(),
            response=_response(),
            generation_id=generation_id,
        )
        is None
    )


def test_paper_runtime_accumulates_left_and_right_controller_activity_for_correction() -> None:
    class Intervention:
        state = InterventionState.CORRECTION

        def __init__(self) -> None:
            self.completed = None

        def complete_correction(self, **kwargs) -> None:
            self.completed = kwargs
            self.state = InterventionState.COMPLETE

    class Session:
        def __init__(self) -> None:
            self.supervisor = SimpleNamespace(mode=SupervisorMode.HUMAN)
            self.coordinator = SimpleNamespace()

        def _control_owner(self) -> str:
            return "human"

        def handle(self, logical_input, **_kwargs):
            if logical_input.name == "human_finish":
                self.supervisor.mode = SupervisorMode.HOLD
                return SimpleNamespace(current=SupervisorMode.HOLD)
            return None

    intervention = Intervention()
    runtime = FlowPROPaperRuntime(
        session=Session(),
        intervention=intervention,
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: None,
    )
    runtime.handle(
        LogicalInput(
            "side_takeover_input",
            1_000_000_000,
            {"left_side": True, "right_side": False},
        )
    )
    runtime.handle(
        LogicalInput(
            "side_takeover_input",
            1_100_000_000,
            {"left_side": False, "right_side": True},
        )
    )
    runtime.handle(
        LogicalInput("human_finish", 1_200_000_000, {}),
        handback_evidence=SimpleNamespace(),
    )

    assert intervention.completed["controller_active_arms"] == (True, True)


def test_capture_invalidation_during_sidecar_io_never_blocks_or_appends_stale_anchor() -> None:
    observation_write_started = threading.Event()
    release_observation_write = threading.Event()

    class BlockingCoordinator(FakeCoordinator):
        def record_observation(self, **kwargs):
            observation_write_started.set()
            assert release_observation_write.wait(1.0)
            return super().record_observation(**kwargs)

    coordinator = BlockingCoordinator()
    history = FlowPROExecutionHistory()
    capture = FlowPROInferenceCapture(
        coordinator,
        native_action_dim=16,
        chunk_size=50,
        execution_history=history,
    )
    generation_id = capture.begin_generation(str(uuid4()))
    result: list[object] = []

    worker = threading.Thread(
        target=lambda: result.append(
            capture.capture_inference(
                request=_request(),
                response=_response(),
                generation_id=generation_id,
            )
        )
    )
    worker.start()
    assert observation_write_started.wait(1.0)

    # invalidate_generation must not wait for slow image/sidecar persistence.
    invalidated = threading.Event()

    def invalidate_and_rebegin_same_uuid() -> None:
        capture.invalidate_generation()
        # Reusing the visible UUID cannot revive the old in-flight capture;
        # the internal epoch must still distinguish this new lifetime.
        capture.begin_generation(generation_id)
        invalidated.set()

    invalidator = threading.Thread(target=invalidate_and_rebegin_same_uuid)
    invalidator.start()
    assert invalidated.wait(0.2)
    release_observation_write.set()
    worker.join(1.0)
    invalidator.join(1.0)

    assert result == [None]
    assert capture.active_generation_id == generation_id
    assert history.anchors == ()
    assert coordinator.candidates == []


def test_native_candidate_validation_rejects_hash_or_request_mismatch() -> None:
    response = _response()
    validate_native_candidate_response(
        response,
        request_id="request-1",
        native_action_dim=16,
        chunk_size=50,
    )
    tampered = deepcopy(response)
    tampered["native_actions"][0, 0] += 1
    with pytest.raises(ValueError, match="content hash"):
        validate_native_candidate_response(
            tampered,
            request_id="request-1",
            native_action_dim=16,
            chunk_size=50,
        )
    with pytest.raises(ValueError, match="request_id"):
        validate_native_candidate_response(
            response,
            request_id="another-request",
            native_action_dim=16,
            chunk_size=50,
        )
    other_model = deepcopy(response)
    other_model["sampling"]["algorithm"] = "other_model"
    validate_native_candidate_response(
        other_model,
        request_id="request-1",
        native_action_dim=16,
        chunk_size=50,
    )
    other_model["sampling"] = "invalid"
    with pytest.raises(ValueError, match="sampling provenance"):
        validate_native_candidate_response(
            other_model,
            request_id="request-1",
            native_action_dim=16,
            chunk_size=50,
        )


def test_capture_rejects_non_uuid_generation() -> None:
    capture = FlowPROInferenceCapture(FakeCoordinator(), native_action_dim=16, chunk_size=50)
    with pytest.raises(ValueError, match="must be a UUID"):
        capture.begin_generation("not-a-generation-uuid")


def test_capture_validates_server_contract() -> None:
    capture = FlowPROInferenceCapture(FakeCoordinator(), native_action_dim=16, chunk_size=50)
    metadata = {
        "chunk_size": 50,
        "action_capture": {
            "available": True,
            "native_action_space": "absolute_unnormalized",
            "native_action_dim": 16,
            "content_sha256": True,
        },
    }
    capture.validate_server_metadata(metadata)
    metadata["chunk_size"] = 32
    with pytest.raises(ValueError, match="chunk size"):
        capture.validate_server_metadata(metadata)


def _logical(name: str, timestamp_ns: int) -> LogicalInput:
    return LogicalInput(name=name, source_timestamp_ns=timestamp_ns, detail={})


def _ready_handback() -> HumanHandbackEvidence:
    return HumanHandbackEvidence(
        side_buttons_released=True,
        accepted_retarget_fresh=False,
        action_src="IDLE",
        robot_settled=True,
        measured_state_fresh=True,
        cameras_fresh=True,
        network_fresh=True,
        motion_healthy=True,
    )


def test_session_controller_reuses_supervisor_generation_and_exits_pending_tablet_commit() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    runtime = FlowPROSessionController(coordinator, capture)

    started = runtime.start_session(
        "/data/record/vendor-episode",
        _logical("flowpro_mode_toggle", 100),
        flowpro_session_id="flowpro-session-1",
    )
    assert started.current == SupervisorMode.HOLD
    auto = runtime.handle(_logical("y_short", 102))
    assert auto is not None and auto.current == SupervisorMode.AUTO
    assert capture.active_generation_id == auto.generation_id
    first_generation = auto.generation_id

    takeover = runtime.handle(_logical("human_takeover", 103))
    assert takeover is not None and takeover.current == SupervisorMode.HUMAN
    assert not capture.active
    handback = runtime.handle(_logical("human_finish", 104), handback_evidence=_ready_handback())
    assert handback is not None and handback.current == SupervisorMode.HOLD

    forward = runtime.handle(_logical("y_long_start", 105))
    assert forward is not None and forward.current == SupervisorMode.FORWARD_RUN
    assert capture.active_generation_id == forward.generation_id
    assert forward.generation_id != first_generation
    stopped = runtime.handle(_logical("y_long_end", 106))
    assert stopped is not None and stopped.current == SupervisorMode.HOLD
    assert not capture.active
    assert [owner for _, owner in coordinator.logical_inputs[:6]] == [
        "hold",
        "hold",
        "policy",
        "human",
        "hold",
        "policy",
    ]

    finished = runtime.handle(_logical("flowpro_mode_toggle", 108))
    assert finished is not None and finished.current == SupervisorMode.DISABLED
    assert coordinator.finished == [
        (
            "pending_recorder_commit",
            {
                "outcome_authority": "external_tablet_recorder",
                "saved_vendor_episode_outcome": "success",
            },
        )
    ]
    assert runtime.flowpro_session_id is None


def test_y_press_starts_session_toggles_policy_and_tablet_final_meta_ends_session() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    runtime = FlowPROSessionController(coordinator, capture)

    started = runtime.start_session(
        "/data/record/vendor-episode",
        _logical("y_press", 200),
        flowpro_session_id="flowpro-session-y-toggle",
    )
    assert started.previous == SupervisorMode.DISABLED
    assert started.current == SupervisorMode.AUTO
    assert capture.active_generation_id == started.generation_id

    stopped = runtime.handle(_logical("y_press", 201))
    assert stopped is not None and stopped.current == SupervisorMode.HOLD
    assert stopped.generation_invalidated
    assert not capture.active

    finished = runtime.finish_from_tablet(timestamp_ns=202)
    assert finished.current == SupervisorMode.DISABLED
    assert runtime.flowpro_session_id is None
    assert coordinator.finished[-1][0] == "pending_recorder_commit"


@pytest.mark.parametrize("event_name", ["vr_stale", "watchdog", "network_lost", "motion_unhealthy"])
def test_transient_health_loss_does_not_bypass_human_handback(event_name: str) -> None:
    supervisor = RolloutSupervisor()
    supervisor.mode = SupervisorMode.HUMAN

    transition = supervisor.handle(event_name)

    assert transition is None
    assert supervisor.mode == SupervisorMode.HUMAN


@pytest.mark.parametrize(
    ("event_name", "start_mode", "end_mode"),
    [
        ("y_long_end", SupervisorMode.FORWARD_RUN, SupervisorMode.HOLD),
        ("vr_stale", SupervisorMode.AUTO, SupervisorMode.HOLD),
        ("side_takeover_input", SupervisorMode.AUTO, SupervisorMode.HOLD),
        ("emergency", SupervisorMode.AUTO, SupervisorMode.ABORT),
    ],
)
def test_policy_stop_is_physically_held_before_durable_event_write(
    event_name: str,
    start_mode: SupervisorMode,
    end_mode: SupervisorMode,
) -> None:
    entered_write = threading.Event()
    allow_write = threading.Event()
    hold_called = threading.Event()

    class Session:
        def __init__(self) -> None:
            self.supervisor = SimpleNamespace(mode=start_mode)

        def handle(self, logical_input, **_kwargs):
            assert hold_called.is_set()
            entered_write.set()
            assert allow_write.wait(1.0)
            previous = self.supervisor.mode
            self.supervisor.mode = end_mode
            return SupervisorTransition(
                previous=previous,
                current=end_mode,
                reason=logical_input.name,
                generation_id=None,
                generation_invalidated=True,
            )

    session = Session()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=SimpleNamespace(state=InterventionState.IDLE),
        rollback_executor=SimpleNamespace(enter_hold=hold_called.set),
        rollback_planner=lambda *_args: None,
    )
    results = []
    worker = threading.Thread(target=lambda: results.append(paper.handle(_logical(event_name, 260))))
    worker.start()

    assert entered_write.wait(1.0)
    assert hold_called.is_set()
    assert worker.is_alive()
    allow_write.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert results[0].current == end_mode


@pytest.mark.parametrize("event_name", ["human_takeover", "external_authority"])
def test_authoritative_human_event_never_reenables_local_hold_writer(event_name: str) -> None:
    hold_called = threading.Event()

    class Session:
        def __init__(self) -> None:
            self.supervisor = SimpleNamespace(mode=SupervisorMode.AUTO)

        def handle(self, logical_input, **_kwargs):
            assert not hold_called.is_set()
            previous = self.supervisor.mode
            self.supervisor.mode = SupervisorMode.HUMAN
            return SupervisorTransition(
                previous=previous,
                current=SupervisorMode.HUMAN,
                reason=logical_input.name,
                generation_id=None,
                generation_invalidated=True,
            )

    paper = FlowPROPaperRuntime(
        session=Session(),
        intervention=SimpleNamespace(state=InterventionState.IDLE),
        rollback_executor=SimpleNamespace(enter_hold=hold_called.set),
        rollback_planner=lambda *_args: None,
    )

    result = paper.handle(_logical(event_name, 261))

    assert result.current == SupervisorMode.HUMAN
    assert not hold_called.is_set()


@pytest.mark.parametrize("mode", [SupervisorMode.AUTO, SupervisorMode.HUMAN, SupervisorMode.ABORT])
def test_non_hold_y_input_reports_a_blocker_without_poisoning_runtime(mode: SupervisorMode) -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    runtime = FlowPROSessionController(coordinator, capture)
    runtime.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 270))
    runtime.supervisor.mode = mode
    paper = FlowPROPaperRuntime(
        session=runtime,
        intervention=SimpleNamespace(state=InterventionState.IDLE),
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: None,
    )

    assert paper.handle(_logical("y_short", 271)) is None
    assert runtime.supervisor.mode == mode
    assert paper.rollback_status.code == "input_blocked"
    assert "y_short:" in paper.rollback_status.detail


def test_release_during_blocking_prepare_cannot_publish_rollback_lease() -> None:
    prepare_started = threading.Event()
    allow_prepare = threading.Event()
    commits = []

    class Driver:
        def prepare_arm(self, _plan, *, lease_id):
            prepare_started.set()
            assert allow_prepare.wait(1.0)
            self.prepared_lease_id = lease_id

        def commit_prepared(self, *, lease_id, deadman_pressed, health):
            commits.append((lease_id, deadman_pressed, health))

    class Session:
        def __init__(self) -> None:
            self.supervisor = SimpleNamespace(mode=SupervisorMode.ROLLBACK_EXEC)
            self.coordinator = FakeCoordinator()

        def set_label_context(self, _context) -> None:
            return None

        def _control_owner(self) -> str:
            return self.supervisor.mode.value

    session = Session()
    intervention = SimpleNamespace(
        state=InterventionState.ROLLBACK_EXEC,
        rollback_plan=object(),
    )
    driver = Driver()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=driver,
        rollback_planner=lambda *_args: None,
    )
    paper._activation_pending = True
    paper._activation_lease_id = "prepared-lease"
    paper._activation_barrier_revision = 1
    paper._execution_requested = True
    health = SimpleNamespace(
        safe=True,
        evaluated_monotonic_ns=time.monotonic_ns(),
    )
    worker = threading.Thread(
        target=lambda: paper.rollback_tick(
            x_pressed=True,
            health=health,
            vr_frame_revision=2,
            now_ns=300,
        )
    )
    worker.start()
    assert prepare_started.wait(1.0)

    paper.handle(_logical("x_release", 301))
    allow_prepare.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert paper.rollback_lease_id is None
    assert paper._activation_prepared
    assert commits == []
    paper.rollback_tick(
        x_pressed=False,
        health=health,
        vr_frame_revision=3,
        now_ns=302,
    )
    assert commits == []

    paper.handle(_logical("x_press", 303))
    fresh_health = SimpleNamespace(
        safe=True,
        evaluated_monotonic_ns=time.monotonic_ns(),
    )
    paper.rollback_tick(
        x_pressed=True,
        health=fresh_health,
        vr_frame_revision=4,
        now_ns=304,
    )
    assert len(commits) == 1
    assert paper.rollback_lease_id == "prepared-lease"


def test_abort_finalization_is_explicit_and_never_controls_tablet_recorder() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    runtime = FlowPROSessionController(coordinator, capture)
    runtime.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 300))
    runtime.handle(_logical("y_short", 301))

    aborted = runtime.handle(_logical("emergency", 302))
    assert aborted is not None and aborted.current == SupervisorMode.ABORT
    assert not capture.active
    state = runtime.finalize_abort(detail={"reason": "physical_estop"})
    assert state["outcome"] == "aborted"
    assert coordinator.finished == [("aborted", {"reason": "physical_estop"})]


def test_internal_resume_failure_forces_a_durable_hold_and_invalidates_generation() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    runtime = FlowPROSessionController(coordinator, capture)
    runtime.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 350))
    running = runtime.handle(_logical("y_short", 351))
    assert running is not None and running.current == SupervisorMode.AUTO
    assert capture.active

    held = runtime.fail_closed_hold(reason="watchdog", timestamp_ns=352)
    assert held is not None and held.current == SupervisorMode.HOLD
    assert not capture.active
    assert coordinator.logical_inputs[-1][0].detail == {"source": "g2_runtime"}


def test_exit_stays_in_hold_when_durable_finish_fails() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    runtime = FlowPROSessionController(coordinator, capture)
    runtime.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 400))
    coordinator.fail_finish = True

    with pytest.raises(OSError, match="durable write failure"):
        runtime.handle(_logical("flowpro_mode_toggle", 401))
    assert runtime.supervisor.mode == SupervisorMode.HOLD
    assert runtime.flowpro_session_id is not None


def test_paper_rollback_continuous_x_plans_runs_pauses_and_resumes() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    snapshot = SceneRestorationSnapshot(
        observation_id=str(uuid4()),
        anchor_timestamp_ns=2_900_000_000,
        task="test",
        state_path=Path("state.npy"),
        image_paths={},
        image_timestamps_ns={},
    )

    class Intervention:
        state = InterventionState.IDLE
        intervention_id = None
        rollback_session_id = None
        rollback_cursor = 0
        rollback_plan = None
        anchor_selected_for_restoration = False
        can_continue_selected_rollback = True

        def freeze_for_continuous_rollback(self, **_kwargs):
            self.intervention_id = str(uuid4())
            self.rollback_session_id = str(uuid4())
            self.state = InterventionState.ROLLBACK_PLANNING
            return SimpleNamespace(), np.zeros(24, dtype=np.float32)

        def complete_continuous_plan(self, plan):
            self.rollback_plan = plan
            self.state = InterventionState.ROLLBACK_READY
            return snapshot

        def begin_rollback(self, *, resume_detail=None, resumed_timestamp_ns=None):
            assert resumed_timestamp_ns is not None
            assert self.state in {
                InterventionState.ROLLBACK_READY,
                InterventionState.ROLLBACK_PAUSED,
            }
            if resume_detail is not None:
                assert resume_detail["resume_from_cursor"] == self.rollback_cursor
            self.anchor_selected_for_restoration = False
            self.state = InterventionState.ROLLBACK_EXEC

        def observe_rollback_progress(self, **_kwargs):
            return ()

        def pause_rollback(self, **kwargs):
            assert self.state == InterventionState.ROLLBACK_EXEC
            assert kwargs["anchor_observation_id"] == snapshot.observation_id
            self.rollback_cursor = kwargs["cursor"]
            self.anchor_selected_for_restoration = True
            self.state = InterventionState.ROLLBACK_PAUSED
            return snapshot

        def complete_rollback(self, **_kwargs):
            assert self.state == InterventionState.ROLLBACK_EXEC
            self.state = InterventionState.SCENE_RESTORE
            return snapshot

    class Executor:
        def __init__(self):
            self.disarmed = False
            self.paused = False
            self.resumed = False
            self.arm_count = 0
            self.auto_finish_count = 0

        def arm(self, _plan, *, lease_id):
            assert lease_id
            self.arm_count += 1

        def resume(self, *, lease_id):
            assert lease_id
            self.resumed = True
            return {"resume_from_cursor": 3, "resume_join_cursor": 3}

        def step(self, *, deadman_pressed, finish_to_anchor_authorized, **_kwargs):
            if not deadman_pressed:
                assert finish_to_anchor_authorized
                self.auto_finish_count += 1
                if self.auto_finish_count == 1:
                    return SimpleNamespace(
                        state=RollbackExecutorState.RUNNING,
                        command=np.zeros(24, dtype=np.float32),
                        measured_state=np.zeros(24, dtype=np.float32),
                        source_timestamp_ns=2_950_000_000,
                        waypoint_index=2,
                        anchor_observation_id=snapshot.observation_id,
                        reason="rollback_waypoint_sent",
                    )
                return SimpleNamespace(
                    state=RollbackExecutorState.HOLD,
                    command=np.zeros(24, dtype=np.float32),
                    measured_state=np.zeros(24, dtype=np.float32),
                    source_timestamp_ns=2_900_000_000,
                    waypoint_index=3,
                    anchor_observation_id=snapshot.observation_id,
                    reason="rollback_anchor_reached_stationary",
                )
            if not self.resumed:
                return SimpleNamespace(
                    state=RollbackExecutorState.RUNNING,
                    command=np.zeros(24, dtype=np.float32),
                    measured_state=np.zeros(24, dtype=np.float32),
                    source_timestamp_ns=2_975_000_000,
                    waypoint_index=1,
                    anchor_observation_id=None,
                    reason="rollback_waypoint_sent",
                )
            return SimpleNamespace(
                state=RollbackExecutorState.COMPLETE,
                command=np.zeros(24, dtype=np.float32),
                measured_state=np.zeros(24, dtype=np.float32),
                source_timestamp_ns=2_000_000_000,
                waypoint_index=10,
                anchor_observation_id=snapshot.observation_id,
                reason="rollback_complete",
            )

        def pause(self):
            self.paused = True

        def disarm(self):
            self.disarmed = True

    intervention = Intervention()
    executor = Executor()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=executor,
        rollback_planner=lambda *_args: SimpleNamespace(),
    )
    session.handle(_logical("y_short", 2_000_000_000))
    stopped = paper.handle(_logical("x_press", 3_000_000_000))
    assert stopped.current == SupervisorMode.HOLD

    for index in range(100):
        paper.rollback_tick(
            x_pressed=True,
            health=SimpleNamespace(safe=False),
            vr_frame_revision=10,
            now_ns=3_100_000_001 + index,
        )
        if intervention.state == InterventionState.ROLLBACK_READY:
            break
        time.sleep(0.001)
    assert intervention.state == InterventionState.ROLLBACK_READY
    assert executor.arm_count == 0

    assert (
        paper.rollback_tick(
            x_pressed=True,
            health=SimpleNamespace(safe=True),
            vr_frame_revision=10,
            now_ns=3_100_000_999,
        )
        is None
    )
    assert paper.rollback_status.code == "awaiting_fresh_vr_frame"
    assert session.supervisor.mode == SupervisorMode.HOLD
    started = paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        vr_frame_revision=11,
        now_ns=3_100_001_000,
    )
    assert started.current == SupervisorMode.ROLLBACK_EXEC
    assert paper.rollback_lease_id is None
    assert (
        paper.rollback_tick(
            x_pressed=True,
            health=SimpleNamespace(safe=True),
            vr_frame_revision=11,
            now_ns=3_100_001_001,
        )
        is None
    )
    assert paper.rollback_lease_id is None
    assert (
        paper.rollback_tick(
            x_pressed=True,
            health=SimpleNamespace(safe=True),
            vr_frame_revision=12,
            now_ns=3_100_001_002,
        )
        is None
    )
    assert executor.arm_count == 1
    original_lease = paper.rollback_lease_id

    paper.handle(_logical("x_release", 3_200_000_000))
    auto = paper.rollback_tick(
        x_pressed=False,
        health=SimpleNamespace(safe=True),
        now_ns=3_200_000_001,
    )
    assert auto.state == RollbackExecutorState.RUNNING
    assert paper.rollback_lease_id == original_lease

    assert paper.handle(_logical("x_press", 3_200_000_002)) is None
    continued = paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        now_ns=3_200_000_003,
    )
    assert continued.state == RollbackExecutorState.RUNNING
    assert paper.rollback_lease_id == original_lease
    assert not executor.resumed

    paper.handle(_logical("x_release", 3_250_000_000))
    held = paper.rollback_tick(
        x_pressed=False,
        health=SimpleNamespace(safe=True),
        now_ns=3_250_000_001,
    )
    assert held.current == SupervisorMode.HOLD
    assert executor.paused
    assert paper.rollback_lease_id is None

    # Second X resumes from the selected cursor directly; no Y is sent.
    assert paper.handle(_logical("x_press", 3_260_000_000)) is None
    resumed = paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        now_ns=3_260_000_001,
    )
    assert resumed.current == SupervisorMode.ROLLBACK_EXEC
    assert paper.rollback_lease_id is None
    assert (
        paper.rollback_tick(
            x_pressed=True,
            health=SimpleNamespace(safe=True),
            now_ns=3_260_000_002,
        )
        is None
    )
    assert paper.rollback_lease_id != original_lease
    assert executor.resumed
    assert any(
        logical_input.name == "rollback_execution_bound" and owner == "rollback"
        for logical_input, owner in paper._deferred_logical_inputs
    )
    completed = paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        now_ns=3_300_000_001,
    )
    assert completed.current == SupervisorMode.HOLD
    assert intervention.state == InterventionState.SCENE_RESTORE
    assert executor.disarmed


def test_terminal_rollback_fault_still_enters_hold_when_wal_is_unavailable() -> None:
    """A diagnostic write failure must not kill the button/DDS coordinator."""

    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    started = session.handle(_logical("rollback_start", 2))
    assert started is not None and started.current == SupervisorMode.ROLLBACK_EXEC

    class Intervention:
        state = InterventionState.ROLLBACK_EXEC

        def invalidate_rollback(self, **_kwargs):
            self.state = InterventionState.REJECTED

    class Executor:
        def __init__(self):
            self.cancelled = False

        def step(self, **_kwargs):
            return SimpleNamespace(
                state=RollbackExecutorState.CANCELLED,
                reason="simulated_terminal_fault",
                command=np.zeros(24, dtype=np.float32),
            )

        def cancel(self, **_kwargs):
            self.cancelled = True

    executor = Executor()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=Intervention(),
        rollback_executor=executor,
        rollback_planner=lambda *_args: None,
    )
    paper.rollback_lease_id = "rollback-lease"
    paper._executor_armed = True

    def fail_wal(*_args, **_kwargs):
        raise OSError("simulated WAL failure")

    coordinator.append_logical_input = fail_wal
    result = paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        now_ns=3,
    )

    assert result is not None and result.current == SupervisorMode.HOLD
    assert session.supervisor.mode == SupervisorMode.HOLD
    assert paper.rollback_lease_id is None
    assert executor.cancelled
    assert paper.rollback_status.code == "invalidated"


def test_continuous_x_can_start_from_hold_after_y_release() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    session.handle(_logical("y_long_start", 2_000_000_000))
    session.handle(_logical("y_long_end", 2_900_000_000))
    last_execution_timestamp_ns = 2_850_000_000
    intervention = SimpleNamespace(
        state=InterventionState.IDLE,
        history=SimpleNamespace(samples=(SimpleNamespace(timestamp_ns=last_execution_timestamp_ns),)),
    )
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: None,
    )

    result = paper.handle(_logical("x_press", 10_000_000_000))

    assert result is None
    assert session.supervisor.mode == SupervisorMode.HOLD
    assert paper.planning_requested
    assert paper.failure_timestamp_ns == last_execution_timestamp_ns
    event, owner = coordinator.logical_inputs[-1]
    assert event.name == "x_press"
    assert owner == "hold"


def test_x_release_during_planning_keeps_plan_but_revokes_execution_permission() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    session.handle(_logical("y_short", 2_000_000_000))
    release_planner = threading.Event()

    class Intervention:
        state = InterventionState.IDLE
        intervention_id = None
        rollback_session_id = None
        rollback_cursor = 0
        rollback_plan = None

        def freeze_for_continuous_rollback(self, **_kwargs):
            self.intervention_id = str(uuid4())
            self.rollback_session_id = str(uuid4())
            self.state = InterventionState.ROLLBACK_PLANNING
            return SimpleNamespace(), np.zeros(24, dtype=np.float32)

        def complete_continuous_plan(self, plan):
            self.rollback_plan = plan
            self.state = InterventionState.ROLLBACK_READY
            return SimpleNamespace(observation_id=str(uuid4()))

        def begin_rollback(self, *, resumed_timestamp_ns=None):
            assert resumed_timestamp_ns is not None
            self.state = InterventionState.ROLLBACK_EXEC

    class Executor:
        armed = 0

        def arm(self, _plan, *, lease_id):
            assert lease_id
            self.armed += 1

    measured_hold = np.linspace(-0.2, 0.2, 24, dtype=np.float32)

    def planner(_segment, _intervention_id, current_state):
        np.testing.assert_allclose(current_state, measured_hold)
        assert release_planner.wait(1)
        return SimpleNamespace()

    intervention = Intervention()
    executor = Executor()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=executor,
        rollback_planner=planner,
        measured_state_provider=lambda: measured_hold.copy(),
    )
    paper.handle(_logical("x_press", 3_000_000_000))
    paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        now_ns=3_100_000_001,
    )
    assert intervention.state == InterventionState.ROLLBACK_PLANNING

    paper.handle(_logical("x_release", 3_200_000_000))
    release_planner.set()
    for index in range(100):
        paper.rollback_tick(
            x_pressed=False,
            health=SimpleNamespace(safe=True),
            now_ns=3_200_000_001 + index,
        )
        if intervention.state == InterventionState.ROLLBACK_READY:
            break
        time.sleep(0.001)
    assert intervention.state == InterventionState.ROLLBACK_READY
    assert executor.armed == 0
    assert session.supervisor.mode == SupervisorMode.HOLD

    resumed = paper.handle(_logical("x_press", 3_300_000_000))
    assert resumed is None
    resumed = paper.rollback_tick(
        x_pressed=True,
        health=SimpleNamespace(safe=True),
        now_ns=3_300_000_001,
    )
    assert resumed.current == SupervisorMode.ROLLBACK_EXEC
    assert executor.armed == 0
    assert (
        paper.rollback_tick(
            x_pressed=True,
            health=SimpleNamespace(safe=True),
            now_ns=3_300_000_002,
        )
        is None
    )
    assert executor.armed == 1


def test_sustained_x_starts_planning_on_next_accepted_vr_frame() -> None:
    class ExpectedFreezeError(RuntimeError):
        pass

    class Intervention:
        state = InterventionState.IDLE
        freeze_calls = 0

        def freeze_for_continuous_rollback(self, **_kwargs):
            self.freeze_calls += 1
            raise ExpectedFreezeError("next frame reached history freeze")

    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    session.handle(_logical("y_short", 2_000_000_000))
    intervention = Intervention()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: pytest.fail("sentinel freeze must stop before planner"),
    )
    paper.handle(
        LogicalInput(
            "x_press",
            3_000_000_000,
            {"accepted_vr_frame_revision": 50},
        )
    )

    result = paper.rollback_tick(
        x_pressed=True,
        y_pressed=False,
        health=SimpleNamespace(safe=True),
        vr_frame_revision=51,
        now_ns=3_500_000_000,
    )
    assert intervention.freeze_calls == 1
    assert result.reason.startswith("ExpectedFreezeError:")


def test_disabled_rollback_capability_stops_policy_without_planning() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    session.handle(_logical("y_short", 2))
    executor = SimpleNamespace(
        arm=lambda *_args, **_kwargs: pytest.fail("disabled rollback must not arm an executor")
    )
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=SimpleNamespace(state=InterventionState.IDLE),
        rollback_executor=executor,
        rollback_planner=lambda *_args: pytest.fail("disabled rollback must not build a plan"),
        rollback_enabled=False,
    )

    stopped = paper.handle(_logical("x_press", 3))
    assert stopped.current == SupervisorMode.HOLD
    assert session.supervisor.mode == SupervisorMode.HOLD
    event, owner = coordinator.logical_inputs[-1]
    assert owner == "hold"
    assert event.name == "rollback_unavailable"
    assert event.detail["reason"] == "rollback_capability_disabled"
    paper.handle(_logical("x_short", 4))
    assert session.supervisor.mode == SupervisorMode.HOLD


def test_ready_y_gate_is_visible_and_recovers_without_an_extra_x_press() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))

    class Intervention:
        state = InterventionState.ROLLBACK_READY
        rollback_plan = SimpleNamespace()
        rollback_session_id = str(uuid4())
        rollback_cursor = 0

        def begin_rollback(self, *, resumed_timestamp_ns=None):
            assert resumed_timestamp_ns is not None
            self.state = InterventionState.ROLLBACK_EXEC

    class Executor:
        armed = 0

        def arm(self, _plan, *, lease_id):
            assert lease_id
            self.armed += 1

    intervention = Intervention()
    executor = Executor()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=executor,
        rollback_planner=lambda *_args: SimpleNamespace(),
    )
    paper._execution_requested = True

    assert (
        paper.rollback_tick(
            x_pressed=True,
            y_pressed=True,
            health=SimpleNamespace(safe=True),
            now_ns=10,
        )
        is None
    )
    assert paper.rollback_status.code == "blocked_y_pressed"
    prepared = paper.rollback_tick(
        x_pressed=True,
        y_pressed=False,
        health=SimpleNamespace(safe=True),
        now_ns=11,
    )
    assert prepared.current == SupervisorMode.ROLLBACK_EXEC
    paper.rollback_tick(
        x_pressed=True,
        y_pressed=False,
        health=SimpleNamespace(safe=True),
        now_ns=12,
    )
    assert executor.armed == 1
    assert paper.rollback_status.code == "rewinding"


def test_rollback_exception_forces_supervisor_hold_and_keeps_next_x_retryable() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    session.supervisor.mode = SupervisorMode.ROLLBACK_EXEC
    session.supervisor.generation_id = str(uuid4())

    class Intervention:
        state = InterventionState.ROLLBACK_EXEC
        history = SimpleNamespace(samples=[])
        rollback_cursor = 0
        segment = SimpleNamespace(generation_ids=("generation-a",))

        def planning_failed(self, *, reason, timestamp_ns):
            assert "injected progress failure" in reason
            assert timestamp_ns == 100
            self.state = InterventionState.REJECTED

    class Executor:
        cancelled = False

        def cancel(self, *, reason):
            assert reason == "rollback_planning_or_resume_failed"
            self.cancelled = True

    intervention = Intervention()
    executor = Executor()
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=executor,
        rollback_planner=lambda *_args: SimpleNamespace(),
    )
    paper._executor_armed = True
    paper.rollback_lease_id = "failed-lease"

    failed = paper._planning_failed(RuntimeError("injected progress failure"), 100)

    assert failed.current == SupervisorMode.HOLD
    assert session.supervisor.mode == SupervisorMode.HOLD
    assert intervention.state == InterventionState.REJECTED
    assert executor.cancelled
    assert paper.rollback_lease_id is None

    # The same tablet episode can issue another X; it no longer gets trapped in
    # the impossible REJECTED + ROLLBACK_EXEC state.
    assert paper.handle(_logical("x_press", 101)) is None
    assert paper.planning_requested


def test_progressed_rejected_rollback_cannot_replan_same_policy_path() -> None:
    coordinator = FakeCoordinator()
    capture = FlowPROInferenceCapture(coordinator, native_action_dim=16, chunk_size=50)
    session = FlowPROSessionController(coordinator, capture)
    session.start_session("/data/record/vendor-episode", _logical("flowpro_mode_toggle", 1))
    session.supervisor.mode = SupervisorMode.HOLD

    intervention = SimpleNamespace(
        state=InterventionState.REJECTED,
        history=SimpleNamespace(samples=[]),
        rollback_cursor=1148,
        segment=SimpleNamespace(generation_ids=("generation-a",)),
    )
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: SimpleNamespace(),
        policy_run_generation_provider=lambda: "generation-a",
    )

    assert paper.handle(_logical("x_press", 101)) is None
    assert not paper.planning_requested
    assert paper.rollback_status.code == "rollback_replay_blocked"
    logical_input, owner = coordinator.logical_inputs[-1]
    assert logical_input.name == "rollback_replay_blocked"
    assert owner == "hold"


def test_live_rollback_repress_defers_contract_owner_not_supervisor_state() -> None:
    coordinator = FakeCoordinator()
    session = SimpleNamespace(
        supervisor=SimpleNamespace(mode=SupervisorMode.ROLLBACK_EXEC),
        coordinator=coordinator,
    )
    intervention = SimpleNamespace(state=InterventionState.ROLLBACK_EXEC)
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: SimpleNamespace(),
    )
    paper.rollback_lease_id = "live-lease"

    assert paper.handle(_logical("x_press", 101)) is None
    assert paper._deferred_logical_inputs[0][1] == "rollback"
    paper._flush_deferred_logical_inputs()
    assert coordinator.logical_inputs[-1][1] == "rollback"


def test_side_button_intent_does_not_cancel_live_rollback() -> None:
    coordinator = FakeCoordinator()
    session = SimpleNamespace(
        supervisor=SimpleNamespace(mode=SupervisorMode.ROLLBACK_EXEC),
        coordinator=coordinator,
        _control_owner=lambda: "rollback",
    )
    intervention = SimpleNamespace(state=InterventionState.ROLLBACK_EXEC)
    paper = FlowPROPaperRuntime(
        session=session,
        intervention=intervention,
        rollback_executor=SimpleNamespace(),
        rollback_planner=lambda *_args: SimpleNamespace(),
    )
    paper.rollback_lease_id = "live-lease"

    assert paper.handle(_logical("side_takeover_input", 102)) is None
    assert intervention.state == InterventionState.ROLLBACK_EXEC
    assert session.supervisor.mode == SupervisorMode.ROLLBACK_EXEC
    assert paper.rollback_lease_id == "live-lease"
    assert paper._deferred_logical_inputs[-1][0].name == "side_takeover_input"


def test_authoritative_takeover_at_anchor_implicitly_verifies_scene_and_starts_correction() -> None:
    calls: list[tuple[str, object]] = []

    class Session:
        def __init__(self) -> None:
            self.supervisor = SimpleNamespace(mode=SupervisorMode.HOLD)

        def handle(self, logical_input, **_kwargs):
            assert logical_input.name == "human_takeover"
            self.supervisor.mode = SupervisorMode.HUMAN
            calls.append(("session", logical_input.source_timestamp_ns))
            return SupervisorTransition(
                previous=SupervisorMode.HOLD,
                current=SupervisorMode.HUMAN,
                reason="human_takeover",
                generation_id=None,
                generation_invalidated=False,
            )

    class Intervention:
        state = InterventionState.ROLLBACK_PAUSED
        anchor_selected_for_restoration = True

        def finalize_selected_anchor(self, *, completed_timestamp_ns):
            calls.append(("finalize", completed_timestamp_ns))
            self.state = InterventionState.SCENE_RESTORE

        def verify_scene(self, *, valid, verified_timestamp_ns, verification_method):
            calls.append(("verify", (valid, verified_timestamp_ns, verification_method)))
            self.state = InterventionState.CORRECTION_READY

        def start_correction(self, *, start_timestamp_ns):
            calls.append(("correction", start_timestamp_ns))
            self.state = InterventionState.CORRECTION

    executor = SimpleNamespace(finish_paused=lambda: calls.append(("finish_paused", True)))
    paper = FlowPROPaperRuntime(
        session=Session(),
        intervention=Intervention(),
        rollback_executor=executor,
        rollback_planner=lambda *_args: SimpleNamespace(),
    )
    paper._executor_armed = True

    result = paper.handle(_logical("human_takeover", 200))

    assert result.current == SupervisorMode.HUMAN
    assert calls == [
        ("finalize", 200),
        ("finish_paused", True),
        ("verify", (True, 200, "implicit_human_takeover")),
        ("correction", 200),
        ("session", 200),
    ]
    assert paper.intervention.state == InterventionState.CORRECTION
