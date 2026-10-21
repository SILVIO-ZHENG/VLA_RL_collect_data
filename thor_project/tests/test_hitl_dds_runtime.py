from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from nexus_deploy.robot.g2a.constants import GDK_BODY_JOINTS
from nexus_deploy.robot.g2a.hitl.g2_runtime.dds_runtime import G2DDSRuntimeCoordinator, G2DDSRuntimeService
from nexus_deploy.robot.g2a.hitl.g2_runtime.g2_dds import (
    AcceptedRetargetSample,
    ActionSourceSample,
    BodyJointSample,
    DeviceInputSnapshot,
    EndEffectorSample,
    G2DDSRead,
    MotionStatusSample,
    PublicFaultSample,
    VRNetworkSample,
    VRTeleopSample,
    WholeBodySafetySample,
)
from nexus_deploy.robot.g2a.hitl.g2_runtime.supervisor import SupervisorMode


class _Application:
    def __init__(self, mode: SupervisorMode = SupervisorMode.HOLD) -> None:
        self.session = SimpleNamespace(supervisor=SimpleNamespace(mode=mode))
        self.frames = []
        self.takeovers = []
        self.external_preempts = []
        self.handbacks = []
        self.handback_prepares = 0
        self.human_restarts = 0
        self.rollback_health = []

    def handle_vr_frame(self, frame, *, received_timestamp_ns=None):
        self.frames.append((frame, received_timestamp_ns))
        return []

    def advance_vr_clock(self, now_ns: int):
        return []

    def human_takeover(self, timestamp_ns: int, *, detail):
        self.takeovers.append((timestamp_ns, detail))
        self.session.supervisor.mode = SupervisorMode.HUMAN

    def external_authority_preempt(self, timestamp_ns: int, *, detail):
        self.external_preempts.append((timestamp_ns, detail))
        # Actual vendor authority is already Human.  The runtime must remain
        # suspended until the same fail-closed handback used by an intentional
        # side-button takeover completes, while keeping this path unlabelled.
        self.session.supervisor.mode = SupervisorMode.HUMAN

    def human_finish(self, timestamp_ns: int, *, evidence):
        self.handbacks.append((timestamp_ns, evidence))
        self.session.supervisor.mode = SupervisorMode.HOLD

    def prepare_human_handback(self):
        self.handback_prepares += 1
        return True

    def restart_human_control_hold(self):
        self.human_restarts += 1

    def update_rollback(self, health):
        self.rollback_health.append(health)


def _read(label: str, sample: object, *, publish_timestamp_ns: int = 1) -> G2DDSRead:
    receive_timestamp_ns = int(getattr(sample, "timestamp_ns", publish_timestamp_ns))
    return G2DDSRead(label, sample, publish_timestamp_ns, receive_timestamp_ns, 3, 4)


def _vr(timestamp_ns: int, *, hand: bool) -> VRTeleopSample:
    return VRTeleopSample(
        timestamp_ns=timestamp_ns,
        frame_index=timestamp_ns,
        connected=True,
        devices=(
            DeviceInputSnapshot(0, False, 0.0, 0.0, 0.0, float(hand), False, False),
            DeviceInputSnapshot(1, False, 0.0, 0.0, 0.0, 0.0, False, False),
            DeviceInputSnapshot(2, False, 0.0, 0.0, 0.0, 0.0, False, False),
        ),
    )


def _body(timestamp_ns: int) -> BodyJointSample:
    return BodyJointSample(
        timestamp_ns=timestamp_ns,
        positions=dict.fromkeys(GDK_BODY_JOINTS, 0.0),
        velocities=dict.fromkeys(GDK_BODY_JOINTS, 0.0),
    )


def _feed_health(coordinator: G2DDSRuntimeCoordinator, timestamp_ns: int, *, hand: bool) -> None:
    coordinator.consume(
        _read(
            "vr_network",
            VRNetworkSample(timestamp_ns, True, 0.0, 1.0, 0.0, "g2", "pico"),
        )
    )
    coordinator.consume(_read("vr_data", _vr(timestamp_ns, hand=hand)))
    coordinator.consume(_read("motion_status", MotionStatusSample(timestamp_ns, 0, 1, "")))
    coordinator.consume(
        _read(
            "whole_body_status",
            WholeBodySafetySample(False, False, {"left_arm_error": 0}),
            publish_timestamp_ns=timestamp_ns,
        )
    )
    coordinator.consume(_read("public_fault", PublicFaultSample(timestamp_ns, 0, 0)))
    coordinator.consume(_read("body_joints", _body(timestamp_ns)))
    for side in ("left", "right"):
        coordinator.consume(
            _read(
                f"{side}_effector",
                EndEffectorSample(timestamp_ns, side, side, 0.025, 0, True),
            )
        )


def test_delayed_coordinator_does_not_turn_queued_vr_frame_into_disconnect() -> None:
    """Rollback history persistence may delay audit without disconnecting VR."""

    received_ns = 3_000_000_000
    clock = [received_ns + 300_000_000]
    application = _Application()
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        clock_ns=lambda: clock[0],
    )
    coordinator.consume(
        _read(
            "vr_network",
            VRNetworkSample(received_ns, True, 0.0, 1.0, 100.0, "g2", "pico"),
        )
    )
    coordinator.consume(_read("vr_data", _vr(received_ns, hand=False)))

    frame, frame_received_ns = application.frames[-1]
    assert frame.connected
    assert frame_received_ns == received_ns
    # This is only an event-time interpretation. The grant-capable health
    # snapshot still sees the 300 ms old evidence and remains fail-closed.
    assert not application.rollback_health[-1].network_fresh


def test_runtime_coordinator_detects_takeover_and_gated_handback() -> None:
    clock = [2_000_000_000]
    application = _Application()
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=True)
    coordinator.consume(
        _read(
            "action_src",
            ActionSourceSample(clock[0], "TAKEOVER", ("TAKEOVER",)),
        )
    )
    assert application.session.supervisor.mode == SupervisorMode.HUMAN
    assert application.takeovers[-1][1]["reason"] == "action_src_takeover"
    coordinator.consume(
        _read(
            "accepted_retarget",
            AcceptedRetargetSample(clock[0] + 1, (2, 4, 5), 0.2),
        )
    )
    assert not application.handbacks

    clock[0] += 300_000_000
    _feed_health(coordinator, clock[0], hand=False)
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    coordinator.tick()
    assert application.handback_prepares == 1
    assert not application.handbacks
    clock[0] += 10_000_000
    _feed_health(coordinator, clock[0], hand=False)
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    coordinator.tick()
    assert application.session.supervisor.mode == SupervisorMode.HUMAN
    clock[0] += 1
    coordinator.consume(_read("vr_data", _vr(clock[0], hand=False)))
    coordinator.tick()
    assert application.session.supervisor.mode == SupervisorMode.HOLD
    assert len(application.handbacks) == 1
    assert application.handbacks[0][1].side_buttons_released


def test_authority_without_side_button_uses_unlabelled_safe_handback() -> None:
    clock = [2_500_000_000]
    application = _Application(mode=SupervisorMode.ROLLBACK_EXEC)
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)

    coordinator.consume(
        _read("action_src", ActionSourceSample(clock[0], "TAKEOVER", ("TAKEOVER",)))
    )

    assert application.session.supervisor.mode == SupervisorMode.HUMAN
    assert application.takeovers == []
    assert len(application.external_preempts) == 1


def test_missing_camera_and_safety_integrations_block_handback() -> None:
    clock = [3_000_000_000]
    application = _Application(mode=SupervisorMode.HUMAN)
    coordinator = G2DDSRuntimeCoordinator(application=application, clock_ns=lambda: clock[0])
    _feed_health(coordinator, clock[0], hand=False)
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    assert not application.handbacks
    assert application.session.supervisor.mode == SupervisorMode.HUMAN


def test_human_handback_reanchors_stale_gdk_target_before_motion_health_recovers() -> None:
    clock = [3_200_000_000]
    application = _Application(mode=SupervisorMode.HUMAN)
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    clock[0] += 1
    coordinator.consume(
        _read(
            "motion_status",
            MotionStatusSample(clock[0], 8, 1, "target joint error over threshold"),
        )
    )
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    coordinator.tick()

    assert application.handback_prepares == 1
    assert application.handbacks == []
    assert application.session.supervisor.mode == SupervisorMode.HUMAN

    # Only a later healthy status may authorize the logical HUMAN -> HOLD.
    clock[0] += 20_000_000
    coordinator.consume(_read("motion_status", MotionStatusSample(clock[0], 0, 1, "")))
    coordinator.tick()
    assert application.handbacks == []
    clock[0] += 1
    coordinator.consume(_read("vr_data", _vr(clock[0], hand=False)))
    coordinator.tick()

    assert len(application.handbacks) == 1
    assert application.handbacks[0][1].motion_healthy
    assert application.session.supervisor.mode == SupervisorMode.HOLD


def test_human_input_after_provisional_reanchor_requires_a_new_reanchor() -> None:
    clock = [3_300_000_000]
    application = _Application(mode=SupervisorMode.HUMAN)
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    coordinator.tick()
    assert application.handback_prepares == 1

    clock[0] += 1
    coordinator.consume(
        _read("accepted_retarget", AcceptedRetargetSample(clock[0], (2,), 0.2))
    )

    assert application.human_restarts == 1
    assert not coordinator._human_hold_reanchored
    assert application.handbacks == []


def test_vr_press_queued_after_camera_capture_cannot_complete_handback() -> None:
    clock = [3_350_000_000]
    application = _Application(mode=SupervisorMode.HUMAN)
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    coordinator.tick()
    assert application.handback_prepares == 1

    # This cycle captures cameras but must not finish with the pre-capture VR frame.
    clock[0] += 1
    coordinator.consume(_read("motion_status", MotionStatusSample(clock[0], 0, 1, "")))
    coordinator.tick()
    assert application.handbacks == []

    # The first post-capture frame says Human control resumed, so the provisional
    # measured HOLD is revoked rather than being used to complete handback.
    clock[0] += 1
    coordinator.consume(_read("vr_data", _vr(clock[0], hand=True)))

    assert application.human_restarts == 1
    assert application.handbacks == []
    assert not coordinator._human_hold_reanchored


def test_human_reanchor_does_not_mask_unrelated_motion_fault() -> None:
    clock = [3_400_000_000]
    application = _Application(mode=SupervisorMode.HUMAN)
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    clock[0] += 1
    coordinator.consume(
        _read("motion_status", MotionStatusSample(clock[0], 7, 1, "motor hardware fault"))
    )
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))

    assert application.handback_prepares == 0
    assert application.handbacks == []
    assert application.session.supervisor.mode == SupervisorMode.HUMAN


def test_tick_rechecks_wall_clock_after_slow_vr_transition_before_stamping_health() -> None:
    clock = [3_500_000_000]
    application = _Application()

    def slow_advance(_now_ns: int):
        clock[0] += 300_000_000
        return []

    application.advance_vr_clock = slow_advance
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    assert application.rollback_health[-1].safe
    application.rollback_health.clear()

    coordinator.tick()

    assert len(application.rollback_health) == 1
    assert not application.rollback_health[0].safe
    assert not application.rollback_health[0].network_fresh


def test_slow_safety_callback_cannot_refresh_expired_rollback_evidence() -> None:
    clock = [3_800_000_000]
    monotonic_clock = [8_000_000_000]
    application = _Application()
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
        monotonic_clock_ns=lambda: monotonic_clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    application.rollback_health.clear()

    def slow_safety():
        clock[0] += 300_000_000
        monotonic_clock[0] += 300_000_000
        return {"emergency_stop_active": False, "collision_detected": False}

    coordinator.safety_state = slow_safety
    coordinator.tick()

    assert len(application.rollback_health) == 1
    assert not application.rollback_health[0].safe
    assert not application.rollback_health[0].network_fresh
    assert application.rollback_health[0].evaluated_monotonic_ns == monotonic_clock[0]


def test_slow_camera_callback_rechecks_handback_and_rollback_freshness() -> None:
    clock = [4_200_000_000]
    monotonic_clock = [9_000_000_000]
    application = _Application()
    coordinator = G2DDSRuntimeCoordinator(
        application=application,
        cameras_fresh=lambda: True,
        safety_state=lambda: {
            "emergency_stop_active": False,
            "collision_detected": False,
        },
        clock_ns=lambda: clock[0],
        monotonic_clock_ns=lambda: monotonic_clock[0],
    )
    _feed_health(coordinator, clock[0], hand=False)
    coordinator.consume(_read("action_src", ActionSourceSample(clock[0], "IDLE", ("IDLE",))))
    application.rollback_health.clear()
    application.session.supervisor.mode = SupervisorMode.HUMAN

    def slow_cameras() -> bool:
        clock[0] += 300_000_000
        monotonic_clock[0] += 300_000_000
        return True

    coordinator.cameras_fresh = slow_cameras
    # First pass installs the measured-pose HOLD barrier; camera and final
    # motion health are deliberately checked only on a later DDS cycle.
    coordinator.tick()
    assert application.handback_prepares == 1
    application.rollback_health.clear()
    coordinator.tick()

    assert application.handbacks == []
    assert application.session.supervisor.mode == SupervisorMode.HUMAN
    assert len(application.rollback_health) == 1
    assert not application.rollback_health[0].safe
    assert not application.rollback_health[0].measured_state_fresh
    assert application.rollback_health[0].evaluated_monotonic_ns == monotonic_clock[0]


def test_runtime_service_refuses_to_drop_evidence_on_queue_overflow() -> None:
    client = SimpleNamespace(specs={"sample": object()})
    service = G2DDSRuntimeService(
        client=client,
        coordinator=SimpleNamespace(),
        labels=("sample",),
        queue_size=1,
        tick_seconds=1e-6,
    )
    service._enqueue(_read("sample", 1))
    with pytest.raises(RuntimeError, match="refusing to drop evidence"):
        service._enqueue(_read("sample", 2))


def test_runtime_service_fast_paths_vr_release_before_queue_consumption() -> None:
    observed = []
    application = SimpleNamespace(observe_transport_vr_frame=observed.append)
    coordinator = SimpleNamespace(application=application)
    client = SimpleNamespace(specs={"vr_data": object()})
    service = G2DDSRuntimeService(
        client=client,
        coordinator=coordinator,
        labels=("vr_data",),
        queue_size=8,
        tick_seconds=0.02,
    )

    service._enqueue(_read("vr_data", _vr(10, hand=False)))

    assert len(observed) == 1
    assert observed[0].x is False
    # The audit/state-machine copy is still queued; the fast path does not
    # consume or discard it.
    assert service.messages.qsize() == 1


def test_runtime_service_fast_path_preserves_full_human_input_sample() -> None:
    observed = []
    application = SimpleNamespace(observe_transport_vr_sample=observed.append)
    coordinator = SimpleNamespace(application=application)
    client = SimpleNamespace(specs={"vr_data": object()})
    service = G2DDSRuntimeService(
        client=client,
        coordinator=coordinator,
        labels=("vr_data",),
        queue_size=8,
        tick_seconds=0.02,
    )
    sample = _vr(11, hand=True)

    service._enqueue(_read("vr_data", sample))

    assert observed == [sample]
    assert not observed[0].side_buttons_released
    assert service.messages.qsize() == 1


def test_runtime_service_drains_queued_vr_before_overdue_stale_tick() -> None:
    consumed = []
    coordinator = SimpleNamespace(consume=lambda read: consumed.append(read.label))
    client = SimpleNamespace(specs={"sample": object(), "vr_data": object()})
    service = G2DDSRuntimeService(
        client=client,
        coordinator=coordinator,
        labels=("sample", "vr_data"),
        queue_size=8,
        tick_seconds=0.02,
    )
    service._enqueue(_read("sample", 1))
    service._enqueue(_read("vr_data", 2))

    service._drain_urgent_before_tick()

    assert consumed == ["vr_data", "sample"]
    assert service.messages.empty()


def test_runtime_service_rechecks_for_vr_arriving_during_slow_ordinary_consume() -> None:
    ordinary_started = threading.Event()
    allow_ordinary = threading.Event()
    consumed = []

    class Coordinator:
        @staticmethod
        def consume(read) -> None:
            consumed.append(read.label)
            if read.label == "sample":
                ordinary_started.set()
                assert allow_ordinary.wait(1.0)

    client = SimpleNamespace(specs={"sample": object(), "vr_data": object()})
    service = G2DDSRuntimeService(
        client=client,
        coordinator=Coordinator(),
        labels=("sample", "vr_data"),
        queue_size=8,
        tick_seconds=0.02,
    )
    service._enqueue(_read("sample", 1))
    worker = threading.Thread(target=service._drain_urgent_before_tick)
    worker.start()
    assert ordinary_started.wait(1.0)

    service._enqueue(_read("vr_data", 2))
    allow_ordinary.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert consumed == ["sample", "vr_data"]
    assert service.messages.empty()


def test_runtime_service_prioritizes_control_then_measured_health_then_ordinary() -> None:
    consumed = []
    coordinator = SimpleNamespace(consume=lambda read: consumed.append(read.label))
    client = SimpleNamespace(specs={"sample": object(), "vr_data": object(), "body_joints": object()})
    service = G2DDSRuntimeService(
        client=client,
        coordinator=coordinator,
        labels=("sample", "vr_data", "body_joints"),
        queue_size=8,
        tick_seconds=0.02,
    )
    service._enqueue(_read("sample", 1))
    service._enqueue(_read("body_joints", 2))
    service._enqueue(_read("vr_data", 3))

    service._drain_urgent_before_tick()

    assert consumed == ["vr_data", "body_joints", "sample"]


def test_runtime_service_ticks_even_while_topic_is_continuously_busy() -> None:
    external_stop = threading.Event()

    class BusyClient:
        specs = {"sample": object()}

        @staticmethod
        def stream(_label, callback, *, stop) -> None:
            while not stop():
                callback(_read("sample", 1))
                time.sleep(0.0001)

    class Coordinator:
        consumed = 0
        ticks = 0

        def consume(self, _read_value) -> None:
            self.consumed += 1

        def tick(self) -> None:
            self.ticks += 1
            external_stop.set()

    coordinator = Coordinator()
    service = G2DDSRuntimeService(
        client=BusyClient(),
        coordinator=coordinator,
        labels=("sample",),
        queue_size=128,
        tick_seconds=0.002,
    )
    service.run(external_stop=external_stop)
    assert coordinator.consumed > 0
    assert coordinator.ticks == 1
