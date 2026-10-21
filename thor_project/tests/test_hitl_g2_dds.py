#!/usr/bin/env python

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest

from nexus_deploy.robot.g2a.constants import GDK_BODY_JOINTS
from nexus_deploy.robot.g2a.hitl.g2_runtime.g2_dds import (
    AcceptedRetargetSample,
    ActionSourceSample,
    BodyJointSample,
    DeviceInputSnapshot,
    EndEffectorSample,
    G2DDSRead,
    G2MeasuredStateAssembler,
    G2ReadOnlyDDSClient,
    G2RuntimeEvidence,
    G2TopicSpec,
    MotionStatusSample,
    PublicFaultSample,
    VRNetworkSample,
    VRTeleopSample,
    WholeBodySafetySample,
    decode_action_source,
    decode_body_joints,
    decode_end_effector,
    decode_operator_event,
    decode_public_fault,
    decode_vr_frame,
    decode_whole_body_safety,
)


def _stamp(timestamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000)


def _header(timestamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(stamp=_stamp(timestamp_ns))


def _device(
    device_type: int,
    *,
    axis_click: bool = False,
    index: float = 0.0,
    hand: float = 0.0,
    key_one: bool = False,
    key_two: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        device_type=device_type,
        axis_click=axis_click,
        axis_x=0.0,
        axis_y=0.0,
        index_trig=index,
        hand_trig=hand,
        key_one=key_one,
        key_two=key_two,
    )


def test_decode_vr_maps_x_y_and_both_side_buttons() -> None:
    raw = SimpleNamespace(
        timestamp=1_788_330_739_297,
        frame_index=155_220,
        device_inputs=[
            _device(0, key_one=True, key_two=True, hand=1.0, index=0.75),
            _device(1, hand=1.0, index=0.25),
            _device(2),
        ],
    )
    sample = decode_vr_frame(raw, connected=True)
    frame = sample.raw_frame()
    assert frame.source_timestamp_ns == 1_788_330_739_297_000_000
    assert frame.frame_index == 155_220
    assert frame.x and frame.y and frame.left_side and frame.right_side
    assert frame.left_index == pytest.approx(0.75)
    assert frame.right_index == pytest.approx(0.25)
    assert sample.side_buttons_pressed
    assert not sample.side_buttons_released

    raw.device_inputs = [_device(0), _device(1), _device(2)]
    assert decode_vr_frame(raw, connected=True).side_buttons_released


@pytest.mark.parametrize(
    "device",
    [
        _device(0, hand=1.0),
        _device(1, hand=1.0),
    ],
)
def test_human_release_gate_tracks_only_side_buttons(device: SimpleNamespace) -> None:
    devices = [_device(0), _device(1), _device(2)]
    devices[device.device_type] = device
    sample = decode_vr_frame(
        SimpleNamespace(timestamp=1_000, frame_index=1, device_inputs=devices),
        connected=True,
    )
    assert not sample.side_buttons_released


@pytest.mark.parametrize(
    "device",
    [
        _device(0, axis_click=True),
        _device(0, index=1.0),
        _device(1, index=1.0),
        _device(1, key_two=True),
    ],
)
def test_non_side_inputs_do_not_request_or_extend_human_control(device: SimpleNamespace) -> None:
    devices = [_device(0), _device(1), _device(2)]
    devices[device.device_type] = device
    sample = decode_vr_frame(
        SimpleNamespace(timestamp=1_000, frame_index=1, device_inputs=devices),
        connected=True,
    )
    assert sample.side_buttons_released
    assert not sample.side_buttons_pressed


def test_action_source_and_operator_event_use_payload_milliseconds() -> None:
    action = decode_action_source(SimpleNamespace(timestamp=1_234, status=1, status_history=[0, 1]))
    assert action.timestamp_ns == 1_234_000_000
    assert action.status == "TAKEOVER"
    assert action.status_history == ("IDLE", "TAKEOVER")

    event = decode_operator_event(
        SimpleNamespace(
            timestamp=1_235,
            operator_event=2,
            operator_event_detail=13,
            operator_device="UnityClient",
        )
    )
    assert event.timestamp_ns == 1_235_000_000
    assert event.detail == "takeover_start"
    assert event.dedup_key == (
        1_235_000_000,
        "takeover",
        "takeover_start",
        "UnityClient",
    )


def test_hardware_safety_decoders_preserve_estop_and_fault_evidence() -> None:
    whole_body = decode_whole_body_safety(
        SimpleNamespace(
            left_arm_estop=True,
            right_arm_estop=False,
            left_arm_error=0,
            right_arm_error=0,
            left_end_error=0,
            right_end_error=7,
            waist_error=0,
            neck_error=0,
            lift_error=0,
            chassis_error=0,
        )
    )
    assert whole_body.emergency_stop_active
    assert whole_body.hardware_fault_active
    assert whole_body.error_codes["right_end_error"] == 7

    public_fault = decode_public_fault(
        SimpleNamespace(
            header=_header(4_000_000_000),
            counter=9,
            fault_codes=[101, 202],
        )
    )
    assert public_fault.timestamp_ns == 4_000_000_000
    assert public_fault.counter == 9
    assert public_fault.fault_count == 2


def test_measured_state_assembler_separates_hal_health_from_canonical_pose() -> None:
    body_time = 10_000_000_000
    body_message = SimpleNamespace(
        header=_header(body_time),
        name=list(reversed(GDK_BODY_JOINTS)),
        position=[float(index) for index in reversed(range(len(GDK_BODY_JOINTS)))],
        velocity=[1.0 if name == GDK_BODY_JOINTS[0] else 0.0 for name in reversed(GDK_BODY_JOINTS)],
    )
    body = decode_body_joints(body_message)
    left = decode_end_effector(
        SimpleNamespace(
            header=_header(body_time + 5_000_000),
            name=["idx31_gripper_l_inner_joint1"],
            end_state=[SimpleNamespace(position=0.025, status=0)],
            controlled=True,
        ),
        side="left",
    )
    right = decode_end_effector(
        SimpleNamespace(
            header=_header(body_time + 10_000_000),
            name=["idx71_gripper_r_inner_joint1"],
            end_state=[SimpleNamespace(position=0.005, status=2)],
            controlled=True,
        ),
        side="right",
    )
    assembler = G2MeasuredStateAssembler()
    assembler.update_body(body)
    assembler.update_effector(left)
    assembler.update_effector(right)
    health = assembler.health(now_ns=body_time + 20_000_000)

    assert health.max_component_skew_ns == 10_000_000
    assert health.timestamp_ns == body_time + 10_000_000
    assert health.max_body_velocity == pytest.approx(math.pi / 3000)
    with pytest.raises(RuntimeError, match="encoder counts"):
        assembler.snapshot(now_ns=body_time + 20_000_000)

    assembler.update_body(
        BodyJointSample(
            body.timestamp_ns + 1,
            body.positions,
            body.velocities,
            position_domain="canonical_radians",
        )
    )
    measured = assembler.snapshot(now_ns=body_time + 20_000_000)

    assert measured.state_24d.shape == (24,)
    assert measured.state_24d[7] == pytest.approx(0.025)
    assert measured.state_24d[15] == pytest.approx(0.005)
    assert measured.max_component_skew_ns == 9_999_999
    assert measured.timestamp_ns == body_time + 10_000_000


def test_measured_state_assembler_rejects_stale_or_skewed_components() -> None:
    assembler = G2MeasuredStateAssembler()
    body_message = SimpleNamespace(
        header=_header(1_000_000_000),
        name=list(GDK_BODY_JOINTS),
        position=[0.0] * len(GDK_BODY_JOINTS),
        velocity=[0.0] * len(GDK_BODY_JOINTS),
    )
    assembler.update_body(decode_body_joints(body_message))
    for side, timestamp in (("left", 1_010_000_000), ("right", 1_100_000_000)):
        assembler.update_effector(
            decode_end_effector(
                SimpleNamespace(
                    header=_header(timestamp),
                    name=[side],
                    end_state=[SimpleNamespace(position=0.0, status=0)],
                    controlled=True,
                ),
                side=side,
            )
        )
    with pytest.raises(RuntimeError, match="component skew"):
        assembler.snapshot(now_ns=1_100_000_000, freshness_ns=200_000_000)
    with pytest.raises(RuntimeError, match="stale"):
        assembler.snapshot(
            now_ns=2_000_000_000,
            freshness_ns=100_000_000,
            max_component_skew_ns=200_000_000,
        )


def test_measured_freshness_uses_local_dds_receive_clock_not_future_vendor_clock() -> None:
    """G2 HAL payload clocks may lead the subscriber clock by about 20 ms."""

    receive_ns = 10_000_000_000
    vendor_ns = receive_ns + 21_000_000
    assembler = G2MeasuredStateAssembler()
    assembler.update_body(
        _runtime_body(vendor_ns),
        received_at_ns=receive_ns,
    )
    for side, offset_ns in (("left", 1_000_000), ("right", 2_000_000)):
        assembler.update_effector(
            EndEffectorSample(vendor_ns + offset_ns, side, side, 0.025, 0, True),
            received_at_ns=receive_ns + offset_ns,
        )

    health = assembler.health(now_ns=receive_ns + 5_000_000)
    assert health.timestamp_ns == vendor_ns + 2_000_000
    assert health.max_component_skew_ns == 2_000_000

    with pytest.raises(RuntimeError, match="stale"):
        assembler.health(
            now_ns=receive_ns + 200_000_000,
            freshness_ns=100_000_000,
        )


def _read(label: str, sample: object, *, publish_timestamp_ns: int = 1) -> G2DDSRead:
    receive_timestamp_ns = int(getattr(sample, "timestamp_ns", publish_timestamp_ns))
    return G2DDSRead(label, sample, publish_timestamp_ns, receive_timestamp_ns, 3, 4)


def _runtime_body(timestamp_ns: int, velocity: float = 0.0) -> BodyJointSample:
    return BodyJointSample(
        timestamp_ns=timestamp_ns,
        positions={name: float(index) for index, name in enumerate(GDK_BODY_JOINTS)},
        velocities=dict.fromkeys(GDK_BODY_JOINTS, velocity),
    )


def _runtime_vr(timestamp_ns: int, *, hand_pressed: bool) -> VRTeleopSample:
    devices = (
        DeviceInputSnapshot(0, False, 0.0, 0.0, 0.0, float(hand_pressed), False, False),
        DeviceInputSnapshot(1, False, 0.0, 0.0, 0.0, 0.0, False, False),
        DeviceInputSnapshot(2, False, 0.0, 0.0, 0.0, 0.0, False, False),
    )
    return VRTeleopSample(timestamp_ns, timestamp_ns, True, devices)


def test_runtime_evidence_accepts_vr_producer_frame_index_reset_when_timestamp_advances(
    caplog: pytest.LogCaptureFixture,
) -> None:
    evidence = G2RuntimeEvidence()
    devices = _runtime_vr(1_000_000_000, hand_pressed=False).devices
    first = VRTeleopSample(1_000_000_000, 100, True, devices)
    restarted = VRTeleopSample(1_010_000_000, 0, True, devices)

    assert evidence.observe(_read("vr_data", first))
    with caplog.at_level(logging.WARNING):
        assert evidence.observe(_read("vr_data", restarted))

    assert evidence.vr is restarted
    assert "frame_index reset accepted" in caplog.text


def _feed_runtime_state(
    evidence: G2RuntimeEvidence,
    *,
    timestamp_ns: int,
    action_src: str,
    hand_pressed: bool,
) -> None:
    evidence.observe(_read("vr_data", _runtime_vr(timestamp_ns, hand_pressed=hand_pressed)))
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(timestamp_ns, True, 0.0, 1.0, 0.0, "g2", "pico"),
        )
    )
    evidence.observe(_read("action_src", ActionSourceSample(timestamp_ns, action_src, (action_src,))))
    evidence.observe(_read("motion_status", MotionStatusSample(timestamp_ns, 0, 1, "")))
    evidence.observe(
        _read(
            "whole_body_status",
            WholeBodySafetySample(False, False, {"left_arm_error": 0}),
            publish_timestamp_ns=timestamp_ns,
        )
    )
    evidence.observe(_read("public_fault", PublicFaultSample(timestamp_ns, 0, 0)))
    evidence.observe(_read("body_joints", _runtime_body(timestamp_ns)))
    for side in ("left", "right"):
        evidence.observe(
            _read(
                f"{side}_effector",
                EndEffectorSample(timestamp_ns, side, side, 0.025, 0, True),
            )
        )


def test_runtime_evidence_requires_fresh_retarget_and_system_accept_for_human() -> None:
    evidence = G2RuntimeEvidence()
    now_ns = 2_000_000_000
    _feed_runtime_state(
        evidence,
        timestamp_ns=now_ns,
        action_src="TAKEOVER",
        hand_pressed=True,
    )
    assert evidence.ownership(now_ns=now_ns).owner == "hold"
    evidence.observe(
        _read(
            "accepted_retarget",
            AcceptedRetargetSample(now_ns + 1, (2, 4, 5), 0.2),
        )
    )
    assert evidence.ownership(now_ns=now_ns + 1).owner == "human"
    blocked = evidence.handback_evidence(now_ns=now_ns + 1, cameras_fresh=True)
    assert not blocked.side_buttons_released
    assert blocked.accepted_retarget_fresh


def test_runtime_handback_and_rollback_health_are_fail_closed() -> None:
    evidence = G2RuntimeEvidence()
    start_ns = 3_000_000_000
    _feed_runtime_state(
        evidence,
        timestamp_ns=start_ns,
        action_src="TAKEOVER",
        hand_pressed=True,
    )
    evidence.observe(_read("accepted_retarget", AcceptedRetargetSample(start_ns, (3, 4, 5), 0.2)))

    settled_ns = start_ns + 300_000_000
    _feed_runtime_state(
        evidence,
        timestamp_ns=settled_ns,
        action_src="IDLE",
        hand_pressed=False,
    )
    ready = evidence.handback_evidence(now_ns=settled_ns, cameras_fresh=True)
    assert ready.side_buttons_released
    assert not ready.accepted_retarget_fresh
    assert ready.action_src == "IDLE"
    assert ready.robot_settled
    assert ready.measured_state_fresh
    assert ready.network_fresh
    assert ready.motion_healthy

    healthy = evidence.rollback_health(now_ns=settled_ns)
    assert healthy.safe
    stale = evidence.rollback_health(now_ns=settled_ns + 2_000_000_000)
    assert not stale.safe

    evidence.observe(
        _read(
            "whole_body_status",
            WholeBodySafetySample(True, False, {"left_arm_error": 0}),
            publish_timestamp_ns=settled_ns + 1,
        )
    )
    estopped = evidence.handback_evidence(now_ns=settled_ns + 1, cameras_fresh=True)
    assert estopped.emergency_stop_active
    assert not evidence.rollback_health(now_ns=settled_ns + 1).safe


def test_low_rate_vr_network_heartbeat_remains_valid_across_three_second_cadence() -> None:
    evidence = G2RuntimeEvidence()
    start_ns = 5_000_000_000
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(start_ns, True, 0.0, 1.0, 0.33, "g2", "pico"),
        )
    )
    vr_ns = start_ns + 3_100_000_000
    evidence.observe(_read("vr_data", _runtime_vr(vr_ns, hand_pressed=False)))

    # The low-rate network status remains valid, while vr_data has its own
    # independent 200 ms freshness gate.
    assert evidence.network_fresh(now_ns=vr_ns)
    assert not evidence.network_fresh(now_ns=start_ns + 5_100_000_000)


def test_single_false_vr_network_sample_uses_bounded_disconnect_grace() -> None:
    evidence = G2RuntimeEvidence(network_disconnect_grace_ns=100_000_000)
    start_ns = 6_000_000_000
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(start_ns, True, 0.0, 1.0, 1.0, "g2", "pico"),
        )
    )
    false_ns = start_ns + 10_000_000
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(false_ns, False, 0.0, 1.0, 1.0, "g2", "pico"),
        )
    )
    evidence.observe(_read("vr_data", _runtime_vr(false_ns, hand_pressed=False)))

    assert evidence.network_fresh(now_ns=false_ns)
    assert evidence.network_fresh(now_ns=false_ns + 100_000_000)
    assert not evidence.network_fresh(now_ns=false_ns + 100_000_001)

    recovered_ns = false_ns + 20_000_000
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(recovered_ns, True, 0.0, 1.0, 1.0, "g2", "pico"),
        )
    )
    evidence.observe(_read("vr_data", _runtime_vr(recovered_ns, hand_pressed=False)))
    assert evidence.network_fresh(now_ns=recovered_ns)


def test_initial_false_vr_network_sample_never_receives_disconnect_grace() -> None:
    evidence = G2RuntimeEvidence(network_disconnect_grace_ns=100_000_000)
    now_ns = 7_000_000_000
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(now_ns, False, 0.0, 1.0, 1.0, "g2", "pico"),
        )
    )
    evidence.observe(_read("vr_data", _runtime_vr(now_ns, hand_pressed=False)))

    assert not evidence.network_fresh(now_ns=now_ns)


def test_bounded_future_network_receive_time_is_not_a_synthetic_disconnect() -> None:
    evidence = G2RuntimeEvidence(max_vr_future_skew_ns=100_000_000)
    vr_ns = 8_000_000_000
    network_ns = vr_ns + 10_000_000
    evidence.observe(
        _read(
            "vr_network",
            VRNetworkSample(network_ns, True, 0.0, 1.0, 1.0, "g2", "pico"),
        )
    )
    evidence.observe(_read("vr_data", _runtime_vr(vr_ns, hand_pressed=False)))

    assert evidence.network_fresh(now_ns=vr_ns)
    assert not evidence.network_fresh(now_ns=network_ns - 100_000_001)


class _FakeMessage:
    @classmethod
    def FromString(cls, payload: bytes) -> SimpleNamespace:  # noqa: N802 - protobuf API
        assert payload == b"payload"
        return SimpleNamespace(value=7)


class _FakeSubscriber:
    instances = 0

    def __init__(self, topic: str, type_name: str, qos: object) -> None:
        type(self).instances += 1
        assert (topic, type_name) == ("/read/only", "example.Message")
        assert qos == "qos"

    def subscribe(self, *, wait: bool) -> tuple[bytes, SimpleNamespace]:
        assert isinstance(wait, bool)
        return b"payload", SimpleNamespace(
            publish_ts_ns=11,
            recv_ts_ns=12,
            seq=13,
            source_ts_ns=0,
        )


def test_gdk_facade_has_only_read_once_and_ignores_zero_source_timestamp() -> None:
    _FakeSubscriber.instances = 0
    fake_gdk = SimpleNamespace(Subscriber=_FakeSubscriber)
    spec = G2TopicSpec(
        label="sample",
        topic="/read/only",
        type_name="example.Message",
        message_class=_FakeMessage,
        decoder=lambda message: message.value,
    )
    client = G2ReadOnlyDDSClient(gdk=fake_gdk, specs={"sample": spec}, qos="qos")
    result = client.read_once("sample")
    assert result.sample == 7
    assert result.publish_timestamp_ns == 11
    assert result.receive_timestamp_ns == 12
    assert result.sequence == 13
    assert result.payload_size == len(b"payload")
    assert not hasattr(client, "publish")


def test_gdk_stream_reuses_one_nonblocking_subscriber() -> None:
    _FakeSubscriber.instances = 0
    fake_gdk = SimpleNamespace(Subscriber=_FakeSubscriber)
    spec = G2TopicSpec(
        label="sample",
        topic="/read/only",
        type_name="example.Message",
        message_class=_FakeMessage,
        decoder=lambda message: message.value,
    )
    client = G2ReadOnlyDDSClient(
        gdk=fake_gdk,
        specs={"sample": spec},
        qos="qos",
        poll_seconds=1e-6,
    )
    reads = []
    client.stream("sample", reads.append, stop=lambda: len(reads) == 3)
    assert [read.sample for read in reads] == [7, 7, 7]
    assert _FakeSubscriber.instances == 1


@pytest.mark.parametrize(
    ("operator_event", "operator_event_detail", "operator_device"),
    (
        (0, 0, ""),  # Protobuf-default idle sample.
        (1, 0, "Unknow"),  # 90 Hz idle heartbeat observed on the audited G2.
    ),
)
def test_gdk_stream_drops_unstamped_operator_event(
    operator_event: int,
    operator_event_detail: int,
    operator_device: str,
) -> None:
    class Message:
        @classmethod
        def FromString(cls, _payload: bytes):  # noqa: N802 - protobuf API
            return SimpleNamespace(
                timestamp=0,
                operator_event=operator_event,
                operator_event_detail=operator_event_detail,
                operator_device=operator_device,
            )

    spec = G2TopicSpec(
        label="operator_event",
        topic="/operator/event",
        type_name="example.OperatorEvent",
        message_class=Message,
        decoder=lambda _message: pytest.fail("unstamped event must not reach the strict decoder"),
    )
    subscriber = SimpleNamespace(
        subscribe=lambda **_kwargs: (
            b"idle",
            SimpleNamespace(publish_ts_ns=1, recv_ts_ns=2, seq=3),
        )
    )

    assert G2ReadOnlyDDSClient._read_subscriber(spec, subscriber, wait=False) is None


def test_gdk_stream_rejects_unstamped_control_operator_event() -> None:
    class Message:
        @classmethod
        def FromString(cls, _payload: bytes):  # noqa: N802 - protobuf API
            return SimpleNamespace(
                timestamp=0,
                operator_event=2,
                operator_event_detail=13,
                operator_device="UnityClient",
            )

    spec = G2TopicSpec(
        label="operator_event",
        topic="/operator/event",
        type_name="example.OperatorEvent",
        message_class=Message,
        decoder=decode_operator_event,
    )
    subscriber = SimpleNamespace(
        subscribe=lambda **_kwargs: (
            b"unstamped-control",
            SimpleNamespace(publish_ts_ns=1, recv_ts_ns=2, seq=3),
        )
    )

    with pytest.raises(ValueError, match="OperatorEventRequest.timestamp must be positive"):
        G2ReadOnlyDDSClient._read_subscriber(spec, subscriber, wait=False)
