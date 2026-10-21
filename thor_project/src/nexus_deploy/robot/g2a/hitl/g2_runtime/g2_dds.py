"""Read-only adapters for the audited G2 DDS topics.

The module has two deliberately separate layers:

* pure protobuf-to-dataclass conversion that is testable off robot;
* a tiny ``agibot_gdk.Subscriber`` wrapper loaded only on G2.

There is no publisher API in this module.  Payload/header timestamps are used
instead of ``SubMsgInfo.source_ts_ns`` because the latter is zero on the
audited robot image.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...constants import (
    GDK_BODY_JOINTS,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
)
from .adapters import MixedUnitTimestampAdapter
from .rollback import RollbackHealth
from .supervisor import (
    HumanHandbackEvidence,
    OwnershipEvidence,
    RawVRFrame,
    SystemArbitrationMonitor,
)

logger = logging.getLogger(__name__)

ACTION_SOURCE_NAMES = {0: "IDLE", 1: "TAKEOVER", 2: "BACKWARD", 3: "MODEL"}
OPERATOR_EVENT_NAMES = {0: "unknown", 1: "autonomy", 2: "takeover", 3: "stopped"}
OPERATOR_DETAIL_NAMES = {
    0: "detail_unknown",
    13: "takeover_start",
    14: "takeover_stop",
    15: "takeover_reset",
    22: "takeover_backward",
    23: "takeover_forward",
    24: "takeover_pending",
    101: "takeover_soft_emergency_stop",
    102: "takeover_soft_emergency_recovery",
    106: "takeover_soft_emergency_stop_recovery",
}
# Retained/heartbeat samples audited on the vendor G2 image.  They have no
# source time and therefore cannot be admitted as ordered authority evidence.
_UNSTAMPED_OPERATOR_EVENT_SENTINELS = frozenset(
    {
        (0, 0, 0, ""),
        (0, 1, 0, "Unknow"),
    }
)
DEVICE_LEFT = 0
DEVICE_RIGHT = 1
DEVICE_HEAD = 2
# The audited G2 recorder converts integer HAL arm-velocity ticks to rad/s
# with this exact scale (e.g. one tick -> 0.0010471975512 rad/s).  Arm
# positions are a different domain: raw motor encoder counts, not radians.
ARM_DDS_VELOCITY_TO_RAD_S = math.pi / 3000.0


def _enum_int(value: Any, *, field: str) -> int:
    raw = getattr(value, "value", value)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer enum value")
    return raw


def _header_timestamp_ns(message: Any, *, field: str) -> int:
    try:
        stamp = message.header.stamp
    except AttributeError as error:
        raise ValueError(f"{field} is missing header.stamp") from error
    return MixedUnitTimestampAdapter.header_stamp(
        sec=stamp.sec,
        nanosec=stamp.nanosec,
        field=field,
    )


@dataclass(frozen=True)
class DeviceInputSnapshot:
    device_type: int
    axis_click: bool
    axis_x: float
    axis_y: float
    index_trig: float
    hand_trig: float
    key_one: bool
    key_two: bool

    @property
    def index_pressed(self) -> bool:
        return self.index_trig > 0.5

    @property
    def hand_pressed(self) -> bool:
        return self.hand_trig > 0.5


@dataclass(frozen=True)
class VRTeleopSample:
    timestamp_ns: int
    frame_index: int
    connected: bool
    devices: tuple[DeviceInputSnapshot, ...]

    def device(self, device_type: int) -> DeviceInputSnapshot | None:
        return next((item for item in self.devices if item.device_type == device_type), None)

    @property
    def side_buttons_pressed(self) -> bool:
        """Return whether either arm-control side button is held."""

        left = self.device(DEVICE_LEFT)
        right = self.device(DEVICE_RIGHT)
        if left is None or right is None:
            return False
        return left.hand_pressed or right.hand_pressed

    @property
    def side_buttons_released(self) -> bool:
        """Return true only when both arm-control side buttons are released."""

        left = self.device(DEVICE_LEFT)
        right = self.device(DEVICE_RIGHT)
        return bool(left is not None and right is not None and not self.side_buttons_pressed)

    def raw_frame(self) -> RawVRFrame:
        left = self.device(DEVICE_LEFT)
        right = self.device(DEVICE_RIGHT)
        if left is None or right is None:
            raise ValueError("VR frame is missing the left or right controller")
        return RawVRFrame(
            source_timestamp_ns=self.timestamp_ns,
            frame_index=self.frame_index,
            connected=self.connected,
            x=left.key_one,
            y=left.key_two,
            left_side=left.hand_pressed,
            right_side=right.hand_pressed,
            left_index=left.index_trig,
            right_index=right.index_trig,
        )


@dataclass(frozen=True)
class ActionSourceSample:
    timestamp_ns: int
    status: str
    status_history: tuple[str, ...]


@dataclass(frozen=True)
class OperatorEventSample:
    timestamp_ns: int
    event: str
    detail: str
    operator_device: str

    @property
    def dedup_key(self) -> tuple[int, str, str, str]:
        return self.timestamp_ns, self.event, self.detail, self.operator_device


@dataclass(frozen=True)
class AcceptedRetargetSample:
    timestamp_ns: int
    group_ids: tuple[int, ...]
    lifetime_s: float


@dataclass(frozen=True)
class MotionStatusSample:
    timestamp_ns: int
    error_code: int
    mode: int
    error_msg: str

    def healthy(self, *, accepted_modes: frozenset[int]) -> bool:
        return self.error_code == 0 and self.mode in accepted_modes


@dataclass(frozen=True)
class VRNetworkSample:
    timestamp_ns: int
    connected: bool
    packet_loss_rate: float
    latency_ms: float
    reported_frequency_hz: float
    ip_address: str
    remote_ip_address: str


@dataclass(frozen=True)
class WholeBodySafetySample:
    left_arm_estop: bool
    right_arm_estop: bool
    error_codes: Mapping[str, int]

    @property
    def emergency_stop_active(self) -> bool:
        return self.left_arm_estop or self.right_arm_estop

    @property
    def hardware_fault_active(self) -> bool:
        return any(code != 0 for code in self.error_codes.values())


@dataclass(frozen=True)
class PublicFaultSample:
    timestamp_ns: int
    counter: int
    fault_count: int


@dataclass(frozen=True)
class BodyJointSample:
    timestamp_ns: int
    positions: Mapping[str, float]
    velocities: Mapping[str, float]
    position_domain: str = "hal_encoder_counts_mixed"


@dataclass(frozen=True)
class EndEffectorSample:
    timestamp_ns: int
    side: str
    name: str
    position: float
    status: int
    controlled: bool


@dataclass(frozen=True)
class MeasuredState24:
    timestamp_ns: int
    state_24d: np.ndarray
    component_timestamps_ns: Mapping[str, int]
    max_component_skew_ns: int
    max_body_velocity: float

    def __post_init__(self) -> None:
        state = np.asarray(self.state_24d, dtype=np.float32)
        if state.shape != (24,) or not np.isfinite(state).all():
            raise ValueError(f"Measured G2 state must be finite [24], got {state.shape}")
        object.__setattr__(self, "state_24d", state.copy())


@dataclass(frozen=True)
class MeasuredStateHealth:
    timestamp_ns: int
    component_timestamps_ns: Mapping[str, int]
    max_component_skew_ns: int
    max_body_velocity: float


def decode_vr_frame(message: Any, *, connected: bool) -> VRTeleopSample:
    timestamp_ns = MixedUnitTimestampAdapter.payload_milliseconds(
        message.timestamp,
        field="RobotFrameData.timestamp",
    )
    frame_index = int(message.frame_index)
    if frame_index < 0:
        raise ValueError("RobotFrameData.frame_index must be non-negative")
    devices: list[DeviceInputSnapshot] = []
    seen: set[int] = set()
    for raw in message.device_inputs:
        device_type = _enum_int(raw.device_type, field="DeviceInputData.device_type")
        if device_type in seen:
            raise ValueError(f"RobotFrameData contains duplicate device_type={device_type}")
        seen.add(device_type)
        values = (float(raw.axis_x), float(raw.axis_y), float(raw.index_trig), float(raw.hand_trig))
        if not np.isfinite(values).all():
            raise ValueError("VR axes and triggers must be finite")
        devices.append(
            DeviceInputSnapshot(
                device_type=device_type,
                axis_click=bool(raw.axis_click),
                axis_x=values[0],
                axis_y=values[1],
                index_trig=values[2],
                hand_trig=values[3],
                key_one=bool(raw.key_one),
                key_two=bool(raw.key_two),
            )
        )
    sample = VRTeleopSample(timestamp_ns, frame_index, bool(connected), tuple(devices))
    sample.raw_frame()
    return sample


def decode_action_source(message: Any) -> ActionSourceSample:
    timestamp_ns = MixedUnitTimestampAdapter.payload_milliseconds(
        message.timestamp,
        field="ActionSrcState.timestamp",
    )
    status_value = _enum_int(message.status, field="ActionSrcState.status")
    if status_value not in ACTION_SOURCE_NAMES:
        raise ValueError(f"Unknown ActionSrcState.status={status_value}")
    history: list[str] = []
    for value in message.status_history:
        history_value = _enum_int(value, field="ActionSrcState.status_history")
        if history_value not in ACTION_SOURCE_NAMES:
            raise ValueError(f"Unknown ActionSrcState.status_history={history_value}")
        history.append(ACTION_SOURCE_NAMES[history_value])
    return ActionSourceSample(timestamp_ns, ACTION_SOURCE_NAMES[status_value], tuple(history))


def decode_operator_event(message: Any) -> OperatorEventSample:
    timestamp_ns = MixedUnitTimestampAdapter.payload_milliseconds(
        message.timestamp,
        field="OperatorEventRequest.timestamp",
    )
    event_value = _enum_int(message.operator_event, field="OperatorEventRequest.operator_event")
    detail_value = _enum_int(
        message.operator_event_detail,
        field="OperatorEventRequest.operator_event_detail",
    )
    return OperatorEventSample(
        timestamp_ns=timestamp_ns,
        event=OPERATOR_EVENT_NAMES.get(event_value, f"event_{event_value}"),
        detail=OPERATOR_DETAIL_NAMES.get(detail_value, f"detail_{detail_value}"),
        operator_device=str(message.operator_device),
    )


def decode_accepted_retarget(message: Any) -> AcceptedRetargetSample:
    timestamp_ns = _header_timestamp_ns(message, field="ReactiveControl.header.stamp")
    group_ids = tuple(sorted(int(group.group_id) for group in message.retarget_groups))
    if not group_ids:
        raise ValueError("Accepted ReactiveControl contains no retarget groups")
    lifetime_s = float(message.lifetime)
    if not np.isfinite(lifetime_s) or lifetime_s <= 0:
        raise ValueError("ReactiveControl.lifetime must be positive and finite")
    return AcceptedRetargetSample(timestamp_ns, group_ids, lifetime_s)


def decode_motion_status(message: Any) -> MotionStatusSample:
    return MotionStatusSample(
        timestamp_ns=_header_timestamp_ns(message, field="MotionControlStatus.header.stamp"),
        error_code=int(message.error_code),
        mode=int(message.mode),
        error_msg=str(message.error_msg),
    )


def decode_vr_network(message: Any) -> VRNetworkSample:
    return VRNetworkSample(
        timestamp_ns=_header_timestamp_ns(message, field="VRNetwork.header.stamp"),
        connected=bool(message.is_connected),
        packet_loss_rate=float(message.packet_loss_rate),
        latency_ms=float(message.latency),
        reported_frequency_hz=float(message.frequency),
        ip_address=str(message.ip_address),
        remote_ip_address=str(message.remote_ip_address),
    )


def decode_whole_body_safety(message: Any) -> WholeBodySafetySample:
    fields = (
        "left_arm_error",
        "right_arm_error",
        "left_end_error",
        "right_end_error",
        "waist_error",
        "neck_error",
        "lift_error",
        "chassis_error",
    )
    return WholeBodySafetySample(
        left_arm_estop=bool(message.left_arm_estop),
        right_arm_estop=bool(message.right_arm_estop),
        error_codes={field: int(getattr(message, field)) for field in fields},
    )


def decode_public_fault(message: Any) -> PublicFaultSample:
    return PublicFaultSample(
        timestamp_ns=_header_timestamp_ns(message, field="PublicFaultMsg.header.stamp"),
        counter=int(message.counter),
        fault_count=len(message.fault_codes),
    )


def decode_body_joints(message: Any) -> BodyJointSample:
    timestamp_ns = _header_timestamp_ns(message, field="JointState.header.stamp")
    names = tuple(str(name) for name in message.name)
    positions = tuple(float(value) for value in message.position)
    velocities = tuple(float(value) for value in message.velocity)
    if len(names) != len(positions) or len(names) != len(velocities):
        raise ValueError("JointState name/position/velocity lengths differ")
    if len(set(names)) != len(names):
        raise ValueError("JointState contains duplicate names")
    if not np.isfinite((*positions, *velocities)).all():
        raise ValueError("JointState positions and velocities must be finite")
    missing = sorted(set(GDK_BODY_JOINTS) - set(names))
    if missing:
        raise ValueError(f"JointState is missing canonical G2 joints: {missing}")
    velocity_by_name = dict(zip(names, velocities, strict=True))
    for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        velocity_by_name[name] *= ARM_DDS_VELOCITY_TO_RAD_S
    return BodyJointSample(
        timestamp_ns=timestamp_ns,
        positions=dict(zip(names, positions, strict=True)),
        velocities=velocity_by_name,
        position_domain="hal_encoder_counts_mixed",
    )


def decode_end_effector(message: Any, *, side: str) -> EndEffectorSample:
    if side not in {"left", "right"}:
        raise ValueError("End-effector side must be left or right")
    names = tuple(str(name) for name in message.name)
    states = tuple(message.end_state)
    if len(names) != 1 or len(states) != 1:
        raise ValueError(f"G2 {side} EndState must contain exactly one joint")
    position = float(states[0].position)
    if not np.isfinite(position):
        raise ValueError(f"G2 {side} end-effector position must be finite")
    return EndEffectorSample(
        timestamp_ns=_header_timestamp_ns(message, field=f"{side} EndState.header.stamp"),
        side=side,
        name=names[0],
        position=position,
        status=int(states[0].status),
        controlled=bool(message.controlled),
    )


class G2MeasuredStateAssembler:
    """Join timing/velocity health without guessing HAL encoder calibration."""

    def __init__(self) -> None:
        self.body: BodyJointSample | None = None
        self.effectors: dict[str, EndEffectorSample] = {}
        # Payload timestamps come from vendor publishers on different G2 SoCs.
        # They are the correct clock for component alignment and dataset
        # lineage, but they can lead this process' wall clock by tens of
        # milliseconds.  Freshness therefore uses the local DDS receive clock.
        self.received_at_ns: dict[str, int] = {}

    def update_body(
        self,
        sample: BodyJointSample,
        *,
        received_at_ns: int | None = None,
    ) -> None:
        if self.body is not None and sample.timestamp_ns <= self.body.timestamp_ns:
            return
        self.body = sample
        self.received_at_ns["body"] = int(sample.timestamp_ns if received_at_ns is None else received_at_ns)

    def update_effector(
        self,
        sample: EndEffectorSample,
        *,
        received_at_ns: int | None = None,
    ) -> None:
        previous = self.effectors.get(sample.side)
        if previous is not None and sample.timestamp_ns <= previous.timestamp_ns:
            return
        self.effectors[sample.side] = sample
        self.received_at_ns[f"{sample.side}_effector"] = int(
            sample.timestamp_ns if received_at_ns is None else received_at_ns
        )

    def health(
        self,
        *,
        now_ns: int,
        freshness_ns: int = 100_000_000,
        max_component_skew_ns: int = 40_000_000,
    ) -> MeasuredStateHealth:
        if now_ns <= 0 or freshness_ns <= 0 or max_component_skew_ns < 0:
            raise ValueError("Measured-state timing thresholds must be positive")
        if self.body is None or set(self.effectors) != {"left", "right"}:
            raise RuntimeError("Measured state is incomplete")
        timestamps = {
            "body": self.body.timestamp_ns,
            "left_effector": self.effectors["left"].timestamp_ns,
            "right_effector": self.effectors["right"].timestamp_ns,
        }
        freshness_timestamps = {
            "body": self.received_at_ns.get("body"),
            "left_effector": self.received_at_ns.get("left_effector"),
            "right_effector": self.received_at_ns.get("right_effector"),
        }
        if any(value is None for value in freshness_timestamps.values()):
            raise RuntimeError("Measured-state DDS receive timestamps are incomplete")
        if any(not 0 <= now_ns - int(value) <= freshness_ns for value in freshness_timestamps.values()):
            raise RuntimeError("Measured state is stale or from the future")
        skew_ns = max(timestamps.values()) - min(timestamps.values())
        if skew_ns > max_component_skew_ns:
            raise RuntimeError(
                f"Measured-state component skew {skew_ns} ns exceeds {max_component_skew_ns} ns"
            )
        max_velocity = max(abs(float(self.body.velocities[name])) for name in GDK_BODY_JOINTS)
        return MeasuredStateHealth(
            timestamp_ns=max(timestamps.values()),
            component_timestamps_ns=timestamps,
            max_component_skew_ns=skew_ns,
            max_body_velocity=max_velocity,
        )

    def snapshot(
        self,
        *,
        now_ns: int,
        freshness_ns: int = 100_000_000,
        max_component_skew_ns: int = 40_000_000,
    ) -> MeasuredState24:
        """Build state24 only from an explicitly canonical body sample.

        The live `/hal/joint_state` decoder deliberately marks its arm
        positions as encoder counts, so it cannot enter this path.  GDK
        `get_joint_states()` remains the authoritative canonical pose source.
        """

        health = self.health(
            now_ns=now_ns,
            freshness_ns=freshness_ns,
            max_component_skew_ns=max_component_skew_ns,
        )
        assert self.body is not None
        if self.body.position_domain != "canonical_radians":
            raise RuntimeError(
                "HAL arm positions are encoder counts; canonical 24D requires an explicit GDK conversion"
            )
        positions = self.body.positions
        state = np.asarray(
            [positions[name] for name in LEFT_ARM_JOINTS]
            + [self.effectors["left"].position]
            + [positions[name] for name in RIGHT_ARM_JOINTS]
            + [self.effectors["right"].position]
            + [positions[name] for name in HEAD_JOINTS]
            + [positions[name] for name in WAIST_JOINTS],
            dtype=np.float32,
        )
        return MeasuredState24(
            timestamp_ns=health.timestamp_ns,
            state_24d=state,
            component_timestamps_ns=health.component_timestamps_ns,
            max_component_skew_ns=health.max_component_skew_ns,
            max_body_velocity=health.max_body_velocity,
        )


class G2RuntimeEvidence:
    """Aggregate latest-only DDS samples into fail-closed runtime evidence."""

    def __init__(
        self,
        *,
        evidence_timeout_ns: int = 200_000_000,
        # /remote/vr_network is a low-rate status heartbeat on the production
        # G2 (about one sample every three seconds), unlike 100 Hz vr_data.
        # Its hold window must exceed that cadence. Live operator input loss is
        # still detected independently by the 200 ms VR frame timeout.
        network_timeout_ns: int = 5_000_000_000,
        # The vendor network topic occasionally publishes a single false
        # sample followed by true about 10 ms later.  Requiring a short,
        # continuous false interval prevents that status-bit chatter from
        # stopping AUTO while retaining a bounded fail-closed disconnect.
        network_disconnect_grace_ns: int = 100_000_000,
        measured_timeout_ns: int = 100_000_000,
        max_component_skew_ns: int = 40_000_000,
        max_vr_future_skew_ns: int = 100_000_000,
        settled_body_velocity: float = 0.02,
        accepted_motion_modes: frozenset[int] = frozenset({1}),
    ) -> None:
        if min(
            evidence_timeout_ns,
            network_timeout_ns,
            network_disconnect_grace_ns,
            measured_timeout_ns,
        ) <= 0:
            raise ValueError("Runtime evidence timeouts must be positive")
        if max_component_skew_ns < 0 or max_vr_future_skew_ns < 0:
            raise ValueError("Runtime skew tolerances must be non-negative")
        if settled_body_velocity <= 0 or not accepted_motion_modes:
            raise ValueError("Runtime settle/motion configuration is incomplete")
        self.evidence_timeout_ns = evidence_timeout_ns
        self.network_timeout_ns = network_timeout_ns
        self.network_disconnect_grace_ns = network_disconnect_grace_ns
        self.measured_timeout_ns = measured_timeout_ns
        self.max_component_skew_ns = max_component_skew_ns
        self.max_vr_future_skew_ns = max_vr_future_skew_ns
        self.settled_body_velocity = settled_body_velocity
        self.accepted_motion_modes = accepted_motion_modes
        self.arbitration = SystemArbitrationMonitor(evidence_timeout_ms=evidence_timeout_ns / 1_000_000)
        self.measured = G2MeasuredStateAssembler()
        self.vr: VRTeleopSample | None = None
        self.vr_revision = 0
        self.network: VRNetworkSample | None = None
        self._network_has_connected = False
        self._network_disconnected_since_ns: int | None = None
        self.action_source: ActionSourceSample | None = None
        self.motion: MotionStatusSample | None = None
        self.accepted_retarget: AcceptedRetargetSample | None = None
        self.operator_event: OperatorEventSample | None = None
        self.whole_body_safety: tuple[WholeBodySafetySample, int] | None = None
        self.public_fault: PublicFaultSample | None = None
        self.received_at_ns: dict[str, int] = {}
        self._last_vr_rejection_reason: str | None = None

    def observe(self, read: G2DDSRead) -> bool:
        """Update one topic and return false only for a duplicate/stale sample."""

        received_ns = int(read.receive_timestamp_ns)
        if received_ns <= 0:
            raise ValueError(f"DDS read {read.label!r} requires a positive receive timestamp")
        sample = read.sample
        if read.label == "vr_data" and isinstance(sample, VRTeleopSample):
            if self.vr is not None:
                if sample.timestamp_ns < self.vr.timestamp_ns:
                    self._note_vr_rejected("source_timestamp_regressed", sample)
                    return False
                if sample.timestamp_ns == self.vr.timestamp_ns and sample.frame_index <= self.vr.frame_index:
                    self._note_vr_rejected("duplicate_or_out_of_order_frame", sample)
                    return False
                if sample.timestamp_ns > self.vr.timestamp_ns and sample.frame_index < self.vr.frame_index:
                    # The epoch timestamp is authoritative across a restart of
                    # the VR producer's process-local frame counter.
                    logger.warning(
                        "VR frame_index reset accepted because source timestamp advanced: %d -> %d",
                        self.vr.frame_index,
                        sample.frame_index,
                    )
            if self._last_vr_rejection_reason is not None:
                logger.warning(
                    "G2 VR DDS evidence recovered after rejected frames: reason=%s",
                    self._last_vr_rejection_reason,
                )
                self._last_vr_rejection_reason = None
            self.vr = sample
            self.vr_revision += 1
        elif read.label == "vr_network" and isinstance(sample, VRNetworkSample):
            if self.network is not None and sample.timestamp_ns <= self.network.timestamp_ns:
                return False
            self.network = sample
            if sample.connected:
                self._network_has_connected = True
                self._network_disconnected_since_ns = None
            elif self._network_disconnected_since_ns is None:
                self._network_disconnected_since_ns = received_ns
        elif read.label == "operator_event" and isinstance(sample, OperatorEventSample):
            accepted = self.arbitration.observe_operator_event(
                dedup_key=sample.dedup_key,
                detail=sample.detail,
                vendor_timestamp_ns=sample.timestamp_ns,
            )
            if not accepted:
                return False
            self.operator_event = sample
        elif read.label == "action_src" and isinstance(sample, ActionSourceSample):
            if self.action_source is not None and sample.timestamp_ns <= self.action_source.timestamp_ns:
                return False
            self.action_source = sample
            self.arbitration.observe_action_src(sample.status)
        elif read.label == "accepted_retarget" and isinstance(sample, AcceptedRetargetSample):
            if (
                self.accepted_retarget is not None
                and sample.timestamp_ns <= self.accepted_retarget.timestamp_ns
            ):
                return False
            self.accepted_retarget = sample
            self.arbitration.observe_accepted_retarget(
                vendor_timestamp_ns=sample.timestamp_ns,
                freshness_timestamp_ns=received_ns,
            )
        elif read.label == "motion_status" and isinstance(sample, MotionStatusSample):
            if self.motion is not None and sample.timestamp_ns <= self.motion.timestamp_ns:
                return False
            self.motion = sample
        elif read.label == "whole_body_status" and isinstance(sample, WholeBodySafetySample):
            timestamp_ns = received_ns
            if self.whole_body_safety is not None and timestamp_ns <= self.whole_body_safety[1]:
                return False
            self.whole_body_safety = (sample, timestamp_ns)
        elif read.label == "public_fault" and isinstance(sample, PublicFaultSample):
            if self.public_fault is not None and sample.timestamp_ns <= self.public_fault.timestamp_ns:
                return False
            self.public_fault = sample
        elif read.label == "body_joints" and isinstance(sample, BodyJointSample):
            previous = self.measured.body
            self.measured.update_body(sample, received_at_ns=received_ns)
            if self.measured.body is previous:
                return False
            self.received_at_ns[read.label] = received_ns
            return True
        elif read.label in {"left_effector", "right_effector"} and isinstance(sample, EndEffectorSample):
            previous = self.measured.effectors.get(sample.side)
            self.measured.update_effector(sample, received_at_ns=received_ns)
            if self.measured.effectors.get(sample.side) is previous:
                return False
            self.received_at_ns[read.label] = received_ns
            return True
        else:
            raise ValueError(f"DDS read {read.label!r} has an incompatible decoded sample")
        self.received_at_ns[read.label] = received_ns
        return True

    def measured_state(self, *, now_ns: int) -> MeasuredState24:
        return self.measured.snapshot(
            now_ns=now_ns,
            freshness_ns=self.measured_timeout_ns,
            max_component_skew_ns=self.max_component_skew_ns,
        )

    def measured_health(self, *, now_ns: int) -> MeasuredStateHealth:
        return self.measured.health(
            now_ns=now_ns,
            freshness_ns=self.measured_timeout_ns,
            max_component_skew_ns=self.max_component_skew_ns,
        )

    def network_fresh(self, *, now_ns: int) -> bool:
        if self.vr is None or self.network is None:
            return False
        vr_received_ns = self.received_at_ns.get("vr_data")
        network_received_ns = self.received_at_ns.get("vr_network")
        if vr_received_ns is None or network_received_ns is None:
            return False
        vr_age_ns = now_ns - vr_received_ns
        network_age_ns = now_ns - network_received_ns
        network_connected = bool(self.network.connected)
        if not network_connected:
            disconnected_since_ns = self._network_disconnected_since_ns
            network_connected = bool(
                self._network_has_connected
                and disconnected_since_ns is not None
                and 0 <= now_ns - disconnected_since_ns <= self.network_disconnect_grace_ns
            )
        return (
            network_connected
            and -self.max_vr_future_skew_ns <= vr_age_ns <= self.evidence_timeout_ns
            # Independent DDS subscriber threads can enqueue a network sample
            # a few milliseconds before an older VR sample is serialized. A
            # bounded future skew is ordering ambiguity, not a disconnect.
            and -self.max_vr_future_skew_ns <= network_age_ns <= self.network_timeout_ns
        )

    def motion_healthy(self, *, now_ns: int) -> bool:
        motion_ok = bool(
            self.motion is not None
            and self._received_fresh("motion_status", now_ns, self.evidence_timeout_ns)
            and self.motion.healthy(accepted_modes=self.accepted_motion_modes)
        )
        platform = self.platform_safety(now_ns=now_ns)
        return motion_ok and not platform["emergency_stop_active"] and not platform["hardware_fault_active"]

    def platform_safety(self, *, now_ns: int) -> dict[str, bool]:
        whole_fresh = bool(
            self.whole_body_safety is not None
            and self._received_fresh("whole_body_status", now_ns, self.evidence_timeout_ns)
        )
        fault_fresh = bool(
            self.public_fault is not None
            and self._received_fresh("public_fault", now_ns, self.evidence_timeout_ns)
        )
        if not whole_fresh or not fault_fresh:
            return {"emergency_stop_active": True, "hardware_fault_active": True}
        assert self.whole_body_safety is not None and self.public_fault is not None
        whole = self.whole_body_safety[0]
        return {
            "emergency_stop_active": whole.emergency_stop_active,
            "hardware_fault_active": whole.hardware_fault_active or self.public_fault.fault_count > 0,
        }

    def action_src(self, *, now_ns: int) -> str:
        if self.action_source is None or not self._received_fresh(
            "action_src", now_ns, self.evidence_timeout_ns
        ):
            return "UNKNOWN"
        return self.action_source.status

    def ownership(self, *, now_ns: int) -> OwnershipEvidence:
        action_src = self.action_src(now_ns=now_ns)
        motion_healthy = self.motion_healthy(now_ns=now_ns)
        if action_src == "UNKNOWN":
            self.arbitration.invalidate_local_leases()
            return OwnershipEvidence("hold", "action_source_stale", None)
        self.arbitration.observe_action_src(action_src)
        self.arbitration.set_motion_health(motion_healthy)
        return self.arbitration.determine_owner(now_ns=now_ns)

    def handback_evidence(
        self,
        *,
        now_ns: int,
        cameras_fresh: bool,
        emergency_stop_active: bool = False,
        collision_detected: bool = False,
    ) -> HumanHandbackEvidence:
        measured: MeasuredStateHealth | None
        try:
            measured = self.measured_health(now_ns=now_ns)
        except RuntimeError:
            measured = None
        accepted_retarget_fresh = self._received_fresh("accepted_retarget", now_ns, self.evidence_timeout_ns)
        vr_fresh = self.network_fresh(now_ns=now_ns)
        platform_safety = self.platform_safety(now_ns=now_ns)
        return HumanHandbackEvidence(
            side_buttons_released=bool(vr_fresh and self.vr and self.vr.side_buttons_released),
            accepted_retarget_fresh=accepted_retarget_fresh,
            action_src=self.action_src(now_ns=now_ns),
            robot_settled=bool(
                measured is not None and measured.max_body_velocity <= self.settled_body_velocity
            ),
            measured_state_fresh=measured is not None,
            cameras_fresh=bool(cameras_fresh),
            network_fresh=vr_fresh,
            motion_healthy=self.motion_healthy(now_ns=now_ns),
            emergency_stop_active=bool(emergency_stop_active or platform_safety["emergency_stop_active"]),
            collision_detected=bool(collision_detected),
        )

    def human_hold_reanchor_safe(
        self,
        *,
        now_ns: int,
        emergency_stop_active: bool = False,
        collision_detected: bool = False,
    ) -> bool:
        """Allow a current-pose HOLD only for healthy or known stale-target status.

        Human correction can leave MotionControlStatus error 8 because the
        lower-priority GDK channel still names its pre-Human target.  That one
        condition is exactly what the measured-pose re-anchor repairs.  Other
        motion errors, bad modes, stale status, and platform faults must remain
        fail-closed rather than being hidden by the handback workaround.
        """

        platform = self.platform_safety(now_ns=now_ns)
        if (
            emergency_stop_active
            or collision_detected
            or platform["emergency_stop_active"]
            or platform["hardware_fault_active"]
            or self.motion is None
            or not self._received_fresh("motion_status", now_ns, self.evidence_timeout_ns)
            or self.motion.mode not in self.accepted_motion_modes
        ):
            return False
        if self.motion.error_code == 0:
            return True
        message = self.motion.error_msg.lower()
        return self.motion.error_code == 8 and "target joint error over threshold" in message

    def rollback_health(
        self,
        *,
        now_ns: int,
        emergency_stop_active: bool = False,
        collision_detected: bool = False,
    ) -> RollbackHealth:
        try:
            self.measured_health(now_ns=now_ns)
            measured_fresh = True
        except RuntimeError:
            measured_fresh = False
        action_src = self.action_src(now_ns=now_ns)
        platform_safety = self.platform_safety(now_ns=now_ns)
        retarget_fresh = self._received_fresh("accepted_retarget", now_ns, self.evidence_timeout_ns)
        return RollbackHealth(
            motion_healthy=self.motion_healthy(now_ns=now_ns),
            measured_state_fresh=measured_fresh,
            network_fresh=self.network_fresh(now_ns=now_ns),
            emergency_stop_active=bool(emergency_stop_active or platform_safety["emergency_stop_active"]),
            collision_detected=bool(collision_detected),
            human_takeover_active=action_src in {"TAKEOVER", "BACKWARD"} or retarget_fresh,
        )

    def accepted_retarget_fresh(self, *, now_ns: int) -> bool:
        """Return freshness in the G2 receive-clock domain."""

        return self._received_fresh("accepted_retarget", now_ns, self.evidence_timeout_ns)

    def _received_fresh(self, label: str, now_ns: int, timeout_ns: int) -> bool:
        timestamp_ns = self.received_at_ns.get(label)
        return timestamp_ns is not None and 0 <= now_ns - timestamp_ns <= timeout_ns

    def _note_vr_rejected(self, reason: str, sample: VRTeleopSample) -> None:
        if reason == self._last_vr_rejection_reason:
            return
        self._last_vr_rejection_reason = reason
        assert self.vr is not None
        logger.warning(
            "G2 VR DDS FRAME REJECTED: reason=%s timestamp_ns=%d frame_index=%d "
            "last_timestamp_ns=%d last_frame_index=%d",
            reason,
            sample.timestamp_ns,
            sample.frame_index,
            self.vr.timestamp_ns,
            self.vr.frame_index,
        )


@dataclass(frozen=True)
class G2TopicSpec:
    label: str
    topic: str
    type_name: str
    message_class: type[Any]
    decoder: Callable[[Any], Any]


@dataclass(frozen=True)
class G2DDSRead:
    label: str
    sample: Any
    publish_timestamp_ns: int
    receive_timestamp_ns: int
    sequence: int
    payload_size: int


def load_g2_topic_specs() -> dict[str, G2TopicSpec]:
    """Load vendor protobufs lazily; this function is called only on G2."""

    from fsm_msgs_pb.action_src_pb2 import ActionSrcState
    from genie_msgs_pb.msg.EndState_pb2 import EndState
    from genie_msgs_pb.msg.JointState_pb2 import JointState
    from genie_msgs_pb.msg.MotionControlStatus_pb2 import MotionControlStatus
    from genie_msgs_pb.msg.PublicFaultMsg_pb2 import PublicFaultMsg
    from genie_msgs_pb.msg.ReactiveControl_pb2 import ReactiveControl
    from genie_msgs_pb.msg.VRNetwork_pb2 import VRNetwork
    from genie_msgs_pb.msg.WholeBodyStatus_pb2 import WholeBodyStatus
    from rh_msgs_v4_pb.operator_event_pb2 import OperatorEventRequest
    from rh_msgs_v4_pb.robot_frame_data_pb2 import RobotFrameData

    specs = (
        G2TopicSpec(
            "vr_data",
            "/remote/vr_data",
            "rh_msgs_v4.RobotFrameData",
            RobotFrameData,
            lambda message: decode_vr_frame(message, connected=True),
        ),
        G2TopicSpec(
            "vr_network",
            "/remote/vr_network",
            "genie_msgs.msg.pb.VRNetwork",
            VRNetwork,
            decode_vr_network,
        ),
        G2TopicSpec(
            "operator_event",
            "/remote/operator_event",
            "rh_msgs_v4.OperatorEventRequest",
            OperatorEventRequest,
            decode_operator_event,
        ),
        G2TopicSpec(
            "action_src",
            "/arbitrator/action_src",
            "action_src_msgs.ActionSrcState",
            ActionSrcState,
            decode_action_source,
        ),
        G2TopicSpec(
            "accepted_retarget",
            "/wbc/retarget",
            "genie_msgs.msg.pb.ReactiveControl",
            ReactiveControl,
            decode_accepted_retarget,
        ),
        G2TopicSpec(
            "motion_status",
            "/wbc/motion_control_status",
            "genie_msgs.msg.pb.MotionControlStatus",
            MotionControlStatus,
            decode_motion_status,
        ),
        G2TopicSpec(
            "whole_body_status",
            "/hal/whole_body_status",
            "genie_msgs.msg.pb.WholeBodyStatus",
            WholeBodyStatus,
            decode_whole_body_safety,
        ),
        G2TopicSpec(
            "public_fault",
            "/monitor/g02_fault_topic",
            "genie_msgs.msg.pb.PublicFaultMsg",
            PublicFaultMsg,
            decode_public_fault,
        ),
        G2TopicSpec(
            "body_joints",
            "/hal/joint_state",
            "genie_msgs.msg.pb.JointState",
            JointState,
            decode_body_joints,
        ),
        G2TopicSpec(
            "left_effector",
            "/hal/left_ee_data",
            "genie_msgs.msg.pb.EndState",
            EndState,
            lambda message: decode_end_effector(message, side="left"),
        ),
        G2TopicSpec(
            "right_effector",
            "/hal/right_ee_data",
            "genie_msgs.msg.pb.EndState",
            EndState,
            lambda message: decode_end_effector(message, side="right"),
        ),
    )
    return {spec.label: spec for spec in specs}


class G2ReadOnlyDDSClient:
    """Synchronous read-only GDK subscriber facade used by runtime threads."""

    def __init__(
        self,
        *,
        gdk: Any | None = None,
        specs: Mapping[str, G2TopicSpec] | None = None,
        qos: Any | None = None,
        poll_seconds: float = 0.002,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("G2 DDS poll_seconds must be positive")
        if gdk is None:
            import agibot_gdk as gdk_module

            gdk = gdk_module
        self.gdk = gdk
        self.specs = dict(specs or load_g2_topic_specs())
        self.qos = qos if qos is not None else gdk.QoS()
        self.poll_seconds = poll_seconds

    def read_once(self, label: str) -> G2DDSRead:
        spec, subscriber = self._open_subscriber(label)
        result = self._read_subscriber(spec, subscriber, wait=True)
        if result is None:
            raise RuntimeError(f"Blocking G2 DDS read for {label} returned no payload")
        return result

    def _open_subscriber(self, label: str) -> tuple[G2TopicSpec, Any]:
        try:
            spec = self.specs[label]
        except KeyError as error:
            raise ValueError(f"Unknown G2 DDS topic label {label!r}") from error
        return spec, self.gdk.Subscriber(spec.topic, spec.type_name, self.qos)

    @staticmethod
    def _read_subscriber(
        spec: G2TopicSpec,
        subscriber: Any,
        *,
        wait: bool,
    ) -> G2DDSRead | None:
        payload, info = subscriber.subscribe(wait=wait)
        if payload is None and not wait:
            return None
        if not isinstance(payload, bytes):
            raise RuntimeError(f"G2 DDS {spec.label} returned a non-bytes payload")
        message = spec.message_class.FromString(payload)
        # The vendor publishes retained/startup OperatorEventRequest samples
        # without a source timestamp. Drop only the exact audited idle
        # sentinels. Other unstamped values still reach the strict decoder and
        # fail closed, especially if they carry takeover-like semantics.
        operator_event_key = (
            getattr(message, "timestamp", None),
            getattr(message, "operator_event", None),
            getattr(message, "operator_event_detail", None),
            getattr(message, "operator_device", None),
        )
        if spec.label == "operator_event" and operator_event_key in _UNSTAMPED_OPERATOR_EVENT_SENTINELS:
            return None
        return G2DDSRead(
            label=spec.label,
            sample=spec.decoder(message),
            publish_timestamp_ns=int(info.publish_ts_ns),
            receive_timestamp_ns=int(info.recv_ts_ns),
            sequence=int(info.seq),
            payload_size=len(payload),
        )

    def stream(self, label: str, callback: Callable[[G2DDSRead], None], *, stop: Callable[[], bool]) -> None:
        """Read until ``stop`` becomes true; intended for one thread per topic."""

        spec, subscriber = self._open_subscriber(label)
        while not stop():
            read = self._read_subscriber(spec, subscriber, wait=False)
            if read is None:
                time.sleep(self.poll_seconds)
                continue
            callback(read)


def require_topic_labels(specs: Mapping[str, G2TopicSpec], labels: Sequence[str]) -> None:
    missing = sorted(set(labels) - specs.keys())
    if missing:
        raise ValueError(f"G2 DDS topic specs are missing labels: {missing}")
