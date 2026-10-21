"""Robot-independent FlowPRO input parsing, ownership state, and episode lifecycle.

The classes in this module never import or call GDK/DDS.  A future G2 adapter
must translate audited vendor messages into ``RawVRFrame``/ownership evidence
without changing the state-machine semantics.  Tablet recorder start/save
remain external operator actions observed through the filesystem lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..common.compat import StrEnum
from ..common.contract import EVENT_SCHEMA_VERSION
from ..common.sidecar import (
    CAMERA_SOURCES,
    SIDECAR_OPEN_SUFFIX,
    VENDOR_META_ACTIVE,
    VENDOR_META_FINAL,
    VENDOR_META_PATH,
    VENDOR_REQUIRED_PATHS,
    FlowPROSidecarWriter,
    classify_vendor_meta,
    read_vendor_meta,
    seal_sidecar,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class RawVRFrame:
    source_timestamp_ns: int
    frame_index: int
    connected: bool
    x: bool = False
    y: bool = False
    left_side: bool = False
    right_side: bool = False
    left_index: float = 0.0
    right_index: float = 0.0


@dataclass(frozen=True)
class LogicalInput:
    name: str
    source_timestamp_ns: int
    detail: dict[str, Any]


@dataclass
class _ButtonState:
    down: bool = False


class VRButtonParser:
    """Parse X/Y and the two Human side-button levels.

    X has only press/release rollback semantics.  Y is a single-button Policy
    toggle: its rising edge starts a logical generation from DISABLED/HOLD or
    stops AUTO immediately.  Physical Policy start is still deferred until a
    later all-released frame.  A rising side-button level requests Human
    takeover; physical ownership still comes from vendor arbitration.
    """

    def __init__(
        self,
        *,
        stale_timeout_ms: float = 200.0,
        index_press_threshold: float = 0.6,
        index_release_threshold: float = 0.2,
        require_initial_neutral: bool = False,
    ) -> None:
        if stale_timeout_ms <= 0:
            raise ValueError("button parser stale timeout must be positive")
        if not 0 <= index_release_threshold < index_press_threshold <= 1:
            raise ValueError("index trigger thresholds must satisfy 0 <= release < press <= 1")
        self.stale_timeout_ns = int(stale_timeout_ms * 1_000_000)
        self.index_press_threshold = float(index_press_threshold)
        self.index_release_threshold = float(index_release_threshold)
        self._buttons = {"x": _ButtonState(), "y": _ButtonState()}
        self._index_triggers = {"left": _ButtonState(), "right": _ButtonState()}
        self._last_timestamp_ns: int | None = None
        self._last_frame_index: int | None = None
        self._last_connected_ns: int | None = None
        self._last_received_ns: int | None = None
        self._stale_emitted = False
        # A stale/disconnected stream loses trustworthy edge state.  Ignore
        # held X/Y levels after reconnection until one explicit all-released
        # frame is observed, otherwise the same physical hold can be parsed as
        # a second press.
        self._neutral_rearm_required = bool(require_initial_neutral)
        self._neutral_rearm_notice_emitted = False
        self._last_frame_rejection_reason: str | None = None
        self._last_side_buttons = (False, False)
        # Local sequence of frames that passed the ordering/shape checks.  It
        # deliberately does not advance for a duplicate or regressed frame:
        # rollback uses it as a fresh-input barrier after durable I/O.
        self._accepted_frame_revision = 0

    @property
    def x_pressed(self) -> bool:
        return not self._neutral_rearm_required and self._buttons["x"].down

    @property
    def y_pressed(self) -> bool:
        return not self._neutral_rearm_required and self._buttons["y"].down

    @property
    def accepted_frame_revision(self) -> int:
        return self._accepted_frame_revision

    @property
    def side_takeover_pressed(self) -> bool:
        return any(self._last_side_buttons)

    @property
    def index_triggers_pressed(self) -> tuple[bool, bool]:
        """Debounced left/right upper-trigger levels used for gripper ownership."""

        if self._neutral_rearm_required:
            return (False, False)
        return (self._index_triggers["left"].down, self._index_triggers["right"].down)

    def advance_gesture_context(self) -> None:
        """Compatibility no-op: the parser no longer buffers click pairs."""

    def feed(
        self,
        frame: RawVRFrame,
        *,
        received_timestamp_ns: int | None = None,
    ) -> list[LogicalInput]:
        if frame.source_timestamp_ns <= 0 or frame.frame_index < 0:
            raise ValueError("VR timestamp must be positive and frame_index non-negative")
        if not all(
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
            for value in (frame.left_index, frame.right_index)
        ):
            raise ValueError("VR index trigger values must be finite values in [0, 1]")
        received_ns = frame.source_timestamp_ns if received_timestamp_ns is None else received_timestamp_ns
        if received_ns <= 0:
            raise ValueError("VR receive timestamp must be positive")
        if self._last_timestamp_ns is not None:
            if frame.source_timestamp_ns < self._last_timestamp_ns:
                self._note_frame_rejected("source_timestamp_regressed", frame)
                return []
            if (
                frame.source_timestamp_ns == self._last_timestamp_ns
                and self._last_frame_index is not None
                and frame.frame_index <= self._last_frame_index
            ):
                self._note_frame_rejected("duplicate_or_out_of_order_frame", frame)
                return []
            if (
                frame.source_timestamp_ns > self._last_timestamp_ns
                and self._last_frame_index is not None
                and frame.frame_index < self._last_frame_index
            ):
                # RobotFrameData.timestamp is the primary ordering key. The
                # teleop publisher may restart its process-local frame counter
                # without resetting the epoch timestamp; accepting that newer
                # frame keeps X/Y edges live across a producer restart.
                logger.warning(
                    "VR frame_index reset accepted because source timestamp advanced: %d -> %d",
                    self._last_frame_index,
                    frame.frame_index,
                )
        if self._last_frame_rejection_reason is not None:
            logger.warning(
                "VR input stream recovered after rejected frames: reason=%s",
                self._last_frame_rejection_reason,
            )
            self._last_frame_rejection_reason = None
        self._last_frame_index = frame.frame_index
        self._last_timestamp_ns = frame.source_timestamp_ns
        self._accepted_frame_revision += 1
        side_buttons = (frame.left_side, frame.right_side)
        events = self.advance(frame.source_timestamp_ns, received_timestamp_ns=received_ns)
        if not frame.connected:
            if not self._stale_emitted:
                events.append(self._event("vr_stale", frame.source_timestamp_ns, reason="disconnected"))
                logger.error(
                    "FLOWPRO VR INPUT BLOCKED: stream disconnected; release X/Y after reconnect to rearm"
                )
                self._stale_emitted = True
            self._release_all(frame.source_timestamp_ns)
            self._neutral_rearm_required = True
            self._neutral_rearm_notice_emitted = False
            return self._translate_to_receive_clock(
                events,
                source_now_ns=frame.source_timestamp_ns,
                received_now_ns=received_ns,
            )

        self._last_connected_ns = frame.source_timestamp_ns
        self._last_received_ns = received_ns
        self._stale_emitted = False
        self._update_index_trigger("left", float(frame.left_index))
        self._update_index_trigger("right", float(frame.right_index))
        if side_buttons != self._last_side_buttons and any(side_buttons):
            events.append(
                self._event(
                    "side_takeover_input",
                    frame.source_timestamp_ns,
                    left_side=frame.left_side,
                    right_side=frame.right_side,
                )
            )
        self._last_side_buttons = side_buttons

        if self._neutral_rearm_required:
            if not frame.x and not frame.y and not any(side_buttons):
                self._neutral_rearm_required = False
                self._neutral_rearm_notice_emitted = False
                logger.warning("FLOWPRO VR INPUT REARMED: neutral X/Y frame observed")
                events.append(self._event("vr_input_rearmed", frame.source_timestamp_ns))
            elif not self._neutral_rearm_notice_emitted:
                logger.warning("FLOWPRO VR INPUT BLOCKED: waiting for one frame with both X and Y released")
                self._neutral_rearm_notice_emitted = True
                events.append(
                    self._event(
                        "vr_neutral_rearm_required",
                        frame.source_timestamp_ns,
                        reason="release_x_and_y_after_vr_recovery",
                    )
                )
            return self._translate_to_receive_clock(
                events,
                source_now_ns=frame.source_timestamp_ns,
                received_now_ns=received_ns,
            )

        previous = {name: state.down for name, state in self._buttons.items()}
        current = {"x": frame.x, "y": frame.y}
        for name in ("x", "y"):
            if current[name] and not previous[name]:
                state = self._buttons[name]
                state.down = True
                if name == "x":
                    events.append(
                        self._event(
                            "x_press",
                            frame.source_timestamp_ns,
                            vr_frame_index=frame.frame_index,
                        )
                    )
                else:
                    events.append(
                        self._event(
                            "y_press",
                            frame.source_timestamp_ns,
                            vr_frame_index=frame.frame_index,
                        )
                    )

        events.extend(self.advance(frame.source_timestamp_ns, received_timestamp_ns=received_ns))
        for name in ("x", "y"):
            if not current[name] and previous[name]:
                events.extend(self._release_button(name, frame.source_timestamp_ns))
        return self._translate_to_receive_clock(
            events,
            source_now_ns=frame.source_timestamp_ns,
            received_now_ns=received_ns,
        )

    def advance(
        self,
        now_ns: int,
        *,
        received_timestamp_ns: int | None = None,
    ) -> list[LogicalInput]:
        events: list[LogicalInput] = []
        freshness_now_ns = now_ns if received_timestamp_ns is None else received_timestamp_ns
        freshness_anchor_ns = (
            self._last_connected_ns if received_timestamp_ns is None else self._last_received_ns
        )
        if freshness_anchor_ns is not None and freshness_now_ns - freshness_anchor_ns > self.stale_timeout_ns:
            if not self._stale_emitted:
                events.append(self._event("vr_stale", now_ns, reason="frame_timeout"))
                logger.error(
                    "FLOWPRO VR INPUT BLOCKED: no fresh frame for %.0f ms; release X/Y after recovery",
                    self.stale_timeout_ns / 1e6,
                )
                self._stale_emitted = True
            self._release_all(now_ns)
            self._neutral_rearm_required = True
        return events

    def advance_received_clock(self, received_now_ns: int) -> list[LogicalInput]:
        """Advance timers using local receive time without mixing clock domains."""

        if received_now_ns <= 0:
            raise ValueError("VR receive clock must be positive")
        if self._last_timestamp_ns is None or self._last_received_ns is None:
            return []
        elapsed_ns = max(0, received_now_ns - self._last_received_ns)
        source_now_ns = self._last_timestamp_ns + elapsed_ns
        events = self.advance(source_now_ns, received_timestamp_ns=received_now_ns)
        return self._translate_to_receive_clock(
            events,
            source_now_ns=source_now_ns,
            received_now_ns=received_now_ns,
        )

    def _release_button(self, name: str, timestamp_ns: int) -> list[LogicalInput]:
        state = self._buttons[name]
        state.down = False
        return [self._event("x_release", timestamp_ns)] if name == "x" else []

    def _release_all(self, timestamp_ns: int) -> None:
        for state in self._buttons.values():
            state.down = False
        for state in self._index_triggers.values():
            state.down = False
        self._last_side_buttons = (False, False)

    def _update_index_trigger(self, side: str, value: float) -> None:
        state = self._index_triggers[side]
        if state.down:
            if value <= self.index_release_threshold:
                state.down = False
        elif value >= self.index_press_threshold:
            state.down = True

    def _note_frame_rejected(self, reason: str, frame: RawVRFrame) -> None:
        if reason == self._last_frame_rejection_reason:
            return
        self._last_frame_rejection_reason = reason
        logger.warning(
            "FLOWPRO VR FRAME REJECTED: reason=%s timestamp_ns=%d frame_index=%d "
            "last_timestamp_ns=%s last_frame_index=%s",
            reason,
            frame.source_timestamp_ns,
            frame.frame_index,
            self._last_timestamp_ns,
            self._last_frame_index,
        )

    @staticmethod
    def _event(name: str, timestamp_ns: int, **detail: Any) -> LogicalInput:
        return LogicalInput(name=name, source_timestamp_ns=timestamp_ns, detail=detail)

    @staticmethod
    def _translate_to_receive_clock(
        events: list[LogicalInput],
        *,
        source_now_ns: int,
        received_now_ns: int,
    ) -> list[LogicalInput]:
        """Align logical events with G2/H5 time while retaining raw VR time."""

        offset_ns = received_now_ns - source_now_ns
        if offset_ns == 0:
            return events
        return [
            LogicalInput(
                name=event.name,
                source_timestamp_ns=event.source_timestamp_ns + offset_ns,
                detail={
                    **event.detail,
                    "vr_source_timestamp_ns": event.source_timestamp_ns,
                    "g2_receive_clock_offset_ns": offset_ns,
                },
            )
            for event in events
        ]


class SupervisorMode(StrEnum):
    DISABLED = "disabled"
    HOLD = "hold"
    AUTO = "auto"
    FORWARD_RUN = "forward_run"
    HUMAN = "human"
    ROLLBACK_EXEC = "rollback_exec"
    ABORT = "abort"


@dataclass(frozen=True)
class SupervisorTransition:
    previous: SupervisorMode
    current: SupervisorMode
    reason: str
    generation_id: str | None
    generation_invalidated: bool


class RolloutSupervisor:
    """Pure control-plane state machine; it never grants physical human control."""

    def __init__(self) -> None:
        self.mode = SupervisorMode.DISABLED
        self.generation_id: str | None = None

    def handle(
        self,
        logical_event: str,
        *,
        handback_evidence: HumanHandbackEvidence | None = None,
    ) -> SupervisorTransition | None:
        previous = self.mode
        invalidated = False
        if logical_event in {"emergency", "fatal_fault"}:
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.ABORT
        elif logical_event == "flowpro_mode_toggle" and self.mode == SupervisorMode.DISABLED:
            self.mode = SupervisorMode.HOLD
        elif (
            logical_event in {
                "flowpro_mode_toggle", "tablet_recording_finished", "tablet_recording_replaced",
            }
            and self.mode == SupervisorMode.HOLD
        ):
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.DISABLED
        elif logical_event == "y_press" and self.mode in {SupervisorMode.DISABLED, SupervisorMode.HOLD}:
            self.generation_id = str(uuid4())
            self.mode = SupervisorMode.AUTO
        elif logical_event == "y_press" and self.mode in {
            SupervisorMode.AUTO,
            SupervisorMode.FORWARD_RUN,
        }:
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.HOLD
        elif logical_event == "y_short" and self.mode == SupervisorMode.HOLD:
            self.generation_id = str(uuid4())
            self.mode = SupervisorMode.AUTO
        elif logical_event == "y_long_start" and self.mode == SupervisorMode.HOLD:
            self.generation_id = str(uuid4())
            self.mode = SupervisorMode.FORWARD_RUN
        elif (
            (logical_event == "y_long_end" and self.mode == SupervisorMode.FORWARD_RUN)
            or (logical_event == "x_press" and self.mode in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN})
            or (
                logical_event == "side_takeover_input"
                and self.mode
                in {
                    SupervisorMode.AUTO,
                    SupervisorMode.FORWARD_RUN,
                }
            )
        ):
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.HOLD
        elif logical_event == "rollback_start" and self.mode == SupervisorMode.HOLD:
            self.generation_id = str(uuid4())
            self.mode = SupervisorMode.ROLLBACK_EXEC
        elif (
            logical_event in {"rollback_pause", "rollback_complete"}
            and self.mode == SupervisorMode.ROLLBACK_EXEC
        ):
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.HOLD
        elif logical_event == "human_takeover" and self.mode not in {
            SupervisorMode.DISABLED,
            SupervisorMode.ABORT,
        }:
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.HUMAN
        elif logical_event == "external_authority" and self.mode not in {
            SupervisorMode.DISABLED,
            SupervisorMode.ABORT,
        }:
            # Vendor authority is physically Human even when FlowPRO did not
            # observe a side-button request. Enter the same safe handback state,
            # but the paper runtime does not create correction labels for it.
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.HUMAN
        elif logical_event == "human_finish" and self.mode == SupervisorMode.HUMAN:
            if handback_evidence is None:
                raise ValueError("human_finish requires fail-closed handback evidence")
            if not evaluate_human_handback(handback_evidence).ready:
                return None
            invalidated = self._invalidate_generation()
            self.mode = SupervisorMode.HOLD
        elif (
            logical_event
            in {"vr_stale", "watchdog", "network_lost", "motion_unhealthy"}
            and self.mode == SupervisorMode.HUMAN
        ):
            # HUMAN is vendor authority, not a local motion lease. Transient
            # local health loss must keep the local GDK stream suspended and
            # defer handback until all evidence recovers. Moving the logical
            # state to HOLD here would strand the Human epoch without its
            # measured-pose re-anchor and could later expose a stale target.
            return None
        elif logical_event in {
            "vr_stale",
            "watchdog",
            "network_lost",
            "motion_unhealthy",
        }:
            invalidated = self._invalidate_generation()
            if self.mode not in {SupervisorMode.DISABLED, SupervisorMode.ABORT}:
                self.mode = SupervisorMode.HOLD
        else:
            return None
        return SupervisorTransition(
            previous=previous,
            current=self.mode,
            reason=logical_event,
            generation_id=self.generation_id,
            generation_invalidated=invalidated,
        )

    def request_human_handback(
        self,
        evidence: HumanHandbackEvidence,
    ) -> tuple[HumanHandbackDecision, SupervisorTransition | None]:
        decision = evaluate_human_handback(evidence)
        if not decision.ready:
            return decision, None
        return decision, self.handle("human_finish", handback_evidence=evidence)

    def _invalidate_generation(self) -> bool:
        existed = self.generation_id is not None
        self.generation_id = None
        return existed


@dataclass(frozen=True)
class OwnershipEvidence:
    owner: str
    reason: str
    generation_id: str | None


@dataclass(frozen=True)
class HumanHandbackEvidence:
    """Fail-closed evidence required before HUMAN may transition to HOLD."""

    side_buttons_released: bool
    accepted_retarget_fresh: bool
    action_src: str
    robot_settled: bool
    measured_state_fresh: bool
    cameras_fresh: bool
    network_fresh: bool
    motion_healthy: bool
    emergency_stop_active: bool = False
    collision_detected: bool = False


@dataclass(frozen=True)
class HumanHandbackDecision:
    ready: bool
    blockers: tuple[str, ...]


def evaluate_human_handback(evidence: HumanHandbackEvidence) -> HumanHandbackDecision:
    """Require explicit release, arbitration clear, settle and health evidence."""

    blockers: list[str] = []
    if not evidence.side_buttons_released:
        blockers.append("side_buttons_not_released")
    if evidence.accepted_retarget_fresh:
        blockers.append("accepted_retarget_still_fresh")
    if evidence.action_src in {"TAKEOVER", "BACKWARD"}:
        blockers.append("human_action_source_still_active")
    elif evidence.action_src not in {"IDLE", "MODEL"}:
        blockers.append("unknown_action_source")
    if not evidence.robot_settled:
        blockers.append("robot_not_settled")
    if not evidence.measured_state_fresh:
        blockers.append("measured_state_stale")
    if not evidence.cameras_fresh:
        blockers.append("camera_data_stale")
    if not evidence.network_fresh:
        blockers.append("network_stale")
    if not evidence.motion_healthy:
        blockers.append("motion_unhealthy")
    if evidence.emergency_stop_active:
        blockers.append("emergency_stop_active")
    if evidence.collision_detected:
        blockers.append("collision_detected")
    return HumanHandbackDecision(ready=not blockers, blockers=tuple(blockers))


@dataclass(frozen=True)
class CompletedHumanIntervention:
    intervention_id: str
    request_timestamp_ns: int
    correction_start_ns: int
    correction_end_ns: int
    handback_timestamp_ns: int


class HumanInterventionTracker:
    """Track repeated interventions within one FlowPRO session."""

    def __init__(self) -> None:
        self.intervention_id: str | None = None
        self.request_timestamp_ns: int | None = None
        self.correction_start_ns: int | None = None
        self.correction_end_ns: int | None = None
        self._completed_ids: set[str] = set()

    def begin(self, *, request_timestamp_ns: int, intervention_id: str | None = None) -> str:
        if self.intervention_id is not None:
            raise RuntimeError("A Human intervention is already active")
        if request_timestamp_ns <= 0:
            raise ValueError("request_timestamp_ns must be positive")
        value = intervention_id or str(uuid4())
        UUID(value)
        if value in self._completed_ids:
            raise ValueError(f"intervention_id has already completed: {value}")
        self.intervention_id = value
        self.request_timestamp_ns = request_timestamp_ns
        self.correction_start_ns = None
        self.correction_end_ns = None
        return value

    def observe_accepted_retarget(self, *, vendor_timestamp_ns: int) -> None:
        if self.intervention_id is None:
            raise RuntimeError("Cannot record accepted retarget without an active intervention")
        if vendor_timestamp_ns <= 0:
            raise ValueError("vendor_timestamp_ns must be positive")
        if self.correction_end_ns is not None and vendor_timestamp_ns <= self.correction_end_ns:
            raise ValueError("Accepted retarget timestamps must be strictly increasing")
        if self.correction_start_ns is None:
            self.correction_start_ns = vendor_timestamp_ns
        self.correction_end_ns = vendor_timestamp_ns

    def finish(
        self,
        *,
        handback_timestamp_ns: int,
        evidence: HumanHandbackEvidence,
    ) -> tuple[HumanHandbackDecision, CompletedHumanIntervention | None]:
        if self.intervention_id is None or self.request_timestamp_ns is None:
            raise RuntimeError("No Human intervention is active")
        decision = evaluate_human_handback(evidence)
        if not decision.ready:
            return decision, None
        if self.correction_start_ns is None or self.correction_end_ns is None:
            raise RuntimeError("Cannot finish an intervention without accepted retarget evidence")
        if handback_timestamp_ns < self.correction_end_ns:
            raise ValueError("handback_timestamp_ns cannot precede the last accepted retarget")
        completed = CompletedHumanIntervention(
            intervention_id=self.intervention_id,
            request_timestamp_ns=self.request_timestamp_ns,
            correction_start_ns=self.correction_start_ns,
            correction_end_ns=self.correction_end_ns,
            handback_timestamp_ns=handback_timestamp_ns,
        )
        self._completed_ids.add(completed.intervention_id)
        self.intervention_id = None
        self.request_timestamp_ns = None
        self.correction_start_ns = None
        self.correction_end_ns = None
        return decision, completed


class SystemArbitrationMonitor:
    """Combine vendor arbitration with local lease/ACK evidence.

    The priority-40 policy channel may leave vendor ``action_src`` at ``IDLE``;
    therefore policy ownership is proven by a fresh local lease and successful
    GDK ACK while human evidence is absent.  This class observes ownership only
    and never publishes vendor events or grants motion authority.
    """

    def __init__(self, *, evidence_timeout_ms: float = 200.0) -> None:
        if evidence_timeout_ms <= 0:
            raise ValueError("evidence_timeout_ms must be positive")
        self.timeout_ns = int(evidence_timeout_ms * 1_000_000)
        self.motion_healthy = True
        self.action_src = "IDLE"
        self.takeover_event_timestamp_ns: int | None = None
        self.accepted_retarget_timestamp_ns: int | None = None
        self.accepted_retarget_freshness_timestamp_ns: int | None = None
        self.policy_generation_id: str | None = None
        self.policy_lease_timestamp_ns: int | None = None
        self.policy_ack_timestamp_ns: int | None = None
        self.rollback_generation_id: str | None = None
        self.rollback_lease_timestamp_ns: int | None = None
        self.rollback_ack_timestamp_ns: int | None = None
        self.rollback_deadman = False
        self.rollback_auto_finish = False
        self._operator_event_keys: set[tuple[Any, ...]] = set()
        # DDS evidence arrives on the coordinator thread while Policy/rollback
        # command acknowledgements arrive on the sole GDK action thread.
        self._lock = threading.RLock()

    def observe_operator_event(
        self,
        *,
        dedup_key: tuple[Any, ...],
        detail: str,
        vendor_timestamp_ns: int,
    ) -> bool:
        with self._lock:
            if dedup_key in self._operator_event_keys:
                return False
            self._operator_event_keys.add(dedup_key)
            if detail in {"takeover_start", "takeover_backward"}:
                self.takeover_event_timestamp_ns = vendor_timestamp_ns
            return True

    def observe_accepted_retarget(
        self,
        *,
        vendor_timestamp_ns: int,
        freshness_timestamp_ns: int | None = None,
    ) -> None:
        with self._lock:
            if vendor_timestamp_ns <= 0:
                raise ValueError("Accepted retarget timestamp must be positive")
            if (
                self.accepted_retarget_timestamp_ns is not None
                and vendor_timestamp_ns <= self.accepted_retarget_timestamp_ns
            ):
                raise ValueError("Accepted retarget timestamps must be strictly increasing")
            self.accepted_retarget_timestamp_ns = vendor_timestamp_ns
            self.accepted_retarget_freshness_timestamp_ns = (
                vendor_timestamp_ns if freshness_timestamp_ns is None else freshness_timestamp_ns
            )

    def observe_action_src(self, action_src: str) -> None:
        if action_src not in {"IDLE", "TAKEOVER", "BACKWARD", "MODEL"}:
            raise ValueError(f"Unknown system action_src: {action_src!r}")
        with self._lock:
            self.action_src = action_src

    def set_motion_health(self, healthy: bool) -> None:
        with self._lock:
            self.motion_healthy = bool(healthy)

    def observe_policy_lease(self, generation_id: str, *, timestamp_ns: int) -> None:
        with self._lock:
            if generation_id != self.policy_generation_id:
                self.policy_ack_timestamp_ns = None
            self.policy_generation_id = generation_id
            self.policy_lease_timestamp_ns = timestamp_ns

    def observe_policy_ack(self, generation_id: str, *, timestamp_ns: int) -> None:
        with self._lock:
            if generation_id != self.policy_generation_id:
                return
            self.policy_ack_timestamp_ns = timestamp_ns

    def observe_rollback_lease(
        self,
        generation_id: str,
        *,
        timestamp_ns: int,
        deadman: bool,
        auto_finish: bool = False,
    ) -> None:
        with self._lock:
            if generation_id != self.rollback_generation_id:
                self.rollback_ack_timestamp_ns = None
            self.rollback_generation_id = generation_id
            self.rollback_lease_timestamp_ns = timestamp_ns
            self.rollback_deadman = bool(deadman)
            self.rollback_auto_finish = bool(auto_finish)

    def observe_rollback_ack(self, generation_id: str, *, timestamp_ns: int) -> None:
        with self._lock:
            if generation_id != self.rollback_generation_id:
                return
            self.rollback_ack_timestamp_ns = timestamp_ns

    def invalidate_local_leases(self) -> None:
        with self._lock:
            self.policy_generation_id = None
            self.policy_lease_timestamp_ns = None
            self.policy_ack_timestamp_ns = None
            self.rollback_generation_id = None
            self.rollback_lease_timestamp_ns = None
            self.rollback_ack_timestamp_ns = None
            self.rollback_deadman = False
            self.rollback_auto_finish = False

    def determine_owner(self, *, now_ns: int) -> OwnershipEvidence:
        with self._lock:
            human_source = self.action_src in {"TAKEOVER", "BACKWARD"}
            if human_source:
                self.invalidate_local_leases()
            if not self.motion_healthy:
                return OwnershipEvidence("hold", "motion_unhealthy", None)
            accepted_retarget_fresh = self._fresh(
                self.accepted_retarget_freshness_timestamp_ns,
                now_ns,
            )
            if accepted_retarget_fresh and human_source:
                return OwnershipEvidence("human", "fresh_accepted_retarget_and_system_accept", None)
            if human_source:
                return OwnershipEvidence("hold", "system_human_source_without_fresh_retarget", None)
            if accepted_retarget_fresh:
                return OwnershipEvidence("hold", "fresh_retarget_without_human_action_source", None)
            if (
                (self.rollback_deadman or self.rollback_auto_finish)
                and self.rollback_generation_id is not None
                and self._fresh(self.rollback_lease_timestamp_ns, now_ns)
                and self._fresh(self.rollback_ack_timestamp_ns, now_ns)
            ):
                return OwnershipEvidence(
                    "rollback",
                    (
                        "fresh_local_rollback_lease_ack_deadman"
                        if self.rollback_deadman
                        else "fresh_local_rollback_lease_ack_finish_to_anchor"
                    ),
                    self.rollback_generation_id,
                )
            if (
                self.policy_generation_id is not None
                and self._fresh(self.policy_lease_timestamp_ns, now_ns)
                and self._fresh(self.policy_ack_timestamp_ns, now_ns)
            ):
                return OwnershipEvidence(
                    "policy",
                    "fresh_local_policy_lease_and_gdk_ack",
                    self.policy_generation_id,
                )
            return OwnershipEvidence("hold", "insufficient_fresh_ownership_evidence", None)

    def _fresh(self, timestamp_ns: int | None, now_ns: int) -> bool:
        return timestamp_ns is not None and 0 <= now_ns - timestamp_ns <= self.timeout_ns


class EpisodeCoordinator:
    """Bind a sidecar to the operator-controlled tablet recorder lifecycle."""

    STATE_SCHEMA_VERSION = "flowpro.coordinator_state/v2"
    PHASE_ATTACHED = "attached_recording"
    PHASE_AWAITING_SAVE = "awaiting_tablet_save"
    PHASE_POSTPROCESSING = "vendor_postprocessing"
    PHASE_READY = "ready_to_seal"
    # Audited active placeholders advertise a 600 s recorder window.  Five
    # additional minutes cover operator/setup latency without allowing an
    # abandoned placeholder from an earlier shift to be attached.
    ACTIVE_META_MAX_AGE_NS = 15 * 60 * 1_000_000_000
    # A live vendor recording updates either DDS chunks under ``record`` or
    # camera streams/timestamp files under ``camera``. The first DDS chunks
    # may be preallocated and remain unchanged for more than ten seconds while
    # the robot is idle, whereas camera timestamp files keep advancing.
    ACTIVE_RECORD_ACTIVITY_MAX_AGE_NS = 10 * 1_000_000_000

    def __init__(
        self,
        *,
        sidecar_root: Path,
        vendor_record_root: Path,
        robot_id: str,
        task_id: str,
        policy: dict[str, Any],
        collection_manifest: dict[str, Any] | None = None,
        completion_stability_seconds: float = 10.0,
    ) -> None:
        if completion_stability_seconds <= 0:
            raise ValueError("completion_stability_seconds must be positive")
        self.sidecar_root = sidecar_root.expanduser().resolve()
        self.vendor_record_root = vendor_record_root.expanduser().resolve()
        self.robot_id = robot_id
        self.task_id = task_id
        self.policy = dict(policy)
        self.collection_manifest = None if collection_manifest is None else dict(collection_manifest)
        self.completion_stability_ns = int(completion_stability_seconds * 1_000_000_000)
        self.writer: FlowPROSidecarWriter | None = None
        self.wall_to_vendor_offset_ns = 0

    @property
    def coordinator_state_path(self) -> Path:
        return self._require_active().open_directory / "coordinator_state.json"

    def attach_active_recording(
        self,
        vendor_episode_path: Path,
        *,
        flowpro_session_id: str | None = None,
    ) -> str:
        if self.writer is not None:
            raise RuntimeError("EpisodeCoordinator already has an active episode")
        episode = self._resolve_vendor_episode(vendor_episode_path)
        discovered = self.discover_active_recording(self.vendor_record_root)
        if discovered != episode:
            raise ValueError(
                f"Requested recording {episode.name!r} is not the unique active recording {discovered.name!r}"
            )
        meta_path = episode / VENDOR_META_PATH
        if not meta_path.is_file() or meta_path.is_symlink():
            raise FileNotFoundError(f"Active recording is missing a regular {VENDOR_META_PATH}: {episode}")
        meta = read_vendor_meta(meta_path)
        if classify_vendor_meta(meta) != VENDOR_META_ACTIVE:
            raise ValueError("FlowPRO may attach only while vendor meta is an active-recording placeholder")
        for open_directory in self.sidecar_root.glob(f"*{SIDECAR_OPEN_SUFFIX}"):
            if open_directory.is_symlink() or not open_directory.is_dir():
                continue
            state_path = open_directory / "coordinator_state.json"
            if not state_path.is_file() or state_path.is_symlink():
                continue
            try:
                existing_state = json.loads(state_path.read_text(encoding="utf-8"))
                existing_episode = Path(existing_state["vendor_episode_path"]).resolve()
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"Cannot verify existing open FlowPRO sidecar binding: {open_directory}"
                ) from error
            if existing_episode == episode:
                raise RuntimeError(
                    "Vendor recording already has an open FlowPRO sidecar; "
                    f"recover it before reattaching: {open_directory}"
                )
        session_id = flowpro_session_id or str(uuid4())
        writer = FlowPROSidecarWriter.create(
            self.sidecar_root,
            flowpro_session_id=session_id,
            robot_id=self.robot_id,
            task_id=self.task_id,
            collection_manifest=self.collection_manifest,
            wall_to_vendor_offset_ns=self.wall_to_vendor_offset_ns,
            **self.policy,
        )
        self.writer = writer
        _write_json_atomic(
            self.coordinator_state_path,
            {
                "schema_version": self.STATE_SCHEMA_VERSION,
                "flowpro_session_id": session_id,
                "phase": self.PHASE_ATTACHED,
                "vendor_episode_path": str(episode),
                "record_name": episode.name,
                "attached_meta_identity": list(self._file_identity(meta_path)),
                "outcome": None,
                "completion_snapshot": None,
                "stable_since_observed_at_ns": None,
            },
        )
        self._append_lifecycle(
            "SessionStarted",
            {
                "record_name": episode.name,
                "vendor_episode_path": str(episode),
                "recorder_control": "external_tablet",
                "collection_manifest": self.collection_manifest,
            },
        )
        return session_id

    def append_event(self, event: dict[str, Any]) -> None:
        self._require_active().append_event(event)

    def attached_recording_finalized(self) -> bool:
        """Return whether the bound tablet recorder has published final meta."""

        state = self._read_state()
        if state["phase"] not in {self.PHASE_ATTACHED, self.PHASE_AWAITING_SAVE}:
            return False
        episode = self._resolve_vendor_episode(Path(state["vendor_episode_path"]))
        meta_path = episode / VENDOR_META_PATH
        if meta_path.is_symlink() or not meta_path.is_file():
            return False
        return classify_vendor_meta(read_vendor_meta(meta_path)) == VENDOR_META_FINAL

    def attached_recording_active(self) -> bool:
        """Check the bound recorder's recent file activity without changing it."""

        state = self._read_state()
        if state["phase"] not in {self.PHASE_ATTACHED, self.PHASE_AWAITING_SAVE}:
            return False
        episode = self._resolve_vendor_episode(Path(state["vendor_episode_path"]))
        return self._recording_is_active(episode, require_fresh_meta=False)

    def append_logical_input(self, logical_input: LogicalInput, *, control_owner: str) -> None:
        writer = self._require_active()
        # Tablet finalization is observed after the vendor clip has ended.  Its
        # source timestamp is the filesystem-observation time, not a timestamp
        # on the robot/camera timeline, so it must not participate in the
        # in-episode vendor-clock join.
        tablet_observation_sources = {
            "tablet_auto_start", "tablet_recording_finished", "tablet_recording_inactive",
            "tablet_recording_replaced",
        }
        vendor_timestamp_ns = (
            None
            if logical_input.name in {"tablet_recording_finished", "tablet_recording_replaced"}
            or logical_input.detail.get("source") in tablet_observation_sources
            else logical_input.source_timestamp_ns
        )
        writer.append_event(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_id": str(uuid4()),
                "flowpro_session_id": writer.flowpro_session_id,
                "event_type": "LogicalInput",
                "producer": "g2_supervisor",
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_utc": _utc_now(),
                "vendor_timestamp_ns": vendor_timestamp_ns,
                "control_owner": control_owner,
                "payload": {"name": logical_input.name, **logical_input.detail},
            }
        )

    def record_observation(
        self,
        *,
        observation_id: str,
        anchor_timestamp_ns: int,
        state_24d: Any,
        task: str,
        images: dict[str, dict[str, Any]],
        control_owner: str = "hold",
    ) -> dict[str, Any]:
        """Persist the complete, exact inference input before policy generation."""

        writer = self._require_active()
        UUID(observation_id)
        if not isinstance(task, str) or len(task) > 4096:
            raise ValueError("Inference task must be a string no longer than 4096 characters")
        if set(images) != set(CAMERA_SOURCES):
            raise ValueError(f"Observation must contain exactly these cameras: {tuple(CAMERA_SOURCES)}")
        for camera_key, image in images.items():
            image_timestamp_ns = int(image["vendor_timestamp_ns"])
            if abs(image_timestamp_ns - anchor_timestamp_ns) > 40_000_000:
                raise ValueError(f"Observation camera {camera_key} exceeds 40 ms anchor skew")
        image_records: dict[str, dict[str, Any]] = {}
        for camera_key, image in images.items():
            image_timestamp_ns = int(image["vendor_timestamp_ns"])
            image_records[camera_key] = writer.store_inference_jpeg(
                bytes(image["data"]),
                camera_key=camera_key,
                vendor_timestamp_ns=image_timestamp_ns,
                width=int(image["width"]),
                height=int(image["height"]),
            )
        state_record = writer.store_observation_state(observation_id, state_24d)
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "ObservationCaptured",
            "producer": "g2_supervisor",
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _utc_now(),
            "vendor_timestamp_ns": anchor_timestamp_ns,
            "observation_id": observation_id,
            "control_owner": control_owner,
            "payload": {
                "anchor_timestamp_ns": anchor_timestamp_ns,
                "state": state_record,
                "task": task,
                "images": image_records,
            },
        }
        writer.append_event(event)
        return event

    def record_policy_candidate(
        self,
        values: Any,
        *,
        observation_id: str,
        request_id: str,
        candidate_id: str,
        generation_id: str,
        control_owner: str = "hold",
        sampling: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._require_active().record_policy_candidate(
            values,
            event_id=str(uuid4()),
            observation_id=observation_id,
            request_id=request_id,
            candidate_id=candidate_id,
            generation_id=generation_id,
            monotonic_ns=time.monotonic_ns(),
            wall_time_utc=_utc_now(),
            control_owner=control_owner,
            sampling=sampling,
        )

    def record_executed_segment(
        self,
        *,
        intervention_id: str,
        selection: Any,
        segment: Any,
        native_action_dim: int,
    ) -> dict[str, Any]:
        """Freeze the actually executed negative trajectory before rollback."""

        writer = self._require_active()
        UUID(intervention_id)
        artifacts = writer.store_executed_segment(intervention_id, segment)
        first = segment.samples[0]
        payload = {
            "protocol": "flowpro.executed_trajectory/v2",
            "intervention_id": intervention_id,
            "failure_timestamp_ns": int(selection.failure_timestamp_ns),
            "requested_horizon_ns": int(selection.requested_horizon_ns),
            "requested_target_timestamp_ns": int(selection.requested_target_timestamp_ns),
            "actual_anchor_timestamp_ns": int(selection.actual_anchor_timestamp_ns),
            "snap_offset_ns": int(selection.snap_offset_ns),
            "sample_count": len(segment.samples),
            "native_action_dim": native_action_dim,
            "execution_quality": {
                "execution_kind_counts": segment.execution_kind_counts,
                "queue_underrun_count": segment.queue_underrun_count,
                "execution_gap_count": segment.execution_gap_count,
                "max_execution_gap_ns": segment.max_execution_gap_ns,
                "physical_history_complete": segment.physical_history_complete,
                "strict_preference_eligible": segment.strict_preference_eligible,
            },
            "artifacts": artifacts,
        }
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "ExecutedSegmentCaptured",
            "producer": "robot_client",
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _utc_now(),
            "vendor_timestamp_ns": int(selection.failure_timestamp_ns),
            "observation_id": segment.anchor.observation_id,
            "candidate_id": first.candidate_id,
            "generation_id": first.generation_id,
            "action_id": first.action_id,
            "control_owner": "hold",
            "payload": payload,
        }
        writer.append_event(event)
        return event

    def record_paper_phase(
        self,
        event_type: str,
        *,
        intervention_id: str,
        vendor_timestamp_ns: int,
        payload: dict[str, Any] | None = None,
        observation_id: str | None = None,
        generation_id: str | None = None,
        control_owner: str = "hold",
        producer: str = "g2_supervisor",
    ) -> dict[str, Any]:
        """Append one durable paper-protocol phase boundary."""

        allowed = {
            "RollbackRequested",
            "RollbackPlanned",
            "RollbackResumed",
            "RollbackPaused",
            "RollbackAnchorCrossed",
            "RollbackInvalidated",
            "RollbackPlanningFailed",
            "RollbackCompleted",
            "SceneRestorationStarted",
            "SceneRestorationVerified",
            "HumanCorrectionStarted",
            "HumanCorrectionCompleted",
        }
        if event_type not in allowed:
            raise ValueError(f"Unsupported paper phase event: {event_type}")
        UUID(intervention_id)
        writer = self._require_active()
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": event_type,
            "producer": producer,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _utc_now(),
            "vendor_timestamp_ns": vendor_timestamp_ns,
            "observation_id": observation_id,
            "generation_id": generation_id,
            "control_owner": control_owner,
            "payload": {"intervention_id": intervention_id, **dict(payload or {})},
        }
        writer.append_event(event)
        return event

    def record_ownership_change(
        self,
        *,
        previous_owner: str,
        current_owner: str,
        reason: str,
        vendor_timestamp_ns: int,
        evidence: dict[str, Any],
        generation_id: str | None,
    ) -> dict[str, Any]:
        writer = self._require_active()
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "ControlOwnershipChanged",
            "producer": "g2_supervisor",
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _utc_now(),
            "vendor_timestamp_ns": vendor_timestamp_ns,
            "generation_id": generation_id,
            "control_owner": current_owner,
            "payload": {
                "previous_owner": previous_owner,
                "current_owner": current_owner,
                "reason": reason,
                "evidence": evidence,
            },
        }
        writer.append_event(event)
        return event

    def finalize_human_correction(
        self,
        *,
        intervention_id: str,
        observation_id: str,
        candidate_id: str,
        correction_start_ns: int,
        correction_end_ns: int,
        scene_verified: bool,
        sensor_skew_ok: bool,
        same_anchor: bool,
        pair_reject: bool,
        invalid_reason: str | None = None,
    ) -> dict[str, Any]:
        writer = self._require_active()
        payload = {
            "phase": "finalized",
            "intervention_id": intervention_id,
            "observation_id": observation_id,
            "candidate_id": candidate_id,
            "correction_start_ns": correction_start_ns,
            "correction_end_ns": correction_end_ns,
            "scene_verified": scene_verified,
            "sensor_skew_ok": sensor_skew_ok,
            "same_anchor": same_anchor,
            "pair_reject": pair_reject,
        }
        if invalid_reason is not None:
            payload["invalid_reason"] = invalid_reason
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "HumanIntervention",
            "producer": "g2_supervisor",
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _utc_now(),
            "vendor_timestamp_ns": correction_end_ns,
            "observation_id": observation_id,
            "candidate_id": candidate_id,
            "control_owner": "hold",
            "payload": payload,
        }
        writer.append_event(event)
        return event

    def finish_control(
        self,
        *,
        outcome: str,
        outcome_detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """End FlowPRO control and wait for the operator to save on the tablet."""

        self._require_active()
        if outcome not in {
            "pending_recorder_commit",
            "success",
            "failure",
            "aborted",
            "unknown",
        }:
            raise ValueError(f"Unsupported episode outcome: {outcome!r}")
        state = self._read_state()
        if state["phase"] == self.PHASE_ATTACHED:
            self._append_lifecycle(
                "EpisodeOutcome",
                {"outcome": outcome, "detail": outcome_detail or {}},
            )
            state = {
                **state,
                "phase": self.PHASE_AWAITING_SAVE,
                "outcome": outcome,
                "outcome_detail": outcome_detail or {},
            }
            _write_json_atomic(self.coordinator_state_path, state)
        elif state["phase"] in {
            self.PHASE_AWAITING_SAVE,
            self.PHASE_POSTPROCESSING,
            self.PHASE_READY,
        }:
            if state.get("outcome") != outcome:
                raise ValueError("Retry outcome differs from the already recorded episode outcome")
        else:
            raise RuntimeError(f"Cannot finish FlowPRO control from phase {state['phase']!r}")
        return state

    def observe_vendor_completion(self, *, observed_at_ns: int | None = None) -> dict[str, Any]:
        """Advance the external tablet lifecycle without sleeping or mutating vendor files."""

        self._require_active()
        now_ns = time.time_ns() if observed_at_ns is None else observed_at_ns
        if now_ns <= 0:
            raise ValueError("observed_at_ns must be positive")
        state = self._read_state()
        if state["phase"] == self.PHASE_ATTACHED:
            raise RuntimeError("finish_control() must run before observing tablet save completion")
        if state["phase"] not in {
            self.PHASE_AWAITING_SAVE,
            self.PHASE_POSTPROCESSING,
            self.PHASE_READY,
        }:
            raise RuntimeError(f"Unsupported coordinator phase: {state['phase']!r}")

        episode = self._resolve_vendor_episode(Path(state["vendor_episode_path"]))
        meta_path = episode / VENDOR_META_PATH
        if not meta_path.is_file() or meta_path.is_symlink():
            raise FileNotFoundError(meta_path)
        meta_phase = classify_vendor_meta(read_vendor_meta(meta_path))
        if meta_phase == VENDOR_META_ACTIVE:
            updated = {
                **state,
                "phase": self.PHASE_AWAITING_SAVE,
                "completion_reason": "tablet_save_not_observed",
                "completion_snapshot": None,
                "stable_since_observed_at_ns": None,
                "last_observed_at_ns": now_ns,
            }
            _write_json_atomic(self.coordinator_state_path, updated)
            return updated
        if meta_phase != VENDOR_META_FINAL:
            raise RuntimeError(f"Unexpected vendor meta phase: {meta_phase!r}")

        # A final, validator-approved vendor meta is the first authoritative
        # evidence that the operator selected Save on the external tablet.
        # Before this point absence of final files remains pending, never a
        # silent failure/discard decision.
        if state.get("outcome") == "pending_recorder_commit":
            state = {
                **state,
                "outcome": "success",
                "outcome_detail": {
                    "outcome_authority": "external_tablet_recorder",
                    "evidence": "final_vendor_meta",
                    "observed_at_ns": now_ns,
                },
            }
            self._append_lifecycle(
                "EpisodeOutcome",
                {
                    "outcome": "success",
                    "detail": state["outcome_detail"],
                },
            )

        snapshot, missing = self._snapshot_required_files(episode)
        if missing:
            updated = {
                **state,
                "phase": self.PHASE_POSTPROCESSING,
                "completion_reason": "required_files_missing",
                "missing_files": missing,
                "completion_snapshot": None,
                "stable_since_observed_at_ns": None,
                "last_observed_at_ns": now_ns,
            }
            _write_json_atomic(self.coordinator_state_path, updated)
            return updated

        previous_snapshot = state.get("completion_snapshot")
        stable_since_ns = state.get("stable_since_observed_at_ns")
        if previous_snapshot != snapshot or not isinstance(stable_since_ns, int) or now_ns < stable_since_ns:
            updated = {
                **state,
                "phase": self.PHASE_POSTPROCESSING,
                "completion_reason": "stability_window_started",
                "missing_files": [],
                "completion_snapshot": snapshot,
                "stable_since_observed_at_ns": now_ns,
                "last_observed_at_ns": now_ns,
            }
            _write_json_atomic(self.coordinator_state_path, updated)
            return updated

        elapsed_ns = now_ns - stable_since_ns
        ready = elapsed_ns >= self.completion_stability_ns
        updated = {
            **state,
            "phase": self.PHASE_READY if ready else self.PHASE_POSTPROCESSING,
            "completion_reason": "stable_and_ready" if ready else "stability_window_pending",
            "missing_files": [],
            "completion_snapshot": snapshot,
            "stable_since_observed_at_ns": stable_since_ns,
            "last_observed_at_ns": now_ns,
            "stable_elapsed_ns": elapsed_ns,
        }
        _write_json_atomic(self.coordinator_state_path, updated)
        return updated

    def finalize(self) -> Path:
        writer = self._require_active()
        state = self._read_state()
        if state["phase"] != self.PHASE_READY:
            raise RuntimeError(
                f"Vendor episode is not ready to seal; coordinator phase is {state['phase']!r}"
            )
        episode = self._resolve_vendor_episode(Path(state["vendor_episode_path"]))
        snapshot, missing = self._snapshot_required_files(episode)
        if missing or snapshot != state.get("completion_snapshot"):
            reset = {
                **state,
                "phase": self.PHASE_POSTPROCESSING,
                "completion_reason": "files_changed_after_completion_barrier",
                "missing_files": missing,
                "completion_snapshot": snapshot if not missing else None,
                "stable_since_observed_at_ns": None,
            }
            _write_json_atomic(self.coordinator_state_path, reset)
            raise RuntimeError("Vendor episode changed after completion barrier; observe stability again")
        writer.close()
        sealed = seal_sidecar(
            writer.open_directory,
            episode,
            vendor_record_root=self.vendor_record_root,
        )
        self.writer = None
        return sealed

    def detach_pending_recording(self) -> Path:
        """Release a stopped, unconfirmed clip while retaining its open sidecar.

        A later final tablet meta can still be sealed by opening this sidecar
        again. No discarded/successful outcome is inferred here.
        """

        writer = self._require_active()
        state = self._read_state()
        if state["phase"] != self.PHASE_AWAITING_SAVE:
            raise RuntimeError("Only an episode awaiting tablet outcome can be detached")
        directory = writer.open_directory
        _write_json_atomic(
            self.coordinator_state_path,
            {
                **state,
                "completion_reason": "replaced_by_new_active_recording",
                "detached_at_ns": time.time_ns(),
            },
        )
        writer.close()
        self.writer = None
        return directory

    @classmethod
    def open_existing(
        cls,
        open_directory: Path,
        *,
        vendor_record_root: Path,
        completion_stability_seconds: float = 10.0,
    ) -> EpisodeCoordinator:
        writer = FlowPROSidecarWriter.open_existing(open_directory)
        policy_manifest = writer.manifest["policy"]
        coordinator = cls(
            sidecar_root=writer.open_directory.parent,
            vendor_record_root=vendor_record_root,
            robot_id=str(writer.manifest["robot_id"]),
            task_id=str(writer.manifest["task_id"]),
            policy={
                "checkpoint_id": policy_manifest["checkpoint_id"],
                "checkpoint_stats_sha256": policy_manifest["checkpoint_stats_sha256"],
                "native_action_dim": policy_manifest["native_action_dim"],
                "chunk_size": policy_manifest["chunk_size"],
                "effector_profile": policy_manifest["effector_profile"],
                "processor_version": policy_manifest["processor_version"],
            },
            completion_stability_seconds=completion_stability_seconds,
        )
        coordinator.writer = writer
        coordinator._read_state()
        return coordinator

    @staticmethod
    def resume_finalize(open_directory: Path, *, vendor_record_root: Path) -> Path:
        directory = open_directory.expanduser().resolve()
        state_path = directory / "coordinator_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != EpisodeCoordinator.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported coordinator state schema")
        if state.get("phase") != EpisodeCoordinator.PHASE_READY:
            raise RuntimeError("Only a ready_to_seal sidecar can resume finalization")
        episode = Path(state["vendor_episode_path"]).expanduser().resolve()
        root = vendor_record_root.expanduser().resolve()
        if episode.parent != root or episode.is_symlink() or not episode.is_dir():
            raise ValueError(f"Invalid bound vendor episode: {episode}")
        snapshot, missing = EpisodeCoordinator._snapshot_required_files(episode)
        if missing or snapshot != state.get("completion_snapshot"):
            raise RuntimeError("Vendor episode changed after persisted completion barrier")
        return seal_sidecar(
            directory,
            episode,
            vendor_record_root=root,
        )

    def _read_state(self) -> dict[str, Any]:
        state = json.loads(self.coordinator_state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Coordinator state must be a JSON object")
        writer = self._require_active()
        if state.get("schema_version") != self.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported coordinator state schema")
        if state.get("flowpro_session_id") != writer.flowpro_session_id:
            raise ValueError("Coordinator state session does not match sidecar")
        episode = self._resolve_vendor_episode(Path(state["vendor_episode_path"]))
        if state.get("record_name") != episode.name:
            raise ValueError("Coordinator record_name does not match vendor episode path")
        return state

    def _resolve_vendor_episode(self, episode_path: Path) -> Path:
        episode = episode_path.expanduser().resolve()
        if episode.parent != self.vendor_record_root or not episode.name or episode.name in {".", ".."}:
            raise ValueError(
                f"Vendor episode must be one direct child of {self.vendor_record_root}: {episode}"
            )
        if episode.is_symlink() or not episode.is_dir():
            raise FileNotFoundError(f"Vendor episode directory does not exist: {episode}")
        return episode

    @classmethod
    def discover_active_recording(cls, vendor_record_root: Path) -> Path:
        """Return the one live tablet recording with fresh recorder output."""

        root = vendor_record_root.expanduser().resolve()
        active = cls.list_active_recordings(root)
        if not active:
            raise RuntimeError(f"No active tablet recording found under {root}")
        if len(active) != 1:
            raise RuntimeError(
                f"Expected exactly one active tablet recording under {root}, found {[p.name for p in active]}"
            )
        return active[0]

    @classmethod
    def list_active_recordings(cls, vendor_record_root: Path) -> list[Path]:
        """List fresh tablet recordings without taking recorder ownership."""

        root = vendor_record_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        active: list[Path] = []
        for episode in sorted(root.iterdir()):
            if cls._recording_is_active(episode):
                active.append(episode.resolve())
        return active

    @classmethod
    def _recording_is_active(cls, episode: Path, *, require_fresh_meta: bool = True) -> bool:
        if episode.is_symlink() or not episode.is_dir():
            return False
        meta_path = episode / VENDOR_META_PATH
        if meta_path.is_symlink() or not meta_path.is_file():
            return False
        try:
            phase = classify_vendor_meta(read_vendor_meta(meta_path))
            if phase != VENDOR_META_ACTIVE:
                return False
            if require_fresh_meta:
                age_ns = time.time_ns() - meta_path.stat().st_mtime_ns
                if not 0 <= age_ns <= cls.ACTIVE_META_MAX_AGE_NS:
                    return False
            record_path = episode / "record"
            if record_path.is_symlink() or not record_path.is_dir():
                return False
            output_files = [
                path for path in record_path.iterdir() if not path.is_symlink() and path.is_file()
            ]
            camera_path = episode / "camera"
            if camera_path.is_dir() and not camera_path.is_symlink():
                output_files.extend(
                    path for path in camera_path.rglob("*") if not path.is_symlink() and path.is_file()
                )
            latest_output_mtime_ns = max(
                (path.stat().st_mtime_ns for path in output_files), default=0,
            )
            # Capture now after all file stats. Live camera timestamps may
            # advance while a large historical root is being scanned.
            activity_age_ns = time.time_ns() - latest_output_mtime_ns
            return 0 <= activity_age_ns <= cls.ACTIVE_RECORD_ACTIVITY_MAX_AGE_NS
        except (OSError, ValueError):
            # Incomplete/discarded recordings are never live evidence.
            return False

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @classmethod
    def _snapshot_required_files(cls, episode: Path) -> tuple[dict[str, list[int]], list[str]]:
        snapshot: dict[str, list[int]] = {}
        missing: list[str] = []
        for relative in VENDOR_REQUIRED_PATHS:
            path = episode / relative
            if path.is_symlink() or not path.is_file():
                missing.append(relative)
                continue
            identity = cls._file_identity(path)
            if identity[2] <= 0:
                missing.append(relative)
                continue
            snapshot[relative] = list(identity)
        return snapshot, sorted(missing)

    def _append_lifecycle(self, event_type: str, payload: dict[str, Any]) -> None:
        writer = self._require_active()
        # Lifecycle transitions are deliberately append-before-state-write so
        # the WAL remains the crash-safe authority.  If the following atomic
        # coordinator-state write fails (for example ENOSPC), a retry must not
        # append a second semantic outcome.
        if event_type == "EpisodeOutcome":
            requested_outcome = payload.get("outcome")
            with (writer.open_directory / "events.wal").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    existing = json.loads(line)
                    if (
                        existing.get("event_type") == "EpisodeOutcome"
                        and existing.get("payload", {}).get("outcome") == requested_outcome
                    ):
                        return
        writer.append_event(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_id": str(uuid4()),
                "flowpro_session_id": writer.flowpro_session_id,
                "event_type": event_type,
                "producer": "g2_supervisor",
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_utc": _utc_now(),
                "vendor_timestamp_ns": None,
                "control_owner": "hold",
                "payload": payload,
            }
        )

    def _require_active(self) -> FlowPROSidecarWriter:
        if self.writer is None:
            raise RuntimeError("EpisodeCoordinator has no active episode")
        return self.writer
