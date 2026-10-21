"""Composition root for the G2 DDS adapter and the standalone robot client.

The missing production-specific code is intentionally narrow: translate DDS
messages into these typed methods and provide operator feedback.  This class
owns no DDS publisher and never starts/stops the tablet recorder.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .executed import FlowPROExecutionHistory
from .feedback import SceneRestorationFeedbackWriter
from .intervention import PaperInterventionController
from .paper_runtime import (
    FlowPROPaperRuntime,
    RobotClientRollbackDriver,
)
from .rollback import RollbackHealth, RollbackTrajectory
from .runtime import FlowPROInferenceCapture, FlowPROSessionController
from .supervisor import (
    EpisodeCoordinator,
    HumanHandbackEvidence,
    LogicalInput,
    RawVRFrame,
    SupervisorMode,
    SupervisorTransition,
    SystemArbitrationMonitor,
    VRButtonParser,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyRunScope:
    """One operator-started Policy run retained for a later X rollback.

    The Policy generation created by Y is the run identifier.  Inference
    replans and temporary robot-client watchdog/queue-underrun HOLDs keep that
    generation, so they must not become physical rollback boundaries.  A Y-up
    only makes the run inactive: its history remains available until X, while
    the next successful Y start creates a new generation and replaces it.
    """

    generation_id: str
    started_timestamp_ns: int
    stopped_timestamp_ns: int | None = None
    stop_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.stopped_timestamp_ns is None


class FlowPROG2Application:
    """Bind data/control state while leaving vendor arbitration authoritative."""

    def __init__(
        self,
        *,
        robot_client_runner: Any,
        sidecar_root: Path,
        vendor_record_root: Path,
        robot_id: str,
        task_id: str,
        policy: dict[str, Any],
        rollback_planner: Callable[[Any, str, Any], RollbackTrajectory],
        arbitration_monitor: SystemArbitrationMonitor | None = None,
        feedback_root: Path | None = None,
        collision_safety_enabled: bool = False,
        rollback_enabled: bool = False,
        max_rollback_ns: int | None = None,
        collection_manifest: dict[str, Any] | None = None,
        effector_gate: Any | None = None,
    ) -> None:
        self.robot_client_runner = robot_client_runner
        self.collision_safety_enabled = bool(collision_safety_enabled)
        self.coordinator = EpisodeCoordinator(
            sidecar_root=sidecar_root,
            vendor_record_root=vendor_record_root,
            robot_id=robot_id,
            task_id=task_id,
            policy=policy,
            collection_manifest=collection_manifest,
        )
        self.history = FlowPROExecutionHistory()
        self.capture = FlowPROInferenceCapture(
            self.coordinator,
            native_action_dim=int(policy["native_action_dim"]),
            chunk_size=int(policy["chunk_size"]),
            execution_history=self.history,
        )
        self.capture.validate_server_metadata(robot_client_runner.policy_client.metadata)
        if getattr(robot_client_runner, "flowpro_capture", None) not in {None, self.capture}:
            raise RuntimeError("Robot client already has a different FlowPRO capture owner")
        robot_client_runner.flowpro_capture = self.capture
        self.arbitration = arbitration_monitor or SystemArbitrationMonitor()
        robot_client_runner.configure_flowpro_arbitration_monitor(self.arbitration)
        self.session = FlowPROSessionController(self.coordinator, self.capture)
        self._input_lock = threading.RLock()
        self.intervention = PaperInterventionController(
            coordinator=self.coordinator,
            history=self.history,
            native_action_dim=int(policy["native_action_dim"]),
            effector_profile=str(policy["effector_profile"]),
            feedback_writer=(
                None if feedback_root is None else SceneRestorationFeedbackWriter(feedback_root)
            ),
        )
        self.policy_run_scope: PolicyRunScope | None = None
        self.rollback_driver = RobotClientRollbackDriver(robot_client_runner)
        self.paper = FlowPROPaperRuntime(
            session=self.session,
            intervention=self.intervention,
            rollback_executor=self.rollback_driver,
            rollback_planner=rollback_planner,
            measured_state_provider=robot_client_runner.flowpro_rollback_start_snapshot,
            policy_run_generation_provider=lambda: (
                None if self.policy_run_scope is None else self.policy_run_scope.generation_id
            ),
            rollback_enabled=rollback_enabled,
            max_rollback_ns=max_rollback_ns,
        )
        # A button held across task startup must never become Policy activation.
        self.parser = VRButtonParser(require_initial_neutral=True)
        self.effector_gate = effector_gate
        self.vendor_record_root = vendor_record_root.expanduser().resolve()
        self.latest_x_pressed = False
        self.latest_y_pressed = False
        self.latest_side_takeover_pressed = False
        self.latest_index_triggers_pressed = (False, False)
        # A trigger held across Human -> Policy is consumed by that handoff.
        # It cannot re-enable vendor EE during a later Human takeover until a
        # complete release has been observed.
        self._trigger_mask_until_release = [False, False]
        self._human_policy_handoff_requested = False
        self._policy_effector_handoff_pending = False
        self._policy_effector_handoff_reported = False
        self._human_effector_enable_pending = False
        self._human_effector_control_reported = False
        # A transport-thread X-up can revoke the action-thread deadman before
        # the serialized DDS queue reaches that frame.  The latch prevents an
        # older queued X-down frame from granting it again.  Only a validated
        # later x_press edge may clear the latch.
        self._transport_x_lock = threading.Lock()
        self._transport_x_release_source_ns: int | None = None
        self._transport_x_release_key: tuple[int, int] | None = None
        self._transport_vr_latest_key: tuple[int, int] | None = None
        # Monotonic local receipt sequence, independent of the producer's frame
        # counter. Policy activation and rollback use this as a
        # post-persistence barrier so cached button/health values can never
        # authorize motion after a slow fsync.
        self.vr_frame_revision = self.parser.accepted_frame_revision
        self._policy_activation_pending = False
        self._policy_activation_barrier_revision: int | None = None

    def bind_initial_recording_in_hold(self, record_name: str) -> None:
        """Attach the tablet-triggering episode while the robot is in measured HOLD."""

        with self._input_lock:
            if self.session.supervisor.mode != SupervisorMode.DISABLED or self.coordinator.writer is not None:
                raise RuntimeError("A tablet episode is already active")
            active = self.coordinator.discover_active_recording(self.vendor_record_root)
            if active.name != record_name:
                raise RuntimeError(
                    f"Tablet start episode changed before HOLD bind: expected {record_name!r}, got {active.name!r}"
                )
            result = self._handle_input_locked(
                LogicalInput(
                    "flowpro_mode_toggle",
                    time.time_ns(),
                    {"source": "tablet_auto_start", "expected_record_name": record_name},
                )
            )
            if result is None or self.session.supervisor.mode != SupervisorMode.HOLD:
                raise RuntimeError("Tablet recording could not be bound in HOLD")

    def handle_vr_frame(
        self,
        frame: RawVRFrame,
        *,
        received_timestamp_ns: int | None = None,
    ) -> list[Any]:
        """Consume one audited raw frame; callers render returned status/ACKs."""

        previous_revision = self.parser.accepted_frame_revision
        events = self.parser.feed(
            frame,
            received_timestamp_ns=received_timestamp_ns,
        )
        # Duplicate/regressed producer frames do not constitute fresh operator
        # evidence and therefore cannot cross a post-fsync motion barrier.
        self.vr_frame_revision = self.parser.accepted_frame_revision
        # Publish the physical button levels before handling audit/state events.
        # In particular, an X-up must become visible to update_rollback without
        # waiting behind any sidecar operation triggered by the same frame.
        self.latest_x_pressed = self.parser.x_pressed
        self.latest_y_pressed = self.parser.y_pressed
        self.latest_side_takeover_pressed = self.parser.side_takeover_pressed
        self.latest_index_triggers_pressed = self.parser.index_triggers_pressed
        self._update_trigger_rearm(
            raw_index_levels=(float(frame.left_index), float(frame.right_index))
            if frame.connected and self.vr_frame_revision > previous_revision
            else None
        )
        self._refresh_vendor_effector_feedback()
        results = [self._handle_input(event) for event in events]
        self._maybe_grant_vendor_effector_control()
        self._maybe_activate_pending_policy_run()
        return results

    def observe_transport_vr_frame(self, frame: RawVRFrame) -> bool:
        """Fail-safe, revoke-only fast path from the DDS subscriber thread."""

        frame_key = (int(frame.source_timestamp_ns), int(frame.frame_index))
        lock = self._transport_lock()
        with lock:
            latest = getattr(self, "_transport_vr_latest_key", None)
            if latest is not None and (
                frame_key[0] < latest[0]
                or (frame_key[0] == latest[0] and frame_key[1] <= latest[1])
            ):
                # Match the authoritative parser's timestamp/frame ordering.
                # An old released frame must not revoke a newer validated X
                # hold and leave rollback apparently unresponsive.
                return False
            self._transport_vr_latest_key = frame_key
            if frame.connected and frame.x:
                return True
            self._transport_x_release_key = frame_key
            # Retain the scalar for old diagnostics; gating uses the complete
            # timestamp/frame key so equal-timestamp frames remain ordered.
            self._transport_x_release_source_ns = frame_key[0]
        revoke = getattr(
            self.robot_client_runner,
            "revoke_flowpro_rollback_deadman_from_transport",
            None,
        )
        if callable(revoke):
            revoke()
        return True

    def observe_transport_vr_sample(self, sample: Any) -> None:
        """Apply revoke-only fast paths before the serialized DDS audit copy.

        If Human presses a side button while a
        measured-HOLD handback is in flight, invalidate that handback epoch on
        the subscriber path so the action thread cannot install its stale
        released-input decision before the queued press is reconciled.
        """

        accepted = self.observe_transport_vr_frame(sample.raw_frame())
        if (
            accepted
            and self.session.supervisor.mode == SupervisorMode.HUMAN
            and not bool(sample.side_buttons_released)
        ):
            revoke = getattr(
                self.robot_client_runner,
                "revoke_flowpro_human_handback_from_transport",
                None,
            )
            if callable(revoke):
                revoke()

    def _transport_x_grant_allowed(self) -> bool:
        lock = self._transport_lock()
        with lock:
            return getattr(self, "_transport_x_release_source_ns", None) is None

    def _acknowledge_validated_x_press(self, logical_input: LogicalInput) -> None:
        source_ns = int(
            logical_input.detail.get("vr_source_timestamp_ns", logical_input.source_timestamp_ns)
        )
        frame_index = int(logical_input.detail.get("vr_frame_index", -1))
        press_key = (source_ns, frame_index)
        lock = self._transport_lock()
        with lock:
            release_key = getattr(self, "_transport_x_release_key", None)
            if release_key is None or press_key > release_key:
                self._transport_x_release_key = None
                self._transport_x_release_source_ns = None

    def _transport_lock(self) -> threading.Lock:
        """Return the transport latch lock (lazy for narrow test doubles)."""

        lock = getattr(self, "_transport_x_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._transport_x_lock = lock
            if not hasattr(self, "_transport_x_release_source_ns"):
                self._transport_x_release_source_ns = None
            if not hasattr(self, "_transport_x_release_key"):
                release_ns = self._transport_x_release_source_ns
                self._transport_x_release_key = (
                    None if release_ns is None else (int(release_ns), -1)
                )
            if not hasattr(self, "_transport_vr_latest_key"):
                self._transport_vr_latest_key = None
        return lock

    def advance_vr_clock(self, now_ns: int) -> list[Any]:
        events = self.parser.advance_received_clock(now_ns)
        self.latest_x_pressed = self.parser.x_pressed
        self.latest_y_pressed = self.parser.y_pressed
        self.latest_side_takeover_pressed = self.parser.side_takeover_pressed
        self.latest_index_triggers_pressed = self.parser.index_triggers_pressed
        self._refresh_vendor_effector_feedback()
        self._maybe_grant_vendor_effector_control()
        results = [self._handle_input(event) for event in events]
        self._maybe_activate_pending_policy_run()
        return results

    def _refresh_vendor_effector_feedback(self) -> None:
        """Continuously consume periodic config snapshots without changing ownership."""

        gate = getattr(self, "effector_gate", None)
        refresh = getattr(gate, "refresh_passive", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as error:
                # A pending ownership transition has its own fail-closed path.
                # Idle telemetry errors must not kill the VR coordinator.
                logger.error("FLOWPRO VENDOR EE PASSIVE FEEDBACK FAILED: %s", error)

    def human_takeover(self, vendor_timestamp_ns: int, *, detail: dict[str, Any] | None = None) -> Any:
        previous_context = self._gesture_context_key()
        self._human_policy_handoff_requested = False
        # DDS has already confirmed vendor Human authority. Suspend the local
        # writer before rollback cleanup or sidecar fsync; otherwise a live
        # rollback cancellation can be stranded when the action thread enters
        # its Human-only wait path.
        self.arbitration.invalidate_local_leases()
        self.robot_client_runner.begin_flowpro_human_control()
        result = self.paper.handle(LogicalInput("human_takeover", vendor_timestamp_ns, dict(detail or {})))
        if not (
            isinstance(result, SupervisorTransition)
            and result.current == SupervisorMode.HUMAN
        ):
            self._synchronize_robot_client(result)
        self._finish_policy_run_transition(result, timestamp_ns=vendor_timestamp_ns)
        self._maybe_grant_vendor_effector_control(timestamp_ns=vendor_timestamp_ns)
        self._advance_gesture_context_if_changed(previous_context)
        return result

    def external_authority_preempt(
        self,
        vendor_timestamp_ns: int,
        *,
        detail: dict[str, Any] | None = None,
    ) -> Any:
        """Fail closed when vendor authority changed without a side-button request.

        This is a safety event, not a Human-correction command. It can stop
        local Policy or rollback, but it never labels or grants Human control.
        """

        self.arbitration.invalidate_local_leases()
        self.robot_client_runner.begin_flowpro_human_control()
        logical_input = LogicalInput(
            "external_authority",
            vendor_timestamp_ns,
            dict(detail or {}),
        )
        previous_context = self._gesture_context_key()
        result = self.paper.handle(logical_input)
        self._synchronize_robot_client(result)
        self._finish_policy_run_transition(result, timestamp_ns=vendor_timestamp_ns)
        self._advance_gesture_context_if_changed(previous_context)
        return result

    def human_finish(
        self,
        vendor_timestamp_ns: int,
        *,
        evidence: HumanHandbackEvidence,
    ) -> Any:
        previous_context = self._gesture_context_key()
        result = self.paper.handle(
            LogicalInput("human_finish", vendor_timestamp_ns, {}),
            handback_evidence=evidence,
        )
        self._synchronize_robot_client(result)
        self._finish_policy_run_transition(result, timestamp_ns=vendor_timestamp_ns)
        self._advance_gesture_context_if_changed(previous_context)
        if (
            self._human_policy_handoff_requested
            and isinstance(result, SupervisorTransition)
            and result.current == SupervisorMode.HOLD
        ):
            self._human_policy_handoff_requested = False
            return self._handle_input(
                LogicalInput(
                    "y_press",
                    vendor_timestamp_ns,
                    {"source": "deferred_human_gripper_handoff"},
                )
            )
        return result

    def prepare_human_handback(self) -> bool:
        """Install an acknowledged measured HOLD before HUMAN may become HOLD."""

        return bool(self.robot_client_runner.prepare_flowpro_human_handback())

    def restart_human_control_hold(self) -> None:
        """Invalidate a provisional handback when Human input resumes."""

        self._human_policy_handoff_requested = False
        self.robot_client_runner.begin_flowpro_human_control()

    def update_rollback(self, health: RollbackHealth) -> Any:
        """Latest-only DDS health/deadman update; GDK send stays on the action thread."""

        previous_context = self._gesture_context_key()
        lease_id = self.paper.rollback_lease_id
        effective_x_pressed = bool(self.latest_x_pressed and self._transport_x_grant_allowed())
        rollback_deadman = bool(
            effective_x_pressed and self.session.supervisor.mode == SupervisorMode.ROLLBACK_EXEC
        )
        auto_finish_authorized = bool(
            not effective_x_pressed and self.session.supervisor.mode == SupervisorMode.ROLLBACK_EXEC
        )
        if lease_id is not None:
            self.arbitration.observe_rollback_lease(
                lease_id,
                timestamp_ns=time.time_ns(),
                deadman=rollback_deadman,
                auto_finish=auto_finish_authorized,
            )
        previous_mode = self.session.supervisor.mode
        result = self.paper.rollback_tick(
            x_pressed=effective_x_pressed,
            y_pressed=getattr(self, "latest_y_pressed", False),
            health=health,
            vr_frame_revision=getattr(self, "vr_frame_revision", None),
            now_ns=time.time_ns(),
        )
        if self.session.supervisor.mode != previous_mode:
            self._prepare_policy_run_transition(result, timestamp_ns=time.time_ns())
            self._synchronize_robot_client(result)
            self._finish_policy_run_transition(result, timestamp_ns=time.time_ns())
        # rollback_tick may have created a fresh physical lease. Register it
        # after _synchronize_robot_client invalidates the previous logical
        # lease, so the first action-thread ACK is not discarded as unknown.
        current_lease_id = self.paper.rollback_lease_id
        if current_lease_id is not None and current_lease_id != lease_id:
            self.arbitration.observe_rollback_lease(
                current_lease_id,
                timestamp_ns=time.time_ns(),
                deadman=bool(
                    effective_x_pressed and self.session.supervisor.mode == SupervisorMode.ROLLBACK_EXEC
                ),
                auto_finish=bool(
                    not effective_x_pressed and self.session.supervisor.mode == SupervisorMode.ROLLBACK_EXEC
                ),
            )
        self._advance_gesture_context_if_changed(previous_context)
        return result

    @property
    def rollback_progress(self) -> Any | None:
        """Latest in-memory progress for an optional rate-limited UI/log bridge."""

        return self.intervention.rollback_progress

    @property
    def rollback_status(self) -> Any:
        """Latest reason why rollback is moving, waiting, blocked, or stopped."""

        return self.paper.rollback_status

    def observe_tablet_completion(self, *, observed_at_ns: int | None = None) -> dict[str, Any]:
        return self.coordinator.observe_vendor_completion(observed_at_ns=observed_at_ns)

    def finalize_saved_episode(self) -> Path:
        return self.coordinator.finalize()

    def finish_from_tablet(self, *, timestamp_ns: int) -> Any:
        """End the bound FlowPRO session after an external tablet Save."""

        with self._input_lock:
            return self._finish_from_tablet_locked(timestamp_ns=timestamp_ns)

    def release_replaced_recording(self, *, next_record_name: str) -> Path | None:
        """Isolate an inactive clip only after the tablet starts a new one."""

        with self._input_lock:
            coordinator = self.coordinator
            if coordinator.writer is None:
                return None
            state = coordinator._read_state()
            if state["phase"] not in {
                coordinator.PHASE_ATTACHED, coordinator.PHASE_AWAITING_SAVE,
            }:
                return None
            if coordinator.attached_recording_finalized():
                return None
            if coordinator.attached_recording_active():
                return None
            active = coordinator.list_active_recordings(self.vendor_record_root)
            if len(active) != 1 or active[0].name != next_record_name:
                return None
            if active[0].name == state["record_name"]:
                return None
            mode = self.session.supervisor.mode
            if self.session.flowpro_session_id is not None:
                if mode != SupervisorMode.HOLD or state["phase"] != coordinator.PHASE_ATTACHED:
                    return None
                timestamp_ns = time.time_ns()
                previous_context = self._gesture_context_key()
                result = self.session.finish_for_replacement(
                    timestamp_ns=timestamp_ns, next_record_name=next_record_name,
                )
                self._synchronize_robot_client(result)
                self._finish_policy_run_transition(result, timestamp_ns=timestamp_ns)
                self._request_vendor_effector_control(timestamp_ns=timestamp_ns)
                self._advance_gesture_context_if_changed(previous_context)
            elif mode != SupervisorMode.DISABLED or state["phase"] != coordinator.PHASE_AWAITING_SAVE:
                return None
            return coordinator.detach_pending_recording()

    def _finish_from_tablet_locked(self, *, timestamp_ns: int) -> Any:
        if self.session.supervisor.mode != SupervisorMode.HOLD:
            raise RuntimeError("Tablet completion is waiting for FlowPRO HOLD")
        previous_context = self._gesture_context_key()
        result = self.session.finish_from_tablet(timestamp_ns=timestamp_ns)
        self._prepare_policy_run_transition(result, timestamp_ns=timestamp_ns)
        self._synchronize_robot_client(result)
        self._finish_policy_run_transition(result, timestamp_ns=timestamp_ns)
        self._request_vendor_effector_control(timestamp_ns=timestamp_ns)
        self._advance_gesture_context_if_changed(previous_context)
        logger.warning("FLOWPRO SESSION ENDED by final tablet recorder metadata")
        return result

    def _handle_input(self, logical_input: LogicalInput) -> Any:
        with self._input_lock:
            return self._handle_input_locked(logical_input)

    def _handle_input_locked(self, logical_input: LogicalInput) -> Any:
        previous_context = self._gesture_context_key()
        previous_mode = self.session.supervisor.mode
        if logical_input.name == "x_press":
            self._acknowledge_validated_x_press(logical_input)
        if logical_input.name == "vr_neutral_rearm_required":
            self.paper.note_vr_rearm_required(logical_input)
            return None
        if logical_input.name == "vr_input_rearmed":
            self.paper.note_vr_rearmed(logical_input)
            return None
        if self.session.supervisor.mode == SupervisorMode.DISABLED:
            if logical_input.name not in {"flowpro_mode_toggle", "y_press"}:
                if logical_input.name.startswith(("x_", "y_")):
                    status = getattr(self.paper, "rollback_status", None)
                    if getattr(status, "code", None) != "activation_blocked":
                        self.paper.note_input_blocked(
                            logical_input,
                            reason=(
                                "FlowPRO is disabled; start the active tablet recording "
                                "with Y first"
                            ),
                        )
                return None
            if self.coordinator.writer is not None:
                # The previous tablet episode has exited FlowPRO but has not
                # completed external save/seal yet. Check this before active
                # recorder discovery: after Save there is intentionally no
                # active placeholder, and that expected condition must not be
                # escalated into a fatal DDS error.
                self.paper.note_activation_blocked(
                    logical_input,
                    reason=(
                        "the previous FlowPRO episode is still awaiting tablet save/seal; "
                        "release X and Y fully, then retry after DATA SEALED is reported"
                    ),
                )
                logger.error("FLOWPRO ACTIVATION BLOCKED: previous episode is pending seal")
                return None
            try:
                active = self.coordinator.discover_active_recording(self.vendor_record_root)
                expected = logical_input.detail.get("expected_record_name")
                if expected is not None and active.name != expected:
                    raise RuntimeError("Tablet recording changed before HOLD bind")
            except (OSError, RuntimeError, ValueError) as error:
                # Recorder discovery is read-only and happens before any
                # session/sidecar mutation.  Transient filesystem failures are
                # therefore operator-retryable and must not kill the DDS input
                # service.
                if (
                    self.session.supervisor.mode != SupervisorMode.DISABLED
                    or self.session.flowpro_session_id is not None
                    or self.coordinator.writer is not None
                ):
                    raise
                self.paper.note_activation_blocked(
                    logical_input,
                    reason=(
                        f"{type(error).__name__}: {error}; release X/Y fully, then press Y to retry"
                    ),
                )
                logger.error("FLOWPRO ACTIVATION BLOCKED: %s", error)
                return None
            try:
                # History is scoped to exactly one tablet episode. Production has
                # no time-based rollback horizon, so an explicit boundary reset is
                # required before accepting the first observation of a new bind.
                self.history.clear()
                self.policy_run_scope = None
                result = self.session.start_session(
                    active,
                    LogicalInput(
                        logical_input.name,
                        logical_input.source_timestamp_ns,
                        {
                            **logical_input.detail,
                            "collision_safety_enabled": self.collision_safety_enabled,
                            "rollback_enabled": self.paper.rollback_enabled,
                        },
                    ),
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                # Missing/multiple tablet recordings and an already-open
                # sidecar are operator-correctable pre-bind conditions.  They
                # must not terminate the DDS thread and permanently kill every
                # later button.  Only contain the error when no partial session
                # was created; partial/durable-write failures still propagate.
                if (
                    self.session.supervisor.mode != SupervisorMode.DISABLED
                    or self.session.flowpro_session_id is not None
                    or self.coordinator.writer is not None
                ):
                    raise
                self.paper.note_activation_blocked(
                    logical_input,
                    reason=(
                        f"{type(error).__name__}: {error}; release X/Y fully, then press Y to retry"
                    ),
                )
                logger.error("FLOWPRO ACTIVATION BLOCKED: %s", error)
                return None
        else:
            if logical_input.name == "x_press" and hasattr(self, "vr_frame_revision"):
                # Bind rollback planning to this accepted VR frame. If the
                # measured-HOLD acknowledgement or sidecar fsync is slow, a
                # later accepted held-X frame must still cross this barrier
                # before history is frozen.
                logical_input = replace(
                    logical_input,
                    detail={
                        **logical_input.detail,
                        "accepted_vr_frame_revision": self.vr_frame_revision,
                    },
                )
            policy_start_input = logical_input.name in {"y_press", "y_short", "y_long_start"}
            bound_recording_active = True
            if (
                policy_start_input
                and self.session.supervisor.mode == SupervisorMode.HOLD
                and self.coordinator.writer is not None
            ):
                try:
                    bound_recording_active = self.coordinator.attached_recording_active()
                except (OSError, RuntimeError, ValueError):
                    bound_recording_active = False
            if (
                policy_start_input
                and self.session.supervisor.mode == SupervisorMode.HOLD
                and self.coordinator.writer is not None
                and not bound_recording_active
            ):
                self.paper.note_input_blocked(
                    logical_input,
                    reason="The bound tablet recording is no longer active; Policy requires a live recording",
                )
                return None
            has_live_button_levels = hasattr(self, "latest_y_pressed")
            x_held = bool(getattr(self, "latest_x_pressed", False))
            side_held = bool(getattr(self, "latest_side_takeover_pressed", False))
            long_y_released = bool(
                logical_input.name == "y_long_start" and has_live_button_levels and not self.latest_y_pressed
            )
            if policy_start_input and has_live_button_levels and (x_held or side_held or long_y_released):
                # A sparse Y-up frame can classify a long-start immediately
                # before the release from that same frame. Likewise, a delayed
                # click/long timer can mature on a frame that already carries X
                # or vendor takeover. Never start Policy against those current
                # physical levels.
                blockers = []
                if x_held:
                    blockers.append("X is held")
                if side_held:
                    blockers.append("a Human side button is held")
                if long_y_released:
                    blockers.append("Y is already released")
                self.paper.note_input_blocked(
                    logical_input,
                    reason=f"Policy start blocked because {'; '.join(blockers)}",
                )
                return None
            if policy_start_input and self.session.supervisor.mode == SupervisorMode.HUMAN:
                # Y is allowed immediately after the side buttons are released,
                # even while vendor arbitration is still completing its
                # roughly one-second TAKEOVER -> IDLE timeout.  Preserve the
                # intent and replay it exactly once after measured-HOLD
                # handback; Policy motion is still gated by vendor EE disable.
                self._human_policy_handoff_requested = True
                logger.warning(
                    "FLOWPRO POLICY HANDOFF QUEUED: waiting for Human arm authority to clear"
                )
                return None
            result = self.paper.handle(logical_input)
        # A successful Y transition has already durably created its generation,
        # but the robot client is still paused here.  Clear the previous run in
        # this interval so no action-thread sample from the new run can race
        # ahead of the boundary.
        self._prepare_policy_run_transition(
            result,
            timestamp_ns=logical_input.source_timestamp_ns,
        )
        self._synchronize_robot_client(result)
        self._finish_policy_run_transition(
            result,
            timestamp_ns=logical_input.source_timestamp_ns,
        )
        self._advance_gesture_context_if_changed(previous_context)
        logger.warning(
            "FLOWPRO INPUT name=%s mode=%s->%s result=%s",
            logical_input.name,
            previous_mode,
            self.session.supervisor.mode,
            type(result).__name__ if result is not None else "none",
        )
        return result

    def _prepare_policy_run_transition(self, result: Any, *, timestamp_ns: int) -> None:
        """Open a new Y-scoped history window before Policy is resumed."""

        if not isinstance(result, SupervisorTransition):
            return
        if result.current not in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            return
        if result.previous not in {SupervisorMode.DISABLED, SupervisorMode.HOLD}:
            return
        if result.generation_id is None:
            raise RuntimeError("Policy run transition lacks a generation identifier")
        # generation_id is deliberately reused as policy_run_id.  A second UUID
        # would create two competing lineage systems for the same Y operation.
        self.history.clear()
        self.policy_run_scope = PolicyRunScope(
            generation_id=result.generation_id,
            started_timestamp_ns=timestamp_ns,
        )
        if hasattr(self, "vr_frame_revision"):
            # Both AUTO and hold-to-run remain physically in HOLD until a
            # post-persistence frame proves the button state is still
            # compatible with the operator's request. This closes the window
            # in which an X/vendor frame can arrive while the Y generation is
            # blocked in sidecar fsync but not be consumed until after resume.
            self._policy_activation_pending = True
            self._policy_activation_barrier_revision = self.vr_frame_revision
            self.paper.note_policy_activation_waiting(
                timestamp_ns=timestamp_ns,
                mode=result.current,
            )
        elif getattr(self, "effector_gate", None) is not None:
            # Production always has the VR revision barrier. Keep narrow
            # embedders fail-closed as well if they opt into the physical EE
            # gate without providing that barrier.
            self._policy_activation_pending = True
        self._begin_policy_effector_handoff(timestamp_ns=timestamp_ns)
        logger.warning(
            "FLOWPRO POLICY RUN REQUESTED generation_id=%s timestamp_ns=%d",
            result.generation_id,
            timestamp_ns,
        )

    def _finish_policy_run_transition(self, result: Any, *, timestamp_ns: int) -> None:
        """Retain a stopped Y run for X, or discard it at a true boundary."""

        if not isinstance(result, SupervisorTransition):
            return
        if result.current not in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            self._policy_activation_pending = False
            self._policy_activation_barrier_revision = None
            self._policy_effector_handoff_pending = False
        scope = getattr(self, "policy_run_scope", None)
        if result.current == SupervisorMode.HOLD and result.previous in {
            SupervisorMode.AUTO,
            SupervisorMode.FORWARD_RUN,
        }:
            if scope is not None and scope.active:
                self.policy_run_scope = replace(
                    scope,
                    stopped_timestamp_ns=timestamp_ns,
                    stop_reason=result.reason,
                )
                logger.warning(
                    "FLOWPRO POLICY RUN RETAIN generation_id=%s reason=%s timestamp_ns=%d",
                    scope.generation_id,
                    result.reason,
                    timestamp_ns,
                )
            return
        if result.current in {
            SupervisorMode.DISABLED,
            SupervisorMode.HUMAN,
            SupervisorMode.ABORT,
        }:
            # Human ownership, episode exit, and abort are real scene/control
            # boundaries.  A frozen intervention already owns its immutable
            # snapshot; live history must not leak into a later Policy run.
            self.history.clear()
            self.policy_run_scope = None

    def _synchronize_robot_client(self, result: Any) -> None:
        if not isinstance(result, SupervisorTransition):
            return
        if result.current == SupervisorMode.HUMAN:
            self.arbitration.invalidate_local_leases()
            self.robot_client_runner.begin_flowpro_human_control()
            return
        if result.current in {
            SupervisorMode.DISABLED,
            SupervisorMode.HOLD,
            SupervisorMode.ABORT,
        }:
            self.arbitration.invalidate_local_leases()
            self.robot_client_runner.enter_flowpro_hold()
            return
        if result.current in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            if result.generation_id is None:
                raise RuntimeError("Policy transition lacks a generation lease")
            self.arbitration.invalidate_local_leases()
            if getattr(self, "_policy_activation_pending", False):
                # The current frame only proved the button state before the
                # durable generation write. A later accepted frame performs
                # the actual physical resume for AUTO or hold-to-run.
                return
            self.arbitration.observe_policy_lease(
                result.generation_id,
                timestamp_ns=time.time_ns(),
            )
            try:
                self.robot_client_runner.resume_flowpro_policy(wait_for_ready=False)
            except Exception:
                # Resume failure must not leave the logical supervisor in a
                # running state while the client has no acknowledged plan.
                self.robot_client_runner.enter_flowpro_hold()
                self.session.fail_closed_hold(reason="watchdog", timestamp_ns=time.time_ns())
                scope = getattr(self, "policy_run_scope", None)
                if scope is not None and scope.active:
                    self.policy_run_scope = replace(
                        scope,
                        stopped_timestamp_ns=time.time_ns(),
                        stop_reason="watchdog",
                    )
                raise
            return
        if result.current == SupervisorMode.ROLLBACK_EXEC:
            # This is only the durable logical intent. The potentially blocking
            # path prepare remains invisible to the action thread; a later
            # fresh held-X frame commits the physical mailbox lease, which is
            # ACKed only by the sole GDK action thread.
            self.arbitration.invalidate_local_leases()

    def _maybe_activate_pending_policy_run(self) -> None:
        """Commit Policy only from a compatible post-fsync VR frame."""

        if not getattr(self, "_policy_activation_pending", False):
            return
        mode = self.session.supervisor.mode
        if mode not in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            self._policy_activation_pending = False
            self._policy_activation_barrier_revision = None
            return
        barrier = self._policy_activation_barrier_revision
        revision = getattr(self, "vr_frame_revision", None)
        if revision is None or barrier is None or revision <= barrier:
            return
        y_level_matches = self.latest_y_pressed == (mode == SupervisorMode.FORWARD_RUN)
        if (
            not y_level_matches
            or self.latest_x_pressed
            or getattr(self, "latest_side_takeover_pressed", False)
        ):
            expected = "only Y pressed" if mode == SupervisorMode.FORWARD_RUN else "all controls released"
            self.paper.note_input_blocked(
                LogicalInput(
                    "y_long_start" if mode == SupervisorMode.FORWARD_RUN else "y_press",
                    time.time_ns(),
                    {},
                ),
                reason=f"Policy remains stationary until a fresh frame has {expected}",
            )
            return
        if not self._policy_effector_handoff_ready():
            return
        scope = self.policy_run_scope
        if scope is None or not scope.active:
            raise RuntimeError("Pending Policy activation lost its generation")
        self.arbitration.invalidate_local_leases()
        try:
            self.robot_client_runner.resume_flowpro_policy(wait_for_ready=False)
        except Exception:
            self.robot_client_runner.enter_flowpro_hold()
            self.session.fail_closed_hold(reason="watchdog", timestamp_ns=time.time_ns())
            self.policy_run_scope = replace(
                scope,
                stopped_timestamp_ns=time.time_ns(),
                stop_reason="watchdog",
            )
            self._policy_activation_pending = False
            self._policy_activation_barrier_revision = None
            raise
        self.arbitration.observe_policy_lease(scope.generation_id, timestamp_ns=time.time_ns())
        self._policy_activation_pending = False
        self._policy_activation_barrier_revision = None
        self.paper.note_policy_activation_active(timestamp_ns=time.time_ns(), mode=mode)

    def _update_trigger_rearm(
        self,
        *,
        raw_index_levels: tuple[float, float] | None,
    ) -> None:
        """Consume a handoff trigger only after a fresh physical release.

        A timeout, disconnect, parser neutral-rearm, duplicate, or regressed
        frame is not release evidence. Treating any of those as an upper-
        trigger release could re-enable vendor EE while the operator still has
        the transfer trigger held.
        """

        if raw_index_levels is None:
            return
        masks = getattr(self, "_trigger_mask_until_release", [False, False])
        threshold = float(getattr(getattr(self, "parser", None), "index_release_threshold", 0.2))
        for side in range(2):
            if masks[side] and raw_index_levels[side] <= threshold:
                masks[side] = False

    def _begin_policy_effector_handoff(self, *, timestamp_ns: int) -> None:
        gate = getattr(self, "effector_gate", None)
        if gate is None:
            self._policy_effector_handoff_pending = False
            return
        pressed = getattr(self, "latest_index_triggers_pressed", (False, False))
        masks = getattr(self, "_trigger_mask_until_release", [False, False])
        for side in range(2):
            masks[side] = bool(masks[side] or pressed[side])
        self._policy_effector_handoff_pending = True
        self._policy_effector_handoff_reported = False
        self._human_effector_enable_pending = False
        self._human_effector_control_reported = False
        try:
            begin_handoff = getattr(gate, "begin_handoff", gate.request)
            status = begin_handoff(False, timestamp_ns=timestamp_ns)
        except Exception as error:
            self._fail_policy_effector_handoff(f"vendor EE disable request failed: {error}")
            return
        logger.warning(
            "FLOWPRO GRIPPER HANDOFF REQUESTED: vendor_ee_current=%s trigger_mask=%s",
            status.current_enabled,
            tuple(masks),
        )

    def _policy_effector_handoff_ready(self) -> bool:
        if not getattr(self, "_policy_effector_handoff_pending", False):
            return True
        gate = getattr(self, "effector_gate", None)
        if gate is None:
            self._policy_effector_handoff_pending = False
            return True
        try:
            status = gate.poll()
        except Exception as error:
            self._fail_policy_effector_handoff(f"vendor EE feedback failed: {error}")
            return False
        if status.confirmed and status.current_enabled is False:
            self._policy_effector_handoff_pending = False
            if not self._policy_effector_handoff_reported:
                logger.warning(
                    "FLOWPRO GRIPPER HANDOFF CONFIRMED: vendor ee.enable=false; "
                    "Policy owns both grippers"
                )
                self._policy_effector_handoff_reported = True
            return True
        if status.timed_out:
            self._fail_policy_effector_handoff(
                "vendor ee.enable=false was not confirmed before timeout"
            )
        return False

    def _fail_policy_effector_handoff(self, reason: str) -> None:
        logger.error("FLOWPRO POLICY START BLOCKED: %s", reason)
        self._policy_effector_handoff_pending = False
        # Keep the pending latch set through the caller's synchronization of
        # the original AUTO transition. The logical supervisor is already put
        # back in HOLD below; the next coordinator cycle clears the latch.
        # Clearing it here would let _synchronize_robot_client resume from the
        # stale AUTO transition that initiated this failed ownership request.
        self._policy_activation_pending = True
        self.robot_client_runner.enter_flowpro_hold()
        self.session.fail_closed_hold(reason="watchdog", timestamp_ns=time.time_ns())
        scope = getattr(self, "policy_run_scope", None)
        if scope is not None and scope.active:
            self.policy_run_scope = replace(
                scope,
                stopped_timestamp_ns=time.time_ns(),
                stop_reason="vendor_ee_handoff_failed",
            )

    def _maybe_grant_vendor_effector_control(self, *, timestamp_ns: int | None = None) -> None:
        """Enable vendor EE only after a new, unmasked Human trigger press."""

        gate = getattr(self, "effector_gate", None)
        session = getattr(self, "session", None)
        supervisor = getattr(session, "supervisor", None)
        if gate is None or getattr(supervisor, "mode", None) != SupervisorMode.HUMAN:
            return
        pressed = getattr(self, "latest_index_triggers_pressed", (False, False))
        masks = getattr(self, "_trigger_mask_until_release", [False, False])
        eligible_trigger_held = any(
            bool(pressed[side]) and not masks[side] for side in range(2)
        )
        if not eligible_trigger_held:
            self._human_effector_control_reported = False
            if getattr(self, "_human_effector_enable_pending", False):
                # The new Human trigger was released before vendor enable was
                # acknowledged. Revoke the delayed grant so it cannot arrive
                # after release and turn that release into an open command.
                try:
                    begin_handoff = getattr(gate, "begin_handoff", gate.request)
                    begin_handoff(False, timestamp_ns=timestamp_ns)
                    logger.warning(
                        "FLOWPRO HUMAN GRIPPER ENABLE CANCELLED: trigger released before ACK"
                    )
                except Exception:
                    logger.exception("Failed to cancel an unacknowledged vendor EE enable")
                self._human_effector_enable_pending = False
            return
        try:
            status = gate.request(True, timestamp_ns=timestamp_ns)
            if status.confirmed:
                self._human_effector_enable_pending = False
                if not getattr(self, "_human_effector_control_reported", False):
                    logger.warning(
                        "FLOWPRO HUMAN GRIPPER CONTROL CONFIRMED: vendor ee.enable=true"
                    )
                    self._human_effector_control_reported = True
            else:
                self._human_effector_enable_pending = True
        except Exception as error:
            # Human arm takeover must remain available as the highest-priority
            # preemption even if gripper ownership cannot be granted.
            logger.error("FLOWPRO HUMAN GRIPPER CONTROL BLOCKED: %s", error)

    def _request_vendor_effector_control(self, *, timestamp_ns: int) -> None:
        gate = getattr(self, "effector_gate", None)
        if gate is None:
            return
        try:
            status = gate.request(True, timestamp_ns=timestamp_ns)
            wait_confirmed = getattr(gate, "wait_confirmed", None)
            if not status.confirmed and callable(wait_confirmed):
                # This path runs in the tablet sealer thread after Policy has
                # reached HOLD. Confirm normal teleop restoration before the
                # completed FlowPRO session is reported as cleanly exited.
                wait_confirmed(True)
            logger.warning("FLOWPRO VENDOR GRIPPER CONTROL RESTORED at session end")
        except Exception:
            logger.exception("Failed to request vendor EE restore at FlowPRO session end")

    def _gesture_context_key(self) -> tuple[Any, Any]:
        intervention = getattr(getattr(self, "paper", None), "intervention", None)
        return (
            getattr(getattr(self.session, "supervisor", None), "mode", None),
            getattr(intervention, "state", None),
        )

    def _advance_gesture_context_if_changed(self, previous: tuple[Any, Any]) -> None:
        parser = getattr(self, "parser", None)
        if parser is not None and self._gesture_context_key() != previous:
            parser.advance_gesture_context()
