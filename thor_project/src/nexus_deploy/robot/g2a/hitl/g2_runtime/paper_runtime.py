"""VR-facing orchestration for continuous-X rollback on one immutable plan."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .intervention import (
    InterventionState,
    PaperInterventionController,
    SceneRestorationSnapshot,
)
from .rollback import RollbackExecutorState, RollbackHealth, RollbackTrajectory
from .runtime import FlowPROSessionController
from .supervisor import HumanHandbackEvidence, LogicalInput, SupervisorMode

logger = logging.getLogger(__name__)

_ROLLBACK_STATES = {
    InterventionState.ROLLBACK_PLANNING,
    InterventionState.ROLLBACK_READY,
    InterventionState.ROLLBACK_EXEC,
    InterventionState.ROLLBACK_PAUSED,
}
_ABANDONABLE_HOLD_INTERVENTION_STATES = _ROLLBACK_STATES | {
    InterventionState.SCENE_RESTORE,
    InterventionState.CORRECTION_READY,
}


@dataclass(frozen=True)
class RollbackPlanningFailed:
    reason: str


@dataclass(frozen=True)
class RollbackRuntimeStatus:
    """Latest operator-facing explanation of the rollback control path."""

    code: str
    detail: str
    timestamp_ns: int
    motion_expected: bool


class FlowPROPaperRuntime:
    """Stop Policy on X-down and rewind one immutable plan toward older anchors."""

    def __init__(
        self,
        *,
        session: FlowPROSessionController,
        intervention: PaperInterventionController,
        rollback_executor: Any,
        rollback_planner: Callable[[Any, str, Any], RollbackTrajectory],
        measured_state_provider: Callable[[], Any] | None = None,
        policy_run_generation_provider: Callable[[], str | None] | None = None,
        rollback_enabled: bool = True,
        max_rollback_ns: int | None = None,
    ) -> None:
        if max_rollback_ns is not None and max_rollback_ns <= 0:
            raise ValueError("max_rollback_ns must be positive")
        self.session = session
        self.intervention = intervention
        self.rollback_executor = rollback_executor
        self.rollback_planner = rollback_planner
        self.measured_state_provider = measured_state_provider
        self.policy_run_generation_provider = policy_run_generation_provider
        self.rollback_enabled = bool(rollback_enabled)
        self.max_rollback_ns = None if max_rollback_ns is None else int(max_rollback_ns)
        self.failure_timestamp_ns: int | None = None
        self.rollback_lease_id: str | None = None
        self._planning_requested = False
        self._planning_not_before_ns = 0
        self._planning_input_barrier_revision: int | None = None
        self._execution_requested = False
        self._planning_lock = threading.Lock()
        self._planning_token: str | None = None
        self._planning_result: RollbackTrajectory | None = None
        self._planning_error: BaseException | None = None
        self._executor_armed = False
        self._health_wait_blockers: tuple[str, ...] | None = None
        self._motion_input_barrier_revision: int | None = None
        self._activation_pending = False
        self._activation_resume = False
        self._activation_lease_id: str | None = None
        self._activation_barrier_revision: int | None = None
        self._activation_prepared = False
        self._activation_resume_detail: dict[str, float | int] | None = None
        self._deferred_logical_inputs: list[tuple[LogicalInput, str]] = []
        # Accumulated automatically from left/right VR hand-trigger edges for
        # the current intervention. It is persisted with correction completion
        # and later cross-checked against measured robot motion offline.
        self._controller_active_arms = [False, False]
        self._rollback_status = RollbackRuntimeStatus(
            code="idle",
            detail="rollback is idle",
            timestamp_ns=0,
            motion_expected=False,
        )

    @property
    def planning_requested(self) -> bool:
        return self._planning_requested

    @property
    def rollback_status(self) -> RollbackRuntimeStatus:
        return self._rollback_status

    def note_input_blocked(self, logical_input: LogicalInput, *, reason: str) -> None:
        """Expose an input gate owned by the application rather than this state machine."""

        self._set_status(
            "input_blocked",
            logical_input.source_timestamp_ns,
            f"{logical_input.name}: {reason}",
            level=logging.WARNING,
        )

    def note_activation_blocked(self, logical_input: LogicalInput, *, reason: str) -> None:
        """Keep a correctable tablet/session bind failure visible and retryable."""

        self._set_status(
            "activation_blocked",
            logical_input.source_timestamp_ns,
            reason,
            level=logging.ERROR,
        )

    def note_policy_activation_waiting(
        self,
        *,
        timestamp_ns: int,
        mode: SupervisorMode,
    ) -> None:
        """Expose the post-fsync fresh-frame barrier for Policy motion."""

        if mode == SupervisorMode.AUTO:
            detail = "AUTO generation is durable; waiting for a newer neutral VR frame"
        elif mode == SupervisorMode.FORWARD_RUN:
            detail = "Y hold is durable; waiting for a newer VR frame that still has only Y pressed"
        else:
            raise ValueError(f"Policy activation cannot wait in mode {mode.value}")

        self._set_status(
            "policy_activation_waiting",
            timestamp_ns,
            detail,
        )

    def note_policy_activation_active(
        self,
        *,
        timestamp_ns: int,
        mode: SupervisorMode,
    ) -> None:
        if mode == SupervisorMode.AUTO:
            detail = "Fresh neutral VR frame accepted; Policy AUTO is active"
        elif mode == SupervisorMode.FORWARD_RUN:
            detail = "Fresh only-Y frame accepted; Policy hold-to-run is active"
        else:
            raise ValueError(f"Policy activation cannot become active in mode {mode.value}")
        self._set_status(
            "policy_running",
            timestamp_ns,
            detail,
            motion_expected=True,
        )

    def note_vr_rearm_required(self, logical_input: LogicalInput) -> None:
        self._set_status(
            "vr_neutral_rearm_required",
            logical_input.source_timestamp_ns,
            "VR input recovered but X/Y are still held; release both buttons once to rearm",
            level=logging.WARNING,
        )

    def note_vr_rearmed(self, logical_input: LogicalInput) -> None:
        self._set_status(
            "vr_input_rearmed",
            logical_input.source_timestamp_ns,
            "Neutral X/Y frame received; buttons are armed again",
        )

    def handle(
        self,
        logical_input: LogicalInput,
        *,
        handback_evidence: HumanHandbackEvidence | None = None,
    ) -> Any:
        name = logical_input.name
        paper_state = self.intervention.state
        if name == "side_takeover_input" and paper_state not in {
            InterventionState.IDLE,
            InterventionState.COMPLETE,
            InterventionState.REJECTED,
        }:
            self._controller_active_arms[0] |= logical_input.detail.get("left_side") is True
            self._controller_active_arms[1] |= logical_input.detail.get("right_side") is True
        if (
            name == "y_press"
            and self.session.supervisor.mode in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}
        ) or (name == "y_long_end" and self.session.supervisor.mode == SupervisorMode.FORWARD_RUN):
            # Deadman release is a physical stop request.  Stop the sole GDK
            # writer before the sidecar append/fsync in session.handle; the
            # application-level synchronization repeats HOLD idempotently once
            # the durable state transition completes.
            enter_hold = getattr(self.rollback_executor, "enter_hold", None)
            if callable(enter_hold):
                enter_hold()
            return self.session.handle(logical_input, handback_evidence=handback_evidence)
        if name in {"human_takeover", "external_authority"} and self.session.supervisor.mode in {
            SupervisorMode.AUTO,
            SupervisorMode.FORWARD_RUN,
        }:
            # The application suspends the local writer before delivering
            # authoritative DDS ownership. Re-entering local HOLD here would
            # undo that suspension and briefly compete with vendor Human
            # control. Persist only the logical transition on this path.
            return self.session.handle(logical_input, handback_evidence=handback_evidence)
        if name in {
            "side_takeover_input",
            "emergency",
            "fatal_fault",
            "vr_stale",
            "watchdog",
            "network_lost",
            "motion_unhealthy",
        } and self.session.supervisor.mode in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            # Safety/authority loss is also a physical stop request.  Persist
            # its audit record only after the sole writer has acknowledged HOLD.
            enter_hold = getattr(self.rollback_executor, "enter_hold", None)
            if callable(enter_hold):
                enter_hold()
            return self.session.handle(logical_input, handback_evidence=handback_evidence)
        if name == "x_press" and paper_state == InterventionState.ROLLBACK_EXEC:
            # During X-up auto-finish this changes the 100 Hz executor intent
            # back to manual rewind on its next mailbox update.  It also
            # preserves the edge if the action thread reached anchor HOLD just
            # before the slower coordinator consumed that acknowledgement.
            self._execution_requested = True
            self._record_logical_input(
                logical_input,
                # SupervisorMode is an internal UI/control state; the durable
                # event contract deliberately uses the authority name
                # ``rollback``.  Persisting ``rollback_exec`` here used to
                # fail only when the deferred input was flushed at an anchor.
                control_owner="rollback",
                defer=self.rollback_lease_id is not None,
            )
            self._set_status(
                "manual_rewind_requested" if not self._activation_pending else "activation_waiting",
                logical_input.source_timestamp_ns,
                (
                    "X is held; continuing the immutable rollback plan"
                    if not self._activation_pending
                    else "X is held; waiting for a fresh post-persistence VR frame"
                ),
                motion_expected=not self._activation_pending,
            )
            return None
        rollback_from_running_policy = self.session.supervisor.mode in {
            SupervisorMode.AUTO,
            SupervisorMode.FORWARD_RUN,
        }
        terminal_retry_without_replay = bool(
            paper_state == InterventionState.REJECTED
            and int(getattr(self.intervention, "rollback_cursor", 0)) == 0
        )
        current_policy_run_generation = (
            None if self.policy_run_generation_provider is None else self.policy_run_generation_provider()
        )
        prior_segment = getattr(self.intervention, "segment", None)
        prior_policy_generations = tuple(getattr(prior_segment, "generation_ids", ()))
        new_policy_run_available = bool(
            current_policy_run_generation is not None
            and prior_policy_generations
            and current_policy_run_generation not in prior_policy_generations
        )
        rollback_from_hold = self.session.supervisor.mode == SupervisorMode.HOLD and (
            paper_state == InterventionState.IDLE
            or terminal_retry_without_replay
            or (
                paper_state in {InterventionState.COMPLETE, InterventionState.REJECTED}
                and new_policy_run_available
            )
        )
        if (
            name == "x_press"
            and self.session.supervisor.mode == SupervisorMode.HOLD
            and paper_state in {InterventionState.COMPLETE, InterventionState.REJECTED}
            and not rollback_from_hold
        ):
            self.session.coordinator.append_logical_input(
                LogicalInput(
                    "rollback_replay_blocked",
                    logical_input.source_timestamp_ns,
                    {
                        "requested_event": name,
                        "intervention_state": paper_state.value,
                        "rollback_cursor": int(getattr(self.intervention, "rollback_cursor", 0)),
                        "reason": "no_new_policy_generation_since_terminal_intervention",
                    },
                ),
                control_owner=SupervisorMode.HOLD.value,
            )
            self._set_status(
                "rollback_replay_blocked",
                logical_input.source_timestamp_ns,
                "The previous rollback cannot be replayed; run a new Policy generation first",
                level=logging.WARNING,
            )
            return None
        if name == "x_press" and (rollback_from_running_policy or rollback_from_hold):
            self._controller_active_arms = [False, False]
            # In the normal hold-to-run workflow, Y-up has already stopped the
            # Policy before the operator asks for rollback.  The rejected
            # trajectory ends at the last physically accepted Policy command,
            # not at a later X wall-clock timestamp spent motionless in HOLD.
            # Starting from a running Policy still uses the X-down timestamp so
            # that the immediate stop is part of the same atomic transition.
            if rollback_from_hold:
                samples = self.intervention.history.samples
                self.failure_timestamp_ns = (
                    samples[-1].timestamp_ns if samples else logical_input.source_timestamp_ns
                )
            else:
                self.failure_timestamp_ns = logical_input.source_timestamp_ns
            if rollback_from_running_policy:
                # Stop the sole action writer before sidecar fsync. The normal
                # application synchronization repeats this idempotently after
                # the logical transition, but a slow disk can no longer leave
                # Policy moving after the operator has pressed X.
                enter_hold = getattr(self.rollback_executor, "enter_hold", None)
                if callable(enter_hold):
                    enter_hold()
            transition = self.session.handle(logical_input)
            if rollback_from_running_policy:
                if transition is None or transition.current != SupervisorMode.HOLD:
                    raise RuntimeError("Immediate X press did not stop Policy in HOLD")
            elif transition is not None or self.session.supervisor.mode != SupervisorMode.HOLD:
                raise RuntimeError("HOLD rollback request changed the supervisor unexpectedly")
            if not self.rollback_enabled:
                self._record_unavailable(logical_input)
                self.failure_timestamp_ns = None
                self._planning_input_barrier_revision = None
            else:
                # Defer snapshot I/O until the application has acknowledged
                # measured HOLD. X has no chord ambiguity: a later accepted
                # held-X frame may start planning immediately.
                self._planning_requested = True
                self._execution_requested = True
                self._planning_not_before_ns = logical_input.source_timestamp_ns
                revision = logical_input.detail.get("accepted_vr_frame_revision")
                self._planning_input_barrier_revision = (
                    int(revision)
                    if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0
                    else None
                )
                self._set_status(
                    "waiting_post_x_frame",
                    logical_input.source_timestamp_ns,
                    "Policy is stopped; keep holding X until a fresh VR frame starts rollback planning",
                    motion_expected=False,
                )
            return transition

        if name == "x_press" and paper_state in {
            InterventionState.ROLLBACK_PLANNING,
            InterventionState.ROLLBACK_READY,
            InterventionState.ROLLBACK_PAUSED,
        }:
            if (
                paper_state == InterventionState.ROLLBACK_PAUSED
                and getattr(self.intervention, "anchor_selected_for_restoration", False)
                and not getattr(self.intervention, "can_continue_selected_rollback", True)
            ):
                self.session.coordinator.append_logical_input(
                    logical_input, control_owner=SupervisorMode.HOLD.value
                )
                self._set_status(
                    "oldest_anchor_reached",
                    logical_input.source_timestamp_ns,
                    "The immutable rollback plan is already at its oldest inference anchor",
                    level=logging.WARNING,
                )
                return None
            self._execution_requested = True
            self.session.coordinator.append_logical_input(
                logical_input, control_owner=SupervisorMode.HOLD.value
            )
            self._set_status(
                "rollback_resume_requested",
                logical_input.source_timestamp_ns,
                "X is held; waiting for rollback health and a fresh execution lease",
                motion_expected=False,
            )
            # Physical arm/resume is performed by rollback_tick, which carries
            # a fresh health sample and the current X level. This avoids
            # publishing a lease with no executable intent and then leaving a
            # 20--50 ms GDK heartbeat gap before the next DDS tick.
            return None

        if name == "x_release":
            self._execution_requested = False
            self._health_wait_blockers = None
            if self._planning_requested:
                # A tap still stops Policy immediately, but no history is
                # frozen if X was released before the fresh-frame barrier.
                self._planning_requested = False
                self.failure_timestamp_ns = None
                self._planning_input_barrier_revision = None
                self._set_status(
                    "released_before_planning",
                    logical_input.source_timestamp_ns,
                    "X was released before rollback history was frozen; Policy remains in HOLD",
                    level=logging.WARNING,
                )
            elif paper_state == InterventionState.ROLLBACK_EXEC and not self._activation_pending:
                self._set_status(
                    "finishing_to_anchor",
                    logical_input.source_timestamp_ns,
                    "X was released; following the same plan to the next older inference anchor",
                    motion_expected=True,
                )
            elif self._activation_pending:
                self._set_status(
                    "awaiting_x_press",
                    logical_input.source_timestamp_ns,
                    "Rollback is prepared but X was released before physical activation; hold X again to start",
                    level=logging.WARNING,
                )
            self._record_logical_input(
                logical_input,
                control_owner=self.session._control_owner(),
                defer=self.rollback_lease_id is not None,
            )
            return None

        # A side-button edge is an intent/audit event. It immediately preempts
        # Policy through the supervisor, but it does not cancel a rollback plan.
        # Only subsequent DDS authority evidence may grant Human control.
        if name == "side_takeover_input" and paper_state in _ROLLBACK_STATES | {
            InterventionState.SCENE_RESTORE,
            InterventionState.CORRECTION_READY,
        }:
            self._record_logical_input(
                logical_input,
                control_owner=self.session._control_owner(),
                defer=self.rollback_lease_id is not None,
            )
            if paper_state in {InterventionState.ROLLBACK_PAUSED, InterventionState.SCENE_RESTORE}:
                self._set_status(
                    "awaiting_human_authority",
                    logical_input.source_timestamp_ns,
                    "Side button pressed at the selected anchor; waiting for vendor Human authority",
                )
            return None

        if name == "human_takeover" and paper_state in {
            InterventionState.ROLLBACK_PAUSED,
            InterventionState.SCENE_RESTORE,
            InterventionState.CORRECTION_READY,
        }:
            if paper_state == InterventionState.ROLLBACK_PAUSED:
                self.intervention.finalize_selected_anchor(
                    completed_timestamp_ns=logical_input.source_timestamp_ns
                )
                if self._executor_armed:
                    self.rollback_executor.finish_paused()
                    self._executor_armed = False
            if self.intervention.state == InterventionState.SCENE_RESTORE:
                self.intervention.verify_scene(
                    valid=True,
                    verified_timestamp_ns=logical_input.source_timestamp_ns,
                    verification_method="implicit_human_takeover",
                )
            self.intervention.start_correction(start_timestamp_ns=logical_input.source_timestamp_ns)
            return self.session.handle(logical_input)

        if name == "human_takeover" and paper_state in _ROLLBACK_STATES:
            self._invalidate_rollback(name, logical_input.source_timestamp_ns)
            return self.session.handle(logical_input)

        if name == "external_authority" and paper_state in _ABANDONABLE_HOLD_INTERVENTION_STATES:
            self._invalidate_rollback(name, logical_input.source_timestamp_ns)
            return self.session.handle(logical_input)

        if (
            name
            in {
                "emergency",
                "fatal_fault",
                "vr_stale",
                "watchdog",
                "network_lost",
                "motion_unhealthy",
                "external_authority",
            }
            and paper_state in _ROLLBACK_STATES
        ):
            self._invalidate_rollback(name, logical_input.source_timestamp_ns)
            return self.session.handle(logical_input)

        if name == "human_finish" and paper_state == InterventionState.CORRECTION:
            transition = self.session.handle(
                logical_input,
                handback_evidence=handback_evidence,
            )
            if transition is not None and transition.current == SupervisorMode.HOLD:
                self.intervention.complete_correction(
                    end_timestamp_ns=logical_input.source_timestamp_ns,
                    sensor_skew_ok=True,
                    same_anchor=True,
                    controller_active_arms=tuple(self._controller_active_arms),
                )
                self._clear_transaction()
            return transition
        if paper_state in _ROLLBACK_STATES | {
            InterventionState.SCENE_RESTORE,
            InterventionState.CORRECTION_READY,
            InterventionState.CORRECTION,
        } and name in {
            "y_press",
            "y_short",
            "y_long_start",
            "y_long_end",
            "flowpro_mode_toggle",
        }:
            # This is an operator input rejection, not a process fault. Killing
            # the DDS thread here would make every later X/Y press appear dead.
            self._record_logical_input(
                LogicalInput(
                    "intervention_input_blocked",
                    logical_input.source_timestamp_ns,
                    {
                        "requested_event": name,
                        "intervention_state": paper_state.value,
                        "reason": "policy_start_or_exit_blocked_during_intervention",
                    },
                ),
                control_owner=self.session._control_owner(),
                defer=self.rollback_lease_id is not None,
            )
            self.note_input_blocked(
                logical_input,
                reason=f"Policy start/FlowPRO exit is blocked in {paper_state.value}",
            )
            return None
        if name in {
            "y_press",
            "y_short",
            "y_long_start",
            "y_long_end",
            "flowpro_mode_toggle",
        } and self.session.supervisor.mode in {
            SupervisorMode.AUTO,
            SupervisorMode.FORWARD_RUN,
            SupervisorMode.HUMAN,
            SupervisorMode.ABORT,
        }:
            mode = self.session.supervisor.mode
            if mode == SupervisorMode.HUMAN:
                reason = "waiting for automatic Human handback gates before Policy may restart"
            elif mode == SupervisorMode.ABORT:
                reason = "ABORT is terminal for this FlowPRO session"
            else:
                reason = "Policy input requires HOLD; stop or hand over the current run first"
            self._record_logical_input(
                LogicalInput(
                    "input_ignored_non_hold",
                    logical_input.source_timestamp_ns,
                    {
                        "requested_event": name,
                        "supervisor_mode": mode.value,
                        "reason": reason,
                    },
                ),
                control_owner=self.session._control_owner(),
                defer=False,
            )
            self.note_input_blocked(logical_input, reason=reason)
            return None
        if name == "x_press":
            self.note_input_blocked(
                logical_input,
                reason=f"intervention_state={paper_state.value}, supervisor_mode={self.session.supervisor.mode.value}",
            )
        if self.rollback_lease_id is not None:
            # All state-changing safety/takeover events were handled above.
            # Remaining events are audit-only and must not fsync on the live
            # 100 Hz lease path (for example side_takeover_input).
            self._record_logical_input(
                logical_input,
                control_owner=self.session._control_owner(),
                defer=True,
            )
            return None
        return self.session.handle(logical_input, handback_evidence=handback_evidence)

    def maybe_start_planning(
        self,
        *,
        now_ns: int,
        x_pressed: bool,
        y_pressed: bool,
        vr_frame_revision: int | None = None,
    ) -> Any:
        """Freeze after a fresh held-X frame, then run the expensive planner off-thread."""

        if not self._planning_requested:
            return None
        if not x_pressed:
            self._planning_requested = False
            self._execution_requested = False
            self._planning_input_barrier_revision = None
            self._set_status(
                "rollback_request_not_held",
                now_ns,
                "X is no longer held; rollback planning was not started",
                level=logging.WARNING,
            )
            return None
        if y_pressed:
            self._set_status(
                "blocked_y_pressed",
                now_ns,
                "Release Y while continuing to hold X; rollback requires X alone",
                level=logging.WARNING,
            )
            return None
        if now_ns < self._planning_not_before_ns:
            return None
        if not self._input_is_newer_than(
            vr_frame_revision,
            self._planning_input_barrier_revision,
        ):
            self._set_status(
                "awaiting_post_x_vr_frame",
                now_ns,
                "Policy is stopped; waiting for a newer held-X VR frame",
            )
            return None
        self._planning_requested = False
        self._planning_input_barrier_revision = None
        self._set_status(
            "freezing_history",
            now_ns,
            "Freezing the current Policy run before rollback planning",
        )
        if self.failure_timestamp_ns is None:
            return self._planning_failed(RuntimeError("Rollback request lost its failure timestamp"), now_ns)
        try:
            expected_policy_run_generation_id = (
                None if self.policy_run_generation_provider is None else self.policy_run_generation_provider()
            )
            if self.policy_run_generation_provider is not None and expected_policy_run_generation_id is None:
                raise RuntimeError("Rollback has no retained operator-started Policy run")
            segment, history_current_state = self.intervention.freeze_for_continuous_rollback(
                failure_timestamp_ns=self.failure_timestamp_ns,
                max_rollback_ns=self.max_rollback_ns,
                expected_policy_run_generation_id=expected_policy_run_generation_id,
            )
            # Read the action thread's fresh rollback-start mailbox after HOLD
            # has been acknowledged. Production supplies the last GDK-accepted
            # command reference so the planner and final trajectory validator
            # share an exact starting point. Never issue a second GDK read from
            # this DDS coordinator thread. The history tail remains a fallback
            # for dependency-injected/offline runtimes.
            current_state = (
                history_current_state
                if self.measured_state_provider is None
                else self.measured_state_provider()
            )
        except Exception as error:
            return self._planning_failed(error, now_ns)
        assert self.intervention.intervention_id is not None
        intervention_id = self.intervention.intervention_id
        token = str(uuid4())
        with self._planning_lock:
            self._planning_token = token
            self._planning_result = None
            self._planning_error = None

        def plan_worker() -> None:
            try:
                result = self.rollback_planner(segment, intervention_id, current_state)
                error: BaseException | None = None
            except BaseException as caught:  # propagate on the coordinator thread
                result = None
                error = caught
            with self._planning_lock:
                if self._planning_token == token:
                    self._planning_result = result
                    self._planning_error = error

        threading.Thread(
            target=plan_worker,
            name="FlowPRORollbackPlanner",
            daemon=True,
        ).start()
        self._set_status(
            "planning",
            now_ns,
            "Building the immutable 100 Hz rollback plan",
        )
        return None

    def rollback_tick(
        self,
        *,
        x_pressed: bool,
        health: RollbackHealth,
        y_pressed: bool = False,
        vr_frame_revision: int | None = None,
        now_ns: int | None = None,
    ) -> Any:
        now_ns = time.time_ns() if now_ns is None else now_ns
        started = self.maybe_start_planning(
            now_ns=now_ns,
            x_pressed=x_pressed,
            y_pressed=y_pressed,
            vr_frame_revision=vr_frame_revision,
        )
        if started is not None:
            return started
        planned = self._poll_planning(
            vr_frame_revision=vr_frame_revision,
            now_ns=now_ns,
        )
        if planned is not None:
            return planned
        if self._activation_pending:
            return self._activate_pending_execution(
                x_pressed=x_pressed,
                y_pressed=y_pressed,
                health=health,
                vr_frame_revision=vr_frame_revision,
                now_ns=now_ns,
            )
        if self.intervention.state in {
            InterventionState.ROLLBACK_READY,
            InterventionState.ROLLBACK_PAUSED,
        }:
            if not self._execution_requested or not x_pressed:
                self._set_status(
                    "awaiting_x_press",
                    now_ns,
                    "Rollback plan is stationary; press and hold X to execute",
                )
                return None
            if y_pressed:
                self._set_status(
                    "blocked_y_pressed",
                    now_ns,
                    "Release Y while continuing to hold X; rollback motion remains in HOLD",
                    level=logging.WARNING,
                )
                return None
            if not self._input_is_newer_than_barrier(vr_frame_revision):
                self._set_status(
                    "awaiting_fresh_vr_frame",
                    now_ns,
                    "Rollback is ready; waiting for a VR frame received after durable preparation",
                )
                return None
            if self._execution_requested and x_pressed:
                return self._start_execution(
                    now_ns,
                    deadman_pressed=True,
                    health=health,
                    vr_frame_revision=vr_frame_revision,
                )
            return None
        if self.intervention.state != InterventionState.ROLLBACK_EXEC:
            return None
        if self.rollback_lease_id is None:
            raise RuntimeError("Rollback execution lost its exclusive lease")
        try:
            result = self.rollback_executor.step(
                deadman_pressed=x_pressed,
                finish_to_anchor_authorized=not x_pressed,
                health=health,
                lease_id=self.rollback_lease_id,
            )
        except Exception as error:
            return self._planning_failed(error, now_ns)
        if result is None:
            return None
        measured = getattr(result, "measured_state", None)
        if measured is None:
            measured = result.command
        source_timestamp_ns = getattr(result, "source_timestamp_ns", None)
        if result.state == RollbackExecutorState.RUNNING:
            self._set_status(
                "rewinding" if x_pressed else "finishing_to_anchor",
                now_ns,
                (
                    "X is held; executing rollback"
                    if x_pressed
                    else "X is released; completing the same plan to the selected anchor"
                ),
                motion_expected=True,
            )
            try:
                self.intervention.observe_rollback_progress(
                    measured_state_24d=measured,
                    source_timestamp_ns=source_timestamp_ns,
                    cursor=result.waypoint_index,
                    observed_timestamp_ns=now_ns,
                )
            except Exception as error:
                return self._planning_failed(error, now_ns)
            return result
        if result.state == RollbackExecutorState.HOLD:
            try:
                self.rollback_executor.pause()
                self.rollback_lease_id = None
                self._flush_deferred_logical_inputs()
                transition = self.session.handle(
                    LogicalInput(
                        "rollback_pause",
                        now_ns,
                        {
                            "source": "finish_to_anchor",
                            "observation_id": getattr(result, "anchor_observation_id", None),
                        },
                    )
                )
                snapshot = self.intervention.pause_rollback(
                    measured_state_24d=measured,
                    source_timestamp_ns=source_timestamp_ns,
                    cursor=result.waypoint_index,
                    anchor_observation_id=getattr(result, "anchor_observation_id", None),
                    paused_timestamp_ns=now_ns,
                )
            except Exception as error:
                # The action thread has already acknowledged stationary HOLD.
                # A persistence/anchor-validation error must reject this
                # transaction, but must not kill the DDS coordinator and make
                # every later button press appear dead.
                return self._planning_failed(error, now_ns)
            if not isinstance(snapshot, SceneRestorationSnapshot):
                return self._planning_failed(
                    RuntimeError("Stationary anchor HOLD did not select a restoration snapshot"),
                    now_ns,
                )
            can_continue = bool(
                self.intervention.state == InterventionState.ROLLBACK_PAUSED
                and (
                    not getattr(self.intervention, "anchor_selected_for_restoration", False)
                    or getattr(self.intervention, "can_continue_selected_rollback", False)
                )
            )
            # Anchor validation, WAL writes and scene-image publication above
            # may take longer than one DDS period.  Never reuse their pre-I/O
            # X/health snapshot.  A held second X continues automatically after
            # the next actually received VR frame; no extra operator action is
            # required.
            self._motion_input_barrier_revision = vr_frame_revision
            self._set_status(
                (
                    "awaiting_fresh_vr_frame"
                    if can_continue and self._execution_requested and x_pressed
                    else "anchor_hold"
                    if can_continue
                    else "oldest_anchor_reached"
                ),
                now_ns,
                (
                    "Stationary anchor selected; waiting for a fresh held-X frame to continue"
                    if can_continue and self._execution_requested and x_pressed
                    else (
                        "Stationary inference anchor selected; press and hold X again to continue farther"
                        if can_continue
                        else "Rollback reached the oldest available inference anchor"
                    )
                ),
                level=logging.WARNING,
            )
            return transition
        if result.state == RollbackExecutorState.COMPLETE:
            try:
                # The action thread already acknowledged terminal HOLD. Revoke
                # and discard the physical lease before any image/WAL I/O.
                self.rollback_executor.disarm()
                self.rollback_lease_id = None
                self._executor_armed = False
                self._flush_deferred_logical_inputs()
                snapshot = self.intervention.complete_rollback(
                    measured_state_24d=measured,
                    completed_timestamp_ns=now_ns,
                )
                transition = self.session.handle(
                    LogicalInput(
                        "rollback_complete",
                        now_ns,
                        {"source": "rollback_endpoint", "observation_id": snapshot.observation_id},
                    )
                )
            except Exception as error:
                return self._planning_failed(error, now_ns)
            if transition is None or transition.current != SupervisorMode.HOLD:
                return self._planning_failed(
                    RuntimeError("Completed rollback did not return the supervisor to HOLD"),
                    now_ns,
                )
            self._set_status(
                "oldest_anchor_reached",
                now_ns,
                "Rollback completed at the oldest available inference anchor",
                level=logging.WARNING,
            )
            return transition
        if result.state in {RollbackExecutorState.FAULT, RollbackExecutorState.CANCELLED}:
            self._set_status(
                "execution_stopped",
                now_ns,
                result.reason,
                level=logging.ERROR,
            )
            self._invalidate_rollback(result.reason, now_ns)
            try:
                return self.session.fail_closed_hold(reason="watchdog", timestamp_ns=now_ns)
            except Exception:
                # The physical lease has already been revoked by
                # _invalidate_rollback().  A full/unavailable sidecar must not
                # kill the DDS consumer and leave every later button looking
                # dead.  Fall back to the pure in-memory supervisor transition
                # and invalidate the active generation there as well.
                logger.exception("FLOWPRO failed to persist terminal rollback HOLD")
                transition = self.session.supervisor.handle("watchdog")
                if transition is not None:
                    self.session._sync_generation(transition)
                return transition
        return result

    def _poll_planning(
        self,
        *,
        vr_frame_revision: int | None,
        now_ns: int,
    ) -> Any:
        if self.intervention.state != InterventionState.ROLLBACK_PLANNING:
            return None
        with self._planning_lock:
            result = self._planning_result
            error = self._planning_error
            done = result is not None or error is not None
            if done:
                self._planning_result = None
                self._planning_error = None
                self._planning_token = None
        if not done:
            return None
        if error is not None:
            return self._planning_failed(error, now_ns)
        assert result is not None
        try:
            self.intervention.complete_continuous_plan(result)
        except Exception as caught:
            return self._planning_failed(caught, now_ns)
        self._set_status(
            "plan_ready",
            now_ns,
            "Rollback plan is durable; waiting for a newer VR frame before execution",
        )
        self._motion_input_barrier_revision = vr_frame_revision
        # This non-None result is an intentional scheduling barrier: the caller
        # exits this DDS reconciliation instead of falling through to READY and
        # arming from X/health values sampled before plan persistence/fsync.
        return InterventionState.ROLLBACK_READY

    def _start_execution(
        self,
        timestamp_ns: int,
        *,
        deadman_pressed: bool,
        health: RollbackHealth,
        vr_frame_revision: int | None,
    ) -> Any:
        if self.intervention.state not in {
            InterventionState.ROLLBACK_READY,
            InterventionState.ROLLBACK_PAUSED,
        }:
            return None
        if not self._execution_requested:
            return None
        # Planning may complete on a DDS tick whose measured-state snapshot is
        # transiently incomplete.  Stay in READY/PAUSED and retry on a later
        # healthy tick; never publish a physical lease carrying an already
        # unsafe initial intent.  X-up clears _execution_requested, so health
        # recovery can never cause delayed motion after the operator releases.
        if not health.safe:
            blockers = tuple(getattr(health, "blockers", ())) or ("unspecified",)
            if blockers != self._health_wait_blockers:
                logger.warning(
                    "FLOWPRO ROLLBACK WAITING IN HOLD: health blockers=%s",
                    ",".join(blockers),
                )
                self._health_wait_blockers = blockers
            self._set_status(
                "blocked_health",
                timestamp_ns,
                ",".join(blockers),
                level=logging.WARNING,
            )
            return None
        if self._health_wait_blockers is not None:
            logger.warning("FLOWPRO ROLLBACK HEALTH RECOVERED: starting held-X execution")
            self._health_wait_blockers = None
        lease_id = str(uuid4())
        resume = self.intervention.state == InterventionState.ROLLBACK_PAUSED
        try:
            # All durable bookkeeping happens while the robot is still in
            # ordinary measured HOLD and no rollback mailbox lease exists.
            # Physical arm is deliberately deferred to a later, fresh VR/health
            # reconciliation so fsync can never starve a live 100 Hz lease.
            self._flush_deferred_logical_inputs()
            self.intervention.begin_rollback(resumed_timestamp_ns=timestamp_ns)
            transition = self.session.handle(
                LogicalInput(
                    "rollback_start",
                    timestamp_ns,
                    {
                        "rollback_session_id": self.intervention.rollback_session_id,
                        "rollback_cursor": self.intervention.rollback_cursor,
                    },
                )
            )
            if transition is None or transition.current != SupervisorMode.ROLLBACK_EXEC:
                raise RuntimeError("Rollback lease did not enter ROLLBACK_EXEC")
        except Exception as error:
            return self._planning_failed(error, timestamp_ns)
        self._activation_pending = True
        self._activation_resume = resume
        self._activation_lease_id = lease_id
        self._activation_barrier_revision = vr_frame_revision
        self._motion_input_barrier_revision = None
        self._set_status(
            "activation_waiting",
            timestamp_ns,
            "Rollback intent is durable; waiting for a newer held-X VR frame before physical lease activation",
        )
        return transition

    def _activate_pending_execution(
        self,
        *,
        x_pressed: bool,
        y_pressed: bool,
        health: RollbackHealth,
        vr_frame_revision: int | None,
        now_ns: int,
    ) -> Any:
        """Publish the physical lease only from post-persistence live input."""

        if not self._activation_pending:
            return None
        if not self._input_is_newer_than(
            vr_frame_revision,
            self._activation_barrier_revision,
        ):
            self._set_status(
                "awaiting_fresh_vr_frame",
                now_ns,
                "Physical rollback is not armed; waiting for a VR frame newer than the durable intent",
            )
            return None
        if not self._execution_requested or not x_pressed:
            self._set_status(
                "awaiting_x_press",
                now_ns,
                "Rollback is prepared but stationary; press and hold X to activate its physical lease",
            )
            return None
        if y_pressed:
            self._set_status(
                "blocked_y_pressed",
                now_ns,
                "Release Y while continuing to hold X; physical rollback remains unarmed",
                level=logging.WARNING,
            )
            return None
        if not health.safe:
            blockers = tuple(getattr(health, "blockers", ())) or ("unspecified",)
            self._set_status(
                "blocked_health",
                now_ns,
                ",".join(blockers),
                level=logging.WARNING,
            )
            return None
        lease_id = self._activation_lease_id
        if lease_id is None:
            return self._planning_failed(
                RuntimeError("Prepared rollback lost its physical lease identifier"),
                now_ns,
            )
        resume_detail = self._activation_resume_detail
        try:
            prepare_arm = getattr(self.rollback_executor, "prepare_arm", None)
            prepare_resume = getattr(self.rollback_executor, "prepare_resume", None)
            commit_prepared = getattr(self.rollback_executor, "commit_prepared", None)
            supports_prepare_commit = callable(commit_prepared) and (
                callable(prepare_resume) if self._activation_resume else callable(prepare_arm)
            )
            if supports_prepare_commit and not self._activation_prepared:
                # This stage may wait for measured HOLD, read GDK state, and
                # build a connector.  It must leave the action thread unaware
                # of the prepared lease.  A later accepted VR frame rechecks X
                # and health before the pure-memory commit.
                if self._activation_resume:
                    assert callable(prepare_resume)
                    resume_detail = prepare_resume(lease_id=lease_id)
                else:
                    assert callable(prepare_arm)
                    assert self.intervention.rollback_plan is not None
                    prepare_arm(self.intervention.rollback_plan, lease_id=lease_id)
                self._activation_prepared = True
                self._activation_resume_detail = resume_detail
                self._activation_barrier_revision = vr_frame_revision
                self._set_status(
                    "activation_prepared",
                    now_ns,
                    "Rollback path is prepared but stationary; waiting for a newer held-X VR frame",
                )
                return None
            if supports_prepare_commit:
                assert callable(commit_prepared)
                evaluated_ns = getattr(health, "evaluated_monotonic_ns", None)
                if evaluated_ns is None:
                    self._set_status(
                        "blocked_health_timestamp",
                        now_ns,
                        "Prepared rollback is stationary; health has no monotonic evaluation timestamp",
                        level=logging.WARNING,
                    )
                    return None
                age_ns = time.monotonic_ns() - int(evaluated_ns)
                if age_ns < 0 or age_ns > 200_000_000:
                    self._set_status(
                        "blocked_health_timestamp",
                        now_ns,
                        f"Prepared rollback is stationary; health evaluation age is {age_ns / 1e6:.1f} ms",
                        level=logging.WARNING,
                    )
                    return None
                commit_prepared(
                    lease_id=lease_id,
                    deadman_pressed=True,
                    health=health,
                )
            elif self._activation_resume:
                resume_with_intent = getattr(self.rollback_executor, "resume_with_intent", None)
                if callable(resume_with_intent):
                    resume_detail = resume_with_intent(
                        lease_id=lease_id,
                        deadman_pressed=True,
                        health=health,
                    )
                else:
                    resume_detail = self.rollback_executor.resume(lease_id=lease_id)
            else:
                assert self.intervention.rollback_plan is not None
                arm_with_intent = getattr(self.rollback_executor, "arm_with_intent", None)
                if callable(arm_with_intent):
                    arm_with_intent(
                        self.intervention.rollback_plan,
                        lease_id=lease_id,
                        deadman_pressed=True,
                        health=health,
                    )
                else:
                    self.rollback_executor.arm(
                        self.intervention.rollback_plan,
                        lease_id=lease_id,
                    )
            self.rollback_lease_id = lease_id
            self._executor_armed = True
        except Exception as error:
            return self._planning_failed(error, now_ns)
        if resume_detail is not None:
            # Preserve bridge provenance without putting a WAL fsync back onto
            # the live-lease critical path. It is flushed after stationary HOLD.
            self._deferred_logical_inputs.append(
                (
                    LogicalInput(
                        "rollback_execution_bound",
                        now_ns,
                        dict(resume_detail),
                    ),
                    # ``rollback_exec`` is an internal supervisor mode, not a
                    # durable authority value accepted by flowpro.event/v1.
                    "rollback",
                )
            )
        self._activation_pending = False
        self._activation_resume = False
        self._activation_lease_id = None
        self._activation_barrier_revision = None
        self._activation_prepared = False
        self._activation_resume_detail = None
        self._health_wait_blockers = None
        self._set_status(
            "rewinding",
            now_ns,
            "Fresh held-X and health accepted; physical rollback lease is active",
            motion_expected=True,
        )
        return None

    def _input_is_newer_than_barrier(self, vr_frame_revision: int | None) -> bool:
        return self._input_is_newer_than(vr_frame_revision, self._motion_input_barrier_revision)

    @staticmethod
    def _input_is_newer_than(
        vr_frame_revision: int | None,
        barrier_revision: int | None,
    ) -> bool:
        # Direct/offline callers without an application receipt counter still
        # receive a one-call scheduling barrier because plan/precommit return
        # immediately. Production always supplies both revisions.
        if vr_frame_revision is None or barrier_revision is None:
            return True
        return vr_frame_revision > barrier_revision

    def _record_logical_input(
        self,
        logical_input: LogicalInput,
        *,
        control_owner: str,
        defer: bool,
    ) -> None:
        if defer:
            self._deferred_logical_inputs.append((logical_input, control_owner))
            return
        self.session.coordinator.append_logical_input(
            logical_input,
            control_owner=control_owner,
        )

    def _flush_deferred_logical_inputs(self) -> None:
        pending = self._deferred_logical_inputs
        self._deferred_logical_inputs = []
        for index, (logical_input, control_owner) in enumerate(pending):
            try:
                self.session.coordinator.append_logical_input(
                    logical_input,
                    control_owner=control_owner,
                )
            except Exception:
                self._deferred_logical_inputs = pending[index:] + self._deferred_logical_inputs
                raise

    def _record_unavailable(self, logical_input: LogicalInput) -> None:
        reason = "rollback_capability_disabled"
        self.session.coordinator.append_logical_input(
            LogicalInput(
                "rollback_unavailable",
                logical_input.source_timestamp_ns,
                {
                    **logical_input.detail,
                    "requested_event": logical_input.name,
                    "reason": reason,
                },
            ),
            control_owner=SupervisorMode.HOLD.value,
        )
        self._set_status(
            "capability_disabled",
            logical_input.source_timestamp_ns,
            "Physical rollback was not enabled for this process",
            level=logging.WARNING,
        )

    def _planning_failed(self, error: BaseException, timestamp_ns: int) -> Any:
        reason = f"{type(error).__name__}: {error}"
        logger.error("FLOWPRO ROLLBACK REJECTED: %s", reason)
        # Revoke physical ownership first. Persistence may itself be the thing
        # that failed (disk full/corrupt sidecar), and must never strand a live
        # rollback lease while we attempt to record the diagnostic.
        if self._executor_armed or self._activation_prepared:
            try:
                self.rollback_executor.cancel(reason="rollback_planning_or_resume_failed")
            except Exception:
                # The application-level HOLD transition below remains the
                # independent fallback. Never let a mailbox cleanup exception
                # strand the button coordinator in ROLLBACK_EXEC.
                logger.exception("FLOWPRO rollback executor cancellation request failed")
            self._executor_armed = False
        self.rollback_lease_id = None
        self._execution_requested = False
        self.failure_timestamp_ns = None
        self._planning_input_barrier_revision = None
        self._health_wait_blockers = None
        self._motion_input_barrier_revision = None
        self._activation_pending = False
        self._activation_resume = False
        self._activation_lease_id = None
        self._activation_barrier_revision = None
        self._activation_prepared = False
        self._activation_resume_detail = None
        try:
            self._flush_deferred_logical_inputs()
        except Exception:
            logger.exception("FLOWPRO deferred input persistence failed while rejecting rollback")
            self._deferred_logical_inputs.clear()

        hold_transition = None
        if self.session.supervisor.mode == SupervisorMode.ROLLBACK_EXEC:
            try:
                hold_transition = self.session.fail_closed_hold(
                    reason="watchdog",
                    timestamp_ns=timestamp_ns,
                )
            except Exception:
                # If the diagnostic WAL itself is unavailable, fail the pure
                # control state to HOLD in memory and synchronize generation.
                # Physical ownership was already revoked above.
                logger.exception("FLOWPRO failed to persist fail-closed HOLD transition")
                hold_transition = self.session.supervisor.handle("watchdog")
                if hold_transition is not None:
                    self.session._sync_generation(hold_transition)
        try:
            if self.intervention.state in _ROLLBACK_STATES:
                self.intervention.planning_failed(reason=reason, timestamp_ns=timestamp_ns)
            else:
                self.session.coordinator.append_logical_input(
                    LogicalInput(
                        "rollback_planning_failed",
                        timestamp_ns,
                        {"reason": reason},
                    ),
                    control_owner=SupervisorMode.HOLD.value,
                )
        except Exception:
            # Diagnostic persistence can fail for the same reason that anchor
            # finalization failed (for example a full/slow disk). Ownership was
            # already revoked above, so retain an in-memory REJECTED state and
            # keep the DDS/button service alive in HOLD.
            logger.exception(
                "FLOWPRO ROLLBACK DIAGNOSTIC PERSISTENCE FAILED after physical ownership was revoked"
            )
            if hasattr(self.intervention, "state"):
                self.intervention.state = InterventionState.REJECTED
        self._set_status(
            "planning_failed",
            timestamp_ns,
            reason,
            level=logging.ERROR,
        )
        return hold_transition or RollbackPlanningFailed(reason=reason)

    def _invalidate_rollback(self, reason: str, timestamp_ns: int) -> None:
        self._planning_requested = False
        self._execution_requested = False
        self._planning_input_barrier_revision = None
        with self._planning_lock:
            self._planning_token = None
            self._planning_result = None
            self._planning_error = None
        if self._executor_armed or self._activation_prepared:
            try:
                abandon_for_human = getattr(
                    self.rollback_executor,
                    "abandon_for_human",
                    None,
                )
                if reason in {"human_takeover", "external_authority"} and callable(abandon_for_human):
                    abandon_for_human()
                else:
                    self.rollback_executor.cancel(reason=reason)
            except Exception:
                logger.exception("FLOWPRO rollback executor cancellation request failed")
            self._executor_armed = False
        self.rollback_lease_id = None
        self._motion_input_barrier_revision = None
        self._activation_pending = False
        self._activation_resume = False
        self._activation_lease_id = None
        self._activation_barrier_revision = None
        self._activation_prepared = False
        self._activation_resume_detail = None
        try:
            self._flush_deferred_logical_inputs()
        except Exception:
            logger.exception("FLOWPRO deferred input persistence failed while invalidating rollback")
            self._deferred_logical_inputs.clear()
        try:
            self.intervention.invalidate_rollback(reason=reason, timestamp_ns=timestamp_ns)
        except Exception:
            # Invalidation persistence is diagnostic after physical ownership
            # has been revoked. Keep the coordinator alive and make the
            # in-memory state retryable from HOLD.
            logger.exception("FLOWPRO rollback invalidation persistence failed")
            if hasattr(self.intervention, "state"):
                self.intervention.state = InterventionState.REJECTED
        self.failure_timestamp_ns = None
        self._health_wait_blockers = None
        self._set_status(
            "invalidated",
            timestamp_ns,
            reason,
            level=logging.ERROR,
        )

    def _clear_transaction(self) -> None:
        self.failure_timestamp_ns = None
        self.rollback_lease_id = None
        self._planning_requested = False
        self._execution_requested = False
        self._planning_input_barrier_revision = None
        self._executor_armed = False
        self._health_wait_blockers = None
        self._motion_input_barrier_revision = None
        self._activation_pending = False
        self._activation_resume = False
        self._activation_lease_id = None
        self._activation_barrier_revision = None
        self._activation_prepared = False
        self._activation_resume_detail = None
        self._controller_active_arms = [False, False]
        try:
            self._flush_deferred_logical_inputs()
        except Exception:
            logger.exception("FLOWPRO deferred input persistence failed while clearing transaction")
            self._deferred_logical_inputs.clear()
        self._set_status("idle", time.time_ns(), "rollback transaction cleared")

    def _set_status(
        self,
        code: str,
        timestamp_ns: int,
        detail: str,
        *,
        motion_expected: bool = False,
        level: int = logging.INFO,
    ) -> None:
        status = RollbackRuntimeStatus(
            code=str(code),
            detail=str(detail),
            timestamp_ns=int(timestamp_ns),
            motion_expected=bool(motion_expected),
        )
        previous = self._rollback_status
        if (
            previous.code == status.code
            and previous.detail == status.detail
            and previous.motion_expected == status.motion_expected
        ):
            return
        self._rollback_status = status
        logger.log(
            level,
            "FLOWPRO ROLLBACK STATUS code=%s motion_expected=%s detail=%s",
            status.code,
            status.motion_expected,
            status.detail,
        )


class RobotClientRollbackDriver:
    """Mailbox adapter: only RobotClientRunner's action thread may call GDK."""

    def __init__(self, robot_client_runner: Any) -> None:
        self.runner = robot_client_runner
        self.runner.configure_flowpro_rollback()

    def enter_hold(self) -> None:
        """Acknowledge measured HOLD before any X-down audit I/O."""

        self.runner.enter_flowpro_hold()

    def arm(self, plan: RollbackTrajectory, *, lease_id: str) -> None:
        self.runner.arm_flowpro_rollback(plan, lease_id=lease_id)

    def arm_with_intent(
        self,
        plan: RollbackTrajectory,
        *,
        lease_id: str,
        deadman_pressed: bool,
        health: RollbackHealth,
    ) -> None:
        self.runner.arm_flowpro_rollback(
            plan,
            lease_id=lease_id,
            initial_deadman_pressed=deadman_pressed,
            initial_health=health,
        )

    def prepare_arm(self, plan: RollbackTrajectory, *, lease_id: str) -> None:
        """Perform potentially blocking path setup without publishing a lease."""

        self.runner.prepare_flowpro_rollback(plan, lease_id=lease_id)

    def commit_prepared(
        self,
        *,
        lease_id: str,
        deadman_pressed: bool,
        health: RollbackHealth,
    ) -> None:
        """Publish a prepared path from a later fresh held-X frame."""

        self.runner.commit_prepared_flowpro_rollback(
            lease_id=lease_id,
            deadman_pressed=deadman_pressed,
            health=health,
        )

    def resume(self, *, lease_id: str) -> dict[str, float | int] | None:
        return self.runner.resume_flowpro_rollback(lease_id=lease_id)

    def resume_with_intent(
        self,
        *,
        lease_id: str,
        deadman_pressed: bool,
        health: RollbackHealth,
    ) -> dict[str, float | int] | None:
        return self.runner.resume_flowpro_rollback(
            lease_id=lease_id,
            initial_deadman_pressed=deadman_pressed,
            initial_health=health,
        )

    def prepare_resume(self, *, lease_id: str) -> dict[str, float | int] | None:
        """Prepare only the untouched paused suffix; keep action thread in HOLD."""

        return self.runner.prepare_resume_flowpro_rollback(lease_id=lease_id)

    def step(
        self,
        *,
        deadman_pressed: bool,
        finish_to_anchor_authorized: bool = False,
        health: RollbackHealth,
        lease_id: str,
    ) -> Any | None:
        return self.runner.poll_or_update_flowpro_rollback_control(
            deadman_pressed=deadman_pressed,
            finish_to_anchor_authorized=finish_to_anchor_authorized,
            health=health,
            lease_id=lease_id,
        )

    def pause(self) -> None:
        self.runner.pause_flowpro_rollback()

    def finish_paused(self) -> None:
        self.runner.finish_paused_flowpro_rollback()

    def cancel(self, *, reason: str) -> None:
        self.runner.cancel_flowpro_rollback(reason=reason)

    def abandon_for_human(self) -> None:
        self.runner.abandon_flowpro_rollback_for_human()

    def disarm(self) -> None:
        self.runner.disarm_flowpro_rollback()
