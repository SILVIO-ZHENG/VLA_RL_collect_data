"""Durable paper-original FlowPRO intervention transaction.

This module coordinates already-existing control primitives.  It never opens
GDK, starts/stops the vendor recorder, or grants physical ownership.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from ..common.compat import StrEnum
from ..common.sidecar import CAMERA_SOURCES, EVENTS_WAL
from .executed import (
    MAX_CAMERA_SKEW_NS,
    ExecutedRejectedSegment,
    FlowPROExecutionHistory,
    InferenceAnchor,
    RollbackAnchorSelection,
)
from .feedback import SceneRestorationFeedbackWriter
from .rollback import RollbackTrajectory
from .supervisor import EpisodeCoordinator, LogicalInput


class InterventionState(StrEnum):
    IDLE = "idle"
    ROLLBACK_PLANNING = "rollback_planning"
    ROLLBACK_READY = "rollback_ready"
    ROLLBACK_EXEC = "rollback_exec"
    ROLLBACK_PAUSED = "rollback_paused"
    SCENE_RESTORE = "scene_restore"
    CORRECTION_READY = "correction_ready"
    CORRECTION = "correction"
    COMPLETE = "complete"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SceneRestorationSnapshot:
    observation_id: str
    anchor_timestamp_ns: int
    task: str
    state_path: Path
    image_paths: dict[str, Path]
    image_timestamps_ns: dict[str, int]


@dataclass(frozen=True)
class RollbackProgress:
    rollback_session_id: str
    state: InterventionState
    cursor: int
    total_waypoints: int
    failure_timestamp_ns: int
    current_history_timestamp_ns: int
    rolled_back_history_ns: int
    remaining_history_ns: int
    crossed_anchor_ids: tuple[str, ...]
    last_crossed_anchor_id: str | None
    next_older_anchor_id: str | None


def load_scene_restoration_snapshot(
    sidecar_directory: Path,
    observation_id: str,
) -> SceneRestorationSnapshot:
    """Resolve and checksum the exact historical inference observation."""

    UUID(observation_id)
    root = sidecar_directory.expanduser().resolve()
    events = []
    for line in (root / EVENTS_WAL).read_bytes().splitlines():
        if line.strip():
            events.append(json.loads(line))
    matches = [
        event
        for event in events
        if event.get("event_type") == "ObservationCaptured" and event.get("observation_id") == observation_id
    ]
    if len(matches) != 1:
        raise ValueError("Scene restoration requires exactly one matching observation")
    event = matches[0]
    payload = event["payload"]

    def resolve(record: dict[str, Any]) -> Path:
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Scene restoration artifact escapes the sidecar")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        if len(data) != int(record["size_bytes"]) or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Scene restoration artifact hash mismatch: {path}")
        return path

    images = payload["images"]
    if set(images) != set(CAMERA_SOURCES):
        raise ValueError("Scene restoration snapshot lacks one or more inference cameras")
    return SceneRestorationSnapshot(
        observation_id=observation_id,
        anchor_timestamp_ns=int(payload["anchor_timestamp_ns"]),
        task=str(payload["task"]),
        state_path=resolve(payload["state"]),
        image_paths={key: resolve(images[key]) for key in CAMERA_SOURCES},
        image_timestamps_ns={key: int(images[key]["vendor_timestamp_ns"]) for key in CAMERA_SOURCES},
    )


class PaperInterventionController:
    """Enforce freeze → rollback → scene restoration → correction ordering."""

    def __init__(
        self,
        *,
        coordinator: EpisodeCoordinator,
        history: FlowPROExecutionHistory,
        native_action_dim: int,
        effector_profile: str,
        feedback_writer: SceneRestorationFeedbackWriter | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.history = history
        self.native_action_dim = native_action_dim
        self.effector_profile = effector_profile
        self.feedback_writer = feedback_writer
        self.state = InterventionState.IDLE
        self.intervention_id: str | None = None
        self.rollback_session_id: str | None = None
        self.history_snapshot_id: str | None = None
        self.selection: Any | None = None
        self.segment: Any | None = None
        self.anchor_segments: dict[str, tuple[RollbackAnchorSelection, ExecutedRejectedSegment]] = {}
        self.rollback_plan: RollbackTrajectory | None = None
        self.correction_start_ns: int | None = None
        self.rollback_cursor = 0
        self.crossed_anchor_ids: set[str] = set()
        # Crossing detection runs on the DDS coordinator while the 100 Hz
        # action thread still owns a live rollback lease.  Never perform WAL
        # fsyncs or copy feedback images from that path: either operation can
        # stall health-mailbox refresh long enough to cancel otherwise healthy
        # motion.  Records are drained only after HOLD/COMPLETE has been
        # acknowledged and the physical lease has been revoked.
        self._pending_anchor_crossings: list[dict[str, Any]] = []
        self.current_history_timestamp_ns: int | None = None
        self.last_crossed_anchor_id: str | None = None
        self.anchor_selected_for_restoration = False
        self.anchor_selected_timestamp_ns: int | None = None

    @property
    def can_continue_selected_rollback(self) -> bool:
        return bool(
            self.state == InterventionState.ROLLBACK_PAUSED
            and self.anchor_selected_for_restoration
            and self.rollback_plan is not None
            and self.rollback_cursor < len(self.rollback_plan.waypoints_24d) - 1
        )

    @property
    def rollback_progress(self) -> RollbackProgress | None:
        if self.rollback_session_id is None or self.selection is None:
            return None
        current_timestamp = (
            self.selection.failure_timestamp_ns
            if self.current_history_timestamp_ns is None
            else self.current_history_timestamp_ns
        )
        ordered = sorted(
            (selection.anchor for selection, _segment in self.anchor_segments.values()),
            key=lambda anchor: anchor.timestamp_ns,
            reverse=True,
        )
        next_older = next(
            (anchor for anchor in ordered if anchor.timestamp_ns < current_timestamp),
            None,
        )
        oldest_timestamp = min(anchor.timestamp_ns for anchor in ordered)
        return RollbackProgress(
            rollback_session_id=self.rollback_session_id,
            state=self.state,
            cursor=self.rollback_cursor,
            total_waypoints=(0 if self.rollback_plan is None else len(self.rollback_plan.waypoints_24d)),
            failure_timestamp_ns=self.selection.failure_timestamp_ns,
            current_history_timestamp_ns=current_timestamp,
            rolled_back_history_ns=max(0, self.selection.failure_timestamp_ns - current_timestamp),
            remaining_history_ns=max(0, current_timestamp - oldest_timestamp),
            crossed_anchor_ids=tuple(
                anchor.observation_id
                for anchor in ordered
                if anchor.observation_id in self.crossed_anchor_ids
            ),
            last_crossed_anchor_id=self.last_crossed_anchor_id,
            next_older_anchor_id=(None if next_older is None else next_older.observation_id),
        )

    def freeze_for_continuous_rollback(
        self,
        *,
        failure_timestamp_ns: int,
        max_rollback_ns: int | None = None,
        expected_policy_run_generation_id: str | None = None,
    ) -> tuple[ExecutedRejectedSegment, np.ndarray]:
        """Freeze the complete measured path of the latest operator-started Policy run.

        Physical rollback and paper-strict preference eligibility are separate
        contracts.  Explicit 100 Hz ``queue_underrun_hold`` samples preserve a
        physically complete path across an asynchronous inference/watchdog HOLD,
        but make the captured negative ineligible for strict training.  A bare
        evidence gap is never planned through.
        """

        if self.state not in {InterventionState.IDLE, InterventionState.COMPLETE, InterventionState.REJECTED}:
            raise RuntimeError("A FlowPRO intervention is already active")
        intervention_id = str(uuid4())
        rollback_session_id = str(uuid4())
        history_snapshot_id = str(uuid4())
        if failure_timestamp_ns <= 0:
            raise ValueError("Rollback failure timestamp must be positive")
        if max_rollback_ns is not None and max_rollback_ns <= 0:
            raise ValueError("Optional rollback compatibility window must be positive")
        lower_bound = None if max_rollback_ns is None else failure_timestamp_ns - max_rollback_ns
        history_samples = tuple(
            sample for sample in self.history.samples if sample.timestamp_ns <= failure_timestamp_ns
        )
        if not history_samples:
            raise ValueError("No accepted Policy execution exists before rollback")
        # generation_id is the Policy-run identifier created by the operator's
        # Y start.  The application-owned PolicyRunScope is authoritative in
        # production; deriving it from the tail is retained only for standalone
        # controller tests and offline tools.
        policy_run_generation_id = (
            history_samples[-1].generation_id
            if expected_policy_run_generation_id is None
            else str(expected_policy_run_generation_id)
        )
        try:
            UUID(policy_run_generation_id)
        except ValueError as error:
            raise ValueError("Policy-run generation identifier must be a UUID") from error
        if history_samples[-1].generation_id != policy_run_generation_id:
            raise ValueError("Latest execution sample does not belong to the active Policy-run scope")
        executed_observation_ids = {
            sample.observation_id
            for sample in history_samples
            if sample.generation_id == policy_run_generation_id
            and sample.control_owner == "policy"
            and sample.command_accepted
        }
        frozen_anchors = tuple(
            anchor
            for anchor in self.history.anchors
            if anchor.timestamp_ns < failure_timestamp_ns
            and (lower_bound is None or anchor.timestamp_ns >= lower_bound)
            and anchor.exact_observation_complete
            and anchor.camera_skew_ns <= MAX_CAMERA_SKEW_NS
            and anchor.generation_id == policy_run_generation_id
            # An inference response that lost the plan-revision race is useful
            # diagnostic data, but it never drove the robot and therefore
            # cannot define an executed negative or a rollback training anchor.
            and anchor.observation_id in executed_observation_ids
        )
        anchor_segments: dict[str, tuple[RollbackAnchorSelection, ExecutedRejectedSegment]] = {}
        anchor_errors: dict[str, str] = {}
        for anchor in frozen_anchors:
            selection = RollbackAnchorSelection(
                anchor=anchor,
                failure_timestamp_ns=failure_timestamp_ns,
                requested_horizon_ns=failure_timestamp_ns - anchor.timestamp_ns,
                requested_target_timestamp_ns=anchor.timestamp_ns,
                actual_anchor_timestamp_ns=anchor.timestamp_ns,
                snap_offset_ns=0,
            )
            try:
                segment = self.history.physical_rollback_segment(
                    selection,
                    native_action_dim=self.native_action_dim,
                    effector_profile=self.effector_profile,
                )
            except ValueError as error:
                anchor_errors[anchor.observation_id] = str(error)
                continue
            if not segment.physical_history_complete:
                anchor_errors[anchor.observation_id] = (
                    "physical rollback history contains "
                    f"{segment.execution_gap_count} unobserved gap(s); "
                    f"maximum={segment.max_execution_gap_ns}ns"
                )
                continue
            if segment.generation_ids != (policy_run_generation_id,):
                anchor_errors[anchor.observation_id] = (
                    "physical rollback history crosses a Policy-run generation boundary: "
                    f"expected={policy_run_generation_id}, observed={segment.generation_ids}"
                )
                continue
            anchor_segments[anchor.observation_id] = (selection, segment)
        if not anchor_segments:
            detail = "; ".join(anchor_errors.values()) or "no complete inference anchor before failure"
            raise ValueError("No physically complete Policy rollback trajectory is available: " + detail)
        selection, segment = min(
            anchor_segments.values(),
            key=lambda item: item[0].anchor.timestamp_ns,
        )
        segment = replace(
            segment,
            rollback_anchors=tuple(
                item[0].anchor
                for item in sorted(anchor_segments.values(), key=lambda item: item[0].anchor.timestamp_ns)
            ),
        )
        self.coordinator.record_paper_phase(
            "RollbackRequested",
            intervention_id=intervention_id,
            vendor_timestamp_ns=failure_timestamp_ns,
            payload={
                "interaction": "continuous_x_hold/v2",
                "rollback_scope": "current_contiguous_policy_history",
                "compatibility_max_rollback_ns": max_rollback_ns,
                "available_rollback_history_ns": failure_timestamp_ns - selection.anchor.timestamp_ns,
                "rollback_session_id": rollback_session_id,
                "history_snapshot_id": history_snapshot_id,
                "available_anchor_ids": list(anchor_segments),
                "policy_run_generation_id": policy_run_generation_id,
                "strict_preference_eligible": segment.strict_preference_eligible,
                "excluded_anchor_reasons": anchor_errors,
            },
        )
        # Persist the full maximum-range snapshot before planning or motion.
        self.coordinator.record_executed_segment(
            intervention_id=intervention_id,
            selection=selection,
            segment=segment,
            native_action_dim=self.native_action_dim,
        )
        self.intervention_id = intervention_id
        self.rollback_session_id = rollback_session_id
        self.history_snapshot_id = history_snapshot_id
        self.selection = selection
        self.segment = segment
        self.anchor_segments = anchor_segments
        self.rollback_plan = None
        self.rollback_cursor = 0
        self.crossed_anchor_ids.clear()
        self._pending_anchor_crossings.clear()
        self.current_history_timestamp_ns = failure_timestamp_ns
        self.last_crossed_anchor_id = None
        self.anchor_selected_for_restoration = False
        self.anchor_selected_timestamp_ns = None
        self.state = InterventionState.ROLLBACK_PLANNING
        return segment, segment.samples[-1].measured_state_24d.copy()

    def complete_continuous_plan(self, plan: RollbackTrajectory) -> SceneRestorationSnapshot:
        if self.state != InterventionState.ROLLBACK_PLANNING:
            raise RuntimeError("Rollback planning completion is out of order")
        assert self.intervention_id is not None and self.selection is not None
        if plan.intervention_id != self.intervention_id:
            raise ValueError("Rollback plan belongs to another intervention")
        if plan.observation_id != self.selection.anchor.observation_id:
            raise ValueError("Rollback plan endpoint differs from the frozen oldest anchor")
        writer = self.coordinator.writer
        if writer is None:
            raise RuntimeError("FlowPRO coordinator lost its active writer")
        if plan.waypoint_source_timestamps_ns is None:
            raise ValueError("Continuous rollback plan lacks historical source timestamps")
        plan_artifact = writer.store_rollback_plan(self.intervention_id, plan.waypoints_24d)
        timestamp_artifact = writer.store_rollback_plan_source_timestamps(
            self.intervention_id, plan.waypoint_source_timestamps_ns
        )
        self.coordinator.record_paper_phase(
            "RollbackPlanned",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=self.selection.failure_timestamp_ns,
            observation_id=self.selection.anchor.observation_id,
            payload={
                "actual_anchor_timestamp_ns": self.selection.actual_anchor_timestamp_ns,
                "duration_s": plan.duration_s,
                "servo_fps": plan.servo_fps,
                "collision_safety_enabled": plan.collision_safety_enabled,
                "safety_config_sha256": plan.safety_config_sha256,
                "rollback_session_id": self.rollback_session_id,
                "history_snapshot_id": self.history_snapshot_id,
                "available_anchor_ids": list(self.anchor_segments),
                "available_anchors": [
                    {
                        "observation_id": selection.anchor.observation_id,
                        "timestamp_ns": selection.anchor.timestamp_ns,
                    }
                    for selection, _segment in sorted(
                        self.anchor_segments.values(),
                        key=lambda item: item[0].anchor.timestamp_ns,
                        reverse=True,
                    )
                ],
                "plan": plan_artifact,
                "plan_source_timestamps": timestamp_artifact,
            },
        )
        self.rollback_plan = plan
        self.state = InterventionState.ROLLBACK_READY
        snapshot = self._snapshot(self.selection.anchor)
        # Historical images are needed only after the robot has reached the
        # selected scene-restoration anchor.  Copying three images and fsyncing
        # their manifest here used to block the DDS/input coordinator just
        # before it granted motion.  Keep READY as a data-only state and publish
        # the exact images at ``scene_restore`` after stationary HOLD instead.
        return snapshot

    def begin_rollback(
        self,
        *,
        resume_detail: dict[str, float | int] | None = None,
        resumed_timestamp_ns: int | None = None,
    ) -> None:
        if self.state not in {
            InterventionState.ROLLBACK_READY,
            InterventionState.ROLLBACK_PAUSED,
            InterventionState.ROLLBACK_EXEC,
        }:
            raise RuntimeError("Rollback is not ready")
        assert self.intervention_id is not None
        if self.anchor_selected_for_restoration:
            self.coordinator.append_logical_input(
                LogicalInput(
                    "rollback_anchor_reopened",
                    self.selection.anchor.timestamp_ns,
                    {
                        "intervention_id": self.intervention_id,
                        "observation_id": self.selection.anchor.observation_id,
                        "rollback_cursor": self.rollback_cursor,
                    },
                ),
                control_owner="hold",
            )
            self.anchor_selected_for_restoration = False
            self.anchor_selected_timestamp_ns = None
        payload: dict[str, Any] = {"rollback_cursor": self.rollback_cursor}
        if resume_detail is not None:
            payload.update(resume_detail)
            payload["execution_mode"] = "immutable_plan_suffix_resume"
        self.coordinator.record_paper_phase(
            "RollbackResumed",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=(
                self.selection.failure_timestamp_ns
                if resumed_timestamp_ns is None
                else int(resumed_timestamp_ns)
            ),
            observation_id=self.selection.anchor.observation_id,
            payload=payload,
            control_owner="rollback",
        )
        self.state = InterventionState.ROLLBACK_EXEC

    def pause_rollback(
        self,
        *,
        measured_state_24d: Any,
        source_timestamp_ns: int | None,
        cursor: int,
        anchor_observation_id: str | None,
        paused_timestamp_ns: int,
        endpoint_tolerance: float = 0.01,
        source_tolerance_ns: int = 100_000_000,
    ) -> SceneRestorationSnapshot:
        """Pause only after the immutable plan has reached a stationary anchor.

        X-up motion is completed by the 100 Hz executor along the untouched
        suffix of the main rollback plan.  Consequently the coordinator no
        longer chooses a nearby pose or constructs a second snap trajectory:
        it only validates the exact anchor/cursor acknowledgement returned by
        that executor.
        """

        if self.state != InterventionState.ROLLBACK_EXEC:
            raise RuntimeError("Rollback pause is out of order")
        measured = np.asarray(measured_state_24d, dtype=np.float32)
        if measured.shape != (24,) or not np.isfinite(measured).all():
            raise ValueError("Rollback pause requires a finite measured [24] state")
        assert self.intervention_id is not None and self.selection is not None
        if self.rollback_plan is None:
            raise ValueError("Rollback pause requires the immutable main plan")
        if anchor_observation_id is None:
            raise ValueError("Rollback executor HOLD lacks an inference-anchor identifier")
        entry = self.anchor_segments.get(anchor_observation_id)
        if entry is None:
            raise ValueError("Rollback executor selected an anchor outside the frozen history")
        selected, selected_segment = entry
        selected_cursor = int(cursor)
        if not 0 <= selected_cursor < len(self.rollback_plan.waypoints_24d):
            raise ValueError("Rollback executor anchor cursor is outside the main plan")
        if selected_cursor < self.rollback_cursor:
            raise ValueError("Rollback executor attempted to replay an already rewound prefix")
        pose_error = float(
            np.max(
                np.abs(
                    measured[: self.native_action_dim] - selected.anchor.state_24d[: self.native_action_dim]
                )
            )
        )
        if pose_error > endpoint_tolerance:
            raise ValueError("Measured robot state did not reach the selected rollback anchor")
        if source_timestamp_ns is None:
            raise ValueError("Rollback executor anchor HOLD lacks a source timestamp")
        time_error = abs(int(source_timestamp_ns) - selected.anchor.timestamp_ns)
        if time_error > source_tolerance_ns:
            raise ValueError("Rollback executor anchor HOLD has mismatched historical time")

        # The last action-thread RUNNING acknowledgement can precede the exact
        # stationary anchor sample.  Queue the selected anchor explicitly, then
        # persist all crossings now that RobotClientRollbackDriver.pause() has
        # already revoked the live motion lease.
        self._queue_anchor_crossing(
            selection=selected,
            measured=measured,
            source_timestamp_ns=int(source_timestamp_ns),
            cursor=selected_cursor,
            observed_timestamp_ns=paused_timestamp_ns,
            endpoint_tolerance=endpoint_tolerance,
            source_tolerance_ns=source_tolerance_ns,
        )
        self._flush_pending_anchor_crossings()

        self.rollback_cursor = selected_cursor
        self.current_history_timestamp_ns = selected.anchor.timestamp_ns
        self.selection, self.segment = selected, selected_segment
        self.crossed_anchor_ids.add(selected.anchor.observation_id)
        self.last_crossed_anchor_id = selected.anchor.observation_id
        self.coordinator.record_paper_phase(
            "RollbackPaused",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=paused_timestamp_ns,
            observation_id=selected.anchor.observation_id,
            payload={
                "rollback_cursor": self.rollback_cursor,
                "source_timestamp_ns": source_timestamp_ns,
                "pause_reason": "finish_to_anchor_complete",
                "anchor_observation_id": selected.anchor.observation_id,
                "anchor_timestamp_ns": selected.anchor.timestamp_ns,
                "endpoint_max_abs_error": pose_error,
            },
        )
        return self._select_anchor_for_restoration(
            measured=measured,
            selected=selected,
            selected_cursor=selected_cursor,
            selected_timestamp_ns=paused_timestamp_ns,
            endpoint_tolerance=endpoint_tolerance,
        )

    def observe_rollback_progress(
        self,
        *,
        measured_state_24d: Any,
        source_timestamp_ns: int | None,
        cursor: int,
        observed_timestamp_ns: int,
        endpoint_tolerance: float = 0.01,
        source_tolerance_ns: int = 100_000_000,
    ) -> tuple[str, ...]:
        """Detect each inference-anchor crossing in memory at most once."""

        if self.state != InterventionState.ROLLBACK_EXEC:
            return ()
        measured = np.asarray(measured_state_24d, dtype=np.float32)
        if measured.shape != (24,) or not np.isfinite(measured).all():
            raise ValueError("Rollback progress requires a finite measured [24] state")
        self.rollback_cursor = int(cursor)
        if source_timestamp_ns is not None:
            self.current_history_timestamp_ns = source_timestamp_ns
        crossed = []
        # Iterate in physical rewind order.  The dictionary is populated from
        # chronological history, which is the opposite order of rollback.
        ordered = sorted(
            (selection for selection, _segment in self.anchor_segments.values()),
            key=lambda selection: selection.anchor.timestamp_ns,
            reverse=True,
        )
        for selection in ordered:
            if self._queue_anchor_crossing(
                selection=selection,
                measured=measured,
                source_timestamp_ns=source_timestamp_ns,
                cursor=self.rollback_cursor,
                observed_timestamp_ns=observed_timestamp_ns,
                endpoint_tolerance=endpoint_tolerance,
                source_tolerance_ns=source_tolerance_ns,
            ):
                crossed.append(selection.anchor.observation_id)
        return tuple(crossed)

    def _queue_anchor_crossing(
        self,
        *,
        selection: RollbackAnchorSelection,
        measured: np.ndarray,
        source_timestamp_ns: int | None,
        cursor: int,
        observed_timestamp_ns: int,
        endpoint_tolerance: float,
        source_tolerance_ns: int,
    ) -> bool:
        """Record one crossing in memory without persistence or image I/O."""

        observation_id = selection.anchor.observation_id
        if observation_id in self.crossed_anchor_ids or source_timestamp_ns is None:
            return False
        if abs(int(source_timestamp_ns) - selection.anchor.timestamp_ns) > source_tolerance_ns:
            return False
        pose_error = float(
            np.max(
                np.abs(
                    measured[: self.native_action_dim] - selection.anchor.state_24d[: self.native_action_dim]
                )
            )
        )
        if pose_error > endpoint_tolerance:
            return False
        self.crossed_anchor_ids.add(observation_id)
        self.last_crossed_anchor_id = observation_id
        pending = getattr(self, "_pending_anchor_crossings", None)
        if pending is None:
            # Compatibility for dependency-injected tests constructing the
            # controller with __new__ instead of running __init__.
            pending = []
            self._pending_anchor_crossings = pending
        pending.append(
            {
                "vendor_timestamp_ns": int(observed_timestamp_ns),
                "observation_id": observation_id,
                "payload": {
                    "anchor_timestamp_ns": selection.anchor.timestamp_ns,
                    "rollback_cursor": int(cursor),
                    "source_timestamp_ns": int(source_timestamp_ns),
                    "pose_max_abs_error": pose_error,
                },
            }
        )
        return True

    def _flush_pending_anchor_crossings(self) -> None:
        """Persist queued crossings only after physical rollback is stationary."""

        pending = getattr(self, "_pending_anchor_crossings", None)
        if not pending:
            return
        assert self.intervention_id is not None
        while pending:
            record = pending[0]
            self.coordinator.record_paper_phase(
                "RollbackAnchorCrossed",
                intervention_id=self.intervention_id,
                vendor_timestamp_ns=record["vendor_timestamp_ns"],
                observation_id=record["observation_id"],
                payload=record["payload"],
                control_owner="rollback",
            )
            pending.pop(0)

    def planning_failed(self, *, reason: str, timestamp_ns: int) -> None:
        if self.state not in {
            InterventionState.ROLLBACK_PLANNING,
            InterventionState.ROLLBACK_READY,
            InterventionState.ROLLBACK_PAUSED,
            InterventionState.ROLLBACK_EXEC,
        }:
            return
        assert self.intervention_id is not None
        self.coordinator.record_paper_phase(
            "RollbackPlanningFailed",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=timestamp_ns,
            observation_id=None if self.selection is None else self.selection.anchor.observation_id,
            payload={"reason": reason, "rollback_cursor": self.rollback_cursor},
        )
        self.state = InterventionState.REJECTED

    def complete_rollback(
        self,
        *,
        measured_state_24d: Any,
        completed_timestamp_ns: int,
        endpoint_tolerance: float = 0.01,
    ) -> SceneRestorationSnapshot:
        if self.state != InterventionState.ROLLBACK_EXEC or self.rollback_plan is None:
            raise RuntimeError("Rollback completion is out of order")
        measured = np.asarray(measured_state_24d, dtype=np.float32)
        if measured.shape != (24,) or not np.isfinite(measured).all():
            raise ValueError("Measured robot state did not reach the rollback anchor")
        endpoint = self.rollback_plan.waypoints_24d[-1]
        error = float(np.max(np.abs(measured[: self.native_action_dim] - endpoint[: self.native_action_dim])))
        if error > endpoint_tolerance:
            raise ValueError("Measured robot state did not reach the rollback anchor")
        self.rollback_cursor = len(self.rollback_plan.waypoints_24d) - 1
        endpoint_entry = self.anchor_segments.get(self.rollback_plan.observation_id)
        if endpoint_entry is None:
            raise ValueError("Rollback endpoint has no matching frozen inference anchor")
        self.selection, self.segment = endpoint_entry
        self.current_history_timestamp_ns = self.selection.anchor.timestamp_ns
        # A few narrow unit-test doubles construct this controller with
        # ``__new__`` to isolate endpoint math. Production instances always
        # initialize the crossing queue in ``__init__``.
        if hasattr(self, "crossed_anchor_ids"):
            self._queue_anchor_crossing(
                selection=self.selection,
                measured=measured,
                source_timestamp_ns=self.selection.anchor.timestamp_ns,
                cursor=self.rollback_cursor,
                observed_timestamp_ns=completed_timestamp_ns,
                endpoint_tolerance=endpoint_tolerance,
                source_tolerance_ns=0,
            )
            self._flush_pending_anchor_crossings()
        return self._complete_selected_anchor(
            measured=measured,
            completed_timestamp_ns=completed_timestamp_ns,
            endpoint_tolerance=endpoint_tolerance,
        )

    def _select_anchor_for_restoration(
        self,
        *,
        measured: np.ndarray,
        selected: RollbackAnchorSelection,
        selected_cursor: int,
        selected_timestamp_ns: int,
        endpoint_tolerance: float,
    ) -> SceneRestorationSnapshot:
        """Select and display an anchor without making the choice irreversible."""

        error = float(
            np.max(
                np.abs(
                    measured[: self.native_action_dim] - selected.anchor.state_24d[: self.native_action_dim]
                )
            )
        )
        if error > endpoint_tolerance:
            raise ValueError("Measured robot state did not reach the selected rollback anchor")
        assert self.intervention_id is not None
        self.selection, self.segment = self.anchor_segments[selected.anchor.observation_id]
        self.rollback_cursor = int(selected_cursor)
        self.current_history_timestamp_ns = selected.anchor.timestamp_ns
        self.last_crossed_anchor_id = selected.anchor.observation_id
        self.anchor_selected_for_restoration = True
        self.anchor_selected_timestamp_ns = int(selected_timestamp_ns)
        self.state = InterventionState.ROLLBACK_PAUSED
        self.coordinator.append_logical_input(
            LogicalInput(
                "rollback_anchor_selected",
                selected_timestamp_ns,
                {
                    "intervention_id": self.intervention_id,
                    "observation_id": selected.anchor.observation_id,
                    "rollback_cursor": self.rollback_cursor,
                    "endpoint_max_abs_error": error,
                    "selection_reopenable": self.can_continue_selected_rollback,
                },
            ),
            control_owner="hold",
        )
        snapshot = self._snapshot(selected.anchor)
        if self.feedback_writer is not None:
            self.feedback_writer.publish_snapshot(
                snapshot,
                intervention_id=self.intervention_id,
                phase="scene_restore",
            )
        return snapshot

    def finalize_selected_anchor(self, *, completed_timestamp_ns: int) -> SceneRestorationSnapshot:
        """Commit the displayed anchor when scene verification actually begins."""

        if self.state != InterventionState.ROLLBACK_PAUSED or not self.anchor_selected_for_restoration:
            raise RuntimeError("No provisional rollback anchor is ready for finalization")
        assert self.selection is not None
        physical_completion_ns = (
            completed_timestamp_ns
            if self.anchor_selected_timestamp_ns is None
            else self.anchor_selected_timestamp_ns
        )
        measured = self.selection.anchor.state_24d
        return self._complete_selected_anchor(
            measured=measured,
            completed_timestamp_ns=physical_completion_ns,
            endpoint_tolerance=0.0,
        )

    def _complete_selected_anchor(
        self,
        *,
        measured: np.ndarray,
        completed_timestamp_ns: int,
        endpoint_tolerance: float,
    ) -> SceneRestorationSnapshot:
        assert self.intervention_id is not None and self.selection is not None
        error = float(
            np.max(
                np.abs(
                    measured[: self.native_action_dim]
                    - self.selection.anchor.state_24d[: self.native_action_dim]
                )
            )
        )
        if error > endpoint_tolerance:
            raise ValueError("Measured robot state did not reach the selected rollback anchor")
        self.coordinator.record_paper_phase(
            "RollbackCompleted",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=completed_timestamp_ns,
            observation_id=self.selection.anchor.observation_id,
            payload={
                "endpoint_max_abs_error": error,
                "selected_anchor_timestamp_ns": self.selection.anchor.timestamp_ns,
                "rollback_cursor": self.rollback_cursor,
                "rollback_session_id": self.rollback_session_id,
                "history_snapshot_id": self.history_snapshot_id,
            },
        )
        self.coordinator.record_paper_phase(
            "SceneRestorationStarted",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=completed_timestamp_ns,
            observation_id=self.selection.anchor.observation_id,
            payload={"display": "exact_historical_three_camera_observation"},
        )
        self.state = InterventionState.SCENE_RESTORE
        self.anchor_selected_for_restoration = False
        self.anchor_selected_timestamp_ns = None
        snapshot = self._snapshot(self.selection.anchor)
        if self.feedback_writer is not None:
            self.feedback_writer.publish_snapshot(
                snapshot,
                intervention_id=self.intervention_id,
                phase="scene_restore",
            )
        return snapshot

    def invalidate_rollback(self, *, reason: str, timestamp_ns: int) -> None:
        if self.state not in {
            InterventionState.ROLLBACK_PLANNING,
            InterventionState.ROLLBACK_READY,
            InterventionState.ROLLBACK_EXEC,
            InterventionState.ROLLBACK_PAUSED,
            # Abandoning before scene verification or Human correction makes
            # the pair ineligible; normal session closure comes from the
            # external tablet recorder lifecycle.
            InterventionState.SCENE_RESTORE,
            InterventionState.CORRECTION_READY,
        }:
            return
        assert self.intervention_id is not None
        self.coordinator.record_paper_phase(
            "RollbackInvalidated",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=timestamp_ns,
            observation_id=None if self.selection is None else self.selection.anchor.observation_id,
            payload={"reason": reason, "rollback_cursor": self.rollback_cursor},
        )
        self.state = InterventionState.REJECTED

    def _snapshot(self, anchor: InferenceAnchor) -> SceneRestorationSnapshot:
        writer = self.coordinator.writer
        if writer is None:
            raise RuntimeError("FlowPRO coordinator lost its active writer")
        return load_scene_restoration_snapshot(writer.open_directory, anchor.observation_id)

    def verify_scene(
        self,
        *,
        valid: bool,
        verified_timestamp_ns: int,
        verification_method: str = "explicit",
    ) -> None:
        if self.state != InterventionState.SCENE_RESTORE:
            raise RuntimeError("Scene verification is out of order")
        assert self.intervention_id is not None and self.selection is not None
        self.coordinator.record_paper_phase(
            "SceneRestorationVerified",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=verified_timestamp_ns,
            observation_id=self.selection.anchor.observation_id,
            payload={
                "scene_valid": bool(valid),
                "operator_confirmed": True,
                "verification_method": str(verification_method),
            },
        )
        self.state = InterventionState.CORRECTION_READY if valid else InterventionState.REJECTED
        if self.feedback_writer is not None:
            self.feedback_writer.publish_verification(
                intervention_id=self.intervention_id,
                observation_id=self.selection.anchor.observation_id,
                valid=valid,
                verified_timestamp_ns=verified_timestamp_ns,
            )

    def start_correction(self, *, start_timestamp_ns: int) -> None:
        if self.state != InterventionState.CORRECTION_READY:
            raise RuntimeError("Human correction requires a verified restored scene")
        assert self.intervention_id is not None and self.selection is not None
        self.coordinator.record_paper_phase(
            "HumanCorrectionStarted",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=start_timestamp_ns,
            observation_id=self.selection.anchor.observation_id,
            payload={"correction_start_ns": start_timestamp_ns},
            control_owner="human",
        )
        self.correction_start_ns = start_timestamp_ns
        self.state = InterventionState.CORRECTION

    def complete_correction(
        self,
        *,
        end_timestamp_ns: int,
        sensor_skew_ok: bool,
        same_anchor: bool,
        controller_active_arms: tuple[bool, bool] | None = None,
    ) -> None:
        if self.state != InterventionState.CORRECTION or self.correction_start_ns is None:
            raise RuntimeError("Human correction completion is out of order")
        if end_timestamp_ns <= self.correction_start_ns:
            raise ValueError("Human correction end must follow its start")
        assert self.intervention_id is not None and self.selection is not None
        if controller_active_arms is None:
            controller_activity = {"left": False, "right": False}
        elif (
            not isinstance(controller_active_arms, tuple)
            or len(controller_active_arms) != 2
            or any(not isinstance(value, bool) for value in controller_active_arms)
        ):
            raise ValueError("controller_active_arms must be (left, right) booleans")
        else:
            controller_activity = {
                "left": controller_active_arms[0],
                "right": controller_active_arms[1],
            }
        self.coordinator.record_paper_phase(
            "HumanCorrectionCompleted",
            intervention_id=self.intervention_id,
            vendor_timestamp_ns=end_timestamp_ns,
            observation_id=self.selection.anchor.observation_id,
            payload={
                "correction_start_ns": self.correction_start_ns,
                "correction_end_ns": end_timestamp_ns,
                "scene_verified": True,
                "sensor_skew_ok": bool(sensor_skew_ok),
                "same_anchor": bool(same_anchor),
                "executed_segment_event_required": True,
                "controller_active_arms": controller_activity,
            },
            control_owner="hold",
        )
        self.state = (
            InterventionState.COMPLETE if sensor_skew_ok and same_anchor else InterventionState.REJECTED
        )
