"""Fail-closed capture of one live G2A inference transaction into a FlowPRO sidecar."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from ..common.sidecar import CAMERA_SOURCES
from .executed import FlowPROExecutionHistory, InferenceAnchor
from .supervisor import (
    EpisodeCoordinator,
    HumanHandbackEvidence,
    LogicalInput,
    RolloutSupervisor,
    SupervisorMode,
    SupervisorTransition,
)


@dataclass(frozen=True)
class CapturedInference:
    observation_id: str
    candidate_id: str
    generation_id: str
    request_id: str


def validate_native_candidate_response(
    response: dict[str, Any],
    *,
    request_id: str,
    native_action_dim: int,
    chunk_size: int,
    max_action_dim: int = 32,
) -> np.ndarray:
    if response.get("request_id") != request_id:
        raise ValueError("Policy response request_id differs from the captured request")
    if response.get("native_action_space") != "absolute_unnormalized":
        raise ValueError("Policy response is missing absolute unnormalized native actions")
    if response.get("native_action_dim") != native_action_dim:
        raise ValueError("Policy response native action dimension differs from the session contract")
    actions = np.asarray(response.get("native_actions"), dtype=np.float32)
    if actions.shape != (chunk_size, native_action_dim) or not np.isfinite(actions).all():
        raise ValueError(
            f"Policy native candidate must be finite [{chunk_size},{native_action_dim}], got {actions.shape}"
        )
    payload = np.ascontiguousarray(actions.astype("<f4", copy=False)).tobytes(order="C")
    if hashlib.sha256(payload).hexdigest() != response.get("native_actions_sha256"):
        raise ValueError("Policy native candidate content hash differs from response metadata")
    sampling = response.get("sampling")
    if sampling is not None and not isinstance(sampling, dict):
        raise ValueError("Policy sampling provenance must be an object when supplied")
    return actions.copy()


class FlowPROInferenceCapture:
    """Bridge an already-active EpisodeCoordinator to robot-client inference."""

    def __init__(
        self,
        coordinator: EpisodeCoordinator,
        *,
        native_action_dim: int,
        chunk_size: int,
        max_action_dim: int = 32,
        execution_history: FlowPROExecutionHistory | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.native_action_dim = native_action_dim
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.execution_history = execution_history
        self._active_generation_id: str | None = None
        # Generation changes must stay responsive to X/Y/Human safety input.
        # In particular, this lock is never held while the coordinator writes
        # images or sidecar records.  The monotonically increasing epoch lets
        # the final, short append critical section detect an invalidate/rebegin
        # cycle even if a caller were to reuse the same UUID.
        self._generation_epoch = 0
        self._generation_lock = threading.RLock()

    @property
    def active(self) -> bool:
        with self._generation_lock:
            return self._active_generation_id is not None

    @property
    def active_generation_id(self) -> str | None:
        with self._generation_lock:
            return self._active_generation_id

    def validate_server_metadata(self, metadata: dict[str, Any]) -> None:
        capability = metadata.get("action_capture")
        if not isinstance(capability, dict) or capability.get("available") is not True:
            raise ValueError("Policy server does not advertise native action capture")
        if capability.get("native_action_space") != "absolute_unnormalized":
            raise ValueError("Policy server native action space is incompatible")
        if capability.get("native_action_dim") != self.native_action_dim:
            raise ValueError("Policy server native action dimension is incompatible")
        if metadata.get("chunk_size") != self.chunk_size:
            raise ValueError("Policy server chunk size is incompatible")
        if capability.get("content_sha256") is not True:
            raise ValueError("Policy server does not provide native action content hashes")

    def begin_generation(self, generation_id: str | None = None) -> str:
        generation_id = str(generation_id or uuid4())
        try:
            UUID(generation_id)
        except ValueError as error:
            raise ValueError("FlowPRO generation_id must be a UUID") from error
        with self._generation_lock:
            if self._active_generation_id is not None:
                raise RuntimeError("FlowPRO capture already has an active generation")
            self._generation_epoch += 1
            self._active_generation_id = generation_id
            return self._active_generation_id

    def invalidate_generation(self) -> None:
        with self._generation_lock:
            self._generation_epoch += 1
            self._active_generation_id = None

    def _generation_snapshot(self, generation_id: str) -> int | None:
        """Return the matching epoch, or ``None`` when this request is stale."""

        with self._generation_lock:
            if self._active_generation_id != generation_id:
                return None
            return self._generation_epoch

    def _generation_matches(self, generation_id: str, epoch: int) -> bool:
        with self._generation_lock:
            return self._active_generation_id == generation_id and self._generation_epoch == epoch

    def capture_inference(
        self,
        *,
        request: dict[str, Any],
        response: dict[str, Any],
        generation_id: str,
        control_owner: str = "hold",
    ) -> CapturedInference | None:
        try:
            UUID(generation_id)
        except ValueError as error:
            raise ValueError("FlowPRO generation_id must be a UUID") from error
        generation_epoch = self._generation_snapshot(generation_id)
        if generation_epoch is None:
            # This is a normal safety race: X/Y/Human invalidated the request
            # while it was in flight.  It must not be turned into a reconnect.
            return None
        request_id = str(request.get("request_id", ""))
        if not request_id:
            raise ValueError("FlowPRO inference capture requires request_id")
        observation = request.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("FlowPRO inference capture requires the exact request observation")
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (24,) or not np.isfinite(state).all():
            raise ValueError("FlowPRO capture requires the exact finite canonical state24")
        task = observation.get("task")
        if not isinstance(task, str) or len(task) > 4096:
            raise ValueError("FlowPRO capture requires the exact task prompt")
        images = observation.get("images")
        timestamps = observation.get("camera_frame_wall_time_ns")
        anchor_timestamp_ns = observation.get("state_capture_wall_time_ns")
        if not isinstance(images, dict) or set(images) != set(CAMERA_SOURCES):
            raise ValueError("FlowPRO capture requires the exact three encoded request images")
        if not isinstance(timestamps, dict) or set(timestamps) != set(CAMERA_SOURCES):
            raise ValueError("FlowPRO capture requires all camera frame timestamps")
        if isinstance(anchor_timestamp_ns, bool) or not isinstance(anchor_timestamp_ns, int):
            raise ValueError("FlowPRO capture requires state_capture_wall_time_ns")
        image_records: dict[str, dict[str, Any]] = {}
        for camera_key, image in images.items():
            if not isinstance(image, dict) or image.get("encoding") != "jpeg":
                raise ValueError("FlowPRO v1 capture requires JPEG transport for every camera")
            image_records[camera_key] = {
                "data": bytes(image["data"]),
                "vendor_timestamp_ns": int(timestamps[camera_key]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }

        candidate = validate_native_candidate_response(
            response,
            request_id=request_id,
            native_action_dim=self.native_action_dim,
            chunk_size=self.chunk_size,
            max_action_dim=self.max_action_dim,
        )
        observation_id = str(uuid4())
        candidate_id = str(uuid4())
        if not self._generation_matches(generation_id, generation_epoch):
            return None
        self.coordinator.record_observation(
            observation_id=observation_id,
            anchor_timestamp_ns=anchor_timestamp_ns,
            state_24d=state,
            task=task,
            images=image_records,
            control_owner=control_owner,
        )
        if not self._generation_matches(generation_id, generation_epoch):
            return None
        self.coordinator.record_policy_candidate(
            candidate,
            observation_id=observation_id,
            request_id=request_id,
            candidate_id=candidate_id,
            generation_id=generation_id,
            control_owner=control_owner,
            sampling=dict(response.get("sampling") or {}),
        )
        anchor = InferenceAnchor(
            observation_id=observation_id,
            generation_id=generation_id,
            timestamp_ns=anchor_timestamp_ns,
            state_24d=state,
            camera_timestamps_ns={key: int(timestamps[key]) for key in CAMERA_SOURCES},
        )
        # Only the final in-memory append is serialized with generation
        # changes.  Sidecar/image I/O above intentionally remains outside the
        # lock so X can invalidate promptly.  If it did, no stale anchor is
        # admitted to rollback history and the client discards the response.
        with self._generation_lock:
            if self._active_generation_id != generation_id or self._generation_epoch != generation_epoch:
                return None
            if self.execution_history is not None:
                self.execution_history.append_anchor(anchor)
            return CapturedInference(
                observation_id=observation_id,
                candidate_id=candidate_id,
                generation_id=generation_id,
                request_id=request_id,
            )


class FlowPROSessionController:
    """Transactionally bind VR control intent, one tablet episode, and generations.

    This class coordinates durable data-plane state only.  It does not publish
    DDS messages, write GDK commands, or claim physical ownership.
    """

    def __init__(
        self,
        coordinator: EpisodeCoordinator,
        capture: FlowPROInferenceCapture,
        *,
        supervisor: RolloutSupervisor | None = None,
    ) -> None:
        if capture.coordinator is not coordinator:
            raise ValueError("FlowPRO capture and session controller must share one coordinator")
        self.coordinator = coordinator
        self.capture = capture
        self.supervisor = supervisor or RolloutSupervisor()
        self.flowpro_session_id: str | None = None

    def start_session(
        self,
        vendor_episode_path: Path,
        logical_input: LogicalInput,
        *,
        flowpro_session_id: str | None = None,
    ) -> SupervisorTransition:
        if logical_input.name not in {"flowpro_mode_toggle", "y_press"}:
            raise ValueError("A FlowPRO session must start from y_press")
        if self.supervisor.mode != SupervisorMode.DISABLED or self.flowpro_session_id is not None:
            raise RuntimeError("FlowPRO session is already active")
        session_id = self.coordinator.attach_active_recording(
            Path(vendor_episode_path),
            flowpro_session_id=flowpro_session_id,
        )
        transition = self.supervisor.handle(logical_input.name)
        expected_mode = (
            SupervisorMode.AUTO
            if logical_input.name == "y_press"
            else SupervisorMode.HOLD
        )
        if transition is None or transition.current != expected_mode:
            raise RuntimeError(f"FlowPRO supervisor did not enter {expected_mode.value} after session attach")
        self.flowpro_session_id = session_id
        self.coordinator.append_logical_input(logical_input, control_owner=SupervisorMode.HOLD.value)
        self._sync_generation(transition)
        return transition

    def fail_closed_hold(self, *, reason: str, timestamp_ns: int) -> SupervisorTransition | None:
        """Persist an internal safety failure and invalidate the active generation."""

        if reason not in {"vr_stale", "watchdog", "network_lost", "motion_unhealthy"}:
            raise ValueError(f"Unsupported fail-closed HOLD reason: {reason!r}")
        logical_input = LogicalInput(
            name=reason,
            source_timestamp_ns=timestamp_ns,
            detail={"source": "g2_runtime"},
        )
        self.coordinator.append_logical_input(
            logical_input,
            control_owner=self._control_owner(),
        )
        transition = self.supervisor.handle(reason)
        if transition is not None:
            self._sync_generation(transition)
        return transition

    def finalize_abort(self, *, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        """Durably end an already-safe ABORT session without touching the tablet recorder."""

        if self.flowpro_session_id is None:
            raise RuntimeError("FlowPRO session is not active")
        if self.supervisor.mode != SupervisorMode.ABORT:
            raise RuntimeError("finalize_abort requires the supervisor to be in ABORT")
        self.capture.invalidate_generation()
        state = self.coordinator.finish_control(outcome="aborted", outcome_detail=detail or {})
        self.flowpro_session_id = None
        return state

    def finish_from_tablet(self, *, timestamp_ns: int) -> SupervisorTransition:
        """Durably close a HOLD session after the tablet publishes final meta."""

        if self.flowpro_session_id is None:
            raise RuntimeError("FlowPRO session is not active")
        if self.supervisor.mode != SupervisorMode.HOLD:
            raise RuntimeError("Tablet completion may finalize FlowPRO only from HOLD")
        logical_input = LogicalInput(
            "tablet_recording_finished",
            timestamp_ns,
            {"outcome_authority": "external_tablet_recorder"},
        )
        self.coordinator.append_logical_input(logical_input, control_owner=SupervisorMode.HOLD.value)
        self.coordinator.finish_control(
            outcome="pending_recorder_commit",
            outcome_detail={
                "outcome_authority": "external_tablet_recorder",
                "saved_vendor_episode_outcome": "success",
            },
        )
        transition = self.supervisor.handle(logical_input.name)
        if transition is None or transition.current != SupervisorMode.DISABLED:
            raise RuntimeError("FlowPRO supervisor did not exit after tablet completion")
        self._sync_generation(transition)
        self.flowpro_session_id = None
        return transition

    def finish_for_replacement(self, *, timestamp_ns: int, next_record_name: str) -> SupervisorTransition:
        """Exit HOLD when a new tablet clip replaces an inactive, unconfirmed one."""

        if self.flowpro_session_id is None:
            raise RuntimeError("FlowPRO session is not active")
        if self.supervisor.mode != SupervisorMode.HOLD:
            raise RuntimeError("Tablet replacement may exit FlowPRO only from HOLD")
        logical_input = LogicalInput(
            "tablet_recording_replaced",
            timestamp_ns,
            {"source": "tablet_recording_replaced", "next_record_name": next_record_name},
        )
        self.coordinator.append_logical_input(logical_input, control_owner=SupervisorMode.HOLD.value)
        self.coordinator.finish_control(
            outcome="pending_recorder_commit",
            outcome_detail={
                "outcome_authority": "external_tablet_recorder",
                "completion_evidence": "new_recording_after_previous_inactive",
                "next_record_name": next_record_name,
            },
        )
        transition = self.supervisor.handle(logical_input.name)
        if transition is None or transition.current != SupervisorMode.DISABLED:
            raise RuntimeError("FlowPRO supervisor did not exit after tablet replacement")
        self._sync_generation(transition)
        self.flowpro_session_id = None
        return transition

    def handle(
        self,
        logical_input: LogicalInput,
        *,
        handback_evidence: HumanHandbackEvidence | None = None,
    ) -> SupervisorTransition | None:
        if self.flowpro_session_id is None:
            raise RuntimeError("FlowPRO session is not active")
        exiting = logical_input.name == "flowpro_mode_toggle" and self.supervisor.mode == SupervisorMode.HOLD
        if exiting:
            self.coordinator.append_logical_input(
                logical_input,
                control_owner=self._control_owner(),
            )
            # Persist AWAITING_TABLET_SAVE before changing the in-memory mode.
            # If this durable write fails, the supervisor remains in HOLD and
            # the operator can retry without reopening policy execution.
            self.coordinator.finish_control(
                outcome="pending_recorder_commit",
                outcome_detail={
                    "outcome_authority": "external_tablet_recorder",
                    "saved_vendor_episode_outcome": "success",
                },
            )
            transition = self.supervisor.handle(logical_input.name)
            if transition is None or transition.current != SupervisorMode.DISABLED:
                raise RuntimeError("FlowPRO supervisor did not exit after durable session finish")
            self._sync_generation(transition)
            self.flowpro_session_id = None
            return transition

        self.coordinator.append_logical_input(
            logical_input,
            control_owner=self._control_owner(),
        )
        transition = self.supervisor.handle(
            logical_input.name,
            handback_evidence=handback_evidence,
        )
        if transition is None:
            return None
        self._sync_generation(transition)
        return transition

    def _control_owner(self) -> str:
        """Map UI/supervisor modes onto the durable control-authority contract."""

        if self.supervisor.mode in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            return "policy"
        if self.supervisor.mode == SupervisorMode.HUMAN:
            return "human"
        if self.supervisor.mode == SupervisorMode.ROLLBACK_EXEC:
            return "rollback"
        return "hold"

    def _sync_generation(self, transition: SupervisorTransition) -> None:
        if transition.generation_invalidated:
            self.capture.invalidate_generation()
        if transition.current in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
            if transition.generation_id is None:
                raise RuntimeError("Policy execution transition is missing generation_id")
            self.capture.begin_generation(transition.generation_id)
