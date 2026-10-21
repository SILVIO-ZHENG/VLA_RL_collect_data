"""Fail-closed validators shared by FlowPRO collection and offline ingestion.

This module deliberately has no LeRobot, GDK, numpy, or JSON-Schema runtime
dependency so the same checks can later run on the G2 sidecar sealer and on the
5090 ingestion service.  JSON Schema files remain the interchange definition;
these checks enforce the cross-field invariants that JSON Schema expresses
poorly (path binding, timestamp ordering, tensor-shape equality, and lineage).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

MANIFEST_SCHEMA_VERSION = "flowpro.sidecar_manifest/v1"
EVENT_SCHEMA_VERSION = "flowpro.event/v1"
PAIR_SCHEMA_VERSION = "flowpro.preference_pair/v1"
DATA_CONTRACT_VERSION = "flowpro.data_contract/v1"

ACTION_DIMS = (16, 24)
ACTION_SPACE = "absolute_unnormalized"
MODEL_FPS = 30  # Legacy default for archived v1 sidecars.
MAX_CAMERA_SKEW_MS = 40.0
EFFECTOR_PROFILES = ("dh", "dahuan", "omnipicker_legacy")
JOIN_CLOCK = "vendor_timestamp_ns"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOINABLE_EVENT_TYPES = {
    "ObservationCaptured",
    "ControlOwnershipChanged",
    "VRTeleopFrame",
    "PlanInstalled",
    "ServoSample",
    "SafetyDecision",
    "HumanIntervention",
    "ExecutedSegmentCaptured",
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
_EVENT_TYPES = _JOINABLE_EVENT_TYPES | {
    "SessionStarted",
    "LogicalInput",
    "PolicyCandidate",
    "EpisodeOutcome",
    "DataSealed",
}
_PRODUCERS = {"g2_supervisor", "robot_client", "policy_server", "annotation"}
_OWNERS = {"policy", "human", "hold", "rollback", None}
_CAMERA_KEYS = {"cam_head", "cam_left_wrist", "cam_right_wrist"}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array")
    return value


def _required(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - value.keys())
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _reject_unknown(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    unknown = sorted(value.keys() - fields)
    if unknown:
        raise ValueError(f"{label} has unknown v1 fields: {unknown}")


def _uuid(value: Any, label: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a UUID string")
    try:
        UUID(value)
    except ValueError as error:
        raise ValueError(f"{label} is not a valid UUID: {value!r}") from error


def _sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")


def _utc_datetime(value: Any, label: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 datetime string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO-8601 datetime: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", ".."}:
        raise ValueError(f"{label} must not be absolute or contain '..': {value!r}")
    return value


def _validate_file_records(value: Any, label: str, *, minimum: int) -> None:
    records = _sequence(value, label)
    if len(records) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} file record(s)")
    seen: set[str] = set()
    allowed = {"relative_path", "size_bytes", "sha256"}
    for index, raw_record in enumerate(records):
        item_label = f"{label}[{index}]"
        record = _mapping(raw_record, item_label)
        _required(record, allowed, item_label)
        _reject_unknown(record, allowed, item_label)
        relative_path = _relative_path(record["relative_path"], f"{item_label}.relative_path")
        if relative_path in seen:
            raise ValueError(f"{label} contains duplicate path {relative_path!r}")
        seen.add(relative_path)
        _positive_int(record["size_bytes"], f"{item_label}.size_bytes", allow_zero=True)
        _sha256(record["sha256"], f"{item_label}.sha256")


def validate_manifest(raw: Any, *, vendor_record_root: Path = Path("/data/record")) -> None:
    """Validate one open or sealed sidecar manifest."""

    manifest = _mapping(raw, "manifest")
    allowed = {
        "schema_version",
        "status",
        "flowpro_session_id",
        "robot_id",
        "task_id",
        "created_at_utc",
        "sealed_at_utc",
        "clock_contract",
        "policy",
        "collection_manifest",
        "vendor_episode",
        "sidecar_files",
    }
    required = allowed - {"sealed_at_utc", "collection_manifest"}
    _required(manifest, required, "manifest")
    _reject_unknown(manifest, allowed, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema_version")
    status = manifest["status"]
    if status not in {"open", "sealed"}:
        raise ValueError("manifest.status must be 'open' or 'sealed'")
    _uuid(manifest["flowpro_session_id"], "manifest.flowpro_session_id")
    for field in ("robot_id", "task_id"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"manifest.{field} must be a non-empty string")
    _utc_datetime(manifest["created_at_utc"], "manifest.created_at_utc")
    _utc_datetime(manifest.get("sealed_at_utc"), "manifest.sealed_at_utc", nullable=status == "open")

    clock = _mapping(manifest["clock_contract"], "manifest.clock_contract")
    required_clock_fields = {"join_clock", "timestamp_unit", "monotonic_scope"}
    clock_fields = required_clock_fields | {"wall_to_vendor_offset_ns"}
    _required(clock, required_clock_fields, "manifest.clock_contract")
    _reject_unknown(clock, clock_fields, "manifest.clock_contract")
    expected_clock = {
        "join_clock": JOIN_CLOCK,
        "timestamp_unit": "ns",
        "monotonic_scope": "single_process_only",
    }
    if any(clock[key] != value for key, value in expected_clock.items()):
        raise ValueError(f"manifest.clock_contract must equal {expected_clock}")
    offset = clock.get("wall_to_vendor_offset_ns", 0)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("manifest.clock_contract.wall_to_vendor_offset_ns must be an integer")

    policy = _mapping(manifest["policy"], "manifest.policy")
    policy_fields = {
        "checkpoint_id",
        "checkpoint_stats_sha256",
        "native_action_dim",
        "action_space",
        "model_fps",
        "chunk_size",
        "effector_profile",
        "processor_version",
    }
    _required(policy, policy_fields, "manifest.policy")
    _reject_unknown(policy, policy_fields, "manifest.policy")
    if not isinstance(policy["checkpoint_id"], str) or not policy["checkpoint_id"]:
        raise ValueError("manifest.policy.checkpoint_id must be non-empty")
    _sha256(policy["checkpoint_stats_sha256"], "manifest.policy.checkpoint_stats_sha256")
    if policy["native_action_dim"] not in ACTION_DIMS:
        raise ValueError(f"manifest.policy.native_action_dim must be one of {ACTION_DIMS}")
    if policy["action_space"] != ACTION_SPACE:
        raise ValueError("manifest policy must use absolute unnormalized actions")
    if (isinstance(policy["model_fps"], bool)
            or not isinstance(policy["model_fps"], (int, float))
            or not math.isfinite(policy["model_fps"])
            or policy["model_fps"] <= 0):
        raise ValueError("manifest.policy.model_fps must be positive and finite")
    _positive_int(policy["chunk_size"], "manifest.policy.chunk_size")
    if policy["effector_profile"] not in EFFECTOR_PROFILES:
        raise ValueError(f"Unknown effector profile: {policy['effector_profile']!r}")
    if not isinstance(policy["processor_version"], str) or not policy["processor_version"]:
        raise ValueError("manifest.policy.processor_version must be non-empty")

    collection = manifest.get("collection_manifest")
    if collection is not None:
        collection = _mapping(collection, "manifest.collection_manifest")
        collection_fields = {
            "schema_version",
            "collection_round_id",
            "sha256",
            "runtime_source_sha256",
        }
        _required(collection, collection_fields, "manifest.collection_manifest")
        _reject_unknown(collection, collection_fields, "manifest.collection_manifest")
        if collection["schema_version"] != "flowpro.collection_manifest/v1":
            raise ValueError("Unsupported collection manifest schema_version")
        if not isinstance(collection["collection_round_id"], str) or not collection["collection_round_id"]:
            raise ValueError("manifest.collection_manifest.collection_round_id must be non-empty")
        _sha256(collection["sha256"], "manifest.collection_manifest.sha256")
        _sha256(
            collection["runtime_source_sha256"],
            "manifest.collection_manifest.runtime_source_sha256",
        )

    vendor = manifest["vendor_episode"]
    if vendor is None:
        if status == "sealed":
            raise ValueError("sealed manifest requires vendor_episode")
    else:
        _validate_vendor_episode(vendor, vendor_record_root=vendor_record_root)
    _validate_file_records(
        manifest["sidecar_files"], "manifest.sidecar_files", minimum=1 if status == "sealed" else 0
    )


def _validate_vendor_episode(raw: Any, *, vendor_record_root: Path) -> None:
    vendor = _mapping(raw, "manifest.vendor_episode")
    allowed = {
        "record_name",
        "episode_path",
        "source_timestamp_start_ns",
        "source_timestamp_end_ns",
        "max_camera_skew_ms",
        "meta_clip_start_time",
        "meta_clip_end_time",
        "files",
    }
    required = allowed - {"meta_clip_start_time", "meta_clip_end_time"}
    _required(vendor, required, "manifest.vendor_episode")
    _reject_unknown(vendor, allowed, "manifest.vendor_episode")
    record_name = vendor["record_name"]
    episode_path = vendor["episode_path"]
    if (
        not isinstance(record_name, str)
        or not record_name
        or "/" in record_name
        or record_name in {".", ".."}
    ):
        raise ValueError("vendor record_name must be one path component")
    if not isinstance(episode_path, str):
        raise ValueError("vendor episode_path must be a string")
    path = Path(episode_path)
    expected_root = vendor_record_root.expanduser().resolve()
    if not path.is_absolute() or path.parent.resolve() != expected_root or path.name != record_name:
        raise ValueError(f"vendor episode_path must be {expected_root}/<record_name>")
    start = _positive_int(vendor["source_timestamp_start_ns"], "vendor start timestamp")
    end = _positive_int(vendor["source_timestamp_end_ns"], "vendor end timestamp")
    if end <= start:
        raise ValueError("vendor source_timestamp_end_ns must be greater than start")
    skew = vendor["max_camera_skew_ms"]
    if isinstance(skew, bool) or not isinstance(skew, (int, float)) or not 0 <= skew <= MAX_CAMERA_SKEW_MS:
        raise ValueError(f"vendor max_camera_skew_ms must be in [0, {MAX_CAMERA_SKEW_MS}]")
    _validate_file_records(vendor["files"], "manifest.vendor_episode.files", minimum=2)


def validate_event(raw: Any) -> None:
    """Validate one append-only FlowPRO event and its join requirements."""

    event = _mapping(raw, "event")
    allowed = {
        "schema_version",
        "event_id",
        "flowpro_session_id",
        "event_type",
        "producer",
        "monotonic_ns",
        "wall_time_utc",
        "vendor_timestamp_ns",
        "observation_id",
        "request_id",
        "candidate_id",
        "generation_id",
        "action_id",
        "control_owner",
        "payload",
    }
    required = {
        "schema_version",
        "event_id",
        "flowpro_session_id",
        "event_type",
        "producer",
        "monotonic_ns",
        "wall_time_utc",
        "vendor_timestamp_ns",
        "payload",
    }
    _required(event, required, "event")
    _reject_unknown(event, allowed, "event")
    if event["schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("Unsupported event schema_version")
    _uuid(event["event_id"], "event.event_id")
    _uuid(event["flowpro_session_id"], "event.flowpro_session_id")
    event_type = event["event_type"]
    if event_type not in _EVENT_TYPES:
        raise ValueError(f"Unsupported event_type: {event_type!r}")
    if event["producer"] not in _PRODUCERS:
        raise ValueError(f"Unsupported event producer: {event['producer']!r}")
    _positive_int(event["monotonic_ns"], "event.monotonic_ns", allow_zero=True)
    _utc_datetime(event["wall_time_utc"], "event.wall_time_utc")
    vendor_timestamp = event["vendor_timestamp_ns"]
    if event_type in _JOINABLE_EVENT_TYPES or vendor_timestamp is not None:
        _positive_int(vendor_timestamp, "event.vendor_timestamp_ns")
    for field in ("observation_id", "candidate_id", "generation_id", "action_id"):
        if field in event:
            _uuid(event[field], f"event.{field}", nullable=True)
    if event.get("control_owner") not in _OWNERS:
        raise ValueError(f"Unsupported control_owner: {event.get('control_owner')!r}")
    _mapping(event["payload"], "event.payload")
    if event_type == "ObservationCaptured":
        if event.get("observation_id") is None:
            raise ValueError("ObservationCaptured requires observation_id")
        payload = _mapping(event["payload"], "ObservationCaptured payload")
        fields = {"anchor_timestamp_ns", "state", "task", "images"}
        _required(payload, fields, "ObservationCaptured payload")
        _reject_unknown(payload, fields, "ObservationCaptured payload")
        anchor_timestamp = _positive_int(
            payload["anchor_timestamp_ns"], "ObservationCaptured payload.anchor_timestamp_ns"
        )
        if anchor_timestamp != vendor_timestamp:
            raise ValueError("ObservationCaptured anchor/vendor timestamps differ")
        if not isinstance(payload["task"], str) or len(payload["task"]) > 4096:
            raise ValueError("ObservationCaptured task must be a string no longer than 4096 characters")
        state = _mapping(payload["state"], "ObservationCaptured payload.state")
        state_fields = {"relative_path", "size_bytes", "sha256", "dtype", "shape", "space"}
        _required(state, state_fields, "ObservationCaptured payload.state")
        _reject_unknown(state, state_fields, "ObservationCaptured payload.state")
        _relative_path(state["relative_path"], "ObservationCaptured payload.state.relative_path")
        if state["dtype"] != "float32" or state["shape"] != [24]:
            raise ValueError("ObservationCaptured state must be float32 [24]")
        if state["space"] != "canonical_hardware_24d":
            raise ValueError("ObservationCaptured state has an invalid space")
        if _positive_int(state["size_bytes"], "ObservationCaptured payload.state.size_bytes") != 96:
            raise ValueError("ObservationCaptured state must contain exactly 96 bytes")
        _sha256(state["sha256"], "ObservationCaptured payload.state.sha256")
        images = _mapping(payload["images"], "ObservationCaptured payload.images")
        if set(images) != _CAMERA_KEYS:
            raise ValueError("ObservationCaptured must contain the exact three inference cameras")
        image_fields = {
            "relative_path",
            "size_bytes",
            "sha256",
            "encoding",
            "camera_key",
            "vendor_timestamp_ns",
            "width",
            "height",
        }
        for camera_key, raw_image in images.items():
            image = _mapping(raw_image, f"ObservationCaptured payload.images.{camera_key}")
            _required(image, image_fields, f"ObservationCaptured payload.images.{camera_key}")
            _reject_unknown(image, image_fields, f"ObservationCaptured payload.images.{camera_key}")
            if image["camera_key"] != camera_key or image["encoding"] != "jpeg":
                raise ValueError("ObservationCaptured image camera_key/encoding is invalid")
            _relative_path(
                image["relative_path"],
                f"ObservationCaptured payload.images.{camera_key}.relative_path",
            )
            _positive_int(image["size_bytes"], f"ObservationCaptured payload.images.{camera_key}.size_bytes")
            _sha256(image["sha256"], f"ObservationCaptured payload.images.{camera_key}.sha256")
            image_timestamp = _positive_int(
                image["vendor_timestamp_ns"],
                f"ObservationCaptured payload.images.{camera_key}.vendor_timestamp_ns",
            )
            if abs(image_timestamp - anchor_timestamp) > 40_000_000:
                raise ValueError("ObservationCaptured image exceeds 40 ms anchor skew")
            _positive_int(image["width"], f"ObservationCaptured payload.images.{camera_key}.width")
            _positive_int(image["height"], f"ObservationCaptured payload.images.{camera_key}.height")
    if event_type == "PolicyCandidate":
        for field in ("observation_id", "candidate_id", "generation_id"):
            if event.get(field) is None:
                raise ValueError(f"PolicyCandidate requires {field}")
        if not isinstance(event.get("request_id"), str) or not event["request_id"]:
            raise ValueError("PolicyCandidate requires request_id")
        payload = event["payload"]
        if payload.get("native_action_dim") not in ACTION_DIMS:
            raise ValueError("PolicyCandidate payload has invalid native_action_dim")
        if payload.get("action_space") != ACTION_SPACE:
            raise ValueError("PolicyCandidate payload must be absolute_unnormalized")
        shape = _sequence(payload.get("shape"), "PolicyCandidate payload.shape")
        if len(shape) != 2 or shape[-1] != payload["native_action_dim"]:
            raise ValueError("PolicyCandidate shape must be [T, native_action_dim]")
        if payload.get("dtype") != "float32":
            raise ValueError("PolicyCandidate payload dtype must be float32")
        _relative_path(payload.get("relative_path"), "PolicyCandidate payload.relative_path")
        _positive_int(payload.get("size_bytes"), "PolicyCandidate payload.size_bytes")
        _sha256(payload.get("sha256"), "PolicyCandidate payload.sha256")
    if event_type == "HumanIntervention" and event["payload"].get("phase") == "finalized":
        payload = event["payload"]
        required = {
            "intervention_id",
            "observation_id",
            "candidate_id",
            "correction_start_ns",
            "correction_end_ns",
            "scene_verified",
            "sensor_skew_ok",
            "same_anchor",
            "pair_reject",
        }
        _required(payload, required, "HumanIntervention payload")
        for field in ("intervention_id", "observation_id", "candidate_id"):
            _uuid(payload[field], f"HumanIntervention payload.{field}")
        if event.get("observation_id") != payload["observation_id"]:
            raise ValueError("HumanIntervention top-level/payload observation_id mismatch")
        if event.get("candidate_id") != payload["candidate_id"]:
            raise ValueError("HumanIntervention top-level/payload candidate_id mismatch")
        start = _positive_int(payload["correction_start_ns"], "correction_start_ns")
        end = _positive_int(payload["correction_end_ns"], "correction_end_ns")
        if end <= start:
            raise ValueError("correction_end_ns must be greater than correction_start_ns")
        for field in ("scene_verified", "sensor_skew_ok", "same_anchor", "pair_reject"):
            if not isinstance(payload[field], bool):
                raise ValueError(f"HumanIntervention payload.{field} must be boolean")
    if event_type == "ExecutedSegmentCaptured":
        for field in ("observation_id", "candidate_id", "generation_id"):
            if event.get(field) is None:
                raise ValueError(f"ExecutedSegmentCaptured requires {field}")
        payload = _mapping(event["payload"], "ExecutedSegmentCaptured payload")
        required = {
            "protocol",
            "intervention_id",
            "failure_timestamp_ns",
            "requested_horizon_ns",
            "requested_target_timestamp_ns",
            "actual_anchor_timestamp_ns",
            "snap_offset_ns",
            "sample_count",
            "native_action_dim",
            "artifacts",
        }
        _required(payload, required, "ExecutedSegmentCaptured payload")
        if payload["protocol"] != "flowpro.executed_trajectory/v2":
            raise ValueError("ExecutedSegmentCaptured protocol must be executed_trajectory/v2")
        _uuid(payload["intervention_id"], "ExecutedSegmentCaptured intervention_id")
        for field in (
            "failure_timestamp_ns",
            "requested_horizon_ns",
            "requested_target_timestamp_ns",
            "actual_anchor_timestamp_ns",
            "sample_count",
        ):
            _positive_int(payload[field], f"ExecutedSegmentCaptured payload.{field}")
        _positive_int(
            payload["snap_offset_ns"],
            "ExecutedSegmentCaptured payload.snap_offset_ns",
            allow_zero=True,
        )
        if payload["native_action_dim"] not in ACTION_DIMS:
            raise ValueError("ExecutedSegmentCaptured has invalid native_action_dim")
        artifacts = _mapping(payload["artifacts"], "ExecutedSegmentCaptured artifacts")
        required_artifacts = {
            "timestamps_ns",
            "measured_model_actions",
            "measured_state_24d",
            "policy_native_actions",
            "safety_targets_24d",
            "gdk_commands_24d",
            "command_accepted",
            "safety_modified",
            "lineage",
        }
        optional_artifacts = {"execution_kind_code"}
        if not required_artifacts.issubset(artifacts) or not set(artifacts).issubset(
            required_artifacts | optional_artifacts
        ):
            raise ValueError("ExecutedSegmentCaptured artifacts are incomplete or unknown")
        for name, raw_artifact in artifacts.items():
            artifact = _mapping(raw_artifact, f"ExecutedSegmentCaptured artifacts.{name}")
            fields = {"relative_path", "size_bytes", "sha256", "dtype", "shape"}
            _required(artifact, fields, f"ExecutedSegmentCaptured artifacts.{name}")
            _reject_unknown(artifact, fields, f"ExecutedSegmentCaptured artifacts.{name}")
            _relative_path(artifact["relative_path"], f"artifact {name}.relative_path")
            _positive_int(artifact["size_bytes"], f"artifact {name}.size_bytes")
            _sha256(artifact["sha256"], f"artifact {name}.sha256")
            shape = _sequence(artifact["shape"], f"artifact {name}.shape")
            if not shape or int(shape[0]) != payload["sample_count"]:
                raise ValueError(f"artifact {name} first dimension must equal sample_count")
        if "execution_quality" in payload:
            quality = _mapping(
                payload["execution_quality"], "ExecutedSegmentCaptured payload.execution_quality"
            )
            required_quality_fields = {
                "execution_kind_counts",
                "queue_underrun_count",
                "strict_preference_eligible",
            }
            physical_quality_fields = {
                "execution_gap_count",
                "max_execution_gap_ns",
                "physical_history_complete",
            }
            allowed_quality_fields = required_quality_fields | physical_quality_fields
            _required(
                quality,
                required_quality_fields,
                "ExecutedSegmentCaptured payload.execution_quality",
            )
            _reject_unknown(
                quality,
                allowed_quality_fields,
                "ExecutedSegmentCaptured payload.execution_quality",
            )
            present_physical_fields = set(quality) & physical_quality_fields
            if present_physical_fields and present_physical_fields != physical_quality_fields:
                raise ValueError("ExecutedSegmentCaptured physical execution-quality fields must be complete")
            counts = _mapping(
                quality["execution_kind_counts"],
                "ExecutedSegmentCaptured payload.execution_quality.execution_kind_counts",
            )
            required_kinds = {"policy_action", "chunk_transition_blend", "queue_underrun_hold"}
            allowed_kinds = required_kinds | {"operator_hold"}
            if not required_kinds.issubset(counts) or not set(counts).issubset(allowed_kinds):
                raise ValueError("ExecutedSegmentCaptured execution_kind_counts are incomplete")
            for kind, count in counts.items():
                _positive_int(
                    count,
                    f"ExecutedSegmentCaptured execution_kind_counts.{kind}",
                    allow_zero=True,
                )
            if sum(int(count) for count in counts.values()) != int(payload["sample_count"]):
                raise ValueError("ExecutedSegmentCaptured execution kind counts disagree with sample_count")
            underruns = _positive_int(
                quality["queue_underrun_count"],
                "ExecutedSegmentCaptured payload.execution_quality.queue_underrun_count",
                allow_zero=True,
            )
            if underruns != counts["queue_underrun_hold"]:
                raise ValueError("ExecutedSegmentCaptured queue underrun counts disagree")
            physical_complete = True
            gap_count = 0
            if present_physical_fields:
                gap_count = _positive_int(
                    quality["execution_gap_count"],
                    "ExecutedSegmentCaptured payload.execution_quality.execution_gap_count",
                    allow_zero=True,
                )
                max_gap_ns = _positive_int(
                    quality["max_execution_gap_ns"],
                    "ExecutedSegmentCaptured payload.execution_quality.max_execution_gap_ns",
                    allow_zero=True,
                )
                physical_complete = quality["physical_history_complete"]
                if not isinstance(physical_complete, bool):
                    raise ValueError("ExecutedSegmentCaptured physical_history_complete must be boolean")
                if (gap_count == 0) != (max_gap_ns == 0):
                    raise ValueError("ExecutedSegmentCaptured execution gap metadata disagrees")
                if physical_complete != (gap_count == 0):
                    raise ValueError("ExecutedSegmentCaptured physical history completeness is inconsistent")
            eligible = quality["strict_preference_eligible"]
            expected_eligible = underruns == 0 and gap_count == 0 and physical_complete
            if not isinstance(eligible, bool) or eligible != expected_eligible:
                raise ValueError("ExecutedSegmentCaptured strict eligibility is inconsistent")
    if event_type in {
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
    }:
        payload = _mapping(event["payload"], f"{event_type} payload")
        _uuid(payload.get("intervention_id"), f"{event_type} payload.intervention_id")
        if event_type == "HumanCorrectionCompleted" and "controller_active_arms" in payload:
            controller_activity = _mapping(
                payload["controller_active_arms"],
                "HumanCorrectionCompleted payload.controller_active_arms",
            )
            _required(
                controller_activity,
                {"left", "right"},
                "HumanCorrectionCompleted payload.controller_active_arms",
            )
            _reject_unknown(
                controller_activity,
                {"left", "right"},
                "HumanCorrectionCompleted payload.controller_active_arms",
            )
            if any(not isinstance(value, bool) for value in controller_activity.values()):
                raise ValueError("HumanCorrectionCompleted controller arm activity must be boolean")


def _validate_tensor_artifact(raw: Any, label: str, *, action: bool) -> tuple[int, ...]:
    artifact = _mapping(raw, label)
    base_fields = {"relative_path", "dtype", "shape", "sha256"}
    allowed = base_fields | ({"source_event_id", "candidate_id", "source"} if action else set())
    required = base_fields | ({"source_event_id"} if action else set())
    _required(artifact, required, label)
    _reject_unknown(artifact, allowed, label)
    _relative_path(artifact["relative_path"], f"{label}.relative_path")
    expected_dtype = "float32" if action else "bool"
    if artifact["dtype"] != expected_dtype:
        raise ValueError(f"{label}.dtype must be {expected_dtype}")
    shape_values = _sequence(artifact["shape"], f"{label}.shape")
    if len(shape_values) != 2:
        raise ValueError(f"{label}.shape must contain [T, D]")
    shape = tuple(_positive_int(value, f"{label}.shape") for value in shape_values)
    _sha256(artifact["sha256"], f"{label}.sha256")
    if action:
        _uuid(artifact["source_event_id"], f"{label}.source_event_id")
        if "candidate_id" in artifact:
            _uuid(artifact["candidate_id"], f"{label}.candidate_id")
    return shape


def validate_preference_pair(raw: Any) -> None:
    """Validate pair shape/domain equality and the strict training gate."""

    pair = _mapping(raw, "pair")
    allowed = {
        "schema_version",
        "pair_id",
        "flowpro_session_id",
        "observation_id",
        "native_action_dim",
        "action_space",
        "fps",
        "effector_profile",
        "checkpoint_stats_sha256",
        "rejected",
        "preferred",
        "valid_mask",
        "same_context",
        "pair_valid",
        "invalid_reason",
        "provenance_event_ids",
        "intervention",
    }
    _required(pair, allowed - {"intervention"}, "pair")
    _reject_unknown(pair, allowed, "pair")
    if pair["schema_version"] != PAIR_SCHEMA_VERSION:
        raise ValueError("Unsupported pair schema_version")
    for field in ("pair_id", "flowpro_session_id", "observation_id"):
        _uuid(pair[field], f"pair.{field}")
    native_dim = pair["native_action_dim"]
    if native_dim not in ACTION_DIMS:
        raise ValueError(f"pair.native_action_dim must be one of {ACTION_DIMS}")
    if pair["action_space"] != ACTION_SPACE:
        raise ValueError("pair must use absolute unnormalized actions")
    if (isinstance(pair["fps"], bool) or not isinstance(pair["fps"], (int, float))
            or not math.isfinite(pair["fps"]) or pair["fps"] <= 0):
        raise ValueError("pair.fps must be positive and finite")
    if pair["effector_profile"] not in EFFECTOR_PROFILES:
        raise ValueError("pair has unknown effector_profile")
    _sha256(pair["checkpoint_stats_sha256"], "pair.checkpoint_stats_sha256")

    rejected_shape = _validate_tensor_artifact(pair["rejected"], "pair.rejected", action=True)
    preferred_shape = _validate_tensor_artifact(pair["preferred"], "pair.preferred", action=True)
    mask_shape = _validate_tensor_artifact(pair["valid_mask"], "pair.valid_mask", action=False)
    if rejected_shape != preferred_shape or rejected_shape != mask_shape:
        raise ValueError("rejected, preferred, and valid_mask shapes must match exactly in v1")
    if rejected_shape[-1] != native_dim:
        raise ValueError("pair tensor shape does not match native_action_dim")
    rejected_source = pair["rejected"].get("source")
    preferred_source = pair["preferred"].get("source")
    if (
        rejected_source not in {"executed_policy_trajectory", "positive_self_pair"}
        and "candidate_id" not in pair["rejected"]
    ):
        raise ValueError("candidate-pair rejected action requires candidate_id")
    if preferred_source not in {
        "measured_human_trajectory",
        "smooth_interpolation",
        "positive_self_pair",
    }:
        raise ValueError("pair.preferred has an unsupported source")
    if rejected_source == "positive_self_pair" and preferred_source != "positive_self_pair":
        raise ValueError("positive self-pair sources must match")

    if "intervention" in pair:
        intervention = _mapping(pair["intervention"], "pair.intervention")
        intervention_required = {
            "intervention_id",
            "active_arms",
            "matching_policy",
            "matched_positive_index",
            "match_distance",
        }
        intervention_fields = intervention_required | {"contact_alignment"}
        _required(intervention, intervention_required, "pair.intervention")
        _reject_unknown(intervention, intervention_fields, "pair.intervention")
        _uuid(intervention["intervention_id"], "pair.intervention.intervention_id")
        active_arms = _sequence(intervention["active_arms"], "pair.intervention.active_arms")
        if (
            not active_arms
            or len(set(active_arms)) != len(active_arms)
            or any(arm not in {"left", "right"} for arm in active_arms)
        ):
            raise ValueError("pair.intervention.active_arms must be a non-empty left/right subset")
        allowed_matching_policies = (
            {"positive_self_pair"}
            if rejected_source == "positive_self_pair"
            else {
                "paper_global_weighted_nearest/v1",
                "fk_contact_phase_nearest/v1",
                "fk_weighted_nearest/v2",
            }
        )
        if intervention["matching_policy"] not in allowed_matching_policies:
            raise ValueError("pair.intervention.matching_policy disagrees with pair source")
        _positive_int(
            intervention["matched_positive_index"],
            "pair.intervention.matched_positive_index",
            allow_zero=True,
        )
        match_distance = intervention["match_distance"]
        if (
            isinstance(match_distance, bool)
            or not isinstance(match_distance, (int, float))
            or not math.isfinite(match_distance)
            or match_distance < 0
        ):
            raise ValueError("pair.intervention.match_distance must be finite and non-negative")
        contact_alignment = intervention.get("contact_alignment")
        if intervention["matching_policy"] in {
            "fk_contact_phase_nearest/v1",
            "fk_weighted_nearest/v2",
        }:
            if contact_alignment is None:
                raise ValueError("event-aware matching requires pair.intervention.contact_alignment")
            contact = _mapping(contact_alignment, "pair.intervention.contact_alignment")
            contact_fields = {
                "schema_version",
                "source_phases",
                "matched_positive_phases",
                "nominal_transition_steps",
                "transition_steps",
                "junction_positive_index",
                "event_limited",
                "next_events",
                "event_guard_frames",
                "event_crossed",
                "gripper_bridge_policy",
                "join_gripper_delta",
            }
            joint_residual = intervention["matching_policy"] == "fk_weighted_nearest/v2"
            if joint_residual:
                contact_fields |= {
                    "matching_phase_gate",
                    "smoothing_backend",
                    "bridge_metrics",
                }
            _required(contact, contact_fields, "pair.intervention.contact_alignment")
            _reject_unknown(contact, contact_fields, "pair.intervention.contact_alignment")
            expected_schema = (
                "flowpro.gripper_event_alignment/v2"
                if joint_residual
                else "flowpro.contact_alignment/v1"
            )
            if contact["schema_version"] != expected_schema:
                raise ValueError("Unsupported contact-alignment schema_version")
            active_set = set(active_arms)
            allowed_phases = {
                "open_free",
                "closing",
                "holding",
                "opening",
                "post_release",
                "stable_intermediate",
            }
            if not joint_residual:
                allowed_phases -= {"closing", "opening"}
            for field in ("source_phases", "matched_positive_phases"):
                phases = _mapping(contact[field], f"pair.intervention.contact_alignment.{field}")
                if set(phases) != active_set or any(value not in allowed_phases for value in phases.values()):
                    raise ValueError(f"contact alignment {field} has invalid active-arm phases")
            if not joint_residual and dict(contact["source_phases"]) != dict(
                contact["matched_positive_phases"]
            ):
                raise ValueError("contact alignment source/matched phases must agree")
            if joint_residual:
                if contact["matching_phase_gate"] != "none":
                    raise ValueError("joint-residual matching may not claim a gripper phase gate")
                if contact["smoothing_backend"] != "joint_minimum_jerk_residual/v1":
                    raise ValueError("joint-residual smoothing backend is invalid")
                metrics = _mapping(
                    contact["bridge_metrics"],
                    "pair.intervention.contact_alignment.bridge_metrics",
                )
                metric_fields = {
                    "max_arm_velocity_rad_s",
                    "max_arm_acceleration_rad_s2",
                    "max_fk_position_step_m",
                    "max_fk_orientation_step_rad",
                    "joint_limits_checked",
                }
                _required(metrics, metric_fields, "pair.intervention.contact_alignment.bridge_metrics")
                _reject_unknown(
                    metrics,
                    metric_fields,
                    "pair.intervention.contact_alignment.bridge_metrics",
                )
                if not isinstance(metrics["joint_limits_checked"], bool):
                    raise ValueError("joint bridge joint_limits_checked must be boolean")
                for field in metric_fields - {"joint_limits_checked"}:
                    value = metrics[field]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError(f"joint bridge metric {field} is invalid")
            nominal = _positive_int(
                contact["nominal_transition_steps"],
                "pair.intervention.contact_alignment.nominal_transition_steps",
            )
            transition = _positive_int(
                contact["transition_steps"],
                "pair.intervention.contact_alignment.transition_steps",
            )
            if transition < 2 or transition > nominal or nominal > preferred_shape[0]:
                raise ValueError("contact alignment transition length is invalid")
            junction = _positive_int(
                contact["junction_positive_index"],
                "pair.intervention.contact_alignment.junction_positive_index",
                allow_zero=True,
            )
            if junction != intervention["matched_positive_index"] + transition - 1:
                raise ValueError("contact alignment junction does not follow M-prime")
            if not isinstance(contact["event_limited"], bool) or contact["event_limited"] != (
                transition < nominal
            ):
                raise ValueError("contact alignment event_limited is inconsistent")
            if contact["event_crossed"] is not False:
                raise ValueError("valid preference pairs may not cross a contact event")
            _positive_int(
                contact["event_guard_frames"],
                "pair.intervention.contact_alignment.event_guard_frames",
                allow_zero=True,
            )
            if contact["gripper_bridge_policy"] != "hold_until_raw_human":
                raise ValueError("contact alignment must hold grippers until the raw Human suffix")
            join_delta = _mapping(
                contact["join_gripper_delta"],
                "pair.intervention.contact_alignment.join_gripper_delta",
            )
            if set(join_delta) != active_set or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in join_delta.values()
            ):
                raise ValueError("contact alignment join_gripper_delta is invalid")
            events = _sequence(contact["next_events"], "pair.intervention.contact_alignment.next_events")
            event_fields = {
                "arm",
                "event_type",
                "onset_index",
                "completion_index",
                "normalized_amplitude",
                "confidence",
                "evidence",
            }
            for event_index, raw_event in enumerate(events):
                label = f"pair.intervention.contact_alignment.next_events[{event_index}]"
                event = _mapping(raw_event, label)
                _required(event, event_fields, label)
                _reject_unknown(event, event_fields, label)
                if event["arm"] not in active_set or event["event_type"] not in {"grasp", "release"}:
                    raise ValueError("contact alignment event has invalid arm or type")
                onset = _positive_int(event["onset_index"], f"{label}.onset_index", allow_zero=True)
                completion = _positive_int(
                    event["completion_index"], f"{label}.completion_index", allow_zero=True
                )
                if onset <= junction or completion < onset:
                    raise ValueError("contact alignment event overlaps the synthetic bridge")
                for field in ("normalized_amplitude", "confidence"):
                    value = event[field]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        or (field == "confidence" and value > 1)
                    ):
                        raise ValueError(f"{label}.{field} is invalid")
                if event["evidence"] != "measured_gripper_position":
                    raise ValueError("contact alignment event evidence is unsupported")
        elif contact_alignment is not None:
            raise ValueError("legacy/self pairs may not claim contact-aware alignment")

    context = _mapping(pair["same_context"], "pair.same_context")
    context_required = {"scene_verified", "sensor_skew_ok", "same_anchor"}
    context_allowed = context_required | {"robot_state_delta_max", "image_similarity"}
    _required(context, context_required, "pair.same_context")
    _reject_unknown(context, context_allowed, "pair.same_context")
    if not all(isinstance(context[field], bool) for field in context_required):
        raise ValueError("same_context gates must be booleans")

    if not isinstance(pair["pair_valid"], bool):
        raise ValueError("pair.pair_valid must be boolean")
    if pair["pair_valid"]:
        if pair["invalid_reason"] is not None:
            raise ValueError("valid pair must have invalid_reason=null")
        if not all(context[field] for field in context_required):
            raise ValueError("valid pair requires every same_context gate")
    elif not isinstance(pair["invalid_reason"], str) or not pair["invalid_reason"]:
        raise ValueError("invalid pair requires a non-empty invalid_reason")

    provenance = _sequence(pair["provenance_event_ids"], "pair.provenance_event_ids")
    if not provenance:
        raise ValueError("pair requires provenance_event_ids")
    for index, event_id in enumerate(provenance):
        _uuid(event_id, f"pair.provenance_event_ids[{index}]")
    if len(set(provenance)) != len(provenance):
        raise ValueError("pair.provenance_event_ids must be unique")
    required_provenance = {
        pair["rejected"]["source_event_id"],
        pair["preferred"]["source_event_id"],
    }
    if not required_provenance.issubset(set(provenance)):
        raise ValueError("pair provenance must include both source event IDs")
