"""Durable FlowPRO sidecar writer, vendor joiner, and atomic sealer.

The module is intentionally offline-first: tests construct a vendor-shaped
episode in a temporary directory and never connect to GDK or DDS.  Production
code can use the same API with ``vendor_record_root=/data/record``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from .contract import (
    ACTION_SPACE,
    EVENT_SCHEMA_VERSION,
    JOIN_CLOCK,
    MANIFEST_SCHEMA_VERSION,
    MAX_CAMERA_SKEW_MS,
    MODEL_FPS,
    validate_event,
    validate_manifest,
)

MANIFEST_PARTIAL = "manifest.partial.json"
MANIFEST_FINAL = "manifest.json"
EVENTS_WAL = "events.wal"
RECOVERY_LOG = "recovery.jsonl"
SIDECAR_OPEN_SUFFIX = ".open"
SIDECAR_SEALED_SUFFIX = ".sealed"

VENDOR_H5_PATH = "record/aligned_joints.h5"
VENDOR_META_PATH = "meta_info.json"
CAMERA_SOURCES = {
    "cam_head": "camera/head_color/head_color",
    "cam_left_wrist": "camera/hand_left_color/hand_left_color",
    "cam_right_wrist": "camera/hand_right_color/hand_right_color",
}
VENDOR_REQUIRED_PATHS = (
    VENDOR_META_PATH,
    VENDOR_H5_PATH,
    *(f"{prefix}.h265" for prefix in CAMERA_SOURCES.values()),
    *(f"{prefix}.txt" for prefix in CAMERA_SOURCES.values()),
)

VENDOR_META_ACTIVE = "active"
VENDOR_META_FINAL = "final"


@dataclass(frozen=True)
class VendorEpisodeInspection:
    record_name: str
    episode_path: Path
    source_timestamp_start_ns: int
    source_timestamp_end_ns: int
    max_camera_skew_ms: float
    camera_skew_ms: dict[str, float]
    files: list[dict[str, Any]]
    meta_clip_start_time: Any = None
    meta_clip_end_time: Any = None

    def manifest_record(self) -> dict[str, Any]:
        return {
            "record_name": self.record_name,
            "episode_path": str(self.episode_path),
            "source_timestamp_start_ns": self.source_timestamp_start_ns,
            "source_timestamp_end_ns": self.source_timestamp_end_ns,
            "max_camera_skew_ms": self.max_camera_skew_ms,
            "meta_clip_start_time": self.meta_clip_start_time,
            "meta_clip_end_time": self.meta_clip_end_time,
            "files": self.files,
        }


@dataclass(frozen=True)
class JoinValidationReport:
    flowpro_session_id: str
    record_name: str
    event_count: int
    observation_count: int
    candidate_count: int
    source_timestamp_start_ns: int
    source_timestamp_end_ns: int
    max_camera_skew_ms: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _parse_utc_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC wall timestamp must include a timezone")
    return int(parsed.timestamp() * 1_000_000_000)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_atomic(path, payload)


def _append_jsonl_fsync(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    )
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_stable(path: Path, *, block_size: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"size_bytes": after.st_size, "sha256": digest.hexdigest()}


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _safe_relative_file(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or relative_path in {"", ".", ".."}:
        raise ValueError(f"Unsafe relative path: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes sidecar root: {relative_path!r}")
    return resolved


def _read_camera_timestamps(path: Path) -> list[int]:
    timestamps: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                timestamp = int(parts[0])
            except ValueError as error:
                raise ValueError(f"Invalid camera timestamp at {path}:{line_number}") from error
            if timestamp <= 0 or (timestamps and timestamp <= timestamps[-1]):
                raise ValueError(f"Camera timestamps are not strictly increasing at {path}:{line_number}")
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError(f"Camera timestamp file is empty: {path}")
    return timestamps


def _read_h5_timestamps(path: Path) -> list[int]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to inspect aligned_joints.h5") from error

    timestamps: list[int] = []
    with h5py.File(path, "r") as h5:
        raw_keys = list(h5.keys())
        try:
            keys = sorted(raw_keys, key=int)
        except ValueError as error:
            raise ValueError(f"H5 frame keys must be decimal integers: {path}") from error
        if not keys or keys != [str(index) for index in range(len(keys))]:
            raise ValueError(f"H5 frame keys must be contiguous integers from zero: {path}")
        for key in keys:
            frame = h5[key]
            if "main_timestamp" not in frame:
                raise ValueError(f"H5 frame {key} is missing main_timestamp")
            timestamp = int(frame["main_timestamp"][()])
            if timestamp <= 0 or (timestamps and timestamp <= timestamps[-1]):
                raise ValueError(f"H5 main_timestamp is not strictly increasing at frame {key}")
            timestamps.append(timestamp)
    return timestamps


def read_vendor_meta(meta_path: Path) -> dict[str, Any]:
    """Read one vendor meta file and require a JSON object."""

    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if not isinstance(meta, dict):
        raise ValueError("meta_info.json must contain an object")
    return meta


def classify_vendor_meta(meta: Mapping[str, Any]) -> str:
    """Classify the audited tablet-recorder meta shape.

    Active recordings contain null validation fields.  A finalized recording
    must contain the exact nested booleans emitted by the G2 recorder.  Any
    partially-final or explicitly-invalid form fails closed.
    """

    data_validate = meta.get("data_validate")
    integrity = meta.get("integrity")
    if data_validate is None and integrity is None:
        return VENDOR_META_ACTIVE
    if not isinstance(data_validate, Mapping):
        raise ValueError("Vendor meta_info.json data_validate must be an object after tablet save")
    if data_validate.get("validate") is not True:
        raise ValueError("Vendor meta_info.json requires data_validate.validate=true")
    if not isinstance(integrity, Mapping):
        raise ValueError("Vendor meta_info.json integrity must be an object after tablet save")
    if integrity.get("integrity") is not True:
        raise ValueError("Vendor meta_info.json requires integrity.integrity=true")
    return VENDOR_META_FINAL


def validate_final_vendor_meta(meta: Mapping[str, Any]) -> None:
    if classify_vendor_meta(meta) != VENDOR_META_FINAL:
        raise ValueError("Vendor meta_info.json is still an active-recording placeholder")


def _camera_alignment_skew_ms(reference: list[int], candidates: list[int]) -> float:
    maximum_ns = 0
    previous_index = -1
    for timestamp in reference:
        right = min(bisect_left(candidates, timestamp), len(candidates) - 1)
        left = max(0, right - 1)
        index = left if abs(candidates[left] - timestamp) < abs(candidates[right] - timestamp) else right
        if index <= previous_index:
            raise ValueError("Camera alignment reuses or reverses a frame index")
        previous_index = index
        maximum_ns = max(maximum_ns, abs(candidates[index] - timestamp))
    return maximum_ns / 1_000_000.0


def inspect_vendor_episode(
    episode_path: Path,
    *,
    vendor_record_root: Path | None = None,
    max_camera_skew_ms: float = MAX_CAMERA_SKEW_MS,
) -> VendorEpisodeInspection:
    """Validate and hash one finalized vendor episode without modifying it."""

    episode = episode_path.expanduser().resolve()
    root = (vendor_record_root or episode.parent).expanduser().resolve()
    if episode.parent != root or not episode.name or episode.name in {".", ".."}:
        raise ValueError(f"Vendor episode must be one direct child of {root}: {episode}")
    if episode.is_symlink() or not episode.is_dir():
        raise FileNotFoundError(f"Vendor episode directory does not exist: {episode}")

    missing = [relative for relative in VENDOR_REQUIRED_PATHS if not (episode / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Vendor episode {episode.name} is missing required files: {missing}")
    initial_identities = {relative: _file_identity(episode / relative) for relative in VENDOR_REQUIRED_PATHS}

    meta_path = episode / VENDOR_META_PATH
    meta = read_vendor_meta(meta_path)
    validate_final_vendor_meta(meta)

    timestamps = _read_h5_timestamps(episode / VENDOR_H5_PATH)
    camera_skew: dict[str, float] = {}
    for camera_key, prefix in CAMERA_SOURCES.items():
        camera_timestamps = _read_camera_timestamps(episode / f"{prefix}.txt")
        skew = _camera_alignment_skew_ms(timestamps, camera_timestamps)
        if skew > max_camera_skew_ms:
            raise ValueError(f"{camera_key} alignment skew {skew:.3f} ms exceeds {max_camera_skew_ms:.3f} ms")
        camera_skew[camera_key] = skew

    file_records: list[dict[str, Any]] = []
    for relative in sorted(VENDOR_REQUIRED_PATHS):
        record = _sha256_stable(episode / relative)
        file_records.append({"relative_path": relative, **record})
    final_identities = {relative: _file_identity(episode / relative) for relative in VENDOR_REQUIRED_PATHS}
    changed = sorted(
        relative
        for relative in VENDOR_REQUIRED_PATHS
        if initial_identities[relative] != final_identities[relative]
    )
    if changed:
        raise RuntimeError(f"Vendor episode changed during inspection: {changed}")

    return VendorEpisodeInspection(
        record_name=episode.name,
        episode_path=episode,
        source_timestamp_start_ns=timestamps[0],
        source_timestamp_end_ns=timestamps[-1],
        max_camera_skew_ms=max(camera_skew.values()),
        camera_skew_ms=camera_skew,
        files=file_records,
        meta_clip_start_time=meta.get("clip_start_time"),
        meta_clip_end_time=meta.get("clip_end_time"),
    )


def recover_open_sidecar(open_directory: Path) -> dict[str, Any]:
    """Truncate only a crash-partial final WAL line and preserve an audit record."""

    directory = open_directory.expanduser().resolve()
    if not directory.name.endswith(SIDECAR_OPEN_SUFFIX):
        raise ValueError(f"Recovery is allowed only for {SIDECAR_OPEN_SUFFIX} directories")
    wal = directory / EVENTS_WAL
    if not wal.is_file():
        raise FileNotFoundError(wal)
    payload = wal.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return {"recovered": False, "discarded_bytes": 0}
    last_newline = payload.rfind(b"\n")
    valid_length = last_newline + 1
    discarded = payload[valid_length:]
    with wal.open("r+b") as stream:
        stream.truncate(valid_length)
        stream.flush()
        os.fsync(stream.fileno())
    report = {
        "recovered_at_utc": _utc_now(),
        "operation": "truncate_partial_wal_tail",
        "discarded_bytes": len(discarded),
        "discarded_sha256": hashlib.sha256(discarded).hexdigest(),
    }
    _append_jsonl_fsync(directory / RECOVERY_LOG, report)
    _fsync_directory(directory)
    return {"recovered": True, **report}


class FlowPROSidecarWriter:
    """Single-process append-only writer for one FlowPRO session."""

    def __init__(self, open_directory: Path, manifest: dict[str, Any]) -> None:
        self.open_directory = open_directory
        self.manifest = manifest
        self.flowpro_session_id = str(manifest["flowpro_session_id"])
        self.native_action_dim = int(manifest["policy"]["native_action_dim"])
        self.wall_to_vendor_offset_ns = int(
            manifest["clock_contract"].get("wall_to_vendor_offset_ns", 0)
        )
        self._lock = threading.Lock()
        self._closed = False
        existing_events = _read_events(self.open_directory / EVENTS_WAL, require_complete_tail=True)
        self._event_ids = {str(event["event_id"]) for event in existing_events}
        self._intervention_ids = {
            str(event["payload"]["intervention_id"])
            for event in existing_events
            if event["event_type"] == "HumanIntervention" and event["payload"].get("phase") == "finalized"
        }

    @classmethod
    def create(
        cls,
        output_root: Path,
        *,
        flowpro_session_id: str,
        robot_id: str,
        task_id: str,
        checkpoint_id: str,
        checkpoint_stats_sha256: str,
        native_action_dim: int,
        chunk_size: int,
        effector_profile: str,
        processor_version: str,
        model_fps: float = MODEL_FPS,
        collection_manifest: dict[str, Any] | None = None,
        wall_to_vendor_offset_ns: int = 0,
    ) -> FlowPROSidecarWriter:
        UUID(flowpro_session_id)
        root = output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / f"{flowpro_session_id}{SIDECAR_OPEN_SUFFIX}"
        directory.mkdir(exist_ok=False)
        (directory / "policy_candidates").mkdir()
        (directory / "inference_inputs").mkdir()
        (directory / "executed_segments").mkdir()
        wal = directory / EVENTS_WAL
        with wal.open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "open",
            "flowpro_session_id": flowpro_session_id,
            "robot_id": robot_id,
            "task_id": task_id,
            "created_at_utc": _utc_now(),
            "sealed_at_utc": None,
            "clock_contract": {
                "join_clock": JOIN_CLOCK,
                "timestamp_unit": "ns",
                "monotonic_scope": "single_process_only",
                **(
                    {"wall_to_vendor_offset_ns": wall_to_vendor_offset_ns}
                    if wall_to_vendor_offset_ns else {}
                ),
            },
            "policy": {
                "checkpoint_id": checkpoint_id,
                "checkpoint_stats_sha256": checkpoint_stats_sha256,
                "native_action_dim": native_action_dim,
                "action_space": ACTION_SPACE,
                "model_fps": model_fps,
                "chunk_size": chunk_size,
                "effector_profile": effector_profile,
                "processor_version": processor_version,
            },
            "collection_manifest": collection_manifest,
            "vendor_episode": None,
            "sidecar_files": [],
        }
        validate_manifest(manifest)
        _write_json_atomic(directory / MANIFEST_PARTIAL, manifest)
        _fsync_directory(directory)
        return cls(directory, manifest)

    @classmethod
    def open_existing(cls, open_directory: Path) -> FlowPROSidecarWriter:
        directory = open_directory.expanduser().resolve()
        if not directory.name.endswith(SIDECAR_OPEN_SUFFIX):
            raise ValueError(f"Expected a {SIDECAR_OPEN_SUFFIX} directory")
        manifest_path = directory / MANIFEST_PARTIAL
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        validate_manifest(manifest)
        if manifest["status"] != "open":
            raise ValueError("Partial manifest must have status=open")
        return cls(directory, manifest)

    def append_event(self, event: Mapping[str, Any]) -> None:
        record = self._vendor_clock_record(dict(event))
        validate_event(record)
        if record["flowpro_session_id"] != self.flowpro_session_id:
            raise ValueError("Event flowpro_session_id does not match writer session")
        event_id = str(record["event_id"])
        intervention_id = None
        if record["event_type"] == "HumanIntervention" and record["payload"].get("phase") == "finalized":
            intervention_id = str(record["payload"]["intervention_id"])
        with self._lock:
            self._require_open()
            if event_id in self._event_ids:
                raise ValueError(f"Duplicate event_id: {event_id}")
            if intervention_id is not None and intervention_id in self._intervention_ids:
                raise ValueError(f"Duplicate intervention_id: {intervention_id}")
            _append_jsonl_fsync(self.open_directory / EVENTS_WAL, record)
            self._event_ids.add(event_id)
            if intervention_id is not None:
                self._intervention_ids.add(intervention_id)

    def _vendor_clock_record(self, value: Any, field: str = "") -> Any:
        """Convert internal wall-clock times only at the durable vendor boundary."""

        offset = self.wall_to_vendor_offset_ns
        if not offset:
            return value
        if isinstance(value, Mapping):
            return {key: self._vendor_clock_record(item, str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [self._vendor_clock_record(item, field) for item in value]
        absolute_time = (
            field.endswith("_timestamp_ns")
            or field in {"timestamp_ns", "correction_start_ns", "correction_end_ns"}
        ) and field != "vr_source_timestamp_ns"
        if absolute_time and isinstance(value, int) and not isinstance(value, bool):
            converted = value - offset
            if converted <= 0:
                raise ValueError(f"Converted vendor timestamp {field} is not positive")
            return converted
        return value

    def store_candidate(self, candidate_id: str, values: Any) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required to store policy candidates") from error

        UUID(candidate_id)
        array = np.asarray(values, dtype=np.float32)
        if (
            array.ndim != 2
            or array.shape[1] != self.native_action_dim
            or array.shape[0] < 1
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                f"Candidate must be finite [T,{self.native_action_dim}] float32, got {array.shape}"
            )
        little_endian = np.ascontiguousarray(array.astype("<f4", copy=False))
        payload = little_endian.tobytes(order="C")
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"policy_candidates/{candidate_id}.f32"
        path = self.open_directory / relative
        with self._lock:
            self._require_open()
            if path.exists():
                existing = _sha256_stable(path)
                if existing["sha256"] != digest or existing["size_bytes"] != len(payload):
                    raise ValueError(f"candidate_id collision with different content: {candidate_id}")
            else:
                _write_atomic(path, payload)
        return {
            "relative_path": relative,
            "size_bytes": len(payload),
            "sha256": digest,
            "dtype": "float32",
            "shape": [int(array.shape[0]), int(array.shape[1])],
        }

    def store_observation_state(self, observation_id: str, values: Any) -> dict[str, Any]:
        """Persist the exact canonical 24D state sent with one inference request."""

        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required to store inference states") from error

        UUID(observation_id)
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (24,) or not np.isfinite(array).all():
            raise ValueError(f"Inference state must be finite canonical [24], got {array.shape}")
        little_endian = np.ascontiguousarray(array.astype("<f4", copy=False))
        payload = little_endian.tobytes(order="C")
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"inference_inputs/{observation_id}.state.f32"
        path = self.open_directory / relative
        with self._lock:
            self._require_open()
            if path.exists():
                existing = _sha256_stable(path)
                if existing["sha256"] != digest or existing["size_bytes"] != len(payload):
                    raise ValueError(f"observation_id collision with a different state: {observation_id}")
            else:
                _write_atomic(path, payload)
        return {
            "relative_path": relative,
            "size_bytes": len(payload),
            "sha256": digest,
            "dtype": "float32",
            "shape": [24],
            "space": "canonical_hardware_24d",
        }

    def store_executed_segment(
        self,
        intervention_id: str,
        segment: Any,
    ) -> dict[str, dict[str, Any]]:
        """Atomically persist a frozen, actually executed Policy segment.

        The 100 Hz path remains memory-only.  These immutable artifacts are
        written once, after an intervention freezes the history window.
        """

        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required to store execution segments") from error

        UUID(intervention_id)
        samples = tuple(segment.samples)
        if not samples:
            raise ValueError("Executed segment must contain at least one sample")
        from ..g2_runtime.executed import EXECUTION_KIND_CODES

        arrays = {
            "timestamps_ns": np.asarray(segment.timestamps_ns, dtype="<i8") - self.wall_to_vendor_offset_ns,
            "measured_model_actions": np.asarray(segment.measured_model_actions, dtype="<f4"),
            "measured_state_24d": np.stack([sample.measured_state_24d for sample in samples]).astype("<f4"),
            "policy_native_actions": np.stack([sample.policy_native_action for sample in samples]).astype(
                "<f4"
            ),
            "safety_targets_24d": np.stack([sample.safety_target_24d for sample in samples]).astype("<f4"),
            "gdk_commands_24d": np.stack([sample.gdk_command_24d for sample in samples]).astype("<f4"),
            "command_accepted": np.asarray([sample.command_accepted for sample in samples], dtype=np.uint8),
            "safety_modified": np.asarray([sample.safety_modified for sample in samples], dtype=np.uint8),
            "execution_kind_code": np.asarray(
                [EXECUTION_KIND_CODES[sample.execution_kind] for sample in samples], dtype=np.uint8
            ),
        }
        lineage = [
            {
                "observation_id": sample.observation_id,
                "candidate_id": sample.candidate_id,
                "generation_id": sample.generation_id,
                "action_id": sample.action_id,
                "chunk_id": sample.chunk_id,
                "action_index": sample.action_index,
                "timestamp_ns": sample.timestamp_ns - self.wall_to_vendor_offset_ns,
                "measured_timestamp_ns": sample.measured_timestamp_ns - self.wall_to_vendor_offset_ns,
                "control_owner": sample.control_owner,
                "execution_kind": sample.execution_kind,
                "request_started_monotonic_ns": sample.request_started_monotonic_ns,
                "response_received_monotonic_ns": sample.response_received_monotonic_ns,
                "chunk_installed_monotonic_ns": sample.chunk_installed_monotonic_ns,
                "latency_stale_steps": sample.latency_stale_steps,
                "installed_first_action_index": sample.installed_first_action_index,
                "first_executed_timestamp_ns": (
                    None if sample.first_executed_timestamp_ns is None
                    else sample.first_executed_timestamp_ns - self.wall_to_vendor_offset_ns
                ),
                "first_executed_action_index": sample.first_executed_action_index,
            }
            for sample in samples
        ]
        directory = self.open_directory / "executed_segments" / intervention_id
        records: dict[str, dict[str, Any]] = {}
        with self._lock:
            self._require_open()
            directory.mkdir(parents=True, exist_ok=False)
            for name, array in arrays.items():
                contiguous = np.ascontiguousarray(array)
                payload = contiguous.tobytes(order="C")
                path = directory / f"{name}.bin"
                _write_atomic(path, payload)
                records[name] = {
                    "relative_path": path.relative_to(self.open_directory).as_posix(),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "dtype": str(contiguous.dtype),
                    "shape": list(contiguous.shape),
                }
            lineage_payload = (
                json.dumps(lineage, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            lineage_path = directory / "lineage.json"
            _write_atomic(lineage_path, lineage_payload)
            records["lineage"] = {
                "relative_path": lineage_path.relative_to(self.open_directory).as_posix(),
                "size_bytes": len(lineage_payload),
                "sha256": hashlib.sha256(lineage_payload).hexdigest(),
                "dtype": "json",
                "shape": [len(lineage)],
            }
        return records

    def store_rollback_plan(self, intervention_id: str, waypoints: Any) -> dict[str, Any]:
        """Persist the exact safety-checked rollback waypoints for audit/replay."""

        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required to store rollback plans") from error

        UUID(intervention_id)
        array = np.asarray(waypoints, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 24 or len(array) < 2 or not np.isfinite(array).all():
            raise ValueError("Rollback plan must be a finite [T,24] array with T>=2")
        contiguous = np.ascontiguousarray(array.astype("<f4", copy=False))
        payload = contiguous.tobytes(order="C")
        relative = f"executed_segments/{intervention_id}/rollback_plan.f32"
        path = self.open_directory / relative
        with self._lock:
            self._require_open()
            if not path.parent.is_dir():
                raise FileNotFoundError("Executed segment must be stored before its rollback plan")
            _write_atomic(path, payload)
        return {
            "relative_path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dtype": "float32",
            "shape": list(contiguous.shape),
        }

    def store_rollback_plan_source_timestamps(
        self, intervention_id: str, source_timestamps_ns: Any
    ) -> dict[str, Any]:
        """Persist the historical time represented by every rollback waypoint."""

        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required to store rollback plans") from error

        UUID(intervention_id)
        array = np.asarray(source_timestamps_ns, dtype=np.int64) - self.wall_to_vendor_offset_ns
        if array.ndim != 1 or len(array) < 2 or np.any(array <= 0) or np.any(np.diff(array) > 0):
            raise ValueError("Rollback source timestamps must be a positive non-increasing [T] array")
        contiguous = np.ascontiguousarray(array.astype("<i8", copy=False))
        payload = contiguous.tobytes(order="C")
        relative = f"executed_segments/{intervention_id}/rollback_plan_source_timestamps.i64"
        path = self.open_directory / relative
        with self._lock:
            self._require_open()
            if not path.parent.is_dir():
                raise FileNotFoundError("Rollback plan must be stored before its source timestamps")
            _write_atomic(path, payload)
        return {
            "relative_path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dtype": "int64",
            "shape": list(contiguous.shape),
        }

    def store_inference_jpeg(
        self,
        data: bytes,
        *,
        camera_key: str,
        vendor_timestamp_ns: int,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        if camera_key not in CAMERA_SOURCES:
            raise ValueError(f"Unknown camera_key: {camera_key!r}")
        if not isinstance(data, bytes) or not data:
            raise ValueError("Inference JPEG must contain bytes")
        if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
            raise ValueError("Inference image is not a complete JPEG byte stream")
        if vendor_timestamp_ns <= 0 or width <= 0 or height <= 0:
            raise ValueError("Inference image timestamp and dimensions must be positive")
        digest = hashlib.sha256(data).hexdigest()
        relative = f"inference_inputs/{digest}.jpg"
        path = self.open_directory / relative
        with self._lock:
            self._require_open()
            if path.exists():
                existing = _sha256_stable(path)
                if existing["sha256"] != digest or existing["size_bytes"] != len(data):
                    raise ValueError(f"Content-addressed image collision: {digest}")
            else:
                _write_atomic(path, data)
        return {
            "relative_path": relative,
            "size_bytes": len(data),
            "sha256": digest,
            "encoding": "jpeg",
            "camera_key": camera_key,
            "vendor_timestamp_ns": vendor_timestamp_ns,
            "width": width,
            "height": height,
        }

    def record_policy_candidate(
        self,
        values: Any,
        *,
        event_id: str,
        observation_id: str,
        request_id: str,
        candidate_id: str,
        generation_id: str,
        monotonic_ns: int,
        wall_time_utc: str,
        control_owner: str,
        sampling: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact = self.store_candidate(candidate_id, values)
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": event_id,
            "flowpro_session_id": self.flowpro_session_id,
            "event_type": "PolicyCandidate",
            "producer": "policy_server",
            "monotonic_ns": monotonic_ns,
            "wall_time_utc": wall_time_utc,
            "vendor_timestamp_ns": None,
            "observation_id": observation_id,
            "request_id": request_id,
            "candidate_id": candidate_id,
            "generation_id": generation_id,
            "action_id": None,
            "control_owner": control_owner,
            "payload": {
                **artifact,
                "native_action_dim": self.native_action_dim,
                "action_space": ACTION_SPACE,
                "sampling": dict(sampling) if sampling is not None else None,
            },
        }
        self.append_event(event)
        return event

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FlowPRO sidecar writer is closed")


def _read_events(path: Path, *, require_complete_tail: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if payload and require_complete_tail and not payload.endswith(b"\n"):
        raise ValueError(f"WAL has a crash-partial final line; run recover_open_sidecar: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Event at {path}:{line_number} is not an object")
        validate_event(record)
        records.append(record)
    return records


def _validate_event_join(
    sidecar_directory: Path,
    events: list[dict[str, Any]],
    inspection: VendorEpisodeInspection,
    flowpro_session_id: str,
) -> tuple[int, int]:
    event_ids: set[str] = set()
    observation_ids: set[str] = set()
    candidate_ids: set[str] = set()
    intervention_ids: set[str] = set()
    last_event_by_producer: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["flowpro_session_id"] != flowpro_session_id:
            raise ValueError("WAL contains an event from another FlowPRO session")
        event_id = str(event["event_id"])
        if event_id in event_ids:
            raise ValueError(f"WAL contains duplicate event_id: {event_id}")
        event_ids.add(event_id)
        producer = str(event["producer"])
        monotonic_ns = int(event["monotonic_ns"])
        previous_event = last_event_by_producer.get(producer)
        if previous_event is not None and monotonic_ns < int(previous_event["monotonic_ns"]):
            # ``monotonic_ns`` is intentionally process-local (see the
            # manifest clock contract).  EpisodeOutcome/DataSealed can be
            # appended by the recovery utility after a G2 process or machine
            # restart, so their monotonic epoch may legitimately reset.  Keep
            # this exception deliberately narrow: trajectory/control events
            # must still be monotonic within their producer process.
            lifecycle_recovery = (
                event["event_type"] in {"EpisodeOutcome", "DataSealed"}
                and event["vendor_timestamp_ns"] is None
                and _parse_utc_ns(str(event["wall_time_utc"]))
                > _parse_utc_ns(str(previous_event["wall_time_utc"]))
            )
            if not lifecycle_recovery:
                raise ValueError(f"monotonic_ns moved backwards for producer {producer}")
        last_event_by_producer[producer] = event
        vendor_timestamp = event["vendor_timestamp_ns"]
        # r11 wrote the time at which final tablet metadata was observed as a
        # vendor timestamp.  That lifecycle event necessarily occurs after
        # clip_end and is not robot trajectory data.  Accept this one legacy
        # representation narrowly; new runtimes write its vendor timestamp as
        # None.  All motion, observation, rollback and other logical events
        # remain strictly bounded by the recorded episode.
        legacy_post_record_tablet_event = (
            event["event_type"] == "LogicalInput"
            and event["control_owner"] == "hold"
            and event["payload"].get("name") == "tablet_recording_finished"
            and vendor_timestamp is not None
            and int(vendor_timestamp) >= inspection.source_timestamp_end_ns
        )
        if vendor_timestamp is not None and not legacy_post_record_tablet_event and not (
            inspection.source_timestamp_start_ns
            <= int(vendor_timestamp)
            <= inspection.source_timestamp_end_ns
        ):
            raise ValueError(f"Event {event_id} vendor timestamp is outside the bound episode")
        if event["event_type"] == "ObservationCaptured":
            observation_ids.add(str(event["observation_id"]))
        elif event["event_type"] == "PolicyCandidate":
            candidate_id = str(event["candidate_id"])
            if candidate_id in candidate_ids:
                raise ValueError(f"WAL contains duplicate candidate_id: {candidate_id}")
            candidate_ids.add(candidate_id)
            payload = event["payload"]
            artifact = _safe_relative_file(sidecar_directory, str(payload["relative_path"]))
            actual = _sha256_stable(artifact)
            if actual["size_bytes"] != payload["size_bytes"] or actual["sha256"] != payload["sha256"]:
                raise ValueError(f"Policy candidate artifact mismatch: {artifact}")
        elif event["event_type"] == "ExecutedSegmentCaptured":
            intervention_id = str(event["payload"]["intervention_id"])
            if intervention_id in intervention_ids:
                raise ValueError(f"WAL contains duplicate intervention_id: {intervention_id}")
            intervention_ids.add(intervention_id)
            for artifact_record in event["payload"]["artifacts"].values():
                artifact = _safe_relative_file(sidecar_directory, str(artifact_record["relative_path"]))
                actual = _sha256_stable(artifact)
                if (
                    actual["size_bytes"] != artifact_record["size_bytes"]
                    or actual["sha256"] != artifact_record["sha256"]
                ):
                    raise ValueError(f"Executed segment artifact mismatch: {artifact}")
        elif event["event_type"] == "RollbackPlanned":
            artifact_record = event["payload"].get("plan")
            if not isinstance(artifact_record, Mapping):
                raise ValueError("RollbackPlanned is missing its immutable plan artifact")
            artifact = _safe_relative_file(sidecar_directory, str(artifact_record["relative_path"]))
            actual = _sha256_stable(artifact)
            if (
                actual["size_bytes"] != artifact_record["size_bytes"]
                or actual["sha256"] != artifact_record["sha256"]
            ):
                raise ValueError(f"Rollback plan artifact mismatch: {artifact}")
            timestamps_record = event["payload"].get("plan_source_timestamps")
            if timestamps_record is not None:
                if not isinstance(timestamps_record, Mapping):
                    raise ValueError("Rollback source timestamps artifact is malformed")
                timestamps_artifact = _safe_relative_file(
                    sidecar_directory, str(timestamps_record["relative_path"])
                )
                timestamps_actual = _sha256_stable(timestamps_artifact)
                if (
                    timestamps_actual["size_bytes"] != timestamps_record["size_bytes"]
                    or timestamps_actual["sha256"] != timestamps_record["sha256"]
                ):
                    raise ValueError(f"Rollback source timestamps artifact mismatch: {timestamps_artifact}")
        elif event["event_type"] == "HumanIntervention" and event["payload"].get("phase") == "finalized":
            intervention_id = str(event["payload"]["intervention_id"])
            if intervention_id in intervention_ids:
                raise ValueError(f"WAL contains duplicate intervention_id: {intervention_id}")
            intervention_ids.add(intervention_id)
    for event in events:
        if event["event_type"] == "PolicyCandidate" and str(event["observation_id"]) not in observation_ids:
            raise ValueError(
                f"PolicyCandidate references missing ObservationCaptured: {event['observation_id']}"
            )
        if event["event_type"] == "HumanIntervention" and event["payload"].get("phase") == "finalized":
            if str(event["observation_id"]) not in observation_ids:
                raise ValueError(
                    f"HumanIntervention references missing ObservationCaptured: {event['observation_id']}"
                )
            if str(event["candidate_id"]) not in candidate_ids:
                raise ValueError(
                    f"HumanIntervention references missing PolicyCandidate: {event['candidate_id']}"
                )
        if event["event_type"] == "ExecutedSegmentCaptured":
            if str(event["observation_id"]) not in observation_ids:
                raise ValueError(
                    "ExecutedSegmentCaptured references missing ObservationCaptured: "
                    f"{event['observation_id']}"
                )
            if str(event["candidate_id"]) not in candidate_ids:
                raise ValueError(
                    f"ExecutedSegmentCaptured references missing PolicyCandidate: {event['candidate_id']}"
                )
    _validate_paper_event_order(events)
    return len(observation_ids), len(candidate_ids)


def _validate_paper_event_order(events: list[dict[str, Any]]) -> None:
    order = {
        "RollbackRequested": 0,
        "ExecutedSegmentCaptured": 1,
        "RollbackPlanned": 2,
        "RollbackCompleted": 3,
        "SceneRestorationStarted": 4,
        "SceneRestorationVerified": 5,
        "HumanCorrectionStarted": 6,
        "HumanCorrectionCompleted": 7,
    }
    seen: dict[str, int] = {}
    verified_valid: set[str] = set()
    for event in events:
        event_type = str(event["event_type"])
        if event_type not in order:
            continue
        intervention_id = str(event["payload"]["intervention_id"])
        stage = order[event_type]
        previous = seen.get(intervention_id, -1)
        if stage <= previous:
            raise ValueError(f"Paper intervention {intervention_id} has duplicate/out-of-order {event_type}")
        if stage > previous + 1:
            raise ValueError(
                f"Paper intervention {intervention_id} skipped a required phase before {event_type}"
            )
        if event_type == "SceneRestorationVerified" and event["payload"].get("scene_valid") is True:
            verified_valid.add(intervention_id)
        if event_type == "HumanCorrectionStarted" and intervention_id not in verified_valid:
            raise ValueError("Human correction started without a valid restored scene")
        seen[intervention_id] = stage


def _sidecar_file_records(directory: Path) -> list[dict[str, Any]]:
    excluded = {MANIFEST_PARTIAL, MANIFEST_FINAL}
    partials = sorted(path for path in directory.rglob("*.partial") if path.is_file())
    if partials:
        raise ValueError(f"Sidecar contains unfinished partial files: {partials}")
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        records.append({"relative_path": relative, **_sha256_stable(path)})
    return records


def _append_data_sealed_event(open_directory: Path, manifest: Mapping[str, Any], record_name: str) -> None:
    events = _read_events(open_directory / EVENTS_WAL, require_complete_tail=True)
    existing = [event for event in events if event["event_type"] == "DataSealed"]
    if existing:
        if len(existing) != 1 or existing[0]["payload"].get("record_name") != record_name:
            raise ValueError("Existing DataSealed event conflicts with requested vendor episode")
        return
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": str(uuid4()),
        "flowpro_session_id": manifest["flowpro_session_id"],
        "event_type": "DataSealed",
        "producer": "g2_supervisor",
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": _utc_now(),
        "vendor_timestamp_ns": None,
        "observation_id": None,
        "request_id": None,
        "candidate_id": None,
        "generation_id": None,
        "action_id": None,
        "control_owner": "hold",
        "payload": {"record_name": record_name},
    }
    validate_event(event)
    _append_jsonl_fsync(open_directory / EVENTS_WAL, event)


def seal_sidecar(
    open_directory: Path,
    vendor_episode_path: Path,
    *,
    vendor_record_root: Path | None = None,
) -> Path:
    """Bind an open sidecar to a finalized vendor episode and atomically seal it."""

    directory = open_directory.expanduser().resolve()
    if not directory.name.endswith(SIDECAR_OPEN_SUFFIX):
        raise ValueError(f"Expected a {SIDECAR_OPEN_SUFFIX} directory: {directory}")
    sealed = directory.with_name(directory.name[: -len(SIDECAR_OPEN_SUFFIX)] + SIDECAR_SEALED_SUFFIX)
    if sealed.exists():
        if directory.exists():
            raise FileExistsError(
                f"Both open and sealed sidecars exist for one session: {directory}, {sealed}"
            )
        validate_sealed_sidecar(
            sealed,
            vendor_episode_path=vendor_episode_path,
            vendor_record_root=vendor_record_root,
        )
        return sealed

    inspection = inspect_vendor_episode(
        vendor_episode_path,
        vendor_record_root=vendor_record_root,
    )
    root = (vendor_record_root or inspection.episode_path.parent).expanduser().resolve()
    partial_path = directory / MANIFEST_PARTIAL
    final_path = directory / MANIFEST_FINAL

    if final_path.is_file() and not partial_path.exists():
        with final_path.open(encoding="utf-8") as stream:
            completed_manifest = json.load(stream)
        validate_manifest(completed_manifest, vendor_record_root=root)
        if completed_manifest["vendor_episode"] != inspection.manifest_record():
            raise ValueError("Recovered final manifest does not match requested vendor episode")
        os.replace(directory, sealed)
        _fsync_directory(sealed.parent)
        validate_sealed_sidecar(
            sealed,
            vendor_episode_path=inspection.episode_path,
            vendor_record_root=root,
        )
        return sealed

    if not partial_path.is_file():
        raise FileNotFoundError(partial_path)
    with partial_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_manifest(manifest, vendor_record_root=root)
    if manifest["status"] != "open":
        raise ValueError("Only status=open sidecars can be sealed")

    events = _read_events(directory / EVENTS_WAL, require_complete_tail=True)
    _validate_event_join(directory, events, inspection, str(manifest["flowpro_session_id"]))
    _append_data_sealed_event(directory, manifest, inspection.record_name)
    events = _read_events(directory / EVENTS_WAL, require_complete_tail=True)
    _validate_event_join(directory, events, inspection, str(manifest["flowpro_session_id"]))

    completed = dict(manifest)
    completed.update(
        {
            "status": "sealed",
            "sealed_at_utc": _utc_now(),
            "vendor_episode": inspection.manifest_record(),
            "sidecar_files": _sidecar_file_records(directory),
        }
    )
    validate_manifest(completed, vendor_record_root=root)
    _write_json_atomic(final_path, completed)
    partial_path.unlink()
    _fsync_directory(directory)
    os.replace(directory, sealed)
    _fsync_directory(sealed.parent)
    validate_sealed_sidecar(
        sealed,
        vendor_episode_path=inspection.episode_path,
        vendor_record_root=root,
    )
    return sealed


def validate_sealed_sidecar(
    sealed_directory: Path,
    *,
    vendor_episode_path: Path | None = None,
    vendor_record_root: Path | None = None,
) -> JoinValidationReport:
    """Re-hash both stores and validate event/candidate/vendor lineage."""

    directory = sealed_directory.expanduser().resolve()
    if not directory.name.endswith(SIDECAR_SEALED_SUFFIX) or not directory.is_dir():
        raise ValueError(f"Expected an existing {SIDECAR_SEALED_SUFFIX} directory")
    manifest_path = directory / MANIFEST_FINAL
    if not manifest_path.is_file() or (directory / MANIFEST_PARTIAL).exists():
        raise ValueError("Sealed sidecar must contain manifest.json and no partial manifest")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest_vendor_path = Path(manifest["vendor_episode"]["episode_path"])
    actual_vendor_path = (vendor_episode_path or manifest_vendor_path).expanduser().resolve()
    root = (vendor_record_root or actual_vendor_path.parent).expanduser().resolve()
    # The sealed manifest records the original G2 absolute path. Validate that
    # immutable provenance against its original root, while allowing an
    # explicitly supplied 5090 staging path for content verification.  The
    # remap is constrained to exactly <local-root>/<record_name>; paths and
    # filenames inside the episode are still taken from the sealed manifest.
    validate_manifest(manifest, vendor_record_root=manifest_vendor_path.parent)
    if manifest["status"] != "sealed":
        raise ValueError("Sealed directory contains a non-sealed manifest")
    record_name = str(manifest["vendor_episode"]["record_name"])
    if actual_vendor_path.parent != root or actual_vendor_path.name != record_name:
        raise ValueError("Remapped vendor episode must be <vendor_record_root>/<record_name>")

    inspection = inspect_vendor_episode(actual_vendor_path, vendor_record_root=root)
    inspection_record = inspection.manifest_record()
    inspection_record["episode_path"] = str(manifest_vendor_path)
    if manifest["vendor_episode"] != inspection_record:
        raise ValueError("Vendor episode no longer matches the sealed manifest")

    actual_sidecar_files = _sidecar_file_records(directory)
    if manifest["sidecar_files"] != actual_sidecar_files:
        raise ValueError("Sidecar files no longer match the sealed manifest")
    events = _read_events(directory / EVENTS_WAL, require_complete_tail=True)
    observation_count, candidate_count = _validate_event_join(
        directory,
        events,
        inspection,
        str(manifest["flowpro_session_id"]),
    )
    return JoinValidationReport(
        flowpro_session_id=str(manifest["flowpro_session_id"]),
        record_name=inspection.record_name,
        event_count=len(events),
        observation_count=observation_count,
        candidate_count=candidate_count,
        source_timestamp_start_ns=inspection.source_timestamp_start_ns,
        source_timestamp_end_ns=inspection.source_timestamp_end_ns,
        max_camera_skew_ms=inspection.max_camera_skew_ms,
    )
