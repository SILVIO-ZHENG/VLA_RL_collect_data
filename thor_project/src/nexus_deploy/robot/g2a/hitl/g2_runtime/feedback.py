"""Filesystem handoff for the G2 operator/VR scene-restoration display.

The runtime cannot modify the vendor teleoperation application.  This module
therefore publishes a small, atomic contract that a tablet/VR bridge can read:
three checksum-preserving copies of the exact inference images plus the current
intervention phase.  It never opens DDS/GDK or controls the recorder.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ..common.sidecar import CAMERA_SOURCES

if TYPE_CHECKING:
    from .intervention import SceneRestorationSnapshot

FEEDBACK_SCHEMA_VERSION = "flowpro.operator_feedback/v1"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
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


class SceneRestorationFeedbackWriter:
    """Publish exact anchor images without mutating the sealed-data source."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    @property
    def current_path(self) -> Path:
        return self.root / "current.json"

    def publish_snapshot(
        self,
        snapshot: SceneRestorationSnapshot,
        *,
        intervention_id: str,
        phase: str,
    ) -> Path:
        UUID(intervention_id)
        if phase not in {"rollback_ready", "anchor_crossed", "scene_restore"}:
            raise ValueError(f"Unsupported operator-feedback phase: {phase!r}")
        destination = self.root / "snapshots" / intervention_id / snapshot.observation_id
        destination.mkdir(parents=True, exist_ok=True)
        image_records: dict[str, dict[str, Any]] = {}
        for camera in CAMERA_SOURCES:
            source = snapshot.image_paths[camera]
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            target = destination / f"{camera}.jpg"
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Operator-feedback image collision: {target}")
            else:
                temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
                try:
                    with temporary.open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                    _fsync_directory(target.parent)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            image_records[camera] = {
                "relative_path": target.relative_to(self.root).as_posix(),
                "sha256": digest,
                "size_bytes": len(data),
                "vendor_timestamp_ns": snapshot.image_timestamps_ns[camera],
            }
        state_data = snapshot.state_path.read_bytes()
        state_target = destination / "state.npy"
        if state_target.exists():
            if state_target.read_bytes() != state_data:
                raise ValueError(f"Operator-feedback state collision: {state_target}")
        else:
            temporary = state_target.with_name(f".{state_target.name}.{uuid4().hex}.partial")
            try:
                with temporary.open("xb") as stream:
                    stream.write(state_data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, state_target)
                _fsync_directory(state_target.parent)
            finally:
                if temporary.exists():
                    temporary.unlink()
        payload = {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "phase": phase,
            "intervention_id": intervention_id,
            "observation_id": snapshot.observation_id,
            "anchor_timestamp_ns": snapshot.anchor_timestamp_ns,
            "task": snapshot.task,
            "state": {
                "relative_path": state_target.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(state_data).hexdigest(),
                "size_bytes": len(state_data),
            },
            "images": image_records,
            "requires_operator_confirmation": phase == "scene_restore",
            "instruction": {
                "rollback_ready": "Rollback plan ready; keep holding X to move backward.",
                "anchor_crossed": "Inference anchor crossed; release X here to select it.",
                "scene_restore": (
                    "Restore the physical scene to these three historical views, then double-click Y."
                ),
            }[phase],
        }
        _write_atomic(self.current_path, payload)
        return self.current_path

    def publish_verification(
        self,
        *,
        intervention_id: str,
        observation_id: str,
        valid: bool,
        verified_timestamp_ns: int,
    ) -> Path:
        UUID(intervention_id)
        UUID(observation_id)
        if verified_timestamp_ns <= 0:
            raise ValueError("verified_timestamp_ns must be positive")
        payload = {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "phase": "correction_ready" if valid else "pair_rejected",
            "intervention_id": intervention_id,
            "observation_id": observation_id,
            "scene_valid": bool(valid),
            "verified_timestamp_ns": verified_timestamp_ns,
            "requires_operator_confirmation": False,
            "instruction": (
                "Scene accepted; begin the Human correction from this anchor."
                if valid
                else "Scene rejected; this intervention will not become a strict preference pair."
            ),
        }
        _write_atomic(self.current_path, payload)
        return self.current_path
