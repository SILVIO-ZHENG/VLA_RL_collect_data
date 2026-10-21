"""Versioned, atomic storage for administrator-defined alignment targets."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .config import AlignmentProfile
from .geometry import matrix_pose, pose_matrix

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class TargetStoreError(ValueError):
    pass


class TargetRepository:
    """Keep immutable versions and one atomically replaceable active target."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, profile: AlignmentProfile) -> AlignmentProfile:
        document = self.active(profile.id)
        if document is None:
            return profile
        pose = _pose(document.get("expected_table_pose"), "expected_table_pose")
        return replace(
            profile,
            version=str(document["version"]),
            base_from_expected_table=pose_matrix(pose),
            target_configured=True,
        )

    def active(self, profile_id: str) -> dict[str, Any] | None:
        path = self._profile_directory(profile_id) / "active.json"
        if not path.exists():
            return None
        return self._read(path, profile_id)

    def publish(
        self,
        profile: AlignmentProfile,
        expected_table_pose: list[float] | tuple[float, ...],
        *,
        robot: Mapping[str, Any],
        robot_host: str,
        calibration_version: str,
    ) -> dict[str, Any]:
        pose = _pose(expected_table_pose, "expected_table_pose")
        now = time.time()
        base_version = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        version = base_version
        profile_dir = self._profile_directory(profile.id)
        versions_dir = profile_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        suffix = 1
        while (versions_dir / f"{version}.json").exists():
            suffix += 1
            version = f"{base_version}-{suffix}"
        tolerance = profile.tolerance
        document: dict[str, Any] = {
            "schema_version": 1,
            "profile_id": profile.id,
            "profile_name": profile.display_name,
            "version": version,
            "created_at": now,
            "expected_table_pose": pose,
            "tag": {
                "family": profile.tag_family,
                "ids": list(profile.tag_ids),
                "size_m": profile.tag_size_m,
            },
            "tag_from_table": list(matrix_pose(profile.tag_from_table)),
            "tolerance": {
                "forward_m": tolerance.forward_m,
                "lateral_m": tolerance.lateral_m,
                "yaw_deg": tolerance.yaw_deg,
                "height_m": tolerance.height_m,
                "tilt_deg": tolerance.tilt_deg,
                "stable_s": tolerance.stable_s,
                "window_s": tolerance.window_s,
                "minimum_samples": tolerance.minimum_samples,
                "minimum_detection_ratio": tolerance.minimum_detection_ratio,
            },
            "inspection_joints": dict(profile.inspection_joints),
            "inspection_joint_tolerance_rad": profile.inspection_joint_tolerance_rad,
            "robot": {
                "id": robot.get("id"),
                "model": robot.get("model"),
                "host": robot_host,
            },
            "calibration_version": calibration_version,
        }
        self._atomic_write(versions_dir / f"{version}.json", document)
        self._atomic_write(profile_dir / "active.json", document)
        return document

    def history(self, profile_id: str) -> list[dict[str, Any]]:
        active = self.active(profile_id)
        active_version = str(active["version"]) if active else None
        directory = self._profile_directory(profile_id) / "versions"
        if not directory.exists():
            return []
        values = []
        for path in directory.glob("*.json"):
            document = self._read(path, profile_id)
            values.append({**document, "active": str(document["version"]) == active_version})
        return sorted(values, key=lambda value: float(value.get("created_at", 0)), reverse=True)

    def activate(self, profile_id: str, version: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(version):
            raise TargetStoreError("invalid target version")
        path = self._profile_directory(profile_id) / "versions" / f"{version}.json"
        if not path.is_file():
            raise TargetStoreError("target version does not exist")
        document = self._read(path, profile_id)
        self._atomic_write(self._profile_directory(profile_id) / "active.json", document)
        return document

    def _profile_directory(self, profile_id: str) -> Path:
        if not _SAFE_ID.fullmatch(profile_id):
            raise TargetStoreError("invalid profile id")
        return self.root / profile_id

    def _read(self, path: Path, profile_id: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TargetStoreError(f"cannot read target configuration: {path.name}") from error
        if not isinstance(value, dict) or value.get("profile_id") != profile_id:
            raise TargetStoreError("target configuration has an invalid profile id")
        _pose(value.get("expected_table_pose"), "expected_table_pose")
        _pose(value.get("tag_from_table"), "tag_from_table")
        _mapping(value.get("tag"), "tag")
        if not _SAFE_ID.fullmatch(str(value.get("version", ""))):
            raise TargetStoreError("target configuration has an invalid version")
        return value

    @staticmethod
    def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetStoreError(f"{name} must be an object")
    return value


def _pose(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise TargetStoreError(f"{name} must contain six numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise TargetStoreError(f"{name} must contain six numbers") from error
    if not all(abs(item) < 1e6 for item in result):
        raise TargetStoreError(f"{name} contains invalid values")
    return result
