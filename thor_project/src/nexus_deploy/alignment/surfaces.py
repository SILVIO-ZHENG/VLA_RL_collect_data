"""Versioned, atomic storage for accepted AprilTag work-surface measurements."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .config import AlignmentProfile

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class SurfaceStoreError(ValueError):
    pass


class SurfaceCalibrationRepository:
    """Store immutable measurements and one atomically selected active value."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def publish(
        self,
        profile: AlignmentProfile,
        measurement: Mapping[str, Any],
        *,
        robot: Mapping[str, Any],
        robot_host: str,
        calibration_version: str,
        config_fingerprint: str,
    ) -> dict[str, Any]:
        now = time.time()
        base_version = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        profile_dir = self._profile_directory(profile.id)
        versions_dir = profile_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        version = base_version
        suffix = 1
        while (versions_dir / f"{version}.json").exists():
            suffix += 1
            version = f"{base_version}-{suffix}"
        document: dict[str, Any] = {
            "schema_version": 1,
            "measurement_id": f"{profile.id}-{version}",
            "profile_id": profile.id,
            "profile_version": profile.version,
            "created_at": now,
            "base_frame": "base_link",
            "surface": dict(measurement),
            "tag": {
                "family": profile.tag_family,
                "ids": list(profile.tag_ids),
                "size_m": profile.tag_size_m,
                "tag_from_table": [
                    float(value) for value in _matrix_pose(profile.tag_from_table)
                ],
            },
            "robot": {
                "id": robot.get("id"),
                "aid": robot.get("aid"),
                "model": robot.get("model"),
                "host": robot_host,
            },
            "camera_calibration_version": calibration_version,
            "alignment_config_fingerprint": config_fingerprint,
            "robot_model": {
                "id": profile.robot_model_id,
                "sha256": profile.robot_model_sha256,
            },
        }
        self._atomic_write(versions_dir / f"{version}.json", document)
        self._atomic_write(profile_dir / "active.json", document)
        return document

    def active(self, profile_id: str) -> dict[str, Any] | None:
        path = self._profile_directory(profile_id) / "active.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SurfaceStoreError("cannot read active surface calibration") from error
        if not isinstance(value, dict) or value.get("profile_id") != profile_id:
            raise SurfaceStoreError("active surface calibration has an invalid profile id")
        return value

    def _profile_directory(self, profile_id: str) -> Path:
        if not _SAFE_ID.fullmatch(profile_id):
            raise SurfaceStoreError("invalid surface profile id")
        return self.root / profile_id

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


def _matrix_pose(transform: Any) -> tuple[float, ...]:
    # Import lazily to keep this storage module independent of detector extras.
    from .geometry import matrix_pose

    return matrix_pose(transform)
