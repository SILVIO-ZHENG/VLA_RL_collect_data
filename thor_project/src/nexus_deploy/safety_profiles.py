"""Generate immutable robot safety profiles from accepted AprilTag measurements."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.+-]+")
GENERATOR_ID = "nexus-platform-alignment-safety-v3"


class SafetyProfileGenerationError(ValueError):
    """An alignment measurement cannot safely produce a runtime profile."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_component(value: object, fallback: str) -> str:
    result = SAFE_COMPONENT.sub("_", str(value or "").strip()).strip("._")
    return (result or fallback)[:80]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class AlignmentSafetyProfileRepository:
    """Materialize one deterministic safety snapshot per accepted measurement."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def robot_directory(self, robot_id: str) -> Path:
        return self.root / _safe_component(robot_id, "robot")

    def sync(
        self,
        *,
        measurements_root: Path,
        robot_id: str,
        template_path: Path,
        expected_robot_aid: str = "",
        allowed_robot_hosts: set[str] | None = None,
    ) -> list[Path]:
        """Generate missing profiles and return all valid generated snapshots."""

        measurements_root = Path(measurements_root).expanduser().resolve()
        candidates = sorted(measurements_root.glob("*/versions/*.json"))
        candidates.extend(sorted(measurements_root.glob("*/active.json")))
        seen_measurements: set[str] = set()
        validated_paths: set[Path] = set()
        for measurement_path in candidates:
            try:
                measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
                measurement_id = str(measurement.get("measurement_id", ""))
                if not measurement_id or measurement_id in seen_measurements:
                    continue
                seen_measurements.add(measurement_id)
                validated_paths.add(
                    self.generate(
                        measurement=measurement,
                        measurement_path=measurement_path,
                        robot_id=robot_id,
                        template_path=template_path,
                        expected_robot_aid=expected_robot_aid,
                        allowed_robot_hosts=allowed_robot_hosts,
                    ).resolve()
                )
            except (OSError, TypeError, ValueError, SafetyProfileGenerationError):
                # Invalid, stale or foreign measurements never become selectable.
                continue
        # Keep older immutable snapshots on disk for audit, but a safety
        # template update (for example a new hard margin) must immediately
        # retire snapshots generated from the previous template.
        template_sha = file_sha256(Path(template_path).expanduser().resolve())
        current: list[Path] = []
        for path in self.list(robot_id):
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
                generated = profile.get("generated_profile", {})
                surface_calibration = profile.get("surface_calibration", {})
                if (
                    isinstance(generated, Mapping)
                    and isinstance(surface_calibration, Mapping)
                    and path.resolve() in validated_paths
                    and generated.get("source_template_sha256") == template_sha
                    and (
                        not expected_robot_aid
                        or surface_calibration.get("robot_aid") == expected_robot_aid
                    )
                ):
                    current.append(path)
            except (OSError, TypeError, ValueError):
                continue
        return current

    def generate(
        self,
        *,
        measurement: Mapping[str, Any],
        measurement_path: Path,
        robot_id: str,
        template_path: Path,
        expected_robot_aid: str = "",
        allowed_robot_hosts: set[str] | None = None,
    ) -> Path:
        if measurement.get("schema_version") != 1:
            raise SafetyProfileGenerationError("unsupported alignment schema")
        if measurement.get("base_frame") != "base_link":
            raise SafetyProfileGenerationError("alignment result is not in base_link")
        surface = measurement.get("surface")
        if not isinstance(surface, Mapping) or surface.get("eligible") is not True:
            raise SafetyProfileGenerationError("alignment surface is not eligible")
        try:
            height = float(surface["height_m"])
            created_at = float(measurement["created_at"])
        except (KeyError, TypeError, ValueError) as error:
            raise SafetyProfileGenerationError("invalid surface height or timestamp") from error
        if not math.isfinite(height) or not 0.30 <= height <= 1.50:
            raise SafetyProfileGenerationError("surface height is outside the G2 workspace")
        if not math.isfinite(created_at) or created_at <= 0:
            raise SafetyProfileGenerationError("invalid measurement timestamp")
        pose = surface.get("pose")
        raw_normal = surface.get("normal")
        if not isinstance(pose, Mapping) or not isinstance(raw_normal, list | tuple):
            raise SafetyProfileGenerationError("surface plane origin or normal is missing")
        try:
            origin = [float(pose[key]) for key in ("x_m", "y_m", "z_m")]
            normal = [float(value) for value in raw_normal]
        except (KeyError, TypeError, ValueError) as error:
            raise SafetyProfileGenerationError("invalid surface plane origin or normal") from error
        norm = math.sqrt(sum(value * value for value in normal))
        if (
            len(normal) != 3
            or not all(math.isfinite(value) for value in (*origin, *normal))
            or norm <= 1e-9
        ):
            raise SafetyProfileGenerationError("invalid surface plane origin or normal")
        normal = [value / norm for value in normal]
        if normal[2] <= 0.95:
            raise SafetyProfileGenerationError("surface plane normal is not an upward tabletop")
        if abs(origin[2] - height) > 0.002:
            raise SafetyProfileGenerationError("surface height disagrees with its plane origin")

        measured_robot = measurement.get("robot")
        if not isinstance(measured_robot, Mapping):
            raise SafetyProfileGenerationError("measurement has no robot identity")
        measured_host = str(measured_robot.get("host", "")).strip()
        measured_robot_id = str(measured_robot.get("id", "")).strip()
        measured_aid = str(measured_robot.get("aid", "")).strip()
        if allowed_robot_hosts and measured_host not in allowed_robot_hosts:
            raise SafetyProfileGenerationError("measurement belongs to another robot address")
        if expected_robot_aid and measured_aid != expected_robot_aid:
            raise SafetyProfileGenerationError("measurement belongs to another robot AID")

        template_path = Path(template_path).expanduser().resolve()
        template = json.loads(template_path.read_text(encoding="utf-8"))
        template_model = template.get("robot_model")
        measured_model = measurement.get("robot_model")
        if not isinstance(template_model, Mapping) or not isinstance(measured_model, Mapping):
            raise SafetyProfileGenerationError("robot model metadata is missing")
        template_urdf_sha = str(template_model.get("sha256", ""))
        measured_urdf_sha = str(measured_model.get("sha256", ""))
        expected_model_id = Path(str(template_model.get("path", ""))).stem
        compatible_alignment_hashes = {
            str(value).strip().lower()
            for value in template_model.get("compatible_alignment_urdf_sha256", ())
            if str(value).strip()
        }
        compatible_alignment_hashes.add(template_urdf_sha.lower())
        if (
            not template_urdf_sha
            or measured_urdf_sha.lower() not in compatible_alignment_hashes
            or str(measured_model.get("id", "")) != expected_model_id
        ):
            raise SafetyProfileGenerationError("alignment URDF does not match safety geometry")

        measurement_id = str(measurement.get("measurement_id", "")).strip()
        profile_id = str(measurement.get("profile_id", "")).strip()
        camera_version = str(measurement.get("camera_calibration_version", "")).strip()
        if not measurement_id or not profile_id or not camera_version:
            raise SafetyProfileGenerationError("alignment provenance is incomplete")
        measurement_sha = file_sha256(Path(measurement_path))
        template_sha = file_sha256(template_path)
        local_time = datetime.fromtimestamp(created_at).astimezone()
        timestamp = local_time.strftime("%Y%m%dT%H%M%S%z")
        suffix = hashlib.sha256(
            f"{measurement_id}:{template_sha}:{GENERATOR_ID}".encode("utf-8")
        ).hexdigest()[:8]
        filename = (
            f"G2_dahuan_{_safe_component(profile_id, 'surface')}_{timestamp}_{suffix}.json"
        )
        output = self.robot_directory(robot_id) / filename

        generated = copy.deepcopy(template)
        generated["display_name"] = (
            f"G2 大寰 · {profile_id} · {local_time.strftime('%Y-%m-%d %H:%M:%S')} · 真机测试"
        )
        # A saved alignment has already passed schema, robot-AID, network-endpoint,
        # camera-calibration and URDF-hash validation above.  The site has enabled
        # physical commissioning, so valid snapshots are selectable for execute;
        # runtime identity, preflight, watchdog and operator confirmation remain
        # independent fail-closed gates.
        generated["calibrated"] = True
        generated["read_only_ready"] = True
        inter_arm = generated.get("inter_arm_collision")
        inter_arm_execute_ready = not isinstance(inter_arm, Mapping) or (
            inter_arm.get("enabled") is not True
            or inter_arm.get("validation_status") == "execute-approved"
        )
        generated["execute_approved"] = inter_arm_execute_ready
        measurement_mode = str(surface.get("measurement_mode", "standard"))
        calibration_source = (
            "g2a_alignment_contact_refined"
            if measurement_mode.startswith("contact-refined")
            else "g2a_alignment_apriltag"
        )
        generated["surface_calibration"] = {
            "source": calibration_source,
            "measurement_mode": measurement_mode,
            "profile_id": profile_id,
            "measurement_id": measurement_id,
            "measurement_sha256": measurement_sha,
            "observed_at": created_at,
            "camera_calibration_version": camera_version,
            "urdf_sha256": template_urdf_sha,
            "alignment_urdf_sha256": measured_urdf_sha,
            "robot_aid": measured_aid,
            "robot_id": measured_robot_id,
            "robot_host": measured_host,
            "surface_origin": [round(value, 8) for value in origin],
            "surface_normal": [round(value, 9) for value in normal],
        }
        refinement = measurement.get("refinement")
        if isinstance(refinement, Mapping):
            generated["surface_calibration"]["refinement"] = copy.deepcopy(refinement)
        surfaces = generated.get("surfaces")
        if not isinstance(surfaces, list) or not surfaces or not isinstance(surfaces[0], dict):
            raise SafetyProfileGenerationError("safety template has no work surface")
        surfaces[0]["height"] = round(height, 5)
        surfaces[0]["origin"] = [
            round(origin[0], 5),
            round(origin[1], 5),
            round(height, 5),
        ]
        surfaces[0]["normal"] = [round(value, 9) for value in normal]
        generated["generated_profile"] = {
            "schema_version": 2,
            "generator": GENERATOR_ID,
            "robot_id": robot_id,
            "source_template": template_path.name,
            "source_template_sha256": template_sha,
            "measurement_id": measurement_id,
            "measurement_sha256": measurement_sha,
            "created_at": created_at,
        }
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != generated:
                raise SafetyProfileGenerationError(
                    "generated profile content changed; immutable snapshot is not selectable"
                )
            return output
        _atomic_json(output, generated)
        return output

    def list(self, robot_id: str) -> list[Path]:
        directory = self.robot_directory(robot_id)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.json"), reverse=True)
