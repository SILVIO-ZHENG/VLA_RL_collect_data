#!/usr/bin/env python3
"""Publish a reviewed contact-refined table plane as a temporary active surface."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nexus_deploy.alignment.config import load_config
from nexus_deploy.alignment.geometry import matrix_pose, pose_matrix
from nexus_deploy.safety_profiles import AlignmentSafetyProfileRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("alignment.example.toml"))
    parser.add_argument("--profile", default="standard_table")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Required acknowledgement: replace active.json but preserve all immutable versions.",
    )
    args = parser.parse_args()
    if not args.activate:
        parser.error("--activate is required")
    return args


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def plane_pose(origin: np.ndarray, normal: np.ndarray, source_pose: Mapping[str, Any]) -> dict[str, float]:
    source = pose_matrix(
        [
            float(source_pose[key])
            for key in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg")
        ]
    )
    x_axis = source[:3, 0] - normal * float(np.dot(source[:3, 0], normal))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, normal)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, normal))
    transform[:3, 3] = origin
    x, y, z, roll, pitch, yaw = matrix_pose(transform)
    return {
        "x_m": round(x, 8),
        "y_m": round(y, 8),
        "z_m": round(z, 8),
        "roll_deg": round(roll, 6),
        "pitch_deg": round(pitch, 6),
        "yaw_deg": round(yaw, 6),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.profile not in config.profiles:
        raise ValueError(f"unknown alignment profile: {args.profile}")
    candidate_path = args.candidate.expanduser().resolve()
    candidate = read_object(candidate_path)
    if candidate.get("status") != "read-only-candidate-not-active":
        raise ValueError("candidate is not an inactive reviewed read-only surface")
    if candidate.get("method") != (
        "retain_apriltag_x_slope_and_fit_y_slope_plus_offset_from_confirmed_contacts"
    ):
        raise ValueError("candidate was produced by an unsupported refinement method")
    coverage = candidate.get("coverage", {})
    if int(coverage.get("confirmed_contact_constraints", 0)) < 5:
        raise ValueError("candidate does not contain five confirmed contact constraints")
    validation = candidate.get("validation", {})
    if float(validation.get("max_abs_error_mm", math.inf)) > 1.0:
        raise ValueError("candidate contact residual exceeds the temporary 1 mm gate")

    surface_root = config.surfaces_directory / args.profile
    active_path = surface_root / "active.json"
    source = read_object(active_path)
    if source.get("measurement_id") != candidate.get("source_measurement_id"):
        raise ValueError("active AprilTag measurement does not match the candidate source")
    candidate_plane = candidate.get("candidate_plane")
    if not isinstance(candidate_plane, Mapping):
        raise ValueError("candidate plane is missing")
    origin = np.asarray(candidate_plane.get("origin_m"), dtype=np.float64)
    normal = np.asarray(candidate_plane.get("normal"), dtype=np.float64)
    if origin.shape != (3,) or normal.shape != (3,) or not np.isfinite((*origin, *normal)).all():
        raise ValueError("candidate plane is invalid")
    normal /= np.linalg.norm(normal)
    if normal[2] <= 0.95 or not 0.3 <= origin[2] <= 1.5:
        raise ValueError("candidate plane is outside the G2 tabletop workspace")

    now = time.time()
    version = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-contact-refined"
    document = copy.deepcopy(source)
    document["created_at"] = now
    document["measurement_id"] = f"{args.profile}-{version}"
    document["robot_model"] = {
        "id": "G2_dahuan_safety",
        "sha256": str(candidate["safety_urdf_sha256"]),
    }
    source_surface = source["surface"]
    surface = document["surface"]
    surface.update(
        {
            "eligible": True,
            "reason": "eligible-contact-refined-temporary",
            "measurement_mode": "contact-refined-temporary",
            "height_m": round(float(origin[2]), 8),
            "pose": plane_pose(origin, normal, source_surface["pose"]),
            "normal": [round(float(value), 9) for value in normal],
            "tilt_deg": round(math.degrees(math.acos(float(normal[2]))), 6),
        }
    )
    context = dict(surface.get("observation_context", {}))
    context["contact_refinement"] = {
        "temporary": True,
        "source_candidate": str(candidate_path),
        "source_candidate_sha256": sha256(candidate_path),
        "confirmed_contact_constraints": int(coverage["confirmed_contact_constraints"]),
        "x_span_m": float(coverage["x_span_m"]),
        "y_span_m": float(coverage["y_span_m"]),
        "x_tilt_status": str(coverage["x_tilt_status"]),
        "y_tilt_status": str(coverage["y_tilt_status"]),
        "rmse_mm": float(validation["rmse_mm"]),
        "max_abs_error_mm": float(validation["max_abs_error_mm"]),
    }
    surface["observation_context"] = context
    document["refinement"] = {
        "type": "contact-plane-temporary",
        "source_measurement_id": str(source["measurement_id"]),
        "source_candidate_sha256": sha256(candidate_path),
        "retained_apriltag_x_slope": True,
        "final_camera_extrinsic_calibration_pending": True,
        "replacement_apriltag_pending": True,
    }

    version_path = surface_root / "versions" / f"{version}.json"
    if version_path.exists():
        raise FileExistsError(version_path)
    atomic_json(version_path, document)
    atomic_json(active_path, document)

    if config.safety_profiles_directory is None or config.safety_profile_template is None:
        raise RuntimeError("alignment config does not define safety profile generation")
    generated = AlignmentSafetyProfileRepository(config.safety_profiles_directory).generate(
        measurement=document,
        measurement_path=active_path,
        robot_id=config.safety_robot_id,
        template_path=config.safety_profile_template,
        expected_robot_aid=config.safety_expected_aid,
        allowed_robot_hosts={str(document["robot"]["host"])},
    )
    print(
        json.dumps(
            {
                "active_surface": str(active_path),
                "immutable_surface": str(version_path),
                "safety_profile": str(generated),
                "measurement_id": document["measurement_id"],
                "mode": surface["measurement_mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
