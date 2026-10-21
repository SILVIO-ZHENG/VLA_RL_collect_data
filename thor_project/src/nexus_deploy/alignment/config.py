"""Versioned configuration loading for the standalone alignment service."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .geometry import pose_matrix


@dataclass(frozen=True)
class Tolerance:
    forward_m: float = 0.02
    lateral_m: float = 0.02
    yaw_deg: float = 2.0
    height_m: float = 0.02
    tilt_deg: float = 2.0
    stable_s: float = 1.0
    window_s: float = 1.0
    minimum_samples: int = 5
    minimum_detection_ratio: float = 0.8


@dataclass(frozen=True)
class AlignmentProfile:
    id: str
    display_name: str
    version: str
    camera_name: str
    tag_family: str
    tag_ids: tuple[int, ...]
    tag_size_m: float
    base_from_expected_table: np.ndarray
    tag_from_table: np.ndarray
    surface_offset_m: float = 0.0
    robot_model_id: str = ""
    robot_model_sha256: str = ""
    tolerance: Tolerance = field(default_factory=Tolerance)
    inspection_joints: Mapping[str, float] = field(default_factory=dict)
    inspection_joint_tolerance_rad: float = 0.005
    # Site-level compatibility for an imperfect print. Only an exactly decoded
    # (Hamming distance 0) fallback marker may be used for a measurement.
    allow_hamming0_fallback_measurement: bool = False
    target_configured: bool = True

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "version": self.version,
            "target_configured": self.target_configured,
            "camera_name": self.camera_name,
            "tag": {
                "family": self.tag_family,
                "ids": list(self.tag_ids),
                "size_m": self.tag_size_m,
            },
            "surface": {
                "frame": "base_link",
                "offset_m": self.surface_offset_m,
                "robot_model_id": self.robot_model_id,
                "robot_model_sha256": self.robot_model_sha256,
                "allow_hamming0_fallback_measurement": self.allow_hamming0_fallback_measurement,
            },
            "tolerance": {
                "forward_mm": self.tolerance.forward_m * 1000,
                "lateral_mm": self.tolerance.lateral_m * 1000,
                "yaw_deg": self.tolerance.yaw_deg,
                "height_mm": self.tolerance.height_m * 1000,
                "tilt_deg": self.tolerance.tilt_deg,
                "stable_s": self.tolerance.stable_s,
            },
        }


@dataclass(frozen=True)
class AdapterConfig:
    id: str
    display_name: str
    plugin: str
    settings: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "default_port": self.settings.get("port"),
        }


@dataclass(frozen=True)
class DetectorConfig:
    id: str
    plugin: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class AlignmentConfig:
    path: Path
    reports_directory: Path
    targets_directory: Path
    surfaces_directory: Path
    safety_profiles_directory: Path | None
    safety_profile_template: Path | None
    safety_robot_id: str
    safety_expected_aid: str
    default_profile: str
    default_adapter: str
    default_detector: str
    default_robot_host: str
    profiles: Mapping[str, AlignmentProfile]
    adapters: Mapping[str, AdapterConfig]
    detectors: Mapping[str, DetectorConfig]
    reconnect_grace_s: float = 30.0
    tag_loss_grace_s: float = 0.7
    update_hz: float = 8.0


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a table")
    return value


def _pose(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be [x, y, z, roll, pitch, yaw]")
    return pose_matrix(value)


def load_config(path: str | Path) -> AlignmentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    service = _mapping(raw.get("service", {}), "service")
    raw_profiles = _mapping(raw.get("profiles", {}), "profiles")
    raw_adapters = _mapping(raw.get("adapters", {}), "adapters")
    raw_detectors = _mapping(raw.get("detectors", {}), "detectors")
    profiles: dict[str, AlignmentProfile] = {}
    for profile_id, item_value in raw_profiles.items():
        item = _mapping(item_value, f"profiles.{profile_id}")
        tolerance_data = _mapping(item.get("tolerance", {}), "tolerance")
        tolerance = Tolerance(
            forward_m=float(tolerance_data.get("forward_m", 0.02)),
            lateral_m=float(tolerance_data.get("lateral_m", 0.02)),
            yaw_deg=float(tolerance_data.get("yaw_deg", 2.0)),
            height_m=float(tolerance_data.get("height_m", 0.02)),
            tilt_deg=float(tolerance_data.get("tilt_deg", 2.0)),
            stable_s=float(tolerance_data.get("stable_s", 1.0)),
            window_s=float(tolerance_data.get("window_s", 1.0)),
            minimum_samples=int(tolerance_data.get("minimum_samples", 5)),
            minimum_detection_ratio=float(tolerance_data.get("minimum_detection_ratio", 0.8)),
        )
        if not 0 < tolerance.minimum_detection_ratio <= 1:
            raise ValueError("minimum_detection_ratio must be in (0, 1]")
        if tolerance.minimum_samples < 2 or tolerance.window_s <= 0 or tolerance.stable_s <= 0:
            raise ValueError("profile timing and minimum_samples must be positive")
        tag_ids = tuple(int(value) for value in item.get("tag_ids", ()))
        if not tag_ids or len(set(tag_ids)) != len(tag_ids):
            raise ValueError(f"profiles.{profile_id}.tag_ids must contain unique ids")
        profiles[str(profile_id)] = AlignmentProfile(
            id=str(profile_id),
            display_name=str(item.get("display_name", profile_id)),
            version=str(item.get("version", "1")),
            camera_name=str(item.get("camera_name", "cam_head")),
            tag_family=str(item.get("tag_family", "tag36h11")),
            tag_ids=tag_ids,
            tag_size_m=float(item.get("tag_size_m", 0.12)),
            base_from_expected_table=_pose(item.get("expected_table_pose"), "expected_table_pose"),
            tag_from_table=_pose(item.get("tag_from_table", [0, 0, 0, 0, 0, 0]), "tag_from_table"),
            surface_offset_m=float(item.get("surface_offset_m", 0.0)),
            robot_model_id=str(item.get("robot_model_id", "")).strip(),
            robot_model_sha256=str(item.get("robot_model_sha256", "")).strip().lower(),
            tolerance=tolerance,
            inspection_joints={
                str(name): float(position)
                for name, position in _mapping(item.get("inspection_joints", {}), "inspection_joints").items()
            },
            inspection_joint_tolerance_rad=float(item.get("inspection_joint_tolerance_rad", 0.005)),
            allow_hamming0_fallback_measurement=bool(
                item.get("allow_hamming0_fallback_measurement", False)
            ),
            # Defaults to true for compatibility with existing site configs. New
            # installs should explicitly mark example/placeholder poses false.
            target_configured=bool(item.get("target_configured", True)),
        )
    adapters = {
        str(adapter_id): AdapterConfig(
            id=str(adapter_id),
            display_name=str(_mapping(value, "adapter").get("display_name", adapter_id)),
            plugin=str(_mapping(value, "adapter")["plugin"]),
            settings=dict(_mapping(value, "adapter").get("settings", {})),
        )
        for adapter_id, value in raw_adapters.items()
    }
    detectors = {
        str(detector_id): DetectorConfig(
            id=str(detector_id),
            plugin=str(_mapping(value, "detector")["plugin"]),
            settings=dict(_mapping(value, "detector").get("settings", {})),
        )
        for detector_id, value in raw_detectors.items()
    }
    if not profiles or not adapters or not detectors:
        raise ValueError("at least one profile, adapter, and detector are required")
    default_profile = str(service.get("default_profile", next(iter(profiles))))
    default_adapter = str(service.get("default_adapter", next(iter(adapters))))
    default_detector = str(service.get("default_detector", next(iter(detectors))))
    if default_profile not in profiles or default_adapter not in adapters or default_detector not in detectors:
        raise ValueError("configured default profile/adapter/detector does not exist")
    reports_value = Path(str(service.get("reports_directory", "runtime/alignment-reports"))).expanduser()
    reports_directory = reports_value if reports_value.is_absolute() else config_path.parent / reports_value
    targets_value = Path(str(service.get("targets_directory", "runtime/alignment-targets"))).expanduser()
    targets_directory = targets_value if targets_value.is_absolute() else config_path.parent / targets_value
    surfaces_value = Path(
        str(service.get("surfaces_directory", "runtime/alignment-surfaces"))
    ).expanduser()
    surfaces_directory = (
        surfaces_value if surfaces_value.is_absolute() else config_path.parent / surfaces_value
    )
    safety_profiles_raw = str(service.get("safety_profiles_directory", "")).strip()
    safety_template_raw = str(service.get("safety_profile_template", "")).strip()
    safety_profiles_value = Path(safety_profiles_raw).expanduser() if safety_profiles_raw else None
    safety_template_value = Path(safety_template_raw).expanduser() if safety_template_raw else None
    safety_profiles_directory = (
        None
        if safety_profiles_value is None
        else (
            safety_profiles_value
            if safety_profiles_value.is_absolute()
            else config_path.parent / safety_profiles_value
        ).resolve()
    )
    safety_profile_template = (
        None
        if safety_template_value is None
        else (
            safety_template_value
            if safety_template_value.is_absolute()
            else config_path.parent / safety_template_value
        ).resolve()
    )
    return AlignmentConfig(
        path=config_path,
        reports_directory=reports_directory.resolve(),
        targets_directory=targets_directory.resolve(),
        surfaces_directory=surfaces_directory.resolve(),
        safety_profiles_directory=safety_profiles_directory,
        safety_profile_template=safety_profile_template,
        safety_robot_id=str(service.get("safety_robot_id", "g2a")).strip() or "g2a",
        safety_expected_aid=str(service.get("safety_expected_aid", "")).strip(),
        default_profile=default_profile,
        default_adapter=default_adapter,
        default_detector=default_detector,
        default_robot_host=str(service.get("default_robot_host", "")).strip(),
        profiles=profiles,
        adapters=adapters,
        detectors=detectors,
        reconnect_grace_s=float(service.get("reconnect_grace_s", 30.0)),
        tag_loss_grace_s=float(service.get("tag_loss_grace_s", 0.7)),
        update_hz=float(service.get("update_hz", 8.0)),
    )


def config_fingerprint(config: AlignmentConfig) -> str:
    payload = {
        "path": str(config.path),
        "profiles": {key: value.public_dict() for key, value in config.profiles.items()},
    }
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
