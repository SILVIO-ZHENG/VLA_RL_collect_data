#!/usr/bin/env python3
"""Operate the dual-model G2A deployment without mixing model environments."""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from nexus_deploy.rtc_modes import (
    rtc_inference_mode_for_args, rtc_metadata_for_args,
    with_rtc_inference_mode, validate_rtc_inference_mode,
)

from nexus_deploy.safety_profiles import (
    AlignmentSafetyProfileRepository,
    GENERATOR_ID,
    SafetyProfileGenerationError,
)

MODEL_ALL = "all"
POLICY_PROTOCOL_NAME = "nexus-policy-websocket"
POLICY_PROTOCOL_VERSION = 2
BODY_RESET_PROFILE_ID = "g2_sorting_home_20260817_v1"
POLICY_PROTOCOL_IDENTITIES = (
    (POLICY_PROTOCOL_NAME, POLICY_PROTOCOL_VERSION),
    ("g2a-policy-websocket", 2),
    ("g2a-pi05-websocket", 1),
)
GENERATED_SAFETY_PROFILE_ID = GENERATOR_ID
G2_SAFETY_RUNTIME_CAPABILITY_VERSION = 15
ROBOT_BACKUP_RETENTION = 2
DEFAULT_ROBOT_SESSION_REMOTE_ROOT = "/home/agi/.cache/nexus-platform-sessions"
DEFAULT_ROBOT_SESSION_REMOTE_RETENTION = 5
G2_ALIGNMENT_SERVICE = "nexus-g2-alignment-agent.service"
ROBOT_CLIENT_REQUIRED_CLI_OPTIONS = (
    "--source-address",
    "--effectors-mode",
    "--effector-profile",
    "--effector-control-mode",
    "--effector-command-mode",
    "--effector-closed-max",
    "--effector-open-min",
    "--fixed-left-effector-state",
    "--fixed-right-effector-state",
    "--control-file",
    "--control-status-file",
    "--platform-telemetry-endpoint",
    "--platform-telemetry-token-file",
    "--platform-frame-dir",
    "--platform-presence-frame-endpoint",
    "--platform-presence-frame-token-file",
    "--platform-presence-job-id",
    "--platform-presence-generation",
    "--platform-presence-frame-fps",
    "--start-paused",
    "--stationary-hold-feedback-max-age-ms",
    "--safety-monitor-only",
    "--async-safety",
    "--rolling-safety",
    "--rolling-geometry",
    "--chunk-smoothing-mode",
    "--chunk-smoothing-window",
    "--chunk-smoothing-max-deviation",
    "--adaptive-smoothing-acceleration-weight",
    "--adaptive-smoothing-jerk-weight",
    "--adaptive-smoothing-velocity-reference",
    "--fixed-horizon-projection",
    "--arm-limit-profile",
    "--arm-tracking-mode",
    "--arm-tracking-omega",
    "--rtc-prefix-mode",
)

DEFAULT_EFFECTOR_PROFILES: dict[str, dict[str, Any]] = {
    "omnipicker_legacy": {
        "display_name": "智元原装 Omnipicker 夹爪",
        "default_task": "Pick up the package and place it on the conveyor belt with the QR code facing up.",
        "unit": "rad",
        "hardware_state_range": [-0.785, 0.0],
        "model_action_range": [-0.785, 0.0],
        "action_mapping": "identity",
        "default_max_velocity": 1.0,
        "default_max_acceleration": 4.0,
        "default_tracking_error": 0.5,
        "default_relative_target_warning": 0.05,
        "state_semantics": "沿用旧模型/GDK 关节位置语义",
        "action_semantics": "模型目标与 GDK 目标同向",
        "gdk_joint_names": [
            "idx31_gripper_l_inner_joint1",
            "idx71_gripper_r_inner_joint1",
        ],
        "collision_geometry": "omnipicker",
        "collision_geometry_status": "verified",
        "ui_hidden": False,
    },
    "dahuan": {
        "display_name": "大寰夹爪",
        "default_task": (
            "Pick a package from the box with the left gripper and place it on the table. "
            "Then pick it up with the right gripper and place it on the conveyor belt, "
            "keeping the QR code facing upward."
        ),
        "unit": "m",
        "hardware_state_range": [0.0, 0.025],
        "model_action_range": [0.0, 0.025],
        "action_mapping": "invert_about_endpoints",
        "default_max_velocity": 0.05,
        "default_max_acceleration": 0.2,
        "default_tracking_error": 0.01,
        "default_relative_target_warning": 0.025,
        "state_semantics": "0=闭合，0.025=张开",
        "action_semantics": "0=张开指令，0.025=闭合指令",
        "gdk_joint_names": [
            "idx31_gripper_l_inner_joint1",
            "idx71_gripper_r_inner_joint1",
        ],
        "collision_geometry": "dahuan",
        "collision_geometry_status": "provisional",
        "ui_hidden": False,
    },
}

EFFECTOR_PROFILE_ALIASES = {"dh": "dahuan"}


def canonical_effector_profile(value: Any) -> str:
    """Return the current profile id while accepting persisted legacy ids."""

    profile = str(value)
    return EFFECTOR_PROFILE_ALIASES.get(profile, profile)


def is_supported_policy_protocol(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    identity = (value.get("name"), value.get("version"))
    return identity in POLICY_PROTOCOL_IDENTITIES


class DeployError(RuntimeError):
    """A deployment precondition or operation failed safely."""


@dataclass(frozen=True)
class EffectorProfileConfig:
    name: str
    display_name: str
    default_task: str
    unit: str
    hardware_state_range: tuple[float, float]
    model_action_range: tuple[float, float]
    action_mapping: str
    default_max_velocity: float
    default_max_acceleration: float
    default_tracking_error: float
    default_relative_target_warning: float
    state_semantics: str
    action_semantics: str
    gdk_joint_names: tuple[str, str]
    collision_geometry: str
    collision_geometry_status: str
    ui_hidden: bool


@dataclass(frozen=True)
class ModelVersionConfig:
    id: str
    display_name: str
    description: str
    effector_profile: str
    base_args: tuple[str, ...]
    real_args: tuple[str, ...]
    mock_args: tuple[str, ...]
    normalization_source: str
    normalization_label: str


@dataclass(frozen=True)
class ModelEffectorVariantConfig:
    effector_profile: str
    base_args: tuple[str, ...]
    real_args: tuple[str, ...]
    mock_args: tuple[str, ...]
    normalization_source: str
    normalization_label: str
    default_version: str
    versions: dict[str, ModelVersionConfig]

    def version(self, version_id: str | None = None) -> ModelVersionConfig:
        selected = self.default_version if not version_id else str(version_id)
        try:
            return self.versions[selected]
        except KeyError as error:
            raise DeployError(
                f"Effector profile {self.effector_profile} has no model version {selected}; "
                f"configured={tuple(self.versions)}"
            ) from error


@dataclass(frozen=True)
class ModelConfig:
    name: str
    repo: Path
    conda_env: str
    python_executable: Path | None
    module: str
    port: int
    cuda_visible_devices: str
    startup_timeout_s: float
    base_args: tuple[str, ...]
    real_args: tuple[str, ...]
    mock_args: tuple[str, ...]
    effector_profile: str
    normalization_source: str
    normalization_label: str
    effector_profiles: tuple[str, ...]
    effector_variants: dict[str, ModelEffectorVariantConfig]

    python_paths: tuple[str, ...] = ()

    def variant(self, effector_profile: str | None = None) -> ModelEffectorVariantConfig:
        selected = (
            self.effector_profile
            if effector_profile is None
            else canonical_effector_profile(effector_profile)
        )
        try:
            return self.effector_variants[selected]
        except KeyError as error:
            raise DeployError(
                f"Model {self.name} has no variant for effector profile {selected}; "
                f"configured={self.effector_profiles}"
            ) from error

    def version(
        self,
        effector_profile: str | None = None,
        version_id: str | None = None,
    ) -> ModelVersionConfig:
        return self.variant(effector_profile).version(version_id)


@dataclass(frozen=True)
class RobotConfig:
    name: str
    source_client: Path
    bundle_directory: Path
    manifest: Path
    source_check: str
    ssh_target: str
    ssh_port: int
    ssh_key: Path
    host_key_alias: str
    remote_directory: str
    legacy_remote_directory: str
    reset_script: str
    effectors_mode: str
    fixed_effector_state: tuple[float, float]
    policy_host: str
    policy_source_host: str
    expected_aid: str
    effector_profile: str
    effector_profiles: tuple[str, ...]


class Deployment:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.expanduser().resolve()
        self.project_root = self.config_path.parent
        with self.config_path.open("rb") as stream:
            self.raw = tomllib.load(stream)

        runtime = self.raw["runtime"]
        self.runtime_dir = self._resolve(runtime["directory"])
        self.conda = Path(runtime["conda"]).expanduser().resolve()
        self.require_unique_model_gpus = bool(runtime.get("require_unique_model_gpus", False))
        self.stationary_hold_feedback_max_age_ms = runtime.get("stationary_hold_feedback_max_age_ms", 40)
        if self.stationary_hold_feedback_max_age_ms not in (40, 100):
            raise DeployError("stationary_hold_feedback_max_age_ms must be 40 or 100")
        self.robot_rt_cadence = runtime.get("robot_rt_cadence", False)
        if not isinstance(self.robot_rt_cadence, bool):
            raise DeployError("robot_rt_cadence must be a boolean")
        self.robot_pause_on_collision = runtime.get("robot_pause_on_collision", False)
        self.robot_rtc_state_aligned_anchor = runtime.get("robot_rtc_state_aligned_anchor", False)
        if not isinstance(self.robot_pause_on_collision, bool) or not isinstance(self.robot_rtc_state_aligned_anchor, bool):
            raise DeployError("Robot motion guard switches must be boolean")
        self.robot_session_archive_root = self._resolve(
            runtime.get("robot_session_archive_directory", self.runtime_dir / "robot-sessions")
        )
        self.robot_session_remote_root = self._validate_robot_session_root(
            str(runtime.get("robot_session_remote_directory", DEFAULT_ROBOT_SESSION_REMOTE_ROOT))
        )
        self.robot_session_remote_retention = int(
            runtime.get("robot_session_remote_retention", DEFAULT_ROBOT_SESSION_REMOTE_RETENTION)
        )
        if not 1 <= self.robot_session_remote_retention <= 100:
            raise DeployError("robot_session_remote_retention must be in [1, 100]")
        self.generated_safety_profiles_root = (self.project_root / "runtime" / "safety-profiles").resolve()
        self.alignment_surfaces_root = (self.project_root / "runtime" / "alignment-surfaces").resolve()
        server = self.raw["server"]
        self.bind_host = str(server["bind_host"])
        self.advertise_host = str(server["advertise_host"])
        raw_effector_profiles = self.raw.get("effector_profiles", DEFAULT_EFFECTOR_PROFILES)
        if not isinstance(raw_effector_profiles, dict) or not raw_effector_profiles:
            raise DeployError("At least one [effector_profiles.<name>] entry is required")
        self.effector_profiles = {}
        for raw_name, data in raw_effector_profiles.items():
            name = canonical_effector_profile(raw_name)
            if name in self.effector_profiles:
                raise DeployError(f"Duplicate effector profile after alias migration: {name}")
            self.effector_profiles[name] = self._load_effector_profile(name, data)
        self.models = {name: self._load_model(name, data) for name, data in self.raw["models"].items()}
        if not self.models:
            raise DeployError("At least one [models.<name>] entry is required")

        # ``[robot]`` is the original single-G2 format.  ``[robots.<name>]`` is
        # the extensible format; accepting both keeps existing deployments valid.
        robots = self.raw.get("robots")
        if robots is None:
            legacy_robot = self.raw.get("robot")
            if legacy_robot is None:
                raise DeployError("Configure [robots.<name>] (or legacy [robot])")
            robots = {"default": legacy_robot}
        self.robots = {name: self._load_robot(name, data) for name, data in robots.items()}
        if not self.robots:
            raise DeployError("At least one [robots.<name>] entry is required")
        # Backward-compatible alias used by existing automation.
        self.robot = next(iter(self.robots.values()))

    @staticmethod
    def _finite_range(name: str, field: str, value: Any) -> tuple[float, float]:
        if not isinstance(value, list | tuple) or len(value) != 2:
            raise DeployError(f"{field} for effector profile {name} must contain two values")
        lower, upper = (float(item) for item in value)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise DeployError(f"{field} for effector profile {name} must be a finite increasing range")
        return lower, upper

    def _load_effector_profile(self, name: str, data: dict[str, Any]) -> EffectorProfileConfig:
        action_mapping = str(data.get("action_mapping", "identity"))
        if action_mapping not in {"identity", "invert_about_endpoints"}:
            raise DeployError(f"Invalid action_mapping for effector profile {name}: {action_mapping}")
        geometry_status = str(data.get("collision_geometry_status", "unavailable"))
        if geometry_status not in {"verified", "provisional", "unavailable"}:
            raise DeployError(f"Invalid collision_geometry_status for effector profile {name}: {geometry_status}")
        names = data.get("gdk_joint_names", ())
        if (
            not isinstance(names, list | tuple)
            or len(names) != 2
            or not all(isinstance(item, str) and item for item in names)
        ):
            raise DeployError(f"gdk_joint_names for effector profile {name} must contain two names")
        tuning = {
            field: float(data[field])
            for field in (
                "default_max_velocity",
                "default_max_acceleration",
                "default_tracking_error",
                "default_relative_target_warning",
            )
        }
        if not all(math.isfinite(value) and value > 0 for value in tuning.values()):
            raise DeployError(f"Effector tuning defaults for profile {name} must be positive")
        default_task = str(data.get("default_task", "")).strip()
        if not default_task or len(default_task) > 4096:
            raise DeployError(f"default_task for effector profile {name} must contain 1..4096 characters")
        return EffectorProfileConfig(
            name=name,
            display_name=str(data.get("display_name", name)),
            default_task=default_task,
            unit=str(data.get("unit", "unit")),
            hardware_state_range=self._finite_range(name, "hardware_state_range", data.get("hardware_state_range")),
            model_action_range=self._finite_range(name, "model_action_range", data.get("model_action_range")),
            action_mapping=action_mapping,
            default_max_velocity=tuning["default_max_velocity"],
            default_max_acceleration=tuning["default_max_acceleration"],
            default_tracking_error=tuning["default_tracking_error"],
            default_relative_target_warning=tuning["default_relative_target_warning"],
            state_semantics=str(data.get("state_semantics", "")),
            action_semantics=str(data.get("action_semantics", "")),
            gdk_joint_names=(str(names[0]), str(names[1])),
            collision_geometry=str(data.get("collision_geometry", "")),
            collision_geometry_status=geometry_status,
            ui_hidden=bool(data.get("ui_hidden", False)),
        )

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _interpreter(self, value: str) -> Path:
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else self.project_root / path
        return Path(os.path.normpath(candidate.absolute()))

    def _load_model_version(
        self,
        model_name: str,
        effector_profile: str,
        version_id: str,
        data: dict[str, Any],
        defaults: dict[str, Any],
    ) -> ModelVersionConfig:
        if effector_profile not in self.effector_profiles:
            raise DeployError(f"Unknown effector_profile for model {model_name}: {effector_profile}")
        normalization_source = str(
            data.get(
                "normalization_source",
                defaults.get("normalization_source", "checkpoint_embedded"),
            )
        )
        if normalization_source not in {"checkpoint_embedded", "external_training_asset"}:
            raise DeployError(f"Invalid normalization_source for model {model_name}: {normalization_source}")
        common_real_args = tuple(str(value) for value in defaults.get("common_real_args", ()))
        real_args = common_real_args + tuple(
            str(value) for value in data.get("real_args", defaults.get("real_args", ()))
        )
        if normalization_source == "external_training_asset" and not any(
            argument.startswith("--robot-norm-path=") for argument in real_args
        ):
            raise DeployError(
                f"Model {model_name}/{effector_profile} declares external_training_asset but has no --robot-norm-path"
            )
        base_args = tuple(str(value) for value in data.get("base_args", defaults.get("base_args", ())))
        if any(argument.startswith("--effector-profile=") for argument in base_args):
            raise DeployError(
                f"Model {model_name}/{effector_profile} base_args must not hard-code "
                "--effector-profile; Nexus injects it from the selected gripper"
            )
        return ModelVersionConfig(
            id=version_id,
            display_name=str(data.get("display_name", version_id)),
            description=str(data.get("description", "")),
            effector_profile=effector_profile,
            base_args=base_args,
            real_args=real_args,
            mock_args=tuple(str(value) for value in data.get("mock_args", defaults.get("mock_args", ()))),
            normalization_source=normalization_source,
            normalization_label=str(
                data.get(
                    "normalization_label",
                    defaults.get("normalization_label", normalization_source),
                )
            ),
        )

    def _load_model_variant(
        self,
        model_name: str,
        effector_profile: str,
        data: dict[str, Any],
        defaults: dict[str, Any],
    ) -> ModelEffectorVariantConfig:
        raw_versions = data.get("versions")
        if raw_versions is None:
            default_version = str(data.get("default_version", "default"))
            versions = {
                default_version: self._load_model_version(
                    model_name,
                    effector_profile,
                    default_version,
                    data,
                    defaults,
                )
            }
        else:
            if not isinstance(raw_versions, dict) or not raw_versions:
                raise DeployError(
                    f"Model {model_name}/{effector_profile} versions must be a non-empty table"
                )
            inherited = {**defaults, **{key: value for key, value in data.items() if key != "versions"}}
            versions = {}
            for raw_id, raw_version in raw_versions.items():
                version_id = str(raw_id)
                if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", version_id):
                    raise DeployError(
                        f"Invalid model version id for {model_name}/{effector_profile}: {version_id}"
                    )
                if not isinstance(raw_version, dict):
                    raise DeployError(
                        f"Model version {model_name}/{effector_profile}/{version_id} must be a table"
                    )
                versions[version_id] = self._load_model_version(
                    model_name,
                    effector_profile,
                    version_id,
                    raw_version,
                    inherited,
                )
            default_version = str(data.get("default_version", next(iter(versions))))
        if default_version not in versions:
            raise DeployError(
                f"Model {model_name}/{effector_profile} default_version {default_version} is not configured"
            )
        selected = versions[default_version]
        return ModelEffectorVariantConfig(
            effector_profile=effector_profile,
            base_args=selected.base_args,
            real_args=selected.real_args,
            mock_args=selected.mock_args,
            normalization_source=selected.normalization_source,
            normalization_label=selected.normalization_label,
            default_version=default_version,
            versions=versions,
        )

    def _load_model(self, name: str, data: dict[str, Any]) -> ModelConfig:
        raw_variants = data.get("effector_variants")
        if raw_variants is None:
            default_profile = str(data.get("effector_profile", "omnipicker_legacy"))
            raw_variants = {default_profile: {}}
        elif not isinstance(raw_variants, dict) or not raw_variants:
            raise DeployError(f"Model {name} effector_variants must be a non-empty table")
        variants = {
            canonical_effector_profile(profile): self._load_model_variant(
                name,
                canonical_effector_profile(profile),
                value if isinstance(value, dict) else {},
                data,
            )
            for profile, value in raw_variants.items()
        }
        default_profile = canonical_effector_profile(
            data.get("default_effector_profile", data.get("effector_profile", next(iter(variants))))
        )
        if default_profile not in variants:
            raise DeployError(f"Model {name} default_effector_profile {default_profile} has no variant")
        default_variant = variants[default_profile]
        return ModelConfig(
            name=name,
            repo=self._resolve(data["repo"]),
            conda_env=str(data["conda_env"]),
            # Keep the venv executable path itself: Path.resolve() follows its
            # symlink to the base interpreter and can bypass the venv.
            python_executable=(
                self._interpreter(data["python_executable"])
                if data.get("python_executable") else None
            ),
            module=str(data["module"]),
            python_paths=tuple(str(self._resolve(p)) for p in data.get("python_paths", [])),
            port=int(data["port"]),
            cuda_visible_devices=str(data["cuda_visible_devices"]),
            startup_timeout_s=float(data["startup_timeout_s"]),
            base_args=default_variant.base_args,
            real_args=default_variant.real_args,
            mock_args=default_variant.mock_args,
            effector_profile=default_profile,
            normalization_source=default_variant.normalization_source,
            normalization_label=default_variant.normalization_label,
            effector_profiles=tuple(variants),
            effector_variants=variants,
        )

    def _load_robot(self, name: str, data: dict[str, Any]) -> RobotConfig:
        legacy_remote = str(data.get("legacy_remote_directory", data["remote_directory"]))
        reset_script = str(
            data.get(
                "reset_script",
                f"{data['remote_directory']}/robot_client/reset_all_joints.sh",
            )
        )
        manifest = self._resolve(str(data.get("manifest", "robot_manifest.txt")))
        ssh_port = int(data.get("ssh_port", 22))
        if not 1 <= ssh_port <= 65535:
            raise DeployError(f"Invalid SSH port for robot {name}: {ssh_port}")
        effectors_mode = str(data.get("effectors_mode", "required"))
        if effectors_mode not in {"required", "absent_fixed"}:
            raise DeployError(f"Invalid effectors_mode for robot {name}: {effectors_mode}")
        effector_profile = canonical_effector_profile(
            data.get("effector_profile", "omnipicker_legacy")
        )
        raw_profiles = data.get("effector_profiles", [effector_profile])
        if not isinstance(raw_profiles, list | tuple) or not raw_profiles:
            raise DeployError(f"effector_profiles for robot {name} must be a non-empty list")
        effector_profiles = tuple(canonical_effector_profile(value) for value in raw_profiles)
        if effector_profile not in effector_profiles:
            raise DeployError(f"Default effector_profile for robot {name} must be included in effector_profiles")
        unknown_profiles = [value for value in effector_profiles if value not in self.effector_profiles]
        if unknown_profiles:
            raise DeployError(f"Unknown effector_profiles for robot {name}: {unknown_profiles}")
        profile = self.effector_profiles[effector_profile]
        raw_fixed_effectors = data.get("fixed_effector_state", profile.hardware_state_range[:1] * 2)
        if not isinstance(raw_fixed_effectors, list | tuple) or len(raw_fixed_effectors) != 2:
            raise DeployError(f"fixed_effector_state for robot {name} must contain two values")
        fixed_effector_state = tuple(float(value) for value in raw_fixed_effectors)
        lower, upper = profile.hardware_state_range
        if not all(math.isfinite(value) and lower - 1e-7 <= value <= upper + 1e-7 for value in fixed_effector_state):
            raise DeployError(f"fixed_effector_state for robot {name} must be finite and within [{lower}, {upper}]")
        return RobotConfig(
            name=name,
            source_client=self._resolve(data["source_client"]),
            bundle_directory=self._resolve(data["bundle_directory"]),
            manifest=manifest,
            source_check=str(data.get("source_check", "g2a")),
            ssh_target=str(data["ssh_target"]),
            ssh_port=ssh_port,
            ssh_key=Path(data["ssh_key"]).expanduser().resolve(),
            host_key_alias=str(data["host_key_alias"]),
            remote_directory=self._validate_remote_directory(str(data["remote_directory"])),
            legacy_remote_directory=self._validate_remote_directory(legacy_remote),
            reset_script=self._validate_remote_reset_script(reset_script),
            effectors_mode=effectors_mode,
            fixed_effector_state=(fixed_effector_state[0], fixed_effector_state[1]),
            policy_host=str(data.get("policy_host", self.advertise_host)),
            policy_source_host=str(data.get("policy_source_host", "")).strip(),
            expected_aid=str(data.get("expected_aid", "")).strip(),
            effector_profile=effector_profile,
            effector_profiles=effector_profiles,
        )

    @staticmethod
    def _validate_remote_directory(value: str) -> str:
        path = Path(value)
        safe_prefix = (
            path.is_absolute()
            and len(path.parts) >= 5
            and path.parts[1] == "home"
            and bool(path.parts[2])
            and path.parts[3] == "deploy"
        )
        if not safe_prefix:
            raise DeployError(f"Robot destination must be a specific /home/<user>/deploy/<name> path: {value}")
        if any(part in ("", ".", "..") for part in path.parts[1:]):
            raise DeployError(f"Unsafe robot destination: {value}")
        return str(path)

    @classmethod
    def _validate_remote_reset_script(cls, value: str) -> str:
        """Accept only the fixed reset entry point inside a robot deployment."""

        path = Path(value)
        if path.name != "reset_all_joints.sh" or path.parent.name != "robot_client":
            raise DeployError("Robot reset_script must end with robot_client/reset_all_joints.sh")
        cls._validate_remote_directory(str(path.parent.parent))
        return str(path)

    def select_models(self, name: str) -> tuple[ModelConfig, ...]:
        if name == MODEL_ALL:
            return tuple(self.models.values())
        try:
            return (self.models[name],)
        except KeyError as error:
            raise DeployError(f"Unknown model: {name}") from error

    def select_robot(self, name: str | None) -> RobotConfig:
        if name is None:
            return self.robot
        try:
            return self.robots[name]
        except KeyError as error:
            raise DeployError(f"Unknown robot: {name}; configured={tuple(self.robots)}") from error

    def robot_network_endpoints(self, name: str) -> tuple[dict[str, Any], ...]:
        """Return ordered, credential-bound network endpoints for one robot."""

        robot = self.select_robot(name)
        raw = self.raw.get("robots", {}).get(name, {})
        primary_host = str(raw.get("control_host", "")).strip()
        primary = {
            "host": primary_host,
            "ssh_port": int(raw.get("ssh_port", robot.ssh_port)),
            "label": str(raw.get("control_host_label", "主地址")),
            "transport": str(raw.get("control_host_transport", "wired")),
            "policy_host": str(raw.get("policy_host", robot.policy_host)),
            "policy_source_host": str(raw.get("policy_source_host", robot.policy_source_host)).strip(),
            "priority": 1,
        }
        endpoints = [primary] if primary_host else []
        alternatives = raw.get("alternate_endpoints", ())
        if not isinstance(alternatives, list | tuple):
            raise DeployError(f"alternate_endpoints for robot {name} must be a list")
        for index, value in enumerate(alternatives, start=2):
            if not isinstance(value, dict):
                raise DeployError(f"alternate endpoint {index} for robot {name} must be a table")
            host = str(value.get("host", "")).strip()
            if not host:
                raise DeployError(f"alternate endpoint {index} for robot {name} has no host")
            endpoints.append(
                {
                    "host": host,
                    "ssh_port": int(value.get("ssh_port", robot.ssh_port)),
                    "label": str(value.get("label", f"备用地址 {index - 1}")),
                    "transport": str(value.get("transport", "alternate")),
                    "policy_host": str(value.get("policy_host", robot.policy_host)),
                    "policy_source_host": str(value.get("policy_source_host", robot.policy_source_host)).strip(),
                    "priority": index,
                }
            )
        if len({(item["host"], item["ssh_port"]) for item in endpoints}) != len(endpoints):
            raise DeployError(f"network endpoints for robot {name} must be unique")
        return tuple(endpoints)

    def select_robot_endpoint(self, name: str, host: str, port: int) -> RobotConfig:
        """Bind a configured robot credential to one declared network endpoint."""

        robot = self.select_robot(name)
        endpoints = self.robot_network_endpoints(name)
        if not endpoints:
            return robot
        endpoint = next(
            (item for item in self.robot_network_endpoints(name) if item["host"] == host and item["ssh_port"] == port),
            None,
        )
        if endpoint is None:
            raise DeployError(
                "The selected robot endpoint is not mapped to a deployment credential: "
                f"robot={name}, selected={host}:{port}"
            )
        user, separator, _configured_host = robot.ssh_target.rpartition("@")
        if not separator or not user:
            raise DeployError(f"Robot {name} ssh_target must include user@host")
        raw = self.raw.get("robots", {}).get(name, {})
        endpoint_raw: dict[str, Any] = raw
        if host != str(raw.get("control_host", "")).strip() or port != int(raw.get("ssh_port", robot.ssh_port)):
            endpoint_raw = next(
                (
                    value
                    for value in raw.get("alternate_endpoints", ())
                    if isinstance(value, dict)
                    and str(value.get("host", "")).strip() == host
                    and int(value.get("ssh_port", robot.ssh_port)) == port
                ),
                {},
            )
        ssh_key_value = endpoint_raw.get("ssh_key")
        ssh_key = Path(str(ssh_key_value)).expanduser().resolve() if ssh_key_value else robot.ssh_key
        return replace(
            robot,
            ssh_target=f"{user}@{host}",
            ssh_port=port,
            policy_host=str(endpoint["policy_host"]),
            policy_source_host=str(endpoint["policy_source_host"]),
            ssh_key=ssh_key,
            host_key_alias=str(endpoint_raw.get("host_key_alias", robot.host_key_alias)),
        )

    def model_context(self, model: ModelConfig) -> dict[str, str | int]:
        return {
            "bind_host": self.bind_host,
            "advertise_host": self.advertise_host,
            "port": model.port,
            "repo": str(model.repo),
            "runtime_dir": str(self.runtime_dir),
        }

    def model_rtc_metadata(self, model, effector_profile=None, model_version=None):
        version = model.version(effector_profile, model_version)
        if model.module != "nexus_deploy.vla.pi05.server":
            return dict(rtc_inference_mode_default="guided", trained_rtc_supported=False,
                        rtc_modes=["guided"])
        args = [argument.format_map(self.model_context(model))
                for argument in (*version.base_args, *version.real_args)]
        return rtc_metadata_for_args(args, model.repo)

    def running_model_rtc_inference_mode(self, model):
        """Read actual managed server arguments; unknown route cannot be reused."""
        if model.module != "nexus_deploy.vla.pi05.server":
            return "guided"
        pgid = self.running_pid(model)
        if pgid is None:
            return None
        for pid, command, _state in self._managed_group_members(model, pgid):
            if model.module in command:
                try:
                    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    arguments = [value.decode("utf-8", "replace") for value in arguments if value]
                    if model.module in arguments:
                        return rtc_inference_mode_for_args(arguments)
                except OSError:
                    pass
                try:
                    return rtc_inference_mode_for_args(shlex.split(command))
                except ValueError:
                    return None
        return None

    def model_command(
        self,
        model: ModelConfig,
        *,
        mock: bool,
        effector_profile: str | None = None,
        model_version: str | None = None,
        rtc_inference_mode: str | None = None,
    ) -> list[str]:
        context = self.model_context(model)
        version = model.version(effector_profile, model_version)
        args = (
            version.base_args
            + (f"--effector-profile={version.effector_profile}",)
            + (version.mock_args if mock else version.real_args)
        )
        formatted = [item.format_map(context) for item in args]
        if rtc_inference_mode is not None and model.module == "nexus_deploy.vla.pi05.server":
            if not mock:
                validate_rtc_inference_mode(
                    rtc_inference_mode, self.model_rtc_metadata(model, effector_profile, model_version)
                )
            formatted = with_rtc_inference_mode(formatted, rtc_inference_mode)
        if model.python_executable is not None:
            return [str(model.python_executable), "-m", model.module, *formatted]
        return [
            str(self.conda),
            "run",
            "--no-capture-output",
            "-n",
            model.conda_env,
            "python",
            "-m",
            model.module,
            *formatted,
        ]

    def model_environment(self, model: ModelConfig, *, cuda_visible_devices: str | None = None) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = (
            model.cuda_visible_devices if cuda_visible_devices is None else str(cuda_visible_devices)
        )
        roots = [str(model.repo), *model.python_paths]
        environment["PYTHONPATH"] = os.pathsep.join(roots)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        return environment

    def pid_file(self, model: ModelConfig) -> Path:
        return self.runtime_dir / f"{model.name}.pid"

    def console_log(self, model: ModelConfig) -> Path:
        return self.runtime_dir / f"{model.name}.console.log"

    def model_selection_file(self, model: ModelConfig) -> Path:
        return self.runtime_dir / f"{model.name}.selection.json"

    def read_pid(self, model: ModelConfig) -> int | None:
        path = self.pid_file(model)
        if not path.is_file():
            return None
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        return pid if pid > 1 else None

    @staticmethod
    def process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def process_command(pid: int) -> str:
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            return ""

    @classmethod
    def process_group_members(cls, pgid: int) -> list[tuple[int, str, str]]:
        """Return live and zombie processes in a managed process group.

        The pid file stores the process-group id created by ``start_new_session``.
        Looking at the whole group matters because ``conda run`` may exit while
        its Python child is still loading a checkpoint.
        """

        members: list[tuple[int, str, str]] = []
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                pid = int(stat_path.parent.name)
                suffix = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
                state = suffix[0]
                process_group = int(suffix[2])
            except (IndexError, OSError, ValueError):
                continue
            if process_group == pgid:
                members.append((pid, cls.process_command(pid), state))
        return members

    @classmethod
    def process_group_exists(cls, pgid: int) -> bool:
        return any(state != "Z" for _, _, state in cls.process_group_members(pgid))

    def _managed_group_members(self, model: ModelConfig, pgid: int) -> list[tuple[int, str, str]]:
        members = [item for item in self.process_group_members(pgid) if item[2] != "Z"]
        if members and not any(model.module in command for _, command, _ in members):
            commands = "; ".join(f"{pid}: {command!r}" for pid, command, _ in members)
            raise DeployError(
                f"Refusing to manage process group {pgid}: no command contains {model.module!r}: {commands}"
            )
        return members

    def running_pid(self, model: ModelConfig) -> int | None:
        pgid = self.read_pid(model)
        if pgid is None:
            return None
        members = self._managed_group_members(model, pgid)
        return pgid if members else None

    def running_model_effector_profile(self, model: ModelConfig) -> str | None:
        """Return the gripper profile declared by the managed model command."""

        pgid = self.running_pid(model)
        if pgid is None:
            return None
        prefix = "--effector-profile="
        for _pid, command, _state in self._managed_group_members(model, pgid):
            for argument in command.split():
                if argument.startswith(prefix):
                    value = argument.removeprefix(prefix).strip()
                    return value or None
        return None

    def running_model_version(self, model: ModelConfig) -> str | None:
        """Identify a managed model version from its allow-listed asset arguments."""

        pgid = self.running_pid(model)
        if pgid is None:
            return None
        commands = "\n".join(command for _pid, command, _state in self._managed_group_members(model, pgid))
        matches: set[str] = set()
        for variant in model.effector_variants.values():
            for version_id, version in variant.versions.items():
                selectors = tuple(
                    argument
                    for argument in version.real_args
                    if argument.startswith(("--checkpoint=", "--model-path=", "--robot-name="))
                )
                if selectors and all(argument in commands for argument in selectors):
                    matches.add(version_id)
        if len(matches) == 1:
            return next(iter(matches))
        try:
            selection = json.loads(
                self.model_selection_file(model).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(selection, dict) or selection.get("pgid") != pgid:
            return None
        version_id = str(selection.get("model_version", ""))
        profile_name = canonical_effector_profile(selection.get("effector_profile", ""))
        if profile_name not in model.effector_profiles:
            return None
        return version_id if version_id in model.variant(profile_name).versions else None

    def running_model_mock(self, model: ModelConfig) -> bool | None:
        """Read the managed process selection, never infer service type from health alone."""
        pgid = self.running_pid(model)
        if pgid is None:
            return None
        try:
            selection = json.loads(self.model_selection_file(model).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(selection, dict) or selection.get("pgid") != pgid:
            return None
        return selection.get("mock") if isinstance(selection.get("mock"), bool) else None

    def health_url(self, model: ModelConfig) -> str:
        return f"http://127.0.0.1:{model.port}/healthz"

    def health(self, model: ModelConfig, timeout_s: float = 1.0) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(self.health_url(model), timeout=timeout_s) as response:
                payload = response.read()
        except (OSError, urllib.error.URLError):
            return None
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return {"raw": payload.decode(errors="replace")}
        return decoded if isinstance(decoded, dict) else {"value": decoded}

    def wait_healthy(
        self,
        model: ModelConfig,
        pgid: int,
        timeout_s: float,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        deadline = time.monotonic() + timeout_s
        next_progress = started_at
        while time.monotonic() < deadline:
            if not self.process_group_exists(pgid):
                tail = self.tail_console(model)
                raise DeployError(f"{model.name} exited before becoming healthy\n{tail}")
            health = self.health(model)
            if health is not None:
                return health
            now = time.monotonic()
            if now >= next_progress:
                elapsed = now - started_at
                last_line = next(
                    (
                        line.strip()
                        for line in reversed(self.tail_console(model, lines=8).splitlines())
                        if line.strip()
                    ),
                    "waiting for model output",
                )
                message = f"{model.name} is loading ({elapsed:.0f}/{timeout_s:.0f}s): {last_line[-240:]}"
                print(message, flush=True)
                if progress_callback is not None:
                    progress_callback(message)
                next_progress = now + 15.0
            time.sleep(0.25)
        tail = self.tail_console(model, lines=20)
        raise DeployError(
            f"Timed out waiting {timeout_s:.1f}s for {model.name} health at {self.health_url(model)}\n{tail}"
        )

    def tail_console(self, model: ModelConfig, lines: int = 30) -> str:
        path = self.console_log(model)
        if not path.is_file():
            return f"No console log: {path}"
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def _terminate_model_group(
        self,
        model: ModelConfig,
        pgid: int,
        *,
        term_timeout_s: float = 10.0,
        kill_timeout_s: float = 5.0,
    ) -> None:
        members = self._managed_group_members(model, pgid)
        if not members:
            self.pid_file(model).unlink(missing_ok=True)
            self.model_selection_file(model).unlink(missing_ok=True)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + term_timeout_s
        while time.monotonic() < deadline and self.process_group_exists(pgid):
            time.sleep(0.1)
        if self.process_group_exists(pgid):
            # The target was verified above, so escalation cannot hit an
            # unrelated process group even when a checkpoint load ignores TERM.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + kill_timeout_s
            while time.monotonic() < deadline and self.process_group_exists(pgid):
                time.sleep(0.1)
        if self.process_group_exists(pgid):
            raise DeployError(f"{model.name} process group {pgid} could not be stopped")
        self.pid_file(model).unlink(missing_ok=True)
        self.model_selection_file(model).unlink(missing_ok=True)

    def start_model(
        self,
        model: ModelConfig,
        *,
        mock: bool,
        wait_timeout_s: float | None,
        progress_callback: Callable[[str], None] | None = None,
        cuda_visible_devices: str | None = None,
        effector_profile: str | None = None,
        model_version: str | None = None,
        rtc_inference_mode: str | None = None,
    ) -> int:
        running = self.running_pid(model)
        if running is not None:
            raise DeployError(f"{model.name} is already running as PID {running}")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file(model).unlink(missing_ok=True)
        self.model_selection_file(model).unlink(missing_ok=True)
        selected_version = model.version(effector_profile, model_version)
        selected_effector_profile = selected_version.effector_profile
        command = self.model_command(
            model,
            mock=mock,
            effector_profile=selected_effector_profile,
            model_version=selected_version.id,
            rtc_inference_mode=rtc_inference_mode,
        )
        log_stream = self.console_log(model).open("ab", buffering=0)
        try:
            marker = (
                f"\n=== Nexus start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"model={model.name} version={selected_version.id} "
                f"gripper={selected_effector_profile} mock={mock} ===\n"
            )
            log_stream.write(marker.encode())
            process = subprocess.Popen(
                command,
                cwd=model.repo,
                env=self.model_environment(model, cuda_visible_devices=cuda_visible_devices),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_stream.close()
        timeout = model.startup_timeout_s if wait_timeout_s is None else wait_timeout_s
        try:
            self.pid_file(model).write_text(f"{process.pid}\n", encoding="utf-8")
            selection_path = self.model_selection_file(model)
            temporary_selection = selection_path.with_suffix(".tmp")
            temporary_selection.write_text(
                json.dumps(
                    {
                        "pgid": process.pid,
                        "model_id": model.name,
                        "effector_profile": selected_effector_profile,
                        "model_version": selected_version.id,
                        "rtc_inference_mode": rtc_inference_mode_for_args(command),
                        "mock": mock,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_selection.replace(selection_path)
            health = self.wait_healthy(
                model,
                process.pid,
                timeout,
                progress_callback=progress_callback,
            )
        except Exception:
            self._terminate_model_group(model, process.pid)
            raise
        print(f"started {model.name}: pid={process.pid} port={model.port} mock={mock} health={health}")
        return process.pid

    def stop_model(self, model: ModelConfig, timeout_s: float = 20.0) -> None:
        pgid = self.running_pid(model)
        if pgid is None:
            self.pid_file(model).unlink(missing_ok=True)
            self.model_selection_file(model).unlink(missing_ok=True)
            print(f"{model.name}: not running")
            return
        self._terminate_model_group(model, pgid, term_timeout_s=timeout_s)
        print(f"stopped {model.name}: process_group={pgid}")

    def ssh_base(self, robot: RobotConfig | None = None) -> list[str]:
        robot = self.robot if robot is None else robot
        return [
            "ssh",
            "-o",
            f"HostKeyAlias={robot.host_key_alias}",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=5",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=60",
            "-o", "ControlPath=/tmp/nexus-platform-ssh-%C",
            "-p",
            str(robot.ssh_port),
            "-i",
            str(robot.ssh_key),
            robot.ssh_target,
        ]

    def rsync_shell(self, robot: RobotConfig | None = None) -> str:
        robot = self.robot if robot is None else robot
        values = [
            "ssh",
            "-o",
            f"HostKeyAlias={robot.host_key_alias}",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=5",
            "-p",
            str(robot.ssh_port),
            "-i",
            str(robot.ssh_key),
        ]
        return shlex.join(values)

    def run_ssh(
        self,
        remote_command: str,
        *,
        robot: RobotConfig | None = None,
        capture: bool = False,
        timeout_s: float = 15.0,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [*self.ssh_base(robot), remote_command],
                check=True,
                text=True,
                capture_output=capture,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            raise DeployError(f"Robot SSH operation timed out after {timeout_s:g}s") from error

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if (
            not session_id
            or len(session_id) > 64
            or any(character not in "0123456789abcdef" for character in session_id)
        ):
            raise DeployError("Invalid deployment session id")
        return session_id

    @staticmethod
    def _validate_robot_session_root(value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or len(path.parts) != 5
            or path.parts[1] != "home"
            or not path.parts[2]
            or path.parts[3:] != (".cache", "nexus-platform-sessions")
        ):
            raise DeployError(
                "Robot session directory must be /home/<user>/.cache/nexus-platform-sessions"
            )
        return str(path)

    def robot_session_directory(self, session_id: str) -> str:
        return f"{self.robot_session_remote_root}/{self._validate_session_id(session_id)}"

    def _robot_session_prelaunch_cleanup(self, session_id: str) -> str:
        """Return a bounded remote cleanup command for archived Nexus sessions.

        Cleanup has to run *before* the new session directory is created.  If
        the filesystem is already full, waiting until the next successful
        archive is too late.  Only generations carrying the durable archive
        marker are eligible; unarchived or running sessions are never removed.
        """

        session_id = self._validate_session_id(session_id)
        script = r'''import re, shutil, sys
from pathlib import Path
root, keep, current = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if not (root.is_absolute() and root.name == "nexus-platform-sessions" and root.parent.name == ".cache"):
    raise SystemExit(42)
uuid_re = re.compile(r"[0-9a-f]{1,64}")
candidates = []
for path in root.iterdir() if root.is_dir() else ():
    if path.is_dir() and uuid_re.fullmatch(path.name):
        if path.name != current and (path / ".archived-to-server").is_file():
            candidates.append(path)
        continue
    if not (path.is_dir() and path.name.endswith(".history")):
        continue
    owner = path.name[:-8]
    if not uuid_re.fullmatch(owner) or owner == current:
        continue
    for generation in path.iterdir():
        if generation.is_dir() and (generation / ".archived-to-server").is_file():
            candidates.append(generation)
candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
for path in candidates[keep:]:
    pid = 0
    try: pid = int((path / "client.pid").read_text().strip())
    except Exception: pass
    if pid > 1 and Path(f"/proc/{pid}").exists():
        continue
    shutil.rmtree(path)
for path in root.glob("*.history"):
    try: path.rmdir()
    except OSError: pass
'''
        return (
            f"python3 -c {shlex.quote(script)} "
            f"{shlex.quote(self.robot_session_remote_root)} "
            f"{self.robot_session_remote_retention} {shlex.quote(session_id)}; "
        )

    def archive_robot_session(
        self,
        robot: RobotConfig,
        session_id: str,
        *,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """Copy durable numerical session evidence from G2 to the server.

        Per-request JPEGs are deliberately excluded: they are large and are
        already used only for the live UI.  Numerical chunks, 100 Hz control
        rows, safety logs, timing, console output and JSON metadata are kept.
        A session is marked remotely only after the local copy is complete;
        retention removes only older sessions carrying that marker.
        """

        session_id = self._validate_session_id(session_id)
        remote = self.robot_session_directory(session_id)
        existing = sorted(self.robot_session_archive_root.glob(f"*/{session_id}/archive.json"))
        if existing:
            try:
                value = json.loads(existing[-1].read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                value = {}
            return {**value, "status": "ready", "path": str(existing[-1].parent)}
        try:
            self.run_ssh(
                f"test -d {shlex.quote(remote)}",
                robot=robot,
                timeout_s=min(15.0, timeout_s),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DeployError(f"Robot session directory is unavailable: {remote}") from error

        day = time.strftime("%Y%m%d")
        parent = self.robot_session_archive_root / day
        destination = parent / session_id
        temporary = parent / f".{session_id}.tmp-{uuid.uuid4().hex[:8]}"
        parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(mode=0o750)
        command = [
            "rsync",
            "--archive",
            "--safe-links",
            "--exclude=/telemetry.token",
            "--exclude=/client.pid",
            "--exclude=*.jpg",
            "--exclude=*.jpeg",
            "--exclude=*.png",
            "-e",
            self.rsync_shell(robot),
            "--",
            f"{robot.ssh_target}:{remote}/",
            f"{temporary}/",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            entries = []
            for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
                relative = path.relative_to(temporary).as_posix()
                entries.append(
                    {
                        "path": relative,
                        "size": path.stat().st_size,
                        "sha256": self._sha256(path),
                    }
                )
            if not entries:
                raise DeployError("Robot session archive contained no diagnostic files")
            manifest = {
                "schema": "nexus.robot-session-archive.v1",
                "status": "ready",
                "session_id": session_id,
                "robot": robot.name,
                "remote_path": remote,
                "path": str(destination),
                "archived_at": time.time(),
                "images_included": False,
                "file_count": len(entries),
                "total_bytes": sum(item["size"] for item in entries),
                "files": entries,
            }
            (temporary / "archive.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        marker_command = f"touch {shlex.quote(remote + '/.archived-to-server')}"
        remote_retention_applied = False
        try:
            self.run_ssh(marker_command, robot=robot, timeout_s=15.0)
            prune_script = r'''import re, shutil, sys
from pathlib import Path
root, keep = Path(sys.argv[1]), int(sys.argv[2])
if not (root.is_absolute() and root.name == "nexus-platform-sessions" and root.parent.name == ".cache"):
    raise SystemExit(42)
archived = sorted(
    (p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[0-9a-f]{1,64}", p.name)
     and (p / ".archived-to-server").is_file()),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
for path in archived[keep:]:
    pid = 0
    try: pid = int((path / "client.pid").read_text().strip())
    except Exception: pass
    if pid > 1 and Path(f"/proc/{pid}").exists():
        continue
    shutil.rmtree(path)
'''
            self.run_ssh(
                f"python3 -c {shlex.quote(prune_script)} "
                f"{shlex.quote(self.robot_session_remote_root)} "
                f"{self.robot_session_remote_retention}",
                robot=robot,
                timeout_s=30.0,
            )
            remote_retention_applied = True
        except (OSError, subprocess.SubprocessError):
            # The server copy is already durable. A marker/prune failure must
            # not turn a safe robot stop into a failed stop operation.
            pass
        manifest["remote_retention_applied"] = remote_retention_applied
        (destination / "archive.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def workspace_safety_profile(self, robot: RobotConfig, profile_id: str) -> tuple[dict[str, Any], Path, str, bool]:
        """Resolve an allow-listed static or generated safety profile."""

        if not profile_id or Path(profile_id).name != profile_id:
            raise DeployError("Invalid workspace safety profile")
        try:
            manifest_entries = {
                line.strip() for line in robot.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
            }
        except OSError as error:
            raise DeployError("Workspace safety manifest is unavailable") from error
        static_path = robot.source_client / profile_id
        if profile_id in manifest_entries and static_path.is_file():
            path = static_path
            generated = False
        else:
            path = self.generated_safety_profiles_root / robot.name / profile_id
            generated = True
        try:
            resolved = path.resolve(strict=True)
            profile = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise DeployError("Workspace safety profile is unavailable") from error
        if generated:
            expected_parent = (self.generated_safety_profiles_root / robot.name).resolve()
            if resolved.parent != expected_parent:
                raise DeployError("Generated workspace safety profile escaped its directory")
            metadata = profile.get("generated_profile")
            surface_calibration = profile.get("surface_calibration")
            if (
                not isinstance(metadata, dict)
                or not isinstance(surface_calibration, dict)
                or metadata.get("generator") != GENERATED_SAFETY_PROFILE_ID
                or metadata.get("robot_id") != robot.name
                or metadata.get("measurement_id") != surface_calibration.get("measurement_id")
            ):
                raise DeployError("Generated workspace safety profile provenance is invalid")
            if robot.expected_aid and surface_calibration.get("robot_aid") != robot.expected_aid:
                raise DeployError("Generated workspace safety profile robot AID is missing or mismatched")
            template_name = str(metadata.get("source_template", ""))
            if Path(template_name).name != template_name or template_name not in manifest_entries:
                raise DeployError("Generated workspace safety profile template is not allow-listed")
            template_path = robot.source_client / template_name
            try:
                template = json.loads(template_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as error:
                raise DeployError("Generated workspace safety template is unavailable") from error
            if metadata.get("source_template_sha256") != self._sha256(template_path) or profile.get(
                "robot_model", {}
            ).get("sha256") != template.get("robot_model", {}).get("sha256"):
                raise DeployError("Generated workspace safety profile geometry has changed")
            measurement_id = str(metadata.get("measurement_id", ""))
            measurement_sha = str(metadata.get("measurement_sha256", ""))
            measurements_root = self.alignment_surfaces_root
            measurement: dict[str, Any] | None = None
            measurement_path: Path | None = None
            candidates = sorted(measurements_root.glob("*/versions/*.json"))
            candidates.extend(sorted(measurements_root.glob("*/active.json")))
            for candidate in candidates:
                try:
                    candidate_resolved = candidate.resolve(strict=True)
                    candidate_value = json.loads(candidate_resolved.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    continue
                if (
                    isinstance(candidate_value, dict)
                    and candidate_value.get("measurement_id") == measurement_id
                    and self._sha256(candidate_resolved) == measurement_sha
                ):
                    measurement = candidate_value
                    measurement_path = candidate_resolved
                    break
            if measurement is None or measurement_path is None:
                raise DeployError("Generated workspace safety profile source measurement is unavailable")
            repository = AlignmentSafetyProfileRepository(self.generated_safety_profiles_root)
            allowed_hosts = {str(endpoint["host"]) for endpoint in self.robot_network_endpoints(robot.name)}
            try:
                verified_path = repository.generate(
                    measurement=measurement,
                    measurement_path=measurement_path,
                    robot_id=robot.name,
                    template_path=template_path,
                    expected_robot_aid=robot.expected_aid,
                    allowed_robot_hosts=allowed_hosts,
                ).resolve(strict=True)
            except (OSError, TypeError, ValueError, SafetyProfileGenerationError) as error:
                raise DeployError(
                    "Generated workspace safety profile no longer matches its immutable source"
                ) from error
            if verified_path != resolved:
                raise DeployError("Generated workspace safety profile path does not match its immutable source")
        remote_path = f"{robot.remote_directory}/robot_client/{profile_id}"
        return profile, resolved, remote_path, generated

    def deploy_generated_safety_profile(self, robot: RobotConfig, profile_id: str, session_id: str) -> None:
        profile, path, remote_path, generated = self.workspace_safety_profile(robot, profile_id)
        del profile
        if not generated:
            return
        payload = path.read_bytes()
        if len(payload) > 512 * 1024:
            raise DeployError("Generated workspace safety profile is too large")
        encoded = base64.b64encode(payload).decode("ascii")
        temporary = f"{remote_path}.tmp-{self._validate_session_id(session_id)}"
        command = (
            "set -eu; umask 077; "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(temporary)}; "
            f"chmod 0600 {shlex.quote(temporary)}; "
            f"mv {shlex.quote(temporary)} {shlex.quote(remote_path)}"
        )
        self.run_ssh(command, robot=robot, timeout_s=15.0)

    def robot_session_command(
        self,
        robot: RobotConfig,
        model: ModelConfig,
        session_id: str,
        config: dict[str, Any],
    ) -> tuple[list[str], str]:
        if config.get("rtc_prefix_mode") == "command" and config.get("hardware_mode") == "execute":
            raise DeployError("Command-prefix inference is currently read-only only")
        if config["hardware_mode"] not in {"dry-run", "read-only", "execute"}:
            raise DeployError("Unsupported robot hardware mode")
        if robot.effectors_mode == "absent_fixed" and config["hardware_mode"] == "execute":
            raise DeployError("This robot has no effectors; fixed synthetic effector state is observation-only")
        effector_profile_name = str(config.get("effector_profile", robot.effector_profile))
        if effector_profile_name not in robot.effector_profiles:
            raise DeployError(
                "Robot does not allow the selected effector profile: "
                f"selected={effector_profile_name}, robot={robot.effector_profiles}"
            )
        if effector_profile_name not in model.effector_profiles:
            raise DeployError(
                "Model has no variant for the selected effector profile: "
                f"selected={effector_profile_name}, model={model.effector_profiles}"
            )
        effector_profile = self.effector_profiles[effector_profile_name]
        effector_control_mode = str(
            config.get("effector_control_mode", "software-trajectory")
        )
        effector_command_mode = (
            "endpoint-latch"
            if effector_profile_name == "dahuan" and effector_control_mode == "device-position"
            else "continuous"
        )
        configured_endpoints = {
            (str(item["host"]), int(item["ssh_port"])) for item in self.robot_network_endpoints(robot.name)
        }
        selected_endpoint = (str(config["robot_host"]), int(config["robot_ssh_port"]))
        if configured_endpoints and selected_endpoint not in configured_endpoints:
            raise DeployError(
                "The selected robot IP is not yet mapped to a deployment credential; "
                f"configured={sorted(configured_endpoints)}, selected={selected_endpoint}"
            )
        session_dir = self.robot_session_directory(session_id)
        hitl_active = bool(config.get("human_in_loop") and config["hardware_mode"] == "execute")
        if hitl_active:
            if not config["safety"].get("workspace_safety_enabled"):
                raise DeployError("Human-in-loop rollback requires enforced workspace safety")
            if not math.isclose(float(config["servo_fps"]), 100.0, rel_tol=0.0, abs_tol=1e-9):
                raise DeployError("Human-in-loop rollback requires a 100 Hz GDK servo")
        command = [
            f"{robot.remote_directory}/robot_client/run.sh",
            f"--host={robot.policy_host}",
            f"--port={model.port}",
            f"--task={config['task']}",
            f"--inference-mode={config['inference_mode']}",
            f"--{config['hardware_mode']}",
            f"--effectors-mode={robot.effectors_mode}",
            f"--effector-profile={effector_profile_name}",
            f"--fixed-left-effector-state={robot.fixed_effector_state[0]}",
            f"--fixed-right-effector-state={robot.fixed_effector_state[1]}",
            f"--max-gripper-relative-target={effector_profile.default_relative_target_warning}",
            f"--duration-s={config['duration_s']}",
            f"--execution-fps={config['execution_fps']}",
            f"--servo-fps={config['servo_fps']}",
            "--safety-check-fps=50",
            "--async-safety",
            f"--prediction-horizon={config['prediction_horizon']}",
            f"--quality-horizon={config['quality_horizon']}",
            f"--min-replan-after-steps={config['min_replan_after_steps']}",
            f"--request-lead-margin-steps={config['request_lead_margin_steps']}",
            f"--min-rtc-blend-steps={config['min_rtc_blend_steps']}",
            f"--latency-window-size={config['latency_window_size']}",
            f"--warmup-requests={config['warmup_requests']}",
            f"--execution-preflight-requests={config['execution_preflight_requests']}",
            f"--watchdog-timeout-s={config['watchdog_timeout_s']}",
            f"--request-timeout-s={config['request_timeout_s']}",
            f"--action-low-pass-cutoff-hz={config['action_low_pass_cutoff_hz']}",
            f"--effector-action-low-pass-cutoff-hz={config['effector_action_low_pass_cutoff_hz']}",
            # The two chunk smoothers are mutually exclusive.  Both preserve
            # the policy clock. Seven-point Savitzky-Golay is the default;
            # adaptive fixed-time remains an explicit experimental option.
            f"--chunk-smoothing-mode={config.get('chunk_smoothing_mode', 'savgol')}",
            f"--action-interpolation={config.get('action_interpolation', 'linear')}",
            "--chunk-smoothing-window=7",
            "--chunk-smoothing-max-deviation=0.005",
            "--adaptive-smoothing-acceleration-weight=20.0",
            "--adaptive-smoothing-jerk-weight=5.0",
            "--adaptive-smoothing-velocity-reference=0.25",
            # Keep the policy time axis unchanged.  C2 reconstruction is
            # intentionally not enabled in production because its velocity /
            # acceleration repair can stretch individual arm segments while
            # Dahuan device-position targets continue on the policy clock.
            # The operator selects uniform or a data-derived per-joint shape;
            # the two UI scalar values always remain hard caps.  The platform
            # default reads the active safety URDF's per-joint arm velocities
            # and applies the explicitly configured 8 rad/s^2 acceleration;
            # the velocity-only experiment is an explicit opt-in.
            f"--arm-limit-profile={config['arm_limit_profile']}",
            f"--max-arm-velocity={config['max_arm_velocity']}",
            f"--max-arm-acceleration={config['max_arm_acceleration']}",
            f"--max-effector-velocity={config['max_effector_velocity']}",
            f"--effector-control-mode={effector_control_mode}",
            f"--effector-command-mode={effector_command_mode}",
            "--effector-closed-max=0.0075",
            "--effector-open-min=0.0175",
            f"--max-effector-acceleration={config['max_effector_acceleration']}",
            f"--max-head-waist-velocity={config['max_head_waist_velocity']}",
            f"--max-head-waist-acceleration={config['max_head_waist_acceleration']}",
            f"--max-arm-tracking-error={config['max_arm_tracking_error']}",
            f"--max-effector-tracking-error={config['max_effector_tracking_error']}",
            f"--max-head-waist-tracking-error={config['max_head_waist_tracking_error']}",
            f"--log-file={session_dir}/client.log",
            f"--timing-log-file={session_dir}/timing.jsonl",
            f"--control-file={session_dir}/control.json",
            f"--control-status-file={session_dir}/control-status.json",
            f"--platform-frame-dir={session_dir}/inference-frames",
        ]
        if hitl_active:
            command[1:1] = ["--human-in-loop"]
            command.extend(
                [
                    "--sidecar-root=/data/logs/flowpro_sidecar",
                    "--vendor-record-root=/data/record",
                    f"--flowpro-feedback-root=/data/logs/flowpro_operator_feedback/{session_id}",
                    f"--robot-id={robot.name}",
                    f"--task-id={session_id}",
                    "--enable-rollback",
                    "--enable-vendor-ee-handoff",
                ]
            )
            if config.get("_tablet_record_name"):
                command.append(f"--initial-vendor-record-name={config['_tablet_record_name']}")
        if robot.policy_source_host:
            command.append(f"--source-address={robot.policy_source_host}")
        if config.get("arm_tracking_mode", "legacy") != "legacy":
            command.extend([f"--arm-tracking-mode={config['arm_tracking_mode']}",
                            f"--arm-tracking-omega={config.get('arm_tracking_omega', 12.0)}"])
        if config.get("rtc_prefix_mode", "model") != "model":
            command.append(f"--rtc-prefix-mode={config['rtc_prefix_mode']}")
        if config["hardware_mode"] == "read-only":
            command.append(f"--action-trace-dir={session_dir}/trace")
        if self.robot_pause_on_collision and config["hardware_mode"] == "execute":
            if not config["safety"].get("workspace_safety_enabled"):
                raise DeployError("Collision pause requires enforced workspace safety")
            command.append("--pause-on-collision")
        if self.robot_rtc_state_aligned_anchor and config["inference_mode"] == "rtc":
            command.append("--rtc-state-aligned-anchor")
        if self.robot_rt_cadence and config["hardware_mode"] == "execute":
            if not config["safety"].get("workspace_safety_enabled"):
                raise DeployError("Realtime cadence requires enforced workspace safety")
            command.append("--rt-cadence")
        if config["safety"].get("hold_effectors"):
            command.append("--hold-effectors")
        if not config["safety"].get("low_pass_filter_enabled", True):
            command.append("--no-action-low-pass")
        if not config["safety"].get("effector_low_pass_filter_enabled", True):
            command.append("--no-effector-action-low-pass")
        safety_enabled = bool(config["safety"].get("workspace_safety_enabled"))
        profile_id = str(config["safety"].get("safety_profile_id", ""))
        if profile_id:
            profile, _path, remote_profile_path, _generated = self.workspace_safety_profile(robot, profile_id)
            calibrated = profile.get("calibrated") is True
            read_only_ready = calibrated or profile.get("read_only_ready") is True
            if config["hardware_mode"] == "read-only" and not read_only_ready:
                raise DeployError("Workspace safety profile is not ready for read-only validation")
            if config["hardware_mode"] == "execute" and (
                not calibrated or profile.get("execute_approved", True) is not True
            ):
                raise DeployError("Workspace safety profile is not approved for physical execution")
            command.append(f"--safety-config={remote_profile_path}")
            if safety_enabled:
                # Both observation and physical execution use the same
                # validated, segmented collision queue. Calibration and
                # execution approval checks above remain mandatory.
                command.extend(["--rolling-safety", "--rolling-geometry=legacy"])
            else:
                command.append("--safety-monitor-only")
        elif safety_enabled:
            raise DeployError("Workspace safety requires a calibrated profile")
        telemetry_endpoint = str(config.get("_platform_telemetry_endpoint") or "").strip()
        telemetry_token = str(config.get("_platform_telemetry_token") or "").strip()
        if bool(telemetry_endpoint) != bool(telemetry_token):
            raise DeployError("Platform telemetry endpoint and token must be configured together")
        if telemetry_endpoint:
            if not telemetry_endpoint.startswith(("http://", "https://", "tcp://")):
                raise DeployError("Platform telemetry endpoint must use HTTP(S) or TCP")
            if len(telemetry_endpoint) > 512 or len(telemetry_token) < 24:
                raise DeployError("Invalid platform telemetry endpoint or token")
            command.extend(
                [
                    f"--platform-telemetry-endpoint={telemetry_endpoint}",
                    f"--platform-telemetry-token-file={session_dir}/telemetry.token",
                    "--platform-telemetry-fps=20",
                ]
            )
        presence_endpoint = str(config.get("_platform_presence_frame_endpoint") or "").strip()
        presence_token = str(config.get("_platform_presence_frame_token") or "").strip()
        presence_job_id = str(config.get("_platform_presence_job_id") or "").strip()
        presence_generation_raw = config.get("_platform_presence_generation")
        presence_configured = (
            bool(presence_endpoint),
            bool(presence_token),
            bool(presence_job_id),
            presence_generation_raw is not None,
        )
        if any(presence_configured) and not all(presence_configured):
            raise DeployError(
                "Package-presence frame endpoint, token, job id, and generation must be configured together"
            )
        if presence_endpoint:
            if not presence_endpoint.startswith("tcp://"):
                raise DeployError("Package-presence frame endpoint must use TCP")
            if len(presence_endpoint) > 512 or len(presence_token) < 24:
                raise DeployError("Invalid package-presence frame endpoint or token")
            if not re.fullmatch(r"[0-9a-f]{32}", presence_job_id):
                raise DeployError("Invalid package-presence deployment ID")
            if isinstance(presence_generation_raw, bool):
                raise DeployError("Invalid package-presence client generation")
            try:
                presence_generation = int(presence_generation_raw)
            except (TypeError, ValueError) as error:
                raise DeployError("Invalid package-presence client generation") from error
            if presence_generation < 1:
                raise DeployError("Package-presence client generation must be positive")
            command.extend(
                [
                    f"--platform-presence-frame-endpoint={presence_endpoint}",
                    f"--platform-presence-frame-token-file={session_dir}/presence-frame.token",
                    f"--platform-presence-job-id={presence_job_id}",
                    f"--platform-presence-generation={presence_generation}",
                    "--platform-presence-frame-fps=10",
                ]
            )
        if bool(config.get("_platform_start_paused", False)):
            command.append("--start-paused")
        command.append(f"--stationary-hold-feedback-max-age-ms={self.stationary_hold_feedback_max_age_ms}")
        return command, session_dir

    def start_robot_session(
        self,
        robot: RobotConfig,
        model: ModelConfig,
        session_id: str,
        config: dict[str, Any],
    ) -> int:
        if config["hardware_mode"] in {"read-only", "execute"}:
            self.require_robot_read_only_ready(robot)
        self.ensure_robot_client_idle(robot)
        profile_id = str(config.get("safety", {}).get("safety_profile_id", ""))
        if profile_id:
            self.deploy_generated_safety_profile(robot, profile_id, session_id)
        command, session_dir = self.robot_session_command(robot, model, session_id, config)
        if config["hardware_mode"] in {"read-only", "execute"}:
            self.release_managed_alignment_worker(robot, hold_s=15.0)
        archive_dir = f"{session_dir}.history/{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        prelaunch_cleanup = ""
        if session_dir == f"{self.robot_session_remote_root}/{session_id}":
            prelaunch_cleanup = self._robot_session_prelaunch_cleanup(session_id)
        telemetry_token = str(config.get("_platform_telemetry_token") or "")
        presence_token = str(config.get("_platform_presence_frame_token") or "")
        # The robot client is shared with 5090. Enable the new three-camera
        # wire format only for Thor sessions which own a presence sidecar.
        robot_command = (
            ["env", "NEXUS_PRESENCE_FRAME_MODE=three", *command]
            if presence_token else command
        )
        token_setup = ""
        if telemetry_token:
            token_setup = f"umask 077; printf %s {shlex.quote(telemetry_token)} >" + '"$session_dir/telemetry.token"; '
        if presence_token:
            token_setup += f"umask 077; printf %s {shlex.quote(presence_token)} >" + '"$session_dir/presence-frame.token"; '
        launch = (
            "set -eu; "
            f"session_dir={shlex.quote(session_dir)}; "
            f"archive_dir={shlex.quote(archive_dir)}; "
            f"{prelaunch_cleanup}"
            'if test -e "$session_dir"; then '
            'mkdir -p "${archive_dir%/*}"; test ! -e "$archive_dir"; '
            'mv -- "$session_dir" "$archive_dir"; fi; '
            'mkdir -p "$session_dir"; '
            f"{token_setup}"
            f"cd {shlex.quote(robot.remote_directory)}; "
            f'nohup {shlex.join(robot_command)} </dev/null >"$session_dir/console.log" 2>&1 & '
            'pid=$!; echo "$pid" >"$session_dir/client.pid"; echo "$pid"'
        )
        try:
            result = self.run_ssh(launch, robot=robot, capture=True)
        except subprocess.CalledProcessError as error:
            detail = "\n".join(
                part.strip()
                for part in (str(error.stderr or ""), str(error.stdout or ""))
                if part and part.strip()
            )
            if "No space left on device" in detail:
                raise DeployError(
                    "G2 /data 分区可用空间不足，无法创建机器人会话；"
                    "请先清理已归档的 Nexus 会话或其他日志"
                ) from error
            raise DeployError(f"Robot client launch failed: {detail or error}") from error
        try:
            pid = int(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as error:
            raise DeployError(f"Robot client did not return a PID: {result.stdout!r}") from error
        if pid <= 1:
            raise DeployError(f"Invalid robot client PID: {pid}")
        return pid

    def release_managed_alignment_worker(
        self,
        robot: RobotConfig,
        *,
        hold_s: float = 15.0,
    ) -> bool:
        """Yield GDK from the managed alignment worker before a robot task.

        This endpoint only stops the child owned by the Nexus alignment agent;
        it never signals vendor teleoperation, collection, or GDK services.
        A bounded hold prevents an open alignment browser from winning the
        small race between release and robot-client lock acquisition.
        """

        if robot.name != "g2a":
            return False
        hold_s = float(hold_s)
        if not math.isfinite(hold_s) or not 1.0 <= hold_s <= 60.0:
            raise DeployError("Alignment release hold must be within [1, 60] seconds")
        _ssh_user, separator, control_host = robot.ssh_target.rpartition("@")
        if not separator or not control_host:
            raise DeployError(f"Robot {robot.name} ssh_target must include user@host")
        endpoint = (
            f"http://{control_host}:8080/api/alignment/release"
            f"?hold_s={hold_s:g}"
        )
        health = f"http://{control_host}:8080/api/alignment/health"
        command = (
            "set -eu; "
            f"if curl --noproxy '*' --connect-timeout 1 --max-time 2 --fail --silent "
            f"{shlex.quote(health)} >/dev/null 2>&1; then "
            f"curl --noproxy '*' --connect-timeout 1 --max-time 4 --fail --silent "
            f"-X POST {shlex.quote(endpoint)}; "
            "else printf '%s\\n' '{\"released\": false, \"gdk_state\": \"unavailable\"}'; fi"
        )
        result = self.run_ssh(command, robot=robot, capture=True, timeout_s=8.0)
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise DeployError("Alignment agent release returned invalid JSON") from error
        if not isinstance(response, dict):
            raise DeployError("Alignment agent release returned an invalid response")
        state = str(response.get("gdk_state", ""))
        if state not in {"idle", "unavailable"}:
            raise DeployError("Alignment agent did not release GDK")
        return state == "idle"

    def robot_runtime_preflight(self, robot: RobotConfig) -> dict[str, Any]:
        """Inspect the non-mutating GDK prerequisites used by read-only sessions."""
        script = r"""import json, os
from pathlib import Path
asset_id = ""
try:
    for line in Path("/data/version/aid_info.yaml").read_text(errors="replace").splitlines():
        if line.strip().startswith("AID:"):
            asset_id = line.split(":", 1)[1].strip().strip('"\'')
            break
except OSError:
    pass
devices = []
for name in ("/dev/ptp_xgi0", "/dev/ptp3"):
    path = Path(name)
    devices.append({"path": name, "exists": path.exists(),
                    "readable": os.access(name, os.R_OK),
                    "writable": os.access(name, os.W_OK)})
processes = []
external_gdk_clients = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit(): continue
    if int(entry.name) == os.getpid(): continue
    try:
        argv = [item.decode(errors="replace") for item in
                (entry / "cmdline").read_bytes().split(b"\0") if item]
    except Exception: continue
    names = {Path(item).name for item in argv[:3]}
    is_gdk = bool(names & {"gdk_service", "gdk_http_server", "camera_service"})
    is_discovery = "fastdds" in names and "discovery" in argv
    if is_gdk or is_discovery:
        processes.append(" ".join(argv))
        continue
    try:
        maps = (entry / "maps").read_text(errors="replace")
    except Exception:
        maps = ""
    if "agibot_gdk" in maps or "libgdk_" in maps:
        external_gdk_clients.append({"pid": int(entry.name), "command": " ".join(argv)})
print(json.dumps({"uid": os.getuid(), "user": os.environ.get("USER", ""),
                  "asset_id": asset_id,
                  "ptp_devices": devices, "gdk_processes": processes,
                  "external_gdk_clients": external_gdk_clients}))"""
        result = self.run_ssh(
            f"python3 -c {shlex.quote(script)}",
            robot=robot,
            capture=True,
        )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise DeployError("Robot runtime preflight returned invalid JSON") from error
        return value if isinstance(value, dict) else {}

    @staticmethod
    def require_robot_identity(robot: RobotConfig, checks: dict[str, Any]) -> None:
        """Fail closed when a network address resolves to another physical robot."""

        expected = robot.expected_aid.strip()
        if not expected:
            return
        observed = str(checks.get("asset_id", "")).strip()
        if not observed:
            raise DeployError(
                f"无法读取机器人硬件身份 AID；期望 {expected}。为避免连接错机器人，部署、观测和真机执行均已阻止。"
            )
        if observed != expected:
            raise DeployError(
                f"机器人硬件身份不匹配：期望 {expected}，实际 {observed}。"
                "该 IP 指向另一台机器人，已阻止部署、观测和真机执行。"
            )

    def require_robot_read_only_ready(self, robot: RobotConfig) -> dict[str, Any]:
        checks = self.robot_runtime_preflight(robot)
        self.require_robot_identity(robot, checks)
        ptp_devices = checks.get("ptp_devices", [])
        usable_ptp = any(
            isinstance(item, dict) and item.get("readable") and item.get("writable") for item in ptp_devices
        )
        checks["ptp_ready"] = usable_ptp
        if not usable_ptp:
            user = str(checks.get("user") or "remote user")
            warnings = list(checks.get("warnings", ()))
            warnings.append(
                f"运行用户 {user} 无权读写 /dev/ptp_xgi0 或 /dev/ptp3；"
                "PTP 诊断仅记录为警告，将以实际 GDK 初始化、关节和相机观测结果判定。"
            )
            checks["warnings"] = warnings
        external_clients = [item for item in checks.get("external_gdk_clients", ()) if isinstance(item, dict)]
        if external_clients:
            warnings = list(checks.get("warnings", ()))
            pids = ", ".join(str(item.get("pid", "?")) for item in external_clients)
            warnings.append(
                f"检测到其他进程正在使用 GDK（PID {pids}）；只读观测可继续，真机执行必须先确认并释放独占控制权。"
            )
            checks["warnings"] = warnings
        process_text = "\n".join(str(item) for item in checks.get("gdk_processes", ()))
        missing = [
            label
            for label, marker in (
                ("GDK runtime", "gdk_service"),
                ("GDK HTTP", "gdk_http_server"),
                ("camera_service", "camera_service"),
                ("Fast DDS discovery", "fastdds discovery"),
            )
            if marker not in process_text
        ]
        if missing:
            raise DeployError(
                "G2 只读观测预检失败：缺少机器人基础服务："
                + "、".join(missing)
                + "。尚未采集图像，也尚未发送推理请求。"
            )
        return checks

    def robot_session_snapshot(self, robot: RobotConfig, session_id: str) -> dict[str, Any]:
        session_dir = self.robot_session_directory(session_id)
        script = r"""import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
def last_json(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2); end = stream.tell(); data = b""; position = end
            while position > 0 and data.count(b"\n") < 2:
                size = min(8192, position); position -= size; stream.seek(position)
                data = stream.read(size) + data
        lines = data.splitlines()
        return json.loads(lines[-1]) if lines else {}
    except Exception:
        return {}
pid = 0
try: pid = int((root / "client.pid").read_text().strip())
except Exception: pass
running = bool(pid > 1 and Path(f"/proc/{pid}").exists())
timing = last_json(root / "timing.jsonl")
control = last_json(root / "control-status.json")
runs = sorted((root / "trace").glob("*")) if (root / "trace").is_dir() else []
trace = last_json(runs[-1] / "requests.jsonl") if runs else {}
frame = {}
try:
    value = json.loads((root / "inference-frames" / "latest.json").read_text())
    if isinstance(value, dict): frame = value
except Exception: pass
run_meta = {}
if runs:
    try: run_meta = json.loads((runs[-1] / "run_metadata.json").read_text())
    except Exception: pass
trace_images = trace.get("images", {}) if isinstance(trace, dict) else {}
frame_images = frame.get("images", {}) if isinstance(frame, dict) else {}
images = frame_images if frame_images else trace_images
def modified(path):
    try: return path.stat().st_mtime
    except Exception: return None
print(json.dumps({"pid": pid, "running": running, "control": control,
                  "timing": timing, "trace": trace,
                  "frame": frame,
                  "run_metadata": run_meta, "image_cameras": sorted(images),
                  "timing_updated_at": modified(root / "timing.jsonl"),
                  "frame_updated_at": modified(root / "inference-frames" / "latest.json"),
                  "trace_updated_at": modified(runs[-1] / "requests.jsonl") if runs else None}))"""
        result = self.run_ssh(
            f"python3 -c {shlex.quote(script)} {shlex.quote(session_dir)}",
            robot=robot,
            capture=True,
        )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise DeployError("Robot telemetry response was not valid JSON") from error
        return value if isinstance(value, dict) else {}

    def set_robot_session_paused(
        self,
        robot: RobotConfig,
        session_id: str,
        paused: bool,
        timeout_s: float = 12.0,
    ) -> dict[str, Any]:
        """Request an acknowledged in-process pause/resume transition."""
        session_dir = self.robot_session_directory(session_id)
        desired = "paused" if paused else "running"
        command = "pause" if paused else "resume"
        script = r"""import json, os, sys, time
from pathlib import Path
root, command, desired, timeout = Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
pid = int((root / "client.pid").read_text().strip())
if pid <= 1 or not Path(f"/proc/{pid}").exists():
    raise SystemExit("robot client is not running")
sequence = time.time_ns()
target = root / "control.json"
temporary = root / f"control.tmp-{os.getpid()}"
temporary.write_text(json.dumps({"sequence": sequence, "command": command}), encoding="utf-8")
temporary.replace(target)
deadline = time.monotonic() + timeout
last = {}
while time.monotonic() < deadline:
    if not Path(f"/proc/{pid}").exists(): raise SystemExit("robot client exited during control transition")
    try: last = json.loads((root / "control-status.json").read_text(encoding="utf-8"))
    except Exception: last = {}
    if int(last.get("sequence", -1)) == sequence:
        if last.get("state") == desired:
            print(json.dumps(last)); raise SystemExit(0)
        if last.get("state") == "error":
            raise SystemExit(str(last.get("message") or "control transition failed"))
    time.sleep(0.05)
raise SystemExit(f"timed out waiting for {desired}; last={last}")"""
        try:
            result = self.run_ssh(
                f"python3 -c {shlex.quote(script)} {shlex.quote(session_dir)} "
                f"{shlex.quote(command)} {shlex.quote(desired)} {float(timeout_s):g}",
                robot=robot,
                capture=True,
                timeout_s=timeout_s + 3.0,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise DeployError(f"Robot soft {command} failed: {str(detail).strip()}") from error
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise DeployError("Robot soft pause response was invalid") from error
        if not isinstance(value, dict) or value.get("state") != desired:
            raise DeployError(f"Robot did not acknowledge {desired}")
        return value

    def tail_robot_session(self, robot: RobotConfig, session_id: str, lines: int = 60) -> str:
        session_dir = self.robot_session_directory(session_id)
        result = self.run_ssh(
            f"tail -n {int(lines)} {shlex.quote(session_dir + '/console.log')} 2>/dev/null || true",
            robot=robot,
            capture=True,
        )
        return result.stdout

    def wait_robot_session_ready(
        self,
        robot: RobotConfig,
        session_id: str,
        timeout_s: float = 75.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_snapshot: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if cancel_check is not None and cancel_check():
                raise DeployError("Robot session startup was cancelled")
            last_snapshot = self.robot_session_snapshot(robot, session_id)
            console = self.tail_robot_session(robot, session_id, lines=50)
            if "Robot client started session=" in console:
                return last_snapshot
            if not last_snapshot.get("running"):
                raise DeployError(f"Robot client exited before becoming ready\n{console}")
            time.sleep(1.0)
        raise DeployError(
            f"Timed out waiting for robot client to enter its control loop\n"
            f"{self.tail_robot_session(robot, session_id, lines=80)}"
        )

    def stop_robot_session(
        self, robot: RobotConfig, session_id: str, timeout_s: float = 10.0
    ) -> dict[str, Any]:
        session_dir = self.robot_session_directory(session_id)
        script = (
            "set -eu; "
            f"pid_file={shlex.quote(session_dir + '/client.pid')}; "
            'test -f "$pid_file" || exit 0; pid=$(cat "$pid_file"); '
            'case "$pid" in ""|*[!0-9]*) exit 43;; esac; '
            'test "$pid" -gt 1 || exit 43; '
            'if test -e "/proc/$pid/cmdline"; then '
            'cmd=$(tr "\\0" " " <"/proc/$pid/cmdline"); '
            'case "$cmd" in *robot_client*) ;; *) exit 44;; esac; '
            'kill -TERM "$pid" 2>/dev/null || true; '
            f"end=$((SECONDS+{max(1, int(timeout_s))})); "
            'while kill -0 "$pid" 2>/dev/null && test "$SECONDS" -lt "$end"; do sleep 0.2; done; '
            'if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi; '
            'fi; rm -f "$pid_file"'
        )
        try:
            self.run_ssh(script, robot=robot)
        except subprocess.CalledProcessError as error:
            raise DeployError(f"Refusing or failing to stop remote robot client (exit={error.returncode})") from error
        try:
            return self.archive_robot_session(robot, session_id)
        except (DeployError, OSError, subprocess.SubprocessError) as error:
            return {
                "status": "failed",
                "session_id": self._validate_session_id(session_id),
                "error": str(error),
            }

    def force_stop_robot_session(
        self,
        robot: RobotConfig,
        session_id: str,
        expected_pid: int,
        timeout_s: float = 5.0,
    ) -> dict[str, Any]:
        """Stop one exact Nexus client after re-checking its PID and session path.

        This is intentionally stricter than ``stop_robot_session`` because it is
        exposed through the occupied-session recovery flow.  Neither an arbitrary
        PID nor a process that merely happens to contain ``robot_client`` can be
        selected by the caller.
        """

        session_dir = self.robot_session_directory(session_id)
        try:
            pid = int(expected_pid)
        except (TypeError, ValueError) as error:
            raise DeployError("Invalid expected robot client pid") from error
        if pid <= 1:
            raise DeployError("Invalid expected robot client pid")
        script = r"""import os, signal, sys, time
from pathlib import Path
root, expected_pid, timeout = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
pid_file = root / "client.pid"
try: recorded_pid = int(pid_file.read_text().strip())
except Exception: raise SystemExit(43)
if recorded_pid != expected_pid or expected_pid <= 1: raise SystemExit(44)
proc = Path(f"/proc/{expected_pid}")
if not proc.exists():
    try: pid_file.unlink()
    except FileNotFoundError: pass
    raise SystemExit(0)
try: command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
except Exception: raise SystemExit(45)
session_marker = str(root) + "/"
if "-m robot_client" not in command or session_marker not in command: raise SystemExit(46)
try: os.kill(expected_pid, signal.SIGTERM)
except ProcessLookupError: pass
deadline = time.monotonic() + max(0.5, timeout)
while proc.exists() and time.monotonic() < deadline: time.sleep(0.05)
if proc.exists():
    try: os.kill(expected_pid, signal.SIGKILL)
    except ProcessLookupError: pass
    deadline = time.monotonic() + 1.0
    while proc.exists() and time.monotonic() < deadline: time.sleep(0.05)
if proc.exists(): raise SystemExit(47)
try: pid_file.unlink()
except FileNotFoundError: pass"""
        try:
            self.run_ssh(
                f"python3 -c {shlex.quote(script)} {shlex.quote(session_dir)} {pid} {float(timeout_s):g}",
                robot=robot,
                timeout_s=max(3.0, float(timeout_s) + 3.0),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            returncode = getattr(error, "returncode", "timeout")
            raise DeployError(
                f"Refusing or failing to force-stop the verified robot client (exit={returncode})"
            ) from error
        try:
            return self.archive_robot_session(robot, session_id)
        except (DeployError, OSError, subprocess.SubprocessError) as error:
            return {
                "status": "failed",
                "session_id": self._validate_session_id(session_id),
                "error": str(error),
            }

    def read_robot_camera(
        self,
        robot: RobotConfig,
        session_id: str,
        camera_name: str,
        frame_id: str | None = None,
    ) -> tuple[str, bytes]:
        if camera_name not in {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
            raise DeployError("Unknown camera")
        session_dir = self.robot_session_directory(session_id)
        requested_frame = str(frame_id or "")
        if requested_frame and (
            len(requested_frame) > 96
            or not all(character.isalnum() or character in "-_" for character in requested_frame)
        ):
            raise DeployError("Invalid camera frame id")
        script = r"""import json, sys
from pathlib import Path
root, camera, requested = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
frame_root = root / "inference-frames"
manifest = frame_root / (f"{requested}.json" if requested else "latest.json")
try:
    meta = json.loads(manifest.read_text())
    relative = meta.get("images", {}).get(camera, {}).get("path")
    if not isinstance(relative, str) or Path(relative).name != relative: raise ValueError
    path = frame_root / relative
    sys.stdout.buffer.write(path.read_bytes()); raise SystemExit(0)
except (OSError, ValueError, TypeError, KeyError): pass
runs = sorted((root / "trace").glob("*")) if (root / "trace").is_dir() else []
if not runs: raise SystemExit(4)
lines = (runs[-1] / "requests.jsonl").read_text().splitlines()
if not lines: raise SystemExit(5)
meta = json.loads(lines[-1]); index = int(meta["request_index"])
relative = meta.get("images", {}).get(camera, {}).get("path")
if not relative: raise SystemExit(6)
path = runs[-1] / "requests" / f"request_{index:04d}" / relative
sys.stdout.buffer.write(path.read_bytes())"""
        try:
            result = subprocess.run(
                [
                    *self.ssh_base(robot),
                    f"python3 -c {shlex.quote(script)} {shlex.quote(session_dir)} "
                    f"{shlex.quote(camera_name)} {shlex.quote(requested_frame)}",
                ],
                check=True,
                capture_output=True,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired as error:
            raise DeployError("Robot camera read timed out") from error
        data = result.stdout
        if data.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif data.startswith(b"P6\n"):
            mime = "image/x-portable-pixmap"
        else:
            mime = "application/octet-stream"
        return mime, data

    def ensure_robot_client_idle(self, robot: RobotConfig | None = None) -> None:
        check = "if pgrep -af '[p]ython(3)? -m robot_client|[r]obot_client.client'; then exit 42; fi"
        try:
            result = subprocess.run(
                [*self.ssh_base(robot), check],
                text=True,
                capture_output=True,
                timeout=8.0,
            )
        except subprocess.TimeoutExpired as error:
            raise DeployError("Robot process-state check timed out") from error
        if result.returncode == 42:
            raise DeployError(f"A robot client is already running; refusing operation:\n{result.stdout}")
        if result.returncode != 0:
            raise DeployError(f"Could not verify robot process state: {result.stderr.strip()}")

    def check_robot_reset_pose(self, robot: RobotConfig) -> dict[str, Any]:
        """Read live 24D feedback through the installed, motion-free GDK probe."""

        self.ensure_robot_client_idle(robot)
        checks = self.robot_runtime_preflight(robot)
        self.require_robot_identity(robot, checks)
        if checks.get("external_gdk_clients"):
            raise DeployError("Other GDK clients are still active; reset pose is unknown")
        source = robot.source_client / "check_reset_pose.py"
        if not source.is_file():
            raise DeployError("Read-only reset-pose probe is missing on Thor")
        dependency_hashes = {}
        for name in ("check_reset_pose.py", "constants.py", "reset_to_dataset_start.py",
                     "robot_lock.py", "debug_python.sh"):
            local = robot.source_client / name
            if not local.is_file():
                raise DeployError(f"Reset-pose probe dependency is missing on Thor: {name}")
            dependency_hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()
        remote_script = f"{robot.remote_directory}/robot_client/check_reset_pose.py"
        remote_python = f"{robot.remote_directory}/robot_client/debug_python.sh"
        checks = []
        for name, expected_hash in dependency_hashes.items():
            remote_path = f"{robot.remote_directory}/robot_client/{name}"
            quoted = shlex.quote(remote_path)
            checks.append(
                f'test -f {quoted} && test "$(sha256sum {quoted} | cut -d " " -f 1)" = '
                f'{shlex.quote(expected_hash)} || '
                f'{{ echo "reset-pose dependency differs from Thor: {name}" >&2; exit 46; }}'
            )
        command = "set -eu; " + "; ".join(checks) + "; " + (
            f'test -x {shlex.quote(remote_python)} || '
            '{ echo "reset-pose Python wrapper is not executable" >&2; exit 45; }; '
            f'{shlex.quote(remote_python)} {shlex.quote(remote_script)}'
        )
        try:
            result = self.run_ssh(command, robot=robot, capture=True, timeout_s=20.0)
            output = result.stdout
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "read-only reset-pose probe failed").strip()
            raise DeployError(detail[-400:]) from error
        lines = [line.removeprefix("HOME_POSE_JSON=") for line in output.splitlines()
                 if line.startswith("HOME_POSE_JSON=")]
        if len(lines) != 1:
            raise DeployError("Reset-pose probe returned no unique machine-readable result")
        try:
            pose = json.loads(lines[0])
        except (TypeError, ValueError) as error:
            raise DeployError("Reset-pose probe returned invalid JSON") from error
        if not isinstance(pose, dict) or pose.get("schema_version") != 1 or pose.get("profile_id") != BODY_RESET_PROFILE_ID:
            raise DeployError("Reset-pose probe profile or schema mismatch")
        status = pose.get("status")
        if status not in {"at_home", "not_home", "unknown"} or pose.get("at_home") is not (
            True if status == "at_home" else False if status == "not_home" else None
        ):
            raise DeployError("Reset-pose probe returned an inconsistent status")
        if status != "unknown" and pose.get("sample_count") != 3:
            raise DeployError("Reset-pose probe did not read three samples")
        return pose

    def reset_robot_with_dahuan_cycle(
        self,
        robot: RobotConfig,
        safety_profile_id: str,
        timeout_s: float = 70.0,
    ) -> dict[str, Any]:
        """Run the fixed Dahuan gripper cycle and body reset on one verified robot.

        The command path and arguments are not caller-controlled.  The reset tool
        owns the cross-process GDK lock and its wrapper hard-codes the validated
        Dahuan open-close-open sequence.
        """

        self.ensure_robot_client_idle(robot)
        checks = self.robot_runtime_preflight(robot)
        self.require_robot_identity(robot, checks)
        external_clients = [item for item in checks.get("external_gdk_clients", ()) if isinstance(item, dict)]
        if external_clients:
            pids = ", ".join(str(item.get("pid", "?")) for item in external_clients)
            raise DeployError(f"Other GDK clients are still active (PID {pids})")
        profile, _local_path, remote_profile_path, _generated = self.workspace_safety_profile(
            robot, safety_profile_id
        )
        if profile.get("calibrated") is not True or profile.get("execute_approved") is not True:
            raise DeployError("Robot reset requires an execute-approved calibrated safety profile")
        profile_name = Path(remote_profile_path).name
        reset_script = shlex.quote(robot.reset_script)
        command = (
            f"set -eu; script={reset_script}; "
            'test -x "$script" || { echo "configured reset tool is not deployed" >&2; exit 45; }; '
            'grep -q -- "--cycle-effectors" "$script" || '
            '{ echo "reset tool does not guarantee the gripper cycle" >&2; exit 46; }; '
            'grep -q -- "--effector-profile=dahuan" "$script" || '
            '{ echo "reset tool does not lock the Dahuan gripper profile" >&2; exit 47; }; '
            'grep -q -- "--safety-config" "$script" || '
            '{ echo "reset tool does not require workspace path validation" >&2; exit 48; }; '
            f'"$script" {shlex.quote(profile_name)}'
        )
        try:
            result = self.run_ssh(
                command,
                robot=robot,
                capture=True,
                timeout_s=max(10.0, float(timeout_s)),
            )
        except subprocess.TimeoutExpired as error:
            raise DeployError("Robot reset timed out and requested hold-current-pose") from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "reset command failed").strip()
            raise DeployError(f"Robot body reset failed: {detail[-1200:]}") from error
        output = result.stdout.strip()
        return {
            "completed": True,
            "effectors_commanded": True,
            "effector_sequence": ["open", "close", "open"],
            "output_tail": output[-2000:],
        }

    def probe_robot_client(self, robot: RobotConfig | None = None) -> dict[str, Any]:
        """Inspect the deployed client and process state without changing the robot."""
        robot = self.robot if robot is None else robot
        script = r"""import json, os, re, sys
from pathlib import Path
root = Path(os.environ["NEXUS_REMOTE"])
required_cli_options = json.loads(sys.argv[1])
info = {}
try:
    value = json.loads((root / "BUNDLE_INFO.json").read_text(encoding="utf-8"))
    if isinstance(value, dict): info = value
except Exception: pass
processes = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit(): continue
    try: command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except Exception: continue
    if not re.search(r"(?:^|\s)-m\s+robot_client(?:\.hitl\.g2_runtime\.service)?(?:\s|$)", command): continue
    session = re.search(
        r"/(?:home/[^/]+/\.cache/nexus-platform-sessions|tmp/nexus_platform)/"
        r"([0-9a-f]{1,64})(?:/|\s|$)",
        command,
    )
    arguments = command.split()
    hardware_mode = ("execute" if "--execute" in arguments else
                     "read-only" if "--read-only" in arguments else
                     "dry-run" if "--dry-run" in arguments else "unknown")
    processes.append({"pid": int(entry.name),
                      "session_id": session.group(1) if session else None,
                      "hardware_mode": hardware_mode,
                      "human_in_loop": "-m robot_client.hitl.g2_runtime.service" in command,
                      "managed": bool(session)})
run_sh = root / "robot_client" / "run.sh"
module = root / "robot_client" / "__main__.py"
client = root / "robot_client" / "client.py"
declared = info.get("files", []) if isinstance(info.get("files"), list) else []
missing = [item for item in declared
           if not isinstance(item, str) or not (root / "robot_client" / item).is_file()]
try: client_source = client.read_text(encoding="utf-8")
except Exception: client_source = ""
missing_cli_options = [option for option in required_cli_options if option not in client_source]
print(json.dumps({"installed": run_sh.is_file() and module.is_file() and bool(declared),
                  "manifest_complete": bool(declared) and not missing,
                  "missing_files": missing,
                  "missing_cli_options": missing_cli_options,
                  "run_sh_executable": os.access(run_sh, os.X_OK),
                  "processes": processes,
                  "protocol": info.get("protocol"),
                  "start_state": info.get("start_state"),
                  "capabilities": info.get("capabilities"),
                  "bundle_robot": info.get("robot")}))"""
        remote_command = (
            f"NEXUS_REMOTE={shlex.quote(robot.remote_directory)} "
            f"python3 -c {shlex.quote(script)} "
            f"{shlex.quote(json.dumps(ROBOT_CLIENT_REQUIRED_CLI_OPTIONS))}"
        )
        try:
            result = subprocess.run(
                [*self.ssh_base(robot), remote_command],
                check=True,
                text=True,
                capture_output=True,
                timeout=6.0,
            )
        except subprocess.TimeoutExpired as error:
            raise DeployError("SSH authenticated, but robot client inspection timed out") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or "SSH command failed"
            raise DeployError(f"Could not inspect robot client: {detail}") from error
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise DeployError("Robot client inspection returned invalid JSON") from error
        if not isinstance(value, dict):
            raise DeployError("Robot client inspection returned an invalid payload")
        processes = value.get("processes") if isinstance(value.get("processes"), list) else []
        protocol = value.get("protocol") if isinstance(value.get("protocol"), dict) else {}
        capabilities = value.get("capabilities") if isinstance(value.get("capabilities"), dict) else {}
        start_state = value.get("start_state") if isinstance(value.get("start_state"), dict) else {}
        approved_start_state_valid = bool(
            start_state.get("profile_id") == BODY_RESET_PROFILE_ID
            and start_state.get("symbol") == "APPROVED_START_STATE_24D"
            and start_state.get("canonical_dim") == 24
            and start_state.get("legacy_aliases") == ["DATASET_START_STATE_24D", "BODY_RESET_TARGET_24D"]
        )
        value["approved_start_state_valid"] = approved_start_state_valid
        body_reset = capabilities.get("body_reset") if isinstance(capabilities.get("body_reset"), dict) else {}
        reset_required = self.raw.get("robots", {}).get(robot.name, {}).get("reset_enabled") is True
        reset_capability_valid = bool(
            body_reset.get("available") is True
            and body_reset.get("entrypoint") == "robot_client/reset_all_joints.sh"
            and body_reset.get("profile_id") == BODY_RESET_PROFILE_ID
            and body_reset.get("commanded_body_joints") == 22
            and body_reset.get("effectors_read") is True
            and body_reset.get("effectors_commanded") is True
            and body_reset.get("effector_profile") == "dahuan"
            and body_reset.get("effector_sequence") == ["open", "close", "open"]
            and body_reset.get("max_stage_joint_travel_rad") == 1.0
            and body_reset.get("max_total_joint_travel_rad") == 4.0
            and body_reset.get("body_path") == "workspace_checked_piecewise_linear_joint_space"
            and body_reset.get("workspace_path_samples") == 64
            and body_reset.get("workspace_surface") == "calibrated_plane_origin_and_normal"
        )
        value["reset_capability_valid"] = reset_capability_valid
        safety_runtime = (
            capabilities.get("safety_runtime") if isinstance(capabilities.get("safety_runtime"), dict) else {}
        )
        safety_runtime_valid = bool(
            safety_runtime.get("version") == G2_SAFETY_RUNTIME_CAPABILITY_VERSION
            and safety_runtime.get("configured_servo_hz") == 100
            and safety_runtime.get("collision_check_hz") == 50
            and safety_runtime.get("collision_check_rate_semantics") == "upper_limit"
            and safety_runtime.get("checked_segment_steps") == 5
            and safety_runtime.get("startup_checked_steps") == 10
            and safety_runtime.get("refill_low_water_steps") == 8
            and safety_runtime.get("max_buffer_steps") == 13
            and safety_runtime.get("queue_refill") == "append_only"
            and safety_runtime.get("geometry_mode") == "fixed_stride2_sampled_segment"
            and safety_runtime.get("surface_geometry") == "calibrated_plane_origin_and_normal"
            and safety_runtime.get("inter_arm_geometry") == "cross_chain_capsules_v2_broad_narrow"
            and safety_runtime.get("kinematic_authority") == "canonical_crsb_v4"
            and safety_runtime.get("inter_arm_geometry_source")
            == "canonical_collision_overlay"
            and safety_runtime.get("inter_arm_waypoint_coverage") == "fixed_stride2_50hz_targets"
            and safety_runtime.get("inter_arm_validation_gate") == "profile_scoped"
            and safety_runtime.get("collision_action_stride") == 2
            and safety_runtime.get("geometry_reference_max_age_ms") == 200
            and safety_runtime.get("motion_feedback_max_age_ms") == 40
            and safety_runtime.get("stationary_hold_feedback_max_age_ms") == self.stationary_hold_feedback_max_age_ms
            and safety_runtime.get("dispatch_deadline_ms") == 200
            and safety_runtime.get("safety_worker") == "process"
            and safety_runtime.get("worker_check_budget_ms") == 20
            and safety_runtime.get("worker_check_budget_policy") == "diagnostic_not_admission"
            and safety_runtime.get("hold_revalidation") == "historical_target_with_worker_liveness"
            and safety_runtime.get("danger_response") == "hold_last_accepted_and_continue"
            and safety_runtime.get("device_position_effector_tracking")
            == "opening_failure_fatal_closing_contact_warning"
            and safety_runtime.get("escape_policy") == "sampled_per_vertex_nonworsening_actual_gripper"
            and safety_runtime.get("projection_adoption") is False
            and safety_runtime.get("soft_pause") == "measured_reanchor"
            and safety_runtime.get("session_generation_fencing") is True
        )
        value["safety_runtime_valid"] = safety_runtime_valid
        valid = bool(
            value.get("installed")
            and value.get("manifest_complete")
            and not value.get("missing_cli_options")
            and value.get("run_sh_executable")
            and value.get("bundle_robot") == robot.name
            and is_supported_policy_protocol(protocol)
            and approved_start_state_valid
            and (not reset_required or reset_capability_valid)
            and (robot.name != "g2a" or safety_runtime_valid)
        )
        value["valid"] = valid
        value["running"] = bool(processes)
        value["status"] = (
            "running"
            if processes and valid
            else "idle"
            if valid
            else "invalid_client"
            if value.get("installed")
            else "not_deployed"
        )
        return value

    def manifest(self, robot: RobotConfig | None = None) -> tuple[str, ...]:
        robot = self.robot if robot is None else robot
        path = robot.manifest
        entries = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not entries or any(Path(item).is_absolute() or ".." in Path(item).parts for item in entries):
            raise DeployError(f"Unsafe or empty robot manifest: {path}")
        return entries

    def verify_dry_run_source(self, source: Path) -> None:
        constants = (source / "constants.py").read_text(encoding="utf-8")
        transport = (source / "transport.py").read_text(encoding="utf-8")
        client = (source / "client.py").read_text(encoding="utf-8")
        adapter = (source / "gdk_adapter.py").read_text(encoding="utf-8")
        safety = source / "safety_layer.py"
        required = {
            "generic protocol": f'PROTOCOL_NAME = "{POLICY_PROTOCOL_NAME}"' in constants,
            "previous protocol compatibility": '"g2a-policy-websocket"' in constants,
            "legacy compatibility": "SUPPORTED_PROTOCOLS" in constants and "SUPPORTED_PROTOCOLS" in transport,
            "dry-run CLI": 'hardware.add_argument("--dry-run"' in client,
            "robot capability CLI": all(option in client for option in ROBOT_CLIENT_REQUIRED_CLI_OPTIONS),
            "dry-run default": 'parser.set_defaults(hardware_mode="dry-run")' in client,
            "dry-run bypasses GDK": (
                'if self.config.hardware_mode == "dry-run":' in adapter
                and adapter.index('if self.config.hardware_mode == "dry-run":') < adapter.index("import agibot_gdk")
            ),
            "dry-run does not send": 'elif self.config.hardware_mode == "dry-run":' in adapter,
            "calibrated table safety layer": (
                safety.is_file()
                and "load_safety_layer" in adapter
                and '"--safety-config"' in client
                and (source / "SAFETY_CONFIG_TEMPLATE.json").is_file()
                and (source / "assets/agibot_g2/G2_omnipicker_fixed_dual.urdf").is_file()
                and (source / "assets/agibot_g2/SOURCE.json").is_file()
            ),
            "100 Hz fail-closed safety runtime": (
                "validate_workspace_safety_servo_rate" in client
                and "reanchor_measured_hold" in client
                and "SafetyDispatchExpiredError" in adapter
                and "allow_projected_actions=False" in adapter
                and "reanchor_measured_hold" in adapter
                and '"--safety-check-fps"' in client
                and "guard_trajectory" in adapter
                and "_consume_certified_step" in adapter
                and "_send_async_action" in adapter
                and (source / "async_safety.py").is_file()
            ),
            "Dahuan gripper-cycle reset tool": (
                (source / "debug_python.sh").is_file()
                and (source / "reset_to_dataset_start.py").is_file()
                and (source / "reset_all_joints.sh").is_file()
                and f'APPROVED_START_STATE_PROFILE_ID = "{BODY_RESET_PROFILE_ID}"' in constants
                and "BODY_RESET_PROFILE_ID = APPROVED_START_STATE_PROFILE_ID" in constants
                and "BODY_RESET_TARGET_24D" in constants
                and "APPROVED_START_STATE_24D" in constants
                and "BODY_RESET_TARGET_24D = APPROVED_START_STATE_24D" in constants
                and "BODY_RESET_TARGET_24D" in (source / "reset_to_dataset_start.py").read_text(encoding="utf-8")
                and "--cycle-effectors" in (source / "reset_all_joints.sh").read_text(encoding="utf-8")
                and "--effector-profile=dahuan" in (source / "reset_all_joints.sh").read_text(encoding="utf-8")
                and "--safety-config" in (source / "reset_all_joints.sh").read_text(encoding="utf-8")
                and "validate_reset_surface_path" in (source / "reset_to_dataset_start.py").read_text(encoding="utf-8")
            ),
        }
        missing = [name for name, valid in required.items() if not valid]
        if missing:
            raise DeployError(f"Robot client failed safety/source checks: {missing}")

    def verify_robot_source(self, robot: RobotConfig, source: Path | None = None) -> None:
        source = robot.source_client if source is None else source
        if robot.source_check == "g2a":
            self.verify_dry_run_source(source)
        elif robot.source_check == "manifest":
            if not source.is_dir():
                raise DeployError(f"Robot source directory does not exist: {source}")
        else:
            raise DeployError(f"Unknown source_check for robot {robot.name}: {robot.source_check}")

    def build_robot_bundle(self, robot: RobotConfig | None = None) -> Path:
        robot = self.robot if robot is None else robot
        source = robot.source_client
        self.verify_robot_source(robot, source)
        entries = self.manifest(robot)
        if "presence_stream.py" not in entries:
            raise DeployError("Robot manifest must include presence_stream.py")
        missing = [item for item in entries if not (source / item).is_file()]
        if missing:
            raise DeployError(f"Robot source is incomplete: {missing}")
        listed = set(entries)
        missing_imports = set()
        for item in entries:
            if not item.endswith(".py"):
                continue
            for node in ast.walk(ast.parse((source / item).read_text(encoding="utf-8"), filename=item)):
                if isinstance(node, ast.ImportFrom) and node.level and node.module:
                    package = Path(item).parent
                    for _ in range(node.level - 1):
                        package = package.parent
                    module = package.joinpath(*node.module.split(".")).as_posix()
                    if f"{module}.py" not in listed and f"{module}/__init__.py" not in listed:
                        missing_imports.add(f"{item}: {node.module}")
        if missing_imports:
            raise DeployError(f"Robot manifest omits imported modules: {sorted(missing_imports)}")
        destination = robot.bundle_directory
        package = destination / "robot_client"
        if destination.exists():
            shutil.rmtree(destination)
        package.mkdir(parents=True)
        for item in entries:
            (package / item).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item, package / item)
        background_services: dict[str, Any] = {}
        if robot.name == "g2a":
            project_root = Path(__file__).resolve().parents[2]
            alignment_source = Path(__file__).resolve().parent / "alignment" / "robot_agent.py"
            alignment_camera = project_root / "scripts" / "robot-alignment" / "alignment-camera.json"
            if not alignment_source.is_file() or not alignment_camera.is_file():
                raise DeployError("G2 alignment agent payload is incomplete")
            shutil.copy2(alignment_source, destination / "alignment_agent.py")
            shutil.copy2(alignment_camera, destination / "alignment-camera.json")
            background_services["alignment_agent"] = {
                "service": G2_ALIGNMENT_SERVICE,
                "entrypoint": "alignment_agent.py",
                "camera_config": "alignment-camera.json",
                "idle_release_s": 60,
                "sha256": hashlib.sha256(alignment_source.read_bytes()).hexdigest(),
            }
        (destination / "BUNDLE_INFO.json").write_text(
            json.dumps(
                {
                    "protocol": {
                        "name": POLICY_PROTOCOL_NAME,
                        "version": POLICY_PROTOCOL_VERSION,
                    },
                    "protocol_aliases": [
                        {"name": name, "version": version} for name, version in POLICY_PROTOCOL_IDENTITIES[1:]
                    ],
                    "legacy_protocol": {"name": "g2a-pi05-websocket", "version": 1},
                    "required_cli_options": list(ROBOT_CLIENT_REQUIRED_CLI_OPTIONS),
                    "start_state": {
                        "profile_id": BODY_RESET_PROFILE_ID,
                        "symbol": "APPROVED_START_STATE_24D",
                        "canonical_dim": 24,
                        "legacy_aliases": [
                            "DATASET_START_STATE_24D",
                            "BODY_RESET_TARGET_24D",
                        ],
                    },
                    "capabilities": {
                        "human_in_loop": {
                            "available": True,
                            "version": 1,
                            "entrypoint": "robot_client.hitl.g2_runtime.service",
                            "rollback_dispatch": "checked_gdk_receipt",
                            "tablet_recorder": "vendor_owned",
                        },
                        "effector_control": {
                            "version": 2,
                            "modes": ["software-trajectory", "device-position"],
                            "device_position_profile": "dahuan",
                            "opening_geometry": "measured_feedback_opening",
                            "policy_sampling": "held_goal_per_policy_interval",
                            "command_hz": 100,
                            "dahuan_command_shaping": "endpoint_latch_7p5mm_17p5mm",
                        },
                        "safety_runtime": {
                            "version": G2_SAFETY_RUNTIME_CAPABILITY_VERSION,
                            "configured_servo_hz": 100,
                            "collision_check_hz": 50,
                            "collision_check_rate_semantics": "upper_limit",
                            "checked_segment_steps": 5,
                            "startup_checked_steps": 10,
                            "refill_low_water_steps": 8,
                            "max_buffer_steps": 13,
                            "queue_refill": "append_only",
                            "geometry_mode": "fixed_stride2_sampled_segment",
                            "surface_geometry": "calibrated_plane_origin_and_normal",
                            "inter_arm_geometry": "cross_chain_capsules_v2_broad_narrow",
                            "kinematic_authority": "canonical_crsb_v4",
                            "inter_arm_geometry_source": "canonical_collision_overlay",
                            "inter_arm_waypoint_coverage": "fixed_stride2_50hz_targets",
                            "inter_arm_validation_gate": "profile_scoped",
                            "collision_action_stride": 2,
                            "geometry_reference_max_age_ms": 200,
                            "motion_feedback_max_age_ms": 40,
                            "stationary_hold_feedback_max_age_ms": self.stationary_hold_feedback_max_age_ms,
                            "dispatch_deadline_ms": 200,
                            "safety_worker": "process",
                            "worker_check_budget_ms": 20,
                            "worker_check_budget_policy": "diagnostic_not_admission",
                            "hold_revalidation": "historical_target_with_worker_liveness",
                            "danger_response": "hold_last_accepted_and_continue",
                            "device_position_effector_tracking": (
                                "opening_failure_fatal_closing_contact_warning"
                            ),
                            "escape_policy": "sampled_per_vertex_nonworsening_actual_gripper",
                            "projection_adoption": False,
                            "soft_pause": "measured_reanchor",
                            "session_generation_fencing": True,
                        },
                        "body_reset": {
                            "available": True,
                            "entrypoint": "robot_client/reset_all_joints.sh",
                            "profile_id": BODY_RESET_PROFILE_ID,
                            "commanded_body_joints": 22,
                            "effectors_read": True,
                            "effectors_commanded": True,
                            "effector_profile": "dahuan",
                            "effector_backend": "move_ee_pos(dual_tool,dahuan,bounded_reissue)",
                            "effector_sequence": ["open", "close", "open"],
                            "effector_cycle_pause_s": 0.15,
                            "effector_reissue_s": 0.75,
                            "effector_max_commands": 3,
                            "phase_feedback_required": True,
                            "failure_final_open_recovery": True,
                            "max_stage_joint_travel_rad": 1.0,
                            "max_total_joint_travel_rad": 4.0,
                            "body_path": "workspace_checked_piecewise_linear_joint_space",
                            "workspace_path_samples": 64,
                            "workspace_surface": "calibrated_plane_origin_and_normal",
                        },
                    },
                    "background_services": background_services,
                    "source": str(source),
                    "robot": robot.name,
                    "files": list(entries),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (package / "run.sh").chmod(0o755)
        print(f"built robot bundle: {destination}")
        return destination

    def deploy_robot_bundle(self, robot: RobotConfig | None = None) -> None:
        robot = self.robot if robot is None else robot
        bundle = robot.bundle_directory
        self.verify_robot_source(robot, bundle / "robot_client")
        if not (bundle / "BUNDLE_INFO.json").is_file():
            raise DeployError(f"Build the robot bundle first: {bundle}")
        runtime = self.robot_runtime_preflight(robot)
        self.require_robot_identity(robot, runtime)
        external_clients = [item for item in runtime.get("external_gdk_clients", ()) if isinstance(item, dict)]
        if external_clients:
            pids = ", ".join(str(item.get("pid", "?")) for item in external_clients)
            raise DeployError(f"Other GDK clients are active (PID {pids}); refusing to update their files")
        self.ensure_robot_client_idle(robot)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        remote = robot.remote_directory
        backup = f"{remote}.backup_{timestamp}"
        prepare = (
            "set -eu; "
            f"remote={shlex.quote(remote)}; backup={shlex.quote(backup)}; "
            'if test -e "$remote"; then test ! -e "$backup"; cp -a "$remote" "$backup"; fi; '
            'mkdir -p "$remote"'
        )
        self.run_ssh(prepare, robot=robot)
        subprocess.run(
            [
                "rsync",
                "-a",
                "--itemize-changes",
                "-e",
                self.rsync_shell(robot),
                f"{bundle}/",
                f"{robot.ssh_target}:{remote}/",
            ],
            check=True,
        )
        self._restart_robot_background_services(robot, bundle)
        removed = self._prune_robot_backups(robot, keep=ROBOT_BACKUP_RETENTION)
        print(f"deployed robot client: {robot.ssh_target}:{remote}")
        print(f"previous target backup (when present): {backup}")
        if removed:
            print(f"removed {len(removed)} expired robot backup(s): {', '.join(removed)}")

    @staticmethod
    def _validated_backup_prune_targets(
        remote_directory: str,
        names: list[str],
        *,
        keep: int,
    ) -> list[str]:
        """Return explicit old backup paths after strict basename validation."""
        remote = PurePosixPath(Deployment._validate_remote_directory(remote_directory))
        if keep < 1:
            raise DeployError("At least one robot rollback backup must be retained")
        expression = re.compile(rf"{re.escape(remote.name)}\.backup_\d{{8}}_\d{{6}}")
        validated: list[str] = []
        for name in names:
            if not name:
                continue
            if PurePosixPath(name).name != name or expression.fullmatch(name) is None:
                raise DeployError(f"Unsafe robot backup name returned by target: {name!r}")
            validated.append(name)
        newest_first = sorted(set(validated), reverse=True)
        return [str(remote.parent / name) for name in newest_first[keep:]]

    def _prune_robot_backups(
        self,
        robot: RobotConfig,
        *,
        keep: int,
    ) -> list[str]:
        remote = PurePosixPath(self._validate_remote_directory(robot.remote_directory))
        # Enumerate only backups produced by deploy_robot_bundle. Operators may
        # keep named rollback directories (for example ``backup_r14``) beside
        # them; those must neither block a deployment nor enter automatic
        # deletion. The strict validator below remains the final safety gate.
        digits = "[0-9]"
        pattern = f"{remote.name}.backup_{digits * 8}_{digits * 6}"
        listed = self.run_ssh(
            "find "
            f"{shlex.quote(str(remote.parent))} -mindepth 1 -maxdepth 1 -type d "
            f"-name {shlex.quote(pattern)} -printf '%f\\0'",
            robot=robot,
            capture=True,
        )
        targets = self._validated_backup_prune_targets(
            robot.remote_directory,
            listed.stdout.split("\0"),
            keep=keep,
        )
        if targets:
            self.run_ssh(
                "rm -rf -- " + " ".join(shlex.quote(path) for path in targets),
                robot=robot,
                timeout_s=180.0,
            )
        return targets

    def _restart_robot_background_services(self, robot: RobotConfig, bundle: Path) -> None:
        if robot.name != "g2a" or not (bundle / "alignment_agent.py").is_file():
            return
        remote = self._validate_remote_directory(robot.remote_directory)
        agent_command = shlex.join(
            [
                f"{remote}/robot_client/debug_python.sh",
                f"{remote}/alignment_agent.py",
                "--calibration",
                f"{remote}/alignment-camera.json",
                "--host",
                "10.42.1.102",
                "--port",
                "8080",
                "--fps",
                "8",
                "--camera-timeout-ms",
                "1500",
                "--idle-release-s",
                "60",
                "--allow-unauthenticated-private-lan",
            ]
        )
        command = (
            "set -eu; service_unit=''; "
            f"if systemctl is-enabled --quiet {G2_ALIGNMENT_SERVICE} 2>/dev/null; then "
            f"service_unit={G2_ALIGNMENT_SERVICE}; "
            "else "
            "for candidate in $(systemctl list-unit-files '*-g2-alignment-agent.service' "
            "--no-legend 2>/dev/null | awk '$2 == \"enabled\" {print $1}'); do "
            'exec_start=$(systemctl show -p ExecStart --value "$candidate"); '
            f'case "$exec_start" in *{shlex.quote(remote + "/alignment_agent.py")}*) '
            "service_unit=$candidate; break;; esac; done; "
            "fi; "
            'if test -n "$service_unit"; then '
            'old_pid=$(systemctl show -p MainPID --value "$service_unit"); '
            'case "$old_pid" in \'\'|*[!0-9]*) exit 48;; esac; test "$old_pid" -gt 1; '
            'kill -TERM "$old_pid"; restarted=0; '
            "for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do "
            'new_pid=$(systemctl show -p MainPID --value "$service_unit"); '
            'if test "$new_pid" -gt 1 2>/dev/null && test "$new_pid" != "$old_pid" && '
            "curl --noproxy '*' --connect-timeout 1 --max-time 1 --fail --silent "
            "http://10.42.1.102:8080/api/alignment/health >/dev/null; then "
            'restarted=1; break; fi; sleep 1; done; test "$restarted" = 1; '
            "else "
            "tmux kill-session -t nexus_alignment_agent 2>/dev/null || true; "
            f"tmux new-session -d -s nexus_alignment_agent {shlex.quote(agent_command)}; "
            "ready=0; for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do "
            "if curl --noproxy '*' --connect-timeout 1 --max-time 1 --fail --silent "
            "http://10.42.1.102:8080/api/alignment/health >/dev/null; then "
            'ready=1; break; fi; sleep 1; done; test "$ready" = 1; '
            "fi"
        )
        self.run_ssh(command, robot=robot, timeout_s=45.0)

    def test_robot_dry_run(self, model: ModelConfig, duration_s: float, robot: RobotConfig | None = None) -> None:
        robot = self.robot if robot is None else robot
        if duration_s <= 0 or duration_s > 30:
            raise DeployError("Dry-run duration must be in (0, 30] seconds")
        health = self.health(model)
        if health is None:
            raise DeployError(f"{model.name} server is not healthy at {self.health_url(model)}")
        self.ensure_robot_client_idle(robot)
        remote = robot.remote_directory
        command = [
            f"{remote}/robot_client/run.sh",
            f"--host={robot.policy_host}",
            f"--port={model.port}",
            "--task=Protocol-only dual-model deployment dry-run",
            "--inference-mode=async",
            "--dry-run",
            f"--duration-s={duration_s}",
            "--execution-fps=30",
            "--servo-fps=100",
            "--prediction-horizon=50",
            "--quality-horizon=50",
            "--warmup-requests=1",
            "--request-timeout-s=30",
            f"--log-file=/tmp/g2a_{model.name}_dry_run.log",
            f"--timing-log-file=/tmp/g2a_{model.name}_dry_run_timing.jsonl",
        ]
        if robot.policy_source_host:
            command.append(f"--source-address={robot.policy_source_host}")
        remote_command = f"cd {shlex.quote(remote)}; exec {shlex.join(command)}"
        print(f"running hardware-free G2 dry-run against {model.name} for {duration_s:.1f}s")
        self.run_ssh(remote_command, robot=robot)
        self.ensure_robot_client_idle(robot)
        print(f"G2 dry-run passed: model={model.name} duration_s={duration_s:.1f}")

    def doctor(self) -> None:
        failures: list[str] = []
        if not self.conda.is_file():
            failures.append(f"missing conda: {self.conda}")
        for model in self.models.values():
            if model.python_executable is not None and not model.python_executable.is_file():
                failures.append(f"missing direct Python executable for {model.name}: {model.python_executable}")
        if len({model.port for model in self.models.values()}) != len(self.models):
            failures.append("model ports are not unique")
        if self.require_unique_model_gpus and len(
            {model.cuda_visible_devices for model in self.models.values()}
        ) != len(self.models):
            failures.append("models are not assigned distinct CUDA_VISIBLE_DEVICES")
        for model in self.models.values():
            if not model.repo.is_dir():
                failures.append(f"missing {model.name} repo: {model.repo}")
            print(
                f"{model.name}: env={model.conda_env} gpu={model.cuda_visible_devices} "
                f"port={model.port} repo={model.repo}"
            )
        for robot in self.robots.values():
            try:
                self.verify_robot_source(robot)
            except (DeployError, OSError) as error:
                failures.append(f"{robot.name}: {error}")
            if not robot.ssh_key.is_file():
                failures.append(f"missing {robot.name} SSH key: {robot.ssh_key}")
            print(f"robot {robot.name}: target={robot.ssh_target} source={robot.source_client}")
        if failures:
            raise DeployError("Doctor failed:\n- " + "\n- ".join(failures))
        print("doctor: ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")

    server = commands.add_parser("server")
    server_commands = server.add_subparsers(dest="server_command", required=True)
    for name in ("command", "start"):
        item = server_commands.add_parser(name)
        item.add_argument("model", help="configured model name or 'all'")
        item.add_argument("--mock", action="store_true")
        if name == "start":
            item.add_argument("--wait-timeout-s", type=float, default=None)
    stop = server_commands.add_parser("stop")
    stop.add_argument("model", help="configured model name or 'all'")
    status = server_commands.add_parser("status")
    status.add_argument("model", default=MODEL_ALL, nargs="?", help="configured model name or 'all'")
    logs = server_commands.add_parser("logs")
    logs.add_argument("model", help="configured model name")
    logs.add_argument("--lines", type=int, default=30)

    client = commands.add_parser("client")
    client_commands = client.add_subparsers(dest="client_command", required=True)
    bundle = client_commands.add_parser("bundle")
    bundle.add_argument("--robot", default=None, help="configured robot name (default: first)")
    deploy = client_commands.add_parser("deploy")
    deploy.add_argument("--robot", default=None, help="configured robot name (default: first)")
    test = client_commands.add_parser("test-dry-run")
    test.add_argument("model", help="configured model name")
    test.add_argument("--robot", default=None, help="configured robot name (default: first)")
    test.add_argument("--duration-s", type=float, default=3.0)
    return parser


def handle_server(deployment: Deployment, args: argparse.Namespace) -> None:
    models = deployment.select_models(args.model)
    if args.server_command == "command":
        for model in models:
            visible_devices = "" if args.mock else model.cuda_visible_devices
            environment = f"CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)}"
            print(f"{model.name}: {environment} {shlex.join(deployment.model_command(model, mock=args.mock))}")
        return
    if args.server_command == "start":
        started: list[ModelConfig] = []
        try:
            for model in models:
                deployment.start_model(
                    model,
                    mock=args.mock,
                    wait_timeout_s=args.wait_timeout_s,
                    cuda_visible_devices="" if args.mock else None,
                )
                started.append(model)
        except Exception:
            for model in reversed(started):
                with contextlib.suppress(Exception):
                    deployment.stop_model(model)
            raise
        return
    if args.server_command == "stop":
        for model in reversed(models):
            deployment.stop_model(model)
        return
    if args.server_command == "status":
        for model in models:
            pid = deployment.running_pid(model)
            health = deployment.health(model) if pid is not None else None
            print(f"{model.name}: pid={pid} healthy={health is not None} port={model.port} health={health}")
        return
    if args.server_command == "logs":
        print(deployment.tail_console(models[0], lines=args.lines))
        return
    raise DeployError(f"Unsupported server command: {args.server_command}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        deployment = Deployment(args.config)
        if args.command == "doctor":
            deployment.doctor()
        elif args.command == "server":
            handle_server(deployment, args)
        elif args.command == "client":
            if args.client_command == "bundle":
                deployment.build_robot_bundle(deployment.select_robot(args.robot))
            elif args.client_command == "deploy":
                deployment.deploy_robot_bundle(deployment.select_robot(args.robot))
            elif args.client_command == "test-dry-run":
                deployment.test_robot_dry_run(
                    deployment.models[args.model],
                    duration_s=args.duration_s,
                    robot=deployment.select_robot(args.robot),
                )
            else:
                raise DeployError(f"Unsupported client command: {args.client_command}")
        else:
            raise DeployError(f"Unsupported command: {args.command}")
    except (DeployError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
