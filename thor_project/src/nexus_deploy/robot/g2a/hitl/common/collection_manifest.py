"""Immutable collection-round contract bound into every FlowPRO sidecar."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

COLLECTION_MANIFEST_SCHEMA_VERSION = "flowpro.collection_manifest/v1"


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def validate_collection_manifest(raw: Any) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("Collection manifest must be an object")
    fields = {
        "schema_version",
        "status",
        "collection_round_id",
        "created_at_utc",
        "robot_id",
        "task_id",
        "task_prompt",
        "policy",
        "runtime",
        "execution",
        "data_gates",
    }
    if set(raw) != fields:
        raise ValueError(f"Collection manifest fields differ from v1: {sorted(set(raw) ^ fields)}")
    if raw["schema_version"] != COLLECTION_MANIFEST_SCHEMA_VERSION or raw["status"] != "frozen":
        raise ValueError("Collection manifest must use v1 and status='frozen'")
    for field in ("collection_round_id", "robot_id", "task_id", "task_prompt"):
        _nonempty(raw[field], f"collection manifest.{field}")
    try:
        timestamp = datetime.fromisoformat(str(raw["created_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("collection manifest.created_at_utc must be ISO-8601") from error
    if timestamp.tzinfo is None:
        raise ValueError("collection manifest.created_at_utc must include a UTC offset")

    policy = raw["policy"]
    policy_fields = {
        "checkpoint_id",
        "checkpoint_stats_sha256",
        "processor_version",
        "native_action_dim",
        "chunk_size",
        "effector_profile",
    }
    if not isinstance(policy, Mapping) or set(policy) != policy_fields:
        raise ValueError("Collection manifest policy fields differ from v1")
    for field in ("checkpoint_id", "processor_version", "effector_profile"):
        _nonempty(policy[field], f"collection manifest.policy.{field}")
    digest = policy["checkpoint_stats_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("collection manifest.policy.checkpoint_stats_sha256 is invalid")
    if policy["native_action_dim"] != 16:
        raise ValueError("Round-1 FlowPRO requires a 16D arm/effector Policy action without head/waist")
    if (
        not isinstance(policy["chunk_size"], int)
        or isinstance(policy["chunk_size"], bool)
        or policy["chunk_size"] <= 0
    ):
        raise ValueError("collection manifest.policy.chunk_size must be positive")

    runtime = raw["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {"schema_version", "source_sha256"}:
        raise ValueError("Collection manifest runtime fields differ from v1")
    if runtime["schema_version"] != "flowpro.g2_runtime_provenance/v1":
        raise ValueError("Collection manifest has an unsupported runtime provenance schema")
    runtime_digest = runtime["source_sha256"]
    if (
        not isinstance(runtime_digest, str)
        or len(runtime_digest) != 64
        or any(c not in "0123456789abcdef" for c in runtime_digest)
    ):
        raise ValueError("collection manifest.runtime.source_sha256 is invalid")

    execution = raw["execution"]
    execution_fields = {
        "inference_mode",
        "execution_fps",
        "servo_fps",
        "prediction_horizon",
        "quality_horizon",
        "min_replan_after_steps",
        "request_lead_margin_steps",
        "rtc_enabled",
        "head_waist_control",
        "rollback_enabled",
        "rollback_scope",
        "x_release_behavior",
        "rollback_plan_immutable",
        "collision_safety_enabled",
    }
    if not isinstance(execution, Mapping) or set(execution) != execution_fields:
        raise ValueError("Collection manifest execution fields differ from v1")
    if execution["inference_mode"] != "async" or execution["rtc_enabled"] is not False:
        raise ValueError("Round-1 collection is frozen to async non-RTC inference")
    if execution["head_waist_control"] is not False:
        raise ValueError("Round-1 FlowPRO must not control head/waist")
    if execution["rollback_enabled"] is not True:
        raise ValueError("Paper-original collection requires rollback")
    if execution["rollback_scope"] != "current_contiguous_policy_history":
        raise ValueError("Round-1 rollback scope must use all current contiguous Policy history")
    if execution["x_release_behavior"] != "same_plan_next_older_inference_anchor":
        raise ValueError("Round-1 X release must finish on the immutable plan's next older anchor")
    if execution["rollback_plan_immutable"] is not True:
        raise ValueError("Round-1 rollback must preserve one immutable plan")
    if not isinstance(execution["collision_safety_enabled"], bool):
        raise ValueError("collision_safety_enabled must be boolean")
    for field in ("execution_fps", "servo_fps"):
        _positive_number(execution[field], f"collection manifest.execution.{field}")
    for field in ("prediction_horizon", "quality_horizon", "min_replan_after_steps"):
        value = execution[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"collection manifest.execution.{field} must be a positive integer")
    lead = execution["request_lead_margin_steps"]
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        raise ValueError("collection manifest.execution.request_lead_margin_steps must be non-negative")
    if execution["quality_horizon"] > execution["prediction_horizon"]:
        raise ValueError("Collection manifest quality_horizon exceeds prediction_horizon")

    gates = raw["data_gates"]
    gate_fields = {
        "require_executed_policy",
        "require_same_anchor",
        "require_scene_confirmation",
        "require_zero_queue_underrun",
        "required_cameras",
        "max_camera_skew_ms",
    }
    if not isinstance(gates, Mapping) or set(gates) != gate_fields:
        raise ValueError("Collection manifest data_gates fields differ from v1")
    for field in (
        "require_executed_policy",
        "require_same_anchor",
        "require_scene_confirmation",
        "require_zero_queue_underrun",
    ):
        if gates[field] is not True:
            raise ValueError(f"Round-1 collection requires data_gates.{field}=true")
    if gates["required_cameras"] != ["cam_head", "cam_left_wrist", "cam_right_wrist"]:
        raise ValueError("Round-1 collection requires the exact ordered three-camera contract")
    skew = _positive_number(gates["max_camera_skew_ms"], "collection manifest.data_gates.max_camera_skew_ms")
    if skew > 40.0:
        raise ValueError("Round-1 max_camera_skew_ms cannot exceed 40 ms")


@dataclass(frozen=True)
class FrozenCollectionManifest:
    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": COLLECTION_MANIFEST_SCHEMA_VERSION,
            "collection_round_id": self.payload["collection_round_id"],
            "sha256": self.sha256,
            "runtime_source_sha256": self.payload["runtime"]["source_sha256"],
        }


def load_frozen_collection_manifest(path: Path) -> FrozenCollectionManifest:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"Collection manifest must be a regular non-symlink file: {resolved}")
    before = resolved.stat()
    content = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Collection manifest changed while loading")
    payload = json.loads(content)
    validate_collection_manifest(payload)
    return FrozenCollectionManifest(
        path=resolved,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=dict(payload),
    )


def validate_runtime_binding(
    manifest: FrozenCollectionManifest,
    *,
    args: Any,
    policy: Mapping[str, Any],
    runtime_source_sha256: str,
) -> None:
    payload = manifest.payload
    expected_top = {
        "robot_id": args.robot_id,
        "task_id": args.task_id,
        "task_prompt": args.task,
    }
    for field, actual in expected_top.items():
        if payload[field] != actual:
            raise ValueError(
                f"Collection manifest {field} differs from runtime: {payload[field]!r} != {actual!r}"
            )
    for field in (
        "checkpoint_id",
        "checkpoint_stats_sha256",
        "processor_version",
        "native_action_dim",
        "chunk_size",
        "effector_profile",
    ):
        if payload["policy"][field] != policy[field]:
            raise ValueError(f"Collection manifest policy.{field} differs from the live Policy server")
    if payload["runtime"]["source_sha256"] != runtime_source_sha256:
        raise ValueError("Collection manifest runtime source differs from the deployed G2 bundle")
    # v1 has no fields to bind these new execution semantics. A future formal
    # round must extend the manifest instead of silently changing a v1 run.
    if (
        getattr(args, "chunk_smoothing_window", 1) != 1
        or getattr(args, "effector_control_mode", "software-trajectory") != "software-trajectory"
        or getattr(args, "effector_command_mode", "continuous") != "continuous"
    ):
        raise ValueError("Collection manifest v1 cannot bind pilot chunk or gripper shaping")
    actual_execution = {
        "inference_mode": args.inference_mode,
        "execution_fps": float(args.fps),
        "servo_fps": float(args.servo_fps),
        "prediction_horizon": args.prediction_horizon,
        "quality_horizon": args.quality_horizon,
        "min_replan_after_steps": args.min_replan_after_steps,
        "request_lead_margin_steps": args.request_lead_margin_steps,
        "rtc_enabled": args.inference_mode == "rtc",
        "head_waist_control": int(policy["native_action_dim"]) == 24,
        "rollback_enabled": bool(args.enable_rollback and args.hardware_mode == "execute"),
        "rollback_scope": "current_contiguous_policy_history",
        "x_release_behavior": "same_plan_next_older_inference_anchor",
        "rollback_plan_immutable": True,
        "collision_safety_enabled": bool(args.hardware_mode == "execute" and args.safety_config is not None),
    }
    if dict(payload["execution"]) != actual_execution:
        differences = {
            key: {"manifest": payload["execution"][key], "runtime": actual_execution[key]}
            for key in actual_execution
            if payload["execution"][key] != actual_execution[key]
        }
        raise ValueError(f"Collection manifest execution differs from runtime: {differences}")
