"""固定哈希导入真机安全实现，用于离线 Episode 的参考判定。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

# Nexus Simulator is intentionally nested under ``tools/``.  The production
# client source is the repository-owned, read-only reference; importing this
# pure geometry module does not initialize GDK or expose a command channel.
NEXUS_PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAFETY_SOURCE = (
    NEXUS_PROJECT_ROOT
    / "src"
    / "nexus_deploy"
    / "robot"
    / "g2a"
    / "safety_layer.py"
)
SAFETY_TEMPLATE = SAFETY_SOURCE.with_name("SAFETY_CONFIG_TEMPLATE.json")
EXPECTED_SOURCE_SHA256 = "1fc47d1ad9ea103ddb4caad13ede47dc05b7fb35d0bc929abfb65bde91e6af14"


class SafetyReferenceError(RuntimeError):
    """真机安全参考代码发生漂移或不能用于离线检查。"""


def _load_module() -> ModuleType:
    if not SAFETY_SOURCE.is_file():
        raise SafetyReferenceError(f"安全参考代码不存在: {SAFETY_SOURCE}")
    digest = hashlib.sha256(SAFETY_SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise SafetyReferenceError(f"安全参考代码哈希已变化: {digest}")
    spec = importlib.util.spec_from_file_location("nexus_pinned_safety_layer", SAFETY_SOURCE)
    if spec is None or spec.loader is None:
        raise SafetyReferenceError("无法构造安全参考模块")
    module = importlib.util.module_from_spec(spec)
    # dataclasses 在处理 postponed annotations 时要求模块已登记。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SafetyReference:
    def __init__(self, asset_root: Path) -> None:
        self.module = _load_module()
        data = json.loads(SAFETY_TEMPLATE.read_text(encoding="utf-8"))
        robot = data["robot_model"]
        model_root = asset_root.expanduser().resolve()
        robot["path"] = str(model_root / "G2_omnipicker_fixed_dual.urdf")
        for mesh in data["collision_meshes"]:
            mesh["path"] = str(model_root / "collision" / Path(mesh["path"]).name)
        points = self.module._load_collision_points(data, SAFETY_TEMPLATE.parent)
        geometry = self.module.GripperGeometry(points).conservative_bounds()
        frames = sorted({point.frame for point in geometry.points})
        self.fk = self.module.URDFForwardKinematics(
            Path(robot["path"]),
            root_link=robot["root_link"],
            required_frames=frames,
            joint_map=robot["joint_map"],
            joint_mimics=robot.get("joint_mimics"),
            fixed_joint_positions=robot.get("fixed_joint_positions"),
            base_transform=np.eye(4),
            expected_sha256=robot["sha256"],
        )
        self.geometry = geometry
        self.motion_indices = data["motion_indices"]
        self.passthrough_indices = data["passthrough_indices"]

    def _layer(self, *, height: float, hard_margin: float) -> Any:
        surface = self.module.TableConstraint(
            name="work_surface",
            height=height,
            hard_margin=hard_margin,
            xy_bounds=None,
        )
        return self.module.ActionSafetyLayer(
            fk=self.fk,
            geometry=self.geometry,
            surface=surface,
            motion_indices=self.motion_indices,
            passthrough_indices=self.passthrough_indices,
            path_samples=2,
            search_iterations=1,
        )

    def evaluate(
        self,
        current: np.ndarray,
        target: np.ndarray,
        *,
        height: float,
        hard_margin: float,
    ) -> dict[str, Any]:
        layer = self._layer(height=height, hard_margin=hard_margin)
        decision = layer.project(current, target)
        return {
            "hard_limit_triggered": decision.hard_limit_triggered,
            "safe": not decision.hard_limit_triggered,
            "reason": decision.reason,
            "current_z_min": decision.current_z_min,
            "predicted_z_min": decision.predicted_z_min,
            "path_z_min": decision.path_z_min,
            "distance_to_surface": decision.distance_to_surface,
            "clearance": (
                None
                if decision.distance_to_surface is None
                else decision.distance_to_surface - hard_margin
            ),
            "limiting_point": decision.limiting_point,
            "limiting_frame": decision.limiting_frame,
            "action_scale": decision.action_scale,
            "latency_ms": decision.projection_latency_ms,
            "implementation_sha256": EXPECTED_SOURCE_SHA256,
            "geometry": "conservative_bounds",
        }

    def collision_response(
        self,
        measured: np.ndarray,
        candidate: np.ndarray,
        last_successful: np.ndarray | None,
        *,
        height: float,
        hard_margin: float,
    ) -> dict[str, Any]:
        """离线复现 GDK adapter 的候选拒绝/上一动作保持选择逻辑。"""

        layer = self._layer(height=height, hard_margin=hard_margin)
        decision = layer.guard(measured, candidate)
        adopted = np.asarray(decision.safe_action, dtype=np.float64)
        response = "allow_candidate"
        reason = decision.reason
        previous_recheck_safe = None
        if (
            decision.reason == "realtime_guard_hold_pending_projection"
            and last_successful is not None
        ):
            previous = np.asarray(last_successful, dtype=np.float64)
            hold = layer.guard(measured, previous)
            previous_recheck_safe = not hold.hard_limit_triggered
            if previous_recheck_safe:
                adopted = previous.copy()
                response = "hold_previous_successful_action"
                reason = "collision_risk_hold_previous_safe_action"
            else:
                adopted = np.asarray(measured, dtype=np.float64).copy()
                response = "hold_measured_pose_previous_recheck_failed"
        elif decision.hard_limit_triggered:
            adopted = np.asarray(measured, dtype=np.float64).copy()
            response = "hold_measured_pose"
        return {
            "candidate_rejected": bool(decision.hard_limit_triggered),
            "collision_response": response,
            "reason": reason,
            "previous_recheck_safe": previous_recheck_safe,
            "adopted_action": adopted.tolist(),
            "candidate_action": np.asarray(candidate, dtype=np.float64).tolist(),
            "measured_action": np.asarray(measured, dtype=np.float64).tolist(),
        }

    def evaluate_action(
        self,
        measured: np.ndarray,
        candidate: np.ndarray,
        last_successful: np.ndarray | None,
        *,
        height: float,
        hard_margin: float,
        enforcement_enabled: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """持续检测候选动作，并按运行时开关决定是否采用安全层输出。"""

        detection = self.evaluate(
            measured,
            candidate,
            height=height,
            hard_margin=hard_margin,
        )
        response = self.collision_response(
            measured,
            candidate,
            last_successful,
            height=height,
            hard_margin=hard_margin,
        )
        collision_detected = bool(response["candidate_rejected"])
        if enforcement_enabled:
            adopted = np.asarray(response["adopted_action"], dtype=np.float64)
            disposition = str(response["collision_response"])
            enforcement_applied = collision_detected
        else:
            adopted = np.asarray(candidate, dtype=np.float64).copy()
            disposition = (
                "monitor_only_allow_unsafe_candidate"
                if collision_detected
                else "allow_candidate"
            )
            enforcement_applied = False
        adopted_pose_safe = True
        if collision_detected:
            adopted_pose_safe = bool(
                self.evaluate(
                    adopted,
                    adopted,
                    height=height,
                    hard_margin=hard_margin,
                )["safe"]
            )
        enforcement_effective = bool(
            not collision_detected
            or (enforcement_enabled and enforcement_applied and adopted_pose_safe)
        )
        if enforcement_enabled and collision_detected and not adopted_pose_safe:
            disposition = "cannot_recover_without_safe_pose"
        detection.update(
            {
                "safe": not collision_detected,
                "hard_limit_triggered": collision_detected,
                "reason": response["reason"],
                "collision_detected": collision_detected,
                "enforcement_enabled": enforcement_enabled,
                "enforcement_applicable": True,
                "enforcement_applied": enforcement_applied,
                "enforcement_effective": enforcement_effective,
                "adopted_pose_safe": adopted_pose_safe,
                "action_disposition": disposition,
            }
        )
        return detection, adopted
