"""G2 URDF 运动学模型；只处理离线资产和内存中的关节状态。"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from yourdfpy import URDF


class ModelError(ValueError):
    """URDF 模型或关节请求无效。"""


@dataclass(frozen=True)
class JointInfo:
    name: str
    group: str
    lower: float
    upper: float
    value: float
    unit: str = "rad"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "lower": self.lower,
            "upper": self.upper,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class VisualMeshRecord:
    key: str
    link_name: str
    mesh_path: Path
    origin: np.ndarray
    scale: tuple[float, float, float] | None


def joint_group(name: str) -> str:
    if "arm_l" in name:
        return "左臂"
    if "arm_r" in name:
        return "右臂"
    if "gripper_l" in name:
        return "左夹爪"
    if "gripper_r" in name:
        return "右夹爪"
    if "head" in name:
        return "头部"
    if "body" in name:
        return "躯干"
    if "chassis" in name:
        return "底盘"
    return "其他"


class G2Model:
    def __init__(self, urdf_path: Path, *, asset_root: Path | None = None) -> None:
        self.urdf_path = urdf_path.expanduser().resolve()
        if not self.urdf_path.is_file():
            raise ModelError(f"G2 visual URDF 不存在: {self.urdf_path}")
        self.asset_root = (
            asset_root.expanduser().resolve() if asset_root is not None else self.urdf_path.parent
        )
        if not self.asset_root.is_dir() or (
            self.asset_root != self.urdf_path.parent
            and self.asset_root not in self.urdf_path.parents
        ):
            raise ModelError(f"G2 visual 资产根目录无效: {self.asset_root}")
        self._lock = threading.RLock()
        self.robot = URDF.load(
            str(self.urdf_path),
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=True,
            load_collision_meshes=False,
            force_mesh=False,
        )
        self._values = {name: 0.0 for name in self.robot.actuated_joint_names}
        self._controllable = self._collect_controllable_joints()
        self.robot.update_cfg(self._values)

    def _collect_controllable_joints(self) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        for joint in self.robot.actuated_joints:
            if joint.type not in {"revolute", "prismatic", "continuous"}:
                continue
            if joint.type == "continuous":
                lower, upper = -math.pi, math.pi
            elif joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
                continue
            else:
                lower, upper = float(joint.limit.lower), float(joint.limit.upper)
            if not math.isfinite(lower) or not math.isfinite(upper) or upper - lower <= 1e-9:
                continue
            result[joint.name] = (lower, upper)
        if not result:
            raise ModelError("URDF 中没有可调节关节")
        return result

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def joint_infos(self) -> list[JointInfo]:
        with self._lock:
            return [
                JointInfo(
                    name=name,
                    group=joint_group(name),
                    lower=limits[0],
                    upper=limits[1],
                    value=self._values[name],
                    unit=("m" if self.robot.joint_map[name].type == "prismatic" else "rad"),
                )
                for name, limits in self._controllable.items()
            ]

    def update_joints(self, values: dict[str, Any]) -> list[JointInfo]:
        if not isinstance(values, dict) or not values:
            raise ModelError("joints 必须是非空对象")
        validated: dict[str, float] = {}
        for name, raw in values.items():
            if name not in self._controllable:
                raise ModelError(f"未知或不可调节关节: {name}")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ModelError(f"关节 {name} 必须是有限数值")
            value = float(raw)
            if not math.isfinite(value):
                raise ModelError(f"关节 {name} 必须是有限数值")
            lower, upper = self._controllable[name]
            if value < lower or value > upper:
                raise ModelError(f"关节 {name}={value} 越界 [{lower}, {upper}]")
            validated[name] = value
        with self._lock:
            self._values.update(validated)
            self.robot.update_cfg(validated)
            return self.joint_infos()

    def reset(self) -> list[JointInfo]:
        with self._lock:
            self._values = {name: 0.0 for name in self.robot.actuated_joint_names}
            self.robot.update_cfg(self._values)
            return self.joint_infos()

    def geometry_records(self) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
        """返回 node、顶点、三角面、变换和近似颜色。"""

        records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
        with self._lock:
            for node in self.robot.scene.graph.nodes_geometry:
                transform, geometry_name = self.robot.scene.graph.get(node)
                geometry = self.robot.scene.geometry[geometry_name]
                vertices = np.asarray(geometry.vertices, dtype=np.float32)
                faces = np.asarray(geometry.faces, dtype=np.int32)
                color = 0xB8C0CC
                try:
                    visual = geometry.visual
                    rgba_value = getattr(visual, "main_color", None)
                    if rgba_value is None:
                        material = getattr(visual, "material", None)
                        rgba_value = getattr(material, "main_color", None)
                        if rgba_value is None:
                            rgba_value = getattr(material, "baseColorFactor", None)
                    rgba = np.asarray(rgba_value, dtype=np.uint8)
                    color = (int(rgba[0]) << 16) | (int(rgba[1]) << 8) | int(rgba[2])
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
                records.append((str(node), vertices, faces, np.asarray(transform), color))
        return records

    def linked_geometry_records(
        self,
    ) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]]:
        """返回 node、所属 link、网格、link 内局部变换和颜色。"""

        records: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]] = []
        with self._lock:
            for node in self.robot.scene.graph.nodes_geometry:
                link_name = str(self.robot.scene.graph.transforms.parents[node])
                local, geometry_name = self.robot.scene.graph.get(node, link_name)
                geometry = self.robot.scene.geometry[geometry_name]
                vertices = np.asarray(geometry.vertices, dtype=np.float32)
                faces = np.asarray(geometry.faces, dtype=np.int32)
                color = 0xB8C0CC
                try:
                    visual = geometry.visual
                    rgba_value = getattr(visual, "main_color", None)
                    if rgba_value is None:
                        material = getattr(visual, "material", None)
                        rgba_value = getattr(material, "main_color", None)
                        if rgba_value is None:
                            rgba_value = getattr(material, "baseColorFactor", None)
                    rgba = np.asarray(rgba_value, dtype=np.uint8)
                    color = (int(rgba[0]) << 16) | (int(rgba[1]) << 8) | int(rgba[2])
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
                records.append(
                    (str(node), link_name, vertices, faces, np.asarray(local), color)
                )
        return records

    def link_transforms(self, link_names: set[str]) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                name: np.asarray(self.robot.scene.graph.get(name)[0])
                for name in link_names
            }

    def geometry_transforms(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                str(node): np.asarray(self.robot.scene.graph.get(node)[0])
                for node in self.robot.scene.graph.nodes_geometry
            }

    def visual_mesh_records(self) -> list[VisualMeshRecord]:
        records: list[VisualMeshRecord] = []
        with self._lock:
            for link in self.robot.robot.links:
                for index, visual in enumerate(link.visuals):
                    mesh = visual.geometry.mesh
                    if mesh is None:
                        continue
                    mesh_path = (self.urdf_path.parent / mesh.filename).resolve()
                    if self.asset_root not in mesh_path.parents or not mesh_path.is_file():
                        raise ModelError(f"visual mesh 不存在或越出资产目录: {mesh.filename}")
                    scale = None
                    if mesh.scale is not None:
                        scale_values = tuple(float(value) for value in mesh.scale)
                        if len(scale_values) != 3 or not np.isfinite(scale_values).all():
                            raise ModelError(f"visual mesh scale 无效: {link.name}")
                        scale = scale_values
                    records.append(
                        VisualMeshRecord(
                            key=f"{link.name}:{index}",
                            link_name=link.name,
                            mesh_path=mesh_path,
                            origin=np.asarray(visual.origin, dtype=np.float64),
                            scale=scale,
                        )
                    )
        return records

    def visual_mesh_transforms(
        self, records: list[VisualMeshRecord]
    ) -> dict[str, np.ndarray]:
        transforms: dict[str, np.ndarray] = {}
        with self._lock:
            for record in records:
                transform = np.asarray(self.robot.get_transform(record.link_name)) @ record.origin
                if record.scale is not None:
                    scaling = np.eye(4, dtype=np.float64)
                    scaling[:3, :3] = np.diag(np.asarray(record.scale, dtype=np.float64))
                    transform = transform @ scaling
                transforms[record.key] = transform
        return transforms

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.robot.robot.name,
                "base_link": self.robot.base_link,
                "degrees_of_freedom": self.robot.num_dofs,
                "actuated_joint_count": self.robot.num_actuated_joints,
                "controllable_joint_count": len(self._controllable),
                "geometry_count": len(self.robot.scene.geometry),
                "visual_mesh_count": len(self.visual_mesh_records()),
                "vertex_count": sum(
                    len(geometry.vertices) for geometry in self.robot.scene.geometry.values()
                ),
                "face_count": sum(
                    len(geometry.faces) for geometry in self.robot.scene.geometry.values()
                ),
            }
