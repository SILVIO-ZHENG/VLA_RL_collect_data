"""G2 visual-URDF model used by the integrated digital-twin renderer."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fast_simplification import simplify
from yourdfpy import URDF

RENDER_LOD_MIN_FACES = 500
RENDER_LOD_REDUCTION = {
    # The Dahuan source uses dense STL meshes.  Keep the articulated arms and
    # especially the gripper sharper while reducing static body/wheel payload.
    "body": 0.75,
    "arm": 0.72,
    "gripper": 0.45,
}


class ModelError(ValueError):
    """The bundled visual model or a joint update is invalid."""


def mesh_material_color(mesh: Any) -> int:
    """Read the DAE sub-mesh material color without losing PBR fallbacks.

    G2's Collada files contain embedded material colors rather than external image
    textures.  ``TextureVisuals.main_color`` is ``None`` with trimesh 5, while the
    actual color lives on its PBR material.  Keep the extraction order compatible
    with the previously validated simulator.
    """

    color = 0xB8C0CC
    try:
        visual = mesh.visual
        rgba_value = getattr(visual, "main_color", None)
        if rgba_value is None:
            material = getattr(visual, "material", None)
            rgba_value = getattr(material, "main_color", None)
            if rgba_value is None:
                rgba_value = getattr(material, "baseColorFactor", None)
        rgba = np.asarray(rgba_value, dtype=np.uint8)
        if rgba.size >= 3:
            color = (int(rgba[0]) << 16) | (int(rgba[1]) << 8) | int(rgba[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return color


class G2Model:
    """Thread-safe FK wrapper around the bundled G2 visual URDF."""

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = urdf_path.resolve()
        if not self.urdf_path.is_file():
            raise ModelError(f"内置 G2 visual URDF 不存在: {self.urdf_path}")
        self._asset_root = self.urdf_path.parent
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
        self._limits = self._joint_limits()
        self._render_records: list[
            tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]
        ] | None = None
        self.robot.update_cfg(self._values)

    def _joint_limits(self) -> dict[str, tuple[float, float]]:
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
            if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
                result[joint.name] = (lower, upper)
        if not result:
            raise ModelError("内置 G2 URDF 中没有可驱动关节")
        return result

    def update_joints(self, values: Any) -> None:
        if not isinstance(values, dict) or not values:
            raise ModelError("joints 必须是非空对象")
        validated: dict[str, float] = {}
        for name, raw in values.items():
            if name not in self._limits:
                raise ModelError(f"未知或不可驱动关节: {name}")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ModelError(f"关节 {name} 必须是有限数值")
            value = float(raw)
            if not math.isfinite(value):
                raise ModelError(f"关节 {name} 必须是有限数值")
            lower, upper = self._limits[name]
            # Telemetry can exceed a visual URDF limit by a tiny calibration or
            # encoder tolerance.  Large mismatches still fail closed.
            tolerance = 1e-3
            if value < lower - tolerance or value > upper + tolerance:
                raise ModelError(f"关节 {name}={value} 越界 [{lower}, {upper}]")
            validated[name] = min(upper, max(lower, value))
        with self._lock:
            self._values.update(validated)
            self.robot.update_cfg(validated)

    def linked_geometry_records(
        self,
    ) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]]:
        records: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]] = []
        with self._lock:
            for node in self.robot.scene.graph.nodes_geometry:
                link_name = str(self.robot.scene.graph.transforms.parents[node])
                local, geometry_name = self.robot.scene.graph.get(node, link_name)
                mesh = self.robot.scene.geometry[geometry_name]
                records.append(
                    (
                        str(node),
                        link_name,
                        np.asarray(mesh.vertices, dtype=np.float32),
                        np.asarray(mesh.faces, dtype=np.int32),
                        np.asarray(local),
                        mesh_material_color(mesh),
                    )
                )
        return records

    @staticmethod
    def _render_lod_reduction(link_name: str) -> float:
        if link_name.startswith("gripper"):
            return RENDER_LOD_REDUCTION["gripper"]
        if link_name.startswith("arm"):
            return RENDER_LOD_REDUCTION["arm"]
        return RENDER_LOD_REDUCTION["body"]

    @staticmethod
    def _weld_render_vertices(
        vertices: np.ndarray, faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weld DAE face-corner duplicates before quadric simplification.

        The vendor Collada meshes repeat positions for many face corners.  A
        simplifier otherwise sees thousands of disconnected triangles and can
        leave visible holes.  Eight-decimal welding is far below the source
        asset's physical precision and keeps the original coordinates.
        """

        rounded = np.round(vertices, decimals=8)
        _, first_indices, inverse = np.unique(
            rounded, axis=0, return_index=True, return_inverse=True
        )
        welded_vertices = vertices[first_indices]
        welded_faces = inverse[faces]
        nondegenerate = (
            (welded_faces[:, 0] != welded_faces[:, 1])
            & (welded_faces[:, 1] != welded_faces[:, 2])
            & (welded_faces[:, 0] != welded_faces[:, 2])
        )
        return welded_vertices, welded_faces[nondegenerate]

    def render_geometry_records(
        self,
    ) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]]:
        """Return a display-only LOD while keeping the source/safety meshes intact.

        The visual DAE bundle contains nearly 600k triangles.  Sending all of
        them through MeshCat for every new browser tab dominates viewer startup
        time.  Quadric decimation is performed once when the isolated renderer
        starts; it never changes the URDF used for FK and is not reachable from
        the robot control or collision-safety paths.
        """

        with self._lock:
            if self._render_records is not None:
                return self._render_records
            records = self.linked_geometry_records()
            result: list[
                tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]
            ] = []
            for node, link, vertices, faces, local, color in records:
                render_vertices = vertices
                render_faces = faces
                if len(faces) >= RENDER_LOD_MIN_FACES:
                    welded_vertices, welded_faces = self._weld_render_vertices(
                        vertices, faces
                    )
                    render_vertices, render_faces = simplify(
                        welded_vertices,
                        welded_faces,
                        target_reduction=self._render_lod_reduction(link),
                        agg=5.0,
                        preserve_border=True,
                    )
                result.append(
                    (
                        node,
                        link,
                        np.asarray(render_vertices, dtype=np.float32),
                        np.asarray(render_faces, dtype=np.int32),
                        local,
                        color,
                    )
                )
            self._render_records = result
            return result

    def link_transforms(self, names: set[str]) -> dict[str, np.ndarray]:
        with self._lock:
            return {name: np.asarray(self.robot.scene.graph.get(name)[0]) for name in names}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            render_records = self.render_geometry_records()
            visual_mesh_count = sum(
                1
                for link in self.robot.robot.links
                for visual in link.visuals
                if visual.geometry.mesh is not None
            )
            return {
                "name": self.robot.robot.name,
                "base_link": self.robot.base_link,
                "degrees_of_freedom": self.robot.num_dofs,
                "actuated_joint_count": self.robot.num_actuated_joints,
                "controllable_joint_count": len(self._limits),
                "geometry_count": len(self.robot.scene.geometry),
                "visual_geometry_count": len(self.robot.scene.geometry),
                "visual_mesh_count": visual_mesh_count,
                "vertex_count": sum(
                    len(geometry.vertices) for geometry in self.robot.scene.geometry.values()
                ),
                "face_count": sum(
                    len(geometry.faces) for geometry in self.robot.scene.geometry.values()
                ),
                "render_vertex_count": sum(len(item[2]) for item in render_records),
                "render_face_count": sum(len(item[3]) for item in render_records),
                "render_lod": "display-only",
                "source": "Nexus bundled asset",
            }
