#!/usr/bin/env python3
"""Build the cacheable browser-only G2 display model.

The output is a standards-compliant GLB containing the display LOD and an
``asset.extras.nexus`` kinematics description.  It is never used by policy,
safety, or robot-control code.  The runtime serves the reproducible gzip file
with an immutable cache policy; the browser performs FK locally from read-only
telemetry.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from nexus_deploy.viewer.model import G2Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    PROJECT_ROOT
    / "src"
    / "nexus_deploy"
    / "viewer"
    / "assets"
    / "g2"
    / "G2_dahuan"
    / "G2_dahuan.urdf"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "src" / "nexus_deploy" / "viewer" / "assets" / "g2-dahuan-display-v1.glb.gz"
)
VISUAL_JOINT_NAMES = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx31_gripper_l_inner_joint1",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx71_gripper_r_inner_joint1",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
)


def _matrix(value: Any) -> list[float]:
    matrix = np.asarray(value if value is not None else np.eye(4), dtype=np.float64)
    return matrix.T.reshape(-1).tolist()


def _pad4(payload: bytes, fill: bytes = b"\0") -> bytes:
    return payload + fill * ((-len(payload)) % 4)


class GlbBuilder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def accessor(
        self,
        values: np.ndarray,
        *,
        component_type: int,
        value_type: str,
        minimum: list[float] | None = None,
        maximum: list[float] | None = None,
        target: int,
    ) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        data = values.tobytes(order="C")
        self.binary.extend(data)
        view_index = len(self.buffer_views)
        self.buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(data),
                "target": target,
            }
        )
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "byteOffset": 0,
            "componentType": component_type,
            "count": len(values),
            "type": value_type,
        }
        if minimum is not None:
            accessor["min"] = minimum
        if maximum is not None:
            accessor["max"] = maximum
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index


def build_glb(urdf_path: Path) -> bytes:
    model = G2Model(urdf_path)
    records = model.render_geometry_records()
    builder = GlbBuilder()

    colors = sorted({int(record[-1]) for record in records})
    material_index = {color: index for index, color in enumerate(colors)}
    materials = []
    for color in colors:
        materials.append(
            {
                "name": f"nexus-{color:06x}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [
                        ((color >> 16) & 255) / 255,
                        ((color >> 8) & 255) / 255,
                        (color & 255) / 255,
                        1.0,
                    ],
                    "metallicFactor": 0.08,
                    "roughnessFactor": 0.56,
                },
            }
        )

    meshes: list[dict[str, Any]] = []
    geometry_nodes: list[tuple[str, str, list[float], int]] = []
    for node_name, link_name, vertices, faces, local, color in records:
        positions = np.ascontiguousarray(vertices, dtype="<f4")
        indices = np.ascontiguousarray(faces.reshape(-1), dtype="<u4")
        position_accessor = builder.accessor(
            positions,
            component_type=5126,
            value_type="VEC3",
            minimum=positions.min(axis=0).astype(float).tolist(),
            maximum=positions.max(axis=0).astype(float).tolist(),
            target=34962,
        )
        index_accessor = builder.accessor(
            indices,
            component_type=5125,
            value_type="SCALAR",
            target=34963,
        )
        mesh_index = len(meshes)
        meshes.append(
            {
                "name": str(node_name),
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor},
                        "indices": index_accessor,
                        "material": material_index[int(color)],
                        "mode": 4,
                    }
                ],
            }
        )
        geometry_nodes.append((str(node_name), str(link_name), _matrix(local), mesh_index))

    links = [str(link.name) for link in model.robot.robot.links]
    nodes: list[dict[str, Any]] = [{"name": name, "children": []} for name in links]
    link_index = {name: index for index, name in enumerate(links)}
    joint_payload: list[dict[str, Any]] = []
    for joint in model.robot.robot.joints:
        origin = _matrix(joint.origin)
        nodes[link_index[str(joint.child)]]["matrix"] = origin
        nodes[link_index[str(joint.parent)]]["children"].append(link_index[str(joint.child)])
        mimic = None
        if joint.mimic is not None:
            mimic = {
                "joint": str(joint.mimic.joint),
                "multiplier": float(joint.mimic.multiplier or 1.0),
                "offset": float(joint.mimic.offset or 0.0),
            }
        joint_payload.append(
            {
                "name": str(joint.name),
                "parent": str(joint.parent),
                "child": str(joint.child),
                "type": str(joint.type),
                "axis": np.asarray(joint.axis, dtype=float).tolist(),
                "origin": origin,
                "mimic": mimic,
            }
        )

    for node_name, link_name, local, mesh_index in geometry_nodes:
        geometry_index = len(nodes)
        nodes.append({"name": node_name, "matrix": local, "mesh": mesh_index})
        nodes[link_index[link_name]]["children"].append(geometry_index)

    summary = model.summary()
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Nexus G2 browser model builder",
            "extras": {
                "nexus": {
                    "schema_version": 1,
                    "robot": "g2a",
                    "base_link": model.robot.base_link,
                    "visual_joint_names": list(VISUAL_JOINT_NAMES),
                    "gripper_aperture_indices": [7, 15],
                    "gripper_mapping": {
                        "dahuan": "urdf_position=max(0,min(0.025,0.025-gdk_state))",
                        "omnipicker_legacy": "urdf_position=-clamp(gdk_state,-pi/4,0)*0.025/(pi/4)",
                    },
                    "joints": joint_payload,
                    "source_vertex_count": summary["vertex_count"],
                    "source_face_count": summary["face_count"],
                    "render_vertex_count": summary["render_vertex_count"],
                    "render_face_count": summary["render_face_count"],
                    "display_only": True,
                    "robot_commands": "disabled",
                }
            },
        },
        "scene": 0,
        "scenes": [{"name": "Nexus G2", "nodes": [link_index[model.robot.base_link]]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": builder.accessors,
        "bufferViews": builder.buffer_views,
        "buffers": [{"byteLength": len(builder.binary)}],
    }
    json_chunk = _pad4(
        json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        b" ",
    )
    binary_chunk = _pad4(bytes(builder.binary))
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary_chunk), b"BIN\0"),
            binary_chunk,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    glb = build_glb(args.urdf.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(glb, compresslevel=9, mtime=0))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "glb_bytes": len(glb),
                "gzip_bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
