#!/usr/bin/env python3
"""Write exact horizontal-plane support-vertex indices into a safety profile.

This is an offline build tool. The robot runtime only consumes the generated
integer indices and does not depend on SciPy.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def read_stl_vertices(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    vertices: np.ndarray
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + triangle_count * 50 == len(payload):
            dtype = np.dtype(
                [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
            )
            triangles = np.frombuffer(payload, dtype=dtype, count=triangle_count, offset=84)
            vertices = np.asarray(triangles["vertices"], dtype=np.float64).reshape(-1, 3)
        else:
            vertices = read_ascii_stl_vertices(payload, path)
    else:
        vertices = read_ascii_stl_vertices(payload, path)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f"STL has no finite vertices: {path}")
    return np.unique(vertices, axis=0)


def read_ascii_stl_vertices(payload: bytes, path: Path) -> np.ndarray:
    values = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            values.append(tuple(float(value) for value in fields[1:]))
    if not values:
        raise ValueError(f"Cannot parse STL vertices: {path}")
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    profile = args.profile.expanduser().resolve()
    data = json.loads(profile.read_text(encoding="utf-8"))
    total_vertices = 0
    total_support = 0
    for mesh in data.get("collision_meshes", ()):
        mesh_path = Path(mesh["path"]).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = (profile.parent / mesh_path).resolve()
        vertices = read_stl_vertices(mesh_path)
        indices = np.sort(np.unique(ConvexHull(vertices).vertices)).astype(int)
        mesh["support_vertex_indices"] = indices.tolist()
        total_vertices += len(vertices)
        total_support += len(indices)
        print(f"{mesh.get('name', mesh_path.stem)}: {len(vertices)} -> {len(indices)}")
    if total_support == 0:
        raise ValueError("Profile contains no collision meshes")
    profile.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"updated {profile}: {total_vertices} -> {total_support} vertices "
        f"({total_vertices / total_support:.1f}x reduction)"
    )


if __name__ == "__main__":
    main()
