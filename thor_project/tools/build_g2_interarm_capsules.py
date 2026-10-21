#!/usr/bin/env python3
"""Build a conservative two-stage inter-arm capsule model from the canonical URDF.

The existing one-capsule-per-link model is retained as a broad phase.  Complex
Dahuan terminal meshes are decomposed into several triangle clusters; every
triangle is assigned to one cluster and every cluster capsule encloses all
three vertices of each assigned triangle.  This keeps the narrow phase
conservative without the very large radius required by one terminal capsule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


TERMINAL_FRAMES = {
    "gripper_l_base_link",
    "gripper_l_inner_link1",
    "gripper_l_outer_link1",
    "gripper_r_base_link",
    "gripper_r_inner_link1",
    "gripper_r_outer_link1",
}
CAPSULE_PADDING_M = 0.0005


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_binary_stl_triangles(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"STL is not binary or is truncated: {path}")
    count = struct.unpack_from("<I", payload, 80)[0]
    if 84 + count * 50 != len(payload):
        raise ValueError(f"Expected a binary STL: {path}")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    records = np.frombuffer(payload, dtype=dtype, count=count, offset=84)
    return np.asarray(records["vertices"], dtype=np.float64)


def _origin_transform(origin: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz = np.asarray([float(v) for v in origin.attrib.get("xyz", "0 0 0").split()])
    roll, pitch, yaw = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
    rx = np.asarray(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    )
    ry = np.asarray(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )
    rz = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    return xyz, rz @ ry @ rx


def _terminal_triangles(urdf_path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    result: dict[str, np.ndarray] = {}
    for link in root.findall("link"):
        frame = link.attrib["name"]
        if frame not in TERMINAL_FRAMES:
            continue
        collision = link.find("collision")
        mesh = None if collision is None else collision.find("geometry/mesh")
        if collision is None or mesh is None:
            raise ValueError(f"Terminal frame has no collision mesh: {frame}")
        path = (urdf_path.parent / mesh.attrib["filename"]).resolve(strict=True)
        scale = np.asarray([float(v) for v in mesh.attrib.get("scale", "1 1 1").split()])
        triangles = _read_binary_stl_triangles(path) * scale
        translation, rotation = _origin_transform(collision.find("origin"))
        result[frame] = triangles @ rotation.T + translation
    if result.keys() != TERMINAL_FRAMES:
        raise ValueError(f"Missing terminal meshes: {sorted(TERMINAL_FRAMES - result.keys())}")
    return result


def _connected_triangle_components(triangles: np.ndarray) -> list[np.ndarray]:
    """Return exact shared-vertex components without a SciPy dependency."""
    parent = np.arange(len(triangles), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    owner: dict[tuple[float, float, float], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        for vertex in triangle:
            key = tuple(np.round(vertex, 9))
            previous = owner.setdefault(key, triangle_index)
            union(triangle_index, previous)
    groups: dict[int, list[int]] = {}
    for triangle_index in range(len(triangles)):
        groups.setdefault(find(triangle_index), []).append(triangle_index)
    return [triangles[indices] for indices in groups.values()]


def _fit_capsule(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = triangles.reshape(-1, 3)
    center = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
    axis = vh[0]
    projection = (points - center) @ axis
    start = center + axis * projection.min()
    end = center + axis * projection.max()
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-16:
        radius = float(np.linalg.norm(points - start, axis=1).max())
    else:
        parameter = np.clip((points - start) @ segment / denominator, 0.0, 1.0)
        closest = start + parameter[:, None] * segment
        radius = float(np.linalg.norm(points - closest, axis=1).max())
    return start, end, radius + CAPSULE_PADDING_M


def _kmeans_component_groups(components: list[np.ndarray], count: int) -> list[np.ndarray]:
    centers = np.asarray([group.reshape(-1, 3).mean(axis=0) for group in components])
    seeds = [centers[0]]
    while len(seeds) < count:
        distances = np.linalg.norm(centers[:, None, :] - np.asarray(seeds)[None, :, :], axis=2)
        seeds.append(centers[int(np.argmax(distances.min(axis=1)))])
    cluster_centers = np.asarray(seeds)
    labels = np.zeros(len(components), dtype=np.int64)
    for _ in range(50):
        distances = np.linalg.norm(
            centers[:, None, :] - cluster_centers[None, :, :], axis=2
        )
        next_labels = distances.argmin(axis=1)
        next_centers = np.asarray(
            [centers[next_labels == index].mean(axis=0) for index in range(count)]
        )
        labels = next_labels
        if np.allclose(next_centers, cluster_centers, rtol=0.0, atol=1e-12):
            break
        cluster_centers = next_centers
    return [
        np.concatenate([group for group, label in zip(components, labels, strict=True) if label == index])
        for index in range(count)
    ]


def _base_groups(triangles: np.ndarray) -> list[tuple[str, np.ndarray]]:
    components = sorted(_connected_triangle_components(triangles), key=len, reverse=True)
    if len(components) < 7:
        raise ValueError("Expected the custom base mesh to contain its separate hardware bodies")
    major = [(f"body_{index}", group) for index, group in enumerate(components[:3])]
    hardware = [
        (f"hardware_{index}", group)
        for index, group in enumerate(_kmeans_component_groups(components[3:], 4))
    ]
    return major + hardware


def _finger_groups(triangles: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Split the L-shaped custom finger into stem and two jaw capsules."""
    centroids = triangles.mean(axis=1)
    low, high = np.quantile(centroids[:, 1], [0.05, 0.45])
    best: tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    for threshold in np.linspace(low, high, 33):
        stem = triangles[centroids[:, 1] < threshold]
        jaw = triangles[centroids[:, 1] >= threshold]
        if min(len(stem), len(jaw)) < 100:
            continue
        jaw_centers = jaw.mean(axis=1)
        center = jaw_centers.mean(axis=0)
        _u, _s, vh = np.linalg.svd(jaw_centers - center, full_matrices=False)
        projection = (jaw_centers - center) @ vh[0]
        for quantile in (0.3, 0.4, 0.5, 0.6, 0.7):
            split = np.quantile(projection, quantile)
            first, second = jaw[projection <= split], jaw[projection > split]
            if min(len(first), len(second)) < 100:
                continue
            radii = [_fit_capsule(group)[2] for group in (stem, first, second)]
            score = max(radii) + 0.1 * sum(radii)
            if best is None or score < best[0]:
                best = score, (stem, first, second)
    if best is None:
        raise ValueError("Could not decompose custom finger mesh")
    names = ("stem", "jaw_0", "jaw_1")
    return list(zip(names, best[1], strict=True))


def _capsule(name: str, frame: str, triangles: np.ndarray) -> dict[str, object]:
    start, end, radius = _fit_capsule(triangles)
    return {
        "name": name,
        "frame": frame,
        "start": [round(float(value), 9) for value in start],
        "end": [round(float(value), 9) for value in end],
        "radius": round(radius, 9),
    }


def build(urdf_path: Path, existing: dict[str, object]) -> dict[str, object]:
    triangles = _terminal_triangles(urdf_path)
    broad_left = existing.get("broad_phase_left_capsules", existing["left_capsules"])
    broad_right = existing.get("broad_phase_right_capsules", existing["right_capsules"])
    broad = {"left": broad_left, "right": broad_right}
    fine: dict[str, list[dict[str, object]]] = {"left": [], "right": []}
    for side in ("left", "right"):
        # Arm capsules were already tight, link-local and independently audited.
        fine[side].extend(
            item for item in broad[side] if "gripper" not in str(item["name"])
        )
        prefix = "l" if side == "left" else "r"
        base_frame = f"gripper_{prefix}_base_link"
        for suffix, group in _base_groups(triangles[base_frame]):
            fine[side].append(
                _capsule(f"{side}_gripper_base_{suffix}", base_frame, group)
            )
        for part in ("inner", "outer"):
            frame = f"gripper_{prefix}_{part}_link1"
            for suffix, group in _finger_groups(triangles[frame]):
                fine[side].append(
                    _capsule(f"{side}_gripper_{part}_{suffix}", frame, group)
                )
    return {
        "schema_version": 2,
        "geometry": "cross-chain-capsules-v2",
        "validation_status": "read-only-candidate",
        "hard_margin": float(existing.get("hard_margin", 0.008)),
        "warning_margin": float(existing.get("warning_margin", 0.020)),
        "broad_phase_activation_margin": 0.030,
        "broad_phase_left_capsules": broad_left,
        "broad_phase_right_capsules": broad_right,
        "left_capsules": fine["left"],
        "right_capsules": fine["right"],
        "canonical_urdf": urdf_path.name,
        "canonical_urdf_sha256": _sha256(urdf_path),
        "capsule_padding_m": CAPSULE_PADDING_M,
        "coverage": "every STL triangle assigned to one enclosing narrow-phase capsule",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    urdf = args.urdf.expanduser().resolve(strict=True)
    overlay = args.overlay.expanduser().resolve(strict=True)
    existing = json.loads(overlay.read_text(encoding="utf-8"))
    result = build(urdf, existing)
    overlay.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"overlay={overlay}")
    print(f"left_capsules={len(result['left_capsules'])}")
    print(f"right_capsules={len(result['right_capsules'])}")


if __name__ == "__main__":
    main()
