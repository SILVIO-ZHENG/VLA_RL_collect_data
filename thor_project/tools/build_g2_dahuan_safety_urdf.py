#!/usr/bin/env python3
"""Build the lightweight G2 Dahuan safety URDF from the canonical model.

The canonical display URDF is the only kinematic authority. This derivative
keeps its complete link/joint tree and the Dahuan terminal collision elements,
while removing visual/inertial data and collision meshes that the realtime
safety layer does not sample. Browser FK and robot-client FK therefore cannot
silently drift because of a legacy arm model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

TERMINAL_COLLISION_LINKS = {
    "gripper_l_base_link",
    "gripper_l_inner_link1",
    "gripper_l_outer_link1",
    "gripper_r_base_link",
    "gripper_r_inner_link1",
    "gripper_r_outer_link1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(canonical_path: Path) -> ET.ElementTree:
    root = copy.deepcopy(ET.parse(canonical_path).getroot())
    for link in root.findall("link"):
        for tag in ("visual", "inertial"):
            for element in list(link.findall(tag)):
                link.remove(element)
        if link.attrib["name"] not in TERMINAL_COLLISION_LINKS:
            for element in list(link.findall("collision")):
                link.remove(element)
    root.attrib["name"] = "G2_dahuan_safety"
    root.insert(
        0,
        ET.Comment(
            " Generated from the Nexus canonical G2_dahuan URDF; "
            "all joint origins/axes and fixed transforms are unchanged. "
        ),
    )
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def sync_inter_arm_geometry(*, overlay_path: Path, template_path: Path) -> None:
    """Generate the self-contained client geometry from the browser source."""
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    required = {
        "geometry",
        "validation_status",
        "hard_margin",
        "warning_margin",
        "left_capsules",
        "right_capsules",
    }
    missing = sorted(required - overlay.keys())
    if missing:
        raise ValueError(f"Collision overlay is missing fields: {missing}")
    optional = {
        key: copy.deepcopy(overlay[key])
        for key in (
            "broad_phase_activation_margin",
            "broad_phase_left_capsules",
            "broad_phase_right_capsules",
            "canonical_urdf",
            "canonical_urdf_sha256",
            "capsule_padding_m",
            "coverage",
        )
        if key in overlay
    }
    template["inter_arm_collision"] = {
        "enabled": True,
        **{key: copy.deepcopy(overlay[key]) for key in required},
        **optional,
        "canonical_source": "G2_dahuan/collision_overlay.json",
        "canonical_source_sha256": sha256(overlay_path),
        "notes": (
            "由 Nexus 权威 G2_dahuan 碰撞几何自动生成；仅比较左右运动链，"
            "运行时夹爪几何使用实测开度。"
        ),
    }
    template_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-overlay", type=Path)
    parser.add_argument("--safety-template", type=Path)
    args = parser.parse_args()
    canonical = args.canonical_urdf.resolve(strict=True)
    output = args.output.resolve()
    result = build(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write(output, encoding="utf-8", xml_declaration=True)
    if bool(args.collision_overlay) != bool(args.safety_template):
        parser.error("--collision-overlay and --safety-template must be supplied together")
    if args.collision_overlay and args.safety_template:
        sync_inter_arm_geometry(
            overlay_path=args.collision_overlay.resolve(strict=True),
            template_path=args.safety_template.resolve(strict=True),
        )
    print(f"safety_urdf={output}")
    print(f"safety_urdf_sha256={sha256(output)}")
    print(f"canonical_urdf_sha256={sha256(canonical)}")


if __name__ == "__main__":
    main()
