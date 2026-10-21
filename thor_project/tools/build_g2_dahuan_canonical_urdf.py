#!/usr/bin/env python3
"""Build the canonical G2 Dahuan visual and kinematic URDF.

The validated legacy G2 base, body, head, wheel, and arm tree is retained,
including its Collada sub-mesh materials.  The OmniPicker adapter, cameras, and
grippers are replaced at each ``arm_*_link7`` by the supplied Dahuan terminal
subtree.  This output is the single platform authority from which display and
safety derivatives are built.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

TERMINAL_ROOTS = ("gripper_l_base_link", "gripper_r_base_link")
LEGACY_TERMINAL_LINKS = {
    "arm_l_end_link",
    "arm_r_end_link",
    "gripper_l_camera_link",
    "gripper_r_camera_link",
}
DAHUAN_LINKS = {
    "gripper_l_base_link",
    "gripper_l_inner_link1",
    "gripper_l_outer_link1",
    "gripper_r_base_link",
    "gripper_r_inner_link1",
    "gripper_r_outer_link1",
}
DAHUAN_JOINTS = {
    "arm_l_link7_to_gripper_l_base_link",
    "idx31_gripper_l_inner_joint1",
    "idx41_gripper_l_outer_joint1",
    "arm_r_link7_to_gripper_r_base_link",
    "idx71_gripper_r_inner_joint1",
    "idx81_gripper_r_outer_joint1",
}
OUTER_MIMIC = {
    "idx41_gripper_l_outer_joint1": "idx31_gripper_l_inner_joint1",
    "idx81_gripper_r_outer_joint1": "idx71_gripper_r_inner_joint1",
}


def _descendants(root: ET.Element, starts: tuple[str, ...]) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = str(joint.find("parent").attrib["link"])
        child = str(joint.find("child").attrib["link"])
        children.setdefault(parent, []).append(child)
    result = set(starts)
    pending = list(starts)
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _remove_legacy_terminal(root: ET.Element) -> None:
    removed_links = _descendants(root, TERMINAL_ROOTS) | LEGACY_TERMINAL_LINKS
    for link in list(root.findall("link")):
        if link.attrib["name"] in removed_links:
            root.remove(link)
    for joint in list(root.findall("joint")):
        parent = str(joint.find("parent").attrib["link"])
        child = str(joint.find("child").attrib["link"])
        if parent in removed_links or child in removed_links:
            root.remove(joint)


def _rewrite_legacy_mesh_paths(root: ET.Element) -> None:
    """Reuse the platform's frozen legacy visual assets without duplicating them."""
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.attrib.get("filename", ""))
        if filename.startswith("meshes/body/") or filename.startswith("meshes/arm/"):
            mesh.attrib["filename"] = f"../../{filename}"


def _normalize_whitespace(root: ET.Element) -> None:
    """Make output independent of insignificant formatting in vendor inputs."""
    for element in root.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        if element.tail is not None and not element.tail.strip():
            element.tail = None


def _add_gripper_mimic(joint: ET.Element) -> None:
    leader = OUTER_MIMIC.get(str(joint.attrib.get("name", "")))
    if leader is None:
        return
    existing = joint.find("mimic")
    if existing is not None:
        joint.remove(existing)
    joint.append(ET.Element("mimic", joint=leader, multiplier="1.0", offset="0.0"))


def build(legacy_visual_path: Path, dahuan_vendor_path: Path) -> ET.ElementTree:
    legacy = ET.parse(legacy_visual_path).getroot()
    dahuan = ET.parse(dahuan_vendor_path).getroot()
    _remove_legacy_terminal(legacy)
    _rewrite_legacy_mesh_paths(legacy)

    source_links = {link.attrib["name"]: link for link in dahuan.findall("link")}
    source_joints = {joint.attrib["name"]: joint for joint in dahuan.findall("joint")}
    if DAHUAN_LINKS - source_links.keys() or DAHUAN_JOINTS - source_joints.keys():
        raise ValueError("Dahuan vendor URDF is missing the expected terminal subtree")

    for name in sorted(DAHUAN_LINKS):
        legacy.append(copy.deepcopy(source_links[name]))
    for name in sorted(DAHUAN_JOINTS):
        joint = copy.deepcopy(source_joints[name])
        _add_gripper_mimic(joint)
        legacy.append(joint)

    legacy.attrib["name"] = "G2_dahuan"
    legacy.insert(
        0,
        ET.Comment(
            " Canonical Nexus G2 Dahuan: validated legacy base/body/arms with "
            "embedded Collada materials, plus the vendor Dahuan terminal subtree. "
        ),
    )
    _normalize_whitespace(legacy)
    ET.indent(legacy, space="  ")
    return ET.ElementTree(legacy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-visual-urdf", type=Path, required=True)
    parser.add_argument("--dahuan-vendor-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.legacy_visual_urdf.resolve(),
        args.dahuan_vendor_urdf.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
