#!/usr/bin/env python3
"""Build a G2 t2/crsB URDF with the platform's modified Dahuan terminals."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ARM_TIPS = ("arm_l_link7", "arm_r_link7")
CUSTOM_LINKS = {
    "gripper_l_base_link",
    "gripper_l_inner_link1",
    "gripper_l_outer_link1",
    "gripper_r_base_link",
    "gripper_r_inner_link1",
    "gripper_r_outer_link1",
}
CUSTOM_JOINTS = {
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

T2_DISPLAY_LINKS = {
    "base_link",
    "body_link1",
    "body_link2",
    "body_link3",
    "body_link4",
    "body_link5",
    "head_link1",
    "head_link2",
    "head_link3",
    "chassis_lwheel_front_link1",
    "chassis_lwheel_front_link2",
    "chassis_lwheel_rear_link1",
    "chassis_lwheel_rear_link2",
    "chassis_rwheel_front_link1",
    "chassis_rwheel_front_link2",
    "chassis_rwheel_rear_link1",
    "chassis_rwheel_rear_link2",
}
ARM_DISPLAY_MATERIALS = {
    1: "black",
    2: "white",
    3: "white",
    4: "white",
    5: "white",
    6: "black",
    7: "black",
}

# The vendor terminal was authored against the legacy CRS arm_link7 mesh.  Its
# flange face ends at x=95 mm and the vendor mount is x=295 mm.  Factory crsB's
# flange face ends at x=90 mm, so preserve the same 200 mm flange-to-tool-frame
# offset by mounting the terminal at x=290 mm.  Keeping the vendor's 295 mm
# value on crsB creates a real 3.3--3.7 mm visual gap and shifts every terminal
# collision point 5 mm away from the wrist.
CRSB_TERMINAL_MOUNT_X_M = 0.290

# The physical Dahuan grippers and GDK state/command contract use 25 mm of
# travel.  The vendor source says 30 mm; retain that file as provenance but
# correct every copied prismatic joint in the controlled hybrid.
DHAUAN_GRIPPER_TRAVEL_M = 0.025

# The factory crsB meshes leave a roughly 5 mm visual-only seam between the
# T2 shoulder cover and arm link 1.  Do not move the calibrated shoulder joint;
# fill only the omitted cosmetic collar geometry.
SHOULDER_COLLAR_CENTER_Z_M = 0.09835
SHOULDER_COLLAR_LENGTH_M = 0.0054
SHOULDER_COLLAR_RADIUS_M = 0.052


def _terminal_descendants(root: ET.Element) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = str(joint.find("parent").attrib["link"])
        child = str(joint.find("child").attrib["link"])
        children.setdefault(parent, []).append(child)

    removed: set[str] = set()
    pending = [child for tip in ARM_TIPS for child in children.get(tip, [])]
    while pending:
        link = pending.pop()
        if link in removed:
            continue
        removed.add(link)
        pending.extend(children.get(link, []))
    return removed


def _remove_factory_terminals(root: ET.Element) -> None:
    removed = _terminal_descendants(root)
    for link in list(root.findall("link")):
        if link.attrib["name"] in removed:
            root.remove(link)
    for joint in list(root.findall("joint")):
        parent = str(joint.find("parent").attrib["link"])
        child = str(joint.find("child").attrib["link"])
        if parent in removed or child in removed:
            root.remove(joint)


def _rewrite_factory_package_paths(root: ET.Element) -> None:
    prefix = "package://genie_robot_description/"
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.attrib.get("filename", ""))
        if filename.startswith(prefix):
            mesh.attrib["filename"] = filename[len(prefix) :]


def _replace_visual_material(visual: ET.Element, name: str) -> None:
    for material in list(visual.findall("material")):
        visual.remove(material)
    visual.append(ET.Element("material", name=name))


def _restore_display_materials(root: ET.Element) -> None:
    """Restore verified T2 material groups without changing collision/FK geometry."""
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for name in T2_DISPLAY_LINKS:
        visual = links[name].find("visual")
        mesh = visual.find("geometry/mesh") if visual is not None else None
        if visual is None or mesh is None:
            raise ValueError(f"factory T2 link has no display mesh: {name}")
        mesh.attrib["filename"] = f"meshes/body/{name}.dae"
        # Collada contains its own black/white/orange sub-materials. An outer
        # URDF material would flatten those regions back to one color.
        for material in list(visual.findall("material")):
            visual.remove(material)

    for side in ("l", "r"):
        for index, material in ARM_DISPLAY_MATERIALS.items():
            visual = links[f"arm_{side}_link{index}"].find("visual")
            if visual is None:
                raise ValueError(f"factory crsB arm link has no visual: {side}{index}")
            _replace_visual_material(visual, material)


def _style_custom_terminals(root: ET.Element) -> None:
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for side in ("l", "r"):
        _replace_visual_material(links[f"gripper_{side}_base_link"].find("visual"), "grey")
        _replace_visual_material(links[f"gripper_{side}_inner_link1"].find("visual"), "silver")
        _replace_visual_material(links[f"gripper_{side}_outer_link1"].find("visual"), "silver")


def _copy_custom_terminals(target: ET.Element, source: ET.Element) -> None:
    links = {link.attrib["name"]: link for link in source.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in source.findall("joint")}
    missing = (CUSTOM_LINKS - links.keys()) | (CUSTOM_JOINTS - joints.keys())
    if missing:
        raise ValueError(f"custom Dahuan URDF is missing: {sorted(missing)}")

    for name in sorted(CUSTOM_LINKS):
        target.append(copy.deepcopy(links[name]))
    for name in sorted(CUSTOM_JOINTS):
        joint = copy.deepcopy(joints[name])
        if name in {
            "arm_l_link7_to_gripper_l_base_link",
            "arm_r_link7_to_gripper_r_base_link",
        }:
            joint.find("origin").attrib["xyz"] = (
                f"{CRSB_TERMINAL_MOUNT_X_M:.3f} 0 0"
            )
        if name in {
            "idx31_gripper_l_inner_joint1",
            "idx41_gripper_l_outer_joint1",
            "idx71_gripper_r_inner_joint1",
            "idx81_gripper_r_outer_joint1",
        }:
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"custom Dahuan joint has no limit: {name}")
            limit.attrib["upper"] = f"{DHAUAN_GRIPPER_TRAVEL_M:.3f}"
        leader = OUTER_MIMIC.get(name)
        if leader is not None:
            existing = joint.find("mimic")
            if existing is not None:
                joint.remove(existing)
            joint.append(
                ET.Element("mimic", joint=leader, multiplier="1.0", offset="0.0")
            )
        target.append(joint)


def _add_shoulder_visual_collars(root: ET.Element) -> None:
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for name in ("arm_l_link1", "arm_r_link1"):
        link = links[name]
        visual = ET.Element("visual", name="nexus_shoulder_adapter_visual")
        ET.SubElement(
            visual,
            "origin",
            xyz=f"0 0 {SHOULDER_COLLAR_CENTER_Z_M:.5f}",
            rpy="0 0 0",
        )
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(
            geometry,
            "cylinder",
            radius=f"{SHOULDER_COLLAR_RADIUS_M:.3f}",
            length=f"{SHOULDER_COLLAR_LENGTH_M:.4f}",
        )
        ET.SubElement(visual, "material", name="grey")
        link.append(visual)


def _validate_tree(root: ET.Element) -> None:
    links = [link.attrib["name"] for link in root.findall("link")]
    if len(links) != len(set(links)):
        raise ValueError("duplicate link names")
    known = set(links)
    children: set[str] = set()
    for joint in root.findall("joint"):
        parent = str(joint.find("parent").attrib["link"])
        child = str(joint.find("child").attrib["link"])
        if parent not in known or child not in known:
            raise ValueError(f"dangling joint {joint.attrib['name']}: {parent} -> {child}")
        if child in children:
            raise ValueError(f"link has multiple parents: {child}")
        children.add(child)
    roots = known - children
    if roots != {"base_link"}:
        raise ValueError(f"expected only base_link as root, got {sorted(roots)}")


def build(factory_crsb_path: Path, custom_dahuan_path: Path) -> ET.ElementTree:
    factory = ET.parse(factory_crsb_path).getroot()
    custom = ET.parse(custom_dahuan_path).getroot()
    _remove_factory_terminals(factory)
    _rewrite_factory_package_paths(factory)
    _restore_display_materials(factory)
    _copy_custom_terminals(factory, custom)
    _style_custom_terminals(factory)
    _add_shoulder_visual_collars(factory)
    factory.attrib["name"] = "G2_dahuan"
    factory.insert(
        0,
        ET.Comment(
            " Nexus controlled hybrid: factory G2 t2/crsB body and arms, "
            "plus the platform-modified Dahuan terminal subtrees. "
        ),
    )
    _validate_tree(factory)
    ET.indent(factory, space="  ")
    return ET.ElementTree(factory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-crsb-urdf", type=Path, required=True)
    parser.add_argument("--custom-dahuan-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.factory_crsb_urdf.resolve(), args.custom_dahuan_urdf.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
