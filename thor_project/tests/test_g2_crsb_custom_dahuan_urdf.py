from __future__ import annotations

import math
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(
    str(PROJECT_ROOT / "tools" / "build_g2_crsb_custom_dahuan_urdf.py")
)["build"]
FACTORY = (
    PROJECT_ROOT
    / "src/nexus_deploy/viewer/assets/g2/G2_dahuan"
    / "G2_t2_crsB_factory.urdf"
)
CUSTOM = (
    PROJECT_ROOT
    / "src/nexus_deploy/viewer/assets/g2/G2_dahuan/G2_dahuan_vendor.urdf"
)


def _joint(root: ET.Element, name: str) -> ET.Element:
    return next(joint for joint in root.findall("joint") if joint.attrib["name"] == name)


def _pose(element: ET.Element) -> np.ndarray:
    xyz = np.asarray([float(value) for value in element.attrib.get("xyz", "0 0 0").split()])
    roll, pitch, yaw = [
        float(value) for value in element.attrib.get("rpy", "0 0 0").split()
    ]
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ]
    )
    ry = np.asarray(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    rz = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


def test_build_uses_crsb_kinematics_and_custom_terminal() -> None:
    root = BUILD(FACTORY, CUSTOM).getroot()

    assert root.attrib["name"] == "G2_dahuan"
    assert _joint(root, "idx24_arm_l_joint4").find("origin").attrib["xyz"] == "0.0105 0 0.332"
    assert _joint(root, "idx64_arm_r_joint4").find("origin").attrib["xyz"] == "0.0105 0 0.332"
    assert _joint(root, "idx26_arm_l_joint6").find("origin").attrib["xyz"] == "0 0 0.278"
    assert _joint(root, "idx66_arm_r_joint6").find("origin").attrib["xyz"] == "0 0 0.278"

    base_visual = next(link for link in root.findall("link") if link.attrib["name"] == "base_link").find("visual")
    assert base_visual.find("geometry/mesh").attrib["filename"] == "meshes/body/base_link.dae"
    assert base_visual.find("material") is None
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for side in ("l", "r"):
        expected = ("black", "white", "white", "white", "white", "black", "black")
        for index, material in enumerate(expected, start=1):
            assert links[f"arm_{side}_link{index}"].find("visual/material").attrib["name"] == material
        assert links[f"gripper_{side}_base_link"].find("visual/material").attrib["name"] == "grey"
        assert links[f"gripper_{side}_inner_link1"].find("visual/material").attrib["name"] == "silver"
        assert links[f"gripper_{side}_outer_link1"].find("visual/material").attrib["name"] == "silver"

    left_mount = _joint(root, "arm_l_link7_to_gripper_l_base_link")
    right_mount = _joint(root, "arm_r_link7_to_gripper_r_base_link")
    assert left_mount.find("origin").attrib["xyz"] == "0.290 0 0"
    assert right_mount.find("origin").attrib["xyz"] == "0.290 0 0"
    assert not any(joint.attrib["name"] == "arm_l_end_joint" for joint in root.findall("joint"))
    assert not any(joint.attrib["name"] == "arm_r_end_joint" for joint in root.findall("joint"))

    for name in (
        "idx31_gripper_l_inner_joint1",
        "idx41_gripper_l_outer_joint1",
        "idx71_gripper_r_inner_joint1",
        "idx81_gripper_r_outer_joint1",
    ):
        assert _joint(root, name).find("limit").attrib["upper"] == "0.025"

    links = {link.attrib["name"] for link in root.findall("link")}
    children = {joint.find("child").attrib["link"] for joint in root.findall("joint")}
    assert links - children == {"base_link"}

    for name in ("arm_l_link1", "arm_r_link1"):
        link = next(link for link in root.findall("link") if link.attrib["name"] == name)
        collar = next(
            visual
            for visual in link.findall("visual")
            if visual.attrib.get("name") == "nexus_shoulder_adapter_visual"
        )
        assert collar.find("origin").attrib["xyz"] == "0 0 0.09835"
        assert collar.find("geometry/cylinder").attrib == {
            "radius": "0.052",
            "length": "0.0054",
        }


def test_custom_finger_meshes_are_authored_in_gripper_base_at_open_zero() -> None:
    """Do not mistake the side-specific CAD origins for an assembly offset.

    The imported workpiece STLs already contain each complete custom finger in
    gripper-base coordinates.  At q=0 the prismatic joint origin and collision
    mesh origin therefore cancel.  Moving either origin independently would
    introduce a real, side-dependent collision error.
    """

    root = BUILD(FACTORY, CUSTOM).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for joint_name in (
        "idx31_gripper_l_inner_joint1",
        "idx41_gripper_l_outer_joint1",
        "idx71_gripper_r_inner_joint1",
        "idx81_gripper_r_outer_joint1",
    ):
        joint = _joint(root, joint_name)
        child = joint.find("child").attrib["link"]
        collision_origin = links[child].find("collision/origin")
        base_from_mesh = _pose(joint.find("origin")) @ _pose(collision_origin)
        np.testing.assert_allclose(base_from_mesh, np.eye(4), rtol=0.0, atol=1e-7)
