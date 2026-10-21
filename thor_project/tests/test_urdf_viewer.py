from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = (
    PROJECT_ROOT
    / "src"
    / "nexus_deploy"
    / "robot"
    / "g2a"
    / "assets"
    / "agibot_g2"
)
VISUAL_URDF = VIEWER_ROOT / "G2_omnipicker_visual.urdf"
SAFETY_URDF = VIEWER_ROOT / "G2_omnipicker_safety_geometry.urdf"
SOURCE_URDF = VIEWER_ROOT / "G2_omnipicker_fixed_dual.urdf"


def parse(path: Path) -> ElementTree.Element:
    return ElementTree.parse(path).getroot()


def test_viewer_urdfs_are_portable_and_all_meshes_resolve() -> None:
    for path in (VISUAL_URDF, SAFETY_URDF):
        root = parse(path)
        assert root.tag == "robot"
        assert not root.findall(".//loop_joint")
        assert not root.findall(".//collision")
        assert not root.findall(".//collision_checking")
        meshes = root.findall(".//visual/geometry/mesh")
        assert meshes
        for mesh in meshes:
            filename = mesh.attrib["filename"]
            assert not filename.startswith(("package://", "file://", "/"))
            assert (VIEWER_ROOT / filename).is_file(), filename


def test_viewer_urdf_is_one_connected_tree() -> None:
    root = parse(VISUAL_URDF)
    links = {link.attrib["name"] for link in root.findall("link")}
    children = [joint.find("child").attrib["link"] for joint in root.findall("joint")]
    parents = [joint.find("parent").attrib["link"] for joint in root.findall("joint")]

    assert len(children) == len(set(children))
    assert set(children) <= links
    assert set(parents) <= links
    assert links - set(children) == {"base_link"}


def test_viewer_preserves_every_official_standard_joint_transform_and_limit() -> None:
    source = parse(SOURCE_URDF)
    viewer = parse(VISUAL_URDF)
    source_joints = {joint.attrib["name"]: joint for joint in source.findall("joint")}
    viewer_joints = {joint.attrib["name"]: joint for joint in viewer.findall("joint")}

    assert viewer_joints.keys() == source_joints.keys()
    for name, source_joint in source_joints.items():
        viewer_joint = viewer_joints[name]
        assert viewer_joint.attrib == source_joint.attrib
        for tag in ("origin", "parent", "child", "axis", "limit"):
            source_element = source_joint.find(tag)
            viewer_element = viewer_joint.find(tag)
            assert (viewer_element is None) == (source_element is None), (name, tag)
            if source_element is not None:
                assert viewer_element.attrib == source_element.attrib, (name, tag)


def test_official_gripper_mimics_have_active_inner_joint_as_root() -> None:
    root = parse(VISUAL_URDF)
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    for side, active_index in (("l", 31), ("r", 71)):
        active = f"idx{active_index}_gripper_{side}_inner_joint1"
        assert joints[active].find("mimic") is None
        followers = [
            joint
            for joint in joints.values()
            if joint.find("mimic") is not None
            and joint.find("mimic").attrib["joint"] == active
        ]
        assert len(followers) == 7


def test_safety_viewer_uses_convex_stl_for_every_gripper_visual() -> None:
    root = parse(SAFETY_URDF)
    gripper_links = [
        link for link in root.findall("link") if link.attrib["name"].startswith("gripper_")
    ]
    displayed = 0
    for link in gripper_links:
        meshes = link.findall("./visual/geometry/mesh")
        if not meshes:
            continue
        displayed += 1
        assert all("meshes/gripper/convex/" in mesh.attrib["filename"] for mesh in meshes)
        assert all(mesh.attrib["filename"].endswith(".STL") for mesh in meshes)
    assert displayed == 20
