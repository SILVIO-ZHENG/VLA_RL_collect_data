from __future__ import annotations

import gzip
import hashlib
import json
import re
import runpy
import struct
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import request as urllib_request

import numpy as np
import pytest

from nexus_deploy.control_api import build_parser
from nexus_deploy.viewer.model import G2Model
from nexus_deploy.viewer.runtime import (
    BROWSER_MODEL_URL,
    IMMUTABLE_CACHE,
    MESHCAT_BROWSER_BUNDLE_URL,
    SimulationState,
    _handler_class,
    _validate_canonical_model_assets,
)
from nexus_deploy.viewer.service import SimulatorProcess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = PROJECT_ROOT / "src" / "nexus_deploy" / "viewer"
BUILD_CANONICAL_URDF = runpy.run_path(
    str(PROJECT_ROOT / "tools" / "build_g2_crsb_custom_dahuan_urdf.py")
)["build"]
SAFETY_BUILDER = runpy.run_path(
    str(PROJECT_ROOT / "tools" / "build_g2_dahuan_safety_urdf.py")
)
DAHUAN_VISUAL_URDF = (
    SIMULATOR_ROOT / "assets" / "g2" / "G2_dahuan" / "G2_dahuan.urdf"
)
DAHUAN_VENDOR_URDF = DAHUAN_VISUAL_URDF.with_name("G2_dahuan_vendor.urdf")
LEGACY_VISUAL_URDF = SIMULATOR_ROOT / "assets" / "G2_omnipicker_visual.urdf"
FACTORY_CRSB_URDF = DAHUAN_VISUAL_URDF.with_name("G2_t2_crsB_factory.urdf")


def test_canonical_dahuan_urdf_is_reproducible(tmp_path: Path) -> None:
    output = tmp_path / "G2_dahuan.urdf"
    result = BUILD_CANONICAL_URDF(FACTORY_CRSB_URDF, DAHUAN_VENDOR_URDF)
    result.write(output, encoding="utf-8", xml_declaration=True)

    assert output.read_bytes() == DAHUAN_VISUAL_URDF.read_bytes()


def test_canonical_dahuan_assets_match_frozen_manifest() -> None:
    assert _validate_canonical_model_assets() == DAHUAN_VISUAL_URDF


def test_safety_urdf_preserves_canonical_kinematics_and_generated_overlay() -> None:
    canonical = ET.parse(DAHUAN_VISUAL_URDF).getroot()
    safety = SAFETY_BUILDER["build"](DAHUAN_VISUAL_URDF).getroot()
    canonical_joints = {
        joint.attrib["name"]: ET.tostring(joint, encoding="unicode")
        for joint in canonical.findall("joint")
    }
    safety_joints = {
        joint.attrib["name"]: ET.tostring(joint, encoding="unicode")
        for joint in safety.findall("joint")
    }
    assert safety_joints == canonical_joints
    assert not safety.findall(".//visual")
    assert not safety.findall(".//inertial")
    collision_links = {
        link.attrib["name"] for link in safety.findall("link") if link.findall("collision")
    }
    assert collision_links == SAFETY_BUILDER["TERMINAL_COLLISION_LINKS"]

    overlay_path = DAHUAN_VISUAL_URDF.with_name("collision_overlay.json")
    overlay = json.loads(overlay_path.read_text())
    template = json.loads(
        (PROJECT_ROOT / "src/nexus_deploy/robot/g2a/G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json").read_text()
    )["inter_arm_collision"]
    assert template["canonical_source_sha256"] == hashlib.sha256(
        overlay_path.read_bytes()
    ).hexdigest()
    for field in (
        "geometry", "validation_status", "hard_margin", "warning_margin",
        "left_capsules", "right_capsules", "broad_phase_activation_margin",
        "broad_phase_left_capsules", "broad_phase_right_capsules",
    ):
        assert template[field] == overlay[field]


def test_bundled_dahuan_visual_urdf_has_no_external_asset_dependency() -> None:
    urdf = DAHUAN_VISUAL_URDF
    assert urdf.is_file()
    mesh_paths = set(re.findall(r'<mesh filename="([^"]+)"', urdf.read_text()))
    assert len(mesh_paths) == 60
    for relative in mesh_paths:
        path = Path(relative)
        assert not path.is_absolute()
        resolved = (urdf.parent / path).resolve()
        assert (SIMULATOR_ROOT / "assets").resolve() in resolved.parents
        assert resolved.is_file(), relative


def test_bundled_visual_model_preserves_legacy_materials_and_dahuan_terminal() -> None:
    model = G2Model(DAHUAN_VISUAL_URDF)
    colors = Counter(record[-1] for record in model.linked_geometry_records())
    summary = model.summary()

    assert sum(colors.values()) == 52
    assert colors == Counter(
        {
            0x000000: 29,
            0xFFFFFF: 8,
            0x333333: 4,
            0xC3C3C3: 4,
            0xE9E9D8: 4,
            0xFA6E00: 3,
        }
    )
    assert summary["base_link"] == "base_link"
    assert summary["geometry_count"] == 52
    assert summary["visual_mesh_count"] == 37
    assert summary["vertex_count"] == 110628
    assert summary["face_count"] == 208608


def test_display_only_lod_reduces_browser_payload_without_dropping_materials() -> None:
    model = G2Model(DAHUAN_VISUAL_URDF)
    source = model.linked_geometry_records()
    rendered = model.render_geometry_records()

    assert len(rendered) == len(source) == 52
    assert [item[-1] for item in rendered] == [item[-1] for item in source]
    assert sum(len(item[3]) for item in rendered) < 250_000
    assert sum(len(item[2]) for item in rendered) < 160_000
    assert model.summary()["render_lod"] == "display-only"


def test_integrated_viewer_and_camera_defaults_are_bundled() -> None:
    assert (SIMULATOR_ROOT / "camera_presets.json").is_file()
    for name in ("viewer.html", "viewer.js", "viewer.css"):
        assert (SIMULATOR_ROOT / "static" / name).is_file()
    viewer = (SIMULATOR_ROOT / "static" / "viewer.js").read_text()
    assert 'params.get("platform_origin")' in viewer
    assert 'origin.hostname === window.location.hostname' in viewer
    assert 'postMessage(message, "*")' not in viewer
    assert "nexus-viewer-ready" in viewer
    assert 'fetch("/api/bootstrap", { cache: "no-store"' in viewer
    assert 'cache: "force-cache"' in viewer
    assert 'message.type === "nexus-measured-joints"' in viewer
    assert "function displayJointValues(values)" in viewer
    assert "async function startCachedViewer(modelUrl, workSurface, collisionOverlay)" in viewer
    assert "bootstrap.collision_overlay" in viewer
    assert "function installCollisionOverlay(config, objectByName)" in viewer
    assert "function closestSegmentPoints(" in viewer
    assert "const COLLISION_VISUAL_HOLD_MS = 1000" in viewer
    assert "function latchedCollisionLevel(" in viewer
    overlay = json.loads(
        (SIMULATOR_ROOT / "assets" / "g2" / "G2_dahuan" / "collision_overlay.json").read_text()
    )
    assert overlay["geometry"] == "cross-chain-capsules-v2"
    assert len(overlay["left_capsules"]) == len(overlay["right_capsules"]) == 19
    assert len(overlay["broad_phase_left_capsules"]) == 9
    assert len(overlay["broad_phase_right_capsules"]) == 9

    presets = json.loads((SIMULATOR_ROOT / "camera_presets.json").read_text())
    near_distance = sum(value * value for value in presets["near"]["position"]) ** 0.5
    overview_distance = sum(
        value * value for value in presets["overview"]["position"]
    ) ** 0.5
    assert near_distance < overview_distance
    assert presets["near"]["fov"] < presets["overview"]["fov"]


def test_cacheable_glb_contains_complete_display_kinematics() -> None:
    path = SIMULATOR_ROOT / "assets" / "g2-dahuan-display-v1.glb.gz"
    assert path.is_file()
    assert path.stat().st_size < 2_500_000
    payload = gzip.decompress(path.read_bytes())
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    document = json.loads(payload[20 : 20 + json_length])
    nexus = document["asset"]["extras"]["nexus"]

    assert magic == b"glTF"
    assert version == 2
    assert total == len(payload)
    assert json_type == b"JSON"
    assert len(document["meshes"]) == 52
    assert len(nexus["visual_joint_names"]) == 24
    assert len(nexus["joints"]) == 39
    assert nexus["render_face_count"] == 74713
    assert nexus["gripper_mapping"]["dahuan"].startswith("urdf_position=")
    assert nexus["display_only"] is True
    assert nexus["robot_commands"] == "disabled"
    source_model = G2Model(DAHUAN_VISUAL_URDF)
    source_joints = {str(joint.name): joint for joint in source_model.robot.robot.joints}
    for exported in nexus["joints"]:
        source = source_joints[exported["name"]]
        assert exported["parent"] == str(source.parent)
        assert exported["child"] == str(source.child)
        assert exported["type"] == str(source.type)
        assert exported["axis"] == pytest.approx(np.asarray(source.axis, dtype=float).tolist())
        assert exported["origin"] == pytest.approx(
            np.asarray(source.origin, dtype=float).T.reshape(-1).tolist()
        )


def test_browser_assets_are_served_gzipped_and_immutable() -> None:
    class FakeState:
        meshcat_port = 18777

        @staticmethod
        def health() -> dict[str, object]:
            return {"ok": True}

        @staticmethod
        def bootstrap() -> dict[str, object]:
            return {"ok": True}

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(FakeState()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib_request.urlopen(f"{base}{BROWSER_MODEL_URL}") as response:  # noqa: S310
            model = response.read()
            assert response.headers["Cache-Control"] == IMMUTABLE_CACHE
            assert response.headers["Content-Encoding"] == "gzip"
            assert response.headers["ETag"].startswith('"')
            assert model.startswith(b"\x1f\x8b")
        with urllib_request.urlopen(  # noqa: S310
            f"{base}{MESHCAT_BROWSER_BUNDLE_URL}"
        ) as response:
            assert response.headers["Cache-Control"] == IMMUTABLE_CACHE
            assert response.headers["Content-Encoding"] == "gzip"
            assert len(gzip.decompress(response.read())) > 800_000
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_control_ui_uses_bundled_simulator_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.simulator_control_url is None
    assert args.disable_integrated_simulator is False
    assert args.simulator_viewer_port == 8767
    assert not hasattr(args, "simulator_python")


def test_simulator_loads_eligible_apriltag_surface_height(tmp_path: Path) -> None:
    path = tmp_path / "alignment-surfaces" / "standard_table" / "active.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurement_id": "surface-test",
                "base_frame": "base_link",
                "created_at": 1.0,
                "surface": {
                    "eligible": True,
                    "height_m": 0.712,
                    "pose": {"x_m": 0.8, "y_m": -0.4, "z_m": 0.712},
                    "normal": [0.004, -0.003, 0.9999875],
                },
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "safety-profiles" / "g2a" / "measured-safety.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            {
                "surface_calibration": {"measurement_id": "surface-test"},
                "surfaces": [
                    {
                        "height": 0.712,
                        "hard_margin": 0.01,
                        "origin": [0.8, -0.4, 0.712],
                        "normal": [0.004, -0.003, 0.9999875],
                        "xy_bounds": [0.3, 1.0, -0.8, 0.8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = object.__new__(SimulationState)
    state.state_dir = tmp_path
    state.alignment_surfaces_directory = tmp_path / "alignment-surfaces"

    surface = state._load_work_surface()

    assert surface["height"] == pytest.approx(0.712)
    assert surface["hard_margin"] == pytest.approx(0.01)
    assert surface["origin"] == pytest.approx([0.8, -0.4, 0.712])
    assert surface["normal"] == pytest.approx([0.004, -0.003, 0.9999875])
    assert surface["xy_bounds"] == pytest.approx([0.3, 1.0, -0.8, 0.8])
    assert surface["source"] == "g2a_alignment_apriltag"
    assert surface["measurement_id"] == "surface-test"
    assert surface["safety_profile_id"] == "measured-safety.json"


def test_missing_platform_interpreter_fails_with_install_hint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="重新安装当前工程"):
        SimulatorProcess.start(
            python=tmp_path / "missing-python",
            runtime_dir=tmp_path / "runtime",
            port=18776,
            viewer_port=18777,
            timeout_s=0.01,
        )
