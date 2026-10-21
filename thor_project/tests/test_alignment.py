from __future__ import annotations

import asyncio
import fcntl
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nexus_deploy.alignment.adapters import (
    HTTPAlignmentRobotAdapter,
    MockAlignmentRobotAdapter,
    _private_ip,
)
from nexus_deploy.alignment.config import load_config
from nexus_deploy.alignment.core import AlignmentEngine, AlignmentError, AlignmentState
from nexus_deploy.alignment.detectors import MockAprilTagDetector, OpenCVAprilTagDetector
from nexus_deploy.alignment.geometry import invert, matrix_pose, pose_matrix, project_points, tag_corners
from nexus_deploy.alignment.robot_agent import (
    G2ReadOnlySource,
    _browser_image,
    _gdk_calibration,
    _gdk_calibration_with_retry,
    _gdk_camera_spec,
    _robot_asset_id,
    _robot_gdk_lock_busy,
    _transform_matrix,
    _validate_bind,
)
from nexus_deploy.alignment.service import AlignmentManager
from nexus_deploy.alignment.targets import TargetRepository
from nexus_deploy.alignment.types import Detection, Frame


def write_config(tmp_path: Path, *, task_running: bool = False) -> Path:
    path = tmp_path / "alignment.toml"
    path.write_text(
        f"""
[service]
default_profile = "table"
default_adapter = "mock"
default_detector = "mock"
default_robot_host = "10.42.1.102"
reports_directory = "reports"
update_hz = 30
tag_loss_grace_s = 0.1
reconnect_grace_s = 1

[profiles.table]
display_name = "Test table"
version = "test-v1"
camera_name = "cam_head"
tag_family = "tag36h11"
tag_ids = [10]
tag_size_m = 0.12
expected_table_pose = [0.8, 0, 1, 0, 0, 0]
tag_from_table = [0, 0, 0, 0, 0, 0]
[profiles.table.inspection_joints]
head = 0
[profiles.table.tolerance]
forward_m = 0.02
lateral_m = 0.02
yaw_deg = 2
stable_s = 0.1
window_s = 0.3
minimum_samples = 2
minimum_detection_ratio = 0.5

[adapters.mock]
display_name = "Mock"
plugin = "nexus_deploy.alignment.adapters:MockAlignmentRobotAdapter"
[adapters.mock.settings]
detector_id = "mock"
task_running = {str(task_running).lower()}
head_joints = {{ head = 0 }}
base_from_camera = [0, 0, 0, 0, 0, 0]

[detectors.mock]
plugin = "nexus_deploy.alignment.detectors:MockAprilTagDetector"
[detectors.mock.settings]
actual_table_pose = [0.805, 0.004, 1, 0, 0, 0.5]
animate = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def make_engine(tmp_path: Path, *, task_running: bool = False) -> AlignmentEngine:
    config = load_config(write_config(tmp_path, task_running=task_running))
    adapter = MockAlignmentRobotAdapter(config.adapters["mock"].settings)
    detector = MockAprilTagDetector(config.detectors["mock"].settings)
    return AlignmentEngine(
        config,
        config.profiles["table"],
        adapter,
        detector,
        host="127.0.0.1",
        adapter_id="mock",
        detector_id="mock",
    )


def test_pose_round_trip_and_inverse() -> None:
    transform = pose_matrix((0.8, -0.1, 1.2, 3, -4, 12))
    assert np.allclose(transform @ invert(transform), np.eye(4), atol=1e-9)
    assert matrix_pose(transform) == pytest.approx((0.8, -0.1, 1.2, 3, -4, 12))


def test_projection_applies_opencv_lens_distortion() -> None:
    cv2 = pytest.importorskip("cv2")
    camera_matrix = np.asarray(
        ((1007.158481, 0, 960.2425436), (0, 1007.129766, 767.6753883), (0, 0, 1)),
        dtype=np.float64,
    )
    distortion = np.asarray(
        (
            30.14113633,
            14.77666421,
            1.20304e-05,
            2.08231e-05,
            0.5692396909,
            30.56955569,
            26.48210739,
            3.652510894,
        )
    )
    camera_from_tag = pose_matrix((0.5, -0.1, 1.2, 2, -3, 8))
    actual = project_points(
        tag_corners(0.1), camera_from_tag, camera_matrix, distortion
    )
    rotation_vector, _ = cv2.Rodrigues(camera_from_tag[:3, :3])
    expected, _ = cv2.projectPoints(
        tag_corners(0.1),
        rotation_vector,
        camera_from_tag[:3, 3],
        camera_matrix,
        distortion,
    )
    assert np.asarray(actual) == pytest.approx(expected.reshape(-1, 2), abs=0.005)


def test_config_exposes_default_robot_host(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    assert config.default_robot_host == "10.42.1.102"
    assert config.surfaces_directory == (tmp_path / "runtime/alignment-surfaces").resolve()
    assert config.profiles["table"].surface_offset_m == 0.0


def test_robot_asset_id_is_attached_to_alignment_provenance(tmp_path: Path) -> None:
    path = tmp_path / "aid_info.yaml"
    path.write_text('AID: "G2A0004C201260"\n', encoding="utf-8")

    assert _robot_asset_id(path) == "G2A0004C201260"


def test_robot_agent_shell_health_does_not_start_gdk(tmp_path: Path) -> None:
    calibration = tmp_path / "camera.json"
    calibration.write_text(
        json.dumps({"robot_id": "g2a", "camera_name": "head_stereo_left"}),
        encoding="utf-8",
    )
    source = G2ReadOnlySource(
        calibration,
        fps=8,
        camera_timeout_ms=1000,
        idle_release_s=60,
        robot_lock_path=tmp_path / "robot.lock",
    )

    health = source.health()

    assert health["status"] == "ok"
    assert health["task_running"] is False
    assert health["gdk_state"] == "idle"
    assert source._process is None


def test_robot_agent_shell_health_observes_external_gdk_owner(tmp_path: Path) -> None:
    calibration = tmp_path / "camera.json"
    calibration.write_text(
        json.dumps({"robot_id": "g2a", "camera_name": "head_stereo_left"}),
        encoding="utf-8",
    )
    lock_path = tmp_path / "robot.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _robot_gdk_lock_busy(lock_path) == (True, None)
        source = G2ReadOnlySource(
            calibration,
            fps=8,
            camera_timeout_ms=1000,
            idle_release_s=60,
            robot_lock_path=lock_path,
        )
        health = source.health()
        assert health["task_running"] is True
        assert health["gdk_state"] == "idle"
        assert source._process is None


def test_robot_agent_release_hold_prevents_immediate_gdk_reacquire(tmp_path: Path) -> None:
    calibration = tmp_path / "camera.json"
    calibration.write_text(
        json.dumps({"robot_id": "g2a", "camera_name": "head_stereo_left"}),
        encoding="utf-8",
    )
    source = G2ReadOnlySource(
        calibration,
        fps=8,
        camera_timeout_ms=1000,
        idle_release_s=60,
        robot_lock_path=tmp_path / "robot.lock",
    )

    assert source.release_worker(hold_s=15) is False
    assert source.health()["release_hold_remaining_s"] > 14
    with pytest.raises(RuntimeError, match="temporarily released for a robot task"):
        source.ensure_started()

    source.close()


def test_robot_agent_state_exposes_canonical_state_without_breaking_head_joints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = tmp_path / "camera.json"
    calibration.write_text(
        json.dumps({"robot_id": "g2a", "camera_name": "head_stereo_left"}),
        encoding="utf-8",
    )
    source = G2ReadOnlySource(
        calibration,
        fps=8,
        camera_timeout_ms=1000,
        idle_release_s=60,
        robot_lock_path=tmp_path / "robot.lock",
    )
    snapshot = {
        "head_joints": {"idx13_head_joint3": 0.417},
        "canonical_state": {"idx11_arm_l_joint1": 0.25},
        "captured_at": 123.0,
    }
    monkeypatch.setattr(source, "_request", lambda command: snapshot)

    assert source.state() == snapshot


def test_robot_agent_state_accepts_legacy_head_only_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = tmp_path / "camera.json"
    calibration.write_text(
        json.dumps({"robot_id": "g2a", "camera_name": "head_stereo_left"}),
        encoding="utf-8",
    )
    source = G2ReadOnlySource(
        calibration,
        fps=8,
        camera_timeout_ms=1000,
        idle_release_s=60,
        robot_lock_path=tmp_path / "robot.lock",
    )
    monkeypatch.setattr(
        source,
        "_request",
        lambda command: {"idx13_head_joint3": 0.417},
    )

    assert source.state() == {"head_joints": {"idx13_head_joint3": 0.417}}


def test_http_adapter_close_requests_immediate_gdk_release(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = HTTPAlignmentRobotAdapter({})
    adapter.base_url = "http://10.42.1.102:8080"
    calls: list[tuple[str, str]] = []

    def opened(path: str, *, method: str = "GET") -> tuple[bytes, str, dict[str, str]]:
        calls.append((path, method))
        return b'{"released": true}', "application/json", {}

    monkeypatch.setattr(adapter, "_open", opened)
    adapter.close()

    assert calls == [("/api/alignment/release", "POST")]
    assert adapter.base_url == ""


def test_example_target_is_explicitly_unconfigured() -> None:
    config = load_config(Path(__file__).parents[1] / "alignment.example.toml")
    assert config.profiles["standard_table"].target_configured is False
    assert config.update_hz == 8
    assert config.tag_loss_grace_s == 1.5
    profile = config.profiles["standard_table"]
    assert profile.tolerance.window_s == 2.0
    assert profile.tolerance.minimum_samples == 4
    assert profile.tolerance.minimum_detection_ratio == pytest.approx(0.45)
    assert profile.tag_size_m == pytest.approx(0.095)
    assert profile.allow_hamming0_fallback_measurement is False


def test_hidden_alignment_overlays_are_forced_out_of_layout() -> None:
    static = Path(__file__).parents[1] / "src" / "nexus_deploy" / "alignment" / "static"
    assert '[hidden] {\n  display: none !important;' in (static / "visibility.css").read_text()
    assert 'href="/assets/visibility.css"' in (static / "alignment.html").read_text()


def test_alignment_csp_allows_fetched_camera_blob_images() -> None:
    from nexus_deploy.alignment.service import _static_headers

    assert "img-src 'self' data: blob:" in _static_headers()["Content-Security-Policy"]


def test_saved_target_immediately_redraws_camera_overlay() -> None:
    script = (
        Path(__file__).parents[1] / "src" / "nexus_deploy" / "alignment" / "static" / "alignment.js"
    ).read_text(encoding="utf-8")
    assert "render(value.session);\n    draw(value.session);" in script
    assert "draw(state.session || session);" in script
    assert "expandPolygon(targetCorners, 10)" in script
    assert 'polygon(context, targetOutline, "#ff5bd6", true, 6)' in script
    assert "function arrow(" not in script


def test_alignment_ui_distinguishes_target_surface_and_safety_profile() -> None:
    static = Path(__file__).parents[1] / "src" / "nexus_deploy" / "alignment" / "static"
    markup = (static / "alignment.html").read_text(encoding="utf-8")
    script = (static / "alignment.js").read_text(encoding="utf-8")

    assert "标准目标位置" in markup
    assert "桌面高度标定" in markup
    assert "安全配置" in markup
    assert "容错结果不能生成桌面安全配置" in script
    assert "id=\"tag-detection\"" in markup
    assert 'session.result?.detection?.method !== "fallback"' in script
    assert "function renderDetectionMethod(result)" in script
    assert 'dom.start.disabled = state.starting || session.state !== "READY";' in script
    assert 'if (state.starting || state.session?.state !== "READY") return;' in script
    assert "state.starting=false;dom.start.disabled=false;" in script
    assert "Number(session.surface_stable_s || 0)" in script
    assert "严格安全测量需要连续稳定保持 2 秒" in script
    assert 'dom.cameraStage.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`' in script
    styles = (static / "target-admin.css").read_text(encoding="utf-8")
    assert ".alignment-grid {\n  align-items: start;" in styles
    assert ".camera-stage {" in styles
    assert "aspect-ratio: 5 / 4;" in styles
    assert ".tag-detection {" in styles
    assert "min-height: 59px;" in styles


def test_ready_session_without_viewer_is_released(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = replace(load_config(write_config(tmp_path)), reconnect_grace_s=0.02)
        manager = AlignmentManager(config)
        engine = await manager.create(
            {"host": "127.0.0.1", "profile_id": "table", "adapter_id": "mock"}
        )
        assert engine.state == AlignmentState.READY
        await asyncio.sleep(0.05)
        assert engine.state == AlignmentState.STOPPED
        await manager.close()

    asyncio.run(scenario())


def test_ready_session_viewer_presence_prevents_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = replace(load_config(write_config(tmp_path)), reconnect_grace_s=0.02)
        manager = AlignmentManager(config)
        engine = await manager.create(
            {"host": "127.0.0.1", "profile_id": "table", "adapter_id": "mock"}
        )
        manager.viewer_joined(engine.id)
        await asyncio.sleep(0.05)
        assert engine.state == AlignmentState.READY
        manager.viewer_left(engine.id)
        await asyncio.sleep(0.05)
        assert engine.state == AlignmentState.STOPPED
        await manager.close()

    asyncio.run(scenario())


def test_same_robot_and_profile_reuses_active_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = load_config(write_config(tmp_path))
        manager = AlignmentManager(config)
        payload = {"host": "127.0.0.1", "profile_id": "table", "adapter_id": "mock"}
        first = await manager.create(payload)
        second = await manager.create(payload)
        assert second is first
        assert len(manager.sessions) == 1
        await manager.close()

    asyncio.run(scenario())


def test_admin_password_publishes_current_stable_pose_as_versioned_target(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = load_config(write_config(tmp_path))
        manager = AlignmentManager(config, admin_password="correct horse")
        engine = await manager.create(
            {"host": "127.0.0.1", "profile_id": "table", "adapter_id": "mock"}
        )
        manager.viewer_joined(engine.id)
        engine.start()
        deadline = time.monotonic() + 2
        while not engine.snapshot()["result"].get("stable") and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert engine.snapshot()["result"]["stable"] is True
        engine._observation_stable_since = time.monotonic() - 2.1
        engine._surface_stable_since = time.monotonic() - 2.1

        with pytest.raises(AlignmentError, match="密码错误"):
            await manager.publish_current_as_target(engine, "wrong")
        assert TargetRepository(config.targets_directory).active("table") is None

        published = await manager.publish_current_as_target(engine, "correct horse")
        assert published["target"]["expected_table_pose"] == pytest.approx(
            [0.805, 0.004, 1.0, 0.0, 0.0, 0.5]
        )
        assert published["session"]["profile"]["target_configured"] is True
        assert TargetRepository(config.targets_directory).history("table")[0]["active"] is True
        manager.viewer_left(engine.id)
        await manager.close()

    asyncio.run(scenario())


def test_active_target_does_not_override_current_tag_geometry(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    repository = TargetRepository(config.targets_directory)
    original = config.profiles["table"]
    repository.publish(
        original,
        [0.81, 0.02, 1.01, 0.0, 0.0, 1.0],
        robot={"id": "g2"},
        robot_host="127.0.0.1",
        calibration_version="camera-v1",
    )

    replacement_tag = replace(
        original,
        tag_family="tag25h9",
        tag_ids=(7,),
        tag_size_m=0.095,
        tag_from_table=pose_matrix([0.01, 0.02, 0.0, 0.0, 0.0, 90.0]),
    )
    resolved = repository.resolve(replacement_tag)

    assert resolved.target_configured is True
    assert resolved.tag_family == "tag25h9"
    assert resolved.tag_ids == (7,)
    assert resolved.tag_size_m == pytest.approx(0.095)
    assert resolved.tag_from_table == pytest.approx(replacement_tag.tag_from_table)
    assert resolved.base_from_expected_table == pytest.approx(
        pose_matrix([0.81, 0.02, 1.01, 0.0, 0.0, 1.0])
    )


def test_stable_flat_tag_publishes_deployment_surface_height(tmp_path: Path) -> None:
    async def scenario() -> None:
        template = tmp_path / "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json"
        template.write_text(
            json.dumps(
                {
                    "version": 2,
                    "calibrated": False,
                    "read_only_ready": True,
                    "execute_approved": False,
                    "base_frame": "base_link",
                    "robot_model": {
                        "type": "urdf",
                        "path": "assets/G2_dahuan_safety.urdf",
                        "sha256": "",
                    },
                    "surfaces": [
                        {
                            "name": "work_surface",
                            "frame": "base_link",
                            "height": 0.75,
                            "hard_margin": 0.02,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        base_config = load_config(write_config(tmp_path))
        configured_profile = replace(
            base_config.profiles["table"],
            robot_model_id="G2_dahuan_safety",
            robot_model_sha256="test-sha",
        )
        config = replace(
            base_config,
            safety_profiles_directory=tmp_path / "runtime/safety-profiles",
            safety_profile_template=template,
            profiles={**base_config.profiles, "table": configured_profile},
        )
        template_data = json.loads(template.read_text(encoding="utf-8"))
        template_data["robot_model"]["sha256"] = "test-sha"
        template.write_text(json.dumps(template_data), encoding="utf-8")
        manager = AlignmentManager(config)
        engine = await manager.create(
            {"host": "127.0.0.1", "profile_id": "table", "adapter_id": "mock"}
        )
        manager.viewer_joined(engine.id)
        engine.start()
        deadline = time.monotonic() + 2
        while not engine.snapshot()["result"].get("surface", {}).get("eligible"):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        engine._observation_stable_since = time.monotonic() - 2.1
        engine._surface_stable_since = time.monotonic() - 2.1

        published = await manager.publish_current_surface(engine)
        active = manager.surfaces.active("table")

        assert published["surface"]["height_m"] == pytest.approx(1.0)
        assert published["surface"]["frame"] == "base_link"
        assert published["surface"]["eligible"] is True
        assert published["robot"]["id"] == "mock-g2"
        assert published["safety_profile"]["generated"] is True
        generated = (
            config.safety_profiles_directory
            / "g2a"
            / published["safety_profile"]["id"]
        )
        assert generated.is_file()
        generated_data = json.loads(generated.read_text(encoding="utf-8"))
        assert generated_data["surfaces"][0]["height"] == pytest.approx(1.0)
        assert active == {
            key: value for key, value in published.items() if key != "safety_profile"
        }
        artifacts = manager._profile_public(configured_profile)["artifacts"]
        assert artifacts["target"]["configured"] is True
        assert artifacts["surface"]["measurement_id"] == published["measurement_id"]
        assert artifacts["surface"]["height_m"] == pytest.approx(1.0)
        assert artifacts["safety_profile"]["generated"] is True
        assert artifacts["safety_profile"]["filename"] == published["safety_profile"]["id"]
        manager.viewer_left(engine.id)
        await manager.close()

    asyncio.run(scenario())


def test_tilted_tag_is_not_eligible_for_surface_safety(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    result = engine._surface_result(
        (0.8, 0.0, 1.0, 2.1, 0.0, 0.0),
        stable=True,
        detection_ratio=1.0,
        spreads=(0.0,) * 6,
    )
    assert result["eligible"] is False
    assert result["reason"] == "surface_tilt_exceeds_limit"
    engine.close()


def test_surface_publish_requires_continuous_strict_stability(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.start()
    deadline = time.monotonic() + 2
    while not engine.snapshot()["result"].get("surface", {}).get("eligible"):
        assert time.monotonic() < deadline
        time.sleep(0.02)

    assert engine.snapshot()["surface_stable_s"] < 2
    with pytest.raises(AlignmentError, match="至少 2 秒"):
        engine.capture_surface_measurement(2.0)

    engine._surface_stable_since = time.monotonic() - 2.1
    assert engine.capture_surface_measurement(2.0)["stable_hold_s"] >= 2
    engine.close()


def test_guidance_stability_is_relaxed_without_relaxing_surface_safety(
    tmp_path: Path,
) -> None:
    engine = make_engine(tmp_path)
    spread = (0.0075, 0.0075, 0.0075, 0.55, 0.55, 0.75)

    assert engine._spread_is_stable(spread) is True
    assert engine._spread_is_surface_stable(spread) is False
    surface = engine._surface_result(
        (0.8, 0.0, 1.0, 0.0, 0.0, 0.0),
        stable=True,
        detection_ratio=1.0,
        spreads=spread,
    )
    assert surface["eligible"] is False
    assert surface["reason"] == "measurement_spread_exceeds_safety_limit"
    engine.close()


def test_http_adapter_only_accepts_private_literal_addresses() -> None:
    assert _private_ip("127.0.0.1") == "127.0.0.1"
    assert _private_ip("192.168.10.8") == "192.168.10.8"
    with pytest.raises(ValueError):
        _private_ip("example.com")
    with pytest.raises(ValueError):
        _private_ip("8.8.8.8")


def test_robot_agent_preserves_encoded_jpeg_without_reencoding() -> None:
    content = b"not-decoded-by-the-agent"
    mime, result, width, height = _browser_image(
        {"encoding": "jpeg", "width": 640, "height": 480, "data": content}
    )
    assert (mime, result, width, height) == ("image/jpeg", content, 640, 480)


def test_robot_agent_converts_gdk_transform() -> None:
    class Value:
        pass

    transform = Value()
    transform.translation = Value()
    transform.rotation = Value()
    transform.translation.x, transform.translation.y, transform.translation.z = (1, 2, 3)
    transform.rotation.x, transform.rotation.y = (0, 0)
    transform.rotation.z, transform.rotation.w = (np.sqrt(0.5), np.sqrt(0.5))
    assert np.allclose(
        _transform_matrix(transform),
        [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
        atol=1e-9,
    )


def test_robot_agent_uses_matching_stereo_intrinsics_and_extrinsics() -> None:
    class Value:
        pass

    class NamedValue:
        def __init__(self, name: str) -> None:
            self.name = name

    class GDK:
        class CameraType:
            kHeadColor = NamedValue("kHeadColor")
            kHeadStereoLeft = NamedValue("kHeadStereoLeft")
            kHeadStereoRight = NamedValue("kHeadStereoRight")

        class SensorExtrinsicType:
            kHeadRGBDToHeadLink3 = NamedValue("kHeadRGBDToHeadLink3")
            kHeadLeftStereoToHeadLink3 = NamedValue("kHeadLeftStereoToHeadLink3")
            kHeadRightStereoToHeadLink3 = NamedValue("kHeadRightStereoToHeadLink3")

    camera_type, extrinsic_type, display_name = _gdk_camera_spec(
        GDK, "head_stereo_left"
    )
    assert camera_type is GDK.CameraType.kHeadStereoLeft
    assert extrinsic_type is GDK.SensorExtrinsicType.kHeadLeftStereoToHeadLink3
    assert display_name == "头部左双目相机"

    transform = Value()
    transform.translation = Value()
    transform.rotation = Value()
    transform.translation.x = transform.translation.y = transform.translation.z = 0
    transform.rotation.x = transform.rotation.y = transform.rotation.z = 0
    transform.rotation.w = 1

    intrinsic = Value()
    intrinsic.intrinsic = [1007.1, 1007.2, 960.2, 767.7]
    intrinsic.distortion = [0.1] * 8

    class Camera:
        def get_camera_intrinsic(self, selected: object) -> object:
            assert selected is camera_type
            return intrinsic

        def get_image_shape(self, selected: object) -> tuple[int, int]:
            assert selected is camera_type
            return (1920, 1536)

    class TF:
        def get_tf_from_base_link(self, link: str) -> object:
            assert link == "head_link3"
            return transform

        def get_tf_from_sensor(self, selected: object) -> object:
            assert selected is extrinsic_type
            return transform

    adapter = Value()
    adapter._gdk = GDK
    adapter._camera = Camera()
    result = _gdk_calibration(adapter, TF(), "head_stereo_left")
    assert result["image_width"] == 1920
    assert result["image_height"] == 1536
    assert result["camera_matrix"] == [
        [1007.1, 0.0, 960.2],
        [0.0, 1007.2, 767.7],
        [0.0, 0.0, 1.0],
    ]
    assert np.asarray(result["base_from_head_link3"]) == pytest.approx(np.eye(4))
    assert np.asarray(result["head_link3_from_camera_factory"]) == pytest.approx(np.eye(4))
    assert np.asarray(result["base_from_camera"]) == pytest.approx(np.eye(4))
    assert "kHeadStereoLeft" in result["intrinsic_source"]
    assert "kHeadLeftStereoToHeadLink3" in result["extrinsic_source"]


def test_robot_agent_retries_cold_gdk_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def calibration(_adapter: object, _tf: object, camera_name: str) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Failed to get tf from base link")
        return {"camera_name": camera_name}

    monkeypatch.setattr("nexus_deploy.alignment.robot_agent._gdk_calibration", calibration)
    result = _gdk_calibration_with_retry(
        object(),
        object(),
        "head_stereo_left",
        timeout_s=1,
        retry_interval_s=0,
    )

    assert attempts == 3
    assert result == {"camera_name": "head_stereo_left"}


def test_robot_agent_private_listener_requires_explicit_auth_choice() -> None:
    with pytest.raises(ValueError, match="requires ALIGNMENT_ROBOT_TOKEN"):
        _validate_bind("10.42.1.102", "")
    _validate_bind("10.42.1.102", "", allow_unauthenticated_private_lan=True)


def test_opencv_detector_recognizes_configured_apriltag(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    config = load_config(write_config(tmp_path))
    profile = config.profiles["table"]
    adapter = MockAlignmentRobotAdapter(config.adapters["mock"].settings)
    adapter.connect("127.0.0.1")
    calibration = adapter.calibration("cam_head")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 10, 300)
    image = np.full((720, 1280), 255, dtype=np.uint8)
    image[210:510, 490:790] = marker
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    detections = OpenCVAprilTagDetector({}).detect(
        Frame(payload.tobytes(), "image/png", 1280, 720, time.time()),
        calibration,
        profile,
    )
    assert len(detections) == 1
    assert detections[0].tag_id == 10
    assert detections[0].quality > 0.8
    assert detections[0].diagnostics["corner_refinement"] == "subpixel"


def test_opencv_detector_enables_subpixel_corner_refinement() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = OpenCVAprilTagDetector({})

    parameters = detector._parameters()

    assert parameters.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX
    assert parameters.cornerRefinementWinSize == 5


def test_opencv_detector_falls_back_for_small_known_apriltag(tmp_path: Path, monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    config = load_config(write_config(tmp_path))
    profile = replace(config.profiles["table"], tag_ids=(0,), tag_size_m=0.10)
    adapter = MockAlignmentRobotAdapter(config.adapters["mock"].settings)
    adapter.connect("127.0.0.1")
    calibration = adapter.calibration("cam_head")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
    canvas = np.full((200, 300), 115, dtype=np.uint8)
    source = np.float32(((0, 0), (159, 0), (159, 159), (0, 159)))
    destination = np.float32(((140, 80), (164, 81), (167, 98), (142, 97)))
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(marker, transform, (300, 200), borderValue=115)
    mask = cv2.warpPerspective(np.full_like(marker, 255), transform, (300, 200))
    canvas[mask > 0] = warped[mask > 0]
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok

    # Exercise fallback deterministically across OpenCV versions.
    from types import SimpleNamespace
    monkeypatch.setattr(cv2.aruco, "ArucoDetector",
                        lambda *args: SimpleNamespace(detectMarkers=lambda image: ([], None, [])))
    detections = OpenCVAprilTagDetector({}).detect(
        Frame(encoded.tobytes(), "image/jpeg", 300, 200, time.time()),
        calibration,
        profile,
    )
    assert len(detections) == 1
    assert detections[0].tag_id == 0
    assert detections[0].diagnostics["fallback"] is True
    assert detections[0].diagnostics["fallback_hamming"] <= 3
    assert detections[0].diagnostics["corner_refinement"] == "subpixel"


def test_fallback_detection_cannot_publish_surface_safety(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert engine.latest_detection is None
    engine.latest_detection = Detection(
        tag_id=10,
        camera_from_tag=np.eye(4),
        corners=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        quality=0.8,
        diagnostics={"fallback": True, "fallback_hamming": 0},
    )

    result = engine._result_from_pose(
        (0.8, 0.0, 1.0, 0.0, 0.0, 0.0),
        stable=True,
        detection_ratio=1.0,
        spreads=(0.0,) * 6,
    )

    assert result["detection"]["method"] == "fallback"
    assert result["surface"]["eligible"] is False
    assert result["surface"]["reason"] == "fallback_detection_not_allowed"
    engine._last_detection_at = time.monotonic()
    engine._observation_stable_since = time.monotonic() - 3
    engine.latest_result = result
    with pytest.raises(AlignmentError, match="容错识别"):
        engine.capture_standard_pose()
    engine.close()


def test_profile_can_explicitly_accept_only_hamming0_fallback_measurement(
    tmp_path: Path,
) -> None:
    engine = make_engine(tmp_path)
    engine.profile = replace(engine.profile, allow_hamming0_fallback_measurement=True)
    engine.latest_detection = Detection(
        tag_id=10,
        camera_from_tag=np.eye(4),
        corners=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        quality=0.8,
        diagnostics={"fallback": True, "fallback_hamming": 0},
    )
    accepted = engine._result_from_pose(
        (0.8, 0.0, 1.0, 0.0, 0.0, 0.0),
        stable=True,
        detection_ratio=1.0,
        spreads=(0.0,) * 6,
    )
    assert accepted["surface"]["eligible"] is True
    assert accepted["surface"]["measurement_mode"] == "relaxed_hamming0_fallback"

    engine.latest_detection = replace(
        engine.latest_detection,
        diagnostics={"fallback": True, "fallback_hamming": 1},
    )
    rejected = engine._result_from_pose(
        (0.8, 0.0, 1.0, 0.0, 0.0, 0.0),
        stable=True,
        detection_ratio=1.0,
        spreads=(0.0,) * 6,
    )
    assert rejected["surface"]["eligible"] is False
    assert rejected["surface"]["reason"] == "fallback_detection_not_allowed"
    engine.close()


def test_engine_reaches_aligned_and_writes_versioned_report(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert engine.state == AlignmentState.READY
    assert all(item["ok"] for item in engine.checks)
    engine.start()
    deadline = time.monotonic() + 2
    while engine.state != AlignmentState.ALIGNED and time.monotonic() < deadline:
        time.sleep(0.03)
    assert engine.state == AlignmentState.ALIGNED
    snapshot = engine.snapshot()
    assert snapshot["result"]["within_tolerance"] is True
    assert snapshot["result"]["guidance"]["mode"] == "table"

    report = engine.complete()
    report_path = Path(engine.completed_report or "")
    assert report["result"] == "aligned"
    assert report["profile_version"] == "test-v1"
    assert report["observation_context"]["head_joints"] == {"head": 0.0}
    assert report["observation_context"]["inspection_joint_tolerance_rad"] == 0.005
    assert len(
        report["observation_context"]["camera_calibration"]["fingerprint_sha256"]
    ) == 64
    assert report["alignment"]["detection"]["method"] == "standard"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["calibration_version"] == "mock-calibration-v1"
    engine.close()


def test_unconfigured_target_never_reports_aligned(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.profile = replace(engine.profile, target_configured=False)
    engine.start()
    deadline = time.monotonic() + 2
    while not engine.snapshot()["result"].get("stable") and time.monotonic() < deadline:
        time.sleep(0.02)
    snapshot = engine.snapshot()
    assert snapshot["state"] == AlignmentState.ADJUSTING
    assert snapshot["profile"]["target_configured"] is False
    assert snapshot["result"]["within_tolerance"] is False
    assert snapshot["target_corners"] is None
    engine.set_adjustment_mode("robot")
    engine.close()


def test_running_robot_cannot_start_alignment(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, task_running=True)
    assert engine.state == AlignmentState.ERROR
    assert any(item["id"] == "idle" and not item["ok"] for item in engine.checks)
    with pytest.raises(AlignmentError):
        engine.start()
    engine.close()


def test_moving_robot_reverses_table_guidance(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    error = {"forward_mm": 30.0, "lateral_mm": -10.0, "yaw_deg": 3.0}
    table = engine._guidance(error)
    engine.set_adjustment_mode("robot")
    robot = engine._guidance(error)
    assert robot["forward_mm"] == -table["forward_mm"]
    assert robot["lateral_mm"] == -table["lateral_mm"]
    assert robot["yaw_deg"] == -table["yaw_deg"]
    engine.close()
