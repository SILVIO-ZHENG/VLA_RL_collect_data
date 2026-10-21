"""Standalone process for Nexus's bundled real-time G2 renderer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import signal
import threading
import time
import xml.etree.ElementTree as ET
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import meshcat

from .model import G2Model, ModelError
from .scene import MeshcatBridge, MeshcatScene, ViewerError

INTEGRATED_SIMULATOR_COMPONENT = "nexus-platform-integrated-simulator"
PACKAGE_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = PACKAGE_ROOT / "assets"
STATIC_ROOT = PACKAGE_ROOT / "static"
CANONICAL_MODEL_ROOT = ASSET_ROOT / "g2" / "G2_dahuan"
CANONICAL_MODEL_MANIFEST = CANONICAL_MODEL_ROOT / "manifest.json"
COLLISION_OVERLAY_CONFIG = CANONICAL_MODEL_ROOT / "collision_overlay.json"
BROWSER_MODEL_GZIP = ASSET_ROOT / "g2-dahuan-display-v1.glb.gz"
MESHCAT_VIEWER_ROOT = Path(meshcat.__file__).resolve().parent / "viewer" / "dist"
MESHCAT_BROWSER_BUNDLE = MESHCAT_VIEWER_ROOT / "main.min.js"
MESHCAT_BROWSER_BUNDLE_GZIP = (
    gzip.compress(MESHCAT_BROWSER_BUNDLE.read_bytes(), compresslevel=9, mtime=0)
    if MESHCAT_BROWSER_BUNDLE.is_file()
    else b""
)
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


def _etag(body: bytes) -> str:
    return f'"{hashlib.sha256(body).hexdigest()}"'


def _asset_url(prefix: str, path: Path, suffix: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "missing"
    return f"/assets/{prefix}-{digest}.{suffix}"


def _referenced_mesh_tree_sha256(urdf_path: Path) -> str:
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise OSError(f"G2_dahuan 权威 URDF 不可解析: {exc}") from exc
    filenames = sorted(
        {
            str(mesh.attrib["filename"])
            for mesh in root.findall(".//mesh")
            if mesh.attrib.get("filename")
        }
    )
    digest = hashlib.sha256()
    asset_root = ASSET_ROOT.resolve()
    for filename in filenames:
        path = (urdf_path.parent / filename).resolve()
        if asset_root not in path.parents or not path.is_file():
            raise OSError(f"G2_dahuan 引用的网格不存在或越出资产目录: {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_canonical_model_assets() -> Path:
    try:
        manifest = json.loads(CANONICAL_MODEL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OSError(f"G2_dahuan 权威模型清单不可读: {exc}") from exc
    if (
        manifest.get("schema_version") != 1
        or manifest.get("id") != "G2_dahuan"
        or manifest.get("model_version") != "G2_dahuan-v5-terminal-aligned-25mm"
        or manifest.get("status") != "frozen"
    ):
        raise OSError("G2_dahuan 权威模型清单版本或冻结状态无效")
    checks = (
        (CANONICAL_MODEL_ROOT / "G2_dahuan.urdf", manifest.get("urdf_sha256")),
        (CANONICAL_MODEL_ROOT / "G2_dahuan_vendor.urdf", manifest.get("vendor_urdf_sha256")),
        (
            CANONICAL_MODEL_ROOT / "G2_t2_crsB_factory.urdf",
            manifest.get("factory_crsb_urdf_sha256"),
        ),
        (BROWSER_MODEL_GZIP, manifest.get("browser_glb_sha256")),
        (COLLISION_OVERLAY_CONFIG, manifest.get("collision_overlay_sha256")),
    )
    for path, expected in checks:
        if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
            raise OSError(f"G2_dahuan 冻结资产或 SHA-256 声明缺失: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise OSError(
                f"G2_dahuan 冻结资产哈希不一致: {path.name} "
                f"expected={expected} actual={actual}"
            )
    urdf_path = CANONICAL_MODEL_ROOT / "G2_dahuan.urdf"
    expected_mesh_tree = manifest.get("referenced_mesh_tree_sha256")
    actual_mesh_tree = _referenced_mesh_tree_sha256(urdf_path)
    if actual_mesh_tree != expected_mesh_tree:
        raise OSError(
            "G2_dahuan 引用网格集合哈希不一致: "
            f"expected={expected_mesh_tree} actual={actual_mesh_tree}"
        )
    return urdf_path


BROWSER_MODEL_URL = _asset_url("g2-display", BROWSER_MODEL_GZIP, "glb")
MESHCAT_BROWSER_BUNDLE_URL = _asset_url(
    "meshcat-main-v0.3.2", MESHCAT_BROWSER_BUNDLE, "min.js"
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    return result


def _camera(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("camera 必须是对象")
    vectors: dict[str, list[float]] = {}
    for key, length in (("position", 3), ("quaternion", 4), ("target", 3)):
        raw = value.get(key)
        if not isinstance(raw, list) or len(raw) != length:
            raise TypeError(f"camera.{key} 必须是长度为 {length} 的数组")
        vectors[key] = [_finite(item, f"camera.{key}") for item in raw]
    norm = math.sqrt(sum(item * item for item in vectors["quaternion"]))
    if norm < 1e-8:
        raise ValueError("camera.quaternion 不能为零")
    vectors["quaternion"] = [item / norm for item in vectors["quaternion"]]
    fov = _finite(value.get("fov"), "camera.fov")
    zoom = _finite(value.get("zoom", 1.0), "camera.zoom")
    if not 10 <= fov <= 120:
        raise ValueError("camera.fov 必须在 [10, 120] 内")
    if not 0 < zoom <= 20:
        raise ValueError("camera.zoom 必须在 (0, 20] 内")
    return {**vectors, "fov": fov, "zoom": zoom}


class SimulationState:
    def __init__(
        self,
        model: G2Model,
        scene: MeshcatScene | None,
        *,
        meshcat_port: int,
        state_dir: Path,
        alignment_surfaces_directory: Path,
        parent_pid: int,
    ) -> None:
        self.model = model
        self.scene = scene
        self.meshcat_port = meshcat_port
        self.parent_pid = parent_pid
        self.state_dir = state_dir
        self.alignment_surfaces_directory = alignment_surfaces_directory
        self.lock = threading.RLock()
        self.revision = 0
        self.started_at = time.time()
        self.presets_path = state_dir / "simulator-camera-presets.json"
        self.presets = self._load_presets()
        self.work_surface = self._load_work_surface()
        self.fallback_error: str | None = None
        if self.scene is not None:
            self._configure_scene_surface(self.scene)

    def _configure_scene_surface(self, scene: MeshcatScene) -> None:
        scene.set_work_surface(
            height=self.work_surface["height"],
            hard_margin=self.work_surface["hard_margin"],
            origin=self.work_surface["origin"],
            normal=self.work_surface["normal"],
            xy_bounds=self.work_surface["xy_bounds"],
        )

    def attach_scene(self, scene: MeshcatScene) -> None:
        """Publish the optional MeshCat fallback after cached-viewer startup."""
        with self.lock:
            self._configure_scene_surface(scene)
            self.scene = scene
            self.fallback_error = None

    def set_fallback_error(self, error: BaseException) -> None:
        with self.lock:
            self.fallback_error = str(error)

    def _load_work_surface(self) -> dict[str, Any]:
        default = {
            "height": 0.75,
            "hard_margin": 0.004,
            "origin": [0.0, 0.0, 0.75],
            "normal": [0.0, 0.0, 1.0],
            "xy_bounds": None,
            "source": "simulation_default",
            "measurement_id": None,
            "safety_profile_id": None,
        }
        path = self.alignment_surfaces_directory / "standard_table" / "active.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            surface = value["surface"]
            height = _finite(surface["height_m"], "surface.height_m")
            pose = surface["pose"]
            origin = [
                _finite(pose["x_m"], "surface.pose.x_m"),
                _finite(pose["y_m"], "surface.pose.y_m"),
                _finite(pose["z_m"], "surface.pose.z_m"),
            ]
            normal = [
                _finite(item, f"surface.normal[{index}]")
                for index, item in enumerate(surface["normal"])
            ]
            if len(normal) != 3:
                raise ValueError("surface.normal 必须包含 3 个分量")
            normal_norm = math.sqrt(sum(item * item for item in normal))
            if normal_norm < 1e-8:
                raise ValueError("surface.normal 不能为零向量")
            normal = [item / normal_norm for item in normal]
            if (
                value.get("schema_version") != 1
                or value.get("base_frame") != "base_link"
                or surface.get("eligible") is not True
                or normal[2] <= 0.95
                or abs(origin[2] - height) > 0.002
            ):
                return default
        except (KeyError, OSError, TypeError, ValueError):
            return default
        measurement_id = str(value.get("measurement_id", "")).strip()
        margin = default["hard_margin"]
        xy_bounds = default["xy_bounds"]
        safety_profile_id = None
        profile_root = self.alignment_surfaces_directory.parent / "safety-profiles" / "g2a"
        try:
            candidates = sorted(
                profile_root.glob("*.json"),
                key=lambda candidate: candidate.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            candidates = []
        for candidate in candidates:
            try:
                profile = json.loads(candidate.read_text(encoding="utf-8"))
                provenance = profile.get("surface_calibration", {})
                surfaces = profile.get("surfaces", [])
                configured_surface = surfaces[0]
                configured_height = _finite(configured_surface["height"], "surfaces[0].height")
                configured_origin = [
                    _finite(item, f"surfaces[0].origin[{index}]")
                    for index, item in enumerate(configured_surface["origin"])
                ]
                configured_normal = [
                    _finite(item, f"surfaces[0].normal[{index}]")
                    for index, item in enumerate(configured_surface["normal"])
                ]
                configured_margin = _finite(
                    configured_surface["hard_margin"], "surfaces[0].hard_margin"
                )
                configured_bounds = [
                    _finite(item, f"surfaces[0].xy_bounds[{index}]")
                    for index, item in enumerate(configured_surface["xy_bounds"])
                ]
                if (
                    not measurement_id
                    or not isinstance(provenance, dict)
                    or provenance.get("measurement_id") != measurement_id
                    or abs(configured_height - height) > 1e-6
                    or len(configured_origin) != 3
                    or len(configured_normal) != 3
                    or any(abs(a - b) > 1e-6 for a, b in zip(configured_origin, origin))
                    or any(abs(a - b) > 1e-6 for a, b in zip(configured_normal, normal))
                    or len(configured_bounds) != 4
                    or configured_bounds[0] >= configured_bounds[1]
                    or configured_bounds[2] >= configured_bounds[3]
                    or not 0 < configured_margin <= 0.20
                ):
                    continue
            except (IndexError, KeyError, OSError, TypeError, ValueError):
                continue
            margin = configured_margin
            xy_bounds = configured_bounds
            safety_profile_id = candidate.name
            break
        return {
            **default,
            "height": height,
            "hard_margin": margin,
            "origin": origin,
            "normal": normal,
            "xy_bounds": xy_bounds,
            "source": "g2a_alignment_apriltag",
            "measurement_id": measurement_id or None,
            "observed_at": value.get("created_at"),
            "safety_profile_id": safety_profile_id,
        }

    def _load_presets(self) -> dict[str, dict[str, Any]]:
        candidates = (self.presets_path, PACKAGE_ROOT / "camera_presets.json")
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            result: dict[str, dict[str, Any]] = {}
            if isinstance(payload, dict):
                for view in ("near", "overview"):
                    try:
                        result[view] = _camera(payload.get(view))
                    except (TypeError, ValueError):
                        pass
            if result:
                return result
        return {}

    def _save_presets(self) -> None:
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.presets_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.presets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.presets_path)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "component": INTEGRATED_SIMULATOR_COMPONENT,
            "simulation_only": True,
            "robot_commands": "disabled",
            "renderer": "缓存 GLB / 浏览器 FK / MeshCat THREE",
            "meshcat_fallback_ready": self.scene is not None,
            "meshcat_fallback_error": self.fallback_error,
            "revision": self.revision,
            "parent_pid": self.parent_pid,
        }

    def bootstrap(self) -> dict[str, Any]:
        refreshed_surface = self._load_work_surface()
        if refreshed_surface != self.work_surface:
            self.work_surface = refreshed_surface
            if self.scene is not None:
                self._configure_scene_surface(self.scene)
        try:
            collision_overlay = json.loads(COLLISION_OVERLAY_CONFIG.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            collision_overlay = None
        return {
            **self.health(),
            "model": self.model.summary(),
            "viewer": {
                "meshcat_port": self.meshcat_port,
                "camera_presets": self.presets,
                "model_url": BROWSER_MODEL_URL,
                "renderer_url": MESHCAT_BROWSER_BUNDLE_URL,
                "cache_policy": IMMUTABLE_CACHE,
            },
            "work_surface": dict(self.work_surface),
            "collision_overlay": collision_overlay,
        }

    def update_joints(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.model.update_joints(payload.get("joints"))
            changed = self.scene.update_robot() if self.scene is not None else 0
            self.revision += 1
            return {"revision": self.revision, "changed_links": changed}

    def set_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        view = payload.get("view")
        if view not in {"near", "overview"}:
            raise ValueError("view 必须是 near 或 overview")
        with self.lock:
            if self.scene is not None:
                self.scene.set_camera_view(str(view))
            return {"camera": view, "preset": self.presets.get(str(view))}

    def save_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        view = payload.get("view")
        if view not in {"near", "overview"}:
            raise ValueError("view 必须是 near 或 overview")
        preset = _camera(payload.get("camera"))
        with self.lock:
            self.presets[str(view)] = preset
            self._save_presets()
            self.revision += 1
            return {"view": view, "preset": preset, "revision": self.revision}


def _handler_class(state: SimulationState) -> type[BaseHTTPRequestHandler]:
    static_files = {
        "/viewer/": ("viewer.html", "text/html; charset=utf-8"),
        "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
        "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "NexusIntegratedSimulator/1.0"

        def _send(
            self,
            body: bytes,
            mime: str,
            status: int = 200,
            *,
            cache_control: str = "no-store",
            content_encoding: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.send_header("ETag", _etag(body))
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'self'; script-src 'self' http://*:{state.meshcat_port}; "
                f"style-src 'self' 'unsafe-inline'; connect-src 'self' ws://*:{state.meshcat_port}; "
                "img-src 'self' data:; frame-ancestors http: https:",
            )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, value: dict[str, Any], status: int = 200) -> None:
            self._send(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def _payload(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Content-Length 无效") from exc
            if not 0 < length <= 64 * 1024:
                raise ValueError("JSON 请求体必须在 1 到 65536 字节之间")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise TypeError("JSON 根节点必须是对象")
            return payload

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/healthz":
                self._json(state.health())
                return
            if path == "/api/bootstrap":
                self._json({"ok": True, "data": state.bootstrap()})
                return
            if path == BROWSER_MODEL_URL:
                self._send(
                    BROWSER_MODEL_GZIP.read_bytes(),
                    "model/gltf-binary",
                    cache_control=IMMUTABLE_CACHE,
                    content_encoding="gzip",
                )
                return
            if path == MESHCAT_BROWSER_BUNDLE_URL:
                self._send(
                    MESHCAT_BROWSER_BUNDLE_GZIP,
                    "text/javascript; charset=utf-8",
                    cache_control=IMMUTABLE_CACHE,
                    content_encoding="gzip",
                )
                return
            static = static_files.get(path)
            if static is None:
                self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            name, mime = static
            self._send((STATIC_ROOT / name).read_bytes(), mime)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            try:
                payload = self._payload()
                if path == "/api/joints":
                    data = state.update_joints(payload)
                elif path == "/api/camera/view":
                    data = state.set_view(payload)
                elif path == "/api/camera/preset":
                    data = state.save_view(payload)
                else:
                    self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
            except (json.JSONDecodeError, ModelError, TypeError, ValueError, ViewerError) as exc:
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "data": data})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(
    host: str,
    port: int,
    meshcat_port: int,
    state_dir: Path,
    alignment_surfaces_directory: Path,
    parent_pid: int,
) -> int:
    bridge: MeshcatBridge | None = None
    state: SimulationState | None = None
    server: ThreadingHTTPServer | None = None
    stop_event = threading.Event()
    try:
        if not BROWSER_MODEL_GZIP.is_file():
            raise OSError(f"缓存 G2 浏览器模型不存在: {BROWSER_MODEL_GZIP}")
        if not COLLISION_OVERLAY_CONFIG.is_file():
            raise OSError(f"G2 碰撞标定叠加配置不存在: {COLLISION_OVERLAY_CONFIG}")
        if not MESHCAT_BROWSER_BUNDLE.is_file():
            raise OSError(f"MeshCat 浏览器渲染器不存在: {MESHCAT_BROWSER_BUNDLE}")
        model = G2Model(_validate_canonical_model_assets())
        bridge = MeshcatBridge.start(host, meshcat_port)
        state = SimulationState(
            model,
            None,
            meshcat_port=meshcat_port,
            state_dir=state_dir,
            alignment_surfaces_directory=alignment_surfaces_directory,
            parent_pid=parent_pid,
        )
        server = ThreadingHTTPServer((host, port), _handler_class(state))
    except (ModelError, OSError, ViewerError) as exc:
        if bridge is not None:
            bridge.close()
        print(f"Nexus 内置仿真启动失败: {exc}", flush=True)
        return 2

    def load_meshcat_fallback() -> None:
        try:
            fallback_scene = MeshcatScene(bridge, model)
            if stop_event.is_set():
                fallback_scene.close()
                return
            state.attach_scene(fallback_scene)
        except Exception as exc:  # noqa: BLE001 - fallback must not stop cached rendering
            state.set_fallback_error(exc)
            print(f"Nexus MeshCat 兼容回退初始化失败: {exc}", flush=True)

    threading.Thread(
        target=load_meshcat_fallback,
        name="nexus-meshcat-fallback-loader",
        daemon=True,
    ).start()

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def monitor_parent() -> None:
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            except PermissionError:
                pass
            time.sleep(1.0)

    threading.Thread(target=monitor_parent, name="nexus-parent-watch", daemon=True).start()
    print(
        f"Nexus 内置仿真已启动: control={host}:{port} meshcat={host}:{meshcat_port} "
        "robot_commands=disabled",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop_event.set()
        server.server_close()
        with state.lock:
            scene = state.scene
        if scene is not None:
            scene.close()
        else:
            bridge.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--meshcat-port", type=int, default=8767)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--alignment-surfaces-directory", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(
        serve(
            args.host,
            args.port,
            args.meshcat_port,
            args.state_dir,
            args.alignment_surfaces_directory,
            args.parent_pid,
        )
    )


if __name__ == "__main__":
    main()
