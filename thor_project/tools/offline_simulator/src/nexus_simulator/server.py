"""阶段 1 HTTP 控制页和 MeshCat 可视化服务。"""

from __future__ import annotations

import json
import math
import os
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nexus_simulator.assets import AssetError, sha256_file, verify_assets
from nexus_simulator.config import SimulationConfig, SimulationConfigError, load_simulation_config
from nexus_simulator.dataset import G2_URDF_JOINTS, DatasetError, EpisodeData, LeRobotDataset
from nexus_simulator.meshcat_scene import MeshcatBridge, MeshcatScene, ViewerError
from nexus_simulator.model import G2Model, ModelError
from nexus_simulator.model_variants import (
    LEGACY_MODEL_ID,
    ModelVariantError,
    ModelVariantSpec,
    model_variant_spec,
    resolve_model_variant,
)
from nexus_simulator.safety_reference import SafetyReference, SafetyReferenceError


class SimulationState:
    def __init__(
        self,
        *,
        project_root: Path,
        config: SimulationConfig,
        model: G2Model,
        scene: MeshcatScene,
        meshcat_port: int,
        asset_manifest: dict[str, Any],
        episode: EpisodeData,
        safety_reference: SafetyReference,
        model_variant: ModelVariantSpec | None = None,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.model = model
        self.scene = scene
        self.meshcat_port = meshcat_port
        self.asset_manifest = asset_manifest
        self.episode = episode
        self.safety_reference = safety_reference
        self.model_variant = model_variant or model_variant_spec(LEGACY_MODEL_ID)
        self.lock = threading.RLock()
        self.surface_height = config.work_surface.height
        self.hard_margin = config.work_surface.hard_margin
        self.safety_limit_enabled = config.guard.enabled
        self.collision_warning_active = False
        self.revision = 0
        self.replay_frame_index = 0
        self.replay_source = "observation"
        self.last_displayed_vectors: dict[str, Any] = {}
        self.last_safe_vectors: dict[str, Any] = {}
        self.camera_presets_path = self.project_root / "configs" / "camera_presets.json"
        self.camera_presets = self._load_camera_presets()

    def _load_camera_presets(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self.camera_presets_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        presets: dict[str, dict[str, Any]] = {}
        for view in ("near", "overview"):
            camera = payload.get(view)
            try:
                presets[view] = _validate_camera(camera)
            except (TypeError, ValueError):
                continue
        return presets

    def _persist_camera_presets(self) -> None:
        self.camera_presets_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.camera_presets_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.camera_presets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.camera_presets_path)

    def bootstrap(self) -> dict[str, Any]:
        with self.lock:
            model_summary = self.model.summary()
            model_summary["variant"] = self.model_variant.metadata()
            return {
                "project": {
                    "name": "Nexus Simulator",
                    "version": "0.1.0",
                    "stage": "URDF/FK 浏览器可视化可行性验证",
                },
                "safety": {
                    "simulation_only": True,
                    "calibrated": False,
                    "deployable": False,
                    "hardware_access": "disabled",
                    "gdk_import": "forbidden",
                    "robot_commands": "disabled",
                    "enforcement_enabled": self.safety_limit_enabled,
                    "detection_enabled": True,
                    "visual_collision_geometry_aligned": (
                        self.model_variant.safety_geometry_aligned
                    ),
                },
                "viewer": {
                    "meshcat_port": self.meshcat_port,
                    "path": "/viewer/",
                    "camera_presets": self.camera_presets,
                },
                "model": model_summary,
                "joints": [info.as_dict() for info in self.model.joint_infos()],
                "work_surface": {
                    "height": self.surface_height,
                    "hard_margin": self.hard_margin,
                    "xy_bounds": None,
                },
                "assets": {
                    "file_count": self.asset_manifest["file_count"],
                    "urdf_sha256": sha256_file(self.model.urdf_path),
                },
                "replay": self.episode.metadata(),
                "revision": self.revision,
            }

    def replay_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_index = payload.get("frame_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError("frame_index 必须是整数")
        source = payload.get("source", "observation")
        if not isinstance(source, str):
            raise TypeError("source 必须是字符串")
        raw_vector = self.episode.vector(raw_index, source)
        raw_previous = self.episode.vector(max(0, raw_index - 1), source)
        vector = self.episode.to_safety_vector(raw_vector, source)
        previous_recorded = self.episode.to_safety_vector(raw_previous, source)
        with self.lock:
            if raw_index == 0:
                self.last_displayed_vectors.pop(source, None)
                self.last_safe_vectors.pop(source, None)
            last_safe = self.last_safe_vectors.get(source)
            measured = self.last_displayed_vectors.get(source, previous_recorded)
            if self.safety_limit_enabled and last_safe is not None:
                # 如果限制关闭时曾显示危险帧，重新开启后从最后验证安全的姿态恢复，
                # 而不是把已经危险的显示姿态继续当作控制起点。
                measured = last_safe
            safety, displayed_vector = self.safety_reference.evaluate_action(
                measured,
                vector,
                last_safe if last_safe is not None else previous_recorded,
                height=self.surface_height,
                hard_margin=self.hard_margin,
                enforcement_enabled=self.safety_limit_enabled,
            )
            if not safety["collision_detected"] or (
                self.safety_limit_enabled and safety["adopted_pose_safe"]
            ):
                self.last_safe_vectors[source] = displayed_vector.copy()
            self.last_displayed_vectors[source] = displayed_vector.copy()
            collision_detected = bool(safety["collision_detected"])
            if collision_detected and not self.collision_warning_active:
                enforcement = "ON" if self.safety_limit_enabled else "OFF"
                print(
                    "[nexus-sim-safety] WARNING "
                    f"collision_detected frame={raw_index} source={source} "
                    f"enforcement={enforcement} reason={safety['reason']}"
                )
            self.collision_warning_active = collision_detected
            displayed_source_vector = self.episode.from_safety_vector(displayed_vector, source)
            values = {
                name: float(displayed_source_vector[index])
                for index, name in enumerate(G2_URDF_JOINTS)
            }
            infos = self.model.update_joints(
                self.model_variant.map_replay_joint_values(
                    values,
                    effector_encoding=self.episode.effector_encoding,
                    source=source,
                )
            )
            self.scene.update_robot()
            self.replay_frame_index = raw_index
            self.replay_source = source
            self.revision += 1
            return {
                "frame_index": raw_index,
                "timestamp_s": float(self.episode.timestamps[raw_index]),
                "source": source,
                "joints": [info.as_dict() for info in infos],
                "safety": safety,
                "revision": self.revision,
            }

    def update_safety_limit(self, payload: dict[str, Any]) -> dict[str, Any]:
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise TypeError("enabled 必须是布尔值")
        with self.lock:
            self.safety_limit_enabled = enabled
            self.collision_warning_active = False
            self.revision += 1
            return {
                "enforcement_enabled": self.safety_limit_enabled,
                "detection_enabled": True,
                "revision": self.revision,
            }

    def update_joints(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = payload.get("joints")
        with self.lock:
            infos = self.model.update_joints(values)
            self.scene.update_robot()
            self.revision += 1
            return {
                "joints": [info.as_dict() for info in infos],
                "revision": self.revision,
            }

    def set_camera_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        view = payload.get("view")
        if view not in {"near", "overview"}:
            raise ValueError("view 必须是 near 或 overview")
        with self.lock:
            self.scene.set_camera_view(view)
            return {"camera": view, "preset": self.camera_presets.get(view)}

    def save_camera_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        view = payload.get("view")
        if view not in {"near", "overview"}:
            raise ValueError("view 必须是 near 或 overview")
        camera = _validate_camera(payload.get("camera"))
        with self.lock:
            self.camera_presets[view] = camera
            self._persist_camera_presets()
            self.revision += 1
            return {"view": view, "preset": camera, "revision": self.revision}

    def reset(self) -> dict[str, Any]:
        with self.lock:
            infos = self.model.reset()
            self.scene.update_robot()
            self.revision += 1
            return {
                "joints": [info.as_dict() for info in infos],
                "revision": self.revision,
            }

    def update_surface(self, payload: dict[str, Any]) -> dict[str, Any]:
        height = _finite(payload.get("height"), "height")
        margin = _finite(payload.get("hard_margin"), "hard_margin")
        if height < 0.2 or height > 1.5:
            raise ValueError("height 必须在 [0.2, 1.5] m")
        if margin < 0.005 or margin > 0.2:
            raise ValueError("hard_margin 必须在 [0.005, 0.2] m")
        with self.lock:
            self.surface_height = height
            self.hard_margin = margin
            # 工作面变化后，旧姿态不能继续被视为已验证安全。
            self.last_displayed_vectors.clear()
            self.last_safe_vectors.clear()
            self.scene.set_work_surface(height=height, hard_margin=margin)
            self.revision += 1
            return {
                "work_surface": {
                    "height": self.surface_height,
                    "hard_margin": self.hard_margin,
                    "xy_bounds": None,
                },
                "revision": self.revision,
            }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    return result


def _finite_vector(value: Any, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(f"{name} 必须是长度为 {length} 的数组")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _validate_camera(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("camera 必须是对象")
    position = _finite_vector(value.get("position"), "camera.position", 3)
    quaternion = _finite_vector(value.get("quaternion"), "camera.quaternion", 4)
    target = _finite_vector(value.get("target"), "camera.target", 3)
    quaternion_norm = math.sqrt(sum(item * item for item in quaternion))
    if quaternion_norm < 1e-8:
        raise ValueError("camera.quaternion 不能为零")
    quaternion = [item / quaternion_norm for item in quaternion]
    fov = _finite(value.get("fov"), "camera.fov")
    if not 10 <= fov <= 120:
        raise ValueError("camera.fov 必须在 [10, 120] 内")
    zoom = _finite(value.get("zoom", 1.0), "camera.zoom")
    if not 0 < zoom <= 20:
        raise ValueError("camera.zoom 必须在 (0, 20] 内")
    return {
        "position": position,
        "quaternion": quaternion,
        "target": target,
        "fov": fov,
        "zoom": zoom,
    }


def _handler_class(state: SimulationState) -> type[BaseHTTPRequestHandler]:
    static_root = files("nexus_simulator").joinpath("static")
    static_files = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/favicon.svg": ("favicon.svg", "image/svg+xml"),
        "/viewer/": ("viewer.html", "text/html; charset=utf-8"),
        "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
        "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "NexusSimulator/0.1"

        def _headers(
            self, status: int, content_type: str, length: int, *, embeddable: bool = False
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if embeddable:
                meshcat_origin = f"{urlsplit(self.path).scheme or 'http'}://*:{state.meshcat_port}"
                self.send_header(
                    "Content-Security-Policy",
                    f"default-src 'self'; script-src 'self' {meshcat_origin}; "
                    f"style-src 'self' 'unsafe-inline'; connect-src 'self' "
                    f"ws://*:{state.meshcat_port}; img-src 'self' data:; "
                    "frame-ancestors 'self' http://*:8765",
                )
            else:
                self.send_header("X-Frame-Options", "SAMEORIGIN")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; frame-src http: https:; img-src 'self' data:",
                )
            self.end_headers()

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: int = 200,
            *,
            embeddable: bool = False,
        ) -> None:
            self._headers(status, content_type, len(body), embeddable=embeddable)
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_video(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header:
                if not range_header.startswith("bytes=") or "," in range_header:
                    self._error("invalid range", HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                first, _, last = range_header[6:].partition("-")
                try:
                    if first:
                        start = int(first)
                        end = int(last) if last else end
                    else:
                        suffix = int(last)
                        start = max(0, size - suffix)
                except ValueError:
                    self._error("invalid range", HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                if start < 0 or start >= size or end < start:
                    self._error("invalid range", HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status)

        def _error(self, message: str, status: int = 400) -> None:
            self._json({"ok": False, "error": message}, status)

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "0")
            except ValueError as exc:
                raise ValueError("Content-Length 无效") from exc
            if length <= 0 or length > 64 * 1024:
                raise ValueError("JSON 请求体必须在 1 到 65536 字节之间")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ValueError("JSON 请求体无效") from exc
            if not isinstance(payload, dict):
                raise TypeError("JSON 根节点必须是对象")
            return payload

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/healthz":
                self._json({"ok": True, "simulation_only": True})
                return
            if path == "/api/bootstrap":
                self._json({"ok": True, "data": state.bootstrap()})
                return
            if path == "/api/replay/video/head":
                self._send_video(state.episode.video_path)
                return
            static = static_files.get(path)
            if static is None:
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            name, content_type = static
            self._send_bytes(
                static_root.joinpath(name).read_bytes(),
                content_type,
                embeddable=path.startswith("/viewer"),
            )

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/joints":
                    result = state.update_joints(payload)
                elif path == "/api/reset":
                    result = state.reset()
                elif path == "/api/work-surface":
                    result = state.update_surface(payload)
                elif path == "/api/safety-limit":
                    result = state.update_safety_limit(payload)
                elif path == "/api/replay/frame":
                    result = state.replay_frame(payload)
                elif path == "/api/camera/view":
                    result = state.set_camera_view(payload)
                elif path == "/api/camera/preset":
                    result = state.save_camera_preset(payload)
                else:
                    self._error("not found", HTTPStatus.NOT_FOUND)
                    return
            except (ModelError, TypeError, ValueError) as exc:
                self._error(str(exc), HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "data": result})

        def log_message(self, format: str, *args: object) -> None:
            request_line = str(args[0]) if args else ""
            if "/api/replay/frame" in request_line or "/api/replay/video/head" in request_line:
                return
            print(f"[nexus-sim-http] {self.address_string()} {format % args}")

    return Handler


def serve(
    *,
    project_root: Path,
    config_path: Path,
    host: str,
    port: int,
    meshcat_port: int,
    dataset_root: Path,
    episode_index: int,
    robot_model: str = LEGACY_MODEL_ID,
) -> int:
    root = project_root.expanduser().resolve()
    config_file = config_path if config_path.is_absolute() else root / config_path
    bridge: MeshcatBridge | None = None
    scene: MeshcatScene | None = None
    try:
        config = load_simulation_config(config_file)
        legacy_assets = verify_assets(root)
        model_variant = resolve_model_variant(root, robot_model)
        urdf = model_variant.urdf_path(root)
        model = G2Model(urdf, asset_root=model_variant.asset_root(root))
        episode = LeRobotDataset(dataset_root).load_episode(episode_index)
        safety_reference = SafetyReference(root / "assets" / "g2" / "G2_omnipicker")
        bridge = MeshcatBridge.start(host=host, port=meshcat_port)
        scene = MeshcatScene(bridge, model)
        scene.set_work_surface(
            height=config.work_surface.height,
            hard_margin=config.work_surface.hard_margin,
        )
        state = SimulationState(
            project_root=root,
            config=config,
            model=model,
            scene=scene,
            meshcat_port=meshcat_port,
            asset_manifest={
                "file_count": model_variant.asset_file_count,
                "safety_file_count": legacy_assets["file_count"],
            },
            episode=episode,
            safety_reference=safety_reference,
            model_variant=model_variant,
        )
        server = ThreadingHTTPServer((host, port), _handler_class(state))
    except (
        AssetError,
        DatasetError,
        ModelError,
        ModelVariantError,
        SafetyReferenceError,
        SimulationConfigError,
        ViewerError,
        OSError,
    ) as exc:
        if scene is not None:
            scene.close()
        elif bridge is not None:
            bridge.close()
        print(f"Nexus Simulator serve: FAILED: {exc}")
        return 2

    stop_event = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        if not stop_event.is_set():
            stop_event.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print("Nexus Simulator: SIMULATION ONLY — ROBOT COMMANDS DISABLED")
    print(f"Control UI: http://{display_host}:{port}")
    print(f"MeshCat UI: http://{display_host}:{meshcat_port}/static/")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        scene.close()
    return 0
