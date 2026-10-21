"""Robot-side, read-only HTTP facade for the G2 GDK adapter.

This process is optional and independent from the inference client. It uses the
same robot process lock, so it cannot accidentally share GDK with a running task.
"""

from __future__ import annotations

import argparse
import fcntl
import hmac
import io
import ipaddress
import json
import math
import multiprocessing
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:
    import numpy as np

MAX_FRAME_BYTES = 16 * 1024 * 1024
DEFAULT_ROBOT_LOCK_PATH = Path("/tmp/g2a_pi05_robot_gdk.lock")

# Public alignment camera names map to the GDK image stream and its matching
# sensor-to-head transform. Keeping the two values together prevents a frame
# from ever being paired with another camera's calibration.
GDK_CAMERA_SPECS: Mapping[str, tuple[str, str, str]] = {
    "cam_head": ("kHeadColor", "kHeadRGBDToHeadLink3", "头部彩色相机"),
    "head_stereo_left": (
        "kHeadStereoLeft",
        "kHeadLeftStereoToHeadLink3",
        "头部左双目相机",
    ),
    "head_stereo_right": (
        "kHeadStereoRight",
        "kHeadRightStereoToHeadLink3",
        "头部右双目相机",
    ),
}


def _robot_asset_id(path: Path = Path("/data/version/aid_info.yaml")) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("AID:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


class G2ReadOnlySource:
    def __init__(
        self,
        calibration_path: Path,
        *,
        fps: float,
        camera_timeout_ms: int,
        idle_release_s: float,
        robot_lock_path: Path = DEFAULT_ROBOT_LOCK_PATH,
    ) -> None:
        calibration = json.loads(calibration_path.expanduser().read_text(encoding="utf-8"))
        if not isinstance(calibration, Mapping):
            raise ValueError("camera calibration must be a JSON object")
        self.calibration_data = dict(calibration)
        self.camera_name = str(calibration.get("camera_name", "cam_head"))
        self.fps = float(fps)
        self.camera_timeout_ms = int(camera_timeout_ms)
        self.idle_release_s = float(idle_release_s)
        self.robot_lock_path = robot_lock_path
        self.started_at = time.time()
        self._lifecycle_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._process: Any = None
        self._connection: Any = None
        self._release_hold_until = 0.0

    def ensure_started(self) -> None:
        with self._lifecycle_lock:
            release_hold_remaining = self._release_hold_until - time.monotonic()
            if release_hold_remaining > 0:
                raise RuntimeError(
                    "Alignment GDK is temporarily released for a robot task "
                    f"({release_hold_remaining:.1f}s remaining)"
                )
            if self._process is not None and self._process.is_alive():
                return
            self._close_worker(force=False)
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(
                target=_gdk_worker,
                args=(
                    child,
                    self.camera_name,
                    self.fps,
                    self.camera_timeout_ms,
                    self.idle_release_s,
                ),
                name="AlignmentGDK",
                daemon=True,
            )
            process.start()
            child.close()
            self._process, self._connection = process, parent
            startup_timeout = max(12.0, self.camera_timeout_ms / 1000 + 8.0)
            if not parent.poll(startup_timeout):
                self._close_worker(force=True)
                raise RuntimeError("GDK alignment worker startup timed out")
            result = parent.recv()
            if not isinstance(result, Mapping) or not result.get("ok"):
                detail = (
                    result.get("error", "GDK alignment worker failed")
                    if isinstance(result, Mapping)
                    else "GDK alignment worker failed"
                )
                self._close_worker(force=False)
                raise RuntimeError(str(detail))

    def _request(self, command: str) -> Any:
        with self._rpc_lock:
            self.ensure_started()
            connection = self._connection
            process = self._process
            if connection is None or process is None:
                raise RuntimeError("GDK alignment worker is unavailable")
            try:
                connection.send(command)
                if not connection.poll(max(10.0, self.camera_timeout_ms / 1000 + 5.0)):
                    raise RuntimeError(f"GDK alignment worker timed out handling {command}")
                result = connection.recv()
            except (BrokenPipeError, EOFError, OSError) as error:
                self._close_worker(force=False)
                raise RuntimeError("GDK alignment worker disconnected") from error
            if not isinstance(result, Mapping) or not result.get("ok"):
                detail = (
                    result.get("error", "GDK request failed")
                    if isinstance(result, Mapping)
                    else "GDK request failed"
                )
                raise RuntimeError(str(detail))
            return result.get("value")

    def info(self) -> dict[str, Any]:
        return {
            "id": str(self.calibration_data.get("robot_id", "g2a")),
            "aid": _robot_asset_id(),
            "model": str(self.calibration_data.get("robot_model", "G2A Logistics")),
            "cameras": [self.camera_name],
        }

    def health(self) -> dict[str, Any]:
        """Report shell/lock health without importing or starting GDK."""
        with self._lifecycle_lock:
            worker_active = bool(self._process is not None and self._process.is_alive())
        lock_busy, lock_error = _robot_gdk_lock_busy(self.robot_lock_path)
        value: dict[str, Any] = {
            "status": "degraded" if lock_error else "ok",
            # Our own worker holding the shared lock is not an external task.
            "task_running": bool(lock_busy and not worker_active),
            "gdk_state": "active" if worker_active else "idle",
            "idle_release_s": self.idle_release_s,
            "release_hold_remaining_s": round(
                max(0.0, self._release_hold_until - time.monotonic()), 1
            ),
            "uptime_s": round(time.time() - self.started_at, 1),
        }
        if lock_error:
            value["error"] = lock_error
        return value

    def deep_health(self) -> dict[str, Any]:
        """Explicitly initialize GDK and verify its live read path."""
        worker = dict(self._request("health"))
        value = self.health()
        value.update(worker)
        value["gdk_state"] = "active"
        return value

    def state(self) -> dict[str, Any]:
        value = self._request("state")
        if not isinstance(value, Mapping):
            raise RuntimeError("GDK alignment worker returned a malformed state")
        # Older workers returned the head-joint mapping directly. Keep that
        # representation readable while exposing the complete canonical state
        # from new workers for read-only geometry/contact calibration.
        if "head_joints" not in value:
            return {"head_joints": dict(value)}
        return dict(value)

    def calibration(self) -> dict[str, Any]:
        value = dict(self._request("calibration"))
        value["version"] = str(self.calibration_data.get("version", "gdk-live-v1"))
        return value

    def frame(self) -> tuple[str, bytes, int, int]:
        value = self._request("frame")
        if not isinstance(value, tuple) or len(value) != 4:
            raise RuntimeError("GDK alignment worker returned a malformed frame")
        return str(value[0]), bytes(value[1]), int(value[2]), int(value[3])

    def close(self) -> None:
        self.release_worker()

    def release_worker(self, *, hold_s: float = 0.0) -> bool:
        """Release this facade's GDK child while keeping HTTP available."""
        hold_s = float(hold_s)
        if not math.isfinite(hold_s) or not 0.0 <= hold_s <= 60.0:
            raise ValueError("hold_s must be within [0, 60]")
        with self._rpc_lock, self._lifecycle_lock:
            self._release_hold_until = max(
                self._release_hold_until, time.monotonic() + hold_s
            )
            had_worker = self._process is not None
            connection = self._connection
            if connection is not None:
                try:
                    connection.send("stop")
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._close_worker(force=True)
            return had_worker

    def _close_worker(self, *, force: bool) -> None:
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=2.0)
            if force and process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def _gdk_worker(
    connection: Any,
    camera_name: str,
    fps: float,
    camera_timeout_ms: int,
    idle_release_s: float,
) -> None:
    adapter: Any = None
    try:
        from robot_client.gdk_adapter import AdapterConfig, G2AAdapter

        adapter = G2AAdapter(
            AdapterConfig(
                hardware_mode="read-only",
                camera_timeout_ms=camera_timeout_ms,
                resize_wrist_images=False,
            )
        )
        # The existing process lock prevents overlap with inference/reset.
        adapter.connect()
        tf = adapter._gdk.TF()
        frame, state_snapshot = _capture_gdk(adapter, camera_name, camera_timeout_ms)
        error: str | None = None
        connection.send({"ok": True})
        last_request = time.monotonic()
        next_capture = last_request + 1.0 / max(1.0, fps)
        while True:
            now = time.monotonic()
            if now - last_request >= idle_release_s:
                break
            wait = min(max(0.0, next_capture - now), max(0.0, idle_release_s - (now - last_request)))
            if connection.poll(wait):
                command = connection.recv()
                last_request = time.monotonic()
                if command == "stop":
                    break
                try:
                    if command == "health":
                        value = {
                            "status": "ok" if error is None else "degraded",
                            "task_running": False,
                            "last_frame_at": frame[4],
                            "error": error,
                        }
                    elif command == "state":
                        value = state_snapshot
                    elif command == "calibration":
                        value = _gdk_calibration_with_retry(
                            adapter,
                            tf,
                            camera_name,
                            timeout_s=4.0,
                        )
                    elif command == "frame":
                        value = (frame[0], frame[1], frame[2], frame[3])
                    else:
                        raise RuntimeError(f"unsupported worker command: {command}")
                    connection.send({"ok": True, "value": value})
                except Exception as request_error:
                    connection.send({"ok": False, "error": str(request_error)})
            now = time.monotonic()
            if now >= next_capture:
                try:
                    frame, state_snapshot = _capture_gdk(
                        adapter, camera_name, camera_timeout_ms
                    )
                    error = None
                except Exception as capture_error:
                    error = str(capture_error)
                next_capture = time.monotonic() + 1.0 / max(1.0, fps)
    except Exception as startup_error:
        try:
            connection.send({"ok": False, "error": str(startup_error)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if adapter is not None:
            adapter.disconnect()
        connection.close()


def _capture_gdk(
    adapter: Any,
    camera_name: str,
    camera_timeout_ms: int,
) -> tuple[tuple[str, bytes, int, int, float], dict[str, Any]]:
    camera_type, _sensor_extrinsic_type, _display_name = _gdk_camera_spec(
        adapter._gdk, camera_name
    )
    image = adapter._camera.get_latest_image(
        camera_type,
        camera_timeout_ms,
    )
    if image is None:
        raise RuntimeError(f"No image received from GDK camera {camera_name}")
    payload = _gdk_image_payload(image, adapter._gdk)
    state = adapter.read_state()
    from robot_client.constants import (
        CANONICAL_STATE_NAMES,
        HEAD_CANONICAL_INDICES,
        HEAD_JOINTS,
    )

    head_values = state[list(HEAD_CANONICAL_INDICES)]
    captured_at = time.time()
    frame = (*_browser_image(payload), captured_at)
    head_joints = {
        name: float(value)
        for name, value in zip(HEAD_JOINTS, head_values, strict=True)
    }
    canonical_state = {
        name: float(value)
        for name, value in zip(CANONICAL_STATE_NAMES, state, strict=True)
    }
    return frame, {
        "head_joints": head_joints,
        "canonical_state": canonical_state,
        "captured_at": captured_at,
    }


def _gdk_calibration(adapter: Any, tf: Any, camera_name: str) -> dict[str, Any]:
    camera_type, sensor_extrinsic_type, display_name = _gdk_camera_spec(
        adapter._gdk, camera_name
    )
    intrinsic = adapter._camera.get_camera_intrinsic(camera_type)
    params = list(intrinsic.intrinsic)
    if len(params) != 4:
        raise RuntimeError(f"GDK returned malformed intrinsics for {camera_name}")
    width, height = adapter._camera.get_image_shape(camera_type)
    base_from_head = _transform_matrix(tf.get_tf_from_base_link("head_link3"))
    head_from_camera = _transform_matrix(
        tf.get_tf_from_sensor(sensor_extrinsic_type)
    )
    return {
        "camera_name": camera_name,
        "camera_display_name": display_name,
        "image_width": int(width),
        "image_height": int(height),
        "camera_matrix": [
            [float(params[0]), 0.0, float(params[2])],
            [0.0, float(params[1]), float(params[3])],
            [0.0, 0.0, 1.0],
        ],
        "distortion": [float(value) for value in intrinsic.distortion],
        # Keep the two factors separate for multi-pose camera-extrinsic
        # calibration. Consumers must not derive either factor back from the
        # composed transform, because head_from_camera is the quantity being
        # calibrated.
        "base_from_head_link3": base_from_head.tolist(),
        "head_link3_from_camera_factory": head_from_camera.tolist(),
        "base_from_camera": (base_from_head @ head_from_camera).tolist(),
        "intrinsic_source": (
            f"GDK Camera.get_camera_intrinsic(CameraType.{camera_type.name})"
        ),
        "extrinsic_source": (
            "GDK TF.get_tf_from_base_link(head_link3) × "
            f"TF.get_tf_from_sensor(SensorExtrinsicType.{sensor_extrinsic_type.name})"
        ),
    }


def _gdk_calibration_with_retry(
    adapter: Any,
    tf: Any,
    camera_name: str,
    *,
    timeout_s: float,
    retry_interval_s: float = 0.1,
) -> dict[str, Any]:
    """Wait briefly for the GDK TF tree after a cold read-only startup."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_error: Exception | None = None
    while True:
        try:
            return _gdk_calibration(adapter, tf, camera_name)
        except Exception as error:
            last_error = error
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"GDK camera TF was not ready within {timeout_s:.1f}s: {last_error}"
                ) from error
            time.sleep(max(0.0, retry_interval_s))


def _gdk_camera_spec(gdk: Any, camera_name: str) -> tuple[Any, Any, str]:
    spec = GDK_CAMERA_SPECS.get(camera_name)
    if spec is None:
        supported = ", ".join(sorted(GDK_CAMERA_SPECS))
        raise RuntimeError(
            f"unsupported alignment camera: {camera_name}; supported: {supported}"
        )
    camera_type_name, sensor_extrinsic_name, display_name = spec
    return (
        getattr(gdk.CameraType, camera_type_name),
        getattr(gdk.SensorExtrinsicType, sensor_extrinsic_name),
        display_name,
    )


def _gdk_image_payload(image: Any, gdk: Any) -> dict[str, Any]:
    if image.encoding == gdk.Encoding.JPEG:
        encoding, color_format = "jpeg", "encoded"
    elif image.encoding == gdk.Encoding.UNCOMPRESSED:
        encoding = "raw"
        if image.color_format == gdk.ColorFormat.RGB:
            color_format = "rgb"
        elif image.color_format == gdk.ColorFormat.BGR:
            color_format = "bgr"
        else:
            raise RuntimeError(f"unsupported GDK color format: {image.color_format!r}")
    else:
        raise RuntimeError(f"unsupported GDK image encoding: {image.encoding!r}")
    return {
        "encoding": encoding,
        "color_format": color_format,
        "width": int(image.width),
        "height": int(image.height),
        "data": bytes(image.data),
    }


def _transform_matrix(transform: Any) -> np.ndarray:
    """Convert a GDK Transform into a homogeneous parent-from-child matrix."""
    import numpy as np

    x = float(transform.rotation.x)
    y = float(transform.rotation.y)
    z = float(transform.rotation.z)
    w = float(transform.rotation.w)
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm <= 1e-12:
        raise RuntimeError("GDK returned a zero-length transform quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = (
        float(transform.translation.x),
        float(transform.translation.y),
        float(transform.translation.z),
    )
    return matrix


def _browser_image(payload: Mapping[str, Any]) -> tuple[str, bytes, int, int]:
    import numpy as np

    width, height = int(payload.get("width", 0)), int(payload.get("height", 0))
    data = payload.get("data")
    if width <= 0 or height <= 0 or not isinstance(data, bytes) or len(data) > MAX_FRAME_BYTES:
        raise RuntimeError("GDK returned a malformed or oversized camera frame")
    encoding = str(payload.get("encoding", "")).lower()
    if encoding in {"jpeg", "jpg"}:
        return "image/jpeg", data, width, height
    if encoding != "raw":
        raise RuntimeError(f"unsupported camera encoding: {encoding}")
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    color = str(payload.get("color_format", "")).lower()
    if color == "bgr":
        pixels = pixels[:, :, ::-1]
    elif color != "rgb":
        raise RuntimeError(f"unsupported raw camera color format: {color}")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required when the GDK camera returns raw frames") from error
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels), mode="RGB").save(output, format="JPEG", quality=90)
    return "image/jpeg", output.getvalue(), width, height


def _robot_gdk_lock_busy(path: Path) -> tuple[bool, str | None]:
    """Probe the shared lock without loading GDK or disturbing its owner."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+", encoding="utf-8")
    except OSError as error:
        return False, f"cannot inspect robot GDK lock: {error}"
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True, None
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False, None
    except OSError as error:
        return False, f"cannot inspect robot GDK lock: {error}"
    finally:
        stream.close()


def _handler(source: G2ReadOnlySource, *, host: str, port: int, token: str) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {host.lower(), f"{host.lower()}:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "G2AlignmentAgent/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.headers.get("Host", "").lower() not in allowed_hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "Host is not allowed"})
                return
            if token:
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied, f"Bearer {token}"):
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "Bearer token is invalid"})
                    return
            path = urlparse(self.path).path
            try:
                if path == "/api/alignment/info":
                    self._json(HTTPStatus.OK, source.info())
                elif path == "/api/alignment/health":
                    self._json(HTTPStatus.OK, source.health())
                elif path == "/api/alignment/health/deep":
                    self._json(HTTPStatus.OK, source.deep_health())
                elif path == "/api/alignment/state":
                    self._json(HTTPStatus.OK, source.state())
                elif path == f"/api/alignment/cameras/{unquote(source.camera_name)}/calibration":
                    self._json(HTTPStatus.OK, source.calibration())
                elif path == f"/api/alignment/cameras/{unquote(source.camera_name)}/frame":
                    mime, content, width, height = source.frame()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("X-Camera-Width", str(width))
                    self.send_header("X-Camera-Height", str(height))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint does not exist"})
            except Exception as error:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            if self.headers.get("Host", "").lower() not in allowed_hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "Host is not allowed"})
                return
            if token:
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied, f"Bearer {token}"):
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "Bearer token is invalid"})
                    return
            parsed = urlparse(self.path)
            if parsed.path != "/api/alignment/release":
                self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint does not exist"})
                return
            try:
                query = parse_qs(parsed.query, keep_blank_values=False)
                hold_s = float(query.get("hold_s", [0.0])[-1])
                self._json(
                    HTTPStatus.OK,
                    {
                        "released": source.release_worker(hold_s=hold_s),
                        "gdk_state": "idle",
                        "release_hold_s": hold_s,
                    },
                )
            except (TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _json(self, status: int, value: Mapping[str, Any]) -> None:
            content = (json.dumps(value, ensure_ascii=False) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _validate_bind(host: str, token: str, *, allow_unauthenticated_private_lan: bool = False) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("--host must be a specific IP address") from error
    if address.is_unspecified or address.is_multicast or not (address.is_private or address.is_loopback):
        raise ValueError("robot agent may only bind to loopback or one private IP")
    if not address.is_loopback and not token and not allow_unauthenticated_private_lan:
        raise ValueError("private-LAN robot agent requires ALIGNMENT_ROBOT_TOKEN")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--idle-release-s",
        type=float,
        default=60.0,
        help="fallback GDK release timeout after the last live-data request",
    )
    parser.add_argument("--robot-lock-path", type=Path, default=DEFAULT_ROBOT_LOCK_PATH)
    parser.add_argument("--token-env", default="ALIGNMENT_ROBOT_TOKEN")
    parser.add_argument(
        "--allow-unauthenticated-private-lan",
        action="store_true",
        help="explicitly allow an unauthenticated listener on one private IP",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        not 1 <= args.port <= 65535
        or not 1 <= args.fps <= 30
        or args.camera_timeout_ms <= 0
        or args.idle_release_s < 1
    ):
        raise ValueError("port, fps, or camera timeout is outside the supported range")
    token = os.environ.get(args.token_env, "")
    _validate_bind(
        args.host,
        token,
        allow_unauthenticated_private_lan=args.allow_unauthenticated_private_lan,
    )
    source = G2ReadOnlySource(
        args.calibration,
        fps=args.fps,
        camera_timeout_ms=args.camera_timeout_ms,
        idle_release_s=args.idle_release_s,
        robot_lock_path=args.robot_lock_path,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler(source, host=args.host, port=args.port, token=token),
    )
    print(f"G2 read-only alignment agent: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        source.close()


if __name__ == "__main__":
    main()
