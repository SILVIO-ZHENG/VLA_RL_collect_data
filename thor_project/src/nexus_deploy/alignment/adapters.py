"""Read-only robot adapters for the standalone alignment service."""

from __future__ import annotations

import ipaddress
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np

from .geometry import pose_matrix
from .types import CameraCalibration, Frame


class AlignmentRobotAdapter(ABC):
    """Narrow read-only contract; implementations must never command motion."""

    @abstractmethod
    def connect(self, host: str) -> None: ...

    @abstractmethod
    def robot_info(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def health(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def calibration(self, camera_name: str) -> CameraCalibration: ...

    @abstractmethod
    def head_joints(self) -> Mapping[str, float]: ...

    @abstractmethod
    def read_frame(self, camera_name: str) -> Frame: ...

    @abstractmethod
    def close(self) -> None: ...


def _private_ip(host: str) -> str:
    try:
        value = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("robot host must be a literal IP address") from error
    if not (value.is_private or value.is_loopback) or value.is_multicast or value.is_unspecified:
        raise ValueError("robot host must be a private or loopback IP address")
    return value.compressed


class HTTPAlignmentRobotAdapter(AlignmentRobotAdapter):
    """Adapter for a robot's read-only alignment HTTP facade.

    Endpoint paths are configurable so a site bridge can map an existing camera
    service without modifying this package. Redirects are deliberately disabled.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.base_url = ""
        self.timeout_s = float(config.get("timeout_s", 2.0))
        self._sequence = 0

    def connect(self, host: str) -> None:
        address = _private_ip(str(host).strip())
        port = int(self.config.get("port", 8080))
        if not 1 <= port <= 65535:
            raise ValueError("adapter port must be in [1, 65535]")
        self.base_url = f"http://{address}:{port}"
        self.robot_info()

    def _path(self, key: str, fallback: str, **values: str) -> str:
        template = str(self.config.get(key, fallback))
        path = template.format(**values)
        if not path.startswith("/") or ".." in path:
            raise ValueError(f"invalid adapter endpoint {key}")
        return path

    def _open(self, path: str, *, method: str = "GET") -> tuple[bytes, str, Mapping[str, str]]:
        if not self.base_url:
            raise RuntimeError("adapter is not connected")
        # Robot hosts are validated private literal IPs.  Never inherit the
        # workstation's HTTP(S)_PROXY for this direct LAN hop: a proxy commonly
        # turns a healthy robot endpoint into a timeout or gateway response.
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}), _NoRedirect)
        request = urllib_request.Request(self.base_url + path, method=method)
        token_env = str(self.config.get("token_env", "ALIGNMENT_ROBOT_TOKEN"))
        token = os.environ.get(token_env, "") if token_env else ""
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            response = opener.open(request, timeout=self.timeout_s)
        except urllib_error.HTTPError as error:
            try:
                detail = json.loads(error.read()).get("error", str(error))
            except Exception:
                detail = str(error)
            raise RuntimeError(f"robot endpoint rejected the request: {detail}") from error
        except urllib_error.URLError as error:
            raise RuntimeError(f"robot endpoint is unreachable: {error.reason}") from error
        with response:
            content = response.read(int(self.config.get("max_response_bytes", 8 * 1024 * 1024)) + 1)
            if len(content) > int(self.config.get("max_response_bytes", 8 * 1024 * 1024)):
                raise RuntimeError("robot response exceeds configured size limit")
            return content, response.headers.get_content_type(), dict(response.headers)

    def _json(self, path: str) -> Mapping[str, Any]:
        content, _mime, _headers = self._open(path)
        value = json.loads(content)
        if not isinstance(value, Mapping):
            raise RuntimeError("robot endpoint did not return a JSON object")
        return value

    def robot_info(self) -> Mapping[str, Any]:
        return self._json(self._path("info_path", "/api/alignment/info"))

    def health(self) -> Mapping[str, Any]:
        return self._json(self._path("health_path", "/api/alignment/health"))

    def calibration(self, camera_name: str) -> CameraCalibration:
        name = urllib_parse.quote(camera_name, safe="")
        value = self._json(self._path("calibration_path", "/api/alignment/cameras/{camera}/calibration", camera=name))
        matrix = np.asarray(value.get("camera_matrix"), dtype=np.float64)
        distortion = np.asarray(value.get("distortion", ()), dtype=np.float64).reshape(-1)
        transform = np.asarray(value.get("base_from_camera"), dtype=np.float64)
        return CameraCalibration(matrix, distortion, transform, str(value.get("version", "unknown")))

    def head_joints(self) -> Mapping[str, float]:
        value = self._json(self._path("state_path", "/api/alignment/state"))
        joints = value.get("head_joints", {})
        if not isinstance(joints, Mapping):
            raise RuntimeError("robot head_joints must be an object")
        return {str(key): float(item) for key, item in joints.items()}

    def read_frame(self, camera_name: str) -> Frame:
        name = urllib_parse.quote(camera_name, safe="")
        path = self._path("frame_path", "/api/alignment/cameras/{camera}/frame", camera=name)
        content, mime, headers = self._open(path)
        self._sequence += 1
        return Frame(
            data=content,
            mime_type=mime,
            width=int(headers.get("X-Camera-Width", self.config.get("frame_width", 0))),
            height=int(headers.get("X-Camera-Height", self.config.get("frame_height", 0))),
            captured_at=time.time(),
            sequence=self._sequence,
        )

    def close(self) -> None:
        if not self.base_url:
            return
        try:
            # New agents release GDK immediately. Older agents return 404/501;
            # swallowing that response preserves compatibility with deployed
            # read-only bridges and their idle timeout.
            self._open(
                self._path("release_path", "/api/alignment/release"),
                method="POST",
            )
        except Exception:
            pass
        finally:
            self.base_url = ""


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise RuntimeError("robot endpoint redirects are not allowed")


class MockAlignmentRobotAdapter(AlignmentRobotAdapter):
    """Deterministic adapter for local UI demos and automated tests."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.host = ""
        self.sequence = 0

    def connect(self, host: str) -> None:
        self.host = str(host).strip() or "127.0.0.1"

    def robot_info(self) -> Mapping[str, Any]:
        return {
            "id": str(self.config.get("robot_id", "mock-g2")),
            "model": str(self.config.get("robot_model", "G2 Alignment Simulator")),
            "host": self.host,
            "cameras": [str(self.config.get("camera_name", "cam_head"))],
        }

    def health(self) -> Mapping[str, Any]:
        return {"status": "ok", "task_running": bool(self.config.get("task_running", False))}

    def calibration(self, camera_name: str) -> CameraCalibration:
        width, height = int(self.config.get("width", 1280)), int(self.config.get("height", 720))
        focal = float(self.config.get("focal_px", 900.0))
        camera_matrix = np.asarray(((focal, 0, width / 2), (0, focal, height / 2), (0, 0, 1)), dtype=np.float64)
        base_from_camera = pose_matrix(self.config.get("base_from_camera", (0, 0, 0, 0, 0, 0)))
        return CameraCalibration(camera_matrix, np.zeros(5), base_from_camera, "mock-calibration-v1")

    def head_joints(self) -> Mapping[str, float]:
        values = self.config.get("head_joints", {})
        return {str(name): float(value) for name, value in values.items()} if isinstance(values, Mapping) else {}

    def read_frame(self, camera_name: str) -> Frame:
        self.sequence += 1
        width, height = int(self.config.get("width", 1280)), int(self.config.get("height", 720))
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop stop-color="#1d2924"/><stop offset="1" stop-color="#0b1210"/>
</linearGradient></defs>
<rect width="100%" height="100%" fill="url(#g)"/><path d="M0 {height * .68:g} H{width}" stroke="#64736d"/>
<text x="42" y="62" fill="#b9c8c1" font-family="sans-serif" font-size="22">ALIGNMENT SIMULATOR · {camera_name}</text>
</svg>'''.encode()
        return Frame(svg, "image/svg+xml", width, height, time.time(), self.sequence)

    def close(self) -> None:
        self.host = ""
