"""Best-effort latest-value telemetry isolated from the servo control path."""

from __future__ import annotations

import http.client
import json
import logging
import re
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

INFERENCE_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class LatestInferenceFrameWriter:
    """Persist inference input JPEGs without blocking inference or servo loops.

    ``submit`` only replaces one in-memory mailbox.  A background thread writes
    immutable, already-encoded image bytes and publishes the manifest last, so
    the platform never observes a partially written frame.  A small bounded
    history lets the platform fetch all three cameras from the same request.
    """

    def __init__(self, directory: Path, *, retained_frames: int = 64) -> None:
        if retained_frames < 3:
            raise ValueError("retained_frames must be at least 3")
        self.directory = directory.expanduser().resolve()
        self.retained_frames = int(retained_frames)
        self._condition = threading.Condition()
        self._pending: tuple[str, float, dict[str, tuple[str, bytes]]] | None = None
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("inference frame writer is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="InferenceFrameWriter",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        request_id: str,
        images: Mapping[str, Any],
        *,
        captured_at: float,
    ) -> None:
        safe_request_id = str(request_id)
        if not safe_request_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in safe_request_id
        ):
            logger.warning("Skipping inference frame with unsafe request id: %r", request_id)
            return
        encoded: dict[str, tuple[str, bytes]] = {}
        for camera_name in INFERENCE_CAMERA_NAMES:
            image = images.get(camera_name)
            if not isinstance(image, Mapping):
                return
            encoding = str(image.get("encoding") or "").lower()
            data = image.get("data")
            if encoding not in {"jpeg", "jpg"} or not isinstance(
                data, (bytes, bytearray, memoryview)
            ):
                return
            payload = bytes(data)
            if not payload.startswith(b"\xff\xd8"):
                return
            encoded[camera_name] = ("image/jpeg", payload)
        with self._condition:
            if self._stopping:
                return
            self._pending = (safe_request_id, float(captured_at), encoded)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                item, self._pending = self._pending, None
                stopping = self._stopping
            if item is not None:
                try:
                    self._write(*item)
                except OSError as error:
                    logger.warning("Unable to persist latest inference frame: %s", error)
            if stopping:
                with self._condition:
                    if self._pending is None:
                        return

    def _write(
        self,
        request_id: str,
        captured_at: float,
        images: dict[str, tuple[str, bytes]],
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        published: dict[str, dict[str, Any]] = {}
        for camera_name, (mime_type, payload) in images.items():
            filename = f"{request_id}-{camera_name}.jpg"
            target = self.directory / filename
            temporary = self.directory / f".{filename}.tmp"
            temporary.write_bytes(payload)
            temporary.replace(target)
            published[camera_name] = {
                "path": filename,
                "mime_type": mime_type,
                "bytes": len(payload),
            }
        manifest = {
            "request_id": request_id,
            "captured_at": captured_at,
            "published_at": time.time(),
            "images": published,
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        frame_manifest = self.directory / f"{request_id}.json"
        frame_temporary = self.directory / f".{request_id}.json.tmp"
        frame_temporary.write_text(manifest_text, encoding="utf-8")
        frame_temporary.replace(frame_manifest)
        latest_temporary = self.directory / ".latest.json.tmp"
        latest_temporary.write_text(manifest_text, encoding="utf-8")
        latest_temporary.replace(self.directory / "latest.json")
        manifests = sorted(
            (
                path
                for path in self.directory.glob("*.json")
                if path.name != "latest.json"
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for obsolete in manifests[self.retained_frames :]:
            try:
                old = json.loads(obsolete.read_text(encoding="utf-8"))
                for value in old.get("images", {}).values():
                    relative = value.get("path") if isinstance(value, Mapping) else None
                    if isinstance(relative, str) and Path(relative).name == relative:
                        (self.directory / relative).unlink(missing_ok=True)
                obsolete.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError):
                continue


class LatestTelemetryPublisher:
    """Upload only the newest control snapshot from a background thread.

    There is deliberately no unbounded queue.  Slow or disconnected platform
    networking drops intermediate visualization frames while the robot's servo
    loop continues unchanged.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        snapshot: Callable[[], Mapping[str, Any] | None],
        *,
        fps: float = 20.0,
        timeout_s: float = 0.35,
    ) -> None:
        if not endpoint.startswith(("http://", "https://", "tcp://")):
            raise ValueError("telemetry endpoint must use http://, https://, or tcp://")
        if not token:
            raise ValueError("telemetry token must not be empty")
        if fps <= 0 or timeout_s <= 0:
            raise ValueError("telemetry fps and timeout must be positive")
        self.endpoint = endpoint
        parsed = urlsplit(endpoint)
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError("telemetry endpoint must have a host and no credentials")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._path = parsed.path or "/"
        if parsed.query:
            self._path += f"?{parsed.query}"
        self._stream_job_id: str | None = None
        if self._scheme == "tcp":
            match = re.search(r"/deployments/([0-9a-f]{32})/robot-telemetry$", parsed.path)
            if match is None:
                raise ValueError("tcp telemetry endpoint must identify one deployment job")
            self._stream_job_id = match.group(1)
        self.token = token
        self.snapshot = snapshot
        self.period_s = 1.0 / fps
        self.timeout_s = timeout_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sequence = -1
        self._last_warning = 0.0
        self._consecutive_failures = 0
        self._connection: http.client.HTTPConnection | None = None
        self._stream_socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry publisher is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="PlatformTelemetry",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.timeout_s * 2))
        self._close_connection()

    def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
        stream, self._stream_socket = self._stream_socket, None
        if stream is not None:
            stream.close()

    def _post_stream(self, body: bytes) -> None:
        if self._stream_socket is None:
            self._stream_socket = socket.create_connection(
                (self._host, self._port or 8768),
                timeout=self.timeout_s,
            )
            self._stream_socket.settimeout(self.timeout_s)
            self._stream_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._stream_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024)
        assert self._stream_socket is not None
        assert self._stream_job_id is not None
        token = self.token.encode("utf-8")
        payload = (
            self._stream_job_id.encode("ascii")
            + struct.pack("!H", len(token))
            + token
            + body
        )
        if len(payload) > 64 * 1024:
            raise RuntimeError("telemetry stream frame exceeds 64 KiB")
        self._stream_socket.sendall(struct.pack("!I", len(payload)) + payload)

    def _post(self, body: bytes) -> None:
        if self._scheme == "tcp":
            self._post_stream(body)
            return
        if self._connection is None:
            connection_type = (
                http.client.HTTPSConnection
                if self._scheme == "https"
                else http.client.HTTPConnection
            )
            self._connection = connection_type(
                self._host,
                self._port,
                timeout=self.timeout_s,
            )
            self._connection.connect()
            if self._connection.sock is not None:
                self._connection.sock.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )
        assert self._connection is not None
        self._connection.request(
            "POST",
            self._path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Nexus-Telemetry-Token": self.token,
            },
        )
        response = self._connection.getresponse()
        response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"HTTP {response.status}")
        if response.will_close:
            self._close_connection()

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            value = self.snapshot()
            if not isinstance(value, Mapping):
                continue
            sequence = int(value.get("sequence", -1))
            if sequence <= self._last_sequence:
                continue
            body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            try:
                self._post(body)
                self._last_sequence = sequence
                self._consecutive_failures = 0
            except (OSError, RuntimeError, http.client.HTTPException) as error:
                self._close_connection()
                self._consecutive_failures += 1
                now = time.monotonic()
                if self._consecutive_failures >= 3 and now - self._last_warning >= 10.0:
                    logger.warning(
                        "Platform telemetry unavailable after %d consecutive attempts; "
                        "visualization frames are being dropped: %s",
                        self._consecutive_failures,
                        error,
                    )
                    self._last_warning = now
