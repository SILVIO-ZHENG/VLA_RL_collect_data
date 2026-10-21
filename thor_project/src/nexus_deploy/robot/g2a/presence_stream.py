"""Best-effort latest-frame publisher for the Nexus presence sidecar.

This module deliberately contains no GDK, policy, or robot-control import.
It receives an already captured JPEG from the production robot client and
forwards only the newest one to Thor.  A slow or unavailable receiver may drop
frames, but it must never delay the policy-inference or 100 Hz servo paths.

Packet format (big endian) matches ``nexus_deploy.vision.parcel.runtime.nexus_frame_stream``::

    uint32 payload_length
    ascii[32] job_id
    uint16 ingress_token_length
    bytes     ingress_token
    uint32 metadata_length
    utf8 JSON metadata
    bytes     JPEG
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import re
import socket
import struct
import threading
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

_JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_MAX_PACKET_BYTES = 4 * 1024 * 1024
_MAX_TOKEN_BYTES = 512
_MAX_METADATA_BYTES = 8 * 1024
_MAX_JPEG_BYTES = _MAX_PACKET_BYTES - 32 - 2 - _MAX_TOKEN_BYTES - 4
_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class PresenceFrameError(ValueError):
    """A frame-publisher configuration or image payload is invalid."""


def _validate_job_id(job_id: str) -> str:
    normalized = str(job_id).strip()
    if not _JOB_ID_PATTERN.fullmatch(normalized):
        raise PresenceFrameError("presence job id must be a 32-character lowercase hex identifier")
    return normalized


def _validate_token(token: str) -> bytes:
    if not isinstance(token, str):
        raise PresenceFrameError("presence ingress token must be text")
    encoded = token.encode("utf-8")
    if not 16 <= len(encoded) <= _MAX_TOKEN_BYTES:
        raise PresenceFrameError("presence ingress token length is invalid")
    return encoded


def encode_head_frame_packet(
    *,
    job_id: str,
    generation: int,
    sequence: int,
    capture_timestamp_ns: int,
    jpeg: bytes,
    ingress_token: str,
) -> bytes:
    """Encode one bounded, authenticated head-camera frame for TCP transport."""

    normalized_job_id = _validate_job_id(job_id)
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise PresenceFrameError("presence generation must be a positive integer")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise PresenceFrameError("presence frame sequence must be a non-negative integer")
    if (
        isinstance(capture_timestamp_ns, bool)
        or not isinstance(capture_timestamp_ns, int)
        or capture_timestamp_ns <= 0
    ):
        raise PresenceFrameError("presence capture timestamp must be a positive integer")
    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8"):
        raise PresenceFrameError("presence frame must be a JPEG byte payload")
    if len(jpeg) > _MAX_JPEG_BYTES:
        raise PresenceFrameError("presence JPEG payload is too large")

    token = _validate_token(ingress_token)
    metadata = json.dumps(
        {
            "generation": generation,
            "sequence": sequence,
            "capture_timestamp_ns": capture_timestamp_ns,
            "encoding": "jpeg",
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(metadata) > _MAX_METADATA_BYTES:
        raise PresenceFrameError("presence frame metadata is too large")
    payload = b"".join(
        (
            normalized_job_id.encode("ascii"),
            struct.pack("!H", len(token)),
            token,
            struct.pack("!I", len(metadata)),
            metadata,
            jpeg,
        )
    )
    if len(payload) > _MAX_PACKET_BYTES:
        raise PresenceFrameError("presence frame packet is too large")
    return struct.pack("!I", len(payload)) + payload


def encode_three_camera_frame_packet(
    *,
    job_id: str,
    generation: int,
    sequence: int,
    capture_timestamp_ns: int,
    images: Mapping[str, bytes],
    ingress_token: str,
) -> bytes:
    """Encode one atomic three-camera capture without decoding JPEGs on G2."""

    if set(images) != set(_CAMERAS):
        raise PresenceFrameError("three-camera presence requires all camera JPEGs")
    frames = tuple(images[name] for name in _CAMERAS)
    if any(not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8") for frame in frames):
        raise PresenceFrameError("three-camera presence requires valid JPEG bytes")
    job_id = _validate_job_id(job_id)
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise PresenceFrameError("presence generation must be positive")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise PresenceFrameError("presence sequence must be non-negative")
    if isinstance(capture_timestamp_ns, bool) or not isinstance(capture_timestamp_ns, int) or capture_timestamp_ns <= 0:
        raise PresenceFrameError("presence capture timestamp must be positive")
    token = _validate_token(ingress_token)
    metadata = json.dumps(
        {
            "generation": generation,
            "sequence": sequence,
            "capture_timestamp_ns": capture_timestamp_ns,
            "encoding": "jpeg_bundle_v1",
            "camera_names": _CAMERAS,
            "image_lengths": [len(frame) for frame in frames],
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = b"".join(
        (
            job_id.encode("ascii"),
            struct.pack("!H", len(token)),
            token,
            struct.pack("!I", len(metadata)),
            metadata,
            *frames,
        )
    )
    if len(metadata) > _MAX_METADATA_BYTES or len(payload) > _MAX_PACKET_BYTES:
        raise PresenceFrameError("three-camera presence packet is too large")
    return struct.pack("!I", len(payload)) + payload


class LatestHeadFramePublisher:
    """Publish the newest head frame, or an opt-in three-camera bundle.

    ``submit`` performs bounded validation and replaces a size-one mailbox; it
    makes no network call and never blocks waiting for Thor.  The caller may
    therefore invoke it directly after the production client's existing camera
    acquisition.  No camera device is opened here.
    """

    def __init__(
        self,
        endpoint: str,
        ingress_token: str,
        *,
        job_id: str,
        generation: int,
        fps: float = 10.0,
        timeout_s: float = 0.35,
    ) -> None:
        parsed = urlsplit(str(endpoint))
        if parsed.scheme != "tcp" or not parsed.hostname or parsed.username or parsed.password:
            raise PresenceFrameError("presence endpoint must be tcp://host:port without credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise PresenceFrameError("presence endpoint must not include a path, query, or fragment")
        if parsed.port is None or not 1 <= parsed.port <= 65535:
            raise PresenceFrameError("presence endpoint must include a valid TCP port")
        if not math.isfinite(fps) or not 0 < fps <= 15:
            raise PresenceFrameError("presence frame fps must be in (0, 15]")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 2:
            raise PresenceFrameError("presence frame timeout must be in (0, 2]")
        self.endpoint = str(endpoint)
        self._host = parsed.hostname
        self._port = int(parsed.port)
        self._token = ingress_token
        _validate_token(self._token)
        self._job_id = _validate_job_id(job_id)
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise PresenceFrameError("presence generation must be a positive integer")
        self._generation = generation
        mode = os.environ.get("NEXUS_PRESENCE_FRAME_MODE", "head")
        if mode not in {"head", "three"}:
            raise PresenceFrameError("NEXUS_PRESENCE_FRAME_MODE must be head or three")
        self._three_camera = mode == "three"
        self._period_s = 1.0 / float(fps)
        self._timeout_s = float(timeout_s)
        self._condition = threading.Condition()
        self._pending: tuple[int, int, bytes | dict[str, bytes]] | None = None
        self._next_sequence = 0
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._socket_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._consecutive_failures = 0
        self._last_warning = 0.0
        self.submitted = 0
        self.replaced_pending = 0
        self.sent = 0
        self.dropped_invalid = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("presence frame publisher is already started")
            if self._stopping:
                raise RuntimeError("presence frame publisher is closed")
            self._thread = threading.Thread(
                target=self._run,
                name="PresenceHeadFrames",
                daemon=True,
            )
            self._thread.start()

    def submit(self, images: Mapping[str, Any], *, capture_timestamp_ns: int) -> bool:
        """Replace the pending capture without waiting for the transport."""

        selected: dict[str, bytes] = {}
        for camera in _CAMERAS if self._three_camera else _CAMERAS[:1]:
            image = images.get(camera) if isinstance(images, Mapping) else None
            if not isinstance(image, Mapping):
                self.dropped_invalid += 1
                return False
            encoding = str(image.get("encoding") or "").lower()
            raw_data = image.get("data")
            if encoding not in {"jpeg", "jpg"} or not isinstance(raw_data, (bytes, bytearray, memoryview)):
                self.dropped_invalid += 1
                return False
            selected[camera] = bytes(raw_data)
        try:
            captured = int(capture_timestamp_ns)
            if captured <= 0 or any(not frame.startswith(b"\xff\xd8") for frame in selected.values()):
                raise PresenceFrameError("presence capture is invalid")
            if sum(map(len, selected.values())) > _MAX_JPEG_BYTES - _MAX_METADATA_BYTES:
                raise PresenceFrameError("presence capture is too large")
        except (PresenceFrameError, TypeError, ValueError):
            self.dropped_invalid += 1
            return False
        with self._condition:
            if self._stopping:
                return False
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._pending is not None:
                self.replaced_pending += 1
            self._pending = (sequence, captured, selected if self._three_camera else selected["cam_head"])
            self.submitted += 1
            self._condition.notify()
        return True

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._close_socket()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._timeout_s * 3))

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "submitted": self.submitted,
                "replaced_pending": self.replaced_pending,
                "sent": self.sent,
                "dropped_invalid": self.dropped_invalid,
                "pending": 1 if self._pending is not None else 0,
            }

    def _run(self) -> None:
        next_send_at = 0.0
        while True:
            with self._condition:
                while not self._stopping and self._pending is None:
                    self._condition.wait()
                if self._stopping:
                    return
                delay = next_send_at - time.monotonic()
                if delay > 0:
                    self._condition.wait(delay)
                    continue
                item, self._pending = self._pending, None
            if item is None:
                continue
            sequence, capture_timestamp_ns, image = item
            encoder = encode_three_camera_frame_packet if self._three_camera else encode_head_frame_packet
            arguments = {"images": image} if self._three_camera else {"jpeg": image}
            packet = encoder(
                job_id=self._job_id,
                generation=self._generation,
                sequence=sequence,
                capture_timestamp_ns=capture_timestamp_ns,
                ingress_token=self._token,
                **arguments,
            )
            try:
                self._send(packet)
            except OSError as error:
                self._close_socket()
                self._consecutive_failures += 1
                now = time.monotonic()
                if self._consecutive_failures >= 3 and now - self._last_warning >= 10.0:
                    logger.warning(
                        "Package-presence frame transport unavailable after %d attempts; "
                        "latest head frames are being dropped: %s",
                        self._consecutive_failures,
                        error,
                    )
                    self._last_warning = now
            else:
                self.sent += 1
                self._consecutive_failures = 0
            next_send_at = time.monotonic() + self._period_s

    def _send(self, packet: bytes) -> None:
        with self._socket_lock:
            if self._socket is None:
                stream = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
                stream.settimeout(self._timeout_s)
                stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                stream.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
                self._socket = stream
            self._socket.sendall(packet)

    def _close_socket(self) -> None:
        with self._socket_lock:
            stream, self._socket = self._socket, None
        if stream is not None:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()
