"""Authenticated latest-frame transport for the Nexus parcel sidecar.

The transport carries JPEG frames *out of* the already connected robot-client
process.  It contains no GDK import, no model loading and no robot-motion API.
That separation is intentional: a second process must not initialise GDK while
the production G2A client already owns it, and the vision model must never gain
the ability to command the robot merely because it received an image.

Packet layout (all lengths are big-endian)::

    uint32 payload_length
    ascii[32] job_id
    uint16 ingress_token_length
    bytes     ingress_token
    uint32 metadata_length
    utf8 JSON metadata
    bytes     JPEG

The receiver keeps one latest-value mailbox.  When inference is slower than
capture, stale images are deliberately dropped rather than queued.
"""

from __future__ import annotations

import hmac
import json
import re
import struct
import threading
import time
from dataclasses import dataclass, replace
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import Any, Mapping


_JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_MAX_PACKET_BYTES = 4 * 1024 * 1024
_MAX_TOKEN_BYTES = 512
_MAX_METADATA_BYTES = 8 * 1024
_MAX_JPEG_BYTES = _MAX_PACKET_BYTES - 32 - 2 - _MAX_TOKEN_BYTES - 4
CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class FrameStreamError(ValueError):
    """A sender produced an invalid or unauthorised frame packet."""


@dataclass(frozen=True)
class HeadFramePacket:
    """A head JPEG, optionally accompanied by both wrist JPEGs."""

    job_id: str
    generation: int
    sequence: int
    capture_timestamp_ns: int
    jpeg: bytes
    received_timestamp_ns: int | None = None
    camera_jpegs: tuple[bytes, bytes, bytes] | None = None

    def validate(self) -> None:
        if not _JOB_ID_PATTERN.fullmatch(self.job_id):
            raise FrameStreamError("job_id must be a 32-character lowercase hex identifier")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise FrameStreamError("generation must be a positive integer")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise FrameStreamError("sequence must be a non-negative integer")
        if (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns <= 0
        ):
            raise FrameStreamError("capture_timestamp_ns must be a positive integer")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise FrameStreamError("jpeg payload must be non-empty bytes")
        if len(self.jpeg) > _MAX_JPEG_BYTES:
            raise FrameStreamError("jpeg payload is too large")
        if self.camera_jpegs is not None:
            if len(self.camera_jpegs) != 3 or self.camera_jpegs[0] != self.jpeg:
                raise FrameStreamError("three-camera payload must include the head JPEG first")
            if any(not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8") for frame in self.camera_jpegs):
                raise FrameStreamError("three-camera payload contains an invalid JPEG")
            if sum(map(len, self.camera_jpegs)) > _MAX_JPEG_BYTES:
                raise FrameStreamError("three-camera JPEG payload is too large")


def _metadata_from_packet(packet: HeadFramePacket) -> bytes:
    metadata = {
        "generation": packet.generation,
        "sequence": packet.sequence,
        "capture_timestamp_ns": packet.capture_timestamp_ns,
        "encoding": "jpeg_bundle_v1" if packet.camera_jpegs is not None else "jpeg",
    }
    if packet.camera_jpegs is not None:
        metadata["camera_names"] = CAMERA_NAMES
        metadata["image_lengths"] = [len(frame) for frame in packet.camera_jpegs]
    encoded = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise FrameStreamError("frame metadata is too large")
    return encoded


def encode_frame_packet(packet: HeadFramePacket, ingress_token: str) -> bytes:
    """Return a length-prefixed packet suitable for a persistent TCP stream."""

    packet.validate()
    if not isinstance(ingress_token, str):
        raise FrameStreamError("ingress token must be text")
    token = ingress_token.encode("utf-8")
    if not 16 <= len(token) <= _MAX_TOKEN_BYTES:
        raise FrameStreamError("ingress token length is invalid")
    metadata = _metadata_from_packet(packet)
    payload = b"".join(
        (
            packet.job_id.encode("ascii"),
            struct.pack("!H", len(token)),
            token,
            struct.pack("!I", len(metadata)),
            metadata,
            *(packet.camera_jpegs if packet.camera_jpegs is not None else (packet.jpeg,)),
        )
    )
    if len(payload) > _MAX_PACKET_BYTES:
        raise FrameStreamError("frame packet is too large")
    return struct.pack("!I", len(payload)) + payload


def decode_frame_payload(
    payload: bytes,
    *,
    expected_job_id: str,
    expected_generation: int,
    expected_ingress_token: str,
    received_timestamp_ns: int | None = None,
) -> HeadFramePacket:
    """Authenticate and parse one packet payload, excluding its uint32 prefix."""

    if not isinstance(payload, bytes) or not (38 <= len(payload) <= _MAX_PACKET_BYTES):
        raise FrameStreamError("frame payload length is invalid")
    if not _JOB_ID_PATTERN.fullmatch(expected_job_id):
        raise FrameStreamError("expected job id is invalid")
    if isinstance(expected_generation, bool) or not isinstance(expected_generation, int) or expected_generation < 1:
        raise FrameStreamError("expected generation is invalid")
    if not isinstance(expected_ingress_token, str):
        raise FrameStreamError("expected ingress token is invalid")

    offset = 0
    try:
        job_id = payload[offset : offset + 32].decode("ascii")
    except UnicodeDecodeError as error:
        raise FrameStreamError("job id is not ASCII") from error
    offset += 32
    if not _JOB_ID_PATTERN.fullmatch(job_id) or job_id != expected_job_id:
        raise FrameStreamError("frame belongs to a different deployment")

    token_length = struct.unpack("!H", payload[offset : offset + 2])[0]
    offset += 2
    if not 16 <= token_length <= _MAX_TOKEN_BYTES or offset + token_length + 4 > len(payload):
        raise FrameStreamError("ingress token length is invalid")
    try:
        token = payload[offset : offset + token_length].decode("utf-8")
    except UnicodeDecodeError as error:
        raise FrameStreamError("ingress token is not UTF-8") from error
    offset += token_length
    if not hmac.compare_digest(token, expected_ingress_token):
        raise FrameStreamError("ingress token is invalid")

    metadata_length = struct.unpack("!I", payload[offset : offset + 4])[0]
    offset += 4
    if not 2 <= metadata_length <= _MAX_METADATA_BYTES or offset + metadata_length >= len(payload):
        raise FrameStreamError("frame metadata length is invalid")
    try:
        metadata = json.loads(payload[offset : offset + metadata_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrameStreamError("frame metadata is invalid JSON") from error
    offset += metadata_length
    if not isinstance(metadata, Mapping):
        raise FrameStreamError("frame metadata must be an object")
    encoding = metadata.get("encoding")
    if not isinstance(encoding, str) or encoding not in {"jpeg", "jpeg_bundle_v1"}:
        raise FrameStreamError("unsupported presence frame encoding")
    generation = metadata.get("generation")
    sequence = metadata.get("sequence")
    captured = metadata.get("capture_timestamp_ns")
    if generation != expected_generation:
        raise FrameStreamError("frame belongs to an old task generation")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise FrameStreamError("frame sequence is invalid")
    if isinstance(captured, bool) or not isinstance(captured, int) or captured <= 0:
        raise FrameStreamError("frame capture timestamp is invalid")
    encoded = payload[offset:]
    camera_jpegs = None
    if encoding == "jpeg_bundle_v1":
        lengths = metadata.get("image_lengths")
        if metadata.get("camera_names") != list(CAMERA_NAMES) or not isinstance(lengths, list) or len(lengths) != 3:
            raise FrameStreamError("three-camera metadata is invalid")
        if any(isinstance(length, bool) or not isinstance(length, int) or length < 2 for length in lengths):
            raise FrameStreamError("three-camera JPEG lengths are invalid")
        if sum(lengths) != len(encoded):
            raise FrameStreamError("three-camera JPEG lengths do not match payload")
        first, second = lengths[0], lengths[0] + lengths[1]
        camera_jpegs = (encoded[:first], encoded[first:second], encoded[second:])
        jpeg = camera_jpegs[0]
    else:
        jpeg = encoded
    packet = HeadFramePacket(
        job_id=job_id,
        generation=generation,
        sequence=sequence,
        capture_timestamp_ns=captured,
        jpeg=jpeg,
        received_timestamp_ns=received_timestamp_ns,
        camera_jpegs=camera_jpegs,
    )
    packet.validate()
    return packet


class LatestFrameMailbox:
    """A thread-safe size-one mailbox that never queues old camera frames."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: HeadFramePacket | None = None
        self._last_received_sequence = -1
        self.accepted = 0
        self.rejected_stale = 0
        self.replaced_pending = 0

    def offer(self, packet: HeadFramePacket) -> bool:
        """Store a newer packet, returning false for duplicate/out-of-order input."""

        packet.validate()
        with self._condition:
            if packet.sequence <= self._last_received_sequence:
                self.rejected_stale += 1
                return False
            self._last_received_sequence = packet.sequence
            if self._packet is not None:
                self.replaced_pending += 1
            self._packet = packet
            self.accepted += 1
            self._condition.notify()
            return True

    def take(self, timeout_s: float | None = None) -> HeadFramePacket | None:
        """Take the newest available frame, waiting at most ``timeout_s`` seconds."""

        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._condition:
            if self._packet is None:
                self._condition.wait(timeout_s)
            packet = self._packet
            self._packet = None
            return packet

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "accepted": self.accepted,
                "rejected_stale": self.rejected_stale,
                "replaced_pending": self.replaced_pending,
                "last_sequence": self._last_received_sequence,
                "pending": 1 if self._packet is not None else 0,
            }


class PresenceFrameIngressServer(ThreadingTCPServer):
    """TCP ingress that validates packets before offering them to a mailbox."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        job_id: str,
        generation: int,
        ingress_token: str,
        mailbox: LatestFrameMailbox | None = None,
    ) -> None:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("job_id must be a 32-character lowercase hex identifier")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if not isinstance(ingress_token, str) or len(ingress_token.encode("utf-8")) < 16:
            raise ValueError("ingress token must contain at least 16 UTF-8 bytes")
        self.job_id = job_id
        self.generation = generation
        self.ingress_token = ingress_token
        self.mailbox = mailbox or LatestFrameMailbox()
        self.rejected_packets = 0
        self.accepted_packets = 0
        self._stats_lock = threading.Lock()
        super().__init__(server_address, _PresenceFrameIngressHandler)

    def ingest_payload(self, payload: bytes) -> bool:
        """Validate and store a payload; intended for both TCP and unit tests."""

        try:
            packet = decode_frame_payload(
                payload,
                expected_job_id=self.job_id,
                expected_generation=self.generation,
                expected_ingress_token=self.ingress_token,
                received_timestamp_ns=time.time_ns(),
            )
            accepted = self.mailbox.offer(packet)
        except FrameStreamError:
            with self._stats_lock:
                self.rejected_packets += 1
            return False
        with self._stats_lock:
            if accepted:
                self.accepted_packets += 1
            else:
                self.rejected_packets += 1
        return accepted

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            result = {
                "accepted_packets": self.accepted_packets,
                "rejected_packets": self.rejected_packets,
            }
        result.update(self.mailbox.stats())
        return result


class _PresenceFrameIngressHandler(StreamRequestHandler):
    """Read a persistent sequence of bounded packets without sending a reply."""

    def handle(self) -> None:
        server = self.server
        if not isinstance(server, PresenceFrameIngressServer):  # pragma: no cover - defensive
            return
        while True:
            header = self.rfile.read(4)
            if not header:
                return
            if len(header) != 4:
                return
            length = struct.unpack("!I", header)[0]
            if not 38 <= length <= _MAX_PACKET_BYTES:
                with server._stats_lock:
                    server.rejected_packets += 1
                return
            payload = self.rfile.read(length)
            if len(payload) != length:
                return
            server.ingest_payload(payload)


def packet_with_received_timestamp(packet: HeadFramePacket) -> HeadFramePacket:
    """Attach a Thor-local arrival timestamp for freshness decisions.

    The robot and Thor wall clocks need not be perfectly synchronized.  The
    control gate therefore uses this local arrival time for freshness while
    retaining the robot capture timestamp for diagnostics.
    """

    packet.validate()
    return replace(packet, received_timestamp_ns=time.time_ns())
