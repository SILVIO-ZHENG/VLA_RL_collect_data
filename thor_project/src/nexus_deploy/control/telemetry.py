"""Token-authenticated, one-way robot telemetry transport."""

from __future__ import annotations

import json
import struct
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import Mapping, Protocol

from nexus_deploy.control.errors import ValidationError


class TelemetrySink(Protocol):
    def ingest_robot_telemetry(self, job_id: str, payload: Mapping, token: str) -> object: ...


class TelemetryTCPServer(ThreadingTCPServer):
    """One-way, token-authenticated latest-value display stream."""

    allow_reuse_address = True
    daemon_threads = True


def telemetry_stream_handler(plane: TelemetrySink) -> type[StreamRequestHandler]:
    class TelemetryStreamHandler(StreamRequestHandler):
        def handle(self) -> None:
            self.request.settimeout(5.0)
            while True:
                try:
                    header = self.rfile.read(4)
                    if not header or len(header) != 4:
                        return
                    length = struct.unpack("!I", header)[0]
                    if not 36 < length <= 64 * 1024:
                        return
                    frame = self.rfile.read(length)
                    if len(frame) != length:
                        return
                    job_id = frame[:32].decode("ascii")
                    token_length = struct.unpack("!H", frame[32:34])[0]
                    token_end = 34 + token_length
                    if token_length < 16 or token_end >= len(frame):
                        return
                    token = frame[34:token_end].decode("utf-8")
                    payload = json.loads(frame[token_end:])
                    if not isinstance(payload, Mapping):
                        return
                    plane.ingest_robot_telemetry(job_id, payload, token)
                except (
                    UnicodeDecodeError,
                    ValueError,
                    ValidationError,
                    json.JSONDecodeError,
                    OSError,
                ):
                    return

    return TelemetryStreamHandler


# Compatibility name used by existing tests and launch code.
_telemetry_stream_handler = telemetry_stream_handler
