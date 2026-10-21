"""WebSocket transport with one in-flight request and metadata handshake."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .codec import pack_message, unpack_message
from .constants import DEFAULT_MAX_MESSAGE_BYTES, SUPPORTED_PROTOCOLS

logger = logging.getLogger(__name__)


def _load_websocket_client():
    try:
        import websockets.sync.client
    except ImportError as error:
        raise ImportError(
            "The robot client needs the bundled/offline `websockets` package. "
            "Run install_offline.sh on the robot first."
        ) from error
    return websockets.sync.client


class PolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        request_timeout_s: float,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        source_address: str | None = None,
    ) -> None:
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.request_timeout_s = request_timeout_s
        self.max_message_bytes = max_message_bytes
        self.source_address = source_address.strip() if source_address else None
        self._lock = threading.Lock()
        self._connection = None
        self._metadata: dict[str, Any] | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Policy client is not connected")
        return self._metadata

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def connect(self, wait: bool = True) -> dict[str, Any]:
        websocket_client = _load_websocket_client()
        while True:
            try:
                connect_kwargs: dict[str, Any] = {
                    "compression": None,
                    "max_size": self.max_message_bytes,
                    "open_timeout": 5,
                    "close_timeout": 2,
                }
                if self.source_address is not None:
                    connect_kwargs["source_address"] = (self.source_address, 0)
                connection = websocket_client.connect(
                    self.uri,
                    **connect_kwargs,
                )
                metadata = unpack_message(
                    connection.recv(timeout=self.request_timeout_s),
                    self.max_message_bytes,
                )
                protocol = metadata.get("protocol", {})
                if protocol not in SUPPORTED_PROTOCOLS:
                    connection.close()
                    raise RuntimeError(
                        f"Incompatible server protocol: {protocol}; supported={SUPPORTED_PROTOCOLS}"
                    )
                self._connection = connection
                self._metadata = metadata
                logger.info(
                    "Connected to %s source=%s profile=%s",
                    self.uri,
                    self.source_address or "system-default",
                    metadata.get("model_profile"),
                )
                return metadata
            except (ConnectionRefusedError, OSError, TimeoutError):
                if not wait:
                    raise
                logger.info("Waiting for policy server at %s", self.uri)
                time.sleep(2)

    def reconnect(self, wait: bool = False) -> dict[str, Any]:
        self.close()
        return self.connect(wait)

    def infer(self, message: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._connection is None:
                raise RuntimeError("Policy client is not connected")

            transport_started = time.perf_counter()
            serialize_started = time.perf_counter()
            request_payload = pack_message(message, self.max_message_bytes)
            serialize_finished = time.perf_counter()

            send_started_wall_time_ns = time.time_ns()
            send_started = time.perf_counter()
            self._connection.send(request_payload)
            send_finished = time.perf_counter()
            send_finished_wall_time_ns = time.time_ns()

            receive_wait_started = time.perf_counter()
            response_payload = self._connection.recv(timeout=self.request_timeout_s)
            response_received = time.perf_counter()
            response_received_wall_time_ns = time.time_ns()

            deserialize_started = time.perf_counter()
            response = unpack_message(response_payload, self.max_message_bytes)
            deserialize_finished = time.perf_counter()
            response_decoded_wall_time_ns = time.time_ns()

            response["client_timing"] = {
                "request_serialize_ms": (serialize_finished - serialize_started) * 1000,
                "request_bytes": len(request_payload),
                "websocket_send_ms": (send_finished - send_started) * 1000,
                "response_wait_ms": (response_received - receive_wait_started) * 1000,
                "response_bytes": len(response_payload),
                "response_deserialize_ms": (deserialize_finished - deserialize_started) * 1000,
                "transport_total_ms": (deserialize_finished - transport_started) * 1000,
                "send_started_wall_time_ns": send_started_wall_time_ns,
                "send_finished_wall_time_ns": send_finished_wall_time_ns,
                "response_received_wall_time_ns": response_received_wall_time_ns,
                "response_decoded_wall_time_ns": response_decoded_wall_time_ns,
            }
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error", "Unknown policy server error")))
        return response

    def close(self) -> None:
        connection, self._connection = self._connection, None
        self._metadata = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logger.debug("Ignoring WebSocket close failure", exc_info=True)
