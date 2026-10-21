"""Generic WebSocket server that hosts any :class:`ModelAdapter`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from .contracts import ContractError, parse_inference_request, validate_action_chunk
from .plugins import ModelAdapter, load_plugin
from .wire import DEFAULT_MAX_BYTES, WireError, pack_message, unpack_message

logger = logging.getLogger(__name__)


class PolicyServer:
    def __init__(self, adapter: ModelAdapter, *, max_message_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.adapter = adapter
        self.max_message_bytes = max_message_bytes
        self.started_ns = time.time_ns()
        self.requests = 0
        # Most VLA runtimes and their RTC/session caches aren't safe for
        # concurrent inference. Keep one in flight while leaving the event loop responsive.
        self._inference_lock = asyncio.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": self.adapter.capabilities.model_id,
            "protocol": self.adapter.capabilities.to_metadata()["protocol"],
            "requests": self.requests,
            "uptime_s": (time.time_ns() - self.started_ns) / 1e9,
        }

    async def handle(self, websocket: Any) -> None:
        await websocket.send(pack_message(self.adapter.capabilities.to_metadata(), self.max_message_bytes))
        async for payload in websocket:
            started = time.perf_counter()
            request_id = "unknown"
            try:
                message = unpack_message(payload, self.max_message_bytes)
                request_id = str(message.get("request_id", "unknown"))
                request = parse_inference_request(message, self.adapter.capabilities)
                async with self._inference_lock:
                    if request.reset:
                        self.adapter.reset(request.session_id)
                    # Inference is intentionally serialized. Production model
                    # adapters normally enqueue work on an accelerator and
                    # return synchronously; adapter-owned worker pools remain
                    # possible without imposing a thread-affinity policy here.
                    chunk = self.adapter.infer(request)
                chunk = validate_action_chunk(chunk, self.adapter.capabilities)
                response = chunk.to_wire(request.request_id)
                response["timing"] = {"server_total_ms": (time.perf_counter() - started) * 1000}
                self.requests += 1
            except (ContractError, WireError, ValueError, RuntimeError) as error:
                logger.warning("Rejected request %s: %s", request_id, error)
                response = {
                    "type": "inference_response",
                    "status": "error",
                    "request_id": request_id,
                    "error": str(error),
                }
            await websocket.send(pack_message(response, self.max_message_bytes))


def _load_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


async def serve(
    host: str,
    port: int,
    server: PolicyServer,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    try:
        from websockets.legacy.server import serve as websocket_serve
    except ImportError as error:
        raise RuntimeError("Install the 'server' dependency group: pip install -e '.[server]'") from error

    async def health_request(path: str, _headers: Any):
        if path != "/healthz":
            return None
        body = (json.dumps(server.health()) + "\n").encode()
        return HTTPStatus.OK, [("Content-Type", "application/json"), ("Content-Length", str(len(body)))], body

    async with websocket_serve(
        server.handle,
        host,
        port,
        compression=None,
        max_size=server.max_message_bytes,
        process_request=health_request,
    ):
        logger.info("Serving %s on %s:%d", server.adapter.capabilities.model_id, host, port)
        await (shutdown_event or asyncio.Event()).wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="model plugin as package.module:Class")
    parser.add_argument("--adapter-config", default="{}", help="JSON or @path/to/config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    adapter = load_plugin(args.adapter, _load_json(args.adapter_config))
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(f"{args.adapter} must produce a ModelAdapter")
    adapter.load()
    try:
        asyncio.run(serve(args.host, args.port, PolicyServer(adapter)))
    except KeyboardInterrupt:
        pass
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
