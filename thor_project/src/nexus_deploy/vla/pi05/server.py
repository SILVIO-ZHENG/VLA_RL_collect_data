#!/usr/bin/env python
"""OpenPI-style WebSocket server for local G2A PI0.5 checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

# This server shares a 64-core host with other highly threaded services. Keep
# numerical libraries from creating one pool per core unless the operator
# explicitly overrides these variables before launch.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs import RTCAttentionSchedule  # noqa: E402

from .dataset_start_state import APPROVED_START_STATE_24D  # noqa: E402
from .policy import CAMERA_KEYS, G2APi05Policy, MockG2APolicy  # noqa: E402
from .profiles import PROFILE_OUTPUT_DIRS, resolve_checkpoint  # noqa: E402
from .protocol import (  # noqa: E402
    DEFAULT_MAX_MESSAGE_BYTES,
    ProtocolError,
    pack_message,
    unpack_message,
)

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_TASK = (
    "Pick a package from the box with the left gripper and place it on the table. "
    "Then pick it up with the right gripper and place it on the conveyor belt, "
    "keeping the QR code facing upward."
)
_WARMUP_SESSION_ID = "__server_startup_warmup__"


def _raise_keyboard_interrupt() -> None:
    raise KeyboardInterrupt


def _require_websockets():
    try:
        import websockets.asyncio.server
    except ImportError as error:
        raise ImportError(
            "The G2A deployment server requires websockets. "
            "Install it in the lerobot conda environment with `python -m pip install websockets`."
        ) from error
    return websockets.asyncio.server


def _warmup_policy(
    policy: G2APi05Policy | MockG2APolicy,
    *,
    requests: int,
    task: str,
    image_height: int,
    image_width: int,
) -> None:
    """Warm ordinary inference and the distinct RTC guidance path."""
    if requests == 0:
        logger.warning("Server startup warmup is disabled (--warmup-requests=0)")
        return

    warmup_state = np.asarray(APPROVED_START_STATE_24D, dtype=np.float32).copy()
    if policy.metadata.get("effector_profile") == "dahuan":
        warmup_state[[7, 15]] = 0.025
    observation = {
        "state": warmup_state,
        "images": {
            key: np.zeros((image_height, image_width, 3), dtype=np.uint8) for key in CAMERA_KEYS
        },
        "task": task,
    }
    rtc_available = bool(policy.metadata.get("rtc", {}).get("available", False))
    logger.info(
        "Starting policy warmup: base_requests=%d rtc_request=%s image=%dx%d task=%r",
        requests,
        rtc_available,
        image_width,
        image_height,
        task,
    )
    total_started = time.perf_counter()
    try:
        previous_response: dict[str, Any] | None = None
        for index in range(requests):
            request_started = time.perf_counter()
            previous_response = policy.infer(
                {
                    "kind": "infer",
                    "request_id": f"server-warmup-{index + 1}",
                    "session_id": _WARMUP_SESSION_ID,
                    "observation": observation,
                    "control": {"enable_rtc": False},
                    "reset": index == 0,
                }
            )
            timing = previous_response.get("server_timing", {})
            logger.info(
                "Policy base warmup %d/%d complete: wall=%.1fms pre/infer/post=%.1f/%.1f/%.1fms",
                index + 1,
                requests,
                (time.perf_counter() - request_started) * 1000,
                float(timing.get("preprocess_ms", 0.0)),
                float(timing.get("infer_ms", 0.0)),
                float(timing.get("postprocess_ms", 0.0)),
            )
        if rtc_available:
            assert previous_response is not None
            trained_rtc = policy.metadata.get("rtc", {}).get("inference_mode") == "trained"
            warmup_delay = 1 if trained_rtc else 0
            request_started = time.perf_counter()
            rtc_response = policy.infer(
                {
                    "kind": "infer",
                    "request_id": "server-warmup-rtc",
                    "session_id": _WARMUP_SESSION_ID,
                    "observation": observation,
                    "control": {
                        "mode": "rtc",
                        "enable_rtc": True,
                        "active_chunk_id": previous_response.get("chunk_id"),
                        "next_action_index": 0,
                        "estimated_delay_steps": warmup_delay,
                        "min_rtc_blend_steps": 0,
                    },
                    "reset": False,
                }
            )
            rtc = rtc_response.get("rtc", {})
            if not rtc.get("prefix_used", False):
                raise RuntimeError(
                    "RTC startup warmup did not exercise prefix guidance: "
                    f"fallback_reason={rtc.get('fallback_reason')}"
                )
            if trained_rtc and (
                rtc.get("inference_mode") != "trained"
                or rtc.get("estimated_delay_steps") != warmup_delay
            ):
                raise RuntimeError("RTC startup warmup did not use the trained prefix path")
            timing = rtc_response.get("server_timing", {})
            logger.info(
                "Policy RTC warmup complete: wall=%.1fms pre/infer/post=%.1f/%.1f/%.1fms "
                "effective_horizon=%d",
                (time.perf_counter() - request_started) * 1000,
                float(timing.get("preprocess_ms", 0.0)),
                float(timing.get("infer_ms", 0.0)),
                float(timing.get("postprocess_ms", 0.0)),
                int(rtc.get("effective_execution_horizon", 0)),
            )
    finally:
        policy.drop_session(_WARMUP_SESSION_ID)
    logger.info("Policy warmup finished in %.1fms", (time.perf_counter() - total_started) * 1000)


class G2AWebsocketPolicyServer:
    """Single-robot WebSocket server with a serialized GPU inference worker."""

    def __init__(
        self,
        policy: G2APi05Policy | MockG2APolicy,
        *,
        host: str,
        port: int,
        max_message_bytes: int,
        timing_log_file: Path,
        event_loop_lag_warning_ms: float,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.max_message_bytes = max_message_bytes
        self.timing_log_file = timing_log_file
        self.event_loop_lag_warning_ms = event_loop_lag_warning_ms
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g2a-policy")
        self._active_connection = asyncio.Lock()
        self.timing_log_file.parent.mkdir(parents=True, exist_ok=True)

    async def _run_inference(self, request: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        queued_at = time.perf_counter()

        def run_policy() -> dict[str, Any]:
            worker_started = time.perf_counter()
            worker_cpu_started = time.thread_time()
            response = self.policy.infer(request)
            worker_cpu_finished = time.thread_time()
            worker_finished = time.perf_counter()
            response.setdefault("server_timing", {}).update(
                {
                    "queue_ms": (worker_started - queued_at) * 1000,
                    "worker_ms": (worker_finished - worker_started) * 1000,
                    "worker_cpu_ms": (worker_cpu_finished - worker_cpu_started) * 1000,
                }
            )
            return response

        return await loop.run_in_executor(self._inference_executor, run_policy)

    async def _run_reset(self, session_id: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._inference_executor, self.policy.reset, session_id)

    def _write_timing_record(self, record: dict[str, Any]) -> None:
        with self.timing_log_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    async def _monitor_event_loop_lag(self) -> None:
        interval_s = 1.0
        expected = time.perf_counter() + interval_s
        while True:
            await asyncio.sleep(interval_s)
            now = time.perf_counter()
            lag_ms = max(0.0, (now - expected) * 1000)
            if lag_ms >= self.event_loop_lag_warning_ms:
                logger.warning("Server asyncio event-loop lag: %.1fms", lag_ms)
            expected = now + interval_s

    async def _handler(self, websocket) -> None:
        if self._active_connection.locked():
            await websocket.close(code=1013, reason="A robot client is already connected")
            return

        async with self._active_connection:
            remote = websocket.remote_address
            logger.info("Client connected: %s", remote)
            session_ids: set[str] = set()
            await websocket.send(pack_message(self.policy.metadata, max_bytes=self.max_message_bytes))

            try:
                async for payload in websocket:
                    request_received = time.perf_counter()
                    request_received_wall_time_ns = time.time_ns()
                    request_bytes = len(payload) if isinstance(payload, bytes) else 0
                    request_id = ""
                    request_deserialize_ms = 0.0
                    request_decoded_wall_time_ns = request_received_wall_time_ns
                    try:
                        deserialize_started = time.perf_counter()
                        request = unpack_message(payload, max_bytes=self.max_message_bytes)
                        deserialize_finished = time.perf_counter()
                        request_deserialize_ms = (deserialize_finished - deserialize_started) * 1000
                        request_decoded_wall_time_ns = time.time_ns()
                        request_id = str(request.get("request_id", ""))
                        operation = request.get("kind")

                        if operation == "ping":
                            response = {
                                "status": "ok",
                                "kind": "pong",
                                "request_id": request_id,
                                "model_profile": self.policy.metadata["model_profile"],
                            }
                        elif operation == "reset":
                            session_id = str(request.get("session_id", ""))
                            if not session_id:
                                raise ProtocolError("`session_id` is required for reset")
                            session_ids.add(session_id)
                            await self._run_reset(session_id)
                            response = {
                                "status": "ok",
                                "kind": "reset_ack",
                                "request_id": request_id,
                                "session_id": session_id,
                            }
                        elif operation == "infer":
                            session_id = str(request.get("session_id", ""))
                            if session_id:
                                session_ids.add(session_id)
                            response = await self._run_inference(request)
                        else:
                            raise ProtocolError(f"Unsupported request kind: {operation!r}")
                    except ProtocolError as error:
                        request_deserialize_ms = (
                            (time.perf_counter() - deserialize_started) * 1000
                            if "deserialize_started" in locals()
                            else 0.0
                        )
                        logger.warning("Rejected request %s: %s", request_id, error)
                        response = {
                            "status": "error",
                            "kind": "error",
                            "request_id": request_id,
                            "error": str(error),
                        }
                    except Exception as error:
                        request_deserialize_ms = (time.perf_counter() - deserialize_started) * 1000
                        logger.exception("Request %s failed", request_id)
                        response = {
                            "status": "error",
                            "kind": "error",
                            "request_id": request_id,
                            "error": f"{type(error).__name__}: {error}",
                        }

                    server_timing = response.setdefault("server_timing", {})
                    server_timing.update(
                        {
                            "request_deserialize_ms": request_deserialize_ms,
                            "request_bytes": request_bytes,
                            "request_received_wall_time_ns": request_received_wall_time_ns,
                            "request_decoded_wall_time_ns": request_decoded_wall_time_ns,
                            # Filled with a close first-pass estimate so it can be carried
                            # inside the response being measured. The exact second-pass
                            # value is written to the server JSONL timing log below.
                            "response_serialize_ms": 0.0,
                            "response_bytes": 0,
                        }
                    )
                    serialize_probe_started = time.perf_counter()
                    probe_payload = pack_message(response, max_bytes=self.max_message_bytes)
                    serialize_probe_ms = (time.perf_counter() - serialize_probe_started) * 1000
                    server_timing["response_serialize_ms"] = serialize_probe_ms
                    server_timing["response_bytes"] = len(probe_payload)
                    serialize_started = time.perf_counter()
                    response_payload = pack_message(response, max_bytes=self.max_message_bytes)
                    response_serialize_ms_exact = (time.perf_counter() - serialize_started) * 1000

                    send_started = time.perf_counter()
                    send_started_wall_time_ns = time.time_ns()
                    await websocket.send(response_payload)
                    send_finished_wall_time_ns = time.time_ns()
                    send_finished = time.perf_counter()
                    websocket_send_ms = (send_finished - send_started) * 1000
                    handler_total_ms = (send_finished - request_received) * 1000
                    record = {
                        "request_id": request_id,
                        "kind": response.get("kind"),
                        "status": response.get("status"),
                        "request_bytes": request_bytes,
                        "response_bytes": len(response_payload),
                        "request_received_wall_time_ns": request_received_wall_time_ns,
                        "send_started_wall_time_ns": send_started_wall_time_ns,
                        "send_finished_wall_time_ns": send_finished_wall_time_ns,
                        "request_deserialize_ms": request_deserialize_ms,
                        **server_timing,
                        "response_serialize_ms_exact": response_serialize_ms_exact,
                        "websocket_send_ms": websocket_send_ms,
                        "handler_total_ms": handler_total_ms,
                    }
                    self._write_timing_record(record)
                    logger.info(
                        "Request timing id=%s bytes=%d/%d decode=%.1fms queue=%.1fms "
                        "worker=%.1fms cpu=%.1fms pre/infer/post=%.1f/%.1f/%.1fms "
                        "encode=%.1fms send=%.1fms handler=%.1fms",
                        request_id,
                        request_bytes,
                        len(response_payload),
                        request_deserialize_ms,
                        float(server_timing.get("queue_ms", 0.0)),
                        float(server_timing.get("worker_ms", 0.0)),
                        float(server_timing.get("worker_cpu_ms", 0.0)),
                        float(server_timing.get("preprocess_ms", 0.0)),
                        float(server_timing.get("infer_ms", 0.0)),
                        float(server_timing.get("postprocess_ms", 0.0)),
                        response_serialize_ms_exact,
                        websocket_send_ms,
                        handler_total_ms,
                    )
            finally:
                for session_id in session_ids:
                    self.policy.drop_session(session_id)
                logger.info("Client disconnected: %s", remote)

    async def run(self) -> None:
        websockets_server = _require_websockets()

        def health_check(connection, request):
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        lag_monitor = asyncio.create_task(self._monitor_event_loop_lag())
        try:
            async with websockets_server.serve(
                self._handler,
                self.host,
                self.port,
                compression=None,
                max_size=self.max_message_bytes,
                ping_interval=20,
                ping_timeout=20,
                process_request=health_check,
            ) as server:
                logger.info(
                    "G2A PI0.5 server listening on ws://%s:%d profile=%s action_space=absolute",
                    self.host,
                    self.port,
                    self.policy.metadata["model_profile"],
                )
                await server.serve_forever()
        finally:
            lag_monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lag_monitor

    def close(self) -> None:
        self._inference_executor.shutdown(wait=False, cancel_futures=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high"),
        default="highest",
        help="PyTorch float32 matrix multiplication precision; 'high' enables faster TF32 on supported GPUs.",
    )
    parser.add_argument("--model-profile", choices=sorted(PROFILE_OUTPUT_DIRS), default="24d_absolute")
    parser.add_argument(
        "--effector-profile",
        choices=("dahuan", "dh", "omnipicker_legacy"),
        default="dahuan",
        help="End-effector semantics advertised to the robot client.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--model-fps",
        "--fps",
        dest="model_fps",
        type=float,
        default=30.0,
        help=(
            "Training-time action frequency advertised as metadata only; "
            "--fps is retained as a compatibility alias"
        ),
    )
    parser.add_argument(
        "--enable-rtc",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--execution-horizon", type=int, default=30)
    parser.add_argument(
        "--rtc-inference-mode",
        choices=("guided", "trained"),
        default="guided",
        help="Use the checkpoint-trained RTC denoising path for RTC requests.",
    )
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--fixed-noise-seed",
        type=int,
        default=None,
        help="Diagnostic only: reuse identical PI0.5 initial noise for every inference request.",
    )
    parser.add_argument(
        "--prefix-attention-schedule",
        choices=[schedule.value for schedule in RTCAttentionSchedule],
        default=RTCAttentionSchedule.EXP.value,
    )
    parser.add_argument(
        "--command-head-waist",
        action="store_true",
        help="Allow a native 24D model to command head/waist. They are held at current values by default.",
    )
    parser.add_argument("--max-message-mb", type=int, default=DEFAULT_MAX_MESSAGE_BYTES // (1024 * 1024))
    parser.add_argument("--mock", action="store_true", help="Use a zero-cost hold-position policy")
    parser.add_argument(
        "--legacy-model-loader",
        action="store_true",
        help="Disable the meta-device deployment loader for diagnosis and parity testing.",
    )
    parser.add_argument(
        "--mock-inference-delay-s",
        type=float,
        default=0.0,
        help="Artificial mock latency for testing stale-step dropping.",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--timing-log-file",
        type=Path,
        default=None,
        help="JSONL request timing output; defaults beside --log-file.",
    )
    parser.add_argument("--event-loop-lag-warning-ms", type=float, default=50.0)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Run this many local inference requests before opening the WebSocket port; 0 disables warmup.",
    )
    parser.add_argument("--warmup-task", default=DEFAULT_WARMUP_TASK)
    parser.add_argument("--warmup-image-width", type=int, default=640)
    parser.add_argument("--warmup-image-height", type=int, default=400)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    return parser.parse_args()


def _configure_logging(log_file: Path | None) -> Path:
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("outputs/g2a_pi05_deploy/logs") / f"server_{timestamp}.log"
    log_file = log_file.expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,
    )
    return log_file


def main() -> None:
    args = parse_args()
    if args.effector_profile == "dh":
        args.effector_profile = "dahuan"
    if (
        args.model_fps <= 0
        or args.execution_horizon <= 0
        or args.max_message_mb <= 0
        or args.event_loop_lag_warning_ms <= 0
        or args.torch_num_threads <= 0
        or args.torch_num_interop_threads <= 0
        or args.warmup_requests < 0
        or args.warmup_image_width <= 0
        or args.warmup_image_height <= 0
    ):
        raise ValueError(
            "fps, horizons, message size, timing threshold, thread counts, and warmup image dimensions "
            "must be positive; warmup requests must be non-negative"
        )
    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(args.torch_num_interop_threads)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    log_file = _configure_logging(args.log_file)
    logger.info("Writing server log to %s", log_file)
    timing_log_file = (
        args.timing_log_file.expanduser().resolve()
        if args.timing_log_file is not None
        else log_file.with_name(f"{log_file.stem}_timing.jsonl")
    )
    logger.info("Writing server request timings to %s", timing_log_file)
    logger.info("Float32 matrix multiplication precision: %s", torch.get_float32_matmul_precision())
    logger.info(
        "Server CPU thread limits torch=%d interop=%d OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        os.environ["OMP_NUM_THREADS"],
        os.environ["MKL_NUM_THREADS"],
        os.environ["OPENBLAS_NUM_THREADS"],
        os.environ["NUMEXPR_NUM_THREADS"],
    )

    if args.mock:
        policy: G2APi05Policy | MockG2APolicy = MockG2APolicy(
            model_fps=args.model_fps,
            execution_horizon=args.execution_horizon,
            inference_delay_s=args.mock_inference_delay_s,
            effector_profile=args.effector_profile,
        )
    else:
        checkpoint = resolve_checkpoint(args.model_profile, args.checkpoint)
        if args.checkpoint is None:
            logger.warning(
                "Automatically selected latest complete checkpoint for profile=%s: %s",
                args.model_profile,
                checkpoint,
            )
        else:
            logger.info("Using explicitly selected checkpoint: %s", checkpoint)
        policy = G2APi05Policy(
            checkpoint,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            model_fps=args.model_fps,
            execution_horizon=args.execution_horizon,
            rtc_inference_mode=args.rtc_inference_mode,
            max_guidance_weight=args.max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule(args.prefix_attention_schedule),
            command_head_waist=args.command_head_waist,
            model_profile=args.model_profile,
            fixed_noise_seed=args.fixed_noise_seed,
            fast_model_load=not args.legacy_model_loader,
            effector_profile=args.effector_profile,
        )

    _warmup_policy(
        policy,
        requests=args.warmup_requests,
        task=args.warmup_task,
        image_height=args.warmup_image_height,
        image_width=args.warmup_image_width,
    )

    server = G2AWebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        max_message_bytes=args.max_message_mb * 1024 * 1024,
        timing_log_file=timing_log_file,
        event_loop_lag_warning_ms=args.event_loop_lag_warning_ms,
    )
    try:
        asyncio.run(server.run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Stopping server")
    finally:
        server.close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: _raise_keyboard_interrupt())
    main()
