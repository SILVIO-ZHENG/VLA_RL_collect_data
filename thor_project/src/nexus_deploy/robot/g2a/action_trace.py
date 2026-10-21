"""Durable read-only traces for diagnosing PI0.5 action chunk smoothness."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ACTION_DIM, CANONICAL_STATE_NAMES


class ControlDiagnostics:
    """Bounded, asynchronous diagnostics; never perform file I/O in control.

    Separate from the mandatory collision audit. Dropped rows/writer errors
    invalidate this diagnostic trace, but must not themselves interrupt motion.
    """
    def __init__(self, path: Path, *, queue_size: int = 2048, max_bytes: int = 128 * 1024 * 1024):
        if queue_size < 1 or max_bytes < 1:
            raise ValueError("Diagnostic queue and file limits must be positive")
        self.path = path
        self.max_bytes = max_bytes
        self.queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="ControlDiagnostics", daemon=True)
        self.thread.start()

    def write(self, record):
        if self.closed or self.error:
            self.dropped += 1
            return
        snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in record.items()}
        try:
            self.queue.put_nowait(snapshot)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                count = 0
                bytes_written = 0
                while True:
                    record = self.queue.get()
                    if record is None:
                        stream.write(json.dumps(dict(record_type="trace_end", dropped=self.dropped)) + "\n")
                        return
                    serialized = json.dumps(_json_value(record), separators=(",", ":")) + "\n"
                    if bytes_written + len(serialized) > self.max_bytes:
                        self.error = "diagnostic_file_size_limit"
                        self.dropped += 1
                        stream.write(json.dumps(dict(record_type="trace_end", dropped=self.dropped,
                                                     truncated=True, reason=self.error)) + "\n")
                        logging.getLogger(__name__).warning(
                            "Control diagnostics reached %d-byte limit; stopping diagnostic capture only", self.max_bytes)
                        return
                    stream.write(serialized)
                    bytes_written += len(serialized)  # JSON is ASCII escaped.
                    count += 1
                    if count % 20 == 0:
                        stream.flush()
        except Exception as error:
            self.error = repr(error)
            logging.getLogger(__name__).warning("Control diagnostics unavailable: %s", self.error)

    def close(self):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + 2
        while self.thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(None, timeout=.05)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=2)
        if self.dropped or self.error or self.thread.is_alive():
            logging.getLogger(__name__).warning(
                "Control diagnostics incomplete: dropped=%d error=%s writer_alive=%s",
                self.dropped, self.error, self.thread.is_alive())


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_image(payload: dict[str, Any], path_stem: Path) -> dict[str, Any]:
    encoding = str(payload.get("encoding", "unknown"))
    color_format = str(payload.get("color_format", "unknown"))
    width = int(payload.get("width", 0))
    height = int(payload.get("height", 0))
    data = bytes(payload.get("data", b""))

    if encoding == "jpeg":
        output_path = path_stem.with_suffix(".jpg")
        output_path.write_bytes(data)
    elif encoding == "raw" and width > 0 and height > 0 and len(data) == width * height * 3:
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if color_format == "bgr":
            pixels = pixels[:, :, ::-1]
        if color_format in {"rgb", "bgr"}:
            output_path = path_stem.with_suffix(".ppm")
            with output_path.open("wb") as stream:
                stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                stream.write(np.ascontiguousarray(pixels).tobytes())
        else:
            output_path = path_stem.with_suffix(".bin")
            output_path.write_bytes(data)
    else:
        output_path = path_stem.with_suffix(".bin")
        output_path.write_bytes(data)

    return {
        "path": output_path.name,
        "encoding": encoding,
        "color_format": color_format,
        "width": width,
        "height": height,
        "bytes": len(data),
    }


class ActionTraceRecorder:
    """Write every inference response immediately and servo samples on close."""

    def __init__(
        self,
        output_root: Path,
        *,
        session_id: str,
        task: str,
        policy_fps: float,
        servo_fps: float,
        inference_mode: str,
        server_metadata: dict[str, Any],
        save_images: bool = True,
        freeze_observation: bool = False,
        action_low_pass_cutoff_hz: float | None = None,
        effector_action_low_pass_cutoff_hz: float | None = None,
        hold_effectors: bool = False,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_root.expanduser().resolve() / f"{timestamp}_{session_id[:8]}"
        self.requests_dir = self.run_dir / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=False)
        self.save_images = save_images
        self._lock = threading.Lock()
        self._request_count = 0
        self._servo_times: list[float] = []
        self._servo_actions: list[np.ndarray] = []
        self._servo_chunk_ids: list[str] = []
        self._servo_action_indices: list[int] = []
        self._closed = False
        self._manifest_path = self.run_dir / "requests.jsonl"

        run_metadata = {
            "format_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "safety": {
                "read_only": True,
                "gdk_actions_sent": 0,
                "description": "Model actions were recorded but never sent to GDK.",
            },
            "session_id": session_id,
            "task": task,
            "policy_fps": policy_fps,
            "servo_fps": servo_fps,
            "inference_mode": inference_mode,
            "observation_mode": "frozen_first_observation" if freeze_observation else "live",
            "client_action_conditioning": {
                "low_pass_cutoff_hz": action_low_pass_cutoff_hz,
                "effector_low_pass_cutoff_hz": effector_action_low_pass_cutoff_hz,
                "hold_effectors": hold_effectors,
            },
            "canonical_state_order": list(CANONICAL_STATE_NAMES),
            "server_metadata": _json_value(server_metadata),
            "images_saved": save_images,
        }
        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def record_request(
        self,
        *,
        request_id: str,
        observation: dict[str, Any],
        response: dict[str, Any],
        anchor: Any,
        estimated_delay_steps: int,
        round_trip_s: float,
        observation_latency_s: float,
        latency_p95_s: float,
        stale_steps: int,
        install_result: Any,
    ) -> Path:
        state = np.asarray(observation["state"], dtype=np.float32)
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"Cannot trace invalid state shape/content: {state.shape}")
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(f"Cannot trace invalid action chunk shape/content: {actions.shape}")

        with self._lock:
            if self._closed:
                raise RuntimeError("Action trace recorder is closed")
            request_index = self._request_count
            self._request_count += 1

        request_dir = self.requests_dir / f"request_{request_index:04d}"
        request_dir.mkdir()
        np.savez_compressed(request_dir / "trace.npz", state=state, actions=actions)

        image_metadata: dict[str, Any] = {}
        if self.save_images:
            image_dir = request_dir / "images"
            image_dir.mkdir()
            for camera_name, payload in observation["images"].items():
                image_metadata[camera_name] = _save_image(payload, image_dir / camera_name)
                image_metadata[camera_name]["path"] = str(
                    Path("images") / image_metadata[camera_name]["path"]
                )
        else:
            for camera_name, payload in observation["images"].items():
                image_metadata[camera_name] = {
                    "saved": False,
                    "encoding": str(payload.get("encoding", "unknown")),
                    "color_format": str(payload.get("color_format", "unknown")),
                    "width": int(payload.get("width", 0)),
                    "height": int(payload.get("height", 0)),
                    "bytes": len(payload.get("data", b"")),
                }

        metadata = {
            "request_index": request_index,
            "request_id": request_id,
            "chunk_id": str(response.get("chunk_id", "")),
            "task": observation.get("task"),
            "round_trip_ms": round_trip_s * 1000,
            "observation_to_response_ms": observation_latency_s * 1000,
            "rolling_p95_latency_ms": latency_p95_s * 1000,
            "stale_steps": stale_steps,
            "server_timing": _json_value(response.get("server_timing", {})),
            "client_timing": _json_value(response.get("client_timing", {})),
            "rtc": _json_value(response.get("rtc", {})),
            "estimated_delay_steps": estimated_delay_steps,
            "anchor": {
                "chunk_id": anchor.chunk_id,
                "next_action_index": anchor.next_action_index,
                "remaining_actions": anchor.remaining_actions,
                "consumed_since_install": anchor.consumed_since_install,
            },
            "install": {
                "installed": bool(install_result.installed),
                "dropped_steps": int(install_result.dropped_steps),
                "reason": str(install_result.reason),
            },
            "capture": {
                key: _json_value(observation[key])
                for key in (
                    "state_capture_wall_time_ns",
                    "observation_monotonic_time",
                    "state_capture_monotonic_time",
                    "camera_capture_start_wall_time_ns",
                    "camera_capture_wall_time_ns",
                    "camera_capture_duration_ms",
                    "camera_frame_wall_time_ns",
                    "camera_skew_ms",
                    "state_camera_skew_ms",
                )
                if key in observation
            },
            "image_transport_timing": _json_value(
                observation.get("image_transport_timing", {})
            ),
            "images": image_metadata,
        }
        (request_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._lock, self._manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        return request_dir

    def record_servo_sample(
        self,
        *,
        monotonic_time: float,
        action: np.ndarray,
        chunk_id: str | None,
        action_index: int,
    ) -> None:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
            return
        with self._lock:
            if self._closed:
                return
            self._servo_times.append(float(monotonic_time))
            self._servo_actions.append(array.copy())
            self._servo_chunk_ids.append(chunk_id or "")
            self._servo_action_indices.append(int(action_index))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            times = np.asarray(self._servo_times, dtype=np.float64)
            actions = (
                np.stack(self._servo_actions).astype(np.float32, copy=False)
                if self._servo_actions
                else np.empty((0, ACTION_DIM), dtype=np.float32)
            )
            chunk_ids = np.asarray(self._servo_chunk_ids, dtype=str)
            action_indices = np.asarray(self._servo_action_indices, dtype=np.int32)
            request_count = self._request_count

        np.savez_compressed(
            self.run_dir / "servo_samples.npz",
            monotonic_time=times,
            actions=actions,
            chunk_ids=chunk_ids,
            action_indices=action_indices,
        )
        (self.run_dir / "completed.json").write_text(
            json.dumps(
                {
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "request_count": request_count,
                    "servo_sample_count": len(times),
                    "gdk_actions_sent": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
