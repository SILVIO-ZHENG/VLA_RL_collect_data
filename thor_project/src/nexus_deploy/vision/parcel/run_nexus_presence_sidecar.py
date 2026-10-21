#!/usr/bin/env python3
"""Run the parcel-presence model on JPEG frames supplied by the G2A client.

This process deliberately does **not** import or initialise GDK.  The production
robot client remains the sole owner of the real G2 camera and robot connection.
The sidecar receives already-encoded head and wrist JPEGs over an
authenticated private-LAN TCP ingress, evaluates the existing YOLO model, and
posts a bounded observation to the Nexus control plane.  It has no SSH,
robot-motion, pause, resume, or reset capability.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError

# Keep the configured interpreter's own packages ahead of inherited CUDA/OpenCV
# packages. No host name, username or Python minor version is hard-coded.
import sysconfig
_vision_site = Path(sysconfig.get_path("purelib"))
_vision_site_text = str(_vision_site)
while _vision_site_text in sys.path:
    sys.path.remove(_vision_site_text)
sys.path.insert(0, _vision_site_text)

# Import the Segment26-capable build before any inherited base-environment
# package can import and cache an older Ultralytics module.
import ultralytics as _vision_ultralytics
from ultralytics.nn.modules.head import Segment26 as _VisionSegment26

if not Path(_vision_ultralytics.__file__).resolve().is_relative_to(_vision_site.resolve()):
    raise ImportError(
        "wrong Ultralytics selected for presence sidecar: "
        f"{_vision_ultralytics.__file__}"
    )

# The vision interpreter also has an older editable nexus_deploy install.
# -E ignores PYTHONPATH, so pin imports to this live deployment source.
_live_src = Path(__file__).resolve().parents[3]
if not (_live_src / "nexus_deploy" / "vision" / "parcel" / "runtime" / "presence.py").is_file():
    raise ImportError(f"sidecar source root missing: {_live_src}")
sys.path.insert(0, str(_live_src))

from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np

from nexus_deploy.vision.parcel.runtime import ParcelStateEstimator, RateLimitedParcelStateService, RuntimeConfig
from nexus_deploy.vision.parcel.runtime.nexus_frame_stream import (
    CAMERA_NAMES,
    HeadFramePacket,
    LatestFrameMailbox,
    PresenceFrameIngressServer,
)
from nexus_deploy.vision.parcel.runtime.presence import (
    NO_PACKAGE,
    PACKAGE_PRESENT,
    UNKNOWN,
    ParcelPresenceGate,
    PresenceResult,
)


LOGGER = logging.getLogger("nexus_presence_sidecar")
MAX_REASON_LENGTH = 240
MAX_CONTROL_BODY_BYTES = 32 * 1024


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be in 1..65535")
    return parsed


def _read_secret(path: Path, *, label: str) -> str:
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not 16 <= len(value.encode("utf-8")) <= 512:
        raise ValueError(f"{label} token length is invalid")
    return value


def _validate_job_id(value: str) -> str:
    normalized = str(value).strip()
    if len(normalized) != 32 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("job id must be a 32-character lowercase hexadecimal identifier")
    return normalized


def _validate_control_url(value: str) -> str:
    normalized = str(value).strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("control URL must be an absolute HTTP(S) URL without credentials or query")
    return normalized


def _configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        resolved = log_file.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(resolved, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class ControlPlanePresencePublisher:
    """POST only one model observation; no control response drives a robot action."""

    def __init__(self, endpoint: str, token: str, *, timeout_s: float = 1.0) -> None:
        self.endpoint = _validate_control_url(endpoint)
        self._token = token
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 10:
            raise ValueError("control timeout must be in (0, 10]")
        self._timeout_s = float(timeout_s)
        # ControlPlane is on the local private LAN. Environment HTTP proxies
        # cannot receive its session-scoped token or delay safety observations.
        self._opener = build_opener(ProxyHandler({}))
        self.published = 0
        self.failed = 0
        self._last_warning_monotonic = 0.0

    def publish(
        self,
        *,
        packet: HeadFramePacket,
        presence: Mapping[str, Any],
        stage: Mapping[str, Any] | None,
        status: str = "ok",
    ) -> bool:
        observed_at_ns = packet.received_timestamp_ns or time.time_ns()
        payload: dict[str, Any] = {
            "generation": packet.generation,
            "sequence": packet.sequence,
            "status": status,
            "presence": {**dict(presence), "observed_at_ns": int(observed_at_ns)},
            "frame": {
                "source": "robot_client_three_cameras",
                "capture_timestamp_ns": packet.capture_timestamp_ns,
                "received_timestamp_ns": int(observed_at_ns),
            },
        }
        if stage is not None:
            payload["stage"] = dict(stage)
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_CONTROL_BODY_BYTES:
            raise ValueError("presence result body exceeds the bounded control-plane limit")
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Nexus-Presence-Token": self._token,
            },
        )
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                response.read(MAX_CONTROL_BODY_BYTES)
                if not 200 <= response.status < 300:
                    raise OSError(f"control plane returned HTTP {response.status}")
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            self.failed += 1
            now = time.monotonic()
            if now - self._last_warning_monotonic >= 10.0:
                LOGGER.warning("Presence observation was not accepted by the control plane: %s", error)
                self._last_warning_monotonic = now
            return False
        self.published += 1
        return True


def _unknown_presence(reason: str) -> dict[str, Any]:
    """Return a bounded UNKNOWN result when image/model processing cannot proceed."""

    return {
        "state": UNKNOWN,
        "confidence": 0.0,
        "changed": False,
        "raw_state": UNKNOWN,
        "raw_confidence": 0.0,
        "candidate_state": None,
        "candidate_frames": 0,
        "window_states": [],
        "camera_windows": {},
        "package_count": 0,
        "qualifying_package_count": 0,
        "ignored_package_count": 0,
        "reason": str(reason).strip()[:MAX_REASON_LENGTH] or "presence observation unavailable",
        "raw_reason": "sidecar could not process a complete three-camera capture",
        "schema_version": "1.0",
    }


def _decode_jpeg(jpeg: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    if encoded.size < 3:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class ThreeCameraPresenceAggregator:
    """Require three consecutive real inference samples from every camera."""

    def __init__(self, head_gate: ParcelPresenceGate) -> None:
        head_config = head_gate.config
        full_frame_config = replace(
            head_config,
            include_roi_names=("frame",),
            exclude_roi_names=(),
            polygons={"frame": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
            window_frames=3,
            allow_gripper_outside_roi=True,
            ambiguous_roi_is_unknown=False,
        )
        self._wrists = {
            "cam_left_wrist": ParcelPresenceGate(full_frame_config),
            "cam_right_wrist": ParcelPresenceGate(full_frame_config),
        }
        self._last_state = UNKNOWN

    def reset(self) -> None:
        for gate in self._wrists.values():
            gate.reset()
        self._last_state = UNKNOWN

    def update(
        self,
        head: PresenceResult,
        wrist_frames: Mapping[str, np.ndarray],
        detector: Any,
    ) -> dict[str, Any]:
        results = {"cam_head": head}
        for camera, gate in self._wrists.items():
            frame = wrist_frames[camera]
            results[camera] = gate.update(detector.predict(frame), frame.shape[:2])
        windows = {camera: list(results[camera].window_states) for camera in CAMERA_NAMES}
        states = {camera: results[camera].state for camera in CAMERA_NAMES}
        if all(state == NO_PACKAGE for state in states.values()):
            state = NO_PACKAGE
        elif any(state == PACKAGE_PRESENT for state in states.values()):
            state = PACKAGE_PRESENT
        else:
            state = UNKNOWN
        result = head.to_dict()
        result["state"] = state
        result["changed"] = state != self._last_state
        result["confidence"] = (
            min(item.confidence for item in results.values()) if state == NO_PACKAGE
            else max((item.confidence for item in results.values() if item.state == PACKAGE_PRESENT), default=0.0)
        )
        result["window_states"] = [state] * 3 if state != UNKNOWN else []
        result["camera_windows"] = windows
        raw_states = [item.raw_state for item in results.values()]
        result["raw_state"] = (
            PACKAGE_PRESENT if PACKAGE_PRESENT in raw_states else
            NO_PACKAGE if all(item == NO_PACKAGE for item in raw_states) else UNKNOWN
        )
        result["raw_confidence"] = max(item.raw_confidence for item in results.values())
        result["candidate_state"] = None
        result["candidate_frames"] = 0
        for count in ("package_count", "qualifying_package_count", "ignored_package_count"):
            result[count] = max(getattr(item, count) for item in results.values())
        result["reason"] = "three-camera presence: " + ", ".join(
            f"{camera}={states[camera]}" for camera in CAMERA_NAMES
        )
        result["raw_reason"] = "raw three-camera presence: " + ", ".join(
            f"{camera}={results[camera].raw_state}" for camera in CAMERA_NAMES
        )
        self._last_state = state
        return result


def build_parser() -> argparse.ArgumentParser:
    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bind-host", required=True, help="Thor private-LAN address accepting robot JPEGs")
    parser.add_argument("--port", type=_port, required=True, help="Authenticated TCP ingress port")
    parser.add_argument("--job-id", required=True, help="Nexus deployment ID")
    parser.add_argument("--generation", type=_positive_int, required=True, help="Current robot-client generation")
    parser.add_argument("--ingress-token-file", type=Path, required=True, help="0600 robot-frame token file")
    parser.add_argument("--control-url", required=True, help="HTTP URL of this job's control-plane endpoint")
    parser.add_argument("--control-token-file", type=Path, required=True, help="0600 sidecar-to-control token file")
    parser.add_argument("--weights", type=Path, default=defaults.weights)
    parser.add_argument("--rules", type=Path, default=defaults.rules)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--detect-hz", type=float, default=defaults.detect_hz)
    parser.add_argument("--camera-fps", type=float, default=defaults.camera_fps)
    parser.add_argument("--ready-file", type=Path, required=True, help="Atomic ready JSON written after model warmup")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--control-timeout-s", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging(args.log_file)
    job_id = _validate_job_id(args.job_id)
    if not math.isfinite(args.detect_hz) or not 0 < args.detect_hz <= 15:
        raise ValueError("detect-hz must be in (0, 15]")
    if not math.isfinite(args.camera_fps) or args.camera_fps < args.detect_hz:
        raise ValueError("camera-fps must be finite and no less than detect-hz")
    ingress_token = _read_secret(args.ingress_token_file, label="ingress")
    control_token = _read_secret(args.control_token_file, label="control")
    control_url = _validate_control_url(args.control_url)
    runtime_config = RuntimeConfig(
        weights=args.weights,
        rules=args.rules,
        device=args.device,
        detect_hz=args.detect_hz,
        camera_fps=args.camera_fps,
    )
    mailbox = LatestFrameMailbox()
    server = PresenceFrameIngressServer(
        (args.bind_host, args.port),
        job_id=job_id,
        generation=args.generation,
        ingress_token=ingress_token,
        mailbox=mailbox,
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="PresenceFrameIngress",
        daemon=True,
    )
    server_started = False
    stop_event = threading.Event()

    def request_stop(*_unused: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    estimator: ParcelStateEstimator | None = None
    try:
        estimator = ParcelStateEstimator(runtime_config)
        estimator.presence_gate = ParcelPresenceGate(
            replace(estimator.presence_gate.config, window_frames=3)
        )
        aggregator = ThreeCameraPresenceAggregator(estimator.presence_gate)
        estimator.warmup()
        scheduler = RateLimitedParcelStateService(estimator)
        publisher = ControlPlanePresencePublisher(
            control_url,
            control_token,
            timeout_s=args.control_timeout_s,
        )
        server_thread.start()
        server_started = True
        _write_json_atomically(
            args.ready_file,
            {
                "status": "ready",
                "pid": os.getpid(),
                "job_id": job_id,
                "generation": args.generation,
                "bind_host": args.bind_host,
                "port": args.port,
                "detect_hz": args.detect_hz,
                "model_sha256": estimator.model_sha256,
                "started_at": time.time(),
            },
        )
        LOGGER.info(
            "Nexus presence sidecar ready: job=%s generation=%s ingress=tcp://%s:%s detect_hz=%.1f",
            job_id,
            args.generation,
            args.bind_host,
            args.port,
            args.detect_hz,
        )
        last_frame_monotonic = None
        # The robot's inference preflight runs before its first presence-frame
        # publisher starts. Exercise the same three YOLO calls meanwhile so
        # policy latency is measured under GPU contention. Synthetic results
        # never enter the parcel gate or the control plane.
        preflight_frame = np.zeros(
            (runtime_config.expected_height, runtime_config.expected_width, 3),
            dtype=np.uint8,
        )
        preflight_deadline = time.monotonic() + 20.0
        next_preflight_detection = time.monotonic()
        preflight_detections = 0
        while not stop_event.is_set():
            now = time.monotonic()
            synthetic_due = (
                last_frame_monotonic is None
                and now < preflight_deadline
                and now >= next_preflight_detection
            )
            packet = mailbox.take(timeout_s=0.02 if synthetic_due else 0.1)
            if packet is None:
                if synthetic_due and not stop_event.is_set():
                    try:
                        for _ in CAMERA_NAMES:
                            estimator.detector.predict(preflight_frame)
                    except Exception:
                        LOGGER.exception("Presence preflight GPU load failed; continuing with live frames")
                        preflight_deadline = time.monotonic()
                    else:
                        preflight_detections += 1
                        next_preflight_detection = time.monotonic() + max(
                            0.0, 1.0 / runtime_config.detect_hz - 0.02
                        )
                continue
            if last_frame_monotonic is None:
                LOGGER.info(
                    "Presence preflight GPU load ended after %d three-camera detections",
                    preflight_detections,
                )
                preflight_frame = None
            now = time.monotonic()
            if last_frame_monotonic is None or now - last_frame_monotonic > 0.75:
                scheduler.reset()
                aggregator.reset()
            last_frame_monotonic = now
            frames = packet.camera_jpegs
            decoded = tuple(_decode_jpeg(jpeg) for jpeg in frames) if frames is not None else ()
            if len(decoded) != 3 or any(frame is None for frame in decoded):
                scheduler.reset()
                aggregator.reset()
                publisher.publish(
                    packet=packet,
                    presence=_unknown_presence("three-camera capture missing or JPEG decode failed"),
                    stage=None,
                    status="decode_error",
                )
                continue
            try:
                result = scheduler.process_with_presence_if_due(
                    decoded[0],
                    color_space="BGR",
                    source_id="robot_client_cam_head",
                    frame_index=packet.sequence,
                    capture_timestamp_ns=packet.capture_timestamp_ns,
                )
                presence = None if result is None else aggregator.update(
                    result.presence,
                    dict(zip(CAMERA_NAMES[1:], decoded[1:])),
                    estimator.detector,
                )
            except Exception as error:  # Model failures must never become a robot action.
                scheduler.reset()
                aggregator.reset()
                LOGGER.exception("Presence model failed for the newest three-camera capture")
                publisher.publish(
                    packet=packet,
                    presence=_unknown_presence(f"presence model error: {error}"),
                    stage=None,
                    status="inference_error",
                )
                continue
            if result is None:
                continue
            publisher.publish(
                packet=packet,
                presence=presence,
                stage=result.stage.to_dict(),
            )
    finally:
        if server_started:
            server.shutdown()
        server.server_close()
        if server_started:
            server_thread.join(timeout=1.0)
        if estimator is not None:
            estimator.close()
        LOGGER.info("Nexus presence sidecar stopped")


if __name__ == "__main__":
    main()
