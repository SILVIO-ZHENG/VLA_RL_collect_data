#!/usr/bin/env python3
"""Algorithm-neutral robot-side G2A client for remote policy inference."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .action_filter import (
    AdaptiveFixedTimeChunkSmoother,
    CausalLowPassFilter,
    ChunkTrajectoryShaper,
    FixedHorizonChunkProjector,
)
from .action_plan import ActionPlan
from .command_prefix import build_command_prefix
from .action_trace import ActionTraceRecorder, ControlDiagnostics
from .async_safety import servo_preparation_time
from .constants import (
    ACTION_DIM,
    APPROVED_START_STATE_24D,
    ARM_CANONICAL_INDICES,
    CAMERA_KEYS,
    CANONICAL_STATE_NAMES,
    EFFECTOR_CANONICAL_INDICES,
    HEAD_CANONICAL_INDICES,
    WAIST_CANONICAL_INDICES,
)
from .effector_command import EffectorEndpointLatch
from .effector_profiles import EFFECTOR_PROFILE_CHOICES, get_effector_profile
from .gdk_adapter import AdapterConfig, G2AAdapter, SafetyDispatchExpiredError
from .presence_stream import LatestHeadFramePublisher
from .grasp_coordinator import GraspEventCoordinator
from .hitl.g2_runtime.runner_mixin import FlowPROControlMixin
from .realtime_cadence import RealtimeCadence, prepare_realtime_cadence
from .telemetry import LatestInferenceFrameWriter, LatestTelemetryPublisher
from .transport import PolicyClient

logger = logging.getLogger(__name__)


@contextmanager
def safety_thread_scheduling(enabled: bool) -> Iterator[None]:
    """Reduce Python-thread handoff latency in this client process only.

    This is not a realtime guarantee or an OS scheduling priority change.
    Preserve a caller's shorter interval and restore it on every exit.
    """
    previous = sys.getswitchinterval()
    interval = min(previous, 0.001) if enabled else previous
    if interval != previous:
        sys.setswitchinterval(interval)
    try:
        if enabled:
            logger.info("Safety Python thread switch interval=%.3fms", interval * 1000)
        yield
    finally:
        if interval != previous:
            sys.setswitchinterval(previous)


@contextmanager
def gc_pause_trace(enabled: bool, path: Path) -> Iterator[None]:
    """Record long GC pauses in memory; write after command streaming stops."""
    if not enabled:
        yield
        return
    starts: dict[int, tuple[float, float]] = {}
    pauses: deque[tuple[float, float, float, int]] = deque(maxlen=128)

    def callback(phase: str, info: dict[str, Any]) -> None:
        tid = threading.get_native_id()
        now = time.perf_counter()
        if phase == "start":
            starts[tid] = (now, time.thread_time())
        elif phase == "stop" and tid in starts:
            started, cpu_started = starts.pop(tid)
            if now - started >= 0.005:
                pauses.append((started, (now - started) * 1000,
                               (time.thread_time() - cpu_started) * 1000,
                               int(info.get("generation", -1))))

    gc.callbacks.append(callback)
    try:
        yield
    finally:
        gc.callbacks.remove(callback)
        if pauses:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as stream:
                for started, wall_ms, cpu_ms, generation in pauses:
                    stream.write(json.dumps(dict(monotonic=started, wall_ms=wall_ms,
                                                 cpu_ms=cpu_ms, generation=generation)) + "\n")


class RobotClientRunner(FlowPROControlMixin):
    def __init__(
        self,
        adapter: G2AAdapter,
        policy_client: PolicyClient,
        *,
        task: str,
        inference_mode: str,
        hardware_mode: str,
        fps: float,
        servo_fps: float,
        prediction_horizon: int,
        quality_horizon: int,
        min_replan_after_steps: int,
        request_lead_margin_steps: int,
        min_rtc_blend_steps: int,
        latency_window_size: int,
        warmup_requests: int,
        execution_preflight_requests: int,
        watchdog_timeout_s: float,
        initial_head_waist_warning_threshold: float,
        max_arm_tracking_error: float,
        max_effector_tracking_error: float,
        max_head_waist_tracking_error: float,
        tracking_error_consecutive_limit: int,
        action_trace_dir: Path | None = None,
        trace_save_images: bool = True,
        freeze_trace_observation: bool = False,
        action_low_pass_cutoff_hz: float | None = None,
        effector_action_low_pass_cutoff_hz: float | None = None,
        effector_command_mode: str = "continuous",
        effector_closed_max: float = 0.0075,
        effector_open_min: float = 0.0175,
        chunk_smoothing_mode: str = "savgol",
        chunk_smoothing_window: int = 7,
        chunk_smoothing_max_deviation: float = 0.005,
        adaptive_smoothing_acceleration_weight: float = 20.0,
        adaptive_smoothing_jerk_weight: float = 5.0,
        adaptive_smoothing_velocity_reference: float = 0.25,
        fixed_horizon_projection: bool = False,
        action_interpolation: str = "linear",
        hold_effectors: bool = False,
        timing_log_file: Path | None = None,
        control_file: Path | None = None,
        control_status_file: Path | None = None,
        platform_frame_dir: Path | None = None,
        presence_frame_publisher: LatestHeadFramePublisher | None = None,
        presence_capture_fps: float = 10.0,
        start_paused: bool = False,
        coordinate_grasp_events: bool = False,
        rt_cadence: bool = False,
        pause_on_collision: bool = False,
        rtc_state_aligned_anchor: bool = False,
        rtc_prefix_mode: str = "model",
    ) -> None:
        self.adapter = adapter
        self.policy_client = policy_client
        self.task = task
        self.inference_mode = inference_mode
        self.hardware_mode = hardware_mode
        self.pause_on_collision = bool(pause_on_collision)
        self.rtc_state_aligned_anchor = bool(rtc_state_aligned_anchor)
        if rtc_prefix_mode not in {"model", "command-shadow", "command"}:
            raise ValueError("Invalid RTC prefix mode")
        if rtc_prefix_mode == "command" and hardware_mode == "execute":
            raise ValueError("Command-prefix inference is restricted to read-only/dry-run testing")
        if rtc_prefix_mode != "model" and (inference_mode != "rtc" or not self.rtc_state_aligned_anchor):
            raise ValueError("Command prefix requires state-aligned RTC")
        self.rtc_prefix_mode = rtc_prefix_mode
        if self.rtc_state_aligned_anchor and freeze_trace_observation:
            raise ValueError("State-aligned RTC cannot reuse a frozen observation")
        self._safety_pause_reason: str | None = None
        self.rt_cadence = bool(rt_cadence)
        if self.rt_cadence and self.hardware_mode != "execute":
            raise ValueError("Realtime cadence is available only for physical execution")
        self.fps = fps
        self.servo_fps = servo_fps
        self.prediction_horizon = prediction_horizon
        self.quality_horizon = quality_horizon
        self.min_replan_after_steps = min_replan_after_steps
        self.request_lead_margin_steps = request_lead_margin_steps
        self.min_rtc_blend_steps = min_rtc_blend_steps
        self.warmup_requests = warmup_requests
        self.execution_preflight_requests = execution_preflight_requests
        self.watchdog_timeout_s = watchdog_timeout_s
        self.initial_head_waist_warning_threshold = initial_head_waist_warning_threshold
        self.max_arm_tracking_error = max_arm_tracking_error
        self.max_effector_tracking_error = max_effector_tracking_error
        self.max_head_waist_tracking_error = max_head_waist_tracking_error
        self.tracking_error_consecutive_limit = tracking_error_consecutive_limit
        self.session_id = uuid.uuid4().hex
        device_position = getattr(getattr(adapter, "config", None), "effector_control_mode", "software-trajectory") == "device-position"
        chunk_projector = None
        if fixed_horizon_projection:
            limiter = getattr(adapter, "_kinematic_limiter", None)
            if limiter is None:
                raise ValueError("fixed-horizon projection requires the adapter kinematic limiter")
            chunk_projector = FixedHorizonChunkProjector(
                policy_hz=fps,
                action_dim=ACTION_DIM,
                projected_indices=tuple(ARM_CANONICAL_INDICES),
                max_velocity=limiter.max_velocity,
                max_acceleration=limiter.max_acceleration,
            )
        self.plan = ActionPlan(
            ACTION_DIM, policy_fps=fps,
            position_target_indices=tuple(EFFECTOR_CANONICAL_INDICES) if device_position else (),
            chunk_projector=chunk_projector,
            interpolation_mode=action_interpolation,
            cubic_indices=tuple(ARM_CANONICAL_INDICES),
        )
        if device_position:
            effector_action_low_pass_cutoff_hz = None
        self.shutdown_event = threading.Event()
        self._execution_armed_event = threading.Event()
        if self.hardware_mode != "execute":
            self._execution_armed_event.set()
        self._control_lock = threading.Lock()
        self._session_lock = threading.RLock()
        self._communication_thread: threading.Thread | None = None
        self._action_thread: threading.Thread | None = None
        self._read_only_telemetry_thread: threading.Thread | None = None
        self._last_response_monotonic: float | None = None
        self._last_round_trip_s: float | None = None
        self._chunk_safety_context: dict[str, dict[str, Any]] = {}
        self.flowpro_capture: Any | None = None
        self._flowpro_chunk_lineage: dict[str, dict[str, Any]] = {}
        self._flowpro_arbitration_monitor: Any | None = None
        self._last_flowpro_sample_ns: int | None = None
        self._flowpro_rollback_lock = threading.Lock()
        self._flowpro_rollback_executor: Any | None = None
        self._flowpro_motion_bridge: Any | None = None
        self._flowpro_rollback_deadman = False
        self._flowpro_rollback_finish_to_anchor = False
        self._flowpro_rollback_health: Any | None = None
        self._flowpro_rollback_health_updated_monotonic: float | None = None
        self._flowpro_rollback_lease_id: str | None = None
        self._flowpro_rollback_prepared_lease_id: str | None = None
        self._flowpro_rollback_last_result: Any | None = None
        self._flowpro_rollback_cancel_reason: str | None = None
        self._flowpro_human_control_active_event = threading.Event()
        self._flowpro_hold_reanchor_requested_event = threading.Event()
        self._flowpro_hold_reanchor_ack_event = threading.Event()
        self._flowpro_human_epoch_lock = threading.Lock()
        self._flowpro_human_epoch = 0
        self._flowpro_hold_reanchor_request_epoch: int | None = None
        self._flowpro_hold_reanchor_ack_epoch: int | None = None
        self._flowpro_hold_reanchor_error: str | None = None
        self._active_safety_chunk_id: str | None = None
        self._active_safety_action_index = -1
        self._observation_latencies_s: deque[float] = deque(maxlen=latency_window_size)
        self._state_latencies_s: deque[float] = deque(maxlen=latency_window_size)
        self._last_action: np.ndarray | None = None
        self._request_number = 0
        self._executed_actions = 0
        self._servo_ticks = 0
        self._logged_image_transport = False
        self._previous_servo_tick: float | None = None
        self._servo_window_started = time.perf_counter()
        self._servo_intervals_ms: list[float] = []
        self._servo_send_ms: list[float] = []
        self._servo_deadline_misses = 0
        # File/console logging can block. Keep periodic servo reports in
        # memory until command streaming has stopped.
        self._servo_timing_reports: deque[tuple[str, tuple[Any, ...]]] = deque(maxlen=600)
        self._freeze_trace_observation = freeze_trace_observation
        self._frozen_trace_observation: dict[str, Any] | None = None
        self._hold_effectors = hold_effectors
        self._fatal_error: str | None = None
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._held_effector_values: np.ndarray | None = None
        self._held_head_waist_values: np.ndarray | None = None
        self._watchdog_holding = False
        self._queue_underrun_active = False
        self._tracking_error_consecutive = 0
        self._effector_tracking_consecutive = [0, 0]
        self._effector_tracking_seen_revision = [0, 0]
        self._effector_command_revision = [0, 0]
        self._effector_last_command: np.ndarray | None = None
        self._timing_log_file = timing_log_file.expanduser().resolve() if timing_log_file else None
        self.control_diagnostics = (ControlDiagnostics(
            self._timing_log_file.parent / f"control-{self.session_id}.jsonl"
        ) if self._timing_log_file else None)
        if self.control_diagnostics is not None:
            self.control_diagnostics.write(dict(
                record_type="trace_start", session_id=self.session_id,
                trace_schema_version=2,
                hardware_mode=self.hardware_mode, policy_fps=self.fps, servo_fps=self.servo_fps,
                inference_mode=self.inference_mode,
                rt_cadence=self.rt_cadence,
                fixed_horizon_projection=bool(fixed_horizon_projection),
                chunk_smoothing_mode=chunk_smoothing_mode,
                chunk_smoothing_window=int(chunk_smoothing_window),
                chunk_smoothing_max_deviation=float(chunk_smoothing_max_deviation),
                adaptive_smoothing_acceleration_weight=float(adaptive_smoothing_acceleration_weight),
                adaptive_smoothing_jerk_weight=float(adaptive_smoothing_jerk_weight),
                adaptive_smoothing_velocity_reference=float(adaptive_smoothing_velocity_reference),
                action_low_pass_cutoff_hz=action_low_pass_cutoff_hz,
                effector_action_low_pass_cutoff_hz=effector_action_low_pass_cutoff_hz,
                effector_command_mode=effector_command_mode,
                effector_closed_max=effector_closed_max,
                effector_open_min=effector_open_min,
                clock_source="monotonic/perf_counter (same OS clock on supported Linux)",
                adapter_config=vars(adapter.config).copy() if hasattr(adapter, "config") else {},
            ))
        self._control_file = control_file.expanduser().resolve() if control_file else None
        self._control_status_file = (
            control_status_file.expanduser().resolve() if control_status_file else None
        )
        self._control_thread: threading.Thread | None = None
        self._inference_frame_writer = (
            LatestInferenceFrameWriter(platform_frame_dir)
            if platform_frame_dir is not None
            else None
        )
        if not math.isfinite(presence_capture_fps) or not 0 < presence_capture_fps <= 15:
            raise ValueError("presence_capture_fps must be in (0, 15]")
        self._presence_frame_publisher = presence_frame_publisher
        self._presence_capture_fps = float(presence_capture_fps)
        self._presence_capture_thread: threading.Thread | None = None
        self._start_paused = bool(start_paused)
        self._camera_capture_lock = threading.Lock()
        self._inference_capture_requested = threading.Event()
        self._last_inference_camera_capture_monotonic = float("-inf")
        self._pause_requested_event = threading.Event()
        self._pause_active_event = threading.Event()
        self._resume_ready_event = threading.Event()
        self._resume_ready_event.set()
        self._grasp_hold_event = threading.Event()
        self._grasp_replan_event = threading.Event()
        if self._timing_log_file is not None:
            self._timing_log_file.parent.mkdir(parents=True, exist_ok=True)
        self._action_filter = (
            CausalLowPassFilter(
                cutoff_hz=action_low_pass_cutoff_hz,
                sample_hz=servo_fps,
                action_dim=ACTION_DIM,
            )
            if action_low_pass_cutoff_hz is not None
            else None
        )
        if chunk_smoothing_mode == "adaptive-fixed-time":
            self._chunk_shaper = AdaptiveFixedTimeChunkSmoother(
                action_dim=ACTION_DIM,
                shaped_indices=tuple(ARM_CANONICAL_INDICES),
                policy_hz=fps,
                max_deviation=chunk_smoothing_max_deviation,
                acceleration_weight=adaptive_smoothing_acceleration_weight,
                jerk_weight=adaptive_smoothing_jerk_weight,
                velocity_reference=adaptive_smoothing_velocity_reference,
            )
        elif chunk_smoothing_mode == "savgol":
            self._chunk_shaper = ChunkTrajectoryShaper(
                action_dim=ACTION_DIM,
                shaped_indices=tuple(ARM_CANONICAL_INDICES),
                window=chunk_smoothing_window,
                max_deviation=chunk_smoothing_max_deviation,
            )
        else:
            raise ValueError(f"Unsupported chunk smoothing mode: {chunk_smoothing_mode}")
        self._grasp_coordinator = (
            GraspEventCoordinator(
                action_dim=ACTION_DIM,
                left_arm=tuple(range(0, 7)), right_arm=tuple(range(8, 15)),
                effector_indices=tuple(EFFECTOR_CANONICAL_INDICES),
                state_range=get_effector_profile(getattr(adapter.config, "effector_profile", "dahuan")).hardware_state_range,
            )
            if coordinate_grasp_events and device_position else None
        )
        self._effector_action_filter = (
            CausalLowPassFilter(
                cutoff_hz=effector_action_low_pass_cutoff_hz,
                sample_hz=servo_fps,
                action_dim=len(EFFECTOR_CANONICAL_INDICES),
            )
            if effector_action_low_pass_cutoff_hz is not None
            else None
        )
        self._effector_endpoint_latch = None
        if effector_command_mode == "endpoint-latch":
            profile = adapter.effector_profile
            closed_endpoint, open_endpoint = profile.hardware_state_range
            self._effector_endpoint_latch = EffectorEndpointLatch(
                dimensions=len(EFFECTOR_CANONICAL_INDICES),
                closed_endpoint=closed_endpoint,
                open_endpoint=open_endpoint,
                closed_max=effector_closed_max,
                open_min=effector_open_min,
            )
        elif effector_command_mode != "continuous":
            raise ValueError(f"Unsupported effector command mode: {effector_command_mode}")
        self.trace_recorder = (
            ActionTraceRecorder(
                action_trace_dir,
                session_id=self.session_id,
                task=self.task,
                policy_fps=self.fps,
                servo_fps=self.servo_fps,
                inference_mode=self.inference_mode,
                server_metadata=self.policy_client.metadata,
                save_images=trace_save_images,
                freeze_observation=freeze_trace_observation,
                action_low_pass_cutoff_hz=action_low_pass_cutoff_hz,
                effector_action_low_pass_cutoff_hz=effector_action_low_pass_cutoff_hz,
                hold_effectors=hold_effectors,
            )
            if action_trace_dir is not None
            else None
        )
        if self.trace_recorder is not None:
            logger.info(
                "READ-ONLY action trace enabled; results will be saved to %s",
                self.trace_recorder.run_dir,
            )

    def validate_server(self) -> None:
        metadata = self.policy_client.metadata
        if metadata.get("canonical_state_dim") != ACTION_DIM:
            raise RuntimeError("Server canonical state dimension must be 24")
        if metadata.get("canonical_action_dim") != ACTION_DIM:
            raise RuntimeError("Server canonical action dimension must be 24")
        if metadata.get("action_space") != "absolute":
            raise RuntimeError("Server must return absolute actions")
        server_effector_profile = str(metadata.get("effector_profile", ""))
        client_profile = getattr(getattr(self, "adapter", None), "effector_profile", None)
        client_effector_profile = str(getattr(client_profile, "name", ""))
        if client_effector_profile and server_effector_profile != client_effector_profile:
            raise RuntimeError(
                "Server/client effector profile mismatch: "
                f"server={server_effector_profile or 'missing'}, "
                f"client={client_effector_profile}"
            )
        if tuple(metadata.get("camera_keys", ())) != CAMERA_KEYS:
            raise RuntimeError(f"Camera key mismatch: {metadata.get('camera_keys')}")
        image_transport = metadata.get("image_transport", {})
        if not image_transport.get("encoded_payloads", False):
            raise RuntimeError("Server does not advertise encoded image payload support")
        if self.inference_mode == "rtc" and not metadata.get("rtc", {}).get("available", False):
            raise RuntimeError("RTC requested but server does not advertise RTC capability")
        if self.inference_mode == "rtc":
            execution_horizon = int(metadata.get("rtc", {}).get("execution_horizon", 0))
            if execution_horizon <= 0 or execution_horizon > self.quality_horizon:
                raise RuntimeError(
                    "Invalid server RTC execution horizon: "
                    f"server={execution_horizon}, client quality_horizon={self.quality_horizon}. "
                    "The server horizon must be positive and no larger than quality_horizon."
                )
            logger.info(
                "Using server-advertised RTC execution_horizon=%d",
                execution_horizon,
            )
        server_chunk_size = int(metadata.get("chunk_size", 0))
        if server_chunk_size != self.prediction_horizon:
            raise RuntimeError(
                "Prediction horizon mismatch: "
                f"server chunk_size={server_chunk_size}, client={self.prediction_horizon}"
            )
        # The server's cadence is model provenance, not an execution
        # constraint.  All wall-clock scheduling is owned by this client.
        # Accept the legacy ``fps`` key while mixed-version deployments are
        # being upgraded.
        advertised_model_fps = metadata.get("model_fps", metadata.get("fps"))
        if advertised_model_fps is not None:
            model_fps = float(advertised_model_fps)
            if not math.isclose(model_fps, self.fps, rel_tol=0.0, abs_tol=1e-6):
                logger.warning(
                    "Model was trained at %.1f FPS but actions will be executed at %.1f FPS; "
                    "the trajectory time scale is %.3fx relative to training",
                    model_fps,
                    self.fps,
                    model_fps / self.fps,
                )

    def _next_request_id(self) -> str:
        self._request_number += 1
        return f"{self.session_id}-{self._request_number:08d}"

    def _capture_observation(self, *, capture_plan_anchor: bool = False) -> dict[str, Any]:
        # Camera calls use the independent GDK Camera object and deliberately
        # do not hold the Robot/action lock.
        started = time.perf_counter()
        camera_start_ns = time.time_ns()
        state_snapshot = None
        state_candidates = []

        def capture_state_snapshot() -> None:
            nonlocal state_snapshot
            state_lock_requested = time.perf_counter()
            with self._control_lock:
                state_lock_acquired = time.perf_counter()
                state = self.adapter.read_state()
                state_read_returned = time.perf_counter()
                # Keep command and feedback from the same protected instant.
                tracking_command = (
                    None
                    if self._last_action is None
                    else np.asarray(self._last_action, dtype=np.float32).copy()
                )
                effector_revisions = tuple(self._effector_command_revision)
                state_capture_monotonic_time = time.perf_counter()
                state_time_ns = time.time_ns()
                plan_anchor = self.plan.snapshot() if capture_plan_anchor else None
                command_prefix = None
                if capture_plan_anchor and getattr(self, "rtc_prefix_mode", "model") != "model":
                    command_prefix = self._capture_rtc_command_prefix(
                        state_capture_monotonic_time, plan_anchor,
                    )
            state_snapshot = (
                state, tracking_command, effector_revisions,
                state_lock_requested, state_lock_acquired, state_read_returned,
                state_capture_monotonic_time, state_time_ns,
                plan_anchor, command_prefix,
            )
            if self.flowpro_capture is not None:
                state_candidates.append(state_snapshot)

        self._inference_capture_requested.set()
        try:
            with self._camera_capture_lock:
                camera_lock_acquired = time.perf_counter()
                if self.flowpro_capture is not None:
                    for attempt in range(2):
                        state_candidates.clear()
                        # The latest frame can be almost one capture tick old.
                        # Bracket selection with real state reads and retain
                        # the one closest to all three camera timestamps.
                        capture_state_snapshot()
                        images, camera_frame_times = self.adapter.capture_images(
                            before_processing=capture_state_snapshot,
                        )
                        clock_offset_ns = self.adapter.camera_wall_to_vendor_offset_ns
                        if clock_offset_ns is None:
                            raise RuntimeError("FlowPRO camera clock bridge is unavailable")
                        camera_frame_times = {
                            key: int(stamp) + clock_offset_ns
                            for key, stamp in camera_frame_times.items()
                        }
                        state_snapshot = min(
                            state_candidates,
                            key=lambda candidate: max(
                                abs(candidate[7] - stamp)
                                for stamp in camera_frame_times.values()
                            ),
                        )
                        skew_ms = max(
                            abs(state_snapshot[7] - stamp)
                            for stamp in camera_frame_times.values()
                        ) / 1e6
                        if skew_ms <= 40.0:
                            break
                        if attempt:
                            raise RuntimeError(
                                f"FlowPRO observation state/camera skew {skew_ms:.1f} ms exceeds 40 ms"
                            )
                        logger.warning(
                            "FlowPRO camera/state skew %.1f ms; retrying one fresh capture",
                            skew_ms,
                        )
                else:
                    images, camera_frame_times = self.adapter.capture_images()
                    clock_offset_ns = None
                image_transport_timing = self.adapter.last_image_transport_timing
                camera_pipeline_timing = getattr(self.adapter, "last_camera_capture_timing", {})
                self._last_inference_camera_capture_monotonic = time.monotonic()
        finally:
            self._inference_capture_requested.clear()
        camera_end_ns = time.time_ns()
        camera_duration_ms = (time.perf_counter() - started) * 1000
        if not self._logged_image_transport:
            summary = {
                key: {
                    "encoding": image.get("encoding"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "bytes": len(image.get("data", b"")),
                }
                for key, image in images.items()
            }
            logger.info("Camera transport payloads: %s", summary)
            self._logged_image_transport = True
        if state_snapshot is None:
            capture_state_snapshot()
        (
            state, tracking_command, effector_revisions,
            state_lock_requested, state_lock_acquired, state_read_returned,
            state_capture_monotonic_time, state_time_ns,
            plan_anchor, command_prefix,
        ) = state_snapshot
        self._check_tracking_error(
            state, command=tracking_command, effector_revisions=effector_revisions,
        )
        frame_times = list(camera_frame_times.values())
        if self.flowpro_capture is not None:
            skew_ms = max(abs(state_time_ns - stamp) for stamp in frame_times) / 1e6
            if skew_ms > 40.0:
                raise RuntimeError(
                    f"FlowPRO observation state/camera skew {skew_ms:.1f} ms exceeds 40 ms"
                )
        observation = {
            "state": state,
            "images": images,
            "task": self.task,
            # Use the beginning of the multi-camera/state acquisition as the
            # conservative age of this complete observation.
            "observation_monotonic_time": started,
            "state_capture_monotonic_time": state_capture_monotonic_time,
            "state_lock_wait_ms": (state_lock_acquired - state_lock_requested) * 1000,
            "state_read_ms": (state_read_returned - state_lock_acquired) * 1000,
            "state_capture_wall_time_ns": state_time_ns,
            "camera_capture_start_wall_time_ns": camera_start_ns,
            "camera_capture_wall_time_ns": camera_end_ns,
            "camera_capture_duration_ms": camera_duration_ms,
            "camera_lock_wait_ms": (camera_lock_acquired - started) * 1000,
            "camera_pipeline_timing": camera_pipeline_timing,
            "camera_frame_wall_time_ns": camera_frame_times,
            "camera_wall_to_vendor_offset_ns": clock_offset_ns,
            "camera_skew_ms": (max(frame_times) - min(frame_times)) / 1e6,
            "state_camera_skew_ms": abs(state_time_ns - max(frame_times)) / 1e6,
            "image_transport_timing": image_transport_timing,
        }
        if capture_plan_anchor:
            # Private metadata: communication removes it before serialization.
            observation["_plan_anchor"] = plan_anchor
            if command_prefix is not None:
                observation["_rtc_command_prefix"] = command_prefix
        return observation

    def _capture_rtc_command_prefix(self, origin, anchor):
        # Caller holds _control_lock, pairing queue/filter/plan and state clocks.
        delay = self._estimated_delay_steps()
        if delay <= 0 or delay > 21 or anchor.chunk_id is None:
            return None
        if getattr(self, "_grasp_coordinator", None) is not None:
            return None  # Future feedback-dependent grasp events are not predictable here.
        count = int(math.ceil((delay - 1) / self.fps / .01 - 1e-9)) + 2
        references = self.plan.preview_at(origin, period=.01, count=count)
        if references is None or self.adapter._last_state is None:
            return None
        references = references.copy()
        arm_filter = None if self._action_filter is None else self._action_filter.fork()
        eff_filter = None if self._effector_action_filter is None else self._effector_action_filter.fork()
        latch = getattr(self, "_effector_endpoint_latch", None)
        latch = None if latch is None else latch.fork()
        for index, point in enumerate(references):
            if self._held_head_waist_values is not None:
                point[list(HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES)] = self._held_head_waist_values
            if self._held_effector_values is not None:
                point[list(EFFECTOR_CANONICAL_INDICES)] = self._held_effector_values
            effector = point[list(EFFECTOR_CANONICAL_INDICES)].copy()
            references[index] = point if arm_filter is None else arm_filter.update(point)
            effector = effector if eff_filter is None else eff_filter.update(effector)
            if latch is not None:
                effector = latch.update(effector)
            references[index, list(EFFECTOR_CANONICAL_INDICES)] = effector
        packet = build_command_prefix(
            self.adapter._kinematic_limiter, self.adapter._last_state,
            list(getattr(self.adapter, "_async_steps", [])), references,
            policy_hz=self.fps, delay_steps=delay,
            effector_profile=self.adapter.effector_profile.name,
            clip_position=lambda q: self.adapter._apply_absolute_position_limits(q, warning_suffix="rtc_preview"),
        )
        packet.update(time_origin=origin, plan_generation=anchor.generation,
                      safety_generation=getattr(self.adapter, "_async_generation", 0),
                      active_chunk_id=anchor.chunk_id, next_action_index=anchor.next_action_index)
        diagnostics = getattr(self, "control_diagnostics", None)
        if diagnostics is not None:
            diagnostics.write(dict(record_type="rtc_command_prefix", mode=self.rtc_prefix_mode, **packet))
        return packet

    def _submit_presence_frame(
        self, images: dict[str, Any], camera_frame_times: dict[str, int]
    ) -> None:
        publisher = self._presence_frame_publisher
        if publisher is None:
            return
        captured_at_ns = camera_frame_times.get("cam_head")
        if captured_at_ns is not None:
            publisher.submit(images, capture_timestamp_ns=int(captured_at_ns))


    def _presence_capture_loop(self) -> None:
        """Keep package detection and the three-camera UI preview fresh.

        This thread only uses the independent camera object. It never reads
        robot state, calls the policy, changes the action plan, arms execution,
        or dispatches a GDK command. A recent inference capture suppresses the
        fallback capture so the normal inference path remains authoritative.
        """
        period_s = 1.0 / min(self._presence_capture_fps, 4.0)
        last_error_logged = 0.0
        while not self.shutdown_event.is_set():
            now = time.monotonic()
            paused = self._pause_requested_event.is_set() or self._pause_active_event.is_set()
            since_inference_capture = now - self._last_inference_camera_capture_monotonic
            if not paused and since_inference_capture < period_s:
                self.shutdown_event.wait(min(0.02, period_s - since_inference_capture))
                continue
            started = time.perf_counter()
            try:
                captured = self._capture_presence_images()
                if captured is None:
                    self.shutdown_event.wait(0.02)
                    continue
                images, camera_frame_times = captured
                capture_finished = time.perf_counter()
                if self.control_diagnostics is not None:
                    self.control_diagnostics.write({
                        "record_type": "presence_capture",
                        "capture_started_monotonic": started,
                        "capture_finished_monotonic": capture_finished,
                    })
                self._submit_presence_frame(images, camera_frame_times)

                if self._inference_frame_writer is not None:
                    captured_at_ns = max(
                        camera_frame_times.values(),
                        default=time.time_ns(),
                    )
                    self._inference_frame_writer.submit(
                        f"{self.session_id}-camera-{captured_at_ns}",
                        images,
                        captured_at=float(captured_at_ns) / 1e9,
                    )
            except Exception as error:
                now = time.monotonic()
                if now - last_error_logged >= 5.0:
                    logger.warning("Package-presence/UI camera capture failed: %s", error)
                    last_error_logged = now
            self.shutdown_event.wait(max(0.0, period_s - (time.perf_counter() - started)))

    def _capture_presence_images(self) -> tuple[dict[str, Any], dict[str, int]] | None:
        if not self.rtc_state_aligned_anchor:
            with self._camera_capture_lock:
                return self.adapter.capture_images()
        # Detection keeps all three cameras and continues during a pause.
        # Do not queue a fallback capture ahead of an inference waiter, or
        # duplicate a just-completed inference capture after waiting for it.
        if self._inference_capture_requested.is_set():
            return None
        if not self._camera_capture_lock.acquire(blocking=False):
            return None
        try:
            if self._inference_capture_requested.is_set():
                return None
            paused = self._pause_requested_event.is_set() or self._pause_active_event.is_set()
            period_s = 1.0 / min(self._presence_capture_fps, 4.0)
            if not paused and time.monotonic() - self._last_inference_camera_capture_monotonic < period_s:
                return None
            return self.adapter.capture_images()
        finally:
            self._camera_capture_lock.release()


    def _read_only_telemetry_loop(self) -> None:
        """Feed the browser viewer from observed state only (no action dispatch).

        The loop exists only for read-only/dry-run sessions, including soft
        pause.  Physical execute sessions retain their existing servo-owned
        telemetry path unchanged.
        """
        period_s = 0.10
        last_error_logged = 0.0
        while not self.shutdown_event.is_set():
            started = time.perf_counter()
            try:
                measured = self.adapter.read_state()
                self.adapter.publish_read_only_telemetry(measured)
            except Exception as error:
                now = time.monotonic()
                if now - last_error_logged >= 5.0:
                    logger.warning("Read-only viewer telemetry failed: %s", error)
                    last_error_logged = now
            self.shutdown_event.wait(max(0.0, period_s - (time.perf_counter() - started)))


    def _request_safety_expiry_pause(
        self, error: SafetyDispatchExpiredError
    ) -> None:
        """Fail closed into the normal measured-pose soft-pause handshake.

        The adapter raises this exception before an expired target reaches
        GDK.  Drop every unsent policy action and request the existing pause
        path, which re-reads the real pose, re-anchors safety, and acknowledges
        pause only after a freshly checked hold has passed through GDK.
        """
        self.plan.clear()
        self._resume_ready_event.clear()
        self._pause_requested_event.set()
        self._write_control_status(
            0,
            "pausing",
            "安全检查结果已过期；未发送该动作，正在从真实姿态建立安全保持",
        )
        logger.error(
            "SAFETY EXPIRY ENTER PAUSE: expired target was not sent; "
            "requesting a fresh measured-pose hold: %s",
            error,
        )


    def _check_tracking_error(
        self,
        measured_state: np.ndarray,
        *,
        command: np.ndarray | None = None,
        effector_revisions: tuple[int, int] | None = None,
    ) -> None:
        """Fail closed after repeated large command-to-feedback disagreement."""
        if (
            self.hardware_mode != "execute"
            or not self._execution_armed_event.is_set()
            or self._last_action is None
        ):
            self._tracking_error_consecutive = 0
            self._effector_tracking_consecutive = [0, 0]
            return
        command = np.asarray(
            self._last_action if command is None else command,
            dtype=np.float32,
        ).copy()
        error = np.abs(np.asarray(measured_state, dtype=np.float32) - command)
        head_waist_indices = HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES
        arm_max = float(np.max(error[list(ARM_CANONICAL_INDICES)]))
        effector_error = error[list(EFFECTOR_CANONICAL_INDICES)]
        closure_contact_max = 0.0
        adapter_config = getattr(getattr(self, "adapter", None), "config", None)
        if getattr(adapter_config, "effector_control_mode", None) == "device-position":
            # A Dahuan position command below feedback asks the fingers to close.
            # Persistent positive feedback is expected when an object is
            # grasped, so it is a contact/stall signal rather than a dangerous
            # tracking failure. Failure to OPEN remains fail-closed because it
            # may carry an object into later arm motion.
            effector_indices = list(EFFECTOR_CANONICAL_INDICES)
            effector_command = command[effector_indices]
            effector_feedback = np.asarray(measured_state, dtype=np.float32)[effector_indices]
            opening_shortfall = np.maximum(effector_command - effector_feedback, 0.0)
            closure_contact = np.maximum(effector_feedback - effector_command, 0.0)
            effector_max = float(np.max(opening_shortfall))
            closure_contact_max = float(np.max(closure_contact))
            if closure_contact_max > self.max_effector_tracking_error:
                now = time.monotonic()
                last_warning = getattr(self, "_effector_contact_warning_last", float("-inf"))
                if now - last_warning >= 1.0:
                    logger.warning(
                        "Dahuan gripper closure stopped above target; treating as object contact "
                        "instead of fatal tracking error left_target/feedback=%.6f/%.6f "
                        "right_target/feedback=%.6f/%.6f closure_residual=%.6fm",
                        command[7],
                        measured_state[7],
                        command[15],
                        measured_state[15],
                        closure_contact_max,
                    )
                    self._effector_contact_warning_last = now
        else:
            effector_max = float(np.max(effector_error))
        head_waist_max = float(np.max(error[list(head_waist_indices)]))
        body_exceeded = (
            arm_max > self.max_arm_tracking_error
            or head_waist_max > self.max_head_waist_tracking_error
        )
        exceeded = body_exceeded or effector_max > self.max_effector_tracking_error
        if not exceeded:
            if self._tracking_error_consecutive or any(self._effector_tracking_consecutive):
                logger.info(
                    "Joint tracking error recovered arm/effector/head_waist=%.4f/%.4f/%.4f",
                    arm_max,
                    effector_max,
                    head_waist_max,
                )
            self._tracking_error_consecutive = 0
            self._effector_tracking_consecutive = [0, 0]
            return
        if getattr(adapter_config, "effector_control_mode", None) == "device-position":
            # Camera observations can be more than a second apart. A gripper
            # may fully open, close, and start opening again between them. Do
            # not count errors from different opening commands as one stall.
            self._tracking_error_consecutive = (
                self._tracking_error_consecutive + 1 if body_exceeded else 0
            )
            revisions = (
                tuple(self._effector_command_revision)
                if effector_revisions is None else effector_revisions
            )
            for side, shortfall in enumerate(opening_shortfall):
                if shortfall <= self.max_effector_tracking_error:
                    self._effector_tracking_consecutive[side] = 0
                elif revisions[side] != self._effector_tracking_seen_revision[side]:
                    self._effector_tracking_consecutive[side] = 1
                else:
                    self._effector_tracking_consecutive[side] += 1
                self._effector_tracking_seen_revision[side] = revisions[side]
            consecutive = max(
                self._tracking_error_consecutive, *self._effector_tracking_consecutive,
            )
        else:
            self._tracking_error_consecutive += 1
            consecutive = self._tracking_error_consecutive
        logger.error(
            "Joint tracking error exceeds limits sample=%d/%d "
            "arm=%.4f/%.4f effector=%.4f/%.4f head_waist=%.4f/%.4f "
            "left_gripper_target/feedback=%.6f/%.6f right_gripper_target/feedback=%.6f/%.6f",
            consecutive,
            self.tracking_error_consecutive_limit,
            arm_max,
            self.max_arm_tracking_error,
            effector_max,
            self.max_effector_tracking_error,
            head_waist_max,
            self.max_head_waist_tracking_error,
            command[7], measured_state[7], command[15], measured_state[15],
        )
        if consecutive >= self.tracking_error_consecutive_limit:
            self._fatal_error = (
                "Persistent joint tracking error; refusing further physical execution "
                f"(arm={arm_max:.4f}, effector={effector_max:.4f}, head_waist={head_waist_max:.4f})"
            )
            logger.critical(self._fatal_error)
            self.shutdown_event.set()
            raise RuntimeError(self._fatal_error)

    def _record_effector_command(self, adopted: np.ndarray) -> None:
        """Mark substantial target changes so opening checks remain per motion."""
        target = np.asarray(adopted, dtype=np.float32)[list(EFFECTOR_CANONICAL_INDICES)]
        previous = self._effector_last_command
        if previous is not None:
            for side in range(2):
                if abs(float(target[side] - previous[side])) > self.max_effector_tracking_error:
                    self._effector_command_revision[side] += 1
        self._effector_last_command = target.copy()

    def _latency_p95_s(self) -> float | None:
        if not self._observation_latencies_s:
            return None
        return float(np.percentile(np.asarray(self._observation_latencies_s), 95))

    def _latency_steps(self) -> int:
        latency_p95_s = self._latency_p95_s()
        if latency_p95_s is None:
            return 0
        return max(0, math.ceil(latency_p95_s * self.fps))

    def _estimated_delay_steps(self) -> int:
        execution_horizon = int(
            self.policy_client.metadata.get("rtc", {}).get("execution_horizon", 0)
        )
        delay_steps = self._latency_steps()
        if getattr(self, "rtc_state_aligned_anchor", False) and self._state_latencies_s:
            delay_steps = max(0, math.ceil(float(np.percentile(self._state_latencies_s, 95)) * self.fps))
        return min(execution_horizon, delay_steps)

    def _rtc_time_origin(self, observation: dict[str, Any]) -> float:
        beginning = float(observation["observation_monotonic_time"])
        if not getattr(self, "rtc_state_aligned_anchor", False):
            return beginning
        state_time = float(observation["state_capture_monotonic_time"])
        if not math.isfinite(state_time) or not beginning <= state_time <= time.perf_counter():
            raise RuntimeError("Invalid state-aligned RTC observation timestamp")
        return state_time

    def _request_lead_steps(self) -> int:
        latency_lead = self._latency_steps() + self.request_lead_margin_steps
        if self.inference_mode != "rtc":
            return latency_lead
        execution_horizon = int(
            self.policy_client.metadata.get("rtc", {}).get("execution_horizon", 0)
        )
        if getattr(self, "rtc_state_aligned_anchor", False) and getattr(self, "_state_latencies_s", None):
            # Capture consumes the old prefix BEFORE the paired state anchor.
            # Account for that gap when requesting, without changing either
            # policy/servo frequency or the conservative continuity proof.
            capture_s = max(0.0, (self._latency_p95_s() or 0.0)
                            - float(np.percentile(self._state_latencies_s, 95)))
            execution_horizon += math.ceil(capture_s * self.fps)
            return min(self.quality_horizon, max(latency_lead, execution_horizon))
        return max(latency_lead, execution_horizon)

    def _rtc_request_enabled(self, anchor: Any, estimated_delay_steps: int) -> bool:
        if self.inference_mode != "rtc" or anchor.chunk_id is None:
            return False
        return anchor.remaining_actions - estimated_delay_steps >= self.min_rtc_blend_steps

    def _continuity_budget(self) -> tuple[bool, int, int]:
        latency_steps = self._latency_steps()
        tail_budget = self.request_lead_margin_steps + self.min_replan_after_steps
        if self.inference_mode == "rtc":
            tail_budget = max(tail_budget, self.min_rtc_blend_steps)
        required_steps = 2 * latency_steps + tail_budget
        return required_steps <= self.quality_horizon, required_steps, latency_steps

    def _enforce_execution_continuity_budget(self) -> bool:
        # Sync deliberately waits for the current chunk to finish before
        # requesting another one, so it accepts a hold interval between chunks.
        # The overlap budget is only meaningful for continuously replanned
        # async/RTC execution.
        if self.inference_mode == "sync":
            return True
        safe, required_steps, latency_steps = self._continuity_budget()
        if safe:
            return True
        latency_p95_s = self._latency_p95_s() or 0.0
        message = (
            "Refusing --execute: the measured rolling-P95 observation-to-response latency "
            f"is {latency_p95_s:.3f}s ({latency_steps} steps), so continuous replanning "
            f"requires more than {required_steps} trusted steps, but quality_horizon="
            f"{self.quality_horizon}. Reduce latency or increase the trusted horizon. "
            "No model action from this response was installed."
        )
        if self.hardware_mode == "execute":
            self._fatal_error = message
            logger.critical(message)
            self.shutdown_event.set()
            return False
        logger.warning("Continuity budget would reject --execute: %s", message)
        return True

    def _enforce_actual_rtc_blend_budget(
        self,
        response: dict[str, Any],
        stale_steps: int,
    ) -> bool:
        rtc = response.setdefault("rtc", {})
        effective_horizon = int(rtc.get("effective_execution_horizon", 0))
        actual_blend_steps = (
            max(0, effective_horizon - stale_steps) if rtc.get("prefix_used", False) else 0
        )
        rtc["actual_blend_steps"] = actual_blend_steps
        if not rtc.get("prefix_used", False) or actual_blend_steps >= self.min_rtc_blend_steps:
            return True
        message = (
            "RTC response has insufficient actual blend after wall-clock stale removal: "
            f"effective_horizon={effective_horizon}, stale_steps={stale_steps}, "
            f"actual_blend={actual_blend_steps}, required_blend={self.min_rtc_blend_steps}."
        )
        if self.hardware_mode == "execute":
            self._fatal_error = f"Refusing --execute: {message} No action from this response was installed."
            logger.critical(self._fatal_error)
            self.shutdown_event.set()
            return False
        logger.warning("%s Read-only/dry-run keeps the response for diagnostics.", message)
        return True

    def _log_action_chunk_diagnostics(self, state: np.ndarray, actions: np.ndarray) -> None:
        """Summarize target jumps before 100 Hz interpolation and GDK clipping."""
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) == 0:
            return
        state = np.asarray(state, dtype=np.float32)
        arm_indices = np.asarray(ARM_CANONICAL_INDICES, dtype=np.int64)
        effector_indices = np.asarray(EFFECTOR_CANONICAL_INDICES, dtype=np.int64)
        with_start = np.concatenate((state[None, :], actions), axis=0)
        policy_deltas = np.abs(np.diff(with_start, axis=0))
        arm_deltas = policy_deltas[:, arm_indices]
        effector_deltas = policy_deltas[:, effector_indices]
        arm_flat_index = int(np.argmax(arm_deltas))
        arm_step, arm_local_index = np.unravel_index(arm_flat_index, arm_deltas.shape)
        arm_index = ARM_CANONICAL_INDICES[arm_local_index]
        effector_first = actions[0, effector_indices] - state[effector_indices]
        servo_scale = self.fps / self.servo_fps
        logger.info(
            "Chunk action diagnostics: observed_state_to_first_arm_max=%.4frad joint=%s "
            "arm_step_p95/max=%.4f/%.4frad estimated_100hz_arm_max=%.4frad "
            "first_effectors=%s effector_step_max=%.4f estimated_100hz_effector_max=%.4f",
            float(np.max(np.abs(actions[0, arm_indices] - state[arm_indices]))),
            CANONICAL_STATE_NAMES[arm_index],
            float(np.percentile(arm_deltas, 95)),
            float(arm_deltas[arm_step, arm_local_index]),
            float(np.max(arm_deltas) * servo_scale),
            np.array2string(effector_first, precision=4),
            float(np.max(effector_deltas)),
            float(np.max(effector_deltas) * servo_scale),
        )

    def _record_request_timing(
        self,
        *,
        phase: str,
        request_id: str,
        observation: dict[str, Any],
        response: dict[str, Any],
        round_trip_s: float,
        observation_latency_s: float,
        used_by_latency_estimator: bool,
    ) -> None:
        client_timing = dict(response.get("client_timing", {}))
        server_timing = dict(response.get("server_timing", {}))
        transport_total_ms = float(client_timing.get("transport_total_ms", round_trip_s * 1000))
        server_handler_ms = float(server_timing.get("handler_total_ms", 0.0))
        # The response cannot contain the exact server send/handler duration,
        # because those finish after its bytes have been encoded. Use worker +
        # request decode + response encode as the observable server subtotal.
        server_observable_ms = (
            float(server_timing.get("request_deserialize_ms", 0.0))
            + float(server_timing.get("queue_ms", 0.0))
            + float(server_timing.get("worker_ms", server_timing.get("total_ms", 0.0)))
            + float(server_timing.get("response_serialize_ms", 0.0))
        )
        known_client_ms = (
            float(client_timing.get("request_serialize_ms", 0.0))
            + float(client_timing.get("websocket_send_ms", 0.0))
            + float(client_timing.get("response_deserialize_ms", 0.0))
        )
        record = {
            "phase": phase,
            "request_id": request_id,
            "used_by_latency_estimator": used_by_latency_estimator,
            "round_trip_ms": round_trip_s * 1000,
            "observation_to_response_ms": observation_latency_s * 1000,
            "camera_capture_ms": float(observation.get("camera_capture_duration_ms", 0.0)),
            "camera_lock_wait_ms": float(observation.get("camera_lock_wait_ms", 0.0)),
            "camera_pipeline_timing": observation.get("camera_pipeline_timing", {}),
            "image_transport_timing": observation.get("image_transport_timing", {}),
            "client_timing": client_timing,
            "server_timing": server_timing,
            "server_observable_ms": server_observable_ms,
            "unattributed_transport_and_scheduling_ms": max(
                0.0,
                transport_total_ms - known_client_ms - server_observable_ms,
            ),
        }
        if server_handler_ms:
            record["server_handler_ms"] = server_handler_ms
        if self._timing_log_file is not None:
            with self._timing_log_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        logger.info(
            "End-to-end timing phase=%s id=%s obs=%.1fms rtt=%.1fms "
            "camera=%.1fms image_transform=%.1fms serialize/send/wait/decode="
            "%.1f/%.1f/%.1f/%.1fms server_decode/queue/worker/pre/infer/post/encode="
            "%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1fms residual=%.1fms bytes=%d/%d estimator=%s",
            phase,
            request_id,
            observation_latency_s * 1000,
            round_trip_s * 1000,
            float(observation.get("camera_capture_duration_ms", 0.0)),
            float(observation.get("camera_pipeline_timing", {}).get("processing_wall_ms",
                sum(float(value.get("total_ms", 0.0))
                    for value in observation.get("image_transport_timing", {}).values()))),
            float(client_timing.get("request_serialize_ms", 0.0)),
            float(client_timing.get("websocket_send_ms", 0.0)),
            float(client_timing.get("response_wait_ms", 0.0)),
            float(client_timing.get("response_deserialize_ms", 0.0)),
            float(server_timing.get("request_deserialize_ms", 0.0)),
            float(server_timing.get("queue_ms", 0.0)),
            float(server_timing.get("worker_ms", 0.0)),
            float(server_timing.get("preprocess_ms", 0.0)),
            float(server_timing.get("infer_ms", 0.0)),
            float(server_timing.get("postprocess_ms", 0.0)),
            float(server_timing.get("response_serialize_ms", 0.0)),
            record["unattributed_transport_and_scheduling_ms"],
            int(client_timing.get("request_bytes", 0)),
            int(client_timing.get("response_bytes", 0)),
            used_by_latency_estimator,
        )

    def _perform_warmup(self) -> None:
        if self.warmup_requests == 0:
            logger.warning("Policy warmup is disabled; the first measured request may include cold start")
            return
        logger.info(
            "Starting %d policy warmup request(s); they will not enter the rolling latency estimator",
            self.warmup_requests,
        )
        previous_response: dict[str, Any] | None = None
        for warmup_index in range(self.warmup_requests):
            observation = self._capture_observation()
            request_id = self._next_request_id()
            request = {
                "kind": "infer",
                "request_id": request_id,
                "session_id": self.session_id,
                "observation": observation,
                "control": {
                    "mode": "async",
                    "enable_rtc": False,
                    "active_chunk_id": None,
                    "next_action_index": 0,
                    "estimated_delay_steps": 0,
                    "min_rtc_blend_steps": self.min_rtc_blend_steps,
                },
                "reset": warmup_index == 0,
            }
            started = time.perf_counter()
            response = self.policy_client.infer(request)
            previous_response = response
            response_time = time.perf_counter()
            round_trip_s = response_time - started
            observation_latency_s = max(
                0.0,
                response_time - float(observation["observation_monotonic_time"]),
            )
            self._record_request_timing(
                phase="warmup",
                request_id=request_id,
                observation=observation,
                response=response,
                round_trip_s=round_trip_s,
                observation_latency_s=observation_latency_s,
                used_by_latency_estimator=False,
            )
        if self.inference_mode == "rtc":
            assert previous_response is not None
            observation = self._capture_observation()
            request_id = self._next_request_id()
            request = {
                "kind": "infer",
                "request_id": request_id,
                "session_id": self.session_id,
                "observation": observation,
                "control": {
                    "mode": "rtc",
                    "enable_rtc": True,
                    "active_chunk_id": previous_response.get("chunk_id"),
                    "next_action_index": 0,
                    "estimated_delay_steps": 0,
                    "min_rtc_blend_steps": self.min_rtc_blend_steps,
                },
                "reset": False,
            }
            started = time.perf_counter()
            response = self.policy_client.infer(request)
            response_time = time.perf_counter()
            rtc = response.get("rtc", {})
            if not rtc.get("prefix_used", False):
                raise RuntimeError(
                    "RTC client warmup did not exercise prefix guidance: "
                    f"fallback_reason={rtc.get('fallback_reason')}"
                )
            self._record_request_timing(
                phase="warmup_rtc",
                request_id=request_id,
                observation=observation,
                response=response,
                round_trip_s=response_time - started,
                observation_latency_s=max(
                    0.0,
                    response_time - float(observation["observation_monotonic_time"]),
                ),
                used_by_latency_estimator=False,
            )
        logger.info("Policy warmup completed successfully")

    def _perform_execution_latency_preflight(self) -> None:
        """Measure requests at the cadence expected during physical execution."""
        if self.hardware_mode != "execute" or self.inference_mode == "sync":
            return
        if self.execution_preflight_requests <= 0:
            raise RuntimeError(
                "Continuous physical execution requires at least one latency preflight request"
            )
        # A tight burst keeps the GPU at its burst clock and understates the
        # latency seen after an RTC chunk has advanced to the request lead.
        # No action is installed or sent while these requests are measured.
        lead_steps = min(self.quality_horizon, self._request_lead_steps())
        interval_s = max(1.0 / self.fps, (self.quality_horizon - lead_steps) / self.fps)
        logger.info(
            "Starting %d execution latency preflight request(s) at %.3fs cadence; "
            "responses will not be installed",
            self.execution_preflight_requests,
            interval_s,
        )
        previous_response: dict[str, Any] | None = None
        next_capture_at: float | None = None
        for preflight_index in range(self.execution_preflight_requests):
            if next_capture_at is not None:
                remaining_s = next_capture_at - time.perf_counter()
                if remaining_s > 0 and self.shutdown_event.wait(remaining_s):
                    raise RuntimeError("Execution latency preflight was interrupted")
            observation = self._capture_observation()
            request_id = self._next_request_id()
            enable_rtc = self.inference_mode == "rtc" and previous_response is not None
            estimated_delay_steps = self._estimated_delay_steps() if enable_rtc else 0
            request = {
                "kind": "infer",
                "request_id": request_id,
                "session_id": self.session_id,
                "observation": observation,
                "control": {
                    "mode": self.inference_mode,
                    "enable_rtc": enable_rtc,
                    "active_chunk_id": previous_response.get("chunk_id") if enable_rtc else None,
                    "next_action_index": 0,
                    "estimated_delay_steps": estimated_delay_steps,
                    "min_rtc_blend_steps": self.min_rtc_blend_steps,
                },
                "reset": preflight_index == 0,
            }
            started = time.perf_counter()
            next_capture_at = float(observation["observation_monotonic_time"]) + interval_s
            response = self.policy_client.infer(request)
            previous_response = response
            response_time = time.perf_counter()
            if enable_rtc and not response.get("rtc", {}).get("prefix_used", False):
                raise RuntimeError(
                    "RTC execution preflight did not exercise prefix guidance: "
                    f"fallback_reason={response.get('rtc', {}).get('fallback_reason')}"
                )
            round_trip_s = response_time - started
            observation_latency_s = max(
                0.0,
                response_time - float(observation["observation_monotonic_time"]),
            )
            self._record_request_timing(
                phase="execution_preflight",
                request_id=request_id,
                observation=observation,
                response=response,
                round_trip_s=round_trip_s,
                observation_latency_s=observation_latency_s,
                used_by_latency_estimator=True,
            )
            # The first live response also pays for action-coordinate mapping,
            # smoothing and diagnostics. Include those costs in the same
            # continuity proof, without installing or sending any action.
            response_actions = self.adapter.model_actions_to_control_targets(
                np.asarray(response.get("actions"), dtype=np.float32)
            )
            response_actions = self._chunk_shaper.shape(response_actions)
            self._log_action_chunk_diagnostics(
                np.asarray(observation["state"], dtype=np.float32), response_actions,
            )
            processed_at = time.perf_counter()
            self._observation_latencies_s.append(max(
                0.0, processed_at - float(observation["observation_monotonic_time"])
            ))
            if self.rtc_state_aligned_anchor:
                self._state_latencies_s.append(max(
                    0.0, processed_at - self._rtc_time_origin(observation)
                ))
        if not self._enforce_execution_continuity_budget():
            raise RuntimeError(self._fatal_error or "Execution latency preflight failed")
        logger.info(
            "Execution latency preflight passed samples=%d p95=%.1fms latency_steps=%d",
            len(self._observation_latencies_s),
            (self._latency_p95_s() or 0.0) * 1000,
            self._latency_steps(),
        )

    def _communication_loop(self) -> None:
        reset_next_request = True
        while not self.shutdown_event.is_set():
            if self._pause_requested_event.is_set() or self._pause_active_event.is_set():
                reset_next_request = True
                self.shutdown_event.wait(0.02)
                continue
            if self._grasp_hold_event.is_set():
                reset_next_request = True
                self.shutdown_event.wait(0.01)
                continue
            if not self.plan.ready_for_request(
                self.inference_mode,
                self.min_replan_after_steps,
                self._request_lead_steps(),
            ):
                self.shutdown_event.wait(0.005)
                continue
            try:
                # Preserve the legacy camera-start anchor when not opted in.
                # State-aligned RTC replaces it with a state/plan pair below;
                # its prefix, delay estimator and drop clock then share that
                # same origin while continuity still includes camera age.
                with self._session_lock:
                    request_session = self.session_id
                    anchor = self.plan.snapshot()
                    capture = self.flowpro_capture
                    request_generation = (
                        None if capture is None else capture.active_generation_id
                    )
                if capture is not None and request_generation is None:
                    reset_next_request = True
                    self.shutdown_event.wait(0.005)
                    continue
                if self._freeze_trace_observation and self._frozen_trace_observation is not None:
                    observation = self._frozen_trace_observation
                else:
                    if self.rtc_state_aligned_anchor:
                        observation = self._capture_observation(capture_plan_anchor=True)
                        anchor = observation.pop("_plan_anchor")
                    else:
                        observation = self._capture_observation()
                    self._submit_presence_frame(observation["images"], observation["camera_frame_wall_time_ns"])
                    if self._freeze_trace_observation:
                        self._frozen_trace_observation = observation
                        logger.info(
                            "Frozen the first observation; all later trace requests reuse identical "
                            "state/image bytes"
                        )
                command_prefix = observation.pop("_rtc_command_prefix", None)
                request_id = self._next_request_id()
                if not self._inference_session_is_current(request_session):
                    reset_next_request = True
                    continue
                if not self._flowpro_generation_current(request_generation):
                    reset_next_request = True
                    continue
                delay_steps = self._estimated_delay_steps()
                enable_rtc = self._rtc_request_enabled(anchor, delay_steps)
                if self.inference_mode == "rtc" and anchor.chunk_id is not None and not enable_rtc:
                    logger.warning(
                        "RTC request falls back to async: remaining=%d estimated_delay=%d "
                        "available_blend=%d required_blend=%d",
                        anchor.remaining_actions,
                        delay_steps,
                        max(0, anchor.remaining_actions - delay_steps),
                        self.min_rtc_blend_steps,
                    )
                request = {
                    "kind": "infer",
                    "request_id": request_id,
                    "session_id": request_session,
                    "observation": observation,
                    "control": {
                        "mode": self.inference_mode,
                        "enable_rtc": enable_rtc,
                        "active_chunk_id": anchor.chunk_id,
                        "next_action_index": anchor.next_action_index,
                        "estimated_delay_steps": delay_steps,
                        "min_rtc_blend_steps": self.min_rtc_blend_steps,
                    },
                    "reset": reset_next_request,
                }
                if getattr(self, "rtc_prefix_mode", "model") == "command" and enable_rtc:
                    if command_prefix is None:
                        raise RuntimeError("Complete command-prefix preview unavailable; refusing ambiguous conditioning")
                    if command_prefix["actions"].shape[0] != delay_steps:
                        raise RuntimeError("Command-prefix delay changed after state capture")
                    request["control"]["command_prefix"] = {
                        key: value for key, value in command_prefix.items()
                        if key not in {"hardware_actions", "offsets_s"}
                    }
                started = time.perf_counter()
                response = self.policy_client.infer(request)
                response_monotonic_time = time.perf_counter()
                if getattr(self, "rtc_prefix_mode", "model") == "command" and command_prefix is not None:
                    if (self.plan.snapshot().generation != command_prefix["plan_generation"]
                            or getattr(self.adapter, "_async_generation", 0) != command_prefix["safety_generation"]):
                        logger.warning("Discarding response after command-prefix generation changed")
                        reset_next_request = True
                        continue
                if not self._inference_session_is_current(request_session):
                    logger.info("Discarding inference response from an inactive/previous session")
                    reset_next_request = True
                    continue
                if not self._flowpro_generation_current(request_generation):
                    reset_next_request = True
                    continue
                captured_inference = None
                if capture is not None:
                    captured_inference = capture.capture_inference(
                        request=request,
                        response=response,
                        generation_id=request_generation,
                        control_owner="policy",
                    )
                    if captured_inference is None or not self._flowpro_generation_current(request_generation):
                        reset_next_request = True
                        continue
                round_trip_s = response_monotonic_time - started
                observation_latency_s = max(
                    0.0,
                    response_monotonic_time - float(observation["observation_monotonic_time"]),
                )
                self._record_request_timing(
                    phase="inference",
                    request_id=request_id,
                    observation=observation,
                    response=response,
                    round_trip_s=round_trip_s,
                    observation_latency_s=observation_latency_s,
                    used_by_latency_estimator=True,
                )
                if self._inference_frame_writer is not None:
                    self._inference_frame_writer.submit(
                        request_id,
                        observation["images"],
                        captured_at=float(observation["camera_capture_wall_time_ns"]) / 1e9,
                    )
                mapping_started = time.perf_counter()
                policy_actions = np.asarray(response.get("actions"), dtype=np.float32)
                response_actions = self.adapter.model_actions_to_control_targets(policy_actions)
                mapping_finished = time.perf_counter()
                response_actions = self._chunk_shaper.shape(response_actions)
                shaping_finished = time.perf_counter()
                self._log_action_chunk_diagnostics(
                    np.asarray(observation["state"], dtype=np.float32),
                    response_actions,
                )
                diagnostics_finished = time.perf_counter()
                # Include mapping, chunk shaping and diagnostics in staleness.
                # Response-arrival latency alone undercounted the adaptive
                # smoother by as much as several 30 Hz policy steps.
                effective_observation_latency_s = max(
                    0.0,
                    diagnostics_finished - float(observation["observation_monotonic_time"]),
                )
                # Camera age remains in the conservative continuity budget.
                # Only prefix indexing/drop use the paired state's origin.
                effective_state_latency_s = max(0.0, diagnostics_finished - self._rtc_time_origin(observation))
                stale_steps = max(0, math.ceil(effective_state_latency_s * self.fps))
                current_snapshot = self.plan.snapshot()
                progress_dropped = (
                    max(0, current_snapshot.next_action_index - anchor.next_action_index)
                    if anchor.chunk_id is not None and current_snapshot.chunk_id == anchor.chunk_id
                    else 0
                )
                effective_dropped_steps = max(stale_steps, progress_dropped)
                with self._session_lock:
                    if not self._inference_session_is_current(request_session):
                        reset_next_request = True
                        continue
                    if not self._flowpro_generation_current(request_generation):
                        reset_next_request = True
                        continue
                    self._last_round_trip_s = round_trip_s
                    self._observation_latencies_s.append(effective_observation_latency_s)
                    if self.rtc_state_aligned_anchor:
                        self._state_latencies_s.append(effective_state_latency_s)
                    if not self._enforce_actual_rtc_blend_budget(response, effective_dropped_steps):
                        break
                    if not self._enforce_execution_continuity_budget():
                        break
                    # Read-only sends no physical command. Park its virtual
                    # action consumer while replacing a chunk so a stream of
                    # zero-cost reservations cannot starve plan.install().
                    # Physical execution keeps the original reservation/GDK
                    # hand-off unchanged.
                    read_only_install_barrier = self.hardware_mode != "execute"
                    if read_only_install_barrier:
                        self._execution_armed_event.clear()
                    flowpro_lineage = None
                    if captured_inference is not None:
                        flowpro_lineage = {
                            "observation_id": captured_inference.observation_id,
                            "candidate_id": captured_inference.candidate_id,
                            "generation_id": captured_inference.generation_id,
                            "native_actions": np.asarray(
                                response["native_actions"], dtype=np.float32
                            ).copy(),
                            "request_started_monotonic_ns": int(started * 1e9),
                            "response_received_monotonic_ns": int(response_monotonic_time * 1e9),
                            "latency_stale_steps": stale_steps,
                            "first_executed_timestamp_ns": None,
                            "first_executed_action_index": None,
                        }

                    def publish_flowpro_lineage(install_result):
                        assert flowpro_lineage is not None
                        chunk_id = str(response.get("chunk_id", ""))
                        self._flowpro_chunk_lineage[chunk_id] = dict(
                            flowpro_lineage,
                            chunk_installed_monotonic_ns=time.perf_counter_ns(),
                            installed_first_action_index=install_result.dropped_steps,
                        )
                        while len(self._flowpro_chunk_lineage) > 16:
                            self._flowpro_chunk_lineage.pop(next(iter(self._flowpro_chunk_lineage)))

                    try:
                        result = self.plan.install(
                            str(response.get("chunk_id", "")),
                            response_actions,
                            anchor,
                            stale_steps=stale_steps,
                            quality_horizon=self.quality_horizon,
                            stale_origin_monotonic=(
                                self._rtc_time_origin(observation)
                                if getattr(self, "rtc_state_aligned_anchor", False) else None
                            ),
                            rtc_handoff_horizon=(
                                int(response["rtc"]["effective_execution_horizon"])
                                if getattr(self, "rtc_state_aligned_anchor", False)
                                and response.get("rtc", {}).get("prefix_used", False) else None
                            ),
                            rtc_handoff_indices=tuple(ARM_CANONICAL_INDICES),
                            rtc_guard_old_tail=(
                                self.inference_mode == "rtc" and self.rtc_state_aligned_anchor
                            ),
                            max_dropped_steps=(
                                int(response["rtc"]["effective_execution_horizon"]) - self.min_rtc_blend_steps
                                if self.hardware_mode == "execute" and response.get("rtc", {}).get("prefix_used", False)
                                else None
                            ),
                            before_publish=(
                                publish_flowpro_lineage if flowpro_lineage is not None else None
                            ),
                        )
                    finally:
                        if (
                            read_only_install_barrier
                            and not self._pause_requested_event.is_set()
                            and not self._pause_active_event.is_set()
            and not self._grasp_hold_event.is_set()
                        ):
                            self._execution_armed_event.set()
                    # install may wait for an in-flight servo commit. A pause
                    # request can arrive during that wait, before its action
                    # thread can acquire this session lock.
                    if (
                        not self._inference_session_is_current(request_session)
                        or not self._flowpro_generation_current(request_generation)
                    ):
                        if result.installed:
                            self.plan.clear()
                            self._flowpro_chunk_lineage.pop(str(response.get("chunk_id", "")), None)
                        reset_next_request = True
                        continue
                    if result.installed:
                        self._last_response_monotonic = time.monotonic()
                        reset_next_request = False
                    if result.installed:
                        if not self._execution_armed_event.is_set():
                            self._execution_armed_event.set()
                            logger.info("Execution armed with a fresh session inference response")
                        # The read-only install barrier may already have armed
                        # its virtual action consumer in the finally block.
                        # A successfully installed fresh-session plan must
                        # independently acknowledge pause/resume readiness.
                        self._resume_ready_event.set()
                chunk_id = str(response.get("chunk_id", ""))
                if response.get("rtc", {}).get("prefix_used", False):
                    response["rtc"]["actual_blend_steps"] = max(
                        0, int(response["rtc"]["effective_execution_horizon"]) - result.dropped_steps
                    )
                if result.rtc_handoff is not None:
                    logger.info("RTC handoff chunk=%s details=%s", chunk_id, result.rtc_handoff)
                self._record_inference_response_diagnostics(
                    request_id=request_id, request_session=request_session, response=response,
                    policy_actions=policy_actions, response_actions=response_actions,
                    observation=observation, anchor=anchor, result=result,
                    response_received_at=response_monotonic_time, stale_steps=stale_steps,
                    response_processing_ms=(diagnostics_finished-response_monotonic_time)*1000,
                    mapping_ms=(mapping_finished-mapping_started)*1000,
                    shaping_ms=(shaping_finished-mapping_finished)*1000,
                    chunk_diagnostics_ms=(diagnostics_finished-shaping_finished)*1000,
                    effective_observation_latency_ms=effective_observation_latency_s*1000,
                )
                if result.installed:
                    self._chunk_safety_context[chunk_id] = {
                        "request_id": request_id,
                        "chunk_id": chunk_id,
                        "chunk_output_length": len(response_actions),
                        "chunk_execute_length": max(
                            0,
                            min(len(response_actions), self.quality_horizon) - result.dropped_steps,
                        ),
                        "policy_latency_ms": round_trip_s * 1000,
                    }
                    # Bound diagnostic state during long-running sessions.
                    while len(self._chunk_safety_context) > 16:
                        self._chunk_safety_context.pop(next(iter(self._chunk_safety_context)))
                    if result.fixed_horizon_projection is not None:
                        projection = result.fixed_horizon_projection
                        logger.info(
                            "Fixed-horizon projection chunk=%s steps=%d changed=%d "
                            "max/rms_deviation=%.4f/%.4frad",
                            chunk_id,
                            int(projection["steps"]),
                            int(projection["changed_values"]),
                            float(projection["max_deviation_rad"]),
                            float(projection["rms_deviation_rad"]),
                        )
                if self.trace_recorder is not None:
                    request_dir = self.trace_recorder.record_request(
                        request_id=request_id,
                        observation=observation,
                        response=response,
                        anchor=anchor,
                        estimated_delay_steps=delay_steps,
                        round_trip_s=round_trip_s,
                        observation_latency_s=observation_latency_s,
                        latency_p95_s=self._latency_p95_s() or observation_latency_s,
                        stale_steps=stale_steps,
                        install_result=result,
                    )
                    logger.info("Saved raw model response to %s", request_dir)
                if not result.installed:
                    logger.warning("Discarded response %s: %s", request_id, result.reason)
                    # Discard only this response. Any still-valid portion of the
                    # active plan remains available while the next request runs.
                    reset_next_request = self.plan.snapshot().chunk_id is None
                    continue
                logger.info(
                    "Installed chunk=%s stale=%d dropped=%d queue=%d lead=%d "
                    "latency_p95=%.1fms rtt=%.1fms camera=%.1fms infer=%.1fms "
                    "rtc_prefix=%s rtc_source/effective/estimated_blend/actual_blend="
                    "%d/%d/%d/%d rtc_fallback=%s",
                    response.get("chunk_id"),
                    stale_steps,
                    result.dropped_steps,
                    self.plan.snapshot().remaining_actions,
                    self._request_lead_steps(),
                    (self._latency_p95_s() or 0.0) * 1000,
                    round_trip_s * 1000,
                    float(observation["camera_capture_duration_ms"]),
                    float(response.get("server_timing", {}).get("infer_ms", 0.0)),
                    response.get("rtc", {}).get("prefix_used", False),
                    int(response.get("rtc", {}).get("prefix_source_steps", 0)),
                    int(response.get("rtc", {}).get("effective_execution_horizon", 0)),
                    int(response.get("rtc", {}).get("blend_steps", 0)),
                    int(response.get("rtc", {}).get("actual_blend_steps", 0)),
                    response.get("rtc", {}).get("fallback_reason"),
                )
            except Exception as error:
                logger.error("Observation/inference failed: %s", error, exc_info=True)
                reset_next_request = True
                if self.shutdown_event.wait(1):
                    break
                try:
                    old_profile = (
                        self.policy_client.metadata.get("model_profile")
                        if self.policy_client.connected
                        else None
                    )
                    self.policy_client.reconnect(wait=False)
                    self.validate_server()
                    new_profile = self.policy_client.metadata.get("model_profile")
                    if old_profile is not None and old_profile != new_profile:
                        self._fatal_error = f"Server model changed during reconnect: {old_profile} -> {new_profile}"
                        logger.critical(self._fatal_error)
                        self.shutdown_event.set()
                        break
                    with self._session_lock:
                        self.plan.clear()
                        self.session_id = uuid.uuid4().hex
                except Exception as reconnect_error:
                    logger.warning("Reconnect failed: %s", reconnect_error)

    def _record_flowpro_policy_dispatch(self) -> None:
        """Append only an actually accepted, provenance-bound policy command."""
        capture = self.flowpro_capture
        if capture is None or self.hardware_mode != "execute":
            return
        receipt = self.adapter.dispatch_receipt()
        if receipt is None:
            return
        if receipt.get("physical_accepted") is not True:
            raise RuntimeError("FlowPRO policy dispatch lacks a GDK acceptance")
        measured = np.asarray(receipt.get("measured_state"), dtype=np.float32)
        age_ms = receipt.get("measured_age_ms")
        measured_ns = receipt.get("measured_timestamp_ns")
        if (
            measured.shape != (ACTION_DIM,)
            or not np.isfinite(measured).all()
            or age_ms is None
            or not math.isfinite(float(age_ms))
            or not 0.0 <= float(age_ms) <= 100.0
            or isinstance(measured_ns, bool)
            or not isinstance(measured_ns, int)
            or measured_ns <= 0
        ):
            raise RuntimeError("FlowPRO policy dispatch lacks fresh measured feedback")
        provenance = receipt.get("provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError("FlowPRO policy dispatch lacks queue provenance")
        if provenance.get("holding_previous_target") is True:
            # A checked hold is real execution evidence, but strict training
            # must see it as an underrun instead of a policy action.
            template = capture.execution_history.latest_sample
            if template is not None and template.generation_id == capture.active_generation_id:
                from .hitl.g2_runtime.executed import EXECUTION_KIND_QUEUE_UNDERRUN

                timestamp_ns = max(
                    measured_ns,
                    0 if self._last_flowpro_sample_ns is None else self._last_flowpro_sample_ns + 1,
                )
                held = replace(
                    template,
                    timestamp_ns=timestamp_ns,
                    measured_timestamp_ns=measured_ns,
                    action_id=str(uuid.uuid4()),
                    safety_target_24d=receipt["action"],
                    gdk_command_24d=receipt["action"],
                    measured_state_24d=measured,
                    execution_kind=EXECUTION_KIND_QUEUE_UNDERRUN,
                    command_accepted=True,
                    safety_modified=False,
                )
                capture.execution_history.append_execution(held)
                self._last_flowpro_sample_ns = timestamp_ns
            return
        chunk_id = provenance.get("chunk_id")
        action_index = provenance.get("policy_action_index")
        lineage = self._flowpro_chunk_lineage.get(chunk_id)
        if lineage is None or isinstance(action_index, bool) or not isinstance(action_index, int):
            raise RuntimeError("FlowPRO policy dispatch cannot be joined to its inference")
        if lineage["generation_id"] != capture.active_generation_id:
            raise RuntimeError("FlowPRO policy dispatch belongs to an inactive generation")
        native_actions = lineage["native_actions"]
        if not 0 <= action_index < len(native_actions):
            raise RuntimeError("FlowPRO policy dispatch index is outside its native candidate")
        target = provenance.get("safety_target_24d")
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
            raise RuntimeError("FlowPRO policy dispatch lacks its checked target")
        command = np.asarray(receipt["action"], dtype=np.float32)
        timestamp_ns = max(
            measured_ns,
            0 if self._last_flowpro_sample_ns is None else self._last_flowpro_sample_ns + 1,
        )
        if lineage["first_executed_timestamp_ns"] is None:
            lineage["first_executed_timestamp_ns"] = timestamp_ns
            lineage["first_executed_action_index"] = action_index
        from .hitl.g2_runtime.executed import PolicyExecutionSample

        sample = PolicyExecutionSample(
            timestamp_ns=timestamp_ns,
            measured_timestamp_ns=measured_ns,
            observation_id=lineage["observation_id"],
            candidate_id=lineage["candidate_id"],
            generation_id=lineage["generation_id"],
            action_id=str(uuid.uuid4()),
            chunk_id=chunk_id,
            action_index=action_index,
            policy_native_action=native_actions[action_index],
            safety_target_24d=target,
            gdk_command_24d=command,
            measured_state_24d=measured,
            command_accepted=True,
            safety_modified=not np.allclose(command, target, rtol=0.0, atol=1e-5),
            request_started_monotonic_ns=lineage["request_started_monotonic_ns"],
            response_received_monotonic_ns=lineage["response_received_monotonic_ns"],
            chunk_installed_monotonic_ns=lineage["chunk_installed_monotonic_ns"],
            latency_stale_steps=lineage["latency_stale_steps"],
            installed_first_action_index=lineage["installed_first_action_index"],
            first_executed_timestamp_ns=lineage["first_executed_timestamp_ns"],
            first_executed_action_index=lineage["first_executed_action_index"],
        )
        capture.execution_history.append_execution(sample)
        self._last_flowpro_sample_ns = timestamp_ns
        monitor = self._flowpro_arbitration_monitor
        if monitor is not None:
            monitor.observe_policy_lease(sample.generation_id, timestamp_ns=timestamp_ns)
            monitor.observe_policy_ack(sample.generation_id, timestamp_ns=timestamp_ns)

    def _record_inference_response_diagnostics(
        self, *, request_id, request_session, response, policy_actions,
        response_actions, observation, anchor, result, response_received_at, stale_steps,
        response_processing_ms=0.0, mapping_ms=0.0, shaping_ms=0.0,
        chunk_diagnostics_ms=0.0, effective_observation_latency_ms=0.0,
    ) -> None:
        """Record replayable numerical chunks, not images, on the bounded writer.

        This is diagnostic only: no inference/actuation or file I/O here. Both
        model and physical coordinates are retained to avoid confusing the DH
        inverse gripper convention with a timing problem.
        """
        diagnostics = getattr(self, "control_diagnostics", None)
        if diagnostics is None:
            return
        diagnostics.write(dict(
            record_type="inference_response", request_id=request_id, session_id=request_session,
            chunk_id=response.get("chunk_id"), response_received_monotonic=response_received_at,
            observation_monotonic=observation.get("observation_monotonic_time"),
            observation_state_monotonic=observation.get("state_capture_monotonic_time"),
            rtc_state_aligned_anchor=getattr(self, "rtc_state_aligned_anchor", False),
            observation_state=np.asarray(observation["state"], dtype=np.float32),
            observation_state_lock_wait_ms=observation.get("state_lock_wait_ms"),
            observation_state_read_ms=observation.get("state_read_ms"),
            model_actions=np.asarray(policy_actions, dtype=np.float32),
            control_targets=np.asarray(response_actions, dtype=np.float32),
            anchor_chunk_id=anchor.chunk_id, anchor_next_action_index=anchor.next_action_index,
            anchor_generation=anchor.generation, requested_stale_steps=stale_steps,
            response_processing_ms=response_processing_ms,
            mapping_ms=mapping_ms,
            shaping_ms=shaping_ms,
            chunk_diagnostics_ms=chunk_diagnostics_ms,
            effective_observation_latency_ms=effective_observation_latency_ms,
            installed=result.installed, dropped_steps=result.dropped_steps, install_reason=result.reason,
            installed_monotonic=result.installed_at, install_wait_ms=result.wait_ms,
            rtc=dict(response.get("rtc", {})),
            fixed_horizon_projection=getattr(result, "fixed_horizon_projection", None),
            rtc_handoff=getattr(result, "rtc_handoff", None),
        ))

    def _inference_session_is_current(self, session_id: str) -> bool:
        return (
            session_id == self.session_id
            and not self.shutdown_event.is_set()
            and not self._pause_requested_event.is_set()
            and not self._pause_active_event.is_set()
        )

    def _flowpro_generation_current(self, generation_id: str | None) -> bool:
        capture = self.flowpro_capture
        return capture is None or (
            generation_id is not None
            and capture.active_generation_id == generation_id
            and not self._pause_requested_event.is_set()
            and not self._pause_active_event.is_set()
        )

    def _send_action(
        self,
        action: np.ndarray,
        *,
        chunk_id: str | None = None,
        action_index: int = -1,
        hold: bool = False,
        trajectory_preview: np.ndarray | Callable[[], np.ndarray] | None = None,
    ) -> np.ndarray:
        if chunk_id is None:
            chunk_id = getattr(self, "_active_safety_chunk_id", None)
        if action_index < 0:
            action_index = getattr(self, "_active_safety_action_index", -1)
        conditioned = np.asarray(action, dtype=np.float32).copy()
        if not hold and self._held_head_waist_values is not None:
            head_waist_indices = HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES
            conditioned[list(head_waist_indices)] = self._held_head_waist_values
        if not hold and self._held_effector_values is not None:
            conditioned[list(EFFECTOR_CANONICAL_INDICES)] = self._held_effector_values
        grasp_coordinator = getattr(self, "_grasp_coordinator", None)
        if not hold and grasp_coordinator is not None:
            feedback = getattr(self.adapter, "_last_feedback_state", None)
            conditioned, grasp_events = grasp_coordinator.update(
                conditioned, feedback, now=time.monotonic(),
                feedback_time=getattr(self.adapter, "_last_feedback_monotonic", None),
            )
            for event in grasp_events:
                level = logging.WARNING if event["event"] == "timeout" else logging.INFO
                logger.log(
                    level, "Grasp coordination side=%s event=%s target=%s",
                    "left" if event["side"] == 0 else "right",
                    event["event"], "closed" if event.get("closed") else "open",
                )
                if event["event"] == "confirmed":
                    getattr(self, "_grasp_replan_event").set()
            if grasp_coordinator.blocked:
                getattr(self, "_grasp_hold_event").set()
            else:
                getattr(self, "_grasp_hold_event").clear()
        effector_target = conditioned[list(EFFECTOR_CANONICAL_INDICES)].copy()
        live_effector_latch = getattr(self, "_effector_endpoint_latch", None)
        safety_context: dict[str, Any] = {}
        preview = None
        if trajectory_preview is not None and not hold:
            # Preview on private filter states: never advance the live filters
            # by the lookahead horizon or reset them at segment boundaries.
            arm_filter = None if self._action_filter is None else self._action_filter.fork()
            eff_filter = (None if self._effector_action_filter is None
                          else self._effector_action_filter.fork())
            effector_latch = (None if live_effector_latch is None
                              else live_effector_latch.fork())
            cached_preview = None

            def build_preview():
                nonlocal cached_preview
                if cached_preview is None:
                    raw = trajectory_preview() if callable(trajectory_preview) else trajectory_preview
                    if isinstance(raw, tuple):
                        raw, preview_indices = raw
                        safety_context["preview_action_indices"] = tuple(preview_indices)
                    value = np.asarray(raw, dtype=np.float32).copy()
                    for i, point in enumerate(value):
                        if self._held_head_waist_values is not None:
                            point[list(HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES)] = self._held_head_waist_values
                        if self._held_effector_values is not None:
                            point[list(EFFECTOR_CANONICAL_INDICES)] = self._held_effector_values
                        eff = point[list(EFFECTOR_CANONICAL_INDICES)].copy()
                        value[i] = point if arm_filter is None else arm_filter.update(point)
                        shaped_effector = eff if eff_filter is None else eff_filter.update(eff)
                        if effector_latch is not None:
                            shaped_effector = effector_latch.update(shaped_effector)
                        value[i, list(EFFECTOR_CANONICAL_INDICES)] = shaped_effector
                    safety_context["preview_control_targets"] = value.copy()
                    cached_preview = value
                return cached_preview

            # The async adapter consumes this factory synchronously only when
            # it can submit a new segment. It must never put it in worker IPC.
            preview = build_preview if callable(trajectory_preview) else build_preview()
        if hold:
            # A plan gap is a hold, not further pursuit of a filtered old goal.
            if self._action_filter is not None:
                self._action_filter.reset(conditioned)
            if self._effector_action_filter is not None:
                self._effector_action_filter.reset(effector_target)
            if live_effector_latch is not None:
                live_effector_latch.reset(effector_target)
            if grasp_coordinator is not None:
                grasp_coordinator.reset(conditioned)
        if self._action_filter is not None:
            conditioned = self._action_filter.update(conditioned)
        if self._effector_action_filter is None:
            # Arm/head/waist smoothing must not add a second, hidden delay to
            # gripper commands. Software-trajectory mode still limits their
            # motion; device-position mode sends the policy-rate target and
            # lets the safety layer check the actual target opening geometry.
            conditioned[list(EFFECTOR_CANONICAL_INDICES)] = effector_target
        else:
            conditioned[list(EFFECTOR_CANONICAL_INDICES)] = self._effector_action_filter.update(
                effector_target
            )
        if live_effector_latch is not None and not hold:
            conditioned[list(EFFECTOR_CANONICAL_INDICES)] = live_effector_latch.update(
                conditioned[list(EFFECTOR_CANONICAL_INDICES)]
            )
        chunk_contexts = getattr(self, "_chunk_safety_context", {})
        safety_context.update(chunk_contexts.get(chunk_id or "", {}))
        safety_context.update(
            {
                # Safety/FK operates in physical GDK coordinates. Raw policy
                # chunks remain available in the request trace; do not add an
                # inverse conversion allocation to every 100 Hz servo tick.
                "raw_action": np.asarray(action, dtype=np.float32),
                "controller_target": conditioned.copy(),
                "chunk_id": chunk_id,
                "action_index": action_index,
                "source_sample_monotonic": getattr(self, "_active_safety_sample_time", None),
                "policy_latency_ms": (
                    safety_context.get("policy_latency_ms")
                    if safety_context
                    else (
                        None
                        if getattr(self, "_last_round_trip_s", None) is None
                        else self._last_round_trip_s * 1000
                    )
                ),
            }
        )
        action_lock_requested = time.perf_counter()
        with self._control_lock:
            action_lock_acquired = time.perf_counter()
            if preview is not None:
                safety_context["trajectory_preview"] = preview
            if hold:
                self.adapter.reset_action_limiter(conditioned)
            action_send_started = time.perf_counter()
            if getattr(self.adapter, "_safety_layer", None) is None:
                adopted = self.adapter.send_action(conditioned)
            else:
                adopted = self.adapter.send_action(conditioned, safety_context=safety_context)
            action_send_returned = time.perf_counter()
            if getattr(self.adapter, "last_action_intervened", False):
                # Rejected targets must not accumulate in the low-pass state.
                # Anchor to the actual adopted action, not sampled telemetry.
                if self._action_filter is not None:
                    self._action_filter.reset(adopted)
                if self._effector_action_filter is not None:
                    self._effector_action_filter.reset(
                        np.asarray(adopted)[list(EFFECTOR_CANONICAL_INDICES)]
                    )
            if getattr(getattr(self.adapter, "config", None), "effector_control_mode", None) == "device-position":
                self._record_effector_command(adopted)
            self._last_action = np.asarray(adopted, dtype=np.float32).copy()
            diagnostics = getattr(self, "control_diagnostics", None)
            if diagnostics is not None:
                worker = getattr(self.adapter, "_async_worker", None)
                feedback = None if worker is None else worker.feedback
                motor_velocity = getattr(self.adapter, "_last_motor_velocity_snapshot", None)
                diagnostics.write(dict(
                    record_type="control", timestamp_ns=time.time_ns(), monotonic=time.perf_counter(),
                    loop_started_monotonic=getattr(self, "_active_safety_sample_time", None),
                    action_send_started_monotonic=action_send_started,
                    action_send_returned_monotonic=action_send_returned,
                    action_lock_wait_ms=(action_lock_acquired - action_lock_requested) * 1000,
                    session_id=self.session_id, hardware_mode=self.hardware_mode,
                    chunk_id=chunk_id, action_index=action_index, explicit_hold=hold,
                    policy_control_target=np.asarray(action), filtered_target=conditioned,
                    controller_candidate=getattr(self.adapter, "_last_controller_candidate", None),
                    adopted_target=np.asarray(adopted),
                    planner_velocity=getattr(getattr(self.adapter, "_kinematic_limiter", None), "velocity", None),
                    measured_state=(feedback[0] if feedback is not None else
                                    getattr(self.adapter, "_last_feedback_state", None)),
                    feedback_monotonic=(getattr(self.adapter, "_last_feedback_monotonic", None)
                                        if feedback is None else feedback[1]),
                    motor_velocity_monotonic=(None if motor_velocity is None else motor_velocity[0]),
                    motor_velocity=(None if motor_velocity is None else motor_velocity[1]),
                    dispatch_monotonic=getattr(self.adapter, "_last_servo_dispatch", None),
                    sdk_call_started=getattr(self.adapter, "_last_servo_call_started", None),
                    sdk_call_ms=getattr(self.adapter, "_last_servo_call_ms", None),
                    collision_response=getattr(self.adapter, "_last_collision_response", None),
                    dispatch_provenance=getattr(self.adapter, "_last_dispatch_provenance", None),
                    command_pending=bool(getattr(self.adapter, "last_command_pending", False)),
                    filter_reanchored=bool(getattr(self.adapter, "last_action_intervened", False)),
                    command_accepted=getattr(self.adapter, "_last_command_accepted", None),
                ))
            return adopted

    def _watchdog_expired(self) -> bool:
        # Sync intentionally waits until the active chunk is exhausted before
        # requesting the next one.  Measuring its watchdog from the previous
        # response makes a normal chunk duration + inference interval look
        # like a communications outage, clears the response anchor, and then
        # forces the valid response to be discarded.  Sync already holds the
        # last successfully sent target while it waits for the next chunk.
        if self.inference_mode == "sync":
            return False
        return (
            self._last_response_monotonic is not None
            and time.monotonic() - self._last_response_monotonic > self.watchdog_timeout_s
        )

    @staticmethod
    def _timing_summary(values: list[float]) -> tuple[float, float, float]:
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(np.percentile(array, 95)), float(array.max())

    def _maybe_log_servo_timing(self, now: float) -> None:
        if now - self._servo_window_started < 1.0 or not self._servo_send_ms:
            return
        send_mean, send_p95, send_max = self._timing_summary(self._servo_send_ms)
        if self._servo_intervals_ms:
            loop_mean, loop_p95, loop_max = self._timing_summary(self._servo_intervals_ms)
        else:
            loop_mean = loop_p95 = loop_max = 0.0
        snapshot = self.plan.snapshot()
        plan_start = getattr(self.plan, "_trajectory_start_time", None)
        plan_elapsed = (
            max(0.0, now - float(plan_start))
            if plan_start is not None
            else 0.0
        )
        expected_index = (
            int(math.floor(plan_elapsed * float(self.plan.policy_fps) + 1e-9))
            if plan_start is not None
            else -1
        )
        sampled_wall_ns = time.time_ns()
        self._servo_timing_reports.append((
            "Servo timing target=%.1fHz ticks=%d policy_steps=%d queue=%d "
            "loop_ms(mean/p95/max)=%.3f/%.3f/%.3f "
            "send_ms(mean/p95/max)=%.3f/%.3f/%.3f deadline_misses=%d sampled_wall_ns=%d",
            (
                self.servo_fps, self._servo_ticks, self._executed_actions,
                snapshot.remaining_actions, loop_mean, loop_p95, loop_max,
                send_mean, send_p95, send_max, self._servo_deadline_misses, sampled_wall_ns,
            ),
        ))
        self._servo_timing_reports.append((
            "ActionPlan clock policy_fps=%.1f elapsed=%.3fs expected_completed=%d "
            "next_index=%d first_index=%d valid_end=%d remaining=%d sampled_wall_ns=%d",
            (
                float(self.plan.policy_fps), plan_elapsed, expected_index,
                snapshot.next_action_index, int(getattr(self.plan, "_first_index", 0)),
                int(getattr(self.plan, "_valid_end_index", 0)), snapshot.remaining_actions,
                sampled_wall_ns,
            ),
        ))
        self._servo_window_started = now
        self._servo_intervals_ms.clear()
        self._servo_send_ms.clear()
        self._servo_deadline_misses = 0

    def _flush_servo_timing_reports(self) -> None:
        while self._servo_timing_reports:
            format_string, values = self._servo_timing_reports.popleft()
            logger.info(format_string, *values)

    def _action_loop(self) -> None:
        cadence = None
        try:
            if self.rt_cadence:
                cadence = RealtimeCadence(self.adapter._async_worker)
            while not self.shutdown_event.is_set() and not self._execution_armed_event.is_set():
                # The first model response may be slow or unavailable. Accept
                # pause/resume without arming or issuing any pre-response GDK
                # commands; otherwise pause acknowledgement deadlocks here.
                if self._pause_requested_event.is_set() and not self._pause_active_event.is_set():
                    self._enter_soft_pause(confirm_command=self.flowpro_capture is not None)
                elif not self._pause_requested_event.is_set() and self._pause_active_event.is_set():
                    self._leave_soft_pause()
                # FlowPRO starts paused and must keep its sole action thread
                # alive for Human handback and rollback before Policy is armed.
                if self.flowpro_capture is not None and self._pause_active_event.is_set():
                    break
                self.shutdown_event.wait(0.01)
            if self.shutdown_event.is_set():
                return
            if cadence is not None:
                cadence.start()
                logger.info("Servo cadence uses SCHED_RR priority 1 for action and feedback threads")
            period = 1.0 / self.servo_fps
            while not self.shutdown_event.is_set():
                loop_started = time.perf_counter()
                pause_hold_sent = False
                if self._pause_requested_event.is_set() and not self._pause_active_event.is_set():
                    self._enter_soft_pause(confirm_command=True)
                    if not self._pause_active_event.is_set():
                        self.shutdown_event.wait(period)
                        continue
                    pause_hold_sent = True
                elif not self._pause_requested_event.is_set() and self._pause_active_event.is_set():
                    self._leave_soft_pause()
                if self._flowpro_action_tick():
                    self.shutdown_event.wait(period)
                    continue
                if self._previous_servo_tick is not None:
                    self._servo_intervals_ms.append((loop_started - self._previous_servo_tick) * 1000)
                self._previous_servo_tick = loop_started
                soft_paused = self._pause_active_event.is_set()
                watchdog_expired = False if soft_paused else self._watchdog_expired()
                if watchdog_expired and not self._watchdog_holding:
                    self._watchdog_holding = True
                    # Retire this generation once, not every holding tick;
                    # otherwise every fresh recovery response becomes stale.
                    self.plan.clear()
                    logger.error(
                        "WATCHDOG ENTER HOLD: no successful inference response for %.3fs; "
                        "clearing the action queue and holding the last successfully sent target",
                        self.watchdog_timeout_s,
                    )
                elif not watchdog_expired and self._watchdog_holding:
                    self._watchdog_holding = False
                    logger.warning("WATCHDOG RECOVERED: successful inference responses resumed")
                snapshot = self.plan.snapshot()
                queue_underrun = (
                    not watchdog_expired
                    and snapshot.chunk_id is not None
                    and snapshot.remaining_actions == 0
                )
                if queue_underrun and not self._queue_underrun_active:
                    self._queue_underrun_active = True
                    if self.inference_mode == "sync":
                        logger.info(
                            "SYNC CHUNK COMPLETE: holding the last successfully sent target "
                            "while waiting for the next chunk"
                        )
                    else:
                        logger.error(
                            "ACTION QUEUE UNDERRUN: holding the last successfully sent target "
                            "until a new chunk arrives"
                        )
                elif not queue_underrun and self._queue_underrun_active:
                    self._queue_underrun_active = False
                    if self.inference_mode == "sync":
                        logger.info("SYNC NEXT CHUNK READY: execution resumed")
                    else:
                        logger.warning("ACTION QUEUE RECOVERED: a new executable chunk is available")
                reservation = (
                    None
                    if watchdog_expired or soft_paused or not self._execution_armed_event.is_set()
                    else self.plan.reserve(now=loop_started)
                )
                if reservation is not None:
                    try:
                        send_started = time.perf_counter()
                        self._active_safety_chunk_id = getattr(reservation, "chunk_id", None)
                        self._active_safety_action_index = getattr(reservation, "action_index", -1)
                        self._active_safety_sample_time = loop_started
                        preview_options = {}
                        adapter = getattr(self, "adapter", None)
                        if (not getattr(reservation, "is_hold", False)
                                and getattr(getattr(adapter, "config", None), "rolling_legacy_geometry", False)
                                and getattr(adapter, "_async_worker", None) is not None):
                            # Eight queued points plus a seven-point refill need
                            # 17 lookahead entries; leave room for the anchor.
                            preview_options["trajectory_preview"] = (
                                lambda r=reservation, t=loop_started, p=period:
                                self.plan.preview_reserved_with_indices(r, now=t, period=p, count=20)
                            )
                        sent = (
                            self._send_action(reservation.action, hold=True)
                            if getattr(reservation, "is_hold", False)
                            else self._send_action(reservation.action, **preview_options)
                        )
                        self._servo_send_ms.append((time.perf_counter() - send_started) * 1000)
                        collision_pause = (
                            getattr(self, "pause_on_collision", False)
                            and not getattr(reservation, "is_hold", False)
                            and getattr(self.adapter, "last_async_response", None) == "async_collision_hold"
                        )
                        if collision_pause:
                            # Fence in-flight inference BEFORE releasing the
                            # reservation. A late reply must not arm the robot.
                            self._safety_pause_reason = "collision_risk"
                            self._pause_requested_event.set()
                            self._resume_ready_event.clear()
                        advance = not self._grasp_hold_event.is_set() and not collision_pause
                        self.plan.commit(
                            reservation, sent,
                            advance=advance,
                        )
                        self._record_flowpro_policy_dispatch()
                        if collision_pause:
                            self.plan.clear()
                            self._write_control_status(
                                0, "pausing", "前瞻碰撞风险：已丢弃旧动作，正在确认实测姿态安全保持",
                                pause_reason="collision_risk",
                            )
                            logger.error("COLLISION RISK ENTER PAUSE: discarded moving plan; explicit resume required")
                        if self._grasp_replan_event.is_set():
                            with self._session_lock:
                                self.plan.clear()
                                self.plan.set_hold_action(sent)
                                self.session_id = uuid.uuid4().hex
                                self._execution_armed_event.clear()
                                self._last_response_monotonic = None
                                self._observation_latencies_s.clear()
                                self._state_latencies_s.clear()
                            self._grasp_replan_event.clear()
                            logger.info(
                                "Gripper confirmed; discarded stale post-grasp plan and "
                                "requested fresh inference from the held state"
                            )
                    except SafetyDispatchExpiredError as error:
                        self.plan.cancel(reservation)
                        self._request_safety_expiry_pause(error)
                        continue
                    except Exception:
                        self.plan.cancel(reservation)
                        raise
                    self._last_action = sent
                    self._servo_ticks += 1
                    if advance:
                        self._executed_actions += reservation.advanced_policy_steps
                    if self.trace_recorder is not None:
                        self.trace_recorder.record_servo_sample(
                            monotonic_time=loop_started,
                            action=sent,
                            chunk_id=reservation.chunk_id,
                            action_index=reservation.action_index,
                        )
                elif self._last_action is not None and not pause_hold_sent:
                    send_started = time.perf_counter()
                    try:
                        self._last_action = self._send_action(self._last_action, hold=True)
                    except SafetyDispatchExpiredError as error:
                        self._request_safety_expiry_pause(error)
                        continue
                    self._servo_send_ms.append((time.perf_counter() - send_started) * 1000)
                    self._servo_ticks += 1
                    if self.trace_recorder is not None:
                        self.trace_recorder.record_servo_sample(
                            monotonic_time=loop_started,
                            action=self._last_action,
                            chunk_id=None,
                            action_index=-1,
                        )

                self._maybe_log_servo_timing(time.perf_counter())
                # Nominal loop scheduling alone cannot bound actual dispatch
                # intervals when guard latency varies. The adapter also paces
                # the final GDK call; never catch up late loop starts here.
                elapsed = time.perf_counter() - loop_started
                current_time = time.perf_counter()
                last_dispatch = getattr(self.adapter, "_last_servo_dispatch", None)
                if getattr(self.adapter, "_async_worker", None) is not None and last_dispatch is not None:
                    # Prepare before the next hardware slot; the adapter paces
                    # the final call. Relative sleeps would compound wake-up
                    # drift with the mandatory dispatch-spacing wait.
                    next_tick = servo_preparation_time(
                        last_dispatch, period,
                        rolling=bool(getattr(getattr(self.adapter, "config", None), "rolling_safety_enabled", False)),
                    )
                    delay = next_tick - current_time
                else:
                    delay = period - elapsed
                if delay > 0:
                    self.shutdown_event.wait(delay)
                if elapsed >= period:
                    missed = max(1, int(math.floor(elapsed / period)))
                    self._servo_deadline_misses += missed
        except Exception as error:
            self._fatal_error = f"Action executor failed; physical command streaming stopped: {error}"
            logger.critical(self._fatal_error, exc_info=True)
            self.shutdown_event.set()
        finally:
            if cadence is not None:
                try:
                    cadence.stop()
                except Exception as error:
                    restore_error = f"Could not restore normal servo scheduling: {error}"
                    self._fatal_error = (
                        f"{self._fatal_error}; {restore_error}"
                        if self._fatal_error else restore_error
                    )
                    logger.critical(self._fatal_error, exc_info=True)
                    self.shutdown_event.set()

    def _flowpro_action_tick(self) -> bool:
        """Give a prepared FlowPRO lease exclusive use of the GDK action slot."""
        if self.flowpro_capture is None:
            return False
        if self._flowpro_hold_reanchor_requested_event.is_set():
            self._execute_flowpro_human_hold_reanchor()
            return True
        # Vendor authority has already changed when this latch is set. Do
        # not dispatch another rollback point while DDS records the takeover.
        if self._flowpro_human_control_active_event.is_set():
            return True
        with self._flowpro_rollback_lock:
            rollback_active = self._flowpro_rollback_lease_id is not None
        if rollback_active:
            self._step_flowpro_rollback()
            receipt = self.adapter.dispatch_receipt()
            if receipt is not None and receipt.get("physical_accepted") is True:
                self._last_action = np.asarray(receipt["action"], dtype=np.float32).copy()
                self._servo_ticks += 1
            return True
        return False

    def _read_only_action_loop(self) -> None:
        """Advance the policy clock for RTC without entering the GDK command path."""
        period = 1.0 / self.servo_fps
        next_tick = time.perf_counter()
        try:
            while not self.shutdown_event.is_set():
                now = time.perf_counter()
                if now < next_tick:
                    self.shutdown_event.wait(next_tick - now)
                    continue
                if now - next_tick >= period:
                    self._servo_deadline_misses += int((now - next_tick) / period)
                next_tick = now + period
                if self._pause_requested_event.is_set() or not self._execution_armed_event.is_set():
                    continue
                if self._watchdog_expired():
                    if not self._watchdog_holding:
                        self._watchdog_holding = True
                        self.plan.clear()
                        logger.error("READ-ONLY WATCHDOG: virtual action queue cleared")
                    continue
                self._watchdog_holding = False
                snapshot = self.plan.snapshot()
                underrun = snapshot.chunk_id is not None and snapshot.remaining_actions == 0
                if underrun and not self._queue_underrun_active:
                    self._queue_underrun_active = True
                    logger.error("READ-ONLY ACTION QUEUE UNDERRUN")
                elif not underrun:
                    self._queue_underrun_active = False
                reservation = self.plan.reserve(now=now)
                if reservation is None:
                    continue
                # This commit advances only the simulated timeline. It never
                # calls _send_action(), prepare_execution(), or GDK.
                self.plan.commit(reservation, reservation.action)
                self._last_action = reservation.action
                self._servo_ticks += 1
                self._executed_actions += reservation.advanced_policy_steps
                if self.trace_recorder is not None:
                    self.trace_recorder.record_servo_sample(
                        monotonic_time=now,
                        action=reservation.action,
                        chunk_id=reservation.chunk_id,
                        action_index=reservation.action_index,
                    )
        except Exception as error:
            self._fatal_error = f"Read-only virtual action clock failed: {error}"
            logger.critical(self._fatal_error, exc_info=True)
            self.shutdown_event.set()

    def _reset_to_measured_hold(self) -> np.ndarray:
        with self._control_lock:
            measured = np.asarray(self.adapter.read_state(), dtype=np.float32)
            self.adapter.reanchor_measured_hold(measured)
        self.plan.clear()
        self.plan.set_hold_action(measured)
        self._last_action = measured.copy()
        if self._action_filter is not None:
            self._action_filter.reset(measured)
        if self._effector_action_filter is not None:
            self._effector_action_filter.reset(measured[list(EFFECTOR_CANONICAL_INDICES)])
        effector_latch = getattr(self, "_effector_endpoint_latch", None)
        if effector_latch is not None:
            effector_latch.reset(measured[list(EFFECTOR_CANONICAL_INDICES)])
        self._tracking_error_consecutive = 0
        self._effector_tracking_consecutive = [0, 0]
        self._effector_tracking_seen_revision = list(self._effector_command_revision)
        self._effector_last_command = measured[list(EFFECTOR_CANONICAL_INDICES)].copy()
        return measured

    def _enter_soft_pause(self, *, confirm_command: bool = False) -> None:
        with self._session_lock:
            self._execution_armed_event.clear()
            self._resume_ready_event.clear()
            measured = getattr(self, "_pending_pause_reference", None)
            if measured is None:
                measured = self._reset_to_measured_hold()
                self._pending_pause_reference = measured
            if confirm_command and not (
                self.flowpro_capture is not None
                and self._flowpro_human_control_active_event.is_set()
            ):
                # A running task is not acknowledged as paused until its hold
                # has passed the normal safety and GDK result path.
                self._last_action = self._send_action(measured, hold=True)
                if getattr(self.adapter, "last_command_pending", False):
                    return
            self._pending_pause_reference = None
            self._pause_active_event.set()
        if getattr(self, "_safety_pause_reason", None) == "collision_risk":
            self._write_control_status(
                0, "paused", "前瞻碰撞风险：已确认安全保持；请检查工作区后手动继续或结束任务",
                pause_reason="collision_risk",
            )
        logger.warning(
            "SOFT PAUSE ACTIVE: action plan cleared; inference suspended; holding measured pose %s",
            np.array2string(measured, precision=5),
        )

    def _leave_soft_pause(self) -> None:
        with self._session_lock:
            self._pending_pause_reference = None
            measured = self._reset_to_measured_hold()
            self.session_id = uuid.uuid4().hex
            self._chunk_safety_context.clear()
            self._observation_latencies_s.clear()
            self._state_latencies_s.clear()
            self._safety_pause_reason = None
            self._last_response_monotonic = None
            self._last_round_trip_s = None
            self._watchdog_holding = False
            self._queue_underrun_active = False
            self._frozen_trace_observation = None
            self._pause_active_event.clear()
        logger.warning(
            "SOFT RESUME REQUESTED: measured pose re-anchored; waiting for reset inference chunk %s",
            np.array2string(measured, precision=5),
        )

    def _write_control_status(self, sequence: int, state: str, message: str, *, pause_reason: str | None = None) -> None:
        if self._control_status_file is None:
            return
        self._control_status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._control_status_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "state": state,
                    "message": message,
                    "pause_reason": pause_reason,
                    "hardware_mode": self.hardware_mode,
                    "updated_at": time.time(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._control_status_file)

    def _wait_for_control_event(self, event: threading.Event, timeout_s: float) -> bool:
        """Wait for a control acknowledgement without delaying shutdown."""

        deadline = time.monotonic() + timeout_s
        while not self.shutdown_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return event.is_set()
            if event.wait(min(0.05, remaining)):
                return True
        return False

    def _control_loop(self) -> None:
        last_sequence = -1
        self._write_control_status(0, "running", "客户端运行中")
        while not self.shutdown_event.wait(0.05):
            if self._control_file is None:
                continue
            try:
                command = json.loads(self._control_file.read_text(encoding="utf-8"))
                sequence = int(command["sequence"])
                requested = str(command["command"])
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if sequence <= last_sequence:
                continue
            last_sequence = sequence
            if requested == "pause":
                self._pause_requested_event.set()
                # Without a physical action thread, read-only/dry-run enters
                # its safe software pause directly. This reads/reanchors the
                # observed state but never calls send_action().
                if self.hardware_mode != "execute":
                    self._enter_soft_pause()
                if self._wait_for_control_event(self._pause_active_event, 5.0):
                    self._write_control_status(sequence, "paused", "动作计划已清空，保持控制已确认")
                elif self.shutdown_event.is_set():
                    break
                else:
                    self._write_control_status(sequence, "error", "暂停确认超时")
            elif requested == "resume":
                self._resume_ready_event.clear()
                self._pause_requested_event.clear()
                # Read-only/dry-run has no action thread to leave the pause.
                # Re-anchor here, then let PolicyCommunication request and
                # install a fresh plan before acknowledging the resume.
                if self.hardware_mode != "execute":
                    self._leave_soft_pause()
                if self._wait_for_control_event(self._resume_ready_event, 10.0):
                    self._write_control_status(sequence, "running", "新推理计划已安装，任务继续")
                elif self.shutdown_event.is_set():
                    break
                else:
                    self._pause_requested_event.set()
                    self._write_control_status(sequence, "error", "继续确认超时，已重新请求保持")

    def warn_initial_state(self) -> np.ndarray:
        with self._control_lock:
            state = self.adapter.read_state()
        logger.info("Initial canonical 24D state: %s", np.array2string(state, precision=5))
        expected = np.asarray(APPROVED_START_STATE_24D, dtype=np.float32)
        head_waist_error = np.abs(state[16:24] - expected[16:24])
        max_abs = float(np.max(head_waist_error))
        if max_abs > self.initial_head_waist_warning_threshold:
            logger.warning(
                "Initial head/waist differs from the approved Nexus start pose by more than %.5f "
                "(max_error=%.5f); continuing",
                self.initial_head_waist_warning_threshold,
                max_abs,
            )
        return state

    def run(
        self,
        duration_s: float,
        *,
        startup_hook: Callable[[], None] | None = None,
        shutdown_hook: Callable[[], None] | None = None,
    ) -> None:
        self.validate_server()
        initial_state = self.warn_initial_state()
        if bool(self.policy_client.metadata.get("head_waist_locked", False)):
            head_waist_indices = HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES
            self._held_head_waist_values = initial_state[list(head_waist_indices)].copy()
            logger.warning(
                "Head/waist commands are locked to the measured startup pose: %s",
                np.array2string(self._held_head_waist_values, precision=5),
            )
        if self._hold_effectors:
            self._held_effector_values = initial_state[list(EFFECTOR_CANONICAL_INDICES)].copy()
            logger.warning(
                "Effectors are held at the measured startup values: %s",
                np.array2string(self._held_effector_values, precision=5),
            )
        if self._action_filter is not None:
            self._action_filter.reset(initial_state)
        if self._effector_action_filter is not None:
            self._effector_action_filter.reset(initial_state[list(EFFECTOR_CANONICAL_INDICES)])
        self.plan.set_hold_action(initial_state)
        self._last_action = initial_state.copy()
        if self._start_paused:
            self._pause_requested_event.set()
            logger.info("Client starts in requested soft-pause state")
        self._perform_warmup()
        self._perform_execution_latency_preflight()
        if self.hardware_mode == "execute":
            if self.rt_cadence:
                config = self.adapter.config
                if not (config.safety_enforcement_enabled and config.async_safety_enabled
                        and config.rolling_safety_enabled and math.isclose(
                            self.servo_fps, 100.0, rel_tol=0.0, abs_tol=1e-9)):
                    raise RuntimeError("Realtime cadence requires the enforced rolling safety chain")
                prepare_realtime_cadence(self.adapter._async_worker)
            # This is the first operation allowed to enter GDK control mode.
            # Everything above this line is observation/model preflight only.
            self.adapter.prepare_execution()
        self._communication_thread = threading.Thread(
            target=self._communication_loop,
            name="PolicyCommunication",
            daemon=True,
        )
        # Observation modes need a virtual policy clock for replan/RTC, but
        # must never traverse the physical action/safety-dispatch path.
        self._action_thread = threading.Thread(
            target=(self._action_loop if self.hardware_mode == "execute"
                    else self._read_only_action_loop),
            name=("ActionExecutor" if self.hardware_mode == "execute" else "VirtualActionClock"),
            daemon=True,
        )
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="SoftPauseControl",
            daemon=True,
        )
        if self.hardware_mode != "execute":
            self._read_only_telemetry_thread = threading.Thread(
                target=self._read_only_telemetry_loop,
                name="ReadOnlyViewerTelemetry",
                daemon=True,
            )
        if self._inference_frame_writer is not None:
            self._inference_frame_writer.start()
        if self._presence_frame_publisher is not None:
            self._presence_frame_publisher.start()
            self._presence_capture_thread = threading.Thread(
                target=self._presence_capture_loop,
                name="PresenceCameraPreview",
                daemon=True,
            )
            self._presence_capture_thread.start()
        self._communication_thread.start()
        if self._action_thread is not None:
            self._action_thread.start()
        self._control_thread.start()
        if self._read_only_telemetry_thread is not None:
            self._read_only_telemetry_thread.start()
        logger.info(
            "Robot client started session=%s mode=%s hardware=%s execution_fps=%.1f servo_fps=%.1f",
            self.session_id,
            self.inference_mode,
            self.hardware_mode,
            self.fps,
            self.servo_fps,
        )
        active_elapsed_s = 0.0
        last_checked = time.monotonic()
        duration_auto_paused = False
        try:
            if startup_hook is not None:
                startup_hook()
            while not self.shutdown_event.wait(0.2):
                now = time.monotonic()
                paused = self._pause_requested_event.is_set() or self._pause_active_event.is_set()
                if duration_auto_paused and not paused:
                    # A confirmed resume starts a fresh bounded run window.
                    duration_auto_paused = False
                    active_elapsed_s = 0.0
                    logger.info("Configured-duration timer restarted after soft resume")
                if not paused:
                    active_elapsed_s += now - last_checked
                last_checked = now
                if duration_s > 0 and active_elapsed_s >= duration_s and not duration_auto_paused:
                    logger.warning(
                        "Configured active duration %.3fs reached; requesting automatic soft pause",
                        duration_s,
                    )
                    self._pause_requested_event.set()
                    if self.hardware_mode != "execute":
                        self._enter_soft_pause()
                    if not self._wait_for_control_event(self._pause_active_event, 5.0):
                        raise RuntimeError(
                            "Configured duration elapsed, but automatic soft pause was not acknowledged"
                        )
                    duration_auto_paused = True
                    active_elapsed_s = 0.0
                    self._write_control_status(
                        0,
                        "paused",
                        "配置运行时长已到，客户端自动软暂停",
                    )
        finally:
            try:
                if shutdown_hook is not None:
                    shutdown_hook()
            finally:
                self.stop()
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self.shutdown_event.set()
        for thread in (self._communication_thread, self._action_thread, self._read_only_telemetry_thread):
            if thread is not None:
                thread.join(timeout=5)
        if self._control_thread is not None:
            self._control_thread.join(timeout=1)
        if self._presence_capture_thread is not None:
            self._presence_capture_thread.join(timeout=1)
        if self._presence_frame_publisher is not None:
            self._presence_frame_publisher.close()
        if self._inference_frame_writer is not None:
            self._inference_frame_writer.close()
        self.plan.clear()
        if self.trace_recorder is not None:
            self.trace_recorder.close()
            logger.info("Action trace finalized at %s", self.trace_recorder.run_dir)
        if getattr(self, "control_diagnostics", None) is not None:
            self.control_diagnostics.close()
        self.policy_client.close()
        self.adapter.disconnect()
        self._flush_servo_timing_reports()
        self._write_control_status(-1, "stopped", "客户端已停止")
        logger.info(
            "Robot client stopped after %d policy steps and %d servo ticks",
            self._executed_actions,
            self._servo_ticks,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="GPU policy server IP or hostname")
    parser.add_argument(
        "--source-address",
        default=None,
        help="Local robot IP used as the policy WebSocket source address",
    )
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--task", required=True, help="Exact task prompt used in training data")
    parser.add_argument("--inference-mode", choices=("sync", "async", "rtc"), default="async")
    parser.add_argument(
        "--execution-fps",
        "--fps",
        dest="fps",
        type=float,
        default=30.0,
        help="Client action execution frequency; --fps is retained as a compatibility alias",
    )
    parser.add_argument(
        "--servo-fps",
        type=float,
        default=100.0,
        help="Fixed GDK command frequency; independent of policy --fps",
    )
    parser.add_argument("--prediction-horizon", type=int, default=50)
    parser.add_argument("--quality-horizon", type=int, default=50)
    parser.add_argument(
        "--min-replan-after-steps",
        "--replan-after-steps",
        dest="min_replan_after_steps",
        type=int,
        default=1,
        help="Minimum new policy steps consumed before adaptive replanning; old flag is an alias.",
    )
    parser.add_argument("--request-lead-margin-steps", type=int, default=8)
    parser.add_argument("--min-rtc-blend-steps", type=int, default=8)
    parser.add_argument("--latency-window-size", type=int, default=20)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Successful non-RTC requests to discard before latency estimation and execution.",
    )
    parser.add_argument(
        "--execution-preflight-requests",
        type=int,
        default=5,
        help="Warm-model requests used to estimate P95 before async/RTC --execute is armed.",
    )
    parser.add_argument("--watchdog-timeout-s", type=float, default=3.0)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--camera-sync-tolerance-ms",
        type=float,
        default=5.0,
        help="Maximum GDK timestamp skew between the three selected camera frames.",
    )
    parser.add_argument(
        "--camera-sync-timeout-ms",
        type=float,
        default=40.0,
        help="Maximum wait for matching wrist frames after selecting the head frame.",
    )
    parser.add_argument(
        "--camera-sync-poll-ms",
        type=float,
        default=2.0,
        help="Retry interval while matching GDK wrist-frame timestamps.",
    )
    parser.add_argument("--dry-camera-width", type=int, default=640)
    parser.add_argument("--dry-camera-height", type=int, default=400)
    parser.add_argument("--wrist-image-width", type=int, default=640)
    parser.add_argument("--wrist-image-height", type=int, default=400)
    parser.add_argument("--wrist-jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--no-resize-wrist-images",
        dest="resize_wrist_images",
        action="store_false",
        help="Send original GDK wrist JPEGs instead of resizing them to the training resolution.",
    )
    parser.add_argument(
        "--left-effector-name",
        "--left-effector-type",
        dest="left_effector_name",
        default=None,
        help="Override the left GDK effector joint name; normally auto-detected.",
    )
    parser.add_argument(
        "--right-effector-name",
        "--right-effector-type",
        dest="right_effector_name",
        default=None,
        help="Override the right GDK effector joint name; normally auto-detected.",
    )
    parser.add_argument(
        "--effectors-mode",
        choices=("required", "absent_fixed"),
        default="required",
        help=(
            "Use required for robots with GDK effectors. absent_fixed skips get_end_state(), "
            "synthesizes two fixed observation values, and is restricted to non-execute modes."
        ),
    )
    parser.add_argument(
        "--effector-profile",
        choices=EFFECTOR_PROFILE_CHOICES,
        default="dahuan",
        help=(
            "Atomic gripper semantics/profile. Dahuan converts model action 0=open and "
            "0.025=close into physical GDK state 0=closed and 0.025=open before "
            "the 100 Hz filter/safety path."
        ),
    )
    parser.add_argument("--effector-control-mode", choices=("software-trajectory", "device-position"),
                        default="software-trajectory", help="Dahuan device-position uses policy-rate held goals and checks the current target opening geometry.")
    parser.add_argument("--fixed-left-effector-state", type=float, default=0.0)
    parser.add_argument("--fixed-right-effector-state", type=float, default=0.0)
    parser.add_argument("--max-relative-target", type=float, default=0.05)
    parser.add_argument("--max-gripper-relative-target", type=float, default=None)
    parser.add_argument("--relative-target-mode", choices=("warn", "clip"), default="warn")
    parser.add_argument("--initial-head-waist-warning-threshold", type=float, default=0.01)
    parser.add_argument("--max-arm-tracking-error", type=float, default=0.4)
    parser.add_argument("--max-effector-tracking-error", type=float, default=None)
    parser.add_argument("--max-head-waist-tracking-error", type=float, default=0.2)
    parser.add_argument("--tracking-error-consecutive-limit", type=int, default=3)
    parser.add_argument("--max-message-mb", type=int, default=64)
    parser.add_argument("--log-file", type=Path, default=Path("logs/robot_client.log"))
    parser.add_argument(
        "--timing-log-file",
        type=Path,
        default=None,
        help="JSONL request timing output; defaults beside --log-file.",
    )
    parser.add_argument(
        "--action-trace-dir",
        type=Path,
        default=None,
        help="Record raw model chunks and 100 Hz virtual commands; requires --read-only.",
    )
    parser.add_argument(
        "--no-trace-images",
        dest="trace_save_images",
        action="store_false",
        help="Do not copy the three encoded input images into each action trace request.",
    )
    parser.add_argument(
        "--freeze-trace-observation",
        action="store_true",
        help="Reuse the first state/images for every traced request to isolate model sampling variance.",
    )
    parser.add_argument(
        "--action-low-pass-cutoff-hz",
        type=float,
        default=2.0,
        help="Causal low-pass cutoff at the final servo rate; defaults to the validated 2 Hz setting.",
    )
    parser.add_argument(
        "--no-action-low-pass",
        dest="action_low_pass_cutoff_hz",
        action="store_const",
        const=None,
        help="Disable the final low-pass filter for read-only diagnostics.",
    )
    parser.add_argument(
        "--effector-action-low-pass-cutoff-hz",
        type=float,
        default=2.0,
        help=(
            "Separate low-pass cutoff for effectors; defaults to 2 Hz and is followed by "
            "the mandatory effector velocity/acceleration limiter."
        ),
    )
    parser.add_argument(
        "--effector-command-mode",
        choices=("continuous", "endpoint-latch"),
        default="continuous",
        help="Use endpoint-latch for binary Dahuan open/close commands with a dead band.",
    )
    parser.add_argument(
        "--effector-closed-max",
        type=float,
        default=0.0075,
        help="Physical opening at or below which endpoint-latch commands fully closed.",
    )
    parser.add_argument(
        "--effector-open-min",
        type=float,
        default=0.0175,
        help="Physical opening at or above which endpoint-latch commands fully open.",
    )
    parser.add_argument(
        "--chunk-smoothing-mode",
        choices=("adaptive-fixed-time", "savgol"),
        default="savgol",
        help="Seven-point arm-only Savitzky-Golay smoothing (default) or experimental adaptive fixed-time smoothing.",
    )
    parser.add_argument(
        "--chunk-smoothing-window", type=int, choices=(1, 5, 7, 9, 11, 13, 15), default=7,
        help="Zero-phase arm-only policy-chunk smoothing window; 1 disables it.",
    )
    parser.add_argument(
        "--chunk-smoothing-max-deviation", type=float, default=0.005,
        help="Maximum per-joint displacement introduced by chunk smoothing in radians.",
    )
    parser.add_argument("--adaptive-smoothing-acceleration-weight", type=float, default=20.0)
    parser.add_argument("--adaptive-smoothing-jerk-weight", type=float, default=5.0)
    parser.add_argument("--adaptive-smoothing-velocity-reference", type=float, default=0.25)
    parser.add_argument(
        "--fixed-horizon-projection",
        action="store_true",
        help=(
            "Project arm waypoints onto fixed policy timestamps using the configured per-joint "
            "velocity/acceleration limits; gripper event indices and chunk duration are unchanged."
        ),
    )
    parser.add_argument(
        "--action-interpolation", choices=("linear", "monotone-cubic"), default="linear",
        help=("Linear interpolation (default), or experimental velocity-continuous "
              "monotone cubic interpolation for arms on fixed policy timestamps. "
              "Neither mode replaces the final velocity/collision checks."),
    )
    parser.add_argument(
        "--no-effector-action-low-pass",
        dest="effector_action_low_pass_cutoff_hz",
        action="store_const",
        const=None,
        help="Disable only the effector low-pass filter; kinematic limits remain mandatory.",
    )
    parser.add_argument("--rtc-prefix-mode", choices=("model", "command-shadow", "command"), default="model",
                        help="Command-prefix conditioning is read-only only; shadow records without changing inference.")
    parser.add_argument("--max-arm-velocity", type=float, default=0.6, help="Final arm limit in rad/s.")
    parser.add_argument(
        "--arm-tracking-mode", choices=("legacy", "damped"), default="legacy",
        help="Experimental arm tracking only; damped keeps existing kinematic/collision limits and policy clock. "
             "Does not change gripper targets. Disabled by default; requires controlled validation.",
    )
    parser.add_argument(
        "--arm-tracking-omega", type=float, default=12.0,
        help="Damped tracking natural frequency in rad/s, NOT Hz; omega * servo period must be <= 0.25.",
    )
    parser.add_argument(
        "--max-arm-acceleration", type=float, default=2.0, help="Final arm limit in rad/s^2."
    )
    parser.add_argument(
        "--arm-limit-profile",
        choices=(
            "uniform", "dataset-v30", "dataset-v30-p99-margin",
            "urdf-velocity-p99-acceleration", "urdf-velocity-uniform-acceleration",
            "urdf-velocity-only",
        ),
        default="uniform",
        help=("Per-joint arm limit profile; scalar arm limits remain hard upper bounds. "
              "urdf-velocity-uniform-acceleration reads the active safety URDF joint velocities "
              "and uses the configured scalar acceleration; the URDF has no acceleration field. "
              "urdf-velocity-only keeps those velocity limits but disables normal arm acceleration limiting; "
              "the scalar acceleration remains a finite braking assumption. "
              "The legacy urdf-velocity-p99-acceleration profile uses a fixed 3.1416 rad/s "
              "velocity array and experimental data acceleration envelope."),
    )
    parser.add_argument(
        "--max-effector-velocity", type=float, default=None, help="Final effector limit in profile units/s."
    )
    parser.add_argument(
        "--max-effector-acceleration",
        type=float,
        default=None,
        help="Final effector limit in profile units/s^2.",
    )
    parser.add_argument(
        "--max-head-waist-velocity", type=float, default=0.3, help="Final head/waist limit in rad/s."
    )
    parser.add_argument(
        "--max-head-waist-acceleration",
        type=float,
        default=1.0,
        help="Final head/waist limit in rad/s^2.",
    )
    parser.add_argument(
        "--gdk-position-limit-margin",
        type=float,
        default=0.01,
        help="Safety margin in radians kept inside every vendor GDK body-joint hard limit.",
    )
    parser.add_argument(
        "--max-consecutive-gdk-rejections",
        type=int,
        default=3,
        help="Hold the last successful action after a transient GDK rejection; stop at this count.",
    )
    parser.add_argument(
        "--hold-effectors",
        action="store_true",
        help="Hold both effectors at their measured startup positions.",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=None,
        help=(
            "Enable calibrated gripper/table FK hard projection from this JSON file. "
            "The file must explicitly contain calibrated=true."
        ),
    )
    parser.add_argument(
        "--safety-log-file",
        type=Path,
        default=None,
        help="Per-servo-step safety JSONL; defaults beside --log-file when safety is enabled.",
    )
    parser.add_argument(
        "--safety-monitor-only",
        action="store_true",
        help=(
            "Evaluate and report calibrated collision risk without changing the command. "
            "Forbidden with --execute."
        ),
    )
    parser.add_argument(
        "--detailed-safety-timing",
        action="store_true",
        help="Opt-in per-thread GDK read/geometry timing diagnostics in the safety audit; adds profiling overhead.",
    )
    parser.add_argument(
        "--async-safety", action="store_true",
        help="Run collision geometry in an isolated process, with bounded checked-action leases.",
    )
    parser.add_argument(
        "--certified-braking", action="store_true",
        help="Candidate stopping-reserve controller. Requires --async-safety; read-only/dry-run only, not --execute.",
    )
    parser.add_argument(
        "--rolling-safety", action="store_true",
        help="Checked segment queue (legacy: 10-point startup, append 5, up to 130 ms); bounds remains a read-only candidate.",
    )
    parser.add_argument("--rolling-geometry", choices=("legacy", "bounds"), default="legacy",
                        help="Legacy sampled path/finite table (default), or experimental bounds requiring --certified-braking.")
    parser.add_argument(
        "--safety-check-fps", type=int, choices=(50, 100), default=50,
        help="Full collision checks per second; 50 validates two frozen 100 Hz targets together.",
    )
    parser.add_argument(
        "--safety-guard-path-samples",
        type=int,
        default=2,
        help="Conservative synchronous samples per 100 Hz current-to-target segment.",
    )
    parser.add_argument(
        "--max-safety-guard-latency-ms",
        type=float,
        default=8.0,
        help="Hold measured pose when the synchronous safety guard exceeds this budget.",
    )
    parser.add_argument(
        "--max-consecutive-safety-deadline-misses",
        type=int,
        default=3,
        help="Stop command streaming after this many consecutive safety guard overruns.",
    )
    parser.add_argument(
        "--safety-log-queue-size",
        type=int,
        default=2048,
        help="Bounded asynchronous safety-audit queue; overflow stops execution.",
    )
    parser.add_argument("--control-file", type=Path, default=None)
    parser.add_argument("--control-status-file", type=Path, default=None)
    parser.add_argument(
        "--platform-telemetry-endpoint",
        default=None,
        help="Optional Nexus latest-value telemetry endpoint.",
    )
    parser.add_argument(
        "--platform-telemetry-token-file",
        type=Path,
        default=None,
        help="0600 file containing the per-deployment telemetry token.",
    )
    parser.add_argument(
        "--platform-telemetry-fps",
        type=float,
        default=20.0,
        help="Maximum background upload rate; it never changes the servo frequency.",
    )
    parser.add_argument(
        "--platform-frame-dir",
        type=Path,
        default=None,
        help="Background latest-inference JPEG cache used by the Nexus UI.",
    )
    parser.add_argument("--platform-presence-frame-endpoint", default=None)
    parser.add_argument("--platform-presence-frame-token-file", type=Path, default=None)
    parser.add_argument("--platform-presence-job-id", default=None)
    parser.add_argument("--platform-presence-generation", type=int, default=None)
    parser.add_argument("--platform-presence-frame-fps", type=float, default=10.0)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument(
        "--rt-cadence", action="store_true",
        help="Experimental bounded RR1 action/feedback scheduling; requires robot RT priority limit >= 1",
    )
    parser.add_argument("--pause-on-collision", action="store_true",
                        help="Latch predicted collision into acknowledged measured-pose pause; never auto-chase skipped targets")
    parser.add_argument("--rtc-state-aligned-anchor", action="store_true",
                        help="Align RTC prefix and dropped steps to the state capture; keep conservative camera-age safety budget")
    parser.add_argument(
        "--smooth-action-interpolation",
        action="store_true",
        help="Compatibility option; v12 chunk smoothing remains authoritative.",
    )
    parser.add_argument("--coordinate-grasp-events", action="store_true")
    parser.add_argument("--stationary-hold-feedback-max-age-ms", type=int, choices=(40, 100), default=40)
    parser.set_defaults(trace_save_images=True, resize_wrist_images=True)
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--dry-run", dest="hardware_mode", action="store_const", const="dry-run")
    hardware.add_argument("--read-only", dest="hardware_mode", action="store_const", const="read-only")
    hardware.add_argument("--execute", dest="hardware_mode", action="store_const", const="execute")
    parser.set_defaults(hardware_mode="dry-run")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def configure_logging(path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
        force=True,
    )


def validate_trace_safety(
    hardware_mode: str,
    action_trace_dir: Path | None,
    freeze_trace_observation: bool = False,
) -> None:
    if action_trace_dir is not None and hardware_mode != "read-only":
        raise ValueError("--action-trace-dir is a safety-locked diagnostic and requires --read-only")
    if freeze_trace_observation and action_trace_dir is None:
        raise ValueError("--freeze-trace-observation requires --action-trace-dir")


def validate_workspace_safety_servo_rate(
    safety_config: Path | None,
    servo_fps: float,
) -> None:
    """Keep collision checks on the only currently validated servo rate."""
    if safety_config is not None and not math.isclose(
        servo_fps, 100.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("workspace safety currently requires --servo-fps=100")


def resolve_initialization_scope(server_metadata: dict[str, Any]) -> str:
    profile = str(server_metadata.get("model_profile", ""))
    native_action_dim = server_metadata.get("native_action_dim")
    if native_action_dim == 16 or profile.startswith("16d_"):
        return "16d"
    if native_action_dim == 24 or profile == "24d":
        return "24d"
    raise RuntimeError(
        "Cannot determine GDK initialization scope from server metadata: "
        f"model_profile={profile!r} native_action_dim={native_action_dim!r}"
    )


def run_robot_client(
    args: argparse.Namespace,
    *,
    runner_setup: Callable[[RobotClientRunner], None] | None = None,
    startup_hook: Callable[[], None] | None = None,
    shutdown_hook: Callable[[], None] | None = None,
    start_paused: bool = False,
) -> None:
    if start_paused:
        args.start_paused = True
    if args.rtc_prefix_mode == "command" and args.hardware_mode == "execute":
        raise ValueError("Command-prefix inference is restricted to read-only/dry-run testing")
    if args.rt_cadence and (args.hardware_mode != "execute" or args.safety_config is None
                            or args.safety_monitor_only or not args.async_safety or not args.rolling_safety):
        raise ValueError("--rt-cadence requires --execute with enforced async rolling workspace safety")
    if args.effectors_mode == "absent_fixed" and args.hardware_mode == "execute":
        raise ValueError("--effectors-mode=absent_fixed cannot be used with --execute")
    if args.safety_monitor_only and args.safety_config is None:
        raise ValueError("--safety-monitor-only requires --safety-config")
    fixed_effector_states = (
        args.fixed_left_effector_state,
        args.fixed_right_effector_state,
    )
    effector_profile = get_effector_profile(args.effector_profile)
    args.effector_profile = effector_profile.name
    if args.max_gripper_relative_target is None:
        args.max_gripper_relative_target = effector_profile.default_relative_target_warning
    if args.max_effector_velocity is None:
        args.max_effector_velocity = effector_profile.default_max_velocity
    if args.max_effector_acceleration is None:
        args.max_effector_acceleration = effector_profile.default_max_acceleration
    if args.max_effector_tracking_error is None:
        args.max_effector_tracking_error = effector_profile.default_tracking_error
    effector_min, effector_max = effector_profile.hardware_state_range
    if not all(
        math.isfinite(value)
        and effector_min - 1e-6 <= value <= effector_max + 1e-6
        for value in fixed_effector_states
    ):
        raise ValueError(
            "fixed effector states must be finite and within the selected effector profile limits"
        )
    if args.fps <= 0 or args.servo_fps <= 0 or args.watchdog_timeout_s <= 0 or args.max_message_mb <= 0:
        raise ValueError("policy/servo fps, watchdog timeout, and max message size must be positive")
    if (
        args.prediction_horizon <= 0
        or args.quality_horizon <= 0
        or args.quality_horizon > args.prediction_horizon
        or args.min_replan_after_steps <= 0
        or args.request_lead_margin_steps < 0
        or args.min_rtc_blend_steps < 0
        or args.latency_window_size <= 0
        or args.warmup_requests < 0
        or args.execution_preflight_requests < 0
        or args.tracking_error_consecutive_limit <= 0
        or args.max_consecutive_gdk_rejections <= 0
        or args.safety_guard_path_samples <= 0
        or args.max_consecutive_safety_deadline_misses <= 0
        or args.safety_log_queue_size <= 0
    ):
        raise ValueError(
            "horizons/window/min-replan must be positive, quality<=prediction, and margins>=0"
        )
    if args.action_low_pass_cutoff_hz is not None and (
        args.action_low_pass_cutoff_hz <= 0 or args.action_low_pass_cutoff_hz >= args.servo_fps * 0.5
    ):
        raise ValueError("action low-pass cutoff must be positive and below servo Nyquist frequency")
    if args.effector_action_low_pass_cutoff_hz is not None and (
        args.effector_action_low_pass_cutoff_hz <= 0
        or args.effector_action_low_pass_cutoff_hz >= args.servo_fps * 0.5
    ):
        raise ValueError("effector action low-pass cutoff must be positive and below servo Nyquist frequency")
    if args.effector_command_mode == "endpoint-latch":
        if args.effector_profile != "dahuan" or args.effector_control_mode != "device-position":
            raise ValueError("endpoint-latch requires the Dahuan device-position control path")
        if not (
            effector_min <= args.effector_closed_max
            < args.effector_open_min <= effector_max
        ):
            raise ValueError(
                "endpoint-latch thresholds must be ordered inside the physical effector range"
            )
    if (not math.isfinite(args.chunk_smoothing_max_deviation)
            or args.chunk_smoothing_max_deviation < 0):
        raise ValueError("chunk smoothing max deviation must be finite and non-negative")
    adaptive_values = (
        args.adaptive_smoothing_acceleration_weight,
        args.adaptive_smoothing_jerk_weight,
        args.adaptive_smoothing_velocity_reference,
    )
    if (
        not all(math.isfinite(value) for value in adaptive_values)
        or args.adaptive_smoothing_acceleration_weight < 0
        or args.adaptive_smoothing_jerk_weight < 0
        or args.adaptive_smoothing_velocity_reference <= 0
    ):
        raise ValueError("adaptive smoothing weights must be non-negative and velocity reference positive")
    kinematic_limits = (
        args.max_arm_velocity,
        args.max_arm_acceleration,
        args.max_effector_velocity,
        args.max_effector_acceleration,
        args.max_head_waist_velocity,
        args.max_head_waist_acceleration,
    )
    if not all(math.isfinite(value) and value > 0 for value in kinematic_limits):
        raise ValueError("all final GDK velocity/acceleration limits must be positive and finite")
    if not math.isfinite(args.gdk_position_limit_margin) or args.gdk_position_limit_margin < 0:
        raise ValueError("GDK position limit margin must be finite and non-negative")
    if (
        not math.isfinite(args.max_safety_guard_latency_ms)
        or args.max_safety_guard_latency_ms <= 0
    ):
        raise ValueError("safety guard latency budget must be positive and finite")
    tracking_limits = (
        args.max_arm_tracking_error,
        args.max_effector_tracking_error,
        args.max_head_waist_tracking_error,
    )
    if not all(math.isfinite(value) and value > 0 for value in tracking_limits):
        raise ValueError("all command-to-feedback tracking error limits must be positive and finite")
    if (
        args.wrist_image_width <= 0
        or args.wrist_image_height <= 0
        or not 1 <= args.wrist_jpeg_quality <= 100
    ):
        raise ValueError("wrist image dimensions must be positive and JPEG quality must be in [1, 100]")
    camera_sync_values = (
        args.camera_sync_tolerance_ms,
        args.camera_sync_timeout_ms,
        args.camera_sync_poll_ms,
    )
    if not all(math.isfinite(value) and value >= 0 for value in camera_sync_values):
        raise ValueError("camera synchronization settings must be finite and non-negative")
    validate_trace_safety(
        args.hardware_mode,
        args.action_trace_dir,
        args.freeze_trace_observation,
    )
    validate_workspace_safety_servo_rate(args.safety_config, args.servo_fps)
    if (args.platform_telemetry_endpoint is None) != (
        args.platform_telemetry_token_file is None
    ):
        raise ValueError(
            "platform telemetry endpoint and token file must be configured together"
        )
    if not math.isfinite(args.platform_telemetry_fps) or not (
        0 < args.platform_telemetry_fps <= 20
    ):
        raise ValueError("platform telemetry fps must be in (0, 20]")
    presence_values = (
        args.platform_presence_frame_endpoint,
        args.platform_presence_frame_token_file,
        args.platform_presence_job_id,
        args.platform_presence_generation,
    )
    if any(value is not None for value in presence_values) and not all(
        value is not None for value in presence_values
    ):
        raise ValueError(
            "presence-frame endpoint, token file, job id, and generation must be configured together"
        )
    if not math.isfinite(args.platform_presence_frame_fps) or not (
        0 < args.platform_presence_frame_fps <= 15
    ):
        raise ValueError("platform presence-frame fps must be in (0, 15]")
    if args.coordinate_grasp_events and args.effector_command_mode != "continuous":
        raise ValueError("grasp coordination and endpoint latch cannot be enabled together")
    configure_logging(args.log_file)
    timing_log_file = (
        args.timing_log_file.expanduser().resolve()
        if args.timing_log_file is not None
        else args.log_file.expanduser().resolve().with_name(f"{args.log_file.stem}_timing.jsonl")
    )
    safety_log_file = (
        args.safety_log_file.expanduser().resolve()
        if args.safety_log_file is not None
        else (
            args.log_file.expanduser().resolve().with_name(f"{args.log_file.stem}_safety.jsonl")
            if args.safety_config is not None
            else None
        )
    )
    logger.info("Writing client request timings to %s", timing_log_file)
    logger.info(
        "Selected effector profile=%s state_range=%s model_action_range=%s mapping=%s",
        effector_profile.name,
        effector_profile.hardware_state_range,
        effector_profile.model_action_range,
        effector_profile.action_mapping,
    )
    if args.hardware_mode == "execute":
        logger.warning("PHYSICAL ACTION EXECUTION IS ENABLED")

    policy_client = PolicyClient(
        args.host,
        args.port,
        request_timeout_s=args.request_timeout_s,
        max_message_bytes=args.max_message_mb * 1024 * 1024,
        source_address=args.source_address,
    )
    adapter: G2AAdapter | None = None
    telemetry_publisher: LatestTelemetryPublisher | None = None
    presence_frame_publisher: LatestHeadFramePublisher | None = None
    try:
        metadata = policy_client.connect(wait=True)
        initialization_scope = resolve_initialization_scope(metadata)
        logger.info(
            "Selected GDK initialization scope=%s for profile=%s native_action_dim=%s",
            initialization_scope,
            metadata.get("model_profile"),
            metadata.get("native_action_dim"),
        )
        adapter = G2AAdapter(
            AdapterConfig(
                hardware_mode=args.hardware_mode,
                initialization_scope=initialization_scope,
                effectors_mode=args.effectors_mode,
                effector_profile=args.effector_profile,
                effector_control_mode=args.effector_control_mode,
                fixed_left_effector_state=args.fixed_left_effector_state,
                fixed_right_effector_state=args.fixed_right_effector_state,
                camera_timeout_ms=args.camera_timeout_ms,
                camera_sync_tolerance_ms=args.camera_sync_tolerance_ms,
                camera_sync_timeout_ms=args.camera_sync_timeout_ms,
                camera_sync_poll_ms=args.camera_sync_poll_ms,
                dry_camera_width=args.dry_camera_width,
                dry_camera_height=args.dry_camera_height,
                resize_wrist_images=args.resize_wrist_images,
                parallel_wrist_images=args.rtc_state_aligned_anchor,
                wrist_image_width=args.wrist_image_width,
                wrist_image_height=args.wrist_image_height,
                wrist_jpeg_quality=args.wrist_jpeg_quality,
                servo_control_period_s=1.0 / args.servo_fps,
                left_effector_name=args.left_effector_name,
                right_effector_name=args.right_effector_name,
                max_relative_target=args.max_relative_target,
                max_gripper_relative_target=args.max_gripper_relative_target,
                relative_target_mode=args.relative_target_mode,
                stationary_hold_feedback_max_age_s=args.stationary_hold_feedback_max_age_ms / 1000.0,
                max_arm_velocity=args.max_arm_velocity,
                max_arm_acceleration=args.max_arm_acceleration,
                arm_limit_profile=args.arm_limit_profile,
                arm_tracking_mode=args.arm_tracking_mode,
                arm_tracking_omega=args.arm_tracking_omega,
                max_effector_velocity=args.max_effector_velocity,
                max_effector_acceleration=args.max_effector_acceleration,
                max_head_waist_velocity=args.max_head_waist_velocity,
                max_head_waist_acceleration=args.max_head_waist_acceleration,
                gdk_position_limit_margin=args.gdk_position_limit_margin,
                max_consecutive_gdk_rejections=args.max_consecutive_gdk_rejections,
                safety_config_path=args.safety_config,
                safety_log_path=safety_log_file,
                safety_enforcement_enabled=not args.safety_monitor_only,
                safety_guard_path_samples=args.safety_guard_path_samples,
                safety_check_fps=args.safety_check_fps,
                async_safety_enabled=args.async_safety,
                certified_braking_enabled=args.certified_braking,
                rolling_safety_enabled=args.rolling_safety,
                rolling_legacy_geometry=args.rolling_safety and args.rolling_geometry == "legacy",
                max_safety_guard_latency_ms=args.max_safety_guard_latency_ms,
                max_consecutive_safety_deadline_misses=(
                    args.max_consecutive_safety_deadline_misses
                ),
                safety_log_queue_size=args.safety_log_queue_size,
                detailed_safety_timing=args.detailed_safety_timing,
                platform_telemetry_enabled=args.platform_telemetry_endpoint is not None,
                platform_telemetry_fps=args.platform_telemetry_fps,
            )
        )
        adapter.connect()
        if args.platform_telemetry_endpoint is not None:
            token_path = args.platform_telemetry_token_file.expanduser().resolve()
            telemetry_token = token_path.read_text(encoding="utf-8").strip()
            telemetry_publisher = LatestTelemetryPublisher(
                args.platform_telemetry_endpoint,
                telemetry_token,
                lambda: adapter.latest_control_telemetry,
                fps=args.platform_telemetry_fps,
            )
            telemetry_publisher.start()
            logger.info(
                "Background platform telemetry enabled endpoint=%s max_fps=%.1f",
                args.platform_telemetry_endpoint,
                args.platform_telemetry_fps,
            )
        if args.platform_presence_frame_endpoint is not None:
            presence_token_path = args.platform_presence_frame_token_file.expanduser().resolve()
            presence_token = presence_token_path.read_text(encoding="utf-8").strip()
            presence_frame_publisher = LatestHeadFramePublisher(
                args.platform_presence_frame_endpoint,
                presence_token,
                job_id=args.platform_presence_job_id,
                generation=args.platform_presence_generation,
                fps=args.platform_presence_frame_fps,
            )
            logger.info(
                "Best-effort package-presence frame transport enabled endpoint=%s max_fps=%.1f",
                args.platform_presence_frame_endpoint,
                args.platform_presence_frame_fps,
            )
        runner = RobotClientRunner(
            adapter,
            policy_client,
            task=args.task,
            inference_mode=args.inference_mode,
            hardware_mode=args.hardware_mode,
            fps=args.fps,
            servo_fps=args.servo_fps,
            prediction_horizon=args.prediction_horizon,
            quality_horizon=args.quality_horizon,
            min_replan_after_steps=args.min_replan_after_steps,
            request_lead_margin_steps=args.request_lead_margin_steps,
            min_rtc_blend_steps=args.min_rtc_blend_steps,
            latency_window_size=args.latency_window_size,
            warmup_requests=args.warmup_requests,
            execution_preflight_requests=args.execution_preflight_requests,
            watchdog_timeout_s=args.watchdog_timeout_s,
            initial_head_waist_warning_threshold=args.initial_head_waist_warning_threshold,
            max_arm_tracking_error=args.max_arm_tracking_error,
            max_effector_tracking_error=args.max_effector_tracking_error,
            max_head_waist_tracking_error=args.max_head_waist_tracking_error,
            tracking_error_consecutive_limit=args.tracking_error_consecutive_limit,
            action_trace_dir=args.action_trace_dir,
            trace_save_images=args.trace_save_images,
            freeze_trace_observation=args.freeze_trace_observation,
            action_low_pass_cutoff_hz=args.action_low_pass_cutoff_hz,
            effector_action_low_pass_cutoff_hz=args.effector_action_low_pass_cutoff_hz,
            effector_command_mode=args.effector_command_mode,
            effector_closed_max=args.effector_closed_max,
            effector_open_min=args.effector_open_min,
            chunk_smoothing_mode=args.chunk_smoothing_mode,
            chunk_smoothing_window=args.chunk_smoothing_window,
            chunk_smoothing_max_deviation=args.chunk_smoothing_max_deviation,
            adaptive_smoothing_acceleration_weight=args.adaptive_smoothing_acceleration_weight,
            adaptive_smoothing_jerk_weight=args.adaptive_smoothing_jerk_weight,
            adaptive_smoothing_velocity_reference=args.adaptive_smoothing_velocity_reference,
            fixed_horizon_projection=args.fixed_horizon_projection,
            action_interpolation=args.action_interpolation,
            hold_effectors=args.hold_effectors,
            timing_log_file=timing_log_file,
            control_file=args.control_file,
            control_status_file=args.control_status_file,
            platform_frame_dir=args.platform_frame_dir,
            presence_frame_publisher=presence_frame_publisher,
            presence_capture_fps=args.platform_presence_frame_fps,
            start_paused=args.start_paused,
            coordinate_grasp_events=args.coordinate_grasp_events,
            rt_cadence=args.rt_cadence,
            pause_on_collision=args.pause_on_collision,
            rtc_state_aligned_anchor=args.rtc_state_aligned_anchor,
            rtc_prefix_mode=args.rtc_prefix_mode,
        )
        if runner_setup is not None:
            runner_setup(runner)
        with safety_thread_scheduling(args.safety_config is not None):
            with gc_pause_trace(args.rt_cadence, timing_log_file.with_name("gc-pauses.jsonl")):
                runner.run(
                    args.duration_s,
                    startup_hook=startup_hook,
                    shutdown_hook=shutdown_hook,
                )
    except Exception:
        logger.exception("Robot client terminated due to an unhandled error")
        raise
    finally:
        if presence_frame_publisher is not None:
            presence_frame_publisher.close()
        if telemetry_publisher is not None:
            telemetry_publisher.close()
        policy_client.close()
        if adapter is not None:
            adapter.disconnect()


def main() -> None:
    run_robot_client(parse_args())


def _interrupt() -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: _interrupt())
    main()
