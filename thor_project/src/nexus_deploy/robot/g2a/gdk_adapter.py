"""LeRobot-independent G2A hardware adapter for the vendor GDK runtime."""

from __future__ import annotations

import inspect
import io
import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

import numpy as np

from .action_filter import JointKinematicLimiter
from .async_safety import (
    AsyncSafetyWorker, CHECK_BUDGET_MS, RESULT_MAX_AGE_MS,
    bounded_hold_is_safe, hold_displacement_coefficients,
    braking_sample, braking_clearance, trajectory_clearance, sampled_feedback_displacement,
)
from .constants import (
    ACTION_DIM,
    ARM_CANONICAL_INDICES,
    CAMERA_KEYS,
    DEFAULT_EFFECTOR_JOINT_NAMES,
    EFFECTOR_CANONICAL_INDICES,
    GDK_BODY_JOINT_LIMITS,
    GDK_BODY_JOINTS,
    HEAD_CANONICAL_INDICES,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_EFFECTOR_INDEX,
    RIGHT_ARM_JOINTS,
    RIGHT_EFFECTOR_INDEX,
    WAIST_CANONICAL_INDICES,
    WAIST_JOINTS,
)
from .effector_profiles import EffectorProfile, get_effector_profile
from .robot_lock import RobotProcessLock
from .safety_layer import (
    ActionSafetyLayer,
    RealtimeSafetyController,
    SafetyJSONLLogger,
    load_safety_layer,
)

logger = logging.getLogger(__name__)

_BODY_LIMIT_INDICES = np.asarray(
    tuple(range(0, 7)) + tuple(range(8, 15)) + tuple(range(16, 24)), dtype=np.intp
)
_BODY_LIMIT_LOWER = np.asarray(
    [GDK_BODY_JOINT_LIMITS[name][0] for name in GDK_BODY_JOINTS], dtype=np.float64
)
_BODY_LIMIT_UPPER = np.asarray(
    [GDK_BODY_JOINT_LIMITS[name][1] for name in GDK_BODY_JOINTS], dtype=np.float64
)

# Geometry remains tied to its 200 ms sampled reference and new motion still
# needs a returned certificate.  This separate limit is only for the latest
# GDK feedback used to renew/dispatch that certificate.  The former 40 ms
# limit was shorter than observed 48--72 ms safety-window tails and could turn
# an identical stationary hold into a fatal task error.
DEFAULT_HOLD_FEEDBACK_MAX_AGE_S = 0.040

# Offline compatibility profile from the smoothed 20260819--20260915 training
# dataset: the larger fitted active-motion P95 from action/feedback per joint.
# These are NOT hardware safety limits.  The operator-configured scalar limits
# remain hard upper bounds, so selecting this profile cannot silently increase
# the currently approved 0.6 rad/s and 2.0 rad/s^2 caps.
DATASET_V30_ARM_VELOCITY = np.asarray(
    (0.911, 0.886, 1.282, 1.052, 0.616, 0.600, 0.681,
     0.600, 0.600, 0.787, 1.044, 0.837, 0.600, 0.646), dtype=np.float32
)
DATASET_V30_ARM_ACCELERATION = np.asarray(
    (2.596, 2.000, 3.393, 3.267, 2.502, 2.000, 2.752,
     2.055, 2.000, 2.493, 3.445, 3.242, 2.485, 2.548), dtype=np.float32
)

# Fitted active-motion P99 envelope from the same action/feedback dataset,
# multiplied by 1.1.  This is an explicitly experimental compatibility
# profile, not a vendor-certified hardware limit.  Scalar limits remain hard
# upper bounds and therefore prevent merely selecting this profile from
# silently increasing command authority.
DATASET_V30_P99_ARM_VELOCITY = np.asarray(
    (1.607, 1.311, 2.185, 1.714, 1.101, 0.774, 1.201,
     1.083, 0.600, 1.336, 1.743, 1.625, 1.079, 1.191), dtype=np.float32
)
DATASET_V30_P99_ARM_ACCELERATION = np.asarray(
    (4.746, 2.859, 6.140, 6.065, 4.725, 3.614, 5.090,
     3.972, 2.000, 4.718, 6.309, 6.459, 5.035, 5.106), dtype=np.float32
)

# The legacy uniform-control URDF declared 3.1416 rad/s for every arm joint.
# The current crsB canonical/safety URDF instead has per-joint velocity values
# and still declares no acceleration limits.  Keep this array for the named
# compatibility profile; do not describe it as dynamically parsed from the
# current URDF. GDK position limits remain enforced independently.
URDF_ARM_VELOCITY = np.full(len(ARM_CANONICAL_INDICES), 3.1416, dtype=np.float32)


def arm_velocity_limits_from_urdf(urdf_path: Path) -> np.ndarray:
    """Read the fourteen GDK arm joint limits from the active URDF, in action order."""
    try:
        root = ElementTree.parse(urdf_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise ValueError(f"Cannot read arm velocity limits from URDF {urdf_path}") from error
    if root.tag != "robot":
        raise ValueError(f"URDF root must be <robot>: {urdf_path}")
    expected = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    found: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in expected:
            continue
        if name in found:
            raise ValueError(f"Duplicate arm joint in URDF: {name}")
        if joint.get("type") not in {"revolute", "continuous"}:
            raise ValueError(f"Arm joint is not revolute in URDF: {name}")
        limit = joint.find("limit")
        try:
            velocity = float(limit.get("velocity")) if limit is not None else math.nan
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid arm velocity in URDF: {name}") from error
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError(f"Invalid arm velocity in URDF: {name}")
        found[name] = velocity
    missing = [name for name in expected if name not in found]
    if missing:
        raise ValueError(f"Missing arm velocity limits in URDF: {missing}")
    return np.asarray([found[name] for name in expected], dtype=np.float32)


class SafetyDispatchExpiredError(RuntimeError):
    """A checked action expired before it could be dispatched; do not send it."""


class ServoInterfaceError(RuntimeError):
    """Servo invocation is ambiguous or incompatible; never retry as a command."""


@dataclass(frozen=True)
class DeliveryEvidence:
    measured_timestamp_ns: int
    measured_state: np.ndarray
    adopted_action: np.ndarray
    command_accepted: bool


@dataclass
class AdapterConfig:
    hardware_mode: str = "dry-run"
    initialization_scope: str = "24d"
    effectors_mode: str = "required"
    effector_profile: str = "dahuan"
    effector_control_mode: str = "software-trajectory"
    fixed_left_effector_state: float = 0.0
    fixed_right_effector_state: float = 0.0
    camera_timeout_ms: int = 1000
    camera_sync_tolerance_ms: float = 5.0
    camera_sync_timeout_ms: float = 40.0
    camera_sync_poll_ms: float = 2.0
    dry_camera_width: int = 640
    dry_camera_height: int = 400
    resize_wrist_images: bool = True
    parallel_wrist_images: bool = False
    wrist_image_width: int = 640
    wrist_image_height: int = 400
    wrist_jpeg_quality: int = 85
    startup_wait_s: float = 2.0
    initialization_attempts: int = 3
    initialization_retry_delay_s: float = 0.2
    stationary_hold_feedback_max_age_s: float = DEFAULT_HOLD_FEEDBACK_MAX_AGE_S
    initialization_settle_s: float = 1.0
    servo_control_period_s: float = 1.0 / 30.0
    enable_low_latency: bool = True
    left_effector_name: str | None = None
    right_effector_name: str | None = None
    max_relative_target: float | None = 0.05
    max_gripper_relative_target: float | None = None
    relative_target_mode: str = "warn"
    max_arm_velocity: float = 0.6
    max_arm_acceleration: float = 2.0
    arm_limit_profile: str = "uniform"
    # Opt-in numeric experiment; production behavior remains legacy by default.
    arm_tracking_mode: str = "legacy"
    arm_tracking_omega: float = 12.0
    max_effector_velocity: float | None = None
    max_effector_acceleration: float | None = None
    max_head_waist_velocity: float = 0.3
    max_head_waist_acceleration: float = 1.0
    gdk_position_limit_margin: float = 0.01
    max_consecutive_gdk_rejections: int = 3
    safety_config_path: Path | None = None
    safety_log_path: Path | None = None
    safety_enforcement_enabled: bool = True
    safety_guard_path_samples: int = 2
    safety_check_fps: int = 100
    async_safety_enabled: bool = False
    certified_braking_enabled: bool = False
    rolling_safety_enabled: bool = False
    rolling_legacy_geometry: bool = False
    max_safety_guard_latency_ms: float = 8.0
    max_consecutive_safety_deadline_misses: int = 3
    safety_log_queue_size: int = 2048
    detailed_safety_timing: bool = False
    platform_telemetry_enabled: bool = False
    platform_telemetry_fps: float = 20.0
    noncontrol_joint_state_grace_s: float = 0.2


@dataclass(frozen=True)
class ObservationSnapshot:
    state: np.ndarray
    images: dict[str, dict[str, Any]]
    state_capture_wall_time_ns: int
    camera_capture_start_wall_time_ns: int
    camera_capture_wall_time_ns: int
    camera_capture_duration_ms: float
    camera_frame_wall_time_ns: dict[str, int]


def _passthrough_image_timing(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", b"")
    return {
        "resized": False,
        "source_width": int(payload.get("width", 0)),
        "source_height": int(payload.get("height", 0)),
        "output_width": int(payload.get("width", 0)),
        "output_height": int(payload.get("height", 0)),
        "source_bytes": len(data),
        "output_bytes": len(data),
        "decode_ms": 0.0,
        "resize_ms": 0.0,
        "encode_ms": 0.0,
        "total_ms": 0.0,
    }


def resize_image_payload_for_transport(
    payload: dict[str, Any],
    *,
    width: int,
    height: int,
    jpeg_quality: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode, directly resize, and JPEG-encode one GDK image payload.

    Direct resize intentionally matches the G2A dataset capture path, which
    used ``cv2.resize(frame, (camera_width, camera_height))`` without padding.
    Pillow is used here because the robot's OpenCV binary isn't compatible
    with its NumPy 2 runtime.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Transport image dimensions must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")

    source_width = int(payload.get("width", 0))
    source_height = int(payload.get("height", 0))
    source_data = payload.get("data", b"")
    if source_width <= 0 or source_height <= 0 or not isinstance(source_data, bytes):
        raise RuntimeError("Malformed GDK image payload cannot be resized")
    if (source_width, source_height) == (width, height) and payload.get("encoding") == "jpeg":
        return dict(payload), _passthrough_image_timing(payload)

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Wrist image resize needs Pillow in the robot Python environment; "
            "install Pillow or pass --no-resize-wrist-images"
        ) from error

    total_started = time.perf_counter()
    decode_started = time.perf_counter()
    encoding = str(payload.get("encoding", "")).lower()
    if encoding == "jpeg":
        with Image.open(io.BytesIO(source_data)) as encoded_image:
            rgb_image = encoded_image.convert("RGB")
            rgb_image.load()
    elif encoding == "raw":
        color_format = str(payload.get("color_format", "")).lower()
        raw = np.frombuffer(source_data, dtype=np.uint8).reshape(source_height, source_width, 3)
        if color_format == "bgr":
            raw = raw[..., ::-1]
        elif color_format != "rgb":
            raise RuntimeError(f"Unsupported raw image color format: {color_format!r}")
        rgb_image = Image.fromarray(np.ascontiguousarray(raw), mode="RGB")
    else:
        raise RuntimeError(f"Unsupported image encoding for resize: {encoding!r}")
    decode_finished = time.perf_counter()

    resize_started = time.perf_counter()
    # Dataset collection used cv2.resize's default bilinear interpolation.
    # Pillow BILINEAR is both closer to that training path and substantially
    # cheaper than LANCZOS on the robot CPU.
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = rgb_image.resize((width, height), resample=resampling)
    resize_finished = time.perf_counter()

    encode_started = time.perf_counter()
    output = io.BytesIO()
    resized.save(output, format="JPEG", quality=jpeg_quality, optimize=False)
    encoded_data = output.getvalue()
    encode_finished = time.perf_counter()
    result = {
        "__g2a_encoded_image__": True,
        "encoding": "jpeg",
        "color_format": "encoded",
        "width": width,
        "height": height,
        "data": encoded_data,
    }
    timing = {
        "resized": True,
        "source_width": source_width,
        "source_height": source_height,
        "output_width": width,
        "output_height": height,
        "source_bytes": len(source_data),
        "output_bytes": len(encoded_data),
        "decode_ms": (decode_finished - decode_started) * 1000,
        "resize_ms": (resize_finished - resize_started) * 1000,
        "encode_ms": (encode_finished - encode_started) * 1000,
        "total_ms": (encode_finished - total_started) * 1000,
    }
    return result, timing


class G2AAdapter:
    """Expose canonical 24D state/actions and encoded camera payloads."""

    def __init__(self, config: AdapterConfig) -> None:
        if config.effectors_mode not in {"required", "absent_fixed"}:
            raise ValueError(f"Unsupported effectors mode: {config.effectors_mode!r}")
        if config.effectors_mode == "absent_fixed" and config.hardware_mode == "execute":
            raise ValueError(
                "effectors_mode=absent_fixed is observation-only and cannot be used with execute"
            )
        if config.camera_timeout_ms <= 0:
            raise ValueError("camera_timeout_ms must be positive")
        if not np.isfinite(config.camera_sync_tolerance_ms) or config.camera_sync_tolerance_ms < 0:
            raise ValueError("camera_sync_tolerance_ms must be finite and non-negative")
        if not np.isfinite(config.camera_sync_timeout_ms) or config.camera_sync_timeout_ms < 0:
            raise ValueError("camera_sync_timeout_ms must be finite and non-negative")
        if not np.isfinite(config.camera_sync_poll_ms) or config.camera_sync_poll_ms < 0:
            raise ValueError("camera_sync_poll_ms must be finite and non-negative")
        if config.safety_guard_path_samples < 1:
            raise ValueError("safety_guard_path_samples must be positive")
        if config.safety_config_path is not None and not math.isclose(
            config.servo_control_period_s,
            0.01,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("workspace safety currently requires a 100 Hz servo period")
        if not np.isfinite(config.max_safety_guard_latency_ms) or config.max_safety_guard_latency_ms <= 0:
            raise ValueError("max_safety_guard_latency_ms must be positive and finite")
        if config.max_consecutive_safety_deadline_misses < 1:
            raise ValueError("max_consecutive_safety_deadline_misses must be positive")
        if config.safety_log_queue_size < 1:
            raise ValueError("safety_log_queue_size must be positive")
        if config.safety_check_fps not in (50, 100):
            raise ValueError("safety_check_fps must be 50 or 100")
        if config.async_safety_enabled and config.safety_check_fps != 50:
            raise ValueError("asynchronous safety requires 50 Hz detection")
        if config.rolling_safety_enabled and config.rolling_legacy_geometry:
            if not config.async_safety_enabled or config.safety_config_path is None:
                raise ValueError("Legacy rolling geometry requires asynchronous surface safety")
            if config.certified_braking_enabled:
                raise ValueError("Legacy rolling geometry does not use certified-braking admission")
        if config.rolling_safety_enabled and not config.rolling_legacy_geometry and not config.certified_braking_enabled:
            raise ValueError("Rolling safety requires certified braking")
        if config.certified_braking_enabled:
            if not config.async_safety_enabled or config.safety_config_path is None:
                raise ValueError("Certified braking requires asynchronous surface safety")
            if config.hardware_mode == "execute":
                raise ValueError("Certified braking candidate is read-only/dry-run only until robot performance validation")
        if (
            not np.isfinite(config.platform_telemetry_fps)
            or not 0 < config.platform_telemetry_fps <= 20
        ):
            raise ValueError("platform_telemetry_fps must be finite and in (0, 20]")
        raw_fixed_effectors = np.asarray(
            [config.fixed_left_effector_state, config.fixed_right_effector_state],
            dtype=np.float64,
        )
        self.effector_profile: EffectorProfile = get_effector_profile(
            config.effector_profile
        )
        if config.max_gripper_relative_target is None:
            config.max_gripper_relative_target = (
                self.effector_profile.default_relative_target_warning
            )
        if config.max_effector_velocity is None:
            config.max_effector_velocity = self.effector_profile.default_max_velocity
        if config.max_effector_acceleration is None:
            config.max_effector_acceleration = self.effector_profile.default_max_acceleration
        gripper_min, gripper_max = self.effector_profile.hardware_state_range
        if not np.all(np.isfinite(raw_fixed_effectors)) or np.any(
            (raw_fixed_effectors < gripper_min - 1e-6)
            | (raw_fixed_effectors > gripper_max + 1e-6)
        ):
            raise ValueError(
                "fixed effector states must be finite and within the selected profile limits"
            )
        if config.stationary_hold_feedback_max_age_s not in {0.04, 0.1}:
            raise ValueError("hold feedback age must be 0.04 or 0.1 seconds")
        self.config = config
        self._fixed_effector_state = np.clip(
            raw_fixed_effectors, gripper_min, gripper_max
        ).astype(np.float32)
        self._gdk: Any = None
        self._robot: Any = None
        self._camera: Any = None
        self._connected = False
        self._gdk_initialized = False
        self._execution_prepared = False
        self._process_lock = RobotProcessLock()
        self._last_state: np.ndarray | None = None
        self._last_successful_action: np.ndarray | None = None
        self._last_command_accepted: bool | None = None
        self._servo_signature_robot: Any = None
        self._servo_uses_latency_keyword = True
        self._consecutive_gdk_rejections = 0
        self._consecutive_safety_deadline_misses = 0
        self._last_servo_dispatch: float | None = None
        self._command_safety_started: float | None = None
        self._certified_step: dict[str, Any] | None = None
        self._certified_dispatch = False
        # Per-thread diagnostics: observation and servo reads must not replace
        # one another's timing records. Never used as cached safety feedback.
        self._state_read_profiler = threading.local()
        self._dry_state = np.zeros(ACTION_DIM, dtype=np.float32)
        self._effector_name_overrides = {
            "left": config.left_effector_name,
            "right": config.right_effector_name,
        }
        self._effector_names = {
            "left": config.left_effector_name or DEFAULT_EFFECTOR_JOINT_NAMES[0],
            "right": config.right_effector_name or DEFAULT_EFFECTOR_JOINT_NAMES[1],
        }
        self._warning_last_time: dict[str, float] = {}
        self._warning_suppressed: dict[str, int] = {}
        self._body_position_cache: dict[str, tuple[float, float]] = {}
        self._body_position_cache_lock = threading.Lock()
        self._last_image_transport_timing: dict[str, dict[str, Any]] = {}
        self._last_camera_capture_timing: dict[str, float] = {}
        self._camera_wall_to_vendor_offset_ns: int | None = None
        self._camera_clock_offset_frozen = False
        self._wrist_image_executor: ThreadPoolExecutor | None = None
        self._last_feedback_state: np.ndarray | None = None
        self._last_feedback_monotonic: float | None = None
        # Diagnostic only: raw motor velocity from the same GDK joint-state
        # read that supplies position feedback. It never authorizes motion.
        self._last_motor_velocity_snapshot: tuple[float, tuple[float | None, ...]] | None = None
        # This is an in-process latest-value mailbox, not a network queue.  The
        # 100 Hz servo path replaces one small snapshot at most 20 times/s;
        # HTTP serialization and upload are owned by a separate publisher
        # thread so visualization can never stall physical command delivery.
        self._control_telemetry_sequence = 0
        self._next_control_telemetry_monotonic = 0.0
        self._next_visual_feedback_monotonic = 0.0
        self._latest_control_telemetry: dict[str, Any] | None = None
        if config.effector_control_mode not in {"software-trajectory", "device-position"}:
            raise ValueError("Invalid effector_control_mode")
        device_position = config.effector_control_mode == "device-position"
        if device_position and (self.effector_profile.name != "dahuan" or config.effectors_mode != "required"):
            raise ValueError("Device-position control requires the Dahuan effector profile")
        # Do not compile the Dahuan gripper's complete 0--25 mm opening range into
        # the collision geometry.  That envelope is intentionally conservative
        # and can reject a valid grasp near the table.  The FK geometry now
        # follows only the measured opening supplied in the 24D feedback. Arm
        # candidates are checked without resizing the fingers to a future
        # target opening; the geometry updates on the next measured sample.
        self._effector_envelope_ranges = None
        self._safety_layer: ActionSafetyLayer | None = (
            load_safety_layer(
                config.safety_config_path,
                allow_read_only_ready=config.hardware_mode == "read-only",
                effector_envelope_ranges=self._effector_envelope_ranges,
            )
            if config.safety_config_path is not None
            else None
        )
        if config.safety_config_path is not None:
            try:
                safety_document = json.loads(
                    config.safety_config_path.read_text(encoding="utf-8")
                )
                declared_profile = str(
                    safety_document.get("robot_model", {}).get("effector_profile", "")
                )
            except (OSError, ValueError, TypeError, AttributeError) as error:
                raise ValueError("Could not inspect safety effector profile") from error
            if declared_profile and declared_profile != self.effector_profile.name:
                raise ValueError(
                    "Safety/robot effector profile mismatch: "
                    f"safety={declared_profile}, robot={self.effector_profile.name}"
                )
        if (
            self._safety_layer is not None
            and self.effector_profile.collision_geometry_status == "provisional"
        ):
            logger.warning(
                "PROVISIONAL COLLISION GEOMETRY: effector_profile=%s temporarily uses %s; "
                "this is not final Dahuan physical-safety validation",
                self.effector_profile.name,
                self.effector_profile.collision_geometry,
            )
        if self._safety_layer is not None:
            expected_passthrough = set(EFFECTOR_CANONICAL_INDICES)
            if set(self._safety_layer.passthrough_indices) != expected_passthrough:
                raise ValueError(
                    "G2A safety passthrough_indices must be exactly the two gripper indices "
                    f"{sorted(expected_passthrough)}"
                )
        if self._safety_layer is not None and config.safety_log_path is None:
            raise ValueError("safety_log_path is required when safety_config_path is set")
        guard_layer: ActionSafetyLayer | None = None
        if self._safety_layer is not None:
            guard_layer = self._safety_layer.realtime_guard(
                path_samples=config.safety_guard_path_samples
            )
        if self._safety_layer is not None:
            logger.warning(
                "SURFACE SAFETY ENABLED mode=%s config=%s log=%s surfaces=%s "
                "collision_points=%d",
                config.hardware_mode,
                config.safety_config_path,
                config.safety_log_path,
                [
                    {
                        "name": surface.name,
                        "height": surface.height,
                        "hard_margin": surface.hard_margin,
                        "warning_margin": surface.warning_margin,
                        "xy_bounds": surface.xy_bounds,
                    }
                    for surface in self._safety_layer.surfaces.surfaces
                ],
                len(self._safety_layer.geometry.points),
            )
            if config.rolling_safety_enabled and config.rolling_legacy_geometry:
                logger.warning(
                    "FIXED COLLISION ACTION STRIDE=2: segment boundaries retained; "
                    "no midpoint interpolation or distance-adaptive densification; "
                    "skipped waypoints are not individually collision-checked")
        max_velocity = np.empty(ACTION_DIM, dtype=np.float32)
        max_acceleration = np.empty(ACTION_DIM, dtype=np.float32)
        if config.arm_limit_profile not in {
            "uniform", "dataset-v30", "dataset-v30-p99-margin",
            "urdf-velocity-p99-acceleration", "urdf-velocity-uniform-acceleration",
            "urdf-velocity-only",
        }:
            raise ValueError("Invalid arm_limit_profile")
        if config.arm_limit_profile in {
            "urdf-velocity-uniform-acceleration", "urdf-velocity-only",
        }:
            urdf_path = (
                getattr(self._safety_layer.fk, "urdf_path", None)
                if self._safety_layer is not None else None
            )
            if self._safety_layer is not None and urdf_path is None:
                raise ValueError("URDF arm velocity profile requires URDF collision kinematics")
            if urdf_path is None:
                urdf_path = Path(__file__).resolve().parent / "assets/agibot_g2/G2_dahuan_safety.urdf"
            arm_urdf_velocity = arm_velocity_limits_from_urdf(Path(urdf_path))
            max_velocity[list(ARM_CANONICAL_INDICES)] = np.minimum(
                arm_urdf_velocity, config.max_arm_velocity
            )
            max_acceleration[list(ARM_CANONICAL_INDICES)] = config.max_arm_acceleration
            logger.info(
                "Arm limits from URDF %s: velocity=%s rad/s, acceleration=%s",
                urdf_path, arm_urdf_velocity.tolist(),
                ("disabled for normal arm tracking (finite braking assumption %.3f rad/s^2)"
                 % config.max_arm_acceleration if config.arm_limit_profile == "urdf-velocity-only"
                 else "%.3f rad/s^2" % config.max_arm_acceleration),
            )
        elif config.arm_limit_profile == "urdf-velocity-p99-acceleration":
            max_velocity[list(ARM_CANONICAL_INDICES)] = np.minimum(
                URDF_ARM_VELOCITY, config.max_arm_velocity
            )
            max_acceleration[list(ARM_CANONICAL_INDICES)] = np.minimum(
                DATASET_V30_P99_ARM_ACCELERATION, config.max_arm_acceleration
            )
        elif config.arm_limit_profile == "dataset-v30-p99-margin":
            max_velocity[list(ARM_CANONICAL_INDICES)] = np.minimum(
                DATASET_V30_P99_ARM_VELOCITY, config.max_arm_velocity
            )
            max_acceleration[list(ARM_CANONICAL_INDICES)] = np.minimum(
                DATASET_V30_P99_ARM_ACCELERATION, config.max_arm_acceleration
            )
        elif config.arm_limit_profile == "dataset-v30":
            max_velocity[list(ARM_CANONICAL_INDICES)] = np.minimum(
                DATASET_V30_ARM_VELOCITY, config.max_arm_velocity
            )
            max_acceleration[list(ARM_CANONICAL_INDICES)] = np.minimum(
                DATASET_V30_ARM_ACCELERATION, config.max_arm_acceleration
            )
        else:
            max_velocity[list(ARM_CANONICAL_INDICES)] = config.max_arm_velocity
            max_acceleration[list(ARM_CANONICAL_INDICES)] = config.max_arm_acceleration
        assert config.max_effector_velocity is not None
        assert config.max_effector_acceleration is not None
        max_velocity[list(EFFECTOR_CANONICAL_INDICES)] = config.max_effector_velocity
        max_acceleration[list(EFFECTOR_CANONICAL_INDICES)] = config.max_effector_acceleration
        head_waist = HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES
        max_velocity[list(head_waist)] = config.max_head_waist_velocity
        max_acceleration[list(head_waist)] = config.max_head_waist_acceleration
        if config.arm_tracking_mode not in {"legacy", "damped"}:
            raise ValueError("Invalid arm_tracking_mode")
        self._kinematic_limiter = JointKinematicLimiter(
            sample_hz=1.0 / config.servo_control_period_s,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            position_target_indices=tuple(EFFECTOR_CANONICAL_INDICES) if device_position else (),
            unlimited_acceleration_indices=(tuple(ARM_CANONICAL_INDICES)
                                            if config.arm_limit_profile == "urdf-velocity-only" else ()),
            tracking_damping_indices=(tuple(ARM_CANONICAL_INDICES)
                                      if config.arm_tracking_mode == "damped" else ()),
            tracking_damping_omega=config.arm_tracking_omega,
        )
        self._async_brake = None
        self._async_braking_active = False
        self._async_last_waypoint = None
        self.last_async_response = None
        # Threads start only after all pure configuration and limiter
        # construction succeeds, so a rejected AdapterConfig cannot leak a
        # background worker.
        self._safety_logger: SafetyJSONLLogger | None = (
            SafetyJSONLLogger(
                config.safety_log_path,
                queue_size=config.safety_log_queue_size,
            )
            if config.safety_log_path is not None
            else None
        )
        self._safety_controller: RealtimeSafetyController | None = None
        try:
            if guard_layer is not None and self._safety_layer is not None:
                self._safety_controller = RealtimeSafetyController(
                    guard_layer,
                    self._safety_layer,
                    # Projection relative to lagging feedback can jump behind
                    # the previous command and bypass the kinematic limiter.
                    # Production uses hold-or-allow, in all hardware modes.
                    allow_projected_actions=False,
                )
        except BaseException:
            if self._safety_logger is not None:
                self._safety_logger.close()
            raise
        self._safety_resources_closed = False
        self._async_worker = None
        self._async_generation = 0
        self._async_request_sequence = 0
        self._async_urgent = False
        self._async_pending = None
        self._async_ready_result = None
        self._async_steps = []
        self._async_hold = None
        self._async_collision_active = False
        self._async_collision_decision = None
        # Latest worker evaluation of the measured GDK pose.  The selected
        # hold/motion certificate can intentionally remain older while a
        # collision is held, so it must not also freeze UI collision distance.
        self._async_latest_live_decision = None
        self._async_latest_live_decision_at = None
        self._async_last_check_at = None
        self._async_started = None
        self._async_reanchor_wait_started = None
        self._async_ever_dispatched = False
        self._async_last_dispatch_at = None
        self._async_dispatch_action = None
        self._async_dispatch_is_hold = False
        self.last_command_pending = False
        self._async_hold_coefficients = (
            hold_displacement_coefficients(guard_layer) if guard_layer is not None else None
        )
        if (config.async_safety_enabled and config.safety_enforcement_enabled
                and self._safety_layer is not None):
            try:
                self._async_worker = AsyncSafetyWorker(
                    config.safety_config_path, config.safety_guard_path_samples, self.read_state,
                    self._audit_async_check,
                )
            except BaseException:
                self._close_safety_resources()
                raise

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_image_transport_timing(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._last_image_transport_timing.items()}

    @property
    def last_camera_capture_timing(self) -> dict[str, float]:
        return dict(self._last_camera_capture_timing)

    @property
    def camera_wall_to_vendor_offset_ns(self) -> int | None:
        return self._camera_wall_to_vendor_offset_ns

    def freeze_camera_clock_offset(self) -> int:
        offset = self._camera_wall_to_vendor_offset_ns
        if offset is None:
            raise RuntimeError("GDK camera clock has not been observed")
        self._camera_clock_offset_frozen = True
        return offset

    @property
    def latest_control_telemetry(self) -> dict[str, Any] | None:
        """Return the newest bounded-rate control snapshot for diagnostics.

        Assignment of the mailbox is atomic under CPython.  Copying happens
        only in the non-realtime publisher thread; the servo thread never
        performs network I/O and never waits for a telemetry consumer.
        """
        value = self._latest_control_telemetry
        if value is None:
            return None
        return {
            **value,
            "measured_state": list(value["measured_state"]),
            "model_candidate": list(value["model_candidate"]),
            "controller_candidate": list(value["controller_candidate"]),
            "adopted_action": list(value["adopted_action"]),
            "safety": dict(value["safety"]),
        }

    def publish_read_only_telemetry(self, measured: np.ndarray) -> None:
        """Publish observed pose for the web viewer without dispatching an action.

        This is deliberately limited to read-only/dry-run presentation.  It
        does not call ``send_action``, does not prepare execution, and does
        not submit work to the async safety/action dispatcher.
        """
        if self.config.hardware_mode == "execute":
            return
        observed = np.asarray(measured, dtype=np.float32).copy()
        self._capture_control_telemetry(
            measured=observed,
            model_candidate=observed,
            controller_candidate=observed,
            adopted_action=observed,
            safety_decision=None,
            safety_guard_latency_ms=None,
            collision_response="read_only_observation",
        )


    def _capture_control_telemetry(
        self,
        *,
        measured: np.ndarray | None,
        model_candidate: np.ndarray,
        controller_candidate: np.ndarray,
        adopted_action: np.ndarray,
        safety_decision: Any,
        safety_guard_latency_ms: float | None,
        collision_response: str,
        live_safety_decision: Any | None = None,
    ) -> None:
        # Local diagnostics use every tick; UI telemetry remains rate limited.
        self._last_controller_candidate = controller_candidate.copy()
        self._last_collision_response = collision_response
        now = time.monotonic()
        if now < self._next_control_telemetry_monotonic:
            return
        self._next_control_telemetry_monotonic = now + 0.05
        measured_value = adopted_action if measured is None else measured
        self._control_telemetry_sequence += 1
        # ``safety_decision`` describes the candidate/path currently being
        # admitted.  It must not be presented as the physical robot pose: a
        # held unsafe candidate can remain blocked after an operator has
        # already lifted the real arm away from the table.  The async worker
        # independently checks the measured feedback as ``live_decision``.
        # Use that for the red, current-pose alarm; retain candidate status as
        # an informational field so the UI can still explain a held action.
        candidate_decision = safety_decision
        physical_decision = (
            live_safety_decision
            if live_safety_decision is not None
            else candidate_decision
        )
        safety_available = physical_decision is not None
        limiting_surface = None
        current_hard_margin = None
        current_warning_margin = None
        work_surface_telemetry = None
        inter_arm_telemetry = None
        if safety_available and self._safety_layer is not None:
            limiting_surface = next(
                (
                    surface
                    for surface in self._safety_layer.surfaces.surfaces
                    if surface.name == physical_decision.limiting_surface
                ),
                None,
            )
            if limiting_surface is not None:
                current_hard_margin = limiting_surface.hard_margin
                current_warning_margin = limiting_surface.warning_margin
            elif (
                physical_decision.limiting_surface == "inter_arm"
                and self._safety_layer.inter_arm_collision is not None
            ):
                current_hard_margin = self._safety_layer.inter_arm_collision.hard_margin
                current_warning_margin = self._safety_layer.inter_arm_collision.warning_margin
            work_surface_name = str(
                getattr(physical_decision, "work_surface_name", "") or ""
            )
            work_surface = next(
                (
                    surface
                    for surface in self._safety_layer.surfaces.surfaces
                    if surface.name == work_surface_name
                ),
                self._safety_layer.surfaces.surfaces[0],
            )
            work_distance = getattr(physical_decision, "work_surface_distance", None)
            work_clearance = getattr(physical_decision, "work_surface_clearance", None)
            work_surface_telemetry = {
                "name": work_surface.name,
                "distance_m": work_distance,
                "clearance_m": work_clearance,
                "hard_margin_m": work_surface.hard_margin,
                "warning_margin_m": work_surface.warning_margin,
                "collision": bool(
                    work_clearance is not None
                    and work_clearance < -self._safety_layer.tolerance
                ),
                "warning": bool(
                    work_distance is not None
                    and work_surface.warning_margin is not None
                    and work_distance < work_surface.warning_margin
                ),
                "limiting_point": str(
                    getattr(physical_decision, "work_surface_point", "") or ""
                ),
            }
            if self._safety_layer.inter_arm_collision is not None:
                inter_arm = self._safety_layer.inter_arm_collision
                separation = getattr(physical_decision, "inter_arm_separation", None)
                clearance = getattr(physical_decision, "inter_arm_clearance", None)
                inter_arm_telemetry = {
                    "minimum_separation_m": separation,
                    "clearance_m": clearance,
                    "hard_margin_m": inter_arm.hard_margin,
                    "warning_margin_m": inter_arm.warning_margin,
                    "collision": bool(
                        clearance is not None
                        and clearance < -self._safety_layer.tolerance
                    ),
                    "warning": bool(
                        separation is not None
                        and separation < inter_arm.warning_margin
                    ),
                    "left_geometry": str(
                        getattr(physical_decision, "inter_arm_left", "") or ""
                    ),
                    "right_geometry": str(
                        getattr(physical_decision, "inter_arm_right", "") or ""
                    ),
                }
        self._latest_control_telemetry = {
            "sequence": self._control_telemetry_sequence,
            "captured_at": time.time(),
            "hardware_mode": self.config.hardware_mode,
            "measured_state": np.asarray(measured_value, dtype=np.float32).tolist(),
            "model_candidate": np.asarray(model_candidate, dtype=np.float32).tolist(),
            "controller_candidate": np.asarray(controller_candidate, dtype=np.float32).tolist(),
            "adopted_action": np.asarray(adopted_action, dtype=np.float32).tolist(),
            "command_delivery": {
                "gdk_accepted": self._last_command_accepted,
                "consecutive_rejections": self._consecutive_gdk_rejections,
            },
            "safety": {
                "geometry_mode": "measured-opening",
                "enabled": bool(
                    self._safety_layer is not None
                    and self.config.safety_enforcement_enabled
                ),
                "monitoring_enabled": self._safety_layer is not None,
                "enforcement_enabled": bool(
                    self._safety_layer is not None
                    and self.config.safety_enforcement_enabled
                ),
                "decision_available": safety_available,
                "collision_detected": bool(
                    safety_available and physical_decision.hard_limit_triggered
                ),
                "proximity_warning": bool(
                    safety_available
                    and physical_decision.soft_limit_triggered
                    and not physical_decision.hard_limit_triggered
                ),
                "intervention_active": bool(
                    candidate_decision is not None
                    and candidate_decision.hard_limit_triggered
                    and self.config.safety_enforcement_enabled
                    and not np.allclose(
                        candidate_decision.controller_target,
                        adopted_action,
                        rtol=0.0,
                        atol=1e-7,
                    )
                ),
                "reason": None if not safety_available else physical_decision.reason,
                "response": collision_response,
                "action_scale": None if not safety_available else physical_decision.action_scale,
                "current_z_min": None if not safety_available else physical_decision.current_z_min,
                "distance_to_surface": (
                    None if not safety_available else physical_decision.distance_to_surface
                ),
                "surface_height": (
                    None if limiting_surface is None else limiting_surface.height
                ),
                "hard_margin": (
                    current_hard_margin
                ),
                "warning_margin": (
                    current_warning_margin
                ),
                "work_surface": work_surface_telemetry,
                "inter_arm": inter_arm_telemetry,
                "predicted_z_min": (
                    None if not safety_available else physical_decision.predicted_z_min
                ),
                "safe_predicted_z_min": (
                    None if not safety_available else physical_decision.safe_predicted_z_min
                ),
                "limiting_surface": (
                    None if not safety_available else physical_decision.limiting_surface
                ),
                "limiting_point": (
                    None if not safety_available else physical_decision.limiting_point
                ),
                "candidate_collision_detected": bool(
                    candidate_decision is not None
                    and candidate_decision.hard_limit_triggered
                ),
                "candidate_proximity_warning": bool(
                    candidate_decision is not None
                    and candidate_decision.soft_limit_triggered
                    and not candidate_decision.hard_limit_triggered
                ),
                "candidate_reason": (
                    None if candidate_decision is None else candidate_decision.reason
                ),
                "candidate_distance_to_surface": (
                    None
                    if candidate_decision is None
                    else candidate_decision.distance_to_surface
                ),
                "guard_latency_ms": safety_guard_latency_ms,
                "guard_budget_ms": self.config.max_safety_guard_latency_ms,
                "check_fps": self.config.safety_check_fps,
                "guard_evaluated": not self._certified_dispatch,
                "deadline_misses": self._consecutive_safety_deadline_misses,
            },
        }

    def _read_visual_feedback_if_due(self, fallback: np.ndarray | None) -> np.ndarray | None:
        """Refresh read-only visualization feedback without touching execute timing.

        The physical execute path deliberately never enters this method.  In
        read-only mode no command is sent, so a bounded GDK state read can be
        performed at the platform telemetry cadence instead of incorrectly
        reusing the state captured by the much slower inference request loop.
        A display-only read failure is logged and the newest valid feedback is
        retained; it cannot create or modify a robot command.
        """

        if not (
            self.config.hardware_mode == "read-only"
            and self.config.platform_telemetry_enabled
        ):
            return fallback
        now = time.monotonic()
        if now < self._next_visual_feedback_monotonic:
            return fallback
        self._next_visual_feedback_monotonic = (
            now + 1.0 / self.config.platform_telemetry_fps
        )
        try:
            return self.read_state()
        except Exception as error:  # display sampling must never arm or stop control
            self._rate_limited_warning(
                "platform_visual_feedback",
                "Read-only platform visualization feedback unavailable; keeping the "
                "last valid measured state: %s",
                error,
            )
            return fallback

    def reset_command_reference(self, state: np.ndarray) -> None:
        """Initialize command, success and fault state from measured feedback."""
        self._certified_step = None
        self._invalidate_async_safety(clear_hold=True)
        reference = np.asarray(state, dtype=np.float32)
        if reference.shape != (ACTION_DIM,) or not np.isfinite(reference).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}D state, got {reference.shape}")
        self._kinematic_limiter.reset(reference)
        self._last_state = reference.copy()
        self._last_successful_action = reference.copy()
        self._consecutive_gdk_rejections = 0

    def reanchor_measured_hold(self, state: np.ndarray) -> None:
        """Re-anchor motion state without claiming GDK success or clearing faults."""
        self._certified_step = None
        self._invalidate_async_safety(clear_hold=True)
        reference = np.asarray(state, dtype=np.float32)
        if reference.shape != (ACTION_DIM,) or not np.isfinite(reference).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}D state, got {reference.shape}")
        self._kinematic_limiter.reset(reference)
        self._last_state = reference.copy()
        if self._async_worker is not None:
            self._async_urgent = True
            # No action is authorized by this timestamp.  It only gives the
            # worker a bounded interval to certify this newly measured,
            # stationary anchor before pause establishment fails closed.
            self._async_reanchor_wait_started = time.perf_counter()
            snapshot = getattr(self._state_read_profiler, "feedback_snapshot", None)
            publish = getattr(self._async_worker, "publish_feedback", None)
            if snapshot is not None and publish is not None and np.array_equal(snapshot[0], reference):
                publish(*snapshot)

    def reset_action_limiter(self, action: np.ndarray) -> None:
        """Hold without erasing GDK rejection history or claiming a new success."""
        if self._async_worker is not None and self._last_state is not None:
            # Repeated plan-gap holds are not a session reset. Cancel motion,
            # but keep a pending independent proof of the accepted hold target.
            # A different requested target must still be limited from that
            # accepted reference; do not pretend it was already sent.
            self._certified_step = None
            self._retire_async_motion()
            return
        step = self._certified_step
        if step is not None and (
            not np.array_equal(step["action"], action)
            or not np.array_equal(step["first"], action)
        ):
            self._certified_step = None
        self._kinematic_limiter.reset(action)
        if self._certified_step is not None:
            self._certified_step["limiter"] = self._kinematic_limiter.fork()

    def cancel_pending_trajectory(self) -> None:
        """Cancel the unsent half of a checked segment on an external hold."""
        self._certified_step = None

    def retire_unissued_motion(self, *, reason: str) -> None:
        """Fence all queued motion before changing a rollback endpoint/owner.

        Call only from the sole action writer. Already accepted GDK commands
        remain in the dispatch receipt history and cannot be withdrawn.
        """
        self._certified_step = None
        if self._async_worker is not None:
            self._retire_async_motion(reason=reason)

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("G2A adapter is already connected")
        if self.config.hardware_mode == "dry-run":
            self._connected = True
            self._kinematic_limiter.reset(self._dry_state)
            return

        self._process_lock.acquire()
        try:
            try:
                import agibot_gdk
            except ImportError as error:
                raise ImportError(
                    "agibot_gdk is unavailable; run inside the robot GDK environment"
                ) from error

            self._gdk = agibot_gdk
            result = self._gdk.gdk_init()
            if result != self._gdk.GDKRes.kSuccess:
                raise RuntimeError(f"GDK initialization failed: {result!r}")
            self._gdk_initialized = True
            self._robot = self._gdk.Robot()
            self._camera = self._gdk.Camera()
            if self.config.startup_wait_s:
                time.sleep(self.config.startup_wait_s)
            self._connected = True
            if self.config.effectors_mode == "absent_fixed":
                logger.warning(
                    "EFFECTORS ABSENT: synthesizing observation-only left/right states %s; "
                    "GDK get_end_state() will not be called and physical execution is disabled",
                    self._fixed_effector_state.tolist(),
                )
            self._last_state = self.read_state()
            self._last_successful_action = self._last_state.copy()
            self._kinematic_limiter.reset(self._last_state)
        except Exception:
            self._connected = False
            self._robot = None
            self._camera = None
            try:
                if self._gdk_initialized:
                    self._gdk.gdk_release()
            finally:
                self._gdk_initialized = False
                self._process_lock.release()
            raise

    def prepare_execution(self) -> None:
        """Enter GDK control mode only after policy-side preflight has passed.

        ``connect()`` intentionally remains observation-only, including when
        the requested hardware mode is ``execute``.  The runner calls this
        explicit second phase after protocol validation, model warmup, camera
        checks, and execution-latency preflight have all succeeded.
        """
        self._require_connected()
        if self.config.hardware_mode != "execute":
            raise RuntimeError("Execution preparation requires hardware_mode=execute")
        if self._execution_prepared:
            raise RuntimeError("GDK physical execution is already prepared")
        self._configure_current_pose()
        measured = self.read_state()
        self.reset_command_reference(measured)
        self._execution_prepared = True
        logger.warning("GDK PHYSICAL EXECUTION PREPARED after all read-only preflight checks")

    def read_state(self) -> np.ndarray:
        read_started = time.perf_counter()
        self._require_connected()
        if self.config.hardware_mode == "dry-run":
            state = self._dry_state.copy()
        else:
            positions = self._read_body_positions(read_started_monotonic=read_started)
            self._mark_state_read_stage("effector_setup")
            if self.config.effectors_mode == "absent_fixed":
                left, right = self._fixed_effector_state
            else:
                self._mark_state_read_stage("end_gdk")
                end_state = self._robot.get_end_state()
                self._mark_state_read_stage("end_decode")
                left = self._read_effector("left", end_state)
                right = self._read_effector("right", end_state)
                lower, upper = self.effector_profile.hardware_state_range
                if not all(lower - 1e-4 <= value <= upper + 1e-4 for value in (left, right)):
                    raise RuntimeError(
                        f"GDK effector state is outside {self.effector_profile.name} "
                        f"range [{lower}, {upper}]: left={left}, right={right}"
                    )
                left = float(min(upper, max(lower, left)))
                right = float(min(upper, max(lower, right)))
            self._mark_state_read_stage("state_pack")
            state = np.asarray(
                [positions[name] for name in LEFT_ARM_JOINTS]
                + [left]
                + [positions[name] for name in RIGHT_ARM_JOINTS]
                + [right]
                + [positions[name] for name in HEAD_JOINTS]
                + [positions[name] for name in WAIST_JOINTS],
                dtype=np.float32,
            )
        self._last_feedback_state = state.copy()
        self._last_feedback_monotonic = time.monotonic()
        self._last_feedback_timestamp_ns = time.time_ns()
        self._state_read_profiler.feedback_snapshot = (state.copy(), read_started)
        return state

    def _mark_state_read_stage(self, name: str | None) -> None:
        """Measure only a protected read, without I/O or cross-thread locks."""
        profiler = getattr(self, "_state_read_profiler", None)
        samples = getattr(profiler, "samples", None)
        if samples is None:
            return
        wall, cpu = time.perf_counter(), time.thread_time()
        previous = getattr(profiler, "current", None)
        if previous is not None:
            label, started_wall, started_cpu = previous
            samples[label] = {
                "wall_ms": (wall - started_wall) * 1000,
                "cpu_ms": (cpu - started_cpu) * 1000,
            }
        profiler.current = None if name is None else (name, wall, cpu)

    def capture_observation(self) -> ObservationSnapshot:
        """Capture cameras on demand and then read the closest available state."""
        self._require_connected()
        start = time.perf_counter()
        start_wall_ns = time.time_ns()
        images, frame_times = self.capture_images()
        camera_end_ns = time.time_ns()
        state = self.read_state()
        state_time_ns = time.time_ns()
        return ObservationSnapshot(
            state=state,
            images=images,
            state_capture_wall_time_ns=state_time_ns,
            camera_capture_start_wall_time_ns=start_wall_ns,
            camera_capture_wall_time_ns=camera_end_ns,
            camera_capture_duration_ms=(time.perf_counter() - start) * 1000,
            camera_frame_wall_time_ns=frame_times,
        )

    def capture_images(
        self, *, before_processing: Callable[[], None] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """Capture encoded images without touching the GDK Robot object."""
        self._require_connected()
        if self.config.hardware_mode == "dry-run":
            width = self.config.dry_camera_width
            height = self.config.dry_camera_height
            raw = bytes(width * height * 3)
            images = {
                key: {
                    "__g2a_encoded_image__": True,
                    "encoding": "raw",
                    "color_format": "rgb",
                    "width": width,
                    "height": height,
                    "data": raw,
                }
                for key in CAMERA_KEYS
            }
            now_ns = time.time_ns()
            frame_times = dict.fromkeys(CAMERA_KEYS, now_ns)
            self._camera_wall_to_vendor_offset_ns = 0
            if before_processing is not None:
                before_processing()
            self._last_image_transport_timing = {
                key: {
                    "resized": False,
                    "source_width": width,
                    "source_height": height,
                    "output_width": width,
                    "output_height": height,
                    "source_bytes": len(raw),
                    "output_bytes": len(raw),
                    "decode_ms": 0.0,
                    "resize_ms": 0.0,
                    "encode_ms": 0.0,
                    "total_ms": 0.0,
                }
                for key in CAMERA_KEYS
            }
            return images, frame_times
        return self._capture_encoded_images(before_processing=before_processing)

    def send_action(
        self,
        action: np.ndarray,
        *,
        safety_context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        control_started = time.perf_counter()
        # A safety submission can return without dispatching anything.  Keep
        # a per-call boundary so rollback consumers never mistake an earlier
        # accepted command for this call's acknowledgement.
        self._send_dispatch_start_sequence = getattr(self, "_dispatch_sequence", 0)
        # Synchronous per-command feedback; telemetry is rate limited.
        self.last_action_intervened = False
        self._last_command_accepted = None
        self._last_controller_candidate = None
        self._last_dispatch_provenance = None
        self._last_collision_response = "pending"
        self._last_servo_call_ms = None
        self._last_servo_call_started = None
        self._command_safety_started = None
        self._certified_dispatch = False
        self._async_dispatch_action = None
        self._async_dispatch_is_hold = False
        self.last_command_pending = False
        self._require_connected()
        if self.config.hardware_mode == "execute" and not self._execution_prepared:
            raise RuntimeError(
                "Physical execution is not prepared; complete policy and latency preflight first"
            )
        target = np.asarray(action, dtype=np.float32)
        if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
            raise ValueError(f"Expected one finite 24D action, got {target.shape}")
        model_candidate = target.copy()
        if safety_context is not None and safety_context.get("raw_action") is not None:
            raw_model_candidate = np.asarray(
                safety_context["raw_action"], dtype=np.float32
            )
            if raw_model_candidate.shape == (ACTION_DIM,) and np.isfinite(
                raw_model_candidate
            ).all():
                model_candidate = raw_model_candidate.copy()
        requested_target = target.copy()
        target = self._apply_action_checks(target)
        if (safety_context is not None and safety_context.get("preplanned_trajectory") is True
                and not np.allclose(target, requested_target, rtol=0.0, atol=1e-7)):
            raise ValueError("Preplanned rollback target exceeded hardware input limits")
        if safety_context is not None and self._async_worker is None:
            # The asynchronous branch replaces this with the selected
            # window's source. Synchronous dispatch belongs to this call.
            self._last_dispatch_provenance = {
                "chunk_id": safety_context.get("chunk_id"),
                "candidate_action_index": safety_context.get("action_index"),
                "policy_action_index": safety_context.get("action_index"),
                "safety_target_24d": (
                    None if safety_context.get("controller_target") is None
                    else np.asarray(safety_context["controller_target"], dtype=np.float32).copy()
                ),
                "reference_preview_index": 0,
                "holding_previous_target": False,
            }
        if self._async_worker is not None:
            return self._send_async_action(target, model_candidate, safety_context, control_started)
        cached = self._consume_certified_step(model_candidate, safety_context, control_started)
        if cached is not None:
            return cached
        limited = (
            self._kinematic_limiter.adopt_preplanned(target)
            if safety_context is not None and safety_context.get("preplanned_trajectory") is True
            else self._kinematic_limiter.update(target)
        )
        if not np.allclose(limited, target, rtol=0.0, atol=1e-7):
            difference = np.abs(target - limited)
            self._rate_limited_action_diagnostic(
                "kinematic_limit",
                "Applied final GDK velocity/acceleration limit: indices=%s max_target_error=%.4f",
                np.flatnonzero(difference > 1e-7).tolist(),
                float(np.max(difference)),
            )
        # The discrete acceleration limiter may overshoot a target by a small
        # bounded amount while braking.  Re-apply non-optional hardware ranges
        # to the value that is actually sent, then re-anchor the limiter so it
        # cannot keep integrating outside the valid range on the next tick.
        final_limited = self._apply_absolute_position_limits(limited, warning_suffix="final")
        if not np.array_equal(final_limited, limited):
            self._kinematic_limiter.reconcile_position_clip(final_limited)
        limited = final_limited
        controller_candidate = limited.copy()
        next_limiter = None
        next_target = None
        if (
            self._safety_layer is not None
            and self.config.safety_enforcement_enabled
            and self.config.safety_check_fps == 50
        ):
            next_limiter = self._kinematic_limiter.fork()
            preview = next_limiter.update(target)
            next_target = self._apply_absolute_position_limits(preview, warning_suffix="preview")
            if not np.array_equal(next_target, preview):
                next_limiter.reconcile_position_clip(next_target)
        measured = (
            self._last_feedback_state
            if self._safety_layer is not None
            else self._read_visual_feedback_if_due(self._last_feedback_state)
        )
        safety_decision = None
        safety_guard_latency_ms = None
        collision_response = "not_applicable"
        if self._safety_layer is not None:
            # A command echo is not sufficient for collision protection. Read
            # real feedback immediately before FK/projection on every protected
            # servo tick. Any GDK/FK/configuration failure propagates and stops
            # physical execution (fail closed).
            if self.config.hardware_mode == "execute":
                # Spend the minimum-spacing wait BEFORE reading fresh feedback
                # and checking the segment. The final dispatch still rechecks
                # spacing and decision age; no deadline is extended or reset.
                self._wait_servo_slot(
                    lead_s=min(
                        self.config.servo_control_period_s,
                        self.config.max_safety_guard_latency_ms / 1000.0,
                    ) * 0.75
                )
            safety_started = time.perf_counter()
            safety_cpu_started = time.thread_time()
            self._command_safety_started = safety_started
            if self.config.detailed_safety_timing:
                self._state_read_profiler.samples = {}
                self._state_read_profiler.current = None
                self._mark_state_read_stage("read_setup")
                try:
                    measured = self.read_state()
                finally:
                    self._mark_state_read_stage(None)
                    state_read_stages = self._state_read_profiler.samples
                    self._state_read_profiler.samples = None
            else:
                measured = self.read_state()
            safety_state_read_ms = (time.perf_counter() - safety_started) * 1000
            if self.config.detailed_safety_timing:
                safety_state_cpu_ms = (time.thread_time() - safety_cpu_started) * 1000
            raw_action = None if safety_context is None else safety_context.get("raw_action")
            if next_target is not None:
                guard = self._safety_controller.guard_layer if self._safety_controller is not None else self._safety_layer
                safety_decision = guard.guard_trajectory(measured, (limited, next_target), raw_action=raw_action)
                if not safety_decision.hard_limit_triggered:
                    safety_decision = replace(safety_decision, safe_action=limited.copy())
            else:
                safety_decision = (
                    self._safety_controller.protect(
                        measured, limited, raw_action=raw_action,
                    )
                    if self._safety_controller is not None
                    else self._safety_layer.guard(
                        measured, limited, raw_action=raw_action,
                    )
                )
            if (
                self.config.hardware_mode == "read-only"
                and next_target is None
                and safety_decision.hard_limit_triggered
                and not bool(
                    getattr(
                        getattr(self._safety_layer, "geometry", None),
                        "horizontal_support_exact",
                        True,
                    )
                )
            ):
                # Legacy profiles use one conservative enclosing sphere per
                # collision frame in the realtime guard. Refine those read-only
                # alarms synchronously before publishing. Profiles with offline
                # convex-hull support vertices already use plane-exact geometry
                # in the realtime guard and deliberately skip this duplicate work.
                detailed_decision = self._safety_layer.project(
                    measured,
                    limited,
                    raw_action=raw_action,
                )
                conservative_safe_action = safety_decision.safe_action
                conservative_action_scale = safety_decision.action_scale
                safety_decision = replace(
                    detailed_decision,
                    safe_action=(
                        conservative_safe_action
                        if detailed_decision.hard_limit_triggered
                        else detailed_decision.safe_action
                    ),
                    action_scale=(
                        conservative_action_scale
                        if detailed_decision.hard_limit_triggered
                        else detailed_decision.action_scale
                    ),
                    reason=(
                        "read_only_detailed_geometry_unsafe"
                        if detailed_decision.hard_limit_triggered
                        else "read_only_detailed_geometry_safe"
                    ),
                    projection_latency_ms=(
                        safety_decision.projection_latency_ms
                        + detailed_decision.projection_latency_ms
                    ),
                )
            safe_limited = np.asarray(safety_decision.safe_action, dtype=np.float32)
            collision_response = (
                "background_projection_revalidated"
                if safety_decision.reason == "background_projection_revalidated"
                else "allow_candidate"
            )
            if (
                self.config.safety_enforcement_enabled
                and
                self.config.hardware_mode == "execute"
                and safety_decision.reason == "realtime_guard_hold_pending_projection"
                and self._last_successful_action is not None
                and self._safety_controller is not None
            ):
                # Keep the last command that GDK actually accepted, but only
                # after rechecking the measured-to-hold segment with the same
                # synchronous guard.  If it is stale/unsafe, the original
                # decision's measured-pose hold remains in force.
                previous = np.asarray(self._last_successful_action, dtype=np.float32)
                hold_decision = self._safety_controller.guard_layer.guard(
                    measured,
                    previous,
                    raw_action=raw_action,
                )
                if not hold_decision.hard_limit_triggered:
                    safe_limited = previous.copy()
                    safety_decision = replace(
                        safety_decision,
                        safe_action=previous.copy(),
                        safe_predicted_z_min=hold_decision.safe_predicted_z_min,
                        path_z_min=min(
                            safety_decision.path_z_min,
                            hold_decision.path_z_min,
                        ),
                        action_scale=0.0,
                        reason="collision_risk_hold_previous_safe_action",
                        projection_latency_ms=(
                            safety_decision.projection_latency_ms
                            + hold_decision.projection_latency_ms
                        ),
                    )
                    collision_response = "hold_previous_successful_action"
                else:
                    collision_response = "hold_measured_pose_previous_recheck_failed"
            elif safety_decision.hard_limit_triggered:
                collision_response = "hold_measured_pose"
            safety_guard_latency_ms = (time.perf_counter() - safety_started) * 1000
            safety_guard_cpu_ms = (time.thread_time() - safety_cpu_started) * 1000
            if safety_guard_latency_ms > self.config.max_safety_guard_latency_ms:
                self._consecutive_safety_deadline_misses += 1
                misses = self._consecutive_safety_deadline_misses
                deadline_fail_closed = (
                    self.config.safety_enforcement_enabled
                    and self.config.hardware_mode != "read-only"
                )
                self._rate_limited_warning(
                    "safety_guard_deadline",
                    "SAFETY GUARD DEADLINE MISSED latency=%.3fms budget=%.3fms "
                    "consecutive=%d/%d; %s",
                    safety_guard_latency_ms,
                    self.config.max_safety_guard_latency_ms,
                    misses,
                    self.config.max_consecutive_safety_deadline_misses,
                    (
                        "holding measured pose"
                        if deadline_fail_closed
                        else (
                            "read-only validation keeps the completed safety decision "
                            "and continues"
                            if self.config.safety_enforcement_enabled
                            else "monitor-only mode leaves the controller candidate unchanged"
                        )
                    ),
                )
                if deadline_fail_closed:
                    safe_limited = measured.copy()
                    safety_decision = replace(
                        safety_decision,
                        safe_action=measured.copy(),
                        safe_predicted_z_min=safety_decision.current_z_min,
                        hard_limit_triggered=True,
                        action_scale=0.0,
                        reason="safety_guard_deadline_hold",
                    )
                    collision_response = "hold_measured_pose_deadline"
                    if misses >= self.config.max_consecutive_safety_deadline_misses:
                        self._kinematic_limiter.reset(safe_limited)
                        raise RuntimeError(
                            "Safety guard exceeded its realtime deadline "
                            f"{misses} consecutive times"
                        )
                elif not self.config.safety_enforcement_enabled:
                    collision_response = "monitor_only_deadline_miss"
            else:
                self._consecutive_safety_deadline_misses = 0
            if (
                not self.config.safety_enforcement_enabled
                and safety_decision.hard_limit_triggered
            ):
                collision_response = "monitor_only_allow_candidate"
            if (
                safety_decision.soft_limit_triggered
                and not safety_decision.hard_limit_triggered
            ):
                warning_margin = next(
                    (
                        surface.warning_margin
                        for surface in self._safety_layer.surfaces.surfaces
                        if surface.name == safety_decision.limiting_surface
                    ),
                    None,
                )
                self._rate_limited_warning(
                    "table_proximity_warning",
                    "SURFACE PROXIMITY WARNING: distance=%.4fm warning_margin=%s "
                    "surface=%s point=%s; candidate remains allowed",
                    safety_decision.distance_to_surface,
                    "unset" if warning_margin is None else f"{warning_margin:.4f}m",
                    safety_decision.limiting_surface,
                    safety_decision.limiting_point,
                )
            self.last_action_intervened = bool(
                self.config.safety_enforcement_enabled
                and safety_decision.hard_limit_triggered
            )
            if self.last_action_intervened:
                self._kinematic_limiter.reset(safe_limited)
            if not np.allclose(safe_limited, limited, rtol=0.0, atol=1e-7):
                self._rate_limited_warning(
                    "table_hard_limit",
                    "COLLISION RISK: %s candidate response=%s reason=%s alpha=%.6f "
                    "current_z_min=%.4f predicted_z_min=%.4f safe_z_min=%.4f "
                    "surface=%s point=%s",
                    (
                        "rejected"
                        if self.config.safety_enforcement_enabled
                        else "monitor-only observed unsafe"
                    ),
                    collision_response,
                    safety_decision.reason,
                    safety_decision.action_scale,
                    safety_decision.current_z_min,
                    safety_decision.predicted_z_min,
                    safety_decision.safe_predicted_z_min,
                    safety_decision.limiting_surface,
                    safety_decision.limiting_point,
                )
                if self.config.safety_enforcement_enabled:
                    # Re-anchor the stateful limiter to the target actually sent.
                    self._kinematic_limiter.reset(safe_limited)
            limited = (
                safe_limited
                if self.config.safety_enforcement_enabled
                else controller_candidate.copy()
            )
        limited = self._dispatch_checked_action(limited)
        self._last_state = limited.copy()
        self._capture_control_telemetry(
            measured=measured,
            model_candidate=model_candidate,
            controller_candidate=controller_candidate,
            adopted_action=limited,
            safety_decision=safety_decision,
            safety_guard_latency_ms=safety_guard_latency_ms,
            collision_response=collision_response,
        )
        if safety_decision is not None:
            assert self._safety_logger is not None
            context = dict(safety_context or {})
            for private_key in ("raw_action", "controller_target", "preview_control_targets"):
                context.pop(private_key, None)
            context["control_latency_ms"] = (time.perf_counter() - control_started) * 1000
            context["safety_guard_latency_ms"] = safety_guard_latency_ms
            context["safety_check_fps"] = self.config.safety_check_fps
            context["guard_evaluated"] = True
            context["trajectory_step"] = 0
            context["safety_state_read_ms"] = safety_state_read_ms
            context["safety_guard_cpu_ms"] = safety_guard_cpu_ms
            if self.config.detailed_safety_timing:
                context["safety_state_cpu_ms"] = safety_state_cpu_ms
                context["state_read_stages"] = state_read_stages
                context["safety_thread_id"] = threading.get_native_id()
                context["safety_geometry_ms"] = safety_guard_latency_ms - safety_state_read_ms
                context["safety_geometry_cpu_ms"] = safety_guard_cpu_ms - safety_state_cpu_ms
            context["safety_guard_budget_ms"] = self.config.max_safety_guard_latency_ms
            context["safety_deadline_misses"] = self._consecutive_safety_deadline_misses
            context["collision_response"] = collision_response
            context["gdk_command_accepted"] = self._last_command_accepted
            context["adopted_action"] = limited.tolist()
            context["feedback_age_ms"] = (
                None
                if self._last_feedback_monotonic is None
                else (time.monotonic() - self._last_feedback_monotonic) * 1000
            )
            self._safety_logger.write(safety_decision, context)
        if (
            next_target is not None
            and self.config.safety_enforcement_enabled
            and safety_guard_latency_ms <= self.config.max_safety_guard_latency_ms
            and self._last_command_accepted is not False
        ):
            if safety_decision.hard_limit_triggered:
                next_target = limited.copy()
                next_limiter = self._kinematic_limiter.fork()
                next_limiter.reset(next_target)
            self._certified_step = {
                "action": next_target.copy(), "first": limited.copy(),
                "measured": measured.copy(), "limiter": next_limiter,
                "decision": replace(safety_decision, safe_action=next_target.copy()),
                "started": safety_started, "context": dict(safety_context or {}),
            }
        return limited

    def _audit_async_check(self, result) -> None:
        if result.get("error"):
            logger.error("Asynchronous safety check failed: %s", result["error"])
            return
        context = dict(result["task"]["context"])
        for private_key in ("raw_action", "controller_target", "preview_control_targets"):
            context.pop(private_key, None)
        context.update(record_type="worker_check", guard_evaluated=True,
                       safety_check_fps=50, worker_mode="process",
                       safety_guard_latency_ms=result["check_ms"],
                       safety_guard_budget_ms=CHECK_BUDGET_MS,
                       check_within_budget=result["check_ms"] <= CHECK_BUDGET_MS,
                       check_started=result["started"],
                       compute_started=result.get("compute_started", result["started"]),
                       check_completed=result.get("check_completed"),
                       submitted_at=result["task"].get("submitted_at"),
                       checked_points=len(result.get("sampled_target_indices", result["task"]["targets"])),
                       target_points=len(result["task"]["targets"]),
                       collision_action_stride=result.get("collision_action_stride", 1),
                       sampled_target_indices=result.get("sampled_target_indices"),
                       sampled_committed_indices=result.get("sampled_committed_indices"),
                       evaluated_configurations=result.get("evaluated_configurations"),
                       geometry_timing=result.get("geometry_timing"),
                       inter_arm=result.get("inter_arm"),
                       escape_approved=result.get("escape_approved", False),
                       live_collision=(result["live_decision"].hard_limit_triggered
                                       if result.get("live_decision") is not None else None),
                       live_collision_reason=(result["live_decision"].reason
                                              if result.get("live_decision") is not None else None),
                       measured_feedback=result["measured"].tolist(),
                       generation=result["task"]["generation"],
                       safety_request_id=result["task"]["request_id"],
                       gdk_command_accepted=None)
        self._safety_logger.write(result["decision"], context)

    def _audit_safety_event(self, event, *, task=None, **details):
        if self._safety_logger is None:
            return
        self._safety_logger.write_event(dict(
            record_type="safety_event", event=event,
            generation=self._async_generation if task is None else task["generation"],
            safety_request_id=None if task is None else task["request_id"],
            certified_buffer_steps=len(self._async_steps), **details))

    def _invalidate_async_safety(self, *, clear_hold: bool) -> None:
        self._async_generation += 1
        self._async_pending = None
        self._async_ready_result = None
        self._async_steps = []
        if clear_hold:
            self._async_last_check_at = None
            self._async_collision_active = False
            self._async_collision_decision = None
            self._async_latest_live_decision = None
            self._async_latest_live_decision_at = None
            self._async_hold = None
            self._async_brake = None
            self._async_braking_active = False
            self._async_last_waypoint = None
        # Never reset the liveness/startup deadline when reanchoring.

    def _retire_async_motion(self, *, allow_rest_recovery: bool = False, reason="external_hold") -> None:
        """Cancel queued motion, retaining only a check of the exact held anchor.

        Window expiry is not a session change. A fresh result for this same
        accepted target can renew hold, but cannot restore pre-hold velocity.
        Only an independent zero-velocity recovery may retain motion authority
        during ordinary expiry; explicit holds cancel even that recovery.
        An unexecuted future anchor never authorizes motion from another pose.
        """
        self._audit_safety_event("motion_retired", task=self._async_pending, reason=reason)
        self._async_steps = []
        if not (self.config.certified_braking_enabled and self._async_brake is not None
                and np.any(self._kinematic_limiter.velocity)):
            self._kinematic_limiter.reset(self._last_state)
        pending = self._async_pending
        if pending is not None and (
            np.array_equal(pending["anchor"], self._last_state)
            or np.array_equal(pending.get("fallback_anchor"), self._last_state)
        ):
            # An independently checked recovery from THIS stationary anchor is
            # not invalidated by expiry of an older window. Explicit external
            # holds still cancel it, as do all pre-hold velocity snapshots.
            recovery_matches = (allow_rest_recovery
                                and np.array_equal(pending["anchor"], self._last_state)
                                and "anchor_velocity" in pending
                                and not np.any(pending["anchor_velocity"]))
            if not recovery_matches:
                pending["motion_cancelled"] = True
        else:
            self._invalidate_async_safety(clear_hold=False)

    def _submit_async_window(self, target, model_candidate, context, *, tail=None,
                             submission_phase="unknown"):
        if self._async_pending is not None or self._async_braking_active:
            return
        window_started = time.perf_counter()
        # Legacy mode appends checked segments. The historical bounds candidate
        # still replaces a suffix behind a six-point frozen anchor.
        preview_steps = 2
        if self.config.rolling_safety_enabled and self.config.rolling_legacy_geometry:
            # Append a checked 70 ms segment; never replace committed points.
            # At most 8 old + 7 new points: 150 ms commitment, within the
            # unchanged 200 ms geometric-reference lifetime.
            if len(self._async_steps) > 8:
                return
            tail = self._async_steps[-1] if self._async_steps else None
            preview_steps = 10 if tail is None else 7
        elif self.config.rolling_safety_enabled:
            tail = self._async_steps[min(5, len(self._async_steps)-1)] if self._async_steps else None
            preview_steps = 10 if tail is None else 4
        limiter = (self._kinematic_limiter if tail is None else tail["limiter"]).fork()
        anchor = self._last_state if tail is None else tail["action"]
        anchor_velocity = limiter._velocity.copy()
        targets, limiters = [], []
        references = (context or {}).get("trajectory_preview")
        preview_started = time.perf_counter()
        if callable(references):
            # Only after pending/queue-capacity checks above. Same inputs and
            # same numerical operations as eager preview; no skipped geometry.
            references = references()
        preview_ms = (time.perf_counter() - preview_started) * 1000
        # Reference zero belongs to this tick's selected output. The frozen
        # suffix begins on the following tick, so skip it as well as the queue.
        offset = len(self._async_steps) + 1 if tail is not None else 0
        preplanned = bool((context or {}).get("preplanned_trajectory") is True)
        if preplanned and references is None:
            raise ValueError("Preplanned safety window requires an explicit trajectory preview")
        if references is not None:
            references = np.asarray(references, dtype=np.float32)
            if (references.ndim != 2 or references.shape[1] != len(target)
                    or len(references) < offset + preview_steps + 1
                    or not np.isfinite(references).all()):
                raise ValueError("Invalid continuous trajectory preview")
        path_started = time.perf_counter()
        for i in range(preview_steps):
            if preplanned:
                preview = limiter.adopt_preplanned(references[offset + i])
            elif references is None:
                preview = limiter.update(target)
            else:
                j = offset + i
                preview = limiter.update_tracking(
                    references[j], (references[j + 1] - references[j]) / limiter.dt)
            checked = self._apply_absolute_position_limits(preview, warning_suffix="async_preview")
            if preplanned and not np.allclose(checked, preview, rtol=0.0, atol=1e-7):
                raise ValueError("Preplanned rollback waypoint exceeded hardware position limits")
            if not np.array_equal(preview, checked):
                limiter.reconcile_position_clip(checked)
            targets.append(checked.copy())
            limiters.append(limiter.fork())
        path_ms = (time.perf_counter() - path_started) * 1000
        self._async_request_sequence += 1
        task = dict(generation=self._async_generation, request_id=self._async_request_sequence,
                    urgent=self._async_urgent,
                    anchor=anchor.copy(), fallback_anchor=self._last_state.copy(),
                    anchor_velocity=anchor_velocity,
                    targets=targets, limiters=limiters, raw=model_candidate.copy(),
                    context={k: v for k, v in (context or {}).items() if k != "trajectory_preview"})
        task["reference_preview_offset"] = offset if references is not None else None
        task["rolling"] = self.config.rolling_safety_enabled
        task["rolling_legacy_geometry"] = self.config.rolling_safety_enabled and self.config.rolling_legacy_geometry
        if task["rolling_legacy_geometry"]:
            task["allow_sampled_escape"] = self._async_ever_dispatched
            # Include the selected but not yet dispatched point during early
            # prefetch, then every immutable queued point up to the tail.
            task["committed"] = ([self._async_dispatch_action.copy()]
                if self._async_dispatch_action is not None else []) + [
                    step["action"].copy() for step in self._async_steps]
        task["anchor_identity"] = (self._async_last_waypoint if tail is None else
                                   (tail["result"]["task"]["generation"],
                                    tail["result"]["task"]["request_id"], tail["step"]))
        if self.config.certified_braking_enabled:
            lower = np.empty(24, dtype=np.float32)
            upper = np.empty(24, dtype=np.float32)
            body_indices = tuple(range(7)) + tuple(range(8, 15)) + tuple(range(16, 24))
            for index, name in zip(body_indices, GDK_BODY_JOINTS, strict=True):
                low, high = GDK_BODY_JOINT_LIMITS[name]
                lower[index] = low + self.config.gdk_position_limit_margin
                upper[index] = high - self.config.gdk_position_limit_margin
            lower[[7, 15]], upper[[7, 15]] = self.effector_profile.hardware_state_range
            task["braking"] = dict(acceleration=limiter.max_acceleration.copy(),
                                   max_velocity=limiter.max_velocity.copy(), dt=limiter.dt,
                                   lower=lower, upper=upper,
                                   target_velocities=[item.velocity for item in limiters])
        self._async_pending = task
        self._async_urgent = False
        submit_started = time.perf_counter()
        self._async_worker.submit(task)
        submit_ms = (time.perf_counter() - submit_started) * 1000
        self._audit_safety_event("request_submitted", task=task,
                                checked_points=len(targets), anchor_identity=task["anchor_identity"],
                                submitted_at=task.get("submitted_at"),
                                submission_phase=submission_phase,
                                window_preview_ms=preview_ms,
                                window_path_ms=path_ms,
                                window_submit_ms=submit_ms,
                                window_total_ms=(time.perf_counter() - window_started) * 1000)

    def _renew_rolling_result(self, result, now):
        if not result.get("task", {}).get("rolling"):
            return result
        self._async_renew_failure = None
        self._async_renew_details = {}
        origin = result.get("geometry_feedback_at", result["started"])
        feedback = self._async_worker.feedback
        # The snapshot must precede the timestamp: a feedback publication in
        # poll/splice must not look as though it came from the future.
        now = time.perf_counter()
        self._async_renew_details.update(
            geometry_age_ms=(now-origin)*1000,
            feedback_age_ms=None if feedback is None else (now-feedback[1])*1000)
        if not np.isfinite(origin) or not 0 <= now-origin < .2:
            self._async_renew_failure = "geometry_reference_expired"
            return None
        if not np.isfinite(result.get("check_ms", float("nan"))):
            self._async_renew_failure = "invalid_check_duration"
            return None
        if feedback is None or not 0 <= now-feedback[1] < .04:
            self._async_renew_failure = "feedback_missing_or_expired"
            return None
        if result["task"].get("rolling_legacy_geometry"):
            if result.get("sampled_snapshot"):
                q = np.asarray(feedback[0])
                if (q.shape != (24,) or not np.isfinite(q).all()
                        or np.any(q[[7, 15]] < 0) or np.any(q[[7, 15]] > .025001)):
                    self._async_renew_failure = "invalid_feedback"
                    return None
                # Never relabel fresh feedback as a fresh geometry check.
                return dict(result, geometry_feedback_at=origin)
            if "sampled_feedback" not in result:
                self._async_renew_failure = "sampled_proof_missing"
                return None
            displacement = sampled_feedback_displacement(result["sampled_feedback"], feedback[0],
                                                           self._async_hold_coefficients)
            self._async_renew_details.update(
                sampled_margin_m=float(result["sampled_feedback"]["margin"]),
                displacement_bound_m=None if self._async_hold_coefficients is None else
                    float(np.dot(self._async_hold_coefficients,
                        np.abs(np.asarray(feedback[0], dtype=np.float64)-result["sampled_feedback"]["reference"]))))
            if not np.isfinite(displacement):
                self._async_renew_failure = "sampled_feedback_displacement"
                return None
            return dict(result, started=feedback[1], measured=feedback[0].copy(),
                        geometry_feedback_at=origin, sampled_feedback_displacement=displacement)
        if "trajectory_box" not in result:
            self._async_renew_failure = "trajectory_box_missing"
            return None
        clearance = trajectory_clearance(result["trajectory_box"], feedback[0], self._async_hold_coefficients)
        if not np.isfinite(clearance) or clearance <= 1e-6:
            self._async_renew_failure = "trajectory_clearance"
            return None
        reserves = result.get("target_brakes")
        if (reserves is None or len(reserves) != len(result["task"]["targets"])
                or any(r is None or braking_clearance(r, feedback[0], self._async_hold_coefficients) <= 1e-6
                       for r in reserves)):
            self._async_renew_failure = "braking_clearance"
            return None  # A fresh motion proof must retain a fresh safe stop too.
        return dict(result, started=feedback[1], measured=feedback[0].copy(),
                    geometry_feedback_at=origin, trajectory_bound_clearance=clearance)

    def _hold_sampled_collision(self, result, now):
        """Reject motion, not the task: hold and independently replan from rest."""
        if self._async_hold is None:
            # Before the first output, there is no historical accepted command.
            proof = result["fallback_hold"]
            if (proof.hard_limit_triggered or not np.array_equal(
                    result["task"]["fallback_anchor"], self._last_state)):
                raise SafetyDispatchExpiredError("No initial accepted target available for collision hold")
            self._async_hold = dict(action=self._last_state.copy(), limiter=self._kinematic_limiter.fork(),
                                    result=result, step=-1, hold_proof=proof)
        # Never append an old moving suffix after zeroing the hold velocity.
        self._audit_safety_event("motion_retired", task=result["task"], reason="collision_hold_replan")
        self._invalidate_async_safety(clear_hold=False)
        self._kinematic_limiter.reset(self._last_state)
        self._async_urgent = True
        self._async_collision_active = True
        self._async_collision_decision = (result["live_decision"]
            if result["live_decision"].hard_limit_triggered else result["decision"])
        inter_arm = result.get("inter_arm") or {}
        if inter_arm.get("collision"):
            self._rate_limited_warning(
                "async_inter_arm_collision",
                "INTER-ARM COLLISION RISK: holding last accepted target; "
                "pair=%s<->%s separation=%.4fm hard_margin=%.4fm; "
                "continuing candidate checks, task remains active",
                inter_arm.get("left_geometry", "left"),
                inter_arm.get("right_geometry", "right"),
                float(inter_arm.get("minimum_separation_m", math.nan)),
                float(inter_arm.get("hard_margin_m", math.nan)),
            )
        else:
            self._rate_limited_warning("async_collision",
                "COLLISION RISK: holding last accepted target; continuing candidate checks, task remains active")

    def _splice_rolling_result(self, result, now):
        """Replace an unexecuted suffix only when its exact q/v anchor survives."""
        pending = self._async_pending
        if (pending is None or result["task"]["generation"] != self._async_generation
                or result["task"]["request_id"] != pending["request_id"]):
            self._audit_safety_event("result_discarded", task=result["task"], reason="request_or_generation_mismatch")
            return None
        if result.get("error"):
            raise RuntimeError(result["error"])
        task = result["task"]
        if result.get("sampled_snapshot"):
            renewed = self._renew_rolling_result(result, now)
            if renewed is None:
                self._async_pending = None
                self._audit_safety_event("result_discarded", task=task, reason=self._async_renew_failure)
                return None
            result = renewed
            self._async_last_check_at = result["started"]
        if result.get("sampled_snapshot") and (
                result["live_decision"].hard_limit_triggered or result["decision"].hard_limit_triggered):
            self._hold_sampled_collision(result, now)
            return None
        matches = [i for i, step in enumerate(self._async_steps)
                   if np.array_equal(step["action"], task["anchor"])
                   and np.array_equal(step["limiter"].velocity, task["anchor_velocity"])
                   and (step["result"]["task"]["generation"], step["result"]["task"]["request_id"], step["step"])
                   == task["anchor_identity"]]
        if not matches and np.array_equal(task["anchor"], self._last_state):
            # Existing admission handles startup and a just-consumed anchor.
            if (not np.array_equal(task["anchor_velocity"], self._kinematic_limiter.velocity)
                    or task["anchor_identity"] != self._async_last_waypoint):
                self._audit_safety_event("result_discarded", task=task, reason="current_anchor_identity_or_velocity")
                self._async_pending = None
                return None
            renewed = self._renew_rolling_result(result, now)
            if self._async_steps:
                # The handoff point can become the CURRENT point while the
                # check is in flight. Install now; legacy deferral until the
                # old suffix drains would miss this valid handoff forever.
                if (renewed is None or result["decision"].hard_limit_triggered
                        or pending.get("motion_cancelled")):
                    reason = (self._async_renew_failure if renewed is None else
                              "collision" if result["decision"].hard_limit_triggered else "motion_cancelled")
                    self._audit_safety_event("result_discarded", task=task, reason=reason,
                                            **(self._async_renew_details if renewed is None else {}))
                    self._async_pending = None
                    return None
                self._async_steps = []
            if renewed is None:
                self._audit_safety_event("result_discarded", task=task, reason=self._async_renew_failure,
                                        **self._async_renew_details)
            return result if renewed is None else renewed
        self._async_pending = None
        renewed = self._renew_rolling_result(result, now)
        reserves = result.get("target_brakes")
        if (not matches or renewed is None or pending.get("motion_cancelled")
                or result["decision"].hard_limit_triggered
                or (not task.get("rolling_legacy_geometry") and
                    (reserves is None or len(reserves) != len(task["targets"]) or any(r is None for r in reserves)))):
            # Existing certified points and their stops remain authoritative.
            reason = ("frozen_anchor_not_found" if not matches else
                      self._async_renew_failure if renewed is None else
                      "motion_cancelled" if pending.get("motion_cancelled") else
                      "collision" if result["decision"].hard_limit_triggered else "braking_reserve_missing")
            self._audit_safety_event("result_discarded", task=task, reason=reason,
                                    **(self._async_renew_details if renewed is None else {}))
            return None
        prefix = self._async_steps[:matches[0]+1]
        suffix = [dict(action=q, limiter=limiter, result=renewed, step=i,
                       hold_proof=result["target_holds"][i], brake_reserve=None if reserves is None else reserves[i],
                       sampled_hold_feedback=(result["target_sampled_feedback"][i]
                                              if task.get("rolling_legacy_geometry") else None))
                  for i, (q, limiter) in enumerate(zip(task["targets"], task["limiters"], strict=True))]
        self._async_steps = prefix + suffix
        self._audit_safety_event("result_admitted", task=task, reason="frozen_anchor_append",
                                checked_points=len(suffix))
        return None

    def _async_window_has_dispatch_slot(self, result, now: float) -> bool:
        """Reserve preparation time at the next actual servo slot, not 10 ms.

        This admission estimate never replaces the expiry check immediately
        before GDK (after pacing). A scheduling overrun still falls back to a
            separately checked hold. Sampled snapshots have a 200 ms lifetime;
            moving feedback has a 40 ms limit; holding uses the site-configured limit.
        """
        next_slot = now
        if self._last_servo_dispatch is not None:
            next_slot = max(next_slot, self._last_servo_dispatch + self.config.servo_control_period_s)
        preparation_s = 0.002
        limit = 200.0 if result.get("sampled_snapshot") else RESULT_MAX_AGE_MS
        return (next_slot + preparation_s - result["started"]) * 1000 < limit

    def _send_async_action(self, target, model_candidate, context, control_started):
        prefetch_error = getattr(self, "_async_prefetch_error", None)
        if prefetch_error is not None:
            raise RuntimeError("Post-dispatch safety prefetch failed; stopping command streaming") from prefetch_error
        self.last_async_response = None
        if self._last_state is None:
            raise RuntimeError("Initialize the command reference from measured feedback before asynchronous control")
        if self._async_braking_active:
            return self._send_reserved_brake(model_candidate, context, control_started)
        now = time.perf_counter()
        if self._async_started is None:
            self._async_started = now
        # Retire the OLD window before stashing a newly returned result. Otherwise
        # invalidation also deletes a fresh, independently checked hold result.
        if self.config.rolling_safety_enabled and self._async_steps:
            renewed = self._renew_rolling_result(self._async_steps[0]["result"], now)
            if renewed is None:
                self._retire_async_motion(allow_rest_recovery=True, reason=self._async_renew_failure)
            else:
                self._async_steps[0]["result"] = renewed
        if self._async_steps and not self._async_window_has_dispatch_slot(self._async_steps[0]["result"], now):
            self._retire_async_motion(allow_rest_recovery=True, reason="dispatch_slot_expired")
        result = self._async_ready_result
        if result is None:
            result = self._async_worker.poll()
            if result is not None:
                self._audit_safety_event("result_polled", task=result["task"],
                                        check_completed=result.get("check_completed"))
                live_decision = result.get("live_decision")
                if result.get("error") is None and live_decision is not None:
                    # Telemetry follows every new measured-pose check, even
                    # when control correctly keeps dispatching an older safe
                    # hold.  Previously the browser kept the hold result's
                    # first distance (often exactly the hard margin) while
                    # newer worker checks already saw deeper penetration or
                    # recovery.
                    self._async_latest_live_decision = live_decision
                    self._async_latest_live_decision_at = result.get(
                        "geometry_feedback_at", result.get("started")
                    )
        now = time.perf_counter()
        self._async_ready_result = None
        if result is not None and self.config.rolling_safety_enabled:
            result = self._splice_rolling_result(result, now)
        if result is not None and self._async_steps:
            self._async_ready_result = result
            result = None
        fresh_check = False
        if (result is not None and self._async_pending is not None
                and result["task"]["generation"] == self._async_generation
                and result["task"]["request_id"] == self._async_pending["request_id"]):
            cancelled = self._async_pending.get("motion_cancelled", False)
            self._async_pending = None
            if result.get("error"):
                raise RuntimeError(result["error"])
            task = result["task"]
            age_ms = (now - result["started"]) * 1000
            result_age_limit = 200.0 if result.get("sampled_snapshot") else RESULT_MAX_AGE_MS
            anchor_matches = (np.array_equal(task["anchor"], self._last_state)
                              and (not result["hold"].hard_limit_triggered or result.get("escape_approved", False)))
            fallback = result.get("fallback_hold")
            if (age_ms < result_age_limit and not anchor_matches and fallback is not None
                    and np.array_equal(task.get("fallback_anchor"), self._last_state)
                    and fallback.hard_limit_triggered):
                raise RuntimeError("Measured-to-hold path is unsafe; no certified fallback target")
            fallback_matches = (fallback is not None and not fallback.hard_limit_triggered
                                and np.array_equal(task.get("fallback_anchor"), self._last_state)
                                and np.array_equal(fallback.safe_action, self._last_state))
            if 0 <= age_ms < result_age_limit and (anchor_matches or fallback_matches):
                fresh_check = True
                decision = result["decision"]
                blocked = decision.hard_limit_triggered
                hold_only = bool(cancelled or task.get("motion_cancelled") or not anchor_matches
                                 or (not task.get("rolling") and result["check_ms"] > CHECK_BUDGET_MS))
                brake_reserves = result.get("target_brakes")
                no_stop_reserve = (self.config.certified_braking_enabled
                                   and (brake_reserves is None or len(brake_reserves) != len(task["targets"])
                                        or any(item is None for item in brake_reserves)))
                hold_only = hold_only or no_stop_reserve
                result["hold_only"] = hold_only
                result["hold_reason"] = (
                    "collision" if blocked else
                    "retired_motion" if cancelled or task.get("motion_cancelled") else
                    "anchor_mismatch" if not anchor_matches else
                    "check_over_budget" if not task.get("rolling") and result["check_ms"] > CHECK_BUDGET_MS else None
                )
                if no_stop_reserve and result["hold_reason"] is None:
                    result["hold_reason"] = "insufficient_braking_clearance"
                hold_proof = result["hold"] if anchor_matches else fallback
                if hold_only and not blocked:
                    decision = result["decision"] = hold_proof
                if blocked:
                    decision = result["decision"] = replace(
                        decision, safe_action=self._last_state.copy(),
                        safe_predicted_z_min=hold_proof.safe_predicted_z_min,
                    )
                    self._rate_limited_warning("async_collision", "COLLISION RISK: rejected candidate; holding last accepted target")
                hold_limiter = self._kinematic_limiter.fork()
                hold_limiter.reset(self._last_state)
                # Preserve the independently checked CURRENT target even if the
                # new motion has too little remaining lifetime for another step.
                self._async_hold = dict(action=self._last_state.copy(), limiter=hold_limiter,
                                        result=result, step=-1, hold_proof=hold_proof,
                                        sampled_hold_feedback=result.get("anchor_sampled_feedback" if anchor_matches
                                                                         else "fallback_sampled_feedback"))
                self._async_steps = []
                for index, (action, limiter) in enumerate(zip(task["targets"], task["limiters"])):
                    if blocked or hold_only:
                        # A hold certificate is not a two-tick motion window.
                        # Replan from rest immediately instead of spending two
                        # more ticks consuming synthetic held trajectory steps.
                        break
                    self._async_steps.append(dict(
                        action=action, limiter=limiter, result=result, step=index,
                        hold_proof=hold_proof if blocked or hold_only else result["target_holds"][index],
                        brake_reserve=None if brake_reserves is None else brake_reserves[index],
                        sampled_hold_feedback=(result["target_sampled_feedback"][index]
                                               if task.get("rolling_legacy_geometry") else None),
                    ))
                if self.config.rolling_safety_enabled and "geometry_feedback_at" not in result:
                    # A sampled proof alone may authorize legacy motion, but
                    # cannot authorize a rolling window after bound failure.
                    self._async_steps = []
                if self._async_steps:
                    self._audit_safety_event("result_admitted", task=task, reason="current_anchor",
                                            checked_points=len(self._async_steps))
            else:
                self._audit_safety_event("result_discarded", task=task, reason="hold_age_or_anchor_mismatch",
                                        certificate_age_ms=age_ms, anchor_matches=anchor_matches,
                                        fallback_matches=fallback_matches)
                self._rate_limited_warning(
                    "async_safety_late", "Safety window discarded: check=%.3fms age=%.3fms; awaiting fresh certificate",
                    result["check_ms"], age_ms,
                )
        if self._async_steps and not self._async_window_has_dispatch_slot(self._async_steps[0]["result"], now):
            self._retire_async_motion(allow_rest_recovery=True, reason="dispatch_slot_expired")
        if not self._async_steps:
            if self.config.certified_braking_enabled and np.any(self._kinematic_limiter.velocity):
                return self._send_reserved_brake(model_candidate, context, control_started)
            # A gap is a true hold, not continued integration of old velocity.
            if np.any(self._kinematic_limiter._velocity):
                # The delayed window may still renew the exact anchor's hold
                # certificate, but must never resume its pre-hold velocity.
                if self._async_pending is not None:
                    self._async_pending["motion_cancelled"] = True
                self._kinematic_limiter.reset(self._last_state)
            self._submit_async_window(target, model_candidate, context, submission_phase="hold")
            held = self._async_hold
            if held is None:
                # Initial startup/re-anchor never sends an unchecked command.
                # Pending must be acknowledged by pause logic before confirming.
                self.last_command_pending = True
                self.last_action_intervened = True
                reanchor_started = self._async_reanchor_wait_started
                waiting_for_reanchor_certificate = (
                    reanchor_started is not None
                    and 0 <= now - reanchor_started <= 5.0
                )
                if not waiting_for_reanchor_certificate:
                    reference_time = (
                        self._async_last_dispatch_at
                        if self._async_ever_dispatched
                        else self._async_started
                    )
                    if now - reference_time > (
                        0.04 if self._async_ever_dispatched else 5.0
                    ):
                        raise SafetyDispatchExpiredError(
                            "No checked hold available after safety re-anchor"
                        )
                return self._last_state.copy()
            selected = self._select_async_hold(now)
        else:
            selected = self._async_steps.pop(0)
        selection_finished = time.perf_counter()
        result = selected["result"]
        action = selected["action"].copy()
        proof_started = time.perf_counter()
        if self.config.rolling_safety_enabled and selected["step"] >= 0:
            renewed = self._renew_rolling_result(result, time.perf_counter())
            if result["task"].get("rolling_legacy_geometry"):
                index = selected["step"]
                expected_previous = result["task"]["anchor"] if index == 0 else result["task"]["targets"][index-1]
                if (not np.array_equal(action, result["task"]["targets"][index])
                        or not np.array_equal(self._last_state, expected_previous)):
                    raise SafetyDispatchExpiredError("Command differs from sampled trajectory")
            else:
                proof = result["trajectory_box"]
                for point in (self._last_state, action):
                    if np.any(np.abs(np.asarray(point, dtype=np.float64)-proof["anchor"]) > proof["radius"]+1e-9):
                        raise SafetyDispatchExpiredError("Command escaped certified trajectory box")
            if renewed is None:
                self._retire_async_motion(allow_rest_recovery=True)
                if result["task"].get("rolling_legacy_geometry"):
                    raise SafetyDispatchExpiredError("Sampled trajectory feedback changed before dispatch")
                return self._send_reserved_brake(model_candidate, context, control_started)
            selected["result"] = result = renewed
            if selected["hold_proof"] is not None:
                lower_z = (selected["hold_proof"].path_z_min if result.get("sampled_snapshot") else
                           selected["hold_proof"].path_z_min-result["sampled_feedback_displacement"]
                           if result["task"].get("rolling_legacy_geometry") else
                           result["trajectory_box"]["plane_limit"] + result["trajectory_bound_clearance"])
                # Never turn an unchecked stride waypoint into a hold proof.
                selected["hold_proof"] = replace(selected["hold_proof"],
                    path_z_min=min(selected["hold_proof"].path_z_min, lower_z),
                    safe_predicted_z_min=min(selected["hold_proof"].safe_predicted_z_min, lower_z))
        proof_finished = time.perf_counter()
        self._command_safety_started = selected.get("dispatch_authority_at", selected.get("hold_checked_at", result["started"]))
        self._async_dispatch_action = action.copy()
        self._async_dispatch_is_hold = selected["step"] < 0
        self._kinematic_limiter = selected["limiter"]
        # Read-only enforces the same certificate expiry as physical dispatch.
        dispatch_entered = time.perf_counter()
        try:
            self._validate_dispatch_age(time.perf_counter())
            adopted = self._dispatch_checked_action(action)
        except SafetyDispatchExpiredError:
            # This exception is raised BEFORE GDK, never after an ambiguous
            # SDK error. Retire motion; one separately certified hold is allowed.
            self._retire_async_motion(allow_rest_recovery=True, reason="final_dispatch_expired")
            if (self.config.certified_braking_enabled and self._async_brake is not None
                    and self._async_brake["index"] < self._async_brake["reserve"]["steps"]):
                return self._send_reserved_brake(model_candidate, context, control_started)
            selected = self._select_async_hold(time.perf_counter())
            result = selected["result"]
            action = selected["action"].copy()
            self._kinematic_limiter.reset(action)
            selected["limiter"] = self._kinematic_limiter.fork()
            self._command_safety_started = selected.get("dispatch_authority_at", selected.get("hold_checked_at", result["started"]))
            self._async_dispatch_action = action.copy()
            self._async_dispatch_is_hold = True
            self._validate_dispatch_age(time.perf_counter())
            adopted = self._dispatch_checked_action(action)
        decision = result["decision"]
        if self._last_command_accepted is False:
            self._invalidate_async_safety(clear_hold=True)
            raise RuntimeError("GDK rejected async target; safety window invalidated")
        self._async_ever_dispatched = True
        self._async_last_dispatch_at = time.perf_counter()
        self._async_reanchor_wait_started = None
        self._last_state = adopted.copy()
        if selected["step"] >= 0:
            self._async_last_waypoint = (result["task"]["generation"], result["task"]["request_id"], selected["step"])
            self._async_collision_active = False
            self._async_collision_decision = None
        elif self._async_collision_active:
            decision = replace(self._async_collision_decision, safe_action=adopted.copy(),
                               safe_predicted_z_min=result["decision"].safe_predicted_z_min,
                               action_scale=0.0, reason="collision_hold_waiting_for_safe_candidate")
        if selected.get("brake_reserve") is not None:
            self._async_brake = dict(reserve=selected["brake_reserve"], index=0, result=result)
        self._async_hold = dict(selected, action=adopted.copy())
        prefetch_started = time.perf_counter()
        # Prepare the next checked segment only after this GDK command is
        # accepted. The current 100 Hz slot no longer waits for preview,
        # path construction or worker IPC. The next slot still requires its
        # own fresh certificate and final dispatch-age check.
        try:
            self._submit_async_window(
                target, model_candidate, context,
                tail=self._async_steps[-1] if self._async_steps else None,
                submission_phase="after_dispatch",
            )
        except Exception as error:
            # The command was already accepted. Return it so the caller
            # commits the reservation, then fail closed before any next
            # command. Raising here would leave an accepted action
            # recorded as uncommitted and make retry semantics ambiguous.
            self._async_prefetch_error = error
            logger.exception("Post-dispatch safety prefetch failed; next action will stop")
        prefetch_finished = time.perf_counter()
        source_context = result["task"]["context"]
        preview_offset = result["task"].get("reference_preview_offset")
        source_time = source_context.get("source_sample_monotonic")
        preview_index = (
            preview_offset + selected["step"]
            if preview_offset is not None and selected["step"] >= 0 else None
        )
        preview_action_indices = source_context.get("preview_action_indices")
        policy_action_index = (
            preview_action_indices[preview_index]
            if preview_action_indices is not None
            and preview_index is not None
            and 0 <= preview_index < len(preview_action_indices)
            else source_context.get("action_index")
        )
        preview_targets = source_context.get("preview_control_targets")
        safety_target = (
            np.asarray(preview_targets[preview_index], dtype=np.float32).copy()
            if preview_targets is not None
            and preview_index is not None
            and 0 <= preview_index < len(preview_targets)
            else source_context.get("controller_target")
        )
        # The current caller may already be on a NEW chunk while a previously
        # checked prefix is dispatched. Preserve the selected queue's identity.
        self._last_dispatch_provenance = dict(
            chunk_id=source_context.get("chunk_id"),
            candidate_action_index=source_context.get("action_index"),
            policy_action_index=policy_action_index,
            safety_target_24d=(
                None if safety_target is None
                else np.asarray(safety_target, dtype=np.float32).copy()
            ),
            safety_request_id=result["task"]["request_id"],
            trajectory_step=selected["step"], reference_preview_index=preview_index,
            reference_sample_monotonic=(
                source_time + preview_index * self._kinematic_limiter.dt
                if source_time is not None and preview_index is not None else None
            ),
            holding_previous_target=selected["step"] < 0,
        )
        response = ("async_collision_hold" if decision.hard_limit_triggered else
                    "async_checked_hold" if selected["step"] < 0 or result.get("hold_only") else
                    "async_checked_motion")
        hold_reason = None
        if response != "async_checked_motion":
            # The cached hold proof's old compute duration is not the cause of
            # today's gap. Lifecycle events identify the actual missing result.
            hold_reason = ("collision" if decision.hard_limit_triggered else
                           "awaiting_fresh_window" if self.config.rolling_safety_enabled else
                           result.get("hold_reason") or "awaiting_fresh_window")
        # Normal checked motion must not reset the upstream low-pass filter
        # on every tick. Reanchor only after actual safety intervention.
        self.last_action_intervened = response != "async_checked_motion"
        self.last_async_response = response
        previous_telemetry = self._latest_control_telemetry
        visual_feedback = getattr(self._async_worker, "feedback", None)
        self._capture_control_telemetry(
            measured=(visual_feedback[0].copy() if visual_feedback is not None
                      else selected.get("hold_feedback", result["measured"])), model_candidate=model_candidate,
            controller_candidate=decision.controller_target, adopted_action=adopted,
            safety_decision=decision, safety_guard_latency_ms=result["check_ms"],
            collision_response=response,
            # ``live_decision`` is the independently checked measured GDK
            # snapshot.  Keep it separate from the held model candidate so a
            # tele-operated recovery immediately clears the physical alarm.
            live_safety_decision=(
                self._async_latest_live_decision
                if self._async_latest_live_decision is not None
                else result.get("live_decision")
            ),
        )
        if self._latest_control_telemetry is not previous_telemetry:
            self._latest_control_telemetry["safety"].update(
                guard_evaluated=fresh_check, guard_budget_ms=CHECK_BUDGET_MS,
                worker_mode="process", certificate_age_ms=(time.perf_counter() - self._command_safety_started) * 1000,
                hold_reason=hold_reason,
                collision_action_stride=result.get("collision_action_stride", 1),
                action_geometry_sampled=selected["hold_proof"] is not None,
                # ``inter_arm`` above is derived from ``live_safety_decision``
                # and therefore describes the measured GDK pose.  The worker
                # report describes the model candidate window.  Never replace
                # the former with the latter: doing so made the browser's
                # real-time capsule overlay and its central hazard notice
                # disagree in read-only observation.
                candidate_inter_arm=result.get("inter_arm"),
            )
        audit = dict(result["task"]["context"])
        for private_key in ("raw_action", "controller_target", "preview_control_targets"):
            audit.pop(private_key, None)
        audit.update(record_type="servo_dispatch", guard_evaluated=False, safety_check_fps=50, worker_mode="process",
                     control_started_monotonic=control_started,
                     dispatch_monotonic=self._last_servo_dispatch,
                     selection_ms=(selection_finished - control_started) * 1000,
                     proof_renew_ms=(proof_finished - proof_started) * 1000,
                     prefetch_ms=(prefetch_finished - prefetch_started) * 1000,
                     prefetch_phase="after_dispatch",
                     final_dispatch_phase_ms=(None if self.config.hardware_mode == "dry-run"
                                             or self._last_servo_dispatch is None else
                                             (self._last_servo_dispatch - dispatch_entered) * 1000),
                     safety_request_id=result["task"]["request_id"], generation=result["task"]["generation"],
                     trajectory_step=selected["step"], safety_guard_latency_ms=None,
                     safety_guard_budget_ms=CHECK_BUDGET_MS, safety_deadline_misses=0,
                     certificate_age_ms=(time.perf_counter() - self._command_safety_started) * 1000,
                     hold_bound_revalidated="hold_checked_at" in selected,
                     dispatch_feedback_age_ms=getattr(self, "_dispatch_feedback_age_ms", None),
                     control_latency_ms=(time.perf_counter() - control_started) * 1000,
                     collision_response=response, adopted_action=adopted.tolist(),
                     hold_reason=hold_reason,
                     rolling_safety=self.config.rolling_safety_enabled,
                     collision_action_stride=result.get("collision_action_stride", 1),
                     inter_arm=result.get("inter_arm"),
                     escape_approved=result.get("escape_approved", False),
                     action_geometry_sampled=selected["hold_proof"] is not None,
                     certified_buffer_steps=len(self._async_steps),
                     proof_renewal=("historical_accepted_target" if selected.get("historical_repeat") else
                                    "sampled_path_snapshot" if result.get("sampled_snapshot") else
                                    "sampled_path_feedback_bound" if result["task"].get("rolling_legacy_geometry")
                                    else "global_trajectory_box") if "geometry_feedback_at" in result else None,
                     geometry_feedback_at=result.get("geometry_feedback_at"),
                     gdk_command_accepted=self._last_command_accepted)
        self._safety_logger.write(replace(decision, safe_action=adopted.copy()), audit)
        return adopted

    def _send_reserved_brake(self, model_candidate, context, control_started):
        """Consume only a finite, pre-certified stop; no unchecked coasting."""
        stored = self._async_brake
        if stored is None:
            raise SafetyDispatchExpiredError("No certified stopping reserve for moving target")
        reserve = stored["reserve"]
        index = stored["index"]
        if index >= reserve["steps"]:
            raise SafetyDispatchExpiredError("Stopping reserve exhausted; refusing extrapolation")
        anchor, _ = braking_sample(reserve, index)
        if not np.array_equal(anchor, self._last_state):
            raise SafetyDispatchExpiredError("Stopping reserve anchor mismatch")
        feedback = self._async_worker.feedback
        now = time.perf_counter()
        if feedback is None or not 0 <= now - feedback[1] < RESULT_MAX_AGE_MS / 1000:
            raise SafetyDispatchExpiredError("Stopping reserve needs fresh feedback")
        clearance = braking_clearance(reserve, feedback[0], self._async_hold_coefficients)
        if not np.isfinite(clearance) or clearance <= 1e-6:
            raise SafetyDispatchExpiredError("Stopping reserve no longer conservatively safe")
        if not self._async_braking_active:
            self._invalidate_async_safety(clear_hold=False)
            self._async_braking_active = True
        action, velocity = braking_sample(reserve, index + 1)
        if not np.array_equal(action, self._apply_absolute_position_limits(action, warning_suffix="braking")):
            raise SafetyDispatchExpiredError("Stopping reserve exceeds physical range")
        self._command_safety_started = feedback[1]
        self._async_dispatch_action = action.copy()
        self._async_dispatch_is_hold = False
        self._validate_dispatch_age(time.perf_counter())
        adopted = self._dispatch_checked_action(action)
        if self._last_command_accepted is False or not np.array_equal(adopted, action):
            raise SafetyDispatchExpiredError("Stopping command was not accepted exactly")
        self._last_state = adopted.copy()
        self._kinematic_limiter._position = adopted.copy()
        self._async_last_waypoint = None
        self._kinematic_limiter._velocity = velocity.copy()
        stored["index"] += 1
        self._async_braking_active = stored["index"] < reserve["steps"]
        self._async_last_dispatch_at = time.perf_counter()
        self._async_ever_dispatched = True
        self.last_action_intervened = True
        self.last_async_response = "async_checked_brake"
        result = dict(stored["result"], measured=feedback[0].copy(), started=feedback[1], hold_only=True)
        proof = replace(result["decision"], safe_action=adopted.copy(),
                        path_z_min=reserve["plane_limit"] + clearance,
                        safe_predicted_z_min=reserve["plane_limit"] + clearance,
                        reason="certified_braking_reserve")
        result["decision"] = proof
        self._async_hold = dict(action=adopted.copy(), limiter=self._kinematic_limiter.fork(),
                                result=result, step=-1, hold_proof=proof)
        self._capture_control_telemetry(measured=feedback[0], model_candidate=model_candidate,
                                       controller_candidate=action, adopted_action=adopted,
                                       safety_decision=proof, safety_guard_latency_ms=0,
                                       collision_response="async_checked_brake")
        audit = dict(context or {})
        for private_key in ("raw_action", "controller_target", "preview_control_targets"):
            audit.pop(private_key, None)
        audit.update(record_type="servo_dispatch", trajectory_step=-2,
                     collision_response="async_checked_brake", hold_reason="certified_deceleration",
                     safety_request_id=stored["result"]["task"]["request_id"],
                     generation=stored["result"]["task"]["generation"],
                     brake_step=stored["index"], brake_steps=reserve["steps"],
                     proof_renewal="global_joint_box", feedback_age_ms=(now-feedback[1])*1000,
                     control_latency_ms=(time.perf_counter()-control_started)*1000,
                     certificate_age_ms=(time.perf_counter()-feedback[1])*1000,
                     adopted_action=adopted.tolist(), gdk_command_accepted=self._last_command_accepted)
        self._safety_logger.write(proof, audit)
        return adopted

    def _select_async_hold(self, now):
        held = self._async_hold
        if held is None or not np.array_equal(held["action"], self._last_state):
            raise SafetyDispatchExpiredError("No certified hold for the last accepted target")
        if held["result"].get("sampled_snapshot") and self._async_ever_dispatched:
            feedback = self._async_worker.feedback
            now = time.perf_counter()
            if (feedback is None or not 0 <= now-feedback[1] < self.config.stationary_hold_feedback_max_age_s
                    or np.asarray(feedback[0]).shape != (24,) or not np.isfinite(feedback[0]).all()):
                raise SafetyDispatchExpiredError("Feedback missing or expired during historical target hold")
            check_age = (float("inf") if self._async_last_check_at is None
                         else now-self._async_last_check_at)
            if not 0 <= check_age < .2:
                # A late rolling-window result must not turn an already accepted,
                # stationary target into a fatal session error.  Repeating this
                # exact target cannot advance the robot along unchecked motion.
                # Permit a short, bounded bridge only while an exact-anchor
                # recheck is actually pending and feedback remains fresh.  New
                # motion still requires the returned geometric certificate.
                pending = self._async_pending
                exact_recheck_pending = (
                    pending is not None
                    and (
                        np.array_equal(pending.get("anchor"), self._last_state)
                        or np.array_equal(pending.get("fallback_anchor"), self._last_state)
                    )
                )
                worker_failed = bool(
                    getattr(self._async_worker, "error", None)
                    or getattr(self._async_worker, "feedback_error", None)
                )
                if not exact_recheck_pending or worker_failed or not 0 <= check_age < .5:
                    raise SafetyDispatchExpiredError(
                        "Collision worker unavailable during historical target hold"
                    )
                limiter = self._kinematic_limiter.fork()
                limiter.reset(self._last_state)
                return dict(
                    held,
                    step=-1,
                    limiter=limiter,
                    historical_repeat=True,
                    recheck_bridge=True,
                    # Dispatch-age validation separately requires fresh robot
                    # feedback.  This timestamp authorizes only an identical
                    # hold command, never a trajectory waypoint.
                    dispatch_authority_at=feedback[1],
                )
            limiter = self._kinematic_limiter.fork()
            limiter.reset(self._last_state)
            return dict(held, step=-1, limiter=limiter, historical_repeat=True,
                        dispatch_authority_at=self._async_last_check_at)
        if held["hold_proof"] is None:
            raise SafetyDispatchExpiredError(
                "Last accepted stride waypoint was not sampled; no checked hold available, command streaming stopped")
        selected = dict(held, step=-1)
        # A held target has zero planner velocity. Do not reinstall the velocity
        # snapshot captured when this target was still part of moving motion.
        selected["limiter"] = self._kinematic_limiter.fork()
        selected["limiter"].reset(self._last_state)
        selected.pop("hold_checked_at", None)
        selected.pop("hold_feedback", None)
        if held["result"].get("sampled_snapshot"):
            result = self._renew_rolling_result(held["result"], now)
            if result is None or held["hold_proof"].hard_limit_triggered:
                raise SafetyDispatchExpiredError(
                    f"Sampled hold unavailable: {self._async_renew_failure}; command streaming stopped")
            selected["result"] = result
            return selected
        feedback = getattr(self._async_worker, "feedback", None)
        now = time.perf_counter()
        feedback_age_ms = None if feedback is None else (now - feedback[1]) * 1000
        renewal_reason = "feedback_missing" if feedback is None else "feedback_expired"
        if feedback is not None and 0 <= (now - feedback[1]) * 1000 < RESULT_MAX_AGE_MS:
            renewal_reason = "hold_displacement_bound_rejected"
            plane = max(s.height + s.hard_margin for s in self._safety_layer.surfaces.surfaces)
            if held["result"]["task"].get("rolling_legacy_geometry"):
                certificate = held.get("sampled_hold_feedback")
                allowed = certificate is not None and np.isfinite(sampled_feedback_displacement(
                    certificate, feedback[0], self._async_hold_coefficients))
                renewal_reason = "sampled_hold_feedback_bound_rejected"
            else:
                allowed = bounded_hold_is_safe(self._async_hold_coefficients, held["result"]["measured"],
                                               feedback[0], held["hold_proof"].path_z_min, plane,
                                               path_surface_clearance=(
                                                   held["hold_proof"].path_surface_clearance
                                               ))
            if allowed:
                selected["hold_checked_at"] = feedback[1]
                selected["hold_feedback"] = feedback[0]
        checked_at = selected.get("hold_checked_at", held["result"]["started"])
        if (now - checked_at) * 1000 >= RESULT_MAX_AGE_MS:
            pending = self._async_pending
            bound = None
            if feedback is not None and self._async_hold_coefficients is not None:
                bound = float(np.dot(self._async_hold_coefficients,
                                     np.abs(feedback[0] - held["result"]["measured"])))
            raise SafetyDispatchExpiredError(
                "Checked hold expired; no fresh safety result, command streaming stopped; "
                f"hold_age_ms={(now - checked_at) * 1000:.3f} "
                f"feedback_age_ms={feedback_age_ms} renewal={renewal_reason} "
                f"hold_path_z_min={held['hold_proof'].path_z_min} displacement_bound_m={bound} "
                f"generation={self._async_generation} "
                f"pending_request_id={None if pending is None else pending['request_id']}"
            )
        return selected

    def _consume_certified_step(
        self, model_candidate: np.ndarray, safety_context: dict[str, Any] | None,
        control_started: float,
    ) -> np.ndarray | None:
        step = self._certified_step
        self._certified_step = None  # Single-use, including every failure path.
        if step is None:
            return None
        if (time.perf_counter() - step["started"]) * 1000 > 20.0:
            return None
        if not np.array_equal(self._last_state, step["first"]):
            return None
        measured = self.read_state()
        if measured.shape != (ACTION_DIM,) or not np.isfinite(measured).all():
            raise ValueError("Invalid feedback while consuming checked trajectory")
        # Cheap drift fence, not a substitute for geometric collision checking.
        # Feedback outside the checked joint-space envelope triggers a new check.
        poses = np.stack((step["measured"], step["first"], step["action"]))
        tolerance = np.full(ACTION_DIM, 1e-3, dtype=np.float32)
        tolerance[list(EFFECTOR_CANONICAL_INDICES)] = 1e-4
        if np.any(measured < poses.min(axis=0) - tolerance) or np.any(
            measured > poses.max(axis=0) + tolerance
        ):
            return None
        if (time.perf_counter() - step["started"]) * 1000 > 20.0:
            return None
        self._command_safety_started = step["started"]
        self._certified_dispatch = True
        self._kinematic_limiter = step["limiter"]
        limited = self._dispatch_checked_action(step["action"].copy())
        self._last_state = limited.copy()
        # Keep runner conditioning anchored to the frozen, actually adopted
        # target, rather than the new input that was deliberately not sent.
        self.last_action_intervened = True
        decision = step["decision"]
        response = "checked_segment_hold" if decision.hard_limit_triggered else "checked_segment_continue"
        self._capture_control_telemetry(
            measured=measured, model_candidate=model_candidate,
            controller_candidate=step["action"], adopted_action=limited,
            safety_decision=decision, safety_guard_latency_ms=None,
            collision_response=response,
        )
        context = dict(step["context"])
        for private_key in ("raw_action", "controller_target", "preview_control_targets"):
            context.pop(private_key, None)
        context.update(
            guard_evaluated=False, trajectory_step=1, safety_check_fps=50,
            certificate_age_ms=(time.perf_counter() - step["started"]) * 1000,
            control_latency_ms=(time.perf_counter() - control_started) * 1000,
            safety_guard_latency_ms=None, collision_response=response,
            adopted_action=limited.tolist(), gdk_command_accepted=self._last_command_accepted,
        )
        assert self._safety_logger is not None
        self._safety_logger.write(decision, context)
        return limited

    def _dispatch_checked_action(self, limited: np.ndarray) -> np.ndarray:
        if self.config.hardware_mode == "read-only":
            if self._async_dispatch_action is not None:
                self._pace_servo_dispatch()
        elif self.config.hardware_mode == "dry-run":
            self._dry_state = limited.copy()
        else:
            self._last_command_accepted = False
            try:
                self._send_combined_action(limited)
            except (SafetyDispatchExpiredError, ServoInterfaceError):
                self._certified_step = None
                # Not a recoverable GDK rejection. No expired candidate may
                # reach hardware, even if the pacing wait caused the expiry.
                raise
            except Exception as error:
                self._certified_step = None
                self._consecutive_gdk_rejections += 1
                consecutive = self._consecutive_gdk_rejections
                maximum = self.config.max_consecutive_gdk_rejections
                if consecutive >= maximum:
                    logger.exception(
                        "GDK rejected JointServoControlReq %d consecutive times; "
                        "stopping physical command streaming targets=%s",
                        consecutive,
                        dict(
                            zip(
                                GDK_BODY_JOINTS
                                + (self._effector_names["left"], self._effector_names["right"]),
                                canonical_to_gdk_servo_positions(
                                    self._control_targets_to_gdk_commands(limited)
                                ).tolist(),
                                strict=True,
                            )
                        ),
                    )
                    raise RuntimeError(
                        f"JointServoControl failed {consecutive} consecutive times"
                    ) from error

                held = self._last_successful_action
                if held is None:
                    held = self._last_state
                if held is None:
                    raise RuntimeError(
                        "JointServoControl failed before a safe hold action was available"
                    ) from error
                limited = self._apply_absolute_position_limits(
                    held, warning_suffix="rejection_hold"
                )
                self._kinematic_limiter.reset(limited)
                # The runner's filters must follow what was actually adopted,
                # including transient GDK rejection, not just collision holds.
                self.last_action_intervened = True
                self._rate_limited_warning(
                    "gdk_servo_rejection",
                    "GDK rejected JointServoControlReq (%d/%d): holding the last successfully "
                    "sent action and continuing; error=%s",
                    consecutive,
                    maximum,
                    error,
                )
            else:
                self._last_command_accepted = True
                if self._consecutive_gdk_rejections:
                    logger.info(
                        "GDK JointServoControlReq recovered after %d consecutive rejection(s)",
                        self._consecutive_gdk_rejections,
                    )
                self._consecutive_gdk_rejections = 0
                self._last_successful_action = limited.copy()
        if self.config.hardware_mode != "execute" or self._last_command_accepted is True:
            self._dispatch_sequence = getattr(self, "_dispatch_sequence", 0) + 1
            self._last_dispatched_action = np.asarray(limited, dtype=np.float32).copy()
            feedback = getattr(self, "_last_feedback_state", None)
            self._last_dispatch_feedback = None if feedback is None else feedback.copy()
            self._last_dispatch_feedback_timestamp_ns = getattr(
                self, "_last_feedback_timestamp_ns", None
            )
            feedback_at = getattr(self, "_last_feedback_monotonic", None)
            self._last_dispatch_feedback_age_ms = (
                None if feedback_at is None
                else max(0.0, (time.monotonic() - feedback_at) * 1000.0)
            )
        return limited

    def dispatch_receipt(self) -> dict[str, Any] | None:
        """Return this send call's accepted dispatch, never a safety submission.

        The caller must query after ``send_action`` returns on the sole action
        thread.  Queue provenance describes the selected certified point and
        may refer to an earlier request than the current method argument.
        """
        sequence = getattr(self, "_dispatch_sequence", 0)
        if sequence <= getattr(self, "_send_dispatch_start_sequence", sequence):
            return None
        action = getattr(self, "_last_dispatched_action", None)
        if action is None:
            return None
        return {
            "sequence": sequence,
            "action": action.copy(),
            "physical_accepted": (
                self.config.hardware_mode == "execute"
                and self._last_command_accepted is True
            ),
            "virtual": self.config.hardware_mode != "execute",
            "measured_state": (
                None if getattr(self, "_last_dispatch_feedback", None) is None
                else self._last_dispatch_feedback.copy()
            ),
            "measured_timestamp_ns": getattr(
                self, "_last_dispatch_feedback_timestamp_ns", None
            ),
            "measured_age_ms": getattr(self, "_last_dispatch_feedback_age_ms", None),
            "provenance": (
                None if self._last_dispatch_provenance is None
                else {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in self._last_dispatch_provenance.items()
                }
            ),
        }

    @property
    def last_delivery_evidence(self) -> DeliveryEvidence | None:
        """Latest accepted command/feedback pair for legacy hold handoff."""
        action = getattr(self, "_last_dispatched_action", None)
        measured = getattr(self, "_last_dispatch_feedback", None)
        timestamp_ns = getattr(self, "_last_dispatch_feedback_timestamp_ns", None)
        if action is None or measured is None or timestamp_ns is None:
            return None
        return DeliveryEvidence(
            measured_timestamp_ns=int(timestamp_ns),
            measured_state=measured.copy(),
            adopted_action=action.copy(),
            command_accepted=self.config.hardware_mode == "execute",
        )

    def command_motion_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the last accepted command and final-limiter velocity."""
        if self._last_state is None:
            raise RuntimeError("Command motion has no accepted GDK anchor")
        return self._last_state.copy(), self._kinematic_limiter.velocity

    def disconnect(self) -> None:
        image_executor = getattr(self, "_wrist_image_executor", None)
        if image_executor is not None:
            image_executor.shutdown(wait=True, cancel_futures=True)
            self._wrist_image_executor = None
        self._certified_step = None
        if self._async_worker is not None:
            self._async_worker.close()
            self._async_worker = None
        self._invalidate_async_safety(clear_hold=True)
        if not self._connected:
            self._process_lock.release()
            self._close_safety_resources()
            return
        if self.config.hardware_mode != "dry-run":
            try:
                if self._camera is not None and hasattr(self._camera, "close_camera"):
                    self._camera.close_camera()
            except Exception:
                logger.warning("Failed to close GDK cameras", exc_info=True)
            finally:
                try:
                    if self._gdk_initialized:
                        self._gdk.gdk_release()
                except Exception:
                    logger.warning("Failed to release GDK", exc_info=True)
                finally:
                    self._gdk_initialized = False
                    self._process_lock.release()
        self._robot = None
        self._camera = None
        self._connected = False
        self._execution_prepared = False
        self._close_safety_resources()

    def _close_safety_resources(self) -> None:
        if self._safety_resources_closed:
            return
        self._safety_resources_closed = True
        if self._async_worker is not None:
            self._async_worker.close()
            self._async_worker = None
        controller_error: BaseException | None = None
        if self._safety_controller is not None:
            try:
                self._safety_controller.close()
            except BaseException as error:
                controller_error = error
        if self._safety_logger is not None:
            self._safety_logger.close()
        if controller_error is not None:
            raise RuntimeError("Failed to stop detailed safety projector") from controller_error

    def model_actions_to_control_targets(self, actions: np.ndarray) -> np.ndarray:
        """Convert policy effector semantics into physical GDK coordinates."""

        return self.effector_profile.model_actions_to_hardware_targets(actions)

    def control_targets_to_model_actions(self, targets: np.ndarray) -> np.ndarray:
        """Convert feedback-coordinate targets back to policy/GDK command semantics."""

        return self.effector_profile.hardware_targets_to_model_actions(targets)

    def _control_targets_to_gdk_commands(self, targets: np.ndarray) -> np.ndarray:
        """Serialize internal targets in the command coordinates expected by GDK.

        Arm, head, and waist targets already use GDK joint coordinates.  DH
        gripper targets intentionally do not: the realtime controller keeps
        them in ``get_end_state`` opening-width coordinates so filtering,
        limiting, tracking, and collision telemetry all compare like with
        like.  The installed GDK accepts the opposite endpoint convention for
        the gripper joint command, identical to the policy action convention,
        so the inverse profile mapping must be applied exactly once here.
        """

        return self.control_targets_to_model_actions(targets)

    def _read_body_positions(self, *, read_started_monotonic: float | None = None) -> dict[str, float]:
        self._mark_state_read_stage("body_gdk")
        started = time.perf_counter() if read_started_monotonic is None else read_started_monotonic
        response = self._robot.get_joint_states()
        self._mark_state_read_stage("body_decode")
        states = response.get("states", [])
        positions: dict[str, float] = {}
        motor_velocities: dict[str, float] = {}
        for state in states:
            name = state.get("name")
            value = state.get("motor_position", state.get("position"))
            if name is not None and value is not None:
                positions[str(name)] = float(value)
            velocity = state.get("motor_velocity")
            if name is not None and velocity is not None:
                try:
                    velocity_value = float(velocity)
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(velocity_value):
                        motor_velocities[str(name)] = velocity_value
        now = time.monotonic()
        noncontrol = set(HEAD_JOINTS + WAIST_JOINTS)
        recovered: list[str] = []
        with self._body_position_cache_lock:
            if (self._last_motor_velocity_snapshot is None
                    or started >= self._last_motor_velocity_snapshot[0]):
                self._last_motor_velocity_snapshot = (
                    started,
                    tuple(motor_velocities.get(name) for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS),
                )
            for name, value in positions.items():
                self._body_position_cache[name] = (value, now)
            for name in GDK_BODY_JOINTS:
                if name in positions or name not in noncontrol:
                    continue
                cached = self._body_position_cache.get(name)
                if cached is not None and now - cached[1] <= self.config.noncontrol_joint_state_grace_s:
                    positions[name] = cached[0]
                    recovered.append(name)
        if recovered:
            self._rate_limited_log(
                logging.WARNING, "transient_noncontrol_joint_feedback",
                "GDK omitted non-controlled joints; reused <=%.0fms cached feedback: %s",
                self.config.noncontrol_joint_state_grace_s * 1000, recovered,
            )
        missing = [name for name in GDK_BODY_JOINTS if name not in positions]
        if missing:
            raise RuntimeError(f"GDK joint state is missing required joints: {missing}")
        return positions

    def _read_effector(self, side: str, end_state: dict[str, Any]) -> float:
        side_state = end_state.get(f"{side}_end_state")
        if not isinstance(side_state, dict) or not side_state.get("end_states"):
            raise RuntimeError(f"GDK end state is missing {side}_end_state")
        reported_names = side_state.get("names")
        detected_name: str | None = None
        if isinstance(reported_names, str) and reported_names:
            detected_name = reported_names
        elif isinstance(reported_names, (list, tuple)) and reported_names:
            detected_name = str(reported_names[0])
        if self._effector_name_overrides[side] is None and detected_name:
            self._effector_names[side] = detected_name
        return float(side_state["end_states"][0]["position"])

    def _camera_type_map(self) -> dict[str, Any]:
        return {
            "cam_head": self._gdk.CameraType.kHeadColor,
            "cam_left_wrist": self._gdk.CameraType.kHandLeftColor,
            "cam_right_wrist": self._gdk.CameraType.kHandRightColor,
        }

    def _capture_encoded_images(
        self, *, before_processing: Callable[[], None] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        acquisition_started = time.perf_counter()
        # Acquire all three GDK frames before doing any JPEG decode/resize work.
        # The head frame is the temporal anchor and the two wrist frames are
        # selected from GDK's timestamped buffers.  A short retry is required
        # because the three camera publications can become visible a few
        # milliseconds apart even when they belong to the same capture tick.
        camera_types = self._camera_type_map()
        anchor = self._camera.get_latest_image(
            camera_types["cam_head"], self.config.camera_timeout_ms
        )
        if anchor is None:
            raise RuntimeError("No image received from GDK camera cam_head")
        anchor_timestamp_ns = int(anchor.timestamp_ns)
        # GDK camera/H5 timestamps can use a different epoch from G2's
        # CLOCK_REALTIME. A latest-frame receipt gives a bounded bridge; take
        # the smallest observed offset during warmup to reduce frame-age bias.
        observed_offset_ns = time.time_ns() - anchor_timestamp_ns
        previous_offset_ns = self._camera_wall_to_vendor_offset_ns
        if self._camera_clock_offset_frozen:
            if previous_offset_ns is None or abs(observed_offset_ns - previous_offset_ns) > 100_000_000:
                raise RuntimeError("GDK camera clock bridge changed by more than 100 ms")
        elif previous_offset_ns is None or observed_offset_ns < previous_offset_ns:
            self._camera_wall_to_vendor_offset_ns = observed_offset_ns
        selected: dict[str, Any] = {"cam_head": anchor}
        pending = {"cam_left_wrist", "cam_right_wrist"}
        tolerance_ns = int(self.config.camera_sync_tolerance_ms * 1e6)
        deadline = time.perf_counter() + self.config.camera_sync_timeout_ms / 1000.0
        last_skew_ns: dict[str, int] = {}
        attempted: set[str] = set()
        while pending:
            for key in tuple(pending):
                # Always make one immediate lookup per wrist, then keep every
                # blocking call inside the one shared synchronization budget.
                if key in attempted and time.perf_counter() >= deadline:
                    continue
                remaining_ms = max(0.0, (deadline - time.perf_counter()) * 1000.0)
                nearest_timeout_ms = max(
                    1,
                    min(self.config.camera_timeout_ms, math.ceil(remaining_ms)),
                )
                image = self._camera.get_nearest_image(
                    camera_types[key], anchor_timestamp_ns, nearest_timeout_ms
                )
                attempted.add(key)
                if image is None:
                    last_skew_ns.pop(key, None)
                    continue
                skew_ns = abs(int(image.timestamp_ns) - anchor_timestamp_ns)
                last_skew_ns[key] = skew_ns
                if skew_ns <= tolerance_ns:
                    selected[key] = image
                    pending.remove(key)
            if not pending:
                break
            if time.perf_counter() >= deadline:
                details = {
                    key: (
                        None
                        if key not in last_skew_ns
                        else round(last_skew_ns[key] / 1e6, 3)
                    )
                    for key in sorted(pending)
                }
                raise RuntimeError(
                    "GDK camera synchronization timed out: "
                    f"anchor_timestamp_ns={anchor_timestamp_ns}, "
                    f"tolerance_ms={self.config.camera_sync_tolerance_ms}, "
                    f"last_skew_ms={details}"
                )
            if self.config.camera_sync_poll_ms:
                time.sleep(self.config.camera_sync_poll_ms / 1000.0)

        acquisition_finished = time.perf_counter()
        payloads: dict[str, dict[str, Any]] = {}
        timestamps: dict[str, int] = {}
        for key in camera_types:
            image = selected[key]
            if image.encoding == self._gdk.Encoding.JPEG:
                encoding = "jpeg"
                color_format = "encoded"
            elif image.encoding == self._gdk.Encoding.UNCOMPRESSED:
                encoding = "raw"
                if image.color_format == self._gdk.ColorFormat.RGB:
                    color_format = "rgb"
                elif image.color_format == self._gdk.ColorFormat.BGR:
                    color_format = "bgr"
                else:
                    raise RuntimeError(f"Unsupported GDK color format: {image.color_format!r}")
            else:
                raise RuntimeError(f"Unsupported GDK image encoding: {image.encoding!r}")
            payload = {
                "__g2a_encoded_image__": True,
                "encoding": encoding,
                "color_format": color_format,
                "width": int(image.width),
                "height": int(image.height),
                "data": bytes(image.data),
            }
            payloads[key] = payload
            timestamps[key] = int(image.timestamp_ns)
        # The FlowPRO state anchor must be sampled next to the selected frame
        # timestamps. Wrist JPEG resize can take longer than the 40 ms join
        # limit, so sample before that CPU work begins.
        if before_processing is not None:
            before_processing()
        processing_started = time.perf_counter()
        images, transport_timing = self._prepare_camera_payloads(payloads)
        processing_finished = time.perf_counter()
        self._last_image_transport_timing = transport_timing
        self._last_camera_capture_timing = {
            "acquisition_ms": (acquisition_finished - acquisition_started) * 1000,
            "payload_copy_ms": (processing_started - acquisition_finished) * 1000,
            "processing_wall_ms": (processing_finished - processing_started) * 1000,
        }
        return images, timestamps

    def _prepare_camera_payloads(
        self, payloads: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Process copied bytes only; the JPEG worker never accesses native GDK.

        One worker plus the calling camera thread bounds processing to two
        concurrent images. Pixels, codec settings, timestamps and frame
        synchronization are identical to the serial path. No frame cache.
        """
        def prepare(key: str) -> tuple[dict[str, Any], dict[str, Any]]:
            payload = payloads[key]
            if self.config.resize_wrist_images and key in ("cam_left_wrist", "cam_right_wrist"):
                return resize_image_payload_for_transport(
                    payload, width=self.config.wrist_image_width,
                    height=self.config.wrist_image_height,
                    jpeg_quality=self.config.wrist_jpeg_quality,
                )
            return payload, _passthrough_image_timing(payload)

        if self.config.parallel_wrist_images and self.config.resize_wrist_images:
            if self._wrist_image_executor is None:
                self._wrist_image_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="WristJPEG"
                )
            left = self._wrist_image_executor.submit(prepare, "cam_left_wrist")
            try:
                right = prepare("cam_right_wrist")
                head = prepare("cam_head")
                prepared = {"cam_head": head, "cam_left_wrist": left.result(),
                            "cam_right_wrist": right}
            finally:
                # Even if either image is malformed, drain this capture before
                # another capture/teardown can reuse its worker. Fail closed.
                wait([left])
        else:
            prepared = {key: prepare(key) for key in payloads}
        return ({key: value[0] for key, value in prepared.items()},
                {key: value[1] for key, value in prepared.items()})

    def _apply_action_checks(self, target: np.ndarray) -> np.ndarray:
        checked = self._apply_absolute_position_limits(target, warning_suffix="input")
        grippers = (LEFT_EFFECTOR_INDEX, RIGHT_EFFECTOR_INDEX)
        current = self._last_state
        if current is not None and self.config.max_relative_target is not None:
            delta = self.config.max_relative_target
            exceeded = np.abs(checked - current) > delta
            exceeded[list(grippers)] = False
            if exceeded.any():
                if self.config.relative_target_mode == "clip":
                    checked = np.clip(checked, current - delta, current + delta)
                else:
                    self._rate_limited_action_diagnostic(
                        "body_relative_target",
                        "Body relative threshold %.4f exceeded without clipping: indices=%s max=%.4f",
                        delta,
                        np.flatnonzero(exceeded).tolist(),
                        float(np.max(np.abs(checked[exceeded] - current[exceeded]))),
                    )
        if current is not None and self.config.max_gripper_relative_target is not None:
            delta = self.config.max_gripper_relative_target
            for index in grippers:
                difference = float(checked[index] - current[index])
                if abs(difference) <= delta:
                    continue
                if self.config.relative_target_mode == "clip":
                    checked[index] = np.clip(checked[index], current[index] - delta, current[index] + delta)
                else:
                    self._rate_limited_action_diagnostic(
                        f"gripper_relative_target_{index}",
                        "Gripper relative threshold %.4f exceeded without clipping: index=%d delta=%.4f",
                        delta,
                        index,
                        difference,
                    )
        return checked.astype(np.float32, copy=False)

    def _apply_absolute_position_limits(
        self,
        target: np.ndarray,
        *,
        warning_suffix: str,
    ) -> np.ndarray:
        """Enforce SDK-valid absolute ranges on an intermediate or final target."""
        checked = np.asarray(target, dtype=np.float32).copy()
        grippers = (LEFT_EFFECTOR_INDEX, RIGHT_EFFECTOR_INDEX)
        requested_grippers = checked[list(grippers)].copy()
        gripper_min, gripper_max = self.effector_profile.hardware_state_range
        checked[list(grippers)] = np.clip(
            checked[list(grippers)],
            gripper_min,
            gripper_max,
        )
        # Always clip exactly. Suppress only sub-10-nanometre numerical noise
        # in the warning stream; this tolerance never expands the valid range.
        if np.any(np.abs(requested_grippers - checked[list(grippers)]) > 1e-8):
            self._rate_limited_warning(
                f"gripper_physical_range_{warning_suffix}",
                "Clipped gripper physical range (%s): requested=%s sent=%s",
                warning_suffix,
                requested_grippers,
                checked[list(grippers)],
            )

        # GDK rejects the complete request if one body target lies outside its
        # absolute range, so this check is mandatory both before and after the
        # stateful kinematic limiter.
        margin = self.config.gdk_position_limit_margin
        lower = _BODY_LIMIT_LOWER + margin
        upper = _BODY_LIMIT_UPPER - margin
        invalid = np.flatnonzero(lower > upper)
        if invalid.size:
            index = int(invalid[0])
            name = GDK_BODY_JOINTS[index]
            raise ValueError(
                f"GDK position limit margin {margin} is too large for {name}: "
                f"[{_BODY_LIMIT_LOWER[index]}, {_BODY_LIMIT_UPPER[index]}]"
            )
        requested = checked[_BODY_LIMIT_INDICES].astype(np.float64)
        clipped = np.clip(requested, lower, upper)
        changed = np.flatnonzero(clipped != requested)
        if changed.size:
            checked[_BODY_LIMIT_INDICES[changed]] = clipped[changed]
            hard_limit_violations = [
                f"{GDK_BODY_JOINTS[index]}={requested[index]:.6f}->"
                f"{clipped[index]:.6f}[{lower[index]:.6f},{upper[index]:.6f}]"
                for index in changed
            ]
            self._rate_limited_action_diagnostic(
                f"gdk_hard_position_limit_{warning_suffix}",
                "Clipped GDK hard position limits (%s): %s",
                warning_suffix,
                "; ".join(hard_limit_violations),
            )
        return checked

    def _rate_limited_warning(self, key: str, message: str, *args: Any) -> None:
        """Log a high-rate servo warning at most once per second."""
        self._rate_limited_log(logging.WARNING, key, message, *args)

    def _rate_limited_action_diagnostic(
        self, key: str, message: str, *args: Any
    ) -> None:
        """Keep virtual-action shaping visible without alarming read-only users."""
        if self.config.hardware_mode == "read-only":
            self._rate_limited_log(
                logging.INFO,
                key,
                "Read-only virtual action diagnostic (no command sent): " + message,
                *args,
            )
            return
        self._rate_limited_warning(key, message, *args)

    def _rate_limited_log(
        self, level: int, key: str, message: str, *args: Any
    ) -> None:
        """Log a high-rate servo message at most once per second."""
        now = time.monotonic()
        last = self._warning_last_time.get(key)
        if last is not None and now - last < 1.0:
            self._warning_suppressed[key] = self._warning_suppressed.get(key, 0) + 1
            return
        suppressed = self._warning_suppressed.pop(key, 0)
        self._warning_last_time[key] = now
        if suppressed:
            logger.log(level, "%s (suppressed %d similar messages)", message % args, suppressed)
        else:
            logger.log(level, message, *args)

    def _send_combined_action(self, target: np.ndarray) -> None:
        if self._async_dispatch_action is not None and not np.array_equal(target, self._async_dispatch_action):
            raise SafetyDispatchExpiredError("Dispatch target differs from the certified action")
        effector_names = (self._effector_names["left"], self._effector_names["right"])
        if not all(effector_names):
            raise RuntimeError(f"Could not resolve both effector joint names: {effector_names}")
        gdk_commands = self._control_targets_to_gdk_commands(target)
        request = self._gdk.JointServoControlReq()
        request.joint_names = list(GDK_BODY_JOINTS + effector_names)
        request.joint_positions = canonical_to_gdk_servo_positions(gdk_commands).tolist()
        request.control_period = self.config.servo_control_period_s
        if hasattr(request, "joint_velocities"):
            request.joint_velocities = [0.0] * len(request.joint_names)
        if hasattr(request, "joint_accelerations"):
            request.joint_accelerations = [0.0] * len(request.joint_names)
        use_keyword = self._servo_keyword_supported(request)
        self._pace_servo_dispatch()
        # Exclude the 100 Hz pacing wait: SDK call latency is a separate cause.
        call_started = self._last_servo_call_started = time.perf_counter()
        try:
            if use_keyword:
                result = self._robot.joint_servo_control(
                    request, enable_low_latency=self.config.enable_low_latency,
                )
            else:
                result = self._robot.joint_servo_control(request)
        except TypeError as error:
            # This may be an error AFTER the vendor function has started.
            # Never guess that it was harmless argument binding and resend.
            raise ServoInterfaceError(
                "GDK servo raised TypeError; refusing an ambiguous command retry. "
                "Check the installed GDK signature and original exception."
            ) from error
        finally:
            self._last_servo_call_ms = (time.perf_counter() - call_started) * 1000
        self._check_result("joint_servo_control", result)

    def _servo_keyword_supported(self, request: Any) -> bool:
        if self._servo_signature_robot is not self._robot:
            method = self._robot.joint_servo_control
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                # Some native bindings expose no signature. Use the configured
                # current API exactly once; an error must fail closed.
                uses_keyword = True
            else:
                try:
                    signature.bind(request, enable_low_latency=self.config.enable_low_latency)
                except TypeError:
                    try:
                        signature.bind(request)
                    except TypeError as error:
                        raise ServoInterfaceError("Unsupported GDK servo signature") from error
                    uses_keyword = False
                else:
                    uses_keyword = True
            self._servo_uses_latency_keyword = uses_keyword
            self._servo_signature_robot = self._robot
        return self._servo_uses_latency_keyword

    def _wait_servo_slot(self, *, lead_s: float = 0.0) -> None:
        if self._last_servo_dispatch is not None:
            deadline = self._last_servo_dispatch + self.config.servo_control_period_s - lead_s
            while (remaining := deadline - time.perf_counter()) > 0:
                time.sleep(remaining)

    def _pace_servo_dispatch(self) -> None:
        self._wait_servo_slot()
        now = time.perf_counter()
        self._validate_dispatch_age(now)
        # Do not reset this clock on collision hold, soft pause, or rejection.
        self._last_servo_dispatch = now

    def _validate_dispatch_age(self, now: float) -> None:
        sampled = (self.config.rolling_safety_enabled and self.config.rolling_legacy_geometry
                   and self._async_dispatch_action is not None)
        if sampled and self.config.safety_enforcement_enabled:
            feedback = self._async_worker.feedback
            now = time.perf_counter()
            feedback_limit = self.config.stationary_hold_feedback_max_age_s if self._async_dispatch_is_hold else .04
            if (feedback is None or not 0 <= now-feedback[1] < feedback_limit
                    or np.asarray(feedback[0]).shape != (24,) or not np.isfinite(feedback[0]).all()):
                raise SafetyDispatchExpiredError("Feedback missing, invalid or expired before dispatch")
            self._dispatch_feedback_age_ms = (now-feedback[1])*1000
        if (
            self.config.safety_enforcement_enabled
            and self._command_safety_started is not None
            and (now - self._command_safety_started) * 1000
            > (200.0 if sampled else RESULT_MAX_AGE_MS if self._async_dispatch_action is not None else
               20.0 if self._certified_dispatch else self.config.max_safety_guard_latency_ms)
        ):
            raise SafetyDispatchExpiredError(
                "Safety decision expired before GDK dispatch (including pacing wait); "
                "physical command streaming stopped without sending the expired target"
            )

    def _configure_current_pose(self) -> None:
        positions = self._read_body_positions()
        request = self._gdk.JointControlReq()
        request.life_time = max(0.1, self.config.servo_control_period_s)
        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        if self.config.initialization_scope == "24d":
            names += WAIST_JOINTS
        elif self.config.initialization_scope != "16d":
            raise ValueError(
                f"Unsupported GDK initialization scope: {self.config.initialization_scope!r}"
            )
        request.joint_names = list(names)
        request.joint_positions = [positions[name] for name in names]
        request.joint_velocities = [0.3] * len(names)
        self._check_result("joint_control_request", self._robot.joint_control_request(request))
        if self.config.initialization_scope == "24d":
            self._initialize_head_at_current_pose(positions)
        if self.config.initialization_settle_s:
            time.sleep(self.config.initialization_settle_s)
        logger.info(
            "GDK current-pose initialization completed scope=%s joints=%d settle=%.2fs",
            self.config.initialization_scope,
            len(names),
            self.config.initialization_settle_s,
        )

    def _initialize_head_at_current_pose(self, positions: dict[str, float]) -> None:
        attempts = self.config.initialization_attempts
        if attempts < 1:
            raise ValueError("GDK initialization attempts must be at least 1")
        head_positions = [positions[name] for name in HEAD_JOINTS]
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = self._robot.move_head_joint(head_positions, [0.2] * len(HEAD_JOINTS))
                self._check_result("move_head_joint", result)
                if attempt > 1:
                    logger.info("GDK head current-pose initialization succeeded on attempt %d", attempt)
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    "GDK head current-pose initialization attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    error,
                )
                if attempt < attempts and self.config.initialization_retry_delay_s:
                    time.sleep(self.config.initialization_retry_delay_s)
        raise RuntimeError(
            f"GDK head current-pose initialization failed after {attempts} attempts"
        ) from last_error

    def _check_result(self, operation: str, result: Any) -> None:
        if result is not None and result != self._gdk.GDKRes.kSuccess and result != 0:
            raise RuntimeError(f"{operation} failed: {result!r}")

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("G2A adapter is not connected")


def canonical_to_gdk_servo_positions(target: np.ndarray) -> np.ndarray:
    """Map canonical 24D to GDK body22 + left/right effector joint order."""
    target = np.asarray(target)
    if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
        raise ValueError(f"Expected one finite canonical 24D target, got {target.shape}")
    return np.concatenate(
        (
            target[:7],
            target[8:15],
            target[16:24],
            target[[LEFT_EFFECTOR_INDEX, RIGHT_EFFECTOR_INDEX]],
        )
    )
