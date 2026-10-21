#!/usr/bin/env python3
"""Safely move G2A body joints toward the packaged fixed reset profile."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .constants import (
        ACTION_DIM,
        ARM_CANONICAL_INDICES,
        BODY_RESET_PROFILE_ID,
        BODY_RESET_TARGET_24D,
        CANONICAL_STATE_NAMES,
        DEFAULT_EFFECTOR_JOINT_NAMES,
        DH_GRIPPER_CLOSED_POSITION,
        DH_GRIPPER_OPEN_POSITION,
        EFFECTOR_CANONICAL_INDICES,
        GDK_BODY_JOINT_LIMITS,
        GDK_BODY_JOINTS,
        GRIPPER_MAX_POSITION,
        GRIPPER_MIN_POSITION,
        HEAD_CANONICAL_INDICES,
        HEAD_JOINTS,
        LEFT_ARM_JOINTS,
        RIGHT_ARM_JOINTS,
        WAIST_CANONICAL_INDICES,
        WAIST_JOINTS,
    )
    from .gdk_adapter import canonical_to_gdk_servo_positions
    from .robot_lock import RobotProcessLock
    from .safety_layer import ActionSafetyLayer, load_safety_layer
else:
    from robot_client.constants import (
        ACTION_DIM,
        ARM_CANONICAL_INDICES,
        BODY_RESET_PROFILE_ID,
        BODY_RESET_TARGET_24D,
        CANONICAL_STATE_NAMES,
        DEFAULT_EFFECTOR_JOINT_NAMES,
        DH_GRIPPER_CLOSED_POSITION,
        DH_GRIPPER_OPEN_POSITION,
        EFFECTOR_CANONICAL_INDICES,
        GDK_BODY_JOINT_LIMITS,
        GDK_BODY_JOINTS,
        GRIPPER_MAX_POSITION,
        GRIPPER_MIN_POSITION,
        HEAD_CANONICAL_INDICES,
        HEAD_JOINTS,
        LEFT_ARM_JOINTS,
        RIGHT_ARM_JOINTS,
        WAIST_CANONICAL_INDICES,
        WAIST_JOINTS,
    )
    from robot_client.gdk_adapter import canonical_to_gdk_servo_positions
    from robot_client.robot_lock import RobotProcessLock
    from robot_client.safety_layer import ActionSafetyLayer, load_safety_layer

if __package__:
    from .effector_profiles import EFFECTOR_PROFILE_CHOICES, EffectorProfile, get_effector_profile
else:
    from robot_client.effector_profiles import EFFECTOR_PROFILE_CHOICES, EffectorProfile, get_effector_profile


class UnknownMotionOutcomeError(RuntimeError):
    """An RPC future broke, so neither retry nor motion cleanup is authorized."""


def is_broken_promise(error: Exception) -> bool:
    message = str(error).lower()
    return "broken promise" in message or "brokenpromise" in message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute a monitored reset toward the packaged fixed body pose. "
            "Without --execute this command only reads state and prints a plan."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("16d", "24d"),
        default="16d",
        help=(
            "Model profile label for the reset report. Both labels restore arms, "
            "head and waist to the same packaged body-reset profile."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Execute the validated physical reset plan.")
    parser.add_argument(
        "--skip-effectors",
        action="store_true",
        help="Do not move either end effector; hold their current positions.",
    )
    parser.add_argument(
        "--cycle-effectors",
        action="store_true",
        help="Run the validated Dahuan open-close-open sequence before the body reset.",
    )
    parser.add_argument(
        "--effector-profile",
        choices=EFFECTOR_PROFILE_CHOICES,
        default="dahuan",
        help="Physical gripper profile used to validate reset endpoints and units.",
    )
    parser.add_argument(
        "--effector-cycle-pause-s",
        type=float,
        default=0.5,
        help="Pause after the first open and close phases of the gripper cycle.",
    )
    parser.add_argument("--body-speed", type=float, default=0.15, help="Arm/waist joint speed in rad/s.")
    parser.add_argument("--head-speed", type=float, default=0.10, help="Head joint speed in rad/s.")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--reset-timeout-s", type=float, default=60.0,
                        help="Overall execution budget; bounded below the web wrapper's 70-second timeout.")
    parser.add_argument("--body-tolerance", type=float, default=0.03)
    parser.add_argument("--effector-tolerance", type=float, default=0.0015)
    parser.add_argument(
        "--effector-timeout-s",
        type=float,
        default=8.0,
        help="Independent feedback timeout for each open/close gripper phase.",
    )
    parser.add_argument("--effector-reissue-s", type=float, default=0.75)
    parser.add_argument("--effector-max-commands", type=int, default=3)
    parser.add_argument(
        "--max-effector-step",
        type=float,
        default=0.0025,
        help="Maximum gripper interpolation increment per servo target, in profile units.",
    )
    parser.add_argument("--effector-servo-period-s", type=float, default=0.02)
    parser.add_argument("--effector-servo-cycles", type=int, default=3)
    parser.add_argument(
        "--max-joint-travel",
        type=float,
        default=1.0,
        help="Maximum travel of any body joint in one reset stage.",
    )
    parser.add_argument(
        "--max-total-joint-travel",
        type=float,
        default=4.0,
        help=(
            "Abort before motion if any body joint's complete reset travel exceeds this. "
            "Travel above --max-joint-travel is split into bounded stages."
        ),
    )
    parser.add_argument(
        "--max-effector-travel",
        type=float,
        default=0.03,
        help="Abort before motion if either commanded effector would travel farther than this.",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        help="Calibrated workspace safety profile used to preflight the complete reset path.",
    )
    parser.add_argument(
        "--safety-path-samples",
        type=int,
        default=64,
        help="Number of collision samples across the complete joint-space reset path.",
    )
    parser.add_argument("--startup-wait-s", type=float, default=2.0)
    parser.add_argument("--status-period-s", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reset_to_dataset_start"))
    args = parser.parse_args()
    args.effector_profile = get_effector_profile(args.effector_profile).name
    positive = (
        "body_speed",
        "head_speed",
        "timeout_s",
        "reset_timeout_s",
        "body_tolerance",
        "effector_tolerance",
        "effector_timeout_s",
        "effector_reissue_s",
        "max_effector_step",
        "effector_servo_period_s",
        "max_joint_travel",
        "max_total_joint_travel",
        "max_effector_travel",
        "effector_cycle_pause_s",
        "status_period_s",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.body_speed > 0.3:
        parser.error("--body-speed must not exceed 0.3 rad/s")
    if args.head_speed > 0.2:
        parser.error("--head-speed must not exceed 0.2 rad/s")
    if not np.isfinite(args.reset_timeout_s) or args.reset_timeout_s > 60.0:
        parser.error("--reset-timeout-s must be finite and must not exceed 60 seconds")
    if args.max_effector_step > 0.005:
        parser.error("--max-effector-step must not exceed 0.005 profile units")
    if args.max_total_joint_travel < args.max_joint_travel:
        parser.error("--max-total-joint-travel must not be smaller than --max-joint-travel")
    if not 16 <= args.safety_path_samples <= 512:
        parser.error("--safety-path-samples must be between 16 and 512")
    if args.execute and args.safety_config is None:
        parser.error("--execute requires --safety-config")
    if not 1 <= args.effector_max_commands <= 5:
        parser.error("--effector-max-commands must be between 1 and 5")
    if not 1 <= args.effector_servo_cycles <= 10:
        parser.error("--effector-servo-cycles must be between 1 and 10")
    if args.startup_wait_s < 0:
        parser.error("--startup-wait-s must be non-negative")
    if args.skip_effectors and args.cycle_effectors:
        parser.error("--skip-effectors and --cycle-effectors are mutually exclusive")
    if args.cycle_effectors and args.effector_profile != "dahuan":
        parser.error("the open-close-open reset sequence is currently validated only for Dahuan grippers")
    return args


def check_result(gdk: Any, operation: str, result: Any) -> None:
    if result is not None and result != gdk.GDKRes.kSuccess and result != 0:
        raise RuntimeError(f"{operation} failed: {result!r}")


def read_motion_mode(robot: Any) -> int:
    """Read GDK's motion status; unknown/error states cannot authorize a reset."""
    status = robot.get_motion_control_status()
    def field(name: str, default: Any = None) -> Any:
        return status.get(name, default) if isinstance(status, dict) else getattr(status, name, default)
    mode = field("mode")
    error = field("error_code")
    if mode is None or error is None:
        raise RuntimeError("GDK motion-control status is missing mode/error_code; reset state is unknown")
    mode = int(mode)
    if int(error) != 0:
        raise RuntimeError(f"GDK motion-control error {error}: {field('error_msg', '')}")
    # The deployed SDK's robot_demo.py defines 0=stop, 1=G1 servo,
    # 2=planning, 5=G2 servo. Planning must finish before another request.
    if mode not in {0, 1, 2, 5}:
        raise RuntimeError(f"Unknown GDK motion-control mode {mode}; reset aborted")
    return mode


def require_request_ready(robot: Any, label: str) -> None:
    mode = read_motion_mode(robot)
    print(f"{label}: GDK motion_mode={mode} error_code=0 before request", flush=True)
    if mode == 2:
        raise RuntimeError(f"{label}: GDK still reports planning; no overlapping request was sent")


def read_body_positions(robot: Any) -> dict[str, float]:
    response = robot.get_joint_states()
    positions: dict[str, float] = {}
    for state in response.get("states", []):
        name = state.get("name")
        value = state.get("motor_position", state.get("position"))
        if name is not None and value is not None:
            positions[str(name)] = float(value)
    required = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS + WAIST_JOINTS
    missing = [name for name in required if name not in positions]
    if missing:
        raise RuntimeError(f"GDK joint state is missing required joints: {missing}")
    return positions


def read_effector(end_state: dict[str, Any], side: str) -> tuple[float, str]:
    side_state = end_state.get(f"{side}_end_state")
    if not isinstance(side_state, dict) or not side_state.get("end_states"):
        raise RuntimeError(f"GDK end state is missing {side}_end_state")
    reported_names = side_state.get("names")
    if isinstance(reported_names, str) and reported_names:
        joint_name = reported_names
    elif isinstance(reported_names, (list, tuple)) and reported_names:
        joint_name = str(reported_names[0])
    else:
        joint_name = DEFAULT_EFFECTOR_JOINT_NAMES[0 if side == "left" else 1]
        print(f"WARNING: GDK did not report the {side} effector name; using fallback {joint_name}")
    return float(side_state["end_states"][0]["position"]), joint_name


def read_canonical_state(
    robot: Any, *, skip_effectors: bool = False
) -> tuple[np.ndarray, dict[str, str]]:
    positions = read_body_positions(robot)
    if skip_effectors:
        # The installed grippers may differ from the dataset hardware.  In body-only
        # mode do not even query the end-effector API: placeholder values are used
        # only to preserve the canonical report shape and are never commanded.
        reference = np.asarray(BODY_RESET_TARGET_24D, dtype=np.float64)
        left = float(reference[EFFECTOR_CANONICAL_INDICES[0]])
        right = float(reference[EFFECTOR_CANONICAL_INDICES[1]])
        left_name, right_name = DEFAULT_EFFECTOR_JOINT_NAMES
    else:
        end_state = robot.get_end_state()
        left, left_name = read_effector(end_state, "left")
        right, right_name = read_effector(end_state, "right")
    state = np.asarray(
        [positions[name] for name in LEFT_ARM_JOINTS]
        + [left]
        + [positions[name] for name in RIGHT_ARM_JOINTS]
        + [right]
        + [positions[name] for name in HEAD_JOINTS]
        + [positions[name] for name in WAIST_JOINTS],
        dtype=np.float64,
    )
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid GDK canonical state: shape={state.shape}")
    return state, {"left": left_name, "right": right_name}


def selected_body_indices(_scope: str) -> tuple[int, ...]:
    return ARM_CANONICAL_INDICES + HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES


def effective_target(
    current: np.ndarray,
    _scope: str,
    skip_effectors: bool,
    *,
    cycle_effectors: bool = False,
) -> np.ndarray:
    target = np.asarray(BODY_RESET_TARGET_24D, dtype=np.float64).copy()
    if skip_effectors:
        target[list(EFFECTOR_CANONICAL_INDICES)] = current[list(EFFECTOR_CANONICAL_INDICES)]
    elif cycle_effectors:
        target[list(EFFECTOR_CANONICAL_INDICES)] = DH_GRIPPER_OPEN_POSITION
    else:
        target[list(EFFECTOR_CANONICAL_INDICES)] = np.clip(
            target[list(EFFECTOR_CANONICAL_INDICES)],
            GRIPPER_MIN_POSITION,
            GRIPPER_MAX_POSITION,
        )
    return target


def print_plan(current: np.ndarray, target: np.ndarray, scope: str, skip_effectors: bool) -> None:
    selected = set(selected_body_indices(scope))
    if not skip_effectors:
        selected.update(EFFECTOR_CANONICAL_INDICES)
    print(f"\nReset plan (packaged profile={BODY_RESET_PROFILE_ID})")
    print(f"scope={scope}, effectors={'hold' if skip_effectors else 'reset'}")
    print(f"{'idx':>3}  {'joint':<25} {'current':>10} {'target':>10} {'delta':>10}  command")
    for index, (name, cur, tgt) in enumerate(zip(CANONICAL_STATE_NAMES, current, target, strict=True)):
        command = "yes" if index in selected else "hold"
        print(f"{index:3d}  {name:<25} {cur:10.5f} {tgt:10.5f} {tgt - cur:10.5f}  {command}")


def validate_plan(
    current: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    effector_profile: EffectorProfile,
) -> None:
    body_indices = selected_body_indices(args.scope)
    body_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS + WAIST_JOINTS
    if len(body_indices) != len(body_names):
        raise RuntimeError("Body reset joint map is inconsistent")
    for index, joint_name in zip(body_indices, body_names, strict=True):
        lower, upper = GDK_BODY_JOINT_LIMITS[joint_name]
        for label, value in (("current", current[index]), ("target", target[index])):
            if value < lower or value > upper:
                raise RuntimeError(
                    "Body hardware position limit exceeded: "
                    f"joint={CANONICAL_STATE_NAMES[index]} gdk_joint={joint_name} "
                    f"{label}={value:.4f} not in [{lower:.4f}, {upper:.4f}] rad"
                )
    body_delta = np.abs(target[list(body_indices)] - current[list(body_indices)])
    worst_local_index = int(np.argmax(body_delta))
    worst_index = body_indices[worst_local_index]
    max_body_travel = float(body_delta[worst_local_index])
    if max_body_travel > args.max_total_joint_travel:
        raise RuntimeError(
            "Body total travel gate exceeded: "
            f"joint={CANONICAL_STATE_NAMES[worst_index]} index={worst_index} "
            f"current={current[worst_index]:.4f} target={target[worst_index]:.4f} "
            f"delta={max_body_travel:.4f} > {args.max_total_joint_travel:.4f} rad"
        )
    if not args.skip_effectors:
        effector_targets = target[list(EFFECTOR_CANONICAL_INDICES)]
        effector_min, effector_max = effector_profile.hardware_state_range
        if np.any(effector_targets < effector_min) or np.any(effector_targets > effector_max):
            raise RuntimeError(f"Effector target is outside physical range: {effector_targets.tolist()}")
        max_effector_travel = float(
            np.max(
                np.abs(target[list(EFFECTOR_CANONICAL_INDICES)] - current[list(EFFECTOR_CANONICAL_INDICES)])
            )
        )
        if max_effector_travel > args.max_effector_travel:
            raise RuntimeError(
                "Effector travel gate exceeded: "
                f"{max_effector_travel:.4f} > {args.max_effector_travel:.4f} rad"
            )


def body_reset_waypoints(
    current: np.ndarray,
    target: np.ndarray,
    body_indices: tuple[int, ...],
    max_stage_travel: float,
) -> list[np.ndarray]:
    """Split one joint-space line into stages without changing its path."""
    if max_stage_travel <= 0:
        raise ValueError("max_stage_travel must be positive")
    maximum = float(
        np.max(np.abs(target[list(body_indices)] - current[list(body_indices)]))
    )
    stages = max(1, int(np.ceil(maximum / max_stage_travel)))
    return [
        current + (target - current) * (stage / stages)
        for stage in range(1, stages + 1)
    ]


def validate_reset_surface_path(
    current: np.ndarray,
    target: np.ndarray,
    safety_layer: ActionSafetyLayer,
    path_samples: int,
) -> None:
    """Fail before control ownership if the full reset line approaches the table."""
    original_samples = safety_layer.path_samples
    safety_layer.path_samples = int(path_samples)
    try:
        decision = safety_layer.guard_trajectory(current, (target,), raw_action=target)
    finally:
        safety_layer.path_samples = original_samples
    if decision.hard_limit_triggered:
        raise RuntimeError(
            "Reset workspace path gate exceeded: "
            f"surface={decision.limiting_surface or 'unknown'} "
            f"point={decision.limiting_point or 'unknown'} "
            f"path_clearance_to_hard_limit={decision.path_surface_clearance * 1000:.3f}mm; "
            "no reset action was sent"
        )


def body_request_values(state: np.ndarray, _scope: str) -> tuple[tuple[str, ...], list[float]]:
    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS
    indices = ARM_CANONICAL_INDICES + WAIST_CANONICAL_INDICES
    return names, state[list(indices)].tolist()


def send_body_target(
    gdk: Any,
    robot: Any,
    state: np.ndarray,
    scope: str,
    speed: float,
    *,
    life_time: float,
) -> None:
    names, positions = body_request_values(state, scope)
    request = gdk.JointControlReq()
    request.life_time = life_time
    request.joint_names = list(names)
    request.joint_positions = positions
    request.joint_velocities = [speed] * len(names)
    require_request_ready(robot, "joint_control_request RPC")
    try:
        result = robot.joint_control_request(request)
    except Exception as error:
        print(f"joint_control_request RPC exception: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        # This SDK can raise only "JointControl failed" while logging a
        # PLANNING transition timeout to stderr. The Python exception cannot
        # prove whether the request was accepted, so cleanup must not submit
        # another body request merely under the name "hold current pose".
        raise UnknownMotionOutcomeError(
            f"joint_control_request outcome is unconfirmed after SDK exception: {error}; "
            "reset aborted without retry or motion cleanup"
        ) from error
    check_result(gdk, "joint_control_request", result)


def send_head_target(
    gdk: Any,
    robot: Any,
    state: np.ndarray,
    speed: float,
    *,
    attempts: int = 3,
    retry_delay_s: float = 0.2,
) -> None:
    positions = state[list(HEAD_CANONICAL_INDICES)].tolist()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        require_request_ready(robot, "head request")
        try:
            check_result(
                gdk,
                "move_head_joint",
                robot.move_head_joint(positions, [speed] * len(HEAD_JOINTS)),
            )
            if attempt > 1:
                print(f"Head target succeeded on attempt {attempt}/{attempts}.")
            return
        except Exception as error:
            if is_broken_promise(error):
                # A broken future cannot prove rejection versus acceptance.
                # Preserve the failed reset rather than replaying that motion.
                raise UnknownMotionOutcomeError("move_head_joint outcome is unknown (broken promise); reset aborted without replay") from error
            last_error = error
            print(
                f"WARNING: head target attempt {attempt}/{attempts} failed: {error}",
                file=sys.stderr,
            )
            if attempt < attempts and retry_delay_s:
                time.sleep(retry_delay_s)
    raise RuntimeError(f"move_head_joint failed after {attempts} attempts") from last_error


def send_combined_servo_target(
    gdk: Any,
    robot: Any,
    target: np.ndarray,
    effector_names: dict[str, str],
    control_period_s: float,
) -> None:
    request = gdk.JointServoControlReq()
    request.joint_names = list(GDK_BODY_JOINTS + (effector_names["left"], effector_names["right"]))
    request.joint_positions = canonical_to_gdk_servo_positions(target).tolist()
    request.control_period = control_period_s
    if hasattr(request, "joint_velocities"):
        request.joint_velocities = [0.0] * len(request.joint_names)
    if hasattr(request, "joint_accelerations"):
        request.joint_accelerations = [0.0] * len(request.joint_names)
    check_result(gdk, "joint_servo_control(effectors)", robot.joint_servo_control(request))


def move_effectors_with_servo(
    gdk: Any,
    robot: Any,
    target: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    current, effector_names = read_canonical_state(robot)
    start_effectors = current[list(EFFECTOR_CANONICAL_INDICES)].copy()
    target_effectors = target[list(EFFECTOR_CANONICAL_INDICES)]
    max_distance = float(np.max(np.abs(target_effectors - start_effectors)))
    steps = max(1, int(np.ceil(max_distance / args.max_effector_step)))
    print(
        "Effector joint-servo plan: "
        f"names={[effector_names['left'], effector_names['right']]} "
        f"start={start_effectors.tolist()} target={target_effectors.tolist()} "
        f"steps={steps} cycles_per_step={args.effector_servo_cycles}"
    )

    command = current.copy()
    for step in range(1, steps + 1):
        ratio = step / steps
        command[list(EFFECTOR_CANONICAL_INDICES)] = (
            start_effectors + (target_effectors - start_effectors) * ratio
        )
        for _ in range(args.effector_servo_cycles):
            send_combined_servo_target(
                gdk,
                robot,
                command,
                effector_names,
                args.effector_servo_period_s,
            )
            time.sleep(args.effector_servo_period_s)
        if step == 1 or step == steps or step % 10 == 0:
            print(f"effectors servo interpolation: step={step}/{steps}")
    return command


def send_dahuan_effector_target(
    gdk: Any,
    robot: Any,
    expected_feedback_position: float,
    *,
    attempts: int = 3,
    retry_delay_s: float = 0.2,
) -> None:
    """Command both Dahuan grippers with the GDK-documented dahuan interface."""

    # get_end_state reports opening width (0=closed, 0.025=open), whereas
    # move_ee_pos(target_type=dahuan) accepts the inverted actuator target
    # (0=open, 0.025=closed).
    raw_position = DH_GRIPPER_OPEN_POSITION - float(expected_feedback_position)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = gdk.JointStates()
            request.group = "dual_tool"
            request.target_type = "dahuan"
            states = []
            for _ in range(2):
                state = gdk.JointState()
                state.position = raw_position
                states.append(state)
            request.states = states
            request.nums = len(states)
            check_result(
                gdk,
                "move_ee_pos(dual_tool,dahuan)",
                robot.move_ee_pos(request),
            )
            if attempt > 1:
                print(f"Dahuan gripper target succeeded on attempt {attempt}/{attempts}.")
            return
        except Exception as error:
            last_error = error
            print(
                f"WARNING: Dahuan gripper target attempt {attempt}/{attempts} failed: {error}",
                file=sys.stderr,
            )
            if attempt < attempts and retry_delay_s:
                time.sleep(retry_delay_s)
    raise RuntimeError("Dahuan dual_tool target failed after 3 attempts") from last_error


def move_and_verify_dahuan_effectors(
    gdk: Any,
    robot: Any,
    position: float,
    args: argparse.Namespace,
    label: str,
) -> np.ndarray:
    current, _ = read_canonical_state(robot)
    target = current.copy()
    target[list(EFFECTOR_CANONICAL_INDICES)] = position
    raw_position = DH_GRIPPER_OPEN_POSITION - position
    print(
        f"Dahuan gripper reset phase={label} expected_feedback={position:.5f}m "
        f"dahuan_raw_target={raw_position:.5f}"
    )
    started = time.monotonic()
    last_command_at = float("-inf")
    last_status = float("-inf")
    command_count = 0
    consecutive_successes = 0
    while True:
        current, _ = read_canonical_state(robot)
        error = np.abs(
            target[list(EFFECTOR_CANONICAL_INDICES)]
            - current[list(EFFECTOR_CANONICAL_INDICES)]
        )
        max_error = float(np.max(error))
        now = time.monotonic()
        if now - last_status >= args.status_period_s:
            values = current[list(EFFECTOR_CANONICAL_INDICES)].tolist()
            print(
                f"effectors:{label}: elapsed={now - started:.1f}s "
                f"feedback={values} max_error={max_error:.5f}"
            )
            last_status = now
        if max_error <= args.effector_tolerance:
            consecutive_successes += 1
            if consecutive_successes >= 3:
                if command_count == 0:
                    print(f"Dahuan gripper phase={label} already at target; verified without a new command")
                return current
        else:
            consecutive_successes = 0
            if (
                command_count < args.effector_max_commands
                and now - last_command_at >= args.effector_reissue_s
            ):
                command_count += 1
                print(
                    f"move_ee_pos issue={command_count}/{args.effector_max_commands} "
                    f"phase={label}"
                )
                send_dahuan_effector_target(gdk, robot, position)
                last_command_at = time.monotonic()
        if now - started >= args.effector_timeout_s:
            raise TimeoutError(
                f"effectors:{label} timed out after {command_count} command(s): "
                f"left_error={error[0]:.5f}m right_error={error[1]:.5f}m"
            )
        time.sleep(0.1)


def run_dahuan_effector_reset_cycle(
    gdk: Any,
    robot: Any,
    args: argparse.Namespace,
) -> np.ndarray:
    """Slowly open, close, and finally open both Dahuan grippers."""

    phases = (
        ("open_before_reset", DH_GRIPPER_OPEN_POSITION, True),
        ("close_self_check", DH_GRIPPER_CLOSED_POSITION, True),
        ("open_final", DH_GRIPPER_OPEN_POSITION, False),
    )
    final: np.ndarray | None = None
    try:
        for label, position, pause_after in phases:
            final = move_and_verify_dahuan_effectors(gdk, robot, position, args, label)
            if pause_after:
                print(
                    f"Dahuan gripper phase={label} verified; "
                    f"pausing {args.effector_cycle_pause_s:.2f}s"
                )
                time.sleep(args.effector_cycle_pause_s)
    except Exception as phase_error:
        print(
            "WARNING: Dahuan gripper reset phase failed; requesting one bounded final-open recovery.",
            file=sys.stderr,
        )
        try:
            move_and_verify_dahuan_effectors(
                gdk,
                robot,
                DH_GRIPPER_OPEN_POSITION,
                args,
                "open_failure_recovery",
            )
        except Exception as recovery_error:
            raise RuntimeError(
                f"Dahuan gripper reset failed ({phase_error}); final-open recovery also failed "
                f"({recovery_error})"
            ) from phase_error
        raise RuntimeError(
            f"Dahuan gripper reset failed ({phase_error}); final-open recovery succeeded, "
            "so body reset was not started"
        ) from phase_error
    assert final is not None
    return final


def hold_current_pose(
    gdk: Any,
    robot: Any,
    scope: str,
    speed: float,
    servo_period_s: float,
    *,
    skip_effectors: bool,
) -> None:
    current, effector_names = read_canonical_state(robot, skip_effectors=skip_effectors)
    # A failed/broken future may leave the original planner active. Cleanup
    # must not submit another body plan while its outcome remains unknown.
    require_request_ready(robot, "failure hold")
    send_body_target(gdk, robot, current, scope, speed, life_time=1.0)
    wait_for_target(robot, current, ARM_CANONICAL_INDICES + WAIST_CANONICAL_INDICES,
                    0.03, 2.0, 0.5, "failure_body_hold", skip_effectors=skip_effectors,
                    require_planner_idle=True)
    send_head_target(gdk, robot, current, speed)
    wait_for_target(robot, current, HEAD_CANONICAL_INDICES, 0.03, 2.0, 0.5,
                    "failure_head_hold", skip_effectors=skip_effectors, require_planner_idle=True)
    if not skip_effectors:
        send_combined_servo_target(gdk, robot, current, effector_names, servo_period_s)


def wait_for_target(
    robot: Any,
    target: np.ndarray,
    indices: tuple[int, ...],
    tolerance: float,
    timeout_s: float,
    status_period_s: float,
    label: str,
    *,
    skip_effectors: bool = False,
    require_planner_idle: bool = False,
) -> np.ndarray:
    started = time.monotonic()
    last_status = float("-inf")
    consecutive_successes = 0
    while True:
        current, _ = read_canonical_state(robot, skip_effectors=skip_effectors)
        error = np.abs(target[list(indices)] - current[list(indices)])
        max_error = float(np.max(error))
        motion_mode = read_motion_mode(robot) if require_planner_idle else None
        planner_ready = not require_planner_idle or motion_mode != 2
        now = time.monotonic()
        if now - last_status >= status_period_s:
            mode_detail = f" motion_mode={motion_mode}" if require_planner_idle else ""
            print(f"{label}: elapsed={now - started:.1f}s max_error={max_error:.5f}{mode_detail}")
            last_status = now
        if max_error <= tolerance and planner_ready:
            consecutive_successes += 1
            if consecutive_successes >= 3:
                return current
        else:
            consecutive_successes = 0
        if now - started >= timeout_s:
            worst_local_index = int(np.argmax(error))
            worst_index = indices[worst_local_index]
            unit = "m" if worst_index in EFFECTOR_CANONICAL_INDICES else "rad"
            raise TimeoutError(
                f"{label} timed out: {CANONICAL_STATE_NAMES[worst_index]} "
                f"error={error[worst_local_index]:.5f} {unit}; motion_mode={motion_mode}; "
                "next body/head request was not sent"
            )
        time.sleep(0.1)


def save_report(
    output_dir: Path,
    args: argparse.Namespace,
    current: np.ndarray,
    target: np.ndarray,
    final: np.ndarray | None,
    status: str,
    error: str | None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "error": error,
        "execute_requested": bool(args.execute),
        "scope": args.scope,
        "reset_head_waist": True,
        "skip_effectors": bool(args.skip_effectors),
        "cycle_effectors": bool(args.cycle_effectors),
        "effector_profile": args.effector_profile,
        "effector_control": {
            "backend": (
                "move_ee_pos(dual_tool,dahuan,bounded_reissue)"
                if args.cycle_effectors
                else "joint_servo_control"
            ),
            "raw_command_semantics": {"open": 0.0, "close": 0.025},
            "feedback_semantics": {"open": 0.025, "close": 0.0},
            "reissue_s": args.effector_reissue_s,
            "max_commands": args.effector_max_commands,
            "max_step": args.max_effector_step,
            "control_period_s": args.effector_servo_period_s,
            "cycles_per_step": args.effector_servo_cycles,
            "cycle_pause_s": args.effector_cycle_pause_s,
            "phase_timeout_s": args.effector_timeout_s,
            "sequence": ["open", "close", "open"] if args.cycle_effectors else [],
        },
        "reset_profile_id": BODY_RESET_PROFILE_ID,
        "body_control": {
            "speed_rad_s": args.body_speed,
            "head_speed_rad_s": args.head_speed,
            "max_stage_joint_travel_rad": args.max_joint_travel,
            "max_total_joint_travel_rad": args.max_total_joint_travel,
            "path": "piecewise_linear_body_then_head",
            "reset_timeout_s": getattr(args, "reset_timeout_s", 60.0),
            "planner_completion": "three_in_tolerance_reads_with_nonplanning_motion_status",
            "workspace_safety_config": (
                None if args.safety_config is None else str(args.safety_config)
            ),
            "workspace_path_samples": args.safety_path_samples,
        },
        "reset_reference": (
            "packaged fixed body pose; Dahuan effectors finish open after open-close-open cycle"
            if args.cycle_effectors
            else "packaged fixed body pose"
        ),
        "joint_order": list(CANONICAL_STATE_NAMES),
        "initial_state": current.tolist(),
        "effective_target": target.tolist(),
        "final_state": None if final is None else final.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def execute_reset(
    gdk: Any,
    robot: Any,
    current: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    safety_layer: ActionSafetyLayer,
) -> np.ndarray:
    reset_deadline = time.monotonic() + getattr(args, "reset_timeout_s", 60.0)
    def remaining(label: str, phase_limit: float | None = None) -> float:
        budget = reset_deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError(f"Overall reset budget exhausted before {label}; no next request was sent")
        return min(budget, args.timeout_s if phase_limit is None else phase_limit)

    print("\nEngaging GDK control at the current pose...")
    require_request_ready(robot, "reset engagement")
    remaining("body engagement")
    send_body_target(gdk, robot, current, args.scope, args.body_speed, life_time=1.0)
    wait_for_target(robot, current, ARM_CANONICAL_INDICES + WAIST_CANONICAL_INDICES,
                    args.body_tolerance, remaining("body engagement feedback"), args.status_period_s,
                    "engagement_body", skip_effectors=args.skip_effectors, require_planner_idle=True)
    remaining("head engagement")
    send_head_target(gdk, robot, current, args.head_speed)
    wait_for_target(robot, current, HEAD_CANONICAL_INDICES, args.body_tolerance,
                    remaining("head engagement feedback"), args.status_period_s, "engagement_head",
                    skip_effectors=args.skip_effectors, require_planner_idle=True)
    time.sleep(1.0)

    current_before_motion, _ = read_canonical_state(
        robot, skip_effectors=args.skip_effectors
    )
    if float(np.max(np.abs(current_before_motion - current))) > 0.05:
        raise RuntimeError("Robot moved by more than 0.05 rad after control engagement; reset aborted")

    if args.cycle_effectors:
        print("Running Dahuan gripper open-close-open reset sequence before body motion...")
        cycle_budget = remaining("gripper self-check", phase_limit=float("inf")) - 2 * args.effector_cycle_pause_s
        if cycle_budget <= 0:
            raise TimeoutError("No execution budget remains for the gripper self-check")
        # Reserve all three phases plus the existing bounded final-open recovery.
        cycle_args = argparse.Namespace(**vars(args))
        cycle_args.effector_timeout_s = min(args.effector_timeout_s, cycle_budget / 4)
        run_dahuan_effector_reset_cycle(gdk, robot, cycle_args)
    elif not args.skip_effectors:
        move_effectors_with_servo(gdk, robot, target, args)
        wait_for_target(
            robot,
            target,
            EFFECTOR_CANONICAL_INDICES,
            args.effector_tolerance,
            args.timeout_s,
            args.status_period_s,
            "effectors",
        )

    body_indices = selected_body_indices(args.scope)
    current_before_body, _ = read_canonical_state(
        robot, skip_effectors=args.skip_effectors
    )
    # Revalidate after the effector cycle in case an external disturbance moved
    # the body while the reset owned GDK.
    validate_plan(current_before_body, target, args, get_effector_profile(args.effector_profile))
    validate_reset_surface_path(
        current_before_body, target, safety_layer, args.safety_path_samples
    )
    waypoints = body_reset_waypoints(
        current_before_body,
        target,
        body_indices,
        args.max_joint_travel,
    )
    # Serialization changes the physical path. Preflight every body-only and
    # head-only segment before moving, rather than reusing the old diagonal.
    planned = current_before_body.copy()
    planner_indices = ARM_CANONICAL_INDICES + WAIST_CANONICAL_INDICES
    for waypoint in waypoints:
        body_target = planned.copy()
        body_target[list(planner_indices)] = waypoint[list(planner_indices)]
        validate_reset_surface_path(planned, body_target, safety_layer, args.safety_path_samples)
        head_target = body_target.copy()
        head_target[list(HEAD_CANONICAL_INDICES)] = waypoint[list(HEAD_CANONICAL_INDICES)]
        validate_reset_surface_path(body_target, head_target, safety_layer, args.safety_path_samples)
        planned = head_target
    print(
        "Moving body joints with the GDK velocity-controlled planner: "
        f"stages={len(waypoints)} max_stage_travel={args.max_joint_travel:.4f}rad"
    )
    final = current_before_body
    for stage, waypoint in enumerate(waypoints, 1):
        deadline = min(reset_deadline, time.monotonic() + args.timeout_s)
        before_stage, _ = read_canonical_state(robot, skip_effectors=args.skip_effectors)
        body_target = before_stage.copy()
        body_target[list(planner_indices)] = waypoint[list(planner_indices)]
        validate_reset_surface_path(before_stage, body_target, safety_layer, args.safety_path_samples)
        require_request_ready(robot, f"body stage {stage}")
        remaining(f"body stage {stage}")
        send_body_target(
            gdk,
            robot,
            body_target,
            args.scope,
            args.body_speed,
            life_time=min(args.timeout_s, remaining(f"body stage {stage}")),
        )
        final = wait_for_target(
            robot,
            body_target,
            planner_indices,
            args.body_tolerance,
            max(0.01, deadline - time.monotonic()),
            args.status_period_s,
            f"body_stage_{stage}_of_{len(waypoints)}",
            skip_effectors=args.skip_effectors,
            require_planner_idle=True,
        )
        head_target = final.copy()
        head_target[list(HEAD_CANONICAL_INDICES)] = waypoint[list(HEAD_CANONICAL_INDICES)]
        validate_reset_surface_path(final, head_target, safety_layer, args.safety_path_samples)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Stage {stage} budget exhausted; no head request was sent")
        remaining(f"head stage {stage}")
        send_head_target(gdk, robot, head_target, args.head_speed)
        final = wait_for_target(robot, head_target, HEAD_CANONICAL_INDICES, args.body_tolerance,
                                max(0.01, deadline - time.monotonic()), args.status_period_s,
                                f"head_stage_{stage}_of_{len(waypoints)}",
                                skip_effectors=args.skip_effectors, require_planner_idle=True)
    if args.cycle_effectors:
        print("Reasserting and verifying the final Dahuan gripper open state after body reset...")
        final_args = argparse.Namespace(**vars(args))
        final_args.effector_timeout_s = min(args.effector_timeout_s,
                                           remaining("final open confirmation", phase_limit=float("inf")))
        final = move_and_verify_dahuan_effectors(
            gdk,
            robot,
            DH_GRIPPER_OPEN_POSITION,
            final_args,
            "open_final_after_body",
        )
    # Head planning and the final gripper phase must not hide body drift.
    # Confirm all 22 body joints and the grippers from one last live sample.
    final = wait_for_target(robot, target, body_indices, args.body_tolerance,
                            remaining("final body confirmation"), args.status_period_s,
                            "final_body_all_22", skip_effectors=args.skip_effectors,
                            require_planner_idle=True)
    if args.cycle_effectors:
        opening_error = np.abs(final[list(EFFECTOR_CANONICAL_INDICES)] - DH_GRIPPER_OPEN_POSITION)
        if np.any(opening_error > args.effector_tolerance):
            raise RuntimeError("Final combined feedback did not confirm both Dahuan grippers open; reset incomplete")
    return final


def main() -> int:
    args = parse_args()
    effector_profile = get_effector_profile(args.effector_profile)
    lock = RobotProcessLock()
    gdk: Any = None
    robot: Any = None
    initialized = False
    current: np.ndarray | None = None
    target: np.ndarray | None = None
    final: np.ndarray | None = None
    status = "failed"
    error_message: str | None = None
    motion_started = False
    safety_layer: ActionSafetyLayer | None = None

    try:
        lock.acquire()
        import agibot_gdk

        gdk = agibot_gdk
        result = gdk.gdk_init()
        if result != gdk.GDKRes.kSuccess:
            raise RuntimeError(f"GDK initialization failed: {result!r}")
        initialized = True
        robot = gdk.Robot()
        if args.startup_wait_s:
            time.sleep(args.startup_wait_s)

        current, _ = read_canonical_state(robot, skip_effectors=args.skip_effectors)
        target = effective_target(
            current,
            args.scope,
            args.skip_effectors,
            cycle_effectors=args.cycle_effectors,
        )
        print_plan(current, target, args.scope, args.skip_effectors)
        validate_plan(current, target, args, effector_profile)
        if args.safety_config is not None:
            safety_layer = load_safety_layer(
                args.safety_config,
            )
            safety_layer.path_samples = args.safety_path_samples
            validate_reset_surface_path(
                current, target, safety_layer, args.safety_path_samples
            )

        if not args.execute:
            status = "preview_only"
            print("\nPREVIEW ONLY: no control mode was configured and no action was sent.")
            return_code = 0
        else:
            if safety_layer is None:
                raise RuntimeError("Physical reset has no calibrated workspace safety layer")
            print("\nPhysical motion requested.")
            print("Executing the validated reset plan without an interactive confirmation prompt.")
            motion_started = True
            final = execute_reset(gdk, robot, current, target, args, safety_layer)
            status = "completed"
            return_code = 0
            print("\nReset completed and verified against live joint feedback.")
    except KeyboardInterrupt:
        error_message = "Interrupted by operator"
        status = "interrupted"
        return_code = 130
        print("\nInterrupted by operator.", file=sys.stderr)
        if motion_started and gdk is not None and robot is not None:
            try:
                hold_current_pose(
                    gdk,
                    robot,
                    args.scope,
                    min(args.body_speed, args.head_speed),
                    args.effector_servo_period_s,
                    # The Dahuan cycle owns its own bounded final-open recovery.
                    # Never overwrite it with a measured near-closed value.
                    skip_effectors=args.skip_effectors or args.cycle_effectors,
                )
                print("Sent a hold-current-pose request.", file=sys.stderr)
            except Exception as hold_error:
                print(f"Failed to request hold-current-pose: {hold_error}", file=sys.stderr)
    except Exception as error:
        error_message = str(error)
        return_code = 1
        print(f"Reset failed: {error}", file=sys.stderr)
        if isinstance(error, UnknownMotionOutcomeError):
            print("Motion RPC outcome is unknown; no retry or motion cleanup request was sent.", file=sys.stderr)
        elif motion_started and gdk is not None and robot is not None:
            try:
                hold_current_pose(
                    gdk,
                    robot,
                    args.scope,
                    min(args.body_speed, args.head_speed),
                    args.effector_servo_period_s,
                    skip_effectors=args.skip_effectors or args.cycle_effectors,
                )
                print("Sent a hold-current-pose request.", file=sys.stderr)
            except Exception as hold_error:
                print(f"Failed to request hold-current-pose: {hold_error}", file=sys.stderr)
    finally:
        if initialized and gdk is not None:
            try:
                gdk.gdk_release()
            except Exception as release_error:
                print(f"GDK release warning: {release_error}", file=sys.stderr)
        lock.release()
        if current is not None and target is not None:
            report_path = save_report(
                args.output_dir,
                args,
                current,
                target,
                final,
                status,
                error_message,
            )
            print(f"Report saved to: {report_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
