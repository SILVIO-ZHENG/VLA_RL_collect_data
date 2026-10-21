#!/usr/bin/env python3
"""Measure one G2 DH-gripper step without constructing any body command.

The program is deliberately single-step and fail-closed.  Motion requires an
explicit confirmation phrase.  Every command contains exactly one selected DH
gripper joint; body joints are only read to detect unexpected movement.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any


NAMES = {
    "left": "idx31_gripper_l_inner_joint1",
    "right": "idx71_gripper_r_inner_joint1",
}
RAW_TARGET = {"open": 0.0, "close": 0.025}
FEEDBACK_TARGET = {"open": 0.025, "close": 0.0}


def compute_metrics(samples: list[dict[str, Any]], action: str) -> dict[str, Any]:
    """Return latency and travel metrics from timestamped physical feedback."""
    if len(samples) < 2:
        raise ValueError("At least two feedback samples are required")
    times = [float(sample["t_s"]) for sample in samples]
    positions = [float(sample["position_m"]) for sample in samples]
    start = positions[0]
    target = FEEDBACK_TARGET[action]
    travel = target - start
    if abs(travel) < 0.005:
        raise ValueError(f"Initial opening {start:.6f} m is too close to target {target:.6f} m")

    direction = 1.0 if travel > 0 else -1.0
    progress = [direction * (position - start) / abs(travel) for position in positions]

    def first_time(predicate) -> float | None:
        for index, value in enumerate(progress):
            if predicate(index, value):
                return times[index]
        return None

    onset = first_time(lambda index, _value: abs(positions[index] - start) >= 0.0005)
    milestones = {
        f"t{percent}_s": first_time(lambda _index, value, p=percent: value >= p / 100)
        for percent in (10, 50, 90)
    }
    settled = None
    for index in range(2, len(samples)):
        if all(abs(positions[item] - target) <= 0.001 for item in range(index - 2, index + 1)):
            settled = times[index - 2]
            break
    intervals = [right - left for left, right in zip(times, times[1:]) if right > left]
    velocities = [
        abs((right_position - left_position) / interval)
        for left_position, right_position, interval in zip(positions, positions[1:], intervals)
    ]
    return {
        "initial_position_m": start,
        "target_position_m": target,
        "final_position_m": positions[-1],
        "onset_s": onset,
        **milestones,
        "settled_s": settled,
        "max_observed_speed_m_s": max(velocities, default=0.0),
        "feedback_samples": len(samples),
        "feedback_rate_hz": 1.0 / statistics.median(intervals) if intervals else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=tuple(NAMES), required=True)
    parser.add_argument("--action", choices=("open", "close"), required=True)
    parser.add_argument(
        "--backend",
        choices=("move-ee", "joint-servo-once", "joint-servo-repeat"),
        required=True,
    )
    parser.add_argument("--repeat-hz", type=float, choices=(15.0, 30.0, 100.0), default=100.0)
    parser.add_argument(
        "--ramp-s",
        type=float,
        choices=(0.0, 0.1, 0.2, 0.3),
        default=0.0,
        help="Linearly ramp the repeated raw target; zero sends a full step.",
    )
    parser.add_argument("--feedback-hz", type=float, default=100.0)
    parser.add_argument("--timeout-s", type=float, default=4.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/gripper_step_benchmark"))
    args = parser.parse_args()
    expected = f"GRIPPER STEP {args.side.upper()} {args.action.upper()}"
    if not args.execute or args.confirmation != expected:
        parser.error(f"motion requires --execute --confirmation {expected!r}")
    if not 20 <= args.feedback_hz <= 200:
        parser.error("--feedback-hz must be in [20, 200]")
    if not 1 <= args.timeout_s <= 8:
        parser.error("--timeout-s must be in [1, 8]")
    if args.ramp_s and args.backend != "joint-servo-repeat":
        parser.error("--ramp-s requires --backend=joint-servo-repeat")
    return args


def _read_body(robot: Any) -> list[float]:
    response = robot.get_joint_states()
    values = [float(item["motor_position"]) for item in response["states"]]
    if len(values) != 22 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Invalid 22D body feedback")
    return values


def _read_grippers(robot: Any) -> dict[str, float]:
    response = robot.get_end_state()
    result: dict[str, float] = {}
    for side, expected_name in NAMES.items():
        group = response[f"{side}_end_state"]
        if group.get("names") != [expected_name] or not group.get("controlled"):
            raise RuntimeError(f"Unexpected {side} gripper identity/control: {group}")
        state = group["end_states"][0]
        position = float(state["position"])
        if (
            not state.get("enable")
            or int(state.get("err_code", -1)) != 0
            or not math.isfinite(position)
            or not -0.001 <= position <= 0.026
        ):
            raise RuntimeError(f"Invalid {side} gripper feedback: {state}")
        result[side] = position
    return result


def _make_move_ee_request(gdk: Any, side: str, raw_target: float) -> Any:
    request = gdk.JointStates()
    request.group = f"{side}_tool"
    request.target_type = "dahuan"
    state = gdk.JointState()
    state.position = raw_target
    request.states = [state]
    request.nums = 1
    return request


def _make_servo_request(gdk: Any, side: str, raw_target: float, period_s: float) -> Any:
    request = gdk.JointServoControlReq()
    request.joint_names = [NAMES[side]]
    request.joint_positions = [raw_target]
    request.control_period = period_s
    if hasattr(request, "joint_velocities"):
        request.joint_velocities = [0.0]
    if hasattr(request, "joint_accelerations"):
        request.joint_accelerations = [0.0]
    return request


def main() -> int:
    args = parse_args()
    # Import only after explicit motion confirmation has been validated.
    from robot_client.robot_lock import RobotProcessLock
    import agibot_gdk as gdk

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{args.side}_{args.action}_{args.backend}.json"
    )
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args) | {"output_dir": str(output_dir)},
        "body_commands": False,
        "commanded_joint_names": [NAMES[args.side]],
        "samples": [],
        "calls": [],
        "status": "failed",
    }
    lock = RobotProcessLock()
    initialized = False
    try:
        lock.acquire()
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK initialization failed")
        initialized = True
        robot = gdk.Robot()
        time.sleep(2.0)
        baseline_body = _read_body(robot)
        time.sleep(0.2)
        if max(abs(a - b) for a, b in zip(baseline_body, _read_body(robot), strict=True)) > 0.01:
            raise RuntimeError("Body is moving; no gripper command was sent")

        feedback = _read_grippers(robot)
        initial = feedback[args.side]
        target = FEEDBACK_TARGET[args.action]
        if abs(initial - target) < 0.005:
            raise RuntimeError(
                f"Selected gripper must start at the opposite endpoint: initial={initial:.6f}, target={target:.6f}"
            )

        period = 1.0 / args.repeat_hz
        raw_target = RAW_TARGET[args.action]
        initial_raw = 0.025 - initial
        request = (
            _make_move_ee_request(gdk, args.side, raw_target)
            if args.backend == "move-ee"
            else _make_servo_request(gdk, args.side, raw_target, period)
        )
        started = time.monotonic()
        next_tick = started
        next_command = started
        next_body_check = started
        sent = 0
        report["samples"].append(
            {
                "t_s": 0.0,
                "position_m": feedback[args.side],
                "other_position_m": feedback["right" if args.side == "left" else "left"],
                "body_delta_rad": 0.0,
            }
        )
        settle_samples = max(3, math.ceil(args.settle_s * args.feedback_hz))
        settled_count = 0
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed > args.timeout_s:
                raise TimeoutError(f"Gripper did not settle within {args.timeout_s:.1f}s")
            command_due = sent == 0 or (
                args.backend == "joint-servo-repeat" and now >= next_command
            )
            if command_due:
                call_started = time.monotonic()
                issued_raw_target = raw_target
                if args.ramp_s:
                    alpha = min(1.0, (call_started - started) / args.ramp_s)
                    issued_raw_target = initial_raw + alpha * (raw_target - initial_raw)
                    request.joint_positions = [issued_raw_target]
                if args.backend == "move-ee":
                    result = robot.move_ee_pos(request)
                else:
                    # The installed G2 binding has already been verified to
                    # support this keyword. Never retry an ambiguous exception.
                    result = robot.joint_servo_control(request, enable_low_latency=True)
                call_finished = time.monotonic()
                if result not in (None, 0, gdk.GDKRes.kSuccess):
                    raise RuntimeError(f"{args.backend} returned {result!r}")
                report["calls"].append(
                    {
                        "t_s": call_started - started,
                        "duration_ms": (call_finished - call_started) * 1000,
                        "raw_target": issued_raw_target,
                    }
                )
                sent += 1
                next_command += period

            feedback = _read_grippers(robot)
            sample_time = time.monotonic() - started
            report["samples"].append(
                {
                    "t_s": sample_time,
                    "position_m": feedback[args.side],
                    "other_position_m": feedback["right" if args.side == "left" else "left"],
                }
            )
            settled_count = settled_count + 1 if abs(feedback[args.side] - target) <= 0.001 else 0
            if settled_count >= settle_samples:
                break
            if now >= next_body_check:
                body = _read_body(robot)
                body_delta = max(abs(a - b) for a, b in zip(baseline_body, body, strict=True))
                report["samples"][-1]["body_delta_rad"] = body_delta
                if body_delta > 0.02:
                    raise RuntimeError(f"Body changed {body_delta:.6f} rad; stopping test")
                next_body_check += 0.05
            next_tick += 1.0 / args.feedback_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

        report["metrics"] = compute_metrics(report["samples"], args.action)
        report["status"] = "completed"
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return 1
    finally:
        if initialized:
            gdk.gdk_release()
        lock.release()
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"REPORT {report_path}", flush=True)
        print(json.dumps({key: report.get(key) for key in ("status", "error", "metrics")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
