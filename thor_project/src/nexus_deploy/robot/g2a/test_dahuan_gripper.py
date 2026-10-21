#!/usr/bin/env python3
"""Isolated G2 Dahuan gripper observation and motion test.

This program never sends body, head, waist, chassis, or whole-body servo targets.
Motion actions use only GDK ``move_ee_pos`` with ``target_type='dahuan'``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__:
    from .robot_lock import RobotProcessLock
else:
    from robot_client.robot_lock import RobotProcessLock

RAW_COMMAND = {"open": 0.0, "close": 0.025}
# get_end_state reports physical opening width, while move_ee_pos(dahuan)
# accepts the inverted actuator target documented by the installed GDK.
EXPECTED_FEEDBACK = {"open": 0.025, "close": 0.0}
SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe or independently test the two Dahuan grippers; no body command exists."
    )
    parser.add_argument("--action", choices=("observe", "open", "close", "cycle"), default="observe")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--backend",
        choices=("move-ee", "joint-servo"),
        default="move-ee",
        help=(
            "Gripper-only GDK backend. joint-servo requests contain only the selected "
            "Dahuan effector joint names and never any body joint."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--observe-s", type=float, default=3.0)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--pause-s", type=float, default=1.0)
    parser.add_argument("--poll-s", type=float, default=0.1)
    parser.add_argument("--reissue-s", type=float, default=0.75)
    parser.add_argument("--max-commands", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logs/test_dahuan_gripper")
    )
    args = parser.parse_args()
    for name in (
        "observe_s",
        "timeout_s",
        "pause_s",
        "poll_s",
        "reissue_s",
        "tolerance",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.max_commands <= 5:
        parser.error("--max-commands must be between 1 and 5")
    if args.action == "observe" and args.execute:
        parser.error("--execute is not used with --action=observe")
    if args.action != "observe":
        expected = confirmation_phrase(args.action, args.side)
        if not args.execute:
            parser.error("motion requires --execute")
        if args.confirmation != expected:
            parser.error(f"motion requires --confirmation={expected!r}")
    return args


def confirmation_phrase(action: str, side: str) -> str:
    return f"TEST DAHUAN GRIPPER {action.upper()} {side.upper()}"


def selected_sides(side: str) -> tuple[str, ...]:
    return SIDES if side == "both" else (side,)


def read_snapshot(robot: Any) -> dict[str, dict[str, Any]]:
    response = robot.get_end_state()
    snapshot: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        value = response.get(f"{side}_end_state")
        if not isinstance(value, dict) or not value.get("end_states"):
            raise RuntimeError(f"GDK end state is missing {side}_end_state")
        state = value["end_states"][0]
        names = value.get("names") or []
        name = str(names[0]) if isinstance(names, (list, tuple)) and names else str(names)
        item = {
            "name": name,
            "type": value.get("type"),
            "controlled": bool(value.get("controlled")),
            "enable": bool(state.get("enable")),
            "position": float(state.get("position", 0.0)),
            "velocity": float(state.get("velocity", 0.0)),
            "status": int(state.get("status", 0)),
            "err_code": int(state.get("err_code", 0)),
        }
        if not item["controlled"] or not item["enable"] or item["err_code"] != 0:
            raise RuntimeError(f"{side} Dahuan gripper is not ready: {item}")
        snapshot[side] = item
    return snapshot


def print_snapshot(snapshot: dict[str, dict[str, Any]], label: str) -> None:
    values = " ".join(
        f"{side}={snapshot[side]['position']:.5f}m"
        f"(vel={snapshot[side]['velocity']:.5f},err={snapshot[side]['err_code']})"
        for side in SIDES
    )
    print(f"{label}: {values}", flush=True)


def make_dahuan_request(gdk: Any, action: str, side: str) -> Any:
    request = gdk.JointStates()
    request.group = {
        "left": "left_tool",
        "right": "right_tool",
        "both": "dual_tool",
    }[side]
    request.target_type = "dahuan"
    positions = [RAW_COMMAND[action]] * len(selected_sides(side))
    states = []
    for position in positions:
        state = gdk.JointState()
        state.position = position
        states.append(state)
    request.states = states
    request.nums = len(states)
    return request


def make_gripper_only_servo_request(
    gdk: Any,
    action: str,
    side: str,
    snapshot: dict[str, dict[str, Any]],
    control_period_s: float,
) -> Any:
    """Build a JointServoControlReq containing Dahuan gripper joints only."""

    request = gdk.JointServoControlReq()
    selected = selected_sides(side)
    request.joint_names = [snapshot[item]["name"] for item in selected]
    if not all(request.joint_names):
        raise RuntimeError(f"Could not resolve selected Dahuan gripper names: {request.joint_names}")
    request.joint_positions = [RAW_COMMAND[action]] * len(selected)
    request.control_period = control_period_s
    if hasattr(request, "joint_velocities"):
        request.joint_velocities = [0.0] * len(selected)
    if hasattr(request, "joint_accelerations"):
        request.joint_accelerations = [0.0] * len(selected)
    return request


def command_and_verify(
    gdk: Any,
    robot: Any,
    action: str,
    side: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    raw_target = RAW_COMMAND[action]
    expected = EXPECTED_FEEDBACK[action]
    print(
        f"COMMAND action={action} side={side} backend={args.backend} "
        f"raw_target={raw_target:.5f} expected_feedback={expected:.5f}",
        flush=True,
    )
    started = time.monotonic()
    last_command_at = float("-inf")
    command_count = 0
    successes = 0
    last: dict[str, dict[str, Any]] | None = None
    while True:
        now = time.monotonic()
        command_due = (
            now - last_command_at >= args.poll_s
            if args.backend == "joint-servo"
            else command_count < args.max_commands and now - last_command_at >= args.reissue_s
        )
        if command_due:
            command_count += 1
            print(
                f"{args.backend} issue={command_count} action={action} side={side}",
                flush=True,
            )
            if args.backend == "move-ee":
                result = robot.move_ee_pos(make_dahuan_request(gdk, action, side))
            else:
                request = make_gripper_only_servo_request(
                    gdk,
                    action,
                    side,
                    last or read_snapshot(robot),
                    args.poll_s,
                )
                try:
                    result = robot.joint_servo_control(request, enable_low_latency=True)
                except TypeError:
                    result = robot.joint_servo_control(request)
            if result not in (None, 0, gdk.GDKRes.kSuccess):
                raise RuntimeError(f"{args.backend} returned {result!r}")
            last_command_at = time.monotonic()
        last = read_snapshot(robot)
        elapsed = time.monotonic() - started
        errors = {
            item: abs(last[item]["position"] - expected)
            for item in selected_sides(side)
        }
        print_snapshot(last, f"feedback action={action} elapsed={elapsed:.1f}s")
        if max(errors.values()) <= args.tolerance:
            successes += 1
            if successes >= 3:
                return last
        else:
            successes = 0
        if elapsed >= args.timeout_s:
            raise TimeoutError(
                f"Dahuan {action} timed out: expected_feedback={expected:.5f}m errors={errors}"
            )
        time.sleep(args.poll_s)


def save_report(args: argparse.Namespace, events: list[dict[str, Any]], status: str, error: str | None) -> Path:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"dahuan_gripper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "status": status,
                "error": error,
                "action": args.action,
                "side": args.side,
                "execute": bool(args.execute),
                "body_commands": False,
                "backend": args.backend,
                "target_type": "dahuan" if args.backend == "move-ee" else "gripper-only",
                "raw_command_semantics": RAW_COMMAND,
                "expected_feedback_semantics": EXPECTED_FEEDBACK,
                "events": events,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()
    lock = RobotProcessLock()
    initialized = False
    gdk: Any = None
    robot: Any = None
    events: list[dict[str, Any]] = []
    status = "failed"
    error_message: str | None = None
    try:
        lock.acquire()
        import agibot_gdk

        gdk = agibot_gdk
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK initialization failed")
        initialized = True
        robot = gdk.Robot()
        time.sleep(2.0)
        initial = read_snapshot(robot)
        print_snapshot(initial, "initial")
        events.append({"phase": "initial", "snapshot": initial})

        if args.action == "observe":
            deadline = time.monotonic() + args.observe_s
            while time.monotonic() < deadline:
                snapshot = read_snapshot(robot)
                print_snapshot(snapshot, "observe")
                events.append({"phase": "observe", "snapshot": snapshot})
                time.sleep(args.poll_s)
            status = "observed"
        else:
            actions = ("open", "close", "open") if args.action == "cycle" else (args.action,)
            for index, action in enumerate(actions):
                snapshot = command_and_verify(gdk, robot, action, args.side, args)
                events.append({"phase": action, "snapshot": snapshot})
                if index + 1 < len(actions):
                    time.sleep(args.pause_s)
            status = "completed"
        return_code = 0
    except Exception as error:
        error_message = str(error)
        print(f"Dahuan gripper test failed: {error}", file=sys.stderr, flush=True)
        return_code = 1
        if args.action == "cycle" and initialized and gdk is not None and robot is not None:
            try:
                print("Attempting one final-open recovery after cycle failure.", file=sys.stderr)
                snapshot = command_and_verify(gdk, robot, "open", args.side, args)
                events.append({"phase": "open_failure_recovery", "snapshot": snapshot})
            except Exception as recovery_error:
                print(f"Final-open recovery failed: {recovery_error}", file=sys.stderr)
    finally:
        if initialized and gdk is not None:
            try:
                gdk.gdk_release()
            except Exception as release_error:
                print(f"GDK release warning: {release_error}", file=sys.stderr)
        lock.release()
        report = save_report(args, events, status, error_message)
        print(f"Report saved to: {report}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
