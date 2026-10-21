"""Bounded G2 gripper-only diagnostic; never constructs body commands."""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

NAMES = {"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=tuple(NAMES), required=True)
    p.add_argument("--action", choices=("open", "close"), required=True)
    p.add_argument("--hz", type=int, choices=(15, 30, 100), default=100)
    p.add_argument("--send-s", type=float, default=2.0)
    p.add_argument("--observe-s", type=float, default=1.0)
    p.add_argument("--shape", choices=("raw", "limited", "lp2-limited", "fast-limited"), default="raw",
                   help="Isolated step-response shaping; not full collision/RTC pipeline replay")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirmation", default="")
    args = p.parse_args()
    if not (0.1 <= args.send_s <= 3 and 0.1 <= args.observe_s <= 3):
        p.error("send/observe duration must be within 0.1..3 seconds")
    if not args.execute or args.confirmation != "GRIPPER ONLY":
        p.error("requires --execute --confirmation 'GRIPPER ONLY'")

    from robot_client.robot_lock import RobotProcessLock

    samples, calls = [], []
    report = {"args": vars(args), "body_commands": False, "samples": samples, "calls": calls}
    lock = RobotProcessLock()
    initialized = False
    g = None
    path = Path("logs/test_dh_gripper") / ("servo_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    lock.acquire()
    try:
        import agibot_gdk as g
        if g.gdk_init() != g.GDKRes.kSuccess:
            raise RuntimeError("GDK initialization failed")
        initialized = True
        robot = g.Robot()
        time.sleep(2)

        def read_body():
            state = robot.get_joint_states()
            values = [float(s["motor_position"]) for s in state["states"]]
            if len(values) != 22 or not all(math.isfinite(v) for v in values):
                raise RuntimeError("Invalid body feedback")
            return values

        baseline = read_body()
        time.sleep(.2)
        if max(abs(a-b) for a,b in zip(baseline, read_body())) > .01:
            raise RuntimeError("Body moving before test; no command sent")
        request = g.JointServoControlReq()
        request.joint_names = [NAMES[args.side]]
        request.joint_positions = [0.0 if args.action == "open" else .025]
        request.control_period = 1.0 / args.hz
        if hasattr(request, "joint_velocities"):
            request.joint_velocities = [0.0]
        if hasattr(request, "joint_accelerations"):
            request.joint_accelerations = [0.0]
        report["request"] = {"names": request.joint_names, "positions": request.joint_positions}
        shaped = None
        if args.shape != "raw":
            import numpy as np
            from robot_client.action_filter import CausalLowPassFilter, JointKinematicLimiter
            end = robot.get_end_state()[f"{args.side}_end_state"]
            initial = float(end["end_states"][0]["position"])
            if not math.isfinite(initial) or not 0 <= initial <= .025:
                raise RuntimeError("Invalid shaping initial position")
            goal = np.array([.025 if args.action == "open" else 0.0], dtype=np.float32)
            filt = CausalLowPassFilter(cutoff_hz=2, sample_hz=args.hz, action_dim=1)
            filt.reset(np.array([initial]))
            velocity, acceleration = (.06, .6) if args.shape == "fast-limited" else (.05, .2)
            report["shaping_limits"] = {"velocity_m_s": velocity, "acceleration_m_s2": acceleration}
            limiter = JointKinematicLimiter(sample_hz=args.hz, max_velocity=np.array([velocity]),
                                             max_acceleration=np.array([acceleration]))
            limiter.reset(np.array([initial]))
            refs = [filt.update(goal) if args.shape == "lp2-limited" else goal.copy()
                    for _ in range(math.ceil(args.send_s * args.hz) + 2)]
            shaped = []
            for a, b in zip(refs, refs[1:]):
                value = limiter.update_tracking(a, (b-a)*args.hz)
                clipped = np.clip(value, 0, .025)
                if not np.array_equal(value, clipped):
                    limiter.reconcile_position_clip(clipped)
                shaped.append(float(.025-clipped[0]))
        started = time.monotonic()
        next_send = started
        next_body = started
        body_delta = 0.0
        command_count = 0
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= args.send_s + args.observe_s:
                break
            if now >= next_body:
                body_delta = max(abs(a-b) for a,b in zip(baseline, read_body()))
                if body_delta > .02:
                    raise RuntimeError(f"Body changed {body_delta:.6f} rad; stopping gripper commands")
                next_body = now + .05
            raw = robot.get_end_state()
            positions = {}
            for side, name in NAMES.items():
                end = raw[f"{side}_end_state"]
                if end["names"] != [name] or not end["controlled"]:
                    raise RuntimeError(f"Unexpected effector identity/control: {side}")
                state = end["end_states"][0]
                pos = float(state["position"])
                if not state["enable"] or state["err_code"] != 0 or not math.isfinite(pos) or not -.001 <= pos <= .026:
                    raise RuntimeError(f"Invalid effector feedback: {side}: {state}")
                positions[side] = pos
            samples.append({"t": elapsed, "positions": positions, "body_delta_rad": body_delta})
            if elapsed < args.send_s and now >= next_send:
                if request.joint_names != [NAMES[args.side]]:
                    raise RuntimeError("Non-gripper target rejected")
                before = time.monotonic()
                if shaped is not None:
                    request.joint_positions = [shaped[command_count]]
                # Installed native signature verified. Never retry an ambiguous exception.
                result = robot.joint_servo_control(request, enable_low_latency=True)
                calls.append({"t": before-started, "duration_ms": (time.monotonic()-before)*1000,
                              "result": int(result), "command_raw": float(request.joint_positions[0])})
                if result != 0:
                    raise RuntimeError(f"Servo returned {result}")
                command_count += 1
                next_send += 1.0 / args.hz
                if next_send < time.monotonic():
                    next_send = time.monotonic() + 1.0 / args.hz  # no catch-up burst
            time.sleep(max(.001, min(.01, next_send-time.monotonic())) if elapsed < args.send_s else .01)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        try:
            if initialized:
                g.gdk_release()
        finally:
            lock.release()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("REPORT", path, flush=True)
        if samples:
            print("SUMMARY", json.dumps({"status": report.get("status"), "calls": len(calls), "first": samples[0], "last": samples[-1], "max_body_delta": max(s["body_delta_rad"] for s in samples)}), flush=True)


if __name__ == "__main__":
    main()
