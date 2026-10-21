#!/usr/bin/env python3
"""Summarize a completed Nexus control trace without accessing the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARM_INDICES = (*range(7), *range(8, 15))
LEFT_ARM = tuple(range(7))
RIGHT_ARM = tuple(range(8, 15))
GRIPPERS = (7, 15)


def percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        name: float(value)
        for name, value in zip(
            ("p50", "p95", "p99", "max"),
            np.percentile(values, (50, 95, 99, 100)),
            strict=True,
        )
    }


def summarize(rows: list[dict], start: float, end: float) -> dict:
    selected = [row for row in rows if start <= row["sdk_call_started"] < end]
    if len(selected) < 3:
        return {"samples": len(selected)}
    t = np.asarray([row["sdk_call_started"] for row in selected], dtype=float)
    policy = np.asarray([row["policy_control_target"] for row in selected], dtype=float)
    filtered = np.asarray([row["filtered_target"] for row in selected], dtype=float)
    adopted = np.asarray([row["adopted_target"] for row in selected], dtype=float)
    measured = np.asarray([row["measured_state"] for row in selected], dtype=float)
    dt = np.diff(t)
    policy_velocity = np.diff(policy[:, ARM_INDICES], axis=0) / dt[:, None]
    adopted_velocity = np.diff(adopted[:, ARM_INDICES], axis=0) / dt[:, None]
    policy_acceleration = np.diff(policy_velocity, axis=0) / dt[1:, None]
    adopted_acceleration = np.diff(adopted_velocity, axis=0) / dt[1:, None]
    deformation = np.abs(policy[:, ARM_INDICES] - adopted[:, ARM_INDICES])
    raw_feedback_error = np.abs(adopted[:, ARM_INDICES] - measured[:, ARM_INDICES])
    return {
        "samples": len(selected),
        "duration_s": float(t[-1] - t[0]),
        "servo_interval_ms": percentiles(dt * 1000),
        "policy_to_adopted_max_joint_rad": percentiles(deformation.max(axis=1)),
        "policy_to_adopted_per_joint_p95_rad": np.percentile(deformation, 95, axis=0).tolist(),
        "same_record_adopted_to_feedback_max_joint_rad": percentiles(raw_feedback_error.max(axis=1)),
        "same_record_adopted_to_feedback_per_joint_p95_rad": np.percentile(raw_feedback_error, 95, axis=0).tolist(),
        "policy_velocity_any_over_0_6_ratio": float(np.any(np.abs(policy_velocity) > 0.6, axis=1).mean()),
        "policy_acceleration_any_over_2_ratio": float(np.any(np.abs(policy_acceleration) > 2.0, axis=1).mean()),
        "adopted_velocity_any_over_0_6_ratio": float(np.any(np.abs(adopted_velocity) > 0.6 + 1e-3, axis=1).mean()),
        "adopted_acceleration_any_over_2_ratio": float(np.any(np.abs(adopted_acceleration) > 2.0 + 0.1, axis=1).mean()),
        "left_arm_range_rad": np.ptp(adopted[:, LEFT_ARM], axis=0).tolist(),
        "right_arm_range_rad": np.ptp(adopted[:, RIGHT_ARM], axis=0).tolist(),
        "gripper_range_m": np.ptp(adopted[:, GRIPPERS], axis=0).tolist(),
        "collision_response_counts": {
            key: sum(row.get("collision_response") == key for row in selected)
            for key in sorted({row.get("collision_response") for row in selected})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terminal-window-s",
        type=float,
        default=2.0,
        help="Keep the final suspected-contact window separate from the preceding trace.",
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    header = next(row for row in records if row.get("record_type") == "trace_start")
    replies = [
        row for row in records
        if row.get("record_type") == "inference_response" and row.get("installed")
    ]
    accepted = [
        row
        for row in records
        if row.get("record_type") == "control"
        and row.get("command_accepted") is True
        and row.get("sdk_call_started") is not None
    ]
    # Repeated diagnostic records can refer to the same physical SDK dispatch.
    rows = sorted(
        {row["sdk_call_started"]: row for row in accepted}.values(),
        key=lambda row: row["sdk_call_started"],
    )
    if len(rows) < 3:
        raise ValueError("No accepted physical command trace")
    first, last = rows[0]["sdk_call_started"], rows[-1]["sdk_call_started"]
    terminal_start = max(first, last - args.terminal_window_s)

    # Record threshold crossings only; grippers are deliberately not smoothed.
    policy_targets = np.asarray([row["policy_control_target"] for row in rows], dtype=float)
    adopted_targets = np.asarray([row["adopted_target"] for row in rows], dtype=float)

    def gripper_events(targets: np.ndarray) -> list[dict]:
        events = []
        for index in GRIPPERS:
            closed = targets[:, index] < 0.0125
            for sample in np.flatnonzero(closed[1:] != closed[:-1]) + 1:
                events.append(
                {
                    "side": "left" if index == 7 else "right",
                    "event": "close" if closed[sample] else "open",
                    "elapsed_s": float(rows[sample]["sdk_call_started"] - first),
                    "target_m": float(targets[sample, index]),
                    "chunk_id": rows[sample].get("chunk_id"),
                    "action_index": rows[sample].get("action_index"),
                }
            )
        return sorted(events, key=lambda row: row["elapsed_s"])

    policy_gripper_events = gripper_events(policy_targets)
    adopted_gripper_events = gripper_events(adopted_targets)
    paired_event_delays = []
    for side in ("left", "right"):
        policy_side = [event for event in policy_gripper_events if event["side"] == side]
        adopted_side = [event for event in adopted_gripper_events if event["side"] == side]
        for policy_event, adopted_event in zip(policy_side, adopted_side):
            paired_event_delays.append({
                "side": side,
                "event": policy_event["event"],
                "policy_elapsed_s": policy_event["elapsed_s"],
                "adopted_elapsed_s": adopted_event["elapsed_s"],
                "delay_ms": (adopted_event["elapsed_s"] - policy_event["elapsed_s"]) * 1000,
                "direction_match": policy_event["event"] == adopted_event["event"],
            })

    active_rows = [
        row for row in rows
        if not row.get("explicit_hold") and row.get("collision_response") == "async_checked_motion"
    ]

    report = {
        "source": str(args.trace.resolve()),
        "session": {
            "session_id": header.get("session_id"),
            "hardware_mode": header.get("hardware_mode"),
            "inference_mode": header.get("inference_mode"),
            "policy_fps": header.get("policy_fps"),
            "servo_fps": header.get("servo_fps"),
            "action_low_pass_cutoff_hz": header.get("action_low_pass_cutoff_hz"),
            "arm_limit_profile": header.get("adapter_config", {}).get("arm_limit_profile"),
            "max_arm_velocity": header.get("adapter_config", {}).get("max_arm_velocity"),
            "max_arm_acceleration": header.get("adapter_config", {}).get("max_arm_acceleration"),
        },
        "accepted_dispatches": len(rows),
        "duration_s": float(last - first),
        "inference": {},
        "segments": {
            "active_checked_motion": summarize(active_rows, first, last + 1e-9),
            "before_terminal_window": summarize(rows, first, terminal_start),
            "terminal_suspected_contact_window": summarize(rows, terminal_start, last + 1e-9),
            "whole_run": summarize(rows, first, last + 1e-9),
        },
        "policy_gripper_threshold_events": policy_gripper_events,
        "adopted_gripper_threshold_events": adopted_gripper_events,
        "paired_gripper_event_delays": paired_event_delays,
        "notes": [
            "The terminal window is separated for diagnosis; it is not an automatic contact detector.",
            "Same-record feedback error is not time-aligned and must not be interpreted as motor tracking error.",
            "Velocity and acceleration ratios use diagnostic dispatch timestamps and include RTC chunk handoffs.",
        ],
    }
    if replies:
        rtt_ms = np.asarray([
            (row["response_received_monotonic"] - row["observation_monotonic"]) * 1000
            for row in replies
        ])
        dropped = np.asarray([row.get("dropped_steps", 0) for row in replies], dtype=float)
        transition_rows = []
        for row in replies:
            raw = np.asarray(row["model_actions"], dtype=float)
            used = np.asarray(row["control_targets"], dtype=float)[int(row.get("dropped_steps", 0)):]
            raw_right = int(np.count_nonzero((raw[1:, 15] < 0.0125) != (raw[:-1, 15] < 0.0125)))
            used_right = int(np.count_nonzero((used[1:, 15] < 0.0125) != (used[:-1, 15] < 0.0125)))
            if raw_right or used_right:
                transition_rows.append({
                    "chunk_id": row.get("chunk_id"),
                    "dropped_steps": int(row.get("dropped_steps", 0)),
                    "raw_model_right_gripper_transitions": raw_right,
                    "installed_right_gripper_transitions": used_right,
                })
        report["inference"] = {
            "installed_replies": len(replies),
            "rtt_ms": percentiles(rtt_ms),
            "dropped_steps": percentiles(dropped),
            "right_gripper_transition_chunks": transition_rows,
            "raw_transition_chunk_count": sum(
                row["raw_model_right_gripper_transitions"] > 0 for row in transition_rows
            ),
            "installed_transition_chunk_count": sum(
                row["installed_right_gripper_transitions"] > 0 for row in transition_rows
            ),
            "warning": "Chunk transition counts overlap in time and must not be summed as executed events.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
