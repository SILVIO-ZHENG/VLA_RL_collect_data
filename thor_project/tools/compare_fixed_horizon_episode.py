#!/usr/bin/env python3
"""Offline 30->100 Hz replay of one complete dual-arm LeRobot episode.

No network, GDK, robot connection, or command path is used.  The same NumPy
projector and final limiter as the robot client are imported so the numerical
comparison stays implementation-aligned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nexus_deploy.robot.g2a.action_filter import (
    FixedHorizonChunkProjector,
    JointKinematicLimiter,
)


ARMS = (*range(7), *range(8, 15))
LEFT = tuple(range(7))
RIGHT = tuple(range(8, 15))
GRIPPERS = (7, 15)


def load_episode(dataset: Path, episode: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pyarrow.parquet as pq

    info = json.loads((dataset / "meta/info.json").read_text())
    if float(info["fps"]) != 30:
        raise ValueError("Expected a 30 Hz dataset")
    names = info["features"]["action"]["names"]
    actions, states = [], []
    for path in sorted(dataset.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "action", "observation.state"]).combine_chunks()
        ids = table["episode_index"].to_numpy()
        mask = ids == episode
        if not np.any(mask):
            continue
        for channel, output in (("action", actions), ("observation.state", states)):
            values = table[channel].chunk(0).values.to_numpy().reshape(-1, len(names))
            output.append(values[mask].astype(np.float32, copy=True))
    if not actions:
        raise ValueError(f"Episode {episode} was not found")
    return np.concatenate(actions), np.concatenate(states), names


def limit_vectors(profile: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.full(24, 99.0, dtype=np.float32)
    acceleration = np.full(24, 99.0, dtype=np.float32)
    velocity[list(ARMS)] = profile["velocity"]
    acceleration[list(ARMS)] = profile["acceleration"]
    return velocity, acceleration


def rolling_project(
    actions: np.ndarray,
    anchor: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    execute_steps: int = 20,
    horizon: int = 50,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    projector = FixedHorizonChunkProjector(
        policy_hz=30,
        action_dim=actions.shape[1],
        projected_indices=ARMS,
        max_velocity=velocity,
        max_acceleration=acceleration,
    )
    previous = np.asarray(anchor, dtype=np.float32).copy()
    previous_velocity = np.zeros(actions.shape[1], dtype=np.float32)
    output, diagnostics = [], []
    for start in range(0, len(actions), execute_steps):
        raw = actions[start : min(len(actions), start + horizon)]
        projected, stats = projector.project(
            raw,
            start_position=previous,
            start_velocity=previous_velocity,
        )
        count = min(execute_steps, len(actions) - start)
        accepted = projected[:count]
        output.append(accepted)
        if len(accepted):
            old = previous
            previous = accepted[-1].copy()
            previous_velocity = (
                (accepted[-1] - accepted[-2]) * 30
                if len(accepted) > 1
                else (accepted[-1] - old) * 30
            ).astype(np.float32)
        diagnostics.append({"start": start, "executed": count, **stats})
    return np.concatenate(output), diagnostics


def interpolate_100hz(anchor: np.ndarray, actions: np.ndarray) -> np.ndarray:
    count = int(np.floor(len(actions) / 30 * 100 + 1e-9)) + 1
    times = np.arange(count, dtype=np.float64) / 100
    phase = np.minimum(times * 30, len(actions))
    completed = np.floor(phase + 1e-9).astype(int)
    alpha = np.clip(phase - completed, 0, 1)[:, None]
    result = np.empty((count, actions.shape[1]), dtype=np.float32)
    for row, (step, blend) in enumerate(zip(completed, alpha[:, 0], strict=True)):
        if step >= len(actions):
            result[row] = actions[-1]
            continue
        left = anchor if step == 0 else actions[step - 1]
        right = actions[step]
        result[row] = left + (right - left) * blend
        result[row, list(GRIPPERS)] = right[list(GRIPPERS)]
    return result


def run_limiter(
    reference: np.ndarray,
    anchor: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    limiter = JointKinematicLimiter(
        sample_hz=100,
        max_velocity=velocity,
        max_acceleration=acceleration,
        position_target_indices=GRIPPERS,
    )
    limiter.reset(anchor)
    output, speeds = [], []
    for index in range(len(reference) - 1):
        target_velocity = (reference[index + 1] - reference[index]) * 100
        output.append(limiter.update_tracking(reference[index], target_velocity))
        speeds.append(limiter.velocity)
    output.append(limiter.update_tracking(reference[-1], np.zeros(reference.shape[1], dtype=np.float32)))
    speeds.append(limiter.velocity)
    return np.asarray(output), np.asarray(speeds)


def event_rows(actions: np.ndarray) -> list[dict[str, int | str]]:
    closed = actions[:, list(GRIPPERS)] <= 0.0125
    rows = []
    for side, column in (("left", 0), ("right", 1)):
        for index in np.flatnonzero(closed[1:, column] != closed[:-1, column]) + 1:
            rows.append(
                {
                    "side": side,
                    "frame": int(index),
                    "event": "close" if closed[index, column] else "open",
                }
            )
    return sorted(rows, key=lambda row: int(row["frame"]))


def policy_dynamic_ratios(
    policy: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
) -> dict[str, float]:
    """Report intrinsic 30 Hz demands, excluding the unknown pre-episode velocity."""
    arm = np.asarray(policy[:, list(ARMS)], dtype=np.float64)
    speed = np.diff(arm, axis=0) * 30
    accel = np.diff(speed, axis=0) * 30
    velocity_ratio = np.abs(speed) / velocity[list(ARMS)]
    acceleration_ratio = np.abs(accel) / acceleration[list(ARMS)]
    return {
        "velocity_interval_exceed_percent": float(
            np.mean(np.any(velocity_ratio > 1.0001, axis=1)) * 100
        ),
        "acceleration_interval_exceed_percent": float(
            np.mean(np.any(acceleration_ratio > 1.0001, axis=1)) * 100
        ),
        "max_velocity_ratio": float(np.max(velocity_ratio)),
        "max_acceleration_ratio": float(np.max(acceleration_ratio)),
    }


def summarize(
    name: str,
    raw_policy: np.ndarray,
    policy: np.ndarray,
    policy_reference: np.ndarray,
    raw_reference: np.ndarray,
    output: np.ndarray,
    speed: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
) -> dict[str, object]:
    policy_error = np.abs(policy[:, list(ARMS)] - raw_policy[:, list(ARMS)])
    tracking_error = np.abs(output[:, list(ARMS)] - policy_reference[:, list(ARMS)])
    total_error = np.abs(output[:, list(ARMS)] - raw_reference[:, list(ARMS)])
    acceleration_actual = np.diff(speed[:, list(ARMS)], axis=0) * 100
    arm_speed = np.abs(speed[:, list(ARMS)])
    events = []
    for event in event_rows(raw_policy):
        frame = int(event["frame"])
        side_indices = LEFT if event["side"] == "left" else RIGHT
        at = min(int(round(frame / 30 * 100)), len(output) - 1)
        events.append(
            {
                **event,
                "time_s": frame / 30,
                "arm_error_at_event_rad": float(
                    np.max(np.abs(output[at, list(side_indices)] - raw_reference[at, list(side_indices)]))
                ),
            }
        )
    return {
        "name": name,
        "duration_s": len(raw_policy) / 30,
        "policy_dynamic_ratios": policy_dynamic_ratios(policy, velocity, acceleration),
        "policy_position_error_rad": {
            "p50": float(np.percentile(policy_error, 50)),
            "p95": float(np.percentile(policy_error, 95)),
            "p99": float(np.percentile(policy_error, 99)),
            "max": float(policy_error.max()),
        },
        "final_limiter_tracking_error_rad": {
            "p95": float(np.percentile(tracking_error, 95)),
            "max": float(tracking_error.max()),
        },
        "total_error_vs_raw_reference_rad": {
            "p95": float(np.percentile(total_error, 95)),
            "max": float(total_error.max()),
        },
        "final_velocity_ratio": float(np.max(arm_speed / velocity[list(ARMS)])),
        "final_acceleration_ratio": float(
            np.max(np.abs(acceleration_actual) / acceleration[list(ARMS)])
        ),
        "gripper_values_exact_at_policy_rate": bool(
            np.array_equal(policy[:, list(GRIPPERS)], raw_policy[:, list(GRIPPERS)])
        ),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--profile-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    actions, states, names = load_episode(args.dataset, args.episode)
    profile_report = json.loads(args.profile_report.read_text())
    candidates = profile_report["candidate_profiles"]
    profiles = {
        "current_uniform": candidates["current_uniform"],
        "dataset_active_p95": candidates["dataset_active_p95"],
        "dataset_active_p99_margin": candidates["dataset_active_p99_margin"],
    }
    anchor = states[0]
    cases = []
    plot_data = {}
    for name, profile in profiles.items():
        velocity, acceleration = limit_vectors(profile)
        raw_reference = interpolate_100hz(anchor, actions)
        raw_output, raw_speed = run_limiter(raw_reference, anchor, velocity, acceleration)
        cases.append(
            summarize(
                f"raw_plus_limiter__{name}",
                actions,
                actions,
                raw_reference,
                raw_reference,
                raw_output,
                raw_speed,
                velocity,
                acceleration,
            )
        )
        projected, windows = rolling_project(actions, anchor, velocity, acceleration)
        projected_reference = interpolate_100hz(anchor, projected)
        projected_output, projected_speed = run_limiter(
            projected_reference, anchor, velocity, acceleration
        )
        row = summarize(
            f"fixed_horizon__{name}",
            actions,
            projected,
            projected_reference,
            raw_reference,
            projected_output,
            projected_speed,
            velocity,
            acceleration,
        )
        row["projection_windows"] = windows
        cases.append(row)
        plot_data[name] = (projected, projected_output)

    report = {
        "schema": "nexus.fixed-horizon-episode-comparison.v1",
        "mode": "offline_only",
        "physical_commands": 0,
        "dataset": str(args.dataset.resolve()),
        "episode": args.episode,
        "frames": len(actions),
        "duration_s": len(actions) / 30,
        "joint_names": names,
        "source_events": event_rows(actions),
        "cases": cases,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    import matplotlib.pyplot as plt

    times = np.arange(len(actions)) / 30
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for axis, indices, title in (
        (axes[0], LEFT, "Left arm: maximum joint deviation from raw action"),
        (axes[1], RIGHT, "Right arm: maximum joint deviation from raw action"),
    ):
        for name, (projected, _) in plot_data.items():
            error = np.max(np.abs(projected[:, list(indices)] - actions[:, list(indices)]), axis=1)
            axis.plot(times, error, label=name)
        axis.set_ylabel("rad")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[2].plot(times, actions[:, 7] * 1000, label="left gripper")
    axes[2].plot(times, actions[:, 15] * 1000, label="right gripper")
    axes[2].set_ylabel("mm")
    axes[2].set_title("Raw gripper events (all projection cases retain exact event indices)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    axes[3].plot(times, np.max(np.abs(actions[:, list(LEFT)] - states[:, list(LEFT)]), axis=1), label="left action-state")
    axes[3].plot(times, np.max(np.abs(actions[:, list(RIGHT)] - states[:, list(RIGHT)]), axis=1), label="right action-state")
    axes[3].set_ylabel("rad")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Raw dataset action versus observation.state")
    axes[3].grid(alpha=0.25)
    axes[3].legend()
    figure.tight_layout()
    figure.savefig(args.output / "完整episode轨迹对比.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"episode": args.episode, "frames": len(actions), "cases": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
