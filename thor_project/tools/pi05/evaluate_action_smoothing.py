#!/usr/bin/env python3
"""Evaluate causal low-pass smoothing on one recorded PI0.5 action chunk."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "g2a_pi05_matplotlib"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from nexus_deploy.robot.g2a.constants import (  # noqa: E402
        ACTION_DIM,
        CANONICAL_STATE_NAMES,
    )
except ModuleNotFoundError:
    from robot_client.constants import (  # type: ignore[no-redef] # noqa: E402, I001
        ACTION_DIM,
        CANONICAL_STATE_NAMES,
    )

GROUPS = {
    "left_arm": tuple(range(0, 7)),
    "effectors": (7, 15),
    "right_arm": tuple(range(8, 15)),
    "head_waist": tuple(range(16, 24)),
}
DEFAULT_CUTOFFS_HZ = (2.0, 3.0, 4.0, 6.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--policy-fps", type=float, default=None)
    parser.add_argument("--servo-fps", type=float, default=None)
    parser.add_argument("--cutoff-hz", type=float, nargs="+", default=DEFAULT_CUTOFFS_HZ)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resample_linear(
    state: np.ndarray,
    actions: np.ndarray,
    *,
    policy_fps: float,
    servo_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ActionPlan timing: action[0] is reached after one policy period."""
    policy_times = np.arange(len(actions) + 1, dtype=np.float64) / policy_fps
    policy_targets = np.concatenate((state[None], actions), axis=0)
    servo_count = int(math.floor(policy_times[-1] * servo_fps)) + 1
    servo_times = np.arange(servo_count, dtype=np.float64) / servo_fps
    servo_actions = np.empty((servo_count, ACTION_DIM), dtype=np.float64)
    for joint_index in range(ACTION_DIM):
        servo_actions[:, joint_index] = np.interp(
            servo_times,
            policy_times,
            policy_targets[:, joint_index],
        )
    return servo_times, servo_actions


def low_pass(
    actions: np.ndarray,
    *,
    cutoff_hz: float,
    servo_fps: float,
) -> np.ndarray:
    """Apply a causal one-pole low-pass filter initialized at the measured state."""
    if cutoff_hz <= 0 or cutoff_hz >= servo_fps * 0.5:
        raise ValueError("cutoff_hz must be between zero and the Nyquist frequency")
    dt = 1.0 / servo_fps
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    filtered = np.empty_like(actions)
    filtered[0] = actions[0]
    for step in range(1, len(actions)):
        filtered[step] = filtered[step - 1] + alpha * (actions[step] - filtered[step - 1])
    return filtered


def trajectory_metrics(
    candidate: np.ndarray,
    raw: np.ndarray,
    *,
    servo_fps: float,
    indices: tuple[int, ...],
) -> dict[str, float]:
    selection = np.asarray(indices)
    values = candidate[:, selection]
    velocity = np.diff(values, axis=0) * servo_fps
    acceleration = np.diff(values, n=2, axis=0) * servo_fps**2
    error = candidate[:, selection] - raw[:, selection]
    return {
        "velocity_p95": float(np.percentile(np.abs(velocity), 95)),
        "velocity_max": float(np.max(np.abs(velocity))),
        "acceleration_p95": float(np.percentile(np.abs(acceleration), 95)),
        "acceleration_max": float(np.max(np.abs(acceleration))),
        "tracking_rmse": float(np.sqrt(np.mean(error**2))),
        "tracking_max": float(np.max(np.abs(error))),
        "endpoint_error_max": float(np.max(np.abs(error[-1]))),
    }


def evaluate(
    raw: np.ndarray,
    *,
    servo_fps: float,
    cutoffs_hz: list[float] | tuple[float, ...],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]]]:
    trajectories = {"raw_linear": raw}
    for cutoff in cutoffs_hz:
        trajectories[f"lowpass_{cutoff:g}hz"] = low_pass(
            raw,
            cutoff_hz=float(cutoff),
            servo_fps=servo_fps,
        )
    metrics = {
        method: {
            group_name: trajectory_metrics(
                trajectory,
                raw,
                servo_fps=servo_fps,
                indices=indices,
            )
            for group_name, indices in GROUPS.items()
        }
        for method, trajectory in trajectories.items()
    }
    return trajectories, metrics


def write_summary(
    path: Path,
    *,
    trace_dir: Path,
    request_index: int,
    policy_fps: float,
    servo_fps: float,
    metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    lines = [
        "# PI0.5 action低通滤波离线评估",
        "",
        f"- trace：`{trace_dir}`",
        f"- request：`{request_index}`",
        f"- policy/servo频率：{policy_fps:g}/{servo_fps:g} Hz",
        "",
        "> 这是离线模拟，没有向机器人发送action。固定OBS trace的chunk边界不代表",
        "> 真实动态执行，因此这里只分析单个chunk内部。",
        "",
    ]
    for group_name in ("left_arm", "right_arm", "effectors"):
        lines.extend(
            [
                f"## {group_name}",
                "",
                "| method | velocity p95/max | acceleration p95/max | RMSE | max error | endpoint error |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, by_group in metrics.items():
            values = by_group[group_name]
            lines.append(
                f"| {method} "
                f"| {values['velocity_p95']:.6f}/{values['velocity_max']:.6f} "
                f"| {values['acceleration_p95']:.6f}/{values['acceleration_max']:.6f} "
                f"| {values['tracking_rmse']:.6f} "
                f"| {values['tracking_max']:.6f} "
                f"| {values['endpoint_error_max']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释",
            "",
            "- cutoff越低，加速度越小，但轨迹滞后越明显。",
            "- `tracking/endpoint error`是相对原始模型轨迹的偏差，不是相对ground truth的误差。",
            "- 真机推荐值需要结合任务成功率和安全速度继续验证，不能只按平滑指标选择。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_worst_joints(
    output_dir: Path,
    *,
    times: np.ndarray,
    trajectories: dict[str, np.ndarray],
    raw: np.ndarray,
    servo_fps: float,
) -> None:
    raw_acceleration = np.abs(np.diff(raw, n=2, axis=0) * servo_fps**2)
    for group_name, indices in GROUPS.items():
        if group_name == "head_waist":
            continue
        local = np.asarray(indices)
        worst_local = int(np.argmax(np.percentile(raw_acceleration[:, local], 95, axis=0)))
        joint_index = indices[worst_local]
        figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for method, values in trajectories.items():
            position = values[:, joint_index]
            axes[0].plot(times, position, label=method, linewidth=1.2)
            axes[1].plot(times[1:], np.diff(position) * servo_fps, linewidth=1.1)
            axes[2].plot(times[2:], np.diff(position, n=2) * servo_fps**2, linewidth=1.1)
        axes[0].set_ylabel("position")
        axes[1].set_ylabel("velocity")
        axes[2].set_ylabel("acceleration")
        axes[2].set_xlabel("time (s)")
        axes[0].legend(fontsize=8, ncol=3)
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.suptitle(f"{group_name}: {CANONICAL_STATE_NAMES[joint_index]}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{group_name}_{joint_index:02d}.png", dpi=160)
        plt.close(figure)


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    run_metadata = json.loads((trace_dir / "run_metadata.json").read_text(encoding="utf-8"))
    policy_fps = float(args.policy_fps or run_metadata["policy_fps"])
    servo_fps = float(args.servo_fps or run_metadata["servo_fps"])
    request_dir = trace_dir / "requests" / f"request_{args.request_index:04d}"
    with np.load(request_dir / "trace.npz", allow_pickle=False) as payload:
        state = np.asarray(payload["state"], dtype=np.float64)
        actions = np.asarray(payload["actions"], dtype=np.float64)
    times, raw = resample_linear(
        state,
        actions,
        policy_fps=policy_fps,
        servo_fps=servo_fps,
    )
    trajectories, metrics = evaluate(
        raw,
        servo_fps=servo_fps,
        cutoffs_hz=args.cutoff_hz,
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else trace_dir / "smoothing_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(
        output_dir / "summary.md",
        trace_dir=trace_dir,
        request_index=args.request_index,
        policy_fps=policy_fps,
        servo_fps=servo_fps,
        metrics=metrics,
    )
    plot_worst_joints(
        output_dir,
        times=times,
        trajectories=trajectories,
        raw=raw,
        servo_fps=servo_fps,
    )
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
