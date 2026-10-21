#!/usr/bin/env python3
"""Analyze a robot-side read-only action trace and generate plots/reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "g2a_pi05_matplotlib"))

import matplotlib
import numpy as np

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


@dataclass(frozen=True)
class TraceRequest:
    name: str
    state: np.ndarray
    actions: np.ndarray
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, help="Timestamped action trace directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory; defaults to TRACE_DIR/analysis.",
    )
    return parser.parse_args()


def load_requests(trace_dir: Path) -> tuple[dict, list[TraceRequest]]:
    run_metadata_path = trace_dir / "run_metadata.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"Missing action trace metadata: {run_metadata_path}")
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if not run_metadata.get("safety", {}).get("read_only", False):
        raise RuntimeError("Refusing to label this as a read-only diagnostic: safety marker is missing")

    requests: list[TraceRequest] = []
    for request_dir in sorted((trace_dir / "requests").glob("request_*")):
        npz_path = request_dir / "trace.npz"
        metadata_path = request_dir / "metadata.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            continue
        with np.load(npz_path, allow_pickle=False) as payload:
            state = np.asarray(payload["state"], dtype=np.float32)
            actions = np.asarray(payload["actions"], dtype=np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"{request_dir}: invalid state shape {state.shape}")
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"{request_dir}: invalid action shape {actions.shape}")
        requests.append(
            TraceRequest(
                name=request_dir.name,
                state=state,
                actions=actions,
                metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
            )
        )
    if not requests:
        raise RuntimeError(f"No complete request traces found below {trace_dir / 'requests'}")
    return run_metadata, requests


def _flat_differences(requests: list[TraceRequest], order: int) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    for request in requests:
        if len(request.actions) > order:
            results.append(np.diff(request.actions, n=order, axis=0))
    return results


def calculate_metrics(
    requests: list[TraceRequest],
    policy_fps: float,
) -> list[dict[str, float | int | str]]:
    first_offsets = np.stack([request.actions[0] - request.state for request in requests])
    d1_chunks = _flat_differences(requests, 1)
    d2_chunks = _flat_differences(requests, 2)
    d1 = np.concatenate(d1_chunks) if d1_chunks else np.empty((0, ACTION_DIM))
    d2 = np.concatenate(d2_chunks) if d2_chunks else np.empty((0, ACTION_DIM))
    min_horizon = min(len(request.actions) for request in requests)
    repeated = np.stack([request.actions[:min_horizon] for request in requests])
    repeated_std = np.std(repeated, axis=0) if len(requests) > 1 else np.zeros_like(repeated[0])
    states = np.stack([request.state for request in requests])

    rows: list[dict[str, float | int | str]] = []
    for joint_index, joint_name in enumerate(CANONICAL_STATE_NAMES):
        joint_d1 = d1[:, joint_index] if len(d1) else np.zeros(1)
        joint_d2 = d2[:, joint_index] if len(d2) else np.zeros(1)
        reversals = 0
        for chunk in d1_chunks:
            signs = np.sign(chunk[:, joint_index])
            signs[np.abs(chunk[:, joint_index]) < 1e-5] = 0
            nonzero = signs[signs != 0]
            reversals += int(np.sum(nonzero[1:] != nonzero[:-1]))
        rows.append(
            {
                "index": joint_index,
                "joint": joint_name,
                "first_offset_abs_mean": float(np.mean(np.abs(first_offsets[:, joint_index]))),
                "first_offset_abs_max": float(np.max(np.abs(first_offsets[:, joint_index]))),
                "step_abs_p95": float(np.percentile(np.abs(joint_d1), 95)),
                "step_abs_max": float(np.max(np.abs(joint_d1))),
                "velocity_abs_p95": float(np.percentile(np.abs(joint_d1 * policy_fps), 95)),
                "velocity_abs_max": float(np.max(np.abs(joint_d1 * policy_fps))),
                "acceleration_abs_p95": float(
                    np.percentile(np.abs(joint_d2 * policy_fps**2), 95)
                ),
                "acceleration_abs_max": float(np.max(np.abs(joint_d2 * policy_fps**2))),
                "direction_reversals": reversals,
                "repeated_chunk_std_mean": float(np.mean(repeated_std[:, joint_index])),
                "repeated_chunk_std_max": float(np.max(repeated_std[:, joint_index])),
                "observed_state_range": float(np.ptp(states[:, joint_index])),
            }
        )
    return rows


def write_metrics_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_group(
    axis: plt.Axes,
    time_axis: np.ndarray,
    values: np.ndarray,
    indices: tuple[int, ...],
    *,
    state: np.ndarray | None = None,
) -> None:
    for index in indices:
        (line,) = axis.plot(time_axis, values[:, index], linewidth=1.2, label=CANONICAL_STATE_NAMES[index])
        if state is not None:
            axis.axhline(float(state[index]), color=line.get_color(), linestyle=":", alpha=0.45)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    axis.tick_params(labelsize=8)


def plot_each_request(
    requests: list[TraceRequest],
    policy_fps: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for request in requests:
        time_axis = np.arange(len(request.actions)) / policy_fps
        figure, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        for axis, (group_name, indices) in zip(axes, GROUPS.items(), strict=True):
            _plot_group(axis, time_axis, request.actions, indices, state=request.state)
            axis.set_ylabel(f"{group_name}\nposition")
        axes[-1].set_xlabel("Chunk time (s)")
        figure.suptitle(
            f"{request.name} | chunk={request.metadata.get('chunk_id')} | "
            f"RTT={request.metadata.get('round_trip_ms', 0):.1f} ms\n"
            "solid=model absolute action, dotted=observed state"
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"{request.name}.png", dpi=160)
        plt.close(figure)


def plot_each_joint(
    requests: list[TraceRequest],
    policy_fps: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(requests)))
    for joint_index, joint_name in enumerate(CANONICAL_STATE_NAMES):
        figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for color, request in zip(colors, requests, strict=True):
            time_axis = np.arange(len(request.actions)) / policy_fps
            label = request.name
            position = request.actions[:, joint_index]
            axes[0].plot(time_axis, position, color=color, alpha=0.7, linewidth=1, label=label)
            axes[0].scatter(
                [0],
                [request.state[joint_index]],
                color=color,
                marker="x",
                s=25,
            )
            if len(position) > 1:
                axes[1].plot(
                    time_axis[1:],
                    np.diff(position) * policy_fps,
                    color=color,
                    alpha=0.7,
                    linewidth=1,
                )
            if len(position) > 2:
                axes[2].plot(
                    time_axis[2:],
                    np.diff(position, n=2) * policy_fps**2,
                    color=color,
                    alpha=0.7,
                    linewidth=1,
                )
        axes[0].set_ylabel("position")
        axes[1].set_ylabel("Δaction × fps")
        axes[2].set_ylabel("Δ²action × fps²")
        axes[2].set_xlabel("Chunk time (s)")
        axes[0].legend(fontsize=6, ncol=3)
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.tick_params(labelsize=8)
        figure.suptitle(f"{joint_index:02d} {joint_name} | x marker=observed state")
        figure.tight_layout()
        figure.savefig(output_dir / f"{joint_index:02d}_{joint_name}.png", dpi=160)
        plt.close(figure)


def plot_servo_trace(trace_dir: Path, servo_fps: float, output_path: Path) -> int:
    path = trace_dir / "servo_samples.npz"
    if not path.is_file():
        return 0
    with np.load(path, allow_pickle=False) as payload:
        times = np.asarray(payload["monotonic_time"], dtype=np.float64)
        actions = np.asarray(payload["actions"], dtype=np.float32)
    if not len(times):
        return 0
    relative_time = times - times[0]
    figure, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    for axis, (group_name, indices) in zip(axes, GROUPS.items(), strict=True):
        _plot_group(axis, relative_time, actions, indices)
        axis.set_ylabel(f"{group_name}\nposition")
    axes[-1].set_xlabel("Read-only virtual servo time (s)")
    figure.suptitle(
        f"100 Hz interpolated virtual commands ({len(times)} samples, configured {servo_fps:g} Hz)\n"
        "These actions were recorded only and were not sent to GDK."
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return len(times)


def calculate_servo_metrics(trace_dir: Path) -> tuple[int, dict[str, dict[str, float]]]:
    path = trace_dir / "servo_samples.npz"
    if not path.is_file():
        return 0, {}
    with np.load(path, allow_pickle=False) as payload:
        times = np.asarray(payload["monotonic_time"], dtype=np.float64)
        actions = np.asarray(payload["actions"], dtype=np.float64)
        chunk_ids = np.asarray(payload["chunk_ids"], dtype=str)
    if len(times) < 3:
        return len(times), {}
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Servo trace timestamps must be strictly increasing")
    steps = np.diff(actions, axis=0)
    velocity = steps / dt[:, None]
    acceleration_dt = (dt[:-1] + dt[1:]) * 0.5
    acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
    boundary_rows = np.flatnonzero(
        (chunk_ids[1:] != chunk_ids[:-1]) & (chunk_ids[1:] != "") & (chunk_ids[:-1] != "")
    )

    results: dict[str, dict[str, float]] = {}
    for group_name, indices in GROUPS.items():
        selection = np.asarray(indices)
        group_velocity = np.abs(velocity[:, selection])
        group_acceleration = np.abs(acceleration[:, selection])
        boundary_step = (
            np.abs(steps[boundary_rows][:, selection])
            if len(boundary_rows)
            else np.zeros((1, len(selection)))
        )
        results[group_name] = {
            "velocity_p95": float(np.percentile(group_velocity, 95)),
            "velocity_max": float(np.max(group_velocity)),
            "acceleration_p95": float(np.percentile(group_acceleration, 95)),
            "acceleration_max": float(np.max(group_acceleration)),
            "chunk_boundary_step_max": float(np.max(boundary_step)),
        }
    return len(times), results


def _group_max(rows: list[dict], indices: tuple[int, ...], key: str) -> float:
    return max(float(rows[index][key]) for index in indices)


def write_summary(
    path: Path,
    run_metadata: dict,
    requests: list[TraceRequest],
    rows: list[dict],
    servo_sample_count: int,
    servo_metrics: dict[str, dict[str, float]],
) -> None:
    policy_fps = float(run_metadata["policy_fps"])
    rtts = [float(request.metadata.get("round_trip_ms", 0.0)) for request in requests]
    infer = [
        float(request.metadata.get("server_timing", {}).get("infer_ms", 0.0))
        for request in requests
    ]
    sampling = run_metadata.get("server_metadata", {}).get("sampling", {})
    ranked_acceleration = sorted(
        rows,
        key=lambda row: float(row["acceleration_abs_p95"]),
        reverse=True,
    )[:10]
    ranked_repeat = sorted(
        rows,
        key=lambda row: float(row["repeated_chunk_std_mean"]),
        reverse=True,
    )[:10]

    lines = [
        "# G2A PI0.5只读模型输出诊断",
        "",
        "> 安全标记：本次采集没有向GDK发送任何action。图中的100Hz数据只是客户端插值结果。",
        "",
        "## 采集信息",
        "",
        f"- 请求数：{len(requests)}",
        f"- 原始模型频率：{policy_fps:g} Hz",
        f"- 虚拟servo样本数：{servo_sample_count}",
        f"- 模型profile：`{run_metadata.get('server_metadata', {}).get('model_profile')}`",
        f"- inference模式：`{run_metadata.get('inference_mode')}`",
        f"- observation模式：`{run_metadata.get('observation_mode', 'live')}`",
        f"- prompt：`{run_metadata.get('task')}`",
        f"- fixed noise：`{sampling.get('fixed_noise', False)}`",
        f"- fixed noise seed：`{sampling.get('fixed_noise_seed')}`",
        f"- inference steps：`{sampling.get('num_inference_steps')}`",
        f"- RTT mean/p95/max：{np.mean(rtts):.1f}/{np.percentile(rtts, 95):.1f}/{np.max(rtts):.1f} ms",
        f"- infer mean/p95/max：{np.mean(infer):.1f}/{np.percentile(infer, 95):.1f}/{np.max(infer):.1f} ms",
    ]
    if run_metadata.get("observation_mode") == "frozen_first_observation":
        lines.extend(
            [
                "",
                "> 固定OBS实验中，机器人真实state保持不变，但客户端虚拟执行进度仍会前进。",
                "> 因此100Hz表中的chunk切换峰值用于观察拼接敏感性，不能直接当成真机动态执行时",
                "> 的边界跳变量；判断模型本身应优先看下面的原始30Hz指标和多次请求方差。",
            ]
        )
    lines.extend(
        [
            "",
            "## 分组抖动上界",
            "",
            "| group | max first offset | max step | max velocity | max p95 acceleration |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group_name, indices in GROUPS.items():
        lines.append(
            f"| {group_name} "
            f"| {_group_max(rows, indices, 'first_offset_abs_max'):.6f} "
            f"| {_group_max(rows, indices, 'step_abs_max'):.6f} "
            f"| {_group_max(rows, indices, 'velocity_abs_max'):.6f} "
            f"| {_group_max(rows, indices, 'acceleration_abs_p95'):.6f} |"
        )

    if servo_metrics:
        lines.extend(
            [
                "",
                "## 100Hz插值轨迹指标（未下发）",
                "",
                "| group | velocity p95/max | acceleration p95/max | max step at chunk switch |",
                "|---|---:|---:|---:|",
            ]
        )
        for group_name, values in servo_metrics.items():
            lines.append(
                f"| {group_name} "
                f"| {values['velocity_p95']:.6f}/{values['velocity_max']:.6f} "
                f"| {values['acceleration_p95']:.6f}/{values['acceleration_max']:.6f} "
                f"| {values['chunk_boundary_step_max']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## 二阶差分P95最大的关节",
            "",
            "| joint | acceleration p95 | acceleration max | reversals |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ranked_acceleration:
        lines.append(
            f"| {row['joint']} | {float(row['acceleration_abs_p95']):.6f} "
            f"| {float(row['acceleration_abs_max']):.6f} | {row['direction_reversals']} |"
        )

    lines.extend(
        [
            "",
            "## 多次请求输出差异最大的关节",
            "",
            "该指标按相同chunk时间索引比较多次请求。机器人静止时，该值越大，说明模型采样、",
            "图像变化或观测变化造成的跨chunk不一致越明显。",
            "",
            "| joint | repeated std mean | repeated std max | observed state range |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ranked_repeat:
        lines.append(
            f"| {row['joint']} | {float(row['repeated_chunk_std_mean']):.6f} "
            f"| {float(row['repeated_chunk_std_max']):.6f} "
            f"| {float(row['observed_state_range']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## 如何判断",
            "",
            "- `per_request/`中单个chunk已经频繁换向：模型原始轨迹本身不平滑。",
            "- 单个chunk平滑但`per_joint/`中不同请求差异大：重点检查随机采样和chunk重规划。",
            "- 原始30Hz曲线平滑、`servo_100hz.png`也平滑：抖动更可能来自GDK执行层或动力学约束。",
            "- 双臂平滑但effectors跳变明显：保持夹爪不动，单独验证双臂。",
            "",
            "详细数值见`joint_metrics.csv`；每次请求的输入图像和原始数组保留在trace的",
            "`requests/request_*/`目录中。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else trace_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata, requests = load_requests(trace_dir)
    policy_fps = float(run_metadata["policy_fps"])
    servo_fps = float(run_metadata["servo_fps"])
    rows = calculate_metrics(requests, policy_fps)
    write_metrics_csv(rows, output_dir / "joint_metrics.csv")
    plot_each_request(requests, policy_fps, output_dir / "per_request")
    plot_each_joint(requests, policy_fps, output_dir / "per_joint")
    servo_sample_count = plot_servo_trace(
        trace_dir,
        servo_fps,
        output_dir / "servo_100hz.png",
    )
    calculated_servo_count, servo_metrics = calculate_servo_metrics(trace_dir)
    if servo_sample_count != calculated_servo_count:
        raise RuntimeError("Servo sample count changed while analysis was running")
    write_summary(
        output_dir / "summary.md",
        run_metadata,
        requests,
        rows,
        servo_sample_count,
        servo_metrics,
    )
    print(f"Analyzed {len(requests)} model requests.")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
