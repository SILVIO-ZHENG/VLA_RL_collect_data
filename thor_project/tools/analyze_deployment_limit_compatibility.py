#!/usr/bin/env python3
"""Offline dataset/model compatibility audit for the G2 deployment limiter.

This tool never imports GDK, opens a robot connection, or changes deployment
configuration.  Dataset-derived envelopes are diagnostics, not hardware-safe
limits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.trajectory_lab.dataset_limits import derivatives
except ModuleNotFoundError:  # Allow direct ``python tools/...py`` execution.
    from trajectory_lab.dataset_limits import derivatives


ARM_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
EFFECTOR_INDICES = (7, 15)
QUANTILES = (50.0, 80.0, 90.0, 95.0, 99.0, 99.9, 100.0)
QUANTILE_NAMES = ("p50", "p80", "p90", "p95", "p99", "p999", "max")


def percentile_rows(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected a non-empty finite 2D array")
    rows = np.percentile(np.abs(array), QUANTILES, axis=0)
    return {name: row.tolist() for name, row in zip(QUANTILE_NAMES, rows, strict=True)}


def masked_percentile_rows(values: np.ndarray, mask: np.ndarray) -> dict[str, list[float | None]]:
    array = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or selected.shape != array.shape:
        raise ValueError("Joint-active mask must match the derivative matrix")
    result: dict[str, list[float | None]] = {name: [] for name in QUANTILE_NAMES}
    for joint in range(array.shape[1]):
        column = np.abs(array[selected[:, joint], joint])
        rows = [None] * len(QUANTILES) if not len(column) else np.percentile(column, QUANTILES).tolist()
        for name, value in zip(QUANTILE_NAMES, rows, strict=True):
            result[name].append(value)
    return result


def contiguous_run_lengths(mask: np.ndarray) -> list[int]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 1:
        raise ValueError("Run mask must be one-dimensional")
    padded = np.r_[False, value, False].astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return (ends - starts).tolist()


def run_summary(lengths: list[int], fps: float) -> dict[str, float | int]:
    if not lengths:
        return {"count": 0, "p50_frames": 0.0, "p95_frames": 0.0, "max_frames": 0,
                "p95_ms": 0.0, "max_ms": 0.0}
    values = np.asarray(lengths, dtype=np.float64)
    p50, p95 = np.percentile(values, (50, 95))
    maximum = int(values.max())
    return {
        "count": len(lengths),
        "p50_frames": float(p50),
        "p95_frames": float(p95),
        "max_frames": maximum,
        "p95_ms": float(p95 / fps * 1000),
        "max_ms": float(maximum / fps * 1000),
    }


def _active_mask_for_order(velocity: np.ndarray, order: int, threshold: float) -> np.ndarray:
    active = np.abs(np.asarray(velocity)) >= threshold
    if order == 1:
        return active
    if order == 2:
        return active[:-1] | active[1:]
    if order == 3:
        return active[:-2] | active[1:-1] | active[2:]
    raise ValueError("Only first through third derivatives are supported")


def summarize_derivatives(
    blocks: list[dict[str, np.ndarray]],
    *,
    active_threshold: float,
    limits: dict[str, np.ndarray],
    fps: float,
) -> dict[str, Any]:
    if not blocks:
        raise ValueError("No derivative blocks")
    result: dict[str, Any] = {}
    velocity_blocks = [block["velocity"] for block in blocks]
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        values = np.concatenate([block[name] for block in blocks])
        active_parts = []
        for velocity, block in zip(velocity_blocks, blocks, strict=True):
            active = np.abs(np.asarray(velocity)) >= active_threshold
            # Local polynomial fitting estimates v/a/jerk at the same center
            # samples. Raw finite differences lose one sample per order.
            active_parts.append(
                active if len(block[name]) == len(velocity)
                else _active_mask_for_order(velocity, order, active_threshold)
            )
        active = np.concatenate(active_parts)
        task_active = np.any(active, axis=1)
        metric: dict[str, Any] = {
            "all": percentile_rows(values),
            "task_active": percentile_rows(values[task_active]) if np.any(task_active) else None,
            "joint_active": masked_percentile_rows(values, active),
            "samples": int(len(values)),
            "task_active_samples": int(task_active.sum()),
        }
        if name in limits:
            limit = np.asarray(limits[name], dtype=np.float64)
            exceeded = np.abs(values) > limit
            runs: list[int] = []
            cursor = 0
            for part in blocks:
                count = len(part[name])
                runs.extend(contiguous_run_lengths(np.any(exceeded[cursor:cursor + count], axis=1)))
                cursor += count
            metric["current_limit"] = {
                "values": limit.tolist(),
                "coordinate_exceedance_ratio": float(exceeded.mean()),
                "any_joint_exceedance_ratio": float(np.any(exceeded, axis=1).mean()),
                "per_joint_exceedance_ratio": exceeded.mean(axis=0).tolist(),
                "runs": run_summary(runs, fps),
            }
        result[name] = metric
    return result


def derivative_blocks(points: np.ndarray, episode_ids: np.ndarray, fps: float, window: int) -> dict[str, list[dict[str, np.ndarray]]]:
    q = np.asarray(points, dtype=np.float64)
    ids = np.asarray(episode_ids)
    if q.ndim != 2 or q.shape[1] != len(ARM_INDICES) or ids.shape != (len(q),):
        raise ValueError("Dataset arm points and episode ids do not align")
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    raw: list[dict[str, np.ndarray]] = []
    fitted: list[dict[str, np.ndarray]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        episode = q[start:end]
        if len(episode) < window:
            continue
        raw.append({
            "velocity": np.diff(episode, n=1, axis=0) * fps,
            "acceleration": np.diff(episode, n=2, axis=0) * fps**2,
            "jerk": np.diff(episode, n=3, axis=0) * fps**3,
        })
        fitted.append(derivatives(episode, fps, window))
    return {"raw": raw, "fitted": fitted}


def candidate_profile(
    channels: dict[str, Any],
    quantile: str,
    *,
    margin: float,
    velocity_floor: float,
    acceleration_floor: float,
) -> dict[str, list[float]]:
    if quantile not in QUANTILE_NAMES:
        raise ValueError(f"Unknown quantile: {quantile}")
    result: dict[str, list[float]] = {}
    for name, floor in (("velocity", velocity_floor), ("acceleration", acceleration_floor)):
        values = [
            np.asarray(channels[channel]["fitted"][name]["task_active"][quantile], dtype=np.float64)
            for channel in ("action", "observation.state")
        ]
        result[name] = np.maximum(floor, margin * np.maximum(*values)).tolist()
    return result


def interpolate_reference(points: np.ndarray, policy_fps: float, servo_fps: float) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("At least two trajectory points are required")
    duration = (len(value) - 1) / policy_fps
    times = np.arange(0.0, duration + 0.5 / servo_fps, 1.0 / servo_fps)
    phase = np.minimum(times * policy_fps, len(value) - 1)
    left = np.minimum(np.floor(phase).astype(int), len(value) - 2)
    alpha = (phase - left)[:, None]
    return value[left] + (value[left + 1] - value[left]) * alpha


def import_limiter(path: Path):
    spec = importlib.util.spec_from_file_location("nexus_limit_audit_action_filter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load limiter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JointKinematicLimiter


def replay_points(
    points: np.ndarray,
    *,
    policy_fps: float,
    servo_fps: float,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    limiter_class,
) -> dict[str, Any]:
    reference = interpolate_reference(points, policy_fps, servo_fps).astype(np.float32)
    limiter = limiter_class(
        sample_hz=servo_fps,
        max_velocity=np.asarray(velocity, dtype=np.float32),
        max_acceleration=np.asarray(acceleration, dtype=np.float32),
    )
    limiter.reset(reference[0])
    adopted = []
    dt = 1.0 / servo_fps
    for index in range(len(reference) - 1):
        adopted.append(limiter.update_tracking(reference[index], (reference[index + 1] - reference[index]) / dt))
    adopted_array = np.asarray(adopted)
    error = np.max(np.abs(reference[:-1] - adopted_array), axis=1)
    reference_velocity = np.diff(reference, axis=0) / dt
    reference_acceleration = np.diff(reference_velocity, axis=0) / dt
    velocity_exceeded = np.any(np.abs(reference_velocity) > np.asarray(velocity), axis=1)
    acceleration_exceeded = np.any(
        np.abs(reference_acceleration) > np.asarray(acceleration), axis=1
    )
    return {
        "samples": len(error),
        "tracking_error_rad": {
            name: float(value)
            for name, value in zip(("p50", "p95", "p99", "max"), np.percentile(error, (50, 95, 99, 100)), strict=True)
        },
        "tracking_error_exceedance_ratio": {
            "0.005_rad": float(np.mean(error > 0.005)),
            "0.02_rad": float(np.mean(error > 0.02)),
            "0.05_rad": float(np.mean(error > 0.05)),
        },
        "reference_demand_exceedance_ratio": {
            "velocity_any_joint": float(velocity_exceeded.mean()),
            "acceleration_any_joint": float(acceleration_exceeded.mean()),
        },
    }


def aggregate_replays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    total = sum(int(row["samples"]) for row in rows)
    return {
        "sequences": len(rows),
        "samples": total,
        "tracking_error_exceedance_ratio_weighted": {
            threshold: float(sum(
                row["tracking_error_exceedance_ratio"][threshold] * row["samples"] for row in rows
            ) / total)
            for threshold in ("0.005_rad", "0.02_rad", "0.05_rad")
        },
        "reference_demand_exceedance_ratio_weighted": {
            metric: float(sum(
                row["reference_demand_exceedance_ratio"][metric] * row["samples"] for row in rows
            ) / total)
            for metric in ("velocity_any_joint", "acceleration_any_joint")
        },
        "tracking_error_p95_rad_across_sequences": {
            name: float(value)
            for name, value in zip(
                ("p50", "p95", "max"),
                np.percentile([row["tracking_error_rad"]["p95"] for row in rows], (50, 95, 100)),
                strict=True,
            )
        },
    }


@dataclass
class DatasetChannel:
    ids: np.ndarray
    arms: np.ndarray
    selected_sequences: dict[int, np.ndarray]


def load_dataset_channel(dataset: Path, channel: str, train_ids: set[int], replay_ids: set[int]) -> DatasetChannel:
    import pyarrow.parquet as pq

    id_parts, arm_parts = [], []
    selected: dict[int, list[np.ndarray]] = {episode: [] for episode in replay_ids}
    for path in sorted(dataset.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", channel]).combine_chunks()
        ids = table["episode_index"].to_numpy()
        array = table[channel].chunk(0)
        width = int(getattr(array.type, "list_size", 0) or len(array[0].as_py()))
        points = array.values.to_numpy().reshape(-1, width)
        mask = np.isin(ids, list(train_ids))
        id_parts.append(ids[mask])
        arm_parts.append(points[mask][:, ARM_INDICES].astype(np.float64, copy=True))
        for episode in replay_ids.intersection(np.unique(ids).tolist()):
            selected[episode].append(points[ids == episode].copy())
    return DatasetChannel(
        ids=np.concatenate(id_parts),
        arms=np.concatenate(arm_parts),
        selected_sequences={key: np.concatenate(parts) for key, parts in selected.items() if parts},
    )


def training_episode_ids(dataset: Path, total_episodes: int) -> tuple[set[int], str]:
    manifest_path = dataset / "smoothing_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        values = manifest.get("validation", {}).get("finalizer", {}).get("train_episode_indices")
        if isinstance(values, list) and values:
            return {int(value) for value in values}, str(manifest_path)
    return set(range(total_episodes)), "all episodes (no train split manifest found)"


def evenly_spaced(values: set[int], count: int) -> set[int]:
    ordered = np.asarray(sorted(values), dtype=int)
    if count >= len(ordered):
        return set(ordered.tolist())
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return set(ordered[indices].tolist())


def markdown_report(report: dict[str, Any]) -> str:
    action = report["dataset"]["channels"]["action"]["raw"]
    profiles = report["candidate_profiles"]
    current_v = action["velocity"]["current_limit"]
    current_a = action["acceleration"]["current_limit"]
    lines = [
        "# G2 部署限速兼容性离线分析",
        "",
        "> 本报告未连接机器人、未发送动作、未修改平台或真机限速。数据分位数不是硬件安全上限。",
        "",
        "## 核心结论",
        "",
        f"- 当前真机统一限制为 `{report['current_limits']['velocity']} rad/s`、"
        f"`{report['current_limits']['acceleration']} rad/s²`，不是从当前训练数据集自动读取。",
        f"- 最新平滑训练 action 中，至少一个手臂关节超过当前速度限制的区间占 "
        f"`{current_v['any_joint_exceedance_ratio']:.2%}`；超过当前加速度限制的区间占 "
        f"`{current_a['any_joint_exceedance_ratio']:.2%}`。",
        "- Mid80/P80 只描述常见运动，且会受静止帧影响，不适合作为硬限速。",
        "- 推荐把硬件安全上限与策略兼容包络分开；下面的 P95/P99 参数均为离线候选。",
        "",
        "## 候选策略兼容包络",
        "",
        "| 候选 | 速度范围 rad/s | 加速度范围 rad/s² | 用途 |",
        "|---|---:|---:|---|",
    ]
    for name, profile in profiles.items():
        velocity = profile["velocity"]
        acceleration = profile["acceleration"]
        lines.append(
            f"| {name} | {min(velocity):.3f}～{max(velocity):.3f} | "
            f"{min(acceleration):.3f}～{max(acceleration):.3f} | {profile['description']} |"
        )
    lines += [
        "",
        "## 复演摘要",
        "",
        "复演只包含 100 Hz 插值和当前 `JointKinematicLimiter`，不包含 RTC、碰撞队列、GDK 或真实反馈。",
        "",
        "| 来源 | FPS | 参数 | 速度需求超限 | 加速度需求超限 | 误差>5mrad | 序列 P95 误差的 P95 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for source, source_rows in report["replay"].items():
        for fps, fps_rows in source_rows.items():
            for profile_name, value in fps_rows.items():
                if not value.get("samples"):
                    continue
                lines.append(
                    f"| {source} | {fps} | {profile_name} | "
                    f"{value['reference_demand_exceedance_ratio_weighted']['velocity_any_joint']:.2%} | "
                    f"{value['reference_demand_exceedance_ratio_weighted']['acceleration_any_joint']:.2%} | "
                    f"{value['tracking_error_exceedance_ratio_weighted']['0.005_rad']:.2%} | "
                    f"{value['tracking_error_p95_rad_across_sequences']['p95']:.4f} rad |"
                )
    lines += [
        "",
        "## 使用边界",
        "",
        "- 模型动作文件是独立的非 RTC chunk，不含跨 chunk 交接。",
        "- 数据集复演从每条 episode 的首帧反馈开始，限幅器在 episode 内保持状态。",
        "- 候选参数必须再受厂商/GDK允许范围约束，并经过低风险真机验证，才能进入生产。",
        "- 不应把 P80、P95 或 P99 直接解释为机器人硬件安全极限。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-actions", required=True, type=Path)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--action-filter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-window", type=int, default=9)
    parser.add_argument("--active-velocity-threshold", type=float, default=0.03)
    parser.add_argument("--replay-episodes", type=int, default=24)
    parser.add_argument("--servo-fps", type=float, default=100.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    dataset_fps = float(info["fps"])
    if not math.isclose(dataset_fps, 30.0) or int(info["features"]["action"]["shape"][0]) < 16:
        raise ValueError("Expected a 30 Hz dataset with canonical arm/effector channels")
    total_episodes = int(info["total_episodes"])
    train_ids, split_source = training_episode_ids(args.dataset, total_episodes)
    replay_ids = evenly_spaced(train_ids, args.replay_episodes)
    current = {
        "velocity": np.full(len(ARM_INDICES), 0.6),
        "acceleration": np.full(len(ARM_INDICES), 2.0),
    }

    channels: dict[str, Any] = {}
    selected: dict[str, dict[int, np.ndarray]] = {}
    for channel in ("action", "observation.state"):
        loaded = load_dataset_channel(args.dataset, channel, train_ids, replay_ids)
        blocks = derivative_blocks(loaded.arms, loaded.ids, dataset_fps, args.fit_window)
        channels[channel] = {
            method: summarize_derivatives(
                method_blocks,
                active_threshold=args.active_velocity_threshold,
                limits=current,
                fps=dataset_fps,
            )
            for method, method_blocks in blocks.items()
        }
        selected[channel] = loaded.selected_sequences

    p95 = candidate_profile(
        channels, "p95", margin=1.0, velocity_floor=0.6, acceleration_floor=2.0
    )
    p99 = candidate_profile(
        channels, "p99", margin=1.1, velocity_floor=0.6, acceleration_floor=2.0
    )
    profiles = {
        "current_uniform": {
            "velocity": current["velocity"].tolist(),
            "acceleration": current["acceleration"].tolist(),
            "description": "当前真机统一限制",
        },
        "dataset_active_p95": {
            **p95,
            "description": "最新数据运动段 fitted P95，离线比较用",
        },
        "dataset_active_p99_margin": {
            **p99,
            "description": "最新数据运动段 fitted P99×1.1，推荐兼容包络候选",
        },
    }

    limiter_class = import_limiter(args.action_filter)
    replay: dict[str, Any] = {"dataset": {}, "model_chunks": {}}
    for fps in (30.0, 15.0):
        replay["dataset"][f"{fps:g}"] = {}
        for profile_name, profile in profiles.items():
            rows = []
            for episode in sorted(replay_ids):
                if episode not in selected["action"] or episode not in selected["observation.state"]:
                    continue
                action = selected["action"][episode][:, ARM_INDICES]
                state = selected["observation.state"][episode][:, ARM_INDICES]
                points = np.vstack((state[0], action))
                rows.append(replay_points(
                    points,
                    policy_fps=fps,
                    servo_fps=args.servo_fps,
                    velocity=np.asarray(profile["velocity"]),
                    acceleration=np.asarray(profile["acceleration"]),
                    limiter_class=limiter_class,
                ))
            replay["dataset"][f"{fps:g}"][profile_name] = aggregate_replays(rows)

    model_actions = np.load(args.model_actions, allow_pickle=False)
    if model_actions.ndim != 3 or model_actions.shape[-1] < 16:
        raise ValueError("Expected model actions shaped [samples, steps, >=16]")
    for fps in (30.0, 15.0):
        replay["model_chunks"][f"{fps:g}"] = {}
        for profile_name, profile in profiles.items():
            rows = [
                replay_points(
                    chunk[:, ARM_INDICES],
                    policy_fps=fps,
                    servo_fps=args.servo_fps,
                    velocity=np.asarray(profile["velocity"]),
                    acceleration=np.asarray(profile["acceleration"]),
                    limiter_class=limiter_class,
                )
                for chunk in model_actions
            ]
            replay["model_chunks"][f"{fps:g}"][profile_name] = aggregate_replays(rows)

    report = {
        "schema": "nexus.deployment-limit-compatibility.v1",
        "mode": "offline_only",
        "hardware_ready": False,
        "physical_commands": 0,
        "dataset": {
            "path": str(args.dataset.resolve()),
            "fps": dataset_fps,
            "training_episode_count": len(train_ids),
            "training_split_source": split_source,
            "fit_window": args.fit_window,
            "active_velocity_threshold_rad_s": args.active_velocity_threshold,
            "channels": channels,
        },
        "model": {
            "actions": str(args.model_actions.resolve()),
            "report": str(args.model_report.resolve()) if args.model_report else None,
            "samples": int(model_actions.shape[0]),
            "steps": int(model_actions.shape[1]),
            "rtc": False,
            "cross_chunk_boundaries": False,
        },
        "current_limits": {"velocity": 0.6, "acceleration": 2.0},
        "candidate_profiles": profiles,
        "replay": replay,
        "limitations": [
            "No GDK, robot, network, collision worker, or physical command was used.",
            "Dataset percentiles are policy compatibility evidence, not hardware-safe limits.",
            "Model chunks are independent non-RTC samples and do not cover chunk hand-off.",
            "Replay isolates interpolation and the current kinematic limiter; it omits safety queue and feedback tracking.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "候选参数.json").write_text(json.dumps({
        "schema": "nexus.offline-policy-compatibility-envelope.v1",
        "hardware_ready": False,
        "source_report": "report.json",
        "profiles": profiles,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "报告.md").write_text(markdown_report(report), encoding="utf-8")
    print(args.output / "报告.md")


if __name__ == "__main__":
    main()
