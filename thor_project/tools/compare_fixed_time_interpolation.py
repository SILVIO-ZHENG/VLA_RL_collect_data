#!/usr/bin/env python3
"""Compare fixed-clock linear and C1 arm interpolation without robot access.

Both methods retain every source knot and its timestamp. Device-position
channels are not interpolated and their source event times are unchanged.
The result is diagnostic only and is never a deployable trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

try:
    from tools.analyze_deployment_limit_compatibility import (
        ARM_INDICES,
        EFFECTOR_INDICES,
        evenly_spaced,
        import_limiter,
        training_episode_ids,
    )
except ModuleNotFoundError:  # Allow direct ``python tools/...py`` execution.
    from analyze_deployment_limit_compatibility import (
        ARM_INDICES,
        EFFECTOR_INDICES,
        evenly_spaced,
        import_limiter,
        training_episode_ids,
    )


def sample_times(point_count: int, policy_fps: float, servo_fps: float) -> np.ndarray:
    if point_count < 2 or policy_fps <= 0 or servo_fps <= 0:
        raise ValueError("Invalid fixed-time interpolation dimensions or rates")
    duration = (point_count - 1) / policy_fps
    # Keep a uniform servo grid and stop at the last tick at or before the
    # policy horizon.  A 30 Hz knot usually lies between two 100 Hz ticks;
    # including the following tick would extrapolate past the fixed horizon.
    last_tick = int(np.floor(duration * servo_fps + 1e-12))
    return np.arange(last_tick + 1, dtype=np.float64) / servo_fps


def evaluate_reference(
    points: np.ndarray,
    times: np.ndarray,
    *,
    policy_fps: float,
    method: str,
) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2 or t.ndim != 1 or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("Expected finite points and one-dimensional sample times")
    knots = np.arange(len(q), dtype=np.float64) / policy_fps
    if np.any(t < 0) or np.any(t > knots[-1] + 1e-12):
        raise ValueError("Sample time outside fixed source duration")
    if method == "linear":
        return np.stack([np.interp(t, knots, q[:, joint]) for joint in range(q.shape[1])], axis=1)
    if method == "pchip_c1":
        return np.asarray(PchipInterpolator(knots, q, axis=0, extrapolate=False)(t), dtype=np.float64)
    raise ValueError(f"Unknown interpolation method: {method}")


def device_event_times(values: np.ndarray, policy_fps: float, midpoint: float = 0.0125) -> list[list[float]]:
    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(EFFECTOR_INDICES):
        raise ValueError("Expected Nx2 device-position targets")
    result = []
    for joint in range(q.shape[1]):
        binary = q[:, joint] <= midpoint
        # The changed target becomes active at the right source knot.
        result.append(((np.flatnonzero(binary[1:] != binary[:-1]) + 1) / policy_fps).tolist())
    return result


def reference_metrics(reference: np.ndarray, servo_fps: float, profile: dict[str, Any]) -> dict[str, float]:
    q = np.asarray(reference, dtype=np.float64)
    dt = 1.0 / servo_fps
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    velocity_limit = np.asarray(profile["velocity"], dtype=np.float64)
    acceleration_limit = np.asarray(profile["acceleration"], dtype=np.float64)
    return {
        "velocity_exceedance_ratio": float(np.mean(np.any(np.abs(velocity) > velocity_limit, axis=1))),
        "acceleration_exceedance_ratio": float(
            np.mean(np.any(np.abs(acceleration) > acceleration_limit, axis=1))
        ),
        "velocity_abs_p95_rad_s": float(np.percentile(np.abs(velocity), 95)),
        "acceleration_abs_p95_rad_s2": float(np.percentile(np.abs(acceleration), 95)),
        "jerk_abs_p95_rad_s3": float(np.percentile(np.abs(jerk), 95)),
    }


def limiter_metrics(
    reference: np.ndarray,
    *,
    servo_fps: float,
    profile: dict[str, Any],
    limiter_class,
) -> dict[str, Any]:
    q = np.asarray(reference, dtype=np.float32)
    limiter = limiter_class(
        sample_hz=servo_fps,
        max_velocity=np.asarray(profile["velocity"], dtype=np.float32),
        max_acceleration=np.asarray(profile["acceleration"], dtype=np.float32),
    )
    limiter.reset(q[0])
    velocity = np.diff(q, axis=0) * servo_fps
    adopted = np.asarray([
        limiter.update_tracking(q[index], velocity[index]) for index in range(len(q) - 1)
    ])
    error = np.max(np.abs(q[:-1] - adopted), axis=1)
    return {
        "samples": len(error),
        "tracking_error_ratio_gt_5mrad": float(np.mean(error > 0.005)),
        "tracking_error_ratio_gt_20mrad": float(np.mean(error > 0.02)),
        "tracking_error_rad": {
            name: float(value)
            for name, value in zip(
                ("p50", "p95", "p99", "max"), np.percentile(error, (50, 95, 99, 100)), strict=True
            )
        },
    }


def compare_sequence(
    points: np.ndarray,
    *,
    policy_fps: float,
    servo_fps: float,
    profiles: dict[str, Any],
    limiter_class,
) -> dict[str, Any]:
    times = sample_times(len(points), policy_fps, servo_fps)
    knots = np.arange(len(points), dtype=np.float64) / policy_fps
    source_duration = float(knots[-1])
    references = {
        method: evaluate_reference(points, times, policy_fps=policy_fps, method=method)
        for method in ("linear", "pchip_c1")
    }
    linear = references["linear"]
    rows = {}
    for method, reference in references.items():
        exact = evaluate_reference(points, knots, policy_fps=policy_fps, method=method)
        deviation = np.max(np.abs(reference - linear), axis=1)
        rows[method] = {
            "duration_s": source_duration,
            "sampled_horizon_s": float(times[-1]),
            "duration_ratio": 1.0,
            "source_knot_max_error_rad": float(np.max(np.abs(exact - points))),
            "linear_path_deviation_rad": {
                "p95": float(np.percentile(deviation, 95)),
                "max": float(np.max(deviation)),
            },
            "profiles": {
                name: {
                    "reference": reference_metrics(reference, servo_fps, profile),
                    "limiter": limiter_metrics(
                        reference,
                        servo_fps=servo_fps,
                        profile=profile,
                        limiter_class=limiter_class,
                    ),
                }
                for name, profile in profiles.items()
            },
        }
    return rows


def load_selected(dataset: Path, episode_ids: set[int]) -> dict[int, dict[str, np.ndarray]]:
    import pyarrow.parquet as pq

    result: dict[int, dict[str, list[np.ndarray]]] = {
        episode: {"action": [], "observation.state": []} for episode in episode_ids
    }
    for path in sorted(dataset.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "action", "observation.state"]).combine_chunks()
        ids = table["episode_index"].to_numpy()
        for channel in ("action", "observation.state"):
            array = table[channel].chunk(0)
            width = int(getattr(array.type, "list_size", 0) or len(array[0].as_py()))
            points = array.values.to_numpy().reshape(-1, width)
            for episode in episode_ids.intersection(np.unique(ids).tolist()):
                result[episode][channel].append(points[ids == episode].copy())
    return {
        episode: {channel: np.concatenate(parts) for channel, parts in channels.items()}
        for episode, channels in result.items()
        if all(channels.values())
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("linear", "pchip_c1"):
        method_rows = [row[method] for row in rows]
        result[method] = {
            "sequences": len(method_rows),
            "duration_ratio_max_error": float(max(abs(row["duration_ratio"] - 1) for row in method_rows)),
            "source_knot_max_error_rad": float(max(row["source_knot_max_error_rad"] for row in method_rows)),
            "linear_path_deviation_rad": {
                "p95_across_sequences": float(np.percentile(
                    [row["linear_path_deviation_rad"]["p95"] for row in method_rows], 95
                )),
                "max": float(max(row["linear_path_deviation_rad"]["max"] for row in method_rows)),
            },
            "profiles": {},
        }
        for profile_name in method_rows[0]["profiles"]:
            profile_rows = [row["profiles"][profile_name] for row in method_rows]
            weights = np.asarray([row["limiter"]["samples"] for row in profile_rows], dtype=float)
            result[method]["profiles"][profile_name] = {
                "reference_velocity_exceedance_ratio_weighted": float(np.average(
                    [row["reference"]["velocity_exceedance_ratio"] for row in profile_rows], weights=weights
                )),
                "reference_acceleration_exceedance_ratio_weighted": float(np.average(
                    [row["reference"]["acceleration_exceedance_ratio"] for row in profile_rows], weights=weights
                )),
                "tracking_error_ratio_gt_5mrad_weighted": float(np.average(
                    [row["limiter"]["tracking_error_ratio_gt_5mrad"] for row in profile_rows], weights=weights
                )),
                "tracking_error_p95_rad_across_sequences": float(np.percentile(
                    [row["limiter"]["tracking_error_rad"]["p95"] for row in profile_rows], 95
                )),
                "reference_jerk_p95_rad_s3_across_sequences": float(np.percentile(
                    [row["reference"]["jerk_abs_p95_rad_s3"] for row in profile_rows], 95
                )),
            }
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 30Hz 固定时轴 C1 插值离线对比",
        "",
        "> 未连接机器人，未发送动作，未修改生产控制链或限速参数。",
        "",
        "两种方法都保持全部源动作节点、节点时间和总时长。`pchip_c1` 只作用于14个手臂关节；夹爪事件不插值、不改时刻。",
        "",
        "| 来源 | FPS | 限制 | 插值 | 速度需求超限 | 加速度需求超限 | 误差>5mrad | 序列P95误差的P95 |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for source, fps_rows in report["results"].items():
        for fps, methods in fps_rows.items():
            for method, value in methods.items():
                for profile, metrics in value["profiles"].items():
                    lines.append(
                        f"| {source} | {fps} | {profile} | {method} | "
                        f"{metrics['reference_velocity_exceedance_ratio_weighted']:.2%} | "
                        f"{metrics['reference_acceleration_exceedance_ratio_weighted']:.2%} | "
                        f"{metrics['tracking_error_ratio_gt_5mrad_weighted']:.2%} | "
                        f"{metrics['tracking_error_p95_rad_across_sequences']:.4f} rad |"
                    )
    lines += [
        "",
        "## 结论",
        "",
        "- 不建议把 `pchip_c1` 直接接入真机控制链。它降低了 jerk，但没有消除固定30Hz时轴中的动态约束冲突。",
        "- 对模型 chunk、当前0.6 rad/s与2.0 rad/s²限制，30Hz加速度需求超限从线性的49.38%升至98.35%，限幅后跟踪误差基本未改善。",
        "- PCHIP严格通过全部源节点，最大节点误差小于3e-16 rad；模型chunk相对线性路径的最大偏移约1.75 mrad。结果变差不是节点或时长被改，而是固定节点之间的曲率仍要求过高加速度。",
        "- 夹爪事件时刻差为0。后续不得通过普通平滑移动夹爪闭合/释放事件。",
        "- 下一步应先记录同一轮真机的模型节点、100Hz参考、限幅输出和关节反馈，验证0.6/2.0是否过度保守；只有硬件验证后才能调整限制。若限制不变，则必须允许轻微改路径或统一重定时，数学上不存在纯插值解。",
        "",
        "## 验收边界",
        "",
        "- 总时长比例固定为1，不做轨迹重定时。",
        "- 所有源节点位置必须精确通过。",
        "- PCHIP 只验证 chunk/episode 内部；RTC跨 chunk 接续仍需单独分析。",
        "- 若固定时轴下仍超出已验证硬件上限，则不存在既不改路径、又不减速、还能严格满足限制的解。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-actions", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--action-filter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--servo-fps", type=float, default=100.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    train_ids, split_source = training_episode_ids(args.dataset, int(info["total_episodes"]))
    replay_ids = evenly_spaced(train_ids, args.episodes)
    dataset = load_selected(args.dataset, replay_ids)
    profile_document = json.loads(args.profiles.read_text())
    profiles = profile_document["profiles"]
    limiter_class = import_limiter(args.action_filter)
    model_actions = np.load(args.model_actions, allow_pickle=False)
    results: dict[str, Any] = {"dataset": {}, "model_chunks": {}}
    device_audit: dict[str, Any] = {"dataset": {}, "model_chunks": {}}
    for fps in (30.0, 15.0):
        dataset_rows, dataset_events = [], []
        for episode in sorted(dataset):
            action = dataset[episode]["action"]
            state = dataset[episode]["observation.state"]
            points = np.vstack((state[0, ARM_INDICES], action[:, ARM_INDICES]))
            dataset_rows.append(compare_sequence(
                points,
                policy_fps=fps,
                servo_fps=args.servo_fps,
                profiles=profiles,
                limiter_class=limiter_class,
            ))
            dataset_events.append(device_event_times(action[:, EFFECTOR_INDICES], fps))
        results["dataset"][f"{fps:g}"] = aggregate(dataset_rows)
        device_audit["dataset"][f"{fps:g}"] = {
            "sequences": len(dataset_events),
            "events": int(sum(len(side) for row in dataset_events for side in row)),
            "event_time_difference_between_methods_s": 0.0,
        }

        model_rows = [
            compare_sequence(
                chunk[:, ARM_INDICES],
                policy_fps=fps,
                servo_fps=args.servo_fps,
                profiles=profiles,
                limiter_class=limiter_class,
            )
            for chunk in model_actions
        ]
        model_events = [device_event_times(chunk[:, EFFECTOR_INDICES], fps) for chunk in model_actions]
        results["model_chunks"][f"{fps:g}"] = aggregate(model_rows)
        device_audit["model_chunks"][f"{fps:g}"] = {
            "sequences": len(model_events),
            "events": int(sum(len(side) for row in model_events for side in row)),
            "event_time_difference_between_methods_s": 0.0,
        }
    report = {
        "schema": "nexus.fixed-time-c1-comparison.v1",
        "mode": "offline_only",
        "hardware_ready": False,
        "physical_commands": 0,
        "dataset": str(args.dataset.resolve()),
        "training_split_source": split_source,
        "selected_episodes": sorted(replay_ids),
        "model_actions": str(args.model_actions.resolve()),
        "profiles": str(args.profiles.resolve()),
        "servo_fps": args.servo_fps,
        "methods": {
            "linear": "current fixed-clock piecewise linear reference",
            "pchip_c1": "fixed-clock shape-preserving C1 cubic arm reference",
        },
        "device_events": device_audit,
        "results": results,
        "limitations": [
            "No GDK, network, collision queue, RTC inference, or physical commands.",
            "Model chunks are independent and do not prove cross-chunk continuity.",
            "A data-derived profile is not a hardware safety limit.",
            "The limiter replay starts each independent sequence with zero velocity.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "报告.md").write_text(markdown(report), encoding="utf-8")
    print(args.output / "报告.md")


if __name__ == "__main__":
    main()
