#!/usr/bin/env python3
"""Pure-compute benchmark for the G2 tabletop safety layers.

This command never imports agibot_gdk, creates a Robot instance, or sends a
servo request.  It only loads a calibrated safety profile and evaluates FK and
collision geometry against two explicitly supplied 24D configurations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .constants import ACTION_DIM
from .safety_layer import load_safety_layer


def _configuration(value: str | None, *, label: str) -> np.ndarray:
    if value is None:
        result = np.zeros(ACTION_DIM, dtype=np.float64)
        result[[7, 15]] = -0.785
        return result
    path = Path(value).expanduser().resolve()
    if path.suffix == ".npy":
        result = np.asarray(np.load(path), dtype=np.float64)
    else:
        result = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite {ACTION_DIM}D vector")
    return result


def _summary(values_ms: list[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def _apply_deltas(configuration: np.ndarray, values: list[str]) -> np.ndarray:
    result = configuration.copy()
    for value in values:
        try:
            index_text, delta_text = value.split("=", 1)
            index = int(index_text)
            delta = float(delta_text)
        except (TypeError, ValueError) as error:
            raise ValueError("--target-delta must use INDEX=DELTA") from error
        if not 0 <= index < ACTION_DIM or not np.isfinite(delta):
            raise ValueError("--target-delta index/value is invalid")
        result[index] += delta
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety-config", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help=(
            "Resolve relative URDF/STL paths from this robot-client directory; "
            "required when benchmarking a generated profile stored elsewhere"
        ),
    )
    parser.add_argument("--current", help="24D JSON or NPY measured configuration")
    parser.add_argument("--target", help="24D JSON or NPY controller target")
    parser.add_argument(
        "--target-delta",
        action="append",
        default=[],
        help="Add DELTA to target INDEX, for example --target-delta=6=-0.5; repeatable.",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--guard-path-samples", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.guard_path_samples < 1:
        raise ValueError("iterations/path samples must be positive and warmup non-negative")

    detailed = load_safety_layer(args.safety_config, asset_root=args.asset_root)
    guard = detailed.realtime_guard(path_samples=args.guard_path_samples)
    current = _configuration(args.current, label="current")
    target = _apply_deltas(_configuration(args.target, label="target"), args.target_delta)

    for _ in range(args.warmup):
        guard.guard(current, target)
        detailed.project(current, target)

    guard_ms: list[float] = []
    projection_ms: list[float] = []
    guard_reason = ""
    projection_reason = ""
    for _ in range(args.iterations):
        started = time.perf_counter()
        guard_decision = guard.guard(current, target)
        guard_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        projection_decision = detailed.project(current, target)
        projection_ms.append((time.perf_counter() - started) * 1000)
        guard_reason = guard_decision.reason
        projection_reason = projection_decision.reason

    print(
        json.dumps(
            {
                "safety_config": str(args.safety_config.expanduser().resolve()),
                "iterations": args.iterations,
                "guard_path_samples": args.guard_path_samples,
                "collision_vertices": len(detailed.geometry.points),
                "guard_bounds": len(guard.geometry.points),
                "guard_reason": guard_reason,
                "projection_reason": projection_reason,
                "realtime_guard": _summary(guard_ms),
                "detailed_projection": _summary(projection_ms),
                "gdk_imported": False,
                "robot_commands_sent": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
