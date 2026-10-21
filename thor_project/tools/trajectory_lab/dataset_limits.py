"""Measure recorded DH dynamics and build an OFFLINE experiment, not hardware limits."""

import argparse
import hashlib
import json
from math import factorial
from pathlib import Path

import numpy as np

ARMS = [*range(7), *range(8, 15)]
QUANTILES = (50, 95, 99, 99.9, 100)
ORDERS = ((1, "velocity"), (2, "acceleration"), (3, "jerk"))


def derivatives(points, fps, window=9):
    """Centered cubic local-fit derivatives; discard edges, never cross episodes.

    This is a noise-reduced *estimate*, not proof of peak physical dynamics.
    The window covers 8 / 30 s at the default, with no extrapolated edge samples.
    """
    q = np.asarray(points, dtype=float)
    if q.ndim != 2 or len(q) < window or not np.isfinite(q).all():
        raise ValueError("Expected finite samples longer than derivative window")
    if type(window) is not int or window < 5 or window % 2 != 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid derivative window or fps")
    x = np.arange(-(window // 2), window // 2 + 1, dtype=float) / fps
    weights = np.linalg.pinv(np.stack([x**i for i in range(4)], axis=1))
    return {
        name: np.stack(
            [np.correlate(q[:, j], weights[order] * factorial(order), mode="valid") for j in range(q.shape[1])],
            axis=1,
        )
        for order, name in ORDERS
    }


def summarize(parts):
    values = np.concatenate(parts)
    result = dict(zip(("p50", "p95", "p99", "p999", "max"), np.percentile(values, QUANTILES, axis=0).tolist()))
    result["sample_count"] = len(values)
    return result


def experimental_limits(stats):
    """Per-joint P99 of smoothed state/target, with explicit experimental floors.

    A percentile is NOT a safety ceiling. Limits exceeding the production values
    remain diagnostic-only and need independent vendor/hardware validation.
    """
    result = {}
    for (_, name), floor in zip(ORDERS, (0.6, 2.0, 10.0)):
        state = np.asarray(stats["observation.state"]["fitted"][name]["p99"])
        action = np.asarray(stats["action"]["fitted"][name]["p99"])
        result[name] = np.maximum(floor, 1.1 * np.maximum(state, action)).tolist()
    return result


def load_profile(path, joint_names):
    profile = json.loads(Path(path).read_text())
    if profile.get("schema") != "nexus.offline-dataset-limits.v1" or profile.get("hardware_ready") is not False:
        raise ValueError("Only explicitly offline experimental profiles are accepted")
    if profile.get("joint_indices") != ARMS or profile.get("joint_names") != [joint_names[i] for i in ARMS]:
        raise ValueError("Profile joint mapping does not match dataset")
    if profile.get("units") != {"velocity": "rad/s", "acceleration": "rad/s^2", "jerk": "rad/s^3"}:
        raise ValueError("Unexpected profile units")
    arrays = {}
    for _, name in ORDERS:
        values = np.asarray(profile["limits"][name], dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Invalid profile {name}")
        arrays[name] = np.ones(16)
        arrays[name][ARMS] = values
    return profile, arrays


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=406, help="Verified checkpoint split: episodes [0, 406)")
    parser.add_argument("--window", type=int, default=9)
    args = parser.parse_args()
    info_path = args.dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["fps"] != 30 or info["features"]["action"]["shape"] != [16]:
        raise ValueError("Expected the verified 30 Hz 16D dataset")
    if not 0 < args.train_episodes < info["total_episodes"]:
        raise ValueError("Training range must leave held-out episodes")
    names = info["features"]["action"]["names"]
    if [names[i] for i in (7, 15)] != ["left_effector_position", "right_effector_position"]:
        raise ValueError("Unexpected effector mapping")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.dataset.glob("data/*/*.parquet"))
    table = pa.concat_tables(
        [
            pq.read_table(p, columns=["episode_index", "frame_index", "timestamp", "action", "observation.state"])
            for p in paths
        ]
    ).combine_chunks()
    ids = table["episode_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    timestamps = table["timestamp"].to_numpy()
    selected = ids < args.train_episodes
    expected = set(range(args.train_episodes))
    if set(ids[selected].tolist()) != expected:
        raise ValueError("Missing training episodes")
    stats = {}
    for channel in ("observation.state", "action"):
        points = table[channel].combine_chunks().values.to_numpy().reshape(-1, 16)
        parts = {method: {name: [] for _, name in ORDERS} for method in ("raw", "fitted")}
        for episode in range(args.train_episodes):
            mask = ids == episode
            if np.any(np.diff(frames[mask]) != 1) or not np.allclose(np.diff(timestamps[mask]), 1 / 30, atol=0.0001):
                raise ValueError(f"Nonuniform or unordered recording in episode {episode}")
            q = points[mask][:, ARMS].astype(float)
            fitted = derivatives(q, 30, args.window)
            for order, name in ORDERS:
                parts["raw"][name].append(abs(np.diff(q, n=order, axis=0) * 30**order).astype(np.float32))
                parts["fitted"][name].append(abs(fitted[name]).astype(np.float32))
        stats[channel] = {
            method: {name: summarize(values) for name, values in data.items()} for method, data in parts.items()
        }
        print(f"Profiled {channel}: {int(selected.sum())} training frames", flush=True)
        del parts
    profile = {
        "schema": "nexus.offline-dataset-limits.v1",
        "hardware_ready": False,
        "dataset": str(args.dataset.resolve()),
        "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "source_files": [
            {"path": str(p.relative_to(args.dataset)), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths
        ],
        "training_episode_range": [0, args.train_episodes - 1],
        "training_frames": int(selected.sum()),
        "excluded_evaluation_episode_range": [args.train_episodes, info["total_episodes"] - 1],
        "fps": 30,
        "fit_window_frames": args.window,
        "fit_window_span_s": (args.window - 1) / 30,
        "joint_indices": ARMS,
        "joint_names": [names[i] for i in ARMS],
        "units": {"velocity": "rad/s", "acceleration": "rad/s^2", "jerk": "rad/s^3"},
        "rule": "max(experimental floor [0.6, 2, 10], 1.1 * max(fitted state P99, fitted action P99)), per joint",
        "limits": experimental_limits(stats),
        "warnings": [
            "Dataset percentiles do not establish hardware-safe limits.",
            "Filtered derivatives suppress peaks; raw and fitted estimates are both retained.",
            "No physical commands, no production configuration changes.",
        ],
    }
    (args.output / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output / "limits.offline.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile["limits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
