#!/usr/bin/env python3
"""Measure native PI0.5 30 Hz output roughness from recorded observations.

This is an offline diagnostic: it loads a checkpoint, reads a local LeRobot
dataset, and never opens a socket or connects to a robot.  It intentionally
measures only *within* each 50-step model chunk, so request hand-off and the
100 Hz execution layer cannot be mistaken for model-internal jitter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from lerobot.configs import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE
from nexus_deploy.vla.pi05.policy import CAMERA_KEYS, G2APi05Policy


ARM_INDICES = np.array([*range(7), *range(8, 15)], dtype=int)


def percentiles(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    return {
        name: float(np.percentile(absolute, percentile))
        for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
    }


def dynamics(actions: np.ndarray, fps: float) -> dict[str, dict[str, float]]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 16:
        raise ValueError(f"Expected [samples, steps, >=16] actions, got {values.shape}")
    # Keep the sample and time axes intact.  Indexing ``values[:, ARM_INDICES]``
    # would accidentally select time rows, which can make a chunk diagnostic
    # look plausible while measuring the wrong signal.
    arms = values[:, :, ARM_INDICES]
    velocity = np.diff(arms, axis=1) * fps
    acceleration = np.diff(velocity, axis=1) * fps
    jerk = np.diff(acceleration, axis=1) * fps
    return {
        "position_step_rad": percentiles(np.diff(arms, axis=1)),
        "velocity_rad_s": percentiles(velocity),
        "acceleration_rad_s2": percentiles(acceleration),
        "jerk_rad_s3": percentiles(jerk),
    }


def selected_indices(dataset: LeRobotDataset, sample_count: int) -> list[int]:
    episodes = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
    # Take an interior frame from evenly distributed episodes; all images and
    # states are real recorded observations, rather than synthetic zeros.
    slots = np.linspace(0, len(starts) - 1, min(sample_count, len(starts))).round().astype(int)
    result: list[int] = []
    for slot in slots:
        start = int(starts[slot])
        end = int(starts[slot + 1]) if slot + 1 < len(starts) else len(episodes)
        result.append(start + min(10, max(0, (end - start - 1) // 2)))
    return result


def as_hwc_uint8(value) -> np.ndarray:
    image = value.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected 3D camera tensor, got {image.shape}")
    if image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image * (255 if image.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-repo-id", default="local/g2a_pi05_jitter_probe")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-noise-seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, return_uint8=True)
    fps = float(dataset.meta.fps)
    policy = G2APi05Policy(
        args.checkpoint,
        device=args.device,
        model_fps=fps,
        execution_horizon=30,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        command_head_waist=False,
        model_profile="16d_absolute_full",
        fixed_noise_seed=args.fixed_noise_seed,
        fast_model_load=True,
        effector_profile="dahuan",
    )
    actions: list[np.ndarray] = []
    samples: list[dict[str, object]] = []
    for number, index in enumerate(selected_indices(dataset, args.samples), start=1):
        item = dataset[index]
        images = {
            key: as_hwc_uint8(item[f"observation.images.{key}"])
            for key in CAMERA_KEYS
        }
        request = {
            "kind": "infer",
            "request_id": f"offline-jitter-{number}",
            "session_id": f"offline-jitter-{number}",
            "reset": True,
            "observation": {
                "state": item[OBS_STATE].detach().cpu().numpy().astype(np.float32),
                "images": images,
                "task": str(item["task"]),
            },
            "control": {"enable_rtc": False},
        }
        started = time.perf_counter()
        response = policy.infer(request)
        chunk = np.asarray(response["actions"], dtype=np.float32)[:, :16]
        actions.append(chunk)
        samples.append(
            {
                "dataset_index": int(item["index"]),
                "episode_index": int(item["episode_index"]),
                "frame_index": int(item["frame_index"]),
                "inference_ms": round((time.perf_counter() - started) * 1000, 3),
                "dynamics": dynamics(chunk[None, ...], fps),
            }
        )
        print(f"{number}/{args.samples}: ep={samples[-1]['episode_index']} frame={samples[-1]['frame_index']}", flush=True)

    stack = np.stack(actions)
    report = {
        "purpose": "offline raw PI0.5 chunk roughness; no robot connection or action execution",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_fps": fps,
        "samples": samples,
        "pooled_within_chunk_dynamics": dynamics(stack, fps),
        "scope": {
            "action_dimensions": 16,
            "arm_indices": ARM_INDICES.tolist(),
            "excluded_dimensions": [7, 15],
            "cross_chunk_boundaries": False,
            "rtc": False,
            "fixed_noise_seed": args.fixed_noise_seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(args.output.with_suffix(".npy"), stack, allow_pickle=False)
    print(json.dumps(report["pooled_within_chunk_dynamics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
