#!/usr/bin/env python3
"""Cold-load LingBot and save a deterministic inference result for parity checks.

This diagnostic never opens a network port and never connects to a robot.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from deploy.g2a_lingbot_policy import CAMERA_KEYS, LingBotG2APolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--qwen3vl-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    os.environ["QWEN3VL_PATH"] = str(args.qwen3vl_path.resolve())

    started_at = time.perf_counter()
    policy = LingBotG2APolicy(
        args.model_path,
        robot_name="g2a_logistics_16d_absolute",
        model_fps=30.0,
        use_bf16=True,
        use_compile=False,
        fast_model_load=not args.legacy,
    )
    loaded_at = time.perf_counter()
    # Legacy construction consumes random values while meta construction does
    # not. Reset immediately before sampling so loader parity compares the
    # model, not two different diffusion-noise trajectories.
    np.random.seed(20260813)
    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    observation = {
        "state": np.zeros(24, dtype=np.float32),
        "images": {key: np.zeros((400, 640, 3), dtype=np.uint8) for key in CAMERA_KEYS},
        "task": "Pick up the package and place it on the conveyor belt with the QR code facing up.",
    }
    response = policy.infer(
        {
            "kind": "infer",
            "request_id": "loader-parity",
            "session_id": "loader-parity",
            "observation": observation,
            "control": {"enable_rtc": False},
            "reset": True,
        }
    )
    finished_at = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, response["actions"], allow_pickle=False)
    print(
        json.dumps(
            {
                "strategy": policy.metadata["model_load_strategy"],
                "load_s": loaded_at - started_at,
                "infer_s": finished_at - loaded_at,
                "shape": list(response["actions"].shape),
                "finite": bool(np.isfinite(response["actions"]).all()),
                "minimum": float(np.min(response["actions"])),
                "maximum": float(np.max(response["actions"])),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
