#!/usr/bin/env python3
"""Cold-load PI0.5 and save a deterministic inference result for parity checks.

This diagnostic never opens a network port and never connects to a robot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from lerobot.configs import RTCAttentionSchedule
from nexus_deploy.vla.pi05.dataset_start_state import APPROVED_START_STATE_24D
from nexus_deploy.vla.pi05.policy import CAMERA_KEYS, G2APi05Policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()

    started_at = time.perf_counter()
    policy = G2APi05Policy(
        args.checkpoint,
        device="cuda:0",
        model_fps=30.0,
        execution_horizon=30,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        command_head_waist=False,
        model_profile="16d_absolute_full",
        fixed_noise_seed=20260813,
        fast_model_load=not args.legacy,
    )
    loaded_at = time.perf_counter()
    observation = {
        "state": np.asarray(APPROVED_START_STATE_24D, dtype=np.float32),
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
                "strategy": policy.model_load_strategy,
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
