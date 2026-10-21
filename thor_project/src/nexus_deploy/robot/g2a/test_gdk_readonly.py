#!/usr/bin/env python3
"""Read G2 state and cameras through GDK without sending any action."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from robot_client.constants import CANONICAL_STATE_NAMES
from robot_client.gdk_adapter import AdapterConfig, G2AAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read canonical 24D state and three GDK cameras. This script never sends actions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gdk_test_output"),
        help="Parent directory for timestamped test results.",
    )
    parser.add_argument("--samples", type=int, default=1, help="Number of observations to capture.")
    parser.add_argument("--interval-s", type=float, default=0.5, help="Delay between observations.")
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--startup-wait-s", type=float, default=2.0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.interval_s < 0:
        parser.error("--interval-s must be non-negative")
    if args.camera_timeout_ms <= 0:
        parser.error("--camera-timeout-ms must be positive")
    if args.startup_wait_s < 0:
        parser.error("--startup-wait-s must be non-negative")
    return args


def save_image(payload: dict[str, Any], path_stem: Path) -> tuple[Path, dict[str, Any]]:
    encoding = str(payload["encoding"])
    color_format = str(payload["color_format"])
    width = int(payload["width"])
    height = int(payload["height"])
    data = bytes(payload["data"])

    if encoding == "jpeg":
        output_path = path_stem.with_suffix(".jpg")
        output_path.write_bytes(data)
    elif encoding == "raw" and len(data) == width * height * 3:
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if color_format == "bgr":
            pixels = pixels[:, :, ::-1]
        elif color_format != "rgb":
            raise RuntimeError(f"Unsupported raw image color format: {color_format}")
        output_path = path_stem.with_suffix(".ppm")
        with output_path.open("wb") as stream:
            stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            stream.write(np.ascontiguousarray(pixels).tobytes())
    else:
        # Preserve an unexpected payload for diagnosis instead of discarding it.
        output_path = path_stem.with_suffix(".bin")
        output_path.write_bytes(data)

    metadata = {
        "path": str(output_path),
        "encoding": encoding,
        "color_format": color_format,
        "width": width,
        "height": height,
        "bytes": len(data),
    }
    return output_path, metadata


def print_state(sample_index: int, state: np.ndarray) -> None:
    print(f"\nSample {sample_index}: canonical 24D state")
    print(f"{'index':>5}  {'joint':<25}  {'position':>12}")
    for index, (name, value) in enumerate(zip(CANONICAL_STATE_NAMES, state, strict=True)):
        print(f"{index:5d}  {name:<25}  {float(value):12.6f}")


def main() -> int:
    args = parse_args()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / run_name
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)

    print("GDK read-only test: no control mode will be configured and no action will be sent.")
    print(f"Output directory: {run_dir}")

    adapter = G2AAdapter(
        AdapterConfig(
            hardware_mode="read-only",
            camera_timeout_ms=args.camera_timeout_ms,
            startup_wait_s=args.startup_wait_s,
        )
    )
    records: list[dict[str, Any]] = []
    try:
        adapter.connect()
        print("GDK initialized successfully.")
        for sample_index in range(args.samples):
            observation = adapter.capture_observation()
            state = observation.state
            if state.shape != (len(CANONICAL_STATE_NAMES),) or not np.isfinite(state).all():
                raise RuntimeError(
                    f"Invalid canonical state: shape={state.shape}, finite={np.isfinite(state).all()}"
                )
            print_state(sample_index, state)

            image_records: dict[str, dict[str, Any]] = {}
            for camera_name, payload in observation.images.items():
                output_path, image_metadata = save_image(
                    payload,
                    image_dir / f"sample_{sample_index:03d}_{camera_name}",
                )
                image_metadata["frame_wall_time_ns"] = observation.camera_frame_wall_time_ns[camera_name]
                image_records[camera_name] = image_metadata
                print(
                    f"Saved {camera_name}: {output_path} "
                    f"({image_metadata['width']}x{image_metadata['height']}, "
                    f"{image_metadata['encoding']}, {image_metadata['bytes']} bytes)"
                )

            records.append(
                {
                    "sample_index": sample_index,
                    "state_capture_wall_time_ns": observation.state_capture_wall_time_ns,
                    "camera_capture_start_wall_time_ns": observation.camera_capture_start_wall_time_ns,
                    "camera_capture_wall_time_ns": observation.camera_capture_wall_time_ns,
                    "camera_capture_duration_ms": observation.camera_capture_duration_ms,
                    "canonical_state": {
                        name: float(value) for name, value in zip(CANONICAL_STATE_NAMES, state, strict=True)
                    },
                    "images": image_records,
                }
            )
            if sample_index + 1 < args.samples and args.interval_s:
                time.sleep(args.interval_s)
    except Exception as error:
        print(f"GDK read-only test failed: {error}", file=sys.stderr)
        return 1
    finally:
        adapter.disconnect()

    summary = {
        "read_only": True,
        "actions_sent": 0,
        "created_at": datetime.now().astimezone().isoformat(),
        "canonical_state_order": list(CANONICAL_STATE_NAMES),
        "samples": records,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "states.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", *CANONICAL_STATE_NAMES])
        for record in records:
            state = record["canonical_state"]
            writer.writerow([record["sample_index"], *(state[name] for name in CANONICAL_STATE_NAMES)])

    print(f"\nRead-only test completed. Results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
