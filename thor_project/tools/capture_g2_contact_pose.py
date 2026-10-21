#!/usr/bin/env python3
"""Capture a stationary G2 contact pose through the read-only alignment agent.

The tool only calls HTTP read endpoints. It never starts a control client and
never sends an action to the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Stable sample label, e.g. near_left")
    parser.add_argument("--base-url", default="http://10.42.1.102:8080")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--interval-s", type=float, default=0.13)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--max-joint-range-rad",
        type=float,
        default=0.001,
        help="Reject a capture if any state channel moves more than this range.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/table-contact-calibration-20260922"),
    )
    args = parser.parse_args()
    if not LABEL_PATTERN.fullmatch(args.label):
        parser.error("--label must contain only letters, digits, '_' or '-'")
    if args.samples < 5:
        parser.error("--samples must be at least 5")
    if args.interval_s <= 0 or args.timeout_s <= 0:
        parser.error("interval and timeout must be positive")
    if args.max_joint_range_rad <= 0:
        parser.error("--max-joint-range-rad must be positive")
    return args


def get_json(base_url: str, path: str, timeout_s: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    with urllib_request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object JSON value")
    return value


def release_worker(base_url: str, timeout_s: float) -> None:
    url = f"{base_url.rstrip('/')}/api/alignment/release"
    request = urllib_request.Request(url, data=b"", method="POST")
    try:
        urllib_request.urlopen(request, timeout=timeout_s).close()  # noqa: S310
    except (OSError, urllib_error.URLError):
        # Capture data is already local; release is best effort and the agent
        # also releases itself after its configured idle timeout.
        pass


def capture_unique_states(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last_captured_at: float | None = None
    deadline = time.monotonic() + max(
        args.timeout_s,
        args.samples * args.interval_s * 3.0 + args.timeout_s,
    )
    while len(records) < args.samples:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out after collecting {len(records)}/{args.samples} unique states"
            )
        payload = get_json(args.base_url, "/api/alignment/state", args.timeout_s)
        captured_at = float(payload.get("captured_at", math.nan))
        state = payload.get("canonical_state")
        if not isinstance(state, dict) or not state:
            raise RuntimeError(
                "alignment agent does not expose canonical_state; deploy the current agent first"
            )
        values = {str(name): float(value) for name, value in state.items()}
        if not math.isfinite(captured_at) or not all(
            math.isfinite(value) for value in values.values()
        ):
            raise RuntimeError("alignment agent returned non-finite state data")
        if last_captured_at is None or captured_at > last_captured_at:
            records.append(
                {
                    "sample_index": len(records),
                    "captured_at": captured_at,
                    "received_at": time.time(),
                    "canonical_state": values,
                }
            )
            last_captured_at = captured_at
        time.sleep(args.interval_s)
    return records


def state_ranges(records: list[dict[str, Any]]) -> dict[str, float]:
    names = tuple(records[0]["canonical_state"])
    if any(tuple(record["canonical_state"]) != names for record in records):
        raise RuntimeError("canonical state channels changed during capture")
    return {
        name: max(float(record["canonical_state"][name]) for record in records)
        - min(float(record["canonical_state"][name]) for record in records)
        for name in names
    }


def main() -> int:
    args = parse_args()
    health = get_json(args.base_url, "/api/alignment/health", args.timeout_s)
    if health.get("task_running") is True:
        raise RuntimeError("robot GDK is occupied by another task; capture was not started")

    try:
        records = capture_unique_states(args)
    finally:
        release_worker(args.base_url, args.timeout_s)

    ranges = state_ranges(records)
    maximum_name = max(ranges, key=ranges.get)
    maximum_range = ranges[maximum_name]
    if maximum_range > args.max_joint_range_rad:
        raise RuntimeError(
            "contact pose moved during capture: "
            f"{maximum_name} range={maximum_range:.6g} rad exceeds "
            f"{args.max_joint_range_rad:.6g} rad; no sample file was written"
        )

    created_at = datetime.now().astimezone().isoformat()
    result = {
        "schema_version": 1,
        "label": args.label,
        "created_at": created_at,
        "read_only": True,
        "actions_sent": 0,
        "source": f"{args.base_url.rstrip('/')}/api/alignment/state",
        "sample_count": len(records),
        "stationarity": {
            "threshold_rad": args.max_joint_range_rad,
            "maximum_range_rad": maximum_range,
            "maximum_range_channel": maximum_name,
            "per_channel_range_rad": ranges,
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir / f"contact_{args.label}_{timestamp}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())
    print(
        f"captured {len(records)} unique read-only states; "
        f"max range {maximum_range:.6g} rad on {maximum_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
