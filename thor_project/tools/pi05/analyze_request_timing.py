#!/usr/bin/env python3
"""Summarize G2A client/server JSONL request timing logs as Markdown."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _image_transform_ms(record: dict[str, Any]) -> float:
    timing = record.get("image_transport_timing", {})
    if not isinstance(timing, dict):
        return 0.0
    return sum(float(item.get("total_ms", 0.0)) for item in timing.values() if isinstance(item, dict))


def _render_table(
    records: list[dict[str, Any]],
    server_by_id: dict[str, dict[str, Any]],
) -> str:
    metrics: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
        ("Observation → response", lambda row: row.get("observation_to_response_ms")),
        ("Client RTT", lambda row: row.get("round_trip_ms")),
        ("Camera capture", lambda row: row.get("camera_capture_ms")),
        ("Client image decode/resize/encode", _image_transform_ms),
        ("Request serialization", lambda row: _nested(row, "client_timing", "request_serialize_ms")),
        ("WebSocket send", lambda row: _nested(row, "client_timing", "websocket_send_ms")),
        ("Response wait", lambda row: _nested(row, "client_timing", "response_wait_ms")),
        ("Response deserialization", lambda row: _nested(row, "client_timing", "response_deserialize_ms")),
        ("Server request deserialization", lambda row: _nested(row, "server_timing", "request_deserialize_ms")),
        ("Server worker queue", lambda row: _nested(row, "server_timing", "queue_ms")),
        ("Server worker total", lambda row: _nested(row, "server_timing", "worker_ms")),
        ("Server preprocessing", lambda row: _nested(row, "server_timing", "preprocess_ms")),
        ("Model inference", lambda row: _nested(row, "server_timing", "infer_ms")),
        ("Server postprocessing", lambda row: _nested(row, "server_timing", "postprocess_ms")),
        ("Response serialization", lambda row: _nested(row, "server_timing", "response_serialize_ms")),
        ("Unattributed transport/scheduling", lambda row: row.get("unattributed_transport_and_scheduling_ms")),
    ]
    if server_by_id:
        metrics.extend(
            [
                (
                    "Server WebSocket send (exact)",
                    lambda row: server_by_id.get(str(row.get("request_id")), {}).get("websocket_send_ms"),
                ),
                (
                    "Server handler total (exact)",
                    lambda row: server_by_id.get(str(row.get("request_id")), {}).get("handler_total_ms"),
                ),
            ]
        )

    lines = [
        "| Stage | Samples | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, getter in metrics:
        values: list[float] = []
        for record in records:
            value = getter(record)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if not values:
            continue
        lines.append(
            f"| {label} | {len(values)} | {sum(values) / len(values):.1f} | "
            f"{_percentile(values, 50):.1f} | {_percentile(values, 95):.1f} | {max(values):.1f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client_timing_log", type=Path)
    parser.add_argument("--server-timing-log", type=Path, default=None)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.client_timing_log)
    if not args.include_warmup:
        records = [record for record in records if record.get("phase") != "warmup"]
    server_records = _read_jsonl(args.server_timing_log) if args.server_timing_log else []
    server_by_id = {str(record.get("request_id")): record for record in server_records}
    markdown = "\n".join(
        [
            "# G2A request timing summary",
            "",
            f"- Client log: `{args.client_timing_log.expanduser().resolve()}`",
            f"- Requests: `{len(records)}`",
            f"- Warmup included: `{args.include_warmup}`",
            "",
            _render_table(records, server_by_id),
            "",
        ]
    )
    if args.output is None:
        print(markdown)
        return
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
