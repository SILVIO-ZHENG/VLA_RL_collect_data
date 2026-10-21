#!/usr/bin/env python3
"""Replay confirmed parcel-presence JSONL through an advisory task gate.

This is deliberately an offline/shadow utility.  It reads a previous
``run_video.py`` JSONL file and writes the actions that a future control-plane
adapter *would* consider.  It has no GDK import, no HTTP client, no SSH use,
and no capability to pause, resume, reset, or otherwise control a robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from nexus_deploy.vision.parcel.runtime import (
    GateAction,
    PresenceTaskGate,
    PresenceTaskGateConfig,
    TaskState,
)


_NS_PER_SECOND = 1_000_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="run_video.py generated JSONL with a nested presence object",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="new shadow-decision JSONL; never overwrites without --overwrite",
    )
    parser.add_argument(
        "--session-id",
        default="offline-shadow-replay",
        help="synthetic task session identifier recorded in the output",
    )
    parser.add_argument(
        "--pause-after-s",
        type=float,
        default=1.5,
        help="continuous confirmed NO_PACKAGE interval before a would_pause",
    )
    parser.add_argument(
        "--reset-after-s",
        type=float,
        default=5.0,
        help="continuous confirmed NO_PACKAGE interval before a would_reset",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_false",
        dest="auto_resume",
        help=(
            "disable the requested same-session automatic resume; by default "
            "a PACKAGE_PRESENT state may only resume a gate-owned pause"
        ),
    )
    parser.set_defaults(auto_resume=True)
    parser.add_argument(
        "--no-simulate-ack",
        action="store_true",
        help=(
            "do not acknowledge would-actions during replay; useful to inspect "
            "only the first pause proposal"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="intentionally replace an existing output JSONL",
    )
    return parser


def _timestamp_ns(payload: dict[str, Any], line_number: int) -> int:
    """Use original video time where available, not offline inference speed."""

    video_time = payload.get("video_timestamp_seconds")
    if video_time is not None:
        try:
            seconds = float(video_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"line {line_number}: video_timestamp_seconds is not numeric"
            ) from exc
        if math.isfinite(seconds) and seconds >= 0:
            # Add one nanosecond so a valid first sample at video time zero
            # remains a positive timestamp for the controller API.
            return max(1, int(round(seconds * _NS_PER_SECOND)))

    raw_timestamp = payload.get("output_timestamp_ns")
    if raw_timestamp is not None:
        try:
            timestamp = int(raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"line {line_number}: output_timestamp_ns is not an integer"
            ) from exc
        if timestamp > 0:
            return timestamp
    raise ValueError(
        f"line {line_number}: need video_timestamp_seconds or output_timestamp_ns"
    )


def _presence_state(payload: dict[str, Any], line_number: int) -> str:
    presence = payload.get("presence")
    if not isinstance(presence, dict):
        raise ValueError(f"line {line_number}: missing object field presence")
    state = presence.get("state")
    if not isinstance(state, str):
        raise ValueError(f"line {line_number}: presence.state must be a string")
    return state


def replay(
    input_path: Path,
    output_path: Path,
    *,
    config: PresenceTaskGateConfig,
    session_id: str,
    simulate_ack: bool,
    overwrite: bool,
) -> dict[str, int]:
    """Write side-effect-free gate decisions for every JSONL input result."""

    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite")

    gate = PresenceTaskGate(config)
    gate.start_session(session_id)
    task_state = TaskState.RUNNING
    action_counts = {action.value: 0 for action in GateAction}
    written = 0
    mode = "w" if overwrite else "x"
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        mode, encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: JSON record must be an object")
            timestamp_ns = _timestamp_ns(payload, line_number)
            state = _presence_state(payload, line_number)
            before = task_state
            decision = gate.observe(
                state,
                observed_at_ns=timestamp_ns,
                now_ns=timestamp_ns,
                task_state=task_state,
            )
            acknowledged: str | None = None
            if simulate_ack and decision.action is not GateAction.NONE:
                gate.acknowledge(decision.action, applied=True)
                acknowledged = decision.action.value
                if decision.action is GateAction.PAUSE:
                    task_state = TaskState.PAUSED
                elif decision.action is GateAction.RESUME:
                    task_state = TaskState.RUNNING
                elif decision.action is GateAction.RESET:
                    task_state = TaskState.ENDED
            action_counts[decision.action.value] += 1
            record = {
                "source_id": payload.get("source_id"),
                "frame_index": payload.get("frame_index"),
                "video_timestamp_seconds": payload.get("video_timestamp_seconds"),
                "presence": payload["presence"],
                "presence_gate_shadow": {
                    "task_state_before": before.value,
                    "task_state_after": task_state.value,
                    "simulated_acknowledged_action": acknowledged,
                    "decision": decision.to_dict(),
                },
            }
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return {"records": written, **action_counts}


def main() -> None:
    args = build_parser().parse_args()
    config = PresenceTaskGateConfig(
        enabled=True,
        pause_after_s=args.pause_after_s,
        reset_after_s=args.reset_after_s,
        auto_resume=args.auto_resume,
    )
    result = replay(
        args.input_jsonl,
        args.output_jsonl,
        config=config,
        session_id=args.session_id,
        simulate_ack=not args.no_simulate_ack,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
