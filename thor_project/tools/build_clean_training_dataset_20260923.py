#!/usr/bin/env python3
"""Build a conservative, auditable clean derivative of the G2 LeRobot dataset.

The source datasets are never modified.  This builder starts from the pre-SG
actions, repairs only high-confidence acquisition artefacts, applies the
existing arm-only Savitzky-Golay filter episode by episode, clips the result to
the URDF position limits, recomputes training statistics, and publishes the
result atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter


DATA_TOOLS = Path("/home/luoxiaolong/workspace/datasets/data_tools")
sys.path.insert(0, str(DATA_TOOLS))
from lerobot_dataset_finalize import finalize_dataset  # noqa: E402


ARM_SIDES = {"left": np.arange(0, 7), "right": np.arange(8, 15)}
ARM_INDICES = np.r_[0:7, 8:15]
GRIPPER_SIDES = {"left": 7, "right": 15}
WINDOW_LENGTH = 11
POLYORDER = 3
OPEN_M = 0.019
CLOSED_M = 0.006
GRIPPER_RANGE_M = (0.0, 0.025)
SHORT_PULSE_MAX_FRAMES = 15  # exclusive at 30 Hz: strictly less than 0.5 s
NO_RESPONSE_SPAN_M = 0.0005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        width = int(array.type.list_size)
        return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, width)
    return np.asarray(array.to_pylist(), dtype=np.float32)


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def episode_ranges(episodes: np.ndarray) -> list[tuple[int, int, int]]:
    boundaries = np.flatnonzero(np.diff(episodes)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(episodes)]
    return [(int(episodes[b]), int(b), int(e)) for b, e in zip(starts, ends)]


def boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def read_source_mapping(path: Path, episode_count: int) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode"])
            result[episode] = {
                "source_episode": int(row["source_episode"]),
                "source_start_frame": int(row["source_start_frame"]),
                "source_end_frame_exclusive": int(row["source_end_frame_exclusive"]),
            }
    if sorted(result) != list(range(episode_count)):
        raise ValueError("Source provenance mapping does not cover every final episode")
    return result


def source_sequences(
    episodes: np.ndarray,
    frames: np.ndarray,
    ranges: list[tuple[int, int, int]],
    mapping: dict[int, dict[str, int]],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return continuous source-frame sequences as (source_episode, source_frame, global_index)."""
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for episode, begin, end in ranges:
        item = mapping[episode]
        expected = item["source_end_frame_exclusive"] - item["source_start_frame"]
        if expected != end - begin:
            raise ValueError(f"Episode {episode}: provenance length does not match data")
        source_frame = item["source_start_frame"] + frames[begin:end]
        grouped[item["source_episode"]].extend(zip(source_frame.tolist(), range(begin, end)))

    result: list[tuple[int, np.ndarray, np.ndarray]] = []
    for source_episode, pairs in grouped.items():
        pairs.sort()
        sf = np.asarray([x[0] for x in pairs], dtype=np.int64)
        gi = np.asarray([x[1] for x in pairs], dtype=np.int64)
        cuts = np.flatnonzero(np.diff(sf) != 1) + 1
        for left, right in zip(np.r_[0, cuts], np.r_[cuts, len(sf)]):
            result.append((source_episode, sf[left:right], gi[left:right]))
    return result


def repair_gripper_state_spikes(
    state: np.ndarray,
    sequences: list[tuple[int, np.ndarray, np.ndarray]],
    episodes: np.ndarray,
    frames: np.ndarray,
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    original = state.copy()
    for source_episode, source_frames, indices in sequences:
        if len(indices) < 3:
            continue
        for side, joint in GRIPPER_SIDES.items():
            values = original[indices, joint]
            mask = (
                (np.abs(values[1:-1] - values[:-2]) > 0.010)
                & (np.abs(values[1:-1] - values[2:]) > 0.010)
                & (np.abs(values[:-2] - values[2:]) < 0.0005)
            )
            for position in np.flatnonzero(mask) + 1:
                index = int(indices[position])
                repaired = float((values[position - 1] + values[position + 1]) / 2.0)
                repairs.append({
                    "side": side,
                    "episode": int(episodes[index]),
                    "frame": int(frames[index]),
                    "source_episode": int(source_episode),
                    "source_frame": int(source_frames[position]),
                    "before_m": float(state[index, joint]),
                    "after_m": repaired,
                })
                state[index, joint] = repaired
    return repairs


def repair_gripper_short_pulses(
    action: np.ndarray,
    original_state: np.ndarray,
    sequences: list[tuple[int, np.ndarray, np.ndarray]],
    episodes: np.ndarray,
    frames: np.ndarray,
) -> list[dict[str, Any]]:
    """Remove only confirmed short O>C>O pulses with no physical response."""
    repairs: list[dict[str, Any]] = []
    original_action = action.copy()
    for source_episode, source_frames, indices in sequences:
        for side, joint in GRIPPER_SIDES.items():
            command: str | None = None
            pending_close: int | None = None
            for position, index_value in enumerate(indices):
                index = int(index_value)
                physical_target = 0.025 - float(original_action[index, joint])
                new = "C" if physical_target <= CLOSED_M else ("O" if physical_target >= OPEN_M else command)
                if command is None:
                    command = new
                    continue
                if new == command or new is None:
                    continue
                if command == "O" and new == "C":
                    pending_close = position
                elif command == "C" and new == "O" and pending_close is not None:
                    duration_frames = int(source_frames[position] - source_frames[pending_close])
                    event_indices = indices[pending_close:position]
                    state_values = original_state[event_indices, joint]
                    start_state = float(original_state[int(indices[pending_close]), joint])
                    state_span = float(np.ptp(state_values)) if len(state_values) else math.inf
                    if (
                        0 < duration_frames < SHORT_PULSE_MAX_FRAMES
                        and start_state >= OPEN_M
                        and state_span < NO_RESPONSE_SPAN_M
                    ):
                        prior_index = int(indices[pending_close - 1]) if pending_close else int(indices[pending_close])
                        replacement = float(original_action[prior_index, joint])
                        action[event_indices, joint] = replacement
                        first = int(event_indices[0])
                        last = int(event_indices[-1])
                        repairs.append({
                            "side": side,
                            "source_episode": int(source_episode),
                            "source_start_frame": int(source_frames[pending_close]),
                            "source_end_frame_exclusive": int(source_frames[position]),
                            "duration_frames": duration_frames,
                            "first_episode": int(episodes[first]),
                            "first_frame": int(frames[first]),
                            "last_episode": int(episodes[last]),
                            "last_frame": int(frames[last]),
                            "state_span_m": state_span,
                            "replacement_native_action_m": replacement,
                        })
                    pending_close = None
                command = new
    return repairs


def repair_arm_feedback_substitution(
    action: np.ndarray,
    state: np.ndarray,
    ranges: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """Interpolate short action==state islands bracketed by target discontinuities."""
    repairs: list[dict[str, Any]] = []
    original = action.copy()
    for episode, begin, end in ranges:
        for side, joints in ARM_SIDES.items():
            equal = np.all(original[begin:end, joints] == state[begin:end, joints], axis=1)
            for local_start, local_end in boolean_runs(equal):
                length = local_end - local_start
                if length > 15 or local_start == 0 or local_end >= end - begin:
                    continue
                left = begin + local_start - 1
                first = begin + local_start
                right = begin + local_end
                last = right - 1
                previous_error = float(np.max(np.abs(original[left, joints] - state[left, joints])))
                entry_step = float(np.max(np.abs(original[first, joints] - original[left, joints])))
                exit_step = float(np.max(np.abs(original[right, joints] - original[last, joints])))
                if previous_error <= 0.05 or entry_step <= 0.03 or exit_step <= 0.03:
                    continue
                for offset, index in enumerate(range(first, right), start=1):
                    alpha = offset / (length + 1)
                    action[index, joints] = (1.0 - alpha) * original[left, joints] + alpha * original[right, joints]
                repairs.append({
                    "episode": episode,
                    "side": side,
                    "start_frame": local_start,
                    "end_frame_exclusive": local_end,
                    "frames": length,
                    "previous_error_rad": previous_error,
                    "entry_step_rad": entry_step,
                    "exit_step_rad": exit_step,
                    "max_repair_rad": float(np.max(np.abs(action[first:right, joints] - original[first:right, joints]))),
                })
    return repairs


def urdf_arm_limits(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = ET.parse(path).getroot()
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for side, prefix in (("l", 20), ("r", 60)):
        for joint_number in range(1, 8):
            name = f"idx{prefix + joint_number}_arm_{side}_joint{joint_number}"
            element = root.find(f"./joint[@name='{name}']/limit")
            if element is None:
                raise ValueError(f"URDF is missing limit for {name}")
            lower.append(float(element.attrib["lower"]))
            upper.append(float(element.attrib["upper"]))
            names.append(name)
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32), names


def smooth_and_clip_arms(
    action: np.ndarray,
    ranges: list[tuple[int, int, int]],
    lower: np.ndarray,
    upper: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    result = action.copy()
    short_episodes = 0
    for _, begin, end in ranges:
        if end - begin < WINDOW_LENGTH:
            short_episodes += 1
            continue
        result[begin:end, ARM_INDICES] = savgol_filter(
            action[begin:end, ARM_INDICES],
            window_length=WINDOW_LENGTH,
            polyorder=POLYORDER,
            axis=0,
            mode="interp",
        ).astype(np.float32)
    before_clip = result[:, ARM_INDICES].copy()
    result[:, ARM_INDICES] = np.clip(before_clip, lower, upper)
    clip_mask = before_clip != result[:, ARM_INDICES]
    clip_counts = {name: int(clip_mask[:, index].sum()) for index, name in enumerate(names)}
    return result, {
        "short_episodes_not_smoothed": short_episodes,
        "position_clip_frames_per_joint": clip_counts,
        "total_joint_values_clipped": int(clip_mask.sum()),
        "max_position_clip_rad": float(np.max(np.abs(before_clip - result[:, ARM_INDICES]))),
    }


def q(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {key: float(np.quantile(values, quantile)) for key, quantile in (
        ("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)
    )}


def strict_pulse_count(
    action: np.ndarray,
    state: np.ndarray,
    sequences: list[tuple[int, np.ndarray, np.ndarray]],
) -> int:
    count = 0
    for _, source_frames, indices in sequences:
        for joint in GRIPPER_SIDES.values():
            command: str | None = None
            start: int | None = None
            for position, index in enumerate(indices):
                target = 0.025 - float(action[int(index), joint])
                new = "C" if target <= CLOSED_M else ("O" if target >= OPEN_M else command)
                if command is None:
                    command = new
                    continue
                if new == command or new is None:
                    continue
                if command == "O" and new == "C":
                    start = position
                elif command == "C" and new == "O" and start is not None:
                    duration = int(source_frames[position] - source_frames[start])
                    values = state[indices[start:position], joint]
                    if 0 < duration < SHORT_PULSE_MAX_FRAMES and state[int(indices[start]), joint] >= OPEN_M and np.ptp(values) < NO_RESPONSE_SPAN_M:
                        count += 1
                    start = None
                command = new
    return count


def residual_metrics(result: np.ndarray, raw: np.ndarray, repaired: np.ndarray) -> dict[str, Any]:
    return {
        "smoothed_vs_original_pre_sg_abs_rad": q(np.abs(result[:, ARM_INDICES] - raw[:, ARM_INDICES])),
        "smoothed_vs_repaired_pre_sg_abs_rad": q(np.abs(result[:, ARM_INDICES] - repaired[:, ARM_INDICES])),
        "repair_vs_original_pre_sg_abs_rad": q(np.abs(repaired[:, ARM_INDICES] - raw[:, ARM_INDICES])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="Current post-SG dataset (metadata/video source)")
    parser.add_argument("--pre-sg", type=Path, required=True, help="Same-frame pre-SG dataset")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    current = args.current.resolve()
    pre_sg = args.pre_sg.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    current_path = next((current / "data").rglob("*.parquet"))
    raw_path = next((pre_sg / "data").rglob("*.parquet"))
    current_table = pq.read_table(current_path)
    raw_table = pq.read_table(raw_path)
    episodes = np.asarray(current_table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    frames = np.asarray(current_table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    raw_episodes = np.asarray(raw_table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    raw_frames = np.asarray(raw_table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if not np.array_equal(episodes, raw_episodes) or not np.array_equal(frames, raw_frames):
        raise ValueError("Current and pre-SG datasets are not frame-aligned")
    ranges = episode_ranges(episodes)
    raw_action = vector_to_numpy(raw_table["action"])
    original_state = vector_to_numpy(raw_table["observation.state"])
    current_state = vector_to_numpy(current_table["observation.state"])
    if not np.array_equal(original_state, current_state):
        raise ValueError("Current and pre-SG state columns differ")
    if raw_action.shape != (len(episodes), 24) or original_state.shape != raw_action.shape:
        raise ValueError(f"Expected Nx24 arrays, got action={raw_action.shape}, state={original_state.shape}")
    if not np.isfinite(raw_action).all() or not np.isfinite(original_state).all():
        raise ValueError("Action/state contain NaN or Inf")

    mapping = read_source_mapping(args.mapping.resolve(), len(ranges))
    sequences = source_sequences(episodes, frames, ranges, mapping)
    repaired_action = raw_action.copy()
    repaired_state = original_state.copy()
    pulses_before = strict_pulse_count(raw_action, original_state, sequences)
    pulse_repairs = repair_gripper_short_pulses(repaired_action, original_state, sequences, episodes, frames)
    spike_repairs = repair_gripper_state_spikes(repaired_state, sequences, episodes, frames)
    arm_repairs = repair_arm_feedback_substitution(repaired_action, repaired_state, ranges)
    lower, upper, joint_names = urdf_arm_limits(args.urdf.resolve())
    cleaned_action, smoothing = smooth_and_clip_arms(repaired_action, ranges, lower, upper, joint_names)
    cleaned_action[:, list(GRIPPER_SIDES.values())] = np.clip(
        repaired_action[:, list(GRIPPER_SIDES.values())], *GRIPPER_RANGE_M
    )
    pulses_after = strict_pulse_count(cleaned_action, repaired_state, sequences)
    residuals = residual_metrics(cleaned_action, raw_action, repaired_action)
    position_excess = np.maximum(lower - cleaned_action[:, ARM_INDICES], cleaned_action[:, ARM_INDICES] - upper)

    validation = {
        "frames": int(len(episodes)),
        "episodes": int(len(ranges)),
        "source_continuous_segments": len(sequences),
        "strict_short_no_response_side_events_before": pulses_before,
        "strict_short_no_response_side_events_repaired": len(pulse_repairs),
        "strict_short_no_response_side_events_after": pulses_after,
        "gripper_state_single_frame_spikes_repaired": len(spike_repairs),
        "arm_feedback_substitution_intervals_repaired": len(arm_repairs),
        "arm_feedback_substitution_frames_repaired": int(sum(x["frames"] for x in arm_repairs)),
        "remaining_arm_position_excess_over_urdf_rad": float(max(0.0, position_excess.max())),
        "gripper_action_in_range": bool(np.all((cleaned_action[:, [7, 15]] >= 0) & (cleaned_action[:, [7, 15]] <= 0.025))),
        "finite": bool(np.isfinite(cleaned_action).all() and np.isfinite(repaired_state).all()),
        "smoothing": smoothing,
        "residuals": residuals,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if pulses_after != 0 or not validation["finite"] or validation["remaining_arm_position_excess_over_urdf_rad"] > 1e-7:
        raise RuntimeError("Pre-publication validation failed")

    work = output.with_name(f".{output.name}.building-{uuid.uuid4().hex[:8]}")
    work.mkdir(parents=True)
    try:
        shutil.copytree(current / "meta", work / "meta")
        shutil.copytree(current / "videos", work / "videos", copy_function=os.link)
        destination = work / "data" / current_path.relative_to(current / "data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = current_table
        state_array = fixed_list(repaired_state)
        action_output = fixed_list(cleaned_action)
        state_index = table.schema.get_field_index("observation.state")
        action_index = table.schema.get_field_index("action")
        table = table.set_column(state_index, pa.field("observation.state", state_array.type), state_array)
        table = table.set_column(action_index, pa.field("action", action_output.type), action_output)
        pq.write_table(table, destination, compression="snappy")

        # Move the eval boundary two episodes so original source episode 886 is
        # wholly in train.  Training must use this exact eval_split value.
        eval_episode_count = 178
        eval_split = eval_episode_count / len(ranges)
        finalizer = finalize_dataset(
            work,
            eval_split=eval_split,
            # The shared validator intentionally accepts only the canonical
            # representation name.  Cleaning details live in this dataset's
            # cleaning_manifest.json rather than overloading this field.
            action_representation="absolute_24d",
        )
        finalizer["dataset_root"] = str(output)
        (work / "stats_view_manifest.json").write_text(
            json.dumps(finalizer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_dataset": str(current),
            "pre_sg_dataset": str(pre_sg),
            "output_dataset": str(output),
            "source_data_sha256": {
                "current": sha256_file(current_path),
                "pre_sg": sha256_file(raw_path),
            },
            "rules": {
                "gripper_state_spike": "center differs from both neighbours by >10 mm while neighbours differ by <0.5 mm; replace with neighbour interpolation",
                "gripper_action_pulse": "O>C>O lasting <15 source frames, start state >=19 mm, state span <0.5 mm; replace pulse by preceding open target",
                "arm_feedback_substitution": "action==state run <=15 frames, bracketed, previous target-state error >0.05 rad and entry/exit target steps >0.03 rad; linearly interpolate across run",
                "arm_smoothing": {"window_length": WINDOW_LENGTH, "polyorder": POLYORDER, "mode": "interp", "per_episode": True},
                "gripper_smoothing": "disabled",
                "arm_position_limits": str(args.urdf.resolve()),
                "frame_deletion": "none",
                "cross_episode_filtering": "none; source-contiguous mapping is used only for gripper artefact detection",
            },
            "validation": validation,
            "repairs": {
                "gripper_action_short_pulses": pulse_repairs,
                "gripper_state_single_frame_spikes": spike_repairs,
                "arm_feedback_substitution_intervals": arm_repairs,
            },
            "split": {
                "eval_episode_count": eval_episode_count,
                "eval_split_required": eval_split,
                "reason": "keeps original source episode 886 wholly in train and removes the only mapped train/eval source overlap",
            },
            "finalizer": finalizer,
            "output_data_sha256": sha256_file(destination),
        }
        (work / "cleaning_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (work / "smoothing_manifest.json").write_text(
            json.dumps({
                "source_dataset": str(pre_sg),
                "operation": manifest["rules"],
                "validation": validation,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(work, output)
    except Exception:
        print(f"Incomplete build retained for diagnosis: {work}", file=sys.stderr)
        raise
    print(json.dumps({"output": str(output), "validation": validation}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
