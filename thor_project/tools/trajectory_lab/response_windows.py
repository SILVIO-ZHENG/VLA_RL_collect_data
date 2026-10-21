"""Replay actual saved reply windows independently; no RTC server or SDK calls.

This is not a closed-loop counterfactual. Moving boundary derivatives come from
the recorded software limiter, not physical feedback or a hardware certificate.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .replay import plan_recorded
from .trajectory import Limits, PlanningError, path_error_bound


def replay(path, profile_path, limit):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if records[-1].get("record_type") != "trace_end" or records[-1].get("dropped") or records[-1].get("truncated"):
        raise ValueError("Complete, non-dropping trace required")
    header = next(r for r in records if r.get("record_type") == "trace_start")
    controls = [r for r in records if r.get("record_type") == "control"]
    replies = [r for r in records if r.get("record_type") == "inference_response" and r.get("installed")]
    if not replies:
        raise ValueError("Numerical inference replies are missing; sampled controls are not raw chunks")
    times = np.array([r["monotonic"] for r in controls])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Missing or non-monotonic control records")
    profile = json.loads(profile_path.read_text())
    if profile.get("hardware_ready") is not False or profile.get("joint_indices") != [*range(7), *range(8, 15)]:
        raise ValueError("Expected the existing offline per-arm dataset profile")
    vectors = []
    for name in ("velocity", "acceleration", "jerk"):
        value = np.ones(16)
        value[profile["joint_indices"]] = profile["limits"][name]
        vectors.append(value)
    lo, hi = np.full(16, -4.0), np.full(16, 4.0)
    lo[[7, 15]], hi[[7, 15]] = 0, 0.025
    limits = Limits(*vectors, lo, hi)
    selected = np.unique(np.linspace(0, len(replies) - 1, min(limit, len(replies)), dtype=int))
    results = []
    for index in selected:
        reply = replies[index]
        at = reply["installed_monotonic"]
        previous = int(np.searchsorted(times, at, side="right")) - 1
        v, a = np.zeros(16), np.zeros(16)
        if previous >= 0:
            record = controls[previous]
            q = np.asarray(record["adopted_target"], dtype=float)[:16]
            if record.get("planner_velocity") is not None:
                v = np.asarray(record["planner_velocity"], dtype=float)[:16]
                if (
                    previous > 0
                    and controls[previous - 1].get("planner_velocity") is not None
                    and not record.get("filter_reanchored")
                ):
                    # Limiter velocities advance by the nominal servo step,
                    # irrespective of diagnostic thread scheduling jitter.
                    a = (v - np.asarray(controls[previous - 1]["planner_velocity"])[:16]) * header["servo_fps"]
        else:
            # Before the first simulated command the observed pose is the hold.
            q = np.asarray(reply["observation_state"], dtype=float)[:16]
        v[[7, 15]], a[[7, 15]] = 0, 0
        dropped = int(reply["dropped_steps"])
        target = np.asarray(reply["control_targets"], dtype=float)[dropped:, :16]
        points = np.vstack([q, target])
        if np.any(points[:, [7, 15]] < -1e-8) or np.any(points[:, [7, 15]] > 0.025 + 1e-8):
            raise ValueError("Physical effector convention or limits mismatch")
        points[:, [7, 15]] = np.clip(points[:, [7, 15]], 0, 0.025)
        phases = np.arange(dropped - 1, dropped + len(target), dtype=float)
        row = dict(
            chunk_id=reply["chunk_id"],
            reply_index=int(index),
            dropped_steps=dropped,
            boundary_source="recorded software limiter",
            rtc=reply["rtc"],
        )
        started = time.perf_counter()
        try:
            trajectory, attempts = plan_recorded(
                points,
                phases=phases,
                policy_fps=header["policy_fps"],
                limits=limits,
                device_indices=(7, 15),
                start_velocity=v,
                start_acceleration=a,
                smoothing_window=9,
                constrained_fallback=True,
            )
            row.update(
                status="bounded",
                attempts=attempts,
                duration_s=trajectory.duration,
                duration_stretch=trajectory.duration / (len(target) / header["policy_fps"]),
                bounds=trajectory.bounds(),
                path_error_rad=path_error_bound(trajectory),
            )
            # Every source event must still be present on the new common clock.
            for k, phase in enumerate(phases):
                t = np.interp(phase, trajectory.phases, trajectory.times)
                np.testing.assert_array_equal(
                    trajectory.sample(min(t + 1e-9, trajectory.duration)).position[[7, 15]], points[k, [7, 15]]
                )
        except PlanningError as error:
            row.update(
                status="rejected",
                error=str(error),
                points=points.tolist(),
                phases=phases.tolist(),
                start_velocity=v.tolist(),
                start_acceleration=a.tolist(),
            )
        row["planning_ms"] = (time.perf_counter() - started) * 1000
        results.append(row)
    return dict(
        source=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_replies=len(replies),
        evaluated=len(results),
        policy_fps=header["policy_fps"],
        inference_mode=header["inference_mode"],
        hardware_ready=False,
        physical_commands=0,
        closed_loop_counterfactual=False,
        limits_profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        cases=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--limits-profile", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("limit must be 1..200")
    from scipy.optimize import linprog

    linprog([1.0], bounds=[(0, 1)], method="highs")  # prepare optional native solver
    report = replay(args.trace, args.limits_profile, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(evaluated=report["evaluated"], bounded=sum(c["status"] == "bounded" for c in report["cases"])),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
