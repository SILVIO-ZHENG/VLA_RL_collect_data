"""Dataset-window handoff experiment with recorded latency, NOT real RTC output.

Simulated 100 Hz clock, measured local planner cost. Optional existing geometry
is evaluated in shadow mode: no SDK import, no hardware or model-server call.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from .dataset_limits import load_profile
from .replay import plan_recorded
from .rolling import RollingTimeline
from .trajectory import Limits, PlanningError


def stats(values):
    if not values:
        return None
    return dict(zip(("p50", "p95", "p99", "max"), np.percentile(values, [50, 95, 99, 100]).tolist()))


def read_delays(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    delays = [
        float(r["observation_to_response_ms"]) / 1000 for r in rows if r.get("used_by_latency_estimator") is True
    ]
    if not delays or not np.isfinite(delays).all() or min(delays) < 0:
        raise ValueError("No valid recorded observation-to-response latencies")
    return delays


def read_episode(dataset, episode):
    import pyarrow.parquet as pq

    parts, states = [], []
    for path in sorted(dataset.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "action", "observation.state"])
        mask = table["episode_index"].to_numpy() == episode
        if mask.any():
            parts.append(table["action"].combine_chunks().values.to_numpy().reshape(-1, 16)[mask])
            states.append(table["observation.state"].combine_chunks().values.to_numpy().reshape(-1, 16)[mask])
    if not parts:
        raise ValueError(f"Missing episode {episode}")
    action = np.concatenate(parts).astype(float)
    anchor = np.concatenate(states)[0].astype(float)
    for q in (action, anchor[None]):
        if not np.isfinite(q).all() or np.any(q[:, [7, 15]] < -1e-8) or np.any(q[:, [7, 15]] > 0.025 + 1e-8):
            raise ValueError("Invalid recorded values")
        q[:, [7, 15]] = np.clip(q[:, [7, 15]], 0, 0.025)
    action[:, [7, 15]] = 0.025 - action[:, [7, 15]]
    return np.vstack((anchor, action))


def shadow_geometry(source, profile, anchor_log):
    spec = importlib.util.spec_from_file_location("offline_geometry_probe", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Explicitly permit the unchanged, non-calibrated read-only TEMPLATE.
    layer = module.load_safety_layer(
        profile, allow_read_only_ready=True, effector_envelope_ranges={7: (0, 0.025), 15: (0, 0.025)}
    )
    with anchor_log.open() as stream:
        anchor = next(
            np.asarray(row["measured_state"], dtype=float)
            for line in stream
            if (row := json.loads(line)).get("measured_state") is not None
        )
    if anchor.shape != (24,) or not np.isfinite(anchor).all():
        raise ValueError("Geometry requires a recorded finite 24D anchor")
    records = []

    def check(previous, target):
        a, b = anchor.copy(), anchor.copy()
        a[:16], b[:16] = previous, target
        started = time.perf_counter()
        decision = layer.guard(a, b)
        records.append(
            {
                "ms": (time.perf_counter() - started) * 1000,
                "blocked": decision.hard_limit_triggered,
                "reason": decision.reason,
                "clearance_m": decision.distance_to_surface,
            }
        )
        # Shadow benchmark does NOT release robot actions or claim a safe path.
        return True

    return check, records


def run_case(
    points, *, fps, limits, delays, seconds, lead_s, checker, constrained_fallback=False, adaptive_lead=False
):
    options = dict(policy_fps=fps, limits=limits, device_indices=(7, 15), smoothing_window=9)
    initial = time.perf_counter()
    trajectory, _ = plan_recorded(points[:51], **options)
    initial_ms = (time.perf_counter() - initial) * 1000
    timeline = RollingTimeline(trajectory)
    events, ticks = [], []
    waiting = None
    next_request = 0.0
    completed = False
    first_failure = None
    for index in range(1, int(seconds * 100) + 1):
        now = index / 100
        previous_phase = timeline.last.phase
        started = time.perf_counter()
        sample = timeline.adopt(now, checker)
        ticks.append(
            {
                "t": now,
                "phase": sample.phase,
                "sample_and_check_ms": (time.perf_counter() - started) * 1000,
                "source_id": timeline.active.source_id,
                "terminal_hold": now >= timeline.start + timeline.active.duration,
                "gripper": sample.position[[7, 15]].tolist(),
            }
        )
        if sample.phase < previous_phase - 1e-8:
            raise AssertionError("Source phase ran backwards")
        if waiting is not None and now >= waiting[0]:
            _, request, proposed, event = waiting
            if proposed is not None:
                try:
                    timeline.submit(proposed, request, now=now)
                    event["status"] = "scheduled"
                except PlanningError as error:
                    event.update(status="response_rejected", error=str(error))
                    timeline.cancel(request)
            else:
                timeline.cancel(request)
            waiting = None
            next_request = now + 0.05
        if waiting is not None or timeline.scheduled is not None or now < next_request:
            continue
        try:
            reserve_lead = lead_s
            if adaptive_lead and len(events) >= 5:
                history = events[-20:]
                reserve_lead = np.clip(
                    (
                        np.percentile([e["recorded_response_delay_ms"] for e in history], 95)
                        + np.percentile([e["planning_ms"] for e in history], 95)
                        + 40
                    )
                    / 1000,
                    0.4,
                    0.8,
                )
            request = timeline.reserve(now, lead_s=float(reserve_lead), allow_terminal_hold=True)
        except PlanningError:
            next_request = now + 0.1
            continue
        phase = request.sample.phase
        first = int(np.floor(phase + 1e-7)) + 1
        if first >= len(points):
            timeline.cancel(request)
            # Reserving the final event is not executing it. Consume the
            # existing prefix before reporting the dataset as completed.
            if timeline.last.phase >= len(points) - 1 - 1e-9:
                completed = True
                break
            next_request = now + 0.05
            continue
        stop = min(first + 50, len(points))
        q = np.vstack((request.sample.position, points[first:stop]))
        phases = np.r_[phase, np.arange(first, stop)]
        event = {
            "requested_at_s": now,
            "handoff_at_s": request.at,
            "source_phase": phase,
            "requested_lead_s": float(reserve_lead),
            "terminal_handoff": request.at > timeline.start + timeline.active.duration + 1e-8,
            "frames": stop - first,
            "recorded_response_delay_ms": delays[len(events) % len(delays)] * 1000,
        }
        started = time.perf_counter()
        proposed = None
        try:
            proposed, attempts = plan_recorded(
                q,
                **options,
                phases=phases,
                start_velocity=request.sample.velocity,
                start_acceleration=request.sample.acceleration,
                constrained_fallback=constrained_fallback,
            )
            event.update(status="planning_complete", attempts=attempts, bounds=proposed.bounds())
        except PlanningError as error:
            event.update(status="planning_rejected", error=str(error))
            if first_failure is None:
                first_failure = {
                    "points": q.tolist(),
                    "phases": phases.tolist(),
                    "start_velocity": request.sample.velocity.tolist(),
                    "start_acceleration": request.sample.acceleration.tolist(),
                    "error": str(error),
                }
        event["planning_ms"] = (time.perf_counter() - started) * 1000
        ready = now + (event["recorded_response_delay_ms"] + event["planning_ms"]) / 1000
        event["simulated_ready_at_s"] = ready
        events.append(event)
        waiting = ready, request, proposed, event
    return {
        "initial_planning_ms": initial_ms,
        "switches": timeline.switches,
        "dataset_end_reached": completed,
        "pending_at_stop": waiting is not None,
        "scheduled_at_stop": timeline.scheduled is not None,
        "first_failed_window": first_failure,
        "planning_ms": stats([e["planning_ms"] for e in events]),
        "sample_and_check_ms": stats([r["sample_and_check_ms"] for r in ticks]),
        "terminal_hold_ticks": sum(r["terminal_hold"] for r in ticks),
        "events": events,
        "ticks": ticks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limits-profile", type=Path, required=True)
    parser.add_argument("--latency-log", type=Path, required=True)
    parser.add_argument("--episodes", default="0,413,415,420")
    parser.add_argument("--fps", nargs="+", type=float, default=[30, 15])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--lead-s", type=float, default=0.6)
    parser.add_argument("--constrained-fallback", action="store_true")
    parser.add_argument("--adaptive-lead", action="store_true")
    parser.add_argument("--safety-source", type=Path)
    parser.add_argument("--geometry-template", type=Path)
    parser.add_argument("--anchor-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120 or not 0 < args.lead_s <= 2 or not all(f in (15, 30) for f in args.fps):
        parser.error("Offline bounds: seconds 1..120, lead 0..2, fps 15/30")
    geometry = [args.safety_source, args.geometry_template, args.anchor_log]
    if any(geometry) and not all(geometry):
        parser.error("Geometry requires source, template, and 24D anchor log together")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["fps"] != 30 or info["features"]["action"]["shape"] != [16]:
        raise ValueError("Requires the verified 30 Hz DH 16D dataset")
    _, vectors = load_profile(args.limits_profile, info["features"]["action"]["names"])
    lower, upper = np.full(16, -4.0), np.full(16, 4.0)
    lower[[7, 15]], upper[[7, 15]] = 0, 0.025
    limits = Limits(vectors["velocity"], vectors["acceleration"], vectors["jerk"], lower, upper)
    delays = read_delays(args.latency_log)
    args.output.mkdir(parents=True, exist_ok=False)
    warmup_ms = 0.0
    if args.constrained_fallback:
        # Solver import/native initialization belongs to preparation, not a
        # deadline-critical first moving-window request. Record its cost.
        before = time.perf_counter()
        from scipy.optimize import linprog

        warm = linprog([1.0], bounds=[(0, 1)], method="highs")
        if not warm.success:
            raise RuntimeError("Offline convex solver warmup failed")
        warmup_ms = (time.perf_counter() - before) * 1000
    report = {
        "mode": "offline_simulated_clock",
        "hardware_ready": False,
        "physical_commands": 0,
        "real_rtc_model_outputs": False,
        "constrained_fallback": args.constrained_fallback,
        "adaptive_lead": args.adaptive_lead,
        "terminal_rest_handoff": True,
        "solver_warmup_ms": warmup_ms,
        "collision_enforced": False,
        "dataset": str(args.dataset.resolve()),
        "candidate_source": "recorded dataset windows, 50 future frames",
        "lead_s": args.lead_s,
        "seconds_per_case": args.seconds,
        "limits_sha256": hashlib.sha256(args.limits_profile.read_bytes()).hexdigest(),
        "latency_log": str(args.latency_log),
        "latency_log_sha256": hashlib.sha256(args.latency_log.read_bytes()).hexdigest(),
        "latency_ms": stats([d * 1000 for d in delays]),
        "geometry_template": str(args.geometry_template) if all(geometry) else None,
        "geometry_template_sha256": hashlib.sha256(args.geometry_template.read_bytes()).hexdigest()
        if all(geometry)
        else None,
        "safety_source": str(args.safety_source) if all(geometry) else None,
        "anchor_log": str(args.anchor_log) if all(geometry) else None,
        "limitations": [
            "Virtual ticks are not a real-time deadline proof.",
            "Recorded delays are reused, not new network measurements.",
            "Geometry uses a template and frozen recorded non-arm joints, not current calibration/feedback.",
            "Shadow geometry results never certify or dispatch hardware commands.",
        ],
        "cases": [],
    }
    for episode in map(int, args.episodes.split(",")):
        points = read_episode(args.dataset, episode)
        for fps in args.fps:
            check, records = (lambda a, b: True), []
            if all(geometry):
                check, records = shadow_geometry(*geometry)
            row = {"episode": episode, "fps": fps}
            try:
                row.update(
                    run_case(
                        points,
                        fps=fps,
                        limits=limits,
                        delays=delays,
                        seconds=args.seconds,
                        lead_s=args.lead_s,
                        checker=check,
                        constrained_fallback=args.constrained_fallback,
                        adaptive_lead=args.adaptive_lead,
                    )
                )
            except PlanningError as error:
                row["startup_error"] = str(error)
            row["geometry"] = {
                "samples": len(records),
                "timing_ms": stats([r["ms"] for r in records]),
                "blocked_samples": sum(r["blocked"] for r in records),
            }
            if records:
                row["geometry"]["records"] = records
            report["cases"].append(row)
            assert not any("agibot_gdk" in name for name in sys.modules)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("ticks", "events", "geometry", "first_failed_window")}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
