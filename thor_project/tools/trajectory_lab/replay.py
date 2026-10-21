"""Replay recorded waypoints locally, without a robot or policy server.

Example: python -m tools.trajectory_lab.replay --dataset ... --output ...
Output is diagnostic data, NOT a robot-deployable trajectory or certificate.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from .dataset_limits import load_profile
from .trajectory import Limits, PlanningError, path_error_bound, plan

ARMS = [*range(7), *range(8, 15)]


def percentiles(values):
    return dict(zip(("p50", "p95", "p99", "max"), np.percentile(values, (50, 95, 99, 100)).tolist()))


def plan_recorded(
    points,
    *,
    tangent_method="auto",
    knot_tolerance=0.003,
    max_path_refinements=6,
    constrained_fallback=False,
    **kwargs,
):
    """Bounded offline search; never relax path or derivative constraints.

    Record unsuccessful attempts instead of concealing spline overshoot. Dense
    knots recover fidelity when simplification fails. Harmonic remains a
    conservative alternative, including for difficult moving boundaries.
    """
    choices = [(tangent_method, knot_tolerance, 0.0)]
    if tangent_method == "auto":
        choices = [("cubic", knot_tolerance, 0.0), ("cubic", 0.0, 0.0), ("harmonic", knot_tolerance, 0.0)]
        if any(
            kwargs.get(name) is not None and np.any(np.asarray(kwargs[name]) != 0)
            for name in ("start_velocity", "start_acceleration")
        ):
            choices = [("cubic", knot_tolerance, 3.0), ("cubic", knot_tolerance, 6.0)] + choices
    attempts = []
    for method, tolerance, blend in dict.fromkeys(choices):
        row = {"method": method, "knot_tolerance_rad": tolerance, "start_blend_phase": blend}
        attempts.append(row)
        try:
            trajectory = plan(
                points,
                tangent_method=method,
                knot_tolerance=tolerance,
                max_path_refinements=max_path_refinements,
                start_blend_phase=blend,
                **kwargs,
            )
        except PlanningError as error:
            row.update(status="rejected", error=str(error))
            if constrained_fallback and (method, tolerance, blend) == choices[-1] and len(points) <= 65:
                # Bounded short-window fallback, never a production dependency.
                from .constrained import fit_window

                fallback_started = time.perf_counter()
                phase = np.asarray(
                    np.arange(len(points)) if kwargs.get("phases") is None else kwargs["phases"], dtype=float
                )
                nominal = np.diff(phase) / kwargs["policy_fps"]
                seed_options = {**kwargs, "start_velocity": None, "start_acceleration": None}
                try:
                    seed, _ = plan_recorded(
                        points,
                        knot_tolerance=knot_tolerance,
                        max_path_refinements=max_path_refinements,
                        **seed_options,
                    )
                    seed_h = np.diff(np.interp(phase, seed.phases, seed.times))
                    for ramp in (len(points) - 1, 24):
                        remaining = 0.35 - (time.perf_counter() - fallback_started)
                        if remaining < 0.05:
                            attempts.append(
                                {
                                    "method": "convex_jerk",
                                    "status": "rejected",
                                    "error": "Shared convex fallback budget exhausted",
                                }
                            )
                            break
                        item = {"method": "convex_jerk", "timing_ramp_frames": ramp}
                        attempts.append(item)
                        h = nominal + np.maximum(0, seed_h - nominal) * np.minimum(1, np.arange(len(nominal)) / ramp)
                        try:
                            trajectory = fit_window(
                                points,
                                phases=phase,
                                policy_fps=kwargs["policy_fps"],
                                limits=kwargs["limits"],
                                device_indices=kwargs.get("device_indices", ()),
                                start_velocity=np.zeros(np.shape(points)[1])
                                if kwargs.get("start_velocity") is None
                                else kwargs["start_velocity"],
                                start_acceleration=np.zeros(np.shape(points)[1])
                                if kwargs.get("start_acceleration") is None
                                else kwargs["start_acceleration"],
                                interval_durations=h,
                                max_path_deviation=kwargs.get("max_path_deviation", 0.02),
                                max_stretch=kwargs.get("max_stretch", 30.0),
                                servo_period=kwargs.get("servo_period", 0.01),
                                budget_s=remaining,
                            )
                        except PlanningError as error:
                            item.update(status="rejected", error=str(error))
                        else:
                            item["status"] = "bounded"
                            return trajectory, attempts
                except PlanningError as error:
                    attempts.append({"method": "convex_timing_seed", "status": "rejected", "error": str(error)})
        else:
            row["status"] = "bounded"
            return trajectory, attempts
    raise PlanningError(f"All offline candidates rejected: {json.dumps(attempts)}")


def legacy_replay(points, fps, legacy):
    dt = 0.01
    times = np.arange(0.0, (len(points) - 1) / fps, dt)
    phases = times * fps
    j = np.minimum(np.floor(phases + 1e-9).astype(int), len(points) - 2)
    u = (phases - j)[:, None]
    raw = points[j] + (points[j + 1] - points[j]) * u
    raw[:, [7, 15]] = points[j + 1][:, [7, 15]]  # existing right-interval device goals
    filt = legacy.CausalLowPassFilter(cutoff_hz=2, sample_hz=100, action_dim=16)
    filt.reset(points[0])
    filtered = np.array([filt.update(q) for q in raw])
    filtered[:, [7, 15]] = raw[:, [7, 15]]
    limiter = legacy.JointKinematicLimiter(
        sample_hz=100,
        max_velocity=np.full(16, 0.6),
        max_acceleration=np.full(16, 2.0),
        position_target_indices=(7, 15),
    )
    limiter.reset(points[0])
    sent = np.array(
        [limiter.update_tracking(filtered[k], (filtered[k + 1] - filtered[k]) / dt) for k in range(len(filtered) - 1)]
    )
    return {
        "body_tracking_error_rad": percentiles(np.max(abs(sent[:, ARMS] - raw[:-1, ARMS]), axis=1)),
        "sampled_body_jerk_rad_s3": percentiles(np.max(abs(np.diff(sent[:, ARMS], n=3, axis=0)) / dt**3, axis=1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", default="0,50,150,250,350,400")
    parser.add_argument("--fps", nargs="+", type=float, default=[30.0, 15.0])
    parser.add_argument("--jerk", type=float, default=10.0, help="OFFLINE experimental value only")
    parser.add_argument(
        "--limits-profile", type=Path, help="Per-joint offline dataset-derived limits, overrides uniform derivatives"
    )
    parser.add_argument("--knot-tolerance", type=float, default=0.003, help="Offline body knot tolerance in radians")
    parser.add_argument("--smoothing-window", type=int, default=9, help="Symmetric body prefit, no device smoothing")
    parser.add_argument(
        "--retiming-damping", type=float, default=0.5, help="Damp interval growth, without relaxing bound checks"
    )
    parser.add_argument("--legacy-client", type=Path, help="Optional path to existing action_filter.py")
    parser.add_argument("--tangent-method", choices=("auto", "cubic", "harmonic"), default="auto")
    parser.add_argument("--save-samples", action="store_true", help="Save large NPZ diagnostic traces")
    parser.add_argument(
        "--path-refinements",
        type=int,
        default=6,
        help="Bounded local path refinement rounds; 0 reproduces no-refinement behavior",
    )
    args = parser.parse_args()
    import pyarrow.parquet as pq  # optional offline dependency, not added to production

    requested = {int(value) for value in args.episodes.split(",")}
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["fps"] != 30 or info["features"]["action"]["shape"] != [16]:
        raise ValueError("This replay requires the verified 30 Hz 16D DH dataset")
    expected = ("left_effector_position", "right_effector_position")
    if tuple(info["features"]["action"]["names"][i] for i in (7, 15)) != expected:
        raise ValueError("Unexpected effector channel mapping")
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = {}
    for path in sorted(args.dataset.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "action", "observation.state"])
        ids = table["episode_index"].to_numpy()
        a = table["action"].combine_chunks().values.to_numpy().reshape(-1, 16)
        state = table["observation.state"].combine_chunks().values.to_numpy().reshape(-1, 16)
        for episode in requested.intersection(np.unique(ids).tolist()):
            selected = ids == episode
            episodes.setdefault(episode, []).append((a[selected].copy(), state[selected].copy()))
    if requested != set(episodes):
        raise ValueError(f"Missing episodes: {requested - set(episodes)}")
    legacy = None
    if args.legacy_client:
        spec = importlib.util.spec_from_file_location("legacy_action_filter", args.legacy_client)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
    lower, upper = np.full(16, -4.0), np.full(16, 4.0)
    lower[[7, 15]], upper[[7, 15]] = 0.0, 0.025
    limits = Limits(np.full(16, 0.6), np.full(16, 2.0), np.full(16, args.jerk), lower, upper)
    profile = None
    if args.limits_profile:
        profile, vectors = load_profile(args.limits_profile, info["features"]["action"]["names"])
        limits = Limits(vectors["velocity"], vectors["acceleration"], vectors["jerk"], lower, upper)
    report = {
        "mode": "offline_only",
        "hardware_ready": False,
        "physical_commands": 0,
        "experimental_limits": {name: getattr(limits, name).tolist() for name in ("velocity", "acceleration", "jerk")},
        "limits_profile": profile,
        "limits_profile_sha256": hashlib.sha256(args.limits_profile.read_bytes()).hexdigest()
        if args.limits_profile
        else None,
        "smoothing_window": args.smoothing_window,
        "knot_tolerance_rad": args.knot_tolerance,
        "retiming_damping": args.retiming_damping,
        "tangent_method": args.tangent_method,
        "max_path_refinements": args.path_refinements,
        "position_bounds": "lab bounds, NOT vendor limits",
        "collision_checked": False,
        "limitations": [
            "No SDK or physics, no RTC inference, no live feedback.",
            "No geometry certificate: these outputs must not be executed on hardware.",
            "Legacy replay isolates filtering/limiting, omits SDK and safety queue.",
        ],
        "cases": [],
    }
    for episode, pieces in sorted(episodes.items()):
        action = np.concatenate([a for a, _ in pieces]).astype(np.float64)
        state = np.concatenate([s for _, s in pieces]).astype(np.float64)
        # float32 represents 0.025 as 0.0250000003725. Normalize only this
        # explicitly bounded encoding error in the diagnostic copy, not the data.
        for array in (action, state):
            if np.any(array[:, [7, 15]] < -1e-8) or np.any(array[:, [7, 15]] > 0.025 + 1e-8):
                raise ValueError("Recorded gripper exceeds endpoint encoding tolerance")
            array[:, [7, 15]] = np.clip(array[:, [7, 15]], 0.0, 0.025)
        action[:, [7, 15]] = np.clip(0.025 - action[:, [7, 15]], 0, 0.025)
        points = np.vstack((state[0], action))
        for fps in args.fps:
            row = {"episode": episode, "policy_fps": fps, "frames": len(action)}
            started = time.perf_counter()
            try:
                trajectory, attempts = plan_recorded(
                    points,
                    policy_fps=fps,
                    limits=limits,
                    device_indices=(7, 15),
                    knot_tolerance=args.knot_tolerance,
                    smoothing_window=args.smoothing_window,
                    retiming_damping=args.retiming_damping,
                    tangent_method=args.tangent_method,
                    max_path_refinements=args.path_refinements,
                )
                row["planning_ms"] = (time.perf_counter() - started) * 1000
                row["planning_attempts"] = attempts
                row["path_refinements"] = trajectory.path_refinements
                row["iterations"] = trajectory.iterations
                row["body_knots"] = len(trajectory.points)
                row["nominal_duration_s"] = (len(points) - 1) / fps
                row["duration_s"] = trajectory.duration
                row["stretch"] = trajectory.duration / row["nominal_duration_s"]
                row["interval_stretch"] = percentiles(trajectory.durations * fps / np.diff(trajectory.phases))
                row["derivative_bound_ratios"] = trajectory.bounds()
                row["continuous_body_path_deviation_bound_rad"] = path_error_bound(trajectory)
                event_times = np.interp(trajectory.original_phases, trajectory.phases, trajectory.times)
                row["source_interval_duration_s"] = percentiles(np.diff(event_times))
                row["source_interval_duration_s"]["min"] = float(np.min(np.diff(event_times)))
                times = np.arange(0.0, trajectory.duration, 0.01)
                before = time.perf_counter()
                samples = [trajectory.sample(t) for t in times]
                row["sample_mean_us"] = (time.perf_counter() - before) * 1e6 / len(times)
                q = np.array([s.position for s in samples])
                phase = np.array([s.phase for s in samples])
                floor = np.minimum(np.floor(phase).astype(int), len(points) - 2)
                straight = points[floor] + (points[floor + 1] - points[floor]) * (phase - floor)[:, None]
                row["body_path_deviation_at_shared_phase_rad"] = percentiles(
                    np.max(abs(q[:, ARMS] - straight[:, ARMS]), axis=1)
                )
                expected_grip = points[floor][:, [7, 15]]
                row["gripper_event_value_error_m"] = float(np.max(abs(q[:, [7, 15]] - expected_grip)))
                row["sampled_body_jerk_rad_s3"] = percentiles(
                    np.max(abs(np.diff(q[:, ARMS], n=3, axis=0)) / 0.01**3, axis=1)
                )
                if args.save_samples:
                    np.savez_compressed(
                        args.output / f"episode-{episode:03d}-{fps:g}hz.npz",
                        time_s=times,
                        source_phase=phase,
                        target=q,
                        velocity=np.array([s.velocity for s in samples]),
                        acceleration=np.array([s.acceleration for s in samples]),
                        jerk=np.array([s.jerk for s in samples]),
                        knot_times=trajectory.times,
                        knot_source_phase=trajectory.phases,
                    )
                row["status"] = "planned_not_collision_checked"
            except PlanningError as error:
                row.update(status="rejected", error=str(error), planning_ms=(time.perf_counter() - started) * 1000)
            if legacy is not None:
                row["legacy"] = legacy_replay(points, fps, legacy)
            report["cases"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        fields = ["episode", "policy_fps", "status", "stretch", "duration_s", "planning_ms", "sample_mean_us"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])


if __name__ == "__main__":
    main()
