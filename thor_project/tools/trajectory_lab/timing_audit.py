"""Estimate stage-relative delays from completed control logs; no robot access.

These are signal-fit delays, not causal hardware latency measurements. A trace
without complete response chunks cannot establish an RTC prefix replay.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = [*range(7), *range(8, 15)]


def stats(values):
    if not len(values):
        return None
    return dict(zip(("p50", "p95", "min", "max"), np.percentile(values, (50, 95, 0, 100)).tolist()))


def fit_delay(source_t, source, target_t, target, *, max_delay=0.6):
    """Fit t_target to source(t_target - delay), excluding moving-window edges.

    Return None for unidentifiable stationary data, never call it zero delay.
    Negative values mean signal lead. Residuals/boundary flags remain visible.
    """
    source_t, target_t = np.asarray(source_t), np.asarray(target_t)
    source, target = np.asarray(source), np.asarray(target)
    if len(source_t) < 3 or len(target_t) < 3:
        return None
    if (
        source.ndim != 2
        or target.shape != (len(target_t), source.shape[1])
        or len(source) != len(source_t)
        or np.any(np.diff(source_t) <= 0)
        or np.any(np.diff(target_t) <= 0)
        or not all(np.isfinite(x).all() for x in (source_t, target_t, source, target))
    ):
        raise ValueError("Expected finite, ordered, dimension-matched signals")
    keep = (target_t >= source_t[0] + max_delay) & (target_t <= source_t[-1] - 0.1)
    if keep.sum() < 20:
        return None
    t, y = target_t[keep], target[keep]
    moving = (np.ptp(source, axis=0) >= 0.025) & (np.ptp(y, axis=0) >= 0.01)
    if not moving.any():
        return None
    source, y = source[:, moving], y[:, moving]
    lags = np.arange(-10, int(round(max_delay * 100)) + 1) / 100
    errors = []
    for lag in lags:
        fitted = np.stack([np.interp(t - lag, source_t, source[:, j]) for j in range(source.shape[1])], axis=1)
        errors.append(float(np.mean((fitted - y) ** 2)))
    best = int(np.argmin(errors))
    baseline = errors[10]
    return dict(
        delay_ms=float(lags[best] * 1000),
        rmse_rad=float(np.sqrt(errors[best])),
        zero_lag_rmse_rad=float(np.sqrt(baseline)),
        error_reduction=0.0 if baseline < 1e-20 else float(1 - errors[best] / baseline),
        search_boundary=best in (0, len(lags) - 1),
        moving_joint_count=int(moving.sum()),
    )


def analyze(path):
    header, end, samples, replies = {}, {}, [], 0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("record_type")
            if kind == "trace_start":
                header = row
            elif kind == "trace_end":
                end = row
            elif kind == "inference_response":
                replies += 1
            elif kind == "control" and row.get("command_accepted") and row.get("sdk_call_started") is not None:
                samples.append(row)
    # A repeated SDK timestamp is the same dispatch, not another 100 Hz action.
    samples = sorted({r["sdk_call_started"]: r for r in samples}.values(), key=lambda r: r["sdk_call_started"])
    if not samples:
        raise ValueError("No accepted SDK dispatch samples")
    t = np.array([r["sdk_call_started"] for r in samples])
    signals = {
        key: np.array([r[key] for r in samples], dtype=float)[:, ARMS]
        for key in ("policy_control_target", "filtered_target", "adopted_target", "measured_state")
    }
    stable = np.array(
        [
            not r.get("explicit_hold")
            and not r.get("filter_reanchored")
            and r.get("collision_response") == "async_checked_motion"
            for r in samples
        ]
    )
    stages = {
        "body_lowpass": ("policy_control_target", "filtered_target"),
        "queue_and_kinematic_tracking": ("filtered_target", "adopted_target"),
        "feedback_response_proxy": ("adopted_target", "measured_state"),
    }
    result = {}
    for stage, (source, target) in stages.items():
        windows = []
        for start in np.arange(t[0], t[-1] - 2, 1):
            selected = (t >= start) & (t < start + 2)
            indices = np.flatnonzero(selected)
            # Exclude intervention, but retain normal rolling chunk changes.
            # Their effects are part of this signal fit, not isolated hardware.
            if len(indices) < 150 or not stable[selected].all():
                continue
            st, sy, tt, ty = t[selected], signals[source][selected], t[selected], signals[target][selected]
            if stage == "feedback_response_proxy":
                raw = [samples[i].get("feedback_monotonic") for i in indices]
                if any(v is None or not np.isfinite(v) for v in raw):
                    continue
                # Assign feedback its recorded read time, not every later send
                # that reused the cached value. It is still not a motor timestamp.
                tt, first = np.unique(np.asarray(raw), return_index=True)
                ty = ty[first]
            fitted = fit_delay(st, sy, tt, ty)
            if fitted is not None:
                windows.append(
                    dict(
                        start_s=float(start - t[0]),
                        candidate_chunks=len({samples[i].get("chunk_id") for i in indices}),
                        **fitted,
                    )
                )
        result[stage] = dict(
            windows=windows,
            usable_windows=len(windows),
            fitted_delay_ms=stats([w["delay_ms"] for w in windows]),
            rmse_rad=stats([w["rmse_rad"] for w in windows]),
            boundary_fits=sum(w["search_boundary"] for w in windows),
        )
    return dict(
        mode="offline_signal_fit",
        hardware_ready=False,
        source=str(Path(path).resolve()),
        trace_complete=bool(end) and not end.get("dropped") and not end.get("truncated"),
        policy_fps=header.get("policy_fps"),
        inference_mode=header.get("inference_mode"),
        numerical_model_reply_count=replies,
        dispatch_provenance_count=sum(r.get("dispatch_provenance") is not None for r in samples),
        sdk_intervals_ms=stats(np.diff(t) * 1000),
        stages=result,
        limitations=[
            "Signal fits are not causal isolation of a filter, limiter, queue or hardware.",
            "Feedback read timestamps are not device-internal timestamps or grasp confirmation.",
            "No stationary/ambiguous window is reported as zero delay.",
            "Old candidate chunk tags cannot identify the adopted queue's exact origin.",
            "No complete model replies means no faithful RTC-prefix counterfactual replay.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = [analyze(path) for path in args.traces]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for case in report:
        print(
            json.dumps(
                {
                    **{k: v for k, v in case.items() if k not in ("stages", "limitations")},
                    "stages": {k: {x: y for x, y in v.items() if x != "windows"} for k, v in case["stages"].items()},
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
