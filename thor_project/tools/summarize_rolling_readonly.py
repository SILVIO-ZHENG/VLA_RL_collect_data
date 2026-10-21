"""Summarize observed read-only evidence without counting pause as active motion.

An unconsumed check is not proof that the controller discarded it. New controller
``safety_event`` records provide that distinction; old logs remain inconclusive.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np


MOTION = "async_checked_motion"
LOG_TIME = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")


def _log_timestamp(line):
    match = LOG_TIME.match(line)
    if match is None:
        return None
    stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f")
    aware = stamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return int(aware.replace(microsecond=0).timestamp()) * 1_000_000_000 + aware.microsecond * 1000


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distribution(values):
    values = [value for value in values if _finite(value)]
    if not values:
        return dict(count=0, min=None, p05=None, median=None, p95=None, max=None, mean=None)
    return dict(count=len(values), min=min(values), p05=float(np.percentile(values, 5)),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=max(values), mean=float(np.mean(values)))


def _rate(timestamps, units_per_second):
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        return None
    return (len(timestamps) - 1) * units_per_second / (timestamps[-1] - timestamps[0])


def _request_key(row):
    return row.get("generation"), row.get("safety_request_id", row.get("request_id"))


def _reference(row):
    generation, request_id = _request_key(row)
    return dict(generation=generation, safety_request_id=request_id)


def _worker_timing(checks):
    # check_started is the monotonic dispatch/start clock, not the audit write
    # timestamp or the configured safety_check_fps.
    starts = sorted(row["check_started"] for row in checks if _finite(row.get("check_started")))
    compute_starts = sorted(row["compute_started"] for row in checks if _finite(row.get("compute_started")))
    periods = [(end - start) * 1000 for start, end in zip(starts, starts[1:])]
    total_latencies = [(row["check_completed"] - row["submitted_at"]) * 1000
                       for row in checks if _finite(row.get("check_completed"))
                       and _finite(row.get("submitted_at"))]
    queue_latencies = [(row["compute_started"] - row["submitted_at"]) * 1000
                       for row in checks if _finite(row.get("compute_started"))
                       and _finite(row.get("submitted_at"))]
    return dict(start_count=len(starts), actual_start_hz=_rate(starts, 1),
                start_interval_ms=_distribution(periods),
                compute_start_hz=_rate(compute_starts, 1),
                submit_to_complete_ms=_distribution(total_latencies),
                submit_to_compute_ms=_distribution(queue_latencies),
                checked_points=_distribution(row.get("checked_points") for row in checks),
                duplicate_start_count=sum(period == 0 for period in periods),
                geometry_latency_ms=_distribution(row.get("safety_guard_latency_ms") for row in checks),
                over_budget_count=sum(row.get("check_within_budget") is False for row in checks))


def _queue_distribution(dispatch):
    values = [row["certified_buffer_steps"] for row in dispatch
              if _finite(row.get("certified_buffer_steps"))]
    return dict(**_distribution(values), counts=dict(sorted(Counter(values).items())),
                zero_count=sum(value == 0 for value in values),
                under_five_count=sum(value < 5 for value in values),
                semantics="recorded certified_buffer_steps; no unrecorded queue state inferred")


def _hold_runs(dispatch, boundary, events):
    runs = []
    index = 0
    motion_seen = False
    while index < len(dispatch):
        if dispatch[index].get("collision_response") == MOTION:
            motion_seen = True
            index += 1
            continue
        start = index
        while index < len(dispatch) and dispatch[index].get("collision_response") != MOTION:
            index += 1
        held = dispatch[start:index]
        first, last = held[0]["timestamp_ns"], held[-1]["timestamp_ns"]
        resumed = dispatch[index] if index < len(dispatch) else None
        end = resumed["timestamp_ns"] if resumed else boundary
        # At EOF we only know the observed span. Do not invent a final 10 ms
        # interval or call a failed tail a completed hold-and-resume cycle.
        until = end if end is not None else last
        references = {_request_key(row) for row in held}
        preceding = dispatch[start - 1] if start else None
        context_start = preceding["timestamp_ns"] if preceding else first
        related_events = [event for event in events
                          if context_start <= event["timestamp_ns"] <= until]
        runs.append(dict(
            phase="steady" if motion_seen else "startup", count=len(held),
            start_timestamp_ns=first, last_hold_timestamp_ns=last,
            start_offset_s=(first - dispatch[0]["timestamp_ns"]) / 1e9,
            preceding_motion_request=_reference(preceding) if preceding else None,
            events_context_start_timestamp_ns=context_start,
            end_timestamp_ns=end, resumed=resumed is not None,
            right_censored=resumed is None,
            duration_ms=(end - first) / 1e6 if end is not None else None,
            observed_hold_span_ms=(last - first) / 1e6,
            responses=dict(Counter(row.get("collision_response", "unknown") for row in held)),
            hold_reasons=dict(Counter(row.get("hold_reason") or "unrecorded" for row in held)),
            hold_certificate_requests=[dict(generation=gen, safety_request_id=req)
                                       for gen, req in sorted(references, key=str)],
            resumed_request=_reference(resumed) if resumed else None,
            queue=_queue_distribution(held), events=related_events,
        ))
    return runs


def _request_outcomes(checks, dispatch, events):
    requests = {}

    def entry(row):
        key = _request_key(row)
        if key not in requests:
            requests[key] = dict(**_reference(row), checks=[], motion_dispatches=0,
                                 hold_dispatches=0, events=[])
        return requests[key]

    for row in checks:
        entry(row)["checks"].append(dict(timestamp_ns=row["timestamp_ns"],
            hard_limit_triggered=row.get("hard_limit_triggered"),
            check_within_budget=row.get("check_within_budget"),
            safety_guard_latency_ms=row.get("safety_guard_latency_ms"),
            checked_points=row.get("checked_points"),
            submitted_at=row.get("submitted_at"), check_started=row.get("check_started"),
            compute_started=row.get("compute_started"), check_completed=row.get("check_completed")))
    for row in dispatch:
        name = "motion_dispatches" if row.get("collision_response") == MOTION else "hold_dispatches"
        entry(row)[name] += 1
    for row in events:
        if _request_key(row)[1] is not None:
            entry(row)["events"].append(row)
    for request in requests.values():
        names = {event.get("event") for event in request["events"]}
        request["admission_observed"] = "result_admitted" in names
        request["admitted_points"] = sum(event["checked_points"] for event in request["events"]
                                         if event.get("event") == "result_admitted"
                                         and _finite(event.get("checked_points")))
        request["discard_observed"] = "result_discarded" in names
        request["motion_consumed"] = request["motion_dispatches"] > 0
        if request["motion_consumed"]:
            outcome = "motion_dispatched"
        elif request["discard_observed"]:
            outcome = "explicit_discard"
        elif request["admission_observed"]:
            outcome = "admitted_without_observed_motion"
        else:
            outcome = "no_motion_observed_reason_unknown"
        request["outcome"] = outcome
    return list(requests.values())


def summarize(directory):
    directory = Path(directory)
    log = (directory / "client.log").read_text()
    lines = [(line, _log_timestamp(line)) for line in log.splitlines()]
    pauses = [stamp for line, stamp in lines if stamp is not None and "SOFT PAUSE ACTIVE" in line]
    pause_confirmed_at = pauses[0] if pauses else None
    requests = [stamp for line, stamp in lines if stamp is not None
                and "Configured active duration" in line
                and "requesting automatic soft pause" in line
                and (pause_confirmed_at is None or stamp <= pause_confirmed_at)]
    pause_requested_at = requests[0] if requests else None
    # A commanded pause is not a collision-buffer underrun. Its request ends
    # active time; its later acknowledgement remains required for completion.
    boundary = pause_requested_at if pause_requested_at is not None else pause_confirmed_at
    rows, malformed = [], []
    audit_path = directory / "client_safety.jsonl"
    if audit_path.exists():
        for number, line in enumerate(audit_path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not _finite(row.get("timestamp_ns")):
                    raise ValueError("missing finite timestamp_ns")
                rows.append(row)
            except (ValueError, TypeError) as error:
                malformed.append(dict(line=number, error=str(error)))
    rows.sort(key=lambda row: row["timestamp_ns"])
    transition = [row for row in rows if pause_requested_at is not None
                  and row["timestamp_ns"] >= pause_requested_at
                  and (pause_confirmed_at is None or row["timestamp_ns"] < pause_confirmed_at)]
    transition_dispatch = [row for row in transition if row.get("record_type") == "servo_dispatch"]
    transition_events = [row for row in transition if row.get("record_type") == "safety_event"]
    active = [row for row in rows if boundary is None or row["timestamp_ns"] < boundary]
    dispatch = [row for row in active if row.get("record_type") == "servo_dispatch"]
    checks = [row for row in active if row.get("record_type") == "worker_check"]
    events = [row for row in active if row.get("record_type") == "safety_event"]
    active_lines = [line for line, stamp in lines if stamp is not None and (boundary is None or stamp < boundary)]
    counts = Counter(row.get("collision_response", "unknown") for row in dispatch)
    command_path = directory / "command.json"
    command = json.loads(command_path.read_text()) if command_path.exists() else []
    legacy = ("--rolling-geometry=legacy" in command or
              any(command[i:i + 2] == ["--rolling-geometry", "legacy"] for i in range(len(command))))
    expected_renewal = "sampled_path_feedback_bound" if legacy else "global_trajectory_box"
    all_proofs = {_request_key(row): row for row in checks}
    bad = []
    for row in dispatch:
        if row.get("collision_response") != MOTION:
            continue
        proof = all_proofs.get(_request_key(row))
        age = row.get("certificate_age_ms")
        snapshot = row.get("proof_renewal") == "sampled_path_snapshot"
        feedback_age = row.get("dispatch_feedback_age_ms")
        stride2 = row.get("collision_action_stride") == 2
        sampled_indices = [] if proof is None else proof.get("sampled_target_indices", [])
        stride_audit_bad = stride2 and (
            proof is None or proof.get("collision_action_stride") != 2
            or not isinstance(sampled_indices, list)
            or row.get("action_geometry_sampled") is not (row.get("trajectory_step") in sampled_indices))
        if (proof is None or proof.get("hard_limit_triggered") is not False
                or stride_audit_bad
                or proof["timestamp_ns"] > row["timestamp_ns"]
                or not _finite(age) or not 0 <= age < (200 if snapshot else 40)
                or (snapshot and (not legacy or not _finite(feedback_age) or not 0 <= feedback_age < 40))
                or (not snapshot and row.get("proof_renewal") != expected_renewal)):
            bad.append(row["timestamp_ns"])
    first_motion = next((row["timestamp_ns"] for row in dispatch
                         if row.get("collision_response") == MOTION), None)
    steady = [row for row in dispatch if first_motion is not None and row["timestamp_ns"] >= first_motion]
    steady_checks = [row for row in checks if first_motion is not None and row["timestamp_ns"] >= first_motion]
    hold_runs = _hold_runs(dispatch, boundary, events)
    outcomes = _request_outcomes(checks, dispatch, events)
    jpgs = list((directory / "inference-frames").glob("*.jpg"))
    durations = _distribution(row.get("safety_guard_latency_ms") for row in checks)
    seconds = ((dispatch[-1]["timestamp_ns"] - dispatch[0]["timestamp_ns"]) / 1e9
               if len(dispatch) >= 2 else None)
    run_path = directory / "result.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    failures = [line for line in log.splitlines()
                if " ERROR " in line or line.startswith(("RuntimeError:", "SafetyDispatchExpiredError:"))]
    completed = bool(pause_confirmed_at is not None and dispatch and not malformed and not failures
                     and not log.count("Traceback") and run.get("ssh_returncode") in (0, 124))
    return dict(
        completed=completed,
        active_end_reason=("automatic_pause_requested" if pause_requested_at is not None else
                           "automatic_pause" if pause_confirmed_at is not None else "log_ended_without_pause"),
        automatic_pause_confirmed=pause_confirmed_at is not None, pause_timestamp_ns=pause_confirmed_at,
        active_end_timestamp_ns=boundary, pause_requested_timestamp_ns=pause_requested_at,
        pause_transition=dict(
            request_observed=pause_requested_at is not None,
            confirmed=pause_confirmed_at is not None,
            duration_ms=((pause_confirmed_at - pause_requested_at) / 1e6
                         if pause_requested_at is not None and pause_confirmed_at is not None else None),
            dispatch_count=len(transition_dispatch),
            hold_tick_count=sum(row.get("collision_response") != MOTION for row in transition_dispatch),
            responses=dict(Counter(row.get("collision_response", "unknown") for row in transition_dispatch)),
            dispatches=transition_dispatch, safety_events=transition_events),
        active_seconds=seconds, dispatch_count=len(dispatch), responses=dict(counts),
        dispatch_hz=_rate([row["timestamp_ns"] for row in dispatch], 1e9),
        check_count=len(checks), check_p95_ms=durations["p95"], check_max_ms=durations["max"],
        motion_fraction=counts[MOTION] / len(dispatch) if dispatch else None,
        steady_dispatch_count=len(steady),
        steady_motion_fraction=(sum(row.get("collision_response") == MOTION for row in steady) / len(steady)
                                if steady else None),
        first_motion_timestamp_ns=first_motion,
        startup_hold_run_count=sum(run["phase"] == "startup" for run in hold_runs),
        steady_hold_run_count=sum(run["phase"] == "steady" for run in hold_runs),
        steady_hold_tick_count=sum(run["count"] for run in hold_runs if run["phase"] == "steady"),
        hold_runs=hold_runs, worker_timing=_worker_timing(checks),
        steady_worker_timing=_worker_timing(steady_checks),
        queue=_queue_distribution(dispatch), steady_queue=_queue_distribution(steady),
        safety_event_counts=dict(Counter(row.get("event", "unknown") for row in events)),
        safety_events=events, request_outcomes=outcomes,
        request_outcome_counts=dict(Counter(row["outcome"] for row in outcomes)),
        event_diagnostics_available=bool(events),
        inference_responses=sum("End-to-end timing phase=inference" in line for line in active_lines),
        images=len(jpgs), all_jpeg_headers_valid=all(path.read_bytes().startswith(b"\xff\xd8") for path in jpgs),
        no_physical_acceptance=(all(row.get("gdk_command_accepted") is None for row in rows)
                                if rows and not malformed else None),
        physical_prepare_seen="GDK PHYSICAL EXECUTION PREPARED" in log,
        tracebacks=log.count("Traceback"), failure_lines=failures,
        queue_underruns=sum("ACTION QUEUE UNDERRUN" in line for line in active_lines),
        motion_audit_failures=bad, proof_renewal=expected_renewal,
        malformed_audit_rows=malformed, audit_file_present=audit_path.exists(),
        ssh_returncode=run.get("ssh_returncode"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="New output path; existing reports are never overwritten")
    args = parser.parse_args()
    result = summarize(args.directory)
    output = args.output or args.directory / "rolling-readonly-summary.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
