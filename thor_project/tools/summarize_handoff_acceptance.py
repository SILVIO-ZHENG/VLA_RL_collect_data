"""Summarize completed read-only logs, excluding explicit pause/resume phases."""
import argparse
import collections
import datetime
import itertools
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def percentiles(values):
    return dict(zip(("p50", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    log = (args.directory / "client.log").read_text()
    rows = [json.loads(line) for line in (args.directory / "client_safety.jsonl").read_text().splitlines()]
    pause = re.search(r"(?m)^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*SOFT PAUSE ACTIVE", log)
    if pause is None:
        raise RuntimeError("No confirmed pause boundary; refusing to mix explicit holds with running statistics")
    boundary = datetime.datetime.strptime(pause[1], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1e9
    active = [r for r in rows if r["timestamp_ns"] < boundary]
    ticks = [r for r in active if r.get("record_type") == "servo_dispatch"]
    checks = [r for r in active if r.get("record_type") == "worker_check"]
    proofs = {(r["generation"], r["safety_request_id"]): r for r in rows if r.get("record_type") == "worker_check"}
    motions = [r for r in ticks if r["collision_response"] == "async_checked_motion"]
    violations = []
    for r in motions:
        proof = proofs.get((r["generation"], r["safety_request_id"]))
        if proof is None or not proof["check_within_budget"] or proof["hard_limit_triggered"]:
            violations.append(r["timestamp_ns"])
    longest_ticks = 0
    longest_ms = 0.0
    index = 0
    for held, group in itertools.groupby(ticks, key=lambda r: r["collision_response"] != "async_checked_motion"):
        group = list(group)
        if held:
            longest_ticks = max(longest_ticks, len(group))
            end = ticks[index + len(group)]["timestamp_ns"] if index + len(group) < len(ticks) else boundary
            longest_ms = max(longest_ms, (end - group[0]["timestamp_ns"]) / 1e6)
        index += len(group)
    result = dict(
        running_only_before_first_pause=True, ticks=len(ticks), checks=len(checks),
        duration_s=(ticks[-1]["timestamp_ns"] - ticks[0]["timestamp_ns"]) / 1e9,
        servo_hz=(len(ticks) - 1) / ((ticks[-1]["timestamp_ns"] - ticks[0]["timestamp_ns"]) / 1e9),
        check_hz=(len(checks) - 1) / (checks[-1]["compute_started"] - checks[0]["compute_started"]),
        response_counts=dict(collections.Counter(r["collision_response"] for r in ticks)),
        hold_reasons=dict(collections.Counter(r.get("hold_reason") or "unspecified" for r in ticks if r["collision_response"] != "async_checked_motion")),
        hold_fraction=1 - len(motions) / len(ticks), longest_hold_ticks=longest_ticks, longest_hold_ms=longest_ms,
        worker_ms=percentiles([r["safety_guard_latency_ms"] for r in checks]),
        certificate_age_ms=percentiles([r["certificate_age_ms"] for r in ticks]),
        worker_over_budget=sum(not r["check_within_budget"] for r in checks),
        motion_proof_violations=violations,
        all_gdk_acceptance_null=all(r.get("gdk_command_accepted") is None for r in rows),
        physical_prepare_seen="GDK PHYSICAL EXECUTION PREPARED" in log,
        critical_count=log.count("CRITICAL"), traceback_count=log.count("Traceback"),
        queue_underruns=log.count("ACTION QUEUE UNDERRUN"),
        transitions=json.loads((args.directory / "transitions.json").read_text()),
    )
    output = args.directory / "handoff-summary.json"
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
