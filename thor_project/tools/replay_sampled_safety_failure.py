"""Offline geometry replay of recorded feedback/commands; never imports GDK."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--stride', type=int, choices=(1, 2), default=2)
    parser.add_argument('--before-ns', type=int)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.client_dir))
    from robot_client.async_safety import check_sampled_segment_window
    from robot_client.safety_layer import load_safety_layer
    layer = load_safety_layer(args.profile, asset_root=args.client_dir / 'robot_client').realtime_guard(path_samples=2)
    records = [json.loads(line) for line in args.log.read_text().splitlines()]
    if args.before_ns is not None:
        records = [r for r in records if r['timestamp_ns'] < args.before_ns]
    checks = [r for r in records if r.get('record_type') == 'worker_check']
    commands = [r for r in records if r.get('record_type') == 'servo_dispatch']
    output = []
    for row in checks:
        stamp = row['check_started']
        measured = np.asarray(row['measured_feedback'], dtype=np.float32)
        before = [r for r in commands if r['event_monotonic'] < stamp]
        after = [r for r in commands if r['event_monotonic'] >= stamp][:13]
        anchor = np.asarray(before[-1]['adopted_action'], dtype=np.float32) if before else measured
        points = [np.asarray(r['adopted_action'], dtype=np.float32) for r in after]
        # Replay only recorded commands, never invent missing failure-time poses.
        committed, targets = points[:8], points[8:]
        tail = committed[-1] if committed else anchor
        if not targets:
            targets = [tail.copy()]
        task = dict(anchor=tail, fallback_anchor=anchor, committed=committed,
                    targets=targets, raw=np.asarray(row['raw_action'], dtype=np.float32))
        started = time.perf_counter()
        result = check_sampled_segment_window(layer, measured, task, stride=args.stride)
        output.append(dict(request=row['safety_request_id'],
                           recorded_future_points=len(points),
                           live_collision=result['live_decision'].hard_limit_triggered,
                           candidate_collision=result['decision'].hard_limit_triggered,
                           path_z_min=result['decision'].path_z_min,
                           stride=args.stride, evaluated_configurations=result['evaluated_configurations'],
                           local_check_ms=(time.perf_counter()-started)*1000))
    report = dict(physical_commands_sent=0, snapshots=len(output), stride=args.stride)
    if args.summary and output:
        report.update(check_ms_p50_p95_p99_max=np.percentile([r['local_check_ms'] for r in output], [50,95,99,100]).tolist(),
                      evaluations_total=sum(r['evaluated_configurations'] for r in output),
                      collisions=sum(r['live_collision'] or r['candidate_collision'] for r in output))
    else:
        report['results'] = output
    print(json.dumps(report, indent=2))
    if not output or any(r['live_collision'] or r['candidate_collision'] for r in output):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
