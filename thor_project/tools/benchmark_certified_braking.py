"""Pure-compute benchmark: no SDK import, connection, or robot command.

Run using a staged robot_client package and an existing calibrated profile.
The 50 Hz feedback callback returns a fixed recorded pose, NOT robot feedback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def statistics(values):
    return dict(p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)), max_ms=float(np.max(values)),
                over_20ms=sum(v > 20 for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--pose-log', type=Path, required=True, help='Only first JSONL row safe_action is used')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--rolling', action='store_true', help='Benchmark the 10-point rolling certificate, without GDK')
    parser.add_argument('--output', type=Path, help='New JSON result file; existing evidence is never overwritten')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 2000:
        parser.error('iterations must be in [1, 2000]')
    if args.output is not None and args.output.exists():
        parser.error('output already exists')
    sys.path.insert(0, str(args.client_dir.resolve()))
    from robot_client.async_safety import AsyncSafetyWorker
    from robot_client.constants import GDK_BODY_JOINTS, GDK_BODY_JOINT_LIMITS
    with args.pose_log.open() as stream:
        measured = np.asarray(json.loads(next(stream))['safe_action'], dtype=np.float32)
    if measured.shape != (24,) or not np.isfinite(measured).all():
        raise ValueError('Invalid recorded pose')
    lower, upper = np.empty(24), np.empty(24)
    indices = tuple(range(7)) + tuple(range(8, 15)) + tuple(range(16, 24))
    for index, name in zip(indices, GDK_BODY_JOINTS, strict=True):
        low, high = GDK_BODY_JOINT_LIMITS[name]
        lower[index], upper[index] = low + .01, high - .01
    lower[[7, 15]], upper[[7, 15]] = 0, np.float32(.025)
    maximum_velocity = np.full(24, .6)
    maximum_velocity[16:] = .3
    maximum_velocity[[7, 15]] = .05
    acceleration = np.full(24, 2.)
    acceleration[16:] = 1.
    acceleration[[7, 15]] = .2
    results = []
    for speed in (.1, .3, .6):
        for enabled in ((True,) if args.rolling else (False, True)):
            worker = AsyncSafetyWorker(args.profile, 2, lambda: measured.copy())
            durations, ages = [], []
            reserves = 0
            rejected = 0
            start = time.perf_counter()
            try:
                for number in range(args.iterations + 10):
                    velocity = np.zeros(24)
                    velocity[0] = speed
                    targets = [(measured + velocity * .01 * i).astype(np.float32)
                               for i in range(1, 11 if args.rolling else 3)]
                    task = dict(request_id=number, generation=0, anchor=measured.copy(),
                                fallback_anchor=measured.copy(), targets=targets, raw=targets[-1])
                    task['rolling'] = args.rolling
                    if enabled:
                        task['braking'] = dict(acceleration=acceleration, max_velocity=maximum_velocity,
                                               lower=lower, upper=upper, dt=.01,
                                               target_velocities=[velocity.copy() for _ in targets])
                    worker.submit(task)
                    deadline = time.perf_counter() + 15
                    result = None
                    while result is None:
                        if time.perf_counter() >= deadline:
                            raise RuntimeError('Synthetic worker timed out')
                        result = worker.poll()
                        if result is None:
                            time.sleep(.0005)
                    if result.get('error'):
                        raise RuntimeError(result['error'])
                    if number >= 10:
                        durations.append(result['check_ms'])
                        ages.append((time.perf_counter() - result['started']) * 1000)
                        reserves += sum(x is not None for x in result.get('target_brakes') or [])
                        rejected += bool(result['decision'].hard_limit_triggered)
            finally:
                worker.close()
            row = dict(braking=enabled, rolling=args.rolling, trajectory_steps=len(targets),
                       synthetic_arm_speed=speed, iterations=args.iterations,
                       check=statistics(durations), result_age=statistics(ages), valid_reserves=reserves,
                       rejected_motion_checks=rejected, elapsed_s=time.perf_counter() - start)
            results.append(row)
            print(json.dumps(row), flush=True)
    assert 'agibot_gdk' not in sys.modules
    summary = dict(synthetic_feedback=True, gdk_imported=False, physical_commands=0,
                   actual_servo_benchmark=False, results=results)
    if args.output is not None:
        with args.output.open('x') as output:
            json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
