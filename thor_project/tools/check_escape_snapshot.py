"""Pure computation from a saved (possibly approximate) pose; never uses GDK."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-dir', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.client_dir))
    from robot_client.action_filter import JointKinematicLimiter
    from robot_client.async_safety import check_sampled_segment_window
    from robot_client.safety_layer import load_safety_layer
    layer = load_safety_layer(a.profile, asset_root=a.client_dir/'robot_client').realtime_guard(path_samples=1)
    data = json.loads(a.snapshot.read_text())
    anchor = np.asarray(data['anchor'], dtype=np.float32)
    measured = np.asarray(data['measured'], dtype=np.float32)
    raw = np.asarray(data['raw'], dtype=np.float32); raw[16:] = anchor[16:]
    velocity = np.full(24, .6); acceleration = np.full(24, 2.)
    velocity[[7,15]] = .05; acceleration[[7,15]] = .2
    velocity[16:] = .3; acceleration[16:] = 1.
    limiter = JointKinematicLimiter(sample_hz=100, max_velocity=velocity, max_acceleration=acceleration)
    limiter.reset(anchor)
    targets = [limiter.update(raw).copy() for _ in range(10)]
    task = dict(anchor=anchor, fallback_anchor=anchor, committed=[], targets=targets, raw=raw)
    old = check_sampled_segment_window(layer, measured, task)
    elapsed = []
    for _ in range(20):
        started = time.perf_counter()
        result = check_sampled_segment_window(layer, measured, dict(task, allow_sampled_escape=True))
        elapsed.append((time.perf_counter()-started)*1000)
    print(json.dumps(dict(approximate_feedback=data.get('approximate_feedback'), physical_commands=0,
        old_blocked=old['decision'].hard_limit_triggered, escape_approved=result['escape_approved'],
        reason=result['decision'].reason, compute_p50_p95_max_ms=np.percentile(elapsed,[50,95,100]).tolist())))
    if not result['escape_approved']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
