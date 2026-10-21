"""Summarize owned dry-run controller evidence, including one-tick brakes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    results = []
    for path in sorted(a.directory.glob('control-*.json')):
        data = json.loads(path.read_text())
        if 'rows' not in data:
            continue
        entry = {k:v for k,v in data.items() if k != 'rows'}
        entry['file'] = str(path)
        audit = path.with_suffix('.audit.jsonl')
        if audit.exists():
            records = [json.loads(x) for x in audit.read_text().splitlines()]
            dispatch = [r for r in records if r.get('record_type') == 'servo_dispatch']
            entry['audit_dispatch_counts_including_warmup'] = dict(Counter(r['collision_response'] for r in dispatch))
            entry['any_physical_acceptance'] = any(r.get('gdk_command_accepted') is not None for r in dispatch)
            checks = [r['safety_guard_latency_ms'] for r in records if r.get('record_type')=='worker_check']
            if checks:
                entry['geometry'] = dict(count=len(checks), p95_ms=float(np.percentile(checks,95)),
                                         max_ms=max(checks), over20=sum(x>20 for x in checks))
        rows = data['rows']
        if len(rows)>2:
            q = np.asarray([r['q'] for r in rows])
            v = np.asarray([r['velocity'] for r in rows])
            entry['nominal_10ms_derivative_consistency_error'] = float(np.max(np.abs(np.diff(q,axis=0)/.01-v[1:])))
            entry['max_nominal_acceleration'] = float(np.max(np.abs(np.diff(v,axis=0)/.01)))
        results.append(entry)
    with a.output.open('x') as f:
        json.dump(dict(physical_commands=0, actual_robot_dynamics_validated=False, results=results), f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
