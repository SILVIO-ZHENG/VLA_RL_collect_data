"""Wall-clock rolling-controller benchmark. Hard-coded dry-run; no GDK import."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-dir', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--pose-log', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--model-fps', type=int, choices=(15,30), default=15)
    p.add_argument('--geometry', choices=('legacy','bounds'), default='legacy')
    p.add_argument('--inject-collision', action='store_true', help='Dry-run only: inject blocked results from 2 to 5 seconds')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.seconds <= 60:
        p.error('seconds must be in [1, 60]')
    if args.inject_collision and (args.geometry != 'legacy' or args.seconds < 6):
        p.error('collision injection requires legacy geometry and at least 6 seconds')
    # Reserve a new result file before starting; never overwrite prior evidence.
    with args.output.open('x') as output:
        sys.path.insert(0, str(args.client_dir.resolve()))
        from robot_client.gdk_adapter import AdapterConfig, G2AAdapter
        with args.pose_log.open() as stream:
            records = (json.loads(line) for line in stream)
            record = next(row for row in records if 'safe_action' in row)
            pose = np.asarray(record.get('measured_feedback', record['safe_action']), dtype=np.float32)
        cfg = AdapterConfig(hardware_mode='dry-run', effector_profile='dahuan',
                            servo_control_period_s=.01, safety_check_fps=50,
                            async_safety_enabled=True, certified_braking_enabled=args.geometry=='bounds',
                            rolling_legacy_geometry=args.geometry=='legacy',
                            rolling_safety_enabled=True, safety_config_path=args.profile,
                            safety_log_path=args.output.with_suffix('.audit.jsonl'))
        adapter = G2AAdapter(cfg)
        rows, failure = [], None
        injected = [0]
        try:
            adapter._dry_state = pose.copy()
            adapter.connect()
            adapter.reset_command_reference(pose)
            # Warm up geometry/feedback at a stationary recorded configuration.
            start = time.perf_counter()
            while not adapter._async_ever_dispatched:
                adapter.send_action(pose)
                if time.perf_counter()-start > 8:
                    raise RuntimeError('Dry-run warmup failed')
                time.sleep(.01)
            start = deadline = time.perf_counter()
            if args.inject_collision:
                original_callback = adapter._async_worker.on_result
                def inject(result):
                    if 2 <= time.perf_counter()-start < 5 and not result.get('error'):
                        result['live_decision'] = replace(result['live_decision'], hard_limit_triggered=True)
                        injected[0] += 1
                    if original_callback is not None:
                        original_callback(result)
                adapter._async_worker.on_result = inject
            last_model_tick = -1
            while time.perf_counter()-start < args.seconds:
                now = time.perf_counter()
                model_tick = int((now-start)*args.model_fps)
                if model_tick != last_model_tick:
                    target = pose.copy()
                    target[0] += .06*math.sin(2*math.pi*.2*(now-start))
                    target[8] -= .04*math.sin(2*math.pi*.2*(now-start))
                    last_model_tick = model_tick
                adopted = adapter.send_action(target)
                rows.append(dict(t=now-start, call_ms=(time.perf_counter()-now)*1000,
                                 buffer=len(adapter._async_steps), intervened=adapter.last_action_intervened,
                                 braking=adapter.last_async_response == 'async_checked_brake',
                                 response=adapter.last_async_response, q=adopted.tolist(),
                                 velocity=adapter._kinematic_limiter.velocity.tolist()))
                deadline += .01
                time.sleep(max(0, deadline-time.perf_counter()))
                # No burst catch-up after a scheduling stall.
                deadline = max(deadline, time.perf_counter()-.001)
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
        finally:
            adapter.disconnect()
        assert 'agibot_gdk' not in sys.modules
        data = dict(hardware_mode='dry-run', gdk_imported=False, physical_commands=0,
                    synthetic_target_fps=args.model_fps,
                    geometry=args.geometry,
                    feedback='simulated following of issued dry-run targets', failure=failure, samples=len(rows),
                    intervention_ticks=sum(r['intervened'] for r in rows),
                    braking_ticks=sum(r['braking'] for r in rows), rows=rows)
        if rows:
            data.update(rate_hz=(len(rows)-1)/max(rows[-1]['t']-rows[0]['t'], 1e-9),
                        call_p95_ms=float(np.percentile([r['call_ms'] for r in rows],95)),
                        call_max_ms=max(r['call_ms'] for r in rows))
        if args.inject_collision:
            held = [r for r in rows if r['response'] == 'async_collision_hold']
            stable = bool(held) and all(r['q'] == held[0]['q'] for r in held)
            resumed = bool(held) and any(r['t'] > held[-1]['t'] and r['response'] == 'async_checked_motion' for r in rows)
            data.update(artificial_collision=True, injected_checks=injected[0],
                        historical_target_constant=stable, automatically_resumed=resumed)
            if not stable or not resumed:
                failure = data['failure'] = failure or 'Collision hold/resume validation failed'
        json.dump(data, output, indent=2)
        print(json.dumps({k:v for k,v in data.items() if k != 'rows'}))
        if failure:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
