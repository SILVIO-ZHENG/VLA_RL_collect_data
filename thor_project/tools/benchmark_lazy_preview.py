"""Pure CPU/equality benchmark on G2; fake adapter only, no SDK connection."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing benchmark")
    sys.path.insert(0, str(args.bundle))
    from robot_client.action_filter import CausalLowPassFilter
    from robot_client.action_plan import ActionPlan
    from robot_client.client import RobotClientRunner

    if "agibot_gdk" in sys.modules:
        raise RuntimeError("SDK imported unexpectedly")
    result, outputs, previews = {}, {}, {}
    for lazy in (False, True):
        obj = object.__new__(RobotClientRunner)
        obj._held_head_waist_values = None
        obj._held_effector_values = None
        obj._action_filter = CausalLowPassFilter(cutoff_hz=2, sample_hz=100, action_dim=24)
        obj._action_filter.reset(np.zeros(24))
        obj._effector_action_filter = None
        obj._control_lock = threading.Lock()
        active, checked, sampled, timings = [0], [], [], []

        def send(target, safety_context):
            if active[0] % 5 == 0:
                raw = safety_context["trajectory_preview"]
                checked.append((raw() if callable(raw) else raw).copy())
            return target.copy()

        obj.adapter = SimpleNamespace(_safety_layer=True, send_action=send, last_action_intervened=False)
        plan = ActionPlan(24, 30, position_target_indices=(7, 15))
        plan.set_hold_action(np.zeros(24))
        q = np.repeat((0.05 * np.sin(np.arange(500) / 50))[:, None], 24, axis=1).astype(np.float32)
        q[:, [7, 15]] = 0.025
        plan.install("test", q, plan.snapshot(), now=0)
        calls = [0]
        for index in range(1000):
            active[0] = index
            started = time.perf_counter()
            reservation = plan.reserve(now=index / 100)

            def source(r=reservation, t=index / 100):
                calls[0] += 1
                return plan.preview_reserved(r, now=t, period=0.01)

            adopted = obj._send_action(reservation.action, trajectory_preview=source if lazy else source())
            plan.commit(reservation, adopted)
            timings.append((time.perf_counter() - started) * 1000)
            sampled.append(adopted)
        name = "lazy" if lazy else "eager"
        outputs[name], previews[name] = np.array(sampled), np.array(checked)
        result[name] = dict(
            preview_calls=calls[0],
            ticks=1000,
            submitted_previews=len(checked),
            cpu_ms=dict(zip(("p50", "p95", "max"), np.percentile(timings, (50, 95, 100)).tolist())),
        )
    np.testing.assert_array_equal(outputs["eager"], outputs["lazy"])
    np.testing.assert_array_equal(previews["eager"], previews["lazy"])
    result.update(
        physical_commands=0,
        gdk_imported="agibot_gdk" in sys.modules,
        bitwise_equal=True,
        scope="fake adapter, nominal one refill per 5 ticks; not a real-time/geometry benchmark",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
