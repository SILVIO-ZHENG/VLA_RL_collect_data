"""Replay recorded references through the real safety queue without a robot SDK."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nexus_deploy.robot.g2a.async_safety import check_sampled_segment_window
from nexus_deploy.robot.g2a.gdk_adapter import AdapterConfig, G2AAdapter

ARM = tuple(range(7)) + tuple(range(8, 15))


class SimulatedWorker:
    """Run real geometry, delivering results on a deterministic simulated clock."""

    def __init__(self, adapter, clock):
        self.adapter, self.clock = adapter, clock
        self.feedback = (adapter._dry_state.copy(), clock[0])
        self.pending = None
        self.result = None
        self.checks = 0

    def submit(self, task):
        self.pending = (task, self.clock[0], self.feedback[0].copy())

    def advance(self):
        if self.pending is None or self.clock[0] - self.pending[1] < 0.03 - 1e-8:
            return
        task, started, measured = self.pending
        self.result = check_sampled_segment_window(self.adapter._safety_controller.guard_layer, measured, task)
        self.result.update(task=task, measured=measured, started=started, check_ms=30.0)
        self.checks += 1
        self.pending = None

    def poll(self):
        result, self.result = self.result, None
        return result

    def close(self):
        pass


def replay(references, start, profile, mode, omega, directory):
    clock = [100.0]
    with (
        patch("nexus_deploy.robot.g2a.gdk_adapter.time.perf_counter", side_effect=lambda: clock[0]),
        patch("nexus_deploy.robot.g2a.gdk_adapter.AsyncSafetyWorker", return_value=None),
    ):
        adapter = G2AAdapter(
            AdapterConfig(
                hardware_mode="dry-run",
                startup_wait_s=0,
                servo_control_period_s=0.01,
                safety_check_fps=50,
                safety_config_path=profile,
                safety_log_path=directory / f"{mode}-{omega}.jsonl",
                async_safety_enabled=True,
                rolling_safety_enabled=True,
                rolling_legacy_geometry=True,
                effector_control_mode="device-position",
                arm_limit_profile="urdf-velocity-uniform-acceleration",
                max_arm_velocity=3.76,
                max_arm_acceleration=8.0,
                arm_tracking_mode=mode,
                arm_tracking_omega=omega,
            )
        )
        adapter._dry_state = start.copy()
        adapter.connect()  # dry-run branch returns before the GDK import.
        adapter.reset_command_reference(start)
        worker = SimulatedWorker(adapter, clock)
        adapter._async_worker = worker
        points, velocities, responses = [], [], []
        try:
            for tick in range(len(references) - 20):
                clock[0] = 100.0 + tick * 0.01
                # Explicit ideal-following assumption, identical in all arms.
                worker.feedback = (adapter._dry_state.copy(), clock[0])
                worker.advance()
                q = adapter.send_action(
                    references[tick], safety_context={"trajectory_preview": references[tick : tick + 20]}
                )
                points.append(q.copy())
                velocities.append(adapter._kinematic_limiter.velocity.copy())
                responses.append(adapter.last_async_response)
        finally:
            adapter.disconnect()
    points = np.stack(points)
    velocities = np.stack(velocities)
    acceleration = np.diff(velocities[:, ARM], axis=0) * 100
    jerk = np.diff(acceleration, axis=0) * 100
    lag = np.max(abs(points[:, ARM] - references[: len(points), ARM]), axis=1)
    return {
        "mode": mode,
        "omega": omega if mode == "damped" else None,
        "ticks": len(points),
        "real_geometry_checks": worker.checks,
        "responses": dict(collections.Counter(responses)),
        "arm_acceleration_max_rad_s2": float(np.max(abs(acceleration))),
        "jerk_p95_rad_s3": float(np.percentile(np.max(abs(jerk), axis=1), 95)),
        "reference_lag_p95_rad": float(np.percentile(lag, 95)),
        "reference_lag_max_rad": float(np.max(lag)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = args.source_root / "runtime/thor/robot-sessions/20261013/71dd8e8308db48cbaa1ecc5fdba67138"
    trace = next(session.glob("control-*.jsonl"))
    rows = [json.loads(line) for line in trace.open()]
    controls = [row for row in rows if row.get("record_type") == "control"]
    selected = controls[600:1620]
    assert all(not row.get("explicit_hold") for row in selected)
    references = np.asarray([row["filtered_target"] for row in selected], dtype=np.float32)
    start = np.asarray(controls[599]["adopted_target"], dtype=np.float32)
    profile = args.source_root / (
        "src/nexus_deploy/robot/g2a/G2_dahuan_standard_table_20260922_contact_refined_safety_execute_v2.json"
    )
    with tempfile.TemporaryDirectory(prefix="nexus-damped-replay-") as temporary:
        results = [
            replay(references, start, profile, mode, omega, Path(temporary))
            for mode, omega in [("legacy", 12.0), ("damped", 16.0), ("damped", 12.0)]
        ]
    report = {
        "physical_commands": 0,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "recorded_control_slice": [600, 1620],
        "limitations": [
            "100 Hz nominal clock; same recorded conditioned references with hindsight preview",
            "Ideal-following simulated feedback; geometry completion fixed at 30 ms",
            "Real queue and collision geometry, without model closed-loop or robot dynamics",
        ],
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
