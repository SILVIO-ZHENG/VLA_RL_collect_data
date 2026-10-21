"""Real-checkpoint, fixed-input RTC prefix comparison; zero robot commands."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.configs import RTCAttentionSchedule
from lerobot.processor import TransitionKey, create_transition

from nexus_deploy.robot.g2a.action_plan import ActionPlan
from nexus_deploy.robot.g2a.command_prefix import build_command_prefix
from nexus_deploy.robot.g2a.effector_command import EffectorEndpointLatch
from nexus_deploy.robot.g2a.gdk_adapter import AdapterConfig, G2AAdapter
from nexus_deploy.vla.pi05.policy import G2APi05Policy, _SessionState

ARM = tuple(range(7)) + tuple(range(8, 15))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    session = args.source_root / "runtime/thor/robot-sessions/20261013/71dd8e8308db48cbaa1ecc5fdba67138"
    trace = next(session.glob("control-*.jsonl"))
    rows = [json.loads(line) for line in trace.open()]
    controls = [r for r in rows if r.get("record_type") == "control" and r.get("command_accepted")]
    replies = [r for r in rows if r.get("record_type") == "inference_response" and r.get("installed")]
    controls.sort(key=lambda r: r["monotonic"])
    replies.sort(key=lambda r: r["installed_monotonic"])
    times = [r["monotonic"] for r in controls]
    by_chunk = {r["chunk_id"]: r for r in replies}
    wanted = {p.name.rsplit("-cam_", 1)[0] for p in args.images.glob("*-cam_head.jpg")}
    adapter = G2AAdapter(
        AdapterConfig(
            hardware_mode="dry-run",
            servo_control_period_s=0.01,
            effector_control_mode="device-position",
            max_arm_velocity=3.76,
            max_arm_acceleration=8.0,
            arm_limit_profile="urdf-velocity-uniform-acceleration",
        )
    )
    plan = ActionPlan(
        24, policy_fps=30, position_target_indices=(7, 15), interpolation_mode="linear", cubic_indices=ARM
    )
    cases = []
    for reply in replies:
        origin = reply["observation_state_monotonic"]
        delay = reply.get("rtc", {}).get("estimated_delay_steps", 0)
        index = bisect.bisect_right(times, origin) - 1
        if reply["request_id"] in wanted and index >= 0 and delay > 0:
            count = int(np.ceil((delay - 1) / 30 / 0.01 - 1e-9)) + 2
            references = plan.preview_at(origin, period=0.01, count=count)
            assert references is not None, "Old trusted horizon cannot cover this recorded delay"
            current = np.asarray(controls[index]["adopted_target"], dtype=np.float32)
            limiter = adapter._kinematic_limiter.fork()
            limiter.reset(current)
            limiter._velocity = np.asarray(controls[index]["planner_velocity"], dtype=np.float32)
            latch = EffectorEndpointLatch(
                dimensions=2, closed_endpoint=0, open_endpoint=0.025, closed_max=0.0075, open_min=0.0175
            )
            latch.reset(current[[7, 15]])
            for point in references:
                point[[7, 15]] = latch.update(point[[7, 15]])
            committed = []
            # Archived queue arrays were not recorded. Reconstruct the short
            # immutable prefix from subsequently accepted outputs; explicitly
            # label this hindsight assumption instead of claiming live export.
            for control in controls[index + 1 : index + 9]:
                assert control.get("collision_response") == "async_checked_motion"
                tail = limiter.fork()
                tail._position = np.asarray(control["adopted_target"], dtype=np.float32)
                tail._velocity = np.asarray(control["planner_velocity"], dtype=np.float32)
                committed.append(dict(action=tail._position.copy(), limiter=tail))
            packet = build_command_prefix(
                limiter,
                current,
                committed,
                references,
                policy_hz=30,
                delay_steps=delay,
                clip_position=lambda q: adapter._apply_absolute_position_limits(q, warning_suffix="offline_prefix"),
            )
            packet.update(
                time_origin=origin,
                active_chunk_id=reply["anchor_chunk_id"],
                next_action_index=reply["anchor_next_action_index"],
            )
            metadata = json.loads((session / "inference-frames" / f"{reply['request_id']}.json").read_text())
            images = {}
            for key, info in metadata["images"].items():
                image = cv2.imread(str(args.images / info["path"]))
                assert image is not None
                images[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cases.append(
                dict(
                    reply=reply,
                    previous=by_chunk[reply["anchor_chunk_id"]],
                    packet=packet,
                    observation=dict(
                        state=np.asarray(reply["observation_state"], dtype=np.float32),
                        images=images,
                    task=(
                        "Pick a package from the box with the left gripper and place it on the table. "
                        "Then pick it up with the right gripper and place it on the conveyor belt, "
                        "keeping the QR code facing upward."
                    ),
                    ),
                )
            )
        last = bisect.bisect_right(times, reply["installed_monotonic"]) - 1
        current = np.asarray(
            controls[last]["adopted_target"] if last >= 0 else reply["observation_state"], dtype=np.float32
        )
        if reply.get("anchor_chunk_id") is None:
            plan = ActionPlan(
                24, policy_fps=30, position_target_indices=(7, 15), interpolation_mode="linear", cubic_indices=ARM
            )
        plan.set_hold_action(current)
        if last >= 1:
            t1 = controls[last].get("loop_started_monotonic") or times[last]
            t0 = controls[last - 1].get("loop_started_monotonic") or times[last - 1]
            if t1 > t0:
                plan._last_sent_velocity = (
                    current - np.asarray(controls[last - 1]["adopted_target"], dtype=np.float32)
                ) / (t1 - t0)
        plan._next_index = reply["anchor_next_action_index"]
        result = plan.install(
            reply["chunk_id"],
            np.asarray(reply["control_targets"]),
            plan.snapshot(),
            stale_steps=reply["dropped_steps"],
            now=reply["installed_monotonic"],
            rtc_handoff_horizon=reply["rtc"]["effective_execution_horizon"]
            if reply["rtc"].get("prefix_used")
            else None,
            rtc_handoff_indices=ARM,
        )
        assert result.installed
    assert len(cases) == 3, f"Expected 3 paired recorded observations, got {len(cases)}"
    torch.set_float32_matmul_precision("high")
    policy = G2APi05Policy(
        root / "models/pi05/dahuan/training_rtc_009500/pretrained_model",
        device="cuda:0",
        model_fps=30,
        execution_horizon=29,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        command_head_waist=False,
        model_profile="16d_absolute_full",
        fixed_noise_seed=1000,
        effector_profile="dahuan",
        tokenizer_path=root / "models/tokenizers/paligemma-3b-pt-224",
        rtc_inference_mode="trained",
    )
    results = []
    for number, case in enumerate(cases):
        reply, previous, packet = case["reply"], case["previous"], case["packet"]
        physical = torch.tensor(previous["model_actions"], dtype=torch.float32)[:, :16]
        transition = create_transition(action=physical)
        normalized = policy._normalizer_step(transition)[TransitionKey.ACTION].to(policy.device)
        outputs = {}
        for mode in ("model", "command", "model_repeat"):
            session_id = f"prefix-ab-{number}-{mode}"
            policy._sessions[session_id] = _SessionState(
                chunk_id=reply["anchor_chunk_id"],
                raw_actions=normalized.clone(),
                processed_native_actions=physical.clone(),
            )
            control = dict(
                enable_rtc=True,
                active_chunk_id=reply["anchor_chunk_id"],
                next_action_index=reply["anchor_next_action_index"],
                estimated_delay_steps=reply["rtc"]["estimated_delay_steps"],
                min_rtc_blend_steps=8,
            )
            if mode == "command":
                control["command_prefix"] = {
                    key: value for key, value in packet.items() if key not in ("hardware_actions", "offsets_s")
                }
            response = policy.infer(
                dict(
                    request_id=f"{session_id}-1",
                    session_id=session_id,
                    observation=case["observation"],
                    control=control,
                    reset=False,
                )
            )
            outputs[mode] = response
        delay = reply["rtc"]["estimated_delay_steps"]
        deterministic = float(np.max(abs(outputs["model"]["actions"] - outputs["model_repeat"]["actions"])))
        assert deterministic < 1e-5
        locked_error = float(np.max(abs(outputs["command"]["actions"][:delay, :16] - packet["actions"])))
        assert locked_error < 1e-5
        row = dict(
            request_id=reply["request_id"],
            delay_steps=delay,
            committed_steps=packet["committed_steps"],
            predicted_steps=packet["predicted_steps"],
            baseline_repeat_error_rad=deterministic,
            frozen_command_prefix_error=locked_error,
            modes={},
        )
        for mode in ("model", "command"):
            response = outputs[mode]
            q = response["actions"][:, list(ARM)]
            row["modes"][mode] = dict(
                rtc=response["rtc"],
                infer_ms=response["server_timing"]["infer_ms"],
                prefix_join_step_rad=float(np.max(abs(q[delay] - q[delay - 1]))),
                prefix_join_acceleration_rad_s2=float(np.max(abs(q[delay] - 2 * q[delay - 1] + q[delay - 2]))) * 900,
                tail_jerk_p95_rad_s3=float(
                    np.percentile(np.max(abs(np.diff(q[delay:], n=3, axis=0)), axis=1) * 27000, 95)
                ),
            )
        results.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        physical_commands=0,
        checkpoint="training_rtc_009500",
        fixed_noise_seed=1000,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        limitations=[
            "Three fixed recorded observations; not closed-loop task success",
            "Committed queue reconstructed retrospectively from accepted commands; "
            "uncommitted suffix forecast from recorded plan",
            "Performance metrics here do not certify altered trajectories for hardware",
        ],
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
