"""Pure command-prefix preview for RTC experiments; never dispatch authority."""

from __future__ import annotations

import math

import numpy as np


def build_command_prefix(
    limiter, current, committed, references, *, policy_hz, delay_steps, effector_profile="dahuan", clip_position=None
):
    """Copy the frozen queue, then privately forecast the uncommitted suffix.

    Sampling uses the policy clock. A committed sample is only a statement
    about this queue snapshot, not a renewed geometric certificate. Planned
    samples can change after a hold and must never authorize robot motion.
    Gripper targets use zero-order hold and are converted back to model units.
    """
    if type(delay_steps) is not int or not 1 <= delay_steps <= 21:
        raise ValueError("Command prefix delay must be an integer in [1, 21]")
    if not math.isfinite(policy_hz) or policy_hz <= 0 or limiter.dt != 0.01:
        raise ValueError("Command prefix requires a positive policy rate and 100 Hz limiter")
    current = np.asarray(current, dtype=np.float32)
    refs = np.asarray(references, dtype=np.float32)
    last_time = (delay_steps - 1) / policy_hz
    ticks = int(math.ceil(last_time / limiter.dt - 1e-9))
    if (
        current.shape != (24,)
        or not np.isfinite(current).all()
        or refs.ndim != 2
        or refs.shape[1] != 24
        or len(refs) < ticks + 2
        or not np.isfinite(refs).all()
    ):
        raise ValueError("Incomplete finite command-prefix references")
    forecast = limiter.fork()
    positions = [current.copy()]
    for tick in range(1, ticks + 1):
        if tick <= len(committed):
            step = committed[tick - 1]
            q = np.asarray(step["action"], dtype=np.float32)
            if q.shape != (24,) or not np.isfinite(q).all():
                raise ValueError("Invalid committed command")
            forecast = step["limiter"].fork()
        else:
            q = forecast.update_tracking(refs[tick], (refs[tick + 1] - refs[tick]) / forecast.dt)
            if clip_position is not None:
                clipped = np.asarray(clip_position(q), dtype=np.float32)
                if clipped.shape != (24,) or not np.isfinite(clipped).all():
                    raise ValueError("Invalid bounded preview command")
                if not np.array_equal(q, clipped):
                    forecast.reconcile_position_clip(clipped)
                q = clipped
        positions.append(q.copy())
    positions = np.stack(positions)
    offsets = np.arange(delay_steps, dtype=np.float64) / policy_hz
    phase = offsets / limiter.dt
    left = np.minimum(np.floor(phase + 1e-9).astype(int), ticks)
    right = np.minimum(left + 1, ticks)
    fraction = (phase - left)[:, None]
    hardware = (positions[left] + fraction * (positions[right] - positions[left])).astype(np.float32)
    hardware[:, [7, 15]] = positions[left][:, [7, 15]]
    model = hardware[:, :16].copy()
    if effector_profile == "dahuan":
        if np.any(hardware[:, [7, 15]] < -1e-7) or np.any(hardware[:, [7, 15]] > 0.025001):
            raise ValueError("Gripper command outside configured opening range")
        model[:, [7, 15]] = np.clip(0.025 - hardware[:, [7, 15]], 0.0, 0.025)
    elif effector_profile != "omnipicker_legacy":
        raise ValueError("Unsupported prefix effector profile")
    committed_steps = int(np.count_nonzero(phase <= len(committed) + 1e-9))
    return {
        "source": "robot_command_preview",
        "actions": np.ascontiguousarray(model),
        "hardware_actions": hardware,
        "offsets_s": offsets,
        "committed_steps": committed_steps,
        "predicted_steps": delay_steps - committed_steps,
        "not_motion_authority": True,
    }


def validate_command_prefix(packet, *, delay_steps, native_dim):
    """Reject malformed/ambiguous input instead of silently using a model tail."""
    if not isinstance(packet, dict) or packet.get("source") != "robot_command_preview":
        raise ValueError("Invalid command-prefix source")
    if packet.get("not_motion_authority") is not True:
        raise ValueError("Command prefix must not claim motion authority")
    committed = packet.get("committed_steps")
    if type(committed) is not int or not 0 <= committed <= delay_steps:
        raise ValueError("Invalid committed prefix length")
    actions = np.asarray(packet.get("actions"), dtype=np.float32)
    if actions.shape != (delay_steps, native_dim) or not np.isfinite(actions).all():
        raise ValueError("Command prefix must contain exactly the finite delayed actions")
    return actions.copy(), committed
