"""Read the idle G2A pose and report whether it matches the approved reset pose.

This command acquires the same GDK lock as the robot client and reset tool. It
only reads feedback; it never configures control mode or sends a joint target.
The caller must separately verify robot identity, client idleness and the
absence of external GDK users before interpreting its result.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from typing import Any, Callable, Sequence

if __package__:
    from .constants import (
        ARM_CANONICAL_INDICES,
        BODY_RESET_PROFILE_ID,
        BODY_RESET_TARGET_24D,
        CANONICAL_STATE_NAMES,
        DH_GRIPPER_OPEN_POSITION,
        EFFECTOR_CANONICAL_INDICES,
        HEAD_CANONICAL_INDICES,
        WAIST_CANONICAL_INDICES,
    )
else:
    from robot_client.constants import (
        ARM_CANONICAL_INDICES,
        BODY_RESET_PROFILE_ID,
        BODY_RESET_TARGET_24D,
        CANONICAL_STATE_NAMES,
        DH_GRIPPER_OPEN_POSITION,
        EFFECTOR_CANONICAL_INDICES,
        HEAD_CANONICAL_INDICES,
        WAIST_CANONICAL_INDICES,
    )


BODY_TOLERANCE_RAD = 0.03
EFFECTOR_TOLERANCE_M = 0.0015
BODY_STABILITY_RAD = 0.01
EFFECTOR_STABILITY_M = 0.001
SAMPLE_COUNT = 3
SAMPLE_INTERVAL_S = 0.1
STARTUP_WAIT_S = 2.0
RESULT_PREFIX = "HOME_POSE_JSON="
BODY_INDICES = ARM_CANONICAL_INDICES + HEAD_CANONICAL_INDICES + WAIST_CANONICAL_INDICES


def _unknown(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "unknown",
        "at_home": None,
        "profile_id": BODY_RESET_PROFILE_ID,
        "checked_at": time.time(),
        "reason": str(reason)[:400],
    }


def assess_samples(samples: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Classify three finite 24D feedback samples against the reset target."""

    if len(samples) != SAMPLE_COUNT:
        raise ValueError(f"expected {SAMPLE_COUNT} pose samples")
    vectors = [tuple(float(value) for value in sample) for sample in samples]
    if any(len(vector) != len(BODY_RESET_TARGET_24D) or
           not all(math.isfinite(value) for value in vector) for vector in vectors):
        raise ValueError("missing or non-finite canonical 24D joint feedback")

    target = list(BODY_RESET_TARGET_24D)
    for index in EFFECTOR_CANONICAL_INDICES:
        target[index] = DH_GRIPPER_OPEN_POSITION

    body_error = max(abs(vector[index] - target[index])
                     for vector in vectors for index in BODY_INDICES)
    effector_error = max(abs(vector[index] - target[index])
                         for vector in vectors for index in EFFECTOR_CANONICAL_INDICES)
    body_drift = max(abs(vector[index] - vectors[0][index])
                     for vector in vectors[1:] for index in BODY_INDICES)
    effector_drift = max(abs(vector[index] - vectors[0][index])
                         for vector in vectors[1:] for index in EFFECTOR_CANONICAL_INDICES)
    within = [
        all(abs(vector[index] - target[index]) <= BODY_TOLERANCE_RAD
            for index in BODY_INDICES)
        and all(abs(vector[index] - target[index]) <= EFFECTOR_TOLERANCE_M
                for index in EFFECTOR_CANONICAL_INDICES)
        for vector in vectors
    ]
    stable = body_drift <= BODY_STABILITY_RAD and effector_drift <= EFFECTOR_STABILITY_M
    if not stable or any(within) != all(within):
        status, reason = "unknown", "feedback moved or crossed the reset-pose threshold"
    elif all(within):
        status, reason = "at_home", "all body joints and Dahuan grippers match the reset pose"
    else:
        status, reason = "not_home", "one or more joints differ from the reset pose"

    worst_index = max(range(len(target)), key=lambda index: max(
        abs(vector[index] - target[index]) / (
            EFFECTOR_TOLERANCE_M if index in EFFECTOR_CANONICAL_INDICES else BODY_TOLERANCE_RAD)
        for vector in vectors
    ))
    return {
        "schema_version": 1,
        "status": status,
        "at_home": True if status == "at_home" else False if status == "not_home" else None,
        "profile_id": BODY_RESET_PROFILE_ID,
        "checked_at": time.time(),
        "sample_count": len(vectors),
        "body_tolerance_rad": BODY_TOLERANCE_RAD,
        "effector_tolerance_m": EFFECTOR_TOLERANCE_M,
        "body_max_error_rad": round(body_error, 6),
        "effector_max_error_m": round(effector_error, 6),
        "body_max_drift_rad": round(body_drift, 6),
        "effector_max_drift_m": round(effector_drift, 6),
        "worst_joint": CANONICAL_STATE_NAMES[worst_index],
        "reason": reason,
    }


def probe_hardware(
    gdk: Any,
    lock: Any,
    read_state: Callable[[Any], tuple[Sequence[float], Any]],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Own GDK exclusively, read feedback, and always release both resources."""

    acquired = False
    initialized = False
    result: dict[str, Any] | None = None
    cleanup_errors: list[str] = []
    try:
        lock.acquire()
        acquired = True
        init_result = gdk.gdk_init()
        if init_result is not None and init_result != 0 and init_result != gdk.GDKRes.kSuccess:
            raise RuntimeError(f"GDK initialization failed: {init_result!r}")
        initialized = True
        robot = gdk.Robot()
        sleep(STARTUP_WAIT_S)
        samples = []
        for index in range(SAMPLE_COUNT):
            if index:
                sleep(SAMPLE_INTERVAL_S)
            # A fallback effector-name warning must not corrupt the JSON line.
            with contextlib.redirect_stdout(sys.stderr):
                state, _names = read_state(robot)
            samples.append(tuple(float(value) for value in state))
        result = assess_samples(samples)
    except Exception as error:
        result = _unknown(str(error))
    finally:
        if initialized:
            try:
                gdk.gdk_release()
            except Exception as error:
                cleanup_errors.append(f"GDK release failed: {error}")
        if acquired:
            try:
                lock.release()
            except Exception as error:
                cleanup_errors.append(f"GDK lock release failed: {error}")
    if cleanup_errors:
        return _unknown("; ".join(cleanup_errors))
    assert result is not None
    return result


def main() -> int:
    try:
        import agibot_gdk
        if __package__:
            from .reset_to_dataset_start import read_canonical_state
            from .robot_lock import RobotProcessLock
        else:
            from robot_client.reset_to_dataset_start import read_canonical_state
            from robot_client.robot_lock import RobotProcessLock
        result = probe_hardware(agibot_gdk, RobotProcessLock(), read_canonical_state)
    except Exception as error:
        result = _unknown(str(error))
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["status"] in {"at_home", "not_home"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
