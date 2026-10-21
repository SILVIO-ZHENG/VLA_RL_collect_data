#!/usr/bin/env python
"""Canonical Nexus start state used by model, client, mock and reset tooling.

The previous per-joint median over 263 dataset episodes is retired.  The
approved profile below was captured from the physical sorting G2 on
2026-08-17 and is now the single runtime reference.  Compatibility aliases
retain the old ``DATASET_START_STATE_*`` API names for downstream callers.
"""

from __future__ import annotations

from nexus_deploy.robot.g2a.constants import CANONICAL_STATE_NAMES

DATASET_BASENAME = "logistics_sorting_640x400_effector_v30"
DATASET_EPISODE_COUNT = 263
APPROVED_START_STATE_PROFILE_ID = "g2_sorting_home_20260817_v1"
APPROVED_START_STATE_METHOD = (
    "physical sorting G2 reference captured on 2026-08-17; right arm mirrored "
    "from the measured left arm using the G2 joint-sign convention"
)

# This tolerance describes the dataset analysis; it is not a robot safety limit.
DATASET_START_STATE_ANALYSIS_ATOL = 0.005
DATASET_16D_EPISODES_WITHIN_ATOL = 221

# Keep this explicit so an adapter ordering change cannot silently remap values.
DATASET_START_STATE_JOINT_NAMES = (
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "left_effector_position",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "right_effector_position",
    "head_joint1",
    "head_joint2",
    "head_joint3",
    "waist_joint1",
    "waist_joint2",
    "waist_joint3",
    "waist_joint4",
    "waist_joint5",
)

APPROVED_START_STATE_24D = (
    0.76465964,
    -0.31351972,
    -0.70621310,
    -1.36066850,
    0.46818197,
    -0.83775747,
    -0.17443806,
    -0.78357828,
    -0.76465964,
    -0.31351972,
    0.70621310,
    -1.36066850,
    -0.46818197,
    -0.83775747,
    0.17443806,
    -0.78500003,
    0.0,
    0.0,
    0.41756022,
    -0.55547464,
    0.81430656,
    -0.08542661,
    0.01562496,
    0.03201678,
)

APPROVED_START_STATE_16D = APPROVED_START_STATE_24D[:16]
APPROVED_START_STATE_HEAD_WAIST_8D = APPROVED_START_STATE_24D[16:24]
APPROVED_START_STATE_JOINT_NAMES = DATASET_START_STATE_JOINT_NAMES

# Backward-compatible names. These now intentionally resolve to the approved
# Nexus profile; the retired dataset median is no longer a runtime constant.
DATASET_START_STATE_METHOD = APPROVED_START_STATE_METHOD
DATASET_START_STATE_24D = APPROVED_START_STATE_24D
DATASET_START_STATE_16D = APPROVED_START_STATE_16D
DATASET_START_STATE_HEAD_WAIST_8D = APPROVED_START_STATE_HEAD_WAIST_8D

# Only these dimensions are bit-for-bit identical across all 263 episode starts.
DATASET_STRICTLY_FIXED_START_JOINTS = (
    ("head_joint1", 0.0),
    ("head_joint2", 0.0),
)

if tuple(CANONICAL_STATE_NAMES) != APPROVED_START_STATE_JOINT_NAMES:
    raise RuntimeError("Approved start-state ordering no longer matches the canonical G2A adapter ordering")


def get_dataset_start_state(native_dim: int) -> tuple[float, ...]:
    """Compatibility wrapper returning the approved Nexus start state."""
    if native_dim == 16:
        return DATASET_START_STATE_16D
    if native_dim == 24:
        return DATASET_START_STATE_24D
    raise ValueError(f"G2A approved start state is only defined for 16D or 24D, got {native_dim}")


def get_dataset_start_state_by_joint() -> dict[str, float]:
    """Compatibility wrapper returning the approved name-to-value mapping."""
    return dict(
        zip(
            DATASET_START_STATE_JOINT_NAMES,
            DATASET_START_STATE_24D,
            strict=True,
        )
    )


def get_approved_start_state(native_dim: int) -> tuple[float, ...]:
    """Return the approved Nexus reference in canonical 16D or 24D order."""
    return get_dataset_start_state(native_dim)


def get_approved_start_state_by_joint() -> dict[str, float]:
    """Return a new approved name-to-value mapping."""
    return get_dataset_start_state_by_joint()
