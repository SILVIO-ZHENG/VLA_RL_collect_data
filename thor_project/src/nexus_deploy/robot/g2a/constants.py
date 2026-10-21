"""Wire-level and GDK constants shared by the standalone robot client."""

PROTOCOL_NAME = "nexus-policy-websocket"
PROTOCOL_VERSION = 2
SUPPORTED_PROTOCOLS = (
    {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
    {"name": "g2a-policy-websocket", "version": 2},
    {"name": "g2a-pi05-websocket", "version": 1},
)
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

CAMERA_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

LEFT_ARM_JOINTS = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
)
RIGHT_ARM_JOINTS = (
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)
HEAD_JOINTS = ("idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3")
WAIST_JOINTS = (
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
)
GDK_BODY_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS + WAIST_JOINTS
DEFAULT_EFFECTOR_JOINT_NAMES = (
    "idx31_gripper_l_inner_joint1",
    "idx71_gripper_r_inner_joint1",
)

# GDK hard position limits from the vendor control_demo.py. Unlike the
# user-configurable relative-target warning, joint_servo_control rejects an
# entire request when any body target is outside these absolute ranges.
GDK_BODY_JOINT_LIMITS = {
    "idx01_body_joint1": (-1.082104, 0.000174),
    "idx02_body_joint2": (-0.000174, 2.652900),
    "idx03_body_joint3": (-1.919862, 1.570970),
    "idx04_body_joint4": (-0.436332, 0.436332),
    "idx05_body_joint5": (-3.045599, 3.045599),
    "idx11_head_joint1": (-1.570970, 1.570970),
    "idx12_head_joint2": (-0.349240, 0.349240),
    "idx13_head_joint3": (-0.534773, 0.534773),
    "idx21_arm_l_joint1": (-3.071796, 3.071796),
    "idx22_arm_l_joint2": (-2.059505, 2.059505),
    "idx23_arm_l_joint3": (-3.071796, 3.071796),
    "idx24_arm_l_joint4": (-2.495838, 1.012308),
    "idx25_arm_l_joint5": (-3.071796, 3.071796),
    "idx26_arm_l_joint6": (-1.012308, 1.012308),
    "idx27_arm_l_joint7": (-1.535907, 1.535907),
    "idx61_arm_r_joint1": (-3.071796, 3.071796),
    "idx62_arm_r_joint2": (-2.059505, 2.059505),
    "idx63_arm_r_joint3": (-3.071796, 3.071796),
    "idx64_arm_r_joint4": (-2.495838, 1.012308),
    "idx65_arm_r_joint5": (-3.071796, 3.071796),
    "idx66_arm_r_joint6": (-1.012308, 1.012308),
    "idx67_arm_r_joint7": (-1.535907, 1.535907),
}

# Canonical network order:
# left arm 7, left effector, right arm 7, right effector, head 3, waist 5.
ACTION_DIM = 24
LEFT_EFFECTOR_INDEX = 7
RIGHT_EFFECTOR_INDEX = 15
GRIPPER_MIN_POSITION = -0.785
GRIPPER_MAX_POSITION = 0.0
DH_GRIPPER_CLOSED_POSITION = 0.0
DH_GRIPPER_OPEN_POSITION = 0.025

CANONICAL_STATE_NAMES = (
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

# Single Nexus runtime start profile.  The previous dataset median is retired;
# the legacy DATASET_START_STATE_24D name below remains only as an API alias.
APPROVED_START_STATE_PROFILE_ID = "g2_sorting_home_20260817_v1"
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

# Compatibility and feature-specific aliases all resolve to the same tuple so
# model warmup, client checks, mock state and reset cannot silently diverge.
DATASET_START_STATE_24D = APPROVED_START_STATE_24D
BODY_RESET_PROFILE_ID = APPROVED_START_STATE_PROFILE_ID
BODY_RESET_TARGET_24D = APPROVED_START_STATE_24D

ARM_CANONICAL_INDICES = tuple(range(0, 7)) + tuple(range(8, 15))
HEAD_CANONICAL_INDICES = tuple(range(16, 19))
WAIST_CANONICAL_INDICES = tuple(range(19, 24))
EFFECTOR_CANONICAL_INDICES = (LEFT_EFFECTOR_INDEX, RIGHT_EFFECTOR_INDEX)

if (
    len(CANONICAL_STATE_NAMES) != ACTION_DIM
    or len(APPROVED_START_STATE_24D) != ACTION_DIM
    or len(DATASET_START_STATE_24D) != ACTION_DIM
    or len(BODY_RESET_TARGET_24D) != ACTION_DIM
):
    raise RuntimeError("Standalone G2A canonical constants must contain exactly 24 dimensions")
