# 中文：
# 默认值的第一层，不是现场最终配置：ControlPlane 会应用站点 TOML，
# 调度 profile 和已保存任务又会覆盖部分值，最后 deployment 形成 argv。
# 数值截止频率存在不等于低通启用；必须同时看 safety 中的布尔开关。
# English:
# First-layer defaults, not effective field configuration. Site TOML, scheduler profiles,
# and saved tasks can override them before deployment builds argv. A cutoff value does
# not imply an enabled filter; inspect the corresponding safety boolean.
"""Service-owned task defaults before site-specific configuration is applied."""

from typing import Any

TASK_DEFAULTS: dict[str, Any] = {
    "robot_id": "",
    "robot_host": "",
    "robot_ssh_port": 22,
    "model_id": "",
    "effector_profile": "dahuan",
    "model_version": "",
    "effector_control_mode": "device-position",
    "arm_limit_profile": "urdf-velocity-uniform-acceleration",
    "chunk_smoothing_mode": "savgol",
    "action_interpolation": "linear",
    "arm_tracking_mode": "legacy",
    "arm_tracking_omega": 12.0,
    "rtc_prefix_mode": "model",
    "rtc_inference_mode": "",
    "gpu_id": "auto",
    "task": (
        "Pick a package from the box with the left gripper and place it on the table. "
        "Then pick it up with the right gripper and place it on the conveyor belt, "
        "keeping the QR code facing upward."
    ),
    "inference_mode": "rtc",
    "hardware_mode": "execute",
    "server_mode": "reuse",
    "duration_s": 0.0,
    # 中文：
    # 策略轨迹每秒推进的步数；改变它改变动作播放时间尺度。
    # servo_fps 为高频下发采样率；prediction/quality horizon 的单位仍是策略步。
    # English:
    # Trajectory steps per second control playback time. servo_fps controls high-rate
    # dispatch sampling; prediction/quality horizons remain measured in policy steps.
    "execution_fps": 30.0,
    "servo_fps": 100.0,
    "prediction_horizon": 50,
    "quality_horizon": 50,
    "request_lead_margin_steps": 8,
    "min_replan_after_steps": 1,
    "latency_window_size": 20,
    "request_timeout_s": 30.0,
    "min_rtc_blend_steps": 8,
    "warmup_requests": 1,
    "execution_preflight_requests": 5,
    "watchdog_timeout_s": 3.0,
    "approval_timeout_s": 300.0,
    "human_in_loop": False,
    "action_low_pass_cutoff_hz": 4.0,
    "effector_action_low_pass_cutoff_hz": 2.0,
    "max_arm_velocity": 3.76,
    "max_arm_acceleration": 8.0,
    "max_effector_velocity": 1.0,
    "max_effector_acceleration": 4.0,
    "max_head_waist_velocity": 0.3,
    "max_head_waist_acceleration": 1.0,
    "max_arm_tracking_error": 0.4,
    "max_effector_tracking_error": 0.5,
    "max_head_waist_tracking_error": 0.2,
    # 中文：
    # 基础速度/跟踪/看门狗、可选在线低通与工作空间碰撞是不同防护层。
    # 此字典值只提供默认；实际站点安全 profile 与最终开关决定执行边界。
    # English:
    # Kinematics/tracking/watchdog, optional online filtering, and workspace collision
    # are separate layers. These are defaults; actual site profile and final switches govern execution.
    "safety": {
        "watchdog_enabled": True,
        "kinematic_limits_enabled": True,
        "tracking_error_enabled": True,
        "low_pass_filter_enabled": False,
        "effector_low_pass_filter_enabled": False,
        "hold_effectors": False,
        "workspace_safety_enabled": False,
        "safety_profile_id": "",
    },
    # 中文：
    # 视觉门禁控制任务暂停/复位，不改模型数值。enabled 默认 False；
    # 2/5 秒参数存在并不代表当前任务启用了自动空场控制。
    # English:
    # Presence gates pause/reset tasks rather than alter policy values. enabled defaults False;
    # 2/5-second settings alone do not establish an active automatic gate.
    "presence_gate": {
        "enabled": False,
        "mode": "automatic",
        "pause_after_s": 2.0,
        "reset_after_s": 5.0,
        "auto_resume": True,
    },
}
