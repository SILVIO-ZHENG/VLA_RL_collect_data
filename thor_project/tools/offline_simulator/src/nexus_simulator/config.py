"""仿真配置的 fail-closed 校验。

当前模块仅使用 Python 标准库，并且刻意不包含任何机器人连接能力。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SimulationConfigError(ValueError):
    """仿真配置不满足隔离或数值约束。"""


@dataclass(frozen=True)
class WorkSurface:
    frame: str
    height: float
    hard_margin: float
    xy_bounds: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class GuardConfig:
    enabled: bool
    path_samples: int
    deadline_ms: float
    max_consecutive_deadline_misses: int
    collision_response: str


@dataclass(frozen=True)
class SimulationConfig:
    profile: str
    simulation_only: bool
    calibrated: bool
    deployable: bool
    work_surface: WorkSurface
    guard: GuardConfig


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationConfigError(f"{field} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise SimulationConfigError(f"{field} 必须是有限数值")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SimulationConfigError(f"{field} 必须是正整数")
    return value


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise SimulationConfigError("work_surface.xy_bounds 必须为 null 或四元素数组")
    xmin, xmax, ymin, ymax = (
        _finite_number(item, f"work_surface.xy_bounds[{index}]")
        for index, item in enumerate(value)
    )
    if xmin >= xmax or ymin >= ymax:
        raise SimulationConfigError("work_surface.xy_bounds 必须满足 xmin<xmax 且 ymin<ymax")
    return xmin, xmax, ymin, ymax


def parse_simulation_config(payload: dict[str, Any]) -> SimulationConfig:
    """解析配置，并强制其不能被误认为真机配置。"""

    if payload.get("simulation_only") is not True:
        raise SimulationConfigError("simulation_only 必须为 true")
    if payload.get("calibrated") is not False:
        raise SimulationConfigError("仿真配置 calibrated 必须为 false")
    if payload.get("deployable") is not False:
        raise SimulationConfigError("仿真配置 deployable 必须为 false")

    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise SimulationConfigError("profile 必须是非空字符串")

    surface_payload = payload.get("work_surface")
    if not isinstance(surface_payload, dict):
        raise SimulationConfigError("work_surface 必须是对象")
    frame = surface_payload.get("frame")
    if frame != "base":
        raise SimulationConfigError("第一阶段仅允许 work_surface.frame='base'")
    height = _finite_number(surface_payload.get("height"), "work_surface.height")
    hard_margin = _finite_number(
        surface_payload.get("hard_margin"), "work_surface.hard_margin"
    )
    if hard_margin <= 0:
        raise SimulationConfigError("work_surface.hard_margin 必须大于 0")

    guard_payload = payload.get("guard")
    if not isinstance(guard_payload, dict):
        raise SimulationConfigError("guard 必须是对象")
    enabled = guard_payload.get("enabled")
    if not isinstance(enabled, bool):
        raise SimulationConfigError("guard.enabled 必须是布尔值")
    deadline_ms = _finite_number(guard_payload.get("deadline_ms"), "guard.deadline_ms")
    if deadline_ms <= 0:
        raise SimulationConfigError("guard.deadline_ms 必须大于 0")
    response = guard_payload.get("collision_response")
    allowed_responses = {"hold_previous_safe_action", "hold_measured_pose"}
    if response not in allowed_responses:
        raise SimulationConfigError(
            "guard.collision_response 必须是 hold_previous_safe_action 或 hold_measured_pose"
        )

    return SimulationConfig(
        profile=profile.strip(),
        simulation_only=True,
        calibrated=False,
        deployable=False,
        work_surface=WorkSurface(
            frame=frame,
            height=height,
            hard_margin=hard_margin,
            xy_bounds=_parse_bounds(surface_payload.get("xy_bounds")),
        ),
        guard=GuardConfig(
            enabled=enabled,
            path_samples=_positive_int(guard_payload.get("path_samples"), "guard.path_samples"),
            deadline_ms=deadline_ms,
            max_consecutive_deadline_misses=_positive_int(
                guard_payload.get("max_consecutive_deadline_misses"),
                "guard.max_consecutive_deadline_misses",
            ),
            collision_response=response,
        ),
    )


def load_simulation_config(path: str | Path) -> SimulationConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationConfigError(f"无法读取仿真配置 {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SimulationConfigError("配置根节点必须是对象")
    return parse_simulation_config(payload)
