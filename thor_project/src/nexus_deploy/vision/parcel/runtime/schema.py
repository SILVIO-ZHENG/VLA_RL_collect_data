"""稳定、可序列化的阶段输出协议。

Stable, JSON-serializable public schema for downstream robot applications.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .core_bridge import STATE_NAMES


# state_id 映射一旦发布就不能随意换序；消费者可同时使用字符串和整数。
# The numeric order is part of the public contract and must remain stable.
SCHEMA_VERSION = "1.0"
STATE_TO_ID = {state: index for index, state in enumerate(STATE_NAMES)}


@dataclass(frozen=True)
class StageResult:
    """单个已处理相机帧对应的一条历史确认结果。

    One history-confirmed result emitted for one actually processed camera frame.

    `state` 是下游应使用的稳定结果；`raw_state` 仅用于诊断当前单帧证据。
    Consumers should act on `state`. `raw_state` is diagnostic single-frame evidence.
    `confidence` 是几何规则证据强度，不是经过校准的统计概率。
    `confidence` is a geometric evidence score, not a calibrated probability.
    """

    # source_id：逻辑相机/视频名称；换值会触发历史重置。
    # Logical camera/video name; changing it resets temporal history.
    source_id: str
    # frame_index：原生相机时间轴上的帧号，而非本次检测的序号。
    # Native-camera timeline index, not merely the inference-call count.
    frame_index: int
    # 两个时间戳均为 Unix 墙钟纳秒；只有同一时钟域时延迟才有意义。
    # Both timestamps are wall-clock nanoseconds; latency requires the same clock domain.
    capture_timestamp_ns: int
    output_timestamp_ns: int

    # 面向下游控制的稳定阶段 / History-confirmed public stage.
    state: str  # 稳定阶段 / history-confirmed stage.
    state_id: int  # 与 state 固定对应的整数 / stable numeric enum.
    confidence: float  # 规则证据分数，不是校准概率 / evidence score, not probability.
    changed: bool  # 本次稳定阶段是否刚切换 / whether stable state changed now.
    # 仅表示 state != UNKNOWN，绝不等于“机器人可以安全运动”。
    # Means only state != UNKNOWN; it is not authorization for robot motion.
    actionable: bool
    # ok=已有稳定值；warming_up=候选累计中；unknown=证据歧义。
    # ok=stable, warming_up=collecting evidence, unknown=ambiguous evidence.
    status: str

    # 单帧诊断和候选确认进度 / Raw evidence and transition progress.
    raw_state: str  # 当前单帧几何状态，只用于诊断 / current-frame diagnostic state.
    raw_confidence: float
    candidate_state: str | None  # 正在等待确认的新阶段 / stage awaiting confirmation.
    candidate_frames: int  # 候选已连续出现的检测次数 / consecutive candidate samples.
    reason: str
    raw_reason: str

    # 当前活动包裹的可选调试信息 / Optional active-package diagnostics.
    # index 仅是当前帧检测列表下标，不是跨帧永久包裹ID。
    # The index is local to one detection result, not a persistent parcel ID.
    active_package_index: int | None
    # 在规范化后 640×400 画面中的像素坐标 / pixel coordinates in the 640×400 view.
    active_centroid: tuple[float, float] | None

    # 性能和可追溯性 / Performance and model traceability.
    # inference_ms 包含预处理、模型、规则和时序层总耗时。
    # inference_ms includes preprocessing, model, geometry rules, and temporal filtering.
    inference_ms: float
    end_to_end_ms: float  # 采集至输出总耗时 / capture-to-output latency.
    model_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """在结果离开推理层前验证协议不变量。

        Validate public-schema invariants before a result leaves the inference layer.
        """

        if self.state not in STATE_TO_ID or self.raw_state not in STATE_TO_ID:
            raise ValueError("state and raw_state must use the published state enum")
        if self.state_id != STATE_TO_ID[self.state]:
            raise ValueError("state_id does not match state")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= float(self.raw_confidence) <= 1.0:
            raise ValueError("raw_confidence must be in [0, 1]")
        if (
            self.frame_index < 0
            or self.capture_timestamp_ns < 0
            or self.output_timestamp_ns < 0
        ):
            raise ValueError("frame index and timestamps must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """转成只包含 JSON 原生类型的字典。

        Convert to a dictionary containing JSON-native values only.
        """

        payload = asdict(self)
        # tuple 在 JSON 中也能编码，但显式 list 可让输出类型跨语言更稳定。
        # An explicit list keeps the centroid type consistent across languages.
        if self.active_centroid is not None:
            payload["active_centroid"] = [
                float(self.active_centroid[0]),
                float(self.active_centroid[1]),
            ]
        return payload

    def to_json(self) -> str:
        """输出 UTF-8 友好的单行 JSON，并拒绝 NaN/Infinity。

        Emit one UTF-8-friendly JSON line and reject NaN/Infinity values.
        """

        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
