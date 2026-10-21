"""接入现有相机循环时使用的可配置、无积压限频层（默认 10 Hz）。

Configurable non-buffering rate gate (10 Hz by default) for existing camera loops.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .estimator import ParcelStateEstimator
from .presence import FrameInferenceResult, PresenceResult
from .schema import StageResult


class RateLimitedParcelStateService:
    """仅在配置的检测时间点处理调用方提供的最新帧（默认 10 Hz）。

    未到时间的帧直接跳过，返回 None，不排队也不伪造输出。这样既能限制延迟，
    又能防止重复旧图像错误满足连续 3/4/5 帧的确认条件。
    Process only the newest caller-supplied frame when the configured gate is due.
    Skipped frames return None, produce no fake results, and are never queued.
    This bounds latency and prevents repeated old frames from satisfying temporal
    confirmation thresholds.
    """

    def __init__(
        self,
        estimator: ParcelStateEstimator,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.estimator = estimator
        # 调度使用 monotonic 时钟，不受系统日期/NTP校时影响。
        # Scheduling uses a monotonic clock, unaffected by wall-clock/NTP changes.
        self.clock_ns = clock_ns
        self.period_ns = int(round(1_000_000_000 / estimator.config.detect_hz))
        self._next_due_ns: int | None = None
        self._latest: StageResult | None = None
        self._latest_presence: PresenceResult | None = None
        self.processed_frames = 0
        self.skipped_frames = 0

    @property
    def latest(self) -> StageResult | None:
        """返回最近一次真实推理结果；从不把它冒充新结果。

        Return the most recent real result without presenting it as a new inference.
        """

        return self._latest

    @property
    def latest_presence(self) -> PresenceResult | None:
        """Return the latest real aggregate-presence result, if one exists."""

        return self._latest_presence

    def reset(self) -> None:
        """重置限频器和全部跟踪历史，适用于重连或新任务。

        Reset the rate gate and all tracking history after reconnects/new tasks.
        """

        self.estimator.reset()
        self._next_due_ns = None
        self._latest = None
        self._latest_presence = None
        self.processed_frames = 0
        self.skipped_frames = 0

    def _claim_due_slot(self) -> bool:
        """Claim one current inference slot without queuing missed camera frames."""

        now = int(self.clock_ns())
        if self._next_due_ns is not None and now < self._next_due_ns:
            self.skipped_frames += 1
            return False
        if self._next_due_ns is None:
            self._next_due_ns = now + self.period_ns
        else:
            # Skip missed periods when overloaded; never build a catch-up backlog.
            periods = max(1, (now - self._next_due_ns) // self.period_ns + 1)
            self._next_due_ns += periods * self.period_ns
        return True

    def process_if_due(
        self,
        frame: np.ndarray,
        *,
        color_space: str,
        source_id: str = "cam_head",
        frame_index: int | None = None,
        capture_timestamp_ns: int | None = None,
    ) -> StageResult | None:
        """若当前周期到期则处理一帧，否则计数并返回 None。

        Process one frame if the current period is due; otherwise count the skip
        and return None. Callers must check for None before reading `.state`.
        """

        result = self.process_with_presence_if_due(
            frame,
            color_space=color_space,
            source_id=source_id,
            frame_index=frame_index,
            capture_timestamp_ns=capture_timestamp_ns,
        )
        return result.stage if result is not None else None

    def process_with_presence_if_due(
        self,
        frame: np.ndarray,
        *,
        color_space: str,
        source_id: str = "cam_head",
        frame_index: int | None = None,
        capture_timestamp_ns: int | None = None,
    ) -> FrameInferenceResult | None:
        """Return stage plus four-sample presence from one due YOLO inference.

        This remains output-only: it never performs pause, resume, or any other
        robot command.  The caller decides whether to run an advisory/shadow
        workflow with the result.
        """

        if not self._claim_due_slot():
            return None
        result = self.estimator.process_frame_with_presence(
            frame,
            color_space=color_space,
            source_id=source_id,
            frame_index=frame_index,
            capture_timestamp_ns=capture_timestamp_ns,
        )
        self._latest = result.stage
        self._latest_presence = result.presence
        self.processed_frames += 1
        return result
