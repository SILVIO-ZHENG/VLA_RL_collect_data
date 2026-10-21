"""面向头部相机连续帧的同步 batch=1 阶段估计器。

Synchronous batch-1 parcel-stage estimator for a chronological head-camera stream.

数据流：RGB/BGR 校验与归一化 → YOLO 实例分割 → 单帧几何状态 → 历史帧确认。
Data flow: validate/normalize RGB or BGR → YOLO instance segmentation →
single-frame geometry decision → temporal confirmation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np
import yaml

from .config import RuntimeConfig
from .core_bridge import (
    Instance,
    RuleConfig,
    StateDecision,
    StateEngine,
    TemporalConfig,
    TemporalStateFilter,
    instances_from_result,
    load_final_model,
)
from .presence import (
    FrameInferenceResult,
    ParcelPresenceGate,
    PresenceGateConfig,
    PresenceResult,
)
from .schema import STATE_TO_ID, StageResult


@runtime_checkable
class SegmentationDetector(Protocol):
    """分割模型的最小协议，用于隔离真实 YOLO 与快速单元测试。

    Minimal segmentation protocol separating real YOLO inference from fast tests.
    """

    model_sha256: str

    def predict(self, frame_bgr: np.ndarray) -> Sequence[Instance]: ...

    def warmup(self, image_shape: tuple[int, int, int]) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FrameReviewArtifacts:
    """Diagnostic evidence from one already-completed inference.

    This type is intentionally for offline/manual review only. It exposes the
    normalized BGR image, instances, and raw stage decision that were produced
    by the same YOLO call as :class:`FrameInferenceResult`; it never serializes
    masks or issues robot commands.
    """

    image_bgr: np.ndarray
    instances: tuple[Instance, ...]
    raw_stage: StateDecision


@dataclass(frozen=True)
class FrameInferenceReview:
    """One public result together with its single-inference review evidence."""

    result: FrameInferenceResult
    artifacts: FrameReviewArtifacts


class UltralyticsSegmentationDetector:
    """固定为实时 batch=1 的 best52 Ultralytics 分割适配器。

    Ultralytics segmentation adapter for the validated best52 weight, fixed to
    realtime batch=1. Model loading also verifies SHA256 and the exact three-class map.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        # load_final_model 会在占用推理资源前校验文件、SHA256 和类别映射。
        # load_final_model validates the file, digest, and class map before use.
        self.model, self.model_sha256, self.weights = load_final_model(
            Path(config.weights),
            expected_sha256=config.expected_weight_sha256,
            skip_hash_check=False,
        )
        self._closed = False

    def _predict_result(self, frame_bgr: np.ndarray) -> Any:
        """运行一次真实 YOLO 调用并强制一帧对应一个结果。

        Execute one YOLO call and enforce the one-input/one-result invariant.
        """

        if self._closed:
            raise RuntimeError("detector is closed")
        # source 使用单元素列表，明确保证实时部署 batch=1。
        # A one-element source list explicitly fixes realtime inference to batch=1.
        results = list(
            self.model.predict(
                source=[frame_bgr],
                imgsz=self.config.imgsz,
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.config.device,
                # 必须保留原分辨率 mask；状态规则不能用检测框替代分割轮廓。
                # Preserve high-resolution masks; state rules reject box-only geometry.
                retina_masks=True,
                stream=True,
                save=False,
                verbose=False,
            )
        )
        if len(results) != 1:
            raise RuntimeError(f"model returned {len(results)} results for one frame")
        return results[0]

    def predict(self, frame_bgr: np.ndarray) -> Sequence[Instance]:
        """把 Ultralytics 输出转换为状态核心使用的 Instance 列表。

        Convert an Ultralytics result into instances consumed by the state core.
        """

        return instances_from_result(self._predict_result(frame_bgr))

    def warmup(self, image_shape: tuple[int, int, int]) -> None:
        """用黑帧预热模型，避免第一张真实图像承担初始化延迟。

        Warm the model with a black frame so initialization does not delay the first live frame.
        """

        frame = np.zeros(image_shape, dtype=np.uint8)
        self._predict_result(frame)

    def close(self) -> None:
        """释放对模型对象的引用；重复调用是安全的。

        Release the model reference; repeated calls are harmless at estimator level.
        """

        self._closed = True
        self.model = None


def _load_rules(path: Path) -> tuple[RuleConfig, TemporalConfig]:
    """读取并分别验证几何规则与时序规则。

    Load and independently validate geometry and temporal rule sections.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"state-rule configuration does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {resolved}")
    return RuleConfig.from_mapping(payload), TemporalConfig.from_mapping(
        payload.get("temporal")
    )


def _load_presence_gate_config(path: Path) -> PresenceGateConfig:
    """Load the independent aggregate-presence configuration from the same YAML."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"state-rule configuration does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {resolved}")
    return PresenceGateConfig.from_mapping(payload)


class ParcelStateEstimator:
    """把按时间顺序到达的图像流转换成稳定的单一包裹阶段。

    Convert a chronological image stream into one stable, global parcel-stage value.

    本对象持有活动包裹轨迹和历史确认状态，只能由一个推理线程处理一路相机。
    相机重连、帧编号回拨、画面/FOV 改变或新任务开始时必须调用 :meth:`reset`。
    This object owns mutable tracking/history state. Use one instance from one
    inference thread for one camera stream. Call :meth:`reset` after reconnects,
    frame-index rewinds, crop/FOV changes, or task-session boundaries.

    当前协议一次只输出一个活动包裹阶段。两臂同时操作不同包裹时会保守输出
    UNKNOWN，而不是猜测其中一个。The current contract emits one active-parcel stage;
    concurrent operations on different parcels conservatively become UNKNOWN.
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        detector: SegmentationDetector | None = None,
        presence_gate: ParcelPresenceGate | None = None,
    ):
        self.config = config or RuntimeConfig.defaults()
        self.rule_config, self.temporal_config = _load_rules(self.config.rules)

        # 运行时图像比例必须和 ROI 标定文件声明完全一致。
        # Runtime geometry must agree with the aspect ratio used to calibrate ROIs.
        configured_ratio = float(self.rule_config.expected_aspect_ratio)
        if abs(configured_ratio - self.config.expected_aspect_ratio) > 1e-9:
            raise ValueError(
                "runtime image aspect ratio does not match the calibrated ROI configuration"
            )
        # detector 参数用于测试或定制后端；正常部署会加载真实 best52。
        # The injected detector supports tests/custom backends; production loads best52.
        self.detector = detector or UltralyticsSegmentationDetector(self.config)
        self.model_sha256 = str(getattr(self.detector, "model_sha256", "test-detector"))
        # 两个对象都包含随帧变化的状态，不能跨线程或不相关视频共享。
        # Both objects are stateful and must not be shared across threads/unrelated streams.
        self._engine = StateEngine(self.rule_config)
        self._temporal = TemporalStateFilter(self.temporal_config)
        # The gate consumes the same per-frame instances as the stage engine.
        # It never invokes YOLO a second time and never sends robot commands.
        self.presence_gate = presence_gate or ParcelPresenceGate(
            _load_presence_gate_config(self.config.rules)
        )
        self._last_source_id: str | None = None
        self._last_frame_index: int | None = None
        self._last_presence: PresenceResult | None = None
        self._next_frame_index = 0
        self._closed = False
        self._process_lock = threading.Lock()

    def warmup(self) -> None:
        """按正式输入尺寸预热模型，但不推进包裹历史状态。

        Warm the detector at production resolution without advancing parcel history.
        """

        self._require_open()
        self.detector.warmup(
            (self.config.expected_height, self.config.expected_width, 3)
        )

    def reset(self) -> None:
        """清空活动包裹轨迹、候选阶段和历史确认状态。

        Clear active-package tracking, candidate stages, and temporal history.
        """

        with self._process_lock:
            self._engine.reset()
            self._temporal.reset()
            self.presence_gate.reset()
            self._last_source_id = None
            self._last_frame_index = None
            self._last_presence = None
            self._next_frame_index = 0

    @property
    def latest_presence(self) -> PresenceResult | None:
        """Return the last aggregate-presence result from this inference stream.

        This is a read-only convenience for a single-threaded output callback.
        Call :meth:`process_frame_with_presence` when the caller needs an atomic
        stage/presence pair for a particular frame.
        """

        return self._last_presence

    def close(self) -> None:
        """关闭模型资源；此方法可重复调用。

        Close detector resources; this method is idempotent.
        """

        if self._closed:
            return
        self.detector.close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("parcel-state estimator is closed")

    def _prepare_frame(self, frame: np.ndarray, color_space: str) -> np.ndarray:
        """校验并生成连续内存中的 640×400 BGR 模型输入。

        Validate a frame and return a contiguous 640×400 BGR model input.

        G2 公共接口输出 RGB，而 OpenCV 视频/Ultralytics 已验证路径使用 BGR。
        因此颜色空间必须由调用方明确声明，禁止自动猜测。
        G2 emits RGB while the validated OpenCV/Ultralytics path consumes BGR;
        callers must declare the color space explicitly instead of relying on guesses.
        """

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.dtype != np.uint8:
            raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
        if (
            frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] <= 0
            or frame.shape[1] <= 0
        ):
            raise ValueError(
                f"frame must have non-empty HWC shape with 3 channels, got {frame.shape}"
            )
        color = color_space.upper()
        if color not in {"RGB", "BGR"}:
            raise ValueError("color_space must be exactly 'RGB' or 'BGR'")

        # 固定 ROI 依赖完整的 1.60 视野。比例不符时拒绝，而不是静默裁剪。
        # Fixed ROIs depend on the complete 1.60 view; reject incompatible aspect ratios.
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        relative_error = (
            abs(aspect_ratio - self.config.expected_aspect_ratio)
            / self.config.expected_aspect_ratio
        )
        if relative_error > float(self.rule_config.aspect_ratio_tolerance):
            raise ValueError(
                f"frame aspect ratio {aspect_ratio:.4f} is incompatible with the "
                f"calibrated {self.config.expected_aspect_ratio:.4f} G2 head view; "
                "do not crop or stretch without recalibrating the ROIs"
            )
        prepared = frame
        target_size = (self.config.expected_width, self.config.expected_height)
        if (width, height) != target_size:
            if not self.config.resize_same_aspect:
                raise ValueError(
                    f"frame must be {target_size[0]}x{target_size[1]}, got {width}x{height}"
                )
            interpolation = (
                cv2.INTER_AREA if width >= target_size[0] else cv2.INTER_LINEAR
            )
            # 这里只对相同画幅做尺寸归一化，不生成 640×640 方图。
            # This normalizes same-aspect resolution; it never creates a 640×640 square.
            prepared = cv2.resize(prepared, target_size, interpolation=interpolation)
        if color == "RGB":
            # G2 RGB → 已验证模型路径 BGR，只转换这一次。
            # Convert G2 RGB to validated-path BGR exactly once.
            prepared = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)
        else:
            # 即使已经是 BGR 也复制，保证绝不原地修改调用方的 ndarray。
            # Copy BGR input so caller-owned arrays are never modified in place.
            prepared = prepared.copy()
        return np.ascontiguousarray(prepared, dtype=np.uint8)

    def _resolve_frame_index(self, frame_index: int | None) -> int:
        """解析源帧编号；缺省时按 30/10=3 的步长自动推进。

        Resolve a source-frame index, advancing by the nominal 30/10=3 step when omitted.
        """

        if frame_index is None:
            resolved = self._next_frame_index
        else:
            resolved = int(frame_index)
            if resolved < 0:
                raise ValueError("frame_index must be non-negative")
        self._next_frame_index = resolved + self.config.nominal_frame_step
        return resolved

    def process_frame_with_review(
        self,
        frame: np.ndarray,
        *,
        color_space: str,
        source_id: str = "cam_head",
        frame_index: int | None = None,
        capture_timestamp_ns: int | None = None,
    ) -> FrameInferenceReview:
        """处理一张全新的连续帧并返回一个稳定阶段结果。

        Process one new chronological frame and return result plus review evidence.

        `color_space` 必须是 RGB 或 BGR。`frame_index` 应单调递增；换 source_id
        或编号回拨会自动清空历史。每次调用都会运行一次模型，不负责 10 Hz 限频；
        高频相机循环应通过 RateLimitedParcelStateService 调用。
        `color_space` must be RGB or BGR. A source change or non-increasing frame index
        resets history. Each call runs inference once; use the rate-limited service in
        camera loops faster than 10 Hz. The returned artifacts are intended
        only for local/offline annotation and come from this same one inference.
        """

        self._require_open()
        if not source_id:
            raise ValueError("source_id cannot be empty")
        # 立即拒绝并发调用，防止不同帧交错污染跟踪和历史窗口。
        # Fail fast on concurrency so interleaved frames cannot corrupt temporal state.
        if not self._process_lock.acquire(blocking=False):
            raise RuntimeError("concurrent process_frame calls are not supported")
        try:
            started = time.perf_counter_ns()
            capture_ns = (
                time.time_ns()
                if capture_timestamp_ns is None
                else int(capture_timestamp_ns)
            )
            if capture_ns < 0:
                raise ValueError("capture_timestamp_ns must be non-negative")
            resolved_index = self._resolve_frame_index(frame_index)
            # 新视频/相机或帧编号回拨不能继承上一段任务状态。
            # A new source or frame rewind must never inherit the previous task history.
            if self._last_source_id is not None and (
                source_id != self._last_source_id
                or (
                    self._last_frame_index is not None
                    and resolved_index <= self._last_frame_index
                )
            ):
                self._engine.reset()
                self._temporal.reset()
                self.presence_gate.reset()

            # 1) 图像规范化 / Image normalization.
            frame_bgr = self._prepare_frame(frame, color_space)
            # 2) YOLO mask 推理 / YOLO mask inference.
            instances = self.detector.predict(frame_bgr)
            # The aggregate gate reads every detected package.  The stage engine
            # below still preserves the old single-active-package contract.
            presence = self.presence_gate.update(instances, frame_bgr.shape[:2])
            # 3) 当前帧几何判断 / Per-frame geometric state.
            raw = self._engine.analyze(
                instances,
                frame_bgr.shape[:2],
                source_id=source_id,
                frame_index=resolved_index,
            )
            # 4) 连续 3/4/5 帧确认 / 3/4/5-frame temporal confirmation.
            stable = self._temporal.update(raw)
            output_ns = time.time_ns()
            inference_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            end_to_end_ms = max(0.0, (output_ns - capture_ns) / 1_000_000.0)
            # status 区分“正在等待足够历史”与“几何证据确实歧义”。
            # status distinguishes warm-up from genuinely ambiguous geometry.
            if stable.state != "UNKNOWN":
                status = "ok"
            elif stable.candidate_state is not None:
                status = "warming_up"
            else:
                status = "unknown"

            self._last_source_id = source_id
            self._last_frame_index = resolved_index
            self._last_presence = presence
            stage = StageResult(
                source_id=source_id,
                frame_index=resolved_index,
                capture_timestamp_ns=capture_ns,
                output_timestamp_ns=output_ns,
                state=stable.state,
                state_id=STATE_TO_ID[stable.state],
                confidence=float(stable.confidence),
                changed=bool(stable.changed),
                # UNKNOWN 永远不可用于直接触发机器人动作。
                # UNKNOWN is never safe as a direct robot-action trigger.
                actionable=stable.state != "UNKNOWN",
                status=status,
                raw_state=raw.state,
                raw_confidence=float(raw.confidence),
                candidate_state=stable.candidate_state,
                candidate_frames=int(stable.candidate_frames),
                reason=stable.reason,
                raw_reason=raw.reason,
                active_package_index=raw.active_package_index,
                active_centroid=raw.active_centroid,
                inference_ms=float(inference_ms),
                end_to_end_ms=float(end_to_end_ms),
                model_sha256=self.model_sha256,
            )
            result = FrameInferenceResult(stage=stage, presence=presence)
            return FrameInferenceReview(
                result=result,
                artifacts=FrameReviewArtifacts(
                    # Keep review rendering isolated from detector/model input
                    # buffers. This extra copy is diagnostic-only and never
                    # happens in the normal realtime presence API.
                    image_bgr=frame_bgr.copy(),
                    instances=tuple(instances),
                    raw_stage=raw,
                ),
            )
        finally:
            self._process_lock.release()

    def process_frame_with_presence(
        self,
        frame: np.ndarray,
        *,
        color_space: str,
        source_id: str = "cam_head",
        frame_index: int | None = None,
        capture_timestamp_ns: int | None = None,
    ) -> FrameInferenceResult:
        """Return stage plus aggregate presence from exactly one YOLO inference.

        This is the normal realtime API. It intentionally discards review-only
        masks and raw geometry evidence so robot-facing callers keep the compact
        JSON-safe result contract.
        """

        return self.process_frame_with_review(
            frame,
            color_space=color_space,
            source_id=source_id,
            frame_index=frame_index,
            capture_timestamp_ns=capture_timestamp_ns,
        ).result

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        color_space: str,
        source_id: str = "cam_head",
        frame_index: int | None = None,
        capture_timestamp_ns: int | None = None,
    ) -> StageResult:
        """Return the original single-active-parcel stage output.

        This backward-compatible method still runs exactly one YOLO inference.
        Consumers that also need the aggregate four-sample presence gate should
        call :meth:`process_frame_with_presence` instead.
        """

        result = self.process_frame_with_presence(
            frame,
            color_space=color_space,
            source_id=source_id,
            frame_index=frame_index,
            capture_timestamp_ns=capture_timestamp_ns,
        )
        return result.stage

    def __enter__(self) -> "ParcelStateEstimator":
        """支持 `with ParcelStateEstimator(...) as estimator`。

        Support use as a context manager.
        """

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """离开上下文时确保释放模型引用。

        Ensure detector resources are released when leaving a context.
        """

        del exc_type, exc, traceback
        self.close()
