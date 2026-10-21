"""实时阶段推理的默认路径与参数配置。

Configuration and default paths for realtime parcel-stage inference.

本模块只保存配置，不加载模型或访问相机，因此导入它不会占用 GPU/GDK 资源。
This module only defines settings. Importing it does not load the model or open GDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# 从当前文件定位部署目录，避免依赖调用命令时的工作目录。
# Resolve paths from this file so they do not depend on the caller's cwd.
DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "parcel_yolo26m_seg_final.pt"
DEFAULT_RULE_CONFIG = DEPLOYMENT_ROOT / "configs" / "g2_head.yaml"

# 这是已经通过 test 集和三段视频验证的 best52 权重摘要。
# This digest pins the exact best52 weight validated on the test set and videos.
FINAL_WEIGHT_SHA256 = "6ca0492ba45211e14e8f5d4f2eeac6bc16c29476f5f0646b4503b4b5c0d40c2c"


def _path_from_env(name: str, fallback: Path) -> Path:
    """读取可选路径环境变量；未设置时使用项目默认路径。

    Read an optional path environment variable, falling back to the project layout.
    """

    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


@dataclass(frozen=True)
class RuntimeConfig:
    """API 与 G2 命令行入口共同使用的不可变配置。

    Immutable validated settings shared by the Python API and G2 CLI.

    `imgsz=640` 交给 Ultralytics 做等比例 letterbox；它不表示把原图拉伸为
    640×640。`expected_width/height` 则对应已经标定 ROI 的 640×400 原始视图。
    `imgsz=640` controls Ultralytics letterboxing; it does not stretch the original
    image to a square. The expected width/height describe the calibrated 640×400 view.
    """

    # 模型、规则与设备 / Model, rule, and accelerator selection.
    weights: Path = DEFAULT_WEIGHTS
    rules: Path = DEFAULT_RULE_CONFIG
    device: str = "0"

    # YOLO 推理参数，与已验证视频测试保持一致。
    # YOLO inference parameters kept identical to the validated video run.
    imgsz: int = 640
    confidence: float = 0.25
    iou: float = 0.60

    # 相机以 30 FPS 采集，阶段检测只取最新帧按 10 Hz 运行。
    # The camera captures at 30 FPS while stage inference samples the newest frame at 10 Hz.
    detect_hz: float = 10.0
    camera_fps: float = 30.0

    # 固定 ROI 对应 640×400。仅允许同宽高比缩放，不允许静默裁剪或拉伸。
    # Fixed ROIs target 640×400. Only same-aspect resizing is allowed.
    expected_width: int = 640
    expected_height: int = 400
    resize_same_aspect: bool = True

    # 加载模型前必须匹配此摘要，避免误用其他轮次权重。
    # Model loading fails fast if the weight digest differs from this value.
    expected_weight_sha256: str = FINAL_WEIGHT_SHA256

    @classmethod
    def defaults(cls) -> "RuntimeConfig":
        """生成默认配置，同时允许环境变量覆盖文件位置。

        Build defaults while honoring explicit deployment path environment variables.
        """

        return cls(
            weights=_path_from_env("PARCEL_STATE_WEIGHTS", DEFAULT_WEIGHTS),
            rules=_path_from_env("PARCEL_STATE_RULES", DEFAULT_RULE_CONFIG),
        )

    def __post_init__(self) -> None:
        # frozen dataclass 仍需把用户传入的 str 规范成 Path。
        # A frozen dataclass uses object.__setattr__ to normalize string paths.
        object.__setattr__(self, "weights", Path(self.weights).expanduser())
        object.__setattr__(self, "rules", Path(self.rules).expanduser())
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")
        if self.detect_hz <= 0 or self.camera_fps <= 0:
            raise ValueError("detect_hz and camera_fps must be positive")
        if self.detect_hz > self.camera_fps:
            raise ValueError("detect_hz cannot exceed camera_fps")
        if self.expected_width <= 0 or self.expected_height <= 0:
            raise ValueError("expected image dimensions must be positive")
        if not self.expected_weight_sha256:
            raise ValueError("expected_weight_sha256 cannot be empty")

    @property
    def expected_aspect_ratio(self) -> float:
        """返回 ROI 标定所要求的原始画面宽高比。

        Return the source-image aspect ratio required by the calibrated ROIs.
        """

        return self.expected_width / self.expected_height

    @property
    def nominal_frame_step(self) -> int:
        """一次检测样本对应多少个原生相机帧。

        Return the native-camera frame increment represented by one detector sample.

        默认 30/10=3，使跟踪器的 frame gap 与离线视频抽帧保持同一语义。
        The default 30/10=3 preserves the offline tracker's frame-gap semantics.
        """

        return max(1, int(round(self.camera_fps / self.detect_hz)))
