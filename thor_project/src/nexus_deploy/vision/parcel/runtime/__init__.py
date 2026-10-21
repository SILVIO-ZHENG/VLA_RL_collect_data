"""G2 头部相机实时包裹阶段推理的公共 API。

Public API for realtime parcel-stage inference from the G2 head camera.

业务代码通常只需导入 ParcelStateEstimator 和 RateLimitedParcelStateService。
Application code normally needs only ParcelStateEstimator and
RateLimitedParcelStateService.
"""

__version__ = "1.0.0"

from .config import RuntimeConfig
from .estimator import (
    FrameInferenceReview,
    FrameReviewArtifacts,
    ParcelStateEstimator,
    SegmentationDetector,
    UltralyticsSegmentationDetector,
)
from .g2_head import (
    G2HeadCameraSource,
    G2HeadRuntime,
    GDKHeadCameraSource,
    SyntheticHeadCameraSource,
)
from .presence import (
    FrameInferenceResult,
    NO_PACKAGE as PRESENCE_NO_PACKAGE,
    PACKAGE_PRESENT,
    PRESENCE_STATES,
    ParcelPresenceGate,
    PresenceGateConfig,
    PresenceResult,
    UNKNOWN as PRESENCE_UNKNOWN,
)
# Shared policy is packaged in the same repository; works for standalone runners.
import sys as _sys
from pathlib import Path as _Path
_repo = str(_Path(__file__).resolve().parents[3])
if _repo not in _sys.path:
    _sys.path.append(_repo)

from .presence_control import (
    GateAction,
    GateDecision,
    PresenceTaskGate,
    PresenceTaskGateConfig,
    TaskState,
)
from .schema import SCHEMA_VERSION, STATE_TO_ID, StageResult
from .service import RateLimitedParcelStateService

__all__ = [
    "G2HeadCameraSource",
    "G2HeadRuntime",
    "GDKHeadCameraSource",
    "FrameInferenceReview",
    "FrameReviewArtifacts",
    "GateAction",
    "GateDecision",
    "FrameInferenceResult",
    "PACKAGE_PRESENT",
    "PRESENCE_NO_PACKAGE",
    "PRESENCE_STATES",
    "PRESENCE_UNKNOWN",
    "ParcelStateEstimator",
    "ParcelPresenceGate",
    "PresenceGateConfig",
    "PresenceResult",
    "PresenceTaskGate",
    "PresenceTaskGateConfig",
    "RateLimitedParcelStateService",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "STATE_TO_ID",
    "SegmentationDetector",
    "StageResult",
    "SyntheticHeadCameraSource",
    "TaskState",
    "UltralyticsSegmentationDetector",
    "__version__",
]
