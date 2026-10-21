"""Robot- and model-independent VLA deployment runtime."""

from .contracts import (
    ActionChunk,
    ImageSpec,
    InferenceRequest,
    Observation,
    PolicyCapabilities,
    TensorSpec,
)

__all__ = [
    "ActionChunk",
    "ImageSpec",
    "InferenceRequest",
    "Observation",
    "PolicyCapabilities",
    "TensorSpec",
]

__version__ = "1.0.0"
