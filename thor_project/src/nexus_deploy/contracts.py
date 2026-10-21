"""Stable, model- and robot-neutral protocol contracts.

The wire representation deliberately contains plain dictionaries and NumPy
arrays.  Adapters own all vendor/model-specific conversion at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

PROTOCOL_NAME = "nexus-policy-websocket"
PROTOCOL_VERSION = 3
LEGACY_PROTOCOL_NAMES = ("g2a-policy-websocket",)


class ContractError(ValueError):
    """A peer or adapter violated the public deployment contract."""


@dataclass(frozen=True)
class TensorSpec:
    """A semantic 1-D state/action space; names define ordering on the wire."""

    names: tuple[str, ...]
    dtype: str = "float32"
    space: str = "absolute"
    lower: tuple[float, ...] | None = None
    upper: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ContractError("TensorSpec names must be non-empty and unique")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ContractError(f"Invalid tensor dtype: {self.dtype!r}") from error
        if dtype.kind != "f":
            raise ContractError(f"State/action dtype must be floating point, got {self.dtype!r}")
        if self.space not in {"absolute", "delta"}:
            raise ContractError(f"Unsupported action space: {self.space}")
        for label, bounds in (("lower", self.lower), ("upper", self.upper)):
            if bounds is not None and len(bounds) != len(self.names):
                raise ContractError(f"{label} must have {len(self.names)} entries")
        if self.lower is not None and self.upper is not None:
            if any(low > high for low, high in zip(self.lower, self.upper, strict=True)):
                raise ContractError("TensorSpec lower bound exceeds upper bound")

    @property
    def dim(self) -> int:
        return len(self.names)

    def validate(self, value: Any, *, rows: bool = False) -> np.ndarray:
        array = np.asarray(value, dtype=self.dtype)
        expected = (None, self.dim) if rows else (self.dim,)
        valid_shape = array.ndim == 2 and array.shape[1] == self.dim if rows else array.shape == expected
        if not valid_shape or not np.isfinite(array).all():
            raise ContractError(f"Expected finite shape {expected}, got {array.shape}")
        if self.lower is not None and np.any(array < np.asarray(self.lower)):
            raise ContractError("Value is below declared lower bound")
        if self.upper is not None and np.any(array > np.asarray(self.upper)):
            raise ContractError("Value is above declared upper bound")
        return np.ascontiguousarray(array)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "names": list(self.names),
            "dim": self.dim,
            "dtype": self.dtype,
            "space": self.space,
        }
        if self.lower is not None:
            result["lower"] = list(self.lower)
        if self.upper is not None:
            result["upper"] = list(self.upper)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSpec:
        names = tuple(str(item) for item in value.get("names", ()))
        if "dim" in value and int(value["dim"]) != len(names):
            raise ContractError("Tensor spec dim does not match names")
        return cls(
            names=names,
            dtype=str(value.get("dtype", "float32")),
            space=str(value.get("space", "absolute")),
            lower=_optional_floats(value.get("lower")),
            upper=_optional_floats(value.get("upper")),
        )


@dataclass(frozen=True)
class ImageSpec:
    name: str
    required: bool = True
    encodings: tuple[str, ...] = ("jpeg", "raw")

    def __post_init__(self) -> None:
        if not self.name or not self.encodings or len(set(self.encodings)) != len(self.encodings):
            raise ContractError("ImageSpec needs a name and unique supported encodings")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "required": self.required, "encodings": list(self.encodings)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImageSpec:
        return cls(
            name=str(value["name"]),
            required=bool(value.get("required", True)),
            encodings=tuple(str(item) for item in value.get("encodings", ("jpeg", "raw"))),
        )


@dataclass(frozen=True)
class PolicyCapabilities:
    model_id: str
    state: TensorSpec
    action: TensorSpec
    images: tuple[ImageSpec, ...]
    chunk_size: int
    model_fps: float
    modes: tuple[str, ...] = ("sync", "async")
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id or self.chunk_size <= 0 or self.model_fps <= 0:
            raise ContractError("Invalid model_id, chunk_size, or model_fps")
        if not self.modes or not set(self.modes) <= {"sync", "async", "rtc"}:
            raise ContractError(f"Unsupported inference modes: {self.modes}")
        image_names = [image.name for image in self.images]
        if len(image_names) != len(set(image_names)):
            raise ContractError("Image names must be unique")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "metadata",
            "protocol": {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            "model_id": self.model_id,
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "images": [item.to_dict() for item in self.images],
            "chunk_size": self.chunk_size,
            "model_fps": self.model_fps,
            "inference_modes": list(self.modes),
            "capabilities": dict(self.extra),
        }

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> PolicyCapabilities:
        protocol = value.get("protocol", {})
        supported_names = (PROTOCOL_NAME, *LEGACY_PROTOCOL_NAMES)
        if not (
            isinstance(protocol, Mapping)
            and protocol.get("name") in supported_names
            and protocol.get("version") == PROTOCOL_VERSION
        ):
            raise ContractError(f"Unsupported protocol: {protocol}")
        return cls(
            model_id=str(value["model_id"]),
            state=TensorSpec.from_dict(value["state"]),
            action=TensorSpec.from_dict(value["action"]),
            images=tuple(ImageSpec.from_dict(item) for item in value.get("images", ())),
            chunk_size=int(value["chunk_size"]),
            model_fps=float(value["model_fps"]),
            modes=tuple(str(item) for item in value.get("inference_modes", ("sync",))),
            extra=dict(value.get("capabilities", {})),
        )


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    images: Mapping[str, Mapping[str, Any]]
    task: str
    timestamp_ns: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, value: Mapping[str, Any], capabilities: PolicyCapabilities) -> Observation:
        state = capabilities.state.validate(value.get("state"))
        images = value.get("images")
        if not isinstance(images, Mapping):
            raise ContractError("observation.images must be a mapping")
        missing = [item.name for item in capabilities.images if item.required and item.name not in images]
        if missing:
            raise ContractError(f"Missing required images: {missing}")
        specs = {item.name: item for item in capabilities.images}
        for name, payload in images.items():
            if name not in specs:
                continue
            if not isinstance(payload, Mapping):
                raise ContractError(f"Image {name!r} payload must be a mapping")
            encoding = str(payload.get("encoding", ""))
            if encoding not in specs[name].encodings:
                raise ContractError(
                    f"Image {name!r} encoding {encoding!r} is unsupported; "
                    f"available={specs[name].encodings}"
                )
            if int(payload.get("width", 0)) <= 0 or int(payload.get("height", 0)) <= 0:
                raise ContractError(f"Image {name!r} must have positive width and height")
            if not isinstance(payload.get("data"), bytes):
                raise ContractError(f"Image {name!r} data must be bytes")
        task = str(value.get("task", "")).strip()
        if not task:
            raise ContractError("observation.task must be non-empty")
        timestamp_ns = int(value.get("timestamp_ns", 0))
        return cls(state, images, task, timestamp_ns, dict(value.get("extra", {})))

    def to_wire(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "images": dict(self.images),
            "task": self.task,
            "timestamp_ns": self.timestamp_ns,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    session_id: str
    observation: Observation
    mode: str = "async"
    reset: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    chunk_id: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_wire(self, request_id: str) -> dict[str, Any]:
        return {
            "type": "inference_response",
            "status": "ok",
            "request_id": request_id,
            "chunk_id": self.chunk_id,
            "actions": self.actions,
            "extra": dict(self.extra),
        }


def parse_inference_request(
    value: Mapping[str, Any], capabilities: PolicyCapabilities
) -> InferenceRequest:
    if value.get("type") not in (None, "inference_request"):
        raise ContractError(f"Unsupported message type: {value.get('type')}")
    request_id = str(value.get("request_id", "")).strip()
    session_id = str(value.get("session_id", "")).strip()
    if not request_id or not session_id:
        raise ContractError("request_id and session_id are required")
    mode = str(value.get("mode", value.get("inference_mode", "async")))
    if mode not in capabilities.modes:
        raise ContractError(f"Mode {mode!r} is not supported; available={capabilities.modes}")
    observation = Observation.from_wire(value.get("observation", {}), capabilities)
    return InferenceRequest(
        request_id=request_id,
        session_id=session_id,
        observation=observation,
        mode=mode,
        reset=bool(value.get("reset", False)),
        context=dict(value.get("context", {})),
    )


def validate_action_chunk(chunk: ActionChunk, capabilities: PolicyCapabilities) -> ActionChunk:
    actions = capabilities.action.validate(chunk.actions, rows=True)
    if len(actions) > capabilities.chunk_size or len(actions) == 0:
        raise ContractError(
            f"Action chunk length must be in [1, {capabilities.chunk_size}], got {len(actions)}"
        )
    if not chunk.chunk_id:
        raise ContractError("chunk_id must be non-empty")
    return ActionChunk(actions, chunk.chunk_id, chunk.extra)


def _optional_floats(value: Any) -> tuple[float, ...] | None:
    return None if value is None else tuple(float(item) for item in value)
