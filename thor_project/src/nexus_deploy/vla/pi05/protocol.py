#!/usr/bin/env python
"""OpenPI-style msgpack protocol with safe NumPy array support."""

from __future__ import annotations

from typing import Any

import numpy as np

PROTOCOL_NAME = "nexus-policy-websocket"
PROTOCOL_VERSION = 2
LEGACY_PROTOCOLS = (
    {"name": "g2a-policy-websocket", "version": 2},
    {"name": "g2a-pi05-websocket", "version": 1},
)
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

_NDARRAY_MARKER = "__g2a_ndarray__"
_NP_SCALAR_MARKER = "__g2a_numpy_scalar__"
_ENCODED_IMAGE_MARKER = "__g2a_encoded_image__"
MAX_IMAGE_PIXELS = 4096 * 4096


class ProtocolError(ValueError):
    """Raised when a WebSocket payload violates the deployment protocol."""


def _require_msgpack():
    try:
        import msgpack
    except ImportError as error:
        raise ImportError(
            "The G2A deployment protocol requires msgpack. "
            "Install it in the lerobot conda environment with `python -m pip install msgpack`."
        ) from error
    return msgpack


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("O", "V", "c"):
            raise ProtocolError(f"Unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_MARKER: True,
            "data": array.tobytes(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    if isinstance(value, np.generic):
        if value.dtype.kind in ("O", "V", "c"):
            raise ProtocolError(f"Unsupported NumPy scalar dtype: {value.dtype}")
        return {
            _NP_SCALAR_MARKER: True,
            "data": value.item(),
            "dtype": value.dtype.str,
        }

    raise TypeError(f"Cannot serialize object of type {type(value).__name__}")


def _unpack_numpy(value: dict[str, Any]) -> Any:
    if value.get(_NDARRAY_MARKER) is True:
        try:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
            data = value["data"]
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError("Malformed NumPy array payload") from error

        if dtype.kind in ("O", "V", "c") or any(dim < 0 for dim in shape):
            raise ProtocolError(f"Unsupported NumPy array metadata: dtype={dtype}, shape={shape}")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected_bytes != len(data):
            raise ProtocolError(
                f"NumPy payload size mismatch: expected {expected_bytes} bytes, received {len(data)}"
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    if value.get(_NP_SCALAR_MARKER) is True:
        try:
            dtype = np.dtype(value["dtype"])
            return dtype.type(value["data"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError("Malformed NumPy scalar payload") from error

    return value


def pack_message(message: dict[str, Any], *, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> bytes:
    """Serialize one protocol message and enforce a bounded payload size."""
    if not isinstance(message, dict):
        raise ProtocolError("Protocol messages must be dictionaries")
    msgpack = _require_msgpack()
    payload = msgpack.packb(message, default=_pack_numpy, use_bin_type=True)
    if not payload or len(payload) > max_bytes:
        raise ProtocolError(f"Encoded message has invalid size: {len(payload)} bytes")
    return payload


def unpack_message(payload: bytes, *, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> dict[str, Any]:
    """Deserialize one bounded protocol message without pickle."""
    if not isinstance(payload, bytes):
        raise ProtocolError(f"Expected a binary WebSocket message, got {type(payload).__name__}")
    if not payload or len(payload) > max_bytes:
        raise ProtocolError(f"Received message has invalid size: {len(payload)} bytes")

    msgpack = _require_msgpack()
    try:
        message = msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False, strict_map_key=False)
    except ProtocolError:
        raise
    except Exception as error:
        raise ProtocolError("Could not decode msgpack payload") from error
    if not isinstance(message, dict):
        raise ProtocolError("Decoded protocol message must be a dictionary")
    return message


def _decode_encoded_image(value: dict[str, Any], camera_key: str) -> np.ndarray:
    """Decode one bounded robot-side JPEG/raw camera payload into HWC RGB."""
    try:
        encoding = str(value["encoding"]).lower()
        color_format = str(value["color_format"]).lower()
        width = int(value["width"])
        height = int(value["height"])
        data = value["data"]
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(f"Camera {camera_key!r} has malformed encoded-image metadata") from error
    if not isinstance(data, bytes):
        raise ProtocolError(f"Camera {camera_key!r} encoded data must be bytes")
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ProtocolError(f"Camera {camera_key!r} has invalid dimensions: {width}x{height}")

    if encoding == "raw":
        if len(data) != width * height * 3:
            raise ProtocolError(
                f"Camera {camera_key!r} raw byte count mismatch: "
                f"expected={width * height * 3} actual={len(data)}"
            )
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if color_format == "rgb":
            return image.copy()
        if color_format == "bgr":
            return np.ascontiguousarray(image[..., ::-1])
        raise ProtocolError(f"Camera {camera_key!r} has unsupported raw color format: {color_format!r}")

    if encoding == "jpeg":
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Server needs OpenCV to decode robot JPEG camera payloads") from error
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ProtocolError(f"Camera {camera_key!r} contains an invalid JPEG")
        if bgr.shape[:2] != (height, width):
            raise ProtocolError(
                f"Camera {camera_key!r} JPEG dimensions differ from metadata: "
                f"decoded={bgr.shape[1]}x{bgr.shape[0]} metadata={width}x{height}"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ProtocolError(f"Camera {camera_key!r} has unsupported encoding: {encoding!r}")


def validate_observation(
    observation: dict[str, Any],
    *,
    state_dim: int,
    camera_keys: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Validate and canonicalize an inference observation."""
    if not isinstance(observation, dict):
        raise ProtocolError("`observation` must be a dictionary")

    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (state_dim,) or not np.isfinite(state).all():
        raise ProtocolError(f"`state` must contain {state_dim} finite float values, got {state.shape}")

    images = observation.get("images")
    if not isinstance(images, dict):
        raise ProtocolError("`observation.images` must be a dictionary")
    missing = [key for key in camera_keys if key not in images]
    extra = [key for key in images if key not in camera_keys]
    if missing or extra:
        raise ProtocolError(f"Invalid camera keys; missing={missing}, extra={extra}")

    canonical_images: dict[str, np.ndarray] = {}
    for key in camera_keys:
        image_value = images[key]
        if isinstance(image_value, dict) and image_value.get(_ENCODED_IMAGE_MARKER) is True:
            image = _decode_encoded_image(image_value, key)
        else:
            image = np.asarray(image_value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ProtocolError(
                f"Camera {key!r} must be an HWC uint8 RGB image, got {image.shape}/{image.dtype}"
            )
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ProtocolError(f"Camera {key!r} has an empty image")
        canonical_images[key] = np.ascontiguousarray(image)

    task = observation.get("task", "")
    if not isinstance(task, str):
        raise ProtocolError("`observation.task` must be a string")
    if len(task) > 4096:
        raise ProtocolError("`observation.task` is too long")

    return state, canonical_images, task
