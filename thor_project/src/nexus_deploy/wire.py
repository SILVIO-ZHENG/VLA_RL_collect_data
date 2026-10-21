"""MessagePack wire codec compatible with the existing v2 ndarray markers."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

NDARRAY_MARKER = "__g2a_ndarray__"
NUMPY_SCALAR_MARKER = "__g2a_numpy_scalar__"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class WireError(ValueError):
    pass


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise WireError(f"Unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            NDARRAY_MARKER: True,
            "data": array.tobytes(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return {NUMPY_SCALAR_MARKER: True, "data": value.item(), "dtype": value.dtype.str}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: dict[str, Any]) -> Any:
    if value.get(NDARRAY_MARKER) is True:
        try:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            array = np.frombuffer(value["data"], dtype=dtype)
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise WireError("ndarray byte count does not match shape")
            return array.reshape(shape).copy()
        except (KeyError, TypeError, ValueError) as error:
            raise WireError(f"Malformed ndarray payload: {error}") from error
    if value.get(NUMPY_SCALAR_MARKER) is True:
        try:
            return np.asarray(value["data"], dtype=np.dtype(value["dtype"])).item()
        except (KeyError, TypeError, ValueError) as error:
            raise WireError(f"Malformed NumPy scalar payload: {error}") from error
    return value


def pack_message(value: dict[str, Any], max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    if not isinstance(value, dict):
        raise WireError("Top-level message must be a dictionary")
    payload = msgpack.packb(_encode(value), use_bin_type=True)
    if not payload or len(payload) > max_bytes:
        raise WireError(f"Encoded message size is invalid: {len(payload)}")
    return payload


def unpack_message(payload: bytes | str, max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    if isinstance(payload, str):
        raise WireError("Binary MessagePack frame required")
    if not payload or len(payload) > max_bytes:
        raise WireError(f"Received message size is invalid: {len(payload)}")
    try:
        value = msgpack.unpackb(payload, raw=False, object_hook=_decode, strict_map_key=False)
    except (msgpack.UnpackException, ValueError) as error:
        raise WireError(f"Malformed MessagePack: {error}") from error
    if not isinstance(value, dict):
        raise WireError("Top-level message must be a dictionary")
    return value
