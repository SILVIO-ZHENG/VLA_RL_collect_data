"""Small dependency-free MessagePack codec for the standalone robot client.

It implements only the MessagePack types used by this deployment protocol and
matches ``nexus_deploy.vla.pi05.protocol`` byte-for-byte.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .constants import DEFAULT_MAX_MESSAGE_BYTES

_NDARRAY_MARKER = "__g2a_ndarray__"
_NP_SCALAR_MARKER = "__g2a_numpy_scalar__"


class ProtocolError(ValueError):
    """Raised when a wire payload is malformed or unsupported."""


def _append_length(
    buf: list[bytes], size: int, fix_base: int, fix_limit: int, tags: tuple[int, int, int]
) -> None:
    if size < fix_limit:
        buf.append(bytes((fix_base | size,)))
    elif size < 2**8:
        buf.append(bytes((tags[0], size)))
    elif size < 2**16:
        buf.append(bytes((tags[1],)) + struct.pack(">H", size))
    elif size < 2**32:
        buf.append(bytes((tags[2],)) + struct.pack(">I", size))
    else:
        raise ProtocolError(f"Container is too large: {size}")


def _pack(value: Any, buf: list[bytes]) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("O", "V", "c"):
            raise ProtocolError(f"Unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        value = {
            _NDARRAY_MARKER: True,
            "data": array.tobytes(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    elif isinstance(value, np.generic):
        if value.dtype.kind in ("O", "V", "c"):
            raise ProtocolError(f"Unsupported NumPy scalar dtype: {value.dtype}")
        value = {
            _NP_SCALAR_MARKER: True,
            "data": value.item(),
            "dtype": value.dtype.str,
        }

    if value is None:
        buf.append(b"\xc0")
    elif value is False:
        buf.append(b"\xc2")
    elif value is True:
        buf.append(b"\xc3")
    elif isinstance(value, int):
        if 0 <= value < 2**7:
            buf.append(bytes((value,)))
        elif -32 <= value < 0:
            buf.append(struct.pack("b", value))
        elif 0 <= value < 2**8:
            buf.append(b"\xcc" + struct.pack(">B", value))
        elif 0 <= value < 2**16:
            buf.append(b"\xcd" + struct.pack(">H", value))
        elif 0 <= value < 2**32:
            buf.append(b"\xce" + struct.pack(">I", value))
        elif 0 <= value < 2**64:
            buf.append(b"\xcf" + struct.pack(">Q", value))
        elif value >= -(2**7):
            buf.append(b"\xd0" + struct.pack(">b", value))
        elif value >= -(2**15):
            buf.append(b"\xd1" + struct.pack(">h", value))
        elif value >= -(2**31):
            buf.append(b"\xd2" + struct.pack(">i", value))
        elif value >= -(2**63):
            buf.append(b"\xd3" + struct.pack(">q", value))
        else:
            raise ProtocolError(f"Integer is outside MessagePack range: {value}")
    elif isinstance(value, float):
        buf.append(b"\xcb" + struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        _append_length(buf, len(encoded), 0xA0, 32, (0xD9, 0xDA, 0xDB))
        buf.append(encoded)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if len(data) < 2**8:
            buf.append(b"\xc4" + struct.pack(">B", len(data)))
        elif len(data) < 2**16:
            buf.append(b"\xc5" + struct.pack(">H", len(data)))
        elif len(data) < 2**32:
            buf.append(b"\xc6" + struct.pack(">I", len(data)))
        else:
            raise ProtocolError(f"Binary value is too large: {len(data)}")
        buf.append(data)
    elif isinstance(value, (list, tuple)):
        size = len(value)
        if size < 16:
            buf.append(bytes((0x90 | size,)))
        elif size < 2**16:
            buf.append(b"\xdc" + struct.pack(">H", size))
        else:
            buf.append(b"\xdd" + struct.pack(">I", size))
        for item in value:
            _pack(item, buf)
    elif isinstance(value, dict):
        size = len(value)
        if size < 16:
            buf.append(bytes((0x80 | size,)))
        elif size < 2**16:
            buf.append(b"\xde" + struct.pack(">H", size))
        else:
            buf.append(b"\xdf" + struct.pack(">I", size))
        for key, item in value.items():
            _pack(key, buf)
            _pack(item, buf)
    else:
        raise ProtocolError(f"Cannot encode {type(value).__name__}")


def pack_message(message: dict[str, Any], max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> bytes:
    if not isinstance(message, dict):
        raise ProtocolError("Protocol messages must be dictionaries")
    chunks: list[bytes] = []
    _pack(message, chunks)
    payload = b"".join(chunks)
    if not payload or len(payload) > max_bytes:
        raise ProtocolError(f"Encoded message has invalid size: {len(payload)}")
    return payload


class _Decoder:
    def __init__(self, payload: bytes) -> None:
        self.data = memoryview(payload)
        self.index = 0

    def _read(self, size: int) -> bytes:
        end = self.index + size
        if size < 0 or end > len(self.data):
            raise ProtocolError("Truncated MessagePack payload")
        value = self.data[self.index : end].tobytes()
        self.index = end
        return value

    def _length(self, size: int, fmt: str) -> int:
        return int(struct.unpack(fmt, self._read(size))[0])

    def decode(self) -> Any:
        code = self._read(1)[0]
        if code <= 0x7F:
            return code
        if code >= 0xE0:
            return code - 256
        if 0xA0 <= code <= 0xBF:
            return self._read(code & 0x1F).decode("utf-8")
        if 0x90 <= code <= 0x9F:
            return [self.decode() for _ in range(code & 0x0F)]
        if 0x80 <= code <= 0x8F:
            return self._decode_map(code & 0x0F)

        if code == 0xC0:
            return None
        if code == 0xC2:
            return False
        if code == 0xC3:
            return True
        if code == 0xC4:
            return self._read(self._length(1, ">B"))
        if code == 0xC5:
            return self._read(self._length(2, ">H"))
        if code == 0xC6:
            return self._read(self._length(4, ">I"))
        if code == 0xCA:
            return struct.unpack(">f", self._read(4))[0]
        if code == 0xCB:
            return struct.unpack(">d", self._read(8))[0]
        if code == 0xCC:
            return self._length(1, ">B")
        if code == 0xCD:
            return self._length(2, ">H")
        if code == 0xCE:
            return self._length(4, ">I")
        if code == 0xCF:
            return self._length(8, ">Q")
        if code == 0xD0:
            return self._length(1, ">b")
        if code == 0xD1:
            return self._length(2, ">h")
        if code == 0xD2:
            return self._length(4, ">i")
        if code == 0xD3:
            return self._length(8, ">q")
        if code == 0xD9:
            return self._read(self._length(1, ">B")).decode("utf-8")
        if code == 0xDA:
            return self._read(self._length(2, ">H")).decode("utf-8")
        if code == 0xDB:
            return self._read(self._length(4, ">I")).decode("utf-8")
        if code == 0xDC:
            return [self.decode() for _ in range(self._length(2, ">H"))]
        if code == 0xDD:
            return [self.decode() for _ in range(self._length(4, ">I"))]
        if code == 0xDE:
            return self._decode_map(self._length(2, ">H"))
        if code == 0xDF:
            return self._decode_map(self._length(4, ">I"))
        raise ProtocolError(f"Unsupported MessagePack byte: 0x{code:02x}")

    def _decode_map(self, size: int) -> Any:
        value = {self.decode(): self.decode() for _ in range(size)}
        if value.get(_NDARRAY_MARKER) is True:
            try:
                dtype = np.dtype(value["dtype"])
                shape = tuple(int(dim) for dim in value["shape"])
                data = value["data"]
            except (KeyError, TypeError, ValueError) as error:
                raise ProtocolError("Malformed NumPy array payload") from error
            if dtype.kind in ("O", "V", "c") or any(dim < 0 for dim in shape):
                raise ProtocolError(f"Unsupported NumPy metadata: {dtype}/{shape}")
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if expected != len(data):
                raise ProtocolError(f"NumPy payload size mismatch: {expected} != {len(data)}")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        if value.get(_NP_SCALAR_MARKER) is True:
            try:
                return np.dtype(value["dtype"]).type(value["data"])
            except (KeyError, TypeError, ValueError) as error:
                raise ProtocolError("Malformed NumPy scalar payload") from error
        return value


def unpack_message(payload: bytes, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise ProtocolError(f"Expected binary WebSocket payload, got {type(payload).__name__}")
    if not payload or len(payload) > max_bytes:
        raise ProtocolError(f"Received message has invalid size: {len(payload)}")
    decoder = _Decoder(payload)
    message = decoder.decode()
    if decoder.index != len(decoder.data):
        raise ProtocolError("MessagePack payload contains trailing bytes")
    if not isinstance(message, dict):
        raise ProtocolError("Decoded protocol message must be a dictionary")
    return message
