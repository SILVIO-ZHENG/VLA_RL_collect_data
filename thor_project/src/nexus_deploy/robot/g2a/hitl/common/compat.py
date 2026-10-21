"""Small runtime compatibility shims shared by the standalone G2 bundle."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 on the audited G2 image.
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042 - required on the G2 Python 3.10 image
        """Python 3.10 equivalent of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
