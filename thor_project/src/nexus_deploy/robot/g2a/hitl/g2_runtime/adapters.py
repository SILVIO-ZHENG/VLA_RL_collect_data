"""Pure adapters for audited G2 payloads.

This module deliberately has no GDK or DDS imports.  The production subscriber
must extract primitive fields from vendor messages and pass them through these
helpers before emitting FlowPRO events.
"""

from __future__ import annotations

MILLISECONDS_TO_NANOSECONDS = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


class MixedUnitTimestampAdapter:
    """Normalize audited vendor timestamps to epoch nanoseconds.

    ``RobotFrameData.timestamp``, ``OperatorEventRequest.timestamp`` and
    ``ActionSrcState.timestamp`` are milliseconds.  Header stamps, accepted
    retarget stamps, vendor H5 and camera timestamp files are nanoseconds.
    Python GDK ``PubMsgInfo.source_ts_ns`` is intentionally not accepted because
    it was observed as zero on the target G2.
    """

    @staticmethod
    def payload_milliseconds(value: object, *, field: str) -> int:
        milliseconds = _positive_int(value, field=field)
        return milliseconds * MILLISECONDS_TO_NANOSECONDS

    @staticmethod
    def header_stamp(*, sec: object, nanosec: object, field: str) -> int:
        if isinstance(sec, bool) or not isinstance(sec, int) or sec < 0:
            raise ValueError(f"{field}.sec must be a non-negative integer")
        if isinstance(nanosec, bool) or not isinstance(nanosec, int):
            raise ValueError(f"{field}.nanosec must be an integer")
        if not 0 <= nanosec < NANOSECONDS_PER_SECOND:
            raise ValueError(f"{field}.nanosec must be in [0, {NANOSECONDS_PER_SECOND})")
        timestamp_ns = sec * NANOSECONDS_PER_SECOND + nanosec
        if timestamp_ns <= 0:
            raise ValueError(f"{field} must resolve to a positive timestamp")
        return timestamp_ns

    @staticmethod
    def nanoseconds(value: object, *, field: str) -> int:
        return _positive_int(value, field=field)
