"""Verified ownership gate for the vendor teleoperation end effectors.

The vendor VR process continues to receive the raw controller stream.  Merely
ignoring an index-trigger release in FlowPRO therefore cannot prevent that
process from opening a gripper.  This module uses the vendor's public
``ee.enable`` configuration and treats the published configuration snapshot as
the acknowledgement; publishing a request alone never grants Policy control.

No arm, gripper, or recorder command is issued here.  The only write performed
by the production transport is the vendor teleoperation configuration request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class VendorEETransport(Protocol):
    """Minimal transport contract, separated for deterministic tests."""

    def publish_enabled(self, enabled: bool, *, timestamp_ns: int) -> None: ...

    def read_current_enabled(self) -> bool | None: ...


@dataclass(frozen=True)
class VendorEEGateStatus:
    current_enabled: bool | None
    target_enabled: bool | None
    confirmed: bool
    pending: bool
    timed_out: bool


class G2VendorEEConfigTransport:
    """GDK DDS transport for the audited vendor teleoperation config topics."""

    SET_TOPIC = "/remote/set_config_robot"
    SET_TYPE = "rh_msgs_v4.SetConfigRobotRequest"
    GET_TOPIC = "/remote/get_config_list_robot"
    GET_TYPE = "rh_msgs_v4.GetConfigListRobotReply"
    SOURCE_APP = "teleop"
    PARAMETER = "ee.enable"

    def __init__(self, *, gdk: Any | None = None, qos: Any | None = None) -> None:
        if gdk is None:
            import agibot_gdk as gdk_module

            gdk = gdk_module
        from rh_msgs_v4_pb.config_robot_pb2 import (
            GetConfigListRobotReply,
            SetConfigRobotRequest,
        )

        self.gdk = gdk
        self.request_class = SetConfigRobotRequest
        self.reply_class = GetConfigListRobotReply
        self.qos = qos if qos is not None else gdk.QoS()
        self.publisher = gdk.Publisher(self.SET_TOPIC, self.SET_TYPE, self.qos)
        self.subscriber = gdk.Subscriber(self.GET_TOPIC, self.GET_TYPE, self.qos)
        self._frame_id = 0

    def publish_enabled(self, enabled: bool, *, timestamp_ns: int) -> None:
        request = self.request_class()
        # Audited /remote payload timestamps use milliseconds while PubMsgInfo
        # uses nanoseconds.
        request.head.timestamp = int(timestamp_ns // 1_000_000)
        request.head.operator_device = "FlowPRO"
        group = request.general_config_list.add()
        group.source_app = self.SOURCE_APP
        parameter = group.bool_params.add()
        parameter.name = self.PARAMETER
        parameter.unit = "bool"
        parameter.set_value = bool(enabled)

        info = self.gdk.PubMsgInfo()
        self._frame_id += 1
        info.frame_id = self._frame_id
        info.source_ts_ns = int(timestamp_ns)
        info.user_process_time = 0
        # Python GDK owns protobuf serialization for Publisher.publish().
        # Subscribers expose bytes, but publishers require the message object;
        # passing bytes makes the binding call SerializeToString() on bytes.
        accepted = self.publisher.publish(request, info)
        if not accepted:
            raise RuntimeError(
                f"Vendor teleop rejected {self.PARAMETER}={str(enabled).lower()} publication"
            )

    def read_current_enabled(self) -> bool | None:
        payload, _info = self.subscriber.subscribe(wait=False)
        if payload is None:
            return None
        if not isinstance(payload, bytes):
            raise RuntimeError("Vendor teleop configuration feedback is not a byte payload")
        reply = self.reply_class.FromString(payload)
        for group in reply.general_config_list:
            if str(group.source_app) != self.SOURCE_APP:
                continue
            for parameter in group.bool_params:
                if str(parameter.name) == self.PARAMETER:
                    return bool(parameter.current_value)
        raise RuntimeError(
            f"Vendor configuration feedback omitted {self.SOURCE_APP}.{self.PARAMETER}"
        )


class VendorEffectorGate:
    """Retry a vendor EE ownership request until feedback confirms it."""

    def __init__(
        self,
        transport: VendorEETransport,
        *,
        timeout_s: float = 1.0,
        retry_s: float = 0.2,
        clock_ns: Any = time.time_ns,
    ) -> None:
        if timeout_s <= 0 or retry_s <= 0 or retry_s > timeout_s:
            raise ValueError("Vendor EE timeout/retry must satisfy 0 < retry <= timeout")
        self.transport = transport
        self.timeout_ns = int(timeout_s * 1_000_000_000)
        self.retry_ns = int(retry_s * 1_000_000_000)
        self.clock_ns = clock_ns
        self.current_enabled: bool | None = None
        self.target_enabled: bool | None = None
        self._requested_at_ns: int | None = None
        self._last_publish_ns: int | None = None
        self._feedback_revision = 0
        self._confirmation_min_revision = 0
        # A subscriber may have accumulated many periodic configuration
        # snapshots while FlowPRO was idle.  Preparing a handoff therefore
        # drains that backlog incrementally before publishing the ownership
        # request.  It must never turn a large but healthy backlog into a
        # permanent button failure.
        self._draining_before_publish = False

    @classmethod
    def from_gdk(
        cls,
        *,
        gdk: Any | None = None,
        timeout_s: float = 1.0,
        retry_s: float = 0.2,
    ) -> VendorEffectorGate:
        return cls(
            G2VendorEEConfigTransport(gdk=gdk),
            timeout_s=timeout_s,
            retry_s=retry_s,
        )

    def request(self, enabled: bool, *, timestamp_ns: int | None = None) -> VendorEEGateStatus:
        now_ns = self.clock_ns() if timestamp_ns is None else int(timestamp_ns)
        if now_ns <= 0:
            raise ValueError("Vendor EE request timestamp must be positive")
        target = bool(enabled)
        new_transaction = self.target_enabled != target
        if new_transaction:
            self._start_handoff(target, now_ns=now_ns)
        elif self._draining_before_publish:
            self._advance_handoff_preparation(now_ns=now_ns)
        else:
            self._refresh_feedback()
        status = self.status(now_ns=now_ns)
        if (
            not new_transaction
            and not self._draining_before_publish
            and not status.confirmed
            and not status.timed_out
            and (
                self._last_publish_ns is None
                or now_ns - self._last_publish_ns >= self.retry_ns
            )
        ):
            self._publish(now_ns)
            status = self.status(now_ns=now_ns)
        return status

    def begin_handoff(
        self,
        enabled: bool,
        *,
        timestamp_ns: int | None = None,
    ) -> VendorEEGateStatus:
        """Start a new boundary even when the previous target was identical."""

        now_ns = self.clock_ns() if timestamp_ns is None else int(timestamp_ns)
        if now_ns <= 0:
            raise ValueError("Vendor EE request timestamp must be positive")
        self._start_handoff(bool(enabled), now_ns=now_ns)
        return self.status(now_ns=now_ns)

    def poll(self, *, timestamp_ns: int | None = None) -> VendorEEGateStatus:
        now_ns = self.clock_ns() if timestamp_ns is None else int(timestamp_ns)
        if self._draining_before_publish:
            self._advance_handoff_preparation(now_ns=now_ns)
        else:
            self._refresh_feedback()
        status = self.status(now_ns=now_ns)
        if (
            self.target_enabled is not None
            and not status.confirmed
            and self._last_publish_ns is not None
            and now_ns - self._last_publish_ns >= self.retry_ns
            and not status.timed_out
        ):
            self._publish(now_ns)
        return self.status(now_ns=now_ns)

    def refresh_passive(self, *, max_samples: int = 32) -> None:
        """Keep an idle DDS subscription near its live edge without writing.

        Vendor configuration feedback is periodic.  Consuming it while no
        handoff is active prevents an hours-long idle period from creating an
        equally large transaction backlog later.
        """

        if max_samples <= 0:
            raise ValueError("Vendor EE passive refresh max_samples must be positive")
        if self._draining_before_publish:
            self._advance_handoff_preparation(
                now_ns=self.clock_ns(),
                max_samples=max_samples,
            )
        else:
            self._refresh_feedback(max_samples=max_samples)

    def status(self, *, now_ns: int | None = None) -> VendorEEGateStatus:
        now_ns = self.clock_ns() if now_ns is None else int(now_ns)
        confirmed = bool(
            self.target_enabled is not None
            and self._last_publish_ns is not None
            and self.current_enabled == self.target_enabled
            and self._feedback_revision >= self._confirmation_min_revision
        )
        timed_out = bool(
            not confirmed
            and self.target_enabled is not None
            and self._requested_at_ns is not None
            and now_ns - self._requested_at_ns >= self.timeout_ns
        )
        return VendorEEGateStatus(
            current_enabled=self.current_enabled,
            target_enabled=self.target_enabled,
            confirmed=confirmed,
            pending=self.target_enabled is not None and not confirmed and not timed_out,
            timed_out=timed_out,
        )

    def wait_confirmed(self, enabled: bool, *, timeout_s: float | None = None) -> None:
        timeout_s = self.timeout_ns / 1_000_000_000 if timeout_s is None else float(timeout_s)
        if timeout_s <= 0:
            raise ValueError("Vendor EE wait timeout must be positive")
        self.request(enabled)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.poll()
            if status.confirmed:
                return
            if status.timed_out:
                break
            time.sleep(0.01)
        raise TimeoutError(f"Vendor ee.enable={str(enabled).lower()} was not confirmed")

    def _publish(self, now_ns: int) -> None:
        assert self.target_enabled is not None
        self.transport.publish_enabled(self.target_enabled, timestamp_ns=now_ns)
        self._last_publish_ns = now_ns
        logger.warning(
            "VENDOR EE CONFIG REQUEST ee.enable=%s",
            str(self.target_enabled).lower(),
        )

    def _start_handoff(self, target: bool, *, now_ns: int) -> None:
        self.target_enabled = bool(target)
        self._requested_at_ns = None
        self._last_publish_ns = None
        self._draining_before_publish = True
        self._advance_handoff_preparation(now_ns=now_ns)

    def _advance_handoff_preparation(self, *, now_ns: int, max_samples: int = 256) -> None:
        # Do not fail just because more than one bounded batch was queued.  The
        # coordinator remains in measured HOLD and subsequent ticks continue
        # draining.  Only the post-publication ACK timeout is safety relevant.
        if not self._draining_before_publish:
            return
        if not self._refresh_feedback(max_samples=max_samples):
            return
        self._draining_before_publish = False
        self._confirmation_min_revision = self._feedback_revision + 1
        self._requested_at_ns = now_ns
        self._publish(now_ns)

    def _refresh_feedback(self, *, max_samples: int = 8) -> bool:
        # Drain a bounded number of retained/queued snapshots and keep the
        # newest one. This method never blocks the urgent VR coordinator.
        for _ in range(max_samples):
            current = self.transport.read_current_enabled()
            if current is None:
                return True
            changed = self.current_enabled is None or self.current_enabled != bool(current)
            self.current_enabled = bool(current)
            self._feedback_revision += 1
            if changed:
                logger.info("VENDOR EE CONFIG FEEDBACK ee.enable=%s", str(current).lower())
        return False
