"""Legacy FlowPRO control mailboxes adapted for the Nexus runner.

These methods are activated only after the Nexus service launcher is wired.
They retain the source lease/health state machine; queue dispatch is supplied
by QueuedRollbackMotionBridge.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ...constants import ACTION_DIM
from .dispatch_ledger import QueuedRollbackMotionBridge
from .rollback import RollbackExecutor

logger = logging.getLogger(__name__)
FLOWPRO_ROLLBACK_HEALTH_MAILBOX_TIMEOUT_S = 0.2


class FlowPROControlMixin:
    def _send_nonrollback_action_if_allowed(self, action: np.ndarray, **_kwargs: Any) -> np.ndarray | None:
        with self._flowpro_rollback_lock:
            if self._flowpro_rollback_lease_id is not None:
                return None
            return self._send_action(action, hold=True)

    def _send_flowpro_rollback_action(self, target: np.ndarray) -> np.ndarray:
        # Keep rollback dispatch serialized with the runner's normal command
        # path and with the command/feedback snapshot taken while arming.
        with self._control_lock:
            return self._flowpro_motion_bridge.send_command(target)

    def _send_flowpro_rollback_hold(self, target: np.ndarray) -> np.ndarray:
        """Keep the sole GDK writer in measured/anchor HOLD until acknowledged."""
        deadline = time.monotonic() + 2.0
        while not self.shutdown_event.is_set() and time.monotonic() < deadline:
            self._send_action(target, hold=True)
            receipt = self.adapter.dispatch_receipt()
            if receipt is None:
                time.sleep(1.0 / self.servo_fps)
                continue
            if receipt.get("physical_accepted") is not True:
                raise RuntimeError("Rollback HOLD was not accepted by GDK")
            sent = np.asarray(receipt["action"], dtype=np.float32)
            if not np.allclose(sent, target, rtol=0.0, atol=1e-5):
                raise RuntimeError("Rollback HOLD differs from its measured/anchor target")
            return sent.copy()
        raise TimeoutError("Rollback checked HOLD was not acknowledged")

    def configure_flowpro_rollback(self) -> None:
        """Create a rollback consumer owned by this runner's sole action thread."""


        with self._flowpro_rollback_lock:
            if self._flowpro_rollback_executor is not None:
                raise RuntimeError("FlowPRO rollback executor is already configured")
            self._flowpro_rollback_executor = RollbackExecutor(
                read_measured=self._read_measured_for_rollback,
                read_command_motion=self.adapter.command_motion_snapshot,
                # The exact target is checked again by adapter.send_action's
                # shared safety layer. Any modification is detected by the
                # executor from the returned adopted command and faults.
                safety_project=lambda _measured, target: target,
                send_command=self._send_flowpro_rollback_action,
                send_hold_command=self._send_flowpro_rollback_hold,
            )
            self._flowpro_motion_bridge = QueuedRollbackMotionBridge(
                self.adapter,
                self._flowpro_rollback_executor,
                require_physical=self.hardware_mode == "execute",
            )


    def flowpro_measured_state_snapshot(self, *, max_age_ms: float = 100.0) -> np.ndarray:
        """Return the action-thread's newest measured state without another GDK caller."""

        if max_age_ms <= 0:
            raise ValueError("FlowPRO measured snapshot max age must be positive")
        evidence = getattr(self.adapter, "last_delivery_evidence", None)
        if evidence is None:
            raise RuntimeError("No servo delivery evidence is available for rollback planning")
        age_ms = max(0.0, (time.time_ns() - int(evidence.measured_timestamp_ns)) / 1e6)
        if age_ms > max_age_ms:
            raise RuntimeError(
                f"Rollback measured state is stale: age={age_ms:.1f}ms limit={max_age_ms:.1f}ms"
            )
        measured = np.asarray(evidence.measured_state, dtype=np.float32)
        if measured.shape != (ACTION_DIM,) or not np.isfinite(measured).all():
            raise RuntimeError("Rollback measured state snapshot is not a finite 24D vector")
        return measured.copy()


    def flowpro_rollback_start_snapshot(self, *, max_age_ms: float = 100.0) -> np.ndarray:
        """Return the exact fresh command reference used by the final limiter.

        Physical start consistency is still checked against measured feedback
        by ``RollbackExecutor.arm``. Planning from the last GDK-accepted
        command makes the first preplanned velocity/acceleration step use the
        same reference as the adapter's stateful validator.
        """

        if max_age_ms <= 0:
            raise ValueError("FlowPRO rollback start max age must be positive")
        evidence = getattr(self.adapter, "last_delivery_evidence", None)
        if evidence is None:
            raise RuntimeError("No servo delivery evidence is available for rollback planning")
        age_ms = max(0.0, (time.time_ns() - int(evidence.measured_timestamp_ns)) / 1e6)
        if age_ms > max_age_ms:
            raise RuntimeError(
                f"Rollback command reference is stale: age={age_ms:.1f}ms limit={max_age_ms:.1f}ms"
            )
        if not bool(evidence.command_accepted):
            raise RuntimeError("Latest rollback command reference was not accepted by GDK")
        adopted = np.asarray(evidence.adopted_action, dtype=np.float32)
        if adopted.shape != (ACTION_DIM,) or not np.isfinite(adopted).all():
            raise RuntimeError("Rollback command reference is not a finite 24D vector")
        return adopted.copy()


    def flowpro_fixed_head_waist_snapshot(self) -> np.ndarray:
        """Return the exact head/waist command frozen at 16D client startup."""

        if self._held_head_waist_values is None:
            raise RuntimeError("This Policy profile does not freeze head/waist control")
        value = np.asarray(self._held_head_waist_values, dtype=np.float32)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise RuntimeError("Frozen head/waist command is not a finite 8D vector")
        return value.copy()


    def configure_flowpro_arbitration_monitor(self, monitor: Any) -> None:
        """Share the DDS ownership monitor with the sole GDK writer.

        The monitor remains observational: a Policy or rollback ACK is emitted
        only after the adapter reports that the corresponding physical command
        was accepted. Human/vendor authority continues to come from DDS.
        """

        required = {
            "observe_policy_lease",
            "observe_policy_ack",
            "observe_rollback_ack",
        }
        if monitor is None or any(not hasattr(monitor, name) for name in required):
            raise TypeError("FlowPRO arbitration monitor does not implement the ACK contract")
        if self._flowpro_arbitration_monitor not in {None, monitor}:
            raise RuntimeError("Robot client already has a different FlowPRO arbitration monitor")
        self._flowpro_arbitration_monitor = monitor


    def enter_flowpro_hold(self, *, timeout_s: float = 2.0) -> None:
        """Clear Policy execution and wait for the action-thread HOLD ACK."""

        if timeout_s <= 0:
            raise ValueError("FlowPRO HOLD timeout must be positive")
        self._pause_requested_event.set()
        if not self._pause_active_event.wait(timeout_s):
            raise TimeoutError("Robot client did not enter measured HOLD for FlowPRO")


    def begin_flowpro_human_control(self, *, timeout_s: float = 2.0) -> None:
        """Suspend local HOLD streaming for the duration of vendor Human control."""

        with self._flowpro_human_epoch_lock:
            self._flowpro_human_epoch += 1
            self._flowpro_hold_reanchor_request_epoch = None
            self._flowpro_hold_reanchor_ack_epoch = None
            self._flowpro_hold_reanchor_requested_event.clear()
            self._flowpro_hold_reanchor_ack_event.clear()
            self._flowpro_hold_reanchor_error = None
            # Publish this latch before waiting for pause ACK so the action
            # thread cannot emit an old HOLD target after takeover.
            self._flowpro_human_control_active_event.set()
        self.enter_flowpro_hold(timeout_s=timeout_s)


    def prepare_flowpro_human_handback(self, *, timeout_s: float = 2.0) -> bool:
        """Re-anchor HOLD to measured state on the action thread before handback.

        The caller invokes this only after vendor retarget has stopped, all
        Human inputs are released, and the robot is settled.  Returning false
        keeps the logical supervisor in HUMAN and therefore cannot grant Policy.
        """

        if timeout_s <= 0:
            raise ValueError("FlowPRO Human handback re-anchor timeout must be positive")
        with self._flowpro_human_epoch_lock:
            if not self._flowpro_human_control_active_event.is_set():
                return True
            request_epoch = self._flowpro_human_epoch
            self._flowpro_hold_reanchor_request_epoch = request_epoch
            self._flowpro_hold_reanchor_ack_epoch = None
            self._flowpro_hold_reanchor_error = None
            self._flowpro_hold_reanchor_ack_event.clear()
            self._flowpro_hold_reanchor_requested_event.set()
        if not self._wait_for_control_event(self._flowpro_hold_reanchor_ack_event, timeout_s):
            with self._flowpro_human_epoch_lock:
                if self._flowpro_hold_reanchor_request_epoch == request_epoch:
                    self._flowpro_hold_reanchor_request_epoch = None
                    self._flowpro_hold_reanchor_requested_event.clear()
            logger.error("FLOWPRO HUMAN HANDBACK RE-ANCHOR timed out; remaining in HUMAN")
            return False
        with self._flowpro_human_epoch_lock:
            ack_matches = self._flowpro_hold_reanchor_ack_epoch == request_epoch
            error = self._flowpro_hold_reanchor_error if ack_matches else "stale handback ACK epoch"
            completed = ack_matches and not self._flowpro_human_control_active_event.is_set()
        if error is not None:
            logger.error(
                "FLOWPRO HUMAN HANDBACK RE-ANCHOR failed; remaining in HUMAN: %s",
                error,
            )
            return False
        return completed


    def revoke_flowpro_human_handback_from_transport(self) -> None:
        """Revoke an in-flight Human handback without waiting for DDS audit I/O."""

        with self._flowpro_human_epoch_lock:
            self._flowpro_human_epoch += 1
            self._flowpro_hold_reanchor_request_epoch = None
            self._flowpro_hold_reanchor_ack_epoch = None
            self._flowpro_hold_reanchor_requested_event.clear()
            self._flowpro_hold_reanchor_ack_event.clear()
            self._flowpro_hold_reanchor_error = None
            self._flowpro_human_control_active_event.set()
        # This is revoke-only: the action thread may enter/sustain HOLD, but no
        # Policy or rollback lease can be granted from the subscriber thread.
        self._pause_requested_event.set()


    def _execute_flowpro_human_hold_reanchor(self) -> None:
        """Action-thread implementation of the measured HOLD handback barrier."""

        with self._flowpro_human_epoch_lock:
            request_epoch = self._flowpro_hold_reanchor_request_epoch
            if (
                request_epoch is None
                or request_epoch != self._flowpro_human_epoch
                or not self._flowpro_hold_reanchor_requested_event.is_set()
                or not self._flowpro_human_control_active_event.is_set()
            ):
                return
        error_message: str | None = None
        sent_value: np.ndarray | None = None
        try:
            measured = self._reset_to_measured_hold()
            with self._flowpro_rollback_lock:
                if self._flowpro_rollback_lease_id is not None:
                    raise RuntimeError("rollback lease prevented the Human handback HOLD")
                sent = self._send_flowpro_rollback_hold(measured)
            sent_value = np.asarray(sent, dtype=np.float32)
            self._last_action = sent_value
            self._servo_ticks += 1
        except Exception as error:
            error_message = str(error)
            logger.exception("FLOWPRO HUMAN HANDBACK RE-ANCHOR action-thread failure")
        finally:
            with self._flowpro_human_epoch_lock:
                # A new Human input or timeout may supersede this request while
                # the GDK call is in flight. Never let the old completion clear
                # the newer Human-control latch or satisfy its ACK waiter.
                if (
                    self._flowpro_human_epoch == request_epoch
                    and self._flowpro_hold_reanchor_request_epoch == request_epoch
                    and self._flowpro_hold_reanchor_requested_event.is_set()
                ):
                    self._flowpro_hold_reanchor_error = error_message
                    self._flowpro_hold_reanchor_ack_epoch = request_epoch
                    self._flowpro_hold_reanchor_request_epoch = None
                    self._flowpro_hold_reanchor_requested_event.clear()
                    if error_message is None:
                        self._flowpro_human_control_active_event.clear()
                    self._flowpro_hold_reanchor_ack_event.set()
                    if error_message is None and sent_value is not None:
                        logger.warning(
                            "FLOWPRO HUMAN HANDBACK RE-ANCHORED epoch=%d: "
                            "GDK HOLD now matches measured pose %s",
                            request_epoch,
                            np.array2string(sent_value, precision=5),
                        )


    def resume_flowpro_policy(
        self,
        *,
        timeout_s: float = 10.0,
        wait_for_ready: bool = True,
    ) -> None:
        """Request a fresh reset generation, optionally waiting for its plan ACK.

        VR/DDS callers must use ``wait_for_ready=False`` so deadman release and
        Human takeover frames can never queue behind model inference.
        """

        if timeout_s <= 0:
            raise ValueError("FlowPRO resume timeout must be positive")
        human_control = getattr(self, "_flowpro_human_control_active_event", None)
        if human_control is not None and human_control.is_set():
            raise RuntimeError("Policy cannot resume before Human measured-HOLD handback ACK")
        with self._flowpro_rollback_lock:
            if (
                self._flowpro_rollback_lease_id is not None
                or getattr(self, "_flowpro_rollback_prepared_lease_id", None) is not None
            ):
                raise RuntimeError("Policy cannot resume while a rollback lease is armed or prepared")
        self._resume_ready_event.clear()
        self._pause_requested_event.clear()
        if not wait_for_ready:
            return
        if not self._resume_ready_event.wait(timeout_s):
            self._pause_requested_event.set()
            raise TimeoutError("Robot client did not install a fresh FlowPRO Policy generation")


    def flowpro_camera_fresh(self, *, max_age_ms: float = 200.0) -> bool:
        """Read one synchronized camera set for the Human handback gate."""

        if max_age_ms <= 0:
            raise ValueError("FlowPRO camera max age must be positive")
        try:
            _images, timestamps = self.adapter.capture_images()
        except Exception:
            logger.exception("FlowPRO camera freshness check failed")
            return False
        if len(timestamps) != 3:
            return False
        offset_ns = self.adapter.camera_wall_to_vendor_offset_ns
        if offset_ns is None:
            return False
        timestamps = {key: int(stamp) + offset_ns for key, stamp in timestamps.items()}
        now_ns = time.time_ns()
        ages_ns = [now_ns - int(timestamp) for timestamp in timestamps.values()]
        max_age_ns = int(max_age_ms * 1_000_000)
        max_skew_ns = int(self.adapter.config.camera_sync_tolerance_ms * 1_000_000)
        return (
            all(0 <= age_ns <= max_age_ns for age_ns in ages_ns)
            and max(timestamps.values()) - min(timestamps.values()) <= max_skew_ns
        )


    def flowpro_safety_state(self, *, max_age_ms: float = 100.0) -> dict[str, bool]:
        """Expose fresh calibrated collision-guard evidence without extra GDK calls."""

        telemetry = self.adapter.latest_control_telemetry
        if max_age_ms <= 0 or telemetry is None:
            return {"emergency_stop_active": False, "collision_detected": True}
        safety = telemetry.get("safety", {})
        age_ms = max(0.0, (time.time() - float(telemetry.get("captured_at", 0.0))) * 1000)
        trustworthy = bool(
            age_ms <= max_age_ms
            and safety.get("enabled") is True
            and safety.get("enforcement_enabled") is True
            and safety.get("decision_available") is True
        )
        return {
            # Arm estop is independently supplied by WholeBodyStatus DDS.
            "emergency_stop_active": False,
            "collision_detected": bool(not trustworthy or safety.get("collision_detected") is True),
        }


    def arm_flowpro_rollback(
        self,
        plan: Any,
        *,
        lease_id: str,
        timeout_s: float = 2.0,
        initial_deadman_pressed: bool = False,
        initial_health: Any | None = None,
    ) -> None:
        """Compatibility wrapper that prepares and immediately publishes a lease.

        Production FlowPRO uses ``prepare_*`` followed by a fresh-frame
        ``commit_prepared_flowpro_rollback``.  Keeping this wrapper preserves
        non-DDS/offline callers without weakening the production path.
        """

        self.prepare_flowpro_rollback(plan, lease_id=lease_id, timeout_s=timeout_s)
        with self._flowpro_rollback_lock:
            self._commit_prepared_flowpro_rollback_locked(
                lease_id=lease_id,
                deadman_pressed=initial_deadman_pressed,
                health=initial_health,
                require_stamped_health=False,
            )


    def prepare_flowpro_rollback(
        self,
        plan: Any,
        *,
        lease_id: str,
        timeout_s: float = 2.0,
    ) -> None:
        """Build an armed path while keeping it invisible to the action thread."""

        self.enter_flowpro_hold(timeout_s=timeout_s)
        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            if executor is None:
                raise RuntimeError("FlowPRO rollback executor is not configured")
            if self._flowpro_rollback_lease_id is not None:
                raise RuntimeError("A rollback execution lease is already active")
            if getattr(self, "_flowpro_rollback_prepared_lease_id", None) is not None:
                raise RuntimeError("A rollback execution lease is already prepared")
            # Serialize the last normal HOLD with a live command/measured
            # snapshot. The off-thread plan may have captured a slightly older
            # boundary; executor.arm inserts a safe connector to plan[0] when
            # needed instead of letting the first preplanned tick fail.
            with self._control_lock:
                measured = np.asarray(self.adapter.read_state(), dtype=np.float32)
                command_position, command_velocity = self.adapter.command_motion_snapshot()
                executor.arm(
                    plan,
                    lease_id=lease_id,
                    measured_state_24d=measured,
                    command_position_24d=command_position,
                    command_velocity_24d=command_velocity,
                )
            self._flowpro_rollback_prepared_lease_id = lease_id
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_health = None
            self._flowpro_rollback_health_updated_monotonic = None
            self._flowpro_rollback_last_result = None
            self._flowpro_rollback_cancel_reason = None


    def resume_flowpro_rollback(
        self,
        *,
        lease_id: str,
        initial_deadman_pressed: bool = False,
        initial_health: Any | None = None,
    ) -> dict[str, float | int] | None:
        """Compatibility wrapper for prepare-resume plus immediate publish."""

        resume_detail = self.prepare_resume_flowpro_rollback(lease_id=lease_id)
        with self._flowpro_rollback_lock:
            self._commit_prepared_flowpro_rollback_locked(
                lease_id=lease_id,
                deadman_pressed=initial_deadman_pressed,
                health=initial_health,
                require_stamped_health=False,
            )
        return resume_detail


    def prepare_resume_flowpro_rollback(
        self,
        *,
        lease_id: str,
    ) -> dict[str, float | int] | None:
        """Rebind a paused suffix without publishing a live action lease."""

        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            if executor is None:
                raise RuntimeError("FlowPRO rollback executor is not configured")
            if self._flowpro_rollback_lease_id is not None:
                raise RuntimeError("A rollback execution lease is already active")
            if getattr(self, "_flowpro_rollback_prepared_lease_id", None) is not None:
                raise RuntimeError("A rollback execution lease is already prepared")
            # Serialize the last normal HOLD send with command-boundary
            # capture and lease publication. The trajectory must start from
            # exactly the position/velocity held by the final limiter, while
            # measured feedback remains a separate tracking/safety input.
            with self._control_lock:
                measured = np.asarray(self.adapter.read_state(), dtype=np.float32)
                command_position, command_velocity = self.adapter.command_motion_snapshot()
                resume_detail = executor.rebind_paused(
                    lease_id=lease_id,
                    measured_state_24d=measured,
                    command_position_24d=command_position,
                    command_velocity_24d=command_velocity,
                )
            self._flowpro_rollback_prepared_lease_id = lease_id
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_health = None
            self._flowpro_rollback_health_updated_monotonic = None
            self._flowpro_rollback_last_result = None
            self._flowpro_rollback_cancel_reason = None
            return resume_detail


    def commit_prepared_flowpro_rollback(
        self,
        *,
        lease_id: str,
        deadman_pressed: bool,
        health: Any,
    ) -> None:
        """Atomically publish a prepared path from current stamped evidence.

        This method intentionally performs no GDK read/send, HOLD wait,
        connector construction, filesystem access, or inference.  Its caller
        therefore can use a newly accepted VR frame as the final deadman gate.
        """

        with self._flowpro_rollback_lock:
            self._commit_prepared_flowpro_rollback_locked(
                lease_id=lease_id,
                deadman_pressed=deadman_pressed,
                health=health,
                require_stamped_health=True,
            )


    def _commit_prepared_flowpro_rollback_locked(
        self,
        *,
        lease_id: str,
        deadman_pressed: bool,
        health: Any | None,
        require_stamped_health: bool,
    ) -> None:
        executor = self._flowpro_rollback_executor
        if executor is None:
            raise RuntimeError("FlowPRO rollback executor is not configured")
        if self._flowpro_rollback_lease_id is not None:
            raise RuntimeError("A rollback execution lease is already active")
        if getattr(self, "_flowpro_rollback_prepared_lease_id", None) != lease_id:
            raise ValueError("FlowPRO prepared rollback lease differs from the requested lease")
        if str(executor.state) != "ready" or getattr(executor, "lease_id", None) != lease_id:
            raise RuntimeError("FlowPRO prepared rollback executor is not READY for this lease")
        if health is None or not bool(getattr(health, "safe", False)):
            raise RuntimeError("FlowPRO prepared rollback requires current safe health")
        evaluated_ns = getattr(health, "evaluated_monotonic_ns", None)
        if evaluated_ns is None:
            if require_stamped_health:
                raise RuntimeError("FlowPRO rollback health has no monotonic evaluation timestamp")
            evaluated_monotonic = time.monotonic()
        else:
            evaluated_ns = int(evaluated_ns)
            age_ns = time.monotonic_ns() - evaluated_ns
            timeout_ns = int(FLOWPRO_ROLLBACK_HEALTH_MAILBOX_TIMEOUT_S * 1e9)
            if age_ns < 0 or age_ns > timeout_ns:
                raise RuntimeError(
                    "FlowPRO rollback health expired before prepared lease commit: "
                    f"age_ms={age_ns / 1e6:.1f} limit_ms={timeout_ns / 1e6:.1f}"
                )
            evaluated_monotonic = evaluated_ns / 1e9
        if not deadman_pressed:
            raise RuntimeError("FlowPRO prepared rollback commit requires held X")
        self._flowpro_rollback_lease_id = lease_id
        self._flowpro_rollback_prepared_lease_id = None
        self._flowpro_rollback_deadman = True
        self._flowpro_rollback_finish_to_anchor = False
        self._flowpro_rollback_health = health
        # Preserve the evidence's original evaluation time; never make old
        # health look fresh merely because prepare/read_state blocked.
        self._flowpro_rollback_health_updated_monotonic = evaluated_monotonic
        self._flowpro_rollback_last_result = None
        self._flowpro_rollback_cancel_reason = None


    def pause_flowpro_rollback(self) -> None:
        """Acknowledge an action-thread anchor HOLD and revoke its motion lease.

        The executor retains the one immutable plan and its monotonic cursor;
        another X press can bind a fresh lease directly to the untouched
        suffix without restoring a suspended/replanned trajectory.
        """

        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            result = self._flowpro_rollback_last_result
            if executor is None or str(executor.state) != "hold":
                raise RuntimeError("Rollback pause requires an executor in measured HOLD")
            if result is None or str(result.state) != "hold":
                raise RuntimeError("Rollback pause has not been acknowledged by the action thread")
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_health = None
            self._flowpro_rollback_health_updated_monotonic = None
            self._flowpro_rollback_lease_id = None
            self._flowpro_rollback_prepared_lease_id = None


    def finish_paused_flowpro_rollback(self) -> None:
        """Discard a paused plan after an anchor has been selected."""

        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            if executor is None:
                raise RuntimeError("FlowPRO rollback executor is not configured")
            executor.discard_paused()
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_health = None
            self._flowpro_rollback_health_updated_monotonic = None
            self._flowpro_rollback_lease_id = None
            self._flowpro_rollback_prepared_lease_id = None
            self._flowpro_rollback_last_result = None
            self._flowpro_rollback_cancel_reason = None


    def cancel_flowpro_rollback(self, *, reason: str) -> None:
        """Request fail-closed cancellation without calling GDK off-thread."""

        if not reason:
            raise ValueError("Rollback cancellation reason must be non-empty")
        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            if executor is None or str(executor.state) == "idle":
                return
            result = self._flowpro_rollback_last_result
            acknowledged_hold = bool(
                str(executor.state) == "hold" and result is not None and str(result.state) == "hold"
            )
            if acknowledged_hold or (
                self._flowpro_rollback_lease_id is None
                and str(executor.state) in {"ready", "hold", "cancelled", "fault"}
            ):
                # An acknowledged executor HOLD is already stationary.  Revoke
                # its still-live lease synchronously instead of queuing a cancel
                # behind the cached HOLD result, where the action thread would
                # otherwise never consume it.
                executor.discard_paused()
                self._flowpro_rollback_deadman = False
                self._flowpro_rollback_finish_to_anchor = False
                self._flowpro_rollback_health = None
                self._flowpro_rollback_health_updated_monotonic = None
                self._flowpro_rollback_lease_id = None
                self._flowpro_rollback_prepared_lease_id = None
                self._flowpro_rollback_last_result = None
                self._flowpro_rollback_cancel_reason = None
                return
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_cancel_reason = reason


    def abandon_flowpro_rollback_for_human(self) -> None:
        """Atomically revoke rollback after DDS confirms vendor Human authority."""

        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            if executor is not None and str(executor.state) != "idle":
                executor.abandon_for_human()
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_health = None
            self._flowpro_rollback_health_updated_monotonic = None
            self._flowpro_rollback_lease_id = None
            self._flowpro_rollback_prepared_lease_id = None
            self._flowpro_rollback_last_result = None
            self._flowpro_rollback_cancel_reason = None


    def update_flowpro_rollback_control(
        self,
        *,
        deadman_pressed: bool,
        finish_to_anchor_authorized: bool = False,
        health: Any,
        lease_id: str,
    ) -> None:
        """Update latest-only intent; the action thread performs the actual GDK call."""

        with self._flowpro_rollback_lock:
            if lease_id != self._flowpro_rollback_lease_id:
                raise ValueError("FlowPRO rollback control lease differs from the armed lease")
            self._flowpro_rollback_deadman = bool(deadman_pressed)
            self._flowpro_rollback_finish_to_anchor = bool(finish_to_anchor_authorized)
            self._flowpro_rollback_health = health
            evaluated_ns = getattr(health, "evaluated_monotonic_ns", None)
            self._flowpro_rollback_health_updated_monotonic = (
                time.monotonic() if evaluated_ns is None else int(evaluated_ns) / 1e9
            )


    def revoke_flowpro_rollback_deadman_from_transport(self) -> None:
        """Apply an X-up revocation without waiting for the DDS state queue.

        This entry point is deliberately one-way: a subscriber may revoke a
        live rollback grant, but it can never grant motion.  The serialized
        DDS coordinator still validates the frame, records the logical edge,
        and is the only path that may accept a later X-down.  Keeping this
        mailbox update under the rollback lock bounds physical X-up latency to
        the next 100 Hz action tick even when WAL or telemetry processing has
        temporarily backlogged the coordinator.
        """

        with self._flowpro_rollback_lock:
            if self._flowpro_rollback_lease_id is None:
                return
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = True


    def poll_or_update_flowpro_rollback_control(
        self,
        *,
        deadman_pressed: bool,
        finish_to_anchor_authorized: bool = False,
        health: Any,
        lease_id: str,
    ) -> Any | None:
        """Atomically observe a terminal ACK or update the active intent.

        Keeping the poll and update under one lock prevents a 20 Hz DDS tick
        from reading a non-terminal result just before the 100 Hz action
        thread completes and revokes the lease.
        """

        with self._flowpro_rollback_lock:
            result = self._flowpro_rollback_last_result
            if result is not None and str(result.state) in {
                "complete",
                "fault",
                "cancelled",
            }:
                return result
            if lease_id != self._flowpro_rollback_lease_id:
                raise ValueError("FlowPRO rollback control lease differs from the armed lease")
            self._flowpro_rollback_deadman = bool(deadman_pressed)
            self._flowpro_rollback_finish_to_anchor = bool(finish_to_anchor_authorized)
            self._flowpro_rollback_health = health
            evaluated_ns = getattr(health, "evaluated_monotonic_ns", None)
            self._flowpro_rollback_health_updated_monotonic = (
                time.monotonic() if evaluated_ns is None else int(evaluated_ns) / 1e9
            )
            return result


    def flowpro_rollback_result(self) -> Any | None:
        with self._flowpro_rollback_lock:
            return self._flowpro_rollback_last_result


    def disarm_flowpro_rollback(self) -> None:
        with self._flowpro_rollback_lock:
            executor = self._flowpro_rollback_executor
            if executor is None:
                raise RuntimeError("FlowPRO rollback executor is not configured")
            result = self._flowpro_rollback_last_result
            if str(executor.state) != "idle" or result is None or str(result.state) != "complete":
                raise RuntimeError("Rollback may be released only after action-thread completion")
            self._flowpro_rollback_deadman = False
            self._flowpro_rollback_finish_to_anchor = False
            self._flowpro_rollback_health = None
            self._flowpro_rollback_health_updated_monotonic = None
            self._flowpro_rollback_lease_id = None
            self._flowpro_rollback_prepared_lease_id = None
            self._flowpro_rollback_last_result = None
            self._flowpro_rollback_cancel_reason = None


    def _read_measured_for_rollback(self) -> np.ndarray:
        with self._control_lock:
            return np.asarray(self.adapter.read_state(), dtype=np.float32)


    def _step_flowpro_rollback(self) -> Any | None:
        rollback_lock = getattr(self, "_flowpro_rollback_lock", None)
        if rollback_lock is None:
            return None
        with rollback_lock:
            # A stationary HOLD must keep the real 100 Hz GDK heartbeat alive,
            # but the first HOLD result carries the exact selected anchor ACK
            # that the slower DDS coordinator still has to consume.  Keep that
            # ACK latched while returning each newly sent heartbeat to the
            # action loop for truthful command tracing.
            latched_hold_ack = None
            executor = self._flowpro_rollback_executor
            health = self._flowpro_rollback_health
            health_updated_monotonic = getattr(
                self,
                "_flowpro_rollback_health_updated_monotonic",
                None,
            )
            lease_id = self._flowpro_rollback_lease_id
            deadman = self._flowpro_rollback_deadman
            finish_to_anchor = getattr(self, "_flowpro_rollback_finish_to_anchor", False)
            cancel_reason = getattr(self, "_flowpro_rollback_cancel_reason", None)
            if executor is None or lease_id is None:
                return None
            if cancel_reason is not None:
                # Cancellation must take priority over republishing a cached
                # HOLD acknowledgement.  A takeover/fault can arrive after the
                # action thread reaches stationary HOLD but before DDS revokes
                # the lease; returning the cached HOLD first would leave the
                # cancellation pending and the old lease armed forever.
                result = executor.cancel(cancel_reason)
            elif (
                health_updated_monotonic is None
                or time.monotonic() - health_updated_monotonic > FLOWPRO_ROLLBACK_HEALTH_MAILBOX_TIMEOUT_S
            ):
                # The latest health value is not a perpetual grant.  A stalled
                # DDS coordinator must stop both held-X motion and autonomous
                # finish-to-anchor even when its final cached value was safe.
                logger.error(
                    "FLOWPRO ROLLBACK STOPPED: health mailbox exceeded %.0f ms; "
                    "sending HOLD and discarding this execution lease",
                    FLOWPRO_ROLLBACK_HEALTH_MAILBOX_TIMEOUT_S * 1000,
                )
                result = executor.cancel("rollback_health_mailbox_stale")
            elif str(executor.state) == "hold":
                if self._flowpro_rollback_last_result is None:
                    logger.error(
                        "FLOWPRO ROLLBACK STOPPED: executor HOLD has no action-thread acknowledgement"
                    )
                    result = executor.cancel("rollback_hold_ack_missing")
                elif health is None:
                    logger.error("FLOWPRO ROLLBACK STOPPED: anchor HOLD lost health evidence")
                    result = executor.cancel("rollback_hold_health_missing")
                else:
                    latched_hold_ack = self._flowpro_rollback_last_result
                    # Keep executor state, lease, physical heartbeat and
                    # published ACK in one critical section. DDS can therefore
                    # revoke L1 and rebind L2 only before or after one complete
                    # 100 Hz send.
                    result = executor.maintain_hold(
                        lease_id=lease_id,
                        health=health,
                    )
            elif health is None:
                # Production arm/resume supplies an initial health sample.  A
                # missing value therefore indicates a broken mailbox contract,
                # not permission to leave a live lease silently idle forever.
                logger.error("FLOWPRO ROLLBACK STOPPED: active lease has no health evidence; sending HOLD")
                result = executor.cancel("rollback_health_missing")
            else:
                # Hold rollback_lock across executor mutation and the GDK send.
                # Every competing DDS transition and every normal GDK send uses
                # the same rollback->control lock order, so the command boundary,
                # executor lease and result mailbox advance atomically.
                result = executor.step(
                    deadman_pressed=deadman,
                    finish_to_anchor_authorized=finish_to_anchor,
                    health=health,
                    lease_id=lease_id,
                )
            if str(result.state) in {"complete", "cancelled", "fault"}:
                # The DDS thread later acknowledges this published result and
                # clears the executor under this same lock.  An anchor HOLD is
                # deliberately not terminal here: it retains the main plan and
                # cursor for a later X press.
                if str(result.state) == "complete":
                    executor.disarm()
                else:
                    # Cancellation/fault has already sent measured HOLD on this
                    # action thread, so the frozen plan can now be discarded.
                    # A GDK/HOLD send exception still propagates and terminates
                    # the action thread; logical health/lease/path faults do not
                    # poison every later X attempt in the tablet episode.
                    executor.discard_paused()
                self._flowpro_rollback_deadman = False
                self._flowpro_rollback_finish_to_anchor = False
                self._flowpro_rollback_health = None
                self._flowpro_rollback_health_updated_monotonic = None
                self._flowpro_rollback_lease_id = None
                self._flowpro_rollback_prepared_lease_id = None
            self._flowpro_rollback_cancel_reason = None
            self._flowpro_rollback_last_result = (
                latched_hold_ack if latched_hold_ack is not None and str(result.state) == "hold" else result
            )
        receipt = self.adapter.dispatch_receipt()
        arbitration_monitor = getattr(self, "_flowpro_arbitration_monitor", None)
        if (
            arbitration_monitor is not None
            and str(result.state) in {"running", "hold", "complete"}
            and result.reason != "rollback_dispatch_pending"
            and receipt is not None
            and receipt.get("physical_accepted") is True
        ):
            arbitration_monitor.observe_rollback_ack(
                lease_id,
                timestamp_ns=int(receipt["measured_timestamp_ns"]),
            )
        if str(result.state) == "fault":
            logger.error(
                "FLOWPRO ROLLBACK FAULT HELD: reason=%s; plan/lease cleared, action thread remains available",
                result.reason,
            )
        return result


