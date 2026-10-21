"""Single-writer orchestration between G2 DDS subscribers and FlowPRO.

DDS topics are read concurrently, but every state-machine mutation is applied
on one coordinator thread.  The service is intentionally bounded and
fail-closed: queue overflow, decoder failure, stale evidence, or unavailable
camera/safety health can never grant Policy or rollback execution.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import count
from typing import Any

from .g2_dds import (
    AcceptedRetargetSample,
    ActionSourceSample,
    G2DDSRead,
    G2ReadOnlyDDSClient,
    G2RuntimeEvidence,
    VRTeleopSample,
)
from .supervisor import SupervisorMode, evaluate_human_handback

DEFAULT_RUNTIME_TOPIC_LABELS = (
    "vr_data",
    "vr_network",
    "operator_event",
    "action_src",
    "accepted_retarget",
    "motion_status",
    "whole_body_status",
    "public_fault",
    "body_joints",
    "left_effector",
    "right_effector",
)

# Operator edges and authority/fault evidence must not sit behind a burst of
# lower-priority telemetry after filesystem I/O.  Health data that has not yet
# been consumed remains fail-closed; prioritizing these topics can therefore
# improve responsiveness without weakening a safety gate.
_URGENT_RUNTIME_TOPICS = {
    "vr_data",
    "vr_network",
    "operator_event",
    "action_src",
    "accepted_retarget",
    "motion_status",
    "whole_body_status",
    "public_fault",
}

# These three streams jointly form the measured 24D health gate.  Keep them
# ahead of unrelated telemetry, but behind operator/authority/fault edges so a
# release or takeover can never be buried by a high-rate joint backlog.
_HEALTH_RUNTIME_TOPICS = {
    "body_joints",
    "left_effector",
    "right_effector",
}
_PRE_TICK_DRAIN_BUDGET = 512


class G2DDSRuntimeCoordinator:
    """Serialize decoded DDS evidence into the FlowPRO application."""

    def __init__(
        self,
        *,
        application: Any,
        evidence: G2RuntimeEvidence | None = None,
        cameras_fresh: Callable[[], bool] | None = None,
        safety_state: Callable[[], Mapping[str, bool]] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.application = application
        self.evidence = evidence or G2RuntimeEvidence()
        # Missing integrations must block handback and rollback rather than be
        # treated as healthy defaults.
        self.cameras_fresh = cameras_fresh or (lambda: False)
        self.safety_state = safety_state or (
            lambda: {"emergency_stop_active": True, "collision_detected": False}
        )
        self.clock_ns = clock_ns
        self.monotonic_clock_ns = monotonic_clock_ns
        self._human_hold_reanchored = False
        self._human_camera_verified_ns: int | None = None
        self._human_camera_barrier_vr_revision: int | None = None
        self._external_authority_active = False

    def consume(self, read: G2DDSRead) -> bool:
        accepted = self.evidence.observe(read)
        if not accepted:
            return False
        if read.label == "vr_data":
            sample = read.sample
            assert isinstance(sample, VRTeleopSample)
            # ``read`` may have waited in the serialized coordinator queue while
            # an X edge freezes/persists rollback history.  Judge whether this
            # historical VR frame arrived on a connected stream at its DDS
            # receive time, not at the later time at which we happen to apply
            # it.  Otherwise the coordinator's own bounded persistence pause
            # turns every queued X-down frame into a synthetic disconnect and
            # invalidates rollback before the background planner can finish.
            #
            # Live loss of the stream remains fail-closed: VRButtonParser uses
            # the same receive timestamp for its 200 ms clock, and periodic
            # ticks plus rollback_health evaluate freshness against wall time.
            connected = self.evidence.network_fresh(now_ns=read.receive_timestamp_ns)
            self.application.handle_vr_frame(
                replace(sample, connected=connected).raw_frame(),
                received_timestamp_ns=read.receive_timestamp_ns,
            )
        # Input handling can perform durable writes while no physical lease is
        # active. Re-read the clock afterwards so old health is never evaluated
        # against the pre-I/O timestamp and accidentally treated as fresh.
        self._reconcile(now_ns=self.clock_ns(), trigger=read.sample)
        return True

    def tick(self) -> None:
        now_ns = self.clock_ns()
        self.application.advance_vr_clock(now_ns)
        # A stale transition may persist diagnostics. Evaluate DDS freshness
        # against the clock after that work, just like consume(), so old
        # evidence cannot receive a new monotonic health timestamp.
        self._reconcile(now_ns=self.clock_ns(), trigger=None)

    def _reconcile(self, *, now_ns: int, trigger: Any | None) -> None:
        mode = self.application.session.supervisor.mode
        if mode in {SupervisorMode.DISABLED, SupervisorMode.ABORT}:
            return

        action_src = self.evidence.action_src(now_ns=now_ns)
        retarget_fresh = self.evidence.accepted_retarget_fresh(now_ns=now_ns)
        human_signal = action_src in {"TAKEOVER", "BACKWARD"} or retarget_fresh
        if mode != SupervisorMode.HUMAN and human_signal:
            if isinstance(trigger, AcceptedRetargetSample):
                takeover_timestamp_ns = trigger.timestamp_ns
                reason = "accepted_retarget"
            elif isinstance(trigger, ActionSourceSample):
                takeover_timestamp_ns = trigger.timestamp_ns
                reason = f"action_src_{trigger.status.lower()}"
            else:
                takeover_timestamp_ns = now_ns
                reason = "fresh_human_evidence"
            side_request_active = bool(
                self.evidence.network_fresh(now_ns=now_ns)
                and self.evidence.vr is not None
                and self.evidence.vr.side_buttons_pressed
            )
            if side_request_active:
                self.application.human_takeover(
                    takeover_timestamp_ns,
                    detail={"reason": reason, "request_source": "side_button"},
                )
                self._external_authority_active = False
                self._human_hold_reanchored = False
                self._clear_human_camera_barrier()
            elif not self._external_authority_active:
                # A vendor authority change without a side-button request must
                # revoke local motion, but it is not an operator-requested Human
                # correction and therefore cannot create preferred data.
                self.application.external_authority_preempt(
                    takeover_timestamp_ns,
                    detail={"reason": reason},
                )
                self._external_authority_active = True
            mode = self.application.session.supervisor.mode
            now_ns = self.clock_ns()
        elif mode == SupervisorMode.HUMAN and human_signal and self._human_hold_reanchored:
            # Human may press a control again while we are waiting for the
            # post-reanchor motion/camera evidence. Suspend the newly installed
            # HOLD immediately and require another measured-pose barrier after
            # this later correction; never reuse the previous handback ACK.
            self.application.restart_human_control_hold()
            self._human_hold_reanchored = False
            self._clear_human_camera_barrier()
        elif not human_signal:
            self._external_authority_active = False

        safety = dict(self.safety_state())
        # Safety/collision integration may read a device or synchronized cache.
        # Evaluate all DDS ages against the clock after that callback returns.
        now_ns = self.clock_ns()
        emergency_stop_active = bool(safety.get("emergency_stop_active", True))
        collision_detected = bool(safety.get("collision_detected", True))
        if mode == SupervisorMode.HUMAN:
            reanchor_completed_this_cycle = False
            handback = self.evidence.handback_evidence(
                now_ns=now_ns,
                cameras_fresh=False,
                emergency_stop_active=emergency_stop_active,
                collision_detected=collision_detected,
            )
            if self._human_hold_reanchored and not handback.side_buttons_released:
                # VR button/trigger level is visible before vendor arbitration
                # necessarily emits a new accepted-retarget/action_src sample.
                # Stop the provisional local HOLD immediately and require a
                # new measured-pose ACK after this additional Human motion.
                self.application.restart_human_control_hold()
                self._human_hold_reanchored = False
                self._clear_human_camera_barrier()
            decision = evaluate_human_handback(handback)
            # motion_status can legitimately report the stale pre-Human GDK
            # target after a correction.  Once every independent Human-release
            # and platform-safety gate passes, re-anchor the local HOLD to the
            # latest measured pose.  Do this before camera capture and before
            # asking motion health to clear; Policy authority remains blocked.
            reanchor_gate = evaluate_human_handback(
                replace(handback, cameras_fresh=True, motion_healthy=True)
            )
            reanchor_safe = self.evidence.human_hold_reanchor_safe(
                now_ns=now_ns,
                emergency_stop_active=emergency_stop_active,
                collision_detected=collision_detected,
            )
            if (
                trigger is None
                and not self._human_hold_reanchored
                and reanchor_gate.ready
                and reanchor_safe
            ):
                self._human_hold_reanchored = bool(self.application.prepare_human_handback())
                # A newly accepted GDK target needs a subsequent DDS
                # motion-status sample. Never complete handback using the
                # status snapshot that preceded the re-anchor ACK.
                reanchor_completed_this_cycle = self._human_hold_reanchored
                if self._human_hold_reanchored:
                    self._clear_human_camera_barrier()

            camera_barrier_ready = self._human_camera_barrier_ready(now_ns=now_ns)
            if camera_barrier_ready:
                handback = self.evidence.handback_evidence(
                    now_ns=now_ns,
                    cameras_fresh=True,
                    emergency_stop_active=emergency_stop_active,
                    collision_detected=collision_detected,
                )
                decision = evaluate_human_handback(handback)
            elif (
                trigger is None
                and
                self._human_hold_reanchored
                and not reanchor_completed_this_cycle
                and decision.blockers == ("camera_data_stale",)
            ):
                cameras_fresh = bool(self.cameras_fresh())
                # Three-camera capture is allowed to block.  Its completion
                # cannot preserve the pre-capture freshness of measured state,
                # network, motion, or arbitration evidence.
                now_ns = self.clock_ns()
                handback = self.evidence.handback_evidence(
                    now_ns=now_ns,
                    # Even a successful capture cannot complete handback in
                    # this cycle: VR/DDS frames received during the blocking
                    # callback may still be queued. A later released VR frame
                    # must cross the barrier first.
                    cameras_fresh=False,
                    emergency_stop_active=emergency_stop_active,
                    collision_detected=collision_detected,
                )
                decision = evaluate_human_handback(handback)
                if cameras_fresh:
                    self._human_camera_verified_ns = now_ns
                    self._human_camera_barrier_vr_revision = self.evidence.vr_revision
            if (
                trigger is None
                and self._human_hold_reanchored
                and not reanchor_completed_this_cycle
                and decision.ready
            ):
                # Human handback is a grant, unlike takeover/revocation. Grant
                # only from the periodic tick after the service has drained all
                # queued urgent VR/arbitration/fault evidence; an older released
                # frame must not win ahead of a newer queued press.
                self.application.human_finish(now_ns, evidence=handback)
                self._human_hold_reanchored = False
                self._clear_human_camera_barrier()

        # Any handback transition above may also perform durable bookkeeping.
        # Stamp rollback health only after taking a final wall-clock sample.
        now_ns = self.clock_ns()
        rollback_health = replace(
            self.evidence.rollback_health(
                now_ns=now_ns,
                emergency_stop_active=emergency_stop_active,
                collision_detected=collision_detected,
            ),
            evaluated_monotonic_ns=self.monotonic_clock_ns(),
        )
        self.application.update_rollback(rollback_health)

    def _clear_human_camera_barrier(self) -> None:
        self._human_camera_verified_ns = None
        self._human_camera_barrier_vr_revision = None

    def _human_camera_barrier_ready(self, *, now_ns: int) -> bool:
        verified_ns = self._human_camera_verified_ns
        barrier_revision = self._human_camera_barrier_vr_revision
        if verified_ns is None or barrier_revision is None:
            return False
        if now_ns < verified_ns or now_ns - verified_ns > self.evidence.evidence_timeout_ns:
            self._clear_human_camera_barrier()
            return False
        return self.evidence.vr_revision > barrier_revision


class G2DDSRuntimeService:
    """Fan DDS reads into one bounded, serialized runtime coordinator."""

    def __init__(
        self,
        *,
        client: G2ReadOnlyDDSClient,
        coordinator: G2DDSRuntimeCoordinator,
        labels: Sequence[str] = DEFAULT_RUNTIME_TOPIC_LABELS,
        queue_size: int = 8192,
        tick_seconds: float = 0.02,
    ) -> None:
        if queue_size <= 0 or tick_seconds <= 0:
            raise ValueError("DDS runtime queue/tick settings must be positive")
        unknown = sorted(set(labels) - client.specs.keys())
        if unknown:
            raise ValueError(f"DDS runtime requested unknown labels: {unknown}")
        self.client = client
        self.coordinator = coordinator
        self.labels = tuple(labels)
        self.tick_seconds = tick_seconds
        self.messages: queue.PriorityQueue[tuple[int, int, G2DDSRead]] = queue.PriorityQueue(
            maxsize=queue_size
        )
        self._message_sequence = count()
        self.errors: queue.Queue[BaseException] = queue.Queue()
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []

    def run(self, *, external_stop: threading.Event | None = None) -> None:
        if self._workers:
            raise RuntimeError("G2 DDS runtime service may run only once")
        external_stop = external_stop or threading.Event()
        self._workers = [
            threading.Thread(
                target=self._subscriber_worker,
                args=(label, external_stop),
                name=f"flowpro-dds-{label}",
                daemon=True,
            )
            for label in self.labels
        ]
        for worker in self._workers:
            worker.start()
        next_tick = time.monotonic() + self.tick_seconds
        try:
            while not external_stop.is_set() and not self._stop.is_set():
                self._raise_worker_error()
                timeout = max(0.0, min(self.tick_seconds, next_tick - time.monotonic()))
                try:
                    _priority, _sequence, read = self.messages.get(timeout=timeout)
                except queue.Empty:
                    pass
                else:
                    self.coordinator.consume(read)
                if time.monotonic() >= next_tick:
                    self._drain_urgent_before_tick()
                    # A bounded drain can leave urgent evidence queued during a
                    # burst. Skipping this grant-capable tick is fail-closed;
                    # the next cycle drains more before reconsidering motion.
                    if not self._priority_evidence_waiting():
                        self.coordinator.tick()
                    next_tick = time.monotonic() + self.tick_seconds
            self._raise_worker_error()
        finally:
            self._stop.set()
            for worker in self._workers:
                worker.join(timeout=2.0)
            alive = [worker.name for worker in self._workers if worker.is_alive()]
            if alive:
                raise RuntimeError(f"G2 DDS subscriber threads did not stop: {alive}")

    def stop(self) -> None:
        self._stop.set()

    def _subscriber_worker(self, label: str, external_stop: threading.Event) -> None:
        try:
            self.client.stream(
                label,
                self._enqueue,
                stop=lambda: self._stop.is_set() or external_stop.is_set(),
            )
        except BaseException as error:
            self.errors.put(error)
            self._stop.set()

    def _enqueue(self, read: G2DDSRead) -> None:
        if read.label == "vr_data" and isinstance(read.sample, VRTeleopSample):
            # Revocation must not wait behind the serialized audit/health
            # queue.  This hook is strictly revoke-only; validated X-down and
            # every state transition remain coordinator-owned.
            observe_sample = getattr(
                getattr(self.coordinator, "application", None),
                "observe_transport_vr_sample",
                None,
            )
            if callable(observe_sample):
                observe_sample(read.sample)
            else:
                observe_frame = getattr(
                    getattr(self.coordinator, "application", None),
                    "observe_transport_vr_frame",
                    None,
                )
                if callable(observe_frame):
                    observe_frame(read.sample.raw_frame())
        priority = (
            0 if read.label in _URGENT_RUNTIME_TOPICS else (1 if read.label in _HEALTH_RUNTIME_TOPICS else 2)
        )
        try:
            self.messages.put(
                (priority, next(self._message_sequence), read),
                timeout=self.tick_seconds,
            )
        except queue.Full as error:
            raise RuntimeError("G2 DDS runtime queue overflowed; refusing to drop evidence") from error

    def _drain_urgent_before_tick(self) -> None:
        """Apply already-received VR/authority evidence before stale timers.

        The fixed budget prevents continuously publishing VR/health from
        starving periodic ticks.  After consuming one ordinary item we inspect
        the queue again: a VR release may have arrived while that item's handler
        was blocked in persistence, and must be applied before the stale tick.
        """

        for _ in range(min(self.messages.maxsize, _PRE_TICK_DRAIN_BUDGET)):
            try:
                priority, _sequence, read = self.messages.get_nowait()
            except queue.Empty:
                return
            self.coordinator.consume(read)
            if priority >= 2 and not self._priority_evidence_waiting():
                return

    def _priority_evidence_waiting(self) -> bool:
        """Atomically inspect whether control/health evidence is queued."""

        # queue.PriorityQueue has no public peek operation.  Its documented
        # mutex protects the heap and lets us inspect without dequeue/requeue,
        # which could otherwise deadlock if a producer fills the freed slot.
        with self.messages.mutex:
            return bool(self.messages.queue and self.messages.queue[0][0] < 2)

    def _raise_worker_error(self) -> None:
        try:
            error = self.errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("G2 DDS subscriber failed") from error
