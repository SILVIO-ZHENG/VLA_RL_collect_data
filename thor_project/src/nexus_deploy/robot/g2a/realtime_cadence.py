"""Opt-in, bounded Linux scheduling for the two servo-critical threads.

The collision geometry process and every camera/policy thread stay on normal
scheduling. This module never grants privilege; the robot account must already
have a one-level RT priority limit. A failed promotion attempts to restore
any thread already promoted and prevents physical command streaming.
"""

from __future__ import annotations

import os
import resource
import threading


PRIORITY = 1
RT_SOFT_US = 50_000
RT_HARD_US = 100_000


def _feedback_tid(worker: object) -> int:
    thread = getattr(worker, "feedback_thread", None)
    tid = getattr(thread, "native_id", None)
    if thread is None or not thread.is_alive() or not isinstance(tid, int) or tid <= 0:
        raise RuntimeError("Realtime cadence requires a running 50 Hz feedback thread")
    return tid


def prepare_realtime_cadence(worker: object) -> None:
    """Fail before entering robot control mode if the RT boundary is unavailable."""
    if not hasattr(os, "SCHED_RR") or not hasattr(resource, "RLIMIT_RTPRIO") or not hasattr(resource, "RLIMIT_RTTIME"):
        raise RuntimeError("Realtime cadence requires Linux RT scheduler and resource limits")
    priority_limit, _ = resource.getrlimit(resource.RLIMIT_RTPRIO)
    if os.geteuid() != 0 and priority_limit != resource.RLIM_INFINITY and priority_limit < PRIORITY:
        raise RuntimeError("Realtime cadence requires RT priority limit >= 1 for the robot account")
    feedback_tid = _feedback_tid(worker)
    if os.sched_getscheduler(feedback_tid) != os.SCHED_OTHER:
        raise RuntimeError("Feedback thread already has a nonstandard scheduler")
    _, hard_limit = resource.getrlimit(resource.RLIMIT_RTTIME)
    if hard_limit != resource.RLIM_INFINITY and hard_limit < RT_HARD_US:
        raise RuntimeError("Realtime CPU-time hard limit is too low for the bounded cadence profile")
    resource.setrlimit(resource.RLIMIT_RTTIME, (RT_SOFT_US, RT_HARD_US))


class RealtimeCadence:
    """Promote feedback first, then the current action thread; undo in reverse."""

    def __init__(self, worker: object) -> None:
        self.worker = worker
        self._promoted: tuple[int, ...] = ()

    def start(self) -> None:
        if self._promoted:
            raise RuntimeError("Realtime cadence already active")
        feedback_tid = _feedback_tid(self.worker)
        action_tid = threading.get_native_id()
        if action_tid == feedback_tid:
            raise RuntimeError("Action and feedback must be distinct threads")
        if any(os.sched_getscheduler(tid) != os.SCHED_OTHER for tid in (action_tid, feedback_tid)):
            raise RuntimeError("Realtime cadence requires normal initial thread scheduling")
        promoted: list[int] = []
        try:
            for tid in (feedback_tid, action_tid):
                os.sched_setscheduler(tid, os.SCHED_RR, os.sched_param(PRIORITY))
                promoted.append(tid)
        except BaseException:
            self._promoted = tuple(reversed(promoted))
            self.stop()
            raise
        self._promoted = (action_tid, feedback_tid)

    def stop(self) -> None:
        promoted = self._promoted
        if not promoted:
            return
        error: OSError | None = None
        failed: list[int] = []
        for tid in promoted:
            try:
                os.sched_setscheduler(tid, os.SCHED_OTHER, os.sched_param(0))
            except OSError as exc:
                error = error or exc
                failed.append(tid)
        self._promoted = tuple(failed)
        if error is not None:
            raise RuntimeError("Could not restore normal servo-thread scheduling") from error
