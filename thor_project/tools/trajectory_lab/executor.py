"""Deterministic OFFLINE queue model, deliberately not wired to GDK or RTC wire.

The checker validates final sample-to-sample paths. Real-time proof lifetimes,
measured feedback, SDK acknowledgements and protocol negotiation are NOT supplied
by this laboratory model; passing its tests never enables hardware execution.
"""

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .trajectory import PlanningError, Sample, Trajectory, vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestAnchor:
    generation: int
    revision: int
    frozen_ticket: int
    executed_phase: float
    executed_source_id: str | None
    frozen_source_id: str | None
    frozen_sample: Sample


class OfflineExecutor:
    """Explicit deterministic ticks; cannot start threads or send robot commands."""

    def __init__(self, anchor, *, period=0.01, capacity=8):
        q = np.asarray(anchor, dtype=float)
        if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
            raise PlanningError("Invalid offline anchor")
        if period != 0.01 or type(capacity) is not int or not 1 <= capacity <= 8:
            raise PlanningError("Lab requires 100 Hz ticks and at most eight committed points")
        z = np.zeros_like(q)
        self.last = Sample(q, z, z, z, 0.0, 0.0)
        self.period = period
        self.capacity = capacity
        self.generation = 0
        self.revision = 0
        self.source_id = None
        self.tick_count = 0
        self.status = "waiting"
        self._next_ticket = 0
        self._last_ticket = 0
        self._queue = deque()
        self._pending = None
        self._pending_time = 0.0

    @property
    def committed(self):
        return tuple(item[1] for item in self._queue)

    def request_anchor(self):
        ticket, sample, source = self._queue[-1] if self._queue else (self._last_ticket, self.last, self.source_id)
        return RequestAnchor(self.generation, self.revision, ticket, self.last.phase, self.source_id, source, sample)

    def propose(self, trajectory, request):
        if self.status in ("paused", "ended"):
            raise PlanningError("Cannot install while paused or ended")
        if not isinstance(trajectory, Trajectory) or not isinstance(request, RequestAnchor):
            raise PlanningError("Expected a planned suffix and explicit request anchor")
        current = self.request_anchor()
        if (
            request.generation != current.generation
            or request.revision != current.revision
            or request.frozen_ticket != current.frozen_ticket
        ):
            raise PlanningError("Stale response: frozen boundary changed; rebase required")
        first = trajectory.sample(0.0)
        for name in ("position", "velocity", "acceleration"):
            if not np.allclose(getattr(first, name), getattr(current.frozen_sample, name), rtol=0.0, atol=1e-9):
                raise PlanningError(f"Suffix must preserve frozen boundary {name}")
        # Retain the already checked prefix exactly; only replace the pending suffix.
        self._pending = trajectory
        self._pending_time = 0.0
        self.revision += 1
        self.status = "ready"

    def _hold(self, status):
        self._queue.clear()
        self._pending = None
        self._pending_time = 0.0
        self.generation += 1
        z = np.zeros_like(self.last.position)
        self.last = Sample(self.last.position, z, z, z, self.last.phase, self.last.time)
        self.status = status

    def prepare(self, checker):
        if self.status in ("paused", "ended", "blocked", "checker_error"):
            return
        while self._pending is not None and len(self._queue) < self.capacity:
            t = min(self._pending_time + self.period, self._pending.duration)
            candidate = self._pending.sample(t)
            previous = self._queue[-1][1] if self._queue else self.last
            # The checker receives copies; it cannot alter certified output.
            try:
                accepted = checker(previous.position.copy(), candidate.position.copy())
                if not isinstance(accepted, (bool, np.bool_)):
                    raise PlanningError("Checker must return an explicit boolean")
            except Exception:
                self._hold("checker_error")
                raise
            if not accepted:
                self._hold("blocked")
                logger.warning("Offline collision rejection: holding last adopted target; new suffix required")
                return
            self._next_ticket += 1
            self._queue.append((self._next_ticket, candidate, self._pending.source_id))
            self._pending_time = t
            if t >= self._pending.duration:
                self._pending = None

    def tick(self, checker):
        self.prepare(checker)
        self.tick_count += 1
        if self._queue:
            self._last_ticket, self.last, self.source_id = self._queue.popleft()
            self.status = "running"
        elif self.status in ("running", "ready"):
            self.status = "waiting"
        return self.last

    def pause(self):
        if self.status != "ended":
            self._hold("paused")

    def resume(self, measured_anchor):
        if self.status != "paused":
            raise PlanningError("Resume requires paused state")
        q = vector(measured_anchor, len(self.last.position), "measured anchor")
        z = np.zeros_like(q)
        self.last = Sample(q, z, z, z, self.last.phase, 0.0)
        self.generation += 1
        self.status = "waiting"

    def end(self):
        self._hold("ended")
