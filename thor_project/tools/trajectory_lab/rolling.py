"""Offline future-boundary handoffs; no SDK, threads, network, or RTC wire API."""

import logging
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .trajectory import PlanningError, Sample, Trajectory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Handoff:
    token: str
    generation: int
    source_id: str
    at: float
    sample: Sample


class RollingTimeline:
    """Separate an immutable future reference from collision-checked outputs.

    Reserving a boundary does NOT certify the intervening path. Each adopted
    sample still passes the supplied checker. This deterministic lab scheduler
    cannot establish robot deadlines or emulate actual physical feedback.
    """

    def __init__(self, trajectory, *, start=0.0):
        if not isinstance(trajectory, Trajectory) or not np.isfinite(start):
            raise PlanningError("Expected an offline trajectory and finite start time")
        self.active = trajectory
        self.start = float(start)
        self.clock = float(start)
        self.last = trajectory.sample(0)
        self.generation = 0
        self.reservation = None
        self.scheduled = None
        self.held = False
        self.switches = 0

    def _time(self, now):
        if not np.isfinite(now) or now < self.clock - 1e-10:
            raise PlanningError("Timeline time must be finite and monotonic")
        self.clock = max(self.clock, float(now))

    def reserve(self, now, *, lead_s, allow_terminal_hold=False):
        self._time(now)
        if self.held or self.reservation is not None or self.scheduled is not None:
            raise PlanningError("A held or reserved timeline cannot reserve another boundary")
        if not np.isfinite(lead_s) or lead_s <= 0:
            raise PlanningError("Handoff lead must be finite and positive")
        # Choose an ORIGINAL source event boundary: do not skip a gripper
        # event just to fit a fractional initial interval into a 100 Hz tick.
        times = np.interp(self.active.original_phases, self.active.phases, self.active.times) + self.start
        index = int(np.searchsorted(times, now + lead_s, side="left"))
        if index >= len(times):
            if not allow_terminal_hold:
                raise PlanningError("Insufficient planned horizon for this handoff lead")
            # A trajectory's rest endpoint is a defined stationary reference,
            # not extrapolated motion. Allow rejoining it after late replies.
            # Do not hide an invalid endpoint behind sample(duration)'s rest.
            endpoint = self.active.sample(self.active.duration)
            body = [j for j in range(len(endpoint.position)) if j not in self.active.device_indices]
            c, h = self.active.coefficients[-1], self.active.durations[-1]
            for order, name in enumerate(("position", "velocity", "acceleration")):
                weights = np.array([np.prod(range(k - order + 1, k + 1)) for k in range(order, 6)])
                actual = weights @ c[order:, :][:, body] / h**order
                if not np.allclose(actual, getattr(endpoint, name)[body], rtol=0, atol=1e-8):
                    raise PlanningError("Terminal handoff requires a polynomial rest endpoint")
            at = float(now + lead_s)
            self.reservation = Handoff(uuid4().hex, self.generation, self.active.source_id, at, endpoint)
            return self.reservation
        # Stay just AFTER the event despite subtraction roundoff in global
        # time. Otherwise the gripper lookup could still select the old goal.
        at = float(times[index]) + 1e-9
        self.reservation = Handoff(
            uuid4().hex, self.generation, self.active.source_id, at, self.active.sample(at - self.start)
        )
        return self.reservation

    def cancel(self, request):
        if self.reservation is request:
            self.reservation = None

    def submit(self, trajectory, request, *, now):
        self._time(now)
        if (
            self.held
            or not isinstance(request, Handoff)
            or self.reservation is not request
            or request.generation != self.generation
            or request.source_id != self.active.source_id
        ):
            raise PlanningError("Stale handoff: active path or generation changed")
        if now >= request.at:
            self.cancel(request)
            raise PlanningError("Late handoff: keep the previously planned path")
        if not isinstance(trajectory, Trajectory):
            raise PlanningError("Expected a bounded trajectory")
        self._compatible(trajectory)
        first = trajectory.sample(0)
        for name in ("position", "velocity", "acceleration"):
            if not np.allclose(getattr(first, name), getattr(request.sample, name), rtol=0, atol=1e-9):
                raise PlanningError(f"Handoff does not preserve {name}")
        if abs(first.phase - request.sample.phase) > 1e-9:
            raise PlanningError("Handoff does not preserve source phase")
        self.scheduled = (request.at, trajectory)
        self.reservation = None

    def hold(self):
        self.held = True
        self.generation += 1
        self.reservation = None
        self.scheduled = None
        z = np.zeros_like(self.last.position)
        self.last = Sample(self.last.position, z, z, z, self.last.phase, self.last.time)

    def _compatible(self, trajectory):
        if trajectory.device_indices != self.active.device_indices:
            raise PlanningError("Handoff changed device mapping")
        for name in ("velocity", "acceleration", "jerk", "lower", "upper"):
            if not np.array_equal(getattr(trajectory.limits, name), getattr(self.active.limits, name)):
                raise PlanningError("Handoff changed trajectory limits")

    def recover(self, trajectory, *, now):
        """Explicit offline replan from held q/v/a; never resume old pending work."""
        self._time(now)
        if not self.held or not isinstance(trajectory, Trajectory):
            raise PlanningError("Recovery requires a held timeline and new trajectory")
        self._compatible(trajectory)
        first = trajectory.sample(0)
        for name in ("position", "velocity", "acceleration"):
            if not np.allclose(getattr(first, name), getattr(self.last, name), rtol=0, atol=1e-9):
                raise PlanningError(f"Recovery changed held {name}")
        if abs(first.phase - self.last.phase) > 1e-9:
            raise PlanningError("Recovery changed held source phase")
        self.active, self.start, self.clock = trajectory, float(now), float(now)
        self.held = False

    def adopt(self, now, checker):
        self._time(now)
        self.clock = float(now)
        if self.held:
            return self.last
        if self.scheduled is not None and now >= self.scheduled[0]:
            self.start, self.active = self.scheduled
            self.scheduled = None
            self.switches += 1
        candidate = self.active.sample(now - self.start)
        try:
            allowed = checker(self.last.position.copy(), candidate.position.copy())
            if not isinstance(allowed, (bool, np.bool_)):
                raise PlanningError("Checker must return an explicit boolean")
        except Exception:
            self.hold()
            raise
        if allowed:
            self.last = candidate
        else:
            self.hold()
            logger.warning("Offline candidate rejected: hold last adopted target; pending handoff invalidated")
        return self.last
