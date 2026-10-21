"""Fail-closed accounting for rollback points accepted by the GDK writer.

Safety-window submission is deliberately absent from this interface.  The
cursor advances only when the adapter reports the selected checked point's
source and the command adopted by GDK.  The action thread owns this ledger;
the DDS thread can only request a new endpoint or revoke a lease.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RollbackDispatch:
    sequence: int
    waypoint_index: int
    action: np.ndarray


class RollbackDispatchLedger:
    """Match accepted queue outputs to one immutable historical path."""

    def __init__(
        self,
        *,
        lease_id: str,
        waypoints: np.ndarray,
        initial_sequence: int = 0,
        controlled_action_dim: int = 24,
    ) -> None:
        path = np.asarray(waypoints, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 24 or not np.isfinite(path).all() or len(path) < 2:
            raise ValueError("Rollback dispatch path must have at least two finite 24D points")
        if not lease_id:
            raise ValueError("Rollback dispatch lease must be nonempty")
        if controlled_action_dim not in (16, 24):
            raise ValueError("Controlled rollback dimension must be 16 or 24")
        if initial_sequence < 0:
            raise ValueError("Dispatch sequence cannot be negative")
        self.lease_id = lease_id
        self.waypoints = path.copy()
        self.waypoints.setflags(write=False)
        self.controlled_action_dim = controlled_action_dim
        self.sequence = initial_sequence
        self.cursor = 0
        self.endpoint = len(path) - 1

    @property
    def complete(self) -> bool:
        return self.cursor == self.endpoint

    def select_endpoint(self, waypoint_index: int) -> None:
        """Select X-release anchor or restore the suffix on a later X press."""
        if not self.cursor <= waypoint_index < len(self.waypoints):
            raise ValueError("Rollback endpoint is outside the untouched suffix")
        self.endpoint = waypoint_index

    def observe(self, receipt: dict[str, Any] | None, *, require_physical: bool = True) -> RollbackDispatch | None:
        if receipt is None:
            return None
        if require_physical and receipt.get("physical_accepted") is not True:
            raise RuntimeError("Rollback point has no GDK acceptance")
        sequence = receipt.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= self.sequence:
            raise RuntimeError("Rollback dispatch receipt is stale or unordered")
        provenance = receipt.get("provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError("Rollback dispatch has no safety provenance")
        if provenance.get("holding_previous_target") is True:
            held = np.asarray(receipt.get("action"), dtype=np.float32)
            if held.shape != (24,) or not np.allclose(
                held[: self.controlled_action_dim],
                self.waypoints[self.cursor, : self.controlled_action_dim],
                rtol=0.0,
                atol=1e-5,
            ):
                raise RuntimeError("Checked hold differs from the accepted rollback point")
            self.sequence = sequence
            return None
        if provenance.get("chunk_id") != self.lease_id:
            raise RuntimeError("Rollback dispatch belongs to another execution lease")
        base = provenance.get("candidate_action_index")
        offset = provenance.get("reference_preview_index")
        if isinstance(base, bool) or not isinstance(base, int):
            raise RuntimeError("Rollback dispatch has no historical waypoint index")
        if offset is None:
            offset = 0
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise RuntimeError("Rollback dispatch has an invalid preview offset")
        index = base + offset
        if index != self.cursor + 1 or index > self.endpoint:
            raise RuntimeError("Rollback dispatch skipped or crossed its selected anchor")
        action = np.asarray(receipt.get("action"), dtype=np.float32)
        if action.shape != (24,) or not np.isfinite(action).all():
            raise RuntimeError("Rollback dispatch action is not a finite 24D point")
        if not np.allclose(
            action[: self.controlled_action_dim],
            self.waypoints[index, : self.controlled_action_dim],
            rtol=0.0,
            atol=1e-5,
        ):
            raise RuntimeError("Safety dispatch modified the immutable rollback point")
        self.sequence = sequence
        self.cursor = index
        return RollbackDispatch(sequence, index, action.copy())


class CheckedRollbackPath:
    """Feed an exact reverse path through the existing safety queue.

    This object must be called by the robot client's sole action thread. It
    never reads GDK itself or treats a queued safety request as execution.
    Ownership handoff and measured HOLD are prerequisites supplied by the
    runner, not inferred from a button event here.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        lease_id: str,
        waypoints: np.ndarray,
        controlled_action_dim: int = 24,
    ) -> None:
        accepted = getattr(adapter, "_last_state", None)
        if accepted is None:
            raise RuntimeError("Rollback requires a known GDK command anchor")
        self.ledger = RollbackDispatchLedger(
            lease_id=lease_id,
            waypoints=waypoints,
            initial_sequence=getattr(adapter, "_dispatch_sequence", 0),
            controlled_action_dim=controlled_action_dim,
        )
        if not np.allclose(
            np.asarray(accepted, dtype=np.float32)[:controlled_action_dim],
            self.ledger.waypoints[0, :controlled_action_dim],
            rtol=0.0,
            atol=1e-5,
        ):
            raise RuntimeError("Rollback path does not begin at the accepted command anchor")
        self.adapter = adapter
        self.adapter.retire_unissued_motion(reason="rollback_lease_start")

    def select_endpoint(self, waypoint_index: int) -> None:
        if not self.ledger.cursor <= waypoint_index < len(self.ledger.waypoints):
            raise ValueError("Rollback endpoint is outside the untouched suffix")
        self.adapter.retire_unissued_motion(reason="rollback_x_release")
        self.ledger.select_endpoint(waypoint_index)

    def tick(self, *, require_physical: bool = True) -> RollbackDispatch | None:
        ledger = self.ledger
        if ledger.complete:
            return None
        next_index = ledger.cursor + 1
        # The rolling checker can request a 10-point first segment or a
        # seven-point refill behind an eight-point queue. Pad only with the
        # selected endpoint, never with a later historical waypoint.
        indices = np.minimum(
            np.arange(next_index, next_index + 21), ledger.endpoint
        )
        preview = ledger.waypoints[indices].copy()
        context = {
            "chunk_id": ledger.lease_id,
            "action_index": next_index,
            "preplanned_trajectory": True,
            "trajectory_preview": preview,
            "raw_action": preview[0].copy(),
        }
        try:
            self.adapter.send_action(preview[0], safety_context=context)
            if getattr(self.adapter, "_last_command_accepted", None) is False:
                raise RuntimeError("GDK rejected a checked rollback command")
            if getattr(self.adapter, "last_async_response", None) == "async_collision_hold":
                raise RuntimeError("Rollback safety checker requested a collision hold")
            receipt = self.adapter.dispatch_receipt()
            if receipt is not None and require_physical:
                measured = receipt.get("measured_state")
                age_ms = receipt.get("measured_age_ms")
                timestamp_ns = receipt.get("measured_timestamp_ns")
                if (
                    measured is None
                    or np.asarray(measured).shape != (24,)
                    or not np.isfinite(measured).all()
                    or isinstance(timestamp_ns, bool)
                    or not isinstance(timestamp_ns, int)
                    or timestamp_ns <= 0
                    or age_ms is None
                    or not np.isfinite(age_ms)
                    or not 0.0 <= age_ms <= 100.0
                ):
                    raise RuntimeError("Rollback dispatch lacks fresh measured feedback")
            dispatched = ledger.observe(
                receipt, require_physical=require_physical
            )
        except Exception:
            self.adapter.retire_unissued_motion(reason="rollback_dispatch_fault")
            raise
        if ledger.complete:
            self.adapter.retire_unissued_motion(reason="rollback_anchor_reached")
        return dispatched


class QueuedRollbackMotionBridge:
    """Adapt the legacy step executor to asynchronous checked dispatches.

    Only ``send_command`` is supplied to the executor. Its stationary/fault
    HOLD command remains the runner's measured-hold responsibility.
    """

    def __init__(self, adapter: Any, executor: Any, *, require_physical: bool = True) -> None:
        self.adapter = adapter
        self.executor = executor
        self.require_physical = require_physical
        self.sender: CheckedRollbackPath | None = None
        self._lease_id: str | None = None
        self._plan_join_index = 0
        self._path_join_index = 0

    def _ensure_sender(self) -> CheckedRollbackPath:
        executor = self.executor
        if executor.plan is None or executor.lease_id is None:
            raise RuntimeError("Rollback motion bridge has no armed plan and lease")
        if self.sender is not None and self._lease_id == executor.lease_id:
            return self.sender
        if self.sender is not None:
            self.adapter.retire_unissued_motion(reason="rollback_lease_rebound")
        bridge = executor._resume_bridge_waypoints
        join_index = executor._resume_join_index
        if bridge is None or join_index is None:
            raise RuntimeError("Rollback execution has no checked resume bridge")
        path = np.concatenate(
            (bridge, executor.plan.waypoints_24d[join_index + 1 :]), axis=0
        )
        self.sender = CheckedRollbackPath(
            self.adapter,
            lease_id=executor.lease_id,
            waypoints=path,
            controlled_action_dim=executor.plan.controlled_action_dim,
        )
        self._lease_id = executor.lease_id
        self._plan_join_index = join_index
        self._path_join_index = len(bridge) - 1
        return self.sender

    def send_command(self, target: np.ndarray) -> np.ndarray:
        from .rollback import RollbackDispatchPending

        sender = self._ensure_sender()
        executor = self.executor
        finish_anchor = executor._finish_to_anchor
        if finish_anchor is None:
            endpoint = len(sender.ledger.waypoints) - 1
        else:
            endpoint = (
                self._path_join_index
                + finish_anchor.waypoint_index
                - self._plan_join_index
            )
        if endpoint != sender.ledger.endpoint:
            sender.select_endpoint(endpoint)
        if sender.ledger.complete:
            raise RuntimeError("Rollback executor requested motion beyond its selected anchor")
        expected = sender.ledger.waypoints[sender.ledger.cursor + 1]
        candidate = np.asarray(target, dtype=np.float32)
        controlled_dim = sender.ledger.controlled_action_dim
        if candidate.shape != (24,) or not np.allclose(
            candidate[:controlled_dim], expected[:controlled_dim], rtol=0.0, atol=1e-5
        ):
            raise RuntimeError("Rollback executor target diverged from the checked queue path")
        dispatched = sender.tick(require_physical=self.require_physical)
        if dispatched is None:
            raise RollbackDispatchPending
        return dispatched.action
