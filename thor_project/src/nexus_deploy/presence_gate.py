# 中文：
# 纯视觉任务门禁状态机：接收 PACKAGE_PRESENT/NO_PACKAGE/UNKNOWN 证据，
# 提出 PAUSE/RESUME/RESET 请求。没有 GDK 导入，不直接执行动作；ControlPlane
# 复核所有权并收到机器人回执后 acknowledge 才完成状态转换。
# UNKNOWN/过期/乱序或断流都不能积累连续空场时间。
# English:
# Pure visual task gate proposes PAUSE/RESUME/RESET from presence evidence, with no GDK
# or direct actuation. ControlPlane checks ownership and acknowledges robot outcomes
# before transitions. Unknown/stale/out-of-order/disconnected evidence cannot accumulate empty time.
"""Pure package-presence task gate owned by the Nexus control plane.

This module has no GDK, SSH, network, subprocess, camera, or model dependency.
It evaluates only already-confirmed visual states and produces an auditable
proposal.  The control-plane adapter must independently check task ownership,
client state, reset capability, and the result of every existing guarded action
before acknowledging a proposal.

The visual sidecar is intentionally unable to call this module's future action
adapter directly: it can only POST a bounded observation to Nexus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


PACKAGE_PRESENT = "PACKAGE_PRESENT"
NO_PACKAGE = "NO_PACKAGE"
UNKNOWN = "UNKNOWN"
PRESENCE_STATES = frozenset({PACKAGE_PRESENT, NO_PACKAGE, UNKNOWN})
_NS_PER_SECOND = 1_000_000_000


class TaskState(str, Enum):
    """Minimal task state supplied by the Nexus control adapter."""

    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


class GateAction(str, Enum):
    """A proposal only; this module never executes it."""

    NONE = "none"
    PAUSE = "pause"
    RESUME = "resume"
    RESET = "reset"


@dataclass(frozen=True)
class PresenceTaskGateConfig:
    """Frozen policy for one human-authorised task session."""

    enabled: bool = False
    pause_after_s: float = 2.0
    reset_after_s: float = 5.0
    max_state_age_s: float = 0.75
    auto_resume: bool = True

    def validate(self) -> None:
        if self.pause_after_s <= 0:
            raise ValueError("pause_after_s must be positive")
        if self.reset_after_s <= self.pause_after_s:
            raise ValueError("reset_after_s must be greater than pause_after_s")
        if self.max_state_age_s <= 0:
            raise ValueError("max_state_age_s must be positive")


@dataclass(frozen=True)
class GateDecision:
    """One side-effect-free result for an authenticated visual observation."""

    session_id: str | None
    action: GateAction
    presence_state: str
    source_fresh: bool
    no_package_elapsed_s: float | None
    auto_paused: bool
    awaiting_reset_ready: bool
    pending_action: GateAction | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["action"] = self.action.value
        if self.pending_action is not None:
            payload["pending_action"] = self.pending_action.value
        return payload


class PresenceTaskGate:
    """Translate fresh, four-frame-confirmed presence states into proposals.

    Rules:

    * Only a continuous sequence of fresh ``NO_PACKAGE`` observations advances
      the empty-area timer. ``UNKNOWN`` breaks that sequence and never asks for
      an action.
    * At 2 seconds the gate proposes a soft pause.
    * At 5 seconds it proposes a reset only after that same pause was applied
      and the task is still paused.
    * ``PACKAGE_PRESENT`` may resume only a still-valid session that this gate
      itself paused. It can never create, replace, or revive a human-ended job.
    * Every proposal waits for the control adapter's explicit acknowledgement;
      no duplicate action is proposed while an acknowledgement is pending.
    """

    def __init__(self, config: PresenceTaskGateConfig | None = None):
        self.config = config or PresenceTaskGateConfig()
        self.config.validate()
        self._session_id: str | None = None
        self._no_package_since_ns: int | None = None
        self._last_observed_ns: int | None = None
        self._pending_action: GateAction | None = None
        self._auto_paused = False
        self._awaiting_reset_ready = False
        self._reset_cycle_complete = False

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def auto_paused(self) -> bool:
        return self._auto_paused

    @property
    def pending_action(self) -> GateAction | None:
        return self._pending_action

    @property
    def awaiting_reset_ready(self) -> bool:
        return self._awaiting_reset_ready

    def start_session(self, session_id: str) -> None:
        normalized = str(session_id).strip()
        if not normalized:
            raise ValueError("session_id cannot be empty")
        self._session_id = normalized
        self._no_package_since_ns = None
        self._last_observed_ns = None
        self._pending_action = None
        self._auto_paused = False
        self._awaiting_reset_ready = False
        self._reset_cycle_complete = False

    def clear_session(self) -> None:
        self._session_id = None
        self._no_package_since_ns = None
        self._last_observed_ns = None
        self._pending_action = None
        self._auto_paused = False
        self._awaiting_reset_ready = False
        self._reset_cycle_complete = False

    def mark_externally_resumed(self) -> None:
        """Release gate pause ownership after a successful human resume."""

        if self._pending_action is not None:
            raise ValueError("cannot reconcile a manual resume while an action is pending")
        if self._awaiting_reset_ready:
            raise ValueError("cannot reconcile a manual resume while reset completion is pending")
        self._auto_paused = False
        self._no_package_since_ns = None
        self._reset_cycle_complete = False

    def invalidate_evidence(self) -> None:
        """ClockTick/stream loss clears evidence without inventing an action."""
        self._clear_empty_timer()
        # Keep the last sequence timestamp and pending command for deduplication.

    def mark_operator_paused(self) -> None:
        self._auto_paused = False
        self._clear_empty_timer()
        self._reset_cycle_complete = False

    def mark_reset_complete(self) -> None:
        """Allow a later package to resume only after the guarded reset completed."""

        if self._pending_action is not None:
            raise ValueError("cannot mark reset complete while an action is pending")
        if not self._awaiting_reset_ready:
            raise ValueError("no gate-owned reset is awaiting completion")
        self._awaiting_reset_ready = False
        self._auto_paused = True
        self._reset_cycle_complete = True

    # 中文：
    # 物理回执才确认门禁请求生效；失败取消 pending 而不伪造自动暂停/复位完成。
    # English:
    # Only physical acknowledgement applies a proposed action; failure clears pending
    # without fabricating successful automatic pause/reset.
    def acknowledge(self, action: GateAction | str, *, applied: bool) -> None:
        """Record a control-plane action result, without performing the action."""

        resolved = GateAction(action)
        if resolved is GateAction.NONE:
            raise ValueError("GateAction.NONE cannot be acknowledged")
        if self._pending_action is not resolved:
            raise ValueError("acknowledgement does not match the pending gate action")
        self._pending_action = None
        if not applied:
            return
        if resolved is GateAction.PAUSE:
            self._auto_paused = True
        elif resolved is GateAction.RESUME:
            self._auto_paused = False
            self._no_package_since_ns = None
            self._reset_cycle_complete = False
        elif resolved is GateAction.RESET:
            self._awaiting_reset_ready = True
            self._auto_paused = True
            self._no_package_since_ns = None

    # 中文：
    # 校验源时间递增和新鲜度，重置不连续空场证据；只有 gate 自己暂停的
    # 同一任务可由有物证据自动恢复。pending/reset 未确认期间不重复提运动。
    # English:
    # Validate increasing/fresh timestamps and reset interrupted-empty evidence. Only a task
    # paused by this gate may auto-resume; pending/unacknowledged reset blocks duplicate proposals.
    def observe(
        self,
        presence_state: str,
        *,
        observed_at_ns: int,
        now_ns: int | None = None,
        task_state: TaskState | str,
    ) -> GateDecision:
        state = str(presence_state)
        if state not in PRESENCE_STATES:
            raise ValueError(f"unsupported presence state: {presence_state!r}")
        observed = int(observed_at_ns)
        if observed <= 0:
            raise ValueError("observed_at_ns must be positive")
        now = observed if now_ns is None else int(now_ns)
        if now < observed:
            raise ValueError("now_ns cannot be earlier than observed_at_ns")
        task = TaskState(task_state)

        if self._last_observed_ns is not None and observed <= self._last_observed_ns:
            return self._decision(
                presence_state=state,
                source_fresh=False,
                elapsed_s=self._elapsed_s(),
                reason="out-of-order or duplicate visual observation ignored",
            )
        if self._last_observed_ns is not None and observed - self._last_observed_ns > int(self.config.max_state_age_s * _NS_PER_SECOND):
            self._clear_empty_timer()
        self._last_observed_ns = observed

        if now - observed > int(self.config.max_state_age_s * _NS_PER_SECOND):
            self._clear_empty_timer()
            return self._decision(
                presence_state=state,
                source_fresh=False,
                elapsed_s=None,
                reason="visual observation is stale; continuous-empty timer cleared",
            )
        if not self.config.enabled:
            self._clear_empty_timer()
            return self._decision(
                presence_state=state,
                source_fresh=True,
                elapsed_s=None,
                reason="package gate is disabled for this task",
            )
        if task is TaskState.ENDED:
            self._clear_empty_timer()
            return self._decision(
                presence_state=state,
                source_fresh=True,
                elapsed_s=None,
                reason="task is ended; no further automatic proposal",
            )
        if self._awaiting_reset_ready:
            self._clear_empty_timer()
            return self._decision(
                presence_state=state,
                source_fresh=True,
                elapsed_s=None,
                reason="gate-owned reset is awaiting physical completion acknowledgement",
            )
        if state != NO_PACKAGE:
            self._clear_empty_timer()
        if self._pending_action is not None:
            return self._decision(
                presence_state=state,
                source_fresh=True,
                elapsed_s=self._elapsed_s(observed),
                reason=f"awaiting acknowledgement of {self._pending_action.value}",
            )
        if state == NO_PACKAGE:
            return self._observe_no_package(observed, task)

        # Both a confirmed package and a mixed/UNKNOWN window break the exact
        # wording "continuous NO_PACKAGE". UNKNOWN never requests an action.
        self._clear_empty_timer()
        if (
            state == PACKAGE_PRESENT
            and self.config.auto_resume
            and task is TaskState.PAUSED
            and self._auto_paused
        ):
            self._reset_cycle_complete = False
            return self._request(
                GateAction.RESUME,
                state,
                None,
                "package present; resuming the same task previously paused by this gate",
            )
        return self._decision(
            presence_state=state,
            source_fresh=True,
            elapsed_s=None,
            reason=(
                "package present; no automatic action is needed"
                if state == PACKAGE_PRESENT
                else "presence unknown; no automatic action and empty timer cleared"
            ),
        )

    # 中文：
    # 从连续 NO_PACKAGE 开始累计，先达到 pause_after_s 提暂停，已确认视觉
    # 暂停后再达到 reset_after_s 提复位。手动暂停永远不自动复位。
    # English:
    # Accumulate continuous NO_PACKAGE; propose pause first, then reset after gate-owned
    # pause acknowledgement and reset threshold. Manual pauses never auto-reset.
    def _observe_no_package(self, observed_at_ns: int, task_state: TaskState) -> GateDecision:
        if self._no_package_since_ns is None:
            self._no_package_since_ns = observed_at_ns
        elapsed_s = self._elapsed_s(observed_at_ns)
        assert elapsed_s is not None
        if task_state is TaskState.RUNNING:
            if self._auto_paused:
                return self._decision(
                    presence_state=NO_PACKAGE,
                    source_fresh=True,
                    elapsed_s=elapsed_s,
                    reason="task reports running after a gate-owned pause; awaiting reconciliation",
                )
            if elapsed_s >= self.config.pause_after_s:
                return self._request(
                    GateAction.PAUSE,
                    NO_PACKAGE,
                    elapsed_s,
                    "continuous NO_PACKAGE reached the configured pause threshold",
                )
            return self._decision(
                presence_state=NO_PACKAGE,
                source_fresh=True,
                elapsed_s=elapsed_s,
                reason="continuous NO_PACKAGE below the pause threshold",
            )
        if task_state is TaskState.PAUSED and self._auto_paused:
            if self._reset_cycle_complete:
                return self._decision(
                    presence_state=NO_PACKAGE,
                    source_fresh=True,
                    elapsed_s=elapsed_s,
                    reason=(
                        "this continuous no-package period already completed a reset; "
                        "waiting for a fresh package before another cycle"
                    ),
                )
            if elapsed_s >= self.config.reset_after_s:
                return self._request(
                    GateAction.RESET,
                    NO_PACKAGE,
                    elapsed_s,
                    "continuous NO_PACKAGE reached the configured reset threshold",
                )
            return self._decision(
                presence_state=NO_PACKAGE,
                source_fresh=True,
                elapsed_s=elapsed_s,
                reason="task is gate-paused; waiting for reset threshold or new package",
            )
        return self._decision(
            presence_state=NO_PACKAGE,
            source_fresh=True,
            elapsed_s=elapsed_s,
            reason="task is manually paused; gate will not request a reset",
        )

    def _request(
        self,
        action: GateAction,
        presence_state: str,
        elapsed_s: float | None,
        reason: str,
    ) -> GateDecision:
        self._pending_action = action
        return self._decision(
            action=action,
            presence_state=presence_state,
            source_fresh=True,
            elapsed_s=elapsed_s,
            reason=reason,
        )

    def _clear_empty_timer(self) -> None:
        self._no_package_since_ns = None

    def _elapsed_s(self, current_ns: int | None = None) -> float | None:
        if self._no_package_since_ns is None:
            return None
        endpoint = self._last_observed_ns if current_ns is None else current_ns
        if endpoint is None:
            return None
        return max(0.0, (endpoint - self._no_package_since_ns) / _NS_PER_SECOND)

    def _decision(
        self,
        *,
        presence_state: str,
        source_fresh: bool,
        elapsed_s: float | None,
        reason: str,
        action: GateAction = GateAction.NONE,
    ) -> GateDecision:
        return GateDecision(
            session_id=self._session_id,
            action=action,
            presence_state=presence_state,
            source_fresh=source_fresh,
            no_package_elapsed_s=elapsed_s,
            auto_paused=self._auto_paused,
            awaiting_reset_ready=self._awaiting_reset_ready,
            pending_action=self._pending_action,
            reason=reason,
        )
