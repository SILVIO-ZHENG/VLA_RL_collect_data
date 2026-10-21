"""History-based confirmation for per-frame parcel-state decisions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from state_engine import STATE_NAMES, StateDecision


@dataclass(frozen=True)
class TemporalConfig:
    history_size: int = 12
    confirm_frames: int = 3
    roi_confirm_frames: int = 4
    no_package_confirm_frames: int = 5
    dropout_hold_frames: int = 3
    held_confidence_decay: float = 0.92
    max_unconfirmed_frames: int = 6
    min_held_confidence: float = 0.35
    transport_evidence_ttl_frames: int = 30

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TemporalConfig":
        payload = payload or {}
        if not isinstance(payload, Mapping):
            raise ValueError("configuration 'temporal' must be a mapping")
        values = {
            name: payload.get(name, definition.default)
            for name, definition in cls.__dataclass_fields__.items()
        }
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "history_size",
            "confirm_frames",
            "roi_confirm_frames",
            "no_package_confirm_frames",
            "dropout_hold_frames",
            "max_unconfirmed_frames",
            "transport_evidence_ttl_frames",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"temporal.{name} must be positive")
        if not 0.0 <= float(self.held_confidence_decay) <= 1.0:
            raise ValueError("temporal.held_confidence_decay must be in [0, 1]")
        if not 0.0 <= float(self.min_held_confidence) <= 1.0:
            raise ValueError("temporal.min_held_confidence must be in [0, 1]")


@dataclass(frozen=True)
class TemporalDecision:
    state: str
    confidence: float
    reason: str
    raw_state: str
    candidate_state: str | None
    candidate_frames: int
    changed: bool


class TemporalStateFilter:
    """Confirm state changes and bridge only short segmentation dropouts.

    Geometric ambiguity (two grippers, multiple active candidates, or an
    untracked bin resident) becomes UNKNOWN immediately.  Only the explicit
    short-dropout reason may temporarily retain the preceding stable state.
    """

    def __init__(self, config: TemporalConfig):
        self.config = config
        self._history: deque[StateDecision] = deque(maxlen=config.history_size)
        self._stable_state = "UNKNOWN"
        self._stable_confidence = 0.0
        self._candidate_state: str | None = None
        self._candidate_frames = 0
        self._dropout_frames = 0
        self._unconfirmed_frames = 0
        self._transport_evidence_frames = 0
        self._support_anchor: str | None = None

    def reset(self) -> None:
        self._history.clear()
        self._stable_state = "UNKNOWN"
        self._stable_confidence = 0.0
        self._candidate_state = None
        self._candidate_frames = 0
        self._dropout_frames = 0
        self._unconfirmed_frames = 0
        self._transport_evidence_frames = 0
        self._support_anchor = None

    def _required_frames(self, state: str) -> int:
        if state == "NO_PACKAGE":
            return self.config.no_package_confirm_frames
        if state in {"ON_TABLE_LEFT", "IN_BIN"}:
            return self.config.roi_confirm_frames
        return self.config.confirm_frames

    def update(self, raw: StateDecision) -> TemporalDecision:
        if raw.state not in STATE_NAMES:
            raise ValueError(f"unsupported raw state: {raw.state}")
        self._history.append(raw)
        previous_stable = self._stable_state
        if self._transport_evidence_frames > 0:
            self._transport_evidence_frames -= 1

        is_short_dropout = (
            raw.state == "UNKNOWN"
            and raw.reason.startswith("short package-mask dropout")
        )
        if raw.state == "UNKNOWN":
            self._candidate_state = None
            self._candidate_frames = 0
            if is_short_dropout and self._stable_state != "UNKNOWN":
                self._dropout_frames += 1
                self._unconfirmed_frames += 1
                if self._dropout_frames <= self.config.dropout_hold_frames:
                    self._stable_confidence *= self.config.held_confidence_decay
                    return TemporalDecision(
                        state=self._stable_state,
                        confidence=float(self._stable_confidence),
                        reason="retained stable state through a short package-mask dropout",
                        raw_state=raw.state,
                        candidate_state=None,
                        candidate_frames=0,
                        changed=False,
                    )
            self._dropout_frames = 0
            self._unconfirmed_frames = 0
            self._stable_state = "UNKNOWN"
            self._stable_confidence = 0.0
            return TemporalDecision(
                state="UNKNOWN",
                confidence=0.0,
                reason=raw.reason,
                raw_state=raw.state,
                candidate_state=None,
                candidate_frames=0,
                changed=previous_stable != "UNKNOWN",
            )

        self._dropout_frames = 0
        if raw.state == self._stable_state:
            self._candidate_state = None
            self._candidate_frames = 0
            self._unconfirmed_frames = 0
            # Continuous confirmed gripper contact is fresh transport evidence.
            # Without this refresh, a long but uninterrupted hold would expire
            # the TTL and incorrectly reject the immediately following IN_BIN.
            if raw.state in {"IN_LEFT_GRIPPER", "IN_RIGHT_GRIPPER"}:
                self._transport_evidence_frames = (
                    self.config.transport_evidence_ttl_frames + 1
                )
            self._stable_confidence = float(
                np.clip(0.75 * self._stable_confidence + 0.25 * raw.confidence, 0.0, 1.0)
            )
            return TemporalDecision(
                state=self._stable_state,
                confidence=self._stable_confidence,
                reason="raw evidence confirms the stable state",
                raw_state=raw.state,
                candidate_state=None,
                candidate_frames=0,
                changed=False,
            )

        self._unconfirmed_frames += 1
        if raw.state == self._candidate_state:
            self._candidate_frames += 1
        else:
            self._candidate_state = raw.state
            self._candidate_frames = 1
        required = self._required_frames(raw.state)
        if self._candidate_frames >= required:
            support_states = {"ON_TABLE_LEFT", "IN_BIN"}
            invalid_support_jump = (
                raw.state in support_states
                and self._support_anchor in support_states
                and raw.state != self._support_anchor
                and self._transport_evidence_frames <= 0
            )
            if invalid_support_jump:
                self._stable_state = "UNKNOWN"
                self._stable_confidence = 0.0
                self._candidate_state = None
                self._candidate_frames = 0
                self._unconfirmed_frames = 0
                return TemporalDecision(
                    state="UNKNOWN",
                    confidence=0.0,
                    reason=(
                        f"rejected direct {self._support_anchor} -> {raw.state} "
                        "transition without confirmed gripper transport"
                    ),
                    raw_state=raw.state,
                    candidate_state=None,
                    candidate_frames=0,
                    changed=previous_stable != "UNKNOWN",
                )
            recent_confidences = [
                item.confidence
                for item in list(self._history)[-self._candidate_frames :]
                if item.state == raw.state
            ]
            self._stable_state = raw.state
            self._stable_confidence = float(
                np.mean(recent_confidences) if recent_confidences else raw.confidence
            )
            self._candidate_state = None
            self._candidate_frames = 0
            self._unconfirmed_frames = 0
            if raw.state in {"IN_LEFT_GRIPPER", "IN_RIGHT_GRIPPER"}:
                self._transport_evidence_frames = (
                    self.config.transport_evidence_ttl_frames + 1
                )
            elif raw.state in support_states:
                self._support_anchor = raw.state
                self._transport_evidence_frames = 0
            elif raw.state == "NO_PACKAGE":
                self._support_anchor = None
                self._transport_evidence_frames = 0
            return TemporalDecision(
                state=self._stable_state,
                confidence=self._stable_confidence,
                reason=f"confirmed by {required} consecutive sampled frames",
                raw_state=raw.state,
                candidate_state=None,
                candidate_frames=0,
                changed=previous_stable != self._stable_state,
            )

        if self._stable_state == "UNKNOWN":
            return TemporalDecision(
                state="UNKNOWN",
                confidence=0.0,
                reason=f"waiting for {required} consecutive frames of {raw.state}",
                raw_state=raw.state,
                candidate_state=raw.state,
                candidate_frames=self._candidate_frames,
                changed=False,
            )
        self._stable_confidence *= self.config.held_confidence_decay
        expired = (
            self._unconfirmed_frames >= self.config.max_unconfirmed_frames
            or self._stable_confidence < self.config.min_held_confidence
        )
        if expired:
            self._stable_state = "UNKNOWN"
            self._stable_confidence = 0.0
            self._unconfirmed_frames = 0
            return TemporalDecision(
                state="UNKNOWN",
                confidence=0.0,
                reason=(
                    "prior state expired while incoming evidence remained "
                    "unconfirmed"
                ),
                raw_state=raw.state,
                candidate_state=self._candidate_state,
                candidate_frames=self._candidate_frames,
                changed=True,
            )
        return TemporalDecision(
            state=self._stable_state,
            confidence=float(self._stable_confidence),
            reason=f"holding prior state while confirming {raw.state}",
            raw_state=raw.state,
            candidate_state=raw.state,
            candidate_frames=self._candidate_frames,
            changed=False,
        )
