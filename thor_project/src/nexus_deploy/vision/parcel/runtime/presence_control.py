"""Compatibility shadow API; the authoritative policy lives in Nexus.

Keep the historical shadow default (auto_resume=False) without a second policy.
"""
from dataclasses import dataclass
from nexus_deploy.presence_gate import (GateAction, GateDecision, TaskState,
    PresenceTaskGate as _Gate, PresenceTaskGateConfig as _Config)

@dataclass(frozen=True)
class PresenceTaskGateConfig(_Config):
    auto_resume: bool = False

class PresenceTaskGate(_Gate):
    def __init__(self, config=None):
        super().__init__(config or PresenceTaskGateConfig())
