"""State records shared by control use cases and infrastructure adapters.

These objects contain data and serialization only.  They deliberately know
nothing about HTTP, task admission, robot transports, or vision processes.
"""

from __future__ import annotations

import copy
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class DeploymentJob:
    """Persisted deployment and task state owned by one ControlPlane."""

    id: str
    config: dict[str, Any]
    status: str = "queued"
    progress: int = 0
    message: str = "部署任务已创建"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    approval_deadline: float | None = None
    model_service: dict[str, Any] = field(default_factory=dict)
    task_session: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    robot_reset: dict[str, Any] = field(default_factory=dict)
    presence_gate: dict[str, Any] = field(default_factory=dict)
    batch: dict[str, Any] = field(default_factory=dict)
    revision: int = 0

    def __post_init__(self) -> None:
        self.telemetry.setdefault(
            "cameras",
            [
                {"id": "cam_head", "label": "头部相机", "status": "waiting", "image_url": None},
                {"id": "cam_left_wrist", "label": "左腕相机", "status": "waiting", "image_url": None},
                {"id": "cam_right_wrist", "label": "右腕相机", "status": "waiting", "image_url": None},
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "revision": self.revision,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": copy.deepcopy(self.events),
            "config": copy.deepcopy(self.config),
            "model_service": copy.deepcopy(self.model_service),
            "task_session": copy.deepcopy(self.task_session),
            "telemetry": copy.deepcopy(self.telemetry),
            "robot_reset": copy.deepcopy(self.robot_reset),
            "presence_gate": copy.deepcopy(self.presence_gate),
            "batch": copy.deepcopy(self.batch),
        }
        started_at = result["model_service"].get("started_at")
        if isinstance(started_at, (int, float)):
            finished_at = result["model_service"].get("finished_at")
            result["model_service"]["elapsed_s"] = round(
                max(0.0, float(finished_at or time.time()) - float(started_at)), 1
            )
        if self.error is not None:
            result["error"] = self.error
        if self.approval_deadline is not None:
            result["approval_deadline"] = self.approval_deadline
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeploymentJob:
        return cls(
            id=str(value["id"]),
            revision=int(value.get("revision", 0)),
            config=copy.deepcopy(dict(value.get("config", {}))),
            status=str(value.get("status", "failed")),
            progress=int(value.get("progress", 0)),
            message=str(value.get("message", "恢复的部署任务")),
            created_at=float(value.get("created_at", time.time())),
            updated_at=float(value.get("updated_at", time.time())),
            events=copy.deepcopy(list(value.get("events", []))),
            error=str(value["error"]) if value.get("error") is not None else None,
            approval_deadline=(
                float(value["approval_deadline"]) if value.get("approval_deadline") is not None else None
            ),
            model_service=copy.deepcopy(dict(value.get("model_service", {}))),
            task_session=copy.deepcopy(dict(value.get("task_session", {}))),
            telemetry=copy.deepcopy(dict(value.get("telemetry", {}))),
            robot_reset=copy.deepcopy(dict(value.get("robot_reset", {}))),
            presence_gate=copy.deepcopy(dict(value.get("presence_gate", {}))),
            batch=copy.deepcopy(dict(value.get("batch", {}))),
        )


@dataclass(frozen=True)
class PresenceRuntimeSettings:
    """Server-owned paths and private ingress settings for visual inference."""

    project_directory: Path
    python_executable: Path
    weights: Path
    rules: Path
    device: str
    bind_host: str
    ingress_port: int
    control_api_url: str
    detect_hz: float
    startup_timeout_s: float


@dataclass
class PresenceSidecar:
    """One control-owned vision subprocess and its private runtime files."""

    job_id: str
    generation: int
    process: subprocess.Popen[Any]
    runtime_directory: Path
    ingress_token_file: Path
    control_token_file: Path
    ready_file: Path
    log_file: Path
    log_handle: Any
