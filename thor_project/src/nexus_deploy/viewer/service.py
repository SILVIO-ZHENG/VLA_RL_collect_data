"""Lifecycle manager for the project-owned simulator subprocess."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from urllib import error as urllib_error
from urllib import request as urllib_request

INTEGRATED_SIMULATOR_COMPONENT = "nexus-platform-integrated-simulator"


def _health(url: str, timeout: float = 0.5) -> dict[str, Any] | None:
    try:
        with urllib_request.urlopen(f"{url}/api/healthz", timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read(32 * 1024))
    except (OSError, ValueError, urllib_error.URLError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class SimulatorProcess:
    process: subprocess.Popen[bytes] | None
    log_stream: IO[bytes] | None
    control_url: str
    viewer_port: int
    log_path: Path

    @classmethod
    def start(
        cls,
        *,
        python: Path,
        runtime_dir: Path,
        alignment_surfaces_directory: Path | None = None,
        host: str = "0.0.0.0",
        port: int = 8766,
        viewer_port: int = 8767,
        timeout_s: float = 45.0,
    ) -> SimulatorProcess:
        control_url = f"http://127.0.0.1:{port}"
        existing = _health(control_url)
        if existing is not None:
            if existing.get("component") != INTEGRATED_SIMULATOR_COMPONENT:
                raise RuntimeError(
                    f"端口 {port} 已被非 Nexus 内置仿真服务占用；请先停止旧仿真进程"
                )
            raise RuntimeError(
                f"端口 {port} 已有另一个 Nexus 内置仿真进程 "
                f"(parent_pid={existing.get('parent_pid')})；请先停止其平台进程"
            )
        if not python.is_file():
            raise RuntimeError(
                f"当前 Nexus Python 解释器不存在: {python}；请重新安装当前工程"
            )
        # Keep a venv interpreter path intact when Nexus itself is installed in a venv.
        python = python.expanduser().absolute()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        alignment_surfaces_directory = (
            Path(alignment_surfaces_directory).expanduser().resolve()
            if alignment_surfaces_directory is not None
            else (runtime_dir / "alignment-surfaces").resolve()
        )
        log_path = runtime_dir / "simulator.log"
        log_stream = log_path.open("ab", buffering=0)
        command = [
            str(python),
            "-m",
            "nexus_deploy.viewer.runtime",
            "--host",
            host,
            "--port",
            str(port),
            "--meshcat-port",
            str(viewer_port),
            "--state-dir",
            str(runtime_dir),
            "--alignment-surfaces-directory",
            str(alignment_surfaces_directory),
            "--parent-pid",
            str(os.getpid()),
        ]
        process = subprocess.Popen(  # noqa: S603 - fixed allow-listed command
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            health = _health(control_url, timeout=0.4)
            if health is not None and health.get("component") == INTEGRATED_SIMULATOR_COMPONENT:
                return cls(process, log_stream, control_url, viewer_port, log_path)
            time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
        log_stream.close()
        try:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:].strip()
        except OSError:
            detail = ""
        raise RuntimeError(f"内置仿真启动失败{': ' + detail if detail else ''}")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
        if self.log_stream is not None:
            self.log_stream.close()
