"""Thor bridge from a fresh tablet recording to normal task admission."""

from __future__ import annotations

import json
import logging
import os
import re
import select
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

from nexus_deploy.robot.g2a.hitl.g2_runtime.tablet_start import EVENT_PREFIX

logger = logging.getLogger(__name__)
TABLET_START_EVENTS = frozenset({"ready", "awaiting_clear", "ambiguous", "unavailable", "detected"})
# Tablet episode names come from the vendor recorder and may contain Chinese.
# Keep the value to one path component before passing it to the G2 client.
RECORD_NAME = re.compile(r"(?!(?:\.{1,2})\Z)[^/\\\x00-\x1f\x7f]{1,128}\Z")


@dataclass(frozen=True)
class TabletStartTarget:
    job_id: str
    config_revision: str
    command: tuple[str, ...]


class TabletStartBridge:
    """Watch one eligible job; stop the watcher before admitting the task."""

    def __init__(
        self,
        *,
        select_target: Callable[[], TabletStartTarget | None],
        report: Callable[[TabletStartTarget, str], None],
        confirm: Callable[[TabletStartTarget, str], None],
    ) -> None:
        self.select_target = select_target
        self.report = report
        self.confirm = confirm
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="G2TabletPrestart", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def _run(self) -> None:
        target: TabletStartTarget | None = None
        process: subprocess.Popen[bytes] | None = None
        pending = b""
        try:
            while not self.stop_event.is_set():
                selected = self.select_target()
                if selected != target:
                    self._stop_process(process)
                    if process is not None and process.stdout is not None:
                        process.stdout.close()
                    process = None
                    pending = b""
                    target = selected
                if target is None:
                    self.stop_event.wait(0.5)
                    continue
                if process is None:
                    try:
                        process = subprocess.Popen(
                            target.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                        )
                        self.report(target, "listening")
                    except OSError as error:
                        logger.warning("G2 tablet start watcher unavailable: %s", error)
                        self.report(target, "unavailable")
                        self.stop_event.wait(5.0)
                        continue
                assert process.stdout is not None
                readable, _, _ = select.select([process.stdout], [], [], 0.5)
                if not readable:
                    if process.poll() is not None:
                        self.report(target, "unavailable")
                        process.stdout.close()
                        process = None
                        self.stop_event.wait(5.0)
                    continue
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    self.report(target, "unavailable")
                    process.stdout.close()
                    process = None
                    self.stop_event.wait(5.0)
                    continue
                pending += chunk
                if len(pending) > 16_384:
                    pending = pending[-4096:]
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line.startswith(EVENT_PREFIX.encode()):
                        continue
                    try:
                        payload = json.loads(line[len(EVENT_PREFIX):])
                        event = payload.get("event")
                    except (ValueError, AttributeError):
                        continue
                    if event not in TABLET_START_EVENTS:
                        continue
                    if event == "detected":
                        record_name = payload.get("record_name")
                        if not isinstance(record_name, str) or not RECORD_NAME.fullmatch(record_name):
                            self.report(target, "rejected")
                            continue
                        self._stop_process(process)
                        process.stdout.close()
                        process = None
                        self.report(target, "detected")
                        if not self.stop_event.is_set():
                            self.confirm(target, record_name)
                        target = None
                        self.stop_event.wait(0.5)
                        break
                    self.report(target, event)
        except Exception:
            logger.exception("G2 tablet start bridge stopped after an unexpected error")
        finally:
            self._stop_process(process)
            if process is not None and process.stdout is not None:
                process.stdout.close()
