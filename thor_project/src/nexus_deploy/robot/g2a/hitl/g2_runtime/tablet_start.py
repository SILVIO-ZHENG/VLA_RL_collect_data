"""Read-only tablet recorder watcher for an approved Nexus task start."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from .supervisor import EpisodeCoordinator

EVENT_PREFIX = "NEXUS_TABLET_START:"
RECORD_ROOT = Path("/data/record")


class NewTabletRecordingDetector:
    """Require an observed empty recorder boundary before accepting a new episode."""

    def __init__(self) -> None:
        self.armed = False
        self.initialized = False
        self.last_status: str | None = None

    def observe(self, active: list[Path]) -> tuple[str, str | None] | None:
        if not self.initialized:
            self.initialized = True
            self.armed = not active
        elif not active:
            self.armed = True
        if len(active) > 1:
            status = "ambiguous"
        elif not self.armed:
            status = "awaiting_clear"
        elif active:
            return "detected", active[0].name
        else:
            status = "ready"
        if status != self.last_status:
            self.last_status = status
            return status, None
        return None


def main() -> None:
    detector = NewTabletRecordingDetector()
    disconnected = threading.Event()

    def watch_owner() -> None:
        try:
            sys.stdin.buffer.read(1)
        finally:
            disconnected.set()

    threading.Thread(target=watch_owner, name="NexusTabletStartOwner", daemon=True).start()
    while not disconnected.is_set():
        try:
            event = detector.observe(EpisodeCoordinator.list_active_recordings(RECORD_ROOT))
        except (OSError, ValueError) as error:
            event = ("unavailable", type(error).__name__)
        if event is not None:
            status, record_name = event
            print(EVENT_PREFIX + json.dumps({"event": status, "record_name": record_name}), flush=True)
            if status == "detected":
                return
        disconnected.wait(0.5)


if __name__ == "__main__":
    main()
