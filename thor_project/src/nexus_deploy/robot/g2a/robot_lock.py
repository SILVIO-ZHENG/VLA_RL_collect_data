"""Cross-process lock for exclusive access to the robot GDK runtime."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO

DEFAULT_ROBOT_LOCK_PATH = Path("/tmp/g2a_pi05_robot_gdk.lock")


class RobotProcessLock:
    def __init__(self, path: Path = DEFAULT_ROBOT_LOCK_PATH) -> None:
        self.path = path
        self._stream: IO[str] | None = None

    @property
    def is_acquired(self) -> bool:
        return self._stream is not None

    def acquire(self) -> None:
        if self._stream is not None:
            raise RuntimeError(f"Robot process lock is already acquired: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.close()
            raise RuntimeError(
                "Another G2A process is already using the robot GDK. "
                "Stop the deployment/read-only/reset process before continuing."
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        self._stream = stream

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            self._stream = None

    def __enter__(self) -> RobotProcessLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
