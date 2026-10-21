"""Standalone G2A robot client entry point."""

import signal

from .client import main


def _interrupt(*_args) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        main()
    except KeyboardInterrupt:
        # SIGTERM is how the deployment control plane asks the remote client
        # to stop.  ``main`` has already completed its cleanup in this path;
        # do not print a crash-looking traceback for a normal operator stop.
        pass
