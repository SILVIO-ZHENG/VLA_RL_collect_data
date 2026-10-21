"""Launch one shared checkout using an explicit machine profile."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import tomllib
from pathlib import Path


def plan(root: Path, site: str, mode: str, *, sorting_config: Path | None = None):
    config = root / "configs" / f"{site}.toml"
    data = tomllib.loads(config.read_text())
    host = data["host"]
    if platform.machine() != host["architecture"]:
        raise ValueError(f"Profile {site} requires {host['architecture']}; this host is {platform.machine()}")
    configured_python = Path(host["web_python"]).expanduser()
    candidate_python = configured_python if configured_python.is_absolute() else config.parent / configured_python
    python = Path(os.path.normpath(candidate_python.absolute()))
    if not python.is_file():
        raise ValueError(f"Web interpreter is missing: {python}")
    command = [str(python), "-m", "nexus_deploy.control_api", "--config", str(config)]
    ports = []
    if mode == "simulate":
        command += ["--host", "127.0.0.1", "--port", "21865", "--disable-integrated-simulator"]
        ports.append(("127.0.0.1", 21865))
    else:
        command += ["--host", host["control_host"], "--port", str(host["control_port"]),
                    "--robot-telemetry-url", f"tcp://{host['telemetry_host']}:{host['telemetry_port']}",
                    "--robot-telemetry-bind-host", host["telemetry_host"], "--enable-actions",
                    "--allow-unauthenticated-private-lan"]
        ports += [(host["control_host"], host["control_port"]),
                  (host["telemetry_host"], host["telemetry_port"])]
        if mode == "execute":
            command += ["--enable-execute", "--enable-reset"]
    if sorting_config:
        command += ["--sorting-config", str(sorting_config.resolve())]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return data, command, ports, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=("5090", "thor"), required=True)
    parser.add_argument("--mode", choices=("simulate", "observe", "execute"), default="simulate")
    parser.add_argument("--check", action="store_true", help="Inspect configuration only; start no service or robot")
    parser.add_argument("--sorting-config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data, command, ports, environment = plan(root, args.site, args.mode, sorting_config=args.sorting_config)
    if args.check:
        config_directory = root / "configs"
        def config_path(value: str) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else config_directory / path

        print(json.dumps({"site": args.site, "architecture": platform.machine(), "root": str(root),
                          "command": command, "hold_feedback_max_age_ms": data["runtime"]["stationary_hold_feedback_max_age_ms"],
                          "presence_weights_available": config_path(data["presence_runtime"]["weights"]).is_file(),
                          "model_interpreters_available": {key: config_path(value["python_executable"]).is_file()
                                                           for key, value in data["models"].items()}}, indent=2))
        return
    # Never replace a running control stack, and fail before launching children.
    for address in ports:
        with socket.socket() as sock:
            try:
                # Permit a clean restart while prior HTTP connections remain
                # in TIME_WAIT; an active listener still makes bind fail.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(address)
            except OSError as error:
                raise SystemExit(f"Required listener {address} unavailable; existing service was left running: {error}")
    os.chdir(root)
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
