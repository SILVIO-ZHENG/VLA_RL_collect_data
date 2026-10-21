"""Command-line assembly for the Nexus control process."""

from __future__ import annotations

import argparse
import errno
import ipaddress
import os
import sys
import threading
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from nexus_deploy.deployment import Deployment
from nexus_deploy.control.http_api import NexusHTTPServer, _handler_factory
from nexus_deploy.control.telemetry import TelemetryTCPServer, telemetry_stream_handler
from nexus_deploy.viewer import SimulatorProcess
from nexus_deploy.sorting_api import SortingController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nexus deployment control plane")
    parser.add_argument("--config", type=Path, default=Path("deploy.toml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--optional-host", action="append", default=[],
        help="Additional explicit private IP; wait for its interface without blocking the primary listener.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--username", default="nexus")
    parser.add_argument(
        "--password-env",
        default="NEXUS_UI_PASSWORD",
        help="Environment variable containing the Basic Auth password for private-LAN binding.",
    )
    parser.add_argument(
        "--allow-unauthenticated-private-lan",
        action="store_true",
        help="Explicitly allow an unauthenticated UI on one specific private IP.",
    )
    parser.add_argument(
        "--enable-actions",
        action="store_true",
        help="Allow fixed model start and supervised robot sessions.",
    )
    parser.add_argument(
        "--enable-execute",
        action="store_true",
        help="Unlock physical execution where the robot configuration also permits it.",
    )
    parser.add_argument(
        "--enable-reset",
        action="store_true",
        help="Unlock the approved Dahuan gripper and body reset sequence.",
    )
    parser.add_argument(
        "--robot-telemetry-url",
        default=None,
        help="Robot-reachable HTTP(S) or TCP endpoint for latest-value telemetry.",
    )
    parser.add_argument(
        "--robot-telemetry-stream-port",
        type=int,
        default=8768,
        help="One-way token-authenticated latest-value telemetry stream port.",
    )
    parser.add_argument(
        "--robot-telemetry-bind-host",
        default=None,
        help="Local private IP used only by the TCP robot telemetry listener.",
    )
    parser.add_argument(
        "--simulator-control-url",
        default=None,
        help="Use an explicitly managed external simulator instead of the bundled one.",
    )
    parser.add_argument(
        "--disable-integrated-simulator",
        action="store_true",
        help="Disable the bundled G2 real-time renderer.",
    )
    parser.add_argument(
        "--simulator-viewer-port",
        type=int,
        default=8767,
        help="MeshCat viewer port exposed to the browser.",
    )
    parser.add_argument(
        "--alignment-url",
        default=None,
        help="Optional standalone workspace-alignment page.",
    )
    parser.add_argument(
        "--alignment-surfaces-directory",
        type=Path,
        default=None,
        help="Directory containing accepted workspace-alignment surfaces.",
    )
    parser.add_argument("--sorting-config", type=Path, default=None)
    return parser


def _control_hosts(primary: str, optional: list[str]) -> tuple[tuple[str, ...], bool]:
    """Validate every listener before constructing the control plane."""
    hosts = tuple(dict.fromkeys([primary, *optional]))
    all_loopback = True
    for host in hosts:
        if host == "localhost":
            continue
        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError("Control listeners must be localhost or specific private IPs") from error
        if address.is_unspecified or address.is_multicast or not (address.is_private or address.is_loopback):
            raise ValueError("The control UI may only bind to loopback or a specific private IP")
        all_loopback = all_loopback and address.is_loopback
    return hosts, all_loopback


def _optional_http_listener(address, handler, stop, initial_server=None):
    """Serve one explicit address, retrying only an absent interface address."""
    server = initial_server
    waiting_logged = False
    try:
        while not stop.is_set():
            if server is None:
                try:
                    server = NexusHTTPServer(address, handler)
                except OSError as error:
                    if error.errno != errno.EADDRNOTAVAIL:
                        raise  # Port/configuration errors are not missing Wi-Fi.
                    if not waiting_logged:
                        print(f"Optional Nexus UI waiting for interface: {address[0]}:{address[1]}", flush=True)
                        waiting_logged = True
                    stop.wait(1.0)
                    continue
                server.timeout = 0.5
                print(f"Nexus optional UI: http://{address[0]}:{address[1]}", flush=True)
            server.handle_request()
    finally:
        if server is not None:
            server.server_close()



def main() -> None:
    # Imported lazily so control_api can keep its historical module entrypoint
    # while process assembly remains separate from the ControlPlane class.
    from nexus_deploy.control_api import ControlPlane

    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.robot_telemetry_stream_port <= 65535:
        raise ValueError("control and telemetry stream ports must be in [1, 65535]")
    listen_hosts, is_loopback = _control_hosts(args.host, args.optional_host)
    password = os.environ.get(args.password_env, "")
    if not is_loopback and not password and not args.allow_unauthenticated_private_lan:
        raise ValueError(f"Private-LAN binding requires {args.password_env} or --allow-unauthenticated-private-lan")
    if args.enable_execute and not args.enable_actions:
        raise ValueError("--enable-execute requires --enable-actions")
    if args.enable_reset and not args.enable_actions:
        raise ValueError("--enable-reset requires --enable-actions")

    robot_telemetry_url = str(args.robot_telemetry_url or "").strip() or None
    if robot_telemetry_url is None and not is_loopback:
        robot_telemetry_url = f"tcp://{args.host}:{args.robot_telemetry_stream_port}"
    telemetry_stream_host: str | None = None
    telemetry_stream_port: int | None = None
    if robot_telemetry_url is not None:
        parsed_telemetry = urlparse(robot_telemetry_url)
        if parsed_telemetry.scheme not in {"http", "https", "tcp"} or not parsed_telemetry.netloc:
            raise ValueError("--robot-telemetry-url must be an absolute HTTP(S) or TCP URL")
        if parsed_telemetry.username or parsed_telemetry.password or parsed_telemetry.query:
            raise ValueError("--robot-telemetry-url must not contain credentials or a query")
        if parsed_telemetry.scheme == "tcp":
            if parsed_telemetry.path not in {"", "/"}:
                raise ValueError("TCP robot telemetry URL must not contain a base path")
            telemetry_stream_host = str(args.robot_telemetry_bind_host or parsed_telemetry.hostname or "").strip()
            try:
                telemetry_bind_address = ipaddress.ip_address(telemetry_stream_host)
            except ValueError as error:
                raise ValueError("--robot-telemetry-bind-host must be a specific private IP address") from error
            if (
                not telemetry_bind_address.is_private
                or telemetry_bind_address.is_unspecified
                or telemetry_bind_address.is_multicast
            ):
                raise ValueError("--robot-telemetry-bind-host must be a specific private IP address")
            telemetry_stream_port = parsed_telemetry.port or args.robot_telemetry_stream_port

    deployment = Deployment(args.config)
    alignment_surfaces_directory = (
        args.alignment_surfaces_directory.expanduser().resolve()
        if args.alignment_surfaces_directory is not None
        else (args.config.expanduser().resolve().parent / "runtime" / "alignment-surfaces")
    )
    configured_alignment = deployment.raw.get("alignment", {})
    if not isinstance(configured_alignment, Mapping):
        raise ValueError("[alignment] must be a TOML table")
    alignment_url = str(args.alignment_url or configured_alignment.get("url", "")).strip() or None
    if alignment_url is not None:
        parsed_alignment = urlparse(alignment_url)
        if parsed_alignment.scheme not in {"http", "https"} or not parsed_alignment.netloc:
            raise ValueError("--alignment-url must be an absolute HTTP(S) URL")
        if parsed_alignment.username or parsed_alignment.password or parsed_alignment.fragment:
            raise ValueError("--alignment-url must not contain credentials or a fragment")

    simulator: SimulatorProcess | None = None
    simulator_control_url = str(args.simulator_control_url or "").strip() or None
    if simulator_control_url is None and not args.disable_integrated_simulator:
        simulator = SimulatorProcess.start(
            python=Path(sys.executable),
            runtime_dir=deployment.runtime_dir,
            alignment_surfaces_directory=alignment_surfaces_directory,
            viewer_port=args.simulator_viewer_port,
        )
        simulator_control_url = simulator.control_url
        print(
            f"Nexus integrated simulator: {simulator.control_url} "
            f"(viewer :{simulator.viewer_port}, log {simulator.log_path})"
        )
    plane = ControlPlane(
        args.config,
        enable_actions=args.enable_actions,
        enable_execute=args.enable_execute,
        enable_reset=args.enable_reset,
        robot_telemetry_url=robot_telemetry_url,
        simulator_control_url=simulator_control_url,
        simulator_viewer_port=args.simulator_viewer_port,
        alignment_url=alignment_url,
        alignment_surfaces_directory=alignment_surfaces_directory,
    )
    sorting = SortingController.from_file(plane, args.sorting_config) if args.sorting_config else None
    telemetry_server: TelemetryTCPServer | None = None
    telemetry_thread: threading.Thread | None = None
    if telemetry_stream_port is not None:
        assert telemetry_stream_host is not None
        telemetry_server = TelemetryTCPServer(
            (telemetry_stream_host, telemetry_stream_port),
            telemetry_stream_handler(plane),
        )
        telemetry_thread = threading.Thread(
            target=telemetry_server.serve_forever,
            name="RobotTelemetryStream",
            daemon=True,
        )
        telemetry_thread.start()
        print(
            f"Nexus one-way robot telemetry: tcp://{telemetry_stream_host}:"
            f"{telemetry_stream_port} (display data only; no command channel)"
        )
    credentials = None if is_loopback or not password else (args.username, password)
    server = NexusHTTPServer(
        (args.host, args.port),
        _handler_factory(
            plane,
            port=args.port,
            allowed_hosts=(args.host,),
            basic_credentials=credentials,
            sorting=sorting,
        ),
    )
    optional_stop = threading.Event()
    optional_threads = []
    for host in listen_hosts[1:]:
        handler = _handler_factory(
            plane, port=args.port, allowed_hosts=(host,),
            basic_credentials=credentials, sorting=sorting,
        )
        # Detect an occupied port now; only an absent interface is optional.
        try:
            optional_server = NexusHTTPServer((host, args.port), handler)
        except OSError as error:
            if error.errno != errno.EADDRNOTAVAIL:
                raise
            optional_server = None
        if optional_server is not None:
            optional_server.timeout = 0.5
            print(f"Nexus optional UI: http://{host}:{args.port}", flush=True)
        thread = threading.Thread(
            target=_optional_http_listener,
            args=((host, args.port), handler, optional_stop, optional_server),
            name="NexusOptionalHTTP", daemon=True,
        )
        thread.start()
        optional_threads.append(thread)
    if sorting is not None:
        sorting.start()
    print(f"Nexus UI: http://{args.host}:{args.port}")
    if not is_loopback and credentials is None:
        print("Warning: unauthenticated private-LAN access is explicitly enabled")
    if not args.enable_actions:
        print("Safety simulation mode: no deployment command will be executed")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        optional_stop.set()
        for thread in optional_threads:
            thread.join(timeout=2.0)
        if sorting is not None:
            sorting.close()
        server.server_close()
        if telemetry_server is not None:
            telemetry_server.shutdown()
            telemetry_server.server_close()
        if telemetry_thread is not None:
            telemetry_thread.join(timeout=2.0)
        if simulator is not None:
            simulator.close()
