"""Operator, scheduler and internal HTTP routing for one control process."""

from __future__ import annotations

import base64
import hmac
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from nexus_deploy.control.errors import TaskError, ValidationError
from nexus_deploy.sorting_api import SortingController, SortingError

STATIC_DIRECTORY = Path(__file__).resolve().parents[1] / "static"
MAX_REQUEST_BYTES = 256 * 1024


class NexusHTTPServer(ThreadingHTTPServer):
    """Suppress expected browser disconnect noise without hiding server faults."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _handler_factory(
    plane: Any,
    *,
    port: int | None = None,
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
    basic_credentials: tuple[str, str] | None = None,
    sorting: SortingController | None = None,
) -> type[BaseHTTPRequestHandler]:
    expected_hosts = {f"{host.lower()}:{port}" if port is not None else host.lower() for host in allowed_hosts}
    expected_hosts.update({"127.0.0.1:18765", "localhost:18765"})
    expected_authorization = None
    if basic_credentials is not None:
        raw_credentials = f"{basic_credentials[0]}:{basic_credentials[1]}".encode()
        expected_authorization = "Basic " + base64.b64encode(raw_credentials).decode("ascii")

    class ControlHandler(BaseHTTPRequestHandler):
        server_version = "NexusPlatform/1.0"
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def do_GET(self) -> None:  # noqa: N802
            if self._sorting_request():
                return
            if not self._request_allowed():
                return
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._landing_redirect()
                    return
                if path == "/api/csrf-token":
                    # The same Host/Origin/auth checks as other UI reads apply.
                    # No GPU query, robot probe, admission change or action.
                    self._json(HTTPStatus.OK, {"csrf_token": plane.csrf_token})
                    return
                if path == "/api/bootstrap":
                    self._json(HTTPStatus.OK, plane.bootstrap())
                    return
                if path == "/api/workstation":
                    self._json(200, plane.tasks.state())
                    return
                if path == "/api/tasks":
                    self._json(200, {"tasks": plane.tasks.list_tasks()})
                    return
                if path == "/api/profiles":
                    self._json(200, {"profiles": plane.tasks.profile_views()})
                    return
                if re.fullmatch(r"/api/tasks/[0-9a-f]{32}", path):
                    self._json(200, plane.tasks.view(path.rsplit("/", 1)[1]))
                    return
                if path == "/api/gpus":
                    self._json(HTTPStatus.OK, plane.gpu_status())
                    return
                if path.startswith("/api/deployments/"):
                    parts = [part for part in path.split("/") if part]
                    if len(parts) == 3:
                        self._json(HTTPStatus.OK, plane.get_job(parts[2]))
                        return
                    if len(parts) == 4 and parts[3] == "control-telemetry":
                        self._json(HTTPStatus.OK, plane.get_control_telemetry(parts[2]))
                        return
                    if len(parts) == 5 and parts[3] == "cameras":
                        mime_type, content = plane.read_camera(parts[2], parts[4])
                        self._binary(HTTPStatus.OK, mime_type, content)
                        return
                self._static(path)
            except (ValidationError, TaskError) as error:
                self._json(
                    error.status,
                    {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                )
            except Exception as error:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                )

        def do_HEAD(self) -> None:  # noqa: N802
            if not self._request_allowed():
                return
            path = urlparse(self.path).path
            if path == "/":
                self._landing_redirect()
                return
            target = self._static_target(path)
            if STATIC_DIRECTORY.resolve() not in target.parents or not target.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "页面不存在"}, head=True)
                return
            self.send_response(HTTPStatus.OK)
            self._static_headers(target, target.stat().st_size)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self._sorting_request():
                return
            path = urlparse(self.path).path
            presence_gate_match = re.fullmatch(r"/api/deployments/([0-9a-f]{32})/presence-gate", path)
            if presence_gate_match is not None:
                # This endpoint is for the local sidecar only.  Browser requests
                # are refused before JSON is interpreted, and the sidecar holds
                # a separate short-lived token rather than the browser CSRF
                # value or robot telemetry credential.
                if not self._host_allowed():
                    return
                if self.headers.get("Origin"):
                    self._json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "包裹检测上报不接受浏览器 Origin"},
                    )
                    return
                try:
                    payload = self._request_json()
                    result = plane.ingest_presence_gate(
                        presence_gate_match.group(1),
                        payload,
                        self.headers.get("X-Nexus-Presence-Token"),
                    )
                    self._json(HTTPStatus.OK, result)
                except (ValidationError, TaskError) as error:
                    self._json(
                        error.status,
                        {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                    )
                except Exception as error:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                    )
                return
            presence_gate_match = re.fullmatch(r"/api/deployments/([0-9a-f]{32})/presence-gate-shadow", path)
            if presence_gate_match is not None:
                if not self._host_allowed():
                    return
                if self.headers.get("Origin"):
                    self._json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "包裹检测遥测不接受浏览器 Origin"},
                    )
                    return
                try:
                    payload = self._request_json()
                    result = plane.ingest_presence_gate_shadow(
                        presence_gate_match.group(1),
                        payload,
                        self.headers.get("X-Nexus-Telemetry-Token"),
                    )
                    self._json(HTTPStatus.OK, result)
                except (ValidationError, TaskError) as error:
                    self._json(
                        error.status,
                        {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                    )
                except Exception as error:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                    )
                return
            telemetry_match = re.fullmatch(r"/api/deployments/([0-9a-f]{32})/robot-telemetry", path)
            if telemetry_match is not None:
                if not self._host_allowed():
                    return
                if self.headers.get("Origin"):
                    self._json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "机器人遥测不接受浏览器 Origin"},
                    )
                    return
                try:
                    payload = self._request_json()
                    result = plane.ingest_robot_telemetry(
                        telemetry_match.group(1),
                        payload,
                        self.headers.get("X-Nexus-Telemetry-Token"),
                    )
                    self._json(HTTPStatus.OK, result)
                except (ValidationError, TaskError) as error:
                    self._json(
                        error.status,
                        {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                    )
                except Exception as error:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                    )
                return
            if not self._request_allowed():
                return
            try:
                payload = self._request_json()
                if path == "/api/workstation/apply-priority":
                    job_id = payload.get("deployment_id")
                    runtime = payload.get("runtime")
                    if (set(payload) != {"deployment_id", "runtime"} or
                            not isinstance(job_id, str) or
                            not re.fullmatch(r"[0-9a-f]{32}", job_id) or
                            not isinstance(runtime, dict)):
                        raise TaskError("INVALID_REQUEST", "真机配置应用参数无效", 400)
                    self._json(HTTPStatus.ACCEPTED, plane.priority_apply_deployment(
                        job_id, runtime, self.headers.get("X-CSRF-Token")))
                    return
                if path == "/api/workstation/activate":
                    job_id = payload.get("deployment_id")
                    if set(payload) != {"deployment_id"} or not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
                        raise TaskError("INVALID_REQUEST", "deployment_id 无效", 400)
                    self._json(HTTPStatus.ACCEPTED, plane.tasks.activate(job_id, self.headers.get("X-CSRF-Token")))
                    return
                if path == "/api/workstation":
                    self._json(200, plane.tasks.configure(payload, self.headers.get("X-CSRF-Token")))
                    return
                if path == "/api/tasks":
                    plane._require_csrf(self.headers.get("X-CSRF-Token"))
                    result, created = plane.tasks.submit(payload, source="operator", identity="operator")
                    self._json(202 if created else 200, result)
                    return
                if re.fullmatch(r"/api/tasks/[0-9a-f]{32}/commands", path):
                    self._json(200, plane.tasks.command(path.split("/")[3], payload, self.headers.get("X-CSRF-Token")))
                    return
                if path == "/api/validate":
                    self._json(HTTPStatus.OK, plane.validate(payload))
                    return
                if path == "/api/robots/probe":
                    result = plane.probe_robot(payload, self.headers.get("X-CSRF-Token"))
                    self._json(HTTPStatus.OK, result)
                    return
                if path == "/api/robots/cleanup/prepare":
                    result = plane.prepare_robot_session_cleanup(payload, self.headers.get("X-CSRF-Token"))
                    self._json(HTTPStatus.OK, result)
                    return
                if path == "/api/robots/cleanup":
                    result = plane.cleanup_robot_session(payload, self.headers.get("X-CSRF-Token"))
                    self._json(HTTPStatus.OK, result)
                    return
                if path == "/api/deployments":
                    job = plane.create_deployment(payload, self.headers.get("X-CSRF-Token"))
                    self._json(HTTPStatus.ACCEPTED, job)
                    return
                parts = [part for part in path.split("/") if part]
                if (
                    len(parts) == 5
                    and parts[:2] == ["api", "deployments"]
                    and parts[3] == "simulator"
                    and parts[4] in {"sync", "view", "preset"}
                ):
                    token = self.headers.get("X-CSRF-Token")
                    if parts[4] == "sync":
                        result = plane.sync_simulator(parts[2], token)
                    elif parts[4] == "view":
                        result = plane.set_simulator_view(parts[2], payload.get("view"), token)
                    else:
                        result = plane.save_simulator_view(parts[2], payload.get("view"), payload.get("camera"), token)
                    self._json(
                        HTTPStatus.OK,
                        result,
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["api", "deployments"]:
                    token = self.headers.get("X-CSRF-Token")
                    if parts[3] == "start":
                        self._json(
                            HTTPStatus.ACCEPTED,
                            plane.start_deployment(parts[2], token),
                        )
                        return
                    if parts[3] == "configure":
                        self._json(
                            HTTPStatus.OK,
                            plane.configure_deployment(parts[2], payload, token),
                        )
                        return
                    if parts[3] == "pause":
                        self._json(HTTPStatus.OK, plane.pause_deployment(parts[2], token))
                        return
                    if parts[3] == "resume":
                        self._json(HTTPStatus.OK, plane.resume_deployment(parts[2], token))
                        return
                    if parts[3] == "stop":
                        self._json(HTTPStatus.OK, plane.stop_deployment(parts[2], token))
                        return
                    if parts[3] == "reconcile-stale-session":
                        self._json(
                            HTTPStatus.OK,
                            plane.reconcile_stale_robot_session(parts[2], token),
                        )
                        return
                    if parts[3] == "reset-robot":
                        self._json(
                            HTTPStatus.ACCEPTED,
                            plane.reset_robot_body(parts[2], token),
                        )
                        return
                    if parts[3] == "close-model":
                        self._json(HTTPStatus.OK, plane.close_model(parts[2], token))
                        return
                raise ValidationError("API 路由不存在", status=HTTPStatus.NOT_FOUND)
            except (ValidationError, TaskError) as error:
                self._json(
                    error.status,
                    {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                )
            except Exception as error:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                )

        def _sorting_request(self) -> bool:
            path = urlparse(self.path).path
            if not path.startswith("/api/sorting/"):
                return False
            # Close the connection even on auth errors with unread POST bodies.
            self.close_connection = True
            if not self._host_allowed():
                return True
            try:
                if sorting is None:
                    raise SortingError("Sorting API is disabled", 404)
                sorting.authorize(self.headers.get("Authorization"), self.headers.get("Origin"))
                if path == "/api/sorting/health" and self.command == "GET":
                    self._json(
                        200,
                        {
                            "service": "sorting",
                            "ok": sorting.last_error is None,
                            "worker_error": sorting.last_error,
                            "profiles": sorted(sorting.profiles),
                            "hardware_actions_enabled": plane.enable_actions,
                        },
                    )
                elif path == "/api/sorting/robot" and self.command == "GET":
                    self._json(200, sorting.robot())
                elif path == "/api/sorting/robot/reset" and self.command == "POST":
                    self._json(202, sorting.reset(self._request_json()))
                elif path == "/api/sorting/robot/reset" and self.command == "GET":
                    query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
                    if set(query) - {"command_id"} or any(len(values) != 1 for values in query.values()):
                        raise SortingError("Only one command_id query parameter is accepted", 400)
                    self._json(200, sorting.reset_status(query.get("command_id", [None])[0]))
                elif path == "/api/sorting/model" and self.command == "GET":
                    self._json(200, sorting.model())
                elif path == "/api/sorting/tasks" and self.command == "POST":
                    result, created = sorting.submit(self._request_json())
                    self._json(202 if created else 200, result)
                elif re.fullmatch(r"/api/sorting/tasks/[0-9a-f]{32}/stop", path) and self.command == "POST":
                    result, created = sorting.stop(path.split("/")[4], self._request_json())
                    self._json(202 if created else 200, result)
                elif re.fullmatch(r"/api/sorting/tasks/[0-9a-f]{32}", path) and self.command == "GET":
                    self._json(200, sorting.view(path.rsplit("/", 1)[1]))
                else:
                    raise SortingError("Sorting API route not found", 404)
            except (SortingError, ValidationError, TaskError) as error:
                self._json(
                    error.status,
                    {"error": str(error), "reason_code": getattr(error, "reason_code", "INVALID_REQUEST")},
                )
            except Exception:
                if self.command == "POST":
                    retry_field = "request_id" if path == "/api/sorting/tasks" else "command_id"
                    self._json(500, {"error": f"Operation failed; query status before retrying the SAME {retry_field}",
                                     "reason_code": "INTERNAL_ERROR", "retry_field": retry_field})
                else:
                    self._json(500, {"error": "Status query failed; retry this GET request",
                                     "reason_code": "INTERNAL_ERROR"})
            return True

        def _request_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValidationError("Content-Length 无效") from error
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValidationError("请求体为空或过大")
            try:
                value = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValidationError("请求不是有效 JSON") from error
            if not isinstance(value, dict):
                raise ValidationError("请求必须是 JSON 对象")
            return value

        def _request_allowed(self) -> bool:
            if not self._host_allowed():
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlparse(origin)
                if parsed.scheme != "http" or parsed.netloc.lower() not in expected_hosts:
                    self._json(HTTPStatus.FORBIDDEN, {"error": "Origin 不受信任"})
                    return False
            if expected_authorization is not None:
                provided = self.headers.get("Authorization", "")
                if not hmac.compare_digest(provided, expected_authorization):
                    self.send_response(HTTPStatus.UNAUTHORIZED)
                    self.send_header("WWW-Authenticate", 'Basic realm="Nexus", charset="UTF-8"')
                    self.send_header("Content-Length", "0")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("X-Frame-Options", "DENY")
                    self.end_headers()
                    return False
            return True

        def _host_allowed(self) -> bool:
            host = self.headers.get("Host", "").strip().lower()
            if host not in expected_hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "Host 不受信任"})
                return False
            return True

        def _landing_redirect(self) -> None:
            with plane._lock:
                target = plane.tasks.landing_url(plane.tasks.console_deployments())
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        @staticmethod
        def _static_target(path: str) -> Path:
            requested = "session.html" if path == "/monitor" else unquote(path).lstrip("/")
            return (STATIC_DIRECTORY / requested).resolve()

        def _static(self, path: str) -> None:
            target = self._static_target(path)
            if STATIC_DIRECTORY.resolve() not in target.parents or not target.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "页面不存在"})
                return
            content = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._static_headers(target, len(content))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _static_headers(self, target: Path, length: int) -> None:
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "frame-src 'self' http://*:8766 http://*:8767 http://*:18766 http://*:18767; frame-ancestors 'none'",
            )
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")

        def _json(self, status: int, value: Any, *, head: bool = False) -> None:
            content = (json.dumps(value, ensure_ascii=False) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _binary(self, status: int, mime_type: str, content: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ControlHandler
