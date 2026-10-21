"""Standalone HTTP/WebSocket service for workspace alignment."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import importlib
import inspect
import ipaddress
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from nexus_deploy.safety_profiles import (
    AlignmentSafetyProfileRepository,
    SafetyProfileGenerationError,
)

from .adapters import AlignmentRobotAdapter
from .config import AlignmentConfig, config_fingerprint, load_config
from .core import AlignmentEngine, AlignmentError, AlignmentState
from .detectors import AprilTagDetector
from .surfaces import SurfaceCalibrationRepository
from .targets import TargetRepository

STATIC_DIRECTORY = Path(__file__).with_name("static")
MAX_JSON_BYTES = 10 * 1024 * 1024


def _load_plugin(target: str, settings: Mapping[str, Any], expected_type: type[Any]) -> Any:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError(f"plugin must use module:object syntax: {target}")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError(f"plugin is not callable: {target}")
    parameters = inspect.signature(factory).parameters
    instance = factory(dict(settings)) if parameters else factory()
    if not isinstance(instance, expected_type):
        raise TypeError(f"plugin {target} must implement {expected_type.__name__}")
    return instance


class AlignmentManager:
    def __init__(self, config: AlignmentConfig, *, admin_password: str = "") -> None:
        self.config = config
        self.admin_password = admin_password
        self.targets = TargetRepository(config.targets_directory)
        self.surfaces = SurfaceCalibrationRepository(config.surfaces_directory)
        self.safety_profiles = (
            AlignmentSafetyProfileRepository(config.safety_profiles_directory)
            if config.safety_profiles_directory is not None
            and config.safety_profile_template is not None
            else None
        )
        self.csrf_token = secrets.token_urlsafe(24)
        self.sessions: dict[str, AlignmentEngine] = {}
        self._lock = asyncio.Lock()
        self._viewers: dict[str, int] = {}
        self._release_tasks: dict[str, asyncio.Task[Any]] = {}
        self._target_lock = asyncio.Lock()

    def bootstrap(self) -> dict[str, Any]:
        return {
            "service": {
                "name": "G2A Workspace Alignment",
                "version": 1,
                "target_admin_enabled": bool(self.admin_password),
            },
            "csrf_token": self.csrf_token,
            "defaults": {
                "profile_id": self.config.default_profile,
                "adapter_id": self.config.default_adapter,
                "adjustment_mode": "table",
                "host": self.config.default_robot_host,
            },
            "profiles": [
                self._profile_public(profile)
                for profile in self.config.profiles.values()
            ],
            "adapters": [adapter.public_dict() for adapter in self.config.adapters.values()],
        }

    def _profile_public(self, profile: Any) -> dict[str, Any]:
        resolved = self.targets.resolve(profile)
        value = resolved.public_dict()
        surface = self.surfaces.active(resolved.id)
        measurement_id = str(surface.get("measurement_id", "")) if surface else ""
        safety_name = ""
        if measurement_id and self.safety_profiles is not None:
            for path in self.safety_profiles.list(self.config.safety_robot_id):
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                generated = document.get("generated_profile", {})
                if isinstance(generated, Mapping) and str(
                    generated.get("measurement_id", "")
                ) == measurement_id:
                    safety_name = path.name
                    break
        value["artifacts"] = {
            "target": {
                "configured": resolved.target_configured,
                "version": resolved.version if resolved.target_configured else None,
            },
            "surface": {
                "configured": surface is not None,
                "measurement_id": measurement_id or None,
                "height_m": (
                    surface.get("surface", {}).get("height_m") if surface else None
                ),
            },
            "safety_profile": {
                "generated": bool(safety_name),
                "filename": safety_name or None,
            },
        }
        return value

    async def create(self, payload: Mapping[str, Any]) -> AlignmentEngine:
        host = str(payload.get("host", "")).strip()
        if not host:
            raise AlignmentError("robot host is required")
        profile_id = str(payload.get("profile_id", self.config.default_profile))
        adapter_id = str(payload.get("adapter_id", self.config.default_adapter))
        requested_detector = payload.get("detector_id")
        adapter_detector = self.config.adapters.get(adapter_id)
        detector_id = str(
            requested_detector
            or (adapter_detector.settings.get("detector_id") if adapter_detector else "")
            or self.config.default_detector
        )
        if profile_id not in self.config.profiles:
            raise AlignmentError("unknown alignment profile")
        if adapter_id not in self.config.adapters:
            raise AlignmentError("unknown robot adapter")
        if detector_id not in self.config.detectors:
            raise AlignmentError("unknown AprilTag detector")
        async with self._lock:
            for session in self.sessions.values():
                if session.host == host and session.state not in {
                    AlignmentState.COMPLETED,
                    AlignmentState.STOPPED,
                    AlignmentState.ERROR,
                }:
                    if session.profile.id == profile_id and session.adapter_id == adapter_id:
                        return session
                    raise AlignmentError("该机器人正在使用另一套对齐配置，请先结束当前对齐")
            adapter_config = self.config.adapters[adapter_id]
            detector_config = self.config.detectors[detector_id]
            adapter_settings = dict(adapter_config.settings)
            if payload.get("port") not in {None, ""}:
                port = int(payload["port"])
                if not 1 <= port <= 65535:
                    raise AlignmentError("robot adapter port must be in [1, 65535]")
                adapter_settings["port"] = port
            adapter = _load_plugin(adapter_config.plugin, adapter_settings, AlignmentRobotAdapter)
            detector = _load_plugin(detector_config.plugin, detector_config.settings, AprilTagDetector)
            profile = self.targets.resolve(self.config.profiles[profile_id])
            engine = await asyncio.to_thread(
                AlignmentEngine,
                self.config,
                profile,
                adapter,
                detector,
                host=host,
                adapter_id=adapter_id,
                detector_id=detector_id,
                adjustment_mode=str(payload.get("adjustment_mode", "table")),
            )
            self.sessions[engine.id] = engine
            self._viewers[engine.id] = 0
            self._schedule_release(engine.id)
            return engine

    async def publish_current_as_target(
        self,
        engine: AlignmentEngine,
        password: str,
    ) -> dict[str, Any]:
        if not self.admin_password:
            raise AlignmentError("标准位置设置功能尚未配置管理员密码")
        if not password or not hmac.compare_digest(password, self.admin_password):
            raise AlignmentError("管理员密码错误")
        if bool(engine.health.get("task_running", False)):
            raise AlignmentError("机器人正在执行任务，不能设置标准位置")
        async with self._target_lock:
            pose = engine.capture_standard_pose(minimum_stable_s=2.0)
            calibration_version = engine.calibration.version if engine.calibration else "unknown"
            document = await asyncio.to_thread(
                self.targets.publish,
                engine.profile,
                pose,
                robot=engine.robot,
                robot_host=engine.host,
                calibration_version=calibration_version,
            )
            profile = self.targets.resolve(self.config.profiles[engine.profile.id])
            engine.apply_target_profile(profile)
            return {
                "target": {
                    "profile_id": document["profile_id"],
                    "version": document["version"],
                    "created_at": document["created_at"],
                    "expected_table_pose": document["expected_table_pose"],
                    "robot": document["robot"],
                    "calibration_version": document["calibration_version"],
                },
                "session": engine.snapshot(),
            }

    async def publish_current_surface(self, engine: AlignmentEngine) -> dict[str, Any]:
        """Persist one immutable, deployment-consumable table measurement."""
        measurement = await asyncio.to_thread(engine.capture_surface_measurement, 2.0)
        calibration_version = engine.calibration.version if engine.calibration else "unknown"
        document = await asyncio.to_thread(
            self.surfaces.publish,
            engine.profile,
            measurement,
            robot=engine.robot,
            robot_host=engine.host,
            calibration_version=calibration_version,
            config_fingerprint=config_fingerprint(self.config),
        )
        result = dict(document)
        if self.safety_profiles is not None and self.config.safety_profile_template is not None:
            active_path = (
                self.config.surfaces_directory / engine.profile.id / "active.json"
            )
            try:
                generated = await asyncio.to_thread(
                    self.safety_profiles.generate,
                    measurement=document,
                    measurement_path=active_path,
                    robot_id=self.config.safety_robot_id,
                    template_path=self.config.safety_profile_template,
                    expected_robot_aid=self.config.safety_expected_aid,
                    allowed_robot_hosts={engine.host},
                )
            except (OSError, TypeError, ValueError, SafetyProfileGenerationError) as error:
                result["safety_profile"] = {
                    "generated": False,
                    "error": str(error),
                }
            else:
                result["safety_profile"] = {
                    "generated": True,
                    "id": generated.name,
                }
        return result

    def get(self, session_id: str) -> AlignmentEngine:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise AlignmentError("alignment session does not exist") from error

    def check_control(self, engine: AlignmentEngine, token: str | None) -> None:
        if not token or not hmac.compare_digest(token, engine.control_token):
            raise AlignmentError("alignment control token is invalid")

    def check_view(self, engine: AlignmentEngine, token: str | None) -> None:
        if not token or not (
            hmac.compare_digest(token, engine.viewer_token)
            or hmac.compare_digest(token, engine.control_token)
        ):
            raise AlignmentError("alignment viewer token is invalid")

    def viewer_joined(self, session_id: str) -> None:
        self._viewers[session_id] = self._viewers.get(session_id, 0) + 1
        task = self._release_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()

    def viewer_left(self, session_id: str) -> None:
        self._viewers[session_id] = max(0, self._viewers.get(session_id, 1) - 1)
        if self._viewers[session_id] == 0:
            self._schedule_release(session_id)

    def _schedule_release(self, session_id: str) -> None:
        previous = self._release_tasks.pop(session_id, None)
        if previous is not None:
            previous.cancel()
        self._release_tasks[session_id] = asyncio.create_task(
            self._release_after_grace(session_id)
        )

    async def _release_after_grace(self, session_id: str) -> None:
        try:
            await asyncio.sleep(self.config.reconnect_grace_s)
            engine = self.sessions.get(session_id)
            if (
                engine is not None
                and self._viewers.get(session_id, 0) == 0
                and engine.state
                in {
                    AlignmentState.READY,
                    AlignmentState.SEARCHING,
                    AlignmentState.UNSTABLE,
                    AlignmentState.ADJUSTING,
                    AlignmentState.ALIGNED,
                    AlignmentState.CONNECTION_LOST,
                }
            ):
                await asyncio.to_thread(engine.stop, "no_viewers")
        except asyncio.CancelledError:
            return
        finally:
            if self._release_tasks.get(session_id) is asyncio.current_task():
                self._release_tasks.pop(session_id, None)

    async def close(self) -> None:
        for task in self._release_tasks.values():
            task.cancel()
        await asyncio.gather(*(asyncio.to_thread(session.close) for session in self.sessions.values()))


def create_app(
    config: AlignmentConfig,
    *,
    host: str,
    port: int,
    username: str = "nexus",
    password: str = "",
    admin_password: str = "",
):
    try:
        from aiohttp import web
    except ImportError as error:
        raise RuntimeError("Install alignment dependencies: pip install -e '.[alignment]'") from error

    manager = AlignmentManager(config, admin_password=admin_password)
    expected_host = f"{host}:{port}".lower()
    authorization = None
    if password:
        authorization = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()

    @web.middleware
    async def security(request: Any, handler: Any):
        request_host = request.headers.get("Host", "").lower()
        if request_host not in {expected_host, host.lower()}:
            return web.json_response({"error": "Host 不受信任"}, status=403)
        origin = request.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != request_host:
                return web.json_response({"error": "Origin 不受信任"}, status=403)
        if authorization is not None and not hmac.compare_digest(
            request.headers.get("Authorization", ""), authorization
        ):
            return web.Response(status=401, headers={"WWW-Authenticate": 'Basic realm="Nexus", charset="UTF-8"'})
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except AlignmentError as error:
            return web.json_response({"error": str(error)}, status=409)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            return web.json_response({"error": str(error)}, status=500)

    app = web.Application(middlewares=[security], client_max_size=MAX_JSON_BYTES)
    app["manager"] = manager

    async def bootstrap(_request: Any):
        return web.json_response(manager.bootstrap())

    async def create_session(request: Any):
        _check_csrf(request, manager)
        engine = await manager.create(await request.json())
        return web.json_response(engine.snapshot(include_token=True), status=201)

    async def get_session(request: Any):
        return web.json_response(manager.get(request.match_info["session_id"]).snapshot())

    async def start_session(request: Any):
        engine = _controlled_engine(request, manager)
        await asyncio.to_thread(engine.start)
        return web.json_response(engine.snapshot())

    async def recheck_session(request: Any):
        engine = _controlled_engine(request, manager)
        await asyncio.to_thread(engine.recheck)
        return web.json_response(engine.snapshot())

    async def update_mode(request: Any):
        engine = _controlled_engine(request, manager)
        body = await request.json()
        engine.set_adjustment_mode(str(body.get("adjustment_mode", "")))
        return web.json_response(engine.snapshot())

    async def complete_session(request: Any):
        engine = _controlled_engine(request, manager)
        body = await request.json()
        surface = await manager.publish_current_surface(engine)
        report = await asyncio.to_thread(
            engine.complete,
            body.get("overlay_data_url"),
            surface_calibration=surface,
        )
        return web.json_response(report)

    async def active_surface(request: Any):
        profile_id = str(request.match_info["profile_id"])
        if profile_id not in manager.config.profiles:
            raise AlignmentError("unknown alignment profile")
        value = manager.surfaces.active(profile_id)
        if value is None:
            raise web.HTTPNotFound(text="surface calibration does not exist")
        return web.json_response(value)

    async def stop_session(request: Any):
        engine = _controlled_engine(request, manager)
        body = await request.json()
        snapshot = await asyncio.to_thread(engine.stop, str(body.get("reason", "user_cancelled")))
        return web.json_response(snapshot)

    async def publish_target(request: Any):
        engine = _controlled_engine(request, manager)
        body = await request.json()
        result = await manager.publish_current_as_target(
            engine,
            str(body.get("admin_password", "")),
        )
        return web.json_response(result, status=201)

    async def frame(request: Any):
        value = manager.get(request.match_info["session_id"]).frame()
        return web.Response(
            body=value.data,
            content_type=value.mime_type,
            headers={"Cache-Control": "no-store", "X-Frame-Sequence": str(value.sequence)},
        )

    async def events(request: Any):
        engine = manager.get(request.match_info["session_id"])
        manager.check_view(engine, request.query.get("token"))
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        manager.viewer_joined(engine.id)
        try:
            while not ws.closed:
                try:
                    await ws.send_json(engine.snapshot())
                except (ConnectionError, RuntimeError):
                    break
                if engine.state in {AlignmentState.COMPLETED, AlignmentState.STOPPED}:
                    break
                await asyncio.sleep(0.12)
        finally:
            manager.viewer_left(engine.id)
            await ws.close()
        return ws

    async def index(_request: Any):
        return web.FileResponse(STATIC_DIRECTORY / "alignment.html", headers=_static_headers())

    async def static_asset(request: Any):
        name = request.match_info["name"]
        if name not in {"alignment.css", "target-admin.css", "visibility.css", "alignment.js"}:
            raise web.HTTPNotFound()
        return web.FileResponse(STATIC_DIRECTORY / name, headers=_static_headers())

    async def on_shutdown(_app: Any) -> None:
        await manager.close()

    app.router.add_get("/", index)
    app.router.add_get("/alignment", index)
    app.router.add_get("/assets/{name}", static_asset)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_post("/api/sessions", create_session)
    app.router.add_get("/api/sessions/{session_id}", get_session)
    app.router.add_post("/api/sessions/{session_id}/start", start_session)
    app.router.add_post("/api/sessions/{session_id}/recheck", recheck_session)
    app.router.add_post("/api/sessions/{session_id}/mode", update_mode)
    app.router.add_post("/api/sessions/{session_id}/complete", complete_session)
    app.router.add_post("/api/sessions/{session_id}/stop", stop_session)
    app.router.add_post("/api/sessions/{session_id}/target", publish_target)
    app.router.add_get("/api/sessions/{session_id}/frame", frame)
    app.router.add_get("/api/sessions/{session_id}/events", events)
    app.router.add_get("/api/surfaces/{profile_id}/active", active_surface)
    app.on_shutdown.append(on_shutdown)
    return app


def _static_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self' ws: wss:; frame-ancestors 'none'"
        ),
    }


def _check_csrf(request: Any, manager: AlignmentManager) -> None:
    supplied = request.headers.get("X-CSRF-Token", "")
    if not supplied or not hmac.compare_digest(supplied, manager.csrf_token):
        raise AlignmentError("CSRF token is invalid")


def _controlled_engine(request: Any, manager: AlignmentManager) -> AlignmentEngine:
    _check_csrf(request, manager)
    engine = manager.get(request.match_info["session_id"])
    manager.check_control(engine, request.headers.get("X-Alignment-Control-Token"))
    return engine


def _validate_bind(host: str, allow_unauthenticated_private_lan: bool, password: str) -> None:
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("--host must be localhost or a specific IP address") from error
    if address.is_unspecified or address.is_multicast or not (address.is_loopback or address.is_private):
        raise ValueError("alignment UI may only bind to loopback or one private IP")
    if not address.is_loopback and not password and not allow_unauthenticated_private_lan:
        raise ValueError("private-LAN binding requires a password or explicit unauthenticated opt-in")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("alignment.toml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--username", default="nexus")
    parser.add_argument("--password-env", default="ALIGNMENT_UI_PASSWORD")
    parser.add_argument("--admin-password-env", default="ALIGNMENT_ADMIN_PASSWORD")
    parser.add_argument("--allow-unauthenticated-private-lan", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    password = os.environ.get(args.password_env, "")
    admin_password = os.environ.get(args.admin_password_env, "")
    if not password and args.password_env == "ALIGNMENT_UI_PASSWORD":
        password = os.environ.get("NEXUS_UI_PASSWORD", "")
    _validate_bind(args.host, args.allow_unauthenticated_private_lan, password)
    config = load_config(args.config)
    from aiohttp import web

    print(f"Alignment UI: http://{args.host}:{args.port}/alignment")
    print("Read-only service: no robot motion or inference command is available")
    print(
        "Set-current-as-standard: enabled"
        if admin_password
        else f"Set-current-as-standard: disabled (set {args.admin_password_env})"
    )
    web.run_app(
        create_app(
            config,
            host=args.host,
            port=args.port,
            username=args.username,
            password=password,
            admin_password=admin_password,
        ),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
