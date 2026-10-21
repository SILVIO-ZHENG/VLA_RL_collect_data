"""MeshCat scene for the Nexus integrated G2 digital twin."""

from __future__ import annotations

import multiprocessing
import re
import signal
import threading
from dataclasses import dataclass
from multiprocessing.connection import Connection
from urllib.parse import urlsplit

import meshcat
import numpy as np
from meshcat import geometry, transformations

from .model import G2Model


class ViewerError(RuntimeError):
    """MeshCat startup or scene initialization failed."""


WORK_SURFACE_SIZE_XY = (0.7, 1.6)
WORK_SURFACE_CENTER_XY = (0.65, 0.0)
WORK_SURFACE_VISUAL_GAP = 0.001
NEAR_CAMERA_TARGET = (0.25, 0.0, 0.62)
NEAR_CAMERA_POSITION = (1.8, 0.18, 0.0)
OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.55)
OVERVIEW_CAMERA_POSITION = (2.7, 0.75, 0.0)
NEAR_SCENE_Z_OFFSET = -0.70
OVERVIEW_SCENE_Z_OFFSET = -0.86


def _bridge_worker(host: str, port: int, connection: Connection) -> None:
    try:
        import asyncio

        import tornado.iostream
        import tornado.websocket
        from meshcat.servers.zmqserver import WebSocketHandler, ZMQWebSocketBridge

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        loop = asyncio.get_event_loop()
        default_handler = loop.get_exception_handler()

        def handle_exception(
            event_loop: asyncio.AbstractEventLoop, context: dict[str, object]
        ) -> None:
            exception = context.get("exception")
            if isinstance(
                exception,
                (tornado.iostream.StreamClosedError, tornado.websocket.WebSocketClosedError),
            ):
                return
            if default_handler is not None:
                default_handler(event_loop, context)
            else:
                event_loop.default_exception_handler(context)

        loop.set_exception_handler(handle_exception)

        def same_host_origin(handler: WebSocketHandler, origin: str) -> bool:
            origin_host = urlsplit(origin).hostname
            request_host = handler.request.host.partition(":")[0]
            return origin_host is not None and origin_host == request_host

        WebSocketHandler.check_origin = same_host_origin
        bridge = ZMQWebSocketBridge(host=host, port=port)
        connection.send({"ok": True, "zmq_url": bridge.zmq_url, "web_url": bridge.web_url})
        connection.close()
        bridge.run()
    except Exception as exc:  # noqa: BLE001 - startup error must reach parent
        try:
            connection.send({"ok": False, "error": repr(exc)})
            connection.close()
        finally:
            raise


@dataclass
class MeshcatBridge:
    process: multiprocessing.Process
    zmq_url: str
    web_url: str

    @classmethod
    def start(cls, host: str, port: int, timeout_s: float = 15.0) -> MeshcatBridge:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_bridge_worker,
            args=(host, port, child),
            name="nexus-integrated-meshcat",
            daemon=True,
        )
        process.start()
        child.close()
        if not parent.poll(timeout_s):
            process.terminate()
            process.join(timeout=2.0)
            raise ViewerError("内置 MeshCat 服务启动超时")
        message = parent.recv()
        parent.close()
        if not message.get("ok"):
            process.join(timeout=2.0)
            raise ViewerError(f"内置 MeshCat 服务启动失败: {message.get('error')}")
        zmq_url = str(message["zmq_url"])
        if host in {"0.0.0.0", "::"}:
            zmq_url = zmq_url.replace(host, "127.0.0.1")
        return cls(process=process, zmq_url=zmq_url, web_url=str(message["web_url"]))

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=3.0)


def _safe_path(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class MeshcatScene:
    def __init__(self, bridge: MeshcatBridge, model: G2Model) -> None:
        self.bridge = bridge
        self.model = model
        self._lock = threading.RLock()
        self.viewer = meshcat.Visualizer(zmq_url=bridge.zmq_url)
        self._link_paths: dict[str, str] = {}
        self._last_transforms: dict[str, np.ndarray] = {}
        self.viewer.delete()
        self._load_robot()
        self.set_work_surface(
            height=0.75,
            hard_margin=0.004,
            origin=(0.0, 0.0, 0.75),
            normal=(0.0, 0.0, 1.0),
            xy_bounds=None,
        )
        self._configure_scene()

    def _load_robot(self) -> None:
        for index, (node, link, vertices, faces, local, color) in enumerate(
            self.model.render_geometry_records()
        ):
            link_path = f"robot/links/{_safe_path(link)}"
            self._link_paths[link] = link_path
            path = f"{link_path}/geometry/{index:03d}-{_safe_path(node)}"
            self.viewer[path].set_object(
                geometry.TriangularMeshGeometry(vertices, faces),
                geometry.MeshPhongMaterial(
                    color=color,
                    shininess=38,
                    specular=0x5A5A5A,
                    reflectivity=0.35,
                ),
            )
            self.viewer[path].set_transform(local)
        transforms = self.model.link_transforms(set(self._link_paths))
        for link, transform in transforms.items():
            self.viewer[self._link_paths[link]].set_transform(transform)
        self._last_transforms = {name: value.copy() for name, value in transforms.items()}

    def _configure_scene(self) -> None:
        self.viewer["/Background"].set_property("top_color", [0.96, 0.97, 0.99])
        self.viewer["/Background"].set_property("bottom_color", [0.88, 0.91, 0.95])
        self.viewer["/Grid"].set_property("visible", False)
        self.viewer["/Axes"].set_property("visible", False)
        self.viewer["/Lights/AmbientLight"].set_property("intensity", 0.82)
        self.viewer["/Lights/PointLightNegativeX"].set_property("intensity", 0.58)
        self.viewer["/Lights/PointLightPositiveX"].set_property("intensity", 0.58)
        self.viewer["/Lights/FillLight"].set_property("intensity", 0.5)
        self.set_camera_view("near")

    def set_camera_view(self, view: str) -> None:
        if view == "near":
            target, position, fov, scene_z = (
                NEAR_CAMERA_TARGET,
                NEAR_CAMERA_POSITION,
                44,
                NEAR_SCENE_Z_OFFSET,
            )
        elif view == "overview":
            target, position, fov, scene_z = (
                OVERVIEW_CAMERA_TARGET,
                OVERVIEW_CAMERA_POSITION,
                50,
                OVERVIEW_SCENE_Z_OFFSET,
            )
        else:
            raise ViewerError(f"未知相机视图: {view}")
        with self._lock:
            transform = transformations.translation_matrix([0.0, 0.0, scene_z])
            self.viewer["robot"].set_transform(transform)
            self.viewer["work_surface"].set_transform(transform)
            self.viewer["/Cameras/default/rotated"].set_object(
                geometry.PerspectiveCamera(fov=fov, near=0.02, far=100.0)
            )
            self.viewer["/Cameras/default"].set_transform(
                transformations.translation_matrix(target)
            )
            self.viewer["/Cameras/default/rotated/<object>"].set_property(
                "position", list(position)
            )

    def update_robot(self) -> int:
        with self._lock:
            current = self.model.link_transforms(set(self._link_paths))
            changed = {
                name: value
                for name, value in current.items()
                if name not in self._last_transforms
                or not np.allclose(value, self._last_transforms[name], rtol=0.0, atol=1e-10)
            }
            for link, transform in changed.items():
                self.viewer[self._link_paths[link]].set_transform(transform)
                self._last_transforms[link] = transform.copy()
            return len(changed)

    def set_work_surface(
        self,
        *,
        height: float,
        hard_margin: float,
        origin: tuple[float, float, float] | list[float],
        normal: tuple[float, float, float] | list[float],
        xy_bounds: tuple[float, float, float, float] | list[float] | None,
    ) -> None:
        surface_origin = np.asarray(origin, dtype=float)
        surface_normal = np.asarray(normal, dtype=float)
        surface_normal /= np.linalg.norm(surface_normal)
        if xy_bounds is None:
            surface_size_xy = WORK_SURFACE_SIZE_XY
            center_xy = np.asarray(WORK_SURFACE_CENTER_XY, dtype=float)
        else:
            xmin, xmax, ymin, ymax = (float(value) for value in xy_bounds)
            surface_size_xy = (xmax - xmin, ymax - ymin)
            center_xy = np.asarray(((xmin + xmax) * 0.5, (ymin + ymax) * 0.5), dtype=float)
        surface_center = np.array(
            [
                center_xy[0],
                center_xy[1],
                surface_origin[2]
                - np.dot(surface_normal[:2], center_xy - surface_origin[:2])
                / surface_normal[2],
            ]
        )
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, surface_normal)
        cross_norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(z_axis, surface_normal), -1.0, 1.0))
        if cross_norm < 1e-12:
            rotation = np.eye(4)
        else:
            rotation = transformations.rotation_matrix(
                np.arccos(dot), cross / cross_norm
            )

        def surface_transform(offset: float) -> np.ndarray:
            result = rotation.copy()
            result[:3, 3] = surface_center + surface_normal * offset
            return result

        table_size = (*surface_size_xy, 0.04)
        margin_size = (*surface_size_xy, hard_margin)
        with self._lock:
            self.viewer["work_surface/table"].set_object(
                geometry.Box(table_size), geometry.MeshLambertMaterial(color=0x8F7968)
            )
            self.viewer["work_surface/table"].set_transform(
                surface_transform(-table_size[2] * 0.5)
            )
            self.viewer["work_surface/hard_margin"].set_object(
                geometry.Box(margin_size),
                geometry.MeshLambertMaterial(color=0xE08A34, transparent=True, opacity=0.28),
            )
            self.viewer["work_surface/hard_margin"].set_transform(
                surface_transform(WORK_SURFACE_VISUAL_GAP + hard_margin * 0.5)
            )

    def close(self) -> None:
        self.bridge.close()
