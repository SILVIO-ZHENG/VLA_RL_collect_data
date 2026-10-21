"""MeshCat 场景桥接；只接收离线 G2 模型的三角网格和 FK 变换。"""

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

from nexus_simulator.model import G2Model


class ViewerError(RuntimeError):
    """MeshCat 服务或场景初始化失败。"""


# G2 base_link: +X 向机器人正前方，+Y 向左，+Z 向上。
# 以下尺寸仅控制浏览器中的有限矩形外观；安全语义仍是 xy_bounds=null 的无限工作面。
# 桌面沿 +X（机器人正前方）的长度为 0.7 m。近端保持在 x=0.3 m，
# 因此缩短桌面不会把它整体推离机器人。
WORK_SURFACE_SIZE_XY = (0.7, 1.6)
WORK_SURFACE_CENTER_XY = (0.65, 0.0)
WORK_SURFACE_VISUAL_GAP = 0.001
# 当前经画面验收的唯一视角是双臂工作区。MeshCat 运行中替换 camera
# 会重置 OrbitControls target，因此在引入可原子设置 pose + target 的
# Three.js 视图前，不向用户暴露未验收的全景或夹爪特写。
NEAR_CAMERA_TARGET = (0.2, 0.0, 1.08)
NEAR_CAMERA_POSITION = (1.55, 0.32, 0.0)
# 整机远景沿用 MeshCat 稳定的世界原点轨道中心，并通过更远的距离和
# 更宽的视场覆盖头部、双臂、工作面与底盘。近景参数保持不变。
OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.55)
OVERVIEW_CAMERA_POSITION = (2.5, 0.9, 0.0)
OVERVIEW_SCENE_Z_OFFSET = -0.86


def work_surface_layout(
    *, height: float, hard_margin: float
) -> tuple[tuple[float, float, float], list[float], tuple[float, float, float], list[float]]:
    table_size = (*WORK_SURFACE_SIZE_XY, 0.04)
    table_center = [
        WORK_SURFACE_CENTER_XY[0],
        WORK_SURFACE_CENTER_XY[1],
        height - table_size[2] * 0.5,
    ]
    margin_size = (*WORK_SURFACE_SIZE_XY, hard_margin)
    margin_center = [
        WORK_SURFACE_CENTER_XY[0],
        WORK_SURFACE_CENTER_XY[1],
        height + WORK_SURFACE_VISUAL_GAP + hard_margin * 0.5,
    ]
    return table_size, table_center, margin_size, margin_center


def changed_link_transforms(
    current: dict[str, np.ndarray], previous: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """只保留发生变化的 link，避免暂停/保持时重复广播整个机器人。"""

    return {
        name: transform
        for name, transform in current.items()
        if name not in previous
        or not np.allclose(transform, previous[name], rtol=0.0, atol=1e-10)
    }


def _bridge_worker(host: str, port: int, connection: Connection) -> None:
    try:
        import asyncio

        import tornado.iostream
        import tornado.websocket
        from meshcat.servers.zmqserver import WebSocketHandler, ZMQWebSocketBridge

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        loop = asyncio.get_event_loop()
        default_exception_handler = loop.get_exception_handler()

        def handle_loop_exception(
            event_loop: asyncio.AbstractEventLoop, context: dict[str, object]
        ) -> None:
            exception = context.get("exception")
            if isinstance(
                exception,
                (tornado.iostream.StreamClosedError, tornado.websocket.WebSocketClosedError),
            ):
                return
            if default_exception_handler is not None:
                default_exception_handler(event_loop, context)
            else:
                event_loop.default_exception_handler(context)

        loop.set_exception_handler(handle_loop_exception)

        def same_host_origin(handler: WebSocketHandler, origin: str) -> bool:
            """Allow the project-owned viewer bridge on a sibling local port."""

            origin_host = urlsplit(origin).hostname
            request_host = handler.request.host.partition(":")[0]
            return origin_host is not None and origin_host == request_host

        WebSocketHandler.check_origin = same_host_origin
        bridge = ZMQWebSocketBridge(host=host, port=port)
        connection.send({"ok": True, "zmq_url": bridge.zmq_url, "web_url": bridge.web_url})
        connection.close()
        bridge.run()
    except Exception as exc:  # noqa: BLE001 - child must report startup failures to parent
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
    port: int

    @classmethod
    def start(cls, host: str, port: int, timeout_s: float = 10.0) -> MeshcatBridge:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_bridge_worker,
            args=(host, port, child),
            name="nexus-meshcat-bridge",
            daemon=True,
        )
        process.start()
        child.close()
        if not parent.poll(timeout_s):
            process.terminate()
            process.join(timeout=2.0)
            raise ViewerError("MeshCat 服务启动超时")
        message = parent.recv()
        parent.close()
        if not message.get("ok"):
            process.join(timeout=2.0)
            raise ViewerError(f"MeshCat 服务启动失败: {message.get('error')}")
        zmq_url = str(message["zmq_url"])
        if host in {"0.0.0.0", "::"}:
            zmq_url = zmq_url.replace(host, "127.0.0.1")
        return cls(process=process, zmq_url=zmq_url, web_url=str(message["web_url"]), port=port)

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
        self._last_link_transforms: dict[str, np.ndarray] = {}
        self.viewer.delete()
        self._load_robot()
        self.set_work_surface(height=0.75, hard_margin=0.01)
        self._configure_camera()

    def _load_robot(self) -> None:
        # yourdfpy 已按 DAE 内嵌材质拆分三角网格。逐材质发送可避免 MeshCat 的
        # Collada loader 丢失 effect/baseColorFactor，让浏览器稳定显示原始配色。
        for index, (node, link, vertices, faces, local_transform, color) in enumerate(
            self.model.linked_geometry_records()
        ):
            link_path = f"robot/links/{_safe_path(link)}"
            self._link_paths[link] = link_path
            path = f"{link_path}/geometry/{index:03d}-{_safe_path(node)}"
            mesh = geometry.TriangularMeshGeometry(vertices, faces)
            # Phong 高光能保留原始深色材质，同时让黑色夹爪的曲面和关节边界
            # 在远景中仍可辨认；这里不改变任何碰撞几何。
            material = geometry.MeshPhongMaterial(
                color=color,
                shininess=38,
                specular=0x5A5A5A,
                reflectivity=0.35,
            )
            self.viewer[path].set_object(mesh, material)
            self.viewer[path].set_transform(local_transform)
        transforms = self.model.link_transforms(set(self._link_paths))
        for link, transform in transforms.items():
            self.viewer[self._link_paths[link]].set_transform(transform)
        self._last_link_transforms = {
            link: transform.copy() for link, transform in transforms.items()
        }

    def _configure_camera(self) -> None:
        self.viewer["/Background"].set_property("top_color", [0.96, 0.97, 0.99])
        self.viewer["/Background"].set_property("bottom_color", [0.88, 0.91, 0.95])
        # 工作面已经提供空间参考；隐藏全局网格可减少背景线条对夹爪轮廓的干扰。
        self.viewer["/Grid"].set_property("visible", False)
        self.viewer["/Axes"].set_property("visible", False)
        self.viewer["/Lights/AmbientLight"].set_property("intensity", 0.82)
        self.viewer["/Lights/PointLightNegativeX"].set_property("intensity", 0.58)
        self.viewer["/Lights/PointLightPositiveX"].set_property("intensity", 0.58)
        self.viewer["/Lights/FillLight"].set_property("intensity", 0.5)
        self.set_camera_view("near")

    def set_camera_view(self, view: str) -> None:
        """应用双臂近景或可见整机的远景。"""

        if view == "near":
            target = NEAR_CAMERA_TARGET
            position = NEAR_CAMERA_POSITION
            fov = 44
            scene_z_offset = 0.0
        elif view == "overview":
            target = OVERVIEW_CAMERA_TARGET
            position = OVERVIEW_CAMERA_POSITION
            fov = 50
            scene_z_offset = OVERVIEW_SCENE_Z_OFFSET
        else:
            raise ViewerError(f"未知相机视图: {view}")

        with self._lock:
            # MeshCat 每次替换 PerspectiveCamera 都会把 OrbitControls 的观察中心
            # 重置到世界原点。远景仅对可视化根节点做等量平移，既能把整机放在
            # 画面中央，也不会改变 FK、碰撞检测或工作面安全参数。
            scene_transform = transformations.translation_matrix(
                [0.0, 0.0, scene_z_offset]
            )
            self.viewer["robot"].set_transform(scene_transform)
            self.viewer["work_surface"].set_transform(scene_transform)
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
            changed = changed_link_transforms(current, self._last_link_transforms)
            if not changed:
                return 0
            # MeshCat animations are transient browser state: replacing a one-frame
            # animation at telemetry rate can leave different links on different
            # clips, and a newly connected viewer falls back to the old base pose.
            # Persist authoritative transforms instead. The platform already limits
            # this display-only bridge to latest-only 8 Hz and this method only sends
            # links whose FK transform actually changed.
            for link, transform in changed.items():
                self.viewer[self._link_paths[link]].set_transform(transform)
                self._last_link_transforms[link] = transform.copy()
            return len(changed)

    def set_work_surface(self, *, height: float, hard_margin: float) -> None:
        with self._lock:
            table_size, table_center, margin_size, margin_center = work_surface_layout(
                height=height, hard_margin=hard_margin
            )
            table_transform = transformations.translation_matrix(table_center)
            self.viewer["work_surface/table"].set_object(
                geometry.Box(table_size),
                geometry.MeshLambertMaterial(color=0x8F7968),
            )
            self.viewer["work_surface/table"].set_transform(table_transform)

            margin_transform = transformations.translation_matrix(margin_center)
            self.viewer["work_surface/hard_margin"].set_object(
                geometry.Box(margin_size),
                geometry.MeshLambertMaterial(color=0xE08A34, transparent=True, opacity=0.28),
            )
            self.viewer["work_surface/hard_margin"].set_transform(margin_transform)

    def close(self) -> None:
        self.bridge.close()
