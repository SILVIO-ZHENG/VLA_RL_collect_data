"""智元 G2A 头部彩色相机适配层与 10 Hz 运行循环。

Adapters and a 10 Hz runtime loop for the Agibot G2A head color camera.

提供两种互斥的相机所有权模式：
1. G2HeadCameraSource：复用已有、已连接的 G2A 对象；不初始化/释放 GDK。
2. GDKHeadCameraSource：独立进程只创建 GDK Camera，并自行 init/release GDK。
Two mutually exclusive ownership modes are provided: reuse an existing G2A owner,
or let a standalone camera-only process own the GDK lifecycle.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from .estimator import ParcelStateEstimator
from .schema import StageResult


@dataclass(frozen=True)
class HeadFrame:
    """一张标准化 RGB 头部图像及其主机侧采集计时。

    One normalized RGB head frame with host-side capture timing.
    """

    image_rgb: np.ndarray  # H×W×3、uint8、RGB / H×W×3 uint8 RGB.
    # 当前GDK路径是在解码完成后记录的主机墙钟近似值，不是相机硬件时间戳。
    # Host wall-clock approximation after decoding, not a hardware sensor timestamp.
    capture_timestamp_ns: int
    capture_duration_ms: float  # 相机调用+解码耗时 / fetch-and-decode duration.


class HeadCameraSource(Protocol):
    """运行循环所需的最小相机协议。

    Minimal camera protocol required by the runtime loop.
    """

    def capture(self) -> HeadFrame: ...


class G2HeadCameraSource:
    """从调用方已经连接的公共 G2A 适配器读取 ``cam_head``。

    Read ``cam_head`` from an already connected public G2A adapter.

    本类不拥有 robot，不会 connect/disconnect，也不会发送动作。需要注意公共
    `capture_camera_frames()` 会按现有 G2A 实现同时读取三路相机。
    This class does not own, connect, disconnect, or command the robot. The current
    public capture method reads all three G2A cameras before returning the head frame.
    """

    def __init__(self, robot: Any):
        self.robot = robot

    def capture(self) -> HeadFrame:
        """获取头部 RGB 帧并保留 G2A 提供的时间戳。

        Capture the head RGB frame and preserve the timestamp supplied by G2A.
        """

        capture = self.robot.capture_camera_frames()
        if "cam_head" not in capture.frames:
            raise RuntimeError("G2A camera capture did not contain 'cam_head'")
        if "cam_head" not in capture.frame_wall_time_ns:
            raise RuntimeError("G2A camera capture did not timestamp 'cam_head'")
        image = capture.frames["cam_head"]
        if not isinstance(image, np.ndarray):
            raise TypeError("G2A cam_head frame must be a numpy.ndarray")
        return HeadFrame(
            image_rgb=image,
            capture_timestamp_ns=int(capture.frame_wall_time_ns["cam_head"]),
            capture_duration_ms=float(capture.capture_duration_ms),
        )


class GDKHeadCameraSource:
    """独立、仅相机的 GDK 数据源；永远不创建 Robot 对象。

    Standalone camera-only GDK source that never creates a Robot object.

    只在独立命令行中使用。如果当前进程已有 GDK/G2A 所有者，绝不能再次初始化
    GDK；应改用 :class:`G2HeadCameraSource` 复用已有连接。
    Use this only in the standalone CLI. If another component already owns GDK,
    wrap its connected G2A object with :class:`G2HeadCameraSource` instead.
    """

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 400,
        timeout_ms: int = 1000,
        gdk: Any | None = None,
    ):
        if width <= 0 or height <= 0 or timeout_ms <= 0:
            raise ValueError("GDK head-camera dimensions and timeout must be positive")
        # 延迟导入让普通开发电脑可以导入/测试本包，不强制安装机器人专属 GDK。
        # Lazy import keeps the package testable off-robot without the vendor GDK.
        if gdk is None:
            try:
                import agibot_gdk as gdk_module
            except ImportError as exc:
                raise ImportError(
                    "agibot_gdk is unavailable; run this command on the G2 robot "
                    "computer, or pass frames through the Python API"
                ) from exc
            gdk = gdk_module
        self.gdk = gdk
        self.width = int(width)
        self.height = int(height)
        self.timeout_ms = int(timeout_ms)
        self._camera: Any = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """相机是否已完成本实例拥有的 GDK 初始化。

        Whether this source currently owns an initialized GDK camera.
        """

        return self._connected

    def connect(self) -> None:
        """初始化 GDK 并只创建 Camera，不创建 Robot/控制接口。

        Initialize GDK and create only Camera—never Robot or a control interface.
        """

        if self._connected:
            raise RuntimeError("GDK head camera is already connected")
        result = self.gdk.gdk_init()
        if result != self.gdk.GDKRes.kSuccess:
            raise RuntimeError(f"GDK initialization failed: {result!r}")
        try:
            self._camera = self.gdk.Camera()
            self._connected = True
        except Exception:
            self.gdk.gdk_release()
            raise

    def _decode(self, image: Any) -> np.ndarray:
        """把 GDK JPEG/裸数据统一解码为连续内存 RGB uint8。

        Decode GDK JPEG or uncompressed data into contiguous RGB uint8.
        """

        if image.encoding == self.gdk.Encoding.JPEG:
            encoded = np.frombuffer(image.data, dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("failed to decode a JPEG frame from GDK")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif image.encoding == self.gdk.Encoding.UNCOMPRESSED:
            rgb = np.frombuffer(image.data, dtype=np.uint8).reshape(
                (image.height, image.width, 3)
            )
            if image.color_format == self.gdk.ColorFormat.BGR:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            elif image.color_format != self.gdk.ColorFormat.RGB:
                raise RuntimeError(
                    f"unsupported GDK color format: {image.color_format!r}"
                )
        else:
            raise RuntimeError(f"unsupported GDK image encoding: {image.encoding!r}")

        # 当前 ROI 只适用于完整 1.60 头部视野；比例不符时不做危险的静默裁剪。
        # Current ROIs require the complete 1.60 head view; never silently crop it.
        aspect_ratio = rgb.shape[1] / rgb.shape[0]
        expected_ratio = self.width / self.height
        if abs(aspect_ratio - expected_ratio) / expected_ratio > 0.03:
            raise RuntimeError(
                f"GDK head frame is {rgb.shape[1]}x{rgb.shape[0]} (aspect "
                f"{aspect_ratio:.4f}); expected the calibrated {expected_ratio:.4f} view"
            )
        if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
            # 仅对相同宽高比做尺寸归一化，不改变相机视野。
            # Normalize same-aspect resolution without changing the field of view.
            interpolation = (
                cv2.INTER_AREA if rgb.shape[1] >= self.width else cv2.INTER_LINEAR
            )
            rgb = cv2.resize(
                rgb, (self.width, self.height), interpolation=interpolation
            )
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def capture(self) -> HeadFrame:
        """只请求 kHeadColor，并在解码后记录主机墙钟时间。

        Request only kHeadColor and timestamp the decoded frame on the host.
        """

        if not self._connected or self._camera is None:
            raise RuntimeError("GDK head camera is not connected")
        started = time.perf_counter_ns()
        image = self._camera.get_latest_image(
            self.gdk.CameraType.kHeadColor,
            self.timeout_ms,
        )
        if image is None:
            raise RuntimeError("no image received from the G2 head camera")
        rgb = self._decode(image)
        captured_ns = time.time_ns()
        return HeadFrame(
            image_rgb=rgb,
            capture_timestamp_ns=captured_ns,
            capture_duration_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
        )

    def disconnect(self) -> None:
        """关闭 Camera，并释放由本实例初始化的 GDK；可重复调用。

        Close Camera and release the GDK owned by this instance; idempotent.
        """

        if not self._connected:
            return
        try:
            if self._camera is not None and hasattr(self._camera, "close_camera"):
                self._camera.close_camera()
        finally:
            self._camera = None
            self._connected = False
            self.gdk.gdk_release()

    def __enter__(self) -> "GDKHeadCameraSource":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.disconnect()


class SyntheticHeadCameraSource:
    """用于非真机命令行烟雾测试的黑色 RGB 帧源。

    Black RGB source for an off-robot CLI smoke test. It replaces only the camera;
    the real model and selected device are still loaded.
    """

    def __init__(self, width: int = 640, height: int = 400):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)

    def capture(self) -> HeadFrame:
        return HeadFrame(
            image_rgb=self.frame.copy(),
            capture_timestamp_ns=time.time_ns(),
            capture_duration_ms=0.0,
        )


class G2HeadRuntime:
    """阻塞、可停止、不追赶积压帧的 G2 头相机 10 Hz 循环。

    Blocking, stoppable G2 head-camera loop targeting 10 Hz without backlog catch-up.
    Actual output rate depends on capture and inference time; this is not hard realtime.
    """

    def __init__(
        self,
        source: HeadCameraSource,
        estimator: ParcelStateEstimator,
        *,
        source_id: str = "cam_head",
    ):
        self.source = source
        self.estimator = estimator
        self.source_id = source_id
        self._stop = threading.Event()
        self._latest: StageResult | None = None

    @property
    def latest(self) -> StageResult | None:
        """最近一次已发布结果，尚未推理时为 None。

        Latest published result, or None before the first inference.
        """

        return self._latest

    def stop(self) -> None:
        """线程安全地请求运行循环退出。

        Request loop termination in a thread-safe manner.
        """

        self._stop.set()

    def run(
        self,
        callback: Callable[[StageResult], None],
        *,
        max_seconds: float | None = None,
        max_results: int | None = None,
    ) -> None:
        """持续采集并把每条真实结果传给 callback。

        Capture continuously and deliver each real result to `callback`.
        `max_seconds` and `max_results` are optional smoke-test limits.
        callback 应快速返回；阻塞式网络/磁盘操作会直接降低检测频率。
        The callback should return quickly; blocking I/O directly reduces output rate.
        """

        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive when provided")
        if max_results is not None and max_results <= 0:
            raise ValueError("max_results must be positive when provided")

        self._stop.clear()
        # 调度使用单调时钟；输出的 capture_timestamp_ns 使用墙钟，便于跨系统记录。
        # Schedule with monotonic time; keep wall time in output for external logging.
        period_ns = int(round(1_000_000_000 / self.estimator.config.detect_hz))
        run_started = time.monotonic()
        next_due_ns = time.monotonic_ns()
        first_capture_ns: int | None = None
        last_frame_index = -1
        emitted = 0

        while not self._stop.is_set():
            if (
                max_seconds is not None
                and time.monotonic() - run_started >= max_seconds
            ):
                break
            remaining_ns = next_due_ns - time.monotonic_ns()
            if remaining_ns > 0:
                self._stop.wait(remaining_ns / 1_000_000_000)
                if self._stop.is_set():
                    break

            head = self.source.capture()
            if first_capture_ns is None:
                first_capture_ns = head.capture_timestamp_ns
            elapsed_s = max(
                0.0, (head.capture_timestamp_ns - first_capture_ns) / 1_000_000_000
            )
            # GDK当前没有暴露可靠的传感器帧号，因此按采集时间映射到名义30FPS编号。
            # GDK exposes no reliable sensor frame id here, so derive a nominal 30 FPS index.
            frame_index = int(round(elapsed_s * self.estimator.config.camera_fps))
            frame_index = max(last_frame_index + 1, frame_index)
            result = self.estimator.process_frame(
                head.image_rgb,
                color_space="RGB",
                source_id=self.source_id,
                frame_index=frame_index,
                capture_timestamp_ns=head.capture_timestamp_ns,
            )
            last_frame_index = frame_index
            self._latest = result
            callback(result)
            emitted += 1
            if max_results is not None and emitted >= max_results:
                break

            # 若采集/推理超过100ms，直接跨过错过的周期，避免延迟不断累积。
            # If capture/inference exceeds 100 ms, skip missed slots instead of queuing them.
            now_ns = time.monotonic_ns()
            missed = max(0, (now_ns - next_due_ns) // period_ns)
            next_due_ns += (missed + 1) * period_ns
