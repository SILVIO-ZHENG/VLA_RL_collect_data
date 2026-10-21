"""Session-scoped alignment state machine and robust pose evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import uuid
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .adapters import AlignmentRobotAdapter
from .config import AlignmentConfig, AlignmentProfile, config_fingerprint
from .detectors import AprilTagDetector
from .geometry import invert, matrix_pose, pose_matrix, project_points, tag_corners, wrapped_degrees
from .types import CameraCalibration, Detection, Frame


class AlignmentState(StrEnum):
    CONNECTING = "CONNECTING"
    READY = "READY"
    SEARCHING = "SEARCHING"
    UNSTABLE = "UNSTABLE"
    ADJUSTING = "ADJUSTING"
    ALIGNED = "ALIGNED"
    CONNECTION_LOST = "CONNECTION_LOST"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AlignmentError(RuntimeError):
    pass


class AlignmentEngine:
    """Own one robot connection and one optional detector worker."""

    def __init__(
        self,
        config: AlignmentConfig,
        profile: AlignmentProfile,
        adapter: AlignmentRobotAdapter,
        detector: AprilTagDetector,
        *,
        host: str,
        adapter_id: str,
        detector_id: str,
        adjustment_mode: str = "table",
    ) -> None:
        if adjustment_mode not in {"table", "robot"}:
            raise AlignmentError("adjustment_mode must be table or robot")
        self.config = config
        self.profile = profile
        self.adapter = adapter
        self.detector = detector
        self.host = host
        self.adapter_id = adapter_id
        self.detector_id = detector_id
        self.adjustment_mode = adjustment_mode
        self.id = uuid.uuid4().hex
        self.control_token = secrets.token_urlsafe(24)
        self.viewer_token = secrets.token_urlsafe(24)
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.state = AlignmentState.CONNECTING
        self.message = "正在连接机器人"
        self.robot: dict[str, Any] = {}
        self.health: dict[str, Any] = {}
        self.head_joints: dict[str, float] = {}
        self.checks: list[dict[str, Any]] = []
        self.calibration: CameraCalibration | None = None
        self.latest_frame: Frame | None = None
        self.latest_detection: Detection | None = None
        self.latest_result: dict[str, Any] = {}
        self.error: str | None = None
        self.completed_report: str | None = None
        self._samples: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=240)
        self._attempts: deque[tuple[float, bool]] = deque(maxlen=240)
        self._last_detection_at: float | None = None
        self._observation_stable_since: float | None = None
        self._surface_stable_since: float | None = None
        self._aligned_since: float | None = None
        self._connection_lost_at: float | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connect()

    def _connect(self) -> None:
        checks: list[dict[str, Any]] = []
        try:
            self.adapter.connect(self.host)
            checks.append(_check("network", "网络连接", True, "机器人可达"))
            self.robot = dict(self.adapter.robot_info())
            identity = str(self.robot.get("model", self.robot.get("id", "已识别")))
            checks.append(_check("identity", "机器人身份", True, identity))
            self.health = dict(self.adapter.health())
            healthy = str(self.health.get("status", "ok")).lower() in {"ok", "ready", "healthy"}
            health_detail = "正常" if healthy else "机器人状态异常"
            checks.append(_check("health", "机器人状态", healthy, health_detail))
            if bool(self.health.get("task_running", False)):
                checks.append(_check("idle", "任务状态", False, "机器人正在执行任务"))
            else:
                checks.append(_check("idle", "任务状态", True, "机器人空闲"))
            if bool(self.health.get("task_running", False)):
                # Never contend with teleoperation, inference or reset.  The
                # shell-only health endpoint can inspect the shared lock without
                # importing GDK, so stop before calibration/state wakes a worker.
                self.checks = checks
                self.state = AlignmentState.ERROR
                self.message = "机器人正在执行任务"
                self.error = None
                self.updated_at = time.time()
                return
            cameras = tuple(str(value) for value in self.robot.get("cameras", ()))
            camera_ok = not cameras or self.profile.camera_name in cameras
            camera_detail = self.profile.camera_name if camera_ok else "相机不存在"
            checks.append(_check("camera", "头部相机", camera_ok, camera_detail))
            self.calibration = self.adapter.calibration(self.profile.camera_name)
            checks.append(_check("calibration", "相机标定", True, self.calibration.version))
            self.head_joints = dict(self.adapter.head_joints())
            head_ok, head_detail = self._head_pose_check(self.head_joints)
            checks.append(_check("head_pose", "头部检查姿态", head_ok, head_detail))
            self.checks = checks
            if not all(bool(item["ok"]) for item in checks):
                self.state = AlignmentState.ERROR
                self.message = next(str(item["detail"]) for item in checks if not item["ok"])
            else:
                self.state = AlignmentState.READY
                self.message = "连接检查完成，可以开始对齐"
            self.error = None
        except Exception as error:
            checks.append(_check("connection", "连接检查", False, str(error)))
            self.checks = checks
            self.state = AlignmentState.ERROR
            self.message = "机器人连接失败"
            self.error = str(error)
            # A failed preflight must never keep the robot process lock or a
            # camera subscription alive until the fallback idle timeout.
            try:
                self.adapter.close()
            except Exception:
                pass
        self.updated_at = time.time()

    def _head_pose_check(self, joints: Mapping[str, float]) -> tuple[bool, str]:
        expected = self.profile.inspection_joints
        if not expected:
            return True, "配置未要求固定关节角"
        missing = [name for name in expected if name not in joints]
        if missing:
            return False, f"缺少头部关节：{', '.join(missing)}"
        errors = {name: float(joints[name]) - target for name, target in expected.items()}
        worst_name = max(errors, key=lambda name: abs(errors[name]))
        worst = errors[worst_name]
        if abs(worst) > self.profile.inspection_joint_tolerance_rad:
            return False, f"请调整 {worst_name}，当前偏差 {worst:+.3f} rad"
        return True, "姿态符合检查配置"

    def start(self) -> None:
        with self._lock:
            if self.state != AlignmentState.READY:
                raise AlignmentError("only a ready session can start alignment")
            if bool(self.health.get("task_running", False)):
                raise AlignmentError("robot is running a task")
            self.state = AlignmentState.SEARCHING
            self.message = (
                "正在寻找 AprilTag"
                if self.profile.target_configured
                else "目标尚未配置，正在寻找 AprilTag"
            )
            self._stop.clear()
            self._surface_stable_since = None
            self._thread = threading.Thread(target=self._worker, name=f"alignment-{self.id[:8]}", daemon=True)
            self._thread.start()

    def recheck(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise AlignmentError("cannot recheck while alignment is running")
            if self.state in {AlignmentState.COMPLETED, AlignmentState.STOPPED}:
                raise AlignmentError("completed session cannot be rechecked")
            self.state = AlignmentState.CONNECTING
            self.message = "正在重新检查机器人"
            self.error = None
        self.adapter.close()
        self._connect()

    def set_adjustment_mode(self, value: str) -> None:
        if value not in {"table", "robot"}:
            raise AlignmentError("adjustment mode must be table or robot")
        with self._lock:
            self.adjustment_mode = value
            if self.latest_result and self.profile.target_configured:
                self.latest_result["guidance"] = self._guidance(self.latest_result["error"])

    def _worker(self) -> None:
        interval = 1.0 / max(1.0, self.config.update_hz)
        next_health_check = 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if started >= next_health_check:
                    self.health = dict(self.adapter.health())
                    if bool(self.health.get("task_running", False)):
                        raise AlignmentError("机器人已开始执行任务，对齐已停止")
                    self.head_joints = dict(self.adapter.head_joints())
                    head_ok, detail = self._head_pose_check(self.head_joints)
                    if not head_ok:
                        raise AlignmentError(detail)
                    next_health_check = started + 1.0
                frame = self.adapter.read_frame(self.profile.camera_name)
                calibration = self.calibration
                if calibration is None:
                    raise AlignmentError("camera calibration is unavailable")
                detections = self.detector.detect(frame, calibration, self.profile)
                detection = max(detections, key=lambda item: item.quality) if detections else None
                with self._lock:
                    self.latest_frame = frame
                    self._attempts.append((time.monotonic(), detection is not None))
                    self._trim_windows()
                    if detection is None:
                        self._handle_missing_detection()
                    else:
                        self._connection_lost_at = None
                        self._consume_detection(detection)
                    self.updated_at = time.time()
            except Exception as error:
                self._handle_worker_error(error)
                if isinstance(error, AlignmentError) and "执行任务" in str(error):
                    self._stop.set()
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, interval - elapsed))

    def _trim_windows(self) -> None:
        cutoff = time.monotonic() - self.profile.tolerance.window_s
        while self._attempts and self._attempts[0][0] < cutoff:
            self._attempts.popleft()
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _consume_detection(self, detection: Detection) -> None:
        assert self.calibration is not None
        base_from_table = (
            self.calibration.base_from_camera @ detection.camera_from_tag @ self.profile.tag_from_table
        )
        pose = matrix_pose(base_from_table)
        now = time.monotonic()
        self.latest_detection = detection
        self._last_detection_at = now
        self._samples.append((now, pose))
        self._trim_windows()
        ratio = sum(1 for _at, found in self._attempts if found) / max(1, len(self._attempts))
        if (
            len(self._samples) < self.profile.tolerance.minimum_samples
            or ratio < self.profile.tolerance.minimum_detection_ratio
        ):
            self.state = AlignmentState.UNSTABLE
            self.message = "已识别 AprilTag，正在等待稳定"
            self._observation_stable_since = None
            self._surface_stable_since = None
            self._aligned_since = None
            self.latest_result = self._result_from_pose(pose, stable=False, detection_ratio=ratio)
            return
        robust_pose, spreads = _robust_pose(tuple(value for _at, value in self._samples))
        stable = self._spread_is_stable(spreads)
        if stable:
            self._observation_stable_since = self._observation_stable_since or now
        else:
            self._observation_stable_since = None
        self.latest_result = self._result_from_pose(robust_pose, stable=stable, detection_ratio=ratio, spreads=spreads)
        if bool(self.latest_result.get("surface", {}).get("eligible")):
            self._surface_stable_since = self._surface_stable_since or now
        else:
            self._surface_stable_since = None
        if stable and not self.profile.target_configured:
            held = now - (self._surface_stable_since or now)
            self.state = AlignmentState.ADJUSTING
            if self._surface_stable_since is None:
                self.message = "位置显示已稳定，正在等待严格标定测量稳定"
            elif held >= 2.0:
                self.message = "严格标定测量稳定，可以将当前位置设为标准"
            else:
                self.message = f"保持当前位置以设为标准（严格测量还需 {2.0 - held:.1f} 秒）"
            self._aligned_since = None
            return
        within = bool(self.latest_result["within_tolerance"])
        if not stable:
            self.state = AlignmentState.UNSTABLE
            self.message = "AprilTag 位姿波动较大，请保持环境稳定"
            self._aligned_since = None
        elif not within:
            self.state = AlignmentState.ADJUSTING
            self.message = "检测稳定，请按提示调整"
            self._aligned_since = None
        else:
            self._aligned_since = self._aligned_since or now
            held = now - self._aligned_since
            if held >= self.profile.tolerance.stable_s:
                self.state = AlignmentState.ALIGNED
                self.message = "已稳定进入允许范围"
            else:
                self.state = AlignmentState.ADJUSTING
                remaining = max(0.0, self.profile.tolerance.stable_s - held)
                self.message = f"已进入允许范围，保持 {remaining:.1f} 秒"

    def _spread_is_stable(self, spread: tuple[float, ...]) -> bool:
        """User-guidance stability; intentionally tolerant to small visual noise."""
        tolerance = self.profile.tolerance
        return (
            spread[0] <= max(0.004, tolerance.forward_m * 0.40)
            and spread[1] <= max(0.004, tolerance.lateral_m * 0.40)
            and spread[2] <= max(0.003, tolerance.height_m * 0.40)
            and spread[3] <= max(0.3, tolerance.tilt_deg * 0.30)
            and spread[4] <= max(0.3, tolerance.tilt_deg * 0.30)
            and spread[5] <= max(0.5, tolerance.yaw_deg * 0.40)
        )

    def _spread_is_surface_stable(self, spread: tuple[float, ...] | None) -> bool:
        """Stricter gate used only for persistent target and safety artifacts."""
        if spread is None:
            return False
        tolerance = self.profile.tolerance
        return (
            spread[0] <= max(0.003, tolerance.forward_m * 0.35)
            and spread[1] <= max(0.003, tolerance.lateral_m * 0.35)
            and spread[2] <= max(0.002, tolerance.height_m * 0.25)
            and spread[3] <= max(0.25, tolerance.tilt_deg * 0.25)
            and spread[4] <= max(0.25, tolerance.tilt_deg * 0.25)
            and spread[5] <= max(0.4, tolerance.yaw_deg * 0.35)
        )

    def _spread_result(self, spreads: tuple[float, ...] | None) -> dict[str, float]:
        values = spreads or (0.0,) * 6
        return {
            "forward_mm": round(values[0] * 1000, 2),
            "lateral_mm": round(values[1] * 1000, 2),
            "height_mm": round(values[2] * 1000, 2),
            "roll_deg": round(values[3], 3),
            "pitch_deg": round(values[4], 3),
            "yaw_deg": round(values[5], 3),
        }

    def _surface_result(
        self,
        pose: tuple[float, ...],
        *,
        stable: bool,
        detection_ratio: float,
        spreads: tuple[float, ...] | None,
    ) -> dict[str, Any]:
        transform = pose_matrix(pose)
        normal = transform[:3, 2]
        tilt = max(abs(float(pose[3])), abs(float(pose[4])))
        surface_stable = stable and self._spread_is_surface_stable(spreads)
        fallback = bool(
            self.latest_detection
            and self.latest_detection.diagnostics.get("fallback", False)
        )
        fallback_hamming = (
            self.latest_detection.diagnostics.get("fallback_hamming")
            if self.latest_detection is not None
            else None
        )
        fallback_allowed = bool(
            fallback
            and self.profile.allow_hamming0_fallback_measurement
            and fallback_hamming == 0
        )
        eligible = bool(
            surface_stable
            and detection_ratio >= self.profile.tolerance.minimum_detection_ratio
            and (not fallback or fallback_allowed)
            and tilt <= self.profile.tolerance.tilt_deg
        )
        if not stable:
            reason = "measurement_unstable"
        elif fallback and not fallback_allowed:
            reason = "fallback_detection_not_allowed"
        elif not surface_stable:
            reason = "measurement_spread_exceeds_safety_limit"
        elif tilt > self.profile.tolerance.tilt_deg:
            reason = "surface_tilt_exceeds_limit"
        else:
            reason = "eligible"
        return {
            "eligible": eligible,
            "reason": reason,
            "frame": "base_link",
            "height_m": round(float(pose[2]) + self.profile.surface_offset_m, 5),
            "tag_origin_height_m": round(float(pose[2]), 5),
            "surface_offset_m": round(self.profile.surface_offset_m, 6),
            "normal": [round(float(value), 7) for value in normal],
            "tilt_deg": round(tilt, 3),
            "spread": self._spread_result(spreads),
            "measurement_mode": "relaxed_hamming0_fallback" if fallback_allowed else "standard",
        }

    def _result_from_pose(
        self,
        pose: tuple[float, ...],
        *,
        stable: bool,
        detection_ratio: float,
        spreads: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        pose_value = {
            "x_m": round(pose[0], 5),
            "y_m": round(pose[1], 5),
            "z_m": round(pose[2], 5),
            "roll_deg": round(pose[3], 3),
            "pitch_deg": round(pose[4], 3),
            "yaw_deg": round(pose[5], 3),
        }
        actual = (
            [list(item) for item in self.latest_detection.corners]
            if self.latest_detection
            else None
        )
        detection_diagnostics = self._detection_diagnostics()
        if not self.profile.target_configured:
            return {
                "pose": pose_value,
                "error": {},
                "guidance": {"mode": self.adjustment_mode, "steps": ["目标尚未配置"]},
                "within_tolerance": False,
                "stable": stable,
                "detection_ratio": round(detection_ratio, 3),
                "spread": self._spread_result(spreads),
                "surface": self._surface_result(
                    pose,
                    stable=stable,
                    detection_ratio=detection_ratio,
                    spreads=spreads,
                ),
                "actual_corners": actual,
                "target_corners": None,
                "tag_id": self.latest_detection.tag_id if self.latest_detection else None,
                "quality": round(self.latest_detection.quality, 3) if self.latest_detection else None,
                "reprojection_error_px": (
                    self.latest_detection.reprojection_error_px
                    if self.latest_detection
                    else None
                ),
                "detection": detection_diagnostics,
            }
        expected = matrix_pose(self.profile.base_from_expected_table)
        error = {
            "forward_mm": round((pose[0] - expected[0]) * 1000, 1),
            "lateral_mm": round((pose[1] - expected[1]) * 1000, 1),
            "height_mm": round((pose[2] - expected[2]) * 1000, 1),
            "roll_deg": round(wrapped_degrees(pose[3] - expected[3]), 2),
            "pitch_deg": round(wrapped_degrees(pose[4] - expected[4]), 2),
            "yaw_deg": round(wrapped_degrees(pose[5] - expected[5]), 2),
        }
        tolerance = self.profile.tolerance
        within = (
            abs(error["forward_mm"]) <= tolerance.forward_m * 1000
            and abs(error["lateral_mm"]) <= tolerance.lateral_m * 1000
            and abs(error["yaw_deg"]) <= tolerance.yaw_deg
        )
        return {
            "pose": pose_value,
            "error": error,
            "guidance": self._guidance(error),
            "within_tolerance": within,
            "stable": stable,
            "detection_ratio": round(detection_ratio, 3),
            "spread": self._spread_result(spreads),
            "surface": self._surface_result(
                pose,
                stable=stable,
                detection_ratio=detection_ratio,
                spreads=spreads,
            ),
            "actual_corners": actual,
            "target_corners": self._target_corners(),
            "tag_id": self.latest_detection.tag_id if self.latest_detection else None,
            "quality": round(self.latest_detection.quality, 3) if self.latest_detection else None,
            "reprojection_error_px": self.latest_detection.reprojection_error_px if self.latest_detection else None,
            "detection": detection_diagnostics,
        }

    def _detection_diagnostics(self) -> dict[str, Any]:
        if self.latest_detection is None:
            return {}
        diagnostics = dict(self.latest_detection.diagnostics)
        fallback = bool(diagnostics.get("fallback", False))
        return {
            "method": "fallback" if fallback else "standard",
            **diagnostics,
        }

    def _guidance(self, error: Mapping[str, float]) -> dict[str, Any]:
        # Error is actual-minus-expected. Move the table by -error, or the robot by +error.
        sign = -1.0 if self.adjustment_mode == "table" else 1.0
        forward = sign * float(error["forward_mm"])
        lateral = sign * float(error["lateral_mm"])
        yaw = sign * float(error["yaw_deg"])
        subject = "桌子" if self.adjustment_mode == "table" else "机器人"
        phrases = []
        if abs(forward) >= 1:
            if subject == "桌子":
                direction = "远离" if forward > 0 else "靠近"
                phrases.append(f"将桌子{direction}机器人方向移动 {abs(forward):.0f} mm")
            else:
                direction = "前方" if forward > 0 else "后方"
                phrases.append(f"将机器人沿自身{direction}移动 {abs(forward):.0f} mm")
        if abs(lateral) >= 1:
            direction = "左侧" if lateral > 0 else "右侧"
            phrases.append(f"将{subject}向机器人{direction}移动 {abs(lateral):.0f} mm")
        if abs(yaw) >= 0.1:
            phrases.append(f"将{subject}{'逆时针' if yaw > 0 else '顺时针'}旋转 {abs(yaw):.1f}°")
        return {
            "mode": self.adjustment_mode,
            "forward_mm": round(forward, 1),
            "lateral_mm": round(lateral, 1),
            "yaw_deg": round(yaw, 2),
            "steps": phrases or ["保持当前位置"],
        }

    def _handle_missing_detection(self) -> None:
        now = time.monotonic()
        self._observation_stable_since = None
        self._surface_stable_since = None
        self._aligned_since = None
        if self._last_detection_at is not None and now - self._last_detection_at <= self.config.tag_loss_grace_s:
            self.state = AlignmentState.UNSTABLE
            age = now - self._last_detection_at
            self.message = f"AprilTag 暂时丢失，最后有效数据 {age:.1f} 秒前"
            if self.latest_result:
                self.latest_result["stale"] = True
                self.latest_result["stale_age_s"] = round(age, 2)
                surface = self.latest_result.get("surface")
                if isinstance(surface, dict):
                    surface["eligible"] = False
                    surface["reason"] = "measurement_stale"
        else:
            self.state = AlignmentState.SEARCHING
            self.message = (
                "未检测到配置的 AprilTag"
                if self.profile.target_configured
                else "目标尚未配置；未检测到 AprilTag"
            )
            self.latest_detection = None
            self.latest_result = {}

    def _handle_worker_error(self, error: Exception) -> None:
        with self._lock:
            now = time.monotonic()
            self.error = str(error)
            self._observation_stable_since = None
            self._surface_stable_since = None
            self._connection_lost_at = self._connection_lost_at or now
            elapsed = now - self._connection_lost_at
            if elapsed >= self.config.reconnect_grace_s:
                self.state = AlignmentState.ERROR
                self.message = "连接持续中断，对齐已停止"
                self._stop.set()
            else:
                self.state = AlignmentState.CONNECTION_LOST
                self.message = f"连接中断，正在重试（{elapsed:.1f} 秒）"
            self.updated_at = time.time()
        try:
            self.adapter.close()
            self.adapter.connect(self.host)
            self.calibration = self.adapter.calibration(self.profile.camera_name)
        except Exception:
            return

    def complete(
        self,
        overlay_data_url: str | None = None,
        *,
        surface_calibration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        release_observation = False
        try:
            with self._lock:
                if self.state != AlignmentState.ALIGNED:
                    raise AlignmentError("alignment can only complete from ALIGNED state")
                release_observation = True
                self._stop.set()
                self.state = AlignmentState.COMPLETED
                self.message = "对齐已完成并保存"
                report = self._report("aligned")
                if surface_calibration is not None:
                    report["surface_calibration"] = dict(surface_calibration)
                directory = self._report_directory()
                if overlay_data_url:
                    image = _decode_overlay(overlay_data_url)
                    image_name = "alignment.png"
                    (directory / image_name).write_bytes(image)
                    report["annotated_image"] = image_name
                report_path = directory / "report.json"
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.completed_report = str(report_path)
                self.updated_at = time.time()
                return report
        finally:
            if release_observation:
                self._release_observation()

    def stop(self, reason: str = "user_cancelled") -> dict[str, Any]:
        try:
            with self._lock:
                if self.state in {AlignmentState.COMPLETED, AlignmentState.STOPPED}:
                    return self.snapshot()
                self._stop.set()
                self.state = AlignmentState.STOPPED
                self.message = "对齐已结束"
                directory = self._report_directory()
                report = self._report(reason[:80] or "user_cancelled")
                path = directory / "report.json"
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.completed_report = str(path)
                self.updated_at = time.time()
                return self.snapshot()
        finally:
            self._release_observation()

    def _report_directory(self) -> Path:
        directory = self.config.reports_directory / time.strftime("%Y%m%d") / self.id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _report(self, result: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.id,
            "result": result,
            "robot": self.robot,
            "robot_host": self.host,
            "profile_id": self.profile.id,
            "profile_version": self.profile.version,
            "config_fingerprint": config_fingerprint(self.config),
            "calibration_version": self.calibration.version if self.calibration else None,
            "observation_context": self._observation_context(),
            "adjustment_mode": self.adjustment_mode,
            "created_at": self.created_at,
            "finished_at": time.time(),
            "alignment": self.latest_result,
            "checks": self.checks,
        }

    def frame(self) -> Frame:
        with self._lock:
            if self.latest_frame is None:
                raise AlignmentError("camera frame is not available yet")
            return self.latest_frame

    def capture_standard_pose(self, minimum_stable_s: float = 2.0) -> tuple[float, ...]:
        """Return the current robust table pose only after a fresh stable hold."""
        with self._lock:
            now = time.monotonic()
            if self._last_detection_at is None or now - self._last_detection_at > self.config.tag_loss_grace_s:
                raise AlignmentError("当前没有新鲜的 AprilTag 检测结果")
            if not self.latest_result or not bool(self.latest_result.get("stable")):
                raise AlignmentError("AprilTag 检测尚未稳定")
            if bool(
                self.latest_detection
                and self.latest_detection.diagnostics.get("fallback", False)
            ):
                raise AlignmentError("当前为容错识别，不能保存为标准位置；请检查标签白色静区和遮挡")
            surface = self.latest_result.get("surface")
            if not isinstance(surface, Mapping) or not bool(surface.get("eligible")):
                raise AlignmentError("当前测量波动尚未达到标定保存要求，请继续保持机器人和桌面静止")
            if self._surface_stable_since is None:
                raise AlignmentError("严格桌面测量尚未稳定")
            held = now - self._surface_stable_since
            if held < minimum_stable_s:
                raise AlignmentError(f"请保持当前位置至少 {minimum_stable_s:.0f} 秒")
            pose = self.latest_result.get("pose", {})
            try:
                return (
                    float(pose["x_m"]),
                    float(pose["y_m"]),
                    float(pose["z_m"]),
                    float(pose["roll_deg"]),
                    float(pose["pitch_deg"]),
                    float(pose["yaw_deg"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AlignmentError("当前 AprilTag 位姿不可用") from error

    def capture_surface_measurement(self, minimum_stable_s: float = 2.0) -> dict[str, Any]:
        """Freeze a fresh, stable, horizontal table measurement for deployment."""
        with self._lock:
            now = time.monotonic()
            if self._last_detection_at is None or now - self._last_detection_at > self.config.tag_loss_grace_s:
                raise AlignmentError("当前没有新鲜的桌面 AprilTag 检测结果")
            surface = self.latest_result.get("surface")
            if not isinstance(surface, Mapping) or not bool(surface.get("eligible")):
                reason = surface.get("reason") if isinstance(surface, Mapping) else "unavailable"
                raise AlignmentError(_surface_reason_message(str(reason)))
            if self._surface_stable_since is None:
                raise AlignmentError("桌面高度检测尚未稳定")
            held = now - self._surface_stable_since
            if held < minimum_stable_s:
                raise AlignmentError(f"请保持桌面与机器人静止至少 {minimum_stable_s:.0f} 秒")
            return {
                **dict(surface),
                "pose": dict(self.latest_result.get("pose", {})),
                "detection_ratio": self.latest_result.get("detection_ratio"),
                "tag_id": self.latest_result.get("tag_id"),
                "quality": self.latest_result.get("quality"),
                "reprojection_error_px": self.latest_result.get("reprojection_error_px"),
                "detection": dict(self.latest_result.get("detection", {})),
                "observation_context": self._observation_context(),
                "stable_hold_s": round(held, 2),
            }

    def _observation_context(self) -> dict[str, Any]:
        calibration = self.calibration
        calibration_value: dict[str, Any] | None = None
        if calibration is not None:
            calibration_value = {
                "version": calibration.version,
                "camera_matrix": np.asarray(calibration.camera_matrix, dtype=np.float64).tolist(),
                "distortion": np.asarray(calibration.distortion, dtype=np.float64).tolist(),
                "base_from_camera": np.asarray(
                    calibration.base_from_camera, dtype=np.float64
                ).tolist(),
            }
            fingerprint_payload = json.dumps(
                calibration_value, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            calibration_value["fingerprint_sha256"] = hashlib.sha256(
                fingerprint_payload
            ).hexdigest()
        return {
            "head_joints": dict(self.head_joints),
            "inspection_joints": dict(self.profile.inspection_joints),
            "inspection_joint_tolerance_rad": self.profile.inspection_joint_tolerance_rad,
            "camera_calibration": calibration_value,
        }

    def apply_target_profile(self, profile: AlignmentProfile) -> None:
        if profile.id != self.profile.id or not profile.target_configured:
            raise AlignmentError("目标配置与当前工位不匹配")
        with self._lock:
            self.profile = profile
            if self.latest_result.get("pose"):
                pose = self.capture_standard_pose(0.0)
                stable = bool(self.latest_result.get("stable"))
                ratio = float(self.latest_result.get("detection_ratio", 0.0))
                self.latest_result = self._result_from_pose(pose, stable=stable, detection_ratio=ratio)
            self._aligned_since = None
            self._surface_stable_since = None
            self.message = f"标准位置已保存并启用（v{profile.version}）"
            self.updated_at = time.time()

    def _target_corners(self) -> list[list[float]] | None:
        if not self.profile.target_configured or self.calibration is None:
            return None
        camera_from_target_tag = (
            invert(self.calibration.base_from_camera)
            @ self.profile.base_from_expected_table
            @ invert(self.profile.tag_from_table)
        )
        target = project_points(
            tag_corners(self.profile.tag_size_m),
            camera_from_target_tag,
            self.calibration.camera_matrix,
            self.calibration.distortion,
        )
        return [list(item) for item in target]

    def snapshot(self, *, include_token: bool = False) -> dict[str, Any]:
        with self._lock:
            value = {
                "id": self.id,
                "state": self.state.value,
                "message": self.message,
                "host": self.host,
                "robot": dict(self.robot),
                "health": dict(self.health),
                "checks": list(self.checks),
                "profile": self.profile.public_dict(),
                "adapter_id": self.adapter_id,
                "detector_id": self.detector_id,
                "adjustment_mode": self.adjustment_mode,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "frame_sequence": self.latest_frame.sequence if self.latest_frame else None,
                "frame_width": self.latest_frame.width if self.latest_frame else None,
                "frame_height": self.latest_frame.height if self.latest_frame else None,
                "calibration_version": self.calibration.version if self.calibration else None,
                "result": dict(self.latest_result),
                "target_corners": self._target_corners(),
                "observation_stable_s": round(
                    max(0.0, time.monotonic() - self._observation_stable_since), 2
                ) if self._observation_stable_since is not None else 0.0,
                "surface_stable_s": round(
                    max(0.0, time.monotonic() - self._surface_stable_since), 2
                ) if self._surface_stable_since is not None else 0.0,
                "error": self.error,
                "report_path": self.completed_report,
            }
            if include_token:
                value["control_token"] = self.control_token
                value["viewer_token"] = self.viewer_token
            return value

    def close(self) -> None:
        self._release_observation()

    def _release_observation(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=2.0)
        self.adapter.close()


def _check(check_id: str, label: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "label": label, "ok": bool(ok), "detail": detail}


def _surface_reason_message(reason: str) -> str:
    messages = {
        "measurement_unstable": "桌面测量尚未稳定，请保持机器人、桌面和标签静止",
        "measurement_spread_exceeds_safety_limit": "桌面测量波动超过安全标定阈值，请继续保持静止",
        "fallback_detection_not_allowed": "当前为容错识别，不能生成安全配置；请更换白色静区完整且无遮挡的 AprilTag",
        "surface_tilt_exceeds_limit": "检测到的桌面倾角超过安全标定限制",
        "measurement_stale": "桌面测量已经失效，请等待 AprilTag 恢复稳定识别",
        "unavailable": "当前没有可用于安全配置的桌面测量",
    }
    return messages.get(reason, "当前桌面测量不能用于安全配置")


def _robust_pose(samples: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = np.asarray(samples, dtype=np.float64)
    angles = np.unwrap(np.radians(values[:, 3:]), axis=0)
    values[:, 3:] = np.degrees(angles)
    median = np.median(values, axis=0)
    deviation = np.abs(values - median)
    mad = np.median(deviation, axis=0)
    score = np.max(deviation / np.maximum(mad * 4.5, (1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05)), axis=1)
    retained = values[score <= 1.0]
    if len(retained) < max(2, len(values) // 2):
        retained = values
    center = np.median(retained, axis=0)
    spread = np.std(retained, axis=0)
    center[3:] = [wrapped_degrees(value) for value in center[3:]]
    return tuple(float(item) for item in center), tuple(float(item) for item in spread)


def _decode_overlay(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise AlignmentError("annotated image must be a PNG data URL")
    try:
        content = base64.b64decode(data_url[len(prefix) :], validate=True)
    except Exception as error:
        raise AlignmentError("annotated image is not valid base64") from error
    if len(content) > 8 * 1024 * 1024 or not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AlignmentError("annotated image is invalid or too large")
    return content
