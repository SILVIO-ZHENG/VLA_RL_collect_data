"""部署封装的无机器人快速回归测试。

Fast deployment-wrapper regression tests that never access a physical robot.

FakeDetector避免加载CUDA/YOLO，但仍执行真实状态引擎与时序过滤器。
FakeDetector avoids CUDA/YOLO while the real geometry and temporal layers still run.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARCEL_ROOT = PROJECT_ROOT / "src" / "nexus_deploy" / "vision" / "parcel"

from nexus_deploy.vision.parcel.runtime import (  # noqa: E402
    G2HeadCameraSource,
    GDKHeadCameraSource,
    ParcelStateEstimator,
    RateLimitedParcelStateService,
    RuntimeConfig,
    STATE_TO_ID,
)
from nexus_deploy.vision.parcel.runtime.core_bridge import Instance  # noqa: E402
from nexus_deploy.vision.parcel.run_video import render_review_frame  # noqa: E402


class FakeDetector:
    """记录模型输入并返回“未检测到实例”的可控测试替身。

    Spy detector that records model inputs and returns a controlled empty detection.
    """

    model_sha256 = "fake-sha256"

    def __init__(self):
        self.frames = []
        self.warmup_shapes = []
        self.closed = False

    def predict(self, frame_bgr):
        self.frames.append(frame_bgr.copy())
        return []

    def warmup(self, image_shape):
        self.warmup_shapes.append(image_shape)

    def close(self):
        self.closed = True


def runtime_config(**overrides):
    """构造指向真实部署规则、但可注入假模型的测试配置。

    Build test settings using the real deployment rules with an injectable fake model.
    """

    values = {
        "rules": PARCEL_ROOT / "configs" / "g2_head.yaml",
        "weights": PROJECT_ROOT / "models" / "vision" / "parcel" / "yolo26m-seg-final.pt",
    }
    values.update(overrides)
    return RuntimeConfig(**values)


class FakeClock:
    """无需sleep即可精确推进纳秒时间的单调时钟替身。

    Manually advanced monotonic nanosecond clock for deterministic rate tests.
    """

    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value


class RuntimeTests(unittest.TestCase):
    """保护颜色、尺寸、频率、历史、GDK所有权和输出协议不变量。

    Protect color, geometry, rate, history, GDK ownership, and schema invariants.
    """

    def test_deployment_rules_match_validated_video_rules(self):
        """部署规则必须与视频复核版本逐项相同 / Rules must match video validation."""

        deployment = yaml.safe_load(
            (PARCEL_ROOT / "configs" / "g2_head.yaml").read_text(encoding="utf-8")
        )
        validated = yaml.safe_load(
            (PARCEL_ROOT / "rules_core" / "configs" / "test_rois.yaml").read_text(
                encoding="utf-8"
            )
        )
        # The original stage engine must remain byte-for-byte configuration
        # compatible with the reviewed video rules.  The deployment wrapper may
        # add its separate presence_gate section because that section is not
        # consumed by the hash-pinned stage core.
        deployment_stage_rules = {
            key: value for key, value in deployment.items() if key != "presence_gate"
        }
        self.assertEqual(deployment_stage_rules, validated)

    def test_g2_rgb_is_converted_to_bgr_once(self):
        """G2 RGB只换色一次且不改原数组 / Convert RGB once without mutating input."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        frame[0, 0] = [11, 22, 33]
        original = frame.copy()
        estimator.process_frame(frame, color_space="RGB", capture_timestamp_ns=1)
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(detector.frames[0][0, 0], [33, 22, 11])

    def test_bgr_input_is_not_channel_swapped(self):
        """OpenCV BGR不能二次换色 / Existing BGR must not be swapped again."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        frame[0, 0] = [33, 22, 11]
        estimator.process_frame(frame, color_space="BGR", capture_timestamp_ns=1)
        np.testing.assert_array_equal(detector.frames[0][0, 0], [33, 22, 11])

    def test_same_aspect_input_is_resized_but_not_square_padded(self):
        """同画幅只归一到640×400 / Same-aspect input normalizes only to 640×400."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        estimator.process_frame(
            np.zeros((200, 320, 3), dtype=np.uint8),
            color_space="RGB",
            capture_timestamp_ns=1,
        )
        self.assertEqual(detector.frames[0].shape, (400, 640, 3))

    def test_wrong_aspect_ratio_is_rejected_before_detector(self):
        """错误FOV在模型调用前拒绝 / Reject incompatible FOV before model inference."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            estimator.process_frame(
                np.zeros((480, 640, 3), dtype=np.uint8),
                color_space="RGB",
                capture_timestamp_ns=1,
            )
        self.assertEqual(detector.frames, [])

    def test_no_package_is_history_confirmed_and_schema_is_json_safe(self):
        """NO_PACKAGE需5帧确认且JSON合法 / Confirm over five frames and emit valid JSON."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        results = [
            estimator.process_frame(
                frame, color_space="RGB", capture_timestamp_ns=index + 1
            )
            for index in range(5)
        ]
        self.assertTrue(all(item.state == "UNKNOWN" for item in results[:4]))
        self.assertEqual(results[-1].state, "NO_PACKAGE")
        self.assertEqual(results[-1].state_id, STATE_TO_ID["NO_PACKAGE"])
        self.assertTrue(results[-1].changed)
        payload = json.loads(results[-1].to_json())
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["model_sha256"], "fake-sha256")

    def test_rate_gate_processes_ten_of_thirty_frames(self):
        """30FPS输入每秒只推理10次 / A 30 FPS caller triggers ten inferences per second."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        clock = FakeClock()
        service = RateLimitedParcelStateService(estimator, clock_ns=clock)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        emitted = 0
        for index in range(30):
            clock.value = int(round(index * 1_000_000_000 / 30))
            emitted += (
                service.process_if_due(
                    frame,
                    color_space="RGB",
                    frame_index=index,
                    capture_timestamp_ns=index,
                )
                is not None
            )
        self.assertEqual(emitted, 10)
        self.assertEqual(len(detector.frames), 10)
        self.assertEqual(service.skipped_frames, 20)

    def test_source_change_resets_temporal_history(self):
        """换视频不能继承旧历史 / A new source must not inherit old temporal history."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        for index in range(4):
            estimator.process_frame(
                frame,
                color_space="RGB",
                source_id="first",
                frame_index=index * 3,
                capture_timestamp_ns=index + 1,
            )
        changed_source = estimator.process_frame(
            frame,
            color_space="RGB",
            source_id="second",
            frame_index=0,
            capture_timestamp_ns=10,
        )
        self.assertEqual(changed_source.state, "UNKNOWN")
        self.assertEqual(changed_source.candidate_frames, 1)

    def test_g2_source_uses_public_head_frame_and_timestamp(self):
        """已有G2A适配器只消费cam_head / Existing G2A adapter consumes cam_head."""

        rgb = np.zeros((400, 640, 3), dtype=np.uint8)
        capture = SimpleNamespace(
            frames={"cam_head": rgb, "cam_left_wrist": rgb, "cam_right_wrist": rgb},
            frame_wall_time_ns={"cam_head": 123},
            capture_duration_ms=4.5,
        )
        robot = SimpleNamespace(capture_camera_frames=lambda: capture)
        head = G2HeadCameraSource(robot).capture()
        self.assertIs(head.image_rgb, rgb)
        self.assertEqual(head.capture_timestamp_ns, 123)
        self.assertEqual(head.capture_duration_ms, 4.5)

    def test_direct_gdk_source_reads_only_head_camera_and_decodes_rgb(self):
        """独立GDK只请求头相机并正确释放 / Direct GDK reads head only and releases it."""

        class FakeImage:
            encoding = 2
            color_format = 4
            height = 400
            width = 640
            data = np.full((400, 640, 3), [33, 22, 11], dtype=np.uint8).tobytes()

        class FakeCamera:
            def __init__(self):
                self.calls = []
                self.closed = False

            def get_latest_image(self, camera_type, timeout_ms):
                self.calls.append((camera_type, timeout_ms))
                return FakeImage()

            def close_camera(self):
                self.closed = True

        camera = FakeCamera()
        fake_gdk = SimpleNamespace(
            gdk_init=lambda: 0,
            gdk_release=lambda: None,
            GDKRes=SimpleNamespace(kSuccess=0),
            Camera=lambda: camera,
            CameraType=SimpleNamespace(kHeadColor="head"),
            Encoding=SimpleNamespace(JPEG=1, UNCOMPRESSED=2),
            ColorFormat=SimpleNamespace(RGB=3, BGR=4),
        )
        source = GDKHeadCameraSource(gdk=fake_gdk)
        source.connect()
        head = source.capture()
        source.disconnect()
        self.assertEqual(camera.calls, [("head", 1000)])
        np.testing.assert_array_equal(head.image_rgb[0, 0], [11, 22, 33])
        self.assertTrue(camera.closed)

    def test_close_is_idempotent(self):
        """模型关闭可重复且关闭后拒绝推理 / Close is idempotent and terminal."""

        detector = FakeDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        estimator.close()
        estimator.close()
        self.assertTrue(detector.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            estimator.process_frame(
                np.zeros((400, 640, 3), dtype=np.uint8),
                color_space="RGB",
            )

    def test_review_renderer_reuses_one_inference_for_old_boxes_masks_and_rois(self):
        """Review rendering uses the old geometry annotation without new YOLO work."""

        mask = np.zeros((400, 640), dtype=bool)
        mask[200:250, 300:350] = True

        class ReviewDetector(FakeDetector):
            def predict(self, frame_bgr):
                self.frames.append(frame_bgr.copy())
                return [
                    Instance(
                        "package",
                        0.90,
                        mask,
                        (300.0, 200.0, 350.0, 250.0),
                    )
                ]

        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        original = frame.copy()
        detector = ReviewDetector()
        estimator = ParcelStateEstimator(runtime_config(), detector=detector)
        try:
            review = estimator.process_frame_with_review(
                frame,
                color_space="BGR",
                source_id="review-video",
                frame_index=0,
                capture_timestamp_ns=1,
            )
            annotated = render_review_frame(
                review,
                estimator.rule_config,
                image_name="review.mp4 | frame=0 t=0.00s",
                timestamp_seconds=0.0,
            )
        finally:
            estimator.close()

        self.assertEqual(len(detector.frames), 1)
        self.assertEqual(len(review.artifacts.instances), 1)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated, frame))
        np.testing.assert_array_equal(frame, original)
        # The original renderer draws the active package in green and the mask
        # beneath it. Checking this point prevents a regression to text-only
        # review videos.
        self.assertGreater(int(annotated[200, 300, 1]), int(annotated[200, 300, 0]))
        self.assertGreater(int(annotated[200, 300, 1]), int(annotated[200, 300, 2]))


if __name__ == "__main__":
    unittest.main()
