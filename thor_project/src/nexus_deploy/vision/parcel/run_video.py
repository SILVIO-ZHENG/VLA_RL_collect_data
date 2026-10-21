#!/usr/bin/env python3
"""用 MP4 离线验证部署 API，每个抽样帧输出一行 JSON。

Exercise the deployment API offline on MP4 and emit one JSON object per sample.

本工具不会按实时速度播放；可选生成带状态叠字的视频，供人工复核。
This tool is not realtime; it can optionally render a state-overlay video for manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2

from nexus_deploy.vision.parcel.runtime import ParcelStateEstimator, RuntimeConfig
from visualization import render_annotation


def _presence_short_name(value: str) -> str:
    """Use compact, stable labels so the review line fits a 640 px frame."""

    return {
        "PACKAGE_PRESENT": "PRESENT",
        "NO_PACKAGE": "NONE",
        "UNKNOWN": "UNKNOWN",
    }.get(value, "UNKNOWN")


def _presence_window(value: tuple[str, ...]) -> str:
    """Render the four actual detector samples as P/N/? without extra inference."""

    symbols = {
        "PACKAGE_PRESENT": "P",
        "NO_PACKAGE": "N",
        "UNKNOWN": "?",
    }
    return "".join(symbols.get(item, "?") for item in value) or "-"


def _outlined_text(image, value: str, origin: tuple[int, int], color) -> None:
    """Draw compact readable text over an annotated BGR image."""

    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        color,
        1,
        cv2.LINE_AA,
    )


def render_review_frame(
    review,
    rule_config,
    *,
    image_name: str,
    timestamp_seconds: float,
):
    """Render old reviewed geometry plus the new four-sample presence evidence.

    ``render_annotation`` is the original review renderer: it draws segmentation
    masks, package/gripper boxes, active-package highlighting and calibrated
    ROI outlines. The small presence line is added from the same completed
    inference. This function performs no model call and cannot control a robot.
    """

    result = review.result
    stage = result.stage
    presence = result.presence
    annotated = render_annotation(
        review.artifacts.image_bgr,
        review.artifacts.instances,
        review.artifacts.raw_stage,
        rule_config,
        image_name=image_name,
        stable_state=stage.state,
        stable_confidence=stage.confidence,
    )

    # Reserve only one short bar below the original two-line stage panel, so
    # its boxes/masks/ROIs remain visible exactly as in the prior review video.
    top = 64
    bottom = min(annotated.shape[0], top + 28)
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (0, top),
        (annotated.shape[1], bottom),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0.0, annotated)
    color = {
        "PACKAGE_PRESENT": (76, 175, 80),
        "NO_PACKAGE": (48, 96, 220),
        "UNKNOWN": (0, 191, 255),
    }.get(presence.state, (0, 191, 255))
    line = (
        f"t={timestamp_seconds:06.2f}s  PRESENCE: {_presence_short_name(presence.state)} "
        f"raw={_presence_short_name(presence.raw_state)} "
        f"4f={_presence_window(presence.window_states)} "
        f"pkg={presence.qualifying_package_count}/{presence.package_count} "
        f"ignored={presence.ignored_package_count}"
    )
    _outlined_text(annotated, line, (8, top + 19), color)
    return annotated


def build_parser() -> argparse.ArgumentParser:
    """创建视频验证命令行参数。

    Build command-line options for offline video verification.
    """

    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True, help="输入MP4 / input MP4")
    parser.add_argument(
        "--weights", type=Path, default=defaults.weights, help="最终权重 / final weight"
    )
    parser.add_argument(
        "--rules", type=Path, default=defaults.rules, help="ROI/时序配置 / ROI and temporal rules"
    )
    parser.add_argument("--device", default=defaults.device, help="设备 / device")
    parser.add_argument(
        "--detect-hz",
        type=float,
        default=defaults.detect_hz,
        help="按视频时间轴抽样频率 / sampling rate on the video timeline",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="JSONL输出；默认打印到终端 / JSONL output, or stdout when omitted",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help=(
            "可人工复核的 MP4：实例掩膜、检测框、ROI 和 Presence；不会额外推理 "
            "/ review MP4 with masks, boxes, ROIs, and presence; no extra inference"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出 / allow replacing an existing output",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="只处理前N秒；0表示全长 / process first N seconds; 0 means full video",
    )
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热 / skip warmup")
    return parser


def main() -> None:
    """按源视频时间轴抽帧，并依序送入同一个有状态 estimator。

    Sample the source timeline and feed frames chronologically to one stateful estimator.
    """

    args = build_parser().parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video does not exist: {video}")
    config = RuntimeConfig(
        weights=args.weights,
        rules=args.rules,
        device=args.device,
        detect_hz=args.detect_hz,
    )
    # OpenCV VideoCapture 返回 BGR；process_frame 必须明确声明，禁止重复换色。
    # OpenCV VideoCapture returns BGR; declare it explicitly to avoid a double swap.
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise SystemExit(f"could not open a valid video: {video}")
    output = args.output_jsonl.expanduser().resolve() if args.output_jsonl else None
    if output:
        if output.exists() and not args.overwrite:
            capture.release()
            raise SystemExit(
                f"output already exists: {output}; pass --overwrite to replace it"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
    output_video = args.output_video.expanduser().resolve() if args.output_video else None
    if output_video:
        if output_video == video:
            capture.release()
            raise SystemExit("output video cannot overwrite the input video")
        if output_video == output:
            capture.release()
            raise SystemExit("output video and JSONL output must use different paths")
        if output_video.exists() and not args.overwrite:
            capture.release()
            raise SystemExit(
                f"output video already exists: {output_video}; pass --overwrite to replace it"
            )
        output_video.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8") if output else None
    writer = None
    review_fps = min(float(config.detect_hz), fps)
    next_sample_seconds = 0.0
    sample_period = 1.0 / config.detect_hz
    frame_index = 0
    sampled_frames = 0
    estimator = ParcelStateEstimator(config)
    try:
        if not args.no_warmup:
            estimator.warmup()
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            timestamp_seconds = frame_index / fps
            if args.max_seconds > 0 and timestamp_seconds >= args.max_seconds:
                break
            # 半帧容差消除浮点边界抖动；while推进目标时刻可避免长期累计漂移。
            # Half-frame tolerance handles floating boundaries; advancing target time
            # in a loop prevents long-run sampling drift.
            if timestamp_seconds + (0.5 / fps) >= next_sample_seconds:
                review = estimator.process_frame_with_review(
                    frame_bgr,
                    color_space="BGR",
                    source_id=video.stem,
                    frame_index=frame_index,
                    # 视频时间另存为 video_timestamp_seconds；这里使用同域墙钟计算延迟。
                    # Source time is stored separately; wall time keeps latency meaningful.
                    capture_timestamp_ns=time.time_ns(),
                )
                result = review.result
                # The nested presence field uses the same one YOLO inference as
                # the historical stage output.  This command remains offline and
                # never sends pause/resume commands to a robot.
                payload = result.to_dict()
                payload["video_timestamp_seconds"] = round(timestamp_seconds, 6)
                line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
                if stream:
                    stream.write(line + "\n")
                else:
                    print(line, flush=True)
                while next_sample_seconds <= timestamp_seconds + (0.5 / fps):
                    next_sample_seconds += sample_period
                sampled_frames += 1
                if output_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            review_fps,
                            (frame_bgr.shape[1], frame_bgr.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"could not create review video: {output_video}")
                    writer.write(
                        render_review_frame(
                            review,
                            estimator.rule_config,
                            image_name=(
                                f"{video.name} | frame={frame_index} "
                                f"t={timestamp_seconds:.2f}s"
                            ),
                            timestamp_seconds=timestamp_seconds,
                        )
                    )
            frame_index += 1
    finally:
        # 文件、解码器和模型在任何异常路径下都必须释放。
        # Close output, decoder, and model on every success/error path.
        if stream:
            stream.close()
        if writer is not None:
            writer.release()
        capture.release()
        estimator.close()
    summary = {
        "status": "complete",
        "video": str(video),
        "output_jsonl": str(output) if output else None,
        "output_video": str(output_video) if output_video else None,
        "sampled_frames": sampled_frames,
        "review_video_fps": review_fps if output_video else None,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
