#!/usr/bin/env python3
"""从 G2A 头部彩色相机实时输出包裹阶段 JSON。

Stream parcel-stage JSON from the G2A head color camera.

该独立入口只创建 GDK Camera，不创建 GDK Robot，也不发送执行器命令。
This standalone entry creates only a GDK Camera, never a Robot or actuator command.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from nexus_deploy.vision.parcel.runtime import (
    G2HeadRuntime,
    GDKHeadCameraSource,
    ParcelStateEstimator,
    RuntimeConfig,
    SyntheticHeadCameraSource,
)

def build_parser() -> argparse.ArgumentParser:
    """创建带中英文说明的命令行参数解析器。

    Build the command-line parser with bilingual operator guidance.
    """

    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=defaults.weights,
        help="最终best52权重路径 / exact best52 weight path",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=defaults.rules,
        help="G2头部视角ROI与时序配置 / G2 head ROI and temporal rules",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="Ultralytics设备，例如0或cpu / Ultralytics device, e.g. 0 or cpu",
    )
    parser.add_argument(
        "--detect-hz",
        type=float,
        default=defaults.detect_hz,
        help="目标检测/输出频率Hz / target inference and output rate in Hz",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=defaults.camera_fps,
        help=(
            "仅用于帧索引时间尺度，不配置GDK硬件帧率 / nominal FPS for "
            "tracking indices; does not configure GDK hardware"
        ),
    )
    parser.add_argument(
        "--camera-timeout-ms",
        type=int,
        default=1000,
        help="单次GDK取帧超时毫秒 / GDK frame timeout in milliseconds",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="达到秒数后退出，便于烟雾测试 / stop after N seconds",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="达到输出条数后退出 / stop after N emitted results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "用黑帧代替GDK相机；仍加载真实模型 / replace camera with black "
            "frames while still loading the real model"
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过GPU黑帧预热（不建议正式部署）/ skip model warmup",
    )
    return parser


def main() -> None:
    """组装模型、相机与调度器，直到收到退出信号。

    Assemble model, camera, and scheduler, then run until a stop condition/signal.
    """

    args = build_parser().parse_args()
    runtime_config = RuntimeConfig(
        weights=args.weights,
        rules=args.rules,
        device=args.device,
        detect_hz=args.detect_hz,
        camera_fps=args.camera_fps,
    )
    estimator = ParcelStateEstimator(runtime_config)
    # dry-run 只替换相机，仍能验证真实权重、模型和JSON输出链路。
    # Dry-run replaces only the camera and still validates the real model/output path.
    if args.dry_run:
        source = SyntheticHeadCameraSource(width=640, height=400)
    else:
        source = GDKHeadCameraSource(
            width=640,
            height=400,
            timeout_ms=args.camera_timeout_ms,
        )
    runner = G2HeadRuntime(source, estimator)

    def request_stop(signum, frame):
        # 信号处理器只设置Event，让主循环在安全边界退出。
        # The signal handler sets an Event so the loop exits at a safe boundary.
        del signum, frame
        runner.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        if isinstance(source, GDKHeadCameraSource):
            source.connect()
        if not args.no_warmup:
            estimator.warmup()
        def emit_result(result):
            # G2HeadRuntime still owns the legacy StageResult callback.  The
            # estimator's presence result was produced from the same inference
            # and is attached only for observation; this CLI never controls a
            # task or sends a robot command.
            payload = result.to_dict()
            presence = estimator.latest_presence
            if presence is not None:
                payload["presence"] = presence.to_dict()
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

        runner.run(
            emit_result,
            max_seconds=args.max_seconds,
            max_results=args.max_results,
        )
    finally:
        # 即使模型、回调或相机抛错，也要释放模型和本进程拥有的GDK。
        # Always release the model and process-owned GDK, even after an exception.
        estimator.close()
        if isinstance(source, GDKHeadCameraSource):
            source.disconnect()


if __name__ == "__main__":
    main()
