#!/usr/bin/env python3
"""部署前快速检查依赖、权重、所选 CUDA 设备与可选 GDK。

Fail fast on dependencies, exact weight, selected CUDA device, and optional GDK.

默认不会打开相机；只有 `--load-model` 会真实加载模型并占用所选设备。
The camera is never opened. Only `--load-model` allocates the selected model device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from nexus_deploy.vision.parcel.runtime import (
    ParcelStateEstimator,
    RuntimeConfig,
    STATE_TO_ID,
    __version__,
)


def sha256_file(path: Path) -> str:
    """分块计算大权重 SHA256，避免一次性读入内存。

    Hash a large weight in chunks instead of loading it entirely into memory.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """创建中英文自检参数。

    Build bilingual deployment-check options.
    """

    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=defaults.weights, help="最终权重 / final weight")
    parser.add_argument("--rules", type=Path, default=defaults.rules, help="阶段规则 / stage rules")
    parser.add_argument("--device", default=defaults.device, help="模型设备 / model device")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="要求所选device是可用CUDA卡 / require the selected device to be CUDA",
    )
    parser.add_argument(
        "--require-gdk",
        action="store_true",
        help="要求agibot_gdk可导入（不连接相机）/ require importable GDK without opening it",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="真实加载并用黑帧预热模型 / load and warm the real model",
    )
    return parser


def selected_cuda_index(device: str) -> int | None:
    """从 Ultralytics 单设备字符串解析 CUDA 序号；CPU/其他后端返回 None。

    Parse a CUDA index from an Ultralytics single-device string; return None for CPU/other backends.
    """

    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        value = value.removeprefix("cuda:")
    if value.isdigit():
        return int(value)
    return None


def main() -> None:
    """执行只读检查，成功时打印一份 JSON 环境清单。

    Run read-only checks and print a JSON environment manifest on success.
    """

    args = build_parser().parse_args()
    config = RuntimeConfig(weights=args.weights, rules=args.rules, device=args.device)
    weights = config.weights.expanduser().resolve()
    rules = config.rules.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"weights do not exist: {weights}")
    if not rules.is_file():
        raise SystemExit(f"rules do not exist: {rules}")
    # 路径参数不能用来替换任意模型：摘要仍必须等于已定版best52。
    # A custom path cannot silently select another model; the digest must remain best52.
    actual_sha256 = sha256_file(weights)
    if actual_sha256.lower() != config.expected_weight_sha256.lower():
        raise SystemExit(
            f"weight SHA256 mismatch: expected {config.expected_weight_sha256}, "
            f"got {actual_sha256}"
        )

    import cv2
    import numpy
    import torch
    import ultralytics
    import yaml

    del yaml
    cuda_available = bool(torch.cuda.is_available())
    cuda_index = selected_cuda_index(args.device)
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    selected_cuda_available = (
        cuda_available and cuda_index is not None and 0 <= cuda_index < cuda_count
    )
    if args.require_cuda and not selected_cuda_available:
        raise SystemExit(
            f"selected device {args.device!r} is not an available CUDA device; "
            f"torch reports {cuda_count} CUDA device(s)"
        )

    # 这里只检查厂商模块能否导入，不会调用gdk_init，也不代表相机一定在线。
    # This checks module availability only; it neither calls gdk_init nor proves camera health.
    has_gdk = importlib.util.find_spec("agibot_gdk") is not None
    if args.require_gdk and not has_gdk:
        raise SystemExit("agibot_gdk is required but is not importable")

    model_loaded = False
    if args.load_model:
        # 此选项会真实分配GPU/CPU内存，并做一张黑帧预热。
        # This option really allocates model memory and performs one black-frame warmup.
        estimator = ParcelStateEstimator(config)
        try:
            estimator.warmup()
            model_loaded = True
        finally:
            estimator.close()

    payload = {
        "status": "ok",
        "runtime_version": __version__,
        "weights": str(weights),
        "weight_sha256": actual_sha256,
        "rules": str(rules),
        "device": args.device,
        "detect_hz": config.detect_hz,
        "camera_fps": config.camera_fps,
        "states": STATE_TO_ID,
        "python_dependencies": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
        },
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
        "selected_cuda_available": selected_cuda_available,
        "gpu": (
            torch.cuda.get_device_name(cuda_index) if selected_cuda_available else None
        ),
        "agibot_gdk_available": has_gdk,
        "model_loaded_and_warmed": model_loaded,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
