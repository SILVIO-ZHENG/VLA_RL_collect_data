"""Small Ultralytics adapter shared by image and video entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from dataset_io import sha256_file
from state_engine import Instance, mask_box


FINAL_WEIGHT_SHA256 = "6ca0492ba45211e14e8f5d4f2eeac6bc16c29476f5f0646b4503b4b5c0d40c2c"
EXPECTED_CLASS_MAP = {0: "package", 1: "left_gripper", 2: "right_gripper"}


def normalized_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def load_final_model(
    weights: Path,
    *,
    expected_sha256: str = FINAL_WEIGHT_SHA256,
    skip_hash_check: bool = False,
) -> tuple[Any, str, Path]:
    weights = weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    actual_hash = sha256_file(weights)
    if (
        not skip_hash_check
        and expected_sha256
        and actual_hash.lower() != expected_sha256.lower()
    ):
        raise ValueError(
            f"weight SHA256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is missing; install this folder's requirements.txt on the cloud node"
        ) from exc
    model = YOLO(str(weights))
    names = normalized_names(model.names)
    if names != EXPECTED_CLASS_MAP:
        raise ValueError(
            f"model class map must be exactly {EXPECTED_CLASS_MAP}, got {names}"
        )
    return model, actual_hash, weights


def instances_from_result(result: Any) -> list[Instance]:
    image_shape = result.orig_img.shape[:2]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = normalized_names(result.names)
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy().astype(float)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
    mask_data: np.ndarray | None = None
    if result.masks is not None and result.masks.data is not None:
        mask_data = result.masks.data.detach().cpu().numpy()
        if len(mask_data) != len(class_ids):
            raise RuntimeError("Ultralytics returned different box and mask counts")
    if mask_data is None:
        raise RuntimeError(
            "segmentation result has boxes but no masks; refusing to substitute box geometry"
        )

    instances: list[Instance] = []
    for index, class_id in enumerate(class_ids):
        class_name = names.get(int(class_id), f"class_{class_id}")
        box = tuple(float(value) for value in xyxy[index])
        raw_mask = mask_data[index]
        if raw_mask.shape != image_shape:
            raw_mask = cv2.resize(
                raw_mask,
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = raw_mask > 0.5
        instances.append(
            Instance(
                class_name=class_name,
                confidence=float(confidences[index]),
                mask=mask,
                box_xyxy=box if box else mask_box(mask),
            )
        )
    return instances
