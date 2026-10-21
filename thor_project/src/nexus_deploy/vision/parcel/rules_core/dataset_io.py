"""Read only the held-out test images and their optional YOLO-seg labels."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from state_engine import Instance, mask_box


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "package", 1: "left_gripper", 2: "right_gripper"}
FRAME_PATTERN = re.compile(r"^(?P<source>.+)__f(?P<frame>\d+)$")


@dataclass(frozen=True)
class TestImage:
    path: Path
    relative_path: Path
    source_id: str
    frame_index: int | None


def parse_frame_key(path: Path) -> tuple[str, int | None]:
    match = FRAME_PATTERN.match(path.stem)
    if not match:
        return path.parent.as_posix() or "ungrouped", None
    return match.group("source"), int(match.group("frame"))


def discover_test_images(dataset_root: Path, max_images: int = 0) -> list[TestImage]:
    """Discover images under dataset_root/test/images; no other split is read."""

    dataset_root = dataset_root.expanduser().resolve()
    image_root = dataset_root / "test" / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"test image directory does not exist: {image_root}")
    paths = [
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"no test images found in {image_root}")

    rows: list[TestImage] = []
    for path in paths:
        source_id, frame_index = parse_frame_key(path)
        rows.append(
            TestImage(
                path=path,
                relative_path=path.relative_to(image_root),
                source_id=source_id,
                frame_index=frame_index,
            )
        )
    rows.sort(
        key=lambda row: (
            row.source_id,
            row.frame_index is None,
            row.frame_index if row.frame_index is not None else 0,
            row.relative_path.as_posix(),
        )
    )
    if max_images < 0:
        raise ValueError("max_images cannot be negative")
    return rows[:max_images] if max_images else rows


class LabelResolver:
    """Resolve canonical labels and the nested CVAT-export layout in this dataset."""

    def __init__(self, dataset_root: Path):
        self.test_root = dataset_root.expanduser().resolve() / "test"
        self.label_root = self.test_root / "labels"
        self._by_stem: dict[str, list[Path]] = {}
        if self.label_root.is_dir():
            for path in self.label_root.rglob("*.txt"):
                # train.txt is an image manifest, not a YOLO annotation.
                if path.name.lower() in {"train.txt", "val.txt", "test.txt"}:
                    continue
                self._by_stem.setdefault(path.stem, []).append(path)

    @property
    def label_count(self) -> int:
        return sum(len(paths) for paths in self._by_stem.values())

    def resolve(self, image: TestImage) -> Path | None:
        canonical = self.label_root / image.relative_path.with_suffix(".txt")
        if canonical.is_file():
            return canonical

        matches = self._by_stem.get(image.path.stem, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            suffix = image.relative_path.with_suffix(".txt").as_posix().lower()
            exact = [p for p in matches if p.as_posix().lower().endswith(suffix)]
            if len(exact) == 1:
                return exact[0]
            raise ValueError(
                f"ambiguous labels for {image.relative_path}: "
                + ", ".join(str(path) for path in matches)
            )
        return None


def load_yolo_segmentation(path: Path, image_shape: tuple[int, int]) -> list[Instance]:
    """Load normalized YOLO segmentation polygons as binary instance masks."""

    height, width = image_shape
    instances: list[Instance] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            class_id = int(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"invalid label at {path}:{line_number}") from exc
        if class_id not in CLASS_NAMES:
            raise ValueError(f"unsupported class id {class_id} at {path}:{line_number}")
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"invalid polygon at {path}:{line_number}")
        points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError(f"out-of-range polygon coordinate at {path}:{line_number}")
        pixel_points = np.empty_like(points, dtype=np.int32)
        pixel_points[:, 0] = np.rint(points[:, 0] * max(width - 1, 0)).astype(np.int32)
        pixel_points[:, 1] = np.rint(points[:, 1] * max(height - 1, 0)).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_points], 1)
        boolean_mask = mask.astype(bool)
        instances.append(
            Instance(
                class_name=CLASS_NAMES[class_id],
                confidence=1.0,
                mask=boolean_mask,
                box_xyxy=mask_box(boolean_mask),
            )
        )
    return instances


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batched(items: list[TestImage], size: int) -> Iterable[list[TestImage]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]
