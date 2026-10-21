"""Create review images and compact per-state contact sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from state_engine import (
    Instance,
    RuleConfig,
    StateDecision,
    normalized_polygon_pixels,
)


COLORS = {
    "package": (0, 210, 255),
    "left_gripper": (255, 120, 30),
    "right_gripper": (80, 80, 255),
}
ROI_COLORS = {"bin": (255, 0, 255), "table_left": (0, 255, 255)}


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_annotation(
    image: np.ndarray,
    instances: Sequence[Instance],
    decision: StateDecision,
    config: RuleConfig,
    *,
    image_name: str,
    reference_state: str | None = None,
    stable_state: str | None = None,
    stable_confidence: float | None = None,
) -> np.ndarray:
    """Overlay masks, configured ROIs, the active instance, and state evidence."""

    canvas = image.copy()
    overlay = canvas.copy()
    for index, instance in enumerate(instances):
        color = COLORS.get(instance.class_name, (180, 180, 180))
        overlay[instance.mask] = color
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        line_width = 3 if index == decision.active_package_index else 1
        cv2.drawContours(canvas, contours, -1, color, line_width, cv2.LINE_AA)
        box = instance.box_xyxy
        if box is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            if index == decision.active_package_index:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            marker = "ACTIVE " if index == decision.active_package_index else ""
            _text(
                canvas,
                f"{marker}{instance.class_name} {instance.confidence:.2f}",
                (max(2, x1), max(16, y1 - 5)),
                scale=0.42,
                color=color,
            )
    canvas = cv2.addWeighted(overlay, 0.26, canvas, 0.74, 0.0)

    shape = canvas.shape[:2]
    for name, points in (
        ("bin", config.bin_roi),
        ("table_left", config.table_roi),
    ):
        pixels = normalized_polygon_pixels(points, shape)
        color = ROI_COLORS[name]
        cv2.polylines(canvas, [pixels], True, color, 2, cv2.LINE_AA)
        x, y = int(pixels[0, 0]), int(pixels[0, 1])
        _text(canvas, f"ROI:{name}", (x + 3, max(18, y - 5)), scale=0.43, color=color)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 64), (20, 20, 20), -1)
    if stable_state is None:
        headline = f"STATE: {decision.state}  confidence={decision.confidence:.3f}"
    else:
        stable_value = 0.0 if stable_confidence is None else stable_confidence
        headline = (
            f"STATE: {stable_state}  confidence={stable_value:.3f} "
            f"| raw={decision.state} ({decision.confidence:.3f})"
        )
    _text(
        canvas,
        headline,
        (8, 23),
        scale=0.58,
        color=(255, 255, 255),
        thickness=2,
    )
    metrics = decision.metrics
    evidence = (
        f"packages={metrics.get('package_count', 0)} "
        f"bin={_fmt(metrics.get('bin_overlap'))} "
        f"table={_fmt(metrics.get('table_overlap'))} "
        f"left_gap={_fmt(metrics.get('left_gap_px'), 1)}px "
        f"right_gap={_fmt(metrics.get('right_gap_px'), 1)}px"
    )
    _text(canvas, evidence, (8, 46), scale=0.43, color=(220, 220, 220))
    suffix = f" | GT-geometry: {reference_state}" if reference_state else ""
    _text(
        canvas,
        f"{image_name}{suffix}",
        (8, canvas.shape[0] - 10),
        scale=0.40,
        color=(240, 240, 240),
    )
    return canvas


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def create_contact_sheet(
    image_paths: Sequence[Path],
    destination: Path,
    *,
    title: str,
    columns: int = 3,
    tile_width: int = 360,
    tile_height: int = 250,
) -> bool:
    """Write a contact sheet. Returns False when no source image is readable."""

    loaded: list[tuple[Path, np.ndarray]] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is not None:
            loaded.append((path, image))
    if not loaded:
        return False
    columns = max(1, columns)
    rows = (len(loaded) + columns - 1) // columns
    title_height = 44
    sheet = np.full(
        (title_height + rows * tile_height, columns * tile_width, 3),
        28,
        dtype=np.uint8,
    )
    _text(sheet, title, (12, 29), scale=0.75, thickness=2)
    for slot, (path, image) in enumerate(loaded):
        row, column = divmod(slot, columns)
        usable_height = tile_height - 24
        scale = min(tile_width / image.shape[1], usable_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        y0 = title_height + row * tile_height
        sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        _text(
            sheet,
            path.stem[:46],
            (column * tile_width + 5, title_height + (row + 1) * tile_height - 7),
            scale=0.36,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(destination), sheet))
