from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridLabelLayout:
    points: np.ndarray
    labels: tuple[str, ...]
    stride: int
    opacity: float
    cell_screen_px: float


def _empty_layout(cell_screen_px: float = 0.0) -> GridLabelLayout:
    return GridLabelLayout(
        points=np.empty((0, 2), dtype=float),
        labels=(),
        stride=0,
        opacity=0.0,
        cell_screen_px=float(cell_screen_px),
    )


def _multiple_count(start: int, stop: int, stride: int) -> int:
    if stop < start:
        return 0
    first = ((start + stride - 1) // stride) * stride
    if first > stop:
        return 0
    return ((stop - first) // stride) + 1


def build_visible_grid_label_layout(
    shape: tuple[int, int],
    grid_step_y: int,
    grid_step_x: int,
    *,
    center_y: float,
    center_x: float,
    viewport_height_px: float,
    viewport_width_px: float,
    zoom: float,
    scale_y: float = 1.0,
    scale_x: float = 1.0,
    max_labels: int = 150,
) -> GridLabelLayout:
    """Create map labels only for readable cells in the visible viewport.

    Screen-space rules provide semantic zoom:
    70+ px cells show every ID, 40-69 px cells show every second ID,
    22-39 px cells show every fifth ID, and smaller cells show no text.
    """
    height, width = (int(shape[0]), int(shape[1]))
    step_y = int(grid_step_y)
    step_x = int(grid_step_x)
    zoom_value = float(zoom)
    sy = abs(float(scale_y))
    sx = abs(float(scale_x))
    viewport_h = float(viewport_height_px)
    viewport_w = float(viewport_width_px)
    label_cap = max(1, int(max_labels))
    if (
        height <= 0
        or width <= 0
        or step_y <= 0
        or step_x <= 0
        or zoom_value <= 0
        or sy <= 0
        or sx <= 0
        or viewport_h <= 0
        or viewport_w <= 0
    ):
        return _empty_layout()

    screen_y = step_y * sy * zoom_value
    screen_x = step_x * sx * zoom_value
    cell_screen_px = min(screen_y, screen_x)
    if cell_screen_px >= 70.0:
        stride = 1
        opacity = 1.0
    elif cell_screen_px >= 40.0:
        stride = 2
        opacity = 0.82
    elif cell_screen_px >= 22.0:
        stride = 5
        opacity = 0.58
    else:
        return _empty_layout(cell_screen_px)

    half_data_y = viewport_h / (2.0 * zoom_value * sy)
    half_data_x = viewport_w / (2.0 * zoom_value * sx)
    row_count = (height + step_y - 1) // step_y
    col_count = (width + step_x - 1) // step_x
    row_start = max(0, int(np.floor((float(center_y) - half_data_y) / step_y)) - 1)
    row_stop = min(
        row_count - 1,
        int(np.floor((float(center_y) + half_data_y) / step_y)) + 1,
    )
    col_start = max(0, int(np.floor((float(center_x) - half_data_x) / step_x)) - 1)
    col_stop = min(
        col_count - 1,
        int(np.floor((float(center_x) + half_data_x) / step_x)) + 1,
    )

    while (
        _multiple_count(row_start, row_stop, stride)
        * _multiple_count(col_start, col_stop, stride)
        > label_cap
    ):
        stride += 1

    points: list[list[float]] = []
    labels: list[str] = []
    for row in range(row_start, row_stop + 1):
        if row % stride != 0:
            continue
        y = min((row * step_y) + (step_y / 2.0), height - 1)
        for col in range(col_start, col_stop + 1):
            if col % stride != 0:
                continue
            x = min((col * step_x) + (step_x / 2.0), width - 1)
            points.append([float(y), float(x)])
            labels.append(f"R{row:02d}C{col:02d}")

    if not points:
        return _empty_layout(cell_screen_px)
    return GridLabelLayout(
        points=np.asarray(points, dtype=float),
        labels=tuple(labels),
        stride=int(stride),
        opacity=float(opacity),
        cell_screen_px=float(cell_screen_px),
    )
