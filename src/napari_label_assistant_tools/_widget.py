from __future__ import annotations

from bisect import bisect_left
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace

import napari
import numpy as np
from scipy import ndimage as ndi
try:
    from skimage.measure import euler_number
except ImportError:
    # scikit-image 0.26 no longer re-exports euler_number from measure.__init__
    from skimage.measure._regionprops import euler_number
from qtpy.QtCore import QItemSelectionModel, Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHeaderView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._grid_map import build_visible_grid_label_layout
from ._dynamic_edit import (
    DEFAULT_TILE_SIZE,
    DynamicLabelsEditController,
    INTERNAL_LAYER_ROLE_KEY,
    is_internal_layer,
)
from ._large_component_controller import LargeComponentController
from ._mask_validation import looks_like_grayscale_image
from ._large_components import (
    LARGE_ANALYSIS_PIXEL_THRESHOLD,
    LARGE_TABLE_PAGE_SIZE,
    LargeComponentIndex,
    build_large_component_index,
)

_PANEL_MIN_WIDTH = 420
_PANEL_MAX_WIDTH = 520


def _auto_refresh_layer_controls(viewer, parent, refresh):
    """Debounce layer-list changes and keep selector widgets synchronized."""
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(50)
    collection_emitters = []
    observed_layers = []

    def _schedule(*_args) -> None:
        if not timer.isActive():
            timer.start()

    def _sync_name_events() -> None:
        current_layers = list(viewer.layers)
        for layer in list(observed_layers):
            if not any(candidate is layer for candidate in current_layers):
                with suppress(Exception):
                    layer.events.name.disconnect(_schedule)
                observed_layers.remove(layer)
        for layer in current_layers:
            if any(candidate is layer for candidate in observed_layers):
                continue
            with suppress(Exception):
                layer.events.name.connect(_schedule)
                observed_layers.append(layer)

    def _run_refresh() -> None:
        if bool(
            getattr(viewer, "_label_assistant_workspace_loading", False)
        ):
            timer.start()
            return
        _sync_name_events()
        refresh()

    def _cleanup(*_args) -> None:
        timer.stop()
        for emitter in collection_emitters:
            with suppress(Exception):
                emitter.disconnect(_schedule)
        for layer in list(observed_layers):
            with suppress(Exception):
                layer.events.name.disconnect(_schedule)
        observed_layers.clear()

    timer.timeout.connect(_run_refresh)
    for event_name in ("inserted", "removed", "reordered"):
        emitter = getattr(viewer.layers.events, event_name, None)
        if emitter is not None:
            emitter.connect(_schedule)
            collection_emitters.append(emitter)
    _sync_name_events()
    parent.destroyed.connect(_cleanup)
    return timer


@dataclass
class _ComponentRecord:
    component_id: int
    label_value: int
    area: int
    euler: int
    centroid_y: float
    centroid_x: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    grid_row: int | None = None
    grid_col: int | None = None
    grid_id: str = ""

    @property
    def bbox_text(self) -> str:
        return (
            f"y[{self.bbox_y0}:{self.bbox_y1}] "
            f"x[{self.bbox_x0}:{self.bbox_x1}]"
        )

    @property
    def bbox(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (self.bbox_y0, self.bbox_y1),
            (self.bbox_x0, self.bbox_x1),
        )


@dataclass
class _FastComponentIndex:
    shape: tuple[int, int]
    component_id_map: np.ndarray
    records: dict[int, _ComponentRecord]
    deleted_component_ids: set[int]

    def active_records(self) -> list[_ComponentRecord]:
        return [
            record
            for component_id, record in sorted(self.records.items())
            if component_id not in self.deleted_component_ids
        ]

    def component_id_at(self, coords: tuple[int, int]) -> int | None:
        y, x = coords
        if y < 0 or x < 0 or y >= self.shape[0] or x >= self.shape[1]:
            return None
        component_id = int(self.component_id_map[y, x])
        if component_id <= 0 or component_id in self.deleted_component_ids:
            return None
        return component_id

    def delete_components(
        self, data: np.ndarray, component_ids: list[int]
    ) -> tuple[np.ndarray, int]:
        out = _as_2d_mask(data).copy()
        changed = 0
        for component_id in [int(value) for value in component_ids]:
            if component_id in self.deleted_component_ids:
                continue
            record = self.records.get(component_id)
            if record is None:
                continue
            ys = slice(record.bbox_y0, record.bbox_y1)
            xs = slice(record.bbox_x0, record.bbox_x1)
            local_map = self.component_id_map[ys, xs]
            mask = local_map == component_id
            count = int(np.count_nonzero(mask))
            if count == 0:
                self.deleted_component_ids.add(component_id)
                continue
            out[ys, xs][mask] = 0
            local_map[mask] = 0
            self.deleted_component_ids.add(component_id)
            changed += count
        return out, changed


class _NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return left < right
        return super().__lt__(other)


class _ComponentTableWidget(QTableWidget):
    def __init__(self, locate_callback=None, theme: str = "dark") -> None:
        super().__init__(0, 10)
        self._locate_callback = locate_callback
        self._context_menu_callback = None
        self.setHorizontalHeaderLabels(
            [
                "Component ID",
                "Label",
                "Pixels",
                "Euler",
                "Grid row",
                "Grid col.",
                "Grid cell",
                "Centroid Y",
                "Centroid X",
                "Bounds",
            ]
        )
        header_tooltips = [
            "Component identifier generated by this analysis.",
            "Original integer value in the source Labels layer.",
            "Component area measured as a pixel count.",
            "Euler number: 1 minus the number of holes.",
            "Zero-based grid row containing the component centroid.",
            "Zero-based grid column containing the component centroid.",
            "Grid-cell identifier containing the component centroid.",
            "Component centroid along the image Y axis.",
            "Component centroid along the image X axis.",
            "Component bounding box in source data coordinates.",
        ]
        for column, tooltip in enumerate(header_tooltips):
            self.horizontalHeaderItem(column).setToolTip(tooltip)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setWordWrap(False)
        self.setToolTip(
            "Select one or more components. Double-click a row to center it "
            "in the viewer; right-click to copy selected components."
        )
        self.verticalHeader().setVisible(False)
        theme_name = str(theme).lower()
        if "light" in theme_name:
            base_color = "#ffffff"
            alternate_color = "#e9ecef"
            text_color = "#111111"
            grid_color = "#adb5bd"
        else:
            base_color = "#25262b"
            alternate_color = "#34363d"
            text_color = "#f8f9fa"
            grid_color = "#555861"
        self.setStyleSheet(
            "QTableWidget {"
            f" background-color: {base_color};"
            f" alternate-background-color: {alternate_color};"
            f" color: {text_color};"
            f" gridline-color: {grid_color};"
            "}"
            "QTableWidget::item {"
            f" color: {text_color};"
            "}"
            "QTableWidget::item:selected {"
            " background-color: #f59f00;"
            " color: #101010;"
            "}"
        )
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.setColumnWidth(0, 100)
        self.setColumnWidth(1, 60)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 60)
        self.setColumnWidth(4, 70)
        self.setColumnWidth(5, 70)
        self.setColumnWidth(6, 90)
        self.setColumnWidth(7, 90)
        self.setColumnWidth(8, 90)
        self.setColumnWidth(9, 150)
        self.setSortingEnabled(True)
        self.itemDoubleClicked.connect(self._locate_item)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.viewport().setContextMenuPolicy(Qt.CustomContextMenu)
        self.viewport().customContextMenuRequested.connect(self._show_context_menu)

    def set_records(self, records: list[_ComponentRecord]) -> None:
        self.setUpdatesEnabled(False)
        self.setSortingEnabled(False)
        self.clearContents()
        self.setRowCount(len(records))
        for row, record in enumerate(records):
            self._set_numeric_item(row, 0, record.component_id)
            self._set_numeric_item(row, 1, record.label_value)
            self._set_numeric_item(row, 2, record.area)
            self._set_numeric_item(row, 3, record.euler)
            self._set_optional_numeric_item(row, 4, record.grid_row)
            self._set_optional_numeric_item(row, 5, record.grid_col)
            self.setItem(row, 6, QTableWidgetItem(record.grid_id))
            self._set_float_item(row, 7, record.centroid_y)
            self._set_float_item(row, 8, record.centroid_x)
            self.setItem(row, 9, QTableWidgetItem(record.bbox_text))
        self.setSortingEnabled(True)
        self.setUpdatesEnabled(True)

    def set_grid_columns_visible(self, visible: bool) -> None:
        for column in (4, 5, 6):
            self.setColumnHidden(column, not visible)

    def selected_component_ids(self) -> list[int]:
        ids: list[int] = []
        for index in self.selectionModel().selectedRows():
            item = self.item(index.row(), 0)
            if item is not None:
                ids.append(int(item.data(Qt.UserRole)))
        return ids

    def select_component_id(self, component_id: int, append: bool = False) -> bool:
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None and int(item.data(Qt.UserRole)) == int(
                component_id
            ):
                self._select_row(row, append=append)
                self.scrollToItem(item)
                return True
        return False

    def selected_component_summary(self, limit: int = 8) -> str:
        ids = self.selected_component_ids()
        if not ids:
            return "Selected components: 0"
        preview = ", ".join(str(component_id) for component_id in ids[:limit])
        if len(ids) > limit:
            preview = f"{preview}, ..."
        return f"Selected components: {len(ids)} | IDs: {preview}"

    def _set_numeric_item(self, row: int, column: int, value: int) -> None:
        item = _NumericTableWidgetItem(str(value))
        item.setData(Qt.UserRole, int(value))
        self.setItem(row, column, item)

    def _set_optional_numeric_item(
        self, row: int, column: int, value: int | None
    ) -> None:
        text = "" if value is None else str(int(value))
        item = _NumericTableWidgetItem(text)
        if value is not None:
            item.setData(Qt.UserRole, int(value))
        self.setItem(row, column, item)

    def _set_float_item(self, row: int, column: int, value: float) -> None:
        item = _NumericTableWidgetItem(f"{value:.2f}")
        item.setData(Qt.UserRole, float(value))
        self.setItem(row, column, item)

    def _locate_item(self, item: QTableWidgetItem) -> None:
        if self._locate_callback is None:
            return
        try:
            component_id = int(self.item(item.row(), 0).data(Qt.UserRole))
        except Exception:
            return
        self._locate_callback(component_id)

    def _select_row(self, row: int, append: bool) -> None:
        model = self.selectionModel()
        if model is None:
            return
        index = self.model().index(row, 0)
        if append:
            flags = QItemSelectionModel.Rows | QItemSelectionModel.Select
        else:
            flags = QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect
        model.select(index, flags)
        model.setCurrentIndex(index, QItemSelectionModel.NoUpdate)

    def _show_context_menu(self, position) -> None:
        item = self.itemAt(position)
        if item is not None:
            row = item.row()
            if row >= 0 and not self.selectionModel().isRowSelected(
                row, self.rootIndex()
            ):
                self.selectRow(row)
        if self._context_menu_callback is None:
            return
        self._context_menu_callback(position)

    def contextMenuEvent(self, event) -> None:
        position = self.viewport().mapFromGlobal(event.globalPos())
        self._show_context_menu(position)
        event.accept()


def _validate_grid_step(step: int, axis_name: str) -> int:
    step = int(step)
    if step <= 0:
        raise ValueError(f"{axis_name} grid size must be a positive integer.")
    return step


def _grid_index_from_xy(
    y: float, x: float, grid_step_y: int, grid_step_x: int
) -> tuple[int, int, str]:
    step_y = _validate_grid_step(grid_step_y, "Y")
    step_x = _validate_grid_step(grid_step_x, "X")
    row = max(0, int(np.floor(float(y) / step_y)))
    col = max(0, int(np.floor(float(x) / step_x)))
    return row, col, f"R{row:02d}C{col:02d}"


def _build_grid_shapes(
    shape: tuple[int, int], grid_step_y: int, grid_step_x: int
) -> list[np.ndarray]:
    height, width = shape
    step_y = _validate_grid_step(grid_step_y, "Y")
    step_x = _validate_grid_step(grid_step_x, "X")
    lines: list[np.ndarray] = []

    for y in range(0, height + 1, step_y):
        yy = min(y, height)
        lines.append(np.array([[yy, 0], [yy, width]], dtype=float))
    if height % step_y != 0:
        lines.append(np.array([[height, 0], [height, width]], dtype=float))

    for x in range(0, width + 1, step_x):
        xx = min(x, width)
        lines.append(np.array([[0, xx], [height, xx]], dtype=float))
    if width % step_x != 0:
        lines.append(np.array([[0, width], [height, width]], dtype=float))

    return lines


def _resolve_layer_data_array(data: object) -> np.ndarray:
    """Resolve napari layer data to a concrete array for Labels and image utilities."""
    resolved = data
    while isinstance(resolved, (list, tuple)) and len(resolved) > 0:
        resolved = resolved[0]
    return np.asarray(resolved)


def _as_2d_mask(data: np.ndarray) -> np.ndarray:
    """Convert supported napari layer data into a 2D Labels array."""
    arr = _resolve_layer_data_array(data)
    if arr.ndim == 2:
        return arr
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D Labels layer. Got shape={arr.shape}. "
            "Select a 2D Labels layer rather than an RGB/RGBA or grayscale image."
        )
    return arr


def _supported_2d_mask_shape(data: object) -> tuple[int, int] | None:
    """Return the resolved 2D shape for Labels-layer data, if supported."""
    resolved = data
    while isinstance(resolved, (list, tuple)) and len(resolved) > 0:
        resolved = resolved[0]
    shape = getattr(resolved, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(resolved).shape
        except Exception:
            return None
    try:
        dims = tuple(int(v) for v in shape)
    except Exception:
        return None
    squeezed = tuple(v for v in dims if v != 1)
    if not squeezed and dims:
        squeezed = (1,)
    if len(squeezed) == 2:
        return squeezed
    return None


def _merge_binary_masks(mask_data: list[np.ndarray]) -> np.ndarray:
    """Merge two or more 2D Labels arrays into a single binary Labels layer."""
    if len(mask_data) < 2:
        raise ValueError("Select at least two Labels layers to merge.")

    masks = [_as_2d_mask(data) > 0 for data in mask_data]
    shape = masks[0].shape
    for idx, mask in enumerate(masks[1:], start=2):
        if mask.shape != shape:
            raise ValueError(
                "All Labels layers must have the same 2D shape. "
                f"Layer 1 has shape {shape}, layer {idx} has shape {mask.shape}."
            )

    merged = np.zeros(shape, dtype=bool)
    for mask in masks:
        merged |= mask
    return merged


def _merge_label_layers_same_class(
    layer_data: list[np.ndarray],
    *,
    target_value: int,
    background_value: int = 0,
) -> np.ndarray:
    """Merge non-zero pixels from many label/Labels layers into one class value."""
    if len(layer_data) < 1:
        raise ValueError("Select at least one Labels layer to merge.")
    if len(layer_data) == 1:
        merged = _as_2d_mask(layer_data[0]) > 0
    else:
        merged = _merge_binary_masks(layer_data)
    out = np.full(merged.shape, int(background_value), dtype=np.int32)
    out[merged] = int(target_value)
    return out


def _compose_label_layers_distinct_classes(
    layer_data: list[np.ndarray],
    *,
    class_values: list[int],
    background_value: int = 0,
) -> np.ndarray:
    """Compose multiple layers into one labels array using one output value per layer."""
    if len(layer_data) < 2:
        raise ValueError("Select at least two layers to compose.")
    if len(class_values) != len(layer_data):
        raise ValueError("Class value count must match selected layer count.")
    values = [int(background_value), *[int(v) for v in class_values]]
    if len(set(values)) != len(values):
        raise ValueError("Background and output class values must all be different.")

    masks = [_as_2d_mask(data) > 0 for data in layer_data]
    shape = masks[0].shape
    for idx, mask in enumerate(masks[1:], start=2):
        if mask.shape != shape:
            raise ValueError(
                "All selected layers must have the same 2D shape. "
                f"Layer 1 has shape {shape}, layer {idx} has shape {mask.shape}."
            )
    out = np.full(shape, int(background_value), dtype=np.int32)
    for mask, value in zip(masks, class_values, strict=False):
        out[mask] = int(value)
    return out


def _count_label_votes(layer_data: list[np.ndarray]) -> np.ndarray:
    """Return a per-pixel vote count across selected non-zero Labels layers."""
    if len(layer_data) < 2:
        raise ValueError("Select at least two layers to compute votes.")
    masks = [_as_2d_mask(data) > 0 for data in layer_data]
    shape = masks[0].shape
    for idx, mask in enumerate(masks[1:], start=2):
        if mask.shape != shape:
            raise ValueError(
                "All selected layers must have the same 2D shape. "
                f"Layer 1 has shape {shape}, layer {idx} has shape {mask.shape}."
            )
    stacked = np.stack(masks, axis=0)
    return np.sum(stacked, axis=0, dtype=np.int32)


def _consensus_label_map(
    layer_data: list[np.ndarray],
    *,
    min_votes: int,
    consensus_value: int = 1,
    background_value: int = 0,
) -> np.ndarray:
    """Keep pixels present in at least min_votes selected layers."""
    votes = _count_label_votes(layer_data)
    threshold = int(min_votes)
    if threshold <= 0:
        raise ValueError("Minimum votes must be a positive integer.")
    if threshold > len(layer_data):
        raise ValueError("Minimum votes cannot exceed selected layer count.")
    out = np.full(votes.shape, int(background_value), dtype=np.int32)
    out[votes >= threshold] = int(consensus_value)
    return out


def _reclassify_single_value_labels(
    data: np.ndarray,
    *,
    new_value: int,
    background_value: int = 0,
    source_value: int | None = None,
) -> np.ndarray:
    """Rewrite one non-background label value in a 2D labels layer."""
    arr = np.asarray(_as_2d_mask(data))
    bg = int(background_value)
    out = arr.astype(np.int32, copy=True)

    non_background = [int(v) for v in np.unique(arr) if int(v) != bg]
    if source_value is None:
        if len(non_background) != 1:
            raise ValueError(
                "Expected exactly one non-background value in the source layer. "
                f"Found: {non_background}"
            )
        source = non_background[0]
    else:
        source = int(source_value)
        if source == bg:
            raise ValueError("Source value cannot be the background value.")
        if source not in non_background:
            raise ValueError(
                f"Source value {source} was not found in the selected layer."
            )

    out[arr == source] = int(new_value)
    return out


def _build_fast_component_index(
    data: np.ndarray,
    *,
    grid_step_y: int | None = None,
    grid_step_x: int | None = None,
) -> _FastComponentIndex:
    """Build a fast connected-component index for a 2D Labels layer."""
    source = _as_2d_mask(data)
    component_id_map = np.zeros(source.shape, dtype=np.int32)
    records: dict[int, _ComponentRecord] = {}
    component_id = 1

    for label_value in [int(v) for v in np.unique(source) if int(v) != 0]:
        component_labels, count = ndi.label(source == label_value)
        if count <= 0:
            continue
        objects = ndi.find_objects(component_labels)
        for local_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            ys, xs = slices
            local_component = component_labels[ys, xs] == local_id
            if not local_component.any():
                continue
            component_id_map[ys, xs][local_component] = component_id
            yy, xx = np.nonzero(local_component)
            yy = yy + int(ys.start)
            xx = xx + int(xs.start)
            area = int(local_component.sum())
            centroid_y = float(yy.mean())
            centroid_x = float(xx.mean())
            grid_row = None
            grid_col = None
            grid_id = ""
            if grid_step_y is not None and grid_step_x is not None:
                grid_row, grid_col, grid_id = _grid_index_from_xy(
                    centroid_y, centroid_x, grid_step_y, grid_step_x
                )
            records[component_id] = _ComponentRecord(
                component_id=component_id,
                label_value=label_value,
                area=area,
                euler=int(euler_number(local_component, connectivity=2)),
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                bbox_y0=int(ys.start),
                bbox_x0=int(xs.start),
                bbox_y1=int(ys.stop),
                bbox_x1=int(xs.stop),
                grid_row=grid_row,
                grid_col=grid_col,
                grid_id=grid_id,
            )
            component_id += 1

    return _FastComponentIndex(
        shape=tuple(source.shape),
        component_id_map=component_id_map,
        records=records,
        deleted_component_ids=set(),
    )


def _copy_mask_components_to_target(
    source_data: np.ndarray,
    target_data: np.ndarray,
    component_labels: np.ndarray,
    component_ids: list[int],
) -> tuple[np.ndarray, int]:
    """Copy selected connected components from source Labels layer into a target Labels layer."""
    source = np.asarray(_as_2d_mask(source_data))
    target = np.asarray(_as_2d_mask(target_data))
    if source.shape != target.shape:
        raise ValueError(
            "Source and target Labels layers must have the same 2D shape. "
            f"Got {source.shape} and {target.shape}."
        )
    if component_labels.shape != source.shape:
        raise ValueError("Component-label map shape does not match source Labels layer.")
    if not component_ids:
        raise ValueError("Select at least one component row to copy.")
    mask = np.isin(component_labels, np.asarray(component_ids, dtype=np.int32))
    copied_pixels = int(np.count_nonzero(mask))
    if copied_pixels <= 0:
        raise ValueError("Selected component copy made no label changes.")
    out_dtype = np.result_type(target.dtype, source.dtype)
    out = target.astype(out_dtype, copy=True)
    out[mask] = source.astype(out_dtype, copy=False)[mask]
    return out, copied_pixels


def _get_or_create_points_layer(
    viewer: napari.Viewer,
    name: str,
    points: np.ndarray,
    features: object,
    text_field: str,
    text_size: float,
    text_scale_with_zoom: bool,
    text_color: str = "yellow",
    anchor: str = "center",
) -> napari.layers.Points:
    """Create or overwrite a Points layer with feature-driven text."""
    if name in viewer.layers:
        layer = viewer.layers[name]
        if not isinstance(layer, napari.layers.Points):
            raise TypeError(
                f"Layer name '{name}' exists but is not a Points layer."
            )
        # Updating data emits a draw before the corresponding feature/text
        # values are replaced. When the visible label count grows, that brief
        # mismatch can make Points text index past the old feature array.
        # Block layer events until both arrays have matching lengths.
        with layer.events.blocker_all():
            layer.data = points
            layer.features = features
        layer.visible = True
    else:
        layer = viewer.add_points(points, name=name, size=0, features=features)

    layer.text = {
        "string": f"{{{text_field}}}",
        "size": float(text_size),
        "color": str(text_color),
        "anchor": str(anchor),
    }
    with suppress(AttributeError):
        layer.text.scaling = bool(text_scale_with_zoom)
    with suppress(AttributeError):
        layer.editable = False
    layer.refresh()
    return layer


def _get_or_create_shapes_layer(
    viewer: napari.Viewer,
    name: str,
    data: list[np.ndarray],
    *,
    shape_type: str = "line",
    edge_color: str = "cyan",
    edge_width: float = 1.0,
    opacity: float = 0.6,
) -> napari.layers.Shapes:
    """Create or overwrite a Shapes layer with stable styling."""
    if name in viewer.layers:
        layer = viewer.layers[name]
        if not isinstance(layer, napari.layers.Shapes):
            raise TypeError(
                f"Layer name '{name}' exists but is not a Shapes layer."
            )
        layer.data = data
        layer.shape_type = [shape_type] * len(data)
        layer.edge_color = edge_color
        layer.edge_width = float(edge_width)
        layer.opacity = float(opacity)
        layer.visible = True
    else:
        layer = viewer.add_shapes(
            data,
            shape_type=shape_type,
            name=name,
            edge_color=edge_color,
            edge_width=float(edge_width),
            face_color="transparent",
            opacity=float(opacity),
        )
    layer.editable = False
    return layer


def label_operations_widget(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    container = QWidget()
    layout = QVBoxLayout(container)
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setMaximumWidth(_PANEL_MAX_WIDTH)

    note = QLabel(
        "Label-focused operations for one or many layers: reassign class values, merge "
        "same-type Labels layers, compose distinct classes, or build vote / consensus maps."
    )
    note.setWordWrap(True)
    layout.addWidget(note)

    form = QFormLayout()
    mode_combo = QComboBox()
    mode_combo.addItems(
        [
            "Reassign Label Value",
            "Merge Layers As Same Class",
            "Compose Layers As Distinct Classes",
            "Vote Count Map",
            "Consensus Map",
        ]
    )
    form.addRow("Mode", mode_combo)
    layout.addLayout(form)

    layer_list = QListWidget()
    layer_list.setSelectionMode(QAbstractItemView.MultiSelection)
    layout.addWidget(layer_list)

    controls = QGroupBox("Operation Settings")
    controls_layout = QFormLayout(controls)
    background_spin = QSpinBox()
    background_spin.setRange(0, 65535)
    background_spin.setValue(0)
    source_value_spin = QSpinBox()
    source_value_spin.setRange(-1, 65535)
    source_value_spin.setValue(-1)
    new_value_spin = QSpinBox()
    new_value_spin.setRange(0, 65535)
    new_value_spin.setValue(1)
    merge_value_spin = QSpinBox()
    merge_value_spin.setRange(0, 65535)
    merge_value_spin.setValue(1)
    class_start_spin = QSpinBox()
    class_start_spin.setRange(1, 65535)
    class_start_spin.setValue(1)
    min_votes_spin = QSpinBox()
    min_votes_spin.setRange(1, 999)
    min_votes_spin.setValue(2)
    consensus_value_spin = QSpinBox()
    consensus_value_spin.setRange(0, 65535)
    consensus_value_spin.setValue(1)
    overwrite_check = QCheckBox("Overwrite first selected layer")
    output_name = QComboBox()
    output_name.setEditable(True)
    output_name.addItem("label_ops_result")

    controls_layout.addRow("Background value", background_spin)
    controls_layout.addRow("Source value (-1 = auto)", source_value_spin)
    controls_layout.addRow("New class value", new_value_spin)
    controls_layout.addRow("Merge class value", merge_value_spin)
    controls_layout.addRow("Class values start at", class_start_spin)
    controls_layout.addRow("Minimum votes", min_votes_spin)
    controls_layout.addRow("Consensus class value", consensus_value_spin)
    controls_layout.addRow("Output name", output_name)
    controls_layout.addRow("", overwrite_check)
    layout.addWidget(controls)

    helper = QLabel(
        "Recommended basics: relabel one layer, merge multiple same-type layers, "
        "compose distinct classes, and build vote or consensus outputs for 3+ Labels layers."
    )
    helper.setWordWrap(True)
    layout.addWidget(helper)

    row = QWidget()
    row_layout = QHBoxLayout(row)
    btn_refresh = QPushButton("Refresh")
    btn_apply = QPushButton("Apply")
    row_layout.addWidget(btn_refresh)
    row_layout.addWidget(btn_apply)
    layout.addWidget(row)

    status = QLabel("Select one or more Labels layers, choose a mode, and apply.")
    status.setWordWrap(True)
    layout.addWidget(status)

    def _eligible_layers():
        return [
            layer
            for layer in viewer.layers
            if isinstance(layer, (napari.layers.Image, napari.layers.Labels))
            and not is_internal_layer(layer)
            and _supported_2d_mask_shape(getattr(layer, "data", None)) is not None
        ]

    def _refresh_layers() -> None:
        selected = {
            getattr(item.data(Qt.UserRole), "name", item.text())
            for item in layer_list.selectedItems()
        }
        layer_list.clear()
        for layer in _eligible_layers():
            item = QListWidgetItem(layer.name)
            item.setData(Qt.UserRole, layer)
            layer_list.addItem(item)
            if layer.name in selected:
                item.setSelected(True)
        _refresh_mode_controls()

    def _selected_layers():
        layers = []
        for item in layer_list.selectedItems():
            layer = item.data(Qt.UserRole)
            if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
                with suppress(ValueError):
                    viewer.layers.index(layer)
                    layers.append(layer)
                    continue
            name = item.text()
            for candidate in viewer.layers:
                if getattr(candidate, "name", "") == name:
                    layers.append(candidate)
                    break
        return layers

    def _set_status(message: str) -> None:
        viewer.status = message
        status.setText(message)

    def _default_output_name(mode: str, layers: list[napari.layers.Layer]) -> str:
        base = layers[0].name if layers else "labels"
        if mode == "Reassign Label Value":
            return f"{base} | class_{int(new_value_spin.value())}"
        if mode == "Merge Layers As Same Class":
            return f"{base} | merged_class_{int(merge_value_spin.value())}"
        if mode == "Compose Layers As Distinct Classes":
            return f"{base} | composed_labels"
        if mode == "Vote Count Map":
            return f"{base} | vote_count"
        return f"{base} | consensus"

    def _refresh_mode_controls() -> None:
        mode = mode_combo.currentText()
        is_reassign = mode == "Reassign Label Value"
        is_merge = mode == "Merge Layers As Same Class"
        is_compose = mode == "Compose Layers As Distinct Classes"
        is_vote = mode == "Vote Count Map"
        is_consensus = mode == "Consensus Map"
        source_value_spin.setEnabled(is_reassign)
        new_value_spin.setEnabled(is_reassign)
        merge_value_spin.setEnabled(is_merge)
        class_start_spin.setEnabled(is_compose)
        min_votes_spin.setEnabled(is_consensus)
        consensus_value_spin.setEnabled(is_consensus)
        overwrite_check.setEnabled(not is_vote)
        layers = _selected_layers()
        output_name.setCurrentText(_default_output_name(mode, layers))
        helper.setText(
            {
                "Reassign Label Value": "Use one layer. Rewrites one non-background value to a new class.",
                "Merge Layers As Same Class": "Use one or more layers. Every non-zero value becomes the same class value.",
                "Compose Layers As Distinct Classes": "Use 2+ layers when each selected layer should become its own class value.",
                "Vote Count Map": "Use 2+ layers to count per-pixel agreement across many Labels layers.",
                "Consensus Map": "Use 2+ layers to keep pixels present in at least N selected Labels layers.",
            }[mode]
        )

    def _write_output(result: np.ndarray, layers: list[napari.layers.Layer], mode: str) -> None:
        if not layers:
            raise ValueError("Select one or more layers first.")
        ref = layers[0]
        target_name = output_name.currentText().strip() or _default_output_name(mode, layers)
        if overwrite_check.isChecked() and mode != "Vote Count Map":
            ref.data = result
            ref.visible = True
            _set_status(f"Updated {ref.name} with {mode.lower()}.")
            return
        viewer.add_labels(
            result,
            name=target_name,
            scale=ref.scale,
            translate=ref.translate,
        )
        _set_status(f"Created {target_name} from {len(layers)} selected layer(s).")
        _refresh_layers()

    def _apply() -> None:
        layers = _selected_layers()
        mode = mode_combo.currentText()
        if mode == "Reassign Label Value":
            if len(layers) != 1:
                raise ValueError("Select exactly one layer for relabeling.")
            result = _reclassify_single_value_labels(
                layers[0].data,
                new_value=int(new_value_spin.value()),
                background_value=int(background_spin.value()),
                source_value=None if int(source_value_spin.value()) < 0 else int(source_value_spin.value()),
            )
            _write_output(result, layers, mode)
            return
        if mode != "Merge Layers As Same Class" and len(layers) < 2:
            raise ValueError("Select at least two layers for this operation.")
        data = [layer.data for layer in layers]
        if mode == "Merge Layers As Same Class":
            if not layers:
                raise ValueError("Select at least one layer to merge into one class.")
            result = _merge_label_layers_same_class(
                data,
                target_value=int(merge_value_spin.value()),
                background_value=int(background_spin.value()),
            )
        elif mode == "Compose Layers As Distinct Classes":
            start = int(class_start_spin.value())
            class_values = [start + i for i in range(len(layers))]
            result = _compose_label_layers_distinct_classes(
                data,
                class_values=class_values,
                background_value=int(background_spin.value()),
            )
        elif mode == "Vote Count Map":
            result = _count_label_votes(data)
        else:
            result = _consensus_label_map(
                data,
                min_votes=int(min_votes_spin.value()),
                consensus_value=int(consensus_value_spin.value()),
                background_value=int(background_spin.value()),
            )
        _write_output(result, layers, mode)

    def _run_safely(fn) -> None:
        try:
            fn()
        except Exception as e:
            _set_status(str(e))

    btn_refresh.clicked.connect(_refresh_layers)
    btn_apply.clicked.connect(lambda: _run_safely(_apply))
    mode_combo.currentTextChanged.connect(lambda _text: _refresh_mode_controls())
    new_value_spin.valueChanged.connect(lambda _value: _refresh_mode_controls())
    merge_value_spin.valueChanged.connect(lambda _value: _refresh_mode_controls())
    layer_list.itemSelectionChanged.connect(_refresh_mode_controls)

    container._mode_combo = mode_combo
    container._layer_list = layer_list
    container._merge_value_spin = merge_value_spin
    container._apply_button = btn_apply
    container._layer_refresh_timer = _auto_refresh_layer_controls(
        viewer, container, _refresh_layers
    )

    _refresh_layers()
    return container


def quick_compare_toggle_widget(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    container = QWidget()
    layout = QVBoxLayout(container)
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setMaximumWidth(_PANEL_MAX_WIDTH)

    note = QLabel(
        "Quickly compare one image layer and one Labels layer without changing the "
        "selected editing layer. Use the buttons below instead of selecting layers "
        "just to press napari's visibility hotkey."
    )
    note.setWordWrap(True)
    layout.addWidget(note)

    form = QFormLayout()
    image_combo = QComboBox()
    mask_combo = QComboBox()
    keep_mask_selected = QCheckBox("Keep Labels layer selected for editing")
    keep_mask_selected.setChecked(True)
    dim_opacity_spin = QSpinBox()
    dim_opacity_spin.setRange(0, 100)
    dim_opacity_spin.setSingleStep(5)
    dim_opacity_spin.setValue(15)
    effect_seconds_combo = QComboBox()
    for seconds in (1, 5, 10):
        effect_seconds_combo.addItem(f"{seconds} second", seconds)
    form.addRow("Image layer", image_combo)
    form.addRow("Labels layer", mask_combo)
    form.addRow("", keep_mask_selected)
    form.addRow("Dim Labels opacity (%)", dim_opacity_spin)
    form.addRow("Effect duration", effect_seconds_combo)
    layout.addLayout(form)

    row1 = QWidget()
    row1_layout = QHBoxLayout(row1)
    btn_refresh = QPushButton("Refresh")
    btn_toggle_mask = QPushButton("Toggle Labels")
    btn_show_both = QPushButton("Show Both")
    row1_layout.addWidget(btn_refresh)
    row1_layout.addWidget(btn_toggle_mask)
    row1_layout.addWidget(btn_show_both)
    layout.addWidget(row1)

    row2 = QWidget()
    row2_layout = QHBoxLayout(row2)
    btn_image_only = QPushButton("Image Only")
    btn_mask_only = QPushButton("Labels Only")
    btn_dim_mask = QPushButton("Dim Labels")
    btn_restore = QPushButton("Restore")
    row2_layout.addWidget(btn_image_only)
    row2_layout.addWidget(btn_mask_only)
    row2_layout.addWidget(btn_dim_mask)
    row2_layout.addWidget(btn_restore)
    layout.addWidget(row2)

    row3 = QWidget()
    row3_layout = QHBoxLayout(row3)
    btn_pulse_mask = QPushButton("Pulse Labels")
    btn_blink_mask = QPushButton("Blink Labels")
    btn_peek_mask = QPushButton("Peek Labels")
    row3_layout.addWidget(btn_pulse_mask)
    row3_layout.addWidget(btn_blink_mask)
    row3_layout.addWidget(btn_peek_mask)
    layout.addWidget(row3)

    status = QLabel("Select an image/Labels pair, then use the quick compare buttons.")
    status.setWordWrap(True)
    layout.addWidget(status)

    state = SimpleNamespace(
        mask_opacity_by_name={},
        animation_timer=QTimer(container),
        animation_mode=None,
        animation_step=0,
        animation_steps=0,
        animation_mask_layer_name=None,
        animation_image_layer_name=None,
    )
    state.animation_timer.setInterval(100)

    def _eligible_image_layers():
        return [
            layer
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Image)
            and not is_internal_layer(layer)
        ]

    def _eligible_mask_layers():
        return [
            layer
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Labels)
            and not is_internal_layer(layer)
        ]

    def _refresh_targets() -> None:
        current_image = image_combo.currentText()
        current_mask = mask_combo.currentText()
        image_combo.clear()
        mask_combo.clear()
        for layer in _eligible_image_layers():
            image_combo.addItem(layer.name)
        for layer in _eligible_mask_layers():
            mask_combo.addItem(layer.name)
        if current_image:
            idx = image_combo.findText(current_image)
            if idx >= 0:
                image_combo.setCurrentIndex(idx)
        if current_mask:
            idx = mask_combo.findText(current_mask)
            if idx >= 0:
                mask_combo.setCurrentIndex(idx)

    def _target_layer(combo: QComboBox):
        name = combo.currentText().strip()
        if not name:
            return None
        with suppress(KeyError):
            return viewer.layers[name]
        return None

    def _set_status(message: str) -> None:
        viewer.status = message
        status.setText(message)

    def _selected_pair():
        image_layer = _target_layer(image_combo)
        mask_layer = _target_layer(mask_combo)
        if image_layer is None or mask_layer is None:
            raise ValueError("Select both an image layer and a Labels layer.")
        return image_layer, mask_layer

    def _remember_mask_opacity(mask_layer) -> None:
        key = str(mask_layer.name)
        if key not in state.mask_opacity_by_name:
            state.mask_opacity_by_name[key] = float(mask_layer.opacity)

    def _restore_mask_selection(mask_layer) -> None:
        if keep_mask_selected.isChecked():
            viewer.layers.selection.active = mask_layer

    def _effect_seconds() -> int:
        return int(effect_seconds_combo.currentData())

    def _stop_animation() -> None:
        if state.animation_timer.isActive():
            state.animation_timer.stop()
        if state.animation_mask_layer_name and state.animation_mask_layer_name in viewer.layers:
            mask_layer = viewer.layers[state.animation_mask_layer_name]
            if state.animation_image_layer_name and state.animation_image_layer_name in viewer.layers:
                viewer.layers[state.animation_image_layer_name].visible = True
            mask_layer.visible = True
            if state.animation_mask_layer_name in state.mask_opacity_by_name:
                mask_layer.opacity = float(
                    state.mask_opacity_by_name[state.animation_mask_layer_name]
                )
            _restore_mask_selection(mask_layer)
        state.animation_mode = None
        state.animation_step = 0
        state.animation_steps = 0
        state.animation_mask_layer_name = None
        state.animation_image_layer_name = None

    def _start_animation(mode: str) -> None:
        image_layer, mask_layer = _selected_pair()
        _remember_mask_opacity(mask_layer)
        _stop_animation()
        image_layer.visible = True
        mask_layer.visible = True
        state.animation_mode = str(mode)
        state.animation_step = 0
        state.animation_steps = max(2, int(_effect_seconds() * 10))
        state.animation_mask_layer_name = str(mask_layer.name)
        state.animation_image_layer_name = str(image_layer.name)
        _restore_mask_selection(mask_layer)
        state.animation_timer.start()

    def _tick_animation() -> None:
        if not state.animation_mask_layer_name or not state.animation_image_layer_name:
            _stop_animation()
            return
        if (
            state.animation_mask_layer_name not in viewer.layers
            or state.animation_image_layer_name not in viewer.layers
        ):
            _stop_animation()
            return

        mask_layer = viewer.layers[state.animation_mask_layer_name]
        image_layer = viewer.layers[state.animation_image_layer_name]
        image_layer.visible = True
        original_opacity = float(
            state.mask_opacity_by_name.get(mask_layer.name, mask_layer.opacity)
        )
        dim_opacity = float(dim_opacity_spin.value()) / 100.0
        progress = state.animation_step / max(1, state.animation_steps - 1)

        if state.animation_mode == "pulse":
            if progress <= 0.5:
                phase = progress / 0.5
            else:
                phase = (1.0 - progress) / 0.5
            mask_layer.visible = True
            mask_layer.opacity = dim_opacity + (original_opacity - dim_opacity) * max(
                0.0, min(1.0, phase)
            )
        elif state.animation_mode == "blink":
            blink_on = (state.animation_step % 2) == 0
            mask_layer.visible = blink_on
            if blink_on:
                mask_layer.opacity = original_opacity
        elif state.animation_mode == "peek":
            mask_layer.visible = False
        else:
            _stop_animation()
            return

        _restore_mask_selection(mask_layer)
        state.animation_step += 1
        if state.animation_step >= state.animation_steps:
            finished_mode = state.animation_mode
            _stop_animation()
            _set_status(
                f"{finished_mode.capitalize()} Labels effect finished for {mask_layer.name}."
            )

    def _show_both() -> None:
        image_layer, mask_layer = _selected_pair()
        _stop_animation()
        _remember_mask_opacity(mask_layer)
        image_layer.visible = True
        mask_layer.visible = True
        mask_layer.opacity = float(state.mask_opacity_by_name.get(mask_layer.name, mask_layer.opacity))
        _restore_mask_selection(mask_layer)
        _set_status(f"Showing both {image_layer.name} and {mask_layer.name}.")

    def _toggle_mask() -> None:
        image_layer, mask_layer = _selected_pair()
        _stop_animation()
        _remember_mask_opacity(mask_layer)
        image_layer.visible = True
        mask_layer.visible = not bool(mask_layer.visible)
        _restore_mask_selection(mask_layer)
        state_text = "visible" if mask_layer.visible else "hidden"
        _set_status(f"Labels layer {mask_layer.name} is now {state_text} over {image_layer.name}.")

    def _show_image_only() -> None:
        image_layer, mask_layer = _selected_pair()
        _stop_animation()
        _remember_mask_opacity(mask_layer)
        image_layer.visible = True
        mask_layer.visible = False
        _restore_mask_selection(mask_layer)
        _set_status(f"Showing image only: {image_layer.name}.")

    def _show_mask_only() -> None:
        image_layer, mask_layer = _selected_pair()
        _stop_animation()
        _remember_mask_opacity(mask_layer)
        image_layer.visible = False
        mask_layer.visible = True
        mask_layer.opacity = float(state.mask_opacity_by_name.get(mask_layer.name, mask_layer.opacity))
        _restore_mask_selection(mask_layer)
        _set_status(f"Showing Labels layer only: {mask_layer.name}.")

    def _dim_mask() -> None:
        image_layer, mask_layer = _selected_pair()
        _stop_animation()
        _remember_mask_opacity(mask_layer)
        image_layer.visible = True
        mask_layer.visible = True
        mask_layer.opacity = float(dim_opacity_spin.value()) / 100.0
        _restore_mask_selection(mask_layer)
        _set_status(
            f"Dimmed {mask_layer.name} to {int(dim_opacity_spin.value())}% opacity over {image_layer.name}."
        )

    def _restore_pair() -> None:
        image_layer, mask_layer = _selected_pair()
        _stop_animation()
        image_layer.visible = True
        mask_layer.visible = True
        if mask_layer.name in state.mask_opacity_by_name:
            mask_layer.opacity = float(state.mask_opacity_by_name[mask_layer.name])
        _restore_mask_selection(mask_layer)
        _set_status(f"Restored paired view for {image_layer.name} and {mask_layer.name}.")

    def _pulse_mask() -> None:
        _start_animation("pulse")
        _set_status(f"Pulsing Labels layer for {_effect_seconds()} second(s).")

    def _blink_mask() -> None:
        _start_animation("blink")
        _set_status(f"Blinking Labels layer for {_effect_seconds()} second(s).")

    def _peek_mask() -> None:
        _start_animation("peek")
        _set_status(f"Peeking image for {_effect_seconds()} second(s) with Labels layer hidden.")

    def _run_safely(fn) -> None:
        try:
            fn()
        except Exception as e:
            _set_status(str(e))

    state.animation_timer.timeout.connect(_tick_animation)
    btn_refresh.clicked.connect(_refresh_targets)
    btn_toggle_mask.clicked.connect(lambda: _run_safely(_toggle_mask))
    btn_show_both.clicked.connect(lambda: _run_safely(_show_both))
    btn_image_only.clicked.connect(lambda: _run_safely(_show_image_only))
    btn_mask_only.clicked.connect(lambda: _run_safely(_show_mask_only))
    btn_dim_mask.clicked.connect(lambda: _run_safely(_dim_mask))
    btn_restore.clicked.connect(lambda: _run_safely(_restore_pair))
    btn_pulse_mask.clicked.connect(lambda: _run_safely(_pulse_mask))
    btn_blink_mask.clicked.connect(lambda: _run_safely(_blink_mask))
    btn_peek_mask.clicked.connect(lambda: _run_safely(_peek_mask))
    container.destroyed.connect(lambda *_args: _stop_animation())
    container._layer_refresh_timer = _auto_refresh_layer_controls(
        viewer, container, _refresh_targets
    )

    _refresh_targets()
    return container


def component_operations_widget(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    container = QWidget()
    layout = QVBoxLayout(container)
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setMaximumWidth(_PANEL_MAX_WIDTH)

    source_form = QFormLayout()
    target_combo = QComboBox()
    target_combo.setToolTip("Labels layer to edit and analyze.")
    source_row = QWidget()
    source_row_layout = QHBoxLayout(source_row)
    source_row_layout.setContentsMargins(0, 0, 0, 0)
    source_row_layout.setSpacing(6)
    source_row_layout.addWidget(target_combo, 1)
    btn_refresh = QPushButton("Refresh layers")
    btn_refresh.setToolTip(
        "Synchronize all layer selectors now. Selectors also update "
        "automatically when layers are opened, added, removed, or renamed."
    )
    source_row_layout.addWidget(btn_refresh)
    source_form.addRow("Labels layer", source_row)
    layout.addLayout(source_form)

    workflow_tabs = QTabWidget()
    workflow_tabs.setDocumentMode(True)
    annotate_page = QWidget()
    annotate_layout = QVBoxLayout(annotate_page)
    annotate_layout.setContentsMargins(0, 6, 0, 0)
    qc_page = QWidget()
    qc_layout = QVBoxLayout(qc_page)
    qc_layout.setContentsMargins(0, 6, 0, 0)
    workflow_tabs.addTab(annotate_page, "Annotate")
    workflow_tabs.addTab(qc_page, "Grid & Components")
    workflow_tabs.setTabToolTip(
        0,
        "Paint and erase the selected Labels layer with bounded-memory "
        "editing when needed.",
    )
    workflow_tabs.setTabToolTip(
        1,
        "Register grid addresses, find connected components, and review or "
        "correct selected regions.",
    )
    layout.addWidget(workflow_tabs, 1)

    form = QFormLayout()
    assign_grid_check = QCheckBox("Assign grid addresses to results")
    assign_grid_check.setChecked(False)
    assign_grid_check.setToolTip(
        "Register each component with the grid row, column, and cell address "
        "containing its centroid."
    )
    display_grid_check = QCheckBox("Show address grid on canvas")
    display_grid_check.setChecked(False)
    display_grid_check.setToolTip(
        "Draw grid lines and readable cell IDs on the canvas. IDs adapt to "
        "zoom and appear only around the visible area. Leave off when only "
        "table IDs are needed."
    )
    grid_y_spin = QSpinBox()
    grid_y_spin.setRange(1, 100000)
    grid_y_spin.setSingleStep(10)
    grid_y_spin.setValue(100)
    grid_y_spin.setToolTip("Grid-cell height in source-image pixels.")
    grid_x_spin = QSpinBox()
    grid_x_spin.setRange(1, 100000)
    grid_x_spin.setSingleStep(10)
    grid_x_spin.setValue(100)
    grid_x_spin.setToolTip("Grid-cell width in source-image pixels.")
    form.addRow("", assign_grid_check)
    form.addRow("", display_grid_check)
    form.addRow("Cell height (px)", grid_y_spin)
    form.addRow("Cell width (px)", grid_x_spin)

    edit_group = QGroupBox("Memory-controlled editing")
    edit_form = QFormLayout(edit_group)
    editing_strategy_combo = QComboBox()
    editing_strategy_combo.addItem("Automatic (recommended)", "auto")
    editing_strategy_combo.addItem("Local-area editing", "tiled")
    editing_strategy_combo.addItem("Full-layer editing", "direct")
    editing_strategy_combo.setToolTip(
        "Automatic uses a bounded in-memory editable area when a full editing "
        "copy would be costly, and standard napari editing otherwise."
    )
    edit_tile_size_combo = QComboBox()
    edit_tile_size_combo.addItem(
        f"Automatic — {DEFAULT_TILE_SIZE} px", 0
    )
    edit_tile_size_combo.addItem("2048 × 2048 px", 2048)
    edit_tile_size_combo.addItem("4096 × 4096 px", 4096)
    edit_tile_size_combo.setToolTip(
        "Smaller editable areas use less memory; larger areas need fewer "
        "automatic reloads while panning."
    )
    show_edit_boundary_check = QCheckBox("Show editable boundary")
    show_edit_boundary_check.setChecked(True)
    show_edit_boundary_check.setToolTip(
        "Outline the area currently loaded for fast Paint and Erase tools."
    )
    auto_save_edit_check = QCheckBox("Apply edits automatically")
    auto_save_edit_check.setChecked(True)
    auto_save_edit_check.setToolTip(
        "When on, save 400 ms after a completed stroke. When off, edits "
        "remain in the editable area until Apply edits is clicked."
    )
    dynamic_edit_status = QLabel("Select a source Labels layer.")
    dynamic_edit_status.setWordWrap(True)
    edit_action_row = QWidget()
    edit_action_layout = QHBoxLayout(edit_action_row)
    edit_action_layout.setContentsMargins(0, 0, 0, 0)
    edit_action_layout.setSpacing(6)
    save_edit_btn = QPushButton("Apply edits")
    undo_edit_btn = QPushButton("Undo")
    redo_edit_btn = QPushButton("Redo")
    save_edit_btn.setToolTip(
        "Write pending editable-area changes to the source layer immediately."
    )
    undo_edit_btn.setToolTip(
        "Undo the most recent paint or erase stroke and synchronize the source."
    )
    redo_edit_btn.setToolTip(
        "Restore the most recently undone stroke and synchronize the source."
    )
    for button in (save_edit_btn, undo_edit_btn, redo_edit_btn):
        button.setEnabled(False)
        edit_action_layout.addWidget(button, 1)
    edit_form.addRow("Editing mode", editing_strategy_combo)
    edit_form.addRow("Local area size", edit_tile_size_combo)
    edit_form.addRow("", show_edit_boundary_check)
    edit_form.addRow("", auto_save_edit_check)
    edit_form.addRow(edit_action_row)
    edit_form.addRow("Status", dynamic_edit_status)
    annotate_layout.addWidget(edit_group)
    annotate_layout.addStretch(1)

    analysis_group = QGroupBox("Grid registration and component QC")
    analysis_layout = QVBoxLayout(analysis_group)
    analysis_note = QLabel(
        "Register full-mask locations with reproducible grid addresses. Find "
        "connected components, trace reviewer findings to exact cells, and "
        "select results in the table or canvas to revisit, delete, or copy them."
    )
    analysis_note.setWordWrap(True)
    analysis_layout.addWidget(analysis_note)
    analysis_layout.addLayout(form)
    qc_layout.addWidget(analysis_group, 1)

    action_row = QWidget()
    action_layout = QHBoxLayout(action_row)
    action_layout.setContentsMargins(0, 0, 0, 0)
    action_layout.setSpacing(6)
    btn_analyze = QPushButton("Find components")
    btn_delete = QPushButton("Delete selected")
    btn_copy = QPushButton("Copy selected")
    btn_copy_last = QPushButton("Copy again")
    action_layout.addWidget(btn_analyze)
    action_layout.addWidget(btn_delete)
    analysis_layout.addWidget(action_row)

    copy_action_row = QWidget()
    copy_action_layout = QHBoxLayout(copy_action_row)
    copy_action_layout.setContentsMargins(0, 0, 0, 0)
    copy_action_layout.setSpacing(6)
    copy_action_layout.addWidget(btn_copy)
    copy_action_layout.addWidget(btn_copy_last)
    analysis_layout.addWidget(copy_action_row)

    btn_analyze.setToolTip(
        "Detect and measure connected components in the selected 2D Labels layer."
    )
    btn_delete.setToolTip(
        "Set all pixels in the selected components to the background value (0)."
    )
    btn_copy.setToolTip(
        "Copy the selected components and their label values to the chosen "
        "destination layer."
    )
    btn_copy_last.setToolTip(
        "Repeat the previous successful copy using the same destination layer."
    )

    copy_target_combo = QComboBox()
    copy_target_combo.setToolTip(
        "Labels layer that will receive copied components. It must have the "
        "same 2D shape as the source layer."
    )
    form.addRow("Copy destination", copy_target_combo)

    click_select_check = QCheckBox("Select components from canvas")
    click_select_check.setToolTip(
        "Click a labeled object in the canvas to select its connected component "
        "in the table."
    )
    analysis_layout.addWidget(click_select_check)

    append_click_select_check = QCheckBox("Add each click to selection")
    append_click_select_check.setToolTip(
        "Keep earlier component selections when clicking additional objects."
    )
    analysis_layout.addWidget(append_click_select_check)

    selection_summary = QLabel("Selected components: 0")
    selection_summary.setWordWrap(True)
    selection_summary.setStyleSheet(
        "QLabel {"
        " background-color: #fff3bf;"
        " color: #5f3b00;"
        " border: 1px solid #f59f00;"
        " border-radius: 4px;"
        " padding: 4px 6px;"
        " font-weight: 600;"
        "}"
    )
    analysis_layout.addWidget(selection_summary)

    results_help = QLabel(
        "Results — click a column heading to sort; double-click a row to "
        "locate that component on the canvas."
    )
    results_help.setWordWrap(True)
    analysis_layout.addWidget(results_help)
    component_table = _ComponentTableWidget(
        theme=str(getattr(viewer, "theme", "dark"))
    )
    component_table.set_grid_columns_visible(False)
    component_table.setMinimumHeight(220)
    analysis_layout.addWidget(component_table, 1)

    analysis_progress = QProgressBar()
    analysis_progress.setVisible(False)
    analysis_layout.addWidget(analysis_progress)
    btn_cancel_analysis = QPushButton("Cancel large-image analysis")
    btn_cancel_analysis.setVisible(False)
    analysis_layout.addWidget(btn_cancel_analysis)

    page_row = QWidget()
    page_layout = QHBoxLayout(page_row)
    page_label = QLabel("Components")
    page_spin = QSpinBox()
    page_spin.setRange(1, 1)
    page_layout.addWidget(page_label)
    page_layout.addWidget(QLabel("Page"))
    page_layout.addWidget(page_spin)
    page_row.setVisible(False)
    analysis_layout.addWidget(page_row)

    euler_help = QLabel(
        "Euler number = 1 minus the number of holes: 1 = no holes, "
        "0 = one hole, and -1 = two holes. Right-click selected components "
        "to copy them to a compatible Labels layer."
    )
    euler_help.setWordWrap(True)
    analysis_layout.addWidget(euler_help)

    status = QLabel("Choose a Labels layer, then click Find components.")
    status.setWordWrap(True)
    analysis_layout.addWidget(status)

    state = SimpleNamespace(
        analyzed_layer_name=None,
        fast_index=None,
        callback_registered=False,
        selected_component_ids=set(),
        syncing_table_selection=False,
        last_copy_target_name=None,
        displayed_grid_layer_names=None,
        large_records=[],
        large_ids=[],
        closed=False,
    )
    grid_label_timer = QTimer(container)
    grid_label_timer.setSingleShot(True)
    grid_label_timer.setInterval(150)

    def _eligible_layers():
        layers = [
            layer
            for layer in viewer.layers
            if isinstance(layer, (napari.layers.Image, napari.layers.Labels))
            and not is_internal_layer(layer)
        ]
        return sorted(layers, key=lambda layer: not isinstance(layer, napari.layers.Labels))

    def _target_layer():
        name = target_combo.currentText().strip()
        if not name:
            return None
        with suppress(KeyError):
            return viewer.layers[name]
        return None

    def _editable_target_layer():
        layer = _target_layer()
        if isinstance(layer, napari.layers.Labels) and not is_internal_layer(layer):
            return layer
        return None

    def _store_grid_registration() -> None:
        layer = _target_layer()
        if not isinstance(layer, napari.layers.Labels) or is_internal_layer(layer):
            return
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata["label_assistant_grid"] = {
            "cell_height": int(grid_y_spin.value()),
            "cell_width": int(grid_x_spin.value()),
            "assign_addresses": bool(assign_grid_check.isChecked()),
            "show_overlay": bool(display_grid_check.isChecked()),
        }
        layer.metadata = metadata

    def _restore_grid_registration() -> None:
        layer = _target_layer()
        metadata = getattr(layer, "metadata", {}) or {}
        registration = (
            metadata.get("label_assistant_grid", {})
            if isinstance(metadata, dict)
            else {}
        )
        controls = (
            grid_y_spin,
            grid_x_spin,
            assign_grid_check,
            display_grid_check,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            grid_y_spin.setValue(int(registration.get("cell_height", 100)))
            grid_x_spin.setValue(int(registration.get("cell_width", 100)))
            assign_grid_check.setChecked(
                bool(registration.get("assign_addresses", False))
            )
            display_grid_check.setChecked(
                bool(registration.get("show_overlay", False))
            )
        finally:
            for control in controls:
                control.blockSignals(False)
        component_table.set_grid_columns_visible(
            assign_grid_check.isChecked()
        )

    def _eligible_copy_targets(source_layer) -> list[napari.layers.Layer]:
        if source_layer is None:
            return []
        source_shape = _supported_2d_mask_shape(getattr(source_layer, "data", None))
        if source_shape is None:
            return []
        targets: list[napari.layers.Layer] = []
        for layer in viewer.layers:
            if not isinstance(layer, napari.layers.Labels):
                continue
            if is_internal_layer(layer):
                continue
            if layer is source_layer:
                continue
            if _supported_2d_mask_shape(getattr(layer, "data", None)) != source_shape:
                continue
            targets.append(layer)
        return targets

    def _refresh_copy_targets() -> None:
        current_copy = getattr(
            copy_target_combo.currentData(),
            "name",
            copy_target_combo.currentText() or state.last_copy_target_name,
        )
        source_layer = _target_layer()
        copy_target_combo.blockSignals(True)
        copy_target_combo.clear()
        for layer in _eligible_copy_targets(source_layer):
            copy_target_combo.addItem(layer.name, layer)
        if current_copy:
            idx = copy_target_combo.findText(current_copy)
            if idx >= 0:
                copy_target_combo.setCurrentIndex(idx)
        copy_target_combo.blockSignals(False)

    def _refresh_targets() -> None:
        current = target_combo.currentText()
        target_combo.blockSignals(True)
        target_combo.clear()
        for layer in _eligible_layers():
            target_combo.addItem(layer.name)
        if current:
            idx = target_combo.findText(current)
            if idx >= 0:
                target_combo.setCurrentIndex(idx)
        target_combo.blockSignals(False)
        _refresh_copy_targets()

    def _copy_target_layer():
        layer = copy_target_combo.currentData()
        if isinstance(layer, napari.layers.Labels):
            with suppress(ValueError):
                viewer.layers.index(layer)
                return layer
        name = copy_target_combo.currentText().strip()
        if not name:
            return None
        with suppress(KeyError):
            return viewer.layers[name]
        return None

    def _set_status(message: str) -> None:
        viewer.status = message
        status.setText(message)

    def _set_dynamic_edit_status(
        message: str, state_name: str = "neutral"
    ) -> None:
        colors = {
            "ready": ("#dff4e5", "#245c36"),
            "busy": ("#fff3cd", "#735c0f"),
            "error": ("#f8d7da", "#7a2830"),
            "neutral": ("#e9edf2", "#39424e"),
        }
        background, foreground = colors.get(
            state_name, colors["neutral"]
        )
        dynamic_edit_status.setText(message)
        dynamic_edit_status.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground}; "
            "border-radius: 3px; padding: 4px 6px; }}"
        )

    def _grid_layer_names(layer) -> tuple[str, str]:
        return (
            f"{layer.name} | grid_overlay",
            f"{layer.name} | grid_labels",
        )

    def _show_grid_overlay() -> None:
        layer = _target_layer()
        if layer is None:
            raise ValueError("Select a Labels layer before showing the grid.")
        shape = _supported_2d_mask_shape(getattr(layer, "data", None))
        if shape is None:
            raise ValueError("The selected layer is not a supported 2D Labels layer.")
        step_y = int(grid_y_spin.value())
        step_x = int(grid_x_spin.value())
        estimated_lines = (shape[0] - 1) // step_y + (shape[1] - 1) // step_x + 4
        if estimated_lines > 2000:
            raise ValueError(
                f"Grid would draw {estimated_lines} lines and may be slow. "
                "Increase Grid height or Grid width."
            )
        grid_name, label_name = _grid_layer_names(layer)
        new_names = (grid_name, label_name)
        previous_names = state.displayed_grid_layer_names
        if previous_names and previous_names != new_names:
            for previous_name in previous_names:
                with suppress(KeyError):
                    viewer.layers[previous_name].visible = False
        state.displayed_grid_layer_names = new_names
        grid_layer = _get_or_create_shapes_layer(
            viewer,
            grid_name,
            _build_grid_shapes(shape, step_y, step_x),
            edge_color="cyan",
            edge_width=1.0,
            opacity=0.55,
        )
        grid_metadata = dict(getattr(grid_layer, "metadata", {}) or {})
        grid_metadata[INTERNAL_LAYER_ROLE_KEY] = "grid_overlay"
        grid_metadata["source_layer_name"] = layer.name
        grid_layer.metadata = grid_metadata
        grid_layer.scale = np.asarray(layer.scale)[-2:]
        grid_layer.translate = np.asarray(layer.translate)[-2:]

    def _canvas_viewport_size() -> tuple[float, float]:
        canvas = viewer.window._qt_viewer.canvas
        size = canvas.size
        if hasattr(size, "width") and hasattr(size, "height"):
            return float(size.height()), float(size.width())
        width, height = size
        return float(height), float(width)

    def _refresh_grid_labels() -> None:
        if not display_grid_check.isChecked():
            return
        layer = _target_layer()
        if layer is None:
            return
        shape = _supported_2d_mask_shape(getattr(layer, "data", None))
        if shape is None:
            return
        viewport_height, viewport_width = _canvas_viewport_size()
        center = np.asarray(viewer.camera.center, dtype=float).ravel()
        if center.size < 2:
            return
        scale = np.asarray(layer.scale, dtype=float).ravel()[-2:]
        translate = np.asarray(layer.translate, dtype=float).ravel()[-2:]
        center_y = (float(center[-2]) - float(translate[-2])) / float(scale[-2])
        center_x = (float(center[-1]) - float(translate[-1])) / float(scale[-1])
        label_layout = build_visible_grid_label_layout(
            shape,
            int(grid_y_spin.value()),
            int(grid_x_spin.value()),
            center_y=center_y,
            center_x=center_x,
            viewport_height_px=viewport_height,
            viewport_width_px=viewport_width,
            zoom=float(viewer.camera.zoom),
            scale_y=float(scale[-2]),
            scale_x=float(scale[-1]),
            max_labels=150,
        )
        _grid_name, label_name = _grid_layer_names(layer)
        if not label_layout.labels:
            with suppress(KeyError):
                viewer.layers[label_name].visible = False
            return
        label_layer = _get_or_create_points_layer(
            viewer,
            label_name,
            label_layout.points,
            {"grid_id": list(label_layout.labels)},
            text_field="grid_id",
            text_size=9.0,
            text_scale_with_zoom=False,
            text_color="cyan",
            anchor="center",
        )
        label_metadata = dict(getattr(label_layer, "metadata", {}) or {})
        label_metadata[INTERNAL_LAYER_ROLE_KEY] = "grid_labels"
        label_metadata["source_layer_name"] = layer.name
        label_layer.metadata = label_metadata
        label_layer.scale = scale
        label_layer.translate = translate
        label_layer.opacity = float(label_layout.opacity)
        label_layer.visible = True

    def _schedule_grid_label_update(*_args) -> None:
        if display_grid_check.isChecked():
            grid_label_timer.start()

    def _hide_grid_overlay() -> None:
        names = state.displayed_grid_layer_names
        if names is None:
            layer = _target_layer()
            names = _grid_layer_names(layer) if layer is not None else ()
        for name in names:
            with suppress(KeyError):
                viewer.layers[name].visible = False
        state.displayed_grid_layer_names = None

    def _selected_component_ids() -> list[int]:
        return sorted(int(value) for value in state.selected_component_ids)

    def _update_selection_summary() -> None:
        ids = _selected_component_ids()
        if not ids:
            selection_summary.setText("Selected components: 0")
            return
        preview = ", ".join(str(component_id) for component_id in ids[:8])
        if len(ids) > 8:
            preview = f"{preview}, ..."
        selection_summary.setText(
            f"Selected components: {len(ids)} | IDs: {preview}"
        )

    def _sync_table_selection_from_state() -> None:
        state.syncing_table_selection = True
        try:
            component_table.clearSelection()
            for component_id in _selected_component_ids():
                component_table.select_component_id(component_id, append=True)
        finally:
            state.syncing_table_selection = False
        _update_selection_summary()

    def _on_table_selection_changed() -> None:
        if state.syncing_table_selection:
            return
        state.selected_component_ids = set(
            component_table.selected_component_ids()
        )
        _update_selection_summary()

    def _render_component_page() -> None:
        index = state.fast_index
        if index is None:
            component_table.set_records([])
            page_row.setVisible(False)
            return
        if isinstance(index, LargeComponentIndex):
            records = state.large_records
            pages = max(1, (len(records) + LARGE_TABLE_PAGE_SIZE - 1) // LARGE_TABLE_PAGE_SIZE)
            page_spin.blockSignals(True)
            page_spin.setRange(1, pages)
            page_spin.blockSignals(False)
            page = int(page_spin.value()) - 1
            start = page * LARGE_TABLE_PAGE_SIZE
            state.syncing_table_selection = True
            try:
                component_table.set_records(
                    records[start : start + LARGE_TABLE_PAGE_SIZE]
                )
            finally:
                state.syncing_table_selection = False
            page_label.setText(f"{len(records):,} components; {LARGE_TABLE_PAGE_SIZE:,} per page")
            page_row.setVisible(pages > 1)
        else:
            state.syncing_table_selection = True
            try:
                component_table.set_records(index.active_records())
            finally:
                state.syncing_table_selection = False
            page_row.setVisible(False)
        _sync_table_selection_from_state()

    def _reassign_grid_ids() -> None:
        index = state.fast_index
        if index is None:
            return
        if assign_grid_check.isChecked():
            step_y = int(grid_y_spin.value())
            step_x = int(grid_x_spin.value())
        else:
            step_y = step_x = None
        if isinstance(index, LargeComponentIndex):
            index.reassign_grid(step_y, step_x)
        else:
            for record in index.records.values():
                if step_y is None or step_x is None:
                    record.grid_row = record.grid_col = None
                    record.grid_id = ""
                else:
                    row, col, grid_id = _grid_index_from_xy(
                        record.centroid_y, record.centroid_x, step_y, step_x
                    )
                    record.grid_row = row
                    record.grid_col = col
                    record.grid_id = grid_id
        component_table.set_grid_columns_visible(assign_grid_check.isChecked())
        _render_component_page()

    def _invalidate_analysis(message: str | None = None) -> None:
        old_index = state.fast_index
        state.analyzed_layer_name = None
        state.fast_index = None
        state.selected_component_ids.clear()
        state.large_records = []
        state.large_ids = []
        if isinstance(old_index, LargeComponentIndex):
            old_index.close()
        component_table.set_records([])
        page_spin.blockSignals(True)
        page_spin.setValue(1)
        page_spin.blockSignals(False)
        page_row.setVisible(False)
        _update_selection_summary()
        if message:
            _set_status(message)

    def _locate_component(component_id: int) -> None:
        if state.fast_index is None:
            return
        record = state.fast_index.records.get(int(component_id))
        if record is None:
            return
        viewer.camera.center = (
            float(record.centroid_y),
            float(record.centroid_x),
        )
        if record.grid_id:
            _set_status(
                f"Centered component {component_id} at {record.grid_id} "
                f"(y={record.centroid_y:.1f}, x={record.centroid_x:.1f})."
            )

    component_table._locate_callback = _locate_component

    def _on_large_ready(layer_name: str, index: LargeComponentIndex) -> None:
        layer = _target_layer()
        if layer is None or layer.name != layer_name or state.closed:
            index.close()
            _set_status("Analysis finished, but the selected layer changed; result discarded.")
            return
        state.analyzed_layer_name = layer_name
        state.fast_index = index
        state.large_records = index.active_records()
        state.large_ids = [record.component_id for record in state.large_records]
        _reassign_grid_ids()
        _set_status(
            f"Analyzed {len(index.records):,} connected component(s) across the "
            f"full {index.shape[0]:,} x {index.shape[1]:,} Labels layer. "
            "Tile seams were joined; results are paged."
        )

    large_controller = LargeComponentController(
        progress=analysis_progress,
        cancel_button=btn_cancel_analysis,
        analyze_button=btn_analyze,
        status=_set_status,
        on_ready=_on_large_ready,
    )
    dynamic_edit_controller = DynamicLabelsEditController(
        viewer=viewer,
        parent=container,
        source_layer=_editable_target_layer,
        strategy=editing_strategy_combo,
        tile_size=edit_tile_size_combo,
        show_boundary=show_edit_boundary_check,
        auto_save=auto_save_edit_check,
        status=_set_dynamic_edit_status,
        action_buttons=(save_edit_btn, undo_edit_btn, redo_edit_btn),
    )
    edit_controllers = getattr(
        viewer, "_label_assistant_edit_controllers", None
    )
    if edit_controllers is None:
        edit_controllers = []
        viewer._label_assistant_edit_controllers = edit_controllers
    edit_controllers.append(dynamic_edit_controller)
    save_edit_btn.clicked.connect(dynamic_edit_controller.save_now)
    undo_edit_btn.clicked.connect(dynamic_edit_controller.undo_stroke)
    redo_edit_btn.clicked.connect(dynamic_edit_controller.redo_stroke)
    auto_save_edit_check.toggled.connect(
        dynamic_edit_controller.on_auto_save_toggled
    )

    def _analyze() -> None:
        layer = _target_layer()
        if layer is None:
            raise ValueError("Select a Labels layer to analyze.")
        if large_controller.busy:
            raise ValueError("Wait for the current large-label operation to finish.")
        source = layer.data[0] if isinstance(layer.data, (list, tuple)) else layer.data
        shape = _supported_2d_mask_shape(source)
        if shape is None:
            raise ValueError("Analyze requires a 2D Labels layer.")
        if isinstance(layer, napari.layers.Image) and looks_like_grayscale_image(source):
            raise ValueError(
                "This target looks like a grayscale image, not a "
                "Labels layer. Choose the correct source Labels layer, then analyze it."
            )
        if shape[0] * shape[1] > LARGE_ANALYSIS_PIXEL_THRESHOLD:
            _invalidate_analysis()
            large_controller.start(source, layer.name, layer=layer)
            return
        _invalidate_analysis()
        use_grid = bool(assign_grid_check.isChecked())
        fast_index = _build_fast_component_index(
            layer.data,
            grid_step_y=int(grid_y_spin.value()) if use_grid else None,
            grid_step_x=int(grid_x_spin.value()) if use_grid else None,
        )
        state.analyzed_layer_name = layer.name
        state.fast_index = fast_index
        component_table.set_grid_columns_visible(use_grid)
        _render_component_page()
        if use_grid:
            _set_status(
                f"Analyzed {len(fast_index.records)} connected component(s) in {layer.name} "
                f"with grid {int(grid_y_spin.value())}x{int(grid_x_spin.value())} px."
            )
        else:
            _set_status(
                f"Analyzed {len(fast_index.records)} connected component(s) in {layer.name}."
            )

    def _delete_selected() -> None:
        layer = _target_layer()
        ids = _selected_component_ids()
        if layer is None or not ids:
            raise ValueError("Select component rows to delete.")
        if state.analyzed_layer_name != layer.name or state.fast_index is None:
            raise ValueError(
                "Component table is stale. Click Analyze before deleting."
            )
        if isinstance(state.fast_index, LargeComponentIndex):
            index = state.fast_index

            def _after_large_delete(changed: int) -> None:
                state.selected_component_ids.difference_update(ids)
                state.large_records = index.active_records()
                state.large_ids = [record.component_id for record in state.large_records]
                _render_component_page()
                _set_status(
                    f"Deleted {len(ids)} component(s), {changed:,} pixels from {layer.name}."
                )

            large_controller.edit_components(
                index=index,
                source_layer=layer,
                target_layer=layer,
                component_ids=ids,
                delete=True,
                on_done=_after_large_delete,
                on_failed=_invalidate_analysis,
            )
            return
        updated, changed = state.fast_index.delete_components(layer.data, ids)
        if changed <= 0:
            raise ValueError("Selected component delete made no label changes.")
        layer.data = updated
        state.selected_component_ids.difference_update(ids)
        component_table.set_records(state.fast_index.active_records())
        _update_selection_summary()
        _set_status(
            f"Deleted {len(ids)} selected component(s) from {layer.name}."
        )

    def _copy_selected_to_layer(target_layer) -> None:
        source_layer = _target_layer()
        ids = _selected_component_ids()
        if source_layer is None or not ids:
            raise ValueError("Select component rows to copy.")
        if state.analyzed_layer_name != source_layer.name or state.fast_index is None:
            raise ValueError(
                "Component table is stale. Click Analyze before copying."
            )
        if target_layer is None or not isinstance(target_layer, napari.layers.Labels):
            raise ValueError("Choose a valid destination labels layer.")
        if isinstance(state.fast_index, LargeComponentIndex):
            index = state.fast_index

            def _after_large_copy(copied_pixels: int) -> None:
                state.last_copy_target_name = str(target_layer.name)
                viewer.layers.selection.active = source_layer
                _set_status(
                    f"Copied {len(ids)} component(s), {copied_pixels:,} pixels "
                    f"from {source_layer.name} to {target_layer.name}."
                )
                _refresh_targets()

            large_controller.edit_components(
                index=index,
                source_layer=source_layer,
                target_layer=target_layer,
                component_ids=ids,
                delete=False,
                on_done=_after_large_copy,
            )
            return
        updated, copied_pixels = _copy_mask_components_to_target(
            source_layer.data,
            target_layer.data,
            state.fast_index.component_id_map,
            ids,
        )
        target_layer.data = updated
        state.last_copy_target_name = str(target_layer.name)
        viewer.layers.selection.active = source_layer
        _set_status(
            f"Copied {len(ids)} component(s) ({copied_pixels} px) from {source_layer.name} "
            f"to {target_layer.name}. Source layer remains active."
        )
        _refresh_targets()

    def _copy_selected_to_last_target() -> None:
        if not state.last_copy_target_name:
            raise ValueError("No previous copy target yet. Choose a target layer first.")
        with suppress(KeyError):
            layer = viewer.layers[state.last_copy_target_name]
            if isinstance(layer, napari.layers.Labels):
                _copy_selected_to_layer(layer)
                return
        raise ValueError("Last copy target is no longer available. Choose a target layer again.")

    def _show_component_context_menu(position) -> None:
        ids = _selected_component_ids()
        if not ids:
            return
        source_layer = _target_layer()
        if source_layer is None:
            return
        menu = QMenu(component_table)
        copy_menu = menu.addMenu("Copy selected to")
        targets = _eligible_copy_targets(source_layer)
        if not targets:
            action = copy_menu.addAction("No same-size labels targets")
            action.setEnabled(False)
        else:
            if state.last_copy_target_name:
                for target_layer in targets:
                    if target_layer.name != state.last_copy_target_name:
                        continue
                    action = copy_menu.addAction(
                        f"Last target: {target_layer.name}"
                    )
                    action.triggered.connect(
                        lambda _checked=False, layer=target_layer: _run_safely(
                            lambda: _copy_selected_to_layer(layer)
                        )
                    )
                    copy_menu.addSeparator()
                    break
            for target_layer in targets:
                action = copy_menu.addAction(target_layer.name)
                action.triggered.connect(
                    lambda _checked=False, layer=target_layer: _run_safely(
                        lambda: _copy_selected_to_layer(layer)
                    )
                )
        menu.exec(component_table.viewport().mapToGlobal(position))

    component_table._context_menu_callback = _show_component_context_menu

    def _mouse_enabled_for_layer(layer) -> bool:
        return (
            click_select_check.isChecked()
            and state.analyzed_layer_name == getattr(layer, "name", None)
            and state.fast_index is not None
        )

    def _handle_mouse_click(_source, event):
        layer = _target_layer()
        if layer is None:
            return
        if not _mouse_enabled_for_layer(layer):
            return
        if str(getattr(event, "type", "")).lower() != "mouse_press":
            return
        position = getattr(event, "position", None)
        if position is None:
            return
        try:
            data_position = layer.world_to_data(position)
            y = int(round(float(data_position[-2])))
            x = int(round(float(data_position[-1])))
        except Exception:
            return
        fast_index = state.fast_index
        if fast_index is None:
            return
        component_id = fast_index.component_id_at((y, x))
        if component_id is None:
            return
        if append_click_select_check.isChecked():
            state.selected_component_ids.add(component_id)
        else:
            state.selected_component_ids = {component_id}
        if isinstance(fast_index, LargeComponentIndex) and component_id in fast_index.records:
            position = bisect_left(state.large_ids, component_id)
            if position < len(state.large_ids) and state.large_ids[position] == component_id:
                target_page = position // LARGE_TABLE_PAGE_SIZE + 1
                if target_page != page_spin.value():
                    page_spin.setValue(target_page)
        _sync_table_selection_from_state()
        record = fast_index.records.get(component_id)
        location = (
            f" ({record.grid_id})"
            if record is not None and record.grid_id
            else ""
        )
        if append_click_select_check.isChecked():
            _set_status(
                f"Added component {component_id}{location} from canvas click. "
                f"{selection_summary.text()}"
            )
        else:
            _set_status(f"Selected component {component_id}{location} from canvas click.")

    def _sync_mouse_callback() -> None:
        layer = _target_layer()
        # Layer callbacks only receive events while that layer is active. A
        # viewer callback keeps picking available under grid and other overlays.
        viewer_callbacks = getattr(viewer, "mouse_drag_callbacks", None)
        if (
            viewer_callbacks is not None
            and _handle_mouse_click in viewer_callbacks
        ):
            viewer_callbacks.remove(_handle_mouse_click)
        for candidate in _eligible_layers():
            callbacks = getattr(candidate, "mouse_drag_callbacks", None)
            if callbacks is not None and _handle_mouse_click in callbacks:
                callbacks.remove(_handle_mouse_click)
        state.callback_registered = False
        if click_select_check.isChecked() and layer is not None:
            if (
                viewer_callbacks is not None
                and _handle_mouse_click not in viewer_callbacks
            ):
                viewer_callbacks.append(_handle_mouse_click)
                state.callback_registered = True

    def _run_safely(fn) -> None:
        try:
            fn()
        except Exception as e:
            _set_status(str(e))

    def _on_target_changed() -> None:
        if large_controller.edit_active:
            target_combo.blockSignals(True)
            target_combo.setCurrentText(state.analyzed_layer_name or "")
            target_combo.blockSignals(False)
            _set_status("Wait for the current tile edit before changing Labels layers.")
            return
        if large_controller.worker is not None:
            large_controller.cancel()
        _hide_grid_overlay()
        _restore_grid_registration()
        _invalidate_analysis("Source layer changed. Click Analyze.")
        _sync_mouse_callback()
        _refresh_copy_targets()
        dynamic_edit_controller.sync()
        if display_grid_check.isChecked():
            try:
                _show_grid_overlay()
                _refresh_grid_labels()
            except Exception as exc:
                _set_status(str(exc))

    def _on_assign_grid_toggled(checked: bool) -> None:
        component_table.set_grid_columns_visible(bool(checked))
        _store_grid_registration()
        _reassign_grid_ids()

    def _on_display_grid_toggled(checked: bool) -> None:
        try:
            if checked:
                _show_grid_overlay()
                _refresh_grid_labels()
                _set_status(
                    f"Displayed {int(grid_y_spin.value())} x "
                    f"{int(grid_x_spin.value())} px grid map. "
                    "Cell IDs adapt to zoom."
                )
            else:
                _hide_grid_overlay()
                _set_status("Grid overlay hidden. Table grid IDs are unchanged.")
        except Exception as exc:
            display_grid_check.blockSignals(True)
            display_grid_check.setChecked(False)
            display_grid_check.blockSignals(False)
            _set_status(str(exc))
        _store_grid_registration()

    def _on_grid_size_changed() -> None:
        _store_grid_registration()
        if display_grid_check.isChecked():
            try:
                _show_grid_overlay()
                _refresh_grid_labels()
            except Exception as exc:
                _set_status(str(exc))
        if assign_grid_check.isChecked() and state.fast_index is not None:
            _reassign_grid_ids()

    def _cleanup_callbacks(*_args) -> None:
        state.closed = True
        with suppress(ValueError, AttributeError):
            viewer._label_assistant_edit_controllers.remove(
                dynamic_edit_controller
            )
        dynamic_edit_controller.close()
        large_controller.close()
        if isinstance(state.fast_index, LargeComponentIndex):
            state.fast_index.close()
            state.fast_index = None
        grid_label_timer.stop()
        with suppress(Exception):
            viewer.camera.events.zoom.disconnect(_schedule_grid_label_update)
        with suppress(Exception):
            viewer.camera.events.center.disconnect(_schedule_grid_label_update)
        with suppress(Exception):
            viewer.window._qt_viewer.canvas.events.resize.disconnect(
                _schedule_grid_label_update
            )
        with suppress(Exception):
            click_select_check.setChecked(False)
        viewer_callbacks = getattr(viewer, "mouse_drag_callbacks", None)
        if (
            viewer_callbacks is not None
            and _handle_mouse_click in viewer_callbacks
        ):
            viewer_callbacks.remove(_handle_mouse_click)
        for candidate in _eligible_layers():
            callbacks = getattr(candidate, "mouse_drag_callbacks", None)
            if callbacks is not None and _handle_mouse_click in callbacks:
                callbacks.remove(_handle_mouse_click)

    def _refresh_and_sync() -> None:
        _refresh_targets()
        dynamic_edit_controller.sync()

    btn_refresh.clicked.connect(_refresh_and_sync)
    btn_analyze.clicked.connect(lambda: _run_safely(_analyze))
    btn_delete.clicked.connect(lambda: _run_safely(_delete_selected))
    btn_copy.clicked.connect(
        lambda: _run_safely(lambda: _copy_selected_to_layer(_copy_target_layer()))
    )
    btn_copy_last.clicked.connect(lambda: _run_safely(_copy_selected_to_last_target))
    btn_copy.setToolTip("Copy all currently selected rows to the chosen target layer.")
    btn_copy_last.setToolTip(
        "Repeat the last successful copy destination without changing layer focus."
    )
    component_table.itemSelectionChanged.connect(_on_table_selection_changed)
    page_spin.valueChanged.connect(lambda _value: _render_component_page())
    target_combo.currentIndexChanged.connect(lambda _idx: _on_target_changed())
    editing_strategy_combo.currentIndexChanged.connect(
        lambda _idx: dynamic_edit_controller.sync(force=True)
    )
    edit_tile_size_combo.currentIndexChanged.connect(
        lambda _idx: dynamic_edit_controller.ensure_tile(force=True)
    )
    show_edit_boundary_check.toggled.connect(
        dynamic_edit_controller.update_boundary_visibility
    )
    assign_grid_check.toggled.connect(_on_assign_grid_toggled)
    display_grid_check.toggled.connect(_on_display_grid_toggled)
    grid_y_spin.valueChanged.connect(lambda _value: _on_grid_size_changed())
    grid_x_spin.valueChanged.connect(lambda _value: _on_grid_size_changed())
    click_select_check.toggled.connect(lambda _checked: _sync_mouse_callback())
    grid_label_timer.timeout.connect(lambda: _run_safely(_refresh_grid_labels))
    viewer.camera.events.zoom.connect(_schedule_grid_label_update)
    viewer.camera.events.center.connect(_schedule_grid_label_update)
    with suppress(Exception):
        viewer.window._qt_viewer.canvas.events.resize.connect(
            _schedule_grid_label_update
        )
    container.destroyed.connect(_cleanup_callbacks)

    _on_assign_grid_toggled(False)
    container._component_table = component_table
    container._workflow_tabs = workflow_tabs
    container._annotate_page = annotate_page
    container._qc_page = qc_page
    container._selection_summary = selection_summary
    container._status_label = status
    container._analyze_button = btn_analyze
    container._target_combo = target_combo
    container._assign_grid_check = assign_grid_check
    container._display_grid_check = display_grid_check
    container._grid_y_spin = grid_y_spin
    container._grid_x_spin = grid_x_spin
    container._refresh_grid_labels = _refresh_grid_labels
    container._large_controller = large_controller
    container._page_spin = page_spin
    container._analysis_progress = analysis_progress
    container._copy_target_combo = copy_target_combo
    container._click_select_check = click_select_check
    container._append_click_select_check = append_click_select_check
    container._copy_button = btn_copy
    container._delete_button = btn_delete
    container._cancel_analysis_button = btn_cancel_analysis
    container._editing_strategy_combo = editing_strategy_combo
    container._edit_tile_size_combo = edit_tile_size_combo
    container._show_edit_boundary_check = show_edit_boundary_check
    container._auto_save_edit_check = auto_save_edit_check
    container._dynamic_edit_status = dynamic_edit_status
    container._dynamic_edit_controller = dynamic_edit_controller
    container._refresh_layers_button = btn_refresh
    container._save_edit_button = save_edit_btn
    container._undo_edit_button = undo_edit_btn
    container._redo_edit_button = redo_edit_btn
    container._layer_refresh_timer = _auto_refresh_layer_controls(
        viewer, container, _refresh_and_sync
    )
    _refresh_targets()
    dynamic_edit_controller.sync()
    return container
