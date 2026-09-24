"""Viewport-driven, bounded-memory editing for large 2D Labels layers."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from napari.qt.threading import create_worker
from qtpy.QtCore import QTimer


INTERNAL_LAYER_ROLE_KEY = "label_assistant_internal_role"
COMPATIBLE_INTERNAL_ROLE_KEYS = (
    INTERNAL_LAYER_ROLE_KEY,
    "sam3_internal_role",
)
EDIT_TILE_ROLE = "editable_tile"
EDIT_BOUNDARY_ROLE = "editable_boundary"
AUTO_PIXEL_LIMIT = 100_000_000
AUTO_MEMORY_LIMIT = 2 * 1024**3
AUTO_AVAILABLE_MEMORY_FRACTION = 0.10
DEFAULT_TILE_SIZE = 4096


@dataclass(frozen=True)
class TileBounds:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def contains(
        self, y: float, x: float, *, margin_fraction: float = 0.0
    ) -> bool:
        height, width = self.shape
        margin_y = max(0.0, height * float(margin_fraction))
        margin_x = max(0.0, width * float(margin_fraction))
        return (
            self.y0 + margin_y <= float(y) < self.y1 - margin_y
            and self.x0 + margin_x <= float(x) < self.x1 - margin_x
        )


def is_internal_layer(layer: Any) -> bool:
    metadata = getattr(layer, "metadata", {}) or {}
    return bool(
        isinstance(metadata, dict)
        and any(metadata.get(key) for key in COMPATIBLE_INTERNAL_ROLE_KEYS)
    )


def available_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def estimated_direct_edit_bytes(data: Any) -> int:
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    pixels = int(np.prod(shape, dtype=np.int64)) if shape else 0
    itemsize = int(np.dtype(getattr(data, "dtype", np.uint32)).itemsize)
    return int(pixels * itemsize * 1.5)


def automatic_uses_tiled_editing(
    data: Any, *, available_bytes: int | None = None
) -> bool:
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    if len(shape) != 2:
        return False
    pixels = int(np.prod(shape, dtype=np.int64))
    available = (
        available_memory_bytes() if available_bytes is None else available_bytes
    )
    memory_limit = AUTO_MEMORY_LIMIT
    if available is not None and available > 0:
        memory_limit = min(
            memory_limit, int(available * AUTO_AVAILABLE_MEMORY_FRACTION)
        )
    return (
        pixels > AUTO_PIXEL_LIMIT
        or estimated_direct_edit_bytes(data) > memory_limit
    )


def resolved_editing_strategy(
    requested: str, data: Any, *, available_bytes: int | None = None
) -> str:
    normalized = str(requested).strip().lower()
    if normalized in {"tiled", "direct"}:
        return normalized
    return (
        "tiled"
        if automatic_uses_tiled_editing(
            data, available_bytes=available_bytes
        )
        else "direct"
    )


def aligned_tile_bounds(
    shape: tuple[int, int],
    center: tuple[float, float],
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    alignment: int = 1,
) -> TileBounds:
    height, width = int(shape[0]), int(shape[1])
    size_y = min(height, max(1, int(tile_size)))
    size_x = min(width, max(1, int(tile_size)))
    align = max(1, int(alignment))
    y0 = int(np.floor((float(center[0]) - size_y / 2) / align) * align)
    x0 = int(np.floor((float(center[1]) - size_x / 2) / align) * align)
    y0 = max(0, min(height - size_y, y0))
    x0 = max(0, min(width - size_x, x0))
    return TileBounds(y0, x0, y0 + size_y, x0 + size_x)


def changed_bounding_box(
    current: np.ndarray, original: np.ndarray
) -> tuple[slice, slice] | None:
    if current.shape != original.shape:
        raise ValueError("Editable tile shape changed unexpectedly.")
    changed = np.asarray(current) != np.asarray(original)
    if not np.any(changed):
        return None
    yy, xx = np.nonzero(changed)
    return (
        slice(int(yy.min()), int(yy.max()) + 1),
        slice(int(xx.min()), int(xx.max()) + 1),
    )


def _layer_is_present(viewer, layer) -> bool:
    if layer is None:
        return False
    with suppress(ValueError):
        viewer.layers.index(layer)
        return True
    return False


class DynamicLabelsEditController:
    """Keep one camera-centered Labels tile editable and write changes back."""

    def __init__(
        self,
        *,
        viewer,
        parent,
        source_layer: Callable[[], Any],
        strategy,
        tile_size,
        show_boundary,
        auto_save,
        status: Callable[[str, str], None],
        action_buttons=(),
    ) -> None:
        self.viewer = viewer
        self.source_layer = source_layer
        self.strategy = strategy
        self.tile_size = tile_size
        self.show_boundary = show_boundary
        self.auto_save = auto_save
        self.status = status
        self.action_buttons = tuple(action_buttons)
        self.source = None
        self.tile = None
        self.boundary = None
        self.bounds: TileBounds | None = None
        self.original: np.ndarray | None = None
        self.dirty = False
        self.dirty_bounds: tuple[int, int, int, int] | None = None
        self.dirty_bounds_unknown = False
        self.stroke_active = False
        self.loading = False
        self.closed = False
        self.request_id = 0
        self.block_data_event = False
        self.worker = None
        self._set_actions_enabled(False)

        self.view_timer = QTimer(parent)
        self.view_timer.setSingleShot(True)
        self.view_timer.setInterval(250)
        self.view_timer.timeout.connect(self.ensure_tile)
        self.commit_timer = QTimer(parent)
        self.commit_timer.setSingleShot(True)
        self.commit_timer.setInterval(400)
        self.commit_timer.timeout.connect(self.commit)

        self.camera = getattr(
            getattr(viewer, "scene", None), "camera", None
        )
        if self.camera is None:
            self.camera = getattr(viewer, "camera", None)
        events = getattr(self.camera, "events", None)
        if events is not None:
            for name in ("center", "zoom"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    emitter.connect(self.schedule)

    def requested_strategy(self) -> str:
        with suppress(RuntimeError):
            return str(self.strategy.currentData() or "auto")
        return "auto"

    def resolved_strategy(self, layer) -> str:
        return resolved_editing_strategy(
            self.requested_strategy(), layer.data
        )

    def selected_tile_size(self) -> int:
        with suppress(RuntimeError, TypeError, ValueError):
            value = int(self.tile_size.currentData() or 0)
            if value > 0:
                return value
        return DEFAULT_TILE_SIZE

    def sync(self, *, force: bool = False) -> None:
        if self.closed:
            return
        layer = self.source_layer()
        shape = tuple(int(v) for v in getattr(getattr(layer, "data", None), "shape", ()))
        if layer is None:
            self.teardown(commit=True)
            self.status("Select a source Labels layer.", "neutral")
            return
        if len(shape) != 2:
            self.teardown(commit=True)
            self.status(
                "Interactive tiled editing currently supports 2D Labels layers.",
                "neutral",
            )
            return
        if (
            self.dirty
            and not self._automatic_save_enabled()
            and (
                layer is not self.source
                or self.resolved_strategy(layer) == "direct"
            )
        ):
            self.status(
                "Unsaved manual edits are still in the editable area. "
                "Click Save now before changing the source or editing strategy.",
                "busy",
            )
            return
        self._remove_stale_helpers(layer)
        if self.resolved_strategy(layer) == "direct":
            self.teardown(commit=True)
            with suppress(Exception):
                layer.editable = True
            estimate = self._format_bytes(estimated_direct_edit_bytes(layer.data))
            prefix = (
                "Automatic selected direct editing."
                if self.requested_strategy() == "auto"
                else "Full-layer editing is active."
            )
            self.status(
                f"{prefix} Estimated editing memory: {estimate}.", "ready"
            )
            return
        if self.source is not None and self.source is not layer:
            self.teardown(commit=True)
        self.source = layer
        with suppress(Exception):
            layer.editable = False
            layer.mode = "pan_zoom"
        self.ensure_tile(force=force)

    def _remove_stale_helpers(self, source) -> None:
        """Remove transient helper layers restored by older workspace files."""
        expected_names = (
            f"{source.name} — editable area",
            f"{source.name} — editable boundary",
        )
        for layer in list(self.viewer.layers):
            if layer is self.tile or layer is self.boundary:
                continue
            name = str(getattr(layer, "name", ""))
            legacy_name = any(
                name == expected
                or (
                    name.startswith(f"{expected} [")
                    and name.endswith("]")
                    and name[len(expected) + 2 : -1].isdigit()
                )
                for expected in expected_names
            )
            metadata = getattr(layer, "metadata", {}) or {}
            same_source_helper = (
                is_internal_layer(layer)
                and isinstance(metadata, dict)
                and metadata.get("source_layer_name") == source.name
            )
            if legacy_name or same_source_helper:
                with suppress(Exception):
                    self.viewer.layers.remove(layer)

    def schedule(self, *_args) -> None:
        if self.closed:
            return
        layer = self.source_layer()
        if layer is not None and self.resolved_strategy(layer) == "tiled":
            self.view_timer.start()

    def ensure_tile(self, force: bool = False) -> None:
        if self.closed:
            return
        layer = self.source_layer()
        shape = tuple(int(v) for v in getattr(getattr(layer, "data", None), "shape", ()))
        if (
            layer is None
            or len(shape) != 2
            or self.resolved_strategy(layer) != "tiled"
        ):
            return
        center = self._center_in_data(layer)
        desired = aligned_tile_bounds(
            shape,
            center,
            tile_size=self.selected_tile_size(),
            alignment=self._chunk_alignment(layer),
        )
        if (
            not force
            and self.source is layer
            and self.bounds is not None
            and self.bounds.contains(*center, margin_fraction=0.20)
        ):
            return
        if self.loading:
            if not force:
                return
            self.request_id += 1
        if self.dirty and not self._automatic_save_enabled():
            self.status(
                "Unsaved manual edits are still in this area. Click Save now "
                "before moving the editable area.",
                "busy",
            )
            return
        self.commit()
        self.source = layer
        self.loading = True
        self.request_id += 1
        request_id = self.request_id
        self.status(
            f"Loading editable area {desired.shape[1]} × {desired.shape[0]} px…",
            "busy",
        )
        worker = create_worker(
            self._read_tile,
            layer.data,
            desired,
            _start_thread=False,
        )
        self.worker = worker
        worker.returned.connect(
            lambda data: self._finish_load(
                request_id, layer, desired, data
            )
        )
        worker.errored.connect(
            lambda error: self._load_failed(request_id, error)
        )
        worker.start()

    @staticmethod
    def _read_tile(data, bounds: TileBounds) -> np.ndarray:
        return np.asarray(data[bounds.slices]).copy()

    def _finish_load(
        self,
        request_id: int,
        source,
        bounds: TileBounds,
        data: np.ndarray,
    ) -> None:
        if (
            self.closed
            or request_id != self.request_id
            or source is not self.source_layer()
        ):
            return
        self.loading = False
        self.worker = None
        previous_mode = str(getattr(self.tile, "mode", "pan_zoom"))
        origin = np.asarray(
            source.data_to_world((bounds.y0, bounds.x0)), dtype=float
        ).ravel()[-2:]
        scale = np.asarray(source.scale, dtype=float)[-2:]
        if not _layer_is_present(self.viewer, self.tile):
            self.tile = self.viewer.add_labels(
                data,
                name=f"{source.name} — editable area",
                scale=scale,
                translate=origin,
                metadata={
                    INTERNAL_LAYER_ROLE_KEY: EDIT_TILE_ROLE,
                    "source_layer_name": source.name,
                },
            )
            self.tile.events.data.connect(self._tile_changed)
            self.tile.events.labels_update.connect(self._tile_changed)
            if self._track_paint_stroke not in self.tile.mouse_drag_callbacks:
                self.tile.mouse_drag_callbacks.append(
                    self._track_paint_stroke
                )
        else:
            self.block_data_event = True
            try:
                self.tile.data = data
                self.tile.scale = scale
                self.tile.translate = origin
            finally:
                self.block_data_event = False
        with suppress(Exception):
            self.tile.opacity = source.opacity
            self.tile.selected_label = source.selected_label
            self.tile.colormap = source.colormap
            self.tile.visible = True
            if previous_mode in {"paint", "erase", "fill", "polygon"}:
                self.tile.mode = previous_mode
        self.source = source
        self.bounds = bounds
        self.original = np.asarray(data).copy()
        self.dirty = False
        self.dirty_bounds = None
        self.dirty_bounds_unknown = False
        self._update_boundary()
        self._set_actions_enabled(True)
        with suppress(Exception):
            self.viewer.layers.selection.active = self.tile
        self.status(
            f"Ready: x {bounds.x0}:{bounds.x1}, y {bounds.y0}:{bounds.y1}. "
            + (
                "Paint or erase inside the outlined area; changes save automatically."
                if self._automatic_save_enabled()
                else "Paint or erase inside the outlined area, then click Save now."
            ),
            "ready",
        )
        self.view_timer.start()

    def _load_failed(self, request_id: int, error: Exception) -> None:
        if request_id != self.request_id:
            return
        self.loading = False
        self.worker = None
        self.status(f"Could not load the editable area: {error}", "error")

    def _tile_changed(self, event=None) -> None:
        if self.closed or self.block_data_event or self.original is None:
            return
        self.dirty = True
        offset = getattr(event, "offset", None)
        event_data = getattr(event, "data", None)
        if offset is None or event_data is None:
            self.dirty_bounds = None
            self.dirty_bounds_unknown = True
        elif not self.dirty_bounds_unknown:
            self._accumulate_dirty_bounds(offset, np.shape(event_data))
        if self._automatic_save_enabled():
            self.status(
                "Editing… changes will be saved automatically.", "busy"
            )
        else:
            self.status(
                "Unsaved edits are in the editable area. Click Save now.",
                "busy",
            )
        if not self.stroke_active and self._automatic_save_enabled():
            self.commit_timer.start()

    def _accumulate_dirty_bounds(
        self, offset, shape: tuple[int, ...]
    ) -> None:
        if len(offset) < 2 or len(shape) < 2:
            self.dirty_bounds = None
            return
        y0, x0 = int(offset[-2]), int(offset[-1])
        update = (y0, x0, y0 + int(shape[-2]), x0 + int(shape[-1]))
        if self.dirty_bounds is None:
            self.dirty_bounds = update
            return
        old = self.dirty_bounds
        self.dirty_bounds = (
            min(old[0], update[0]),
            min(old[1], update[1]),
            max(old[2], update[2]),
            max(old[3], update[3]),
        )

    def _track_paint_stroke(self, layer, event):
        mode = str(getattr(layer, "mode", "")).lower()
        if mode not in {"paint", "erase", "fill", "polygon"}:
            return
        self.stroke_active = True
        self.commit_timer.stop()
        try:
            yield
            while getattr(event, "type", "") == "mouse_move":
                yield
        finally:
            self.stroke_active = False
            if (
                self.dirty
                and not self.closed
                and self._automatic_save_enabled()
            ):
                self.commit_timer.start()

    def _automatic_save_enabled(self) -> bool:
        with suppress(RuntimeError):
            return bool(self.auto_save.isChecked())
        return True

    def on_auto_save_toggled(self, checked: bool) -> None:
        if not checked:
            self.commit_timer.stop()
            if self.dirty:
                self.status(
                    "Automatic saving is off. Click Save now to write pending "
                    "edits to the source.",
                    "busy",
                )
            return
        if self.dirty and not self.stroke_active:
            self.status(
                "Automatic saving is on. Pending edits will be saved shortly.",
                "busy",
            )
            self.commit_timer.start()

    def save_now(self, *_args) -> None:
        if self.tile is None:
            self.status("Local-area editing is not active.", "neutral")
            return
        if self.stroke_active:
            self.status(
                "Finish the current stroke before saving.", "busy"
            )
            return
        if not self.dirty:
            self.status(
                f"No pending edits. Source layer “{self.source.name}” is current.",
                "ready",
            )
            return
        if self.commit():
            # The camera or selected source may have changed while a manual
            # tile was intentionally held in place.
            self.sync()

    def undo_stroke(self, *_args) -> None:
        self._replay_stroke(undo=True)

    def redo_stroke(self, *_args) -> None:
        self._replay_stroke(undo=False)

    def _replay_stroke(self, *, undo: bool) -> None:
        if self.tile is None or self.source is None:
            self.status("Local-area editing is not active.", "neutral")
            return
        if self.stroke_active:
            self.status(
                "Finish the current stroke before using Undo or Redo.", "busy"
            )
            return
        history_name = "_undo_history" if undo else "_redo_history"
        history = getattr(self.tile, history_name, None)
        if not history:
            self.status(
                "No paint stroke is available to "
                + ("undo." if undo else "redo."),
                "neutral",
            )
            return
        bounds = self._history_item_bounds(history[-1])
        self.commit_timer.stop()
        if undo:
            self.tile.undo()
        else:
            self.tile.redo()
        self.dirty = True
        self.dirty_bounds = bounds
        self.dirty_bounds_unknown = bounds is None
        if not self._automatic_save_enabled():
            self.status(
                ("Undo" if undo else "Redo")
                + " applied in the editable area. Click Save now to update "
                f"“{self.source.name}”.",
                "busy",
            )
            return
        if self.commit():
            action = "Undid" if undo else "Restored"
            self.status(
                f"{action} the last stroke and saved the result to "
                f"“{self.source.name}”.",
                "ready",
            )

    @staticmethod
    def _history_item_bounds(history_item) -> tuple[int, int, int, int] | None:
        bounds = None
        try:
            atoms = list(history_item)
        except TypeError:
            return None
        for atom in atoms:
            key = getattr(atom, "slice_key", None)
            if key is not None and len(key) >= 2:
                ys, xs = key[-2], key[-1]
                if not isinstance(ys, slice) or not isinstance(xs, slice):
                    return None
                update = (
                    int(ys.start or 0),
                    int(xs.start or 0),
                    int(ys.stop),
                    int(xs.stop),
                )
            else:
                try:
                    indices = atom[0]
                    yy = np.asarray(indices[-2])
                    xx = np.asarray(indices[-1])
                    update = (
                        int(yy.min()),
                        int(xx.min()),
                        int(yy.max()) + 1,
                        int(xx.max()) + 1,
                    )
                except Exception:
                    return None
            if bounds is None:
                bounds = update
            else:
                bounds = (
                    min(bounds[0], update[0]),
                    min(bounds[1], update[1]),
                    max(bounds[2], update[2]),
                    max(bounds[3], update[3]),
                )
        return bounds

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in self.action_buttons:
            with suppress(RuntimeError):
                button.setEnabled(bool(enabled))

    def commit(self, *_args, update_status: bool = True) -> bool:
        self.commit_timer.stop()
        if (
            not self.dirty
            or self.source is None
            or self.tile is None
            or self.bounds is None
            or self.original is None
        ):
            return True
        current = np.asarray(self.tile.data)
        try:
            candidate = (
                (
                    slice(self.dirty_bounds[0], self.dirty_bounds[2]),
                    slice(self.dirty_bounds[1], self.dirty_bounds[3]),
                )
                if (
                    self.dirty_bounds is not None
                    and not self.dirty_bounds_unknown
                )
                else (slice(None), slice(None))
            )
            bounded = changed_bounding_box(
                current[candidate], self.original[candidate]
            )
            if bounded is None:
                local = None
            else:
                ys, xs = bounded
                base_y = int(candidate[0].start or 0)
                base_x = int(candidate[1].start or 0)
                local = (
                    slice(base_y + ys.start, base_y + ys.stop),
                    slice(base_x + xs.start, base_x + xs.stop),
                )
            if local is None:
                self.dirty = False
                self.dirty_bounds = None
                self.dirty_bounds_unknown = False
                return True
            ys, xs = local
            destination = (
                slice(self.bounds.y0 + ys.start, self.bounds.y0 + ys.stop),
                slice(self.bounds.x0 + xs.start, self.bounds.x0 + xs.stop),
            )
            changed_region = np.asarray(current[local]).copy()
            previous_region = np.asarray(self.original[local]).copy()
            changed_mask = changed_region != previous_region
            self.source.data[destination] = changed_region
            preview_updated = self._refresh_source_region(
                changed_region,
                destination,
                changed_mask,
            )
            self.original[local] = changed_region
            self.dirty = False
            self.dirty_bounds = None
            self.dirty_bounds_unknown = False
            if update_status:
                if preview_updated:
                    self.status(
                        f"Saved to source layer “{self.source.name}” and "
                        "updated its local preview.",
                        "ready",
                    )
                else:
                    self.status(
                        f"Saved to source layer “{self.source.name}”. Its "
                        "preview will update on the next view reload.",
                        "ready",
                    )
            return True
        except Exception as error:
            self.status(f"Could not save label edits: {error}", "error")
            return False

    def _refresh_source_region(
        self,
        region_data: np.ndarray,
        destination: tuple[slice, slice],
        changed_mask: np.ndarray,
    ) -> bool:
        """Patch napari's visible Labels cache without refreshing the layer."""
        refresh_caches = getattr(
            self.source, "_refresh_caches_from_region", None
        )
        accumulate = getattr(self.source, "_accumulate_updated_slice", None)
        partial_refresh = getattr(
            self.source, "_partial_labels_refresh", None
        )
        if not all(
            callable(method)
            for method in (refresh_caches, accumulate, partial_refresh)
        ):
            return False
        try:
            # napari's cache helper accepts one new label value at a time.
            # A debounced edit may contain paint and erase changes together,
            # so update each changed value while retaining one bounding box.
            values = np.unique(region_data[changed_mask])
            for value in values:
                value_mask = changed_mask & (region_data == value)
                refresh_caches(
                    region_data,
                    destination,
                    value_mask,
                    int(value),
                )
            accumulate(destination)
            partial_refresh()
            return True
        except Exception:
            # The data write already succeeded. Private napari display APIs
            # can vary by release, so never fall back to a potentially costly
            # full-layer refresh for a huge source.
            return False

    def _update_boundary(self) -> None:
        if self.source is None or self.bounds is None:
            return
        b = self.bounds
        rectangle = np.asarray(
            [[b.y0, b.x0], [b.y0, b.x1], [b.y1, b.x1], [b.y1, b.x0]],
            dtype=float,
        )
        if not _layer_is_present(self.viewer, self.boundary):
            self.boundary = self.viewer.add_shapes(
                [rectangle],
                shape_type="rectangle",
                name=f"{self.source.name} — editable boundary",
                edge_color="#35e86f",
                face_color="transparent",
                edge_width=3,
                scale=np.asarray(self.source.scale)[-2:],
                translate=np.asarray(self.source.translate)[-2:],
                metadata={
                    INTERNAL_LAYER_ROLE_KEY: EDIT_BOUNDARY_ROLE,
                    "source_layer_name": self.source.name,
                },
            )
            self.boundary.editable = False
        else:
            self.boundary.data = [rectangle]
        self.update_boundary_visibility()

    def update_boundary_visibility(self, *_args) -> None:
        if self.boundary is not None:
            with suppress(RuntimeError):
                self.boundary.visible = self.show_boundary.isChecked()

    def teardown(self, *, commit: bool) -> None:
        self.view_timer.stop()
        self.commit_timer.stop()
        self.request_id += 1
        self.loading = False
        if commit:
            self.commit(update_status=False)
        if self.tile is not None:
            with suppress(Exception):
                self.tile.events.data.disconnect(self._tile_changed)
            with suppress(Exception):
                self.tile.events.labels_update.disconnect(self._tile_changed)
            with suppress(ValueError):
                self.tile.mouse_drag_callbacks.remove(
                    self._track_paint_stroke
                )
        for layer in (self.boundary, self.tile):
            if _layer_is_present(self.viewer, layer):
                with suppress(Exception):
                    self.viewer.layers.remove(layer)
        if _layer_is_present(self.viewer, self.source):
            with suppress(Exception):
                self.source.editable = True
        self.source = None
        self.tile = None
        self.boundary = None
        self.bounds = None
        self.original = None
        self.dirty = False
        self.dirty_bounds = None
        self.dirty_bounds_unknown = False
        self.stroke_active = False
        self.worker = None
        self._set_actions_enabled(False)

    def close(self) -> None:
        if self.closed:
            return
        self.teardown(commit=True)
        events = getattr(self.camera, "events", None)
        if events is not None:
            for name in ("center", "zoom"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    with suppress(Exception):
                        emitter.disconnect(self.schedule)
        self.closed = True

    def _center_in_data(self, layer) -> tuple[float, float]:
        center = getattr(self.camera, "center", None)
        if center is None:
            shape = layer.data.shape
            return float(shape[-2] / 2), float(shape[-1] / 2)
        world = tuple(float(value) for value in np.asarray(center).ravel())
        with suppress(Exception):
            data = np.asarray(layer.world_to_data(world), dtype=float).ravel()
            if data.size >= 2:
                return float(data[-2]), float(data[-1])
        return float(world[-2]), float(world[-1])

    @staticmethod
    def _chunk_alignment(layer) -> int:
        chunks = getattr(layer.data, "chunks", None)
        if not chunks or len(chunks) < 2:
            return 1
        with suppress(TypeError, ValueError):
            return max(1, min(int(chunks[-2]), int(chunks[-1])))
        return 1

    @staticmethod
    def _format_bytes(byte_count: int) -> str:
        value = float(max(0, int(byte_count)))
        for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or suffix == "TiB":
                return f"{value:.1f} {suffix}"
            value /= 1024.0
        return f"{value:.1f} TiB"
