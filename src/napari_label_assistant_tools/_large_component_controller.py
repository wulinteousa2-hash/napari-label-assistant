"""Responsive Qt orchestration for large, disk-backed component analysis."""

from __future__ import annotations

from threading import Event
from typing import Callable

import numpy as np
from napari.qt.threading import create_worker
from qtpy.QtCore import QTimer

from ._large_components import (
    LargeAnalysisCancelled,
    LargeComponentIndex,
    build_large_component_index,
)


def _writable_array_like(data) -> bool:
    if isinstance(data, (list, tuple)):
        return False
    if isinstance(data, np.ndarray):
        return bool(data.flags.writeable)
    return type(data).__module__.split(".")[0] == "zarr" and hasattr(
        data, "__setitem__"
    )


class LargeComponentController:
    def __init__(
        self,
        *,
        progress,
        cancel_button,
        analyze_button,
        status: Callable[[str], None],
        on_ready: Callable[[str, LargeComponentIndex], None],
    ) -> None:
        self.progress = progress
        self.cancel_button = cancel_button
        self.analyze_button = analyze_button
        self.status = status
        self.on_ready = on_ready
        self.worker = None
        self.cancel_event: Event | None = None
        self.edit_active = False
        self.closed = False
        cancel_button.clicked.connect(self.cancel)

    @property
    def busy(self) -> bool:
        return self.worker is not None or self.edit_active

    def start(self, source, layer_name: str, layer=None) -> None:
        if self.busy:
            raise ValueError("A large-mask operation is already running.")
        self.cancel_event = Event()
        self.analyze_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status("Large-mask analysis started. You can keep using napari.")
        worker = create_worker(
            build_large_component_index,
            source,
            cancel_event=self.cancel_event,
            _start_thread=False,
        )
        self.worker = worker
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(
            lambda index: self._on_returned(layer_name, index)
        )
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        if layer is not None and hasattr(layer, "editable"):
            was_editable = bool(layer.editable)
            if was_editable:
                layer.editable = False
                worker.finished.connect(
                    lambda: setattr(layer, "editable", was_editable)
                )
        worker.start()

    def _on_progress(self, update: tuple[int, int, str]) -> None:
        if self.closed:
            return
        completed, total, message = update
        self.progress.setValue(int(100 * completed / max(total, 1)))
        self.progress.setFormat(f"{self.progress.value()}% - {message}")
        self.status(message)

    def _on_returned(self, layer_name: str, index: LargeComponentIndex) -> None:
        if self.closed or (self.cancel_event and self.cancel_event.is_set()):
            index.close()
            return
        self.on_ready(layer_name, index)

    def _on_error(self, error: Exception) -> None:
        if self.closed:
            return
        if isinstance(error, LargeAnalysisCancelled):
            self.status("Large-mask analysis canceled; no mask data was changed.")
        else:
            self.status(f"Large-mask analysis failed: {error}")

    def _on_finished(self) -> None:
        self.worker = None
        self.cancel_event = None
        if self.closed:
            return
        self.analyze_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)

    def cancel(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            if not self.closed:
                self.status("Canceling after the current tile...")

    def edit_components(
        self,
        *,
        index: LargeComponentIndex,
        source_layer,
        target_layer,
        component_ids: list[int],
        delete: bool,
        on_done: Callable[[int], None],
        on_failed: Callable[[], None] | None = None,
    ) -> None:
        if self.busy:
            raise ValueError("Wait for the current large-mask operation to finish.")
        target_data = target_layer.data
        if not _writable_array_like(target_data):
            raise ValueError(
                "Large-mask editing needs a writable NumPy/memmap or Zarr "
                "target. A multiscale display layer is not directly editable."
            )
        if tuple(target_data.shape) != index.shape:
            raise ValueError("Copy target shape does not match the analyzed mask.")
        source_data = source_layer.data
        if not delete and isinstance(source_data, (list, tuple)):
            source_data = source_data[0]
        iterator = iter(index.selected_tiles(component_ids))
        changed = 0
        tile_count = 0
        self.edit_active = True
        self.analyze_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Applying selected components tile by tile...")

        def step() -> None:
            nonlocal changed, tile_count
            if self.closed:
                self.edit_active = False
                return
            try:
                tile, mask = next(iterator)
            except StopIteration:
                self.edit_active = False
                self.analyze_button.setEnabled(True)
                self.progress.setVisible(False)
                if delete:
                    index.deleted_component_ids.update(map(int, component_ids))
                refresh_error = None
                try:
                    target_layer.refresh()
                except Exception as error:
                    refresh_error = error
                on_done(changed)
                if refresh_error is not None:
                    self.status(
                        f"Pixels were written, but napari refresh failed: {refresh_error}"
                    )
                return
            try:
                slices = (slice(tile.y0, tile.y1), slice(tile.x0, tile.x1))
                updated = np.asarray(target_data[slices]).copy()
                if delete:
                    updated[mask] = 0
                else:
                    source_region = np.asarray(source_data[slices])
                    updated[mask] = source_region[mask]
                target_data[slices] = updated
                changed += int(np.count_nonzero(mask))
                tile_count += 1
                self.status(
                    f"Updated {tile_count:,} tile(s), {changed:,} selected pixels."
                )
            except Exception as error:
                self.edit_active = False
                self.analyze_button.setEnabled(True)
                self.progress.setVisible(False)
                if on_failed is not None:
                    on_failed()
                self.status(
                    f"Large-mask edit stopped after {changed:,} pixels; "
                    f"earlier tiles may have changed: {error}. Reanalyze before editing."
                )
                return
            QTimer.singleShot(0, step)

        QTimer.singleShot(0, step)

    def close(self) -> None:
        self.closed = True
        self.cancel()
