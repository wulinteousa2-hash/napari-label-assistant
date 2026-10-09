"""Paged Review/QC library with durable, foreground-only training checkboxes."""
from contextlib import suppress
from threading import Event

import napari
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)
from superqt.utils import create_worker

from ._component_library import (
    annotation_changed, ensure_library, library_for, notify_library,
)
from ._quick_train import array_2d
from ._quick_train_widget import _run_task

PAGE_SIZE = 200


class ComponentLibraryWidget(QGroupBox):
    def __init__(self, viewer, source_layer, parent=None):
        super().__init__("Training component library", parent)
        self.viewer, self.source_layer = viewer, source_layer
        self.layer = None
        self.library = None
        self.worker = None
        self.cancel_event = None
        self._connections = []
        self._locked = []
        self._stamp = None
        self._rendering = False
        self._auto_paused = False
        self.closed = False
        layout = QVBoxLayout(self)
        note = QLabel("Include object foreground. Confirm background under Reviewed cells.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.update_button = QPushButton("Build / update library")
        self.cancel_button = QPushButton("Cancel scan")
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.update_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Train", "ID", "Label pixels", "Bounds"])
        self.table.horizontalHeaderItem(0).setToolTip("Include this object's foreground. Background needs reviewed regions.")
        self.table.horizontalHeaderItem(1).setToolTip("Permanent library ID; original Curated label values are preserved.")
        self.table.horizontalHeaderItem(2).setToolTip("Component pixel count, not bounding-box area.")
        self.table.horizontalHeaderItem(3).setToolTip("Source pixel coordinates. Training uses fixed-size patches and counts shared patches once.")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(140)
        self.table.setMaximumHeight(260)
        layout.addWidget(self.table)
        page_row = QHBoxLayout()
        self.page_spin = QSpinBox()
        self.page_spin.setRange(1, 1)
        page_row.addWidget(QLabel("Page"))
        page_row.addWidget(self.page_spin)
        self.summary = QLabel("No library yet.")
        self.summary.setWordWrap(True)
        page_row.addWidget(self.summary, 1)
        layout.addLayout(page_row)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel("Build once; IDs and training selections are saved with the workspace.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_button.clicked.connect(lambda: self.start_update(verify_all=True))
        self.cancel_button.clicked.connect(self.cancel)
        self.page_spin.valueChanged.connect(self.render)
        self.table.itemChanged.connect(self._checked)
        self.table.cellClicked.connect(self._locate)
        self.poll = QTimer(self)
        self.poll.setInterval(1000)
        self.poll.timeout.connect(self.refresh)
        self.poll.start()
        self.destroyed.connect(lambda *_: self.shutdown())
        widgets = getattr(viewer, "_label_assistant_library_widgets", None)
        if widgets is None:
            widgets = []
            viewer._label_assistant_library_widgets = widgets
        widgets.append(self)
        self.refresh()

    def refresh(self):
        if self.closed or self.worker is not None:
            return
        target = self.source_layer()
        target = target if isinstance(target, napari.layers.Labels) else None
        if target is not self.layer:
            for emitter in self._connections:
                with suppress(Exception):
                    emitter.disconnect(self._source_changed)
            self._connections.clear()
            self.layer = target
            self._stamp = None
            self.page_spin.setValue(1)
            if target is not None:
                for name in ("paint", "data", "labels_update"):
                    emitter = getattr(target.events, name, None)
                    if emitter is not None:
                        emitter.connect(self._source_changed)
                        self._connections.append(emitter)
        self.update_button.setEnabled(target is not None)
        try:
            self.library = library_for(target) if target is not None else None
            state = self.library.state() if self.library else None
            stamp = (id(target), str(self.library.path), state["revision"], state["edit_revision"]) if state else (id(target),)
            if stamp != self._stamp:
                self._stamp = stamp
                self.render()
            if state and state["pending"]:
                if self._auto_paused:
                    self.status.setText("Curated changed. Click Build / update library when ready.")
                    return
                self.status.setText("Curated changed. Updating affected components; changed objects need selection again.")
                # Wait for Editable Area's pending writes before taking a snapshot.
                controllers = getattr(self.viewer, "_label_assistant_edit_controllers", ())
                if not getattr(target, "_label_assistant_component_editing", False) and not any(c.source is target and (c.dirty or c.loading or c.stroke_active) for c in controllers):
                    self.start_update(verify_all=False)
        except Exception as error:
            self.library = None
            self.table.setRowCount(0)
            self.status.setText(str(error))

    def _source_changed(self, event):
        if self.layer is None or getattr(event, "library_selection", False):
            return
        if getattr(self.layer, "_label_assistant_library_rebinding", False) or getattr(self.layer, "_label_assistant_library_precise_edit", False):
            return
        try:
            if self.cancel_event is not None:
                # Release the worker's SQLite transaction before recording a
                # concurrent producer's edit on the GUI thread.
                self.cancel_event.set()
            bounds = None
            offset = getattr(event, "offset", None)
            data = getattr(event, "data", None)
            if offset is not None and data is not None:
                from numbers import Integral
                shape = getattr(data, "shape", ())
                if len(offset)>=2 and len(shape)==2 and all(isinstance(v,Integral) for v in offset[-2:]):
                    y,x = map(int,offset[-2:])
                    bounds = (y,x,y+shape[0],x+shape[1])
            annotation_changed(self.viewer, self.layer, bounds)
            self._stamp = None
        except Exception as error:
            self.status.setText(str(error))

    def render(self, *_):
        self._rendering = True
        try:
            if self.library is None:
                self.table.setRowCount(0)
                self.summary.setText("No library yet.")
                return
            total, selected, needs_review = self.library.counts()
            self.page_spin.blockSignals(True)
            self.page_spin.setRange(1, max(1, (total+PAGE_SIZE-1)//PAGE_SIZE))
            self.page_spin.blockSignals(False)
            records = self.library.records(offset=(self.page_spin.value()-1)*PAGE_SIZE, limit=PAGE_SIZE)
            pending = self.library.state()["pending"]
            self.table.setRowCount(len(records))
            for row, record in enumerate(records):
                check = QTableWidgetItem()
                check.setData(Qt.UserRole, record["id"])
                check.setFlags(Qt.ItemIsSelectable | Qt.ItemIsUserCheckable |
                               (Qt.ItemIsEnabled if not pending else Qt.NoItemFlags))
                check.setCheckState(Qt.Checked if record["included"] else Qt.Unchecked)
                if record["needs_review"]:
                    check.setToolTip("Object changed. Review its new boundaries before including it again.")
                self.table.setItem(row, 0, check)
                values = [str(record["id"]), f'{record["pixels"]:,}',
                          f'y[{record["y0"]}:{record["y1"]}] x[{record["x0"]}:{record["x1"]}]']
                for col, value in enumerate(values, 1):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.UserRole, record["id"])
                    if col == 1:
                        item.setToolTip(f'Curated label value: {record["label_value"]}')
                    self.table.setItem(row, col, item)
            self.summary.setText(f"{total:,} objects · {selected:,} included · {needs_review:,} changed")
        finally:
            self._rendering = False

    def _checked(self, item):
        if self._rendering or item.column() != 0 or self.library is None:
            return
        try:
            self.library = library_for(self.layer)
            self.library.select(item.data(Qt.UserRole), item.checkState() == Qt.Checked)
            notify_library(self.layer)
            self.status.setText("Training selection saved.")
        except Exception as error:
            self.status.setText(str(error))
        self.render()

    def _locate(self, row, column):
        if column == 0 or self.library is None:
            return
        records = self.library.records(offset=(self.page_spin.value()-1)*PAGE_SIZE, limit=PAGE_SIZE)
        if row >= len(records):
            return
        record = records[row]
        position = self.layer.data_to_world((record["cy"], record["cx"]))
        center = list(self.viewer.camera.center)
        center[-2:] = position[-2:]
        self.viewer.camera.center = tuple(center)

    def start_update(self, *, verify_all):
        if self.worker is not None or self.closed or self.layer is None:
            return
        try:
            if verify_all:
                self._auto_paused = False
            from .workspace.service import _commit_pending_local_edits
            _commit_pending_local_edits(self.viewer)
            self.library = ensure_library(self.layer)
            source = array_2d(self.layer.data)
            self.cancel_event = Event()
            self.layer._label_assistant_library_indexing = True
            self._locked = [(self.layer, self.layer.editable)]
            for c in getattr(self.viewer, "_label_assistant_edit_controllers", ()):
                if c.source is self.layer and c.tile is not None:
                    self._locked.append((c.tile, c.tile.editable))
            for layer, _ in self._locked:
                layer.editable = False
            self.worker = create_worker(_run_task, self.library.update, (source,),
                                        {"cancel": self.cancel_event, "verify_all": verify_all},
                                        _start_thread=False,
                                        _connect={"errored": self._error})
            self.worker.yielded.connect(self._progress)
            self.worker.returned.connect(self._ready)
            self.worker.finished.connect(self._finished)
            self.update_button.setEnabled(False)
            self.table.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress.show()
            self.worker.start()
        except Exception as error:
            self._error(error)
            self._finished()

    def _progress(self, progress):
        done, total, message = progress
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(message)

    def _ready(self, state):
        if not self.closed:
            notify_library(self.layer)
            self.status.setText("Library saved. Check objects to include; review background in Local Editing.")

    def _error(self, error):
        self._auto_paused = True
        if not self.closed:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.status.setText("Scan cancelled. Saved IDs and selections are retained.")
            else:
                self.status.setText(f"Library scan stopped: {error}")

    def _finished(self):
        if self.layer is not None:
            self.layer._label_assistant_library_indexing = False
        for layer, editable in self._locked:
            if any(layer is item for item in self.viewer.layers):
                layer.editable = editable
        self._locked.clear()
        self.worker = None
        self.cancel_event = None
        if self.closed:
            with suppress(ValueError):
                self.viewer._label_assistant_library_widgets.remove(self)
        if not self.closed:
            self.table.setEnabled(True)
            self.update_button.setEnabled(self.layer is not None)
            self.cancel_button.setEnabled(False)
            self.progress.hide()
            self._stamp = None
            self.render()

    def cancel(self):
        self._auto_paused = True
        if self.cancel_event is not None:
            self.cancel_event.set()

    def shutdown(self):
        self.closed = True
        self.cancel()
        with suppress(RuntimeError):
            self.poll.stop()
        for emitter in self._connections:
            with suppress(Exception):
                emitter.disconnect(self._source_changed)
        self._connections.clear()
        if self.worker is None:
            with suppress(ValueError):
                self.viewer._label_assistant_library_widgets.remove(self)
