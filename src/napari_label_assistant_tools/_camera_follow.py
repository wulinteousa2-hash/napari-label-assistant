"""Camera-driven bounded edit windows. Background workers only read arrays."""

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import QApplication

STEP = 512
MAX_WINDOW = 4096
MAX_READ_BYTES = 128 * 1024**2
ENTER_PIXEL_SIZE = 0.5
EXIT_PIXEL_SIZE = 0.35


@dataclass(frozen=True)
class FollowWindow:
    origin: tuple[int, int]
    size: int


def plan_window(bounds, shape, pixel_size, *, current=None, bytes_per_pixel=4):
    """Plan source-coordinate windows without reading image or mask pixels."""
    threshold = EXIT_PIXEL_SIZE if current is not None else ENTER_PIXEL_SIZE
    if pixel_size < threshold:
        return None
    y0, x0, y1, x1 = bounds
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(shape[0], y1), min(shape[1], x1)
    if y1 <= y0 or x1 <= x0:
        return None
    span = max(y1 - y0, x1 - x0)
    padding = max(32, min(128, span * 0.1))
    required = span + STEP + 2 * padding
    size = 1024
    while size < required:
        size *= 2
    if size > MAX_WINDOW or size * size * bytes_per_pixel > MAX_READ_BYTES:
        return None
    if current is not None:
        a, b = current.origin
        margin = min(128, current.size * 0.1)
        # Retain a warm buffer while the viewport stays inside its guard band.
        if (
            a + margin <= y0
            and b + margin <= x0
            and y1 <= a + current.size - margin
            and x1 <= b + current.size - margin
            and current.size <= 2 * size
        ):
            return current
    starts = []
    for lo, hi in ((y0, y1), (x0, x1)):
        minimum = max(0, ceil((hi - size) / STEP) * STEP)
        maximum = floor(lo / STEP) * STEP
        centered = floor(((lo + hi - size) / 2) / STEP) * STEP
        starts.append(max(minimum, min(maximum, centered)))
    return FollowWindow(tuple(starts), size)


def read_regions(specification):
    """Worker entry point: no Qt objects or layer mutation."""
    return tuple(
        (
            bounds,
            np.asarray(array[bounds]).copy()
            if bounds is not None
            else np.zeros((1, 1), dtype=array.dtype),
        )
        for array, bounds in specification
    )


class CameraFollowController(QObject):
    DEBOUNCE_MS = 180

    def __init__(self, project):
        super().__init__(project)
        self.project = project
        self.host = project.host
        self.enabled = False
        self.closed = False
        self.updating = False
        self.window = None
        self.context = None
        self.generation = 0
        self.edit_revision = 0
        self.deadline = 0
        self._manual_depth = 0
        self._manual_view = None
        self._future = None
        self._request = None
        self._executor = None
        self._modes = {}
        self._edit_connections = {}
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.evaluate)
        self.poll = QTimer(self)
        self.poll.setInterval(30)
        self.poll.timeout.connect(self._poll)
        self.host.adapter.camera.events.connect(self.schedule)
        self.host.adapter.native.installEventFilter(self)
        self.state = "manual"
        self._status("Manual — use Edit at View Center.")

    @property
    def edit_ready(self):
        return (self.window is not None and self.state == "following") or (
            self._manual_view is not None and self.state == "manual"
        )

    def _view(self):
        camera = self.host.adapter.camera
        native = self.host.adapter.native
        return (
            tuple(camera.center),
            float(camera.zoom),
            native.width(),
            native.height(),
        )

    def _status(self, text, state=None):
        if state:
            self.state = state
        self.project.auto_status.setText(text)
        self.project._update()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.generation += 1
        self._manual_view = None
        self.timer.stop()
        if not enabled:
            self._status("Manual — use Edit at View Center.", "manual")
        else:
            self._status(
                "Following camera — waiting for the view to settle.", "waiting"
            )
            self.schedule()

    def schedule(self, event=None):
        if self.closed or not self.enabled:
            return
        self.generation += 1
        self.deadline = time.monotonic() + self.DEBOUNCE_MS / 1000
        if self._manual_depth or self.project.busy:
            self.timer.stop()
            return
        if self._manual_view is not None and self._view() == self._manual_view:
            return
        self._manual_view = None
        self.timer.start(self.DEBOUNCE_MS)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self.schedule()
        return False

    def pause_for_operation(self):
        self.generation += 1
        self.timer.stop()
        if self.enabled:
            self._status("Following pauses during the project operation.", "waiting")

    @contextmanager
    def manual_control(self):
        self._manual_depth += 1
        self.generation += 1
        self.timer.stop()
        try:
            yield
        finally:
            self._manual_depth -= 1
            self._manual_view = self._view()
            self.window = None
            self._status(
                "Manual area — camera follow resumes when the view changes.", "manual"
            )

    def _sync_edits(self, manager):
        layers = {entry.layer for entry in manager.entries}
        for layer in set(self._edit_connections) - layers:
            for emitter in self._edit_connections.pop(layer):
                emitter.disconnect(self._edited)
        for layer in layers - set(self._edit_connections):
            emitters = (layer.events.data, layer.events.labels_update)
            for emitter in emitters:
                emitter.connect(self._edited)
            self._edit_connections[layer] = emitters

    def _edited(self, event=None):
        if not self.updating:
            self.edit_revision += 1

    def _waiting(self, manager):
        return (
            manager.stroke_active
            or QApplication.mouseButtons() != Qt.MouseButton.NoButton
            or any(
                entry.bounds is not None and not entry.layer.editable
                for entry in manager.entries
            )
        )

    def evaluate(self):
        if (
            self.closed
            or not self.enabled
            or self.project.busy
            or not self.host.isVisible()
        ):
            return
        if self._manual_depth or (
            self._manual_view is not None and self._view() == self._manual_view
        ):
            return
        remaining = self.deadline - time.monotonic()
        if remaining > 0:
            self.timer.start(max(1, ceil(remaining * 1000)))
            return
        if not self.host.adapter.supported_view():
            self._status("Camera follow waits for a normal 2D view.", "waiting")
            return
        backend = self.project.backend
        manager = backend.manager(self.host.viewer) if backend else None
        if manager is None or not manager.entries or manager.loading:
            return
        self._sync_edits(manager)
        if self._waiting(manager):
            self._status(
                "Waiting for the current stroke or navigation drag to finish.",
                "waiting",
            )
            self.timer.start(100)
            return
        if self._future is not None:
            return
        try:
            index = max(0, self.project.masks.currentIndex())
            context = backend.follow_signature(manager)
            if context != self.context:
                self.context = context
                self.window = None
                self._modes.clear()
            native = self.host.adapter.native
            if native.width() < 2 or native.height() < 2:
                self._status("Following waits for a visible canvas.", "waiting")
                return
            world = [
                self.host.adapter.world_at(point)
                for point in (
                    (0, 0),
                    (native.width(), 0),
                    (native.width(), native.height()),
                    (0, native.height()),
                )
            ]
            bounds, shape, basis = backend.follow_view(manager, world, index)
            center = np.mean(np.asarray(world), axis=0)[-2:]
            points = self.host.adapter.canvas_points(
                [center, center + basis[:, 0], center + basis[:, 1]]
            )
            pixel_size = min(
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[0]),
            )
            window = plan_window(
                bounds,
                shape,
                pixel_size,
                current=self.window,
                bytes_per_pixel=sum(
                    np.dtype(entry.source.data.dtype).itemsize
                    for entry in manager.entries
                ),
            )
            if window is None:
                if any(entry.bounds is not None for entry in manager.entries):
                    self.updating = True
                    try:
                        for entry in manager.entries:
                            self._modes[entry.layer] = str(entry.layer.mode)
                        backend.suspend_follow(manager)
                    finally:
                        self.updating = False
                self.window = None
                self._status(
                    "Overview — zoom in to load a full-resolution edit area.",
                    "overview",
                )
                return
            if window == self.window:
                self._status(
                    f"Following camera — full-resolution {window.size} × {window.size} area.",
                    "following",
                )
                return
            backend.flush(self.host.viewer)
            specification = backend.follow_reads(manager, window.origin, window.size)
            self._request = (self.generation, self.edit_revision, context, window)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="spatial-mask-read"
                )
            self._future = self._executor.submit(read_regions, specification)
            self.poll.start()
            self._status("Loading full-resolution edit area…", "loading")
        except Exception as error:
            self._fail(error)

    def _poll(self):
        try:
            self._finish_poll()
        except Exception as error:
            self._future = None
            self.poll.stop()
            self._fail(error)

    def _finish_poll(self):
        if self.closed or self._future is None:
            self.poll.stop()
            return
        if not self._future.done():
            return
        if self.project.busy or self._manual_depth:
            return
        backend = self.project.backend
        manager = backend.manager(self.host.viewer) if backend else None
        generation, revision, context, window = self._request
        if (
            not self.enabled
            or generation != self.generation
            or manager is None
            or context != backend.follow_signature(manager)
            or revision != self.edit_revision
        ):
            self._future = None
            self.poll.stop()
            if self.enabled:
                self.timer.start(0)
            return
        if self._waiting(manager):
            self._status(
                "Waiting for the current stroke or navigation drag to finish.",
                "waiting",
            )
            return
        if backend.follow_dirty(manager):
            self._future = None
            self.poll.stop()
            self.timer.start(0)
            return
        future, self._future = self._future, None
        self.poll.stop()
        try:
            prepared = future.result()
            backend.flush(self.host.viewer)
            self.updating = True
            try:
                modes = {
                    entry.layer: self._modes.get(entry.layer, str(entry.layer.mode))
                    for entry in manager.entries
                }
                backend.apply_follow(manager, window.origin, window.size, prepared)
                for entry in manager.entries:
                    if entry.bounds is not None:
                        entry.layer.mode = modes[entry.layer]
                self._modes.clear()
            finally:
                self.updating = False
            self.window = window
            self.project.mark_dirty()
            self._status(
                f"Following camera — full-resolution {window.size} × {window.size} area.",
                "following",
            )
        except Exception as error:
            self._fail(error)

    def remember_tool(self, layer, mode):
        if layer in self._modes:
            self._modes[layer] = mode

    def _fail(self, error):
        self.project.auto_follow.setChecked(False)
        self.project.status.setText(f"Automatic area update paused: {error}")
        self._status("Paused — current edit buffers retained.", "error")

    def cleanup(self):
        if self.closed:
            return
        self.closed = True
        self.generation += 1
        self.timer.stop()
        self.poll.stop()
        self.host.adapter.camera.events.disconnect(self.schedule)
        try:
            self.host.adapter.native.removeEventFilter(self)
        except RuntimeError:
            pass
        for emitters in self._edit_connections.values():
            for emitter in emitters:
                emitter.disconnect(self._edited)
        self._edit_connections.clear()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
