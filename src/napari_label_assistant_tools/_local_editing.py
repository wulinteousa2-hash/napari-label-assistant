"""Local editing controls using Spatial Assistant's camera-follow planner."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QLabel, QPushButton

from ._camera_follow import CameraFollowController
from ._editing_backend import LocalEditingBackend
from ._editing_canvas import CanvasAdapter


class LocalEditingControls(QWidget):
    def __init__(self, viewer, size_control, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.host = self
        self.adapter = CanvasAdapter(viewer)
        self.backend = LocalEditingBackend()
        self.busy = False
        self._manager = None
        self._size_control = size_control
        self.active_view = self.isVisible
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.auto_follow = QCheckBox("Follow camera automatically")
        self.auto_follow.setToolTip(
            "Zoom in to load a bounded full-resolution edit area. Pan to follow; "
            "zoom out for a read-only overview. Edits use the existing mask save and recovery workflow."
        )
        self.auto_status = QLabel()
        self.auto_status.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.manual_edit = QPushButton("Edit at View Center")
        for control in (self.auto_follow, self.manual_edit, self.auto_status, self.status):
            layout.addWidget(control)
        self.follow = CameraFollowController(self)
        self.auto_follow.toggled.connect(self._toggled)
        self.manual_edit.clicked.connect(self.edit_at_center)
        size_control.currentIndexChanged.connect(self._size_changed)

    def _size_changed(self, _index):
        if self._manager is not None:
            self._manager.edit_size = self.follow_size
        self.follow._manual_view = None
        self.follow.schedule()

    @property
    def follow_size(self):
        return int(self._size_control.currentData())

    def _update(self):
        pass  # Status labels are updated directly by the follow controller.

    def mark_dirty(self):
        pass  # MaskEntry owns dirty tracking, recovery, and workspace saving.

    def _toggled(self, enabled):
        if self._manager is not None:
            self._manager.follow_enabled = bool(enabled)
        self.follow.set_enabled(enabled)

    def refresh(self, manager):
        if manager is self._manager:
            return
        if self._manager is not None and getattr(self._manager, 'follow', None) is self.follow:
            self._manager.follow = None
        self.follow.set_enabled(False)
        self._manager = manager
        if manager is not None:
            manager.follow = self.follow
            index = self._size_control.findData(manager.edit_size)
            if index < 0:
                self._size_control.addItem(f'{manager.edit_size} px', manager.edit_size)
                index = self._size_control.count() - 1
            self._size_control.setCurrentIndex(index)
        enabled = bool(manager is not None and manager.follow_enabled)
        self.auto_follow.blockSignals(True)
        self.auto_follow.setChecked(enabled)
        self.auto_follow.blockSignals(False)
        self.follow.set_enabled(enabled)

    def edit_at_center(self):
        try:
            with self.follow.manual_control():
                native = self.adapter.native
                world = self.adapter.world_at((native.width() / 2, native.height() / 2))
                self.backend.edit_at(self.viewer, world, self.currentIndex(),
                                     int(self._size_control.currentData()))
            self.status.clear()
        except Exception as error:
            self.status.setText(str(error))

    @property
    def masks(self):
        return self

    def currentIndex(self):
        manager = self.backend.manager(self.viewer)
        active = self.viewer.layers.selection.active
        if manager is not None:
            return next((i for i, entry in enumerate(manager.entries)
                         if active is entry.layer or active is entry.overview), 0)
        return 0

    def shutdown(self):
        self.follow.cleanup()
        if self._manager is not None and getattr(self._manager, 'follow', None) is self.follow:
            self._manager.follow = None
