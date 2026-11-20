"""Simple/Advanced curation shell connected to an independent training dock."""
import napari
import weakref
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QHBoxLayout, QRadioButton,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from ._widget import _LayerActivityIndicator, component_operations_widget
from ._workspace_widget import WorkspaceManagerWidget
from ._quick_train_widget import quick_train_widget, open_training
from ._activity import activity_for


class _ModeStack(QStackedWidget):
    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() else super().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() else super().minimumSizeHint()


class LabelAssistantWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.mode = 'advanced'
        self._closed = False
        self.setObjectName('labelAssistantShell')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        mode_bar = QHBoxLayout()
        self.simple_radio = QRadioButton('Simple')
        self.advanced_radio = QRadioButton('Advanced')
        self.advanced_radio.setChecked(True)
        mode_bar.addWidget(self.simple_radio)
        mode_bar.addWidget(self.advanced_radio)
        mode_bar.addStretch(1)
        layout.addLayout(mode_bar)
        self.activity = activity_for(viewer)
        self._activity_header = _LayerActivityIndicator(self, hub=self.activity)
        self._activity_header.set_context_tooltip('Current task, confirmed tiles, and active annotation batch. Details opens recent activity.')
        layout.addWidget(self._activity_header)
        self.stack = _ModeStack(self)
        self.advanced = QWidget()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self._tool_tabs = QTabWidget()
        self.workspace = WorkspaceManagerWidget(napari_viewer=viewer)
        self._tool_tabs.addTab(self.workspace, 'Workspace')
        unconfigured = {id(layer) for layer in viewer.layers
                        if isinstance(layer, napari.layers.Labels) and 'label_assistant_grid' not in layer.metadata}
        self.components = component_operations_widget(viewer, activity_indicator=self._activity_header)
        self._tool_tabs.addTab(self.components, 'Labels')
        self.training = quick_train_widget(viewer)
        if self.training.parentWidget() is None:
            self.training.setParent(self)
            self.training.hide()
        self.training.attach_annotation_widgets(self.components)
        from ._annotation_batches import events as batch_events
        self._batch_events = batch_events(viewer)
        self._batch_events.changed.connect(self.workspace.mark_changed)
        for panel in (self.components, self.training):
            for control in panel.findChildren(QComboBox):
                control.currentIndexChanged.connect(self.workspace.mark_changed)
            for control in panel.findChildren(QAbstractSpinBox):
                control.valueChanged.connect(self.workspace.mark_changed)
            for control in panel.findChildren(QCheckBox):
                control.toggled.connect(self.workspace.mark_changed)
        for index, tip in enumerate(('Create, open, save, and transfer workspaces.',
                                     'Edit Labels and inspect connected components.')):
            self._tool_tabs.setTabToolTip(index, tip)
        advanced_layout.addWidget(self._tool_tabs)
        from ._simple_curation import SimpleCurationPanel
        self.simple = SimpleCurationPanel(viewer, self.components, self.workspace,
                                          self.training, self, unconfigured)
        viewer._label_assistant_progress_journal = self.simple.journal
        self.stack.addWidget(self.advanced)
        self.stack.addWidget(self.simple)
        layout.addWidget(self.stack, 1)
        local = self.components._local_editing_controls
        local.active_view = lambda: self.isVisible() and (
            self.mode == 'simple' or self._tool_tabs.currentWidget() is self.components)
        self.simple_radio.toggled.connect(lambda checked: self.set_mode('simple') if checked else None)
        self.advanced_radio.toggled.connect(lambda checked: self.set_mode('advanced') if checked else None)
        self._apply_theme()
        viewer.events.theme.connect(self._apply_theme)
        window = viewer.window._qt_window
        self._window = weakref.ref(window)
        window.installEventFilter(self)
        window.destroyed.connect(lambda *_: self._shutdown())
        self.destroyed.connect(lambda *_: self._shutdown())
        self.set_mode('advanced')

    def _apply_theme(self, *_args):
        from napari.utils.theme import get_theme
        theme = get_theme(self.viewer.theme)
        foreground = theme.foreground.as_rgb()
        border = theme.highlight.as_rgb()
        accent = theme.current.as_rgb()
        text = theme.text.as_rgb()
        self.setStyleSheet(f'''
QWidget#labelAssistantShell QToolButton#simpleSectionToggle {{
    padding: 4px 8px; border: 1px solid {border}; border-radius: 6px;
    background: {foreground}; color: {text}; font-weight: bold;
}}
QWidget#labelAssistantShell QFrame#simpleSectionBody {{
    border: 1px solid {border}; border-left: 3px solid {accent};
    border-radius: 6px;
}}
QWidget#labelAssistantShell QPushButton#simplePrimaryAction {{
    border: 1px solid {accent}; border-radius: 4px; padding: 5px;
    font-weight: bold;
}}
''')

    def set_mode(self, mode):
        if self.simple.busy:
            return
        self.mode = 'simple' if mode == 'simple' else 'advanced'
        for radio, checked in ((self.simple_radio, self.mode == 'simple'),
                               (self.advanced_radio, self.mode == 'advanced')):
            radio.blockSignals(True)
            radio.setChecked(checked)
            radio.blockSignals(False)
        self.stack.setCurrentWidget(self.simple if self.mode == 'simple' else self.advanced)
        self.simple.set_active(self.mode == 'simple')
        self.setMinimumWidth(360 if self.mode == 'simple' else 420)
        self.setMaximumWidth(520 if self.mode == 'simple' else 16777215)
        self.stack.updateGeometry()
        self.updateGeometry()

    def show_advanced(self, tab=1, *, qc=False):
        self.set_mode('advanced')
        self._tool_tabs.setCurrentIndex(tab)
        if tab == 1:
            self.components._workflow_tabs.setCurrentIndex(1 if qc else 0)

    def show_training(self):
        return open_training(self.viewer)

    def eventFilter(self, obj, event):
        if obj is self._window() and event.type() == QEvent.Close:
            self.simple.journal.flush()
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.simple.journal.flush()
        super().closeEvent(event)

    def hideEvent(self, event):
        if hasattr(self, 'simple') and self.simple.active:
            self.simple.set_active(False)
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'simple') and self.mode == 'simple':
            self.simple.set_active(True)

    def _shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._batch_events.changed.disconnect(self.workspace.mark_changed)
        self.viewer.events.theme.disconnect(self._apply_theme)
        self.simple.shutdown()
        self.training.shutdown()
        self.activity.has_display = False
        if getattr(self.viewer, '_label_assistant_curation_widget', None) is self:
            self.viewer._label_assistant_curation_widget = None
        if getattr(self.viewer, '_label_assistant_progress_journal', None) is self.simple.journal:
            self.viewer._label_assistant_progress_journal = None
