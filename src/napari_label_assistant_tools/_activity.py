"""Shared, bounded activity state for the persistent curation header."""
from collections import deque
from datetime import datetime
from time import monotonic

from qtpy.QtCore import QObject, QTimer, Qt, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
    QSizePolicy, QToolButton, QVBoxLayout,
)


class ActivityHub(QObject):
    changed = Signal(object)
    summary_changed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.closed = False
        self.destroyed.connect(lambda *_: setattr(self, 'closed', True))
        self.records = {}
        self.history = deque(maxlen=100)
        self.summary = ('Confirmed — / —', 'Choose an image and mask')
        self.has_display = False
        self._sequence = 0
        self._last_current = None
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.expire)

    def report(self, source, state, message, *, completed=None, total=None,
               details=None, cancel=None, priority=50, progress_format=None,
               transient=True, log=True):
        if self.closed:
            return
        previous = self.records.get(source)
        record = dict(source=source, state=state, message=str(message).split('\n')[0],
                      details=str(details or message), completed=completed, total=total,
                      cancel=cancel, priority=priority, progress_format=progress_format)
        # Repeated readiness/poll updates must not extend task completions or fill the log.
        if previous and all(previous.get(key) == value for key, value in record.items()):
            return
        self._sequence += 1
        record['sequence'] = self._sequence
        record['expires'] = monotonic() + 4 if transient and state == 'ready' else None
        self.records[source] = record
        if log and source not in ('view', 'camera', 'context'):
            entry = (source, state, record['details'])
            if not self.history or self.history[-1][1:] != entry:
                self.history.append((datetime.now().strftime('%H:%M:%S'), *entry))
        self._publish()

    def finish(self, source, message=None):
        record = self.records.get(source)
        if record is None or record['state'] in ('error', 'warning'):
            return
        self.report(source, 'ready', message or record['message'],
                    details=message or record['details'], priority=record['priority'],
                    completed=record['completed'] if record['completed'] is not None else 1,
                    total=record['total'] or 1)

    def clear(self, source):
        if self.records.pop(source, None) is not None:
            self._publish()

    def set_summary(self, confirmed, batch):
        if self.closed:
            return
        summary = (str(confirmed), str(batch))
        if summary != self.summary:
            self.summary = summary
            self.summary_changed.emit(*summary)

    @staticmethod
    def _rank(record):
        state, source = record['state'], record['source']
        if source == 'view' and state != 'error':
            rank = -100
        elif source == 'context':
            rank = 25
        elif source == 'camera' and state != 'error':
            rank = 50 if state == 'busy' else -50
        else:
            rank = {'error': 400, 'busy': 200, 'warning': 120, 'ready': 100}.get(state, 0)
        return rank + record['priority'], record['sequence']

    @property
    def current(self):
        return max(self.records.values(), key=self._rank) if self.records else dict(
            source='view', state='idle', message='Ready', details='Ready', cancel=None,
            completed=None, total=None, progress_format=None)

    def _publish(self):
        if self.closed:
            return
        current = self.current
        if current != self._last_current:
            self._last_current = current.copy()
            self.changed.emit(current)
        if any(record['expires'] for record in self.records.values()):
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.timer.stop()

    def expire(self):
        expired = [source for source, record in self.records.items()
                   if record['expires'] is not None and record['expires'] <= monotonic()]
        for source in expired:
            del self.records[source]
        if expired:
            self._publish()


def activity_for(viewer):
    hub = getattr(viewer, '_label_assistant_activity', None)
    if hub is None:
        hub = ActivityHub(viewer.window._qt_window)
        viewer._label_assistant_activity = hub
    return hub


class ActivityIndicator(QFrame):
    """Fixed footprint: current task, global counts, active batch, task progress."""
    _BUSY_FRAMES = ('◐', '◓', '◑', '◒')

    def __init__(self, parent=None, *, hub=None):
        super().__init__(parent)
        self.hub = hub or ActivityHub(self)
        self.hub.has_display = True
        self.setObjectName('layerActivityIndicator')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setMinimumWidth(0)
        self.setFixedHeight(max(94, self.fontMetrics().lineSpacing() * 3 + 32))
        self.state = 'idle'
        self._frame = 0
        self._context_tooltip = ''
        self._full_message = 'Ready'
        self._summary_text = self.hub.summary
        self._record = self.hub.current
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.icon = QLabel('●')
        self.icon.setObjectName('layerActivityIcon')
        self.icon.setFixedWidth(18)
        self.message = self._label()
        self.dismiss_button = QToolButton(text='×')
        self.dismiss_button.setToolTip('Dismiss this message')
        self.dismiss_button.clicked.connect(lambda: self.hub.clear(self._record['source']))
        self.details_button = QToolButton(text='⋯')
        self.details_button.setToolTip('Recent task details')
        self.details_button.clicked.connect(self._show_details)
        row.addWidget(self.icon)
        row.addWidget(self.message, 1)
        row.addWidget(self.dismiss_button)
        row.addWidget(self.details_button)
        layout.addLayout(row)
        self.confirmed_label = self._label()
        self.batch_label = self._label()
        layout.addWidget(self.confirmed_label)
        layout.addWidget(self.batch_label)
        progress_row = QHBoxLayout()
        progress_row.setSpacing(4)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(15)
        self.progress.setTextVisible(True)
        policy = self.progress.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.progress.setSizePolicy(policy)
        self.cancel_button = QToolButton(text='Stop')
        self.cancel_button.setToolTip('Stop the current task after its next safe boundary')
        self.cancel_button.clicked.connect(self._cancel)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.cancel_button)
        layout.addLayout(progress_row)
        self._animation = QTimer(self)
        self._animation.setInterval(140)
        self._animation.timeout.connect(self._advance_busy_frame)
        self.hub.changed.connect(self._display)
        self.hub.summary_changed.connect(self.set_summary)
        self._display(self.hub.current)
        self.set_summary(*self.hub.summary)

    def _label(self):
        label = QLabel()
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return label

    def set_context_tooltip(self, text):
        self._context_tooltip = str(text)
        self._display(self._record)

    def set_summary(self, confirmed, batch):
        self._summary_text = (confirmed, batch)
        self._render_message()

    def set_state(self, state, message, *, details=None):
        self.hub.report('view', state, message, details=details, priority=0,
                        transient=False, log=False)

    def _display(self, record):
        self._record = record
        self.state = state = record['state']
        self._full_message = record['message']
        icons = dict(idle='●', busy=self._BUSY_FRAMES[0], ready='✓', warning='⚠', error='✕')
        icons['busy'] = self._BUSY_FRAMES[self._frame]
        self.icon.setText(icons.get(state, '●'))
        if state == 'busy':
            if not self._animation.isActive():
                self._animation.start()
        else:
            self._animation.stop()
        accent = dict(idle='#868e96', busy='#228be6', ready='#2f9e44', warning='#e67700', error='#e03131').get(state, '#868e96')
        self.setStyleSheet(
            'QFrame#layerActivityIndicator { background-color: #e7f5ff; color: #12344d;'
            ' border: 1px solid #74c0fc; border-radius: 4px; }'
            'QFrame#layerActivityIndicator QLabel { color: #12344d; background-color: transparent; border: none; }'
            f'QFrame#layerActivityIndicator QLabel#layerActivityIcon {{ color: {accent}; font-weight: 700; }}'
            'QFrame#layerActivityIndicator QToolButton { color: #12344d; background: #d0ebff; padding: 0 4px; border: 0; }'
            'QFrame#layerActivityIndicator QProgressBar { color: #12344d; background: #d0ebff; border: 0; border-radius: 2px; text-align: center; }'
            'QFrame#layerActivityIndicator QProgressBar::chunk { background: #74c0fc; border-radius: 2px; }')
        total, completed = record.get('total'), record.get('completed')
        show_progress = state == 'busy' or total is not None
        if show_progress:
            maximum = max(0, int(total or 0))
            self.progress.setRange(0, maximum)
            if maximum:
                self.progress.setValue(max(0, min(int(completed or 0), maximum)))
            self.progress.setFormat(record.get('progress_format') or ('%v / %m · %p%' if maximum else 'Working…'))
        self.progress.setVisible(show_progress)
        self.cancel_button.setVisible(state == 'busy' and record.get('cancel') is not None)
        self.dismiss_button.setVisible(state in ('error', 'warning') and record['source'] != 'context')
        tooltip = '\n\n'.join(filter(None, (record['details'], self._context_tooltip)))
        self.setToolTip(tooltip)
        self.message.setToolTip(tooltip)
        self._render_message()

    def _render_message(self):
        for label, text in ((self.message, self._full_message),
                            (self.confirmed_label, self._summary_text[0]),
                            (self.batch_label, self._summary_text[1])):
            width = max(1, label.width() - 2)
            if label is self.batch_label and ' · ' in text:
                name, _, counts = text.rpartition(' · ')
                suffix = ' · ' + counts
                name_width = width - label.fontMetrics().horizontalAdvance(suffix)
                visible = (label.fontMetrics().elidedText(name, Qt.ElideRight, name_width) + suffix
                           if name_width > 12 else label.fontMetrics().elidedText(counts, Qt.ElideRight, width))
            else:
                visible = label.fontMetrics().elidedText(text, Qt.ElideRight, width)
            label.setText(visible)
            if label is not self.message:
                label.setToolTip(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_message()

    def _advance_busy_frame(self):
        self._frame = (self._frame + 1) % len(self._BUSY_FRAMES)
        self.icon.setText(self._BUSY_FRAMES[self._frame])

    def _cancel(self):
        callback = self._record.get('cancel')
        if callback is not None:
            try:
                callback()
            except Exception as error:
                self.hub.report(self._record['source'], 'error', f'Could not stop task: {error}')

    def _show_details(self):
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle('Recent activity')
        dialog.resize(620, 360)
        layout = QVBoxLayout(dialog)
        log = QPlainTextEdit()
        log.setReadOnly(True)
        lines = [f'{time}  {message}' for time, _source, _state, message in self.hub.history]
        log.setPlainText('\n\n'.join(lines) or self._record['details'])
        log.moveCursor(QTextCursor.End)
        layout.addWidget(log)
        self._details_dialog = dialog
        dialog.show()
