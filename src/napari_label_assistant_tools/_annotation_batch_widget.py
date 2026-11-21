"""Shared quick-action buttons for Simple, Labels, and Training views."""
from contextlib import suppress

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel,
                           QComboBox, QMenu, QInputDialog, QMessageBox, QSizePolicy)

from . import _annotation_batches as batches


class AnnotationBatchWidget(QWidget):
    changed = Signal()

    def __init__(self, viewer, parent=None, *, review=None, selected_regions=None):
        super().__init__(parent)
        self.viewer, self.review = viewer, review
        self.selected_regions = selected_regions
        self.image = self.truth = None
        self.closed = False
        self.busy = False
        self._stamp = None
        self.buttons = {}
        from ._batch_overlays import controller
        self.overlay = controller(viewer)
        self.overlay.clients.add(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.title = QLabel('Annotation batches')
        self.title.setToolTip(
            'A batch groups reviewed regions. The Training menu independently selects which '
            'reviewed regions to use for training.'
        )
        layout.addWidget(self.title)
        self.batch_help = QLabel('Each batch groups reviewed regions. Choose which batch to train on below.')
        self.batch_help.setWordWrap(True)
        layout.addWidget(self.batch_help)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        layout.addLayout(self.grid)
        self.new_button = QPushButton('+ New batch')
        self.new_button.setToolTip('Start an empty batch for annotation and training. Global review coverage and saved models are preserved.')
        self.new_button.clicked.connect(lambda: self.run(lambda: self._change('new')))
        self.scope = QComboBox()
        self.scope.setMinimumContentsLength(10)
        self.scope.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.scope.setToolTip('Explicit training selection. Viewing another batch does not change this setting.')
        self.scope.currentIndexChanged.connect(lambda: self.run(self._training_changed))
        layout.addWidget(self.scope)
        self.display = QComboBox()
        self.display.addItem('Show: Active batch', 'active')
        self.display.addItem('Show: All batches', 'all')
        self.display.addItem('Show: Hide', 'hide')
        self.display.setToolTip('Display only. Yellow = pending selection; red X = needs review. Training selection is unchanged.')
        self.display.currentIndexChanged.connect(lambda: self.run(lambda: self._change('view', name=self.display.currentData())))
        layout.addWidget(self.display)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.setStyleSheet('QPushButton[annotationBatch="true"]:checked { border: 2px solid palette(highlight); font-weight: bold; }')
        self._events = batches.events(viewer)
        self._events.changed.connect(self.refresh)
        self.destroyed.connect(lambda *_: self.shutdown())
        self.refresh()

    def set_context(self, image, truth, review=None):
        self.image, self.truth = image, truth
        if review is not None:
            self.review = review
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.overlay.set_context(self.image, self.truth)

    def _training_changed(self):
        value = self.scope.currentData()
        if value in ('all', 'active'):
            self._change(value)
        elif value is not None:
            self._change('training', batch_id=value)

    def _blocked(self):
        from ._mask_tiles import manager_for
        trainer = getattr(self.viewer, '_label_assistant_training_widget', None)
        shell = getattr(self.viewer, '_label_assistant_curation_widget', None)
        manager = manager_for(self.viewer)
        return (self.busy or (trainer is not None and trainer.worker is not None) or
                bool(manager is not None and manager.stroke_active) or
                bool(shell is not None and hasattr(shell, 'simple') and shell.simple.busy))

    def set_busy(self, busy):
        self.busy = busy
        self.refresh()

    def _change(self, action, **kwargs):
        if self.image is None or self.truth is None:
            raise ValueError('Choose a matching image and Labels layer first.')
        if self._blocked():
            raise ValueError('Wait for the current operation to finish before changing annotation batches.')
        self.overlay.set_context(self.image, self.truth)
        batches.change(self.viewer, self.image, self.truth, action, **kwargs)
        self.overlay.render()
        self.changed.emit()

    def refresh(self, *_args):
        if self.closed:
            return
        from ._reviewed_regions import effective_areas, region_reviewed
        from ._mask_tiles import full_source
        from ._batch_overlays import review_state
        _, state = review_state(self.viewer, self.image, self.truth)
        if self.isVisible():
            self.overlay.set_context(self.image, self.truth)
        root = full_source(self.truth).metadata.get(batches.KEY) if self.truth is not None else None
        stamp = (id(self.image), id(self.truth), id(root), repr(state.get('areas')), repr(state.get('invalid_regions')), self._blocked())
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            pair = batches.get_pair(self.image, self.truth)
        except ValueError as error:
            self.summary.setText(str(error))
            self.scope.setEnabled(False)
            self.display.setEnabled(False)
            self.new_button.setEnabled(False)
            for button in self.buttons.values():
                button.setEnabled(False)
            return
        records = pair['batches'] if pair else [dict(id='initial', number=1, name='Batch 1', regions=state.get('review_cells', state.get('areas', [])),
                created_at_utc=None, updated_at_utc=None, finished_at_utc=None, revision=0)]
        ids = tuple(b['id'] for b in records)
        if ids != tuple(self.buttons):
            while self.grid.count():
                item = self.grid.takeAt(0)
                if item.widget() is not self.new_button:
                    item.widget().hide()
                    item.widget().deleteLater()
            self.buttons = {}
            for i, batch in enumerate(records):
                button = QPushButton()
                button.setProperty('annotationBatch', True)
                button.setCheckable(True)
                button.setMinimumWidth(0)
                button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                identifier = batch['id']
                button.clicked.connect(lambda _checked=False, key=identifier: self.run(lambda: self.activate(key)))
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                button.customContextMenuRequested.connect(lambda point, key=identifier, control=button: self._menu(key, control, point))
                self.buttons[identifier] = button
                self.grid.addWidget(button, i//3, i%3)
            self.grid.addWidget(self.new_button, len(records)//3, len(records)%3)
        available = self.image is not None and self.truth is not None
        enabled = available and not self._blocked()
        active = batches.active_batch(pair) if pair else records[0]
        areas = effective_areas(state)
        exact = {tuple(r) for r in areas}
        for batch in records:
            button = self.buttons[batch['id']]
            button.setText(f"{batch['name']}\n{len(batch['regions'])} region(s)" + (' · finished' if batch['finished_at_utc'] else ''))
            button.setChecked(batch['id'] == active['id'])
            button.setEnabled(enabled)
            color = batches.batch_color(batch)
            if button.property('batchColor') != color:
                pixmap = QPixmap(12,12)
                pixmap.fill(QColor(color))
                button.setIcon(QIcon(pixmap))
                button.setProperty('batchColor', color)
            button.setToolTip(f"{batch['name']}\nCreated: {batch['created_at_utc'] or 'Earlier approval dates not recorded'}\n"
                              f"Updated: {batch['updated_at_utc'] or 'Not recorded'}\nMembership revision: {batch['revision']}\n"
                              'Click to view and make active for annotation. Training selection is unchanged.\nRight-click for actions and history.')
        ready = sum(tuple(r) in exact or region_reviewed(r, areas) for r in active['regions'])
        self.scope.blockSignals(True)
        self.scope.clear()
        self.scope.addItem('Training: all reviewed annotations', 'all')
        for record in records:
            self.scope.addItem(f"Training: {record['name']}", record['id'] if pair else 'active')
        value = batches.training_batch(pair)['id'] if pair and pair['scope'] == 'active' else 'all'
        self.scope.setCurrentIndex(self.scope.findData(value))
        self.scope.blockSignals(False)
        self.display.blockSignals(True)
        self.display.setCurrentIndex(self.display.findData(pair.get('view_mode', 'active') if pair else 'active'))
        self.display.blockSignals(False)
        self.display.setEnabled(enabled)
        self.scope.setEnabled(enabled)
        self.new_button.setEnabled(enabled)
        self.summary.setText(f"Active for annotation: {active['name']} · {ready} / {len(active['regions'])} regions reviewed.\n"
                             'Yellow: pending · Red ×: needs review. Colors show review membership, not training coverage.')

    def activate(self, identifier):
        if identifier == 'initial':
            pair = batches.ensure_pair(self.viewer, self.image, self.truth)
            identifier = pair['active_id']
        self._change('select', batch_id=identifier)

    def show_regions(self):
        self._change('view', name='active')

    def _selected(self):
        regions = self.selected_regions() if self.selected_regions is not None else []
        if not regions and self.review is not None and self.review.selected_cell is not None:
            _, b, _, _ = self.review.selected_cell
            regions = [(b.y0,b.x0,b.y1,b.x1)]
        if not regions:
            raise ValueError('Select review tiles on the canvas first.')
        return regions

    def add_reviewed(self, regions=None):
        from ._reviewed_regions import associated_mask, effective_areas, region_reviewed
        mask = associated_mask(self.viewer, self.image, self.truth)
        areas = effective_areas(mask.metadata.get('label_assistant_review', {})) if mask is not None else []
        regions = self._selected() if regions is None else regions
        if not all(region_reviewed(r, areas) for r in regions):
            raise ValueError('Review and approve every selected region before adding it to a batch.')
        self._change('add', regions=regions)

    def _menu(self, identifier, button, point):
        if self._blocked():
            return
        menu = QMenu(self)
        for text, action, needs_regions in (
                ('Show batch regions', self.show_regions, False),
                ('Add selected reviewed regions', self.add_reviewed, True),
                ('Remove selected regions from batch', lambda regions: self._change('remove', regions=regions), True),
                ('Rename…', self.rename, False), ('Finish batch', lambda: self._change('finish'), False),
                ('Reopen batch', lambda: self._change('reopen'), False), ('Details / review history…', self.details, False)):
            menu.addAction(text).triggered.connect(
                lambda _checked=False, fn=action, selected=needs_regions:
                self.run(lambda: self._menu_action(identifier, fn, selected)))
        menu.exec_(button.mapToGlobal(point))

    def _menu_action(self, identifier, action, needs_regions=False):
        # Switching batches clears pending canvas selection. Capture the explicit
        # membership action's regions before that switch, including multi-select.
        regions = list(self._selected()) if needs_regions else None
        self.activate(identifier)
        if needs_regions:
            action(regions)
        else:
            action()

    def rename(self):
        pair = batches.ensure_pair(self.viewer, self.image, self.truth)
        value, accepted = QInputDialog.getText(self, 'Rename annotation batch', 'Name', text=batches.active_batch(pair)['name'])
        if accepted:
            self._change('rename', name=value)

    def details(self):
        batch = batches.active_batch(batches.ensure_pair(self.viewer, self.image, self.truth))
        details = (f"{batch['name']}\nCreated: {batch['created_at_utc'] or 'Not recorded for earlier annotations'}\n"
                   f"Updated: {batch['updated_at_utc'] or 'Not recorded'}\nMembership revision: {batch['revision']}\n"
                   f"Regions: {len(batch['regions'])}\nOrigin: {batch['origin']}\n\nRecent review actions:\n")
        details += '\n'.join(f"{event['at_utc']} · {event['action']} · {event.get('bounds', '')}" for event in batch['events'][-20:])
        from ._model_project import project
        uses = []
        for model in project(self.viewer)['models']:
            for history in model.get('history', []):
                for selection in history.get('source', {}).get('annotation_batches', []):
                    if selection.get('id') == batch['id']:
                        uses.append(f"{model['name']} · {history.get('started_at_utc','date not recorded')} · membership revision {selection['revision']}")
        details += '\n\nRecorded training rounds using this batch:\n' + ('\n'.join(uses) or 'No batch-specific training use recorded.')
        details += '\n\nMembership and review events are recorded. Historical label pixels are not frozen copies.'
        QMessageBox.information(self, 'Annotation batch', details)

    def run(self, action):
        try:
            action()
        except Exception as error:
            self.summary.setText(str(error))

    def shutdown(self):
        if not self.closed:
            self.closed = True
            with suppress(Exception):
                self._events.changed.disconnect(self.refresh)
            with suppress(Exception):
                self.overlay.release(self)
