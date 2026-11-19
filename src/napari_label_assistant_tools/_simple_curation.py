"""A tile-centered presentation over the existing annotation controllers."""
from copy import deepcopy
from contextlib import suppress

import napari
import numpy as np
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import (
    QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget,
)

from ._coordinates import mask_world_affine, matching_mask_transforms
from ._dynamic_edit import TileBounds, is_internal_layer
from ._mask_tiles import full_source, full_data, manager_for
from ._quick_train import array_2d
from ._reviewed_regions import KEY as REVIEW_KEY, associated_mask, effective_areas, identity
from ._tile_progress import KEY, ProgressJournal, active_path


def _shape_2d(layer):
    try:
        return tuple(array_2d(full_data(layer)).shape)
    except (ValueError, TypeError):
        return None


class StepSection(QWidget):
    def __init__(self, title, *, expanded=True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = QToolButton(text=title.replace('&', '&&'))
        self.toggle.setObjectName('simpleSectionToggle')
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.body = QFrame()
        self.body.setObjectName('simpleSectionBody')
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(8, 8, 8, 8)
        self.content.setSpacing(6)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self._show)
        self._show(expanded)

    def _show(self, expanded):
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)


class SimpleCurationPanel(QScrollArea):
    def __init__(self, viewer, components, workspace, training, shell, unconfigured=()):
        super().__init__(shell)
        self.viewer, self.c, self.workspace, self.training, self.shell = viewer, components, workspace, training, shell
        self.review = components._reviewed_area_widget
        self.local = components._local_editing_controls
        self.active = False
        self.busy = False
        self.closed = False
        self._syncing = False
        self._writing = False
        self._pair = None
        self._focus_cell = None
        self.pending = []
        self._undo = None
        self._revision = 0
        self._save_activity = None
        self._tile_progress = None
        self._unconfigured = set(unconfigured)
        self._connections = []
        self._watched = {}
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumWidth(360)
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)
        self.workspace_section = StepSection('1  Workspace')
        self.project_label = QLabel()
        self.project_label.setWordWrap(True)
        self.workspace_section.content.addWidget(self.project_label)
        row, buttons = self._buttons(('Open', workspace.open_workspace), ('Save', workspace.save))
        self.workspace_section.content.addWidget(row)
        self.open_button, self.save_button = buttons
        layout.addWidget(self.workspace_section)

        self.edit_section = StepSection('2  Annotate & Review')
        form = QFormLayout()
        self.image_combo, self.mask_combo = QComboBox(), QComboBox()
        for combo in (self.image_combo, self.mask_combo):
            combo.setMinimumWidth(0)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self.grid_combo = QComboBox()
        for size in (512, 1024, 2048, 4096):
            self.grid_combo.addItem(f'{size} × {size} px', (size, size))
        self.grid_combo.setToolTip('Review area size. The camera-follow buffer and training patch size are managed separately.')
        form.addRow('Image', self.image_combo)
        form.addRow('Mask', self.mask_combo)
        form.addRow('Review tile', self.grid_combo)
        self.edit_section.content.addLayout(form)
        self.canvas_mode = QComboBox()
        self.canvas_mode.addItem('Edit — paint / erase', 'edit')
        self.canvas_mode.addItem('Review — select tiles', 'review')
        self.edit_section.content.addWidget(self.canvas_mode)
        self.follow_label = QLabel('Camera follow · automatic')
        self.follow_label.setWordWrap(True)
        self.edit_section.content.addWidget(self.follow_label)
        row, buttons = self._buttons(('Paint', lambda: self._tool('paint')), ('Erase', lambda: self._tool('erase')))
        self.paint_button, self.erase_button = buttons
        self.edit_section.content.addWidget(row)
        self.current_label = QLabel('Choose an image and mask.')
        self.current_label.setWordWrap(True)
        self.edit_section.content.addWidget(self.current_label)
        self.pending_label = QLabel('No pending tile selection')
        self.pending_label.setWordWrap(True)
        self.edit_section.content.addWidget(self.pending_label)
        row, buttons = self._buttons(('Reviewed & Next', self.approve), ('Skip', self.skip))
        self.approve_button, self.skip_button = buttons
        self.approve_button.setObjectName('simplePrimaryAction')
        self.edit_section.content.addWidget(row)
        description = QLabel('Approve only after checking all objects and background.')
        description.setWordWrap(True)
        self.edit_section.content.addWidget(description)
        layout.addWidget(self.edit_section)

        self.progress_section = StepSection('3  Progress & Dataset')
        self.progress_label, self.dataset_label, self.persistence_label = QLabel(), QLabel(), QLabel()
        for label in (self.progress_label, self.dataset_label, self.persistence_label):
            label.setWordWrap(True)
            self.progress_section.content.addWidget(label)
        from ._annotation_batch_widget import AnnotationBatchWidget
        self.batch_widget = AnnotationBatchWidget(viewer, self, review=self.review,
                                                   selected_regions=lambda: self.pending)
        self.progress_section.content.insertWidget(1, self.batch_widget)
        self.workflow_progress = QProgressBar()
        self.workflow_progress.setFixedHeight(20)
        self.workflow_progress.setTextVisible(True)
        self.workflow_progress.hide()
        self.progress_section.content.addWidget(self.workflow_progress)
        row, buttons = self._buttons(('Show Coverage', self.review.show_coverage), ('Prepare Dataset', self.prepare))
        self.coverage_button, self.prepare_button = buttons
        self.progress_section.content.addWidget(row)
        row, buttons = self._buttons(('Training amount…', self._open_patch_selection),
                                     ('Open Training', lambda: shell.show_advanced(2)))
        self.patch_selection_button, self.open_training_button = buttons
        self.progress_section.content.addWidget(row)
        self.cancel_button = QPushButton('Stop tile operation')
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(lambda: setattr(self, '_cancelled', True))
        self.progress_section.content.addWidget(self.cancel_button)
        layout.addWidget(self.progress_section)

        self.tools_section = StepSection('Tools', expanded=False)
        actions = (
            ('Clear pending selection', self.clear_selection),
            ('Revoke selected / current tile', self.revoke),
            ('Revoke all reviewed areas…', self.revoke_all),
            ('Undo last review action', self.undo_review),
            ('View & Compare', lambda: self._advanced_section(self.c._view_compare_section)),
            ('Label Tools', lambda: self._advanced_section(self.c._label_tools_section)),
            ('Screen objects / mask cleanup', lambda: shell.show_advanced(1, qc=True)),
            ('Workspace settings / Save As', lambda: shell.show_advanced(0)),
        )
        self.tool_buttons = []
        for text, callback in actions:
            button = QPushButton(text.replace('&', '&&'))
            button.clicked.connect(lambda _checked=False, fn=callback: self.run(fn))
            self.tools_section.content.addWidget(button)
            self.tool_buttons.append(button)
        layout.addWidget(self.tools_section)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addStretch(1)

        self.journal = ProgressJournal(viewer, self)
        self.journal.status.connect(self.persistence_label.setText)
        self.journal.activity.connect(self._progress_activity)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(30)
        self.refresh_timer.timeout.connect(self.refresh_sources)
        self.poll = QTimer(self)
        self.poll.setInterval(250)
        self.poll.timeout.connect(self.refresh_status)
        self.poll.start()
        self._connect(viewer.layers.events.inserted, self._inserted)
        self._connect(viewer.layers.events.removed, self._removed)
        self._connect(self.review.cell_selected, self._picked)
        self._connect(self.review.coverage_changed, self._coverage_changed)
        from ._annotation_batches import events as batch_events
        self._connect(batch_events(viewer).changed, self._batch_changed)
        self._connect(self.c._target_combo.currentTextChanged, self._schedule_refresh)
        self._connect(self.review.image_combo.currentIndexChanged, self._schedule_refresh)
        self.mask_combo.currentIndexChanged.connect(lambda _i: self._choose_mask())
        self.image_combo.currentIndexChanged.connect(lambda _i: self._choose_image())
        self.grid_combo.currentIndexChanged.connect(lambda _i: self._grid_changed())
        self.canvas_mode.currentIndexChanged.connect(lambda _i: self.run(self._apply_canvas_mode))
        self.refresh_sources()

    def _buttons(self, *actions):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        buttons = []
        for text, callback in actions:
            button = QPushButton(text.replace('&', '&&'))
            button.clicked.connect(lambda _checked=False, fn=callback: self.run(fn))
            layout.addWidget(button)
            buttons.append(button)
        return row, buttons

    def _connect(self, emitter, callback):
        emitter.connect(callback)
        self._connections.append((emitter, callback))

    def _progress_activity(self, state):
        self._save_activity = state
        self._update_workflow_progress()
        if state == 'saving' and self.workflow_progress.isVisible():
            # Paint the busy state before synchronous bounded writes, without
            # processing other events or allowing reentrant editing actions.
            self.workflow_progress.repaint()

    def _update_workflow_progress(self):
        bar, tiles = self.workflow_progress, self._tile_progress
        if self.busy and tiles is not None:
            bar.setRange(0, tiles['total'])
            bar.setValue(tiles['completed'])
            bar.setFormat(f"{tiles['action']} · %v / %m tiles")
        elif self._save_activity == 'error':
            bar.setRange(0, 1)
            bar.setValue(0)
            bar.setFormat('Tile progress not saved')
        elif tiles is not None and 'stopped' in tiles:
            bar.setRange(0, tiles['total'])
            bar.setValue(tiles['completed'])
            bar.setFormat(('Stopped' if tiles['stopped'] else 'Updated') + ' · %v / %m tiles')
        elif self._save_activity in ('queued', 'saving'):
            bar.setRange(0, 0)
            bar.setFormat('Saving tile progress…')
        elif tiles is not None:
            bar.setRange(0, tiles['total'])
            bar.setValue(tiles['completed'])
            bar.setFormat(('Stopped' if tiles.get('stopped') else 'Updated') + ' · %v / %m tiles')
        elif self._save_activity == 'saved':
            bar.setRange(0, 1)
            bar.setValue(1)
            bar.setFormat('Tile progress saved')
        else:
            bar.hide()
            return
        bar.show()

    @property
    def truth(self):
        name = self.c._target_combo.currentText()
        layer = self.viewer.layers[name] if name in self.viewer.layers else None
        if (not isinstance(layer, napari.layers.Labels) or is_internal_layer(layer)
                or layer.metadata.get(REVIEW_KEY, {}).get('role') in ('reviewed_regions', 'reviewed_annotations')
                or _shape_2d(layer) is None):
            return None
        return layer

    @property
    def image(self):
        return self.image_combo.currentData()

    def _inserted(self, event):
        layer = event.value
        if isinstance(layer, napari.layers.Labels) and not layer.metadata.get('label_assistant_grid'):
            if not getattr(self.viewer, '_label_assistant_workspace_loading', False):
                self._unconfigured.add(id(layer))
        self._schedule_refresh()

    def _removed(self, event):
        for emitter, callback in self._watched.pop(event.value, []):
            with suppress(ValueError, RuntimeError):
                emitter.disconnect(callback)
        self._schedule_refresh()

    def _schedule_refresh(self, *_args):
        if not self.closed:
            self.refresh_timer.start()

    def refresh_sources(self):
        if self.closed or self.busy:
            return
        if getattr(self.viewer, '_label_assistant_workspace_loading', False):
            self.refresh_timer.start()
            return
        self._syncing = True
        try:
            source = self.truth
            masks = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)
                     and not is_internal_layer(layer)
                     and _shape_2d(layer) is not None
                     and layer.metadata.get(REVIEW_KEY, {}).get('role') not in ('reviewed_regions', 'reviewed_annotations')]
            self.mask_combo.clear()
            self.mask_combo.addItem('Choose mask…', None)
            for layer in masks:
                self.mask_combo.addItem(layer.name, layer)
            if source in masks:
                self.mask_combo.setCurrentIndex(self.mask_combo.findData(source))
            elif len(masks) == 1 and self.active:
                source = masks[0]
                self.c._target_combo.setCurrentText(source.name)
                self.mask_combo.setCurrentIndex(1)
            else:
                source = None
            images = []
            if source is not None:
                for layer in self.viewer.layers:
                    if isinstance(layer, napari.layers.Image) and not layer.rgb and not is_internal_layer(layer):
                        if (_shape_2d(layer) == _shape_2d(source)
                                and matching_mask_transforms(layer, full_source(source))):
                            images.append(layer)
            session = full_source(source).metadata.get(KEY, {}) if source is not None else {}
            chosen = next((image for image in images if image.metadata.get(REVIEW_KEY, {}).get('id') == session.get('image_id')), None)
            if chosen is None and len(images) == 1:
                chosen = images[0]
            self.image_combo.clear()
            self.image_combo.addItem('Choose matching image…', None)
            for image in images:
                self.image_combo.addItem(image.name, image)
            if chosen is not None:
                self.image_combo.setCurrentIndex(self.image_combo.findData(chosen))
                self.review.image_combo.setCurrentIndex(self.review.image_combo.findData(chosen))
            pair = (source, chosen)
            if self.active and pair != self._pair:
                self._pair = pair
                self._tile_progress = None
                self._undo = None
                self.pending = []
                if source is not None and chosen is not None:
                    shape = list(array_2d(full_data(source)).shape)
                    if session.get('image_id') == identity(chosen) and session.get('shape') == shape:
                        self.pending = [tuple(map(int, rect)) for rect in session.get('pending', [])
                                        if len(rect) == 4 and 0 <= rect[0] < rect[2] <= shape[0]
                                        and 0 <= rect[1] < rect[3] <= shape[1]]
                        registration = full_source(source).metadata.get('label_assistant_grid', {})
                        grid = (registration.get('cell_height'), registration.get('cell_width'))
                        if all(grid):
                            self.c._grid_y_spin.setValue(grid[0])
                            self.c._grid_x_spin.setValue(grid[1])
                    elif self.active and id(source) in self._unconfigured:
                        self.c._grid_y_spin.setValue(1024)
                        self.c._grid_x_spin.setValue(1024)
                        self._unconfigured.discard(id(source))
                if self.review.selected_cell is None or self.review.selected_cell[0] is not source:
                    self.review.clear_cell()
                self.c._set_pending_training_tiles(self.pending)
            grid = (self.c._grid_y_spin.value(), self.c._grid_x_spin.value())
            index = self.grid_combo.findData(grid)
            if index < 0:
                self.grid_combo.addItem(f'{grid[0]} × {grid[1]} px', grid)
                index = self.grid_combo.count() - 1
            self.grid_combo.setCurrentIndex(index)
            for layer in self.viewer.layers:
                if is_internal_layer(layer) or layer in self._watched:
                    continue
                connections = []
                for name, callback in (('metadata', self._metadata_changed), ('data', self._pixels_changed),
                                       ('labels_update', self._pixels_changed), ('name', self._schedule_refresh)):
                    emitter = getattr(layer.events, name, None)
                    if emitter is not None:
                        emitter.connect(callback)
                        connections.append((emitter, callback))
                self._watched[layer] = connections
        finally:
            self._syncing = False
        if self.active:
            self.run(self._apply_canvas_mode)
            self._draw_pending()
        self.refresh_status()

    def _choose_mask(self):
        if not self._syncing and self.mask_combo.currentData() is not None:
            self.c._target_combo.setCurrentText(self.mask_combo.currentData().name)
            self.refresh_sources()

    def _choose_image(self):
        if not self._syncing and self.image is not None:
            self.review.image_combo.setCurrentIndex(self.review.image_combo.findData(self.image))
            self.pending = []
            self._pair = (self.truth, self.image)
            self.review.clear_cell()
            self._persist()
            self.run(self._apply_canvas_mode)

    def _grid_changed(self):
        if not self._syncing and self.grid_combo.currentData():
            sy, sx = self.grid_combo.currentData()
            self.c._grid_y_spin.setValue(sy)
            self.c._grid_x_spin.setValue(sx)
            self._persist()
            self.refresh_status()

    def _metadata_changed(self, *_args):
        if not self._writing and not getattr(self.viewer, '_label_assistant_workspace_loading', False):
            self.journal.schedule()

    def _pixels_changed(self, event):
        if event.source is not self.truth:
            return
        entry = getattr(self.truth, '_label_assistant_mask_entry', None)
        if (entry is not None and entry.manager.loading) or self.local.follow.updating:
            return
        self._revision += 1
        if not self.busy:
            self._tile_progress = None
        self._undo = None
        self.journal.schedule()

    def _persist(self):
        truth, image = self.truth, self.image
        if truth is None or image is None or self._syncing:
            return
        self.c._set_pending_training_tiles(self.pending)
        session = dict(image_id=identity(image), shape=list(array_2d(full_data(truth)).shape),
                       grid=[self.c._grid_y_spin.value(), self.c._grid_x_spin.value()],
                       pending=[list(rect) for rect in self.pending])
        self._writing = True
        try:
            for owner in (truth, full_source(truth)):
                owner.metadata[KEY] = session
        finally:
            self._writing = False
        self.journal.schedule()

    def set_active(self, active):
        self.active = active
        if not active:
            self._restore_pending_style()
        if self.busy:
            if not active:
                self._cancelled = True
            return
        if active:
            self.c._workflow_tabs.setCurrentIndex(0)
            self.c._click_select_check.setChecked(False)
            self.c._tile_select_check.setChecked(False)
            self.review.advanced_check.setChecked(False)
            self.refresh_sources()
        else:
            self.review.mouse_review_check.setChecked(False)
            for attr in ('simple_focus_overlay', 'selection_overlay'):
                layer = getattr(self.review, attr, None)
                if layer is not None:
                    layer.visible = False
            self.journal.flush()

    def _apply_canvas_mode(self):
        if not self.active or self.busy or self.training.worker is not None or self.truth is None or self.image is None:
            return
        from .workspace.service import _zarr_reference
        if manager_for(self.viewer) is None or not manager_for(self.viewer).entries:
            if _zarr_reference(self.truth) is None:
                self.message.setText('Save the workspace once to enable bounded editing and progress recovery for this mask.')
                return
            from ._mask_tiles import set_mode
            name = self.truth.name
            set_mode(self.viewer, 'grid')
            self.c._target_combo.setCurrentText(name)
            self.local.refresh(manager_for(self.viewer))
            self.refresh_timer.start()
        self.local.auto_follow.setChecked(True)
        review_mode = self.canvas_mode.currentData() == 'review'
        self.review.advanced_check.setChecked(False)
        self.review.mouse_action_combo.setCurrentIndex(self.review.mouse_action_combo.findData('select'))
        self.review.mouse_review_check.setChecked(review_mode)
        focus = getattr(self.review, 'simple_focus_overlay', None)
        if focus is not None:
            focus.visible = not review_mode
        if not review_mode:
            self.local.follow.resume_from_review()
            self.viewer.layers.selection.active = self.truth
            if self.truth.editable and str(self.truth.mode) not in ('paint', 'erase'):
                self.truth.mode = 'paint'
        self.refresh_status()

    def _tool(self, mode):
        self.canvas_mode.setCurrentIndex(self.canvas_mode.findData('edit'))
        self._apply_canvas_mode()
        if self.truth is not None and self.truth.editable:
            self.viewer.layers.selection.active = self.truth
            self.truth.mode = mode

    def _current_from_view(self):
        truth = self.truth
        if truth is None:
            return
        manager = manager_for(self.viewer)
        if (manager is not None and manager.stroke_active) or self.c._dynamic_edit_controller.stroke_active:
            return
        native = self.local.adapter.native
        world = self.local.adapter.world_at((native.width()/2, native.height()/2))
        transform = mask_world_affine(full_source(truth))
        y, x = np.floor(np.linalg.solve(transform[:2,:2], np.asarray(world)[-2:] - transform[:2,2])).astype(int)
        h, w = array_2d(full_data(truth)).shape
        if not (0 <= y < h and 0 <= x < w):
            return
        sy, sx = self.c._grid_y_spin.value(), self.c._grid_x_spin.value()
        bounds = TileBounds(y//sy*sy, x//sx*sx, min((y//sy+1)*sy,h), min((x//sx+1)*sx,w))
        current = self.review.selected_cell
        if current is None or current[0] is not truth or current[1] != bounds:
            self.review.selected_cell = (truth, bounds, sy, sx)
            self.review.update_status()
        if self._focus_cell != (truth, bounds):
            active = self.viewer.layers.selection.active
            self.review.make_outline(truth, [[bounds.y0,bounds.x0,bounds.y1,bounds.x1]],
                                     'Current review tile', 'cyan', 'simple_focus_overlay')
            if active is not None and any(layer is active for layer in self.viewer.layers):
                self.viewer.layers.selection.active = active
            self._focus_cell = (truth, bounds)

    def _picked(self):
        if not self.active or self.busy or self.canvas_mode.currentData() != 'review' or self.review.selected_cell is None:
            return
        _, b, _, _ = self.review.selected_cell
        self._tile_progress = None
        rect = (b.y0,b.x0,b.y1,b.x1)
        if rect in self.pending:
            self.pending.remove(rect)
        else:
            self.pending.append(rect)
        self._draw_pending()
        self._persist()
        self.refresh_status()

    def _draw_pending(self):
        if self.truth is not None and self.active:
            active = self.viewer.layers.selection.active
            layer = self.review.make_outline(self.truth, self.pending, 'Pending review tiles', 'yellow', 'selection_overlay')
            if layer is not getattr(self, '_pending_styled_layer', None):
                self._pending_styled_layer = layer
                self._pending_original_opacity = layer.opacity
            layer.edge_color = '#ffd600'
            layer.face_color = np.asarray([[1.0, 0.84, 0.0, 0.16]])
            layer.opacity = 1.0
            self._pending_edge_stamp = None
            self._update_pending_edge_width()
            if active is not None and any(layer is active for layer in self.viewer.layers):
                self.viewer.layers.selection.active = active

    def _restore_pending_style(self):
        # The same Shapes layer is reused by Advanced's existing review tools.
        # Do not leave a zoom-compensated data width on their ordinary outline.
        layer = getattr(self, '_pending_styled_layer', None)
        if layer is not None and any(item is layer for item in self.viewer.layers):
            layer.edge_width = 2
            layer.edge_color = 'yellow'
            layer.face_color = 'transparent'
            layer.opacity = self._pending_original_opacity
        self._pending_styled_layer = None
        self._pending_edge_stamp = None

    def _update_pending_edge_width(self):
        """Keep selections legible at overview zoom without reading pixels."""
        layer = getattr(self.review, 'selection_overlay', None)
        if layer is None or not self.pending or not any(item is layer for item in self.viewer.layers):
            return
        matrix = mask_world_affine(layer)
        native = self.local.adapter.native
        stamp = (id(layer), len(layer.data), float(self.viewer.camera.zoom),
                 native.width(), native.height(), tuple(matrix.ravel()))
        if stamp == getattr(self, '_pending_edge_stamp', None):
            return
        # Napari Shapes widths use data units. Project two unit vectors so
        # registration, anisotropic scale, and dock resizing are accounted for.
        points = np.asarray([[0,0],[1,0],[0,1]]) @ matrix[:2,:2].T + matrix[:2,2]
        projected = self.local.adapter.canvas_points(points)
        scale = np.linalg.svd((projected[1:]-projected[0]).T, compute_uv=False)[-1]
        if not np.isfinite(scale) or scale <= 0:
            return
        layer.edge_width = 3.0 / scale
        self._pending_edge_stamp = stamp

    def clear_selection(self):
        self._tile_progress = None
        self.pending.clear()
        self._draw_pending()
        self._persist()
        self.refresh_status()

    def _targets(self):
        if self.pending:
            return list(self.pending)
        if self.review.selected_cell is None:
            raise ValueError('Move over the image or select a review tile first.')
        _, b, _, _ = self.review.selected_cell
        return [(b.y0,b.x0,b.y1,b.x1)]

    def approve(self):
        if self.canvas_mode.currentData() == 'review' and not self.pending:
            raise ValueError('Select fully checked tiles on the canvas first.')
        self._operate(1, self._targets(), next_tile=not self.pending)

    def revoke(self):
        self._operate(0, self._targets())

    def revoke_all(self):
        mask = associated_mask(self.viewer, self.image, self.truth)
        areas = effective_areas(mask.metadata.get(REVIEW_KEY, {})) if mask is not None else []
        if not areas:
            raise ValueError('No saved reviewed areas to revoke for this image/mask pair.')
        answer = QMessageBox.question(self, 'Revoke review approval',
            f'Revoke all reviewed areas for {self.image.name} / {self.truth.name}?\n'
            'Biological annotations and training history are preserved.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self._operate(0, areas)

    def _operate(self, value, rectangles, *, next_tile=False, undo_state=None, undo_batches=None):
        if self.busy or self.training.worker is not None:
            raise ValueError('Wait for the current operation to finish.')
        if active_path(self.viewer) is None:
            raise ValueError('Save the workspace once before confirming tiles, so review progress can be recovered.')
        if self.truth is None or self.image is None:
            raise ValueError('Choose the matching image and mask.')
        truth, image = self.truth, self.image
        from ._annotation_batches import ensure_pair, restore_membership
        before_batches = deepcopy(ensure_pair(self.viewer, image, truth))
        self.review.context(current_tile=True, review_bounds=rectangles[0])
        mask = associated_mask(self.viewer, self.image, self.truth)
        before = deepcopy(mask.metadata.get(REVIEW_KEY, {})) if mask is not None else {}
        self._cancelled = False
        self.busy = True
        self.message.clear()
        self._tile_progress = dict(total=len(rectangles), completed=0,
                                  action='Restoring review' if undo_state is not None else
                                  'Approving' if value else 'Revoking')
        self._update_workflow_progress()
        self.local.busy = True
        self.training.setEnabled(False)
        self.cancel_button.show()
        previous_mode = str(self.truth.mode)
        self.truth.mode = 'pan_zoom'
        self.local.follow.pause_for_operation()
        self.refresh_status()
        if self.workflow_progress.isVisible():
            self.workflow_progress.repaint()
        completed = []
        todo = list(map(tuple, rectangles))
        revision = self._revision

        def finish(error=None):
            self.busy = False
            self.local.busy = False
            self.training.setEnabled(True)
            self.cancel_button.hide()
            if completed and undo_state is None:
                self._undo = (before, completed, revision, before_batches)
            if undo_state is not None:
                self._undo = None
            if completed and undo_state is not None and not error and not self._cancelled:
                target = associated_mask(self.viewer, image, truth)
                restored_state = deepcopy(undo_state)
                if not restored_state:
                    restored_state = {key: value for key, value in target.metadata[REVIEW_KEY].items()
                                      if key not in ('areas', 'invalid_regions', 'review_cells')}
                    restored_state.update(areas=[], invalid_regions=[], review_cells=[])
                for owner in (target, full_source(target)):
                    owner.metadata[REVIEW_KEY] = deepcopy(restored_state)
                if undo_batches is not None:
                    restore_membership(self.viewer, image, truth, undo_batches)
                self.review.coverage_changed.emit()
            if not error and not self._cancelled:
                self.pending = [rect for rect in self.pending if rect not in completed]
            self._persist()
            self._draw_pending()
            if completed and value == 1 and undo_state is None and self.truth is truth and self.image is image:
                self.review.show_coverage()
            if any(layer is truth for layer in self.viewer.layers) and truth.editable:
                truth.mode = previous_mode
            self.local.follow.resume_from_review()
            self.journal.flush()
            if not self.active:
                self.review.mouse_review_check.setChecked(False)
            self.message.setText(str(error) if error else '')
            if next_tile and not error and not self._cancelled:
                self.skip()
            self._tile_progress = dict(total=len(rectangles), completed=len(completed),
                                      stopped=bool(error or self._cancelled))
            self._update_workflow_progress()
            self.refresh_sources()
            self.refresh_status()

        def step():
            if self.closed:
                return
            if self._cancelled or not todo:
                finish()
                return
            rect = todo.pop(0)
            try:
                if self.truth is not truth or self.image is not image or not any(layer is truth for layer in self.viewer.layers):
                    raise ValueError('The annotation source changed; remaining tile updates were stopped.')
                self.review.mark(value, current_tile=True, review_bounds=rect)
                if undo_state is not None:
                    for a,b,c,d in effective_areas(undo_state):
                        restored = (max(a,rect[0]),max(b,rect[1]),min(c,rect[2]),min(d,rect[3]))
                        if restored[0]<restored[2] and restored[1]<restored[3]:
                            self.review.mark(1,current_tile=True,review_bounds=restored)
                completed.append(rect)
                self._tile_progress['completed'] = len(completed)
                self._update_workflow_progress()
            except Exception as error:
                finish(error)
                return
            QTimer.singleShot(0, step)
        step()

    def undo_review(self):
        if self._undo is None or self._undo[2] != self._revision:
            raise ValueError('No review action can be undone after these annotations changed.')
        before, rectangles, _, batch_state = self._undo
        self._operate(0, rectangles, undo_state=before, undo_batches=batch_state)

    def skip(self):
        if self.truth is None:
            return
        current = self.review.selected_cell
        if current is None:
            self._current_from_view()
            current = self.review.selected_cell
        if current is None:
            return
        if not self.review.next_unreviewed():
            self.message.setText(self.review.status.text())
            return
        truth, bounds, _, _ = self.review.selected_cell
        native = self.local.adapter.native
        matrix = mask_world_affine(full_source(truth))[:2,:2]
        extent = np.abs(matrix) @ np.asarray(bounds.shape)
        self.viewer.camera.zoom = min(native.height()/max(extent[0],1e-8),
                                      native.width()/max(extent[1],1e-8)) * 0.8
        self._draw_pending()
        self.local.follow.schedule()

    def _coverage_changed(self):
        self.journal.schedule()
        self.refresh_status()

    def _batch_changed(self, event):
        from ._annotation_batches import _id
        if (getattr(event, 'action', None) in ('new', 'select') and not self.busy and
                getattr(event, 'truth_id', None) == _id(self.truth) and
                getattr(event, 'image_id', None) == _id(self.image)):
            self.pending = []
            self.c._set_pending_training_tiles([])
            self._persist()
            if self.active:
                self._draw_pending()
        self.refresh_status()

    def prepare(self):
        if self.busy:
            raise ValueError('Finish the current tile operation first.')
        self.journal.flush()
        training = self.training
        self._preparing = True
        if not training._multiple_pairs() and not training._is_multilabel():
            training.image_combo.setCurrentIndex(training.image_combo.findData(self.image))
            training.truth_combo.setCurrentIndex(training.truth_combo.findData(self.truth))
            training.annotation_combo.setCurrentIndex(training.annotation_combo.findData('partial'))
            training.workflow_combo.setCurrentIndex(training.workflow_combo.findData('coverage'))
            training._sources_changed()
        training.prepare_button.click()

    def _advanced_section(self, section):
        self.shell.show_advanced(1)
        section.setChecked(True)
        self.c._workflow_tabs.widget(0).ensureWidgetVisible(section)

    def _open_patch_selection(self):
        self.shell.show_advanced(2)
        self.training.workflow_tabs.setCurrentIndex(0)
        self.training.dataset_scroll.ensureWidgetVisible(self.training.patch_selection_group)

    def refresh_status(self):
        if self.closed:
            return
        self.project_label.setText(self.workspace.current_label.text())
        self.batch_widget.set_context(self.image, self.truth)
        if active_path(self.viewer) is None:
            self.persistence_label.setText('Save workspace once to enable tile progress recovery.')
        if not self.active:
            return
        self._update_pending_edge_width()
        shared = self.c._pending_training_tiles()
        if shared != self.pending and not self.busy:
            self.pending = shared
            self._draw_pending()
        if not self.busy and self.canvas_mode.currentData() == 'edit':
            with suppress(ValueError, RuntimeError, np.linalg.LinAlgError):
                self._current_from_view()
        self.current_label.setText(self.review.cell_label.text())
        self.progress_label.setText(self.review.progress_label.text())
        detail = self.training.status_label.text() if getattr(self, '_preparing', False) else ''
        scope = 'Configured multiple-image/channel dataset\n' if self.training._multiple_pairs() or self.training._is_multilabel() else ''
        from ._annotation_batches import preparation_error
        batch_error = (self.training.batch_preparation_error() if self.training._multiple_pairs() or self.training._is_multilabel()
                       else preparation_error(self.image, self.truth))
        self.dataset_label.setText(scope + (batch_error or self.training.dataset_state_label.text()) + ('\n' + detail if detail else ''))
        review_mode = self.canvas_mode.currentData() == 'review'
        if review_mode:
            self.follow_label.setText('Review selection active · click tiles on the image, then approve.'
                                     if self.review.mouse_review_check.isChecked() else
                                     'Review selection unavailable · ' + self.review.status.text())
        else:
            self.follow_label.setText(self.local.auto_status.text() or 'Camera follow · automatic')
        self.pending_label.setText(f'{len(self.pending)} pending tile(s) · click again to deselect' if self.pending else 'No pending tile selection')
        self.approve_button.setText(f'Approve {len(self.pending)} selected tiles' if self.pending else
                                    'Click tiles on image first' if review_mode else 'Reviewed && Next')
        available = self.truth is not None and self.image is not None and active_path(self.viewer) is not None
        busy = self.busy or self.training.worker is not None or self.c._large_controller.busy
        self.batch_widget.set_busy(busy)
        reason = ('Wait for the current operation to finish.' if busy else
                  'Choose an aligned 2D image and mask.' if self.truth is None or self.image is None else
                  'Save the workspace once to enable durable tile review.' if active_path(self.viewer) is None else '')
        for control in (self.mask_combo,self.image_combo,self.grid_combo,self.canvas_mode):
            control.setEnabled(not busy)
        for control in (self.approve_button,self.skip_button,self.coverage_button,self.prepare_button):
            control.setEnabled(available and not busy)
            control.setToolTip(reason)
        if review_mode and not self.pending:
            self.approve_button.setEnabled(False)
            self.approve_button.setToolTip(reason or 'Click fully checked tiles on the canvas to select them.')
        if batch_error:
            self.prepare_button.setEnabled(False)
            self.prepare_button.setToolTip(batch_error)
        self.save_button.setEnabled(not busy)
        self.open_button.setEnabled(not busy)
        self.open_training_button.setEnabled(not self.busy)
        self.patch_selection_button.setEnabled(not busy)
        self.shell.simple_radio.setEnabled(not self.busy)
        self.shell.advanced_radio.setEnabled(not self.busy)
        for button in self.tool_buttons:
            button.setEnabled(not busy)
        self.tool_buttons[3].setEnabled(not busy and self._undo is not None)
        editable = self.truth is not None and self.truth.editable and self.canvas_mode.currentData() == 'edit'
        self.paint_button.setEnabled(editable and not busy)
        self.erase_button.setEnabled(editable and not busy)

    def run(self, action):
        try:
            action()
        except Exception as error:
            self.message.setText(str(error))

    def shutdown(self):
        if self.closed:
            return
        self.journal.shutdown()
        self.batch_widget.shutdown()
        self.closed = True
        self.poll.stop()
        self.refresh_timer.stop()
        for emitter, callback in self._connections:
            with suppress(ValueError, RuntimeError):
                emitter.disconnect(callback)
        for connections in self._watched.values():
            for emitter, callback in connections:
                with suppress(ValueError, RuntimeError):
                    emitter.disconnect(callback)
        self._watched.clear()
