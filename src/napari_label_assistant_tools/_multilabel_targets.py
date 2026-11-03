"""Channel-to-mask assignments live in a dialog, not the compact Dataset page."""
from copy import deepcopy

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                           QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
                           QWidget, QLineEdit, QPushButton, QHBoxLayout)

from ._dataset_pairs import source_layers, is_prediction
from ._model_project import layer_id
from ._quick_train import array_2d


def suggest_channels(viewer, names, definitions=()):
    images, labels, masks = source_layers(viewer)
    result = deepcopy(list(definitions))
    existing = {row['image_id']: row for row in result}
    for image in images:
        identifier = layer_id(image)
        if identifier in existing:
            row = existing[identifier]
        else:
            row = dict(image_id=identifier, image_name=image.name, included=False,
                       split_group=image.name, channels={})
            result.append(row)
        for name in names:
            if name in row['channels']:
                continue
            candidates = [label for label in labels if (
                label.metadata.get('quick_train', {}).get('source_id') == identifier and
                label.metadata.get('quick_train', {}).get('channel_name') == name) or
                label.name.casefold() in {f'{image.name} — {name}'.casefold(), f'{image.name} {name}'.casefold()} or
                (len(images) == 1 and label.name.casefold() == name.casefold())]
            target = candidates[0] if len(candidates) == 1 else None
            mask = next((mask for mask in masks if target is not None and
                         mask.metadata.get('label_assistant_review', {}).get('image_id') == identifier and
                         mask.metadata.get('label_assistant_review', {}).get('truth_id') == layer_id(target)), None)
            row['channels'][name] = dict(target_id=layer_id(target) if target is not None else None,
                                          reviewed_id=layer_id(mask) if mask is not None else None,
                                          coverage='partial' if mask is not None else 'complete', confirmed=False)
    return result


def resolve_channels(viewer, names, definitions):
    images, labels, masks = source_layers(viewer)
    by_id = {layer_id(layer): layer for layer in images + labels + masks}
    resolved, seen = [], set()
    for row in definitions:
        if not row.get('included'):
            continue
        image = by_id.get(row.get('image_id'))
        if image is None or not any(image is item for item in images):
            raise ValueError(f"Missing image {row.get('image_name', '')}. Repair it in Manage channels.")
        if row['image_id'] in seen:
            raise ValueError('Each image can appear only once in a channel dataset.')
        seen.add(row['image_id'])
        shape = tuple(array_2d(image.data).shape)
        targets, used = [], set()
        for name in names:
            assignment = row.get('channels', {}).get(name, {})
            coverage = assignment.get('coverage')
            if coverage == 'unknown':
                targets.append(None)
                continue
            if coverage not in ('partial', 'complete'):
                raise ValueError(f'{image.name}: configure coverage for channel {name!r}.')
            if assignment.get('foreground_mode', 'binary') not in ('binary', 'nonzero'):
                raise ValueError(f'{image.name}: choose how foreground pixels are read for {name!r}.')
            target = by_id.get(assignment.get('target_id'))
            if target is None or not any(target is item for item in labels):
                raise ValueError(f'{image.name}: assign a target mask for channel {name!r}, or explicitly choose Unknown.')
            if assignment['target_id'] in used:
                raise ValueError(f'{image.name}: assign a distinct mask layer to each output channel.')
            used.add(assignment['target_id'])
            if is_prediction(target) and not assignment.get('confirmed'):
                raise ValueError(f'{image.name}: confirm reviewed/corrected prediction for channel {name!r}.')
            mask = by_id.get(assignment.get('reviewed_id')) if coverage == 'partial' else None
            if coverage == 'partial':
                if mask is None or not any(mask is item for item in masks):
                    raise ValueError(f'{image.name}: choose reviewed coverage for channel {name!r}.')
                state = mask.metadata.get('label_assistant_review', {})
                if state.get('image_id') != row['image_id'] or state.get('truth_id') != assignment['target_id']:
                    raise ValueError(f'{image.name}: reviewed coverage for {name!r} belongs to a different target.')
            for layer in [target] + ([mask] if mask is not None else []):
                if tuple(array_2d(layer.data).shape) != shape or not np.allclose(
                    [image.data_to_world(p) for p in ((0, 0), (1, 0), (0, 1))],
                    [layer.data_to_world(p) for p in ((0, 0), (1, 0), (0, 1))]):
                    raise ValueError(f'{image.name}: channel {name!r} masks must match the image shape and transform.')
            targets.append((assignment, target, mask))
        if not any(target is not None for target in targets):
            raise ValueError(f'{image.name}: all channels are unknown; assign supervision or exclude this image.')
        resolved.append((row, image, targets))
    if not resolved:
        raise ValueError('Include at least one image and assign its named masks in Manage channels.')
    return resolved


class _ChannelAssignmentsTable(QWidget):
    def __init__(self, viewer, names, definitions, parent=None):
        super().__init__(parent)
        self.names = tuple(names)
        self.definitions = suggest_channels(viewer, names, definitions)
        images, labels, masks = source_layers(viewer)
        by_id = {layer_id(layer): layer for layer in images}
        message = QLabel('Check Include image, assign each channel its mask, then choose coverage. Complete means all positives AND negatives for this channel are checked. Partial uses a reviewed mask; Unknown ignores missing annotations. Any nonzero ID reads instance labels as foreground without changing them. Related images should share a split group.')
        message.setWordWrap(True)
        self.table = QTableWidget(len(self.definitions) * len(names), 9)
        self.table.setHorizontalHeaderLabels(['Include image', 'Image', 'Channel', 'Target mask', 'Coverage', 'Reviewed mask', 'Prediction reviewed', 'Specimen / split group', 'Foreground pixels'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().moveSection(8, 4)
        # Qt does not size columns from the text in row-spanning image cells.
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.setColumnWidth(1, 180)
        for image_index, row in enumerate(self.definitions if names else []):
            start = image_index * len(names)
            include = QCheckBox()
            include.setChecked(row.get('included', False))
            self.table.setCellWidget(start, 0, include)
            image = by_id.get(row['image_id'])
            self.table.setItem(start, 1, QTableWidgetItem(image.name if image is not None else row.get('image_name', '') + ' (missing)'))
            self.table.item(start, 1).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(start, 7, QTableWidgetItem(row.get('split_group') or row['image_id']))
            if len(names) > 1:
                for column in (0, 1, 7):
                    self.table.setSpan(start, column, len(names), 1)
            for channel_index, name in enumerate(names):
                index = start + channel_index
                assignment = row['channels'][name]
                item = QTableWidgetItem(name)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(index, 2, item)
                for column, choices, key in ((3, labels, 'target_id'), (5, masks, 'reviewed_id')):
                    combo = QComboBox()
                    combo.addItem('Choose mask…' if column == 3 else 'None', None)
                    for layer in choices:
                        combo.addItem(layer.name, layer_id(layer))
                    identifier = assignment.get(key)
                    if identifier and combo.findData(identifier) < 0:
                        combo.addItem('Missing layer — choose replacement', identifier)
                    combo.setCurrentIndex(max(0, combo.findData(identifier)))
                    self.table.setCellWidget(index, column, combo)
                coverage = QComboBox()
                coverage.addItem('Complete — this channel', 'complete')
                coverage.addItem('Partial — reviewed regions', 'partial')
                coverage.addItem('Unknown / not annotated', 'unknown')
                coverage.setCurrentIndex(max(0, coverage.findData(assignment.get('coverage'))))
                self.table.setCellWidget(index, 4, coverage)
                confirmed = QCheckBox('Confirmed')
                confirmed.setChecked(assignment.get('confirmed', False))
                self.table.setCellWidget(index, 6, confirmed)
                foreground = QComboBox()
                foreground.addItem('Binary 0/1', 'binary')
                foreground.addItem('Any nonzero ID', 'nonzero')
                foreground.setCurrentIndex(max(0, foreground.findData(assignment.get('foreground_mode', 'binary'))))
                self.table.setCellWidget(index, 8, foreground)
                self.table.cellWidget(index, 3).currentIndexChanged.connect(lambda _index, control=confirmed: control.setChecked(False))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(message)
        layout.addWidget(self.table)

    def assignments(self):
        result = deepcopy(self.definitions)
        if not self.names:
            return result
        for image_index, row in enumerate(result):
            start = image_index * len(self.names)
            row['included'] = self.table.cellWidget(start, 0).isChecked()
            row['split_group'] = self.table.item(start, 7).text().strip() or row['image_id']
            row['channels'] = {name: dict(target_id=self.table.cellWidget(start + c, 3).currentData(),
                                         coverage=self.table.cellWidget(start + c, 4).currentData(),
                                         reviewed_id=self.table.cellWidget(start + c, 5).currentData(),
                                         foreground_mode=self.table.cellWidget(start + c, 8).currentData(),
                                         confirmed=self.table.cellWidget(start + c, 6).isChecked())
                               for c, name in enumerate(self.names)}
        return result


class MultilabelTargetsDialog(QDialog):
    def __init__(self, viewer, names, definitions, parent=None):
        super().__init__(parent)
        self.viewer, self.names = viewer, tuple(names)
        self.setWindowTitle('Name channels and assign masks')
        self.resize(1450, 620)
        self.names_edit = QLineEdit(', '.join(names))
        self.names_edit.setPlaceholderText('Type feature names, e.g. cell, mitochondria — or use your mask layer names')
        update = QPushButton('Update channels')
        update.clicked.connect(self._update_channels)
        self.use_layers_button = QPushButton('Use mask layers as channels')
        self.use_layers_button.setToolTip('For one loaded image: name channels after its mask layers, assign them, and read any nonzero IDs as foreground. Original masks are not changed.')
        self.use_layers_button.setEnabled(len(source_layers(viewer)[0]) == 1)
        self.use_layers_button.clicked.connect(self._use_layers)
        row = QHBoxLayout()
        row.addWidget(QLabel('Channel names'))
        row.addWidget(self.names_edit, 1)
        row.addWidget(update)
        row.addWidget(self.use_layers_button)
        self.message = QLabel('Name each feature here. Names describe the output channels; mask pixel values provide supervision. You do not need to rename layers or return to Models.')
        self.message.setWordWrap(True)
        self._editor = _ChannelAssignmentsTable(viewer, names, definitions, self)
        self.layout_box = QVBoxLayout(self)
        self.layout_box.addLayout(row)
        self.layout_box.addWidget(self.message)
        self.layout_box.addWidget(self._editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        self.layout_box.addWidget(buttons)

    @property
    def table(self):
        return self._editor.table

    def assignments(self):
        return self._editor.assignments()

    def _replace_editor(self, names, definitions):
        old = self._editor
        self.names = tuple(names)
        self._editor = _ChannelAssignmentsTable(self.viewer, self.names, definitions, self)
        self.layout_box.replaceWidget(old, self._editor)
        old.hide()
        old.deleteLater()

    def _update_channels(self):
        from ._quick_train import QuickTrainConfig
        names = tuple(part.strip() for part in self.names_edit.text().split(','))
        try:
            QuickTrainConfig(class_names=names, multilabel=True).validate()
        except ValueError as error:
            self.message.setText(str(error))
            return False
        definitions = self.assignments()
        if len(names) == len(self.names):
            for row in definitions:
                for old, new in zip(self.names, names):
                    if new not in row['channels'] and old in row['channels']:
                        row['channels'][new] = deepcopy(row['channels'][old])
        self._replace_editor(names, definitions)
        self.message.setText('Channels updated. Assign masks and coverage, check Include image, then Apply.')
        return True

    def _use_layers(self):
        images, labels, _ = source_layers(self.viewer)
        if len(images) != 1 or not labels:
            self.message.setText('This shortcut needs one image and at least one target-mask layer. Otherwise type shared channel names and assign masks manually.')
            return
        names = tuple(layer.name.strip() for layer in labels)
        if any(',' in name for name in names):
            self.message.setText('A layer name contains a comma. Type simple channel names and assign masks manually.')
            return
        from ._quick_train import QuickTrainConfig
        try:
            QuickTrainConfig(class_names=names, multilabel=True).validate()
        except ValueError as error:
            self.message.setText(str(error))
            return
        self.names_edit.setText(', '.join(names))
        definitions = suggest_channels(self.viewer, names, self.assignments())
        row = next(row for row in definitions if row['image_id'] == layer_id(images[0]))
        row['included'] = True
        for name, layer in zip(names, labels):
            assignment = row['channels'][name]
            if assignment.get('target_id') != layer_id(layer):
                assignment['confirmed'] = False
            assignment.update(target_id=layer_id(layer), foreground_mode='nonzero')
        self._replace_editor(names, definitions)
        self.message.setText('Mask layers assigned as channels; any nonzero ID is foreground during training. Original masks stay unchanged. Check Complete/Partial/Unknown coverage for each feature before Apply.')

    def _apply(self):
        typed = tuple(part.strip() for part in self.names_edit.text().split(','))
        if typed != self.names or not self.names:
            if not self._update_channels():
                return
        self.accept()
