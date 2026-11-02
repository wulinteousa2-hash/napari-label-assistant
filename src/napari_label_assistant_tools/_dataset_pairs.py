"""Workspace-stable image/label pair definitions and an off-panel editor."""
from copy import deepcopy

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                           QHeaderView, QLabel, QPushButton, QTableWidget,
                           QTableWidgetItem, QVBoxLayout)

from ._dynamic_edit import is_internal_layer
from ._model_project import layer_id
from ._quick_train import array_2d


def source_layers(viewer):
    images, labels, masks = [], [], []
    for layer in viewer.layers:
        if is_internal_layer(layer):
            continue
        role = layer.metadata.get('label_assistant_review', {}).get('role')
        if isinstance(layer, napari.layers.Image) and not layer.rgb:
            images.append(layer)
        elif isinstance(layer, napari.layers.Labels):
            if role == 'reviewed_regions':
                masks.append(layer)
            elif role != 'reviewed_annotations':
                labels.append(layer)
    return images, labels, masks


def is_prediction(layer):
    return bool(layer.metadata.get('quick_train', {}).get('experimental')) or layer.name.startswith(('Model Training prediction', 'Model Training preview'))


def suggest_pairs(viewer, definitions=()):
    """Reuse IDs and only suggest labels from an association or unique name."""
    images, labels, masks = source_layers(viewer)
    result = deepcopy(list(definitions))
    known = {row['image_id'] for row in result}
    for image in images:
        identifier = layer_id(image)
        if identifier in known:
            continue
        associated = [mask for mask in masks if mask.metadata.get('label_assistant_review', {}).get('image_id') == identifier]
        truth_ids = {mask.metadata['label_assistant_review'].get('truth_id') for mask in associated}
        truth_ids.add(image.metadata.get('label_assistant_annotation_id'))
        candidates = [label for label in labels if layer_id(label) in truth_ids]
        if not candidates:
            candidates = [label for label in labels if label.metadata.get('quick_train', {}).get('source_id') == identifier or label.metadata.get('quick_train', {}).get('source') == image.name or label.name in (
                f'Model Training prediction — {image.name}', f'Model Training preview — {image.name}', f'{image.name} labels')]
        if not candidates:
            # Layer adjacency is a suggestion only; no pair is included silently.
            index = next(i for i, item in enumerate(viewer.layers) if item is image)
            following = []
            for layer in list(viewer.layers)[index + 1:]:
                if isinstance(layer, napari.layers.Image):
                    break
                if any(layer is label for label in labels):
                    following.append(layer)
            candidates = following
        truth = candidates[0] if len(candidates) == 1 else None
        mask = next((mask for mask in associated if truth is not None and mask.metadata['label_assistant_review'].get('truth_id') == layer_id(truth)), None)
        result.append(dict(image_id=identifier, image_name=image.name,
                           truth_id=layer_id(truth) if truth is not None else None,
                           reviewed_id=layer_id(mask) if mask is not None else None,
                           coverage='partial' if mask is not None else 'complete',
                           split_group=image.name, included=False,
                           confirmed=False))
    return result


def resolve_pairs(viewer, definitions, *, minimum=2):
    """Validate explicit pairs without reading full arrays or changing pixels."""
    images, labels, masks = source_layers(viewer)
    all_layers = {layer_id(layer): layer for layer in images + labels + masks}
    result, seen = [], set()
    for row in definitions:
        if not row.get('included'):
            continue
        image, truth = (all_layers.get(row.get(key)) for key in ('image_id', 'truth_id'))
        if image is None or truth is None or not isinstance(image, napari.layers.Image) or not any(truth is item for item in labels):
            raise ValueError(f"Missing image or ground truth for {row.get('image_name', 'pair')}. Open Manage pairs to repair it.")
        if row['image_id'] in seen:
            raise ValueError('Each image can appear only once in a dataset.')
        seen.add(row['image_id'])
        if is_prediction(truth) and not row.get('confirmed'):
            raise ValueError(f"Confirm Labels reviewed for {image.name} after checking/correcting its model prediction.")
        if row.get('coverage') not in ('partial', 'complete'):
            raise ValueError(f"Choose annotation coverage for {image.name}.")
        mask = all_layers.get(row.get('reviewed_id')) if row['coverage'] == 'partial' else None
        if row['coverage'] == 'partial' and (mask is None or not any(mask is item for item in masks)):
            raise ValueError(f"Choose an associated reviewed-region mask for {image.name}.")
        shape = tuple(array_2d(image.data).shape)
        import numpy as np
        for layer in [truth] + ([mask] if mask is not None else []):
            if tuple(array_2d(layer.data).shape) != shape:
                raise ValueError(f"{image.name}: image, labels, and reviewed mask must have matching shapes.")
            if not np.allclose([image.data_to_world(p) for p in ((0, 0), (1, 0), (0, 1))],
                               [layer.data_to_world(p) for p in ((0, 0), (1, 0), (0, 1))]):
                raise ValueError(f"{image.name}: image and annotation transforms must align.")
        if mask is not None:
            state = mask.metadata.get('label_assistant_review', {})
            if state.get('image_id') != row['image_id'] or state.get('truth_id') != row['truth_id']:
                raise ValueError(f"{image.name}: the reviewed mask belongs to a different image/label pair.")
        result.append((row, image, truth, mask))
    if len(result) < minimum:
        raise ValueError('Include at least two image–label pairs, or select Current pair.')
    return result


class DatasetPairsDialog(QDialog):
    def __init__(self, viewer, definitions, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.review_pair = None
        self.setWindowTitle('Manage dataset pairs')
        self.resize(1200, 520)
        self.definitions = suggest_pairs(viewer, definitions)
        images, labels, masks = source_layers(viewer)
        by_id = {layer_id(layer): layer for layer in images + labels + masks}
        self.message = QLabel('Include image–label pairs to train one model. Coverage applies per pair. Confirm reviewed predictions before using them as ground truth. Give related images the same split group so they stay together in validation. Pixel scale should be consistent across images.')
        self.message.setWordWrap(True)
        self.table = QTableWidget(len(self.definitions), 7)
        self.table.setHorizontalHeaderLabels(['Include', 'Image', 'Ground truth', 'Coverage', 'Reviewed mask', 'Specimen / split group', 'Labels reviewed'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for index, row in enumerate(self.definitions):
            include = QCheckBox()
            include.setChecked(row.get('included', False))
            self.table.setCellWidget(index, 0, include)
            image = by_id.get(row['image_id'])
            item = QTableWidgetItem(image.name if image is not None else row.get('image_name', 'Missing image') + ' (missing)')
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(index, 1, item)
            for column, choices, key, placeholder in ((2, labels, 'truth_id', 'Choose labels…'), (4, masks, 'reviewed_id', 'None')):
                combo = QComboBox()
                combo.addItem(placeholder, None)
                for layer in choices:
                    combo.addItem(layer.name, layer_id(layer))
                value = row.get(key)
                if value and combo.findData(value) < 0:
                    combo.addItem('Missing layer — choose a replacement', value)
                combo.setCurrentIndex(max(0, combo.findData(value)))
                self.table.setCellWidget(index, column, combo)
            coverage = QComboBox()
            coverage.addItem('Complete — whole image', 'complete')
            coverage.addItem('Partial — reviewed regions', 'partial')
            coverage.setCurrentIndex(max(0, coverage.findData(row.get('coverage'))))
            self.table.setCellWidget(index, 3, coverage)
            self.table.setItem(index, 5, QTableWidgetItem(row.get('split_group') or row['image_id']))
            confirmed = QCheckBox('Confirmed')
            confirmed.setChecked(row.get('confirmed', False))
            self.table.setCellWidget(index, 6, confirmed)
            # A replacement label must be reviewed independently.
            self.table.cellWidget(index, 2).currentIndexChanged.connect(lambda _index, control=confirmed: control.setChecked(False))
        review = QPushButton('Review selected pair')
        review.clicked.connect(self._review)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message)
        layout.addWidget(self.table)
        layout.addWidget(review)
        layout.addWidget(buttons)

    def pairs(self):
        return [{**row, 'included': self.table.cellWidget(i, 0).isChecked(),
                 'truth_id': self.table.cellWidget(i, 2).currentData(),
                 'coverage': self.table.cellWidget(i, 3).currentData(),
                 'reviewed_id': self.table.cellWidget(i, 4).currentData(),
                 'split_group': self.table.item(i, 5).text().strip() or row['image_id'],
                 'confirmed': self.table.cellWidget(i, 6).isChecked()}
                for i, row in enumerate(self.definitions)]

    def _apply(self):
        # Keep incomplete definitions editable; preparation reports what needs repair.
        self.accept()

    def _review(self):
        index = self.table.currentRow()
        if index < 0:
            self.message.setText('Select an image row to review its annotations.')
            return
        self.review_pair = self.pairs()[index]
        self.accept()
