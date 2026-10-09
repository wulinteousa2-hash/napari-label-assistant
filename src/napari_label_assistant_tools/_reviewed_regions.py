"""Image-associated review coverage, written in bounded chunks."""
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4
from contextlib import suppress

import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QPushButton, QSpinBox, QVBoxLayout

KEY = "label_assistant_review"


def identity(layer):
    metadata = dict(layer.metadata)
    state = dict(metadata.get(KEY, {}))
    state.setdefault("id", str(uuid4()))
    metadata[KEY] = state
    layer.metadata = metadata
    return state["id"]


def associated_mask(viewer, image):
    if image is None:
        return None
    image_id = image.metadata.get(KEY, {}).get("id")
    if not image_id:
        return None
    return next((layer for layer in viewer.layers
                 if layer.metadata.get(KEY, {}).get("role") == "reviewed_regions"
                 and layer.metadata[KEY].get("image_id") == image_id), None)


def associate(mask, image):
    mask.metadata = {**mask.metadata, KEY: {"role": "reviewed_regions", "image_id": identity(image)}}


def write_area(mask, bounds, value):
    for y in range(bounds.y0, bounds.y1, 512):
        for x in range(bounds.x0, bounds.x1, 512):
            mask.data[y:min(y + 512, bounds.y1), x:min(x + 512, bounds.x1)] = value
    mask.refresh()
    from ._component_library import notify_library
    notify_library(mask)


class ReviewedAreaWidget(QGroupBox):
    def __init__(self, viewer, controller, parent=None):
        super().__init__("Reviewed cells", parent)
        self.viewer, self.controller = viewer, controller
        self.closed = False
        widgets = getattr(viewer, "_label_assistant_review_widgets", None)
        if widgets is None:
            widgets = []
            viewer._label_assistant_review_widgets = widgets
        widgets.append(self)
        self.selected_cell = None
        self.grid_config = None
        self.selection_overlay = None
        self.review_overlay = None
        self._previous_modes = []
        layout = QVBoxLayout(self)
        self.image_combo = QComboBox()
        self.image_combo.setToolTip("Select the image corresponding to the Curated layer being reviewed.")
        layout.addWidget(self.image_combo)
        self.mouse_review_check = QCheckBox("Select an editable/review cell on the canvas")
        self.mouse_review_check.setToolTip("Enable this mode to select or mark whole address-grid cells. Turn it off to resume painting objects.")
        self.grid_form = QFormLayout()
        layout.addLayout(self.grid_form)
        self.mouse_action_combo = QComboBox()
        self.mouse_action_combo.addItem("Click to select a cell", "select")
        self.mouse_action_combo.addItem("Click to mark a cell reviewed", "mark")
        self.mouse_action_combo.addItem("Click to remove reviewed status", "remove")
        self.mouse_action_combo.setEnabled(False)
        layout.addWidget(self.mouse_review_check)
        self.advanced_check = QCheckBox("Mark already-checked cells with clicks")
        self.advanced_check.setToolTip("Use only for cells you have already checked completely. The normal workflow selects a cell first, then finishes its review.")
        layout.addWidget(self.advanced_check)
        layout.addWidget(self.mouse_action_combo)
        self.mouse_action_combo.hide()
        self.advanced_check.toggled.connect(self._advanced_changed)
        self.cell_label = QLabel("No grid cell selected. Enable mouse review, then click a cell.")
        self.cell_label.setWordWrap(True)
        layout.addWidget(self.cell_label)
        self.mouse_review_check.toggled.connect(self.toggle_mouse_review)

        note = QLabel("Confirm every target and the background in this cell before including it.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.mark_button = QPushButton("Finish review — include this cell in training")
        self.mark_button.setToolTip("Confirms every target in the selected cell has been annotated. Remaining pixels are confirmed background.")
        self.mark_button.setEnabled(False)
        self.remove_button = QPushButton("Remove reviewed status")
        self.remove_button.setEnabled(False)
        self.preview_button = QPushButton("Show reviewed annotations in this area")
        for button, action in ((self.mark_button, self.finish_review), (self.remove_button, self.remove_review), (self.preview_button, self.preview)):
            if button is not self.preview_button:
                layout.addWidget(button)
            else:
                button.hide()
            button.clicked.connect(lambda _checked=False, fn=action: self.run(fn))
        self.status = QLabel("Use local-area editing to define the review boundary.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.image_combo.currentIndexChanged.connect(self.clear_cell)
        self.image_combo.currentIndexChanged.connect(self.update_status)
        viewer.layers.events.inserted.connect(self.refresh_images)
        viewer.layers.events.removed.connect(self.refresh_images)
        self.refresh_images()

    def _advanced_changed(self, enabled):
        self.mouse_action_combo.setVisible(enabled)
        if enabled:
            self.mouse_action_combo.setCurrentIndex(self.mouse_action_combo.findData("mark"))
            self.mouse_review_check.setChecked(True)
            if not self.mouse_review_check.isChecked():
                self.advanced_check.setChecked(False)
            else:
                self.status.setText("Click already-checked cells to include them in training. Turn this option off to resume editing.")
        else:
            self.mouse_action_combo.setCurrentIndex(self.mouse_action_combo.findData("select"))
            self.mouse_review_check.setChecked(False)

    def remove_review(self):
        if self.selected_cell is None:
            raise ValueError("Select the cell whose reviewed status you want to remove.")
        self.mark(0)

    def finish_review(self):
        if self.selected_cell is None:
            raise ValueError("Select a grid cell on the canvas first, then check all targets inside its yellow boundary.")
        self.mark(1)

    def configure_grid(self, source, height, width, show):
        self.grid_config = (source, height, width, show)
        for label, original in (("Review-cell height (px)", height), ("Review-cell width (px)", width)):
            spin = QSpinBox()
            spin.setRange(original.minimum(), original.maximum())
            spin.setValue(original.value())
            spin.setToolTip("Uses the same cell size as the address grid in Review/QC. Changing size clears the current selection; existing reviewed coverage is preserved.")
            spin.valueChanged.connect(original.setValue)
            original.valueChanged.connect(spin.setValue)
            self.grid_form.addRow(label, spin)
        height.valueChanged.connect(self.clear_cell)
        width.valueChanged.connect(self.clear_cell)
        show.toggled.connect(lambda checked: self.mouse_review_check.setChecked(False) if not checked else None)
        self.viewer.mouse_drag_callbacks.append(self.mouse_click)

    def clear_cell(self, *_args):
        self.selected_cell = None
        self.mark_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.cell_label.setText("No grid cell selected. Enable mouse review, then click a cell.")
        if self.selection_overlay is not None:
            self.selection_overlay.visible = False
        self.mark_button.setText("Finish review — include this cell in training")
        self.remove_button.setText("Remove reviewed status")

    def toggle_mouse_review(self, enabled):
        self.mouse_action_combo.setEnabled(enabled)
        if enabled:
            if self.grid_config is None:
                self.mouse_review_check.setChecked(False)
                return
            source, _, _, show = self.grid_config
            truth = source()
            if truth is None:
                self.status.setText("Select the Curated Labels layer first.")
                self.mouse_review_check.setChecked(False)
                return
            show.setChecked(True)
            if not show.isChecked():
                self.status.setText("The address grid could not be displayed. Increase its cell size, then enable mouse review again.")
                self.mouse_review_check.setChecked(False)
                return
            for layer in (truth, self.controller.tile):
                if layer is not None and not any(layer is previous for previous, _ in self._previous_modes):
                    self._previous_modes.append((layer, layer.mode))
                    layer.mode = "pan_zoom"
            self.viewer.layers.selection.active = self.make_outline(truth, [[self.selected_cell[1].y0, self.selected_cell[1].x0, self.selected_cell[1].y1, self.selected_cell[1].x1]] if self.selected_cell is not None and self.selected_cell[0] is truth else [], "Selected review cell", "yellow", "selection_overlay")
            self.status.setText("Click a grid cell to select it. Check every object before marking the cell reviewed.")
            self.update_review_overlay(truth)
        else:
            if self.advanced_check.isChecked():
                self.advanced_check.setChecked(False)
            for layer, mode in self._previous_modes:
                if any(layer is candidate for candidate in self.viewer.layers):
                    layer.mode = mode
            self._previous_modes.clear()
            if self.selection_overlay is not None and self.selected_cell is None:
                self.selection_overlay.visible = False
            if self.grid_config is not None:
                truth = self.grid_config[0]()
                if truth is not None:
                    tile = self.controller.tile
                    self.viewer.layers.selection.active = tile if self.controller.source is truth and tile is not None and any(tile is layer for layer in self.viewer.layers) else truth

    def make_outline(self, truth, rectangles, name, color, attribute):
        from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY
        data = [np.asarray([[y0,x0], [y0,x1], [y1,x1], [y1,x0]], dtype=float) for y0,x0,y1,x1 in rectangles]
        layer = getattr(self, attribute)
        if layer is None or not any(layer is candidate for candidate in self.viewer.layers):
            layer = self.viewer.add_shapes(data, shape_type="rectangle", name=name,
                face_color="transparent", edge_color=color, edge_width=2,
                scale=truth.scale, translate=truth.translate, rotate=truth.rotate,
                shear=truth.shear, affine=truth.affine,
                metadata={INTERNAL_LAYER_ROLE_KEY: "review_cells"})
            layer.editable = False
            setattr(self, attribute, layer)
        else:
            layer.data = data
            for attribute_name in ("scale", "translate", "rotate", "shear", "affine"):
                setattr(layer, attribute_name, getattr(truth, attribute_name))
            layer.visible = True
        return layer

    def update_review_overlay(self, truth):
        mask = associated_mask(self.viewer, self.image_combo.currentData())
        areas = mask.metadata.get(KEY, {}).get("areas", []) if mask is not None else []
        self.make_outline(truth, areas, "Reviewed areas — green outlines", "lime", "review_overlay")

    def mouse_click(self, _viewer, event):
        if not self.mouse_review_check.isChecked() or self.grid_config is None:
            return
        if str(getattr(event, "type", "")).lower() != "mouse_press" or getattr(event, "button", 1) != 1:
            return
        source, height, width, show = self.grid_config
        truth = source()
        if truth is None or not show.isChecked():
            return
        try:
            from ._dynamic_edit import TileBounds
            position = truth.world_to_data(event.position)
            y, x = np.floor(position[-2:]).astype(int)
            h, w = truth.data.shape
            if not (0 <= y < h and 0 <= x < w):
                return
            sy, sx = height.value(), width.value()
            row, col = y // sy, x // sx
            bounds = TileBounds(row*sy, col*sx, min((row+1)*sy,h), min((col+1)*sx,w))
            self.selected_cell = (truth, bounds, sy, sx)
            self.mark_button.setEnabled(True)
            self.remove_button.setEnabled(True)
            self.cell_label.setText(f"Selected R{row:02d}C{col:02d} · {bounds.shape[1]} × {bounds.shape[0]} px")
            self.mark_button.setText("Finish review — include this cell in training")
            self.remove_button.setText("Remove reviewed status")
            layer = self.make_outline(truth, [[bounds.y0,bounds.x0,bounds.y1,bounds.x1]], "Selected review cell", "yellow", "selection_overlay")
            self.viewer.layers.selection.active = layer
            action = self.mouse_action_combo.currentData()
            if action != "select":
                self.mark(1 if action == "mark" else 0)
        except Exception as error:
            self.status.setText(str(error))

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        with suppress(ValueError):
            self.viewer._label_assistant_review_widgets.remove(self)
        self.mouse_review_check.setChecked(False)
        with suppress(ValueError):
            self.viewer.mouse_drag_callbacks.remove(self.mouse_click)
        for emitter in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed):
            with suppress(Exception):
                emitter.disconnect(self.refresh_images)

    def run(self, action):
        try:
            action()
        except Exception as error:
            self.status.setText(str(error))

    def refresh_images(self, *_args):
        import napari
        from ._dynamic_edit import is_internal_layer
        current = self.image_combo.currentData()
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image) and not layer.rgb and not is_internal_layer(layer):
                self.image_combo.addItem(layer.name, layer)
        for i in range(self.image_combo.count()):
            if self.image_combo.itemData(i) is current:
                self.image_combo.setCurrentIndex(i)
        self.image_combo.blockSignals(False)
        self.update_status()

    def update_status(self, *_args):
        mask = associated_mask(self.viewer, self.image_combo.currentData())
        if mask is not None:
            count = len(mask.metadata[KEY].get("areas", []))
            self.status.setText(f"{count} marked area(s) · Training selects this mask in Partial mode. Preview patches again after changes.")
        else:
            self.status.setText("No reviewed coverage yet. Marking an area creates a separate mask.")

    def context(self):
        from ._quick_train import array_2d
        c = self.controller
        image = self.image_combo.currentData()
        if image is None or not any(image is layer for layer in self.viewer.layers):
            raise ValueError("Select the corresponding image.")
        truth, bounds = c.source, c.bounds
        if self.selected_cell is not None:
            truth, bounds, sy, sx = self.selected_cell
            source, height, width, _ = self.grid_config
            if source() is not truth or (sy, sx) != (height.value(), width.value()):
                self.clear_cell()
                raise ValueError("The source or grid changed. Click a cell again.")
        elif c.bounds is None or c.tile is None or c.source is None or c.loading:
            raise ValueError("Enable mouse review and click a grid cell, or load a local editable area first.")
        if c.stroke_active:
            raise ValueError("Finish the current stroke first.")
        if truth.metadata.get(KEY, {}).get("role") in ("reviewed_regions", "reviewed_annotations"):
            raise ValueError("Select the Curated object layer, rather than the review-coverage mask.")
        tile_editable = c.source is truth and c.tile is not None and c.tile.editable
        if not truth.editable and not tile_editable:
            raise ValueError("Wait for training or indexing to finish before changing review coverage.")
        if tuple(array_2d(image.data).shape) != tuple(truth.data.shape):
            raise ValueError("The selected image and Curated layer must have matching dimensions.")
        for attribute in ("scale", "translate", "rotate", "shear", "affine"):
            a, b = getattr(image, attribute), getattr(truth, attribute)
            a, b = getattr(a, "affine_matrix", a), getattr(b, "affine_matrix", b)
            if not np.allclose(a, b):
                raise ValueError("The selected image and Curated layer must have matching spatial transforms.")
        if c.source is truth and c.dirty and not c.commit():
            raise ValueError("Could not apply pending edits. Review status was not changed.")
        return image, truth, bounds

    def mark(self, value):
        from .workspace.service import _require_zarr
        image, truth, bounds = self.context()
        mask = associated_mask(self.viewer, image)
        if mask is None:
            if not value:
                raise ValueError("This image has no reviewed-region mask.")
            path, _ = QFileDialog.getSaveFileName(self, "Store Reviewed Areas", "reviewed-regions.zarr", "Zarr mask (*.zarr)")
            if not path:
                return
            if not path.endswith(".zarr"):
                path += ".zarr"
            data = _require_zarr().open_array(path, mode="w-", shape=truth.data.shape,
                                             chunks=(512, 512), dtype="uint8", fill_value=0)
            mask = self.viewer.add_labels(data, name="Reviewed regions", scale=image.scale,
                                          translate=image.translate, rotate=image.rotate,
                                          shear=image.shear, affine=image.affine, opacity=.15,
                                          metadata={KEY: {"role": "reviewed_regions", "image_id": identity(image), "truth_id": identity(truth)}})
        state = dict(mask.metadata[KEY])
        truth_id = identity(truth)
        if state.get("truth_id", truth_id) != truth_id:
            raise ValueError("This mask belongs to another ground-truth layer. Select that layer before reviewing.")
        if tuple(mask.data.shape) != tuple(truth.data.shape) or not mask.editable:
            raise ValueError("Reviewed mask must match the source dimensions and be available for editing.")
        rect = [bounds.y0, bounds.x0, bounds.y1, bounds.x1]
        areas = list(state.get("areas", []))
        if value:
            if rect not in areas:
                areas.append(rect)
        else:
            # Subtract the rectangle, retaining the reviewed parts of overlapping areas.
            remaining = []
            y0, x0, y1, x1 = rect
            for a, b, d, e in areas:
                u, v, w, z = max(a, y0), max(b, x0), min(d, y1), min(e, x1)
                if u >= w or v >= z:
                    remaining.append([a, b, d, e])
                else:
                    remaining.extend(r for r in ([a,b,u,e], [w,b,d,e], [u,b,w,v], [u,z,w,e]) if r[0] < r[2] and r[1] < r[3])
            areas = remaining
        self.update_annotations(image, truth, mask, bounds, value)
        write_area(mask, bounds, value)
        from ._component_library import subtract_rectangle
        review_cells = list(state.get("review_cells", state.get("areas", [])))
        if value:
            if rect not in review_cells:
                review_cells.append(rect)
        else:
            review_cells = [part for cell in review_cells for part in subtract_rectangle(cell, rect)]
        invalid_regions = list(state.get("invalid_regions", []))
        if value:
            invalid_regions = [
                piece for invalid in invalid_regions
                for piece in subtract_rectangle(invalid, rect)
            ]
        mask.metadata = {**mask.metadata, KEY: {**state, "truth_id": truth_id,
                         "areas": areas, "invalid_regions": invalid_regions, "review_cells": review_cells}}
        from ._component_library import notify_library
        notify_library(mask)
        self.update_status()
        if self.grid_config is not None:
            self.update_review_overlay(truth)

    def update_annotations(self, image, truth, mask, bounds, value):
        """Merge trusted foreground into a binary disk-backed annotation layer."""
        from .workspace.service import _require_zarr
        image_id, truth_id = identity(image), identity(truth)
        annotations = next((layer for layer in self.viewer.layers
            if layer.metadata.get(KEY, {}).get("role") == "reviewed_annotations"
            and layer.metadata[KEY].get("image_id") == image_id
            and layer.metadata[KEY].get("truth_id") == truth_id), None)
        if annotations is None:
            if not value:
                return
            root = getattr(getattr(mask.data, "store", None), "root", None)
            folder = Path(root).parent if root else Path(mkdtemp(prefix="label-assistant-reviewed-"))
            path = folder / f"reviewed-annotations-{uuid4().hex[:8]}.zarr"
            data = _require_zarr().open_array(str(path), mode="w-", shape=truth.data.shape,
                                            chunks=(512,512), dtype="uint8", fill_value=0)
            annotations = self.viewer.add_labels(data, name="Reviewed annotations",
                scale=truth.scale, translate=truth.translate, rotate=truth.rotate,
                shear=truth.shear, affine=truth.affine,
                metadata={KEY: {"role": "reviewed_annotations", "image_id": image_id, "truth_id": truth_id}})
            annotations.editable = False
        for y in range(bounds.y0, bounds.y1, 512):
            for x in range(bounds.x0, bounds.x1, 512):
                slices = (slice(y,min(y+512,bounds.y1)), slice(x,min(x+512,bounds.x1)))
                annotations.data[slices] = (np.asarray(truth.data[slices]) > 0).astype("uint8") if value else 0
        annotations.refresh()
        from ._component_library import notify_library
        notify_library(annotations)

    def preview(self):
        from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY
        image, truth, bounds = self.context()
        mask = associated_mask(self.viewer, image)
        if mask is None:
            raise ValueError("Mark an area as reviewed first.")
        if bounds.shape[0] * bounds.shape[1] > 16_777_216:
            raise ValueError("This cell is too large for a local preview. Select a smaller cell; reviewed annotations remain available in their disk-backed layer.")
        data = np.asarray(truth.data[bounds.slices]).copy()
        data[np.asarray(mask.data[bounds.slices]) == 0] = 0
        old = getattr(self, "preview_layer", None)
        if old is not None and any(old is layer for layer in self.viewer.layers):
            self.viewer.layers.remove(old)
        translate = np.asarray(truth.translate).copy()
        translate += truth._transforms["data2physical"].linear_matrix @ np.asarray([bounds.y0, bounds.x0])
        self.preview_layer = self.viewer.add_labels(data, name="Reviewed annotations — current area",
            scale=truth.scale, translate=translate, rotate=truth.rotate, shear=truth.shear,
            affine=truth.affine, metadata={INTERNAL_LAYER_ROLE_KEY: "reviewed_preview"})
        self.preview_layer.editable = False
        self.status.setText("Read-only snapshot of Curated object IDs in reviewed coverage in this area. Refresh this preview after edits; original annotations are preserved.")
