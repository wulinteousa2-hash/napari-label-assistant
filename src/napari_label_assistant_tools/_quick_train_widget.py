"""Simple in-napari training and local mask preview workflow."""

from __future__ import annotations

from contextlib import suppress
from threading import Event
from types import GeneratorType

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from superqt.utils import create_worker

from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY, is_internal_layer
from ._quick_train import (
    QuickTrainCancelled, QuickTrainConfig, array_2d, build_crop_atlas, crop_summary,
    load_model, predict_area, prepare_crops, save_model, train_prepared_unet,
)


class QuickTrainWorkerError(Exception):
    """Report runtime failures without superqt mistaking them for deleted workers."""


def _run_task(function, args, kwargs):
    try:
        result = function(*args, **kwargs)
        if isinstance(result, GeneratorType):
            return (yield from result)
        return result
    except QuickTrainCancelled:
        raise
    except Exception as error:
        # GeneratorWorker treats RuntimeError as a deleted Qt worker and skips
        # both errored and finished. Torch CUDA errors also inherit RuntimeError.
        raise QuickTrainWorkerError(str(error)) from error


class QuickTrainWidget(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.model = None
        self.prepared = None
        self._prepared_signature = None
        self._generation = 0
        self._source_emitters = []
        self._crop_preview_layers = []
        self.worker = None
        self.cancel_event = None
        self.closed = False
        self._locked_layer = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 6, 0, 0)
        note = QLabel(
            "Train a small U-Net from a grayscale image and complete ground truth. "
            "All nonzero labels are targets; every unlabeled pixel is background. "
            "Inspect the training crops before training, then preview a new image."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        group = QGroupBox("1. Prepare & preview crops")
        form = QFormLayout(group)
        self.image_combo = QComboBox()
        self.truth_combo = QComboBox()
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(8, 128)
        self.samples_spin.setValue(32)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 10000)
        self.steps_spin.setValue(200)
        self.crop_combo = QComboBox()
        for size in (256, 512, 1024, 2048):
            self.crop_combo.addItem(f"{size} × {size} px", size)
        self.crop_combo.setCurrentIndex(1)
        self.crop_combo.setToolTip(
            "Source area covered by each patch. Use smaller crops for tiny targets, "
            "or larger crops for more surrounding context."
        )
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("256 × 256 (faster)", 256)
        self.resolution_combo.addItem("512 × 512 (more detail)", 512)
        self.resolution_combo.setCurrentIndex(1)
        self.resolution_combo.setToolTip(
            "The grid shows this exact training resolution. Smaller resolutions "
            "reduce boundary detail. Foreground-preserving mask pooling retains tiny targets."
        )
        self.device_combo = QComboBox()
        for label, value in (("Automatic (CUDA or CPU)", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")):
            self.device_combo.addItem(label, value)
        form.addRow("Training image", self.image_combo)
        form.addRow("Ground truth", self.truth_combo)
        form.addRow("Crop context", self.crop_combo)
        form.addRow("Training resolution", self.resolution_combo)
        form.addRow("Crop count", self.samples_spin)
        validation = QLabel(
            "Find targets across the whole mask, then sample positives, nearby "
            "background, and ordinary background. Separate spatial blocks are "
            "held out for validation."
        )
        validation.setWordWrap(True)
        form.addRow(validation)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1000000)
        self.seed_spin.setValue(42)
        form.addRow("Crop sampling seed", self.seed_spin)
        self.seed_spin.setToolTip("Change this number and prepare again to review a different set of crops.")
        self.train_button = QPushButton("Quick Train — start new model")
        self.continue_button = QPushButton("Continue training current model")
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setRange(0.00001, 0.1)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(0.001)
        self.prepare_button = QPushButton("Prepare & Preview Crops")
        self.prepare_button.setToolTip("Scan the ground truth in bounded tiles, then display the actual image/mask inputs as a grid. Large masks can take time to scan.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.prepare_button)
        row.addWidget(self.cancel_button)
        form.addRow(row)
        self.crop_summary_label = QLabel("Prepare crops to inspect positive and background examples.")
        self.crop_summary_label.setWordWrap(True)
        form.addRow(self.crop_summary_label)
        layout.addWidget(group)

        train_group = QGroupBox("2. Train reviewed crops")
        train_form = QFormLayout(train_group)
        train_form.addRow("Training steps", self.steps_spin)
        train_form.addRow("Compute", self.device_combo)
        train_form.addRow("Learning rate", self.learning_rate_spin)
        self.train_button.setToolTip("Start a new model from scratch using the reviewed crops.")
        train_form.addRow(self.train_button)
        train_form.addRow(self.continue_button)
        training_help = QLabel("Continue updates the current or loaded model. For a new round, change the crop seed or source, prepare and review again. Include earlier examples to reduce forgetting. A lower learning rate (such as 0.0001) is useful for fine-tuning.")
        training_help.setWordWrap(True)
        train_form.addRow(training_help)
        self.metrics_label = QLabel("No model trained or loaded.")
        self.metrics_label.setWordWrap(True)
        train_form.addRow(self.metrics_label)
        model_row = QHBoxLayout()
        self.save_button = QPushButton("Save model…")
        self.load_button = QPushButton("Load model…")
        model_row.addWidget(self.save_button)
        model_row.addWidget(self.load_button)
        train_form.addRow(model_row)
        layout.addWidget(train_group)

        preview_group = QGroupBox("3. Preview on a new image")
        preview_form = QFormLayout(preview_group)
        self.prediction_combo = QComboBox()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setToolTip("Higher thresholds select fewer pixels.")
        self.predict_button = QPushButton("Predict current area")
        self.predict_button.setToolTip(
            "Add an experimental Labels preview centered at the current canvas "
            "position. Only one crop is read; the full image is not segmented."
        )
        preview_form.addRow("Prediction image", self.prediction_combo)
        preview_form.addRow("Threshold", self.threshold_spin)
        preview_form.addRow(self.predict_button)
        preview_note = QLabel(
            "Pan to an area of interest, then predict. The new Labels layer covers "
            "only that area. Review its boundaries and false positives."
        )
        preview_note.setWordWrap(True)
        preview_form.addRow(preview_note)
        layout.addWidget(preview_group)
        self.refresh_button = QPushButton("Refresh layers")
        layout.addWidget(self.refresh_button)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QLabel("Choose a training image and its complete ground truth.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.train_button.clicked.connect(lambda: self._safe(self._start_training))
        self.prepare_button.clicked.connect(lambda: self._safe(self._start_preparing))
        self.predict_button.clicked.connect(lambda: self._safe(self._start_prediction))
        self.cancel_button.clicked.connect(self._cancel)
        self.refresh_button.clicked.connect(self.refresh_layers)
        self.save_button.clicked.connect(lambda: self._safe(self._save))
        self.load_button.clicked.connect(lambda: self._safe(self._load))
        for control in (self.crop_combo, self.resolution_combo):
            control.currentIndexChanged.connect(self._invalidate_crops)
        self.continue_button.clicked.connect(lambda: self._safe(lambda: self._start_training(continue_training=True)))
        self.seed_spin.valueChanged.connect(self._invalidate_crops)
        self.samples_spin.valueChanged.connect(self._invalidate_crops)
        self.image_combo.currentIndexChanged.connect(self._sources_changed)
        self.truth_combo.currentIndexChanged.connect(self._sources_changed)
        self.destroyed.connect(lambda *_args: self.shutdown())
        # Reuse the plugin's layer lifecycle handling without importing PyTorch.
        from ._widget import _auto_refresh_layer_controls

        self._layer_refresh_timer = _auto_refresh_layer_controls(viewer, self, self.refresh_layers)
        self.refresh_layers()
        self._update_controls()

    def _safe(self, function):
        try:
            function()
        except Exception as error:
            self._status(str(error))

    def _status(self, message):
        if not self.closed:
            self.status_label.setText(message)
            self.viewer.status = message

    def refresh_layers(self):
        if self.closed or self.worker is not None:
            return
        previous = (self.image_combo.currentData(), self.truth_combo.currentData())
        for combo, layer_type in (
            (self.image_combo, napari.layers.Image),
            (self.truth_combo, napari.layers.Labels),
            (self.prediction_combo, napari.layers.Image),
        ):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for layer in self.viewer.layers:
                if isinstance(layer, layer_type) and not is_internal_layer(layer):
                    if layer_type is napari.layers.Image and layer.rgb:
                        continue
                    combo.addItem(layer.name, layer)
            for index in range(combo.count()):
                if combo.itemData(index) is current:
                    combo.setCurrentIndex(index)
                    break
            combo.blockSignals(False)
        if previous[0] is not self.image_combo.currentData() or previous[1] is not self.truth_combo.currentData():
            self._sources_changed()

    def _sources_changed(self, *_args):
        for emitter in self._source_emitters:
            with suppress(Exception):
                emitter.disconnect(self._invalidate_crops)
        self._source_emitters.clear()
        for layer in (self.image_combo.currentData(), self.truth_combo.currentData()):
            if layer is None:
                continue
            for name in ("data", "paint"):
                emitter = getattr(layer.events, name, None)
                if emitter is not None:
                    emitter.connect(self._invalidate_crops)
                    self._source_emitters.append(emitter)
        self._invalidate_crops()

    def _invalidate_crops(self, *_args):
        if self.closed:
            return
        self._generation += 1
        was_prepared = self.prepared is not None
        self.prepared = None
        self._prepared_signature = None
        if was_prepared:
            self.crop_summary_label.setText("Source or crop settings changed. Prepare and preview again before training.")
        self._update_controls()

    def _layer(self, combo):
        layer = combo.currentData()
        if layer is None or not any(layer is item for item in self.viewer.layers):
            raise ValueError("Choose an available image or Labels layer.")
        return layer

    def _update_controls(self):
        busy = self.worker is not None
        for control in (
            self.prepare_button, self.image_combo, self.truth_combo, self.samples_spin,
            self.steps_spin, self.seed_spin, self.learning_rate_spin, self.crop_combo, self.device_combo, self.load_button,
            self.resolution_combo, self.prediction_combo, self.threshold_spin, self.refresh_button,
        ):
            control.setEnabled(not busy)
        self.save_button.setEnabled(not busy and self.model is not None)
        self.train_button.setEnabled(not busy and self.prepared is not None)
        self.continue_button.setEnabled(not busy and self.prepared is not None and self.model is not None)
        self.predict_button.setEnabled(not busy and self.model is not None)
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)

    def _start_worker(self, function, on_done, *args, **kwargs):
        if self.worker is not None:
            raise ValueError("Wait for the current Quick Train operation to finish.")
        self.cancel_event = Event()
        kwargs["cancel"] = self.cancel_event
        worker = create_worker(
            _run_task, function, args, kwargs, _start_thread=False,
            _connect={"errored": self._on_error},
        )
        self.worker = worker
        if hasattr(worker, "yielded"):
            worker.yielded.connect(self._on_progress)
        worker.returned.connect(lambda result: self._returned(result, on_done))
        worker.finished.connect(self._finished)
        self.progress.setRange(0, 0)
        self._update_controls()
        worker.start()

    def _on_progress(self, update):
        if self.closed:
            return
        self.progress.setRange(0, update.total)
        self.progress.setValue(update.completed)
        self._status(update.message)

    def _returned(self, result, on_done):
        if self.closed:
            return
        if self.cancel_event is not None and self.cancel_event.is_set():
            self._status("Quick Train operation canceled. Existing layers and model are unchanged.")
            return
        self._safe(lambda: on_done(result))

    def _on_error(self, error):
        if isinstance(error, QuickTrainCancelled):
            self._status("Quick Train operation canceled. Existing layers and model are unchanged.")
        else:
            self._status(f"Quick Train failed: {error}")

    def _finished(self):
        self.worker = None
        self.cancel_event = None
        if self._locked_layer is not None:
            layer, editable = self._locked_layer
            layer.editable = editable
            self._locked_layer = None
        if not self.closed:
            self._update_controls()
            self.refresh_layers()

    def _cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._status("Canceling after the current crop or training step…")

    def _config(self):
        return QuickTrainConfig(
            crop_pixels=int(self.crop_combo.currentData()),
            input_pixels=int(self.resolution_combo.currentData()),
            samples=self.samples_spin.value(), steps=self.steps_spin.value(),
            device=self.device_combo.currentData(), seed=self.seed_spin.value(),
            base_channels=self.model.config.base_channels if self.model is not None else 8,
        )

    def _signature(self):
        return (id(self.image_combo.currentData()), id(self.truth_combo.currentData()),
                self.crop_combo.currentData(), self.resolution_combo.currentData(),
                self.samples_spin.value(), self.seed_spin.value())

    @staticmethod
    def _prepare_task(image, truth, config, *, cancel):
        crops = yield from prepare_crops(image, truth, config, cancel=cancel)
        atlas = build_crop_atlas(crops)
        return crops, atlas

    def _start_preparing(self):
        image = self._layer(self.image_combo)
        truth = self._layer(self.truth_combo)
        image_data, truth_data = array_2d(image.data), array_2d(truth.data)
        if tuple(image_data.shape) != tuple(truth_data.shape):
            raise ValueError("Training image and ground truth must have the same 2D shape.")
        config = self._config()
        signature, generation = self._signature(), self._generation
        self._status("Scanning ground truth and preparing crops. No training will start until you review the grid and click Quick Train.")
        self._locked_layer = (truth, bool(truth.editable))
        truth.editable = False
        try:
            self._start_worker(
                self._prepare_task,
                lambda result: self._preparation_done(result, image, truth, signature, generation),
                image_data, truth_data, config,
            )
        except Exception:
            truth.editable = self._locked_layer[1]
            self._locked_layer = None
            raise

    def _preparation_done(self, result, image, truth, signature, generation):
        if generation != self._generation or not all(any(layer is item for item in self.viewer.layers) for layer in (image, truth)):
            self._status("Source changed during preparation; crops discarded. Prepare again.")
            return
        crops, atlas = result
        self._show_crop_grid(atlas, image, truth)
        self.prepared = crops
        self._prepared_signature = signature
        counts = crop_summary(crops)
        self.crop_summary_label.setText("\n".join(
            f"{split.title()}: {c['total']} crops · {c['positive']} positive · {c['background']} background ({c['nearby_background']} nearby)."
            for split, c in counts.items()
        ))
        extra = " Some requested negatives contain foreground; inspect the actual counts." if any(p["kind"] == "mixed context" for p in crops.locations) else ""
        self._status("Crops ready. Inspect the aligned image/mask grid, then click Quick Train. Double-click a tile to locate it in the source." + extra)

    def _show_crop_grid(self, atlas, image, truth):
        for layer in self._crop_preview_layers:
            if any(layer is item for item in self.viewer.layers):
                self.viewer.layers.remove(layer)
        self._crop_preview_layers = []
        translation = (float(image.extent.world[0, 0]), float(image.extent.world[0, 1]) - atlas.image.shape[1] - 64)
        metadata = {
            INTERNAL_LAYER_ROLE_KEY: "quick_train_crops",
            "quick_train_crops": {"image_source": image.name, "mask_source": truth.name,
                                  "tiles": atlas.locations, "read_only_preview": True},
        }
        grid_image = self.viewer.add_image(atlas.image, name="Quick Train — crop images", contrast_limits=(0, 1), translate=translation, metadata=metadata.copy())
        grid_mask = self.viewer.add_labels(atlas.mask, name="Quick Train — crop ground truth", translate=translation, metadata=metadata.copy(), opacity=0.5)
        grid_mask.editable = False
        rectangles, captions, colors = [], [], []
        for tile in atlas.locations:
            y, x, size = tile["atlas_y0"], tile["atlas_x0"], tile["model_size"]
            rectangles.append([[y, x], [y, x + size - 1], [y + size - 1, x + size - 1], [y + size - 1, x]])
            captions.append(f"{tile['tile']:02d} {tile['split']} · {tile['kind']}\ny={tile['y0']}, x={tile['x0']}")
            colors.append("orange" if tile["split"] == "validation" else "cyan")
        frames = self.viewer.add_shapes(
            rectangles, shape_type="rectangle", name="Quick Train — crop coordinates",
            features={"caption": captions}, edge_color=colors, edge_width=1,
            face_color="transparent", translate=translation, metadata=metadata.copy(),
            text={"string": "{caption}", "anchor": "upper_left", "color": "white",
                  "translation": [-size / 2 - 8, -size / 2], "size": 10},
        )
        frames.editable = False
        self._crop_preview_layers = [grid_image, grid_mask, frames]

        def show_tile(_layer, event, *, locate=False):
            position = grid_image.world_to_data(tuple(event.position)[-2:])
            row, col = int(np.floor(position[0] / atlas.stride_y)), int(np.floor(position[1] / atlas.stride_x))
            index = row * atlas.columns + col
            if row < 0 or not 0 <= col < atlas.columns or not 0 <= index < len(atlas.locations):
                return
            tile = atlas.locations[index]
            self._status(f"Tile {tile['tile']}: {tile['split']}, {tile['kind']}; source y={tile['y0']}, x={tile['x0']}; foreground {tile['foreground_pixels_original']} source pixels → {tile['foreground_pixels_model']} model pixels.")
            if locate and any(image is item for item in self.viewer.layers):
                center = image.data_to_world((tile["y0"] + tile["size"] / 2, tile["x0"] + tile["size"] / 2))
                y, x, size = tile["y0"], tile["x0"], tile["size"]
                world_corners = np.array([image.data_to_world(p) for p in ((y, x), (y + size, x), (y, x + size), (y + size, x + size))])
                extent = np.maximum(np.ptp(world_corners, axis=0), 1e-6)
                self._focus(center, *extent)
                self.viewer.layers.selection.active = image

        for layer in self._crop_preview_layers:
            layer.mouse_drag_callbacks.append(lambda layer, event: show_tile(layer, event) if event.type == "mouse_press" else None)
            layer.mouse_double_click_callbacks.append(lambda layer, event: show_tile(layer, event, locate=True))
        center = grid_image.data_to_world((atlas.image.shape[0] / 2, atlas.image.shape[1] / 2))
        self._focus(center, *atlas.image.shape)

    def _focus(self, center, height, width):
        previous = tuple(self.viewer.camera.center)
        self.viewer.camera.center = (*previous[:-2], *center)
        canvas_width, canvas_height = self.viewer.window._qt_viewer.canvas.size
        self.viewer.camera.zoom = max(1e-5, min(canvas_width / width, canvas_height / height) * 0.9)

    def _start_training(self, continue_training=False):
        if self.prepared is None or self._signature() != self._prepared_signature:
            raise ValueError("Prepare and preview crops before training.")
        if continue_training and self.model is None:
            raise ValueError("Train or load a model before continuing training.")
        self._status("Continuing the current model on reviewed crops." if continue_training else "Training a new model on reviewed crops.")
        self._start_worker(train_prepared_unet, self._training_done, self.prepared, self._config(),
                           initial_model=self.model if continue_training else None,
                           learning_rate=self.learning_rate_spin.value(),
                           source_info={"image_layer": self._layer(self.image_combo).name,
                                        "ground_truth_layer": self._layer(self.truth_combo).name})

    def _training_done(self, model):
        self.model = model
        self._show_metrics()
        self._status(f"Model ready ({model.training_device.upper()}). Choose a prediction image and click Predict current area. Preview masks need review.")

    def _show_metrics(self):
        metrics = self.model.metrics
        self.metrics_label.setText(
            f"Training: {self.model.training_rounds} rounds · {self.model.training_steps} total steps\n"
            f"Validation: Dice {metrics['dice']:.1%} · IoU {metrics['iou']:.1%}\n"
            f"Precision {metrics['precision']:.1%} · Recall {metrics['recall']:.1%}\n"
            f"Context: {self.model.crop_pixels} × {self.model.crop_pixels} px. "
            "These scores do not measure accuracy on a new image."
        )

    def _start_prediction(self):
        if self.model is None:
            raise ValueError("Train or load a model first.")
        layer = self._layer(self.prediction_combo)
        data = array_2d(layer.data)
        center = layer.world_to_data(tuple(self.viewer.camera.center)[-2:])
        threshold = self.threshold_spin.value()
        self._status("Predicting the current area; only one image crop is read.")
        self._start_worker(
            predict_area, lambda result: self._prediction_done(layer, threshold, result),
            self.model, data, center, device=self.device_combo.currentData(),
        )

    def _prediction_done(self, image_layer, threshold, prediction):
        if not any(image_layer is layer for layer in self.viewer.layers):
            self._status("The prediction image was removed; preview discarded.")
            return
        # Derive the complete 2D pixel-to-world transform, including rotation,
        # shear and affine transforms, then shift it to the crop's data origin.
        origin = np.asarray(image_layer.data_to_world(prediction.origin))
        zero = np.asarray(image_layer.data_to_world((0, 0)))
        affine = np.eye(3)
        affine[:2, 0] = np.asarray(image_layer.data_to_world((1, 0))) - zero
        affine[:2, 1] = np.asarray(image_layer.data_to_world((0, 1))) - zero
        affine[:2, 2] = origin
        mask = (prediction.probability >= threshold).astype(np.uint8)
        self.viewer.add_labels(
            mask, name=f"Quick Train preview — {image_layer.name}", affine=affine,
            metadata={"quick_train": {"experimental": True, "source": image_layer.name,
                                      "origin_yx": list(prediction.origin), "threshold": threshold}},
        )
        self._status(f"Added experimental {mask.shape[0]} × {mask.shape[1]} preview at y={prediction.origin[0]}, x={prediction.origin[1]}. Only this area was segmented.")

    def _save(self):
        if self.model is None:
            raise ValueError("Train or load a model first.")
        path, _ = QFileDialog.getSaveFileName(self, "Save Quick Train model", "quick-train.pt", "Quick Train model (*.pt)")
        if path:
            self._start_worker(self._save_task, lambda _result: self._status(f"Model saved to {path}; training history is embedded and exported beside it as .history.json."), self.model, path)

    @staticmethod
    def _save_task(model, path, *, cancel):
        if cancel.is_set():
            raise QuickTrainCancelled()
        save_model(model, path)
        return path

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Quick Train model", "", "Quick Train model (*.pt)")
        if path:
            self._start_worker(self._load_task, self._training_done, path)

    @staticmethod
    def _load_task(path, *, cancel):
        if cancel.is_set():
            raise QuickTrainCancelled()
        return load_model(path)

    def shutdown(self):
        self.closed = True
        for emitter in self._source_emitters:
            with suppress(Exception):
                emitter.disconnect(self._invalidate_crops)
        self._source_emitters.clear()
        if self.cancel_event is not None:
            self.cancel_event.set()


def quick_train_widget(viewer=None, **kwargs):
    if viewer is None:
        viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found.")
    return QuickTrainWidget(viewer)
