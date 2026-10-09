"""Simple in-napari training and local mask preview workflow."""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
from uuid import uuid4
from contextlib import suppress
from threading import Event
from types import GeneratorType

import napari
import numpy as np
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)
from superqt.utils import create_worker

from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY, is_internal_layer
from ._quick_train import (
    QuickTrainCancelled, QuickTrainConfig, array_2d, build_crop_atlas, crop_summary,
    load_model, predict_area, prepare_crops, save_model, train_prepared_unet,
)


from ._coverage_train import IndexedPatches, index_patches, preview_index, train_indexed, train_kfold, train_final


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
        self.cross_validation = None
        self._validation_preview_layer = None
        self.prepared = None
        self._prepared_signature = None
        self._generation = 0
        self._source_emitters = []
        self._crop_preview_layers = []
        self._gallery_return_view = None
        self.worker = None
        self.cancel_event = None
        self.closed = False
        self._locked_layer = None
        self._locked_extra_layers = []
        self._coverage_splits = {}
        self.model_state_label = QLabel("No model in memory.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        self.workflow_tabs = QTabWidget()
        dataset_page = QWidget()
        dataset_layout = QVBoxLayout(dataset_page)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.addWidget(scroll, 1)
        self.workflow_tabs.addTab(dataset_page, "1. Dataset")
        outer.addWidget(self.workflow_tabs)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 6, 0, 0)
        note = QLabel(
            "Choose annotations, then preview the combined dataset."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.dataset_task_label = QLabel()
        self.dataset_task_label.setWordWrap(True)
        layout.addWidget(self.dataset_task_label)

        group = QGroupBox("Sources")
        form = QFormLayout(group)
        self.image_combo = QComboBox()
        self.truth_combo = QComboBox()
        self.reviewed_combo = QComboBox()
        self.reviewed_combo.setToolTip(
            "Choose a separate Labels layer marking fully reviewed regions with nonzero pixels. "
            "Outside those regions, pixels are ignored. Mark a region reviewed only after "
            "all target objects there have been annotated; accepted objects alone are not complete ground truth."
        )
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItem("All patches (recommended)", "coverage")
        self.workflow_combo.addItem("Sample patches", "sampled")
        self.annotation_combo = QComboBox()
        self.annotation_combo.addItem("Partial — reviewed cells + selected objects", "partial")
        self.annotation_combo.addItem("Complete — whole image", "complete")
        self.create_reviewed_button = QPushButton("Create reviewed mask…")
        self.create_reviewed_button.setToolTip("Create an empty disk-backed Labels layer matching the training image. In Labels, mark only regions with complete annotations as reviewed using label 1.")
        self.annotation_note = QLabel()
        self.annotation_note.setWordWrap(True)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(5)
        self.epochs_spin.setToolTip("Maximum epochs. Plateau control can finish earlier; with patience 10 + 10, allow more than 20 epochs to give it time to act.")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(2)
        self.batch_spin.setToolTip("Patches processed in one optimizer update. Larger batches and model inputs need more device memory; reduce batch size if training runs out of memory.")
        self.extra_targets_spin = QSpinBox()
        self.extra_targets_spin.setRange(0, 100)
        self.extra_targets_spin.setValue(25)
        self.extra_targets_spin.setSuffix(" %")
        self.extra_targets_spin.setToolTip("Every training patch is visited each epoch. Add extra target-containing patches to balance rare foreground. These extra visits do not replace coverage.")
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(8, 128)
        self.samples_spin.setValue(32)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 10000)
        self.steps_spin.setValue(200)
        self.crop_combo = QComboBox()
        for size in (128, 256, 512, 1024, 2048, 4096):
            self.crop_combo.addItem(f"{size} × {size} px", size)
        self.crop_combo.setCurrentIndex(self.crop_combo.findData(512))
        self.crop_combo.setToolTip(
            "Source area covered by each patch. Use smaller crops for tiny targets, "
            "or larger crops for more surrounding context."
        )
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("256 × 256 (faster)", 256)
        self.resolution_combo.addItem("512 × 512 (more detail)", 512)
        self.resolution_combo.addItem("1024 × 1024", 1024)
        self.resolution_combo.addItem("2048 × 2048", 2048)
        self.resolution_combo.setCurrentIndex(1)
        self.resolution_combo.setToolTip(
            "Pixel size fed into the model after resizing the source patch. Matching source patch and model input sizes preserves resolution. Smaller inputs "
            "reduce boundary detail. Foreground-preserving mask pooling retains tiny targets."
        )
        self.device_combo = QComboBox()
        for label, value in (("Automatic (CUDA or CPU)", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")):
            self.device_combo.addItem(label, value)
        form.addRow("Workflow", self.workflow_combo)
        form.addRow("Annotation coverage", self.annotation_combo)
        form.addRow("Training image", self.image_combo)
        form.addRow("Ground truth", self.truth_combo)
        form.addRow("Reviewed-region mask", self.reviewed_combo)
        form.addRow(self.create_reviewed_button)
        form.addRow("Source patch size", self.crop_combo)
        form.addRow("Model input size", self.resolution_combo)
        form.addRow("Crop count", self.samples_spin)
        self.dataset_form = form
        self.validation_note = validation = QLabel(
            "Find targets across the whole mask, then sample positives, nearby "
            "background, and ordinary background. Separate spatial blocks are "
            "held out for validation."
        )
        validation.setWordWrap(True)
        self.workflow_combo.setToolTip(validation.text())
        validation.hide()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1000000)
        self.seed_spin.setValue(42)
        form.addRow("Shuffle / sampling seed", self.seed_spin)
        self.seed_spin.setToolTip("Coverage mode shuffles every eligible training patch each epoch; validation regions stay fixed. Sampled mode uses this seed to choose crops.")
        self.train_button = QPushButton("Train new model")
        self.continue_button = QPushButton("Continue training")
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setRange(0.00001, 0.1)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(0.001)
        self.prepare_button = QPushButton("Prepare dataset")
        self.patch_preview_button = QPushButton("Show patch preview")
        self.patch_preview_button.setToolTip("Optional: show representative model inputs in a separate contact sheet beside the source image. Double-click a tile to locate its source.")
        self.return_image_button = QPushButton("Return to image")
        self.return_image_button.setToolTip("Hide the contact sheet and restore the canvas view from before opening it.")
        self.prepare_button.setToolTip("Scan the ground truth in bounded tiles, then display the actual image/mask inputs as a grid. Large masks can take time to scan.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.prepare_button)
        row.addWidget(self.patch_preview_button)
        row.addWidget(self.return_image_button)
        prepare_group = QGroupBox("Prepare dataset")
        prepare_layout = QVBoxLayout(prepare_group)
        prepare_layout.addLayout(row)
        self.crop_summary_label = QLabel("Prepare crops to inspect positive and background examples.")
        self.crop_summary_label.setWordWrap(True)
        prepare_layout.addWidget(self.crop_summary_label)
        layout.addWidget(group)
        self.annotation_tabs = QTabWidget()
        self.reviewed_cells_page = QWidget()
        self.reviewed_cells_layout = QVBoxLayout(self.reviewed_cells_page)
        self.reviewed_cells_layout.setContentsMargins(0, 0, 0, 0)
        self.reviewed_cells_hint = QLabel("Select an editable cell in Labels → Local Editing, correct its masks, then finish its review here.")
        self.reviewed_cells_hint.setWordWrap(True)
        self.reviewed_cells_layout.addWidget(self.reviewed_cells_hint)
        self.selected_objects_page = QWidget()
        self.selected_objects_layout = QVBoxLayout(self.selected_objects_page)
        self.selected_objects_layout.setContentsMargins(0, 0, 0, 0)
        self.selected_objects_hint = QLabel("Open the main Label Assistant to select library objects for this dataset.")
        self.selected_objects_hint.setWordWrap(True)
        self.selected_objects_layout.addWidget(self.selected_objects_hint)
        self.annotation_tabs.addTab(self.reviewed_cells_page, "Reviewed cells")
        self.annotation_tabs.addTab(self.selected_objects_page, "Selected objects")
        layout.addWidget(self.annotation_tabs, 1)
        prepare_layout.insertWidget(0, self.annotation_note)
        dataset_layout.addWidget(prepare_group)
        self.reviewed_area_widget = None
        self.component_library_widget = None
        train_scroll = QScrollArea()
        train_scroll.setWidgetResizable(True)
        train_scroll.setFrameShape(QScrollArea.NoFrame)
        train_body = QWidget()
        train_layout = QVBoxLayout(train_body)
        train_scroll.setWidget(train_body)
        self.train_page = train_scroll
        self.workflow_tabs.addTab(train_scroll, "3. Train")
        self.training_model_label = QLabel()
        self.training_model_label.setWordWrap(True)
        train_layout.addWidget(self.training_model_label)
        train_group = QGroupBox("Train and evaluate")
        train_form = QFormLayout(train_group)
        self.train_form = train_form
        train_form.addRow("Maximum epochs", self.epochs_spin)
        train_form.addRow("Training steps", self.steps_spin)
        train_form.addRow("Batch size", self.batch_spin)
        train_form.addRow("Extra visits to patches with objects (%)", self.extra_targets_spin)
        self.workload_label = QLabel()
        self.workload_label.setWordWrap(True)
        train_form.addRow(self.workload_label)
        train_form.addRow("Compute", self.device_combo)
        train_form.addRow("Learning rate", self.learning_rate_spin)
        self.train_button.setToolTip("Start a new model from scratch using the reviewed crops.")
        training_buttons = QHBoxLayout()
        training_buttons.addWidget(self.train_button, 1)
        training_buttons.addWidget(self.continue_button, 1)
        train_form.addRow(training_buttons)
        self.continuation_note = QLabel()
        self.continuation_note.setWordWrap(True)
        self.results_toggle = QToolButton()
        self.results_toggle.setText("Show metrics and history")
        self.results_toggle.setCheckable(True)
        self.results_toggle.toggled.connect(self.continuation_note.setVisible)
        train_form.addRow(self.results_toggle)
        train_form.addRow(self.continuation_note)
        self.continuation_note.hide()
        training_help = QLabel("Continue starts from the current or loaded model. Each coverage epoch visits all training patches; no manual resampling is needed. The best validation checkpoint is retained. Review prediction errors and update annotations before reindexing. Models stay in memory until you save them to disk.")
        training_help.setWordWrap(True)
        self.continue_button.setToolTip(training_help.text())
        training_help.deleteLater()
        self.metrics_label = QLabel("No model trained or loaded.")
        self.metrics_label.setWordWrap(True)
        train_form.addRow(self.metrics_label)
        self.metrics_label.hide()
        self.results_toggle.toggled.connect(self.metrics_label.setVisible)
        self.model_state_label.setWordWrap(True)
        train_form.addRow(self.model_state_label)
        model_row = QHBoxLayout()
        self.save_button = QPushButton("Save model…")
        self.load_button = QPushButton("Load model…")
        model_row.addWidget(self.save_button)
        model_row.addWidget(self.load_button)
        train_form.addRow(model_row)
        train_layout.addWidget(train_group)
        self.validation_group = QGroupBox("Validation")
        validation_form = QFormLayout(self.validation_group)
        self.validation_method_combo = QComboBox()
        self.validation_method_combo.addItem("Spatial holdout (recommended)", "holdout")
        self.validation_method_combo.addItem("Spatial k-fold (advanced)", "kfold")
        self.holdout_spin = QSpinBox()
        self.holdout_spin.setRange(5, 50)
        self.holdout_spin.setValue(25)
        self.holdout_spin.setSuffix(" %")
        self.fold_spin = QSpinBox()
        self.fold_spin.setRange(2, 10)
        self.fold_spin.setValue(5)
        self.validation_seed_spin = QSpinBox()
        self.validation_seed_spin.setRange(0, 1000000)
        self.validation_seed_spin.setValue(42)
        validation_form.addRow("Method", self.validation_method_combo)
        validation_form.addRow("Validation holdout", self.holdout_spin)
        validation_form.addRow("Number of folds", self.fold_spin)
        validation_form.addRow("Split seed", self.validation_seed_spin)
        self.validation_form = validation_form
        self.validation_summary = QLabel("Prepare the dataset to inspect spatial validation coverage.")
        self.validation_summary.setWordWrap(True)
        validation_form.addRow(self.validation_summary)
        self.preview_validation_button = QPushButton("Preview validation regions on canvas")
        self.preview_validation_button.clicked.connect(lambda: self._safe(self._preview_validation))
        validation_form.addRow(self.preview_validation_button)
        from ._learning_chart import LearningChart
        self.learning_chart = LearningChart()
        self.chart_mode_combo = QComboBox()
        self.chart_mode_combo.addItem("Validation scores", "scores")
        self.chart_mode_combo.addItem("Validation loss", "loss")
        self.chart_mode_combo.setToolTip("Validation loss is measured on held-out pixels after each evaluated epoch. Lower is better; its scale differs from percentage scores.")
        self.chart_mode_combo.currentIndexChanged.connect(lambda: self.learning_chart.set_loss_mode(self.chart_mode_combo.currentData() == "loss"))
        validation_form.addRow("Chart", self.chart_mode_combo)
        validation_form.addRow(self.learning_chart)
        self.export_csv_button = QPushButton("Export metrics CSV…")
        self.export_csv_button.setEnabled(False)
        self.export_csv_button.setToolTip("Export evaluated checkpoints with validation loss and scores. During training this saves a snapshot; export again after completion for the full run.")
        self.export_csv_button.clicked.connect(lambda: self._safe(self._export_metrics_csv))
        validation_form.addRow(self.export_csv_button)
        self.validation_results_label = QLabel("No validation run completed.")
        self.validation_results_label.setWordWrap(True)
        self.validation_results_label.setMinimumHeight(self.validation_results_label.fontMetrics().lineSpacing()*3)
        validation_form.addRow(self.validation_results_label)
        self.save_validation_button = QPushButton("Save fold models and validation report…")
        self.save_validation_button.clicked.connect(lambda: self._safe(self._save_validation))
        validation_form.addRow(self.save_validation_button)
        self.final_train_button = QPushButton("Train final model on all reviewed regions")
        self.final_train_button.setToolTip("Starts fresh weights and trains on every reviewed patch. No holdout scores are reported for this final model; cross-validation results are retained in its history.")
        self.final_train_button.clicked.connect(lambda: self._safe(lambda: self._start_training(final_training=True)))
        validation_form.addRow(self.final_train_button)
        train_layout.addWidget(self.validation_group)
        for control in (self.validation_method_combo, self.holdout_spin, self.fold_spin, self.validation_seed_spin):
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._validation_changed)
            else:
                control.valueChanged.connect(self._validation_changed)

        train_layout.addStretch(1)
        self.advanced_panel = QWidget()
        advanced_form = QFormLayout(self.advanced_panel)
        advanced_form.setContentsMargins(8, 0, 8, 6)
        for source_form, control in (
            (form, self.crop_combo), (form, self.resolution_combo),
            (form, self.seed_spin), (train_form, self.batch_spin),
            (train_form, self.device_combo), (train_form, self.learning_rate_spin),
        ):
            row = source_form.takeRow(control)
            advanced_form.addRow(row.labelItem.widget(), control)
        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_scroll.setFrameShape(QScrollArea.NoFrame)
        advanced_body = QWidget()
        advanced_layout = QVBoxLayout(advanced_body)
        advanced_layout.addWidget(self.advanced_panel)
        augment_group = QGroupBox("Augmentation — training only")
        augment_form = QFormLayout(augment_group)
        self.rotation_check = QCheckBox("Random 90° rotations")
        self.flip_check = QCheckBox("Random flips")
        self.intensity_check = QCheckBox("Brightness and contrast variation")
        for control in (self.rotation_check, self.flip_check, self.intensity_check):
            control.setChecked(True)
            augment_form.addRow(control)
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0, 50)
        self.contrast_spin.setValue(10)
        self.contrast_spin.setSuffix(" %")
        self.contrast_spin.setToolTip("Random contrast multiplier between 1 − variation and 1 + variation.")
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0, 25)
        self.brightness_spin.setValue(3)
        self.brightness_spin.setSuffix(" %")
        self.brightness_spin.setToolTip("Random offset as a percentage of the normalized image range; masks are unchanged.")
        augment_form.addRow("Contrast ±", self.contrast_spin)
        augment_form.addRow("Brightness ±", self.brightness_spin)
        advanced_layout.addWidget(augment_group)
        self.plateau_group = QGroupBox("Reduce learning rate, then early stop")
        self.plateau_group.setCheckable(True)
        self.plateau_group.setChecked(False)
        plateau_form = QFormLayout(self.plateau_group)
        plateau_note = QLabel("All patches only. Monitor validation loss each epoch; retain the best validation Dice checkpoint.")
        plateau_note.setWordWrap(True)
        plateau_form.addRow(plateau_note)
        self.reduce_patience_spin = QSpinBox()
        self.stop_patience_spin = QSpinBox()
        for control in (self.reduce_patience_spin, self.stop_patience_spin):
            control.setRange(1, 200)
            control.setValue(10)
            control.setSuffix(" epochs")
        self.lr_factor_spin = QDoubleSpinBox()
        self.lr_factor_spin.setRange(0.1, 0.9)
        self.lr_factor_spin.setSingleStep(0.1)
        self.lr_factor_spin.setValue(0.5)
        self.lr_factor_spin.setToolTip("Multiply the learning rate once after the first patience period. Minimum resulting learning rate is 0.0000001.")
        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setDecimals(6)
        self.min_delta_spin.setRange(0, 1)
        self.min_delta_spin.setSingleStep(0.0001)
        self.min_delta_spin.setValue(0.0001)
        self.min_delta_spin.setToolTip("Absolute decrease in validation loss required to reset patience.")
        plateau_form.addRow("Reduce after no improvement", self.reduce_patience_spin)
        plateau_form.addRow("Learning-rate multiplier", self.lr_factor_spin)
        plateau_form.addRow("Stop after reduction and no improvement", self.stop_patience_spin)
        plateau_form.addRow("Minimum loss improvement", self.min_delta_spin)
        advanced_layout.addWidget(self.plateau_group)
        advanced_layout.addStretch(1)
        advanced_scroll.setWidget(advanced_body)
        self.workflow_tabs.addTab(advanced_scroll, "5. Advanced")

        inference_scroll = QScrollArea()
        inference_scroll.setWidgetResizable(True)
        inference_scroll.setFrameShape(QScrollArea.NoFrame)
        inference_body = QWidget()
        inference_layout = QVBoxLayout(inference_body)
        inference_scroll.setWidget(inference_body)
        self.workflow_tabs.addTab(inference_scroll, "4. Inference")
        self.inference_model_label = QLabel()
        self.inference_model_label.setWordWrap(True)
        inference_layout.addWidget(self.inference_model_label)
        inference_intro = QLabel("Train or load a model, choose an image, then pan to the area to predict.")
        inference_intro.setWordWrap(True)
        inference_layout.addWidget(inference_intro)
        self.inference_load_button = QPushButton("Load model…")
        self.inference_load_button.clicked.connect(lambda: self._safe(self._load))
        inference_layout.addWidget(self.inference_load_button)
        preview_group = QGroupBox("Predict an area")
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
        inference_layout.addWidget(preview_group)
        inference_layout.addStretch(1)
        self.workflow_tabs.removeTab(self.workflow_tabs.indexOf(advanced_scroll))
        self.workflow_tabs.addTab(advanced_scroll, "5. Advanced")
        from ._models_widget import ModelsWidget
        self.models_panel = ModelsWidget(self)
        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setWidget(self.models_panel)
        self.workflow_tabs.insertTab(1, model_scroll, "2. Models")
        for index, tooltip in enumerate(("Choose reviewed data and prepare patches", "Define biological tasks, select checkpoints and compare models", "Train and evaluate the selected configuration", "Predict with the current trained model", "Patch context, input size, optimization and augmentation")):
            self.workflow_tabs.setTabToolTip(index, tooltip)
        self.viewer._label_assistant_training_widget = self
        for control in (self.models_panel.provider, self.models_panel.architecture, self.models_panel.encoder, self.models_panel.initialization, self.models_panel.width):
            control.currentIndexChanged.connect(self._update_model_summary)
        self.models_panel.encoder.editTextChanged.connect(self._update_model_summary)
        self.models_panel.architecture.editTextChanged.connect(self._update_model_summary)
        self.refresh_button = QPushButton("Refresh layers")
        outer.addWidget(self.refresh_button)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)
        self.status_label = QLabel("Choose the training image, ground truth, and annotation coverage.")
        self.status_label.setWordWrap(True)
        self.status_label.setFixedHeight(self.status_label.fontMetrics().lineSpacing()*2+8)
        outer.addWidget(self.status_label)
        self.log_toggle = QToolButton()
        self.log_toggle.setText("Activity log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(True)
        outer.addWidget(self.log_toggle)
        self.activity_log = QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumBlockCount(2000)
        self.activity_log.setFixedHeight(100)
        self.activity_log.setPlaceholderText("Training, validation, and errors appear here.")
        self.log_toggle.toggled.connect(self.activity_log.setVisible)
        outer.addWidget(self.activity_log)
        outer.addWidget(self.cancel_button)

        self.intensity_check.toggled.connect(self._update_controls)
        self.create_reviewed_button.clicked.connect(lambda: self._safe(self._create_reviewed_mask))
        self.train_button.clicked.connect(lambda: self._safe(self._start_training))
        self.prepare_button.clicked.connect(lambda: self._safe(self._start_preparing))
        self.patch_preview_button.clicked.connect(lambda: self._safe(self._start_patch_preview))
        self.return_image_button.clicked.connect(self._return_to_image)
        self.predict_button.clicked.connect(lambda: self._safe(self._start_prediction))
        self.cancel_button.clicked.connect(self._cancel)
        self.refresh_button.clicked.connect(self.refresh_layers)
        self.save_button.clicked.connect(lambda: self._safe(self._save))
        self.load_button.clicked.connect(lambda: self._safe(self._load))
        for control in (self.crop_combo, self.resolution_combo):
            control.currentIndexChanged.connect(self._invalidate_crops)
        self.continue_button.clicked.connect(lambda: self._safe(lambda: self._start_training(continue_training=True)))
        self.seed_spin.valueChanged.connect(lambda *_args: self._invalidate_crops() if self.workflow_combo.currentData() == "sampled" else None)
        self.samples_spin.valueChanged.connect(self._invalidate_crops)
        for control in (self.epochs_spin, self.batch_spin, self.extra_targets_spin):
            control.valueChanged.connect(self._update_workload)
        self.workflow_combo.currentIndexChanged.connect(self._workflow_changed)
        self.annotation_combo.currentIndexChanged.connect(self._annotation_changed)
        self.reviewed_combo.currentIndexChanged.connect(self._sources_changed)
        self.image_combo.currentIndexChanged.connect(self._sources_changed)
        self.truth_combo.currentIndexChanged.connect(self._sources_changed)
        self.destroyed.connect(lambda *_args: self.shutdown())
        # Reuse the plugin's layer lifecycle handling without importing PyTorch.
        from ._widget import _auto_refresh_layer_controls

        self._layer_refresh_timer = _auto_refresh_layer_controls(viewer, self, self.refresh_layers)
        self.viewer.layers.events.removed.connect(self._source_removed)
        self.refresh_layers()
        self._workflow_changed()
        self._update_model_summary()
        from ._model_project import project
        settings = project(self.viewer).get("settings")
        if settings:
            self.restore_project_settings(settings)

    def project_settings(self):
        combo_names = ("workflow_combo", "annotation_combo", "crop_combo", "resolution_combo", "device_combo", "validation_method_combo")
        spin_names = ("samples_spin", "steps_spin", "seed_spin", "epochs_spin", "batch_spin", "extra_targets_spin", "learning_rate_spin", "holdout_spin", "fold_spin", "validation_seed_spin", "contrast_spin", "brightness_spin", "reduce_patience_spin", "stop_patience_spin", "lr_factor_spin", "min_delta_spin")
        check_names = ("rotation_check", "flip_check", "intensity_check", "plateau_group")
        from ._model_project import layer_id
        return {"combos": {name: getattr(self,name).currentData() for name in combo_names},
                "values": {name: getattr(self,name).value() for name in spin_names},
                "checks": {name: getattr(self,name).isChecked() for name in check_names},
                "sources": {name: layer_id(getattr(self,name).currentData()) for name in ("image_combo", "truth_combo", "reviewed_combo", "prediction_combo")},
                "new_model": self.models_panel.spec().to_dict(), "model_name": self.models_panel.name.text(),
                "comparison_candidates": self.models_panel.candidates}

    def restore_project_settings(self, settings):
        combos = settings.get("combos", {})
        # Coverage determines whether sampled workflow is eligible. Restore it first.
        ordered = [name for name in ("annotation_combo", "workflow_combo") if name in combos]
        ordered += [name for name in combos if name not in ordered]
        for name in ordered:
            value = combos[name]
            control=getattr(self,name,None)
            if control is not None:
                index=control.findData(value)
                if index>=0:
                    control.setCurrentIndex(index)
        for name,value in settings.get("values",{}).items():
            control=getattr(self,name,None)
            if control is not None:
                control.setValue(value)
        for name,value in settings.get("checks",{}).items():
            control=getattr(self,name,None)
            if control is not None:
                control.setChecked(value)
        for name,identifier in settings.get("sources",{}).items():
            control=getattr(self,name,None)
            if control is not None and identifier:
                for i in range(control.count()):
                    layer=control.itemData(i)
                    if layer is not None and layer.metadata.get("label_assistant_review",{}).get("id")==identifier:
                        control.setCurrentIndex(i)
                        break
        panel=self.models_panel
        spec=settings.get("new_model")
        if spec and panel.provider.findData(spec["provider"]) >= 0:
            panel.provider.setCurrentIndex(panel.provider.findData(spec["provider"]))
            panel.architecture.setCurrentIndex(panel.architecture.findData(spec["architecture"]))
            if panel.architecture.isEditable():
                panel.architecture.setCurrentText(spec["architecture"])
            panel.encoder.setCurrentText(spec["encoder"])
            panel.initialization.setCurrentIndex(panel.initialization.findData(spec["initialization"]))
            width_index = panel.width.findData(spec["width"])
            if width_index >= 0:
                panel.width.setCurrentIndex(width_index)
        panel.name.setText(settings.get("model_name",""))
        panel.candidates=list(settings.get("comparison_candidates",[]))
        panel.candidate_list.clear()
        from ._model_providers import ModelSpec
        for spec in panel.candidates:
            panel.candidate_list.addItem(ModelSpec(**spec).label())
        self.model_state_label.setText("Workspace restored. Select its saved model in Models to continue or predict.")
        self._update_controls()

    def _update_model_summary(self, *_args):
        from ._model_providers import spec_for
        if not hasattr(self, "models_panel"):
            return
        current = spec_for(self.model.config).label() if self.model is not None and hasattr(self.model, "config") else "None selected"
        if self.model is not None:
            from ._model_project import project
            task_id = getattr(self.model, "identity", {}).get("task_id")
            task = next((t for t in project(self.viewer)["tasks"] if t["id"] == task_id), None)
            if task:
                current = task["name"] + " · " + current
        try:
            new_label = self.models_panel.spec().label()
        except (ValueError, TypeError):
            return
        self.training_model_label.setText("New: " + new_label + "\nContinue: " + current)
        self.inference_model_label.setText("Current model: " + current)

    def _safe(self, function):
        try:
            function()
        except Exception as error:
            if self.worker is None:
                self._unlock_sources()
            self._status(str(error))

    def _status(self, message):
        if not self.closed:
            self.status_label.setText(message)
            self.status_label.setToolTip(message)
            if hasattr(self, "activity_log"):
                scrollbar = self.activity_log.verticalScrollBar()
                old_position = scrollbar.value()
                at_end = old_position >= scrollbar.maximum()
                self.activity_log.appendPlainText(datetime.now().strftime("%H:%M:%S") + "  " + message)
                scrollbar.setValue(scrollbar.maximum() if at_end else old_position)
            self.viewer.status = message

    def refresh_layers(self):
        if self.closed or self.worker is not None:
            return
        previous = (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData())
        for combo, layer_type in (
            (self.image_combo, napari.layers.Image),
            (self.truth_combo, napari.layers.Labels),
            (self.reviewed_combo, napari.layers.Labels),
            (self.prediction_combo, napari.layers.Image),
        ):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if combo is self.reviewed_combo:
                combo.addItem("Choose reviewed-region mask…", None)
            for layer in self.viewer.layers:
                if isinstance(layer, layer_type) and not is_internal_layer(layer):
                    if layer_type is napari.layers.Image and layer.rgb:
                        continue
                    review_role = layer.metadata.get("label_assistant_review", {}).get("role")
                    if combo is self.truth_combo and review_role in ("reviewed_regions", "reviewed_annotations"):
                        continue
                    if combo is self.reviewed_combo and review_role == "reviewed_annotations":
                        continue
                    combo.addItem(layer.name, layer)
            for index in range(combo.count()):
                if combo.itemData(index) is current:
                    combo.setCurrentIndex(index)
                    break
            combo.blockSignals(False)
        self._select_associated_reviewed_mask()
        if any(a is not b for a, b in zip(previous, (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData()))):
            self._sources_changed()

    def _source_removed(self, event):
        if any(event.value is layer for layer in (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData())):
            self._invalidate_crops()

    def _select_associated_reviewed_mask(self):
        from ._reviewed_regions import associated_mask, KEY
        if self.annotation_combo.currentData() != "partial":
            return
        image = self.image_combo.currentData()
        mask = associated_mask(self.viewer, image)
        current = self.reviewed_combo.currentData()
        if mask is not None or (current is not None and current.metadata.get(KEY, {}).get("role") == "reviewed_regions"):
            self.reviewed_combo.blockSignals(True)
            self.reviewed_combo.setCurrentIndex(max(0, self.reviewed_combo.findData(mask)))
            self.reviewed_combo.blockSignals(False)
        if mask is not None:
            truth_id = mask.metadata.get(KEY, {}).get("truth_id")
            truth = next((layer for layer in self.viewer.layers if truth_id and layer.metadata.get(KEY, {}).get("id") == truth_id), None)
            if truth is not None:
                self.truth_combo.blockSignals(True)
                self.truth_combo.setCurrentIndex(self.truth_combo.findData(truth))
                self.truth_combo.blockSignals(False)

    def _sources_changed(self, *_args):
        self._select_associated_reviewed_mask()
        for emitter in self._source_emitters:
            with suppress(Exception):
                emitter.disconnect(self._invalidate_crops)
        self._source_emitters.clear()
        for layer in (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData()):
            if layer is None:
                continue
            from ._component_library import ensure_library_event
            ensure_library_event(layer)
            for name in ("data", "paint", "labels_update", "component_library"):
                emitter = getattr(layer.events, name, None)
                if emitter is not None:
                    emitter.connect(self._invalidate_crops)
                    self._source_emitters.append(emitter)
        self._invalidate_crops()
        self._update_annotation_note()

    def _invalidate_crops(self, *_args):
        if self.closed:
            return
        self._generation += 1
        if self.cancel_event is not None:
            self.cancel_event.set()
        self.cross_validation = None
        if hasattr(self, "validation_results_label"):
            self.validation_results_label.setText("No validation run for the current dataset settings. Reindex after changes.")
        if self._validation_preview_layer is not None:
            self._validation_preview_layer.visible = False
        if self._gallery_return_view is not None:
            center, zoom, active = self._gallery_return_view
            self.viewer.camera.center = center
            self.viewer.camera.zoom = zoom
            if active is not None and any(layer is active for layer in self.viewer.layers):
                self.viewer.layers.selection.active = active
            self._gallery_return_view = None
        for layer in self._crop_preview_layers:
            layer.visible = False
        was_prepared = self.prepared is not None
        self.prepared = None
        self._prepared_signature = None
        if was_prepared:
            self.crop_summary_label.setText("Sources or patch settings changed. Prepare the dataset again before training.")
        self._update_controls()
        self._update_annotation_note()

    def _layer(self, combo):
        layer = combo.currentData()
        if layer is None or not any(layer is item for item in self.viewer.layers):
            raise ValueError("Choose an available image or Labels layer.")
        return layer

    def _update_controls(self):
        busy = self.worker is not None
        for control in (
            self.validation_method_combo, self.holdout_spin, self.fold_spin, self.validation_seed_spin,
            self.rotation_check, self.flip_check, self.intensity_check,
            self.prepare_button, self.image_combo, self.truth_combo, self.samples_spin,
            self.steps_spin, self.seed_spin, self.learning_rate_spin, self.crop_combo, self.device_combo, self.load_button, self.inference_load_button,
            self.resolution_combo, self.prediction_combo, self.threshold_spin, self.refresh_button,
            self.workflow_combo, self.annotation_combo, self.reviewed_combo, self.epochs_spin, self.batch_spin, self.extra_targets_spin, self.create_reviewed_button,
        ):
            control.setEnabled(not busy)
        self.contrast_spin.setEnabled(not busy and self.intensity_check.isChecked())
        self.brightness_spin.setEnabled(not busy and self.intensity_check.isChecked())
        self.plateau_group.setEnabled(not busy and self.workflow_combo.currentData() == "coverage")
        self.patch_preview_button.setEnabled(not busy and self.prepared is not None)
        self.return_image_button.setEnabled(not busy and any(layer.visible for layer in self._crop_preview_layers))
        self.save_button.setEnabled(not busy and self.model is not None)
        self.train_button.setEnabled(not busy and self.prepared is not None)
        seen_holdout = isinstance(self.prepared, IndexedPatches) and self.prepared.validation_seen_before
        kfold = self.validation_method_combo.currentData() == "kfold" and self.workflow_combo.currentData() == "coverage"
        self.continue_button.setEnabled(not busy and self.prepared is not None and self.model is not None and not seen_holdout and not kfold)
        self._update_validation_controls(busy)
        self.export_csv_button.setEnabled(bool(self.learning_chart.points))
        self.continuation_note.setText("The current model has already trained in the reserved validation regions. Review new regions for independent validation, or train a new model." if seen_holdout and self.model is not None else "")
        self.continuation_note.setVisible(seen_holdout and self.model is not None)
        self.predict_button.setEnabled(not busy and self.model is not None)
        self.models_panel.set_busy(busy)
        self._update_model_summary()
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)
        if not busy:
            self._update_annotation_note()
            self._update_workload()

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
        try:
            worker.start()
        except Exception:
            self.worker = None
            self.cancel_event = None
            self._unlock_sources()
            self._update_controls()
            raise

    def _on_progress(self, update):
        if self.closed:
            return
        self.progress.setRange(0, update.total)
        self.progress.setValue(update.completed)
        self._status(update.message)
        if update.metrics is not None and update.epoch is not None:
            self.learning_chart.add_point(update.epoch, update.metrics, update.fold, update.training_step if update.training_step is not None else update.completed, update.training_loss_last_batch, "step" if self.workflow_combo.currentData() == "sampled" else "epoch", series=update.series)
            self.export_csv_button.setEnabled(True)
            loss = update.metrics.get('loss')
            prefix = (update.series + " · ") if update.series else (f"Fold {update.fold} · " if update.fold else "")
            unit = "Step" if self.workflow_combo.currentData() == "sampled" else "Epoch"
            text = prefix + f"{unit} {update.epoch}: Dice {update.metrics.get('dice', 0):.1%} · IoU {update.metrics.get('iou', 0):.1%}"
            if loss is not None:
                text += f" · validation loss {loss:.4f}"
            self.validation_results_label.setText(text)

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
        self._unlock_sources()
        if not self.closed:
            self._update_controls()
            self.refresh_layers()

    def _cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._status("Canceling after the current crop or training step…")

    def _training_config(self, continuing):
        from dataclasses import replace
        from ._model_providers import spec_for
        config = self._config()
        if continuing and self.model is not None:
            spec = spec_for(self.model.config)
            config = replace(config, model_spec=spec.to_dict(), base_channels=self.model.config.base_channels)
        config.validate()
        return config

    def _config(self):
        from ._model_providers import spec_for
        spec = self.models_panel.spec()
        return QuickTrainConfig(
            model_spec=spec.to_dict(),
            crop_pixels=int(self.crop_combo.currentData()),
            input_pixels=int(self.resolution_combo.currentData()),
            samples=self.samples_spin.value(), steps=self.steps_spin.value(),
            device=self.device_combo.currentData(), seed=self.seed_spin.value(), batch_size=self.batch_spin.value(),
            base_channels=spec.width if spec.provider == "custom" else 8,
            augment_rotations=self.rotation_check.isChecked(),
            augment_flips=self.flip_check.isChecked(),
            augment_intensity=self.intensity_check.isChecked(),
            contrast_jitter=self.contrast_spin.value() / 100,
            brightness_jitter=self.brightness_spin.value() / 100,
        )

    def _signature(self):
        from ._component_library import library_for
        truth = self.truth_combo.currentData()
        library = library_for(truth) if truth is not None else None
        state = library.state() if library else None
        library_signature = (state["revision"], state["edit_revision"], state["pending"]) if state else None
        return (id(self.image_combo.currentData()), id(self.truth_combo.currentData()),
                self.crop_combo.currentData(), self.resolution_combo.currentData(),
                self.samples_spin.value() if self.workflow_combo.currentData() == "sampled" else None,
                self.seed_spin.value() if self.workflow_combo.currentData() == "sampled" else None, self.workflow_combo.currentData(),
                self.annotation_combo.currentData(), id(self.reviewed_combo.currentData()), library_signature, self.validation_method_combo.currentData(), self.holdout_spin.value(), self.fold_spin.value(), self.validation_seed_spin.value())

    def _validation_changed(self, *_args):
        self._coverage_splits.clear()
        self._invalidate_crops()
        self._update_validation_controls(self.worker is not None)

    def _update_validation_controls(self, busy):
        coverage = self.workflow_combo.currentData() == "coverage"
        kfold = coverage and self.validation_method_combo.currentData() == "kfold"
        self.validation_group.setVisible(coverage)
        for control, visible in ((self.holdout_spin, not kfold), (self.fold_spin, kfold)):
            control.setVisible(visible)
            self.validation_form.labelForField(control).setVisible(visible)
        self.train_button.setText("Run spatial k-fold validation" if kfold else "Train new model")
        self.final_train_button.setVisible(kfold)
        self.save_validation_button.setVisible(kfold)
        self.final_train_button.setEnabled(not busy and self.cross_validation is not None and self.prepared is not None)
        self.save_validation_button.setEnabled(not busy and self.cross_validation is not None)
        self.preview_validation_button.setEnabled(not busy and isinstance(self.prepared, IndexedPatches))
        if not isinstance(self.prepared, IndexedPatches):
            self.validation_summary.setText("Prepare the dataset after choosing validation settings.")
            return
        dataset = self.prepared
        groups = len({item['block'] for item in dataset.locations})
        if kfold:
            count = len(dataset.validation.get('folds', []))
            lines = [f"{count} folds · {groups} independent groups · {len(dataset.locations):,} patches · {count} fresh-model training runs."]
            for number, blocks in enumerate(dataset.validation.get('folds', []), 1):
                validation = sum(item['block'] in blocks for item in dataset.locations)
                lines.append(f"Fold {number}: {len(dataset.locations)-validation:,} training / {validation:,} validation patches")
            self.validation_summary.setText("\n".join(lines))
        else:
            validation = len(dataset.indices('validation'))
            self.validation_summary.setText(f"{groups} spatial groups · {len(dataset.indices('training')):,} training / {validation:,} validation patches ({validation/len(dataset.locations):.1%} actual holdout). Group sizes can change the requested ratio.")

    def _cross_validation_done(self, result):
        self.cross_validation = result
        from ._model_project import remember_model
        task_id = self.models_panel.ensure_task()["id"]
        for number, model in enumerate(result.models, 1):
            remember_model(self.viewer, model, task_id=task_id, select=False, name=f"{self.models_panel.spec().label()} · fold {number}")
        self.models_panel.refresh()
        report = result.report
        lines = [f"Fold {r['fold']}: Dice {r['metrics']['dice']:.1%} · IoU {r['metrics']['iou']:.1%} · loss {r['metrics']['loss']:.4f} · best epoch {r['best_epoch']}" for r in report['fold_results']]
        for key in ('dice', 'iou', 'precision', 'recall', 'loss'):
            summary = report['aggregate'][key]
            lines.append(f"Validation loss: mean {summary['mean']:.4f} · fold SD {summary['std']:.4f}" if key == "loss" else f"{key.title()}: mean {summary['mean']:.1%} · fold SD {summary['std']:.1%}")
        lines.append("Scores use validation-selected checkpoints; they are not independent test-set scores. Save the report or train a final model separately.")
        self.validation_results_label.setText("\n".join(lines))
        from ._training_report import history_points
        points = []
        for number, model in enumerate(result.models, 1):
            points.extend({**point, 'series': f'Fold {number}'} for point in history_points(model.history))
        self.learning_chart.set_points(points)
        self._status("Spatial k-fold validation complete. Current model retained. Save fold models/report or train a final model on all reviewed regions.")

    def _preview_validation(self):
        if not isinstance(self.prepared, IndexedPatches):
            raise ValueError("Prepare the dataset first to index spatial assignments.")
        dataset = self.prepared
        image = self._layer(self.image_combo)
        canvas_width, canvas_height = self.viewer.window._qt_viewer.canvas.size
        center = np.asarray(self.viewer.camera.center[-2:])
        half = np.asarray([canvas_height, canvas_width]) / (2*max(self.viewer.camera.zoom, 1e-6))
        corners = np.asarray([image.world_to_data(tuple(center+offset*half)) for offset in (np.array([-1,-1]), np.array([-1,1]), np.array([1,-1]), np.array([1,1]))])
        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
        rows = [p for p in dataset.locations if p['y1'] >= minimum[-2] and p['y0'] <= maximum[-2] and p['x1'] >= minimum[-1] and p['x0'] <= maximum[-1]]
        if len(rows) > 5000:
            raise ValueError("Zoom into an area to preview validation regions; more than 5,000 indexed patches are visible.")
        if not rows:
            raise ValueError("Pan to a reviewed region on the source image, then preview validation regions.")
        if self._validation_preview_layer is not None and any(self._validation_preview_layer is layer for layer in self.viewer.layers):
            self.viewer.layers.remove(self._validation_preview_layer)
        folds = dataset.validation.get('folds', [])
        palette = ['orange','cyan','lime','magenta','yellow','red','blue','white','pink','gold']
        rectangles, colors, captions = [], [], []
        for item in rows:
            fold = next((i for i, blocks in enumerate(folds) if item['block'] in blocks), None)
            caption = f"Fold {fold+1}" if fold is not None else item['split']
            rectangles.append([[item['y0'],item['x0']], [item['y0'],item['x1']], [item['y1'],item['x1']], [item['y1'],item['x0']]])
            colors.append(palette[fold] if fold is not None else ('orange' if item['split']=='validation' else 'cyan'))
            captions.append(caption)
        self._validation_preview_layer = self.viewer.add_shapes(rectangles, shape_type='rectangle', name='Validation regions — visible patches',
            edge_color=colors, face_color='transparent', edge_width=2, scale=image.scale, translate=image.translate, rotate=image.rotate, shear=image.shear, affine=image.affine,
            features={'partition': captions}, metadata={INTERNAL_LAYER_ROLE_KEY: 'validation_preview'})
        self._validation_preview_layer.editable = False
        self._status("Validation preview: " + ("; ".join(f"Fold {i+1} = {palette[i]}" for i in range(len(folds))) if folds else "orange = validation, cyan = training") + ". Refresh this preview after panning.")

    def _export_metrics_csv(self):
        from ._training_report import write_metrics_csv
        points = [dict(point) for point in self.learning_chart.points]
        if not points:
            raise ValueError("No validation metrics available to export yet.")
        path, _ = QFileDialog.getSaveFileName(self, "Export Validation Metrics", "validation-metrics.csv", "CSV (*.csv)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != '.csv':
            destination = destination.with_suffix('.csv')
        write_metrics_csv(points, destination)
        self._status(f"Exported {len(points)} evaluated checkpoints to {destination}." + (" Training is still running; this is a snapshot." if self.worker is not None else ""))

    def _save_validation(self):
        if self.cross_validation is None:
            raise ValueError("Run spatial k-fold validation first.")
        folder = QFileDialog.getExistingDirectory(self, "Save Validation Results")
        if not folder:
            return
        destination = Path(folder) / ("validation-" + datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + uuid4().hex[:6])
        self._start_worker(self._save_validation_task, lambda path: self._status(f"Saved fold models, report, and learning chart to {path}"), self.cross_validation, destination)

    @staticmethod
    def _save_validation_task(result, destination, *, cancel):
        from ._training_report import history_points, learning_svg, write_metrics_csv
        destination.mkdir(parents=True, exist_ok=False)
        points = []
        for number, model in enumerate(result.models, 1):
            if cancel.is_set():
                raise QuickTrainCancelled()
            save_model(model, destination / f"fold-{number}.pt")
            points.extend({**point, 'series': f'Fold {number}'} for point in history_points(model.history))
        (destination / 'validation-report.json').write_text(json.dumps(result.report, indent=2, allow_nan=False), encoding='utf-8')
        (destination / 'learning-progress.svg').write_text(learning_svg(points, 'Spatial k-fold validation learning progress'), encoding='utf-8')
        (destination / "validation-loss.svg").write_text(learning_svg(points, title="Validation loss", loss=True), encoding="utf-8")
        write_metrics_csv(points, destination / "validation-metrics.csv")
        return destination

    def _update_workload(self, *_args):
        coverage = self.workflow_combo.currentData() == "coverage"
        self.workload_label.setVisible(coverage)
        if not coverage:
            return
        if not isinstance(self.prepared, IndexedPatches):
            self.workload_label.setText("Index the dataset to calculate updates per epoch.")
            return
        count = len(self.prepared.indices('training'))
        extra = int(np.ceil(count * self.extra_targets_spin.value() / 100))
        updates = (count + extra + self.batch_spin.value() - 1) // self.batch_spin.value()
        self.workload_label.setText(
            f"Per epoch: {count:,} unique patches + {extra:,} target repeats = {updates:,} updates. "
            f"Maximum: {self.epochs_spin.value()} epochs, {updates * self.epochs_spin.value():,} updates, "
            "plus validation after each epoch."
        )

    def _annotation_changed(self, *_args):
        self._workflow_changed()
        self._sources_changed()

    def _workflow_changed(self, *_args):
        partial = self.annotation_combo.currentData() == "partial"
        sampled_index = self.workflow_combo.findData("sampled")
        self.workflow_combo.model().item(sampled_index).setEnabled(not partial)
        if partial and self.workflow_combo.currentData() != "coverage":
            self.workflow_combo.blockSignals(True)
            self.workflow_combo.setCurrentIndex(self.workflow_combo.findData("coverage"))
            self.workflow_combo.blockSignals(False)
        coverage = self.workflow_combo.currentData() == "coverage"
        for form, control, visible in (
            (self.dataset_form, self.samples_spin, not coverage),
            (self.train_form, self.steps_spin, not coverage),
            (self.train_form, self.epochs_spin, coverage),
            (self.train_form, self.extra_targets_spin, coverage),
        ):
            control.setVisible(visible)
            form.labelForField(control).setVisible(visible)
        self.prepare_button.setText("Prepare dataset")
        self.prepare_button.setToolTip(
            "Scan the mask in bounded patches and store a coordinate index. Training streams every eligible patch; the gallery shows only representative examples."
            if coverage else "Prepare a fixed crop sample for a quick experiment. Training reuses this sample."
        )
        self.validation_note.setText(
            "Each epoch visits every eligible training patch. Separate spatial blocks are reserved for validation. The gallery is a sample, not the complete dataset."
            if coverage else "Train on a fixed crop sample; separate spatial blocks are held out for validation."
        )
        self.workflow_combo.setToolTip(self.validation_note.text())
        self._invalidate_crops()
        self.crop_summary_label.setText(
            "Index the dataset to inspect coverage and representative examples."
            if coverage else "Prepare crops to inspect positive and background examples."
        )
        self._update_annotation_note()

    def _update_annotation_note(self):
        partial = self.annotation_combo.currentData() == "partial"
        self.reviewed_combo.setVisible(partial)
        self.create_reviewed_button.setVisible(False)
        self.dataset_form.labelForField(self.reviewed_combo).setVisible(partial)
        self.annotation_note.setToolTip(
            "Reviewed cells supply foreground and background. Selected objects supply "
            "foreground only. Shared pixels are counted once; unknown pixels are ignored."
        )
        if not partial:
            self.annotation_note.setText("Complete annotations · zero pixels are background")
            return
        from ._component_library import library_for
        from ._reviewed_regions import KEY
        truth = self.truth_combo.currentData()
        included = changed = 0
        try:
            library = library_for(truth) if truth is not None else None
            if library is not None:
                _, included, changed = library.counts()
        except Exception as error:
            self.annotation_note.setText(str(error))
            return
        mask = self.reviewed_combo.currentData()
        state = mask.metadata.get(KEY, {}) if mask is not None else {}
        count = len(state.get("areas", [])) if state.get("role") == "reviewed_regions" else 0
        self.annotation_note.setText(
            f"{count:,} reviewed cells · {included:,} selected objects"
            + (f" · {changed:,} need review" if changed else "")
        )

    def _create_reviewed_mask(self):
        from .workspace.service import _require_zarr
        image = self._layer(self.image_combo)
        shape = tuple(array_2d(image.data).shape)
        chosen, _ = QFileDialog.getSaveFileName(self, "Create Reviewed-Region Mask", "reviewed-regions.zarr", "Zarr mask (*.zarr)")
        if not chosen:
            return
        path = Path(chosen).expanduser()
        if path.suffix.lower() != ".zarr":
            path = path.with_suffix(".zarr")
        if path.exists():
            raise ValueError("Choose a new path; existing masks will not be replaced.")
        zarr = _require_zarr()
        data = zarr.open_array(str(path), mode="w-", shape=shape,
                               chunks=tuple(min(512, n) for n in shape), dtype=np.uint8, fill_value=0)
        layer = self.viewer.add_labels(
            data, name="Reviewed regions", scale=image.scale, translate=image.translate,
            rotate=image.rotate, shear=image.shear, affine=image.affine,
        )
        from ._reviewed_regions import associate
        associate(layer, image)
        self.refresh_layers()
        self.reviewed_combo.setCurrentIndex(self.reviewed_combo.findData(layer))
        self._status("Empty reviewed-region mask created. Correct the selected cell in Labels, then choose Finish review under Dataset → Reviewed cells.")

    def _unlock_sources(self):
        if self._locked_layer is not None:
            layer, editable = self._locked_layer
            with suppress(Exception):
                layer.editable = editable
            self._locked_layer = None
        for layer, editable in self._locked_extra_layers:
            with suppress(Exception):
                layer.editable = editable
        self._locked_extra_layers.clear()

    def _lock_sources(self):
        truth = self._layer(self.truth_combo)
        controllers = getattr(self.viewer, "_label_assistant_edit_controllers", ()) or ()
        relevant = [c for c in controllers if not c.closed and (c.source is truth or c.source is self.reviewed_combo.currentData())]
        if any(c.dirty or c.stroke_active for c in relevant):
            raise ValueError("Apply pending ground-truth or reviewed-region edits before preparing or training.")
        layers = [truth]
        if self.annotation_combo.currentData() == "partial":
            layers.append(self._layer(self.reviewed_combo))
        layers += [c.tile for c in relevant if c.tile is not None]
        self._locked_layer = (truth, bool(truth.editable))
        for layer in layers:
            if layer is not truth:
                self._locked_extra_layers.append((layer, bool(layer.editable)))
            layer.editable = False

    def _previous_sampled_training(self, image):
        regions = []
        if self.model is not None:
            for history in self.model.history:
                source = history.get('source', {})
                image_id = image.metadata.get('label_assistant_review', {}).get('id')
                if (source.get('image_id') and source['image_id'] == image_id) or source.get('image_layer') == image.name:
                    locations = history.get('locations', self.model.locations if history.get('workflow') == 'coverage' else [])
                    regions.extend(p for p in locations if p['split'] == 'training')
        return regions

    def _partition_key(self, image, config):
        return (id(image), tuple(array_2d(image.data).shape), config.crop_pixels)

    @staticmethod
    def _index_task(image, truth, config, *, reviewed, validation_blocks, prior_training_regions=(), cancel, validation_fraction=0.25, split_seed=42, validation_method="holdout", folds=5):
        dataset = yield from index_patches(image, truth, config, reviewed=reviewed, validation_blocks=validation_blocks, prior_training_regions=prior_training_regions, cancel=cancel, validation_fraction=validation_fraction, split_seed=split_seed, validation_method=validation_method, folds=folds)
        return dataset

    @staticmethod
    def _prepare_task(image, truth, config, *, cancel):
        crops = yield from prepare_crops(image, truth, config, cancel=cancel)
        return crops

    def _start_preparing(self):
        image = self._layer(self.image_combo)
        truth = self._layer(self.truth_combo)
        image_data, truth_data = array_2d(image.data), array_2d(truth.data)
        if tuple(image_data.shape) != tuple(truth_data.shape):
            raise ValueError("Training image and ground truth must have the same 2D shape.")
        config = self._config()
        partial = self.annotation_combo.currentData() == "partial"
        coverage = self.workflow_combo.currentData() == "coverage"
        if partial and not coverage:
            raise ValueError("Partial annotations require coverage training with a reviewed-region mask.")
        if partial and self.reviewed_combo.currentData() is None:
            raise ValueError("Select a cell in Labels → Local Editing, then choose Finish review under Dataset → Reviewed cells.")
        reviewed_layer = self._layer(self.reviewed_combo) if partial else None
        from ._reviewed_regions import KEY
        if reviewed_layer is not None:
            expected = reviewed_layer.metadata.get(KEY, {}).get("truth_id")
            if expected and truth.metadata.get(KEY, {}).get("id") != expected:
                raise ValueError("Select the Curated ground-truth layer associated with these reviewed areas.")
        if reviewed_layer is truth:
            raise ValueError("Use a separate reviewed-region mask, not the target ground-truth layer.")
        reviewed_data = array_2d(reviewed_layer.data) if reviewed_layer is not None else None
        if partial:
            from ._component_library import library_for, TrainingSupervision
            library = library_for(truth)
            if library is not None and library.state()["pending"]:
                raise ValueError("Update the component library under Dataset → Selected objects after editing Curated.")
            invalid = reviewed_layer.metadata.get(KEY, {}).get("invalid_regions", []) if reviewed_layer else []
            review_state = reviewed_layer.metadata.get(KEY, {}) if reviewed_layer is not None else {}
            known_areas = review_state.get("areas") if review_state.get("role") == "reviewed_regions" else None
            reviewed_data = TrainingSupervision(truth_data, reviewed_data, library=library,
                                                invalid_regions=invalid, reviewed_areas=known_areas)
        if reviewed_data is not None and tuple(reviewed_data.shape) != tuple(image_data.shape):
            raise ValueError("Reviewed-region mask must match the image and ground-truth shape.")
        signature, generation = self._signature(), self._generation
        self._status("Preparing the dataset and validation partitions. Patch preview is optional; training starts only when you choose Train.")
        self._lock_sources()
        try:
            if coverage:
                partition_key = self._partition_key(image, config)
                partition = self._coverage_splits.get(partition_key)
                self._start_worker(
                    self._index_task,
                    lambda result: self._index_done(result, image, truth, signature, generation, partition_key),
                    image_data, truth_data, config, reviewed=reviewed_data, validation_blocks=partition,
                    prior_training_regions=self._previous_sampled_training(image),
                    validation_fraction=self.holdout_spin.value()/100, split_seed=self.validation_seed_spin.value(),
                    validation_method=self.validation_method_combo.currentData(), folds=self.fold_spin.value(),
                )
                return
            self._start_worker(
                self._prepare_task,
                lambda result: self._preparation_done(result, image, truth, signature, generation),
                image_data, truth_data, config,
            )
        except Exception:
            self._unlock_sources()
            raise

    def _index_done(self, result, image, truth, signature, generation, partition_key):
        if generation != self._generation or signature != self._signature() or not all(any(layer is item for item in self.viewer.layers) for layer in (image, truth)):
            self._status("Source changed during indexing. Prepare again.")
            return
        dataset = result
        self.prepared = dataset
        self._prepared_signature = signature
        self._coverage_splits[partition_key] = [b['id'] for b in dataset.blocks if b['split'] == 'validation']
        summary = dataset.summary()
        self.crop_summary_label.setText(
            f"Indexed {len(dataset.locations):,} patches · supervised area {dataset.reviewed_pixels / dataset.total_pixels:.1%} of image\n"
            f"Training: {summary['training']['total']:,} patches ({summary['training']['positive']:,} with targets)\n"
            f"Validation: {summary['validation']['total']:,} patches in reserved spatial blocks\n"
            "Optional patch preview shows representative examples. Training uses the entire index."
        )
        self._update_annotation_note()
        self._update_validation_controls(False)
        self.validation_results_label.setText("Dataset indexed. Validation metrics will appear after the first evaluated epoch.")
        self._status("Dataset ready. Train now or inspect the optional patch preview. Every training patch is visited each epoch; validation regions are excluded from training.")

    def _preparation_done(self, result, image, truth, signature, generation):
        if generation != self._generation or not all(any(layer is item for item in self.viewer.layers) for layer in (image, truth)):
            self._status("Source changed during preparation; crops discarded. Prepare again.")
            return
        crops = result
        self.prepared = crops
        self._prepared_signature = signature
        counts = crop_summary(crops)
        self.crop_summary_label.setText("\n".join(
            f"{split.title()}: {c['total']} crops · {c['positive']} positive · {c['background']} background ({c['nearby_background']} nearby)."
            for split, c in counts.items()
        ))
        extra = " Some requested negatives contain foreground; inspect the actual counts." if any(p["kind"] == "mixed context" for p in crops.locations) else ""
        self._status("Dataset ready. Train now or inspect the optional patch preview. Double-click a preview tile to locate it in the source." + extra)

    @staticmethod
    def _patch_preview_task(prepared, *, cancel):
        if cancel.is_set():
            raise QuickTrainCancelled()
        crops = preview_index(prepared, cancel=cancel) if isinstance(prepared, IndexedPatches) else prepared
        return build_crop_atlas(crops), getattr(crops, "validity", None)

    def _start_patch_preview(self):
        if self.prepared is None or self._signature() != self._prepared_signature:
            raise ValueError("Prepare the dataset before opening its patch preview.")
        image, truth = self._layer(self.image_combo), self._layer(self.truth_combo)
        generation = self._generation
        self._lock_sources()
        def done(result):
            if generation != self._generation:
                self._status("Dataset changed; preview discarded. Prepare again.")
                return
            atlas, validity = result
            self._show_crop_grid(atlas, image, truth, validity=validity)
            self._status("Patch preview: a separate contact sheet beside the image. Cyan frames are training; orange frames are validation. Double-click a tile to locate its source, or Return to image.")
        self._start_worker(self._patch_preview_task, done, self.prepared)

    def _return_to_image(self):
        for layer in self._crop_preview_layers:
            layer.visible = False
        if self._gallery_return_view is not None:
            center, zoom, active = self._gallery_return_view
            self.viewer.camera.center = center
            self.viewer.camera.zoom = zoom
            if active is not None and any(layer is active for layer in self.viewer.layers):
                self.viewer.layers.selection.active = active
        self._gallery_return_view = None
        self._update_controls()
        self._status("Returned to the source image. Prepared training data is retained.")

    def _show_crop_grid(self, atlas, image, truth, *, validity=None):
        if self._gallery_return_view is None:
            self._gallery_return_view = (tuple(self.viewer.camera.center), self.viewer.camera.zoom, self.viewer.layers.selection.active)
        for layer in self._crop_preview_layers:
            if any(layer is item for item in self.viewer.layers):
                self.viewer.layers.remove(layer)
        self._crop_preview_layers = []
        origin = np.asarray(image.data_to_world((0, 0)), dtype=float)
        scale = tuple(max(float(np.linalg.norm(np.asarray(image.data_to_world(basis)) - origin)), 1e-12) for basis in ((1, 0), (0, 1)))
        world = image.extent.world
        translation = (float(world[0, -2]), float(world[1, -1]) + 64 * scale[1])
        metadata = {
            INTERNAL_LAYER_ROLE_KEY: "quick_train_crops",
            "quick_train_crops": {"image_source": image.name, "mask_source": truth.name,
                                  "tiles": atlas.locations, "read_only_preview": True},
        }
        grid_image = self.viewer.add_image(atlas.image, name="Quick Train — crop images", contrast_limits=(0, 1), translate=translation, scale=scale, metadata=metadata.copy())
        grid_mask = self.viewer.add_labels(atlas.mask, name="Quick Train — crop ground truth", translate=translation, scale=scale, metadata=metadata.copy(), opacity=0.5)
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
            face_color="transparent", translate=translation, scale=scale, metadata=metadata.copy(),
            text={"string": "{caption}", "anchor": "upper_left", "color": "white",
                  "translation": [-size / 2 - 8, -size / 2], "size": 10},
        )
        frames.editable = False
        self._crop_preview_layers = [grid_image, grid_mask, frames]
        if validity is not None and not np.all(validity):
            ignored = np.zeros_like(atlas.mask, dtype=np.uint8)
            for tile, valid in zip(atlas.locations, validity):
                y, x, size = tile['atlas_y0'], tile['atlas_x0'], tile['model_size']
                ignored[y:y + size, x:x + size] = valid == 0
            ignored_layer = self.viewer.add_labels(ignored, name="Training — ignored pixels", translate=translation, scale=scale, metadata=metadata.copy(), opacity=0.25)
            ignored_layer.editable = False
            self._crop_preview_layers.append(ignored_layer)

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
                for preview_layer in self._crop_preview_layers:
                    preview_layer.visible = False
                self._gallery_return_view = None
                self.viewer.layers.selection.active = image
                self._update_controls()

        for layer in self._crop_preview_layers:
            layer.mouse_drag_callbacks.append(lambda layer, event: show_tile(layer, event) if event.type == "mouse_press" else None)
            layer.mouse_double_click_callbacks.append(lambda layer, event: show_tile(layer, event, locate=True))
        center = grid_image.data_to_world((atlas.image.shape[0] / 2, atlas.image.shape[1] / 2))
        self._focus(center, atlas.image.shape[0] * scale[0], atlas.image.shape[1] * scale[1])

    def _focus(self, center, height, width):
        previous = tuple(self.viewer.camera.center)
        self.viewer.camera.center = (*previous[:-2], *center)
        canvas_width, canvas_height = self.viewer.window._qt_viewer.canvas.size
        self.viewer.camera.zoom = max(1e-5, min(canvas_width / width, canvas_height / height) * 0.9)

    def _start_training(self, continue_training=False, final_training=False):
        if self.prepared is None or self._signature() != self._prepared_signature:
            raise ValueError("Prepare the dataset before training. Patch preview is optional.")
        kfold = isinstance(self.prepared, IndexedPatches) and self.validation_method_combo.currentData() == "kfold"
        if continue_training and kfold:
            raise ValueError("K-fold evaluation starts independent models from fresh weights. Continue training is available with spatial holdout.")
        if final_training and (not kfold or self.cross_validation is None):
            raise ValueError("Complete spatial k-fold validation on this dataset before training a final model.")
        self.learning_chart.set_points([])
        self.export_csv_button.setEnabled(False)
        self.validation_results_label.setText("Final training uses all reviewed patches; no held-out validation loss is measured." if final_training else "Training started. Waiting for the first evaluated epoch.")
        if continue_training and self.model is None:
            raise ValueError("Train or load a model before continuing training.")
        if continue_training and isinstance(self.prepared, IndexedPatches) and self.prepared.validation_seen_before:
            raise ValueError("Review new regions for independent validation, or train a new model.")
        task_info = self.models_panel.bind_sources()
        model_task = getattr(self.model, "identity", {}).get("task_id") if self.model is not None else None
        if continue_training and model_task and model_task != task_info["task_id"]:
            raise ValueError("The selected model belongs to another biological task. Select its task before continuing.")
        self._status("Continuing the current model on the prepared dataset." if continue_training else "Training a new model on the prepared dataset.")
        if isinstance(self.prepared, IndexedPatches):
            self._lock_sources()
            generation = self._generation
            source_info = {**task_info, "image_layer": self._layer(self.image_combo).name,
                           "image_id": self._layer(self.image_combo).metadata.get("label_assistant_review", {}).get("id"),
                           "ground_truth_layer": self._layer(self.truth_combo).name,
                           "image_shape": list(self.prepared.image.shape),
                           "reviewed_layer": self.reviewed_combo.currentText() if self.prepared.reviewed is not None else None}
            supervision = self.prepared.reviewed
            if getattr(supervision, "library", None) is not None:
                source_info["component_library"] = {
                    "source_id": supervision.snapshot["source_id"],
                    "revision": supervision.snapshot["revision"],
                    "selected_ids": list(supervision.selected),
                }
            function = train_final if final_training else (train_kfold if kfold else train_indexed)
            done = self._training_done if final_training or not kfold else self._cross_validation_done
            extra = {"cross_validation_report": self.cross_validation.report} if final_training else ({"initial_model": self.model if continue_training else None} if not kfold else {})
            self._start_worker(function,
                lambda result: done(result) if generation == self._generation else self._status("Annotations changed during training; previous model retained."),
                self.prepared, self._training_config(continue_training), epochs=self.epochs_spin.value(),
                extra_targets=self.extra_targets_spin.value() / 100,
                **extra,
                learning_rate=self.learning_rate_spin.value(), source_info=source_info,
                plateau_enabled=self.plateau_group.isChecked(),
                reduce_patience=self.reduce_patience_spin.value(),
                stop_patience=self.stop_patience_spin.value(),
                lr_factor=self.lr_factor_spin.value(), min_delta=self.min_delta_spin.value())
            return
        self._start_worker(train_prepared_unet, self._training_done, self.prepared, self._training_config(continue_training),
                           initial_model=self.model if continue_training else None,
                           learning_rate=self.learning_rate_spin.value(),
                           source_info={**task_info, "image_layer": self._layer(self.image_combo).name,
                                        "ground_truth_layer": self._layer(self.truth_combo).name})

    def _start_comparison(self):
        if not isinstance(self.prepared, IndexedPatches) or self._signature() != self._prepared_signature:
            raise ValueError("Prepare the coverage dataset before comparing models.")
        from ._model_comparison import compare_models
        from ._model_providers import ModelSpec, provider_available
        specs = list(self.models_panel.candidates)
        if len(specs) < 2:
            raise ValueError("Add at least two configurations in Models.")
        for spec in specs:
            if not provider_available(spec["provider"]):
                raise ValueError("Install the SMP optional dependency before running this comparison.")
        source_info = self.models_panel.bind_sources()
        source_info.update(image_layer=self._layer(self.image_combo).name, ground_truth_layer=self._layer(self.truth_combo).name)
        self.learning_chart.set_points([])
        self.validation_results_label.setText("Model comparison running on shared spatial splits.")
        generation = self._generation
        self._lock_sources()
        self._start_worker(compare_models,
            lambda result: self._comparison_done(result) if generation == self._generation else self._status("Dataset changed; comparison discarded."),
            self.prepared, self._config(), specs, source_info=source_info,
            epochs=self.epochs_spin.value(), extra_targets=self.extra_targets_spin.value()/100,
            learning_rate=self.learning_rate_spin.value(), plateau_enabled=self.plateau_group.isChecked(),
            reduce_patience=self.reduce_patience_spin.value(), stop_patience=self.stop_patience_spin.value(),
            lr_factor=self.lr_factor_spin.value(), min_delta=self.min_delta_spin.value())

    def _comparison_done(self, result):
        from ._model_project import remember_model, project
        state = project(self.viewer)
        task_id = self.models_panel.ensure_task()["id"]
        ids = []
        for model in result.models:
            row = remember_model(self.viewer, model, task_id=task_id, select=False)
            ids.append(row["id"])
        for row in result.report["results"]:
            row["model_ids"] = [ids[i] for i in row["model_indices"]]
        state["comparisons"].append(result.report)
        self.models_panel.refresh()
        self.validation_results_label.setText("Comparison complete. See Models for results; current model retained.")
        self._status("Comparison complete. Save workspace to keep candidate checkpoints and results.")

    def _training_done(self, model, register=True):
        self.model = model
        if register:
            from ._model_project import remember_model
            remember_model(self.viewer, model, task_id=self.models_panel.ensure_task()["id"], name=self.models_panel.name.text().strip() or None)
        self.models_panel.refresh()
        self._update_model_summary()
        if isinstance(self.prepared, IndexedPatches) and model.history and model.history[-1].get("mode") == "new":
            self.prepared.validation_seen_before = False
        self.model_state_label.setText("Model updated in memory — Save workspace or Export current model to keep these weights.")
        self._show_metrics()
        if model.metrics:
            text = f"Retained checkpoint: Dice {model.metrics.get('dice',0):.1%} · IoU {model.metrics.get('iou',0):.1%}"
            if 'loss' in model.metrics:
                text += f" · validation loss {model.metrics['loss']:.4f}"
            self.validation_results_label.setText(text)
        else:
            self.validation_results_label.setText("Final model trained on all reviewed patches. Historical fold results are retained; these weights have no holdout score.")
        stopped = model.history[-1].get("stop_reason") if model.history else None
        prefix = "Early stop; best validation checkpoint retained. " if stopped else ""
        self._status(prefix + f"Model ready ({model.training_device.upper()}). Choose a prediction image and click Predict current area. Preview masks need review.")

    def _show_metrics(self):
        from ._training_report import history_points
        self.learning_chart.set_points(history_points(self.model.history))
        self.export_csv_button.setEnabled(bool(self.learning_chart.points))
        metrics = self.model.metrics
        if not metrics:
            self.metrics_label.setText("Final model trained on all reviewed patches. No holdout metrics were measured for these weights; cross-validation results are stored in history.")
            report = self.model.history[-1].get('cross_validation_report', {}) if self.model.history else {}
            if report.get('aggregate'):
                self.metrics_label.setText(self.metrics_label.text() + "\nPrior fold evaluation: " + " · ".join(f"{key.title()} {report['aggregate'][key]['mean']:.1%} ± {report['aggregate'][key]['std']:.1%}" for key in ('dice','iou')))
            return
        self.metrics_label.setText(
            f"Training: {self.model.training_rounds} rounds · {self.model.training_steps} total steps\n"
            f"Validation: Dice {metrics['dice']:.1%} · IoU {metrics['iou']:.1%}\n"
            f"Precision {metrics['precision']:.1%} · Recall {metrics['recall']:.1%}\n"
            f"Source patch: {self.model.crop_pixels} × {self.model.crop_pixels} px. "
            "These scores do not measure accuracy on a new image."
        )
        if self.model.history and self.model.history[-1].get('workflow') == 'coverage':
            history = self.model.history[-1]
            self.metrics_label.setText(self.metrics_label.text() +
                f"\nCoverage: {history['epochs']} complete epoch(s). Retained checkpoint: " +
                (f"epoch {history['retained_epoch']}" if history['retained_epoch'] else "starting model"))
            if history.get('stop_reason'):
                self.metrics_label.setText(self.metrics_label.text() + "\nEarly stop: " + history['stop_reason'])
            if history.get('learning_rate_events'):
                self.metrics_label.setText(self.metrics_label.text() + f"\nLearning rate reduced to {history['learning_rate_events'][-1]['new_lr']:g}.")

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
            self._start_worker(self._save_task, lambda _result: self._model_saved(path), self.model, path)

    def _model_saved(self, path):
        from ._model_project import project
        identifier = getattr(self.model, "identity", {}).get("model_id")
        row = next((r for r in project(self.viewer)["models"] if r["id"] == identifier), None)
        if row is not None:
            revision = project(self.viewer).get("current_checkpoint") or row["revision"]
            entry = row if revision == row["revision"] else next((r for r in row.get("runs", []) if r["revision"] == revision), None)
            if entry is not None:
                entry.update(checkpoint=str(Path(path).resolve()), saved_revision=revision)
                for run in row.get("runs", []):
                    if run["revision"] == revision:
                        run.update(checkpoint=entry["checkpoint"], saved_revision=revision)
        self.models_panel.refresh()
        self.model_state_label.setText(f"Model saved: {path}")
        self._status(f"Model saved to {path}; training history is embedded and exported beside it as .history.json.")

    @staticmethod
    def _save_task(model, path, *, cancel):
        if cancel.is_set():
            raise QuickTrainCancelled()
        save_model(model, path)
        return path

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Quick Train model", "", "Quick Train model (*.pt)")
        if path:
            self._start_worker(self._load_task, lambda model: self._model_loaded(model, path), path)

    def _model_loaded(self, model, path, register=True):
        self._invalidate_crops()
        image = self.image_combo.currentData()
        if image is not None:
            for key in list(self._coverage_splits):
                if key[0] == id(image):
                    del self._coverage_splits[key]
        self._training_done(model, register=False)
        if register:
            from ._model_project import remember_model
            remember_model(self.viewer, model, task_id=self.models_panel.ensure_task()["id"], path=path)
            from ._model_project import project
            project(self.viewer)["current_task"] = model.identity.get("task_id")
            self.models_panel.refresh()
        self.model_state_label.setText(f"Loaded model: {path}")
        config = model.config
        self.rotation_check.setChecked(config.augment_rotations)
        self.flip_check.setChecked(config.augment_flips)
        self.intensity_check.setChecked(config.augment_intensity)
        self.contrast_spin.setValue(config.contrast_jitter * 100)
        self.brightness_spin.setValue(config.brightness_jitter * 100)
        for history in reversed(model.history):
            if history.get("workflow") == "coverage":
                plateau = history.get("plateau_control", {})
                self.plateau_group.setChecked(plateau.get("enabled", False))
                self.reduce_patience_spin.setValue(plateau.get("reduce_patience", 10))
                self.stop_patience_spin.setValue(plateau.get("stop_patience", 10))
                self.lr_factor_spin.setValue(plateau.get("factor", 0.5))
                self.min_delta_spin.setValue(plateau.get("min_delta", 0.0001))
                break
        for record in reversed(model.history):
            settings = record.get('validation', {})
            if settings.get('method') in ('holdout', 'kfold'):
                self.validation_method_combo.setCurrentIndex(self.validation_method_combo.findData(settings['method']))
                self.holdout_spin.setValue(round(settings.get('holdout_fraction', .25)*100))
                self.validation_seed_spin.setValue(settings.get('seed', 42))
                self.fold_spin.setValue(settings.get('fold_count', 5) if settings['method'] == 'kfold' else 5)
                break
        for history in reversed(model.history):
            if history.get('workflow') == 'coverage':
                image = self.image_combo.currentData()
                if image is not None and history.get('source', {}).get('image_layer') == image.name and history.get('source', {}).get('image_shape') == list(array_2d(image.data).shape):
                    key = (id(image), tuple(array_2d(image.data).shape), history['config']['crop_pixels'])
                    self._coverage_splits[key] = history['validation_blocks']
                break
        self._status('Model loaded. Prepare the dataset and review examples before continuing training.')

    @staticmethod
    def _load_task(path, *, cancel):
        if cancel.is_set():
            raise QuickTrainCancelled()
        return load_model(path)

    def attach_annotation_widgets(self, components):
        """Share annotation controls with the Labels editor without duplicating state."""
        review = components._reviewed_area_widget
        library = components.component_library_widget
        self.reviewed_area_widget = review
        self.component_library_widget = library
        self.reviewed_cells_hint.hide()
        self.selected_objects_hint.hide()
        self.reviewed_cells_layout.addWidget(review)
        self.selected_objects_layout.addWidget(library)
        review.show()
        library.show()
        review.image_combo.hide()  # Uses the Dataset image selector.
        library.source_layer = lambda: self.truth_combo.currentData()
        library.table.setMaximumHeight(16777215)
        library.layout().setStretchFactor(library.table, 1)

        def sync_sources(*_args):
            image = self.image_combo.currentData()
            image_index = review.image_combo.findData(image)
            if image_index >= 0:
                review.image_combo.setCurrentIndex(image_index)
            truth = self.truth_combo.currentData()
            target_index = components._target_combo.findText(truth.name) if truth is not None else -1
            if review.selected_cell is not None and review.selected_cell[0] is not truth:
                review.clear_cell()
            if target_index >= 0:
                components._target_combo.setCurrentIndex(target_index)
            library.refresh()

        def sync_editor(*_args):
            truth = next((layer for layer in self.viewer.layers
                          if layer.name == components._target_combo.currentText()), None)
            index = self.truth_combo.findData(truth)
            if index >= 0:
                self.truth_combo.setCurrentIndex(index)

        self.image_combo.currentIndexChanged.connect(sync_sources)
        self.truth_combo.currentIndexChanged.connect(sync_sources)
        components._target_combo.currentIndexChanged.connect(sync_editor)
        sync_sources()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        if getattr(self.viewer, "_label_assistant_training_widget", None) is self:
            self.viewer._label_assistant_training_widget = None
        if self.component_library_widget is not None:
            self.component_library_widget.shutdown()
        if self.reviewed_area_widget is not None:
            self.reviewed_area_widget.shutdown()
        with suppress(Exception):
            self.viewer.layers.events.removed.disconnect(self._source_removed)
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
