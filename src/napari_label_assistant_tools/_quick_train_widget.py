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
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)
from superqt.utils import create_worker

from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY, is_internal_layer
from ._mask_tiles import full_data, full_source
from ._quick_train import (
    QuickTrainCancelled, QuickTrainConfig, array_2d, build_crop_atlas, crop_summary,
    load_model, predict_area, predict_region, prediction_labels, prepare_crops, save_model, train_prepared_unet,
)


from ._coverage_train import IndexedPatches, index_patches, index_pairs, preview_index, train_indexed, train_kfold, train_final
from ._patch_subset import select_training_patches, select_trial_validation
from ._activity import activity_for


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
        self.activity = activity_for(viewer)
        self.model = None
        self.cross_validation = None
        self._validation_preview_layer = None
        self.prepared = None
        self._patch_subset_cache = None
        self._prepared_signature = None
        self._dataset_outdated = False
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
        self.dataset_pairs = []
        self.channel_assignments = []
        self.model_state_label = QLabel("No model in memory.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        self.dataset_scroll = scroll
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
        self.source_context = QLabel()
        self.source_context.setWordWrap(True)
        outer.addWidget(self.source_context)
        outer.addWidget(self.workflow_tabs)
        layout = QVBoxLayout(body)
        self.dataset_source_layout = layout
        layout.setContentsMargins(0, 6, 0, 0)
        note = QLabel(
            "Prepare confirmed tiles from Edit & Confirm. Unconfirmed areas are excluded."
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
        self.dataset_source_mode = QComboBox()
        self.dataset_source_mode.addItem('This image', 'current')
        self.dataset_source_mode.addItem('Multiple images', 'multiple')
        self.manage_pairs_button = QPushButton('Manage pairs…')
        self.manage_pairs_button.clicked.connect(lambda: self._safe(self._manage_pairs))
        self.dataset_source_row = QWidget()
        source_row = QHBoxLayout(self.dataset_source_row)
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(self.dataset_source_mode, 1)
        source_row.addWidget(self.manage_pairs_button)
        form.addRow('Source mode', self.dataset_source_row)
        self.pairs_summary = QLabel()
        self.pairs_summary.setWordWrap(True)
        form.addRow(self.pairs_summary)
        self.reviewed_combo = QComboBox()
        self.reviewed_combo.setToolTip(
            "Choose a separate Labels layer marking fully reviewed regions with nonzero pixels. "
            "Outside those regions, pixels are ignored. Mark a region reviewed only after "
            "all target objects there have been annotated; accepted objects alone are not complete ground truth."
        )
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItem("All / selected training patches", "coverage")
        self.workflow_combo.addItem("Sample patches", "sampled")
        self.annotation_combo = QComboBox()
        self.annotation_combo.addItem("Partial — confirmed tiles", "partial")
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
        for combo in (self.image_combo, self.truth_combo, self.reviewed_combo,
                      self.dataset_source_mode, self.workflow_combo, self.annotation_combo):
            combo.setMinimumContentsLength(8)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        form.addRow("Workflow", self.workflow_combo)
        form.addRow("Coverage", self.annotation_combo)
        form.addRow("Image", self.image_combo)
        form.addRow("Ground truth", self.truth_combo)
        form.addRow(self.create_reviewed_button)
        form.addRow("Source patch size", self.crop_combo)
        form.addRow("Model input size", self.resolution_combo)
        form.addRow("Crop count", self.samples_spin)
        self.dataset_form = form
        self.dataset_advanced_toggle = QToolButton()
        self.dataset_advanced_toggle.setText('Advanced source options')
        self.dataset_advanced_toggle.setCheckable(True)
        self.dataset_advanced_toggle.toggled.connect(self._update_dataset_advanced_visibility)
        layout.addWidget(self.dataset_advanced_toggle)
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
        overview = QGroupBox("Training data")
        overview_layout = QVBoxLayout(overview)
        self.annotation_sources_label = QLabel()
        self.supervision_label = QLabel()
        self.dataset_state_label = QLabel("No eligible annotations")
        for label in (self.annotation_sources_label, self.supervision_label, self.dataset_state_label):
            label.setWordWrap(True)
            overview_layout.addWidget(label)
        layout.addWidget(overview)
        from ._annotation_batch_widget import AnnotationBatchWidget
        self.batch_widget = AnnotationBatchWidget(viewer, self)
        layout.addWidget(self.batch_widget)
        self.patch_selection_group = QGroupBox('Training patch selection')
        self.patch_selection_form = QFormLayout(self.patch_selection_group)
        self.patch_selection_mode = QComboBox()
        for text, value in (('All eligible training patches', 'all'), ('Percentage of training patches', 'percentage'),
                            ('Number of training patches', 'count')):
            self.patch_selection_mode.addItem(text, value)
        self.patch_percentage_spin = QDoubleSpinBox()
        self.patch_percentage_spin.setRange(.1, 100)
        self.patch_percentage_spin.setDecimals(1)
        self.patch_percentage_spin.setValue(10)
        self.patch_percentage_spin.setSuffix(' %')
        self.patch_count_spin = QSpinBox()
        self.patch_count_spin.setRange(1, 1000000000)
        self.patch_count_spin.setValue(100)
        self.patch_count_spin.setToolTip('Unique training patches in the fixed trial subset. Extra target-patch visits are configured separately in Train.')
        self.patch_subset_seed = QSpinBox()
        self.patch_subset_seed.setRange(0, 1000000)
        self.patch_subset_seed.setValue(42)
        self.patch_subset_seed.setToolTip('Choose a reproducible training subset. Changing this seed never changes the reserved validation patches.')
        for title, control in (('Use', self.patch_selection_mode), ('Percentage', self.patch_percentage_spin),
                               ('Patch count', self.patch_count_spin), ('Subset seed', self.patch_subset_seed)):
            self.patch_selection_form.addRow(title, control)
        self.quick_trial_check = QCheckBox('Quick trial — smaller validation set')
        self.quick_trial_check.setToolTip('Use a fixed sample of the existing reserved validation pool. The spatial split stays unchanged. Evaluate the full validation set before accepting a model.')
        self.trial_validation_percentage = QDoubleSpinBox()
        self.trial_validation_percentage.setRange(.1, 100)
        self.trial_validation_percentage.setDecimals(1)
        self.trial_validation_percentage.setValue(10)
        self.trial_validation_percentage.setSuffix(' %')
        self.trial_validation_percentage.setToolTip('Percentage of reserved validation patches. The sample is fixed across epochs and training-subset changes; it follows Validation → Split seed.')
        self.patch_selection_form.addRow(self.quick_trial_check)
        self.patch_selection_form.addRow('Validation percentage', self.trial_validation_percentage)
        self.patch_selection_summary = QLabel()
        self.patch_selection_summary.setWordWrap(True)
        self.patch_selection_form.addRow(self.patch_selection_summary)
        layout.addWidget(self.patch_selection_group)
        self.training_tile_actions = QGroupBox('Confirmed tiles for dataset')
        tile_layout = QVBoxLayout(self.training_tile_actions)
        self.edit_confirm_button = QPushButton('Edit && confirm tiles…')
        self.edit_confirm_button.setEnabled(False)
        tile_layout.addWidget(self.edit_confirm_button)
        self.show_confirmed_tiles_button = QPushButton('Show confirmed tiles on image')
        self.show_confirmed_tiles_button.setEnabled(False)
        tile_layout.addWidget(self.show_confirmed_tiles_button)
        self.show_dataset_split_button = QPushButton('Show training / validation patches on image')
        self.show_dataset_split_button.setEnabled(False)
        self.show_dataset_split_button.clicked.connect(lambda: self._safe(self._preview_validation))
        tile_layout.addWidget(self.show_dataset_split_button)
        tile_hint = QLabel('Confirm reviewed objects and background in Edit & Confirm, then prepare the dataset here. Unconfirmed areas remain excluded.')
        tile_hint.setWordWrap(True)
        tile_layout.addWidget(tile_hint)
        layout.addWidget(self.training_tile_actions)
        self.legacy_tile_controls = QGroupBox('Advanced: confirm loaded edit areas')
        legacy_layout = QVBoxLayout(self.legacy_tile_controls)
        tile_form = QFormLayout()
        self.review_tile_height = QSpinBox()
        self.review_tile_width = QSpinBox()
        for spin in (self.review_tile_height, self.review_tile_width):
            spin.setRange(1, 100000)
            spin.setSuffix(' px')
            spin.setValue(1024)
        tile_form.addRow('Training tile height', self.review_tile_height)
        tile_form.addRow('Training tile width', self.review_tile_width)
        legacy_layout.addLayout(tile_form)
        self.confirm_tiles_on_canvas = QCheckBox('Click tiles to confirm reviewed masks for dataset')
        self.confirm_tiles_on_canvas.setToolTip('Clicking approves every object and the remaining background inside the chosen grid tile. Only confirm tiles you have checked completely. Turn off to paint again.')
        self.confirm_tiles_on_canvas.setEnabled(False)
        legacy_layout.addWidget(self.confirm_tiles_on_canvas)
        self.cleanup_masks_button = QPushButton('Optional: find small masks in current edit area…')
        self.cleanup_masks_button.setEnabled(False)
        self.cleanup_masks_button.setToolTip('Index connected objects in the loaded edit area, then preview an area threshold before removing small masks. This does not clean the entire image.')
        legacy_layout.addWidget(self.cleanup_masks_button)
        self.select_training_tiles_button = QPushButton('Select checked tiles on canvas…')
        self.confirm_training_tiles_button = QPushButton('Confirm edited / selected tiles for training')
        for button in (self.select_training_tiles_button, self.confirm_training_tiles_button):
            button.setEnabled(False)
            legacy_layout.addWidget(button)
        self.selection_toggle = QToolButton()
        self.selection_toggle.setText("Advanced: saved object selections and coverage")
        self.selection_toggle.setCheckable(True)
        layout.addWidget(self.selection_toggle)
        self.training_selection_panel = QWidget()
        self.training_selection_layout = QVBoxLayout(self.training_selection_panel)
        self.training_selection_layout.setContentsMargins(0, 0, 0, 0)
        self.selection_form = QFormLayout()
        self.selection_form.addRow("Reviewed-region mask", self.reviewed_combo)
        self.training_selection_layout.addLayout(self.selection_form)
        self.selection_hint = QLabel("Existing foreground-only selections are preserved here. For new training data, confirm complete tiles in Edit & Confirm.")
        self.selection_hint.setWordWrap(True)
        self.training_selection_layout.addWidget(self.selection_hint)
        self.training_selection_layout.addWidget(self.legacy_tile_controls)
        self.training_selection_panel.hide()
        self.selection_toggle.toggled.connect(self._update_source_mode_visibility)
        self.review_pair_toggle = QToolButton()
        self.review_pair_toggle.setText('Review selected pair')
        self.review_pair_toggle.setCheckable(True)
        self.review_pair_toggle.toggled.connect(self._update_source_mode_visibility)
        self.training_selection_layout.insertWidget(0, self.review_pair_toggle)
        layout.addWidget(self.training_selection_panel)
        layout.addStretch(0)
        prepare_layout.insertWidget(0, self.annotation_note)
        self.dataset_feedback = QLabel()
        self.dataset_feedback.setWordWrap(True)
        prepare_layout.addWidget(self.dataset_feedback)
        self.smaller_patches_button = QPushButton()
        self.smaller_patches_button.hide()
        self.smaller_patches_button.setToolTip('Keep the confirmed tile selection and retry preparation using smaller source and model patches. Validation still requires separate spatial groups.')
        self.smaller_patches_button.clicked.connect(lambda: self._safe(self._prepare_smaller_patches))
        prepare_layout.addWidget(self.smaller_patches_button)
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
        from ._cluster_widget import TrainingComputeWidget
        self.training_compute = TrainingComputeWidget(self)
        train_layout.addWidget(self.training_compute)
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
        self.full_validation_button = QPushButton('Evaluate full validation set')
        self.full_validation_button.clicked.connect(lambda: self._safe(self._start_full_validation))
        validation_form.addRow(self.full_validation_button)
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
        inference_intro = QLabel("Train or load a model, choose an image, then predict a current area, the entire image, or a drawn box.")
        inference_intro.setWordWrap(True)
        inference_layout.addWidget(inference_intro)
        self.inference_classes_label = QLabel()
        self.inference_classes_label.setWordWrap(True)
        inference_layout.addWidget(self.inference_classes_label)
        self.inference_load_button = QPushButton("Load model…")
        self.inference_load_button.clicked.connect(lambda: self._safe(self._load))
        inference_layout.addWidget(self.inference_load_button)
        preview_group = QGroupBox("Predict")
        preview_form = QFormLayout(preview_group)
        self.prediction_combo = QComboBox()
        self.prediction_scope = QComboBox()
        self.prediction_scope.addItem("Current area (one crop)", "area")
        self.prediction_scope.addItem("Entire image", "image")
        self.prediction_scope.addItem("Drawn box", "box")
        self.prediction_scope.currentIndexChanged.connect(self._update_prediction_scope)
        self.prediction_box_layer = None
        self.draw_prediction_box_button = QPushButton("Draw prediction box")
        self.draw_prediction_box_button.clicked.connect(lambda: self._safe(self._draw_prediction_box))
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
        preview_form.addRow("Predict", self.prediction_scope)
        preview_form.addRow(self.draw_prediction_box_button)
        preview_form.addRow("Threshold", self.threshold_spin)
        preview_form.addRow(self.predict_button)
        preview_note = QLabel(
            "Current area predicts one crop at the canvas center. Entire image and "
            "drawn box use overlapping tiles with progress and Cancel. Each run adds "
            "a new Labels layer for review. A box limits the region; it does not "
            "teach this model which object to segment."
        )
        preview_note.setWordWrap(True)
        preview_form.addRow(preview_note)
        self._update_prediction_scope()
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
        self.readiness_label = QLabel()
        self.readiness_label.setWordWrap(True)
        outer.addWidget(self.readiness_label)
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
        self.models_panel.class_names_edit.textChanged.connect(self._invalidate_crops)
        self.models_panel.output_mode.currentIndexChanged.connect(self._output_mode_changed)
        for control in (self.epochs_spin, self.batch_spin, self.extra_targets_spin):
            control.valueChanged.connect(self._update_workload)
        self.patch_selection_mode.currentIndexChanged.connect(self._patch_selection_changed)
        self.quick_trial_check.toggled.connect(self._quick_trial_changed)
        self.trial_validation_percentage.valueChanged.connect(self._patch_selection_changed)
        for control in (self.patch_percentage_spin, self.patch_count_spin, self.patch_subset_seed):
            control.valueChanged.connect(self._patch_selection_changed)
        self.workflow_combo.currentIndexChanged.connect(self._workflow_changed)
        self.annotation_combo.currentIndexChanged.connect(self._annotation_changed)
        self.reviewed_combo.currentIndexChanged.connect(self._sources_changed)
        self.dataset_source_mode.currentIndexChanged.connect(self._dataset_mode_changed)
        self.image_combo.currentIndexChanged.connect(self._sources_changed)
        self.truth_combo.currentIndexChanged.connect(self._sources_changed)
        self.destroyed.connect(lambda *_args: self.shutdown())
        # Reuse the plugin's layer lifecycle handling without importing PyTorch.
        from ._widget import _auto_refresh_layer_controls

        self._layer_refresh_timer = _auto_refresh_layer_controls(viewer, self, self.refresh_layers)
        self.viewer.layers.events.removed.connect(self._source_removed)
        self.refresh_layers()
        self._output_mode_changed()
        self._update_model_summary()
        from ._model_project import project
        settings = project(self.viewer).get("settings")
        if settings:
            self.restore_project_settings(settings)
        from ._annotation_batches import events as batch_events
        self._batch_events = batch_events(viewer)
        self._batch_events.changed.connect(self._batches_changed)
        self._safe(self.models_panel.restore_current_model)

    def _batch_pairs(self):
        from ._annotation_batches import _id
        if self._is_multilabel():
            identifiers = {(row['image_id'], item.get('target_id')) for row in self.channel_assignments if row.get('included')
                           for item in row.get('channels', {}).values() if item.get('coverage') != 'unknown'}
        elif self._multiple_pairs():
            identifiers = {(row['image_id'], row.get('truth_id')) for row in self.dataset_pairs if row.get('included')}
        else:
            return [(self.image_combo.currentData(), self.truth_combo.currentData())]
        by_id = {_id(layer): layer for layer in self.viewer.layers if _id(layer)}
        return [(by_id.get(i), by_id.get(t)) for i,t in sorted(identifiers, key=str)]

    def _batch_signatures(self):
        from ._annotation_batches import signature
        result = []
        for i,t in self._batch_pairs():
            if i is not None and t is not None:
                try:
                    value = signature(i,t)
                except ValueError as error:
                    value = ('invalid batch', str(error))
                result.append((id(i),id(t),value))
        return tuple(result)

    def _batch_contexts(self):
        from ._annotation_batches import selection
        return [selection(i,t) for i,t in self._batch_pairs() if i is not None and t is not None]

    def batch_preparation_error(self):
        from ._annotation_batches import preparation_error
        for image,truth in self._batch_pairs():
            error = preparation_error(image, truth)
            if error:
                return error
        return ''

    def _batches_changed(self, *_args):
        if self.closed:
            return
        if self.prepared is not None and self._prepared_signature != self._signature():
            self._invalidate_crops()
        else:
            self._update_controls()

    def _protect_batch_holdouts(self, dataset):
        if self.model is not None and dataset.validation.get('method') != 'kfold' and any(c['scope'] == 'active' for c in self._batch_contexts()):
            from ._annotation_batches import protect_validation, _id
            image = self.image_combo.currentData()
            dataset = protect_validation(dataset, self.model, image_id=_id(image), image_name=image.name if image is not None else '')
        return dataset

    def project_settings(self):
        combo_names = ("workflow_combo", "annotation_combo", "crop_combo", "resolution_combo", "device_combo", "validation_method_combo", "dataset_source_mode", "patch_selection_mode")
        spin_names = ("samples_spin", "steps_spin", "seed_spin", "epochs_spin", "batch_spin", "extra_targets_spin", "learning_rate_spin", "holdout_spin", "fold_spin", "validation_seed_spin", "contrast_spin", "brightness_spin", "reduce_patience_spin", "stop_patience_spin", "lr_factor_spin", "min_delta_spin", "patch_percentage_spin", "patch_count_spin", "patch_subset_seed", "trial_validation_percentage")
        check_names = ("rotation_check", "flip_check", "intensity_check", "plateau_group", "quick_trial_check")
        from ._model_project import layer_id
        return {"combos": {name: getattr(self,name).currentData() for name in combo_names},
                "values": {name: getattr(self,name).value() for name in spin_names},
                "checks": {name: getattr(self,name).isChecked() for name in check_names},
                "sources": {name: layer_id(getattr(self,name).currentData()) for name in ("image_combo", "truth_combo", "reviewed_combo", "prediction_combo")},
                "new_model": self.models_panel.spec().to_dict(), "model_name": self.models_panel.name.text(),
                "class_names_text": self.models_panel.class_names_edit.text(),
                "dataset_pairs": self.dataset_pairs,
                'channel_assignments': self.channel_assignments,
                'multilabel': self.models_panel.is_multilabel(),
                "comparison_candidates": self.models_panel.candidates}

    def restore_project_settings(self, settings):
        self.quick_trial_check.setChecked(bool(settings.get('checks', {}).get('quick_trial_check', False)))
        self.dataset_pairs = list(settings.get('dataset_pairs', []))
        self.channel_assignments = list(settings.get('channel_assignments', []))
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
        panel.output_mode.setCurrentIndex(panel.output_mode.findData(bool(settings.get('multilabel', False))))
        if "class_names_text" in settings:
            panel.class_names_edit.setText(settings["class_names_text"])
        panel.candidates=list(settings.get("comparison_candidates",[]))
        panel.candidate_list.clear()
        from ._model_providers import ModelSpec
        for spec in panel.candidates:
            panel.candidate_list.addItem(ModelSpec(**spec).label())
        self.model_state_label.setText("Workspace restored. Select its saved model in Models to continue or predict.")
        self._use_single_included_pair()
        self._update_controls()

    def _multiple_pairs(self):
        return self.dataset_source_mode.currentData() == 'multiple'

    def _is_multilabel(self):
        return hasattr(self, 'models_panel') and self.models_panel.is_multilabel()

    def _output_mode_changed(self, *_args):
        multilabel = self._is_multilabel()
        self.dataset_source_mode.model().item(0).setEnabled(not multilabel)
        self.dataset_source_mode.setItemText(1, 'Images / target channels' if multilabel else 'Multiple images')
        if multilabel:
            self.dataset_source_mode.setCurrentIndex(1)
        self._workflow_changed()
        self._sources_changed()

    def _dataset_mode_changed(self, *_args):
        if self._is_multilabel() and not self._multiple_pairs():
            self.dataset_source_mode.setCurrentIndex(1)
            return
        self._use_single_included_pair()
        self._workflow_changed()
        self._sources_changed()

    def _use_single_included_pair(self):
        """A single selected image uses spatial validation, including old projects."""
        if not self._multiple_pairs() or self._is_multilabel():
            return
        included = [row for row in self.dataset_pairs if row.get('included')]
        if len(included) != 1:
            return
        from ._dataset_pairs import associate_pair_coverage, source_layers, is_prediction
        associate_pair_coverage(self.viewer, self.dataset_pairs)
        row = included[0]
        if not row.get('image_id') or not row.get('truth_id'):
            return
        images, labels, masks = source_layers(self.viewer)
        def matches(layer, identifier):
            return layer.metadata.get('label_assistant_review', {}).get('id') == identifier
        image = next((layer for layer in images if matches(layer, row.get('image_id'))), None)
        truth = next((layer for layer in labels if matches(layer, row.get('truth_id'))), None)
        if image is None or truth is None or row.get('coverage') not in ('partial', 'complete'):
            return  # Keep invalid definitions visible to pair validation.
        if is_prediction(truth) and not row.get('confirmed'):
            return
        if row['coverage'] == 'partial':
            candidates = [mask for mask in masks
                          if mask.metadata['label_assistant_review'].get('image_id') == row['image_id']
                          and mask.metadata['label_assistant_review'].get('truth_id') == row['truth_id']]
            if row.get('reviewed_id'):
                if not any(matches(mask, row['reviewed_id']) for mask in candidates):
                    return
            elif len(candidates) > 1:
                return
        self.dataset_source_mode.blockSignals(True)
        self.dataset_source_mode.setCurrentIndex(self.dataset_source_mode.findData('current'))
        self.dataset_source_mode.blockSignals(False)
        for combo, key in ((self.image_combo, 'image_id'), (self.truth_combo, 'truth_id'), (self.reviewed_combo, 'reviewed_id')):
            for index in range(combo.count()):
                layer = combo.itemData(index)
                if layer is not None and layer.metadata.get('label_assistant_review', {}).get('id') == row.get(key):
                    combo.setCurrentIndex(index)
                    break
        self.annotation_combo.setCurrentIndex(self.annotation_combo.findData(row.get('coverage', 'partial')))

    def _update_source_mode_visibility(self, *_args):
        multiple = self._multiple_pairs()
        reviewing = not multiple or self.review_pair_toggle.isChecked()
        self.review_pair_toggle.setVisible(multiple)
        self.review_pair_toggle.setText('Review selected mask' if self._is_multilabel() else 'Review selected pair')
        self.manage_pairs_button.setText('Manage channels…' if self._is_multilabel() else 'Manage pairs…')
        self.training_selection_panel.setVisible(self.selection_toggle.isChecked())
        if self.component_library_widget is not None:
            self.component_library_widget.setVisible(reviewing)
        self.selection_hint.setVisible(self.component_library_widget is None or (multiple and not reviewing))
        self.dataset_source_layout.setStretch(self.dataset_source_layout.count() - 1, 0 if reviewing else 1)
        for control in (self.image_combo, self.truth_combo, self.annotation_combo):
            control.setVisible(reviewing)
            self.dataset_form.labelForField(control).setVisible(reviewing)
        self.pairs_summary.setVisible(multiple)
        if self._is_multilabel():
            from ._multilabel_targets import resolve_channels
            names = tuple(part.strip() for part in self.models_panel.class_names_edit.text().split(',') if part.strip())
            included = [row for row in self.channel_assignments if row.get('included')]
            issues = []
            for row in included:
                try:
                    resolve_channels(self.viewer, names, [row])
                except ValueError as error:
                    issues.append(str(error))
            detail = f'{len(issues)} need attention' if issues else 'Ready to prepare' if included and names else 'Open Manage channels to name features and assign masks'
            image_word = 'image' if len(included) == 1 else 'images'
            channel_word = 'channel' if len(names) == 1 else 'channels'
            self.pairs_summary.setText(f'{len(included)} {image_word} · {len(names)} {channel_word} · {detail}')
            self.pairs_summary.setToolTip('\n'.join(issues))
            self.reviewed_combo.setVisible(reviewing and self.annotation_combo.currentData() == 'partial')
            self.selection_form.labelForField(self.reviewed_combo).setVisible(reviewing and self.annotation_combo.currentData() == 'partial')
            self._update_dataset_advanced_visibility()
            return
        included = [row for row in self.dataset_pairs if row.get('included')]
        from ._dataset_pairs import resolve_pairs
        issues = []
        for row in included:
            try:
                resolve_pairs(self.viewer, [row], minimum=1)
            except ValueError as error:
                issues.append(str(error))
        detail = f'{len(issues)} need attention' if issues else 'Ready to prepare' if len(included) >= 2 else 'Choose at least two pairs'
        self.pairs_summary.setText(f'{len(included)} pairs selected · {detail}')
        self.pairs_summary.setToolTip('\n'.join(issues))
        self.reviewed_combo.setVisible(reviewing and self.annotation_combo.currentData() == 'partial')
        self.selection_form.labelForField(self.reviewed_combo).setVisible(reviewing and self.annotation_combo.currentData() == 'partial')
        self._update_dataset_advanced_visibility()

    def _update_dataset_advanced_visibility(self, *_args):
        advanced = self.dataset_advanced_toggle.isChecked()
        for control in (self.workflow_combo, self.samples_spin):
            visible = advanced and (control is not self.samples_spin or self.workflow_combo.currentData() != 'coverage')
            control.setVisible(visible)
            self.dataset_form.labelForField(control).setVisible(visible)
        source_visible = advanced or self._multiple_pairs() or self._is_multilabel()
        self.dataset_source_row.setVisible(source_visible)
        self.dataset_form.labelForField(self.dataset_source_row).setVisible(source_visible)

    def _manage_pairs(self):
        if self._is_multilabel():
            from ._multilabel_targets import MultilabelTargetsDialog
            names = tuple(part.strip() for part in self.models_panel.class_names_edit.text().split(',') if part.strip())
            dialog = MultilabelTargetsDialog(self.viewer, names, self.channel_assignments, self)
            if dialog.exec_():
                self.models_panel.apply_channel_names(dialog.names)
                self.channel_assignments = dialog.assignments()
                self._sources_changed()
                self._status('Channel names and masks saved in the current setup. Check annotation coverage, then Prepare dataset. Save the workspace to keep these assignments.')
            return
        from ._dataset_pairs import DatasetPairsDialog
        dialog = DatasetPairsDialog(self.viewer, self.dataset_pairs, self)
        if not dialog.exec_():
            return
        self.dataset_pairs = dialog.pairs()
        self.dataset_source_mode.setCurrentIndex(self.dataset_source_mode.findData('multiple'))
        if dialog.review_pair is not None:
            row = dialog.review_pair
            self.annotation_combo.setCurrentIndex(self.annotation_combo.findData(row['coverage']))
            from ._model_project import layer_id
            for combo, key in ((self.image_combo, 'image_id'), (self.truth_combo, 'truth_id'), (self.reviewed_combo, 'reviewed_id')):
                for i in range(combo.count()):
                    layer = combo.itemData(i)
                    if layer is not None and layer_id(layer) == row.get(key):
                        combo.setCurrentIndex(i)
                        break
            self.review_pair_toggle.setChecked(True)
        self._sources_changed()

    def _pair_layers(self):
        """Watch every included source, including unresolved rows, for removal/edits."""
        if not self._multiple_pairs():
            return [layer for layer in (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData()) if layer is not None]
        identifiers = self._channel_layer_ids() if self._is_multilabel() else {row.get(key) for row in self.dataset_pairs if row.get('included') for key in ('image_id', 'truth_id', 'reviewed_id')}
        from ._model_project import layer_id
        return [layer for layer in self.viewer.layers if layer_id(layer) in identifiers]

    def _channel_layer_ids(self):
        identifiers = set()
        for row in self.channel_assignments:
            if row.get('included'):
                identifiers.add(row.get('image_id'))
                for channel in row.get('channels', {}).values():
                    identifiers.update((channel.get('target_id'), channel.get('reviewed_id')))
        return identifiers

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
            self.activity.report('training', 'error', str(error), priority=80)

    def _status(self, message):
        if not self.closed:
            if self.worker is not None:
                current = self.activity.records.get('training', {})
                if current.get('state') not in ('error', 'warning'):
                    self.activity.report('training', 'busy', message, priority=80,
                                         completed=max(0, self.progress.value()), total=self.progress.maximum(),
                                         cancel=self._cancel)
            self._update_training_overview()
            self.status_label.setText(message)
            self.status_label.setToolTip(message)
            self.dataset_feedback.setText(message)
            smaller = self._smaller_patch_size()
            retry = 'Reviewed targets need at least two separate spatial blocks' in message and smaller is not None
            self.smaller_patches_button.setVisible(retry)
            if retry:
                self.smaller_patches_button.setText(f'Try {smaller} px model patches and prepare again')
            if hasattr(self, "activity_log"):
                scrollbar = self.activity_log.verticalScrollBar()
                old_position = scrollbar.value()
                at_end = old_position >= scrollbar.maximum()
                self.activity_log.appendPlainText(datetime.now().strftime("%H:%M:%S") + "  " + message)
                scrollbar.setValue(scrollbar.maximum() if at_end else old_position)
            self.viewer.status = message

    def _smaller_patch_size(self):
        current = self.crop_combo.currentData()
        sizes = [self.crop_combo.itemData(i) for i in range(self.crop_combo.count())]
        return max((size for size in sizes if size is not None and size < current), default=None)

    def _prepare_smaller_patches(self):
        size = self._smaller_patch_size()
        if size is None or self.worker is not None:
            return
        self.crop_combo.setCurrentIndex(self.crop_combo.findData(size))
        if self.resolution_combo.findData(size) < 0:
            self.resolution_combo.addItem(f'{size} × {size}', size)
        self.resolution_combo.setCurrentIndex(self.resolution_combo.findData(size))
        self._start_preparing()

    def refresh_layers(self):
        if self.closed or self.worker is not None:
            return
        from ._dataset_pairs import associate_pair_coverage
        pair_coverage_changed = associate_pair_coverage(self.viewer, self.dataset_pairs)
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
        if pair_coverage_changed or any(a is not b for a, b in zip(previous, (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData()))):
            self._sources_changed()
        else:
            self._update_controls()

    def _source_removed(self, event):
        from ._model_project import layer_id
        if any(event.value is layer for layer in (self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_combo.currentData())) or (self._is_multilabel() and layer_id(event.value) in self._channel_layer_ids()) or (self._multiple_pairs() and any(layer_id(event.value) == row.get(key) for row in self.dataset_pairs if row.get('included') for key in ('image_id', 'truth_id', 'reviewed_id'))):
            self._invalidate_crops()

    def _select_associated_reviewed_mask(self):
        from ._reviewed_regions import associated_mask, KEY
        if self.annotation_combo.currentData() != "partial":
            return
        image = self.image_combo.currentData()
        mask = associated_mask(self.viewer, image, self.truth_combo.currentData())
        current = self.reviewed_combo.currentData()
        if mask is not None or (current is not None and current.metadata.get(KEY, {}).get("role") == "reviewed_regions"):
            self.reviewed_combo.blockSignals(True)
            self.reviewed_combo.setCurrentIndex(max(0, self.reviewed_combo.findData(mask)))
            self.reviewed_combo.blockSignals(False)

    def _source_event(self, event):
        layer = getattr(event, 'source', None)
        entry = getattr(layer, '_label_assistant_mask_entry', None)
        if (entry is not None and entry.manager.loading
                and getattr(event, 'type', '') in ('data', 'paint', 'labels_update',
                                                  'scale', 'translate', 'rotate', 'shear', 'affine')):
            # Loading a different crop/overview does not edit the full s0 store
            # or its registration. Actual edits notify separately.
            return
        if (getattr(event, 'type', '') in ('data', 'paint', 'labels_update')
                and isinstance(layer, napari.layers.Labels)
                and layer.metadata.get('label_assistant_review', {}).get('role') not in ('reviewed_regions', 'reviewed_annotations')
                and not getattr(layer, '_label_assistant_library_precise_edit', False)
                and not getattr(layer, '_label_assistant_library_rebinding', False)):
            # The shared library widget handles its current target. Other image
            # pairs/channels still need their review approval revoked on edits.
            if not any(widget.layer is layer for widget in getattr(self.viewer, '_label_assistant_library_widgets', ()) if not widget.closed):
                from ._component_library import annotation_changed, annotation_bounds
                annotation_changed(self.viewer, layer, annotation_bounds(event))
        self._invalidate_crops()

    def _sources_changed(self, *_args):
        self._use_single_included_pair()
        self._select_associated_reviewed_mask()
        for emitter in self._source_emitters:
            with suppress(Exception):
                emitter.disconnect(self._source_event)
        self._source_emitters.clear()
        for layer in self._pair_layers():
            if layer is None:
                continue
            from ._component_library import ensure_library_event
            ensure_library_event(layer)
            for name in ("data", "paint", "labels_update", "component_library", "scale", "translate", "rotate", "shear", "affine"):
                emitter = getattr(layer.events, name, None)
                if emitter is not None:
                    emitter.connect(self._source_event)
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
        self._dataset_outdated = self._dataset_outdated or was_prepared
        self._overview_dataset = None
        self.prepared = None
        self._patch_subset_cache = None
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
        self.training_compute.set_busy(busy)
        self.batch_widget.set_context(self.image_combo.currentData(), self.truth_combo.currentData(), self.reviewed_area_widget)
        self.batch_widget.set_busy(busy)
        for control in (
            self.select_training_tiles_button, self.confirm_training_tiles_button,
            self.validation_method_combo, self.holdout_spin, self.fold_spin, self.validation_seed_spin,
            self.rotation_check, self.flip_check, self.intensity_check,
            self.prepare_button, self.image_combo, self.truth_combo, self.samples_spin,
            self.steps_spin, self.seed_spin, self.learning_rate_spin, self.crop_combo, self.device_combo, self.load_button, self.inference_load_button,
            self.dataset_source_mode, self.manage_pairs_button, self.review_pair_toggle,
            self.resolution_combo, self.prediction_combo, self.prediction_scope, self.draw_prediction_box_button, self.threshold_spin, self.refresh_button,
            self.workflow_combo, self.annotation_combo, self.reviewed_combo, self.epochs_spin, self.batch_spin, self.extra_targets_spin, self.create_reviewed_button,
        ):
            control.setEnabled(not busy)
        for button in (self.select_training_tiles_button, self.confirm_training_tiles_button,
                       self.confirm_tiles_on_canvas, self.show_confirmed_tiles_button,
                       self.review_tile_height, self.review_tile_width, self.cleanup_masks_button,
                       self.edit_confirm_button):
            button.setEnabled(not busy and self.reviewed_area_widget is not None)
        self.smaller_patches_button.setEnabled(not busy)
        self.show_dataset_split_button.setEnabled(not busy and isinstance(self.prepared, IndexedPatches))
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
        compatible = (isinstance(self.prepared, IndexedPatches) and self.model is not None and
                      (getattr(self.model, 'crop_pixels', None), self.model.config.input_pixels) == (self.prepared.crop_pixels, self.prepared.input_pixels) and
                      tuple(self.model.config.class_names) == tuple(self.prepared.class_names) and
                      self.model.config.multilabel == self.prepared.multilabel)
        self.full_validation_button.setEnabled(not busy and compatible and not seen_holdout and not kfold)
        self.full_validation_button.setToolTip('Evaluate fixed model weights on every reserved patch; no training or checkpoint reselection.'
            if compatible and not seen_holdout and not kfold else 'Prepare a matching spatial/image holdout dataset and load a model. Reserved validation regions must be independent of its training.')
        self.export_csv_button.setEnabled(bool(self.learning_chart.points))
        self.continuation_note.setText(
            "No model loaded. To continue a previous batch, select its checkpoint in Models → Saved and in-memory models → Select this model."
            if self.model is None else
            "The current model has already trained in the reserved validation regions. Review new regions for independent validation, or train a new model."
            if seen_holdout else "")
        self.continuation_note.setVisible(bool(self.continuation_note.text()))
        self.predict_button.setEnabled(not busy and self.model is not None)
        names = getattr(getattr(self.model, "config", None), "class_names", ())
        multilabel = getattr(getattr(self.model, 'config', None), 'multilabel', False)
        self.threshold_spin.setEnabled(not busy and (not names or multilabel))
        self.inference_classes_label.setText(
            "Classes: 0 = background; " + "; ".join(f"{i} = {name}" for i, name in enumerate(names, 1)) + ". Highest-probability class wins; binary threshold is unused."
            if names else "Binary model: background / foreground. Threshold controls foreground selection.")
        if multilabel:
            self.inference_classes_label.setText('Independent channels: ' + ', '.join(names) + '. Threshold applies to each channel; overlapping masks are preserved as separate layers.')
        self.models_panel.set_busy(busy)
        self._update_source_mode_visibility()
        self._update_model_summary()
        image, truth = self.image_combo.currentData(), self.truth_combo.currentData()
        pair_context = f"Image: {image.name if image is not None else 'Choose an image'} · Annotations: {truth.name if truth is not None else 'Choose a Labels layer'}"
        definitions = self.channel_assignments if self._is_multilabel() else self.dataset_pairs
        self.source_context.setText(f"Dataset: {sum(bool(row.get('included')) for row in definitions)} selected images" if self._multiple_pairs() else pair_context)
        if hasattr(self, "editor_pair_context"):
            self.editor_pair_context.setText(pair_context)
        reason = ("An operation is running. Wait or use Cancel." if busy else
                  "Choose an image and annotation layer in Dataset." if image is None or truth is None else
                  "Prepare the dataset in Dataset to enable training." if self.prepared is None else
                  "Dataset ready. Train a new model in Train.")
        predict_reason = "Train or load a model to enable prediction." if self.model is None else {
            "area": "Predict one training-sized crop centered at the current canvas position.",
            "image": "Predict the entire image with overlapping tiles. Adds a new Labels layer.",
            "box": "Predict the latest drawn box with overlapping tiles. Adds a new Labels layer.",
        }[self.prediction_scope.currentData()]
        if busy:
            predict_reason = "Prediction is unavailable while an operation is running."
        if self.model is not None and self.prepared is not None:
            if kfold:
                reason += " Continue training requires spatial holdout instead of k-fold validation."
            elif seen_holdout:
                reason += " " + self.continuation_note.text()
        self.readiness_label.setText(reason + "\n" + predict_reason)
        self.train_button.setToolTip(reason)
        self.predict_button.setToolTip(predict_reason)
        self.continue_button.setToolTip("Continue training requires spatial holdout instead of k-fold validation." if kfold else self.continuation_note.text() if self.model is None or seen_holdout else "Prepare a dataset first." if self.prepared is None else reason)
        subset_error = self._update_patch_selection_controls()
        if subset_error and isinstance(self.prepared, IndexedPatches):
            self.train_button.setEnabled(False)
            self.continue_button.setEnabled(False)
            self.patch_preview_button.setEnabled(False)
            self.train_button.setToolTip(subset_error)
            self.continue_button.setToolTip(subset_error)
            self.readiness_label.setText(subset_error + '\n' + predict_reason)
        self.cancel_button.setEnabled(busy)
        batch_error = self.batch_preparation_error()
        self.prepare_button.setToolTip(batch_error or 'Prepare the selected annotation scope. Patch reads stay bounded; preview is optional.')
        if batch_error:
            self.prepare_button.setEnabled(False)
            self.prepare_button.setToolTip(batch_error)
            if self.prepared is None:
                self.readiness_label.setText(batch_error + '\n' + predict_reason)
        self.training_selection_panel.setEnabled(not busy)
        self.selection_toggle.setEnabled(not busy)
        self.progress.setVisible(busy)
        if not busy:
            self._update_annotation_note()
            self._update_workload()

    def _start_worker(self, function, on_done, *args, **kwargs):
        if self.worker is not None:
            raise ValueError("Wait for the current Model Training operation to finish.")
        self.cancel_event = Event()
        kwargs["cancel"] = self.cancel_event
        worker = create_worker(
            _run_task, function, args, kwargs, _start_thread=False,
            _connect={"errored": self._on_error},
        )
        self.worker = worker
        name = getattr(function, '__name__', '').lower()
        title = next((text for part, text in (
            ('predict', 'Predicting labels…'), ('train', 'Training model…'),
            ('load', 'Loading model…'), ('save', 'Saving model…'),
            ('export', 'Exporting model…'), ('preview', 'Preparing patch preview…'),
            ('compar', 'Comparing models…'), ('evaluat', 'Evaluating model…'),
            ('index', 'Preparing dataset…'), ('prepare', 'Preparing dataset…')) if part in name),
            'Preparing model operation…')
        self.activity.report('training', 'busy', title, priority=80, cancel=self._cancel)
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
            self.activity.report('training', 'error', 'Model operation could not start', priority=80)
            raise

    def _on_progress(self, update):
        if self.closed:
            return
        self.progress.setRange(0, update.total)
        self.progress.setValue(update.completed)
        self._status(update.message)
        if update.metrics is not None and update.epoch is not None:
            self.learning_chart.add_point(update.epoch, update.metrics, update.fold, update.training_step if update.training_step is not None else update.completed, update.training_loss_last_batch, "step" if self.workflow_combo.currentData() == "sampled" else "epoch", series=update.series)
            if update.series and 'Trial validation' in update.series:
                evaluation = self._selected_training_dataset().validation.get('evaluation', {})
                self.learning_chart.points[-1].update(validation_scope='trial', validation_patches=evaluation.get('selected_patches', ''))
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
            self._status("Model Training operation canceled. Existing layers and model are unchanged.")
            self.activity.report('training', 'warning', 'Model operation stopped',
                                 details=self.status_label.text(), priority=80)
            return
        self._safe(lambda: on_done(result))

    def _on_error(self, error):
        if self.closed:
            return
        if isinstance(error, QuickTrainCancelled):
            self._status("Model Training operation canceled. Existing layers and model are unchanged.")
        else:
            self._status(f"Model Training failed: {error}")
        self.activity.report('training', 'warning' if isinstance(error, QuickTrainCancelled) else 'error',
                             self.status_label.text(), priority=80)

    def _finished(self):
        self.worker = None
        self.cancel_event = None
        self._unlock_sources()
        if not self.closed:
            self.activity.finish('training', self.status_label.text())
            self._update_controls()
            self.refresh_layers()
            shell = getattr(self.viewer, '_label_assistant_curation_widget', None)
            if shell is not None and hasattr(shell, 'simple') and not shell.simple.closed:
                shell.simple.refresh_status()

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
            class_names=self.models_panel.class_names(),
            multilabel=self._is_multilabel(),
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
                self.annotation_combo.currentData(), id(self.reviewed_combo.currentData()), library_signature, self.validation_method_combo.currentData(), self.holdout_spin.value(), self.fold_spin.value(), self.validation_seed_spin.value(), self.models_panel.class_names_edit.text(), self.dataset_source_mode.currentData(), json.dumps(self.dataset_pairs, sort_keys=True) if self._multiple_pairs() else None, self._is_multilabel(), json.dumps(self.channel_assignments, sort_keys=True) if self._is_multilabel() else None, self._batch_signatures())

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
        grouped = self._multiple_pairs() and (not self._is_multilabel() or sum(bool(row.get('included')) for row in self.channel_assignments) > 1)
        blocked = self.validation_method_combo.blockSignals(True)
        self.validation_method_combo.setItemText(0, 'Image/specimen holdout' if grouped else 'Spatial holdout (recommended)')
        self.validation_method_combo.setItemText(1, 'Image/specimen k-fold' if grouped else 'Spatial k-fold (advanced)')
        self.validation_method_combo.blockSignals(blocked)
        self.train_button.setText(("Run image-group k-fold validation" if grouped else "Run spatial k-fold validation") if kfold else "Train new model")
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
            unit = 'image/specimen groups' if dataset.sources else 'spatial groups'
            self.validation_summary.setText(f"Eligible pool: {groups} {unit} · {len(dataset.indices('training')):,} training / {validation:,} validation patches ({validation/len(dataset.locations):.1%} actual holdout). Group sizes can change the requested ratio.")

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
        dataset = self._selected_training_dataset()
        image = self._layer(self.image_combo)
        canvas_width, canvas_height = self.viewer.window._qt_viewer.canvas.size
        center = np.asarray(self.viewer.camera.center[-2:])
        half = np.asarray([canvas_height, canvas_width]) / (2*max(self.viewer.camera.zoom, 1e-6))
        corners = np.asarray([image.world_to_data(tuple(center+offset*half)) for offset in (np.array([-1,-1]), np.array([-1,1]), np.array([1,-1]), np.array([1,1]))])
        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
        rows = [p for p in dataset.locations if p['y1'] >= minimum[-2] and p['y0'] <= maximum[-2] and p['x1'] >= minimum[-1] and p['x0'] <= maximum[-1]]
        if dataset.sources:
            image_id = image.metadata.get('label_assistant_review', {}).get('id')
            rows = [p for p in rows if p.get('image_id') == image_id]
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

    def _selected_training_dataset(self):
        dataset = self.prepared
        if not isinstance(dataset, IndexedPatches) or self.validation_method_combo.currentData() == 'kfold':
            return dataset
        key = (id(dataset), dataset.validation_seen_before, self.patch_selection_mode.currentData(),
               self.patch_percentage_spin.value(), self.patch_count_spin.value(), self.patch_subset_seed.value(),
               self.quick_trial_check.isChecked(), self.trial_validation_percentage.value(), self.validation_seed_spin.value())
        cached = self._patch_subset_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        selected = select_training_patches(dataset, mode=key[2], percentage=key[3], count=key[4], seed=key[5])
        if key[6]:
            selected = select_trial_validation(selected, percentage=key[7], seed=key[8])
        self._patch_subset_cache = (key, selected)
        return selected

    def _quick_trial_changed(self, checked):
        if checked and self.patch_selection_mode.currentData() == 'all':
            self.patch_selection_mode.setCurrentIndex(self.patch_selection_mode.findData('percentage'))
        self._patch_selection_changed()

    def _patch_selection_changed(self, *_args):
        self._patch_subset_cache = None
        if self._validation_preview_layer is not None:
            self._validation_preview_layer.visible = False
        if self._gallery_return_view is not None:
            self._return_to_image()
        self._update_workload()
        self._update_controls()

    def _update_patch_selection_controls(self):
        coverage = self.workflow_combo.currentData() == 'coverage'
        kfold = self.validation_method_combo.currentData() == 'kfold'
        mode = self.patch_selection_mode.currentData()
        self.patch_selection_group.setVisible(coverage)
        for control, visible in ((self.patch_percentage_spin, mode == 'percentage'),
                                 (self.patch_count_spin, mode == 'count'),
                                 (self.patch_subset_seed, mode != 'all')):
            control.setVisible(visible)
            self.patch_selection_form.labelForField(control).setVisible(visible)
        for control in (self.patch_selection_mode, self.patch_percentage_spin, self.patch_count_spin, self.patch_subset_seed):
            control.setEnabled(coverage and not kfold and self.worker is None)
        self.quick_trial_check.setEnabled(coverage and not kfold and self.worker is None)
        self.trial_validation_percentage.setVisible(self.quick_trial_check.isChecked())
        self.patch_selection_form.labelForField(self.trial_validation_percentage).setVisible(self.quick_trial_check.isChecked())
        self.trial_validation_percentage.setEnabled(coverage and not kfold and self.worker is None)
        if kfold:
            self.patch_selection_summary.setText('K-fold and final training use the full index. Trial subsets are available with spatial/image holdout.')
            return None
        if not isinstance(self.prepared, IndexedPatches):
            self.patch_selection_summary.setText('Prepare the dataset to count eligible patches. Confirmed review tiles remain approved.')
            return None
        try:
            selected = self._selected_training_dataset()
        except ValueError as error:
            self.patch_selection_summary.setText(str(error))
            return str(error)
        available = len(self.prepared.indices('training'))
        rows = [p for p in selected.locations if p['split'] == 'training']
        trial = selected.validation.get('evaluation', {})
        validation_text = (f"Trial validation: {trial['selected_patches']:,} / {trial['available_patches']:,} reserved patches "
                           f"({trial['positive_patches']:,} with targets; {trial['spatial_groups']:,} spatial/image groups). "
                           'Fixed each epoch. Evaluate the full set before accepting the model.'
                           if trial.get('mode') == 'trial' else
                           f"Validation: {len(selected.indices('validation')):,} reserved patches unchanged.")
        self.patch_selection_summary.setText(
            f"Using {len(rows):,} / {available:,} training patches ({sum(p['contains_foreground'] for p in rows):,} with targets).\n"
            + validation_text + '\n' + ('The same training subset is reused each epoch.' if mode != 'all' else 'All eligible training patches are used.'))
        return None

    def _update_workload(self, *_args):
        coverage = self.workflow_combo.currentData() == "coverage"
        self.workload_label.setVisible(coverage)
        if not coverage:
            return
        if not isinstance(self.prepared, IndexedPatches):
            self.workload_label.setText("Index the dataset to calculate updates per epoch.")
            return
        try:
            selected = self._selected_training_dataset()
        except ValueError as error:
            self.workload_label.setText(str(error))
            return
        count = len(selected.indices('training'))
        extra = int(np.ceil(count * self.extra_targets_spin.value() / 100))
        updates = (count + extra + self.batch_spin.value() - 1) // self.batch_spin.value()
        label = 'trial validation' if selected.validation.get('evaluation', {}).get('mode') == 'trial' else 'reserved validation'
        self.workload_label.setText(
            f"Per epoch: {count:,} selected patches + {extra:,} target repeats = {updates:,} updates. "
            f"Maximum: {self.epochs_spin.value()} epochs, {updates * self.epochs_spin.value():,} updates, "
            f"plus {len(selected.indices('validation')):,} {label} patches after each epoch."
        )

    def _annotation_changed(self, *_args):
        self._workflow_changed()
        self._sources_changed()

    def _workflow_changed(self, *_args):
        partial = self.annotation_combo.currentData() == "partial"
        multiple = self._multiple_pairs()
        sampled_index = self.workflow_combo.findData("sampled")
        self.workflow_combo.model().item(sampled_index).setEnabled(not partial and not multiple)
        if (partial or multiple) and self.workflow_combo.currentData() != "coverage":
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
            "Index eligible patches in bounded memory. Choose all patches or a trial subset in Training patch selection; validation remains reserved."
            if coverage else "Prepare a fixed crop sample for a quick experiment. Training reuses this sample."
        )
        self.validation_note.setText(
            "Each epoch visits the selected training patches. Separate spatial blocks are reserved for validation; trial sampling never moves them into training."
            if coverage else "Train on a fixed crop sample; separate spatial blocks are held out for validation."
        )
        if multiple:
            self.validation_note.setText('Choose all eligible training patches or a fixed trial subset across selected pairs. Whole image/specimen groups remain held out for validation.')
            if self._is_multilabel():
                self.validation_note.setText('Independent target channels support all training patches or a fixed trial subset. Spatial/image validation groups remain reserved.')
        self.workflow_combo.setToolTip(self.validation_note.text())
        self._invalidate_crops()
        self.crop_summary_label.setText(
            "Index the dataset to inspect coverage and representative examples."
            if coverage else "Prepare crops to inspect positive and background examples."
        )
        self._update_annotation_note()

    def _update_training_overview(self):
        """Use persisted selection metadata or the existing patch index, never scan pixels."""
        from ._component_library import library_for
        from ._reviewed_regions import KEY, effective_areas, region_reviewed
        from ._model_project import layer_id
        if not hasattr(self, "annotation_sources_label"):
            return
        lines, eligible, objects, needs_review = [], False, 0, 0
        active_truths = {truth for _,truth,scope in self._batch_signatures() if scope[0] == 'active'}
        rows = []
        by_id = {layer_id(layer): layer for layer in self.viewer.layers
                 if isinstance(layer, (napari.layers.Image, napari.layers.Labels)) and not is_internal_layer(layer)} if self._multiple_pairs() else {}
        if self._multiple_pairs():
            for row in self.channel_assignments if self._is_multilabel() else self.dataset_pairs:
                if not row.get('included'):
                    continue
                if self._is_multilabel():
                    names = [name.strip() for name in self.models_panel.class_names_edit.text().split(',') if name.strip()]
                    for name in names:
                        assignment = row.get('channels', {}).get(name, {})
                        rows.append((f"{row.get('image_name', 'Image')} / {name}", assignment.get('coverage'),
                                     by_id.get(assignment.get('target_id')) if row.get('image_id') in by_id else None, by_id.get(assignment.get('reviewed_id'))))
                else:
                    rows.append((row.get('image_name', 'Image'), row.get('coverage'),
                                 by_id.get(row.get('truth_id')) if row.get('image_id') in by_id else None, by_id.get(row.get('reviewed_id'))))
        else:
            image, truth = self.image_combo.currentData(), self.truth_combo.currentData()
            rows.append((image.name if image is not None else "Choose an image", self.annotation_combo.currentData(), truth if image is not None else None, self.reviewed_combo.currentData()))
        for name, coverage, truth, mask in rows:
            included = changed = 0
            skipped_pending = False
            if truth is not None and coverage == 'partial':
                try:
                    library = library_for(truth)
                    if library is not None:
                        _, included, changed = library.counts()
                        pending = library.state()['pending']
                        skipped_pending = bool(pending and id(truth) in active_truths)
                        if skipped_pending:
                            included = changed = 0
                        else:
                            needs_review += int(bool(included and pending))
                except Exception:
                    needs_review += 1
            objects += included
            if coverage == 'unknown':
                lines.append(f"{name}: unknown channel")
                continue
            if truth is None:
                lines.append(f"{name}: choose ground truth")
                continue
            if coverage == 'complete':
                lines.append(f"{name}: complete coverage (explicit)")
                eligible = True
                continue
            state = mask.metadata.get(KEY, {}) if mask is not None else {}
            areas = effective_areas(state)
            cells = state.get('review_cells', state.get('areas', []))
            count = sum(region_reviewed(cell, areas) for cell in cells)
            external = mask is not None and 'areas' not in state
            eligible |= bool(areas) or external
            detail = "external coverage mask; measured on preparation" if external else f"{count:,} confirmed tiles available"
            legacy = f" · {included:,} saved object selections" if included else ''
            lines.append(f"{name}: {detail}{legacy}" + (f" · {changed:,} objects need review" if changed else "") +
                         (' · pending object selections excluded from this batch' if skipped_pending else ''))
        visible_lines = lines[:4] + ([f"{len(lines) - 4:,} more sources; see Manage pairs / channels."] if len(lines) > 4 else [])
        self.annotation_sources_label.setText(('Global annotation sources\n' if active_truths else 'Annotation sources\n') + "\n".join(visible_lines))
        self.annotation_sources_label.setToolTip("\n".join(lines))
        dataset = self.prepared
        self.annotation_combo.setItemText(self.annotation_combo.findData('partial'),
            'Partial — confirmed tiles + saved object selections' if objects else 'Partial — confirmed tiles')
        if isinstance(dataset, IndexedPatches):
            self._dataset_outdated = False
            if dataset is getattr(self, '_overview_dataset', None):
                self.supervision_label.setText(self._overview_supervision)
                self.dataset_state_label.setText(self._overview_state)
                return
            positive = sum(p['foreground_pixels_original'] for p in dataset.locations)
            known = dataset.reviewed_pixels
            detail = f"Confirmed foreground: {positive:,} pixels\nConfirmed background: {known - positive:,} pixels\nUnknown pixels excluded: {dataset.total_pixels - known:,}"
            if dataset.multilabel:
                for channel, name in enumerate(dataset.class_names):
                    fg = sum(p['channel_positive_pixels'][channel] for p in dataset.locations)
                    valid = sum(p['channel_reviewed_pixels'][channel] for p in dataset.locations)
                    detail += f"\n{name}: {fg:,} foreground · {valid - fg:,} background (independent validity)"
            self.supervision_label.setText(detail)
            self.dataset_state_label.setText(f"Prepared pool · {len(dataset.indices('training')):,} training patches · {len(dataset.indices('validation')):,} validation patches")
            self._overview_dataset = dataset
            self._overview_supervision = self.supervision_label.text()
            self._overview_state = self.dataset_state_label.text()
        elif dataset is not None:
            self._dataset_outdated = False
            self.supervision_label.setText("Explicit complete coverage · foreground and background checked during preparation.")
            self.dataset_state_label.setText("Prepared")
        else:
            self.supervision_label.setText(
                "Foreground annotations available. Background has not been verified.\nUnknown pixels are excluded. Review background in Labels; training and validation require targets and verified background."
                if objects and not eligible else
                "Confirmed tiles include annotated objects and checked background.\nUnconfirmed areas are excluded."
                + ("\nSaved object selections add foreground only; they do not approve surrounding background." if objects else ''))
            self.dataset_state_label.setText(
                "Outdated because source annotations or dataset settings changed. Prepare again." if self._dataset_outdated else
                "Update the component library after edits before preparation." if needs_review else
                "Ready to prepare" if eligible else
                "Foreground only — confirm reviewed tiles in Edit & Confirm before preparation." if objects else "No confirmed tiles yet — use Edit & Confirm.")
            error = self.batch_preparation_error()
            if error:
                self.dataset_state_label.setText(error)

    def _update_annotation_note(self):
        partial = self.annotation_combo.currentData() == "partial"
        partial_tiles = partial if not self._multiple_pairs() else any(row.get('included') and row.get('coverage') == 'partial' for row in self.dataset_pairs)
        self.training_tile_actions.setVisible(partial_tiles and not self._is_multilabel())
        self.reviewed_combo.setVisible(partial)
        self.create_reviewed_button.setVisible(False)
        self.selection_form.labelForField(self.reviewed_combo).setVisible(partial)
        self.annotation_note.setToolTip(
            "Reviewed cells supply foreground and background. Selected objects supply "
            "foreground only. Shared pixels are counted once; unknown pixels are ignored."
        )
        self._update_training_overview()
        if self._multiple_pairs():
            self.annotation_note.setText('Each channel has independent coverage in Manage channels. One image uses spatial validation; several images use image/specimen groups.' if self._is_multilabel() else 'Coverage is configured per pair in Manage pairs. Validation keeps image/specimen groups separate.')
            self._update_source_mode_visibility()
            return
        if not partial:
            self.annotation_note.setText("Complete annotations · zero pixels are background")
            return
        self.annotation_note.setText("Partial supervision · unknown pixels excluded")

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
        associate(layer, image, self.truth_combo.currentData() if self._is_multilabel() else None)
        self.refresh_layers()
        self.reviewed_combo.setCurrentIndex(self.reviewed_combo.findData(layer))
        self._status("Empty coverage created. Correct masks and confirm tiles in Labels → Edit & Confirm.")

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
        if self._multiple_pairs():
            layers = self._pair_layers()
            controllers = getattr(self.viewer, '_label_assistant_edit_controllers', ()) or ()
            relevant = [c for c in controllers if not c.closed and any(c.source is layer for layer in layers)]
            if any(c.dirty or c.stroke_active for c in relevant):
                raise ValueError('Apply pending annotation edits in every included pair before preparing or training.')
            layers += [c.tile for c in relevant if c.tile is not None]
            for layer in layers:
                if isinstance(layer, napari.layers.Labels):
                    self._locked_extra_layers.append((layer, bool(layer.editable)))
                    layer.editable = False
            return
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
                if source.get('pairs'):
                    regions.extend(p for p in history.get('locations', self.model.locations) if p.get('image_id') == image_id and p['split'] == 'training')
                    continue
                if (source.get('image_id') and source['image_id'] == image_id) or source.get('image_layer') == image.name:
                    locations = history.get('locations', self.model.locations if history.get('workflow') == 'coverage' else [])
                    regions.extend(p for p in locations if p['split'] == 'training')
        return regions

    def _partition_key(self, image, config):
        from ._annotation_batches import signature
        key = (id(image), tuple(array_2d(image.data).shape), config.crop_pixels)
        scope = signature(image, self.truth_combo.currentData())
        return key + scope if scope[0] == 'active' else key

    @staticmethod
    def _index_task(image, truth, config, *, reviewed, validation_blocks, prior_training_regions=(), cancel, validation_fraction=0.25, split_seed=42, validation_method="holdout", folds=5):
        dataset = yield from index_patches(image, truth, config, reviewed=reviewed, validation_blocks=validation_blocks, prior_training_regions=prior_training_regions, cancel=cancel, validation_fraction=validation_fraction, split_seed=split_seed, validation_method=validation_method, folds=folds)
        return dataset

    @staticmethod
    def _prepare_task(image, truth, config, *, cancel):
        crops = yield from prepare_crops(image, truth, config, cancel=cancel)
        return crops

    def _start_preparing(self):
        self.refresh_layers()
        from .workspace.service import _commit_pending_local_edits
        _commit_pending_local_edits(self.viewer)
        if self._is_multilabel():
            return self._start_preparing_channels()
        if self._multiple_pairs():
            return self._start_preparing_pairs()
        image = self._layer(self.image_combo)
        truth = self._layer(self.truth_combo)
        image_data, truth_data = array_2d(image.data), array_2d(full_data(truth))
        if tuple(image_data.shape) != tuple(truth_data.shape):
            raise ValueError("Training image and ground truth must have the same 2D shape.")
        from ._coordinates import matching_mask_transforms
        if not matching_mask_transforms(image, full_source(truth)):
            raise ValueError("Training image and ground truth must have matching spatial transforms.")
        config = self._config()
        partial = self.annotation_combo.currentData() == "partial"
        coverage = self.workflow_combo.currentData() == "coverage"
        from ._annotation_batches import restrict_supervision, selection
        if selection(image, truth)['scope'] == 'active' and not coverage:
            raise ValueError('Annotation batches use All / selected training patches. Switch to that workflow before preparing this batch.')
        if partial and not coverage:
            raise ValueError("Partial annotations require coverage training with a reviewed-region mask.")
        if partial and self.reviewed_combo.currentData() is None:
            raise ValueError("No checked tiles are confirmed for training. Open Edit & Confirm, choose the review-tile size, and confirm fully checked objects and background.")
        reviewed_layer = self._layer(self.reviewed_combo) if partial else None
        from ._reviewed_regions import KEY
        if reviewed_layer is not None:
            expected = reviewed_layer.metadata.get(KEY, {}).get("truth_id")
            if expected and truth.metadata.get(KEY, {}).get("id") != expected:
                raise ValueError("Select the Curated ground-truth layer associated with these reviewed areas.")
        if reviewed_layer is truth:
            raise ValueError("Use a separate reviewed-region mask, not the target ground-truth layer.")
        if reviewed_layer is not None and not matching_mask_transforms(image, full_source(reviewed_layer)):
            raise ValueError("Reviewed coverage must have matching spatial transforms with the training image.")
        reviewed_data = array_2d(full_data(reviewed_layer)) if reviewed_layer is not None else None
        if partial:
            from ._component_library import library_for, TrainingSupervision
            library = library_for(truth)
            # Whole-tile coverage does not depend on an unused object library.
            if library is not None and library.counts()[1] == 0:
                library = None
            if library is not None and library.state()["pending"]:
                if selection(image, truth)['scope'] == 'active':
                    library = None  # Confirmed batch regions need no global component reindex.
                else:
                    raise ValueError("Update the component library in Manage Training Selection after editing annotations.")
            invalid = reviewed_layer.metadata.get(KEY, {}).get("invalid_regions", []) if reviewed_layer else []
            review_state = reviewed_layer.metadata.get(KEY, {}) if reviewed_layer is not None else {}
            known_areas = review_state.get("areas") if review_state.get("role") == "reviewed_regions" else None
            reviewed_data = TrainingSupervision(truth_data, reviewed_data, library=library,
                                                invalid_regions=invalid, reviewed_areas=known_areas)
        reviewed_data = restrict_supervision(self.viewer, image, truth, reviewed_data)
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

    def _prior_training_image_ids(self):
        result = set()
        if self.model is None:
            return result
        for record in self.model.history:
            source = record.get('source', {})
            locations = record.get('locations', [])
            if source.get('pairs'):
                result.update(item.get('image_id') for item in locations if item.get('split') == 'training')
            elif source.get('image_id'):
                result.add(source['image_id'])
        result.discard(None)
        return result

    def _start_preparing_pairs(self):
        from ._dataset_pairs import resolve_pairs
        from ._component_library import library_for, TrainingSupervision
        from ._reviewed_regions import KEY
        from ._annotation_batches import selection, restrict_supervision
        resolved = resolve_pairs(self.viewer, self.dataset_pairs)
        config = self._config()
        pairs = []
        for row, image, truth, mask in resolved:
            reviewed = None
            if row['coverage'] == 'partial':
                library = library_for(truth)
                if library is not None and library.state()['pending']:
                    if selection(image, truth)['scope'] == 'active':
                        library = None
                    else:
                        raise ValueError(f'{image.name}: update the component library after editing labels.')
                state = mask.metadata.get(KEY, {})
                reviewed = TrainingSupervision(array_2d(full_data(truth)), array_2d(full_data(mask)), library=library,
                                               invalid_regions=state.get('invalid_regions', []),
                                               reviewed_areas=state.get('areas'))
            reviewed = restrict_supervision(self.viewer, image, truth, reviewed)
            info = dict(image_id=row['image_id'], ground_truth_id=row['truth_id'],
                        reviewed_id=row.get('reviewed_id') if mask is not None else None,
                        image_layer=image.name, ground_truth_layer=truth.name,
                        coverage=row['coverage'], split_group=row['split_group'])
            info['annotation_batch'] = selection(image, truth)
            if reviewed is not None and getattr(reviewed, 'library', None) is not None:
                info['component_library'] = dict(source_id=reviewed.snapshot['source_id'],
                                                 revision=reviewed.snapshot['revision'], selected_ids=list(reviewed.selected))
            pairs.append(dict(image=array_2d(image.data), ground_truth=array_2d(full_data(truth)), reviewed=reviewed, info=info))
        signature, generation = self._signature(), self._generation
        self._lock_sources()
        self._status(f'Preparing {len(pairs)} pairs. Entire image/specimen groups are held out for validation.')
        def done(dataset):
            if generation != self._generation or signature != self._signature():
                self._status('A dataset source changed; prepare the pairs again.')
                return
            dataset = self._protect_batch_holdouts(dataset)
            self.prepared = dataset
            self._prepared_signature = signature
            summary = dataset.summary()
            self.crop_summary_label.setText(f"{len(dataset.sources)} pairs · {len(dataset.locations):,} patches · {dataset.reviewed_pixels / dataset.total_pixels:.1%} supervised\nTraining: {summary['training']['total']:,} patches · Validation: {summary['validation']['total']:,} patches in held-out image/specimen groups.")
            self._status('Multiple-pair dataset ready. Train now or inspect the optional patch preview.')
        try:
            self._start_worker(index_pairs, done, pairs, config,
                               validation_fraction=self.holdout_spin.value() / 100,
                               split_seed=self.validation_seed_spin.value(),
                               validation_method=self.validation_method_combo.currentData(), folds=self.fold_spin.value(),
                               prior_training_image_ids=self._prior_training_image_ids())
        except Exception:
            self._unlock_sources()
            raise

    def _start_preparing_channels(self):
        from ._multilabel_targets import resolve_channels
        from ._multilabel import index_multilabel
        if not self.models_panel.class_names_edit.text().strip():
            self._manage_pairs()
            if not self.models_panel.class_names_edit.text().strip():
                self._status('Open Manage channels to name features and assign their mask layers before preparation.')
                return
        config = self._config()
        resolved = resolve_channels(self.viewer, config.class_names, self.channel_assignments)
        pairs = []
        for row, image, targets in resolved:
            channels, channel_info = [], []
            for name, target in zip(config.class_names, targets):
                if target is None:
                    channels.append(None)
                    channel_info.append(dict(name=name, coverage='unknown'))
                else:
                    assignment, layer, mask = target
                    from ._annotation_batches import restrict_supervision, selection
                    reviewed = self._channel_supervision(layer, mask, ignore_pending=selection(image, layer)['scope'] == 'active') if assignment['coverage'] == 'partial' else None
                    reviewed = restrict_supervision(self.viewer, image, layer, reviewed)
                    channels.append(dict(data=array_2d(full_data(layer)), reviewed=reviewed,
                                         foreground_mode=assignment.get('foreground_mode', 'binary')))
                    info = dict(name=name, **assignment)
                    info['annotation_batch'] = selection(image, layer)
                    if reviewed is not None and reviewed.library is not None:
                        info['component_library'] = dict(source_id=reviewed.snapshot['source_id'], revision=reviewed.snapshot['revision'], selected_ids=list(reviewed.selected))
                    channel_info.append(info)
            pairs.append(dict(image=array_2d(image.data), channels=channels,
                              info=dict(image_id=row['image_id'], image_layer=image.name,
                                        split_group=row.get('split_group') or row['image_id'], channels=channel_info)))
        signature, generation = self._signature(), self._generation
        def done(dataset):
            if generation != self._generation or signature != self._signature():
                self._status('Channel annotations changed; prepare again.')
                return
            dataset = self._protect_batch_holdouts(dataset)
            self.prepared, self._prepared_signature = dataset, signature
            self.crop_summary_label.setText(f"{len(pairs)} images · {len(config.class_names)} independent channels · {len(dataset.locations):,} patches\nTraining: {len(dataset.indices('training')):,} · Validation: {len(dataset.indices('validation')):,}. Unknown channel pixels are ignored.")
            self._status('Multilabel dataset ready. Overlapping target masks stay independent during training.')
        self._lock_sources()
        try:
            self._start_worker(index_multilabel, done, pairs, config,
                               validation_fraction=self.holdout_spin.value() / 100,
                               split_seed=self.validation_seed_spin.value(),
                               validation_method=self.validation_method_combo.currentData(), folds=self.fold_spin.value(),
                               prior_training_image_ids=self._prior_training_image_ids(),
                               prior_training_regions=self._previous_sampled_training(resolved[0][1]) if len(resolved) == 1 else ())
        except Exception:
            self._unlock_sources()
            raise

    @staticmethod
    def _channel_supervision(layer, mask, *, ignore_pending=False):
        from ._component_library import TrainingSupervision, library_for
        from ._reviewed_regions import KEY
        library = library_for(layer)
        if library is not None and library.state()['pending']:
            if ignore_pending:
                library = None
            else:
                raise ValueError(f"{layer.name}: update the component library after edits.")
        state = mask.metadata.get(KEY, {})
        return TrainingSupervision(array_2d(full_data(layer)), array_2d(full_data(mask)), library=library,
            invalid_regions=state.get('invalid_regions', []),
            reviewed_areas=state.get('areas') if state.get('role') == 'reviewed_regions' else None)

    def _index_done(self, result, image, truth, signature, generation, partition_key):
        if generation != self._generation or signature != self._signature() or not all(any(layer is item for item in self.viewer.layers) for layer in (image, truth)):
            self._status("Source changed during indexing. Prepare again.")
            return
        dataset = self._protect_batch_holdouts(result)
        self.prepared = dataset
        self._prepared_signature = signature
        self._coverage_splits[partition_key] = [b['id'] for b in dataset.blocks if b['split'] == 'validation']
        summary = dataset.summary()
        self.crop_summary_label.setText(
            f"Indexed {len(dataset.locations):,} patches · supervised area {dataset.reviewed_pixels / dataset.total_pixels:.1%} of image\n"
            f"Training: {summary['training']['total']:,} patches ({summary['training']['positive']:,} with targets)\n"
            f"Validation: {summary['validation']['total']:,} patches in reserved spatial blocks\n"
            "Training patch selection controls the amount used. The optional preview shows representative selected patches."
        )
        self._update_annotation_note()
        self._update_validation_controls(False)
        self.validation_results_label.setText("Dataset indexed. Validation metrics will appear after the first evaluated epoch.")
        self._update_workload()
        self._status("Dataset ready. Choose all training patches or a trial subset, then train. Reserved validation patches remain excluded from training.")

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
        selected = self._selected_training_dataset()
        generation = self._generation
        self._lock_sources()
        def done(result):
            if generation != self._generation:
                self._status("Dataset changed; preview discarded. Prepare again.")
                return
            atlas, validity = result
            self._show_crop_grid(atlas, image, truth, validity=validity)
            self._status("Patch preview: a separate contact sheet beside the image. Cyan frames are training; orange frames are validation. Double-click a tile to locate its source, or Return to image.")
        self._start_worker(self._patch_preview_task, done, selected)

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
        grid_image = self.viewer.add_image(atlas.image, name="Model Training — crop images", contrast_limits=(0, 1), translate=translation, scale=scale, metadata=metadata.copy())
        grid_masks = []
        for name, mask in (zip(atlas.channel_names, atlas.mask) if atlas.channel_names else [('', atlas.mask)]):
            grid_mask = self.viewer.add_labels(mask, name='Model Training — crop ground truth' + (f' — {name}' if name else ''), translate=translation, scale=scale, metadata=metadata.copy(), opacity=0.5)
            grid_mask.editable = False
            grid_masks.append(grid_mask)
        rectangles, captions, colors = [], [], []
        for tile in atlas.locations:
            y, x, size = tile["atlas_y0"], tile["atlas_x0"], tile["model_size"]
            rectangles.append([[y, x], [y, x + size - 1], [y + size - 1, x + size - 1], [y + size - 1, x]])
            captions.append(f"{tile['tile']:02d} {tile['split']} · {tile['kind']}\n{tile.get('image_name', '')} y={tile['y0']}, x={tile['x0']}")
            colors.append("orange" if tile["split"] == "validation" else "cyan")
        frames = self.viewer.add_shapes(
            rectangles, shape_type="rectangle", name="Model Training — crop coordinates",
            features={"caption": captions}, edge_color=colors, edge_width=1,
            face_color="transparent", translate=translation, scale=scale, metadata=metadata.copy(),
            text={"string": "{caption}", "anchor": "upper_left", "color": "white",
                  "translation": [-size / 2 - 8, -size / 2], "size": 10},
        )
        frames.editable = False
        self._crop_preview_layers = [grid_image, *grid_masks, frames]
        if validity is not None and not np.all(validity):
            channel_names = atlas.channel_names or ('',)
            for channel, name in enumerate(channel_names):
                ignored = np.zeros(atlas.image.shape, dtype=np.uint8)
                for tile, valid in zip(atlas.locations, validity):
                    y, x, size = tile['atlas_y0'], tile['atlas_x0'], tile['model_size']
                    ignored[y:y + size, x:x + size] = (valid[channel] if atlas.channel_names else valid) == 0
                ignored_layer = self.viewer.add_labels(ignored, name='Training — ignored pixels' + (f' — {name}' if name else ''), translate=translation, scale=scale, metadata=metadata.copy(), opacity=0.25)
                ignored_layer.editable = False
                self._crop_preview_layers.append(ignored_layer)

        def show_tile(_layer, event, *, locate=False):
            position = grid_image.world_to_data(tuple(event.position)[-2:])
            row, col = int(np.floor(position[0] / atlas.stride_y)), int(np.floor(position[1] / atlas.stride_x))
            index = row * atlas.columns + col
            if row < 0 or not 0 <= col < atlas.columns or not 0 <= index < len(atlas.locations):
                return
            tile = atlas.locations[index]
            source_image = image
            if tile.get('image_id'):
                source_image = next((layer for layer in self.viewer.layers if layer.metadata.get('label_assistant_review', {}).get('id') == tile['image_id']), None)
            self._status(f"Tile {tile['tile']}: {tile['split']}, {tile['kind']}; source y={tile['y0']}, x={tile['x0']}; foreground {tile['foreground_pixels_original']} source pixels → {tile['foreground_pixels_model']} model pixels.")
            if locate and source_image is not None and any(source_image is item for item in self.viewer.layers):
                source_image.visible = True
                center = source_image.data_to_world((tile["y0"] + tile["size"] / 2, tile["x0"] + tile["size"] / 2))
                y, x, size = tile["y0"], tile["x0"], tile["size"]
                world_corners = np.array([source_image.data_to_world(p) for p in ((y, x), (y + size, x), (y, x + size), (y + size, x + size))])
                extent = np.maximum(np.ptp(world_corners, axis=0), 1e-6)
                self._focus(center, *extent)
                for preview_layer in self._crop_preview_layers:
                    preview_layer.visible = False
                self._gallery_return_view = None
                self.viewer.layers.selection.active = source_image
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
        cluster = None
        if self.training_compute.two_nodes():
            from ._model_providers import spec_for
            config = self._training_config(continue_training)
            if not isinstance(self.prepared, IndexedPatches) or kfold or final_training:
                raise ValueError('Two-node training requires All patches (coverage) with holdout validation. Choose local compute for sampled or k-fold training.')
            if spec_for(config).provider != 'custom':
                raise ValueError('Two-node training currently supports Custom small U-Net models.')
            if config.device == 'cpu' or config.batch_size < 2:
                raise ValueError('Two-node training needs Automatic/CUDA compute and a global batch size of at least 2.')
            cluster = self.training_compute.settings()
        self.learning_chart.set_points([])
        self.export_csv_button.setEnabled(False)
        self.validation_results_label.setText("Final training uses all reviewed patches; no held-out validation loss is measured." if final_training else "Training started. Waiting for the first evaluated epoch.")
        if continue_training and self.model is None:
            raise ValueError("Train or load a model before continuing training.")
        if continue_training and isinstance(self.prepared, IndexedPatches) and self.prepared.validation_seen_before:
            raise ValueError("Review new regions for independent validation, or train a new model.")
        task_info = self.models_panel.bind_sources()
        task_info['annotation_batches'] = self._batch_contexts()
        model_task = getattr(self.model, "identity", {}).get("task_id") if self.model is not None else None
        if continue_training and model_task and model_task != task_info["task_id"]:
            raise ValueError("The selected model belongs to another biological task. Select its task before continuing.")
        self._status("Continuing the current model on the prepared dataset." if continue_training else "Training a new model on the prepared dataset.")
        if isinstance(self.prepared, IndexedPatches):
            selected = self.prepared if final_training else self._selected_training_dataset()
            evaluation = selected.validation.get('evaluation', {})
            if evaluation.get('mode') == 'trial':
                self.validation_results_label.setText(
                    f"Quick trial: {len(selected.indices('training')):,} training patches; "
                    f"trial validation {evaluation['selected_patches']:,} / {evaluation['available_patches']:,} reserved patches each epoch.")
                self._status(self.validation_results_label.text())
            self._lock_sources()
            generation = self._generation
            source_info = {**task_info, "image_layer": self._layer(self.image_combo).name,
                           "image_id": self._layer(self.image_combo).metadata.get("label_assistant_review", {}).get("id"),
                           "ground_truth_layer": self._layer(self.truth_combo).name,
                           "image_shape": list(self.prepared.image.shape),
                           "reviewed_layer": self.reviewed_combo.currentText() if self.prepared.reviewed is not None else None}
            if self.prepared.sources or self.prepared.multilabel:
                source_info.update(pairs=self.prepared.source_info, multilabel=self.prepared.multilabel, image_layer=f'{len(self.prepared.source_info)} annotated images', image_id=None, image_shape=None, ground_truth_layer=None, reviewed_layer=None)
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
            if cluster is not None:
                from ._cluster_training import train_two_nodes
                function = train_two_nodes
                extra['cluster'] = cluster
            self._start_worker(function,
                lambda result: done(result) if generation == self._generation else self._status("Annotations changed during training; previous model retained."),
                selected, self._training_config(continue_training), epochs=self.epochs_spin.value(),
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
        selected = self._selected_training_dataset()
        from ._model_providers import ModelSpec, provider_available
        specs = list(self.models_panel.candidates)
        if len(specs) < 2:
            raise ValueError("Add at least two configurations in Models.")
        for spec in specs:
            if not provider_available(spec["provider"]):
                raise ValueError("Install the SMP optional dependency before running this comparison.")
        source_info = self.models_panel.bind_sources()
        source_info['annotation_batches'] = self._batch_contexts()
        source_info.update(image_layer=self._layer(self.image_combo).name, ground_truth_layer=self._layer(self.truth_combo).name)
        if self.prepared.sources or self.prepared.multilabel:
            source_info.update(pairs=self.prepared.source_info, multilabel=self.prepared.multilabel, image_id=None, image_layer=f'{len(self.prepared.source_info)} annotated images', ground_truth_layer=None)
        self.learning_chart.set_points([])
        self.validation_results_label.setText("Model comparison running on shared spatial splits.")
        generation = self._generation
        self._lock_sources()
        self._start_worker(compare_models,
            lambda result: self._comparison_done(result) if generation == self._generation else self._status("Dataset changed; comparison discarded."),
            selected, self._config(), specs, source_info=source_info,
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

    def _start_full_validation(self):
        from ._coverage_train import evaluate_full_validation
        from ._model_project import layer_id
        if self.model is None or not isinstance(self.prepared, IndexedPatches):
            raise ValueError('Prepare a matching dataset and load or train a model first.')
        if self._signature() != self._prepared_signature:
            raise ValueError('Dataset inputs changed. Prepare the dataset again.')
        current, generation = self.model, self._generation
        image, truth = self._layer(self.image_combo), self._layer(self.truth_combo)
        source = dict(image_id=layer_id(image), ground_truth_id=layer_id(truth),
                      image_layer=image.name, ground_truth_layer=truth.name,
                      pairs=self.prepared.source_info, annotation_batches=self._batch_contexts())
        self._lock_sources()
        self._status('Evaluating fixed model weights on the complete reserved validation pool. No training updates.')
        self._start_worker(evaluate_full_validation,
            lambda result: self._full_validation_done(result, current, generation),
            current, self.prepared, device=self.device_combo.currentData(), source_info=source)

    def _full_validation_done(self, result, model, generation):
        if self.model is not model or self._generation != generation:
            self._status('Model or annotations changed during evaluation; result discarded.')
            return
        from ._model_project import record_model_evaluation
        record_model_evaluation(self.viewer, model, result)
        self.models_panel.refresh()
        self._show_metrics()
        metrics = result['metrics']
        self.validation_results_label.setText(
            f"Full validation · {result['patches']:,} patches: Dice {metrics['dice']:.1%} · "
            f"IoU {metrics['iou']:.1%} · loss {metrics['loss']:.4f}. Fixed retained weights; no training updates.")
        self.model_state_label.setText('Full validation recorded in memory — Save workspace or export the model to keep this evaluation.')
        self._status('Full validation complete. Trial checkpoint-selection scores remain separate. Save workspace or export the model to keep the evaluation.')

    def _training_done(self, model, register=True):
        self.model = model
        from ._reviewed_regions import record_training_usage
        record_training_usage(self.viewer, model)
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
            trial = bool(model.history and model.history[-1].get('validation', {}).get('evaluation', {}).get('mode') == 'trial')
            label = 'Retained checkpoint · Trial validation' if trial else 'Retained checkpoint'
            text = f"{label}: Dice {model.metrics.get('dice',0):.1%} · IoU {model.metrics.get('iou',0):.1%}"
            if 'loss' in model.metrics:
                text += f" · validation loss {model.metrics['loss']:.4f}"
            evaluations = model.history[-1].get('evaluations', []) if model.history else []
            if evaluations:
                full = evaluations[-1]
                text += (f"\nFull validation: Dice {full['metrics']['dice']:.1%} · IoU {full['metrics']['iou']:.1%} · "
                         f"{full['patches']:,} patches; fixed retained weights.")
            self.validation_results_label.setText(text)
        else:
            self.validation_results_label.setText("Final model trained on all reviewed patches. Historical fold results are retained; these weights have no holdout score.")
        stopped = model.history[-1].get("stop_reason") if model.history else None
        prefix = "Early stop; best validation checkpoint retained. " if stopped else ""
        self._status(prefix + f"Model ready ({model.training_device.upper()}). Choose a prediction image and region in Inference. Prediction masks need review.")

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
            evaluations = self.model.history[-1].get('evaluations', []) if self.model.history else []
            if evaluations:
                full = evaluations[-1]
                self.metrics_label.setText(self.metrics_label.text() +
                    f"\nFull validation: Dice {full['metrics']['dice']:.1%} · IoU {full['metrics']['iou']:.1%} · {full['patches']:,} patches.")
            return
        self.metrics_label.setText(
            f"Training: {self.model.training_rounds} rounds · {self.model.training_steps} total steps\n"
            f"Validation: Dice {metrics['dice']:.1%} · IoU {metrics['iou']:.1%}\n"
            f"Precision {metrics['precision']:.1%} · Recall {metrics['recall']:.1%}\n"
            f"Source patch: {self.model.crop_pixels} × {self.model.crop_pixels} px. "
            "These scores do not measure accuracy on a new image."
        )
        trial = self.model.history[-1].get('validation', {}).get('evaluation', {}) if self.model.history else {}
        if trial.get('mode') == 'trial':
            self.metrics_label.setText(self.metrics_label.text().replace('Validation:', 'Trial validation:') +
                f"\nTrial validation: {trial['selected_patches']:,} / {trial['available_patches']:,} reserved patches · seed {trial['seed']}. "
                'These scores selected the checkpoint; they are not full-validation scores.')
        evaluations = self.model.history[-1].get('evaluations', []) if self.model.history else []
        if evaluations:
            full = evaluations[-1]
            values = full['metrics']
            self.metrics_label.setText(self.metrics_label.text() +
                f"\nFull validation: Dice {values['dice']:.1%} · IoU {values['iou']:.1%} · "
                f"{full['patches']:,} patches · fixed retained weights. This is held-out validation, not an independent test set.")
        if self.model.config.class_names:
            self.metrics_label.setText(self.metrics_label.text() + ('\nMultilabel: macro-average over independent channels.' if self.model.config.multilabel else "\nMulticlass: macro-average over foreground classes; background excluded.") +
                "\n" + " · ".join(f"{name}: IoU {metrics.get(f'class_{i}_iou', 0):.1%}, Dice {metrics.get(f'class_{i}_dice', 0):.1%}" for i, name in enumerate(self.model.config.class_names, 1)))
        if self.model.history and self.model.history[-1].get('workflow') == 'coverage':
            history = self.model.history[-1]
            self.metrics_label.setText(self.metrics_label.text() +
                f"\nCoverage: {history['epochs']} complete epoch(s). Retained checkpoint: " +
                (f"epoch {history['retained_epoch']}" if history['retained_epoch'] else "starting model"))
            selection = history.get('training_selection', {})
            if selection.get('mode') in ('percentage', 'count'):
                self.metrics_label.setText(self.metrics_label.text() +
                    f"\nTraining subset: {selection['selected_training_patches']:,} / "
                    f"{selection['available_training_patches']:,} patches · seed {selection['seed']}.")
            if history.get('stop_reason'):
                self.metrics_label.setText(self.metrics_label.text() + "\nEarly stop: " + history['stop_reason'])
            if history.get('learning_rate_events'):
                self.metrics_label.setText(self.metrics_label.text() + f"\nLearning rate reduced to {history['learning_rate_events'][-1]['new_lr']:g}.")

    def _update_prediction_scope(self):
        scope = self.prediction_scope.currentData()
        self.predict_button.setText({"area": "Predict current area", "image": "Predict entire image", "box": "Predict drawn box"}[scope])
        self.draw_prediction_box_button.setVisible(scope == "box")

    def _draw_prediction_box(self):
        layer = self.prediction_box_layer
        if layer is None or not any(layer is item for item in self.viewer.layers):
            layer = self.viewer.add_shapes(name="Model prediction box", ndim=2,
                                           edge_color="yellow", face_color="transparent")
            self.prediction_box_layer = layer
        self.viewer.layers.selection.active = layer
        layer.mode = "add_rectangle"
        self._status("Draw a box over the image, then click Predict drawn box. The latest box is used.")

    def _start_prediction(self):
        if self.model is None:
            raise ValueError("Train or load a model first.")
        layer = self._layer(self.prediction_combo)
        data = array_2d(layer.data)
        threshold = self.threshold_spin.value()
        scope = self.prediction_scope.currentData()
        if scope != "area":
            bounds = None
            if scope == "box":
                box = self.prediction_box_layer
                if box is None or not any(box is item for item in self.viewer.layers) or not len(box.data):
                    raise ValueError("Click Draw prediction box and draw a rectangle first.")
                corners = np.asarray([layer.world_to_data(box.data_to_world(point)) for point in box.data[-1]])
                # Transform round trips can put an integer boundary a few
                # floating-point ulps below its original pixel coordinate.
                corners = np.round(corners, decimals=8)
                bounds = (*np.floor(corners.min(axis=0)).astype(int), *np.ceil(corners.max(axis=0)).astype(int))
                box.mode = "pan_zoom"
            self._status("Predicting the entire image." if scope == "image" else "Predicting inside the drawn box.")
            self._start_worker(
                predict_region, lambda result: self._prediction_done(layer, threshold, result, scope),
                self.model, data, bounds, device=self.device_combo.currentData(),
            )
            return
        self._status("Predicting the current area; only one image crop is read.")
        center = layer.world_to_data(tuple(self.viewer.camera.center)[-2:])
        self._start_worker(
            predict_area, lambda result: self._prediction_done(layer, threshold, result),
            self.model, data, center, device=self.device_combo.currentData(),
        )

    def _prediction_done(self, image_layer, threshold, prediction, scope="area"):
        if not any(image_layer is layer for layer in self.viewer.layers):
            self._status("The prediction image was removed; preview discarded.")
            return
        from ._coordinates import crop_transform_kwargs
        transform = crop_transform_kwargs(image_layer, prediction.origin)
        names = getattr(getattr(self.model, "config", None), "class_names", ())
        multilabel = getattr(getattr(self.model, 'config', None), 'multilabel', False)
        mask = prediction_labels(prediction.probability, threshold, multilabel=multilabel)
        from ._model_project import layer_id
        if multilabel:
            if mask.ndim != 3 or mask.shape[0] != len(names):
                raise ValueError('Prediction channels do not match the loaded model definition.')
            for channel, name in enumerate(names):
                self.viewer.add_labels(mask[channel], name=f'Model Training prediction — {image_layer.name} — {name}', **transform,
                    metadata={'quick_train': dict(experimental=True, source=image_layer.name, source_id=layer_id(image_layer),
                        origin_yx=list(prediction.origin), scope=scope, threshold=threshold, multilabel=True,
                        channel_name=name, channel_index=channel, class_names=list(names), label_names={'0': 'background', '1': name})})
            self._status(f'Added {len(names)} independent {mask.shape[-2]} × {mask.shape[-1]} channel masks. Overlaps are preserved; review each output before reuse.')
            return
        self.viewer.add_labels(
            mask, name=f"Model Training {'preview' if scope == 'area' else 'prediction'} — {image_layer.name}", **transform,
            metadata={"quick_train": {"experimental": True, "source": image_layer.name, "source_id": layer_id(image_layer),
                                      "origin_yx": list(prediction.origin), "threshold": None if names else threshold, "scope": scope,
                                      "class_names": list(names), "label_names": {"0": "background", **{str(i): name for i, name in enumerate(names, 1)}}}},
        )
        extent = {"area": "Only this area was segmented.", "image": "The entire image was segmented.", "box": "The drawn box region was segmented."}[scope]
        self._status(f"Added experimental {mask.shape[0]} × {mask.shape[1]} prediction at y={prediction.origin[0]}, x={prediction.origin[1]}. {extent} Review the mask before use.")

    def _save(self):
        if self.model is None:
            raise ValueError("Train or load a model first.")
        from dataclasses import asdict
        from ._model_project import project
        from ._model_providers import spec_for
        from ._model_naming import checkpoint_filename
        state = project(self.viewer)
        identifier = self.model.identity.get("model_id")
        record = next((row for row in state["models"] if row["id"] == identifier), None)
        if record is None:
            record = {"spec": spec_for(self.model.config).to_dict(), "config": asdict(self.model.config),
                      "history": self.model.history}
        revision = state.get("current_checkpoint") if state.get("current_model") == identifier else None
        entry = next((run for run in record.get("runs", []) if run["revision"] == revision), record)
        filename = checkpoint_filename(record, entry)
        path, _ = QFileDialog.getSaveFileName(self, "Save trained model", filename, "Model checkpoint (*.pt)")
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
        path, _ = QFileDialog.getOpenFileName(self, "Load trained model", "", "Model checkpoint (*.pt)")
        if path:
            self._start_worker(self._load_task, lambda model: self._model_loaded(model, path), path)

    def _model_loaded(self, model, path, register=True):
        prepared = self.prepared
        reuse_prepared = prepared is not None and self._prepared_signature == self._signature()
        if not reuse_prepared:
            self._invalidate_crops()
        image = self.image_combo.currentData()
        if image is not None and not reuse_prepared:
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
        # A prepared dataset owns its validation settings and partition. Loading
        # weights must not silently replace them with checkpoint defaults.
        reuse_prepared = reuse_prepared and self.prepared is prepared and self._prepared_signature == self._signature()
        for record in reversed(model.history) if not reuse_prepared else ():
            settings = record.get('validation', {})
            if settings.get('method') in ('holdout', 'kfold'):
                self.validation_method_combo.setCurrentIndex(self.validation_method_combo.findData(settings['method']))
                self.holdout_spin.setValue(round(settings.get('holdout_fraction', .25)*100))
                self.validation_seed_spin.setValue(settings.get('seed', 42))
                self.fold_spin.setValue(settings.get('fold_count', 5) if settings['method'] == 'kfold' else 5)
                break
        for history in reversed(model.history) if not reuse_prepared else ():
            if history.get('workflow') == 'coverage':
                image = self.image_combo.currentData()
                if image is not None and history.get('source', {}).get('image_layer') == image.name and history.get('source', {}).get('image_shape') == list(array_2d(image.data).shape):
                    key = (id(image), tuple(array_2d(image.data).shape), history['config']['crop_pixels'])
                    from ._annotation_batches import selection
                    current = selection(image, self.truth_combo.currentData())
                    if current['scope'] == 'active' and any(b.get('id') == current['id'] and b.get('revision') == current['revision']
                            for b in history.get('source', {}).get('annotation_batches', [])):
                        key += ('active', current['id'], current['revision'])
                    self._coverage_splits[key] = history['validation_blocks']
                break
        if reuse_prepared and isinstance(prepared, IndexedPatches):
            batched = any(c['scope'] == 'active' for c in self._batch_contexts())
            try:
                if batched:
                    from dataclasses import replace
                    prepared = replace(prepared, validation_seen_before=False)
                prepared = self._protect_batch_holdouts(prepared)
            except ValueError:
                self._invalidate_crops()
                raise
            self.prepared = prepared
            self._patch_subset_cache = None
            # Reassess this checkpoint against the current holdout. A loaded
            # model may have trained on regions reserved by another model.
            size = prepared.crop_pixels
            held_out = {(p['y0'] // size, p['x0'] // size)
                        for p in prepared.locations if p['split'] == 'validation'}
            seen = prepared.validation_seen_before if batched else False
            for region in self._previous_sampled_training(image):
                y0, x0 = max(0, region['y0']), max(0, region['x0'])
                y1 = min(prepared.image.shape[0], region['y0'] + region['size'])
                x1 = min(prepared.image.shape[1], region['x0'] + region['size'])
                if y1 <= y0 or x1 <= x0:
                    continue
                if any((row, col) in held_out
                       for row in range(y0 // size, (y1 - 1) // size + 1)
                       for col in range(x0 // size, (x1 - 1) // size + 1)):
                    seen = True
                    break
            if prepared.sources:
                prior = self._prior_training_image_ids()
                seen = any(p['split'] == 'validation' and p.get('image_id') in prior for p in prepared.locations)
            prepared.validation_seen_before = seen
        self._update_controls()
        if reuse_prepared:
            self._status(f'Model loaded. Prepared dataset retained; {self.train_button.text()} is available.' +
                         (' Continue training is blocked because this model has trained in the reserved validation regions.'
                          if isinstance(prepared, IndexedPatches) and prepared.validation_seen_before else ''))
        else:
            self._status('Model loaded. Prepare the dataset before training; source or dataset settings changed, or no dataset was prepared.')

    @staticmethod
    def _load_task(path, *, cancel):
        if cancel.is_set():
            raise QuickTrainCancelled()
        return load_model(path)

    def attach_annotation_widgets(self, components):
        """Share annotation controls with the Labels editor without duplicating state."""
        if getattr(self, '_annotation_components', None) is components:
            return
        self._annotation_components = components
        review = components._reviewed_area_widget
        library = components.component_library_widget
        self.reviewed_area_widget = review
        self.batch_widget.selected_regions = components._pending_training_tiles
        self.component_library_widget = library
        _, height, width, _ = review.grid_config
        for spin, original in ((self.review_tile_height, height), (self.review_tile_width, width)):
            spin.setRange(original.minimum(), original.maximum())
            spin.setValue(original.value())
            spin.valueChanged.connect(original.setValue)
            original.valueChanged.connect(spin.setValue)
        self.confirm_tiles_on_canvas.setEnabled(True)
        def toggle_canvas_confirmation(enabled):
            if enabled and self._multiple_pairs():
                self.review_pair_toggle.setChecked(True)
            review.advanced_check.setChecked(enabled)
            if enabled and not review.mouse_review_check.isChecked():
                self._status(review.status.text())

        self.confirm_tiles_on_canvas.toggled.connect(toggle_canvas_confirmation)
        review.advanced_check.toggled.connect(self.confirm_tiles_on_canvas.setChecked)
        review.coverage_changed.connect(self.refresh_layers)
        review.feedback.connect(self._status)
        self.show_confirmed_tiles_button.setEnabled(True)
        self.show_confirmed_tiles_button.clicked.connect(lambda: self._safe(review.show_coverage))
        self.select_training_tiles_button.setEnabled(True)
        self.confirm_training_tiles_button.setEnabled(True)
        self.edit_confirm_button.setEnabled(True)
        review.prepare_dataset_button.setEnabled(True)

        def show_labels():
            shell = getattr(self.viewer, '_label_assistant_curation_widget', None)
            if shell is not None:
                shell.show_advanced(1)
                _reveal_panel(shell)
                return
            ancestor = components.parentWidget()
            while ancestor is not None:
                if isinstance(ancestor, QTabWidget) and ancestor.indexOf(components) >= 0:
                    ancestor.setCurrentWidget(components)
                    break
                ancestor = ancestor.parentWidget()

        def select_training_tiles():
            show_labels()
            components._workflow_tabs.setCurrentIndex(0)
            components._loaded_tile_section.setChecked(True)
            components._tile_select_check.setChecked(True)
            scroll = components._workflow_tabs.widget(0)
            scroll.ensureWidgetVisible(components._tile_select_check)

        def edit_confirm_tiles():
            show_labels()
            components._workflow_tabs.setCurrentIndex(0)
            components._workflow_tabs.widget(0).ensureWidgetVisible(review)

        def show_dataset():
            open_training(self.viewer)
            self.workflow_tabs.setCurrentIndex(0)
            self.refresh_layers()

        self.edit_confirm_button.clicked.connect(edit_confirm_tiles)
        review.prepare_dataset_button.clicked.connect(show_dataset)

        def cleanup_masks():
            self.confirm_tiles_on_canvas.setChecked(False)
            show_labels()
            components._workflow_tabs.setCurrentIndex(1)
            components._analysis_scope_combo.setCurrentIndex(components._analysis_scope_combo.findData('area'))
            components._analyze_button.click()
            components._workflow_tabs.widget(1).ensureWidgetVisible(components._small_component_size_spin)

        self.cleanup_masks_button.setEnabled(True)
        self.cleanup_masks_button.clicked.connect(cleanup_masks)

        def confirm_training_tiles():
            components._confirm_training_tiles()
            self.refresh_layers()
            self._status(components._status_label.text())
            self._sources_changed()

        self.select_training_tiles_button.clicked.connect(select_training_tiles)
        self.confirm_training_tiles_button.clicked.connect(confirm_training_tiles)
        self.selection_hint.hide()
        self.training_selection_layout.addWidget(library)
        library.show()
        review.image_combo.show()
        pair_context = QLabel()
        self.editor_pair_context = pair_context
        pair_context.setWordWrap(True)
        components._annotate_page.layout().insertWidget(0, pair_context)
        library.source_layer = lambda: self.truth_combo.currentData()
        library.table.setMaximumHeight(16777215)
        library.layout().setStretchFactor(library.table, 1)

        def sync_sources(*_args):
            from ._reviewed_regions import identity
            image = self.image_combo.currentData()
            image_index = review.image_combo.findData(image)
            if image_index >= 0:
                review.image_combo.setCurrentIndex(image_index)
            truth = self.truth_combo.currentData()
            if image is not None and truth is not None:
                annotation_id = identity(truth)
                if image.metadata.get("label_assistant_annotation_id") != annotation_id:
                    image.metadata = {**image.metadata, "label_assistant_annotation_id": annotation_id}
            pair_context.setText(f"Image: {image.name if image is not None else 'Choose in Training → Dataset'}\nAnnotations: {truth.name if truth is not None else 'Choose in Training → Dataset'}")
            target_index = components._target_combo.findText(truth.name) if truth is not None else -1
            if review.selected_cell is not None and review.selected_cell[0] is not truth:
                review.clear_cell()
            if target_index >= 0:
                components._target_combo.setCurrentIndex(target_index)
            library.refresh()
            review.update_status()

        def sync_editor(*_args):
            truth = next((layer for layer in self.viewer.layers
                          if layer.name == components._target_combo.currentText()), None)
            index = self.truth_combo.findData(truth)
            if index >= 0:
                self.truth_combo.setCurrentIndex(index)

        def restore_pair(*_args):
            image = self.image_combo.currentData()
            annotation_id = image.metadata.get("label_assistant_annotation_id") if image is not None else None
            if annotation_id:
                from ._reviewed_regions import KEY
                for index in range(self.truth_combo.count()):
                    layer = self.truth_combo.itemData(index)
                    if layer.metadata.get(KEY, {}).get("id") == annotation_id:
                        self.truth_combo.setCurrentIndex(index)
                        break

        self.image_combo.currentIndexChanged.connect(restore_pair)
        self.image_combo.currentIndexChanged.connect(sync_sources)
        self.truth_combo.currentIndexChanged.connect(sync_sources)
        def sync_review_image(*_args):
            index = self.image_combo.findData(review.image_combo.currentData())
            if index >= 0:
                self.image_combo.setCurrentIndex(index)

        review.image_combo.currentIndexChanged.connect(sync_review_image)
        components._target_combo.currentIndexChanged.connect(sync_editor)
        restore_pair()
        sync_sources()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.batch_widget.shutdown()
        with suppress(Exception):
            self._batch_events.changed.disconnect(self._batches_changed)
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
                emitter.disconnect(self._source_event)
        self._source_emitters.clear()
        if self.cancel_event is not None:
            self.cancel_event.set()


def quick_train_widget(viewer=None, **kwargs):
    if viewer is None:
        viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found.")
    widget = getattr(viewer, '_label_assistant_training_widget', None)
    if widget is None or widget.closed:
        widget = QuickTrainWidget(viewer)
    return widget


def _reveal_panel(widget):
    """Reveal an existing dock without constructing another panel."""
    ancestor = widget.parentWidget()
    while ancestor is not None:
        if isinstance(ancestor, QDockWidget):
            ancestor.show()
            ancestor.raise_()
            widget.show()
            return True
        ancestor = ancestor.parentWidget()
    widget.show()
    return False


def open_training(viewer):
    """Open the shared training workflow as an independent dock."""
    widget = quick_train_widget(viewer)
    ancestor = widget.parentWidget()
    shell = getattr(viewer, '_label_assistant_curation_widget', None)
    while ancestor is not None and ancestor is not shell and not isinstance(ancestor, QDockWidget):
        ancestor = ancestor.parentWidget()
    if ancestor is None or ancestor is shell:
        viewer.window.add_plugin_dock_widget('napari-label-assistant', 'Model Training')
    _reveal_panel(widget)
    return widget
