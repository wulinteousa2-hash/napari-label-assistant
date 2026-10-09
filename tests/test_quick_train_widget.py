from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import napari_label_assistant_tools._quick_train_widget as widget_module
from napari_label_assistant_tools._quick_train import (
    QuickTrainCancelled, QuickTrainPrediction, QuickTrainProgress,
)


def setup_widget(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((256, 256), dtype=np.uint8), name="image")
    truth = viewer.add_labels(np.zeros((256, 256), dtype=np.uint8), name="truth")
    widget = widget_module.QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("complete"))
    widget.workflow_combo.setCurrentIndex(widget.workflow_combo.findData("sampled"))
    return viewer, image, truth, widget


def test_selectors_update_and_preserve_renamed_layer(make_napari_viewer, qtbot):
    viewer, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    second = viewer.add_image(np.zeros((256, 256)), name="second")
    qtbot.waitUntil(lambda: widget.prediction_combo.findText("second") >= 0)
    widget.prediction_combo.setCurrentText("second")
    second.name = "renamed"
    widget.refresh_button.click()
    assert widget.prediction_combo.currentData() is second
    assert widget.prediction_combo.currentText() == "renamed"
    assert not widget.predict_button.isEnabled()


def test_worker_cancellation_restores_ground_truth_without_qt_error(
    make_napari_viewer, qtbot, monkeypatch
):
    _, _, truth, widget = setup_widget(make_napari_viewer, qtbot)
    started = Event()

    def cancellable(*args, cancel):
        started.set()
        yield QuickTrainProgress(1, 2, "Waiting")
        assert cancel.wait(5)
        raise QuickTrainCancelled()

    monkeypatch.setattr(widget, "_prepare_task", cancellable)
    widget.prepare_button.click()
    qtbot.waitUntil(started.is_set)
    assert not truth.editable
    assert not widget.prepare_button.isEnabled()
    widget.cancel_button.click()
    qtbot.waitUntil(lambda: widget.worker is None)
    assert truth.editable
    assert widget.prepare_button.isEnabled()
    assert not widget.train_button.isEnabled()
    assert not widget.predict_button.isEnabled()
    assert "canceled" in widget.status_label.text()


def test_training_panel_unwraps_real_napari_multiscale_layers(
    make_napari_viewer, qtbot, monkeypatch
):
    viewer = make_napari_viewer()
    image = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
    truth = (image > 100).astype(np.uint8)
    viewer.add_image([image, image[::2, ::2]], multiscale=True)
    truth_layer = viewer.add_labels([truth, truth[::2, ::2]], multiscale=True)
    editable_before = truth_layer.editable
    widget = widget_module.QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("complete"))
    widget.workflow_combo.setCurrentIndex(widget.workflow_combo.findData("sampled"))
    captured = []

    def check_sources(image_data, truth_data, _config, *, cancel):
        captured.append((image_data, truth_data))
        yield QuickTrainProgress(1, 2, "Checking sources")
        assert image_data[10:20, 10:20].shape == (10, 10)
        assert truth_data[10:20, 10:20].shape == (10, 10)
        raise QuickTrainCancelled()

    monkeypatch.setattr(widget, "_prepare_task", check_sources)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None)
    assert captured[0][0] is image
    assert captured[0][1] is truth
    assert truth_layer.editable == editable_before
    assert "canceled" in widget.status_label.text()


def test_training_failure_restores_controls(make_napari_viewer, qtbot, monkeypatch):
    _, _, truth, widget = setup_widget(make_napari_viewer, qtbot)

    def failing(*args, cancel):
        yield QuickTrainProgress(1, 2, "Starting")
        raise RuntimeError("Missing training dependency")

    monkeypatch.setattr(widget, "_prepare_task", failing)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None)
    assert truth.editable
    assert "Missing training dependency" in widget.status_label.text()
    assert widget.prepare_button.isEnabled()
    assert not widget.train_button.isEnabled()


def test_preview_is_aligned_and_does_not_modify_source(make_napari_viewer, qtbot):
    viewer, image, _, widget = setup_widget(make_napari_viewer, qtbot)
    image.scale = (2, 3)
    image.translate = (7, 11)
    image.rotate = 30
    image.shear = (0.2,)
    prediction = QuickTrainPrediction(np.ones((32, 32), dtype=np.float32), (70, 90))
    widget._prediction_done(image, 0.5, prediction)
    preview = viewer.layers[-1]
    assert preview.data.shape == (32, 32)
    assert preview.data.all()
    np.testing.assert_allclose(preview.data_to_world((0, 0)), image.data_to_world((70, 90)))
    np.testing.assert_allclose(preview.data_to_world((10, 10)), image.data_to_world((80, 100)))
    assert not image.data.any()
    assert preview.metadata["quick_train"]["experimental"]


def test_prediction_worker_adds_only_one_bounded_preview(
    make_napari_viewer, qtbot, monkeypatch
):
    viewer, image, _, widget = setup_widget(make_napari_viewer, qtbot)
    widget.model = SimpleNamespace()
    widget._update_controls()

    def fake_predict(*args, device, cancel):
        return QuickTrainPrediction(np.ones((32, 32), dtype=np.float32), (40, 50))

    monkeypatch.setattr(widget_module, "predict_area", fake_predict)
    count = len(viewer.layers)
    widget.predict_button.click()
    qtbot.waitUntil(lambda: widget.worker is None)
    assert len(viewer.layers) == count + 1
    assert viewer.layers[-1].data.shape == (32, 32)
    assert "Only this area was segmented" in widget.status_label.text()


def test_real_training_and_prediction_through_panel_buttons(
    make_napari_viewer, qtbot, monkeypatch
):
    pytest.importorskip("torch")
    from napari_label_assistant_tools._quick_train import QuickTrainConfig

    viewer, image, truth, widget = setup_widget(make_napari_viewer, qtbot)
    yy, xx = np.indices(image.data.shape)
    truth.data = ((xx // 16) % 2).astype(np.uint8)
    image.data = (truth.data * 120 + 50 + yy % 8).astype(np.uint8)
    config = QuickTrainConfig(crop_pixels=64, input_pixels=32, samples=8, steps=2, base_channels=4, device="cpu")

    monkeypatch.setattr(widget, "_config", lambda: config)
    widget.device_combo.setCurrentText("CPU")
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert widget.prepared is not None, widget.status_label.text()
    assert widget.model is None
    assert widget._crop_preview_layers == []
    widget.patch_preview_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert len(widget._crop_preview_layers) == 3
    assert widget.train_button.isEnabled()
    widget.train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert widget.model is not None, widget.status_label.text()
    assert "Validation: Dice" in widget.metrics_label.text()
    assert widget.save_button.isEnabled()
    assert widget.predict_button.isEnabled()
    assert truth.editable
    widget.predict_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert len(viewer.layers) == 6
    assert viewer.layers[-1].data.shape == (64, 64)


def test_crop_preview_does_not_become_a_training_source_and_stales_on_edit(
    make_napari_viewer, qtbot, monkeypatch
):
    from napari_label_assistant_tools._quick_train import QuickTrainConfig

    viewer, image, truth, widget = setup_widget(make_napari_viewer, qtbot)
    yy, xx = np.indices(image.data.shape)
    truth.data = ((xx // 16) % 2).astype(np.uint8)
    image.data = (truth.data * 120 + 50 + yy % 8).astype(np.uint8)
    config = QuickTrainConfig(crop_pixels=64, input_pixels=32, samples=8, steps=2, base_channels=4, device="cpu")
    monkeypatch.setattr(widget, "_config", lambda: config)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert widget.prepared is not None, widget.status_label.text()
    assert widget.image_combo.count() == 1
    assert widget.truth_combo.count() == 1
    widget.patch_preview_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    grid_image, grid_mask, frames = widget._crop_preview_layers
    assert not grid_mask.editable
    assert grid_image.extent.world[0, 1] > image.extent.world[1, 1]
    assert grid_mask.data.shape == grid_image.data.shape
    assert len(frames.data) == 8
    assert "Training: 6 crops" in widget.crop_summary_label.text()
    truth.events.paint()
    assert widget.prepared is None
    assert not widget.train_button.isEnabled()
    assert "changed" in widget.crop_summary_label.text()


def test_continue_button_passes_current_model_and_seed_invalidates_crops(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    _, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    assert not widget.continue_button.isEnabled()
    original = SimpleNamespace(config=QuickTrainConfig(base_channels=4))
    widget.model = original
    widget.prepared = object()
    widget._prepared_signature = widget._signature()
    widget._update_controls()
    assert widget.continue_button.isEnabled()
    captured = []
    monkeypatch.setattr(widget, "_start_worker", lambda *args, **kwargs: captured.append((args, kwargs)))
    widget.learning_rate_spin.setValue(0.0001)
    widget.continue_button.click()
    assert captured[-1][1]["initial_model"] is original
    assert captured[-1][1]["learning_rate"] == 0.0001
    assert captured[-1][0][3].base_channels == 4
    widget.train_button.click()
    assert captured[-1][1]["initial_model"] is None
    widget.seed_spin.setValue(43)
    assert widget.prepared is None
    assert not widget.continue_button.isEnabled()


def test_coverage_is_default_and_partial_annotations_require_review_mask(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((256, 256), dtype=np.uint8))
    viewer.add_labels(np.zeros((256, 256), dtype=np.uint8))
    widget = widget_module.QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    assert widget.workflow_combo.currentData() == "coverage"
    assert widget.annotation_combo.currentData() == "partial"
    assert widget.prepare_button.text() == "Prepare dataset"
    widget.prepare_button.click()
    assert widget.worker is None
    assert widget.prepared is None
    assert "Finish review" in widget.status_label.text()


def test_coverage_panel_indexes_trains_and_preserves_review_mask(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._coverage_train import IndexedPatches
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    viewer, image, truth, widget = setup_widget(make_napari_viewer, qtbot)
    widget.workflow_combo.setCurrentIndex(widget.workflow_combo.findData("coverage"))
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("partial"))
    yy, xx = np.indices(image.data.shape)
    truth.data = ((xx // 8) % 2).astype(np.uint8)
    image.data = (truth.data * 120 + 50 + yy % 8).astype(np.uint8)
    reviewed = viewer.add_labels(np.ones_like(truth.data), name="reviewed")
    reviewed.data[::2, ::2] = 0
    original = reviewed.data.copy()
    widget.refresh_layers()
    widget.reviewed_combo.setCurrentIndex(widget.reviewed_combo.findData(reviewed))
    config = QuickTrainConfig(crop_pixels=64, input_pixels=32, samples=8, steps=2, base_channels=4, device="cpu", batch_size=4)
    monkeypatch.setattr(widget, "_config", lambda: config)
    widget.epochs_spin.setValue(1)
    # Mixed reviewed/unreviewed bins would be excluded when downsampling. Preserve resolution here.
    from dataclasses import replace
    config = replace(config, input_pixels=64)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert isinstance(widget.prepared, IndexedPatches), widget.status_label.text()
    assert len(widget.prepared.locations) == 16
    assert "Training: 12 patches" in widget.crop_summary_label.text()
    assert "Training — ignored pixels" not in viewer.layers
    widget.train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget.model is not None, widget.status_label.text()
    assert widget.model.history[-1]['workflow'] == 'coverage'
    assert widget.model.history[-1]['checkpoints'][-1]['unique_patches'] == 12
    assert "in memory" in widget.model_state_label.text()
    widget.patch_preview_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert "Training — ignored pixels" in viewer.layers
    assert widget.truth_combo.findText("Training — ignored pixels") == -1
    np.testing.assert_array_equal(reviewed.data, original)
    assert truth.editable and reviewed.editable
    reviewed.events.paint()
    assert widget.prepared is None
    assert not widget.continue_button.isEnabled()


def test_creating_review_mask_is_disk_backed_and_matches_image_transform(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    viewer, image, truth, widget = setup_widget(make_napari_viewer, qtbot)
    image.scale = (2, 3)
    image.translate = (7, 11)
    path = tmp_path / "reviewed.zarr"
    monkeypatch.setattr(widget_module.QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    widget._create_reviewed_mask()
    reviewed = widget.reviewed_combo.currentData()
    assert reviewed is not truth
    assert type(reviewed.data).__module__.startswith('zarr')
    assert tuple(reviewed.data.shape) == tuple(image.data.shape)
    assert not np.any(np.asarray(reviewed.data))
    np.testing.assert_allclose(reviewed.data_to_world((10, 20)), image.data_to_world((10, 20)))


def test_annotation_edits_cancel_active_coverage_job_and_unlock_sources(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    viewer, image, truth, widget = setup_widget(make_napari_viewer, qtbot)
    widget.workflow_combo.setCurrentIndex(widget.workflow_combo.findData('coverage'))
    reviewed = viewer.add_labels(np.ones_like(truth.data), name='reviewed')
    widget.refresh_layers()
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('partial'))
    widget.reviewed_combo.setCurrentIndex(widget.reviewed_combo.findData(reviewed))
    monkeypatch.setattr(widget, '_config', lambda: QuickTrainConfig(crop_pixels=64, input_pixels=32, samples=8, steps=2, base_channels=4, device='cpu'))
    started = Event()
    def waiting(*args, cancel, **kwargs):
        started.set()
        yield QuickTrainProgress(1, 2, 'Indexing')
        assert cancel.wait(5)
        raise QuickTrainCancelled()
    monkeypatch.setattr(widget, '_index_task', waiting)
    widget.prepare_button.click()
    qtbot.waitUntil(started.is_set)
    assert not truth.editable and not reviewed.editable
    reviewed.events.labels_update(data=np.ones((1, 1), dtype=np.uint8), offset=(0, 0))
    qtbot.waitUntil(lambda: widget.worker is None)
    assert truth.editable and reviewed.editable
    assert widget.prepared is None


def test_coverage_shuffle_change_does_not_require_reindexing(make_napari_viewer, qtbot):
    _, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    widget.workflow_combo.setCurrentIndex(widget.workflow_combo.findData('coverage'))
    widget.prepared = object()
    signature = widget._signature()
    widget.seed_spin.setValue(100)
    assert widget.prepared is not None
    assert widget._signature() == signature


def test_loaded_coverage_model_restores_validation_partition(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel
    _, image, _, widget = setup_widget(make_napari_viewer, qtbot)
    config = QuickTrainConfig(crop_pixels=64, input_pixels=32, base_channels=4)
    history = [{'workflow': 'coverage', 'epochs': 2, 'retained_epoch': 1,
                'config': {'crop_pixels': 64}, 'validation_blocks': ['0:1'],
                'source': {'image_layer': image.name, 'image_shape': [256, 256]}}]
    model = QuickTrainModel(None, config, 64, {'dice': .5, 'iou': .4, 'precision': .5, 'recall': .5}, 'cpu', history=history)
    widget._model_loaded(model, '/tmp/model.pt')
    assert widget._coverage_splits[(id(image), (256, 256), 64)] == ['0:1']
    assert 'Loaded model' in widget.model_state_label.text()


def test_five_workflow_tabs_and_model_loading_lock(make_napari_viewer, qtbot):
    _, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    widget.show()
    assert [widget.workflow_tabs.tabText(i) for i in range(5)] == [
        "1. Dataset", "2. Models", "3. Train", "4. Inference", "5. Advanced",
    ]
    assert widget.workflow_tabs.count() == 5
    assert widget.image_combo.isVisible()
    assert not widget.crop_combo.isVisible()
    assert not widget.prediction_combo.isVisible()
    widget.workflow_tabs.setCurrentIndex(2)
    assert widget.train_button.isVisible()
    assert not widget.image_combo.isVisible()
    widget.workflow_tabs.setCurrentIndex(4)
    assert widget.crop_combo.isVisible()
    assert widget.learning_rate_spin.isVisible()
    assert not widget.image_combo.isVisible()
    widget.workflow_tabs.setCurrentIndex(3)
    assert widget.prediction_combo.isVisible()
    assert not widget.crop_combo.isVisible()
    assert widget.inference_load_button.isEnabled()
    widget.worker = object()
    widget._update_controls()
    assert not widget.inference_load_button.isEnabled()
    assert widget.cancel_button.isVisible()
    assert widget.cancel_button.isEnabled()
    widget.worker = None
    widget._update_controls()


def test_partial_mode_automatically_uses_coverage(make_napari_viewer, qtbot):
    _, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    assert widget.workflow_combo.currentData() == "sampled"
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("partial"))
    assert widget.workflow_combo.currentData() == "coverage"
    assert widget.prepare_button.text() == "Prepare dataset"
    assert widget.samples_spin.isHidden()
    sampled = widget.workflow_combo.findData("sampled")
    assert not widget.workflow_combo.model().item(sampled).isEnabled()
    widget.workflow_combo.setCurrentIndex(sampled)
    assert widget.workflow_combo.currentData() == "coverage"
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("complete"))
    assert widget.workflow_combo.model().item(sampled).isEnabled()
    widget.workflow_combo.setCurrentIndex(sampled)
    assert widget.workflow_combo.currentData() == "sampled"


def test_advanced_augmentation_plateau_and_larger_options(make_napari_viewer, qtbot):
    _, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    widget.show()
    widget.workflow_tabs.setCurrentIndex(4)
    assert widget.crop_combo.findData(4096) >= 0
    assert widget.resolution_combo.findData(2048) >= 0
    assert widget.batch_spin.maximum() == 128
    widget.batch_spin.setValue(16)
    widget.rotation_check.setChecked(False)
    widget.flip_check.setChecked(False)
    widget.intensity_check.setChecked(False)
    config = widget._config()
    assert config.batch_size == 16
    assert not config.augment_rotations and not config.augment_flips and not config.augment_intensity
    assert not widget.contrast_spin.isEnabled()
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("partial"))
    widget.plateau_group.setChecked(True)
    assert widget.reduce_patience_spin.isEnabled()
    widget.reduce_patience_spin.setValue(20)
    widget.stop_patience_spin.setValue(10)
    widget.worker = object()
    widget._update_controls()
    assert not widget.rotation_check.isEnabled()
    assert not widget.plateau_group.isEnabled()
    widget.worker = None
    widget._update_controls()
    assert widget.plateau_group.isEnabled()
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("complete"))
    widget.workflow_combo.setCurrentIndex(widget.workflow_combo.findData("sampled"))
    assert not widget.plateau_group.isEnabled()


def test_validation_controls_and_real_kfold_panel(make_napari_viewer, qtbot, tmp_path):
    import json
    from napari_label_assistant_tools._coverage_train import index_patches
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    def consume(generator):
        while True:
            try:
                next(generator)
            except StopIteration as done:
                return done.value, []
    def fixture_data():
        yy, xx = np.indices((128,128))
        truth = ((xx // 8) % 2).astype(np.uint8)
        return (truth*120+50+yy%8).astype(np.uint8), truth, QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=2, base_channels=4, device='cpu', batch_size=4)
    viewer = make_napari_viewer()
    image, truth, config = fixture_data()
    viewer.add_image(image)
    viewer.add_labels(truth)
    widget = widget_module.QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    widget.validation_method_combo.setCurrentIndex(widget.validation_method_combo.findData('kfold'))
    widget.fold_spin.setValue(2)
    dataset, _ = consume(index_patches(image, truth, config, validation_method='kfold', folds=2))
    widget.prepared = dataset
    widget._prepared_signature = widget._signature()
    widget._update_controls()
    assert not widget.continue_button.isEnabled()
    assert '2 fresh-model' in widget.validation_summary.text()
    widget._config = lambda: config
    widget.epochs_spin.setValue(1)
    widget.train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget.cross_validation is not None
    assert widget.model is None
    assert 'IoU' in widget.validation_results_label.text()
    assert widget.learning_chart.points
    assert widget.final_train_button.isEnabled()
    widget._save_validation_task(widget.cross_validation, tmp_path/'report', cancel=Event())
    assert (tmp_path/'report'/'fold-1.pt').exists()
    report = json.loads((tmp_path/'report'/'validation-report.json').read_text())
    assert report['validation']['fold_count'] == 2
    assert (tmp_path/'report'/'learning-progress.svg').exists()
    widget.final_train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget.model.history[-1]['mode'] == 'final_all_reviewed'
    assert not widget.model.metrics
    assert 'No holdout' in widget.metrics_label.text()
    widget.validation_seed_spin.setValue(10)
    assert widget.prepared is None
    assert widget.cross_validation is None


def test_loss_csv_and_stable_activity_log(make_napari_viewer, qtbot, monkeypatch, tmp_path):
    import csv
    _, _, _, widget = setup_widget(make_napari_viewer, qtbot)
    initial_height = widget.status_label.height()
    widget._status('Validating')
    widget._status('Epoch 30/38: ' + 'long progress message '*30)
    assert widget.status_label.height() == initial_height
    log = widget.activity_log.toPlainText()
    assert 'Validating' in log and 'Epoch 30/38' in log
    update = QuickTrainProgress(100, 200, 'Epoch 1 complete', metrics={'loss': 1.25, 'dice': .8, 'iou': .7}, epoch=1, training_step=100)
    widget._on_progress(update)
    assert 'validation loss 1.2500' in widget.validation_results_label.text()
    assert 'Reindex' not in widget.validation_results_label.text()
    assert widget.export_csv_button.isEnabled()
    widget.chart_mode_combo.setCurrentIndex(widget.chart_mode_combo.findData('loss'))
    assert widget.learning_chart.loss_mode
    path = tmp_path / 'metrics.csv'
    monkeypatch.setattr(widget_module.QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    widget.export_csv_button.click()
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row['validation_loss']) == 1.25
    assert row['step'] == '100'
    assert widget.train_button.parent() is widget.continue_button.parent()


def test_prepare_dataset_leaves_canvas_unchanged_and_enables_training(make_napari_viewer,qtbot,monkeypatch):
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    viewer,image,truth,widget=setup_widget(make_napari_viewer,qtbot)
    truth.data=(np.indices(image.data.shape)[1]%16<8).astype(np.uint8)
    monkeypatch.setattr(widget,'_config',lambda:QuickTrainConfig(crop_pixels=64,input_pixels=32,samples=8,steps=1,base_channels=4,device='cpu'))
    viewer.camera.center=(0,90,110)
    viewer.camera.zoom=2.5
    center,zoom=tuple(viewer.camera.center),viewer.camera.zoom
    layers=list(viewer.layers)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    assert widget.prepared is not None
    assert widget.train_button.isEnabled()
    assert widget.patch_preview_button.isEnabled()
    assert widget._crop_preview_layers==[]
    assert list(viewer.layers)==layers
    np.testing.assert_allclose(viewer.camera.center,center)
    assert viewer.camera.zoom==zoom


def test_gallery_world_scale_alignment_and_return_to_source(make_napari_viewer,qtbot):
    from napari_label_assistant_tools._quick_train import PairedCrops,build_crop_atlas
    viewer,image,truth,widget=setup_widget(make_napari_viewer,qtbot)
    image.scale=(.001,.002)
    image.translate=(-5,6)
    image.rotate=30
    image.shear=(.2,)
    locations=[{'split':split,'kind':'target','y0':32,'x0':64,'size':32,
                'foreground_pixels_original':1024,'foreground_pixels_model':1024} for split in ('training','validation')]
    crops=PairedCrops(np.ones((1,32,32),np.float32),np.ones((1,32,32),np.uint8),
                     np.ones((1,32,32),np.float32),np.ones((1,32,32),np.uint8),32,locations)
    atlas=build_crop_atlas(crops)
    viewer.layers.selection.active=image
    original_center,original_zoom=tuple(viewer.camera.center),viewer.camera.zoom
    validity=np.ones((2,32,32),bool)
    validity[0,:,16:]=False
    widget._show_crop_grid(atlas,image,truth,validity=validity)
    assert len(widget._crop_preview_layers)==4
    gallery=widget._crop_preview_layers[0]
    assert gallery.extent.world[0,-1] > image.extent.world[1,-1]
    for layer in widget._crop_preview_layers:
        np.testing.assert_allclose(layer.scale,gallery.scale)
        np.testing.assert_allclose(layer.translate,gallery.translate)
    np.testing.assert_allclose(gallery.scale,[np.linalg.norm(np.array(image.data_to_world(b))-image.data_to_world((0,0))) for b in ((1,0),(0,1))])
    widget.prepared=crops
    widget._return_to_image()
    assert widget.prepared is crops
    assert all(not layer.visible for layer in widget._crop_preview_layers)
    np.testing.assert_allclose(viewer.camera.center,original_center)
    assert viewer.camera.zoom==original_zoom
    assert viewer.layers.selection.active is image
