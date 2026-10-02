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
    grid_image, grid_mask, frames = widget._crop_preview_layers
    assert not grid_mask.editable
    assert grid_image.translate[1] < 0
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
