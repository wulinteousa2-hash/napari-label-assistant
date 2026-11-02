"""Semantic classes survive preparation, optimization, checkpoints, and inference."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from napari_label_assistant_tools._quick_train import (
    QuickTrainConfig, QuickTrainModel, class_targets, resize_classes,
    prepare_crops, train_prepared_unet, predict_area, predict_region,
    prediction_labels, segmentation_loss, save_model, load_model, _scores,
)
from napari_label_assistant_tools._coverage_train import index_patches, train_indexed
from napari_label_assistant_tools._model_providers import build_model, ModelSpec
from napari_label_assistant_tools._model_project import add_task, remember_model, project


def consume(task):
    while True:
        try:
            next(task)
        except StopIteration as result:
            return result.value


def data_and_config():
    yy, xx = np.indices((128, 128))
    labels = ((xx // 4) % 3).astype(np.uint8)
    image = (labels * 80 + yy % 4).astype(np.uint8)
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8,
                              steps=1, base_channels=4, device="cpu",
                              class_names=("axon", "myelin"))
    return image, labels, config


@pytest.mark.parametrize("names", [("axon",), ("axon", "AXON"), ("axon", ""), ("background", "axon")])
def test_invalid_class_definitions_are_rejected(names):
    with pytest.raises(ValueError, match="class names"):
        QuickTrainConfig(class_names=names).validate()


@pytest.mark.parametrize("workflow", ["coverage", "sampled"])
def test_multiclass_training_checkpoint_and_both_inference_paths(tmp_path, workflow):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    image, truth, config = data_and_config()
    if workflow == "coverage":
        dataset = consume(index_patches(image, truth, config))
        assert set(np.unique(dataset.read(0)[1])) == {0, 1, 2}
        model = consume(train_indexed(dataset, config, epochs=1))
    else:
        dataset = consume(prepare_crops(image, truth, config))
        assert set(np.unique(dataset.masks)) == {0, 1, 2}
        model = consume(train_prepared_unet(dataset, config))
    assert model.network.output.out_channels == 3
    assert np.isfinite(model.metrics["loss"])
    assert all(f"class_{k}_iou" in model.metrics for k in (1, 2))
    path = tmp_path / "axon-myelin.pt"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded.config.class_names == ("axon", "myelin")
    assert json.loads(path.with_suffix(".history.json").read_text())["history"] == loaded.history
    for prediction in (predict_area(loaded, image, (40, 50), device="cpu"),
                       consume(predict_region(loaded, image, device="cpu"))):
        assert prediction.probability.shape[0] == 3
        np.testing.assert_allclose(prediction.probability.sum(axis=0), 1, atol=1e-5)
        mask = prediction_labels(prediction.probability)
        assert mask.ndim == 2
        assert set(np.unique(mask)) <= {0, 1, 2}
    assert consume(predict_region(loaded, image, (5, 7, 67, 99), device="cpu")).probability.shape == (3, 62, 92)


def test_multiclass_loss_ignores_unreviewed_pixels():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 3, 8, 8), requires_grad=True)
    target = torch.zeros((1, 1, 8, 8))
    target[:, :, :, 2:4] = 1
    target[:, :, :, 4:6] = 2
    valid = torch.ones_like(target)
    valid[:, :, :, 6:] = 0
    loss = segmentation_loss(torch, logits, target, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[:, :, :, 6:]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, :6]) > 0


def test_semantic_metrics_penalize_swapped_classes():
    torch = pytest.importorskip("torch")
    target = np.tile(np.array([0, 1, 2, 0], dtype=np.uint8), (4, 1))
    swapped = np.where(target == 1, 2, np.where(target == 2, 1, 0))
    logits = torch.nn.functional.one_hot(torch.from_numpy(swapped).long(), 3).permute(2, 0, 1)[None].float() * 20

    class Constant(torch.nn.Module):
        def forward(self, x):
            return logits

    metrics = _scores(torch, Constant(), np.zeros((1, 4, 4), np.float32), target[None], "cpu", None)
    assert metrics["iou"] == metrics["class_1_iou"] == metrics["class_2_iou"] == 0
    np.testing.assert_array_equal(prediction_labels(logits.numpy()[0]), swapped)


def test_class_ids_are_explicit_and_nearest_resize_keeps_ids():
    labels = np.array([[0, 1], [2, 99]])
    assert class_targets(labels, ()).sum() == 3  # existing instance-label behavior
    with pytest.raises(ValueError, match="label IDs"):
        class_targets(labels, ("axon", "myelin"))
    valid = np.array([[True, True], [True, False]])
    np.testing.assert_array_equal(class_targets(labels, ("axon", "myelin"), valid), [[0, 1], [2, 0]])
    assert set(np.unique(resize_classes(labels[:1, :], (3, 8)))) == {0, 1}


def test_missing_class_and_class_loss_on_downsampling_are_actionable():
    image, truth, config = data_and_config()
    truth[truth == 2] = 1
    with pytest.raises(ValueError, match="missing classes: myelin"):
        consume(index_patches(image, truth, config))
    from napari_label_assistant_tools._coverage_train import _model_supervision
    truth = np.zeros((64, 64), dtype=np.uint8)
    truth[0, 0] = 2
    with pytest.raises(ValueError, match="matching source patch"):
        _model_supervision(truth, np.ones_like(truth, bool), 64, 32, config.class_names)


def test_changed_classes_require_reindexing_and_cannot_continue_old_weights():
    pytest.importorskip("torch")
    image, truth, config = data_and_config()
    dataset = consume(index_patches(image, truth, config))
    with pytest.raises(ValueError, match="Prepare the dataset again"):
        consume(train_indexed(dataset, replace(config, class_names=("myelin", "axon")), epochs=1))
    old_config = replace(config, class_names=("myelin", "axon"))
    old = QuickTrainModel(build_model(old_config), old_config, 32, {}, "cpu")
    with pytest.raises(ValueError, match="class definitions"):
        consume(train_indexed(dataset, config, initial_model=old, epochs=1))


@pytest.mark.parametrize("provider,width", [("custom", 4), ("standard", 16)])
def test_builtin_multiclass_heads(provider, width):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    _, _, config = data_and_config()
    network = build_model(replace(config, model_spec=ModelSpec(provider=provider, width=width).to_dict()))
    with torch.no_grad():
        assert network(torch.zeros((1, 1, 32, 48))).shape == (1, 3, 32, 48)


def test_loading_different_semantics_keeps_existing_task_unchanged():
    pytest.importorskip("torch")
    viewer = SimpleNamespace()
    task = add_task(viewer, "Binary objects", "Any nonzero instance is foreground")
    _, _, config = data_and_config()
    model = QuickTrainModel(build_model(config), config, 32, {}, "cpu")
    remember_model(viewer, model, task_id=task["id"])
    assert task["class_names"] == []
    assert model.identity["task_id"] != task["id"]
    imported = next(t for t in project(viewer)["tasks"] if t["id"] == model.identity["task_id"])
    assert imported["class_names"] == ["axon", "myelin"]


def test_old_binary_checkpoint_without_class_metadata_still_loads(tmp_path):
    torch = pytest.importorskip("torch")
    _, _, config = data_and_config()
    config = replace(config, class_names=())
    model = QuickTrainModel(build_model(config), config, 32, {}, "cpu")
    path = tmp_path / "legacy.pt"
    save_model(model, path)
    payload = torch.load(path, weights_only=True)
    payload["config"].pop("class_names")
    payload.pop("contract")
    torch.save(payload, path)
    loaded = load_model(path)
    assert loaded.config.class_names == ()
    assert loaded.network.output.out_channels == 1


def test_multiclass_task_panel_training_inference_and_task_protection(make_napari_viewer, qtbot):
    pytest.importorskip("torch")
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    image_data, truth_data, _ = data_and_config()
    viewer = make_napari_viewer()
    viewer.add_image(image_data, name="TEM")
    viewer.add_labels(truth_data, name="Tissue classes")
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    panel = widget.models_panel
    panel.task_name.setText("Axon and myelin")
    panel.definition.setText("Axon interiors and myelin sheath")
    panel.class_names_edit.setText("axon, myelin")
    panel.add_task_button.click()
    task = panel.task()
    assert task["class_names"] == ["axon", "myelin"]


    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData("complete"))
    widget.crop_combo.setCurrentIndex(widget.crop_combo.findData(32))
    widget.resolution_combo.setCurrentIndex(widget.resolution_combo.findData(32))
    # Some production presets start at 64; add a small test preset if needed.
    for combo in (widget.crop_combo, widget.resolution_combo):
        if combo.findData(32) < 0:
            combo.addItem("32 px (test)", 32)
        combo.setCurrentIndex(combo.findData(32))
    widget.epochs_spin.setValue(1)
    widget.device_combo.setCurrentIndex(widget.device_combo.findData("cpu"))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert widget.prepared is not None, widget.status_label.text()
    widget.train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=30000)
    assert widget.model is not None, widget.status_label.text()
    assert widget.model.config.class_names == ("axon", "myelin")
    assert not widget.threshold_spin.isEnabled()
    assert "myelin: IoU" in widget.metrics_label.text()
    widget.prediction_scope.setCurrentIndex(widget.prediction_scope.findData("image"))
    widget.predict_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=30000)
    output = viewer.layers[-1]
    assert output.data.shape == truth_data.shape
    assert output.metadata["quick_train"]["class_names"] == ["axon", "myelin"]
    panel.class_names_edit.setText("myelin, axon")
    assert widget.prepared is None
    with pytest.raises(ValueError, match="another task"):
        panel._update_task()
    assert task["class_names"] == ["axon", "myelin"]


def test_multiclass_workspace_preserves_classes_and_checkpoints(tmp_path, make_napari_viewer, qtbot):
    pytest.importorskip("torch")
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    image, truth, config = data_and_config()
    viewer = make_napari_viewer()
    viewer.add_image(image)
    viewer.add_labels(truth, metadata={"quick_train": {"class_names": ["axon", "myelin"], "label_names": {"0": "background", "1": "axon", "2": "myelin"}}})
    task = add_task(viewer, "Axon and myelin", "Two tissue classes", class_names=config.class_names)
    model = QuickTrainModel(build_model(config), config, 32, {}, "cpu")
    remember_model(viewer, model, task_id=task["id"])
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    path = tmp_path / "classes.label-assistant.json"
    save_workspace(viewer, path, complete_package=True)
    restored = make_napari_viewer()
    load_workspace(restored, path)
    state = project(restored)
    assert state["tasks"][0]["class_names"] == ["axon", "myelin"]
    assert restored.layers[1].metadata["quick_train"]["class_names"] == ["axon", "myelin"]
    checkpoint = state["models"][0]["checkpoint"]
    assert load_model(checkpoint).config.class_names == config.class_names


def test_multiclass_checkpoint_relinking_in_model_library(tmp_path):
    from dataclasses import asdict
    from napari_label_assistant_tools._model_library import catalog
    from napari_label_assistant_tools._model_files import verify_relocated_checkpoint
    pytest.importorskip("torch")
    viewer = SimpleNamespace()
    _, _, config = data_and_config()
    task = add_task(viewer, "Tissue", "Axon and myelin", class_names=config.class_names)
    model = QuickTrainModel(build_model(config), config, 32, {}, "cpu")
    remember_model(viewer, model, task_id=task["id"])
    path = tmp_path / "relocated.pt"
    save_model(model, path)
    row = catalog(viewer)[0]
    assert verify_relocated_checkpoint(path, row, asdict(config)) == str(path)


def test_smp_multiclass_contract_is_forwarded(monkeypatch):
    import sys
    torch = pytest.importorskip("torch")
    _, _, config = data_and_config()
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return torch.nn.Conv2d(1, kwargs["classes"], 1)

    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", SimpleNamespace(Unet=constructor, UnetPlusPlus=constructor, DeepLabV3Plus=constructor))
    config = replace(config, model_spec=ModelSpec(provider="smp").to_dict())
    network = build_model(config, initialize=False)
    assert calls[0]["classes"] == 3
    assert network.out_channels == 3
