import numpy as np
import pytest

from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel, save_model
from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
from napari_label_assistant_tools._model_providers import build_model
from napari_label_assistant_tools._model_project import remember_model


@pytest.mark.parametrize('action', ['new_configuration', 'copy_configuration', 'select_checkpoint', 'load_checkpoint'])
def test_prepare_then_pick_model(make_napari_viewer, qtbot, monkeypatch, tmp_path, action):
    viewer = make_napari_viewer()
    truth = (np.indices((128, 128))[1] % 16 < 8).astype(np.uint8)
    viewer.add_image(truth.astype(np.float32), name='image')
    viewer.add_labels(truth, name='annotations')
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.show()
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    cfg = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1, base_channels=4, device='cpu')
    monkeypatch.setattr(widget, '_config', lambda: cfg)
    panel = widget.models_panel
    trained = QuickTrainModel(build_model(cfg), cfg, 32, {'dice': .5, 'iou': .4, 'precision': .6, 'recall': .5}, 'cpu', history=[{'mode': 'new', 'config': {}, 'checkpoints': [], 'steps': 1}])
    task = panel.ensure_task()
    remember_model(viewer, trained, task_id=task['id'])
    panel.refresh()
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert widget.prepared is not None
    prepared = widget.prepared
    assert widget.train_button.isEnabled()
    widget.workflow_tabs.setCurrentIndex(1)
    if action == 'new_configuration':
        panel.provider.setCurrentIndex(panel.provider.findData('standard'))
    elif action == 'copy_configuration':
        panel.configuration_button.click()
    elif action == 'select_checkpoint':
        panel.use_button.click()
    else:
        path = tmp_path / 'model.pt'
        save_model(trained, path)
        monkeypatch.setattr('napari_label_assistant_tools._quick_train_widget.QFileDialog.getOpenFileName', lambda *a, **k: (str(path), ''))
        widget.load_button.click()
        qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    panel.to_train.click()
    assert widget.train_button.isVisible()
    assert widget.prepared is prepared
    assert widget.train_button.isEnabled()
    if action in ('select_checkpoint', 'load_checkpoint'):
        source = widget.truth_combo.currentData()
        source.data = source.data.copy()
        assert widget.prepared is None
        panel.use_button.click()
        qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
        assert widget.prepared is None
        assert not widget.train_button.isEnabled()
        assert 'Prepare the dataset' in widget.status_label.text()


@pytest.mark.parametrize('overlap', [False, True])
def test_checkpoint_rechecks_holdout_without_replacing_prepared_partition(make_napari_viewer, qtbot, monkeypatch, overlap):
    viewer = make_napari_viewer()
    truth = (np.indices((128, 128))[1] % 16 < 8).astype(np.uint8)
    image = viewer.add_image(truth.astype(np.float32), name='image')
    viewer.add_labels(truth, name='annotations')
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    cfg = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1, base_channels=4, device='cpu')
    monkeypatch.setattr(widget, '_config', lambda: cfg)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    prepared = widget.prepared
    signature = widget._prepared_signature
    split_cache = dict(widget._coverage_splits)
    split = 'validation' if overlap else 'training'
    location = next(p for p in prepared.locations if p['split'] == split)
    history = [{'workflow': 'coverage', 'mode': 'new', 'epochs': 1, 'retained_epoch': 1,
                'config': {'crop_pixels': 32}, 'validation_blocks': ['different:partition'],
                'validation': {'method': 'kfold', 'fold_count': 3, 'seed': 99},
                'source': {'image_layer': image.name, 'image_shape': [128, 128]},
                'locations': [{**location, 'split': 'training'}]}]
    model = QuickTrainModel(build_model(cfg), cfg, 32, {'dice': .5, 'iou': .4, 'precision': .6, 'recall': .5}, 'cpu', history=history)
    widget._model_loaded(model, 'in memory', register=False)
    assert widget.prepared is prepared
    assert widget._prepared_signature == signature
    assert widget._coverage_splits == split_cache
    assert widget.validation_method_combo.currentData() == 'holdout'
    assert prepared.validation_seen_before is overlap
    assert widget.train_button.isEnabled()
    assert widget.continue_button.isEnabled() is not overlap
    if overlap:
        assert 'Continue training is blocked' in widget.status_label.text()
    # Switching to an unexposed checkpoint must clear the previous restriction.
    model.history = []
    widget._model_loaded(model, 'unexposed checkpoint', register=False)
    assert not prepared.validation_seen_before
    assert widget.continue_button.isEnabled()
