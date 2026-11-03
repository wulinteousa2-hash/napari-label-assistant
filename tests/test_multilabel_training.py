"""Overlapping channels are supervised, validated and predicted independently."""
from dataclasses import asdict, replace
from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest

from napari_label_assistant_tools._quick_train import (
    QuickTrainConfig, QuickTrainModel, QuickTrainCancelled, segmentation_loss,
    prediction_labels, predict_area, predict_region, save_model, load_model, build_crop_atlas,
)
from napari_label_assistant_tools._coverage_train import train_indexed, preview_index, train_kfold, train_final
from napari_label_assistant_tools._multilabel import index_multilabel
from napari_label_assistant_tools._model_providers import build_model, ModelSpec
from napari_label_assistant_tools._model_project import layer_id, add_task, remember_model, project
from napari_label_assistant_tools._multilabel_targets import resolve_channels, suggest_channels, MultilabelTargetsDialog


NAMES = ('cell', 'mitochondria', 'nucleus', 'vesicles')


def consume(task):
    while True:
        try:
            next(task)
        except StopIteration as done:
            return done.value


def fixture_data(count=1, channels=4):
    yy, xx = np.indices((128, 128))
    y, x = yy % 32, xx % 32
    masks = [(y >= 2) & (y < 30) & (x >= 2) & (x < 30),
             (y >= 10) & (y < 20) & (x >= 10) & (x < 20),
             (y >= 3) & (y < 8) & (x >= 3) & (x < 8),
             (y >= 23) & (y < 27) & (x >= 23) & (x < 27)]
    masks = [mask.astype(np.uint8) for mask in masks[:channels]]
    image = (masks[0] * 80 + masks[-1] * 100 + xx % 4).astype(np.uint8)
    pairs = [dict(image=image.copy(), channels=[dict(data=mask.copy(), reviewed=None) for mask in masks],
                  info=dict(image_id=f'image-{i}', image_layer=f'Image {i}', split_group=f'subject-{i}',
                            channels=[dict(name=name, target_id=f'{i}-{name}', coverage='complete') for name in NAMES[:channels]]))
             for i in range(count)]
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1,
                              base_channels=4, device='cpu', class_names=NAMES[:channels], multilabel=True)
    return pairs, config


@pytest.mark.parametrize('count', [1, 3])
def test_channel_preparation_preserves_nesting_and_source_identity(count):
    pairs, config = fixture_data(count)
    dataset = consume(index_multilabel(pairs, config))
    assert dataset.multilabel
    assert dataset.class_names == NAMES
    x, y, valid = dataset.read(0)
    assert x.shape == (32, 32)
    assert y.shape == valid.shape == (4, 32, 32)
    assert y[0, 15, 15] == y[1, 15, 15] == 1
    assert y[0, 5, 5] == y[2, 5, 5] == 1
    assert valid.all()
    assert dataset.total_pixels == count * 128 * 128 * 4
    if count > 1:
        for image_id in {p['image_id'] for p in dataset.locations}:
            assert len({p['split'] for p in dataset.locations if p['image_id'] == image_id}) == 1
    preview = preview_index(dataset)
    atlas = build_crop_atlas(preview)
    assert atlas.mask.shape[0] == 4
    assert atlas.channel_names == NAMES


def test_loss_masks_unknown_channels_and_uses_independent_sigmoids():
    torch = pytest.importorskip('torch')
    logits = torch.zeros((1, 4, 8, 8), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :2, 2:6, 2:6] = 1
    valid = torch.ones_like(logits)
    valid[:, 2] = 0
    valid[:, 3, :, 4:] = 0
    loss = segmentation_loss(torch, logits, targets, valid, multilabel=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[:, 2]) == 0
    assert torch.count_nonzero(logits.grad[:, 3, :, 4:]) == 0
    assert torch.count_nonzero(logits.grad[:, :2]) > 0
    probabilities = np.full((4, 8, 8), 0.9)
    assert prediction_labels(probabilities, multilabel=True).all()
    assert not prediction_labels(probabilities, threshold=0.95, multilabel=True).any()


@pytest.mark.parametrize('count', [1, 3])
def test_real_training_checkpoint_and_all_prediction_scopes(tmp_path, count):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    pairs, config = fixture_data(count)
    dataset = consume(index_multilabel(pairs, config))
    model = consume(train_indexed(dataset, config, epochs=1, source_info={'pairs': dataset.source_info, 'multilabel': True}))
    assert np.isfinite(model.metrics['loss'])
    assert all(f'class_{i}_iou' in model.metrics for i in range(1, 5))
    path = tmp_path / 'overlapping.pt'
    save_model(model, path)
    loaded = load_model(path)
    assert loaded.config.multilabel
    assert loaded.config.class_names == NAMES
    assert loaded.config.output_channels == 4  # no extra background channel
    for prediction in (predict_area(loaded, pairs[0]['image'], (20, 20), device='cpu'),
                       consume(predict_region(loaded, pairs[0]['image'], device='cpu')),
                       consume(predict_region(loaded, pairs[0]['image'], (5, 7, 67, 99), device='cpu'))):
        assert prediction.probability.shape[0] == 4
        assert np.isfinite(prediction.probability).all()
    assert loaded.history[-1]['source']['pairs'] == dataset.source_info


def test_unknown_and_partial_channels_have_independent_validity():
    pairs, config = fixture_data(3, channels=2)
    # One channel is not annotated on image 0; enough independent images remain
    # for a split containing positive/negative examples of every channel.
    pairs[0]['channels'][1] = None
    partial = np.ones((128, 128), np.uint8)
    partial[:, 16:32] = 0
    pairs[1]['channels'][1]['reviewed'] = partial
    dataset = consume(index_multilabel(pairs, config, split_seed=42))
    source = dataset.sources[0]
    _, targets, valid = source.read(0)
    assert not targets[1].any() and not valid[1].any()
    assert valid[0].all()
    _, _, valid = dataset.sources[1].read(0)
    assert valid[0].all() and not valid[1, :, 16:32].any()
    pairs[1]['channels'][1]['data'][0, 20] = 99  # unknown pixels are not negatives
    with pytest.raises(ValueError, match='changed'):
        dataset.sources[1].read(0)  # edits still invalidate the old index
    refreshed = consume(index_multilabel(pairs, config, split_seed=42))
    refreshed.sources[1].read(0)  # nonbinary IDs in unknown pixels are ignored


def test_nonbinary_targets_missing_examples_and_cancellation():
    pairs, config = fixture_data(channels=2)
    pairs[0]['channels'][1]['data'][15, 15] = 2
    with pytest.raises(ValueError, match='binary 0/1'):
        consume(index_multilabel(pairs, config))
    pairs, config = fixture_data(channels=2)
    pairs[0]['channels'][1]['data'][:] = 0
    with pytest.raises(ValueError, match='positive and negative'):
        consume(index_multilabel(pairs, config))
    cancel = Event()
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        consume(index_multilabel(pairs, config, cancel=cancel))


def test_multilabel_cannot_continue_exclusive_model_with_same_head_size():
    pytest.importorskip('torch')
    pairs, config = fixture_data(channels=4)
    dataset = consume(index_multilabel(pairs, config))
    exclusive = replace(config, class_names=('cell', 'mitochondria', 'nucleus'), multilabel=False)
    old = QuickTrainModel(build_model(exclusive), exclusive, 32, {}, 'cpu')
    with pytest.raises(ValueError, match='class definitions'):
        consume(train_indexed(dataset, config, initial_model=old, epochs=1))


def test_one_named_channel_retains_channel_axis_at_inference():
    torch = pytest.importorskip('torch')
    _, config = fixture_data(channels=1)
    model = QuickTrainModel(build_model(config), config, 32, {}, 'cpu')
    image = np.zeros((45, 53), np.uint8)
    assert predict_area(model, image, (10, 10), device='cpu').probability.shape == (1, 32, 32)
    assert consume(predict_region(model, image, device='cpu')).probability.shape == (1, 45, 53)


def test_sigmoid_predictions_can_all_be_positive_at_the_same_pixel():
    torch = pytest.importorskip('torch')
    pairs, config = fixture_data()
    network = torch.nn.Conv2d(1, 4, 1)
    with torch.no_grad():
        network.weight.zero_()
        network.bias.fill_(float(np.log(9)))
    model = QuickTrainModel(network, config, 32, {}, 'cpu')
    for prediction in (predict_area(model, pairs[0]['image'], (20, 20), device='cpu'),
                       consume(predict_region(model, pairs[0]['image'], device='cpu'))):
        np.testing.assert_allclose(prediction.probability, .9, atol=1e-5)
        assert prediction_labels(prediction.probability, multilabel=True).all()


def test_nested_features_receive_per_channel_perfect_scores():
    torch = pytest.importorskip('torch')
    from napari_label_assistant_tools._coverage_train import _validate
    pairs, config = fixture_data()
    dataset = consume(index_multilabel(pairs, config))
    target = np.stack([target['data'][:32, :32] for target in pairs[0]['channels']])
    logits = torch.from_numpy(target.astype(np.float32) * 20 - 10)[None]

    class Perfect(torch.nn.Module):
        def forward(self, x):
            return logits

    metrics = consume(_validate(torch, Perfect(), dataset, 'cpu', None, step=1, total=1))
    assert metrics['iou'] == metrics['dice'] == 1
    assert all(metrics[f'class_{i}_iou'] == 1 for i in range(1, 5))


def test_multilabel_kfold_final_and_comparison_keep_channel_semantics():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    from napari_label_assistant_tools._model_comparison import compare_models
    pairs, config = fixture_data(2, channels=2)
    dataset = consume(index_multilabel(pairs, config, validation_method='kfold', folds=2))
    result = consume(train_kfold(dataset, config, epochs=1))
    assert len(result.models) == 2
    assert all(model.config.multilabel for model in result.models)
    final = consume(train_final(dataset, config, cross_validation_report=result.report, epochs=1))
    assert final.metrics == {} and final.config.multilabel
    dataset = consume(index_multilabel(pairs, config))
    comparison = consume(compare_models(dataset, config, [ModelSpec(width=4).to_dict(), ModelSpec(width=8).to_dict()], epochs=1))
    assert all(model.config.multilabel for model in comparison.models)
    assert comparison.report['dataset']['multilabel']


def test_target_reads_are_bounded_and_never_materialize_whole_images():
    class SliceOnly:
        def __init__(self, data):
            self.data, self.shape = data, data.shape
            self.reads = 0

        def __array__(self, *args, **kwargs):
            raise AssertionError('Full-source conversion is forbidden')

        def __getitem__(self, region):
            self.reads += 1
            result = self.data[region]
            assert max(result.shape) <= 32
            return result

    pairs, config = fixture_data()
    pairs[0]['image'] = SliceOnly(pairs[0]['image'])
    for target in pairs[0]['channels']:
        target['data'] = SliceOnly(target['data'])
    dataset = consume(index_multilabel(pairs, config))
    before = pairs[0]['image'].reads
    assert dataset.read(0)[1].shape == (4, 32, 32)
    assert pairs[0]['image'].reads == before + 1  # image normalization is shared


def viewer_targets(viewer, count=1, channels=4):
    pairs, config = fixture_data(count, channels)
    definitions = []
    for pair in pairs:
        image = viewer.add_image(pair['image'], name=pair['info']['image_layer'])
        assignments = {}
        for name, target in zip(config.class_names, pair['channels']):
            layer = viewer.add_labels(target['data'], name=f'{image.name} — {name}')
            assignments[name] = dict(target_id=layer_id(layer), coverage='complete', reviewed_id=None, confirmed=False)
        definitions.append(dict(image_id=layer_id(image), image_name=image.name,
                                included=True, split_group=image.name, channels=assignments))
    return definitions, config


def test_channel_dialog_assignments_are_explicit_stable_and_independent(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    definitions, config = viewer_targets(viewer)
    suggested = suggest_channels(viewer, config.class_names)
    assert not suggested[0]['included']
    assert all(suggested[0]['channels'][name]['target_id'] == definitions[0]['channels'][name]['target_id'] for name in NAMES)
    dialog = MultilabelTargetsDialog(viewer, NAMES, definitions)
    qtbot.addWidget(dialog)
    assert len(resolve_channels(viewer, NAMES, dialog.assignments())) == 1
    viewer.layers[1].name = 'Renamed cell'
    assert resolve_channels(viewer, NAMES, dialog.assignments())[0][2][0][1].name == 'Renamed cell'
    dialog.table.cellWidget(1, 4).setCurrentIndex(2)  # explicit unknown channel
    dialog.table.cellWidget(1, 3).setCurrentIndex(0)
    assert resolve_channels(viewer, NAMES, dialog.assignments())[0][2][1] is None
    viewer.layers.remove(viewer.layers[1])
    with pytest.raises(ValueError, match='assign a target mask'):
        resolve_channels(viewer, NAMES, dialog.assignments())


def test_multilabel_panel_trains_previews_and_outputs_separate_masks(make_napari_viewer, qtbot):
    pytest.importorskip('torch')
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    definitions, config = viewer_targets(viewer)
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    panel = widget.models_panel
    panel.output_mode.setCurrentIndex(panel.output_mode.findData(True))
    panel.class_names_edit.setText(', '.join(NAMES))
    widget.channel_assignments = definitions
    widget._sources_changed()
    assert widget._multiple_pairs()
    assert widget.manage_pairs_button.text() == 'Manage channels…'
    for combo in (widget.crop_combo, widget.resolution_combo):
        combo.addItem('32 px', 32)
        combo.setCurrentIndex(combo.findData(32))
    widget.device_combo.setCurrentIndex(widget.device_combo.findData('cpu'))
    widget.epochs_spin.setValue(1)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget.prepared is not None, widget.status_label.text()
    widget.train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=30000)
    assert widget.model is not None, widget.status_label.text()
    assert widget.model.config.multilabel
    assert widget.threshold_spin.isEnabled()
    widget.patch_preview_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert len([layer for layer in widget._crop_preview_layers if 'crop ground truth' in layer.name]) == 4
    from napari_label_assistant_tools._quick_train import QuickTrainPrediction
    widget._prediction_done(viewer.layers[0], 0.5, QuickTrainPrediction(np.full((4, 32, 32), .9, np.float32), (5, 7)))
    outputs = list(viewer.layers)[-4:]
    assert all(layer.data.all() for layer in outputs)
    assert [layer.metadata['quick_train']['channel_name'] for layer in outputs] == list(NAMES)
    for layer in outputs:
        np.testing.assert_allclose(layer.data_to_world((0, 0)), viewer.layers[0].data_to_world((5, 7)))


def test_multilabel_workspace_and_model_library_relinking(tmp_path, make_napari_viewer, qtbot):
    pytest.importorskip('torch')
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    from napari_label_assistant_tools._model_files import verify_relocated_checkpoint
    from napari_label_assistant_tools._model_library import catalog
    viewer = make_napari_viewer()
    definitions, config = viewer_targets(viewer)
    task = add_task(viewer, 'Cell compartments', 'Nested masks', class_names=NAMES, multilabel=True)
    model = QuickTrainModel(build_model(config), config, 32, {}, 'cpu')
    remember_model(viewer, model, task_id=task['id'])
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.channel_assignments = definitions
    path = tmp_path / 'channels.label-assistant.json'
    save_workspace(viewer, path, complete_package=True)
    restored = make_napari_viewer()
    load_workspace(restored, path)
    other = QuickTrainWidget(restored)
    qtbot.addWidget(other)
    assert other._is_multilabel()
    assert other._multiple_pairs()
    assert not other.dataset_source_mode.model().item(0).isEnabled()
    assert other.channel_assignments == definitions
    assert project(restored)['tasks'][0]['multilabel']
    checkpoint = project(restored)['models'][0]['checkpoint']
    assert load_model(checkpoint).config.multilabel
    assert verify_relocated_checkpoint(checkpoint, catalog(restored)[0], asdict(config)) == checkpoint


def test_review_coverage_is_separate_for_each_nested_target(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._reviewed_regions import associate, associated_mask, ReviewedAreaWidget
    from napari_label_assistant_tools._dynamic_edit import TileBounds
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    definitions, config = viewer_targets(viewer, channels=2)
    image, cell, mito = list(viewer.layers)
    cell_coverage = viewer.add_labels(np.zeros((128, 128), np.uint8), name='Cell reviewed')
    mito_coverage = viewer.add_labels(np.zeros((128, 128), np.uint8), name='Mito reviewed')
    associate(cell_coverage, image, cell)
    associate(mito_coverage, image, mito)
    controller = SimpleNamespace(source=cell, tile=cell, bounds=TileBounds(0, 0, 32, 32),
                                 loading=False, stroke_active=False, dirty=False)
    review = ReviewedAreaWidget(viewer, controller)
    qtbot.addWidget(review)
    review.mark(1)
    controller.source = controller.tile = mito
    controller.bounds = TileBounds(64, 64, 96, 96)
    review.mark(1)
    assert cell_coverage.data[:32, :32].all()
    assert not cell_coverage.data[64:96, 64:96].any()
    assert mito_coverage.data[64:96, 64:96].all()
    assert not mito_coverage.data[:32, :32].any()
    assert associated_mask(viewer, image, cell) is cell_coverage
    assert associated_mask(viewer, image, mito) is mito_coverage
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.models_panel.output_mode.setCurrentIndex(1)
    widget.models_panel.class_names_edit.setText('cell, mitochondria')
    widget.truth_combo.setCurrentIndex(widget.truth_combo.findData(mito))
    assert widget.truth_combo.currentData() is mito
    assert widget.reviewed_combo.currentData() is mito_coverage


def test_channel_dialog_can_start_empty_and_use_existing_mask_layers(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    definitions, config = viewer_targets(viewer, channels=2)
    dialog = MultilabelTargetsDialog(viewer, (), [])
    qtbot.addWidget(dialog)
    assert dialog.table.rowCount() == 0
    dialog._apply()
    assert not dialog.result()  # empty names show guidance instead of accepting
    dialog.use_layers_button.click()
    assert dialog.names == tuple(layer.name for layer in list(viewer.layers)[1:])
    assert dialog.table.rowCount() == 2
    assignments = dialog.assignments()
    assert assignments[0]['included']
    assert all(value['foreground_mode'] == 'nonzero' for value in assignments[0]['channels'].values())
    assert len(resolve_channels(viewer, dialog.names, assignments)) == 1
    original_ids = [value['target_id'] for value in assignments[0]['channels'].values()]
    dialog.names_edit.setText('macrophage, organelles')
    assert dialog._update_channels()
    assert [value['target_id'] for value in dialog.assignments()[0]['channels'].values()] == original_ids


def test_nonzero_instance_ids_are_an_explicit_read_only_channel_option():
    pairs, config = fixture_data(channels=2)
    pairs[0]['channels'][0]['data'] *= 4
    before = pairs[0]['channels'][0]['data'].copy()
    with pytest.raises(ValueError, match='binary 0/1'):
        consume(index_multilabel(pairs, config))
    pairs[0]['channels'][0]['foreground_mode'] = 'nonzero'
    dataset = consume(index_multilabel(pairs, config))
    _, targets, _ = dataset.read(0)
    assert targets[0].max() == 1
    np.testing.assert_array_equal(targets[0], before[:32, :32] != 0)
    np.testing.assert_array_equal(pairs[0]['channels'][0]['data'], before)


def test_manage_channels_names_masks_and_preserves_existing_trained_task(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    pairs, config = fixture_data(channels=2)
    image = viewer.add_image(pairs[0]['image'], name='macrophage_img')
    cell = viewer.add_labels(pairs[0]['channels'][0]['data'] * 3, name='macrophage_m')
    viewer.add_labels(pairs[0]['channels'][1]['data'], name='macrophage_o')
    original = add_task(viewer, 'macrophage', 'cell')
    old_config = replace(config, class_names=(), multilabel=False)
    remember_model(viewer, QuickTrainModel(None, old_config, 32, {}, 'cpu'), task_id=original['id'])
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.models_panel.output_mode.setCurrentIndex(1)
    assert not widget.models_panel.class_names_edit.text()

    def accept_existing_masks(dialog):
        assert not dialog.names
        dialog.use_layers_button.click()
        dialog._apply()
        return dialog.result()

    monkeypatch.setattr(MultilabelTargetsDialog, 'exec_', accept_existing_masks)
    widget.manage_pairs_button.click()
    assert widget.models_panel.class_names() == ('macrophage_m', 'macrophage_o')
    assert widget.models_panel.task()['id'] != original['id']
    assert not original.get('multilabel')
    assert not original['class_names']
    assert len(project(viewer)['models']) == 1
    assert widget.channel_assignments[0]['included']
    for combo in (widget.crop_combo, widget.resolution_combo):
        combo.addItem('32 px', 32)
        combo.setCurrentIndex(combo.findData(32))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget.prepared is not None, widget.status_label.text()
    assert cell.data.max() == 3  # no source pixels were rewritten


def test_using_layer_names_does_not_mix_multiple_images(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    viewer_targets(viewer, count=2, channels=2)
    dialog = MultilabelTargetsDialog(viewer, (), [])
    qtbot.addWidget(dialog)
    assert not dialog.use_layers_button.isEnabled()
    dialog.names_edit.setText('cell, mitochondria')
    assert dialog._update_channels()
    assert dialog.table.rowCount() == 4
    assert not any(row['included'] for row in dialog.assignments())


def test_empty_name_preparation_opens_setup_and_cancel_keeps_data(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    viewer_targets(viewer, channels=2)
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.models_panel.output_mode.setCurrentIndex(1)
    opened = []

    def cancel(dialog):
        opened.append(dialog)
        return False

    monkeypatch.setattr(MultilabelTargetsDialog, 'exec_', cancel)
    widget.prepare_button.click()
    assert len(opened) == 1
    assert widget.worker is None
    assert widget.prepared is None
    assert 'Open Manage channels' in widget.status_label.text()
