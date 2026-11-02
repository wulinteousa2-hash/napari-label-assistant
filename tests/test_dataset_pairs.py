from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from napari_label_assistant_tools._coverage_train import (
    index_pairs, train_indexed, preview_index, partition_dataset, train_kfold, train_final,
)
from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainCancelled, load_model, save_model
from napari_label_assistant_tools._dataset_pairs import DatasetPairsDialog, suggest_pairs, resolve_pairs


def consume(task):
    while True:
        try:
            next(task)
        except StopIteration as done:
            return done.value


def config(**kwargs):
    return QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4,
                            samples=8, steps=1, device='cpu', **kwargs)


def sources(count=4, classes=()):
    result = []
    for i in range(count):
        yy, xx = np.indices((65 + 5 * i, 77 + i))
        truth = ((xx // 4 + yy // 8 + i) % (3 if classes else 2)).astype(np.uint8)
        image = (truth * 70 + yy % 4).astype(np.uint8)
        result.append(dict(image=image, ground_truth=truth, reviewed=None,
                           info=dict(image_id=f'image-{i}', ground_truth_id=f'labels-{i}',
                                     image_layer=f'Image {i}', ground_truth_layer=f'Labels {i}',
                                     split_group=f'subject-{i}', coverage='complete')))
    return result


class SliceOnly:
    def __init__(self, data):
        self.data, self.shape, self.reads = data, data.shape, []

    def __array__(self, *args, **kwargs):
        raise AssertionError('Never materialize an entire source')

    def __getitem__(self, region):
        result = self.data[region]
        assert max(result.shape) <= 32
        self.reads.append(region)
        return result


def test_pairs_stream_sources_and_keep_specimens_in_one_split():
    pairs = sources()
    pairs[1]['info']['split_group'] = pairs[0]['info']['split_group']
    for pair in pairs:
        pair['image'] = SliceOnly(pair['image'])
        pair['ground_truth'] = SliceOnly(pair['ground_truth'])
    dataset = consume(index_pairs(pairs, config(), split_seed=7))
    assert len(dataset.sources) == 4
    assert dataset.total_pixels == sum(np.prod(pair['image'].shape) for pair in pairs)
    assert all(not pair['image'].reads for pair in pairs)
    for image_id in {p['image_id'] for p in dataset.locations}:
        assert len({p['split'] for p in dataset.locations if p['image_id'] == image_id}) == 1
    assert len({p['split'] for p in dataset.locations if p['source_index'] in (0, 1)}) == 1
    for i, item in enumerate(dataset.locations):
        _, y, valid = dataset.read(i)
        original = pairs[item['source_index']]['ground_truth'].data[item['y0']:item['y1'], item['x0']:item['x1']]
        np.testing.assert_array_equal(y[:original.shape[0], :original.shape[1]], original)
        assert not y[valid == 0].any()
    preview = preview_index(dataset)
    assert all('image_id' in item for item in preview.locations)


def test_per_pair_partial_coverage_is_not_reused_for_other_images():
    pairs = sources(2)
    reviewed = np.zeros(pairs[0]['image'].shape, dtype=np.uint8)
    reviewed[:32, :32] = 1
    pairs[0]['reviewed'] = reviewed
    pairs[0]['info']['coverage'] = 'partial'
    dataset = consume(index_pairs(pairs, config()))
    assert dataset.sources[0].reviewed_pixels == 32 * 32
    assert dataset.sources[1].reviewed_pixels == pairs[1]['image'].size
    assert len(dataset.sources[0].locations) == 1  # no within-image split required


@pytest.mark.parametrize('classes', [(), ('axon', 'myelin')])
def test_pooled_training_and_checkpoint_provenance(tmp_path, classes):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    cfg = config(class_names=classes)
    dataset = consume(index_pairs(sources(classes=classes), cfg))
    model = consume(train_indexed(dataset, cfg, epochs=1,
                                 source_info={'pairs': dataset.source_info}))
    assert np.isfinite(model.metrics['loss'])
    assert len({p['image_id'] for p in model.locations}) == 4
    assert model.config.class_names == classes
    path = tmp_path / 'pooled.pt'
    save_model(model, path)
    loaded = load_model(path)
    assert loaded.history[-1]['source']['pairs'] == dataset.source_info
    assert loaded.locations == model.locations


def test_kfold_and_final_training_use_whole_source_groups():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    cfg = config()
    dataset = consume(index_pairs(sources(), cfg, validation_method='kfold', folds=2))
    for fold in dataset.validation['folds']:
        current = partition_dataset(dataset, fold)
        train_ids = {p['image_id'] for p in current.locations if p['split'] == 'training'}
        valid_ids = {p['image_id'] for p in current.locations if p['split'] == 'validation'}
        assert not train_ids & valid_ids
        assert current.read(0)[0].shape == (32, 32)
    result = consume(train_kfold(dataset, cfg, epochs=1))
    assert len(result.models) == 2
    final = consume(train_final(dataset, cfg, cross_validation_report=result.report, epochs=1))
    assert final.metrics == {}
    assert all(item['split'] == 'training' for item in final.locations)


def test_comparison_fingerprints_the_correct_source():
    pytest.importorskip('torch')
    from napari_label_assistant_tools._model_comparison import compare_models
    from napari_label_assistant_tools._model_providers import ModelSpec
    cfg = config()
    dataset = consume(index_pairs(sources(2), cfg))
    result = consume(compare_models(dataset, cfg, [ModelSpec(width=4).to_dict(), ModelSpec(width=8).to_dict()], epochs=1))
    assert len(result.models) == 2
    assert result.report['dataset']['sources'] == dataset.source_info
    assert all('image_digest' in p for p in result.models[0].locations)


def test_previously_trained_images_block_continuation_into_reserved_sources():
    pairs = sources()
    dataset = consume(index_pairs(pairs, config()))
    reserved = {p['image_id'] for p in dataset.locations if p['split'] == 'validation'}
    again = consume(index_pairs(pairs, config(), prior_training_image_ids=reserved))
    assert again.validation_seen_before
    pairs[0]['info']['split_group'] = 'same'
    for pair in pairs[1:]:
        pair['info']['split_group'] = 'same'
    with pytest.raises(ValueError, match='two independent'):
        consume(index_pairs(pairs, config()))
    cancel = Event()
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        consume(index_pairs(sources(), config(), cancel=cancel))


def viewer_pairs(viewer):
    from napari_label_assistant_tools._model_project import layer_id
    rows = []
    for i, pair in enumerate(sources(2)):
        image = viewer.add_image(pair['image'], name=f'Image {i}')
        labels = viewer.add_labels(pair['ground_truth'], name=f'Model Training prediction — Image {i}')
        rows.append(dict(image_id=layer_id(image), image_name=image.name, truth_id=layer_id(labels),
                         included=True, coverage='complete', reviewed_id=None,
                         split_group=f'subject-{i}', confirmed=True))
    return rows


def test_dialog_suggests_without_silently_including_predictions(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    original = viewer_pairs(viewer)
    suggestions = suggest_pairs(viewer)
    assert [row['truth_id'] for row in suggestions] == [row['truth_id'] for row in original]
    assert not any(row['included'] or row['confirmed'] for row in suggestions)
    dialog = DatasetPairsDialog(viewer, original)
    qtbot.addWidget(dialog)
    assert resolve_pairs(viewer, dialog.pairs())
    dialog.table.cellWidget(0, 2).setCurrentIndex(0)
    assert not dialog.table.cellWidget(0, 6).isChecked()
    with pytest.raises(ValueError, match='Missing image or ground truth'):
        resolve_pairs(viewer, dialog.pairs())
    unreviewed = [{**row, 'confirmed': False} for row in original]
    with pytest.raises(ValueError, match='Labels reviewed'):
        resolve_pairs(viewer, unreviewed)


def test_multi_pair_panel_prepares_trains_and_invalidates_any_source(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    definitions = viewer_pairs(viewer)
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.dataset_pairs = definitions
    widget.dataset_source_mode.setCurrentIndex(widget.dataset_source_mode.findData('multiple'))
    assert widget.image_combo.isHidden()
    assert widget.annotation_tabs.isHidden()
    assert widget.workflow_combo.currentData() == 'coverage'
    assert '2 pairs selected' in widget.pairs_summary.text()
    for combo in (widget.crop_combo, widget.resolution_combo):
        combo.addItem('32 px', 32)
        combo.setCurrentIndex(combo.findData(32))
    widget.epochs_spin.setValue(1)
    widget.device_combo.setCurrentIndex(widget.device_combo.findData('cpu'))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget.prepared is not None, widget.status_label.text()
    assert len(widget.prepared.sources) == 2
    widget.train_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=30000)
    assert widget.model is not None, widget.status_label.text()
    assert len(widget.model.history[-1]['source']['pairs']) == 2
    assert all(layer.editable for layer in viewer.layers if hasattr(layer, 'editable'))
    widget.patch_preview_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)
    assert widget._crop_preview_layers
    viewer.layers[3].events.data()  # Not the current single-pair selector's labels.
    assert widget.prepared is None
    widget.dataset_source_mode.setCurrentIndex(widget.dataset_source_mode.findData('current'))
    assert not widget.image_combo.isHidden()
    assert not widget.annotation_tabs.isHidden()


def test_pair_definitions_survive_workspace_save_and_rename(tmp_path, make_napari_viewer, qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer = make_napari_viewer()
    definitions = viewer_pairs(viewer)
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.dataset_pairs = definitions
    widget.dataset_source_mode.setCurrentIndex(widget.dataset_source_mode.findData('multiple'))
    viewer.layers[0].name = 'Renamed image'
    assert resolve_pairs(viewer, definitions)[0][1].name == 'Renamed image'
    path = tmp_path / 'multi.label-assistant.json'
    save_workspace(viewer, path, complete_package=True)
    restored = make_napari_viewer()
    load_workspace(restored, path)
    other = QuickTrainWidget(restored)
    qtbot.addWidget(other)
    assert other._multiple_pairs()
    assert other.dataset_pairs == definitions
    assert len(resolve_pairs(restored, other.dataset_pairs)) == 2
    restored.layers.remove(restored.layers[3])
    with pytest.raises(ValueError, match='Missing image or ground truth'):
        resolve_pairs(restored, other.dataset_pairs)


def test_manage_pairs_can_open_review_controls_without_adding_table_to_panel(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    definitions = viewer_pairs(viewer)
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)

    class AcceptedDialog:
        def __init__(self, *args):
            self.review_pair = definitions[1]

        def exec_(self):
            return True

        def pairs(self):
            return definitions

    monkeypatch.setattr('napari_label_assistant_tools._dataset_pairs.DatasetPairsDialog', AcceptedDialog)
    widget.manage_pairs_button.click()
    assert widget._multiple_pairs()
    assert widget.review_pair_toggle.isChecked()
    assert not widget.annotation_tabs.isHidden()
    assert widget.image_combo.currentData() is viewer.layers[2]
    assert widget.truth_combo.currentData() is viewer.layers[3]
    widget.review_pair_toggle.setChecked(False)
    assert widget.annotation_tabs.isHidden()
    assert widget.image_combo.isHidden()
