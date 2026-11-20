"""Annotation semantics and the unified review-to-dataset workflow."""
import numpy as np
import pytest

from napari_label_assistant_tools._component_library import TrainingSupervision
from napari_label_assistant_tools._reviewed_regions import (
    KEY, effective_areas, region_reviewed, review_progress,
)
from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget


def test_stale_coverage_pixels_outside_saved_approvals_are_not_training_data():
    truth = np.zeros((32, 48), dtype=np.uint8)
    coverage = np.ones_like(truth)
    areas = [[3, 5, 13, 15], [10, 12, 20, 25]]
    supervision = TrainingSupervision(truth, coverage, reviewed_areas=areas,
                                     invalid_regions=[[8, 8, 12, 12]])
    expected = np.zeros_like(truth)
    expected[3:13, 5:15] = 1
    expected[10:20, 12:25] = 1
    expected[8:12, 8:12] = 0
    np.testing.assert_array_equal(supervision[slice(0,32),slice(0,48)], expected)
    np.testing.assert_array_equal(supervision[slice(7,18),slice(9,30)], expected[7:18,9:30])
    empty = TrainingSupervision(truth, coverage, reviewed_areas=[])
    assert not empty[slice(0,32),slice(0,48)].any()
    # External masks without rectangle metadata retain their explicit coverage.
    external = TrainingSupervision(truth, coverage)
    assert external[slice(0,32),slice(0,48)].all()


def test_review_progress_matches_pixel_reference_without_image_reads():
    rng = np.random.default_rng(17)
    for _ in range(100):
        shape = (23, 31)
        rectangles = []
        for i in range(8):
            y = sorted(rng.choice(shape[0] + 1, 2, replace=False))
            x = sorted(rng.choice(shape[1] + 1, 2, replace=False))
            rectangles.append([int(y[0]), int(x[0]), int(y[1]), int(x[1])])
        state = dict(areas=rectangles[:6], invalid_regions=rectangles[6:])
        valid = np.zeros(shape, bool)
        for a, b, c, d in state['areas']:
            valid[a:c, b:d] = True
        for a, b, c, d in state['invalid_regions']:
            valid[a:c, b:d] = False
        sy, sx = map(int, rng.integers(2, 9, size=2))
        expected = sum(valid[y:y+sy, x:x+sx].all()
                       for y in range(0, shape[0], sy) for x in range(0, shape[1], sx))
        actual, total = review_progress(shape, (sy, sx), state)
        assert actual == expected
        assert total == ((23 + sy - 1) // sy) * ((31 + sx - 1) // sx)
    assert review_progress((100000, 100000), (100, 100),
                           dict(areas=[[0, 0, 100000, 100000]])) == (1000000, 1000000)


def test_reviewed_foreground_background_and_unknown_are_distinct():
    truth = np.zeros((8, 8), np.uint8)
    truth[1, 1] = 1
    coverage = np.zeros_like(truth)
    coverage[:4, :4] = 1
    coverage[4:6, 4:6] = 1  # Reviewed background-only cell.
    supervision = TrainingSupervision(truth, coverage)
    valid = supervision[slice(0, 8), slice(0, 8)]
    assert valid[1, 1] and truth[1, 1] == 1
    assert valid[2, 2] and truth[2, 2] == 0
    assert valid[5, 5] and truth[5, 5] == 0
    assert not valid[7, 7] and truth[7, 7] == 0
    from napari_label_assistant_tools._quick_train import segmentation_loss
    torch = pytest.importorskip('torch')
    logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
    segmentation_loss(torch, logits, torch.tensor(truth, dtype=torch.float32)[None, None],
                      torch.tensor(valid, dtype=torch.float32)[None, None]).backward()
    assert logits.grad[0, 0, 1, 1] < 0
    assert logits.grad[0, 0, 2, 2] > 0
    assert logits.grad[0, 0, 5, 5] > 0
    assert logits.grad[0, 0, 7, 7] == 0


def test_positive_only_supervision_does_not_train_unknown_pixels(tmp_path):
    from test_component_library import make_library
    from napari_label_assistant_tools._quick_train import segmentation_loss
    torch = pytest.importorskip('torch')
    truth = np.zeros((8, 8), np.uint8)
    truth[2:4, 2:4] = 1
    library = make_library(tmp_path, truth)
    library.select(library.records()[0]['id'], True)
    valid = TrainingSupervision(truth, library=library)[slice(0, 8), slice(0, 8)]
    np.testing.assert_array_equal(valid, truth)
    logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
    target = torch.tensor(truth, dtype=torch.float32)[None, None]
    usable = torch.tensor(valid, dtype=torch.float32)[None, None]
    segmentation_loss(torch, logits, target, usable).backward()
    assert torch.count_nonzero(logits.grad[usable == 0]) == 0
    assert torch.count_nonzero(logits.grad[usable != 0]) == 4


def test_review_progress_revoked_until_explicit_reapproval(make_napari_viewer, qtbot):
    from test_reviewed_regions import setup, configure_mouse, click
    viewer, _, truth, mask, _, review = setup(make_napari_viewer, qtbot)
    configure_mouse(review, truth, qtbot, cell_size=16)
    click(review, truth, 4, 4)
    review.mark(1)
    assert review.progress_label.text() == "Confirmed grid tiles: 1 / 4"
    assert "Status: Confirmed for dataset" in review.cell_label.text()
    from napari_label_assistant_tools._component_library import annotation_changed
    truth.data[4, 4] = 0
    annotation_changed(viewer, truth, (4, 4, 5, 5))
    assert review.progress_label.text() == "Confirmed grid tiles: 0 / 4"
    assert "Needs recheck" in review.cell_label.text()
    assert not region_reviewed((0, 0, 16, 16), effective_areas(mask.metadata[KEY]))
    review.mark(1)
    assert review.progress_label.text() == "Confirmed grid tiles: 1 / 4"
    review.remove_review()
    assert review.progress_label.text() == "Confirmed grid tiles: 0 / 4"


def test_unified_overview_and_outdated_state(make_napari_viewer, qtbot):
    from test_reviewed_regions import setup
    viewer, _, truth, _, _, review = setup(make_napari_viewer, qtbot)
    training = QuickTrainWidget(viewer)
    qtbot.addWidget(training)
    assert not hasattr(training, 'annotation_tabs')
    assert training.training_selection_panel.isHidden()
    training.selection_toggle.setChecked(True)
    assert not training.training_selection_panel.isHidden()
    review.mark(1)
    assert '1 confirmed tiles available' in training.annotation_sources_label.text()
    assert training.dataset_state_label.text() == 'Ready to prepare'
    training.prepared = object()
    training._update_training_overview()
    assert training.dataset_state_label.text() == 'Prepared'
    truth.events.data()
    assert training.prepared is None
    assert 'Outdated' in training.dataset_state_label.text()
    assert 'invalid_regions' in training.reviewed_combo.currentData().metadata[KEY]


def test_dataset_fits_narrow_dock(make_napari_viewer, qtbot):
    from napari_label_assistant_tools import label_assistant_widget
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), np.uint8), name='Image with a long descriptive microscopy filename')
    viewer.add_labels(np.zeros((32, 32), np.uint8), name='Curated')
    panel = label_assistant_widget(viewer)
    qtbot.addWidget(panel)
    panel.resize(480, 950)
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    panel.show_training()
    panel.show()
    training = panel.training
    qtbot.waitUntil(lambda: training.dataset_scroll.horizontalScrollBar().maximum() == 0)
    training.selection_toggle.setChecked(True)
    qtbot.waitUntil(lambda: training.dataset_scroll.horizontalScrollBar().maximum() == 0)


def test_foreground_only_overview_does_not_claim_background(tmp_path, make_napari_viewer, qtbot):
    from test_component_library import consume
    from napari_label_assistant_tools._component_library import ensure_library, notify_library
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), np.uint8))
    truth = viewer.add_labels(np.eye(32, dtype=np.uint8))
    library = ensure_library(truth)
    consume(library.update(truth.data))
    training = QuickTrainWidget(viewer)
    qtbot.addWidget(training)
    library.select(library.records()[0]['id'], True)
    notify_library(truth)
    assert 'saved object selections' in training.annotation_sources_label.text()
    assert 'Background has not been verified' in training.supervision_label.text()
    assert 'Foreground only' in training.dataset_state_label.text()


def test_prepared_overview_counts_supervision_from_index(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._coverage_train import index_patches
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    from test_component_library import consume
    image = np.zeros((64, 64), np.uint8)
    truth = image.copy()
    truth[4:8, 4:8] = truth[52:56, 52:56] = 1
    reviewed = image.copy()
    reviewed[:32, :32] = reviewed[32:, 32:] = 1
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1)
    dataset = consume(index_patches(image, truth, config, reviewed=reviewed))
    training = QuickTrainWidget(make_napari_viewer())
    qtbot.addWidget(training)
    training.prepared = dataset
    training._update_training_overview()
    assert 'Confirmed foreground: 32 pixels' in training.supervision_label.text()
    assert 'Confirmed background: 2,016 pixels' in training.supervision_label.text()
    assert 'Unknown pixels excluded: 2,048' in training.supervision_label.text()
    assert '1 training patches · 1 validation patches' in training.dataset_state_label.text()


def test_channel_review_validity_is_independent_and_revocable(make_napari_viewer, qtbot):
    from test_reviewed_regions import setup
    from napari_label_assistant_tools._quick_train import segmentation_loss
    viewer, _, truth, coverage, _, review = setup(make_napari_viewer, qtbot)
    review.mark(1)
    state = coverage.metadata[KEY]
    coverage.metadata = {**coverage.metadata, KEY: {**state, 'invalid_regions': [[0, 0, 8, 8]]}}
    partial = QuickTrainWidget._channel_supervision(truth, coverage)
    usable = partial[slice(0, 16), slice(0, 16)]
    assert not usable[:8, :8].any()
    assert usable[8:, :].all()
    valid = np.stack([np.ones((16, 16), np.uint8), usable])
    torch = pytest.importorskip('torch')
    logits = torch.zeros((1, 2, 16, 16), requires_grad=True)
    loss = segmentation_loss(torch, logits, torch.zeros_like(logits), torch.tensor(valid)[None], multilabel=True)
    loss.backward()
    assert logits.grad[0, 0, :8, :8].abs().sum() > 0
    assert logits.grad[0, 1, :8, :8].abs().sum() == 0


def test_noncurrent_pair_edits_revoke_only_intersecting_review_cells(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._reviewed_regions import associate
    from napari_label_assistant_tools._model_project import layer_id
    viewer = make_napari_viewer()
    rows, targets, masks = [], [], []
    for index in range(2):
        image = viewer.add_image(np.zeros((32, 32), np.uint8), name=f'image{index}')
        truth = viewer.add_labels(np.zeros((32, 32), np.uint8), name=f'truth{index}')
        pixels = np.zeros((32, 32), np.uint8)
        pixels[:16] = 1
        mask = viewer.add_labels(pixels, name=f'coverage{index}')
        associate(mask, image, truth)
        cells = [[0, 0, 16, 16], [0, 16, 16, 32]]
        mask.metadata[KEY].update(areas=cells.copy(), review_cells=cells.copy())
        targets.append(truth)
        masks.append(mask)
        rows.append(dict(included=True, image_id=layer_id(image), image_name=image.name,
                         truth_id=layer_id(truth), reviewed_id=layer_id(mask),
                         coverage='partial', split_group=str(index), confirmed=True))
    training = QuickTrainWidget(viewer)
    qtbot.addWidget(training)
    training.dataset_pairs = rows
    training.dataset_source_mode.setCurrentIndex(training.dataset_source_mode.findData('multiple'))
    targets[1].events.labels_update(data=np.zeros((2, 2), np.uint8), offset=(1, 1))
    assert not masks[0].metadata[KEY].get('invalid_regions')
    assert masks[1].metadata[KEY]['invalid_regions'] == [[0, 0, 16, 16]]
    assert masks[1].metadata[KEY]['areas'] == [[0, 16, 16, 32]]


def test_channel_preparation_passes_effective_supervision(make_napari_viewer, qtbot, monkeypatch):
    from test_multilabel_training import fixture_data
    from napari_label_assistant_tools._reviewed_regions import associate
    from napari_label_assistant_tools._model_project import layer_id
    pairs, config = fixture_data(channels=1)
    viewer = make_napari_viewer()
    image = viewer.add_image(pairs[0]['image'])
    truth = viewer.add_labels(pairs[0]['channels'][0]['data'])
    mask = viewer.add_labels(np.ones(truth.data.shape, np.uint8))
    associate(mask, image, truth)
    mask.metadata[KEY].update(areas=[[0, 0, 128, 128]], invalid_regions=[[0, 0, 32, 32]])
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.models_panel.class_names_edit.setText('cell')
    widget.channel_assignments = [dict(included=True, image_id=layer_id(image), image_name=image.name,
        split_group='specimen', channels=dict(cell=dict(target_id=layer_id(truth), reviewed_id=layer_id(mask), coverage='partial')))]
    monkeypatch.setattr(widget, '_config', lambda: config)
    monkeypatch.setattr(widget, '_lock_sources', lambda: None)
    captured = []
    monkeypatch.setattr(widget, '_start_worker', lambda function, done, pairs, *args, **kwargs: captured.extend(pairs))
    widget._start_preparing_channels()
    supervision = captured[0]['channels'][0]['reviewed']
    assert isinstance(supervision, TrainingSupervision)
    assert not supervision[slice(0, 32), slice(0, 32)].any()
    assert supervision[slice(32, 64), slice(32, 64)].all()


def test_mixed_selections_and_review_survive_workspace_roundtrip(tmp_path, make_napari_viewer, qtbot):
    from test_reviewed_regions import setup
    from test_component_library import consume
    from napari_label_assistant_tools._component_library import ensure_library, library_for
    from napari_label_assistant_tools._reviewed_regions import associated_mask
    from napari_label_assistant_tools._mask_tiles import full_data
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer, image, truth, _, _, review = setup(make_napari_viewer, qtbot)
    truth.data[24:27, 24:27] = 2
    review.mark(1)
    library = ensure_library(truth)
    consume(library.update(truth.data))
    selected = next(record['id'] for record in library.records() if record['label_value'] == 2)
    library.select(selected, True)
    path = tmp_path / 'mixed.label-assistant.json'
    save_workspace(viewer, path, complete_package=True)
    restored = make_napari_viewer()
    load_workspace(restored, path)
    loaded_truth = restored.layers[truth.name]
    mask = associated_mask(restored, restored.layers[image.name], loaded_truth)
    loaded_library = library_for(loaded_truth)
    assert loaded_library.selected_ids() == [selected]
    state = mask.metadata[KEY]
    assert state['areas'] == [[0, 0, 16, 16]]
    supervision = TrainingSupervision(full_data(loaded_truth), full_data(mask), library=loaded_library,
        invalid_regions=state.get('invalid_regions', []), reviewed_areas=state['areas'])
    valid = supervision[slice(0, 32), slice(0, 32)]
    assert valid[0, 0] and valid[4, 4] and valid[25, 25]
    assert not valid[20, 20]


def test_huge_review_write_and_index_only_touch_bounded_regions(tmp_path):
    from napari.utils.events import EmitterGroup
    from napari_label_assistant_tools._reviewed_regions import write_area
    from napari_label_assistant_tools._dynamic_edit import TileBounds
    from napari_label_assistant_tools._quick_train import QuickTrainConfig
    from napari_label_assistant_tools._coverage_train import index_patches
    from test_component_library import consume
    import zarr

    class BoundedSource:
        shape = (100000, 100000)
        dtype = np.dtype('uint8')

        def __array__(self, *args, **kwargs):
            raise AssertionError('Full image materialized')

        def __getitem__(self, region):
            y, x = region
            assert y.stop - y.start <= 32 and x.stop - x.start <= 32
            block = np.zeros((y.stop - y.start, x.stop - x.start), np.uint8)
            block[2:6, 2:6] = 1
            return block

    coverage = zarr.open_array(str(tmp_path / 'coverage.zarr'), mode='w',
                               shape=BoundedSource.shape, chunks=(512, 512), dtype='uint8')
    class Mask:
        def refresh(self):
            pass
    mask = Mask()
    mask.data = coverage
    mask.events = EmitterGroup(source=mask)
    regions = [[0, 0, 32, 32], [99968, 99968, 100000, 100000]]
    for rect in regions:
        write_area(mask, TileBounds(*rect), 1)
    source = BoundedSource()
    supervision = TrainingSupervision(source, coverage, reviewed_areas=regions)
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1)
    dataset = consume(index_patches(source, source, config, reviewed=supervision))
    assert len(dataset.locations) == 2
    assert dataset.reviewed_pixels == 2 * 32 * 32
    assert dataset.total_pixels == 100000**2
    _, _, usable = dataset.read(0)
    assert usable.all()
