"""Removing managed masks must not mutate napari's in-flight deletion list."""
import numpy as np
import pytest
import zarr

from napari_label_assistant_tools._mask_tiles import MaskTileManager
from napari_label_assistant_tools.workspace.service import _persist_labels_to_zarr


@pytest.fixture
def managed_mask(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    pixels = np.zeros((64, 64), dtype='uint32')
    pixels[3:8, 4:9] = 17
    path = tmp_path / 'mask.ome.zarr'
    _persist_labels_to_zarr(pixels, path, xy_chunk=32, scale=(1, 1))
    array = zarr.open_array(str(path / 's0'), mode='r+')
    manager = MaskTileManager(viewer, dict(tile_size=32, camera_follow=False))
    manager.add(dict(name='SAM3 tiled exemplar labels'), array, dict(path=path, array_path='s0'))
    entry = manager.entries[0]
    assert entry.overview is not None
    yield viewer, manager, entry, array
    manager.dispose()


@pytest.mark.parametrize('overview_first', [False, True])
def test_remove_selected_crop_and_overview_in_either_order(managed_mask, monkeypatch, qtbot, overview_first):
    viewer, manager, entry, array = managed_mask
    crop, overview = entry.layer, entry.overview
    entry.layer.data[10, 11] = 81
    viewer.layers.selection.clear()
    viewer.layers.selection.update([crop, overview])
    # Selection is a set: both valid snapshot orders must work reliably.
    selection = viewer.layers.selection
    original_iter = type(selection).__iter__
    first = overview if overview_first else crop
    def ordered(current):
        items = list(original_iter(current))
        return iter(sorted(items, key=lambda item: item is not first)) if current is selection else iter(items)
    monkeypatch.setattr(type(selection), '__iter__', ordered)
    viewer.layers.remove_selected()
    qtbot.waitUntil(lambda: len(viewer.layers) == 0)
    assert not viewer.layers.selection
    assert not manager.entries
    assert array[10, 11] == 81


def test_remove_crop_after_reordering_overview_before_it(managed_mask, qtbot):
    viewer, manager, entry, _ = managed_mask
    viewer.layers.move(viewer.layers.index(entry.overview), 0)
    viewer.layers.remove(entry.layer)
    qtbot.waitUntil(lambda: len(viewer.layers) == 0)
    assert not viewer.layers.selection
    assert not manager.entries


def test_clear_reordered_layers_keeps_saved_pixels(managed_mask, qtbot):
    viewer, manager, entry, array = managed_mask
    entry.layer.data[10, 11] = 81
    viewer.layers.move(viewer.layers.index(entry.overview), 0)
    viewer.layers.clear()
    qtbot.waitUntil(lambda: len(viewer.layers) == 0)
    assert not viewer.layers.selection
    assert array[10, 11] == 81


def test_manual_overview_removal_keeps_editable_source(managed_mask):
    viewer, manager, entry, _ = managed_mask
    viewer.layers.remove(entry.overview)
    assert entry.overview is None
    assert entry.layer in viewer.layers and entry in manager.entries
    entry.refresh_overview()
    assert entry.overview in viewer.layers


def test_failed_flush_prevents_removal_without_losing_layers(managed_mask):
    viewer, manager, entry, array = managed_mask
    crop, overview = entry.layer, entry.overview
    crop.data[10, 11] = 81
    array[10, 11] = 99
    with pytest.raises(ValueError, match='another writer'):
        viewer.layers.remove(crop)
    assert crop in viewer.layers and overview in viewer.layers
    assert entry in manager.entries and not manager._flushing_removal
    assert crop.data[10, 11] == 81 and array[10, 11] == 99
    crop.data[10, 11] = 99
    manager.flush()


def test_deleting_one_managed_mask_does_not_remove_unrelated_layers(managed_mask, qtbot):
    viewer, _, entry, _ = managed_mask
    other = viewer.add_image(np.ones((64, 64)), name='reference image')
    viewer.layers.move(viewer.layers.index(entry.overview), 0)
    viewer.layers.remove(entry.layer)
    qtbot.waitUntil(lambda: len(viewer.layers) == 1)
    assert viewer.layers[0] is other
    assert viewer.layers.selection.issubset(set(viewer.layers))
