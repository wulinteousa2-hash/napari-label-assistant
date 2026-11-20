"""Start a project with an image and no pre-existing segmentation mask."""
import json
from pathlib import Path

import numpy as np
import pytest

from napari_label_assistant_tools._label_assistant import label_assistant_widget
from napari_label_assistant_tools._mask_tiles import create_empty_mask, full_data, full_source, manager_for
from napari_label_assistant_tools._coordinates import matching_mask_transforms
from napari_label_assistant_tools.workspace import load_workspace, save_workspace
from napari_label_assistant_tools._tile_progress import KEY as TILE_KEY


@pytest.fixture(autouse=True)
def no_modal_warnings(monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, message: pytest.fail(f'{title}: {message}'))


def open_simple(viewer, qtbot):
    shell = label_assistant_widget(viewer)
    qtbot.addWidget(shell)
    shell.resize(480, 850)
    shell.show()
    shell.set_mode('simple')
    # Synthetic images have no original file to link; include their data in saves.
    shell.workspace.storage_mode_combo.blockSignals(True)
    shell.workspace.storage_mode_combo.setCurrentIndex(1)
    shell.workspace.storage_mode_combo.blockSignals(False)
    return shell


def test_image_and_grid_can_start_annotation_without_an_existing_mask(make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    pixels = np.arange(48 * 64, dtype=np.float32).reshape(48, 64)
    image = viewer.add_image(pixels.copy(), name='New image', scale=(2, 3), translate=(4, 5), rotate=17,
                             shear=(.1,), affine=np.array([[1, .1, 7], [.2, 1, 9], [0, 0, 1]]))
    viewer.add_points([[4, 5]], name='New image | grid_labels')
    shell = open_simple(viewer, qtbot)
    panel = shell.simple
    assert panel.image is image and panel.truth is None
    assert panel.image_combo.count() == 2
    assert panel.create_mask_button.isVisible() and panel.create_mask_button.isEnabled()
    assert not panel.paint_button.isEnabled()
    assert shell._activity_header.confirmed_label.toolTip() == 'No annotation mask yet'
    shell.activity.report('training', 'warning', 'Model Training operation canceled. Existing layers and model are unchanged.')
    panel.refresh_status()
    assert 'training' not in shell.activity.records

    manifest = tmp_path / 'new.label-assistant.json'
    shell.workspace._save_path(manifest)
    assert all(row['layer_type'] != 'Labels' for row in json.loads(manifest.read_text())['layers'])
    panel.create_mask_button.click()
    truth = panel.truth
    assert truth is not None, panel.message.text()
    assert panel.image is image
    assert full_data(truth).shape == pixels.shape
    assert matching_mask_transforms(image, full_source(truth))
    assert not np.asarray(full_data(truth)[:]).any()
    assert panel.create_mask_button.isHidden()
    assert shell.training.image_combo.currentData() is image
    assert shell.training.truth_combo.currentData() is truth
    assert panel.review.image_combo.currentData() is image
    np.testing.assert_array_equal(image.data, pixels)

    truth.brush_size = 1
    truth.paint((10, 10), 7)
    manager_for(viewer).flush()
    shell.workspace._save_path(manifest)
    restored_viewer = make_napari_viewer()
    result = load_workspace(restored_viewer, manifest)
    assert not result['skipped_layers']
    reopened = open_simple(restored_viewer, qtbot).simple
    assert reopened.image.name == image.name
    assert full_data(reopened.truth)[10, 10] == 7
    assert matching_mask_transforms(reopened.image, full_source(reopened.truth))


def test_manual_layer_replacement_uses_new_image_and_preserves_previous_project(make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), dtype=np.uint8), name='Old image')
    viewer.add_labels(np.eye(32, dtype=np.uint8), name='Curated')
    old_manifest = tmp_path / 'old.label-assistant.json'
    save_workspace(viewer, old_manifest, complete_package=True)
    shell = open_simple(viewer, qtbot)
    shell.workspace._associate_workspace(old_manifest)
    old_bytes = old_manifest.read_bytes()
    old_manager = manager_for(viewer)
    viewer.layers.clear()
    image = viewer.add_image(np.zeros((40, 48), dtype=np.uint8), name='Next image')
    viewer.add_points([[0, 0]], name='Next image | grid_labels')
    qtbot.waitUntil(lambda: shell.simple.image is image and shell.simple.truth is None)
    assert shell.workspace.workspace_path is None
    manifest = tmp_path / 'next.label-assistant.json'
    shell.workspace._save_path(manifest)
    shell.simple.create_mask_button.click()
    truth = shell.simple.truth
    assert truth is not None, shell.simple.message.text()
    assert manager_for(viewer) is not old_manager
    assert full_data(truth).shape == (40, 48)
    assert shell.simple.image is image
    assert not shell.simple.pending
    assert '0 /' in shell.simple.review.progress_label.text()
    shell.workspace._save_path(manifest)
    assert old_manifest.read_bytes() == old_bytes
    assert all(record['name'] != 'Old image' for record in json.loads(manifest.read_text())['layers'])


def test_multiple_new_images_remain_selectable_before_any_masks_exist(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    first = viewer.add_image(np.zeros((24, 32), dtype=np.uint8), name='First')
    second = viewer.add_image(np.zeros((40, 48), dtype=np.uint8), name='Second')
    panel = open_simple(viewer, qtbot).simple
    assert panel.image_combo.count() == 3
    assert panel.image is None
    panel.image_combo.setCurrentIndex(panel.image_combo.findData(second))
    panel.refresh_sources()
    assert panel.image is second
    panel.create_mask_button.click()
    second_truth = panel.truth
    assert second_truth is not None
    assert full_data(second_truth).shape == (40, 48)
    previous_session = dict(full_source(second_truth).metadata[TILE_KEY])
    assert previous_session['image_id'] == second.metadata['label_assistant_review']['id']
    # A mask for another image must not make this new image disappear.
    panel.image_combo.setCurrentIndex(panel.image_combo.findData(first))
    assert panel.image is first and panel.truth is None
    assert full_source(second_truth).metadata[TILE_KEY] == previous_session
    assert panel.create_mask_button.isEnabled()
    panel.create_mask_button.click()
    assert full_data(panel.truth).shape == (24, 32)
    assert panel.truth is not second_truth


def test_huge_image_creates_sparse_mask_without_reading_source_pixels(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    import zarr
    root = zarr.open_group(str(tmp_path / 'image.zarr'), mode='w')
    from napari_label_assistant_tools.workspace.service import _create_zarr_array
    base = _create_zarr_array(root, 's0', shape=(100000, 100000), chunks=(512, 512), dtype='uint8')
    overview = _create_zarr_array(root, 's6', shape=(1563, 1563), chunks=(256, 256), dtype='uint8')
    viewer = make_napari_viewer()
    image = viewer.add_image([base, overview], multiscale=True, name='Huge image')
    source_type = type(base)
    original = source_type.__getitem__
    def guarded(array, key):
        assert array is not base, 'Creating a mask must not read the source image'
        result = original(array, key)
        assert result.size <= 4096 ** 2, 'Mask creation must keep reads bounded'
        return result
    monkeypatch.setattr(source_type, '__getitem__', guarded)
    layer = create_empty_mask(viewer, image, manifest=tmp_path / 'large.label-assistant.json')
    assert full_data(layer).shape == (100000, 100000)
    assert layer.data.size <= 1024 ** 2
    storage = full_source(layer).metadata['label_assistant_workspace_storage']
    store_path = Path(storage['path'])
    assert sum(path.stat().st_size for path in store_path.rglob('*') if path.is_file()) < 1024 ** 2


@pytest.mark.parametrize('rgb', [False, True])
def test_unsupported_images_cannot_create_annotation_masks(make_napari_viewer, rgb):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((8, 12, 3) if rgb else (3, 8, 12), dtype=np.uint8), rgb=rgb)
    with pytest.raises(ValueError, match='2D'):
        create_empty_mask(viewer, image)
    assert len(viewer.layers) == 1


@pytest.mark.parametrize('confirm', [False, True])
def test_simple_workspace_new_confirms_and_preserves_saved_project(make_napari_viewer, qtbot, tmp_path, monkeypatch, confirm):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), dtype=np.uint8), name='Old image')
    viewer.add_labels(np.eye(32, dtype=np.uint8), name='Curated')
    manifest = tmp_path / 'old.label-assistant.json'
    save_workspace(viewer, manifest, complete_package=True)
    before = manifest.read_bytes()
    shell = open_simple(viewer, qtbot)
    shell.workspace._associate_workspace(manifest)
    panel = shell.simple
    assert [button.text() for button in (panel.new_button, panel.open_button, panel.save_button)] == ['New', 'Open', 'Save']
    previous_layers = tuple(viewer.layers)
    prompts = []
    monkeypatch.setattr(shell.workspace, '_confirm_clear', lambda message: prompts.append(message) or confirm)
    panel.new_button.click()
    assert len(prompts) == 1
    if confirm:
        qtbot.waitUntil(lambda: panel.image is None and panel.truth is None)
        assert len(viewer.layers) == 0
        assert shell.workspace.workspace_path is None
        assert 'Unsaved workspace' in shell.workspace.current_label.text()
        assert not panel.create_mask_button.isEnabled()
        image = viewer.add_image(np.zeros((40, 48), dtype=np.uint8), name='Next image')
        qtbot.waitUntil(lambda: panel.image is image)
        assert panel.create_mask_button.isEnabled()
    else:
        assert tuple(viewer.layers) == previous_layers
        assert shell.workspace.workspace_path == manifest
    assert manifest.read_bytes() == before
