"""Review sizes must stay unique and preserve existing review coordinates."""
import pytest

from test_simple_curation import setup_simple as setup_simple, pick, finish
from napari_label_assistant_tools._simple_curation import REVIEW_TILE_SIZES
from napari_label_assistant_tools._mask_tiles import manager_for
from napari_label_assistant_tools._reviewed_regions import associated_mask, effective_areas, KEY
from napari_label_assistant_tools._label_assistant import label_assistant_widget
from napari_label_assistant_tools.workspace import load_workspace, save_workspace


def choices(panel):
    return [tuple(panel.grid_combo.itemData(i)) for i in range(panel.grid_combo.count())]


def test_repeated_refresh_does_not_duplicate_custom_review_sizes(setup_simple):
    _, shell, _ = setup_simple
    panel = shell.simple
    shell.components._grid_y_spin.setValue(100)
    shell.components._grid_x_spin.setValue(100)
    for _ in range(20):
        panel.refresh_sources()
    expected = [(size, size) for size in REVIEW_TILE_SIZES] + [(100, 100)]
    assert choices(panel) == expected
    assert panel.grid_combo.currentData() == [100, 100]
    assert panel.grid_combo.currentText() == '100 × 100 px (custom)'
    panel.grid_combo.setCurrentIndex(panel.grid_combo.findData([512, 512]))
    for _ in range(20):
        panel.refresh_sources()
    assert choices(panel) == [(size, size) for size in REVIEW_TILE_SIZES]
    assert panel.grid_combo.currentData() == [512, 512]


@pytest.mark.parametrize('size', REVIEW_TILE_SIZES)
def test_review_presets_update_grid_without_enlarging_edit_or_training_buffers(setup_simple, size):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    before = (manager_for(viewer).edit_size, shell.training.crop_combo.currentData())
    index = panel.grid_combo.findData([size, size])
    assert index >= 0
    panel.grid_combo.setCurrentIndex(index)
    panel.refresh_sources()
    assert panel.grid_combo.currentData() == [size, size]
    assert (shell.components._grid_y_spin.value(), shell.components._grid_x_spin.value()) == (size, size)
    assert (manager_for(viewer).edit_size, shell.training.crop_combo.currentData()) == before
    assert choices(panel) == [(value, value) for value in REVIEW_TILE_SIZES]


def test_review_size_changes_preserve_approvals_and_restore_custom_grid(setup_simple, qtbot, make_napari_viewer):
    viewer, shell, manifest = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    coverage = associated_mask(viewer, panel.image, panel.truth)
    before = effective_areas(coverage.metadata[KEY])
    panel.grid_combo.setCurrentIndex(panel.grid_combo.findData([8192, 8192]))
    panel.refresh_sources()
    assert effective_areas(coverage.metadata[KEY]) == before
    shell.components._grid_y_spin.setValue(100)
    shell.components._grid_x_spin.setValue(200)
    panel.refresh_sources()
    assert panel.grid_combo.currentData() == [100, 200]
    save_workspace(viewer, manifest)
    restored_viewer = make_napari_viewer()
    assert not load_workspace(restored_viewer, manifest)['skipped_layers']
    restored = label_assistant_widget(restored_viewer)
    qtbot.addWidget(restored)
    restored.set_mode('simple')
    for _ in range(10):
        restored.simple.refresh_sources()
    assert restored.simple.grid_combo.currentData() == [100, 200]
    assert choices(restored.simple).count((100, 200)) == 1
    assert len(choices(restored.simple)) == len(REVIEW_TILE_SIZES) + 1
    restored_coverage = associated_mask(restored_viewer, restored.simple.image, restored.simple.truth)
    assert effective_areas(restored_coverage.metadata[KEY]) == before
