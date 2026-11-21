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
    assert panel.grid_combo.currentData() == [size, size], (panel._grid_locked, panel._grid_busy(), panel._stroke_active(), panel.message.text())
    assert (shell.components._grid_y_spin.value(), shell.components._grid_x_spin.value()) == (size, size)
    assert (manager_for(viewer).edit_size, shell.training.crop_combo.currentData()) == before
    assert choices(panel) == [(value, value) for value in REVIEW_TILE_SIZES]


def test_review_size_changes_preserve_approvals_and_restore_custom_grid(setup_simple, qtbot, make_napari_viewer, monkeypatch):
    viewer, shell, manifest = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    coverage = associated_mask(viewer, panel.image, panel.truth)
    before = effective_areas(coverage.metadata[KEY])
    monkeypatch.setattr(panel, '_choose_review_grid', lambda current: (8192, 8192))
    panel.change_grid_button.click()
    panel.refresh_sources()
    assert panel.grid_combo.currentData() == [8192, 8192]
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
    assert restored.simple._grid_locked
    assert not restored.simple.grid_combo.isEnabled()
    restored_coverage = associated_mask(restored_viewer, restored.simple.image, restored.simple.truth)
    assert effective_areas(restored_coverage.metadata[KEY]) == before


def test_selected_size_locks_and_accidental_changes_are_rejected(setup_simple):
    _, shell, _ = setup_simple
    panel = shell.simple
    panel.grid_combo.setCurrentIndex(panel.grid_combo.findData([2048, 2048]))
    assert panel._grid_locked
    assert not panel.grid_combo.isEnabled()
    assert panel.change_grid_button.isEnabled()
    assert panel.change_grid_button.text() == 'Change…'
    panel.grid_combo.setCurrentIndex(panel.grid_combo.findData([4096, 4096]))
    assert panel.grid_combo.currentData() == [2048, 2048]
    assert (shell.components._grid_y_spin.value(), shell.components._grid_x_spin.value()) == (2048, 2048)
    for control in (panel.grid_label, panel.grid_combo):
        assert 'Locked' in control.toolTip()
        assert 'does not set the orange outlined editing area' in control.toolTip()
        assert 'Change review grid' in control.toolTip()
        assert 'orange outline' in control.toolTip()


def test_painting_locks_default_review_size_and_saves_it(setup_simple, make_napari_viewer, qtbot):
    from napari_label_assistant_tools._mask_tiles import full_source
    from napari_label_assistant_tools._tile_progress import KEY as TILE_KEY
    viewer, shell, manifest = setup_simple
    panel = shell.simple
    assert not panel._grid_locked
    panel.truth.brush_size = 1
    panel.truth.paint((15, 15), 77)
    assert panel._grid_locked
    assert full_source(panel.truth).metadata[TILE_KEY]['review_grid_locked'] is True
    save_workspace(viewer, manifest)
    restored_viewer = make_napari_viewer()
    load_workspace(restored_viewer, manifest)
    restored = label_assistant_widget(restored_viewer)
    qtbot.addWidget(restored)
    restored.set_mode('simple')
    assert restored.simple._grid_locked
    assert not restored.simple.grid_combo.isEnabled()


def test_cancelled_grid_change_preserves_pending_tiles(setup_simple, monkeypatch):
    _, shell, _ = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    pending = list(panel.pending)
    current = panel.grid_combo.currentData()
    monkeypatch.setattr(panel, '_choose_review_grid', lambda value: None)
    panel.change_grid_button.click()
    assert panel.pending == pending
    assert panel.grid_combo.currentData() == current
    assert panel._grid_locked


def test_explicit_grid_change_flushes_pixels_and_keeps_approved_regions(setup_simple, qtbot, monkeypatch):
    from napari_label_assistant_tools._mask_tiles import full_data
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    coverage = associated_mask(viewer, panel.image, panel.truth)
    approved = effective_areas(coverage.metadata[KEY])
    pick(panel, 70, 70)
    panel.truth.brush_size = 1
    panel.truth.paint((80, 80), 99)
    monkeypatch.setattr(panel, '_choose_review_grid', lambda current: (4096, 4096))
    panel.change_grid_button.click()
    assert full_data(panel.truth)[80, 80] == 99
    assert effective_areas(coverage.metadata[KEY]) == approved
    assert not panel.pending
    assert panel.review.selected_cell is None
    assert panel.grid_combo.currentData() == [4096, 4096]
    assert panel._grid_locked


def test_grid_change_is_rejected_during_an_active_stroke(setup_simple, monkeypatch):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    manager = manager_for(viewer)
    prompts = []
    monkeypatch.setattr(panel, '_choose_review_grid', lambda current: prompts.append(current))
    manager.stroke_active = True
    try:
        with pytest.raises(ValueError, match='Finish the current'):
            panel._grid_action()
    finally:
        manager.stroke_active = False
    assert not prompts


def test_editable_outline_tracks_loaded_crop_instead_of_review_tile(setup_simple, qtbot):
    from napari_label_assistant_tools._coordinates import matching_mask_transforms
    from napari_label_assistant_tools._mask_tiles import full_source
    viewer, shell, _ = setup_simple
    panel = shell.simple
    panel.grid_combo.setCurrentIndex(panel.grid_combo.findData([4096, 4096]))
    manager_for(viewer).go(0, 0, size=64)
    active = viewer.layers.selection.active
    panel.refresh_status()
    assert 'Review tile: 4096 × 4096 px' in panel.area_label.text()
    assert 'Editable area: 64 × 64 px' in panel.area_label.text()
    outline = panel.review.simple_edit_overlay
    assert outline.visible and not outline.editable
    assert tuple(outline.data[0].max(axis=0)) == (64, 64)
    assert matching_mask_transforms(outline, full_source(panel.truth))
    assert viewer.layers.selection.active is active
    before = outline.edge_width[0]
    viewer.camera.zoom /= 10
    qtbot.wait(50)
    panel._update_review_size_feedback()
    assert outline.edge_width[0] > before * 5
    shell.set_mode('advanced')
    assert not outline.visible


def test_change_dialog_applies_a_deliberately_selected_size(setup_simple, qtbot):
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QLabel
    _, shell, _ = setup_simple
    panel = shell.simple
    if not panel._grid_locked:
        panel.change_grid_button.click()  # Lock the default size.
    def choose():
        dialog = QApplication.activeModalWidget()
        try:
            combo = dialog.findChild(QComboBox)
            assert 'orange outlined editing area' in combo.toolTip()
            assert any('approved regions are kept' in label.text() for label in dialog.findChildren(QLabel))
            combo.setCurrentIndex(combo.findData([8192, 8192]))
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        finally:
            if dialog.isVisible():
                dialog.reject()
    QTimer.singleShot(20, choose)
    panel.change_grid_button.click()
    assert panel.grid_combo.currentData() == [8192, 8192]
    assert panel._grid_locked and not panel.grid_combo.isEnabled()
