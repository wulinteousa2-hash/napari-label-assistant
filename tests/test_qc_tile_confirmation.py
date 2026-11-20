"""Editing-to-QC picking and explicit whole-tile training confirmation."""
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from napari_label_assistant_tools._label_assistant import label_assistant_widget
from napari_label_assistant_tools._mask_tiles import MaskTileManager, full_data, manager_for
from napari_label_assistant_tools._reviewed_regions import KEY, associated_mask, effective_areas


@pytest.fixture
def qc_editor(tmp_path, make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    transform = dict(scale=(2, 3), translate=(11, 17), rotate=23, shear=(.2,))
    image = viewer.add_image(np.zeros((1030, 1530), dtype='uint8'), name='EM image', **transform)
    path = tmp_path/'mask.zarr'
    stored = zarr.open_array(str(path), mode='w', shape=(1030, 1530),
                             chunks=(256, 256), dtype='uint32')
    stored[520:524, 520:524] = 7
    stored[550:554, 560:564] = 7
    stored[5:10, 5:10] = 9
    from napari_label_assistant_tools.workspace.service import _common_record
    record = _common_record(image, 'Labels')
    record['name'] = 'Curated'
    manager = MaskTileManager(viewer, dict(tile_size=512, row=1, col=1))
    truth = manager.add(record, stored, {'path': path, 'array_path': ''})
    page = label_assistant_widget(viewer)
    qtbot.addWidget(page)
    page.show()
    page._tool_tabs.setCurrentIndex(1)
    widget = page._tool_tabs.widget(1)
    widget._target_combo.setCurrentText('Curated')
    widget._analysis_scope_combo.setCurrentIndex(widget._analysis_scope_combo.findData('area'))
    widget._workflow_tabs.setCurrentIndex(1)
    return viewer, image, truth, stored, page, widget


def click(viewer, layer, point, button=1):
    callback = next(cb for cb in viewer.mouse_drag_callbacks
                    if getattr(cb, '__name__', '') == '_handle_mouse_click')
    callback(viewer, SimpleNamespace(type='mouse_press', button=button,
                                    position=layer.data_to_world(point)))


def test_canvas_pick_analyzes_automatically_and_appends_without_painting(qc_editor):
    viewer, _, truth, _, _, widget = qc_editor
    truth.mode = 'paint'
    original = truth.data.copy()
    widget._append_click_select_check.setChecked(True)
    assert widget._click_select_check.isChecked()
    assert widget._component_table.rowCount() == 2
    assert str(truth.mode) == 'pan_zoom'
    click(viewer, truth, (8, 8))
    click(viewer, truth, (40, 50))
    assert widget._component_table.selected_component_ids() == [1, 2]
    np.testing.assert_array_equal(truth.data, original)
    click(viewer, truth, (8, 8), button=2)
    assert widget._component_table.selected_component_ids() == [1, 2]
    widget._workflow_tabs.setCurrentIndex(0)
    assert str(truth.mode) == 'paint'
    assert not any(getattr(cb, '__name__', '') == '_handle_mouse_click' for cb in viewer.mouse_drag_callbacks)


def test_source_edits_and_crop_navigation_do_not_use_stale_component_ids(qc_editor):
    viewer, _, truth, _, _, widget = qc_editor
    widget._click_select_check.setChecked(True)
    click(viewer, truth, (8, 8))
    truth.data[80:84, 80:84] = 7
    truth.events.data()
    click(viewer, truth, (81, 81))
    assert widget._component_table.rowCount() == 3
    assert widget._component_table.selected_component_ids() == [3]
    truth._label_assistant_mask_entry.manager.go(0, 0)
    click(viewer, truth, (6, 6))
    assert widget._component_table.rowCount() == 1
    assert widget._component_table.selected_component_ids() == [1]


def test_cell_review_mode_does_not_block_qc_component_picking(qc_editor):
    viewer, _, truth, _, _, widget = qc_editor
    widget._workflow_tabs.setCurrentIndex(0)
    widget._reviewed_area_widget.mouse_review_check.setChecked(True)
    assert widget._reviewed_area_widget.mouse_review_check.isChecked()
    widget._workflow_tabs.setCurrentIndex(1)
    widget._click_select_check.setChecked(True)
    assert not widget._reviewed_area_widget.mouse_review_check.isChecked()
    click(viewer, truth, (8, 8))
    assert widget._component_table.selected_component_ids() == [1]


def test_canvas_cell_review_stays_enabled_after_click_and_confirmation(qc_editor, qtbot):
    viewer, _, truth, _, _, widget = qc_editor
    widget._workflow_tabs.setCurrentIndex(0)
    review = widget._reviewed_area_widget
    review.mouse_review_check.setChecked(True)
    from napari_label_assistant_tools._mask_tiles import full_source
    for point in ((520, 520), (620, 620)):
        review.mouse_click(viewer, SimpleNamespace(type='mouse_press', button=1,
            position=full_source(truth).data_to_world(point)))
        assert review.mouse_review_check.isChecked()
        review.next_button.click()
        assert review.mouse_review_check.isChecked(), review.status.text()


def confirm(qc_editor, qtbot):
    viewer, image, truth, stored, page, widget = qc_editor
    widget._workflow_tabs.setCurrentIndex(0)
    widget._loaded_tile_section.setChecked(True)
    assert widget._confirm_tile_button.isEnabled()
    widget._confirm_tile_button.click()
    coverage = associated_mask(viewer, image, truth)
    assert coverage is not None, widget._status_label.text()
    training = page.training
    qtbot.waitUntil(lambda: training.reviewed_combo.currentData() is coverage, timeout=5000)
    return coverage, training


def test_whole_tile_confirmation_includes_foreground_background_and_excludes_outside(qc_editor, qtbot, monkeypatch):
    from napari_label_assistant_tools._component_library import TrainingSupervision
    from napari_label_assistant_tools._reviewed_regions import QFileDialog
    viewer, image, truth, stored, _, widget = qc_editor
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: pytest.fail('Bounded confirmation must not require a legacy-mode file dialog'))
    truth.data[20, 20] = 17
    truth.events.data()
    original_get = zarr.Array.__getitem__
    original_set = zarr.Array.__setitem__
    writes = []
    def bounded_get(array, key):
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(s, slice) for s in key):
            assert int(np.prod([len(range(*s.indices(n))) for s, n in zip(key, array.shape)])) <= 512**2
        return original_get(array, key)
    def bounded_set(array, key, value):
        assert np.size(value) <= 512**2
        writes.append(np.shape(value))
        return original_set(array, key, value)
    with monkeypatch.context() as guarded:
        guarded.setattr(zarr.Array, '__getitem__', bounded_get)
        guarded.setattr(zarr.Array, '__setitem__', bounded_set)
        coverage, training = confirm(qc_editor, qtbot)
    assert writes
    assert stored[532, 532] == 17
    assert coverage.metadata[KEY]['areas'] == [[512, 512, 1024, 1024]]
    assert full_data(coverage).shape == full_data(truth).shape == (1030, 1530)
    assert np.all(full_data(coverage)[512:1024, 512:1024] == 1)
    assert full_data(coverage)[5, 5] == 0
    assert stored[5, 5] == 9
    supervision = TrainingSupervision(full_data(truth), full_data(coverage), reviewed_areas=coverage.metadata[KEY]['areas'])
    assert supervision[slice(520, 530), slice(520, 530)].all()
    assert not supervision[slice(0, 16), slice(0, 16)].any()
    assert 'Ready to prepare' in training.dataset_state_label.text()
    outline = widget._reviewed_area_widget.review_overlay
    np.testing.assert_allclose(outline.data_to_world((512, 512)), truth._label_assistant_mask_entry.source.data_to_world((512, 512)))
    assert 'confirmed for training' in widget._status_label.text()
    frame = widget._reviewed_area_widget.tile_confirmation_overlay
    assert frame.visible
    np.testing.assert_array_equal(frame.data[0][0], (512, 512))
    np.testing.assert_allclose(frame.data_to_world((512, 512)), truth._label_assistant_mask_entry.source.data_to_world((512, 512)))
    widget._workflow_tabs.setCurrentIndex(1)
    assert not frame.visible


def test_confirmed_tile_survives_save_reload_and_edits_require_reconfirmation(qc_editor, qtbot, tmp_path):
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    from napari_label_assistant_tools._component_library import TrainingSupervision
    viewer, image, truth, _, _, widget = qc_editor
    coverage, training = confirm(qc_editor, qtbot)
    truth.data[8, 8] = 19
    truth.events.data()
    manager_for(viewer).flush()
    assert not effective_areas(coverage.metadata[KEY])
    supervision = TrainingSupervision(full_data(truth), full_data(coverage),
        invalid_regions=coverage.metadata[KEY]['invalid_regions'], reviewed_areas=coverage.metadata[KEY]['areas'])
    assert not supervision[slice(512, 528), slice(512, 528)].any()
    confirm(qc_editor, qtbot)
    assert effective_areas(coverage.metadata[KEY]) == [[512, 512, 1024, 1024]]
    path = tmp_path/'confirmed.label-assistant.json'
    save_workspace(viewer, path, complete_package=True)
    load_workspace(viewer, path)
    restored_truth = viewer.layers['Curated']
    restored_coverage = associated_mask(viewer, viewer.layers['EM image'], restored_truth)
    assert restored_coverage.metadata[KEY]['areas'] == [[512, 512, 1024, 1024]]
    assert full_data(restored_coverage)[520, 520] == 1
    assert full_data(restored_truth)[520, 520] == 19


def test_confirmation_refuses_active_strokes_and_misaligned_edit_crops(qc_editor):
    viewer, image, truth, _, _, widget = qc_editor
    widget._workflow_tabs.setCurrentIndex(0)
    widget._loaded_tile_section.setChecked(True)
    manager_for(viewer).stroke_active = True
    widget._confirm_tile_button.click()
    assert associated_mask(viewer, image, truth) is None
    assert 'stroke' in widget._status_label.text()
    manager_for(viewer).stroke_active = False
    truth.translate = truth.translate + (1, 0)
    widget._confirm_tile_button.click()
    assert associated_mask(viewer, image, truth) is None
    assert 'registration' in widget._status_label.text()


def test_confirmed_tile_can_prepare_partial_training_without_unknown_background(qc_editor, qtbot):
    viewer, _, truth, _, _, widget = qc_editor
    # Put foreground in a second independent spatial group for holdout.
    truth.data[400:404, 400:404] = 7
    truth.events.data()
    coverage, training = confirm(qc_editor, qtbot)
    training.crop_combo.addItem('32 px', 32)
    training.crop_combo.setCurrentIndex(training.crop_combo.findData(32))
    training.resolution_combo.addItem('32 px', 32)
    training.resolution_combo.setCurrentIndex(training.resolution_combo.findData(32))
    assert training.annotation_combo.currentData() == 'partial'
    training.prepare_button.click()
    qtbot.waitUntil(lambda: training.worker is None, timeout=15000)
    assert training.prepared is not None, training.status_label.text()
    dataset = training.prepared
    assert dataset.reviewed_pixels == 512*512
    assert len(dataset.locations) == 16*16
    assert dataset.indices('training') and dataset.indices('validation')
    assert sum(p['foreground_pixels_original'] for p in dataset.locations) == 48
    assert all(p['y0'] >= 512 and p['x0'] >= 512 for p in dataset.locations)
    assert full_data(coverage)[5, 5] == 0
    assert training.show_dataset_split_button.isEnabled()
    training.show_dataset_split_button.click()
    assert training._validation_preview_layer is not None
    manager_for(viewer).go(0, 0)
    assert training.prepared is dataset


def test_huge_tile_confirmation_never_reads_or_writes_a_full_mask(tmp_path, make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._widget import component_operations_widget
    viewer = make_napari_viewer()
    root = zarr.open_group(str(tmp_path/'image.zarr'), mode='w')
    levels = [root.create_array(f's{i}', shape=(80000//2**i, 80000//2**i),
                                chunks=(512, 512), dtype='uint8') for i in range(7)]
    image = viewer.add_image(levels, multiscale=True, name='Large image', contrast_limits=(0, 255))
    path = tmp_path/'large-mask.zarr'
    data = zarr.open_array(str(path), mode='w', shape=(80000, 80000), chunks=(512, 512), dtype='uint32')
    manager = MaskTileManager(viewer, dict(tile_size=512, row=100, col=110))
    y, x = 100*512, 110*512
    data[y+5, x+5] = 17
    truth = manager.add({'name': 'Curated'}, data, {'path': path, 'array_path': ''})
    widget = component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._workflow_tabs.setCurrentIndex(0)
    widget._loaded_tile_section.setChecked(True)
    original_get, original_set = zarr.Array.__getitem__, zarr.Array.__setitem__
    def bounded_get(array, key):
        if array.shape == (80000, 80000) and isinstance(key, tuple) and all(isinstance(s, slice) for s in key):
            assert int(np.prod([len(range(*s.indices(n))) for s, n in zip(key, array.shape)])) <= 512**2
        return original_get(array, key)
    def bounded_set(array, key, value):
        assert np.size(value) <= 512**2
        return original_set(array, key, value)
    monkeypatch.setattr(zarr.Array, '__getitem__', bounded_get)
    monkeypatch.setattr(zarr.Array, '__setitem__', bounded_set)
    widget._confirm_tile_button.click()
    coverage = associated_mask(viewer, image, truth)
    assert coverage is not None, widget._status_label.text()
    assert full_data(coverage).shape == (80000, 80000)
    assert full_data(coverage)[y+5, x+5] == full_data(coverage)[y+20, x+20] == 1
    assert full_data(coverage)[0, 0] == 0
    assert data[y+5, x+5] == 17
    assert coverage.metadata[KEY]['areas'] == [[y, x, y+512, x+512]]


def tile_click(viewer, layer, point, button=1):
    callback = next(cb for cb in viewer.mouse_drag_callbacks
                    if getattr(cb, '__name__', '') == '_select_training_tile')
    callback(viewer, SimpleNamespace(type='mouse_press', button=button,
                                    position=layer.data_to_world(point)))


@pytest.mark.parametrize('unused_pending_library', [False, True])
def test_dataset_select_additive_tiles_confirm_and_prepare_with_default_patch_sizes(qc_editor, qtbot, tmp_path, unused_pending_library):
    viewer, image, truth, _, page, widget = qc_editor
    training = page.training
    widget._workflow_tabs.setCurrentIndex(0)
    truth.mode = 'paint'
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    page.show_training()
    assert training.training_tile_actions.isVisible()
    training.select_training_tiles_button.click()
    assert page._tool_tabs.currentWidget() is widget
    assert widget._workflow_tabs.currentIndex() == 0
    assert widget._tile_select_check.isChecked()
    widget._tile_append_check.setChecked(True)
    assert str(truth.mode) == 'pan_zoom'
    # Background clicks select the whole checked tile, not a component.
    tile_click(viewer, truth, (100, 100))
    tile_click(viewer, truth, (100, 100))
    assert '1 checked training tiles' in widget._tile_selection_summary.text()
    assert associated_mask(viewer, image, truth) is None
    assert widget._component_table.rowCount() == 0
    manager_for(viewer).go(0, 0)
    tile_click(viewer, truth, (100, 100))
    assert '2 checked training tiles' in widget._tile_selection_summary.text()
    # Navigating to an unselected third tile must not include it on confirmation.
    manager_for(viewer).go(0, 2)
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    page.show_training()
    training.confirm_training_tiles_button.click()
    coverage = associated_mask(viewer, image, truth)
    assert coverage is not None, training.dataset_feedback.text()
    assert len(coverage.metadata[KEY]['areas']) == 2
    assert '2 confirmed tiles available' in training.annotation_sources_label.text(), (coverage.metadata[KEY], training.reviewed_combo.currentData())
    assert training.reviewed_combo.currentData() is coverage
    assert training.crop_combo.currentData() == 512
    if unused_pending_library:
        from napari_label_assistant_tools._component_library import ensure_library
        library = ensure_library(truth, tmp_path/'unused.sqlite')
        library.mark_dirty()
        assert library.counts()[1] == 0 and library.state()['pending']
    training.prepare_button.click()
    qtbot.waitUntil(lambda: training.worker is None, timeout=15000)
    assert training.prepared is not None, training.dataset_feedback.text()
    dataset = training.prepared
    assert dataset.reviewed_pixels == 2*512**2
    assert len(dataset.locations) == 2
    assert len(dataset.indices('training')) == len(dataset.indices('validation')) == 1
    assert sum(p['foreground_pixels_original'] for p in dataset.locations) == 57
    assert full_data(coverage)[1000, 100] == 0
    assert full_data(coverage)[100, 1100] == 0


def test_tile_picker_replaces_clears_and_rejects_overview_clicks(qc_editor):
    viewer, image, truth, _, _, widget = qc_editor
    widget._workflow_tabs.setCurrentIndex(0)
    widget._loaded_tile_section.setChecked(True)
    widget._tile_select_check.setChecked(True)
    tile_click(viewer, truth, (100, 100))
    manager_for(viewer).go(0, 0)
    tile_click(viewer, truth, (100, 100))
    assert '1 checked training tiles' in widget._tile_selection_summary.text()
    tile_click(viewer, truth, (600, 600))
    assert 'inside the loaded full-resolution tile' in widget._status_label.text()
    assert associated_mask(viewer, image, truth) is None
    widget._clear_training_tiles_button.click()
    assert 'No training tiles' in widget._tile_selection_summary.text()
    widget._click_select_check.setChecked(True)
    assert not widget._tile_select_check.isChecked()


def test_prepare_without_confirmed_tiles_explains_the_next_action_in_dataset(qc_editor):
    viewer, _, _, _, page, _ = qc_editor
    training = page.training
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    page.show_training()
    training.prepare_button.click()
    assert training.worker is None
    assert 'No checked tiles are confirmed' in training.dataset_feedback.text()
    assert training.edit_confirm_button.isVisible()
    assert not training.select_training_tiles_button.isVisible()
    assert not training.confirm_training_tiles_button.isVisible()


def test_edit_confirm_happy_path_keeps_training_and_cleanup_separate(qc_editor, qtbot):
    viewer, image, truth, _, page, widget = qc_editor
    training = page.training
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    page.show_training()
    training.edit_confirm_button.click()
    assert page._tool_tabs.currentWidget() is widget
    assert widget._workflow_tabs.currentIndex() == 0
    assert widget._workflow_tabs.tabText(0).replace('&&','&') == 'Edit & Confirm'
    assert widget._workflow_tabs.tabText(1).replace('&&','&') == 'Mask Analysis & Cleanup'
    review = widget._reviewed_area_widget
    review.grid_config[1].setValue(256)
    review.grid_config[2].setValue(256)
    review.advanced_check.setChecked(True)
    from napari_label_assistant_tools._mask_tiles import full_source
    for point in ((520,520), (20,20)):
        review.mouse_click(viewer, SimpleNamespace(type='mouse_press',button=1,
            position=full_source(truth).data_to_world(point)))
        assert review.advanced_check.isChecked()
        assert page._tool_tabs.currentWidget() is widget
    mask = associated_mask(viewer,image,truth)
    assert mask.metadata[KEY]['areas'] == [[512,512,768,768], [0,0,256,256]]
    assert 'Confirmed foreground and background' in training.dataset_feedback.text()
    qtbot.waitUntil(lambda: training.reviewed_combo.currentData() is mask, timeout=5000)
    review.advanced_check.setChecked(False)
    assert not review.mouse_review_check.isChecked()
    review.prepare_dataset_button.click()
    assert training.isVisible()
    assert page._tool_tabs.indexOf(training) == -1
    assert '2 confirmed tiles available' in training.annotation_sources_label.text()
    assert training.workflow_combo.isHidden()
    training.dataset_advanced_toggle.setChecked(True)
    assert not training.workflow_combo.isHidden()


def test_too_few_spatial_groups_offers_smaller_patches_without_changing_approvals(qc_editor, qtbot):
    coverage, training = confirm(qc_editor, qtbot)
    areas = list(coverage.metadata[KEY]['areas'])
    training.prepare_button.click()
    qtbot.waitUntil(lambda: training.worker is None, timeout=15000)
    assert training.prepared is None
    assert 'two separate spatial blocks' in training.dataset_feedback.text()
    assert not training.smaller_patches_button.isHidden()
    assert '256 px' in training.smaller_patches_button.text()
    training.smaller_patches_button.click()
    qtbot.waitUntil(lambda: training.worker is None, timeout=15000)
    assert training.crop_combo.currentData() == training.resolution_combo.currentData() == 256
    assert coverage.metadata[KEY]['areas'] == areas


def test_dataset_cleanup_shortcut_indexes_current_area_before_any_removal(qc_editor):
    viewer, image, truth, _, page, widget = qc_editor
    training = page.training
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    page.show_training()
    original = truth.data.copy()
    training.cleanup_masks_button.click()
    assert page._tool_tabs.currentWidget() is widget
    assert widget._workflow_tabs.currentIndex() == 1
    assert widget._component_table.rowCount() == 2
    np.testing.assert_array_equal(truth.data, original)
    assert not widget._analysis_options_section.isChecked()
    assert not widget._confirm_tile_button.isVisible()
    assert not widget._reviewed_area_widget.isVisible()
    widget._component_table.selectRow(0)
    assert associated_mask(viewer, image, truth) is None


def test_qc_switch_and_layout_changes_preserve_full_resolution_crop(qc_editor, qtbot):
    viewer, _, truth, _, _, widget = qc_editor
    widget._workflow_tabs.setCurrentIndex(0)
    follow = widget._local_editing_controls.follow
    pixels = truth.data.copy()
    bounds = truth._label_assistant_mask_entry.bounds
    center, zoom = tuple(viewer.camera.center), viewer.camera.zoom
    widget._workflow_tabs.setCurrentIndex(1)
    assert not widget._local_editing_controls.isVisible()
    assert widget._local_editing_controls.active_view()
    follow.schedule()  # A layout resize must not turn the existing crop into an overview.
    follow.evaluate()
    qtbot.wait(300)
    assert truth._label_assistant_mask_entry.bounds == bounds
    np.testing.assert_array_equal(truth.data, pixels)
    assert truth.visible and truth.editable
    assert tuple(viewer.camera.center) == center and viewer.camera.zoom == zoom
