"""One tile workflow across Simple/Advanced, with durable pending selections."""
from types import SimpleNamespace

import numpy as np
import pytest

from napari_label_assistant_tools._label_assistant import label_assistant_widget
from napari_label_assistant_tools._mask_tiles import full_data, full_source, manager_for
from napari_label_assistant_tools._reviewed_regions import KEY as REVIEW_KEY, associated_mask, effective_areas
from napari_label_assistant_tools._tile_progress import KEY, progress_path
from napari_label_assistant_tools.workspace import load_workspace, save_workspace


@pytest.fixture
def setup_simple(make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((128, 128), dtype='uint8'), name='Image')
    pixels = np.zeros((128, 128), dtype='uint32')
    pixels[4:12, 4:12] = 17
    pixels[68:76, 68:76] = 81
    viewer.add_labels(pixels, name='Curated')
    path = tmp_path/'case.label-assistant.json'
    save_workspace(viewer, path, complete_package=True, xy_chunk=32)
    shell = label_assistant_widget(viewer)
    qtbot.addWidget(shell)
    shell.resize(480, 900)
    shell.show()
    shell.set_mode('simple')
    shell.components._grid_y_spin.setValue(32)
    shell.components._grid_x_spin.setValue(32)
    shell.simple.refresh_sources()
    viewer.camera.center = (0, 64, 64)
    viewer.camera.zoom = 6
    qtbot.waitUntil(lambda: manager_for(viewer) is not None and bool(manager_for(viewer).entries), timeout=10000)
    qtbot.waitUntil(lambda: shell.simple.local.follow.state == 'following', timeout=10000)
    return viewer, shell, path


def pick(panel, y, x):
    panel.canvas_mode.setCurrentIndex(panel.canvas_mode.findData('review'))
    assert panel.review.mouse_review_check.isChecked(), (panel.message.text(), panel.review.status.text())
    event = SimpleNamespace(type='mouse_press', button=1, position=panel.image.data_to_world((y, x)))
    panel.review.mouse_click(panel.viewer, event)
    assert panel.review.selected_cell is not None, panel.review.status.text()
    assert panel.active, (panel.shell.mode, panel.message.text())


def finish(panel, qtbot):
    qtbot.waitUntil(lambda: not panel.busy, timeout=15000)


def test_simple_and_advanced_share_pending_tiles_and_controllers(setup_simple):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    assert label_assistant_widget(viewer) is shell
    assert shell._tool_tabs.count() == 2
    pick(panel, 10, 10)
    pick(panel, 70, 70)
    expected = [(0,0,32,32), (64,64,96,96)]
    assert panel.pending == expected
    assert shell.components._pending_training_tiles() == expected
    truth = panel.truth
    follow = panel.local.follow
    shell.set_mode('advanced')
    assert shell.components._pending_training_tiles() == expected
    shell.set_mode('simple')
    assert panel.pending == expected
    assert panel.truth is truth and panel.local.follow is follow
    pick(panel, 10, 10)
    assert panel.pending == [(64,64,96,96)]
    panel.clear_selection()
    assert not panel.pending and not shell.components._pending_training_tiles()


def test_simple_training_amount_opens_existing_dataset_controls(setup_simple):
    viewer,shell,_=setup_simple
    viewer.window._qt_viewer.hide()  # Dock visibility without offscreen OpenGL rendering.
    viewer.window._qt_window.show()
    shell.simple.patch_selection_button.click()
    assert shell.mode == 'simple'
    assert shell.training.isVisible()
    assert shell._tool_tabs.indexOf(shell.training) == -1
    assert shell.training.workflow_tabs.currentIndex() == 0
    assert not shell.training.patch_selection_group.isHidden()


def test_annotation_batch_buttons_share_state_across_simple_and_advanced(setup_simple,qtbot):
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,_ = setup_simple
    panel = shell.simple
    assert panel.batch_widget.parent() is panel.progress_section.body
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    first = batches.active_batch(batches.get_pair(panel.image,panel.truth))
    assert first['name'] == 'Batch 1' and first['regions'] == [[0,0,32,32]]
    pick(panel,70,70)
    assert panel.pending
    reviewed = associated_mask(viewer,panel.image,panel.truth)
    before = effective_areas(reviewed.metadata[REVIEW_KEY])
    panel.batch_widget.new_button.click()
    second = batches.active_batch(batches.get_pair(panel.image,panel.truth))
    assert second['name'] == 'Batch 2' and second['regions'] == []
    assert panel.pending == []
    assert effective_areas(reviewed.metadata[REVIEW_KEY]) == before
    pick(panel,70,70)
    panel.approve()
    finish(panel,qtbot)
    pair = batches.get_pair(panel.image,panel.truth)
    assert pair['batches'][0]['regions'] == [[0,0,32,32]]
    assert pair['batches'][1]['regions'] == [[64,64,96,96]]
    shell.set_mode('advanced')
    panel.review.update_status()
    shell.training._update_controls()
    selectors = (panel.batch_widget,panel.review.batch_widget,shell.training.batch_widget)
    assert all(second['id'] in selector.buttons for selector in selectors)
    assert all(selector.overlay is panel.batch_widget.overlay for selector in selectors)
    panel.review.batch_widget.buttons[first['id']].click()
    assert batches.active_batch(batches.get_pair(panel.image,panel.truth))['id'] == first['id']
    assert batches.selection(panel.image,panel.truth)['id'] == second['id']
    assert shell.training.batch_widget.buttons[first['id']].isChecked()
    shell.set_mode('simple')
    assert panel.batch_widget.buttons[first['id']].isChecked()


def test_batch_color_remains_visible_after_approval_with_shared_display_filter(setup_simple,qtbot):
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,_ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    panel.batch_widget.new_button.click()
    second = batches.active_batch(batches.get_pair(panel.image,panel.truth))['id']
    pick(panel,70,70)
    panel.approve_button.click()
    finish(panel,qtbot)
    overlay = panel.batch_widget.overlay
    qtbot.waitUntil(lambda:overlay.layer is not None and overlay.layer.visible and
                   overlay.layer.features['batch_id'].tolist() == [second],timeout=3000)
    assert panel.pending == []
    assert viewer.layers.index(overlay.layer) > viewer.layers.index(panel.review.review_overlay)
    assert len(panel.review.selection_overlay.data) == 0
    signature = batches.signature(panel.image,panel.truth)
    panel.batch_widget.display.setCurrentIndex(panel.batch_widget.display.findData('all'))
    assert len(overlay.layer.data) == 2
    assert batches.signature(panel.image,panel.truth) == signature
    assert panel.review.batch_widget.display.currentData() == 'all'
    assert shell.training.batch_widget.display.currentData() == 'all'


def test_new_batch_can_select_and_approve_tiles_while_mask_is_in_overview(setup_simple,qtbot):
    from napari._vispy.mouse_event import NapariMouseEvent
    from napari.utils.interactions import mouse_press_callbacks
    from vispy.app.canvas import MouseEvent
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,_ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    panel.canvas_mode.setCurrentIndex(panel.canvas_mode.findData('edit'))
    shell.components._display_grid_check.setChecked(False)
    panel.local.follow.pause_for_operation()
    panel.local.backend.suspend_follow(manager_for(viewer))
    assert panel.truth.data.shape == (1,1)
    panel.batch_widget.new_button.click()
    panel.canvas_mode.setCurrentIndex(panel.canvas_mode.findData('review'))
    assert panel.review.mouse_review_check.isChecked(),panel.review.status.text()
    assert shell.components._display_grid_check.isChecked()
    grid = viewer.layers[f'{panel.truth.name} | grid_overlay']
    assert np.max(np.concatenate(grid.data),axis=0).tolist() == [128,128]
    assert 'Review selection active' in panel.follow_label.text()
    event = NapariMouseEvent(MouseEvent(type='mouse_press',pos=(10,10),button=1),
                            view_direction=np.zeros(3),up_direction=None,camera_zoom=viewer.camera.zoom,
                            position=panel.image.data_to_world((70,70)),dims_displayed=[0,1],
                            dims_point=[0,0],viewbox=(0,0))
    mouse_press_callbacks(viewer,event)
    assert panel.pending == [(64,64,96,96)]
    assert panel.approve_button.isEnabled()
    assert panel.approve_button.text() == 'Approve 1 selected tiles'
    assert batches.selection(panel.image,panel.truth)['regions'] == []
    panel.approve_button.click()
    finish(panel,qtbot)
    assert batches.selection(panel.image,panel.truth)['name'] == 'Batch 2'
    assert batches.selection(panel.image,panel.truth)['regions'] == [[64,64,96,96]]


def test_batch_state_recovers_without_full_save_and_review_undo_restores_membership(setup_simple,qtbot,make_napari_viewer):
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,path = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    panel.batch_widget.new_button.click()
    pick(panel,70,70)
    panel.approve()
    finish(panel,qtbot)
    assert batches.selection(panel.image,panel.truth)['regions'] == [[64,64,96,96]]
    panel.undo_review()
    finish(panel,qtbot)
    assert batches.selection(panel.image,panel.truth)['regions'] == []
    pick(panel,70,70)
    panel.approve()
    finish(panel,qtbot)
    panel.journal.flush()
    reopened_viewer = make_napari_viewer()
    result = load_workspace(reopened_viewer,path)
    assert not result['progress_warnings']
    reopened = label_assistant_widget(reopened_viewer)
    qtbot.addWidget(reopened)
    reopened.set_mode('simple')
    info = batches.selection(reopened.simple.image,reopened.simple.truth)
    assert info['name'] == 'Batch 2' and info['regions'] == [[64,64,96,96]]
    assert batches.get_pair(reopened.simple.image,reopened.simple.truth)['batches'][0]['regions'] == [[0,0,32,32]]


def test_simple_prepares_only_active_batch_and_switching_invalidates_index(setup_simple,qtbot):
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,_ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    first = batches.active_batch(batches.get_pair(panel.image,panel.truth))['id']
    panel.batch_widget.new_button.click()
    full_data(panel.truth)[32:40,96:104] = 23
    pick(panel,40,110)
    pick(panel,70,70)
    panel.approve()
    finish(panel,qtbot)
    for combo in (shell.training.crop_combo,shell.training.resolution_combo):
        combo.addItem('32 px (test)',32)
        combo.setCurrentIndex(combo.findData(32))
    panel.prepare()
    qtbot.waitUntil(lambda:shell.training.worker is None,timeout=15000)
    dataset = shell.training.prepared
    assert dataset is not None,shell.training.status_label.text()
    assert {(p['y0'],p['x0']) for p in dataset.locations} == {(32,96),(64,64)}
    assert (0,0) not in {(p['y0'],p['x0']) for p in dataset.locations}
    shell.training.device_combo.setCurrentIndex(shell.training.device_combo.findData('cpu'))
    shell.training.models_panel.width.setCurrentIndex(shell.training.models_panel.width.findData(4))
    shell.training.epochs_spin.setValue(1)
    shell.training.extra_targets_spin.setValue(0)
    shell.training.train_button.click()
    qtbot.waitUntil(lambda:shell.training.worker is None,timeout=20000)
    assert shell.training.model is not None,shell.training.status_label.text()
    recorded = shell.training.model.history[-1]['source']['annotation_batches'][0]
    assert recorded['name'] == 'Batch 2' and recorded['regions'] == [[32,96,64,128],[64,64,96,96]]
    prepared = shell.training.prepared
    panel.batch_widget.buttons[first].click()
    assert shell.training.prepared is prepared
    panel.batch_widget.scope.setCurrentIndex(panel.batch_widget.scope.findData(first))
    assert shell.training.prepared is None


def test_batch_selector_refresh_is_read_only_and_does_not_starve_progress_save(setup_simple,qtbot):
    viewer,shell,_ = setup_simple
    panel = shell.simple
    changes = []
    panel.truth.events.metadata.connect(lambda event: changes.append(event))
    for _ in range(30):
        panel.batch_widget.refresh()
    assert not changes
    pick(panel,10,10)
    qtbot.waitUntil(lambda:panel._save_activity == 'saved',timeout=5000)


def test_batch_context_action_keeps_multiple_selected_regions_when_switching(setup_simple,qtbot):
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,_ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    pick(panel,70,70)
    panel.approve()
    finish(panel,qtbot)
    first = batches.active_batch(batches.get_pair(panel.image,panel.truth))['id']
    panel.batch_widget.new_button.click()
    second = batches.active_batch(batches.get_pair(panel.image,panel.truth))['id']
    panel.batch_widget.buttons[first].click()
    pick(panel,10,10)
    pick(panel,70,70)
    assert len(panel.pending) == 2
    selector = panel.batch_widget
    selector._menu_action(second,selector.add_reviewed,True)
    assert batches.selection(panel.image,panel.truth)['regions'] == [[0,0,32,32],[64,64,96,96]]


def test_batch_metadata_survives_full_workspace_save_and_empty_batch_disables_prepare(setup_simple,qtbot,make_napari_viewer):
    from napari_label_assistant_tools import _annotation_batches as batches
    viewer,shell,path = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    panel.batch_widget.new_button.click()
    panel.refresh_status()
    assert not panel.prepare_button.isEnabled()
    assert not shell.training.prepare_button.isEnabled()
    assert 'empty' in panel.dataset_label.text()
    save_workspace(viewer,path)
    reopened_viewer = make_napari_viewer()
    load_workspace(reopened_viewer,path)
    reopened = label_assistant_widget(reopened_viewer)
    qtbot.addWidget(reopened)
    reopened.set_mode('simple')
    pair = batches.get_pair(reopened.simple.image,reopened.simple.truth)
    assert pair['scope'] == 'active' and len(pair['batches']) == 2
    assert batches.active_batch(pair)['name'] == 'Batch 2'
    assert pair['batches'][0]['regions'] == [[0,0,32,32]]


def test_pending_tiles_restore_without_full_workspace_save(setup_simple, make_napari_viewer, qtbot):
    viewer, shell, path = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    pick(panel, 70, 70)
    before = path.read_bytes()
    panel.journal.flush()
    assert path.read_bytes() == before
    assert progress_path(path).exists()
    restored = make_napari_viewer()
    result = load_workspace(restored, path)
    assert not result['progress_warnings']
    reopened = label_assistant_widget(restored)
    qtbot.addWidget(reopened)
    reopened.set_mode('simple')
    assert reopened.simple.pending == [(0,0,32,32),(64,64,96,96)]
    assert associated_mask(restored, reopened.simple.image, reopened.simple.truth) is None


def test_selection_save_shows_busy_then_saved_progress(setup_simple):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    bar = panel.workflow_progress
    assert not bar.isHidden() and bar.maximum() == 0 and bar.minimum() == 0
    panel.journal.flush()
    assert bar.maximum() == 1 and bar.value() == 1
    assert bar.format() == 'Tile progress saved'
    assert panel.pending == [(0,0,32,32)]
    assert associated_mask(viewer,panel.image,panel.truth) is None


def test_header_keeps_counts_and_batch_visible_when_scrolled_or_switched(setup_simple, qtbot):
    viewer, shell, _ = setup_simple
    panel, header = shell.simple, shell._activity_header
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    qtbot.waitUntil(lambda: '1 / 16' in header.confirmed_label.toolTip())
    assert panel.workflow_progress is header.progress
    assert header.progress.parentWidget() is header
    assert 'Batch 1' in header.batch_label.toolTip()
    assert '1 / 1' in header.batch_label.toolTip()
    for width in (360, 480, 520):
        shell.resize(width, 640)
        panel.verticalScrollBar().setValue(panel.verticalScrollBar().maximum())
        qtbot.wait(20)
        assert header.isVisible()
        assert header.geometry().bottom() < shell.stack.geometry().top()
        assert 'Confirmed' in header.confirmed_label.text()
    shell.set_mode('advanced')
    assert header.isVisible()
    assert '1 / 16' in header.confirmed_label.toolTip()
    # Annotation changes revoke approval and remain visible above the scroll area.
    shell.set_mode('simple')
    panel.canvas_mode.setCurrentIndex(panel.canvas_mode.findData('edit'))
    panel.truth.brush_size = 1
    panel.truth.paint((15, 15), 55)
    manager_for(viewer).flush()
    panel.review.update_status()
    panel.refresh_status()
    assert '0 / 16' in header.confirmed_label.toolTip()


def test_selection_progress_finishes_automatically_without_manual_save(setup_simple,qtbot):
    from json import loads
    _, shell, path = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    pick(panel,70,70)
    qtbot.waitUntil(lambda: panel._save_activity == 'saved',timeout=5000)
    state = loads(progress_path(path).read_text())
    assert any(record.get(KEY,{}).get('pending') == [[0,0,32,32],[64,64,96,96]] for record in state['layers'])
    assert panel.workflow_progress.value() == panel.workflow_progress.maximum() == 1


def test_approval_progress_counts_completed_tiles(setup_simple,qtbot):
    _, shell, _ = setup_simple
    panel = shell.simple
    for y,x in [(10,10),(10,40),(70,70)]:
        pick(panel,y,x)
    counts = []
    bar = panel.workflow_progress
    bar.valueChanged.connect(lambda count: counts.append((count,bar.maximum())))
    panel.approve()
    finish(panel,qtbot)
    assert all((count,3) in counts for count in (0,1,2,3))
    panel.journal.flush()
    assert bar.value() == 3 and bar.maximum() == 3
    assert bar.format() == 'Updated · %v / %m tiles'
    assert panel.message.text() == ''


def test_cancelled_progress_keeps_actual_partial_count(setup_simple,qtbot):
    _, shell, _ = setup_simple
    panel = shell.simple
    panel._operate(1,[(0,0,32,32),(0,32,32,64),(64,64,96,96)])
    panel._cancelled = True
    finish(panel,qtbot)
    panel.journal.flush()
    bar = panel.workflow_progress
    assert bar.maximum() == 3 and bar.value() == 1
    assert bar.format().startswith('Stopped')
    assert panel.message.text() == ''


def test_failed_journal_does_not_show_saved_progress(setup_simple,monkeypatch):
    from napari_label_assistant_tools import _tile_progress
    _, shell, _ = setup_simple
    panel = shell.simple
    pick(panel,10,10)

    def fail(*args,**kwargs):
        raise OSError('Test storage unavailable')

    monkeypatch.setattr(_tile_progress,'write_progress',fail)
    panel.journal.flush()
    assert panel.workflow_progress.maximum() == 1
    assert panel.workflow_progress.value() == 0
    assert panel.workflow_progress.format() == 'Tile progress not saved'
    assert 'Test storage unavailable' in panel.persistence_label.text()


def test_clearing_review_selection_cannot_approve_previous_cell(setup_simple):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.clear_selection()
    assert not panel.approve_button.isEnabled()
    panel.approve_button.click()
    assert associated_mask(viewer,panel.image,panel.truth) is None
    with pytest.raises(ValueError,match='Select fully checked tiles'):
        panel.approve()


def test_pending_overlay_preserves_active_paint_layer(setup_simple):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    viewer.layers.selection.active = panel.truth
    panel.truth.mode = 'erase'
    panel._draw_pending()
    assert viewer.layers.selection.active is panel.truth
    assert str(panel.truth.mode) == 'erase'


def test_pending_highlight_has_translucent_fill_and_scales_with_zoom(setup_simple,qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    original = np.asarray(full_data(panel.truth)[:]).copy()
    pick(panel,10,10)
    overlay = panel.review.selection_overlay
    assert 0.1 < overlay.face_color[0,3] < 0.25
    assert overlay.opacity == 1.0
    viewer.camera.zoom = 2
    qtbot.wait(50)
    panel.refresh_status()
    close_width = overlay.edge_width[0]
    viewer.camera.zoom = 0.02
    qtbot.wait(50)
    panel.refresh_status()
    assert overlay.edge_width[0] > close_width*50
    assert panel.pending == [(0,0,32,32)]
    np.testing.assert_array_equal(full_data(panel.truth)[:],original)


def test_clearing_and_reselecting_retains_highlight_style(setup_simple):
    _, shell, _ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.clear_selection()
    assert len(panel.review.selection_overlay.data) == 0
    pick(panel,70,70)
    overlay = panel.review.selection_overlay
    assert len(overlay.data) == 1 and overlay.face_color[0,3] > 0.1
    assert overlay.edge_color[0,3] == 1
    assert panel.pending == [(64,64,96,96)]


def test_advanced_outlines_do_not_inherit_overview_border_width(setup_simple,qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    viewer.camera.zoom = 0.02
    qtbot.wait(50)
    panel.refresh_status()
    overlay = panel.review.selection_overlay
    assert overlay.edge_width[0] > 2
    shell.set_mode('advanced')
    assert overlay.edge_width[0] == 2
    assert overlay.face_color[0,3] == 0
    shell.set_mode('simple')
    assert overlay.edge_width[0] > 2 and overlay.face_color[0,3] > 0.1
    assert panel.pending == [(0,0,32,32)]


def test_revoke_all_requires_confirmation_and_preserves_source(setup_simple,qtbot,monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    viewer, shell, _ = setup_simple
    panel = shell.simple
    original = np.asarray(full_data(panel.truth)[:]).copy()
    panel._operate(1,[(0,0,32,32),(64,64,96,96)])
    finish(panel,qtbot)
    mask = associated_mask(viewer,panel.image,panel.truth)
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.No)
    panel.revoke_all()
    assert len(effective_areas(mask.metadata[REVIEW_KEY])) == 2
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.Yes)
    panel.revoke_all()
    finish(panel,qtbot)
    assert not effective_areas(mask.metadata[REVIEW_KEY])
    np.testing.assert_array_equal(full_data(panel.truth)[:],original)
    panel.undo_review()
    finish(panel,qtbot)
    assert len(effective_areas(mask.metadata[REVIEW_KEY])) == 2


def test_review_is_durable_and_preserves_label_values(setup_simple, make_napari_viewer, qtbot):
    viewer, shell, path = setup_simple
    panel = shell.simple
    truth = panel.truth
    original = np.asarray(full_data(truth)[:]).copy()
    pick(panel, 10, 10)
    pick(panel, 70, 70)
    panel.approve()
    finish(panel, qtbot)
    mask = associated_mask(viewer, panel.image, truth)
    assert mask is not None, panel.message.text()
    assert not getattr(full_source(mask), '_label_assistant_mask_temporary', False)
    assert full_data(mask)[10,10] == 1 and full_data(mask)[70,70] == 1
    np.testing.assert_array_equal(full_data(truth)[:], original)
    restored = make_napari_viewer()
    result = load_workspace(restored, path)
    assert not result['progress_warnings']
    restored_mask = next(layer for layer in restored.layers if layer.metadata.get(REVIEW_KEY,{}).get('role') == 'reviewed_regions')
    assert full_data(restored_mask)[10,10] == 1 and full_data(restored_mask)[70,70] == 1
    assert len(effective_areas(restored_mask.metadata[REVIEW_KEY])) == 2


@pytest.mark.parametrize('points', [[(10,10)], [(10,10),(70,70)]])
def test_approval_keeps_colored_coverage_visible(setup_simple,qtbot,points):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    original = np.asarray(full_data(panel.truth)[:]).copy()
    for y,x in points:
        pick(panel,y,x)
    panel.approve()
    finish(panel,qtbot)
    mask = associated_mask(viewer,panel.image,panel.truth)
    entry = mask._label_assistant_mask_entry
    assert full_source(mask).visible
    assert mask.visible and entry.overview.visible
    assert panel.review.review_overlay.visible
    assert not panel.pending
    # Camera-follow overview transitions must retain the approved colors.
    panel.local.follow.pause_for_operation()
    panel.local.backend.suspend_follow(manager_for(viewer))
    assert not mask.visible  # The 1x1 placeholder is deliberately hidden.
    assert full_source(mask).visible and entry.overview.visible
    entry.refresh_overview()
    assert entry.overview.visible
    np.testing.assert_array_equal(full_data(panel.truth)[:],original)


def test_show_coverage_reveals_hidden_mask_without_changing_edit_focus(setup_simple,qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel,10,10)
    panel.approve()
    finish(panel,qtbot)
    mask = associated_mask(viewer,panel.image,panel.truth)
    mask.visible = False
    full_source(mask).visible = False
    mask._label_assistant_mask_entry.sync_overview_display()
    panel.canvas_mode.setCurrentIndex(panel.canvas_mode.findData('edit'))
    viewer.layers.selection.active = panel.truth
    panel.truth.mode = 'erase'
    panel.review.show_coverage()
    assert full_source(mask).visible and mask._label_assistant_mask_entry.overview.visible
    assert viewer.layers.selection.active is panel.truth
    assert str(panel.truth.mode) == 'erase'


def test_revoke_and_undo_only_change_review_approval(setup_simple, qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    truth = panel.truth
    original = np.asarray(full_data(truth)[:]).copy()
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    pick(panel, 10, 10)
    panel.revoke()
    finish(panel, qtbot)
    mask = associated_mask(viewer, panel.image, truth)
    assert not effective_areas(mask.metadata[REVIEW_KEY])
    panel.undo_review()
    finish(panel, qtbot)
    assert full_data(mask)[10,10] == 1, panel.message.text()
    np.testing.assert_array_equal(full_data(truth)[:], original)


def test_prepare_uses_existing_training_widget(setup_simple, qtbot):
    _, shell, _ = setup_simple
    panel, training = shell.simple, shell.training
    pick(panel, 10, 10)
    pick(panel, 70, 70)
    panel.approve()
    finish(panel, qtbot)
    for combo in (training.crop_combo, training.resolution_combo):
        combo.addItem('32 px (test)', 32)
        combo.setCurrentIndex(combo.findData(32))
    panel.prepare()
    qtbot.waitUntil(lambda: training.worker is None, timeout=15000)
    assert training.prepared is not None, training.status.text() if hasattr(training,'status') else panel.dataset_label.text()
    assert training.truth_combo.currentData() is panel.truth
    assert training.annotation_combo.currentData() == 'partial'
    assert training.workflow_combo.currentData() == 'coverage'
    assert training.prepared.reviewed_pixels == 2*32*32


def test_prepare_preserves_configured_multiple_image_selection(setup_simple,monkeypatch):
    from copy import deepcopy
    _, shell, _ = setup_simple
    training = shell.training
    training.dataset_source_mode.setCurrentIndex(training.dataset_source_mode.findData('multiple'))
    before = deepcopy(training.dataset_pairs)
    called = []
    monkeypatch.setattr(training.prepare_button,'click',lambda:called.append(True))
    shell.simple.prepare()
    assert called
    assert training._multiple_pairs()
    assert training.dataset_pairs == before


def test_pixel_edits_revoke_review_but_keep_training_history(setup_simple, qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    truth = panel.truth
    for owner in (truth, full_source(truth)):
        owner.metadata[REVIEW_KEY]['training_areas'] = [[0,0,32,32]]
    panel.canvas_mode.setCurrentIndex(panel.canvas_mode.findData('edit'))
    truth.brush_size = 1
    truth.paint((15,15), 55)
    manager_for(viewer).flush()
    mask = associated_mask(viewer, panel.image, truth)
    # Coverage pixels are retained; approval metadata excludes edited cells.
    assert not effective_areas(mask.metadata[REVIEW_KEY])
    pick(panel, 10, 10)
    panel.review.update_status()
    assert 'Needs recheck' in panel.review.cell_label.text()
    assert truth.metadata[REVIEW_KEY]['training_areas'] == [[0,0,32,32]]
    assert panel._undo is None


def test_progress_is_scoped_to_manifest_revision(setup_simple, make_napari_viewer):
    _, shell, path = setup_simple
    pick(shell.simple, 10,10)
    shell.simple.journal.flush()
    path.write_text(path.read_text()+'\n')
    other = make_napari_viewer()
    result = load_workspace(other,path)
    assert 'different workspace revision' in result['progress_warnings'][0]
    truth = next(layer for layer in other.layers if layer.name == 'Curated')
    assert not full_source(truth).metadata.get(KEY,{}).get('pending')


def test_undo_first_approval_keeps_coverage_association(setup_simple, qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    pick(panel, 10, 10)
    panel.approve()
    finish(panel, qtbot)
    mask = associated_mask(viewer, panel.image, panel.truth)
    panel.undo_review()
    finish(panel, qtbot)
    assert associated_mask(viewer, panel.image, panel.truth) is mask
    assert not effective_areas(mask.metadata[REVIEW_KEY])
    assert full_data(mask)[10, 10] == 0
    pick(panel, 70, 70)
    panel.approve()
    finish(panel, qtbot)
    assert associated_mask(viewer, panel.image, panel.truth) is mask
    assert full_data(mask)[70, 70] == 1


def test_cancelled_undo_does_not_restore_unprocessed_approval(setup_simple, qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    rectangles = [(0,0,32,32), (64,64,96,96)]
    panel._operate(1, rectangles)
    finish(panel, qtbot)
    panel._operate(0, rectangles)
    finish(panel, qtbot)
    panel.undo_review()  # First rectangle is processed synchronously.
    panel._cancelled = True
    finish(panel, qtbot)
    mask = associated_mask(viewer, panel.image, panel.truth)
    assert full_data(mask)[10,10] == 1
    assert full_data(mask)[70,70] == 0
    assert not any(a<=70<c and b<=70<d for a,b,c,d in effective_areas(mask.metadata[REVIEW_KEY]))
    assert panel._undo is None


def test_skip_moves_to_unreviewed_cell_without_approving_it(setup_simple, qtbot):
    viewer, shell, _ = setup_simple
    panel = shell.simple
    panel._operate(1, [(0,32,32,64)])
    finish(panel, qtbot)
    pick(panel, 10, 10)
    panel.clear_selection()
    panel.skip()
    _, bounds, _, _ = panel.review.selected_cell
    assert (bounds.y0,bounds.x0,bounds.y1,bounds.x1) == (0,64,32,96)
    mask = associated_mask(viewer, panel.image, panel.truth)
    assert effective_areas(mask.metadata[REVIEW_KEY]) == [[0,32,32,64]]


def test_pending_snapshot_survives_layer_teardown(setup_simple, make_napari_viewer, qtbot):
    from napari_label_assistant_tools._tile_progress import capture_progress
    viewer, shell, path = setup_simple
    pick(shell.simple,10,10)
    cached = capture_progress(viewer,path)
    assert any(KEY in record for record in cached.values()), cached
    viewer.layers.clear()  # Viewer.close clears layers before closing the Qt window.
    assert any(KEY in record for record in capture_progress(viewer,path).values())
    shell.simple.journal.flush()
    restored = make_napari_viewer()
    result = load_workspace(restored,path)
    assert not result['progress_warnings']
    truth = next(layer for layer in restored.layers if layer.name == 'Curated')
    assert full_source(truth).metadata[KEY]['pending'] == [[0,0,32,32]]


def test_progress_snapshot_never_reads_source_pixels(setup_simple, monkeypatch):
    from napari_label_assistant_tools._tile_progress import capture_progress, write_progress
    viewer, shell, path = setup_simple
    pick(shell.simple,10,10)
    array_type = type(full_data(shell.simple.truth))

    def forbidden(*args, **kwargs):
        raise AssertionError('A progress snapshot must not read full-resolution pixels')

    with monkeypatch.context() as patch:
        patch.setattr(array_type,'__getitem__',forbidden)
        patch.setattr(array_type,'__array__',forbidden)
        assert capture_progress(viewer,path)
        assert write_progress(viewer,path) == progress_path(path)


def test_simple_and_advanced_menu_reuse_one_dock(make_napari_viewer):
    from napari_label_assistant_tools._label_assistant import _open_mode
    from npe2 import PluginManifest
    from pathlib import Path
    import napari_label_assistant_tools
    manifest = PluginManifest.from_file(Path(napari_label_assistant_tools.__file__).with_name('napari.yaml'))
    assert [item.command for item in manifest.contributions.menus['napari-label-assistant.main']] == [
        'napari-label-assistant.simple_curation', 'napari-label-assistant.advanced_curation',
        'napari-label-assistant.quick_train', 'napari-label-assistant.model_library']
    viewer = make_napari_viewer()
    simple = _open_mode(viewer,'simple')
    assert simple.mode == 'simple'
    advanced = _open_mode(viewer,'advanced')
    assert advanced is simple and simple.mode == 'advanced'
    assert len(viewer.window.dock_widgets) == 1
    from napari._app_model import get_app_model
    get_app_model().commands.execute_command('napari-label-assistant.simple_curation').result()
    assert simple.mode == 'simple'
    get_app_model().commands.execute_command('napari-label-assistant.advanced_curation').result()
    assert simple.mode == 'advanced' and len(viewer.window.dock_widgets) == 1


def test_unrelated_volume_does_not_break_simple_view(make_napari_viewer,qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((32,32),dtype='uint8'),name='Aligned image')
    viewer.add_image(np.zeros((4,32,32),dtype='uint8'),name='Unrelated volume')
    truth = viewer.add_labels(np.zeros((32,32),dtype='uint8'),name='Curated')
    shell = label_assistant_widget(viewer)
    qtbot.addWidget(shell)
    shell.set_mode('simple')
    assert shell.simple.truth is truth
    assert shell.simple.image is image
    assert shell.simple.image_combo.count() == 2  # Placeholder plus compatible image.


def test_sparse_100k_mask_can_be_reviewed_from_overview(make_napari_viewer,qtbot,tmp_path,monkeypatch):
    import json
    import zarr
    from napari_label_assistant_tools.workspace.service import _create_zarr_array, WORKSPACE_FORMAT
    image_root = zarr.open_group(str(tmp_path/'image.zarr'),mode='w')
    for name, size in (('s0',100000),('s6',1563)):
        _create_zarr_array(image_root,name,shape=(size,size),chunks=(256,256),dtype='uint8',fill_value=0)
    array = zarr.open_array(str(tmp_path/'mask.zarr'),mode='w',shape=(100000,100000),chunks=(512,512),dtype='uint32')
    layers = [dict(layer_type='Image',name='Large image',label_assistant_review=dict(id='image'),
                   storage=dict(kind='zarr',path='image.zarr',array_path='',multiscale=True,datasets=['s0','s6'])),
              dict(layer_type='Labels',name='Curated',label_assistant_review=dict(id='truth'),
                   storage=dict(kind='zarr',path='mask.zarr',array_path=''))]
    path = tmp_path/'large.label-assistant.json'
    path.write_text(json.dumps(dict(format=WORKSPACE_FORMAT,version=2,layers=layers,
                                  mask_view=dict(mode='grid',tile_size=512))))
    viewer = make_napari_viewer()
    assert not load_workspace(viewer,path)['skipped_layers']
    shell = label_assistant_widget(viewer)
    qtbot.addWidget(shell)
    shell.set_mode('simple')
    panel = shell.simple
    panel.grid_combo.setCurrentIndex(panel.grid_combo.findData((1024,1024)))
    shell.components._display_grid_check.setChecked(False)
    panel.local.follow.pause_for_operation()
    panel.local.backend.suspend_follow(manager_for(viewer))
    assert panel.truth._label_assistant_mask_entry.bounds is None
    panel.batch_widget.overlay.set_context(panel.image,panel.truth)
    original = type(array).__getitem__
    reads = []

    def bounded(source, region):
        if tuple(source.shape) == (100000,100000):
            assert isinstance(region,tuple) and len(region) == 2
            assert all(isinstance(s,slice) and s.stop is not None and s.stop-(s.start or 0)<=1024 for s in region)
            reads.append(region)
        return original(source,region)

    with monkeypatch.context() as patch:
        patch.setattr(type(array),'__getitem__',bounded)
        pick(panel,98500,98500)
        assert panel.pending == [(98304,98304,99328,99328)]
        panel.approve_button.click()
        finish(panel,qtbot)
        panel.batch_widget.overlay.render()
        assert panel.batch_widget.overlay.layer.visible
        assert len(panel.batch_widget.overlay.layer.data) == 1
        panel.journal.flush()
    coverage = associated_mask(viewer,panel.image,panel.truth)
    assert full_data(coverage)[98500,98500] == 1
    assert full_data(coverage)[100,100] == 0
    assert full_data(panel.truth).shape == (100000,100000)
    assert array[98500,98500] == 0  # Review approval never changes biological annotations.
