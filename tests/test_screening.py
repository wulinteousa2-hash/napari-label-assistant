"""Human feature groups keep arbitrary source IDs and checked pending picks."""
import json
from types import SimpleNamespace

import numpy as np
from napari_label_assistant_tools import _screening as screening
from napari_label_assistant_tools._large_components import build_large_component_index
from napari_label_assistant_tools._mask_tiles import full_data
from napari_label_assistant_tools._widget import component_operations_widget, _build_fast_component_index
from napari_label_assistant_tools.workspace import load_workspace, save_workspace


def setup(viewer, qtbot):
    pixels = np.zeros((24, 28), dtype=np.uint32)
    pixels[2:5, 3:7] = 17
    pixels[14:18, 18:22] = 81
    source = viewer.add_labels(pixels, name='SAM3 objects', scale=(2, 3), translate=(10, 20))
    widget = component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._workflow_tabs.setCurrentIndex(1)
    widget._analyze_button.click()
    widget._click_select_check.setChecked(True)
    widget._append_click_select_check.setChecked(True)
    return widget, source


def pick(viewer, source, position):
    callback = next(callback for callback in viewer.mouse_drag_callbacks
                    if getattr(callback, '__name__', '') == '_handle_mouse_click')
    callback(viewer, SimpleNamespace(type='mouse_press', position=source.data_to_world(position)))


def test_mixed_object_values_toggle_undo_clear_and_lightweight_boxes(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget, source = setup(viewer, qtbot)
    original = source.data.copy()
    pick(viewer, source, (3, 4))
    pick(viewer, source, (15, 19))
    assert widget._component_table.selected_component_ids() == [1, 2]
    overlay = widget._selection_overlay.layer
    assert len(overlay.data) == 2
    assert viewer.layers.selection.active is source
    np.testing.assert_allclose(overlay.data_to_world((2, 3)), source.data_to_world((2, 3)))
    pick(viewer, source, (3, 4))
    assert widget._component_table.selected_component_ids() == [2]
    widget._undo_selection_button.click()
    assert widget._component_table.selected_component_ids() == [1, 2]
    widget._clear_selection_button.click()
    assert widget._component_table.selected_component_ids() == []
    assert not overlay.visible
    np.testing.assert_array_equal(source.data, original)


def test_new_group_copies_original_values_and_clears_only_on_success(make_napari_viewer, qtbot, monkeypatch):
    viewer = make_napari_viewer()
    widget, source = setup(viewer, qtbot)
    original = source.data.copy()
    pick(viewer, source, (3, 4))
    pick(viewer, source, (15, 19))
    monkeypatch.setattr('napari_label_assistant_tools._widget.QInputDialog.getText', lambda *args: ('Remak bundles', True))
    widget._new_group_button.click()
    assert widget._copy_target_combo.currentText() == 'Remak bundles'
    assert widget._component_table.selected_component_ids() == [1, 2]
    widget._clear_after_copy_check.setChecked(True)
    widget._copy_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy)
    np.testing.assert_array_equal(full_data(viewer.layers['Remak bundles'])[:], original)
    np.testing.assert_array_equal(source.data, original)
    assert widget._component_table.selected_component_ids() == []
    assert source.metadata[screening.KEY]['groups']['Remak bundles']['components'] == []


def test_cancel_keeps_pending_selection(make_napari_viewer, qtbot, monkeypatch):
    viewer = make_napari_viewer()
    widget, source = setup(viewer, qtbot)
    destination = viewer.add_labels(np.zeros((24, 28), dtype=np.uint32), name='Remak bundles', scale=(2, 3), translate=(10, 20))
    widget._refresh_layers_button.click()
    pick(viewer, source, (3, 4))
    widget._manual_copy_target_check.setChecked(True)
    widget._copy_target_combo.setCurrentText(destination.name)
    widget._clear_after_copy_check.setChecked(True)
    monkeypatch.setattr('napari_label_assistant_tools._copy_destination.confirm_manual_destination', lambda *args: False)
    widget._copy_button.click()
    assert widget._component_table.selected_component_ids() == [1]
    assert not np.any(destination.data)


def test_pending_groups_can_switch_before_creating_output_layers(make_napari_viewer, qtbot, monkeypatch):
    viewer = make_napari_viewer()
    widget, source = setup(viewer, qtbot)
    pick(viewer, source, (3, 4))
    names = iter(['Remak bundles', 'Axons'])
    monkeypatch.setattr('napari_label_assistant_tools._widget.QInputDialog.getText', lambda *args: (next(names), True))
    widget._new_group_button.click()
    widget._clear_selection_button.click()
    pick(viewer, source, (3, 4))
    widget._new_group_button.click()
    widget._clear_selection_button.click()
    pick(viewer, source, (15, 19))
    widget._refresh_layers_button.click()
    assert widget._copy_target_combo.findText('Remak bundles') >= 0
    widget._copy_target_combo.setCurrentText('Remak bundles')
    widget._switch_screening_group()
    assert widget._component_table.selected_component_ids() == [1]
    widget._copy_target_combo.setCurrentText('Axons')
    widget._switch_screening_group()
    assert widget._component_table.selected_component_ids() == [2]
    assert 'Remak bundles' not in viewer.layers and 'Axons' not in viewer.layers


def test_edited_source_disables_copy_and_changed_object_needs_reselection(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget, source = setup(viewer, qtbot)
    pick(viewer, source, (3, 4))
    widget._click_select_check.setChecked(False)
    source.data[3, 4] = 0
    source.events.data()
    assert not widget._copy_button.isEnabled()
    assert not widget._selection_overlay.layer.visible
    assert 'Source annotations changed' in widget._screening_hint.text()
    widget._analyze_button.click()
    assert widget._component_table.selected_component_ids() == []
    assert '1 saved selection changed' in widget._screening_hint.text()
    widget._clear_selection_button.click()
    assert 'changed' not in widget._screening_hint.text()


def test_pending_groups_survive_workspace_roundtrip(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    viewer = make_napari_viewer()
    widget, source = setup(viewer, qtbot)
    pick(viewer, source, (3, 4))
    monkeypatch.setattr('napari_label_assistant_tools._widget.QInputDialog.getText', lambda *args: ('Remak bundles', True))
    widget._new_group_button.click()
    saved = source.metadata[screening.KEY]
    path = tmp_path / 'screening.label-assistant.json'
    save_workspace(viewer, path, xy_chunk=16)
    payload = json.loads(path.read_text())
    assert all(layer['name'] != 'Screening selection' for layer in payload['layers'])
    restored = make_napari_viewer()
    load_workspace(restored, path)
    reopened = component_operations_widget(restored)
    qtbot.addWidget(reopened)
    reopened._workflow_tabs.setCurrentIndex(1)
    assert reopened._copy_target_combo.currentText() == 'Remak bundles'
    assert '1 saved selection' in reopened._screening_hint.text()
    reopened._analyze_button.click()
    qtbot.waitUntil(lambda: not reopened._large_controller.busy, timeout=15000)
    assert len(reopened._component_table.selected_component_ids()) == 1
    assert source.metadata[screening.KEY] == saved


def finish(generator):
    while True:
        try:
            next(generator)
        except StopIteration as result:
            return result.value


def test_footprints_survive_index_renumbering_but_reject_changed_pixels():
    pixels = np.zeros((16, 16), dtype=np.uint32)
    pixels[4:12, 4:12] = 81
    pixels[6, 6] = 0
    small = _build_fast_component_index(pixels)
    saved = dict(context={}, components=[screening.describe(small, 1)])
    pixels[0, 0] = 7  # Renumbers the original object; its footprint is unchanged.
    large = finish(build_large_component_index(pixels, tile_size=6))
    try:
        assert screening.restore(large, saved, {}, {}) == ({2}, 0)
    finally:
        large.close()
    pixels[6, 6], pixels[7, 7] = 81, 0  # Same value, area, and bounds, different footprint.
    changed = _build_fast_component_index(pixels)
    assert screening.restore(changed, saved, {}, {}) == (set(), 1)


def test_large_screening_uses_cached_bounded_masks_without_source_reads():
    class Lazy:
        shape = (2050, 2050)
        dtype = np.dtype('uint32')
        blocked = False

        def __array__(self, *args, **kwargs):
            raise AssertionError('Whole-image conversion is forbidden')

        def __getitem__(self, slices):
            assert not self.blocked, 'Screening must not reread source pixels'
            ys, xs = slices
            block = np.zeros((ys.stop-ys.start, xs.stop-xs.start), dtype=self.dtype)
            if ys.start <= 1025 < ys.stop and xs.start <= 1025 < xs.stop:
                block[1025-ys.start, 1025-xs.start] = 81
            return block

    source = Lazy()
    index = finish(build_large_component_index(source, tile_size=1024))
    source.blocked = True
    try:
        descriptor = screening.describe(index, 1)
        assert screening.restore(index, dict(context={}, components=[descriptor]), {}, {}) == ({1}, 0)
        assert screening.restore(index, dict(context={}, components=[descriptor]), {'shape': [1, 1]}, {}) == (set(), 1)
    finally:
        index.close()
