"""Entire-mask indexing and cleanup are independent of the displayed crop."""
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from napari_label_assistant_tools._editing_backend import LocalEditingBackend
from napari_label_assistant_tools._mask_tiles import MaskTileManager
from napari_label_assistant_tools._widget import component_operations_widget
from napari_label_assistant_tools.workspace.service import _persist_labels_to_zarr


@pytest.fixture
def entire_mask(tmp_path, make_napari_viewer, qtbot):
    pixels = np.zeros((2200,2300), dtype='uint8')
    pixels[100:105,1100:1105] = 7
    pixels[600:604,600:604] = 7
    pixels[2046:2051,2046:2051] = 7  # Connected across both 2048-pixel seams.
    pixels[2100,2200] = 9
    path = tmp_path/'mask.ome.zarr'
    _persist_labels_to_zarr(pixels,path,xy_chunk=256,scale=(2,3))
    source = zarr.open_group(str(path),mode='r+')['s0']
    viewer = make_napari_viewer()
    manager = MaskTileManager(viewer,dict(tile_size=512,camera_follow=False))
    truth = manager.add(dict(name='Mask',scale=[2,3],translate=[11,17]),source,
                        dict(path=path,array_path='s0'))
    widget = component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._workflow_tabs.setCurrentIndex(1)
    return viewer,manager,truth,source,widget


def find_all(widget, qtbot):
    assert widget._analysis_scope_combo.currentData() == 'whole'
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: widget._large_controller.worker is None,timeout=15000)
    assert widget._component_table.rowCount() == 4, widget._status_label.text()


def test_entire_search_joins_seams_and_survives_camera_and_crop_navigation(entire_mask,qtbot,monkeypatch):
    viewer,manager,truth,source,widget = entire_mask
    assert not np.any(truth.data)  # Empty displayed crop, objects elsewhere.
    original = zarr.Array.__getitem__
    def bounded(array,key):
        if array.shape == source.shape and isinstance(key,tuple) and all(isinstance(s,slice) for s in key):
            assert np.prod([len(range(*s.indices(n))) for s,n in zip(key,array.shape)]) <= 2048**2
        return original(array,key)
    monkeypatch.setattr(zarr.Array,'__getitem__',bounded)
    viewer.camera.zoom = .001
    find_all(widget,qtbot)
    assert sorted(int(widget._component_table.item(row,2).text()) for row in range(4)) == [1,16,25,25]
    manager.go(1,1)
    widget._click_select_check.setChecked(True)
    assert widget._component_table.rowCount() == 4
    callback = next(cb for cb in viewer.mouse_drag_callbacks if getattr(cb,'__name__','') == '_handle_mouse_click')
    point = truth._label_assistant_mask_entry.source.data_to_world((2100,2200))
    callback(viewer,SimpleNamespace(type='mouse_press',button=1,position=point))
    assert widget._component_table.selected_component_ids() == [4]
    widget._component_table._locate_callback(4)
    np.testing.assert_allclose(viewer.camera.center[-2:],point)


def test_whole_mask_cleanup_and_undo_work_in_overview_only_state(entire_mask,qtbot):
    _,manager,truth,source,widget = entire_mask
    LocalEditingBackend().suspend_follow(manager)
    assert truth._label_assistant_mask_entry.bounds is None
    find_all(widget,qtbot)
    widget._small_component_size_spin.setValue(2)
    widget._remove_small_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy,timeout=15000)
    assert source[2100,2200] == 0, widget._status_label.text()
    assert source[2047,2047] == 7
    assert widget._component_table.rowCount() == 3
    widget._undo_delete_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy,timeout=15000)
    assert source[2100,2200] == 9, widget._status_label.text()
    assert source[2047,2047] == 7
    find_all(widget,qtbot)


def test_global_component_copy_keeps_original_image_coordinates(entire_mask,tmp_path,qtbot):
    viewer,manager,_,source,widget = entire_mask
    path = tmp_path/'destination.zarr'
    target = zarr.open_array(str(path),mode='w',shape=source.shape,chunks=(256,256),dtype='uint8')
    destination = manager.add(dict(name='Curated',scale=[2,3],translate=[11,17]),target,
                              dict(path=path,array_path=''))
    widget._refresh_layers_button.click()
    widget._target_combo.setCurrentText('Mask')
    find_all(widget,qtbot)
    widget._component_table.select_component_id(4)
    widget._copy_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy,timeout=15000)
    assert target[2100,2200] == 9, widget._status_label.text()
    assert source[2100,2200] == 9
    assert target[52,152] == 0  # No crop-relative offset applied.
