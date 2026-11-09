"""Component searches use s0, with explicit bounded-area feedback."""
import numpy as np
import zarr

from napari_label_assistant_tools import _widget as module
from napari_label_assistant_tools._mask_tiles import MaskTileManager
from napari_label_assistant_tools.workspace.service import _persist_labels_to_zarr


def pyramid(tmp_path):
    pixels = np.zeros((129, 135), dtype='uint32')
    pixels[10:14, 10:14] = 7
    pixels[100:103, 110:115] = 9
    path = tmp_path/'mask.ome.zarr'
    _persist_labels_to_zarr(pixels, path, xy_chunk=32, scale=(1,1))
    root = zarr.open_group(str(path), mode='r+')
    # A component search must use s0, regardless of the display resolution.
    for name in ('s1', 's2', 's3', 's4', 's5', 's6'):
        root[name][:] = 0
    return path, [root[f's{i}'] for i in range(7)]


def test_native_ome_pyramid_large_search_reads_s0(tmp_path, make_napari_viewer, qtbot, monkeypatch):
    _, levels = pyramid(tmp_path)
    viewer = make_napari_viewer()
    viewer.add_labels(levels, multiscale=True, name='Actual mask')
    monkeypatch.setattr(module, 'LARGE_ANALYSIS_PIXEL_THRESHOLD', 1)
    widget = module.component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._workflow_tabs.setCurrentIndex(1)
    widget._target_combo.setCurrentText('Actual mask')
    original_array = zarr.Array.__array__
    def no_full_s0(array, *args, **kwargs):
        if array.path == 's0':
            raise AssertionError('Large-mask search must stream s0 slices, not load the full array')
        return original_array(array, *args, **kwargs)
    monkeypatch.setattr(zarr.Array, '__array__', no_full_s0)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: widget._large_controller.worker is None, timeout=15000)
    assert widget._component_table.rowCount() == 2, widget._status_label.text()


def test_native_ome_pyramid_small_search_reads_s0(tmp_path, make_napari_viewer, qtbot):
    _, levels = pyramid(tmp_path)
    viewer = make_napari_viewer()
    viewer.add_labels(levels, multiscale=True, name='Actual mask')
    widget = module.component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._workflow_tabs.setCurrentIndex(1)
    widget._target_combo.setCurrentText('Actual mask')
    widget._analyze_button.click()
    assert widget._component_table.rowCount() == 2, widget._status_label.text()


def test_bounded_ome_search_reports_empty_area_and_finds_objects_elsewhere(tmp_path, make_napari_viewer, qtbot):
    path = tmp_path/'bounded.ome.zarr'
    pixels = np.zeros((1050, 1530),dtype='uint8')
    pixels[520:524, 520:524] = 7
    _persist_labels_to_zarr(pixels,path,xy_chunk=256, scale=(1,1))
    root = zarr.open_group(str(path),mode='r+')
    viewer = make_napari_viewer()
    manager = MaskTileManager(viewer, dict(tile_size=512, camera_follow=False))
    manager.add(dict(name='Actual mask'), root['s0'], dict(path=path,array_path='s0'))
    widget = module.component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._analysis_scope_combo.setCurrentIndex(widget._analysis_scope_combo.findData('area'))
    widget._workflow_tabs.setCurrentIndex(1)
    widget._analyze_button.click()
    assert widget._component_table.rowCount() == 0
    assert 'No components in this edit area' in widget._status_label.text()
    manager.go(1,1)
    widget._analyze_button.click()
    assert widget._component_table.rowCount() == 1
    assert 'current edit area' in widget._status_label.text()


def test_overview_only_mask_can_load_search_area_without_switching_tabs(tmp_path, make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._editing_backend import LocalEditingBackend
    path, levels = pyramid(tmp_path)
    viewer = make_napari_viewer()
    manager = MaskTileManager(viewer, dict(tile_size=512, camera_follow=False))
    truth = manager.add(dict(name='Actual mask'), levels[0], dict(path=path,array_path='s0'))
    widget = module.component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._analysis_scope_combo.setCurrentIndex(widget._analysis_scope_combo.findData('area'))
    widget._workflow_tabs.setCurrentIndex(1)
    LocalEditingBackend().suspend_follow(manager)
    widget._analyze_button.click()
    assert 'No full-resolution edit area loaded' in widget._status_label.text()
    assert widget._analysis_scope_label.isVisibleTo(widget._workflow_tabs.widget(1).widget())
    monkeypatch.setattr(widget._local_editing_controls.adapter, 'world_at', lambda point: (30.,30.))
    widget._load_analysis_area_button.click()
    assert widget._workflow_tabs.currentIndex() == 1
    assert truth._label_assistant_mask_entry.bounds is not None
    assert widget._component_table.rowCount() == 2, widget._status_label.text()
    assert levels[0][10,10] == 7
    assert levels[0][100,110] == 9
