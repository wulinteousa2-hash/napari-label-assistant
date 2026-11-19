"""Automatic local editing keeps journaled masks authoritative."""

import threading

import numpy as np
import pytest
import zarr
from qtpy.QtWidgets import QComboBox

from napari_label_assistant_tools._camera_follow import FollowWindow, plan_window
from napari_label_assistant_tools._local_editing import LocalEditingControls
from napari_label_assistant_tools._mask_tiles import MaskTileManager, set_mode


def test_review_keeps_crop_on_layout_change_and_follows_camera_with_editor_hidden(editor, qtbot):
    viewer, manager, controls, _ = editor
    controls.auto_follow.setChecked(True)
    ready(controls, qtbot)
    entry = manager.entries[0]
    bounds, pixels = entry.bounds, entry.layer.data.copy()
    controls.active_view = lambda: True  # Labels panel remains visible in QC.
    controls.hide()
    controls.follow.retain_review_area()
    controls.follow.schedule()
    controls.follow.evaluate()
    qtbot.wait(250)
    assert entry.bounds == bounds
    np.testing.assert_array_equal(entry.layer.data, pixels)
    viewer.camera.center = (0, 20000, 20000)
    qtbot.waitUntil(lambda: entry.bounds != bounds, timeout=10000)
    assert entry.bounds[0].start <= 20000 < entry.bounds[0].stop
    assert entry.layer.visible and entry.layer.editable
    assert entry.layer.data.nbytes <= 4096**2 * 4


def test_window_planner():
    shape = (100000, 100000)
    window = plan_window((20000, 30000, 20500, 30600), shape, 1)
    assert window.size == 2048
    assert all(n % 512 == 0 for n in window.origin)
    assert plan_window((20020, 30020, 20520, 30620), shape, 1, current=window) == window
    assert plan_window((0, 0, 100000, 100000), shape, .01) is None
    assert plan_window((-200, -200, -1, -1), shape, 1) is None
    assert plan_window((0, 0, 3000, 3000), shape, 1, bytes_per_pixel=16) is None
    assert plan_window((0, 0, 200, 200), shape, .4) is None
    assert plan_window((100, 100, 200, 200), shape, .4,
                       current=FollowWindow((0, 0), 1024)) is not None


def test_window_planner_honors_selected_size():
    bounds, shape = (10000, 10000, 10500, 10600), (100000, 100000)
    large = plan_window(bounds, shape, 1, requested_size=4096)
    assert large.size == 4096
    small = plan_window(bounds, shape, 1, current=large, requested_size=1024)
    assert small.size == 1024
    assert plan_window(bounds, shape, 1, requested_size=4090).size == 4090


def test_native_canvas_mapping_round_trip(make_napari_viewer):
    from napari_label_assistant_tools._editing_canvas import CanvasAdapter
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((128, 256), dtype='uint8'),
                      scale=(2, 3), translate=(11, 17))
    adapter = CanvasAdapter(viewer)
    assert adapter.supported_view()
    pixels = [(0, 0), (100, 150), (250, 300)]
    world = [adapter.world_at(point)[-2:] for point in pixels]
    np.testing.assert_allclose(adapter.canvas_points(world), pixels, atol=1e-6)


@pytest.fixture
def editor(tmp_path, make_napari_viewer, qtbot, monkeypatch, request):
    viewer = make_napari_viewer()
    manager = MaskTileManager(viewer)
    path = tmp_path / 'mask.zarr'
    extent = getattr(request, 'param', 100000)
    array = zarr.open_array(str(path), mode='w', shape=(extent, extent),
                            chunks=(256, 256), dtype='uint32')
    manager.add({'name': 'mask'}, array, {'path': path, 'array_path': ''})
    set_mode(viewer, 'grid')
    size = QComboBox()
    size.addItem('1024', 1024)
    controls = LocalEditingControls(viewer, size)
    qtbot.addWidget(controls)
    controls.show()
    controls.refresh(manager)
    # Deterministic canvas geometry; exercise real camera events, workers,
    # mask buffers, durable writes, and all follow state transitions.
    adapter = controls.adapter
    def world_at(point):
        center = np.asarray(viewer.camera.center)[-2:]
        offset = np.asarray((point[1] - adapter.native.height()/2,
                             point[0] - adapter.native.width()/2))
        return center + offset / viewer.camera.zoom
    monkeypatch.setattr(adapter, 'world_at', world_at)
    monkeypatch.setattr(adapter, 'canvas_points',
                        lambda points: np.asarray(points)[:, ::-1] * viewer.camera.zoom)
    center = min(10000, extent // 2)
    viewer.camera.center = (0, center, center)
    viewer.camera.zoom = 2
    yield viewer, manager, controls, array
    controls.shutdown()
    manager.dispose()


def ready(controls, qtbot):
    qtbot.waitUntil(lambda: controls.follow.state == 'following', timeout=10000)


def test_follow_uses_custom_size_and_keeps_it_separate_from_navigation_step(editor, qtbot):
    _, manager, controls, _ = editor
    controls._size_control.addItem('4090 px', 4090)
    controls._size_control.setCurrentIndex(controls._size_control.findData(4090))
    controls.auto_follow.setChecked(True)
    qtbot.waitUntil(lambda: controls.follow.window is not None and controls.follow.window.size == 4090,
                    timeout=10000)
    bounds = manager.entries[0].bounds
    assert tuple(s.stop - s.start for s in bounds) == (4090, 4090)
    assert manager.state()['edit_tile_size'] == 4090
    assert manager.size == 512  # Camera-follow navigation stride is independent.


def test_follow_paint_pan_overview_manual_and_save(editor, qtbot):
    viewer, manager, controls, array = editor
    controls.auto_follow.setChecked(True)
    ready(controls, qtbot)
    entry = manager.entries[0]
    first = entry.bounds
    assert first[0].start <= 10000 < first[0].stop
    assert entry.layer.data.nbytes <= 4096**2 * 4
    entry.layer.mode = 'paint'
    entry.layer.data[10, 20] = 17
    entry.layer.events.data()
    manager.stroke_active = True
    viewer.camera.center = (0, 20000, 20000)
    qtbot.wait(400)
    assert entry.bounds == first
    manager.stroke_active = False
    qtbot.waitUntil(lambda: entry.bounds != first, timeout=10000)
    assert array[first[0].start+10, first[1].start+20] == 17
    assert str(entry.layer.mode) == 'paint'
    assert tuple(viewer.camera.center)[-2:] == (20000, 20000)
    viewer.camera.zoom = .001
    qtbot.waitUntil(lambda: controls.follow.state == 'overview', timeout=10000)
    assert entry.bounds is None and not entry.layer.editable
    viewer.camera.zoom = 2
    ready(controls, qtbot)
    controls.manual_edit.click()
    bounds = entry.bounds
    qtbot.wait(400)
    assert controls.follow.state == 'manual' and entry.bounds == bounds
    assert manager.state()['camera_follow'] is True
    viewer.camera.center = (0, 30000, 30000)
    ready(controls, qtbot)
    controls.auto_follow.setChecked(False)
    bounds = entry.bounds
    viewer.camera.center = (0, 40000, 40000)
    qtbot.wait(400)
    assert entry.bounds == bounds


def test_stale_reads_and_read_failure_retain_buffers(editor, qtbot, monkeypatch):
    from napari_label_assistant_tools import _camera_follow
    viewer, manager, controls, array = editor
    original = _camera_follow.read_regions
    gate = threading.Event()
    started = threading.Event()
    def slow(spec):
        started.set()
        gate.wait(5)
        return original(spec)
    monkeypatch.setattr(_camera_follow, 'read_regions', slow)
    controls.auto_follow.setChecked(True)
    qtbot.waitUntil(started.is_set, timeout=5000)
    viewer.camera.center = (0, 30000, 30000)
    entry = manager.entries[0]
    entry.layer.data[5, 6] = 29
    entry.layer.events.data()
    gate.set()
    ready(controls, qtbot)
    assert entry.bounds[0].start <= 30000 < entry.bounds[0].stop
    assert array[5, 6] == 29
    buffer, bounds = entry.layer.data, entry.bounds
    def fail(spec):
        raise OSError('read failed')
    monkeypatch.setattr(_camera_follow, 'read_regions', fail)
    viewer.camera.center = (0, 50000, 50000)
    qtbot.waitUntil(lambda: controls.follow.state == 'error', timeout=10000)
    assert not controls.auto_follow.isChecked()
    assert entry.layer.data is buffer and entry.bounds == bounds


@pytest.mark.parametrize('select_overview', [True, False])
def test_zooming_in_restores_only_selected_overviews(editor, qtbot, tmp_path, select_overview):
    from napari_label_assistant_tools import _label_pyramid as pyramid
    from napari_label_assistant_tools.workspace.service import _remember_layer_storage
    viewer, manager, controls, _ = editor
    root = zarr.open_group(str(tmp_path / 'overview.ome.zarr'), mode='w')
    names = []
    for i, shape in enumerate(pyramid.level_shapes((100000, 100000))):
        root.create_array(f's{i}', shape=shape, chunks=(256, 256), dtype='uint32')
        names.append(f's{i}')
    root.attrs[pyramid.KEY] = dict(method=pyramid.METHOD, complete=True, dirty_regions=[], datasets=names)
    entry = manager.entries[0]
    entry.source.data = root['s0']
    _remember_layer_storage(entry.source, dict(path=tmp_path / 'overview.ome.zarr', array_path='s0'))
    entry.refresh_overview()
    overview = entry.overview
    assert overview is not None
    controls.auto_follow.setChecked(True)
    ready(controls, qtbot)
    entry.layer.mode = 'paint'
    viewer.camera.zoom = .001
    qtbot.waitUntil(lambda: controls.follow.state == 'overview', timeout=10000)
    selected = overview if select_overview else viewer.add_image(np.zeros((16, 16)))
    viewer.layers.selection.active = selected
    viewer.camera.zoom = 2
    ready(controls, qtbot)
    assert entry.layer.editable
    assert viewer.layers.selection.active is (entry.layer if select_overview else selected)
    assert str(entry.layer.mode) == 'paint'
    if select_overview:
        entry.layer.brush_size = 1
        entry.layer.paint((20, 20), 17)
        manager.flush()
        assert root['s0'][entry.bounds[0].start + 20, entry.bounds[1].start + 20] == 17


def test_conflicting_writer_pauses_without_losing_edits(editor, qtbot):
    viewer, manager, controls, array = editor
    controls.auto_follow.setChecked(True)
    ready(controls, qtbot)
    entry = manager.entries[0]
    bounds = entry.bounds
    entry.layer.data[5, 6] = 13
    entry.layer.events.data()
    array[bounds[0].start+5, bounds[1].start+6] = 99
    viewer.camera.center = (0, 40000, 40000)
    qtbot.waitUntil(lambda: controls.follow.state == 'error', timeout=10000)
    assert entry.bounds == bounds
    assert entry.layer.data[5, 6] == 13
    assert array[bounds[0].start+5, bounds[1].start+6] == 99
    assert 'conflict' in controls.status.text().lower()


def test_follow_maps_registered_masks_and_rejects_different_grids(editor):
    viewer, manager, controls, array = editor
    entry = manager.entries[0]
    entry.source.scale = np.array([2., 3.])
    entry.source.translate = np.array([11., 17.])
    entry.source.rotate = np.array([[0., -1.], [1., 0.]])
    points = [(100, 200), (100, 400), (300, 400), (300, 200)]
    world = [entry.source.data_to_world(point) for point in points]
    bounds, shape, basis = controls.backend.follow_view(manager, world, 0)
    np.testing.assert_allclose(bounds, (100, 200, 300, 400))
    manager.add({'name': 'different grid'}, array, entry.reference)
    with pytest.raises(ValueError, match='aligned mask grids'):
        controls.backend.follow_view(manager, world, 0)
    # Restore the original registration for fixture teardown.
    entry.source.scale = np.ones(2)
    entry.source.translate = np.zeros(2)
    entry.source.rotate = np.eye(2)


@pytest.mark.parametrize('editor', [2048], indirect=True)
def test_follow_setting_round_trips_workspace(editor, qtbot, tmp_path):
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer, manager, controls, array = editor
    controls.auto_follow.setChecked(True)
    ready(controls, qtbot)
    entry = manager.entries[0]
    bounds = entry.bounds
    entry.layer.data[5, 6] = 31
    entry.layer.events.data()
    path = tmp_path / 'saved.label-assistant.json'
    save_workspace(viewer, path)
    assert array[bounds[0].start+5, bounds[1].start+6] == 31
    controls.refresh(None)
    load_workspace(viewer, path)
    from napari_label_assistant_tools._mask_tiles import manager_for
    restored = manager_for(viewer)
    controls.refresh(restored)
    assert controls.auto_follow.isChecked() and restored.follow_enabled
    controls.shutdown()
