import numpy as np
import pytest
import zarr

from napari_label_assistant_tools import _label_pyramid as pyramid
from napari_label_assistant_tools.workspace.service import _persist_labels_to_zarr


def store(tmp_path, pixels):
    path = tmp_path / 'labels.ome.zarr'
    _persist_labels_to_zarr(pixels, path, xy_chunk=32, scale=(2, 3))
    return dict(path=path, array_path='s0')


def test_seven_categorical_levels_preserve_source_and_alignment(tmp_path):
    pixels = np.zeros((129, 135), np.uint64)
    pixels[1, 1] = 2**40 + 17
    pixels[-1, -1] = 9
    pixels[12:16, 17:21] = 37
    reference = store(tmp_path, pixels)
    assert pyramid.datasets(reference) == [f's{i}' for i in range(7)]
    levels = pyramid.open_levels(reference)
    np.testing.assert_array_equal(levels[0][:], pixels)
    expected = pixels
    for number, level in enumerate(levels):
        assert level.dtype == pixels.dtype
        assert level.shape == expected.shape
        np.testing.assert_array_equal(level[:], expected)
        assert set(np.unique(level[:])) <= {0, 9, 37, 2**40 + 17}
        expected = pyramid.reduce_labels(expected)
    root = zarr.open_group(str(reference['path']), mode='r')
    metadata = root.attrs['ome']['multiscales'][0]['datasets']
    assert [item['path'] for item in metadata] == [f's{i}' for i in range(7)]
    assert metadata[6]['coordinateTransformations'][0]['scale'] == [128, 192]


def test_foreground_ids_are_not_averaged_or_skipped():
    pixels = np.array([[0, 19, 0], [4, 8, 0], [0, 0, 255]], np.uint32)
    np.testing.assert_array_equal(pyramid.reduce_labels(pixels), [[19, 0], [0, 255]])


def test_local_refresh_propagates_paint_and_erase(tmp_path):
    reference = store(tmp_path, np.zeros((129, 135), np.uint32))
    root = zarr.open_group(str(reference['path']), mode='r+')
    assert pyramid.mark_dirty(reference, (101, 109, 102, 110))
    assert not pyramid.datasets(reference)
    root['s0'][101, 109] = 23
    pyramid.build(reference, scale=(2, 3))
    for i in range(7):
        assert root[f's{i}'][101 // 2**i, 109 // 2**i] == 23
    pyramid.mark_dirty(reference, (101, 109, 102, 110))
    root['s0'][101, 109] = 0
    pyramid.build(reference, scale=(2, 3))
    for i in range(7):
        assert not np.any(root[f's{i}'][:])


def test_interrupted_build_is_not_published_and_can_resume(tmp_path):
    path = tmp_path / 'legacy.ome.zarr'
    root = zarr.open_group(str(path), mode='w')
    source = root.create_array('s0', shape=(129, 135), chunks=(32, 32), dtype='uint32')
    source[1, 1] = 18
    reference = dict(path=path, array_path='s0')
    def interrupted(*args):
        raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError, match='interruption'):
        pyramid.build(reference, progress=interrupted)
    assert not pyramid.datasets(reference)
    assert source[1, 1] == 18
    source[1, 1] = 0
    source[-1, -1] = 5
    pyramid.build(reference)
    for i in range(7):
        level = root[f's{i}'][:]
        assert not np.any(level == 18)
        assert level[-1, -1] == 5


def test_build_reads_source_in_bounded_blocks(tmp_path, monkeypatch):
    root = zarr.open_group(str(tmp_path / 'bounded.zarr'), mode='w')
    class Source:
        shape = (257, 259)
        dtype = np.dtype('uint32')
        def __array__(self, *args, **kwargs):
            raise AssertionError('Full source materialized')
        def __getitem__(self, region):
            a, b = region
            assert a.stop - a.start <= 32 and b.stop - b.start <= 32
            output = np.zeros((a.stop-a.start, b.stop-b.start), self.dtype)
            if a.start == b.start == 0:
                output[1, 1] = 7
            return output
    class Root:
        attrs = root.attrs
        create_array = root.create_array
        def __getitem__(self, key):
            return source if key == 's0' else root[key]
        def __contains__(self, key):
            return key == 's0' or key in root
    source = Source()
    monkeypatch.setattr(pyramid, 'BLOCK_SIZE', 16)
    monkeypatch.setattr(pyramid, '_root', lambda *args, **kwargs: Root())
    pyramid.build(dict(path=tmp_path / 'bounded.zarr', array_path='s0'))
    assert root['s6'][0, 0] == 7


@pytest.mark.parametrize('shape,count', [((2, 2), 2), ((1, 1), 1), ((128, 128), 7)])
def test_small_labels_stop_before_duplicate_one_pixel_levels(shape, count):
    assert len(pyramid.level_shapes(shape)) == count


def manifest(tmp_path, reference):
    import json
    from napari_label_assistant_tools.workspace import WORKSPACE_FORMAT
    path = tmp_path / 'case.label-assistant.json'
    path.write_text(json.dumps(dict(format=WORKSPACE_FORMAT, version=2,
        mask_view=dict(mode='grid', tile_size=512, row=0, col=0),
        layers=[dict(layer_type='Labels', name='Curated', visible=True, opacity=.6,
                     scale=[2, 3], translate=[7, 11],
                     storage=dict(kind='zarr', path=str(reference['path']), array_path='s0'))])))
    return path


def test_whole_mask_overview_and_editable_tile_stay_linked(tmp_path, make_napari_viewer, qtbot):
    from types import SimpleNamespace
    from napari_label_assistant_tools.workspace import load_workspace, save_workspace
    from napari_label_assistant_tools._mask_tiles import manager_for, full_data
    from napari_label_assistant_tools._dynamic_edit import is_internal_layer
    import json
    pixels = np.zeros((1030, 1530), np.uint32)
    pixels[700, 900] = 19
    reference = store(tmp_path, pixels)
    path = manifest(tmp_path, reference)
    viewer = make_napari_viewer()
    result = load_workspace(viewer, path)
    assert not result['skipped_layers']
    manager = manager_for(viewer)
    entry = manager.entries[0]
    overview = entry.overview
    assert overview is not None, manager.error
    assert overview.multiscale and not overview.editable
    assert is_internal_layer(overview)
    assert len(overview.data) == 7
    assert full_data(entry.layer).shape == pixels.shape
    assert entry.layer.data.shape == (512, 512)
    assert overview.data[0][700, 900] == 19
    np.testing.assert_allclose(overview.data_to_world((700, 900)), entry.source.data_to_world((700, 900)))
    entry.layer.opacity = .35
    assert overview.opacity == .35
    entry.layer.visible = False
    assert not overview.visible
    entry.layer.visible = True
    assert overview.visible
    viewer.layers.selection.active = entry.layer
    event = SimpleNamespace(position=entry.source.data_to_world((700, 900)))
    manager.locate_from_overview(viewer, event)
    assert (manager.row, manager.col) == (1, 1)
    assert viewer.layers.selection.active is entry.layer
    assert entry.layer.data[188, 388] == 19
    # The overview leaves the displayed crop to the editable layer, preventing
    # double opacity. It does not alter the stored pyramid data.
    assert overview.data[0][slice(700, 701), slice(900, 901)][0, 0] == 0
    assert full_data(entry.layer)[700, 900] == 19
    entry.layer.data[188, 388] = 37
    manager.flush()
    root = zarr.open_group(str(reference['path']), mode='r')
    assert root['s6'][700 // 64, 900 // 64] == 37
    entry.layer.data[188, 388] = 0
    manager.flush()
    assert root['s6'][700 // 64, 900 // 64] == 0
    save_workspace(viewer, path)
    payload = json.loads(path.read_text())
    assert len(payload['layers']) == 1  # Derived display layers are not annotations.
    assert payload['layers'][0]['storage']['datasets'] == [f's{i}' for i in range(7)]
    manager.show_whole_mask()
    viewer.layers.remove(entry.layer)
    # Derived layers leave after napari finishes its deletion snapshot.
    qtbot.waitUntil(lambda: not viewer.layers)


def test_huge_pyramid_load_never_materializes_full_resolution(tmp_path, make_napari_viewer, monkeypatch):
    from napari_label_assistant_tools.workspace import load_workspace
    from napari_label_assistant_tools._mask_tiles import manager_for
    root_path = tmp_path / 'huge.ome.zarr'
    root = zarr.open_group(str(root_path), mode='w')
    names = []
    for i, shape in enumerate(pyramid.level_shapes((100000, 100000))):
        name = f's{i}'
        array = root.create_array(name, shape=shape, chunks=(512, 512), dtype='uint32')
        array[80000 // 2**i, 80000 // 2**i] = 7
        names.append(name)
    root.attrs[pyramid.KEY] = dict(method=pyramid.METHOD, complete=True, dirty_regions=[], datasets=names)
    original = type(root['s0']).__getitem__
    reads = []
    def bounded(array, region):
        if isinstance(region, tuple) and len(region) == 2 and all(isinstance(s, slice) for s in region):
            counts = [len(range(*s.indices(n))) for s, n in zip(region, array.shape)]
            assert np.prod(counts) <= 4 * 1024**2, (array.path, counts)
            reads.append((array.path, counts))
        return original(array, region)
    monkeypatch.setattr(type(root['s0']), '__getitem__', bounded)
    viewer = make_napari_viewer()
    result = load_workspace(viewer, manifest(tmp_path, dict(path=root_path, array_path='s0')))
    assert not result['skipped_layers']
    entry = manager_for(viewer).entries[0]
    assert entry.overview is not None, manager_for(viewer).error
    levels = entry.overview.data if entry.overview.multiscale else [entry.overview.data]
    assert max(levels[0].shape) <= 2048
    assert levels[-1].shape == (1563, 1563)
    assert levels[-1][1250, 1250] == 7
    assert reads


def test_comparison_hides_overview_without_changing_edit_selection(tmp_path, make_napari_viewer, qtbot):
    from napari_label_assistant_tools.workspace import load_workspace
    from napari_label_assistant_tools._mask_tiles import manager_for
    from napari_label_assistant_tools._widget import quick_compare_toggle_widget
    from qtpy.QtWidgets import QPushButton
    reference = store(tmp_path, np.zeros((1030, 1530), np.uint32))
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((1030, 1530), np.uint8), name='image')
    load_workspace(viewer, manifest(tmp_path, reference), clear_existing=False)
    entry = manager_for(viewer).entries[0]
    widget = quick_compare_toggle_widget(viewer)
    qtbot.addWidget(widget)
    viewer.layers.selection.active = entry.layer
    assert widget._mask_combo.count() == 1
    buttons = {button.text(): button for button in widget.findChildren(QPushButton)}
    buttons['Image Only'].click()
    assert not entry.layer.visible and not entry.overview.visible
    buttons['Overlay'].click()
    assert entry.layer.visible and entry.overview.visible
    widget._opacity_spin.setValue(25)
    assert entry.overview.opacity == .25
    assert viewer.layers.selection.active is entry.layer


def test_preview_failure_keeps_annotation_write_and_hides_stale_view(tmp_path, make_napari_viewer, monkeypatch):
    from napari_label_assistant_tools.workspace import load_workspace
    from napari_label_assistant_tools._mask_tiles import manager_for, full_data
    reference = store(tmp_path, np.zeros((1030, 1530), np.uint32))
    viewer = make_napari_viewer()
    load_workspace(viewer, manifest(tmp_path, reference))
    manager = manager_for(viewer)
    entry = manager.entries[0]
    def unavailable(*args, **kwargs):
        raise OSError('simulated overview write failure')
    monkeypatch.setattr(pyramid, 'build', unavailable)
    entry.layer.data[10, 10] = 41
    manager.flush()
    assert full_data(entry.layer)[10, 10] == 41
    assert entry.overview is None
    assert not pyramid.datasets(reference)


def test_concurrent_source_edit_cannot_publish_a_fresh_pyramid(tmp_path):
    reference = store(tmp_path, np.zeros((129, 135), np.uint32))
    def changed():
        raise RuntimeError('source changed while building')
    with pytest.raises(RuntimeError, match='source changed'):
        pyramid.build(reference, force=True, validate_source=changed)
    assert not pyramid.datasets(reference)


@pytest.mark.parametrize('mode', ['auto', 'grid'])
def test_switch_back_from_full_source_uses_bounded_renderer(tmp_path, make_napari_viewer, monkeypatch, mode):
    from napari_label_assistant_tools._mask_tiles import set_mode, manager_for, full_data
    from napari_label_assistant_tools.workspace.service import _remember_layer_storage
    from napari_label_assistant_tools._dynamic_edit import is_internal_layer
    path = tmp_path / 'large.ome.zarr'
    root = zarr.open_group(str(path), mode='w')
    names = []
    for i, shape in enumerate(pyramid.level_shapes((100000, 100000))):
        array = root.create_array(f's{i}', shape=shape, chunks=(512, 512), dtype='uint32')
        names.append(f's{i}')
    root.attrs[pyramid.KEY] = dict(method=pyramid.METHOD, complete=True, dirty_regions=[], datasets=names)
    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.zeros((32, 32), np.uint32), name='source')
    # Simulate the backend retained by Full-layer mode without first rendering
    # its 40 GB source. The transition itself must never read that full array.
    layer._data = root['s0']
    _remember_layer_storage(layer, dict(path=path, array_path='s0'))
    original = type(root['s0']).__getitem__
    source_reads = []
    def bounded(array, key):
        if array.path == 's0' and isinstance(key, tuple) and all(isinstance(s, slice) for s in key):
            shape = tuple(len(range(*s.indices(n))) for s, n in zip(key, array.shape))
            # Native overview rendering can request a viewport slightly larger
            # than the edit tile. It must still remain a bounded viewport.
            assert np.prod(shape) <= 4 * 1024**2
            source_reads.append(shape)
        return original(array, key)
    monkeypatch.setattr(type(root['s0']), '__getitem__', bounded)
    set_mode(viewer, mode)
    manager = manager_for(viewer)
    assert manager is not None
    entry = manager.entries[0]
    assert entry.layer is not layer
    assert entry.layer.data.shape == (1024, 1024)
    assert full_data(entry.layer).shape == (100000, 100000)
    assert entry.overview is not None and is_internal_layer(entry.overview), manager.error
    assert source_reads
    # Re-selecting Auto is a no-op; it must not rebuild the overview or render s0.
    overview = entry.overview
    set_mode(viewer, 'auto')
    assert entry.overview is overview
