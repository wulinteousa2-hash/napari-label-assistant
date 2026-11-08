"""Spatial Assistant's overview policy, including native napari construction."""
import numpy as np
import pytest

from napari_label_assistant_tools._label_overview import (
    POLICY_KEY,
    BoundedLabelsViewer,
    LabelOverviewGuard,
    bounded_overview_arguments,
    plan_overview,
)


class LazyLevel:
    def __init__(self, shape, dtype='uint32'):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = 2
        self.size = int(np.prod(shape))
        self.reads = []

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:, :], dtype=dtype)

    def __getitem__(self, region):
        if region is Ellipsis:
            region = (slice(None), slice(None))
        if not isinstance(region, tuple):
            region = (region,)
        region += (slice(None),) * (2-len(region))
        for key, length in zip(region, self.shape):
            if isinstance(key, int) and not -length <= key < length:
                raise IndexError(key)
        shape = tuple(len(range(*s.indices(n))) for s, n in zip(region, self.shape)
                      if isinstance(s, slice))
        self.reads.append(shape)
        assert int(np.prod(shape)) * self.dtype.itemsize <= 64 * 1024**2
        assert max(shape) <= 4096
        return np.zeros(shape, self.dtype)


def pyramid(shape=(80000, 80000), dtype='uint32'):
    return [LazyLevel(tuple((n + 2**i - 1) // 2**i for n in shape), dtype)
            for i in range(7)]


ROLE = {'label_assistant_internal_role': 'label_pyramid_overview'}


@pytest.mark.parametrize('limit,index', [(32768, 5), (4096, 5), (2048, 6)])
def test_planner_matches_spatial_without_reading_pixels(limit, index):
    levels = pyramid()
    plan = plan_overview(levels, limit)
    assert plan.first_level == index
    assert plan.levels[0] is levels[index]
    assert plan.source_shape == (80000, 80000)
    assert plan.factor == (2**index, 2**index)
    assert not any(level.reads for level in levels)


def test_byte_limit_and_no_safe_preview():
    levels = [LazyLevel((4096, 4096), 'uint64'), LazyLevel((2048, 2048), 'uint64')]
    assert plan_overview(levels, 32768).first_level == 1
    with pytest.raises(ValueError, match='No texture-safe'):
        plan_overview(pyramid()[:2], 4096)


@pytest.mark.parametrize('limit,multiscale', [(4096, True), (2048, False)])
def test_constructor_zoom_and_resolution_lock_cannot_read_full_mask(make_napari_viewer, limit, multiscale):
    viewer = make_napari_viewer()
    levels = pyramid()
    layer = BoundedLabelsViewer(viewer, limit).add_labels(
        levels, multiscale=True, metadata=ROLE, scale=(2, 3), translate=(11, 17),
        rotate=27, shear=(.2,), affine=np.array([[1, .1, 7], [0, 1, 13], [0, 0, 1]]),
    )
    assert layer.multiscale is multiscale
    layer.editable = False
    from napari.utils.transforms import Affine
    physical = Affine(scale=(2, 3), translate=(11, 17), rotate=27, shear=(.2,))
    extra = Affine(affine_matrix=np.array([[1, .1, 7], [0, 1, 13], [0, 0, 1]]))
    factor = np.asarray(layer.metadata[POLICY_KEY]['factor'])
    for point in ((0, 0), (10, 20), (1000, 1000)):
        np.testing.assert_allclose(layer.data_to_world(point), extra(physical(np.asarray(point)*factor)))
    layer.locked_data_level = 0
    for zoom in (.01, .25, 2, .015):
        viewer.scene.camera.zoom = zoom
        layer.refresh()
    assert not any(level.reads for level in levels[:5])
    assert any(level.reads for level in levels[5:])
    assert not layer.editable


def test_edit_crops_and_unrelated_layers_are_untouched():
    levels = pyramid()
    for kwargs in ({'multiscale': True, 'metadata': {'other_plugin': 'mask'}},
                   {'metadata': ROLE}):
        data, options = bounded_overview_arguments(levels, kwargs, 4096)
        assert data is levels and options is kwargs


def test_legacy_overview_repair_and_cleanup(make_napari_viewer):
    viewer = make_napari_viewer()
    levels = pyramid()
    layer = viewer.add_labels(levels, multiscale=True, metadata=ROLE,
                              scale=(2, 3), translate=(7, 11))
    guard = LabelOverviewGuard(viewer, 2048)
    assert layer.data[0] is levels[6]
    np.testing.assert_allclose(layer.data_to_world((10, 20)),
                               np.asarray((10, 20))*64*(2, 3)+(7, 11))
    layer.locked_data_level = 0
    assert not any(level.reads for level in levels[:6])
    guard.close()
    guard.close()
    assert not guard._layers


def test_manager_bounds_before_constructor_and_after_save(tmp_path, make_napari_viewer, monkeypatch):
    import zarr

    from napari_label_assistant_tools import _label_overview, _label_pyramid
    from napari_label_assistant_tools._mask_tiles import MaskTileManager
    from napari_label_assistant_tools.workspace import save_workspace
    monkeypatch.setattr(_label_overview, 'texture_limit', lambda: 4096)
    path = tmp_path / 'mask.ome.zarr'
    root = zarr.open_group(str(path), mode='w')
    names = []
    for i, shape in enumerate(_label_pyramid.level_shapes((80000, 80000))):
        root.create_array(f's{i}', shape=shape, chunks=(512, 512), dtype='uint32')
        names.append(f's{i}')
    root.attrs[_label_pyramid.KEY] = {'method': _label_pyramid.METHOD, 'complete': True,
                                        'dirty_regions': [], 'datasets': names}
    viewer = make_napari_viewer()
    original_add = type(viewer).add_labels
    previews = []
    def bounded_add(self, data, **kwargs):
        if kwargs.get('multiscale'):
            previews.append([level.shape for level in data])
            assert max(data[0].shape) <= 4096
        return original_add(self, data, **kwargs)
    monkeypatch.setattr(type(viewer), 'add_labels', bounded_add)
    manager = MaskTileManager(viewer)
    manager.add({'name': 'mask'}, root['s0'], {'path': path, 'array_path': 's0'})
    entry = manager.entries[0]
    assert entry.overview is not None, manager.error
    assert entry.layer.data.shape == (1024, 1024)
    assert entry.source.data.shape == (80000, 80000)
    entry.layer.data[10, 20] = 17
    manager.go(1, 1)
    assert root['s0'][10, 20] == 17
    save_workspace(viewer, tmp_path / 'saved.label-assistant.json')
    assert len(previews) >= 2
    assert all(shapes == [(2500, 2500), (1250, 1250)] for shapes in previews)
    # Rebasing the display leaves crop exclusion in global source coordinates.
    entry.bounds = (slice(32000, 33024), slice(32000, 33024))
    root['s5'][1010, 1010] = 19
    assert entry.overview.data[0][slice(1010, 1011), slice(1010, 1011)][0, 0] == 0
    entry.bounds = None
    assert entry.overview.data[0][slice(1010, 1011), slice(1010, 1011)][0, 0] == 19


def test_preview_limit_error_does_not_block_mask_save(tmp_path, make_napari_viewer, monkeypatch):
    from napari_label_assistant_tools import _label_pyramid
    from napari_label_assistant_tools._mask_tiles import MaskTileManager
    from napari_label_assistant_tools.workspace import save_workspace
    from napari_label_assistant_tools.workspace.service import _persist_labels_to_zarr

    path = tmp_path / 'mask.ome.zarr'
    array = _persist_labels_to_zarr(np.zeros((128, 128), np.uint32), path,
                                    xy_chunk=32, scale=(1, 1))
    viewer = make_napari_viewer()
    manager = MaskTileManager(viewer)
    manager.add({'name': 'mask'}, array, {'path': path, 'array_path': 's0'})
    entry = manager.entries[0]
    entry.layer.data[5, 6] = 23
    monkeypatch.setattr(_label_pyramid, 'open_levels', lambda ref: pyramid()[:2])
    result = save_workspace(viewer, tmp_path / 'saved.label-assistant.json')
    assert result['saved_layers'] == 1
    assert array[5, 6] == 23 and entry.layer.editable
    assert entry.overview is None
    assert 'No texture-safe' in manager.error
