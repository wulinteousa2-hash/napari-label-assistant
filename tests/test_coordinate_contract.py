"""Scientific pixel/world registration across persistence, editing and copying."""
import json

import numpy as np
import pytest

from napari_label_assistant_tools._coordinates import (
    crop_transform_kwargs, matching_mask_transforms,
)
from napari_label_assistant_tools._quick_train import QuickTrainPrediction
from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
from napari_label_assistant_tools._widget import component_operations_widget, label_operations_widget
from napari_label_assistant_tools.workspace import save_workspace, load_workspace


TRANSFORMS = [
    {},
    dict(scale=(2, 3), translate=(7, 11)),
    dict(rotate=25),
    dict(shear=(.2,)),
    dict(affine=np.array([[1, .15, 8], [-.2, 1, -6], [0, 0, 1]])),
    dict(scale=(2, 3), translate=(7, 11), rotate=25, shear=(.2,),
         affine=np.array([[1.1, .15, 8], [-.2, .9, -6], [0, 0, 1]])),
    dict(scale=(-2, 3), translate=(7, 11), rotate=-30, shear=(-.3,)),
]


def world(layer, points):
    return np.array([layer.data_to_world(point) for point in points])


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_A01_layers_and_cropped_prediction_roundtrip(transform, make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    image_pixels = np.arange(32*47, dtype=np.uint16).reshape(32, 47)
    label_pixels = np.zeros((32, 47), np.uint64)
    label_pixels[0:4, 2:6] = 2**53+1
    image = viewer.add_image(image_pixels.copy(), **transform)
    viewer.add_labels(label_pixels.copy(), **transform)
    widget = QuickTrainWidget(viewer); qtbot.addWidget(widget)
    origin = np.array([9, 13])
    widget._prediction_done(image, .5, QuickTrainPrediction(np.ones((5, 7), np.float32), tuple(origin)))
    points = np.array([(0, 0), (1, 0), (0, 1), (4, 6)])
    np.testing.assert_allclose(world(viewer.layers[-1], points), world(image, points+origin), atol=1e-10)
    expected_world = [world(layer, points) for layer in viewer.layers]
    expected_pixels = [layer.data.copy() for layer in viewer.layers]
    path = tmp_path/'coordinates.label-assistant.json'
    for _ in range(2):
        save_workspace(viewer, path, complete_package=True)
        viewer = make_napari_viewer()
        result = load_workspace(viewer, path)
        assert not result['skipped_layers']
        for layer, pixels, coordinates in zip(viewer.layers, expected_pixels, expected_world):
            np.testing.assert_array_equal(layer.data[:], pixels)
            np.testing.assert_allclose(world(layer, points), coordinates, rtol=0, atol=1e-10)


def test_legacy_scale_translation_workspace_still_loads(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.ones((13, 17), np.uint8), scale=(2, 3), translate=(7, 11))
    expected = world(layer, [(0, 0), (12, 16)])
    path = tmp_path/'legacy.label-assistant.json'
    save_workspace(viewer, path)
    payload = json.loads(path.read_text())
    for record in payload['layers']:
        for key in ('rotate', 'shear', 'affine'):
            record.pop(key, None)
    path.write_text(json.dumps(payload))
    restored = make_napari_viewer()
    assert not load_workspace(restored, path)['skipped_layers']
    np.testing.assert_allclose(world(restored.layers[0], [(0, 0), (12, 16)]), expected)


def test_three_dimensional_workspace_transform_is_not_truncated(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    affine = np.eye(4); affine[0, 1] = .15; affine[:3, 3] = (11, 13, 17)
    source = viewer.add_labels(np.arange(3*11*15, dtype=np.uint16).reshape(3, 11, 15),
        scale=(1, 2, 3), translate=(4, 5, 6), rotate=(10, 20, 30), shear=(.1, .2, .3), affine=affine)
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 10, 14)]
    expected = world(source, points)
    path = tmp_path/'3d.label-assistant.json'
    save_workspace(viewer, path)
    restored = make_napari_viewer()
    assert not load_workspace(restored, path)['skipped_layers']
    np.testing.assert_allclose(world(restored.layers[0], points), expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(restored.layers[0].data[:], source.data[:])


@pytest.mark.parametrize('attribute', ['scale', 'translate', 'rotate', 'shear', 'affine'])
def test_invalid_saved_coordinates_do_not_leave_default_position_layer(attribute, make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    viewer.add_labels(np.ones((13, 17), np.uint8), **TRANSFORMS[5])
    path = tmp_path/'invalid.label-assistant.json'
    save_workspace(viewer, path)
    payload = json.loads(path.read_text())
    value = np.asarray(payload['layers'][0][attribute], dtype=float)
    value.flat[0] = np.nan
    payload['layers'][0][attribute] = value.tolist()
    path.write_text(json.dumps(payload))
    restored = make_napari_viewer()
    result = load_workspace(restored, path)
    assert not result['restored_layers']
    assert len(result['skipped_layers']) == 1
    assert f'Invalid {attribute} transform' in result['skipped_layers'][0]['reason']
    assert not len(restored.layers)


def test_A02_local_crop_paint_boundary_and_live_transform_changes(make_napari_viewer, qtbot, monkeypatch):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.zeros((96, 128), np.uint32), **TRANSFORMS[5])
    widget = component_operations_widget(viewer); qtbot.addWidget(widget)
    controller = widget._dynamic_edit_controller
    monkeypatch.setattr(controller, '_center_in_data', lambda _: (40, 55))
    controller.tile_size.addItem('32 px', 32)
    controller.tile_size.setCurrentIndex(controller.tile_size.findData(32))
    widget._editing_strategy_combo.setCurrentIndex(widget._editing_strategy_combo.findData('tiled'))
    qtbot.waitUntil(lambda: controller.tile is not None and controller.bounds.shape == (32, 32), timeout=10000)
    bounds = controller.bounds
    assert bounds.y0 > 0 and bounds.x0 > 0
    points = np.array([(0, 0), (1, 0), (0, 1), (31, 31)])

    def check_geometry():
        np.testing.assert_allclose(world(controller.tile, points),
                                   world(source, points + [bounds.y0, bounds.x0]), atol=1e-10)
        boundary_points = np.asarray(controller.boundary.data[0])
        np.testing.assert_allclose(world(controller.boundary, boundary_points),
                                   world(source, boundary_points), atol=1e-10)
    check_geometry()
    controller.tile.brush_size = 1
    controller.tile.paint((4, 6), 17)
    controller.commit()
    expected = np.zeros(source.data.shape, source.data.dtype)
    expected[bounds.y0+4, bounds.x0+6] = 17
    np.testing.assert_array_equal(source.data, expected)
    source.rotate = 41
    source.shear = (.4,)
    source.translate = (19, -8)
    source.affine = np.array([[.9, .2, -5], [.1, 1.2, 16], [0, 0, 1]])
    check_geometry()
    controller.tile.paint((4, 6), 0)
    controller.commit()
    assert not source.data.any()
    controller.close()
    source.rotate = 10  # Disconnected helper transform listeners must be safe.


@pytest.mark.parametrize('tiled', [False, True])
def test_A04_copy_filters_misalignment_and_manual_matching_preserves_original(
        tiled, make_napari_viewer, qtbot, monkeypatch):
    monkeypatch.setattr('napari_label_assistant_tools._copy_destination.confirm_manual_destination',lambda *args:True)
    import napari_label_assistant_tools._widget as module
    if tiled:
        monkeypatch.setattr(module, 'LARGE_ANALYSIS_PIXEL_THRESHOLD', 1)
    viewer = make_napari_viewer()
    pixels = np.zeros((16, 20), np.uint16); pixels[2:5, 4:8] = 7
    source = viewer.add_labels(pixels.copy(), name='candidate', **TRANSFORMS[5])
    target = viewer.add_labels(np.zeros_like(pixels), name='misaligned', **TRANSFORMS[5])
    target.translate = target.translate + (1, 0)
    aligned = viewer.add_labels(np.zeros_like(pixels), name='aligned', **TRANSFORMS[5])
    widget = component_operations_widget(viewer); qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: widget._component_table.rowCount() > 0 and not widget._large_controller.busy, timeout=15000)
    widget._component_table.select_component_id(1)
    assert widget._copy_target_combo.findData(target) == -1
    assert widget._copy_target_combo.findData(aligned) >= 0
    widget._manual_copy_target_check.setChecked(True)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(target))
    widget._copy_button.click()
    assert not target.data.any()
    qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=10000)
    matched=viewer.layers['misaligned']
    np.testing.assert_array_equal(matched.data,pixels)
    assert target.name.startswith('misaligned — original')
    np.testing.assert_allclose(world(matched,[(0,0),(15,19)]),world(source,[(0,0),(15,19)]),atol=1e-10)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(aligned))
    widget._copy_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy, timeout=10000)
    np.testing.assert_array_equal(aligned.data, pixels)
    widget._copy_target_combo.setEditText('new curated')
    widget._copy_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy, timeout=10000)
    created = viewer.layers['new curated']
    np.testing.assert_array_equal(created.data, pixels)
    np.testing.assert_allclose(world(created, [(0, 0), (1, 0), (0, 1), (15, 19)]),
                               world(source, [(0, 0), (1, 0), (0, 1), (15, 19)]), atol=1e-10)


@pytest.mark.parametrize('scale,origin', [(1, 1e9), (1e-9, 1e-3)])
def test_registration_checks_pixel_error_not_relative_world_origin(scale, origin, make_napari_viewer):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.zeros((11, 15), np.uint8), scale=(scale, scale), translate=(origin, origin))
    target = viewer.add_labels(np.zeros((11, 15), np.uint8), scale=(scale, scale), translate=(origin, origin))
    assert matching_mask_transforms(source, target)
    target.translate = target.translate + (scale, 0)
    assert not matching_mask_transforms(source, target)


@pytest.mark.parametrize('attribute,value', [
    ('scale', (2, 4)), ('rotate', 31), ('shear', (.5,)),
    ('affine', np.array([[1, .2, 0], [0, 1, 0], [0, 0, 1]])),
])
def test_registration_detects_each_transform_difference(attribute, value, make_napari_viewer):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.zeros((16, 20), np.uint8), **TRANSFORMS[5])
    target = viewer.add_labels(np.zeros((16, 20), np.uint8), **TRANSFORMS[5])
    assert matching_mask_transforms(source, target)
    setattr(target, attribute, value)
    assert not matching_mask_transforms(source, target)


def test_tiled_controller_rejects_misalignment_before_mutation(make_napari_viewer, qtbot):
    from qtpy.QtWidgets import QPushButton, QProgressBar
    from napari_label_assistant_tools._large_components import build_large_component_index
    from napari_label_assistant_tools._large_component_controller import LargeComponentController
    viewer = make_napari_viewer()
    pixels = np.zeros((8, 9), np.uint16); pixels[1:4, 2:6] = 7
    source = viewer.add_labels(pixels)
    target = viewer.add_labels(np.zeros_like(pixels), translate=(1, 0))
    task = build_large_component_index(pixels, tile_size=4)
    while True:
        try:
            next(task)
        except StopIteration as result:
            index = result.value
            break
    controller = LargeComponentController(progress=QProgressBar(), cancel_button=QPushButton(),
        analyze_button=QPushButton(), status=lambda _: None, on_ready=lambda *_: None)
    try:
        with pytest.raises(ValueError, match='matching spatial transforms'):
            controller.edit_components(index=index, source_layer=source, target_layer=target,
                component_ids=list(index.records), delete=False, on_done=lambda _: None)
        assert not target.data.any() and not controller.busy
    finally:
        controller.close(); index.close()


def test_equivalent_decompositions_and_singleton_masks(make_napari_viewer):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.zeros((16, 20), np.uint8), **TRANSFORMS[5])
    zero = world(source, [(0, 0)])[0]
    affine = np.eye(3)
    affine[:2, :2] = (world(source, [(1, 0), (0, 1)]) - zero).T
    affine[:2, 2] = zero
    target = viewer.add_labels(np.zeros((16, 20), np.uint8), affine=affine)
    assert matching_mask_transforms(source, target)
    singleton = viewer.add_labels(np.zeros((1, 16, 20), np.uint8), scale=(1, 2, 3), translate=(0, 7, 11))
    projected = viewer.add_labels(np.zeros((16, 20), np.uint8), **crop_transform_kwargs(singleton))
    assert matching_mask_transforms(singleton, projected)
    singleton.translate = (4, 7, 11)
    with pytest.raises(ValueError, match='leaves the 2D plane'):
        crop_transform_kwargs(singleton)
    wrong_plane = viewer.add_labels(np.zeros((16, 20, 1), np.uint8))
    with pytest.raises(ValueError, match='leaves the 2D plane'):
        crop_transform_kwargs(wrong_plane)


def test_crop_at_large_origin_does_not_lose_linear_precision(make_napari_viewer):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.zeros((11, 15), np.uint8),
                               scale=(2, 3), translate=(1e9, -1e9), rotate=25, shear=(.2,))
    origin = np.array([90000, 80000])
    crop = viewer.add_labels(np.zeros((11, 15), np.uint8), **crop_transform_kwargs(source, origin))
    points = np.array([(0, 0), (1, 0), (0, 1), (10, 14)])
    # Floating point evaluation at this world magnitude has ~1e-7 resolution.
    np.testing.assert_allclose(world(crop, points), world(source, points+origin), rtol=0, atol=5e-7)


def test_combine_checks_registration_and_preserves_complete_transform(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    pixels = np.zeros((16, 20), np.uint8); pixels[2:5, 4:8] = 1
    source = viewer.add_labels(pixels, name='source', **TRANSFORMS[5])
    other = viewer.add_labels(pixels.copy(), name='other', **TRANSFORMS[5])
    widget = label_operations_widget(viewer); qtbot.addWidget(widget)
    widget._mode_combo.setCurrentText('Merge Layers As Same Class')
    for i in range(widget._layer_list.count()):
        widget._layer_list.item(i).setSelected(True)
    widget._apply_button.click()
    created = viewer.layers[-1]
    np.testing.assert_allclose(world(created, [(0, 0), (15, 19)]), world(source, [(0, 0), (15, 19)]), atol=1e-10)
    other.translate = other.translate + (1, 0)
    for i in range(widget._layer_list.count()):
        item = widget._layer_list.item(i)
        item.setSelected(item.text() in {'source', 'other'})
    count = len(viewer.layers)
    widget._apply_button.click()
    assert len(viewer.layers) == count
