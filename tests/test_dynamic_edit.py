from types import SimpleNamespace

import numpy as np
import pytest

from napari_label_assistant_tools._dynamic_edit import (
    EDIT_BOUNDARY_ROLE,
    EDIT_TILE_ROLE,
    INTERNAL_LAYER_ROLE_KEY,
    aligned_tile_bounds,
    automatic_uses_tiled_editing,
    changed_bounding_box,
    resolved_editing_strategy,
)
from napari_label_assistant_tools._widget import component_operations_widget


class _ArrayDescription:
    def __init__(self, shape, dtype=np.uint32):
        self.shape = shape
        self.dtype = np.dtype(dtype)


def test_automatic_strategy_uses_tiles_only_for_large_arrays():
    large = _ArrayDescription((84_175, 79_966))
    small = _ArrayDescription((2048, 2048), np.uint16)

    assert automatic_uses_tiled_editing(
        large, available_bytes=128 * 1024**3
    )
    assert not automatic_uses_tiled_editing(
        small, available_bytes=8 * 1024**3
    )
    assert resolved_editing_strategy("auto", large) == "tiled"
    assert resolved_editing_strategy("direct", large) == "direct"


def test_tile_bounds_are_aligned_and_changed_area_is_bounded():
    bounds = aligned_tile_bounds(
        (84_175, 79_966),
        (42_086, 39_982),
        tile_size=4096,
        alignment=1024,
    )
    assert bounds.shape == (4096, 4096)
    assert bounds.y0 % 1024 == 0
    assert bounds.x0 % 1024 == 0
    assert bounds.contains(42_086, 39_982)

    original = np.zeros((20, 30), dtype=np.uint32)
    current = original.copy()
    current[4:8, 10:17] = 3
    assert changed_bounding_box(current, original) == (
        slice(4, 8),
        slice(10, 17),
    )


def test_tiled_paint_data_writes_back_and_helpers_stay_internal(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    source = viewer.add_labels(
        np.zeros((96, 128), dtype=np.uint32), name="mask"
    )
    widget = component_operations_widget(viewer)
    qtbot.addWidget(widget)

    stale_tile = viewer.add_labels(
        np.ones((12, 12), dtype=np.uint32),
        name="mask — editable area",
    )
    stale_boundary = viewer.add_shapes(
        [np.asarray([[0, 0], [0, 12], [12, 12], [12, 0]])],
        shape_type="rectangle",
        name="mask — editable boundary",
    )
    tiled_index = widget._editing_strategy_combo.findData("tiled")
    widget._editing_strategy_combo.setCurrentIndex(tiled_index)
    controller = widget._dynamic_edit_controller
    qtbot.waitUntil(lambda: controller.tile is not None, timeout=5000)

    assert stale_tile not in viewer.layers
    assert stale_boundary not in viewer.layers
    assert controller.tile.name == "mask — editable area"
    assert controller.boundary.name == "mask — editable boundary"
    assert controller.tile.metadata[INTERNAL_LAYER_ROLE_KEY] == EDIT_TILE_ROLE
    assert (
        controller.boundary.metadata[INTERNAL_LAYER_ROLE_KEY]
        == EDIT_BOUNDARY_ROLE
    )
    assert widget._target_combo.findText(controller.tile.name) == -1
    assert source.editable is False

    local_updates = []
    source.events.labels_update.connect(local_updates.append)
    controller.tile.brush_size = 1
    controller.tile.paint((6, 10), 7)
    assert controller.dirty
    qtbot.waitUntil(
        lambda: int(source.data[6, 10]) == 7 and not controller.dirty,
        timeout=3000,
    )
    assert int(source.data[6, 10]) == 7
    assert len(local_updates) == 1
    assert tuple(local_updates[0].offset) == (6, 10)
    assert tuple(local_updates[0].data.shape[:2]) == (1, 1)
    assert "Saved to source layer" in widget._dynamic_edit_status.text()
    assert "local preview" in widget._dynamic_edit_status.text()

    assert widget._auto_save_edit_check.isChecked()
    widget._auto_save_edit_check.setChecked(False)
    controller.tile.paint((7, 13), 4)
    qtbot.wait(550)
    assert int(source.data[7, 13]) == 0
    assert controller.dirty

    widget._save_edit_button.click()
    qtbot.waitUntil(
        lambda: int(source.data[7, 13]) == 4 and not controller.dirty,
        timeout=3000,
    )
    assert "Saved to source layer" in widget._dynamic_edit_status.text()
    widget._auto_save_edit_check.setChecked(True)

    controller.tile.mode = "paint"
    mouse_event = SimpleNamespace(type="mouse_press")
    stroke = controller._track_paint_stroke(controller.tile, mouse_event)
    next(stroke)
    controller.tile.paint((14, 20), 8)
    assert controller.dirty_bounds == (14, 20, 15, 21)
    qtbot.wait(550)
    assert int(source.data[14, 20]) == 0
    assert controller.dirty
    mouse_event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(stroke)
    qtbot.waitUntil(
        lambda: int(source.data[14, 20]) == 8 and not controller.dirty,
        timeout=3000,
    )

    direct_index = widget._editing_strategy_combo.findData("direct")
    widget._editing_strategy_combo.setCurrentIndex(direct_index)
    assert controller.tile is None
    assert controller.boundary is None
    assert source.editable is True


def test_native_brush_stroke_persists_to_zarr_source(
    make_napari_viewer, qtbot, tmp_path
):
    zarr = __import__("zarr")
    path = tmp_path / "mask.zarr"
    stored = zarr.open_array(
        str(path),
        mode="w",
        shape=(96, 128),
        chunks=(32, 32),
        dtype=np.uint32,
    )
    viewer = make_napari_viewer()
    viewer.add_labels(stored, name="disk mask")
    widget = component_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._editing_strategy_combo.setCurrentIndex(
        widget._editing_strategy_combo.findData("tiled")
    )
    controller = widget._dynamic_edit_controller
    qtbot.waitUntil(lambda: controller.tile is not None, timeout=5000)

    controller.tile.brush_size = 1
    controller.tile.paint((11, 17), 9)
    qtbot.waitUntil(
        lambda: int(stored[11, 17]) == 9 and not controller.dirty,
        timeout=3000,
    )

    reopened = zarr.open_array(str(path), mode="r")
    assert int(reopened[11, 17]) == 9
