import numpy as np

from napari_label_assistant_tools._widget import (
    _get_or_create_points_layer,
    component_operations_widget,
)


def test_display_grid_overlay_controls_adaptive_cell_ids(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((1000, 1000), dtype=np.uint8), name="map mask")

    widget = component_operations_widget(viewer)
    widget._target_combo.setCurrentText("map mask")
    widget._grid_y_spin.setValue(100)
    widget._grid_x_spin.setValue(100)
    viewer.camera.center = (0, 500, 500)
    viewer.camera.zoom = 1.0

    widget._display_grid_check.setChecked(True)
    widget._refresh_grid_labels()

    grid = viewer.layers["map mask | grid_overlay"]
    labels = viewer.layers["map mask | grid_labels"]
    assert grid.visible
    assert labels.visible
    assert 0 < len(labels.data) <= 150
    assert labels.features["grid_id"].str.match(r"R\d+C\d+").all()

    # Editing synchronization may remove obsolete editable-area helpers, but
    # must not mistake the grid helpers for stale editing layers.
    widget._dynamic_edit_controller.sync()
    assert "map mask | grid_overlay" in viewer.layers
    assert "map mask | grid_labels" in viewer.layers

    viewer.camera.zoom = 0.1
    widget._refresh_grid_labels()
    assert not labels.visible
    viewer.camera.zoom = 1.0
    widget._refresh_grid_labels()
    assert labels.visible
    assert viewer.layers["map mask | grid_overlay"] is grid
    assert viewer.layers["map mask | grid_labels"] is labels
    widget._display_grid_check.setChecked(False)
    assert not grid.visible
    assert not labels.visible


def test_grid_label_points_and_text_resize_atomically(
    make_napari_viewer, qapp
):
    viewer = make_napari_viewer()
    text_refresh_visibility = []

    for count in (2, 60, 5, 90):
        points = np.column_stack(
            (np.arange(count, dtype=float), np.arange(count, dtype=float))
        )
        layer = _get_or_create_points_layer(
            viewer,
            "grid labels",
            points,
            {"grid_id": [f"R{index}" for index in range(count)]},
            text_field="grid_id",
            text_size=9.0,
            text_scale_with_zoom=False,
        )
        if not text_refresh_visibility:
            layer.text.events.connect(
                lambda _event: text_refresh_visibility.append(layer.visible)
            )
        qapp.processEvents()
        assert len(layer.data) == count
        assert len(layer.features) == count
        assert len(layer._view_text) == count

    # Every renderer-facing text refresh during a resize happens while the
    # layer is hidden, so Vispy cannot draw new indices against old strings.
    assert text_refresh_visibility
    assert not any(text_refresh_visibility)
