import numpy as np

from napari_label_assistant_tools._widget import component_operations_widget


def test_grid_ids_and_visible_overlay_are_independent(make_napari_viewer):
    viewer = make_napari_viewer()
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    viewer.add_labels(
        mask,
        name="working mask",
        scale=(2.0, 3.0),
        translate=(4.0, 5.0),
    )

    widget = component_operations_widget(viewer)
    widget._target_combo.setCurrentText("working mask")
    widget._grid_y_spin.setValue(5)
    widget._grid_x_spin.setValue(4)

    widget._assign_grid_check.setChecked(True)
    assert "working mask | grid_overlay" not in viewer.layers

    widget._analyze_button.click()
    assert widget._component_table.item(0, 6).text() == "R00C00"

    widget._display_grid_check.setChecked(True)
    grid = viewer.layers["working mask | grid_overlay"]
    assert len(grid.data) == 7
    assert np.array_equal(grid.scale, np.array([2.0, 3.0]))
    assert np.array_equal(grid.translate, np.array([4.0, 5.0]))

    widget._display_grid_check.setChecked(False)
    assert not grid.visible
    assert widget._component_table.rowCount() == 1


def test_dense_grid_display_is_rejected_before_rendering(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((1200, 1200), dtype=np.uint8), name="large mask")

    widget = component_operations_widget(viewer)
    widget._target_combo.setCurrentText("large mask")
    widget._grid_y_spin.setValue(1)
    widget._grid_x_spin.setValue(1)
    widget._display_grid_check.setChecked(True)

    assert not widget._display_grid_check.isChecked()
    assert "may be slow" in widget._status_label.text()
    assert "large mask | grid_overlay" not in viewer.layers
