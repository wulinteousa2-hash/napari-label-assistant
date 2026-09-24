import numpy as np

from napari_label_assistant_tools._widget import component_operations_widget


def test_grid_settings_reuse_component_analysis(make_napari_viewer):
    viewer = make_napari_viewer()
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:7, 5:7] = 1
    viewer.add_labels(mask, name="working mask")

    widget = component_operations_widget(viewer)
    widget._target_combo.setCurrentText("working mask")
    widget._analyze_button.click()
    component_map = widget._component_table.item(0, 0).text()

    widget._assign_grid_check.setChecked(True)
    widget._grid_y_spin.setValue(4)
    widget._grid_x_spin.setValue(4)

    assert widget._component_table.item(0, 0).text() == component_map
    assert widget._component_table.item(0, 6).text() == "R01C01"
    assert "stale" not in widget._status_label.text().lower()
