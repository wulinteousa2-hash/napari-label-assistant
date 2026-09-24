import numpy as np

import napari_label_assistant_tools._widget as widget_module


def test_large_mask_routes_to_worker_and_keeps_grid_edits_cheap(
    make_napari_viewer, monkeypatch, qtbot
):
    monkeypatch.setattr(widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 50)
    viewer = make_napari_viewer()
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    mask[4:8, 4:8] = 0
    source = viewer.add_labels(mask, name="large mask")
    target = viewer.add_labels(np.zeros_like(mask), name="target")

    widget = widget_module.component_operations_widget(viewer)
    widget._target_combo.setCurrentText("large mask")
    widget._analyze_button.click()
    assert not source.editable
    qtbot.waitUntil(
        lambda: widget._large_controller.worker is None
        and widget._component_table.rowCount() == 1,
        timeout=15000,
    )

    assert source.editable
    assert widget._component_table.item(0, 3).text() == "0"
    assert "Tile seams were joined" in widget._status_label.text()

    widget._assign_grid_check.setChecked(True)
    widget._grid_y_spin.setValue(4)
    widget._grid_x_spin.setValue(4)
    assert widget._component_table.item(0, 6).text() == "R01C01"
    assert widget._large_controller.worker is None

    widget._component_table.select_component_id(1)
    widget._copy_target_combo.setCurrentText("target")
    widget._copy_button.click()
    qtbot.waitUntil(
        lambda: not widget._large_controller.edit_active, timeout=15000
    )
    assert np.array_equal(target.data, mask)

    widget._component_table.select_component_id(1)
    widget._delete_button.click()
    qtbot.waitUntil(
        lambda: not widget._large_controller.edit_active, timeout=15000
    )
    assert not np.any(viewer.layers["large mask"].data)
    assert widget._component_table.rowCount() == 0
