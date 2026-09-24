from types import SimpleNamespace

import numpy as np

import napari_label_assistant_tools._widget as widget_module


def test_large_component_table_pages_thousands_of_objects(
    make_napari_viewer, monkeypatch, qtbot
):
    monkeypatch.setattr(widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 50)
    viewer = make_napari_viewer()
    y, x = np.indices((48, 48))
    mask = ((y + x) % 2 == 0).astype(np.uint8)
    layer = viewer.add_labels(mask, name="many objects")

    widget = widget_module.component_operations_widget(viewer)
    widget._target_combo.setCurrentText("many objects")
    widget._analyze_button.click()
    qtbot.waitUntil(
        lambda: widget._large_controller.worker is None
        and widget._page_spin.maximum() == 2,
        timeout=15000,
    )

    assert widget._component_table.rowCount() == 1000
    widget._page_spin.setValue(2)
    assert widget._component_table.rowCount() == 152
    shown_ids = {
        int(widget._component_table.item(row, 0).text())
        for row in range(widget._component_table.rowCount())
    }
    assert shown_ids == set(range(1001, 1153))

    widget._click_select_check.setChecked(True)
    widget._append_click_select_check.setChecked(True)
    callback = next(
        callback
        for callback in viewer.mouse_drag_callbacks
        if getattr(callback, "__name__", "") == "_handle_mouse_click"
    )
    first_position = tuple(np.argwhere(mask)[0])
    later_position = tuple(np.argwhere(mask)[1000])
    callback(
        viewer,
        SimpleNamespace(
            type="mouse_press",
            position=layer.data_to_world(first_position),
        ),
    )
    callback(
        viewer,
        SimpleNamespace(
            type="mouse_press",
            position=layer.data_to_world(later_position),
        ),
    )

    assert (
        widget._selection_summary.text()
        == "Selected components: 2 | IDs: 1, 1001"
    )
    assert widget._component_table.selected_component_ids() == [1001]
    widget._page_spin.setValue(1)
    assert widget._component_table.selected_component_ids() == [1]
