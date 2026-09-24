from types import SimpleNamespace

import numpy as np

from napari_label_assistant_tools._widget import component_operations_widget


def test_canvas_component_pick_uses_layer_data_coordinates(make_napari_viewer):
    viewer = make_napari_viewer()
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:6, 4:7] = 1
    layer = viewer.add_labels(
        mask, name="transformed mask", scale=(2.0, 3.0), translate=(4.0, 5.0)
    )
    widget = component_operations_widget(viewer)
    widget._target_combo.setCurrentText("transformed mask")
    widget._analyze_button.click()

    click_checkbox = widget._click_select_check
    click_checkbox.setChecked(True)
    callback = next(
        callback
        for callback in viewer.mouse_drag_callbacks
        if getattr(callback, "__name__", "") == "_handle_mouse_click"
    )
    viewer.add_points([(0, 0)], name="active overlay")
    event = SimpleNamespace(
        type="mouse_press", position=layer.data_to_world((4, 5))
    )
    callback(viewer, event)

    assert widget._component_table.selected_component_ids() == [1]


def test_canvas_component_pick_can_append_selections(make_napari_viewer):
    viewer = make_napari_viewer()
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[7:9, 7:9] = 1
    layer = viewer.add_labels(mask, name="mask")
    widget = component_operations_widget(viewer)
    widget._target_combo.setCurrentText("mask")
    widget._analyze_button.click()
    widget._click_select_check.setChecked(True)
    widget._append_click_select_check.setChecked(True)
    callback = next(
        callback
        for callback in viewer.mouse_drag_callbacks
        if getattr(callback, "__name__", "") == "_handle_mouse_click"
    )

    callback(
        viewer,
        SimpleNamespace(
            type="mouse_press", position=layer.data_to_world((1, 1))
        ),
    )
    callback(
        viewer,
        SimpleNamespace(
            type="mouse_press", position=layer.data_to_world((7, 7))
        ),
    )

    assert sorted(widget._component_table.selected_component_ids()) == [1, 2]
    assert (
        widget._selection_summary.text()
        == "Selected components: 2 | IDs: 1, 2"
    )
