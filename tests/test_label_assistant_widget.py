import numpy as np
from qtpy.QtWidgets import QPushButton

from napari_label_assistant_tools import label_assistant_widget


def test_label_assistant_widget_contains_all_tabs(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), dtype=np.uint8), name="image")
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")

    widget = label_assistant_widget(viewer)

    assert widget._tool_tabs.count() == 3
    assert [
        widget._tool_tabs.tabText(index)
        for index in range(widget._tool_tabs.count())
    ] == ["Edit & Review", "Visual Compare", "Combine Layers"]


def test_components_tab_exposes_full_component_controls(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")

    widget = label_assistant_widget(viewer)
    components = widget._tool_tabs.widget(0)

    assert components._target_combo.findText("mask") >= 0
    assert components._component_table.columnCount() == 10
    assert components._analyze_button.text() == "Find components"
    assert components._delete_button.text() == "Delete selected"



def test_image_labels_view_changes_visibility_without_losing_labels_selection(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((16, 16), dtype=np.uint8), name="image")
    mask = viewer.add_labels(np.zeros((16, 16), dtype=np.uint8), name="mask")
    widget = label_assistant_widget(viewer)
    image_mask_view = widget._tool_tabs.widget(1)

    buttons = {
        button.text(): button
        for button in image_mask_view.findChildren(QPushButton)
    }
    buttons["Image Only"].click()

    assert image.visible
    assert not mask.visible
    assert viewer.layers.selection.active is mask


def test_combine_labels_merges_a_single_multivalue_layer(make_napari_viewer):
    viewer = make_napari_viewer()
    labels = np.array([[0, 2], [255, 0]], dtype=np.uint16)
    source = viewer.add_labels(labels, name="source mask")
    widget = label_assistant_widget(viewer)
    combine_masks = widget._tool_tabs.widget(2)

    combine_masks._mode_combo.setCurrentText("Merge Layers As Same Class")
    for index in range(combine_masks._layer_list.count()):
        item = combine_masks._layer_list.item(index)
        item.setSelected(item.text() == source.name)
    combine_masks._merge_value_spin.setValue(7)
    combine_masks._apply_button.click()

    result = viewer.layers["source mask | merged_class_7"]
    assert np.array_equal(
        result.data, np.array([[0, 7], [7, 0]], dtype=np.int32)
    )
