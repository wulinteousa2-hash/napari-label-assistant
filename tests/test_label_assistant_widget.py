import numpy as np
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QLabel, QPushButton

from napari_label_assistant_tools import label_assistant_widget


def test_label_assistant_stylesheets_parse_without_qt_warnings(
    make_napari_viewer, qapp
):
    messages = []

    def _message_handler(_message_type, _context, message):
        messages.append(str(message))

    previous_handler = qInstallMessageHandler(_message_handler)
    try:
        viewer = make_napari_viewer()
        viewer.add_labels(
            np.zeros((32, 32), dtype=np.uint8), name="mask"
        )
        widget = label_assistant_widget(viewer)
        widget.show()
        qapp.processEvents()
    finally:
        qInstallMessageHandler(previous_handler)

    parse_messages = [
        message
        for message in messages
        if "Could not parse stylesheet" in message
    ]
    label_styles = [
        (repr(label), label.text(), label.styleSheet())
        for label in widget.findChildren(QLabel)
        if label.styleSheet()
    ]
    assert not parse_messages, label_styles


def test_label_assistant_widget_contains_all_tabs(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), dtype=np.uint8), name="image")
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")

    widget = label_assistant_widget(viewer)

    assert widget._tool_tabs.count() == 4
    assert [
        widget._tool_tabs.tabText(index)
        for index in range(widget._tool_tabs.count())
    ] == ["Project", "Labels", "Visual Compare", "Mask Tools"]


def test_components_tab_exposes_full_component_controls(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")

    widget = label_assistant_widget(viewer)
    components = widget._tool_tabs.widget(1)

    assert components._workflow_tabs.count() == 2
    assert [
        components._workflow_tabs.tabText(index)
        for index in range(components._workflow_tabs.count())
    ] == ["Annotate", "Grid & Components"]
    assert components._target_combo.findText("mask") >= 0
    assert components._component_table.columnCount() == 10
    assert components._analyze_button.text() == "Find components"
    assert components._delete_button.text() == "Delete selected"


def test_layer_selectors_refresh_automatically_and_keep_visible_fallback(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="first")
    widget = label_assistant_widget(viewer)
    labels = widget._tool_tabs.widget(1)

    assert labels._refresh_layers_button.text() == "Refresh layers"

    second = viewer.add_labels(
        np.zeros((32, 32), dtype=np.uint8), name="second"
    )
    qtbot.waitUntil(lambda: labels._target_combo.findText("second") >= 0)

    second.name = "renamed"
    qtbot.waitUntil(lambda: labels._target_combo.findText("renamed") >= 0)



def test_image_labels_view_changes_visibility_without_losing_labels_selection(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((16, 16), dtype=np.uint8), name="image")
    mask = viewer.add_labels(np.zeros((16, 16), dtype=np.uint8), name="mask")
    widget = label_assistant_widget(viewer)
    image_mask_view = widget._tool_tabs.widget(2)

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
    combine_masks = widget._tool_tabs.widget(3)

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
