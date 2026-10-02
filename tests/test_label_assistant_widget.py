import numpy as np
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QLabel, QPushButton

from napari_label_assistant_tools import label_assistant_widget
from napari_label_assistant_tools._widget import component_operations_widget


def test_copy_destination_refresh_and_manual_selection(make_napari_viewer):
    viewer = make_napari_viewer()
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    source = viewer.add_labels(mask, name="source", scale=(2, 3), translate=(4, 5))
    widget = component_operations_widget(viewer)
    curated = viewer.add_labels(np.zeros_like(mask), name="Curated")
    widget._refresh_layers_button.click()
    assert widget._copy_target_combo.findText("Curated") >= 0

    other = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="other")
    widget._refresh_layers_button.click()
    assert widget._copy_target_combo.findText("other") == -1
    widget._manual_copy_target_check.setChecked(True)
    assert widget._copy_target_combo.findText("other") >= 0
    widget._copy_target_combo.setEditText("other")
    widget._refresh_layers_button.click()
    assert widget._copy_target_combo.currentText() == "other"

    widget._analyze_button.click()
    widget._component_table.select_component_id(1)
    widget._copy_button.click()
    matching = viewer.layers["other (source shape)"]
    assert np.array_equal(matching.data, mask)
    assert np.array_equal(matching.scale, source.scale)
    assert np.array_equal(matching.translate, source.translate)
    assert widget._copy_target_combo.currentText() == matching.name
    assert viewer.layers.selection.active is source
    assert not np.any(other.data)

    widget._copy_target_combo.setEditText("missing")
    widget._refresh_layers_button.click()
    assert widget._copy_target_combo.currentText() == "missing"
    widget._copy_button.click()
    assert np.array_equal(viewer.layers["missing"].data, mask)

    layer_count = len(viewer.layers)
    widget._copy_button.click()
    assert len(viewer.layers) == layer_count

    widget._copy_target_combo.setEditText("Curated")
    widget._copy_button.click()
    assert np.array_equal(curated.data, mask)
    widget._manual_copy_target_check.setChecked(False)
    assert not widget._copy_target_combo.isEditable()
    assert widget._copy_target_combo.findText("other") == -1


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

    assert widget._tool_tabs.count() == 5
    assert [
        widget._tool_tabs.tabText(index)
        for index in range(widget._tool_tabs.count())
    ] == ["Workspace", "Labels", "Training", "Compare", "Combine"]
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    assert isinstance(widget._tool_tabs.widget(2), QuickTrainWidget)
    labels = widget._tool_tabs.widget(1)
    assert labels._layer_activity_indicator is widget._activity_header
    assert widget._activity_header.height() == 46


def test_activity_header_paints_high_contrast_background(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")
    widget = label_assistant_widget(viewer)
    qtbot.addWidget(widget)
    widget.show()
    qtbot.wait(50)

    image = widget._activity_header.grab().toImage()
    background = image.pixelColor(8, 8)

    assert background.red() > 220
    assert background.green() > 235
    assert background.blue() > 245


def test_components_tab_exposes_full_component_controls(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")

    widget = label_assistant_widget(viewer)
    components = widget._tool_tabs.widget(1)

    assert components._workflow_tabs.count() == 2
    assert [
        components._workflow_tabs.tabText(index)
        for index in range(components._workflow_tabs.count())
    ] == ["Local Editing", "Review & QC"]
    assert components._target_combo.findText("mask") >= 0
    assert components._component_table.columnCount() == 10
    assert components._analyze_button.text() == "Find components"
    assert components._delete_button.text() == "Delete selected"
    components._workflow_tabs.setCurrentIndex(0)
    assert not components._target_combo.parentWidget().isHidden()


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


def test_layer_activity_indicator_tracks_napari_selection(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((32, 32), dtype=np.uint8), name="reference image"
    )
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")
    widget = label_assistant_widget(viewer)
    labels = widget._tool_tabs.widget(1)

    viewer.layers.selection.active = image

    qtbot.waitUntil(
        lambda: labels._layer_activity_indicator.state == "ready"
        and "reference image"
        in labels._layer_activity_indicator.message.text()
    )


def test_layer_activity_indicator_tracks_source_selector(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((32, 32), dtype=np.uint8), name="reference image"
    )
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="mask")
    widget = label_assistant_widget(viewer)
    labels = widget._tool_tabs.widget(1)

    labels._target_combo.setCurrentText("reference image")

    qtbot.waitUntil(
        lambda: labels._layer_activity_indicator.state == "ready"
        and "reference image"
        in labels._layer_activity_indicator.message.text()
    )



def test_image_labels_view_changes_visibility_without_losing_labels_selection(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((16, 16), dtype=np.uint8), name="image")
    mask = viewer.add_labels(np.zeros((16, 16), dtype=np.uint8), name="mask")
    widget = label_assistant_widget(viewer)
    image_mask_view = widget._tool_tabs.widget(3)

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
    combine_masks = widget._tool_tabs.widget(4)

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
