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


def test_dataset_owns_annotations_and_shares_editor_selection(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._dynamic_edit import TileBounds

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((32, 32), dtype=np.float32), name="image")
    curated = viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="Curated")
    other = viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="other")
    panel = label_assistant_widget(viewer)
    qtbot.addWidget(panel)
    panel.resize(540, 850)
    panel.show()
    labels = panel._tool_tabs.widget(1)
    training = panel._tool_tabs.widget(2)
    review = labels._reviewed_area_widget
    library = labels.component_library_widget
    panel._tool_tabs.setCurrentIndex(2)
    assert training.reviewed_area_widget is review
    assert training.component_library_widget is library
    assert review.parentWidget() is training.reviewed_cells_page
    assert library.parentWidget() is training.selected_objects_page
    assert training.image_combo.currentData() is image
    assert review.image_combo.currentData() is image
    training.truth_combo.setCurrentIndex(training.truth_combo.findData(curated))
    assert labels._target_combo.currentText() == "Curated"
    selected = (curated, TileBounds(0, 0, 16, 16), 16, 16)
    review.selected_cell = selected
    training.annotation_tabs.setCurrentIndex(1)
    assert library.isVisible()
    assert not review.isVisible()
    assert library.source_layer() is curated
    training.workflow_tabs.setCurrentIndex(next(
        i for i in range(training.workflow_tabs.count())
        if training.workflow_tabs.tabText(i).endswith(". Train")
    ))
    assert training.train_button.isVisible()
    assert not library.isVisible()
    training.workflow_tabs.setCurrentIndex(0)
    training.annotation_tabs.setCurrentIndex(0)
    assert review.isVisible()
    assert review.selected_cell is selected
    labels._target_combo.setCurrentText("other")
    assert training.truth_combo.currentData() is other
    assert library.source_layer() is other
    assert review.selected_cell is None
    panel._tool_tabs.setCurrentIndex(1)
    assert not library.isVisible()
    assert not review.mark_button.isVisible()
    assert review.mouse_review_check.parentWidget() is labels._annotate_page
    training.workflow_tabs.setCurrentIndex(next(
        i for i in range(training.workflow_tabs.count())
        if training.workflow_tabs.tabText(i).endswith(". Train")
    ))
    assert training.metrics_label.isHidden()
    training.results_toggle.setChecked(True)
    assert not training.metrics_label.isHidden()


def test_dataset_click_marking_activates_canvas_and_restores_editing(make_napari_viewer, qtbot):
    from types import SimpleNamespace
    from napari_label_assistant_tools._reviewed_regions import KEY, associate, identity

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((64, 64), dtype=np.float32), name="image")
    curated = viewer.add_labels(np.zeros((64, 64), dtype=np.uint8), name="Curated")
    curated.data[2:6, 2:6] = 7
    mask = viewer.add_labels(np.zeros((64, 64), dtype=np.uint8), name="Reviewed regions")
    associate(mask, image)
    mask.metadata = {**mask.metadata, KEY: {**mask.metadata[KEY], "truth_id": identity(curated)}}
    panel = label_assistant_widget(viewer)
    qtbot.addWidget(panel)
    labels = panel._tool_tabs.widget(1)
    training = panel._tool_tabs.widget(2)
    review = training.reviewed_area_widget
    training.truth_combo.setCurrentIndex(training.truth_combo.findData(curated))
    labels._grid_y_spin.setValue(32)
    labels._grid_x_spin.setValue(32)
    curated.mode = "paint"
    assert not review.mouse_review_check.isChecked()
    review.advanced_check.setChecked(True)
    assert review.mouse_review_check.isChecked()
    assert review.mouse_action_combo.currentData() == "mark"
    assert labels._display_grid_check.isChecked()
    assert str(curated.mode) == "pan_zoom"
    review.mouse_click(viewer, SimpleNamespace(type="mouse_press", button=1,
                       position=curated.data_to_world((4, 4))))
    assert mask.data[:32, :32].all()
    assert not mask.data[32:, :].any()
    assert mask.metadata[KEY]["areas"] == [[0, 0, 32, 32]]
    review.advanced_check.setChecked(False)
    assert not review.mouse_review_check.isChecked()
    assert review.mouse_action_combo.currentData() == "select"
    assert str(curated.mode) == "paint"
    review.mouse_click(viewer, SimpleNamespace(type="mouse_press", button=1,
                       position=curated.data_to_world((40, 40))))
    assert not mask.data[32:, 32:].any()
    review.advanced_check.setChecked(True)
    labels._display_grid_check.setChecked(False)
    assert not review.mouse_review_check.isChecked()
    assert not review.advanced_check.isChecked()
