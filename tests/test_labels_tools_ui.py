import numpy as np
import pytest
from qtpy.QtWidgets import QFormLayout, QGroupBox, QPushButton

from napari_label_assistant_tools._widget import (
    component_operations_widget, label_operations_widget, quick_compare_toggle_widget,
)


def buttons(widget):
    return {button.text(): button for button in widget.findChildren(QPushButton)}


def select(widget, *names):
    for i in range(widget._layer_list.count()):
        item = widget._layer_list.item(i)
        item.setSelected(item.text() in names)


def test_sections_collapsed_in_local_editing(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = component_operations_widget(viewer)
    qtbot.addWidget(widget)
    for section, content in ((widget._view_compare_section, widget._view_compare),
                             (widget._label_tools_section, widget._label_tools)):
        assert section.parentWidget() is widget._workflow_tabs.widget(0).widget()
        assert not section.isChecked()
        assert content.isHidden()
        section.setChecked(True)
        assert not content.isHidden()
        section.setChecked(False)
        assert content.isHidden()


def test_compare_modes_opacity_and_effects_keep_edit_focus(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.arange(16).reshape(4, 4))
    mask = viewer.add_labels(np.eye(4, dtype=np.uint8), name="comparison")
    edit = viewer.add_labels(np.zeros((4, 4), dtype=np.uint8), name="editing")
    original = mask.data.copy()
    widget = quick_compare_toggle_widget(viewer)
    qtbot.addWidget(widget)
    widget._mask_combo.setCurrentText(mask.name)
    mask.opacity = .6
    assert widget._opacity_spin.value() == 60
    viewer.layers.selection.active = edit
    controls = buttons(widget)
    for mode, visibility in (("Image Only", (True, False)), ("Overlay", (True, True)),
                             ("Labels Only", (False, True))):
        controls[mode].click()
        assert (image.visible, mask.visible) == visibility
        assert viewer.layers.selection.active is edit
    widget._opacity_spin.setValue(35)
    assert mask.opacity == pytest.approx(.35)
    controls["Overlay"].click()
    assert mask.opacity == pytest.approx(.35)
    for effect in ("Peek Labels", "Blink Labels", "Pulse Labels"):
        advanced = next(group for group in widget.findChildren(QGroupBox)
                        if group.title() == "Advanced view effects")
        advanced.setChecked(True)
        controls[effect].click()
        state = widget._animation_state
        state.animation_timer.stop()
        widget._tick_animation()
        if effect == "Peek Labels":
            assert not mask.visible
        elif effect == "Blink Labels":
            widget._tick_animation()
            assert not mask.visible
        else:
            assert mask.opacity == pytest.approx(.15)
        while state.animation_mode:
            widget._tick_animation()
        assert image.visible and mask.visible
        assert mask.opacity == pytest.approx(.35)
        assert viewer.layers.selection.active is edit
    np.testing.assert_array_equal(mask.data, original)


@pytest.mark.parametrize("mode,fields", [
    ("Reassign Label Value", {"Background value", "Source value (-1 = auto)", "New class value"}),
    ("Merge Layers As Same Class", {"Background value", "Merge class value"}),
    ("Compose Layers As Distinct Classes", {"Background value", "Class values start at"}),
    ("Vote Count Map", set()),
    ("Consensus Map", {"Background value", "Minimum votes", "Consensus class value"}),
])
def test_operation_only_shows_relevant_fields(make_napari_viewer, qtbot, mode, fields):
    widget = label_operations_widget(make_napari_viewer())
    qtbot.addWidget(widget)
    widget._mode_combo.setCurrentText(mode)
    form = widget._settings_form
    visible = {form.itemAt(row, QFormLayout.LabelRole).widget().text()
               for row in range(form.rowCount() - 1)
               if not form.itemAt(row, QFormLayout.FieldRole).widget().isHidden()}
    assert visible == fields | {"Output name"}


@pytest.mark.parametrize("mode,expected", [
    ("Merge Layers As Same Class", [[1, 1], [1, 0]]),
    ("Compose Layers As Distinct Classes", [[1, 2], [2, 0]]),
    ("Vote Count Map", [[1, 1], [2, 0]]),
    ("Consensus Map", [[0, 0], [1, 0]]),
])
def test_operations_create_new_outputs(make_napari_viewer, qtbot, mode, expected):
    viewer = make_napari_viewer()
    a = np.array([[2, 0], [2, 0]], dtype=np.uint8)
    b = np.array([[0, 3], [3, 0]], dtype=np.uint8)
    first = viewer.add_labels(a.copy(), name="a")
    second = viewer.add_labels(b.copy(), name="b")
    viewer.add_image(a, name="image")
    widget = label_operations_widget(viewer)
    qtbot.addWidget(widget)
    assert widget._layer_list.count() == 2
    widget._mode_combo.setCurrentText(mode)
    select(widget, "a", "b")
    assert widget._apply_button.isEnabled()
    widget._apply_button.click()
    np.testing.assert_array_equal(viewer.layers[-1].data, expected)
    np.testing.assert_array_equal(first.data, a)
    np.testing.assert_array_equal(second.data, b)


def test_reassign_and_explicit_overwrite(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.array([[0, 2], [2, 0]], dtype=np.uint8), name="source")
    widget = label_operations_widget(viewer)
    qtbot.addWidget(widget)
    select(widget, source.name)
    widget._new_value_spin.setValue(7)
    widget._output_name.setCurrentText("custom output")
    widget._apply_button.click()
    np.testing.assert_array_equal(source.data, [[0, 2], [2, 0]])
    np.testing.assert_array_equal(viewer.layers["custom output"].data, [[0, 7], [7, 0]])
    select(widget, source.name)
    widget._overwrite_check.setChecked(True)
    count = len(viewer.layers)
    widget._apply_button.click()
    assert len(viewer.layers) == count
    np.testing.assert_array_equal(source.data, [[0, 7], [7, 0]])


def test_invalid_combinations_and_values_are_safe(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    a = viewer.add_labels(np.array([[0, 1], [2, 0]], dtype=np.uint8), name="a")
    viewer.add_labels(np.zeros((3, 3), dtype=np.uint8), name="wrong shape")
    other = viewer.add_labels(np.zeros((2, 2), dtype=np.uint8), name="offset", translate=(1, 0))
    viewer.add_labels(np.zeros((2, 2, 2), dtype=np.uint8), name="3d")
    widget = label_operations_widget(viewer)
    qtbot.addWidget(widget)
    assert widget._layer_list.count() == 3
    widget._mode_combo.setCurrentText("Merge Layers As Same Class")
    select(widget, "a", "wrong shape")
    assert not widget._apply_button.isEnabled()
    assert "shape" in widget._status.text()
    select(widget, "a", "offset")
    assert not widget._apply_button.isEnabled()
    assert "transform" in widget._status.text()
    other.translate = (0, 0)
    widget._mode_combo.setCurrentText("Consensus Map")
    select(widget, "a")
    assert not widget._apply_button.isEnabled()
    widget._mode_combo.setCurrentText("Reassign Label Value")
    count = len(viewer.layers)
    widget._apply_button.click()
    assert len(viewer.layers) == count
    assert "exactly one non-background" in widget._status.text()
    np.testing.assert_array_equal(a.data, [[0, 1], [2, 0]])
