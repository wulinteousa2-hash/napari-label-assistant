import numpy as np
import pytest

import napari_label_assistant_tools._widget as widget_module
import napari_label_assistant_tools._large_component_controller as controller_module


def test_cancel_large_analysis_restores_controls_without_raising(
    make_napari_viewer, monkeypatch, qtbot
):
    build_index = controller_module.build_large_component_index

    def cancellable_analysis(source, *, cancel_event):
        yield 1, 2, "Waiting for cancellation"
        assert cancel_event.wait(5)
        return (yield from build_index(source, tile_size=4, cancel_event=cancel_event))

    monkeypatch.setattr(controller_module, "build_large_component_index", cancellable_analysis)
    monkeypatch.setattr(widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 50)
    viewer = make_napari_viewer()
    mask = np.ones((12, 12), dtype=np.uint8)
    source = viewer.add_labels(mask.copy(), name="source")
    widget = widget_module.component_operations_widget(viewer)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: widget._analysis_progress.value() == 50)
    assert not source.editable
    widget._cancel_analysis_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy)
    assert "analysis canceled" in widget._status_label.text()
    assert widget._analyze_button.isEnabled()
    assert widget._cancel_analysis_button.isHidden()
    assert widget._analysis_progress.isHidden()
    assert source.editable
    assert np.array_equal(source.data, mask)
    assert widget._component_table.rowCount() == 0

    monkeypatch.setattr(controller_module, "build_large_component_index", build_index)
    widget._analyze_button.click()
    qtbot.waitUntil(
        lambda: not widget._large_controller.busy
        and widget._component_table.rowCount() == 1,
        timeout=15000,
    )


def test_large_analysis_failure_is_reported_without_reraising(
    make_napari_viewer, monkeypatch, qtbot
):
    def failing_analysis(source, *, cancel_event):
        yield 0, 1, "Starting"
        raise ValueError("Cannot read tile")

    monkeypatch.setattr(controller_module, "build_large_component_index", failing_analysis)
    monkeypatch.setattr(widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 50)
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.ones((12, 12), dtype=np.uint8), name="source")
    widget = widget_module.component_operations_widget(viewer)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy)
    assert "analysis failed: Cannot read tile" in widget._status_label.text()
    assert widget._analyze_button.isEnabled()
    assert source.editable


@pytest.mark.parametrize("large", [False, True])
def test_remove_small_components_uses_strict_area_threshold(
    make_napari_viewer, monkeypatch, qtbot, large
):
    monkeypatch.setattr(
        widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 50 if large else 1000
    )
    viewer = make_napari_viewer()
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1, 1] = 3
    mask[4, 4:6] = 3
    mask[8:10, 8:10] = 7
    source = viewer.add_labels(mask.copy(), name="source")
    widget = widget_module.component_operations_widget(viewer)

    widget._remove_small_button.click()
    assert "Find components" in widget._status_label.text()
    assert np.array_equal(source.data, mask)
    widget._analyze_button.click()
    qtbot.waitUntil(
        lambda: not widget._large_controller.busy
        and widget._component_table.rowCount() == 3,
        timeout=15000,
    )
    widget._remove_small_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy, timeout=15000)
    expected = mask.copy()
    expected[1, 1] = 0
    assert np.array_equal(source.data, expected)
    assert widget._component_table.rowCount() == 2

    widget._remove_small_button.click()
    assert "No components below 2 pixels" in widget._status_label.text()
    widget._small_component_size_spin.setValue(3)
    widget._remove_small_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy, timeout=15000)
    expected[4, 4:6] = 0
    assert np.array_equal(source.data, expected)
    assert widget._component_table.rowCount() == 1


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

    widget._manual_copy_target_check.setChecked(True)
    widget._copy_target_combo.setEditText("new curated")
    widget._copy_button.click()
    qtbot.waitUntil(
        lambda: not widget._large_controller.edit_active, timeout=15000
    )
    assert np.array_equal(viewer.layers["new curated"].data, mask)
    assert widget._copy_target_combo.currentText() == "new curated"

    widget._component_table.select_component_id(1)
    widget._delete_button.click()
    qtbot.waitUntil(
        lambda: not widget._large_controller.edit_active, timeout=15000
    )
    assert not np.any(viewer.layers["large mask"].data)
    assert widget._component_table.rowCount() == 0
