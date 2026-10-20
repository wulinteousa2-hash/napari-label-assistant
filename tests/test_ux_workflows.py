import numpy as np
import pytest
from types import SimpleNamespace

from napari_label_assistant_tools import label_assistant_widget
from napari_label_assistant_tools._dynamic_edit import TileBounds
from napari_label_assistant_tools._reviewed_regions import associate, identity, KEY
import napari_label_assistant_tools._widget as widget_module


def test_disk_backed_recovery_rejects_replaced_data(tmp_path):
    import zarr
    from napari_label_assistant_tools._deletion_history import DeletionHistory
    data = zarr.open_array(str(tmp_path / "mask.zarr"), mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    data[1:3, 1:3] = 42
    layer = SimpleNamespace(data=data)
    history = DeletionHistory(layer)
    slices = (slice(0, 4), slice(0, 4))
    history.capture(slices, np.asarray(data[slices]) == 42)
    data[1:3, 1:3] = 0
    assert list(history.restore())[-1][:2] == (4, 0)
    assert np.all(np.asarray(data[1:3, 1:3]) == 42)
    layer.data = np.zeros((8, 8))
    with pytest.raises(ValueError, match="replaced"):
        list(history.restore())
    history.close()


@pytest.mark.parametrize("large", [False, True])
def test_cleanup_undo_preserves_later_edits(make_napari_viewer, monkeypatch, qtbot, large):
    monkeypatch.setattr(widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 50 if large else 1000)
    viewer = make_napari_viewer()
    original = np.zeros((12, 12), dtype=np.uint16)
    original[1, 1:3] = 7
    original[8:11, 8:11] = 12
    source = viewer.add_labels(original.copy())
    widget = widget_module.component_operations_widget(viewer)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy and widget._component_table.rowCount() == 2)
    widget._small_component_size_spin.setValue(3)
    widget._remove_small_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy)
    assert not source.data[1, 1:3].any()
    source.data[1, 1] = 99
    widget._undo_delete_button.click()
    qtbot.waitUntil(lambda: not widget._large_controller.busy)
    expected = original.copy()
    expected[1, 1] = 99
    assert np.array_equal(source.data, expected)
    assert "preserved 1 later edits" in widget._status_label.text()
    assert not widget._undo_delete_button.isEnabled()


def test_review_next_skips_reviewed_cells_and_wraps(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((8, 8), dtype=np.uint8))
    truth = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8))
    mask = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8))
    associate(mask, image)
    mask.metadata[KEY]["truth_id"] = identity(truth)
    widget = widget_module.component_operations_widget(viewer)
    widget._target_combo.setCurrentText(truth.name)
    widget._grid_y_spin.setValue(4)
    widget._grid_x_spin.setValue(4)
    review = widget._reviewed_area_widget
    review.selected_cell = (truth, TileBounds(0, 4, 4, 8), 4, 4)
    review.mark(1)
    review.selected_cell = (truth, TileBounds(0, 0, 4, 4), 4, 4)
    review.finish_and_next()
    assert review.selected_cell[1] == TileBounds(4, 0, 8, 4)
    assert mask.data[:4].all()
    review.finish_and_next()
    assert review.selected_cell[1] == TileBounds(4, 4, 8, 8)
    review.finish_and_next()
    assert "All cells" in review.editor_summary.text()


def test_canceled_review_storage_does_not_advance(make_napari_viewer, monkeypatch):
    from napari_label_assistant_tools._reviewed_regions import QFileDialog
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((8, 8), dtype=np.uint8))
    truth = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8))
    widget = widget_module.component_operations_widget(viewer)
    review = widget._reviewed_area_widget
    widget._grid_y_spin.setValue(4)
    widget._grid_x_spin.setValue(4)
    selected = (truth, TileBounds(0, 0, 4, 4), 4, 4)
    review.selected_cell = selected
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    review.finish_and_next()
    assert review.selected_cell == selected


def test_shared_pair_restored_and_prerequisites_visible(make_napari_viewer):
    viewer = make_napari_viewer()
    image1 = viewer.add_image(np.zeros((8, 8)), name="image one")
    image2 = viewer.add_image(np.zeros((8, 8)), name="image two")
    truth1 = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="truth one")
    truth2 = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="truth two")
    page = label_assistant_widget(viewer)
    training = page._tool_tabs.widget(2)
    training.image_combo.setCurrentIndex(training.image_combo.findData(image1))
    training.truth_combo.setCurrentIndex(training.truth_combo.findData(truth1))
    training.image_combo.setCurrentIndex(training.image_combo.findData(image2))
    training.truth_combo.setCurrentIndex(training.truth_combo.findData(truth2))
    training.image_combo.setCurrentIndex(training.image_combo.findData(image1))
    assert training.truth_combo.currentData() is truth1
    assert "image one" in training.source_context.text()
    assert "Prepare the dataset" in training.readiness_label.text()
    assert "Train or load a model" in training.readiness_label.text()
    workspace = page._tool_tabs.widget(0)
    workspace.save_state_label.setText("Saved")
    truth1.name = "renamed annotations"
    assert "changed since last save" in workspace.save_state_label.text()
