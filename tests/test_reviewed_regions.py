from types import SimpleNamespace

import numpy as np

from napari_label_assistant_tools._dynamic_edit import TileBounds
from napari_label_assistant_tools._reviewed_regions import KEY, ReviewedAreaWidget, associate, associated_mask, identity
from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
from napari_label_assistant_tools.workspace.service import _common_record, _apply_common_state


def setup(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((32, 32), dtype=np.uint8))
    truth = viewer.add_labels(np.zeros((32, 32), dtype=np.uint32))
    truth.data[3:6, 3:6] = 17
    mask = viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="coverage")
    associate(mask, image)
    mask.metadata[KEY]["truth_id"] = identity(truth)
    controller = SimpleNamespace(bounds=TileBounds(0, 0, 16, 16), tile=truth, source=truth,
                                 loading=False, stroke_active=False, dirty=False)
    widget = ReviewedAreaWidget(viewer, controller)
    qtbot.addWidget(widget)
    return viewer, image, truth, mask, controller, widget


def test_mark_remove_preserve_objects_and_preview(make_napari_viewer, qtbot):
    viewer, image, truth, mask, controller, widget = setup(make_napari_viewer, qtbot)
    widget.mark(1)
    assert mask.data[:16, :16].all()
    assert not mask.data[16:].any()
    assert truth.data[4, 4] == 17
    widget.preview()
    assert widget.preview_layer.data[4, 4] == 17
    assert not widget.preview_layer.editable
    controller.bounds = TileBounds(0, 0, 8, 8)
    widget.mark(0)
    assert not mask.data[:8, :8].any()
    assert mask.data[8:16, :16].all()
    assert truth.data[4, 4] == 17
    assert len(mask.metadata[KEY]["areas"]) == 2


def test_training_association_partial_only_and_reindex(make_napari_viewer, qtbot):
    viewer, image, truth, mask, controller, review = setup(make_napari_viewer, qtbot)
    training = QuickTrainWidget(viewer)
    qtbot.addWidget(training)
    assert training.reviewed_combo.currentData() is mask
    image.name = "renamed"
    mask.name = "also renamed"
    assert associated_mask(viewer, image) is mask
    training.prepared = object()
    review.mark(1)
    assert training.prepared is None
    training.annotation_combo.setCurrentIndex(training.annotation_combo.findData("complete"))
    training.reviewed_combo.setCurrentIndex(0)
    training.refresh_layers()
    assert training.reviewed_combo.currentData() is None
    training.annotation_combo.setCurrentIndex(training.annotation_combo.findData("partial"))
    assert training.reviewed_combo.currentData() is mask
    other = viewer.add_image(np.zeros((32, 32)))
    training.refresh_layers()
    training.image_combo.setCurrentIndex(training.image_combo.findData(other))
    assert training.reviewed_combo.currentData() is None


def test_workspace_metadata_preserves_association(make_napari_viewer, qtbot):
    viewer, image, truth, mask, _, _ = setup(make_napari_viewer, qtbot)
    records = [_common_record(layer, layer._type_string.title()) for layer in (image, truth, mask)]
    for layer in (image, truth, mask):
        layer.metadata = {}
    for layer, record in zip((image, truth, mask), records):
        _apply_common_state(layer, record)
    assert associated_mask(viewer, image) is mask
    assert mask.metadata[KEY]["truth_id"] == truth.metadata[KEY]["id"]


def test_portable_workspace_roundtrip(make_napari_viewer, qtbot, tmp_path):
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer, image, truth, mask, _, review = setup(make_napari_viewer, qtbot)
    review.mark(1)
    path = tmp_path / "reviewed.label-assistant.json"
    save_workspace(viewer, path, complete_package=True)
    restored = make_napari_viewer()
    load_workspace(restored, path)
    loaded_image = restored.layers[image.name]
    loaded_mask = associated_mask(restored, loaded_image)
    assert loaded_mask is not None
    assert np.asarray(loaded_mask.data[:16, :16]).all()
    assert loaded_mask.metadata[KEY]["areas"] == [[0, 0, 16, 16]]
    training = QuickTrainWidget(restored)
    qtbot.addWidget(training)
    assert training.reviewed_combo.currentData() is loaded_mask


def test_first_mark_creates_associated_disk_mask(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    from napari_label_assistant_tools import _reviewed_regions as module
    viewer, image, truth, mask, _, review = setup(make_napari_viewer, qtbot)
    viewer.layers.remove(mask)
    training = QuickTrainWidget(viewer)
    qtbot.addWidget(training)
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "review.zarr"), ""))
    review.mark(1)
    mask = associated_mask(viewer, image)
    qtbot.waitUntil(lambda: training.reviewed_combo.currentData() is mask)
    assert mask.data[:16, :16].all()
    assert mask.metadata[KEY]["truth_id"] == identity(truth)


def configure_mouse(review, truth, qtbot, cell_size=10):
    from qtpy.QtWidgets import QCheckBox, QSpinBox
    height, width, show = QSpinBox(), QSpinBox(), QCheckBox()
    for spin in (height, width):
        spin.setValue(cell_size)
        qtbot.addWidget(spin)
    qtbot.addWidget(show)
    review.configure_grid(lambda: truth, height, width, show)
    review.mouse_review_check.setChecked(True)
    return height, width, show


def click(review, truth, y, x, button=1):
    review.mouse_click(None, SimpleNamespace(type="mouse_press", button=button,
                                            position=truth.data_to_world((y, x))))


def test_mouse_selects_background_cell_and_clips_edge(make_napari_viewer, qtbot):
    _, _, truth, mask, _, review = setup(make_napari_viewer, qtbot)
    configure_mouse(review, truth, qtbot)
    click(review, truth, 31, 31)
    assert review.selected_cell[1] == TileBounds(30, 30, 32, 32)
    assert "R03C03" in review.cell_label.text()
    review.mark_button.click()
    assert mask.data[30:, 30:].all()
    assert not mask.data[:30].any()
    assert truth.data[4, 4] == 17
    assert review.review_overlay.visible


def test_mouse_direct_mark_remove_and_grid_change(make_napari_viewer, qtbot):
    _, _, truth, mask, _, review = setup(make_napari_viewer, qtbot)
    height, _, _ = configure_mouse(review, truth, qtbot)
    review.mouse_action_combo.setCurrentIndex(review.mouse_action_combo.findData("mark"))
    click(review, truth, 12, 12)
    assert mask.data[10:20, 10:20].all()
    click(review, truth, -1, 12)
    click(review, truth, 1, 1, button=2)
    assert not mask.data[:10].any()
    review.mouse_action_combo.setCurrentIndex(review.mouse_action_combo.findData("remove"))
    click(review, truth, 12, 12)
    assert not mask.data.any()
    assert not mask.metadata[KEY]["areas"]
    height.setValue(8)
    assert review.selected_cell is None
    review.mouse_review_check.setChecked(False)
    click(review, truth, 3, 3)
    assert review.selected_cell is None
    review.shutdown()
    assert review.mouse_click not in review.viewer.mouse_drag_callbacks


def test_grid_mouse_works_without_loaded_edit_tile(make_napari_viewer, qtbot):
    _, _, truth, mask, controller, review = setup(make_napari_viewer, qtbot)
    controller.source = controller.bounds = controller.tile = None
    configure_mouse(review, truth, qtbot)
    review.mouse_action_combo.setCurrentIndex(review.mouse_action_combo.findData("mark"))
    click(review, truth, 5, 5)
    assert mask.data[:10, :10].all()


def test_mark_merges_objects_as_one_value_and_unmark_clears(make_napari_viewer, qtbot):
    viewer, _, truth, mask, controller, review = setup(make_napari_viewer, qtbot)
    truth.data[12:14,12:14] = 91
    original = truth.data.copy()
    configure_mouse(review, truth, qtbot)
    review.mouse_action_combo.setCurrentIndex(review.mouse_action_combo.findData("mark"))
    click(review, truth, 4, 4)
    annotations = next(layer for layer in viewer.layers if layer.metadata.get(KEY, {}).get("role") == "reviewed_annotations")
    assert annotations.data[4,4] == 1
    assert annotations.data[12,12] == 0
    assert annotations.data[0,0] == 0
    assert mask.data[0,0] == 1
    click(review, truth, 12, 12)
    assert annotations.data[12,12] == 1
    assert annotations.data[4,4] == 1
    assert set(np.unique(np.asarray(annotations.data[:]))) == {0,1}
    review.mouse_action_combo.setCurrentIndex(review.mouse_action_combo.findData("remove"))
    click(review, truth, 4, 4)
    assert annotations.data[4,4] == 0
    assert annotations.data[12,12] == 1
    np.testing.assert_array_equal(truth.data, original)
    assert not annotations.editable


def test_simple_workflow_and_automatic_ground_truth(make_napari_viewer, qtbot):
    viewer, image, truth, mask, _, review = setup(make_napari_viewer, qtbot)
    configure_mouse(review, truth, qtbot)
    assert not review.mark_button.isEnabled()
    assert review.mouse_action_combo.isHidden()
    assert review.preview_button.isHidden()
    click(review, truth, 4, 4)
    assert review.mark_button.isEnabled()
    assert "Finish review" in review.mark_button.text()
    review.mark_button.click()
    annotations = next(layer for layer in viewer.layers if layer.metadata.get(KEY, {}).get("role") == "reviewed_annotations")
    other = viewer.add_labels(np.zeros((32,32), dtype=np.uint8), name="Unreviewed SAM3")
    training = QuickTrainWidget(viewer)
    qtbot.addWidget(training)
    assert training.truth_combo.currentData() is truth
    assert training.reviewed_combo.currentData() is mask
    assert training.truth_combo.findData(annotations) == -1
    assert training.truth_combo.findData(mask) == -1
    assert training.create_reviewed_button.isHidden()
    training.truth_combo.setCurrentIndex(training.truth_combo.findData(other))
    assert training.truth_combo.currentData() is truth
    review.advanced_check.setChecked(True)
    review.mouse_action_combo.setCurrentIndex(review.mouse_action_combo.findData("mark"))
    review.advanced_check.setChecked(False)
    assert review.mouse_action_combo.currentData() == "select"


def test_cell_selection_returns_to_editable_tile(make_napari_viewer, qtbot):
    viewer, _, truth, _, controller, review = setup(make_napari_viewer, qtbot)
    tile = viewer.add_labels(truth.data[:16,:16].copy(), name="Editable Curated area")
    controller.tile = tile
    truth.editable = False
    tile.mode = "paint"
    configure_mouse(review, truth, qtbot)
    click(review, truth, 4, 4)
    review.mouse_review_check.setChecked(False)
    assert viewer.layers.selection.active is tile
    assert str(tile.mode) == "paint"
    assert review.selection_overlay.visible
    assert review.selected_cell[1] == TileBounds(0,0,10,10)
