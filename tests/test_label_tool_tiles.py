import numpy as np
import pytest

from napari_label_assistant_tools import _label_tool_tiles as tools


class BoundedArray:
    """Fail if an operation requests the full source or an oversized tile."""
    def __init__(self, pixels):
        self.pixels = pixels
        self.shape = pixels.shape

    def __array__(self, *args, **kwargs):
        raise AssertionError("Full source materialization")

    def __getitem__(self, region):
        block = self.pixels[region]
        assert max(block.shape) <= 2
        return block


@pytest.mark.parametrize("mode,expected", [
    ("Merge Layers As Same Class", [[7, 7, 7], [7, 0, 7]]),
    ("Compose Layers As Distinct Classes", [[1, 2, 2], [2, 0, 1]]),
    ("Vote Count Map", [[1, 1, 2], [2, 0, 1]]),
    ("Consensus Map", [[0, 0, 9], [9, 0, 0]]),
])
def test_disk_adapter_is_bounded_and_matches_original_operations(monkeypatch, mode, expected):
    monkeypatch.setattr(tools, "TILE_SIZE", 2)
    data = [BoundedArray(np.array([[2, 0, 2], [2, 0, 2]], dtype=np.uint8)),
            BoundedArray(np.array([[0, 3, 3], [3, 0, 0]], dtype=np.uint8))]
    result, owner = tools.run_label_tool(data, mode, background=0, source=None,
        new=5, merge=7, start=1, votes=2, consensus=9)
    try:
        np.testing.assert_array_equal(result[:], expected)
    finally:
        owner.cleanup()


def test_disk_reassign_validates_across_tiles(monkeypatch):
    monkeypatch.setattr(tools, "TILE_SIZE", 2)
    pixels = np.array([[0, 2, 0], [0, 0, 2]], dtype=np.uint8)
    settings = dict(background=0, source=None, new=5, merge=7, start=1, votes=2, consensus=9)
    result, owner = tools.run_label_tool([BoundedArray(pixels)], "Reassign Label Value", **settings)
    try:
        np.testing.assert_array_equal(result[:], [[0, 5, 0], [0, 0, 5]])
    finally:
        owner.cleanup()
    pixels[-1, -1] = 3
    with pytest.raises(ValueError, match="exactly one non-background"):
        tools.run_label_tool([BoundedArray(pixels)], "Reassign Label Value", **settings)
    settings["source"] = 2
    result, owner = tools.run_label_tool([BoundedArray(pixels)], "Reassign Label Value", **settings)
    try:
        np.testing.assert_array_equal(result[:], [[0, 5, 0], [0, 0, 3]])
    finally:
        owner.cleanup()


@pytest.mark.parametrize("overwrite", [False, True])
def test_gui_operates_on_full_source_and_saves_output(tmp_path, make_napari_viewer, qtbot, overwrite):
    from test_mask_tiles import workspace
    from napari_label_assistant_tools._mask_tiles import full_data
    from napari_label_assistant_tools._widget import label_operations_widget
    from napari_label_assistant_tools.workspace import load_workspace, save_workspace

    path, arrays = workspace(tmp_path, shape=(515, 520))
    viewer = make_napari_viewer()
    load_workspace(viewer, path)
    source = viewer.layers["Curated"]
    widget = label_operations_widget(viewer)
    qtbot.addWidget(widget)
    widget._mode_combo.setCurrentText("Merge Layers As Same Class")
    for i in range(widget._layer_list.count()):
        item = widget._layer_list.item(i)
        item.setSelected(item.text() == source.name)
    widget._merge_value_spin.setValue(9)
    widget._overwrite_check.setChecked(overwrite)
    widget._apply_button.click()
    qtbot.waitUntil(lambda: widget.isEnabled(), timeout=30000)
    output = source if overwrite else viewer.layers["Curated | merged_class_9"]
    assert int(full_data(output)[-1, -1]) == 9  # Outside the visible crop.
    assert int(arrays[0][-1, -1]) == 2**31 + 17
    destination = tmp_path / "saved" / "case.label-assistant.json"
    save_workspace(viewer, destination)
    restored = make_napari_viewer()
    load_workspace(restored, destination)
    assert int(full_data(restored.layers[output.name])[-1, -1]) == 9
