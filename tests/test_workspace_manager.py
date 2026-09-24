import json
from types import SimpleNamespace

import numpy as np
import pytest

from napari_label_assistant_tools.workspace import (
    WORKSPACE_FORMAT,
    load_workspace,
    read_workspace,
    save_workspace,
)


class Labels:
    def __init__(self, data, name="labels"):
        self.data = data
        self.name = name
        self.visible = True
        self.opacity = 0.7
        self.blending = "translucent"
        self.scale = (1.0,) * len(data.shape)
        self.translate = (0.0,) * len(data.shape)
        self.metadata = {}
        self.source = SimpleNamespace(path=None, reader_plugin=None)


class LayerList(list):
    def __init__(self, values=()):
        super().__init__(values)
        self.selection = SimpleNamespace(active=self[0] if self else None)

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)


class FakeViewer:
    def __init__(self, layers=()):
        self.layers = LayerList(layers)
        self.dims = SimpleNamespace(current_step=(0, 0))

    def add_labels(self, data, name):
        layer = Labels(data, name)
        self.layers.append(layer)
        return layer


def test_workspace_round_trip_uses_writable_zarr_and_restores_grid(tmp_path):
    zarr = pytest.importorskip("zarr")
    layer = Labels(np.zeros((24, 32), dtype=np.uint16), "mask")
    layer.data[3:7, 5:9] = 4
    layer.metadata["label_assistant_grid"] = {
        "cell_height": 512,
        "cell_width": 768,
        "assign_addresses": True,
        "show_overlay": False,
    }
    helper = Labels(np.ones((8, 8), dtype=np.uint16), "mask — editable area")
    helper.metadata["label_assistant_internal_role"] = "editable_tile"
    path = tmp_path / "case.label-assistant.json"

    result = save_workspace(FakeViewer([layer, helper]), path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert result["saved_layers"] == 1
    assert payload["format"] == WORKSPACE_FORMAT
    assert payload["layers"][0]["label_assistant_grid"]["cell_height"] == 512

    restored_viewer = FakeViewer()
    progress = []
    loaded = load_workspace(
        restored_viewer,
        path,
        progress=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )
    restored = restored_viewer.layers[0]

    assert loaded["restored_layers"] == ["mask"]
    assert progress[0] == (0, 1, "Preparing project…")
    assert progress[-1] == (1, 1, "Restoring viewer state…")
    assert restored_viewer._label_assistant_workspace_loading is False
    assert restored.metadata["label_assistant_grid"]["cell_width"] == 768
    assert type(restored.data).__module__.startswith("zarr")
    restored.data[0, 0] = 9
    storage = payload["layers"][0]["storage"]
    stored = zarr.open_array(
        str(tmp_path / storage["path"] / storage["array_path"]), mode="r"
    )
    assert int(stored[0, 0]) == 9


def test_save_commits_pending_local_edits_before_serializing(tmp_path):
    pytest.importorskip("zarr")
    layer = Labels(np.zeros((8, 8), dtype=np.uint8), "mask")

    class Controller:
        closed = False
        dirty = True
        stroke_active = False

        def __init__(self):
            self.called = False

        def commit(self, *, update_status):
            self.called = True
            layer.data[2, 3] = 7
            self.dirty = False
            return True

    controller = Controller()
    viewer = FakeViewer([layer])
    viewer._label_assistant_edit_controllers = [controller]

    save_workspace(viewer, tmp_path / "case.label-assistant.json")

    assert controller.called is True
    assert int(layer.data[2, 3]) == 7


def test_repeated_save_keeps_previous_manifest_backup(tmp_path):
    pytest.importorskip("zarr")
    layer = Labels(np.zeros((8, 8), dtype=np.uint8), "first name")
    viewer = FakeViewer([layer])
    path = tmp_path / "case.label-assistant.json"

    save_workspace(viewer, path)
    layer.name = "second name"
    save_workspace(viewer, path)

    backup = tmp_path / "case.label-assistant.backup.json"
    previous = json.loads(backup.read_text(encoding="utf-8"))
    current = json.loads(path.read_text(encoding="utf-8"))
    assert previous["layers"][0]["name"] == "first name"
    assert current["layers"][0]["name"] == "second name"


def test_reads_existing_sam3_workspace_for_migration(tmp_path):
    path = tmp_path / "legacy.sam3.json"
    path.write_text(
        json.dumps(
            {
                "format": "napari-sam3-workspace",
                "version": 1,
                "viewer": {},
                "layers": [{"name": "mask"}],
            }
        ),
        encoding="utf-8",
    )

    _resolved, payload = read_workspace(path)

    assert payload["format"] == WORKSPACE_FORMAT
    assert payload["imported_from"] == "napari-sam3-workspace-v1"
