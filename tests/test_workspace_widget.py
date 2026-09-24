from pathlib import Path

import numpy as np
from qtpy.QtCore import QSettings

from napari_label_assistant_tools._workspace_widget import (
    LAST_WORKSPACE_KEY,
    VIEWER_WORKSPACE_LAYER_IDS_ATTR,
    VIEWER_WORKSPACE_PATH_ATTR,
    WorkspaceManagerWidget,
)


def _settings(path: Path) -> QSettings:
    return QSettings(str(path), QSettings.IniFormat)


def test_last_recent_project_is_not_assumed_active(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((8, 8), dtype=np.uint8), name="new image")
    settings = _settings(tmp_path / "settings.ini")
    settings.setValue(LAST_WORKSPACE_KEY, str(tmp_path / "previous.json"))

    widget = WorkspaceManagerWidget(viewer, settings=settings)

    assert widget.workspace_path is None
    assert "Unsaved project" in widget.current_label.text()


def test_removing_all_project_layers_detaches_save_target(
    make_napari_viewer, qapp, tmp_path
):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.zeros((8, 8), dtype=np.uint8), name="old image")
    project = tmp_path / "previous.label-assistant.json"
    setattr(viewer, VIEWER_WORKSPACE_PATH_ATTR, str(project))
    setattr(viewer, VIEWER_WORKSPACE_LAYER_IDS_ATTR, {id(layer)})
    widget = WorkspaceManagerWidget(
        viewer, settings=_settings(tmp_path / "settings.ini")
    )

    viewer.layers.remove(layer)
    qapp.processEvents()

    assert widget.workspace_path is None
    assert "Unsaved project" in widget.current_label.text()
    assert "Save will ask for a new name" in widget.status_label.text()
