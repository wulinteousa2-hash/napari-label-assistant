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
    assert "Unsaved workspace" in widget.current_label.text()


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
    assert "Unsaved workspace" in widget.current_label.text()
    assert "Save will ask for a new name" in widget.status_label.text()


def test_large_image_optimization_preference_is_explicit_and_persistent(
    make_napari_viewer, tmp_path
):
    settings_path = tmp_path / "settings.ini"
    viewer = make_napari_viewer()
    widget = WorkspaceManagerWidget(
        viewer, settings=_settings(settings_path)
    )

    assert widget.optimize_images_check.isChecked() is False
    assert "multiscale OME-Zarr" in widget.optimize_images_note.text()
    assert "Off keeps links" in widget.optimize_images_note.text()

    widget.optimize_images_check.setChecked(True)
    settings = _settings(settings_path)
    restored = WorkspaceManagerWidget(viewer, settings=settings)

    assert restored.optimize_images_check.isChecked() is True


def test_optimized_save_rebinds_real_napari_image_as_multiscale(
    make_napari_viewer, tmp_path, monkeypatch
):
    from napari_label_assistant_tools.workspace import (
        load_workspace,
        save_workspace,
    )
    from napari_label_assistant_tools.workspace import service

    monkeypatch.setattr(service, "LARGE_IMAGE_AXIS_THRESHOLD", 4)
    monkeypatch.setattr(service, "PYRAMID_SMALLEST_LEVEL", 2)
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.arange(63, dtype=np.uint8).reshape(7, 9), name="reference"
    )

    path = tmp_path / "case.label-assistant.json"
    save_workspace(
        viewer,
        path,
        xy_chunk=4,
        optimize_large_images=True,
    )

    restored_viewer = make_napari_viewer()
    load_workspace(restored_viewer, path)
    restored = restored_viewer.layers[0]

    assert layer.multiscale is False
    assert restored.multiscale is True
    assert [tuple(level.shape) for level in restored.data] == [
        (7, 9),
        (4, 5),
        (2, 3),
        (1, 2),
    ]
