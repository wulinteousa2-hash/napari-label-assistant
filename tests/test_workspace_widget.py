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
    assert "navigation" in widget.optimize_images_note.text()
    assert "multiscale" in widget.optimize_images_check.toolTip()

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


def test_save_as_requests_local_mask_copy(make_napari_viewer, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QFileDialog
    widget = WorkspaceManagerWidget(make_napari_viewer(), settings=_settings(tmp_path / "settings.ini"))
    destination = tmp_path / "local.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    calls = []
    monkeypatch.setattr(widget, "_save_path", lambda path, **kwargs: calls.append((path, kwargs)))
    widget.save_as()
    assert calls == [(destination, {"copy_labels": True})]


def test_save_failure_is_visible_and_preserves_project(make_napari_viewer, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    widget = WorkspaceManagerWidget(make_napari_viewer(), settings=_settings(tmp_path / "settings.ini"))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    def fail():
        raise PermissionError("Shared mount is read-only")

    widget._run(fail, success=lambda result: "saved", completed_path=tmp_path / "new.json")
    assert widget.workspace_path is None
    assert "read-only" in widget.status_label.text()
    assert len(warnings) == 1


def test_complete_package_setting_persists_and_reaches_save(make_napari_viewer, tmp_path, monkeypatch):
    from napari_label_assistant_tools import _workspace_widget as module
    settings_path = tmp_path / "settings.ini"
    viewer = make_napari_viewer()
    widget = WorkspaceManagerWidget(viewer, settings=_settings(settings_path))
    assert widget.storage_mode_combo.currentData() is False
    widget.storage_mode_combo.setCurrentIndex(1)
    restored = WorkspaceManagerWidget(viewer, settings=_settings(settings_path))
    assert restored.storage_mode_combo.currentData() is True
    calls = []
    def fake_save(viewer, path, **kwargs):
        calls.append(kwargs)
        return {"path": str(path), "saved_layers": 1, "complete_package": True}
    monkeypatch.setattr(module, "save_workspace", fake_save)
    restored._save_path(tmp_path / "project.json")
    assert calls[0]["complete_package"] is True
    assert "Keep the workspace file" in restored.status_label.text()


def test_partial_load_warns_about_missing_layers(make_napari_viewer, tmp_path, monkeypatch):
    from napari_label_assistant_tools import _workspace_widget as module
    viewer = make_napari_viewer()
    widget = WorkspaceManagerWidget(viewer, settings=_settings(tmp_path / "settings.ini"))
    monkeypatch.setattr(module, "load_workspace", lambda *args, **kwargs: {
        "path": "project.json", "restored_layers": ["mask"],
        "skipped_layers": [{"name": "image", "reason": "missing /shared/image.tiff"}],
    })
    warnings = []
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *args: warnings.append(args))
    widget._load_with_progress(tmp_path / "project.json")
    assert len(warnings) == 1
    assert "image" in warnings[0][2]
    assert "Include all data (portable)" in warnings[0][2]


def test_complete_package_with_optimization_preserves_real_multiscale_image(
    make_napari_viewer, tmp_path, monkeypatch
):
    import shutil
    from napari_label_assistant_tools.workspace import service
    monkeypatch.setattr(service, "LARGE_IMAGE_AXIS_THRESHOLD", 4)
    viewer = make_napari_viewer()
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image = viewer.add_image([pixels, pixels[::2, ::2]], multiscale=True, name="reference")
    viewer.add_labels(np.ones((8, 8), dtype=np.uint8), name="Curated")
    assert image.multiscale
    assert not service._needs_image_pyramid(image)
    package = tmp_path / "package"
    service.save_workspace(viewer, package / "project.json", complete_package=True,
                           optimize_large_images=True, xy_chunk=3)
    moved = tmp_path / "moved"
    shutil.move(str(package), moved)
    restored = make_napari_viewer()
    result = service.load_workspace(restored, moved / "project.json")
    assert result["skipped_layers"] == []
    assert restored.layers["reference"].multiscale
    assert len(restored.layers["reference"].data) == 2
    assert np.array_equal(np.asarray(restored.layers["reference"].data[0]), pixels)
    assert np.all(np.asarray(restored.layers["Curated"].data) == 1)


def test_saved_state_is_independent_of_next_save_mode(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.ones((8, 8), dtype=np.uint8))
    setattr(viewer, VIEWER_WORKSPACE_PATH_ATTR, str(tmp_path / "project.json"))
    setattr(viewer, VIEWER_WORKSPACE_LAYER_IDS_ATTR, {id(layer)})
    viewer._label_assistant_workspace_storage_state = "linked"
    widget = WorkspaceManagerWidget(viewer, settings=_settings(tmp_path / "settings.ini"))
    widget.storage_mode_combo.setCurrentIndex(1)
    assert "Saved data: linked" in widget.storage_state_label.text()
    assert "Copy all images" in widget.package_note.text()


def test_incomplete_workspace_stays_protected_after_panel_recreation(make_napari_viewer, tmp_path, monkeypatch):
    from napari_label_assistant_tools import _workspace_widget as module
    viewer = make_napari_viewer()
    viewer.add_labels(np.ones((8, 8), dtype=np.uint8))
    viewer._label_assistant_missing_workspace_layers = [{"name": "image", "reason": "missing"}]
    widget = WorkspaceManagerWidget(viewer, settings=_settings(tmp_path / "settings.ini"))
    warnings = []
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(module, "save_workspace", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Save must not run")))
    widget._save_path(tmp_path / "project.json")
    assert "saving protected" in widget.storage_state_label.text()
    assert len(warnings) == 1
    assert not (tmp_path / "project.json").exists()


def test_recent_entries_keep_paths_and_menu_disables_unavailable_files(make_napari_viewer, tmp_path, monkeypatch):
    from napari_label_assistant_tools._workspace_widget import RECENT_WORKSPACES_KEY
    existing = tmp_path / "existing.json"
    existing.write_text("{}")
    missing = tmp_path / "missing.json"
    settings = _settings(tmp_path / "settings.ini")
    settings.setValue(RECENT_WORKSPACES_KEY, [str(existing), str(missing)])
    widget = WorkspaceManagerWidget(make_napari_viewer(), settings=settings)
    actions = widget.recent_menu.actions()
    assert actions[0].isEnabled()
    assert not actions[1].isEnabled()
    assert "unavailable" in widget.recent_list.item(1).text()
    opened = []
    monkeypatch.setattr(widget, "_load_path", lambda path: opened.append(path))
    actions[0].trigger()
    widget.recent_list.setCurrentRow(0)
    widget.open_selected_recent()
    assert opened == [existing, existing]
