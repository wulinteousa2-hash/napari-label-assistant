from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import shutil
import errno

import numpy as np
import pytest

from napari_label_assistant_tools._model_files import (
    BUNDLE_SUFFIXES, associate_workspace, workspace_locations, file_status,
    copy_checkpoint_bundle, verify_relocated_checkpoint, update_reference,
)
from napari_label_assistant_tools._model_library import catalog
from napari_label_assistant_tools._model_project import project, add_task, remember_model, KEY


def test_locations_only_use_active_workspace(tmp_path):
    viewer = SimpleNamespace(_last_workspace_path=tmp_path / "old.json")
    assert workspace_locations(viewer) == (None, None)
    associate_workspace(viewer, tmp_path / "Study.label-assistant.json")
    manifest, folder = workspace_locations(viewer)
    assert manifest == tmp_path / "Study.label-assistant.json"
    assert folder == tmp_path / "Study_label_assistant_data" / "models"
    path = tmp_path / "external.pt"
    path.write_bytes(b"weights")
    assert file_status(path, dirty=False, models_folder=folder) == "Linked externally"
    assert file_status(path, dirty=True, models_folder=folder) == "Unsaved"
    path.unlink()
    assert file_status(path, dirty=False, models_folder=folder) == "Missing"


def test_copy_preserves_bundle_and_does_not_overwrite(tmp_path):
    source = tmp_path / "source" / "external.pt"
    source.parent.mkdir()
    for suffix in BUNDLE_SUFFIXES:
        source.with_suffix(suffix).write_bytes(suffix.encode())
    folder = tmp_path / "workspace" / "models"
    folder.mkdir(parents=True)
    conflict = folder / "model1-rev1.pt"
    conflict.write_bytes(b"do not overwrite")
    copied = Path(copy_checkpoint_bundle(source, folder, "model1", "rev1"))
    assert copied != conflict
    assert conflict.read_bytes() == b"do not overwrite"
    for suffix in BUNDLE_SUFFIXES:
        assert copied.with_suffix(suffix).read_bytes() == source.with_suffix(suffix).read_bytes()
    assert copy_checkpoint_bundle(copied, folder, "model1", "rev1") == str(copied)


def test_copy_failure_publishes_no_partial_bundle(tmp_path, monkeypatch):
    source = tmp_path / "external.pt"
    source.write_bytes(b"weights")
    source.with_suffix(".metrics.csv").write_text("dice\n0.8\n")
    original_copy = shutil.copy2
    def failing_copy(src, dst):
        if str(src).endswith(".metrics.csv"):
            raise OSError("Disk full")
        return original_copy(src, dst)
    monkeypatch.setattr(shutil, "copy2", failing_copy)
    folder = tmp_path / "models"
    with pytest.raises(OSError, match="Disk full"):
        copy_checkpoint_bundle(source, folder, "model1", "rev1")
    assert not list(folder.iterdir())
    assert source.read_bytes() == b"weights"


def test_copy_supports_filesystems_without_hard_links(tmp_path, monkeypatch):
    source = tmp_path / "external.pt"
    source.write_bytes(b"weights")
    def unsupported(*a, **k):
        raise OSError(errno.EOPNOTSUPP, "Hard links not supported")
    monkeypatch.setattr("napari_label_assistant_tools._model_files.os.link", unsupported)
    destination = copy_checkpoint_bundle(source, tmp_path / "models", "model1", "rev1")
    assert Path(destination).read_bytes() == b"weights"


@pytest.fixture
def saved_model(make_napari_viewer, tmp_path):
    pytest.importorskip("torch")
    from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel, save_model
    from napari_label_assistant_tools._model_providers import build_model
    viewer = make_napari_viewer()
    image_path = tmp_path / "EM.npy"
    np.save(image_path, np.zeros((32, 32), dtype=np.uint8))
    viewer.open(str(image_path), name="EM")
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name="Curated")
    task = add_task(viewer, "Myelin", "Sheath pixels")
    cfg = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4)
    metrics = {"dice": .8, "iou": .7, "precision": .8, "recall": .8}
    history = {"mode": "new", "workflow": "coverage", "epochs": 1, "retained_epoch": 1,
               "config": asdict(cfg), "retained_validation": metrics, "dataset_snapshot_id": "dataset1",
               "started_at_utc": "2026-10-20T12:00:00Z", "validation": {"method": "holdout", "seed": 42},
               "checkpoints": [{"epoch": 1, "validation": metrics}], "source": {"image_layer": "EM"}}
    model = QuickTrainModel(build_model(cfg), cfg, 32, metrics, "cpu", history=[history])
    record = remember_model(viewer, model, task_id=task["id"])
    source = tmp_path / "external" / "model.pt"
    source.parent.mkdir()
    save_model(model, source)
    record.update(checkpoint=str(source), saved_revision=record["revision"])
    record["runs"][0].update(checkpoint=str(source), saved_revision=record["revision"], notes="Preserve my observations")
    return viewer, model, source


def test_relink_verifies_metadata_without_constructing_network(saved_model, tmp_path, monkeypatch):
    viewer, model, source = saved_model
    row = catalog(viewer)[0]
    relocated = tmp_path / "relocated.pt"
    shutil.move(source, relocated)
    assert catalog(viewer)[0]["file_status"] == "Missing"
    def no_network(*a, **k):
        raise AssertionError("Relinking must not construct a network")
    monkeypatch.setattr("napari_label_assistant_tools._model_providers.build_model", no_network)
    verified = verify_relocated_checkpoint(relocated, row, asdict(model.config))
    update_reference(viewer, row, verified, expected_state=project(viewer), expected_workspace=None)
    current = catalog(viewer)[0]
    assert current["file_status"] == "Linked externally"
    assert current["model_id"] == row["model_id"]
    assert current["run_id"] == row["run_id"]
    assert current["revision"] == row["revision"]
    assert current["notes"] == "Preserve my observations"
    assert len(project(viewer)["models"][0]["runs"]) == 1


@pytest.mark.parametrize("wrong", ["model", "revision", "configuration"])
def test_relink_rejects_wrong_file(saved_model, tmp_path, wrong):
    import torch
    viewer, model, source = saved_model
    row = catalog(viewer)[0]
    payload = torch.load(source, weights_only=True)
    if wrong == "model":
        payload["identity"]["model_id"] = "differentmodel"
    elif wrong == "revision":
        payload["history"][-1]["started_at_utc"] = "2026-10-21T12:00:00Z"
    else:
        payload["config"]["crop_pixels"] = 64
    other = tmp_path / "wrong.pt"
    torch.save(payload, other)
    with pytest.raises(ValueError, match="another model|history does not match|configuration does not match"):
        verify_relocated_checkpoint(other, row, asdict(model.config))
    assert catalog(viewer)[0]["checkpoint"] == str(source)


def test_reference_update_rejects_workspace_change(saved_model, tmp_path):
    viewer, _, source = saved_model
    old_state = project(viewer)
    row = catalog(viewer)[0]
    setattr(viewer, KEY, deepcopy(old_state))
    with pytest.raises(ValueError, match="workspace changed"):
        update_reference(viewer, row, source, expected_state=old_state, expected_workspace=None)
    current = project(viewer)
    associate_workspace(viewer, tmp_path / "different.json")
    with pytest.raises(ValueError, match="workspace changed"):
        update_reference(viewer, row, source, expected_state=current, expected_workspace=None)


def test_relink_older_run_does_not_replace_latest_checkpoint(saved_model, tmp_path):
    viewer, model, source = saved_model
    old = catalog(viewer)[0]
    model.history = deepcopy(model.history) + deepcopy(model.history)
    model.history[-1]["mode"] = "continue"
    model.history[-1]["started_at_utc"] = "2026-10-21T12:00:00Z"
    latest = remember_model(viewer, model, task_id=model.identity["task_id"])
    revision = latest["revision"]
    path_before = latest["checkpoint"]
    relocated = tmp_path / "older-run.pt"
    shutil.copy2(source, relocated)
    verify_relocated_checkpoint(relocated, old, asdict(model.config))
    update_reference(viewer, old, relocated, expected_state=project(viewer), expected_workspace=None)
    assert latest["revision"] == revision
    assert latest["checkpoint"] == path_before
    assert latest["runs"][0]["checkpoint"] == str(relocated)
    assert latest["runs"][0]["notes"] == "Preserve my observations"


def test_managed_copy_save_as_and_moved_workspace(saved_model, tmp_path):
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer, _, source = saved_model
    manifest = tmp_path / "project" / "study.label-assistant.json"
    save_workspace(viewer, manifest)
    assert workspace_locations(viewer)[0] == manifest
    assert catalog(viewer)[0]["file_status"] == "Linked externally"
    row = catalog(viewer)[0]
    destination = copy_checkpoint_bundle(source, workspace_locations(viewer)[1], row["model_id"], row["revision"])
    update_reference(viewer, row, destination, expected_state=project(viewer), expected_workspace=manifest)
    assert catalog(viewer)[0]["file_status"] == "Managed"
    save_workspace(viewer, manifest)
    portable = tmp_path / "portable" / "study.label-assistant.json"
    save_workspace(viewer, portable, complete_package=True, copy_labels=True)
    moved = tmp_path / "moved"
    shutil.move(portable.parent, moved)
    load_workspace(viewer, moved / portable.name)
    current = catalog(viewer)[0]
    assert current["file_status"] == "Managed"
    assert Path(current["checkpoint"]).is_relative_to(moved)
    assert Path(current["checkpoint"]).is_file()
    assert Path(current["checkpoint"]).with_suffix(".history.json").is_file()
    assert current["notes"] == "Preserve my observations"
    assert source.is_file()


def test_library_file_actions_and_search_scope(saved_model, tmp_path, qtbot, monkeypatch):
    from napari_label_assistant_tools._model_library_widget import ModelLibraryWidget, QFileDialog
    from napari_label_assistant_tools.workspace import save_workspace
    viewer, _, source = saved_model
    manifest = tmp_path / "workspace.label-assistant.json"
    save_workspace(viewer, manifest)
    widget = ModelLibraryWidget(viewer)
    qtbot.addWidget(widget)
    assert manifest.name in widget.scope_label.text()
    assert str(workspace_locations(viewer)[1]) == widget.models_location.text()
    assert widget.copy_file_button.isEnabled()
    widget.file_filter.setCurrentIndex(widget.file_filter.findData("Missing"))
    assert widget.models.rowCount() == 0
    widget.file_filter.setCurrentIndex(0)
    assert widget.models.rowCount() == 1
    widget.copy_file_button.click()
    qtbot.waitUntil(lambda: widget.file_worker is None, timeout=15000)
    current = catalog(viewer)[0]
    assert current["file_status"] == "Managed"
    assert "Save the workspace" in widget.status.text()
    assert source.is_file()
    moved = tmp_path / "relocated"
    moved.mkdir()
    original = Path(current["checkpoint"])
    destination = moved / "renamed.pt"
    for suffix in BUNDLE_SUFFIXES:
        item = original.with_suffix(suffix)
        if item.is_file():
            shutil.move(item, destination.with_suffix(suffix))
    widget.refresh()
    assert widget.locate_file_button.isEnabled()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(destination), ""))
    widget.locate_file_button.click()
    qtbot.waitUntil(lambda: widget.file_worker is None, timeout=15000)
    assert catalog(viewer)[0]["checkpoint"] == str(destination)
    assert "Verified and repaired" in widget.status.text()
    widget.search.setText("renamed.pt")
    assert widget.models.rowCount() == 1
    widget.search.setText("not-registered.pt")
    assert widget.models.rowCount() == 0
    assert not widget.copy_file_button.isEnabled()
    assert not widget.locate_file_button.isEnabled()


def test_workspace_panel_tracks_service_open_and_save_as(saved_model, tmp_path, qtbot):
    from napari_label_assistant_tools._workspace_widget import WorkspaceManagerWidget
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer, _, _ = saved_model
    widget = WorkspaceManagerWidget(viewer=viewer)
    qtbot.addWidget(widget)
    first = tmp_path / "first.label-assistant.json"
    second = tmp_path / "second.label-assistant.json"
    save_workspace(viewer, first)
    assert widget.workspace_path == first
    save_workspace(viewer, second, complete_package=True, copy_labels=True)
    assert widget.workspace_path == second
    load_workspace(viewer, first)
    assert widget.workspace_path == first
    assert "1 models" in widget.model_catalog_label.text()
