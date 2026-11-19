"""Readable names must not change checkpoint identity or overwrite old bundles."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from napari_label_assistant_tools._model_naming import (
    BUNDLE_SUFFIXES, available_checkpoint_path, checkpoint_filename,
)
from napari_label_assistant_tools._model_project import (
    KEY, add_task, project, remember_model, resolve_references, save_project,
)
from napari_label_assistant_tools._model_providers import ModelSpec
from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel


def record():
    config = asdict(QuickTrainConfig(base_channels=8, input_pixels=1024))
    return {"id": "a02ed1718cb5424e8a1215410ee3d47d", "name": "Remak",
            "revision": "7a72a5188a434283adda50bc706b7b25",
            "spec": ModelSpec(width=8).to_dict(), "config": config,
            "history": [{"source": {"image_layer": "EM08"},
                         "started_at_utc": "2026-11-18T14:32:55Z", "config": config}]}


def test_readable_filename_uses_recorded_metadata():
    row = record()
    before = deepcopy(row)
    assert checkpoint_filename(row) == "Remak_EM08_unet-w8_in1024_run01_20261118_7a72a518.pt"
    assert row == before


def test_historical_run_uses_its_own_architecture_input_and_date():
    row = record()
    first = {key: deepcopy(row[key]) for key in ("revision", "spec", "config")}
    first["history"] = deepcopy(row["history"][0])
    second = deepcopy(first)
    second["revision"] = "22222222abcdef"
    second["spec"]["width"] = 32
    second["config"]["input_pixels"] = 512
    second["history"]["started_at_utc"] = "2026-11-19T10:00:00Z"
    row.update({key: deepcopy(second[key]) for key in ("revision", "spec", "config")})
    row["history"].append(second["history"])
    row["runs"] = [first, second]
    assert checkpoint_filename(row, first) == "Remak_EM08_unet-w8_in1024_run01_20261118_7a72a518.pt"
    assert checkpoint_filename(row) == "Remak_EM08_unet-w32_in512_run02_20261119_22222222.pt"


def test_smp_name_uses_encoder_instead_of_hidden_width():
    row = record()
    row["spec"] = ModelSpec(provider="smp", architecture="unetplusplus", encoder="resnet34").to_dict()
    name = checkpoint_filename(row)
    assert "smp-unetplusplus-resnet34" in name
    assert "-w8" not in name


def test_date_is_utc_and_legacy_config_can_supply_width():
    row = record()
    row.pop("spec")
    row["history"][0]["started_at_utc"] = "2026-11-18T23:30:00-06:00"
    assert "_unet-w8_in1024_run01_20261119_" in checkpoint_filename(row)


@pytest.mark.parametrize("date", (None, "", "bad date", 123))
def test_missing_or_invalid_date_is_not_invented(date):
    row = record()
    row["history"][0]["started_at_utc"] = date
    assert "_undated_" in checkpoint_filename(row)


def test_filename_is_bounded_and_safe_for_unusual_names():
    row = record()
    row["name"] = "../Remák: " + "long name/" * 100
    row["history"][0]["source"]["image_layer"] = "../../EM \\ test " * 100
    name = checkpoint_filename(row)
    assert Path(name).name == name
    assert re.fullmatch(r"[A-Za-z0-9_-]+\.pt", name)
    assert len(name.encode()) < 240
    assert name.startswith("Remak-")


@pytest.mark.parametrize("suffix", BUNDLE_SUFFIXES)
def test_collision_with_any_companion_keeps_existing_file(tmp_path, suffix):
    filename = checkpoint_filename(record())
    conflict = (tmp_path / filename).with_suffix(suffix)
    conflict.write_bytes(b"preserve existing file")
    destination = available_checkpoint_path(tmp_path, filename)
    assert destination.name != filename
    assert not destination.exists()
    assert conflict.read_bytes() == b"preserve existing file"


def trained_model():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    from napari_label_assistant_tools._model_providers import build_model
    config = QuickTrainConfig(base_channels=4, crop_pixels=32, input_pixels=32)
    history = [{"source": {"image_layer": "EM08"}, "started_at_utc": "2026-11-18T14:32:55Z",
                "config": asdict(config), "mode": "new", "checkpoints": []}]
    return QuickTrainModel(build_model(config), config, 32, {}, "cpu", history=history)


def test_workspace_new_rounds_are_readable_and_repeated_saves_keep_names(tmp_path):
    viewer = SimpleNamespace()
    task = add_task(viewer, "Remak", "Confirmed target pixels")
    model = trained_model()
    remember_model(viewer, model, task_id=task["id"], name="Remak")
    manifest, data = tmp_path / "workspace.json", tmp_path / "data"

    def save():
        saved = save_project(viewer, manifest, data, False)
        setattr(viewer, KEY, resolve_references(deepcopy(saved), manifest))
        return project(viewer)["models"][0]

    row = save()
    first = Path(row["checkpoint"])
    assert first.name.startswith("Remak_EM08_unet-w4_in32_run01_20261118_")
    original = first.read_bytes()
    timestamp = first.stat().st_mtime_ns
    row["name"] = "Renamed Remak"
    assert save()["checkpoint"] == str(first)
    assert first.stat().st_mtime_ns == timestamp
    model.history = deepcopy(model.history) + deepcopy(model.history)
    model.history[-1].update(mode="continue", started_at_utc="2026-11-19T12:00:00Z")
    remember_model(viewer, model)
    row = save()
    second = Path(row["checkpoint"])
    assert second.name.startswith("Renamed-Remak_EM08_unet-w4_in32_run02_20261119_")
    assert row["runs"][0]["checkpoint"] == str(first)
    assert first.read_bytes() == original
    for path in (first, second):
        assert all(path.with_suffix(suffix).is_file() for suffix in BUNDLE_SUFFIXES)
        assert json.loads(path.with_suffix(".history.json").read_text())["model_file"] == path.name


@pytest.mark.parametrize("extension", (".pt", ".PT", ".pth", ""))
def test_legacy_saved_names_survive_save_as_and_reload(tmp_path, extension):
    from napari_label_assistant_tools._quick_train import load_model, save_model
    viewer = SimpleNamespace()
    model = trained_model()
    legacy = tmp_path / "old" / "models" / ("modelid-revisionid" + extension)
    legacy.parent.mkdir(parents=True)
    save_model(model, legacy)
    remember_model(viewer, model, path=legacy, name="Remak")
    timestamp = legacy.stat().st_mtime_ns
    manifest = tmp_path / "old" / "workspace.json"
    saved = save_project(viewer, manifest, legacy.parent.parent, False)
    assert (manifest.parent / saved["models"][0]["checkpoint"]).resolve() == legacy
    assert legacy.stat().st_mtime_ns == timestamp
    copied_manifest = tmp_path / "new" / "workspace.json"
    copied = save_project(viewer, copied_manifest, tmp_path / "new" / "data", True)
    destination = copied_manifest.parent / copied["models"][0]["checkpoint"]
    if extension.lower() == ".pt":
        assert destination.name == legacy.name
    else:
        assert destination.name.startswith("Remak_EM08_unet-w4_in32_run01_20261118_")
    assert legacy.is_file()
    assert destination.is_file()
    assert all(destination.with_suffix(suffix).is_file() for suffix in BUNDLE_SUFFIXES if suffix != ".pt")
    assert load_model(destination).config == model.config


def test_workspace_save_avoids_existing_checkpoint_bundle(tmp_path):
    from napari_label_assistant_tools._quick_train import load_model
    viewer = SimpleNamespace()
    model = trained_model()
    row = remember_model(viewer, model, name="Remak")
    manifest, data = tmp_path / "workspace.json", tmp_path / "data"
    folder = data / "models"
    folder.mkdir(parents=True)
    conflict = folder / checkpoint_filename(row)
    conflict.write_bytes(b"unrelated existing checkpoint")
    conflict.with_suffix(".metrics.csv").write_bytes(b"unrelated existing metrics")
    saved = save_project(viewer, manifest, data, False)
    destination = manifest.parent / saved["models"][0]["checkpoint"]
    assert destination != conflict
    assert conflict.read_bytes() == b"unrelated existing checkpoint"
    assert conflict.with_suffix(".metrics.csv").read_bytes() == b"unrelated existing metrics"
    assert load_model(destination).config == model.config
    assert json.loads(destination.with_suffix(".history.json").read_text())["model_file"] == destination.name


def test_managed_copy_uses_readable_stem_and_updates_history_filename(tmp_path):
    from napari_label_assistant_tools._model_files import copy_checkpoint_bundle
    original = tmp_path / "external.pt"
    original.write_bytes(b"weights")
    sidecar = {"model_file": original.name, "history": [{"mode": "new"}]}
    original.with_suffix(".history.json").write_text(json.dumps(sidecar))
    filename = checkpoint_filename(record())
    destination = Path(copy_checkpoint_bundle(original, tmp_path / "models", "modelid", "revisionid", filename))
    assert destination.name == filename
    copied = json.loads(destination.with_suffix(".history.json").read_text())
    assert copied["model_file"] == filename
    assert copied["history"] == sidecar["history"]
    assert json.loads(original.with_suffix(".history.json").read_text()) == sidecar


def test_export_suggests_actual_checkpoint_configuration(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget, QFileDialog
    viewer = make_napari_viewer()
    model = trained_model()
    row = remember_model(viewer, model, name="Remak")
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.model = model
    widget.models_panel.width.setCurrentIndex(widget.models_panel.width.findData(64))
    suggestions = []

    def cancel_dialog(parent, title, filename, file_filter):
        suggestions.append(filename)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", cancel_dialog)
    widget._save()
    assert suggestions == [checkpoint_filename(row)]
    assert "_unet-w4_in32_" in suggestions[0]
    original_revision = row["revision"]
    model.history = deepcopy(model.history) + deepcopy(model.history)
    remember_model(viewer, model)
    project(viewer)["current_checkpoint"] = original_revision
    widget._save()
    assert suggestions[-1] == checkpoint_filename(row, row["runs"][0])
    assert "_run01_" in suggestions[-1]
    project(viewer)["models"].clear()
    widget._save()
    assert "EM08_unet-w4_in32_run02_" in suggestions[-1]
