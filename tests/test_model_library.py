from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from qtpy.QtCore import Qt, QItemSelectionModel

from napari_label_assistant_tools._model_project import project, add_task, remember_model, RUNTIME, KEY
from napari_label_assistant_tools._model_library import (catalog, comparison_note, comparison_key,
    annotate, export_csv, chart_svg, run_curve, weight_statistics)
from napari_label_assistant_tools._model_library_widget import ModelLibraryWidget
from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel


def model(score=.7, dataset="dataset-a", width=4):
    cfg = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=width)
    metrics = {"dice": score, "iou": score-.1, "precision": score, "recall": score, "loss": 1-score}
    history = {"mode": "new", "workflow": "coverage", "config": asdict(cfg),
               "source": {"image_layer": "EM", "ground_truth_layer": "Curated"},
               "started_at_utc": "2026-10-20T12:00:00+00:00", "epochs": 2, "steps": 4,
               "retained_epoch": 2, "retained_validation": metrics,
               "validation": {"method": "holdout", "seed": 42, "holdout_fraction": .25},
               "dataset_snapshot_id": dataset, "duration_seconds": 10,
               "dataset_summary": {"training": {"total": 12}, "validation": {"total": 4}},
               "checkpoints": [{"epoch": 1, "validation": metrics}]}
    return QuickTrainModel(None, cfg, 32, metrics, "cpu", history=[history])


def populate(viewer):
    task = add_task(viewer, "Myelin", "Myelin sheath pixels")
    first = model()
    remember_model(viewer, first, task_id=task["id"], name="Small U-Net")
    second = model(.8)
    second.identity = dict(first.identity)
    second.history = deepcopy(first.history) + second.history
    second.history[-1]["mode"] = "continue"
    second.history[-1]["started_at_utc"] = "2026-10-21T12:00:00+00:00"
    remember_model(viewer, second, task_id=task["id"])
    other = model(.75, width=8)
    remember_model(viewer, other, task_id=task["id"], name="Wide U-Net")
    return first, second, other


def test_catalog_preserves_metrics_and_config_per_run():
    viewer = SimpleNamespace()
    populate(viewer)
    rows = catalog(viewer)
    assert len(rows) == 3
    assert [r["dice"] for r in rows] == [.7, .8, .75]
    assert [r["run"] for r in rows] == [1, 2, 1]
    assert rows[0]["training_patches"] == 12
    assert rows[1]["mode"] == "continue"
    assert rows[2]["width"] == 8
    assert "Small U-Net / run 2" in comparison_note(rows)
    assert "Small U-Net / run 2" in comparison_note(rows, "loss")
    rows[2]["dataset_id"] = "different"
    assert "No best-model ranking" in comparison_note(rows)
    rows[0]["dataset_id"] = ""
    assert comparison_key(rows[0]) is None


def test_legacy_runs_do_not_inherit_latest_score():
    viewer = SimpleNamespace()
    populate(viewer)
    state = project(viewer)
    old = state["models"][0]["runs"][0]
    old.pop("metrics")
    old["history"].pop("retained_validation")
    rows = catalog(viewer)
    assert rows[0]["dice"] is None
    assert rows[1]["dice"] == .8
    old.pop("spec")
    old["config"] = {"base_channels": 12}
    assert catalog(viewer)[0]["width"] == 12


def test_annotations_and_full_csv_export(tmp_path):
    viewer = SimpleNamespace()
    populate(viewer)
    row = catalog(viewer)[0]
    annotate(viewer, row, name="Candidate <A>", dataset_name="Reviewed set v1", notes="Narrow boundaries", tags="small, best, small", favorite=True)
    rows = catalog(viewer)
    assert rows[0]["notes"] == "Narrow boundaries"
    assert rows[0]["tags"] == "small, best"
    assert rows[1]["favorite"]
    assert all(r["dataset"] == "Reviewed set v1" for r in rows)
    path = tmp_path / "history.csv"
    export_csv(rows, path)
    saved = list(csv.DictReader(path.open()))
    assert len(saved) == 3
    assert saved[0]["dataset_id"] == "dataset-a"
    assert saved[0]["notes"] == "Narrow boundaries"


@pytest.mark.parametrize("kind", ["metrics", "size", "time"])
def test_charts_are_standalone_valid_svg(kind):
    viewer = SimpleNamespace()
    populate(viewer)
    rows = catalog(viewer)
    for row in rows:
        row["parameters"] = 1000
    svg = chart_svg(rows, kind=kind)
    ET.fromstring(svg)
    assert "Small U-Net / run 2" in svg
    assert "Parameters" in svg if kind == "size" else True
    ET.fromstring(chart_svg([], kind=kind))


def test_sidecar_curves_without_loading_weights(tmp_path):
    viewer = SimpleNamespace()
    first, _, _ = populate(viewer)
    row = catalog(viewer)[0]
    getattr(viewer, RUNTIME).clear()
    row["checkpoint"] = str(tmp_path / "model.pt")
    Path(row["checkpoint"]).with_suffix(".history.json").write_text(json.dumps({"history": first.history}))
    points = run_curve(viewer, row)
    assert points[0]["metrics"]["dice"] == .7
    assert points[0]["series"].startswith("Small U-Net / run 1 [")
    Path(row["checkpoint"]).with_suffix(".history.json").write_text("invalid json")
    assert run_curve(viewer, row) == []


def test_library_refresh_filters_and_documentation(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = ModelLibraryWidget(viewer)
    qtbot.addWidget(widget)
    assert widget.models.rowCount() == 0
    populate(viewer)
    assert widget.models.rowCount() == 2
    assert widget.history.rowCount() == 3
    widget.name.setText("Renamed model")
    widget.dataset_name.setText("Reviewed image revision 1")
    widget.notes.setPlainText("Useful for thin boundaries")
    widget.tags.setText("candidate, small")
    widget.favorite.setChecked(True)
    widget.save_notes.click()
    assert "Documentation updated" in widget.status.text()
    assert any(r["notes"] == "Useful for thin boundaries" for r in catalog(viewer))
    widget.favorites.setChecked(True)
    assert widget.models.rowCount() == 1
    widget.favorites.setChecked(False)
    widget.search.setText("Wide U-Net")
    assert widget.models.rowCount() == 1
    widget.search.clear()
    assert widget.history.rowCount() == 3
    widget.tabs.setCurrentIndex(1)
    widget.history.clearSelection()
    for row in range(widget.history.rowCount()):
        widget.history.selectionModel().select(widget.history.model().index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    widget.compare_button.click()
    assert widget.tabs.currentIndex() == 2
    assert "Best recorded" in widget.comparison_note.text()
    widget.chart_kind.setCurrentIndex(widget.chart_kind.findData("curves"))
    assert "polyline" in widget.chart.svg
    widget.resize(1000, 900)
    widget.show()
    qtbot.wait(20)
    assert widget.chart.geometry().bottom() < widget.legend.geometry().top()
    assert not widget.grab().isNull()


def test_documentation_workspace_roundtrip(make_napari_viewer, tmp_path, qtbot):
    torch = pytest.importorskip("torch")
    from napari_label_assistant_tools._model_providers import build_model
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), dtype=np.uint8))
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8))
    trained = model()
    trained.network = build_model(trained.config)
    task = add_task(viewer, "Myelin", "Sheath pixels")
    remember_model(viewer, trained, task_id=task["id"])
    annotate(viewer, catalog(viewer)[0], name="Saved candidate", dataset_name="Dataset v1", notes="Reviewed", tags="candidate", favorite=True)
    widget = ModelLibraryWidget(viewer)
    qtbot.addWidget(widget)
    path = tmp_path / "project.json"
    save_workspace(viewer, path, complete_package=True)
    load_workspace(viewer, path)
    rows = catalog(viewer)
    assert len(rows) == 1
    assert rows[0]["notes"] == "Reviewed"
    assert rows[0]["dataset"] == "Dataset v1"
    assert rows[0]["favorite"]
    assert rows[0]["status"] == "Saved"
    assert getattr(viewer, RUNTIME) == {}
    assert widget.models.rowCount() == 1
    assert run_curve(viewer, rows[0])


def test_weight_statistics_leave_parameters_unchanged():
    torch = pytest.importorskip("torch")
    network = torch.nn.Linear(1000, 100)
    before = network.weight.detach().clone()
    rows = weight_statistics(SimpleNamespace(network=network))
    assert rows[0]["parameters"] == 100000
    assert rows[0]["sampled"] <= 65536
    assert rows[0]["nonfinite_sampled"] == 0
    assert torch.equal(network.weight, before)
    network.weight = torch.nn.Parameter(network.weight.detach().T)
    rows = weight_statistics(SimpleNamespace(network=network))
    assert rows[0]["sampled"] <= 65536
    assert rows[0]["parameters"] == 100000


def test_plugin_menu_names():
    import yaml
    path = Path(__file__).parents[1] / "src/napari_label_assistant_tools/napari.yaml"
    contributions = yaml.safe_load(path.read_text())["contributions"]
    assert [w["display_name"] for w in contributions["widgets"]] == ["Label Curation", "Model Training", "Model Library"]
    assert contributions["widgets"][0]["command"] == "napari-label-assistant.label_assistant"


def test_library_selects_checkpoint_without_discarding_prepared_dataset(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    truth = (np.indices((128, 128))[1] % 16 < 8).astype(np.uint8)
    viewer.add_image(truth.astype(np.float32), name="EM")
    viewer.add_labels(truth, name="Curated")
    trainer = QuickTrainWidget(viewer)
    qtbot.addWidget(trainer)
    trainer.annotation_combo.setCurrentIndex(trainer.annotation_combo.findData("complete"))
    cfg = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4)
    monkeypatch.setattr(trainer, "_config", lambda: cfg)
    populate(viewer)
    trainer.prepare_button.click()
    qtbot.waitUntil(lambda: trainer.worker is None, timeout=15000)
    prepared = trainer.prepared
    assert prepared is not None
    widget = ModelLibraryWidget(viewer)
    qtbot.addWidget(widget)
    selected = widget.selected_rows()[0]
    widget.select_button.click()
    assert trainer.model is getattr(viewer, RUNTIME)[selected["model_id"] + ":" + selected["revision"]]
    assert trainer.prepared is prepared
    assert trainer.train_button.isEnabled()
    assert project(viewer)["current_checkpoint"] == selected["revision"]
    assert selected["model_name"] in widget.current_label.text()


def test_chart_exports_and_numeric_sort(make_napari_viewer, qtbot, monkeypatch, tmp_path):
    from napari_label_assistant_tools._model_library_widget import QFileDialog
    viewer = make_napari_viewer()
    populate(viewer)
    widget = ModelLibraryWidget(viewer)
    qtbot.addWidget(widget)
    widget.comparison_ids = [row["run_id"] for row in widget.rows]
    widget.tabs.setCurrentIndex(2)
    widget.draw_chart()
    for suffix in ("svg", "png"):
        path = tmp_path / f"chart.{suffix}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, p=path, **k: (str(p), ""))
        widget._export_chart()
        assert path.stat().st_size > 100
    column = [widget.history.horizontalHeaderItem(i).text() for i in range(widget.history.columnCount())].index("Dice")
    widget.history.sortItems(column, Qt.DescendingOrder)
    assert widget.history.item(0, column).text() == "0.8"
