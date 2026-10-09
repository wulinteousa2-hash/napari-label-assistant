"""Workspace-owned task/model metadata; networks stay outside the JSON manifest."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
from uuid import uuid4

KEY = "_label_assistant_model_project"
RUNTIME = "_label_assistant_project_models"

def project(viewer):
    state = getattr(viewer, KEY, None)
    if state is None:
        state = {"version": 1, "tasks": [], "models": [], "comparisons": [], "current_task": None, "current_model": None}
        setattr(viewer, KEY, state)
    state.setdefault("configurations", [])
    state.setdefault("datasets", [])
    if not hasattr(viewer, RUNTIME):
        setattr(viewer, RUNTIME, {})
    return state


def layer_id(layer):
    if layer is None:
        return None
    metadata = dict(layer.metadata.get("label_assistant_review", {}))
    if not metadata.get("id"):
        metadata["id"] = uuid4().hex
        layer.metadata = {**layer.metadata, "label_assistant_review": metadata}
    return metadata["id"]


def add_task(viewer, name, definition):
    if not name.strip() or not definition.strip():
        raise ValueError("Give the task a name and define which pixels count as the target.")
    state = project(viewer)
    task = {"id": uuid4().hex, "name": name.strip(), "target_definition": definition.strip(), "sources": {}}
    state["tasks"].append(task)
    state["current_task"] = task["id"]
    return task


def remember_model(viewer, model, *, task_id=None, path=None, select=True, name=None):
    from ._model_providers import spec_for
    state = project(viewer)
    identity = model.identity
    identifier = identity.get("model_id") or uuid4().hex
    identity.update(model_id=identifier, task_id=identity.get("task_id") or task_id)
    task = next((t for t in state["tasks"] if t["id"] == identity.get("task_id")), None)
    if task is None and identity.get("task_id"):
        source = model.history[-1].get("source", {}) if model.history else {}
        task = {"id": identity["task_id"], "name": source.get("task_name", "Imported segmentation task"),
                "target_definition": source.get("target_definition", "See imported model training history."), "sources": {}}
        state["tasks"].append(task)
    # Every update is a new run/checkpoint revision; previous saved files remain valid.
    revision = uuid4().hex
    row = next((r for r in state["models"] if r["id"] == identifier), None)
    if row is None:
        row = {"id": identifier, "name": name or spec_for(model.config).label(), "runs": []}
        state["models"].append(row)
    from hashlib import sha256
    spec = spec_for(model.config).to_dict()
    configuration_id = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if not any(c["id"] == configuration_id for c in state["configurations"]):
        state["configurations"].append({"id": configuration_id, "spec": spec})
    for history in model.history:
        dataset_id = history.get("dataset_snapshot_id")
        if dataset_id and not any(d["id"] == dataset_id for d in state["datasets"]):
            state["datasets"].append({"id": dataset_id, "source": history_summary(history).get("source", {}),
                "summary": history.get("dataset_summary", {}), "crop_pixels": model.config.crop_pixels,
                "input_pixels": model.config.input_pixels})
    row.update(configuration_id=configuration_id, task_id=identity.get("task_id"), spec=spec,
               metrics=dict(model.metrics), config=asdict(model.config), revision=revision,
               parameters=sum(p.numel() for p in model.network.parameters()) if model.network is not None else 0, history=[history_summary(h) for h in model.history])
    if model.history:
        run = {"id": uuid4().hex, "revision": revision, "history": history_summary(model.history[-1])}
        if path:
            run.update(checkpoint=str(Path(path).resolve()), saved_revision=revision)
        row["runs"].append(run)
    if path:
        row["checkpoint"] = str(Path(path).resolve())
        row["saved_revision"] = revision
    getattr(viewer, RUNTIME)[identifier] = model
    getattr(viewer, RUNTIME)[identifier + ":" + revision] = model
    if select:
        state["current_model"] = identifier
        state["current_checkpoint"] = revision
    return row


def save_project(viewer, manifest, data_root, complete_package):
    from ._quick_train import save_model
    widget = getattr(viewer, "_label_assistant_training_widget", None)
    if widget is not None:
        project(viewer)["settings"] = widget.project_settings()
    state = deepcopy(project(viewer))
    runtime = getattr(viewer, RUNTIME)
    folder = data_root / "models"
    def persist(entry, identifier, model):
        revision = entry.get("revision", "")
        if not isinstance(identifier, str) or not identifier.isalnum() or not isinstance(revision, str) or not revision.isalnum():
            raise ValueError("Invalid model identifier or checkpoint revision in project.")
        source = Path(entry["checkpoint"]) if entry.get("checkpoint") else None
        dirty = entry.get("saved_revision") != revision
        owned = source is not None and source.resolve().is_relative_to(data_root.resolve())
        if dirty or complete_package or owned:
            if source is None or not source.is_file() or dirty:
                if model is None:
                    raise ValueError("Cannot save updated model: checkpoint weights are unavailable.")
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / f"{identifier}-{revision}.pt"
                save_model(model, destination)
                source = destination
            elif complete_package and not owned:
                import shutil
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / f"{identifier}-{revision}.pt"
                if source.resolve() != destination.resolve():
                    for suffix in ('.pt', '.history.json', '.learning.svg', '.loss.svg', '.metrics.csv'):
                        item = source.with_suffix(suffix)
                        if item.is_file():
                            temporary = destination.with_suffix(suffix + '.' + uuid4().hex + '.tmp')
                            try:
                                shutil.copy2(item, temporary)
                                os.replace(temporary, destination.with_suffix(suffix))
                            finally:
                                temporary.unlink(missing_ok=True)
                source = destination
            entry["saved_revision"] = revision
        elif source is None:
            raise ValueError("Model has no saved checkpoint.")
        if source is not None:
            if complete_package and not source.is_file():
                raise FileNotFoundError(f"Cannot package missing checkpoint: {source}")
            entry["checkpoint"] = os.path.relpath(source.resolve(), manifest.parent.resolve())

    for row in state["models"]:
        identifier = row["id"]
        persist(row, identifier, runtime.get(identifier))
        for run in row.get("runs", []):
            if run["revision"] == row["revision"]:
                run.update(checkpoint=row["checkpoint"], saved_revision=row["saved_revision"])
            else:
                persist(run, identifier, runtime.get(identifier + ":" + run["revision"]))
    reports_folder = data_root / "comparisons"
    for report in state["comparisons"]:
        identifier = report.get("id", "")
        if not isinstance(identifier, str) or not identifier.isalnum():
            raise ValueError("Invalid comparison identifier.")
        if "locations" in report.get("dataset", {}):
            reports_folder.mkdir(parents=True, exist_ok=True)
            destination = reports_folder / (identifier + ".json")
            temporary = destination.with_suffix("." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            report["report_file"] = os.path.relpath(destination, manifest.parent)
            report["dataset"] = {key: value for key, value in report["dataset"].items() if key != "locations"}
            for result in report["results"]:
                result["validation"] = {key: value for key,value in result["validation"].items()
                                        if key not in ("locations", "fold_results", "folds")}
        elif report.get("report_file"):
            source = Path(report["report_file"])
            if complete_package and not source.is_file():
                raise FileNotFoundError("Comparison report unavailable: " + str(source))
            if complete_package:
                import shutil
                reports_folder.mkdir(parents=True, exist_ok=True)
                destination = reports_folder / (identifier + ".json")
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                source = destination
            report["report_file"] = os.path.relpath(source.resolve(), manifest.parent)
    return state


def restore_project(viewer, state, manifest):
    restored = deepcopy(state or {"version": 1, "tasks": [], "models": [], "comparisons": [], "current_task": None, "current_model": None})
    validate_project_state(restored)
    resolve_references(restored, manifest)
    for index, report in enumerate(restored["comparisons"]):
        if report.get("report_file"):
            try:
                full = json.loads(Path(report["report_file"]).read_text(encoding="utf-8"))
                if isinstance(full, dict) and full.get("id") == report["id"] and isinstance(full.get("dataset"), dict) and isinstance(full.get("results"), list):
                    full["report_file"] = report["report_file"]
                    restored["comparisons"][index] = full
            except (OSError, ValueError):
                report["report_unavailable"] = True
    setattr(viewer, KEY, restored)
    setattr(viewer, RUNTIME, {})
    widget = getattr(viewer, "_label_assistant_training_widget", None)
    if widget is not None:
        widget.model = None
        widget.cross_validation = None
        widget._invalidate_crops()
        widget.refresh_layers()
        widget.models_panel.refresh()
        widget.restore_project_settings(restored.get("settings", {}))


def resolve_references(state, manifest):
    for row in state["models"]:
        for entry in [row, *row.get("runs", [])]:
            if entry.get("checkpoint"):
                entry["checkpoint"] = str((manifest.parent / entry["checkpoint"]).resolve())
    for report in state["comparisons"]:
        if report.get("report_file"):
            report["report_file"] = str((manifest.parent / report["report_file"]).resolve())
    return state


def history_summary(record):
    """Keep manifests small; full coordinates and curves remain in checkpoints."""
    result = {key: deepcopy(value) for key, value in record.items()
              if key not in ("locations", "checkpoints", "cross_validation_report", "validation_blocks")}
    if "validation" in result:
        result["validation"] = {key: value for key, value in result["validation"].items()
                                if key not in ("folds", "validation_blocks")}
    library = result.get("source", {}).get("component_library")
    if library and "selected_ids" in library:
        library["selected_count"] = len(library.pop("selected_ids"))
    result["evaluated_checkpoints"] = len(record.get("checkpoints", []))
    return result


def validate_project_state(state):
    """Reject malformed project metadata before replacing the viewer's layers."""
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Unsupported model-project metadata.")
    for key in ("tasks", "models", "comparisons"):
        if not isinstance(state.get(key), list) or any(not isinstance(row, dict) for row in state[key]):
            raise ValueError(f"Invalid model-project {key} records.")
    for task in state["tasks"]:
        if any(not isinstance(task.get(key), str) for key in ("id", "name", "target_definition")) or not isinstance(task.get("sources", {}), dict):
            raise ValueError("Invalid biological-task record.")
    for row in state["models"]:
        if not isinstance(row.get("id"), str) or not isinstance(row.get("name"), str) or not isinstance(row.get("revision"), str) or not isinstance(row.get("spec"), dict):
            raise ValueError("Invalid model record.")
        if any(key not in row["spec"] for key in ("provider", "architecture", "encoder", "initialization", "width", "preprocessing")):
            raise ValueError("Incomplete model specification.")
        if not isinstance(row.get("metrics", {}), dict) or not isinstance(row.get("config", {}), dict):
            raise ValueError("Invalid model summary.")
        if not isinstance(row.get("runs", []), list) or not isinstance(row.get("history", []), list) or any(not isinstance(h, dict) for h in row.get("history", [])):
            raise ValueError("Invalid model history.")
        for run in row.get("runs", []):
            if not isinstance(run, dict) or not isinstance(run.get("id"), str) or not isinstance(run.get("revision"), str) or not isinstance(run.get("history"), dict):
                raise ValueError("Invalid checkpoint run.")
        if row.get("checkpoint") is not None and not isinstance(row["checkpoint"], str):
            raise ValueError("Invalid checkpoint path.")
    for report in state["comparisons"]:
        if not isinstance(report.get("id"), str) or not isinstance(report.get("dataset"), dict) or not isinstance(report.get("results"), list):
            raise ValueError("Invalid comparison report.")
    if not isinstance(state.get("settings", {}), dict):
        raise ValueError("Invalid training settings.")
