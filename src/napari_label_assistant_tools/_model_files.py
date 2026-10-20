"""Workspace-owned checkpoint bundles and identity-preserving file references."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import os
import shutil
import errno

from ._model_project import project, notify_project, history_summary

WORKSPACE_PATH = "_label_assistant_workspace_path"
WORKSPACE_LAYERS = "_label_assistant_workspace_layer_ids"
BUNDLE_SUFFIXES = (".pt", ".history.json", ".learning.svg", ".loss.svg", ".metrics.csv")


def workspace_locations(viewer):
    """Return only the active workspace, never a recent-workspace preference."""
    value = getattr(viewer, WORKSPACE_PATH, "")
    if not value:
        return None, None
    from .workspace.service import workspace_data_root
    manifest = Path(value).expanduser().resolve()
    return manifest, workspace_data_root(manifest) / "models"


def associate_workspace(viewer, manifest):
    setattr(viewer, WORKSPACE_PATH, str(Path(manifest).expanduser().resolve()))
    setattr(viewer, WORKSPACE_LAYERS, {id(layer) for layer in getattr(viewer, "layers", ())})


def file_status(checkpoint, *, dirty, models_folder):
    if dirty or not checkpoint:
        return "Unsaved"
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        return "Missing"
    if models_folder is not None and path.is_relative_to(models_folder.resolve()):
        return "Managed"
    return "Linked externally"


def entry_for(viewer, row):
    model = next((m for m in project(viewer)["models"] if m["id"] == row["model_id"]), None)
    if model is None:
        raise ValueError("This model is no longer in the current workspace. Refresh the library.")
    entry = next((r for r in model.get("runs", []) if r["id"] == row["run_id"]), None)
    if entry is None and row["run_id"] == model["id"]:
        entry = model
    if entry is None or entry["revision"] != row["revision"]:
        raise ValueError("This checkpoint changed. Refresh the library before managing its files.")
    return model, entry


def update_reference(viewer, row, path, *, expected_state, expected_workspace):
    """Commit on the UI thread only if the operation still belongs to this project."""
    if project(viewer) is not expected_state or workspace_locations(viewer)[0] != expected_workspace:
        raise ValueError(f"The workspace changed during the operation. No model reference was changed. The file is at {path}.")
    model, entry = entry_for(viewer, row)
    latest = model["revision"] == row["revision"]
    previous = entry.get("checkpoint") or (model.get("checkpoint", "") if latest else "")
    if previous != row["checkpoint"]:
        raise ValueError("The checkpoint reference changed during the operation. Refresh the library.")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("The selected checkpoint is no longer available.")
    entry.update(checkpoint=str(path), saved_revision=row["revision"])
    if latest:
        model.update(checkpoint=str(path), saved_revision=row["revision"])
    notify_project(viewer)


def copy_checkpoint_bundle(source, models_folder, model_id, revision):
    """Stage all companion files, then publish; never replace an existing bundle."""
    source = Path(source).expanduser().resolve()
    folder = Path(models_folder).expanduser().resolve()
    if not source.is_file():
        raise ValueError("The checkpoint is missing. Locate it before copying it into the workspace.")
    if source.suffix.lower() != ".pt":
        raise ValueError("Choose a plugin model checkpoint (*.pt).")
    if not model_id.isalnum() or not revision.isalnum():
        raise ValueError("Invalid model or checkpoint identifier.")
    if source.is_relative_to(folder):
        return str(source)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{model_id}-{revision}.pt"
    while any(destination.with_suffix(suffix).exists() for suffix in BUNDLE_SUFFIXES):
        destination = folder / f"{model_id}-{revision}-{uuid4().hex[:12]}.pt"
    published = []
    try:
        with TemporaryDirectory(prefix=".checkpoint-copy-", dir=folder) as staging:
            staged = []
            for suffix in BUNDLE_SUFFIXES:
                original = source if suffix == ".pt" else source.with_suffix(suffix)
                if not original.is_file():
                    continue
                before = original.stat()
                temporary = Path(staging) / ("checkpoint" + suffix)
                shutil.copy2(original, temporary)
                after = original.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("A model file changed during copying. Save training first, then retry.")
                staged.append((temporary, destination.with_suffix(suffix)))
            for temporary, target in staged:
                # Hard-link publication fails rather than overwriting a file
                # created by another writer after the destination check.
                try:
                    os.link(temporary, target)
                except OSError as error:
                    if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS):
                        raise
                    # Some external/network filesystems do not support links.
                    # Exclusive creation still prevents replacing another file.
                    with target.open("xb") as output:
                        published.append(target)
                        with temporary.open("rb") as source_stream:
                            shutil.copyfileobj(source_stream, output)
                    shutil.copystat(temporary, target)
                else:
                    published.append(target)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    return str(destination)


def verify_relocated_checkpoint(path, row, expected_config):
    """Read checkpoint metadata without constructing a network or loading GPU tensors."""
    import torch
    from ._quick_train import QuickTrainConfig
    from ._model_providers import spec_for
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Choose an available checkpoint file.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        raise ValueError(f"Could not verify this checkpoint: {error}") from error
    if not isinstance(payload, dict) or payload.get("quick_train_version") not in (1, 2):
        raise ValueError("This is not a supported plugin checkpoint. Import the correct model instead.")
    identity = payload.get("identity", {})
    if identity.get("model_id") and identity["model_id"] != row["model_id"]:
        raise ValueError("This file belongs to another model. Import it instead of relinking this run.")
    if identity.get("task_id") and row.get("task_id") and identity["task_id"] != row["task_id"]:
        raise ValueError("This file belongs to another task.")
    def normalized(config):
        value = QuickTrainConfig(**config)
        result = asdict(value)
        result["model_spec"] = spec_for(value).to_dict()
        return result
    if not expected_config or normalized(payload.get("config", {})) != normalized(expected_config):
        raise ValueError("The checkpoint configuration does not match this recorded run.")
    expected_history = dict(row.get("history", {}))
    expected_history.setdefault("evaluated_checkpoints", 0)
    history = payload.get("history", [])
    if row.get("history"):
        if not history or history_summary(history[-1]) != expected_history:
            raise ValueError("This checkpoint's training history does not match the selected run. Choose the matching checkpoint revision.")
    elif not identity.get("model_id"):
        raise ValueError("This legacy checkpoint has no identifying history. Import it as a model instead of relinking a run.")
    metrics = payload.get("metrics", {})
    for metric in ("dice", "iou", "precision", "recall", "loss"):
        if row.get(metric) is not None and metrics.get(metric) != row[metric]:
            raise ValueError("The checkpoint scores do not match the selected run.")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("The checkpoint contains no model weights.")
    return str(path)
