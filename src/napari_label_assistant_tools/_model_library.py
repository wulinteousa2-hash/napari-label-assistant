"""Model/run catalog and exports, independent of loaded networks and Qt widgets."""
from copy import deepcopy
from html import escape
from pathlib import Path
import csv
import json
import math

from ._model_project import project, RUNTIME, notify_project

METRICS = ("dice", "iou", "precision", "recall", "loss")
FIELDS = ("model_name", "run", "date", "task", "dataset", "provider", "architecture",
          "encoder", "width", "initialization", "parameters", *METRICS, "validation",
          "epochs", "learning_rate", "batch_size", "crop_pixels", "input_pixels",
          "image", "annotations", "coverage", "training_patches", "validation_patches", "steps", "augmentation",
          "seconds", "device", "mode", "favorite", "tags", "notes", "status",
          "model_id", "run_id", "revision", "dataset_id", "checkpoint", "file_status", "reports")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def catalog(viewer):
    """One row per recorded checkpoint/run, never substitute latest-run scores."""
    state = project(viewer)
    tasks = {t["id"]: t["name"] for t in state["tasks"]}
    datasets = {d["id"]: d for d in state["datasets"]}
    runtime = getattr(viewer, RUNTIME, {})
    from ._model_files import workspace_locations, file_status
    models_folder = workspace_locations(viewer)[1]
    rows = []
    for model in state["models"]:
        runs = model.get("runs") or [{"id": model["id"], "revision": model["revision"],
                                      "history": (model.get("history") or [{}])[-1]}]
        for index, run in enumerate(runs, 1):
            history = run.get("history", {})
            latest = run["revision"] == model["revision"]
            config = run.get("config", history.get("config", {}))
            spec = run.get("spec") or config.get("model_spec")
            if not spec:
                spec = model["spec"] if latest else (
                    {"provider": "custom", "architecture": "unet", "width": config["base_channels"], "initialization": "random"}
                    if config.get("base_channels") else {})
            metrics = run.get("metrics", history.get("retained_validation", model.get("metrics", {}) if latest else {}))
            source = history.get("source", {})
            dataset_id = history.get("dataset_snapshot_id", "")
            dataset = datasets.get(dataset_id, {})
            validation = history.get("validation", {})
            method = validation.get("method", "sampled holdout" if history.get("crop_summary") else "unknown")
            checkpoint = run.get("checkpoint") or (model.get("checkpoint", "") if latest else "")
            loaded = model["id"] + ":" + run["revision"] in runtime or (latest and model["id"] in runtime)
            status = "In memory" if loaded else "Not saved"
            if checkpoint:
                status = "Saved" if Path(checkpoint).is_file() else "Missing checkpoint"
            if run.get("saved_revision", model.get("saved_revision") if latest else None) != run["revision"]:
                status += " · unsaved weights"
            dirty = run.get("saved_revision", model.get("saved_revision") if latest else None) != run["revision"]
            storage = file_status(checkpoint, dirty=dirty, models_folder=models_folder)
            row = {"model_id": model["id"], "model_name": model["name"], "run_id": run["id"],
                   "revision": run["revision"], "run": index, "date": history.get("started_at_utc", ""),
                   "task": tasks.get(model.get("task_id"), "Unassigned"), "task_id": model.get("task_id"),
                   "dataset_id": dataset_id, "dataset": dataset.get("name") or source.get("image_layer") or "Not recorded",
                   "provider": spec.get("provider", ""), "architecture": spec.get("architecture", ""),
                   "encoder": spec.get("encoder", "") if spec.get("provider") == "smp" else "",
                   "width": spec.get("width"), "initialization": spec.get("initialization", ""),
                   "parameters": run.get("parameters", model.get("parameters") if latest else None),
                   "validation": method, "epochs": history.get("epochs"), "steps": history.get("steps"),
                   "learning_rate": history.get("learning_rate"), "batch_size": config.get("batch_size"),
                   "crop_pixels": config.get("crop_pixels", history.get("crop_pixels")),
                   "input_pixels": config.get("input_pixels"), "seconds": history.get("duration_seconds"),
                   "device": history.get("device", ""), "mode": history.get("mode", "unknown"),
                   "favorite": bool(model.get("favorite")), "tags": ", ".join(model.get("tags", [])),
                   "notes": run.get("notes", ""), "status": status, "checkpoint": checkpoint,
                   "file_status": storage, "reports": ", ".join(suffix for suffix in (".history.json", ".metrics.csv", ".learning.svg", ".loss.svg") if checkpoint and Path(checkpoint).with_suffix(suffix).is_file()),
                   "history": deepcopy(history)}
            summary = history.get("dataset_summary", history.get("crop_summary", {}))
            total = history.get("total_pixels", 0)
            row.update(image=source.get("image_layer", ""), annotations=source.get("ground_truth_layer", ""),
                       coverage=history.get("reviewed_pixels", 0) / total if total else None,
                       training_patches=summary.get("training", {}).get("total"),
                       validation_patches=summary.get("validation", {}).get("total"),
                       augmentation=", ".join(label for key, label in (("augment_rotations", "rotation"), ("augment_flips", "flip"), ("augment_intensity", "intensity")) if config.get(key)))
            row.update({metric: metrics.get(metric) if number(metrics.get(metric)) else None for metric in METRICS})
            rows.append(row)
    return rows


def comparison_key(row):
    """Unknown provenance and final-all-data runs cannot be automatically ranked."""
    if not row.get("dataset_id") or row["validation"] in ("none", "unknown"):
        return None
    validation = row["history"].get("validation", {})
    return (row.get("task_id"), row["dataset_id"], row.get("crop_pixels"), row.get("input_pixels"),
            json.dumps(validation, sort_keys=True))


def comparison_note(rows, metric="dice"):
    if len(rows) < 2:
        return "Select at least two runs in Training History to compare."
    keys = {comparison_key(row) for row in rows}
    if None in keys or len(keys) != 1:
        return "Exploratory comparison: datasets or validation splits differ, or provenance is missing. No best-model ranking is assigned."
    scored = [r for r in rows if number(r.get(metric))]
    if not scored:
        return "Comparable recorded dataset and split; no recorded scores for this metric."
    best = (min if metric == "loss" else max)(scored, key=lambda r: r[metric])
    return (f"Best recorded validation {metric}: {best['model_name']} / run {best['run']} ({best[metric]:.4g}). "
            "Matching recorded dataset and split; validation-selected scores are not independent test results.")


def annotate(viewer, row, *, name, dataset_name, notes, tags, favorite):
    if not name.strip():
        raise ValueError("Give the model a name.")
    state = project(viewer)
    model = next(m for m in state["models"] if m["id"] == row["model_id"])
    model.update(name=name.strip(), tags=list(dict.fromkeys(t.strip() for t in tags.split(",") if t.strip())), favorite=bool(favorite))
    run = next((r for r in model.get("runs", []) if r["id"] == row["run_id"]), None)
    if run is not None:
        run["notes"] = notes
    if row.get("dataset_id"):
        dataset = next((d for d in state["datasets"] if d["id"] == row["dataset_id"]), None)
        if dataset is not None:
            dataset["name"] = dataset_name.strip()
    notify_project(viewer)


def export_csv(rows, path):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_curve(viewer, row):
    """Read curves from runtime or JSON sidecars; never deserialize weights."""
    from ._training_report import history_points
    runtime = getattr(viewer, RUNTIME, {})
    model = runtime.get(row["model_id"] + ":" + row["revision"])
    if model is not None:
        history = model.history
    elif row.get("checkpoint"):
        path = Path(row["checkpoint"]).with_suffix(".history.json")
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        history = payload.get("history", []) if isinstance(payload, dict) else []
    else:
        return []
    # Each checkpoint carries cumulative history; show only its own training run.
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        return []
    points = history_points(history[-1:])
    for point in points:
        point["series"] = f"{row['model_name']} / run {row['run']} [{row['model_id'][:6]}]"
    return points


def chart_svg(rows, *, kind="metrics", metric="dice"):
    """Standalone SVG for selected-run metrics, model size, or run chronology."""
    width, height = 800, 420 + len(rows) * 18
    scored = [r for r in rows if number(r.get(metric)) and (kind != "size" or number(r.get("parameters")))]
    if kind == "time":
        scored = sorted(scored, key=lambda r: r.get("date", ""))
    maximum = max([r[metric] for r in scored] + [1]) if metric == "loss" else 1
    xmax = max([r["parameters"] for r in scored] + [1]) if kind == "size" else max(1, len(scored))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="800" height="{height}" fill="#fafafa"/>',
           f'<text x="55" y="28" fill="#333" font-size="17">{escape(metric.title())} — {escape({"metrics":"selected runs", "size":"model size", "time":"runs in date order"}[kind])}</text>']
    for tick in range(6):
        y = 300 - tick * 48
        svg.append(f'<path d="M55 {y} H765" stroke="#ddd"/><text x="5" y="{y+4}" font-size="11">{maximum*tick/5:.3g}</text>')
    colors = ("#147d64", "#2468ac", "#b65b18", "#884eb5", "#ad3f69")
    for index, row in enumerate(scored):
        x = 65 + 680 * (row["parameters"] / xmax if kind == "size" else (index + .5) / xmax)
        y = 300 - 240 * row[metric] / max(maximum, 1e-6)
        color = colors[index % len(colors)]
        label = escape(f"{row['model_name']} / run {row['run']}")
        if kind == "metrics":
            bar = min(55, 550 / max(len(scored), 1))
            svg.append(f'<rect x="{x-bar/2:.2f}" y="{y:.2f}" width="{bar:.2f}" height="{300-y:.2f}" fill="{color}"><title>{label}: {row[metric]:.5g}</title></rect>')
        else:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"><title>{label}: {row[metric]:.5g}</title></circle>')
        svg.append(f'<text x="{x:.2f}" y="{max(45,y-8):.2f}" font-size="10" text-anchor="middle">{index+1}</text>')
    axis = f"Parameters (0–{xmax:,})" if kind == "size" else "Runs (numbers match the selected-run list)"
    svg.append(f'<text x="55" y="333" font-size="12">{escape(axis)}</text>')
    note = "No recorded scores for this selection." if not scored else "Validation scores; compare matching datasets and evaluation conditions."
    svg.append(f'<text x="55" y="370" font-size="12">{note}</text>')
    for index, row in enumerate(scored, 1):
        label = escape(f"{index}. {row['model_name']} / run {row['run']} · {row.get('date') or 'date unknown'} · {metric} {row[metric]:.5g}")
        svg.append(f'<text x="55" y="{400+18*(index-1)}" font-size="11">{label}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def weight_statistics(model):
    """Inspect bounded samples of already-loaded tensors without changing weights."""
    import torch
    rows = []
    with torch.no_grad():
        for name, tensor in model.network.named_parameters():
            count = tensor.numel()
            stride = max(1, (count + 65535) // 65536)
            if tensor.is_contiguous():
                sample = tensor.detach().view(-1)[::stride][:65536]
            else:
                # Reshaping a noncontiguous custom-model tensor could copy the
                # entire parameter; construct only the sampled coordinates.
                indices = torch.arange(0, count, stride, device=tensor.device)[:65536]
                coordinates = []
                for dimension in reversed(tensor.shape):
                    coordinates.insert(0, indices % dimension)
                    indices = torch.div(indices, dimension, rounding_mode="floor")
                sample = tensor.detach()[tuple(coordinates)]
            sample = sample.float().cpu()
            finite = sample[torch.isfinite(sample)]
            rows.append({"layer": name, "shape": str(tuple(tensor.shape)), "parameters": count,
                         "sampled": sample.numel(), "nonfinite_sampled": sample.numel()-finite.numel(),
                         "mean": finite.mean().item() if finite.numel() else None,
                         "std": finite.std(unbiased=False).item() if finite.numel() else None,
                         "min": finite.min().item() if finite.numel() else None,
                         "max": finite.max().item() if finite.numel() else None})
    return rows
