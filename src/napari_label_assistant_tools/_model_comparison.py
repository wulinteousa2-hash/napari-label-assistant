"""Sequential, controlled architecture comparisons on one indexed dataset."""
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256, blake2b
import json
from datetime import datetime, timezone
from ._quick_train import QuickTrainProgress, _check_cancel
from ._coverage_train import train_indexed, train_kfold
from ._model_providers import ModelSpec

@dataclass
class ComparisonResult:
    models: list
    report: dict


def dataset_snapshot(dataset, source=None):
    snapshot = {"source": deepcopy(source or {}), "image_shape": list(dataset.image.shape),
                "crop_pixels": dataset.crop_pixels, "input_pixels": dataset.input_pixels,
                "locations": deepcopy(dataset.locations), "validation": deepcopy(dataset.validation)}
    snapshot["id"] = sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return snapshot


def compare_models(dataset, config, specs, *, cancel=None, source_info=None, **settings):
    if len(specs) < 2:
        raise ValueError("Select at least two model configurations to compare.")
    # Fingerprint only eligible source patches; never materialize the image.
    dataset = replace(dataset, locations=deepcopy(dataset.locations))
    import numpy as np
    for number, location in enumerate(dataset.locations, 1):
        _check_cancel(cancel)
        region = (slice(location["y0"], location["y1"]), slice(location["x0"], location["x1"]))
        raw = np.asarray(dataset.image[region])
        location["image_digest"] = blake2b(raw.tobytes(), digest_size=16).hexdigest()
        if number % max(1, len(dataset.locations)//100) == 0:
            yield QuickTrainProgress(0, len(specs)*1000, f"Checking comparison inputs: {number}/{len(dataset.locations)} eligible patches.")
    configs = [replace(config, model_spec=spec) for spec in specs]
    for candidate in configs:
        candidate.validate()
    report = {"id": __import__("uuid").uuid4().hex,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": dataset_snapshot(dataset, source_info), "settings": deepcopy(settings), "results": [],
              "note": "Shared indexed patches and spatial splits; fresh weights for every candidate and fold. Validation-selected scores are not independent test scores."}
    models = []
    kfold = dataset.validation.get("method") == "kfold"
    for number, candidate in enumerate(configs, 1):
        _check_cancel(cancel)
        function = train_kfold if kfold else train_indexed
        generator = function(dataset, candidate, cancel=cancel, source_info=source_info, **settings)
        while True:
            try:
                update = next(generator)
            except StopIteration as done:
                result = done.value
                break
            fraction = update.completed/max(1, update.total)
            yield QuickTrainProgress(int((number-1+fraction)*1000), len(configs)*1000,
                f"Candidate {number}/{len(configs)} · {update.message}", metrics=update.metrics,
                epoch=update.epoch, fold=update.fold, training_loss_last_batch=update.training_loss_last_batch,
                training_step=update.training_step, series=f"Candidate {number}" + (f" / Fold {update.fold}" if update.fold else ""))
        candidate_models = result.models if kfold else [result]
        row = {"spec": candidate.model_spec, "model_indices": list(range(len(models), len(models)+len(candidate_models))),
               "parameters": sum(p.numel() for p in candidate_models[0].network.parameters()),
               "seconds": sum(m.history[-1].get("duration_seconds", 0) for m in candidate_models),
               "metrics": deepcopy(result.report["aggregate"]) if kfold else {k: {"mean": v, "std": None} for k,v in result.metrics.items()},
               "validation": deepcopy(result.report) if kfold else deepcopy(dataset.validation)}
        report["results"].append(row)
        for model in candidate_models:
            model.history[-1]["comparison_id"] = report["id"]
            model.history[-1]["dataset_snapshot_id"] = report["dataset"]["id"]
        models.extend(candidate_models)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return ComparisonResult(models, report)


def export_comparison(report, path):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fields = ["comparison_id", "dataset_id", "provider", "architecture", "encoder", "initialization", "parameters", "seconds"]
        fields += [f"{metric}_{stat}" for metric in ("dice", "iou", "precision", "recall", "loss") for stat in ("mean", "std")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            row = {"comparison_id": report["id"], "dataset_id": report["dataset"]["id"],
                   **{key: result["spec"][key] for key in ("provider", "architecture", "encoder", "initialization")},
                   "parameters": result["parameters"], "seconds": result["seconds"]}
            for metric, values in result["metrics"].items():
                for stat, value in values.items():
                    if f"{metric}_{stat}" in fields:
                        row[f"{metric}_{stat}"] = value
            writer.writerow(row)
