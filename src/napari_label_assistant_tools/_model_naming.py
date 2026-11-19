"""Readable checkpoint names derived from metadata, without loading weights."""
from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

BUNDLE_SUFFIXES = (".pt", ".history.json", ".learning.svg", ".loss.svg", ".metrics.csv")


def _part(value, limit=40):
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:limit].rstrip("-")


def checkpoint_filename(record, entry=None):
    """Use a model record and optional historical run; preserve its provenance."""
    entry = record if entry is None else entry
    revision = entry.get("revision", record.get("revision", ""))
    latest = revision == record.get("revision")
    history = entry.get("history", {})
    rounds = len(history) if isinstance(history, list) else 0
    if isinstance(history, list):
        history = history[-1] if history else {}
    config = entry.get("config") or history.get("config") or (record.get("config", {}) if latest else {})
    spec = entry.get("spec") or config.get("model_spec") or (record.get("spec", {}) if latest else {})
    source = history.get("source", {})
    parts = [_part(record.get("name") or source.get("task_name")) or "model"]
    image = source.get("image_layer") or ("multiple-images" if source.get("pairs") else "")
    if image:
        parts.append(_part(image, 32) or "image")
    memberships = [b for b in source.get('annotation_batches', []) if b.get('scope') == 'active']
    if len(memberships) == 1 and len(source.get('annotation_batches', [])) == 1:
        parts.append(_part(memberships[0]['name'], 24) or 'batch')
    provider = spec.get("provider", "custom")
    architecture = spec.get("architecture", "unet")
    if provider != "custom":
        architecture = f"{provider}-{architecture}"
    if provider == "smp":
        architecture += "-" + spec.get("encoder", "encoder")
    else:
        width = spec.get("width", config.get("base_channels"))
        if width is not None:
            architecture += f"-w{width}"
    parts.append(_part(architecture, 48) or "network")
    if config.get("input_pixels"):
        parts.append("in" + _part(config["input_pixels"], 8))
    run = next((i for i, row in enumerate(record.get("runs", []), 1)
                if row.get("revision") == revision), max(1, rounds))
    parts.append(f"run{run:02d}")
    try:
        date = datetime.fromisoformat(str(history.get("started_at_utc") or "").replace("Z", "+00:00"))
        if date.tzinfo is not None:
            date = date.astimezone(timezone.utc)
        parts.append(date.strftime("%Y%m%d"))
    except (ValueError, TypeError):
        parts.append("undated")
    parts.append(_part(revision, 8) or uuid4().hex[:8])
    return "_".join(parts) + ".pt"


def available_checkpoint_path(folder, filename):
    """Avoid collisions with weights or any companion report already on disk."""
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".pt":
        raise ValueError("Choose a checkpoint filename (*.pt), without a directory.")
    destination = Path(folder) / filename
    while any((destination if suffix == ".pt" else destination.with_suffix(suffix)).exists()
              for suffix in BUNDLE_SUFFIXES):
        destination = Path(folder) / f"{Path(filename).stem}-{uuid4().hex[:12]}{Path(filename).suffix}"
    return destination
