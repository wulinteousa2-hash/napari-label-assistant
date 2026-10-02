"""Bounded-memory U-Net training. PyTorch is imported only when requested."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from threading import Event
from typing import Any

import numpy as np
from scipy.ndimage import find_objects, label, zoom


class QuickTrainCancelled(Exception):
    """Cooperative cancellation between crops, training steps, and predictions."""


@dataclass(frozen=True)
class QuickTrainConfig:
    crop_pixels: int = 512
    input_pixels: int = 512
    samples: int = 32
    steps: int = 200
    batch_size: int = 2
    base_channels: int = 8
    device: str = "auto"
    seed: int = 42

    def validate(self) -> None:
        if not 32 <= self.crop_pixels <= 4096:
            raise ValueError("Crop size must be between 32 and 4096 pixels.")
        if not 32 <= self.input_pixels <= 512 or self.input_pixels % 16:
            raise ValueError("Model input size must be a multiple of 16, between 32 and 512.")
        if not 8 <= self.samples <= 256:
            raise ValueError("Choose between 8 and 256 training crops.")
        if not 1 <= self.steps <= 10000:
            raise ValueError("Choose between 1 and 10,000 training steps.")
        if not 1 <= self.batch_size <= 8:
            raise ValueError("Batch size must be between 1 and 8.")
        if self.base_channels not in (4, 8, 12, 16):
            raise ValueError("Unsupported U-Net channel width.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Choose Automatic, CPU, or CUDA.")


@dataclass
class QuickTrainProgress:
    completed: int
    total: int
    message: str


@dataclass
class PairedCrops:
    images: np.ndarray
    masks: np.ndarray
    validation_images: np.ndarray
    validation_masks: np.ndarray
    crop_pixels: int
    locations: list[dict] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)


@dataclass
class CropAtlas:
    image: np.ndarray
    mask: np.ndarray
    locations: list[dict]
    columns: int
    stride_y: int
    stride_x: int
    header: int


@dataclass
class QuickTrainModel:
    network: Any
    config: QuickTrainConfig
    crop_pixels: int
    metrics: dict[str, float]
    training_device: str
    locations: list[dict] = field(default_factory=list)
    training_steps: int = 0
    training_rounds: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class QuickTrainPrediction:
    probability: np.ndarray
    origin: tuple[int, int]


def _check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise QuickTrainCancelled()


def array_2d(data):
    """Resolve the highest-resolution multiscale level without materializing it."""
    # Newer napari uses MultiScaleData, a Sequence with an array-like shape.
    # Its tuple indexing still indexes the underlying list of pyramid levels.
    while isinstance(data, Sequence) and not isinstance(data, (str, bytes, np.ndarray)) and len(data):
        data = data[0]
    shape = getattr(data, "shape", ())
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Choose a single-channel 2D image and a 2D Labels layer.")
    return data


def _resize(array: np.ndarray, shape: tuple[int, int], *, labels=False):
    if array.shape == shape:
        return array.astype(np.float32, copy=True)
    if all(out <= source for out, source in zip(shape, array.shape)):
        # Preserve even one-pixel targets when reducing a binary mask. Each
        # model pixel represents the maximum foreground value in its source bin.
        edges_y = np.linspace(0, array.shape[0], shape[0] + 1).astype(int)
        edges_x = np.linspace(0, array.shape[1], shape[1] + 1).astype(int)
        reducer = np.maximum if labels else np.add
        result = reducer.reduceat(
            reducer.reduceat(array.astype(np.float32), edges_y[:-1], axis=0),
            edges_x[:-1], axis=1,
        )
        if not labels:
            result /= np.diff(edges_y)[:, None] * np.diff(edges_x)[None, :]
        return result.astype(np.float32)
    return zoom(
        array.astype(np.float32),
        (shape[0] / array.shape[0], shape[1] / array.shape[1]),
        order=0 if labels else 1,
        prefilter=False,
    )


def _normalize_crop(array: np.ndarray) -> np.ndarray:
    """Identical percentile normalization for training and prediction crops."""
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("The selected image contains NaN or infinite pixels.")
    low, high = np.percentile(array, (1, 99))
    return np.clip((array - low) / max(float(high - low), 1e-6), 0, 1)


def _scan_annotations(ground_truth, crop_pixels, rng, cancel):
    """Find sparse targets at full resolution, retaining bounded anchor lists."""
    height, width = map(int, ground_truth.shape)
    side = min(max(512, 2 * crop_pixels), max(height, width) // 2)
    rows, cols = max(1, height // side), max(1, width // side)
    ys = np.linspace(0, height, rows + 1).astype(int)
    xs = np.linspace(0, width, cols + 1).astype(int)
    blocks = []
    for row in range(rows):
        for col in range(cols):
            blocks.append({
                "id": len(blocks), "y0": int(ys[row]), "y1": int(ys[row + 1]),
                "x0": int(xs[col]), "x1": int(xs[col + 1]),
                "anchors": [], "seen": 0,
            })
    scan_side = 2048
    total = sum(
        ((b["y1"] - b["y0"] + scan_side - 1) // scan_side)
        * ((b["x1"] - b["x0"] + scan_side - 1) // scan_side)
        for b in blocks
    )
    done = 0
    for block in blocks:
        for y0 in range(block["y0"], block["y1"], scan_side):
            for x0 in range(block["x0"], block["x1"], scan_side):
                _check_cancel(cancel)
                region = (
                    slice(y0, min(y0 + scan_side, block["y1"])),
                    slice(x0, min(x0 + scan_side, block["x1"])),
                )
                foreground = np.asarray(ground_truth[region]) != 0
                components, _ = label(foreground)
                for component_id, box in enumerate(find_objects(components), 1):
                    if box is None:
                        continue
                    _check_cancel(cancel)
                    pixels = np.argwhere(components[box] == component_id)
                    point = pixels[len(pixels) // 2]
                    anchor = (y0 + box[0].start + int(point[0]), x0 + box[1].start + int(point[1]))
                    block["seen"] += 1
                    if len(block["anchors"]) < 16:
                        block["anchors"].append(anchor)
                    else:
                        slot = int(rng.integers(block["seen"]))
                        if slot < 16:
                            block["anchors"][slot] = anchor
                done += 1
                yield QuickTrainProgress(done, total, f"Scanning ground truth for sparse targets: {done}/{total} tiles.")
    annotated = [b for b in blocks if b["anchors"]]
    if not annotated:
        raise ValueError("No labeled targets found in the ground truth.")
    if len(annotated) < 2:
        raise ValueError("Targets occupy only one spatial block. Reduce crop context to create separate annotated training and validation blocks.")
    # Keep holdout blocks stable across repeated crop-sampling rounds.
    split_rng = np.random.default_rng(42)
    split_rng.shuffle(annotated)
    validation_ids = {b["id"] for b in annotated[:max(1, len(annotated) // 4)]}
    background = [b for b in blocks if not b["anchors"]]
    split_rng.shuffle(background)
    validation_ids.update(b["id"] for b in background[:len(background) // 4])
    for block in blocks:
        block["split"] = "validation" if block["id"] in validation_ids else "training"
    return blocks


def _origin(block, size, rng, anchor=None):
    if anchor is None:
        return (
            int(rng.integers(block["y0"], block["y1"] - size + 1)),
            int(rng.integers(block["x0"], block["x1"] - size + 1)),
        )
    jitter = rng.integers(-size // 4, size // 4 + 1, size=2)
    return (
        int(np.clip(anchor[0] - size // 2 + jitter[0], block["y0"], block["y1"] - size)),
        int(np.clip(anchor[1] - size // 2 + jitter[1], block["x0"], block["x1"] - size)),
    )


def prepare_crops(image, ground_truth, config, *, cancel=None):
    """Sample guaranteed positives and contextual negatives in disjoint blocks."""
    config.validate()
    image, ground_truth = array_2d(image), array_2d(ground_truth)
    if tuple(image.shape) != tuple(ground_truth.shape):
        raise ValueError("Training image and ground truth must have the same 2D shape.")
    height, width = map(int, image.shape)
    crop_pixels = min(config.crop_pixels, height, width, max(height, width) // 2)
    if crop_pixels < 32:
        raise ValueError("The image needs at least 32 pixels on each axis and 64 on one axis for separate training and validation areas.")
    rng = np.random.default_rng(config.seed)
    blocks = yield from _scan_annotations(ground_truth, crop_pixels, rng, cancel)
    validation_count = max(2, config.samples // 4)
    locations, datasets = [], []
    done = 0
    for split, count in (
        ("training", config.samples - validation_count),
        ("validation", validation_count),
    ):
        rng = np.random.default_rng(config.seed if split == "training" else 42)
        pool = [b for b in blocks if b["split"] == split]
        positives = [b for b in pool if b["anchors"]]
        rng.shuffle(positives)
        images, masks = [], []
        for i in range(count):
            requested = "positive" if i % 2 == 0 else ("nearby background" if i % 4 == 1 else "background")
            kind = requested
            for attempt in range(1 if requested == "positive" else 64):
                _check_cancel(cancel)
                if requested == "positive":
                    block = positives[(i // 2) % len(positives)]
                    anchor = block["anchors"][int(rng.integers(len(block["anchors"])))]
                    y0, x0 = _origin(block, crop_pixels, rng, anchor)
                elif requested == "nearby background" and attempt < 32:
                    block = positives[int(rng.integers(len(positives)))]
                    anchor = np.array(block["anchors"][int(rng.integers(len(block["anchors"])))])
                    direction = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)][int(rng.integers(8))])
                    center = anchor + direction * rng.uniform(0.8, 1.4) * crop_pixels
                    y0 = int(np.clip(center[0] - crop_pixels / 2, block["y0"], block["y1"] - crop_pixels))
                    x0 = int(np.clip(center[1] - crop_pixels / 2, block["x0"], block["x1"] - crop_pixels))
                else:
                    block = pool[int(rng.integers(len(pool)))]
                    y0, x0 = _origin(block, crop_pixels, rng)
                    kind = "background"
                region = (slice(y0, y0 + crop_pixels), slice(x0, x0 + crop_pixels))
                truth = np.asarray(ground_truth[region]) != 0
                if requested == "positive" or not truth.any():
                    break
            if requested == "positive" and not truth.any():
                raise ValueError("Ground truth changed during preparation. Prepare the crops again.")
            if requested != "positive" and truth.any():
                kind = "mixed context"
            raw = np.asarray(image[region])
            small_shape = (config.input_pixels, config.input_pixels)
            images.append(_resize(_normalize_crop(raw), small_shape))
            model_mask = _resize(truth, small_shape, labels=True).astype(np.uint8)
            masks.append(model_mask)
            locations.append({
                "split": split, "y0": y0, "x0": x0, "size": crop_pixels,
                "block": block["id"], "kind": kind, "requested": requested,
                "contains_foreground": bool(model_mask.any()),
                "foreground_pixels_original": int(np.count_nonzero(truth)),
                "foreground_pixels_model": int(np.count_nonzero(model_mask)),
            })
            done += 1
            yield QuickTrainProgress(done, config.samples, f"Preparing {split} crops: {done}/{config.samples}.")
        images, masks = np.stack(images), np.stack(masks)
        if not masks.any():
            raise ValueError(f"No labeled targets found in the {split} crops.")
        if masks.all():
            raise ValueError(f"No background found in the {split} crops. Ground truth must include positive and negative examples.")
        datasets.extend([images, masks])
    public_blocks = [{k: v for k, v in b.items() if k not in ("anchors", "seen")} for b in blocks]
    return PairedCrops(*datasets, crop_pixels, locations, public_blocks)


def crop_summary(crops):
    counts = {}
    for split in ("training", "validation"):
        rows = [p for p in crops.locations if p["split"] == split]
        counts[split] = {
            "total": len(rows), "positive": sum(p["contains_foreground"] for p in rows),
            "background": sum(not p["contains_foreground"] for p in rows),
            "nearby_background": sum(p["kind"] == "nearby background" for p in rows),
        }
    return counts


def build_crop_atlas(crops):
    """Pack the exact normalized model inputs into an aligned, annotated grid."""
    images = np.concatenate((crops.images, crops.validation_images))
    masks = np.concatenate((crops.masks, crops.validation_masks))
    size = images.shape[-1]
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    header, gutter = 36, 8
    stride_y, stride_x = size + header + gutter, size + gutter
    atlas_image = np.zeros((rows * stride_y, columns * stride_x), dtype=np.float32)
    atlas_mask = np.zeros(atlas_image.shape, dtype=np.uint8)
    locations = []
    for i, (image, mask, source) in enumerate(zip(images, masks, crops.locations)):
        y0, x0 = (i // columns) * stride_y + header, (i % columns) * stride_x
        atlas_image[y0:y0 + size, x0:x0 + size] = image
        atlas_mask[y0:y0 + size, x0:x0 + size] = mask
        locations.append({**source, "tile": i + 1, "atlas_y0": y0, "atlas_x0": x0, "model_size": size})
    return CropAtlas(atlas_image, atlas_mask, locations, columns, stride_y, stride_x, header)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Quick Train requires PyTorch in napari's Python environment. "
            "Install the optional dependency: pip install 'napari-label-assistant[train]'."
        ) from error
    return torch


def build_unet(base_channels=8):
    """Five-level U-Net with GroupNorm, skip connections, and binary output."""
    torch = _torch()
    nn, functional = torch.nn, torch.nn.functional

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [base_channels * 2**i for i in range(5)]
            self.down = nn.ModuleList([Block(1 if i == 0 else widths[i - 1], w) for i, w in enumerate(widths)])
            self.up = nn.ModuleList([Block(widths[i + 1] + widths[i], widths[i]) for i in range(3, -1, -1)])
            self.output = nn.Conv2d(widths[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, block in enumerate(self.down):
                if i:
                    x = functional.max_pool2d(x, 2)
                x = block(x)
                skips.append(x)
            for block, skip in zip(self.up, reversed(skips[:-1])):
                x = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = block(torch.cat((x, skip), dim=1))
            return self.output(x)

    return SmallUNet()


def _device(torch, requested):
    if requested == "cpu":
        return "cpu"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        torch.empty(1, device="cuda")
        if torch.cuda.mem_get_info()[0] < 1024**3:
            raise RuntimeError("Less than 1 GB of CUDA memory is free.")
        return "cuda"
    except RuntimeError as error:
        if requested == "cuda":
            raise RuntimeError("CUDA is unavailable or has insufficient memory. Choose CPU or free GPU memory.") from error
        return "cpu"


def _scores(torch, network, images, masks, device, cancel):
    network.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(len(images)):
            _check_cancel(cancel)
            x = torch.from_numpy(images[i:i + 1]).unsqueeze(1).to(device)
            predicted = torch.sigmoid(network(x)).cpu().numpy()[0, 0] >= 0.5
            truth = masks[i] > 0.5
            tp += int(np.count_nonzero(predicted & truth))
            fp += int(np.count_nonzero(predicted & ~truth))
            fn += int(np.count_nonzero(~predicted & truth))
    return {
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1),
    }


def train_quick_unet(image, ground_truth, config, *, cancel=None):
    """Generator executed by a worker; neither reads nor mutates full layers."""
    _check_cancel(cancel)
    crops = yield from prepare_crops(image, ground_truth, config, cancel=cancel)
    return (yield from train_prepared_unet(crops, config, cancel=cancel))


def train_prepared_unet(crops, config, *, cancel=None, initial_model=None, learning_rate=0.001, source_info=None):
    """Train only on the already inspected inputs; do not resample locations."""
    config.validate()
    if crops.images.shape[1:] != (config.input_pixels,) * 2 or len(crops.locations) != config.samples:
        raise ValueError("Crop settings changed. Prepare and preview crops again.")
    if not 0 < learning_rate <= 0.1:
        raise ValueError("Learning rate must be greater than zero and at most 0.1.")
    if initial_model is not None and config.base_channels != initial_model.config.base_channels:
        raise ValueError("Continue training requires the existing model architecture.")
    torch = _torch()
    _check_cancel(cancel)
    device = _device(torch, config.device)
    yield QuickTrainProgress(0, config.steps, f"Training small U-Net on {device.upper()}; {crops.crop_pixels}px context resized to {config.input_pixels}px.")
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = monotonic()
    checkpoints = []
    best_step = 0
    old_threads = torch.get_num_threads()
    network = None
    try:
        if device == "cpu":
            torch.set_num_threads(min(old_threads, 4))
        # Restore the host application's CPU random state after initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            network = build_unet(config.base_channels).to(device)
        if initial_model is not None:
            # Use a separate network: cancellation must preserve the current model.
            network.load_state_dict(initial_model.network.state_dict())
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        rng = np.random.default_rng(config.seed + (initial_model.training_steps if initial_model else 0))
        best_score, best_state, best_metrics = -1, None, {}
        if initial_model is not None:
            best_metrics = _scores(torch, network, crops.validation_images, crops.validation_masks, device, cancel)
            best_score = best_metrics["dice"]
            checkpoints.append({"step": 0, "validation": dict(best_metrics)})
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        batch_size = min(config.batch_size, len(crops.images))
        for step in range(1, config.steps + 1):
            _check_cancel(cancel)
            network.train()
            indices = rng.choice(len(crops.images), batch_size, replace=False)
            x, y = crops.images[indices], crops.masks[indices]
            turns = int(rng.integers(4))
            x, y = np.rot90(x, turns, axes=(-2, -1)), np.rot90(y, turns, axes=(-2, -1))
            if rng.random() < 0.5:
                x, y = x[..., ::-1], y[..., ::-1]
            x = np.clip(x * rng.uniform(0.9, 1.1) + rng.uniform(-0.03, 0.03), 0, 1)
            x = torch.from_numpy(x.copy()).unsqueeze(1).to(device)
            y = torch.from_numpy(y.copy()).unsqueeze(1).float().to(device)
            logits = network(x)
            probability = torch.sigmoid(logits)
            dice_loss = 1 - (2 * (probability * y).sum() + 1) / (probability.sum() + y.sum() + 1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == config.steps:
                metrics = _scores(torch, network, crops.validation_images, crops.validation_masks, device, cancel)
                checkpoints.append({"step": step, "loss": float(loss.detach()), "validation": dict(metrics)})
                if metrics["dice"] >= best_score:
                    best_step = step
                    best_score, best_metrics = metrics["dice"], metrics
                    best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
                yield QuickTrainProgress(step, config.steps, f"Step {step}/{config.steps} · loss {float(loss.detach()):.3f} · validation Dice {metrics['dice']:.1%}.")
            else:
                yield QuickTrainProgress(step, config.steps, f"Training step {step}/{config.steps} on {device.upper()}.")
        _check_cancel(cancel)
        network.cpu()
        network.load_state_dict(best_state)
        network.eval()
        history = deepcopy(initial_model.history) if initial_model else []
        history.append({
            "round": (initial_model.training_rounds if initial_model else 0) + 1,
            "mode": "continue" if initial_model else "new",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": monotonic() - started_clock,
            "config": asdict(config), "learning_rate": learning_rate,
            "optimizer": "AdamW (fresh each round)", "device": device,
            "source": deepcopy(source_info or {}), "crop_pixels": crops.crop_pixels,
            "crop_summary": crop_summary(crops), "locations": deepcopy(crops.locations),
            "checkpoints": checkpoints, "retained_step": best_step,
            "retained_validation": dict(best_metrics),
        })
        return QuickTrainModel(network, config, crops.crop_pixels, best_metrics, device, crops.locations,
                               (initial_model.training_steps if initial_model else 0) + config.steps,
                               (initial_model.training_rounds if initial_model else 0) + 1, history)
    finally:
        if network is not None:
            network.cpu()
        if device == "cpu":
            torch.set_num_threads(old_threads)


def predict_area(model, image, center, *, device="auto", cancel=None):
    """Predict one bounded region centered at data coordinates."""
    torch = _torch()
    image = array_2d(image)
    _check_cancel(cancel)
    height, width = map(int, image.shape)
    size = min(model.crop_pixels, height, width)
    y0 = max(0, min(height - size, int(center[0]) - size // 2))
    x0 = max(0, min(width - size, int(center[1]) - size // 2))
    raw = np.asarray(image[y0:y0 + size, x0:x0 + size])
    normalized = _resize(_normalize_crop(raw), (model.config.input_pixels,) * 2)
    actual_device = _device(torch, device)
    old_threads = torch.get_num_threads()
    try:
        if actual_device == "cpu":
            torch.set_num_threads(min(old_threads, 4))
        model.network.to(actual_device).eval()
        with torch.no_grad():
            x = torch.from_numpy(normalized)[None, None].to(actual_device)
            probability = torch.sigmoid(model.network(x)).cpu().numpy()[0, 0]
        _check_cancel(cancel)
        return QuickTrainPrediction(np.clip(_resize(probability, (size, size)), 0, 1), (y0, x0))
    finally:
        model.network.cpu()
        if actual_device == "cpu":
            torch.set_num_threads(old_threads)


def save_model(model, path):
    torch = _torch()
    torch.save({
        "quick_train_version": 1, "config": asdict(model.config),
        "crop_pixels": model.crop_pixels, "metrics": model.metrics,
        "training_device": model.training_device, "locations": model.locations,
        "training_steps": model.training_steps, "training_rounds": model.training_rounds,
        "history": model.history,
        "state_dict": model.network.cpu().state_dict(),
    }, path)
    history_path = Path(path).with_suffix(".history.json")
    history_path.write_text(json.dumps({
        "history_version": 1, "model_file": Path(path).name,
        "training_rounds": model.training_rounds, "training_steps": model.training_steps,
        "history": model.history,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_model(path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("quick_train_version") != 1:
        raise ValueError("Choose a model saved by Label Assistant Quick Train.")
    config = QuickTrainConfig(**payload["config"])
    config.validate()
    crop_pixels = int(payload["crop_pixels"])
    if not 32 <= crop_pixels <= config.crop_pixels:
        raise ValueError("The model has an invalid crop size.")
    network = build_unet(config.base_channels)
    network.load_state_dict(payload["state_dict"], strict=True)
    network.eval()
    return QuickTrainModel(network, config, crop_pixels, payload["metrics"], payload["training_device"], payload.get("locations", []),
                           payload.get("training_steps", config.steps), payload.get("training_rounds", 1), payload.get("history", []))
