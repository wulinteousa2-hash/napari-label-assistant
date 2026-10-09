"""Coordinate-indexed, bounded-memory training over reviewed image regions."""
from __future__ import annotations

from copy import deepcopy
from hashlib import blake2b
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from time import monotonic

import numpy as np

from ._quick_train import (
    PairedCrops, QuickTrainModel, QuickTrainProgress, _check_cancel, _device,
    _normalize_crop, _resize, _torch, array_2d, augment_batch, build_unet,
)


def _model_supervision(truth, valid, crop_pixels, input_pixels):
    pad = ((0, crop_pixels - truth.shape[0]), (0, crop_pixels - truth.shape[1]))
    reviewed = np.pad(valid, pad)
    targets = np.pad(truth & valid, pad)
    shape = (input_pixels,) * 2
    usable = 1 - _resize(~reviewed, shape, labels=True)
    targets = _resize(targets, shape, labels=True) * usable
    if not usable.any() or ((truth & valid).any() and not targets.any()):
        raise ValueError(
            'Reviewed pixels or targets disappear at model input size. '
            'Use matching source patch and model input sizes, or fully review a larger region.'
        )
    return targets.astype(np.float32), usable.astype(np.float32)


@dataclass
class IndexedPatches:
    image: object
    ground_truth: object
    reviewed: object | None
    crop_pixels: int
    input_pixels: int
    locations: list[dict]
    blocks: list[dict]
    reviewed_pixels: int
    total_pixels: int
    validation_seen_before: bool = False
    validation: dict = field(default_factory=dict)

    def indices(self, split):
        return [i for i, item in enumerate(self.locations) if item['split'] == split]

    def summary(self):
        return {
            split: {
                'total': len(rows := [p for p in self.locations if p['split'] == split]),
                'positive': sum(p['contains_foreground'] for p in rows),
                'background': sum(not p['contains_foreground'] for p in rows),
                'reviewed_pixels': sum(p['reviewed_pixels'] for p in rows),
            } for split in ('training', 'validation')
        }

    def read(self, index):
        """Read just one patch; padding and unreviewed pixels are excluded from loss."""
        item = self.locations[index]
        region = (slice(item['y0'], item['y1']), slice(item['x0'], item['x1']))
        raw = np.asarray(self.image[region])
        if item.get("image_digest") and blake2b(raw.tobytes(), digest_size=16).hexdigest() != item["image_digest"]:
            raise ValueError("Image pixels changed during comparison. Prepare the dataset and restart the comparison.")
        source_labels = np.asarray(self.ground_truth[region])
        truth = source_labels != 0
        valid = np.ones(truth.shape, bool) if self.reviewed is None else np.asarray(self.reviewed[region]) != 0
        if int(valid.sum()) != item['reviewed_pixels'] or int((truth & valid).sum()) != item['foreground_pixels_original']:
            raise ValueError('Annotations changed after indexing. Prepare the dataset again.')
        digest = blake2b(truth.tobytes()+valid.tobytes(), digest_size=16).hexdigest()
        if item.get('annotation_digest', digest) != digest:
            raise ValueError('Annotations changed after indexing. Prepare the dataset again.')
        label_digest = blake2b(source_labels.tobytes(), digest_size=16).hexdigest()
        if item.get("source_label_digest", label_digest) != label_digest:
            raise ValueError("Source label IDs changed after indexing. Update and prepare the dataset again.")
        normalized = _normalize_crop(raw)
        pad = ((0, self.crop_pixels - raw.shape[0]), (0, self.crop_pixels - raw.shape[1]))
        normalized = np.pad(normalized, pad, mode='edge')
        shape = (self.input_pixels,) * 2
        x = _resize(normalized, shape)
        y, valid = _model_supervision(truth, valid, self.crop_pixels, self.input_pixels)
        return x.astype(np.float32), y.astype(np.float32), valid.astype(np.float32)


def index_patches(image, ground_truth, config, *, reviewed=None, validation_blocks=None, prior_training_regions=(), cancel=None, validation_fraction=0.25, split_seed=42, validation_method="holdout", folds=5):
    """Visit every grid patch, storing coordinates and counts rather than image pixels."""
    from ._model_providers import spec_for
    config.validate()
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("Validation holdout must be between 5% and 50%.")
    image, ground_truth = array_2d(image), array_2d(ground_truth)
    reviewed = array_2d(reviewed) if reviewed is not None else None
    if tuple(image.shape) != tuple(ground_truth.shape) or (reviewed is not None and tuple(reviewed.shape) != tuple(image.shape)):
        raise ValueError('Image, ground truth, and reviewed-region mask must have the same 2D shape.')
    h, w = map(int, image.shape)
    size = config.crop_pixels
    if min(h, w) < size * 2:
        raise ValueError('Coverage training needs at least two source patches on each axis. Choose a smaller source patch size.')
    locations, groups = [], {}
    component_blocks = {}
    total = ((h + size - 1) // size) * ((w + size - 1) // size)
    reviewed_pixels = 0
    group_rows = 2 if (h + size - 1) // size >= 4 else 1
    group_cols = 2 if (w + size - 1) // size >= 4 else 1
    candidates = getattr(reviewed, "candidate_tiles", lambda *_: None)(size)
    if candidates is None:
        coordinates = ((row, col) for row in range((h+size-1)//size)
                       for col in range((w+size-1)//size))
    else:
        coordinates = candidates
        total = len(candidates)
    for done, (row, col) in enumerate(coordinates, 1):
        y0, x0 = row*size, col*size
        _check_cancel(cancel)
        region = (slice(y0, min(y0 + size, h)), slice(x0, min(x0 + size, w)))
        source_labels = np.asarray(ground_truth[region])
        truth = source_labels != 0
        valid = np.ones(truth.shape, bool) if reviewed is None else np.asarray(reviewed[region]) != 0
        count, positive = int(valid.sum()), int((truth & valid).sum())
        if count:
            _model_supervision(truth, valid, size, config.input_pixels)
            block = f'{row // group_rows}:{col // group_cols}'
            for component in getattr(reviewed, "component_ids", lambda *args: [])(region, truth, valid):
                component_blocks.setdefault(component, set()).add(block)
            groups.setdefault(block, []).append(len(locations))
            locations.append({
                'y0': y0, 'x0': x0, 'y1': region[0].stop, 'x1': region[1].stop,
                'size': size, 'block': block, 'reviewed_pixels': count,
                'foreground_pixels_original': positive, 'contains_foreground': bool(positive),
                'kind': 'target' if positive else 'reviewed background',
                'annotation_digest': blake2b(truth.tobytes()+valid.tobytes(), digest_size=16).hexdigest(),
                'source_label_digest': blake2b(source_labels.tobytes(), digest_size=16).hexdigest(),
            })
            reviewed_pixels += count
        if done == total or done % max(1, total // 200) == 0:
            yield QuickTrainProgress(done, total, f'Indexing reviewed regions: {done:,}/{total:,} patches; {len(locations):,} eligible.')
    if not locations:
        raise ValueError('No reviewed pixels found. Mark complete reviewed regions in the reviewed-region mask.')
    if getattr(reviewed, "require_background", False) and not any(
        p["reviewed_pixels"] > p["foreground_pixels_original"] for p in locations
    ):
        raise ValueError("Selected objects supply foreground only. Review background in Local Editing before training.")
    forbidden = set()
    for region in prior_training_regions:
        y0, x0 = max(0, region['y0']), max(0, region['x0'])
        y1, x1 = min(h, region['y0'] + region['size']), min(w, region['x0'] + region['size'])
        if y1 <= y0 or x1 <= x0:
            continue
        for row in range(y0 // (size * group_rows), (y1 - 1) // (size * group_rows) + 1):
            for col in range(x0 // (size * group_cols), (x1 - 1) // (size * group_cols) + 1):
                forbidden.add(f'{row}:{col}')
    parents = {block: block for block in groups}
    def root(block):
        while parents[block] != block:
            parents[block] = parents[parents[block]]
            block = parents[block]
        return block
    for touched in component_blocks.values():
        touched = sorted(touched)
        for block in touched[1:]:
            left, right = root(touched[0]), root(block)
            parents[max(left, right)] = min(left, right)
    remap = {block: root(block) for block in groups}
    grouped = {}
    for block, ids in groups.items():
        grouped.setdefault(remap[block], []).extend(ids)
    groups = grouped
    for item in locations:
        item["spatial_block"] = item["block"]
        item["block"] = remap[item["block"]]
    forbidden = {remap.get(block, block) for block in forbidden}
    if validation_blocks is not None:
        validation_blocks = {remap.get(block, block) for block in validation_blocks}
    if validation_method == "kfold":
        partitions = spatial_folds(locations, folds, split_seed)
        validation_blocks = set(partitions[0])
    elif validation_method != "holdout":
        raise ValueError("Choose spatial holdout or spatial k-fold validation.")
    if validation_blocks is None:
        positives = sorted(k for k, ids in groups.items() if any(locations[i]['contains_foreground'] for i in ids))
        if len(positives) < 2:
            raise ValueError('Reviewed targets need at least two separate spatial blocks. Review more regions or reduce source patch size.')
        rng = np.random.default_rng(split_seed)
        candidates = [k for k in positives if k not in forbidden]
        if not candidates:
            # A new model can still use this dataset; continuation must not use seen holdouts.
            candidates = list(positives)
        rng.shuffle(candidates)
        background = sorted(set(groups) - set(positives) - forbidden)
        rng.shuffle(background)
        validation_blocks = set(candidates[:max(1, int(len(positives) * validation_fraction))]) | set(background[:int(len(background) * validation_fraction)])
    else:
        validation_blocks = set(validation_blocks)
    for item in locations:
        item['split'] = 'validation' if item['block'] in validation_blocks else 'training'
    # Retain reserved blocks even if annotations temporarily disappear from them.
    blocks = [{'id': k, 'split': 'validation' if remap.get(k, k) in validation_blocks else 'training'}
              for k in sorted(set(remap) | validation_blocks)]
    for split in ('training', 'validation'):
        rows = [p for p in locations if p['split'] == split]
        if not rows or not any(p['contains_foreground'] for p in rows):
            raise ValueError(f'{split.title()} regions need reviewed targets. Review more regions; keep validation separate.')
        if not any(p['reviewed_pixels'] > p['foreground_pixels_original'] for p in rows):
            raise ValueError(f'{split.title()} regions need reviewed background as well as targets.')
    return IndexedPatches(image, ground_truth, reviewed, size, config.input_pixels, locations, blocks, reviewed_pixels, h * w, bool(validation_blocks & forbidden), {"method": validation_method, "holdout_fraction": validation_fraction, "seed": split_seed, "fold_count": folds if validation_method == "kfold" else 1, "folds": partitions if validation_method == "kfold" else []})


def preview_index(dataset, count=16, *, cancel=None):
    """A bounded, representative gallery; training still uses the entire index."""
    images, masks, validity, locations = [], [], [], []
    per_group = max(1, count // 4)
    for split in ('training', 'validation'):
        xs, ys, vs = [], [], []
        for positive in (True, False):
            ids = [i for i in dataset.indices(split) if dataset.locations[i]['contains_foreground'] == positive]
            chosen = [ids[i] for i in np.linspace(0, len(ids) - 1, min(per_group, len(ids)), dtype=int)] if ids else []
            for index in chosen:
                _check_cancel(cancel)
                x, y, v = dataset.read(index)
                xs.append(x); ys.append(y); vs.append(v)
                locations.append({**dataset.locations[index], 'foreground_pixels_model': int(y.sum())})
        images.append(np.stack(xs)); masks.append(np.stack(ys)); validity.append(np.stack(vs))
    crops = PairedCrops(images[0], masks[0], images[1], masks[1], dataset.crop_pixels, locations, dataset.blocks)
    crops.validity = np.concatenate(validity)
    return crops


class PlateauControl:
    """Reduce LR once, then stop after a further period without improvement."""
    def __init__(self, reduce_patience=10, stop_patience=10, factor=0.5, min_delta=0.0001):
        if reduce_patience < 1 or stop_patience < 1 or not 0 < factor < 1 or min_delta < 0:
            raise ValueError("Choose positive patience, an LR factor between 0 and 1, and nonnegative minimum improvement.")
        self.reduce_patience = reduce_patience
        self.stop_patience = stop_patience
        self.factor = factor
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.reduced = False

    def update(self, loss):
        if not np.isfinite(loss):
            raise ValueError("Validation loss is not finite. Training cannot monitor improvement.")
        if loss < self.best - self.min_delta:
            self.best = loss
            self.bad_epochs = 0
            return None
        self.bad_epochs += 1
        patience = self.stop_patience if self.reduced else self.reduce_patience
        if self.bad_epochs < patience:
            return None
        if not self.reduced:
            self.reduced = True
            self.bad_epochs = 0
            return "reduce"
        return "stop"


def epoch_indices(dataset, rng, extra_targets=0.25):
    ids = np.array(dataset.indices('training'), dtype=int)
    positives = ids[[dataset.locations[i]['contains_foreground'] for i in ids]]
    extra = int(np.ceil(len(ids) * extra_targets)) if extra_targets else 0
    if extra:
        ids = np.concatenate([ids, rng.choice(positives, extra, replace=True)])
    return rng.permutation(ids)


def masked_binary_loss(torch, logits, truth, valid):
    probability = torch.sigmoid(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    dice = 1 - (2 * (probability * truth * valid).sum() + 1) / ((probability * valid).sum() + (truth * valid).sum() + 1)
    return bce + dice


def _validate(torch, network, dataset, device, cancel, *, step, total):
    network.eval()
    tp = fp = fn = 0
    losses = []
    ids = dataset.indices('validation')
    with torch.no_grad():
        for done, index in enumerate(ids, start=1):
            _check_cancel(cancel)
            x, y, valid = dataset.read(index)
            logits = network(torch.from_numpy(x)[None, None].to(device))
            losses.append(float(masked_binary_loss(torch, logits,
                torch.from_numpy(y)[None, None].to(device),
                torch.from_numpy(valid)[None, None].to(device))))
            prediction = torch.sigmoid(logits).cpu().numpy()[0, 0] >= 0.5
            truth, valid = y > 0.5, valid > 0.5
            tp += int((prediction & truth & valid).sum())
            fp += int((prediction & ~truth & valid).sum())
            fn += int((~prediction & truth & valid).sum())
            if done == len(ids) or done % max(1, len(ids) // 100) == 0:
                yield QuickTrainProgress(step, total, f'Validating reserved regions: {done:,}/{len(ids):,} patches.')
    return {'loss': float(np.mean(losses)), 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'dice': 2 * tp / max(2 * tp + fp + fn, 1), 'iou': tp / max(tp + fp + fn, 1)}


def train_indexed(dataset, config, *, epochs=5, extra_targets=0.25, learning_rate=0.001,
                  initial_model=None, source_info=None, cancel=None,
                  plateau_enabled=False, reduce_patience=10, stop_patience=10,
                  lr_factor=0.5, min_delta=0.0001, validation_enabled=True):
    from ._model_providers import build_model, spec_for, training_mode, library_versions
    from ._model_comparison import dataset_snapshot
    config.validate()
    if not 1 <= epochs <= 1000 or not 0 <= extra_targets <= 1 or not 0 < learning_rate <= 0.1:
        raise ValueError('Choose 1–1000 epochs, 0–100% additional target patches, and a positive learning rate up to 0.1.')
    if (config.crop_pixels, config.input_pixels) != (dataset.crop_pixels, dataset.input_pixels):
        raise ValueError('Patch settings changed. Index the dataset again.')
    if initial_model is not None and dataset.validation_seen_before:
        raise ValueError('The current model has already trained in these validation regions. Review new regions or train a new model.')
    if initial_model is not None and spec_for(config) != spec_for(initial_model.config):
        raise ValueError('Continue training requires the current model architecture.')
    control = PlateauControl(reduce_patience, stop_patience, lr_factor, min_delta) if plateau_enabled and validation_enabled else None
    lr_events = []
    stop_reason = None
    torch = _torch()
    device = _device(torch, config.device)
    old_threads = torch.get_num_threads()
    network = None
    started = monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    train_count = len(dataset.indices('training'))
    patches_per_epoch = train_count + (int(np.ceil(train_count * extra_targets)) if extra_targets else 0)
    steps_per_epoch = (patches_per_epoch + config.batch_size - 1) // config.batch_size
    total, step = epochs * steps_per_epoch, 0
    checkpoints, best_metrics, best_score, best_state, retained_epoch = [], {}, -1, None, 0
    try:
        if device == 'cpu':
            torch.set_num_threads(min(old_threads, 4))
        _check_cancel(cancel)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            network = build_model(config, initialize=initial_model is None).to(device)
        if initial_model is not None:
            network.load_state_dict(initial_model.network.state_dict())
            yield QuickTrainProgress(0, total, 'Evaluating current model on reserved validation regions…')
            best_metrics = yield from _validate(torch, network, dataset, device, cancel, step=step, total=total)
            best_score = best_metrics['dice']
            if control is not None:
                control.update(best_metrics['loss'])
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            checkpoints.append({'epoch': 0, 'validation': dict(best_metrics)})
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        rng = np.random.default_rng(config.seed + (initial_model.training_steps if initial_model else 0))
        for epoch in range(1, epochs + 1):
            visited = set()
            order = epoch_indices(dataset, rng, extra_targets)
            for start in range(0, len(order), config.batch_size):
                _check_cancel(cancel)
                batch_ids = order[start:start + config.batch_size]
                batches = [dataset.read(int(i)) for i in batch_ids]
                x, y, valid = [np.stack(parts) for parts in zip(*batches)]
                x, y, valid = augment_batch(x, y, valid, rng, config)
                x, y, valid = [torch.from_numpy(a.copy()).unsqueeze(1).to(device) for a in (x, y, valid)]
                training_mode(network, len(batch_ids))
                optimizer.zero_grad(set_to_none=True)
                loss = masked_binary_loss(torch, network(x), y, valid)
                loss.backward(); optimizer.step()
                step += 1
                visited.update(map(int, batch_ids))
                if step % max(1, steps_per_epoch // 200) == 0 or start + config.batch_size >= len(order):
                    yield QuickTrainProgress(step, total, f'Epoch {epoch}/{epochs} · coverage {len(visited):,}/{train_count:,} training patches ({len(visited)/train_count:.0%}) · step {step:,}/{total:,} · batch training loss {float(loss.detach()):.3f}.')
            if validation_enabled:
                yield QuickTrainProgress(step, total, f'Epoch {epoch}/{epochs}: validating all reserved patches…')
                metrics = yield from _validate(torch, network, dataset, device, cancel, step=step, total=total)
                checkpoints.append({'epoch': epoch, 'step': step, 'unique_patches': len(visited),
                    'learning_rate': optimizer.param_groups[0]['lr'], 'training_loss_last_batch': float(loss.detach()), 'validation': dict(metrics)})
                if metrics['dice'] >= best_score:
                    best_score, best_metrics, retained_epoch = metrics['dice'], metrics, epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
                loss_summary = f" · validation loss {metrics['loss']:.4f}" if 'loss' in metrics else ''
                yield QuickTrainProgress(step, total, f'Epoch {epoch}/{epochs} complete · coverage 100% · validation Dice {metrics["dice"]:.1%} · IoU {metrics["iou"]:.1%}{loss_summary}.', metrics=dict(metrics), epoch=epoch, training_loss_last_batch=float(loss.detach()), training_step=step)
            else:
                retained_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
                checkpoints.append({'epoch': epoch, 'step': step, 'unique_patches': len(visited), 'learning_rate': optimizer.param_groups[0]['lr']})
                yield QuickTrainProgress(step, total, f'Epoch {epoch}/{epochs} complete · all reviewed training patches visited · no holdout evaluation.')
            if control is not None:
                action = control.update(metrics['loss'])
                if action == "reduce":
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = min(old_lr, max(1e-7, old_lr * lr_factor))
                    for group in optimizer.param_groups:
                        group['lr'] = new_lr
                    lr_events.append({'epoch': epoch, 'old_lr': old_lr, 'new_lr': new_lr})
                    yield QuickTrainProgress(step, total, f'Validation loss stalled for {reduce_patience} epochs. Learning rate reduced: {old_lr:g} → {new_lr:g}.')
                elif action == "stop":
                    stop_reason = f"No validation-loss improvement for {stop_patience} epochs after learning-rate reduction."
                    yield QuickTrainProgress(step, step, f'Early stop at epoch {epoch}: {stop_reason} Retaining best validation Dice checkpoint.')
                    break
        _check_cancel(cancel)
        network.cpu().load_state_dict(best_state)
        network.eval()
        history = deepcopy(initial_model.history) if initial_model else []
        history.append({'round': (initial_model.training_rounds if initial_model else 0) + 1,
            'workflow': 'coverage', 'mode': 'continue' if initial_model else ('new' if validation_enabled else 'final_all_reviewed'),
            'validation': deepcopy(dataset.validation) if validation_enabled else {'method': 'none', 'purpose': 'final model on all reviewed data'},
            'locations': deepcopy(dataset.locations),
            'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'duration_seconds': monotonic() - started, 'config': asdict(config),
            'dataset_snapshot_id': dataset_snapshot(dataset, source_info)['id'], 'library_versions': library_versions(), 'epochs': epoch, 'max_epochs': epochs,
            'plateau_control': {'enabled': bool(plateau_enabled and validation_enabled), 'requested': plateau_enabled, 'reduce_patience': reduce_patience,
                'stop_patience': stop_patience, 'factor': lr_factor, 'min_delta': min_delta},
            'learning_rate_events': lr_events, 'stop_reason': stop_reason,
            'steps': step, 'extra_targets': extra_targets, 'learning_rate': learning_rate,
            'optimizer': 'AdamW (fresh each round)', 'singleton_batch_norm': 'running_statistics', 'device': device, 'source': deepcopy(source_info or {}),
            'dataset_summary': dataset.summary(), 'reviewed_pixels': dataset.reviewed_pixels,
            'total_pixels': dataset.total_pixels, 'validation_blocks': [b['id'] for b in dataset.blocks if b['split'] == 'validation'],
            'checkpoints': checkpoints, 'retained_epoch': retained_epoch, 'retained_validation': dict(best_metrics)})
        return QuickTrainModel(network, config, dataset.crop_pixels, best_metrics, device, deepcopy(dataset.locations),
            (initial_model.training_steps if initial_model else 0) + step,
            (initial_model.training_rounds if initial_model else 0) + 1, history, deepcopy(initial_model.identity) if initial_model else {})
    finally:
        if network is not None:
            network.cpu()
        if device == 'cpu':
            torch.set_num_threads(old_threads)


def spatial_folds(locations, count=5, seed=42):
    """Keep each spatial/component group intact and balance patch counts."""
    if not 2 <= count <= 10:
        raise ValueError("Choose 2–10 spatial folds.")
    groups = {}
    for item in locations:
        groups.setdefault(item['block'], []).append(item)
    positive = [key for key, rows in groups.items() if any(r['contains_foreground'] for r in rows)]
    if len(positive) < count:
        raise ValueError(f"{count}-fold validation needs at least {count} independent spatial groups containing targets; found {len(positive)}. Review more separate regions or choose fewer folds.")
    rng = np.random.default_rng(seed)
    result, sizes = [[] for _ in range(count)], [0]*count
    for category in (positive, [key for key in groups if key not in positive]):
        category = sorted(category)
        rng.shuffle(category)
        category.sort(key=lambda key: len(groups[key]), reverse=True)
        for key in category:
            fold = min(range(count), key=lambda i: (sizes[i], i))
            result[fold].append(key)
            sizes[fold] += len(groups[key])
    for fold, blocks in enumerate(result, 1):
        for validation in (True, False):
            rows = [r for r in locations if (r['block'] in blocks) == validation]
            if not rows or not any(r['contains_foreground'] for r in rows) or not any(r['reviewed_pixels'] > r['foreground_pixels_original'] for r in rows):
                raise ValueError(f"Fold {fold} needs reviewed targets and background in both training and validation. Review more complete regions or choose fewer folds.")
    return [sorted(blocks) for blocks in result]


def partition_dataset(dataset, validation_blocks):
    blocks = set(validation_blocks)
    locations = [{**item, 'split': 'validation' if item['block'] in blocks else 'training'} for item in dataset.locations]
    return replace(dataset, locations=locations,
                   blocks=[{'id': key, 'split': 'validation' if key in blocks else 'training'} for key in sorted({r['block'] for r in locations})],
                   validation_seen_before=False)


@dataclass
class CrossValidationResult:
    models: list
    report: dict


def train_kfold(dataset, config, *, cancel=None, **settings):
    """Train each fold independently; do not replace the user's current model."""
    if settings.get("initial_model") is not None:
        raise ValueError("Spatial k-fold validation must start fresh weights for every fold.")
    folds = dataset.validation.get('folds')
    if not folds:
        raise ValueError("Index the dataset with spatial k-fold selected before running validation.")
    models, scores = [], []
    started = datetime.now(timezone.utc).isoformat()
    for number, blocks in enumerate(folds, 1):
        _check_cancel(cancel)
        current = partition_dataset(dataset, blocks)
        current.validation = {**dataset.validation, 'fold': number, 'validation_blocks': blocks}
        generator = train_indexed(current, replace(config, seed=config.seed+number-1), cancel=cancel, **settings)
        while True:
            try:
                progress = next(generator)
            except StopIteration as done:
                model = done.value
                break
            fraction = progress.completed / max(1, progress.total)
            yield QuickTrainProgress(int((number-1+fraction)*1000), len(folds)*1000, f'Fold {number}/{len(folds)} · {progress.message}', metrics=progress.metrics, epoch=progress.epoch, fold=number, training_loss_last_batch=progress.training_loss_last_batch, training_step=progress.training_step)
        models.append(model)
        scores.append({'fold': number, 'metrics': dict(model.metrics), 'best_epoch': model.history[-1]['retained_epoch'],
                       'training_patches': len(current.indices('training')), 'validation_patches': len(current.indices('validation')),
                       'history': [{key: deepcopy(value) for key, value in record.items() if key != 'locations'} for record in model.history]})
    aggregate = {key: {'mean': float(np.mean([m.metrics[key] for m in models])),
                       'std': float(np.std([m.metrics[key] for m in models], ddof=1))} for key in models[0].metrics}
    report = {'workflow': 'spatial_cross_validation', 'validation': deepcopy(dataset.validation),
              'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
              'fold_results': scores, 'aggregate': aggregate, 'locations': deepcopy(dataset.locations),
              'note': 'Each fold starts from fresh weights. Scores use checkpoints selected on the validation fold; they are not independent test-set scores.'}
    for model in models:
        model.history[-1]['cross_validation_report'] = deepcopy(report)
    return CrossValidationResult(models, report)


def train_final(dataset, config, *, cross_validation_report, cancel=None, **settings):
    if settings.get('initial_model') is not None:
        raise ValueError('Final all-reviewed training starts fresh weights.')
    current = partition_dataset(dataset, [])
    model = yield from train_indexed(current, config, cancel=cancel, validation_enabled=False, **settings)
    model.history[-1]['cross_validation_report'] = deepcopy(cross_validation_report)
    return model
