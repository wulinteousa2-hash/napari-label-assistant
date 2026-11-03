"""Independent, overlapping target masks with per-channel supervision."""
from dataclasses import dataclass, field, replace
from hashlib import blake2b

import numpy as np

from ._coverage_train import (IndexedPatches, index_patches, index_pairs,
                             partition_dataset, spatial_folds, _check_class_coverage)
from ._quick_train import QuickTrainProgress, _check_cancel, array_2d, _normalize_crop, _resize


class BinaryTarget:
    """Validate binary mask pixels only where this channel is supervised."""
    def __init__(self, data, reviewed, name, foreground_mode='binary'):
        self.data, self.reviewed, self.name = data, reviewed, name
        self.shape = data.shape
        if foreground_mode not in ('binary', 'nonzero'):
            raise ValueError('Choose binary 0/1 or explicit nonzero foreground interpretation.')
        self.foreground_mode = foreground_mode

    def __getitem__(self, region):
        values = np.asarray(self.data[region])
        known = values if self.reviewed is None else values[np.asarray(self.reviewed[region]) != 0]
        if self.foreground_mode == 'nonzero':
            if not np.isfinite(known).all():
                raise ValueError(f'Channel {self.name!r} contains non-finite supervised label pixels.')
            return (values != 0).astype(np.uint8)
        if not np.isin(known, (0, 1)).all():
            raise ValueError(f'Channel {self.name!r} needs a binary 0/1 mask in supervised regions. Convert instance/class IDs explicitly before training.')
        return values


@dataclass
class MultilabelPatches(IndexedPatches):
    channel_datasets: list = field(default_factory=list)

    def read(self, index):
        if self.sources:
            return super().read(index)
        item = self.locations[index]
        region = (slice(item['y0'], item['y1']), slice(item['x0'], item['x1']))
        raw = np.asarray(self.image[region])
        if item.get('image_digest'):
            if blake2b(raw.tobytes(), digest_size=16).hexdigest() != item['image_digest']:
                raise ValueError('Image pixels changed during comparison. Prepare the dataset again.')
        normalized = np.pad(_normalize_crop(raw),
                            ((0, self.crop_pixels - raw.shape[0]), (0, self.crop_pixels - raw.shape[1])), mode='edge')
        image_patch = (raw, _resize(normalized, (self.input_pixels,) * 2))
        x, targets, validity = None, [], []
        for source, patch_index in zip(self.channel_datasets, item['channel_patch_indices']):
            if source is None or patch_index is None:
                y = np.zeros((self.input_pixels,) * 2, np.float32)
                valid = np.zeros_like(y)
            else:
                image, y, valid = source.read(patch_index, _image_patch=image_patch)
                if x is None:
                    x = image
            targets.append(y)
            validity.append(valid)
        if x is None:
            raise ValueError('No supervised channels remain in this patch. Prepare the dataset again.')
        return x, np.stack(targets), np.stack(validity)


def index_multilabel(pairs, config, *, cancel=None, validation_fraction=0.25,
                     split_seed=42, validation_method='holdout', folds=5,
                     prior_training_image_ids=(), prior_training_regions=()):
    """Index each channel independently; missing labels never become negatives."""
    config.validate()
    if not config.multilabel or not pairs:
        raise ValueError('Choose multilabel mode, name channels, and include at least one image.')
    binary = replace(config, class_names=(), multilabel=False, model_spec=None)
    sources, infos = [], []
    for number, pair in enumerate(pairs):
        _check_cancel(cancel)
        image = array_2d(pair['image'])
        if len(pair['channels']) != len(config.class_names):
            raise ValueError('Target assignments must match the named output channels. Open Manage channels.')
        channels, by_region = [], {}
        parents = {}

        def root(value):
            parents.setdefault(value, value)
            while parents[value] != value:
                parents[value] = parents[parents[value]]
                value = parents[value]
            return value

        for channel, (name, target) in enumerate(zip(config.class_names, pair['channels'])):
            _check_cancel(cancel)
            if target is None:
                channels.append(None)
                continue
            data = array_2d(target['data'])
            reviewed = target.get('reviewed')
            if tuple(data.shape) != tuple(image.shape) or (reviewed is not None and tuple(reviewed.shape) != tuple(image.shape)):
                raise ValueError(f'Channel {name!r} mask and reviewed coverage must match its image.')
            data = BinaryTarget(data, reviewed, name, target.get('foreground_mode', 'binary'))
            source = yield from index_patches(image, data, binary, reviewed=reviewed,
                                              cancel=cancel, _defer_split=True)
            channels.append(source)
            for patch_index, item in enumerate(source.locations):
                left, right = root(item['spatial_block']), root(item['block'])
                parents[max(left, right)] = min(left, right)
                region = tuple(item[key] for key in ('y0', 'x0', 'y1', 'x1'))
                by_region.setdefault(region, {})[channel] = (patch_index, item)
        if not by_region:
            raise ValueError(f"{pair['info']['image_layer']}: no supervised channels; assign masks or exclude this image.")
        locations = []
        for region, parts in sorted(by_region.items()):
            first = next(iter(parts.values()))[1]
            positives = [parts[c][1]['foreground_pixels_original'] if c in parts else 0 for c in range(len(channels))]
            known = [parts[c][1]['reviewed_pixels'] if c in parts else 0 for c in range(len(channels))]
            locations.append({**first, 'block': root(first['spatial_block']),
                              'channel_patch_indices': [parts[c][0] if c in parts else None for c in range(len(channels))],
                              'channel_positive_pixels': positives, 'channel_reviewed_pixels': known,
                              'reviewed_pixels': sum(known), 'foreground_pixels_original': sum(positives),
                              'contains_foreground': any(positives),
                              'kind': 'target' if any(positives) else 'reviewed background',
                              'class_ids': [c + 1 for c, value in enumerate(positives) if value],
                              'image_id': pair['info']['image_id'], 'image_name': pair['info']['image_layer']})
        source = MultilabelPatches(
            image=image, ground_truth=next(s.ground_truth for s in channels if s is not None),
            reviewed=None, crop_pixels=config.crop_pixels, input_pixels=config.input_pixels,
            locations=locations, blocks=[], reviewed_pixels=sum(p['reviewed_pixels'] for p in locations),
            total_pixels=int(np.prod(image.shape)) * len(channels), class_names=config.class_names,
            multilabel=True, channel_datasets=channels,
        )
        sources.append(source)
        infos.append({**pair['info'], 'image_shape': list(image.shape), 'multilabel': True})
        yield QuickTrainProgress(number + 1, len(pairs), f"Indexed channels for {pair['info']['image_layer']} ({number + 1}/{len(pairs)} images).")
    if len(sources) > 1:
        dummy_pairs = [dict(image=s.image, ground_truth=s.ground_truth, info=info) for s, info in zip(sources, infos)]
        return (yield from index_pairs(dummy_pairs, config, cancel=cancel,
                                       validation_fraction=validation_fraction, split_seed=split_seed,
                                       validation_method=validation_method, folds=folds,
                                       prior_training_image_ids=prior_training_image_ids,
                                       _indexed_sources=sources))
    dataset = sources[0]
    if validation_method == 'kfold':
        partitions = spatial_folds(dataset.locations, folds, split_seed)
        reserved = partitions[0]
    elif validation_method == 'holdout':
        if not 0.05 <= validation_fraction <= 0.5:
            raise ValueError('Validation holdout must be between 5% and 50%.')
        groups = sorted({p['block'] for p in dataset.locations if p['contains_foreground']})
        if len(groups) < 2:
            raise ValueError('Single-image multilabel validation needs at least two spatial groups containing targets. Review more regions or reduce patch size.')
        np.random.default_rng(split_seed).shuffle(groups)
        reserved, partitions = groups[:max(1, int(len(groups) * validation_fraction))], []
    else:
        raise ValueError('Choose holdout or k-fold validation.')
    dataset = partition_dataset(dataset, reserved)
    dataset.source_info = infos
    dataset.validation = dict(method=validation_method, unit='spatial', holdout_fraction=validation_fraction,
                              seed=split_seed, fold_count=folds if partitions else 1, folds=partitions,
                              validation_blocks=list(reserved))
    # A loaded model's previously trained pixels must not become a new holdout.
    dataset.validation_seen_before = any(
        p['split'] == 'validation' and p['y0'] < r['y0'] + r['size'] and p['y1'] > r['y0']
        and p['x0'] < r['x0'] + r['size'] and p['x1'] > r['x0']
        for p in dataset.locations for r in prior_training_regions)
    _check_class_coverage(dataset)
    return dataset
