"""Reproducible, metadata-only sampling after the spatial validation split."""
from dataclasses import replace
from collections import deque
from hashlib import sha256
import json

import numpy as np


def select_training_patches(dataset, *, mode='all', percentage=10, count=100, seed=42):
    """Select training coordinates; keep every reserved validation patch.

    Source arrays, reviewed masks, and spatial partitions are shared unchanged.
    The chosen subset stays fixed across epochs; the training loop still shuffles
    it and applies its existing additional target-patch visits.
    """
    if mode == 'all':
        return dataset
    if mode not in ('percentage', 'count'):
        raise ValueError('Choose all training patches, a percentage, or a patch count.')
    training = dataset.indices('training')
    if not training:
        raise ValueError('No eligible training patches are available.')
    if mode == 'percentage':
        if not np.isfinite(percentage) or not 0 < percentage <= 100:
            raise ValueError('Training percentage must be greater than 0 and at most 100.')
        requested = max(1, int(np.ceil(len(training) * percentage / 100)))
    else:
        if int(count) != count or count < 1:
            raise ValueError('Choose at least one training patch.')
        requested = min(int(count), len(training))
    rng = np.random.default_rng(seed)
    positive = [i for i in training if dataset.locations[i]['contains_foreground']]
    background = [i for i in training if not dataset.locations[i]['contains_foreground']]
    if not positive:
        raise ValueError('The training partition has no confirmed foreground.')
    if requested == len(training):
        selected = set(training)
    elif requested == 1 and not dataset.multilabel:
        mixed = [i for i in positive if dataset.locations[i]['reviewed_pixels'] >
                 dataset.locations[i]['foreground_pixels_original']]
        if not mixed:
            raise ValueError('One patch cannot supply both confirmed foreground and background. Choose more patches.')
        selected = {int(rng.choice(mixed))}
    else:
        # Proportional positive/background sampling avoids an accidentally empty
        # foreground subset while retaining the pool's patch-level balance.
        n_positive = max(1, int(np.floor(requested * len(positive) / len(training) + .5)))
        if background and requested >= 2:
            n_positive = min(n_positive, requested - 1)
        n_background = min(len(background), requested - min(n_positive, len(positive)))
        n_positive = requested - n_background
        selected = set(map(int, rng.permutation(positive)[:n_positive]))
        selected.update(map(int, rng.permutation(background)[:n_background]))
    locations = [item for i, item in enumerate(dataset.locations)
                 if item['split'] == 'validation' or i in selected]
    rows = [dataset.locations[i] for i in selected]
    if (not dataset.multilabel and sum(p['reviewed_pixels'] for p in rows) <=
            sum(p['foreground_pixels_original'] for p in rows)):
        raise ValueError('This subset has no confirmed background. Increase the patch amount or change the subset seed.')
    result = replace(dataset, locations=locations,
                     reviewed_pixels=sum(p['reviewed_pixels'] for p in locations),
                     training_selection=dict(mode=mode, percentage=float(percentage) if mode == 'percentage' else None,
                         requested_count=int(count) if mode == 'count' else None, seed=int(seed),
                         available_training_patches=len(training), selected_training_patches=len(selected),
                         validation_patches=len(dataset.indices('validation')), validation_unchanged=True,
                         strategy='fixed foreground/background-stratified subset'))
    from ._coverage_train import _check_class_coverage
    try:
        _check_class_coverage(result)
    except ValueError as error:
        if str(error).startswith('Training'):
            raise ValueError(f'{error} Increase the training patch amount or change the subset seed.') from error
        raise
    return result


def _spatial_order(dataset, ids, rng):
    """Round-robin shuffled reserved groups without reading any image pixels."""
    sources = {}
    for index in ids:
        row = dataset.locations[index]
        groups = sources.setdefault(row.get('source_index', 0), {})
        groups.setdefault(row.get('block', ''), []).append(index)
    sources_queues = []
    for groups in sources.values():
        queues = [rng.permutation(rows).tolist() for rows in groups.values()]
        sources_queues.append(deque(queues[i] for i in rng.permutation(len(queues))))
    sources_queues = deque(sources_queues[i] for i in rng.permutation(len(sources_queues)))
    order = []
    while sources_queues:
        queues = sources_queues.popleft()
        queue = queues.popleft()
        order.append(queue.pop())
        if queue:
            queues.append(queue)
        if queues:
            sources_queues.append(queues)
    return order


def check_validation_supervision(dataset):
    """Reject undefined target scores rather than reporting background-only Dice."""
    from ._coverage_train import _check_class_coverage
    _check_class_coverage(dataset)
    rows = [p for p in dataset.locations if p['split'] == 'validation']
    if not rows:
        raise ValueError('No reserved validation patches are available.')
    if not dataset.class_names:
        foreground = sum(p['foreground_pixels_original'] for p in rows)
        known = sum(p['reviewed_pixels'] for p in rows)
        if foreground <= 0 or known <= foreground:
            raise ValueError('Validation needs confirmed foreground and background for meaningful target scores.')


def select_trial_validation(dataset, *, percentage=10, seed=42):
    """Sample only existing holdout coordinates; training and sources stay shared."""
    if dataset.validation.get('method') == 'kfold':
        raise ValueError('Quick trial validation requires spatial/image holdout, not k-fold.')
    if not np.isfinite(percentage) or not 0 < percentage <= 100:
        raise ValueError('Trial validation percentage must be greater than 0 and at most 100.')
    validation = dataset.indices('validation')
    check_validation_supervision(dataset)
    requested = max(1, int(np.ceil(len(validation) * percentage / 100)))
    if requested == len(validation):
        return dataset
    rng = np.random.default_rng(seed)
    positive = [i for i in validation if dataset.locations[i]['contains_foreground']]
    background = [i for i in validation if not dataset.locations[i]['contains_foreground']]
    if requested == 1:
        mixed = [i for i in positive if dataset.locations[i]['reviewed_pixels'] >
                 dataset.locations[i]['foreground_pixels_original']]
        selected = set(_spatial_order(dataset, mixed, rng)[:1])
    else:
        n_positive = max(1, int(np.floor(requested * len(positive) / len(validation) + .5)))
        if background:
            n_positive = min(n_positive, requested - 1)
        n_background = min(len(background), requested - min(n_positive, len(positive)))
        n_positive = requested - n_background
        selected = set(_spatial_order(dataset, positive, rng)[:n_positive])
        selected.update(_spatial_order(dataset, background, rng)[:n_background])
    locations = [p for i, p in enumerate(dataset.locations) if p['split'] == 'training' or i in selected]
    rows = [p for p in locations if p['split'] == 'validation']
    info = dict(mode='trial', percentage=float(percentage), seed=int(seed),
                available_patches=len(validation), selected_patches=len(rows),
                positive_patches=sum(p['contains_foreground'] for p in rows),
                images_selected=len({p.get('source_index', 0) for p in rows}),
                images_available=len({dataset.locations[i].get('source_index', 0) for i in validation}),
                spatial_groups=len({(p.get('source_index', 0), p.get('block', '')) for p in rows}),
                strategy='fixed foreground/background-stratified sample spread across reserved spatial/image groups',
                subset_id=sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    selection = dict(dataset.training_selection)
    if selection:
        selection.update(validation_unchanged=False, validation_patches=len(rows))
    result = replace(dataset, locations=locations, reviewed_pixels=sum(p['reviewed_pixels'] for p in locations),
                     validation={**dataset.validation, 'evaluation': info}, training_selection=selection)
    try:
        check_validation_supervision(result)
    except ValueError as error:
        raise ValueError(f'Trial validation: {error} Increase validation percentage.') from error
    return result
