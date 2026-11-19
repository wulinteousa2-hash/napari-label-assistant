"""Named annotation worksets: coordinates and review events, never pixel copies."""
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

KEY = 'label_assistant_annotation_batches'


def batch_color(batch):
    """Stable fallback for older batches; new records persist their color."""
    color = batch.get('color', '')
    if isinstance(color, str) and len(color) == 7 and color.startswith('#'):
        try:
            int(color[1:], 16)
            return color
        except ValueError:
            pass
    palette = ('#56b4e9', '#bb86fc', '#2ec4b6', '#f781bf', '#ff8c42', '#a6cee3', '#b3de69', '#fb8072')
    number = max(1, int(batch.get('number', 1)))
    if number <= len(palette):
        return palette[number-1]
    from colorsys import hsv_to_rgb
    hue = (number * .61803398875) % 1
    if .11 < hue < .19:  # Yellow belongs to pending selections.
        hue += .12
    return '#' + ''.join(f'{round(c*255):02x}' for c in hsv_to_rgb(hue, .55, .95))


def events(viewer):
    from napari.utils.events import EmitterGroup
    value = getattr(viewer, '_label_assistant_batch_events', None)
    if value is None:
        value = EmitterGroup(changed=None)
        viewer._label_assistant_batch_events = value
    return value


def _owner(truth):
    from ._mask_tiles import full_source
    return full_source(truth)


def _id(layer):
    return layer.metadata.get('label_assistant_review', {}).get('id') if layer is not None else None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _shape(truth):
    from ._quick_train import array_2d
    from ._mask_tiles import full_data
    return tuple(array_2d(full_data(truth)).shape)


def _save(viewer, truth, root, **event):
    owner = _owner(truth)
    owner.metadata = {**owner.metadata, KEY: root}
    if truth is not owner:
        truth.metadata = {**truth.metadata, KEY: root}
    entry = getattr(truth, '_label_assistant_mask_entry', None)
    if entry is not None:
        entry.sync_metadata()
    events(viewer).changed(**event)
    journal = getattr(viewer, '_label_assistant_progress_journal', None)
    if journal is not None:
        journal.schedule()


def get_pair(image, truth):
    if image is None or truth is None:
        return None
    root = _owner(truth).metadata.get(KEY, {})
    if root and root.get('version') != 1:
        raise ValueError('Unsupported annotation-batch metadata. Existing annotations are preserved.')
    pair = root.get('pairs', {}).get(_id(image))
    if pair is not None:
        from ._coordinates import mask_world_affine
        if pair.get('truth_id') != _id(truth):
            raise ValueError('Annotation batches belong to another Labels source.')
        if tuple(pair['shape']) != _shape(truth) or not np.allclose(
                pair['registration'], mask_world_affine(_owner(truth)), rtol=0, atol=1e-8):
            raise ValueError('Annotation batch dimensions/registration changed. Restore the source registration before using these batches.')
    return pair


def _commit(viewer, image, truth, pair, **event):
    root = dict(_owner(truth).metadata.get(KEY, {'version': 1, 'pairs': {}}))
    root['pairs'] = {**root.get('pairs', {}), _id(image): pair}
    _save(viewer, truth, root, image_id=_id(image), truth_id=_id(truth), **event)


def ensure_pair(viewer, image, truth):
    from ._reviewed_regions import identity, associated_mask
    from ._coordinates import mask_world_affine
    from ._coordinates import matching_mask_transforms
    from ._quick_train import array_2d
    if _shape(truth) != tuple(array_2d(image.data).shape) or not matching_mask_transforms(image, _owner(truth)):
        raise ValueError('Annotation batches require matching 2D image/Labels dimensions and registration.')
    identity(image)
    identity(truth)
    pair = get_pair(image, truth)
    if pair is not None:
        return pair
    mask = associated_mask(viewer, image, truth)
    state = mask.metadata.get('label_assistant_review', {}) if mask is not None else {}
    regions = [list(map(int, r)) for r in state.get('review_cells', state.get('areas', []))]
    identifier = uuid4().hex
    batch = dict(id=identifier, number=1, name='Batch 1', created_at_utc=None if regions else _now(),
                 updated_at_utc=None, finished_at_utc=None, revision=0, regions=regions, events=[],
                 origin='existing review membership; historical approval times unavailable' if regions else 'new')
    batch['color'] = batch_color(batch)
    pair = dict(image_id=_id(image), truth_id=_id(truth), shape=list(_shape(truth)),
                registration=mask_world_affine(_owner(truth)).tolist(),
                active_id=identifier, training_id=identifier, scope='all', view_mode='active', batches=[batch])
    _commit(viewer, image, truth, pair)
    return pair


def active_batch(pair):
    return next((b for b in pair['batches'] if b['id'] == pair['active_id']), None) if pair else None


def training_batch(pair):
    identifier = pair.get('training_id', pair['active_id'])
    return next(b for b in pair['batches'] if b['id'] == identifier)


def change(viewer, image, truth, action, *, batch_id=None, name=None, regions=()):
    from ._component_library import subtract_rectangle
    pair = deepcopy(ensure_pair(viewer, image, truth))
    pair.setdefault('training_id', pair['active_id'])
    for record in pair['batches']:
        record['color'] = batch_color(record)
    batch = active_batch(pair)
    stamp = _now()
    if action == 'new':
        if not (len(pair['batches']) == 1 and not batch['regions'] and not batch['events']
                and not batch['finished_at_utc'] and pair['scope'] == 'all'):
            number = max(b['number'] for b in pair['batches']) + 1
            batch = dict(id=uuid4().hex, number=number, name=f'Batch {number}', created_at_utc=stamp,
                         updated_at_utc=stamp, finished_at_utc=None, revision=0, regions=[], events=[], origin='new')
            batch['color'] = batch_color(batch)
            pair['batches'].append(batch)
        pair.update(active_id=batch['id'], training_id=batch['id'], scope='active')
        if pair.get('view_mode') == 'hide':
            pair['view_mode'] = 'active'
    elif action == 'select':
        if not any(b['id'] == batch_id for b in pair['batches']):
            raise ValueError('This annotation batch is no longer available.')
        pair.update(active_id=batch_id)
        batch = active_batch(pair)
        if pair.get('view_mode') == 'hide':
            pair['view_mode'] = 'active'
    elif action == 'training':
        if not any(b['id'] == batch_id for b in pair['batches']):
            raise ValueError('This training batch is no longer available.')
        pair.update(training_id=batch_id, scope='active')
    elif action == 'view':
        if name not in ('active', 'all', 'hide'):
            raise ValueError('Choose Active batch, All batches, or Hide.')
        pair['view_mode'] = name
    elif action in ('all', 'active'):
        pair['scope'] = action
        if action == 'active':
            pair['training_id'] = pair['active_id']
    elif action == 'rename':
        if not name or not name.strip():
            raise ValueError('Give the annotation batch a name.')
        batch['name'] = name.strip()
    elif action in ('finish', 'reopen'):
        batch['finished_at_utc'] = stamp if action == 'finish' else None
    elif action in ('approve', 'add', 'revoke', 'remove'):
        if batch['finished_at_utc'] and action in ('approve', 'add'):
            raise ValueError('This batch is finished. Reopen it or start a new batch before approving tiles.')
        for rectangle in regions:
            rect = list(map(int, rectangle))
            if not (len(rect) == 4 and 0 <= rect[0] < rect[2] <= pair['shape'][0]
                    and 0 <= rect[1] < rect[3] <= pair['shape'][1]):
                raise ValueError('Annotation batch region is outside this source image.')
            if action in ('approve', 'add') and rect not in batch['regions']:
                batch['regions'].append(rect)
            if action == 'remove':
                batch['regions'] = [piece for area in batch['regions'] for piece in subtract_rectangle(area, rect)]
            batch['events'].append(dict(action=action, bounds=rect, at_utc=stamp))
        batch['revision'] += 1
    else:
        raise ValueError('Unknown annotation batch action.')
    if action not in ('select', 'all', 'active', 'training', 'view'):
        batch['updated_at_utc'] = stamp
    _commit(viewer, image, truth, pair, action=action)
    return batch


def restore_membership(viewer, image, truth, before):
    pair = deepcopy(get_pair(image, truth))
    if pair is None:
        return
    for batch in pair['batches']:
        original = next((b for b in before['batches'] if b['id'] == batch['id']), None)
        if original is not None and batch['regions'] != original['regions']:
            batch['regions'] = deepcopy(original['regions'])
            batch['revision'] += 1
            batch['updated_at_utc'] = _now()
            batch['events'].append(dict(action='undo review', at_utc=batch['updated_at_utc']))
    _commit(viewer, image, truth, pair)


def selection(image, truth):
    pair = get_pair(image, truth)
    if pair is None or pair['scope'] == 'all':
        return dict(scope='all', image_id=_id(image), truth_id=_id(truth))
    batch = training_batch(pair)
    return dict(scope='active', image_id=pair['image_id'], truth_id=pair['truth_id'],
                id=batch['id'], name=batch['name'], number=batch['number'], revision=batch['revision'],
                regions=deepcopy(batch['regions']), created_at_utc=batch['created_at_utc'])


def signature(image, truth):
    pair = get_pair(image, truth)
    if pair is None or pair['scope'] == 'all':
        return ('all',)
    batch = training_batch(pair)
    return ('active', batch['id'], batch['revision'])


def preparation_error(image, truth):
    try:
        pair = get_pair(image, truth)
        if pair is not None and pair['scope'] == 'active' and not training_batch(pair)['regions']:
            return f"{training_batch(pair)['name']} is empty. Approve tiles or add existing reviewed regions before preparing this batch."
    except ValueError as error:
        return str(error)
    return ''


class BatchSupervision:
    """Restrict existing validity to batch coordinates; do not copy large masks."""
    def __init__(self, base, regions):
        self.base = base
        self.regions = tuple(tuple(r) for r in regions)
        self.shape, self.dtype = base.shape, np.dtype('uint8')

    def __getattr__(self, name):
        return getattr(self.base, name)

    def candidate_tiles(self, size):
        return sorted({(y, x) for a,b,c,d in self.regions
                       for y in range(a//size, (c-1)//size+1) for x in range(b//size, (d-1)//size+1)})

    def __getitem__(self, region):
        ys, xs = region
        allowed = np.zeros((ys.stop-ys.start, xs.stop-xs.start), dtype=bool)
        for a,b,c,d in self.regions:
            y0,x0,y1,x1 = max(a,ys.start),max(b,xs.start),min(c,ys.stop),min(d,xs.stop)
            if y0 < y1 and x0 < x1:
                allowed[y0-ys.start:y1-ys.start,x0-xs.start:x1-xs.start] = True
        return (np.asarray(self.base[region], dtype=bool) & allowed).astype('uint8')


def restrict_supervision(viewer, image, truth, base):
    info = selection(image, truth)
    if info['scope'] == 'all':
        return base
    if not info['regions']:
        raise ValueError(f"{info['name']} is empty. Approve tiles or add existing reviewed tiles to this batch first.")
    if base is None:
        from ._reviewed_regions import associated_mask
        from ._component_library import TrainingSupervision
        from ._mask_tiles import full_data
        mask = associated_mask(viewer, image, truth)
        if mask is None:
            raise ValueError('Confirm reviewed regions before preparing an annotation batch.')
        state = mask.metadata.get('label_assistant_review', {})
        base = TrainingSupervision(full_data(truth), full_data(mask), reviewed_areas=state.get('areas'),
                                   invalid_regions=state.get('invalid_regions', []))
    return BatchSupervision(base, info['regions'])


def protect_validation(dataset, model, *, image_id=None, image_name=''):
    """Keep earlier holdout coordinates reserved when preparing another batch."""
    from ._coverage_train import partition_dataset, _check_class_coverage
    from dataclasses import replace
    size = dataset.crop_pixels
    reserved, trained, reserved_images = set(), set(), set()

    def add(target, identifier, rect):
        a,b,c,d = rect
        for y in range(max(0,a)//size, (c-1)//size+1):
            for x in range(max(0,b)//size, (d-1)//size+1):
                target.add((identifier,y,x))

    for history in model.history:
        source = history.get('source', {})
        pairs = source.get('pairs', [])
        identifier = source.get('image_id') or (image_id if source.get('image_layer') == image_name else None)
        if len(pairs) == 1:
            identifier = pairs[0].get('image_id')
        if len(pairs) > 1:
            held = set(history.get('validation_blocks', []))
            reserved_images.update(p['image_id'] for p in pairs
                                   if str(p.get('split_group') or p['image_id']) in held)
        for patch in history.get('locations', []):
            target = reserved if patch['split'] == 'validation' else trained
            key = patch.get('image_id', identifier)
            if key is None:
                continue
            add(target, key, (patch['y0'],patch['x0'],patch.get('y1',patch['y0']+patch.get('size',size)),
                              patch.get('x1',patch['x0']+patch.get('size',size))))
        # Quick trials retain only selected validation patches, but history also
        # records all held-out raw spatial blocks. Protect their omitted areas.
        shape = source.get('image_shape') or (pairs[0].get('image_shape') if len(pairs) == 1 else None)
        old_size = history.get('config', {}).get('crop_pixels')
        if identifier is not None and shape and old_size and len(pairs) <= 1:
            height, width = shape[-2:]
            gh = old_size * (2 if (height+old_size-1)//old_size >= 4 else 1)
            gw = old_size * (2 if (width+old_size-1)//old_size >= 4 else 1)
            for block in history.get('validation_blocks', []):
                parts = str(block).split(':')
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    y,x = map(int, parts)
                    add(reserved, identifier, (y*gh,x*gw,min((y+1)*gh,height),min((x+1)*gw,width)))
    if not reserved and not reserved_images:
        return dataset
    held = {b['id'] for b in dataset.blocks if b['split'] == 'validation'}
    held.update(p['block'] for p in dataset.locations if p.get('image_id',image_id) in reserved_images or
                (p.get('image_id',image_id),p['y0']//size,p['x0']//size) in reserved)
    result = partition_dataset(dataset, held)
    seen = dataset.validation_seen_before
    if dataset.sources:
        trained_images = {key for key,_,_ in trained}
        seen = seen or any(p['split'] == 'validation' and p.get('image_id') in trained_images for p in result.locations)
    for p in result.locations:
        if p['split'] != 'validation':
            continue
        key = p.get('image_id',image_id)
        source = dataset.sources[p['source_index']] if dataset.sources else dataset
        h,w = source.image.shape[-2:]
        gh = 2 if (h+size-1)//size >= 4 else 1
        gw = 2 if (w+size-1)//size >= 4 else 1
        y,x = p['y0']//(size*gh)*gh, p['x0']//(size*gw)*gw
        if any((key,row,col) in trained for row in range(y,y+gh) for col in range(x,x+gw)):
            seen = True
            break
    result = replace(result, validation_seen_before=seen,
                     validation={**dataset.validation, 'preserved_prior_validation': True})
    for split in ('training', 'validation'):
        rows = [p for p in result.locations if p['split'] == split]
        if not rows or not any(p['contains_foreground'] for p in rows) or not any(
                p['reviewed_pixels'] > p['foreground_pixels_original'] for p in rows):
            raise ValueError(f'{split.title()} needs independent reviewed targets and background. Earlier holdout regions stay reserved; add more reviewed regions to this batch.')
    _check_class_coverage(result)
    return result
