"""Compact, checked screening selections; no source pixel arrays are copied.

Analysis component numbers are temporary. Persist geometry and a fingerprint of
the component's footprint instead, and match against a newly built index.
"""
from hashlib import blake2b

import numpy as np

KEY = "label_assistant_screening"


def context(shape, geometry):
    from ._coordinates import mask_world_affine
    return dict(shape=list(map(int, shape)), affine=mask_world_affine(geometry).tolist())


def record_key(record):
    return tuple(int(getattr(record, name)) for name in (
        'label_value', 'area', 'bbox_y0', 'bbox_x0', 'bbox_y1', 'bbox_x1'))


def describe(index, component_id):
    """Hash only cached ID tiles intersecting this component, never source data."""
    record = index.records[component_id]
    digest = blake2b(digest_size=20)
    # Canonical blocks make the fingerprint independent of index numbering,
    # index chunk size, and switching between in-memory and disk-backed indices.
    for y in range(record.bbox_y0, record.bbox_y1, 1024):
        for x in range(record.bbox_x0, record.bbox_x1, 1024):
            y1, x1 = min(y + 1024, record.bbox_y1), min(x + 1024, record.bbox_x1)
            bounds = [y, x, y1, x1]
            if hasattr(index, 'component_mask'):
                mask = index.component_mask(component_id, bounds)
            else:
                mask = index.component_id_map[y:y1, x:x1] == component_id
            digest.update(np.asarray(bounds, dtype='<i8').tobytes())
            digest.update(np.packbits(mask).tobytes())
    return dict(key=list(record_key(record)), footprint=digest.hexdigest())


def restore(index, saved, current_context, cache):
    """Changed/ambiguous objects are excluded, even when their numbers match."""
    components = saved.get('components', [])
    if saved.get('context') != current_context:
        return set(), len(components)
    by_key = {}
    for record in index.active_records():
        by_key.setdefault(record_key(record), []).append(record.component_id)
    selected = set()
    for component in components:
        if not isinstance(component, dict):
            continue
        key = component.get('key')
        if not isinstance(key, list) or len(key) != 6 or not all(isinstance(value, int) for value in key):
            continue
        matches = by_key.get(tuple(key), [])
        if len(matches) == 1:
            component_id = matches[0]
            if component_id not in cache:
                cache[component_id] = describe(index, component_id)
            if cache[component_id] == component:
                selected.add(component_id)
    return selected, len(components) - len(selected)


class SelectionOverlay:
    """Selection boxes cost four coordinates per object, not image-sized RAM."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.layer = None

    def update(self, records, geometry, visible):
        from ._coordinates import apply_crop_transform, crop_transform_kwargs
        from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY
        if not records or not visible:
            self.hide()
            return
        boxes = []
        for record in records:
            y0, x0 = record.bbox_y0 - .5, record.bbox_x0 - .5
            y1, x1 = record.bbox_y1 - .5, record.bbox_x1 - .5
            boxes.append(np.asarray([[y0, x0], [y0, x1], [y1, x1], [y1, x0]]))
        if self.layer is None or not any(layer is self.layer for layer in self.viewer.layers):
            active = self.viewer.layers.selection.active
            self.layer = self.viewer.add_shapes(
                boxes, shape_type='rectangle', name='Screening selection',
                edge_color='#00ffff', edge_width=2, face_color='transparent',
                metadata={INTERNAL_LAYER_ROLE_KEY: 'screening_selection'},
                **crop_transform_kwargs(geometry),
            )
            self.layer.editable = False
            if active is not None and any(layer is active for layer in self.viewer.layers):
                self.viewer.layers.selection.active = active
        else:
            self.layer.data = boxes
            apply_crop_transform(self.layer, geometry)
        self.layer.visible = True

    def hide(self):
        if self.layer is not None:
            self.layer.visible = False

    def close(self):
        if self.layer is not None and any(layer is self.layer for layer in self.viewer.layers):
            self.viewer.layers.remove(self.layer)
        self.layer = None
