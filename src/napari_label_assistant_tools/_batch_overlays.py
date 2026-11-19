"""Coordinate-only batch history display shared by all annotation views."""
from contextlib import suppress
from weakref import WeakSet

import numpy as np
from qtpy.QtCore import QObject, QTimer

from . import _annotation_batches as batches
from ._mask_tiles import full_source


def review_state(viewer, image, truth):
    image_id, truth_id = batches._id(image), batches._id(truth)
    mask = next((layer for layer in viewer.layers if image_id and truth_id and
                 layer.metadata.get('label_assistant_review', {}).get('role') == 'reviewed_regions' and
                 layer.metadata['label_assistant_review'].get('image_id') == image_id and
                 layer.metadata['label_assistant_review'].get('truth_id') in (truth_id, None)), None)
    return mask, mask.metadata.get('label_assistant_review', {}) if mask is not None else {}


def display_regions(pair, state):
    """Clip older overlaps so the active batch's color always takes priority."""
    from ._component_library import subtract_rectangle
    from ._reviewed_regions import effective_areas, region_reviewed
    from math import ceil, sqrt
    if pair.get('view_mode', 'active') == 'hide':
        return []
    active = batches.active_batch(pair)
    records = [active]
    if pair.get('view_mode') == 'all':
        records += [record for record in reversed(pair['batches']) if record is not active]
    areas, covered, items = effective_areas(state), [], []
    exact = {tuple(area) for area in areas}
    # Bound the coordinate index to roughly the number of member regions, even
    # if one old region spans the entire image. Ordinary grid tiles then compare
    # only with nearby rectangles rather than every previously drawn tile.
    count = sum(len(record['regions']) for record in records)
    step = max(1, ceil(max(pair['shape']) / max(1, sqrt(count))))
    buckets = {}

    def keys(bounds):
        a,b,c,d = bounds
        return ((y,x) for y in range(a//step, (c-1)//step+1) for x in range(b//step, (d-1)//step+1))

    for record in records:
        for region in record['regions']:
            fragments = [list(region)]
            candidates = {index for key in keys(region) for index in buckets.get(key, ())}
            for index in sorted(candidates):
                previous = covered[index]
                fragments = [piece for fragment in fragments for piece in subtract_rectangle(fragment, previous)]
                if not fragments:
                    break
            ready = tuple(region) in exact or region_reviewed(region, areas)
            for bounds in fragments:
                items.append(dict(bounds=bounds, batch_id=record['id'], name=record['name'],
                                  color=batches.batch_color(record), active=record is active, ready=ready))
                index = len(covered)
                covered.append(bounds)
                for key in keys(bounds):
                    buckets.setdefault(key, []).append(index)
    return items


def controller(viewer):
    value = getattr(viewer, '_label_assistant_batch_overlay', None)
    if value is None or value.closed:
        value = BatchOverlay(viewer)
        viewer._label_assistant_batch_overlay = value
    return value


class BatchOverlay(QObject):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.clients = WeakSet()
        self.image = self.truth = self.layer = None
        self.closed = self.rendering = False
        self._stamp = None
        self._watched = []
        self._watch_ids = ()
        self._widths = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(75)
        self.timer.timeout.connect(self.render)
        self.events = batches.events(viewer)
        self.events.changed.connect(self.schedule)
        viewer.camera.events.zoom.connect(self.update_width)
        viewer.layers.events.removed.connect(self._removed)

    def release(self, widget):
        self.clients.discard(widget)
        if not self.clients:
            self.shutdown()

    def set_context(self, image, truth):
        if self.closed:
            return
        self.image, self.truth = image, truth
        mask, _ = review_state(self.viewer, image, truth)
        watched = [layer for layer in (image, truth, full_source(truth) if truth is not None else None, mask)
                   if layer is not None]
        watched = list({id(layer): layer for layer in watched}.values())
        identifiers = tuple(id(layer) for layer in watched)
        if identifiers != self._watch_ids:
            for emitter in self._watched:
                with suppress(Exception):
                    emitter.disconnect(self.schedule)
            self._watched = []
            self._watch_ids = identifiers
            for layer in watched:
                for name in ('metadata', 'scale', 'translate', 'rotate', 'shear', 'affine'):
                    emitter = getattr(getattr(layer, 'events', None), name, None)
                    if emitter is not None:
                        emitter.connect(self.schedule)
                        self._watched.append(emitter)
        self.schedule()

    def schedule(self, *_args):
        if not self.closed and not self.timer.isActive():
            self.timer.start()

    def render(self):
        if self.closed or self.rendering:
            return
        shell = getattr(self.viewer, '_label_assistant_curation_widget', None)
        if shell is not None and hasattr(shell, 'simple') and shell.simple.busy:
            self.schedule()
            return
        self.rendering = True
        try:
            self._render()
        finally:
            self.rendering = False

    def _render(self):
        from ._coordinates import mask_world_affine, crop_transform_kwargs
        from ._dynamic_edit import INTERNAL_LAYER_ROLE_KEY
        from qtpy.QtGui import QColor
        image, truth = self.image, self.truth
        if image is None or truth is None or not all(any(layer is item for item in self.viewer.layers) for layer in (image, truth)):
            self.hide()
            return
        try:
            pair = batches.get_pair(image, truth)
        except ValueError:
            self.hide()
            return
        if pair is None:
            self.hide()
            return
        _, state = review_state(self.viewer, image, truth)
        source = full_source(truth)
        root = source.metadata.get(batches.KEY)
        stamp = (id(image), id(truth), id(root), repr(state.get('areas')), repr(state.get('invalid_regions')),
                 tuple(mask_world_affine(source).ravel()))
        if stamp == self._stamp:
            self.raise_overlay()
            return
        self._stamp = stamp
        items = display_regions(pair, state)
        if not items:
            self.hide()
            return
        data, kinds, edges, faces, widths = [], [], [], [], []
        features = dict(batch_id=[], batch=[], review_status=[])
        for item in items:
            a,b,c,d = item['bounds']
            color = np.array(QColor(item['color']).getRgbF())
            face = color.copy()
            color[3] = 1 if item['active'] else .55
            face[3] = .10 if item['active'] else .025
            shapes = [('rectangle', [[a,b],[a,d],[c,d],[c,b]], color, face)]
            if not item['ready']:
                # Keep batch identity on the border; a red X flags stale review.
                face[3] = .02
                shapes += [('line', line, np.array([1,.25,.25,1]), np.zeros(4))
                           for line in ([[a,b],[c,d]], [[a,d],[c,b]])]
            for kind, points, edge, fill in shapes:
                data.append(np.asarray(points, dtype=float))
                kinds.append(kind)
                edges.append(edge)
                faces.append(fill)
                widths.append(2.5 if item['active'] or kind == 'line' else 1.5)
                features['batch_id'].append(item['batch_id'])
                features['batch'].append(item['name'])
                features['review_status'].append('Reviewed' if item['ready'] else 'Needs review')
        selected = list(self.viewer.layers.selection)
        active = self.viewer.layers.selection.active
        try:
            if self.layer is None or not any(self.layer is layer for layer in self.viewer.layers):
                self.layer = self.viewer.add_shapes(data, shape_type=kinds, name='Annotation batches',
                    edge_color=np.asarray(edges), face_color=np.asarray(faces), edge_width=widths,
                    opacity=1, features=features, **crop_transform_kwargs(source),
                    metadata={INTERNAL_LAYER_ROLE_KEY: 'annotation_batches'})
                self.layer.editable = False
            else:
                self.layer.data = list(zip(data, kinds))
                self.layer.features = features
                self.layer.edge_color = np.asarray(edges)
                self.layer.face_color = np.asarray(faces)
                for name in ('scale','translate','rotate','shear','affine'):
                    setattr(self.layer, name, getattr(source, name))
                self.layer.visible = True
            self._widths = widths
            self.update_width()
            self.raise_overlay()
        finally:
            selection = self.viewer.layers.selection
            selection.clear()
            selection.update(layer for layer in selected if any(layer is item for item in self.viewer.layers))
            if active is not None and any(active is layer for layer in self.viewer.layers):
                selection.active = active

    def raise_overlay(self):
        layer = self.layer
        if layer is None or not layer.visible or not any(layer is item for item in self.viewer.layers):
            return
        # Overall coverage can be created after approval. Keep batch colors above
        # it, while yellow pending-selection outlines remain on top.
        pending = [item for item in self.viewer.layers if item.name in ('Pending review tiles','Selected review cell')
                   and item.metadata.get('label_assistant_internal_role') == 'review_cells'
                   and item.visible and len(item.data)]
        desired = [item for item in self.viewer.layers if item is not layer and all(item is not p for p in pending)] + [layer] + pending
        if list(self.viewer.layers) != desired:
            for position, item in enumerate(desired):
                current = self.viewer.layers.index(item)
                if current != position:
                    self.viewer.layers.move(current, position)

    def update_width(self, *_args):
        if self.layer is None or not self._widths or not any(self.layer is layer for layer in self.viewer.layers):
            return
        from ._coordinates import mask_world_affine
        scale = np.linalg.svd(mask_world_affine(self.layer)[:2,:2], compute_uv=False)[-1] * self.viewer.camera.zoom
        self.layer.edge_width = (np.asarray(self._widths) / max(float(scale), 1e-8)).tolist()

    def hide(self):
        self._stamp = None
        if self.layer is not None and any(self.layer is layer for layer in self.viewer.layers):
            self.layer.visible = False

    def _removed(self, event):
        if event.value is self.image or event.value is self.truth:
            self.set_context(None, None)
            self.hide()
        if event.value is self.layer:
            self.layer = None
            self._stamp = None

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        for emitter, callback in [(self.events.changed, self.schedule),
                                  (self.viewer.camera.events.zoom, self.update_width),
                                  (self.viewer.layers.events.removed, self._removed)]:
            with suppress(Exception):
                emitter.disconnect(callback)
        for emitter in self._watched:
            with suppress(Exception):
                emitter.disconnect(self.schedule)
        with suppress(Exception):
            self.hide()
