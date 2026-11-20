"""Small, workspace-scoped recovery records for annotation workflow state.

This journal contains coordinates and metadata, never image/label pixel arrays.
Pixel writes continue to use MaskEntry's existing bounded recovery protocol.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, QTimer, Signal

KEY = 'label_assistant_tile_workflow'
FORMAT = 'label-assistant-tile-progress'


def active_path(viewer):
    value = getattr(viewer, '_label_assistant_progress_manifest', None)
    return Path(value) if value else None


def progress_path(manifest):
    from .workspace.service import workspace_data_root
    return workspace_data_root(Path(manifest)) / 'tile-progress.json'


def _reference(layer):
    from .workspace.service import _zarr_reference, _layer_file_source
    reference = _zarr_reference(layer)
    if reference:
        return dict(kind='zarr', path=str(Path(reference['path']).resolve()),
                    array_path=reference.get('array_path', ''))
    path, reader = _layer_file_source(layer)
    if path:
        return dict(kind='file', path=str(path.resolve()), reader_plugin=reader)
    return None


def _description(layer):
    from ._coordinates import mask_world_affine
    from ._quick_train import array_2d
    from .workspace.service import _common_record
    reference = _reference(layer)
    if reference is None:
        return None
    try:
        shape = list(array_2d(layer.data).shape)
        registration = mask_world_affine(layer).tolist()
    except (ValueError, np.linalg.LinAlgError):
        # Review tiles are 2D; unrelated RGB/volume layers remain in the main
        # workspace without preventing recovery of the annotation workflow.
        return None
    record = _common_record(layer, type(layer).__name__)
    record.update(storage=reference, shape=shape, registration=registration)
    return record


def _key(record):
    ref = record['storage']
    path = Path(ref['path'])
    if ref['kind'] == 'zarr':
        path = path / ref.get('array_path', '')
    return (record['layer_type'], ref['kind'], str(path.resolve()))


def capture_progress(viewer, manifest, *, reset=False):
    from ._dynamic_edit import is_internal_layer
    from ._mask_tiles import full_source
    manifest = Path(manifest).resolve()
    cache_key = str(manifest)
    cache = getattr(viewer, '_label_assistant_progress_records', {})
    records = {} if reset else dict(cache.get(cache_key, {}))
    for visible in viewer.layers:
        if is_internal_layer(visible):
            continue
        layer = full_source(visible)
        if type(layer).__name__ not in ('Labels', 'Image'):
            continue
        record = _description(layer)
        if record is not None:
            records[_key(record)] = record
    cache[cache_key] = records
    viewer._label_assistant_progress_records = cache
    return records


def write_progress(viewer, manifest, *, reset=False):
    from .workspace.service import _atomic_write_json
    manifest = Path(manifest).resolve()
    cache_key = str(manifest)
    records = capture_progress(viewer, manifest, reset=reset)
    digest = sha256(manifest.read_bytes()).hexdigest()
    bases = getattr(viewer, '_label_assistant_progress_bases', {})
    if not reset and cache_key in bases and bases[cache_key] != digest:
        raise ValueError('Workspace changed on disk. Save or reopen it before recording more tile progress.')
    bases[cache_key] = digest
    viewer._label_assistant_progress_bases = bases
    target = progress_path(manifest)
    needed = any(KEY in record or 'label_assistant_annotation_batches' in record or record.get('label_assistant_review', {}).get('role') == 'reviewed_regions'
                 for record in records.values())
    if not needed and not target.exists():
        return None
    payload = dict(format=FORMAT, version=1, manifest_digest=digest,
                   layers=list(records.values()))
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, payload)
    return target


def restore_progress(viewer, manifest):
    """Restore only records matching this manifest and registered source stores."""
    import json
    from ._dynamic_edit import is_internal_layer
    from ._mask_tiles import full_source
    from ._coordinates import mask_world_affine
    from ._quick_train import array_2d
    from .workspace.service import _restore_layer, _apply_common_state, workspace_data_root
    manifest = Path(manifest).resolve()
    viewer._label_assistant_progress_manifest = str(manifest)
    bases = getattr(viewer, '_label_assistant_progress_bases', {})
    bases[str(manifest)] = sha256(manifest.read_bytes()).hexdigest()
    viewer._label_assistant_progress_bases = bases
    path = progress_path(manifest)
    if not path.is_file():
        return []
    state = json.loads(path.read_text())
    if state.get('format') != FORMAT or state.get('version') != 1:
        raise ValueError('Unsupported tile progress record; workspace annotations were opened normally.')
    if state.get('manifest_digest') != sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Tile progress belongs to a different workspace revision and was not applied.')
    descriptions = state.get('layers', [])
    restored = []
    for record in descriptions:
        candidates = [layer for layer in viewer.layers if not is_internal_layer(layer)
                      and type(full_source(layer)).__name__ == record.get('layer_type')]
        match = None
        for layer in candidates:
            owner = full_source(layer)
            current = _description(owner)
            if current is not None and _key(current) == _key(record):
                if (current['shape'] != record['shape']
                        or not np.allclose(current['registration'], record['registration'], rtol=0, atol=1e-8)):
                    continue
                old_id = owner.metadata.get('label_assistant_review', {}).get('id')
                new_id = record.get('label_assistant_review', {}).get('id')
                if old_id and new_id and old_id != new_id:
                    continue
                match = layer
                break
        if match is None and record.get('layer_type') != 'Image':
            review = record.get('label_assistant_review', {})
            ref = record.get('storage', {})
            if (record.get('layer_type') == 'Labels' and review.get('role') == 'reviewed_regions'
                    and ref.get('kind') == 'zarr'
                    and Path(ref['path']).resolve().is_relative_to(workspace_data_root(manifest).resolve())):
                by_id = {full_source(layer).metadata.get('label_assistant_review', {}).get('id'): layer
                         for layer in viewer.layers if not is_internal_layer(layer)}
                truth, image = by_id.get(review.get('truth_id')), by_id.get(review.get('image_id'))
                if truth is not None and image is not None:
                    owner = full_source(truth)
                    if (list(array_2d(owner.data).shape) == record.get('shape')
                            and np.allclose(mask_world_affine(owner), record['registration'], rtol=0, atol=1e-8)):
                        match = _restore_layer(viewer, record, manifest_path=manifest)
                        if not hasattr(match, '_label_assistant_mask_entry'):
                            _apply_common_state(match, record, manifest_path=manifest)
        if match is not None:
            for owner in (match, full_source(match)):
                for key in ('label_assistant_review', 'label_assistant_grid', KEY, 'label_assistant_screening', 'label_assistant_annotation_batches'):
                    if key in record:
                        owner.metadata[key] = deepcopy(record[key])
            restored.append(match.name)
    return restored


class ProgressJournal(QObject):
    status = Signal(str)
    activity = Signal(str)

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.closed = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.flush)

    def schedule(self):
        if not self.closed and active_path(self.viewer) and not getattr(self.viewer, '_label_assistant_workspace_loading', False):
            capture_progress(self.viewer, active_path(self.viewer))
            self.activity.emit('queued')
            self.status.emit('Saving tile progress…')
            from ._activity import activity_for
            activity_for(self.viewer).report('journal', 'busy', 'Saving tile progress…', priority=20, log=False)
            self.timer.start()

    def flush(self):
        self.timer.stop()
        path = active_path(self.viewer)
        if self.closed or path is None or getattr(self.viewer, '_label_assistant_workspace_loading', False):
            return
        try:
            from ._activity import activity_for
            hub = activity_for(self.viewer)
            hub.report('journal', 'busy', 'Saving tile progress…', priority=20, log=False)
            self.activity.emit('saving')
            from .workspace.service import _commit_pending_local_edits
            _commit_pending_local_edits(self.viewer)
            write_progress(self.viewer, path)
            self.status.emit('Tile progress saved')
            hub.report('journal', 'ready', 'Tile progress saved', completed=1, total=1,
                       priority=20, progress_format='Tile progress saved')
            self.activity.emit('saved')
        except Exception as error:
            self.status.emit(f'Tile progress not saved: {error}')
            hub.report('journal', 'error', 'Tile progress not saved', details=str(error),
                       completed=0, total=1, priority=20, progress_format='Tile progress not saved')
            self.activity.emit('error')

    def shutdown(self):
        if not self.closed:
            self.flush()
            self.closed = True
            self.timer.stop()
