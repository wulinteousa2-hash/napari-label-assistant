"""Keep full mask stores outside napari; expose only bounded editable crops."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import uuid

import numpy as np
from napari.utils.transforms import Affine

from ._coordinates import crop_transform_kwargs

KEY = 'label_assistant_mask_tile'
MANAGER = '_label_assistant_mask_tiles'
MODE = '_label_assistant_mask_loading_mode'
DEFAULT_SIZE = 1024


def manager_for(viewer):
    return getattr(viewer, MANAGER, None)


def full_source(layer):
    """Full source for scientific operations, never for adding to the viewer."""
    entry = getattr(layer, '_label_assistant_mask_entry', None)
    return entry.source if entry is not None else layer


def full_data(layer):
    return full_source(layer).data


class Labels(SimpleNamespace):
    """Serialization/geometry adapter; deliberately not a napari Layer."""
    def __init__(self, **values):
        super().__init__()
        for name,value in values.items():
            setattr(self,name,value)

    def __setattr__(self, name, value):
        if name == 'affine' and not isinstance(value, Affine):
            value = Affine(affine_matrix=value)
        super().__setattr__(name, value)
    def data_to_world(self, point):
        base = Affine(scale=self.scale, translate=self.translate,
                      rotate=self.rotate, shear=self.shear)
        return tuple(self.affine(base(point)))


class MaskEntry:
    def __init__(self, manager, record, array, reference):
        self.manager = manager
        self.layer = None
        self.bounds = None
        self.original = None
        self.pending_invalidations = []
        self.pending_pixels = None
        self.source = Labels(data=array, name=record.get('name', 'Labels'),
            scale=np.asarray(record.get('scale') or [1, 1]),
            translate=np.asarray(record.get('translate') or [0, 0]),
            rotate=np.asarray(record.get('rotate', np.eye(2))),
            shear=np.asarray(record.get('shear', [0.])),
            affine=Affine(affine_matrix=np.asarray(record.get('affine', np.eye(3)))),
            visible=record.get('visible', True), opacity=record.get('opacity', .7),
            blending=record.get('blending', 'translucent'), metadata={})
        for key in ('spectral_assistant', 'quick_train', 'label_assistant_grid',
                    'label_assistant_review', 'label_assistant_component_library'):
            if key in record:
                self.source.metadata[key] = deepcopy(record[key])
        from .workspace.service import _remember_layer_storage
        _remember_layer_storage(self.source, reference)
        self.reference = reference

    def _recovery_dir(self):
        reference = self.source.metadata['label_assistant_workspace_storage']
        return Path(reference['path']).parent / '.label_assistant_tile_recovery'

    def flush(self):
        if self.layer is None or self.original is None:
            return
        if self.manager.stroke_active:
            raise ValueError('Finish the current paint or erase stroke before saving or changing tiles.')
        if getattr(self,'pending_roi_writes',False):
            changed=np.asarray(self.layer.data)!=self.original
            desired=np.asarray(self.layer.data).copy()
            self.recover()
            if self.bounds is not None:
                baseline=np.asarray(self.source.data[self.bounds]).copy()
                pixels=baseline.copy();pixels[changed]=desired[changed]
                self.manager.loading=True
                try:self.layer.data=pixels
                finally:self.manager.loading=False
                self.original=baseline
            self.pending_roi_writes=False
            for bounds in self.pending_invalidations:
                self.invalidate(bounds)
            self.pending_invalidations.clear()
        if self.pending_pixels is not None:
            # Complete an interrupted write before comparing subsequent edits,
            # including an erase back to the old pre-failure background value.
            self.recover()
            self.original = self.pending_pixels
            self.pending_pixels = None
            for bounds in self.pending_invalidations:
                self.invalidate(bounds)
            self.pending_invalidations.clear()
        pixels = np.asarray(self.layer.data)
        if pixels.shape != self.original.shape or pixels.dtype.kind not in 'ui':
            raise ValueError(f'{self.layer.name}: tile shape/type changed. Restore this tile before continuing.')
        changed = pixels != self.original
        self.sync_metadata()
        if self.bounds is None:
            if changed.any():
                raise ValueError('This mask has no pixels in the current tile; edits cannot be applied here.')
            return
        if not changed.any():
            return
        from ._coordinates import mask_world_affine
        expected = mask_world_affine(self.source)
        expected[:2, 2] += expected[:2, :2] @ [self.bounds[0].start, self.bounds[1].start]
        actual = mask_world_affine(self.layer)
        if not np.allclose(actual, expected, rtol=0, atol=1e-8):
            raise ValueError('The crop transform changed. Restore its original registration before applying edits.')
        y, x = np.nonzero(changed)
        local = (slice(int(y.min()), int(y.max())+1), slice(int(x.min()), int(x.max())+1))
        region = tuple(slice(base.start+part.start,base.start+part.stop) for base,part in zip(self.bounds,local))
        mask, patch, original = changed[local], pixels[local], self.original[local]
        limits = np.iinfo(self.source.data.dtype)
        values = patch[mask]
        if values.min() < limits.min or values.max() > limits.max:
            raise ValueError(f'{self.layer.name}: label values exceed the full mask dtype {self.source.data.dtype}. No pixels written.')
        current = np.asarray(self.source.data[region]).copy()
        if np.any(mask & (current != original) & (current != patch)):
            raise ValueError(f'{self.layer.name}: another writer changed these pixels. Keep this tile open and resolve the conflict before saving.')
        root = self._recovery_dir()
        root.mkdir(parents=True, exist_ok=True)
        journal = root / f'{uuid.uuid4().hex}.npz'
        reference = self.source.metadata['label_assistant_workspace_storage']
        info = dict(path=str(Path(reference['path']).resolve()), array_path=reference.get('array_path', ''),
                    bounds=[[s.start, s.stop] for s in region], shape=list(self.source.data.shape),
                    dtype=str(self.source.data.dtype))
        # A durable bounded recovery record exists before touching Zarr chunks.
        with journal.open('wb') as handle:
            np.savez_compressed(handle, info=np.asarray(json.dumps(info)),
                                indices=np.flatnonzero(mask), old=current[mask], new=values)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        self.pending_pixels = pixels.copy()
        current[mask] = values
        self.source.data[region] = current
        if not np.array_equal(np.asarray(self.source.data[region])[mask], values):
            raise ValueError(f'Mask write verification failed; recovery retained at {journal}')
        bounds = (region[0].start, region[1].start, region[0].stop, region[1].stop)
        self.invalidate(bounds)
        self.original = pixels.copy()
        journal.unlink()
        self.pending_pixels = None

    def sync_metadata(self):
        # Preserve full-source identity/library metadata, not crop coordinates.
        self.source.name = self.layer.name
        if self.bounds is not None:
            self.source.visible = self.layer.visible
        self.source.opacity = self.layer.opacity
        self.source.blending = self.layer.blending
        self.source.metadata.update({key: deepcopy(value) for key, value in self.layer.metadata.items()
                                     if key not in (KEY, 'label_assistant_workspace_storage')})

    def invalidate(self, bounds):
        from ._component_library import library_for, invalidate_reviewed_regions
        library = library_for(self.source)
        if library is not None:
            library.mark_dirty(bounds)
        invalidate_reviewed_regions(self.manager.viewer, self.source, bounds)

    def recover(self):
        root = self._recovery_dir()
        if not root.exists():
            return
        reference = self.source.metadata['label_assistant_workspace_storage']
        for path in root.glob('*.npz'):
            with np.load(path, allow_pickle=False) as saved:
                info = json.loads(str(saved['info']))
                if info['path'] != str(Path(reference['path']).resolve()) or info['array_path'] != reference.get('array_path', ''):
                    continue
                if info['shape'] != list(self.source.data.shape):
                    raise ValueError(f'Recovery dimensions changed; retain and inspect {path}')
                if info.get('dtype', str(self.source.data.dtype)) != str(self.source.data.dtype):
                    raise ValueError(f'Recovery dtype changed; retain and inspect {path}')
                bounds = tuple(slice(*pair) for pair in info['bounds'])
                if len(bounds) != 2 or any(s.start < 0 or s.stop > n or not 0 < s.stop-s.start <= 4096
                                            for s,n in zip(bounds,self.source.data.shape)):
                    raise ValueError(f'Invalid recovery region; retain and inspect {path}')
                current = np.asarray(self.source.data[bounds]).copy()
                ids, old, new = saved['indices'], saved['old'], saved['new']
                existing = current.ravel()[ids]
                if np.any((existing != old) & (existing != new)):
                    raise ValueError(f'Conflicting mask edits; recovery retained at {path}')
                current.flat[ids] = new
                self.source.data[bounds] = current
                self.pending_invalidations.append((bounds[0].start,bounds[1].start,bounds[0].stop,bounds[1].stop))
                from ._component_library import library_for
                library = library_for(self.source)
                if library is not None:
                    library.mark_dirty(self.pending_invalidations[-1])
            path.unlink()

    def prepare(self, row, col, size):
        height, width = self.source.data.shape
        y0, x0 = row*size, col*size
        if y0 >= height or x0 >= width:
            return None, np.zeros((1,1), dtype=self.source.data.dtype)
        bounds = (slice(y0, min(height, y0+size)), slice(x0, min(width, x0+size)))
        return bounds, np.asarray(self.source.data[bounds]).copy()

    def show(self, prepared=None):
        from .workspace.service import _apply_common_state, _common_record
        size = self.manager.size
        bounds, pixels = prepared if prepared is not None else self.prepare(self.manager.row,self.manager.col,size)
        y0, x0 = self.manager.row*size, self.manager.col*size
        transform = crop_transform_kwargs(self.source, (y0, x0))
        metadata = deepcopy(self.source.metadata)
        # Crop buffers are not stores or independent global component libraries.
        metadata.pop('label_assistant_workspace_storage', None)
        metadata.pop('label_assistant_component_library', None)
        metadata[KEY] = dict(origin=[y0, x0], source_shape=list(self.source.data.shape),
                             scope='active tile only')
        if self.layer is None:
            self.layer = self.manager.viewer.add_labels(pixels, name=self.source.name,
                metadata=metadata, **transform)
            state = _common_record(self.source, 'Labels')
            # Restore non-spatial state; the visible layer must keep crop mapping.
            for key in ('scale', 'translate', 'rotate', 'shear', 'affine'):
                state.pop(key, None)
            state.pop('label_assistant_component_library', None)
            _apply_common_state(self.layer, state)
            self.layer._label_assistant_mask_entry = self
            self.layer.events.data.connect(self.manager.schedule_flush)
            self.layer.events.labels_update.connect(self.manager.schedule_flush)
            self.layer.mouse_drag_callbacks.append(self.manager.track_stroke)
        else:
            self.layer.data = pixels
            for name, value in transform.items():
                setattr(self.layer, name, value)
            self.layer.metadata = metadata
            self.layer._reset_history()
        self.bounds = bounds
        self.original = pixels.copy()
        self.layer.visible = self.source.visible if bounds is not None else False
        self.layer.editable = bounds is not None


class MaskTileManager:
    def __init__(self, viewer, state=None):
        state = state or {}
        self.viewer = viewer
        self.entries = []
        self.size = int(state.get('tile_size', DEFAULT_SIZE))
        if self.size not in (512, 1024, 2048, 4096):
            self.size = DEFAULT_SIZE
        self.row = max(0, int(state.get('row', 0)))
        self.col = max(0, int(state.get('col', 0)))
        self.progress = dict(state.get('progress', {}))
        self.loading = False
        self.stroke_active = False
        self.error = ''
        self.timer = None
        self._temporary_stores = []
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
        if QApplication.instance() is not None:
            self.timer = QTimer()
            self.timer.setSingleShot(True)
            self.timer.setInterval(500)
            self.timer.timeout.connect(self._auto_flush)
        viewer.layers.events.removing.connect(self._removing)
        viewer.layers.events.inserted.connect(self._inserted)
        setattr(viewer, MANAGER, self)

    def add(self, record, array, reference, manifest_path=None):
        entry = MaskEntry(self, record, array, reference)
        from .workspace.service import _apply_common_state
        _apply_common_state(entry.source, record, manifest_path=manifest_path)
        entry.recover()
        self.loading = True
        try:
            entry.show()
            self.entries.append(entry)
        finally:
            self.loading = False
        return entry.layer

    def schedule_flush(self, event=None):
        if not self.loading and self.timer is not None:
            self.timer.start()

    def _inserted(self, event):
        """New aligned mask crops become sparse full-size managed masks."""
        if self.loading or getattr(self.viewer, '_label_assistant_workspace_loading', False):
            return
        import napari
        from ._dynamic_edit import is_internal_layer
        from ._coordinates import matching_mask_transforms
        layer = event.value
        if not isinstance(layer, napari.layers.Labels) or is_internal_layer(layer) or hasattr(layer, '_label_assistant_mask_entry'):
            return
        reference = next((entry for entry in self.entries if entry.bounds is not None
                          and layer.data.shape == entry.layer.data.shape
                          and matching_mask_transforms(layer, entry.layer)), None)
        if reference is None:
            return
        from tempfile import TemporaryDirectory
        from .workspace.service import _common_record
        import zarr
        temporary = TemporaryDirectory(prefix='label-assistant-new-mask-')
        self._temporary_stores.append(temporary)
        path = Path(temporary.name)/'mask.zarr'
        array = zarr.open_array(str(path),mode='w',shape=reference.source.data.shape,
                                chunks=tuple(min(n,1024) for n in reference.source.data.shape),dtype=layer.data.dtype)
        record = _common_record(layer,'Labels')
        base = _common_record(reference.source,'Labels')
        for name in ('scale','translate','rotate','shear','affine'):
            record[name] = base[name]
        entry = MaskEntry(self,record,array,{'path':path,'array_path':''})
        entry.source._label_assistant_mask_temporary = True
        entry.layer, entry.bounds = layer, reference.bounds
        entry.original = np.zeros_like(layer.data)
        layer._label_assistant_mask_entry = entry
        layer.metadata = {**layer.metadata,KEY:deepcopy(reference.layer.metadata[KEY])}
        layer.events.data.connect(self.schedule_flush)
        layer.events.labels_update.connect(self.schedule_flush)
        layer.mouse_drag_callbacks.append(self.track_stroke)
        self.entries.append(entry)
        self.schedule_flush()

    def _auto_flush(self):
        if self.stroke_active:
            self.timer.start()
            return
        try:
            self.flush()
        except Exception as error:
            self.error = str(error)
            self.viewer.status = f'Tile edits not saved: {error}'

    def track_stroke(self, layer, event):
        self.stroke_active = True
        try:
            yield
            while getattr(event, 'type', '') == 'mouse_move':
                yield
        finally:
            self.stroke_active = False
            self.schedule_flush()

    def flush(self):
        for entry in self.entries:
            entry.flush()
        for entry in self.entries:
            entry.sync_metadata()  # Capture invalidations even when coverage precedes truth in layer order.
        self.error = ''

    def finish_loading(self):
        for entry in self.entries:
            for bounds in entry.pending_invalidations:
                entry.invalidate(bounds)
            entry.pending_invalidations.clear()
        for entry in self.entries:
            entry.sync_metadata()

    def _removing(self, event):
        layer = self.viewer.layers[event.index]
        entry = getattr(layer, '_label_assistant_mask_entry', None)
        if entry is not None and entry in self.entries:
            entry.flush()
            self.entries.remove(entry)
            if not self.entries:
                self.dispose()

    def dispose(self):
        if self.timer is not None:
            self.timer.stop()
        for emitter, callback in ((self.viewer.layers.events.removing,self._removing),
                                  (self.viewer.layers.events.inserted,self._inserted)):
            try:
                emitter.disconnect(callback)
            except ValueError:
                pass

    @property
    def shape(self):
        return tuple(max(entry.source.data.shape[i] for entry in self.entries) for i in (0, 1))

    def state(self):
        return dict(mode=getattr(self.viewer, MODE, 'auto'), tile_size=self.size,
                    row=self.row, col=self.col, progress=self.progress,
                    editing_strategy=getattr(self.viewer, '_label_assistant_editing_strategy', 'auto'))

    def go(self, row, col, size=None):
        from .workspace.service import _commit_pending_local_edits, _ensure_no_training, _ensure_no_library_scan
        _ensure_no_training(self.viewer)
        _ensure_no_library_scan(self.viewer)
        if any(entry.bounds is not None and not entry.layer.editable for entry in self.entries):
            raise ValueError('Finish the current mask operation before changing tiles.')
        _commit_pending_local_edits(self.viewer)
        self.flush()
        new_size = self.size if size is None else int(size)
        if new_size not in (512, 1024, 2048, 4096):
            raise ValueError('Choose a tile size of 512, 1024, 2048 or 4096 pixels.')
        h, w = self.shape
        row, col = int(row), int(col)
        if not (0 <= row < (h+new_size-1)//new_size and 0 <= col < (w+new_size-1)//new_size):
            raise ValueError('Tile address is outside the source mask.')
        # Read every destination crop before replacing any editable buffers.
        prepared = [entry.prepare(row,col,new_size) for entry in self.entries]
        self.row, self.col, self.size = row, col, new_size
        self.loading = True
        try:
            for entry, pixels in zip(self.entries,prepared):
                entry.show(pixels)
        finally:
            self.loading = False
        self.progress.setdefault(f'{self.size}:{row}:{col}', 'visited')
        self.focus()

    def focus(self):
        if not self.entries:
            return
        entry = next((entry for entry in self.entries if entry.bounds is not None), None)
        if entry is None:
            return
        center = [(s.start+s.stop-1)/2 for s in entry.bounds]
        camera = getattr(getattr(self.viewer, 'scene', None), 'camera', None)
        if camera is None:
            camera = self.viewer.camera
        camera.center = entry.source.data_to_world(center)
        corners = np.asarray([entry.source.data_to_world((y,x))
                              for y in (entry.bounds[0].start,entry.bounds[0].stop)
                              for x in (entry.bounds[1].start,entry.bounds[1].stop)])
        camera.zoom = 700/max(float(np.ptp(corners,axis=0).max()),1.)

    def restore_full_layers(self):
        from .workspace.service import _commit_pending_local_edits, _apply_common_state, _common_record
        _commit_pending_local_edits(self.viewer)
        self.flush()
        self.loading = True
        try:
            for entry in list(self.entries):
                # Explicitly requested full loading; never used by automatic opening.
                source = entry.source
                self.viewer.layers.remove(entry.layer)
                layer = self.viewer.add_labels(source.data, name=source.name, metadata=deepcopy(source.metadata))
                _apply_common_state(layer, _common_record(source, 'Labels'))
                if getattr(source,'_label_assistant_mask_temporary',False):
                    layer._label_assistant_mask_temporary = True
                    layer._label_assistant_mask_temp_resources = self._temporary_stores
        finally:
            self.loading = False
        setattr(self.viewer, MANAGER, None)


def loading_state(viewer, payload):
    """Resolve preference before any Labels layer is added to napari."""
    state = dict(payload.get('mask_view') or {})
    override = getattr(viewer, '_label_assistant_mask_loading_override', None)
    mode = override or state.get('mode', 'auto')
    if mode not in ('auto', 'grid', 'full'):
        raise ValueError('Unknown mask loading mode. Choose Automatic, Grid tile, or Full-layer editing.')
    setattr(viewer, MODE, mode)
    if mode in ('grid','auto'):
        viewer._label_assistant_editing_strategy = mode
    else:
        strategy = (getattr(viewer,'_label_assistant_editing_strategy','direct')
                    if override else state.get('editing_strategy','direct'))
        viewer._label_assistant_editing_strategy = strategy if strategy in ('tiled','direct') else 'direct'
    state['mode'] = mode
    return state


def should_tile(array, mode):
    if len(array.shape) != 2 or mode == 'full':
        return False
    if mode == 'grid':
        return True
    from ._dynamic_edit import automatic_uses_tiled_editing
    return automatic_uses_tiled_editing(array)


def set_mode(viewer, mode):
    """Existing editing selector also determines how the next project opens."""
    requested = 'grid' if mode == 'grid' else 'auto' if mode == 'auto' else 'full'
    viewer._label_assistant_editing_strategy = mode
    setattr(viewer, '_label_assistant_mask_loading_override', requested)
    setattr(viewer, MODE, requested)
    manager = manager_for(viewer)
    if manager is not None and manager.entries:
        if requested == 'full':
            manager.restore_full_layers()
        return
    if requested != 'grid':
        return
    from .workspace.service import _common_record, _zarr_reference, _commit_pending_local_edits
    _commit_pending_local_edits(viewer)
    import napari
    candidates = [(layer, _zarr_reference(layer)) for layer in list(viewer.layers)
                  if isinstance(layer, napari.layers.Labels) and len(layer.data.shape) == 2]
    if not candidates:
        return  # Preference is remembered for the next workspace open.
    if any(reference is None for _, reference in candidates):
        raise ValueError('Save in-memory masks to a workspace first, then choose Grid tile editing.')
    # Preflight bounded reads before removing any currently visible mask.
    for layer, _ in candidates:
        np.asarray(layer.data[:min(layer.data.shape[0],DEFAULT_SIZE), :min(layer.data.shape[1],DEFAULT_SIZE)])
    manager = MaskTileManager(viewer)
    for layer, reference in candidates:
        record, data = _common_record(layer, 'Labels'), layer.data
        library = layer.metadata.get('label_assistant_component_library')
        if library is not None:
            record['label_assistant_component_library'] = deepcopy(library)
        viewer.layers.remove(layer)
        manager.add(record, data, reference)
    manager.focus()


def tile_controls(viewer, parent=None, strategy=None):
    """Compact grid navigator; no per-tile image files or giant table needed."""
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QComboBox
    panel = QWidget(parent)
    layout = QVBoxLayout(panel); layout.setContentsMargins(0, 0, 0, 0)
    note = QLabel(); note.setWordWrap(True); layout.addWidget(note)
    row, col = QSpinBox(), QSpinBox()
    row.setPrefix('R'); col.setPrefix('C')
    row.setRange(0, 1000000); col.setRange(0, 1000000)
    size = QComboBox()
    for value in (512, 1024, 2048, 4096):
        size.addItem(f'{value} px', value)
    address = QHBoxLayout()
    for control in (row, col, size):
        address.addWidget(control)
    layout.addLayout(address)
    buttons = QHBoxLayout()
    previous, go, following, done = (QPushButton(text) for text in ('Previous', 'Go', 'Save & Next', 'Mark done'))
    for button in (previous, go, following, done):
        buttons.addWidget(button)
    layout.addLayout(buttons)
    stamp = [None]
    def refresh():
        if strategy is not None:
            desired = getattr(viewer, '_label_assistant_editing_strategy', 'auto')
            index = strategy.findData(desired)
            if index >= 0 and strategy.currentIndex() != index:
                strategy.blockSignals(True); strategy.setCurrentIndex(index); strategy.blockSignals(False)
        manager = manager_for(viewer)
        active = manager is not None and bool(manager.entries)
        panel.setVisible(active)
        if not active:
            return
        key = (id(manager), manager.row, manager.col, manager.size)
        h, w = manager.shape
        if stamp[0] != key:
            row.setValue(manager.row); col.setValue(manager.col)
            size.setCurrentIndex(size.findData(manager.size)); stamp[0] = key
        progress = manager.progress.get(f'{manager.size}:{manager.row}:{manager.col}', 'unvisited')
        bounds = next((entry.bounds for entry in manager.entries if entry.bounds is not None), (slice(0,0),slice(0,0)))
        note.setText(manager.error or f'Grid tile R{manager.row:02d}C{manager.col:02d} · {progress}\n'
                     f'y[{bounds[0].start}:{bounds[0].stop}] x[{bounds[1].start}:{bounds[1].stop}]\n'
                     f'Only mask crops are loaded. Full mask: {h:,} × {w:,}. Components here describe this tile; boundary objects may continue next door.')
    def run(action):
        try:
            action(); refresh()
        except Exception as error:
            note.setText(str(error))
            viewer.status = str(error)
    def move(direction):
        manager = manager_for(viewer)
        columns = (manager.shape[1]+manager.size-1)//manager.size
        index = manager.row*columns+manager.col+direction
        manager.go(index//columns, index%columns)
    previous.clicked.connect(lambda: run(lambda: move(-1)))
    following.clicked.connect(lambda: run(lambda: move(1)))
    go.clicked.connect(lambda: run(lambda: manager_for(viewer).go(row.value(), col.value(), size.currentData())))
    def mark_done():
        manager = manager_for(viewer); manager.flush()
        manager.progress[f'{manager.size}:{manager.row}:{manager.col}'] = 'done'
        layer = manager.entries[0].layer
        layer.metadata = {**layer.metadata,KEY:{**layer.metadata[KEY],'completion':'done'}}
    done.clicked.connect(lambda: run(mark_done))
    panel._poll = QTimer(panel); panel._poll.setInterval(500); panel._poll.timeout.connect(refresh); panel._poll.start()
    panel._row, panel._col, panel._size = row, col, size
    panel._go, panel._next, panel._done = go, following, done
    panel._note = note
    refresh()
    return panel
