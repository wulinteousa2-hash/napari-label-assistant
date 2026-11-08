"""Derived categorical OME-Zarr pyramids; s0 remains the annotation source."""
from copy import deepcopy
from pathlib import Path

import numpy as np

KEY = 'label_assistant_pyramid'
METHOD = 'first-nonzero-2x2'
MAX_LEVELS = 7
BLOCK_SIZE = 512


def level_shapes(shape):
    shapes = [tuple(shape)]
    while len(shapes) < MAX_LEVELS and max(shapes[-1]) > 1:
        shapes.append(tuple((n + 1) // 2 for n in shapes[-1]))
    return shapes


def reduce_labels(block):
    """Keep an existing foreground ID in each 2x2 bin, never average IDs.

    The first nonzero ID in row-major order wins if several IDs share a bin.
    This preserves isolated foreground pixels, but is visualization only.
    """
    h, w = block.shape
    output = np.zeros(((h + 1) // 2, (w + 1) // 2), dtype=block.dtype)
    for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        values = block[y::2, x::2]
        target = output[:values.shape[0], :values.shape[1]]
        np.copyto(target, values, where=(target == 0))
    return output


def _root(reference, mode='r'):
    import zarr
    if str(reference.get('array_path', '')).strip('/') != 's0':
        return None
    return zarr.open_group(str(reference['path']), mode=mode)


def datasets(reference):
    root = _root(reference)
    if root is None or KEY not in root.attrs:
        return []
    state = root.attrs[KEY]
    if not state.get('complete') or state.get('dirty_regions'):
        return []
    expected = level_shapes(root['s0'].shape)
    names = state.get('datasets', [])
    if len(names) < 2 or names != [f's{i}' for i in range(len(expected))]:
        return []
    for name, shape in zip(names, expected):
        if name not in root or tuple(root[name].shape) != shape or root[name].dtype != root['s0'].dtype:
            return []
    return names


def open_levels(reference):
    names = datasets(reference)
    root = _root(reference)
    return [root[name] for name in names] if names else []


def build(reference, *, scale=(1, 1), progress=None, bounds=None, force=False, validate_source=None):
    """Build/refresh s1-s6 in bounded blocks, without writing s0.

    An interrupted build leaves a persistent dirty marker. OME resolution
    metadata is published only after all derived arrays have been written.
    """
    from .workspace.service import _create_zarr_array
    root = _root(reference, 'r+')
    if root is None or len(root['s0'].shape) != 2:
        return []
    source = root['s0']
    if np.dtype(source.dtype).kind not in 'uib':
        raise ValueError('Label pyramids require integer Labels data.')
    shapes = level_shapes(source.shape)
    names = [f's{i}' for i in range(len(shapes))]
    previous = dict(root.attrs.get(KEY, {}))
    if not previous and any(name in root for name in names[1:]):
        raise ValueError('Existing lower-resolution data was not created by Label Assistant; it was left unchanged.')
    ready = (previous.get('complete') and previous.get('method') == METHOD
             and all(name in root and tuple(root[name].shape) == shape
                     and root[name].dtype == source.dtype for name, shape in zip(names, shapes)))
    full = [0, 0, *source.shape]
    dirty = list(previous.get('dirty_regions', []))
    if bounds is not None:
        dirty.append(list(map(int, bounds)))
    if not ready or force:
        dirty = [full]
    elif not dirty:
        return names
    rect = [max(0, min(r[0] for r in dirty)), max(0, min(r[1] for r in dirty)),
            min(source.shape[0], max(r[2] for r in dirty)), min(source.shape[1], max(r[3] for r in dirty))]
    root.attrs[KEY] = dict(method=METHOD, complete=bool(ready), datasets=names, dirty_regions=[rect])
    arrays, existed = [source], []
    for name, shape in zip(names[1:], shapes[1:]):
        existed.append(name in root)
        if name in root:
            array = root[name]
            if tuple(array.shape) != shape or array.dtype != source.dtype:
                raise ValueError(f'{name} has incompatible dimensions or dtype; s0 was left unchanged.')
        else:
            array = _create_zarr_array(root, name, shape=shape,
                chunks=tuple(min(n, BLOCK_SIZE) for n in shape), dtype=source.dtype, fill_value=0)
        arrays.append(array)
    regions = []
    for shape in shapes[1:]:
        rect = [max(0, rect[0] // 2), max(0, rect[1] // 2),
                min(shape[0], (rect[2] + 1) // 2), min(shape[1], (rect[3] + 1) // 2)]
        regions.append(rect)
    total = sum(((r[2]-r[0]+BLOCK_SIZE-1)//BLOCK_SIZE) *
                ((r[3]-r[1]+BLOCK_SIZE-1)//BLOCK_SIZE) for r in regions)
    done = 0
    for number, (fine, coarse, rect, was_present) in enumerate(zip(arrays, arrays[1:], regions, existed), start=1):
        y0, x0, y1, x1 = rect
        for y in range(y0, y1, BLOCK_SIZE):
            for x in range(x0, x1, BLOCK_SIZE):
                stop_y, stop_x = min(y + BLOCK_SIZE, y1), min(x + BLOCK_SIZE, x1)
                block = np.asarray(fine[2*y:min(2*stop_y, fine.shape[0]), 2*x:min(2*stop_x, fine.shape[1])])
                reduced = reduce_labels(block)
                if was_present or np.any(reduced):
                    coarse[y:stop_y, x:stop_x] = reduced
                done += 1
                if progress is not None:
                    progress(done, total, f'Building Labels pyramid s{number}: {Path(reference["path"]).name}…')
    if validate_source is not None:
        validate_source()
    ome = deepcopy(dict(root.attrs.get('ome', {})))
    ome['version'] = '0.5'
    multiscales = ome.get('multiscales') or [dict(name=Path(reference['path']).stem,
                                               axes=[dict(name=n, type='space') for n in ('y', 'x')])]
    base_scale = tuple(scale[-2:]) if len(scale) >= 2 else (1, 1)
    multiscales[0].update(type='first-nonzero', datasets=[
        dict(path=name, coordinateTransformations=[dict(type='scale', scale=[float(s) * 2**i for s in base_scale])])
        for i, name in enumerate(names)])
    ome['multiscales'] = multiscales
    root.attrs['ome'] = ome
    root.attrs[KEY] = dict(method=METHOD, complete=True, datasets=names, dirty_regions=[])
    return names


def mark_dirty(reference, bounds):
    root = _root(reference, 'r+')
    if root is None or KEY not in root.attrs:
        return False
    state = dict(root.attrs[KEY])
    rect = list(map(int, bounds)) if bounds is not None else [0, 0, *root['s0'].shape]
    previous = state.get('dirty_regions', [])
    state['dirty_regions'] = [[min([rect[0]] + [r[0] for r in previous]),
                               min([rect[1]] + [r[1] for r in previous]),
                               max([rect[2]] + [r[2] for r in previous]),
                               max([rect[3]] + [r[3] for r in previous])]]
    root.attrs[KEY] = state
    return True


def refresh_layer(layer, bounds=None):
    from ._mask_tiles import full_source
    from .workspace.service import _zarr_reference
    source = full_source(layer)
    reference = _zarr_reference(source)
    if reference is None:
        return None
    try:
        if not mark_dirty(reference, bounds):
            return None
        if bounds is not None:
            build(reference, scale=tuple(source.scale), bounds=bounds)
        return True
    except Exception as error:
        source._label_assistant_pyramid_error = str(error)
        return False


class OverviewLevel:
    """A lazy display level with the editable crop excluded from its overlay."""
    def __init__(self, array, entry):
        self.array, self.entry = array, entry
        self.shape, self.dtype = array.shape, array.dtype
        self.ndim, self.size = 2, int(np.prod(array.shape))

    def __array__(self, dtype=None, copy=None):
        if self.size * self.dtype.itemsize > 64 * 1024**2:
            raise ValueError('Request a bounded slice of this label pyramid level.')
        return np.asarray(self[:, :], dtype=dtype)

    def __getitem__(self, region):
        if region is Ellipsis:
            region = (slice(None), slice(None))
        result = np.asarray(self.array[region])
        bounds = self.entry.bounds
        if bounds is None or not isinstance(region, tuple) or len(region) != 2:
            return result
        if not all(isinstance(s, slice) and (s.step is None or s.step == 1) for s in region):
            return result
        h, w = self.entry.source.data.shape
        y_scale, x_scale = h / self.shape[0], w / self.shape[1]
        y0, x0 = int(np.ceil(bounds[0].start / y_scale)), int(np.ceil(bounds[1].start / x_scale))
        y1, x1 = int(bounds[0].stop / y_scale), int(bounds[1].stop / x_scale)
        ys, xs = region
        start_y, stop_y, _ = ys.indices(self.shape[0])
        start_x, stop_x, _ = xs.indices(self.shape[1])
        a, b, c, d = max(y0, start_y), max(x0, start_x), min(y1, stop_y), min(x1, stop_x)
        if a < c and b < d:
            result = result.copy()
            result[a-start_y:c-start_y, b-start_x:d-start_x] = 0
        return result
