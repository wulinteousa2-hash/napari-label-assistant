"""Bounded adapter for the existing label operations on disk-backed masks."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


MEMORY_LIMIT = 64 * 1024**2
TILE_SIZE = 1024


def run_label_tool(data, mode, *, background, source, new, merge, start, votes, consensus):
    """Reuse the original algorithms on tiles; never coerce a full lazy array.

    Return the output and its temporary-store owner. The caller retains the owner
    for the lifetime of the output layer, just like other temporary mask stores.
    """
    from ._widget import (
        _reclassify_single_value_labels, _merge_label_layers_same_class,
        _compose_label_layers_distinct_classes, _count_label_votes,
        _consensus_label_map,
    )
    import zarr

    shape = tuple(data[0].shape)
    if len(shape) != 2 or any(tuple(array.shape) != shape for array in data):
        raise ValueError("Selected Labels layers must have the same 2D shape.")

    def regions():
        for y in range(0, shape[0], TILE_SIZE):
            for x in range(0, shape[1], TILE_SIZE):
                yield (slice(y, min(y + TILE_SIZE, shape[0])),
                       slice(x, min(x + TILE_SIZE, shape[1])))

    # Relabel auto-detection must consider the full source, not individual tiles.
    if mode == "Reassign Label Value":
        found = False
        auto_value = None
        if source == background:
            raise ValueError("Source value cannot be the background value.")
        for region in regions():
            values = np.unique(np.asarray(data[0][region]))
            values = values[values != background]
            if source is None:
                if values.size:
                    if values.size > 1 or (auto_value is not None and auto_value != int(values[0])):
                        raise ValueError("Expected exactly one non-background value in the source layer.")
                    auto_value = int(values[0])
            elif np.any(values == source):
                found = True
        if source is None:
            source = auto_value
            found = source is not None
        if not found:
            raise ValueError("Source value was not found in the selected layer.")

    temporary = TemporaryDirectory(prefix="label-assistant-label-tool-")
    try:
        output = zarr.open_array(str(Path(temporary.name) / "mask.zarr"), mode="w",
                                 shape=shape, chunks=tuple(min(n, TILE_SIZE) for n in shape),
                                 dtype=np.int32)
        for region in regions():
            blocks = [np.asarray(array[region]) for array in data]
            if mode == "Reassign Label Value":
                block = blocks[0]
                if np.any(block == source):
                    result = _reclassify_single_value_labels(block, new_value=new,
                        background_value=background, source_value=source)
                else:
                    result = block.astype(np.int32)
            elif mode == "Merge Layers As Same Class":
                result = _merge_label_layers_same_class(blocks, target_value=merge,
                                                        background_value=background)
            elif mode == "Compose Layers As Distinct Classes":
                result = _compose_label_layers_distinct_classes(blocks,
                    class_values=list(range(start, start + len(blocks))), background_value=background)
            elif mode == "Vote Count Map":
                result = _count_label_votes(blocks)
            elif mode == "Consensus Map":
                result = _consensus_label_map(blocks, min_votes=votes,
                    consensus_value=consensus, background_value=background)
            else:
                raise ValueError(f"Unknown label operation: {mode}")
            output[region] = result
        return output, temporary
    except Exception:
        temporary.cleanup()
        raise
