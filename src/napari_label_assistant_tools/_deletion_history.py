"""Disk-backed, single-operation recovery for component cleanup."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class DeletionHistory:
    def __init__(self, layer, *, full_resolution=False):
        self.layer = layer
        self.full_resolution = full_resolution
        from ._mask_tiles import full_data
        self.data = full_data(layer) if full_resolution else layer.data
        self.directory = TemporaryDirectory(prefix="label-assistant-undo-")
        self.entries = []

    def capture(self, slices, mask):
        """Persist original selected values before changing a bounded tile."""
        indices = np.flatnonzero(mask)
        values = np.asarray(self.data[slices]).ravel()[indices]
        path = Path(self.directory.name) / f"{len(self.entries)}.npz"
        np.savez_compressed(path, indices=indices, values=values)
        self.entries.append((slices, path))

    def restore(self):
        """Restore one tile per yield, preserving any subsequent nonzero edits."""
        from ._mask_tiles import full_data
        current = full_data(self.layer) if self.full_resolution else self.layer.data
        if current is not self.data:
            raise ValueError("The layer data was replaced; this deletion cannot be restored.")
        restored = skipped = 0
        for slices, path in self.entries:
            with np.load(path) as record:
                tile = np.asarray(self.data[slices]).copy()
                indices, values = record["indices"], record["values"]
                available = tile.ravel()[indices] == 0
                tile.flat[indices[available]] = values[available]
                if self.full_resolution:
                    from ._copy_destination import write_region
                    entry = self.layer._label_assistant_mask_entry
                    selected = np.zeros(tile.shape, dtype=bool)
                    selected.flat[indices[available]] = True
                    write_region(entry.manager.viewer, self.layer, slices, selected, tile, (0, 0))
                else:
                    self.data[slices] = tile
                restored += int(available.sum())
                skipped += int((~available).sum())
            yield restored, skipped, slices

    def close(self):
        self.directory.cleanup()
