"""Disk-backed, single-operation recovery for component cleanup."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class DeletionHistory:
    def __init__(self, layer):
        self.layer = layer
        self.data = layer.data
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
        if self.layer.data is not self.data:
            raise ValueError("The layer data was replaced; this deletion cannot be restored.")
        restored = skipped = 0
        for slices, path in self.entries:
            with np.load(path) as record:
                tile = np.asarray(self.data[slices]).copy()
                indices, values = record["indices"], record["values"]
                available = tile.ravel()[indices] == 0
                tile.flat[indices[available]] = values[available]
                self.data[slices] = tile
                restored += int(available.sum())
                skipped += int((~available).sum())
            yield restored, skipped, slices

    def close(self):
        self.directory.cleanup()
