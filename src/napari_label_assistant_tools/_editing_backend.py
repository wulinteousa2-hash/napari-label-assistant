"""Camera-follow bridge to the existing journaled mask editing engine."""

import numpy as np

from . import _coordinates as coordinates, _mask_tiles as masks


class LocalEditingBackend:
    coordinates = coordinates

    def manager(self, viewer):
        return masks.manager_for(viewer)

    def flush(self, viewer):
        manager = self.manager(viewer)
        if manager is not None:
            manager.flush()

    def edit_at(self, viewer, world, entry_index=0, size=1024):
        manager = self.manager(viewer)
        if manager is None or not manager.entries:
            raise ValueError("This project has no editable Zarr mask")
        entry = manager.entries[entry_index]
        affine = self.coordinates.mask_world_affine(entry.source)
        y, x = np.linalg.solve(affine[:2, :2], np.asarray(world)[-2:] - affine[:2, 2])
        if not (
            0 <= y < entry.source.data.shape[0] and 0 <= x < entry.source.data.shape[1]
        ):
            raise ValueError("The chosen image area is outside this mask")
        manager.go(int(y // size), int(x // size), size=size)
        layer = entry.layer
        viewer.layers.selection.active = layer
        return layer

    def follow_signature(self, manager):
        return (
            id(manager),
            tuple(
                (
                    id(entry),
                    id(entry.source.data),
                    tuple(entry.source.data.shape),
                    self.coordinates.mask_world_affine(entry.source).tobytes(),
                )
                for entry in manager.entries
            ),
        )

    def follow_view(self, manager, world, entry_index):
        anchor = manager.entries[entry_index].source
        for entry in manager.entries:
            if not self.coordinates.matching_mask_transforms(anchor, entry.source):
                raise ValueError(
                    "Automatic following requires aligned mask grids; use manual editing for differently registered masks"
                )
        matrix = self.coordinates.mask_world_affine(anchor)
        points = np.linalg.solve(
            matrix[:2, :2], (np.asarray(world)[:, -2:] - matrix[:2, 2]).T
        ).T
        bounds = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )
        return bounds, tuple(anchor.data.shape), matrix[:2, :2]

    def _check_follow(self, manager):
        from napari_label_assistant_tools.workspace.service import (
            _ensure_no_library_scan,
            _ensure_no_training,
        )

        _ensure_no_training(manager.viewer)
        _ensure_no_library_scan(manager.viewer)
        if manager.stroke_active:
            raise ValueError(
                "Finish the current stroke before moving the editable area"
            )
        for entry in manager.entries:
            if entry.bounds is None:
                continue
            if not entry.layer.editable:
                raise ValueError("A mask operation is still using the editable area")
            expected = self.coordinates.mask_world_affine(entry.source).copy()
            expected[:2, 2] += expected[:2, :2] @ np.array(
                [part.start for part in entry.bounds]
            )
            if not np.allclose(
                expected,
                self.coordinates.mask_world_affine(entry.layer),
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(
                    "The edit crop transform changed; restore its registration before following"
                )

    def follow_reads(self, manager, origin, size):
        self._check_follow(manager)
        result = []
        for entry in manager.entries:
            h, w = entry.source.data.shape
            bounds = (
                None
                if origin[0] >= h or origin[1] >= w
                else (
                    slice(origin[0], min(h, origin[0] + size)),
                    slice(origin[1], min(w, origin[1] + size)),
                )
            )
            result.append((entry.source.data, bounds))
        return tuple(result)

    def follow_dirty(self, manager):
        return any(
            entry.original is not None
            and (
                entry.layer.data.shape != entry.original.shape
                or np.any(np.asarray(entry.layer.data) != entry.original)
            )
            for entry in manager.entries
        )

    def apply_follow(self, manager, origin, size, prepared):
        self._check_follow(manager)
        if len(prepared) != len(manager.entries):
            raise ValueError("Project masks changed while the area was loading")
        # Origins move in 512-pixel increments, allowing overlapping windows.
        # Each entry's true bounds/shape govern writes and journal recovery.
        manager.size = 512
        manager.row, manager.col = origin[0] // 512, origin[1] // 512
        previous = manager.loading
        manager.loading = True
        if manager.timer is not None:
            manager.timer.stop()
        try:
            for entry, buffer in zip(manager.entries, prepared):
                entry.show(buffer)
        finally:
            manager.loading = previous
        manager.progress.setdefault(f"512:{manager.row}:{manager.col}", "visited")

    def suspend_follow(self, manager):
        self._check_follow(manager)
        manager.flush()
        previous = manager.loading
        manager.loading = True
        if manager.timer is not None:
            manager.timer.stop()
        try:
            for entry in manager.entries:
                # bounds=None is the backend's noneditable state. It preserves
                # source visibility and removes the hole from the overview.
                entry.bounds = None
                entry.layer.mode = "pan_zoom"
                entry.layer.visible = False
                empty = np.zeros((1, 1), dtype=entry.source.data.dtype)
                entry.layer.data = empty
                entry.original = empty.copy()
                entry.layer.editable = False
                if entry.overview is not None:
                    entry.overview.refresh()
                    entry.sync_overview_display()
        finally:
            manager.loading = previous
