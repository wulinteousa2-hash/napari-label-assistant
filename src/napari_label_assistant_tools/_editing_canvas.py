"""Canvas mapping for automatic local editing.

Private canvas integration is isolated here.
World coordinates come from actual canvas pixels, never camera zoom alone.
"""


class CanvasAdapter:
    def __init__(self, viewer):
        self.viewer = viewer
        self.canvas = viewer.window._qt_viewer.canvas
        for attribute in ("_map_canvas2world", "view", "native"):
            if not hasattr(self.canvas, attribute):
                raise RuntimeError(f"Unsupported napari canvas: missing {attribute}")
        self._canvas_to_world = self.canvas._map_canvas2world
        self.native = self.canvas.native
        self.camera = viewer.scene.camera if hasattr(viewer, "scene") else viewer.camera
        self.grid = viewer.canvas.grid if hasattr(viewer, "canvas") else viewer.grid

    def world_at(self, xy):
        return self._canvas_to_world(tuple(xy), self.canvas.view)

    def canvas_points(self, world_yx):
        """Project world Y/X positions into Qt logical canvas X/Y pixels."""
        import numpy as np

        world_xy = np.asarray(world_yx)[:, ::-1]
        transform = self.canvas.view.transform * self.canvas.view.scene.transform
        return np.asarray(transform.map(world_xy))[:, :2]

    def supported_view(self):
        return (
            self.viewer.dims.ndisplay == 2
            and tuple(self.viewer.dims.displayed) == (0, 1)
            and not self.grid.enabled
            and self.viewer.dims.ndim == 2
        )
