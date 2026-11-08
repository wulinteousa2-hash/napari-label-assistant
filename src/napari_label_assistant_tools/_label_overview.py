"""Bounded, read-only label previews following Spatial Assistant's policy.

Ported from Spatial Assistant's rendering/label_overview.py.
Only Label Assistant/SAM3 overview helpers are adapted. Editable masks and
their full-resolution source stores are never resampled or replaced.
"""

import weakref
from dataclasses import dataclass

import numpy as np
from napari.layers import Labels

MAX_OVERVIEW_AXIS = 4096
MAX_OVERVIEW_PIXELS = 16 * 1024**2
MAX_OVERVIEW_BYTES = 64 * 1024**2
POLICY_KEY = "label_assistant_bounded_overview"
ROLE_KEYS = ("label_assistant_internal_role", "sam3_internal_role")


def texture_limit():
    """Use the renderer's texture limit, with Spatial's conservative fallback."""
    from qtpy.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is not None and app.platformName() not in ("offscreen", "minimal"):
        try:
            from napari._vispy.utils.gl import get_max_texture_sizes

            maximum = get_max_texture_sizes()[0]
            if maximum:
                return int(maximum)
        except (ImportError, RuntimeError):
            pass
    return 2048


def overview_metadata(metadata):
    return any(metadata.get(key) == "label_pyramid_overview" for key in ROLE_KEYS)


@dataclass(frozen=True)
class OverviewPlan:
    levels: tuple
    factor: tuple[float, float]
    first_level: int
    source_shape: tuple[int, int]


def plan_overview(levels, texture_limit=2048):
    """Inspect shapes only; return a bounded suffix without reading any pixels."""
    levels = tuple(levels)
    if not levels or any(len(level.shape) != 2 for level in levels):
        raise ValueError("A label overview requires a nonempty 2D pyramid")
    limit = min(MAX_OVERVIEW_AXIS, int(texture_limit or 2048))
    for index, level in enumerate(levels):
        shape = tuple(int(n) for n in level.shape)
        if (
            max(shape) <= limit
            and shape[0] * shape[1] <= MAX_OVERVIEW_PIXELS
            and shape[0] * shape[1] * np.dtype(level.dtype).itemsize
            <= MAX_OVERVIEW_BYTES
        ):
            suffix = levels[index:]
            if any(
                max(item.shape) > limit
                or int(np.prod(item.shape)) > MAX_OVERVIEW_PIXELS
                or int(np.prod(item.shape)) * np.dtype(item.dtype).itemsize
                > MAX_OVERVIEW_BYTES
                for item in suffix
            ):
                raise ValueError("The label overview pyramid must decrease in size")
            source_shape = tuple(int(n) for n in levels[0].shape)
            factor = tuple(a / b for a, b in zip(source_shape, shape))
            return OverviewPlan(suffix, factor, index, source_shape)
    raise ValueError(
        "No texture-safe label overview level is available; the full-resolution edit crop remains usable"
    )


def bounded_overview_arguments(data, kwargs, texture_limit=2048):
    """Prepare an overview before Labels construction can read a large level."""
    if not kwargs.get("multiscale") or not overview_metadata(
        kwargs.get("metadata") or {}
    ):
        return data, kwargs
    plan = plan_overview(data, texture_limit)
    if plan.first_level == 0 and len(plan.levels) > 1:
        return data, kwargs
    kwargs = dict(kwargs)
    kwargs["scale"] = np.asarray(kwargs.get("scale", (1, 1)), dtype=float) * plan.factor
    kwargs["metadata"] = {
        **kwargs.get("metadata", {}),
        POLICY_KEY: {
            "source_shape": list(plan.source_shape),
            "factor": list(plan.factor),
            "first_level": plan.first_level,
            "texture_limit": int(texture_limit or 2048),
        },
    }
    if len(plan.levels) == 1:
        # napari's constructor unwraps a one-level multiscale list incorrectly
        # for these lazy arrays. A single bounded display level is ordinary 2D.
        kwargs["multiscale"] = False
        return plan.levels[0], kwargs
    return list(plan.levels), kwargs


class BoundedLabelsViewer:
    """Delegate a backend's public viewer calls, bounding overview construction.

    The Label Assistant mask manager keeps this delegate, so overview rebuilds
    after paint/save/navigation use the same policy without global patches.
    """

    def __init__(self, viewer, texture_limit=2048):
        object.__setattr__(self, "_viewer", viewer)
        object.__setattr__(self, "_texture_limit", texture_limit)

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def __setattr__(self, name, value):
        setattr(self._viewer, name, value)

    def add_labels(self, data, **kwargs):
        data, kwargs = bounded_overview_arguments(data, kwargs, self._texture_limit)
        return self._viewer.add_labels(data, **kwargs)


class LabelOverviewGuard:
    """Repair preexisting/externally created overview helpers, with cleanup."""

    def __init__(self, viewer, texture_limit=2048):
        self.viewer = viewer
        self.texture_limit = texture_limit
        self._layers = weakref.WeakSet()
        self._repairing = False
        self._closed = False
        viewer.layers.events.inserted.connect(self._inserted, position="first")
        viewer.layers.events.removed.connect(self._removed)
        for layer in viewer.layers:
            self._attach(layer)

    def _inserted(self, event):
        self._attach(event.value)

    def _removed(self, event):
        layer = event.value
        if layer in self._layers:
            layer.events.data.disconnect(self._data_changed)
            self._layers.discard(layer)

    def _attach(self, layer):
        if (
            not isinstance(layer, Labels)
            or not layer.multiscale
            or not overview_metadata(layer.metadata)
        ):
            return
        if layer not in self._layers:
            layer.events.data.connect(self._data_changed, position="first")
            self._layers.add(layer)
        try:
            self._repair(layer)
        except ValueError as error:
            # Keep unsupported legacy previews hidden; editable crops remain
            # usable and managed overview construction reports the error.
            layer.visible = False
            self.viewer.status = str(error)

    def _data_changed(self, event):
        self._attach(event.source)

    def _repair(self, layer):
        if self._repairing:
            return
        plan = plan_overview(layer.data, self.texture_limit)
        if plan.first_level == 0:
            return
        self._repairing = True
        visible = layer.visible
        try:
            previous = layer.metadata.get(POLICY_KEY, {}).get("factor", (1, 1))
            scale = np.asarray(layer.scale) / previous * np.asarray(plan.factor)
            layer.visible = False
            # A manual resolution lock must not force the old full array.
            layer.locked_data_level = None
            if len(plan.levels) == 1:
                # Existing layers cannot switch from multiscale to single-scale.
                # Preserve their type without Labels unwrapping the one-item list.
                from napari.layers._multiscale_data import MultiScaleData

                layer.data = MultiScaleData(plan.levels)
            else:
                layer.data = list(plan.levels)
            layer.scale = scale
            layer.editable = False
            layer.metadata = {
                **layer.metadata,
                POLICY_KEY: {
                    "source_shape": list(plan.source_shape),
                    "factor": list(plan.factor),
                    "first_level": plan.first_level,
                    "texture_limit": int(self.texture_limit or 2048),
                },
            }
        finally:
            layer.visible = visible
            self._repairing = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.viewer.layers.events.inserted.disconnect(self._inserted)
        self.viewer.layers.events.removed.disconnect(self._removed)
        for layer in tuple(self._layers):
            layer.events.data.disconnect(self._data_changed)
        self._layers.clear()
