from __future__ import annotations

import napari
from qtpy.QtWidgets import QWidget


def label_assistant_widget(viewer=None, **kwargs) -> QWidget:
    """Edit, review, and organize segmentation labels."""
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    from ._curation_shell import LabelAssistantWidget
    page = getattr(viewer, '_label_assistant_curation_widget', None)
    if page is None:
        page = LabelAssistantWidget(viewer)
        viewer._label_assistant_curation_widget = page
    return page
