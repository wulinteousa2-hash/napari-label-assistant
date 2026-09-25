from __future__ import annotations

import napari
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ._widget import (
    _LayerActivityIndicator,
    label_operations_widget,
    component_operations_widget,
    quick_compare_toggle_widget,
)
from ._workspace_widget import WorkspaceManagerWidget


def label_assistant_widget(viewer=None, **kwargs) -> QWidget:
    """Create the standalone label-assistant tool."""
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    page = QWidget()
    layout = QVBoxLayout(page)
    activity_header = _LayerActivityIndicator(page)
    activity_header.set_context_tooltip(
        "Annotate full-resolution 2D images at 100,000 × 100,000-pixel scale "
        "without cropping. Inspect and curate connected components, then use "
        "reproducible grid addresses to report, correct, and re-review mask "
        "regions."
    )
    layout.addWidget(activity_header)
    tools = QTabWidget()
    tools.addTab(
        WorkspaceManagerWidget(napari_viewer=viewer),
        "Workspace",
    )
    tools.addTab(
        component_operations_widget(
            viewer, activity_indicator=activity_header
        ),
        "Labels",
    )
    tools.addTab(quick_compare_toggle_widget(viewer), "Compare")
    tools.addTab(label_operations_widget(viewer), "Combine")
    tools.setTabToolTip(
        0,
        "Create, open, save, and transfer complete Label Assistant workspaces.",
    )
    tools.setTabToolTip(
        1,
        "Edit Labels locally, then review the full layer through grid-addressed QC.",
    )
    tools.setTabToolTip(
        2,
        "Compare an image with its Labels layer while preserving edit focus.",
    )
    tools.setTabToolTip(
        3,
        "Relabel, merge, compose, or calculate agreement across Labels layers.",
    )
    layout.addWidget(tools)
    page._activity_header = activity_header
    page._tool_tabs = tools
    return page
