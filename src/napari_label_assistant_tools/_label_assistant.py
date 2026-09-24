from __future__ import annotations

import napari
from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ._widget import (
    label_operations_widget,
    component_operations_widget,
    quick_compare_toggle_widget,
)
from ._workspace_widget import WorkspaceManagerWidget


def _intro(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setToolTip(
        "Annotate full-resolution 2D images at 100,000 × 100,000-pixel scale "
        "without cropping. Use reproducible grid addresses to report, find, "
        "correct, and re-review missing or incorrect mask regions."
    )
    label.setStyleSheet(
        "QLabel { background-color: #e7f5ff; border: 1px solid #74c0fc; "
        "color: #12344d; border-radius: 4px; padding: 7px; }"
    )
    return label


def label_assistant_widget(viewer=None, **kwargs) -> QWidget:
    """Create the standalone label-assistant tool."""
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(
        _intro(
            "Edit 100,000 × 100,000-pixel Labels layers, then review them "
            "with traceable grid addresses."
        )
    )
    tools = QTabWidget()
    tools.addTab(
        WorkspaceManagerWidget(napari_viewer=viewer),
        "Project",
    )
    tools.addTab(component_operations_widget(viewer), "Labels")
    tools.addTab(quick_compare_toggle_widget(viewer), "Visual Compare")
    tools.addTab(label_operations_widget(viewer), "Mask Tools")
    tools.setTabToolTip(
        0,
        "Create, open, save, and transfer Label Assistant projects.",
    )
    tools.setTabToolTip(
        1,
        "Annotate Labels layers, then inspect components through grid-based QC.",
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
    page._tool_tabs = tools
    return page
