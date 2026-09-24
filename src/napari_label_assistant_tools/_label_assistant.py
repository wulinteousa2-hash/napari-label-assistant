from __future__ import annotations

import napari
from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ._widget import (
    label_operations_widget,
    component_operations_widget,
    quick_compare_toggle_widget,
)


def _intro(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
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
            "Annotate 2D images at 100,000 × 100,000-pixel scale without "
            "cropping, then review and correct the mask through traceable grid "
            "addresses. Return to each reported cell to resolve missing, "
            "over-labeled, or incorrectly segmented regions."
        )
    )
    tools = QTabWidget()
    tools.addTab(component_operations_widget(viewer), "Edit & Review")
    tools.addTab(quick_compare_toggle_widget(viewer), "Visual Compare")
    tools.addTab(label_operations_widget(viewer), "Combine Layers")
    tools.setTabToolTip(
        0,
        "Paint or erase Labels layers and inspect connected components.",
    )
    tools.setTabToolTip(
        1,
        "Compare an image with its Labels layer while preserving edit focus.",
    )
    tools.setTabToolTip(
        2,
        "Relabel, merge, compose, or calculate agreement across Labels layers.",
    )
    layout.addWidget(tools)
    page._tool_tabs = tools
    return page
