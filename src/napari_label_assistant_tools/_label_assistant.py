from __future__ import annotations

import napari
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget, QComboBox, QAbstractSpinBox, QCheckBox

from ._widget import (
    _LayerActivityIndicator,
    label_operations_widget,
    component_operations_widget,
    quick_compare_toggle_widget,
)
from ._workspace_widget import WorkspaceManagerWidget
from ._quick_train_widget import QuickTrainWidget


def label_assistant_widget(viewer=None, **kwargs) -> QWidget:
    """Edit, review, and organize segmentation labels."""
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    page = QWidget()
    layout = QVBoxLayout(page)
    activity_header = _LayerActivityIndicator(page)
    activity_header.set_context_tooltip(
        "Edit, review, and organize segmentation labels. "
        "Annotate full-resolution 2D images at 100,000 × 100,000-pixel scale "
        "without cropping. Inspect and curate connected components, then use "
        "reproducible grid addresses to report, correct, and re-review mask "
        "regions."
    )
    layout.addWidget(activity_header)
    tools = QTabWidget()
    workspace = WorkspaceManagerWidget(napari_viewer=viewer)
    tools.addTab(workspace, "Workspace")
    components = component_operations_widget(viewer, activity_indicator=activity_header)
    tools.addTab(components, "Labels")
    training = QuickTrainWidget(viewer)
    training.attach_annotation_widgets(components)
    tools.addTab(training, "Training")
    tools.addTab(quick_compare_toggle_widget(viewer), "Compare")
    tools.addTab(label_operations_widget(viewer), "Combine")
    for panel in (components, training):
        for control in panel.findChildren(QComboBox):
            control.currentIndexChanged.connect(workspace.mark_changed)
        for control in panel.findChildren(QAbstractSpinBox):
            control.valueChanged.connect(workspace.mark_changed)
        for control in panel.findChildren(QCheckBox):
            control.toggled.connect(workspace.mark_changed)
    tools.setTabToolTip(
        0,
        "Create, open, save, and transfer complete workspaces.",
    )
    tools.setTabToolTip(
        1,
        "Edit Labels and review connected components.",
    )
    tools.setTabToolTip(
        3,
        "Compare an image with its Labels layer while preserving edit focus.",
    )
    tools.setTabToolTip(
        4,
        "Relabel, merge, compose, or calculate agreement across Labels layers.",
    )
    tools.setTabToolTip(
        2,
        "Prepare and review crops, train or continue a U-Net, and preview predictions.",
    )
    layout.addWidget(tools)
    page._activity_header = activity_header
    page._tool_tabs = tools
    return page
