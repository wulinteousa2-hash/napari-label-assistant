"""Regenerate the system map: stdlib SVG; optional PNG via local PySide6."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1800, 1570
COLORS = dict(green="#98d4b2", blue="#94c8df", gold="#e9c35b", red="#df9683")

# Each card names implementation modules, two responsibilities, and their contracts.
CARDS = [
    ("A / CLIENT & ANNOTATION WORKBENCH", "green",
     "napari.yaml · _label_assistant.py · _widget.py",
     ("Label Curation", "Workspace / Labels / Training", "Shared napari viewer + layer events"),
     ("Separate plugin panels", "Model Training / Model Library", "Shared project and checkpoint state"),
     ["Edit & Confirm keeps painting and review in one panel.",
      "Mask Analysis & Cleanup inspects and curates components.",
      "View & Compare toggles candidate and curated layers.",
      "Label Tools provides repeatable mask operations.",
      "Qt panels coordinate actions on the live viewer."]),
    ("B / LARGE-IMAGE STORAGE & DISPLAY", "blue",
     "_mask_tiles.py · _label_pyramid.py · _label_overview.py",
     ("Full-resolution source", "Disk-backed Zarr masks / s0", "Full-source access for analysis"),
     ("Bounded viewer layers", "GPU-safe categorical overviews", "Full-resolution editable crops"),
     ["Bounded mask loading is the default workspace policy.",
      "Full stores stay outside napari in the bounded path.",
      "Derived label levels retain IDs; they are display data.",
      "Image pyramids use intensity downsampling separately.",
      "Scientific operations resolve back to the full source."]),
    ("C / LOCAL EDITING & COORDINATES", "green",
     "_camera_follow.py · _editing_backend.py · _coordinates.py",
     ("Camera-follow editing", "Plan and read bounded windows", "Paint / erase / undo / redo"),
     ("Registered writes", "Crop pixels ↔ full mask ↔ world", "Journal changed source regions"),
     ["_local_editing.py exposes Edit at View Center controls.",
      "_mask_tiles.py commits edits and supports write recovery.",
      "Affine transforms keep crops aligned with their source.",
      "_dynamic_edit.py also supports direct / automatic editing.",
      "Changed annotations invalidate affected approvals."]),
    ("D / COMPONENT ANALYSIS & CLEANUP", "green",
     "_large_components.py · _large_component_controller.py",
     ("Entire mask / local area", "Read full-resolution bounded chunks", "Join connected regions at seams"),
     ("Inspect and curate", "Paged table / locate / pick / copy", "Noise cleanup + deletion recovery"),
     ["Whole-mask analysis is independent of zoom and edit crop.",
      "Disconnected regions of one label remain components.",
      "Measurements include pixel area, bounds and centroid.",
      "Copy destinations use explicit coordinate matching.",
      "_deletion_history.py supports undo of the last deletion."]),
    ("E / SPATIAL QC & REVIEWED REGIONS", "blue",
     "_grid_map.py · _reviewed_regions.py · _component_library.py",
     ("Reproducible addresses", "Grid origin + cell size → R12C08", "Locate findings in source space"),
     ("Confirm tile & next", "Approve objects AND background", "Persist exact reviewed rectangles"),
     ["Grid addresses and training approvals are separate state.",
      "Confirmed regions define known annotation coverage.",
      "Unreviewed pixels remain unknown during partial training.",
      "Edits require rechecking affected approved regions.",
      "The grid is a spatial reference, not a reviewer audit log."]),
    ("F / COMPONENT LIBRARY & LABEL TOOLS", "green",
     "_component_library.py · _label_tool_tiles.py · _widget.py",
     ("Persistent object library", "SQLite IDs / selections / coverage", "Separate from temporary analysis IDs"),
     ("Full-source mask tools", "Relabel / merge / class composition", "Vote count / consensus masks"),
     ["Library-backed selections survive workspace reopening.",
      "Advanced object selections can supplement supervision.",
      "Label tools process disk-backed sources in bounded tiles.",
      "New outputs keep the full source image dimensions.",
      "Workspace Save makes temporary output assets durable."]),
    ("G / DATASET & SUPERVISION CONTRACT", "gold",
     "_dataset_pairs.py · _coverage_train.py · _multilabel_targets.py",
     ("Images + annotations", "Current pair / multiple pairs", "Complete / partial / unknown"),
     ("Prepare dataset", "Index eligible patches + validity", "Spatial / specimen-group splits"),
     ["Stable layer IDs preserve pair and channel associations.",
      "Binary / exclusive multiclass / overlapping multilabel.",
      "Partial coverage masks exclude unknown supervision.",
      "Preparation is separate from the optional crop preview.",
      "Annotation and source changes invalidate prepared data."]),
    ("H / OPTIONAL LOCAL MODEL TRAINING", "gold",
     "_quick_train_widget.py · _quick_train.py · _model_providers.py",
     ("Model providers", "Custom / standard U-Net", "Optional SMP / registered providers"),
     ("Train and evaluate", "PyTorch CPU / CUDA when available", "Holdout / spatial k-fold / comparisons"),
     ["2D grayscale inputs use per-crop percentile normalization.",
      "Task mode defines logits, masked loss and metrics.",
      "_coverage_train.py streams indexed supervised patches.",
      "Continuation loads weights with a fresh optimizer.",
      "Training runs in workers with progress and cancellation."]),
    ("I / INFERENCE & ANNOTATION FEEDBACK", "gold",
     "_quick_train.py · _quick_train_widget.py · _multilabel.py",
     ("Predict a region", "Current area / entire image / box", "Overlapping tiled prediction"),
     ("Return Labels to viewer", "Binary threshold / class argmax", "Independent multilabel channels"),
     ["Prediction uses the selected compatible checkpoint.",
      "Large-region inference supports progress and Cancel.",
      "Task and class/channel metadata accompany the output.",
      "Review predictions before using them as ground truth.",
      "Correct and confirm to begin another training round."]),
    ("J / WORKSPACE SAVE & RESTORE", "blue",
     "workspace/service.py · _workspace_widget.py · _model_project.py",
     ("JSON manifest · format 2", "Layer / viewer / transform state", "Review + task + model references"),
     ("Workspace data folder", "Images / Zarr masks / SQLite", "Checkpoints / histories / reports"),
     ["Save flushes pending local edits before recording state.",
      "Existing Zarr sources may remain linked externally.",
      "Complete-package Save As copies assets for transfer.",
      "Open restores summaries; weights load when selected.",
      "Save/Open waits for training, prediction and model export."]),
    ("K / MODEL LIBRARY & RUN PROVENANCE", "blue",
     "_model_library_widget.py · _model_library.py · _model_files.py",
     ("Catalog models and runs", "Task / config / dataset / scores", "Notes / tags / favorites / history"),
     ("Manage checkpoint files", "Managed / external / missing", "Copy bundle / locate / select"),
     ["Browsing catalog records does not load network weights.",
      "Save or export persists in-memory trained checkpoints.",
      "Compare runs using recorded evaluation provenance.",
      "Export history CSV and learning / comparison charts.",
      "Checkpoint selection is shared with Model Training."]),
    ("L / IMPLEMENTATION BOUNDARIES", "red",
     "pyproject.toml · tests/ · validation_reports/",
     ("Local annotation core", "napari / Qt / NumPy / SciPy", "scikit-image / Zarr / SQLite"),
     ("Optional model stack", "PyTorch training extra / SMP extra", "External masks enter as Labels"),
     ["No built-in chat, LLM router or SAM inference backend.",
      "Bounded reads do not make every operation asynchronous.",
      "Practical scale depends on dtype, chunks and hardware.",
      "Validation-selected scores are not independent test scores.",
      "Review approval records inspection, not biological truth."]),
]


def build_svg() -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
             '<title id="title">napari-label-assistant / System Design</title>',
             '<desc id="description">Implementation map of version 6.4.0: local annotation, full-resolution storage, curation, spatial review, optional training and inference, and workspace persistence. Twelve panels document the source modules and their boundaries.</desc>',
             '<defs>' + ''.join(f'<marker id="arrow-{color[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{color}"/></marker>' for color in COLORS.values()) + '</defs>',
             f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#171f1c"/>',
             '<g font-family="DejaVu Sans, sans-serif">']

    def text(x, y, value, size=15, color="#dce2da", bold=False):
        weight = ' font-weight="700"' if bold else ''
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}"{weight}>{escape(value)}</text>')

    def box(x, y, w, h, color, dashed=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#1c2521" stroke="{color}" stroke-width="1.4"{dash}/>')

    def arrow(x1, y1, x2, y2, color, dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        parts.append(f'<path d="M{x1} {y1} H{x2} V{y2}" fill="none" stroke="{color}" stroke-width="1.8" marker-end="url(#arrow-{color[1:]})"{dash}/>')

    text(24, 44, "napari-label-assistant / System Design", 36, "#edf1e8", True)
    text(24, 73, "Full-resolution annotation · connected-component curation · spatial QC · optional local training", 20, COLORS['green'])
    text(24, 99, "Implementation map | v6.4.0 | source baseline c5c02e3 | 10 November 2026", 14, "#a9b7ad")
    arrow(1120, 96, 1160, 96, COLORS['green'])
    text(1172, 101, "core workflow", 13, COLORS['green'])
    arrow(1310, 96, 1350, 96, COLORS['gold'], True)
    text(1362, 101, "optional model path", 13, COLORS['gold'])
    text(1585, 101, "red = boundary", 13, COLORS['red'])

    flow = [
        (24, 300, "1 Open workspace", "Images + candidate / curated masks", "Restore layers, sources and state", 'green'),
        (358, 316, "2 Edit and curate", "Bounded painting + component cleanup", "Full-resolution source coordinates", 'green'),
        (708, 316, "3 Review and confirm", "Grid addresses + exact tile approvals", "Objects and checked background", 'green'),
        (1058, 328, "4 Optional model loop", "Prepare → train → predict → review", "Checkpoint + dataset provenance", 'gold'),
        (1420, 356, "5 Save and resume", "Manifest + image / mask / model assets", "Reopen the same spatial workspace", 'blue'),
    ]
    for x, width, title, line1, line2, color in flow:
        box(x, 129, width, 87, COLORS[color])
        text(x + 12, 153, title, 18, COLORS[color], True)
        text(x + 12, 177, line1, 14)
        text(x + 12, 198, line2, 14)
    for left, right in zip(flow, flow[1:]):
        arrow(left[0] + left[1], 172, right[0], 172, COLORS[right[5]], right[5] == 'gold')
    # The core annotation path can save without training.
    parts.append(f'<path d="M866 216 V243 H1598 V216" fill="none" stroke="{COLORS["green"]}" stroke-width="1.6" marker-end="url(#arrow-{COLORS["green"][1:]})"/>')
    text(1040, 265, "Annotation and curation can save directly", 14, COLORS['green'])
    text(24, 287, "Predictions return to editing and review; only approved annotation becomes training supervision.", 16, COLORS['green'])

    for index, (title, color, modules, left, right, lines) in enumerate(CARDS):
        x = 24 + (index % 3) * 592
        y = 316 + (index // 3) * 290
        ink = COLORS[color]
        box(x, y, 568, 272, ink, True)
        text(x + 14, y + 26, title, 18, ink, True)
        text(x + 14, y + 48, modules, 12, "#a9b7ad")
        for offset, mini in ((14, left), (292, right)):
            box(x + offset, y + 64, 262, 82, ink)
            text(x + offset + 11, y + 87, mini[0], 17, ink, True)
            text(x + offset + 11, y + 108, mini[1], 13)
            text(x + offset + 11, y + 128, mini[2], 13)
        for row, line in enumerate(lines):
            text(x + 14, y + 172 + row * 20, line, 14)

    text(24, 1512, "Read with system-design.md for source links, workflow details, boundaries and regeneration instructions.", 16, COLORS['green'])
    text(24, 1540, "Architecture documentation only · no runtime changes · editable SVG + PNG preview", 14, "#a9b7ad")
    parts.extend(['</g>', '</svg>'])
    return '\n'.join(parts) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true', help='Also render PNG using PySide6 QtSvg')
    args = parser.parse_args()
    svg = ROOT / 'system-design.svg'
    svg.write_text(build_svg(), encoding='utf-8')
    print(svg)
    if args.png:
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtGui import QGuiApplication, QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer
        app = QGuiApplication.instance() or QGuiApplication([])
        renderer = QSvgRenderer(str(svg))
        if not renderer.isValid():
            raise RuntimeError('Invalid SVG')
        image = QImage(WIDTH * 2, HEIGHT * 2, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        png = ROOT / 'system-design.png'
        if not image.save(str(png)):
            raise RuntimeError(f'Could not save {png}')
        print(png)
        del app


if __name__ == '__main__':
    main()
