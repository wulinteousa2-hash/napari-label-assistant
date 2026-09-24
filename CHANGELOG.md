# Changelog

All notable changes to napari-label-assistant are documented here. The project
uses [Semantic Versioning](https://semver.org/).

## [1.0.2] - 2026-09-24

### Changed

- A recent project path is now used only as a file-dialog location; unrelated
  viewer contents no longer inherit it as their active Save destination.
- Removing every layer associated with an open project now changes the viewer
  to an unsaved project, so the next Save requests a new filename.
- Every manifest overwrite now preserves the immediately previous manifest as
  a recoverable `.backup.json` file.

### Fixed

- Prevented adaptive grid-address redraws from briefly pairing new point
  indices with stale text values during pan and zoom.
- Corrected a malformed editing-status stylesheet that produced harmless Qt
  stylesheet-parser warnings in the backend.

## [1.0.1] - 2026-09-24

### Major changes

- Established a project-centered workflow: open or save the annotation
  project, edit the full-resolution mask, review it through reproducible grid
  addresses, and perform supporting mask operations in one assistant.
- Added responsive local-area editing for very large 2D Labels layers, with
  stroke-aware changed-region writes back to the full-resolution source.
- Added durable standalone project persistence with writable OME-Zarr masks,
  SAM3 project migration, portable snapshots, and visible loading progress.
- Reorganized the interface around **Project**, **Labels**, **Visual Compare**,
  and **Mask Tools**, including dedicated **Annotate** and **Grid & Components**
  views.

### Added

- Added a standalone Project tab with New, Open, Save, Save As, recent
  projects, and Portable Snapshot actions.
- Added the `.label-assistant.json` project format. In-memory Labels layers
  are persisted once as writable, chunked OME-Zarr arrays and reused by later
  saves.
- Added import support for version-1 `.sam3.json` projects so existing
  projects can migrate without the SAM3 Assistant plugin.
- Added project persistence for per-layer grid dimensions and grid
  display/assignment preferences.
- Added automatic, debounced layer-selector updates when layers are opened,
  added, removed, reordered, or renamed.
- Added an always-visible **Refresh layers** fallback beside the shared Labels
  layer selector.
- Added an automatically closing project-loading dialog with the current
  layer name and overall layer progress.

### Changed

- Reorganized the main interface into **Project**, **Labels**, **Visual
  Compare**, and **Mask Tools**.
- Split Labels work into **Annotate** and **Grid & Components** views so the
  component table can use the available vertical space.
- Renamed **Combine Layers** to **Mask Tools** to reflect its broader relabel,
  merge, composition, voting, and consensus operations.
- Renamed the local editing action from **Save now** to **Apply edits**, and
  **Save edits automatically** to **Apply edits automatically**, distinguishing
  source-layer synchronization from project persistence.
- Shortened the persistent header while retaining the complete product
  explanation in its tooltip and README.

### Fixed

- Ensured project Save applies pending local-area edits to the source Labels
  layer before writing the project manifest.
- Prevented editable-area, editable-boundary, and grid-overlay helper layers
  from being stored or duplicated after project reload.
- Refreshed only the changed source region after a local edit, improving visual
  feedback without refreshing the entire large mask.
- Made automatic local writes stroke-aware and limited commits to changed
  bounding regions to reduce painting latency.
- Corrected canvas click-to-select and append-selection behavior.
- Guarded callbacks against deleted Qt controls and synchronized point text and
  feature updates to avoid reload- and redraw-time exceptions.
- Prevented local-edit helper cleanup from removing address-grid layers during
  pan, zoom, or automatic layer-list synchronization.

## [1.0.0] - 2026-09-24

- Initial standalone release for large-image Labels annotation, connected
  component analysis, grid-addressed review, visual comparison, and mask
  operations.
