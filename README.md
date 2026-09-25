# napari-label-assistant

**Performance-aware 2D mask annotation, connected-component curation, and
traceable spatial review—from ordinary images to datasets approaching
100,000 × 100,000 pixels.**

napari-label-assistant connects the complete mask-development workflow:
annotate at full resolution, inspect connected components, collect accepted
regions into a curated Labels layer, review the result by reproducible grid
address, correct reported locations, and resume the project later.

The plugin is model-agnostic. Candidate masks can come from manual annotation,
SAM3, or another segmentation workflow. The final curated layer can then be
used for training, measurement, quantification, or export.

## Version 1.0.2 highlights

- Project Save no longer reuses a recent project's filename for unrelated
  viewer contents, and removing all project layers returns the viewer to an
  explicit unsaved state.
- Before replacing a project manifest, Save retains its previous version as a
  recoverable `.backup.json` file.
- Grid-address labels now update atomically while panning and zooming, avoiding
  transient point/text indexing errors in napari's renderer.
- Editing-status styles no longer produce harmless Qt parser warnings in the
  backend.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes.

## Workflow at a glance

1. **Annotate:** create, extend, erase, or reshape labels at full resolution.
2. **Inspect:** analyze each nonzero label value as one or more connected
   components and sort the results by size, topology, position, or grid
   address.
3. **Curate:** remove artifacts and copy accepted components from candidate
   masks into a curated destination Labels layer.
4. **Review:** register findings with grid addresses such as **R12C08**, so an
   independent reviewer and annotator can refer to the same location.
5. **Correct and resume:** return to a reported cell, update the source mask,
   save the project, and continue later with the same layer and viewer state.

## Connected-component inquiry and curation

The component table turns a segmentation mask into an inspectable set of
regions. It is designed to remain usable when the mask contains many
components and the full image is too large for naive in-memory analysis.

- Detect connected components independently within each nonzero label value.
- Sort by label value, pixel count, Euler number, centroid, bounds, or grid
  address.
- Double-click a table row to locate its component on the canvas.
- Click a labeled region on the canvas to select its table row; optionally add
  successive clicks to a multi-component selection.
- Delete noise and other unwanted regions individually or in batches.
- Copy accepted components individually or in batches to a compatible curated
  destination Labels layer.

Euler number provides a compact 2D topology check: `1` indicates a solid
component with no holes, `0` indicates one hole, and `-1` indicates two holes.
Combined with area sorting, this helps distinguish small artifacts, solid
blobs, ring-like structures, and components that merit closer inspection.

Component IDs identify results within the current analysis. Editing the mask
and running the analysis again may change those IDs. Grid addresses provide
the reproducible spatial reference used across review sessions.

### Candidate and curated Labels layers

A practical workflow uses two roles rather than requiring every segmentation
result to be edited in place:

- **Candidate layers** contain model predictions, partial masks, or regions
  awaiting review.
- **Curated destination layers** collect accepted annotations for downstream
  training, measurement, quantification, or export.

Candidate regions produced by SAM3 are one example, but the same workflow
works with any compatible napari Labels layer. Users can inspect candidates
visually, select them from the table or canvas, and copy only accepted regions
into the curated layer.

## 100,000 × 100,000-pixel workflows

A 100,000 × 100,000 label mask contains 10 billion pixels. Local-area editing
avoids creating another full-size editable copy: only a camera-centered region
is loaded for interaction, and only changed pixels are written back.

At this scale, the source should use an efficiently sliceable, preferably
chunked or on-disk array such as Zarr. Actual responsiveness depends on array
storage, chunk geometry, data type, disk performance, available memory, and
hardware. The numeric scale describes the intended workflow, not an
unconditional hardware-independent limit.

## Interface

The dock widget is organized into four workflow areas:

### Project

Open **Plugins → napari-label-assistant → Label Assistant**, then use the first
**Project** tab to create, open, and save a complete project without SAM3
Assistant.

- A normal save writes a small `.label-assistant.json` manifest.
- New in-memory Labels layers are persisted once as writable, chunked
  OME-Zarr data beside the manifest. Later saves reuse the same store instead
  of copying the full mask again.
- Source images already loaded from files remain external references, keeping
  routine saves fast. **Portable Snapshot** explicitly copies referenced local
  images and masks into one folder for transfer.
- Pending local-area edits are applied to the source Labels layer before the
  project is saved.
- The JSON file is a lightweight project manifest rather than a container for
  the full image. Large mask pixels remain in writable OME-Zarr storage.
- A recent project is not treated as the active Save destination for unrelated
  viewer contents. Before a manifest is overwritten, its previous version is
  retained as a `.backup.json` file.
- Grid cell dimensions and grid display/assignment choices are restored with
  each Labels layer.
- Opening a project displays layer-by-layer progress in a temporary dialog;
  the dialog closes automatically when loading succeeds or fails.
- Temporary editable-area, editable-boundary, and grid-overlay layers are not
  stored; the plugin recreates them when needed.

Existing `.sam3.json` projects created by SAM3 Assistant can be opened for
migration. Use **Save As** to continue with the standalone Label Assistant
format.

### Labels

Choose one Labels layer, then switch between two task-focused tabs:

- **Annotate** contains the memory-controlled Paint and Erase workflow.
- **Grid & Components** contains grid registration, component actions, canvas
  selection, and an expandable results table for large result sets.

Layer selectors update automatically when layers are opened, added, removed,
or renamed. **Refresh layers** remains beside the shared Labels-layer selector
as an always-visible fallback; it is no longer buried among component actions.

**Memory-controlled editing** appears first because it controls how napari
accesses the selected layer:

- **Automatic (recommended)** uses standard full-layer editing for ordinary
  arrays and local-area editing when the layer is large relative to available
  memory.
- **Local-area editing** always loads a bounded region around the current
  camera position.
- **Full-layer editing** uses napari's standard direct editing behavior.

**Connected-component analysis** provides measurement and region-management
tools. Click **Find components**, then:

- click a column heading to sort the table;
- double-click a result row to center that component in the viewer;
- enable **Select components from canvas** to select a table row by clicking
  the labeled region;
- enable **Add each click to selection** to build a multi-component selection;
- use **Delete selected**, **Copy selected**, or **Copy again** as needed.

The results table reports component ID, source label, pixel count, Euler
number, centroid, and bounds. Optional grid columns report the grid row,
column, and cell containing each component centroid.

### Grid registration and QC traceability

The registration grid gives reviewers a reproducible address for every part of
a full 2D mask. With the same cell height, cell width, grid origin, and source
coordinate system, a reference such as **R12C08** identifies the same location
across masking, review, correction, and follow-up QC sessions—even between
collaborators.

Grid-cell size is user-defined, with a default of 100 × 100 pixels. The grid
serves both as an annotation map and as a shared review vocabulary:
screenshots, review notes, and correction requests can carry the same address
without creating or exchanging cropped coordinate systems.

- Enable **Assign grid addresses to results** to register each component by the
  grid row, column, and cell containing its centroid.
- Enable **Show address grid on canvas** to see cell boundaries and identifiers
  around the current viewport.
- Use a grid address together with a component ID—for example, “R12C08,
  component 431”—to identify a labeled region that needs correction.
- Use the grid address alone to register an unlabeled location where a
  component is missing and therefore cannot appear in the results table.

Grid addresses make reviewer findings and corrections traceable without
creating cropped copies of the source. Reuse the same grid dimensions, origin,
and source coordinate system when addresses must remain stable. The current
1.0 grid is a spatial registration and reference system; it does not store
reviewer comments, assignments, completion states, or a change-history audit
log.

### Visual Compare

Compare an image with a Labels layer while keeping the Labels layer active for
editing. The view controls can show either layer, show both, dim the labels, or
temporarily pulse, blink, and peek between them.

### Mask Tools

Perform mask and label operations across one or more compatible layers:

- reassign one label value;
- merge foreground regions into one class;
- compose layers as distinct classes;
- create a per-pixel vote-count map;
- create a consensus map from a minimum vote threshold.

## Editing at 100,000 × 100,000-pixel scale

Local-area editing keeps only a camera-centered working region editable in
memory. Pan or zoom to another location and the working region follows
automatically. The green boundary shows exactly where Paint and Erase are
currently available.

The original Labels layer remains the source of truth. Changes made in the
temporary editable-area layer are written back only for pixels that changed,
and the corresponding part of the source preview is refreshed locally. The
editable-area and boundary layers are transient helpers rather than additional
label results.

### Saving

With **Apply edits automatically** enabled, synchronization is stroke-aware:

1. Saving never starts while the paint or erase mouse button is held.
2. Releasing the mouse starts a 400 ms inactivity delay.
3. Only the accumulated changed region is compared and written to the source.

Disable automatic apply to keep changes in the local area until **Apply edits**
is clicked. A local area with unsaved manual changes remains in place during
pan or zoom, preventing accidental loss. **Undo** and **Redo** synchronize
immediately in automatic mode and remain pending until **Apply edits** in manual
mode.

Tile size controls the memory/performance tradeoff. A 2048 × 2048 area uses
less memory; a 4096 × 4096 area requires fewer reloads while navigating.

## Component analysis at full-image scale

Full-resolution masks are analyzed in tiles and displayed as paged results so the table
does not need to create a row for every component at once. Analysis can be
cancelled from the progress control. Component IDs and optional grid
coordinates remain expressed in the full source-layer coordinate system.

## Requirements and scope

- Python 3.10 or newer
- napari 0.4.19 or newer
- 2D Labels layers for local-area editing and component analysis
- matching 2D shapes when copying or combining layers

Practical dataset size depends on array storage, data type, chunking, available
memory, and hardware. Automatic mode chooses a conservative strategy, while
Local-area editing can be selected explicitly for datasets such as
100,000 × 100,000-pixel masks.

## Installation

Install the released package:

```bash
python -m pip install napari-label-assistant
```

When upgrading from the pre-1.0 development package, remove its old plugin
registration before installing 1.0:

```bash
python -m pip uninstall napari-label-assistant-tools
python -m pip install napari-label-assistant
```

Start napari and open:

**Plugins → napari-label-assistant → Label Assistant**

## Development

Clone the repository and install it in editable mode with test dependencies:

```bash
python -m pip install -e ".[all,test]"
pytest
```

## License

Distributed under the terms of the MIT license.
