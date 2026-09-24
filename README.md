# napari-label-assistant

**Annotate 2D images at 100,000 × 100,000-pixel scale without cropping,
then review and correct the mask through traceable grid addresses.**

napari-label-assistant closes the loop between annotation and quality control.
The annotator works on the full-resolution Labels layer through a
memory-controlled local area designed to remain responsive. The reviewer works
in the same source coordinate system and uses reproducible addresses such as
**R12C08** to report missing, under-labeled, over-labeled, or incorrectly
segmented regions. The annotator can return to that exact cell, correct the
mask, and submit it for re-review.

## Version 1.0.1 highlights

- A standalone **Project** tab saves and restores annotation projects,
  writable masks, grid registration, and viewer state without SAM3 Assistant.
- Layer selectors update automatically when napari layers are opened, added,
  removed, reordered, or renamed.
- The interface now follows the working sequence **Project → Labels → Visual
  Compare → Mask Tools**, with separate **Annotate** and **Grid & Components**
  views inside Labels.
- Local tile synchronization is called **Apply edits**, clearly separating it
  from durable project **Save**.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes.

## Two-part workflow

### 1. Full-resolution annotation

- Create and extend labels with napari Paint, Fill, and Polygon tools.
- Remove or reshape labels with Erase.
- Move continuously across the image while a bounded camera-centered region is
  loaded for editing.
- Apply only changed pixels back to the original Labels layer automatically or
  with **Apply edits**.
- Preserve the original image dimensions and coordinates without producing
  cropped working files that later need to be reassembled.

### 2. Grid-addressed review and correction

- Register the full mask with reproducible grid row, column, and cell addresses.
- Display those addresses around the current viewport while retaining the same
  registration across pan and zoom.
- Assign each connected component to the grid address containing its centroid.
- Let reviewers identify a labeled component with a grid address and component
  ID, or identify a missing label with the grid address alone.
- Return directly to the reported cell for correction and follow-up review.
- Keep annotation, reviewer feedback, correction, and QC discussion traceable
  to the same full-resolution location.

## Component and label analysis

- Detect connected components independently within each nonzero label value.
- Sort components by label value, pixel count, Euler number, centroid, bounds,
  or grid address.
- Use Euler number to review component topology: `1` indicates no holes, `0`
  indicates one hole, and `-1` indicates two holes.
- Locate components on the canvas, build multi-component selections, delete
  selected regions, or copy them to a compatible Labels layer.
- Compare image and Labels layers, reassign values, combine layers, and create
  vote-count or consensus results.

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
