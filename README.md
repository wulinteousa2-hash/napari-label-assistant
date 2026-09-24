# napari-label-assistant

Edit, navigate, and review very large napari Labels layers without cropping.

napari-label-assistant is designed for label masks that range from ordinary
images to extremely large 2D datasets. It lets you paint and erase a responsive
local area while preserving the full-size source layer—no manual cropping,
editing separate files, or stitching results back together.

## Highlights

- Edit very large Labels layers through a bounded-memory local area.
- Continue using napari's familiar Paint, Erase, Fill, and Polygon tools.
- Save each completed stroke automatically or control saving manually.
- Analyze connected components independently within every nonzero label value.
- Sort results, locate components on the canvas, and select one or many regions.
- Delete selected components or copy them between compatible Labels layers.
- Add optional grid row, column, and cell identifiers from component centroids.
- Review an image/Labels pair without losing the active editing layer.
- Relabel, merge, compose, and calculate agreement across Labels layers.

## Interface

The dock widget follows three common tasks:

### Edit & Review

Choose one Labels layer, edit it, and inspect its connected components.

**Large-label editing** appears first because it controls how napari accesses
the selected layer:

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

### Grid-based review and spatial traceability

The optional grid gives reviewers a reproducible address for every part of a
large 2D mask. With the same cell height, cell width, and source coordinate
system, a reference such as **R12C08** identifies the same location across
review sessions and between collaborators.

- Enable **Add grid coordinates to results** to assign each component a grid
  row, column, and cell from its centroid.
- Enable **Show grid on canvas** to see cell boundaries and identifiers around
  the current viewport.
- Use a grid cell together with a component ID—for example, “R12C08,
  component 431”—to identify a region that needs correction.
- Use the cell reference alone to report an unlabeled area where a component
  is missing and therefore cannot appear in the results table.

Grid references make review discussions traceable without creating cropped
copies of the source. Reuse the same grid dimensions when a reference must
remain stable. The current 1.0 grid is a spatial registration system; it does
not store reviewer comments, assignments, completion states, or a change-history

audit log.
### Visual Compare

Compare an image with a Labels layer while keeping the Labels layer active for
editing. The view controls can show either layer, show both, dim the labels, or
temporarily pulse, blink, and peek between them.

### Combine Layers

Perform label-level operations across one or more compatible layers:

- reassign one label value;
- merge foreground regions into one class;
- compose layers as distinct classes;
- create a per-pixel vote-count map;
- create a consensus map from a minimum vote threshold.

## Editing very large Labels layers

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

With **Save edits automatically** enabled, saving is stroke-aware:

1. Saving never starts while the paint or erase mouse button is held.
2. Releasing the mouse starts a 400 ms inactivity delay.
3. Only the accumulated changed region is compared and written to the source.

Disable automatic saving to keep changes in the local area until **Save now**
is clicked. A local area with unsaved manual changes remains in place during
pan or zoom, preventing accidental loss. **Undo** and **Redo** synchronize
immediately in automatic mode and remain pending until **Save now** in manual
mode.

Tile size controls the memory/performance tradeoff. A 2048 × 2048 area uses
less memory; a 4096 × 4096 area requires fewer reloads while navigating.

## Large component analysis

Large masks are analyzed in tiles and displayed as paged results so the table
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
