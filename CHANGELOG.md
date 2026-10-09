# Changelog

All notable changes to napari-label-assistant are documented here. The project
uses [Semantic Versioning](https://semver.org/).

## [3.1.0] - Unreleased

- Separated required Prepare dataset from optional Show patch preview. Preparation leaves the canvas unchanged; contact-sheet previews respect source world scale, sit beside the image, and offer Return to image. Outdated previews are hidden when the dataset changes.

- Documented direct SMP installation, interpreter verification, restart steps, and troubleshooting of missing packages versus dependency import failures; synchronized training-page documentation.

- SMP import errors now distinguish a missing package from a broken dependency, display the actual napari Python executable, and preserve the underlying error. Installation guidance uses the SMP package directly rather than an unreleased plugin extra.

- Added a task-oriented Models page between Dataset and Train, with target definitions, architecture configurations, checkpoint/run selection, configuration reuse, and training history.
- Added a shared model factory for the existing small custom U-Net, standard U-Net, optional SMP U-Net/U-Net++/DeepLabV3+ encoders, and registered custom Python providers. Checkpoint version 2 records model specifications and dependency versions; version-1 checkpoints remain readable.
- Added controlled sequential model comparisons using identical indexed data and spatial partitions, fresh candidate/fold weights, bounded input fingerprints, validation metrics, parameter counts, training time, and CSV export.
- Extended workspace format to version 2 while retaining version-1 loading. Workspace Save persists changed model weights and per-run checkpoints; complete-package Save As includes checkpoints and comparison reports using relative references. Large histories remain in sidecars, and networks load only when selected.
- Handle singleton batches with running BatchNorm statistics and retain the previous valid workspace manifest when checkpoint persistence fails.

- Reorganized Training into Dataset, Models, Train, Inference, and Advanced pages. Dataset
  owns reviewed-cell confirmation and the training component library; Labels
  retains canvas cell selection, mask editing, and component QC. Detailed training
  metrics expand on demand; annotation tables share the existing editing state.


See [the version 3 release plan](docs/V3_RELEASE_PLAN.md) for completed UI
changes and the implemented persistent component-library training workflow.

- Connect local editing to reviewed-area marking and removal; preserve Curated object IDs, automatically select image-associated coverage in partial Training mode, and persist associations in workspaces. Add a bounded read-only preview of reviewed annotations.

### Fixed

- Dataset's “Mark already-checked cells with clicks” now activates canvas review
  mode and selects the marking action directly. Turning it off restores editing;
  hiding the address grid clears click-marking mode as well.

### Added

- Added a Validation box in Training → Train with configurable spatial holdout, independent spatial k-fold evaluation, visible fold assignments and per-fold/mean/standard-deviation metrics. Added a separate final-model action using all reviewed patches, fold-model/report exports, and exact validation provenance in model history.
- Added live validation learning curves for Dice, IoU, precision and recall, plus standalone SVG charts saved with models and validation reports. Renamed additional target sampling to Extra visits to patches with objects (%).

- Advanced augmentation controls for rotations, flips, and brightness/contrast;
  optional validation-loss plateau handling with one learning-rate reduction and
  configurable early stopping. Persist settings/events and actual epoch counts
  in model history while retaining the best validation Dice checkpoint.
- Batch sizes up to 128, source patches up to 4096, model input sizes up to 2048,
  and maximum training rounds up to 1000 epochs.


- Added a paged, disk-backed training component library in Training → Dataset → Selected objects, with
  persistent IDs, label pixel counts, bounds, and Train checkboxes. Library scans
  skip Euler calculations and run in a cancellable worker; checkbox changes
  update SQLite records without refreshing the full image.
- Persist library records, compressed ID tiles, source fingerprints, selections,
  and retired IDs with workspace Save/Open, Save As, and portable exports.
- Combine selected foreground components with manually reviewed regions, ignore
  unchecked surroundings, deduplicate patch coordinates, and keep each component
  out of the opposite validation split.
- Update affected components after Editable Area and QC copy/delete changes.
  Preserve unambiguous unchanged IDs; new objects start unchecked; splits/merges
  allocate new IDs. Edited review cells require background review again.

- Automatically use all-patch coverage training for partial annotations and disable the incompatible sampled-patch workflow, preventing reviewed cells from failing at preview.

- Simplified cell review to select, edit Curated, and Finish review. Moved direct click marking into Advanced, removed the redundant local-preview action, and automatically select associated Curated ground truth and reviewed coverage in Partial Training mode. Hide manual mask creation and exclude generated helper masks from ground-truth choices.

- Added mouse-driven address-grid review: select cells, mark/unmark with clicks, yellow selection and green reviewed outlines, and shared grid-size controls. Marking merges accepted objects into a disk-backed Reviewed annotations layer with value 1 while preserving original object IDs.

- Added default **Coverage training**: index eligible image patches once, stream
  batches from disk, and visit every training patch each epoch. Reserved spatial
  validation regions remain separate; progress reports unique patch coverage.
- Added partial-annotation support with a separate reviewed-region mask. Pixels
  outside fully reviewed regions and image-edge padding are excluded from loss
  and validation. Empty disk-backed reviewed masks can be created from Training.
- Added configurable epochs, batch size, and additional target-patch visits.
  Representative galleries stay bounded instead of loading the full dataset.
- Show model state explicitly as in memory or saved to disk. Continue training
  preserves existing weights on cancellation and retains the best validation
  checkpoint. Coverage history and validation block assignments are saved with
  model weights; sampled-crop training remains available for quick experiments.

### Changed

- Increment the package version from 3.0.0 to 3.1.0 and refresh README highlights
  for the current training, model-management, validation, and workspace workflows.

- Organize Quick Train into Dataset, Train, Advanced, and Inference pages with concise
  guidance. Use saved bounds to index only candidate patches where available.
- Replace duplicate manual-review data refresh notifications with lightweight
  training notifications.

- Refined Workspace labels and tooltips, including **Include all images and
  masks** for portable saves intended for sharing, transfer, or archiving.
- Removed the overlapping snapshot action. Grouped **New**, **Open**,
  **Open Recent**, **Save**, and **Save As** in one file-operation row.
  Open Recent provides a menu; recent entries show names, folders, and availability.
- Added explicit linked and portable save modes, with storage selection above
  image optimization and a separate indicator for the last saved workspace.

### Fixed

- Stopped repeated editable-area reloads and status flashing at image edges.
  Camera navigation preserves unrelated layer focus and pending manual edits;
  active paint strokes are not interrupted by area reloads.

- Retained unavailable layer records after partial loading and blocked saves
  until the workspace can be fully reopened, preventing accidental data loss.

## [2.0.2] - 2026-10-02

### Fixed

- Complete package saves with image optimization now recognize napari
  MultiScaleData and preserve existing pyramid levels instead of attempting
  to index the pyramid wrapper as a single image array.

- Detect incomplete or shadowed Zarr installations with the imported module
  location and actionable diagnostics before clearing existing viewer layers.
- Distinguish dependency failures from missing workspace data in load warnings.
- Support the Zarr 2 array-creation API as well as Zarr 3.

## [2.0.1] - 2026-10-02

### Added

- Added **Complete package copy (images and masks)** beside image optimization.
  Saves include all image and mask arrays with relative paths, including fileless,
  RGB, and multiscale images. Transfer the manifest and data folder together to
  reopen on another computer without the original mounts or reader plugins.

### Fixed

- Opening a workspace with missing linked data now displays the affected layer
  names and failure reasons, with guidance for making a complete package.

## [2.0.0] - 2026-10-02

- Embed completed training-round history in saved Quick Train models and export it as a neighboring `.history.json` file, preserving it across loading and continuation.

- Continue training current or loaded Quick Train weights on reviewed crops, with adjustable learning rate, sampling seed, saved round/step counts, and cancellation preserving the existing model.
- Keep spatial validation blocks fixed across sampling seeds and allow up to 10,000 training steps.

### Added

- Added a dedicated **Training** tab alongside Workspace, Labels, Compare,
  and Combine.
- Added **Training → Quick Train** for optional PyTorch U-Net training from
  complete image/ground-truth pairs, spatial validation scores, local mask
  previews, and model save/load. Training and prediction use background workers
  with cancellation and bounded reads of large images.
- Added **Prepare & Preview Crops** with aligned image/mask galleries, source
  coordinates, source navigation, and actual positive/background counts. Sparse
  annotations are discovered with a tiled full-resolution ground-truth scan;
  sampling uses guaranteed positives and contextual negatives in separate spatial
  training/validation blocks. Reviewed crops are reused unchanged for training.
- Exposed 256/512-pixel training resolution and changed the default context and
  input to 512 pixels to preserve detail. Foreground-preserving mask pooling
  keeps tiny targets when downsampling.

### Changed

- Save As to a new destination now creates independent writable mask copies;
  source images remain linked. Ordinary Save reuses existing mask stores.

### Fixed

- Save As can copy read-only shared Zarr masks into local writable storage.
- Saves show progress and visible error dialogs. Mask bindings are updated
  only after the manifest is successfully written.

## [1.1.0] - 2026-09-30

### Added

- Added manual copy-destination selection and automatic creation of a Labels
  layer matching the source's 2D shape. Incompatible existing destinations are
  preserved while a matching destination is created and selected.
- Added **Remove small components** with an adjustable area threshold (default:
  below 2 pixels) to clean the source mask and results across all pages.

### Changed

- Grouped **Find components**, **Delete selected**, and **Copy selected** together
  and removed the redundant **Copy again** button.

### Fixed

- Large-image analysis cancellation no longer raises a main-thread traceback.
  Controls and layer editability are restored, and analysis can be restarted.
- Analysis failures are reported through the status message without reraising
  worker exceptions in the Qt event loop.

## [1.0.3] - 2026-09-25

### Added

- Added an optional **Optimize large images for viewing** Workspace setting.
  On Save, oversized single-scale 2D images can be copied once into a local,
  chunked multiscale OME-Zarr pyramid with visible tile-level progress. The
  original file is preserved, unchecked mode retains file references, and the
  optimized view becomes active when the workspace is reopened.

- Replaced the static top header with a fixed-size layer-activity indicator
  that reports viewport preparation, local editable-area loading, and
  readiness without resizing the dock or requiring users to watch system CPU
  or memory monitors. The header retains the original high-contrast light
  background and blue border while communicating state through its icon.

### Changed

- Clarified the interface hierarchy as **Workspace**, **Labels**, **Compare**,
  and **Combine**. Within Labels, **Local Editing** now identifies the
  bounded-memory paint workflow, while **Review & QC** identifies full-layer,
  grid-addressed component inspection and curation.

### Fixed

- Preserved the active paint label value across pan- or zoom-driven editable
  area reloads and when returning from local-area to full-layer editing.

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
