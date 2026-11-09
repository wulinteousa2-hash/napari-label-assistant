# Entire-mask component analysis independent of zoom

## User-facing behavior

Mask Analysis & Cleanup defaults to **Entire mask — full resolution**. The
**Current edit area** alternative retains local inspection. The entire-mask
search uses the original full-resolution backing store rather than the editable
crop or a downsampled overview. It does not enable full-layer rendering.

The existing streaming index joins components across chunk boundaries, reports
progress, supports cancellation, and pages large result tables. Global results
remain valid across viewing-crop navigation; real annotation edits require
reanalysis. Picking and locating use the original source transform.

Global cleanup writes selected components in bounded regions through the existing
recovery journal. Undo captures selected original values on disk and preserves
later nonzero edits. Copy uses full-image placement and retains bounded matching
destinations. Current-area operations remain available explicitly.

## Complete validation of the actual mask

Source: `/home/wteox/Works/EM_core/EM data July 10th/P2MPZ/p2mpz EM25 32.json`,
layer `curate`, full-resolution `s0`, opened read-only in `napari-dev2`.

- Dimensions: **49,976 × 59,716** pixels.
- Complete search: **16 connected components**.
- Total foreground: **22,211,355 pixels**.
- Measured elapsed time: **30.19 seconds** in this environment.
- The search visited all 750 source chunks and joined seams before reporting
  global objects. A second bounded pass calculated topology measurements.
- Original JSON, mask arrays, and images were not modified. Temporary index data
  and the detailed result were written only under `/tmp`.

This is a whole-mask count, superseding the two-region sampling reported in
validation report 16.

## Regression checks

Disposable OME-Zarr tests cover an empty loaded crop with objects elsewhere,
overview-only state, objects crossing both 2048-pixel chunk seams, bounded reads,
global picking/locating after crop navigation, global copying, noise removal, and
undo. The main regression run passed **82 tests** across full/local mask analysis,
OME pyramids, coordinates, paging, cleanup, and mask persistence. Additional
checks exercise pre-existing manual destination matching, cancellation,
cleanup/undo, and bounded destination preservation.
