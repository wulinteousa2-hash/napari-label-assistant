# Managed grid-mask loading

Implemented 2026-11-05 on uncommitted changes based on
`cb95a5e97717aca24db658c488fd810647955bcd`, retaining the prior coordinate patch.
Original audit findings/evidence describe the reviewed commit and remain historical.

## User workflow

Open the existing workspace. In Automatic mode, large 2D Zarr Labels stores stay
outside napari's layer list; only current mask crops are added. The multiscale
image path is unchanged. This applies to Curated and associated full-size review
masks, rather than leaving hidden full-size review layers loaded.

The existing Labels → Local Editing → Editing mode selector includes Grid tile
editing (low memory). Its navigator provides 512/1024/2048/4096-pixel sizes,
zero-based R/C addresses, Previous, Go, Save & Next and Mark done. Global source
pixel bounds are displayed. Irrelevant local-area settings are hidden while the
grid navigator is active. A manual mode selection before opening overrides saved
preferences. Local-area and Full-layer modes remain available explicitly.

Normal saves write full-source references, original coordinate transforms and
grid-mode preferences/progress. Portable saves and Save As copy complete masks,
then rebind subsequent edits to the copies. Newly created aligned tile masks use
sparse full-size temporary stores and are persisted on Save. No separate stitch
operation or user-managed crop files are required.

## Correctness and recovery

- Painting and erasing write to original global pixel coordinates. Source image
  intensities are unaffected. Labels are not resampled or renumbered.
- Only the changed bounding rectangle is written, preserving unchanged pixels
  within it. Out-of-range label values and altered crop registration are rejected.
- Edits flush after a short idle interval, before navigation/save, and before
  managed layers are removed or the viewer closes. Active strokes block writes.
- A bounded, fsynced recovery record is written beside the mask store before a
  Zarr write. Reopening replays compatible interrupted writes; conflicting values
  or changed recovery dimensions/dtype are refused and the record retained.
- A retry after partial write completion establishes the recovered baseline before
  applying later edits, including an erase back to the original background.
- Concurrent changes detected on edited pixels are refused. This is designed for
  one editing session per store, not a distributed multi-user locking protocol.
- Undo history is reset when changing crop buffers to avoid applying an old tile's
  stroke to another region. Native undo remains available within the active tile.

## Existing scientific workflows

Current-pair, paired-image and multilabel dataset preparation resolve managed
masks to their full backing arrays and full transforms. They do not treat the
visible crop as a whole-image training target. Full-mask training libraries also
read disk-backed sources; locating an object loads its grid tile. Changes dirty
the correct global library regions and invalidate affected existing review metadata.

Ordinary component inspection operates on the visible tile, with an explicit
scope notice. Objects at boundaries can continue in neighboring tiles; local
counts/IDs are not global scientific measurements. The existing reviewed-cell
coverage editor currently requires Local-area or Full-layer mode. Its saved masks
and associations are preserved. Mark done records navigation progress only and
does not certify positive/negative supervision for ML.

Navigation shares a grid address across managed workspace masks. Masks with
smaller extents are hidden/disabled where the active cell does not intersect them;
an unrelated edge crop is not substituted. Unsupported non-2D Labels and external
reader-based mask loading retain their existing paths.

## Executed verification

| Check | Result | Evidence |
|---|---|---|
| Full suite, including coordinate and grid regressions | 333 passed, no failures/errors/skips; 227.93 s | [Log](evidence/grid_tiles/full_suite.log), [JUnit](evidence/grid_tiles/full_suite.xml) |
| Final grid and widget checks, including compact controls | 29 passed, no failures/errors/skips; 28.88 s | [Log](evidence/grid_tiles/final_widgets.log), [JUnit](evidence/grid_tiles/final_widgets.xml) |
| Component-library/workspace integration | 46 passed, no failures/errors/skips; 35.32 s | [Log](evidence/grid_tiles/library_workspace_integration.log) |
| Patch whitespace / new-module syntax | Passed | `git diff --check`; Python `ast.parse` |

Covered cases: automatic pre-display loading; exact global paint/erase and edge
tiles; repeated save/reopen; full portable-copy equivalence and relocation; Save
As independence; saved preferences and explicit overrides; new managed masks;
close-time flush; bounded writes; conflict/dtype/registration rejection; interrupted
and partially completed writes; different mask extents; full-mask training; global
library dimensions/locations; and workspace dirty-state changes for tile completion.

Tests use synthetic stores under `/tmp/nla-mask-tiles`, Qt offscreen, isolated
config/cache and the napari-dev2 interpreter. Existing Pydantic/napari deprecation
warnings remain. The full run preceded the final control-visibility adjustment;
the subsequent focused run covers that adjustment. No dependency was installed.

## Opening-memory experiment

Three sparse local Zarr masks, each **84,175 × 79,966**, with dtypes uint32,
uint8 and uint8: **40,386,828,300 logical source bytes** in total. Automatic loading
produced exactly three **1024 × 1024 NumPy mask buffers**:

- Visible mask pixels: **6 MiB**.
- Original snapshots used to identify edits: **6 MiB**.
- Process RSS before opening: **384,786,432 bytes**.
- Process RSS after opening/event processing: **481,460,224 bytes**.
- RSS increase: **96,673,792 bytes**, about **92.2 MiB**.
- Opening time: **0.4564 seconds** on this machine.

[Raw measurement](evidence/grid_tiles/opening_memory.json) and
[reproduction script](repro/benchmark_grid_mask_open.py). The test independently
asserts that full-sized sources never enter `viewer.add_labels`; sources remain
Zarr arrays in lightweight adapters, not offscreen full-sized napari Labels objects.

This measurement isolates masks, uses highly compressible sparse local stores and
Qt offscreen, and ran on the available high-end machine. It demonstrates bounded
display buffers, not equivalent timings for real masks, network storage, image
rendering, or lower-end hardware. Source chunk geometry still determines Zarr
decompression memory. Real desktop responsiveness on the affected computer remains
to be measured. Full-image inference output allocation is a separate limitation.

## Reproduction and provenance

From the repository root, use the napari-dev2 interpreter with
`PYTHONPATH=/tmp/label-assistant-test-deps:src`, `QT_QPA_PLATFORM=offscreen`,
`PYTHONDONTWRITEBYTECODE=1`, OMP/MKL threads=2, and isolated XDG/Matplotlib paths:

```bash
python -m pytest tests -q -p no:cacheprovider --basetemp=/tmp/nla-mask-tiles/reproduce
python -m pytest tests/test_mask_tiles.py tests/test_label_assistant_widget.py -q -p no:cacheprovider
python validation_reports/repro/benchmark_grid_mask_open.py
```

[summary.json](evidence/grid_tiles/summary.json) records suite totals, times,
base commit and changed-file hashes. Source/docs changes are uncommitted. No
production project was opened with the new loader or modified during validation.
Nothing was pushed. This implementation does not resolve every original audit
finding or independently establish release readiness.
