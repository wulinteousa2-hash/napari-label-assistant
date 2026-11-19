# Persistent batch history colors

Implemented locally on 2026-11-19.

## Behavior

- Simple **3. Progress & Dataset**, Advanced Labels, and Advanced Training Dataset
  share colored batch-button markers and a **Show: Active batch / All batches /
  Hide** control.
- Newly approved regions retain their batch color automatically. Pending canvas
  selections remain yellow and appear above batch-history overlays.
- Active batch borders stay about 2.5 screen pixels wide with a light fill.
  Earlier batches use dimmer borders and lighter fill in All batches.
- Overlapping history regions are clipped by coordinates: the active batch takes
  priority, followed by the most recent earlier batch. Colors do not blend into
  contradictory identities. Fragment counts do not change stored membership counts.
- A red × marks member regions that are no longer fully reviewed after edits or
  revocation. Their borders retain their batch identity.
- Clicking a batch button makes it active for annotation and viewing. The separate
  **Training: Batch N** dropdown names the training selection explicitly. Viewing
  another batch or changing the display filter leaves that training selection and
  prepared dataset intact. Explicit training-selection changes invalidate preparation.
  **+ New batch** continues to select the new batch for annotation and training.
- Settings and colors persist per image/Labels pair using the existing optional
  annotation-batch metadata. Older records receive deterministic fallback colors;
  their colors are stored on the next batch-setting change. Historical dates and
  workspace format/version are unchanged.

## Implementation and data safety

One shared viewer controller renders a Shapes layer containing coordinate geometry
and review-status features. Internal display geometry is excluded from workspace
layer serialization. It does not read or duplicate full-resolution pixel arrays,
modify label values/colormaps, or alter supervision validity. Rendering is coalesced
and deferred during multi-tile approval to keep the tile operation responsive.
Overlap clipping uses a compact coordinate index to compare nearby regions,
avoiding an all-pairs scan for ordinary review grids.
Active editing-layer selection is preserved when the overlay is created or updated.

Colors represent review membership. Actual training-patch use remains recorded
in model histories; colors do not imply every patch was trained or freeze previous
annotation pixels.

## Files

- New: `src/napari_label_assistant_tools/_batch_overlays.py`
- Updated batch metadata/settings: `_annotation_batches.py`
- Updated shared controls: `_annotation_batch_widget.py`
- Cleanup integration: `_simple_curation.py`
- New regression coverage: `tests/test_batch_overlays.py`
- Updated workflow tests: `tests/test_annotation_batches.py`,
  `tests/test_simple_curation.py`
- Updated documentation: `README.md`, `docs/TRAINING.md`,
  `docs/model-management.md`
- This report.

Existing local changes were preserved. No user workspace was modified, and no
commit, push, dependency addition, model change, or inference change was made.

## Validation

Broad regression suite: **332 passed**, 1,216 warnings, in 497.58 seconds.
It covered annotation batches, Simple/Advanced curation, reviewed regions, coverage
training, model management/naming/selection, training-patch subsets, multiclass and
multilabel training, workspace manager/widget, copy destinations, mask lifecycle,
grid display, and coordinate contracts. No failures remained.

Focused final checks: **12 passed**, including automatic color retention after
approval, shared controls, stable colors, overlap priority, pending-selection
layer order, source switching, edited/revoked regions, independent training
selection, and workspace reopening. This final run also included a **10,000-tile**
coordinate-history comparison-count check and a strengthened sparse **100,000 ×
100,000** disk-backed mask test with bounded-read assertions while creating the
batch overlay. The focused run verified the final coordinate-index optimization
and replacement-button visibility cleanup after the broad run was collected.

A standalone offscreen harness in the user's **napari-dev2** environment passed:

- Batch 2 overview selection and approval.
- Persistent colors, All/Hide, edited-region crosses, and independent training
  selection.
- Advanced review selection from overview.

A dark-theme Qt grab verified the narrow Simple progress panel, colored button
markers, explicit training name, display selector, and review-status legend.
`git diff --check` passed.

## Limits

Automated GUI checks use Qt offscreen. Interactive rendering/input should be checked
after saving the active session and restarting napari. A finite palette supplies
the first eight batch colors; additional batch numbers use deterministic hues.
Button names and stored IDs remain available alongside color for identification.
Exact historical annotation-pixel restoration is outside this change.
