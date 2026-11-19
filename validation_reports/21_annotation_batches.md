# Annotation batches

Implemented locally on 2026-11-19. This extends the existing Simple/Advanced
workflow; it does not redesign the model or change the optimizer.

## Workflow

- Simple: **3. Progress & Dataset → Annotation batches**.
- Advanced Labels: **Local Editing → Confirm tiles for dataset**.
- Advanced Training: **Dataset → Annotation batches**.
- The same numbered buttons, active selection, and training scope appear in all
  three locations. Buttons show member-region counts; selecting one highlights
  its regions without changing annotation pixels.
- **+ New batch** starts an empty workset. Subsequent approvals join that batch.
  Global review coverage remains cumulative. A new selection clears pending canvas
  selections so they cannot accidentally be approved into the next workset.
- **Training: active batch only** limits eligible supervision to its coordinates.
  **Training: all reviewed annotations** restores the cumulative sources.
  Percentage/count patch sampling and Quick trial remain available after indexing.
- Right-click a batch to add selected reviewed regions, remove membership without
  revoking review, rename, finish/reopen, or inspect review history and training use.
  Membership actions preserve multi-tile selections when switching batches.
- Prepare the desired batch, select the earlier checkpoint, and use the existing
  **Continue training** action. Model architecture/weights are retained. A batch
  does not automatically create a new model or mark every member patch trained.

## Safety and persistence

- Optional `label_assistant_annotation_batches` metadata stores stable batch IDs,
  names, coordinate membership, membership revisions, and UTC review events.
  It uses existing workspace serialization and the tile-progress journal; the
  workspace format/version is unchanged.
- Workspaces without this metadata retain cumulative preparation. On first
  initialization, existing review membership becomes Batch 1. Missing historical
  approval dates remain unknown. Historical training runs are not retroactively
  attributed to Batch 1.
- Global reviewed masks, unknown-pixel exclusion, selected-object validity, and
  independent multilabel channel validity remain authoritative. Batch membership
  only restricts existing supervision; it cannot approve unknown pixels.
- Pending unrelated component selections do not force a global component-library
  rebuild for an active reviewed-region batch. Those selections are excluded;
  cumulative preparation retains its existing library-update requirement.
- Membership restrictions allocate patch-sized validity arrays, with no full-image
  annotation copies or scans. A synthetic 100,000 × 100,000 validity source rejects
  whole-array conversion in the regression test.
- Earlier checkpoint holdout coordinates remain reserved, including omitted areas
  of Quick-trial validation blocks. Loading a checkpoint after preparation also
  reassesses the holdout. Inadequate independent supervision gives an actionable
  error; earlier training exposure continues to block unsafe continuation.
- Newly copied/combined Labels outputs do not inherit another source's batch IDs.
- Training provenance records the selected batch ID/name/revision/coordinates and
  actual patches used. Readable checkpoint filenames include the batch name for a
  single active source. Metadata does not freeze historical annotation pixels.

## Files

New implementation:

- `src/napari_label_assistant_tools/_annotation_batches.py`
- `src/napari_label_assistant_tools/_annotation_batch_widget.py`

Integration into existing local work:

- `_simple_curation.py`, `_reviewed_regions.py`, `_quick_train_widget.py`
- `_curation_shell.py`, `_widget.py`, `_copy_destination.py`
- `_mask_tiles.py`, `_tile_progress.py`, `_model_naming.py`
- `workspace/service.py`

Tests/documentation:

- `tests/test_annotation_batches.py` (new)
- `tests/test_simple_curation.py`, `tests/test_copy_destination.py`
- `README.md`, `docs/TRAINING.md`, `docs/model-management.md`
- This report.

Existing unrelated local changes were preserved. No commit or push was performed.

## Validation

Final results: **293 passed** in the broad regression run (341.88 s), plus
**1 passed** in the additional checkpoint-after-preparation regression test.
No failures remained. `git diff --check` passed.

The broad run included annotation batches, Simple curation, reviewed regions,
coverage training, Quick Train widgets, model management/naming, training-patch
subsets, multiclass and multilabel training, workspace manager/widget, copy
destinations, mask-layer lifecycle, and model-selection dataset tests. It emitted
1,213 warnings with warning details suppressed by pytest.

Environment: `napari-dev` Python, `QT_QPA_PLATFORM=offscreen`, with its bundled
`libsqlite3.so.3.53.4` preloaded. All model training checks used CPU. The additional
checkpoint test specifically verifies that loading an earlier checkpoint after
preparing a batch moves its old held-out spatial groups into validation and keeps
safe continuation available.

An offscreen Qt preview checked the actual Simple section at 468 px wide and the
Advanced selector at 446 px wide. Regression tests exercise real Qt interactions,
CPU training provenance, workspace save/reopen, progress-journal recovery, review
undo, batch-only indexing, empty-batch feedback, validation protection, independent
channel validity, and disk-backed mask destinations.

## Limits

- Full interactive napari rendering and GPU throughput still require manual use;
  automated GUI checks use Qt's offscreen platform.
- Batch membership is per image/Labels pair. Multi-image/channel configurations
  retain explicit source inclusion and each pair's selected scope.
- Finish prevents new approvals until reopening; subsequent pixel edits still
  invalidate existing review according to the existing review policy.
- Exact historical annotation-pixel restoration is not provided. Keep a separate
  workspace snapshot when an immutable annotation version is needed.
