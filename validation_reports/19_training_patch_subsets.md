# Training-patch trial subsets

Date: 2026-11-18. Implemented locally; no release or push.

## Behavior

Confirmed review tiles define trusted annotations. Training selection now defines
which eligible **model patches** are used in a run, independently of approval.
Dataset preparation still builds the full bounded coordinate index. Under
**Training → Dataset → Training patch selection**, choose All, Percentage, or
Number of training patches, plus a subset seed for trials. Simple mode includes
a **Training amount…** shortcut to these controls.

Percentages apply only to eligible training patches after the existing spatial
or image/specimen split. For 1,225 eligible training patches, 10% selects 123.
The full reserved validation set remains unchanged. Changing the percentage,
count, or subset seed reuses the prepared index and does not revoke review status
or change annotations. The same subset is reused each epoch; the existing shuffle
and additional target-patch visits operate within it.

Positive and background-only patches are sampled proportionally where available,
with foreground retained. A single-patch trial requires a patch containing known
foreground and background. Class/channel coverage is checked; insufficient
subsets produce actionable messages rather than inferring negatives for unknown
pixels. Model-patch previews and canvas partition overlays use the chosen subset.

The default All option preserves existing behavior. Partial annotations can use
these subsets without enabling the separate legacy Sample patches workflow.
K-fold and final all-reviewed training retain the full-index behavior; trial
controls are disabled for k-fold.

## Implementation and persistence

- `_patch_subset.py`: deterministic metadata-only selection. Array and supervision
  objects are shared; no source image or whole-mask copies are made.
- `_coverage_train.py`: optional dataset selection metadata and completed-round
  history. Training architecture, losses, optimizer, and validation calculations
  are unchanged.
- `_quick_train_widget.py`: budget controls, caching, correct train/comparison/
  preview dispatch, workload and model-history summaries, and optional workspace
  settings. The full prepared pool stays available when returning to All.
- `_model_comparison.py`: selected-patch policy in dataset snapshots.
- `_simple_curation.py`: shortcut to the existing training selection controls.
- `README.md`, `docs/TRAINING.md`: workflow, units, and validation-time guidance.

Workspace JSON version remains 2. Missing settings default to All. Completed
checkpoints record the policy, seed, available/selected counts, and actual patch
coordinates. Existing training-footprint recording therefore marks only patches
used in training, excluding reserved validation. Previous local work was retained.

## Validation

- **114 tests passed** in the broader training run, including multiclass,
  multilabel, spatial validation, checkpoint loading, and image/specimen pairs.
- **136 tests passed** in the final workflow run, including Simple access, the
  existing training widget, annotation review, models/checkpoints, workspace
  persistence, and the new subset tests. These runs overlap; they are not 250
  unique tests. The complete repository suite was not rerun for this change.
- Fifteen new test cases cover all/default identity, percentage rounding,
  count limits, deterministic/nested subsets, stable validation, missing
  supervision, per-channel validity, real CPU training and checkpoint history,
  exact training footprints, no source pixel reads during selection, source
  approval preservation, selected-dataset dispatch, workspace roundtrip, k-fold,
  and the Simple shortcut.
- The controls were rendered offscreen in `napari-dev2` using a synthetic dataset:
  25% used 3 of 12 training patches, with all 4 validation patches retained.
- `git diff --check`: passed. No dependencies added.

Tests ran offscreen in the existing `napari-dev` environment with its SQLite
preload workaround. No researcher workspaces, annotations, or running training
were modified for validation. Live GPU throughput was not benchmarked.

## Practical limits

The validation set is deliberately not subsampled. A 10% training budget does
not imply a 90% reduction in total run time, because full validation still runs
after each epoch. Initial indexing still examines eligible annotation patches;
subsequent budget changes avoid that scan. This change adds percentage/count
sampling, not individual manual patch checkboxes or per-epoch resampling.
