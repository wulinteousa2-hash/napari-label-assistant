# Partial-training workflow: real workspace validation

Validated read-only on 2026-11-08 using:
`/home/wteox/Works/EM_core/EM data July 10th/P2MPZ/p2mpz EM25 32.json`.
Original JSON, images, and Zarr masks were not changed. No training was run.

## Findings

- Image, `curate`, and reviewed-region stores open read-only and have matching
  dimensions: 49,976 × 59,716. Saved transforms match.
- Reviewed-region metadata correctly associates the image and `curate` IDs.
- Metadata records 11 nonoverlapping 100 × 100 approved cells with no invalidated
  regions: 110,000 pixels, including 66,159 foreground and 43,841 background.
- Training settings select multiple-pair mode with only one included pair.
  Its `reviewed_id` is null, although the current-pair `reviewed_combo` setting
  references the correct reviewed-region layer. Pair resolution therefore rejects
  the missing association. After that repair, multiple-pair mode would still
  require at least two included image–label pairs.
- The annotation summary uses the pair's null reviewed ID and consequently reports
  zero reviewed cells. Tile confirmation actions are hidden in multiple-pair mode
  by `_update_annotation_note`. Users cannot fix the problem from the visible panel.
- Saved review-grid dimensions are 100 × 100, bounded navigation size is 512,
  and source/model patches are 512 × 512. These are separate settings despite
  similar tile terminology.

## Coverage discrepancy

The stored coverage pixels disagree with saved approval rectangles. In the four
512 × 512 patches intersecting the approved rectangles, all 1,048,576 stored pixels
are nonzero, whereas metadata approves only 110,000 pixels there. The store's fill
value is zero. The origin of the discrepancy has not been established.

`TrainingSupervision` uses metadata rectangles to choose candidate patches but
reads coverage pixels directly for validity. It does not intersect those pixels
with the saved approval rectangles. This can include pixels outside the saved
selection. This must be reconciled before recommending the current preparation
output as representing the user's intended selection.

## Bounded preparation experiments

Ran `index_patches` on read-only full-resolution arrays with saved split seed 44.
Also tested an in-memory coverage wrapper intersecting coverage pixels with the
11 saved rectangles; this changed no files.

| Source/model patch | Result restricted to saved approvals |
|---|---|
| 512 / 512 | Rejected: reviewed targets lack two separate spatial blocks |
| 256 / 256 | 6 training patches / 1 validation patch; 101,600 / 8,400 reviewed pixels |
| 128 / 128 | 12 training patches / 4 validation patches; 69,760 / 40,240 reviewed pixels |

These demonstrate preparation feasibility, not model quality or an independently
representative validation dataset. All approvals are concentrated in a small area.

## Proposed user flow

1. Load workspace.
2. Zoom and edit the curated mask. Optional mask cleanup belongs here.
3. Open one visible **Training tiles** section and choose review-tile size.
4. Review and confirm successive tiles on the canvas, with persistent selection
   mode, visible tile status, and automatic image/mask/coverage association.
5. Prepare a partial dataset from confirmed tiles. Show training and validation
   counts and locate their tiles on the image. Unconfirmed pixels remain unknown.

Use separate visible states for confirmed approval, prepared training/validation
membership, and historical model-training use. Keep memory-crop and model-patch
settings advanced; neither should silently change the user's review grid.

For optional SAM3 cleanup, expose component indexing, an area threshold in pixels,
a removal preview, explicit current-tile/whole-mask scope, and undo. Whole-mask
cleanup must handle connected components crossing storage boundaries. Cleanup
after confirmation invalidates affected approvals until re-review.

Prioritize association consistency and coverage validity before reorganizing UI.
For one selected image, use the single-image preparation path automatically rather
than requiring users to understand pair modes or manually select coverage masks.

## Implemented fixes and revalidation

- Pair resolution and widget refresh recover a missing coverage reference only
  when exactly one reviewed mask matches both source IDs. Explicit wrong references
  and ambiguous matches still produce errors.
- Restoring a one-image selection in multiple-pair mode uses the single-image
  preparation path and spatial validation automatically.
- `TrainingSupervision` now intersects stored coverage with saved approval
  rectangles. Empty rectangle metadata approves no pixels; external coverage
  without rectangle metadata retains its explicit validity mask.
- Dataset exposes review-tile dimensions, persistent click-to-confirm mode,
  coverage outlines, immediate confirmation feedback, and a current-area cleanup
  shortcut. Missing spatial groups offer an explicit smaller-patch retry.
- Repeated read-only indexing of the actual source stores with the fixed code
  reports zero supervised pixels outside metadata approvals. At 256 px it produces
  6 training / 1 validation patches, covering exactly 110,000 approved pixels.
  At 512 px it retains the insufficient-spatial-groups error rather than weakening
  validation or silently approving more pixels.
- Broader regression run: 113 tests passed across annotation semantics, component
  libraries, dataset pairs, tile confirmation, training widgets, reviewed regions,
  and multilabel training. Additional checks cover subsequent UI feedback and
  cleanup-shortcut changes.
