# Editing, confirmation, and mask-analysis interface

## Result

- **Edit & Confirm** retains the existing camera-follow, Paint/Erase, bounded
  editing, edit undo/redo, and workspace saving/loading controls. The review grid
  and its confirmation actions are visible in the same panel. Review-tile size
  remains independent of the memory-limited editing crop and model input patches.
- **Confirm tile for dataset**, **Confirm tile & next**, and continuous canvas
  confirmation approve objects and checked background. Status distinguishes
  unconfirmed, confirmed, and edited tiles needing recheck. Historical training
  use remains a separate indicator.
- **Mask Analysis & Cleanup** focuses on Find components, sortable index tables,
  canvas/table inspection, area thresholds, cleanup preview, removal, copy, and
  undo. The scope label identifies a bounded edit area versus the selected mask
  layer. Grid-address and destination settings are collapsed advanced controls.
  Analysis/table selection never grants dataset approval.
- **Training → Dataset** shows confirmed-tile counts, a return-to-editor action,
  coverage display, preparation, and a training/validation patch overlay. Older
  foreground-only selections, coverage-mask controls, and loaded-area confirmation
  remain under advanced controls; saved selections are preserved.
- Loaded-area confirmation moved out of the analysis tab into a collapsed editor
  section. The standard grid-confirmation workflow does not require that section.

## Validation

Rendered and visually inspected all three panels with disposable image/mask data.
Preview files: `/tmp/edit-confirm-ui.png`, `/tmp/mask-analysis-ui.png`, and
`/tmp/training-dataset-ui.png`. Escaped Qt tab mnemonics so both ampersands display
literally. Inspected the collapsed cleanup settings and the expanded component table.

The broad regression run covered 154 tests across tile confirmation, reviewed
regions, annotation semantics, label tools, top-level navigation, dataset pairs,
training widgets, camera following, bounded mask persistence, and dynamic editing.
153 passed initially; one assertion expected the previous missing-coverage guidance.
After updating that expected wording, three targeted checks passed, including
that assertion, prepared-split display, and selection without training approval.
No functional regression remained from this run. `git diff --check` passed.

No user image, mask, or workspace file was modified for this validation.
