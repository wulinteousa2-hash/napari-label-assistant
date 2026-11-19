# Review selection from mask overview

2026-11-19: fixed the reported inability to select tiles after starting Batch 2.

## Cause and correction

The visible managed Labels layer can contain a 1 × 1 placeholder while the
disk-backed mask is displayed through its overview. Grid construction read that
placeholder's shape, rejected it as an unsupported 2D mask, and turned mouse
review off. Simple mode still displayed “Review — select tiles,” leaving its
approval action disabled with no pending selection. This was reproducible without
the user's workspace and was not a missing approval step.

Grid construction and viewport address labels now use the full source mask's
dimensions and registration. They inspect shape/transform metadata without
materializing the mask. Full-source coordinates also keep grid lines and addresses
aligned when the editable layer contains an offset crop. Rotation, shear, and
affine registration are carried to both display helpers.

Simple Review mode now explicitly says to click tiles on the image and then
approve. Its disabled empty-selection button reads “Click tiles on image first.”
If mouse review cannot activate, the reason appears in the visible Review status
instead of the unrelated overview/editing message.

Batch counts remain approval-based: a yellow pending selection enables approval,
but only approved regions increase the batch's region count. Review validity and
annotation pixel storage are unchanged.

## Files

- `src/napari_label_assistant_tools/_widget.py`: full-source grid geometry/labels.
- `src/napari_label_assistant_tools/_simple_curation.py`: explicit Review guidance.
- `tests/test_simple_curation.py`: new regression reproducing the exact mode
  transition; strengthened 100k disk-backed test to select and approve through UI
  controls instead of invoking the review operation directly.

## Verification

The new reproduction failed before the correction because mouse review silently
switched off. It now passes, dispatching a real napari mouse event through the
viewer callback registry, checking full-mask grid extents, enabling approval, and
recording the approved region in Batch 2.

Initial focused grid/selection checks: **8 passed**.
Broader relevant suite: **102 passed**, 1,039 warnings, in 264.82 seconds.
It covered Simple curation, annotation batches, reviewed regions, grid display,
canvas component coordinates, grid layout/large shapes, coordinate contracts,
and semantic grid widgets. This includes the strengthened 100,000 × 100,000
disk-backed mask selection/approval test with bounded-read assertions.
`git diff --check` passed.

In the actual `napari-dev2` environment, pytest could not start GUI tests because
`pytest-qt`/the `qtbot` fixture is absent. A standalone offscreen Qt harness then
verified **Simple Batch 2 overview selection and approval** and **Advanced review
selection from overview** successfully, using only temporary synthetic data. No
dependencies were installed and no user workspace was modified.

Automated rendering uses Qt offscreen; final interactive appearance and input
should still be checked after saving the active session and restarting napari.
