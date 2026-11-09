# Checked training tiles and QC mask display

Validated locally on 2026-11-07 using Python 3.11, napari 0.9.1, PySide6,
and the `napari-dev2` environment with Qt offscreen rendering. All masks and
workspaces used for validation were disposable test data.

## Reproduced causes and changes

- Component-table selection did not add whole tiles to training. QC now has
  separate training-tile controls. Background clicks select the loaded tile;
  additive clicks retain prior selections. Explicit confirmation approves
  foreground and background while retaining original annotation IDs.
- Dataset concealed the annotation-selection path and could briefly lack the
  new coverage layer. Visible tile-selection/confirmation actions and synchronous
  coverage refresh make the next Prepare dataset action use the confirmed tiles.
- The camera-follow controller checked the visibility of Local Editing controls,
  which are hidden in QC. It now checks the Labels panel. QC retains the current
  detailed crop across layout changes and resumes following on camera navigation.
- An unused stale component library no longer blocks tile-only preparation.
  Selected individual objects retain their library freshness checks.

## Regression evidence

- Dataset → canvas selection → two background clicks in distinct checked tiles
  → confirmation → Prepare dataset succeeds at the default 512-pixel patch size.
  It indexes two patches in independent training/validation blocks, accounts for
  both checked tiles including background, and excludes unreviewed pixels.
- Duplicate/replacement/additive tile selection, overview-click rejection, and
  actionable missing-coverage messages are covered.
- QC tab/layout changes preserve crop pixels, bounds, visibility, camera center,
  and zoom. Camera panning still loads bounded full-resolution crops while the
  editing controls are hidden.
- Existing confirmation tests verify journaled bounded writes on an 80,000 ×
  80,000 logical mask, transformed coordinates, save/reload, and invalidation
  after later edits. No full-mask display allocation is used for confirmation.
- Targeted tile-workflow and camera-follow run: **21 passed**.
- Full regression run: **438 passed**, with one obsolete assertion expecting
  the previous missing-coverage instructions. Updated that assertion to check
  the new Dataset instructions. The training-widget, tile-workflow, and camera
  rerun passed **45 tests**. After the final outline wording change, all **13**
  tile-workflow tests passed again, including exclusion of an unselected third tile.
- Final `git diff --check` and Ruff fatal/syntax checks passed (existing unused
  imports excluded from the Ruff check).

The offscreen checks validate full-resolution buffers and display state; they
do not benchmark GPU rendering or the user's storage hardware. Preparation keeps
independent spatial validation: small or sparsely annotated selections may need
more checked tiles or a smaller source patch size.
