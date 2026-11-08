# Release 6.3.0 validation

Date: **2026-11-07**. Release base: `93f30df` (5.3.0).

This release combines automatic camera-follow editing, bounded mask previews,
legacy-project loading migration, the consolidated annotation/training UI, and
streamed label tools. Version declarations in `pyproject.toml` and the package
are **6.3.0**. Minimum requirements are Python 3.11 and napari 0.9.1.

## Full regression suite

```bash
QT_QPA_PLATFORM=offscreen QT_API=pyside6 PYTHONPATH=src \
python -m pytest -q --disable-warnings --junitxml=/tmp/label-assistant-6.3.0-tests.xml
```

**Final result: 425 passed, 0 failed, 0 skipped, 495 warnings in 396.60 seconds
(6 minutes 36 seconds).** Warnings are retained in the test output rather than
treated as failures; this count does not claim warning-free dependencies.

Validation environment: Python **3.11.16**, napari **0.9.1**, PySide6/Qt
**6.11.2**, Zarr **3.1.6**, pytest **9.1.1**, Linux ARM64. The runner used
pytest-qt and the local editable source tree.

The initial full run found outdated assertions counting derived overview layers
as saved annotations, and an ROI-copy integration error in the temporary journal
adapter. The assertions now verify persisted annotation layers separately from
internal helpers. ROI-copy invalidation is attached to the destination entry
that owns the preview. Expanded regression cases copy into managed masks both
with and without pyramids, from different initial grid positions, and check the
full backing pixels, displayed crop, and derived label values.

## Existing large-project loading

A read-only validation used an existing six-layer project containing a
multiscale image, three Shapes layers, and two **73,595 × 94,479 uint32** masks.
Its JSON saved `mode: full` and `editing_strategy: tiled`.

Before migration, ordinary loading attempted a **27,812,728,020-byte** source
read for each mask. The diagnostic intercepted both reads before allocation;
it did not reproduce the entire reported multi-minute load or allocate those
arrays. Spatial Assistant overrides this old preference with bounded loading.

After automatic migration, the normal Label Assistant loader restored all six
layers without an override or skipped layers:

| Check | Observed result |
|---|---|
| First backend load after migration | approximately 0.61 seconds |
| Progress-dialog loader with the complete Label Assistant widget | approximately 0.81 seconds |
| Managed full-resolution edit buffers | two 1024 × 1024 crops |
| Largest allowed source read in the comparison | 6,794,200 bytes |
| Oversized read attempts after migration | zero |
| Mask write attempts | zero; writes were forbidden by the diagnostic |
| Project JSON | SHA-256 unchanged before/after loading |

These are local offscreen load-path measurements with warmed imports/storage,
not rendered-frame or hardware-independent latency guarantees. The original
project and mask stores were not rewritten. A later explicit Workspace Save
records the resolved bounded mode. Explicit Advanced session overrides remain
available and are covered by regression tests.

## Display and annotation contracts

- The preview planner inspects metadata only. For an 80k × 80k mask at a
  4096-pixel texture cap, napari receives 2500 × 2500 and 1250 × 1250 levels.
  A 2048-pixel limit uses the single smaller level safely. Levels also respect
  the 64 MiB byte budget, including wider integer IDs.
- Coordinate tests cover scale, translation, rotation, shear, additional affine
  transforms, cropped predictions, repeated workspace round trips, and source
  changes. Overview rebasing leaves full-resolution annotation coordinates and
  categorical IDs intact.
- Resolution locks, navigation, overview rebuilding, and legacy preview repair
  cannot expose the full mask through the bounded helper. Unrelated Labels and
  editable crop data are unchanged by the preview policy.
- Camera-follow tests cover zoom/pan, manual area selection, stroke waiting,
  stale-read rejection, failed reads, external-write conflicts, and saved follow
  settings. The default workflow runs with Advanced collapsed.
- Save/recovery tests cover changed-region writes, erasure, interrupted writes,
  conflict refusal, portable/save-as persistence, and data-only viewer clients.
- Annotation/training tests distinguish reviewed foreground/background,
  selected-object foreground, and unknown pixels; edits invalidate intersecting
  review cells and prepared datasets, including independent channel validity.

## Packaging

The release wheel builds locally without dependency downloads using:

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/release-dist .
```

Artifact: `napari_label_assistant-6.3.0-py3-none-any.whl`.
SHA-256: `8980c8fbfba2da681fbadff6bf83561cfe222f2b655d5ab56249b0bfc4b9fe3b`.
The package includes all seven added camera-follow, canvas, editing-backend,
local-control, bounded-overview, categorical-pyramid, and streamed-label-tool
modules. Source changes, tests, and documentation are committed; local diagnostic
scripts and real-project paths are excluded.

## Remaining scope limits

- Qt checks run offscreen. They validate canvas mapping, layer/control behavior,
  arrays, and state; they do not establish interactive OpenGL rendering speed.
- Creating/approving reviewed-cell coverage still uses explicit Local-area or
  Full-layer editing for the session. Grid completion is not training approval.
- Save As, portable packaging, and the first categorical-pyramid build can process
  complete source datasets. Fast opening does not make those operations free.
- Storage chunks, image rendering, I/O, and model operations still affect memory
  and responsiveness. External-edit conflict checks are not multi-user locking.
