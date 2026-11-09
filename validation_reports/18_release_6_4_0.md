# Release 6.4.0 validation

Date: **2026-11-08**. Release base: `5cd9614` (6.3.0).
Version declarations in `pyproject.toml`, the package, and the built wheel are
**6.4.0**. Minimum requirements remain Python 3.11 and napari 0.9.1.

## Full regression suite

**464 passed, 0 failures, 0 errors, 0 skipped, 637 warnings in 540.83 seconds.**
Warnings are retained in the output; this is not a warning-free dependency claim.

```bash
LD_PRELOAD=/home/wteox/miniforge3/envs/napari-dev/lib/libsqlite3.so.3.53.4 \
QT_QPA_PLATFORM=offscreen QT_API=pyside6 \
/home/wteox/miniforge3/envs/napari-dev/bin/python -m pytest -q --disable-warnings \
  --junitxml=/tmp/napari-label-assistant-6.4.0-tests.xml
```

Evidence: [test output](evidence/release_6_4_0/tests.log),
[JUnit XML](evidence/release_6_4_0/tests.xml),
and [wheel build](evidence/release_6_4_0/build.log).

Environment: Python 3.11, napari runtime **0.9.1**, NumPy 2.3.5, Zarr 3.1.6,
PySide6 6.11.2, pytest 9.0.2, pytest-qt 4.5.0, Linux ARM64. The napari-dev
environment has stale napari distribution metadata reporting 0.7.1rc2; imported
runtime code reports 0.9.1. The user's napari-dev2 environment reports 0.9.1 in
both places and was used for the complete read-only real-mask scan.
The versioned SQLite library is preloaded to avoid an environment loader mismatch.

## Covered contracts

- Edit & Confirm retains camera following, bounded crops, Paint/Erase,
  edit undo/redo, and workspace persistence. Review-tile sizes remain independent
  of editable crop and model patch sizes.
- Exact confirmed rectangles bound partial supervision. Missing unambiguous
  coverage references recover automatically. One-image saved pair configurations
  use spatial validation without bypassing source or prediction-review checks.
- Viewing-crop changes preserve unchanged prepared datasets; annotation edits
  invalidate affected approvals and datasets. Historical training use is separate.
- Entire-mask analysis streams full-resolution s0 independently of zoom, joins
  chunk seams, and retains global picking/locating through crop navigation.
- Whole-mask cleanup, copying, undo, destination preservation, and cancellation
  respect full-image coordinates and the existing bounded recovery journal.
- Review shutdown detaches callbacks without selecting layers or touching
  destroyed Qt controls.
- Existing binary, exclusive multiclass, overlapping multilabel, model management,
  inference, coordinate, dtype, and data-persistence regression tests pass.

## Packaging and manifest

Built without downloading dependencies:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /tmp/napari-label-assistant-6.4.0-dist .
npe2 validate src/napari_label_assistant_tools/napari.yaml
```

The plugin manifest is valid. ZIP integrity, wheel metadata version, embedded
package version, and inclusion of `napari.yaml` were checked.

- Wheel: `napari_label_assistant-6.4.0-py3-none-any.whl`
- SHA-256: `0e2f55d3cc0fd7c65ef32c8397a81a4b3055867b78fa34a7dd7e4d57315f2d64`
- `git diff --check`: passed.
- GitHub source publication is requested; publishing a PyPI distribution is a
  separate operation and was not performed.

## Real-mask validation and limits

The read-only complete scan of the **49,976 × 59,716** curated mask found **16
connected components** covering **22,211,355 foreground pixels** in **30.19
seconds** on this machine. See [entire-mask validation](17_entire_mask_analysis.md).
This is a dataset-specific measurement, not a hardware-independent performance
guarantee or a biological segmentation-accuracy evaluation. Original user images,
masks, and workspace JSON were not modified for validation.

See the [partial-training investigation](14_partial_training_example_review.md),
[workflow reorganization](15_workflow_clarity.md), and
[OME-Zarr search reproduction](16_ome_component_search.md) for development evidence.
Those reports retain their intermediate findings; the final release behavior and
complete-suite result are recorded here.
