# Release 7.0.0 validation

Date: **2026-11-19**. Release base: `c5c02e3` (6.4.0).

This major version includes the accumulated Simple/Advanced annotation workflow,
recoverable tile progress, annotation batches and history colors, component
screening, reproducible training-patch subsets, Quick trial validation, wider
custom U-Nets, readable checkpoint naming, and workspace/overview fixes. See the
[changelog](../CHANGELOG.md#700---2026-11-19) and feature reports below.

## Version and packaging

The version in `pyproject.toml`, package `__version__`, and wheel metadata is
**7.0.0**. Python 3.11+ and napari 0.9.1+ requirements are unchanged. Existing
workspace format version 2, source stores, and supported model formats remain
compatible. Earlier review dates and historical annotation pixels are not
reconstructed.

Built locally without downloading dependencies:

```bash
/home/wteox/miniforge3/envs/napari-dev/bin/python -m pip wheel \
  --no-deps --no-build-isolation --no-index \
  --wheel-dir /tmp/napari-label-assistant-7.0.0-dist .
```

- Wheel: `napari_label_assistant-7.0.0-py3-none-any.whl`.
- SHA-256: `771511bd1a894e50a5197430facb9c9746957e470b9c9696d69576fc80d66cd9`.
- ZIP integrity passed; all **49 Python source modules** and `napari.yaml` are
  included. The manifest contains both Simple and Advanced entry points.
- `npe2 validate src/napari_label_assistant_tools/napari.yaml`: passed.
- `git diff --check`: passed.

Evidence: [build log](evidence/release_7_0_0/build.log),
[package inspection](evidence/release_7_0_0/package.json).

## Full regression suite

**631 passed, 0 failures, 0 errors, 0 skipped**, with 1,918 warnings in
**977.19 seconds (16:17)**. Warnings were retained in the summary; this is not a
warning-free dependency claim. All repository tests ran with Qt offscreen in the
`napari-dev` environment.

```bash
QT_QPA_PLATFORM=offscreen QT_API=pyside6 \
LD_PRELOAD=/home/wteox/miniforge3/envs/napari-dev/lib/libsqlite3.so.3.53.4 \
/home/wteox/miniforge3/envs/napari-dev/bin/python -m pytest -q \
  --disable-warnings --junitxml=/tmp/napari-label-assistant-7.0.0-tests.xml
```

Evidence: [test output](evidence/release_7_0_0/tests.log),
[JUnit XML](evidence/release_7_0_0/tests.xml).

## Feature validation

- [Simple curation and progress recovery](18_simple_curation.md).
- [Training patch subsets](19_training_patch_subsets.md).
- [Quick trial validation](20_quick_trial_validation.md).
- [Annotation batch metadata and training selection](21_annotation_batches.md).
- [Overview review-selection repair](22_batch_overview_selection.md).
- [Persistent batch colors and independent display](23_batch_history_colors.md).

The final feature-level checks included a 10,000-tile coordinate history, a sparse
100,000 × 100,000 disk-backed mask with bounded-read assertions, workspace
save/reopen, prepared-dataset preservation, validation protection, and real CPU
training. Standalone Qt checks in `napari-dev2` verified Simple and Advanced
overview selection, batch colors, All/Hide, and stale-review warnings.

## Scope and publication

Automated GUI checks use Qt offscreen. They do not replace interactive microscopy
annotation review or establish biological model accuracy, GPU throughput, or a
full 100k dense-image performance benchmark. User workspaces and annotations were
not modified for release validation.

The system-design diagram deliberately documents the earlier 6.4.0 baseline;
the release notes and current workflow documentation describe 7.0.0. Existing
intermediate reports retain their original test counts and findings.

GitHub source publication is requested on `main`. PyPI publication and GitHub
release assets are separate operations and are not performed by this source push.
