# Quick trial validation budgets

Implemented locally on 2026-11-18. No user workspace, annotation store, or running training session was modified.

## Behavior

- Dataset → Training patch selection now offers **Quick trial — smaller validation set**, off by default. Enabling it switches All training patches to the existing percentage setting; explicit percentage/count choices are preserved.
- Validation percentage defaults to 10% and is independent of the training budget. With 5,257 training and 1,813 reserved validation patches, 10% of each gives 526 unique training patches and 182 validation patches. Extra foreground visits retain their existing behavior.
- Selection follows the original spatial/image holdout. Omitted validation patches remain reserved and never become training patches. Sampling uses metadata only, shares all source arrays/masks, and spreads foreground/background selections across reserved image/spatial groups.
- The validation subset follows the split seed and stays fixed across epochs, training amounts, and training subset seeds. Required binary background/foreground and multiclass/multilabel coverage is checked; insufficient subsets produce actionable errors rather than misleading scores.
- Quick trial labels appear in progress, model details/library, curves, and metric/comparison CSVs. Trial/full evaluation provenance remains distinct.
- Train → Validation → **Evaluate full validation set** evaluates the retained weights once on the entire reserved pool. It uses the existing masked validation implementation, makes no optimizer updates, does not reselect weights, and records results separately on the same run. Cancellation preserves model state.
- Full-evaluation metrics, coordinates, source associations, timestamps, and dataset fingerprint are saved with the model/history. Workspace summaries omit the nested coordinate list. Save/export preserves new evaluation metadata; existing saved checkpoints remain intact.
- Existing workspaces without the new settings use full validation. Initial preparation still indexes the eligible pool. K-fold and final all-reviewed workflows retain their full-index behavior.

## Files

- `src/napari_label_assistant_tools/_patch_subset.py`
- `src/napari_label_assistant_tools/_coverage_train.py`
- `src/napari_label_assistant_tools/_quick_train_widget.py`
- `src/napari_label_assistant_tools/_model_project.py`
- `src/napari_label_assistant_tools/_model_comparison.py`
- `src/napari_label_assistant_tools/_model_library.py`
- `src/napari_label_assistant_tools/_models_widget.py`
- `src/napari_label_assistant_tools/_training_report.py`
- `tests/test_patch_subset.py`
- `tests/test_patch_subset_widget.py`
- `README.md`, `docs/TRAINING.md`, `docs/model-management.md`

## Verification

**241 passed**, 65 warnings, 79.19 seconds, using offscreen Qt and CPU model training/evaluation. Covered patch subsets, training engines and GUI, model/project persistence, readable checkpoint names, wider channels, multiclass/multilabel training, dataset selection, and charts.

```bash
QT_QPA_PLATFORM=offscreen \
LD_PRELOAD=/home/wteox/miniforge3/envs/napari-dev/lib/libsqlite3.so.3.53.4 \
/home/wteox/miniforge3/envs/napari-dev/bin/python -m pytest -q --disable-warnings \
  tests/test_patch_subset.py tests/test_patch_subset_widget.py \
  tests/test_coverage_train.py tests/test_quick_train.py tests/test_quick_train_widget.py \
  tests/test_model_management.py tests/test_model_files.py tests/test_model_library.py \
  tests/test_model_naming.py tests/test_channel_widths.py \
  tests/test_multiclass_training.py tests/test_multilabel_training.py \
  tests/test_model_selection_dataset.py tests/test_learning_chart.py --tb=short
```

New regressions cover the reported 526/182 budgets, fixed validation under training changes, image/spatial diversity, missing class/channel/background handling, unchanged weights and training steps, cancellation, history/CSV scopes, saving additional evaluations without a new training round, workspace settings, and full validation using bounded 32-pixel reads from a virtual 100,000 × 100,000 source that forbids whole-array materialization.

`git diff --check` passed. GPU timing and a complete interactive desktop training session were not benchmarked. Synthetic Qt preview: `/tmp/quick-trial-selection.png`. The new setting takes effect in a new training run after napari is restarted.
