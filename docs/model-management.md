# Model management and benchmarking

## Workflow

1. **Dataset:** Choose **Current pair** or **Multiple pairs → Manage pairs…**. Configure image/ground-truth associations and coverage per pair. Click **Prepare dataset** to index patches and validation partitions. Preparation is required for training; **Show patch preview** is optional. Multiple-pair datasets keep whole image/specimen groups together in validation. Review controls open on demand.
2. **Models:** Create a biological task such as *Myelin segmentation*. Leave **Foreground classes** empty for binary foreground/background, or enter semantic names such as `axon, myelin` for labels 1 and 2, with background 0. Define class meanings consistently across all pairs. Choose a new architecture or select a saved checkpoint.
3. **Train:** Start fresh training or continue the current model. A new model uses the configuration in Models; continuation always uses the current checkpoint's architecture. Reprepare patches when data or spatial settings change.
4. **Inference:** Predict the current crop, entire image, or a drawn box. Larger regions use overlapping tiles with progress and cancellation. Predictions require review and correction in Curated; class names persist in their workspace metadata.
5. **Advanced:** Adjust source context, model input size, augmentation, compute, batch size, optimizer settings, and optional plateau control.

A biological task describes the annotation meaning, not a pretrained biological model. ImageNet initialization supplies an encoder initialization; it does not provide trained myelin or Remak bundle segmentation weights. Tasks with trained models cannot change their target definition; create another task for a different target meaning. Task names can be updated.

The task's dataset associations use stable layer IDs. Training history records the task definition at the time of the run, so later renaming does not rewrite previous results. A default foreground task keeps the existing training workflow usable if no named biological task has been created.

## Patch preview and canvas placement

Annotation batches separate cumulative review progress from the annotation selection
used for a training run. Choose the named batch in the **Training** dropdown in
Labels, Simple Progress & Dataset, or Dataset; then prepare it and continue the
selected checkpoint. Batch buttons and the colored display controls let you
inspect review history without changing that training selection. Batch identity,
membership revision, and bounds are saved in run provenance. Changing a batch does
not create a model or change its weights. New checkpoint names include the batch name
for a single active image/Labels pair. Earlier runs keep their saved provenance.

Preparation does not add preview layers or move the canvas. **Show patch preview** opens a separate contact sheet to the right of the source image and focuses it. Placement accounts for the source image's world extent and pixel scale, including rotated or sheared images. The gallery is a rearrangement of patches, not an overlay at their source coordinates.

- **Crop images:** Normalized model input patches.
- **Crop ground truth:** Aligned foreground masks.
- **Crop coordinates:** Source coordinates and split labels; cyan frames indicate training, orange indicate validation.
- **Ignored pixels:** Areas excluded from supervision, shown when a patch has unreviewed pixels or padding.

Double-click a tile to hide the gallery and focus its original source region. **Return to image** hides the preview and restores the previous canvas view. Prepared training data is retained. Editing annotations or changing dataset settings invalidates the preparation and hides outdated previews.

## Architectures

- **Custom small U-Net:** The existing five-level GroupNorm model, with widths 4, 8, 12, 16, 32, or 64; the default remains 8. The selector displays the parameter count and estimated float32 weight storage for the current output classes/channels without allocating a model. Binary widths 8, 16, 32, and 64 have approximately 0.49, 1.97, 7.85, and 31.39 million parameters. Wider models require more computation and memory; reduce batch size if needed. Changing width applies to **Train new model**. **Continue training** retains the loaded checkpoint's width and weights.
- **Standard U-Net:** Four levels, BatchNorm, transpose-convolution upsampling, and widths 16, 32, or 64.
- **SMP:** U-Net, U-Net++, or DeepLabV3+, with selectable encoders. Common encoder choices appear in the dropdown; another supported encoder name can be typed.

From the repository directory, install dependencies in the Python environment running napari:

```bash
python -m pip install -e '.[train]'
python -m pip install -e '.[smp]'
```

### Installing and checking SMP

Activate the environment used to launch napari first (for example, `conda activate napari-dev`, if that is your napari environment). Install SMP directly and verify its import:

```bash
python -c "import sys; print(sys.executable)"
python -m pip install segmentation-models-pytorch
python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"
```

Use `python -m pip` so installation targets that Python interpreter. For a source checkout, `python -m pip install -e '.[smp]'` from the repository directory also installs the optional dependencies. The direct SMP command works independently of whether a published plugin release provides the `[smp]` extra.

After the import check succeeds, close napari and relaunch it from the same environment:

```bash
python -m napari
```

Then select SMP in **Training → Models** and retry training.

### Troubleshooting SMP import failures

A training message about importing SMP means model construction stopped; it does not establish whether a previous pip installation succeeded.

| Message | Meaning and next step |
|---|---|
| **SMP is not installed** | The napari interpreter cannot find `segmentation_models_pytorch`. Use the exact Python executable shown in the error with `-m pip install segmentation-models-pytorch`. |
| **SMP dependency import failed** | Importing SMP encountered a missing dependency. The message preserves its name and the underlying error. Run the import check above to capture the full traceback. |
| **SMP is present but could not be imported** | The package was found, but initialization failed. Inspect the preserved error, including any PyTorch/torchvision compatibility or shared-library failure. |

For a dependency failure, capture the full import traceback and inspect installed versions in the same environment:

```bash
python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"
python -m pip show segmentation-models-pytorch torch torchvision timm
python -m pip check
```

Keep the traceback and Python executable path when reporting the problem. A successful pip installation or `pip check` alone does not prove that compiled dependencies can import. Resolve the reported dependency error before retrying training; reinstalling the plugin alone may not fix it.

SMP is optional. Missing providers do not prevent access to workspace annotations or stored model summaries. Constructing a new pretrained encoder may download weights; loading a saved checkpoint reconstructs the architecture without fetching pretrained weights.

Providers use one grayscale input channel. Binary tasks use one foreground output logit; multiclass tasks use one logit per semantic class including background. Multilabel tasks use one independent sigmoid logit per named binary target mask, without a background output. `ModelSpec.output_channels` records the head size; the task/checkpoint also stores whether outputs compete or are independent. Training and inference use per-patch 1st–99th percentile normalization to [0, 1], recorded in the model specification. This shared normalization also applies to pretrained encoders; it is not ImageNet RGB normalization. Model input size must be a multiple of 32 for SMP. Single-patch training batches use running BatchNorm statistics, allowing pooled heads and small final batches to remain usable.

## Checkpoints and history

The library groups trained models by biological task and displays configuration, parameter count, metrics, source definition, run history, and save status. Select the latest checkpoint or an earlier run. **Use configuration for a new model** copies architecture settings without copying trained weights. **Select this model** makes the chosen checkpoint current for continuation or inference.

Workspace Save persists changed model weights, including separate revisions for previous runs. **Export current model** writes a standalone `.pt` file and readable history, metric CSV, and learning/loss SVG files. Weights are held in memory until one of these saves occurs; training completion itself does not save to disk.

New workspace checkpoints and export suggestions use readable names, for example:

```text
Remak_EM08_unet-w8_in1024_run01_20261118_7a72a518.pt
```

The fields identify the model, source image, architecture/width (or encoder), model input size, run number, recorded UTC training date, and a short revision ID. Names are sanitized and shortened for filenames. Missing historical dates are shown as `undated`; they are not reconstructed. Full model and revision IDs remain in workspace metadata. History, metrics, and chart files share the checkpoint stem. Export filenames remain editable. Copying an external checkpoint into the workspace also uses a readable name, without moving the original.

Existing saved files are not renamed. Complete-package Save As preserves existing `.pt` filenames; legacy exports with another extension receive a readable `.pt` name in the new package. Collisions receive an additional unique suffix. Renaming a model in the library changes its display name and future save suggestions, not already-saved files.

Checkpoints include weights, architecture specification, preprocessing, task/model identity, library versions, configuration, and complete training history. Version-1 checkpoints are recognized as the original custom U-Net. Continue training starts a fresh optimizer from stored weights; this is not exact recovery of an interrupted optimizer/scheduler state.

With **Dataset → Quick trial — smaller validation set**, checkpoint selection uses a fixed subset of the reserved validation pool. These metrics are labeled **Trial validation** in the model library and reports. **Train → Validation → Evaluate full validation set** evaluates the retained weights on the complete pool without training or replacing the trial-selection scores. The additional evaluation is saved on the same run, with separate metrics, source coordinates and dataset provenance. It does not create a training round or add optimizer steps. Save the workspace or export the model to preserve it. Scores from trial and full validation should not be treated as the same evaluation protocol; neither is an independent test-image score.

## Controlled comparisons

Prepare a coverage dataset, add at least two configurations in Models, then choose **Run comparison**. The candidates share source patches, reviewed supervision, spatial partitions, input resolution, augmentation, learning rate, and training budget. Each candidate and each k-fold model starts fresh. Candidates run sequentially. The current prediction model is retained until a result is selected.

Only eligible source patches are fingerprinted, in bounded reads, before the comparison. Image pixel changes and indexed annotation changes abort the comparison rather than silently comparing different inputs. A report records the dataset fingerprint, settings, parameter counts, training time, metrics, and fold details. Export the comparison table as CSV; Workspace Save also stores its full JSON report.

Holdout results are validation-selected checkpoint scores. K-fold results show means and sample standard deviations across folds. These are not independent test-image results. Separate reports can use different data and should not be ranked as though they shared an evaluation protocol. This release compares explicit configurations; automatic parameter sweeps and nnU-Net integration are deferred.

## Workspace storage

Workspace format 2 extends the existing manifest rather than replacing layer storage. It records biological tasks, configuration IDs, dataset summaries, model/run summaries, selected task/checkpoint, and training controls. Full coordinates and curves remain in model/history artifacts; full comparison reports are sidecars, keeping large-image manifests manageable.

Complete-package Save As includes images, masks, checkpoints, and comparison reports using relative references. Transfer the manifest and its entire data folder together. Workspace Open restores summaries and controls; network weights are loaded when selected. Existing version-1 workspaces remain readable. Training, prediction, and model export must finish or be cancelled before workspace Save/Open.

## Custom provider development

Register trusted Python code before creating the Training widget:

```python
from napari_label_assistant_tools._model_providers import register_model_provider

register_model_provider(
    "my_unet",
    lambda spec, initialize: MyUNet(base_channels=spec.width, output_channels=spec.output_channels),
)
```

The builder receives `ModelSpec` and `initialize`, and must return a PyTorch `nn.Module` accepting `N × 1 × H × W` inputs and returning raw logits of shape `N × spec.output_channels × H × W`. Adapt constructor parameter names to your model implementation. An optional `validator(spec)` can reject unsupported settings. The registered provider appears in Models; the architecture name is editable for custom providers. Extend the provider's trusted code for additional custom options. Provider code must be registered again before reloading its checkpoints; executable code is never imported from a checkpoint.

The SMP adapter follows the [official model constructor interface](https://smp.readthedocs.io/en/latest/models.html).
