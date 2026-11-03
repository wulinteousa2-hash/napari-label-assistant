# Training large images

The Training tab supports binary, exclusive multiclass, and overlapping
multilabel segmentation. With **One mask — binary / exclusive classes** and
**Foreground classes** empty, all nonzero ground-truth labels are foreground;
individual object IDs are not learned separately. Define names such as
`axon, myelin` in **Models** for semantic IDs `0 = background`, `1 = axon`,
`2 = myelin`. Up to 254 foreground classes are supported. Class definitions
persist per task/checkpoint; trained tasks require a new task to change meanings.

For nested features, choose **Separate masks — overlapping channels** in Models.
Open **Dataset → Manage channels…** to name channels (for example
`cell, mitochondria, nucleus`) and assign separate aligned mask layers. The
dialog opens even when no names have been entered in Models. For one image,
**Use mask layers as channels** fills channel names and target assignments from
the loaded Labels layers and includes the image. Names are descriptive; masks
still supply numeric supervision. The whole-cell mask includes organelles;
organelle masks overlap it.

Use **Foreground pixels → Binary 0/1** for binary masks. **Any nonzero ID** treats
all nonzero instance IDs within that mask as one feature during training and
preserves source pixels. The one-image shortcut selects this mode. If a selected
task already has trained models with different meanings, applying the channel
setup creates another task rather than changing its old models.
Multilabel output uses one independent sigmoid channel per name, without a
separate background channel. Do not merge these masks into exclusive class IDs.

Every channel has its own supervision coverage. **Complete** means its positive
and negative pixels have all been checked. **Partial** uses that target's
associated reviewed-region mask. **Unknown / not annotated** excludes that
channel on that image. Missing masks never become negative examples implicitly.
Use reviewed cells rather than selected-component foreground in this mode.
**All patches** streams the selected channels; sampled single-mask preparation
is unavailable. One image uses spatial validation; multiple images reserve whole
image/specimen groups. Each channel needs known positive and negative examples
in both training and validation.

Multilabel validation reports per-channel Dice/IoU and macro averages. Prediction
applies the adjustable threshold independently and adds one binary Labels layer
per channel. It preserves overlaps without forcing biological containment or
inferring cell-to-organelle instance ownership. Class/channel order and output
mode are stored in checkpoints, with mask assignments saved in the workspace.

Training has five pages: **Dataset**, **Models**, **Train**, **Inference**, and **Advanced**.
Choose annotations and prepare the dataset in Dataset; patch preview is optional. Select a biological task and
model configuration or checkpoint in Models, then open Train to start or
continue a model. Detailed metrics/history expand on demand. Local Editing keeps
the cell selection and mask correction tools; Review & QC keeps component sorting,
copying, and deletion. Switching pages preserves the selected cell and annotations.

## Optional model dependencies

Install SMP in the environment running napari with
`python -m pip install segmentation-models-pytorch`. Verify the import and restart
napari from the same environment before training. See
[SMP installation and troubleshooting](model-management.md#installing-and-checking-smp)
for commands and how to distinguish missing SMP from a broken dependency import.

## Two ways to include annotations

These annotation workflows apply to **Current pair** mode. For several images,
choose **Dataset sources → Multiple pairs → Manage pairs…**. Select pairs,
configure coverage and reviewed masks separately, and confirm corrected model
predictions before using them as ground truth. Pair definitions persist by layer
ID in the workspace. The dialog's **Review selected pair** opens the regular
review controls; those controls stay collapsed otherwise.

Multiple-pair preparation streams eligible patches across all included images.
Holdout and k-fold validation reserve entire image/specimen groups. Give related
images the same split group, and use comparable image scales. Each split needs
targets and background; semantic classes need examples in both splits. The
single-pair workflow continues to reserve spatial regions within its image.

Keep **Partial — reviewed cells + selected objects** when Curated contains only some targets.
Zero pixels may contain missing objects, so they must not automatically become
background. Partial mode uses **All patches (recommended)**.

### Dataset → Reviewed cells: review a region

1. In **Training → Dataset**, choose the image and Curated Labels layer.
2. In **Labels → Local Editing**, select a review cell on the canvas.
3. Correct or add Curated masks through **Editable Area**. Check every target
   and its background inside the cell.
4. Return to **Training → Dataset → Reviewed cells** and choose
   **Finish review — include this cell in training**. Coverage is created
   automatically; no manual mask creation is needed.
5. Review several independent regions with both targets and genuine background.

For cells you have already checked, enable **Mark already-checked cells with
clicks** under **Dataset → Reviewed cells**. This activates canvas review mode
and selects **Click to mark a cell reviewed** automatically. Click cells on the
canvas to include them; turn the option off to return to editing. Disabling the
address grid also turns off click marking.

### Dataset → Selected objects: select components

1. In **Training → Dataset**, choose Curated as ground truth and open
   **Selected objects**.
2. Click **Build / update library**. The first scan is a background operation
   with progress and Cancel; subsequent local edits update affected components.
3. Check **Train** beside objects to include. Clicking another column locates
   that object. The table contains ID, label pixels, and bounds; no Euler
   calculation is performed for the library.
4. Save the workspace once to include the library asset. Later checkbox changes
   are written immediately to that library.
5. Keep background confirmed under **Reviewed cells**. A checked component
   supplies foreground, not confirmation of its surroundings.

Both methods feed the same training dataset. Shared patches and pixels are
counted once. Unchecked surroundings and invalidated review cells are ignored.
Objects sharing one Curated label value can have different library IDs. Library
IDs are separate from Curated label values; building the library never relabels
the source mask.

### Continue editing and reopen projects

Workspace Save/Open preserves the library, stable IDs, training checkboxes,
review coverage, and source associations. Save As and portable exports copy the
library so the new project has independent selections. Existing workspaces
without a library remain supported.

- Unchanged objects retain their IDs and selections.
- New masks receive new IDs and start unchecked.
- Boundary edits retain an ID when pixel overlap is unambiguous, but clear the
  selection and require review again.
- Splits and merges of previously indexed objects retire their old IDs and
  allocate new IDs. Deleted IDs are never reused.
- Editing a reviewed cell invalidates that cell's background confirmation.
  Finish its review again before treating its background as training data.

Editable Area writes and QC copy/delete operations notify the library. If masks
were changed externally, click **Build / update library** to check the saved
index against source pixels. Training refuses stale library data. Cancel stops
automatic scan retries until Build / update is requested again.

Workspace saves include managed model checkpoints and training history.
After restarting napari, open the workspace and select its checkpoint in Model
Library before continuing. **Export current model…** and **Load model…** also
support standalone checkpoint files.

No reviewed mask is needed when every target is annotated across the image;
choose **Complete — whole image** explicitly in that case.

## Advanced training settings

- **Batch size:** 1–128 patches per optimizer update. Larger batches and model
  inputs require more device memory. Start with the existing default of 2;
  increase it when your hardware has room.
- **Source patch size:** up to 4096 × 4096 pixels. This changes the image context,
  not the model resolution. Model inputs now include 1024 and 2048; a 4096 source
  patch with 512 input is downsampled eightfold and may lose fine detail.
  Changing patch/input sizes requires Preview training patches again.
- **Augmentation:** independently enable random 90-degree rotations, random flips,
  and brightness/contrast variation. The defaults preserve the previous behavior:
  contrast ±10%, brightness ±3%. Geometry transforms apply equally to image,
  ground truth, and valid-pixel masks. Validation and inference are not augmented.
- **Reduce learning rate, then early stop:** optional and available for All
  patches. Monitor mean validation patch loss (masked BCE + Dice) each epoch.
  Choose the initial patience, learning-rate multiplier, second patience, and
  minimum absolute loss decrease that counts as improvement.

For example, set maximum epochs to 50, enable plateau control, choose **10 epochs**
before reduction, multiplier **0.5**, and **10 epochs** before stopping. After ten
consecutive epochs without meaningful validation-loss improvement, the learning
rate halves once. After ten further consecutive non-improving epochs, training
stops. Improvement resets the current patience counter; the learning rate remains
reduced. The maximum epoch limit still applies. The reduced learning rate has a
floor of 1e-7 and never increases above its previous value.

The best **validation Dice** checkpoint is retained, even if early stopping is
triggered by validation loss. Model history records the settings, validation
loss, per-epoch learning rates, reduction events, actual completed epochs, and
stop reason. Model loading restores augmentation and plateau controls. Sampled
training uses update counts rather than epochs, so the plateau controls are
unavailable there.

## Coverage training

**All patches (recommended)** is the coverage workflow for large images.

- **Source patch size** is the area read from the original image.
- **Model input size** is the resolution fed into the model after resizing.
  Matching sizes, such as 512 × 512 for both, preserve spatial resolution.
- **Preview training patches** scans annotation patches and stores coordinates and
  counts. It does not load the complete image into memory.
- The gallery contains representative examples. It is not the whole training
  dataset. The dataset summary reports total patches, train/validation counts,
  and reviewed area as a percentage of the image.
- Each **epoch** visits every eligible training patch in shuffled order. The
  **batch size** is the number of patches processed in one model update.
- **Additional target patches** adds repeat visits to patches containing targets;
  these visits supplement, rather than replace, coverage of the full training set.
- Validation uses reserved spatial blocks and does not update model weights.
  Patch reads, loss, and validation exclude image-edge padding and unknown pixels.

To protect partial annotations when downsampling, a model pixel is supervised
only if every source pixel contributing to it is reviewed. If reviewed pixels
or targets disappear at the selected input size, preparation reports an error;
use matching patch/input sizes or fully review a larger region.

After indexing, set epochs, batch size, and compute device, then **Train new model**
or **Continue training**. You do not need to change the crop seed
or rebuild the index between epochs or continuation rounds. Changing the shuffle
seed changes patch order without requiring reindexing. Apply annotation corrections
and index again when the image, ground truth, reviewed mask, or patch size changes.
The same source and patch size retain reserved validation blocks within the panel;
coverage models save those block assignments and restore them when loaded with a
matching image name, image dimensions, and patch size. Loading a model requires
preparing the dataset again. Known sampled-training regions from the same image
name are excluded when selecting new validation blocks. If no independent reviewed
target blocks remain, review new regions or start a new model.

The panel calculates updates per epoch and total updates before training.
Progress reports epochs, unique training patches visited, model updates, and
validation progress. The checkpoint with the best validation Dice is retained.
Continuation can retain the starting weights if no epoch improves validation.
A continuation round uses a fresh optimizer. Cancel preserves the previous model.

## Saving and reviewing the model

Training updates the model **in memory**. Save the workspace to persist managed
checkpoints, training history, and pair definitions. **Export current model…**
writes a standalone checkpoint and companion reports. Model Library selects
saved checkpoints for continuation or inference. The panel reports whether the
model was updated in memory, loaded, or saved.

In **Inference**, choose **Current area**, **Entire image**, or **Drawn box**.
The latter modes blend overlapping tiles with progress and Cancel. Predictions
create a separate Labels layer; review and correct them before training on them.
Multiclass predictions choose the most probable class including background, and
the binary threshold is disabled. Validation within one image
does not establish performance on a different image or specimen. Keep a separate
image for final evaluation when possible.

## Sampled crops

**Sample patches** retains the smaller, fixed-crop workflow for
complete annotations. It trains by steps and reuses the prepared crop sample.
Change the sampling seed and prepare again to use different sampled locations.
Partial annotations require coverage training with a reviewed-region mask.

### Review Curated cells for training

1. Select the SAM3 **Curated** source layer and the corresponding image under
   **Reviewed areas for training**. Set the review-cell height and width; these
   share the address-grid settings in Review/QC.
2. Enable **Select a review cell on the canvas**, then click a cell. Its yellow
   outline and address identify the whole region that needs review.
3. Turn cell selection off while editing Curated. Correct false positives and
   add every missing target throughout the yellow cell boundary. The editable
   area may cover only part of that cell; inspect and edit the remaining parts
   before declaring the cell complete. Check objects crossing cell boundaries.
4. Click **Finish review — include this cell in training**. Pending edits are
   applied first. The first use asks where to store the disk-backed review mask;
   no separate Create mask action is needed. Green outlines show reviewed areas.
5. In Training choose **Partial — reviewed cells + selected objects**. This automatically
   uses **All patches (recommended)**; sampled-patch training is available only
   for complete annotations. The associated
   **Curated** ground truth and **Reviewed regions** coverage are selected
   automatically. Click **Preview training patches**, then train.

**Remove reviewed status** clears the selected cell's coverage without deleting
Curated objects. Marking, unmarking, or editing sources requires dataset reindexing.
Changing grid size clears the selection but preserves previously marked pixels.

**Reviewed annotations** is a read-only binary copy: objects from reviewed cells
have value 1, background 0. It is updated when a cell is marked or remarked, and
is excluded from Training ground-truth choices. Curated retains its object IDs.
**Reviewed regions** records entire trusted cells including background; Training
uses this coverage to ignore unreviewed pixels. Green outlines are viewing helpers.
Workspace saves preserve both data layers and their source associations.

**Mark already-checked cells with clicks** activates direct click-to-mark
and click-to-remove actions. Use these only after the cells have been checked.
Normal review selects a cell first and finishes its review explicitly.

## Library bounds and training patches

Source patches have a fixed size, such as 512 × 512. Bounds identify candidate
patches; exact foreground and usable pixel counts come from their mask pixels.
A small object can share a patch with other objects; a large object can occupy
several patches. The gallery is a bounded sample, not the full training set.

Every eligible training patch is visited each epoch. Larger objects can
contribute more patches; the library does not assign equal weight to each
object. Additional target-patch visits oversample foreground patches. Patches
belonging to one component stay together in training or validation, including
components crossing spatial blocks. Independent target/background groups are
required for both splits.

### Validation and learning curves

In **Training → Train**, the **Validation** box sits below **Train and evaluate**.
Choose validation settings before clicking Prepare dataset. Changing the
method, percentage, fold count or split seed requires a new dataset index.
These controls apply to the **All patches** workflow; sampled experiments retain
their existing fixed spatial holdout.

- **Spatial holdout (recommended):** reserve 5–50% of spatial groups (default 25%).
  Patch counts and the actual patch percentage are displayed because groups can
  have different sizes. Continue training reuses the reserved split; regions already
  used to train the current model cannot become independent validation regions.
- **Spatial k-fold (advanced):** choose 2–10 folds (default 5). Each independent
  model starts with fresh weights, trains on the other folds, and validates on its
  own held-out groups. Continue training is disabled for this mode. At least one
  target-containing group per fold is required; training and validation must both
  include targets and trusted background. Groups linked by library components
  stay together across splits.

**Preview validation regions on canvas** draws the indexed patches currently in
view. Holdout uses orange for validation and cyan for training. K-fold uses a
separate color per fold, listed in the status message. Pan/zoom on the source image
and refresh this preview as needed; helpers are excluded from workspace saves.
The optional representative image gallery shows the first fold's split after k-fold indexing. Show patch preview opens it beside the image; Return to image hides it without discarding the prepared dataset.

The live learning chart updates after each evaluated epoch with Dice, IoU,
precision and recall. K-fold results show each fold's scores and best epoch,
plus mean and sample standard deviation across folds. Best checkpoints are selected
using their validation fold, so these are not independent test-set measurements.

**Save fold models and validation report…** creates a new result folder with each
fold's weights and history, exact spatial/patch assignments, aggregate scores, and
an SVG learning chart. Workspace saves also persist registered fold checkpoints;
the current prediction model is retained throughout evaluation.

After k-fold evaluation, **Train final model on all reviewed regions** starts a
fresh model and visits all eligible patches. It has no held-out validation data;
validation plateau stopping is therefore inactive. Its history retains the earlier
cross-validation report, and its weights are retained from the last requested epoch.
Save the workspace or use **Export current model…** to persist the final model.
Standalone model exports also write a
`.learning.svg` beside the model and `.history.json`.

### Validation loss, CSV, and activity log

Use **Chart → Validation loss** to see held-out loss on its own numeric scale
(lower is better), or **Validation scores** for Dice, IoU, precision and recall.
The last-batch training loss shown in progress messages is different from loss
measured across the held-out validation patches after each evaluated epoch.

**Export metrics CSV…** exports the observed checkpoints, including epoch, step,
fold/series, validation loss, Dice, IoU, precision and recall. The optional
`training_loss_last_batch` column is explicitly the final training batch's loss,
not an epoch average. Missing legacy fields are empty. During training, export
captures the checkpoints available at that moment; export again after completion
for the full run. Model saves also write `.metrics.csv` and `.loss.svg`; fold-result
exports include a combined CSV and loss chart.

The fixed-height progress summary and **Activity log** keep the layout stable
between training and validation updates. The log retains up to 2,000 recent
messages, can be scrolled without being forced back to the bottom, and can be
collapsed. **Train new model** and **Continue training** share one action row.
