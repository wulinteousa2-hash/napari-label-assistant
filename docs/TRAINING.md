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
Partial channels combine their own reviewed cells and selected-component foreground;
neither another channel's coverage nor its unannotated zeros become negatives.
Edits revoke approval for the affected channel's reviewed regions.
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
continue a model. Detailed metrics/history expand on demand. Edit & Confirm keeps
the cell selection and mask correction tools; Mask Analysis & Cleanup keeps component sorting,
copying, and deletion. Switching pages preserves the selected cell and annotations.

## Annotation batches

Simple **3. Progress & Dataset**, Advanced **Labels → Local Editing → Confirm
tiles for dataset**, and **Training → Dataset** show the same batch buttons.
The active button is highlighted. **+ New batch** creates the next numbered,
empty workset and selects that named batch in the **Training** dropdown. New approvals join
that workset, with UTC review-event timestamps. Batch creation, selection, and
training never automatically approve annotation pixels.

Existing confirmed-region membership is retained as Batch 1 when first initializing
batches. Its earlier approval dates are not reconstructed, and it is not presented
as an exact historical training-dataset snapshot. Workspaces without batch metadata
retain cumulative preparation. The first explicit new batch on an unreviewed image
is Batch 1; the next batch after existing review work is Batch 2.

Click a button to highlight its regions and make it active for subsequent
annotation approvals. This leaves the explicit **Training: Batch N** selection
unchanged, so viewing history does not invalidate a prepared dataset. Changing
training selection or its membership invalidates prepared data; selecting another
annotation batch clears pending tile selections. Global review coverage is separate and counts each region
once. Right-click a button to add selected, already-reviewed regions; remove regions
from membership without revoking global review; rename; finish/reopen; or inspect
its dated review actions and model-run references. Finished batches remain available
for training, but must be reopened before adding approvals. Grid-size changes do
not move stored coordinates; counts describe member regions rather than a newly
resized grid. Current review validity still controls supervised pixels.

**Training: all reviewed annotations** restores the cumulative annotation sources.
**Training: Batch N** clips existing supervision to that batch's member coordinates,
including for multiple pairs and independent channels. Each image/Labels pair owns
its selection; configure all included pairs/channels deliberately. Unknown pixels
retain their existing validity masks. A reviewed-region batch can be prepared
without rebuilding unrelated pending component selections; those pending selections
are excluded from that run. Cumulative preparation retains its existing library
update requirements. No global connected-component analysis is introduced by batch
selection.

Each batch has a persistent color and a matching button marker. **Show: Active
batch**, **Show: All batches**, and **Show: Hide** only affect display. Approval
updates the colored overlay automatically. In All batches, earlier batches have
dim outlines and lighter fill. Overlap fragments belong visually to the active
batch first, then the most recent earlier batch, preventing blended identities.
Yellow remains reserved for pending selections. A red × marks any member region
that is no longer fully reviewed; its border retains its batch color. These
coordinate overlays do not allocate image-sized masks or change training validity.
Colors, display preference, and the independent training selection persist in
workspace metadata; older batches receive stable fallback colors without altering
historical approval dates. Colors indicate membership, not that every patch was
used in training or that historical annotation pixels were frozen.

After training Batch 1, save the checkpoint. Create Batch 2, review additional
regions, prepare Batch 2, select the saved model, and click **Continue training**.
This retains the model architecture and weights, including width 64. Training data
comes from the selected batch; older examples are included only through an explicit
selection or the cumulative scope. Earlier validation coordinates remain held out
when preparing batches for the loaded model. If a batch cannot supply independent
training and validation targets/background, preparation explains what is missing.
Loading a model after preparation reassesses the reserved regions before continuation.

Training histories record batch IDs, names, membership revisions, and selected
coordinates in addition to the actual model patches used. A trained batch is not
automatically marked fully used: percentage sampling may visit only some patches.
Historical runs are not retroactively assigned to imported batches. Source pixels
are shared, not frozen historical copies; annotation changes still require review
and re-preparation. Exact restoration of old annotation pixels is a separate capability.

## Optional model dependencies

Install SMP in the environment running napari with
`python -m pip install segmentation-models-pytorch`. Verify the import and restart
napari from the same environment before training. See
[SMP installation and troubleshooting](model-management.md#installing-and-checking-smp)
for commands and how to distinguish missing SMP from a broken dependency import.

## Confirmed tiles and saved object selections

**Label Curation — Simple** provides a compact route: edit with automatic camera
follow, approve the current review tile or select several checked tiles in Review
mode, then **Prepare Dataset → Open Training**. Selected yellow tiles remain
pending until approved. Approval includes both annotated foreground and verified
background. Subsequent painting requires re-review. Tools can clear pending
selection, revoke approval, or undo the last review action without changing mask
values. The Advanced presentation retains all existing source, channel, and
training settings; switching presentations shares the same controllers.

Review-tile size determines approval boundaries. The edit-buffer size controls
memory usage, and the training crop size controls model inputs; these sizes need
not match. Tile-progress recovery becomes available after the first workspace
save, while full workspace Save remains necessary for new layers and other settings.

These annotation workflows apply to **This image** mode. For several images,
open **Advanced source options → Source mode → Multiple images → Manage pairs…**. Select pairs,
configure coverage and reviewed masks separately, and confirm corrected model
predictions before using them as ground truth. Pair definitions persist by layer
ID in the workspace. The dialog's **Review selected pair** opens the regular
review controls; those controls stay collapsed otherwise.

Multiple-pair preparation streams eligible patches across all included images.
Holdout and k-fold validation reserve entire image/specimen groups. Give related
images the same split group, and use comparable image scales. Each split needs
targets and background; semantic classes need examples in both splits. The
single-pair workflow continues to reserve spatial regions within its image.

Keep **Partial — confirmed tiles** when Curated contains only some targets.
Zero pixels may contain missing objects, so they must not automatically become
background. Partial mode uses **All / selected training patches** and supports
percentage/count trial subsets while preserving its supervision-validity masks.

Whole-mask Labels overviews use derived OME-Zarr resolution levels for display.
Dataset indexing and supervision always read full-resolution `s0`, including
when the active editable tile shows only part of the source. Overview layers
never become training masks, and visually overlapping preview objects do not
change annotation IDs or review validity.

### Edit and confirm tiles

1. Load the workspace and choose the image and curated mask.
2. Open **Labels → Edit & Confirm**. Zoom, paint, erase, and save as before.
3. Under **Confirm tiles for dataset**, set **Review-tile height/width**. These
   dimensions define approval areas independently of the memory-limited edit crop.
4. Select a tile, check all target objects and background, then click **Confirm
   tile for dataset** or **Confirm tile & next**. For repeated approvals, enable
   **Click tiles to confirm for dataset** and click fully checked tiles. Turn
   selection off before painting again.
5. Use **Go to dataset preparation**. In **Training → Dataset**, keep **Partial —
   confirmed tiles** and click **Prepare dataset**. Unconfirmed areas are excluded.

Tile status distinguishes **Not confirmed**, **Confirmed for dataset**, and
**Needs recheck after edits**. Historical training use is a separate status.
Green outlines show current approval; blue outlines show recorded training patch
footprints. Save the workspace to preserve approvals and history. Existing small
approvals remain valid when the review grid changes, even if they do not cover
entire cells in the new grid.

Dataset shows confirmed-tile counts, **Edit & confirm tiles…**, and **Show
confirmed tiles on image**. After preparation, **Show training / validation
patches on image** displays dataset membership in the current view. Model patch
and input sizes remain under **5. Advanced**. If there are too few separate
spatial groups, confirm more distant regions or use the smaller-patch retry;
this does not change review-grid dimensions or approvals.

Single-image projects saved in Multiple pairs automatically use spatial
validation. Missing review references recover when exactly one coverage mask
matches the image and curated mask. Saved approval rectangles bound partial
supervision even if the coverage array contains extra nonzero pixels. External
coverage without rectangle metadata retains its explicit validity mask.

The older loaded-edit-area confirmation controls remain under **Edit & Confirm →
Advanced: confirm loaded edit areas**. They approve the loaded crop, which may
differ from the review grid. **Save navigation progress** is only a navigation
marker and never grants dataset approval.

### Mask Analysis & Cleanup

Use this tab to **Find components**, sort the index table, inspect components on
the canvas, preview **Remove below** thresholds, and explicitly remove noise.
**Undo last deletion** restores the latest cleanup. **Entire mask — full resolution**
is the default search scope, independent of zoom and the current edit crop. Choose
**Current edit area** for a quicker local search. Objects may continue beyond a
local area's edge. Grid addresses and copy-destination settings remain available
in a collapsed advanced section.

OME-Zarr `s0` is the annotation data; `s1`–`s6` are display resolutions of that
same mask. Component search uses full-resolution data, never the currently
displayed coarse resolution. Entire-mask search streams `s0` in bounded chunks,
joins components across chunk boundaries, and pages results with progress and
Cancel. It works while zoomed out with only an overview displayed. Component
coordinates, canvas picking, cleanup, copy, and undo use full-image coordinates.
It does not switch to full-layer rendering or allocate a full-image ID map.

In **Current edit area** mode, only the loaded crop is searched and its bounds
are displayed. An empty crop does not imply an empty mask. Pan and use **Load
edit area at view center and find components**, or choose **Entire mask** instead.

Selecting a table row or a canvas component does not approve training data.
Return to **Edit & Confirm** to approve complete tiles. Cleanup affecting an
already confirmed area invalidates its approval until rechecked.

### Dataset → Advanced: saved object selections and coverage: select components

1. In **Training → Dataset**, choose Curated as ground truth and open
   **Advanced: saved object selections and coverage**. The panel also selects reviewed coverage;
   cell approval itself stays in Labels.
2. Click **Build / update library**. The first scan is a background operation
   with progress and Cancel; subsequent local edits update affected components.
3. Check **Train** beside objects to include. Clicking another column locates
   that object. The table contains ID, label pixels, and bounds; no Euler
   calculation is performed for the library.
4. Save the workspace once to include the library asset. Later checkbox changes
   are written immediately to that library.
5. Confirm complete checked tiles under **Labels → Edit & Confirm → Confirm tiles for dataset**. A checked component
   supplies foreground, not confirmation of its surroundings.

Both methods feed the same training dataset. Shared patches and pixels are
counted once. Unchecked surroundings and invalidated review cells are ignored.
Reviewed cells include all verified foreground and zero-valued background. A cell
with no targets can supply negative supervision. Selected objects confirm only
their foreground pixels. The existing split policy requires targets and verified
background in training and validation, so a dataset consisting solely of selected
objects cannot yet be prepared for training. Positive-only patches are supported
within datasets that also contain adequate reviewed background.

The Dataset overview shows actual source selections and preparation status.
Exact supervised foreground/background and excluded unknown counts are calculated
from the patch index after preparation, without allocating full-image masks.
Source edits or changed dataset settings mark the dataset outdated and disable
training until it is prepared again. External coverage masks without recorded
cell rectangles are measured during indexing rather than assigned guessed counts.
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

**All / selected training patches** (formerly **All patches**) is the coverage
workflow for large images. **All** remains the default training amount.

- **Source patch size** is the area read from the original image.
- **Model input size** is the resolution fed into the model after resizing.
  Matching sizes, such as 512 × 512 for both, preserve spatial resolution.
- **Preview training patches** scans annotation patches and stores coordinates and
  counts. It does not load the complete image into memory.
- The gallery contains representative examples. It is not the whole training
  dataset. The dataset summary reports total patches, train/validation counts,
  and reviewed area as a percentage of the image.
- Each **epoch** visits every selected training patch in shuffled order. The
  **batch size** is the number of patches processed in one model update.
- **Additional target patches** adds repeat visits to patches containing targets;
  these visits supplement, rather than replace, coverage of the chosen training set.
- Validation uses reserved spatial blocks and does not update model weights.
  Patch reads, loss, and validation exclude image-edge padding and unknown pixels.

### Quick trials from confirmed annotations

In **Dataset → Training patch selection**, choose **All**, **Percentage**, or
**Number of training patches** after preparing the index. These are model patches,
not review-grid tiles. A 10% budget over 1,225 eligible training patches uses 123
unique training patches. Counts larger than the available pool use all available
training patches. The summary reports the actual number and target-containing
count; Train reports additional target repeats separately.

Subset selection happens after the spatial/image-group split and never selects
validation patches for training. By default, every reserved validation patch is
evaluated each epoch. Enable **Quick trial — smaller validation set** to reduce
that work too. If training was set to All, enabling Quick trial switches it to the
training percentage setting. Existing percentage/count choices are retained.
Set **Validation percentage** independently (default 10%). For example, a pool of
5,257 training and 1,813 validation patches uses 526 training and 182 validation
patches when both percentages are 10%.

The validation sample stays fixed across epochs, training percentages, and training
subset seeds. It follows **Validation → Split seed** and spreads samples across
reserved spatial/image groups, with proportional foreground/background sampling.
Small subsets must still supply the required foreground/background and class/channel
coverage; otherwise the controls explain what is missing. Training and validation
selection uses metadata only and reuses the prepared index; initial preparation
still indexes all eligible patches. Changing **Subset seed** chooses different
training coordinates without changing validation or rescanning annotations.

Quick-trial metrics are labeled **Trial validation** in the UI, curves, model library,
and CSV reports. They select the trial checkpoint and are approximate subset scores.
After training, select **Train → Validation → Evaluate full validation set** to
evaluate the retained weights once on the complete reserved pool. This makes no
optimizer updates and does not reselect weights. Full-validation metrics and their
dataset provenance are stored separately from trial checkpoint-selection scores.
Save the workspace or export the model afterward to preserve that evaluation.
Full validation is still held-out validation, not an independent test-image score.

Approvals, foreground-only object selections, and channel-specific unknown pixels
remain unchanged. Patch previews and canvas split overlays use the chosen subset.
Workspace settings store both budgets and their seeds. Existing workspaces without
Quick trial settings default to full validation. Completed model histories record
the budgets, available/selected counts, and actual selected coordinates; training
footprints are recorded only for patches used in training. Choosing **All** restores
the full eligible pool. K-fold validation and final all-reviewed training use their
existing full-index workflows; trial-budget controls are disabled for k-fold.

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
matching image name, image dimensions, and patch size. Loading a model can reuse a compatible prepared dataset; otherwise prepare
the dataset again. Known sampled-training regions from the same image
name are excluded when selecting new validation blocks. If no independent reviewed
target blocks remain, review new regions or start a new model.

The panel calculates updates per epoch and total updates before training.
Progress reports epochs, unique training patches visited, model updates, and
validation progress. The checkpoint with the best validation Dice is retained.
Continuation retains the starting weights if every evaluated epoch scores lower;
ties retain the latest evaluated weights.
A continuation round uses a fresh optimizer. Cancel preserves the previous model.

## Continue training an existing model

**Continue training** improves the model currently in memory, including one
loaded from a `.pt` checkpoint exported by this plugin. **Train new model**
creates another model using the selected architecture and initialization; it
does not start from the current model's trained weights.

1. Train a model or load its checkpoint in **Models**.
2. Select its biological task and keep its class/channel definitions compatible.
   Continue reuses the current model's architecture automatically.
3. In **Dataset**, prepare the dataset if it is missing or outdated. A compatible
   prepared dataset can be reused. **Show patch preview** is optional.
4. Set **Learning rate** in **Advanced**, then open **Train** and click
   **Continue training**.
5. Save the model or workspace again to persist the completed round.

Continue starts with the existing weights but creates a **new AdamW optimizer**.
It does not resume the old optimizer's internal state. A smaller learning rate,
such as `0.0001`, is an optional fine-tuning setting, not a required value or a
guarantee of improvement.

At the beginning of a continuation round, the starting model is evaluated on
the current validation data. After further training, the weights with the best
validation Dice in that round are retained. If every evaluated checkpoint is
worse, the starting weights are kept. A tie keeps the latest evaluated weights.
Cancel discards the unfinished round and preserves the previous model and its
completed history. Completed-round and total optimizer-update counts still
increase when a round finishes, even if the starting weights win.

In **Sample patches** mode, increasing crop count or changing the sampling seed
can select different training locations. The coverage workflow visits every
chosen training patch each epoch; its trial budget can select a subset of the
eligible pool. Training longer on a fixed sample does not expand its
spatial coverage. Retaining representative earlier examples in later rounds can
help reduce forgetting.

Compatible single-image All-patches preparation reuses reserved spatial blocks;
changing annotations requires preparation again but does not necessarily move
those blocks. Multiple-image datasets keep entire image/specimen groups together.
Compare round scores only when validation regions, targets, and preprocessing
are equivalent. All-patches continuation is blocked when the plugin detects that
the model already trained in the selected holdout; choose independent validation
regions or train a new model. Spatial k-fold evaluation starts fresh models and
does not support Continue training.

Saving writes completed history inside the `.pt` checkpoint and also exports
`model.history.json` beside `model.pt`. Each completed round records timestamps,
duration, source information, settings, learning rate, and patch locations/counts.
Sampled training records metrics at evaluation steps and a `retained_step`;
All-patches training records epoch evaluations and a `retained_epoch`. These are
periodic evaluation records, not a log of every optimizer update. Loading and
continuing appends another completed round; saving to another path carries the
history forward. Supported older plugin checkpoints remain loadable, but missing
historical rounds cannot be reconstructed.

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
   **Edit & Confirm → Confirm tiles for dataset**. Set the review-tile height and
   width independently of the editable crop size.
2. Enable **Select review tiles on canvas**, then click a cell. Its yellow
   outline and address identify the whole region that needs review.
3. Turn cell selection off while editing Curated. Correct false positives and
   add every missing target throughout the yellow cell boundary. The editable
   area may cover only part of that cell; inspect and edit the remaining parts
   before declaring the cell complete. Check objects crossing cell boundaries.
4. Click **Confirm tile for dataset**. Pending edits are
   applied first. Bounded editing creates a sparse review mask automatically;
   legacy full-layer editing asks for a storage location. Green outlines show reviewed areas.
5. In Training choose **Partial — confirmed tiles**. This automatically
   uses **All / selected training patches**. The separate legacy **Sample patches**
   workflow remains available only for complete annotations; use **Training patch
   selection** for trials with partial annotations. The associated
   **Curated** ground truth and **Reviewed regions** coverage are selected
   automatically. Click **Prepare dataset**, then train. Patch preview is optional.

**Remove reviewed status** clears the selected cell's coverage without deleting
Curated objects. Marking, unmarking, or editing sources requires dataset reindexing.
Changing grid size clears the selection but preserves previously marked pixels.

In the legacy workflow, **Reviewed annotations** is a read-only binary copy: objects from reviewed cells
have value 1, background 0. It is updated when a cell is marked or remarked, and
is excluded from Training ground-truth choices. Curated retains its object IDs.
**Reviewed regions** records entire trusted cells including background; Training
uses this coverage to ignore unreviewed pixels. Green outlines are viewing helpers.
Bounded editing reads original Curated IDs through this coverage without creating
another foreground copy. Workspace saves preserve coverage and source associations.

**Click tiles to confirm for dataset** activates direct click-to-mark
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
