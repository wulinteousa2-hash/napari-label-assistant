# Architecture, state and scientific scope

Reviewed SHA: `cb95a5e97717aca24db658c488fd810647955bcd`. This is a focused engineering assessment of the entire tracked tree's structure and the high-risk implementations. Every tracked file was inventoried and hashed; every Python file was AST-indexed. That does not mean every branch was independently exercised or every line received equal review depth. See the [feature map](01_feature_inventory.md) and [complete function/test index](evidence/source_index.md).

## Architecture and maintainability

The main separation is sound: numerical array functions, streamed dataset generators, providers, model/project metadata, workspace service and Qt panels are distinct modules. Lazy PyTorch/SMP imports avoid imposing training costs on annotation-only users. Zarr chunk iteration, temporary component caches and SQLite selections address important scale problems without requiring whole-image copies in every operation.

There are measurable boundary inconsistencies worth fixing before any broad refactor:

| Boundary | Observed engineering problem | Narrow correction justified by evidence |
|---|---|---|
| Small vs tiled component copy | A03: promotion in one path, silent narrowing in another | One selected-value/range policy with NumPy and Zarr adapters; preflight before mutation. |
| Current pair vs grouped/channel preparation | A07/A20: duplicated prerequisite logic with different safety checks | Shared image/label/coverage/provenance compatibility validator. |
| Pixel coordinates vs layer/world state | A01/A02/A04: transforms reconstructed differently across save, helper and copy | Shared exact coordinate-contract helper, tested independently at basis/corner points. |
| Layer object vs remembered asset | A05: metadata can disagree with replacement data | Explicit backing-array identity validation/invalidation. |
| Dataset record vs content identity | A08: normal indexing and controlled comparison fingerprint different inputs | Version one bounded content identity contract. |

`_widget.py` and `_quick_train_widget.py` contain large UI/action closures and access private state across panels/controllers. This increases coupling and makes validation behavior easy to duplicate. Real Qt tests make many paths testable, but the failures above show that helper-level success is insufficient. Extract only the demonstrated shared contracts; there is no measured justification here for replacing Qt, napari, providers, or the whole UI architecture.

State is distributed across layer metadata, viewer private attributes, widget fields, project dictionaries, runtime model caches, JSON, Zarr and SQLite. This supports rich persistence but needs explicit invariants: which layer ID denotes the current data, which revision is reviewed, which checkpoint is loaded, and whether source data is locked. Existing tests cover source invalidation, class-task isolation, dirty model revisions, cancellation and missing providers. Offscreen execution does not prove all concurrent transitions safe.

## Error handling, transactions and threads

- Manifest and checkpoint writes use temporary files and replacement. Existing tests verify preserving the prior manifest when saving fails. This does not make all assets a single atomic transaction: mask stores can already have been edited, sidecar writing can fail after checkpoint publication, and failed staging may leave unreferenced files.
- Missing layers/checkpoints have visible states; incomplete workspace saves are blocked. A06 is narrower: asset access after clearing lacks rollback. Do not report that the plugin silently treats all missing checkpoints as success; the existing missing-model tests passed.
- Component library updates use SQLite transactions; cancellation tests preserve prior usable selections. Disk-backed deletion journals preserve later nonzero edits, but recovery is session-scoped and only covers the most recent deletion.
- Dataset indexing/training use generators with cancellation boundaries and napari workers. Local tile reads have request IDs to discard stale returned work. Existing tests exercise pending strokes, camera navigation and focus preservation.
- Large component edits use `QTimer.singleShot` between tiles; individual tile read/decompression/write and NumPy work still run in that callback on the UI thread. Progress callbacks in workspace loading call `QApplication.processEvents`. Neither approach guarantees responsiveness during slow individual operations. No deadlock or memory leak was demonstrated. A long-run leak/latency test remains required.
- No repository CI configuration was present in the reviewed tracked files. Broad Python/napari/Zarr minima and optional provider versions need a tested matrix rather than assumption. Exact executed environment is in [environment.json](evidence/environment.json).

## Image/label contracts

Training accepts single-channel 2D data, resolving multiscale inputs to the highest-resolution level. Input intensities are normalized per crop with 1st/99th percentiles, clipped to [0,1]; the source is not intentionally rewritten by normalization. Images are averaged when reducing model input, binary targets use foreground-preserving reduction, and semantic IDs use nearest sampling. Reviewed masks suppress unknown/padded pixels; disappeared semantic classes/coverage can raise actionable preparation errors. The same crop normalization is used for inference, but local percentile variation can still make real tile boundaries context-dependent. Constant-output stitching success does not establish seam-free biological inference.

Small component measurements are pixel counts, index-space centroids and half-open bounding boxes, not calibrated µm² by default. Four-neighbor detection preserves disconnected regions sharing a label as separate components. Analysis IDs can change on reanalysis; persistent library IDs have a different lifecycle. Exact uint64 pixel persistence was tested through 2^53+1; a full maximum-ID guarantee across all consumers is not established. A10 explains the topology convention mismatch.

Generated viewing-pyramid levels explicitly transform intensities by averaging; full-resolution originals should remain exact. Existing pyramid tests passed, but independent exact intensity checks in this audit apply to base-resolution ordinary saves, not every reader, RGB stack, pyramid boundary and dtype combination. Arbitrary 3D, RGB/multichannel model inputs and nonaffine registration are outside the implemented training contract.

## ML contract

| Mode | Supervision | Network/output | Loss and inference | Evidence |
|---|---|---|---|---|
| Binary | One label array; any nonzero ID is foreground | 1 logit channel | Sigmoid; masked BCE + soft Dice; threshold | Independent CPU workflow; existing tests |
| Exclusive multiclass | One array with 0 background and foreground IDs 1…K | K+1 logits | Softmax; masked cross-entropy + foreground Dice; argmax includes background | Independent CPU workflow, metric oracle and class tests |
| Overlapping multilabel | K independent binary/nonzero masks with per-channel validity | K independent logits; no background channel | Sigmoid per channel; masked BCE + Dice; independent threshold | Independent CPU workflow, gradient oracle and existing nesting tests |

Config permits 2–254 foreground names for exclusive multiclass and 1–254 names for multilabel. Binary mode is unnamed nonzero foreground; this practical naming limit is not evidence that training 254 channels is computationally feasible. Instance IDs are not channel indices. Channels do not force containment or assign organelles to parent cell instances.

The custom U-Net has five encoder widths, GroupNorm, max-pooling, bilinear upsampling and skip concatenation. The standard provider uses BatchNorm/transposed convolution and handles singleton training batches specially. SMP offers selected architectures/encoders; external pretrained download paths were mocked in relevant existing tests. Native U-Net CPU runs and provider contracts executed; all optional real pretrained model combinations did not.

AdamW starts fresh for each continuation round, which documentation/history states. Best validation Dice weights are retained, while plateau control monitors validation loss. Scores therefore describe a validation-selected checkpoint, not independent test performance. Model versions 1–4 and task/output contracts are checked when loading; corruption and incompatible contracts were independently rejected.

## Splits, leakage and metrics

Coverage indexing assigns source patches to spatial blocks, consolidates known component dependencies, and reserves whole blocks; multiple-image mode keeps the provided image/specimen group intact. Existing tests cover prior training exposure, k-fold/final retraining, component-crossing exclusions and comparison partitions. A17 confirms that unique IDs/groups are disjoint but byte-identical clones can be assigned to opposite splits: software enforces supplied groups, not biological independence. Spatial adjacency, common specimen provenance, near-duplicate crops and repeated validation-driven model selection can inflate estimates even without array overlap. No quantitative inflation was measured here.

Metrics accumulate TP/FP/FN over eligible validation pixels before deriving scores. Binary empty/empty foreground yields zero. For multiclass/multilabel, foreground per-class scores are reported; macro averages include classes with nonzero TP+FP+FN and exclude entirely absent classes. Background is not included in foreground macro scores. Validation loss is an average over validation patch losses, so it is not necessarily the same as a single global pixel-weighted loss when patch validity differs. That is a weighting choice to document, not a proven arithmetic error. Independent per-mode reserved-patch Dice/IoU matched exactly within floating tolerance.

Default GPU and CPU kernels can use different precision. The GB10 smoke experiment found a small default cross-device difference and exact repeated/same-GPU checkpoint outputs; TF32-disabled GPU inference reduced the CPU difference substantially. Device and precision settings should accompany reproducibility claims; the experiment does not establish nondeterministic long training or corrupted checkpoints.

## Scientific curation and reproducibility limits

Original and curated layers can coexist and component selections can persist. Nevertheless, raw/predicted/reviewed/curated are partly metadata and user workflow roles, not universally immutable data types. Copy collisions, wrong transform acceptance and stale store references can violate the user's intent. A review checkbox certifies a user's inspection, not biological truth. Component selections and grid cells do not constitute a complete reviewer audit trail with immutable historical masks. Exact specimen IDs, acquisition calibration and annotation completeness remain researcher responsibilities.

The synthetic test supports computational mechanics. It does not validate axon/myelin segmentation accuracy in real images, inter-rater consistency, g-ratio biology, morphology generalization, independent statistical inference, or external analysis plugins' custom-group charts. Those claims require appropriate data and separate studies.
