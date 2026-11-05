# Prioritized technical findings

Reviewed SHA: `cb95a5e97717aca24db658c488fd810647955bcd`, 2026-11-03. All reproductions use disposable synthetic data. **Verified** means the described observation was executed; downstream scientific consequences may also depend on the stated conditions. Tests asserting safer behavior intentionally fail on this commit. No application repair was applied.

## Findings at a glance

| ID | Priority | Affected operation | Evidence | Observed result |
|---|---|---|---|---|
| A01 | P0 | Workspace spatial integrity / prediction persistence | Verified | Saving and reopening removes the prediction's full affine; unchanged mask pixels move to a different physical location. |
| A02 | P0 | Local editing on rotated/sheared layers | Verified geometry; write consequence code-reviewed | Editable helper axes differ from source axes, although the helper's origin matches. |
| A03 | P0 | Tiled component copy into narrower dtype | Verified | Label 300 becomes 44 in uint8 destination; completion reports success. |
| A04 | P0 | Component copy between spatially incompatible layers | Verified | Same-shaped layers at different world positions are accepted; copied pixels are physically misplaced. |
| A05 | P0 | Save after replacing Zarr layer data | Verified | Live pixels all 9 reopen as old pixels all 2. |
| A06 | P1 | Opening a workspace whose assets are unavailable | Verified service behavior | Previous live layers are removed before assets fail; no rollback. UI does ask permission to clear layers. |
| A07 | P1 | Current-pair dataset alignment | Verified | Shifted truth is accepted and “Dataset ready” appears; grouped/channel paths correctly reject misalignment. |
| A15 | P1 | Very large full-image inference | Verified allocation request; size extrapolation | Tiled reads still request a full float32 output before reading a source tile. |
| A19 | P1 | Copy over existing curated labels | Verified | Existing ID 9 is overwritten by ID 7 with no collision confirmation or plugin copy recovery. |
| A08 | P2 | Normal-training dataset identity | Verified | Changing raw image pixels leaves the snapshot ID unchanged. Dedicated Compare has stronger checks. |
| A10 | P2 | Topology interpretation | Verified | Components use 4-neighbor foreground, Euler uses 8-neighbor foreground; diagonal openings give a different hole count. |
| A20 | P2 | Prediction-to-ground-truth safeguards | Verified | Current pair accepts experimental predictions without the explicit review confirmation used for pairs/channels. |

P0 denotes observed silent scientific pixel-value or coordinate corruption in the specified path. A02 is restricted to unsupported transform combinations that the interface nevertheless accepts; no wrong biological annotation was created in a production project. A15 does not mean the documented bounded local-editing implementation is fictitious.

## A01 — Full spatial transform is not serialized

- **Source:** `workspace/service.py:_common_record` L1199 and `_apply_common_state` L1240; `_quick_train_widget.py:_prediction_done` L1928.
- **Reproduce:** Run `test_independent_audit.py::test_A01_prediction_world_coordinates_survive_workspace_roundtrip`. Create a source with scale `(2,3)`, translate `(7,11)`, rotation 25°, shear 0.2; create a 5×7 current-area prediction at pixel origin `(9,13)` using the actual prediction completion handler; portable-save and reopen.
- **Expected:** Identical data-to-world mapping and mask pixels.
- **Actual:** Prediction `(0,0)` moves from world `(13.9006286977,57.2495548473)` to `(0,0)`; other basis points also change. Pixel array remains exact. The prediction transform is stored in `affine`, but only scale/translate are persisted. This also affects ordinary rotated/sheared layers. Even an unrotated cropped prediction can have its crop translation solely in affine.
- **Evidence:** [A01 JSON](evidence/independent_observations/A01_affine_roundtrip.json), [failing assertion](evidence/independent_suite.log).
- **Impact:** Saved segmentation may overlay the wrong physical pixels despite apparently successful saving and matching array checksums.
- **Correction:** Persist/restore the complete transform chain with a versioned, backward-compatible representation. Alternatively decompose the exact composite affine consistently. Preserve existing scale/translate-only projects.
- **Regression:** Compare source and result world coordinates at origin, both basis vectors, and corners before/after save; include identity, translated ROI, anisotropic scaling, rotation, shear and additional affine. Verify pixels independently.

## A02 — Local editable area does not inherit source axes

- **Source:** `_dynamic_edit.py:DynamicLabelsEditController._finish_load` L409; `commit` L701 and boundary construction.
- **Reproduce:** Run A02. Force local-area editing for a 64×80 source with scale `(2,3)`, translate `(7,11)`, rotation 30°, shear 0.2. Compare helper `(1,0)`, `(0,1)` and `(6,9)` with corresponding source pixel world coordinates.
- **Expected:** Every helper pixel occupies the source pixel's physical position, or editing is explicitly rejected.
- **Actual:** Origin is derived with `data_to_world`, but helper axes use only `source.scale`. Non-origin world coordinates differ. Commit writes helper indices back into the rectangular source-index bounds.
- **Evidence:** [A02 JSON](evidence/independent_observations/A02_edit_transform.json). The coordinate mismatch was executed; misdirected paint is inferred from the reviewed commit path, not a human stroke experiment on a rotated image.
- **Impact:** A user painting the visible helper can select a physically different location from the source image beneath it.
- **Correction:** Reuse an exact crop-to-world transform for helper, boundary and navigation; immediately block unsupported transforms until alignment is guaranteed.
- **Regression:** Paint one synthetic local pixel and verify its source index and both layers' world positions, including a crop away from origin and near each edge.

## A03 — Tiled copy silently narrows integer labels

- **Source:** `_large_component_controller.py:LargeComponentController.edit_components` L127, especially target-tile assignment; contrast `_widget.py:_copy_mask_components_to_target` L798, which promotes dtypes.
- **Reproduce:** Run A03. Build a tiled component index for an 8×9 uint16 source containing twelve pixels with label 300. Copy into a same-shaped uint8 Labels destination using the real Qt controller.
- **Expected:** Preserve 300 using an appropriate integer destination, or reject before any writes with a useful message.
- **Actual:** Destination contains 44 (`300 modulo 256`); controller reports twelve updated pixels without an error.
- **Evidence:** [A03 JSON](evidence/independent_observations/A03_large_copy_overflow.json).
- **Impact:** Biological class/object identities change silently. Tiny fixture forces the same tiled controller used for large masks; no large allocation was needed.
- **Correction:** Preflight all selected values against destination range. For disk-backed stores, reject or explicitly create a promoted store; perform no partial writes on failure. Share one exact integer policy between small and large paths. Also guard signed/unsigned mixed promotion into float.
- **Regression:** uint16→uint8 with 300; uint64 values above 2^53; signed/unsigned destinations; Zarr targets; assert exact values or unchanged destination after explicit rejection.

## A04 — Equal shape is mistaken for spatial compatibility during copy

- **Source:** `_widget.py:_eligible_copy_targets` L1902 and `_copy_selected_to_layer` L2536. New destination creation also copies only scale/translate. Combine `_apply` / `_write_output` L1098 has a similar **code-reviewed** omission, not separately executed here.
- **Reproduce:** Run A04. Analyze a label-7 component at indices `(2:5,4:8)` in a 16×20 source at world origin. Select an equal-shaped destination translated `(20,30)` and press Copy.
- **Expected:** Reject incompatible transforms or perform an explicit, documented coordinate mapping.
- **Actual:** Twelve pixels are copied. Source pixel `(2,4)` is at world `(2,4)`; target pixel `(2,4)` is at `(22,34)`. Status reports a successful copy.
- **Evidence:** [A04 JSON](evidence/independent_observations/A04_copy_alignment.json).
- **Impact:** Accepted annotations can be placed at the wrong specimen location.
- **Correction:** Use the same basis-point transform check already present in `resolve_pairs` and `resolve_channels`; preserve full transforms for new outputs. Do not silently resample labels.
- **Regression:** Matching shapes with differing translation, scale, rotation, shear or affine must fail before writes; matching full transforms must copy exactly. Extend to Combine separately.

## A05 — Remembered storage overrides the current Zarr data object

- **Source:** `workspace/service.py:_zarr_reference` L1144, `_remember_layer_storage` L1174 and label serialization.
- **Reproduce:** Run A05. Create same-shaped stores A (all 2) and B (all 9). Save a layer referencing A; assign B to `layer.data`; save to the same manifest and reopen.
- **Expected:** Reopen the actual current data, B, preserving all 9s.
- **Actual:** Reopens A, all 2s. `_zarr_reference` checks remembered metadata before discovering the replacement Zarr object's real store. Its NumPy replacement guard does not cover Zarr replacements.
- **Evidence:** [A05 JSON](evidence/independent_observations/A05_stale_zarr_binding.json).
- **Impact:** Save success does not guarantee the displayed scientific data is what the project will reopen. Store B is not destroyed, but its live association is lost.
- **Correction:** Derive identity from the current array/store and verify it against remembered storage; invalidate stale bindings on data replacement. Check array path, not just store root.
- **Regression:** NumPy→Zarr, Zarr A→B, another array within the same store, altered shapes/dtypes, multiscale replacements, Save and Save As.

## A06 — All-failed open has no rollback

- **Source:** `workspace/service.py:load_workspace` L255; `_workspace_widget.py:_load_path` L441 and `_load_with_progress` L458.
- **Reproduce:** Run A06. Start with a live unsaved synthetic Labels layer; open a structurally valid manifest whose sole Zarr asset is missing.
- **Expected:** Keep the previous workspace accessible if no layer can be restored, or stage the new project before replacing it.
- **Actual:** No layers remain; result contains one skipped layer. Model schema/dependency checks occur before clearing, but actual asset access occurs afterward.
- **Evidence:** [A06 JSON](evidence/independent_observations/A06_failed_open.json).
- **Qualification:** The UI explicitly confirms removal of current layers and blocks pending local helper edits. This is a failed-operation recovery weakness, not an undisclosed bypass of that confirmation. Source Zarr files are not deleted. Partial-load errors are exposed and incomplete projects are blocked from saving.
- **Correction:** Preflight asset access and preserve old layer objects until at least a recoverable new state exists; offer restore on total failure.
- **Regression:** Missing/corrupt first, middle and all assets; unreadable files; missing checkpoints; assert rollback or explicit retained-layer recovery.

## A07 — Current-pair preparation bypasses transform validation

- **Source:** `_quick_train_widget.py:_start_preparing` L1406; contrast `_dataset_pairs.py:resolve_pairs` L75 and `_multilabel_targets.py:resolve_channels` L43.
- **Reproduce:** Run A07. Pair a 128×160 image at origin with same-shaped truth translated `(9,11)`, choose Complete, patch/input 32 and Prepare dataset.
- **Expected:** Reject the spatial mismatch consistently with multi-image/channel datasets.
- **Actual:** `prepared` is populated and status says Dataset ready. Only shape is checked in this path.
- **Evidence:** [A07 JSON](evidence/independent_observations/A07_current_pair_alignment.json).
- **Impact:** Training pairs disagree with what the researcher sees overlaid. This is conditional on transformed/misregistered layers; it does not prove every current-pair dataset is wrong.
- **Correction:** Consolidate pair compatibility checks across preparation paths, including reviewed masks.
- **Regression:** Single-pair translated/scaled/rotated truth and reviewed mask rejected before locking sources/indexing; aligned pairs still prepare.

## A15 — Tile-based inference retains image-sized accumulators

- **Source:** `_quick_train.py:predict_region` L714.
- **Reproduce:** A15 uses a fake 100,000² source and an allocation guard that intercepts large `np.zeros` requests. No huge array is created, and the source raises if read before the guard.
- **Expected:** A feasible out-of-core output path or a preflight memory estimate with an actionable size limit before allocating.
- **Actual:** Requests a 100,000×100,000 float32 probability array: 40,000,000,000 bytes, before any source tile read. Also allocates a full float32 weight image. Binary lower bound is 80 GB decimal (74.5 GiB); four-channel probability plus weights is 200 GB decimal (186.3 GiB), excluding output masks, model and temporary buffers.
- **Evidence:** [A15 JSON](evidence/independent_observations/A15_allocation_request.json); bounded real prediction measurement in [performance.json](evidence/performance.json).
- **Qualification:** These totals are code-derived allocation requirements, not measured 100k inference RSS or an attempted OOM. README qualifies 100k workflows as local editing/chunked access; its broad scale must not be extrapolated to full-image inference.
- **Impact:** The full-image button can exhaust RAM on the same project that local editing handles successfully.
- **Correction:** First show a memory budget and enforce a feasible limit; then consider streamed Zarr accumulators/output. Keep ROI prediction available. Avoid broad refactoring before a measured need.
- **Regression:** Allocation-spy tests for requested bounds/channel count, budget rejection before allocation, cancellation during safe tiled work, eventual streamed-output equivalence on small known arrays.

## A19 — Component copy can destroy prior curated values without recovery

- **Source:** `_widget.py:_copy_selected_to_layer` L2536; `_copy_mask_components_to_target` L798; tiled copy controller. Deletion journaling is a separate path.
- **Reproduce:** Run supplemental A19. Candidate component ID 7 overlaps twelve curated pixels with ID 9; select Copy to curated.
- **Expected:** Existing conflicting annotations are protected by explicit overwrite confirmation or usable undo/journaling.
- **Actual:** ID 9 disappears and becomes 7; Copy reports success. The plugin's Undo last deletion remains disabled; whole-array replacement does not add a native paint stroke to napari undo.
- **Evidence:** [A19 JSON](evidence/independent_observations/A19_copy_collision.json). This probe establishes lack of plugin copy recovery, not the impossibility of recovery from an external backup or retained source file.
- **Impact:** Copying accepted predictions can erase curated identities. On disk-backed tiled targets, writes may already be durable.
- **Correction:** Detect nonzero conflicting pixels before copy; provide a conflict summary and explicit overwrite choice, with a bounded recovery journal. Avoid prompting when copying only onto background.
- **Regression:** Conflict/no-conflict copies on NumPy and Zarr; cancelled conflict prompt leaves destination exact; undo restores old identities without erasing later edits.

## A08 — Normal-training fingerprint omits raw image content

- **Source:** `_model_comparison.py:dataset_snapshot` L17; `_coverage_train.py:train_indexed` history construction. `compare_models` L32 explicitly adds image digests first.
- **Reproduce:** Run A08. Index fixed truth, alter a small raw-intensity rectangle, reindex with identical settings and source metadata; calculate snapshots.
- **Expected:** Different training input gives a different content identity, or the UI clearly labels the ID as coordinate/annotation identity only.
- **Actual:** Identical SHA256 snapshot IDs; no raw pixel digest in normal indexing. Annotation digests correctly detect changed labels.
- **Evidence:** [A08 JSON](evidence/independent_observations/A08_snapshot_identity.json).
- **Impact:** “Same dataset” provenance/ranking can conflate changed acquisitions or preprocessing. Dedicated controlled comparisons are stronger and were not shown invalid by this probe.
- **Correction:** Use bounded patch content fingerprints consistently, including preprocessing contract and validation assignment, or restrict the meaning of existing IDs. Version identities to preserve old records.
- **Regression:** Intensity-only changes and source replacement change IDs; harmless layer renaming policy documented; fixed content/splits reproduce IDs; mutation during comparison rejected.

## A10 — Component and Euler connectivity conventions differ

- **Source:** `_widget.py:_build_fast_component_index` L735; `_large_components.py:_euler_numerators` L127. README describes Euler 0 as one hole without discussing diagonal connectivity.
- **Reproduce:** A10 uses `[[1,1,1],[1,0,1],[1,1,0]]`. The center background connects diagonally to the exterior under 8-neighbor background.
- **Expected under component convention:** Four-connected foreground implies the dual eight-connected background; Euler is 1 for this connected foreground without a hole.
- **Actual:** Euler is 0 because the code explicitly uses eight-connected foreground for Euler, while component detection uses four-connected foreground.
- **Evidence:** [A10 JSON](evidence/independent_observations/A10_connectivity.json).
- **Qualification:** Eight-connected Euler is mathematically legitimate. The finding is mixed, insufficiently explained semantics, not a universal arithmetic failure. A10's expected value deliberately enforces the dual convention of the component detector. It is not valid to say all reported Euler numbers are wrong.
- **Impact:** Topology-based filtering of diagonal openings can produce a different scientific interpretation than users expect from the component partition.
- **Correction:** Choose one documented foreground/background pair across detection and topology, or explicitly expose which convention each measurement uses. Preserve old results' interpretation if changing it.
- **Regression:** Diagonal contacts/openings, rings, holes, tile seams and borders against an independently specified topology oracle, not only agreement between two app implementations.

## A20 — Prediction review confirmation differs by dataset mode

- **Source:** `_quick_train_widget.py:_start_preparing` L1406; `_dataset_pairs.py:resolve_pairs`; `_multilabel_targets.py:resolve_channels`.
- **Reproduce:** Supplemental A20 creates an experimental prediction with the actual prediction metadata/name, chooses it as current-pair truth and presses Prepare dataset with Complete coverage.
- **Expected:** Consistent explicit reviewed-prediction acknowledgement across all dataset modes.
- **Actual:** Dataset is ready without a confirmation; multi-image and channel resolvers demand reviewed-prediction confirmation.
- **Evidence:** [A20 JSON](evidence/independent_observations/A20_prediction_as_ground_truth.json).
- **Qualification:** Choosing Complete is already a coverage assertion, and some predictions may have been manually corrected. The probe demonstrates an inconsistent safeguard, not proof that the user trained on unreviewed pixels or that pseudo-labeling can never be intentional.
- **Impact:** Researchers may mistake a previous model output for verified ground truth and reinforce its errors.
- **Correction:** Reuse the existing reviewed-prediction state in current-pair mode and make prediction provenance visible beside truth selection.
- **Regression:** Experimental prediction rejected until acknowledged; genuine/manual truth allowed; changing prediction data invalidates stale acknowledgement where feasible.

## Additional limits and risks, not independently established defects

| Area | Evidence and limit | Practical next validation |
|---|---|---|
| Specimen leakage | A17 verified that identical acquisitions assigned distinct image/group IDs can cross the split. Groups are manual; implementation does keep correctly assigned groups disjoint. | Audit specimen identifiers and original acquisitions; duplicate-content warning is useful but cannot infer all related tissue. |
| Spatial correlation | Static inspection: nonoverlapping spatial blocks have no mandatory physical exclusion buffer. No inflated numeric score was proven. | Spatial-distance sensitivity experiment and independent specimen holdout. |
| Full uint64 library support | Code-reviewed SQLite INTEGER stores label values as signed 64-bit. Existing independent pixel roundtrip covers 2^53+1, not >2^63−1 library records. | Exercise max uint64 library ingestion; either declare a limit or use exact text/blob representation. |
| Large component count | Disk tiling bounds image working sets, but Python records, union-find and provisional fragments grow with component count. Euler accumulation uses global-count arrays per tile. | Adversarial fragmented-mask benchmarks with controlled component count and disk budget. |
| In-place mutable stores | Save/manifest backup does not version every mask pixel; editing writable Zarr changes the underlying store before the next manifest save. | Explain durable pixels vs saved project state; interrupted-write/recovery tests on isolated stores. |
| GUI reentrancy | Workspace progress calls `QApplication.processEvents`; layer restore work is synchronous. Long chunk writes/readers can block despite periodic progress. | Real desktop latency/close/cancel tests on slow storage; do not infer responsiveness from offscreen success. |
| Installation/version matrix | `napari>=0.4.19`, `zarr>=2.18`, Python>=3.10 have no upper bounds; no CI workflow in tracked tree. Only one modern stack was executed. | Clean installations and compatibility matrix before broad stable-release claims. |

No serious unconditional training-data leakage, universally invalid Dice/IoU arithmetic, or biological generalization failure was claimed without evidence. The release blocker is the demonstrated coordinate/value/save integrity failures.
