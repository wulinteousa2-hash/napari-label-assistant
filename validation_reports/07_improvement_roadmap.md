# Evidence-based improvement roadmap

Reviewed SHA: `cb95a5e97717aca24db658c488fd810647955bcd`. Recommendations follow completed inspection and executed evidence. They are **proposals only**: this audit does not authorize or implement application source changes. Preserve existing workspace/checkpoint formats and functionality while addressing the demonstrated failures.

Effort and regression risk below are provisional engineering estimates, not measured delivery commitments. S=localized change; M=several coordinated paths; L=storage/transaction design or wide compatibility work. Scientific risk reduction takes priority over visual polish or feature expansion.

## Immediate critical fixes

| Rank | Work / findings | Scientific risk reduction | User impact | Effort | Regression risk | Acceptance gate |
|---|---|---|---|---|---|---|
| 1 | Complete coordinate contract across save/load, helper/crop and copy (A01/A02/A04) | Very high: prevent silent physical displacement | All transformed/ROI workflows | M | Medium: older manifests and helper navigation | A01/A02/A04 pass; plain scale/translation projects unchanged; full basis/corner coordinate roundtrips exact. Explicitly reject unsupported transforms until implemented. |
| 2 | Validate current backing-store identity before every save (A05) | Very high: persist the actual displayed scientific data | Any layer.data replacement with a different Zarr source | S–M | Medium: remembered assets and Save As | Zarr A→B, same-store different-array and NumPy replacement cases reopen exact; portable/multiscale tests retain behavior. |
| 3 | Range-safe integer copy policy shared by small/tiled paths (A03) | Very high: preserve label/class identities | Especially mixed dtype and large-label projects | M | Medium: writable Zarr dtype cannot simply be changed | No silent narrowing, signed/unsigned float promotion or partial writes; supported values exact, unsupported targets rejected before mutation. |
| 4 | Protect conflicting curated destination pixels (A19) | High: prevent loss of accepted annotations | Common curation workflow | M | Medium: partial-write and undo handling | Conflict summary/explicit choice and bounded journal; no extra confirmation on conflict-free copies; recovery preserves subsequent edits. |

Do not solve these by broadly deleting transforms, coercing all arrays to a small dtype, automatically relabeling biological classes, resampling accepted masks, or materializing huge stores. Targeted invariants and reversible preflight are sufficient starting points.

## High-priority reliability improvements

| Rank | Work / findings | Scientific risk reduction | User impact | Effort | Regression risk | Acceptance gate |
|---|---|---|---|---|---|---|
| 5 | Share pair/coverage/provenance validation (A07/A20) | High: image/truth registration and prediction review | Single and multiple-image training | S–M | Low–medium | Same prerequisites enforced across Current pair, Manage pairs and Manage channels; aligned/manual ground truth remains frictionless. |
| 6 | Add inference memory preflight and scope-specific limits (A15) | High: avoid OOM and interrupted workflows | Large images and many channels | S initially | Low | Budget includes probabilities, weights, output and known temporary needs; oversized tasks stop before allocation and explain usable ROI scope. |
| 7 | Stage/preflight loading with prior-context recovery (A06) | High recovery benefit | Missing/relocated/partially available projects | M | Medium | Missing all assets does not discard previous live state; partial projects remain clearly incomplete and cannot overwrite the original manifest. |
| 8 | Version content-based normal-run dataset identity (A08) | Medium–high: trustworthy comparisons/provenance | Model history and ranking | M | Medium: legacy identity interpretation | Intensity-only changes alter identity; source, class, preprocessing and split contracts included; old IDs retained as legacy, not reinterpreted silently. |
| 9 | Specify consistent topology semantics (A10) | Medium: reliable hole interpretation | Topology-driven curation | S–M | Medium: scientific output definition changes | Explicit foreground/background connectivity; independent diagonal/seam fixtures; old exported results remain interpretable. |
| 10 | Compatibility/integrity automation | High: prevent reintroductions | Every release | M | Low in product, medium in test maintenance | Existing tests plus independent regression assertions in supported CI; minimum/current napari and Zarr 2/3; no publication as stable based on one stack. |

The independent scripts deliberately live outside the default tests. A future authorized fix should promote suitable probes into the normal suite with proper fixtures rather than merely editing expected values until they pass. A10 needs an explicitly chosen scientific convention; A15 tests a limitation, not a behavior to preserve indefinitely.

## Workflow and UX simplifications

1. Keep the compact Dataset page and off-panel source/channel editors. Use existing direct channel naming and mask-layer shortcut; do not duplicate the table on the main panel.
2. Show a short current-state summary using existing metadata: image/target association, coverage, output mode/class names, dataset split and active checkpoint. Make raw/predicted/reviewed roles evident without demanding additional memorized names.
3. Put actionable validation messages beside the invalid source/coverage row; preserve current helpful unknown/partial/complete explanations. Newly assigned masks should not silently imply checked negative pixels.
4. Explain that writable Zarr edits can be durable before project Save and that JSON backup is not a full mask snapshot. Show recovery scope for stroke Undo vs last-deletion Undo.
5. Retain cancellation and progress controls, but measure real responsiveness; button presence is not evidence of timely cancellation. Use real-desktop research tasks to determine which panel changes reduce errors/clicks.

These changes have lower implementation effort and lower regression exposure than a new all-purpose wizard, additional permanent panels, or an interface rewrite.

## Longer-term work, justified by further measurement

| Candidate | Trigger/evidence required | Scope and caution |
|---|---|---|
| Disk-backed tiled inference outputs | A15 proves output RAM scaling; demonstrate demand after memory preflight | Stream probability/weight accumulation to chunked stores, preserve overlap and per-channel semantics, support recovery/cancel. Higher effort and I/O/precision regression risk. |
| Bounded component metadata/Euler improvements | Controlled fragmented-mask runtime/RSS/disk benchmark | Optimize global-count temporary arrays and metadata only if component-count measurements show limits. Pixel tiling alone does not bound fragment metadata. |
| Stronger asset transaction/recovery design | Failure injection across multi-asset save/load | Stage assets/manifests, track publication, retain recoverable original context; do not promise immutable mask history from JSON backups. |
| Typed project/source contracts | Defect rate or continued validator duplication after narrow fixes | Replace informal cross-widget state incrementally. No evidence here requires replacing working numerical services or Qt. |
| Reproducibility controls | GPU precision smoke observation plus repeated real training study | Record device/library/precision flags and define tolerances; optional strict mode if needed. Do not promise universal bitwise CPU/GPU equality. |
| Specimen/duplicate provenance assistance | A17 manual-group limitation and real acquisition workflow study | Warn about exact duplicates and incomplete group metadata; retain researcher-defined biological grouping. Content hashes cannot identify all related specimens. |

## Release re-evaluation checklist

Resolve the five P0 conditions and protect curated conflicts, rerun unchanged existing tests plus the independent assertions, document accepted connectivity/device/memory conventions, perform the stated compatibility and real-desktop checks, and independently inspect the fixes. Preserve old workspaces/checkpoints through versioned optional metadata and explicit migration tests. A clean synthetic pipeline remains a mechanics gate, not evidence of biological generalization.

Current conclusion remains **Requires critical fixes before reliable scientific use**. After those gates, controlled research testing is an appropriate next stage; broader scientific-use claims need the intended workflow/version/data evidence.
