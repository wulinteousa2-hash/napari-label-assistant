# Executive assessment and release recommendation

Reviewed repository: [napari-label-assistant](https://github.com/wulinteousa2-hash/napari-label-assistant). Reviewed commit: `cb95a5e97717aca24db658c488fd810647955bcd`; `git ls-remote --heads origin main` returned that same SHA during the audit. Date: 2026-11-03. Declared source version: 4.3.0, with overlapping-channel work documented as Unreleased. Audit source and tests were imported from this checkout, not inferred from installed package metadata.

**Release recommendation: Requires critical fixes before reliable scientific use.** Five reproduced paths can silently change coordinates or label values, or save the wrong backing data. These are release blockers even though all 289 existing tests pass. This is not a claim that every existing annotation/project is corrupt; each finding specifies its triggering conditions.

## What the evidence supports

The project has substantial working functionality: ordinary exact pixel round trips, component measurements, bounded local reads, channel-specific unknown-pixel masking, grouped splits, three segmentation modes, real CPU training and checkpoint reloads. Independent binary, multiclass and overlapping multilabel synthetic workflows completed successfully. Existing tests also exercised native label painting, review persistence, model management, missing-model handling and continuation.

The strongest weaknesses are at boundaries between otherwise working subsystems. Workspace persistence preserves pixel arrays but omits full coordinate transforms. Local edit helpers also omit source rotation/shear. Component copying has inconsistent integer policies and insufficient spatial compatibility checks. Replacing a Zarr source can leave stale saved storage metadata. These are precisely cases that end-user visual checks and passing implementation-oriented tests can miss.

Full-scale readiness is not established. The 100k claim is qualified in README as local, chunked access dependent on hardware. That access pattern worked in one sparse synthetic test. It is insufficient evidence of responsive rendering, exhaustive component analysis or full-image U-Net inference at that scale. The latter currently has an image-sized output-memory requirement despite bounded source reads.

## Provisional scores

Scores are audit judgments, not statistical estimates or certifications. Rubric: 0–2 = unsuitable core behavior; 3–4 = demonstrated integrity problems; 5–6 = useful routine behavior with material gaps; 7–8 = broad adversarial evidence and recovery; 9–10 = strong version/hardware coverage, independent end-to-end evidence and well-established limits. No category receives a high score from unit-test counts alone.

| Category | Provisional score | Criteria and evidence | Confidence |
|---|---:|---|---|
| Engineering reliability | 6/10 | Clear numerical services, lazy optional ML imports, atomic manifest/checkpoint writes and 289 passing tests; inconsistent validation and tightly coupled UI/state paths caused verified defects. | Moderate, one stack |
| Scientific validity of implemented operations | 5/10 | Independent component and metric arithmetic passes; properly masked supervision and grouped splits. Spatial integrity defects, mixed topology conventions and manual specimen provenance limit dependable interpretation. No biological accuracy study. | Moderate for mechanics; biological validity unverified |
| Image and label integrity | 3/10 | Untransformed uint64 pixels and uint16 intensities round-trip exactly, but A01–A05 show silent world-coordinate, integer-value or backing-store failures. | High for reproduced conditions |
| AI training reliability | 6/10 | Binary/multiclass/multilabel synthetic training and CPU reload pass; loss/activation choices appropriate, reserved metrics independently match. A07 registration bypass and A08 weak normal-run content identity remain; real specimen generalization is unverified. | Moderate |
| 100k large-image readiness | **Unverified** | Bounded 100k tile read and up to 2048×3072 tiled-component measurements executed. No full 100k rendering/scan/end-to-end benchmark; image-sized inference buffers are verified. | Insufficient for a global score |
| UX quality | 5/10 | Guided stages, useful coverage text and direct channel naming exist. Inconsistent safety rules, copy recovery, missing-asset rollback and dense configuration tables remain. This is an expert heuristic/offscreen assessment, not a human study. | Limited |
| Release readiness | 2/10 | Scientific integrity gate fails on five verified P0 paths. Stable classifier and broad version minimums are not supported by a compatibility matrix in this review. | High for current blocker decision |

## Decision gates

Before recommending reliable scientific use, require:

1. Exact world-coordinate round trips and crop/helper alignment for all accepted transforms, or explicit rejection before editing/saving unsupported transforms.
2. Exact selected label values across small/tiled copy paths, range preflight, spatial compatibility checks, and protection/recovery for conflicting curated pixels.
3. Current backing-array identity used on every save; replaced Zarr data must reopen exactly.
4. Alignment and prediction-review rules shared across all dataset modes.
5. Published workflow-specific memory limits and a safe inference memory preflight.
6. The existing suite plus independent failing probes pass after targeted, reviewed fixes; compatibility and real-desktop follow-up gates in the testing report are completed for the intended release scope.

The safe immediate direction is targeted reliability work, not adding architectures, new panels, or a wholesale rewrite. Existing datasets were not opened or modified. No application source, existing tests, README, changelog or Git commit was changed by this audit. Reports and disposable audit reproductions live only under `validation_reports/`; transient test assets live under `/tmp`.

See [technical findings](02_technical_findings.md), [scientific evidence](05_scientific_validation.md), [testing limits](06_testing_summary.md), and the [ranked roadmap](07_improvement_roadmap.md).
