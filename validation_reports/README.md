# Independent software and scientific audit

**Current release: 5.3.0 (2026-11-05).** The final release working tree passed
**353 tests**, with 0 failures/skips and 353 warnings, in 250.88 seconds
(Qt offscreen). See [release validation](11_release_5_3_0.md). Initial audit
scores and failures below remain historical; follow-up reports record the fixes.

Reviewed GitHub repository: [wulinteousa2-hash/napari-label-assistant](https://github.com/wulinteousa2-hash/napari-label-assistant). Exact reviewed/local/remote-main SHA during execution: `cb95a5e97717aca24db658c488fd810647955bcd`. Audit date: **2026-11-03**. Source declares version **4.3.0**; multilabel work is documented under Unreleased.

**Authorized follow-up, 2026-11-04:** A01/A02/A04 have now been patched in the
working tree. See [coordinate patch validation](08_coordinate_patch_validation.md).
The initial audit reports, counts, and evidence below describe the original
reviewed commit and are preserved as historical findings.

**Authorized follow-up, 2026-11-05:** Managed grid-mask loading is implemented;
see [grid-mask validation and memory measurement](09_grid_mask_loading_validation.md).

**Authorized follow-up, 2026-11-05:** Manual copy destinations now support
confirmation, automatic matching and retained originals. See
[copy-destination validation](10_copy_destination_validation.md).

**Recommendation: Requires critical fixes before reliable scientific use.** Five P0 conditions were reproduced involving saved spatial transforms, local edit geometry, component-copy dtype/alignment, and stale Zarr storage bindings. Ordinary paths also passed substantial independent checks; findings specify their triggering conditions rather than alleging universal corruption.

Existing suite: **289 passed, 0 failed/skipped**. Final independent suites: **19 passed, 11 failed** across 30 cases. Three CPU synthetic workflows and a real GPU smoke check passed their defined final contracts; an exploratory default CPU/GPU precision check failed and is explicitly retained/documented. No application source or existing tests were changed, and no Git commit/push was performed. Real user datasets were not used.

## Reports

| Deliverable | Report |
|---|---|
| A: Executive assessment, provisional scores and G: release recommendation | [00_executive_assessment.md](00_executive_assessment.md) |
| Phase 1: Architecture/feature inventory, interactions, implementation and tests | [01_feature_inventory.md](01_feature_inventory.md) |
| B: Prioritized technical findings, reproduction/evidence/correction/regression | [02_technical_findings.md](02_technical_findings.md) |
| C: Ten consequential UX issues and first-use walkthrough | [03_ux_findings.md](03_ux_findings.md) |
| Architecture, state, scientific and ML contracts / unsupported scope | [04_architecture_and_scientific_scope.md](04_architecture_and_scientific_scope.md) |
| D: Numerical correctness, synthetic end-to-end workflows and performance | [05_scientific_validation.md](05_scientific_validation.md) |
| E: Executed counts/times/warnings, coverage gaps and manual requirements | [06_testing_summary.md](06_testing_summary.md) |
| F: Risk-ranked improvements and release gates | [07_improvement_roadmap.md](07_improvement_roadmap.md) |

## Reproducibility and scope

Start with [executed commands](evidence/commands.md), [environment](evidence/environment.json), [suite summary](evidence/all_suite_summary.json), [tracked-file integrity](evidence/repository_integrity.json), and [full source/test inventory](evidence/source_index.md). Raw JSON observations, logs and JUnit XML are under [evidence/](evidence/). Reproduction code is under [repro/](repro/); its expected failures are findings, not unfinished application repairs.

**Verified** = executed observation; **Code-reviewed** = inspected behavior without independent execution; **Unverified** = blocked/not performed and accompanied by a follow-up procedure. Qt controls/layer objects were tested offscreen; this is not a human usability study or rendered-overlay evaluation. Sparse logical 100k access is not a full 100k performance demonstration. Toy axon/myelin-like data tests pipeline mechanics and cannot establish biological accuracy or generalization.

The reports propose targeted corrections; they do not implement them. Earlier requests to publish features do not override this audit's explicit instruction against source modification or committing.
