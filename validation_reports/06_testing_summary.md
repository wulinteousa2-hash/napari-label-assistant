# Testing summary and remaining validation

Reviewed SHA: `cb95a5e97717aca24db658c488fd810647955bcd`. Execution date 2026-11-03. All source imports used checkout `src`; original tests ran unchanged. Audit tests were added only under `validation_reports/repro/` and remain outside the configured default `tests/` collection.

## Executed final suites

| Suite | Collected/executed | Passed | Failed | Errors | Skipped | JUnit seconds | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| existing | 289 | 289 | 0 | 0 | 0 | 139.798 | [existing log](evidence/existing_suite.log), [XML](evidence/existing_suite.xml) |
| independent | 21 | 12 | 9 | 0 | 0 | 7.179 | [independent log](evidence/independent_suite.log), [XML](evidence/independent_suite.xml) |
| supplemental | 5 | 3 | 2 | 0 | 0 | 2.732 | [supplemental log](evidence/supplemental_suite.log), [XML](evidence/supplemental_suite.xml) |
| synthetic | 3 | 3 | 0 | 0 | 0 | 16.146 | [synthetic log](evidence/synthetic_suite.log), [XML](evidence/synthetic_suite.xml) |
| cuda | 1 | 1 | 0 | 0 | 0 | 2.640 | [cuda log](evidence/cuda_suite.log), [XML](evidence/cuda_suite.xml) |

The existing suite collected and passed **289 cases**, with no failures/errors/skips. Its pytest console reported **139.80 seconds**; wrapper wall time including startup/collection was **143.96 seconds**. There were 258 source test functions before parametrization, not 258 executed cases. Independent final suites comprise **30 cases: 19 passed, 11 failed**, with no errors/skips. The eleven failures intentionally expose unmet acceptance/safety requirements: A01–A08, A10, A19, A20. They were not repaired, xfailed or omitted. Combined final suite executions are 319 cases: 308 passed and 11 failed; keeping existing vs independent totals separate is more informative than that aggregate.

Exploratory history is preserved: the first synthetic workflow harness passed the wrong component-copy argument order (three harness failures), then the harness alone was corrected and all three modes passed. The first benchmark used an incorrect controller class name, then was corrected; both original errors remain in evidence. The initial default GPU/CPU tolerance assertion failed; diagnostics established same-device consistency and documented default cross-device precision differences instead of treating them as a checkpoint defect. These initial runs are additional historical executions, not included in final-suite totals or application-defect counts. See `synthetic_harness_initial_error.log`, `performance_initial_harness_error.json`, and `cuda_initial_cross_device_failure.*`.

## Environment

- Python `3.11.16 (main, Sep  2 2026, 23:41:26) [GCC 15.3.0]`, Linux aarch64 `Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39`, 20 logical CPUs; approximately 121.7 GiB reported system RAM.
- napari 0.9.1, NumPy 2.4.6, SciPy 1.17.1, scikit-image 0.26.0, Zarr 3.1.6.
- PyTorch 2.14.0, CUDA 13.0; NVIDIA GB10; PySide6 Qt 6.11.2.
- pytest 9.1.1; pytest-qt 4.5.0 available through the pre-existing `/tmp/label-assistant-test-deps` path. No dependency installation was needed for this audit.
- Installed distribution metadata reports napari-label-assistant 3.3.0, while the reviewed pyproject declares 4.3.0. Tests explicitly import the checkout, so installed-version metadata is not the reviewed source identity.
- Qt `offscreen`; XDG config/cache and Matplotlib config isolated under `/tmp/nla-audit-cb95a5e`; OMP/MKL threads=2; bytecode/cacheprovider writes disabled; disposable pytest basetemps. Test package/plugin settings do not target user production projects.

Full inventory: [environment.json](evidence/environment.json), [repository hashes and AST](evidence/repository_inventory.json). This environment is powerful hardware; small tests do not establish ordinary laptop performance.

## Warnings and operational limits

Existing pytest output: 294 warnings, principally Pydantic v2 `json_encoders` deprecations and napari `viewer.camera` deprecations. Independent core suite: 22 warnings; supplemental: 20; synthetic: 16; GPU: 16. Qt emitted `QOpenGLWidget is not supported on this platform` during offscreen viewer creation. No rendered-overlay quality claim follows from programmatic widget success. Warnings are retained in logs rather than silently filtered.

The execution sandbox intermittently failed before commands with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. Relevant commands were retried through the approved outside-sandbox execution mechanism. This is an execution-environment limitation, not a repository defect. It did not prevent final test completion.

## Coverage and evidence levels

| Requirement area | Executed evidence | Remaining limitation |
|---|---|---|
| Image shape/intensities/dtype | Independent uint16 raw-image and uint64-label checks; base-level workspace and synthetic pipeline checks | Arbitrary readers, RGB/3D and all pyramid downsampling cases not independently checked |
| Coordinates/crops/local editing | A01/A02/A04/A07; existing native paint, tile boundary, camera and pan tests | Real rotated-image brush/viewport experience and visual rendering unverified |
| Components/holes/disconnection | BFS oracle, large IDs, borders; existing seam/ring/random tests | Topology conventions differ; worst-case millions of fragments not tested |
| Copy/delete/undo | Real Qt copy defects; independent deletion recovery; existing small/large cleanup undo | Copy recovery absent; interrupted disk-write recovery not exhaustively tested |
| Grid/review | Independent non-divisible boundary arithmetic; existing reviewed-cell clipping/next/association/portable tests | Reopening a real desktop and manually relocating by grid not performed |
| Workspace lifecycle | Ordinary exact pixel checks, three synthetic pipelines, existing Save As/portable/missing model/backward-format tests | Affine/storage replacement failures; not every legacy project/reader/version tested |
| Binary/multiclass/multilabel | Real CPU training and metrics; per-channel unknown gradients; existing overlapping-target tests | 254-channel training feasibility, production annotation completeness and biology unverified |
| Leakage | Group-disjoint synthetic checks, clone limitation; existing prior-exposure/k-fold/component tests | No real specimen provenance audit or estimate of correlation-driven optimism |
| Model management | Checkpoint reload, corruption/version/contract rejection, existing library/relink/export tests | Real pretrained downloads/provider/version matrix not executed |
| Inference | Constant-output tiled edges/ROI/cancellation; full-image synthetic U-Net and existing all-scope tests | Context-dependent seams, massive outputs, slow I/O cancellation and real images not established |
| GPU | Real small forward/backward, repeat, checkpoint and precision diagnostics | Full training loop/device combinations and long-run deterministic repeatability unverified |
| Performance | Independent subprocess runtime/RSS, up to 2048×3072 components, sparse logical 100k edge tile | No full 100k scan/render, human responsiveness, soak/leak or interrupted operations benchmark |
| UX/accessibility | Offscreen controls, prerequisites, naming and stylesheet tests; expert code review | No human usability study, screen-reader or real-desktop high-DPI evaluation |

No code-coverage percentage was measured; test counts are not line/branch coverage. Existing tests demonstrate useful behavior, but several compare app paths sharing the same convention and cannot serve as independent scientific oracles.

## Required manual and environment follow-up

1. **Real desktop pixel placement:** On copied synthetic and microscopy images, include scale/translation/rotation/shear, paint a known coordinate, pan/zoom, crop prediction, save/restart/reopen and verify overlays plus world-coordinate logs. Blocker: offscreen OpenGL cannot establish visual behavior. Expected: exact accepted-coordinate placement or a clear refusal before edits.
2. **Interrupted writes:** On disposable NumPy/Zarr projects and checkpoints, interrupt at asset write, manifest replacement, library transaction and sidecar write. Expected: previous manifest usable, no silent partial scientific mask, explicit recovery or documented partial mutation. Blocker: no destructive process-kill matrix performed here.
3. **Storage and relocation:** Move the entire Portable Snapshot folder; separately break each image/model/library path. Expected: portable sources resolve; external missing assets named; annotations accessible when checkpoints fail; failed open recovers prior context. Existing tests cover selected cases, not every reader/store/permission combination.
4. **Version compatibility:** Clean install Python 3.10+ on the intended OSs, napari minimum/current, Zarr 2.18/current 3, supported Qt backends and optional training providers. Expected: plugin discovery, exact integrity probes, save/reopen and tiny training work. Blocker: only one installed modern environment tested; no new environments/dependencies were installed.
5. **Real ML evaluation:** Fix original specimen IDs, independently annotate an untouched test set, lock preprocessing/classes/model-selection protocol, compare against a baseline, report per-specimen/per-class metrics and uncertainty. Expected: documented generalization evidence, not selected-validation score. Blocker: no authorized biological benchmark/annotations supplied for audit.
6. **Scale/latency/soak:** Use increasing realistic Zarr sizes and fragmentation, record input/output/disk budgets, peak RSS, event-loop latency and cancellation latency over repeated cycles. Start small; never allocate 100k dense arrays just to OOM. Expected: bounded behavior for the advertised workflow or actionable limits. Blocker: no production storage/interactive workload exercised.
7. **Researcher task study:** Run the UX report's fourteen-step workflow with novice/experienced researchers; measure errors, recovery and state comprehension. Expected: identify active sources/coverage/checkpoint without memorization. Blocker: no human study was conducted.

Commands are provided in [commands.md](evidence/commands.md). These are remaining requirements, not claims that verification already occurred.
