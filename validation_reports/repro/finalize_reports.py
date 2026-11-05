"""Generate report tables from executed evidence, never assumed results."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'validation_reports'
E=REPORT/'evidence'
SHA='cb95a5e97717aca24db658c488fd810647955bcd'


def read(name): return json.loads((E/name).read_text())


def suite(name):
    tree=ET.parse(E/f'{name}_suite.xml')
    cases=tree.findall('.//testcase')
    failed=sum(c.find('failure') is not None for c in cases)
    errors=sum(c.find('error') is not None for c in cases)
    skipped=sum(c.find('skipped') is not None for c in cases)
    elapsed=sum(float(s.get('time',0)) for s in tree.findall('.//testsuite'))
    return dict(name=name,collected=len(cases),passed=len(cases)-failed-errors-skipped,
                failed=failed,errors=errors,skipped=skipped,seconds=elapsed)


suites=[suite(name) for name in ('existing','independent','supplemental','synthetic','cuda')]
(E/'all_suite_summary.json').write_text(json.dumps(suites,indent=2))
env=read('environment.json');perf=read('performance.json');cuda=read('cuda_smoke.json')
synthetics=[read(f'synthetic_{mode}.json') for mode in ('binary','multiclass','multilabel')]

scientific='''# Scientific and computational validation

Reviewed SHA: `SHA_TOKEN`. Date: 2026-11-03. **Verified** results below were executed using synthetic data. Values are from committed-source imports in the working checkout, with original tracked files preserved. Evidence includes JSON observations, full logs and JUnit XML, not screenshots alone.

## Numerical and integrity probes

| Probe | Expected / independently specified reference | Observed | Outcome |
|---|---|---|---|
| A01 saved prediction transform | Same world coordinates and pixels | Origin changes from (13.90063,57.24955) to (0,0); pixels exact | FAIL spatial integrity |
| A02 local helper alignment | Source/helper world coordinates match at pixel basis points | Only anchor matches; rotated/sheared axes differ | FAIL geometry |
| A03 tiled integer copy | 300 preserved or rejected before writes | 44 in uint8 target, reported completion | FAIL value integrity |
| A04 destination compatibility | Shifted target rejected or explicitly mapped | Twelve pixels copied into a target translated (20,30) | FAIL alignment |
| A05 current Zarr binding | Reopen replacement pixels all 9 | Reopens old store, all 2 | FAIL saved-data identity |
| A06 all-missing open | Prior layers retained/recoverable | Live viewer cleared; requested layer skipped | FAIL rollback requirement; UI clear confirmation exists |
| A07 single-pair registration | Shifted truth rejected | Dataset ready with truth offset (9,11) | FAIL prerequisite |
| A08 raw-data identity | Intensity changes alter content identity | Snapshot IDs identical | FAIL provenance requirement |
| A09 component oracle | Pure Python 4-neighbor BFS: four components, areas 1,1,28,44; exact centroids/half-open bboxes | Small and tiled paths match every record, including disconnected common ID 2^40+9 and borders | PASS |
| A10 topology convention | Dual of 4-foreground: diagonal opening has Euler 1 | App Euler 0 using 8-foreground | FAIL consistency requirement; valid alternate Euler convention |
| A11 metric TP=2, FP=1, FN=1 | Precision=recall=Dice=2/3; IoU=1/2 | Exact agreement | PASS |
| A11 empty/empty | App's declared-by-code convention: all zero | All zero | PASS arithmetic; document convention |
| A11 complete miss / perfect | (0,3,2) -> all zero; (7,0,0) -> all one | Exact agreement | PASS |
| A12 unknown channel pixels | Zero loss gradient for every ignored pixel; nonzero known gradients | Matches | PASS |
| A13 augmentation | Image/truth/valid geometry identical; source arrays unchanged | Exact equality | PASS |
| A14 overlap/edges/ROI | Logit ln(4) -> probability 0.8 everywhere; 43×97 full image; ROI origin (3,5), shape 37×86; cancellation | Matches within 1e-6, cancellation raised between tiles | PASS; constant network only |
| A15 allocation guard | Intercept allocation without making huge arrays | First request 40 GB for binary probability; weights additionally image-sized | PASS diagnostic, establishes unsafe scale |
| A16 grid edges | 4×6 cells map (0,0),(3.999,5.999),(4,6),(10,14) -> R00C00,R00C00,R01C01,R02C02 | Exact agreement | PASS |
| A16 deletion recovery | Restore original large ID except later nonzero 77 | Restored 14 pixels, skipped one; ID 2^40+9 exact | PASS |
| A17 grouped split | Supplied image/group IDs never cross splits | Disjoint IDs, but byte-identical clones accepted in different groups | PASS grouping; biological independence not inferred |
| A18 ordinary persistence | uint16 intensities and uint64 labels including 2^53+1 exact after portable save | Exact dimensions/dtype/pixels, SHA256 recorded | PASS pixels; complete affine not covered by this probe |
| A19 curated collision | Protect or recover overwritten curated IDs | 9 replaced by 7; plugin deletion undo disabled | FAIL recovery requirement |
| A20 prediction provenance | Same review confirmation rules in all dataset modes | Single-pair prediction accepted without explicit confirmation | FAIL safeguard consistency |
| A21 invalid checkpoints | Corrupt bytes, unsupported version and incompatible channel contract rejected | All rejected; exception messages recorded | PASS rejection; raw error wording can be technical |

Per-probe arrays, values and statuses: [independent observations](evidence/independent_observations/). Expected-failure assertions are not xfailed or hidden. A15/A17 are successful diagnostic probes identifying a limitation, not endorsements of safe full-scale inference or specimen independence.

## Reproducible synthetic axon/myelin-like pipeline

`repro/test_synthetic_workflow.py` generates three 96×128 uint16 images from elliptical interiors and annular myelin-like masks with varying sizes/aspect ratios/intensities and independent seeded noise. In exclusive mode, axon=1/myelin=2/background=0; binary merges foreground. Multilabel uses overlapping whole-region and axon masks, with the first eight rows unknown per channel. Class meanings differ intentionally by mode and are named explicitly.

Each case creates real napari Image/Labels objects, selects/copies one independently indexed component into an isolated curated array, portable-saves, reopens, checks every source pixel, prepares pooled dataset patches, verifies split IDs are disjoint, trains a width-4 custom U-Net for two CPU epochs, saves/reloads the checkpoint, runs entire-image tiled inference, independently recomputes reserved-patch Dice/IoU from predictions, creates output layer(s) through the actual prediction handler, saves again and verifies every final pixel exactly. Source IDs/checksums and class contracts are retained in JSON.

This is a service/Qt-assisted pipeline, not a completed human annotation session. Native brush/erase/local commits and reviewed-cell GUI actions were exercised separately by the existing suite; the synthetic pipeline supplies known coverage masks programmatically rather than certifying a human review. Model-library reference relinking/relocated projects were exercised by existing tests, while this pipeline directly reloads checkpoint files. These distinctions avoid overstating one integrated test's scope.

All three images share known object geometry and differ in noise. Their group IDs are controlled test inputs, not proof of independent biological specimens. Full-image scores below are on an already used image (resubstitution), and held-out scores are validation-selected. **Neither measures real microscopy generalization.** Success means mechanics and calculations passed, not that two epochs produced useful masks.

| Mode | Train / validation patches | Steps | Retained validation Dice / IoU | Independent reserved-patch Dice / IoU | CPU checkpoint max probability difference | Pixel roundtrips |
|---|---:|---:|---|---|---:|---|
'''.replace('SHA_TOKEN',SHA)
for item in synthetics:
    metrics=item['retained_validation']; independent=item['independent_reserved_patch_metrics']
    scientific+=f"| {item['mode']} | {item['patch_summary']['training']['total']} / {item['patch_summary']['validation']['total']} | {item['training_steps']} | {metrics['dice']:.8f} / {metrics['iou']:.8f} | {independent['dice']:.8f} / {independent['iou']:.8f} | {item['model_reload_max_absolute_difference']:.3g} | Exact |\n"
scientific+='''
Evidence: [binary](evidence/synthetic_binary.json), [multiclass](evidence/synthetic_multiclass.json), [multilabel](evidence/synthetic_multilabel.json), [test log](evidence/synthetic_suite.log). Per-class TP/FP/FN, full-image resubstitution metrics, hashes and settings are stored rather than selectively presenting an appealing score. All checkpoint prediction comparisons used atol=1e-7, rtol=0 on CPU.

## GPU numerical smoke check

'''
scientific+=f"""Real device: {cuda['device']}. One 32×32 forward/backward/AdamW step completed with finite loss {cuda['loss']:.8f}. Repeated GPU output maximum difference: {cuda['gpu_repeat_max_abs']}; same-GPU checkpoint reload difference: {cuda['same_gpu_reload_max_abs']}. The default GPU-vs-CPU logit difference was {cuda['gpu_cpu_reload_max_abs']:.8f}; it did **not** meet atol=rtol=0.0003. With TF32 disabled for inference, maximum difference was {cuda['strict_gpu_cpu_max_abs']:.8f}, which met that tolerance. The default comparison changed {cuda['default_device_threshold_flips']} zero-logit threshold decisions in this fixture.

The initial strict default cross-device test failed and is retained in [its original log](evidence/cuda_initial_cross_device_failure.log). The diagnostic rerun keeps that observation and tests separate same-device and high-precision contracts; it does not silently relax the failed default-device claim. Default cuDNN allow_tf32={cuda['default_cudnn_allow_tf32']}; matmul allow_tf32={cuda['default_matmul_allow_tf32']}. This is expected precision sensitivity to investigate/document, not evidence of a damaged checkpoint. A single successful repeat does not prove deterministic long training. [Diagnostics](evidence/cuda_smoke.json).

## Safe large-image and runtime measurements

Fresh subprocess per case; synthetic sparse rectangular components; tile size 256. Operation time excludes import/source construction; total time includes them. Process peak RSS is Linux ru_maxrss (includes Python, Qt/napari/numerical/PyTorch imports), not isolated array memory. It may differ slightly from psutil's RSS snapshot because the counters are not sampled identically. No repeated-trial confidence interval was measured.

| Workload | Components / tile scope | Operation seconds | Cold total seconds | Process peak RSS MiB | Stored component cache / result |
|---|---|---:|---:|---:|---|
"""
for item in perf:
    label='×'.join(map(str,item.get('shape',item.get('logical_shape',[]))))
    detail=str(item.get('components', '512×512 edge read' if item['kind']=='logical100k' else f"{item.get('tiles')} tiles, constant network"))
    output=f"{item['cache_bytes']} bytes cache" if 'cache_bytes' in item else f"{item['tile_bytes']} bytes tile; {item['stored_bytes']} bytes compressed store" if 'tile_bytes' in item else f"{item['probability_bytes']} bytes probability"
    scientific+=f"| {item['kind']} {label} | {detail} | {item['operation_seconds']:.6f} | {item['elapsed_seconds']:.6f} | {item['peak_process_rss_bytes']/2**20:.2f} | {output} |\n"
scientific+='''
The logical 100k store is sparse, fill-zero and unusually compressible: only a small known high-ID rectangle is written. Its 40-GB logical uint32 array is never materialized. A 1-MiB boundary tile read with the production local-read helper returned exact values. This does not benchmark realistic densely annotated storage, remote access, full image rendering, painting, component scanning or U-Net inference at 100k.

For the measured low-fragmentation component cases, four times as many pixels took approximately four times the operation time, with component counts rising similarly. This is one favorable scaling example, not a bound for checkerboards or millions of fragments. Metadata and provisional IDs scale with components; temporary disk demand depends on tile compressibility. Prediction benchmark uses a constant network and tests tile/output mechanics only, not realistic network throughput.

The guarded allocation test establishes 80 GB binary / 200 GB four-channel accumulator lower bounds at 100k², without allocating those buffers. Bounded source reading alone does not imply bounded output memory. Full-scale GPU memory, interactive event latency, cancellation latency during slow tile writes, leaks over many cycles, and network-storage performance remain unverified. [Raw performance measurements](evidence/performance.json).

## Conclusion of the numerical work

Core component/metric/supervision mathematics passes the exercised fixtures, but the scientific acceptance criterion includes coordinates and identities as well as array bytes. A01–A05 fail that criterion under their specified conditions. Topology conventions, validation selection and specimen provenance need explicit interpretation. No synthetic result is presented as biological validation.
'''
(REPORT/'05_scientific_validation.md').write_text(scientific)

testing=f'''# Testing summary and remaining validation

Reviewed SHA: `{SHA}`. Execution date 2026-11-03. All source imports used checkout `src`; original tests ran unchanged. Audit tests were added only under `validation_reports/repro/` and remain outside the configured default `tests/` collection.

## Executed final suites

| Suite | Collected/executed | Passed | Failed | Errors | Skipped | JUnit seconds | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
'''
for item in suites:
    testing+=f"| {item['name']} | {item['collected']} | {item['passed']} | {item['failed']} | {item['errors']} | {item['skipped']} | {item['seconds']:.3f} | [{item['name']} log](evidence/{item['name']}_suite.log), [XML](evidence/{item['name']}_suite.xml) |\n"
testing+='''
The existing suite collected and passed **289 cases**, with no failures/errors/skips. Its pytest console reported **139.80 seconds**; wrapper wall time including startup/collection was **143.96 seconds**. There were 258 source test functions before parametrization, not 258 executed cases. Independent final suites comprise **30 cases: 19 passed, 11 failed**, with no errors/skips. The eleven failures intentionally expose unmet acceptance/safety requirements: A01–A08, A10, A19, A20. They were not repaired, xfailed or omitted. Combined final suite executions are 319 cases: 308 passed and 11 failed; keeping existing vs independent totals separate is more informative than that aggregate.

Exploratory history is preserved: the first synthetic workflow harness passed the wrong component-copy argument order (three harness failures), then the harness alone was corrected and all three modes passed. The first benchmark used an incorrect controller class name, then was corrected; both original errors remain in evidence. The initial default GPU/CPU tolerance assertion failed; diagnostics established same-device consistency and documented default cross-device precision differences instead of treating them as a checkpoint defect. These initial runs are additional historical executions, not included in final-suite totals or application-defect counts. See `synthetic_harness_initial_error.log`, `performance_initial_harness_error.json`, and `cuda_initial_cross_device_failure.*`.

## Environment

'''
testing+=f"""- Python `{env['python'].splitlines()[0]}`, Linux aarch64 `{env['platform']}`, {env['logical_cpus']} logical CPUs; approximately {int(env['meminfo'][0].split()[1])/1024**2:.1f} GiB reported system RAM.
- napari {env['packages']['napari']}, NumPy {env['packages']['numpy']}, SciPy {env['packages']['scipy']}, scikit-image {env['packages']['scikit-image']}, Zarr {env['packages']['zarr']}.
- PyTorch {env['packages']['torch']}, CUDA {env['torch_cuda_version']}; {', '.join(env['cuda_devices'])}; {env['qt_api']} Qt {env['qt_version']}.
- pytest {env['packages']['pytest']}; pytest-qt {env['packages']['pytest-qt']} available through the pre-existing `/tmp/label-assistant-test-deps` path. No dependency installation was needed for this audit.
- Installed distribution metadata reports napari-label-assistant 3.3.0, while the reviewed pyproject declares 4.3.0. Tests explicitly import the checkout, so installed-version metadata is not the reviewed source identity.
- Qt `offscreen`; XDG config/cache and Matplotlib config isolated under `/tmp/nla-audit-cb95a5e`; OMP/MKL threads=2; bytecode/cacheprovider writes disabled; disposable pytest basetemps. Test package/plugin settings do not target user production projects.

Full inventory: [environment.json](evidence/environment.json), [repository hashes and AST](evidence/repository_inventory.json). This environment is powerful hardware; small tests do not establish ordinary laptop performance.

"""
testing+='''## Warnings and operational limits

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
'''
(REPORT/'06_testing_summary.md').write_text(testing)

commands='''# Commands and reproducibility

Reviewed SHA: `SHA_TOKEN`. Commands below were executed from the repository root. Paths identify this machine; use the equivalent environment when reproducing elsewhere. Outputs are stored under `validation_reports/evidence/`. No git commit/push, source repair, dependency installation, or production-dataset mutation was performed.

```bash
git rev-parse HEAD
git status --short
git ls-files
git ls-remote --heads origin main
```

Tracked source/docs/tests were inspected using `rg`, `rg --files`, `sed`, AST indexing and SHA256 hashing. README, CHANGELOG, pyproject, plugin manifest, docs and test definitions are included in the inventory. Function-level references are in source_index.md; targeted `sed` reads examined the high-risk branches. Public GitHub landing-page content was also opened, but exact revision validation uses Git rather than a cached web-page version.

Common executed test environment:

```bash
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH=/tmp/label-assistant-test-deps:src
export PYTHONDONTWRITEBYTECODE=1
export XDG_CONFIG_HOME=/tmp/nla-audit-cb95a5e/config
export XDG_CACHE_HOME=/tmp/nla-audit-cb95a5e/cache
export MPLCONFIGDIR=/tmp/nla-audit-cb95a5e/mpl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
AUDIT_PYTHON=/home/wteox/miniforge3/envs/napari-dev2/bin/python

"$AUDIT_PYTHON" validation_reports/repro/collect_inventory.py
"$AUDIT_PYTHON" validation_reports/repro/run_existing_suite.py

"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_independent_audit.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/independent-tests --junitxml=validation_reports/evidence/independent_suite.xml
"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_synthetic_workflow.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/synthetic-tests-final --junitxml=validation_reports/evidence/synthetic_suite.xml
"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_supplemental_audit.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/supplemental-tests --junitxml=validation_reports/evidence/supplemental_suite.xml
"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_cuda_smoke.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/cuda-tests-diagnostics --junitxml=validation_reports/evidence/cuda_suite.xml
"$AUDIT_PYTHON" validation_reports/repro/benchmark.py
"$AUDIT_PYTHON" validation_reports/repro/finalize_reports.py
```

The wrapper also executed `python -m pytest tests --collect-only -q -p no:cacheprovider` and the full existing command recorded in existing_suite_summary.json (including `--durations=20`). stdout/stderr were redirected/captured into the corresponding logs. The independent/supplemental suites are expected to return exit code 1 on the reviewed source; that is audit evidence, not a command-execution failure to hide.

Historical runs: synthetic initial `--basetemp=.../synthetic-tests`; GPU initial `--basetemp=.../cuda-tests`; benchmark initial controller import failure. The audit harness was corrected, application source was not. Preserved logs identify each failure. Performance was repeated after separating import time from operation time; published measurements use the final script/output.

Re-run individual findings with `-k A01`, `-k A03`, etc. Run all added probes with `python -m pytest validation_reports/repro -q -p no:cacheprovider --basetemp=/tmp/nla-audit-rerun`; use a new basetemp if preserving original disposable test files. Existing test selection remains `tests/`, so future default test runs will not automatically include these independent regression proposals.

Array checksums and final tracked-file verification are in repository_integrity.json. No coverage-instrumentation, build/wheel install, real-desktop user study or massive dense allocation command was executed.
'''.replace('SHA_TOKEN',SHA)
(E/'commands.md').write_text(commands)

original=read('repository_inventory.json')
changed=[]
for record in original['files']:
    path=ROOT/record['path']
    actual=sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual!=record['sha256']:changed.append(dict(path=record['path'],before=record['sha256'],after=actual))
integrity=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),reviewed_commit=SHA,
               current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               tracked_files_checked=len(original['files']),changed_tracked_files=changed,
               git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
               tracked_diff=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True))
(E/'repository_integrity.json').write_text(json.dumps(integrity,indent=2))
assert not changed,changed
print(json.dumps(dict(suites=suites,tracked_files_unchanged=len(original['files'])),indent=2))
