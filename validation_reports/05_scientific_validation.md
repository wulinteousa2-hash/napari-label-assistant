# Scientific and computational validation

Reviewed SHA: `cb95a5e97717aca24db658c488fd810647955bcd`. Date: 2026-11-03. **Verified** results below were executed using synthetic data. Values are from committed-source imports in the working checkout, with original tracked files preserved. Evidence includes JSON observations, full logs and JUnit XML, not screenshots alone.

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
| binary | 24 / 12 | 24 | 0.19692750 / 0.10921774 | 0.19692750 / 0.10921774 | 0 | Exact |
| multiclass | 24 / 12 | 24 | 0.07373915 / 0.03833704 | 0.07373915 / 0.03833704 | 0 | Exact |
| multilabel | 24 / 12 | 24 | 0.13973237 / 0.07570046 | 0.13973237 / 0.07570046 | 0 | Exact |

Evidence: [binary](evidence/synthetic_binary.json), [multiclass](evidence/synthetic_multiclass.json), [multilabel](evidence/synthetic_multilabel.json), [test log](evidence/synthetic_suite.log). Per-class TP/FP/FN, full-image resubstitution metrics, hashes and settings are stored rather than selectively presenting an appealing score. All checkpoint prediction comparisons used atol=1e-7, rtol=0 on CPU.

## GPU numerical smoke check

Real device: NVIDIA GB10. One 32×32 forward/backward/AdamW step completed with finite loss 0.33369640. Repeated GPU output maximum difference: 0.0; same-GPU checkpoint reload difference: 0.0. The default GPU-vs-CPU logit difference was 0.00201106; it did **not** meet atol=rtol=0.0003. With TF32 disabled for inference, maximum difference was 0.00001246, which met that tolerance. The default comparison changed 0 zero-logit threshold decisions in this fixture.

The initial strict default cross-device test failed and is retained in [its original log](evidence/cuda_initial_cross_device_failure.log). The diagnostic rerun keeps that observation and tests separate same-device and high-precision contracts; it does not silently relax the failed default-device claim. Default cuDNN allow_tf32=True; matmul allow_tf32=False. This is expected precision sensitivity to investigate/document, not evidence of a damaged checkpoint. A single successful repeat does not prove deterministic long training. [Diagnostics](evidence/cuda_smoke.json).

## Safe large-image and runtime measurements

Fresh subprocess per case; synthetic sparse rectangular components; tile size 256. Operation time excludes import/source construction; total time includes them. Process peak RSS is Linux ru_maxrss (includes Python, Qt/napari/numerical/PyTorch imports), not isolated array memory. It may differ slightly from psutil's RSS snapshot because the counters are not sampled identically. No repeated-trial confidence interval was measured.

| Workload | Components / tile scope | Operation seconds | Cold total seconds | Process peak RSS MiB | Stored component cache / result |
|---|---|---:|---:|---:|---|
| components-1 512×768 | 84 | 0.017912 | 0.782802 | 281.19 | 4128 bytes cache |
| components-2 1024×1536 | 345 | 0.068253 | 0.843225 | 281.66 | 16902 bytes cache |
| components-4 2048×3072 | 1334 | 0.264673 | 1.032979 | 292.64 | 67040 bytes cache |
| logical100k 100000×100000 | 512×512 edge read | 0.002446 | 0.786812 | 291.27 | 1048576 bytes tile; 727 bytes compressed store |
| prediction 1024×1536 | 40 tiles, constant network | 0.051703 | 1.368706 | 761.26 | 6291456 bytes probability |

The logical 100k store is sparse, fill-zero and unusually compressible: only a small known high-ID rectangle is written. Its 40-GB logical uint32 array is never materialized. A 1-MiB boundary tile read with the production local-read helper returned exact values. This does not benchmark realistic densely annotated storage, remote access, full image rendering, painting, component scanning or U-Net inference at 100k.

For the measured low-fragmentation component cases, four times as many pixels took approximately four times the operation time, with component counts rising similarly. This is one favorable scaling example, not a bound for checkerboards or millions of fragments. Metadata and provisional IDs scale with components; temporary disk demand depends on tile compressibility. Prediction benchmark uses a constant network and tests tile/output mechanics only, not realistic network throughput.

The guarded allocation test establishes 80 GB binary / 200 GB four-channel accumulator lower bounds at 100k², without allocating those buffers. Bounded source reading alone does not imply bounded output memory. Full-scale GPU memory, interactive event latency, cancellation latency during slow tile writes, leaks over many cycles, and network-storage performance remain unverified. [Raw performance measurements](evidence/performance.json).

## Conclusion of the numerical work

Core component/metric/supervision mathematics passes the exercised fixtures, but the scientific acceptance criterion includes coordinates and identities as well as array bytes. A01–A05 fail that criterion under their specified conditions. Topology conventions, validation selection and specimen provenance need explicit interpretation. No synthetic result is presented as biological validation.
