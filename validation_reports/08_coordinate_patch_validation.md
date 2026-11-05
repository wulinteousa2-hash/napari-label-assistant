# Coordinate contract patch: A01, A02 and A04

Date: 2026-11-04. User-authorized source changes on base commit
`cb95a5e97717aca24db658c488fd810647955bcd`. This is an uncommitted working-tree
patch. The original audit evidence remains unchanged; its failures describe the
pre-patch commit.

## Implemented changes

- **A01:** Workspace manifests now retain separate rotation, shear and extra
  affine matrices alongside scale/translation. Optional fields keep old manifests
  readable and avoid applying scale/translation twice. Invalid coordinate state
  is reported and its newly created layer removed instead of leaving a mask
  displayed at a default mapping. Native 3D transforms are preserved too.
- **A02:** Local edit areas use the complete source transform with the crop's
  exact world-space displacement. Boundaries use the same source mapping.
  Source-transform events update existing helpers, and teardown disconnects the
  listeners. Paint and erase still write to original full-resolution indices.
- **A04:** Compatible component-copy destinations must match shape and spatial
  registration. Both the UI path and tiled controller reject incompatible
  transforms before mutation. Manually created destinations inherit the complete
  transform. Combine validates registration and preserves its reference mapping.
- `_coordinates.py` supplies shared mapping, crop and compatibility functions.
  It composes public napari Affine properties analytically, avoiding subtraction
  error at large world origins. Compatibility is measured over mask corners
  within 1e-6 source pixels, not a relative tolerance on large world coordinates.
- Singleton-axis masks can be projected only when their mapping remains in the
  viewer's final two world axes; otherwise the operation explains that aligned
  2D layers are needed. No implicit resampling or axis relocation is performed.
- Cropped prediction creation uses the same shared contract. README and the
  Unreleased changelog describe the updated behavior.

## Verification

| Executed check | Result | Evidence |
|---|---|---|
| Full existing suite plus new regressions | 316 passed, no failures/errors/skips; 188.10 seconds | [Full log](evidence/coordinate_patch/full_suite.log), [XML](evidence/coordinate_patch/full_suite.xml) |
| Final focused coordinate suite | 27 passed, no failures/errors/skips | [Coordinate log](evidence/coordinate_patch/coordinate_regressions.log), [XML](evidence/coordinate_patch/coordinate_regressions.xml) |
| Original independent A01/A02/A04 audit assertions | 3 passed, 18 deselected | [Original probes log](evidence/coordinate_patch/original_A01_A02_A04.log), [XML](evidence/coordinate_patch/original_A01_A02_A04.xml) |
| Patch whitespace and new-module parsing | Passed | `git diff --check`; `ast.parse` on new Python modules |

The first targeted run had 71 passes and one failure in an old positive-copy
test whose destination had identity coordinates while its source was scaled and
translated. That destination fixture now has matching source coordinates.
New negative tests explicitly verify rejection; the safety rule was not relaxed.
The first expanded full run passed 305 tests; additional invalid-transform,
singleton-plane and direct-controller checks bring the final full total to 316.
The final focused run covers the last singleton-plane guard adjustment.

Cases include identity, anisotropic scale, translation, rotation, shear, separate
affine, combined transforms, reflection, nonzero crop offsets, repeated save/load,
exact uint64 pixels, legacy records, native 3D layers, malformed coordinate state,
live transform changes, actual local paint/erase, small/tiled copying, explicit
misaligned destinations, newly created destinations, equivalent transform
decompositions, small physical pixel sizes, and large world origins.

Tests use the existing napari-dev2 environment with Qt offscreen, isolated
config/cache and disposable `/tmp/nla-coordinate-patch` data. Offscreen testing
verifies layer geometry and controls, not a human visual usability study. Existing
Pydantic and napari deprecation warnings remain; `ruff` was unavailable, so no
lint success is claimed. No new dependency was installed.

## Reproduction

Use the environment exports in [the original commands](evidence/commands.md),
substituting isolated `/tmp/nla-coordinate-patch` config/cache paths, then run:

```bash
python -m pytest tests -q -p no:cacheprovider --basetemp=/tmp/nla-coordinate-patch/reproduce-full
python -m pytest tests/test_coordinate_contract.py -q -p no:cacheprovider --basetemp=/tmp/nla-coordinate-patch/reproduce-coordinates
```

[run_original_probes.py](evidence/coordinate_patch/run_original_probes.py)
redirects the original audit assertions' JSON observations into a disposable
directory so the pre-patch evidence is not overwritten. The actual executable
was `/home/wteox/miniforge3/envs/napari-dev2/bin/python`, with
`PYTHONPATH=/tmp/label-assistant-test-deps:src`, `QT_QPA_PLATFORM=offscreen`,
OMP/MKL threads=2, and bytecode/cacheprovider writes disabled.
[summary.json](evidence/coordinate_patch/summary.json) records exact suite totals,
times, base commit and changed-file hashes.

## Scope and remaining limits

This patch prevents future loss of accepted transforms. It cannot reconstruct
rotation/shear/affine values already absent from an older saved manifest.
It does not change mask pixels or interpolate labels to force registration.
A03 (tiled-copy integer overflow), A05 (stale replacement Zarr bindings), and the
other audit findings still need their own fixes; this patch alone does not
establish overall release readiness. Production datasets were not opened or
modified. Nothing was committed or pushed.
