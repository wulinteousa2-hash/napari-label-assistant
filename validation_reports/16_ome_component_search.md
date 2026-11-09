# OME-Zarr component-search validation

## Actual workspace, read-only

Inspected `/home/wteox/Works/EM_core/EM data July 10th/P2MPZ/p2mpz EM25 32.json`
and opened its curated OME-Zarr source in mode `r`. The authoritative array is
`s0`, shape 49,976 × 59,716, dtype uint8. Lower-resolution arrays are display
representations of the same mask, not separate annotation sources.

| Full-resolution region (y, x) | Foreground pixels | Connected components |
|---|---:|---:|
| Saved navigation area: y 20992:21504, x 28160:28672 | 0 | 0 |
| Annotated area: y 19900:20412, x 29100:29612 | 177,199 | 1 |

The saved bounded area is genuinely empty. Searching that area cannot establish
whether the rest of the mask contains components. This validation sampled these
two bounded regions; it was not a whole-image component census.

## Reproduced bugs

Native napari multiscale Labels use `MultiScaleData`, not a Python list/tuple.
Previous code only unwrapped lists/tuples. On disposable OME pyramids with two
objects in s0 and empty coarse levels:

- Small-mask analysis returned zero components because array conversion selected
  the coarse pyramid representation.
- Streaming analysis failed with `list indices must be integers or slices, not
  tuple`, because the worker received the pyramid container rather than s0.
- Bounded-mask searches did not clearly distinguish an empty current crop from
  an empty full source. Overview-only masks required loading an edit area, with
  the error previously positioned below the results table.

## Fixes

- Resolve `MultiScaleData`, lists, and tuples to level zero without materializing
  the array. Small and streaming component analysis both use this source. Large
  component copying also unwraps native multiscale source data.
- Keep results/errors above Find components and the table. Report zero objects
  in a bounded area explicitly, state that other areas were not searched, and
  show full-image bounds for the selected crop.
- Add **Load edit area at view center and find components** in analysis. Users
  can load/search full-resolution data from overview mode without changing tabs.
- Ignore viewing-crop replacement events when invalidating training data; those
  events do not change the full-resolution source. Actual edit notifications
  continue to invalidate prepared datasets.

## Validation

Regression coverage includes native small/streaming pyramid searches using s0,
rejection of whole-s0 materialization during streaming, empty versus annotated
bounded crops, loading a search area from overview-only state, paging, canvas
coordinates, component operations, pyramids, and training source invalidation.

Also ran both UI search paths on a disposable 512 × 512 multiscale copy of the
annotated region from the real mask. Both reported exactly one component. The
actual workspace and image/mask stores were not modified.

The real-region UI validation also passed in the user's `napari-dev2` environment:
small and streamed searches both found one component with the same source pixels.

The broad regression run passed 96 checks and exposed one crop-navigation dataset
invalidation failure. After excluding data and display-registration events during
managed crop loading, all 29 targeted checks passed, including that regression,
the four OME component-search tests, and training-widget source-edit invalidation.
`git diff --check` passed. No whole-image component count is claimed.
