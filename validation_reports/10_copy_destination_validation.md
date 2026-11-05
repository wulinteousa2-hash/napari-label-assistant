# Copy destination matching and recovery

Date: 2026-11-05. Base commit: `cb95a5e97717aca24db658c488fd810647955bcd`.
Results describe the pre-release working tree, including the earlier coordinate
and managed-grid-mask patches. No production project or user mask was changed.

## Behavior

An existing manually selected destination shows both layer names and dimensions
in a Copy/Cancel dialog. Cancel is the default and leaves pixels and layer names
unchanged. Repeated copies to the same confirmed destination do not prompt;
switching destinations asks again.

A smaller, registered source crop copies into the correct full-mask coordinates
without shrinking the destination. Otherwise matching prepares a destination
under the chosen name and retains the original as a hidden `— original` layer.
Existing overlapping annotations transfer only where pixel grids align exactly;
labels are never interpolated. Unaligned or out-of-frame annotations remain in
the original. The copy menu's restore action restores the original name and
visibility and retains the copied result separately. This recovery action applies
to matching replacements, not ordinary overwrites of a compatible destination.

Default selection prefers compatible Curated layers and excludes review-coverage
masks and retained originals. If no compatible target exists, Copy prepares
Curated instead of leaving a disabled dead end. Explicit manual selection can
still choose other masks after confirmation.

Managed grid destinations retain their canonical dimensions and disk backing;
only bounded crops enter napari. Long transfers run in chunks with progress and
cancellation. Cancelling during a transfer retains completed pixels; it differs
from cancelling the initial confirmation, which performs no copy. Existing
model/review associations remain with the retained original when a replacement
is created; the replacement needs its own scientific review/dataset selection.

## Executed validation

Environment: Python 3.11, napari 0.9.1, PySide6 6.11.2, NumPy 2.4.6,
Zarr 3.1.6, Linux ARM64; Qt offscreen. Full suite: **352 passed**, 349 warnings,
253.73 seconds. Subsequent focused tests cover the final managed-source guard
and UI wording: **32 passed**, 150 warnings, 19.64 seconds. Raw logs and JUnit records are in
[evidence/copy_destination/](evidence/copy_destination/).

The focused suite tests confirmation and cancellation, destination switching,
overlap preservation and restoration, integer IDs above 2^53, exact coordinate
mapping, sparse chunked preservation, background cancellation/read errors,
actual Qt dialogs/context menus, coverage-mask exclusion, creation without an
existing destination, and managed source/destination canonical dimensions.
Synthetic masks and disposable sparse stores were used throughout.

Commands (same environment prefix for both runs):

```bash
env QT_QPA_PLATFORM=offscreen PYTHONPATH=/tmp/label-assistant-test-deps:src \
 PYTHONDONTWRITEBYTECODE=1 XDG_CONFIG_HOME=/tmp/nla-copy-match/config \
 XDG_CACHE_HOME=/tmp/nla-copy-match/cache MPLCONFIGDIR=/tmp/nla-copy-match/mpl \
 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/wteox/miniforge3/envs/napari-dev2/bin/python -m pytest \
 -q -p no:cacheprovider --junitxml=/tmp/nla-copy-match-full-final.xml

# Final focus, with the same env prefix:
python -m pytest tests/test_copy_destination.py tests/test_label_assistant_widget.py \
 -q -p no:cacheprovider --basetemp=/tmp/nla-copy-match/reviewed-ui-2 \
 --junitxml=/tmp/nla-copy-match-reviewed-ui-2.xml
git diff --check
```

An intermediate wording assertion expected the old hint text and was updated to
check the actionable “prepare Curated” guidance. Earlier native context-menu
test harness hangs were resolved using Qt timers to close the real popup;
these were not recorded as successful runs.

## Limits

Verified: the synthetic contracts and offscreen interactions above. This is not
a human usability study, a real 84k-image performance benchmark, or evidence of
biological accuracy. Other original audit findings are not globally resolved by
this patch. Source changes are concentrated in `_copy_destination.py`,
`_widget.py`, `_large_component_controller.py` and `_mask_tiles.py`; regression
cases are in `tests/test_copy_destination.py` with updated coordinate/widget
expectations. No dependency was added. These observations were recorded before
the 5.3.0 release commit; release validation is recorded separately.
