# Simple tile curation workflow — local validation

Date: 2026-11-17. Changes are local; no release or GitHub push was performed.

## Result and scope

The plugin submenu offers **Label Curation — Simple** and **Label Curation —
Advanced**. Both open the same dock and annotation/training controllers. Advanced
retains Workspace, Labels, and Training. Simple adds compact, collapsible
Workspace, Annotate & Review, Progress & Dataset, and Tools sections, with colors
derived from the napari theme.

Existing camera-follow editing, review coverage, supervision semantics, component
selection, dataset preparation, model training, and validation algorithms are
reused. Review approval already established foreground and verified background;
edits already invalidated affected approval through metadata. Neither behavior
was reimplemented. Review tiles, edit buffers, and training crops remain distinct.

New workflow behavior includes shared pending tile selection, explicit Edit/Review
modes, reviewed-and-next navigation, clear/revoke/undo actions, batch cancellation,
and a small workspace-scoped recovery journal. Approval and revocation never alter
biological label values. Historical training use remains recorded after edits.
Multiple-image and multilabel preparation retains its configured source assignments.

## Persistence and memory

After the first workspace save, `tile-progress.json` records coordinates, source
identities, registration, and review metadata in the workspace data directory.
It contains no image or label arrays. Restoring requires a matching manifest
digest and source identity/shape/registration. Incompatible recovery is reported.
New managed review-coverage masks use sparse, durable Zarr storage. The workspace
JSON remains version 2, with optional workflow metadata; older projects need no
migration. Normal workspace Save is still needed for new layers, model changes,
and other project settings.

The existing bounded pixel-write/recovery path is unchanged. A regression test
reviews a sparse **100,000 × 100,000** mask from its noneditable overview state and
rejects full-resolution reads larger than 1024 pixels per side. A separate test
forbids pixel reads entirely while capturing/writing progress metadata. These are
allocation/read-bound checks, not a real-dataset performance benchmark.

## Files for this workflow

- `src/napari_label_assistant_tools/_curation_shell.py`: shared dock and theme.
- `src/napari_label_assistant_tools/_simple_curation.py`: Simple presentation.
- `src/napari_label_assistant_tools/_tile_progress.py`: recovery journal.
- `src/napari_label_assistant_tools/_label_assistant.py`, `napari.yaml`: menu commands
  and reuse of the existing dock contribution.
- `src/napari_label_assistant_tools/_widget.py`: shared pending selection and grid state.
- `src/napari_label_assistant_tools/_reviewed_regions.py`: shared next-unreviewed
  navigation, durable sparse coverage, and explicitly bounded overview review.
- `src/napari_label_assistant_tools/workspace/service.py`, `_workspace_widget.py`:
  optional workflow metadata, recovery loading/status, and pre-detach flushing.
- `tests/test_simple_curation.py`: 19 regression tests.
- `tests/test_label_pyramid.py`: wait for deferred overview cleanup, as required by
  the existing local fix for napari layer-removal snapshot errors.
- `README.md`, `docs/TRAINING.md`: usage and persistence guidance.

Earlier local camera-follow, screening, workspace-saving, overview-lifecycle, and
documentation changes were retained.

## Validation

Environment: Python 3.11, Qt/PySide6, napari 0.9.1; Qt tests ran offscreen in
`napari-dev`. The panel was also rendered in `napari-dev2` using temporary synthetic
data and napari's dark/light styles. No researcher workspace or annotation store
was opened or modified for these tests.

- Complete suite run: **504 passed, 1 failed** in 728.91 seconds. The failure was
  the older assertion that overview removal is synchronous. Cleanup is intentionally
  deferred until napari finishes deleting its selected-layer snapshot.
- After updating that assertion and completing the workflow checks, the final
  focused run passed **127 tests** in 148.44 seconds, including all **19 Simple
  regressions**, pyramids, lifecycle, camera follow, review, workspace save/load,
  and existing curation controls. The complete suite was not repeated afterward.
- Actual napari command-registry execution verifies both mode entries reuse one dock.
- `npe2 validate src/napari_label_assistant_tools/napari.yaml`: valid.
- `git diff --check`: passed.

Test command prefix:

```bash
QT_QPA_PLATFORM=offscreen \
LD_PRELOAD=/home/wteox/miniforge3/envs/napari-dev/lib/libsqlite3.so.3.53.4 \
/home/wteox/miniforge3/envs/napari-dev/bin/python -m pytest -q --disable-warnings --tb=short
```

The preload addresses that test environment's SQLite library mismatch. It is not
a plugin dependency or a required end-user configuration change.

## Manual checks remaining

Use a freshly started napari session to load the updated plugin menu. Check painting,
camera following, coverage visibility, and dock sizing on a real large-image
workspace with an actual GPU canvas. Offscreen Qt cannot validate live OpenGL
rendering or SAM3 running concurrently. Tile recovery is debounced and relies on
the existing bounded pixel recovery; it does not promise recovery of every action
interrupted by a process kill or hardware/storage failure.
