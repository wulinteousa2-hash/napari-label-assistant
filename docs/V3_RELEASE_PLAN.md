# Version 3 release plan

This records the training and review workflow discussion and its implementation.
The work started with the 3.0.0 major-version increment after 2.0.2.
The current version is 3.2.0. It includes the 3.1.0 development work and the
editing, review-navigation, deletion-recovery, and save-state improvements
documented in [CHANGELOG.md](../CHANGELOG.md).
The persistent library, combined supervision, workspace persistence, and local
update paths are implemented. The user workflow is in [TRAINING.md](TRAINING.md).

## Completed UI changes

- Reduce wording and vertical space in Quick Train for a 2K monitor, using
  napari-unet-assistant as a reference.
- Use five pages: **1. Dataset**, **2. Models**, **3. Train**, **4. Inference**,
  **5. Advanced**.
- Dataset owns reviewed-cell confirmation and the component training library;
  Labels retains local mask editing, canvas cell selection, and component QC.
- Keep patch size, model input size, seed, batch size, compute device, and
  learning rate visible in Advanced, rather than expanding fields elsewhere.
- Keep progress, status, and Cancel available across tabs.
- Advanced exposes augmentation, optional validation-loss plateau LR reduction
  followed by early stopping, batches up to 128, 4096 source patches, and 2048
  model inputs. Model history preserves settings and actual completed epochs.
- Use short instructions and tooltips; retain annotation coverage guidance.
- Inference remains prediction of the current area, not full-image inference.
- Widget checks cover the four pages and library selection/persistence.

## Implemented: two compatible ways to supply training annotations

### Local Editing: reviewed regions

Users select a grid cell, correct or add Curated objects with Editable Area,
check all targets and background, then finish review to include the region.
Reviewed regions supply confirmed foreground and confirmed background.
Keep this workflow alongside the component library.

### Training → Dataset: component library

Build a persistent library of all components, with these basic columns:

| Include for training | Library/component ID | Label pixels | Bounds |
| --- | --- | --- | --- |
| Checkbox | Permanent identifier | Pixel count | y0, x0, y1, x1 |

- Omit Euler computation from this library.
- Clicking a row navigates to its object.
- Checking a row saves its selection, without immediately copying pixels or
  rewriting a large reviewed-region mask.
- Label the action **Include for training**, not **Reviewed**: accepting an
  object does not confirm its surrounding background.
- Retain source label value internally, separately from permanent library ID;
  one label value can contain multiple disconnected components.

## Implemented: shared training dataset and patch rules

- Combine reviewed regions and selected components in the same training dataset.
- Selected components supply confirmed foreground. Their unchecked surroundings
  are ignored, rather than assumed to be background.
- Reviewed regions supply background supervision needed to reject background.
- Use fixed-size source patches, such as 512 × 512, rather than a different
  model input size for each component.
- Bounds identify candidate patches; exact occupied-patch counts require checking
  component pixels. A small object may share a patch with other objects; a large
  object can occupy several patches.
- Deduplicate overlapping patch coordinates and supervised pixels.
- Preserve surrounding image context while excluding unconfirmed pixels from loss.
- Prepare patches when the user requests Preview training patches.
- Preserve spatial separation of training and validation; shared patches and
  overlapping objects must not leak across the split.
- Use uniform patch weighting; large objects may contribute multiple patches
  (see Implementation choices for the current weighting limitation).

## Implemented: persistence and stable identity

Save the library as a project asset linked from the workspace, including:

- Permanent library ID and source label value.
- Pixel count, bounds, and training-selection state.
- Source image/layer identity and annotation revision.
- Retired IDs and the next available ID, so IDs are never reused.

Open Workspace should load the saved library directly, retaining IDs and checked
rows. Sorting, filtering, and reopening must not renumber objects. Save small
selection changes immediately; include them in Save/Save As and portable projects.
Training patches can be regenerated and do not define permanent object identity.
Existing workspaces without a library must remain supported.

| Annotation change | Intended library behavior |
| --- | --- |
| Unchanged object | Preserve ID and training selection |
| Boundary edit with clear identity | Preserve ID, update measurements, require re-review |
| New object | Allocate a new ID; start unchecked |
| Split or merge | Retire affected IDs; allocate new IDs; require selection again |
| Deletion | Retire ID and remove it from active training selections |

Geometry matching alone cannot guarantee identity. The implementation tracks
edit bounds and handles ambiguous identity conservatively.

## Implemented: continue SAM3/Curated editing after reopening

- Reopen the workspace and continue adding or correcting SAM3 masks in Curated
  through Editable Area.
- Update the library for the edited area and affected objects crossing its
  boundary, avoiding full-image rescans when possible.
- Preserve unrelated objects and selections.
- Flag affected previously reviewed cells for re-review before the
  next training preparation.
- Keep masks, library revisions, selections, and workspace references consistent
  when saving; detect stale records after interrupted or external edits.

## Implemented: responsiveness on 80K × 80K images

Users reported about a four-second pause when clicking to mark a cell reviewed.
Current marking performs synchronous annotation copying, coverage writes, and
layer refreshes. Duplicate refreshes have been reduced; the manual pixel writes
remain synchronous. The library avoids those writes when selecting objects.

- Build the initial library once in a background worker using bounded memory.
- Keep table display paginated or virtualized for many components.
- Make checkbox selection a small persistent update, independent of image size.
- Defer expensive patch preparation and pixel writes; show progress and allow
  cancellation for long operations.
- Further profiling of the manual review path remains future work.

## Acceptance checks before release

- Open/save/reopen preserves library IDs, selections, and reviewed-region links.
- Linked, Save As, and portable workspaces retain required library assets.
- New, edited, deleted, split, and merged objects follow the identity rules.
- Incremental updates handle objects crossing Editable Area boundaries.
- Selected-object background stays ignored unless explicitly reviewed.
- Overlaps do not duplicate training patches or contaminate validation.
- Mixed component/region datasets provide both foreground and background.
- Large-image selection stays responsive; background scans cancel safely.
- Existing projects without a component library still open correctly.

## Implementation choices

- SQLite schema version 1 stores records and compressed spatial ID tiles;
  optional workspace references retain compatibility with existing version-1
  manifests. Missing or unsupported libraries block unsafe project saves.
- Known edit bounds expand to complete old/new boundary-crossing components.
  One-to-one same-label pixel overlap preserves identity; ambiguous splits/merges
  allocate new IDs. Coordinate fingerprints identify unchanged geometry.
- Fixed source-grid patches provide context; coverage weights patches uniformly.
  Larger objects can contribute more patches. Equal weighting per object is not
  implemented.
- Explicit Build / update checks external source edits using tile fingerprints.
  Training verifies selection revisions and annotation/label fingerprints.
- SQLite updates publish records and ID tiles in one transaction. Cancellation
  during saving rolls back staged changes and retains dirty edit bounds.
- Workspace saves/opening wait for active library or component mutations.
  Source masks remain normal workspace assets; model weights are saved separately.

## Validation on the supplied accepted workspace

Read-only source: final_p2mpzem08.label-assistant.json, with Curated dimensions
84,175 × 79,966 and 20 stored reviewed areas.

- Built a library of 673 components without Euler computation.
- Indexed 76 candidate patches: 60 training and 16 validation.
- Prepared 8 bounded previews; reopened selections retained their IDs.
- Added a 9-pixel mask in an isolated editable overlay: allocated ID 674 and
  retained all 673 previous IDs and previous selections.
- The local update took approximately 0.3 seconds; selection database writes
  took 10–13 ms in that test.
- The accepted manifest remained unchanged and source arrays were opened read-only.
- The optimized initial full-image build took about 200 seconds on this machine.
  Its progress and cancellation controls remain available during the scan.
