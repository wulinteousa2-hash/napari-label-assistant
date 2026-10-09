import json
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari_label_assistant_tools._component_library import (
    KEY, annotation_changed, ensure_library, library_for,
)
from napari_label_assistant_tools._component_library_widget import ComponentLibraryWidget
from napari_label_assistant_tools.workspace.service import (
    WorkspaceError, create_portable_snapshot, load_workspace, save_workspace,
)
from test_component_library import consume


def source_data():
    truth = np.zeros((64,64),np.uint16)
    truth[5:12,5:12] = 17
    truth[45:52,45:52] = 17
    return truth


def test_library_checkboxes_persist_and_auto_update_local_edits(
    make_napari_viewer, qtbot, tmp_path
):
    viewer = make_napari_viewer()
    source = viewer.add_labels(source_data(),name="Curated")
    library = ensure_library(source,tmp_path/"library.sqlite")
    widget = ComponentLibraryWidget(viewer,lambda:source)
    qtbot.addWidget(widget)
    widget.start_update(verify_all=True)
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    assert library.counts()[0] == 2, widget.status.text()
    assert widget.table.rowCount() == 2
    widget.table.item(0,0).setCheckState(Qt.Checked)
    original_id = widget.table.item(0,0).data(Qt.UserRole)
    assert library.selected_ids() == [original_id]
    assert "selection saved" in widget.status.text()
    source.data[5:12,12:14]=17
    annotation_changed(viewer,source,(5,12,12,14))
    widget.refresh()
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    first = library.records()[0]
    assert first["id"] == original_id
    assert first["needs_review"] == 1
    assert not first["included"]
    assert not library.state()["pending"]
    widget.shutdown()


def test_workspace_reopen_save_as_and_portable_export_preserve_library(
    make_napari_viewer,tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((64,64),np.uint8),name="image")
    source = viewer.add_labels(source_data(),name="Curated")
    original = ensure_library(source,tmp_path/"initial.sqlite")
    consume(original.update(source.data))
    selected_id = original.records()[0]["id"]
    original.select(selected_id,True)
    first_path = tmp_path/"original"/"project.json"
    save_workspace(viewer,first_path,complete_package=True,xy_chunk=16)
    durable_path = Path(source.metadata[KEY]["path"])
    assert durable_path != original.path
    assert durable_path.is_relative_to(first_path.parent)
    restored = make_napari_viewer()
    result = load_workspace(restored,first_path)
    assert result["skipped_layers"] == []
    restored_source = restored.layers["Curated"]
    library = library_for(restored_source)
    assert library.selected_ids() == [selected_id]
    assert library.records() == original.records()
    copy_path = tmp_path/"copy"/"project.json"
    save_workspace(restored,copy_path,copy_labels=True,complete_package=True,xy_chunk=16)
    independent = library_for(restored_source)
    assert independent.path != durable_path
    independent.select(selected_id,False)
    assert library_for(source).selected_ids() == [selected_id]
    portable = create_portable_snapshot(copy_path,tmp_path/"portable")
    portable_viewer = make_napari_viewer()
    load_workspace(portable_viewer,portable["path"])
    portable_library = library_for(portable_viewer.layers["Curated"])
    assert portable_library.path.is_relative_to(tmp_path/"portable")
    assert portable_library.records() == independent.records()
    assert not portable_library.selected_ids()


def test_missing_library_is_reported_and_prevents_overwriting_project(
    make_napari_viewer,tmp_path
):
    viewer = make_napari_viewer()
    source = viewer.add_labels(source_data(),name="Curated")
    library = ensure_library(source,tmp_path/"initial.sqlite")
    consume(library.update(source.data))
    path = tmp_path/"project.json"
    save_workspace(viewer,path)
    payload = json.loads(path.read_text())
    payload["layers"][0][KEY]["path"]="missing.sqlite"
    path.write_text(json.dumps(payload))
    restored = make_napari_viewer()
    result = load_workspace(restored,path)
    assert result["skipped_layers"]
    assert "library is unavailable" in result["skipped_layers"][0]["reason"]
    with pytest.raises(WorkspaceError,match="incomplete workspace"):
        save_workspace(restored,path)


def test_editing_invalidates_reviewed_background_and_new_masks_start_unchecked(
    make_napari_viewer,tmp_path
):
    from napari_label_assistant_tools._reviewed_regions import KEY as REVIEW_KEY, identity
    viewer = make_napari_viewer()
    source = viewer.add_labels(source_data(),name="Curated")
    library = ensure_library(source,tmp_path/"initial.sqlite")
    consume(library.update(source.data))
    original = library.records()
    for record in original:
        library.select(record["id"],True)
    reviewed = viewer.add_labels(np.ones((64,64),np.uint8),metadata={
        REVIEW_KEY:{"role":"reviewed_regions","truth_id":identity(source),
                    "areas":[[0,0,64,64]]}})
    source.data[20:25,20:25]=19
    annotation_changed(viewer,source,(20,20,25,25))
    consume(library.update(source.data))
    state = reviewed.metadata[REVIEW_KEY]
    assert state["invalid_regions"] == [[0,0,64,64]]
    assert sum((r[2]-r[0])*(r[3]-r[1]) for r in state["areas"]) == 0
    new = next(r for r in library.records() if r["label_value"]==19)
    assert new["id"] > max(r["id"] for r in original)
    assert not new["included"]
    assert library.selected_ids() == [r["id"] for r in original]


def test_checkbox_avoids_pixel_refresh_and_uses_new_save_as_library_immediately(
    make_napari_viewer,qtbot,tmp_path
):
    viewer = make_napari_viewer()
    source = viewer.add_labels(source_data(),name="Curated")
    library = ensure_library(source,tmp_path/"initial.sqlite")
    consume(library.update(source.data))
    widget = ComponentLibraryWidget(viewer,lambda:source)
    qtbot.addWidget(widget)
    save_workspace(viewer,tmp_path/"first"/"project.json")
    widget.refresh()
    original = library_for(source)
    save_workspace(viewer,tmp_path/"second"/"project.json",copy_labels=True)
    current = library_for(source)
    assert current.path != original.path
    events = []
    source.events.data.connect(lambda event: events.append(event))
    widget.table.item(0,0).setCheckState(Qt.Checked)
    selected = widget.table.item(0,0).data(Qt.UserRole)
    assert current.selected_ids() == [selected]
    assert original.selected_ids() == []
    assert events == []  # No full-mask refresh when a training checkbox changes.
    widget.shutdown()


def test_cancelled_scan_does_not_restart_until_requested(
    make_napari_viewer,qtbot,tmp_path,monkeypatch
):
    from threading import Event
    from napari_label_assistant_tools._component_library import ComponentLibrary
    from napari_label_assistant_tools._quick_train import QuickTrainCancelled
    viewer=make_napari_viewer()
    source=viewer.add_labels(source_data(),name="Curated")
    library=ensure_library(source,tmp_path/"initial.sqlite")
    consume(library.update(source.data))
    widget=ComponentLibraryWidget(viewer,lambda:source)
    qtbot.addWidget(widget)
    started=Event()
    calls=[]
    def waiting(self,data,*,cancel,verify_all):
        calls.append(True)
        started.set()
        yield 0,1,"Waiting"
        assert cancel.wait(5)
        raise QuickTrainCancelled()
    monkeypatch.setattr(ComponentLibrary,"update",waiting)
    library.mark_dirty((0,0,5,5))
    widget.start_update(verify_all=False)
    qtbot.waitUntil(started.is_set)
    widget.cancel_button.click()
    qtbot.waitUntil(lambda:widget.worker is None)
    widget.refresh()
    widget.refresh()
    assert len(calls)==1
    assert library.state()["pending"]
    assert widget.worker is None
    widget.shutdown()


def test_open_workspace_during_scan_preserves_current_layers(
    make_napari_viewer,qtbot,tmp_path
):
    viewer=make_napari_viewer()
    source=viewer.add_labels(source_data(),name="Curated")
    library=ensure_library(source,tmp_path/"initial.sqlite")
    consume(library.update(source.data))
    path=tmp_path/"project.json"
    save_workspace(viewer,path)
    widget=ComponentLibraryWidget(viewer,lambda:source)
    qtbot.addWidget(widget)
    widget.worker=object()
    try:
        with pytest.raises(WorkspaceError,match="component library"):
            load_workspace(viewer,path)
        assert len(viewer.layers)==1 and viewer.layers[0] is source
    finally:
        widget.worker=None
        widget.shutdown()
