from threading import Event

import numpy as np
import pytest

from napari_label_assistant_tools._component_library import (
    ComponentLibrary, TrainingSupervision,
)
from napari_label_assistant_tools._coverage_train import index_patches
from napari_label_assistant_tools._quick_train import QuickTrainCancelled, QuickTrainConfig


def consume(task):
    while True:
        try:
            next(task)
        except StopIteration as result:
            return result.value


def make_library(tmp_path, truth, size=8):
    library = ComponentLibrary.create(tmp_path/"components.sqlite",truth.shape,"source",tile_size=size)
    consume(library.update(truth))
    return library


def test_ids_survive_reopen_and_unchanged_refresh_without_euler(tmp_path, monkeypatch):
    import napari_label_assistant_tools._large_components as components
    monkeypatch.setattr(components, "_euler_numerators",
                        lambda *args: pytest.fail("Library must not compute Euler"))
    truth = np.zeros((32,48),np.uint16)
    truth[6:20,7:19] = 17
    truth[24:28,30:35] = 17  # one source label, two distinct component IDs
    library = make_library(tmp_path,truth)
    first,second = library.records()
    library.select(first["id"],True)
    reopened = ComponentLibrary(library.path)
    consume(reopened.update(truth,verify_all=True))
    assert reopened.records() == library.records()
    assert reopened.selected_ids() == [first["id"]]
    assert first["pixels"] == 14*12
    assert first["label_value"] == second["label_value"] == 17
    assert first["id"] != second["id"]
    assert reopened.read_ids((slice(0,32),slice(0,48)))[10,8] == first["id"]


def test_incremental_boundary_edit_preserves_id_but_requires_reselection(tmp_path):
    truth = np.zeros((48,48),np.uint8)
    truth[7:26,7:15] = 1
    truth[40:44,40:44] = 2
    library = make_library(tmp_path,truth)
    first,unrelated = library.records()
    library.select(first["id"],True)
    library.select(unrelated["id"],True)
    truth[7:26,15:17] = 1
    library.mark_dirty(np.asarray((7,15,26,17),dtype=np.int64))
    consume(library.update(truth))
    changed,untouched = library.records()
    assert changed["id"] == first["id"]
    assert changed["pixels"] == 19*10
    assert changed["needs_review"] == 1
    assert changed["included"] == 0
    assert untouched["id"] == unrelated["id"]
    assert untouched["included"] == 1
    assert not library.state()["pending"]


def test_splits_merges_deletion_and_new_objects_never_reuse_ids(tmp_path):
    truth = np.zeros((32,48),np.uint8)
    truth[6:12,4:20] = 1
    library = make_library(tmp_path,truth)
    original = library.records()[0]["id"]
    library.select(original,True)
    truth[6:12,11:13] = 0
    library.mark_dirty((6,11,12,13))
    consume(library.update(truth))
    split = [r["id"] for r in library.records()]
    assert len(split) == 2 and original not in split
    assert not library.selected_ids()
    truth[6:12,11:13] = 1
    library.mark_dirty((6,11,12,13))
    consume(library.update(truth))
    merged = library.records()[0]["id"]
    assert merged > max(split)
    truth[:]=0
    library.mark_dirty()
    consume(library.update(truth))
    assert library.counts()[0] == 0
    truth[15:18,15:18]=4
    consume(library.update(truth,verify_all=True))
    assert library.records()[0]["id"] > merged


def test_cancel_rolls_back_published_library_and_keeps_dirty_edits(tmp_path):
    truth = np.zeros((32,32),np.uint8)
    truth[3:6,3:6]=1
    library = make_library(tmp_path,truth)
    previous = library.records()
    truth[20:23,20:23]=2
    library.mark_dirty((20,20,23,23))
    cancel = Event()
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        consume(library.update(truth,cancel=cancel))
    assert library.records() == previous
    assert library.state()["pending"]
    consume(library.update(truth))
    assert len(library.records()) == 2


def test_mixed_supervision_ignores_unselected_objects_and_unknown_background(tmp_path):
    truth = np.zeros((64,64),np.uint8)
    truth[4:8,4:8]=1
    truth[4:8,12:14]=2
    truth[52:56,52:56]=3
    library = make_library(tmp_path,truth)
    for record in library.records():
        if record["label_value"] != 2:
            library.select(record["id"],True)
    reviewed = np.zeros_like(truth)
    reviewed[16:32,16:32]=1
    reviewed[48:50,48:50]=1
    supervision = TrainingSupervision(truth,reviewed,library=library)
    valid = supervision[slice(0,64),slice(0,64)]
    assert valid[5,5] == valid[53,53] == valid[20,20] == 1
    assert valid[5,12] == valid[5,10] == 0
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,samples=8,steps=2)
    dataset = consume(index_patches(truth,truth,config,reviewed=supervision))
    assert len({(p["y0"],p["x0"]) for p in dataset.locations}) == len(dataset.locations)
    assert dataset.reviewed_pixels == 260+16+16
    for i in range(len(dataset.locations)):
        _,target,usable = dataset.read(i)
        assert not target[usable==0].any()
    only_objects = TrainingSupervision(truth,library=library)
    with pytest.raises(ValueError,match="foreground only"):
        consume(index_patches(truth,truth,config,reviewed=only_objects))


def test_source_and_selection_changes_block_stale_training(tmp_path):
    truth = np.zeros((32,32),np.uint8)
    truth[2:6,2:6]=1
    library = make_library(tmp_path,truth)
    library.select(library.records()[0]["id"],True)
    stale = TrainingSupervision(truth,library=library)
    library.select(library.records()[0]["id"],False)
    with pytest.raises(ValueError,match="changed"):
        stale[slice(0,8),slice(0,8)]
    library.select(library.records()[0]["id"],True)
    stale = TrainingSupervision(truth,library=library)
    # Same foreground pixel count; the stored footprint still must not be trusted.
    truth[2,2]=0
    truth[1,2]=1
    with pytest.raises(ValueError,match="differ"):
        stale[slice(0,8),slice(0,8)]


def test_invalidated_reviewed_background_is_ignored(tmp_path):
    truth = np.zeros((16,16),np.uint8)
    reviewed = np.ones_like(truth)
    valid = TrainingSupervision(truth,reviewed,invalid_regions=[(3,4,7,9)])[
        slice(0,16),slice(0,16)]
    assert valid.sum() == 256-20


def test_incremental_update_never_materializes_full_source(tmp_path):
    class BoundedSource:
        shape = (64,64)
        dtype = np.dtype("uint8")

        def __init__(self):
            self.data = np.zeros(self.shape,self.dtype)
            self.data[5:20,7:17]=1
            self.reads = []

        def __array__(self,*args,**kwargs):
            raise AssertionError("No full-array conversion")

        def __getitem__(self,region):
            result = self.data[region]
            assert max(result.shape) <= 8
            self.reads.append(region)
            return result

    source = BoundedSource()
    library = ComponentLibrary.create(tmp_path/"components.sqlite",source.shape,"source",tile_size=8)
    consume(library.update(source))
    source.reads.clear()
    source.data[5:20,17:19]=1
    library.mark_dirty((5,17,20,19))
    consume(library.update(source))
    assert source.reads
    assert all(r[0].stop<=25 and r[1].stop<=25 for r in source.reads)
    assert library.records()[0]["pixels"] == 15*12


def test_one_component_cannot_leak_across_training_and_validation(tmp_path):
    truth = np.zeros((128,128),np.uint8)
    truth[5:10,5:70] = 1  # crosses several spatial blocks
    truth[100:105,100:105] = 2
    library = make_library(tmp_path,truth)
    for record in library.records():
        library.select(record["id"],True)
    valid = TrainingSupervision(truth,np.ones_like(truth),library=library)
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,samples=8,steps=2)
    dataset = consume(index_patches(truth,truth,config,reviewed=valid))
    splits = set()
    for item in dataset.locations:
        region = slice(item["y0"],item["y1"]),slice(item["x0"],item["x1"])
        if (truth[region]==1).any():
            splits.add(item["split"])
    assert len(splits) == 1


def test_cancel_during_save_does_not_publish_partial_records_or_id_tiles(tmp_path):
    truth = np.zeros((32,32),np.uint8)
    truth[2:6,2:6]=1
    library = make_library(tmp_path,truth)
    library.select(library.records()[0]["id"],True)
    previous = library.records()
    previous_counts = library.counts()
    truth[20:25,20:25]=2
    library.mark_dirty((20,20,25,25))
    cancel = Event()
    task = library.update(truth,cancel=cancel)
    for done,total,message in task:
        if message == "Saving the component library":
            cancel.set()
            break
    with pytest.raises(QuickTrainCancelled):
        consume(task)
    assert library.records() == previous
    assert library.counts() == previous_counts
    assert library.state()["pending"]
    assert library.read_ids((slice(20,25),slice(20,25))).sum() == 0


def test_source_label_id_changes_after_indexing_are_detected(tmp_path):
    truth = np.zeros((64,64),np.uint32)
    truth[4:8,4:8]=1
    truth[52:56,52:56]=2
    library = make_library(tmp_path,truth)
    for record in library.records():
        library.select(record["id"],True)
    supervision = TrainingSupervision(truth,np.ones_like(truth),library=library)
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,samples=8,steps=2)
    dataset = consume(index_patches(truth,truth,config,reviewed=supervision))
    item = next(i for i,p in enumerate(dataset.locations) if p["y0"]==0 and p["x0"]==0)
    truth[4:8,4:8]=9  # Same foreground and valid pixels; raw IDs still changed.
    with pytest.raises(ValueError,match="label IDs"):
        dataset.read(item)
