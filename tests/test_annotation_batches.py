"""Annotation worksets remain separate from global approval and pixel storage."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from napari_label_assistant_tools import _annotation_batches as batches
from napari_label_assistant_tools._component_library import TrainingSupervision
from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel
from napari_label_assistant_tools._coverage_train import index_patches
from napari_label_assistant_tools._reviewed_regions import associate


def consume(generator):
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value


@pytest.fixture
def annotated(make_napari_viewer):
    viewer = make_napari_viewer()
    yy,xx = np.indices((256,256))
    target = ((xx%16)<8).astype('uint8')
    image = viewer.add_image((target*100+50+yy%8).astype('uint8'), name='EM')
    truth = viewer.add_labels(target, name='Curated')
    reviewed = viewer.add_labels(np.ones_like(target), name='Reviewed')
    associate(reviewed, image, truth)
    reviewed.metadata['label_assistant_review'].update(areas=[[0,0,128,256]], review_cells=[[0,0,128,256]])
    return viewer,image,truth,reviewed


def test_new_batches_keep_approval_and_pixels_and_have_stable_ids(annotated):
    viewer,image,truth,reviewed = annotated
    pixels = truth.data.copy()
    state = deepcopy(reviewed.metadata)
    first = batches.ensure_pair(viewer,image,truth)
    assert first['scope'] == 'all'
    assert first['batches'][0]['regions'] == [[0,0,128,256]]
    assert first['batches'][0]['created_at_utc'] is None
    second = batches.change(viewer,image,truth,'new')
    assert second['name'] == 'Batch 2' and second['regions'] == []
    assert batches.selection(image,truth)['scope'] == 'active'
    assert second['id'] != first['batches'][0]['id']
    assert second['created_at_utc'].endswith('+00:00')
    assert reviewed.metadata == state
    np.testing.assert_array_equal(truth.data,pixels)
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    assert batches.selection(image,truth)['regions'] == [[128,0,256,256]]
    assert len(batches.active_batch(batches.get_pair(image,truth))['events']) == 2
    snapshot = batches.selection(image,truth)
    batches.change(viewer,image,truth,'rename',name='New boundaries')
    assert batches.selection(image,truth)['id'] == snapshot['id']
    assert snapshot['name'] == 'Batch 2'
    batches.change(viewer,image,truth,'select',batch_id=first['active_id'])
    assert batches.selection(image,truth)['regions'] == [[128,0,256,256]]
    batches.change(viewer,image,truth,'training',batch_id=first['active_id'])
    assert batches.selection(image,truth)['regions'] == [[0,0,128,256]]


def test_first_empty_batch_number_and_finished_batch_guard(annotated):
    viewer,image,truth,reviewed = annotated
    reviewed.metadata['label_assistant_review'].update(areas=[],review_cells=[])
    first = batches.change(viewer,image,truth,'new')
    assert first['number'] == 1
    batches.change(viewer,image,truth,'finish')
    before = deepcopy(truth.metadata[batches.KEY])
    with pytest.raises(ValueError,match='finished'):
        batches.change(viewer,image,truth,'approve',regions=[[0,0,32,32]])
    assert truth.metadata[batches.KEY] == before
    batches.change(viewer,image,truth,'reopen')
    batches.change(viewer,image,truth,'approve',regions=[[0,0,32,32]])
    assert batches.change(viewer,image,truth,'new')['number'] == 2


def test_batch_indexing_excludes_other_batches_and_keeps_known_background(annotated):
    viewer,image,truth,reviewed = annotated
    first = batches.ensure_pair(viewer,image,truth)
    batches.change(viewer,image,truth,'new')
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    reviewed.metadata['label_assistant_review']['areas'].append([128,0,256,256])
    base = TrainingSupervision(truth.data,reviewed.data, reviewed_areas=reviewed.metadata['label_assistant_review']['areas'])
    restricted = batches.restrict_supervision(viewer,image,truth,base)
    assert restricted.base is base
    assert not restricted[np.s_[0:32,0:32]].any()
    assert restricted[np.s_[128:160,0:32]].all()
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,base_channels=4,device='cpu')
    dataset = consume(index_patches(image.data,truth.data,config,reviewed=restricted))
    assert len(dataset.locations) == 32
    assert all(p['y0'] >= 128 for p in dataset.locations)
    assert all(p['reviewed_pixels'] > p['foreground_pixels_original'] > 0 for p in dataset.locations)
    batches.change(viewer,image,truth,'select',batch_id=first['active_id'])
    batches.change(viewer,image,truth,'training',batch_id=first['active_id'])
    other = batches.restrict_supervision(viewer,image,truth,base)
    assert other[np.s_[0:32,0:32]].all()
    assert not other[np.s_[128:160,0:32]].any()
    batches.change(viewer,image,truth,'all')
    assert batches.restrict_supervision(viewer,image,truth,base) is base


def test_empty_batches_fail_and_removed_members_keep_global_review(annotated):
    viewer,image,truth,reviewed = annotated
    batches.change(viewer,image,truth,'new')
    with pytest.raises(ValueError,match='empty'):
        batches.restrict_supervision(viewer,image,truth,None)
    batches.change(viewer,image,truth,'add',regions=[[0,0,32,32]])
    before = deepcopy(reviewed.metadata)
    batches.change(viewer,image,truth,'remove',regions=[[0,0,32,32]])
    assert batches.selection(image,truth)['regions'] == []
    assert reviewed.metadata == before


def test_batch_bounds_and_registration_are_validated(annotated):
    viewer,image,truth,_ = annotated
    batches.ensure_pair(viewer,image,truth)
    with pytest.raises(ValueError,match='outside'):
        batches.change(viewer,image,truth,'approve',regions=[[0,0,257,32]])
    truth.translate = (1,0)
    with pytest.raises(ValueError,match='registration'):
        batches.get_pair(image,truth)


def test_batch_supervision_only_reads_requested_patch_and_keeps_unknown_channels():
    class Source:
        shape = (100000,100000)
        def __array__(self,*args):
            raise AssertionError('Do not materialize large validity masks')
        def __getitem__(self,region):
            assert region == (slice(32,64),slice(32,64))
            valid = np.ones((32,32),'uint8')
            valid[:8] = 0
            return valid
    selected = batches.BatchSupervision(Source(), [[32,32,48,64]])
    result = selected[np.s_[32:64,32:64]]
    assert not result[:8].any() and result[8:16].all() and not result[16:].any()
    assert selected.candidate_tiles(32) == [(1,1)]


def test_prior_holdouts_including_unsampled_trial_areas_never_become_training(annotated):
    _,image,truth,reviewed = annotated
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,base_channels=4,device='cpu')
    dataset = consume(index_patches(image.data,truth.data,config))
    identifier = batches._id(image)
    history = dict(config=asdict(config), source={'image_id':identifier,'image_shape':[256,256]},
                   validation_blocks=['0:0'], locations=[dict(split='validation',y0=0,x0=0,size=32)])
    model = QuickTrainModel(None,config,32,{},'cpu',history=[history])
    protected = batches.protect_validation(dataset,model,image_id=identifier)
    assert all(p['split'] == 'validation' for p in protected.locations if p['y0']<64 and p['x0']<64)
    assert protected.image is dataset.image and protected.ground_truth is dataset.ground_truth
    assert protected.validation['preserved_prior_validation']
    history['locations'].append(dict(split='training',y0=32,x0=32,size=32))
    assert batches.protect_validation(dataset,model,image_id=identifier).validation_seen_before


def test_multilabel_batches_preserve_independent_channel_validity(annotated):
    from napari_label_assistant_tools._multilabel import index_multilabel
    _,image,truth,reviewed = annotated
    base = TrainingSupervision(truth.data, reviewed.data, reviewed_areas=[[0,0,256,256]])
    channels = [dict(data=truth.data, reviewed=batches.BatchSupervision(
                    base, [[0,x,256,x+32] for x in range(offset,256,64)]))
                for offset in (0,32)]
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,base_channels=4,device='cpu',
                              class_names=('Remak','Axon'),multilabel=True)
    pairs = [dict(image=image.data,channels=channels,
                  info=dict(image_id='EM',image_layer='EM',split_group='EM',channels=[
                      dict(name=name,target_id=name,coverage='partial') for name in config.class_names]))]
    dataset = consume(index_multilabel(pairs, config))
    for index,patch in enumerate(dataset.locations):
        _,_,valid = dataset.read(index)
        known = (patch['x0']//32)%2
        assert valid[known].all()
        assert not valid[1-known].any()


def test_checkpoint_names_include_single_selected_batch(annotated):
    from napari_label_assistant_tools._model_naming import checkpoint_filename
    viewer,image,truth,_ = annotated
    batches.change(viewer,image,truth,'new')
    info = batches.selection(image,truth)
    record = dict(name='Remak',revision='12345678abcdef',config={'input_pixels':512,'base_channels':64},
                  history=[dict(source={'image_layer':'EM08','annotation_batches':[info]},started_at_utc='2026-11-19T12:00:00Z')])
    assert 'Remak_EM08_Batch-2_unet-w64_in512' in checkpoint_filename(record)


def test_reviewed_batch_does_not_require_rebuilding_old_pending_component_library(annotated,tmp_path,qtbot):
    from napari_label_assistant_tools._component_library import ensure_library
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer,image,truth,reviewed = annotated
    library = ensure_library(truth,tmp_path/'objects.sqlite')
    consume(library.update(truth.data))
    library.select(library.records()[0]['id'],True)
    library.mark_dirty((0,0,32,32))
    assert library.state()['pending'] and library.counts()[1] == 1
    batches.change(viewer,image,truth,'new')
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    reviewed.metadata['label_assistant_review']['areas'].append([128,0,256,256])
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.image_combo.setCurrentIndex(widget.image_combo.findData(image))
    widget.truth_combo.setCurrentIndex(widget.truth_combo.findData(truth))
    widget.reviewed_combo.setCurrentIndex(widget.reviewed_combo.findData(reviewed))
    for combo in (widget.crop_combo,widget.resolution_combo):
        combo.addItem('32 px (test)',32)
        combo.setCurrentIndex(combo.findData(32))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    assert widget.prepared is not None,widget.status_label.text()
    assert all(p['y0'] >= 128 for p in widget.prepared.locations)
    assert library.state()['pending'] and library.counts()[1] == 1


def test_loading_checkpoint_after_preparation_preserves_earlier_batch_holdouts(annotated,qtbot,monkeypatch):
    from napari_label_assistant_tools._model_providers import build_model
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer,image,truth,reviewed = annotated
    batches.change(viewer,image,truth,'new')
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    reviewed.metadata['label_assistant_review']['areas'].append([128,0,256,256])
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.reviewed_combo.setCurrentIndex(widget.reviewed_combo.findData(reviewed))
    config = QuickTrainConfig(crop_pixels=32,input_pixels=32,base_channels=4,device='cpu')
    monkeypatch.setattr(widget,'_config',lambda:config)
    widget.prepare_button.click()
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    assert widget.prepared is not None,widget.status_label.text()
    patch = next(p for p in widget.prepared.locations if p['split'] == 'training')
    history = dict(workflow='coverage',mode='new',epochs=1,retained_epoch=1,config=asdict(config),
                   source=dict(image_id=batches._id(image),image_layer=image.name,image_shape=[256,256]),
                   validation_blocks=[patch['block']],locations=[{**patch,'split':'validation'}])
    model = QuickTrainModel(build_model(config),config,32,
                            dict(dice=.5,iou=.4,precision=.6,recall=.5),'cpu',history=[history])
    widget._model_loaded(model,'previous batch checkpoint',register=False)
    assert widget.prepared is not None
    assert all(p['split'] == 'validation' for p in widget.prepared.locations if p['block'] == patch['block'])
    assert widget.prepared.validation['preserved_prior_validation']
    assert not widget.prepared.validation_seen_before
    assert widget.continue_button.isEnabled()
