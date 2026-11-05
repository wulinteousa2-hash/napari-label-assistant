"""Disposable scientific pipeline validation; no production data or source edits."""
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from napari_label_assistant_tools._coverage_train import index_pairs, train_indexed
from napari_label_assistant_tools._multilabel import index_multilabel
from napari_label_assistant_tools._quick_train import (
    QuickTrainConfig, predict_region, prediction_labels, save_model, load_model,
)
from napari_label_assistant_tools._widget import _build_fast_component_index, _copy_mask_components_to_target
from napari_label_assistant_tools.workspace import save_workspace, load_workspace

OUT = Path(__file__).resolve().parents[1] / 'evidence'


def consume(generator):
    updates = []
    while True:
        try:
            updates.append(next(generator))
        except StopIteration as done:
            return done.value, updates


def digest(array):
    return sha256(np.asarray(array).tobytes()).hexdigest()


def synthetic(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((96, 128))
    axon, myelin = np.zeros(yy.shape, bool), np.zeros(yy.shape, bool)
    for y, x, ry, rx in ((18, 19, 7, 10), (30, 63, 9, 8), (69, 34, 10, 13), (70, 96, 12, 8)):
        radius = ((yy-y)/ry)**2 + ((xx-x)/rx)**2
        axon |= radius < .48
        myelin |= (radius >= .48) & (radius < 1)
    truth = axon.astype(np.uint8) + 2*myelin.astype(np.uint8)
    image = np.clip(1000 + 1200*axon + 4000*myelin + yy*3 + rng.normal(0,80,yy.shape), 0,65535).astype(np.uint16)
    # Nested independent channels: outer cell region includes the axon interior.
    return image, truth, (axon | myelin).astype(np.uint8), axon.astype(np.uint8)


@pytest.mark.parametrize('mode', ['binary', 'multiclass', 'multilabel'])
def test_synthetic_pipeline(mode, make_napari_viewer, qtbot, tmp_path):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    import torch
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1,
                              base_channels=4, batch_size=2, device='cpu', seed=71,
                              class_names=() if mode=='binary' else ('axon','myelin') if mode=='multiclass' else ('whole region','axon'),
                              multilabel=mode=='multilabel')
    source = make_napari_viewer()
    originals = []
    for seed in (71,72,73):
        image, truth, outer, inner = synthetic(seed)
        source.add_image(image, name=f'image {seed}')
        target = (truth != 0).astype(np.uint8) if mode=='binary' else truth
        source.add_labels(target, name=f'truth {seed}')
        originals.extend([image.copy(), target.copy()])
        if mode=='multilabel':
            source.add_labels(outer, name=f'whole region {seed}')
            source.add_labels(inner, name=f'axon {seed}')
            originals.extend([outer.copy(), inner.copy()])
    # Representative selection/copy acceptance on an isolated disposable mask.
    raw = originals[1]
    components = _build_fast_component_index(raw)
    accepted = list(components.records)[:1]
    curated, changed = _copy_mask_components_to_target(raw, np.zeros_like(raw), components.component_id_map, accepted)
    expected = np.where(components.component_id_map == accepted[0], raw, 0)
    np.testing.assert_array_equal(curated, expected)
    assert changed == np.count_nonzero(expected)
    manifest = tmp_path/'initial.label-assistant.json'
    save_workspace(source, manifest, complete_package=True)
    reopened = make_napari_viewer(); load_workspace(reopened, manifest)
    for layer, expected_pixels in zip(reopened.layers, originals):
        np.testing.assert_array_equal(layer.data[:], expected_pixels)
    pairs=[]
    stride=4 if mode=='multilabel' else 2
    for n, seed in enumerate((71,72,73)):
        info=dict(image_id=f'synthetic-{seed}', image_layer=f'image {seed}', split_group=f'specimen-{seed}')
        image=reopened.layers[n*stride].data
        truth=reopened.layers[n*stride+1].data
        if mode=='multilabel':
            # Missing supervision for one channel in one image is intentional.
            reviewed=np.ones(image.shape,np.uint8); reviewed[:8]=0
            channels=[dict(data=reopened.layers[n*stride+2].data, reviewed=reviewed),
                      dict(data=reopened.layers[n*stride+3].data, reviewed=reviewed)]
            pairs.append(dict(image=image, channels=channels, info=info))
        else:
            pairs.append(dict(image=image, ground_truth=truth, info=info))
    dataset, indexing_updates=consume(index_multilabel(pairs,config) if mode=='multilabel' else index_pairs(pairs,config))
    train_ids={p['image_id'] for p in dataset.locations if p['split']=='training'}
    validation_ids={p['image_id'] for p in dataset.locations if p['split']=='validation'}
    assert not train_ids & validation_ids
    model, training_updates=consume(train_indexed(dataset,config,epochs=2,extra_targets=0))
    checkpoint=tmp_path/'synthetic.pt'; save_model(model,checkpoint); loaded=load_model(checkpoint)
    prediction,_=consume(predict_region(model,originals[0],device='cpu'))
    repeated,_=consume(predict_region(loaded,originals[0],device='cpu'))
    np.testing.assert_allclose(prediction.probability,repeated.probability,rtol=0,atol=1e-7)
    labels=prediction_labels(prediction.probability,multilabel=config.multilabel)
    target=np.stack([originals[2],originals[3]]) if mode=='multilabel' else originals[1]
    expected_shape=(2,96,128) if mode=='multilabel' else (96,128)
    assert labels.shape==expected_shape and prediction.origin==(0,0)
    metrics=[]
    for k in range(2 if mode=='multilabel' else 2 if mode=='multiclass' else 1):
        pred=labels[k]>0 if mode=='multilabel' else labels==k+1 if mode=='multiclass' else labels>0
        truth=target[k]>0 if mode=='multilabel' else target==k+1 if mode=='multiclass' else target>0
        tp=int((pred&truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum())
        metrics.append(dict(tp=tp,fp=fp,fn=fn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1)))
    # Independently verify aggregation on exactly the reserved model-input patches.
    from napari_label_assistant_tools._coverage_train import _validate
    independent_counts=np.zeros((len(config.class_names) or 1,3),np.int64)
    for index in dataset.indices('validation'):
        x,y,valid=dataset.read(index)
        with torch.no_grad():
            logits=loaded.network(torch.from_numpy(x)[None,None])
        probabilities=(torch.sigmoid(logits) if mode!='multiclass' else torch.softmax(logits,dim=1)).numpy()[0]
        predicted=probabilities>=.5 if mode=='multilabel' else probabilities.argmax(0) if mode=='multiclass' else probabilities[0]>=.5
        for k in range(len(independent_counts)):
            p=predicted[k] if mode=='multilabel' else predicted==k+1 if mode=='multiclass' else predicted
            t=y[k]>.5 if mode=='multilabel' else y==k+1 if mode=='multiclass' else y>.5
            v=valid[k]>.5 if mode=='multilabel' else valid>.5
            independent_counts[k]+=((p&t&v).sum(),(p&~t&v).sum(),(~p&t&v).sum())
    tp,fp,fn=independent_counts.T; eligible=(tp+fp+fn)>0
    expected_dice=float(np.mean((2*tp/np.maximum(2*tp+fp+fn,1))[eligible])) if eligible.any() else 0.
    expected_iou=float(np.mean((tp/np.maximum(tp+fp+fn,1))[eligible])) if eligible.any() else 0.
    observed_validation,_=consume(_validate(torch,loaded.network,dataset,'cpu',None,step=0,total=1))
    assert observed_validation['dice']==pytest.approx(expected_dice)
    assert observed_validation['iou']==pytest.approx(expected_iou)
    widget=QuickTrainWidget(reopened); qtbot.addWidget(widget)
    widget.model=loaded
    widget._prediction_done(reopened.layers[0],.5,prediction,scope='image')
    final_arrays=[np.asarray(layer.data[:]).copy() for layer in reopened.layers]
    final_manifest=tmp_path/'final.label-assistant.json'
    save_workspace(reopened,final_manifest,complete_package=True)
    final=make_napari_viewer(); load_workspace(final,final_manifest)
    for layer,expected_pixels in zip(final.layers,final_arrays):
        np.testing.assert_array_equal(layer.data[:],expected_pixels)
    record=dict(mode=mode, config=config.__dict__, shape=[96,128], training_steps=model.training_steps,
                train_image_ids=sorted(train_ids),validation_image_ids=sorted(validation_ids),
                patch_summary=dataset.summary(), retained_validation=model.metrics,
                independent_reserved_patch_metrics=dict(dice=expected_dice,iou=expected_iou,counts=independent_counts.tolist()),
                full_image_resubstitution_metrics=metrics, initial_hashes=[digest(a) for a in originals],
                final_hashes=[digest(a) for a in final_arrays], model_reload_max_absolute_difference=float(np.max(np.abs(prediction.probability-repeated.probability))),
                initial_and_final_pixel_roundtrips=True, component_copy_pixels=changed,
                checkpoint_workspace_reference_relinking='covered by existing suite; direct checkpoint reload exercised here',
                biological_generalization='not tested; these scores are pipeline checks only')
    (OUT/f'synthetic_{mode}.json').write_text(json.dumps(record,indent=2,default=str))
