"""Independent assertions against scientific/data-integrity requirements.

Failures are audit findings, not repairs or xfails. All data are synthetic.
"""
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from napari_label_assistant_tools._widget import (
    _build_fast_component_index, _copy_mask_components_to_target, component_operations_widget,
    _grid_index_from_xy, _compose_label_layers_distinct_classes,
)
from napari_label_assistant_tools._large_components import build_large_component_index
from napari_label_assistant_tools._deletion_history import DeletionHistory
from napari_label_assistant_tools._coverage_train import index_patches, index_pairs
from napari_label_assistant_tools._quick_train import (
    QuickTrainConfig, QuickTrainModel, QuickTrainPrediction, augment_batch,
    metrics_from_counts, segmentation_loss, predict_region, predict_area, QuickTrainCancelled,
)
from napari_label_assistant_tools.workspace import save_workspace, load_workspace, WORKSPACE_FORMAT

OUT = Path(__file__).resolve().parents[1] / 'evidence' / 'independent_observations'
OUT.mkdir(parents=True, exist_ok=True)


def observe(name, **values):
    (OUT / (name + '.json')).write_text(json.dumps(values, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def consume(generator):
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value


def cfg(**kwargs):
    return QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1,
                            base_channels=4, device='cpu', **kwargs)


def striped():
    yy, xx = np.indices((128, 160))
    target = ((xx // 4 + yy // 8) % 2).astype(np.uint8)
    return (target * 130 + yy % 7).astype(np.uint8), target


def oracle_components(array):
    """Pure Python 4-neighbor oracle; no scipy/skimage component routines."""
    seen, records = set(), []
    height, width = array.shape
    for y in range(height):
        for x in range(width):
            if not array[y, x] or (y, x) in seen:
                continue
            value, queue, points = int(array[y, x]), deque([(y, x)]), []
            seen.add((y, x))
            while queue:
                py, px = queue.popleft()
                points.append((py, px))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = py+dy, px+dx
                    if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and int(array[ny, nx]) == value:
                        seen.add((ny, nx)); queue.append((ny, nx))
            ys, xs = zip(*points)
            records.append((value, len(points), float(np.mean(ys)), float(np.mean(xs)),
                            min(ys), min(xs), max(ys)+1, max(xs)+1))
    return sorted(records)


def actual_records(records):
    return sorted((r.label_value, r.area, r.centroid_y, r.centroid_x,
                   r.bbox_y0, r.bbox_x0, r.bbox_y1, r.bbox_x1) for r in records.values())


def test_A01_prediction_world_coordinates_survive_workspace_roundtrip(make_napari_viewer, qtbot, tmp_path):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    image = viewer.add_image(np.arange(32*40, dtype=np.uint16).reshape(32, 40),
                             scale=(2, 3), translate=(7, 11), rotate=25, shear=(.2,))
    widget = QuickTrainWidget(viewer); qtbot.addWidget(widget)
    widget._prediction_done(image, .5, QuickTrainPrediction(np.ones((5, 7), np.float32), (9, 13)))
    prediction = viewer.layers[-1]
    points = [(0, 0), (1, 0), (0, 1), (4, 6)]
    before = np.asarray([prediction.data_to_world(p) for p in points])
    pixels = prediction.data.copy()
    path = tmp_path / 'prediction.label-assistant.json'
    save_workspace(viewer, path, complete_package=True)
    restored = make_napari_viewer(); load_workspace(restored, path)
    after = np.asarray([restored.layers[-1].data_to_world(p) for p in points])
    observe('A01_affine_roundtrip', expected_world=before, actual_world=after,
            pixels_equal=np.array_equal(restored.layers[-1].data[:], pixels),
            stored_prediction_record=json.loads(path.read_text())['layers'][-1])
    np.testing.assert_allclose(after, before, atol=1e-9)


def test_A02_edit_tile_matches_rotated_sheared_source(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    source = viewer.add_labels(np.zeros((64, 80), np.uint32), scale=(2, 3),
                               translate=(7, 11), rotate=30, shear=(.2,))
    widget = component_operations_widget(viewer); qtbot.addWidget(widget)
    widget._editing_strategy_combo.setCurrentIndex(widget._editing_strategy_combo.findData('tiled'))
    controller = widget._dynamic_edit_controller
    qtbot.waitUntil(lambda: controller.tile is not None, timeout=10000)
    points = [(0, 0), (1, 0), (0, 1), (6, 9)]
    expected = np.asarray([source.data_to_world((p[0]+controller.bounds.y0, p[1]+controller.bounds.x0)) for p in points])
    actual = np.asarray([controller.tile.data_to_world(p) for p in points])
    observe('A02_edit_transform', expected_world=expected, actual_world=actual)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_A03_large_copy_cannot_silently_wrap_label_values(make_napari_viewer, qtbot):
    from qtpy.QtWidgets import QPushButton, QProgressBar
    from napari_label_assistant_tools._large_component_controller import LargeComponentController
    viewer = make_napari_viewer()
    source_data = np.zeros((8, 9), np.uint16); source_data[1:4, 2:6] = 300
    source = viewer.add_labels(source_data)
    target = viewer.add_labels(np.zeros(source_data.shape, np.uint8))
    index = consume(build_large_component_index(source_data, tile_size=4))
    status, done = [], []
    controller = LargeComponentController(progress=QProgressBar(), cancel_button=QPushButton(),
        analyze_button=QPushButton(), status=status.append, on_ready=lambda *args: None)
    try:
        controller.edit_components(index=index, source_layer=source, target_layer=target,
            component_ids=list(index.records), delete=False, on_done=done.append)
        qtbot.waitUntil(lambda: not controller.busy, timeout=10000)
        values = np.unique(target.data).tolist()
        observe('A03_large_copy_overflow', source_value=300, target_dtype=str(target.data.dtype),
                actual_values=values, completed_pixels=done, status=status)
        assert 44 not in values, '300 wrapped to 44 in uint8 destination without rejection/promotion'
    finally:
        controller.close(); index.close()


def test_A04_copy_rejects_spatially_incompatible_destination(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((16, 20), np.uint16); data[2:5, 4:8] = 7
    source = viewer.add_labels(data, name='source')
    target = viewer.add_labels(np.zeros_like(data), name='shifted target', translate=(20, 30))
    widget = component_operations_widget(viewer); qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda: widget._component_table.rowCount() > 0)
    widget._component_table.selectRow(0)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(target))
    widget._copy_button.click()
    observe('A04_copy_alignment', target_nonzero=int(np.count_nonzero(target.data)),
            source_world=source.data_to_world((2,4)), target_world=target.data_to_world((2,4)),
            status=widget._status_label.text())
    assert not target.data.any(), 'Copy accepted equal shapes with different spatial transforms'


def test_A05_zarr_replacement_does_not_save_stale_original_store(make_napari_viewer, tmp_path):
    import zarr
    first = zarr.open_array(str(tmp_path/'first.zarr'), mode='w', shape=(12,13), chunks=(4,5), dtype='uint32')
    second = zarr.open_array(str(tmp_path/'second.zarr'), mode='w', shape=(12,13), chunks=(4,5), dtype='uint32')
    first[:] = 2; second[:] = 9
    viewer = make_napari_viewer(); layer = viewer.add_labels(first)
    path = tmp_path/'case.label-assistant.json'
    save_workspace(viewer, path)
    layer.data = second
    save_workspace(viewer, path)
    restored = make_napari_viewer(); load_workspace(restored, path)
    actual = restored.layers[0].data[:]
    observe('A05_stale_zarr_binding', expected_value=9, actual_values=np.unique(actual),
            storage=json.loads(path.read_text())['layers'][0]['storage'])
    np.testing.assert_array_equal(actual, np.full((12,13),9,np.uint32))


def test_A06_failed_open_retains_previous_live_layers(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer(); previous = viewer.add_labels(np.full((12,13),7,np.uint16), name='unsaved')
    path = tmp_path/'missing.label-assistant.json'
    path.write_text(json.dumps(dict(format=WORKSPACE_FORMAT, version=2, layers=[
        dict(layer_type='Labels', name='unavailable', storage=dict(kind='zarr', path='absent.zarr',array_path=''))])))
    result = load_workspace(viewer, path)
    observe('A06_failed_open', remaining_names=[x.name for x in viewer.layers], result=result,
            qualified_scope='service path; Workspace UI asks confirmation before removing existing layers')
    assert any(layer is previous for layer in viewer.layers), 'All requested layers failed after current live layers were cleared'


def test_A07_current_pair_preparation_rejects_misaligned_labels(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    image, truth = striped()
    viewer = make_napari_viewer(); viewer.add_image(image)
    labels = viewer.add_labels(truth, translate=(9,11))
    widget = QuickTrainWidget(viewer); qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    for combo in (widget.crop_combo,widget.resolution_combo):
        combo.addItem('32 px',32); combo.setCurrentIndex(combo.findData(32))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    observe('A07_current_pair_alignment', accepted=widget.prepared is not None,
            label_world=labels.data_to_world((0,0)), image_world=viewer.layers[0].data_to_world((0,0)),
            status=widget.status_label.text())
    assert widget.prepared is None, 'Current-pair preparation accepted a world-shifted target'


def test_A08_dataset_fingerprint_changes_with_image_pixels():
    from napari_label_assistant_tools._model_comparison import dataset_snapshot
    image, truth = striped()
    before = dataset_snapshot(consume(index_patches(image,truth,cfg())))['id']
    changed = image.copy(); changed[3:6,7:12] += 1
    after = dataset_snapshot(consume(index_patches(changed,truth,cfg())))['id']
    observe('A08_snapshot_identity', before=before, after=after, raw_images_equal=False)
    assert before != after, 'Normal training dataset identity omitted actual image pixels'


def test_A09_components_agree_with_independent_bfs():
    mask = np.zeros((19,27),np.uint64)
    mask[0:7,0:4]=2**40+9; mask[3:11,9:16]=2**40+9
    mask[5:9,11:14]=0; mask[17,25]=3; mask[18,26]=3
    expected=oracle_components(mask)
    fast=_build_fast_component_index(mask)
    tiled=consume(build_large_component_index(mask,tile_size=6))
    try:
        observe('A09_bfs_components', expected=expected, fast=actual_records(fast.records), tiled=actual_records(tiled.records))
        assert actual_records(fast.records)==expected
        assert actual_records(tiled.records)==expected
    finally:
        tiled.close()


def test_A10_topology_convention_matches_four_connected_foreground():
    # Center background connects diagonally to outside: no hole under the dual
    # 8-connected-background convention for 4-connected foreground.
    mask=np.array([[1,1,1],[1,0,1],[1,1,0]],np.uint8)
    actual=next(iter(_build_fast_component_index(mask).records.values())).euler
    observe('A10_connectivity', mask=mask, foreground_neighbors=4, expected_dual_euler=1,
            actual_euler=actual, qualification='The implementation explicitly computes 8-foreground Euler while identifying components with 4-neighbor connectivity.')
    assert actual==1


@pytest.mark.parametrize('tp,fp,fn',[(2,1,1),(0,0,0),(0,3,2),(7,0,0)])
def test_A11_independent_metric_arithmetic(tp,fp,fn):
    expected=dict(precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                  dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1))
    actual=metrics_from_counts(tp,fp,fn)
    observe(f'A11_metrics_{tp}_{fp}_{fn}', expected=expected, actual=actual,
            empty_convention='zero, not perfect score, when both foreground sets are empty')
    assert actual==expected


def test_A12_unknown_multilabel_pixels_have_zero_gradient():
    import torch
    logits=torch.zeros((1,3,5,7),requires_grad=True)
    truth=torch.zeros_like(logits);truth[:,0,1:4,1:5]=1;truth[:,1,2:4,2:5]=1
    valid=torch.ones_like(logits);valid[:,2]=0;valid[:,1,:,5:]=0
    loss=segmentation_loss(torch,logits,truth,valid,multilabel=True);loss.backward()
    observe('A12_unknown_gradient', loss=float(loss.detach()),
            unknown_gradients=int(torch.count_nonzero(logits.grad[valid==0])), known_gradients=int(torch.count_nonzero(logits.grad[valid!=0])))
    assert torch.count_nonzero(logits.grad[valid==0])==0
    assert torch.count_nonzero(logits.grad[valid!=0])>0


def test_A13_augmentation_geometry_is_joint_and_sources_unchanged():
    original=np.arange(2*7*7,dtype=np.float32).reshape(2,7,7)
    truth=original.copy();valid=original.copy()
    config=cfg(augment_intensity=False)
    x,y,v=augment_batch(original,truth,valid,np.random.default_rng(71),config)
    assert np.array_equal(x,y) and np.array_equal(x,v)
    np.testing.assert_array_equal(original,np.arange(98,dtype=np.float32).reshape(2,7,7))
    observe('A13_augmentation', geometric_alignment=True, source_unchanged=True)


def test_A14_tile_edges_overlap_and_cancellation():
    import torch
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.full_like(x,np.log(4))
    model=QuickTrainModel(Constant(),cfg(),32,{},'cpu')
    image=np.arange(43*97,dtype=np.float32).reshape(43,97)
    prediction=consume(predict_region(model,image,device='cpu'))
    np.testing.assert_allclose(prediction.probability,.8,atol=1e-6)
    assert prediction.probability.shape==image.shape
    cropped=consume(predict_region(model,image,(3,5,40,91),device='cpu'))
    assert cropped.origin==(3,5) and cropped.probability.shape==(37,86)
    cancel=Event();task=predict_region(model,image,device='cpu',cancel=cancel);next(task);cancel.set()
    with pytest.raises(QuickTrainCancelled):next(task)
    observe('A14_tiling', full_shape=prediction.probability.shape,
            probability_min=float(prediction.probability.min()), probability_max=float(prediction.probability.max()),
            crop_shape=cropped.probability.shape, crop_origin=cropped.origin,cancelled=True)


def test_A15_huge_inference_allocation_guard(monkeypatch):
    import napari_label_assistant_tools._quick_train as module
    class Description:
        shape=(100000,100000)
        def __getitem__(self,key):raise AssertionError('No source pixels should be read before the guard')
    allocations=[];original=np.zeros
    class GuardedAllocation(Exception):pass
    def zeros(shape,*args,**kwargs):
        if np.prod(shape)>1_000_000:
            allocations.append(dict(shape=list(shape),bytes=int(np.prod(shape))*np.dtype(kwargs.get('dtype',float)).itemsize))
            raise GuardedAllocation()
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(module.np,'zeros',zeros)
    model=SimpleNamespace(config=cfg(),crop_pixels=32)
    with pytest.raises(GuardedAllocation):consume(module.predict_region(model,Description(),device='cpu'))
    observe('A15_allocation_request', actual_guarded_requests=allocations,
            binary_probability_plus_weight_bytes=80_000_000_000,
            four_channel_probability_plus_weight_bytes=200_000_000_000,
            qualification='Allocation request intercepted; no huge array allocated. Byte totals are code-derived, not measured peak RSS.')
    assert allocations[0]['bytes']==40_000_000_000


def test_A16_grid_boundaries_and_deletion_recovery(tmp_path):
    rows=[_grid_index_from_xy(y,x,4,6) for y,x in ((0,0),(3.999,5.999),(4,6),(10,14))]
    assert rows==[(0,0,'R00C00'),(0,0,'R00C00'),(1,1,'R01C01'),(2,2,'R02C02')]
    layer=SimpleNamespace(data=np.zeros((11,15),np.uint64));layer.data[2:5,3:8]=2**40+9
    history=DeletionHistory(layer);slices=(slice(0,7),slice(0,10));mask=layer.data[slices]>0
    history.capture(slices,mask);tile=layer.data[slices].copy();tile[mask]=0;layer.data[slices]=tile
    layer.data[3,4]=77
    try:
        events=list(history.restore())
        assert layer.data[3,4]==77 and layer.data[2,3]==2**40+9
        observe('A16_grid_undo', addresses=rows, restored=events[-1][:2], high_id=int(layer.data[2,3]))
    finally:history.close()


def test_A17_grouped_validation_requires_explicit_correct_specimen_ids():
    image,truth=striped()
    pairs=[dict(image=image.copy(),ground_truth=truth.copy(),info=dict(image_id=f'id-{i}',
                image_layer=f'clone-{i}',ground_truth_id=f'truth-{i}',split_group=f'group-{i}')) for i in range(3)]
    dataset=consume(index_pairs(pairs,cfg()))
    train={p['image_id'] for p in dataset.locations if p['split']=='training'}
    valid={p['image_id'] for p in dataset.locations if p['split']=='validation'}
    assert not train&valid
    observe('A17_manual_grouping', training_ids=sorted(train), validation_ids=sorted(valid),
            all_input_images_byte_identical=True,
            qualification='Group labels are enforced, but biological independence/deduplication is not inferred.')


def test_A18_plain_roundtrip_preserves_uint64_labels_and_raw_intensity(make_napari_viewer,tmp_path):
    viewer=make_napari_viewer()
    image=np.arange(31*47,dtype=np.uint16).reshape(31,47)
    labels=np.zeros(image.shape,np.uint64);labels[0:5,0:3]=2**40+9;labels[30,46]=2**53+1
    viewer.add_image(image.copy(),scale=(.2,.3),translate=(4,5))
    viewer.add_labels(labels.copy(),scale=(.2,.3),translate=(4,5))
    path=tmp_path/'exact.label-assistant.json';save_workspace(viewer,path,complete_package=True)
    restored=make_napari_viewer();load_workspace(restored,path)
    np.testing.assert_array_equal(restored.layers[0].data[:],image)
    np.testing.assert_array_equal(restored.layers[1].data[:],labels)
    assert restored.layers[1].data.dtype==labels.dtype
    observe('A18_exact_roundtrip', image_sha256=sha256(image.tobytes()).hexdigest(),
            labels_sha256=sha256(labels.tobytes()).hexdigest(), exact=True,
            transforms='scale and translation only; full affine tested separately in A01')
