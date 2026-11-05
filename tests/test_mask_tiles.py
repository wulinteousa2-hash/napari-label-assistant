"""Bounded mask loading and exact global persistence on disposable stores."""
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from napari_label_assistant_tools._mask_tiles import manager_for, full_data, set_mode
from napari_label_assistant_tools.workspace import WORKSPACE_FORMAT, load_workspace, save_workspace


def workspace(tmp_path, shape=(1050, 1530), mode='grid', row=0, col=0, size=512):
    layers = []
    arrays = []
    for index, (name, dtype) in enumerate((('Curated', 'uint32'), ('Reviewed regions', 'uint8'), ('Reviewed annotations', 'uint8'))):
        path = tmp_path/f'{index}.zarr'
        array = zarr.open_array(str(path), mode='w', shape=shape, chunks=(256,256), dtype=dtype)
        array[5:8, 6:9] = 7 if index == 0 else 1
        array[-2:, -3:] = 2**31+17 if index == 0 else 1
        record = dict(layer_type='Labels', name=name, scale=[2,3], translate=[7,11],
            rotate=[[.866025403784,-.5],[.5,.866025403784]], shear=[.2],
            affine=[[1,.1,4],[0,1,5],[0,0,1]],
            storage=dict(kind='zarr',path=path.name,array_path='',shape=list(shape),chunks=[256,256],dtype=dtype))
        record['label_assistant_review'] = dict(id=f'mask-{index}')
        if index:
            record['label_assistant_review'].update(role='reviewed_regions' if index==1 else 'reviewed_annotations',truth_id='mask-0')
        layers.append(record); arrays.append(array)
    payload = dict(format=WORKSPACE_FORMAT,version=2,layers=layers,
                   mask_view=dict(mode=mode,tile_size=size,row=row,col=col))
    path = tmp_path/'case.label-assistant.json'; path.write_text(json.dumps(payload))
    return path, arrays


def test_automatic_huge_open_never_passes_full_masks_to_napari(tmp_path,make_napari_viewer,monkeypatch):
    path, arrays = workspace(tmp_path,shape=(84175,79966),mode='auto',size=1024)
    viewer = make_napari_viewer()
    original = type(viewer).add_labels
    calls=[]
    def bounded(self,data,*args,**kwargs):
        assert isinstance(data,np.ndarray)
        assert max(data.shape)<=1024 and data.nbytes<=4*1024**2
        calls.append(data.shape)
        return original(self,data,*args,**kwargs)
    monkeypatch.setattr(type(viewer),'add_labels',bounded)
    result=load_workspace(viewer,path)
    assert not result['skipped_layers'] and len(calls)==3
    manager=manager_for(viewer)
    assert len(manager.entries)==3
    assert all(full_data(layer).shape==(84175,79966) for layer in viewer.layers)
    assert all(layer.data.shape==(1024,1024) for layer in viewer.layers)
    manager.go(82,78)
    assert manager.entries[0].layer.data.shape==(207,94)
    assert int(manager.entries[0].layer.data[-1,-1])==2**31+17


def test_navigation_paint_erase_global_alignment_and_progress(tmp_path,make_napari_viewer):
    path,arrays=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);entry=manager.entries[0]
    entry.layer.brush_size=1
    entry.layer.paint((5,6),0)
    entry.layer.paint((10,12),19)
    manager.go(1,1)
    assert arrays[0][5,6]==0 and arrays[0][10,12]==19
    entry.layer.paint((4,6),23)
    manager.go(2,2)
    assert arrays[0][516,518]==23
    assert int(arrays[0][-1,-1])==2**31+17
    assert entry.layer.data.shape==(26,506)
    for point in ((0,0),(1,0),(0,1),(25,505)):
        expected=entry.source.data_to_world((point[0]+1024,point[1]+1024))
        np.testing.assert_allclose(entry.layer.data_to_world(point),expected,atol=1e-9)
    manager.progress['512:2:2']='done'
    save_workspace(viewer,path)
    payload=json.loads(path.read_text())
    assert payload['layers'][0]['storage']['shape']==[1050,1530]
    assert payload['layers'][0]['translate']==[7.,11.]
    assert payload['mask_view']['row']==2 and payload['mask_view']['progress']['512:2:2']=='done'
    other=make_napari_viewer();load_workspace(other,path)
    restored=manager_for(other)
    assert (restored.row,restored.col,restored.size)==(2,2,512)
    assert int(other.layers['Curated'].data[-1,-1])==2**31+17


def test_include_all_data_copies_entire_masks_and_rebinds_stores(tmp_path,make_napari_viewer):
    path,arrays=workspace(tmp_path)
    originals=[array[:] for array in arrays]
    viewer=make_napari_viewer();load_workspace(viewer,path)
    destination=tmp_path/'portable'/'project.label-assistant.json'
    save_workspace(viewer,destination,complete_package=True,copy_labels=True,xy_chunk=256)
    payload=json.loads(destination.read_text())
    assert all(record['storage']['shape']==[1050,1530] for record in payload['layers'])
    assert all(not Path(record['storage']['path']).is_absolute() for record in payload['layers'])
    manager=manager_for(viewer)
    manager.entries[0].layer.data[10,10]=37
    save_workspace(viewer,destination)
    assert arrays[0][10,10]==0  # Save As isolates subsequent edits from originals.
    moved=tmp_path/'moved';destination.parent.rename(moved)
    restored=make_napari_viewer();load_workspace(restored,moved/destination.name)
    for index,layer in enumerate(restored.layers):
        expected=originals[index]
        if index==0: expected[10,10]=37
        np.testing.assert_array_equal(full_data(layer)[:],expected)


def test_plain_and_full_modes_keep_existing_loading_behavior(tmp_path,make_napari_viewer):
    path,_=workspace(tmp_path,mode='full')
    viewer=make_napari_viewer();load_workspace(viewer,path)
    assert manager_for(viewer) is None
    assert viewer.layers[0].data.shape==(1050,1530)


def test_explicit_pre_open_choice_overrides_saved_mode(tmp_path,make_napari_viewer):
    path,_=workspace(tmp_path,mode='full')
    viewer=make_napari_viewer()
    set_mode(viewer,'grid')
    load_workspace(viewer,path)
    assert manager_for(viewer) is not None
    assert viewer._label_assistant_editing_strategy=='grid'
    set_mode(viewer,'tiled')
    load_workspace(viewer,path)
    assert manager_for(viewer) is None
    assert viewer._label_assistant_editing_strategy=='tiled'
    set_mode(viewer,'grid')
    assert viewer.layers[0].data.shape==(1024,1024)
    set_mode(viewer,'tiled')
    assert manager_for(viewer) is None
    assert viewer.layers[0].data.shape==(1050,1530)


def test_conflicting_writer_is_not_overwritten(tmp_path,make_napari_viewer):
    path,arrays=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);entry=manager.entries[0]
    entry.layer.data[10,10]=17
    arrays[0][10,10]=22
    with pytest.raises(ValueError,match='another writer'):
        manager.go(1,1)
    assert manager.row==0 and arrays[0][10,10]==22 and entry.layer.data[10,10]==17
    entry.layer.data[10,10]=22;manager.flush()


def test_failed_write_retains_recovery_and_reopen_replays_it(tmp_path,make_napari_viewer,monkeypatch):
    path,arrays=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);entry=manager.entries[0]
    source=entry.source.data
    class Fails:
        shape=source.shape;dtype=source.dtype
        def __getitem__(self,key): return source[key]
        def __setitem__(self,key,value): raise OSError('simulated disk write interruption')
    entry.source.data=Fails()
    entry.layer.data[10,10]=31
    with pytest.raises(OSError,match='interruption'):
        manager.go(1,1)
    assert manager.row==0 and entry.layer.data[10,10]==31
    assert list((tmp_path/'.label_assistant_tile_recovery').glob('*.npz'))
    entry.source.data=source
    reopened=make_napari_viewer();result=load_workspace(reopened,path)
    assert not result['skipped_layers']
    assert arrays[0][10,10]==31
    assert not list((tmp_path/'.label_assistant_tile_recovery').glob('*.npz'))
    manager.flush()


def test_erase_after_partial_write_failure_is_not_lost(tmp_path,make_napari_viewer):
    path,arrays=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);entry=manager.entries[0];source=entry.source.data
    class Partial:
        shape=source.shape;dtype=source.dtype
        def __getitem__(self,key):return source[key]
        def __setitem__(self,key,value):
            source[key]=value
            raise OSError('interrupted after pixels reached disk')
    entry.source.data=Partial();entry.layer.data[10,10]=31
    with pytest.raises(OSError):manager.flush()
    assert arrays[0][10,10]==31
    entry.source.data=source
    entry.layer.data[10,10]=0
    manager.flush()
    assert arrays[0][10,10]==0


def test_viewer_close_flushes_pending_edits(tmp_path,make_napari_viewer):
    path,arrays=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    viewer.layers[0].data[10,10]=41
    viewer.close()
    assert arrays[0][10,10]==41


def test_tile_controls_and_existing_mode_selector(tmp_path,make_napari_viewer,qtbot):
    from napari_label_assistant_tools._widget import component_operations_widget
    path,_=workspace(tmp_path)
    viewer=make_napari_viewer();widget=component_operations_widget(viewer);qtbot.addWidget(widget)
    load_workspace(viewer,path)
    manager=manager_for(viewer)
    controls=widget._grid_tiles
    qtbot.waitUntil(lambda:widget._edit_tile_size_combo.isHidden(),timeout=2000)
    controls._row.setValue(1);controls._col.setValue(1);controls._size.setCurrentIndex(controls._size.findData(512))
    controls._go.click()
    assert (manager.row,manager.col)==(1,1)
    controls._done.click()
    assert manager.progress['512:1:1']=='done'
    controls._next.click()
    assert (manager.row,manager.col)==(1,2)


def test_new_crop_mask_is_managed_and_saved_at_full_extent(tmp_path,make_napari_viewer):
    from napari_label_assistant_tools._coordinates import crop_transform_kwargs
    path,_=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);manager.go(1,1)
    reference=manager.entries[0]
    pixels=np.zeros(reference.layer.data.shape,np.uint16);pixels[4,6]=300
    layer=viewer.add_labels(pixels,name='new objects',**crop_transform_kwargs(reference.layer))
    assert len(manager.entries)==4
    assert full_data(layer).shape==(1050,1530)
    save_workspace(viewer,path)
    restored=make_napari_viewer();load_workspace(restored,path)
    assert full_data(restored.layers['new objects'])[516,518]==300
    assert full_data(restored.layers['new objects'])[0,0]==0
    assert restored.layers['new objects'].data.shape==(512,512)


def test_only_dirty_bounding_region_is_written(tmp_path,make_napari_viewer):
    path,_=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);entry=manager.entries[0];source=entry.source.data
    writes=[]
    class Spy:
        shape=source.shape;dtype=source.dtype
        def __getitem__(self,key): return source[key]
        def __setitem__(self,key,value):
            writes.append(key);source[key]=value
    entry.source.data=Spy()
    entry.layer.data[10:12,20:23]=19
    manager.flush()
    assert writes==[(slice(10,12),slice(20,23))]
    entry.source.data=source


def test_dtype_overflow_and_transform_change_refuse_to_write(tmp_path,make_napari_viewer):
    from napari_label_assistant_tools._coordinates import apply_crop_transform
    path,arrays=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);entry=manager.entries[1]
    pixels=entry.layer.data.astype(np.uint16);pixels[10,10]=300;entry.layer.data=pixels
    with pytest.raises(ValueError,match='exceed'):
        manager.flush()
    assert arrays[1][10,10]==0
    entry.layer.data[10,10]=23
    entry.layer.translate=entry.layer.translate+(1,0)
    with pytest.raises(ValueError,match='transform changed'):
        manager.flush()
    apply_crop_transform(entry.layer,entry.source)
    manager.flush();assert arrays[1][10,10]==23


def test_different_mask_extents_do_not_show_an_unrelated_edge_tile(tmp_path,make_napari_viewer):
    path,_=workspace(tmp_path)
    payload=json.loads(path.read_text())
    small=zarr.open_array(str(tmp_path/'small.zarr'),mode='w',shape=(128,256),chunks=(128,128),dtype='uint8')
    small[0,0]=3
    payload['layers'].append(dict(layer_type='Labels',name='small',storage=dict(kind='zarr',path='small.zarr',array_path='')))
    path.write_text(json.dumps(payload))
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);manager.go(1,1)
    entry=viewer.layers['small']._label_assistant_mask_entry
    assert entry.bounds is None and not entry.layer.visible and not entry.layer.editable
    manager.go(0,0)
    assert viewer.layers['small'].data[0,0]==3 and viewer.layers['small'].editable


def test_training_prepares_full_backend_not_active_crop(tmp_path,make_napari_viewer,qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    path,arrays=workspace(tmp_path,shape=(1024,1536))
    yy,xx=np.indices(arrays[0].shape)
    truth=((xx//32+yy//32)%2).astype(np.uint32)
    arrays[0][:]=truth
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager_for(viewer).go(1,1)
    image=viewer.add_image((truth*130).astype(np.uint8),name='image',scale=(2,3),translate=(7,11),
        rotate=[[.866025403784,-.5],[.5,.866025403784]],shear=(.2,),affine=[[1,.1,4],[0,1,5],[0,0,1]])
    widget=QuickTrainWidget(viewer);qtbot.addWidget(widget)
    widget.image_combo.setCurrentIndex(widget.image_combo.findData(image))
    widget.truth_combo.setCurrentIndex(widget.truth_combo.findData(viewer.layers['Curated']))
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    widget.crop_combo.setCurrentIndex(widget.crop_combo.findData(512))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda:widget.worker is None,timeout=20000)
    assert widget.prepared is not None,widget.status_label.text()
    assert widget.prepared.ground_truth.shape==(1024,1536)
    assert max(item['x1'] for item in widget.prepared.locations)==1536


def test_global_library_uses_full_store_and_global_dirty_bounds(tmp_path,make_napari_viewer,qtbot):
    from napari_label_assistant_tools._component_library_widget import ComponentLibraryWidget
    from napari_label_assistant_tools._component_library import library_for
    path,_=workspace(tmp_path)
    viewer=make_napari_viewer();load_workspace(viewer,path)
    manager=manager_for(viewer);layer=viewer.layers['Curated']
    manager.go(1,1)
    widget=ComponentLibraryWidget(viewer,lambda:layer);qtbot.addWidget(widget)
    widget.start_update(verify_all=True)
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    library=library_for(layer)
    assert library.shape==(1050,1530)
    records=library.records()
    assert len(records)==2
    index=next(i for i,item in enumerate(records) if item['y0']>1000)
    widget._locate(index,1)
    assert (manager.row,manager.col)==(2,2)
    layer.data[0,0]=23
    manager.flush()
    assert library.state()['pending']
    save_workspace(viewer,path)
    restored=make_napari_viewer();load_workspace(restored,path)
    assert library_for(restored.layers['Curated']).shape==(1050,1530)


def test_mark_done_marks_workspace_changed(tmp_path,make_napari_viewer,qtbot):
    from napari_label_assistant_tools._label_assistant import label_assistant_widget
    path,_=workspace(tmp_path)
    viewer=make_napari_viewer();page=label_assistant_widget(viewer);qtbot.addWidget(page)
    load_workspace(viewer,path)
    workspace_widget=page._tool_tabs.widget(0)
    workspace_widget.save_state_label.setText('Saved')
    page._tool_tabs.widget(1)._grid_tiles._done.click()
    assert 'changed since last save' in workspace_widget.save_state_label.text()
