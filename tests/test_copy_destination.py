"""Manual destination intent, cancellation, matching, and full-mask ROI copying."""
import json

import numpy as np
import pytest
import zarr
from qtpy.QtCore import QPoint, QTimer
from qtpy.QtWidgets import QApplication, QMessageBox, QMenu

from napari_label_assistant_tools import _copy_destination as destination
from napari_label_assistant_tools._widget import component_operations_widget
from napari_label_assistant_tools._mask_tiles import full_data, manager_for
from napari_label_assistant_tools.workspace import WORKSPACE_FORMAT, load_workspace, save_workspace


def setup(viewer, qtbot, shape=(12,14), target_shape=(8,9), target_value=0, source_translate=(0,0)):
    pixels=np.zeros(shape,np.uint16);pixels[1:3,2:5]=300
    source=viewer.add_labels(pixels,name='Candidates',translate=source_translate)
    target=None
    if target_shape is not None:
        data=np.zeros(target_shape,np.uint16);data[0,0]=target_value
        target=viewer.add_labels(data,name='Curated')
    widget=component_operations_widget(viewer);qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda:widget._component_table.rowCount()>0 and not widget._large_controller.busy,timeout=15000)
    widget._component_table.select_component_id(1)
    widget._manual_copy_target_check.setChecked(True)
    if target is not None:
        widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(target))
    return widget,source,target,pixels


def test_cancel_leaves_existing_destination_and_layer_names_exact(make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer();widget,source,target,pixels=setup(viewer,qtbot,target_value=77)
    before=target.data.copy();names=[layer.name for layer in viewer.layers]
    calls=[]
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:calls.append(args) or False)
    widget._copy_button.click()
    assert len(calls)==1 and calls[0][-1] is True
    np.testing.assert_array_equal(target.data,before)
    np.testing.assert_array_equal(source.data,pixels)
    assert [layer.name for layer in viewer.layers]==names
    assert 'cancelled' in widget._status_label.text()


@pytest.mark.parametrize('tiled',[False,True])
def test_matching_retains_overlap_backup_and_restore(tiled,make_napari_viewer,qtbot,monkeypatch):
    if tiled:
        monkeypatch.setattr('napari_label_assistant_tools._widget.LARGE_ANALYSIS_PIXEL_THRESHOLD',1)
    viewer=make_napari_viewer();widget,source,target,pixels=setup(viewer,qtbot,target_value=77)
    original=target.data.copy();calls=[]
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:calls.append(args) or True)
    widget._copy_button.click();qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=15000)
    matched=viewer.layers['Curated']
    assert matched is not target and matched.data.shape==source.data.shape
    assert matched.data[0,0]==77
    np.testing.assert_array_equal(matched.data[1:3,2:5],pixels[1:3,2:5])
    np.testing.assert_array_equal(target.data,original)
    assert target.name.startswith('Curated — original')
    widget._copy_button.click();qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=15000)
    assert len(calls)==1
    widget._restore_copy_destination()
    assert viewer.layers['Curated'] is target
    np.testing.assert_array_equal(target.data,original)
    assert matched.name.startswith('Curated — copied result')


def test_realignment_keeps_original_coordinates_and_pixels(make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer();widget,source,target,pixels=setup(viewer,qtbot,target_shape=(12,14),target_value=77)
    target.rotate=35;target.scale=(2,3)
    points=[(0,0),(11,13)]
    old_world=np.asarray([target.data_to_world(p) for p in points])
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:True)
    widget._copy_button.click()
    matched=viewer.layers['Curated']
    np.testing.assert_allclose([matched.data_to_world(p) for p in points],[source.data_to_world(p) for p in points])
    np.testing.assert_allclose([target.data_to_world(p) for p in points],old_world)
    assert target.data[0,0]==77
    np.testing.assert_array_equal(matched.data,pixels)


def test_matching_retains_large_overlap_in_bounded_background_work(make_napari_viewer,qtbot,monkeypatch):
    monkeypatch.setattr(destination,'MEMORY_LIMIT',64)
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:True)
    viewer=make_napari_viewer();widget,source,target,pixels=setup(viewer,qtbot,target_value=77)
    widget._copy_button.click()
    qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=15000)
    matched=viewer.layers['Curated']
    assert 'Copy stopped' not in widget._status_label.text(),widget._status_label.text()
    expected=pixels.copy();expected[0,0]=77
    np.testing.assert_array_equal(full_data(matched)[:],expected)
    assert target.data[0,0]==77 and not target.visible
    assert source.editable and matched.editable


def test_mixed_integer_types_do_not_round_existing_ids(make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer()
    pixels=np.zeros((12,14),np.int64);pixels[1:3,2:5]=7
    source=viewer.add_labels(pixels,name='Candidates')
    old=np.zeros((12,14),np.uint64);old[0,0]=2**53+1
    target=viewer.add_labels(old,name='Curated')
    widget=component_operations_widget(viewer);qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name);widget._analyze_button.click()
    qtbot.waitUntil(lambda:widget._component_table.rowCount()>0)
    widget._component_table.select_component_id(1)
    widget._manual_copy_target_check.setChecked(True)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(target))
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:True)
    widget._copy_button.click();qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=10000)
    assert target.data.dtype==np.uint64 and int(target.data[0,0])==2**53+1
    assert np.all(target.data[1:3,2:5]==7)


@pytest.mark.parametrize('accept',[False,True])
def test_default_copy_prepares_mismatched_curated_after_confirmation(accept,make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer();widget,_,target,pixels=setup(viewer,qtbot)
    widget._manual_copy_target_check.setChecked(False)
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:accept)
    widget._copy_button.click()
    assert widget._manual_copy_target_check.isChecked()
    if accept:
        np.testing.assert_array_equal(viewer.layers['Curated'].data,pixels)
        assert target.name.startswith('Curated — original')
    else:
        assert viewer.layers['Curated'] is target and not target.data.any()


def test_roi_copies_into_larger_destination_without_resizing(make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer();widget,source,target,pixels=setup(viewer,qtbot,shape=(4,6),target_shape=(32,40),target_value=77,source_translate=(10,12))
    old=target.data.copy();events=[]
    target.events.data.connect(lambda event:events.append(event))
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:True)
    widget._copy_button.click();qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=10000)
    assert viewer.layers['Curated'] is target and target.data.shape==(32,40)
    old[11:13,14:17]=300
    np.testing.assert_array_equal(target.data,old)
    assert events


def test_copy_button_creates_destination_without_a_dead_end(make_napari_viewer,qtbot):
    viewer=make_napari_viewer();widget,source,_,pixels=setup(viewer,qtbot,target_shape=None)
    widget._manual_copy_target_check.setChecked(False)
    assert 'prepare Curated' in widget._copy_destination_hint.text()
    widget._copy_button.click()
    np.testing.assert_array_equal(viewer.layers['Curated'].data,pixels)
    assert widget._copy_target_combo.currentText()=='Curated'


def test_switching_destination_requests_confirmation_again(make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer();widget,source,target,_=setup(viewer,qtbot,target_shape=(12,14))
    other=viewer.add_labels(np.zeros_like(target.data),name='Other destination')
    widget._refresh_layers_button.click()
    widget._target_combo.setCurrentText(source.name)
    if widget._component_table.rowCount()==0:
        widget._analyze_button.click()
        qtbot.waitUntil(lambda:widget._component_table.rowCount()>0)
        widget._component_table.select_component_id(1)
    calls=[]
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda parent,src,dst,matching:calls.append(dst.name) or True)
    for selected in (target,target,other,other,target):
        widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(selected))
        widget._copy_button.click()
        qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=10000)
    assert calls==['Curated','Other destination','Curated']


def test_coverage_mask_is_not_an_automatic_copy_destination(make_napari_viewer,qtbot):
    viewer=make_napari_viewer();widget,_,_,pixels=setup(viewer,qtbot,target_shape=None)
    coverage=viewer.add_labels(np.zeros_like(pixels),name='Reviewed regions',
        metadata={'label_assistant_review':{'role':'reviewed_regions'}})
    widget._manual_copy_target_check.setChecked(False)
    assert widget._copy_target_combo.findData(coverage)==-1
    widget._copy_button.click()
    assert not coverage.data.any()
    np.testing.assert_array_equal(viewer.layers['Curated'].data,pixels)


@pytest.mark.parametrize('failure',['cancel','read_error'])
def test_background_copy_failure_releases_editing_locks(failure,make_napari_viewer,qtbot):
    from qtpy.QtWidgets import QWidget,QPushButton,QProgressBar
    from napari_label_assistant_tools._large_component_controller import LargeComponentController
    viewer=make_napari_viewer()
    source=viewer.add_labels(np.ones((8,9),np.uint8))
    target=viewer.add_labels(np.zeros((8,9),np.uint8))
    parent=QWidget();qtbot.addWidget(parent)
    done=[]
    controller=LargeComponentController(progress=QProgressBar(parent),cancel_button=QPushButton('Cancel',parent),
        analyze_button=QPushButton(parent),status=lambda _:None,on_ready=lambda *_:None)
    def regions():
        if failure=='read_error':raise OSError('simulated read failure')
        yield (1,)
        yield (2,)
    def write(value):
        target.data[0,0]=7
        controller.cancel()
        return 1
    try:
        controller.copy_regions(regions(),source,target,write,lambda count,message:done.append((count,message)))
        assert not source.editable and not target.editable
        qtbot.waitUntil(lambda:not controller.busy,timeout=10000)
        assert source.editable and target.editable
        assert not source._label_assistant_component_editing and not target._label_assistant_component_editing
        assert done and done[0][1]
        assert done[0][0]==(1 if failure=='cancel' else 0)
    finally:controller.close()


def test_context_menu_uses_manual_targets_and_offers_creation(make_napari_viewer,qtbot,monkeypatch):
    viewer=make_napari_viewer();widget,_,target,_=setup(viewer,qtbot)
    menus=[]
    def dismiss():
        menu=QApplication.activePopupWidget()
        if isinstance(menu,QMenu):
            menus.append(menu);menu.close()
        else:
            QTimer.singleShot(10,dismiss)
    QTimer.singleShot(0,dismiss)
    widget._component_table._context_menu_callback(QPoint(0,0))
    submenu=menus[-1].actions()[0].menu()
    assert any(action.text()==target.name and action.isEnabled() for action in submenu.actions())
    widget._manual_copy_target_check.setChecked(False)
    QTimer.singleShot(0,dismiss)
    widget._component_table._context_menu_callback(QPoint(0,0))
    submenu=menus[-1].actions()[0].menu()
    assert any('Prepare matching Curated' in action.text() and action.isEnabled() for action in submenu.actions())


@pytest.mark.parametrize('accept',[False,True])
def test_confirmation_dialog_names_both_layers_and_supports_cancel(accept,make_napari_viewer,qtbot):
    viewer=make_napari_viewer()
    source=viewer.add_labels(np.zeros((12,14),np.uint8),name='Candidates')
    target=viewer.add_labels(np.zeros((8,9),np.uint8),name='Curated')
    seen=[]
    def answer():
        box=QApplication.activeModalWidget()
        assert isinstance(box,QMessageBox)
        seen.append(box.text())
        box.button(QMessageBox.Yes if accept else QMessageBox.Cancel).click()
    QTimer.singleShot(0,answer)
    assert destination.confirm_manual_destination(None,source,target,True)==accept
    assert 'Candidates' in seen[0] and 'Curated' in seen[0] and '14 × 12' in seen[0] and '9 × 8' in seen[0]


@pytest.mark.parametrize('starting_row',[0,1])
def test_grid_destination_retains_full_extent_when_copying_small_roi(starting_row,tmp_path,make_napari_viewer,qtbot,monkeypatch):
    store=zarr.open_array(str(tmp_path/'full.zarr'),mode='w',shape=(1000,1300),chunks=(256,256),dtype='uint32')
    store[0,0]=77
    path=tmp_path/'project.label-assistant.json'
    path.write_text(json.dumps(dict(format=WORKSPACE_FORMAT,version=2,
        mask_view=dict(mode='grid',tile_size=512,row=starting_row,col=starting_row),
        layers=[dict(layer_type='Labels',name='Curated',storage=dict(kind='zarr',path='full.zarr',array_path=''))])))
    viewer=make_napari_viewer();load_workspace(viewer,path)
    pixels=np.zeros((4,6),np.uint16);pixels[1:3,2:5]=300
    source=viewer.add_labels(pixels,name='Candidates',translate=(520,530))
    target=viewer.layers['Curated']
    widget=component_operations_widget(viewer);qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name);widget._analyze_button.click()
    qtbot.waitUntil(lambda:widget._component_table.rowCount()>0)
    widget._component_table.select_component_id(1)
    widget._manual_copy_target_check.setChecked(True)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(target))
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:True)
    widget._copy_button.click();qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=15000)
    assert full_data(target).shape==(1000,1300) and target.data.shape==(488,512)
    assert 'Copy stopped' not in widget._status_label.text(),widget._status_label.text()
    np.testing.assert_array_equal(full_data(target)[521:523,532:535],np.full((2,3),300))
    np.testing.assert_array_equal(target.data[9:11,20:23],np.full((2,3),300))
    assert full_data(target)[0,0]==77
    save_workspace(viewer,path)
    assert json.loads(path.read_text())['layers'][0]['storage']['shape']==[1000,1300]


def test_matching_a_grid_source_keeps_new_destination_grid_managed(tmp_path,make_napari_viewer,qtbot,monkeypatch):
    store=zarr.open_array(str(tmp_path/'source.zarr'),mode='w',shape=(1000,1300),chunks=(256,256),dtype='uint16')
    store[517:519,514:517]=300
    path=tmp_path/'grid.label-assistant.json'
    path.write_text(json.dumps(dict(format=WORKSPACE_FORMAT,version=2,
        mask_view=dict(mode='grid',tile_size=512,row=1,col=1),
        layers=[dict(layer_type='Labels',name='Candidates',storage=dict(kind='zarr',path='source.zarr',array_path=''))])))
    viewer=make_napari_viewer();load_workspace(viewer,path)
    old=np.zeros((8,8),np.uint16);old[0,0]=77
    original=viewer.add_labels(old,name='Curated')
    source=viewer.layers['Candidates']
    widget=component_operations_widget(viewer);qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name);widget._analyze_button.click()
    qtbot.waitUntil(lambda:widget._component_table.rowCount()>0)
    widget._component_table.select_component_id(1)
    widget._manual_copy_target_check.setChecked(True)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(original))
    monkeypatch.setattr(destination,'confirm_manual_destination',lambda *args:True)
    widget._copy_button.click();qtbot.waitUntil(lambda:not widget._large_controller.busy,timeout=15000)
    matched=viewer.layers['Curated']
    assert hasattr(matched,'_label_assistant_mask_entry') and matched.data.shape==(488,512)
    manager_for(viewer).flush()
    assert full_data(matched).shape==(1000,1300) and full_data(matched)[0,0]==77
    np.testing.assert_array_equal(full_data(matched)[517:519,514:517],np.full((2,3),300))
