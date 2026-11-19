"""Batch history colors stay independent of training and annotation pixels."""
from copy import deepcopy

import numpy as np
import pytest
from qtpy.QtGui import QColor

from napari_label_assistant_tools import _annotation_batches as batches
from napari_label_assistant_tools._annotation_batch_widget import AnnotationBatchWidget
from napari_label_assistant_tools._batch_overlays import display_regions
from napari_label_assistant_tools._reviewed_regions import associate


@pytest.fixture
def colored(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((256,256),'uint8'),name='EM')
    truth = viewer.add_labels(np.zeros((256,256),'uint32'),name='Curated')
    mask = viewer.add_labels(np.ones((256,256),'uint8'),name='Reviewed')
    associate(mask,image,truth)
    mask.metadata['label_assistant_review'].update(areas=[[0,0,256,256]],review_cells=[[0,0,128,256]])
    first = batches.ensure_pair(viewer,image,truth)['active_id']
    second = batches.change(viewer,image,truth,'new')['id']
    widget = AnnotationBatchWidget(viewer)
    qtbot.addWidget(widget)
    widget.set_context(image,truth)
    widget.show()
    viewer.layers.selection.active = truth
    return viewer,image,truth,mask,widget,first,second


def test_colors_stay_stable_and_viewing_does_not_change_training(colored):
    viewer,image,truth,_,widget,first,second = colored
    before = batches.selection(image,truth)
    signature = batches.signature(image,truth)
    pair = batches.get_pair(image,truth)
    colors = [batches.batch_color(b) for b in pair['batches']]
    assert len(set(colors)) == 2
    widget.buttons[first].click()
    assert batches.active_batch(batches.get_pair(image,truth))['id'] == first
    assert batches.selection(image,truth) == before
    for mode in ('all','hide','active'):
        widget.display.setCurrentIndex(widget.display.findData(mode))
        assert batches.signature(image,truth) == signature
    batches.change(viewer,image,truth,'rename',name='First round')
    assert [batches.batch_color(b) for b in batches.get_pair(image,truth)['batches']] == colors
    widget.scope.setCurrentIndex(widget.scope.findData(first))
    assert batches.selection(image,truth)['id'] == first
    assert batches.signature(image,truth) != signature
    assert batches.training_batch(batches.get_pair(image,truth))['id'] != second


def test_approved_regions_automatically_keep_color_and_layer_selection(colored,qtbot):
    viewer,image,truth,mask,widget,_,second = colored
    pixels, review_pixels = truth.data.copy(),mask.data.copy()
    active = viewer.layers.selection.active
    selected = set(viewer.layers.selection)
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    qtbot.waitUntil(lambda:widget.overlay.layer is not None and widget.overlay.layer.visible,timeout=3000)
    layer = widget.overlay.layer
    assert layer.features['batch_id'].tolist() == [second]
    color = QColor(batches.batch_color(batches.active_batch(batches.get_pair(image,truth))))
    np.testing.assert_allclose(layer.edge_color[0],color.getRgbF(),atol=.005)
    assert .05 <= layer.face_color[0,3] <= .12
    assert viewer.layers.selection.active is active
    assert set(viewer.layers.selection) == selected
    np.testing.assert_array_equal(truth.data,pixels)
    np.testing.assert_array_equal(mask.data,review_pixels)
    assert widget.buttons[second].property('batchColor') == color.name()


def test_all_batches_clip_overlaps_and_dim_history(colored,qtbot):
    viewer,image,truth,mask,widget,first,second = colored
    batches.change(viewer,image,truth,'approve',regions=[[64,64,192,192]])
    widget.display.setCurrentIndex(widget.display.findData('all'))
    items = display_regions(batches.get_pair(image,truth),mask.metadata['label_assistant_review'])
    assert items[0]['batch_id'] == second
    for item in items[1:]:
        a,b,c,d = item['bounds']
        assert not (a < 192 and c > 64 and b < 192 and d > 64)
        assert item['batch_id'] == first
    qtbot.waitUntil(lambda:widget.overlay.layer is not None and len(widget.overlay.layer.data) == len(items),timeout=3000)
    layer = widget.overlay.layer
    assert set(layer.features['batch_id']) == {first,second}
    assert all(layer.edge_color[i,3] < 1 for i,b in enumerate(layer.features['batch_id']) if b == first)
    assert layer.edge_color[0,3] == 1


def test_edited_regions_gain_red_cross_without_losing_batch_identity(colored,qtbot):
    from napari_label_assistant_tools._component_library import invalidate_reviewed_regions
    viewer,image,truth,mask,widget,_,second = colored
    region = [128,0,256,256]
    mask.metadata['label_assistant_review']['review_cells'].append(region)
    batches.change(viewer,image,truth,'approve',regions=[region])
    widget.overlay.render()
    assert len(widget.overlay.layer.data) == 1
    invalidate_reviewed_regions(viewer,truth,(160,32,170,40))
    qtbot.waitUntil(lambda:len(widget.overlay.layer.data) == 3,timeout=3000)
    layer = widget.overlay.layer
    assert layer.features['batch_id'].tolist() == [second]*3
    assert set(layer.features['review_status']) == {'Needs review'}
    np.testing.assert_allclose(layer.edge_color[1],[1,.25,.25,1],atol=.005)
    assert 'Red ×' in widget.summary.text()
    batches.change(viewer,image,truth,'revoke',regions=[region])
    widget.overlay.render()
    assert len(layer.data) == 3


def test_hide_survives_approval_and_view_controls_do_not_invalidate_training(colored,qtbot):
    viewer,image,truth,_,widget,_,_ = colored
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    widget.overlay.render()
    layer = widget.overlay.layer
    signature = batches.signature(image,truth)
    widget.display.setCurrentIndex(widget.display.findData('hide'))
    assert not layer.visible
    assert batches.signature(image,truth) == signature
    batches.change(viewer,image,truth,'approve',regions=[[128,0,160,32]])
    widget.overlay.render()
    assert not layer.visible
    widget.display.setCurrentIndex(widget.display.findData('all'))
    assert layer.visible


def test_old_batches_gain_colors_without_changing_previous_training_selection(colored):
    viewer,image,truth,_,widget,first,second = colored
    pair = batches.get_pair(image,truth)
    pair.pop('training_id')
    pair.pop('view_mode')
    for record in pair['batches']:
        record.pop('color')
    colors = [batches.batch_color(record) for record in pair['batches']]
    batches.change(viewer,image,truth,'select',batch_id=first)
    assert batches.selection(image,truth)['id'] == second
    assert [record['color'] for record in batches.get_pair(image,truth)['batches']] == colors


def test_batch_display_survives_workspace_reopen(colored,tmp_path,make_napari_viewer):
    from napari_label_assistant_tools.workspace import save_workspace,load_workspace
    viewer,image,truth,_,widget,first,second = colored
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    widget.buttons[first].click()
    widget.display.setCurrentIndex(widget.display.findData('all'))
    expected = deepcopy(batches.get_pair(image,truth))
    path = tmp_path/'colors.label-assistant.json'
    save_workspace(viewer,path,complete_package=True)
    restored = make_napari_viewer()
    assert not load_workspace(restored,path)['skipped_layers']
    pair = batches.get_pair(restored.layers['EM'],restored.layers['Curated'])
    assert pair == expected
    assert pair['active_id'] == first and pair['training_id'] == second
    assert pair['view_mode'] == 'all'
    assert not any(layer.name == 'Annotation batches' for layer in restored.layers)


def test_pending_yellow_selection_stays_above_batch_history(colored):
    from napari_label_assistant_tools._dynamic_edit import INTERNAL_LAYER_ROLE_KEY
    viewer,image,truth,_,widget,_,_ = colored
    yellow = viewer.add_shapes([np.array([[128,0],[128,32],[160,32],[160,0]])],shape_type='rectangle',
                               name='Pending review tiles',edge_color='yellow',face_color='transparent',
                               metadata={INTERNAL_LAYER_ROLE_KEY:'review_cells'})
    viewer.layers.selection.active = truth
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    widget.overlay.render()
    assert viewer.layers.index(yellow) > viewer.layers.index(widget.overlay.layer)
    assert viewer.layers.selection.active is truth
    viewer.camera.zoom *= 2
    assert len(widget.overlay.layer.edge_width) == len(widget.overlay.layer.data)


def test_switching_sources_hides_previous_overlay_and_restores_it(colored,qtbot):
    viewer,image,truth,_,widget,_,_ = colored
    batches.change(viewer,image,truth,'approve',regions=[[128,0,256,256]])
    widget.overlay.render()
    layer = widget.overlay.layer
    other_image = viewer.add_image(np.zeros((32,32),'uint8'),name='Other image')
    other_truth = viewer.add_labels(np.zeros((32,32),'uint8'),name='Other mask')
    widget.set_context(other_image,other_truth)
    qtbot.waitUntil(lambda:not layer.visible,timeout=3000)
    widget.set_context(image,truth)
    qtbot.waitUntil(lambda:layer.visible,timeout=3000)
    assert widget.overlay.layer is layer


def test_large_grid_history_uses_local_coordinate_comparisons(monkeypatch):
    from napari_label_assistant_tools import _component_library
    regions = [[y,x,y+1000,x+1000] for y in range(0,100000,1000) for x in range(0,100000,1000)]
    batch = dict(id='batch',number=1,name='Batch 1',regions=regions)
    pair = dict(active_id='batch',view_mode='all',shape=[100000,100000],batches=[batch])
    calls = []
    original = _component_library.subtract_rectangle
    def counted(area,cut):
        calls.append(1)
        return original(area,cut)
    monkeypatch.setattr(_component_library,'subtract_rectangle',counted)
    items = display_regions(pair,dict(areas=regions))
    assert len(items) == 10000 and all(item['ready'] for item in items)
    assert len(calls) < len(regions)*4
