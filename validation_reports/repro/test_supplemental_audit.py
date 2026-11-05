"""Additional provenance, recovery and incompatible-checkpoint probes."""
import json
from pathlib import Path

import numpy as np
import pytest

OUT=Path(__file__).resolve().parents[1]/'evidence'/'independent_observations'


def observe(name,**values):
    (OUT/f'{name}.json').write_text(json.dumps(values,indent=2,default=str))


def test_A19_copy_collision_has_recovery(make_napari_viewer,qtbot):
    from napari_label_assistant_tools._widget import component_operations_widget
    viewer=make_napari_viewer()
    data=np.zeros((16,20),np.uint16); data[2:5,4:8]=7
    source=viewer.add_labels(data,name='candidate')
    destination=np.zeros_like(data); destination[2:5,4:8]=9
    target=viewer.add_labels(destination,name='curated')
    widget=component_operations_widget(viewer);qtbot.addWidget(widget)
    widget._target_combo.setCurrentText(source.name)
    widget._analyze_button.click()
    qtbot.waitUntil(lambda:widget._component_table.rowCount()>0)
    widget._component_table.selectRow(0)
    widget._copy_target_combo.setCurrentIndex(widget._copy_target_combo.findData(target))
    widget._copy_button.click()
    observe('A19_copy_collision',before_value=9,after_values=np.unique(target.data).tolist(),
            deletion_undo_available=widget._undo_delete_button.isEnabled(),
            status=widget._status_label.text(),scope='programmatic Qt button activation; no human usability study')
    assert np.all(target.data[2:5,4:8]==9) or widget._undo_delete_button.isEnabled(), 'Copy overwrote curated values without collision confirmation or plugin recovery'


def test_A20_single_pair_predictions_require_review_confirmation(make_napari_viewer,qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    yy,xx=np.indices((128,160)); target=((xx//4+yy//8)%2).astype(np.uint8)
    viewer=make_napari_viewer(); viewer.add_image((target*130+yy%7).astype(np.uint8))
    viewer.add_labels(target,name='Model Training prediction — Image',metadata={'quick_train':{'experimental':True}})
    widget=QuickTrainWidget(viewer);qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    for combo in (widget.crop_combo,widget.resolution_combo):
        combo.addItem('32 px',32);combo.setCurrentIndex(combo.findData(32))
    widget.prepare_button.click();qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    observe('A20_prediction_as_ground_truth',accepted=widget.prepared is not None,
            status=widget.status_label.text(),explicit_review_confirmation_requested=False)
    assert widget.prepared is None,'Current-pair path accepted an experimental prediction without the explicit confirmation required by Manage pairs'


@pytest.mark.parametrize('kind',['corrupt','unsupported_version','incompatible_contract'])
def test_A21_bad_checkpoints_are_rejected(kind,tmp_path):
    import torch
    from napari_label_assistant_tools._quick_train import QuickTrainConfig,load_model
    path=tmp_path/f'{kind}.pt'
    if kind=='corrupt': path.write_bytes(b'not a pytorch checkpoint')
    elif kind=='unsupported_version':torch.save({'quick_train_version':999},path)
    else:torch.save({'quick_train_version':2,'config':QuickTrainConfig().__dict__,
                    'contract':{'output_channels':3}},path)
    with pytest.raises(Exception) as error:
        load_model(path)
    observe(f'A21_{kind}',rejected=True,error_type=type(error.value).__name__,message=str(error.value),
            scope='loader rejection; native error text may still require user guidance')
