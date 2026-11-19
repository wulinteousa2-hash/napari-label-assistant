from copy import deepcopy

import numpy as np
import pytest

from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
from napari_label_assistant_tools._reviewed_regions import associate
from napari_label_assistant_tools.workspace import load_workspace, save_workspace


@pytest.fixture
def prepared_widget(make_napari_viewer,qtbot):
    viewer=make_napari_viewer()
    yy,xx=np.indices((128,128))
    truth=((xx//8)%2).astype('uint8')
    image=viewer.add_image((truth*120+50+yy%8).astype('uint8'),name='Image')
    labels=viewer.add_labels(truth,name='Curated')
    reviewed=viewer.add_labels(np.ones_like(truth),name='Reviewed regions')
    associate(reviewed,image,labels)
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    for combo in (widget.crop_combo,widget.resolution_combo):
        combo.addItem('32 px (test)',32)
        combo.setCurrentIndex(combo.findData(32))
    widget.image_combo.setCurrentIndex(widget.image_combo.findData(image))
    widget.truth_combo.setCurrentIndex(widget.truth_combo.findData(labels))
    widget.prepare_button.click()
    qtbot.waitUntil(lambda:widget.worker is None,timeout=15000)
    assert widget.prepared is not None,widget.status_label.text()
    return viewer,widget,reviewed


def test_change_budget_reuses_index_and_preserves_approval(prepared_widget):
    _,widget,reviewed=prepared_widget
    base=widget.prepared
    signature=widget._prepared_signature
    metadata=deepcopy(reviewed.metadata)
    pixels=reviewed.data.copy()
    validation=[p for p in base.locations if p['split']=='validation']
    widget.patch_selection_mode.setCurrentIndex(widget.patch_selection_mode.findData('percentage'))
    widget.patch_percentage_spin.setValue(50)
    selected=widget._selected_training_dataset()
    assert widget.prepared is base and widget._prepared_signature == signature
    assert len(selected.indices('training')) == 6
    assert [p for p in selected.locations if p['split']=='validation'] == validation
    assert 'Using 6 / 12 training patches' in widget.patch_selection_summary.text()
    assert 'selected patches' in widget.workload_label.text()
    assert widget.patch_count_spin.isHidden() and not widget.patch_percentage_spin.isHidden()
    assert reviewed.metadata == metadata
    np.testing.assert_array_equal(reviewed.data,pixels)
    widget.patch_selection_mode.setCurrentIndex(widget.patch_selection_mode.findData('all'))
    assert widget._selected_training_dataset() is base


def test_training_receives_selected_dataset_not_full_pool(prepared_widget,monkeypatch):
    from napari_label_assistant_tools._coverage_train import train_indexed
    _,widget,_=prepared_widget
    widget.patch_selection_mode.setCurrentIndex(widget.patch_selection_mode.findData('count'))
    widget.patch_count_spin.setValue(3)
    calls=[]
    monkeypatch.setattr(widget.models_panel,'bind_sources',lambda:{'task_id':'trial'})
    monkeypatch.setattr(widget,'_lock_sources',lambda:None)
    monkeypatch.setattr(widget,'_start_worker',lambda function,done,*args,**kwargs:calls.append((function,args,kwargs)))
    widget._start_training()
    function,args,_=calls[0]
    assert function is train_indexed
    assert len(args[0].indices('training')) == 3
    assert len(args[0].indices('validation')) == len(widget.prepared.indices('validation'))
    assert len(widget.prepared.indices('training')) == 12


def test_quick_trial_changes_both_budgets_without_reindexing_or_approval_changes(prepared_widget):
    _, widget, reviewed = prepared_widget
    base, signature = widget.prepared, widget._prepared_signature
    before = deepcopy(reviewed.metadata)
    widget.quick_trial_check.setChecked(True)
    selected = widget._selected_training_dataset()
    assert widget.patch_selection_mode.currentData() == 'percentage'
    assert len(selected.indices('training')) == 2
    assert len(selected.indices('validation')) == 1
    assert 'Trial validation: 1 / 4' in widget.patch_selection_summary.text()
    assert '1 trial validation patches' in widget.workload_label.text()
    validation = [p for p in selected.locations if p['split'] == 'validation']
    widget.patch_percentage_spin.setValue(50)
    widget.patch_subset_seed.setValue(9)
    assert [p for p in widget._selected_training_dataset().locations if p['split'] == 'validation'] == validation
    assert widget.prepared is base and widget._prepared_signature == signature
    assert reviewed.metadata == before
    widget.quick_trial_check.setChecked(False)
    assert len(widget._selected_training_dataset().indices('validation')) == 4


def test_quick_trial_settings_restore_and_old_workspaces_default_off(prepared_widget):
    _, widget, _ = prepared_widget
    widget.quick_trial_check.setChecked(True)
    widget.trial_validation_percentage.setValue(50)
    settings = widget.project_settings()
    widget.quick_trial_check.setChecked(False)
    widget.restore_project_settings(settings)
    assert widget.quick_trial_check.isChecked()
    assert widget.trial_validation_percentage.value() == 50
    assert len(widget._selected_training_dataset().indices('validation')) == 2
    widget.restore_project_settings({})
    assert not widget.quick_trial_check.isChecked()


def test_trial_training_and_full_evaluation_receive_different_validation_pools(prepared_widget, monkeypatch):
    from napari_label_assistant_tools._coverage_train import evaluate_full_validation, train_indexed
    from napari_label_assistant_tools._quick_train import QuickTrainModel
    _, widget, _ = prepared_widget
    widget.quick_trial_check.setChecked(True)
    calls = []
    monkeypatch.setattr(widget.models_panel, 'bind_sources', lambda: {'task_id':'trial'})
    monkeypatch.setattr(widget, '_lock_sources', lambda: None)
    monkeypatch.setattr(widget, '_start_worker', lambda function, done, *args, **kwargs: calls.append((function, args, kwargs)))
    widget._start_training()
    function, args, _ = calls[-1]
    assert function is train_indexed
    assert len(args[0].indices('training')) == 2 and len(args[0].indices('validation')) == 1
    widget.model = QuickTrainModel(None, widget._config(), 32, {}, 'cpu')
    widget._start_full_validation()
    function, args, _ = calls[-1]
    assert function is evaluate_full_validation
    assert args[1] is widget.prepared
    assert len(args[1].indices('validation')) == 4


def test_full_evaluation_gui_saves_without_new_training_round(prepared_widget, qtbot, tmp_path):
    pytest.importorskip('torch').set_num_threads(2)
    from napari_label_assistant_tools._coverage_train import train_indexed
    from napari_label_assistant_tools._model_project import project
    from napari_label_assistant_tools._quick_train import load_model
    viewer, widget, _ = prepared_widget
    widget.device_combo.setCurrentIndex(widget.device_combo.findData('cpu'))
    widget.quick_trial_check.setChecked(True)
    dataset, config = widget._selected_training_dataset(), widget._config()
    generator = train_indexed(dataset, config, epochs=1, extra_targets=0)
    while True:
        try:
            next(generator)
        except StopIteration as done:
            model = done.value
            break
    widget._training_done(model)
    widget._update_controls()
    assert widget.full_validation_button.isEnabled()
    assert 'Trial validation' in widget.validation_results_label.text()
    state = project(viewer)
    run_count, steps = len(state['models'][0]['runs']), model.training_steps
    widget.full_validation_button.click()
    qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)
    assert 'Full validation' in widget.validation_results_label.text(), widget.status_label.text()
    assert model.history[-1]['evaluations'][-1]['patches'] == 4
    assert model.training_steps == steps
    assert len(project(viewer)['models'][0]['runs']) == run_count
    manifest = tmp_path / 'trial.label-assistant.json'
    save_workspace(viewer, manifest, complete_package=True, copy_labels=True)
    saved = project(viewer)['models'][0]
    restored_model = load_model(saved['checkpoint'])
    assert restored_model.history[-1]['evaluations'][-1]['patches'] == 4
    assert 'locations' not in saved['history'][-1]['evaluations'][-1]
    widget.quick_trial_check.setChecked(False)
    load_workspace(viewer, manifest)
    assert widget.quick_trial_check.isChecked()


def test_subset_settings_roundtrip_in_existing_workspace_format(prepared_widget,make_napari_viewer,qtbot,tmp_path):
    viewer,widget,_=prepared_widget
    widget.patch_selection_mode.setCurrentIndex(widget.patch_selection_mode.findData('count'))
    widget.patch_count_spin.setValue(7)
    widget.patch_subset_seed.setValue(99)
    path=tmp_path/'selection.label-assistant.json'
    save_workspace(viewer,path,complete_package=True)
    restored=make_napari_viewer()
    assert not load_workspace(restored,path)['skipped_layers']
    other=QuickTrainWidget(restored)
    qtbot.addWidget(other)
    assert other.patch_selection_mode.currentData() == 'count'
    assert other.patch_count_spin.value() == 7 and other.patch_subset_seed.value() == 99
    assert other.patch_percentage_spin.isHidden() and not other.patch_count_spin.isHidden()


def test_kfold_keeps_full_fold_workflow(prepared_widget):
    _,widget,_=prepared_widget
    widget.patch_selection_mode.setCurrentIndex(widget.patch_selection_mode.findData('percentage'))
    widget.patch_percentage_spin.setValue(10)
    widget.validation_method_combo.setCurrentIndex(widget.validation_method_combo.findData('kfold'))
    assert not widget.patch_selection_mode.isEnabled()
    assert 'full index' in widget.patch_selection_summary.text()
    assert widget._selected_training_dataset() is widget.prepared
