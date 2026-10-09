from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np
import pytest

from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel, load_model, save_model
from napari_label_assistant_tools._model_providers import ModelSpec, build_model, spec_for
from napari_label_assistant_tools._model_project import (
    project, add_task, remember_model, save_project, restore_project, resolve_references, KEY, RUNTIME,
)


def config(spec=None):
    return QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=1,
                            base_channels=4, device='cpu', model_spec=spec)


def model(cfg=None):
    cfg = cfg or config()
    return QuickTrainModel(build_model(cfg), cfg, 32, {'dice': .5, 'iou': .4, 'precision': .6, 'recall': .5}, 'cpu',
                           history=[{'mode': 'new', 'config': {}, 'checkpoints': [], 'steps': 1}])


def exhaust(generator):
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value


@pytest.mark.parametrize('spec', [ModelSpec(width=4), ModelSpec(provider='standard', width=16)])
def test_builtin_logit_contract_and_checkpoint_roundtrip(tmp_path, spec):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    trained = model(config(spec.to_dict()))
    trained.identity = {'task_id': 'myelin', 'model_id': 'model1'}
    trained.network.eval()
    x = torch.rand(1, 1, 32, 48)
    with torch.no_grad():
        expected = trained.network(x)
    assert expected.shape == x.shape
    path = tmp_path / 'weights.pt'
    save_model(trained, path)
    loaded = load_model(path)
    assert spec_for(loaded.config) == spec
    assert loaded.identity == trained.identity
    with torch.no_grad():
        assert torch.equal(expected, loaded.network(x))


def test_legacy_checkpoint_without_model_spec_or_identity(tmp_path):
    torch = pytest.importorskip('torch')
    trained = model()
    path = tmp_path / 'old.pt'
    save_model(trained, path)
    payload = torch.load(path, weights_only=True)
    payload['quick_train_version'] = 1
    payload['config'].pop('model_spec')
    payload.pop('identity')
    torch.save(payload, path)
    loaded = load_model(path)
    assert spec_for(loaded.config) == ModelSpec(width=4)
    assert loaded.identity == {}


def test_pretrained_smp_loading_never_downloads_weights(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    import sys
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return torch.nn.Conv2d(1,1,1)
    monkeypatch.setitem(sys.modules, 'segmentation_models_pytorch',
                        SimpleNamespace(Unet=constructor, UnetPlusPlus=constructor, DeepLabV3Plus=constructor))
    cfg = config(ModelSpec(provider='smp', initialization='imagenet').to_dict())
    trained = model(cfg)
    assert calls[-1]['encoder_weights'] == 'imagenet'
    assert calls[-1]['activation'] is None
    assert calls[-1]['in_channels'] == calls[-1]['classes'] == 1
    path = tmp_path / 'smp.pt'
    save_model(trained,path)
    load_model(path)
    assert calls[-1]['encoder_weights'] is None


def test_run_checkpoints_survive_save_as_and_directory_move(tmp_path):
    pytest.importorskip('torch')
    viewer = SimpleNamespace()
    task = add_task(viewer,'Myelin','Myelin sheath pixels')
    first = model()
    remember_model(viewer,first,task_id=task['id'])
    second = model()
    second.identity = dict(first.identity)
    second.history = deepcopy(first.history) + [{'mode':'continue','config':{},'checkpoints':[],'steps':2}]
    with pytest.importorskip('torch').no_grad():
        for parameter in second.network.parameters():
            parameter.add_(.1)
    remember_model(viewer,second,task_id=task['id'])
    source = tmp_path/'source'
    source.mkdir()
    manifest = source/'workspace.json'
    saved = save_project(viewer,manifest,source/'data',True)
    setattr(viewer,KEY,resolve_references(deepcopy(saved),manifest))
    row = saved['models'][0]
    assert len(row['runs']) == 2
    original_checkpoints = [source/r['checkpoint'] for r in row['runs']]
    assert all(p.is_file() for p in original_checkpoints)
    assert original_checkpoints[0] != original_checkpoints[1]
    destination = tmp_path/'destination'
    destination.mkdir()
    copied = save_project(viewer,destination/'workspace.json',destination/'data',True)
    shutil.rmtree(source)
    moved = tmp_path/'moved'
    destination.rename(moved)
    restored = SimpleNamespace()
    restore_project(restored,copied,moved/'workspace.json')
    assert getattr(restored,RUNTIME) == {}
    checkpoints = [Path(r['checkpoint']) for r in project(restored)['models'][0]['runs']]
    assert all(p.is_file() for p in checkpoints)
    old_weights = load_model(checkpoints[0]).network.state_dict()
    new_weights = load_model(checkpoints[1]).network.state_dict()
    assert any(not pytest.importorskip('torch').equal(old_weights[k],new_weights[k]) for k in old_weights)


def test_comparison_real_training_uses_identical_dataset_and_splits(tmp_path):
    pytest.importorskip('torch').set_num_threads(2)
    from napari_label_assistant_tools._coverage_train import index_patches
    from napari_label_assistant_tools._model_comparison import compare_models, export_comparison
    yy,xx = np.indices((128,128))
    truth=((xx % 16)<8).astype(np.uint8)
    image=(truth*100+yy%10).astype(np.uint8)
    cfg=config()
    dataset=exhaust(index_patches(image,truth,cfg))
    snapshot=deepcopy(dataset.locations)
    specs=[ModelSpec(width=4).to_dict(),ModelSpec(provider='standard',width=16).to_dict()]
    result=exhaust(compare_models(dataset,cfg,specs,epochs=1,extra_targets=0))
    assert len(result.models)==2
    assert dataset.locations==snapshot
    assert result.models[0].history[-1]['validation_blocks']==result.models[1].history[-1]['validation_blocks']
    assert result.models[0].history[-1]['dataset_snapshot_id']==result.models[1].history[-1]['dataset_snapshot_id']
    assert result.report['results'][0]['parameters'] != result.report['results'][1]['parameters']
    assert all(r['seconds']>0 for r in result.report['results'])
    path=tmp_path/'comparison.csv'
    export_comparison(result.report,path)
    import csv
    with path.open() as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==2
    assert rows[0]['dataset_id']==rows[1]['dataset_id']
    assert float(rows[0]['loss_mean'])>=0


def test_workspace_saves_in_memory_model_and_old_workspace_still_opens(make_napari_viewer,tmp_path):
    pytest.importorskip('torch')
    from napari_label_assistant_tools.workspace import save_workspace,load_workspace
    viewer=make_napari_viewer()
    viewer.add_image(np.zeros((32,32),np.uint8),name='EM')
    viewer.add_labels(np.ones((32,32),np.uint8),name='Curated')
    task=add_task(viewer,'Myelin','Sheath pixels')
    remember_model(viewer,model(),task_id=task['id'])
    path=tmp_path/'project.json'
    save_workspace(viewer,path,complete_package=True)
    payload=json.loads(path.read_text())
    assert payload['version']==2
    assert not Path(payload['model_project']['models'][0]['checkpoint']).is_absolute()
    restored=make_napari_viewer()
    load_workspace(restored,path)
    assert len(project(restored)['models'])==1
    assert getattr(restored,RUNTIME)=={}
    payload.pop('model_project')
    payload['version']=1
    path.write_text(json.dumps(payload))
    load_workspace(restored,path)
    assert project(restored)['models']==[]


def test_failed_checkpoint_save_preserves_existing_manifest(make_napari_viewer,tmp_path,monkeypatch):
    from napari_label_assistant_tools.workspace import save_workspace
    import napari_label_assistant_tools._quick_train as engine
    viewer=make_napari_viewer()
    viewer.add_labels(np.ones((32,32),np.uint8))
    path=tmp_path/'workspace.json'
    save_workspace(viewer,path)
    original=path.read_bytes()
    remember_model(viewer,model())
    def fail(*args):
        raise OSError('Disk full')
    monkeypatch.setattr(engine,'save_model',fail)
    with pytest.raises(OSError,match='Disk full'):
        save_workspace(viewer,path)
    assert path.read_bytes()==original


def test_models_ui_task_binding_new_vs_continue_and_restore(make_napari_viewer,qtbot,tmp_path):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    from napari_label_assistant_tools.workspace import save_workspace,load_workspace
    viewer=make_napari_viewer()
    viewer.add_image(np.zeros((64,64),np.uint8),name='EM')
    viewer.add_labels(np.ones((64,64),np.uint8),name='Curated')
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    panel=widget.models_panel
    panel.task_name.setText('Myelin')
    panel.definition.setText('Sheath pixels')
    panel._add_task()
    info=panel.bind_sources()
    widget.model=model()
    panel.provider.setCurrentIndex(panel.provider.findData('standard'))
    assert widget._training_config(False).model_spec['provider']=='standard'
    assert widget._training_config(True).model_spec['provider']=='custom'
    panel._add_candidate()
    panel.provider.setCurrentIndex(panel.provider.findData('custom'))
    panel._add_candidate()
    widget.epochs_spin.setValue(7)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    widget._training_done(widget.model)
    path=tmp_path/'workspace.json'
    save_workspace(viewer,path,complete_package=True)
    assert 'saved with workspace' in widget.model_state_label.text()
    assert 'updated weights not saved' not in panel.details.toPlainText()
    widget.epochs_spin.setValue(2)
    load_workspace(viewer,path)
    assert widget.model is None
    assert widget.epochs_spin.value()==7
    assert panel.task()['name']=='Myelin'
    assert len(panel.candidates)==2
    assert len(project(viewer)['models'])==1
    assert panel.selected()['task_id']==info['task_id']
    panel.definition.setText('Whole fiber pixels')
    with pytest.raises(ValueError,match='Create another task'):
        panel._update_task()


def test_custom_registered_provider_can_reload_without_executing_checkpoint_code(monkeypatch,tmp_path):
    torch=pytest.importorskip('torch')
    import napari_label_assistant_tools._model_providers as providers
    monkeypatch.setattr(providers,'_REGISTERED_PROVIDERS',dict(providers._REGISTERED_PROVIDERS))
    providers.register_model_provider('test_reference',lambda spec,initialize: torch.nn.Conv2d(1,1,1))
    cfg=config(ModelSpec(provider='test_reference',architecture='reference').to_dict())
    path=tmp_path/'custom.pt'
    save_model(model(cfg),path)
    loaded=load_model(path)
    assert spec_for(loaded.config).architecture=='reference'
    providers._REGISTERED_PROVIDERS.pop('test_reference')
    with pytest.raises(ValueError,match='unavailable'):
        load_model(path)


def test_image_change_detection_is_bounded():
    from hashlib import blake2b
    from napari_label_assistant_tools._coverage_train import index_patches
    yy,xx=np.indices((128,128))
    truth=(xx%16<8).astype(np.uint8)
    image=truth.copy()
    dataset=exhaust(index_patches(image,truth,config()))
    row=dataset.locations[0]
    region=(slice(row['y0'],row['y1']),slice(row['x0'],row['x1']))
    row['image_digest']=blake2b(image[region].tobytes(),digest_size=16).hexdigest()
    dataset.read(0)
    image[0,0]+=1
    with pytest.raises(ValueError,match='Image pixels changed'):
        dataset.read(0)


def test_exporting_previous_checkpoint_does_not_replace_latest(make_napari_viewer,qtbot,tmp_path):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer=make_napari_viewer()
    viewer.add_image(np.zeros((64,64),np.uint8))
    viewer.add_labels(np.ones((64,64),np.uint8))
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    first=model()
    widget._training_done(first)
    second=model()
    second.identity=dict(first.identity)
    second.history=deepcopy(first.history)+[{'mode':'continue','checkpoints':[],'config':{},'steps':2}]
    widget._training_done(second)
    row=project(viewer)['models'][0]
    latest=row['revision']
    first_revision=row['runs'][0]['revision']
    panel=widget.models_panel
    panel.run_combo.setCurrentIndex(panel.run_combo.findData(first_revision))
    panel.use_selected()
    assert widget.model is first
    widget._model_saved(tmp_path/'previous-export.pt')
    row=project(viewer)['models'][0]
    assert row['revision']==latest
    assert 'checkpoint' not in row
    assert row['runs'][0]['checkpoint'].endswith('previous-export.pt')


def test_comparison_report_is_external_and_portable(tmp_path):
    from napari_label_assistant_tools._model_comparison import compare_models
    from napari_label_assistant_tools._coverage_train import index_patches
    yy,xx=np.indices((128,128))
    truth=(xx%16<8).astype(np.uint8)
    image=truth.copy()
    dataset=exhaust(index_patches(image,truth,config()))
    result=exhaust(compare_models(dataset,config(),[ModelSpec(width=4).to_dict(),ModelSpec(width=8).to_dict()],epochs=1,extra_targets=0))
    viewer=SimpleNamespace()
    project(viewer)['comparisons'].append(result.report)
    folder=tmp_path/'original'
    folder.mkdir()
    saved=save_project(viewer,folder/'workspace.json',folder/'data',True)
    assert 'locations' not in saved['comparisons'][0]['dataset']
    assert not Path(saved['comparisons'][0]['report_file']).is_absolute()
    destination=tmp_path/'moved'
    folder.rename(destination)
    restored=SimpleNamespace()
    restore_project(restored,saved,destination/'workspace.json')
    assert project(restored)['comparisons'][0]['dataset']['locations']


def test_malformed_model_project_does_not_clear_existing_layers(make_napari_viewer,tmp_path):
    from napari_label_assistant_tools.workspace import save_workspace, load_workspace
    viewer=make_napari_viewer()
    original=viewer.add_labels(np.ones((32,32),np.uint8),name="Keep this layer")
    path=tmp_path/"workspace.json"
    save_workspace(viewer,path)
    payload=json.loads(path.read_text())
    payload["model_project"]["models"]={"broken":"record"}
    path.write_text(json.dumps(payload))
    with pytest.raises(Exception,match="Existing workspace retained"):
        load_workspace(viewer,path)
    assert original in viewer.layers


def test_unavailable_custom_provider_preserves_workspace_annotation_access(make_napari_viewer,qtbot,tmp_path):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    from napari_label_assistant_tools.workspace import save_workspace,load_workspace
    viewer=make_napari_viewer()
    viewer.add_labels(np.ones((32,32),np.uint8),name="Annotations")
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    path=tmp_path/"workspace.json"
    save_workspace(viewer,path)
    payload=json.loads(path.read_text())
    payload["model_project"]["settings"]["new_model"]["provider"]="unavailable_custom_provider"
    path.write_text(json.dumps(payload))
    load_workspace(viewer,path)
    assert len(viewer.layers)==1
    assert widget.models_panel.provider.currentData()=="custom"


def test_smp_configuration_reuse_handles_hidden_width(make_napari_viewer,qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer=make_napari_viewer()
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    cfg=config(ModelSpec(provider='smp',encoder='resnet34').to_dict())
    trained=QuickTrainModel(None,cfg,32,{'dice':.5,'iou':.4,'precision':.6,'recall':.5},'cpu')
    widget._training_done(trained)
    panel=widget.models_panel
    panel.use_configuration()
    assert panel.spec().provider=='smp'
    assert panel.spec().encoder=='resnet34'
    assert panel.spec().width==8
    panel.provider.setCurrentIndex(panel.provider.findData('custom'))
    assert panel.spec().encoder=='resnet18'


def test_restore_complete_sampled_workflow_before_model_selection(make_napari_viewer,qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer=make_napari_viewer()
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    assert widget.annotation_combo.currentData()=='partial'
    settings={'combos':{'workflow_combo':'sampled','annotation_combo':'complete'}}
    widget.restore_project_settings(settings)
    assert widget.annotation_combo.currentData()=='complete'
    assert widget.workflow_combo.currentData()=='sampled'


def test_model_library_marks_updated_weights_unsaved(make_napari_viewer,qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer=make_napari_viewer()
    widget=QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget._training_done(model())
    assert 'updated weights not saved' in widget.models_panel.details.toPlainText()


@pytest.mark.parametrize('error,expected',[
    (ModuleNotFoundError("No module named SMP",name='segmentation_models_pytorch'),'SMP is not installed'),
    (ModuleNotFoundError("No module named timm",name='timm'),'SMP dependency import failed'),
    (RuntimeError('operator torchvision::nms does not exist'),'SMP is present but could not be imported'),
])
def test_smp_import_diagnostics_preserve_environment_and_cause(monkeypatch,error,expected):
    import importlib
    import sys
    from napari_label_assistant_tools._model_providers import require_smp
    def fail(name):
        raise error
    monkeypatch.setattr(importlib,'import_module',fail)
    with pytest.raises(ValueError) as caught:
        require_smp()
    assert expected in str(caught.value)
    assert sys.executable in str(caught.value)
    if error.name == 'segmentation_models_pytorch' if isinstance(error,ModuleNotFoundError) else False:
        assert '-m pip install segmentation-models-pytorch' in str(caught.value)
    else:
        assert str(error) in str(caught.value)
