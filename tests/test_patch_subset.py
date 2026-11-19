"""Trial subsets preserve approval, disk-backed sources, and validation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_label_assistant_tools._coverage_train import IndexedPatches, index_patches, train_indexed
from napari_label_assistant_tools._patch_subset import select_training_patches, select_trial_validation
from napari_label_assistant_tools._quick_train import QuickTrainConfig, load_model, save_model


class NoPixelReads:
    shape = (100000, 100000)

    def __array__(self, *args):
        raise AssertionError('Subset selection must not materialize arrays')

    def __getitem__(self, key):
        raise AssertionError('Subset selection must not read pixels')


def pool(training=100, validation=20):
    source = NoPixelReads()
    locations = []
    for split, count in (('training',training), ('validation',validation)):
        for i in range(count):
            positive = i % 4 == 0
            row = i if split == 'training' else training+i
            locations.append(dict(y0=row*32,x0=0,y1=row*32+32,x1=32,size=32,
                split=split, block=f'{split}:{i//4}', reviewed_pixels=1024,
                contains_foreground=positive,foreground_pixels_original=128 if positive else 0))
    return IndexedPatches(source,source,source,32,32,locations,[],1024*len(locations),100000**2,
                          validation={'method':'holdout','seed':42})


def coords(dataset,split='training'):
    return [(p['y0'],p['x0']) for p in dataset.locations if p['split']==split]


def test_all_retains_existing_dataset_identity():
    dataset=pool()
    assert select_training_patches(dataset) is dataset


def test_percentage_is_applied_only_after_validation_split_without_pixel_reads():
    dataset=pool(training=1225,validation=337)
    before=deepcopy(dataset.locations)
    selected=select_training_patches(dataset,mode='percentage',percentage=10,seed=7)
    assert len(selected.indices('training')) == 123
    assert coords(selected,'validation') == coords(dataset,'validation')
    assert selected.validation is dataset.validation
    assert selected.blocks is dataset.blocks
    assert selected.image is dataset.image and selected.reviewed is dataset.reviewed
    assert selected.ground_truth is dataset.ground_truth
    assert dataset.locations == before
    assert selected.training_selection['available_training_patches'] == 1225
    assert selected.training_selection['selected_training_patches'] == 123
    assert selected.reviewed_pixels == sum(p['reviewed_pixels'] for p in selected.locations)


def test_subsets_are_reproducible_nested_and_have_positive_and_background_patches():
    dataset=pool()
    small=select_training_patches(dataset,mode='percentage',percentage=10,seed=2)
    same=select_training_patches(dataset,mode='percentage',percentage=10,seed=2)
    bigger=select_training_patches(dataset,mode='percentage',percentage=50,seed=2)
    other=select_training_patches(dataset,mode='percentage',percentage=10,seed=3)
    assert coords(small) == coords(same)
    assert set(coords(small)) <= set(coords(bigger))
    assert coords(small) != coords(other)
    assert coords(small,'validation') == coords(other,'validation')
    flags={p['contains_foreground'] for p in small.locations if p['split']=='training'}
    assert flags == {False,True}
    assert set(coords(small)).isdisjoint(coords(small,'validation'))


def test_count_cap_and_single_patch_must_include_verified_background():
    dataset=pool()
    selected=select_training_patches(dataset,mode='count',count=7)
    assert len(selected.indices('training')) == 7
    assert len(select_training_patches(dataset,mode='count',count=999).indices('training')) == 100
    single=select_training_patches(dataset,mode='count',count=1)
    chosen=next(p for p in single.locations if p['split']=='training')
    assert chosen['foreground_pixels_original'] > 0
    assert chosen['reviewed_pixels'] > chosen['foreground_pixels_original']
    foreground_only=replace(dataset,locations=[{**p,'reviewed_pixels':p['foreground_pixels_original']}
                            if p['split']=='training' and p['contains_foreground'] else p for p in dataset.locations])
    with pytest.raises(ValueError,match='both confirmed foreground and background'):
        select_training_patches(foreground_only,mode='count',count=1)


@pytest.mark.parametrize('settings',[dict(mode='bad'),dict(mode='percentage',percentage=0),
                                     dict(mode='percentage',percentage=101),dict(mode='count',count=0)])
def test_invalid_budgets_are_actionable(settings):
    with pytest.raises(ValueError):
        select_training_patches(pool(),**settings)


def test_multilabel_sampling_checks_each_channel_without_inferring_negatives():
    dataset=pool(training=2,validation=1)
    dataset.multilabel=True
    dataset.class_names=('A','B')
    dataset.locations=[
        {**dataset.locations[0],'contains_foreground':True,'channel_positive_pixels':[10,0],'channel_reviewed_pixels':[20,0]},
        {**dataset.locations[1],'contains_foreground':True,'channel_positive_pixels':[0,10],'channel_reviewed_pixels':[0,20]},
        {**dataset.locations[2],'channel_positive_pixels':[10,10],'channel_reviewed_pixels':[20,20]},
    ]
    with pytest.raises(ValueError,match='Training channel'):
        select_training_patches(dataset,mode='count',count=1)
    selected=select_training_patches(dataset,mode='count',count=2)
    assert selected.multilabel and selected.class_names == dataset.class_names
    assert selected.reviewed is dataset.reviewed
    assert selected.locations[0]['channel_reviewed_pixels'] == [20,0]
    assert selected.locations[1]['channel_reviewed_pixels'] == [0,20]


def consume(generator):
    while True:
        try: next(generator)
        except StopIteration as done: return done.value


def test_real_training_and_checkpoint_history_record_only_selected_training_patches(tmp_path):
    pytest.importorskip('torch')
    yy,xx=np.indices((128,128))
    truth=((xx//8)%2).astype('uint8')
    image=(truth*120+50+yy%8).astype('uint8')
    config=QuickTrainConfig(crop_pixels=32,input_pixels=32,base_channels=4,batch_size=4,device='cpu')
    dataset=consume(index_patches(image,truth,config))
    selected=select_training_patches(dataset,mode='count',count=4,seed=9)
    model=consume(train_indexed(selected,config,epochs=1,extra_targets=0,
                               source_info={'ground_truth_id':'truth'}))
    assert model.training_steps == 1
    assert model.history[-1]['training_selection']['seed'] == 9
    assert model.history[-1]['training_selection']['selected_training_patches'] == 4
    assert len([p for p in model.history[-1]['locations'] if p['split']=='training']) == 4
    assert len(dataset.indices('training')) == 12
    from types import SimpleNamespace
    from napari_label_assistant_tools._reviewed_regions import KEY, record_training_usage
    layer=SimpleNamespace(metadata={KEY:{'id':'truth'}})
    record_training_usage(SimpleNamespace(layers=[layer]),model)
    footprints={tuple(r) for r in layer.metadata[KEY]['training_areas']}
    assert footprints == {(p['y0'],p['x0'],p['y1'],p['x1']) for p in selected.locations if p['split']=='training'}
    assert len(footprints) == 4
    path=tmp_path/'trial.pt'
    save_model(model,path)
    assert load_model(path).history[-1]['training_selection'] == model.history[-1]['training_selection']


def test_quick_trial_matches_reported_budget_without_pixels_or_split_changes():
    dataset = pool(training=5257, validation=1813)
    original = deepcopy(dataset.locations)
    training = select_training_patches(dataset, mode='percentage', percentage=10, seed=42)
    trial = select_trial_validation(training, percentage=10, seed=42)
    assert len(trial.indices('training')) == 526
    assert len(trial.indices('validation')) == 182
    assert coords(trial) == coords(training)
    assert set(coords(trial, 'validation')) <= set(coords(dataset, 'validation'))
    assert set(coords(trial)).isdisjoint(coords(dataset, 'validation'))
    assert trial.blocks is dataset.blocks
    assert trial.image is dataset.image and trial.reviewed is dataset.reviewed
    assert trial.ground_truth is dataset.ground_truth
    assert dataset.locations == original
    assert dataset.validation == {'method': 'holdout', 'seed': 42}
    info = trial.validation['evaluation']
    assert info['available_patches'] == 1813 and info['selected_patches'] == 182
    assert 0 < info['positive_patches'] < 182
    assert info['spatial_groups'] > 1


def test_trial_validation_stays_fixed_when_training_budget_or_seed_changes():
    dataset = pool(training=100, validation=80)
    trials = [select_trial_validation(select_training_patches(dataset, mode='percentage', percentage=p, seed=s),
                                      percentage=25, seed=17) for p, s in ((10, 1), (50, 2), (100, 3))]
    assert all(coords(ds, 'validation') == coords(trials[0], 'validation') for ds in trials)
    assert len({ds.validation['evaluation']['subset_id'] for ds in trials}) == 1
    other = select_trial_validation(dataset, percentage=25, seed=18)
    assert coords(other, 'validation') != coords(trials[0], 'validation')
    assert select_trial_validation(dataset, percentage=100) is dataset


def test_trial_validation_spreads_across_images_and_reserved_groups():
    dataset = pool(training=100, validation=80)
    for i, row in enumerate(dataset.locations):
        row['source_index'] = i % 2
    trial = select_trial_validation(dataset, percentage=25)
    rows = [p for p in trial.locations if p['split'] == 'validation']
    assert {row['source_index'] for row in rows} == {0, 1}
    assert len({row['block'] for row in rows}) > 4
    assert trial.validation['evaluation']['images_selected'] == 2


@pytest.mark.parametrize('percentage', (0, 101, float('nan')))
def test_invalid_trial_validation_percentage(percentage):
    with pytest.raises(ValueError, match='percentage'):
        select_trial_validation(pool(), percentage=percentage)


def test_trial_rejects_missing_supervision_and_kfold():
    dataset = pool()
    with pytest.raises(ValueError, match='k-fold'):
        select_trial_validation(replace(dataset, validation={'method': 'kfold'}))
    background_only = replace(dataset, locations=[{**p, 'contains_foreground':False, 'foreground_pixels_original':0}
                                                  if p['split'] == 'validation' else p for p in dataset.locations])
    with pytest.raises(ValueError, match='foreground and background'):
        select_trial_validation(background_only)
    foreground_only = replace(dataset, locations=[{**p, 'reviewed_pixels':p['foreground_pixels_original']}
                                                  if p['contains_foreground'] else p for p in dataset.locations])
    with pytest.raises(ValueError, match='Increase validation percentage'):
        select_trial_validation(foreground_only, percentage=1)


def test_trial_checks_channel_specific_validity_without_inferring_background():
    dataset = pool(training=2, validation=2)
    dataset.multilabel, dataset.class_names = True, ('A', 'B')
    for i, row in enumerate(dataset.locations):
        row.update(contains_foreground=True,
                   channel_positive_pixels=[10, 0] if i % 2 == 0 else [0, 10],
                   channel_reviewed_pixels=[20, 0] if i % 2 == 0 else [0, 20])
    with pytest.raises(ValueError, match='Validation channel.*Increase validation percentage'):
        select_trial_validation(dataset, percentage=50)
    assert select_trial_validation(dataset, percentage=100) is dataset


def test_trial_does_not_silently_omit_a_multiclass_target():
    dataset = pool(training=4, validation=4)
    dataset.class_names = ('axon', 'myelin')
    for i, row in enumerate(dataset.locations):
        row['contains_foreground'] = True
        row['foreground_pixels_original'] = 128
        row['class_ids'] = [1, 2] if row['split'] == 'training' else [1] if i % 2 else [2]
    with pytest.raises(ValueError, match='missing classes.*Increase validation percentage'):
        select_trial_validation(dataset, percentage=25)
    assert select_trial_validation(dataset, percentage=100) is dataset


def test_full_validation_preserves_weights_and_records_separate_saved_scores(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    from types import SimpleNamespace
    from napari_label_assistant_tools._coverage_train import evaluate_full_validation
    from napari_label_assistant_tools._model_project import remember_model, record_model_evaluation, history_summary
    from napari_label_assistant_tools._training_report import history_points, write_metrics_csv, learning_svg
    from napari_label_assistant_tools._model_library import catalog
    import csv
    yy, xx = np.indices((128, 128))
    truth = ((xx // 8) % 2).astype('uint8')
    image = (truth * 120 + 50 + yy % 8).astype('uint8')
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4, batch_size=4, device='cpu')
    dataset = consume(index_patches(image, truth, config))
    trial = select_trial_validation(select_training_patches(dataset, mode='percentage', percentage=50), percentage=25)
    model = consume(train_indexed(trial, config, epochs=1, extra_targets=0))
    assert model.history[-1]['validation']['evaluation']['selected_patches'] == 1
    weights = {name:value.clone() for name,value in model.network.state_dict().items()}
    original_metrics, steps = dict(model.metrics), model.training_steps
    reads, original_read = [], dataset.read
    def read(index):
        reads.append(index)
        return original_read(index)
    dataset.read = read
    result = consume(evaluate_full_validation(model, dataset, device='cpu'))
    assert reads == dataset.indices('validation')
    assert result['patches'] == 4 and result['scope'] == 'full_reserved_validation'
    assert model.metrics == original_metrics and model.training_steps == steps
    assert all(torch.equal(value, model.network.state_dict()[name]) for name,value in weights.items())
    viewer = SimpleNamespace()
    row = remember_model(viewer, model)
    revision, run_count = row['revision'], len(row['runs'])
    original_path = tmp_path / 'original.pt'
    save_model(model, original_path)
    row.update(checkpoint=str(original_path), saved_revision=revision)
    row['runs'][-1].update(checkpoint=str(original_path), saved_revision=revision)
    record_model_evaluation(viewer, model, result)
    assert row['revision'] == revision and len(row['runs']) == run_count
    assert row['saved_revision'] is None
    assert row['metrics'] == original_metrics
    assert 'locations' not in history_summary(model.history[-1])['evaluations'][0]
    assert 'Trial validation' in catalog(viewer)[0]['validation']
    from napari_label_assistant_tools._model_project import save_project
    manifest = tmp_path / 'workspace.json'
    saved = save_project(viewer, manifest, tmp_path / 'data', False)
    saved_path = manifest.parent / saved['models'][0]['checkpoint']
    assert load_model(saved_path).history[-1]['evaluations'][0]['patches'] == 4
    assert not load_model(original_path).history[-1].get('evaluations')
    path = tmp_path / 'trial.pt'
    save_model(model, path)
    restored = load_model(path)
    assert restored.history[-1]['evaluations'][0]['metrics'] == result['metrics']
    assert restored.metrics == original_metrics
    points = history_points(restored.history)
    assert [point['validation_scope'] for point in points] == ['trial', 'full']
    assert [point['validation_patches'] for point in points] == [1, 4]
    assert 'trial / full' in learning_svg(points)
    metrics_path = tmp_path / 'metrics.csv'
    write_metrics_csv(points, metrics_path)
    with metrics_path.open() as stream:
        exported = list(csv.DictReader(stream))
    assert [p['validation_scope'] for p in exported] == ['trial', 'full']


def test_full_validation_cancellation_and_incompatible_inputs_preserve_model():
    torch = pytest.importorskip('torch')
    from threading import Event
    from napari_label_assistant_tools._coverage_train import evaluate_full_validation
    from napari_label_assistant_tools._quick_train import QuickTrainModel, QuickTrainCancelled, build_unet
    dataset = pool()
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4, device='cpu')
    model = QuickTrainModel(build_unet(4).eval(), config, 32, {}, 'cpu')
    event = Event()
    event.set()
    with pytest.raises(QuickTrainCancelled):
        consume(evaluate_full_validation(model, dataset, device='cpu', cancel=event))
    with pytest.raises(ValueError, match='trained in the reserved'):
        consume(evaluate_full_validation(model, replace(dataset, validation_seen_before=True), device='cpu'))
    with pytest.raises(ValueError, match='matching the loaded model'):
        consume(evaluate_full_validation(model, replace(dataset, input_pixels=64), device='cpu'))
    with pytest.raises(ValueError, match='full prepared validation pool'):
        consume(evaluate_full_validation(model, select_trial_validation(dataset), device='cpu'))
    assert model.history == [] and model.metrics == {} and not model.network.training


def test_large_image_full_validation_reads_only_small_patches_and_cancels():
    pytest.importorskip('torch').set_num_threads(2)
    from threading import Event
    from napari_label_assistant_tools._coverage_train import evaluate_full_validation
    from napari_label_assistant_tools._quick_train import QuickTrainModel, QuickTrainCancelled, build_unet

    class PatchOnlyArray:
        shape = (100000, 100000)
        ndim = 2

        def __init__(self, labels=False):
            self.labels, self.reads = labels, []

        def __array__(self, *args):
            raise AssertionError('Never materialize the full large image/mask')

        def __getitem__(self, region):
            assert isinstance(region, tuple) and len(region) == 2
            height, width = (s.stop - s.start for s in region)
            assert height <= 32 and width <= 32
            self.reads.append(region)
            yy, xx = np.indices((height, width))
            truth = (xx % 16 < 8).astype('uint8')
            return truth if self.labels else (50 + truth * 100 + yy % 8).astype('uint8')

    image, truth = PatchOnlyArray(), PatchOnlyArray(labels=True)
    rows = [dict(split=split, block=split, y0=y, x0=0, y1=y+32, x1=32,
                 size=32, contains_foreground=True, foreground_pixels_original=512, reviewed_pixels=1024)
            for split, y in (('training', 0), ('validation', 99000), ('validation', 99032))]
    dataset = IndexedPatches(image, truth, None, 32, 32, rows, [], 3072, 100000**2,
                             validation={'method':'holdout', 'seed':42})
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4, device='cpu')
    model = QuickTrainModel(build_unet(4).eval(), config, 32, {}, 'cpu')
    assert consume(evaluate_full_validation(model, dataset, device='cpu'))['patches'] == 2
    assert len(image.reads) == len(truth.reads) == 2
    event = Event()
    evaluation = evaluate_full_validation(model, dataset, device='cpu', cancel=event)
    assert next(evaluation).completed == 1
    event.set()
    with pytest.raises(QuickTrainCancelled):
        consume(evaluation)
    assert not model.network.training and model.history == [] and model.training_steps == 0


def test_trial_comparison_preserves_validation_scope_and_csv(tmp_path):
    pytest.importorskip('torch').set_num_threads(2)
    from napari_label_assistant_tools._model_comparison import compare_models, export_comparison
    from napari_label_assistant_tools._model_providers import ModelSpec
    import csv
    yy, xx = np.indices((128, 128))
    truth = ((xx // 8) % 2).astype('uint8')
    image = (truth * 120 + 50 + yy % 8).astype('uint8')
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4, batch_size=4, device='cpu')
    dataset = select_trial_validation(consume(index_patches(image, truth, config)), percentage=25)
    result = consume(compare_models(dataset, config, [ModelSpec(width=w).to_dict() for w in (4, 8)],
                                    epochs=1, extra_targets=0))
    assert all(model.history[-1]['validation']['evaluation']['selected_patches'] == 1 for model in result.models)
    assert 'Trial validation' in result.report['note']
    path = tmp_path / 'comparison.csv'
    export_comparison(result.report, path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['validation_scope'] for row in rows] == ['trial', 'trial']
    assert [row['validation_patches'] for row in rows] == ['1', '1']
