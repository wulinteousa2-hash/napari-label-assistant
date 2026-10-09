from threading import Event

import numpy as np
import pytest

from napari_label_assistant_tools._coverage_train import (
    epoch_indices, index_patches, masked_binary_loss, preview_index, train_indexed,
)
from napari_label_assistant_tools._quick_train import QuickTrainCancelled, QuickTrainConfig, load_model, save_model


class SliceOnly:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('Full-array conversion is forbidden')

    def __getitem__(self, region):
        self.reads.append(region)
        result = self.data[region]
        assert max(result.shape) <= 32
        return result


def consume(generator):
    updates = []
    while True:
        try:
            updates.append(next(generator))
        except StopIteration as result:
            return result.value, updates


def fixture_data(shape=(128, 128)):
    yy, xx = np.indices(shape)
    truth = ((xx // 8) % 2).astype(np.uint8)
    image = (truth * 120 + 50 + yy % 8).astype(np.uint8)
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, samples=8, steps=2,
                              base_channels=4, device='cpu', batch_size=4)
    return image, truth, config


def test_index_covers_image_edges_and_reads_only_bounded_patches():
    image, truth, config = fixture_data((137, 141))
    lazy_image, lazy_truth = SliceOnly(image), SliceOnly(truth)
    dataset, _ = consume(index_patches(lazy_image, lazy_truth, config))
    assert lazy_image.reads == []  # indexing reads only annotation data
    coverage = np.zeros(image.shape, dtype=np.uint8)
    for item in dataset.locations:
        coverage[item['y0']:item['y1'], item['x0']:item['x1']] += 1
    assert np.all(coverage == 1)
    assert dataset.reviewed_pixels == image.size
    x, y, valid = dataset.read(len(dataset.locations) - 1)
    assert x.shape == y.shape == valid.shape == (32, 32)
    assert valid.sum() == (137 % 32) * (141 % 32)
    assert not y[valid == 0].any()
    previews = preview_index(dataset, count=8)
    assert len(previews.locations) <= 8
    assert len(dataset.locations) == 25


def test_partial_review_excludes_unknown_foreground_and_masks_loss():
    torch = pytest.importorskip('torch')
    image, truth, config = fixture_data()
    reviewed = np.zeros(truth.shape, dtype=np.uint8)
    reviewed[:, ::2] = 1
    dataset, _ = consume(index_patches(image, truth, config, reviewed=reviewed))
    assert dataset.reviewed_pixels == image.size // 2
    _, y, valid = dataset.read(0)
    assert not valid[:, 1::2].any()
    assert not y[:, 1::2].any()
    logits = torch.zeros((1, 1, 32, 32), requires_grad=True)
    loss = masked_binary_loss(torch, logits, torch.from_numpy(y)[None, None], torch.from_numpy(valid)[None, None])
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, :, 1::2]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, ::2]) > 0


def test_each_epoch_visits_every_training_patch_and_never_validation():
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(image, truth, config))
    training, validation = set(dataset.indices('training')), set(dataset.indices('validation'))
    rng = np.random.default_rng(4)
    order = epoch_indices(dataset, rng, extra_targets=0.5)
    assert set(order) == training
    assert not set(order) & validation
    assert len(order) == len(training) + int(np.ceil(len(training) * 0.5))
    train_blocks = {p['block'] for p in dataset.locations if p['split'] == 'training'}
    validation_blocks = {p['block'] for p in dataset.locations if p['split'] == 'validation'}
    assert not train_blocks & validation_blocks


def test_reserved_validation_blocks_survive_reindexing_and_seed_change():
    from dataclasses import replace
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(image, truth, config))
    reserved = [b['id'] for b in dataset.blocks if b['split'] == 'validation']
    again, _ = consume(index_patches(image, truth, replace(config, seed=99), validation_blocks=reserved))
    assert [(p['y0'], p['x0'], p['split']) for p in again.locations] == [(p['y0'], p['x0'], p['split']) for p in dataset.locations]


def test_empty_review_and_changed_annotations_are_actionable():
    image, truth, config = fixture_data()
    with pytest.raises(ValueError, match='No reviewed pixels'):
        consume(index_patches(image, truth, config, reviewed=np.zeros_like(truth)))
    dataset, _ = consume(index_patches(image, truth, config))
    truth[0, 0] = 1
    with pytest.raises(ValueError, match='Annotations changed'):
        dataset.read(0)


def test_indexing_cancellation():
    image, truth, config = fixture_data()
    cancel = Event()
    generator = index_patches(image, truth, config, cancel=cancel)
    next(generator)
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        next(generator)


def test_actual_streamed_training_continuation_and_model_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(SliceOnly(image), SliceOnly(truth), config))
    model, updates = consume(train_indexed(dataset, config, epochs=1, extra_targets=0))
    assert model.training_steps == 3
    assert model.history[-1]['checkpoints'][-1]['unique_patches'] == 12
    assert any('coverage 100%' in p.message for p in updates)
    original = {k: v.clone() for k, v in model.network.state_dict().items()}
    cancel = Event()
    generator = train_indexed(dataset, config, epochs=1, extra_targets=0, initial_model=model, cancel=cancel)
    next(generator)
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        next(generator)
    assert all(torch.equal(v, model.network.state_dict()[k]) for k, v in original.items())
    continued, _ = consume(train_indexed(dataset, config, epochs=1, extra_targets=0, initial_model=model))
    assert continued.training_steps == 6
    assert continued.training_rounds == 2
    assert all(torch.equal(v, model.network.state_dict()[k]) for k, v in original.items())
    path = tmp_path / 'coverage.pt'
    save_model(continued, path)
    restored = load_model(path)
    assert restored.history[-1]['workflow'] == 'coverage'
    assert restored.history[-1]['validation_blocks']
    assert restored.training_steps == 6


def test_downsampling_does_not_silently_discard_reviewed_targets():
    from dataclasses import replace
    image, truth, config = fixture_data()
    reviewed = np.zeros_like(truth)
    reviewed[:, ::2] = 1
    with pytest.raises(ValueError, match='disappear at model input size'):
        consume(index_patches(image, truth, replace(config, crop_pixels=64), reviewed=reviewed))


def test_continuation_retains_starting_weights_if_validation_declines(monkeypatch):
    torch = pytest.importorskip('torch')
    from napari_label_assistant_tools import _coverage_train as module
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(image, truth, config))
    initial, _ = consume(train_indexed(dataset, config, epochs=1, extra_targets=0))
    original = {k: v.clone() for k, v in initial.network.state_dict().items()}
    evaluations = iter([0.9, 0.1])
    def fake_validate(*args, step, total):
        score = next(evaluations)
        yield module.QuickTrainProgress(step, total, 'Validating')
        return {'dice': score, 'iou': score, 'precision': score, 'recall': score}
    monkeypatch.setattr(module, '_validate', fake_validate)
    model, _ = consume(train_indexed(dataset, config, epochs=1, extra_targets=0, initial_model=initial))
    assert model.history[-1]['retained_epoch'] == 0
    assert model.metrics['dice'] == 0.9
    assert all(torch.equal(v, model.network.state_dict()[k]) for k, v in original.items())


def test_validation_excludes_regions_used_by_prior_sampled_training():
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(image, truth, config))
    reserved = next(b['id'] for b in dataset.blocks if b['split'] == 'validation')
    row, col = map(int, reserved.split(':'))
    previous = [{'y0': row * 64, 'x0': col * 64, 'size': 64}]
    revised, _ = consume(index_patches(image, truth, config, prior_training_regions=previous))
    assert all(p['split'] == 'training' for p in revised.locations if p['block'] == reserved)
    assert any(p['split'] == 'validation' for p in revised.locations)


def test_seen_holdout_allows_new_model_but_rejects_continuation():
    image, truth, config = fixture_data()
    previous = [{'y0': 0, 'x0': 0, 'size': 128}]
    dataset, _ = consume(index_patches(image, truth, config, prior_training_regions=previous))
    assert dataset.validation_seen_before
    with pytest.raises(ValueError, match='already trained'):
        consume(train_indexed(dataset, config, initial_model=object()))
    pytest.importorskip('torch')
    model, _ = consume(train_indexed(dataset, config, epochs=1, extra_targets=0))
    assert model.history[-1]['mode'] == 'new'


def test_plateau_control_patience_and_improvement_reset():
    from napari_label_assistant_tools._coverage_train import PlateauControl
    control = PlateauControl(2, 2, 0.5, 0.01)
    assert control.update(1.0) is None
    assert control.update(0.995) is None  # Below minimum meaningful improvement.
    assert control.update(1.0) == "reduce"
    assert control.update(1.0) is None
    assert control.update(0.8) is None  # Improvement resets post-reduction patience.
    assert control.update(0.8) is None
    assert control.update(0.8) == "stop"
    with pytest.raises(ValueError, match="finite"):
        control.update(float("nan"))


def test_training_reduces_lr_then_stops_and_saves_history(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    from napari_label_assistant_tools import _coverage_train as module
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(image, truth, config))
    def plateau_validation(*args, step, total):
        yield module.QuickTrainProgress(step, total, 'Validating')
        return {'loss': 1.0, 'dice': 0.5, 'iou': 0.4, 'precision': 0.5, 'recall': 0.5}
    monkeypatch.setattr(module, '_validate', plateau_validation)
    model, updates = consume(train_indexed(dataset, config, epochs=20, extra_targets=0,
        plateau_enabled=True, reduce_patience=2, stop_patience=2, learning_rate=0.001))
    history = model.history[-1]
    assert history['max_epochs'] == 20
    assert history['epochs'] == 5
    assert model.training_steps == 15
    assert history['learning_rate_events'] == [{'epoch': 3, 'old_lr': 0.001, 'new_lr': 0.0005}]
    assert [c['learning_rate'] for c in history['checkpoints']] == [0.001]*3 + [0.0005]*2
    assert history['stop_reason']
    assert any('Early stop at epoch 5' in p.message for p in updates)
    assert updates[-1].completed == updates[-1].total
    path = tmp_path / 'early-stop.pt'
    save_model(model, path)
    restored = load_model(path)
    assert restored.history[-1]['learning_rate_events'] == history['learning_rate_events']
    assert restored.history[-1]['epochs'] == 5


def test_configurable_holdout_and_spatial_folds():
    from napari_label_assistant_tools._coverage_train import partition_dataset
    image, truth, config = fixture_data((256, 256))
    quarter, _ = consume(index_patches(image, truth, config, validation_fraction=.25))
    half, _ = consume(index_patches(image, truth, config, validation_fraction=.5))
    assert len(half.indices('validation')) == 2*len(quarter.indices('validation'))
    dataset, _ = consume(index_patches(image, truth, config, validation_method='kfold', folds=5, split_seed=7))
    same, _ = consume(index_patches(image, truth, config, validation_method='kfold', folds=5, split_seed=7))
    assert dataset.validation == same.validation
    folds = dataset.validation['folds']
    seen = set()
    for blocks in folds:
        assert not seen.intersection(blocks)
        seen.update(blocks)
        current = partition_dataset(dataset, blocks)
        assert {current.locations[i]['block'] for i in current.indices('training')}.isdisjoint({current.locations[i]['block'] for i in current.indices('validation')})
    assert seen == {p['block'] for p in dataset.locations}
    with pytest.raises(ValueError, match='independent spatial groups'):
        consume(index_patches(*fixture_data((64,64)), validation_method='kfold', folds=5))


def test_real_kfold_final_training_and_saved_chart(tmp_path):
    from napari_label_assistant_tools._coverage_train import train_kfold, train_final
    image, truth, config = fixture_data()
    dataset, _ = consume(index_patches(image, truth, config, validation_method='kfold', folds=2))
    result, updates = consume(train_kfold(dataset, config, epochs=1, extra_targets=0))
    assert len(result.models) == 2
    assert all(model.training_rounds == 1 and model.history[-1]['mode'] == 'new' for model in result.models)
    assert {u.fold for u in updates if u.metrics} == {1,2}
    assert all(u.epoch == 1 for u in updates if u.metrics)
    assert result.report['aggregate']['dice']['mean'] == pytest.approx(np.mean([m.metrics['dice'] for m in result.models]))
    assert result.report['aggregate']['dice']['std'] == pytest.approx(np.std([m.metrics['dice'] for m in result.models], ddof=1))
    final, _ = consume(train_final(dataset, config, cross_validation_report=result.report, epochs=1, extra_targets=0, plateau_enabled=True))
    assert not final.metrics
    assert all(p['split'] == 'training' for p in final.locations)
    assert final.history[-1]['mode'] == 'final_all_reviewed'
    assert not final.history[-1]['plateau_control']['enabled']
    assert final.history[-1]['checkpoints'][0]['unique_patches'] == len(dataset.locations)
    assert final.history[-1]['cross_validation_report'] == result.report
    path = tmp_path / 'final.pt'
    save_model(final, path)
    loaded = load_model(path)
    assert loaded.history[-1]['cross_validation_report'] == result.report
    assert path.with_suffix('.learning.svg').exists()
    assert 'Dice' in path.with_suffix('.learning.svg').read_text()
    assert result.models[0].history[-1]['validation']['fold'] == 1
