from dataclasses import asdict
from threading import Event
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from napari_label_assistant_tools._cluster_training import (
    ClusterSettings, read_dataset, run_checked, ssh_command, train_two_nodes, write_dataset,
)
from napari_label_assistant_tools._coverage_train import index_patches, train_indexed
from napari_label_assistant_tools._quick_train import QuickTrainCancelled, QuickTrainConfig, segmentation_loss


def consume(generator):
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value


def dataset_fixture():
    yy, xx = np.indices((64, 96))
    truth = ((xx // 8) % 2).astype(np.uint8)
    config = QuickTrainConfig(crop_pixels=32, input_pixels=32, base_channels=4,
                              batch_size=4, device='cpu', augment_rotations=False,
                              augment_flips=False, augment_intensity=False)
    dataset = consume(index_patches(truth * 120 + 50 + yy % 8, truth, config))
    return dataset, config


def test_snapshot_preserves_inputs_provenance_and_independence(tmp_path):
    from napari_label_assistant_tools._model_comparison import dataset_snapshot
    dataset, _ = dataset_fixture()
    dataset.training_selection = {'mode': 'all'}
    consume(write_dataset(dataset, tmp_path))
    frozen = read_dataset(tmp_path)
    assert dataset_snapshot(frozen) == dataset_snapshot(dataset)
    for index in range(len(dataset.locations)):
        for expected, actual in zip(dataset.read(index), frozen.read(index)):
            np.testing.assert_array_equal(expected, actual)
    dataset.ground_truth[:] = 0
    assert frozen.read(0)[1].any()  # staged inputs cannot change with later edits


def test_cancellation_stops_snapshot_and_subprocess(tmp_path):
    import os
    from threading import Timer
    dataset, _ = dataset_fixture()
    cancel = Event()
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        consume(write_dataset(dataset, tmp_path, cancel))
    cancel.clear()
    pid_path = tmp_path / 'child.pid'
    program = 'import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)'
    timer = Timer(1, cancel.set)
    timer.start()
    try:
        with pytest.raises(QuickTrainCancelled):
            run_checked([sys.executable, '-c', program, str(pid_path)], cancel)
    finally:
        timer.cancel()
    assert pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


@pytest.mark.parametrize('worker_error', ['out of memory', None])
def test_preflight_checks_cuda_allocation_before_submission(monkeypatch, worker_error):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from napari_label_assistant_tools import _cluster_training as module
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (100, 200))
    monkeypatch.setattr(torch.distributed, 'is_nccl_available', lambda: True)
    monkeypatch.setattr(module, 'socket', SimpleNamespace(socket=lambda: nullcontext(SimpleNamespace(bind=lambda _: None))))
    report = dict(host='worker', torch=torch.__version__, cuda=True, nccl=True)
    if worker_error:
        report['error'] = worker_error
    monkeypatch.setattr(module, 'run_checked', lambda *args, **kwargs: json.dumps(report))
    settings = ClusterSettings('worker', '/env/python', '192.168.100.10')
    if worker_error:
        with pytest.raises(ValueError, match='Free GPU/unified memory'):
            module.check_cluster(settings)
    else:
        assert module.check_cluster(settings) == report


@pytest.mark.parametrize('host', ['-oProxyCommand=x', 'worker;touch /tmp/unwanted', 'worker\ncommand'])
def test_invalid_hosts_rejected(host):
    with pytest.raises(ValueError):
        ClusterSettings(host, '/env/bin/python', '192.168.100.10').validate()


def test_remote_command_quotes_arguments():
    import shlex
    settings = ClusterSettings('worker', '/env with spaces/bin/python', '192.168.100.10')
    arguments = [settings.python, '-c', 'print("literal $HOME; `text`")']
    assert shlex.split(ssh_command(settings, arguments)[-1]) == arguments


def _distributed_worker(rank, rendezvous, folder):
    import torch
    torch.set_num_threads(2)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        # Uneven masks and a zero-validity rank exercise global loss normalization.
        for mode, channels in [('binary', 1), ('classes', 3), ('multilabel', 2)]:
            generator = torch.Generator().manual_seed(3)
            logits = torch.randn(3, channels, 4, 4, generator=generator)
            if mode == 'classes':
                target = torch.randint(0, channels, (3, 1, 4, 4), generator=generator).float()
            else:
                target = torch.randint(0, 2, logits.shape, generator=generator).float()
            valid = torch.ones_like(target)
            valid[0, ..., :2, :] = 0
            local = logits[rank::2].clone().requires_grad_()
            loss = segmentation_loss(torch, local, target[rank::2], valid[rank::2],
                                     multilabel=mode == 'multilabel', distributed=True)
            loss.backward()
            np.save(Path(folder) / f'{mode}-{rank}.npy', local.grad.numpy() / 2)
            if rank == 0:
                whole = logits.clone().requires_grad_()
                expected = segmentation_loss(torch, whole, target, valid, multilabel=mode == 'multilabel')
                expected.backward()
                np.save(Path(folder) / f'{mode}-expected.npy', whole.grad.numpy())
                assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
        dataset = read_dataset(folder)
        _, config = dataset_fixture()
        result = consume(train_indexed(dataset, config, distributed=True, epochs=1, extra_targets=0))
        torch.save(result.network.state_dict(), Path(folder) / f'rank-{rank}.pt')
        (Path(folder) / f'metrics-{rank}.json').write_text(json.dumps(result.metrics))
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_training_matches_local_including_uneven_last_batch(tmp_path):
    import torch
    torch.set_num_threads(2)
    dataset, config = dataset_fixture()
    assert len(dataset.indices('training')) % config.batch_size == 1
    consume(write_dataset(dataset, tmp_path))
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2)
    for mode in ('binary', 'classes', 'multilabel'):
        expected = np.load(tmp_path / f'{mode}-expected.npy')
        for rank in range(2):
            np.testing.assert_allclose(np.load(tmp_path / f'{mode}-{rank}.npy'), expected[rank::2], atol=1e-7, rtol=1e-5)
    local = consume(train_indexed(dataset, config, epochs=1, extra_targets=0))
    # Per-channel GroupNorm removes these convolution biases mathematically.
    # Near-zero floating-point gradients can make Adam update them differently
    # across batch partitions; verify the identifiable weights and outputs.
    invariant_biases = set()
    for name, module in local.network.named_modules():
        if isinstance(module, torch.nn.Sequential):
            children = list(module.children())
            for index, child in enumerate(children):
                if (index and isinstance(child, torch.nn.GroupNorm)
                        and child.num_groups == child.num_channels
                        and isinstance(children[index - 1], torch.nn.Conv2d)):
                    invariant_biases.add(f'{name}.{index - 1}.bias')
    rank_zero = torch.load(tmp_path / 'rank-0.pt', weights_only=True)
    for rank in range(2):
        state = torch.load(tmp_path / f'rank-{rank}.pt', weights_only=True)
        for name, tensor in local.network.state_dict().items():
            torch.testing.assert_close(state[name], rank_zero[name], atol=0, rtol=0)
            if name not in invariant_biases:
                torch.testing.assert_close(state[name], tensor, atol=3e-5, rtol=3e-4)
        metrics = json.loads((tmp_path / f'metrics-{rank}.json').read_text())
        assert metrics == pytest.approx(local.metrics, abs=1e-5)
    from napari_label_assistant_tools._model_providers import build_model
    parallel = build_model(config)
    parallel.load_state_dict(rank_zero)
    parallel.eval()
    with torch.no_grad():
        for index in range(len(dataset.locations)):
            inputs = torch.from_numpy(dataset.read(index)[0])[None, None]
            torch.testing.assert_close(parallel(inputs), local.network(inputs), atol=2e-5, rtol=1e-4)


def test_compute_defaults_local_and_shared_settings_do_not_enable_remote(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    widget = QuickTrainWidget(make_napari_viewer())
    qtbot.addWidget(widget)
    assert not widget.training_compute.two_nodes()
    assert widget.training_compute.options.isHidden()
    settings = widget.project_settings()
    assert 'cluster' not in settings
    widget.restore_project_settings(settings)
    assert not widget.training_compute.two_nodes()


def test_two_node_button_dispatches_loaded_checkpoint(make_napari_viewer, qtbot, monkeypatch):
    from types import SimpleNamespace
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    viewer = make_napari_viewer()
    dataset, config = dataset_fixture()
    viewer.add_image(dataset.image)
    viewer.add_labels(dataset.ground_truth)
    widget = QuickTrainWidget(viewer)
    qtbot.addWidget(widget)
    widget.annotation_combo.setCurrentIndex(widget.annotation_combo.findData('complete'))
    widget.prepared = dataset
    widget._prepared_signature = widget._signature()
    widget.model = SimpleNamespace(config=config, identity={})
    widget.training_compute.mode.setCurrentIndex(1)
    cluster = ClusterSettings('worker', '/env/python', '192.168.100.10')
    monkeypatch.setattr(widget.training_compute, 'settings', lambda: cluster)
    monkeypatch.setattr(widget, '_training_config', lambda _: QuickTrainConfig(**{**asdict(config), 'device': 'cuda'}))
    captured = []
    monkeypatch.setattr(widget, '_start_worker', lambda *args, **kwargs: captured.append((args, kwargs)))
    widget._start_training(continue_training=True)
    args, kwargs = captured[-1]
    assert args[0] is train_two_nodes
    assert kwargs['cluster'] is cluster
    assert kwargs['initial_model'] is widget.model
    assert widget._locked_layer is not None
    widget._unlock_sources()
