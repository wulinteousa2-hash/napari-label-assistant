"""Opt-in two-host training. No discovery, services, or remote dependencies at startup."""
from dataclasses import asdict, dataclass
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
from uuid import uuid4

import numpy as np

from ._coverage_train import IndexedPatches
from ._quick_train import QuickTrainProgress, _check_cancel, load_model, save_model


@dataclass(frozen=True)
class ClusterSettings:
    host: str
    python: str
    address: str
    interface: str = ''

    def validate(self):
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:-]*', self.host):
            raise ValueError('Enter an SSH host alias or user@hostname for the second node.')
        if not self.python.startswith('/') or any(c in self.python for c in '\n\r\0'):
            raise ValueError('Enter the absolute Python executable path on the second node.')
        try:
            address = ipaddress.ip_address(self.address)
        except ValueError as error:
            raise ValueError('Enter this computer’s IP address on the inter-node network.') from error
        if address.is_loopback or address.is_unspecified or address.version != 4:
            raise ValueError('Use this computer’s reachable IPv4 address on the inter-node network.')
        if self.interface and not re.fullmatch(r'[A-Za-z0-9_.:-]+', self.interface):
            raise ValueError('Enter a valid network interface name.')


def ssh_command(settings, arguments):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=5',
            '-o', 'ServerAliveCountMax=3', settings.host, shlex.join(arguments)]


def run_checked(command, cancel, *, timeout=60):
    """Cancellable transfers and checks; capture stderr without pipe deadlocks."""
    _check_cancel(cancel)
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            import time
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                _check_cancel(cancel)
                if time.monotonic() > deadline:
                    raise ValueError('Compute-node command timed out.')
                if cancel is not None:
                    cancel.wait(0.1)
                else:
                    time.sleep(0.1)
            output.seek(0)
            result = output.read().decode(errors='replace')
            if process.returncode:
                raise ValueError(f'Compute-node command failed: {result[-3000:]}')
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def check_cluster(settings, *, cancel=None):
    settings.validate()
    import torch
    if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        raise ValueError('Two-node training requires a CUDA GPU and NCCL on this computer.')
    try:
        torch.cuda.mem_get_info()
    except Exception as error:
        raise ValueError('This computer cannot initialize CUDA. Free GPU/unified memory before retrying. ' + str(error)) from error
    # Bind now so an incorrect local address fails before staging any data.
    with socket.socket() as connection:
        connection.bind((settings.address, 0))
    probe = ('import json,socket,torch,numpy,scipy\n'
             'report=dict(host=socket.gethostname(),torch=torch.__version__, '
             'cuda=torch.cuda.is_available(),nccl=torch.distributed.is_nccl_available())\n'
             'try:\n report["free_bytes"],report["total_bytes"]=torch.cuda.mem_get_info()\n'
             'except Exception as error:\n report["error"]=str(error)\n'
             'print(json.dumps(report))')
    report = json.loads(run_checked(ssh_command(settings, [settings.python, '-c', probe]), cancel))
    if not report['cuda'] or not report['nccl']:
        raise ValueError('The second node needs a CUDA-enabled PyTorch environment with NCCL.')
    if report.get('error'):
        raise ValueError(f"Worker {report['host']} cannot initialize CUDA. Free GPU/unified memory before retrying. {report['error']}")
    if report['torch'] != torch.__version__:
        raise ValueError(f"Use matching PyTorch versions: local {torch.__version__}, worker {report['torch']}.")
    return report


def write_dataset(dataset, folder, cancel=None):
    """Freeze just the selected model inputs, one patch at a time, to disk."""
    folder = Path(folder)
    metadata = {name: getattr(dataset, name) for name in (
        'crop_pixels', 'input_pixels', 'locations', 'blocks', 'reviewed_pixels', 'total_pixels',
        'validation_seen_before', 'validation', 'class_names', 'source_info', 'multilabel', 'training_selection')}
    metadata['image_shape'] = list(dataset.image.shape)
    if not dataset.locations:
        raise ValueError('The prepared dataset has no patches.')
    arrays = None
    try:
        for index in range(len(dataset.locations)):
            _check_cancel(cancel)
            patch = dataset.read(index)
            if arrays is None:
                arrays = [np.lib.format.open_memmap(folder / f'{name}.npy', mode='w+',
                          dtype=np.float32, shape=(len(dataset.locations), *value.shape))
                          for name, value in zip(('images', 'targets', 'validity'), patch)]
            for array, value in zip(arrays, patch):
                array[index] = value
            yield QuickTrainProgress(index + 1, len(dataset.locations),
                                    f'Preparing two-node inputs: {index + 1}/{len(dataset.locations)} patches.')
        for array in arrays:
            array.flush()
        (folder / 'dataset.json').write_text(json.dumps(metadata), encoding='utf-8')
    finally:
        if arrays:
            for array in arrays:
                array._mmap.close()


class FrozenPatches(IndexedPatches):
    def read(self, index, **kwargs):
        return tuple(array[index].copy() for array in self._arrays)


def read_dataset(folder):
    from types import SimpleNamespace
    folder = Path(folder)
    metadata = json.loads((folder / 'dataset.json').read_text())
    image = SimpleNamespace(shape=tuple(metadata.pop('image_shape')))
    metadata['class_names'] = tuple(metadata['class_names'])
    dataset = FrozenPatches(image=image, ground_truth=None, reviewed=None, **metadata)
    dataset._arrays = [np.load(folder / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                       for name in ('images', 'targets', 'validity')]
    return dataset


def stage_code(folder):
    """Ship this exact code revision without requiring a GUI install on the worker."""
    package = Path(folder) / 'napari_label_assistant_tools'
    package.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        if source.name != '__init__.py':
            shutil.copy2(source, package / source.name)
    (package / '__init__.py').write_text('')


def train_two_nodes(dataset, config, *, cluster, cancel=None, initial_model=None, **settings):
    """Launch rank 0 locally and rank 1 through an explicitly configured SSH host."""
    from ._model_providers import spec_for
    if not isinstance(dataset, IndexedPatches) or dataset.validation.get('method') == 'kfold':
        raise ValueError('Two-node training requires a prepared coverage dataset with holdout validation.')
    if spec_for(config).provider != 'custom':
        raise ValueError('Two-node training currently supports Custom small U-Net models.')
    if config.device == 'cpu' or config.batch_size < 2:
        raise ValueError('Two-node training needs Automatic/CUDA compute and a global batch size of at least 2.')
    yield QuickTrainProgress(0, 1, 'Checking both training nodes…')
    report = check_cluster(cluster, cancel=cancel)
    remote = '/tmp/nla-cluster-' + uuid4().hex
    processes = []
    staged = False
    with tempfile.TemporaryDirectory(prefix='nla-cluster-') as temporary:
        folder = Path(temporary)
        try:
            yield from write_dataset(dataset, folder, cancel)
            if initial_model is not None:
                save_model(initial_model, folder / 'initial.pt')
            stage_code(folder)
            with socket.socket() as connection:
                connection.bind((cluster.address, 0))
                port = connection.getsockname()[1]
            job = dict(config=asdict(config), settings=settings, address=cluster.address, port=port,
                       nodes=[socket.gethostname(), report['host']], interface=cluster.interface,
                       initial_model=initial_model is not None)
            (folder / 'job.json').write_text(json.dumps(job), encoding='utf-8')
            archive = folder / 'inputs.tar'
            with tarfile.open(archive, 'w') as bundle:
                for item in folder.iterdir():
                    if item != archive:
                        bundle.add(item, arcname=item.name)
            yield QuickTrainProgress(0, 1, 'Sending the prepared inputs and checkpoint to the second node…')
            run_checked(ssh_command(cluster, ['mkdir', '-m', '700', remote]), cancel)
            staged = True
            run_checked(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                         '-o', 'ConnectTimeout=8', str(archive), f'{cluster.host}:{remote}/inputs.tar'],
                        cancel, timeout=1800)
            run_checked(ssh_command(cluster, ['tar', '-xf', remote + '/inputs.tar', '-C', remote]), cancel)
            messages = queue.Queue()
            def collect(process, rank):
                for line in process.stdout:
                    messages.put((rank, line.rstrip()))
            for rank in (0, 1):
                command = [sys.executable if rank == 0 else cluster.python, '-u', '-m',
                           'napari_label_assistant_tools._cluster_worker',
                           str(folder) if rank == 0 else remote, str(rank)]
                if rank:
                    command = ssh_command(cluster, ['env', f'PYTHONPATH={remote}', *command])
                environment = {**os.environ, 'PYTHONPATH': str(folder)}
                process = subprocess.Popen(command, cwd=folder, env=environment, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                processes.append(process)
                threading.Thread(target=collect, args=(process, rank), daemon=True).start()
            errors = []
            yield QuickTrainProgress(0, 1, 'Both nodes launched. Connecting GPUs and synchronizing the model…')
            while any(process.poll() is None for process in processes) or not messages.empty():
                _check_cancel(cancel)
                try:
                    rank, line = messages.get(timeout=0.1)
                except queue.Empty:
                    if any(process.poll() not in (None, 0) for process in processes):
                        raise ValueError('Two-node training failed: ' + '\n'.join(errors[-12:]))
                    continue
                if line.startswith('NLA_PROGRESS '):
                    if rank == 0:
                        update = json.loads(line[len('NLA_PROGRESS '):])
                        update['message'] = 'Two nodes · ' + update['message']
                        yield QuickTrainProgress(**update)
                else:
                    errors.append(f'Node {rank + 1}: {line}')
                    errors = errors[-40:]
            if any(process.returncode for process in processes):
                raise ValueError('Two-node training failed: ' + '\n'.join(errors[-12:]))
            _check_cancel(cancel)
            return load_model(folder / 'result.pt')
        finally:
            # The sentinel is checked between patches and steps on both nodes.
            (folder / 'cancel').touch()
            if staged:
                try:
                    cleanup = ('import pathlib,runpy,shutil,sys; p=pathlib.Path(sys.argv[1]); '
                               'w=p/"napari_label_assistant_tools"/"_cluster_worker.py"; '
                               'runpy.run_path(str(w))["stop"](p) if w.is_file() else shutil.rmtree(p,ignore_errors=True)')
                    run_checked(ssh_command(cluster, [cluster.python, '-c', cleanup, remote]), None, timeout=15)
                except (ValueError, OSError):
                    pass  # stdin EOF also cancels the remote worker after a lost SSH connection.
            for process in processes:
                if process.stdin:
                    process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stdout:
                    process.stdout.close()
