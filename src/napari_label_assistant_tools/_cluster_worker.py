"""Process entry point for one rank; no napari or Qt imports."""
import json
import os
from pathlib import Path
import signal
import sys
import threading


def stop(folder):
    """Stop only the worker belonging to this private, randomly named job."""
    import shutil
    import time
    folder = Path(folder)
    (folder / 'cancel').touch()
    pid_file = folder / 'worker.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text())
        # Check identity as well as PID, since a completed process's PID may be reused.
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists() and str(folder).encode() in command.read_bytes().split(b'\0'):
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):
                if not command.exists():
                    break
                time.sleep(0.1)
            if command.exists() and str(folder).encode() in command.read_bytes().split(b'\0'):
                os.kill(pid, signal.SIGKILL)
    shutil.rmtree(folder)


def main(folder, rank):
    from dataclasses import asdict, replace
    from datetime import timedelta
    import subprocess
    import torch
    from ._cluster_training import read_dataset
    from ._coverage_train import train_indexed
    from ._quick_train import QuickTrainConfig, load_model, save_model
    folder = Path(folder)
    (folder / 'worker.pid').write_text(str(os.getpid()))
    stopped = threading.Event()
    class Cancellation:
        def is_set(self):
            return stopped.is_set() or (folder / 'cancel').exists()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    def watch_connection():
        sys.stdin.read()
        stopped.set()
    threading.Thread(target=watch_connection, daemon=True).start()
    job = json.loads((folder / 'job.json').read_text())
    # Select the interface that routes to the coordinator; never fall back to Wi-Fi.
    interface = job['interface']
    if rank or not interface:
        route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', job['address']], text=True))[0]
        if rank:
            interface = route['dev']
        elif not interface:
            addresses = json.loads(subprocess.check_output(['ip', '-j', 'address'], text=True))
            interface = next(row['ifname'] for row in addresses if any(
                a.get('local') == job['address'] for a in row['addr_info']))
    os.environ['NCCL_SOCKET_IFNAME'] = '=' + interface
    os.environ['GLOO_SOCKET_IFNAME'] = interface
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl', init_method=f"tcp://{job['address']}:{job['port']}",
                                         rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        config = QuickTrainConfig(**{**job['config'], 'class_names': tuple(job['config']['class_names'])})
        config = replace(config, device='cuda')
        initial = load_model(folder / 'initial.pt') if job['initial_model'] else None
        generator = train_indexed(read_dataset(folder), config, initial_model=initial,
                                  cancel=Cancellation(), distributed=True, **job['settings'])
        while True:
            try:
                progress = next(generator)
            except StopIteration as done:
                result = done.value
                break
            if rank == 0:
                print('NLA_PROGRESS ' + json.dumps(asdict(progress)), flush=True)
        if rank == 0:
            result.history[-1]['compute'].update(nodes=job['nodes'], backend='nccl', mode='two_nodes')
            save_model(result, folder / 'result.pt')
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    if sys.argv[1] == '--stop':
        stop(sys.argv[2])
    else:
        main(sys.argv[1], int(sys.argv[2]))
