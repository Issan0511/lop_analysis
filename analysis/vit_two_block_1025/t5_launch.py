"""Sequential registered T5 cases; wait five minutes for unavailable checkpoints."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
import common as T
from t5_run import source_hashes


def completed(path):
    if not (path / 'complete.json').exists():
        return False
    result = json.loads((path / 'complete.json').read_text())
    assert result['status'] == 'complete'
    assert result['config']['source_hashes'] == source_hashes()
    for item in json.loads((path / 'manifest.json').read_text()):
        file = path / item['path']
        assert file.stat().st_size == item['bytes'] and T.sha256(file) == item['sha256'], file
    return True


def main():
    out = T.ARCHIVE / 't5'
    out.mkdir(exist_ok=True)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = json.loads((T.ARCHIVE / 'gate/t5.json').read_text())
    assert gate['status'] == 'passed' and gate['source_hashes'] == source_hashes()
    attempt = str(time.time_ns())
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    for mode in ('fixed', 'adam'):
        pending = [(seed, task, step, layer) for task in (10, 60) for seed in T.SEEDS
                   for step in (100, 500) for layer in (1, 2)]
        while pending:
            ready = None
            for case in list(pending):
                seed, task, step, layer = case
                name = f'seed{seed}_t{task:02d}_s{step:03d}_L{layer}_{mode}'
                if completed(out / name):
                    pending.remove(case)
                    continue
                ckpt = T.ARCHIVE / f'runs/seed{seed}/ckpt/t{task:02d}_s{step:03d}.pt'
                if ckpt.exists():
                    ready = case, name
                    break
            if ready is None:
                if not pending:
                    break
                row = dict(status='waiting_for_checkpoint', mode=mode, pending=pending, updated=time.time())
                T.atomic_json(out / 'controller_status.json', row)
                print(json.dumps(row), flush=True)
                time.sleep(300)
                continue
            case, name = ready
            seed, task, step, layer = case
            log = logs / f'{name}_{attempt}.log'
            row = dict(status='running', case=name, mode=mode, pending=len(pending), log=str(log), updated=time.time())
            T.atomic_json(out / 'controller_status.json', row)
            print(json.dumps(row), flush=True)
            try:
                with log.open('x') as output:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('t5_run.py')),
                        '--seed', str(seed), '--task', str(task), '--step', str(step), '--layer', str(layer),
                        '--mode', mode], cwd=T.ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
                assert completed(out / name)
            except Exception as exc:
                T.atomic_json(out / 'controller_status.json', dict(row, status='failed', error=str(exc)))
                raise
            pending.remove(case)
    T.atomic_json(out / 'controller_status.json', dict(status='complete', updated=time.time()))
    print('All T5 cases completed', flush=True)


if __name__ == '__main__':
    main()
