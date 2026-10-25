"""Sequential diagnostics on all 20 registered checkpoints as they become ready."""
import argparse
import fcntl
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
import common as T


def main(kind):
    module_name = {'alignment': 'alignment', 'global_noise': 'global_noise'}[kind]
    module = importlib.import_module(module_name)
    current_hashes = module.hashes()
    gate = json.loads((T.ARCHIVE / f'gate/{kind}.json').read_text())
    assert gate['status'] == 'passed' and gate['source_hashes'] == current_hashes
    out = T.ARCHIVE / kind
    out.mkdir(exist_ok=True)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    attempt = str(time.time_ns())
    pending = [(seed, task, step) for task in (10, 60) for seed in T.SEEDS for step in (100, 500)]
    while pending:
        ready = None
        for case in list(pending):
            seed, task, step = case
            name = f'seed{seed}_t{task:02d}_s{step:03d}'
            marker = out / name / 'complete.json'
            if marker.exists():
                result = json.loads(marker.read_text())
                assert result['status'] == 'complete' and result['config']['source_hashes'] == current_hashes
                for item in json.loads((marker.parent / 'manifest.json').read_text()):
                    file = marker.parent / item['path']
                    assert file.stat().st_size == item['bytes'] and T.sha256(file) == item['sha256'], file
                pending.remove(case)
                continue
            ckpt = T.ARCHIVE / f'runs/seed{seed}/ckpt/t{task:02d}_s{step:03d}.pt'
            if ckpt.exists():
                ready = case, name
                break
        if ready is None:
            if not pending:
                break
            row = dict(status='waiting_for_checkpoint', pending=pending, updated=time.time())
            T.atomic_json(out / 'controller_status.json', row)
            print(json.dumps(row), flush=True)
            time.sleep(300)
            continue
        case, name = ready
        seed, task, step = case
        log = logs / f'{name}_{attempt}.log'
        row = dict(status='running', case=name, pending=len(pending), log=str(log), updated=time.time())
        T.atomic_json(out / 'controller_status.json', row)
        print(json.dumps(row), flush=True)
        try:
            with log.open('x') as output:
                subprocess.run([sys.executable, '-u', str(Path(__file__).with_name(module_name + '.py')),
                    '--seed', str(seed), '--task', str(task), '--step', str(step)],
                    cwd=T.ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
            assert (out / name / 'complete.json').exists()
        except Exception as exc:
            T.atomic_json(out / 'controller_status.json', dict(row, status='failed', error=str(exc)))
            raise
        # Leave this case in pending until its manifest is verified on the next iteration.
    T.atomic_json(out / 'controller_status.json', dict(status='complete', updated=time.time()))
    print(f'All {kind} cases completed', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=('alignment', 'global_noise'), required=True)
    main(ap.parse_args().kind)
