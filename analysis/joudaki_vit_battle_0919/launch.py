"""One GPU worker, durable per-run logs, bounded retries and automatic final report."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .model import ARM_ORDER
from .run import RAW, REPO, atomic_json, source_hashes


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for filename in ('checks_compile.json', 'speed_checks.json', 'report_checks.json'):
        check = REPO / 'results/joudaki_vit_battle_0919' / filename
        if not check.exists() or not json.loads(check.read_text()).get('all_pass'):
            raise RuntimeError(f'Required preflight missing or failed: {check}')
        if json.loads(check.read_text()).get('source_hashes_at_validation') != source_hashes():
            raise RuntimeError(f'Source changed after validation: {check}')
    lock = (RAW / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    # Seed outermost so all arms receive attention early, without partial ranking.
    for seed in range(10):
        for arm in ARM_ORDER:
            if (RAW / 'STOP').exists():
                atomic_json(RAW / 'status.json', dict(status='paused', arm=arm, seed=seed))
                return
            out = RAW / 'runs' / arm / f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'done.json').exists():
                continue
            command = [sys.executable, '-m', 'analysis.joudaki_vit_battle_0919.run',
                       '--arm', arm, '--seed', str(seed), '--out', str(out), '--engine', 'compile']
            for attempt in range(3):
                atomic_json(RAW / 'status.json', dict(status='running', arm=arm, seed=seed,
                                                     attempt=attempt, updated=time.time()))
                with (out / 'console.log').open('a') as log:
                    code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                if (out / 'done.json').exists():
                    break
                if code == 0:
                    atomic_json(RAW / 'status.json', dict(status='paused', arm=arm, seed=seed))
                    return
                if attempt == 2:
                    atomic_json(RAW / 'status.json', dict(status='failed', arm=arm, seed=seed, code=code))
                    raise RuntimeError(f'{arm} seed{seed} failed; inspect {out}/console.log')
                time.sleep(30)
    subprocess.run([sys.executable, '-m', 'analysis.joudaki_vit_battle_0919.report'], cwd=REPO, check=True)
    atomic_json(RAW / 'status.json', dict(status='complete', updated=time.time(), report=str(REPO / 'results/joudaki_vit_battle_0919/summary.md')))


if __name__ == '__main__':
    main()
