"""Run the registered seed/stage order once, with resumable task checkpoints."""
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
import common as T


def main():
    T.ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (T.ARCHIVE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        basic = json.loads((T.ARCHIVE / 'gate/basic.json').read_text())
        resume = json.loads((T.ARCHIVE / 'gate/resume.json').read_text())
        assert basic['status'] == resume['status'] == 'passed'
        assert basic['source_hashes'] == T.own_hashes()
        head, dirty = T.git_state()
        assert not dirty, dirty
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        logs = T.ARCHIVE / 'logs'
        logs.mkdir(exist_ok=True)
        completed = []

        def report(status, **extra):
            row = dict(status=status, attempt=stamp, git_hash=head,
                       completed=completed, updated=time.time(), **extra)
            T.atomic_json(T.ARCHIVE / 'controller_status.json', row)
            print(json.dumps(row), flush=True)

        for until in (10, 60):
            for seed in T.SEEDS:
                if (T.ARCHIVE / 'STOP').exists():
                    report('paused_by_file')
                    return
                log = logs / f'{stamp}_seed{seed}_through{until}.log'
                report('running', seed=seed, until_task=until, log=str(log))
                try:
                    with log.open('x') as output:
                        proc = subprocess.run(
                            [sys.executable, '-u', str(T.ROOT / 'analysis/vit_two_block_1025/train.py'),
                             '--seed', str(seed), '--until-task', str(until)],
                            cwd=T.ROOT, stdout=output, stderr=subprocess.STDOUT)
                    assert proc.returncode == 0, f'train exit {proc.returncode}: {log}'
                    run = T.ARCHIVE / 'runs' / f'seed{seed}'
                    state = json.loads((run / 'stage_complete.json').read_text())
                    assert state['status'] == 'complete' and state['through_task'] >= until, state
                except Exception as exc:
                    report('failed', seed=seed, until_task=until, log=str(log), error=str(exc))
                    raise
                completed.append(dict(seed=seed, through_task=until, log=str(log)))
        report('complete')


if __name__ == '__main__':
    main()
