"""One GPU worker for vit_long_1024 (spec §10.3): exclusive lock, seed-outer job order,
task-boundary resume, `STOP` file, at most two retries per job, and the final summary.

    python -m analysis.vit_long_1024.launch            # production (refuses without passing checks)

Paths follow VIT_LONG_RAW / VIT_LONG_DATA; child runs use VIT_LONG_PY or this interpreter.
`--plan` (a JSON list of jobs) is for the S-diverge check only and skips the preflight.
"""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time

from analysis.joudaki_vit_battle_0919.run import atomic_json

from . import common as C


def production_jobs():
    # Seed outermost: the four arms and the seed's fresh tasks stay paired if the queue stops.
    # VIT_LONG_SEEDS (e.g. "0,1,2") keeps only those seeds, to split the queue across rented GPUs
    # with each seed's four arms and fresh tasks on one machine (spec §12); unset = all seeds.
    prod = C.production()
    only = os.environ.get('VIT_LONG_SEEDS', '').strip()
    keep = {int(x) for x in only.split(',')} if only else None
    common = ['--n-labels', str(prod['n_labels']), '--images-per-class', str(prod['images_per_class']),
              '--steps', str(prod['steps']), '--augment', prod['augment'], '--dropout', str(prod['dropout'])]
    for seed in [x for x in prod['seeds'] if keep is None or x in keep]:
        for arm in prod['arms']:                               # R0, S, R0L, SL
            yield dict(arm=arm, seed=seed, args=common + ['--tasks', str(prod['tasks'])])
        for arm in prod['fresh_arms']:                         # fresh (no LN), freshL (LN)
            for task in prod['fresh_tasks']:
                yield dict(arm=arm, seed=seed, fresh_task=task, args=common)


def job_dir(root, job):
    base = root / 'runs' / job['arm'] / f"seed{job['seed']}"
    return base / f"task{job['fresh_task']:02d}" if C.is_fresh(job['arm']) else base


def job_name(job):
    return f"{job['arm']} seed{job['seed']}" + (f" task{job['fresh_task']}" if C.is_fresh(job['arm']) else '')


def job_command(python, job, out):
    command = [python, '-m', 'analysis.vit_long_1024.run', '--arm', job['arm'], '--seed', str(job['seed']),
               '--out', str(out)]
    if C.is_fresh(job['arm']):
        command += ['--fresh-task', str(job['fresh_task'])]
    return command + list(job.get('args', []))


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:
        return f'unavailable: {exc!r}'


def environment_record():
    """S-env: python processes and GPU users before starting (spec §8)."""
    processes = [line for line in _run(['pgrep', '-fa', 'python']).splitlines() if line.strip()]
    apps = [line for line in _run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader']).splitlines() if line.strip()]
    return dict(time=time.time(), host=socket.gethostname(), launcher_pid=os.getpid(),
                python_processes=processes, python_count=len(processes),
                gpu=_run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader']).strip(),
                compute_apps=apps)


def gpu_python_users(record):
    users = []
    for line in record['compute_apps']:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and 'python' in parts[1] and parts[0].isdigit() and int(parts[0]) != os.getpid():
            users.append(line)
    return users


def preflight():
    hashes = C.source_hashes()
    for name in ('checks_nodata.json', 'checks_data.json'):
        path = C.RESULTS / 'checks' / name
        record = json.loads(path.read_text()) if path.exists() else {}
        if not record.get('all_pass'):
            raise RuntimeError(f'Required preflight missing or failed: {path}')
        if record.get('source_hashes_at_validation') != hashes:
            raise RuntimeError(f'Source changed after validation: {path}')
        if record.get('production') != C.production_layout():
            raise RuntimeError(f'Checks were run for {record.get("production")}, not {C.production_layout()}: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', help='JSON list of jobs; checks only (skips preflight and report)')
    parser.add_argument('--retry-sleep', type=float, default=30.)
    args = parser.parse_args(argv)
    root = C.RAW
    root.mkdir(parents=True, exist_ok=True)
    test = args.plan is not None
    if not test:
        preflight()
    env = environment_record()
    atomic_json(root / f'env_{int(env["time"])}.json', env)
    users = gpu_python_users(env)
    if users and not test:
        atomic_json(root / 'status.json', dict(status='refused', reason='GPU already used by python', users=users,
                                               updated=time.time()))
        raise RuntimeError(f'GPU already used by python: {users}')
    lock = (root / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    jobs = json.loads(args.plan) if test else list(production_jobs())
    python = os.environ.get('VIT_LONG_PY', sys.executable)
    for index, job in enumerate(jobs):
        if (root / 'STOP').exists():
            atomic_json(root / 'status.json', dict(status='paused', reason='STOP', next=job_name(job),
                                                   updated=time.time()))
            return
        out = job_dir(root, job)
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'done.json').exists():
            continue
        command = job_command(python, job, out)
        for attempt in range(3):
            atomic_json(root / 'status.json', dict(status='running', job=job_name(job), index=index, jobs=len(jobs),
                                                   attempt=attempt, out=str(out), updated=time.time()))
            with (out / 'console.log').open('a') as log:
                log.write(f'\n# {time.strftime("%Y-%m-%d %H:%M:%S")} attempt {attempt}: {" ".join(command)}\n')
                log.flush()
                code = subprocess.call(command, cwd=C.REPO, stdout=log, stderr=subprocess.STDOUT)
            if (out / 'done.json').exists():
                break
            if code == 0:
                run_status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
                atomic_json(root / 'status.json', dict(status='paused', job=job_name(job), run=run_status,
                                                       updated=time.time()))
                return
            if attempt == 2:
                atomic_json(root / 'status.json', dict(status='failed', job=job_name(job), code=code,
                                                       updated=time.time()))
                raise RuntimeError(f'{job_name(job)} failed three times; inspect {out}/console.log')
            time.sleep(args.retry_sleep)
    if test:
        atomic_json(root / 'status.json', dict(status='complete', jobs=len(jobs), updated=time.time()))
        return
    python_args = [python, '-m']
    subprocess.run(python_args + ['analysis.vit_long_1024.collect'], cwd=C.REPO, check=True)
    subprocess.run(python_args + ['analysis.vit_long_1024.judge'], cwd=C.REPO, check=True)
    atomic_json(root / 'status.json', dict(status='complete', updated=time.time(),
                                           summary=str(root / 'summary/judge.md')))
    atomic_json(root / 'DONE', dict(status='complete', time=time.time(), host=socket.gethostname(),
                                    summary=str(root / 'summary/judge.md')))


if __name__ == '__main__':
    main()
