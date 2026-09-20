#!/usr/bin/env python3
"""Fail-closed, sequential reuse qualification for spec §7 (before H training).

Stops at the first failed prerequisite. An incomplete suite never qualifies a main run.
Frozen engines come from their recorded launch commits; host dependencies must match.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from src import relu_doors_0919 as E

RUN = 'relu_doors_h_0919'
OUT = ROOT / 'results' / RUN / '_checks'
PREREG = '9aac824ba06e837b077dce41ee47c6db7a99ffeb'
REF_SHA = 'e7069acf585ece6914e2d28b3b14def4e94aff11'
DOOR_SHA = 'ab983d9802f273af5135e7a3aa3bef55bedc46cf'
HASHES = {
    'results/rlcifar_mlp_battle_0918/R/per_task.csv': 'f6d1d0a2a5c9d96dad1a3aa58bbba4c41f0711fe38cd14e1dcd2f429ca0d8328',
    'results/relu_doors_0919/C/per_task.csv': '7de7284dc8c292a032f750f94cb9f51a8bf0eb1a982529704bd5e2a7dbbfd1dd',
    'results/relu_doors_0919/CH/per_task.csv': '3259e819ec8ad1d06b3bc861f4daa77121dc2acd8d2646422cde9cbaf52467f1',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, obj):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

def frozen(commit, filename):
    for dep in ('src/pmnist_0905.py', 'src/pmnist_rlcifar_0907.py'):
        original = subprocess.check_output(['git', 'show', f'{commit}:{dep}'], cwd=ROOT)
        if original != (ROOT / dep).read_bytes():
            raise ValueError(f'frozen dependency differs: {commit}:{dep}')
    source = subprocess.check_output(['git', 'show', f'{commit}:src/{filename}.py'], cwd=ROOT)
    path = OUT / 'frozen' / f'{filename}_{commit}.py'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(f'frozen_{filename}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def rows(path):
    df = pd.read_csv(path, float_precision='round_trip')
    df = df[df.task.isin([1, 2]) & (df.cond == 'raw')]
    if len(df) != 20 or df.duplicated(['seed', 'task']).any():
        raise ValueError(f'expected exactly 20 unique raw rows: {path}')
    if set(map(tuple, df[['seed', 'task']].to_numpy())) != {(s,t) for s in range(10) for t in (1,2)}:
        raise ValueError('wrong seed/task coverage')
    return df.sort_values(['seed', 'task']).reset_index(drop=True)

def compare(a, b):
    # Textual arm/iv are not measurements. All shared numeric columns except slot
    # are compared after the same parent CSV serialization, without any tolerance.
    cols = [c for c in a.select_dtypes(include='number').columns if c in b and c != 'slot']
    if not cols or len(a) != len(b) or len(a) == 0:
        return {'pass': False, 'compared_values': 0, 'reason': 'empty/schema/shape'}
    aa, bb = a[cols].to_numpy(float), b[cols].to_numpy(float)
    equal = (aa == bb) | (np.isnan(aa) & np.isnan(bb))
    failures = []
    for i,j in np.argwhere(~equal)[:12]:
        failures.append({'seed': int(a.iloc[i].seed), 'task': int(a.iloc[i].task),
                         'column': cols[j], 'left': float(aa[i,j]), 'right': float(bb[i,j])})
    return {'pass': bool(equal.all()), 'compared_values': int(equal.size),
            'mismatch_count': int((~equal).sum()), 'columns': cols, 'first_mismatches': failures}

def states(left, right):
    a = torch.load(left / 'ckpt.pt', map_location='cpu', weights_only=False)
    b = torch.load(right / 'ckpt.pt', map_location='cpu', weights_only=False)
    count = 0
    def eq(x, y):
        nonlocal count
        if isinstance(x, torch.Tensor):
            count += x.numel()
            return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
        if isinstance(x, dict):
            return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(eq(xx, yy) for xx, yy in zip(x, y))
        count += 1
        return x == y
    checks = {k: eq(a[k], b[k]) for k in ('P', 'm', 'v', 'tc', 'g_lab', 'g_batch', 'alive')}
    if 'act_state' in a and 'act_state' in b:
        checks['act_state'] = eq(a['act_state'], b['act_state'])
    return {'pass': all(checks.values()) and count > 0, 'compared_values': count, 'fields': checks}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'S-reuse.json').exists():
        raise RuntimeError('refusing to overwrite qualification')
    if (ROOT / 'results' / RUN / '_launch' / 'STOP').exists():
        raise RuntimeError('STOP present')
    torch.set_num_threads(2)
    dev = E.H.setup('cuda')
    start = {'prereg_commit': PREREG, **E.git_state(),
             'started_at': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
             'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
             'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                               (Path(__file__), ROOT / 'src/relu_doors_0919.py')},
             'git_status': subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).splitlines()}
    save('provenance_start.json', start)
    checks = {}
    def record(name, result):
        checks[name] = result
        save('S-reuse.json', {'pass': False, 'complete': False, 'checks': checks,
                              'reason': 'qualification in progress', 'provenance': start})
        print(name, 'PASS' if result['pass'] else 'FAIL', flush=True)
        if not result['pass']:
            save('S-reuse.json', {'pass': False, 'complete': False, 'checks': checks,
                                  'reason': 'STOP: failed prerequisite ' + name, 'provenance': start})
            save('checks.json', {'all_pass': False, 'status': 'STOP', 'failed_prerequisite': name,
                                'remaining_checks': 'NOT_RUN', 'H_main_run': 'NOT_RUN'})
            return False
        return True
    hashes = {p: sha(ROOT / p) == v for p,v in HASHES.items()}
    if not record('reference_hashes', {'pass': all(hashes.values()), 'files': hashes}):
        return 2
    cifar = E.RC.Cifar10()
    def train(mod, arm, conds, tag):
        dest = OUT / 'runs' / tag
        if dest.exists():
            raise RuntimeError('refusing overwrite ' + str(dest))
        print('START', tag, flush=True)
        started = time.monotonic()
        mod.run(arm, list(range(10)), conds, 2, 400, dev, dest, cifar=cifar,
                snapshots=False, checkpoint=True,
                progress=lambda message: print(tag, message, flush=True))
        print('DONE', tag, 'seconds', time.monotonic()-started, flush=True)
        torch.cuda.empty_cache()
        return dest
    parent = frozen(REF_SHA, 'rlcifar_mlp_battle_0918')
    original = train(parent, 'R', ['raw', 'std'], 'frozen_ref_R20')
    legacy = rows(ROOT / 'results/rlcifar_mlp_battle_0918/R/per_task.csv')
    base = rows(original / 'per_task.csv')
    if not record('frozen_ref_vs_registered_csv', compare(base, legacy)):
        return 2
    # Verify the same comparator rejects corrupted values, seed correspondence,
    # and the wrong condition; no H outcome is generated here.
    bad = legacy.copy()
    bad.loc[0, 'online_acc'] = np.nextafter(bad.loc[0, 'online_acc'], np.inf)
    shifted = legacy.copy()
    shifted['seed'] = (shifted.seed + 1) % 10
    wrong = pd.read_csv(original / 'per_task.csv', float_precision='round_trip')
    wrong = wrong[(wrong.cond == 'std') & wrong.task.isin([1,2])].sort_values(['seed','task']).reset_index(drop=True)
    mutants = {'one_csv_ulp': not compare(base, bad)['pass'],
               'seed_shift': not compare(base, shifted)['pass'],
               'wrong_condition': not compare(base, wrong)['pass']}
    if not record('reference_comparator_mutations', {'pass': all(mutants.values()), 'detected': mutants}):
        return 2
    current = train(E, 'ref', ['raw','std'], 'current_ref_R20')
    if not record('current_ref_R20_rows', compare(rows(current / 'per_task.csv'), base)):
        return 2
    if not record('current_ref_R20_state', states(current, original)):
        return 2
    narrow = train(E, 'ref', ['raw'], 'current_ref_R10')
    if not record('ref_R10_vs_registered_R20_raw', compare(rows(narrow / 'per_task.csv'), base)):
        return 2
    frozen_doors = frozen(DOOR_SHA, 'relu_doors_0919')
    for arm in ('C', 'CH'):
        old = train(frozen_doors, arm, ['raw'], 'frozen_' + arm)
        new = train(E, arm, ['raw'], 'current_' + arm)
        for name, result in (
            ('frozen_' + arm + '_vs_registered', compare(rows(old / 'per_task.csv'), rows(ROOT / 'results/relu_doors_0919' / arm / 'per_task.csv'))),
            ('current_' + arm + '_rows', compare(rows(new / 'per_task.csv'), rows(old / 'per_task.csv'))),
            ('current_' + arm + '_state', states(new, old))):
            if not record(name, result):
                return 2
    # Other registered checks and the C->H mutation are still mandatory.
    save('S-reuse.json', {'pass': False, 'complete': False, 'checks': checks,
                          'reason': 'numeric prerequisite passed; remaining mutation and full suite pending',
                          'provenance': start})
    return 0

if __name__ == '__main__':
    sys.exit(main())
