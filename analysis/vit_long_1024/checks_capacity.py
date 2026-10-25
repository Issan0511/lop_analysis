"""Checks K1 and K2 of addendum C (specs/spec_vit_long_1024.md 追補 C.4), before the runs.

K1: checkpoint.pt of S, R0, SL, R0L holds task 100, and its weights and Adam moments equal
    ckpt/model_t100.pt and ckpt/adam_t100.pt bit for bit.
K2: capacity.py --replay-task 100 on fresh and freshL reproduces addendum B's fresh task 100
    (mem_correct, mem_loss, online_correct, online_loss) bit for bit.

    python -m analysis.vit_long_1024.checks_capacity
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .capacity import RUNS, SEED

RAW = RUNS.parent / 'capacity'
OUT = C.REPO / 'results/vit_long_1024_2blk_t100/capacity'


def same(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    return x == y


def k1():
    out = {}
    for arm in C.ARMS:
        d = RUNS / arm / f'seed{SEED}'
        state = torch.load(d / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model = torch.load(d / 'ckpt/model_t100.pt', map_location='cpu', weights_only=False)
        adam = torch.load(d / 'ckpt/adam_t100.pt', map_location='cpu', weights_only=False)
        out[arm] = dict(task=state['task'], weights=same(state['model'], model),
                        adam=same(state['optimizer'], adam['optimizer']), adam_task=adam['task'])
        out[arm]['pass'] = out[arm]['task'] == 100 and out[arm]['weights'] and out[arm]['adam']
    return out


def k2():
    out = {}
    env = dict(os.environ, PYTHONPATH=str(C.REPO), VIT_LONG_PRODUCTION='production_2blk_t100.json',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for state in C.FRESH_ARMS:
        target = RAW / '_checks' / f'K2_{state}'
        subprocess.run([sys.executable, '-m', 'analysis.vit_long_1024.capacity', '--state', state,
                        '--replay-task', '100', '--out', str(target)], cwd=C.REPO, env=env, check=True)
        mine = np.load(target / 'intask.npz')
        ref = np.load(RUNS / state / f'seed{SEED}/task100/intask.npz')
        res = {key: bool(np.array_equal(mine[key], ref[key][0])) for key in
               ('mem_correct', 'mem_loss', 'online_correct', 'online_loss', 'batch_size')}
        res['T90'] = json.loads((target / 'result.json').read_text())['T90']
        res['pass'] = all(v for k, v in res.items() if k != 'T90')
        out[state] = res
    return out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    head, dirty = C.git_state()
    report = dict(git_hash=head, dirty=bool(dirty), K1=k1(), K2=k2())
    report['pass'] = all(v['pass'] for v in report['K1'].values()) and all(v['pass'] for v in report['K2'].values())
    (OUT / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if report['pass'] else 1)
