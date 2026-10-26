"""Checks of addendum E (specs/spec_vit_long_1024.md 追補 E.4) before the run. Needs a clean, committed tree.

E1  RMSNorm model: every LayerNorm replaced (5 RMSNorm, 0 LayerNorm), gain 1, eps 1e-5, and every other
    initial weight bit-identical to the LayerNorm arm's (seeds 10252010 and 10252011); forward finite.
E2  Smoke runs of SRMS and R0RMS (2 tasks x 100 updates, seed 10252010): R0RMS zeroes the head at both
    tasks and SRMS only at task 1, loss rows are the fixed classes, R0RMS starts task 2 at ln 10, SRMS above it.
E3  The code change leaves the old arms unchanged: smoke S and R0 (no norm) equal addendum D's white-san
    smoke records bit for bit, and smoke SL equals the same smoke run with the previous commit (de93b02a).
E4  The fixed image set of each new seed: task 101's ten classes, images 251-500, same for every task,
    labels from ('randlabel', seed, task).
E5  git_state: new files under specs/ or analysis/vit_long_1024/posthoc_2blk/ do not make a run dirty,
    a new top-level .py under analysis/vit_long_1024/ does.

    python -m analysis.vit_long_1024.checks_rlfix_e
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix'
OUT = REPO / 'results/vit_long_1024_2blk_rlfix'
OLD_COMMIT = 'de93b02a'
TIMING = {'train_seconds', 'record_seconds', 'other_seconds', 'seconds'}
COMMON = ['--tasks', '2', '--steps', '100', '--smoke', '--n-labels', '10', '--images-per-class', '250',
          '--augment', 'off', '--dropout', '0.0', '--depth', '2', '--mlp-ratio', '8.0']


def env(production, repo=REPO):
    return dict(os.environ, PYTHONPATH=str(repo), VIT_LONG_PRODUCTION=production, CUBLAS_WORKSPACE_CONFIG=':4096:8')


def child(production, code):
    out = subprocess.run([PY, '-c', code], cwd=REPO, env=env(production), check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def smoke(arm, seed, production, out, repo=REPO):
    if out.exists():
        subprocess.run(['rm', '-rf', str(out)], check=True)
    subprocess.run([PY, '-m', 'analysis.vit_long_1024.run', '--arm', arm, '--seed', str(seed), '--out', str(out), *COMMON],
                   cwd=repo, env=env(production, repo), check=True, capture_output=True, text=True)
    return out


def same_records(a, b):
    ra, rb = list(csv.DictReader(open(a / 'per_task.csv'))), list(csv.DictReader(open(b / 'per_task.csv')))
    rows = len(ra) == len(rb) and all(x[c] == y[c] for x, y in zip(ra, rb) for c in x if c not in TIMING)
    xa, xb = np.load(a / 'intask.npz'), np.load(b / 'intask.npz')
    arrays = xa.files == xb.files and all(np.array_equal(xa[k], xb[k], equal_nan=True) for k in xa.files)
    return dict(per_task_equal=rows, intask_equal=bool(arrays))


E1 = r'''
import json, torch
from analysis.vit_long_1024 import common as C
def same(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    return x == y
out = {}
for seed in (10252010, 10252011):
    ln, rms = C.build_model(seed, 'cpu', 0.0, True, 2, 8.0), C.build_model(seed, 'cpu', 0.0, 'rms', 2, 8.0)
    sl, sr = ln.state_dict(), rms.state_dict()
    norms = [m for m in rms.modules() if isinstance(m, torch.nn.RMSNorm)]
    x = torch.randn(4, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    y = rms.eval()(x)
    out[seed] = dict(rms=len(norms), layernorm=sum(isinstance(m, torch.nn.LayerNorm) for m in rms.modules()),
                     gains_one=all(bool((m.weight == 1).all()) for m in norms), eps=sorted({m.eps for m in norms}),
                     only_ln_keys=sorted(k for k in sl if k not in sr), only_rms_keys=sorted(k for k in sr if k not in sl),
                     shared_equal=all(same(sl[k], sr[k]) for k in sr),
                     n_params_ln=sum(p.numel() for p in ln.parameters()), n_params_rms=sum(p.numel() for p in rms.parameters()),
                     forward_finite=bool(torch.isfinite(y).all()))
print(json.dumps(out))
'''

E4 = r'''
import json, torch
from analysis.vit_long_1024 import common as C
from analysis.joudaki_vit_battle_0919.model import stream_seed
src = C.image_source('real')
out = {}
for seed in C.production()['seeds']:
    plans = [C.task_plan(seed, t, 10, 250) for t in range(1, 101)]
    same_plan = all({k: v for k, v in p.items() if k != 'task'} == {k: v for k, v in plans[0].items() if k != 'task'} for p in plans)
    task101 = list(C.cycle_classes(seed, 5, 10)[0])
    d1, _ = C.task_dataset(src, seed, 1, 10, 250, augment=False)
    want = [str(q.relative_to(src.root)) for cls in task101 for q in src.train_paths(cls)[250:500]]
    lab = [C.task_labels(2500, seed, t, 10) for t in (1, 2)]
    exp = [torch.randint(0, 10, (2500,), generator=torch.Generator().manual_seed(stream_seed('randlabel', seed, t))) for t in (1, 2)]
    out[seed] = dict(same_plan=same_plan, classes=plans[0]['classes'], classes_task101=plans[0]['classes'] == task101,
                     paths_251_500=d1.paths == want, labels_stream=all(bool(torch.equal(a, b)) for a, b in zip(lab, exp)),
                     labels_differ=float((lab[0] == lab[1]).double().mean()))
print(json.dumps(out))
'''


def e5():
    from analysis.vit_long_1024 import common as C
    _, before = C.git_state()
    probes = [REPO / 'specs/_e5_probe.md', REPO / 'analysis/vit_long_1024/posthoc_2blk/_e5_probe.py']
    for f in probes:
        f.write_text('probe\n')
    try:
        _, harmless = C.git_state()
        top = REPO / 'analysis/vit_long_1024/_e5_probe.py'
        top.write_text('probe\n')
        try:
            _, flagged = C.git_state()
        finally:
            top.unlink()
    finally:
        for f in probes:
            f.unlink()
    return dict(clean_before=before == '', specs_and_posthoc_ignored=harmless == '', top_level_py_flagged='_e5_probe.py' in flagged)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    report = dict(git_hash=head)
    e1 = child('production_2blk_rlfix_e.json', E1)
    e1['pass'] = all(v['rms'] == 5 and v['layernorm'] == 0 and v['gains_one'] and v['eps'] == [1e-05] and v['shared_equal']
                     and len(v['only_ln_keys']) == 5 and all(k.endswith('.bias') for k in v['only_ln_keys'])
                     and not v['only_rms_keys'] and v['n_params_ln'] - v['n_params_rms'] == 5 * 384 and v['forward_finite']
                     for v in e1.values())
    report['E1'] = e1
    C_DIR = RAW / '_checks_e'
    rows = {}
    for arm in ('SRMS', 'R0RMS'):
        d = smoke(arm, 10252010, 'production_2blk_rlfix_e.json', C_DIR / f'smoke_{arm}')
        rr = list(csv.DictReader(open(d / 'per_task.csv')))
        rows[arm] = dict(head_zeroed=[int(r['head_zeroed']) for r in rr], loss_rows=[json.loads(r['loss_rows']) for r in rr],
                         mem_loss_start=[float(r['mem_loss_start']) for r in rr],
                         ms_per_update=[1000 * float(r['train_seconds']) / 100 for r in rr])
    fixed = json.loads(list(csv.DictReader(open(C_DIR / 'smoke_SRMS/per_task.csv')))[0]['classes'])
    e2 = dict(rows, fixed_classes=fixed)
    e2['pass'] = (rows['SRMS']['head_zeroed'] == [1, 0] and rows['R0RMS']['head_zeroed'] == [1, 1]
                  and all(r == fixed for a in rows.values() for r in a['loss_rows'])
                  and abs(rows['R0RMS']['mem_loss_start'][1] - math.log(10)) < 1e-5
                  and rows['SRMS']['mem_loss_start'][1] > rows['R0RMS']['mem_loss_start'][1])
    report['E2'] = e2
    e3 = {}
    for arm in ('S', 'R0'):
        new = smoke(arm, 10252010, 'production_2blk_rlfix.json', C_DIR / f'smoke_{arm}_newcode')
        e3[arm] = same_records(new, RAW / '_checks' / f'smoke_{arm}')
    old_tree = Path('/tmp/claude-1000/-home-issan-Projects-claude/17bddd02-1e83-428c-94a6-1c785e8e100a/scratchpad/old_de93b02a')
    if old_tree.exists():
        subprocess.run(['git', 'worktree', 'remove', '--force', str(old_tree)], cwd=REPO, check=False)
    subprocess.run(['git', 'worktree', 'add', '--detach', str(old_tree), OLD_COMMIT], cwd=REPO, check=True, capture_output=True)
    try:
        old = smoke('SL', 10252010, 'production_2blk_rlfix.json', C_DIR / 'smoke_SL_oldcode', repo=old_tree)
        new = smoke('SL', 10252010, 'production_2blk_rlfix.json', C_DIR / 'smoke_SL_newcode')
        e3['SL_old_vs_new'] = same_records(new, old)
    finally:
        subprocess.run(['git', 'worktree', 'remove', '--force', str(old_tree)], cwd=REPO, check=False)
    e3['pass'] = all(v['per_task_equal'] and v['intask_equal'] for v in e3.values())
    report['E3'] = e3
    e4 = child('production_2blk_rlfix_e.json', E4)
    e4['pass'] = all(v['same_plan'] and v['classes_task101'] and v['paths_251_500'] and v['labels_stream']
                     and v['labels_differ'] < .2 for v in e4.values())
    report['E4'] = e4
    os.environ['VIT_LONG_PRODUCTION'] = 'production_2blk_rlfix_e.json'
    e5r = e5()
    e5r['pass'] = all(e5r.values())
    report['E5'] = e5r
    report['pass_'] = all(report[k]['pass'] for k in ('E1', 'E2', 'E3', 'E4', 'E5'))
    (OUT / 'checks_e.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in report.items()}, ensure_ascii=False))
    sys.exit(0 if report['pass_'] else 1)
