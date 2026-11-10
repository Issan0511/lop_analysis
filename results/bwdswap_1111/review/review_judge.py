#!/usr/bin/env python3
"""Reviewer's check of analysis/bwdswap_1111/judge.py on a synthetic results tree with known values
(no real B numbers are read).  Writes review_judge.json next to this file."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/issan/Projects/claude/wt/bwdswap_1111')
HERE = Path(__file__).resolve().parent
FAKE, OUTD = HERE / 'judge_fake', HERE / 'judge_out'
PY = '/home/issan/Projects/claude/proj_004_drift/.venv/bin/python'
T975, T95 = 2.6850108468, 2.2621571628  # t quantiles 0.9875 and 0.975, df = 9 (Simpson on the t pdf)
ARMS = {'RC': {'restore': ('N10', 'R1_10', 'R1_10_bwd'), 'reverse': ('N1r', 'S1_10r', 'S1_10r_bwd')},
        'RM': {'restore': ('N10', 'R2_10', 'R2_10_bwd'), 'reverse': ('N2r', 'S2_10r', 'S2_10r_bwd')},
        'F51': {'restore': ('N_c', 'R_ch', 'R_ch_bwd'), 'reverse': ('N_hr', 'S_hcr', 'S_hcr_bwd')}}
rng = np.random.default_rng(7)
truth = {}
for st, dirs in ARMS.items():
    for dr, names in dirs.items():
        for name in names:
            for k in (1, 2):
                truth[(st, dr, name, k)] = {s: float(rng.uniform(0.1, 0.9)) for s in range(10)}
# a tie: B == N on seed 3 of RC restore (k = 1) -> sign '0' for that seed
truth[('RC', 'restore', 'R1_10_bwd', 1)][3] = truth[('RC', 'restore', 'N10', 1)][3]


def put(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


for st in ('RC', 'F51'):
    put(FAKE / st / 'run_checks.json', dict(all_pass=True, checks={}))
    for dr, names in ARMS[st].items():
        for name in names:
            put(FAKE / st / 'arms' / name / 'done.json', dict(complete=True))
            put(FAKE / st / 'arms' / name / 'rows.json',
                [dict(seed=s, online_acc=truth[(st, dr, name, 1)][s], finite=True) for s in range(10)])
for s in range(10):
    d = FAKE / 'RM' / f's{s}'
    put(d / 'done.json', dict(complete=True))
    put(d / 'run_checks.json', dict(all_pass=True, checks={}))
    with (d / 'arms.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'arm', 'k', 'task', 'online_acc', 'finite'])
        w.writeheader()
        for dr, names in ARMS['RM'].items():
            for name in names:
                for k in (1, 2):
                    w.writerow(dict(seed=s, arm=name, k=k, task=0, online_acc=repr(truth[('RM', dr, name, k)][s]), finite=True))

p = subprocess.run([PY, str(ROOT / 'analysis/bwdswap_1111/judge.py'), '--src', str(FAKE), '--out', str(OUTD)],
                   capture_output=True, text=True, cwd=ROOT)
res = dict(exit=p.returncode, stderr_tail=p.stderr[-800:])
paired = {(r['setting'], r['direction'], r['comparison']): r for r in csv.DictReader((OUTD / 'paired.csv').open())}
bad = []
for st, dirs in ARMS.items():
    for dr, (n, x, b) in dirs.items():
        E = {kk: truth[(st, dr, nm, 1)] for kk, nm in zip('NXB', (n, x, b))}
        for comp, a_, b_ in (('X-N', 'X', 'N'), ('B-N', 'B', 'N'), ('B-X', 'B', 'X')):
            diff = np.array([E[a_][s] - E[b_][s] for s in range(10)])
            m, sd = diff.mean(), diff.std(ddof=1)
            exp = dict(mean=m, low975=m - T975 * sd / math.sqrt(10), high975=m + T975 * sd / math.sqrt(10),
                       low95=m - T95 * sd / math.sqrt(10), high95=m + T95 * sd / math.sqrt(10),
                       n_pos=int((diff > 0).sum()), n_neg=int((diff < 0).sum()), n_zero=int((diff == 0).sum()))
            got = paired[(st, dr, comp)]
            for key, v in exp.items():
                if abs(float(got[key]) - v) > 2e-6:
                    bad.append((st, dr, comp, key, float(got[key]), v))
res.update(mismatches=bad, n_checked=len(paired))
verdict = json.loads((OUTD / 'verdict.json').read_text())
res['complete'] = verdict['complete']
res['rm_uses_k1'] = all(
    abs(verdict['settings']['RM']['directions'][dr]['means'][kk] - np.mean(list(truth[('RM', dr, nm, 1)].values()))) < 1e-12
    for dr, names in ARMS['RM'].items() for kk, nm in zip('NXB', names))
(HERE / 'review_judge.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
