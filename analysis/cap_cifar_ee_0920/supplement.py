"""Descriptive tables added after completion; no registered labels are changed.

Run from the repository root after report.py. Raw files resolve via the archive
manifest after cleanup. Terms in ledger_mean.csv are sums of adjacent intervals,
then arithmetic means over seeds (units were averaged in ledger_summary.csv).
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.cifar_interventions_0920 import artifact

P = Path('results/cap_cifar_ee_0920')
ARMS = ['ref', 'cap1', 'cap2', 'cap12', 'cap12_bfix']
assert json.loads((P / 'status.json').read_text())['stage'] == 'completed'


def read(name):
    return list(csv.DictReader((P / name).open()))


def write(name, rows):
    with (P / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


ledger = read('ledger_summary.csv')
out = []
for arm in ARMS:
    for window in ['1-10', '1-50', '30-50']:
        for term in sorted({r['term'] for r in ledger}):
            rr = [r for r in ledger if (r['arm'], r['window'], r['term']) == (arm, window, term)]
            assert len(rr) == 10
            out.append(dict(arm=arm, window=window, term=term,
                            total=st.mean(float(r['total']) for r in rr),
                            per_task=st.mean(float(r['per_task']) for r in rr),
                            mean_unit_median=st.mean(float(r['unit_median']) for r in rr),
                            delta=st.mean(float(r['delta']) for r in rr),
                            tier='REPORT_ONLY'))
write('ledger_mean.csv', out)

tasks = read('per_task.csv')
out = []
for arm in ARMS:
    rr = [r for r in tasks if r['arm'] == arm and int(r['task']) > 1]
    assert len(rr) == 490
    for layer in [1, 2]:
        row = dict(arm=arm, layer=layer)
        for key in ['hits', 'removed', 'violations', 'bfix_removed', 'bfix_hits', 'bfix_errors']:
            row[key] = sum(float(r[f'{key}_l{layer}']) for r in rr)
        for key in ['max_ratio', 'max_excess']:
            row[key] = max(float(r[f'{key}_l{layer}']) for r in rr)
        row['tier'] = 'OPERATION_CHECK'
        out.append(row)
write('operation_summary.csv', out)

out = []
for arm in ARMS:
    with np.load(artifact(P / 'arms' / arm / 'ledger_state_t01.npz')) as f:
        first = {k: f[k] for k in ['mu', 'b']}
    with np.load(artifact(P / 'arms' / arm / 'ledger_state_t50.npz')) as f:
        last = {k: f[k] for k in ['mu', 'b']}
    for seed in range(10):
        out.append(dict(arm=arm, seed=seed,
                        mu_norm_t1=float(np.linalg.norm(first['mu'][seed])),
                        mu_norm_t50=float(np.linalg.norm(last['mu'][seed])),
                        mu_vector_change_norm=float(np.linalg.norm(last['mu'][seed] - first['mu'][seed])),
                        b2_mean_change=float((last['b'][seed] - first['b'][seed]).mean()),
                        tier='REPORT_ONLY'))
write('endpoint_movement.csv', out)
print('Wrote ledger_mean.csv, operation_summary.csv, endpoint_movement.csv')
