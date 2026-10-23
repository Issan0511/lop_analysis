"""Refuse partial production rankings; summarize the preregistered paired endpoints."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .model import ARM_ORDER
from .data import task_classes
from .run import RAW, REPO, atomic_json, write_rows


def sign_test(values):
    x = np.asarray(values)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    n = pos + neg
    p = min(1., 2 * sum(math.comb(n, j) for j in range(min(pos, neg) + 1)) / 2**n) if n else 1.
    winner = 1 if neg <= 1 and p < .05 and pos > neg else -1 if pos <= 1 and p < .05 and neg > pos else 0
    return dict(n_pairs=len(x), positive=pos, negative=neg, ties=len(x)-n,
                p=p, median_difference=float(np.median(x)) if len(x) else None, winner=winner)


def holm(comparisons):
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]['p'])
    previous = 0.
    for rank, idx in enumerate(order):
        previous = max(previous, min(1., comparisons[idx]['p'] * (len(order) - rank)))
        comparisons[idx]['p_holm'] = previous


def summarize(root, output):
    root, output = Path(root), Path(output)
    runs, failures, missing = {}, [], []
    common_hashes = None
    for arm in ARM_ORDER:
        for seed in range(10):
            folder = root / arm / f'seed{seed}'
            done = folder / 'done.json'
            if not done.exists():
                missing.append(f'{arm}/seed{seed}')
                continue
            status = json.loads(done.read_text())
            provenance = json.loads((folder / 'provenance.json').read_text())
            if common_hashes is None:
                common_hashes = provenance['source_hashes']
            if common_hashes != provenance['source_hashes']:
                raise ValueError('Source hashes differ across production runs')
            if provenance['classes'] != task_classes(seed):
                raise ValueError('Class sequence differs from paired seed definition')
            config = provenance['config']
            if config['smoke'] or config['tasks'] != 40 or config['steps'] != 500 or config.get('engine') != 'compile':
                raise ValueError('Smoke/nonregistered run in production folder')
            if config['arm'] != arm or config['seed'] != seed:
                raise ValueError('Run identity mismatch')
            if status['status'] == 'diverged':
                failures.append(dict(arm=arm, seed=seed, **status))
                continue
            if status['status'] != 'complete':
                raise ValueError('Unexpected terminal status')
            with (folder / 'per_task.csv').open() as f:
                rows = list(csv.DictReader(f))
            if [int(r['task']) for r in rows] != list(range(1, 41)):
                raise ValueError(f'Incomplete tasks in {folder}')
            if any(int(r['samples']) != 62500 or int(r['steps']) != 500 for r in rows):
                raise ValueError('Training budget differs from preregistration')
            acc = np.array([float(r['online_acc']) for r in rows])
            if not np.isfinite(acc).all():
                raise ValueError('Nonfinite endpoint')
            runs[arm, seed] = dict(window=float(acc[20:40].mean()),
                                   global_window=float(np.mean([float(r['online_global_acc']) for r in rows[20:40]])),
                                   drop=float(acc[:10].mean() - acc[30:40].mean()),
                                   val_window=float(np.mean([float(r['val_acc']) for r in rows[20:40]])),
                                   train_min=min(float(r['train_acc']) for r in rows),
                                   acc=acc.tolist())
    if missing:
        raise ValueError(f'Incomplete production: {len(missing)}/130 runs missing; ranking refused')
    comparisons = []
    # 12 SNA comparisons plus the two additional structural contrasts; KKA-SNA is already included.
    pairs = [('SNA', a) for a in ARM_ORDER if a != 'SNA'] + [('KKA23', 'KKA'), ('KKT1', 'KKA')]
    for first, second in pairs:
        seeds = [s for s in range(10) if (first, s) in runs and (second, s) in runs]
        comparisons.append(dict(first=first, second=second, seeds=seeds,
            **sign_test([runs[first, s]['window']-runs[second, s]['window'] for s in seeds])))
    holm(comparisons)
    rows = []
    for arm in ARM_ORDER:
        rr = [runs[arm, s] for s in range(10) if (arm, s) in runs]
        med = lambda key: float(np.median([r[key] for r in rr])) if rr else None
        rows.append(dict(arm=arm, completed=len(rr), diverged=10-len(rr), window_median=med('window'),
                         global_window_median=med('global_window'),
                         drop_median=med('drop'), val_window_median=med('val_window'), train_min_median=med('train_min'),
                         collapsed=sum(r['window'] < .5 for r in rr)))
    sna = comparisons[:12]
    label_a = 'SNA_BEATEN' if any(c['winner'] == -1 for c in sna) else 'SNA_TOP' if all(c['winner'] == 1 for c in sna) else 'SNA_TIED_TOP'
    kk = next(c for c in sna if c['second'] == 'KKA')['winner']
    scale, tail = comparisons[-2]['winner'], comparisons[-1]['winner']
    labels = dict(A=label_a, K1={-1:'PERIOD_HURTS', 0:'PERIOD_FREE', 1:'PERIOD_HELPS'}[kk],
                  K2={-1:'SCALE_MATTERS_DOWN', 0:'SCALE_FREE', 1:'SCALE_MATTERS_UP'}[scale],
                  K3={-1:'TAIL2_BETTER', 0:'TAIL_FREE', 1:'TAIL1_BETTER'}[tail])
    if failures:
        labels['qualification'] = 'DIVERGENCE_EXCLUDED; inspect paired counts'
    if any(c['n_pairs'] == 0 for c in comparisons):
        labels = {'qualification': 'INCONCLUSIVE_NO_PAIRS'}
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'verdict.csv', rows)
    atomic_json(output / 'verdict.json', dict(labels=labels, comparisons=comparisons, failures=failures))
    write_rows(output / 'paired_tests.csv', comparisons)
    lines = ['# Joudaki ViT / Tiny ImageNet: 13 activations', '',
             '40 tasks × 500 steps; 10 seeds; online window = tasks 21–40.',
             'Raw paired sign-test labels are exploratory. Holm-adjusted p values accompany all 14 distinct comparisons.',
             'Non-significance does not establish equivalence. This is an activation comparison, not a causal mechanism test.', '',
             '| Arm | n | Window (5-class) | Window (200-class) | Early−late | Validation window | Collapsed |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'NA' if x is None else f'{x:.4f}'
    for r in sorted(rows, key=lambda r: r['window_median'] if r['window_median'] is not None else -1, reverse=True):
        lines.append(f"| {r['arm']} | {r['completed']} | {fmt(r['window_median'])} | {fmt(r['global_window_median'])} | {fmt(r['drop_median'])} | {fmt(r['val_window_median'])} | {r['collapsed']} |")
    lines += ['', 'Registered labels (unadjusted): `' + json.dumps(labels) + '`', '',
              f'Diverged runs: {len(failures)}. Raw data: {root}.',
              'Source: Joudaki et al., arXiv:2510.00304v3 Appendix B; upstream commit 161217078ba52107c94a16602af958a321d62ce3.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm in ARM_ORDER:
        rr = [runs[arm, s]['acc'] for s in range(10) if (arm, s) in runs]
        if rr:
            array = np.array(rr)
            axes[0].plot(range(1, 41), np.median(array, axis=0), label=arm)
    axes[0].set(xlabel='Task', ylabel='Online accuracy (seed median)', ylim=(0, 1), title='500 updates per task')
    axes[0].axhline(.2, color='grey', linestyle=':')
    axes[0].legend(ncol=2, fontsize=8)
    data = [[runs[a, s]['window'] for s in range(10) if (a, s) in runs] for a in ARM_ORDER]
    for i, values in enumerate(data):
        axes[1].scatter([i] * len(values), values, s=16)
    axes[1].set(xticks=range(13), xticklabels=ARM_ORDER, ylabel='Online accuracy, tasks 21–40', ylim=(0, 1))
    axes[1].tick_params(axis='x', rotation=60)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=str(RAW / 'runs'))
    p.add_argument('--output', default=str(REPO / 'results/joudaki_vit_battle_0919'))
    args = p.parse_args()
    summarize(args.root, args.output)
