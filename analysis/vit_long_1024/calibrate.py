"""Small T90 calibration on fresh networks (not a registered run; spec §12).

A freshly initialised net learns one task for U updates, with the memorisation rate every
25 updates, for a chosen layout (label values per task, images per class). Each (seed, task)
is its own process, as a production fresh task is. Nothing is saved but the records.

    python -m analysis.vit_long_1024.calibrate --n-labels 10 --images-per-class 25 --steps 500
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from analysis.joudaki_vit_battle_0919.run import atomic_json

from . import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-labels', type=int, default=C.N_LABELS)
    p.add_argument('--images-per-class', type=int, default=C.PER_CLASS)
    p.add_argument('--steps', type=int, default=C.STEPS)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--tasks', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--data-kind', choices=('real', 'synthetic'), default='real')
    p.add_argument('--arm', choices=('fresh', 'freshL'), default='fresh', help='freshL: with affine LayerNorm')
    p.add_argument('--depth', type=int, default=C.DEPTH, help='number of blocks (default 6)')
    p.add_argument('--mlp-ratio', type=float, default=C.MLP_RATIO, help='FFN width / 384 (default 4)')
    p.add_argument('--augment', choices=('on', 'off'), default='on', help='off: no augmentation in training')
    p.add_argument('--dropout', type=float, default=C.DROPOUT, help='dropout_p and attn_drop_rate (default 0.1)')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    tag = (f'{args.arm}_labels{args.n_labels}_per{args.images_per_class}_U{args.steps}_aug{args.augment}'
           f'_drop{args.dropout:g}_depth{args.depth}_mlp{args.mlp_ratio:g}')
    root = Path(args.out) if args.out else C.RAW / 'calibrate' / f'{tag}_{int(time.time())}'
    root.mkdir(parents=True, exist_ok=True)
    python = os.environ.get('VIT_LONG_PY', sys.executable)
    rows = []
    for seed in args.seeds:
        for task in args.tasks:
            out = root / f'seed{seed}_task{task:02d}'
            command = [python, '-m', 'analysis.vit_long_1024.run', '--arm', args.arm, '--fresh-task', str(task),
                       '--seed', str(seed), '--smoke', '--steps', str(args.steps), '--save-mode', 'none',
                       '--n-labels', str(args.n_labels), '--images-per-class', str(args.images_per_class),
                       '--data-kind', args.data_kind, '--augment', args.augment, '--dropout', str(args.dropout),
                       '--depth', str(args.depth), '--mlp-ratio', str(args.mlp_ratio),
                       '--out', str(out)]
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'console.log').open('w') as log:
                subprocess.run(command, cwd=C.REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (out / 'per_task.csv').open() as f:
                row = next(csv.DictReader(f))
            rows.append(dict(seed=seed, task=task, n_images=args.n_labels * args.images_per_class, online_acc=float(row['online_acc']),
                             mem_acc_end=float(row['mem_acc_end']), T50=int(float(row['T50'])),
                             T75=int(float(row['T75'])), T90=int(float(row['T90'])),
                             train_ms_per_update=1000 * float(row['train_seconds']) / args.steps))
            print(json.dumps(rows[-1]), flush=True)
    unreached = args.steps + C.EVAL_EVERY
    summary = dict(layout=dict(n_labels=args.n_labels, images_per_class=args.images_per_class, steps=args.steps,
                               augment=args.augment, dropout=args.dropout, arm=args.arm, depth=args.depth,
                               mlp_ratio=args.mlp_ratio),
                   rows=rows, note=f'T = {unreached} means not reached within {args.steps} updates',
                   data_kind=args.data_kind, time=time.time())
    atomic_json(root / 'calibrate.json', summary)
    print(f'| seed | task | online | memorised at end | T50 | T75 | T90 |')
    for r in rows:
        print(f"| {r['seed']} | {r['task']} | {r['online_acc']:.3f} | {r['mem_acc_end']:.3f} | {r['T50']} | {r['T75']} | {r['T90']} |")
    print(f'wrote {root / "calibrate.json"}')


if __name__ == '__main__':
    main()
