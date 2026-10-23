"""Width x normalization factorial on the random-label ViT cell.

The random-label cell (hidden 1536, LayerNorm on) showed no plasticity loss at all:
window .940, fresh gap -.261, early-late -.062, none of them in the direction of loss.
So the difference from relu_doors_0919's collapsing MLP is not the labels. Two candidate
differences remain, and this runs the 2x2 that separates them:

              hidden 1536      hidden 96
    LN on     already run      W
    LN off    N                WN   <- the MLP box's own combination

Everything else is held at the random-label cell's setting: arm R, seeds 0-9, 40 tasks
of 500 updates, 250 images per task with iid uniform labels redrawn each task, same
image sets, batch order and augmentations.

Outside `analysis/joudaki_vit_battle_0919/` on purpose: that directory is frozen by
sha256 in every battle run's provenance.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from analysis.joudaki_vit_battle_0919.model import MODEL_CONFIG, make_model, train_forward
from analysis.joudaki_vit_battle_0919.run import RAW, atomic_json, write_rows
from analysis.randlabel_control_joudaki_vit_0919 import (ARM, READOUT_TASKS, STEPS, TASKS,
                                                         make_task_dataset, readout,
                                                         train_accuracy, train_one_task)

CELLS = {
    # name: (mlp_ratio, normalization)
    'W':  (0.25, 'layer'),     # narrow FFN, normalisation kept
    'N':  (4.0, 'none'),       # full width, normalisation removed
    'WN': (0.25, 'none'),      # both -- the relu_doors_0919 combination
}


def cell_config(mlp_ratio, normalization):
    return dict(MODEL_CONFIG, mlp_ratio=mlp_ratio, normalization=normalization)


def build(seed, config, device):
    model = make_model(ARM, seed, device, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., fused=True)
    return model, optimizer, train_forward(model, 'compile')


def run_seed(seed, config, device='cuda'):
    """Returns (per-task rows, readout rows, fresh gap row). Divergence is recorded, not raised."""
    model, optimizer, forward = build(seed, config, device)
    rows, readouts, diverged = [], [], None
    for task in range(1, TASKS + 1):
        began = time.monotonic()
        dataset, classes = make_task_dataset(seed, task, device)
        try:
            online = train_one_task(model, forward, optimizer, dataset, classes, seed, task, device)
        except FloatingPointError as error:
            diverged = dict(seed=seed, task=task, reason=str(error))
            print(json.dumps(dict(diverged=diverged)), flush=True)
            break
        rows.append(dict(seed=seed, task=task, online_acc=online,
                         train_acc=train_accuracy(model, dataset, classes, device),
                         seconds=round(time.monotonic() - began, 2)))
        if task in READOUT_TASKS:
            for entry in readout(model, dataset, classes, device):
                readouts.append(dict(seed=seed, task=task, **entry))
        print(json.dumps(rows[-1]), flush=True)
    fresh = None
    if diverged is None:
        dataset, classes = make_task_dataset(seed, TASKS, device)
        model, optimizer, forward = build(seed, config, device)
        try:
            value = train_one_task(model, forward, optimizer, dataset, classes, seed, TASKS, device)
            fresh = dict(seed=seed, task=TASKS, fresh_online_acc=value,
                         continual_online_acc=rows[TASKS - 1]['online_acc'],
                         fresh_gap=value - rows[TASKS - 1]['online_acc'])
            print(json.dumps(fresh), flush=True)
        except FloatingPointError as error:
            diverged = dict(seed=seed, task='fresh', reason=str(error))
    return rows, readouts, fresh, diverged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True, choices=sorted(CELLS))
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    mlp_ratio, normalization = CELLS[args.cell]
    config = cell_config(mlp_ratio, normalization)
    out = RAW.parent / f'factorial_joudaki_vit_0919/{args.cell}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'cell {args.cell}: hidden={int(config["embed_dim"] * mlp_ratio)} '
          f'normalization={normalization}', flush=True)
    rows, readouts, fresh, failures = [], [], [], []
    for seed in range(args.seeds):
        seed_rows, seed_readouts, seed_fresh, diverged = run_seed(seed, config)
        rows.extend(seed_rows)
        readouts.extend(seed_readouts)
        if seed_fresh is not None:
            fresh.append(seed_fresh)
        if diverged is not None:
            failures.append(diverged)
        write_rows(out / 'per_task.csv', rows)
        write_rows(out / 'readouts.csv', readouts)
        write_rows(out / 'fresh_control.csv', fresh)
        atomic_json(out / 'config.json', dict(
            cell=args.cell, arm=ARM, tasks=TASKS, steps=STEPS, mlp_ratio=mlp_ratio,
            hidden=int(config['embed_dim'] * mlp_ratio), normalization=normalization,
            seeds=args.seeds, diverged=failures,
            spec='specs/spec_factorial_joudaki_vit_0919.md'))
    print(f'wrote {out}  diverged={len(failures)}', flush=True)


if __name__ == '__main__':
    main()
