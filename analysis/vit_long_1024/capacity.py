"""Addendum C (specs/spec_vit_long_1024.md 追補 C): teach a saved task-100 state or a fresh net one
random-label task of N unused Tiny ImageNet images and record T50/T75/T90. One run per process,
so that each run compiles its own forward.

    python -m analysis.vit_long_1024.capacity --state S --n 1000 --out <dir>
    python -m analysis.vit_long_1024.capacity --state fresh --replay-task 100 --out <dir>   # check K2

The inner loop is train_task's (run.py) without the end-of-task layer summaries; the Recorder
is run.py's. With --replay-task the run uses that task's own images, labels and batch order
for the production 500 updates and never stops early (check K2).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import (prepare_noise, reset_head, stream_seed, train_forward,
                                                    update_adaptive)
from analysis.joudaki_vit_battle_0919.run import atomic_json, set_rng

from . import common as C
from .run import Recorder, first_reach

SEED, N_LABELS, PER_CLASS, DEPTH, MLP_RATIO = 0, 10, 25, 2, 8.0
U_MAX = {250: 2000, 1000: 4000, 2500: 8000}
FIRST_UNUSED = 250                  # 0-based; tasks 1-100 use images 0-124 of each class
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs'


def capacity_task(source, n):
    """N unused images of task 101's ten classes (n/10 per class, path order) and iid labels."""
    classes = C.task_plan(SEED, 101, N_LABELS, PER_CLASS)['classes']
    images, paths = [], []
    for cls in classes:
        array, names = source.train(cls, FIRST_UNUSED, n // N_LABELS)
        images.append(array)
        paths.extend(names)
    generator = torch.Generator().manual_seed(stream_seed('capacity_labels', SEED, n))
    labels = torch.randint(0, N_LABELS, (n,), generator=generator)
    return C.TaskData(np.concatenate(images), labels, paths, augment=False), classes


def load_state(state, device):
    model = C.build_model(SEED, device, 0.0, state in C.LN_ARMS, DEPTH, MLP_RATIO)
    optimizer = C.build_optimizer(model)
    if not C.is_fresh(state):
        saved = torch.load(RUNS / state / f'seed{SEED}/checkpoint.pt', map_location=device, weights_only=False)
        if saved['task'] != 100:
            raise ValueError(f'{state}: checkpoint is at task {saved["task"]}, expected 100')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        set_rng(saved['rng'])
    return model, optimizer


def run(args):
    C.set_determinism(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        print('already done', flush=True)
        return
    device = 'cuda'
    git_hash, dirty = C.git_state()
    if dirty and not args.allow_dirty:
        raise RuntimeError('Commit source before running:\n' + dirty)
    source = C.image_source('real')
    model, optimizer = load_state(args.state, device)
    forward = train_forward(model, 'compile')
    if args.replay_task:
        task_id = args.replay_task
        data, plan = C.task_dataset(source, SEED, task_id, N_LABELS, PER_CLASS, augment=False)
        classes, steps, early_stop = plan['classes'], 500, False
        rows_list = C.loss_classes(args.state, SEED, task_id, N_LABELS, PER_CLASS)
    else:
        data, classes = capacity_task(source, args.n)
        task_id, steps, early_stop = 1000 + args.n, U_MAX[args.n], True
        rows_list = (C.loss_classes(args.state, SEED, 101, N_LABELS, PER_CLASS) if args.state in C.SHARED_ARMS
                     else classes)
    n = len(data)
    rows_t = torch.tensor(rows_list, device=device)
    zeroed = args.state not in C.SHARED_ARMS
    if zeroed:
        reset_head(model)
    model.train()
    points = steps // C.EVAL_EVERY + 1
    record = Recorder(model, C.eval_tensor(data.images, device), data.labels.to(device), rows_t, points, device)
    begin = time.monotonic()
    record.take(0)
    correct = torch.zeros(steps, dtype=torch.int64, device=device)
    losses = torch.zeros(steps, dtype=torch.float64, device=device)
    sizes = np.zeros(steps, dtype=np.int64)
    loader = train_loader(data, SEED, task_id, steps, args.workers)
    step, last, status = 0, 0, 'budget'
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        logits = forward(x).index_select(1, rows_t)
        loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            status = 'diverged'
            break
        correct[step - 1] = (logits.detach().argmax(1) == y).sum()
        losses[step - 1] = loss.detach().double()
        sizes[step - 1] = len(y)
        loss.backward()
        try:
            # run.py's infinity cap: checks gradients without changing finite gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
        except RuntimeError as exc:
            if 'non-finite' not in str(exc):
                raise
            status = 'diverged'
            break
        optimizer.step()
        update_adaptive(model)
        if step % C.EVAL_EVERY == 0:
            last = step // C.EVAL_EVERY
            record.take(last)
            if early_stop and int(record.correct[last]) * 100 >= 90 * n:
                status = 'reached90'
                break
    torch.cuda.synchronize()
    seconds = time.monotonic() - begin
    rec = {key: value[:last + 1] for key, value in record.numpy().items()}
    done = step if status != 'diverged' else step - 1
    np.savez_compressed(out / 'intask.npz', mem_correct=rec['correct'], mem_loss=rec['loss'],
                        up_norm_med=rec['up'], head_row_norm=rec['head'], head_bias=rec['bias'],
                        online_correct=correct.cpu().numpy()[:done], online_loss=losses.cpu().numpy()[:done],
                        batch_size=sizes[:done], u=np.arange(0, last + 1) * C.EVAL_EVERY)
    result = dict(state=args.state, n=n, replay_task=args.replay_task or None, task_id=task_id, steps_budget=steps,
                  steps_run=done, status=status, classes=classes, first_image=FIRST_UNUSED if not args.replay_task
                  else None, loss_rows=rows_list, head_zeroed=zeroed,
                  T50=first_reach(rec['correct'], n, 50), T75=first_reach(rec['correct'], n, 75),
                  T90=first_reach(rec['correct'], n, 90),
                  mem_acc_start=float(rec['correct'][0] / n), mem_loss_start=float(rec['loss'][0] / n),
                  mem_acc_last=float(rec['correct'][-1] / n), mem_loss_last=float(rec['loss'][-1] / n),
                  seconds=round(seconds, 1), git_hash=git_hash, dirty=bool(dirty), source_hashes=C.source_hashes(),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, finished=time.time())
    atomic_json(out / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('state', 'n', 'status', 'T50', 'T75', 'T90', 'mem_acc_start',
                                             'mem_loss_start', 'mem_acc_last', 'steps_run', 'seconds')}), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True, choices=C.ARMS + C.FRESH_ARMS)
    p.add_argument('--n', type=int, choices=sorted(U_MAX), default=250)
    p.add_argument('--replay-task', type=int, default=0, help='check K2: a fresh arm on that task of addendum B')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--allow-dirty', action='store_true', help='checks only')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
