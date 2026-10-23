"""Fresh-network control for joudaki_vit_battle_0919 (**post-hoc, not preregistered**).

The battle's online accuracy rises from .72 to .88 across the 40 tasks, so there is no
net degradation. That does not establish that plasticity is intact: representation
learning could be masking a declining ability to fit a new task. The two are separated
by a fresh network trained on the *same* task 40 -- same five classes, same batch order,
same augmentations, all of which depend only on (seed, task) and not on run history.

    fresh_gap = fresh_online_acc(task 40) - continual_online_acc(task 40)

    gap <= 0  the continual network fits the new task at least as well as a fresh one
              -> plasticity is genuinely intact in this box
    gap >  0  the continual network fits it worse -> plasticity loss, masked by transfer

Two differences from the continual run are inherent to the comparison and are NOT
controlled; both are recorded rather than hidden:
  * Adam moments. The continual network carries them across tasks; a fresh one starts
    empty. Part of any negative gap may be moment warm-up rather than plasticity.
  * Dropout draws. These come from the global RNG, whose history cannot be shared with
    a network that has no history. Batch order and augmentation ARE shared.

This file lives outside `analysis/joudaki_vit_battle_0919/` on purpose: that directory is
frozen by sha256 in every run's provenance, and adding a file to it would break resume
and mixed-report guards.
"""
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import TaskImages, task_classes, train_loader
from analysis.joudaki_vit_battle_0919.model import (ARM_ORDER, activations, make_model, prepare_noise,
                                                    reset_head, train_forward, update_adaptive)
from analysis.joudaki_vit_battle_0919.run import DATA, RAW, atomic_json, write_rows

TASK = 40                 # the last task of the continual sequence
STEPS = 500
PRESENTATIONS = 62500     # the registered budget the report also checks
OUT = RAW / 'fresh_control_0919'


def continual_accuracy(arm, seed, task=TASK):
    with (RAW / 'runs' / arm / f'seed{seed}' / 'per_task.csv').open() as f:
        for row in csv.DictReader(f):
            if int(row['task']) == task:
                return float(row['online_acc'])
    raise ValueError(f'task {task} missing for {arm} seed{seed}')


def fresh_accuracy(arm, seed, device='cuda', workers=2):
    """One task, from scratch, with the continual run's own data stream for that task."""
    class_ids = task_classes(seed)[TASK - 1]
    classes = torch.tensor(class_ids, device=device)
    model = make_model(arm, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., fused=True)
    forward = train_forward(model, 'compile')
    reset_head(model)                      # the continual run resets the head every task
    model.train()
    dataset = TaskImages(DATA, class_ids, 'train')
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for x, y in train_loader(dataset, seed, TASK, STEPS, workers):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        logits = forward(x).index_select(1, classes)
        loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite loss in fresh {arm} seed{seed}')
        correct += (logits.detach().argmax(1) == y).sum()
        total += len(y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'),
                                       error_if_nonfinite=True, foreach=True)
        optimizer.step()
        update_adaptive(model)
    # 2500 images, 20 batches per epoch including a 68-image tail, 25 epochs.
    assert total == PRESENTATIONS, total
    return float(correct / total)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ARM_ORDER:
        for seed in range(10):
            began = time.monotonic()
            fresh = fresh_accuracy(arm, seed)
            continual = continual_accuracy(arm, seed)
            rows.append(dict(arm=arm, seed=seed, task=TASK, fresh_online_acc=fresh,
                             continual_online_acc=continual, fresh_gap=fresh - continual,
                             seconds=round(time.monotonic() - began, 2)))
            print(json.dumps(rows[-1]), flush=True)
            write_rows(OUT / 'fresh_control.csv', rows)
    atomic_json(OUT / 'fresh_control.json', dict(
        task=TASK, steps=STEPS, arms=list(ARM_ORDER), seeds=list(range(10)),
        note='post-hoc, not preregistered; Adam moments and dropout history not controlled',
        rows=rows))
    print(f'wrote {OUT}/fresh_control.csv', flush=True)


if __name__ == '__main__':
    main()
