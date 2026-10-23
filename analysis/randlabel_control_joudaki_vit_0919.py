"""ViT x random labels, one cell (see specs/spec_randlabel_joudaki_vit_0919.md).

Everything is the battle's registered setting except the labels: the same five-class
image set per task, the same batch order and the same augmentations -- all of which
depend only on (seed, task) -- with labels drawn iid uniform over the task's five
classes and redrawn every task, following pmnist_rlcifar_0907's convention.

The battle showed no plasticity loss at all (fresh gap -.249, 0/130 runs positive).
Random labels remove the transferable structure between tasks, so this cell decides
whether that "easiness" came from the labels or from the transformer.

This file lives outside `analysis/joudaki_vit_battle_0919/`: that directory is frozen by
sha256 in every run's provenance and adding a file to it would break the resume and
mixed-report guards.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import TaskImages, stream_seed, task_classes, train_loader
from analysis.joudaki_vit_battle_0919.model import (activations, make_model, masked_logits, prepare_noise,
                                                    reset_head, train_forward, update_adaptive)
from analysis.joudaki_vit_battle_0919.run import DATA, RAW, atomic_json, write_rows

ARM = 'R'
TASKS = 40
STEPS = 500
CLASSES = 5
# 50 of each class's 500 training images [spec 2.1]. At 2500 images the ViT cannot
# memorise random labels in 500 updates at all (task-1 fit .229 against a .2 floor),
# which would make the collapse label true by floor effect rather than by plasticity
# loss. 250 images fit to .932 on task 1 with online .687, above the .5 threshold.
# Presentations stay at 62500: 250 images x 250 epochs, as 2500 x 25.
IMAGES_PER_CLASS = 50
PRESENTATIONS = 62500
READOUT_TASKS = (1, 21, 40)
OUT = RAW.parent / 'randlabel_joudaki_vit_0919'


def random_labels(n, seed, task):
    """iid uniform over the task's five classes, redrawn per task, on its own stream."""
    generator = torch.Generator().manual_seed(stream_seed('randlabel', seed, task))
    return torch.randint(0, CLASSES, (n,), generator=generator)


@torch.no_grad()
def readout(model, dataset, classes, device):
    """Negative-preactivation mass and fully dead channels, on 16 fixed images."""
    model.eval()
    modules = activations(model)
    indices = np.linspace(0, len(dataset) - 1, 16, dtype=int).tolist()
    x = torch.stack([dataset[i][0] for i in indices]).to(device)
    for module in modules:
        module.capture = True
    masked_logits(model, x, classes)
    rows = []
    for i, module in enumerate(modules):
        module.capture = False
        z = module.last_z.reshape(-1, module.last_z.shape[-1]).float()
        module.last_z = None
        rows.append(dict(layer=i + 1, negative_mass=float((z < 0).float().mean()),
                         dead_channels=float((z.max(0).values <= 0).float().mean()),
                         channels=int(z.shape[1])))
    model.train()
    return rows


def train_one_task(model, forward, optimizer, dataset, classes, seed, task, device, workers=2):
    """The battle's inner loop verbatim, on a dataset whose labels have been replaced."""
    reset_head(model)
    model.train()
    correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for x, y in train_loader(dataset, seed, task, STEPS, workers):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        logits = forward(x).index_select(1, classes)
        loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite loss at seed{seed} task{task}')
        correct += (logits.detach().argmax(1) == y).sum()
        total += len(y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'),
                                       error_if_nonfinite=True, foreach=True)
        optimizer.step()
        update_adaptive(model)
    assert total == PRESENTATIONS, total
    return float(correct / total)


@torch.no_grad()
def train_accuracy(model, dataset, classes, device):
    model.eval()
    from analysis.joudaki_vit_battle_0919.data import eval_loader
    correct, seen = 0, 0
    for x, y in eval_loader(dataset, 2):
        x, y = x.to(device), y.to(device)
        correct += int((model(x).index_select(1, classes).argmax(1) == y).sum())
        seen += len(y)
    model.train()
    return correct / seen


def make_task_dataset(seed, task, device):
    class_ids = task_classes(seed)[task - 1]
    dataset = TaskImages(DATA, class_ids, 'train')
    # TaskImages lays the five classes out in blocks of 500, so a head slice of each
    # block keeps the classes balanced.
    keep = [c * 500 + j for c in range(CLASSES) for j in range(IMAGES_PER_CLASS)]
    dataset.images = dataset.images[keep]
    dataset.paths = [dataset.paths[i] for i in keep]
    dataset.labels = random_labels(len(keep), seed, task)          # the one change
    return dataset, torch.tensor(class_ids, device=device)


def run_seed(seed, device='cuda'):
    model = make_model(ARM, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., fused=True)
    forward = train_forward(model, 'compile')
    rows, readouts = [], []
    for task in range(1, TASKS + 1):
        began = time.monotonic()
        dataset, classes = make_task_dataset(seed, task, device)
        online = train_one_task(model, forward, optimizer, dataset, classes, seed, task, device)
        row = dict(seed=seed, task=task, online_acc=online,
                   train_acc=train_accuracy(model, dataset, classes, device),
                   samples=PRESENTATIONS, steps=STEPS, seconds=round(time.monotonic() - began, 2))
        rows.append(row)
        if task in READOUT_TASKS:
            for entry in readout(model, dataset, classes, device):
                readouts.append(dict(seed=seed, task=task, **entry))
        print(json.dumps(row), flush=True)
    return rows, readouts


def fresh_at_last_task(seed, device='cuda'):
    """Same control as the battle's: a from-scratch network on this seed's task 40."""
    dataset, classes = make_task_dataset(seed, TASKS, device)
    model = make_model(ARM, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., fused=True)
    forward = train_forward(model, 'compile')
    return train_one_task(model, forward, optimizer, dataset, classes, seed, TASKS, device)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, readouts, fresh = [], [], []
    for seed in range(10):
        seed_rows, seed_readouts = run_seed(seed)
        rows.extend(seed_rows)
        readouts.extend(seed_readouts)
        continual = seed_rows[TASKS - 1]['online_acc']
        value = fresh_at_last_task(seed)
        fresh.append(dict(seed=seed, task=TASKS, fresh_online_acc=value,
                          continual_online_acc=continual, fresh_gap=value - continual))
        print(json.dumps(fresh[-1]), flush=True)
        write_rows(OUT / 'per_task.csv', rows)
        write_rows(OUT / 'readouts.csv', readouts)
        write_rows(OUT / 'fresh_control.csv', fresh)
    atomic_json(OUT / 'config.json', dict(arm=ARM, tasks=TASKS, steps=STEPS, classes=CLASSES,
                                          seeds=list(range(10)), labels='iid uniform, redrawn per task',
                                          spec='specs/spec_randlabel_joudaki_vit_0919.md'))
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
