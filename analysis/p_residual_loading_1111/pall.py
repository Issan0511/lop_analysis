"""Canonical persistent-P MLP replay for p_residual_loading_1111.

The archived persistent run contains measurements, not parameter checkpoints.
This adapter validates EVERY available run_task record with np.array_equal at
EVERY reconstructed task end. It does not claim weight-bit equality to an
archived Pall model: no such weight archive is available here.

Constructing Pall performs the full canonical replay. Importing this module or
calling compare_records on synthetic arrays performs no training. The caller
must preregister/commit before constructing a real-cell object.
"""
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from analysis.loading_bridge_1111 import dynamic as D
from src import stage1_ctrl_1111 as M

RAW = Path('/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw')


def compare_records(saved, rec, task):
    """Fail closed on missing/extra fields, dtype changes, or any unequal value."""
    prefix = f'Pall|t{task:02d}|'
    saved_fields = {key[len(prefix):] for key in saved.files if key.startswith(prefix)}
    generated_fields = set(rec)
    if not saved_fields or saved_fields != generated_fields:
        raise AssertionError(
            f'Pall task {task} record fields differ: '
            f'archived_only={sorted(saved_fields - generated_fields)}, '
            f'generated_only={sorted(generated_fields - saved_fields)}')
    checked = []
    for field in sorted(saved_fields):
        actual = np.asarray(rec[field])
        expected = saved[prefix + field]
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            message = (f'Pall task {task} record mismatch in {field}: '
                       f'actual shape={actual.shape}, dtype={actual.dtype}; '
                       f'archived shape={expected.shape}, dtype={expected.dtype}')
            if actual.shape == expected.shape and np.issubdtype(actual.dtype, np.number):
                difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                if difference.size:
                    message += f'; max_abs={float(np.max(difference)):.17g}'
            raise AssertionError(message)
        checked.append(field)
    return dict(task=int(task), exact_record_match=True, checked_fields=checked,
                n_checked_fields=len(checked), comparison='np.array_equal plus dtype equality',
                archived_weights_available=False)


class Pall(D.MLP):
    """MLP-compatible replay of the stage1 persistent-P arm.

    At the returned task-end snapshot, that task's persistent P intervention has
    already occurred (for tasks >=2). The NEXT task's P intervention is NOT
    applied here: the root runner owns it and must use the canonical stream
    M.own(f'persist|P|adam|{next_task}', seed).
    """

    def __init__(self, seed, act='ELU', tasks=None):
        tasks = [29] if tasks is None else list(tasks)
        if (not tasks or any(not isinstance(t, (int, np.integer)) for t in tasks)
                or any(t < 1 or t > 29 for t in tasks) or len(set(tasks)) != len(tasks)):
            raise ValueError('tasks must be unique integer task ends in 1..29')
        tasks = sorted(map(int, tasks))
        if act not in ('R', 'LR', 'ELU', 'SILU', 'GELU'):
            raise ValueError(f'Unsupported activation: {act}')
        torch.set_num_threads(1)
        self.seed, self.act = int(seed), act
        self.source = RAW / f'persist_P_adam_lr0.001_{act}_s{seed}.npz'
        self.sources = [self.source]
        self.X, self.subset = M.load_X(seed)
        self.ctx = M.Ctx(self.X, act)
        initial = M.initial(seed, (784, M.H, M.K))
        self.V_init = initial[2].clone()
        self.P = M.new_params(initial)
        self.opt = M.make_opt('adam', self.P, .001)
        self.labels, self.orders, self.bases = {}, {}, {}
        self.replay_checks = []
        self.n, self.eval_every, self.recovery_limit = M.N, M.EVAL_EVERY, M.ENGINE_STEPS
        labels_rng = M.stream('env_labels_0913', seed)
        batches_rng = M.stream('env_batch_0913', seed)
        started = time.monotonic()
        with np.load(self.source, allow_pickle=False) as saved:
            # Check source coverage before spending time on a partial replay.
            for task in range(1, max(tasks) + 1):
                prefix = f'Pall|t{task:02d}|'
                if not any(key.startswith(prefix) for key in saved.files):
                    raise AssertionError(f'Pall source has no archived task {task}: {self.source}')
            for task in range(1, max(tasks) + 2):
                y = torch.randint(M.K, (M.N,), generator=labels_rng)
                order = M.order_for(batches_rng, M.ENGINE_STEPS)
                self.labels[task], self.orders[task] = y, order
                if task > max(tasks):
                    break  # Generate the next task's labels/batches, but do not train or apply P.
                if task >= 2:
                    M.intervene('P', self.P, self.opt, self.ctx,
                                M.own(f'persist|P|adam|{task}', seed), self.V_init, 'adam')
                one_hot = F.one_hot(y, M.K).to(torch.float32)
                record, _ = M.run_task(self.P, self.opt, self.ctx, y, one_hot, order,
                                       M.ENGINE_STEPS, freeze_body=0)
                self.replay_checks.append(compare_records(saved, record, task))
                if task in tasks:
                    self.bases[task] = M.snapshot(self.P, self.opt)
                print(f'Pall replay {act} seed{seed} task{task:02d}: '
                      f'all {len(record)} archived fields exactly matched', flush=True)
        self.replay_elapsed_s = time.monotonic() - started
        self.replay_validation = (
            'All available archived run_task measurements matched exactly at every task; '
            'archived Pall weights are unavailable, so weight-bit equality is not asserted.')
