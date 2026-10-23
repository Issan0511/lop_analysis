"""One vit_long_1024 run (specs/spec_vit_long_1024.md §4.3, §6).

    --arm R0     60 tasks, head (200 rows) zeroed at every task start, loss on classes_t
    --arm S      60 tasks, head zeroed only before task 1, loss on classes_1 for every task
    --arm fresh  one task (--fresh-task 25/50/60) on a freshly initialised net, R0's protocol

The inner loop is the randlabel/factorial loop verbatim (index_select of the five loss
rows, masked cross entropy, non-finite checks, fused Adam, compiled forward). Recorders
use the uncompiled module in eval mode and never touch the global RNG. Continual runs
write a resumable checkpoint at every task boundary; a resumed process first replays the
compile sequence of task 1's first two batch shapes so that it holds the same graphs as
an uninterrupted process (spec §4.1).
"""
import argparse
import json
import math
import os
import platform
import shutil
import socket
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import (UPSTREAM_COMMIT, prepare_noise, reset_head,
                                                    train_forward, update_adaptive)
from analysis.joudaki_vit_battle_0919.run import atomic_json, atomic_save, get_rng, set_rng, sha256, write_rows

from . import common as C


class Diverged(FloatingPointError):
    def __init__(self, task, step, reason):
        super().__init__(f'task {task} step {step}: {reason}')
        self.task, self.step, self.reason = task, step, reason


class Paused(Exception):
    pass


def first_reach(correct, n, percent):
    """First evaluation point u = 25, 50, ... where correct/n >= percent/100, else U + 25."""
    for k in range(1, len(correct)):
        if int(correct[k]) * 100 >= percent * n:
            return k * C.EVAL_EVERY
    return (len(correct) - 1) * C.EVAL_EVERY + C.EVAL_EVERY


class Recorder:
    """The within-task record every 25 updates (spec §6.2), kept on the GPU until task end."""

    def __init__(self, model, x_eval, y_eval, rows, points, device):
        self.model, self.x, self.y, self.rows = model, x_eval, y_eval, rows
        self.correct = torch.zeros(points, dtype=torch.int64, device=device)
        self.loss = torch.zeros(points, dtype=torch.float64, device=device)
        self.up = torch.zeros(points, C.DEPTH, dtype=torch.float64, device=device)
        self.head = torch.zeros(points, len(rows), dtype=torch.float64, device=device)
        self.bias = torch.zeros(points, len(rows), dtype=torch.float64, device=device)

    @torch.no_grad()
    def take(self, k):
        model = self.model
        model.eval()
        for start in range(0, len(self.y), C.BATCH):
            logits = model(self.x[start:start + C.BATCH]).index_select(1, self.rows)
            y = self.y[start:start + C.BATCH]
            self.correct[k] += (logits.argmax(1) == y).sum()
            self.loss[k] += F.cross_entropy(logits, y, reduction='sum').double()
        model.train()
        for i in range(C.DEPTH):
            self.up[k, i] = torch.quantile(C.fc1(model, i).weight.double().norm(dim=1), .5)
        head = model.layers['out']
        self.head[k] = head.weight.index_select(0, self.rows).double().norm(dim=1)
        self.bias[k] = head.bias.index_select(0, self.rows).double()

    def numpy(self):
        return {key: getattr(self, key).cpu().numpy() for key in ('correct', 'loss', 'up', 'head', 'bias')}


def warm_compile(model, forward, rows, n_images, device):
    """Replay task 1's first batch shapes (128, then 122 for 250 images) before resuming (spec §4.1).

    The compiled forward first builds a static graph for 128 and then, at the first 122,
    a graph with a dynamic batch dimension that serves every later batch; the two differ
    in gradient rounding. RNG state is restored, gradients are discarded, Adam is untouched.
    """
    state = get_rng()
    model.train()
    for size in C.first_epoch_sizes(n_images):
        x = torch.zeros(size, 3, 64, 64, device=device)
        y = torch.zeros(size, dtype=torch.long, device=device)
        F.cross_entropy(forward(x).index_select(1, rows), y).backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    set_rng(state)


def save_bytes(n_params):
    weights = 4 * n_params
    return dict(weights=weights, adam=2 * weights, checkpoint=3 * weights)


def remaining_save_bytes(args, tasks_left, n_params):
    size = save_bytes(n_params)
    if args.arm == 'fresh':
        return 2 * size['weights'] * len(tasks_left)
    if args.save_mode == 'none':
        return 0
    if args.save_mode == 'minimal':
        return size['weights'] * len(tasks_left) + size['checkpoint']
    total = sum(size['weights'] for t in tasks_left if t in C.END_TASKS)
    total += sum(size['adam'] + size['weights'] for t in tasks_left if t in C.ADAM_TASKS)
    return total + 2 * size['checkpoint']           # the resume file and its temporary copy


def validate(args):
    if args.steps % C.EVAL_EVERY or args.steps < C.EVAL_EVERY:
        raise ValueError('steps must be a positive multiple of 25')
    if args.arm == 'fresh' and not args.fresh_task:
        raise ValueError('--arm fresh needs --fresh-task')
    if args.smoke:
        return
    prod = C.production()
    expected = dict(steps=prod['steps'], n_labels=prod['n_labels'], images_per_class=prod['images_per_class'],
                    augment=prod['augment'], dropout=prod['dropout'],
                    engine='compile', data_kind='real', recorders=True, save_mode='full', inject_nan='',
                    no_warmup=False, free_bytes_override=0)
    actual = dict(steps=args.steps, n_labels=args.n_labels, images_per_class=args.images_per_class,
                  augment=args.augment, dropout=args.dropout,
                  engine=args.engine, data_kind=args.data_kind, recorders=not args.no_recorders,
                  save_mode=args.save_mode, inject_nan=args.inject_nan, no_warmup=args.no_warmup,
                  free_bytes_override=args.free_bytes_override)
    if args.seed not in prod['seeds'] or actual != expected:
        raise ValueError(f'production needs a seed in {prod["seeds"]} and {expected} (production.json); '
                         f'got seed {args.seed}, {actual}. Use --smoke for anything else.')
    if args.arm in C.ARMS and args.tasks != prod['tasks']:
        raise ValueError(f'production continual runs have {prod["tasks"]} tasks')
    if args.arm == 'fresh' and args.fresh_task not in prod['fresh_tasks']:
        raise ValueError(f'production fresh tasks are {prod["fresh_tasks"]}')
    if args.arm not in C.ARMS + ('fresh',):
        raise ValueError(f'unknown arm {args.arm}')


def config_of(args):
    return dict(arm=args.arm, seed=args.seed, tasks=args.tasks if args.arm != 'fresh' else 1,
                fresh_task=args.fresh_task if args.arm == 'fresh' else None, steps=args.steps,
                batch=C.BATCH, lr=C.LR, act=C.ACT, model=C.CONFIG, engine=args.engine,
                data_kind=args.data_kind, recorders=not args.no_recorders, save_mode=args.save_mode,
                n_labels=args.n_labels, images_per_class=args.images_per_class, augment=args.augment,
                dropout=args.dropout, model_config=C.model_config(args.dropout),
                eval_every=C.EVAL_EVERY, end_tasks=list(C.END_TASKS), adam_tasks=list(C.ADAM_TASKS),
                mid_tasks=list(C.MID_TASKS), mid_step=C.MID_STEP, inject_nan=args.inject_nan,
                no_warmup=args.no_warmup, smoke=args.smoke)


def stop_requested(out):
    return (C.RAW / 'STOP').exists() or (out / 'STOP').exists()


def mean_layer(rows, key):
    values = [r[key] for r in rows]
    return float(np.mean(values)) if values else math.nan


def run(args):
    C.set_determinism(args.threads)
    validate(args)
    out = Path(args.out)
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        print('already terminal', flush=True)
        return
    git_hash, dirty = C.git_state()
    if dirty and not args.smoke:
        raise RuntimeError('Commit source and checks before production:\n' + dirty)
    device = args.device
    recorders = not args.no_recorders
    tasks = [args.fresh_task] if args.arm == 'fresh' else list(range(1, args.tasks + 1))
    inject = tuple(int(v) for v in args.inject_nan.split(':')) if args.inject_nan else None
    cfg = config_of(args)
    hashes = C.source_hashes()
    source = C.image_source(args.data_kind)
    model, optimizer = None, None
    model = C.build_model(args.seed, device, args.dropout)
    optimizer = C.build_optimizer(model)
    n_params = sum(p.numel() for p in model.parameters())
    forward = train_forward(model, 'compile') if args.engine == 'compile' else model
    probe = None
    if recorders:
        probe = C.eval_tensor(source.val(C.probe_names(source, args.seed)), device)
    checkpoint = out / 'checkpoint.pt'
    rows, layer_rows, manifest = [], [], []
    intask = {}
    done = 0
    if checkpoint.exists() and args.arm != 'fresh':
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['config'] != cfg or state['source_hashes'] != hashes:
            raise ValueError('Checkpoint config/source mismatch; refuse mixed experiment')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rows, layer_rows, intask, manifest = state['rows'], state['layer_rows'], state['intask'], state['manifest']
        done = state['task']
        if args.engine == 'compile' and not args.no_warmup:
            rows_t = torch.tensor(C.loss_classes(args.arm, args.seed, done + 1 if done < len(tasks) else done,
                                                 args.n_labels, args.images_per_class), device=device)
            warm_compile(model, forward, rows_t, args.n_labels * args.images_per_class, device)
        set_rng(state['rng'])
        print(json.dumps(dict(resumed_after_task=done)), flush=True)
    else:
        for name in ('per_task.csv', 'layer_stats.csv', 'intask.npz', 'manifest.json', 'status.json'):
            (out / name).unlink(missing_ok=True)
        dataset_record = C.RESULTS / 'checks/dataset.json'
        atomic_json(out / 'provenance.json', dict(
            config=cfg, git_hash=git_hash, dirty=dirty, source_hashes=hashes, upstream_commit=UPSTREAM_COMMIT,
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            driver=_driver(), python=platform.python_version(), host=socket.gethostname(),
            data_root=str(C.DATA_ROOT) if args.data_kind == 'real' else 'synthetic',
            dataset=json.loads(dataset_record.read_text()) if dataset_record.exists() else None,
            plan={t: C.task_plan(args.seed, t, args.n_labels, args.images_per_class) for t in tasks},
            loss_rows={t: C.loss_classes(args.arm, args.seed, t, args.n_labels, args.images_per_class)
                       for t in tasks},
            probe=C.probe_names(source, args.seed), n_params=n_params, started=time.time()))
        if args.arm != 'fresh':
            path = out / 'ckpt/model_t00.pt'
            atomic_save(path, model.state_dict())
            manifest.append(_record(path, out))
            if recorders:
                for entry in C.layer_summary(model, probe):
                    layer_rows.append(dict(task=0, state='init', **entry, readout_neg_mass=math.nan,
                                           readout_dead=math.nan))
    task = done
    step = 0
    try:
        for task in tasks[done:] if args.arm != 'fresh' else tasks:
            if stop_requested(out):
                atomic_json(out / 'status.json', dict(status='paused', reason='STOP', completed=rows[-1]['task'] if rows else 0))
                return
            free = args.free_bytes_override or shutil.disk_usage(out).free
            need = C.FREE_FLOOR + remaining_save_bytes(args, [t for t in tasks if t >= task], n_params)
            if free < need:
                atomic_json(out / 'status.json', dict(status='paused', reason='disk', free=free, need=need,
                                                      completed=rows[-1]['task'] if rows else 0))
                print(json.dumps(dict(paused='disk', free=free, need=need)), flush=True)
                return
            row, layers, record = train_task(args, model, optimizer, forward, source, probe, task, tasks[0],
                                             inject, out, manifest)
            rows.append(row)
            layer_rows.extend(layers)
            for key, value in record.items():
                intask.setdefault(key, []).append(value)
            step = 0
            if args.arm != 'fresh':
                atomic_save(checkpoint, dict(config=cfg, source_hashes=hashes, task=task, model=model.state_dict(),
                                             optimizer=optimizer.state_dict(), rng=get_rng(), rows=rows,
                                             layer_rows=layer_rows, intask=intask, manifest=manifest))
            write_outputs(out, rows, layer_rows, intask, manifest)
            atomic_json(out / 'status.json', dict(status='running', task=task, step=args.steps,
                                                  tasks=len(tasks), updated=time.time()))
            print(json.dumps({k: row[k] for k in ('task', 'online_acc', 'mem_acc_end', 'T90', 'probe_dead005',
                                                    'train_seconds', 'record_seconds', 'seconds')}), flush=True)
            if args.stop_after_task == task and task < tasks[-1]:
                atomic_json(out / 'status.json', dict(status='paused', reason='stop-after-task', completed=task))
                return
        atomic_json(out / 'done.json', dict(status='complete', tasks=len(tasks), last_task=tasks[-1], time=time.time()))
        atomic_json(out / 'status.json', dict(status='complete', tasks=len(tasks), updated=time.time()))
    except Diverged as exc:
        write_outputs(out, rows, layer_rows, intask, manifest)
        atomic_json(out / 'done.json', dict(status='diverged', task=exc.task, step=exc.step, reason=exc.reason,
                                            completed=rows[-1]['task'] if rows else 0, time=time.time()))
        atomic_json(out / 'status.json', dict(status='diverged', task=exc.task, step=exc.step, updated=time.time()))
        print(json.dumps(dict(diverged=dict(task=exc.task, step=exc.step, reason=exc.reason))), flush=True)
    except Exception as exc:
        atomic_json(out / 'status.json', dict(status='error', task=task, reason=repr(exc), time=time.time()))
        raise


def train_task(args, model, optimizer, forward, source, probe, task, first_task, inject, out, manifest):
    device = args.device
    recorders = not args.no_recorders
    begin = time.monotonic()
    data, plan = C.task_dataset(source, args.seed, task, args.n_labels, args.images_per_class,
                                augment=args.augment == 'on')
    rows_list = C.loss_classes(args.arm, args.seed, task, args.n_labels, args.images_per_class)
    rows_t = torch.tensor(rows_list, device=device)
    zeroed = C.zero_head_at(args.arm, task, first_task)
    if zeroed:
        reset_head(model)
    model.train()
    head = model.layers['out']
    head_full_start = float(head.weight.detach().double().norm())
    points = args.steps // C.EVAL_EVERY + 1
    record = None
    record_seconds = 0.
    loop_record = 0.
    if recorders:
        x_eval = C.eval_tensor(data.images, device)
        y_eval = data.labels.to(device)
        x16 = C.eval_tensor(data.images[C.readout_indices(len(data))], device)
        record = Recorder(model, x_eval, y_eval, rows_t, points, device)
        torch.cuda.synchronize()
        r0 = time.monotonic()
        record.take(0)
        torch.cuda.synchronize()
        record_seconds += time.monotonic() - r0
    correct = torch.zeros(args.steps, dtype=torch.int64, device=device)
    losses = torch.zeros(args.steps, dtype=torch.float64, device=device)
    sizes = np.zeros(args.steps, dtype=np.int64)
    save_mid = args.save_mode == 'full' and (args.arm == 'fresh' or task in C.MID_TASKS)
    mid_seconds = 0.
    loader = train_loader(data, args.seed, task, args.steps, args.workers)
    train_start = time.monotonic()
    step = 0
    try:
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prepare_noise(model, len(y))
            logits = forward(x).index_select(1, rows_t)
            loss = F.cross_entropy(logits, y)
            if inject is not None and inject == (task, step):
                loss = loss * float('nan')                    # S-diverge only (smoke runs)
            if not bool(torch.isfinite(loss)):
                raise Diverged(task, step, 'nonfinite training loss')
            correct[step - 1] = (logits.detach().argmax(1) == y).sum()
            losses[step - 1] = loss.detach().double()
            sizes[step - 1] = len(y)
            loss.backward()
            try:
                # Infinity cap checks gradients without changing finite gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
            except RuntimeError as exc:
                if 'non-finite' in str(exc):
                    raise Diverged(task, step, 'nonfinite gradient') from exc
                raise
            optimizer.step()
            update_adaptive(model)
            if recorders and step % C.EVAL_EVERY == 0:
                torch.cuda.synchronize()
                r0 = time.monotonic()
                record.take(step // C.EVAL_EVERY)
                torch.cuda.synchronize()
                loop_record += time.monotonic() - r0
            if save_mid and step == C.MID_STEP:
                m0 = time.monotonic()
                path = out / f'ckpt/mid_t{task:02d}.pt'
                atomic_save(path, model.state_dict())
                manifest.append(_record(path, out))
                mid_seconds += time.monotonic() - m0
            if step % 100 == 0:
                atomic_json(out / 'status.json', dict(status='running', task=task, step=step, updated=time.time()))
    except Diverged:
        _save_partial(out, task, step, correct, losses, sizes)
        raise
    torch.cuda.synchronize()
    if step != args.steps:
        raise RuntimeError(f'loader produced {step} steps, expected {args.steps}')
    train_seconds = time.monotonic() - train_start - loop_record - mid_seconds
    record_seconds += loop_record
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise Diverged(task, args.steps, 'nonfinite parameter')
    other_start = time.monotonic()
    correct_np, losses_np = correct.cpu().numpy(), losses.cpu().numpy()
    n = len(data)
    first = min(100, args.steps)
    chunks = [float(correct_np[i:i + 100].sum() / sizes[i:i + 100].sum()) for i in range(0, args.steps, 100)]
    row = dict(task=task, arm=args.arm, seed=args.seed, cycle=plan['cycle'], position=plan['position'],
               classes=json.dumps(plan['classes']), loss_rows=json.dumps(rows_list), head_zeroed=int(zeroed),
               samples=int(sizes.sum()), steps=args.steps,
               online_acc=float(correct_np.sum() / sizes.sum()),
               online_acc_first100=float(correct_np[:first].sum() / sizes[:first].sum()),
               online_chunks=json.dumps(chunks), online_loss=float((losses_np * sizes).sum() / sizes.sum()),
               head_full_norm_start=head_full_start)
    record_np = dict(online_correct=correct_np, online_loss=losses_np, batch_size=sizes.copy())
    layers = []
    if recorders:
        rec = record.numpy()
        record_np.update(mem_correct=rec['correct'], mem_loss=rec['loss'], up_norm_med=rec['up'],
                         head_row_norm=rec['head'], head_bias=rec['bias'])
        r0 = time.monotonic()
        summary = C.layer_summary(model, probe)
        readout = C.factorial_readout(model, x16)
        record_seconds += time.monotonic() - r0
        for entry, extra in zip(summary, readout):
            layers.append(dict(task=task, state='end', **entry, readout_neg_mass=extra['readout_neg_mass'],
                               readout_dead=extra['readout_dead']))
        row.update(mem_acc_start=float(rec['correct'][0] / n), mem_loss_start=float(rec['loss'][0] / n),
                   mem_acc_end=float(rec['correct'][-1] / n), mem_loss_end=float(rec['loss'][-1] / n),
                   T50=first_reach(rec['correct'], n, 50), T75=first_reach(rec['correct'], n, 75),
                   T90=first_reach(rec['correct'], n, 90),
                   head_norm_start=float(np.sqrt((rec['head'][0] ** 2).sum())),
                   head_norm_end=float(np.sqrt((rec['head'][-1] ** 2).sum())),
                   up_norm_med_start=float(rec['up'][0].mean()), up_norm_med_end=float(rec['up'][-1].mean()),
                   probe_dead005=mean_layer(summary, 'dead005'), probe_dead0=mean_layer(summary, 'dead0'),
                   probe_neg_mass=mean_layer(summary, 'neg_mass'),
                   readout_neg_mass=mean_layer(readout, 'readout_neg_mass'),
                   readout_dead=mean_layer(readout, 'readout_dead'))
        row['reached90'] = int(row['T90'] <= args.steps)
    else:
        for key in ('mem_acc_start', 'mem_loss_start', 'mem_acc_end', 'mem_loss_end', 'T50', 'T75', 'T90',
                    'head_norm_start', 'head_norm_end', 'up_norm_med_start', 'up_norm_med_end', 'probe_dead005',
                    'probe_dead0', 'probe_neg_mass', 'readout_neg_mass', 'readout_dead', 'reached90'):
            row[key] = math.nan
    keep_end = args.save_mode != 'none' and (args.arm == 'fresh' or args.save_mode == 'minimal'
                                              or task in C.END_TASKS)
    if keep_end:
        path = out / f'ckpt/model_t{task:02d}.pt'
        atomic_save(path, model.state_dict())
        manifest.append(_record(path, out))
    if args.save_mode == 'full' and args.arm != 'fresh' and task in C.ADAM_TASKS:
        path = out / f'ckpt/adam_t{task:02d}.pt'
        atomic_save(path, dict(task=task, optimizer=optimizer.state_dict(), rng=get_rng()))
        manifest.append(_record(path, out))
    row.update(train_seconds=round(train_seconds, 3), record_seconds=round(record_seconds, 3),
               other_seconds=round(time.monotonic() - begin - train_seconds - record_seconds, 3),
               seconds=round(time.monotonic() - begin, 3))
    return row, layers, record_np


def write_outputs(out, rows, layer_rows, intask, manifest):
    write_rows(out / 'per_task.csv', rows)
    write_rows(out / 'layer_stats.csv', layer_rows)
    if intask:
        arrays = {key: np.stack(values) for key, values in intask.items()}
        arrays['tasks'] = np.array([r['task'] for r in rows])
        arrays['u'] = np.arange(0, arrays['online_correct'].shape[1] + 1, C.EVAL_EVERY)
        temp = out / 'intask.tmp.npz'
        np.savez_compressed(temp, **arrays)
        os.replace(temp, out / 'intask.npz')
    atomic_json(out / 'manifest.json', manifest)


def _save_partial(out, task, step, correct, losses, sizes):
    temp = out / 'diverged_task.tmp.npz'
    np.savez_compressed(temp, task=task, step=step, online_correct=correct.cpu().numpy()[:step],
                        online_loss=losses.cpu().numpy()[:step], batch_size=sizes[:step])
    os.replace(temp, out / 'diverged_task.npz')


def _record(path, root):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path))


def _driver():
    try:
        import subprocess
        return subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       text=True).strip()
    except Exception:
        return None


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=('R0', 'S', 'fresh'))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tasks', type=int, default=C.TASKS)
    p.add_argument('--steps', type=int, default=C.STEPS)
    p.add_argument('--fresh-task', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--data-kind', choices=('real', 'synthetic'), default='real')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--engine', choices=('compile', 'eager'), default='compile')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after-task', type=int, default=0)
    p.add_argument('--no-recorders', action='store_true')
    p.add_argument('--save-mode', choices=('full', 'minimal', 'none'), default='full')
    p.add_argument('--n-labels', type=int, default=C.N_LABELS, help='label values per task (registered: 5)')
    p.add_argument('--images-per-class', type=int, default=C.PER_CLASS, help='registered: 50')
    p.add_argument('--augment', choices=('on', 'off'), default='on',
                   help="off: training sees the evaluation transform (no augmentation); batch order unchanged")
    p.add_argument('--dropout', type=float, default=C.DROPOUT, help='dropout_p and attn_drop_rate (default 0.1)')
    p.add_argument('--inject-nan', default='', help='TASK:STEP, smoke runs only (S-diverge)')
    p.add_argument('--no-warmup', action='store_true', help='smoke only: resume without the compile replay')
    p.add_argument('--free-bytes-override', type=int, default=0, help='smoke only: S-disk')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
