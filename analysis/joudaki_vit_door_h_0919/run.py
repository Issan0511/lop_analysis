"""Task-boundary resumable, paired ViT activation experiment (see registered spec)."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from .data import TaskImages, eval_loader, task_classes, train_loader
from .model import MODEL_CONFIG, UPSTREAM_COMMIT, activations, make_model, masked_logits, reset_head, prepare_noise, train_forward, update_adaptive

REPO = Path(__file__).resolve().parents[2]
RAW = Path.home() / 'Projects/obsidian-research-data/joudaki_vit_door_h_0919'
DATA = Path.home() / 'Projects/obsidian-research-data/datasets/tiny-imagenet-200'


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_save(path, obj):
    temp = Path(str(path) + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def write_rows(path, rows):
    if not rows:
        return
    temp = Path(str(path) + '.tmp')
    with temp.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    paths = sorted((REPO / 'analysis/joudaki_vit_door_h_0919').rglob('*.py'))
    paths += [REPO / 'src/rlcifar_mlp_battle_0918.py', REPO / 'src/pmnist_0905.py',
              REPO / 'src/pmnist_rlcifar_0907.py']
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def record_file(path, root):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path))


def get_rng():
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@torch.no_grad()
def evaluate(model, dataset, classes, device, workers):
    model.eval()
    correct, global_correct, total, loss_sum = 0, 0, 0, 0.
    for x, y in eval_loader(dataset, workers):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        all_logits = model(x)
        logits = all_logits.index_select(1, classes)
        loss_sum += float(F.cross_entropy(logits, y, reduction='sum'))
        correct += int((logits.argmax(1) == y).sum())
        global_correct += int((all_logits.argmax(1) == classes[y]).sum())
        total += len(y)
    return correct / total, loss_sum / total, global_correct / total


def diagnose(model, dataset, classes, device, out, task):
    """Fixed unaugmented validation images; actual autograd activation derivatives."""
    model.eval()
    modules = activations(model)
    # Spread 16 indices over the sorted validation set, fixed for this seed/task.
    indices = np.linspace(0, len(dataset) - 1, 16, dtype=int).tolist()
    x = torch.stack([dataset[i][0] for i in indices]).to(device)
    for module in modules:
        module.capture = True
    with torch.no_grad():
        masked_logits(model, x, classes)
    arrays, metrics = {}, []
    for i, module in enumerate(modules):
        module.capture = False
        z = module.last_z
        module.last_z = None
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError('nonfinite diagnostic preactivation')
        with torch.enable_grad():
            leaf = z.detach().requires_grad_(True)
            gate = torch.autograd.grad(module.phi(leaf).sum(), leaf)[0].detach()
        zcpu = z.cpu().numpy()
        arrays[f'z{i+1}'] = zcpu.astype(np.float16) if np.abs(zcpu).max() <= 65504 else zcpu
        row = dict(task=task, layer=i+1, z_mean=float(z.mean()), z_std=float(z.std(unbiased=False)),
                   z_min=float(z.min()), z_max=float(z.max()),
                   dphi_abs_mean=float(gate.abs().mean()), dphi_zero=float((gate == 0).float().mean()))
        fc1 = model.layers[f'block_{i}'].layers['mlp'].layers['fc1']
        row.update(weight_norm=float(fc1.weight.detach().norm()), bias_norm=float(fc1.bias.detach().norm()))
        if module.V is not None:
            a = (.6 / module.V.sqrt()).clamp(.005, 3.)
            theta = 2 * a * z
            lower, upper = (-2*np.pi, np.pi) if module.arm == 'KKT1' else (-1.5*np.pi, .5*np.pi)
            row.update(alpha_median=float(a.median()), alpha_min=float(a.min()), alpha_max=float(a.max()),
                       outside_band=float(((theta < lower) | (theta > upper)).float().mean()))
            arrays[f'alpha{i+1}'] = a.cpu().numpy()
        metrics.append(row)
    arrays['indices'] = np.array(indices)
    arrays['paths'] = np.array([str(dataset.paths[i].relative_to(DATA))
                               if dataset.paths[i].is_relative_to(DATA) else str(dataset.paths[i]) for i in indices])
    path = out / f'preact_t{task:02d}.npz'
    temp = Path(str(path) + '.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)
    return metrics, path


def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        print('already terminal', flush=True)
        return
    if not args.smoke and (args.seed not in range(10) or args.tasks != 40 or args.steps != 500):
        raise ValueError('Production requires seed0–9, 40 tasks, 500 steps. Use --smoke separately.')
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'analysis', 'specs'],
                                    cwd=REPO, text=True)
    if dirty and not args.smoke:
        raise RuntimeError('Commit source and checks before production: ' + dirty)
    if args.door_frozen and not args.smoke:
        raise ValueError('--door-frozen is a preflight control (S-off); never production')
    cfg = dict(arm=args.arm, seed=args.seed, tasks=args.tasks, steps=args.steps, batch=128,
               lr=1e-4, dtype='float32', engine=args.engine, smoke=args.smoke,
               door_frozen=args.door_frozen, model=MODEL_CONFIG)
    model = make_model(args.arm, args.seed, args.device)
    if args.door_frozen:
        for module in activations(model):
            module.door_frozen = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., **({'foreach': True} if args.engine == 'eager' else {'fused': True}))
    forward = train_forward(model, args.engine)
    classes_by_task = task_classes(args.seed)
    checkpoint = out / 'checkpoint.pt'
    rows, diagnostics, manifest, start = [], [], [], 0
    hashes = source_hashes()
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if state['config'] != cfg or state['source_hashes'] != hashes:
            raise ValueError('Checkpoint config/source mismatch; refuse mixed experiment')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        set_rng(state['rng'])
        rows, diagnostics, manifest = state['rows'], state['diagnostics'], state['manifest']
        start = state['task']
        write_rows(out / 'per_task.csv', rows)
    else:
        atomic_json(out / 'provenance.json', dict(config=cfg, git_hash=git_hash, dirty=dirty,
            source_hashes=hashes, upstream_commit=UPSTREAM_COMMIT, torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), python=platform.python_version(),
            classes=classes_by_task, dataset=str(Path(args.data).resolve()),
            dataset_zip_sha256=json.loads((REPO / 'results/joudaki_vit_door_h_0919/checks.json').read_text())['dataset_zip_sha256']
                if (REPO / 'results/joudaki_vit_door_h_0919/checks.json').exists() else None,
            started=time.time()))
        path = out / 'model_t00.pt'
        atomic_save(path, model.state_dict())
        manifest.append(record_file(path, out))
    task = start
    try:
        for task in range(start + 1, args.tasks + 1):
            if (RAW / 'STOP').exists() or (out / 'STOP').exists():
                atomic_json(out / 'status.json', dict(status='paused', completed=start if not rows else rows[-1]['task']))
                return
            if shutil.disk_usage(out).free < 25 * 1024**3:
                raise RuntimeError('Less than 25 GiB free: pause rather than lose snapshots')
            begin = time.monotonic()
            class_ids = classes_by_task[task-1]
            classes = torch.tensor(class_ids, device=args.device)
            train = TaskImages(args.data, class_ids, 'train')
            val = TaskImages(args.data, class_ids, 'val')
            if task == 1 and start == 0:
                metrics, path = diagnose(model, val, classes, args.device, out, 0)
                diagnostics.extend(metrics)
                manifest.append(record_file(path, out))
            reset_head(model)
            model.train()
            correct = torch.zeros((), dtype=torch.int64, device=args.device)
            global_correct = torch.zeros((), dtype=torch.int64, device=args.device)
            loss_sum = torch.zeros((), dtype=torch.float64, device=args.device)
            total, interval_correct, interval_total = 0, 0, 0
            chunk_acc = []
            loader = train_loader(train, args.seed, task, args.steps, args.workers)
            train_start = time.monotonic()
            for step, (x, y) in enumerate(loader, 1):
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prepare_noise(model, len(y))
                all_logits = forward(x)
                logits = all_logits.index_select(1, classes)
                loss = F.cross_entropy(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite training loss')
                wins = (logits.detach().argmax(1) == y).sum()
                correct += wins
                global_correct += (all_logits.detach().argmax(1) == classes[y]).sum()
                interval_correct = interval_correct + wins
                total += len(y)
                interval_total += len(y)
                loss_sum += loss.detach().double() * len(y)
                loss.backward()
                # Infinity cap checks gradients without changing finite gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
                optimizer.step()
                update_adaptive(model)
                if step % 100 == 0 or step == args.steps:
                    chunk_acc.append(float(interval_correct / interval_total))
                    interval_correct, interval_total = 0, 0
                    atomic_json(out / 'status.json', dict(status='running', task=task, step=step,
                                                          tasks=args.tasks, updated=time.time()))
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - train_start
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite parameter')
            train_acc, train_loss, train_global_acc = evaluate(model, train, classes, args.device, args.workers)
            val_acc, val_loss, val_global_acc = evaluate(model, val, classes, args.device, args.workers)
            metrics, path = diagnose(model, val, classes, args.device, out, task)
            diagnostics.extend(metrics)
            manifest.append(record_file(path, out))
            path = out / f'model_t{task:02d}.pt'
            atomic_save(path, model.state_dict())
            manifest.append(record_file(path, out))
            row = dict(task=task, arm=args.arm, seed=args.seed, online_acc=float(correct / total),
                       online_loss=float(loss_sum / total), train_acc=train_acc, train_loss=train_loss,
                       val_acc=val_acc, val_loss=val_loss, samples=total, steps=args.steps,
                       online_global_acc=float(global_correct / total), train_global_acc=train_global_acc,
                       val_global_acc=val_global_acc,
                       train_seconds=train_seconds, seconds=time.monotonic()-begin,
                       online_chunks=json.dumps(chunk_acc))
            rows.append(row)
            state = dict(config=cfg, source_hashes=hashes, task=task, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), rng=get_rng(), rows=rows,
                         diagnostics=diagnostics, manifest=manifest)
            atomic_save(checkpoint, state)
            write_rows(out / 'per_task.csv', rows)
            atomic_json(out / 'diagnostics.json', diagnostics)
            atomic_json(out / 'manifest.json', manifest)
            print(json.dumps(dict(task=task, tasks=args.tasks, train_seconds=round(train_seconds, 2),
                                  seconds=round(row['seconds'], 2))), flush=True)
            if args.stop_after_task == task and task < args.tasks:
                atomic_json(out / 'status.json', dict(status='paused', completed=task))
                return
        atomic_json(out / 'done.json', dict(status='complete', tasks=args.tasks, time=time.time()))
    except (FloatingPointError, RuntimeError) as exc:
        is_nonfinite = isinstance(exc, FloatingPointError) or 'non-finite' in str(exc)
        if is_nonfinite:
            atomic_json(out / 'done.json', dict(status='diverged', task=task, reason=str(exc), time=time.time()))
        else:
            atomic_json(out / 'status.json', dict(status='error', task=task, reason=str(exc), time=time.time()))
            raise


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tasks', type=int, default=40)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--out', required=True)
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--door-frozen', action='store_true')
    p.add_argument('--stop-after-task', type=int, default=0)
    p.add_argument('--engine', choices=['eager', 'fused', 'compile'], default='eager')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
