"""Data-free preflight of vit_long_1024 (spec §8), on deterministic synthetic images.

    python -m analysis.vit_long_1024.checks_nodata   ->  results/vit_long_1024/checks/checks_nodata.json

Short runs go to RAW/checks/nodata_<time>/ as separate processes, like production runs.
"""
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.factorial_joudaki_vit_0919 import cell_config
from analysis.joudaki_vit_battle_0919.data import task_classes
from analysis.joudaki_vit_battle_0919.model import BattleActivation, activations, reset_head
from analysis.joudaki_vit_battle_0919.run import atomic_json, get_rng, sha256
from analysis.randlabel_control_joudaki_vit_0919 import random_labels

from . import common as C
from . import judge as J
from .run import Recorder

PY = os.environ.get('VIT_LONG_PY', sys.executable)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


PROD = C.production_layout()
LAYOUT_ARGS = ['--n-labels', str(PROD['n_labels']), '--images-per-class', str(PROD['images_per_class']),
               '--augment', PROD['augment'], '--dropout', str(PROD['dropout']),
               '--depth', str(PROD['depth']), '--mlp-ratio', str(PROD['mlp_ratio'])]


def run_job(out, arm, seed, *extra, check=True):
    """A short synthetic run in the production layout (labels, images per class), small T and U."""
    command = [PY, '-m', 'analysis.vit_long_1024.run', '--arm', arm, '--seed', str(seed), '--smoke',
               '--data-kind', 'synthetic', '--out', str(out), *LAYOUT_ARGS, *map(str, extra)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'console.log').open('a') as log:
        log.write(' '.join(command) + '\n')
        log.flush()
        return subprocess.run(command, cwd=C.REPO, stdout=log, stderr=subprocess.STDOUT, check=check).returncode


def intask(out):
    with np.load(out / 'intask.npz') as f:
        return {k: f[k] for k in f.files}


def rows(out):
    return J.read_csv(out / 'per_task.csv')


def weights(out, task):
    return torch.load(out / f'ckpt/model_t{task:02d}.pt', map_location='cpu', weights_only=False)


def strip_seconds(table):
    return [{k: v for k, v in r.items() if not k.endswith('seconds')} for r in table]


# src/rlcifar_mlp_battle_0918.py was extended after the battle's validation (0923: altlabels and
# sgd_postfit engines). The ViT uses only these definitions from it; their source text at the
# validated version (commit 2fa57c30, file sha256 27507ffb...) hashes to this value.
USED_DEFINITIONS = ('Act', 'ReLU', 'make_act', 'ARM_ORDER')
USED_DEFINITIONS_SHA256 = 'cb4f0916a9015fb40d5a72d1e6fdfe4dfe673101d475a95c9b49cbbd38c12dd3'


def used_definitions_sha256(path):
    import ast
    source = Path(path).read_text()
    found = {}
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in USED_DEFINITIONS:
            found[name] = ast.get_source_segment(source, node)
    import hashlib
    return hashlib.sha256('\n'.join(found[k] for k in USED_DEFINITIONS).encode()).hexdigest()


def check_code():
    record = json.loads((C.REPO / 'results/joudaki_vit_battle_0919/checks_compile.json').read_text())
    validated = record['source_hashes_at_validation']
    now = {k: sha256(C.REPO / k) for k in validated}
    battle_now = {str(p.relative_to(C.REPO)): sha256(p)
                  for p in sorted((C.REPO / 'analysis/joudaki_vit_battle_0919').rglob('*'))
                  if p.is_file() and '__pycache__' not in p.parts}
    extra = sorted(k for k in battle_now if k.endswith('.py') and k not in validated)
    changed = [k for k in validated if now[k] != validated[k]]
    battle_dir_ok = not extra and not [k for k in changed if k.startswith('analysis/')]
    src_changed = [k for k in changed if k.startswith('src/')]
    used_ok = used_definitions_sha256(C.REPO / 'src/rlcifar_mlp_battle_0918.py') == USED_DEFINITIONS_SHA256
    src_ok = all(k == 'src/rlcifar_mlp_battle_0918.py' for k in src_changed) and used_ok
    return dict(pass_=battle_dir_ok and src_ok, files=len(validated), changed=changed, added_py_files=extra,
                battle_dir_unchanged=battle_dir_ok, used_definitions=list(USED_DEFINITIONS),
                used_definitions_equal_validated=used_ok,
                note='src/rlcifar_mlp_battle_0918.py changed after 0919 (0923 engines); the definitions the ViT '
                     'uses are textually identical to the validated version')


def check_model(device):
    out = {}
    out['config_is_factorial_N'] = C.CONFIG == cell_config(4.0, 'none')
    model = C.build_model(0, device)
    norms = [m for name, m in model.named_modules() if name.endswith(('norm1', 'norm2')) or name == 'layers.norm']
    out['identity_norms'] = len(norms)
    out['all_norms_identity'] = len(norms) == 13 and all(isinstance(m, nn.Identity) for m in norms)
    out['n_params'] = sum(p.numel() for p in model.parameters())
    out['relu_acts'] = all(isinstance(a, BattleActivation) and a.arm == 'R' for a in activations(model))
    x = torch.randn(8, 3, 64, 64, generator=torch.Generator().manual_seed(5)).to(device)
    model.eval()
    with torch.no_grad():
        direct = model(x)
        traced, blocks = C.trace(model, x)
    out['trace_logits_bit_equal'] = bool(torch.equal(direct, traced))
    out['trace_blocks'] = len(blocks)
    # The production model (depth and FFN width from the production file) for the loss-row checks.
    model = C.build_model(0, device, PROD['dropout'], False, PROD['depth'], PROD['mlp_ratio'])
    out['production_depth'] = model.depth
    out['production_n_params'] = sum(p.numel() for p in model.parameters())
    out['production_n_params_expected'] = C.n_params_expected(PROD['depth'], PROD['mlp_ratio'], False)
    norms_p = [m for name, m in model.named_modules() if name.endswith(('norm1', 'norm2')) or name == 'layers.norm']
    out['production_norms_identity'] = len(norms_p) == 2 * PROD['depth'] + 1 and all(isinstance(m, nn.Identity) for m in norms_p)
    model.eval()
    with torch.no_grad():
        out['production_trace_logits_bit_equal'] = bool(torch.equal(model(x), C.trace(model, x)[0]))
    model.train()
    for arm, task in (('R0', 2), ('S', 2)):
        rows_t = torch.tensor(C.loss_classes(arm, 0, task, PROD['n_labels'], PROD['images_per_class']), device=device)
        model.zero_grad(set_to_none=True)
        y = torch.randint(0, PROD['n_labels'], (8,), generator=torch.Generator().manual_seed(6)).to(device)
        loss = F.cross_entropy(model(x).index_select(1, rows_t), y)
        loss.backward()
        grad = model.layers['out'].weight.grad
        outside = torch.ones(200, dtype=torch.bool, device=device)
        outside[rows_t] = False
        out[f'{arm}_loss_rows'] = len(rows_t)
        out[f'{arm}_finite_backward'] = bool(torch.isfinite(loss)) and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        out[f'{arm}_zero_grad_outside_loss_rows'] = bool((grad[outside] == 0).all()) and bool(
            (model.layers['out'].bias.grad[outside] == 0).all())
        out[f'{arm}_nonzero_grad_on_loss_rows'] = bool(grad[rows_t].abs().sum() > 0)
    # LN arms (R0L, SL): the randlabel cell's affine LayerNorms, same initial shared weights.
    ln = C.build_model(0, device, PROD['dropout'], True, PROD['depth'], PROD['mlp_ratio'])
    plain = C.build_model(0, device, PROD['dropout'], False, PROD['depth'], PROD['mlp_ratio'])
    lns = [m for name, m in ln.named_modules() if name.endswith(('norm1', 'norm2')) or name == 'layers.norm']
    out['ln_config_is_randlabel_cell'] = C.model_config(C.DROPOUT, True) == cell_config(4.0, 'layer')
    out['ln_affine_layernorms'] = len(lns) == 2 * PROD['depth'] + 1 and all(
        isinstance(m, nn.LayerNorm) and m.elementwise_affine for m in lns)
    out['ln_n_params'] = sum(p.numel() for p in ln.parameters())
    a, b = plain.state_dict(), ln.state_dict()
    out['ln_shared_initial_weights_equal'] = all(torch.equal(a[k], b[k]) for k in a if isinstance(a[k], torch.Tensor))
    out['production_dropout_applied'] = all(m.p == PROD['dropout'] for m in ln.modules() if isinstance(m, nn.Dropout))
    ln.eval()
    with torch.no_grad():
        out['ln_trace_logits_bit_equal'] = bool(torch.equal(ln(x), C.trace(ln, x)[0]))
    out['pass_'] = (out['config_is_factorial_N'] and out['all_norms_identity'] and out['n_params'] == 10814024
                    and out['relu_acts'] and out['trace_logits_bit_equal']
                    and all(out[f'{a}_{k}'] for a in ('R0', 'S')
                            for k in ('finite_backward', 'zero_grad_outside_loss_rows', 'nonzero_grad_on_loss_rows'))
                    and out['ln_config_is_randlabel_cell'] and out['ln_affine_layernorms']
                    and out['ln_n_params'] == C.n_params_expected(PROD['depth'], PROD['mlp_ratio'], True)
                    and out['production_n_params'] == out['production_n_params_expected']
                    and out['production_norms_identity'] and out['production_trace_logits_bit_equal']
                    and out['ln_shared_initial_weights_equal']
                    and out['production_dropout_applied'] and out['ln_trace_logits_bit_equal'])
    return out


def check_plan():
    out = dict(seeds=list(C.SEEDS), tasks='1-61')
    ok = True
    for seed in C.SEEDS:
        ok &= C.cycle_classes(seed, 0) == task_classes(seed)
        for cycle in (0, 1):
            groups = C.cycle_classes(seed, cycle)
            ok &= len(groups) == 40 and sorted(sum(groups, [])) == list(range(200))
        seen = set()
        for task in range(1, 62):
            plan = C.task_plan(seed, task)
            ok &= plan['first'] == (0 if task <= 40 else 50) and plan['count'] == 50
            pairs = {(c, j) for c in plan['classes'] for j in range(plan['first'], plan['first'] + plan['count'])}
            ok &= not (pairs & seen) and len(pairs) == 250
            seen |= pairs
            ok &= C.loss_classes('S', seed, task) == C.task_plan(seed, 1)['classes']
            ok &= C.loss_classes('R0', seed, task) == plan['classes']
            labels = C.task_labels(250, seed, task)
            ok &= torch.equal(labels, random_labels(250, seed, task)) and torch.equal(labels, C.task_labels(250, seed, task))
            ok &= int(labels.min()) >= 0 and int(labels.max()) <= 4
        ok &= not torch.equal(C.task_labels(250, seed, 1), C.task_labels(250, seed, 2))
    out['registered_layout_ok'] = bool(ok)
    # The production layout (Issa 1024: 10 labels x 25 images): 20 tasks per pass over the 200
    # classes, each pass with the next 25 images of every class; task 61 (J5's last probe) included.
    K, m = PROD['n_labels'], PROD['images_per_class']
    alt = True
    per_cycle = 200 // K
    for seed in C.SEEDS:
        seen = set()
        for cycle in range(3):
            groups = C.cycle_classes(seed, cycle, K)
            alt &= len(groups) == per_cycle and sorted(sum(groups, [])) == list(range(200))
        for task in range(1, 62):
            plan = C.task_plan(seed, task, K, m)
            pairs = {(c, j) for c in plan['classes'] for j in range(plan['first'], plan['first'] + plan['count'])}
            alt &= len(plan['classes']) == K and plan['first'] == m * ((task - 1) // per_cycle) and not (pairs & seen)
            alt &= len(pairs) == K * m
            seen |= pairs
            alt &= C.loss_classes('S', seed, task, K, m) == C.task_plan(seed, 1, K, m)['classes']
            alt &= C.loss_classes('R0', seed, task, K, m) == plan['classes']
            labels = C.task_labels(K * m, seed, task, K)
            alt &= torch.equal(labels, C.task_labels(K * m, seed, task, K)) and int(labels.min()) >= 0
            alt &= int(labels.max()) <= K - 1
    out['production'] = PROD
    out['production_layout_ok'] = bool(alt)
    out['pass_'] = bool(ok and alt)
    return out


def check_rng(device):
    source = C.SyntheticImages()
    model = C.build_model(0, device)
    model.train()
    data, _ = C.task_dataset(source, 0, 1, PROD['n_labels'], PROD['images_per_class'])
    x = C.eval_tensor(data.images, device)
    rows_t = torch.tensor(C.loss_classes('R0', 0, 1, PROD['n_labels'], PROD['images_per_class']), device=device)
    probe = C.eval_tensor(source.val(C.probe_names(source, 0)), device)
    before = get_rng()
    recorder = Recorder(model, x, data.labels.to(device), rows_t, 3, device)
    recorder.take(0)
    recorder.take(1)
    C.layer_summary(model, probe)
    C.factorial_readout(model, x[:16])
    model.eval()
    C.trace(model, probe)
    model.train()
    after = get_rng()
    return dict(pass_=tree_equal(before, after) and model.training, still_training=model.training)


def check_runs(root):
    """S-head, S-arm1, S-recorder, S-resume, S-fresh1 on synthetic images (T = 3 or 2, U = 300)."""
    res = {}
    full = {arm: root / f'{arm}_full' for arm in C.ARMS}
    for arm in C.ARMS:
        run_job(full[arm], arm, 0, '--tasks', 3, '--steps', 300)
    r0, s = rows(full['R0']), rows(full['S'])
    plan = [C.task_plan(0, t, PROD['n_labels'], PROD['images_per_class'])['classes'] for t in (1, 2, 3)]
    # S-head
    s_model = weights(full['S'], 3)
    outside = torch.ones(200, dtype=torch.bool)
    outside[torch.tensor(plan[0])] = False
    head = dict(
        R0_head_zero_at_every_task_start=all(float(r['head_full_norm_start']) == 0. for r in r0),
        S_head_zero_at_task1=float(s[0]['head_full_norm_start']) == 0.,
        S_head_nonzero_at_task2_3=all(float(r['head_full_norm_start']) > 0 for r in s[1:]),
        S_loss_rows_nonzero_at_task2_3_u0=all(float(r['head_norm_start']) > 0 for r in s[1:]),
        R0_loss_rows_are_classes_t=[json.loads(r['loss_rows']) for r in r0] == plan,
        S_loss_rows_are_classes_1=all(json.loads(r['loss_rows']) == plan[0] for r in s),
        S_other_rows_zero_at_task3_end=bool((s_model['layers.out.weight'][outside] == 0).all())
        and bool((s_model['layers.out.bias'][outside] == 0).all()),
        R0_zeroed_flags=[int(r['head_zeroed']) for r in r0], S_zeroed_flags=[int(r['head_zeroed']) for r in s])
    head['pass_'] = all(v for k, v in head.items() if k not in ('R0_zeroed_flags', 'S_zeroed_flags')) and \
        head['R0_zeroed_flags'] == [1, 1, 1] and head['S_zeroed_flags'] == [1, 0, 0]
    res['S_head'] = head
    # S-arm1
    a, b = intask(full['R0']), intask(full['S'])
    task1_equal = all(np.array_equal(a[k][0], b[k][0]) for k in ('online_correct', 'online_loss', 'batch_size',
                                                                   'mem_correct', 'mem_loss', 'up_norm_med'))
    w1_equal = C.state_sha256(weights(full['R0'], 1)) == C.state_sha256(weights(full['S'], 1))
    task2_differs = not np.array_equal(a['online_loss'][1], b['online_loss'][1])
    res['S_arm1_synthetic'] = dict(pass_=task1_equal and w1_equal and task2_differs, task1_records_equal=task1_equal,
                                   task1_weights_equal=w1_equal, task2_differs=task2_differs,
                                   task1_online=float(r0[0]['online_acc']), task2_online_R0=float(r0[1]['online_acc']),
                                   task2_online_S=float(s[1]['online_acc']))
    # S-recorder
    rec = {}
    for arm in C.ARMS:
        bare = root / f'{arm}_bare'
        run_job(bare, arm, 0, '--tasks', 2, '--steps', 300, '--no-recorders', '--save-mode', 'minimal')
        x, y = intask(full[arm]), intask(bare)
        same_steps = all(np.array_equal(x[k][:2], y[k]) for k in ('online_correct', 'online_loss', 'batch_size'))
        same_weights = C.state_sha256(weights(full[arm], 2)) == C.state_sha256(weights(bare, 2))
        rec[arm] = dict(online_per_update_equal=same_steps, task2_weights_equal=same_weights)
    res['S_recorder'] = dict(pass_=all(v['online_per_update_equal'] and v['task2_weights_equal'] for v in rec.values()),
                             **rec)
    # S-resume
    resume = {}
    for arm in C.ARMS:
        cut = root / f'{arm}_resumed'
        run_job(cut, arm, 0, '--tasks', 3, '--steps', 300, '--stop-after-task', 2)
        stopped_early = not (cut / 'done.json').exists() and len(rows(cut)) == 2
        run_job(cut, arm, 0, '--tasks', 3, '--steps', 300)
        x = torch.load(full[arm] / 'checkpoint.pt', map_location='cpu', weights_only=False)
        y = torch.load(cut / 'checkpoint.pt', map_location='cpu', weights_only=False)
        parts = {key: tree_equal(x[key], y[key]) for key in ('config', 'model', 'optimizer', 'rng', 'task', 'layer_rows')}
        parts['rows_except_seconds'] = strip_seconds(x['rows']) == strip_seconds(y['rows'])
        ia, ib = intask(full[arm]), intask(cut)
        parts['intask'] = ia.keys() == ib.keys() and all(np.array_equal(ia[k], ib[k]) for k in ia)
        files = sorted(p.name for p in (full[arm] / 'ckpt').glob('*.pt'))
        parts['saved_files'] = files == sorted(p.name for p in (cut / 'ckpt').glob('*.pt')) and all(
            tree_equal(torch.load(full[arm] / 'ckpt' / f, map_location='cpu', weights_only=False),
                       torch.load(cut / 'ckpt' / f, map_location='cpu', weights_only=False)) for f in files)
        parts['stopped_after_task2'] = stopped_early
        resume[arm] = parts
    nowarm = root / 'R0_resumed_nowarmup'
    run_job(nowarm, 'R0', 0, '--tasks', 3, '--steps', 300, '--stop-after-task', 2, '--no-warmup')
    run_job(nowarm, 'R0', 0, '--tasks', 3, '--steps', 300, '--no-warmup')
    x = torch.load(full['R0'] / 'checkpoint.pt', map_location='cpu', weights_only=False)
    y = torch.load(nowarm / 'checkpoint.pt', map_location='cpu', weights_only=False)
    nowarm_record = dict(online_equal=bool(np.array_equal(intask(full['R0'])['online_loss'][2], intask(nowarm)['online_loss'][2])),
                         weights_equal=tree_equal(x['model'], y['model']),
                         adam_state_equal=tree_equal(x['optimizer'], y['optimizer']))
    res['S_resume'] = dict(pass_=all(all(v.values()) for v in resume.values()), **resume,
                           record_resume_without_compile_replay=nowarm_record)
    # S-fresh1
    fresh = root / 'fresh_task1'
    run_job(fresh, 'fresh', 0, '--fresh-task', 1, '--steps', 300)
    fa, ca = intask(fresh), intask(full['R0'])
    same = all(np.array_equal(fa[k][0], ca[k][0]) for k in ('online_correct', 'online_loss', 'mem_correct'))
    same_w = C.state_sha256(weights(fresh, 1)) == C.state_sha256(weights(full['R0'], 1))
    res['S_fresh1'] = dict(pass_=same and same_w, records_equal=same, weights_equal=same_w)
    # sizes of the saved files (for §6.4 / §10.4)
    size = {name: (full['R0'] / 'ckpt' / name).stat().st_size for name in ('model_t01.pt', 'adam_t01.pt', 'mid_t01.pt')}
    size['checkpoint.pt'] = (full['R0'] / 'checkpoint.pt').stat().st_size
    res['file_bytes'] = size
    return res


def check_ln_runs(root):
    """The LN arms: head protocol, task-1 identity of R0L and SL, and SL's exact resume (T = 2, U = 300)."""
    full = {arm: root / f'{arm}_ln' for arm in ('R0L', 'SL')}
    for arm in full:
        run_job(full[arm], arm, 0, '--tasks', 2, '--steps', 300)
    r0, s = rows(full['R0L']), rows(full['SL'])
    a, b = intask(full['R0L']), intask(full['SL'])
    task1 = all(np.array_equal(a[k][0], b[k][0]) for k in ('online_correct', 'online_loss', 'mem_correct', 'up_norm_med'))
    w1 = C.state_sha256(weights(full['R0L'], 1)) == C.state_sha256(weights(full['SL'], 1))
    head = (float(r0[1]['head_full_norm_start']) == 0. and float(s[1]['head_full_norm_start']) > 0
            and [int(r['head_zeroed']) for r in r0] == [1, 1] and [int(r['head_zeroed']) for r in s] == [1, 0])
    ln_keys = any('norm1.weight' in k for k in weights(full['SL'], 2))
    cut = root / 'SL_ln_resumed'
    run_job(cut, 'SL', 0, '--tasks', 2, '--steps', 300, '--stop-after-task', 1)
    run_job(cut, 'SL', 0, '--tasks', 2, '--steps', 300)
    x = torch.load(full['SL'] / 'checkpoint.pt', map_location='cpu', weights_only=False)
    y = torch.load(cut / 'checkpoint.pt', map_location='cpu', weights_only=False)
    resume = all(tree_equal(x[k], y[k]) for k in ('model', 'optimizer', 'rng', 'layer_rows')) and \
        strip_seconds(x['rows']) == strip_seconds(y['rows'])
    return dict(pass_=task1 and w1 and head and ln_keys and resume, task1_records_equal=task1, task1_weights_equal=w1,
                head_protocol=head, layernorm_in_state=ln_keys, SL_resume_bit_equal=resume,
                online=dict(R0L=[float(r['online_acc']) for r in r0], SL=[float(r['online_acc']) for r in s]))


def check_diverge(root):
    lroot = root / 'launcher_root'
    common = ['--smoke', '--data-kind', 'synthetic', '--tasks', '2', '--steps', '50', '--save-mode', 'minimal',
              *LAYOUT_ARGS]
    plan = [dict(arm='R0', seed=0, args=common + ['--inject-nan', '2:7']), dict(arm='S', seed=0, args=common)]
    env = dict(os.environ, VIT_LONG_RAW=str(lroot))
    code = subprocess.run([PY, '-m', 'analysis.vit_long_1024.launch', '--plan', json.dumps(plan), '--retry-sleep', '1'],
                          cwd=C.REPO, env=env, capture_output=True, text=True)
    first, second = lroot / 'runs/R0/seed0', lroot / 'runs/S/seed0'
    done1 = json.loads((first / 'done.json').read_text()) if (first / 'done.json').exists() else {}
    done2 = json.loads((second / 'done.json').read_text()) if (second / 'done.json').exists() else {}
    partial = np.load(first / 'diverged_task.npz') if (first / 'diverged_task.npz').exists() else None
    out = dict(launcher_exit=code.returncode, done_first=done1, done_second=done2,
               first_rows=len(rows(first)) if (first / 'per_task.csv').exists() else 0,
               first_task1_weights=(first / 'ckpt/model_t01.pt').exists(),
               partial_steps=int(partial['step']) if partial is not None else None)
    out['pass_'] = (code.returncode == 0 and done1.get('status') == 'diverged' and done1.get('task') == 2
                    and done1.get('step') == 7 and out['first_rows'] == 1 and out['first_task1_weights']
                    and out['partial_steps'] == 7 and done2.get('status') == 'complete')
    return out


def check_disk(root):
    out_dir = root / 'disk_guard'
    run_job(out_dir, 'R0', 0, '--tasks', 2, '--steps', 50, '--free-bytes-override', 1)
    status = json.loads((out_dir / 'status.json').read_text())
    return dict(pass_=status.get('status') == 'paused' and status.get('reason') == 'disk'
                and not (out_dir / 'done.json').exists() and not (out_dir / 'per_task.csv').exists(), status=status)


def synthetic_run(arm, seed, online_late, t90_early, t90_late, dead):
    tasks = {}
    for t in range(1, 61):
        late = t >= 41
        tasks[t] = dict(steps=500, online_acc=online_late if late else .8, online_acc_first100=online_late if late else .6,
                        T90=t90_late[(t - 41) % len(t90_late)] if late else t90_early[(t - 1) % len(t90_early)])
    end = {t: [dict(dead005=dead, dead0=dead / 2, readout_dead=dead) for _ in range(6)] for t in (40, 60)}
    return dict(tasks=tasks, end=end, done={})


def check_judge():
    cases = {}

    def runs_of(r0, s):
        return {'R0': {k: synthetic_run('R0', k, *r0[k]) for k in range(5)},
                'S': {k: synthetic_run('S', k, *s[k]) for k in range(5)}}
    good = (.9, [100], [150], .01)
    lost = (.25, [100], [525], .3)
    slow = (.6, [300], [525], .1)
    # J0: S lower in all five seeds -> SHARED_WORSE; J1/J3 per arm.
    runs = runs_of([good] * 5, [lost] * 5)
    cases['worse'] = (J.j0(runs)['label'], J.j1(runs, 'R0')['label'], J.j1(runs, 'S')['label'],
                      J.j3(runs, 'R0')['label'], J.j3(runs, 'S')['label'])
    runs = runs_of([lost] * 5, [good] * 5)
    cases['better'] = (J.j0(runs)['label'],)
    runs = runs_of([good, lost, good, lost, good], [lost, good, lost, good, lost])
    cases['mixed'] = (J.j0(runs)['label'],)
    runs = runs_of([slow] * 5, [slow] * 5)                    # E = 300 > 262.5: never LOSS
    cases['slow'] = (J.j1(runs, 'R0')['label'],)
    runs = runs_of([(.25, [525], [525], .3)] * 5, [good] * 5)
    cases['not_learned'] = (J.j1(runs, 'R0')['label'],)
    runs = runs_of([good] * 5, [good] * 5)
    del runs['S'][0], runs['S'][1]
    cases['insufficient'] = (J.j0(runs)['label'], J.j1(runs, 'S')['label'], J.j3(runs, 'S')['label'])
    expected = dict(worse=('SHARED_WORSE', 'NO_LOSS', 'LOSS', 'NO_DEATH', 'DEATH'), better=('SHARED_BETTER',),
                    mixed=('MIXED',), slow=('NO_LOSS',), not_learned=('NOT_JUDGEABLE',),
                    insufficient=('INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT'))
    onset_run = synthetic_run('R0', 0, .3, [100], [525], .0)
    lab = lambda x: dict(label=x)
    jln = dict(break_=J.jln(lab('NO_LOSS'), lab('LOSS')), hold=J.jln(lab('NO_LOSS'), lab('NO_LOSS')),
               none=J.jln(lab('LOSS'), lab('LOSS')), mixed=J.jln(lab('LOSS'), lab('NO_LOSS')),
               insufficient=J.jln(lab('INSUFFICIENT'), lab('LOSS')))
    jln_ok = jln == dict(break_='PHASE1_BREAKS_SHIELD', hold='SHIELD_HOLDS', none='NO_SHIELD', mixed='MIXED',
                         insufficient='INSUFFICIENT')
    # A task that reaches .9 at u = 200 and then has a block of loss 3.0 > ln 10 is a spike; it is a
    # breakdown if it ends below .5. A task that never reaches .9 has no spike by definition.
    mem = np.zeros((3, 21), dtype=np.int64)
    mem[0, 8:] = 240                                   # reaches at u = 200, stays
    mem[1, 8:] = 240
    mem[1, 20] = 50                                    # ends at .2
    mem[2, :] = 100                                    # never reaches
    loss = np.full((3, 500), .1)
    loss[:, 300:325] = 3.0
    record = dict(tasks=np.array([1, 2, 3]), mem_correct=mem, online_loss=loss,
                  batch_size=np.tile(np.array([128, 122] * 250), (3, 1)))
    sp = J.spikes_of(record, 10, 500)
    spikes_ok = [(r['spike'], r['breakdown']) for r in sp] == [(True, False), (True, True), (False, False)]
    # online .8 for tasks 1-40 and .3 from 41: the 5-task mean first drops below .5 at task 44.
    return dict(pass_=cases == expected and J.onset(onset_run) == 44 and jln_ok and spikes_ok, cases=cases,
                onset_example=J.onset(onset_run), jln=jln, spikes_example=sp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    args = parser.parse_args()
    C.set_determinism()
    device = 'cuda'
    root = Path(args.root) if args.root else C.RAW / 'checks' / f'nodata_{int(time.time())}'
    root.mkdir(parents=True, exist_ok=True)
    results = dict(started=time.time(), root=str(root), host=socket.gethostname(), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(), python=platform.python_version(), production=PROD,
                   note='synthetic runs use the production layout with T = 2-3 and U = 300 (U = 50 for S-diverge); '
                        'R0/S without LN for the full set, R0L/SL with LN for S_ln')
    for name, fn in (('S_code', check_code), ('S_model', lambda: check_model(device)), ('S_plan', check_plan),
                     ('S_rng', lambda: check_rng(device)), ('S_judge', check_judge),
                     ('runs', lambda: check_runs(root)), ('S_ln', lambda: check_ln_runs(root)),
                     ('S_diverge', lambda: check_diverge(root)),
                     ('S_disk', lambda: check_disk(root))):
        began = time.time()
        try:
            value = fn()
        except Exception:
            value = dict(pass_=False, error=traceback.format_exc())
        if name == 'runs' and 'error' not in value:
            results.update(value)
        else:
            results[name] = value
        print(json.dumps({name: value if name != 'runs' else {k: v.get('pass_') if isinstance(v, dict) else v
                                                                  for k, v in value.items()},
                          'seconds': round(time.time() - began, 1)}, default=str)[:2000], flush=True)
    passes = {k: v['pass_'] for k, v in results.items() if isinstance(v, dict) and 'pass_' in v}
    results['passes'] = passes
    results['all_pass'] = all(passes.values()) and len(passes) >= 13
    results['source_hashes_at_validation'] = C.source_hashes()
    results['finished'] = time.time()
    path = C.RESULTS / 'checks/checks_nodata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, json.loads(json.dumps(results, default=str).replace('NaN', 'null')))
    print(json.dumps(dict(all_pass=results['all_pass'], passes=passes, record=str(path))), flush=True)


if __name__ == '__main__':
    main()
