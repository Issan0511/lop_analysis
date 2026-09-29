"""Source-faithful RL-MNIST reconstruction and bounded causal intervention arms.

Registered actual-system supplement 0fde22b. CPU float32, single thread,
source ELU backward expm1(z)+1, same streams, native torch.optim.Adam.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
SOURCE_ROOT = Path('/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928')
DATA = Path('/home/issan/Projects/claude/proj_004_drift/data/mnist/train-images-idx3-ubyte.gz')
N, H, K, BATCH, STEPS = 1200, 100, 10, 16, 4000
NAMES = ('W1', 'b1', 'W2', 'b2')
LR, EPS, BETAS = .001, 1e-8, (.9, .999)
CHECKPOINTS = (0, 1, 10, 100, 500, 1000, 2000, 4000)


def load_helpers():
    spec = importlib.util.spec_from_file_location('real_source_helpers', SOURCE_ROOT / 'alpha_ladder_1layer.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


AL = load_helpers()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''): h.update(data)
    return h.hexdigest()


RUNNER_SHA = sha(__file__)


def elu(z):
    return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))


def gate(z):
    return torch.where(z > 0, torch.ones_like(z), torch.expm1(z.clamp_max(0)) + 1.0)


class SourceELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z); return elu(z)

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        return g * gate(z)


def native_optimizer(P):
    return torch.optim.Adam([{'params': [P[0], P[1]], 'lr': LR}, {'params': [P[2]]}, {'params': [P[3]]}],
                            lr=LR, betas=BETAS, eps=EPS)


def initial(seed):
    torch.manual_seed(seed)
    ax = AL.read_idx(DATA).reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=AL.stream('rl_subset', seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    P = [q.to(torch.float32).requires_grad_(True) for q in AL.initial(seed)]
    return X, P, native_optimizer(P), AL.stream('env_labels_0913', seed), AL.stream('env_batch_0913', seed), subset


def task_draw(glabel, gbatch):
    y = torch.randint(K, (N,), generator=glabel)
    order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(-(-(STEPS * BATCH) // N))]).reshape(-1)[:STEPS * BATCH].reshape(STEPS, BATCH)
    return y, torch.nn.functional.one_hot(y, K).to(torch.float32), order


def backward(P, optim, xb, yb, reference_gradient_logits=None, capture=False):
    z = xb @ P[0].T + P[1]
    a = SourceELU.apply(z)
    logits = a @ P[2].T + P[3]
    if capture: logits.retain_grad()
    loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
    optim.zero_grad(set_to_none=True); P[2].grad = None
    if reference_gradient_logits is None:
        loss.backward()
    else:
        # Inject exact native CE logits gradient, avoiding softmax/CE rounding differences.
        logits.backward(reference_gradient_logits)
    if capture:
        return {'logits': logits.detach().clone(), 'gradient_logits': logits.grad.detach().clone(),
                'z_batch': z.detach().clone(), 'gate_batch': gate(z.detach()), 'loss_batch': float(loss.detach())}
    return None


def native_step(P, optim, xb, yb, reference_gradient_logits=None, capture=False, replay_second=None, scale=1.0):
    captured = backward(P, optim, xb, yb, reference_gradient_logits, capture)
    old_hidden = [P[j].detach().clone() for j in (0, 1)] if replay_second is not None else None
    if capture:
        captured.update({f'gradient_{name}': p.grad.detach().clone() for name, p in zip(NAMES, P)})
    optim.step()
    if replay_second is not None:
        # Native Adam updates the moments first; replace hidden S with reference's
        # same-step *updated* raw S, restore the old parameter, and redo only its
        # addcdiv. W2 and b2 keep their native steps; hidden first moments are native.
        with torch.no_grad():
            for j in (0, 1):
                st = optim.state[P[j]]
                st['exp_avg_sq'].copy_(replay_second[j]).mul_(scale * scale)
                t = float(st['step'])
                denom = st['exp_avg_sq'].sqrt().div_(math.sqrt(1 - BETAS[1] ** t)).add_(optim.param_groups[0]['eps'])
                P[j].copy_(old_hidden[j]).addcdiv_(st['exp_avg'], denom, value=-LR / (1 - BETAS[0] ** t))
    return captured


@torch.no_grad()
def diagnostics(P, X64, Y64):
    W1, b1, W2, b2 = [q.detach().double() for q in P]
    z = X64 @ W1.T + b1
    ordered, indices = z.sort(0, descending=True)
    body, body_index = z.median(0)
    s = z.std(0)
    assert bool(torch.all(torch.isfinite(z))) and bool(torch.all(s > 0))
    logits = elu(z) @ W2.T + b2
    unit = {'top': ordered[0], 'body': body, 's': s, 'q99': ordered[11], 'q95': ordered[59], 'q90': ordered[119],
            'k': (z > 0).sum(0).double(), 'T': (ordered[0] - body) / s, 'top_over_s': ordered[0] / s,
            'pplus': (z > 0).double().mean(0), 'top_input_id': indices[0], 'body_input_id': body_index,
            'wnorm': W1.norm(dim=1), 'vnorm': W2.norm(dim=0)}
    ce = float((logits.logsumexp(-1) - (logits * Y64).sum(-1)).mean())
    acc = float((logits.argmax(-1) == Y64.argmax(-1)).double().mean())
    return {k: v.numpy().copy() for k, v in unit.items()}, ce, acc


def snapshot(P, optim):
    out = {}
    for name, p in zip(NAMES, P):
        st = optim.state[p]
        out[name] = p.detach().numpy().copy()
        out[f'm_{name}'] = st['exp_avg'].numpy().copy()
        out[f'q_{name}'] = st['exp_avg_sq'].numpy().copy()
        out[f'step_{name}'] = np.asarray(float(st['step']))
    out.update(lr=np.float64(LR), eps=np.float64(optim.param_groups[0]['eps']), betas=np.array(BETAS),
               output_bias_eps=np.float64(optim.param_groups[2]['eps']), readout_lr=np.float64(optim.param_groups[1]['lr']))
    return out


def clone_arm(P, optim, mode, scale):
    params = [q.detach().clone().requires_grad_(True) for q in P]
    opt = native_optimizer(params)
    opt.load_state_dict(copy.deepcopy(optim.state_dict()))
    with torch.no_grad():
        params[2].mul_(scale)
        if mode in ('B', 'C', 'D', 'E'):
            for j in (0, 1):
                opt.state[params[j]]['exp_avg'].mul_(scale)
                opt.state[params[j]]['exp_avg_sq'].mul_(scale * scale)
    opt.param_groups[1]['lr'] = 0.0
    if mode in ('C', 'D', 'E'): opt.param_groups[0]['eps'] = scale * EPS
    tag = f'{mode}_c{str(scale).replace(".", "p")}'
    return {'P': params, 'optim': opt, 'mode': mode, 'scale': scale, 'tag': tag}


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pilot(out):
    X, P, opt, gl, gb, subset = initial(0)
    y, Y, order = task_draw(gl, gb)
    xb, yb = X[order[0]], Y[order[0]]
    cap = backward(P, opt, xb, yb, capture=True)
    with torch.no_grad():
        z = xb @ P[0].T + P[1]; a = elu(z); lg = a @ P[2].T + P[3]
        glogit = (torch.exp(lg - lg.logsumexp(-1, keepdim=True)) - yb) / BATCH
        gz = (glogit @ P[2]) * gate(z)
        expected_g = [gz.T @ xb, gz.sum(0), glogit.T @ a, glogit.sum(0)]
        gradient_errors = {name: float((p.grad - g).abs().max()) for name, p, g in zip(NAMES, P, expected_g)}
        before = [p.detach().clone() for p in P]
        expected = []
        for value, g in zip(before, expected_g):
            m = torch.zeros_like(g).lerp_(g, .1)
            q = torch.zeros_like(g).addcmul_(g, g, value=.001)
            denom = q.sqrt().div_(math.sqrt(1 - .999)).add_(EPS)
            expected.append(value.addcdiv(m, denom, value=-LR / (1 - .9)))
    opt.step()
    parameter_errors = {name: float((p.detach() - q).abs().max()) for name, p, q in zip(NAMES, P, expected)}
    # D at c=1 must reproduce a native step from the same warm state, bit for bit.
    ref = clone_arm(P, opt, 'REF', 1.0); dr = clone_arm(P, opt, 'D', 1.0)
    e1 = clone_arm(P, opt, 'E', 1.0)
    capref = native_step(ref['P'], ref['optim'], X[order[1]], Y[order[1]], capture=True)
    sq = [ref['optim'].state[ref['P'][j]]['exp_avg_sq'].detach().clone() for j in (0, 1)]
    native_step(dr['P'], dr['optim'], X[order[1]], Y[order[1]], replay_second=sq, scale=1.0)
    native_step(e1['P'], e1['optim'], X[order[1]], Y[order[1]], reference_gradient_logits=capref['gradient_logits'])
    replay_c1_errors = {mode: {name: float((arm['P'][j] - ref['P'][j]).detach().abs().max()) for j, name in enumerate(NAMES)} for mode, arm in [('D', dr), ('E', e1)]}
    started = time.perf_counter()
    for it in range(2000): native_step(P, opt, X[order[it]], Y[order[it]])
    seconds = time.perf_counter() - started
    checks = {'seed': 0, 'pilot_updates': 2000, 'seconds': seconds, 'updates_per_second': 2000 / seconds,
              'estimated_two_seed_native_seconds': seconds / 2000 * (2 * 50 * STEPS + 2 * 11 * 3 * STEPS),
              'gradient_max_abs_error': gradient_errors, 'first_parameter_max_abs_error': parameter_errors,
              'replay_c1_parameter_max_abs_error': replay_c1_errors,
              'torch': torch.__version__, 'threads': torch.get_num_threads(), 'source_sha256': RUNNER_SHA}
    assert max(gradient_errors.values()) < 2e-6
    assert max(parameter_errors.values()) < 2e-7
    assert max(max(v.values()) for v in replay_c1_errors.values()) < 2e-7
    (out / 'real_pilot.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks), flush=True)


def reconstruct(seed, out):
    started = time.perf_counter()
    X, P, opt, gl, gb, subset = initial(seed); X64 = X.double()
    shape = {}; end_ce = None
    for task in range(1, 51):
        y, Y, order = task_draw(gl, gb); Y64 = Y.double()
        if task == 50:
            unit, _, _ = diagnostics(P, X64, Y64)
            shape.update({f'sw_{k}': unit[k].astype(np.float32) for k in ('top', 'q99', 'q95', 'q90', 'body', 'k', 's')})
        for it in range(STEPS):
            native_step(P, opt, X[order[it]], Y[order[it]])
            if task == 50 and it + 1 == 500:
                unit, _, _ = diagnostics(P, X64, Y64)
                shape.update({f'mid_{k}': unit[k].astype(np.float32) for k in ('top', 'q99', 'q95', 'q90', 'body', 'k', 's')})
        if task % 10 == 0: print(json.dumps({'event': 'pretrain', 'seed': seed, 'task': task, 'seconds': time.perf_counter() - started}), flush=True)
    unit, end_ce, end_acc = diagnostics(P, X64, Y64)
    shape.update({f'end_{k}': unit[k].astype(np.float32) for k in ('top', 'q99', 'q95', 'q90', 'body', 'k', 's')})
    archived_path = SOURCE_ROOT / 'vfreeze' / 'runs_vf' / f'EEQ_s{seed}.npz'
    with np.load(archived_path) as archived:
        comparisons = {k: {'bit_equal': bool(np.array_equal(value, archived[k][49])), 'max_abs_error': float(np.max(abs(value.astype(float) - archived[k][49])))} for k, value in shape.items()}
        comparisons['ce_end'] = {'bit_equal': bool(end_ce == float(archived['ce_end'][49])), 'abs_error': abs(end_ce - float(archived['ce_end'][49])), 'new': end_ce, 'archived': float(archived['ce_end'][49])}
        comparisons['acc_end'] = {'new': end_acc, 'archived': float(archived['acc_end'][49])}
    label_state = gl.get_state().numpy().copy(); batch_state = gb.get_state().numpy().copy()
    plans = {task: task_draw(gl, gb) for task in (51, 52, 53)}
    state = snapshot(P, opt)
    state.update(X=X.numpy(), subset=subset, label_state=label_state, batch_state=batch_state,
                 y_t51=plans[51][0].numpy(), order_t51=plans[51][2].numpy(), seed=np.int64(seed))
    np.savez_compressed(out / f'real_s{seed}_t50.npz', **state)
    np.savez_compressed(out / f'real_s{seed}_task_plan.npz', **{f'{name}_t{task}': (trip[0] if name == 'y' else trip[2]).numpy() for task, trip in plans.items() for name in ('y', 'order')})
    np.savez_compressed(out / f'real_s{seed}_source_shapes.npz', **shape)
    comparisons.update(seed=seed, seconds=time.perf_counter() - started, source_npz_sha256=sha(archived_path), source_sha256=RUNNER_SHA,
                       strict_shape_replay=all(v['bit_equal'] for k, v in comparisons.items() if k != 'acc_end'))
    (out / f'real_s{seed}_source_comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    print(json.dumps({'event': 'checkpoint_ready', 'seed': seed, 'path': str(out / f'real_s{seed}_t50.npz'), 'strict_source_replay': comparisons['strict_shape_replay'], 'seconds': time.perf_counter() - started}), flush=True)
    return X, P, opt, plans


def intervene(seed, X, P, opt, plans, out):
    started = time.perf_counter(); X64 = X.double()
    ref = clone_arm(P, opt, 'REF', 1.0)
    arms = [ref] + [clone_arm(P, opt, mode, scale) for scale in (.1, 10.) for mode in ('A', 'B', 'C', 'D', 'E')]
    frozen_readouts = {a['tag']: a['P'][2].detach().clone() for a in arms}
    unit_rows, summary_rows, contrasts, null_rows = [], [], [], []
    all_diag = {}
    archive_checks = []
    archived_runs = {}
    for tag, source in ((ref['tag'], 'EEQ'), ('A_c0p1', 'VF0.1'), ('A_c10p0', 'VF10')):
        with np.load(SOURCE_ROOT / 'vfreeze' / 'runs_vf' / f'{source}_s{seed}.npz') as saved:
            archived_runs[tag] = {k: saved[k] for k in saved.files if k.startswith(('sw_', 'mid_', 'end_', 'ce_end'))}
    initial_states = {'arm_names': np.array([a['tag'] for a in arms]), 'gains': np.array([a['scale'] for a in arms]), 'policies': np.array([a['mode'] for a in arms])}
    for arm in arms:
        initial_states.update({f'{arm["tag"]}__{k}': v for k, v in snapshot(arm['P'], arm['optim']).items()})
    np.savez_compressed(out / f'real_s{seed}_t51_initial_states.npz', **initial_states)

    def record(task, step, Y64):
        byarm = {}
        for arm in arms:
            tag = arm['tag']; unit, ce, acc = diagnostics(arm['P'], X64, Y64); byarm[tag] = unit
            if step == 0: arm['switch_alive'] = unit['k'] > 0
            alive = arm['switch_alive']
            for i in range(H):
                unit_rows.append({'seed': seed, 'task': task, 'step': step, 'arm': tag, 'policy': arm['mode'], 'gain': arm['scale'], 'unit': i,
                                  **{k: int(v[i]) if k.endswith('_id') else float(v[i]) for k, v in unit.items()}})
            summary_rows.append({'seed': seed, 'task': task, 'step': step, 'arm': tag, 'policy': arm['mode'], 'gain': arm['scale'],
                                 'median_T_all': float(np.median(unit['T'])), 'median_raw_top_all': float(np.median(unit['top'])),
                                 'median_s_all': float(np.median(unit['s'])), 'mean_pplus_all': float(np.mean(unit['pplus'])),
                                 'median_T_switch_alive': float(np.median(unit['T'][alive])) if alive.any() else float('nan'),
                                 'ce': ce, 'accuracy': acc, 'switch_alive_fraction': float(alive.mean())})
            all_diag[tag] = unit
            if tag in archived_runs and step in (0, 500, 4000):
                phase = {0: 'sw', 500: 'mid', 4000: 'end'}[step]
                for key in ('top', 'q99', 'q95', 'q90', 'body', 'k', 's'):
                    old = archived_runs[tag][f'{phase}_{key}'][task - 1]
                    current = unit[key].astype(np.float32)
                    archive_checks.append({'seed': seed, 'arm': tag, 'task': task, 'step': step, 'metric': key,
                                           'bit_equal': bool(np.array_equal(current, old)), 'max_abs_error': float(np.max(abs(current.astype(float) - old)))})
        for mode in ('A', 'B', 'C', 'D', 'E'):
            small = byarm[f'{mode}_c0p1']; large = byarm[f'{mode}_c10p0']
            contrasts.append({'seed': seed, 'task': task, 'step': step, 'policy': mode,
                              'paired_median_T_small_minus_large': float(np.median(small['T'] - large['T'])),
                              'paired_mean_T_small_minus_large': float(np.mean(small['T'] - large['T'])),
                              'fraction_T_small_greater': float(np.mean(small['T'] > large['T'])),
                              'paired_median_raw_top_small_minus_large': float(np.median(small['top'] - large['top']))})
        for arm in arms:
            if arm['mode'] != 'E': continue
            tag = arm['tag']
            null_rows.append({'seed': seed, 'task': task, 'step': step, 'arm': tag,
                              'max_abs_hidden_parameter_difference': max(float((arm['P'][j] - ref['P'][j]).detach().abs().max()) for j in (0, 1)),
                              'max_abs_output_bias_difference': float((arm['P'][3] - ref['P'][3]).detach().abs().max()),
                              'max_abs_T_difference': float(np.max(abs(byarm[tag]['T'] - byarm[ref['tag']]['T']))),
                              'max_abs_raw_top_difference': float(np.max(abs(byarm[tag]['top'] - byarm[ref['tag']]['top'])))})
            assert null_rows[-1]['max_abs_output_bias_difference'] == 0.0

    one_step = {'arm_names': np.array([a['tag'] for a in arms]), 'gains': np.array([a['scale'] for a in arms]), 'policies': np.array([a['mode'] for a in arms])}
    for task in (51, 52, 53):
        y, Y, order = plans[task]; Y64 = Y.double()
        if task == 51:
            record(task, 0, Y64)
        else:
            for arm in arms: arm['switch_alive'] = all_diag[arm['tag']]['k'] > 0
        for it in range(STEPS):
            xb, yb = X[order[it]], Y[order[it]]
            refcap = native_step(ref['P'], ref['optim'], xb, yb, capture=True)
            ref_second = [ref['optim'].state[ref['P'][j]]['exp_avg_sq'].detach() for j in (0, 1)]
            capture = task == 51 and it == 0
            if capture:
                one_step.update({f'{ref["tag"]}__{k}': v.detach().numpy() if isinstance(v, torch.Tensor) else np.asarray(v) for k, v in refcap.items()})
                one_step.update({f'{ref["tag"]}__{k}': v for k, v in snapshot(ref['P'], ref['optim']).items()})
            for arm in arms[1:]:
                cap = native_step(arm['P'], arm['optim'], xb, yb,
                                  reference_gradient_logits=refcap['gradient_logits'] if arm['mode'] == 'E' else None,
                                  capture=capture, replay_second=ref_second if arm['mode'] == 'D' else None, scale=arm['scale'])
                if capture:
                    one_step.update({f'{arm["tag"]}__{k}': v.detach().numpy() if isinstance(v, torch.Tensor) else np.asarray(v) for k, v in cap.items()})
                    one_step.update({f'{arm["tag"]}__{k}': v for k, v in snapshot(arm['P'], arm['optim']).items()})
            step = it + 1
            if capture:
                np.savez_compressed(out / f'real_s{seed}_t51_step1.npz', **one_step)
                print(json.dumps({'event': 'first_step_ready', 'seed': seed, 'path': str(out / f'real_s{seed}_t51_step1.npz')}), flush=True)
            if (task == 51 and step in CHECKPOINTS) or step == STEPS:
                record(task, step, Y64)
        if task in (51, 53):
            full = {'arm_names': np.array([a['tag'] for a in arms]), 'gains': np.array([a['scale'] for a in arms]), 'policies': np.array([a['mode'] for a in arms])}
            for arm in arms: full.update({f'{arm["tag"]}__{k}': v for k, v in snapshot(arm['P'], arm['optim']).items()})
            np.savez_compressed(out / f'real_s{seed}_t{task}_states.npz', **full)
        csv_write(out / f'real_s{seed}_unit_diagnostics.csv', unit_rows)
        csv_write(out / f'real_s{seed}_summary.csv', summary_rows)
        csv_write(out / f'real_s{seed}_contrasts.csv', contrasts)
        csv_write(out / f'real_s{seed}_null_checks.csv', null_rows)
        csv_write(out / f'real_s{seed}_archive_checks.csv', archive_checks)
        print(json.dumps({'event': 'intervention_task_done', 'seed': seed, 'task': task, 'seconds': time.perf_counter() - started,
                          'contrasts': [r for r in contrasts if r['task'] == task and r['step'] == STEPS]}), flush=True)
    frozen = {arm['tag']: bool(torch.equal(arm['P'][2], frozen_readouts[arm['tag']])) for arm in arms}
    assert all(frozen.values())
    meta = {'seed': seed, 'seconds': time.perf_counter() - started, 'readout_values_unchanged': frozen,
            'native_reference_gradient_replay': True, 'second_replay_uses_updated_raw_reference_state': True,
            'source_sha256': RUNNER_SHA, 'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
            'data_sha256': sha(DATA), 'source_helper_sha256': sha(SOURCE_ROOT / 'alpha_ladder_1layer.py'),
            'registration_commit': '0fde22b', 'training_dtype': 'float32', 'diagnostic_dtype': 'float64', 'threads': torch.get_num_threads()}
    (out / f'real_s{seed}_provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'event': 'seed_complete', 'seed': seed, 'intervention_seconds': meta['seconds']}), flush=True)


def summarize(out):
    summaries = []
    for seed in (0, 1):
        with (out / f'real_s{seed}_contrasts.csv').open() as f: rows = list(csv.DictReader(f))
        for task in (51, 52, 53):
            rows_t = {r['policy']: float(r['paired_median_T_small_minus_large']) for r in rows if int(r['task']) == task and int(r['step']) == 4000}
            summaries.append({'seed': seed, 'task': task, **rows_t,
                              'B_over_A_if_A_gt_005': rows_t['B'] / rows_t['A'] if rows_t['A'] > .05 else '',
                              'D_over_C_if_C_gt_005': rows_t['D'] / rows_t['C'] if rows_t['C'] > .05 else ''})
    csv_write(out / 'real_paired_summary.csv', summaries)
    lines = ['# 実系のtask50再構成とtask51–53媒介経路介入', '',
             '登録commit: 0fde22b。各seedでtask50まで再構成し、同一の状態・入力・ラベル・バッチから分岐した。各seedを個別表示し、unit/taskを独立反復として検定しない。', '',
             'A=元のmoment保持、B=hidden momentsのc/c²整合化、C=Bにhidden εのc倍を追加、D=Cのhidden二次momentだけc²倍した基準からreplay、E=Cの残差を基準からreplay。W2は値を固定。基準c=1は共通。', '',
             '## 主対比：median_unit(T[c=.1]−T[c=10])', '',
             'T=(max−lower_median)/sample_sd。同一100unitの対応差を取ってから中央値。', '',
             '| seed | task | A | B | C | D | E | B/A (A>.05) | D/C (C>.05) |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in summaries:
        lines.append('| ' + ' | '.join(str(r[k]) if k in ('seed', 'task') or r[k] == '' else f'{r[k]:.7g}' for k in ('seed', 'task', 'A', 'B', 'C', 'D', 'E', 'B_over_A_if_A_gt_005', 'D_over_C_if_C_gt_005')) + ' |')
    lines.extend(['', '## Source再現と厳密nullの数値確認', ''])
    for seed in (0, 1):
        source = json.loads((out / f'real_s{seed}_source_comparison.json').read_text())
        with (out / f'real_s{seed}_null_checks.csv').open() as f: null = list(csv.DictReader(f))
        with (out / f'real_s{seed}_archive_checks.csv').open() as f: archived = list(csv.DictReader(f))
        lines.append(f'- seed{seed}: task50の21形状統計＋CEのbit再現={source["strict_shape_replay"]}。postのREF/Aで原保存形状とbit一致={sum(r["bit_equal"] == "True" for r in archived)}/{len(archived)}項。')
        lines.append(f'  Eの最大差（全診断checkpoint）: hidden parameter {max(float(r["max_abs_hidden_parameter_difference"]) for r in null):.6g}, T {max(float(r["max_abs_T_difference"]) for r in null):.6g}, output bias {max(float(r["max_abs_output_bias_difference"]) for r in null):.6g}。')
    lines.extend(['', '## 解釈上の限定', '',
                  '- 自然腕Aがこの短期窓で対象の方向を示すかを最初に確認する。長期task151–200の効果と同一量とはみなさない。',
                  '- B−Aは持ち越しmomentsの振幅不整合、C−Bはε変更、D−Cはv別の二次momentフィードバックに対する介入。これらを足して寄与率にしない。',
                  '- Dが効果を減らしても、過去の大きな残差の保持と、バッチ分散によるRMS増大をそれだけで分けることはできない。',
                  '- Eは通常の自分自身の損失最小化ではない。理論上hidden軌道が同じになる対照で、長期float32差があれば同一状態の一歩の検算と区別する。output biasはexact replayのためbit一致をassertした。',
                  '- グラフの符号が同じという再現だけで機構同定とは呼ばない。第一歩の状態からの予測と媒介経路を固定した対照をあわせて解釈する。', '',
                  'full parameters/moments/counters: real_s*_t50.npz、real_s*_t51_initial_states.npz、real_s*_t51_states.npz、real_s*_t53_states.npz。固定入力Xはt50に一度だけ保存。初回native勾配・logitsと更新後状態はreal_s*_t51_step1.npz。'])
    (out / 'real_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true'); parser.add_argument('--seed', type=int, choices=(0, 1))
    parser.add_argument('--pretrain-only', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('results/logistic_v_height_0929'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_flush_denormal(True)
    if args.pilot: pilot(args.out)
    if args.summarize: summarize(args.out)
    if args.seed is not None:
        if args.resume:
            with np.load(args.out / f'real_s{args.seed}_t50.npz') as saved:
                X = torch.tensor(saved['X'], dtype=torch.float32)
                P = [torch.tensor(saved[name], dtype=torch.float32, requires_grad=True) for name in NAMES]
                opt = native_optimizer(P)
                for name, param in zip(NAMES, P):
                    opt.state[param] = {'step': torch.tensor(float(saved[f'step_{name}'])),
                                        'exp_avg': torch.tensor(saved[f'm_{name}']), 'exp_avg_sq': torch.tensor(saved[f'q_{name}'])}
                plans = {}
                with np.load(args.out / f'real_s{args.seed}_task_plan.npz') as plan:
                    for task in (51, 52, 53):
                        y = torch.tensor(plan[f'y_t{task}'], dtype=torch.int64)
                        plans[task] = (y, torch.nn.functional.one_hot(y, K).to(torch.float32), torch.tensor(plan[f'order_t{task}'], dtype=torch.int64))
        else:
            X, P, opt, plans = reconstruct(args.seed, args.out)
        if not args.pretrain_only: intervene(args.seed, X, P, opt, plans, args.out)


if __name__ == '__main__':
    main()
