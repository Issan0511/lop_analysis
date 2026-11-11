#!/usr/bin/env python3
"""Preregistered loading_bridge_1111 A: frozen readout interventions.

Original body features are float32; all analytic readout operations are float64.
The per-example feature gradient is unnormalised by dataset size. The expected
MLP mean-preactivation push is mean_x [(x^T xbar+1) phi'(z) g_h]; stage1's raw
S_i is N times this value. Positive push is the gradient that lowers m under SGD.
No training or optimizer edits occur here. See specs/spec_loading_bridge_1111.md.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import stage1_ctrl_1111 as M

torch.set_num_threads(1)
DATA_ROOT = Path('/home/issan/Projects/obsidian-research-data')
MLP_ROOT = DATA_ROOT / 'stage1_ctrl_1111/results/stage1_ctrl_1111/raw'
VIT_ROOT = DATA_ROOT / 'vit_long_1024_2blk_rlfix/runs/SL'
ARMS = ('S', 'P0', 'P', 'RC', 'RR', 'H10', 'Q', 'T', 'G', 'TG', 'H0')
RTOL, ATOL = 1e-8, 1e-10


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def rng(role, seed, task=0, architecture='', activation=''):
    """Private, order-independent streams, shared by B if this helper is imported."""
    key = f'loading_bridge_1111|{role}|{architecture}|{activation}|{seed}|{task}'
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63) - 1)
    return torch.Generator().manual_seed(value)


def close(name, a, b, records, rtol=RTOL, atol=ATOL):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    diff = float((a - b).abs().max()) if a.numel() else 0.0
    scale = float(b.abs().max()) if b.numel() else 0.0
    # Elementwise tolerance, not a global-scale allowance for small entries.
    ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    records[name] = dict(passed=ok, max_abs=diff, reference_max_abs=scale, rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError(f'{name}: max_abs={diff:.6g}, reference_max={scale:.6g}')


def make_arms(W, b, h, seed, task, architecture, activation='', W_init=None):
    """Return (W,b,forward c,backward multiplier,permutation) in float64."""
    W, b, h = W.double(), b.double(), h.double()
    K, D = W.shape
    mu = h.mean(0)
    ans = {'S': (W, b, 1., 1., None)}
    gen = lambda role: rng(role, seed, task, architecture, activation)
    g = gen('P0')
    pi = torch.randperm(K, generator=g)
    while bool((pi == torch.arange(K)).any()):
        pi = torch.randperm(K, generator=g)
    ans['P0'] = (W[pi], b[pi], 1., 1., pi)
    g = gen('P')
    idx = torch.stack([torch.randperm(K, generator=g) for _ in range(D)], dim=1)
    wp = W.gather(0, idx)
    ans['P'] = (wp, b + (W - wp) @ mu, 1., 1., idx)
    wc = W - W.mean(0, keepdim=True)
    r = torch.randn(W.shape, generator=gen('RC'), dtype=torch.float64)
    r -= r.mean(0, keepdim=True)
    wr = W.mean(0, keepdim=True) + r / r.norm(dim=0, keepdim=True) * wc.norm(dim=0, keepdim=True)
    ans['RC'] = (wr, b + (W - wr) @ mu, 1., 1., None)
    r = torch.randn(W.shape, generator=gen('RR'), dtype=torch.float64)
    wr = r / r.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True)
    ans['RR'] = (wr, b, 1., 1., None)
    ans['H10'] = (W * .1, b * .1, 1., 1., None)
    if W_init is not None:
        wi = W_init.double()
        norms, targets = wc.norm(dim=0), (wi - wi.mean(0, keepdim=True)).norm(dim=0)
        scale = torch.where(norms > 0, targets / norms.clamp_min(1e-30), torch.zeros_like(norms))
        wq = W.mean(0, keepdim=True) + wc * scale
        ans['Q'] = (wq, b + (W - wq) @ mu, 1., 1., None)
    ans.update(T=(W, b, .1, 1., None), G=(W, b, 1., .1, None), TG=(W, b, .1, .1, None),
               H0=(torch.zeros_like(W), torch.zeros_like(b), 1., 1., None))
    return ans


def quantiles(name, tensor):
    a = torch.as_tensor(tensor).double().flatten()
    if not a.numel():
        return {f'{name}_{k}': float('nan') for k in ('min', 'q25', 'median', 'q75', 'max', 'mean')}
    q = torch.quantile(a, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist()
    return dict(zip([f'{name}_{k}' for k in ('min', 'q25', 'median', 'q75', 'max', 'mean')], q + [float(a.mean())]))


def analyse_arm(h, y, arm, raw, body=None):
    W, b, cf, cb, perm = raw
    h = h.double()
    z = cf * (h @ W.T + b)
    zc = z - z.mean(1, keepdim=True)
    logp = torch.log_softmax(z, dim=1)
    p = logp.exp()
    wc = W - W.mean(0, keepdim=True)
    gh = cb * (p @ wc)
    entropy = -(p * logp).sum(1)
    gn = gh.norm(dim=1)
    radial = (gh * h).sum(1)
    hc = h - h.mean(0, keepdim=True)
    correct_margin = zc.gather(1, y[:, None]).squeeze(1)
    row = dict(arm=arm, forward_coefficient=cf, backward_multiplier=cb,
               head_norm=float(W.norm()), head_centered_norm=float(wc.norm()),
               bias_norm=float(b.norm()), feature_mean_norm=float(h.mean(0).norm()),
               feature_centered_rms=float(hc.square().mean().sqrt()),
               feature_rms=float(h.square().mean().sqrt()),
               centered_logit_rms=float(zc.square().mean().sqrt()),
               sample_mean_logit_rms=float(zc.mean(0).square().mean().sqrt()),
               old_accuracy=float((z.argmax(1) == y).double().mean()),
               old_centered_correct_margin=float(correct_margin.mean()),
               entropy=float(entropy.mean()), max_probability=float(p.max(1).values.mean()),
               expected_uniform_ce=float((z.logsumexp(1) - z.mean(1)).mean()),
               feature_gradient_norm_mean=float(gn.mean()), feature_gradient_rms=float(gh.square().mean().sqrt()),
               feature_gradient_mean_norm=float(gh.mean(0).norm()), radial_product_mean=float(radial.mean()),
               centered_radial_product_mean=float((gh * hc).sum(1).mean()))
    for label, a in [('row_norm', W.norm(dim=1)), ('column_norm', W.norm(dim=0)),
                     ('centered_column_norm', wc.norm(dim=0)), ('gradient_norm', gn)]:
        row.update(quantiles(label, a))
    if arm == 'P0':
        y_relabel = torch.argsort(perm)[y]
        row['old_accuracy_relabelled'] = float((z.argmax(1) == y_relabel).double().mean())
        row['old_centered_correct_margin_relabelled'] = float(zc.gather(1, y_relabel[:, None]).mean())
    else:
        row['old_accuracy_relabelled'] = row['old_accuracy']
        row['old_centered_correct_margin_relabelled'] = row['old_centered_correct_margin']
    arrays = dict(logits=z.numpy(), probability=p.numpy(), gradient_norm=gn.numpy(), radial_product=radial.numpy(),
                  row_norm=W.norm(dim=1).numpy(), column_norm=W.norm(dim=0).numpy(),
                  centered_column_norm=wc.norm(dim=0).numpy(), head_weight=W.numpy(), head_bias=b.numpy())
    if body is not None:
        Z, kernel, act = body
        d = M.dphi64(Z.double(), act)
        push = (kernel[:, None] * d * gh).mean(0)
        active = (Z > 0).any(0)
        row.update(push_signed_mean=float(push.mean()), push_abs_mean=float(push.abs().mean()),
                   push_l2=float(push.norm()), push_stage1_sum_signed_mean=float(push.mean() * len(h)),
                   active_units=int(active.sum()),
                   active_push_signed_mean=float(push[active].mean()) if active.any() else float('nan'),
                   active_push_abs_mean=float(push[active].abs().mean()) if active.any() else float('nan'),
                   active_push_l2=float(push[active].norm()))
        arrays['expected_mean_preactivation_push'] = push.numpy()
        arrays['stage1_S'] = (len(h) * push).numpy()
    return row, arrays, dict(p=p, gradient=gh, ce=(z.logsumexp(1) - z.mean(1)).mean(), z=z)


def check_invariants(h, y, arms, probes, W_init=None):
    checks = {}
    W, b = arms['S'][:2]
    ws, wp = W.double(), arms['P'][0]
    close('P_column_multisets', wp.sort(dim=0).values, ws.sort(dim=0).values, checks, 0., 0.)
    wrc = arms['RC'][0]
    close('RC_column_means', wrc.mean(0), ws.mean(0), checks)
    close('RC_centered_column_norms', (wrc - wrc.mean(0)).norm(dim=0), (ws - ws.mean(0)).norm(dim=0), checks)
    close('RR_row_norms', arms['RR'][0].norm(dim=1), ws.norm(dim=1), checks)
    for a in ('P', 'RC', 'Q'):
        if a in arms:
            close(f'{a}_mean_logits', (h.double() @ arms[a][0].T + arms[a][1]).mean(0),
                  (h.double() @ ws.T + b).mean(0), checks)
    if 'Q' in arms:
        wi = W_init.double()
        n = (ws - ws.mean(0)).norm(dim=0)
        target = (wi - wi.mean(0)).norm(dim=0)
        target = torch.where(n > 0, target, torch.zeros_like(target))
        close('Q_initial_centered_norms', (arms['Q'][0] - arms['Q'][0].mean(0)).norm(dim=0), target, checks)
    close('P0_expected_ce', probes['P0']['ce'], probes['S']['ce'], checks)
    close('P0_feature_gradient', probes['P0']['gradient'], probes['S']['gradient'], checks)
    close('P0_logits', probes['P0']['z'], probes['S']['z'][:, arms['P0'][4]], checks)
    close('T_TG_probabilities', probes['T']['p'], probes['TG']['p'], checks)
    close('T_TG_gradient_x10', probes['T']['gradient'], 10 * probes['TG']['gradient'], checks)
    close('S_G_probabilities', probes['S']['p'], probes['G']['p'], checks)
    close('S_G_gradient_x10', probes['S']['gradient'], 10 * probes['G']['gradient'], checks)
    close('H10_T_probabilities', probes['H10']['p'], probes['T']['p'], checks)
    close('H10_T_gradient_x01', probes['H10']['gradient'], .1 * probes['T']['gradient'], checks)
    close('H10_TG_gradient', probes['H10']['gradient'], probes['TG']['gradient'], checks)
    close('H0_zero_gradient', probes['H0']['gradient'], torch.zeros_like(probes['H0']['gradient']), checks, 0., 0.)
    return checks


def run_cell(h, W, b, y, seed, task, architecture, activation='', W_init=None, body=None):
    arms = make_arms(W, b, h, seed, task, architecture, activation, W_init)
    rows, raw, probes = [], {}, {}
    for arm in ARMS:
        if arm not in arms:
            continue
        row, arrays, probes[arm] = analyse_arm(h, y, arm, arms[arm], body)
        row.update(architecture=architecture, activation=activation, seed=seed, task=task, n=len(h))
        rows.append(row)
        raw.update({f'{arm}__{k}': v for k, v in arrays.items()})
        if arms[arm][4] is not None:
            raw[f'{arm}__permutation'] = arms[arm][4].numpy()
    checks = check_invariants(h, y, arms, probes, W_init)
    raw.update(features=h.float().numpy(), labels=y.numpy())
    if body is not None:
        z = body[0].double()
        m, s = z.mean(0), z.std(0, unbiased=False)
        r = torch.where(s > 0, m / s, torch.full_like(s, float('nan')))
        d = M.dphi64(z, activation).abs()
        stats = dict(m=m.numpy(), s=s.numpy(), r=r.numpy(), opening_fraction=(z > 0).double().mean(0).numpy(),
                     mean_abs_derivative=d.mean(0).numpy(), active=(z > 0).any(0).numpy())
        raw.update({f'unit__{k}': v for k, v in stats.items()})
        raw['input_mean_kernel'] = body[1].numpy()
        for row in rows:
            for k in ('m', 's', 'r', 'opening_fraction', 'mean_abs_derivative'):
                row.update(quantiles(k, stats[k]))
    return rows, raw, checks


def synthetic_checks():
    g = rng('synthetic', 927)
    N, D, K = 31, 7, 5
    h = torch.randn(N, D, generator=g, dtype=torch.float64)
    W = torch.randn(K, D, generator=g, dtype=torch.float64)
    b = torch.randn(K, generator=g, dtype=torch.float64)
    y = torch.randint(K, (N,), generator=g)
    wi = torch.randn(K, D, generator=g, dtype=torch.float64) * .1
    rows, raw, checks = run_cell(h, W, b, y, 927, 10, 'synthetic', W_init=wi)
    hh = h.clone().requires_grad_()
    z = hh @ W.T + b
    objective = (z.logsumexp(1) - z.mean(1)).mean()
    grad, = torch.autograd.grad(objective, hh)
    _, _, prob = analyse_arm(h, y, 'S', (W, b, 1., 1., None))
    close('analytic_vs_autograd_feature_gradient', N * grad, prob['gradient'], checks)
    for arm, cf, cb in (('T', .1, 1.), ('G', 1., .1), ('TG', .1, .1)):
        hh = h.clone().requires_grad_()
        zz = hh @ W.T + b
        # Straight-through value cf*z with a separately chosen derivative cb.
        transformed = cf * zz.detach() + cb * (zz - zz.detach())
        obj = (transformed.logsumexp(1) - transformed.mean(1)).mean()
        actual, = torch.autograd.grad(obj, hh)
        _, _, expected = analyse_arm(h, y, arm, (W, b, cf, cb, None))
        close(f'{arm}_analytic_vs_custom_backward', N * actual, expected['gradient'], checks)
    # Exercise every MLP derivative (including the sign-changing GELU/SILU),
    # with exact float64 finite-dataset mean-preactivation chain rule.
    x = torch.randn(N, 11, generator=g, dtype=torch.float64)
    w1 = torch.randn(D, 11, generator=g, dtype=torch.float64).requires_grad_()
    b1 = torch.randn(D, generator=g, dtype=torch.float64).requires_grad_()
    for act in ('R', 'LR', 'ELU', 'SILU', 'GELU'):
        z1 = x @ w1.T + b1
        a = M.activ(z1, act)
        zz = a @ W.T + b
        obj = (zz.logsumexp(1) - zz.mean(1)).mean()
        gw, gb = torch.autograd.grad(obj, (w1, b1))
        kernel = x @ x.mean(0) + 1
        _, arr, _ = analyse_arm(a.detach(), y, 'S', (W, b, 1., 1., None), (z1.detach(), kernel, act))
        close(f'{act}_mean_preactivation_push_vs_autograd', arr['expected_mean_preactivation_push'], gw @ x.mean(0) + gb, checks)
    return checks


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    leading = [k for k in ('architecture', 'activation', 'task', 'seed', 'arm') if k in fields]
    fields = leading + [k for k in fields if k not in leading]
    with path.open('w', newline='') as f:
        out = csv.DictWriter(f, fields)
        out.writeheader()
        out.writerows(rows)


def bootstrap(rows):
    """Paired bootstrap of ratio of means (primary) and mean seed ratios (auxiliary).

    Both numerator and denominator seed pairs are resampled together. The fixed
    spec denominator threshold is 1e-10. No unit is independently resampled.
    """
    metrics = ('feature_gradient_norm_mean', 'push_signed_mean', 'push_abs_mean', 'push_l2',
               'active_push_signed_mean', 'active_push_abs_mean', 'radial_product_mean',
               'centered_logit_rms', 'old_accuracy', 'old_centered_correct_margin')
    groups = sorted(set((r['architecture'], r['activation'], r['task']) for r in rows))
    output = []
    for arch, act, task in groups:
        cell = [r for r in rows if (r['architecture'], r['activation'], r['task']) == (arch, act, task)]
        by = {(r['seed'], r['arm']): r for r in cell}
        seeds = sorted({r['seed'] for r in cell})
        for arm in sorted({r['arm'] for r in cell} - {'S'}):
            for metric in metrics:
                valid = [s for s in seeds if metric in by[s, 'S'] and metric in by[s, arm]
                         and np.isfinite(by[s, 'S'][metric]) and np.isfinite(by[s, arm][metric])
                         and abs(by[s, 'S'][metric]) >= 1e-10]
                if not valid:
                    continue
                ref = np.array([by[s, 'S'][metric] for s in valid])
                num = np.array([by[s, arm][metric] for s in valid])
                gen = np.random.default_rng(int(rng(f'bootstrap|{arm}|{metric}', 0, task, arch, act).initial_seed()))
                ix = gen.integers(0, len(valid), (10000, len(valid)))
                den = ref[ix].mean(1)
                draws = num[ix].mean(1) / np.where(abs(den) >= 1e-10, den, np.nan)
                sample_means = (num / ref)[ix].mean(1)
                ratio = num.mean() / ref.mean() if abs(ref.mean()) >= 1e-10 else np.nan
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                ml, mh = np.percentile(sample_means, [2.5, 97.5])
                # A signed push that reverses sign is not a positive retained force.
                judgement = ('substantial_retention' if lo > .5 else 'majority_removal' if hi < .5 else 'unresolved')
                output.append(dict(architecture=arch, activation=act, task=task, arm=arm, metric=metric,
                                   n=len(valid), seeds=';'.join(map(str, valid)), excluded_seed_count=len(seeds)-len(valid),
                                   baseline_mean=float(ref.mean()), intervention_mean=float(num.mean()),
                                   ratio_of_means=float(ratio), ci_low=float(lo), ci_high=float(hi),
                                   mean_seed_ratio=float((num/ref).mean()), mean_seed_ratio_ci_low=float(ml),
                                   mean_seed_ratio_ci_high=float(mh), ratio_judgement=judgement,
                                   nonfinite_bootstrap_count=int((~np.isfinite(draws)).sum())))
    return output


def mlp_cell(act, seed, task):
    source = MLP_ROOT / f'states_adam_lr0.001_{act}_s{seed}.npz'
    with np.load(source) as saved:
        X, subset = M.load_X(seed)
        assert np.array_equal(subset, saved['subset'])
        P = [torch.from_numpy(saved[f'end{i}'][task - 1].copy()) for i in range(4)]
        wi = torch.from_numpy(saved['init2'].copy())
        y = torch.from_numpy(saved['y'][task - 1].copy()).long()
        init = M.initial(seed, (784, M.H, M.K))
        for j in range(4):
            assert np.array_equal(init[j].numpy(), saved[f'init{j}']), f'initial mismatch parameter{j}'
        labels = M.stream('env_labels_0913', seed)
        for _ in range(task):
            canonical_y = torch.randint(M.K, (M.N,), generator=labels)
        assert torch.equal(y, canonical_y)
    with torch.no_grad():
        z = X @ P[0].T + P[1]
        h = M.activ(z, act)
        kernel = X.double() @ X.double().mean(0) + 1
        rows, raw, checks = run_cell(h, P[2], P[3], y, seed, task, 'MLP', act, wi, (z, kernel, act))
        direct = h @ P[2].T + P[3]
        close('float32_head_reconstruction', raw['S__logits'], direct, checks, rtol=2e-5, atol=2e-5)
        # Original stage1's helper also recomputes the BODY in double precision.
        # Record its signed mean and per-unit S_i; A main is defined at float32 features.
        prior, conf, norm = M.expected_push(P, M.Ctx(X, act))
        raw['stage1_float64_body_S'] = prior[0]
        raw['subset_indices'] = subset
        for row in rows:
            row['stage1_float64_body_baseline_push_signed_mean'] = float(prior[0].mean() / len(X))
            row['stage1_float64_body_baseline_max_probability'] = conf
        metadata = dict(source=str(source), source_sha256=sha(source), dataset_input_sha256=array_sha(X.numpy()),
                        labels_sha256=array_sha(y.numpy()), subset_sha256=array_sha(subset),
                        saved_source_accuracy=float((direct.argmax(1) == y).float().mean()))
    return rows, raw, checks, metadata


def vit_cell(seed, task, device):
    from analysis.vit_long_1024 import common as C
    from analysis.vit_long_1024.posthoc_2blk.scale_counterfactual import cls_features
    C.set_determinism(1)
    path = VIT_ROOT / f'seed{seed}/ckpt/model_t{task:02d}.pt'
    model = C.build_arm_model('SL', seed, device, 0.0, 2, 8.0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    source = C.image_source('real')
    data, plan = C.task_dataset(source, seed, task, 10, 250, augment=False)
    indices = C.loss_classes('SL', seed, task, 10, 250)
    h, direct = [], []
    with torch.no_grad():
        for j in range(0, len(data), 128):
            xx = C.eval_tensor(data.images[j:j + 128], device)
            h.append(cls_features(model, xx).cpu())
            direct.append(model(xx)[:, indices].cpu())
    h, direct = torch.cat(h), torch.cat(direct)
    out = model.layers['out']
    W, b = out.weight[indices].detach().cpu(), out.bias[indices].detach().cpu()
    y = data.labels.cpu().long()
    rows, raw, checks = run_cell(h, W, b, y, seed, task, 'ViT', 'SL')
    close('float32_head_reconstruction', raw['S__logits'], direct, checks, rtol=2e-5, atol=2e-5)
    raw['loss_rows'] = np.asarray(indices)
    metadata = dict(source=str(path), source_sha256=sha(path), dataset_images_sha256=array_sha(data.images),
                    labels_sha256=array_sha(y.numpy()), plan=plan, loss_rows=indices,
                    dataset_root=str(source.root), image_paths=data.paths,
                    saved_source_accuracy=float((direct.argmax(1) == y).float().mean()),
                    model_head_rows=int(out.weight.shape[0]), analytic_loss_rows=int(W.shape[0]))
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return rows, raw, checks, metadata


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('selftest', 'mlp', 'vit', 'all', 'summarize'), default='selftest')
    parser.add_argument('--smoke', action='store_true', help='MLP R seed0 t1; ViT seed10252010 t100, isolated output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/loading_bridge_1111')
    parser.add_argument('--raw', type=Path, default=ROOT / 'raw/loading_bridge_1111/static')
    args = parser.parse_args()
    out = args.out / 'static_smoke' if args.smoke else args.out
    rawdir = args.raw / 'smoke' if args.smoke else args.raw
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks = synthetic_checks()
    if args.mode == 'selftest':
        (out / 'static_selftest.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps(dict(status='PASS', synthetic_checks=len(checks), output=str(out / 'static_selftest.json'))))
        return
    if args.mode == 'summarize':
        files = sorted(rawdir.glob('*__summary.json'))
        rows = [r for file in files for r in json.loads(file.read_text())]
        write_csv(out / 'static_summary.csv', rows)
        write_csv(out / 'static_ratios.csv', bootstrap(rows))
        print(json.dumps(dict(cells=len(files), rows=len(rows), output=str(out))))
        return
    rawdir.mkdir(parents=True, exist_ok=True)
    all_rows, cells = [], []
    jobs = []
    if args.mode in ('mlp', 'all'):
        jobs += [('MLP', act, seed, task) for act in (('R',) if args.smoke else ('R', 'LR', 'ELU', 'SILU', 'GELU'))
                 for seed in ((0,) if args.smoke else range(10)) for task in ((1,) if args.smoke else (1, 10, 30))]
    if args.mode in ('vit', 'all'):
        jobs += [('ViT', 'SL', seed, task) for seed in ((10252010,) if args.smoke else range(10252010, 10252015))
                 for task in ((100,) if args.smoke else (1, 100))]
    provenance = dict(command=sys.argv, cwd=str(Path.cwd()), git_hash=git('rev-parse', 'HEAD'),
                      working_tree_status=git('status', '--porcelain'), python=platform.python_version(),
                      torch=torch.__version__, numpy=np.__version__, device=args.device, smoke=args.smoke,
                      spec_sha256=sha(ROOT / 'specs/spec_loading_bridge_1111.md'),
                      script_sha256=sha(Path(__file__)), synthetic_checks=checks,
                      definitions=dict(feature_gradient='per-example E_uniform_y dCE/dh, without 1/N',
                                       expected_push='mean_x ((x dot mean_x(x)+1) phi_prime(z) feature_gradient)',
                                       push_stage1_S='N times expected_push',
                                       logit_centering='class centering per example',
                                       sample_mean_logit_rms='RMS across classes of sample-averaged class-centered logits',
                                       active='z>0 on at least one fixed input; auxiliary subset',
                                       bootstrap='10,000 paired seed resamples; primary ratio of means; mean ratios auxiliary'),
                      cells=cells)
    provpath = out / f'static_provenance_{args.mode}.json'
    for arch, act, seed, task in jobs:
        key = f'{arch}_{act}_s{seed}_t{task}'
        t = time.monotonic()
        try:
            rows, raw, sanity, source = mlp_cell(act, seed, task) if arch == 'MLP' else vit_cell(seed, task, args.device)
        except Exception as exc:
            provenance.update(failed_cell=key, error=repr(exc), elapsed_s=time.monotonic()-start)
            provpath.write_text(json.dumps(provenance, indent=2) + '\n')
            raise
        file = rawdir / f'{key}.npz'
        np.savez_compressed(file, **raw)
        (rawdir / f'{key}__summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        source.update(cell=key, raw_file=str(file), raw_sha256=sha(file), sanity=sanity, elapsed_s=time.monotonic()-t)
        cells.append(source)
        all_rows.extend(rows)
        provenance['elapsed_s'] = time.monotonic() - start
        provpath.write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'{key} PASS ({time.monotonic()-t:.2f}s)', flush=True)
    write_csv(out / f'static_summary_{args.mode}.csv', all_rows)
    write_csv(out / f'static_ratios_{args.mode}.csv', bootstrap(all_rows))
    print(json.dumps(dict(status='PASS', cells=len(cells), rows=len(all_rows), elapsed_s=time.monotonic()-start)))


if __name__ == '__main__':
    main()
