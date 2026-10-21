#!/usr/bin/env python3
"""Read-only checkpoint analysis of preregistered (A)/(B); never train or commit.

Run with the project venv, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE=1.
The legacy extraction/window implementation is imported, not copied.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CHARGPT_NLAYER'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/issan/Projects/claude/wt/chargpt_cl_1019')
sys.path.insert(0, str(SOURCE / 'analysis/chargpt_cl_1019_quicklook'))
import decomp_dmu as D
C, K = D.C, D.K
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
ROOT = SOURCE / 'results/chargpt_cl_1019/M2'
OUT = HERE / 'results/chargpt_l0_carrier_1021'
TASKS = list(range(1, 11)) + list(range(40, 51)) + list(range(90, 101))
END_TASKS = sorted(set(TASKS + [25, 39, 89]))
ARMS = ['C3', 'ReLUcap']
ATOL = 1e-10


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


@lru_cache(maxsize=160)
def probe(seed, task):
    chain = C.ChainM(seed, task, 0.01)
    return C.probes_M(chain, seed, task)[0]


def ckpt(d, name):
    p = d / 'ckpts' / f'{name}.pt'
    if not p.exists() and name.startswith('u100_task'):
        p = d / 'ckpts' / f'task{name[9:]}_u100.pt'
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_model(arm, d, name, seed):
    # The full saved state overwrites every parameter/buffer. Avoid redundant
    # task-1 calibration on each checkpoint load, using K's original builder.
    m = K._orig_build(arm, 'M', seed, 'cpu')
    m.load_state_dict(torch.load(ckpt(d, name), map_location='cpu', weights_only=True))
    return m.eval()


@torch.no_grad()
def snapshot(m, p):
    # Preserve the legacy float32 forward, float64 token moments / arithmetic.
    return [dict(W=x['W'], y=x['y'].mean(0), sd=x['z'].std(0))
            for x in D.ffn_io(m, p)]


def decompose(a, b):
    # Three bilinear contributions; rows = own, upstream, interaction.
    weights = torch.stack([b['W'] - a['W'], a['W'], b['W'] - a['W']])
    inputs = torch.stack([a['y'], b['y'] - a['y'], b['y'] - a['y']])
    raw = torch.einsum('kud,kd->ku', weights, inputs)
    delta = b['W'].mv(b['y']) - a['W'].mv(a['y'])
    residual = float((delta - raw.sum(0)).abs().max())
    assert torch.all(a['sd'] > 0), 'Zero sigma; normalized estimand undefined'
    normed = raw / a['sd']
    dn = delta / a['sd']
    normalized_residual = float((dn - normed.sum(0)).abs().max())
    assert max(residual, normalized_residual) <= ATOL
    return dict(dmu=dn.numpy(), own=normed[0].numpy(), up=normed[1].numpy(),
                cross=normed[2].numpy()), residual, normalized_residual


def summarize_decomp(u):
    sink = u['dmu'] < 0
    r = dict(n_units=len(sink), n_sinking=int(sink.sum()), sinking_fraction=float(sink.mean()))
    for suffix, mask in [('', np.ones_like(sink)), ('_sinking', sink)]:
        for k, v in u.items():
            r[k + '_mean' + suffix] = float(v[mask].mean()) if mask.any() else float('nan')
            r['neg_' + k + suffix] = float(np.maximum(-v[mask], 0).mean()) if mask.any() else float('nan')
        r['neg_up_cross' + suffix] = float(np.maximum(-(u['up'] + u['cross'])[mask], 0).mean()) if mask.any() else float('nan')
        denom = sum(r['neg_' + k + suffix] for k in ('own', 'up', 'cross'))
        r['up_share' + suffix] = r['neg_up' + suffix] / denom if denom > 0 else float('nan')
        legacy_denom = r['neg_own' + suffix] + r['neg_up_cross' + suffix]
        r['legacy_up_share' + suffix] = r['neg_up_cross' + suffix] / legacy_denom if legacy_denom > 0 else float('nan')
    r['dmu_median'] = float(np.median(u['dmu']))
    return r


def scale(a, b):
    dy = b['y'] - a['y']
    normw, normdy = a['W'].norm(dim=1), dy.norm()
    upabs = a['W'].mv(dy).abs()
    align = upabs / (normw * normdy) if normdy > 0 else torch.full_like(normw, float('nan'))
    return dict(up_abs_median=float(np.median(upabs)), w_norm_median=float(np.median(normw)),
                dy_norm=float(normdy), alignment_median=float(np.median(align)),
                lr_over_w_median=float(np.median(1e-3 / normw)))


@torch.no_grad()
def carrier(m, p):
    """Conditional LN decomposition with the *total* residual's denominator.

    y_source[n] = gamma * (x_source[n] - mean_feature(x_source[n]))
                  / sqrt(var_feature(total_x[n]) + eps).
    Project its token mean on total ybar. This is additive bookkeeping,
    not LN(source) and not the causal effect of removing that source.
    """
    h = m.wte(p)
    src = {'emb': h}
    for li, block in enumerate(m.h):
        at = block.sc * block.attn(block.n1(h))
        h = h + at
        src[f'attn{li}'] = at
        if li == 1:
            break
        f = block.mlp(block.n2(h))
        h = h + f
        src[f'ffn{li}'] = f
    x = h.reshape(-1, C.D).double()
    n2 = m.h[1].n2
    denom = (x.var(-1, correction=0, keepdim=True) + n2.eps).sqrt()
    y = (x - x.mean(-1, keepdim=True)) * n2.weight.double() / denom
    ybar, xbar = y.mean(0), x.mean(0)
    vectors = {}
    r = dict(ybar_norm=float(ybar.norm()), xbar_norm=float(xbar.norm()))
    for k, v in src.items():
        s = v.reshape(-1, C.D).double()
        ys = ((s - s.mean(-1, keepdim=True)) * n2.weight.double() / denom).mean(0)
        vectors[k] = ys
        r[k] = float(ys.dot(ybar) / ybar.square().sum())
        r[k + '_preln'] = float(s.mean(0).dot(xbar) / xbar.square().sum())
    # Float32 residual additions create tiny non-additive roundoff; retain it.
    closure = sum(vectors.values()) - ybar
    r['projection_sum'] = sum(r[k] for k in src)
    r['closure_relative'] = float(closure.norm() / ybar.norm())
    r['ln64_vs_forward_relative'] = float((ybar - n2(h).double().reshape(-1, C.D).mean(0)).norm() / ybar.norm())
    assert r['closure_relative'] < 1e-5
    return r


def preflight():
    start = time.monotonic()
    source_files = ['src/chargpt_cl_1019.py', 'src/chargpt_cl_1019_kune.py',
                    'analysis/chargpt_cl_1019_quicklook/decomp_dmu.py',
                    'analysis/chargpt_cl_1019_quicklook/judge_m2.py',
                    'analysis/chargpt_cl_1019_quicklook/carrier_source.py']
    git = lambda *a: subprocess.check_output(['git', '-C', str(SOURCE), *a], text=True).strip()
    checks = dict(status='running_preflight', source_head=git('rev-parse', 'HEAD'),
                  requested_source_head='a970f12c',
                  source_diff_since_requested=git('diff', '--name-only', 'a970f12c', '--', *source_files),
                  source_sha256={f: sha(SOURCE / f) for f in source_files},
                  spec_sha256=sha(HERE / 'specs/spec_chargpt_l0_carrier_1021.md'),
                  python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                  device='cpu', threads=torch.get_num_threads(),
                  definitions=dict(up_share='neg(up)/(neg(own)+neg(up)+neg(cross))',
                      legacy_up_share='neg(up+cross)/(neg(own)+neg(up+cross))',
                      cap='sqrt(V), not row norm or probe sigma',
                      precision='float32 model forward; float64 means, std(ddof=1), decomposition',
                      carrier='tokenwise centered sources / total residual LN denominator, gamma, token mean, projection'))
    dump_json(OUT / 'checks.json', checks)
    try:
        assert checks['source_diff_since_requested'] == ''
        expected = ['init'] + [f'task{t}' for t in END_TASKS] + [f'u100_task{t}' for t in TASKS]
        for arm in ARMS:
            for seed in range(10):
                d = ROOT / arm / f'seed{seed}'
                for name in expected:
                    ckpt(d, name)
                rows = list(csv.DictReader((d / 'per_task.csv').open()))
                assert len(rows) == 100 and all(int(r['n']) == 10000 for r in rows)
        checks['inventory'] = dict(pass_=True, runs=20, checkpoints_per_run=len(expected))
        # Legacy expected values use cross bundled into upstream and all units.
        d = SOURCE / 'results/chargpt_cl_1019/quicklook/M2/C3/seed1000'
        neg = np.zeros(2)
        nu = no = resid = comparison_error = 0.0
        for t in range(2, 11):
            p = probe(1000, t)
            ma = load_model('C3', d, f'task{t-1}', 1000)
            mb = load_model('C3', d, f'u100_task{t}', 1000)
            a, b = snapshot(ma, p), snapshot(mb, p)
            # Explicit import-based parity against the original window on t2.
            legacy = D.window('C3', d, f'task{t-1}', f'u100_task{t}', p, False) if t == 2 else None
            for li in range(2):
                u, rr, nr = decompose(a[li], b[li])
                resid = max(resid, rr, nr)
                neg[li] += np.maximum(-u['dmu'], 0).mean()
                if li == 1:
                    nu += np.maximum(-(u['up'] + u['cross']), 0).mean()
                    no += np.maximum(-u['own'], 0).mean()
                if legacy is not None:
                    comparison_error = max(comparison_error, *(float(np.max(np.abs(u[k] - legacy[li][0][k].numpy()))) for k in u))
        share, concentration = nu / (nu + no), neg[1] / neg.sum()
        checks['seed1000'] = dict(pass_=abs(share - .958) <= .0005 and abs(concentration - .937) <= .0005,
                                 upstream_share=share, concentration=concentration,
                                 identity_max=resid, imported_window_max_difference=comparison_error,
                                 tasks=list(range(2, 11)), denominator='all units, upstream includes cross')
        assert checks['seed1000']['pass_'] and comparison_error < ATOL
        checks['identity_preflight'] = dict(pass_=resid <= ATOL, max_abs=resid, tolerance=ATOL)
        # Cap is activated at task 2, anchored to task 1 end sqrt(V).
        d = ROOT / 'ReLUcap/seed0'
        ref = torch.load(ckpt(d, 'task1'), map_location='cpu', weights_only=True)
        cap_max = np.zeros(2)
        cap_match = 0.0
        ncheck = 0
        for name in expected:
            if name in ('init', 'u100_task1'):
                continue
            st = torch.load(ckpt(d, name), map_location='cpu', weights_only=True)
            for li in range(2):
                key = f'h.{li}.mlp.'
                ref_s = ref[key+'V'].sqrt().double()
                cap_max[li] = max(cap_max[li], float((st[key+'V'].double().sqrt() / ref_s).max()))
                if name != 'task1':
                    cap_match = max(cap_match, float((st[key+'cap'].double() - ref_s).abs().max()))
                    assert float(st[key+'cap_on']) == 1.0
            ncheck += 1
        checks['cap_seed0'] = dict(pass_=bool(np.all(cap_max <= 1 + 5e-7) and cap_match == 0),
                                  max_tracked_s_ratio_by_layer=cap_max.tolist(), cap_anchor_max_error=cap_match,
                                  checkpoints=ncheck, relative_tolerance=5e-7)
        assert checks['cap_seed0']['pass_']
        # Mutations apply to the extracted fixed-state ledger, not training.
        ma = load_model('C3', ROOT / 'C3/seed0', 'task1', 0)
        mb = load_model('C3', ROOT / 'C3/seed0', 'u100_task2', 0)
        a, b = snapshot(ma, probe(0, 2)), snapshot(mb, probe(0, 2))
        c = snapshot(mb, probe(0, 3))
        zero_error = swap_error = change = 0.0
        for li in range(2):
            frozen = dict(b[li], W=a[li]['W'])
            u, _, _ = decompose(a[li], frozen)
            zero_error = max(zero_error, float(np.abs(u['own']).max()), float(np.abs(u['cross']).max()),
                             float(np.abs(u['dmu'] - u['up']).max()))
            base, _, _ = decompose(a[li], b[li])
            data = (b[li]['W'].mv(c[li]['y'] - b[li]['y']) / a[li]['sd']).numpy()
            combined = ((b[li]['W'].mv(c[li]['y']) - a[li]['W'].mv(a[li]['y'])) / a[li]['sd']).numpy()
            swap_error = max(swap_error, float(np.abs(combined - base['dmu'] - data).max()))
            change = max(change, float(np.abs(data).max()))
            # Evaluate another probe without mutating the fixed-probe endpoints.
            repeat, _, _ = decompose(a[li], b[li])
            assert all(np.array_equal(base[k], repeat[k]) for k in base)
        checks['mutations'] = dict(pass_=max(zero_error, swap_error) <= ATOL and change > 0,
                                  zero_dW_max_error=zero_error, probe_swap_identity_max_error=swap_error,
                                  probe_swap_data_change_max=change,
                                  scope='swap the data-only probe; keep the analysis probe fixed. Replacing the analysis probe can change every decomposition term.')
        assert checks['mutations']['pass_']
        cr = carrier(ma, probe(0, 1))
        checks['carrier_preflight'] = dict(pass_=cr['closure_relative'] < 1e-5,
                                           closure_relative=cr['closure_relative'], projection_sum=cr['projection_sum'])
        checks['status'] = 'preflight_passed'
    except Exception as e:
        checks.update(status='preflight_failed', error=repr(e))
        dump_json(OUT / 'checks.json', checks)
        raise
    checks['preflight_seconds'] = time.monotonic() - start
    dump_json(OUT / 'checks.json', checks)
    print((OUT / 'checks.json').read_text(), flush=True)
    return checks


class Table:
    def __init__(self, name):
        self.f = (OUT / name).open('w', newline='')
        self.writer = None
        self.count = 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.f, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1

    def flush(self):
        self.f.flush()


def main_run(checks):
    start = time.monotonic()
    tabs = {k: Table(f'table_{k}.csv') for k in ('decomp', 'scale', 'data_shift', 'carrier_source')}
    maxres = maxnormres = maxclosure = 0.0
    counts = []
    for arm in ARMS:
        for seed in range(10):
            d = ROOT / arm / f'seed{seed}'
            units = {}
            def evaluate(name, t):
                return snapshot(load_model(arm, d, name, seed), probe(seed, t))

            def record_window(t, w, an, bn, a, b):
                nonlocal maxres, maxnormres
                for li in range(2):
                    key = dict(arm=arm, seed=seed, layer=li, task=t, window=w, start=an, end=bn, probe_task=t)
                    u, rr, nr = decompose(a[li], b[li])
                    maxres, maxnormres = max(maxres, rr), max(maxnormres, nr)
                    tabs['decomp'].add(dict(key, residual=rr, normalized_residual=nr, **summarize_decomp(u)))
                    tabs['scale'].add(dict(key, **scale(a[li], b[li]), rotation_cos_median=float('nan'), one_minus_cos_median=float('nan')))
                    for k, v in u.items():
                        units[f't{t}_{w}_L{li}_{k}'] = v.astype(np.float32)

            for t in TASKS:
                an, bn, en = ('init' if t == 1 else f'task{t-1}'), f'u100_task{t}', f'task{t}'
                a, b = evaluate(an, t), evaluate(bn, t)
                m_end = load_model(arm, d, en, seed)
                e = snapshot(m_end, probe(seed, t))
                record_window(t, 'i', an, bn, a, b)
                record_window(t, 'ii', bn, en, b, e)
                cr = carrier(m_end, probe(seed, t))
                maxclosure = max(maxclosure, cr['closure_relative'])
                tabs['carrier_source'].add(dict(arm=arm, seed=seed, layer=1, task=t, **cr))
            for ta, tb in [(10, 25), (25, 39)]:
                record_window(tb, 'coarse', f'task{ta}', f'task{tb}', evaluate(f'task{ta}', tb), evaluate(f'task{tb}', tb))
            for t in END_TASKS:
                m = load_model(arm, d, f'task{t}', seed)
                a, b = snapshot(m, probe(seed, t)), snapshot(m, probe(seed, t+1))
                for li in range(2):
                    shift = (a[li]['W'].mv(b[li]['y'] - a[li]['y']) / a[li]['sd']).numpy()
                    tabs['data_shift'].add(dict(arm=arm, seed=seed, layer=li, task=t, probe_from=t, probe_to=t+1,
                                               mean=float(shift.mean()), median=float(np.median(shift)),
                                               abs_median=float(np.median(np.abs(shift))), n_units=len(shift)))
                    units[f't{t}_data_L{li}'] = shift.astype(np.float32)
                if t-1 not in END_TASKS:
                    continue  # t1 initialization is not a consecutive task-end pair.
                prev = torch.load(ckpt(d, f'task{t-1}'), map_location='cpu', weights_only=True)
                for li in range(2):
                    wa, wb = prev[f'h.{li}.mlp.up.weight'].double(), a[li]['W']
                    cos = (wa * wb).sum(1) / (wa.norm(dim=1) * wb.norm(dim=1))
                    tabs['scale'].add(dict(arm=arm, seed=seed, layer=li, task=t, window='end_to_end',
                                          start=f'task{t-1}', end=f'task{t}', probe_task='',
                                          up_abs_median=float('nan'), w_norm_median=float(np.median(wb.norm(dim=1))),
                                          dy_norm=float('nan'), alignment_median=float('nan'),
                                          lr_over_w_median=float(np.median(1e-3 / wb.norm(dim=1))),
                                          rotation_cos_median=float(np.median(cos)), one_minus_cos_median=float(np.median(1-cos))))
                    units[f't{t}_rotation_L{li}'] = cos.numpy().astype(np.float32)
            np.savez_compressed(OUT / 'units' / f'{arm}_seed{seed}.npz', **units)
            for tab in tabs.values():
                tab.flush()
            counts.append(dict(arm=arm, seed=seed))
            checks.update(status='running', completed_runs=counts, full_identity_max=maxres,
                          full_normalized_identity_max=maxnormres, full_carrier_closure_max=maxclosure)
            dump_json(OUT / 'checks.json', checks)
            print(f'{arm} seed{seed} complete; elapsed={time.monotonic()-start:.1f}s', flush=True)
    checks.update(status='complete', runtime_seconds=time.monotonic()-start,
                  table_rows={k: v.count for k, v in tabs.items()},
                  analysis_sha256=sha(Path(__file__)),
                  spec_unchanged=sha(HERE / 'specs/spec_chargpt_l0_carrier_1021.md') == checks['spec_sha256'])
    assert checks['spec_unchanged']
    dump_json(OUT / 'checks.json', checks)
    print('COMPLETE ' + json.dumps(checks['table_rows']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checks-only', action='store_true')
    ap.add_argument('--run-after-checks', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / 'units').mkdir(exist_ok=True)
    if args.run_after_checks:
        checks = json.loads((OUT / 'checks.json').read_text())
        assert checks['status'] == 'preflight_passed'
    else:
        checks = preflight()
    if not args.checks_only:
        main_run(checks)
