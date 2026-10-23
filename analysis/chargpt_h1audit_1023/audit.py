#!/usr/bin/env python3
"""Fixed-state H1 diagnostics. No optimizer, parameter updates, or training."""
import argparse
import csv
import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis/chargpt_loadsign_1023'))
import measure as M
from report import WINDOWS
A, C, torch, np = M.A, M.C, M.torch, M.np
OUT = ROOT/'results/chargpt_h1audit_1023'
SOURCES = ['self', 'other', 'residual']
LAYERS = ['direct', 'exact']


def share(a, b, alive):
    d = a[alive].abs()+b[alive].abs()
    nz = d > 0
    return M.median(a[alive][nz].abs()/d[nz]), int((~nz).sum())


def prank(g):
    return float(g.trace().square()/g.square().sum().clamp_min(1e-300))


@torch.no_grad()
def calculate(m, x, labels, truth):
    hs, rs, zs, inputs = [], [], [], []
    for _, xx in A.chunks(x):
        cap = []; hs.append(m.body(xx, cap).reshape(-1, C.D))
        rs.append(cap[2].reshape(-1, C.D)); zs.append(cap[3].reshape(-1, C.FF))
        inputs.append(m.h[1].n2(cap[2]).reshape(-1, C.D))
    h, r, z, xf = map(torch.cat, [hs, rs, zs, inputs])
    mlp = m.h[1].mlp; a = mlp.phi(z); gate = mlp.dphi(z)
    U = mlp.down.weight; Uc = U-U.mean(0, keepdim=True)
    gamma = m.nf.weight; T = gamma[:, None]*Uc; G = T.T@T
    ybar = xf.mean(0); k = xf@ybar; alive = (z > 0).any(0)
    t = r+mlp.down(a); sig = (M.cent(t).square().mean(-1, keepdim=True)+m.nf.eps).sqrt()
    qln = M.cent(t)/sig
    hr = M.cent(r)*gamma/sig; hf = (a@T.T)/sig
    target = torch.nn.functional.one_hot(labels.reshape(-1), C.V).to(A.DT)
    xi = target-1/C.V
    vc = m.head.weight-m.head.weight.mean(0); F = xi.T@h
    kappa = (vc*F).sum()/F.square().sum(); vl = kappa*F; vp = vc-vl
    logits = m.head(h); p = logits.softmax(-1)
    B = xi.T@(a/sig); Fr = xi.T@hr
    Qself = kappa*B*G.diag(); Qr = kappa*Fr@T; Qo = vl@T-Qself-Qr
    qU = qln@U; qgT = qln@(gamma[:, None]*T)
    weight = k[:, None]*gate
    def avg(pull): return (weight*pull).mean(0)
    acc = {}
    err_sources = [('actual', p-1/C.V), ('oracle', xi), ('truth', truth-1/C.V)]
    for en, e in err_sources:
        b = gamma*(e@vl)
        rad = -qU*(qln*b).mean(-1, keepdim=True)/sig
        rr = -qU*(qln*(gamma*(e@(kappa*Fr)))).mean(-1, keepdim=True)/sig
        rs = -qU*qgT*(e@(kappa*B))/(C.D*sig)
        ro = rad-rr-rs
        for sn, Q, radpart in [('self', Qself, rs), ('other', Qo, ro), ('residual', Qr, rr)]:
            direct = (e@Q)/sig
            acc[f'{en}_{sn}_direct'] = avg(direct)
            acc[f'{en}_{sn}_exact'] = avg(direct+radpart)
        for ln in LAYERS:
            acc[f'{en}_load_{ln}'] = sum(acc[f'{en}_{s}_{ln}'] for s in SOURCES)
        for vn, v in [('full', vc), ('headperp', vp)]:
            b = gamma*(e@v); direct = M.cent(b)@U/sig
            radial = -qU*(qln*b).mean(-1, keepdim=True)/sig
            acc[f'{en}_{vn}_direct'] = avg(direct)
            acc[f'{en}_{vn}_exact'] = avg(direct+radial)
        for ln in LAYERS:
            acc[f'{en}_unitrest_{ln}'] = acc[f'{en}_full_{ln}']-acc[f'{en}_self_{ln}']
    c = 1-1/C.V
    # Diagonal m=n of xi_m . xi_n, without forming the N x N label Gram.
    ds = kappa*c*a*G.diag()/sig.square()
    rs = -kappa*c*a*qU*qgT/(C.D*sig.square())
    for name, feat in [('ffn', hf), ('residual', hr), ('load', h)]:
        b = feat*gamma
        dd = kappa*c*(M.cent(b)@U)/sig
        dr = -kappa*c*qU*(qln*b).mean(-1, keepdim=True)/sig
        acc[f'diag_{name}_direct'] = avg(dd)
        acc[f'diag_{name}_exact'] = avg(dd+dr)
    acc['diag_self_direct'] = avg(ds); acc['diag_self_exact'] = avg(ds+rs)
    acc['diag_self_direct_absK'] = (k.abs()[:, None]*gate*ds).mean(0)
    for ln in LAYERS:
        acc[f'diag_other_{ln}'] = acc.pop(f'diag_ffn_{ln}')-acc[f'diag_self_{ln}']
        for sn in SOURCES+['load']:
            acc[f'off_{sn}_{ln}'] = acc[f'oracle_{sn}_{ln}']-acc[f'diag_{sn}_{ln}']
        for sn in SOURCES+['load', 'full', 'headperp']:
            acc[f'fit_{sn}_{ln}'] = acc[f'actual_{sn}_{ln}']-acc[f'oracle_{sn}_{ln}']
    # Independently evaluate the error correction and summed direct/exact loading.
    checks = [M.relmax(h-hr-hf, h), M.relmax(F-B@T.T-Fr, F)]
    for en, e in err_sources:
        b = gamma*(e@vl); dd = M.cent(b)@U/sig
        ex = dd-qU*(qln*b).mean(-1, keepdim=True)/sig
        for ln, pull in [('direct', dd), ('exact', ex)]:
            checks.append(M.relmax(acc[f'{en}_load_{ln}']-avg(pull), avg(pull)))
            checks.append(M.relmax(acc[f'{en}_full_{ln}']-acc[f'{en}_load_{ln}']-acc[f'{en}_headperp_{ln}'], acc[f'{en}_full_{ln}']))
    b = gamma*((p-target)@vl); dd = M.cent(b)@U/sig
    ex = dd-qU*(qln*b).mean(-1, keepdim=True)/sig
    for ln, pull in [('direct', dd), ('exact', ex)]:
        checks.append(M.relmax(acc[f'fit_load_{ln}']-avg(pull), acc[f'actual_load_{ln}'], acc[f'oracle_load_{ln}']))
        checks.append(M.relmax(acc[f'diag_load_{ln}']-sum(acc[f'diag_{s}_{ln}'] for s in SOURCES), acc[f'diag_load_{ln}']))
    identity = max(checks); assert identity <= 1e-10, identity
    xe = xf-xf.mean(0); gn = G/G.diag().sqrt()[:, None].clamp_min(1e-300)/G.diag().sqrt()[None, :].clamp_min(1e-300)
    row = dict(n_alive=int(alive.sum()), kappa=float(kappa), cos2=float((vc*F).sum().square()/(vc.square().sum()*F.square().sum())),
               identity_error=identity, K_positive=float((k>0).double().mean()),
               input_rank=prank(xf.T@xf), input_centered_rank=prank(xe.T@xe),
               down_rank=prank(G), down_cos2=float((gn.square().sum()-gn.diag().square().sum())/(C.FF*(C.FF-1))),
               accuracy=float((p.argmax(-1)==labels.reshape(-1)).double().mean()), mean_p_true=float((p*target).sum(-1).mean()),
               true_entropy=float(-(truth*truth.clamp_min(1e-300).log()).sum(-1).mean()),
               KL_true_pred=float((truth*(truth.clamp_min(1e-300).log()-logits.log_softmax(-1))).sum(-1).mean()))
    for name, v in acc.items():
        row['s_'+name] = A.sink(v, alive); row['abs_'+name] = M.median(v[alive].abs())
    for ln in LAYERS:
        for sn in ['self', 'load']:
            names = [(f'off_{sn}_{ln}', f'diag_{sn}_{ln}'), (f'fit_{sn}_{ln}', f'oracle_{sn}_{ln}')]
            for aa, bb in names:
                sh, nz = share(acc[aa], acc[bb], alive)
                row[f'share_{aa}'] = sh; row[f'share_zero_{aa}'] = nz
        for en in ['actual', 'oracle', 'truth']:
            aa, bb = acc[f'{en}_unitrest_{ln}'], acc[f'{en}_self_{ln}']
            row[f'share_{en}_unitrest_{ln}'], row[f'share_zero_{en}_unitrest_{ln}'] = share(aa, bb, alive)
            for pair, base, total in [
                ('other_flips', acc[f'{en}_self_{ln}']+acc[f'{en}_residual_{ln}'], acc[f'{en}_load_{ln}']),
                ('rest_flips', bb, bb+aa)]:
                row[f'{en}_{ln}_{pair}'] = float(((base[alive]>0)&(total[alive]<=0)).double().mean())
    acc['alive'] = alive
    return row, acc, ybar, vl


def oracle_gradient(m, x, labels, v):
    g = torch.zeros_like(m.h[1].mlp.up.weight)
    for i, xx in A.chunks(x):
        h = m.body(xx)
        e = torch.nn.functional.one_hot(labels[i:i+len(xx)], C.V).to(A.DT)-1/C.V
        g += torch.autograd.grad(h, m.h[1].mlp.up.weight, grad_outputs=(e@v)/x.numel())[0].detach()
    return g


class Inputs(M.Inputs):
    def __init__(self):
        super().__init__()
        self.probe_ref = {(r['arm'], int(r['seed']), int(r['task'])): r for r in csv.DictReader((ROOT/'results/chargpt_loadsign_1023/probe_control_states.csv').open())}
        self.truth_cache = {}

    def truth(self, seed, task, x):
        key = (seed, task)
        if key not in self.truth_cache:
            chain = C.ChainM(seed, task, .01)
            p = torch.from_numpy(chain.P); pi = torch.from_numpy(chain.pi)
            q = torch.empty((*x.shape, C.V), dtype=A.DT)
            q[:, 0] = pi[x[:, 0]]/pi[x[:, 0]].sum(-1, keepdim=True)
            q[:, 1:] = p[x[:, :-1], x[:, 1:]]
            assert float((q.sum(-1)-1).abs().max()) < 1e-12
            self.truth_cache[key] = q.reshape(-1, C.V)
        return self.truth_cache[key]


def state(inp, arm, seed, task, autograd=False):
    m = inp.load(arm, seed, task); x, labels, _, _ = A.data(seed, task)
    row, raw, ybar, vl = calculate(m, x, labels, inp.truth(seed, task, x))
    ref = inp.probe_ref[arm, seed, task]
    assert row['n_alive'] == int(ref['n_alive'])
    assert row['s_actual_load_exact'] == float(ref['s_switch_load'])
    assert row['s_actual_full_exact'] == float(ref['s_switch_full'])
    assert A.rel(row['kappa'], ref['kappa']) <= 1e-10
    agerr = 0.
    if autograd:
        for en, g in [('actual', M.P.fixed_error_gradients(m, x, [vl])[0][1]), ('oracle', oracle_gradient(m, x, labels, vl))]:
            exact = raw[f'{en}_load_exact']; agerr = max(agerr, M.relmax(exact-(g*ybar).sum(1), exact))
        assert agerr <= 1e-10, agerr
    return dict(arm=arm, seed=seed, task=task, autograd_error=agerr, **row), raw


@torch.no_grad()
def tensor_check():
    """Independent explicit m,n sums + exhaustive independent-label expectation."""
    gen = torch.Generator().manual_seed(102322)
    def rand(*sz): return torch.randn(*sz, generator=gen, dtype=A.DT)
    N, D, Cc, I = 4, 5, 3, 6
    a = rand(N, I).abs(); r = rand(N, D); U = rand(D, I); gamma = rand(D).abs()+.2
    t = r+a@U.T; sig = (M.cent(t).square().mean(-1, keepdim=True)+1e-5).sqrt(); q = M.cent(t)/sig
    h = gamma*q; T = gamma[:, None]*(U-U.mean(0)); G = T.T@T; k = rand(N); gate = (rand(N, I)>0).double()
    J = torch.stack([torch.diag(gamma)@(torch.eye(D, dtype=A.DT)-torch.ones(D, D, dtype=A.DT)/D-qq[:, None]*qq[None, :]/D)/ss for qq, ss in zip(q, sig.flatten())])
    deriv = J@U; c = 1-1/Cc
    analytic = c*(k[:, None]*gate*(h[:, :, None]*deriv).sum(1)).mean(0)
    samples = []
    worst = 0.
    for labels in itertools.product(range(Cc), repeat=N):
        xi = torch.nn.functional.one_hot(torch.tensor(labels), Cc).to(A.DT)-1/Cc
        F = xi.T@h
        fast = (k[:, None]*gate*torch.einsum('nc,cd,ndi->ni', xi, F, deriv)).mean(0)
        explicit = torch.zeros(I, dtype=A.DT); diagonal = torch.zeros(I, dtype=A.DT)
        for n in range(N):
            for mm in range(N):
                term = k[n]*gate[n]*(xi[mm]@xi[n])*(h[mm]@deriv[n])/N
                explicit += term
                if mm == n: diagonal += term
        worst = max(worst, M.relmax(fast-explicit, fast), M.relmax(diagonal-analytic, analytic))
        samples.append(fast)
    worst = max(worst, M.relmax(torch.stack(samples).mean(0)-analytic, analytic))
    # Direct/exact self diagonal derived independently from each token's own feature.
    self_feature = a[:, :, None]*T.T[None, :, :]/sig[:, :, None]
    directJ = torch.diag(gamma)@(torch.eye(D, dtype=A.DT)-torch.ones(D, D, dtype=A.DT)/D)
    direct = (self_feature*(directJ@U).T[None, :, :]).sum(-1)/sig
    exact = (self_feature*deriv.transpose(1, 2)).sum(-1)
    ds = a*G.diag()/sig.square()
    ex = ds-a*(q@U)*(q@(gamma[:, None]*T))/(D*sig.square())
    worst = max(worst, M.relmax(direct-ds, ds), M.relmax(exact-ex, ex))
    assert worst <= 1e-10, worst
    return dict(max_error=worst, exhaustive_label_assignments=len(samples), passed=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['gate', 'run']); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); inp = Inputs()
    if args.mode == 'gate':
        bundle_sha = M.sha(M.BUNDLE); assert bundle_sha == M.BUNDLE_SHA
        toy = tensor_check(); rows = []; start = time.monotonic()
        for arm in A.ARMS:
            for t in A.TASKS:
                row, _ = state(inp, arm, 0, t, t in [1, 9, 50, 100]); rows.append(row)
        M.dump(OUT/'gate.json', dict(passed=True, rows=len(rows), code_sha256=M.sha(__file__), bundle_sha256=bundle_sha,
            toy=toy, max_autograd=max(r['autograd_error'] for r in rows), max_identity=max(r['identity_error'] for r in rows),
            previous_signs_exact=True, n_alive_exact=True, kappa_tolerance=1e-10, seconds=time.monotonic()-start))
        print('gate passed: 70 states, 8 autograd, explicit double sums and exhaustive IID labels', flush=True)
        return
    g = json.loads((OUT/'gate.json').read_text()); assert g['passed'] and g['code_sha256'] == M.sha(__file__)
    rows = []; start = time.monotonic()
    for seed in range(10):
        for arm in A.ARMS:
            ts = time.monotonic(); rr = []; arrays = {}
            for task in A.TASKS:
                row, raw = state(inp, arm, seed, task); rr.append(row)
                arrays.update({f't{task}_{key}': value.numpy() for key, value in raw.items()})
            part = OUT/'parts'/f'{arm}_seed{seed}.csv'; M.write_csv(part, rr)
            rawfile = OUT/'raw'/f'{arm}_seed{seed}.npz'; rawfile.parent.mkdir(exist_ok=True)
            np.savez_compressed(rawfile, **arrays)
            M.dump(part.with_suffix('.json'), dict(sha256=M.sha(part), raw_sha256=M.sha(rawfile), code_sha256=M.sha(__file__), seconds=time.monotonic()-ts))
            rows.extend(rr); print(f'{arm} seed{seed}: {time.monotonic()-ts:.1f}s', flush=True)
        A.CACHE.clear(); A.UCACHE.clear(); inp.truth_cache.clear()
    M.write_csv(OUT/'states.csv', rows)
    fields = [k for k in rows[0] if k not in ['arm', 'seed', 'task']]; wins = []
    for arm in A.ARMS:
        for seed in range(10):
            for window, tasks in WINDOWS.items():
                rr = [r for r in rows if r['arm']==arm and r['seed']==seed and r['task'] in tasks]
                wins.append(dict(arm=arm, seed=seed, window=window, **{f:float(np.median([r[f] for r in rr])) for f in fields}))
    M.write_csv(OUT/'windows.csv', wins)
    M.dump(OUT/'provenance.json', dict(git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256=M.sha(__file__), spec_sha256=M.sha(ROOT/'specs/spec_chargpt_h1audit_1023.md'), bundle_sha256=M.BUNDLE_SHA,
        input_code={str(p.relative_to(ROOT)):M.sha(p) for p in [ROOT/'analysis/chargpt_loadsign_1023/measure.py', ROOT/'analysis/chargpt_latepush_1023/measure.py', ROOT/'analysis/chargpt_seed10_1022/load10.py']},
        rows=len(rows), training_steps=0, optimizer_created=False, checkpoint_extracted=False, device='cpu', dtype='float64', threads=torch.get_num_threads(),
        seconds=time.monotonic()-start, finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), max_identity_error=max(r['identity_error'] for r in rows)))


if __name__ == '__main__': main()
