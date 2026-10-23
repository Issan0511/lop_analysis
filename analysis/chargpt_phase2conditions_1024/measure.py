#!/usr/bin/env python3
"""Frozen-state CE/LS/label-mean ledger and Gaussian direction diagnostic."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location('silu_parent', ROOT/'analysis/chargpt_siluphase2_1024/measure.py')
S = importlib.util.module_from_spec(sp)
sp.loader.exec_module(S)
E, Q, M, A, C, torch, np = S.E, S.Q, S.M, S.A, S.C, S.torch, S.np
OUT = ROOT/'results/chargpt_phase2conditions_1024'
SPEC = ROOT/'specs/spec_chargpt_phase2conditions_1024.md'
BASES = ['P2', 'P2ctx', 'ls_task', 'sq_range', 'sq_full', 'ce', 'actual']
PARTS = ['iid_mean', 'context_shift', 'task_departure', 'fit_gap', 'null', 'ce']
METRICS = ['SGD', 'Adam_fixed']
EARLY = [(arm, seed, t, 10000) for arm in ['C3', 'ReLUcap'] for seed in range(10) for t in range(1, 10)]
SILU = [(arm, 1000, t, 10000) for arm in ['SiLU', 'SiLUcap'] for t in [50, 100]]
GATES = [('C3', 0, t, 10000) for t in [1, 6, 9]] + [('ReLUcap', 0, 9, 10000), ('SiLU', 1000, 100, 10000)]
ZV = torch.randn(64, C.D, generator=torch.Generator().manual_seed(1024024), dtype=A.DT)
dump, arr, rel, csvwrite = M.dump, Q.arr, E.rel, E.csvwrite
CODE = None


def norm(x):
    return float(torch.linalg.vector_norm(x)) if torch.is_tensor(x) else float(np.linalg.norm(x))


def ratio(a, b):
    return norm(a)/max(norm(b), 1e-300)


def check(checks, key, a, b):
    aa, bb = arr(a), arr(b)
    value = float(np.max(np.abs(aa-bb)))/max(float(np.max(np.abs(aa))), float(np.max(np.abs(bb))), 1e-300)
    checks[key] = float(value)
    assert value <= 1e-8, (key, value)


@torch.no_grad()
def signals(m, seed, task):
    x = A.data(seed, task)[0]
    h = torch.cat([m.body(xx)[:, 1:].reshape(-1, C.D) for _, xx in A.chunks(x)])
    n = len(h)
    eig, vec = torch.linalg.eigh(h.T@h)
    keep = eig > eig[-1]*1e-10
    u, ev = vec[:, keep], eig[keep]
    gi, root = (u/ev)@u.T, (u/ev.sqrt())@u.T
    vc = (m.head.weight-m.head.weight.mean(0)).T
    vr = u@(u.T@vc)
    logits = h@vc
    p = logits.softmax(-1)
    qt = torch.from_numpy(C.ChainM(seed, task, .01).P)
    q = qt[x[:, :-1], x[:, 1:]].reshape(-1, C.V)
    target = C.V*(q-1/C.V)
    vs = gi@h.T@target
    delta = p-1/C.V-logits/C.V
    dual = h@gi
    contexts = (x[:, :-1]*C.V+x[:, 1:]).reshape(-1)
    uq, ix = torch.unique(contexts, return_inverse=True)
    summed = torch.zeros(len(uq), C.D, dtype=A.DT)
    summed.index_add_(0, ix, dual)
    beta = C.V*(C.V-1)/(C.V*.01+1)
    scale = beta/(C.V*n)
    out = dict(P2=dual, P2ctx=h@(summed.T@summed), ls_task=h@vs@vs.T/(C.V*n),
               sq_range=h@vr@vr.T/(C.V*n), sq_full=h@vc@vc.T/(C.V*n),
               ce=delta@vc.T/n, actual=(p-1/C.V)@vc.T/n)
    checks = {}
    check(checks, 'feature_ledger', out['sq_full']+out['ce'], out['actual'])
    rhead = h.T@(p-q)
    check(checks, 'head_gap_identity', vr-vs, C.V*gi@(rhead-h.T@delta))
    # Closed-form LS normal equations restricted to the retained subspace.
    check(checks, 'ls_normal_equation', u.T@h.T@(h@vs), u.T@h.T@target)
    b, bs = ev.sqrt()[:, None]*(u.T@vr), ev.sqrt()[:, None]*(u.T@vs)
    stats = dict(rank=int(keep.sum()), rank_1e8=int((eig > eig[-1]*1e-8).sum()),
                 rank_1e12=int((eig > eig[-1]*1e-12).sum()), condition=float(ev[-1]/ev[0]),
                 n=n, contexts=len(uq), beta=beta, scale=scale,
                 fit_error=ratio(b-bs, bs), head_ce_ratio=ratio(rhead, h.T@(p-1/C.V)),
                 head_gap_ce_part=ratio(C.V*gi@rhead, vr-vs),
                 head_gap_delta_part=ratio(C.V*gi@h.T@delta, vr-vs),
                 logit_rms=float(logits.square().mean().sqrt()),
                 delta_ratio=ratio(delta, p-1/C.V), pmax=float(p.max(-1).values.mean()),
                 qmax=float(q.max(-1).values.mean()))
    aux = dict(h=h, gi=gi, root=root, vc=vc, vs=vs, vr=vr, delta=delta,
               p=p, q=q, b=b, bs=bs, u=u, ev=ev, wp=root@ZV.T, n=n, scale=scale)
    return x, out, aux, stats, checks


def backward(m, x, sig, aux, raw, den, early):
    allparams = dict(m.named_parameters())
    params = {k: w for k, w in allparams.items() if (not early or k in ['h.0.mlp.up.weight', 'h.1.mlp.up.weight'])}
    gs = {name: {key: torch.zeros_like(w) for key, w in params.items()} for name in sig}
    gauss = {metric: torch.zeros(64, 2, C.FF, dtype=A.DT) for metric in METRICS} if early else None
    for start, xx in A.chunks(x):
        hh = m.body(xx)[:, 1:].reshape(-1, C.D)
        lo, hi = start*(C.T-1), start*(C.T-1)+len(hh)
        for name, signal in sig.items():
            gg = torch.autograd.grad(hh, list(params.values()), grad_outputs=signal[lo:hi],
                                     retain_graph=True, allow_unused=True)
            for (key, _), v in zip(params.items(), gg):
                if v is not None:
                    gs[name][key] += v.detach()
        if early:
            for start_probe in range(0, 64, 8):
                ww = aux['wp'][:, start_probe:start_probe+8]
                so = (hh.detach()@ww).T[:, :, None]*ww.T[:, None, :]
                gg = torch.autograd.grad(hh, list(params.values()), grad_outputs=so,
                                         retain_graph=True, is_grads_batched=True)
                for layer, vv in enumerate(gg):
                    for metric in METRICS:
                        direction = raw[layer]['kr'] if metric == 'SGD' else raw[layer]['kr']/den[layer]
                        gauss[metric][start_probe:start_probe+8, layer] += (vv.detach()*direction).sum(-1)
    checks = {}
    for key in params:
        check(checks, 'gradient_ledger_'+key, gs['sq_full'][key]+gs['ce'][key], gs['actual'][key])
    if not early:
        direct = {key: torch.zeros_like(w) for key, w in params.items()}
        direct['wte.weight'] = (aux['p']-1/C.V).T@aux['h']/aux['n']
        gs['direct'] = direct
        # Independent full CE autograd, including tied-head parameter appearances.
        full = {key: torch.zeros_like(w) for key, w in params.items()}
        for _, xx in A.chunks(x):
            f = m(xx)[:, 1:].reshape(-1, C.V)
            loss = (f.logsumexp(-1)-f.mean(-1)).sum()/aux['n']
            gg = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
            for (key, _), v in zip(params.items(), gg):
                if v is not None:
                    full[key] += v.detach()
        for key in params:
            check(checks, 'tied_full_'+key, full[key], gs['actual'][key]+direct[key])
        gs['actual_full'] = full
    return gs, gauss, checks


def exact_a(m, raw, aux, gauss, projected, seed):
    base = {k: v.detach() for k, v in m.named_parameters()}
    x = A.data(seed, exact_a.task)[0]
    results, checks = {}, {}
    for unit in [0, 127, 255, 511]:
        tangent = {k: torch.zeros_like(v) for k, v in base.items()}
        tangent['h.0.mlp.up.weight'][unit] = raw[0]['kr'][unit]
        dd = []
        for _, xx in A.chunks(x):
            def fun(pp):
                # forward returns logits, so functional-call a body wrapper.
                return torch.func.functional_call(Body(m), {f'model.{k}': v for k, v in pp.items()}, (xx,), tie_weights=True)
            with torch.no_grad():
                _, dh = torch.func.jvp(fun, (base,), (tangent,))
            dd.append(dh[:, 1:].reshape(-1, C.D))
        d = torch.cat(dd)
        mat = aux['root']@aux['h'].T@d@aux['root']
        mat = (mat+mat.T)/2
        tr = mat.trace()
        check(checks, f'u{unit}_trace', tr, projected['P2'][unit])
        quad = (ZV@mat*ZV).sum(1)
        check(checks, f'u{unit}_gaussian', quad, gauss[:, 0, unit])
        vals = torch.linalg.eigvalsh(mat)
        f = norm(mat)
        b = aux['u']@(aux['b'])
        bs = aux['u']@(aux['bs'])
        gap = b-bs
        fit = (2*(gap*(mat@bs)).sum()+(gap*(mat@gap)).sum())/(C.V*aux['n'])
        check(checks, f'u{unit}_fit', fit, projected['sq_range'][unit]-projected['ls_task'][unit])
        bound = (2*norm(mat@bs)*norm(gap)+float(vals.abs().max())*norm(gap)**2)/(C.V*aux['n'])
        ce_bound = norm(d@aux['vc'])*norm(aux['delta'])/aux['n']
        assert abs(float(fit)) <= bound+1e-12
        assert abs(float(projected['ce'][unit])) <= ce_bound+1e-12
        results[str(unit)] = dict(trace=float(tr), frobenius=f, eig_min=float(vals[0]), eig_max=float(vals[-1]),
                                  Z=None if f == 0 else float((32**.5)*tr.abs()/f),
                                  fit=float(fit), fit_bound=bound, ce=float(projected['ce'][unit]), ce_bound=ce_bound,
                                  Zhat=None if f == 0 else float(8*tr.abs()/((gauss[:, 0, unit]-tr).square().mean().sqrt())))
    return results, checks


class Body(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x):
        return self.model.body(x)


def algebra_gate():
    gen = torch.Generator().manual_seed(1024025)
    h = torch.randn(11, 4, generator=gen, dtype=A.DT)
    gi = torch.linalg.inv(h.T@h)
    ctx = torch.nn.functional.one_hot(torch.arange(11) % 3, 3).double()
    pred = h@gi@h.T@ctx@ctx.T@h@gi
    expected = torch.zeros_like(h)
    for y in ctx.T:
        v = gi@h.T@y
        expected += (h@v)[:, None]*v[None, :]
    checks = {}
    check(checks, 'context_LS_basis_sum', pred, expected)
    return checks


def old_reference(arm, seed, task, step, arrays, early):
    if early:
        p = M.DATA/'chargpt_earlyphase_1024/results/chargpt_earlyphase_1024/raw'/f'{arm}_seed{seed}_task{task}.npz'
    else:
        p = M.DATA/'chargpt_siluphase2_1024/results/chargpt_siluphase2_1024/raw'/f'{arm}_t{task}_u{step}.npz'
    checks = {}
    with np.load(p) as old:
        for layer in ([0, 1] if early else [2, 3]):
            check(checks, f'L{layer}_r', arrays[f'L{layer}_r'], old[f'L{layer}_r'])
            for metric in METRICS:
                prefix = f'L{layer}_{metric}'
                op = f'{prefix}_r' if early else f'{prefix}_r_full'
                check(checks, prefix+'_P2', arrays[prefix+'_P2'], old[op+'_P2'])
                check(checks, prefix+'_actual', arrays[prefix+'_actual' if early else prefix+'_actual_full'],
                      old[op+'_switch' if early else op+'_switch_full'])
    return dict(path=str(p), sha256=M.sha(p), checks=checks)


def state(inp, arm, seed, task, step, gate=False):
    tag = f'{arm}_seed{seed}_task{task}_u{step}'
    dest = OUT/'raw'/f'{tag}.npz'
    early = arm in ['C3', 'ReLUcap']
    if early:
        m = inp.load(arm, seed, task)
        den = inp.adam(m, arm, seed, task)
        raw = E.fields(m, seed)
    else:
        m, den, folder = S.load(arm, task, step)
        raw = S.fields(m)
    x, sig, aux, stats, checks = signals(m, seed, task)
    gs, gauss, gg_checks = backward(m, x, sig, aux, raw, den, early)
    checks.update(gg_checks)
    arrays, exact = {}, None
    layers = [0, 1] if early else [2, 3]
    for layer in layers:
        for k in ['r', 'mu', 's']:
            arrays[f'L{layer}_{k}'] = arr(raw[layer][k])
        arrays[f'L{layer}_mask'] = arr(raw[layer]['alive' if early else 'responsive'])
    for metric in METRICS:
        projections = {layer: {} for layer in layers}
        for name, g0 in gs.items():
            if early:
                for layer in layers:
                    gg = g0[f'h.{layer}.mlp.up.weight']
                    if metric == 'Adam_fixed':
                        gg = gg/den[layer]
                    projections[layer][name] = (gg*raw[layer]['kr']).sum(1)
            else:
                g = g0 if metric == 'SGD' else {k: v/den[k] for k, v in g0.items()}
                pp = S.full(m, raw, g)
                for layer in layers:
                    projections[layer][name] = pp[layer]['r']
        for layer, pp in projections.items():
            pp['iid_mean'] = pp['P2']*aux['scale']
            pp['ctx_mean'] = pp['P2ctx']*aux['scale']
            pp['context_shift'] = pp['ctx_mean']-pp['iid_mean']
            pp['task_departure'] = pp['ls_task']-pp['ctx_mean']
            pp['fit_gap'] = pp['sq_range']-pp['ls_task']
            pp['null'] = pp['sq_full']-pp['sq_range']
            check(checks, f'L{layer}_{metric}_r_ledger', sum(pp[k] for k in PARTS), pp['actual'])
            for name, value in pp.items():
                arrays[f'L{layer}_{metric}_{name}'] = arr(value)
            if early:
                v = gauss[metric][:, layer]
                arrays[f'L{layer}_{metric}_gaussian'] = arr(v)
                for name, select in [('Zhat', slice(None)), ('Zhat_first', slice(0, 32)), ('Zhat_last', slice(32, 64))]:
                    vv = (v[select]-pp['P2']).square().mean(0)
                    zz = 8*pp['P2'].abs()/vv.sqrt()
                    zz = torch.where(vv > 0, zz, torch.full_like(zz, float('nan')))
                    arrays[f'L{layer}_{metric}_{name}'] = arr(zz)
        if gate and arm == 'C3' and metric == 'SGD':
            exact_a.task = task
            exact, cc = exact_a(m, raw, aux, gauss[metric], projections[0], seed)
            checks.update(cc)
    metadata = dict(arm=arm, seed=seed, task=task, step=step, early=early, stats=stats, checks=checks,
                    exact_A=exact, code_sha256=CODE, spec_sha256=M.sha(SPEC))
    if gate:
        metadata['reference'] = old_reference(arm, seed, task, step, arrays, early)
    np.savez_compressed(dest, **arrays)
    dump(dest.with_suffix('.json'), metadata)
    return metadata


def main():
    global CODE
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    CODE = M.sha(__file__)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'raw').mkdir(exist_ok=True)
    start = time.time()
    assert M.sha(M.BUNDLE) == M.BUNDLE_SHA
    inp = E.Inputs()
    if args.gate:
        algebra = algebra_gate()
        selected = GATES
    else:
        gate = json.loads((OUT/'gate.json').read_text())
        assert gate['passed'] and gate['code_sha256'] == CODE
        selected = EARLY+SILU
    records = []
    for arm, seed, task, step in selected:
        path = OUT/'raw'/f'{arm}_seed{seed}_task{task}_u{step}.json'
        if not args.gate and path.exists():
            meta = json.loads(path.read_text())
            assert meta['code_sha256'] == CODE
        else:
            meta = state(inp, arm, seed, task, step, args.gate)
        records.append(meta)
        print(arm, seed, task, step, 'complete', round(time.time()-start, 1), 'seconds', flush=True)
    if args.gate:
        dump(OUT/'gate.json', dict(passed=True, code_sha256=CODE, algebra=algebra, states=records,
                                  inputs=inp.used, silu_inputs=S.USED, elapsed_seconds=time.time()-start))
    else:
        dump(OUT/'checks.json', records)
        gate = json.loads((OUT/'gate.json').read_text())
        dump(OUT/'provenance.json', dict(git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             code_sha256=CODE, spec_sha256=M.sha(SPEC), inputs={**gate['inputs'], **inp.used}, silu_inputs={**gate['silu_inputs'], **S.USED},
             bundle_sha256=M.BUNDLE_SHA, states=len(records), training_steps=0, optimizer_created=False,
             gaussian_probes=64, gaussian_seed=1024024, elapsed_seconds=time.time()-start,
             imported_sources={str(p.relative_to(ROOT)): M.sha(p) for p in [Path(E.__file__), Path(S.__file__), Path(Q.__file__), Path(A.__file__)]}))


if __name__ == '__main__':
    main()
