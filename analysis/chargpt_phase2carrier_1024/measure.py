#!/usr/bin/env python3
"""Frozen C3 checkpoints: full-parameter phase-2/CE carrier ledger."""
import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location('early', ROOT/'analysis/chargpt_earlyphase_1024/measure.py')
E = importlib.util.module_from_spec(sp)
sp.loader.exec_module(E)
torch, np, A, C, Q, M = E.torch, E.np, E.A, E.C, E.Q, E.M
OUT = ROOT/'results/chargpt_phase2carrier_1024'
SPEC = ROOT/'specs/spec_chargpt_phase2carrier_1024.md'
TASKS = [1, 4, 5, 6, 9, 39, 50, 89, 100]
GROUPS = ['embedding', 'attention0', 'ln0', 'up0', 'down0', 'attention1', 'ln1', 'up1', 'downstream']
SIGNALS = ['P2', 'P2ctx', 'switch', 'fit']
METRICS = ['SGD', 'Adam_fixed']
dump, arr, rel = E.dump, Q.arr, E.rel


def check(checks, key, aa, bb, tol=1e-8):
    error = rel(aa, bb)
    checks[key] = float(error)
    assert error <= tol, (key, error)


def group_name(key):
    if key.startswith(('wte.', 'wpe.')):
        return 'embedding'
    for l in [0, 1]:
        if key.startswith((f'h.{l}.attn.', f'h.{l}.n1.')):
            return f'attention{l}'
        if key.startswith(f'h.{l}.n2.'):
            return f'ln{l}'
        if key.startswith(f'h.{l}.mlp.up.'):
            return f'up{l}'
        if l == 0 and key.startswith('h.0.mlp.down.'):
            return 'down0'
    if key.startswith(('h.1.mlp.down.', 'nf.')):
        return 'downstream'
    raise ValueError('Unassigned parameter '+key)


def load(inp, seed, task):
    m = inp.load('C3', seed, task)
    params = dict(m.named_parameters())
    membership = {k: group_name(k) for k in params}
    assert set(membership.values()) == set(GROUPS)
    assert not any(k.endswith('.bias') for k in params)
    st = inp.read(f'results/chargpt_cl_1019/M2/C3/seed{seed}/ckpts/task{task}_adam.pt')
    pg = st['param_groups'][0]
    assert len(params) == len(pg['params'])
    den = {}
    for idx, (key, w) in zip(pg['params'], params.items()):
        ss = st['state'][idx]
        assert ss['exp_avg_sq'].shape == w.shape
        den[key] = (ss['exp_avg_sq'].double()/(1-pg['betas'][1]**float(ss['step']))).sqrt()+pg['eps']
    return m, den, membership


def gradients(m, seed, task):
    x = A.data(seed, task)[0]
    n = len(x)*(C.T-1)
    params = dict(m.named_parameters())
    with torch.no_grad():
        h = torch.cat([m.body(xx)[:, 1:].reshape(-1, C.D) for _, xx in A.chunks(x)])
        ev, uu = torch.linalg.eigh(h.T@h)
        ranks = {f'{tol:g}': int((ev > ev[-1]*tol).sum()) for tol in [1e-10, 1e-8, 1e-12]}
        signals = {}
        for tol in [1e-10, 1e-8, 1e-12]:
            if tol != 1e-10 and ranks[f'{tol:g}'] == ranks['1e-10']:
                continue
            keep = ev > ev[-1]*tol
            v = uu[:, keep]
            signals['P2' if tol == 1e-10 else f'P2tol{tol:g}'] = (h@v/ev[keep])@v.T
        ctx = (x[:, :-1]*C.V+x[:, 1:]).reshape(-1)
        uq, ix = torch.unique(ctx, return_inverse=True)
        grouped = torch.zeros(len(uq), C.D, dtype=A.DT)
        grouped.index_add_(0, ix, signals['P2'])
        signals['P2ctx'] = h@(grouped.T@grouped)
        f = m.head(h)
        p = f.softmax(-1)
        qt = torch.from_numpy(C.ChainM(seed, task, .01).P)
        q = qt[x[:, :-1], x[:, 1:]].reshape(-1, C.V)
        vc = m.head.weight.detach()-m.head.weight.detach().mean(0)
        signals['switch_body'] = (p-1/C.V)@vc/n
        signals['fit_body'] = (p-q)@vc/n
        direct = {'switch': (p-1/C.V).T@h/n, 'fit': (p-q).T@h/n}
    gs = {name: {k: torch.zeros_like(w) for k, w in params.items()} for name in [*signals, 'switch', 'fit']}
    for start, xx in A.chunks(x):
        hh = m.body(xx)[:, 1:].reshape(-1, C.D)
        lo, hi = start*(C.T-1), start*(C.T-1)+len(hh)
        for name, signal in signals.items():
            gg = torch.autograd.grad(hh, list(params.values()), grad_outputs=signal[lo:hi], retain_graph=True, allow_unused=True)
            for (key, _), v in zip(params.items(), gg):
                if v is not None:
                    gs[name][key] += v.detach()
        ff = m.head(hh)
        losses = {'switch': (ff.logsumexp(-1)-ff.mean(-1)).sum()/n,
                  'fit': -(q[lo:hi]*ff.log_softmax(-1)).sum()/n}
        for name, loss in losses.items():
            gg = torch.autograd.grad(loss, list(params.values()), retain_graph=True, allow_unused=True)
            for (key, _), v in zip(params.items(), gg):
                if v is not None:
                    gs[name][key] += v.detach()
    checks = {}
    for error in ['switch', 'fit']:
        gs[error+'_head'] = {k: torch.zeros_like(w) for k, w in params.items()}
        gs[error+'_head']['wte.weight'] = direct[error]
        for key in params:
            check(checks, error+'_head_body_'+key, gs[error][key], gs[error+'_body'][key]+gs[error+'_head'][key])
    stats = dict(n_tokens=n, rank=ranks['1e-10'], rank_1e8=ranks['1e-08'], rank_1e12=ranks['1e-12'],
                 pmax=float(p.max(-1).values.mean()), ce=float(-(q*f.log_softmax(-1)).sum(-1).mean()),
                 condition=float(ev[-1]/ev[ev > ev[-1]*1e-10].min()))
    return gs, stats, checks


class Moments(torch.nn.Module):
    def __init__(self, m, raw):
        super().__init__()
        self.model = m
        self.raw = raw

    def forward(self, x):
        capture = []
        self.model.body(x, capture)
        vals = []
        for l in [0, 1]:
            z = capture[2*l+1]
            vals += [z.mean((0, 1)), (z-self.raw[l]['mu']).square().mean((0, 1))]
        return torch.stack(vals)


def moment_jvps(m, seed, raw, directions, batch=4):
    """Independent parameter tangents are vectorized, without learning updates."""
    wrapper = Moments(m, raw)
    params = {f'model.{k}': w.detach() for k, w in m.named_parameters()}
    x = A.uniform(seed)
    result = torch.zeros(len(directions), 4, C.FF, dtype=A.DT)
    for start in range(0, len(directions), batch):
        dd = directions[start:start+batch]
        tangent = {f'model.{k}': torch.stack([g[k] for g in dd]) for k, _ in m.named_parameters()}
        for _, xx in A.chunks(x):
            def fn(pp):
                return torch.func.functional_call(wrapper, pp, (xx,), tie_weights=True)
            def one(tt):
                return torch.func.jvp(fn, (params,), (tt,))[1]
            with torch.no_grad():
                value = torch.vmap(one)(tangent)
            result[start:start+len(dd)] += value.detach()*len(xx)/len(x)
    # The mu in the centered second moment is frozen; the correction for unbiased sd is exact.
    out = []
    n = len(x)*C.T
    for value in result:
        layers = []
        for l in [0, 1]:
            r = raw[l]
            dmu, dvar = value[2*l], value[2*l+1]*n/(n-1)
            ds = dvar/(2*r['s'])
            mean, width = dmu/r['s'], -r['mu']*ds/r['s']**2
            layers.append(dict(mu=dmu, sd=ds, mean=mean, width=width, r=mean+width))
        out.append(layers)
    return out


def projections(m, seed, raw, gs, den, membership):
    params = dict(m.named_parameters())
    zeros = {k: torch.zeros_like(w) for k, w in params.items()}
    tags, dirs = [], []
    names = SIGNALS+[s for s in gs if s.startswith('P2tol')]
    for metric in METRICS:
        for name in names:
            gg = gs[name] if metric == 'SGD' else {k: v/den[k] for k, v in gs[name].items()}
            for route in ['full', *GROUPS]:
                tags.append((metric, name, route))
                dirs.append(gg if route == 'full' else {k: gg[k] if membership[k] == route else zeros[k] for k in params})
        for name in ['switch_body', 'fit_body', 'switch_head', 'fit_head']:
            gg = gs[name] if metric == 'SGD' else {k: v/den[k] for k, v in gs[name].items()}
            tags.append((metric, name, 'full'))
            dirs.append(gg)
    pp = dict(zip(tags, moment_jvps(m, seed, raw, dirs)))
    checks = {}
    for metric in METRICS:
        for name in names:
            for layer in [0, 1]:
                tag = f'L{layer}_{metric}_{name}'
                full = pp[metric, name, 'full'][layer]
                for target in ['mu', 'sd', 'mean', 'width', 'r']:
                    summed = sum(pp[metric, name, group][layer][target] for group in GROUPS)
                    check(checks, tag+'_'+target+'_carrier_sum', full[target], summed)
                gg = gs[name][f'h.{layer}.mlp.up.weight']
                if metric == 'Adam_fixed':
                    gg = gg/den[f'h.{layer}.mlp.up.weight']
                analytic = (gg*raw[layer]['kr']).sum(1)
                check(checks, tag+'_own_analytic', pp[metric, name, f'up{layer}'][layer]['r'], analytic)
                check(checks, tag+'_mean_width', full['r'], full['mean']+full['width'])
                # True causal exclusion: subsequent parameters cannot alter this preactivation.
                excluded = ['down0', 'attention1', 'ln1', 'up1', 'downstream'] if layer == 0 else ['downstream']
                for group in excluded:
                    assert torch.count_nonzero(pp[metric, name, group][layer]['r']) == 0, (tag, group)
        for error in ['switch', 'fit']:
            for layer in [0, 1]:
                for target in ['mu', 'sd', 'mean', 'width', 'r']:
                    check(checks, f'L{layer}_{metric}_{error}_{target}_head_body',
                          pp[metric, error, 'full'][layer][target],
                          pp[metric, error+'_body', 'full'][layer][target]+pp[metric, error+'_head', 'full'][layer][target])
    return pp, checks


def finite_gate(m, seed, task, raw, gs, rank):
    base = {k: w.detach().clone() for k, w in m.named_parameters()}
    params = dict(m.named_parameters())
    records = []
    for name in ['P2', 'switch']:
        gn = sum(v.square().sum() for v in gs[name].values()).sqrt()
        direction = {k: v/gn for k, v in gs[name].items()}
        pred = moment_jvps(m, seed, raw, [direction], batch=1)[0]
        errors = {0: [], 1: [], 'capacity': []}
        for eps in [1e-4, 1e-5, 1e-6]:
            vals, capacities = [], []
            for sign in [-1, 1]:
                with torch.no_grad():
                    for k, w in params.items():
                        w.copy_(base[k]+sign*eps*direction[k])
                vals.append(E.fields(m, seed))
                if name == 'P2':
                    with torch.no_grad():
                        hh = torch.cat([m.body(xx)[:, 1:].reshape(-1, C.D) for _, xx in A.chunks(A.data(seed, task)[0])])
                        ev = torch.linalg.eigvalsh(hh.T@hh)
                        assert int((ev > ev[-1]*1e-10).sum()) == rank
                        capacities.append(float(ev[-rank:].log().sum()/2))
            with torch.no_grad():
                for k, w in params.items():
                    w.copy_(base[k])
            for layer in [0, 1]:
                fd = (vals[1][layer]['r']-vals[0][layer]['r'])/(2*eps)
                error = rel(fd, pred[layer]['r'])
                errors[layer].append(error)
                records.append(dict(signal=name, epsilon=eps, layer=layer, error=float(error)))
            if name == 'P2':
                fd = (capacities[1]-capacities[0])/(2*eps)
                error = abs(fd-float(gn))/max(abs(fd), float(gn))
                errors['capacity'].append(error)
                records.append(dict(signal='P2_capacity', epsilon=eps, layer=-1, error=error))
        assert all(min(vals) <= 1e-4 for vals in errors.values() if vals), errors
    assert all(torch.equal(w, base[k]) for k, w in params.items())
    return records


def old_reference(seed, task, raw, pp):
    path = M.DATA/'chargpt_earlyphase_1024/results/chargpt_earlyphase_1024/raw'/f'C3_seed{seed}_task{task}.npz'
    if not path.exists():
        return None
    checks = {}
    with np.load(path) as old:
        for layer in [0, 1]:
            assert np.array_equal(arr(raw[layer]['alive']), old[f'L{layer}_alive'])
            check(checks, f'L{layer}_r', raw[layer]['r'], torch.from_numpy(old[f'L{layer}_r']))
            for metric in METRICS:
                for signal, oldname in [('P2', 'P2'), ('switch', 'switch')]:
                    check(checks, f'L{layer}_{metric}_{signal}', pp[metric, signal, f'up{layer}'][layer]['r'],
                          torch.from_numpy(old[f'L{layer}_{metric}_r_{oldname}']))
    return dict(path=str(path), sha256=M.sha(path), checks=checks)


def state(inp, seed, task, gate):
    start = time.monotonic()
    m, den, membership = load(inp, seed, task)
    raw = E.fields(m, seed)
    gs, stats, checks = gradients(m, seed, task)
    pp, cc = projections(m, seed, raw, gs, den, membership)
    checks.update(cc)
    reference = old_reference(seed, task, raw, pp)
    finite = finite_gate(m, seed, task, raw, gs, stats['rank']) if gate else None
    arrays = {}
    for layer in [0, 1]:
        for key in ['r', 'mu', 's', 'p', 'alive']:
            arrays[f'L{layer}_{key}'] = arr(raw[layer][key])
    for (metric, signal, route), layers in pp.items():
        for layer, values in enumerate(layers):
            for target, value in values.items():
                arrays[f'L{layer}_{metric}_{signal}_{route}_{target}'] = arr(value)
    tag = f'C3_seed{seed}_task{task}'
    path = OUT/'raw'/f'{tag}.npz'
    np.savez_compressed(path, **arrays)
    meta = dict(seed=seed, task=task, stats=stats, checks=checks, finite=finite, reference=reference,
                parameter_groups=membership, seconds=time.monotonic()-start,
                raw_sha256=M.sha(path), code_sha256=M.sha(Path(__file__)), spec_sha256=M.sha(SPEC))
    dump(path.with_suffix('.json'), meta)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    (OUT/'raw').mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    assert M.sha(M.BUNDLE) == M.BUNDLE_SHA
    inp = E.Inputs()
    code = M.sha(Path(__file__))
    selected = [(0, t) for t in [1, 9, 100]] if args.gate else [(s, t) for s in range(10) for t in TASKS]
    if not args.gate:
        gg = json.loads((OUT/'gate.json').read_text())
        assert gg['passed'] and gg['code_sha256'] == code
    records = []
    for seed, task in selected:
        path = OUT/'raw'/f'C3_seed{seed}_task{task}.json'
        if not args.gate and path.exists():
            meta = json.loads(path.read_text())
            assert meta['code_sha256'] == code
            assert meta['raw_sha256'] == M.sha(path.with_suffix('.npz'))
        else:
            meta = state(inp, seed, task, args.gate)
        records.append(meta)
        print(f'seed{seed} task{task} complete; state {meta["seconds"]:.1f}s; elapsed {time.monotonic()-start:.1f}s', flush=True)
        A.CACHE.clear()
    if args.gate:
        dump(OUT/'gate.json', dict(passed=True, code_sha256=code, states=records, inputs=inp.used,
                                  seconds=time.monotonic()-start))
    else:
        dump(OUT/'checks.json', records)
        gate = json.loads((OUT/'gate.json').read_text())
        dump(OUT/'provenance.json', dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             code_sha256=code, spec_sha256=M.sha(SPEC), states=len(records), seeds=10, tasks=TASKS,
             inputs={**gate['inputs'], **inp.used}, bundle_path=str(M.BUNDLE), bundle_sha256=M.BUNDLE_SHA,
             source_sha256={str(p.relative_to(ROOT)):M.sha(p) for p in [Path(E.__file__),Path(A.__file__)]},
             training_steps=0, optimizer_created=False, checkpoint_extracted=False,
             device='cpu', dtype='float64', threads=torch.get_num_threads(), seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
