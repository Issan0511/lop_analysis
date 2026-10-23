#!/usr/bin/env python3
"""Matched early phase-1/phase-2 conditional predictors; frozen checkpoints only."""
import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location('phase2_parent', ROOT/'analysis/chargpt_l0phase2_1023/measure.py')
Q = importlib.util.module_from_spec(sp)
sp.loader.exec_module(Q)
torch, np, M, A, C = Q.torch, Q.np, Q.M, Q.A, Q.C
OUT = ROOT/'results/chargpt_earlyphase_1024'
OLD = M.DATA/'chargpt_l0phase2_1023/results/chargpt_l0phase2_1023'
SPEC = ROOT/'specs/spec_chargpt_earlyphase_1024.md'
TOLS = [1e-8, 1e-10, 1e-12]


def dump(path, obj):
    M.dump(path, obj)


def csvwrite(path, rows):
    Q.csvwrite(path, rows)


def rel(a, b):
    return M.relmax(a-b, a, b)


class Inputs(M.Inputs):
    def __init__(self):
        super().__init__()
        self.used = {}

    def read(self, name):
        data = self.tar.extractfile(name).read()
        self.used[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)

    def load(self, arm, seed, task):
        sd = self.read(f'results/chargpt_cl_1019/M2/{arm}/seed{seed}/ckpts/task{task}.pt')
        C.L = 2
        m = C.GPT(arm, 'M')
        m.load_state_dict(sd)
        m.eval().double()
        for block in m.h:
            block.mlp.track_v = False
        assert m.wte.weight is m.head.weight
        return m

    def adam(self, m, arm, seed, task):
        sd = self.read(f'results/chargpt_cl_1019/M2/{arm}/seed{seed}/ckpts/task{task}_adam.pt')
        group = sd['param_groups'][0]
        params = list(m.named_parameters())
        assert len(params) == len(group['params'])
        den = {}
        for index, (name, param) in zip(group['params'], params):
            for layer in [0, 1]:
                if name == f'h.{layer}.mlp.up.weight':
                    state = sd['state'][index]
                    assert state['exp_avg_sq'].shape == param.shape
                    k = float(state['step'])
                    den[layer] = (state['exp_avg_sq'].double()/(1-group['betas'][1]**k)).sqrt()+group['eps']
        assert set(den) == {0, 1}
        return den


@torch.no_grad()
def fields(m, seed):
    raws = []
    for layer, (xf, z) in enumerate(A.moments(m, A.uniform(seed))):
        w = m.h[layer].mlp.up.weight.detach()
        assert m.h[layer].mlp.up.bias is None
        mean = xf.mean(0)
        mu = w@mean
        s = z.std(0)
        xc = xf-mean
        cov = xc.T@xc/(len(xf)-1)
        kr = mean/s[:, None]-mu[:, None]*(w@cov)/s[:, None]**3
        assert float((z-xf@w.T).abs().max()) < 1e-9
        assert (s > 0).all()
        raws.append(dict(W=w.clone(), m=mean, mu=mu, s=s, r=mu/s, cov=cov, kr=kr,
                         alive=(z > 0).any(0), p=(z > 0).double().mean(0)))
    return raws


def compare(x, y, mask):
    x, y, mask = Q.arr(x), Q.arr(y), Q.arr(mask).astype(bool)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    a, b = x[mask], y[mask]
    pos, neg = b > 0, b < 0
    ba = ((a[pos] > 0).mean()+(a[neg] < 0).mean())/2 if pos.sum() >= 10 and neg.sum() >= 10 else None
    return dict(n=len(a), n_sink=int(pos.sum()), n_float=int(neg.sum()),
                agreement=float((np.sign(a) == np.sign(b)).mean()) if len(a) else None,
                ba=None if ba is None else float(ba), rho=Q.rho(a, b),
                pred_sink=float((a > 0).mean()) if len(a) else None,
                actual_sink=float(pos.mean()) if len(a) else None)


def algebra_gate():
    gen = torch.Generator().manual_seed(1024001)
    h = torch.randn(13, 5, generator=gen, dtype=A.DT)
    gi = torch.linalg.inv(h.T@h)
    ctx = torch.arange(13) % 4
    checks = {}
    for name, y in [('iid', torch.eye(13, dtype=A.DT)), ('ctx', torch.nn.functional.one_hot(ctx, 4).double())]:
        c = y@y.T
        v1, v2 = h.T@y, gi@h.T@y
        # Explicit sum over label covariance factors, independent of formula.
        p1 = sum(y[:, k, None]*v1[:, k][None, :] for k in range(y.shape[1]))
        p2 = sum((h@v2[:, k])[:, None]*v2[:, k][None, :] for k in range(y.shape[1]))
        checks[name+'_P1'] = rel(p1, c@h)
        checks[name+'_P2'] = rel(p2, h@gi@h.T@c@h@gi)
    assert max(checks.values()) < 1e-10
    return checks


def state(inp, arm, seed, task, gate=False):
    m = inp.load(arm, seed, task)
    raw = fields(m, seed)
    den = inp.adam(m, arm, seed, task)
    x = A.data(seed, task)[0]
    n = len(x)*(C.T-1)
    qt = torch.from_numpy(C.ChainM(seed, task, .01).P)
    with torch.no_grad():
        h = torch.cat([m.body(xx)[:, 1:].reshape(-1, C.D) for _, xx in A.chunks(x)])
        vc = m.head.weight.detach()-m.head.weight.detach().mean(0)
        f = h@m.head.weight.detach().T
        p = f.softmax(-1)
        q = qt[x[:, :-1], x[:, 1:]].reshape(-1, C.V)
        e, target = p-1/C.V, q-1/C.V
        gram = h.T@h
        val, vec = torch.linalg.eigh(gram)
        ranks = {tol: int((val > val[-1]*tol).sum()) for tol in TOLS}
        keep = val > val[-1]*1e-10
        v = vec[:, keep]
        gi = (v/val[keep])@v.T
        context = (x[:, :-1]*C.V+x[:, 1:]).reshape(-1)
        uq, ix = torch.unique(context, return_inverse=True)
        grouped_h = torch.zeros(len(uq), C.D, dtype=A.DT)
        grouped_h.index_add_(0, ix, h)
        dual = (h@v/val[keep])@v.T
        grouped_dual = torch.zeros_like(grouped_h)
        grouped_dual.index_add_(0, ix, dual)
        ff = target.T@h
        kappa = (vc*ff).sum()/ff.square().sum()
        vl = kappa*ff
        signals = dict(P1=h, P2=h@gi, P1ctx=grouped_h[ix], P2ctx=h@(grouped_dual.T@grouped_dual),
                       switch=(e@vc)/n, fit=((p-q)@vc)/n,
                       load_actual=(e@vl)/n, load_q=(target@vl)/n, residual=(e@(vc-vl))/n)
        for tol in [1e-8, 1e-12]:
            if ranks[tol] != ranks[1e-10]:
                kk = val > val[-1]*tol
                vv = vec[:, kk]
                signals[f'P2tol{tol:g}'] = h@((vv/val[kk])@vv.T)
        hn_fit, hn_switch = (p-q).T@h, e.T@h
        error_cos = (e*target).sum(-1)/(e.norm(dim=-1)*target.norm(dim=-1)).clamp_min(1e-300)
        stats = dict(n_tokens=n, rank=ranks[1e-10], rank_tol1e8=ranks[1e-8], rank_tol1e12=ranks[1e-12],
                     condition=float(val[-1]/val[keep].min()),
                     pmax=float(p.max(-1).values.mean()), qmax=float(q.max(-1).values.mean()),
                     Hq=float(-(q*q.clamp_min(1e-300).log()).sum(-1).mean()),
                     KL=float((q*(q.clamp_min(1e-300).log()-f.log_softmax(-1))).sum(-1).mean()),
                     error_cos_mean=float(error_cos.mean()), kappa_q=float(kappa),
                     cos2_head_q=float((vc*ff).sum().square()/(vc.square().sum()*ff.square().sum())),
                     head_fit_ratio=float(hn_fit.norm()/hn_switch.norm()),
                     final_gamma_cv=float(m.nf.weight.std()/m.nf.weight.abs().mean()))
    params = [b.mlp.up.weight for b in m.h]
    grads = {name: [torch.zeros_like(w) for w in params] for name in [*signals, 'local_self']}
    checks = {}
    for start, xx in A.chunks(x):
        cap = []
        hh = m.body(xx, cap)
        hs = hh[:, 1:].reshape(-1, C.D)
        lo, hi = start*(C.T-1), (start+len(xx))*(C.T-1)
        for name, signal in signals.items():
            gs = torch.autograd.grad(hs, params, grad_outputs=signal[lo:hi], retain_graph=True)
            for accum, value in zip(grads[name], gs):
                accum.add_(value.detach())
        with torch.no_grad():
            for layer in [0, 1]:
                xf = m.h[layer].n2(cap[2*layer].detach())[:, 1:].reshape(-1, C.D)
                z = cap[2*layer+1].detach()[:, 1:].reshape(-1, C.FF)
                mlp = m.h[layer].mlp
                grads['local_self'][layer] += (mlp.phi(z)*mlp.dphi(z)).T@xf
        if gate:
            if 'direct_P1' not in grads:
                for name in ['direct_P1', 'direct_switch', 'direct_fit', 'direct_local_self']:
                    grads[name] = [torch.zeros_like(w) for w in params]
            logits = m.head(hh)[:, 1:]
            cq = q[lo:hi].reshape_as(logits)
            losses = dict(direct_P1=hs.square().sum()/2,
                          direct_switch=(logits.logsumexp(-1)-logits.mean(-1)).sum()/n,
                          direct_fit=-(cq*logits.log_softmax(-1)).sum()/n)
            for name, loss in losses.items():
                gs = torch.autograd.grad(loss, params, retain_graph=True)
                for accum, value in zip(grads[name], gs):
                    accum.add_(value.detach())
            for layer, w in enumerate(params):
                # Grad only with respect to the same layer's row (not other layer's activity).
                loss = m.h[layer].mlp.phi(cap[2*layer+1][:, 1:]).square().sum()/2
                grads['direct_local_self'][layer] += torch.autograd.grad(loss, w, retain_graph=True)[0].detach()
    for layer in [0, 1]:
        checks[f'L{layer}_decomposition'] = rel(grads['switch'][layer], grads['load_actual'][layer]+grads['residual'][layer])
        if gate:
            for name in ['P1', 'switch', 'fit', 'local_self']:
                checks[f'L{layer}_{name}_autograd'] = rel(grads[name][layer], grads['direct_'+name][layer])
    assert max(checks.values()) < 1e-9, checks
    if gate:
        for layer in [0, 1]:
            w = params[layer]
            direction = grads['P2'][layer]/grads['P2'][layer].norm()
            ww = w.detach().clone()
            pred = float((grads['P2'][layer]*direction).sum())
            for eps in [1e-4, 1e-5, 1e-6]:
                vals = []
                with torch.no_grad():
                    for sign in [-1, 1]:
                        w.copy_(ww+sign*eps*direction)
                        hp = torch.cat([m.body(xx)[:, 1:].reshape(-1, C.D) for _, xx in A.chunks(x)])
                        eig = torch.linalg.eigvalsh(hp.T@hp)
                        assert int((eig > eig[-1]*1e-10).sum()) == ranks[1e-10]
                        vals.append(eig[-ranks[1e-10]:].log().sum()/2)
                    w.copy_(ww)
                fd = float((vals[1]-vals[0])/(2*eps))
                checks[f'L{layer}_P2_fd_eps{eps:g}'] = abs(fd-pred)/max(abs(pred), 1e-300)
            checks[f'L{layer}_P2_fd'] = checks[f'L{layer}_P2_fd_eps1e-06']
            assert checks[f'L{layer}_P2_fd'] < 1e-4, checks
        mom = A.moments(m, A.uniform(seed))
        for layer, (xf, z) in enumerate(mom):
            d = grads['switch'][layer]/grads['switch'][layer].norm()
            dz = xf@d.T
            zp, zm = z+1e-4*dz, z-1e-4*dz
            fd = (zp.mean(0)/zp.std(0)-zm.mean(0)/zm.std(0))/(2e-4)
            checks[f'L{layer}_r_fd'] = rel(fd, (raw[layer]['kr']*d).sum(1))
            assert checks[f'L{layer}_r_fd'] < 1e-6
    arrays, rows, comparisons = {}, [], []
    names = [k for k in grads if not k.startswith('direct_')]
    for layer in [0, 1]:
        rr = raw[layer]
        for key in ['alive', 'r', 'mu', 's', 'p']:
            arrays[f'L{layer}_{key}'] = Q.arr(rr[key])
        for metric in ['SGD', 'Adam_fixed']:
            for target_name, direction in [('r', rr['kr']), ('mu', rr['m'][None, :].expand_as(rr['W']))]:
                kk = direction if metric == 'SGD' else direction/den[layer]
                pushes = {name: (grads[name][layer]*kk).sum(1) for name in names}
                arrays.update({f'L{layer}_{metric}_{target_name}_{name}':Q.arr(v) for name, v in pushes.items()})
                base = dict(arm=arm, seed=seed, task=task, layer=layer, metric=metric, target=target_name,
                            n_alive=int(rr['alive'].sum()), r_med=Q.med(rr['r']), p_open_med=Q.med(rr['p']))
                for name, sv in pushes.items():
                    row = dict(base, predictor=name, positive=float((sv[rr['alive']] > 0).double().mean()),
                               median=Q.med(sv[rr['alive']]), abs_median=Q.med(sv[rr['alive']].abs()),
                               n_zero=int((sv[rr['alive']] == 0).sum()), gradient_norm=float(grads[name][layer].norm()), **stats)
                    for er in ['switch', 'fit']:
                        mask = rr['alive'] & (sv != 0) & (pushes[er] != 0)
                        row.update({er+'_'+k:v for k,v in compare(sv, pushes[er], mask).items()})
                    rows.append(row)
                for er in ['switch', 'fit']:
                    for family, p1, p2 in [('iid', 'P1', 'P2'), ('ctx', 'P1ctx', 'P2ctx')]:
                        mask = rr['alive'] & (pushes[er] != 0) & (pushes[p1] != 0) & (pushes[p2] != 0)
                        c1, c2 = compare(pushes[p1], pushes[er], mask), compare(pushes[p2], pushes[er], mask)
                        comparisons.append(dict(base, error=er, family=family, n_common=int(mask.sum()),
                            **{'p1_'+k:v for k,v in c1.items()}, **{'p2_'+k:v for k,v in c2.items()},
                            delta_ba=c2['ba']-c1['ba'] if c2['ba'] is not None and c1['ba'] is not None else None))
    if gate:
        with np.load(OLD/'raw'/f'headcov_{arm}_seed{seed}_t9_u10000.npz') as old, np.load(OLD/'raw'/f'scalar_{arm}_seed{seed}_t9_u10000.npz') as scalar:
            assert np.array_equal(arrays['L0_alive'], old['alive'])
            for metric in ['SGD', 'Adam_fixed']:
                for name, oldkey, source in [('P2', metric+'_iid', old), ('P2ctx', metric+'_ctx', old),
                                             ('switch', metric+'_switch_r', scalar), ('fit', metric+'_fit_r', scalar)]:
                    got = arrays[f'L0_{metric}_r_{name}']
                    expected = source[oldkey]
                    err = float(np.max(np.abs(got-expected))/max(np.max(np.abs(got)), np.max(np.abs(expected)), 1e-300))
                    checks[f'{metric}_{name}_reference'] = err
                    assert err < 1e-9 and np.array_equal(np.sign(got), np.sign(expected)), checks
            for key in ['r', 'mu', 's']:
                got, expected = arrays['L0_'+key], scalar[key]
                err = float(np.max(np.abs(got-expected))/max(np.max(np.abs(expected)), 1e-300))
                checks[key+'_reference'] = err
                assert err < 1e-9
    return rows, comparisons, arrays, checks


def save(arm, seed, task, result, inp):
    rows, comp, arrays, checks = result
    dest = OUT/'raw'/f'{arm}_seed{seed}_task{task}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)
    dump(dest.with_suffix('.json'), dict(code_sha256=M.sha(__file__), raw_sha256=M.sha(dest),
         rows=rows, comparisons=comp, checks=checks, inputs=inp.used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gate', 'run'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    code_hash = M.sha(__file__)
    start = time.monotonic()
    if args.mode == 'gate':
        assert M.sha(M.BUNDLE) == M.BUNDLE_SHA
        inp = Inputs()
        algebra = algebra_gate()
        checks = []
        for arm in A.ARMS:
            result = state(inp, arm, 0, 9, gate=True)
            save(arm, 0, 9, result, inp)
            checks.append(dict(arm=arm, **result[3]))
            print('gate', arm, 'passed', round(time.monotonic()-start, 1), flush=True)
        dump(OUT/'gate.json', dict(passed=True, code_sha256=code_hash, algebra=algebra, checks=checks,
             bundle_sha256=M.BUNDLE_SHA, seconds=time.monotonic()-start))
        return
    gate = json.loads((OUT/'gate.json').read_text())
    assert gate['passed'] and gate['code_sha256'] == code_hash
    inp = Inputs()
    rows, comparisons, checks = [], [], []
    for seed in range(10):
        for arm in A.ARMS:
            ts = time.monotonic()
            for task in range(1, 10):
                path = OUT/'raw'/f'{arm}_seed{seed}_task{task}.npz'
                if not path.exists():
                    save(arm, seed, task, state(inp, arm, seed, task), inp)
                meta = json.loads(path.with_suffix('.json').read_text())
                assert meta['code_sha256'] == code_hash and meta['raw_sha256'] == M.sha(path)
                rows.extend(meta['rows'])
                comparisons.extend(meta['comparisons'])
                checks.append(dict(arm=arm, seed=seed, task=task, **meta['checks']))
                inp.used.update(meta['inputs'])
            print(arm, seed, '9 states', round(time.monotonic()-ts, 1), 's', flush=True)
        A.CACHE.clear()
        A.UCACHE.clear()
    csvwrite(OUT/'states.csv', rows)
    csvwrite(OUT/'comparisons.csv', comparisons)
    dump(OUT/'checks.json', checks)
    dump(OUT/'provenance.json', dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
         code_sha256=code_hash, spec_sha256=M.sha(SPEC), parent_code_sha256=M.sha(Q.__file__),
         bundle_path=str(M.BUNDLE), bundle_sha256=M.BUNDLE_SHA, inputs=inp.used,
         states=180, training_steps=0, optimizer_created=False, checkpoint_extracted=False,
         dtype='float64', device='cpu', threads=torch.get_num_threads(), seconds=time.monotonic()-start))
    print('complete', len(rows), len(comparisons), round(time.monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    main()
