"""clamp_drift_0912: does the width clamp RAISE the drift?  (spec_clamp_drift_0912)

Two levers improve/worsen plasticity and rho does not connect them:
  N1  (pre-Adam noise c=1): rho 38.1, L 2.62 pt
  N0w (width clamp):        rho 38.4, L 0.60 pt

drift2 -- the part of the realised weight change that lies along the raw (pre-noise)
gradient of that step -- was only introduced on 9/11 and has never been measured on a
clamped arm.  This run measures it, splitting the update in two places:

    Delta_adam  = the Adam step                        (before the clamp projection)
    Delta_net   = what actually moved the weights      (after the projection)
    clamp_removed = Delta_adam - Delta_net             (exactly the projection)

Everything else -- the injector, the Adam update, the clamp -- is the committed code
(grad_coherence_0911 / clamp_horizon_0910), so all four arms must reproduce their
existing trajectories bit for bit (G1); the added projection maths is read-only.
"""
from pathlib import Path
import argparse, concurrent.futures, json, math, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from src import width_sink_clamp_0909 as C
from src import clamp_horizon_0910 as CH
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import grad_coherence_0911 as GC

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/clamp_drift_0912'
SMOKE = ROOT / 'results/_smoke_clamp_drift_0912'
TASKS = 120
CLAMP_FROM = 21
LATE = (61, 120)
LR0 = .001
TIME_CAP = 1800.
SPEC = ROOT / 'specs/spec_clamp_drift_0912.md'

# name -> (noise c, clamp)   -- every arm has a committed anchor
ARMS = {'N0': (0.0, 'ref'), 'N0w': (0.0, 'wclamp'), 'N1': (1.0, 'ref'), 'N1w': (1.0, 'wclamp')}


def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


def split(dW, ghat):
    """(along g, orthogonal) energies per row."""
    dr = (dW * ghat).sum(1, keepdim=True) * ghat
    return (dr ** 2).sum(1), ((dW - dr) ** 2).sum(1)


def loop(p, act, adam, gens, ngen, mnist, probe, t_from, t_to, cfg, base, rows, units, ck,
         instrument=True, mut=None, cfrom=CLAMP_FROM):
    c_noise, clamp = cfg
    gp, gd, gb = gens
    lr = LR0
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        ce0 = probe_ce(p, act, probe, perm)
        late = instrument and LATE[0] <= task <= LATE[1]
        if late:
            z = lambda: torch.zeros(100, dtype=torch.float64)
            S2, dA, fA, dN, fN, rem, remd = z(), z(), z(), z(), z(), z(), z()
            graw2, kap2 = z(), z()
            tot = torch.zeros(100, 784, dtype=torch.float64)
            Wt0 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double()).clone()
        ce20 = None
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                Wb = p[0].detach().double().clone() if late else None
                if late:
                    g_raw = gr[0].double().clone()
                    graw2 += (g_raw ** 2).sum(1)
                if c_noise > 0:
                    raw0 = gr[0]
                    gr = list(gr)
                    gr[0] = GC.inject(gr[0], c_noise, ngen)
                    if ck is not None and step % 100 == 0:
                        rel = float((gr[0] - raw0).double().norm() / raw0.double().norm().clamp(min=1e-300))
                        ck['g4_inj_sum'] = ck.get('g4_inj_sum', 0.) + rel
                        ck['g4_inj_n'] = ck.get('g4_inj_n', 0) + 1
                        ck['g4_inj_lo'] = min(ck.get('g4_inj_lo', 9e9), rel)
                        ck['g4_inj_hi'] = max(ck.get('g4_inj_hi', 0.), rel)
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                W_adam = p[0].detach().double().clone() if late else None
                if clamp != 'ref' and task >= cfrom:
                    CH.apply_clamp(p, clamp, base, ck, mut)
                    CH.verify_clamped(p, clamp, base, ck)
                if late:
                    gn = g_raw.norm(dim=1, keepdim=True).clamp(min=1e-300)
                    ghat = g_raw / gn
                    dw_a = W_adam - Wb                       # the Adam step
                    dw_n = p[0].detach().double() - Wb       # what actually moved the weights
                    a1, b1 = split(dw_a, ghat); a2, b2 = split(dw_n, ghat)
                    dA += a1; fA += b1; dN += a2; fN += b2
                    cut = dw_a - dw_n                        # exactly the clamp projection
                    rem += (cut ** 2).sum(1)
                    remd += split(cut, ghat)[0]
                    kap2 += ((dw_a / lr) ** 2).mean(1)
                    dwt = dw_n - dw_n.mean(1, keepdim=True)
                    S2 += (dwt ** 2).sum(1); tot += dwt
                    ck['g2_ident'] = max(ck.get('g2_ident', 0.), float(
                        ((a2 + b2) - (dw_n ** 2).sum(1)).abs().max() / max(float((dw_n ** 2).sum(1).max()), 1e-300)))
                    if clamp == 'ref':
                        ck['g2_clamp_zero'] = max(ck.get('g2_clamp_zero', 0.), float(cut.abs().max()))
            if step == 20:
                ce20 = probe_ce(p, act, probe, perm)
        if rows is None:
            continue
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, _ = GS.gate_block(p, act, probe, perm)
        r.update(gr_)
        r.update(task=task, step=625, clamp=clamp, ce0=ce0, ce20=ce20)
        with torch.no_grad():
            r['w2_fro'] = float(p[2].double().norm())
        for k, v in list(u.items()) + list(gu.items()):
            units[f'ref_{k}_t{task}'] = v     # grad_coherence_0911 labels these 'ref_' for every arm
        if late:
            D2 = (tot ** 2).sum(1)
            rho = torch.where(S2 > 0, D2 / S2.clamp(min=1e-300), torch.full_like(S2, float('nan')))
            Wt1 = (lambda W: W - W.mean(1, keepdim=True))(p[0].detach().double())
            ck['g2_tot_ident'] = max(ck.get('g2_tot_ident', 0.), float((tot - (Wt1 - Wt0)).abs().max()))
            n = 625 * lr ** 2
            r.update(S2=float(S2.mean()), D2=float(D2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625),
                     drift2_adam=float(dA.mean() / n), diff2_adam=float(fA.mean() / n),
                     drift2_net=float(dN.mean() / n), diff2_net=float(fN.mean() / n),
                     f_diff_net=float(fN.sum() / (dN + fN).sum()),
                     clamp_removed=float(rem.mean() / n),
                     clamp_removed_drift=float(remd.mean() / n),
                     clamp_drift_share=float(remd.sum() / rem.sum()) if float(rem.sum()) > 0 else 0.)
            units[f'ref_drift2_net_i_t{task}'] = (dN / n).numpy().copy()
        rows.append(r)
    return p, act, adam


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, clamp_from=CLAMP_FROM):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    c_noise, clamp = ARMS[arm]
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi('LR'))
    assert ck['g2_dphi'] < 1e-6, ('phi-prime', ck)
    ck.update(noise_c=c_noise, clamp=clamp, lr=LR0)
    if c_noise > 0 and controls:
        ck.update(GC.check_unbiased(c_noise, seed))
        assert ck['g2_mean_rel'] < ck['g2_mean_tol'] and ck['g2_var_rel'] < ck['g2_var_tol'], ('G4 unbiased', ck)
        assert ck['g2_mean_rel_mutctl'] > 3 * ck['g2_mean_tol'], ('vacuous control', ck)

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act('LR')
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ngen = H.stream('gradnoise', seed)
    st0 = [g.get_state().clone() for g in gens]
    base = None
    if clamp != 'ref':
        loop(p, act, adam, gens, ngen, mnist, probe, 1, clamp_from - 1, (c_noise, clamp), None,
             rows, units, ck, cfrom=clamp_from)
        base = C.baselines(p[0])
        ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
        if controls:
            ck.update(_controls(arm, seed, C.snapshot(p, act, adam, gens), base, mnist, probe, ck, clamp_from))
        loop(p, act, adam, gens, ngen, mnist, probe, clamp_from, tasks, (c_noise, clamp), base,
             rows, units, ck, cfrom=clamp_from)
    else:
        loop(p, act, adam, gens, ngen, mnist, probe, 1, tasks, (c_noise, clamp), None, rows, units, ck)
    T.finite_guard(rows, tag)

    g2s = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ck['g4_streams_intact'] = all(torch.equal(a, b.get_state()) for a, b in zip(st0, g2s))
    assert ck['g4_streams_intact']
    if c_noise > 0:
        ck['g4_inj_mean'] = ck['g4_inj_sum'] / ck['g4_inj_n']
        assert 0.98 * c_noise <= ck['g4_inj_mean'] <= 1.02 * c_noise, ('G4 injection mean', ck['g4_inj_mean'])
    else:
        ck['g4_ngen_untouched'] = bool(torch.equal(H.stream('gradnoise', seed).get_state(), ngen.get_state()))
        assert ck['g4_ngen_untouched']
    assert ck.get('g2_ident', 0.) <= 1e-10, ('G2 projection identity', ck['g2_ident'])
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck['g2_tot_ident'])
    if clamp == 'ref':
        assert ck.get('g2_clamp_zero', 0.) == 0., ('unclamped arm shows a projection', ck['g2_clamp_zero'])
    else:
        for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff'):
            assert k in ck and ck[k] <= CH.limit(ck, k, 'f32_rowmean_bound_t20'), ('G3', k, ck.get(k))

    # ---- G1: every arm reproduces a committed trajectory bit for bit
    if arm == 'N0':
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        w, n = 0., 0
        for t in range(1, min(tasks, 120) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
        exp = 3 * min(tasks, 120); anchor = 'elu_growth_0909/LR'
    else:
        ref = np.load(ROOT / 'results/grad_coherence_0911' / f'{arm}_s{seed}_units.npz')
        # the anchor was produced with CLAMP_FROM; if this run clamps earlier (smoke) only the
        # shared pre-clamp prefix can match.
        g1_to = tasks if clamp_from == CLAMP_FROM else min(tasks, clamp_from - 1)
        w, n = 0., 0
        for key in units:
            if key in ref and int(key.rsplit('_t', 1)[1]) <= g1_to:
                w = max(w, float(np.abs(units[key].astype(float) - ref[key].astype(float)).max())); n += 1
        exp = None; anchor = f'grad_coherence_0911/{arm}'
    ck.update(g1_anchor=anchor, g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=exp)
    assert w <= 1e-10, ('G1 failed', arm, seed, w)
    assert n > 0 and (exp is None or n == exp), ('G1 count guard', n, exp)
    if exp is None:
        assert n >= 6 * g1_to, ('G1 compared too few arrays', n, g1_to)
    if controls and seed == 0 and arm == 'N0':
        p3 = H.init_params(seed, torch.device('cpu'))
        with torch.no_grad():
            p3[0].data[0, 0] += 1e-3
        ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        loop(p3, GS.make_act('LR'), ad3, g3, H.stream('gradnoise', seed), mnist, probe, 1, 1,
             (0.0, 'ref'), None, [], u3, {}, instrument=False)
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
        assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, clamp_from=clamp_from, cfg=dict(c=c_noise, clamp=clamp, lr=LR0),
                late=LATE, checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                grad_coherence_sha256=T.sha(Path(GC.__file__)), clamp_horizon_sha256=T.sha(Path(CH.__file__)),
                base_sha256=T.sha(Path(C.__file__)), host_sha256=T.sha(Path(H.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; drift2 split before/after the width-clamp projection'))
    print('FINISHED', tag, 'G1', ck['g1_units_maxabs'], f"({n})", round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def _controls(arm, seed, snap, base, mnist, probe, ck_main, clamp_from):
    """The committed clamp's mutation controls (skip / rowscale / tail), run at t20."""
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if key in CH.FLOOR_OF else None
        return 100 * CH.limit(ck_main, key, b)
    for mut, keys in [('skip', ['c3_cnorm_rel']), ('rowscale', ['c3_m_absdiff']), ('tail', ['clamp_tail_absdiff'])]:
        p2, act2, adam2, gens2 = C.restore(snap, 'LR')
        ck2 = {}
        if mut == 'skip':
            gp, gd, gb = gens2
            perm = torch.randperm(784, generator=gp)
            idx = H.stratified_draw(mnist, gd)
            order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
            xs = mnist.train_x[idx][:, perm][order]; ys = mnist.train_y[idx][order]
            for step in range(1, 626):
                out = H.forward(p2, xs[(step - 1) * 16:step * 16], act2)
                g = torch.autograd.grad(torch.nn.functional.cross_entropy(
                    out[4], ys[(step - 1) * 16:step * 16]), p2)
                with torch.no_grad():
                    m_, v_, tc = adam2
                    tc[0] += 1; c1, c2 = 1 - .9 ** tc[0], 1 - .999 ** tc[0]
                    for q, gg, mi, vi in zip(p2, g, m_, v_):
                        mi.mul_(.9).add_(gg, alpha=.1); vi.mul_(.999).addcmul_(gg, gg, value=.001)
                        q -= LR0 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                    if step != 300:
                        CH.apply_clamp(p2, 'wclamp', base, ck2)
                    CH.verify_clamped(p2, 'wclamp', base, ck2)
        else:
            CH.loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, clamp_from, 'wclamp',
                    base, None, {}, ck2, mut=mut)
        for k in keys:
            got = ck2.get(k, 0.)
            res[f'ctl_{mut}_{k}'] = got
            assert got > floor(k), (f'vacuous control {mut}: {k} = {got:g} <= {floor(k):g}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        global LATE
        LATE = (3, 4)
        for arm in ARMS:
            run(arm, 0, tasks=4, out=SMOKE, controls=True, clamp_from=3)
        return
    if a.all:
        def job(j):
            subprocess.run([sys.executable, '-m', 'src.clamp_drift_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
