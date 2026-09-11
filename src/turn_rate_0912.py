"""turn_rate_0912: is the width harm just the turn rate?  (spec_turn_rate_0912)

leaky is positively homogeneous, so a unit's function is set by the DIRECTION of
W~_i; its length can be absorbed downstream.  An Adam step has |dW~_i| ~ lr*sqrt(784)
regardless of that length, so the angle a unit turns per update is

    omega ~ |dW~_i|_perp / |W~_i|  ∝  lr / |W~_i|

A long unit turns less per step and cannot finish turning inside the 625 updates of
a task.  If that single scalar is the whole story, then moving the clamp height kappa
and lr by the SAME factor in OPPOSITE directions leaves omega unchanged, and the two
arms of such a pair must lose the same accuracy despite 2x different lengths.

kappa is applied by scaling base['c'] -- the committed apply_clamp is NOT edited.
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

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/turn_rate_0912'
SMOKE = ROOT / 'results/_smoke_turn_rate_0912'
TASKS = 120
CLAMP_FROM = 21
LATE = (61, 120)
LR0 = .001
TIME_CAP = 1800.
SPEC = ROOT / 'specs/spec_turn_rate_0912.md'

# name -> (kappa, lr);  kappa None = no clamp.   omega ∝ lr/kappa
ARMS = {
    'N0':  (None, LR0),          # reference                      G1: elu_growth_0909/LR
    'W1':  (1.0,  LR0),          # omega 1.0                      G1: grad_coherence_0911/N0w
    'W1h': (1.0,  LR0 / 2),      # omega 0.5   short + slow lr
    'W2':  (2.0,  LR0),          # omega 0.5   LONG  + normal lr   <- pair with W1h
    'W1d': (1.0,  2 * LR0),      # omega 2.0   short + fast lr
    'W05': (0.5,  LR0),          # omega 2.0   SHORT + normal lr   <- pair with W1d
}
PAIRS = (('W1h', 'W2'), ('W1d', 'W05'))


def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


def angles(a, b):
    """Per-row angle between two centred weight matrices (float64, radians)."""
    na = a.norm(dim=1).clamp(min=1e-300); nb = b.norm(dim=1).clamp(min=1e-300)
    cos = ((a * b).sum(1) / (na * nb)).clamp(-1., 1.)
    return torch.arccos(cos), cos, na, nb


def loop(p, act, adam, gens, mnist, probe, t_from, t_to, cfg, base, rows, units, ck,
         instrument=True, mut=None, cfrom=CLAMP_FROM):
    kappa, lr = cfg
    gp, gd, gb = gens
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
            S2, dN, fN, graw2, kap2, dth = z(), z(), z(), z(), z(), z()
            tot = torch.zeros(100, 784, dtype=torch.float64)
            Wt_start = C.rows_of(p[0])[2].clone()
            Wt0 = Wt_start.clone()
        ce20 = None
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                if late:
                    Wb = p[0].detach().double().clone()
                    Wt_b = C.rows_of(p[0])[2].clone()
                    g_raw = gr[0].double().clone()
                    graw2 += (g_raw ** 2).sum(1)
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if kappa is not None and task >= cfrom:
                    CH.apply_clamp(p, 'wclamp', base, ck, mut)
                    CH.verify_clamped(p, 'wclamp', base, ck)
                if late:
                    Wt_a = C.rows_of(p[0])[2]
                    th, cos, na, nb = angles(Wt_b, Wt_a)
                    dth += th
                    ck['g2_cos_range'] = max(ck.get('g2_cos_range', 0.), float((cos.abs() - 1).clamp(min=0).max()))
                    # G2: cross-check the REPORTED angle two ways.  arccos of a dot product
                    # and the chord form are affected by rounding differently, and the chord
                    # form is the stable one at theta ~ 3e-3 rad, so their agreement bounds
                    # the error in what we report.  (The raw chord IDENTITY cannot be checked
                    # at 1e-8: |dW~|^2 loses eps64/theta^2 to cancellation.)
                    # normalise FIRST: ||a_hat - b_hat|| = 2 sin(theta/2) holds for any norms,
                    # so this is valid on the growing (unclamped) arm too.
                    # angles(Wt_b, Wt_a) returns na = ||Wt_b||, nb = ||Wt_a||  -- do not swap.
                    th_c = 2 * torch.arcsin((((Wt_a / nb.unsqueeze(1)) - (Wt_b / na.unsqueeze(1)))
                                             .norm(dim=1) / 2).clamp(max=1.))
                    ck['g2_angle_rel'] = max(ck.get('g2_angle_rel', 0.), float(
                        ((th_c - th).abs() / th.clamp(min=1e-300)).max()))
                    dw_n = p[0].detach().double() - Wb
                    gn = g_raw.norm(dim=1, keepdim=True).clamp(min=1e-300)
                    ghat = g_raw / gn
                    dr = (dw_n * ghat).sum(1, keepdim=True) * ghat
                    dN += (dr ** 2).sum(1); fN += ((dw_n - dr) ** 2).sum(1)
                    kap2 += ((dw_n / lr) ** 2).mean(1)
                    dwt = dw_n - dw_n.mean(1, keepdim=True)
                    S2 += (dwt ** 2).sum(1); tot += dwt
            if step == 20:
                ce20 = probe_ce(p, act, probe, perm)
        if rows is None:
            continue
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, _ = GS.gate_block(p, act, probe, perm)
        r.update(gr_)
        r.update(task=task, step=625, clamp='ref', ce0=ce0, ce20=ce20, lr=lr,
                 kappa=(kappa if kappa is not None else 0.0))   # 0 = no clamp (nan trips finite_guard)
        with torch.no_grad():
            r['w2_fro'] = float(p[2].double().norm())
        for k, v in list(u.items()) + list(gu.items()):
            units[f'ref_{k}_t{task}'] = v
        if late:
            Wt_end = C.rows_of(p[0])[2]
            Th = angles(Wt_start, Wt_end)[0]
            ck['g2_triangle'] = max(ck.get('g2_triangle', 0.), float((Th - dth).max()))
            D2 = (tot ** 2).sum(1)
            rho = torch.where(S2 > 0, D2 / S2.clamp(min=1e-300), torch.full_like(S2, float('nan')))
            Wt1 = C.rows_of(p[0])[2]
            ck['g2_tot_ident'] = max(ck.get('g2_tot_ident', 0.), float((tot - (Wt1 - Wt0)).abs().max()))
            n = 625 * lr ** 2
            om = dth / 625
            r.update(omega_step=float(om.mean()), theta_task=float(Th.mean()),
                     turn_eff=float(Th.sum() / dth.sum()),
                     S2=float(S2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625),
                     drift2_net=float(dN.mean() / n), diff2_net=float(fN.mean() / n))
            units[f'ref_omega_i_t{task}'] = om.numpy().copy()
            units[f'ref_theta_i_t{task}'] = Th.numpy().copy()
        rows.append(r)
    return p, act, adam


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, clamp_from=CLAMP_FROM):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    kappa, lr = ARMS[arm]
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi('LR'))
    assert ck['g2_dphi'] < 1e-6, ('phi-prime', ck)
    ck.update(kappa=(kappa if kappa is not None else 'none'), lr=lr, lr_rel_err=abs(lr / ARMS[arm][1] - 1))

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act('LR')
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    base = None
    if kappa is not None:
        # t1..t20 is run at the REFERENCE lr for every arm, so all arms branch from one
        # state and the clamp height kappa*c_ref is common.  Otherwise a smaller lr would
        # also give a smaller t20 norm, and kappa/lr would not be independent (smoke:
        # W1h came out at 0.76x the reference turn rate instead of 0.50x).
        loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, (kappa, LR0), None,
             rows, units, ck, cfrom=clamp_from)
        b0 = C.baselines(p[0])
        base = dict(b0); base['c'] = b0['c'] * kappa          # kappa enters ONLY here
        ck['kappa_applied'] = float((base['c'] / b0['c']).mean())
        assert abs(ck['kappa_applied'] - kappa) < 1e-12, ('kappa not applied', ck['kappa_applied'])
        ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
        if controls:
            ck.update(_controls(seed, C.snapshot(p, act, adam, gens), base, mnist, probe, ck, clamp_from, lr))
        loop(p, act, adam, gens, mnist, probe, clamp_from, tasks, (kappa, lr), base,
             rows, units, ck, cfrom=clamp_from)
    else:
        loop(p, act, adam, gens, mnist, probe, 1, tasks, (kappa, lr), None, rows, units, ck)
    T.finite_guard(rows, tag)

    assert ck.get('g2_cos_range', 0.) == 0., ('G2 cos outside [-1,1]', ck.get('g2_cos_range'))
    assert ck.get('g2_triangle', -1.) <= 1e-8, ('G2 net turn exceeds path length', ck.get('g2_triangle'))
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck.get('g2_tot_ident'))
    assert ck.get('g2_angle_rel', 1.) <= 1e-6, ('G2 angle cross-check', ck.get('g2_angle_rel'))
    if kappa is not None:
        for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff'):
            assert k in ck and ck[k] <= CH.limit(ck, k, 'f32_rowmean_bound_t20'), ('G3', k, ck.get(k))

    # ---- G1: the two default arms reproduce committed trajectories bit for bit
    anchor, w, n, exp = 'none (new kappa/lr)', None, 0, None
    if kappa is not None and arm != 'W1':
        # every arm shares t1..t20 with the reference at lr=1e-3
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        w = 0.
        for t in range(1, min(tasks, clamp_from - 1) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
        exp = 3 * min(tasks, clamp_from - 1)
        anchor = f'elu_growth_0909/LR (t1-{clamp_from - 1} prefix)'
    if arm == 'N0':
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        w = 0.
        for t in range(1, min(tasks, 120) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
        exp = 3 * min(tasks, 120); anchor = 'elu_growth_0909/LR'
    elif arm == 'W1':
        ref = np.load(ROOT / 'results/grad_coherence_0911' / f'N0w_s{seed}_units.npz')
        g1_to = tasks if clamp_from == CLAMP_FROM else min(tasks, clamp_from - 1)
        w = 0.
        for key in units:
            if key in ref and int(key.rsplit('_t', 1)[1]) <= g1_to:
                w = max(w, float(np.abs(units[key].astype(float) - ref[key].astype(float)).max())); n += 1
        exp = None; anchor = 'grad_coherence_0911/N0w'
        assert n >= 6 * g1_to, ('G1 compared too few arrays', n, g1_to)
    ck.update(g1_anchor=anchor, g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=exp)
    if w is not None:
        assert w <= 1e-10, ('G1 failed', arm, seed, w)
        assert n > 0 and (exp is None or n == exp), ('G1 count guard', n, exp)
    if controls and seed == 0 and arm == 'N0':
        p3 = H.init_params(seed, torch.device('cpu'))
        with torch.no_grad():
            p3[0].data[0, 0] += 1e-3
        ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        loop(p3, GS.make_act('LR'), ad3, g3, mnist, probe, 1, 1, (None, LR0), None, [], u3, {},
             instrument=False)
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
        assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, clamp_from=clamp_from,
                cfg=dict(kappa=kappa, lr=lr), late=LATE, checks=ck,
                spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), data_sha256=mnist.sha256,
                torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; clamp height kappa x t20 norm, per-arm lr, per-update turn angle'))
    print('FINISHED', tag, 'G1', ck['g1_units_maxabs'], f'({n})', round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def _controls(seed, snap, base, mnist, probe, ck_main, clamp_from, lr):
    """The committed clamp's mutation controls (skip / rowscale / tail) at t20."""
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if key in CH.FLOOR_OF else None
        return 100 * CH.limit(ck_main, key, b)
    for mut, keys in [('skip', ['c3_cnorm_rel']), ('rowscale', ['c3_m_absdiff']), ('tail', ['clamp_tail_absdiff'])]:
        p2, act2, adam2, gens2 = C.restore(snap, 'LR')
        ck2 = {}
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
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if not (mut == 'skip' and step == 300):
                    CH.apply_clamp(p2, 'wclamp', base, ck2, None if mut == 'skip' else mut)
                CH.verify_clamped(p2, 'wclamp', base, ck2)
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
            subprocess.run([sys.executable, '-m', 'src.turn_rate_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
