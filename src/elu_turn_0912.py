"""elu_turn_0912: does saturation act through the turn rate?  (spec_elu_turn_0912)

omega ~ |dW~_i| / |W~_i|  =  (step length, set by the gate) / (unit length, set by width)

The tau dial (CELU tau = 0.3 / 1 / 3) changes saturation at ~constant width; the ELU
width clamp changes the denominator at ~constant gate.  Both are measured here with
the same read-only instrumentation as turn_rate_0912 (angle per update, net step
norm, per-unit arrays).  Every arm must reproduce a committed trajectory bit for bit.
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
OUT = ROOT / 'results/elu_turn_0912'
SMOKE = ROOT / 'results/_smoke_elu_turn_0912'
TASKS = 120
CLAMP_FROM = 21
LATE = (61, 120)
LR0 = .001
TIME_CAP = 1800.
SPEC = ROOT / 'specs/spec_elu_turn_0912.md'

# name -> (activation arm for GS.make_act, clamp)
ARMS = {
    'CELU03': ('CELU03', None),
    'CELU1':  ('CELU1',  None),
    'CELU3':  ('CELU3',  None),
    'CELU1w': ('CELU1',  'wclamp'),
    'LR':     ('LR',     None),
}


def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


ANGLE_XCHECK_MIN = 1e-4     # arccos loses eps64/theta^2 relative; at 1e-4 rad that is ~2e-8


def angles(a, b):
    """Per-row angle.  The normalised-chord form 2 asin(|a_hat - b_hat|/2) is the primary
    (stable at any theta); arccos of the dot product is kept for the cross-check, which is
    only meaningful where theta > ANGLE_XCHECK_MIN (near-dead ELU units turn by ~1e-6 rad
    per step, where arccos is off by 1e-4 relative -- the smoke caught exactly that)."""
    na = a.norm(dim=1).clamp(min=1e-300); nb = b.norm(dim=1).clamp(min=1e-300)
    cos = ((a * b).sum(1) / (na * nb)).clamp(-1., 1.)
    th = 2 * torch.arcsin((((a / na.unsqueeze(1)) - (b / nb.unsqueeze(1))).norm(dim=1) / 2).clamp(max=1.))
    return th, cos, na, nb


def loop(p, act, adam, gens, mnist, probe, t_from, t_to, clamp, base, rows, units, ck,
         instrument=True, mut=None, cfrom=CLAMP_FROM, lr=LR0):
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
            S2, dN, fN, graw2, kap2, dth, stp = z(), z(), z(), z(), z(), z(), z()
            tot = torch.zeros(100, 784, dtype=torch.float64)
            Wt_start = C.rows_of(p[0])[2].clone(); Wt0 = Wt_start.clone()
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
                if clamp is not None and task >= cfrom:
                    CH.apply_clamp(p, clamp, base, ck, mut)
                    CH.verify_clamped(p, clamp, base, ck)
                if late:
                    Wt_a = C.rows_of(p[0])[2]
                    th, cos, na, nb = angles(Wt_b, Wt_a)
                    dth += th
                    stp += (Wt_a - Wt_b).norm(dim=1)                 # the NUMERATOR
                    ck['g2_cos_range'] = max(ck.get('g2_cos_range', 0.), float((cos.abs() - 1).clamp(min=0).max()))
                    th_dot = torch.arccos(cos)
                    big = th > ANGLE_XCHECK_MIN
                    if bool(big.any()):
                        ck['g2_angle_rel'] = max(ck.get('g2_angle_rel', 0.), float(
                            ((th_dot[big] - th[big]).abs() / th[big]).max()))
                    ck['g2_tiny_angle_frac'] = ck.get('g2_tiny_angle_frac', 0.) + float((~big).double().mean()) / (625 * (LATE[1] - LATE[0] + 1))
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
        r.update(task=task, step=625, clamp=(clamp or 'ref'), ce0=ce0, ce20=ce20, lr=lr)
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
            om = dth / 625; st = stp / 625
            cn = Wt1.norm(dim=1)
            r.update(omega_step=float(om.mean()), theta_task=float(Th.mean()),
                     step_norm=float(st.mean()), turn_eff=float(Th.sum() / dth.sum()),
                     omega_decomp_gap=float((torch.log(om) - torch.log(st / cn)).abs().mean()),
                     S2=float(S2.mean()), rho_mean=float(torch.nanmean(rho)),
                     kap2=float(kap2.mean() / 625), graw2=float(graw2.mean() / 625),
                     drift2_net=float(dN.mean() / n), diff2_net=float(fN.mean() / n))
            units[f'ref_omega_i_t{task}'] = om.numpy().copy()
            units[f'ref_step_i_t{task}'] = st.numpy().copy()
            units[f'ref_theta_i_t{task}'] = Th.numpy().copy()
        rows.append(r)
    return p, act, adam


def _cmp(units, ref, prefix_mine, prefix_ref, t_lo, t_hi, kinds=None):
    w, n = 0., 0
    for key in units:
        if not key.startswith(prefix_mine + '_'):
            continue
        kind, t = key[len(prefix_mine) + 1:].rsplit('_t', 1)
        t = int(t)
        if not (t_lo <= t <= t_hi) or (kinds is not None and kind not in kinds):
            continue
        rk = f'{prefix_ref}_{kind}_t{t}'
        if rk in ref:
            w = max(w, float(np.abs(units[key].astype(float) - ref[rk].astype(float)).max())); n += 1
    return w, n


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, clamp_from=CLAMP_FROM):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    act_arm, clamp = ARMS[arm]
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(act_arm))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G4 phi-prime', ck)
    ck.update(act=act_arm, clamp=(clamp or 'ref'), lr=LR0)

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(act_arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    base = None
    if clamp is not None:
        loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, clamp, None, rows, units, ck, cfrom=clamp_from)
        base = C.baselines(p[0])
        ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
        if controls:
            ck.update(_controls(act_arm, seed, C.snapshot(p, act, adam, gens), base, mnist, probe, ck, clamp_from))
        loop(p, act, adam, gens, mnist, probe, clamp_from, tasks, clamp, base, rows, units, ck, cfrom=clamp_from)
    else:
        loop(p, act, adam, gens, mnist, probe, 1, tasks, None, None, rows, units, ck)
    T.finite_guard(rows, tag)

    assert ck.get('g2_cos_range', 0.) == 0., ('G2 cos', ck.get('g2_cos_range'))
    assert ck.get('g2_angle_rel', 1.) <= 1e-6, ('G2 angle cross-check', ck.get('g2_angle_rel'))
    assert ck.get('g2_triangle', -1.) <= 1e-8, ('G2 triangle', ck.get('g2_triangle'))
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck.get('g2_tot_ident'))
    if clamp is not None:
        for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff'):
            assert k in ck and ck[k] <= CH.limit(ck, k, 'f32_rowmean_bound_t20'), ('G3', k, ck.get(k))

    # ---- G1
    if arm == 'LR':
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        w, n = 0., 0
        for t in range(1, min(tasks, 120) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
        exp = 3 * min(tasks, 120); anchor = 'elu_growth_0909/LR'
    elif arm == 'CELU1w':
        ref = np.load(ROOT / 'results/unit_triage_0911' / f'ELU1_none_s{seed}_units.npz')
        kinds = ('cnorm_i', 'gbar_i', 'bias_i')
        w1, n1 = _cmp(units, ref, 'ref', 'ref', 1, min(tasks, clamp_from - 1), kinds)
        # the anchor's wclamp branch starts at CLAMP_FROM=21; a smoke run that clamps
        # earlier can only be checked on the shared pre-clamp prefix.
        if clamp_from == CLAMP_FROM:
            g1_hi = min(tasks, 100)
            w2, n2 = _cmp(units, ref, 'ref', 'wclamp', clamp_from, g1_hi, kinds)
        else:
            g1_hi, w2, n2 = clamp_from - 1, 0., 0
        w, n = max(w1, w2), n1 + n2
        exp = 3 * min(tasks, clamp_from - 1) + (3 * (g1_hi - clamp_from + 1) if g1_hi >= clamp_from else 0)
        anchor = 'unit_triage_0911/ELU1_none (ref t1-20, wclamp t21-100)'
    else:
        ref = np.load(ROOT / 'results/gate_shape_0911' / f'{arm}_s{seed}_units.npz')
        w, n = _cmp(units, ref, 'ref', 'ref', 1, min(tasks, 120))
        exp = None; anchor = f'gate_shape_0911/{arm}'
        assert n >= 12 * min(tasks, 120), ('G1 compared too few arrays', n)
    ck.update(g1_anchor=anchor, g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=exp)
    assert w <= 1e-10, ('G1 failed', arm, seed, w)
    assert n > 0 and (exp is None or n == exp), ('G1 count guard', n, exp)
    if controls and seed == 0 and arm == 'LR':
        p3 = H.init_params(seed, torch.device('cpu'))
        with torch.no_grad():
            p3[0].data[0, 0] += 1e-3
        ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        loop(p3, GS.make_act('LR'), ad3, g3, mnist, probe, 1, 1, None, None, [], u3, {}, instrument=False)
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
        assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, clamp_from=clamp_from, cfg=dict(act=act_arm, clamp=clamp, lr=LR0),
                late=LATE, checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                gate_shape_sha256=T.sha(Path(GS.__file__)), clamp_horizon_sha256=T.sha(Path(CH.__file__)),
                base_sha256=T.sha(Path(C.__file__)), host_sha256=T.sha(Path(H.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; CELU tau dial + ELU wclamp + leaky, per-update turn angle and net step norm'))
    print('FINISHED', tag, 'G1', ck['g1_units_maxabs'], f'({n})', round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def _controls(act_arm, seed, snap, base, mnist, probe, ck_main, clamp_from):
    """The committed clamp's mutation controls at t20; margin = 10x the check tolerance."""
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if key in CH.FLOOR_OF else None
        return 10 * CH.limit(ck_main, key, b)
    for mut, keys in [('skip', ['c3_cnorm_rel']), ('rowscale', ['c3_m_absdiff']), ('tail', ['clamp_tail_absdiff'])]:
        p2, act2, adam2, gens2 = C.restore(snap, act_arm if act_arm == 'LR' else 'LR')
        act2 = GS.make_act(act_arm)
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
                    q -= LR0 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if not (mut == 'skip' and step == 300):
                    CH.apply_clamp(p2, 'wclamp', base, ck2, None if mut == 'skip' else mut)
                CH.verify_clamped(p2, 'wclamp', base, ck2)
        for k in keys:
            got = ck2.get(k, 0.)
            b = 'f32_rowmean_bound_t20' if k in CH.FLOOR_OF else None
            res[f'ctl_{mut}_{k}'] = got
            res[f'ctl_{mut}_{k}_x_tol'] = got / max(CH.limit(ck_main, k, b), 1e-300)
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
            subprocess.run([sys.executable, '-m', 'src.elu_turn_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
