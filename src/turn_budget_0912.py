"""turn_budget_0912: can the lost rotation be bought back with more updates?
(spec_turn_budget_0912)

Reference measurement (turn_rate_0912/N0, 3 seeds): the net rotation a unit achieves in
one task falls as Theta ~ ||W~_i||^-0.63, and turn_eff ~ 0.33 (ballistic would be 1.0,
diffusive 1/sqrt(625) = 0.04).  Rather than guess the exponent of the compensation, the
decisive arm TRAINS UNTIL the rotation reaches its t20 value and REPORTS the updates it
needed.

The data budget is identical in every arm: the 10000-example task pool is reused (freshly
shuffled from a dedicated 'reshuffle' stream) for updates beyond 625, so "more updates"
is separated from "more data".
"""
from pathlib import Path
import argparse, concurrent.futures, json, math, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import elu_turn_0912 as ET

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/turn_budget_0912'
SMOKE = ROOT / 'results/_smoke_turn_budget_0912'
ACT = 'LR'
TASKS = 120
BRANCH = 21
LATE = (61, 120)
LR0 = .001
BASE_STEPS = 625
MAX_STEPS = 6250           # 10x cap (raised pre-run: the pilot puts the exponent near 0.35,
                           # so closing the t120 deficit needs ~6x); binds are recorded, T3 excludes them
CHECK_EVERY = 25           # rotation is tested this often (mutation control: 625)
TIME_CAP = 7200.
SPEC = ROOT / 'specs/spec_turn_budget_0912.md'

# name -> policy
ARMS = {'R0': 'fixed', 'TH': 'theta', 'P1': 'linear', 'FLAT': 'flat2'}


def probe_ce(p, act, probe, perm):
    with torch.no_grad():
        return float(F.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))


def n_steps_for(policy, cn, cn20, theta_star):
    """Updates planned before the task starts.  'theta' is closed-loop (planned = cap)."""
    if policy == 'fixed':
        return BASE_STEPS
    if policy == 'flat2':
        return 2 * BASE_STEPS
    if policy == 'linear':
        return int(min(MAX_STEPS, max(BASE_STEPS, round(BASE_STEPS * cn / cn20))))
    return MAX_STEPS                                   # 'theta': stop when Theta >= target


def loop(p, act, adam, gens, rgen, mnist, probe, t_from, t_to, policy, rows, units, ck,
         cn20=None, theta_star=None, instrument=True, check_every=CHECK_EVERY):
    gp, gd, gb = gens
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        pool_x = mnist.train_x[idx][:, perm]
        pool_y = mnist.train_y[idx]
        xs, ys = pool_x[order], pool_y[order]
        if ck is not None and task == t_from:
            ck['g4_pool_set_ok'] = bool(torch.equal(torch.sort(idx).values, torch.sort(idx).values))
        ce0 = probe_ce(p, act, probe, perm)
        with torch.no_grad():
            Wt_start = C.rows_of(p[0])[2].clone()
            cn_now = float(Wt_start.norm(dim=1).mean())
        plan = n_steps_for(policy, cn_now, cn20 or cn_now, theta_star)
        late = instrument and LATE[0] <= task <= LATE[1]
        if late:
            z = lambda: torch.zeros(100, dtype=torch.float64)
            S2, dN, fN, dth, stp = z(), z(), z(), z(), z()
            tot = torch.zeros(100, 784, dtype=torch.float64)
            Wt0 = Wt_start.clone()
        ce20 = None
        step, epoch, hit_cap, stopped_at = 0, 0, False, None
        while step < plan:
            j = step % BASE_STEPS
            if step and j == 0:                         # reshuffle the SAME pool
                epoch += 1
                o2 = torch.randperm(H.TASK_EXAMPLES, generator=rgen)
                xs, ys = pool_x[o2], pool_y[o2]
            step += 1
            out = H.forward(p, xs[j * 16:(j + 1) * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                out[4], ys[j * 16:(j + 1) * 16]), p)
            with torch.no_grad():
                if late:
                    Wb = p[0].detach().double().clone()
                    Wt_b = C.rows_of(p[0])[2].clone()
                    g_raw = gr[0].double().clone()
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= LR0 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if late:
                    Wt_a = C.rows_of(p[0])[2]
                    th, cos, na, nb = ET.angles(Wt_b, Wt_a)
                    dth += th; stp += (Wt_a - Wt_b).norm(dim=1)
                    ck['g2_cos_range'] = max(ck.get('g2_cos_range', 0.), float((cos.abs() - 1).clamp(min=0).max()))
                    big = th > ET.ANGLE_XCHECK_MIN
                    if bool(big.any()):
                        td = torch.arccos(cos)
                        ck['g2_angle_rel'] = max(ck.get('g2_angle_rel', 0.), float(
                            ((td[big] - th[big]).abs() / th[big]).max()))
                    dw = p[0].detach().double() - Wb
                    gn = g_raw.norm(dim=1, keepdim=True).clamp(min=1e-300)
                    ghat = g_raw / gn
                    dr = (dw * ghat).sum(1, keepdim=True) * ghat
                    dN += (dr ** 2).sum(1); fN += ((dw - dr) ** 2).sum(1)
                    dwt = dw - dw.mean(1, keepdim=True)
                    S2 += (dwt ** 2).sum(1); tot += dwt
                if policy == 'theta' and step >= BASE_STEPS and step % check_every == 0:
                    cur = float(ET.angles(Wt_start, C.rows_of(p[0])[2])[0].mean())
                    if cur >= theta_star:
                        stopped_at = step
                        break
            if step == 20:
                ce20 = probe_ce(p, act, probe, perm)
        if policy == 'theta':
            hit_cap = stopped_at is None
        with torch.no_grad():
            Th = ET.angles(Wt_start, C.rows_of(p[0])[2])[0]
        if rows is None:
            continue
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, _ = GS.gate_block(p, act, probe, perm)
        r.update(gr_)
        r.update(task=task, step=step, clamp='ref', ce0=ce0, ce20=ce20, lr=LR0,
                 n_steps=step, epochs=epoch + 1, hit_cap=int(hit_cap),
                 theta_task=float(Th.mean()),
                 theta_ratio=(float(Th.mean()) / theta_star if theta_star else 0.0),   # 0 = prefix (no target yet)
                 ce625=probe_ce(p, act, probe, perm))
        with torch.no_grad():
            r['w2_fro'] = float(p[2].double().norm())
        for k, v in list(u.items()) + list(gu.items()):
            units[f'ref_{k}_t{task}'] = v
        if late:
            D2 = (tot ** 2).sum(1)
            rho = torch.where(S2 > 0, D2 / S2.clamp(min=1e-300), torch.full_like(S2, float('nan')))
            Wt1 = C.rows_of(p[0])[2]
            ck['g2_tot_ident'] = max(ck.get('g2_tot_ident', 0.), float((tot - (Wt1 - Wt0)).abs().max()))
            ck['g2_triangle'] = max(ck.get('g2_triangle', 0.), float((Th - dth).max()))
            n = step * LR0 ** 2
            r.update(omega_step=float((dth / step).mean()), step_norm=float((stp / step).mean()),
                     turn_eff=float(Th.sum() / dth.sum()), S2=float(S2.mean()),
                     rho_mean=float(torch.nanmean(rho)),
                     drift2_net=float(dN.mean() / n), diff2_net=float(fN.mean() / n))
            units[f'ref_theta_i_t{task}'] = Th.numpy().copy()
        rows.append(r)
    return p, act, adam


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, branch=BRANCH, check_every=CHECK_EVERY):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    policy = ARMS[arm]
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(ACT))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G4 phi-prime', ck)
    ck.update(act=ACT, policy=policy, lr=LR0, base_steps=BASE_STEPS, max_steps=MAX_STEPS,
              check_every=check_every)

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(ACT)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    rgen = H.stream('reshuffle', seed)
    st0 = [g.get_state().clone() for g in gens]
    # ---- shared prefix t1..branch-1 at 625 updates for every arm
    loop(p, act, adam, gens, rgen, mnist, probe, 1, branch - 1, 'fixed', rows, units, ck)
    with torch.no_grad():
        cn20 = float(C.rows_of(p[0])[2].norm(dim=1).mean())
    win = [r for r in rows if max(1, branch - 5) <= r['task'] <= branch - 1]
    theta_star = float(np.median([r['theta_task'] for r in win if np.isfinite(r.get('theta_task', np.nan))])) \
        if any(np.isfinite(r.get('theta_task', np.nan)) for r in win) else None
    if theta_star is None:                       # prefix is outside LATE: measure it directly
        theta_star = _measure_theta(p, act, adam, gens, rgen, mnist, probe, branch, ck)
    ck.update(cn20=cn20, theta_star=theta_star)
    loop(p, act, adam, gens, rgen, mnist, probe, branch, tasks, policy, rows, units, ck,
         cn20=cn20, theta_star=theta_star, check_every=check_every)
    T.finite_guard(rows, tag)

    g2 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ck['g4_streams_intact'] = all(torch.equal(a, b.get_state()) for a, b in zip(st0, g2))
    assert ck['g4_streams_intact']
    if policy == 'fixed':
        ck['g4_rgen_untouched'] = bool(torch.equal(H.stream('reshuffle', seed).get_state(), rgen.get_state()))
        assert ck['g4_rgen_untouched']
    assert ck.get('g2_cos_range', 0.) == 0., ('G2 cos', ck.get('g2_cos_range'))
    assert ck.get('g2_angle_rel', 1.) <= 1e-6, ('G2 angle', ck.get('g2_angle_rel'))
    assert ck.get('g2_triangle', -1.) <= 1e-8, ('G2 triangle', ck.get('g2_triangle'))
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck.get('g2_tot_ident'))
    ns = [r['n_steps'] for r in rows if r['task'] >= branch]
    ck.update(n_steps_min=min(ns), n_steps_max=max(ns), n_steps_mean=float(np.mean(ns)),
              cost=float(np.sum(ns)) / (BASE_STEPS * len(ns)),
              cap_frac=float(np.mean([r['hit_cap'] for r in rows if r['task'] >= branch])))
    if policy == 'theta':
        # G3: every non-capped task stopped on a multiple of check_every, at or after 625
        bad = [r['n_steps'] for r in rows if r['task'] >= branch and not r['hit_cap']
               and (r['n_steps'] % check_every or r['n_steps'] < BASE_STEPS)]
        ck['g3_stop_grid_violations'] = len(bad)
        assert not bad, ('G3 stop rule', bad[:5])

    # ---- G1
    eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
    g1_to = min(tasks, 120) if policy == 'fixed' else min(tasks, branch - 1)
    w, n = 0., 0
    for t in range(1, g1_to + 1):
        for k in ('zbar_i', 'sd_i', 'cnorm_i'):
            key = f'ref_{k}_t{t}'
            if key in units:
                w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
    ck.update(g1_anchor='elu_growth_0909/LR' + ('' if policy == 'fixed' else f' (t1-{g1_to} prefix)'),
              g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=3 * g1_to)
    assert w <= 1e-10, ('G1 failed', arm, seed, w)
    assert n == 3 * g1_to, ('G1 count guard', n, 3 * g1_to)
    if controls and seed == 0 and arm == 'R0':
        p3 = H.init_params(seed, torch.device('cpu'))
        with torch.no_grad():
            p3[0].data[0, 0] += 1e-3
        ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        loop(p3, GS.make_act(ACT), ad3, g3, H.stream('reshuffle', seed), mnist, probe, 1, 1,
             'fixed', [], u3, {}, instrument=False)
        ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
        assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, branch=branch,
                cfg=dict(act=ACT, policy=policy, lr=LR0, base_steps=BASE_STEPS, max_steps=MAX_STEPS),
                late=LATE, checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                elu_turn_sha256=T.sha(Path(ET.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                base_sha256=T.sha(Path(C.__file__)), host_sha256=T.sha(Path(H.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; per-task update budget tied to unit length or achieved rotation'))
    print('FINISHED', tag, 'G1', ck['g1_units_maxabs'], f"cost {ck['cost']:.2f}", round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def _measure_theta(p, act, adam, gens, rgen, mnist, probe, branch, ck):
    raise RuntimeError('theta_star must come from the shared prefix; widen LATE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        global LATE, MAX_STEPS
        LATE = (1, 8); MAX_STEPS = 1875
        for arm in ARMS:
            run(arm, 0, tasks=6, out=SMOKE, controls=True, branch=4, check_every=25)
        return
    if a.all:
        def job(j):
            subprocess.run([sys.executable, '-m', 'src.turn_budget_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
