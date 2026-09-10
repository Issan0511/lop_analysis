"""clamp_horizon_0910: separate sinking from width at a long horizon (spec_clamp_horizon_0910).

Same branch point and clamps as width_sink_clamp_0909 (init -> t20 once, then
bit-identical branches with a per-update clamp), continued to t400, plus a
`wcap2` arm that caps ||W~_i|| at 2x its t20 value instead of pinning it.

Self-contained on purpose: width_sink_clamp_0909.py is NOT edited, so the
provenance of its committed results keeps matching its file.  The training loop
below is character-identical to C.loop; only the clamp set differs.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/clamp_horizon_0910'
REF = ROOT / 'results/width_sink_clamp_0909'
ARMS = [('LR', 'none'), ('ELU1', 'none')]
CLAMPS = ['ref', 'dclamp', 'wclamp', 'wcap2']
CAP = 2.0                      # wcap2 ceiling, in units of the t20 norm
TASKS = 400
CLAMP_FROM = 21
G1_LAST = 100                  # tasks 21..100 must reproduce width_sink_clamp_0909
TIME_CAP = 2400.
EPS32 = C.EPS32
TOL = dict(c3_cnorm_rel=1e-6, c3_m_absdiff=1e-9, c5_mbar_abs=1e-8, c5_sd_rel=1e-6,
           c5_cnorm_rel=1e-6, cap_excess_rel=1e-6, cap_m_absdiff=1e-9, clamp_tail_absdiff=0.,
           c6_var_rel=1e-9, c6_mean_abs=1e-9, decompose_crosscheck=1e-9, cap_restore_rel=1e-6)
FLOOR_OF = dict(c3_m_absdiff='f32_rowmean_bound', c5_mbar_abs='f32_rowmean_bound', cap_m_absdiff='f32_rowmean_bound')
REL_CHECKS = ('c3_cnorm_rel', 'c5_sd_rel', 'c5_cnorm_rel', 'cap_excess_rel')
REL_FLOOR = 10 * EPS32


def limit(ck, k, bound_key=None):
    """Tolerance for check `k`.  Absolute row-mean checks are floored by the
    provable float32 bound; `bound_key` selects WHICH bound (the whole-run running
    max for the run itself, the t20 snapshot for the mutation controls, which are
    always re-run from t20 and therefore do not grow with the horizon)."""
    if k in REL_CHECKS:
        return max(TOL[k], REL_FLOOR)
    return max(TOL[k], ck.get(bound_key or FLOOR_OF.get(k, ''), 0.))


def clamped_W(kind, Wd, m, Wt, base, mut=None):
    """float64 clamped weights.  `wclamp`/`dclamp` are C's; `wcap2` is a ceiling."""
    if kind == 'wcap2':
        cap = (CAP if mut != 'cap4' else 4 * CAP) * base['c']
        n = Wt.norm(dim=1, keepdim=True)
        s = torch.where(n > cap, cap / n, torch.ones_like(n))
        return m + Wt * s
    return C.clamped_W(kind, Wd, m, Wt, base, mut)


def apply_clamp(p, kind, base, ck, mut=None):
    """In place on p[0]; running maxima of the invariance checks into `ck`."""
    Wd, m, Wt = C.rows_of(p[0])
    cn0 = Wt.norm(dim=1)
    sd0 = float(m.std(unbiased=False))
    mb0 = float(m.mean())
    tail = [q.detach().clone() for q in p[1:]]
    ck['f32_rowmean_bound'] = max(ck.get('f32_rowmean_bound', 0.), EPS32 * float(Wd.abs().mean(1).max()))
    binding = bool((cn0 > CAP * base['c'].squeeze(1)).any()) if kind == 'wcap2' else True
    if kind == 'wcap2' and not binding and mut is None:
        ck['cap_bind_steps'] = ck.get('cap_bind_steps', 0)
        ck['cap_idle_steps'] = ck.get('cap_idle_steps', 0) + 1
        return dict(cnorm_pre=cn0, sd_pre=sd0, mbar_pre=mb0, binding=False)
    New = clamped_W(kind, Wd, m, Wt, base, mut)
    p[0].data.copy_(New.float())
    if mut == 'tail':                       # control for clamp_tail_absdiff
        p[1].data.add_(1e-4)
    Wd1, m1, Wt1 = C.rows_of(p[0])
    cn1 = Wt1.norm(dim=1)

    def up(k, v):
        ck[k] = max(ck.get(k, 0.), float(v))

    if kind == 'wclamp':
        up('c3_m_absdiff', (m1 - m).abs().max())
    elif kind == 'dclamp':
        up('c5_sd_rel', abs(float(m1.std(unbiased=False)) - sd0) / sd0)
        up('c5_cnorm_rel', ((cn1 - cn0).abs() / cn0).max())
    else:                                   # wcap2 (only reached when the cap binds)
        # rows that were over the ceiling must come back to exactly CAP*c_i.  This is
        # what the `cap4` mutation breaks (with an 8c ceiling nothing is projected).
        over = cn0 > CAP * base['c'].squeeze(1)
        if over.any():
            tgt = CAP * base['c'].squeeze(1)[over]
            up('cap_restore_rel', ((cn1[over] - tgt).abs() / tgt).max())
        up('cap_m_absdiff', (m1 - m).abs().max())
        ck['cap_bind_steps'] = ck.get('cap_bind_steps', 0) + 1
    up('clamp_tail_absdiff', max(float((a.detach() - b).abs().max()) for a, b in zip(p[1:], tail)))
    return dict(cnorm_pre=cn0, sd_pre=sd0, mbar_pre=mb0, binding=binding)


def verify_clamped(p, kind, base, ck):
    """Does the CURRENT state meet the target?  Every step, clamp applied or not."""
    Wd, m, Wt = C.rows_of(p[0])
    c = base['c'].squeeze(1)

    def up(k, v):
        ck[k] = max(ck.get(k, 0.), float(v))

    if kind == 'wclamp':
        up('c3_cnorm_rel', ((Wt.norm(dim=1) - c).abs() / c).max())
    elif kind == 'dclamp':
        up('c5_mbar_abs', abs(float(m.mean()) - base['mbar0']))
    else:
        up('cap_excess_rel', ((Wt.norm(dim=1) - CAP * c).clamp(min=0.) / c).max())


def measure(p, act, probe, perm, want_acc=False, mnist=None, ck=None, mut_decomp=False):
    r, u = C.measure(p, act, probe, perm, want_acc, mnist, ck, mut_decomp)
    with torch.no_grad():
        z = probe.px[:, perm] @ p[0].detach().T + p[1].detach()
        g = act.dphi(z, 0) if isinstance(act, H.AdaptiveSnake) else act.dphi(z)
        r['dead_hard'] = int((g.max(0).values < 1e-6).sum())
        r['dead_soft'] = int((g.max(0).values < .05).sum())
        r['sat'] = float((g < .05).float().mean())
    return r, u


def loop(p, act, adam, gens, mnist, probe, t_from, t_to, clamp, base, rows, units, ck, mut=None):
    gp, gd, gb = gens
    for task in range(t_from, t_to + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        ce20 = None
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]
                c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
                if clamp != 'ref':
                    pre = apply_clamp(p, clamp, base, ck, mut)
                    verify_clamped(p, clamp, base, ck)
                    if clamp == 'wclamp':
                        ck['pressure'] = ck.get('pressure', 0.) + float((pre['cnorm_pre'] / base['c'].squeeze(1)).log().mean())
                    elif clamp == 'dclamp':
                        ck['pressure'] = ck.get('pressure', 0.) + (base['mbar0'] - pre['mbar_pre'])
                    else:
                        ck['pressure'] = ck.get('pressure', 0.) + float(pre['binding'])   # binding steps so far
            if step == 20:                       # task-start reference for the G3 CE gate
                with torch.no_grad():
                    ce20 = float(torch.nn.functional.cross_entropy(H.forward(p, probe.px[:, perm], act)[4], probe.py))
            if step == 625 and rows is not None:
                r, u = measure(p, act, probe, perm, True, mnist, ck)
                r.update(task=task, step=625, clamp=clamp, ce20=ce20, pressure=ck.get('pressure', 0.))
                rows.append(r)
                for k, v in u.items():
                    units[f'{clamp}_{k}_t{task}'] = v
    return p, act, adam


def g1_check(arm, iv, seed, rows, units, ck, tasks=TASKS):
    """t21..100 must reproduce the committed width_sink_clamp_0909 outputs."""
    import csv
    ref_rows = list(csv.DictReader(open(REF / f'{arm}_{iv}_s{seed}_rows.csv')))
    ref_u = np.load(REF / f'{arm}_{iv}_s{seed}_units.npz')
    worst_row, worst_u, n = 0., 0., 0
    have = {(r['clamp'], int(r['task'])): r for r in rows}
    for x in ref_rows:
        if x['step'] != '625' or x['clamp'] not in ('ref', 'dclamp', 'wclamp'):
            continue
        t = int(x['task'])
        if not (CLAMP_FROM <= t <= G1_LAST) or (x['clamp'], t) not in have:
            continue
        r = have[(x['clamp'], t)]
        for k in ('zbar_inv', 'sigma_inv', 'star_sd', 'acc'):
            worst_row = max(worst_row, abs(float(x[k]) - r[k]))
        n += 1
    nu = 0
    for c in ('ref', 'dclamp', 'wclamp'):
        for t in range(CLAMP_FROM, min(tasks, G1_LAST) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i', 'm_i', 'star_i'):
                key = f'{c}_{k}_t{t}'
                if key in ref_u and key in units:
                    worst_u = max(worst_u, float(np.abs(ref_u[key] - units[key]).max())); nu += 1
    ck.update(g1_rows_maxabs=worst_row, g1_units_maxabs=worst_u, g1_rows_compared=n,
              g1_rows_expected=3 * (min(tasks, G1_LAST) - CLAMP_FROM + 1),
              g1_units_compared=nu, g1_units_expected=15 * (min(tasks, G1_LAST) - CLAMP_FROM + 1))
    return worst_row, worst_u, n


def _controls(arm, seed, snap, base, mnist, probe, ck_main, bound20):
    """Every check gets a mutation that must break it by 100x its own limit."""
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if (bound20 and key in FLOOR_OF) else None
        return 100 * max(limit(ck_main, key, b), ck_main.get(key, 0.))

    # --- wcap2: at t21 the norm is below the cap, so a plain trajectory mutation is
    # vacuous.  Build a state above the ceiling and drive it through `loop` itself, so
    # the in-loop verify_clamped call is what fires (a hand-call would not test the wiring).
    def above_cap():
        p3, act3, adam3, gens3 = C.restore(snap, arm)
        with torch.no_grad():
            Wd, m, Wt = C.rows_of(p3[0])
            p3[0].data.copy_((m + Wt * (3 * base['c'] / Wt.norm(dim=1, keepdim=True))).float())
        return p3, act3, adam3, gens3
    p3, act3, adam3, gens3 = above_cap()
    ck3 = {}
    loop(p3, act3, adam3, gens3, mnist, probe, CLAMP_FROM, CLAMP_FROM, 'wcap2', base, None, {}, ck3, mut='cap4')
    res['ctl_wcap2_cap4_excess_rel'] = ck3.get('cap_excess_rel', 0.)
    assert res['ctl_wcap2_cap4_excess_rel'] > floor('cap_excess_rel'), ('vacuous cap4 control', ck3)
    p3, act3, adam3, gens3 = above_cap()
    ck3b = {}
    loop(p3, act3, adam3, gens3, mnist, probe, CLAMP_FROM, CLAMP_FROM, 'wcap2', base, None, {}, ck3b)
    res['ctl_wcap2_restored_excess_rel'] = ck3b.get('cap_excess_rel', 0.)
    res['ctl_wcap2_restore_rel'] = ck3b.get('cap_restore_rel', 0.)
    res['ctl_wcap2_bind_steps'] = ck3b.get('cap_bind_steps', 0)
    assert ck3b.get('cap_excess_rel', 1.) <= limit(ck_main, 'cap_excess_rel'), ('cap does not restore', ck3b)
    assert ck3b.get('cap_bind_steps', 0) > 0, 'the above-cap control never bound'

    # --- trajectory mutations for the pinning clamps
    for clamp, mut, keys in [('wclamp', 'skip', ['c3_cnorm_rel']),
                             ('wclamp', 'rowscale', ['c3_m_absdiff']),
                             ('wclamp', 'tail', ['clamp_tail_absdiff']),
                             ('dclamp', 'p90', ['c5_mbar_abs']),
                             ('dclamp', 'rowscale', ['c5_cnorm_rel', 'c5_sd_rel'])]:
        p2, act2, adam2, gens2 = C.restore(snap, arm)
        ck2 = {}
        if mut == 'skip':
            gp, gd, gb = gens2
            perm = torch.randperm(784, generator=gp)
            idx = H.stratified_draw(mnist, gd); order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
            xs = mnist.train_x[idx][:, perm][order]; ys = mnist.train_y[idx][order]
            for step in range(1, 626):
                out = H.forward(p2, xs[(step - 1) * 16:step * 16], act2)
                gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p2)
                with torch.no_grad():
                    m_, v_, tc = adam2; tc[0] += 1; c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                    for q, g, mi, vi in zip(p2, gr, m_, v_):
                        mi.mul_(.9).add_(g, alpha=1 - .9); vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                        q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                    if isinstance(act2, H.AdaptiveSnake): act2.update(out[0], out[2])
                    if step != 300:
                        apply_clamp(p2, clamp, base, ck2)
                    verify_clamped(p2, clamp, base, ck2)
        else:
            loop(p2, act2, adam2, gens2, mnist, probe, CLAMP_FROM, CLAMP_FROM, clamp, base, None, {}, ck2, mut=mut)
        for key in keys:
            got = ck2.get(key, 0.)
            res[f'ctl_{clamp}_{mut}_{key}'] = got
            assert got > floor(key), (f'vacuous control {clamp}/{mut}: {key} = {got:g} <= {floor(key):g}')

    # --- C6: scaling the decomposition coefficient must break both identities
    p4, act4, adam4, gens4 = C.restore(snap, arm)
    ck4 = {}
    measure(p4, act4, probe, probe.refs[0], False, mnist, ck4, mut_decomp=True)
    res['ctl_c6_var_rel'] = ck4['c6_var_rel']; res['ctl_c6_mean_abs'] = ck4['c6_mean_abs']
    assert ck4['c6_var_rel'] > floor('c6_var_rel') and ck4['c6_mean_abs'] > floor('c6_mean_abs'), ('vacuous C6 control', ck4)

    # --- G1: a perturbed init must break the reproduction of t21..100
    p5 = H.init_params(seed, torch.device('cpu'))
    with torch.no_grad():
        p5[0].data[0, 0] += 1e-3
    act5 = C.make_act(arm)
    adam5 = ([torch.zeros_like(q) for q in p5], [torch.zeros_like(q) for q in p5], [0])
    gens5 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    r5, u5, ck5 = [], {}, {}
    loop(p5, act5, adam5, gens5, mnist, probe, 1, CLAMP_FROM, 'ref', None, r5, u5, ck5)
    ref_u = np.load(REF / f'{arm}_none_s{seed}_units.npz')
    d = max(float(np.abs(u5[f'ref_{k}_t{CLAMP_FROM}'] - ref_u[f'ref_{k}_t{CLAMP_FROM}']).max())
            for k in ('zbar_i', 'sd_i', 'cnorm_i', 'm_i', 'star_i'))
    res['ctl_g1_init'] = d
    assert d > 100 * 1e-10, f'vacuous G1 control: {d:g}'
    return res


def _assert(ck, tasks=TASKS):
    missing = [k for k in TOL if k not in ck]
    assert not missing, ('checks never recorded (a silently unexercised code path)', missing)
    bad = {k: (ck[k], limit(ck, k)) for k in TOL if not ck[k] <= limit(ck, k)}
    assert not bad, ('check failed (value, limit)', bad)
    assert ck['g1_rows_maxabs'] <= 1e-10 and ck['g1_units_maxabs'] <= 1e-10, ('G1 failed', ck)
    assert ck['g1_rows_compared'] >= ck['g1_rows_expected'], ('G1 rows', ck['g1_rows_compared'], ck['g1_rows_expected'])
    assert ck['g1_units_compared'] >= ck['g1_units_expected'], ('G1 units', ck['g1_units_compared'], ck['g1_units_expected'])
    if tasks >= 100:
        assert ck.get('cap_bind_steps', 0) > 0, 'wcap2 never bound: the arm is identical to ref'


def run(arm, iv, seed, tasks=TASKS, clamp_from=CLAMP_FROM, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu')
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{arm}_{iv}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    p = H.init_params(seed, torch.device('cpu'))
    act = C.make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'ref', None, rows, units, ck)
    base = C.baselines(p[0])
    ck['f32_rowmean_bound_t20'] = EPS32 * float(p[0].detach().double().abs().mean(1).max())
    ck['decompose_crosscheck'] = C.decompose_crosscheck(p, probe)
    ck.update(base_sigma_m=base['sigma_m'], base_mbar0=base['mbar0'], base_cnorm_mean=float(base['c'].mean()))
    snap = C.snapshot(p, act, adam, gens)
    for clamp in CLAMPS:
        p2, act2, adam2, gens2 = C.restore(snap, arm)
        ck['pressure'] = 0.
        loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, tasks, clamp, base, rows, units, ck)
        ck[f'pressure_{clamp}'] = ck.pop('pressure', 0.)
        print('DONE', prefix, clamp, round(time.monotonic() - t0, 1), 's', flush=True)
    g1_check(arm, iv, seed, rows, units, ck, tasks)
    if controls:
        ck.update(_controls(arm, seed, snap, base, mnist, probe, ck, bound20=True))
    _assert(ck, tasks)
    keys = []
    [keys.append(k) for r in rows for k in r if k not in keys]
    for r in rows:
        r.update(arm=arm, iv=iv, seed=seed)
    keys += ['arm', 'iv', 'seed']
    C.G.B.csvwrite(out / f'{prefix}_rows.csv', [{k: r.get(k) for k in keys} for r in rows])
    np.savez_compressed(out / f'{prefix}_units.npz', **units)
    (out / f'{prefix}_provenance.json').write_text(json.dumps(dict(
        arm=arm, iv=iv, seed=seed, tasks=tasks, clamp_from=clamp_from, clamps=CLAMPS, cap=CAP, checks=ck,
        spec_sha256=C.sha(ROOT / 'specs/spec_clamp_horizon_0910.md'), code_sha256=C.sha(Path(__file__)),
        base_sha256=C.sha(Path(C.__file__)), host_sha256=C.sha(Path(H.__file__)), data_sha256=mnist.sha256,
        wall_seconds=time.monotonic() - t0,
        scope='CPU continuation from the shared t20 state; ref reproduces width_sink_clamp_0909 t21-100'), indent=1, default=str))
    print('FINISHED', prefix, 'G1', ck['g1_rows_maxabs'], ck['g1_units_maxabs'], round(time.monotonic() - t0, 1), 's', flush=True)
    assert time.monotonic() - t0 < TIME_CAP, 'runtime cap'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--iv', default='none'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        run('LR', 'none', 0, tasks=24, clamp_from=21, out=ROOT / 'results/_smoke_clamp_horizon', controls=True)
        return
    if a.all:
        def job(j):
            subprocess.run([sys.executable, '-m', 'src.clamp_horizon_0910', '--arm', j[0], '--iv', j[1], '--seed', str(j[2])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(job, [(arm, iv, s) for arm, iv in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.iv, a.seed)


if __name__ == '__main__':
    main()
