"""gate_wclamp_0911: sub-run of spec_gate_shape_0911 -- `ref` and `wclamp` continuations
t21-100 from the shared t20 state, for the Snake ladder (SN02 / SN06 / SN15) and the
leak ladder (LR03 / LR001), whose wclamp trajectories do not exist yet.

Committed machinery, imported and never edited: CH.loop (training step + clamp +
invariance checks + measure with dead counters), CH.apply_clamp / CH.verify_clamped,
C.baselines / C.snapshot / C.decompose_crosscheck.  Local: make_act (the committed
dispatchers do not know these arms), restore (C.restore KeyErrors on them), the G1
anchor (ref-only), and a controls table restricted to wclamp.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import numpy as np
import torch
from src import clamp_horizon_0910 as CH
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/gate_wclamp_0911'
SMOKE = ROOT / 'results/_smoke_gate_wclamp_0911'
ARMS = ['SN02', 'SN06', 'SN15', 'LR03', 'LR001']
CLAMPS = ['ref', 'wclamp']
TASKS = 100
CLAMP_FROM = 21
TIME_CAP = 1200.
TOL = {k: CH.TOL[k] for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff',
                              'c6_var_rel', 'c6_mean_abs', 'decompose_crosscheck')}
SPEC = ROOT / 'specs/spec_gate_shape_0911.md'
CONTROL_OF = {'ctl_wclamp_skip_c3_cnorm_rel': 'c3_cnorm_rel',
              'ctl_wclamp_rowscale_c3_m_absdiff': 'c3_m_absdiff',
              'ctl_wclamp_tail_clamp_tail_absdiff': 'clamp_tail_absdiff',
              'ctl_c6_var_rel': 'c6_var_rel', 'ctl_c6_mean_abs': 'c6_mean_abs'}


def restore(snap, arm):
    p = [q.clone().requires_grad_(True) for q in snap['p']]
    act = GS.make_act(arm)
    adam = ([x.clone() for x in snap['adam'][0]], [x.clone() for x in snap['adam'][1]], [snap['adam'][2]])
    gens = []
    for s in snap['gens']:
        g = torch.Generator(); g.set_state(s.clone()); gens.append(g)
    return p, act, adam, gens


def limit(ck, k, bound_key=None):
    return CH.limit(ck, k, bound_key)


def expected_count(arm, seed, tasks):
    if arm in ('LR03', 'LR001'):
        return 6 * min(tasks, 100)
    ref = np.load(ROOT / 'results/gate_scale_invariance_0909' / f'{arm}_s{seed}_units.npz')
    return sum(1 for t in ref['tasks'] if int(t) <= tasks)


def _controls(arm, seed, snap, base, mnist, probe, ck_main, clamp_from):
    """wclamp mutation controls (skip / rowscale / tail), C6, and the G1 init control.
    Run BEFORE the main loop, floored by the t20 state (追補 1 R1 of clamp_horizon)."""
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if key in CH.FLOOR_OF else None
        return 100 * limit(ck_main, key, b)

    for mut, keys in [('skip', ['c3_cnorm_rel']), ('rowscale', ['c3_m_absdiff']), ('tail', ['clamp_tail_absdiff'])]:
        p2, act2, adam2, gens2 = restore(snap, arm)
        ck2 = {}
        if mut == 'skip':
            gp, gd, gb = gens2
            perm = torch.randperm(784, generator=gp)
            idx = H.stratified_draw(mnist, gd)
            order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
            xs = mnist.train_x[idx][:, perm][order]
            ys = mnist.train_y[idx][order]
            for step in range(1, 626):
                out = H.forward(p2, xs[(step - 1) * 16:step * 16], act2)
                gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p2)
                with torch.no_grad():
                    m_, v_, tc = adam2
                    tc[0] += 1
                    c1, c2 = 1 - .9 ** tc[0], 1 - .999 ** tc[0]
                    for q, g, mi, vi in zip(p2, gr, m_, v_):
                        mi.mul_(.9).add_(g, alpha=1 - .9)
                        vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                        q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                    if isinstance(act2, H.AdaptiveSnake):
                        act2.update(out[0], out[2])
                    if step != 300:
                        CH.apply_clamp(p2, 'wclamp', base, ck2)
                    CH.verify_clamped(p2, 'wclamp', base, ck2)
        else:
            CH.loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, clamp_from, 'wclamp', base, None, {}, ck2, mut=mut)
        for key in keys:
            got = ck2.get(key, 0.)
            res[f'ctl_wclamp_{mut}_{key}'] = got
            assert got > floor(key), (f'vacuous control wclamp/{mut}: {key} = {got:g} <= {floor(key):g}')

    p4, act4, adam4, gens4 = restore(snap, arm)
    ck4 = {}
    CH.measure(p4, act4, probe, probe.refs[0], False, mnist, ck4, mut_decomp=True)
    res['ctl_c6_var_rel'] = ck4['c6_var_rel']; res['ctl_c6_mean_abs'] = ck4['c6_mean_abs']
    assert ck4['c6_var_rel'] > floor('c6_var_rel') and ck4['c6_mean_abs'] > floor('c6_mean_abs'), ('vacuous C6 control', ck4)

    # G1: a perturbed init must break the reproduction of the anchor at its first task
    cmp, _, first, _ = GS.anchor(arm, seed)
    p5 = H.init_params(seed, torch.device('cpu'))
    with torch.no_grad():
        p5[0].data[0, 0] += 1e-3
    act5 = GS.make_act(arm)
    adam5 = ([torch.zeros_like(q) for q in p5], [torch.zeros_like(q) for q in p5], [0])
    gens5 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    u5, ck5 = {}, {}
    CH.loop(p5, act5, adam5, gens5, mnist, probe, 1, first, 'ref', None, [], u5, ck5)
    d, n5 = cmp(u5, first, first)
    res['ctl_g1_init'] = d; res['ctl_g1_compared'] = n5
    assert n5 > 0 and d > 1e-4, ('vacuous G1 control', d, n5)
    return res


def _assert(ck, tasks, controls):
    missing = [k for k in TOL if k not in ck]
    assert not missing, ('checks never recorded', missing)
    bad = {k: (ck[k], limit(ck, k)) for k in TOL if not ck[k] <= limit(ck, k)}
    assert not bad, ('check failed (value, limit)', bad)
    assert ck['g1_units_maxabs'] <= 1e-10, ('G1 failed', ck['g1_units_maxabs'])
    assert ck['g1_units_compared'] == ck['g1_units_expected'], ('G1 count guard', ck['g1_units_compared'], ck['g1_units_expected'])
    if controls:
        for cname, k in CONTROL_OF.items():
            assert cname in ck, ('a registered mutation control never ran', cname)
            assert ck[cname] > ck[k], ('control no longer beats the observed value', cname, ck[cname], k, ck[k])
            ck[f'margin_{k}'] = ck[cname] / (100 * ck[k]) if ck[k] > 0 else float('inf')


def run(arm, seed, tasks=TASKS, clamp_from=CLAMP_FROM, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{arm}_none_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(arm))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G2', ck)

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    CH.loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'ref', None, rows, units, ck)
    base = C.baselines(p[0])
    ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
    ck['decompose_crosscheck'] = C.decompose_crosscheck(p, probe)
    ck.update(base_sigma_m=base['sigma_m'], base_mbar0=base['mbar0'], base_cnorm_mean=float(base['c'].mean()))
    snap = C.snapshot(p, act, adam, gens)
    ck['controls_run'] = bool(controls)
    if controls:
        ck.update(_controls(arm, seed, snap, base, mnist, probe, ck, clamp_from))
        print('CONTROLS OK', prefix, round(time.monotonic() - t0, 1), 's', flush=True)

    for clamp in CLAMPS:
        p2, act2, adam2, gens2 = restore(snap, arm)
        ck['pressure'] = 0.
        CH.loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, tasks, clamp, base, rows, units, ck)
        ck[f'pressure_{clamp}'] = ck.pop('pressure', 0.)
        T.finite_guard([r for r in rows if r['clamp'] == clamp], f'{prefix}/{clamp}')
        print('DONE', prefix, clamp, round(time.monotonic() - t0, 1), 's', flush=True)

    cmp, _, first, note = GS.anchor(arm, seed)
    w, n = cmp(units, 1, tasks)
    ck.update(g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=expected_count(arm, seed, tasks), g1_anchor=note)
    _assert(ck, tasks, controls)

    for r in rows:
        r.update(arm=arm, iv='none', seed=seed)
    T.write_rows(out / f'{prefix}_rows.csv', rows)
    np.savez_compressed(out / f'{prefix}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{prefix}_provenance.json',
           dict(arm=arm, iv='none', seed=seed, tasks=tasks, clamp_from=clamp_from, clamps=CLAMPS, checks=ck,
                spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='sub-run of spec_gate_shape_0911: CPU continuation from the shared t20 state; '
                      'G1 covers the ref arm only (no committed wclamp trajectory for these arms)'))
    print('FINISHED', prefix, 'G1', w, f'({n} arrays vs {note})', round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--arms', default=None)
    ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        run('SN02', 0, tasks=24, clamp_from=21, out=SMOKE, controls=True)
        run('LR03', 0, tasks=24, clamp_from=21, out=SMOKE, controls=True)
        return
    if a.all:
        arms = a.arms.split(',') if a.arms else ARMS

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.gate_wclamp_0911', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
