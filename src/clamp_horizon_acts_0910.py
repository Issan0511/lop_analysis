"""Sub-runs B and C of spec_transport_holes_0910: the clamp_horizon intervention
on ReLU (B) and on GELU / SiLU (C).

`src/clamp_horizon_0910.py` is imported, never edited -- its sha256 is in the
provenance of its own committed LR/ELU1 results.  The training loop, the clamps
and their invariance checks are used AS THEY STAND (`CH.loop`, `CH.apply_clamp`,
`CH.verify_clamped`, `CH.clamped_W`), so the arm-independent physics is
byte-identical to the committed run and G1 is what proves it.

What is re-implemented locally, and why (spec §7.5):

  run()        CH.run captures TASKS / CLAMP_FROM / OUT as default arguments at
               def time, so rebinding the module globals silently does nothing.
  restore()    C.restore dispatches through C.make_act, which KeyErrors on
               GELU/SILU.  (R needs none of this: C.make_act('R') already works.)
  g1_check()   CH.g1_check hard-wires the count guards to a four-clamp
               width_sink reference (240 rows / 1200 arrays).  These arms have a
               ref-only anchor: 600 arrays, and for GELU/SiLU no committed
               anchor at all until sub-run A has run.
  _controls()  CH._controls reads the module-global CLAMP_FROM instead of the
               argument, and -- the reason this matters -- runs AFTER the main
               loop, so a failing control throws away ~15 minutes of finished
               computation.  Here it runs FIRST, and the "control must also beat
               the observed value by 100x" half that ordering costs is restored
               by re-asserting it after the loop.

CH.measure is wrapped (with try/finally, unlike long_horizon_0910) to add the
gate counters of spec §2.1: dead_hard/dead_soft/sat threshold the SIGNED gate
and therefore read an inverted valley unit as dead.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, json, subprocess, sys, time

import numpy as np
import torch

from src import clamp_horizon_0910 as CH
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/clamp_horizon_acts_0910'
ARMS = ['R', 'GELU', 'SILU']
CLAMPS = CH.CLAMPS                      # ref, dclamp, wclamp, wcap2
CAP = CH.CAP                            # 2.0
TASKS = 400
CLAMP_FROM = 21
G1_LAST = 100
TIME_CAP = 2400.
TOL = CH.TOL


def limit(ck, k, bound_key=None):
    return CH.limit(ck, k, bound_key)


# ------------------------------------------------------------------ measurement
def _wrap_measure(arm):
    """Return (install, uninstall) for the gate-counter wrapper around CH.measure."""
    orig = CH.measure

    def measure(p, act, probe, perm, want_acc=False, mnist=None, ck=None, mut_decomp=False):
        r, u = orig(p, act, probe, perm, want_acc, mnist, ck, mut_decomp)
        r.update(T.gate_stats(p, act, probe, perm, arm))
        return r, u
    return orig, measure


# ------------------------------------------------------------------------- G1
def g1_check(arm, seed, units, ck, tasks):
    """The `ref` arm's per-unit arrays over t1..min(tasks,100) against the anchor.

    Six arrays, not the five CH.g1_check uses: the anchors carry bias_i too.
    Only the `ref` arm is covered -- neither R nor the valley arms have committed
    dclamp/wclamp trajectories -- so this is weaker than the committed run's G1
    and the provenance says which anchor it leaned on."""
    t_to = min(tasks, G1_LAST)
    worst, n = T.g1_ref_units(arm, seed, units, 1, t_to, ck)
    return worst, n


# -------------------------------------------------------------------- controls
def _controls(arm, seed, snap, base, mnist, probe, ck_main, clamp_from):
    """Ten mutation controls, each of which must break its own check by 100x.

    Runs BEFORE the main loop.  The floor therefore cannot use the run's own
    running maxima, which is exactly right: 追補 1 R1 showed that flooring an
    absolute row-mean check by the whole-run running max makes the control fail
    at t400 even though nothing is wrong.  The controls all re-run from t20, so
    `f32_rowmean_bound_t20` is the state they actually live in."""
    res = {}

    def floor(key):
        b = 'f32_rowmean_bound_t20' if key in CH.FLOOR_OF else None
        return 100 * limit(ck_main, key, b)

    def above_cap():
        """At t21 no row is near the ceiling, so a plain trajectory mutation of
        wcap2 would be vacuous.  Build a state 3x over it and drive it through
        `loop`, so the in-loop check is what fires rather than a hand call."""
        p3, act3, adam3, gens3 = T.restore(snap, arm)
        with torch.no_grad():
            Wd, m, Wt = C.rows_of(p3[0])
            p3[0].data.copy_((m + Wt * (3 * base['c'] / Wt.norm(dim=1, keepdim=True))).float())
        return p3, act3, adam3, gens3

    p3, act3, adam3, gens3 = above_cap()
    ck3 = {}
    CH.loop(p3, act3, adam3, gens3, mnist, probe, clamp_from, clamp_from, 'wcap2',
            base, None, {}, ck3, mut='cap4')
    res['ctl_wcap2_cap4_excess_rel'] = ck3.get('cap_excess_rel', 0.)
    assert res['ctl_wcap2_cap4_excess_rel'] > floor('cap_excess_rel'), ('vacuous cap4 control', ck3)

    p3, act3, adam3, gens3 = above_cap()
    ck3b = {}
    CH.loop(p3, act3, adam3, gens3, mnist, probe, clamp_from, clamp_from, 'wcap2',
            base, None, {}, ck3b)
    res['ctl_wcap2_restored_excess_rel'] = ck3b.get('cap_excess_rel', 0.)
    res['ctl_wcap2_restore_rel'] = ck3b.get('cap_restore_rel', 0.)
    res['ctl_wcap2_bind_steps'] = ck3b.get('cap_bind_steps', 0)
    assert ck3b.get('cap_excess_rel', 1.) <= limit(ck_main, 'cap_excess_rel'), \
        ('cap does not restore', ck3b)
    assert ck3b.get('cap_bind_steps', 0) > 0, 'the above-cap control never bound'

    for clamp, mut, keys in [('wclamp', 'skip', ['c3_cnorm_rel']),
                             ('wclamp', 'rowscale', ['c3_m_absdiff']),
                             ('wclamp', 'tail', ['clamp_tail_absdiff']),
                             ('dclamp', 'p90', ['c5_mbar_abs']),
                             ('dclamp', 'rowscale', ['c5_cnorm_rel', 'c5_sd_rel'])]:
        p2, act2, adam2, gens2 = T.restore(snap, arm)
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
                gr = torch.autograd.grad(torch.nn.functional.cross_entropy(
                    out[4], ys[(step - 1) * 16:step * 16]), p2)
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
                        CH.apply_clamp(p2, clamp, base, ck2)
                    CH.verify_clamped(p2, clamp, base, ck2)
        else:
            CH.loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, clamp_from, clamp,
                    base, None, {}, ck2, mut=mut)
        for key in keys:
            got = ck2.get(key, 0.)
            res[f'ctl_{clamp}_{mut}_{key}'] = got
            assert got > floor(key), \
                (f'vacuous control {clamp}/{mut}: {key} = {got:g} <= {floor(key):g}')

    p4, act4, adam4, gens4 = T.restore(snap, arm)
    ck4 = {}
    CH.measure(p4, act4, probe, probe.refs[0], False, mnist, ck4, mut_decomp=True)
    res['ctl_c6_var_rel'] = ck4['c6_var_rel']
    res['ctl_c6_mean_abs'] = ck4['c6_mean_abs']
    assert ck4['c6_var_rel'] > floor('c6_var_rel') and ck4['c6_mean_abs'] > floor('c6_mean_abs'), \
        ('vacuous C6 control', ck4)

    # G1: a perturbed init must break the reproduction of the anchor at t21
    p5 = H.init_params(seed, torch.device('cpu'))
    with torch.no_grad():
        p5[0].data[0, 0] += 1e-3
    act5 = T.make_act(arm)
    adam5 = ([torch.zeros_like(q) for q in p5], [torch.zeros_like(q) for q in p5], [0])
    gens5 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    u5, ck5 = {}, {}
    CH.loop(p5, act5, adam5, gens5, mnist, probe, 1, clamp_from, 'ref', None, [], u5, ck5)
    d, n5 = T.g1_ref_units(arm, seed, u5, clamp_from, clamp_from, ck5, prefix='ctl_g1')
    res['ctl_g1_init'] = d
    res['ctl_g1_compared'] = n5
    assert n5 == len(T.UNIT_KEYS), ('vacuous G1 control: nothing compared', n5)
    assert d > 100 * 1e-10, f'vacuous G1 control: {d:g}'
    return res


# Which mutation control guards which check.  Written out rather than matched by
# suffix: a suffix rule that happens to match nothing turns the whole re-check
# into a no-op, which is the exact shape of the seven vacuous checks this
# codebase has already shipped.
CONTROL_OF = {
    'ctl_wcap2_cap4_excess_rel': 'cap_excess_rel',
    'ctl_wclamp_skip_c3_cnorm_rel': 'c3_cnorm_rel',
    'ctl_wclamp_rowscale_c3_m_absdiff': 'c3_m_absdiff',
    'ctl_wclamp_tail_clamp_tail_absdiff': 'clamp_tail_absdiff',
    'ctl_dclamp_p90_c5_mbar_abs': 'c5_mbar_abs',
    'ctl_dclamp_rowscale_c5_cnorm_rel': 'c5_cnorm_rel',
    'ctl_dclamp_rowscale_c5_sd_rel': 'c5_sd_rel',
    'ctl_c6_var_rel': 'c6_var_rel',
    'ctl_c6_mean_abs': 'c6_mean_abs',
}


def _assert(ck, arm, tasks, controls=True):
    missing = [k for k in TOL if k not in ck]
    assert not missing, ('checks never recorded (a silently unexercised code path)', missing)
    bad = {k: (ck[k], limit(ck, k)) for k in TOL if not ck[k] <= limit(ck, k)}
    assert not bad, ('check failed (value, limit)', bad)
    assert ck['g1_units_maxabs'] <= 1e-10, ('G1 failed', ck['g1_units_maxabs'])
    assert ck['g1_units_compared'] >= ck['g1_units_expected'], \
        ('G1 count guard', ck['g1_units_compared'], ck['g1_units_expected'])
    if tasks >= 100:
        assert ck.get('cap_bind_steps', 0) > 0, 'wcap2 never bound: the arm is identical to ref'
    if not controls:
        return
    # _controls already required every control to beat 100x its tolerance, at the
    # t20 state it runs in.  Running the controls first costs the other half of
    # the old rule -- "and it beats what the run actually produced" -- so restore
    # that here.  The 100x form is NOT re-asserted: at t400 the f32 row-mean bound
    # is ~2x its t20 value, so a 100x re-assert would fail for a healthy run,
    # which is 追補 1 R1 in a new disguise.  The margin is recorded instead.
    seen = 0
    for cname, k in CONTROL_OF.items():
        assert cname in ck, ('a registered mutation control never ran', cname)
        seen += 1
        assert ck[cname] > ck[k], \
            ('control no longer beats the observed value', cname, ck[cname], k, ck[k])
        ck[f'margin_{k}'] = ck[cname] / (100 * ck[k]) if ck[k] > 0 else float('inf')
    assert seen == len(CONTROL_OF), 'control table did not run'


def run(arm, seed, tasks=TASKS, clamp_from=CLAMP_FROM, out=OUT, controls=True):
    torch.set_num_threads(1)
    H.setup('cpu')
    T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{arm}_none_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}

    orig_measure, wrapped = _wrap_measure(arm)
    try:
        CH.measure = wrapped
        p = H.init_params(seed, torch.device('cpu'))
        act = T.make_act(arm)
        adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
        gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        CH.loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'ref', None,
                rows, units, ck)
        base = C.baselines(p[0])
        ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
        ck['decompose_crosscheck'] = C.decompose_crosscheck(p, probe)
        ck.update(base_sigma_m=base['sigma_m'], base_mbar0=base['mbar0'],
                  base_cnorm_mean=float(base['c'].mean()))
        snap = C.snapshot(p, act, adam, gens)

        ck['controls_run'] = bool(controls)
        if controls:                                   # FIRST, not last: spec §6
            ck.update(_controls(arm, seed, snap, base, mnist, probe, ck, clamp_from))
            print('CONTROLS OK', prefix, round(time.monotonic() - t0, 1), 's', flush=True)

        for clamp in CLAMPS:
            p2, act2, adam2, gens2 = T.restore(snap, arm)
            ck['pressure'] = 0.
            CH.loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, tasks, clamp,
                    base, rows, units, ck)
            ck[f'pressure_{clamp}'] = ck.pop('pressure', 0.)
            T.finite_guard([r for r in rows if r['clamp'] == clamp], f'{prefix}/{clamp}')
            print('DONE', prefix, clamp, round(time.monotonic() - t0, 1), 's', flush=True)
    finally:
        CH.measure = orig_measure
    assert CH.measure is orig_measure, 'committed module left monkeypatched'

    g1_check(arm, seed, units, ck, tasks)
    T.check_gates([r for r in rows if r['clamp'] == 'ref'], arm, ck)
    _assert(ck, arm, tasks, controls)

    for r in rows:
        r.update(arm=arm, iv='none', seed=seed)
    T.write_rows(out / f'{prefix}_rows.csv', rows)
    np.savez_compressed(out / f'{prefix}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{prefix}_provenance.json',
           dict(arm=arm, iv='none', seed=seed, tasks=tasks, clamp_from=clamp_from,
                clamps=CLAMPS, cap=CAP, checks=ck,
                code_sha256=T.sha(Path(__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)),
                wall_seconds=wall,
                scope='sub-run B/C of spec_transport_holes_0910: CPU continuation from the '
                      'shared t20 state. G1 covers the ref arm only (no committed dclamp/'
                      'wclamp trajectory exists for this activation).',
                **T.provenance_base(mnist)))
    print('FINISHED', prefix, 'G1', ck['g1_units_maxabs'],
          f"({ck['g1_units_compared']} arrays vs {ck['g1_anchor']})",
          round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--arms', default=None, help='comma list for --all')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-controls', dest='controls', action='store_false')
    a = ap.parse_args()
    if a.smoke:
        # R, 32 tasks: ReLU's fastest row crosses 2x its t20 norm at t27-28, so
        # wcap2 actually binds and `cap_m_absdiff` / `cap_restore_rel` get
        # recorded.  (The committed --smoke cannot pass for LR/ELU1 for exactly
        # this reason: no row is within 25% of the ceiling by t24.)
        run('R', 0, tasks=32, clamp_from=21,
            out=ROOT / 'results/_smoke_clamp_horizon_acts', controls=True)
        return
    if a.all:
        arms = a.arms.split(',') if a.arms else ARMS

        def job(j):
            subprocess.run([sys.executable, '-m', 'src.clamp_horizon_acts_0910',
                            '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in arms for s in range(3)]):
                pass
        return
    run(a.arm, a.seed, controls=a.controls)


if __name__ == '__main__':
    main()
