"""Mediation for spec_sign_ladder_0912.md 追補 3: is the reversal's ~3 pt carried
by the width, by the depth, or by neither?

The ladder showed that removing GELU's deep-side reversal moves L_ref from 7.68 to
4.62 pt, but the intervention also moves the width (cnorm 9.13 -> 14.25) and the
depth (z_bar -6.89 -> -3.67).  Here the trunk t1..20 is run ONCE with GELU and the
branches switch BOTH the activation and the clamp at t21, so all six branches share
one snapshot and one set of clamp targets (`base`).  gap(c) = L(GELU,c) - L(GELUA,c)
is then the reversal's effect with mediator c held fixed at a value common to both.

Committed modules are used as they stand (CH.loop / CH.apply_clamp carry the clamp
and its invariance checks); the only local re-implementations are the ones
clamp_horizon_acts_0910 already documents, plus a `restore` that dispatches through
sign_ladder_0912.make_act so the branch can change activation.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time

import numpy as np
import torch

from src import clamp_horizon_0910 as CH
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
from src import sign_ladder_0912 as SL

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/mediation_0912'
TRUNK = 'GELU'                       # the shared t1..20 trunk
ACTS = ['GELU', 'GELUA']
CLAMPS = ['ref', 'wclamp', 'dclamp']
TASKS = 400
CLAMP_FROM = 21
TIME_CAP = 9000.          # 追補 3: GELUA+wclamp の 1 枝が 2635 s。上限は書き出しの後に走る


def restore(snap, arm):
    """C.restore / T.restore, but through sign_ladder's make_act so a branch can
    switch activation at the clamp point."""
    p = [q.clone().requires_grad_(True) for q in snap['p']]
    act = SL.make_act(arm)
    if snap['V'] is not None:
        act.V = [v.clone() for v in snap['V']]
    adam = ([x.clone() for x in snap['adam'][0]], [x.clone() for x in snap['adam'][1]],
            [snap['adam'][2]])
    gens = []
    for s in snap['gens']:
        g = torch.Generator(); g.set_state(s.clone()); gens.append(g)
    return p, act, adam, gens


def _wrap_measure(arm):
    orig = CH.measure

    def measure(p, act, probe, perm, want_acc=False, mnist=None, ck=None, mut_decomp=False):
        r, u = orig(p, act, probe, perm, want_acc, mnist, ck, mut_decomp)
        r.update(T.gate_stats(p, act, probe, perm, arm))
        r.update(SL.past_zc(p, act, probe, perm, SL.base_of(arm)))
        return r, u
    return orig, measure


def run(seed, tasks=TASKS, clamp_from=CLAMP_FROM, out=OUT):
    torch.set_num_threads(1)
    H.setup('cpu')
    T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck['selftest'] = SL._selftest(bases=('GELU',))

    orig_measure, wrapped = _wrap_measure(TRUNK)
    try:
        CH.measure = wrapped
        p = H.init_params(seed, torch.device('cpu'))
        act = SL.make_act(TRUNK)
        adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
        gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        trunk_units = {}
        CH.loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'ref', None,
                rows, trunk_units, ck)
        base = C.baselines(p[0])
        ck.update(base_cnorm_mean=float(base['c'].mean()), base_sigma_m=base['sigma_m'],
                  base_mbar0=base['mbar0'])
        snap = C.snapshot(p, act, adam, gens)
        for r in rows:
            r['act'] = TRUNK
        units.update({f'{TRUNK}_{k}': v for k, v in trunk_units.items()})
        print('TRUNK OK', seed, round(time.monotonic() - t0, 1), 's', flush=True)

        for a in ACTS:
            CH.measure = _wrap_measure(a)[1]
            for clamp in CLAMPS:
                p2, act2, adam2, gens2 = restore(snap, a)
                assert act2.name == a or (a in SL.BASES and type(act2).__name__ in ('GELU', 'SiLU')), \
                    ('branch got the wrong activation', a)
                br_rows, br_units = [], {}
                ck['pressure'] = 0.
                CH.loop(p2, act2, adam2, gens2, mnist, probe, clamp_from, tasks, clamp,
                        base, br_rows, br_units, ck)
                ck[f'pressure_{a}_{clamp}'] = ck.pop('pressure', 0.)
                T.finite_guard(br_rows, f's{seed}/{a}/{clamp}')
                for r in br_rows:
                    r['act'] = a
                rows.extend(br_rows)
                if a == TRUNK and clamp == 'ref':            # GM1 anchor
                    g1u = dict(trunk_units); g1u.update(br_units)
                    worst, n = T.g1_ref_units(TRUNK, seed, g1u, 1, tasks, ck)
                    assert n == len(T.UNIT_KEYS) * tasks, ('G1 count guard', n)
                    assert worst == 0., ('GM1 failed: ref branch does not reproduce the anchor', worst)
                units.update({f'{a}_{k}': v for k, v in br_units.items()})
                print('DONE', seed, a, clamp, round(time.monotonic() - t0, 1), 's', flush=True)
    finally:
        CH.measure = orig_measure
    assert CH.measure is orig_measure, 'committed module left monkeypatched'

    # clamp invariances, at the committed tolerances
    present = [k for k in CH.TOL if k in ck]
    assert {'c3_m_absdiff', 'c5_sd_rel', 'c5_cnorm_rel'} <= set(present), \
        ('a clamp never recorded its invariance -- unexercised code path', present)
    bad = {k: (ck[k], CH.limit(ck, k)) for k in present if not ck[k] <= CH.limit(ck, k)}
    assert not bad, ('clamp invariance failed (value, limit)', bad)
    ck['tol_checked'] = present

    T.write_rows(out / f's{seed}_rows.csv', rows)
    np.savez_compressed(out / f's{seed}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f's{seed}_provenance.json',
           dict(seed=seed, tasks=tasks, clamp_from=clamp_from, trunk=TRUNK,
                acts=ACTS, clamps=CLAMPS, checks=ck,
                scope='spec_sign_ladder_0912.md 追補 3: mediation. One GELU trunk to t20, '
                      'then six branches switching activation AND clamp at t21 so that '
                      'base and snapshot are common to all six.',
                code_sha256=T.sha(Path(__file__)),
                sign_ladder_sha256=T.sha(Path(SL.__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)),
                wall_seconds=wall, **T.provenance_base(mnist)))
    print('FINISHED', seed, round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        run(0, tasks=26, out=ROOT / 'results/_smoke_mediation_0912')
        return
    if a.all:
        def job(s):
            subprocess.run([sys.executable, '-m', 'src.mediation_0912', '--seed', str(s)], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, range(3)):
                pass
        return
    run(a.seed)


if __name__ == '__main__':
    main()
