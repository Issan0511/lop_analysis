"""Sub-run A of spec_transport_holes_0910: GELU / SiLU reference trajectories to t400.

Box B has never run a valley activation.  This makes the trajectory that sub-runs
C and D then anchor on, and answers A1/A2/A3.

`src/long_horizon_0910.py` is deliberately NOT imported: importing it assigns
`C.MEAS = [625]` process-wide and monkeypatches `C.measure` with no try/finally,
so an exception anywhere leaves a committed module permanently wrapped.  Its
forty lines are reproduced here with both mutations scoped.

There is NO committed reference for GELU/SiLU, so there is no G1 and the
provenance says so (`no_committed_reference: true`).  In its place, spec §3.1:
  1. measurement non-invasiveness   (measured vs unmeasured -> bit-identical state)
  2. stream arm-independence        (the perms/data/batch draws during real
                                     training match LR's, byte for byte)
  3. valley_acts_0910._selftest     (phi vs torch, dphi vs autograd, the valley
                                     bottom, the sign structure, min_dphi)
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, subprocess, sys, time

import numpy as np
import torch

from src import width_sink_clamp_0909 as C
from src import valley_acts_0910 as VA
from src import transport_common_0910 as T

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/long_horizon_acts_0910'
ARMS = ['GELU', 'SILU']
TASKS = 400
TIME_CAP = 1800.


def _setup(seed):
    torch.set_num_threads(1)
    H.setup('cpu')
    T.data_dir()
    mnist = H.Mnist(torch.device('cpu'))
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    return mnist, probe


def _fresh(arm, seed, mutate=False):
    p = H.init_params(seed, torch.device('cpu'))
    if mutate:
        with torch.no_grad():
            p[0].data[0, 0] += 1e-3
    act = T.make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    return p, act, adam, gens


def _train(arm, p, act, adam, gens, mnist, probe, t_to, rows, units, ck, measure_mut=False):
    """C.loop with C.MEAS and C.measure both scoped, and the gate counters added."""
    orig_meas, orig_measure = C.MEAS, C.measure

    def measure(p_, act_, probe_, perm, want_acc=False, mnist_=None, ck_=None, mut_decomp=False):
        r, u = orig_measure(p_, act_, probe_, perm, want_acc, mnist_, ck_, mut_decomp)
        r.update(T.gate_stats(p_, act_, probe_, perm, arm))
        return r, u
    try:
        C.MEAS = [625]                                   # task ends only
        C.measure = measure
        C.loop(p, act, adam, gens, mnist, probe, 1, t_to, 'ref', None, rows, units, ck,
               measure_mut=measure_mut)
    finally:
        C.MEAS, C.measure = orig_meas, orig_measure
    assert C.measure is orig_measure and C.MEAS == orig_meas, 'committed module left mutated'
    return p


# --------------------------------------------------------------- spec §3.1 (1)
def ctl_noninvasive(arm, seed, mnist, probe, n=60):
    """The measured run and an unmeasured run must end in bit-identical state,
    and the `measure_mut` variant (W1 += 1e-9 at each measurement) must break it."""
    out = {}
    for tag, rows, mut in [('measured', [], False), ('unmeasured', None, False),
                           ('mutated', [], True)]:
        p, act, adam, gens = _fresh(arm, seed)
        _train(arm, p, act, adam, gens, mnist, probe, n, rows, {}, {}, measure_mut=mut)
        out[tag] = [q.detach().clone() for q in p]
    same = max(float((a - b).abs().max()) for a, b in zip(out['measured'], out['unmeasured']))
    broke = max(float((a - b).abs().max()) for a, b in zip(out['measured'], out['mutated']))
    assert same == 0., ('measurement perturbs the trajectory', arm, seed, same)
    assert broke > 1e-6, ('vacuous non-invasiveness control', arm, seed, broke)
    return dict(noninvasive_tasks=n, noninvasive_maxabs=same, noninvasive_control=broke)


# --------------------------------------------------------------- spec §3.1 (2)
def ctl_streams(arm, seed, mnist, n=5):
    """The activation must not enter the RNG.  Hashing the draws made DURING real
    training (not a dry draw, which would be trivially equal) makes this
    falsifiable: any arm-dependent consumption of a generator changes the hash."""

    def digest(a, sd):
        p, act, adam, gens = _fresh(a, sd)
        gp, gd, gb = gens
        h = hashlib.sha256()
        for _ in range(n):
            perm = torch.randperm(784, generator=gp)
            idx = H.stratified_draw(mnist, gd)
            order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
            for t in (perm, idx, order):
                h.update(t.numpy().tobytes())
            xs = mnist.train_x[idx][:, perm][order]
            ys = mnist.train_y[idx][order]
            for step in range(1, 626):
                out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
                gr = torch.autograd.grad(
                    torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p)
                with torch.no_grad():
                    m_, v_, tc = adam
                    tc[0] += 1
                    c1, c2 = 1 - .9 ** tc[0], 1 - .999 ** tc[0]
                    for q, g, mi, vi in zip(p, gr, m_, v_):
                        mi.mul_(.9).add_(g, alpha=1 - .9)
                        vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                        q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
        return h.hexdigest()

    mine, ref = digest(arm, seed), digest('LR', seed)
    other = digest('LR', seed + 1)
    assert mine == ref, ('the activation entered the RNG', arm, seed, mine, ref)
    assert ref != other, 'vacuous stream control: a different seed gave the same draws'
    return dict(stream_tasks=n, stream_sha256=mine, stream_matches_LR=True,
                stream_control_differs=True)


def run(arm, seed, tasks=TASKS, out=OUT, controls=True):
    t0 = time.monotonic()
    mnist, probe = _setup(seed)
    rows, units, ck = [], {}, {}
    ck['selftest'] = VA._selftest()                                   # §3.1 (3)
    ck['controls_run'] = bool(controls)
    if controls:
        ck.update(ctl_noninvasive(arm, seed, mnist, probe))           # §3.1 (1)
        ck.update(ctl_streams(arm, seed, mnist))                      # §3.1 (2)
        print('CONTROLS OK', arm, seed, round(time.monotonic() - t0, 1), 's', flush=True)
    p, act, adam, gens = _fresh(arm, seed)
    _train(arm, p, act, adam, gens, mnist, probe, tasks, rows, units, ck)
    T.finite_guard(rows, f'{arm}_s{seed}')
    T.check_gates(rows, arm, ck)
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.update(arm=arm, iv='none', seed=seed)
    T.write_rows(out / f'{arm}_none_s{seed}_rows.csv', rows)
    np.savez_compressed(out / f'{arm}_none_s{seed}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{arm}_none_s{seed}_provenance.json',
           dict(arm=arm, iv='none', seed=seed, tasks=tasks, meas=[625],
                no_committed_reference=True,
                scope='sub-run A of spec_transport_holes_0910: first box-B trajectory for a '
                      'valley activation. There is no committed reference to reproduce, so '
                      'there is NO G1 here; §3.1 controls stand in its place.',
                checks=ck, code_sha256=T.sha(Path(__file__)), wall_seconds=wall,
                **T.provenance_base(mnist)))
    print('FINISHED', arm, seed, 'inv_units(last)', rows[-1]['inv_units'],
          'acc(last)', round(rows[-1]['acc'], 4), round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-controls', dest='controls', action='store_false')
    a = ap.parse_args()
    if a.smoke:
        # 8 tasks: enough for the gate check to see the valley, and the §3.1
        # controls run at their own (fixed) horizons regardless.
        run('GELU', 0, tasks=8, out=ROOT / 'results/_smoke_long_horizon_acts', controls=True)
        return
    if a.all:
        # The §3.1 controls test properties of the harness, not of the seed, so
        # they run once per arm (seed 0) rather than six times.
        def job(j):
            cmd = [sys.executable, '-m', 'src.long_horizon_acts_0910',
                   '--arm', j[0], '--seed', str(j[1])]
            if j[1] != 0:
                cmd.append('--no-controls')
            subprocess.run(cmd, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed, controls=a.controls)


if __name__ == '__main__':
    main()
