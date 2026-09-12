"""Sign ladder on GELU's deep-side gate (box B): does the reversal CAUSE the loss?

Box B leaves this undecided.  GELU loses 7.68 pt against ReLU 4.78 and leaky 4.36,
and at matched absolute depth GELU units are about half as mobile as every
non-valley arm (scale-free P(rise 0.7 sd within 50 tasks): GELU 0.39-0.43, leaky
0.62-0.65, ReLU 0.74-0.75, ELU1 0.83-0.85).  The cross-arm ordering of mobility
matches the ordering of loss (Spearman 0.90, n=5), but the within-arm clamps
disagree with it: dclamp restores as much mobility as wclamp and a third of the
accuracy, and ReLU's wclamp halves mobility while restoring 73%.  Correlation
across five arms cannot settle it; an intervention on the reversal itself can.

This ladder changes ONLY the sign of the gate below the valley bottom.  The
forward map above z_c, the depth, the width, the optimiser, the data and the
random streams are all held fixed.

    arm     phi'(z), z < z_c     phi(z), z < z_c            what it is
    GELU    phi'(z) < 0          phi(z)            ->  0     committed reference
    GELUF   0                    phi(z_c)           = -0.16997   reversal removed
    GELUA   |phi'(z)| > 0        2 phi(z_c) - phi(z)-> -0.33994   reversal sign-flipped

GELUA is the tight arm: |gate| is identical to GELU's at EVERY z, so a GELU/GELUA
difference can only be the sign.  GELUF is the loose arm: it deletes the reversal
without putting anything in its place (box A ran this shape as `clamp0`).  Both
are needed -- GELUF alone confounds "wrong sign" with "no signal", and GELUA alone
confounds the sign with the bounded output it must have to carry a positive gate.

Controls, in the shape sub-run A used:
  G1  the GELU arm reproduces the committed long_horizon_acts_0910 per-unit
      arrays bit for bit (same Raptor Lake host class, AVX2)
  1.  measurement non-invasiveness, with a mutated variant that must break it
  2.  stream arm-independence: the new activations must not enter the RNG, with
      an extra-draw control showing the digest can see a stream being consumed
  3.  _selftest below: the two new gates against autograd, the sign structure,
      the |gate| identity for GELUA, and mutation controls for each
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
OUT = ROOT / 'results/sign_ladder_0912'
ARMS = ['GELU', 'GELUF', 'GELUA', 'SILU', 'SILUF', 'SILUA']
TASKS = 400
TIME_CAP = 2400.

# base -> (class, z_c, phi(z_c)).  phi(z_c) is tabulated and asserted in _selftest;
# for SiLU it is exactly z_c + 1 (valley_acts_0910 docstring).
BASES = {'GELU': (VA.GELU, VA.ZC['GELU'], -0.16997120747990369),
         'SILU': (VA.SiLU, VA.ZC['SILU'], -0.27846454276107385)}
ZC = VA.ZC['GELU']          # kept so the GELU ladder's code path is unchanged


class _Floor:
    """base above z_c, flat at phi(z_c) below it.  Gate is exactly 0 past the valley."""

    def __init__(self, base):
        cls, self.zc, self.phi_c = BASES[base]
        self._g = cls(); self.base = base
        self.name = base + 'F'; self.kind = self._g.kind + '_floor'; self.param = 1.0

    def phi(self, z):
        return self._g.phi(torch.clamp(z, min=self.zc))

    def dphi(self, z):
        # clamp: in float32 the base's phi' can be negative just ABOVE z_c (it
        # changes sign there), which would make this "non-valley" arm show a
        # negative gate.  See spec 追補 1.
        return torch.where(z >= self.zc, self._g.dphi(z).clamp(min=0.), torch.zeros_like(z))


class _Abs:
    """base above z_c, reflected about y = phi(z_c) below it.  |gate| == the base's
    at every z; only the sign on the far side differs."""

    def __init__(self, base):
        cls, self.zc, self.phi_c = BASES[base]
        self._g = cls(); self.base = base
        self.name = base + 'A'; self.kind = self._g.kind + '_abs'; self.param = 1.0

    def phi(self, z):
        g = self._g.phi(z)
        return torch.where(z >= self.zc, g, 2.0 * self.phi_c - g)

    def dphi(self, z):
        # |phi'| everywhere: phi' above z_c (where phi' >= 0), -phi' below it.
        # abs() rather than a branch so float32's sign flip at z_c cannot leak a
        # negative gate.  See spec 追補 1.
        return self._g.dphi(z).abs()


def base_of(arm):
    return arm[:-1] if arm not in BASES else arm


def make_act(arm):
    if arm in BASES:
        return BASES[arm][0]()
    return (_Floor if arm.endswith('F') else _Abs)(base_of(arm))


def _selftest(bases=('GELU', 'SILU')):
    """Checks that can fail, each with a mutation control."""
    return {b: _selftest_base(b) for b in bases}


def _selftest_base(b):
    z = torch.linspace(-14, 6, 200001, dtype=torch.float64)
    cls_b, ZC, PHI_C = BASES[b]
    base = cls_b()
    above, below = z >= ZC, z < ZC
    out = {'zc': ZC, 'phi_c': PHI_C, 'phi_at_zc': float(base.phi(torch.tensor(ZC, dtype=torch.float64)))}
    assert abs(out['phi_at_zc'] - PHI_C) < 1e-12, (b, 'PHI_C wrong', out['phi_at_zc'])
    for name, cls in [(b + 'F', _Floor), (b + 'A', _Abs)]:
        a = cls(b)
        # 1. identical to GELU above z_c (this is what "everything else held fixed" means)
        d_above = float((a.phi(z[above]) - base.phi(z[above])).abs().max())
        g_above = float((a.dphi(z[above]) - base.dphi(z[above])).abs().max())
        # 2. dphi is the true derivative of phi
        zg = z.clone().requires_grad_(True)
        auto = torch.autograd.grad(a.phi(zg).sum(), zg)[0]
        d_auto = float((a.dphi(z) - auto).abs().max())
        # 3. the gate is non-negative EVERYWHERE (the whole point of the ladder)
        min_gate = float(a.dphi(z).min())
        # 4. continuity of phi and of dphi at z_c
        e = torch.tensor([ZC - 1e-9, ZC + 1e-9], dtype=torch.float64)
        jump_phi = float((a.phi(e)[0] - a.phi(e)[1]).abs())
        jump_gate = float((a.dphi(e)[0] - a.dphi(e)[1]).abs())
        # 5. arm-specific identity
        if name.endswith('F'):
            ident = float(a.dphi(z[below]).abs().max())            # gate == 0 below
            ident_name = 'max|gate| below z_c (must be 0)'
        else:
            ident = float((a.dphi(z).abs() - base.dphi(z).abs()).abs().max())
            ident_name = 'max| |gate| - |GELU gate| | (must be 0)'
        out[name] = dict(phi_vs_gelu_above=d_above, gate_vs_gelu_above=g_above,
                         dphi_vs_autograd=d_auto, min_gate=min_gate,
                         jump_phi_at_zc=jump_phi, jump_gate_at_zc=jump_gate,
                         identity=ident, identity_is=ident_name)
        assert d_above == 0. and g_above == 0., (name, 'differs from GELU above z_c', d_above, g_above)
        assert d_auto < 1e-9, (name, 'dphi is not the derivative of phi', d_auto)
        assert min_gate >= 0., (name, 'gate goes negative -- the reversal was not removed', min_gate)
        assert jump_phi < 1e-8 and jump_gate < 1e-8, (name, 'discontinuous at z_c', jump_phi, jump_gate)
        assert ident < 1e-12, (name, ident_name, ident)
        # mutation controls: each check must be able to fail
        zb = torch.tensor([ZC - 0.75], dtype=torch.float64)   # near the most reversed point
        assert float(base.dphi(zb)) < -0.09, (b, 'vacuous: base not reversed below z_c')
        assert float(base.dphi(torch.tensor([ZC - 3.0], dtype=torch.float64))) < 0., \
            (b, 'vacuous: base not still reversed 3 below z_c')
        assert float(a.dphi(zb)) >= 0., (name, 'vacuous sign check')
        bad = cls(b); bad_phi = bad.phi(zb) + 1e-3
        assert float((bad_phi - a.phi(zb)).abs()) > 1e-6, 'vacuous: perturbed phi compares equal'
        if name.endswith('A'):       # the |gate| identity must not be trivially true
            assert float((a.dphi(zb) - base.dphi(zb)).abs()) > 1e-3, \
                'vacuous |gate| identity: abs arm gate equals the base gate'
    # 6. float32: the gate must not go negative THERE either, and the guard must
    #    not be vacuous -- raw GELU phi' really does flip sign near z_c in float32
    z32 = torch.linspace(ZC - 0.45, ZC + 0.45, 2000001, dtype=torch.float32)
    raw32 = base.dphi(z32)
    flips = int(((z32 >= ZC) & (raw32 < 0)).sum() + ((z32 < ZC) & (raw32 > 0)).sum())
    # The guard is DEMONSTRATED only where the raw base really flips sign in
    # float32 (GELU does; SiLU does not in this grid).  Recording the count keeps
    # that distinction visible instead of implying both were exercised.
    out['float32_sign_flips_in_raw_base'] = flips
    out['float32_guard_demonstrated'] = flips > 0
    for name, cls in [(b + 'F', _Floor), (b + 'A', _Abs)]:
        m32 = float(cls(b).dphi(z32).min())
        out[name]['float32_min_gate'] = m32
        assert m32 >= 0., (name, 'float32 gate goes negative near z_c', m32)

    # 7. the three arms must actually differ on the far side
    gates = {}
    for dz in (0.75, 3.0):
        zb = torch.tensor([ZC - dz], dtype=torch.float64)
        g = {k: float(make_act(k).dphi(zb)) for k in (b, b + 'F', b + 'A')}
        assert g[b] < 0 < g[b + 'A'] and g[b + 'F'] == 0., ('ladder collapsed', b, dz, g)
        gates['z=zc-%g' % dz] = g
    out['gates'] = gates
    return out


def past_zc(p, act, probe, perm, arm='GELU'):
    """beyond_frac / inv_units in T.gate_stats are hard-coded 0 for an arm that
    valley_acts does not tabulate, so the depth counter the ladder needs is
    recomputed here against GELU's z_c for all three arms alike."""
    with torch.no_grad():
        z = probe.px[:, perm] @ p[0].detach().T + p[1].detach()
        below = z < BASES[base_of(arm)][1]
        return dict(past_zc_frac=float(below.float().mean()),
                    past_zc_units=int(((below.float().mean(0)) > .5).sum()))


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
    act = make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    return p, act, adam, gens


def _train(arm, p, act, adam, gens, mnist, probe, t_to, rows, units, ck, measure_mut=False):
    """C.loop with C.MEAS and C.measure both scoped (see long_horizon_acts_0910)."""
    orig_meas, orig_measure = C.MEAS, C.measure

    def measure(p_, act_, probe_, perm, want_acc=False, mnist_=None, ck_=None, mut_decomp=False):
        r, u = orig_measure(p_, act_, probe_, perm, want_acc, mnist_, ck_, mut_decomp)
        r.update(T.gate_stats(p_, act_, probe_, perm, arm))
        r.update(past_zc(p_, act_, probe_, perm, arm))
        return r, u
    try:
        C.MEAS = [625]
        C.measure = measure
        C.loop(p, act, adam, gens, mnist, probe, 1, t_to, 'ref', None, rows, units, ck,
               measure_mut=measure_mut)
    finally:
        C.MEAS, C.measure = orig_meas, orig_measure
    assert C.measure is orig_measure and C.MEAS == orig_meas, 'committed module left mutated'
    return p


def ctl_noninvasive(arm, seed, mnist, probe, n=60):
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


def ctl_streams(arm, seed, mnist, n=5):
    """The activation must not enter the RNG.  Same digest-over-real-training
    construction as sub-run A, with the same extra-draw control."""
    def digest(a, sd, extra_draw=False):
        p, act, adam, gens = _fresh(a, sd) if a in LADDER else (None,) * 4
        if p is None:
            p = H.init_params(sd, torch.device('cpu')); act = T.make_act(a)
            adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
            gens = [H.stream('perm', sd), H.stream('data', sd), H.stream('batch', sd)]
        gp, gd, gb = gens
        h = hashlib.sha256()
        for _ in range(n):
            if extra_draw:
                torch.randperm(2, generator=gb)
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
    consumed = digest('LR', seed, extra_draw=True)
    assert mine == ref, ('the activation entered the RNG', arm, seed, mine, ref)
    assert ref != other, 'vacuous stream control: a different seed gave the same draws'
    assert ref != consumed, 'vacuous stream control: an extra draw did not change the digest'
    return dict(stream_tasks=n, stream_sha256=mine, stream_matches_LR=True,
                stream_control_seed_differs=True, stream_control_extra_draw_differs=True)


def run(arm, seed, tasks=TASKS, out=OUT, controls=True):
    t0 = time.monotonic()
    mnist, probe = _setup(seed)
    rows, units, ck = [], {}, {}
    ck['selftest'] = _selftest()
    ck['controls_run'] = bool(controls)
    if controls:
        ck.update(ctl_noninvasive(arm, seed, mnist, probe))
        ck.update(ctl_streams(arm, seed, mnist))
        print('CONTROLS OK', arm, seed, round(time.monotonic() - t0, 1), 's', flush=True)
    p, act, adam, gens = _fresh(arm, seed)
    _train(arm, p, act, adam, gens, mnist, probe, tasks, rows, units, ck)
    T.finite_guard(rows, f'{arm}_s{seed}')
    T.check_gates(rows, arm, ck)
    if arm in BASES:                                    # G1: bit identity with sub-run A
        worst, n = T.g1_ref_units(arm, seed, units, 1, tasks, ck)
        assert n == len(T.UNIT_KEYS) * tasks, ('G1 compared the wrong number of keys', n)
        assert worst == 0., ('G1 failed: this host does not reproduce sub-run A', worst)
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.update(arm=arm, iv='none', seed=seed)
    T.write_rows(out / f'{arm}_none_s{seed}_rows.csv', rows)
    np.savez_compressed(out / f'{arm}_none_s{seed}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{arm}_none_s{seed}_provenance.json',
           dict(arm=arm, iv='none', seed=seed, tasks=tasks, meas=[625],
                scope='sign ladder on the deep-side gate of GELU; spec_sign_ladder_0912.md',
                checks=ck, code_sha256=T.sha(Path(__file__)), wall_seconds=wall,
                **T.provenance_base(mnist)))
    print('FINISHED', arm, seed, 'past_zc_units', rows[-1]['past_zc_units'],
          'acc(last)', round(rows[-1]['acc'], 4), round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--arms', default=None)
    ap.add_argument('--no-controls', dest='controls', action='store_false')
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(_selftest(), indent=1)); return
    if a.smoke:
        for arm in (a.arms.split(',') if a.arms else ARMS):
            run(arm, 0, tasks=8, out=ROOT / 'results/_smoke_sign_ladder_0912', controls=False)
        return
    if a.all:
        def job(j):
            cmd = [sys.executable, '-m', 'src.sign_ladder_0912', '--arm', j[0], '--seed', str(j[1])]
            if j[1] != 0:
                cmd.append('--no-controls')
            subprocess.run(cmd, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            todo = a.arms.split(',') if a.arms else ARMS
            for _ in pool.map(job, [(arm, s) for arm in todo for s in range(3)]):
                pass
        return
    run(a.arm, a.seed, controls=a.controls)


if __name__ == '__main__':
    main()
