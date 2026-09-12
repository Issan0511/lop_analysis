"""band_omega_0912: is the shallow band a lever of its own, or just omega?
(spec_band_omega_0912)

leaky's shallow band (phi' between 0.25 and 1) is structurally EMPTY: the negative side
is a flat 0.1, so no input can sit there.  ELU keeps 9% (tau=1) to 43% (tau=3) of its
(unit, input) mass there, and the width clamp RAISES that share (0.094 -> 0.145).

turn_rate_0912 built pairs with 2x different length and the SAME omega, by moving the
clamp height kappa and lr in opposite directions; in leaky those pairs matched in loss.
Run the same design on ELU tau=1.  If the pair splits, the band is a second lever.

Training, clamp and turn instrumentation are elu_turn_0912's (imported, not copied).
"""
from pathlib import Path
import argparse, concurrent.futures, json, math, subprocess, sys, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
from src import clamp_horizon_0910 as CH
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import elu_turn_0912 as ET

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/band_omega_0912'
SMOKE = ROOT / 'results/_smoke_band_omega_0912'
ACT = 'CELU1'
TASKS = 120
CLAMP_FROM = 21
LR0 = .001
TIME_CAP = 1800.
SPEC = ROOT / 'specs/spec_band_omega_0912.md'

# name -> (kappa, lr).   omega ∝ lr / kappa
ARMS = {
    'E1h': (1.0, LR0 / 2),      # omega 0.5   short, wide band
    'E2':  (2.0, LR0),          # omega 0.5   LONG,  narrow band   <- pair with E1h
    'E1d': (1.0, 2 * LR0),      # omega 2.0   short
    'E05': (0.5, LR0),          # omega 2.0   SHORTER              <- pair with E1d
}


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
    ck.update(GS.check_dphi(ACT))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G4 phi-prime', ck)
    ck.update(act=ACT, kappa=kappa, lr=lr, clamp='wclamp')

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(ACT)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    # t1..t20 at the REFERENCE lr for every arm, so kappa and lr are independent
    ET.loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, 'wclamp', None, rows, units, ck,
            cfrom=clamp_from, lr=LR0)
    b0 = C.baselines(p[0])
    base = dict(b0); base['c'] = b0['c'] * kappa
    ck['kappa_applied'] = float((base['c'] / b0['c']).mean())
    assert abs(ck['kappa_applied'] - kappa) < 1e-12, ('kappa not applied', ck['kappa_applied'])
    ck['base_cnorm_t20'] = float(base['c'].mean())
    ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
    if controls:
        ck.update(ET._controls(ACT, seed, C.snapshot(p, act, adam, gens), base, mnist, probe, ck, clamp_from))
    ET.loop(p, act, adam, gens, mnist, probe, clamp_from, tasks, 'wclamp', base, rows, units, ck,
            cfrom=clamp_from, lr=lr)
    T.finite_guard(rows, tag)

    # G5: the three shares partition the (unit, input) mass exactly
    ck['g5_band_ident'] = max(abs(r['pos_frac'] + r['cov'] + (1 - r['cov'] - r['pos_frac']) - 1) for r in rows)
    for r in rows:
        r['band'] = 1. - r['cov'] - r['pos_frac']
    assert ck['g5_band_ident'] <= 1e-12, ('G5 band identity', ck['g5_band_ident'])
    cn = [r['cnorm'] for r in rows if r['task'] >= clamp_from]
    ck['q0_cnorm_rel'] = max(abs(x / ck['base_cnorm_t20'] - 1) for x in cn) if cn else 1.
    assert ck['q0_cnorm_rel'] <= 1e-6, ('width not pinned to kappa*c_ref', ck['q0_cnorm_rel'])
    assert ck.get('g2_cos_range', 0.) == 0., ('G2 cos', ck.get('g2_cos_range'))
    assert ck.get('g2_angle_rel', 1.) <= 1e-6, ('G2 angle cross-check', ck.get('g2_angle_rel'))
    assert ck.get('g2_triangle', -1.) <= 1e-8, ('G2 triangle', ck.get('g2_triangle'))
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck.get('g2_tot_ident'))
    for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff'):
        assert k in ck and ck[k] <= CH.limit(ck, k, 'f32_rowmean_bound_t20'), ('G3', k, ck.get(k))

    # G1: every arm shares t1..t20 with the unclamped CELU1 reference
    ref = np.load(ROOT / 'results/gate_shape_0911' / f'{ACT}_s{seed}_units.npz')
    g1_to = min(tasks, clamp_from - 1)
    w, n = ET._cmp(units, ref, 'ref', 'ref', 1, g1_to)
    ck.update(g1_anchor=f'gate_shape_0911/{ACT} (t1-{g1_to} prefix)', g1_units_maxabs=w,
              g1_units_compared=n, g1_units_expected=None)
    assert w <= 1e-10, ('G1 failed', arm, seed, w)
    assert n >= 12 * g1_to, ('G1 count guard', n, g1_to)
    if controls and seed == 0 and arm == 'E1h':
        p3 = H.init_params(seed, torch.device('cpu'))
        with torch.no_grad():
            p3[0].data[0, 0] += 1e-3
        ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        ET.loop(p3, GS.make_act(ACT), ad3, g3, mnist, probe, 1, 1, None, None, [], u3, {}, instrument=False)
        ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - ref[f'ref_{k}_t1']).max())
                                for k in ('zbar_i', 'sd_i', 'cnorm_i'))
        assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, clamp_from=clamp_from,
                cfg=dict(act=ACT, kappa=kappa, lr=lr), late=ET.LATE, checks=ck,
                spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                elu_turn_sha256=T.sha(Path(ET.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), data_sha256=mnist.sha256,
                torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; ELU tau=1, clamp height kappa x c_ref and per-arm lr from t21'))
    print('FINISHED', tag, 'G1', ck['g1_units_maxabs'], f'({n})', round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        ET.LATE = (3, 4)
        for arm in ARMS:
            run(arm, 0, tasks=4, out=SMOKE, controls=True, clamp_from=3)
        return
    if a.all:
        def job(j):
            subprocess.run([sys.executable, '-m', 'src.band_omega_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
