"""tau_clamp_0912: does ELU's saturation harm run THROUGH the width?  (spec_tau_clamp_0912)

The tau dial moves the loss by 2.0 pt at nearly constant width (2.68 / 1.86 / 0.65 pt for
tau = 0.3 / 1 / 3), so saturation IS harmful inside the ELU family.  What is special about
ELU is that pinning the width removes essentially ALL of its loss at tau = 1, while leaky
keeps 0.5 pt.  The missing point is tau = 0.3 WITH the clamp:

    S(tau) = L(wclamp)              what survives the width clamp
    R_c(tau) = L(ref) - L(wclamp)   what the width clamp removes

Training, clamp and instrumentation are elu_turn_0912's (imported, not copied); only the
arm table and the G1 anchors differ.  tau = 1 is carried in from that run.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
from src import clamp_horizon_0910 as CH
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import elu_turn_0912 as ET

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/tau_clamp_0912'
SMOKE = ROOT / 'results/_smoke_tau_clamp_0912'
TASKS = 120
CLAMP_FROM = 21
LR0 = .001
TIME_CAP = 1800.
SPEC = ROOT / 'specs/spec_tau_clamp_0912.md'

# name -> (activation arm, clamp, gate_shape_0911 anchor)
ARMS = {
    'T03':  ('CELU03', None,     'CELU03'),
    'T03w': ('CELU03', 'wclamp', 'CELU03'),
    'T3':   ('CELU3',  None,     'CELU3'),
    'T3w':  ('CELU3',  'wclamp', 'CELU3'),
}


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, clamp_from=CLAMP_FROM):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    act_arm, clamp, anch = ARMS[arm]
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(act_arm))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G4 phi-prime', ck)
    ck.update(act=act_arm, clamp=(clamp or 'ref'), lr=LR0, tau=GS.make_act(act_arm).param)

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(act_arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    base = None
    if clamp is not None:
        ET.loop(p, act, adam, gens, mnist, probe, 1, clamp_from - 1, clamp, None, rows, units, ck, cfrom=clamp_from)
        base = C.baselines(p[0])
        ck['base_cnorm_t20'] = float(base['c'].mean())
        ck['f32_rowmean_bound_t20'] = CH.EPS32 * float(p[0].detach().double().abs().mean(1).max())
        if controls:
            ck.update(ET._controls(act_arm, seed, C.snapshot(p, act, adam, gens), base, mnist, probe, ck, clamp_from))
        ET.loop(p, act, adam, gens, mnist, probe, clamp_from, tasks, clamp, base, rows, units, ck, cfrom=clamp_from)
        # N0: the realised width really is pinned to its t20 value
        cn = [r['cnorm'] for r in rows if r['task'] >= clamp_from]
        ck['n0_cnorm_rel'] = max(abs(x / ck['base_cnorm_t20'] - 1) for x in cn) if cn else 1.
        assert ck['n0_cnorm_rel'] <= 1e-6, ('N0 width not pinned', ck['n0_cnorm_rel'])
    else:
        ET.loop(p, act, adam, gens, mnist, probe, 1, tasks, None, None, rows, units, ck)
    T.finite_guard(rows, tag)

    assert ck.get('g2_cos_range', 0.) == 0., ('G2 cos', ck.get('g2_cos_range'))
    assert ck.get('g2_angle_rel', 1.) <= 1e-6, ('G2 angle cross-check', ck.get('g2_angle_rel'))
    assert ck.get('g2_triangle', -1.) <= 1e-8, ('G2 triangle', ck.get('g2_triangle'))
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck.get('g2_tot_ident'))
    if clamp is not None:
        for k in ('c3_cnorm_rel', 'c3_m_absdiff', 'clamp_tail_absdiff'):
            assert k in ck and ck[k] <= CH.limit(ck, k, 'f32_rowmean_bound_t20'), ('G3', k, ck.get(k))

    # ---- G1: clamped arms share t1..t20 with the unclamped anchor; unclamped arms match all of it
    ref = np.load(ROOT / 'results/gate_shape_0911' / f'{anch}_s{seed}_units.npz')
    g1_to = min(tasks, 120) if clamp is None else min(tasks, clamp_from - 1)
    w, n = ET._cmp(units, ref, 'ref', 'ref', 1, g1_to)
    ck.update(g1_anchor=f'gate_shape_0911/{anch}' + ('' if clamp is None else f' (t1-{g1_to} prefix)'),
              g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=None)
    assert w <= 1e-10, ('G1 failed', arm, seed, w)
    assert n >= 12 * g1_to, ('G1 count guard', n, g1_to)
    if controls and seed == 0 and arm == 'T3':
        p3 = H.init_params(seed, torch.device('cpu'))
        with torch.no_grad():
            p3[0].data[0, 0] += 1e-3
        ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
        g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
        u3 = {}
        ET.loop(p3, GS.make_act(act_arm), ad3, g3, mnist, probe, 1, 1, None, None, [], u3, {}, instrument=False)
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
                cfg=dict(act=act_arm, clamp=clamp, lr=LR0), late=ET.LATE, checks=ck,
                spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                elu_turn_sha256=T.sha(Path(ET.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                clamp_horizon_sha256=T.sha(Path(CH.__file__)), base_sha256=T.sha(Path(C.__file__)),
                host_sha256=T.sha(Path(H.__file__)), data_sha256=mnist.sha256,
                torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; CELU tau 0.3 and 3, with and without the width clamp'))
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
            subprocess.run([sys.executable, '-m', 'src.tau_clamp_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
