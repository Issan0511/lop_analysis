"""budget_ladder_0912: after the update deficit is filled, how much decline is left?
(spec_budget_ladder_0912)

Every task from t1 gets the SAME fixed budget (625 / 1250 / 2500 / 5000 updates), so an
early window and a late window can be compared at equal compute.  D(B) = acc(t21-30) -
acc(t101-120) at budget B is the decline compute cannot remove at that budget.

Training loop and instrumentation are turn_budget_0912's (imported).  Only the per-task
budget policy is replaced -- at runtime, in this module's namespace; the committed file
is not edited, and the replacement's source is recorded in the provenance.
"""
from pathlib import Path
import argparse, concurrent.futures, inspect, json, subprocess, sys, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import turn_budget_0912 as TB

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/budget_ladder_0912'
SMOKE = ROOT / 'results/_smoke_budget_ladder_0912'
ACT = 'LR'
TASKS = 120
LR0 = .001
TIME_CAP = 7200.
SPEC = ROOT / 'specs/spec_budget_ladder_0912.md'
ARMS = {'B1': 625, 'B2': 1250, 'B4': 2500, 'B8': 5000}


def fixed_budget(policy, cn, cn20, theta_star):
    """Replacement for turn_budget_0912.n_steps_for: the policy IS the budget."""
    return int(policy)


TB.n_steps_for = fixed_budget          # runtime override, recorded below


def run(arm, seed, tasks=TASKS, out=OUT, controls=True):
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    budget = ARMS[arm]
    mnist = H.Mnist(torch.device('cpu'))
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    t0 = time.monotonic()
    rows, units, ck = [], {}, {}
    ck.update(GS.check_dphi(ACT))
    assert ck['g2_dphi'] < 1e-6 and ck['g2_dphi_mutctl'] > 1e-3, ('G4 phi-prime', ck)
    ck.update(act=ACT, budget=budget, lr=LR0, policy_override=inspect.getsource(fixed_budget))

    p = H.init_params(seed, torch.device('cpu'))
    act = GS.make_act(ACT)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    rgen = H.stream('reshuffle', seed)
    st0 = [g.get_state().clone() for g in gens]
    TB.loop(p, act, adam, gens, rgen, mnist, probe, 1, tasks, budget, rows, units, ck)
    T.finite_guard(rows, tag)

    g2 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    ck['g4_streams_intact'] = all(torch.equal(a, b.get_state()) for a, b in zip(st0, g2))
    assert ck['g4_streams_intact']
    if budget == TB.BASE_STEPS:
        ck['g4_rgen_untouched'] = bool(torch.equal(H.stream('reshuffle', seed).get_state(), rgen.get_state()))
        assert ck['g4_rgen_untouched']
    ck['u0_steps_exact'] = all(r['n_steps'] == budget for r in rows)
    assert ck['u0_steps_exact'], ('budget not applied', sorted({r['n_steps'] for r in rows}))
    assert ck.get('g2_cos_range', 0.) == 0., ('G2 cos', ck.get('g2_cos_range'))
    assert ck.get('g2_angle_rel', 1.) <= 1e-6, ('G2 angle', ck.get('g2_angle_rel'))
    assert ck.get('g2_triangle', -1.) <= 1e-8, ('G2 triangle', ck.get('g2_triangle'))
    assert ck.get('g2_tot_ident', 0.) <= 1e-10, ('G2 telescoping', ck.get('g2_tot_ident'))
    ck['cost'] = budget / TB.BASE_STEPS

    # ---- G1: only the 625 arm has a committed anchor
    if budget == TB.BASE_STEPS:
        eg = np.load(ROOT / 'results/elu_growth_0909' / f'LR_none_s{seed}_units.npz')
        w, n = 0., 0
        for t in range(1, min(tasks, 120) + 1):
            for k in ('zbar_i', 'sd_i', 'cnorm_i'):
                key = f'ref_{k}_t{t}'
                if key in units:
                    w = max(w, float(np.abs(units[key] - eg[k][t - 1]).max())); n += 1
        ck.update(g1_anchor='elu_growth_0909/LR', g1_units_maxabs=w, g1_units_compared=n, g1_units_expected=3 * min(tasks, 120))
        assert w <= 1e-10, ('G1 failed', arm, seed, w)
        assert n == 3 * min(tasks, 120), ('G1 count guard', n)
        if controls and seed == 0:
            p3 = H.init_params(seed, torch.device('cpu'))
            with torch.no_grad():
                p3[0].data[0, 0] += 1e-3
            ad3 = ([torch.zeros_like(q) for q in p3], [torch.zeros_like(q) for q in p3], [0])
            g3 = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
            u3 = {}
            TB.loop(p3, GS.make_act(ACT), ad3, g3, H.stream('reshuffle', seed), mnist, probe, 1, 1,
                    budget, [], u3, {}, instrument=False)
            ck['ctl_g1_init'] = max(float(np.abs(u3[f'ref_{k}_t1'] - eg[k][0]).max()) for k in ('zbar_i', 'sd_i', 'cnorm_i'))
            assert ck['ctl_g1_init'] > 1e-8, ('vacuous G1 control', ck['ctl_g1_init'])
    else:
        ck.update(g1_anchor='none (budget != 625)', g1_units_maxabs=None, g1_units_compared=0, g1_units_expected=0)

    for r in rows:
        r.update(arm=arm, seed=seed)
    T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    wall = time.monotonic() - t0
    T.dump(out / f'{tag}_provenance.json',
           dict(arm=arm, seed=seed, tasks=tasks, cfg=dict(act=ACT, budget=budget, lr=LR0), late=TB.LATE,
                checks=ck, spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)),
                turn_budget_sha256=T.sha(Path(TB.__file__)), gate_shape_sha256=T.sha(Path(GS.__file__)),
                base_sha256=T.sha(Path(C.__file__)), host_sha256=T.sha(Path(H.__file__)),
                data_sha256=mnist.sha256, torch_version=torch.__version__, wall_seconds=wall,
                scope='CPU; fixed per-task update budget from t1, same 10000-example pool reshuffled'))
    print('FINISHED', tag, 'G1', ck['g1_units_maxabs'], f"cost {ck['cost']:.2f}", round(wall, 1), 's', flush=True)
    assert wall < TIME_CAP, ('runtime cap', wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        TB.LATE = (1, 4)
        for arm in ARMS:
            run(arm, 0, tasks=4, out=SMOKE, controls=True)
        return
    if a.all:
        def job(j):
            subprocess.run([sys.executable, '-m', 'src.budget_ladder_0912', '--arm', j[0], '--seed', str(j[1])], check=True)
        # biggest budgets first so they overlap with the small ones
        order = [(arm, s) for arm in ('B8', 'B4', 'B2', 'B1') for s in range(3)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for _ in pool.map(job, order):
                pass
        return
    run(a.arm, a.seed)


if __name__ == '__main__':
    main()
