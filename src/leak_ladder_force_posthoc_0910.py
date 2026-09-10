"""Leak ladder for the force curve (post-hoc, unregistered, 2026-09-10).

Question from Leak距離補償仮説_0908: does the per-unit equilibrium occupancy scale
with the leak a (residual-weighted balance, §2.3) or with a^2 (self-moment
balance, §2.2)?  Reference trajectories only (no clamp), init -> t100, same
streams as every other run, per-unit m_i / zbar_i / sd_i at every task end.
LR (a=0.1) already exists in width_sink_clamp_0909; here a = 0.3, 0.01, 0 (ReLU).
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
H = C.H; ROOT = C.ROOT
OUT = ROOT / 'results/leak_ladder_force_posthoc_0910'
ARMS = {'LR03': ('leaky', .3), 'LR001': ('leaky', .01), 'R': ('relu', 0.)}


def run(arm, seed):
    torch.set_num_threads(1); H.setup('cpu'); mnist = H.Mnist(torch.device('cpu'))
    kind, a = ARMS[arm]
    act = H.Activation('R', 'relu') if kind == 'relu' else H.Activation('LRx', 'leaky', a)
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    p = H.init_params(seed, torch.device('cpu'))
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    rows, units, ck = [], {}, {}
    t0 = time.monotonic()
    C.loop(p, act, adam, gens, mnist, probe, 1, 100, 'ref', None, rows, units, ck)
    # non-invasiveness / reproduction: cnorm_i at the 11 checkpoints of gate_scale_invariance_0909
    ref = np.load(ROOT / f'results/gate_scale_invariance_0909/{arm}_s{seed}_units.npz')
    worst = max(float(np.abs(units[f'ref_cnorm_i_t{int(t)}'] - ref['cnorm_i'][i]).max())
                for i, t in enumerate(ref['tasks']) if f'ref_cnorm_i_t{int(t)}' in units)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows: r.update(arm=arm, seed=seed, leak=a)
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    C.G.B.csvwrite(OUT / f'{arm}_s{seed}_rows.csv', [{k: r.get(k) for k in keys} for r in rows])
    np.savez_compressed(OUT / f'{arm}_s{seed}_units.npz', **units)
    (OUT / f'{arm}_s{seed}_provenance.json').write_text(json.dumps(dict(
        arm=arm, seed=seed, leak=a, gate_scale_cnorm_maxabs=worst, checks=ck,
        code_sha256=C.sha(Path(__file__)), wall_seconds=time.monotonic() - t0,
        scope='reference trajectory only; post-hoc leak ladder for the force curve'), indent=1, default=str))
    print('FINISHED', arm, seed, 'vs gate_scale cnorm maxabs', worst, round(time.monotonic() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm'); ap.add_argument('--seed', type=int); ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    if a.all:
        def job(j): subprocess.run([sys.executable, '-m', 'src.leak_ladder_force_posthoc_0910', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ARMS for s in range(3)]): pass
    else:
        run(a.arm, a.seed)
