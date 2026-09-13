"""long_horizon_0910: reference trajectories to t400 (spec_long_horizon_0910)."""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import numpy as np, torch
from src import width_sink_clamp_0909 as C
H = C.H; ROOT = C.ROOT; OUT = ROOT / 'results/long_horizon_0910'; TASKS = 400
C.MEAS = [625]   # task ends only


def run(arm, seed, mutate=False):
    torch.set_num_threads(1); H.setup('cpu'); mnist = H.Mnist(torch.device('cpu'))
    act = C.make_act(arm)
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    p = H.init_params(seed, torch.device('cpu'))
    if mutate:
        with torch.no_grad(): p[0].data[0, 0] += 1e-3
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gens = [H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)]
    rows, units, ck = [], {}, {}; t0 = time.monotonic()
    # dead-unit counters need the current-perm gate on the probe: hook measure via a wrapper
    orig = C.measure
    def measure(p_, act_, probe_, perm, want_acc=False, mnist_=None, ck_=None, mut_decomp=False):
        r, u = orig(p_, act_, probe_, perm, want_acc, mnist_, ck_, mut_decomp)
        with torch.no_grad():
            z = probe_.px[:, perm] @ p_[0].detach().T + p_[1].detach()
            g = act_.dphi(z, 0) if isinstance(act_, H.AdaptiveSnake) else act_.dphi(z)
            r['dead_hard'] = int((g.max(0).values < 1e-6).sum()); r['dead_soft'] = int((g.max(0).values < .05).sum())
            r['sat'] = float((g < .05).float().mean())
        return r, u
    C.measure = measure
    C.loop(p, act, adam, gens, mnist, probe, 1, TASKS, 'ref', None, rows, units, ck)
    C.measure = orig
    eg = np.load(ROOT / 'results/elu_growth_0909' / f'{arm}_none_s{seed}_units.npz')
    worst = max(float(np.abs(units[f'ref_cnorm_i_t{t}'] - eg['cnorm_i'][t - 1]).max()) for t in range(1, 121))
    if mutate:
        return worst
    assert worst <= 1e-10, ('G1', worst)
    assert arm != 'LR' or max(r['dead_hard'] for r in rows) == 0, 'leaky cannot have hard-dead units'
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows: r.update(arm=arm, iv='none', seed=seed)
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    C.G.B.csvwrite(OUT / f'{arm}_none_s{seed}_rows.csv', [{k: r.get(k) for k in keys} for r in rows])
    np.savez_compressed(OUT / f'{arm}_none_s{seed}_units.npz', **units)
    (OUT / f'{arm}_none_s{seed}_provenance.json').write_text(json.dumps(dict(arm=arm, seed=seed, tasks=TASKS, g1_units_maxabs=worst,
        checks=ck, spec_sha256=C.sha(ROOT / 'specs/spec_long_horizon_0910.md'), code_sha256=C.sha(Path(__file__)),
        wall_seconds=time.monotonic() - t0), indent=1, default=str))
    print('FINISHED', arm, seed, 'G1', worst, round(time.monotonic() - t0), 's', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm'); ap.add_argument('--seed', type=int); ap.add_argument('--all', action='store_true'); ap.add_argument('--control', action='store_true')
    a = ap.parse_args()
    if a.control:
        C.loop.__globals__  # noqa
        w = run('LR', 0, mutate=True); print('G1 mutation control (init +1e-3): maxabs', w); assert w > 1e-4
    elif a.all:
        def job(j): subprocess.run([sys.executable, '-m', 'src.long_horizon_0910', '--arm', j[0], '--seed', str(j[1])], check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(job, [(arm, s) for arm in ['LR', 'ELU1', 'SNA'] for s in range(3)]): pass
    else:
        run(a.arm, a.seed)
