"""Why does the loss want the row mean DOWN?  (post-hoc, unregistered, 2026-09-10)

dL/dm_i = sum_x phi'(z_i(x)) e_i(x) S(x),   e_i(x) = dL/da_i(x)  (activation error)

e_i(x) > 0 means "on sample x the loss would fall if unit i were LESS active".
We record e_i(x) per (unit, sample) inside the same two task windows as
growth_engine_posthoc_0910 and split it by
    side      z_i(x) > 0  vs  <= 0
    ink       S(x) = total pixel sum, quartiles of the training distribution
    correct   the network's argmax == y at that step
    within    updates 1-20 / 21-100 / 101-625
and per unit by its occupancy at the switch.  Trajectory must stay on the
committed one (checked against elu_growth_0909 per-unit arrays at 1e-10).
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
H = C.H; ROOT = C.ROOT
OUT = ROOT / 'results/why_down_posthoc_0910'
WINDOWS = [(21, 25), (96, 100)]
PHASES = [(1, 20), (21, 100), (101, 625)]


def run(arm, seed=0):
    torch.set_num_threads(1); H.setup('cpu'); mnist = H.Mnist(torch.device('cpu'))
    act = C.make_act(arm)
    S_all = mnist.train_x.double().sum(1); qs = torch.quantile(S_all, torch.tensor([.25, .5, .75], dtype=torch.float64))
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]; px = mnist.test_x[pidx]
    p = H.init_params(seed, torch.device('cpu'))
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    # accumulators: [window][phase][side][ink q][correct] -> sum e, sum e*S, count
    W = len(WINDOWS); acc = np.zeros((W, 3, 2, 4, 2, 3))
    unit = {}   # (window, unit) -> dict of sums by side/phase
    ends = {}; t0 = time.monotonic()
    for task in range(1, 101):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd); order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]; ys = mnist.train_y[idx][order]
        wi = next((k for k, (lo, hi) in enumerate(WINDOWS) if lo <= task <= hi), None)
        if wi is not None:
            with torch.no_grad():
                z0 = px[:, perm].double() @ p[0].detach().double().T + p[1].double()
                pos0 = (z0 > 0).double().mean(0).numpy()
            usum = np.zeros((100, 2, 3, 3))   # unit, side, phase, (sum e, sum eS, n)
        for step in range(625):
            xb = xs[step * 16:(step + 1) * 16]; yb = ys[step * 16:(step + 1) * 16]
            out = H.forward(p, xb, act); loss = torch.nn.functional.cross_entropy(out[4], yb)
            if wi is not None:
                gg = torch.autograd.grad(loss, list(p) + [out[1]]); gr, ea = gg[:6], gg[6]
                with torch.no_grad():
                    e = ea.double(); z1 = out[0].detach().double(); Sx = xb.double().sum(1)
                    side = (z1 > 0).long()                                   # (16,100)
                    inkq = torch.bucketize(Sx, qs).long()                    # (16,) 0..3
                    corr = (out[4].detach().argmax(1) == yb).long()          # (16,)
                    ph = next(k for k, (a, b) in enumerate(PHASES) if a <= step + 1 <= b)
                    eS = e * Sx[:, None]
                    for s_ in (0, 1):
                        m = side == s_
                        for q in range(4):
                            mq = m & (inkq == q)[:, None]
                            for c in (0, 1):
                                mc = mq & (corr == c)[:, None]
                                if mc.any():
                                    acc[wi, ph, s_, q, c] += [float(e[mc].sum()), float(eS[mc].sum()), float(mc.sum())]
                        usum[:, s_, ph, 0] += (e * m).sum(0).numpy(); usum[:, s_, ph, 1] += (eS * m).sum(0).numpy(); usum[:, s_, ph, 2] += m.sum(0).numpy()
            else:
                gr = torch.autograd.grad(loss, p)
            with torch.no_grad():
                m_, v_, tc = adam; tc[0] += 1; c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9); vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake): act.update(out[0], out[2])
        if wi is not None:
            for i in range(100):
                unit[(task, i)] = dict(task=task, unit=i, pos0=float(pos0[i]), **{f'{k}_{s}_{ph}': float(usum[i, s, ph, j])
                                       for s in (0, 1) for ph in range(3) for j, k in enumerate(['e', 'eS', 'n'])})
            with torch.no_grad():
                Wd = p[0].detach().double(); ends[task] = (Wd - Wd.mean(1, keepdim=True)).norm(dim=1).numpy().copy()
            print(arm, 'task', task, round(time.monotonic() - t0, 1), flush=True)
    eg = np.load(ROOT / 'results/elu_growth_0909' / f'{arm}_none_s{seed}_units.npz')
    worst = max(float(np.abs(ends[t] - eg['cnorm_i'][t - 1]).max()) for t in ends)
    assert worst <= 1e-10, worst
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f'{arm}_s{seed}_cells.npy', acc)
    C.G.B.csvwrite(OUT / f'{arm}_s{seed}_units.csv', list(unit.values()))
    (OUT / f'{arm}_s{seed}_provenance.json').write_text(json.dumps(dict(arm=arm, seed=seed, windows=WINDOWS, phases=PHASES,
        ink_quartiles=qs.tolist(), trajectory_maxabs=worst, code_sha256=C.sha(Path(__file__)), wall_seconds=time.monotonic() - t0), indent=1))
    print('FINISHED', arm, worst, round(time.monotonic() - t0, 1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='LR'); a = ap.parse_args(); run(a.arm)
