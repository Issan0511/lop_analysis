"""Does the "reduce activation on wrong samples" force shrink the width?  (post-hoc, 2026-09-10)

Centred gradient on a row:  g~_i = sum_x phi'(z_i(x)) e_i(x) x~(x),  x~ = x - mean_j x.
Split by the sample's correctness at that step, and project each part onto the
task-start centred weight W~_i0 (unit vector) and its orthogonal complement:
    along   = <g~_part, W~_i0/||W~_i0||>      (>0: descent SHRINKS the existing pattern)
    orth    = ||g~_part - along * W~_i0/||..|| ||  (deposits a new direction)
Also the realised Adam update split the same way (cannot be split by sample).
Trajectory checked against elu_growth_0909 per-unit arrays at 1e-10.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
H = C.H; ROOT = C.ROOT
OUT = ROOT / 'results/width_shrink_posthoc_0910'
WINDOWS = [(21, 25), (96, 100)]; PHASES = [(1, 20), (21, 100), (101, 625)]


def run(arm, seed=0):
    torch.set_num_threads(1); H.setup('cpu'); mnist = H.Mnist(torch.device('cpu'))
    act = C.make_act(arm)
    p = H.init_params(seed, torch.device('cpu'))
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    # acc[win, phase, correct, (along_grad, orth_grad_sq, n_samples)] summed over units (mean over units at the end)
    acc = np.zeros((2, 3, 2, 3)); real = np.zeros((2, 3, 3))   # realised: (along, orth_sq, steps)
    ends = {}; t0 = time.monotonic()
    for task in range(1, 101):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd); order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]; ys = mnist.train_y[idx][order]
        wi = next((k for k, (lo, hi) in enumerate(WINDOWS) if lo <= task <= hi), None)
        if wi is not None:
            with torch.no_grad():
                W0 = p[0].detach().double(); Wt0 = W0 - W0.mean(1, keepdim=True); U0 = Wt0 / Wt0.norm(dim=1, keepdim=True)
        for step in range(625):
            xb = xs[step * 16:(step + 1) * 16]; yb = ys[step * 16:(step + 1) * 16]
            out = H.forward(p, xb, act); loss = torch.nn.functional.cross_entropy(out[4], yb)
            if wi is not None:
                gg = torch.autograd.grad(loss, list(p) + [out[0]]); gr, dz = gg[:6], gg[6]
                with torch.no_grad():
                    ph = next(k for k, (a, b) in enumerate(PHASES) if a <= step + 1 <= b)
                    d = dz.double()                                   # (16,100) = dL/dz1 (gate included)
                    xd = xb.double(); xt = xd - xd.mean(1, keepdim=True)   # centred inputs (16,784)
                    corr = (out[4].detach().argmax(1) == yb)
                    for c, m in ((1, corr), (0, ~corr)):
                        if m.any():
                            gt = d[m].T @ xt[m]                         # (100,784) centred gradient from this subset
                            along = (gt * U0).sum(1)                     # per unit
                            orth = gt - along[:, None] * U0
                            acc[wi, ph, c] += [float(along.mean()), float((orth ** 2).sum(1).mean()), float(m.sum())]
                    Wb = p[0].detach().double().clone()
            else:
                gr = torch.autograd.grad(loss, p)
            with torch.no_grad():
                m_, v_, tc = adam; tc[0] += 1; c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9); vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake): act.update(out[0], out[2])
                if wi is not None:
                    dW = p[0].detach().double() - Wb; dWt = dW - dW.mean(1, keepdim=True)
                    along = (dWt * U0).sum(1); orth = dWt - along[:, None] * U0
                    real[wi, ph] += [float(along.mean()), float((orth ** 2).sum(1).mean()), 1.]
        if wi is not None:
            with torch.no_grad():
                Wd = p[0].detach().double(); ends[task] = (Wd - Wd.mean(1, keepdim=True)).norm(dim=1).numpy().copy()
    eg = np.load(ROOT / 'results/elu_growth_0909' / f'{arm}_none_s{seed}_units.npz')
    worst = max(float(np.abs(ends[t] - eg['cnorm_i'][t - 1]).max()) for t in ends); assert worst <= 1e-10, worst
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f'{arm}_s{seed}_grad.npy', acc); np.save(OUT / f'{arm}_s{seed}_real.npy', real)
    (OUT / f'{arm}_s{seed}_provenance.json').write_text(json.dumps(dict(arm=arm, seed=seed, trajectory_maxabs=worst,
        code_sha256=C.sha(Path(__file__)), wall_seconds=time.monotonic() - t0), indent=1))
    PH = ['1-20', '21-100', '101-625']
    print(f'FINISHED {arm} maxabs {worst} {time.monotonic() - t0:.0f}s')
    print('gradient on the CENTRED weights, per task (mean over units), split by sample correctness')
    print(f"{'win':8s} {'phase':8s} | {'along W~0: wrong':>17s} {'correct':>9s} | {'orth^2: wrong':>14s} {'correct':>9s} | {'realised along':>15s} {'realised orth^2':>16s}")
    for w, wn in enumerate(['t21-25', 't96-100']):
        for ph in range(3):
            n = 5.
            print(f"{wn:8s} {PH[ph]:8s} | {acc[w, ph, 0, 0] / n:+17.4f} {acc[w, ph, 1, 0] / n:+9.4f} | {acc[w, ph, 0, 1] / n:14.4f} {acc[w, ph, 1, 1] / n:9.4f} | {real[w, ph, 0] / n:+15.4f} {real[w, ph, 1] / n:16.4f}")
    print('  along > 0 for the GRADIENT means descent shrinks the existing pattern; realised along < 0 means it did shrink')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='LR'); a = ap.parse_args(); run(a.arm)
