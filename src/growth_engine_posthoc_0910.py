"""What pushes the first layer?  Post-hoc, unregistered diagnostic (2026-09-10).

Replays the reference trajectory (init -> t100, same streams as elu_growth_0909)
and, inside two windows of tasks, logs at every step:

  raw gradient g on W1 and the realised Adam step dW, each split into
      row-mean part  (brightness gain)   g_row, dm
      centred part   (pattern)           g~,   dW~
  energy fraction landing in the row-mean direction, Adam vs gradient
  m-vector budget:  d mbar,  d var(m) = 2 cov(m, dm) + var(dm)
  W~ budget:        d||W~||^2 = 2<W~, dW~> + ||dW~||^2   (alignment + deposition)
  brightness gradient split by the input's side of the kink (z>0 vs z<=0)
and per unit per task: total dm_i vs ||W~_i||, fossil response and start position.

The instrumentation (extra autograd output on z1) must leave the trajectory on
the committed one: checked against elu_growth_0909 per-unit arrays at 1e-10.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np
import torch
from src import width_sink_clamp_0909 as C

H = C.H
ROOT = C.ROOT
OUT = ROOT / 'results/growth_engine_posthoc_0910'
WINDOWS = [(21, 25), (96, 100)]


def run(arm, seed=0):
    torch.set_num_threads(1); H.setup('cpu')
    mnist = H.Mnist(torch.device('cpu'))
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:512]
    px = mnist.test_x[pidx]
    p = H.init_params(seed, torch.device('cpu'))
    act = C.make_act(arm)
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    steprows, unitrows, ends = [], [], {}
    t0 = time.monotonic()
    for task in range(1, 101):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        logged = any(lo <= task <= hi for lo, hi in WINDOWS)
        if logged:
            with torch.no_grad():
                W0 = p[0].detach().double().clone(); m0 = W0.mean(1); Wt0 = W0 - m0[:, None]
                xpd = px[:, perm].double()
                fossil = (xpd @ Wt0.T).var(0, unbiased=False)
                z0 = xpd @ W0.T + p[1].double()
            dm_acc = torch.zeros(100, dtype=torch.float64)
            dWt_acc = torch.zeros(100, 784, dtype=torch.float64)
            gpos_acc = torch.zeros(100, dtype=torch.float64); gneg_acc = torch.zeros(100, dtype=torch.float64)
        for step in range(625):
            xb = xs[step * 16:(step + 1) * 16]; yb = ys[step * 16:(step + 1) * 16]
            out = H.forward(p, xb, act)
            loss = torch.nn.functional.cross_entropy(out[4], yb)
            if logged:
                gg = torch.autograd.grad(loss, list(p) + [out[0]])
                gr, dz = gg[:6], gg[6]
                Wb = p[0].detach().double().clone()
            else:
                gr = torch.autograd.grad(loss, p)
            with torch.no_grad():
                m_, v_, tc = adam; tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
                if logged:
                    g = gr[0].double(); Wa = p[0].detach().double()
                    dW = Wa - Wb
                    mb = Wb.mean(1); Wtb = Wb - mb[:, None]
                    dm = dW.mean(1); dWt = dW - dm[:, None]
                    grow = g.mean(1); gt = g - grow[:, None]
                    # brightness gradient by side of the kink:  g_row,i = sum_x dz_i(x) S(x) / 784
                    S = xb.double().sum(1); z1 = out[0].detach().double(); d = dz.double()
                    pos = (z1 > 0).double()
                    grow_pos = ((d * pos) * S[:, None]).sum(0) / 784
                    grow_neg = ((d * (1 - pos)) * S[:, None]).sum(0) / 784
                    ident = float((grow_pos + grow_neg - grow).abs().max())
                    g2 = (g ** 2).sum(1); ok = g2 > 0          # fully-underflowed units give 0/0
                    fr_grad = float((784 * grow[ok] ** 2 / g2[ok]).mean()) if ok.any() else float('nan')
                    steprows.append(dict(
                        arm=arm, task=task, step=step + 1,
                        dmbar=float(dm.mean()),
                        cov2=float(2 * ((mb - mb.mean()) * (dm - dm.mean())).mean()),
                        vdm=float(dm.var(unbiased=False)),
                        align=float((2 * (Wtb * dWt).sum(1)).mean()),
                        dep=float((dWt ** 2).sum(1).mean()),
                        fr_adam=float((784 * dm ** 2 / (dW ** 2).sum(1)).mean()),
                        fr_grad=fr_grad, n_zero_grad_units=int((~ok).sum()),
                        align0=float((2 * (Wt0 * dWt).sum(1)).mean()),
                        sign_agree=float((torch.sign(dm) * torch.sign(-grow)).mean()),
                        grow_mean=float(grow.mean()), grow_pos=float(grow_pos.mean()),
                        grow_neg=float(grow_neg.mean()), side_identity=ident,
                        pos_frac=float(pos.mean()),
                        adam_rowstep=float(dm.abs().mean()), grad_rowstep=float((.001 * grow).abs().mean()),
                        adam_ctrstep=float(dWt.norm(dim=1).mean()), grad_ctrstep=float((.001 * gt).norm(dim=1).mean()),
                        loss=float(loss)))
                    dm_acc += dm; dWt_acc += dWt; gpos_acc += grow_pos; gneg_acc += grow_neg
        if logged:
            with torch.no_grad():
                Wd = p[0].detach().double(); Wt = Wd - Wd.mean(1, keepdim=True)
                ends[task] = Wt.norm(dim=1).numpy().copy()
                dep_task = (dWt_acc ** 2).sum(1); align_task = 2 * (Wt0 * dWt_acc).sum(1)
                dnorm2 = Wt.norm(dim=1) ** 2 - Wt0.norm(dim=1) ** 2
                split_identity = float((align_task + dep_task - dnorm2).abs().max())
                cos_task = (Wt0 * dWt_acc).sum(1) / (Wt0.norm(dim=1) * dWt_acc.norm(dim=1))
            for i in range(100):
                unitrows.append(dict(arm=arm, task=task, unit=i, m0=float(m0[i]), dm=float(dm_acc[i]),
                                     cnorm0=float(Wt0[i].norm()), fossil=float(fossil[i]),
                                     zbar0=float(z0[:, i].mean()), pos0=float((z0[:, i] > 0).double().mean()),
                                     dep_task=float(dep_task[i]), align_task=float(align_task[i]),
                                     dnorm2=float(dnorm2[i]), cos_task=float(cos_task[i]),
                                     gpos=float(gpos_acc[i]), gneg=float(gneg_acc[i]),
                                     split_identity=split_identity))
            print(arm, 'logged task', task, round(time.monotonic() - t0, 1), 's', flush=True)
    # non-invasiveness: the logged windows must still be on the committed trajectory
    eg = np.load(ROOT / 'results/elu_growth_0909' / f'{arm}_none_s{seed}_units.npz')
    worst = max(float(np.abs(ends[t] - eg['cnorm_i'][t - 1]).max()) for t in ends)
    assert worst <= 1e-10, ('instrumentation moved the trajectory', worst)
    OUT.mkdir(parents=True, exist_ok=True)
    C.G.B.csvwrite(OUT / f'{arm}_s{seed}_steps.csv', steprows)
    C.G.B.csvwrite(OUT / f'{arm}_s{seed}_units.csv', unitrows)
    (OUT / f'{arm}_s{seed}_provenance.json').write_text(json.dumps(dict(
        arm=arm, seed=seed, windows=WINDOWS, trajectory_maxabs_vs_elu_growth=worst,
        code_sha256=C.sha(Path(__file__)), wall_seconds=time.monotonic() - t0,
        scope='post-hoc unregistered diagnostic on the reference trajectory'), indent=1))
    print('FINISHED', arm, 'trajectory maxabs', worst, round(time.monotonic() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='LR'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); run(a.arm, a.seed)
