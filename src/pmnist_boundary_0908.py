"""Boundary-window probe for Permuted MNIST — separate the jump from the refit.

Registered question ([[負に押す力の正体_0902]] §5-1): there, a boundary record is
"jump + 1000 steps" mixed, so rho < 2/15 can only be *attributed* to "refit chases",
never separated.  Permuted MNIST separates them exactly, because the permutation
change is instantaneous and the setpoint has a closed form.  For hidden unit i of
layer 1, under permutation P of the 784 input pixels,

    zbar_i(P) = W1[i] . P(mu) + b_i ,     zbar*_i = E_P[zbar_i(P)] = (sum_j W1[i,j]) * mean(mu) + b_i

because averaging a permutation over all orderings replaces every pixel by the grand
mean.  So at each boundary we record, per unit:

    end    zbar under the OLD permutation, after the last step of task t
    jump   zbar under the NEW permutation, BEFORE any gradient step of task t+1
    dense  zbar under the new permutation after s = 1..n_dense steps
    star   the closed-form setpoint from the current weights

  strip = jump - end        (0902's flip term, here exact and instantaneous)
  refit = dense[-1] - jump  (what the task's own gradient puts back)
  disease = drift of star   (0902 §4: "zbar* itself moves below the wall")

The training loop is copied from the host's `run_one` (same streams, same stratified
draw, same order, same update rule), so a run here follows the same trajectory as the
main runs.  The host is imported and never edited.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmnist_0905 as H
from pmnist_lopcmp_0905 import Intervention          # frozen module, imported only

PROBE_N = 512


@torch.no_grad()
def _zbar(params, x, act):
    z1, _, z2, _, _ = H.forward(params, x, act)
    return (z1.mean(0).cpu().numpy().astype(np.float32),
            z2.mean(0).cpu().numpy().astype(np.float32))


@torch.no_grad()
def _star(params, mu_mean):
    """E_P[zbar] for layer 1; for layer 2 the permutation does not act, so we return
    the row sum of W2 (its drift is what moves layer 2's operating point)."""
    return ((params[0].sum(1) * mu_mean + params[1]).cpu().numpy().astype(np.float32),
            (params[2].sum(1) + params[3]).cpu().numpy().astype(np.float32))


def run(arm, seed, lr, n_tasks, first_dense, n_dense, mnist, device, optimizer, iv, c, beta):
    act = H.ARMS[arm]
    if act.kind == "adaptive_snake":
        act = H.AdaptiveSnake(c, beta, device)
    params = H.init_params(seed, device)
    # NOT `Intervention(iv)`: that is the dataclass constructor, which stores the raw
    # string as `kind` so `kind in ("l2","l2init")` is False and the penalty is silently
    # dropped.  The host uses the `parse` classmethod; so must we.
    ivo = Intervention.parse(iv) if iv != "none" else None
    p0 = [q.detach().clone() for q in params] if (ivo is not None and ivo.kind == "l2init") else None
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params],
            [0]) if optimizer == "adam" else None
    g_perm, g_data, g_batch = H.stream("perm", seed), H.stream("data", seed), H.stream("batch", seed)
    pidx = torch.randperm(mnist.test_x.shape[0], generator=H.stream("boundary_probe", seed))[:PROBE_N]
    px = mnist.test_x[pidx].to(device)
    mu_mean = float(mnist.train_x.mean())
    R = {k: [] for k in ("task", "end1", "end2", "jump1", "jump2", "star1", "star2",
                         "dense1", "dense2", "acc")}
    perm_prev = None

    for t in range(1, n_tasks + 1):
        dense = t > first_dense and perm_prev is not None
        if dense:                                        # --- state at the end of task t-1
            e1, e2 = _zbar(params, px[:, perm_prev], act)
        perm = torch.randperm(784, generator=g_perm).to(device)
        if dense:                                        # --- same weights, new permutation
            j1, j2 = _zbar(params, px[:, perm], act)
            s1, s2 = _star(params, mu_mean)
            R["task"].append(t); R["end1"].append(e1); R["end2"].append(e2)
            R["jump1"].append(j1); R["jump2"].append(j2); R["star1"].append(s1); R["star2"].append(s2)
            d1, d2 = [], []

        idx = H.stratified_draw(mnist, g_data).to(device)
        order = torch.randperm(H.TASK_EXAMPLES, generator=g_batch).to(device)
        xs = mnist.train_x[idx][:, perm][order]; ys = mnist.train_y[idx][order]

        for s in range(H.STEPS_PER_TASK):
            xb = xs[s * H.BATCH:(s + 1) * H.BATCH]; yb = ys[s * H.BATCH:(s + 1) * H.BATCH]
            out = H.forward(params, xb, act)
            loss = torch.nn.functional.cross_entropy(out[4], yb)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                if ivo is not None and ivo.kind in ("l2", "l2init"):
                    grads = [gr + 2.0 * ivo.lam * (q - (p0[i] if p0 is not None else 0.0))
                             for i, (q, gr) in enumerate(zip(params, grads))]
                if adam is None:
                    for q, gr in zip(params, grads): q -= lr * gr
                else:
                    m, v, tc = adam; tc[0] += 1
                    b1, b2, eps = 0.9, 0.999, 1e-8; c1, c2 = 1 - b1 ** tc[0], 1 - b2 ** tc[0]
                    for q, gr, mi, vi in zip(params, grads, m, v):
                        mi.mul_(b1).add_(gr, alpha=1 - b1); vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        q -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                if isinstance(act, H.AdaptiveSnake): act.update(out[0], out[2])
            if dense and s < n_dense:
                a, b_ = _zbar(params, px[:, perm], act); d1.append(a); d2.append(b_)

        if dense:
            R["dense1"].append(np.stack(d1)); R["dense2"].append(np.stack(d2))
            with torch.no_grad():
                R["acc"].append(float((H.forward(params, mnist.test_x[:, perm], act)[4].argmax(1)
                                       == mnist.test_y).float().mean()))
        perm_prev = perm
    return {k: (np.stack(v) if v and isinstance(v[0], np.ndarray) else np.asarray(v))
            for k, v in R.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lr", type=float, default=0.001); ap.add_argument("--tasks", type=int, default=120)
    ap.add_argument("--first-dense", type=int, default=100)
    ap.add_argument("--n-dense", type=int, default=300)
    ap.add_argument("--optimizer", default="adam"); ap.add_argument("--iv", default="none")
    ap.add_argument("--c", type=float, default=0.6); ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = H.setup(a.device); mnist = H.Mnist(dev)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    (od / "provenance.json").write_text(json.dumps(
        {"run_id": "pmnist_boundary_0908", "git_hash": H.git_hash(), "arm": a.arm, "iv": a.iv,
         "seeds": a.seeds, "lr": a.lr, "tasks": a.tasks, "first_dense": a.first_dense,
         "n_dense": a.n_dense, "optimizer": a.optimizer, "c": a.c, "beta": a.beta,
         "probe_n": PROBE_N, "dims": list(H.DIMS), "batch": H.BATCH,
         "steps_per_task": H.STEPS_PER_TASK, "data_sha256": mnist.sha256}, indent=1))
    for s in [int(x) for x in a.seeds.split(",")]:
        t0 = time.time(); rec = run(a.arm, s, a.lr, a.tasks, a.first_dense, a.n_dense,
                                    mnist, dev, a.optimizer, a.iv, a.c, a.beta)
        f = od / f"{a.arm}_{a.iv.replace(':', '-')}_s{s}.npz"
        np.savez_compressed(f, **rec)
        print(f"[{time.time()-t0:7.1f}s] {a.arm} iv={a.iv} seed={s} acc_last={rec['acc'][-1]:.4f} -> {f}", flush=True)


if __name__ == "__main__":
    main()
