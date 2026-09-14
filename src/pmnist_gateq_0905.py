"""Gate *quality* probe for SNA (and a fixed-alpha reference), box A.

`mobility` in per_task.csv is the median over units of E_test[phi'(z_i)].  A median
of 0.7 says nothing about (a) the low tail of units, (b) layer 2, or (c) how phi'
is *distributed across inputs* for one unit -- a Snake unit at mean gain 0.7 can be
"half the inputs at 1.4, half near 0", which is not the same gate as leaky at 0.7.
Per task this records, per layer:
  mob_p10 / mob_min          low tail of per-unit E[phi']
  frac_units_mob_lt_0p3      units whose mean gain is below 0.3
  offfrac_med / offfrac_p90  per unit, fraction of test inputs with phi' < 0.25 ("off"), median/p90 over units
  gain_cv_med                per unit, sd_x[phi']/E_x[phi'] (how oscillatory the gate is), median over units
  gsign_med                  per unit, |E_x[phi' * (dL/da)]| / E_x[|phi' * (dL/da)|] -- coherence of the
                             backpropagated signal through the gate on the test set (1 = all inputs agree)
Runs: SNA(c=0.6, beta=0.01) and SN02 and LR, seed 0, Adam lr=0.001, 200 tasks.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmnist_0905 import (ARMS, AdaptiveSnake, Mnist, TASK_EXAMPLES, BATCH, STEPS_PER_TASK, REPO,
                         init_params, forward, stream, stratified_draw, setup)


def probe(params, mnist, perm, act):
    x = mnist.test_x[:, perm]; y = mnist.test_y
    z1, a1, z2, a2, logits = forward(params, x, act)
    # backprop signal at the activation outputs, per sample (mean-reduced CE scaled back by N)
    a1r = a1.detach().requires_grad_(True); a2r = a2.detach().requires_grad_(True)
    W2, b2, W3, b3 = params[2], params[3], params[4], params[5]
    z2r = a1r @ W2.T + b2
    a2r2 = (act.phi(z2r, 1) if isinstance(act, AdaptiveSnake) else act.phi(z2r))
    lo = a2r2 @ W3.T + b3
    loss = torch.nn.functional.cross_entropy(lo, y, reduction="sum")
    g1, = torch.autograd.grad(loss, a1r, retain_graph=True)          # dL/da1 per sample
    loss2 = torch.nn.functional.cross_entropy(a2r @ W3.T + b3, y, reduction="sum")
    g2, = torch.autograd.grad(loss2, a2r)
    out = {}
    with torch.no_grad():
        for li, (z, g) in enumerate(((z1, g1), (z2, g2))):
            d = act.dphi(z, li) if isinstance(act, AdaptiveSnake) else act.dphi(z)   # [N, h]
            mob = d.mean(0)
            off = (d < 0.25).float().mean(0)
            cv = d.std(0) / mob.clamp_min(1e-8)
            s = d * g
            coh = s.mean(0).abs() / s.abs().mean(0).clamp_min(1e-12)
            t = f"l{li+1}"
            out.update({f"mob_med_{t}": float(mob.median()), f"mob_p10_{t}": float(mob.quantile(0.10)),
                        f"mob_min_{t}": float(mob.min()), f"frac_units_mob_lt_0p3_{t}": float((mob < 0.3).float().mean()),
                        f"offfrac_med_{t}": float(off.median()), f"offfrac_p90_{t}": float(off.quantile(0.90)),
                        f"gain_cv_med_{t}": float(cv.median()), f"gsign_med_{t}": float(coh.median()),
                        f"gsign_p10_{t}": float(coh.quantile(0.10))})
    out["acc"] = float((logits.argmax(1) == y).float().mean())
    return out


def run(arm, seed, mnist, device, lr=0.001, n_tasks=200):
    act = ARMS[arm]
    if act.kind == "adaptive_snake": act = AdaptiveSnake(0.6, 0.01, device)
    p = init_params(seed, device)
    gp, gd, gb = stream("perm", seed), stream("data", seed), stream("batch", seed)
    m = [torch.zeros_like(q) for q in p]; v = [torch.zeros_like(q) for q in p]; tc = 0
    rows = []
    for t in range(1, n_tasks + 1):
        perm = torch.randperm(784, generator=gp).to(device)
        idx = stratified_draw(mnist, gd).to(device); order = torch.randperm(TASK_EXAMPLES, generator=gb).to(device)
        xs, ys = mnist.train_x[idx][:, perm][order], mnist.train_y[idx][order]
        for s in range(STEPS_PER_TASK):
            out = forward(p, xs[s*BATCH:(s+1)*BATCH], act)
            loss = torch.nn.functional.cross_entropy(out[4], ys[s*BATCH:(s+1)*BATCH])
            g = torch.autograd.grad(loss, p)
            with torch.no_grad():
                tc += 1; b1, b2, eps = 0.9, 0.999, 1e-8; c1, c2 = 1-b1**tc, 1-b2**tc
                for q, gr, mi, vi in zip(p, g, m, v):
                    mi.mul_(b1).add_(gr, alpha=1-b1); vi.mul_(b2).addcmul_(gr, gr, value=1-b2)
                    q -= lr * (mi/c1) / ((vi/c2).sqrt() + eps)
                if isinstance(act, AdaptiveSnake): act.update(out[0], out[2])
        rows.append({"arm": arm, "seed": seed, "task": t, **probe(p, mnist, perm, act)})
    return rows


def main():
    device = setup("auto"); mnist = Mnist(device)
    out = Path(REPO / "results" / "_diag_gateq_0905"); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    allrows = []
    for arm in ("SNA", "SN02", "LR", "R"):
        allrows += run(arm, 0, mnist, device); print("done", arm, flush=True)
        pd.DataFrame(allrows).to_csv(out / "gateq.csv", index=False)
    print("wrote", out / "gateq.csv")


if __name__ == "__main__":
    main()
