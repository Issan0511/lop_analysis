#!/usr/bin/env python3
"""Post-hoc (not registered): per-unit state at every task end of one MLP run of
swish_battle_0917, recorded by wrapping the host's evaluate_rl (read-only, after the
host's own metrics are computed), so the training trajectory is untouched.  The rows the
host returns are compared with the stored per_task.csv as a reproduction check.

usage: posthoc_unit_state_mlp.py ARM SEED OUT.npz
(run on CPU with one thread, as the mlp box was; the reproduction check needs that)
"""
import csv, json, math, sys, time
from pathlib import Path
REPO = Path("/home/issan/Projects/claude/wt/swish_battle_0917")
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from src import swish_battle_0917 as SB
from src import pmnist_0905 as H
from src import pmnist_rlmnist_0906 as RL

arm, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
device = H.setup("cpu")
torch.set_num_threads(1)
data = SB.load_data("mlp", device)
rec = []


def gate(act, z, l):
    return act.dphi(z, l) if isinstance(act, H.AdaptiveSnake) else act.dphi(z)


def alpha(act, l, n):
    if isinstance(act, H.AdaptiveSnake):
        return act.alpha(l).double()
    p = getattr(act, "param", float("nan"))
    return torch.full((n,), float(p), dtype=torch.float64)


def eff_rank(a):
    s = torch.linalg.svdvals(a.double())
    p = s / s.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))


def unit_stats(params, x, y, act):
    W1, b1, W2, b2, W3, b3 = [p.detach() for p in params]
    ps = [p.detach().clone().requires_grad_(True) for p in params]
    z1, a1, z2, a2, logits = H.forward(ps, x, act)
    z1.retain_grad(); z2.retain_grad()
    loss = F.cross_entropy(logits, y)
    loss.backward()
    out = {}
    with torch.no_grad():
        for l, (z, a) in enumerate(((z1, a1), (z2, a2))):
            z, a, gz = z.double(), a.double(), z.grad.double() * len(y)   # per-sample dL/dz
            g = gate(act, z.float(), l).double()
            zb, zs = z.mean(0), z.std(0)
            ab, asd = a.mean(0), a.std(0)
            n = z.shape[1]
            al = alpha(act, l, n)
            out.update({
                f"zbar{l+1}": zb, f"zsd{l+1}": zs, f"u{l+1}": zb / zs.clamp_min(1e-30),
                f"alpha{l+1}": al, f"alphaW{l+1}": al * zs,
                f"gmean{l+1}": g.mean(0), f"gneg{l+1}": (g < 0).double().mean(0),
                f"gtiny{l+1}": (g.abs() < 1e-3).double().mean(0),
                f"abar{l+1}": ab, f"asd{l+1}": asd,
                # push: the direction plain GD moves each unit's mean preactivation
                f"push{l+1}": -gz.mean(0),
                f"erank_unc{l+1}": torch.tensor(eff_rank(a)),
                f"erank_cen{l+1}": torch.tensor(eff_rank(a - ab)),
                # share of the activation energy in the common (mean) vector
                f"mean_energy{l+1}": torch.tensor(float((ab ** 2).sum() / (a ** 2).mean(0).sum())),
            })
        # layer 2 input decomposition along the mean direction of layer-1 outputs
        a1d = a1.double(); mu = a1d.mean(0); mhat = mu / mu.norm()
        proj = (a1d - mu) @ mhat                       # fluctuation along mu-hat
        w = W2.double() @ mhat                         # each unit's weight on mu-hat
        zc = (a1d - mu) @ W2.double().T                # centered z2
        rho = (w[None, :] * proj[:, None]).var(0) / zc.var(0).clamp_min(1e-30)
        out.update({"mu1_norm": mu.norm(), "proj_sd": proj.std(), "c2": mu.norm() / proj.std(),
                    "w_mu": w, "rho_mu": rho, "b2_over_sd": b2.double() / z2.double().std(0),
                    "m2_over_sd": (W2.double() @ mu) / z2.double().std(0),
                    "loss": torch.tensor(float(loss))})
    return {k: v.cpu().numpy() for k, v in out.items()}


orig = RL.evaluate_rl


def spy(params, x, y, act):
    m = orig(params, x, y, act)
    rec.append(unit_stats(params, x, y, act))
    return m


RL.evaluate_rl = spy
t0 = time.time()
rows, div = SB.run_host("mlp", arm, seed, data, device)
ref = list(csv.DictReader(open(REPO / f"results/swish_battle_0917/mlp/{arm}/seed{seed}/per_task.csv")))
fmt = lambda v: f"{v:.10g}" if isinstance(v, float) else str(v)
bad = sum(1 for r, g in zip(ref, rows) for k in r if fmt(g.get(k)) != r[k])
np.savez_compressed(out, **{k: np.stack([r[k] for r in rec]) for k in rec[0]},
                    repro_mismatch=bad, n_rows=len(rows))
print(f"{arm} s{seed}: {len(rows)} tasks, repro mismatched cells {bad}, {time.time()-t0:.0f}s", flush=True)
