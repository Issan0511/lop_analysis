"""Checks for spec pmnist_adapt_0905 §4, run before the main launch.

S-ema-off : SNA(beta=0, c=0.6) == fixed Snake alpha=0.6, bit-for-bit
S-act     : per-unit-alpha phi' vs autograd (relative where |phi'|>1e-9, absolute <= 4eps, zeros in grid)
S-detach  : alpha carries no grad; a backward through phi leaves V untouched
S-clip    : alpha stays inside [lo, hi]; clip fraction reported
S-repro   : known arms, box A, seed 0, first 20 tasks == the 9/5 Adam runs (same code path)
"""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
import pmnist_0905 as H

REPO = Path(".")
OUT = REPO / "results" / "_checks_pmnist_adapt_0905"; OUT.mkdir(parents=True, exist_ok=True)
dev = H.setup("auto"); res = {}

def run(args, out):
    subprocess.run([sys.executable, "src/pmnist_0905.py", *args, "--out", str(out)],
                   check=True, capture_output=True)
    return pd.read_csv(out / "per_task.csv")

# ---- S-ema-off
a = run(["--stage", "main", "--optimizer", "adam", "--arms", "SNA", "--seeds", "0", "--lrs", "0.001",
         "--tasks", "5", "--beta", "0", "--c", "0.6"], OUT / "emaoff_sna")
b = run(["--stage", "main", "--optimizer", "adam", "--arms", "SN06", "--seeds", "0", "--lrs", "0.001",
         "--tasks", "5"], OUT / "emaoff_sn06")
common = [c for c in b.columns if c in a.columns and c not in ("arm",)]
res["S-ema-off"] = {"pass": bool(all(np.array_equal(a[c].values, b[c].values) for c in common)),
                    "cols_compared": len(common),
                    "max_abs_acc_diff": float(np.abs(a.acc.values - b.acc.values).max())}

# ---- S-act (per-unit alpha)
sna = H.AdaptiveSnake(0.6, 0.01, dev)
torch.manual_seed(0)
sna.V[0] = torch.rand(H.DIMS[1], device=dev) * 8 + 0.05        # spread of W across units
eps = float(np.finfo(np.float64).eps); ok = True; worst = 0.0
alpha = sna.alpha(0).double()
zs = torch.linspace(-12, 12, 20001, dtype=torch.float64, device=dev)
grid = [zs]
for k in range(-60, 61):                                       # zeros of 1+sin(2 a z), all units
    grid.append(((-math.pi / 2 + 2 * math.pi * k) / (2 * alpha)).flatten())
z = torch.cat(grid); z = z[z.abs() <= 12]
Z = z[:, None].expand(-1, H.DIMS[1]).clone().requires_grad_(True)
phi = Z + torch.sin(alpha * Z) ** 2 / alpha
(g,) = torch.autograd.grad(phi.sum(), Z)
ana = 1.0 + torch.sin(2 * alpha * Z.detach())
dev_ = (g - ana).abs(); worst = float(dev_.max())
mask = ana.abs() > 1e-9
rel = np.allclose(g[mask].cpu().numpy(), ana[mask].cpu().numpy(), atol=0.0)
res["S-act"] = {"pass": bool(rel and worst <= 4 * eps), "rel_allclose_atol0": bool(rel),
                "max_abs_dev_ulp": worst / eps, "n_points": int(z.numel()), "n_units": H.DIMS[1]}

# ---- S-detach
sna2 = H.AdaptiveSnake(0.6, 0.01, dev)
V_before = [v.clone() for v in sna2.V]
zz = torch.randn(16, H.DIMS[1], device=dev, requires_grad=True)
sna2.phi(zz, 0).sum().backward()
res["S-detach"] = {"pass": bool(all(not v.requires_grad for v in sna2.V)
                                 and all(torch.equal(x, y) for x, y in zip(V_before, sna2.V))
                                 and zz.grad is not None),
                   "V_requires_grad": [bool(v.requires_grad) for v in sna2.V]}

# ---- S-clip (on the 5-task SNA run with beta on)
c = run(["--stage", "main", "--optimizer", "adam", "--arms", "SNA", "--seeds", "0", "--lrs", "0.001",
         "--tasks", "10"], OUT / "clip_sna")
res["S-clip"] = {"pass": bool((c.alpha_min_l1 >= 0.05 - 1e-9).all() and (c.alpha_max_l1 <= 3.0 + 1e-9).all()
                              and (c.alpha_min_l2 >= 0.05 - 1e-9).all() and (c.alpha_max_l2 <= 3.0 + 1e-9).all()),
                 "alpha_med_l1_task1_to_10": [round(float(x), 3) for x in c.alpha_med_l1.values],
                 "clip_frac_max": float(max(c.alpha_clip_frac_l1.max(), c.alpha_clip_frac_l2.max())),
                 "two_alpha_W_med_l1_last": float(c.two_alpha_W_med_l1.iloc[-1])}

# ---- S-repro: box A, seed 0, 20 tasks, known arms vs the 9/5 runs
known = pd.concat([pd.read_csv(f) for f in
                   list((REPO / "results" / "_diag_adam_0905").glob("shard*/per_task.csv"))
                   + [REPO / "results" / "_diag_alpha_adam_0905" / "per_task.csv",
                      REPO / "results" / "_diag_alpha_adam_0905" / "low" / "per_task.csv"]])
known = known[(known.seed == 0) & (known.task <= 20)]
d = run(["--stage", "main", "--optimizer", "adam", "--arms", "R,LR,LIN,SN1,SN3,SN02", "--seeds", "0",
         "--lrs", "0.001", "--tasks", "20"], OUT / "repro_boxA")
rep = {}
for arm in ("R", "LR", "LIN", "SN1", "SN3", "SN02"):
    x = d[d.arm == arm].sort_values("task"); y = known[known.arm == arm].sort_values("task")
    cols = [k for k in ("acc", "mob_l1", "zbar_l1", "w_norm_l1") if k in x and k in y]
    rep[arm] = bool(len(x) == len(y) == 20 and all(np.array_equal(x[k].values, y[k].values) for k in cols))
res["S-repro"] = {"pass": bool(all(rep.values())), "per_arm": rep}

res["all_pass"] = bool(all(v["pass"] for k, v in res.items() if k.startswith("S-")))
(OUT / "checks.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
