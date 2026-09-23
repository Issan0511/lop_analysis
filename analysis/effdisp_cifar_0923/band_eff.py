#!/usr/bin/env python3
"""POST HOC (0923 night).
||D||^2 over input-PCA bands (top 0-10, mid1 10-100, mid2 100-439, low 439-1199, comp = orthogonal to the data).
Shares per band, Adam/SGD ratios per band, and the effective/plain ratio per band."""
import numpy as np, os, pandas as pd
ARCH = os.path.expanduser("~/Projects/obsidian-research-data"); SP = f"{ARCH}/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923"
B = f"{ARCH}/sgd_postfit_cifar_0923/basis_cache"
BANDS = (("top", 0, 10), ("mid1", 10, 100), ("mid2", 100, 439), ("low", 439, 1199))
rows = []
for s in range(10):
    z = np.load(f"{B}/basis_std_s{s}.npz"); Q = z["Q"][:, :1199].astype(np.float64); lam = z["lam"].astype(np.float64)
    for arm in ("A", "F", "S_hi", "SA_iid"):
        Wp = np.load(f"{SP}/{arm}/snap/LR_std_seed{s}/t40.npz")["W1"].astype(np.float64)
        acc = {}
        for t in range(41, 51):
            W = np.load(f"{SP}/{arm}/snap/LR_std_seed{s}/t{t:02d}.npz")["W1"].astype(np.float64)
            D = W - Wp; C = D @ Q
            plain_tot = float((D * D).sum()); inspan = (C * C).sum(0)
            for name, lo, hi in BANDS:
                acc.setdefault(("eff", name), []).append(float((lam[lo:hi] * inspan[lo:hi]).sum()))
                acc.setdefault(("plain", name), []).append(float(inspan[lo:hi].sum()))
            acc.setdefault(("plain", "comp"), []).append(plain_tot - float(inspan.sum()))
            acc.setdefault(("eff", "comp"), []).append(0.0)
            Wp = W
        for (kind, name), v in acc.items():
            rows.append(dict(arm=arm, seed=s, kind=kind, band=name, val=float(np.mean(v))))
d = pd.DataFrame(rows)
tot = d.groupby(["arm", "seed", "kind"])["val"].transform("sum"); d["share"] = d.val / tot
pd.set_option("display.width", 200)
sh = d.groupby(["arm", "kind", "band"])["share"].median().unstack("band")[["top", "mid1", "mid2", "low", "comp"]]
print("share of per-task displacement^2 by band (median over seeds):"); print(sh.round(3).to_string())
v = d.groupby(["arm", "kind", "band"])["val"].median().unstack("band")[["top", "mid1", "mid2", "low", "comp"]]
print("\nAdam / pure-SGD ratio of displacement^2 per band:"); print((v.loc["A"] / v.loc["SA_iid"]).round(1).to_string())
print("\nAdam / frozen-after-fit (F) ratio per band:"); print((v.loc["A"] / v.loc["F"]).round(2).to_string())
tot_eff = d[d.kind == "eff"].groupby(["arm", "seed"]).val.sum().groupby("arm").median()
print("\nsqrt(total effective displacement^2) per arm:", np.sqrt(tot_eff).round(1).to_dict())
