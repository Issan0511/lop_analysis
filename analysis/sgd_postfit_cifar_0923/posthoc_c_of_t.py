#!/usr/bin/env python3
"""POST HOC (0923 night, Issa: is c constant in time?).
net relative growth rho^2 + 2 rho c, and the absolute dN.  Median over 10 seeds, by task block."""
import numpy as np, os, pandas as pd
SP = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
rows = []
for arm in ("A", "SA_iid", "F", "S_hi"):
    for s in range(10):
        W = np.stack([np.load(f"{SP}/{arm}/snap/LR_std_seed{s}/t{t:02d}.npz")["W1"].astype(np.float64).ravel() for t in range(1, 51)])
        for k in range(1, 50):                       # D_t = W(t) - W(t-1), t = k+1
            D = W[k] - W[k - 1]; nW, nD = np.linalg.norm(W[k - 1]), np.linalg.norm(D)
            c = float(D @ W[k - 1] / (nW * nD)); rho = nD / nW
            rows.append(dict(arm=arm, seed=s, t=k + 1, nW=nW, nD=nD, rho=rho, c=c, bal=-rho / 2, ratio=c / (-rho / 2),
                             net_rel=rho ** 2 + 2 * rho * c, dN=float(W[k] @ W[k] - W[k - 1] @ W[k - 1])))
df = pd.DataFrame(rows)
df["block"] = pd.cut(df.t, [1, 5, 10, 20, 30, 40, 50], labels=["t2-5", "t6-10", "t11-20", "t21-30", "t31-40", "t41-50"])
pd.set_option("display.width", 220)
for arm in ("A", "SA_iid", "F", "S_hi"):
    d = df[df.arm == arm]
    g = d.groupby(["block", "seed"], observed=True)[["nW", "nD", "rho", "c", "bal", "ratio", "net_rel", "dN"]].mean().groupby("block", observed=True).median()
    print(f"\n== {arm}"); print(g.round(4).to_string())
    # per-seed slope of c over t11-50
    sl = [np.polyfit(x.t, x.c, 1)[0] * 40 for _, x in d[d.t >= 11].groupby("seed")]
    print(f"   c: change over t11-50 from a per-seed linear fit, median {np.median(sl):+.4f}, seeds positive {sum(v > 0 for v in sl)}/10")
df.to_csv("/home/issan/Projects/claude/wt/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923/posthoc_c_of_t.csv", index=False)
