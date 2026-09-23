#!/usr/bin/env python3
"""POST HOC (0923 night, Issa: redo the slope with the effective, width-making component of the input).
V = sum_ij lambda_j C_ij^2 with C = W1 Q (seed's input PCA, 1199 PCs); per task D = W1(t) - W1(t-1):
dV = |D|_S^2 + 2 <W, D>_S, rho_S = |D|_S/|W|_S, c_S = <W,D>_S/(|W|_S |D|_S), balance ratio c_S/(-rho_S/2).
Also per band (top 0-10, mid1 10-100, mid2 100-439, low 439-1199; unweighted within band) and the unweighted total."""
import numpy as np, os, pandas as pd
SP = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
B = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/basis_cache")
BANDS = (("top", 0, 10), ("mid1", 10, 100), ("mid2", 100, 439), ("low", 439, 1199))
rows = []
for arm in ("A", "SA_iid", "F", "S_hi"):
    for s in range(10):
        z = np.load(f"{B}/basis_std_s{s}.npz"); Q = z["Q"][:, :1199].astype(np.float64); lam = z["lam"].astype(np.float64)
        Wp = np.load(f"{SP}/{arm}/snap/LR_std_seed{s}/t01.npz")["W1"].astype(np.float64); Cp = Wp @ Q
        for t in range(2, 51):
            W = np.load(f"{SP}/{arm}/snap/LR_std_seed{s}/t{t:02d}.npz")["W1"].astype(np.float64); C = W @ Q
            CD = C - Cp; D = W - Wp
            def acc(cw, cd, w=None):
                w = np.ones(cw.shape[1]) if w is None else w
                V, dd, wd = float((w * cw * cw).sum()), float((w * cd * cd).sum()), float((w * cw * cd).sum())
                rho, c = np.sqrt(dd / V), wd / np.sqrt(V * dd)
                return dict(V=V, dV=dd + 2 * wd, rho=rho, c=c, ratio=c / (-rho / 2), rel=rho ** 2 + 2 * rho * c)
            r = dict(arm=arm, seed=s, t=t)
            r.update({f"S_{k}": v for k, v in acc(Cp, CD, lam).items()})                     # input-weighted (width-making)
            r.update({f"N_{k}": v for k, v in acc(Wp, D).items()})                             # plain norm
            for name, lo, hi in BANDS:
                r.update({f"{name}_{k}": v for k, v in acc(Cp[:, lo:hi], CD[:, lo:hi]).items()})
            r["sigma_med"] = float(np.median(np.sqrt((lam * C * C).sum(1))))
            rows.append(r); Wp, Cp = W, C
df = pd.DataFrame(rows)
df["block"] = pd.cut(df.t, [1, 5, 10, 20, 30, 40, 50], labels=["t2-5", "t6-10", "t11-20", "t21-30", "t31-40", "t41-50"])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
L = []
def P(x): L.append(str(x)); print(x)
for arm in ("A", "SA_iid", "F", "S_hi"):
    d = df[df.arm == arm]
    cols = ["S_V", "S_dV", "S_rho", "S_c", "S_ratio", "N_V", "N_dV", "N_rho", "N_c", "N_ratio", "sigma_med"]
    g = d.groupby(["block", "seed"], observed=True)[cols].mean().groupby("block", observed=True).median()
    P(f"\n== {arm}: input-weighted (S_*) vs plain norm (N_*)"); P(g.round(3).to_string())
    cols = [f"{b}_{k}" for b in ("top", "mid1", "mid2", "low") for k in ("V", "dV", "c", "rho", "ratio")]
    g = d.groupby(["block", "seed"], observed=True)[cols].mean().groupby("block", observed=True).median()
    P(f"== {arm}: per band (unweighted within band)"); P(g.loc[["t6-10", "t41-50"]].round(3).T.to_string())
    ex = []
    for _, x in d[d.t >= 11].groupby("seed"):
        ex.append((np.polyfit(np.log(x.t), np.log(x.S_V), 1)[0], np.polyfit(np.log(x.t), np.log(x.N_V), 1)[0], np.polyfit(np.log(x.t), np.log(x.sigma_med), 1)[0]))
    ex = np.median(np.array(ex), 0); P(f"   exponents t11-50 (median over seeds): V_weighted {ex[0]:.2f}, norm {ex[1]:.2f}, sigma_med {ex[2]:.2f}")
open("/home/issan/Projects/claude/wt/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923/posthoc_c_eff.txt", "w").write("\n".join(L) + "\n")
df.to_csv("/home/issan/Projects/claude/wt/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923/posthoc_c_eff.csv", index=False)
