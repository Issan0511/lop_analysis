#!/usr/bin/env python3
"""POST HOC (0923 night).
cycles 14-24) and a residual; are the residuals of different cycles mutually orthogonal (generic randomness)?"""
import numpy as np, os, pandas as pd
SP = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
AL = os.path.expanduser("~/Projects/obsidian-research-data/altlabels_cifar_0923/results/altlabels_cifar_0923")
rows = []
for arm, base in {"Adam LR_abab": f"{AL}/LR_abab", "SGD SA_abab": f"{SP}/SA_abab"}.items():
    for s in range(10):
        W = {t: np.load(f"{base}/snap/LR_std_seed{s}/t{t:02d}.npz")["W1"].astype(np.float64).ravel() for t in range(27, 51)}
        R = np.stack([(W[t + 1] - W[t]) * (1 if t % 2 == 1 else -1) for t in range(27, 50)])   # 23 transitions, aligned
        m = R.mean(0); shared = (R @ m)[:, None] * m[None, :] / (m @ m)
        res = R - shared
        G = res @ res.T; n = np.sqrt(np.diag(G)); C = G / np.outer(n, n)
        off = C[~np.eye(len(C), dtype=bool)]
        rows.append(dict(arm=arm, seed=s, shared_frac=float((shared ** 2).sum() / (R ** 2).sum()), resid_frac=float((res ** 2).sum() / (R ** 2).sum()),
                         resid_cos_mean=float(off.mean()), resid_cos_absmean=float(np.abs(off).mean()),
                         resid_PR=float(G.trace() ** 2 / (G ** 2).sum()), n_trans=len(R),
                         R_norm=float(np.median(np.linalg.norm(R, axis=1))), resid_norm=float(np.median(n))))
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(df.groupby("arm")[["shared_frac", "resid_frac", "resid_cos_mean", "resid_cos_absmean", "resid_PR", "n_trans", "R_norm", "resid_norm"]].median().round(3).to_string())
