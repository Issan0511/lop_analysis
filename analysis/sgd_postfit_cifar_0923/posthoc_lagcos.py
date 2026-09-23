#!/usr/bin/env python3
"""POST HOC (0923 night, Issa: is the orthogonality just high dimension?).
Gram -> cos between displacements by lag; radial cos(D_t, W_{t-1}); participation ratio of the ensemble; random baseline 1/sqrt(d)."""
import numpy as np, os, pandas as pd
SP = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
AL = os.path.expanduser("~/Projects/obsidian-research-data/altlabels_cifar_0923/results/altlabels_cifar_0923")
ARMS = {"A (Adam iid)": f"{SP}/A", "SA_iid (SGD iid)": f"{SP}/SA_iid", "LR_abab (Adam abab)": f"{AL}/LR_abab", "SA_abab (SGD abab)": f"{SP}/SA_abab"}
d = 100 * 3072
rows = []
for arm, base in ARMS.items():
    for s in range(10):
        W = np.stack([np.load(f"{base}/snap/LR_std_seed{s}/t{t:02d}.npz")["W1"].astype(np.float64).ravel() for t in range(1, 51)])
        D = W[1:] - W[:-1]                                   # 49 x d, D[k] = W(t=k+2) - W(t=k+1)
        G = D @ D.T; n = np.sqrt(np.diag(G)); C = G / np.outer(n, n)
        rad = np.array([(D[k] @ W[k]) / (n[k] * np.linalg.norm(W[k])) for k in range(49)])
        pr = G.trace() ** 2 / (G ** 2).sum()
        late = slice(24, 49)                                  # t = 26..50
        def lagmean(L):
            v = [C[i, i + L] for i in range(24, 49 - L)]
            return float(np.mean(v)) if v else np.nan
        rows.append(dict(arm=arm, seed=s, lag1=lagmean(1), lag2=lagmean(2), lag3=lagmean(3), lag4=lagmean(4),
                         lag10plus=float(np.mean([C[i, j] for i in range(24, 49) for j in range(i + 10, 49)])),
                         radial=float(rad[late].mean()), PR=float(pr), PR_late=float(G[late, late].trace() ** 2 / (G[late, late] ** 2).sum()),
                         rel_disp=float(np.mean(n[late] / np.linalg.norm(W[24:49], axis=1)))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"random baseline |cos| ~ 1/sqrt(d) = {1/np.sqrt(d):.4f}")
print(df.groupby("arm")[["lag1", "lag2", "lag3", "lag4", "lag10plus", "radial", "PR", "PR_late", "rel_disp"]].median().round(4).to_string())
df.to_csv("/home/issan/Projects/claude/wt/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923/posthoc_lagcos.csv", index=False)
