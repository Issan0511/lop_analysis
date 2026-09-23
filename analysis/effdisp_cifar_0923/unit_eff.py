#!/usr/bin/env python3
"""POST HOC (0923 night).
sigma_i = sqrt(w_i^T Sigma w_i) (pre-activation width, centered input covariance from the seed's PCA, 1199 PCs),
d_i = sqrt(dw_i^T Sigma dw_i) (effective displacement), c_i = w_i^T Sigma dw_i / (sigma_i d_i) (radial cosine),
out_i = ||W2[:, i]|| (outgoing weight).  Per unit: window means; predicted sigma* = d / (2|c|) with c the aggregate cosine
sum(x)/sum(sigma d).  Cross-unit statistics per arm x seed, then medians over seeds."""
import numpy as np, os, pandas as pd, sys
ARCH = os.path.expanduser("~/Projects/obsidian-research-data")
SP = f"{ARCH}/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923"; EW = f"{ARCH}/eps_width_cifar_0923/results/eps_width_cifar_0923"
B = f"{ARCH}/sgd_postfit_cifar_0923/basis_cache"
ARMS = {"A": (f"{SP}/A", "LR", range(41, 51)), "A_ce": (f"{SP}/A_ce", "LR", range(41, 51)), "F": (f"{SP}/F", "LR", range(41, 51)),
        "S_hi": (f"{SP}/S_hi", "LR", range(41, 51)), "SA_iid": (f"{SP}/SA_iid", "LR", range(41, 51)),
        "LL_e8": (f"{EW}/LL_e8", "LL", range(11, 21)), "LL_e6": (f"{EW}/LL_e6", "LL", range(11, 21)),
        "LL_e4": (f"{EW}/LL_e4", "LL", range(11, 21)), "LL_e3": (f"{EW}/LL_e3", "LL", range(11, 21))}
units, summ = [], []
for s in range(10):
    z = np.load(f"{B}/basis_std_s{s}.npz"); Q = z["Q"][:, :1199].astype(np.float64); lam = z["lam"].astype(np.float64)
    for arm, (d, pre, win) in ARMS.items():
        acc = {k: [] for k in ("sig", "d", "x", "sd", "out", "V", "dV")}
        prev = None
        for t in range(win.start - 1, win.stop):
            P = np.load(f"{d}/snap/{pre}_std_seed{s}/t{t:02d}.npz"); C = P["W1"].astype(np.float64) @ Q
            if prev is not None:
                Cp, W2p = prev; D = C - Cp
                V = (lam * Cp * Cp).sum(1); dd = (lam * D * D).sum(1); x = (lam * Cp * D).sum(1)
                acc["sig"].append(np.sqrt(V)); acc["d"].append(np.sqrt(dd)); acc["x"].append(x); acc["sd"].append(np.sqrt(V * dd))
                acc["out"].append(np.linalg.norm(W2p, axis=0)); acc["V"].append(V); acc["dV"].append((lam * C * C).sum(1) - V)
            prev = (C, P["W2"].astype(np.float64))
        a = {k: np.mean(np.array(v), 0) for k, v in acc.items()}
        c = np.array(acc["x"]).sum(0) / np.array(acc["sd"]).sum(0)
        pred = a["d"] / (2 * np.abs(c))
        bal = c / (-(a["d"] / a["sig"]) / 2)
        for i in range(100):
            units.append(dict(arm=arm, seed=s, unit=i, sigma=a["sig"][i], d=a["d"][i], c=c[i], pred=pred[i], bal=bal[i], out=a["out"][i], rel_dV=a["dV"][i] / a["V"][i]))
        ok = c < 0
        ls, lp, ld, lc, lo = np.log(a["sig"][ok]), np.log(pred[ok]), np.log(a["d"][ok]), np.log(np.abs(c[ok])), np.log(a["out"][ok])
        X = np.column_stack([np.ones(ok.sum()), ld, lc]); coef = np.linalg.lstsq(X, ls, rcond=None)[0]
        vd, vc, cov = np.var(ld), np.var(lc), np.cov(ld, lc, bias=True)[0, 1]
        summ.append(dict(arm=arm, seed=s, frac_c_neg=ok.mean(), slope_sig_on_pred=np.polyfit(lp, ls, 1)[0], r_sig_pred=np.corrcoef(lp, ls)[0, 1],
                         med_ratio=np.median(np.exp(ls - lp)), med_bal=np.median(bal[ok]), coef_d=coef[1], coef_c=coef[2],
                         share_d=(vd + cov) / (vd + vc + 2 * cov) if False else (vd - cov) / (vd + vc - 2 * cov),
                         cv_d=np.std(a["d"][ok]) / np.mean(a["d"][ok]), cv_c=np.std(c[ok]) / np.abs(np.mean(c[ok])), cv_sig=np.std(a["sig"][ok]) / np.mean(a["sig"][ok]),
                         slope_d_on_sig=np.polyfit(ls, ld, 1)[0], r_d_out=np.corrcoef(ld, lo)[0, 1], slope_d_on_out=np.polyfit(lo, ld, 1)[0],
                         med_c=np.median(c[ok]), med_rel_dV=np.median(a["dV"][ok] / a["V"][ok])))
U = pd.DataFrame(units); Sm = pd.DataFrame(summ)
out = str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'results/effdisp_cifar_0923')
U.to_csv(f"{out}/unit_eff_units.csv", index=False); Sm.to_csv(f"{out}/unit_eff_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
order = ["SA_iid", "S_hi", "F", "A_ce", "A", "LL_e3", "LL_e4", "LL_e6", "LL_e8"]
print(Sm.groupby("arm").median(numeric_only=True).drop(columns=["seed"]).loc[order].round(3).T.to_string())
