"""V4 (verifier): LR_ref_s0 (a01s0) at T = 100 and at the initial weights:
(1) regression over the 1,200 inputs of the cross-unit sd (alive units) of u_perp_kn on |x~perp_n| (slope, intercept, R^2, intercept / mean sd),
    same for u/sperp_k; same for an iso draw with the same H and sperp (null reference);
(2) top eigenvalue of the alive x alive pattern-correlation matrix vs 20 iso draws (H units, unit-free: corr is scale-free);
(3) isotropy test (extra): per unit q_k = sperp_k^2 / (|w_perp_k|^2 tr(Sigma_perp)/784) (= 1 +- sqrt(2/PR) under (M'1));
    K-octile rms of u/sperp vs the model prediction |x~perp_n|/sqrt(tr Sigma_perp)."""
import json, math, os
import numpy as np
from v_common import load, weights, RES

def linfit(x, y):
    A = np.c_[x, np.ones_like(x)]; c, *_ = np.linalg.lstsq(A, y, rcond=None); yh = A @ c
    return dict(slope=float(c[0]), intercept=float(c[1]), R2=float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()), rel_intercept=float(c[1] / y.mean()))

def main():
    d, Xt, xbar, K, Kbar = load("a01s0"); N, D = Xt.shape
    Xp = Xt - np.outer(K / Kbar, xbar); nperp = np.linalg.norm(Xp, axis=1); Sig = Xp.T @ Xp / N; trS = float(np.trace(Sig))
    edges = np.quantile(K, np.arange(1, 8) / 8.0); oc = np.searchsorted(edges, K, side="right")
    model_oct = [float(np.sqrt((nperp[oc == o] ** 2).mean() / trS)) for o in range(8)]
    rng = np.random.default_rng(14100); out = dict(model_octile_rms_u_over_sperp=model_oct)
    for T in (0, 100):
        W = weights(d, T); Z = Xt @ W.T; m = W @ xbar; U = Z - np.outer(K / Kbar, m); alive = np.flatnonzero((Z > 0).sum(0) > 0); H = len(alive)
        Ua = U[:, alive]; sp = Ua.std(0); Wp = W[alive] - np.outer(m[alive] / Kbar, xbar)
        r = dict(H=H, raw=linfit(nperp, Ua.std(1)), norm=linfit(nperp, (Ua / sp).std(1)))
        q = sp ** 2 / ((Wp ** 2).sum(1) * trS / D); r["isotropy_q"] = dict(median=float(np.median(q)), q10=float(np.quantile(q, .1)), q90=float(np.quantile(q, .9)), mean=float(q.mean()))
        r["octile_rms_u_over_sperp"] = [float(np.sqrt(((Ua[oc == o] / sp) ** 2).mean())) for o in range(8)]
        C = np.corrcoef(Ua.T); ev = np.linalg.eigvalsh(C); lam1 = float(ev[-1]); nulls = []; nreg = []
        for _ in range(20):
            G = rng.standard_normal((D, H)); G -= np.outer(xbar, xbar @ G) / Kbar; Ui = Xt @ G; Ui *= (sp / Ui.std(0))[None, :]
            nulls.append(float(np.linalg.eigvalsh(np.corrcoef(Ui.T))[-1])); nreg.append(linfit(nperp, Ui.std(1)))
        r["lambda1"] = lam1; r["null_lambda1_mean"] = float(np.mean(nulls)); r["null_lambda1_max"] = float(np.max(nulls)); r["lambda1_over_null"] = lam1 / float(np.mean(nulls))
        r["null_reg_raw"] = {k: [float(np.min([x[k] for x in nreg])), float(np.max([x[k] for x in nreg]))] for k in ("slope", "rel_intercept", "R2")}
        iu = np.triu_indices(H, 1); r["pair_corr_mean"] = float(C[iu].mean()); r["pair_corr_sd"] = float(C[iu].std())
        out[f"T{T}"] = r
    json.dump(out, open(os.path.join(RES, "v4_spread.json"), "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
