"""V1 (verifier): numeric checks of THEORY_XBAR.md lemmas 2, 4, 5(a)(c) on the seed-0 1,200-input subset (LR_ref_s0 subset).
MC: g ~ N(0, I_785), g <- g - xbar (xbar.g)/Kbar (sigma = 1), u = Xt g; 200 units per rep, 20 reps."""
import json, math, os
import numpy as np
from v_common import load, weights, RES

def normcdf_bar(x): return 0.5 * math.erfc(x / math.sqrt(2))

def main():
    d, Xt, xbar, K, Kbar = load("a01s0"); N, D = Xt.shape
    nx = math.sqrt(Kbar); Xp = Xt - np.outer(K / Kbar, xbar); nperp = np.linalg.norm(Xp, axis=1)
    Sig = Xp.T @ Xp / N; ev = np.linalg.eigvalsh(Sig)[::-1]; trS = float(ev.sum()); trS2 = float((ev ** 2).sum()); PR = trS ** 2 / trS2
    cot = K / (nx * nperp); cos = K / (np.linalg.norm(Xt, axis=1) * nx); sin = np.sqrt(1 - cos ** 2)
    out = dict(N=N, D=D, xbar_norm=nx, Kbar=Kbar, trSigma_perp=trS, trSigma_perp2=trS2, PR=PR, inv_sqrt_PR=1 / math.sqrt(PR),
               K=dict(min=float(K.min()), max=float(K.max()), mean=float(K.mean()), sd=float(K.std())),
               nperp=dict(min=float(nperp.min()), mean=float(nperp.mean()), max=float(nperp.max())),
               cot=dict(min=float(cot.min()), mean=float(cot.mean()), max=float(cot.max())), pearson_cot_K=float(np.corrcoef(cot, K)[0, 1]))
    # PR variants (to locate THEORY's 30.4)
    Xc = Xt - Xt.mean(0); evc = np.linalg.eigvalsh(Xc.T @ Xc / N); out["PR_centered_cov"] = float(evc.sum() ** 2 / (evc ** 2).sum())
    Xp784 = Xp[:, :784]; ev784 = np.linalg.eigvalsh(Xp784.T @ Xp784 / N); out["PR_Sigma_perp_pixels_only"] = float(ev784.sum() ** 2 / (ev784 ** 2).sum())
    # identities
    out["id_sum_xperp_maxabs"] = float(np.abs(Xp.sum(0)).max())
    out["id_lemma4_maxrel"] = float(np.abs(K / nperp - nx * cot).max() / (K / nperp).max())
    out["id_cot_eq_cos_over_sin_maxrel"] = float(np.abs(cot - cos / sin).max() / cot.max())
    W = weights(d, 100); Z = Xt @ W.T; m = W @ xbar; U = Z - np.outer(K / Kbar, m)
    Wp = W - np.outer(m / Kbar, xbar); out["id_lemma2_u_eq_wperp_xperp_maxabs"] = float(np.abs(U - Xp @ Wp.T).max())
    out["id_lemma2_u_mean_maxabs"] = float(np.abs(U.mean(0)).max()); out["id_m_eq_mean_z_maxabs"] = float(np.abs(m - Z.mean(0)).max())
    # open-condition identity (lemma 4 / 5b) on T100: z > 0  <=>  u/|xperp| > -(m/|xbar|) cot  (any sign of m)
    lhs = Z > 0; rhs = (U / nperp[:, None]) > (-(m / nx)[None, :] * cot[:, None]); out["id_open_cond_mismatch"] = int((lhs != rhs).sum())
    # ---- MC lemma 5 (a)(c)
    rng = np.random.default_rng(7140); R = 20; H = 200
    sl, ic, r2, ratio_q, s2rel, kurt, tails, corr_sd, corr_sd100, corr_mean, xi_var = [], [], [], [], [], [], [], [], [], [], []
    tail_pts = [1.0, 2.0, 3.0]
    tail_acc = np.zeros(len(tail_pts)); tail_acc_known = np.zeros(len(tail_pts)); cnt = 0
    for r in range(R):
        G = rng.standard_normal((D, H)); G -= np.outer(xbar, xbar @ G) / Kbar
        U = Xt @ G                                      # (N, H), sigma = 1
        sd_n = U.std(1)                                 # cross-unit sd per input (mean over units ~ 0)
        A = np.c_[nperp, np.ones(N)]; coef, *_ = np.linalg.lstsq(A, sd_n, rcond=None); yh = A @ coef
        sl.append(coef[0]); ic.append(coef[1] / sd_n.mean()); r2.append(1 - ((sd_n - yh) ** 2).sum() / ((sd_n - sd_n.mean()) ** 2).sum())
        ratio = sd_n / nperp; ratio_q.append([float(np.quantile(ratio, q)) for q in (0.05, 0.5, 0.95)])
        sp = U.std(0); s2rel.append(sp ** 2 / trS)      # s_perp^2 / (sigma^2 tr Sigma_perp)
        xi = U * math.sqrt(trS) / (sp[None, :] * nperp[:, None]); xk = U / nperp[:, None]   # realized-s rescaled vs known-sigma
        xi_var.append(float(xi.var())); kurt.append(float(((xi - xi.mean()) ** 4).mean() / xi.var() ** 2 - 3))
        for j, t in enumerate(tail_pts): tail_acc[j] += (xi > t).sum(); tail_acc_known[j] += (xk > t).sum()
        cnt += xi.size
        C = np.corrcoef(U.T); iu = np.triu_indices(H, 1); corr_sd.append(float(C[iu].std())); corr_mean.append(float(C[iu].mean()))
        C1 = C[:100, :100]; iu1 = np.triu_indices(100, 1); corr_sd100.append(float(C1[iu1].std()))
    s2rel = np.concatenate(s2rel)
    out["MC"] = dict(R=R, H=H,
        spread_slope_vs_nperp=dict(mean=float(np.mean(sl)), min=float(np.min(sl)), max=float(np.max(sl))),
        spread_rel_intercept=dict(mean=float(np.mean(ic)), min=float(np.min(ic)), max=float(np.max(ic))),
        spread_R2=dict(mean=float(np.mean(r2)), min=float(np.min(r2)), max=float(np.max(r2))),
        sd_over_nperp_q05_q50_q95_mean=np.mean(ratio_q, 0).tolist(),
        s2_over_trS=dict(mean=float(s2rel.mean()), sd=float(s2rel.std()), q05=float(np.quantile(s2rel, .05)), q95=float(np.quantile(s2rel, .95)), predicted_rel_sd_sqrt_2_over_PR=math.sqrt(2 / PR)),
        xi_var=float(np.mean(xi_var)), xi_excess_kurt=float(np.mean(kurt)),
        xi_tail=[dict(t=t, P_realized_s=float(tail_acc[j] / cnt), P_known_sigma=float(tail_acc_known[j] / cnt), Phibar=normcdf_bar(t), ratio_realized_over_Phibar=float(tail_acc[j] / cnt / normcdf_bar(t))) for j, t in enumerate(tail_pts)],
        corr_mean=float(np.mean(corr_mean)), corr_sd_H200=dict(mean=float(np.mean(corr_sd)), min=float(np.min(corr_sd)), max=float(np.max(corr_sd))),
        corr_sd_H100=dict(mean=float(np.mean(corr_sd100)), min=float(np.min(corr_sd100)), max=float(np.max(corr_sd100)), q05=float(np.quantile(corr_sd100, .05)), q95=float(np.quantile(corr_sd100, .95))),
        corr_sd_over_inv_sqrt_PR=float(np.mean(corr_sd) * math.sqrt(PR)))
    json.dump(out, open(os.path.join(RES, "v1_lemma5.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
