"""Sufficient statistics that replace the assumption "pattern directions isotropic and independent across units"
(THEORY_XBAR.md 第 4 版 §2). Per condition and state T:
  PR(M)   participation ratio of the unit-averaged normalized-pattern covariance M = (1/H) sum_k uhat_k uhat_k^T
          (identity: mean_{k!=l} R_kl^2 = (H/(H-1)) (1/PR(M) - 1/H), R = pairwise pattern correlation over inputs)
  PR_pair = 1 / mean_{k!=l} R_kl^2, meanR, sdR
  share_common = |ubar|^2 / mean_k |u_k|^2  (ubar = mean_k u_k), lam1_share = top eigenvalue share of M
  S1 spread law: y_n = mean_k (u_kn/s_k)^2 vs x_n = |x_perp_n|^2 : R^2, relative intercept, log-log exponent
  S3 tail: xi_kn = u_kn / (s_k |x_perp_n| / rms|x_perp|): P(|xi|>t)/(2 Phibar(t)) for t = 2, 3; kurtosis
Isotropic reference per data subset: Gaussian directions orthogonal to xbar, H = 100, 20 draws (positions irrelevant
for normalized patterns). T = 0 uses the actual (uniform) initialization.
usage: suffstat.py [cond ...]  -> results/xbar_arrangement_1014/suffstat_<cond>.json, suffstat_summary.md"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v_common import load, weights, CONDS
import numpy as np
from math import erfc, sqrt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "xbar_arrangement_1014"))
STATES_SHOW = [0, 1, 2, 5, 10, 20, 40, 60, 80, 100]

def Phibar(x): return 0.5 * erfc(x / sqrt(2))

def pattern_stats(W, Xt, xbar, K, Kbar):
    Z = Xt @ W.T; m = W @ xbar
    U = Z - np.outer(K / Kbar, m); s = U.std(0); alive = s > 0
    U = U[:, alive]; s = s[alive]; H = U.shape[1]
    Uh = U / np.linalg.norm(U, axis=0, keepdims=True)
    R = Uh.T @ Uh; off = R[~np.eye(H, dtype=bool)]
    meanR2 = float((off ** 2).mean()); meanR = float(off.mean()); sdR = float(off.std())
    ev = np.linalg.eigvalsh(R / H); ev = ev[ev > 1e-12]
    PR_M = float(ev.sum() ** 2 / (ev ** 2).sum())
    ident = (H / (H - 1)) * (1 / PR_M - 1 / H)
    ubar = U.mean(1); share_common = float(ubar @ ubar / (U ** 2).sum(0).mean())
    lam1_share = float(ev.max() / ev.sum())
    Xp = Xt - np.outer(K / Kbar, xbar); xp2 = (Xp ** 2).sum(1)
    y = ((U / s[None, :]) ** 2).mean(1)
    A = np.c_[xp2, np.ones_like(xp2)]; coef = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ coef; R2 = float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rel_icpt = float(coef[1] / (coef[0] * xp2.mean())); expo = float(np.polyfit(np.log(xp2), np.log(y), 1)[0])
    scale = np.sqrt(xp2 / xp2.mean()); xi = U / (s[None, :] * scale[:, None])
    tail = {f"p{t}": float((np.abs(xi) > t).mean() / (2 * Phibar(t))) for t in (2, 3)}
    kurt = float(((xi - xi.mean()) ** 4).mean() / xi.var() ** 2)
    xi2 = (U / s[None, :]) / np.sqrt(y)[:, None]            # per-input empirical scale (S3 separated from S1)
    tail_emp = {f"p{t}": float((np.abs(xi2) > t).mean() / (2 * Phibar(t))) for t in (2, 3)}
    kurt_emp = float(((xi2 - xi2.mean()) ** 4).mean() / xi2.var() ** 2)
    return dict(tail_emp=tail_emp, xi_emp_kurtosis=kurt_emp, H_alive=int(H), PR_M=PR_M, PR_pair=1 / meanR2, meanR2=meanR2, identity_rhs=float(ident), meanR=meanR, sdR=sdR,
                share_common=share_common, lam1_share=lam1_share, S1_R2=R2, S1_rel_intercept=rel_icpt, S1_loglog_exponent=expo,
                tail_ratio=tail, xi_kurtosis=kurt, m_neg_frac=float((m[alive] < 0).mean()))

KEYS = ["PR_M", "PR_pair", "meanR", "sdR", "share_common", "lam1_share", "S1_R2", "S1_rel_intercept", "S1_loglog_exponent", "xi_kurtosis", "xi_emp_kurtosis"]

def iso_reference(Xt, xbar, K, Kbar, H=100, draws=20, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(draws):
        G = rng.standard_normal((Xt.shape[1], H)); G -= np.outer(xbar, (xbar @ G) / Kbar)
        out.append(pattern_stats(G.T, Xt, xbar, K, Kbar))
    agg = {k: [float(np.mean([o[k] for o in out])), float(np.std([o[k] for o in out]))] for k in KEYS}
    agg["tail_ratio"] = {t: float(np.mean([o["tail_ratio"][t] for o in out])) for t in ("p2", "p3")}
    agg["tail_emp"] = {t: float(np.mean([o["tail_emp"][t] for o in out])) for t in ("p2", "p3")}
    return agg

def PR_sigma_perp(Xt, xbar, K, Kbar):
    Xp = Xt - np.outer(K / Kbar, xbar); S = Xp.T @ Xp / len(Xp); ev = np.linalg.eigvalsh(S)
    return float(ev.sum() ** 2 / (ev ** 2).sum())

def run(cond):
    d, Xt, xbar, K, Kbar = load(cond)
    res = dict(cond=cond, PR_sigma_perp=PR_sigma_perp(Xt, xbar, K, Kbar), iso=iso_reference(Xt, xbar, K, Kbar), states={})
    for T in range(0, 101):
        res["states"][T] = pattern_stats(weights(d, T), Xt, xbar, K, Kbar)
    ph2 = [res["states"][T] for T in range(11, 101)]
    res["phase2"] = {k: [float(np.mean([p[k] for p in ph2])), float(np.min([p[k] for p in ph2])), float(np.max([p[k] for p in ph2]))] for k in KEYS}
    res["phase2"]["tail_p3"] = [float(np.mean([p["tail_ratio"]["p3"] for p in ph2])), float(np.min([p["tail_ratio"]["p3"] for p in ph2])), float(np.max([p["tail_ratio"]["p3"] for p in ph2]))]
    res["phase2"]["tail_emp_p3"] = [float(np.mean([p["tail_emp"]["p3"] for p in ph2])), float(np.min([p["tail_emp"]["p3"] for p in ph2])), float(np.max([p["tail_emp"]["p3"] for p in ph2]))]
    res["phase2"]["identity_max_abs_err"] = float(max(abs(p["meanR2"] - p["identity_rhs"]) for p in ph2))
    with open(os.path.join(OUT, f"suffstat_{cond}.json"), "w") as f: json.dump(res, f, indent=1)
    return res

def fmt(r, k, T=None):
    v = r["states"][T][k] if T is not None else r["phase2"][k]
    return f"{v:.3f}" if T is not None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

if __name__ == "__main__":
    conds = sys.argv[1:] or list(CONDS)
    rows = []
    for c in conds:
        r = run(c); iso = r["iso"]
        rows.append((c, r))
        print(f"{c}: PR(Σ⊥) {r['PR_sigma_perp']:.1f} | iso PR(M) {iso['PR_M'][0]:.1f}±{iso['PR_M'][1]:.1f} sdR {iso['sdR'][0]:.3f} S1 R² {iso['S1_R2'][0]:.2f} icpt {iso['S1_rel_intercept'][0]:+.2f} expo {iso['S1_loglog_exponent'][0]:.2f} tail3 {iso['tail_ratio']['p3']:.2f} tail3emp {iso['tail_emp']['p3']:.2f} PRpair {iso['PR_pair'][0]:.1f} common {iso['share_common'][0]:.3f} λ1 {iso['lam1_share'][0]:.3f}"
              f" | T0 PR {fmt(r,'PR_M',0)} sdR {fmt(r,'sdR',0)} | T1 PR {fmt(r,'PR_M',1)} | T10 PR {fmt(r,'PR_M',10)}"
              f" | ph2 PR(M) {fmt(r,'PR_M')} PR_pair {fmt(r,'PR_pair')} meanR {fmt(r,'meanR')} sdR {fmt(r,'sdR')} common {fmt(r,'share_common')} λ1 {fmt(r,'lam1_share')}"
              f" S1 R² {fmt(r,'S1_R2')} icpt {fmt(r,'S1_rel_intercept')} expo {fmt(r,'S1_loglog_exponent')} tail3 {r['phase2']['tail_p3'][0]:.2f} tail3emp {r['phase2']['tail_emp_p3'][0]:.2f} [{r['phase2']['tail_emp_p3'][1]:.2f}, {r['phase2']['tail_emp_p3'][2]:.2f}] kurt {fmt(r,'xi_kurtosis')} kurtemp {fmt(r,'xi_emp_kurtosis')} T0: tail3emp {r['states'][0]['tail_emp']['p3']:.2f} common {fmt(r,'share_common',0)} expo {fmt(r,'S1_loglog_exponent',0)} | T1: common {fmt(r,'share_common',1)} meanR {fmt(r,'meanR',1)} expo {fmt(r,'S1_loglog_exponent',1)} ident_err {r['phase2']['identity_max_abs_err']:.1e}", flush=True)
    lines = ["| 条件 | PR(Σ⊥) | 等方の参照 PR(M)（H=100） | 初期 T0 | T1 | T10 | 相 2 PR(M) 平均 [min, max] | 相 2 PR_pair | 相 2 相関の平均 | 相 2 相関の sd | 共通成分の分担 | λ1 の分担 | S1 R² | S1 相対切片 | S1 指数 | 裾 3σ 比（等方の尺度） | 裾 3σ 比（入力ごとの尺度） |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in rows:
        iso = r["iso"]
        lines.append(f"| {c} | {r['PR_sigma_perp']:.1f} | {iso['PR_M'][0]:.1f} ± {iso['PR_M'][1]:.1f}（sdR {iso['sdR'][0]:.3f}、S1 R² {iso['S1_R2'][0]:.2f}、切片 {iso['S1_rel_intercept'][0]:+.2f}、裾 {iso['tail_ratio']['p3']:.2f}） | {fmt(r,'PR_M',0)} | {fmt(r,'PR_M',1)} | {fmt(r,'PR_M',10)} | {fmt(r,'PR_M')} | {fmt(r,'PR_pair')} | {fmt(r,'meanR')} | {fmt(r,'sdR')} | {fmt(r,'share_common')} | {fmt(r,'lam1_share')} | {fmt(r,'S1_R2')} | {fmt(r,'S1_rel_intercept')} | {fmt(r,'S1_loglog_exponent')} | {r['phase2']['tail_p3'][0]:.2f} | {r['phase2']['tail_emp_p3'][0]:.2f} [{r['phase2']['tail_emp_p3'][1]:.2f}, {r['phase2']['tail_emp_p3'][2]:.2f}]（参照 {iso['tail_emp']['p3']:.2f}） |")
    with open(os.path.join(OUT, "suffstat_summary.md"), "w") as f: f.write("\n".join(lines) + "\n")
