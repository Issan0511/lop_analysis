"""xbar_arrangement_1014 / agent A, follow-up: isotropic-directions model (coordinator request, 1014).
Model: w_perp ~ N(0, I) projected orthogonal to xbar, u_kn = w_perp . x~_n, so sd(u_kn | n) is proportional to |x~perp_n|,
x~perp_n = x~_n - (K_n/Kbar) xbar, and cot(theta_n) = K_n / (|xbar| |x~perp_n|).
(1) every phase-2 state: Spearman(f, cot theta) vs Spearman(f, K); f_iso from 20 draws per state (per alive unit g ~ N(0, I_785),
    g <- g - xbar (xbar.g)/Kbar, u = Xt g rescaled to sd_n(u) = s_perp_k, z_iso = m_k K_n/Kbar + u); f/f_iso per K-octile,
    Pearson(f, f_iso), max f vs max f_iso, mean p actual vs iso.
(2) every phase-2 state and init: regression over inputs of sd_k(u_kn) (alive units) on |x~perp_n|: slope, intercept, R^2
    (also for u_kn / s_perp_k, extra).
(3) T = 0 (init), 1, 2, 5, 10, 20, ..., 100: mean pairwise pattern corr, sd(ubar)*sqrt(H)/mean s_perp, eigenvalues of the
    alive x alive pattern-correlation matrix vs the iso null (H units, 20 draws).
(4) last phase-2 state (T = 100): mean pattern direction wbar_perp vs the top-5 principal directions of Sigma_perp; Spearman of
    ubar_n with cot theta and |x~perp|; argmax of ubar vs the persistent argmax-of-f input.
(5) PR(Sigma_perp) and the sd of the pairwise corr under the iso null (100 units, 20 draws).
usage: a2_iso.py <tag>   -> adds key "iso_followup" to results/xbar_arrangement_1014/a_occ_<tag>.json"""
import os, sys, json, math, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_occupancy import FILES, PHASE2, ROOT, load_X, pearson, spearman
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
R_ISO = 20
T3 = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def summ(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(med=float(np.median(v)), q10=float(np.quantile(v, 0.1)), q90=float(np.quantile(v, 0.9)), min=float(v.min()), max=float(v.max()))

def linfit(x, y):
    A = np.c_[x, np.ones_like(x)]; coef, *_ = np.linalg.lstsq(A, y, rcond=None); yh = A @ coef
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(); return float(coef[0]), float(coef[1]), float(r2)

def iso_patterns(rng, xbar, Kbar, Xt, H):
    G = rng.standard_normal((Xt.shape[1], H)); G -= np.outer(xbar, xbar @ G) / Kbar
    return Xt @ G                                                              # (N, H), mean_n = 0 exactly (xbar.g = 0)

def main(tag):
    t0 = time.time(); rng = np.random.default_rng(20261014)
    d = np.load(os.path.join(ROOT, FILES[tag])); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]; N = len(X)
    xbar = Xt.mean(0); K = Xt @ xbar; Kbar = float(xbar @ xbar); nx = math.sqrt(Kbar)
    Xp = Xt - np.outer(K / Kbar, xbar); nperp = np.linalg.norm(Xp, axis=1); cot = K / (nx * nperp)
    Sig = Xp.T @ Xp / N; ev, evec = np.linalg.eigh(Sig); ev = ev[::-1]; evec = evec[:, ::-1]
    PR = float(ev.sum() ** 2 / (ev ** 2).sum())
    edges = np.quantile(K, np.arange(1, 8) / 8.0); oc = np.searchsorted(edges, K, side="right")
    def state(T):
        if T == 0: W = d["init0"].astype(np.float64); b = d["init1"].astype(np.float64)
        else: W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64)
        Wt = np.c_[W, b]; Z = Xt @ Wt.T; m = Wt @ xbar; U = Z - np.outer(K / Kbar, m); O = Z > 0; idx = np.flatnonzero(O.mean(0) > 0)
        return Wt, Z, m, U, O, idx
    out = dict(PR_Sigma_perp=PR, Sigma_perp_top5_eig_frac=(ev[:5] / ev.sum()).tolist(), inv_sqrt_PR=1 / math.sqrt(PR),
               nperp=dict(min=float(nperp.min()), max=float(nperp.max()), mean=float(nperp.mean())),
               cot=dict(min=float(cot.min()), max=float(cot.max()), mean=float(cot.mean())),
               spearman_K_cot=spearman(K, cot), pearson_K_nperp=pearson(K, nperp), R_iso=R_ISO)
    # ---------- (1) and (2) over phase 2 ----------
    q1 = dict(sp_f_cot=[], sp_f_K=[], p_f_cot=[], p_f_K=[], p_f_fiso=[], sp_f_fiso=[], max_f=[], max_fiso_avg=[], max_fiso_draw=[], mean_p=[], mean_p_iso=[],
              p_fiso_K=[], oct_f=[], oct_fiso=[], ratio=[])
    q2 = dict(slope=[], intercept=[], r2=[], rel_intercept=[], n_slope=[], n_intercept=[], n_r2=[], n_rel_intercept=[])
    argmax_f = []
    for T in PHASE2:
        Wt, Z, m, U, O, idx = state(T); H = len(idx); f = O[:, idx].mean(1); sp = U[:, idx].std(0)
        argmax_f.append(int(np.argmax(f)))
        fiso = np.zeros(N); pis = []; mxd = []
        for _ in range(R_ISO):
            Ui = iso_patterns(rng, xbar, Kbar, Xt, H); Ui *= sp / Ui.std(0)
            Oi = (np.outer(K / Kbar, m[idx]) + Ui) > 0; fi = Oi.mean(1); fiso += fi / R_ISO; pis.append(float(Oi.mean())); mxd.append(float(fi.max()))
        q1["sp_f_cot"].append(spearman(f, cot)); q1["sp_f_K"].append(spearman(f, K)); q1["p_f_cot"].append(pearson(f, cot)); q1["p_f_K"].append(pearson(f, K))
        q1["p_f_fiso"].append(pearson(f, fiso)); q1["sp_f_fiso"].append(spearman(f, fiso)); q1["p_fiso_K"].append(pearson(fiso, K))
        q1["max_f"].append(float(f.max())); q1["max_fiso_avg"].append(float(fiso.max())); q1["max_fiso_draw"].append(float(np.mean(mxd)))
        q1["mean_p"].append(float(O[:, idx].mean())); q1["mean_p_iso"].append(float(np.mean(pis)))
        of = np.array([f[oc == o].mean() for o in range(8)]); oi = np.array([fiso[oc == o].mean() for o in range(8)])
        q1["oct_f"].append(of.tolist()); q1["oct_fiso"].append(oi.tolist()); q1["ratio"].append((of / np.where(oi > 0, oi, np.nan)).tolist())
        sdk = U[:, idx].std(1); s, c, r2 = linfit(nperp, sdk)
        q2["slope"].append(s); q2["intercept"].append(c); q2["r2"].append(r2); q2["rel_intercept"].append(c / sdk.mean())
        Un = U[:, idx] / sp; sdn = Un.std(1); s, c, r2 = linfit(nperp, sdn)
        q2["n_slope"].append(s); q2["n_intercept"].append(c); q2["n_r2"].append(r2); q2["n_rel_intercept"].append(c / sdn.mean())
    o1 = {k: summ(v) for k, v in q1.items() if k not in ("oct_f", "oct_fiso", "ratio")}
    for k in ("oct_f", "oct_fiso", "ratio"):
        M = np.array(q1[k], float); o1[k] = dict(med=np.nanmedian(M, 0).tolist(), q10=np.nanquantile(M, 0.1, axis=0).tolist(), q90=np.nanquantile(M, 0.9, axis=0).tolist())
    o1["n_states_cot_more_negative"] = int(sum(a < b for a, b in zip(q1["sp_f_cot"], q1["sp_f_K"])))
    o1["n_states_cot_abs_larger"] = int(sum(abs(a) > abs(b) for a, b in zip(q1["sp_f_cot"], q1["sp_f_K"])))
    o1["n_states"] = len(PHASE2)
    out["q1"] = o1
    o2 = {k: summ(v) for k, v in q2.items()}
    Wt, Z, m, U, O, idx = state(0); sdk = U[:, idx].std(1); s, c, r2 = linfit(nperp, sdk); sp0 = U[:, idx].std(0); sdn = (U[:, idx] / sp0).std(1); s2, c2, r22 = linfit(nperp, sdn)
    o2["init"] = dict(slope=s, intercept=c, r2=r2, rel_intercept=c / sdk.mean(), n_slope=s2, n_intercept=c2, n_r2=r22, n_rel_intercept=c2 / sdn.mean())
    out["q2"] = o2
    # ---------- (3) common component ----------
    q3 = {}
    for T in T3:
        Wt, Z, m, U, O, idx = state(T); H = len(idx); Ua = U[:, idx]; sp = Ua.std(0)
        C = np.corrcoef(Ua.T); iu = np.triu_indices(H, 1); lam, vec = np.linalg.eigh(C); lam = lam[::-1]; v1 = vec[:, -1]
        ubar = Ua.mean(1)
        nl = []; nmc = []
        for _ in range(R_ISO):
            Ui = iso_patterns(rng, xbar, Kbar, Xt, H); Ci = np.corrcoef(Ui.T); li = np.linalg.eigvalsh(Ci)[::-1]; nl.append(li[:3]); nmc.append(Ci[iu].mean())
        nl = np.array(nl)
        q3[str(T)] = dict(H=int(H), mean_corr=float(C[iu].mean()), sd_corr=float(C[iu].std()), sd_ubar_sqrtH_over_mean_sp=float(ubar.std() * math.sqrt(H) / sp.mean()),
                          lam=lam[:3].tolist(), null_lam_mean=nl.mean(0).tolist(), null_lam1_max=float(nl[:, 0].max()),
                          ratio_lam=(lam[:3] / nl.mean(0)).tolist(), n_above_null_max=int((lam > nl[:, 0].max()).sum()),
                          cos_v1_ones=float(abs(v1.sum()) / math.sqrt(H)), null_mean_corr=float(np.mean(nmc)))
    out["q3"] = q3
    # ---------- (4) last phase-2 state ----------
    Wt, Z, m, U, O, idx = state(100); wperp = Wt[idx] - np.outer(m[idx] / Kbar, xbar); wbar = wperp.mean(0)
    ubar = U[:, idx].mean(1); assert np.allclose(ubar, Xp @ wbar, atol=1e-8)
    cosv = [float(wbar @ evec[:, i] / np.linalg.norm(wbar)) for i in range(5)]
    var_share = [float(ev[i] * (wbar @ evec[:, i]) ** 2 / (wbar @ Sig @ wbar)) for i in range(5)]
    vals, cnts = np.unique(argmax_f, return_counts=True); persist = int(vals[np.argmax(cnts)])
    f100 = O[:, idx].mean(1); am_u = int(np.argmax(ubar)); rank_desc = np.argsort(-ubar)
    out["q4"] = dict(T=100, H=int(len(idx)), cos_wbar_pc=cosv, var_share_pc=var_share, pc_eig_frac=(ev[:5] / ev.sum()).tolist(),
                     sp_ubar_cot=spearman(ubar, cot), sp_ubar_nperp=spearman(ubar, nperp), sp_ubar_K=spearman(ubar, K),
                     argmax_ubar=am_u, K_argmax_ubar=float(K[am_u]), persistent_argmax_f=persist, K_persistent=float(K[persist]),
                     persistent_count=int(cnts.max()), argmax_f_T100=int(np.argmax(f100)), K_argmax_f_T100=float(K[int(np.argmax(f100))]),
                     coincide_ubar_persistent=bool(am_u == persist), rank_of_persistent_in_ubar=int(np.flatnonzero(rank_desc == persist)[0]) + 1,
                     cot_persistent=float(cot[persist]), cot_rank_of_persistent=int((cot > cot[persist]).sum()) + 1,
                     nperp_persistent=float(nperp[persist]))
    # ---------- (5) null sd of pairwise corr, 100 units ----------
    sds = []
    for _ in range(R_ISO):
        Ui = iso_patterns(rng, xbar, Kbar, Xt, 100); Ci = np.corrcoef(Ui.T); sds.append(Ci[np.triu_indices(100, 1)])
    allc = np.concatenate(sds); out["q5"] = dict(PR=PR, inv_sqrt_PR=1 / math.sqrt(PR), null_sd=float(allc.std()), null_mean=float(allc.mean()))
    out["elapsed_s"] = time.time() - t0
    p = os.path.join(RES, f"a_occ_{tag}.json"); J = json.load(open(p)); J["iso_followup"] = out; json.dump(J, open(p, "w"), indent=1)
    print(tag, "done", round(out["elapsed_s"], 1), "s PR", round(PR, 2), "sp_cot", round(o1["sp_f_cot"]["med"], 3), "sp_K", round(o1["sp_f_K"]["med"], 3),
          "ratio", [round(x, 2) for x in o1["ratio"]["med"]], flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
