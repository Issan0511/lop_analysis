"""slopes_counts.py (cobirth_seeds_1013, TASK B tag 'slopes', 1014).

Generalized copy of analysis/cobirth_1013/m1_counts.py (M1: N_i(T) presence and births B_i(T) per alive target, per-target Fano, pooled vs Poisson /
per-target Poisson mixture, null 1 (per-candidate circular shift / permutation), null 1b (width-stratified permutation), null 2 (target shuffle),
time windows) + the section-4 variance decomposition of analysis/cobirth_1013/m3_hubs.py (target identity / width bin / task / threshold pool C_i /
(C_i, width); Fano of C_i vs the width-preserving uniform null U) + Cantelli with per-target n_beta at two states (nearest state, per-target minimum,
each state alone, constants) + the break-even n* (n* at which the Cantelli bound equals c: n*_be = mu + sigma sqrt((1-c)/c)).
Box: 784-100-10 MLP, 1200 fixed MNIST inputs, hard leaky ReLU slope a, CE, Adam 1e-3, batch 16, 4000 updates/task, iid uniform 10-class labels,
100 tasks. Saved task-end weights only (no training).

Task convention: M1 (T = 1..100, state end0[T-1]; Phase 2 = T 11..100, 90 task ends).
  stoch_bridges / p159 / p161 / p170 / p172c use t = T + 1 (state end0[t-2]); the reproduction block converts explicitly (stoch rows t = 10..100 = T 9..99,
  births t = 11..100 = T 10..99; P159 reference t = 40 / 100 = T 39 / 99, along t = 2/10/20/40/100 = T 1/9/19/39/99; P161 turnover t = 10..60 = T 9..59).
  M3 used t = T + 1 with phase 2 = t 10..101 (= T 9..100, 92 states); the decomposition here is on T 11..100 unless --decomp-m3 (then also on T 9..100).
Bridge (lifter) criterion: bridges_at copied verbatim (stoch_bridges.py -> m1_counts.py). The threshold inputs T_i and the counts |O_j & T_i| used for the
pool C_i are recomputed with the same two lines (as m3_hubs.py does).

usage:
  slopes_counts.py run    --npz <path> --a <slope> --tag <cond> [--nbeta <p172c json> ...] [--rnull 1000] [--decomp-m3]
  slopes_counts.py anchor --npz <path> --a <slope> --tag <cond> [--decomp-m3]      (reproduction only)
  slopes_counts.py combine                                                          (RAW/slopes_*_full.json -> results/cobirth_seeds_1013/slopes.json)
CPU, 1 thread, float64."""
import os, sys, json, math, time, glob, argparse
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/slopes"
WT = "/home/issan/Projects/claude/wt/cobirth_seeds_1013"
OUT_JSON = os.path.join(WT, "results/cobirth_seeds_1013/slopes.json")
H = 100; NT = 100; PH2 = list(range(11, 101)); WINDOWS = [(11, 40), (41, 70), (71, 100)]; R_NULL = 1000; SEED = 0; MIN_ALIVE = 30; MIN_ALIVE_WIN = 15
KBINS_FINE = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 1200)]   # m3_hubs.py
R_MC_U = 20; SEED_U = 20261013                                                                                                      # m3_hubs.py
C_LEVELS = (0.32, 0.26, 0.19, 0.14, 0.11, 0.08)   # rho_R thresholds of cumulative_push section 3 (c = 1: 32/26/19 %, c = 2.8: 14/11/8 % for a0.1/0.3/0.5)
F_UNIFORM = (3.0, 6.4)                             # README 4.1 uniform F-bar values
nan = float("nan")
_LOG = None
def P(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True)
    if _LOG is not None: _LOG.write(s + "\n"); _LOG.flush()

# published numbers used as anchors (keyed by npz basename). stoch: stoch_counts.log (t convention); p159/p161: P177 table (stage16_1008_seed1/logs/p177_table.log)
PUB = {"LR_ref_s0.npz": dict(stoch=dict(rows=1511, births=1399, tt=1078, nt_mean=1.38, nt_max=7, B_mean=0.198, B_max=6), p159_ref=dict(T=39, max=5, ge3=2, ge10=0, n=82), p159_along=[54, 4, 3, 5, 2], p161=dict(count_median=17, persist=0.07, life1=0.94)),
       "LRa0.3_s0.npz": dict(stoch=dict(rows=2836, births=2557, tt=1840, nt_mean=1.53, nt_max=14, B_mean=0.484, B_max=11), p159_ref=dict(T=99, max=5, ge3=1, ge10=0, n=100), p159_along=[60, 4, 5, 4, 5], p161=dict(count_median=34, persist=0.08, life1=0.92),
                             m3=dict(src="obsidian-research-data/cobirth_1013/m3/m3_a03.json (CU_total_fano 0.98695; M3_REPORT prose rounds it to 1.00)", n=9190, N_unit_share=0.020, N_unit_fano_within=1.92, N_Ck_fano_within=0.91, N_C_fano_within=1.41, C_fano=1.94, CU_total_fano=0.99)),
       "LRa0.5_s0.npz": dict(stoch=dict(rows=7712, births=6649, tt=4685, nt_mean=1.64, nt_max=10, B_mean=0.783, B_max=10), p159_ref=dict(T=99, max=4, ge3=2, ge10=0, n=100), p159_along=[70, 8, 4, 7, 4], p161=dict(count_median=87, persist=0.11, life1=0.88)),
       "LR_ref_s1.npz": dict(p159_ref=dict(T=39, max=6, ge3=1, ge10=0, n=91), p159_along=[71, 3, 2, 6, 1], p161=dict(count_median=16, persist=0.03, life1=0.95)),
       "LRa0.3_s1.npz": dict(p159_ref=dict(T=99, max=4, ge3=1, ge10=0, n=100), p159_along=[71, 5, 6, 2, 4], p161=dict(count_median=30, persist=0.06, life1=0.93)),
       "LRa0.5_s1.npz": dict(p159_ref=dict(T=99, max=5, ge3=19, ge10=0, n=100), p159_along=[73, 6, 6, 4, 5], p161=dict(count_median=88, persist=0.10, life1=0.89))}

def bridges_at(O, Dp, Kv):
    """verbatim from stoch_bridges.py bridges_at (rows = target i, cols = candidate j)."""
    alive = O.any(0); k = O.sum(0); need = (k + 1) // 2; Oi = O.astype(np.int32); ov = Oi.T @ Oi
    E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    isb = (thr >= 2) & (ov >= need[:, None]); isb[np.diag_indices(H)] = False; isb[~alive, :] = False; isb[:, ~alive] = False
    return isb, alive, k

def thresholds_and_counts(O, Dp, Kv):
    """the same two lines as bridges_at (and m3_hubs.py): T_i = top-3 of K_n dd/dz, thr[i, j] = |O_j & T_i|."""
    E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    return T, thr

def pois_tail(lam, k):
    """P(Pois(lam) >= k), k real -> ceil."""
    if not np.isfinite(k): return 0.0
    k = int(math.ceil(k - 1e-12))
    if k <= 0: return 1.0
    if lam <= 0: return 0.0
    return float(max(0.0, 1.0 - sum(math.exp(-lam + x * math.log(lam) - math.lgamma(x + 1)) for x in range(k))))

def pois_pmf(lam, v):
    if lam <= 0: return 1.0 if v == 0 else 0.0
    return math.exp(-lam + v * math.log(lam) - math.lgamma(v + 1))

def cantelli(mu, var, nstar):
    if not np.isfinite(nstar): return 0.0
    if not np.isfinite(var) or nstar <= mu: return 1.0
    if var <= 0: return 0.0
    return float(var / (var + (nstar - mu) ** 2))

def ind_stats(M):
    """M: (n tasks, m candidates) 0/1 indicator matrix for one target. Row sums N_t. Returns dict incl. pairwise excess."""
    n = M.shape[0]; Nn = M.sum(1).astype(float); mu = float(Nn.mean()); var0 = float(Nn.var()); var1 = float(Nn.var(ddof=1)) if n > 1 else nan
    p = M.mean(0); sp2 = float((p ** 2).sum())
    G = (M.T.astype(float) @ M.astype(float)) / n - np.outer(p, p); delta = float(G.sum() - np.trace(G))
    return dict(n=n, mu=mu, var0=var0, var1=var1, fano=(var1 / mu if mu > 0 else nan), max=int(Nn.max()), ge2=float((Nn >= 2).mean()), ge3=float((Nn >= 3).mean()),
                sp2=sp2, delta=delta, delta_id=var0 - mu + sp2, var1_minus_mu=(var1 - mu) if n > 1 else nan, hist=np.bincount(Nn.astype(int)).tolist(), N=Nn)

def null_cols(M, R, rng, mode):
    """mode 'shift': independent circular shift of each candidate's series; 'perm': independent permutation. Marginals kept.
    Returns array (R, 5): fano, max, ge2, ge3, delta (= var0 - mu + sum p^2, since sum p^2 and mu are invariant)."""
    n, m = M.shape; ar = np.arange(n); p = M.mean(0); sp2 = float((p ** 2).sum()); mu = float(M.sum(1).mean()); out = np.empty((R, 5))
    if mode == "strat":   # independent permutation of each candidate's series WITHIN strata of the target's width k_i(T) (quantile bins); keeps P(X_j | k-bin)
        kb = null_cols.strata; groups = [np.where(kb == g)[0] for g in np.unique(kb)]
    for r in range(R):
        if mode == "perm": idx = np.argsort(rng.random((n, m)), axis=0)
        elif mode == "strat":
            idx = np.tile(ar[:, None], (1, m))
            for g in groups:
                if len(g) > 1: idx[g] = g[np.argsort(rng.random((len(g), m)), axis=0)]
        else: off = rng.integers(0, n, size=m); idx = (ar[:, None] - off[None, :]) % n
        Nn = np.take_along_axis(M, idx, axis=0).sum(1).astype(float); var1 = Nn.var(ddof=1); var0 = Nn.var()
        out[r] = (var1 / mu if mu > 0 else nan, Nn.max(), (Nn >= 2).mean(), (Nn >= 3).mean(), var0 - mu + sp2)
    return out

def pval2(obs, nulls):
    """two-sided MC p-value (with +1 correction); NaN-safe."""
    v = nulls[np.isfinite(nulls)]
    if not np.isfinite(obs) or len(v) == 0: return nan
    hi = (1 + (v >= obs).sum()) / (len(v) + 1); lo = (1 + (v <= obs).sum()) / (len(v) + 1); return float(min(1.0, 2 * min(hi, lo)))

def q(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0: return {str(x): None for x in qs} | {"n": 0, "max": None, "min": None}
    return {str(x): float(np.quantile(v, x)) for x in qs} | {"n": int(len(v)), "max": float(v.max()), "min": float(v.min()), "mean": float(v.mean())}

def rnd(o):
    if isinstance(o, dict): return {str(k): rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v) for v in o]
    if isinstance(o, np.ndarray): return rnd(o.tolist())
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        if not np.isfinite(o): return None if np.isnan(o) else ("inf" if o > 0 else "-inf")
        return float(f"{o:.6g}")
    return o

def full(o):
    """json-safe without rounding (drops per-target N arrays)."""
    if isinstance(o, dict): return {str(k): full(v) for k, v in o.items() if k != "N"}
    if isinstance(o, (list, tuple)): return [full(v) for v in o]
    if isinstance(o, np.ndarray): return full(o.tolist())
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        if not np.isfinite(o): return None if np.isnan(o) else ("inf" if o > 0 else "-inf")
        return float(o)
    return o

# ---------------------------------------------------------------- M1 analysis (verbatim except: Cantelli moved to cantelli_all) ----
def analyze(kind, IND, aliveT, rng, kmat):
    """kind 'presence' or 'births'. IND: dict T -> (H,H) bool indicator (lifter / birth). aliveT: (NT, H) bool. Phase 2 per-target stats, nulls."""
    res = {}; per = []; Mats = {}
    cols_all = np.arange(H)
    for i in range(H):
        Ts = [T for T in PH2 if aliveT[T - 1, i]]
        if len(Ts) == 0: per.append(None); continue
        M = np.stack([IND[T][i, :] for T in Ts], 0).astype(np.int8); M = M[:, cols_all != i]  # (n_alive, 99)
        Mats[i] = (Ts, M); st = ind_stats(M); st["i"] = i; st["Ts"] = Ts; per.append(st)
    ok = [st for st in per if st is not None and st["n"] >= MIN_ALIVE]
    res["n_targets_alive_any"] = len([s for s in per if s is not None]); res["n_targets_used"] = len(ok); res["min_alive_tasks"] = MIN_ALIVE
    res["n_alive_tasks_per_used_target"] = q([s["n"] for s in ok], (0.0, 0.1, 0.5, 0.9, 1.0))
    res["identity_max_abs_err"] = float(max(abs(s["delta"] - s["delta_id"]) for s in ok)) if ok else None
    mu = np.array([s["mu"] for s in ok]); var1 = np.array([s["var1"] for s in ok]); var0 = np.array([s["var0"] for s in ok]); fano = np.array([s["fano"] for s in ok]); mx = np.array([s["max"] for s in ok]); n_i = np.array([s["n"] for s in ok], float)
    delta = np.array([s["delta"] for s in ok]); v1mm = np.array([s["var1_minus_mu"] for s in ok]); sp2 = np.array([s["sp2"] for s in ok])
    res["per_target"] = dict(i=[s["i"] for s in ok], n_alive=[s["n"] for s in ok], mu=mu, var1=var1, var0=var0, fano=fano, max=mx, ge2=[s["ge2"] for s in ok], ge3=[s["ge3"] for s in ok], delta=delta, var1_minus_mu=v1mm, sum_p2=sp2)
    res["fano_quantiles"] = q(fano); res["mu_quantiles"] = q(mu); res["share_fano_gt1"] = float(np.nanmean(fano > 1)) if np.isfinite(fano).any() else None
    res["n_targets_mu0"] = int((mu == 0).sum())
    res["delta_quantiles"] = q(delta); res["share_delta_gt0"] = float((delta > 0).mean()); res["sum_delta"] = float(delta.sum()); res["sum_var1_minus_mu"] = float(np.nansum(v1mm)); res["sum_sum_p2"] = float(sp2.sum())
    Nall = np.concatenate([s["N"] for s in ok]); ntot = len(Nall); lam = float(Nall.mean()); vmax = int(Nall.max())
    hist = np.bincount(Nall.astype(int), minlength=vmax + 1).astype(float)
    pois = np.array([ntot * pois_pmf(lam, v) for v in range(vmax + 1)]); pois_tail_ge = ntot * pois_tail(lam, vmax + 1)
    mix = np.array([sum(s["n"] * pois_pmf(s["mu"], v) for s in ok) for v in range(vmax + 1)])
    within_var = float((n_i * var0).sum() / n_i.sum()); between_var = float((n_i * (mu - lam) ** 2).sum() / n_i.sum())
    res["pooled"] = dict(n_target_tasks=ntot, mean=lam, var=float(Nall.var(ddof=1)), fano=float(Nall.var(ddof=1) / lam) if lam > 0 else None, max=vmax,
                         ge2=float((Nall >= 2).mean()), ge3=float((Nall >= 3).mean()), ge5=float((Nall >= 5).mean()), ge10=float((Nall >= 10).mean()),
                         n_ge2=int((Nall >= 2).sum()), n_ge3=int((Nall >= 3).sum()), n_ge5=int((Nall >= 5).sum()), n_ge10=int((Nall >= 10).sum()),
                         hist_obs=hist.tolist(), hist_poisson_same_mean=pois.tolist(), poisson_tail_beyond_max=float(pois_tail_ge), hist_mixture_per_target_poisson=mix.tolist(),
                         poisson_ge2=float(pois_tail(lam, 2)), poisson_ge3=float(pois_tail(lam, 3)), poisson_ge5=float(pois_tail(lam, 5)),
                         mixture_ge2=float(sum(s["n"] * pois_tail(s["mu"], 2) for s in ok) / ntot), mixture_ge3=float(sum(s["n"] * pois_tail(s["mu"], 3) for s in ok) / ntot), mixture_ge5=float(sum(s["n"] * pois_tail(s["mu"], 5) for s in ok) / ntot),
                         var_within_targets=within_var, var_between_targets=between_var, fano_within=(within_var / lam if lam > 0 else None), fano_between=(between_var / lam if lam > 0 else None))
    nulls = {}
    NSTRATA = 4
    for mode in ("shift", "perm", "strat"):
        arr = np.full((len(ok), R_NULL, 5), nan); pf = []; pm = []; pd = []
        for a_idx, s in enumerate(ok):
            Ts, M = Mats[s["i"]]
            if mode == "strat":
                kk = np.array([kmat[T - 1, s["i"]] for T in Ts], float); edges = np.quantile(kk, np.linspace(0, 1, NSTRATA + 1)[1:-1]); null_cols.strata = np.searchsorted(edges, kk, side="right")
            out = null_cols(M, R_NULL, rng, mode); arr[a_idx] = out
            pf.append(pval2(s["fano"], out[:, 0])); pm.append(pval2(s["max"], out[:, 1])); pd.append(pval2(s["delta"], out[:, 4]))
        pf = np.array(pf); pm = np.array(pm); pd = np.array(pd)
        medF = np.nanmedian(arr[:, :, 0], axis=0); meanF = np.nanmean(arr[:, :, 0], axis=0); maxN = arr[:, :, 1].max(axis=0)
        ge2 = (arr[:, :, 2] * n_i[:, None]).sum(0) / n_i.sum(); ge3 = (arr[:, :, 3] * n_i[:, None]).sum(0) / n_i.sum(); sdel = arr[:, :, 4].sum(0)
        obs = dict(median_fano=float(np.nanmedian(fano)), mean_fano=float(np.nanmean(fano)), max_N=int(mx.max()), pooled_ge2=res["pooled"]["ge2"], pooled_ge3=res["pooled"]["ge3"], sum_delta=float(delta.sum()))
        nulls[mode] = dict(R=R_NULL, observed=obs,
                           null=dict(median_fano=q(medF, (0.025, 0.5, 0.975)), mean_fano=q(meanF, (0.025, 0.5, 0.975)), max_N=q(maxN, (0.025, 0.5, 0.975)), pooled_ge2=q(ge2, (0.025, 0.5, 0.975)), pooled_ge3=q(ge3, (0.025, 0.5, 0.975)), sum_delta=q(sdel, (0.025, 0.5, 0.975))),
                           p_two_sided=dict(median_fano=pval2(obs["median_fano"], medF), mean_fano=pval2(obs["mean_fano"], meanF), max_N=pval2(obs["max_N"], maxN), pooled_ge2=pval2(obs["pooled_ge2"], ge2), pooled_ge3=pval2(obs["pooled_ge3"], ge3), sum_delta=pval2(obs["sum_delta"], sdel)),
                           per_target=dict(p_fano=pf, p_max=pm, p_delta=pd, n_p_fano_lt05=int(np.nansum(pf < 0.05)), n_p_max_lt05=int(np.nansum(pm < 0.05)), n_p_delta_lt05=int(np.nansum(pd < 0.05)),
                                           n_fano_above_null_median=int(np.nansum(fano > np.nanmedian(arr[:, :, 0], axis=1))), n_fano_tested=int(np.isfinite(pf).sum()),
                                           null_fano_q025_med=q(np.nanquantile(arr[:, :, 0], 0.025, axis=1), (0.5,)), null_fano_q975_med=q(np.nanquantile(arr[:, :, 0], 0.975, axis=1), (0.5,)),
                                           n_max_above_null_q975=int((mx > np.quantile(arr[:, :, 1], 0.975, axis=1)).sum())))
    nulls["strat"]["note"] = f"independent permutation of each candidate's series within {NSTRATA} quantile strata of the target's width k_i(T) over its alive Phase-2 tasks (keeps each candidate's rate per stratum; destroys same-task pairing within strata)"
    res["null_candidate_series"] = nulls
    kb_edges = [0, 3, 8, 30, 100, 300, 10 ** 9]; kb_lab = ["1-3", "4-8", "9-30", "31-100", "101-300", ">300"]; kt = {}
    kall = np.concatenate([[kmat[T - 1, s["i"]] for T in s["Ts"]] for s in ok]).astype(float)
    for lo, hi, lab in zip(kb_edges[:-1], kb_edges[1:], kb_lab):
        sel = (kall > lo) & (kall <= hi)
        if sel.sum() == 0: continue
        v = Nall[sel]; m_ = v.mean()
        kt[lab] = dict(n_target_tasks=int(sel.sum()), mean=float(m_), fano=float(v.var(ddof=1) / m_) if m_ > 0 and len(v) > 1 else None, ge2=float((v >= 2).mean()), ge3=float((v >= 3).mean()), max=int(v.max()), poisson_ge2=float(pois_tail(m_, 2)), poisson_ge3=float(pois_tail(m_, 3)))
    res["by_target_width"] = kt
    Nmat = np.full((len(PH2), H), -1.0)
    for st in per:
        if st is None: continue
        for Tt, v in zip(st["Ts"], st["N"]): Nmat[Tt - PH2[0], st["i"]] = v
    used_idx = np.array([s["i"] for s in ok]); fanoN = np.full((R_NULL, len(ok)), nan); maxN2 = np.zeros((R_NULL, len(ok)))
    for r in range(R_NULL):
        Nr = Nmat.copy()
        for row in range(len(PH2)):
            al = np.where(Nr[row] >= 0)[0]; Nr[row, al] = Nr[row, al[rng.permutation(len(al))]]
        for a_idx, i in enumerate(used_idx):
            v = Nr[:, i]; v = v[v >= 0]; m_ = v.mean(); fanoN[r, a_idx] = v.var(ddof=1) / m_ if m_ > 0 else nan; maxN2[r, a_idx] = v.max()
    pf2 = np.array([pval2(fano[a], fanoN[:, a]) for a in range(len(ok))]); pm2 = np.array([pval2(mx[a], maxN2[:, a]) for a in range(len(ok))])
    medF2 = np.nanmedian(fanoN, axis=1); sdF2 = np.nanstd(fanoN, axis=1); obs_sd = float(np.nanstd(fano))
    res["null_target_shuffle"] = dict(R=R_NULL, observed=dict(median_fano=float(np.nanmedian(fano)), sd_fano_across_targets=obs_sd, sd_mu_across_targets=float(mu.std()), max_of_per_target_max=int(mx.max())),
                                      null=dict(median_fano=q(medF2, (0.025, 0.5, 0.975)), sd_fano_across_targets=q(sdF2, (0.025, 0.5, 0.975))),
                                      p_two_sided=dict(median_fano=pval2(float(np.nanmedian(fano)), medF2), sd_fano_across_targets=pval2(obs_sd, sdF2)),
                                      per_target=dict(p_fano=pf2, p_max=pm2, n_p_fano_lt05=int(np.nansum(pf2 < 0.05)), n_p_max_lt05=int(np.nansum(pm2 < 0.05)), n_fano_tested=int(np.isfinite(pf2).sum())),
                                      note="")
    sdmu2 = np.zeros(R_NULL)
    for r in range(R_NULL):
        Nr = Nmat.copy()
        for row in range(len(PH2)):
            al = np.where(Nr[row] >= 0)[0]; Nr[row, al] = Nr[row, al[rng.permutation(len(al))]]
        mus = np.array([Nr[:, i][Nr[:, i] >= 0].mean() for i in used_idx]); sdmu2[r] = mus.std()
    res["null_target_shuffle"]["null"]["sd_mu_across_targets"] = q(sdmu2, (0.025, 0.5, 0.975)); res["null_target_shuffle"]["p_two_sided"]["sd_mu_across_targets"] = pval2(float(mu.std()), sdmu2)
    res["null_target_shuffle"]["note"] = "mu_i is not invariant under this null; sd(mu_i) across targets compares target heterogeneity with the cross-sectional draw. With few alive targets per task (small a) this null is coarse."
    win = {}
    for (lo, hi) in WINDOWS:
        rows = []
        for s in ok:
            sel = np.array([(lo <= T <= hi) for T in s["Ts"]])
            if sel.sum() < MIN_ALIVE_WIN: continue
            v = s["N"][sel]; m_ = v.mean(); rows.append((m_, v.var(ddof=1), v.var(ddof=1) / m_ if m_ > 0 else nan, v.max(), v))
        if not rows: continue
        allv = np.concatenate([r[4] for r in rows]); mus = np.array([r[0] for r in rows]); fs = np.array([r[2] for r in rows])
        win[f"T{lo}-{hi}"] = dict(n_targets=len(rows), mean_mu=float(mus.mean()), median_fano=float(np.nanmedian(fs)), mean_fano=float(np.nanmean(fs)), share_fano_gt1=float(np.nanmean(fs > 1)),
                                  pooled_mean=float(allv.mean()), pooled_fano=float(allv.var(ddof=1) / allv.mean()) if allv.mean() > 0 else None, pooled_ge2=float((allv >= 2).mean()), pooled_ge3=float((allv >= 3).mean()), max=int(allv.max()), n_target_tasks=int(len(allv)))
    res["windows"] = win
    return res, per, ok

# ---------------------------------------------------------------- Cantelli with per-target n_beta at several states + break-even n* ----
def load_nbeta(paths):
    """list of (T_M1, {i: n_b}) from p172c real jsons. The original files carry 'task' = t (state end0[t-2]) -> T = t - 1."""
    out = []
    for p in paths:
        jd = json.load(open(p)); T = int(jd["T_M1"]) if "T_M1" in jd else int(jd["task"]) - 1
        out.append((T, {int(r["i"]): float(r["n_b"]) for r in jd["real"]}, p))
    return sorted(out, key=lambda x: x[0])

def cantelli_all(ok, NB):
    """ok: per-target stats (mu, var1 ddof=1, N over the target's alive Phase-2 tasks Ts). NB: list of (T, {i: n_b}, path).
    Rules: 'nearest' = n_beta of the state nearest to each task T (ties -> earlier state; a target without a value there takes the other state);
    'min' = per-target minimum over states; 'T<s>' = that state alone (missing -> that state's median); constants; and uniform F-bar versions."""
    res = {}
    mu = np.array([s["mu"] for s in ok]); var = np.array([s["var1"] for s in ok]); ids = [s["i"] for s in ok]
    allv = np.array([v for _, d, _ in NB for v in d.values() if np.isfinite(v)]); fill_all = float(np.median(allv)) if len(allv) else nan
    res["states"] = [dict(T=T, path=p, n_values=len(d), n_finite=int(np.isfinite(list(d.values())).sum()), n_inf=int(np.isinf(list(d.values())).sum()), n_zero=int((np.array(list(d.values())) <= 0).sum()),
                          min=float(np.nanmin([v for v in d.values() if np.isfinite(v)])), median=float(np.median([v for v in d.values() if np.isfinite(v)])), q10=float(np.quantile([v for v in d.values() if np.isfinite(v)], 0.1))) for T, d, p in NB]
    res["fill_value_all_states_median"] = fill_all
    def nstar_nearest(i, T):
        for Ts, d, _ in sorted(NB, key=lambda s: (abs(T - s[0]), s[0])):
            if i in d: return d[i], 0
        return fill_all, 1
    def nstar_min(i):
        v = [d[i] for _, d, _ in NB if i in d]
        return (min(v), 0) if v else (fill_all, 1)
    def block(nstar_tasks, filled, label, var_use=None):
        """nstar_tasks: list (per target) of arrays over its alive tasks."""
        vv = var if var_use is None else var_use
        b_tasks = [np.array([cantelli(mu[a], vv[a], n) for n in nstar_tasks[a]]) for a in range(len(ok))]
        b_avg = np.array([bt.mean() for bt in b_tasks]); b_maxtask = np.array([bt.max() for bt in b_tasks])
        pz = np.array([np.mean([pois_tail(mu[a], n) for n in nstar_tasks[a]]) for a in range(len(ok))])
        emp_n = np.array([int((ok[a]["N"] >= nstar_tasks[a]).sum()) for a in range(len(ok))]); emp_f = emp_n / np.array([len(ok[a]["N"]) for a in range(len(ok))])
        amax = int(np.argmax(b_avg)); ns_all = np.concatenate(nstar_tasks); nmm = np.array([np.min(nstar_tasks[a]) - mu[a] for a in range(len(ok))])
        exc = [dict(i=int(ids[a]), T=int(ok[a]["Ts"][t]), N=int(ok[a]["N"][t]), nstar=float(nstar_tasks[a][t])) for a in range(len(ok)) for t in np.where(ok[a]["N"] >= nstar_tasks[a])[0]][:60]
        return dict(label=label, n_targets=len(ok), n_targets_filled=int(filled), nstar=q(ns_all, (0.0, 0.1, 0.5, 0.9)), nstar_minus_mu_min=float(nmm.min()), nstar_minus_mu_argmin_target=int(ids[int(np.argmin(nmm))]),
                    cantelli=dict(max=float(b_avg.max()), argmax_target=int(ids[amax]), argmax_mu=float(mu[amax]), argmax_var=float(vv[amax]), argmax_nstar_min=float(np.min(nstar_tasks[amax])),
                                  median=float(np.median(b_avg)), q90=float(np.quantile(b_avg, .9)), mean=float(b_avg.mean()), sum_targets=float(b_avg.sum()), max_over_target_tasks=float(b_maxtask.max()),
                                  n_targets_gt_014=int((b_avg > 0.14).sum()), n_targets_gt_008=int((b_avg > 0.08).sum())),
                    poisson_indep=dict(max=float(pz.max()), median=float(np.median(pz))), empirical=dict(n_target_tasks_ge_nstar=int(emp_n.sum()), max_freq=float(emp_f.max()), n_targets_with_any=int((emp_n > 0).sum()), exceedances=exc),
                    per_target=dict(i=ids, bound_time_avg=b_avg, nstar_min_over_tasks=[float(np.min(x)) for x in nstar_tasks], n_ge_nstar=emp_n))
    # nearest
    ns_near = []; fill_near = 0
    for s in ok:
        arr = []; fl = 0
        for T in s["Ts"]:
            v, f = nstar_nearest(s["i"], T); arr.append(v); fl = max(fl, f)
        ns_near.append(np.array(arr)); fill_near += fl
    res["nearest"] = block(ns_near, fill_near, "per-target n_beta of the nearest state (ties to the earlier); fallback other state; none -> all-state median")
    ns_min = []; fill_min = 0
    for s in ok:
        v, f = nstar_min(s["i"]); ns_min.append(np.full(len(s["Ts"]), v)); fill_min += f
    res["min"] = block(ns_min, fill_min, "per-target minimum of n_beta over the states")
    gmin = float(np.min(allv)); have_any = np.array([any(s["i"] in d for _, d, _ in NB) for s in ok])
    ns_min_c = [x if have_any[a] else np.full(len(x), gmin) for a, x in enumerate(ns_min)]
    res["min_fill_armmin"] = block(ns_min_c, int((~have_any).sum()), f"per-target minimum; targets without any n_beta value (dead at every n_beta state) -> the minimum over all states ({gmin:.2f}) [conservative]")
    res["targets_without_nbeta"] = [dict(i=int(s["i"]), n_alive=int(s["n"]), mu=float(s["mu"]), var=float(s["var1"]), max_N=int(s["max"])) for a, s in enumerate(ok) if not have_any[a]]
    for T, d, _ in NB:
        med = float(np.median([v for v in d.values() if np.isfinite(v)])); arrs = []; fl = 0
        for s in ok:
            if s["i"] in d: arrs.append(np.full(len(s["Ts"]), d[s["i"]]))
            else: arrs.append(np.full(len(s["Ts"]), med)); fl += 1
        res[f"T{T}_only"] = block(arrs, fl, f"n_beta of state T={T} only; targets without a value -> that state's median {med:.2f}")
        mn = float(np.min([v for v in d.values() if np.isfinite(v)]))
        res[f"T{T}_arm_min"] = block([np.full(len(s["Ts"]), mn) for s in ok], 0, f"constant n* = min n_beta at T={T} ({mn:.2f})")
    res["ten"] = block([np.full(len(s["Ts"]), 10.0) for s in ok], 0, "constant n* = 10")
    for F in F_UNIFORM:
        res[f"nearest_F{F:g}"] = block(ns_near, fill_near, f"nearest rule with uniform F-bar = {F:g} (var = F mu)", var_use=F * mu)
        res[f"min_F{F:g}"] = block(ns_min, fill_min, f"min rule with uniform F-bar = {F:g}", var_use=F * mu)
    res["min_F1"] = block(ns_min, fill_min, "min rule with F = 1 (var = mu; the independent-variance Cantelli)", var_use=1.0 * mu)
    # break-even n*: Cantelli bound sigma^2/(sigma^2 + (n* - mu)^2) = c  <=>  n* = mu + sigma sqrt((1-c)/c)
    be = {}; nmin = np.array([x[0] for x in ns_min]); sd = np.sqrt(var); hv = have_any & (mu > 0)   # comparisons only for targets with a real n_beta and mu > 0
    for c in C_LEVELS:
        f = math.sqrt((1 - c) / c); nbe = mu + sd * f; a = int(np.argmax(nbe))
        rec = dict(c=c, factor_sqrt=f, max=float(nbe.max()), argmax_target=int(ids[a]), argmax_mu=float(mu[a]), argmax_sd=float(sd[a]), argmax_fano=float(var[a] / mu[a]) if mu[a] > 0 else None,
                   median=float(np.median(nbe)), q90=float(np.quantile(nbe, .9)),
                   n_targets_compared=int(hv.sum()), n_targets_mu0=int((mu == 0).sum()), n_targets_without_nbeta=int((~have_any).sum()),
                   n_targets_nbeta_min_below_breakeven=int((nmin[hv] < nbe[hv]).sum()), min_margin_nbeta_min_minus_breakeven=float((nmin[hv] - nbe[hv]).min()), min_ratio_nbeta_min_over_breakeven=float((nmin[hv] / nbe[hv]).min()),
                   argmin_margin_target=int(np.array(ids)[hv][int(np.argmin(nmin[hv] - nbe[hv]))]), max_breakeven_targets_without_nbeta=(float(nbe[~have_any].max()) if (~have_any).any() else None))
        for F in F_UNIFORM:
            nbeF = mu + np.sqrt(F * mu) * f; rec[f"max_F{F:g}"] = float(nbeF.max())
        be[f"c{c:g}"] = rec
    res["breakeven"] = be
    res["breakeven_per_target"] = dict(i=ids, mu=mu, sd=sd, nstar_min=nmin, nbe_014=mu + sd * math.sqrt(0.86 / 0.14), nbe_008=mu + sd * math.sqrt(0.92 / 0.08))
    return res

# ---------------------------------------------------------------- M3 section-4 decomposition (copied from m3_hubs.py) ----
def avg_rank(x):
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x), float); sx = x[order]
    i = 0; n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]: j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
    return ranks

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3: return float("nan")
    rx = avg_rank(x[ok]); ry = avg_rank(y[ok])
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def var_decomp(vals, groups):
    """population variance of vals decomposed into across-group (variance of group means) and within-group parts (m3_hubs.py)"""
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    n = len(vals); mu = vals.mean(); tot = vals.var()
    ug, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv); sums = np.bincount(inv, weights=vals); means = sums / cnt
    across = float((cnt * (means - mu) ** 2).sum() / n); within = float(((vals - means[inv]) ** 2).sum() / n)
    return dict(n=int(n), mean=float(mu), var=float(tot), across=across, within=within, share_across=(across / tot if tot > 0 else None), share_within=(within / tot if tot > 0 else None),
                fano=(tot / mu if mu > 0 else None), fano_within=(within / mu if mu > 0 else None), n_groups=int(len(ug)))

def kbin_of(k, bins):
    for b, (lo, hi) in enumerate(bins):
        if lo <= k <= hi: return b
    return len(bins) - 1

def decomp(Tlist, ISB, ALIVE, KK, TIDX, THR, OALL, label):
    """M3 section 4 on the states Tlist (M1 convention). Null U (m3_hubs.py): each candidate's open set -> uniform random subset of size k_j; target's T_i, O_i actual."""
    rng = np.random.default_rng(SEED_U); Nn = OALL.shape[1]; recs = []; NU1 = []; CU1 = []
    for T in Tlist:
        O = OALL[T - 1]; alive = ALIVE[T - 1]; k = KK[T - 1].astype(np.int64); need = (k + 1) // 2; Tix = TIDX[T - 1]; thr = THR[T - 1]
        NU = np.zeros((R_MC_U, H)); CU = np.zeros((R_MC_U, H))
        for rep in range(R_MC_U):
            U = rng.random((Nn, H)); ranks = np.argsort(np.argsort(U, axis=0), axis=0); On = ranks < k[None, :]
            thr_n = np.stack([On[Tix[:, i], :].sum(0) for i in range(H)], 0); ov_n = O.astype(np.int32).T @ On.astype(np.int32)
            isb_n = (thr_n >= 2) & (ov_n >= need[:, None]); isb_n[np.diag_indices(H)] = False; isb_n[~alive, :] = False; isb_n[:, ~alive] = False
            NU[rep] = isb_n.sum(1)
            pool_n = (thr_n >= 2) & alive[None, :]; pool_n[np.diag_indices(H)] = False; CU[rep] = pool_n.sum(1)
        pool = (thr >= 2) & alive[None, :]; pool[np.diag_indices(H)] = False; Cobs = pool.sum(1); Nobs = ISB[T - 1].sum(1)
        for i in np.where(alive)[0]:
            recs.append((T, int(i), int(k[i]), int(Nobs[i]), int(Cobs[i]), float(NU[:, i].mean()), float(NU[:, i].var()), float(CU[:, i].mean()), float(CU[:, i].var())))
            NU1.append(NU[0, i]); CU1.append(CU[0, i])
    R = np.array(recs, float); tt = R[:, 0].astype(int); ii = R[:, 1].astype(int); kk = R[:, 2].astype(int); NN = R[:, 3]; CC = R[:, 4]; LU = R[:, 5]; LUv = R[:, 6]; CUm = R[:, 7]; CUv = R[:, 8]
    kbf = np.array([kbin_of(k, KBINS_FINE) for k in kk]); NU1 = np.array(NU1); CU1 = np.array(CU1)
    vd = dict(label=label, states=[int(Tlist[0]), int(Tlist[-1])], n_states=len(Tlist), R_MC_U=R_MC_U, seed_U=SEED_U,
              N_by_unit=var_decomp(NN, ii), N_by_kbin_fine=var_decomp(NN, kbf), N_by_unit_x_kbin_fine=var_decomp(NN, ii * 100 + kbf), N_by_task=var_decomp(NN, tt),
              N_by_pool=var_decomp(NN, np.minimum(CC, 30)), N_by_pool_x_kbin_fine=var_decomp(NN, kbf * 100 + np.minimum(CC, 30)),
              C_by_unit=var_decomp(CC, ii), C_by_kbin_fine=var_decomp(CC, kbf), C_by_task=var_decomp(CC, tt), CU_mean_by_kbin_fine=var_decomp(CUm, kbf),
              CU_rep0=var_decomp(CU1, ii), CU_rep0_by_kbin_fine=var_decomp(CU1, kbf), NU_rep0_by_unit=var_decomp(NU1, ii), NU_rep0_by_kbin_fine=var_decomp(NU1, kbf))
    vd["pool"] = dict(C_mean=float(CC.mean()), C_var=float(CC.var()), C_fano=float(CC.var() / CC.mean()) if CC.mean() > 0 else None, C_max=int(CC.max()),
                      CU_mean=float(CUm.mean()), CU_hetero_var=float(CUm.var()), CU_within_var=float(CUv.mean()), CU_total_fano=float((CUm.var() + CUv.mean()) / CUm.mean()) if CUm.mean() > 0 else None,
                      CU_fano_within_kbin=float((vd["CU_mean_by_kbin_fine"]["within"] + CUv.mean()) / CUm.mean()) if CUm.mean() > 0 else None,
                      ratio_C_CU=float(CC.sum() / CUm.sum()) if CUm.sum() > 0 else None, pass_rate_N_over_C=float(NN.sum() / CC.sum()) if CC.sum() > 0 else None, sp_N_C=spearman(NN, CC), sp_C_k=spearman(CC, kk))
    vd["nullU_N"] = dict(lamU_mean=float(LU.mean()), lamU_hetero_var=float(LU.var()), lamU_within_state_var_mean=float(LUv.mean()), NU_total_fano=float((LU.var() + LUv.mean()) / LU.mean()) if LU.mean() > 0 else None,
                         ratio_obs_U_total=float(NN.sum() / LU.sum()) if LU.sum() > 0 else None)
    vd["arrays"] = dict(T=tt, i=ii, k=kk, N=NN, C=CC, lamU=LU, CU=CUm)
    return vd

# ---------------------------------------------------------------- states and reproduction ----
def compute_states(dd, a, Xt, Kv):
    Nn = Xt.shape[0]
    ISB = np.zeros((NT, H, H), bool); ALIVE = np.zeros((NT, H), bool); KK = np.zeros((NT, H), np.int16); TIDX = np.zeros((NT, 3, H), np.int16); THR = np.zeros((NT, H, H), np.int8); OALL = np.zeros((NT, Nn, H), bool)
    for T in range(1, NT + 1):
        W = np.concatenate([dd["end0"][T - 1].astype(np.float64), dd["end1"][T - 1].astype(np.float64)[:, None]], 1); Z = Xt @ W.T; tau = 0.1 * Z.std(0) + 1e-12
        s = 0.5 * (1 + np.tanh(0.5 * Z / tau)); O = Z > 0; Dp = (1 - a) * s * (1 - s) / tau
        isb, alive, k = bridges_at(O, Dp, Kv); Tix, thr = thresholds_and_counts(O, Dp, Kv)
        assert not (isb & (thr < 2)).any()
        ISB[T - 1] = isb; ALIVE[T - 1] = alive; KK[T - 1] = k; TIDX[T - 1] = Tix; THR[T - 1] = thr; OALL[T - 1] = O
    return ISB, ALIVE, KK, TIDX, THR, OALL

def reproduction(ISB, ALIVE, base):
    BIRTH = np.zeros_like(ISB); BIRTH[1:] = ISB[1:] & ~ISB[:-1]
    rows_stoch = int(sum(ISB[T - 1].sum() for T in range(9, 100)))                       # stoch rows: t = 10..100 -> T = 9..99
    births_stoch = int(sum(BIRTH[T - 1][ALIVE[T - 1]].sum() for T in range(10, 100)))     # stoch births t = 11..100 -> T = 10..99
    recs = []
    for T in range(9, 99):                                                                  # stoch (t,i) with n_t >= 1, t = 10..99 -> T = 9..98
        for i in np.where(ALIVE[T - 1])[0]:
            n_t = int(ISB[T - 1][i].sum())
            if n_t >= 1:
                S = int((ISB[T - 1][i] & ISB[T][i]).sum()); n1 = int(ISB[T][i].sum()); recs.append((n_t, S, n1 - S, n1))
    rec = np.array(recs) if recs else np.zeros((1, 4))
    out = dict(stoch=dict(rows_t10_100=rows_stoch, births_t11_100=births_stoch, target_tasks_nt_ge1=len(recs), nt_mean=float(rec[:, 0].mean()), nt_max=int(rec[:, 0].max()), S_mean=float(rec[:, 1].mean()), S_max=int(rec[:, 1].max()),
                          B_mean=float(rec[:, 2].mean()), B_max=int(rec[:, 2].max()), n1_mean=float(rec[:, 3].mean()), n1_max=int(rec[:, 3].max())))
    # P159 counting at a reference state and along training (t = 2/10/20/40/100 -> T = 1/9/19/39/99)
    def cnt(T): c = ISB[T - 1].sum(1)[ALIVE[T - 1]]; return dict(T=T, n=int(ALIVE[T - 1].sum()), max=int(c.max()) if len(c) else 0, ge3=int((c >= 3).sum()), ge10=int((c >= 10).sum()))
    out["p159_ref_T39"] = cnt(39); out["p159_ref_T99"] = cnt(99); out["p159_along_max"] = [cnt(T)["max"] for T in (1, 9, 19, 39, 99)]
    # P161 turnover over t = 10..60 -> T = 9..59
    Bset = {T: set(zip(*np.where(ISB[T - 1]))) for T in range(9, 60)}
    persist = [len(Bset[T] & Bset[T + 1]) / max(len(Bset[T]), 1) for T in range(9, 59)]
    ap = {}
    for T in range(9, 60):
        for pr in Bset[T]: ap.setdefault(pr, []).append(T)
    lifetimes = []
    for pr, ts in ap.items():
        ts = sorted(ts); run = 1
        for kx in range(1, len(ts)):
            if ts[kx] == ts[kx - 1] + 1: run += 1
            else: lifetimes.append(run); run = 1
        lifetimes.append(run)
    out["p161"] = dict(count_median=float(np.median([len(Bset[T]) for T in Bset])), persist=float(np.median(persist)), life1=float(np.mean(np.array(lifetimes) == 1)) if lifetimes else None, n_runs=len(lifetimes))
    # phase-2 (T 11..100) counts in M1 convention
    Nc = np.concatenate([ISB[T - 1].sum(1)[ALIVE[T - 1]] for T in PH2]) if len(PH2) else np.zeros(1)
    out["phase2_M1"] = dict(rows=int(sum(ISB[T - 1].sum() for T in PH2)), births=int(sum(BIRTH[T - 1][ALIVE[T - 1]].sum() for T in PH2)), alive_target_tasks=int(len(Nc)), N_max=int(Nc.max()), N_ge3=int((Nc >= 3).sum()), N_ge10=int((Nc >= 10).sum()))
    pub = PUB.get(base); checks = {}
    if pub:
        if "stoch" in pub:
            s = out["stoch"]; p = pub["stoch"]
            checks["stoch"] = bool(s["rows_t10_100"] == p["rows"] and s["births_t11_100"] == p["births"] and s["target_tasks_nt_ge1"] == p["tt"] and s["nt_max"] == p["nt_max"] and s["B_max"] == p["B_max"] and abs(s["B_mean"] - p["B_mean"]) < 6e-4 and abs(s["nt_mean"] - p["nt_mean"]) < 6e-3)
        if "p159_ref" in pub:
            p = pub["p159_ref"]; s = out[f"p159_ref_T{p['T']}"]; checks["p159_ref"] = bool(s["max"] == p["max"] and s["ge3"] == p["ge3"] and s["ge10"] == p["ge10"] and s["n"] == p["n"])
            checks["p159_along"] = bool(out["p159_along_max"] == pub["p159_along"])
        if "p161" in pub:
            p = pub["p161"]; s = out["p161"]; checks["p161"] = bool(round(s["life1"], 2) == p["life1"] and round(s["persist"], 2) == p["persist"] and round(s["count_median"]) == p["count_median"])
    out["published"] = pub; out["match"] = checks
    return out, BIRTH

def run(args, anchor_only=False):
    global PH2, R_NULL, _LOG
    R_NULL = args.rnull; base = os.path.basename(args.npz); src = args.npz if os.path.isabs(args.npz) else os.path.join(ROOT, args.npz); a = float(args.a); tag = args.tag
    os.makedirs(RAW, exist_ok=True); _LOG = open(os.path.join(RAW, f"slopes_{tag}{'_anchor' if anchor_only else ''}.log"), "w")
    t0 = time.time(); dd = np.load(src)
    assert abs(float(dd["slope"]) - a) < 1e-12 if "slope" in dd.files else True, f"slope in npz {dd['slope']} != --a {a}"
    X = load_X(dd["subset"]); Nn = X.shape[0]; Xt = np.concatenate([X, np.ones((Nn, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar
    ISB, ALIVE, KK, TIDX, THR, OALL = compute_states(dd, a, Xt, Kv)
    P(f"[{tag}] npz {src} a={a} seed {int(dd['seed']) if 'seed' in dd.files else '?'} | states + bridges done ({time.time() - t0:.1f}s) | alive units per T: T1 {ALIVE[0].sum()} T11 {ALIVE[10].sum()} T39 {ALIVE[38].sum()} T50 {ALIVE[49].sum()} T99 {ALIVE[98].sum()} T100 {ALIVE[99].sum()}")
    repro, BIRTH = reproduction(ISB, ALIVE, base)
    s = repro["stoch"]
    P(f"[{tag}] REPRO stoch convention (t = T+1): rows t10..100 {s['rows_t10_100']} | births t11..100 {s['births_t11_100']} | target-tasks n_t>=1 {s['target_tasks_nt_ge1']} | n_t mean {s['nt_mean']:.3f} max {s['nt_max']} | B mean {s['B_mean']:.3f} max {s['B_max']} | n1 mean {s['n1_mean']:.3f} max {s['n1_max']}")
    P(f"[{tag}] REPRO P159 counting: ref T39 (t40) {repro['p159_ref_T39']} | ref T99 (t100) {repro['p159_ref_T99']} | along max at T 1/9/19/39/99 (t 2/10/20/40/100) {repro['p159_along_max']}")
    P(f"[{tag}] REPRO P161 turnover T 9..59 (t 10..60): bridges/task median {repro['p161']['count_median']:.0f} persist median {repro['p161']['persist']:.2f} lifetime-1 share {repro['p161']['life1']} (runs {repro['p161']['n_runs']})")
    P(f"[{tag}] phase 2 (M1, T 11..100): rows {repro['phase2_M1']['rows']} births {repro['phase2_M1']['births']} alive target-tasks {repro['phase2_M1']['alive_target_tasks']} N max {repro['phase2_M1']['N_max']} N>=3 {repro['phase2_M1']['N_ge3']} N>=10 {repro['phase2_M1']['N_ge10']}")
    P(f"[{tag}] published: {repro['published']} | MATCH: {repro['match']}")
    ledger = dict(T=list(range(1, NT + 1)), n_alive=ALIVE.sum(1).tolist(), pairs=[int(ISB[T - 1].sum()) for T in range(1, NT + 1)], births=[int(BIRTH[T - 1].sum()) if T > 1 else None for T in range(1, NT + 1)],
                  max_N=[int(ISB[T - 1].sum(1)[ALIVE[T - 1]].max()) if ALIVE[T - 1].any() else 0 for T in range(1, NT + 1)], k_median_alive=[float(np.median(KK[T - 1][ALIVE[T - 1]])) if ALIVE[T - 1].any() else None for T in range(1, NT + 1)])
    al2 = np.array([ALIVE[T - 1].sum() for T in PH2]); ever = ALIVE[np.array(PH2) - 1].any(0); nal = ALIVE[np.array(PH2) - 1].sum(0)
    alive_summary = dict(phase2_alive_per_task=q(al2, (0.0, 0.1, 0.5, 0.9, 1.0)), units_alive_at_some_phase2_task=int(ever.sum()), units_alive_ge30_tasks=int((nal >= MIN_ALIVE).sum()), units_alive_all_90=int((nal == len(PH2)).sum()),
                         alive_tasks_per_unit=q(nal[ever], (0.0, 0.1, 0.5, 0.9, 1.0)),
                         revivals=int(sum(((~ALIVE[T - 2]) & ALIVE[T - 1]).sum() for T in PH2)), deaths=int(sum((ALIVE[T - 2] & ~ALIVE[T - 1]).sum() for T in PH2)))
    P(f"[{tag}] alive units in phase 2: per task min/q10/med/q90/max {[round(v) for v in (alive_summary['phase2_alive_per_task'][k] for k in ('0.0', '0.1', '0.5', '0.9', '1.0'))]} | units alive at some phase-2 task {alive_summary['units_alive_at_some_phase2_task']}, >= {MIN_ALIVE} tasks {alive_summary['units_alive_ge30_tasks']}, all 90 {alive_summary['units_alive_all_90']} | deaths {alive_summary['deaths']} revivals {alive_summary['revivals']} (unit-transitions within T 10->11..99->100)")
    OUT = dict(tag=tag, npz=src, a=a, seed=int(dd["seed"]) if "seed" in dd.files else None, convention="M1: T = 1..100 is the state end0[T-1]; Phase 2 = T 11..100. stoch/p159/p161/p172c use t = T+1.",
               criterion="verbatim bridges_at (stoch_bridges.py via m1_counts.py)", repro=repro, ledger=ledger, alive=alive_summary)
    decs = {}
    if args.decomp_m3:
        vd = decomp(list(range(9, 101)), ISB, ALIVE, KK, TIDX, THR, OALL, "M3 states t = 10..101 = T 9..100 (92)"); decs["m3_states"] = vd
        pub = (PUB.get(base) or {}).get("m3")
        line = (f"[{tag}] M3-states decomposition (T 9..100): n {vd['N_by_unit']['n']} | N Fano {vd['N_by_unit']['fano']:.2f} unit share {vd['N_by_unit']['share_across']:.3f} unit-within Fano {vd['N_by_unit']['fano_within']:.2f} | C-within Fano {vd['N_by_pool']['fano_within']:.2f} "
                f"(C,k)-within Fano {vd['N_by_pool_x_kbin_fine']['fano_within']:.2f} | C Fano {vd['pool']['C_fano']:.2f} null-U {vd['pool']['CU_total_fano']:.2f}")
        if pub:
            m = dict(n=vd["N_by_unit"]["n"] == pub["n"], N_unit_share=round(vd["N_by_unit"]["share_across"], 3) == pub["N_unit_share"], N_unit_fano_within=round(vd["N_by_unit"]["fano_within"], 2) == pub["N_unit_fano_within"],
                     N_Ck_fano_within=round(vd["N_by_pool_x_kbin_fine"]["fano_within"], 2) == pub["N_Ck_fano_within"], N_C_fano_within=round(vd["N_by_pool"]["fano_within"], 2) == pub["N_C_fano_within"],
                     C_fano=round(vd["pool"]["C_fano"], 2) == pub["C_fano"], CU_total_fano=round(vd["pool"]["CU_total_fano"], 2) == pub["CU_total_fano"])
            vd["published"] = pub; vd["match"] = m; line += f" | published {pub} | MATCH {m}"
        P(line)
    if anchor_only:
        OUT["decomposition"] = {k: {kk: vv for kk, vv in v.items() if kk != "arrays"} for k, v in decs.items()}
        json.dump(full(OUT), open(os.path.join(RAW, f"slopes_{tag}_anchor_full.json"), "w"), indent=1); P(f"[{tag}] anchor written ({time.time() - t0:.1f}s)"); return
    # ---- M1 analysis ----
    NB = load_nbeta(args.nbeta) if args.nbeta else []
    rng = np.random.default_rng(SEED)
    IND_P = {T: ISB[T - 1] for T in range(1, NT + 1)}; IND_B = {T: BIRTH[T - 1] for T in range(1, NT + 1)}
    resP, perP, okP = analyze("presence", IND_P, ALIVE, rng, KK); P(f"[{tag}] presence analysis done ({time.time() - t0:.1f}s)")
    resB, perB, okB = analyze("births", IND_B, ALIVE, rng, KK); P(f"[{tag}] births analysis done ({time.time() - t0:.1f}s)")
    for nm, r in (("presence", resP), ("births", resB)):
        po = r["pooled"]; fq = r["fano_quantiles"]; n1 = r["null_candidate_series"]["shift"]; n2 = r["null_target_shuffle"]; n3 = r["null_candidate_series"]["strat"]
        P(f"[{tag}] {nm}: targets used {r['n_targets_used']} (alive any {r['n_targets_alive_any']}, mu=0 {r['n_targets_mu0']}) | pooled mean {po['mean']:.4f} Fano {po['fano']:.3f} (within {po['fano_within']:.3f} between {po['fano_between']:.3f}) max {po['max']} | P>=2 {po['ge2']:.4f} (Pois {po['poisson_ge2']:.4f}, mix {po['mixture_ge2']:.4f}) P>=3 {po['ge3']:.5f} (Pois {po['poisson_ge3']:.5f}, mix {po['mixture_ge3']:.5f}) P>=5 {po['ge5']:.5f} ({po['n_ge5']}) N>=10 {po['n_ge10']}")
        P(f"[{tag}] {nm}: per-target Fano q10 {fq['0.1']:.3f} q50 {fq['0.5']:.3f} q90 {fq['0.9']:.3f} max {fq['max']:.3f} share>1 {r['share_fano_gt1']:.3f} | mu q10/q50/q90 {r['mu_quantiles']['0.1']:.3f}/{r['mu_quantiles']['0.5']:.3f}/{r['mu_quantiles']['0.9']:.3f} | sum delta {r['sum_delta']:.3f} share delta>0 {r['share_delta_gt0']:.3f} | identity err {r['identity_max_abs_err']:.1e}")
        P(f"[{tag}] {nm}: null1 shift: median Fano obs {n1['observed']['median_fano']:.3f} null [{n1['null']['median_fano']['0.025']:.3f},{n1['null']['median_fano']['0.975']:.3f}] p {n1['p_two_sided']['median_fano']:.3f} | max N obs {n1['observed']['max_N']} null [{n1['null']['max_N']['0.025']:.0f},{n1['null']['max_N']['0.975']:.0f}] p {n1['p_two_sided']['max_N']:.3f} | pooled P>=3 null [{n1['null']['pooled_ge3']['0.025']:.4f},{n1['null']['pooled_ge3']['0.975']:.4f}] p {n1['p_two_sided']['pooled_ge3']:.3f} | sum delta null [{n1['null']['sum_delta']['0.025']:.2f},{n1['null']['sum_delta']['0.975']:.2f}] | per-target p_fano<.05 {n1['per_target']['n_p_fano_lt05']}/{n1['per_target']['n_fano_tested']}")
        P(f"[{tag}] {nm}: null1b strat(k_i): median Fano null [{n3['null']['median_fano']['0.025']:.3f},{n3['null']['median_fano']['0.975']:.3f}] p {n3['p_two_sided']['median_fano']:.3f} | max N null [{n3['null']['max_N']['0.025']:.0f},{n3['null']['max_N']['0.975']:.0f}] p {n3['p_two_sided']['max_N']:.3f} | pooled P>=3 null [{n3['null']['pooled_ge3']['0.025']:.4f},{n3['null']['pooled_ge3']['0.975']:.4f}] p {n3['p_two_sided']['pooled_ge3']:.3f} | sum delta null [{n3['null']['sum_delta']['0.025']:.2f},{n3['null']['sum_delta']['0.975']:.2f}] p {n3['p_two_sided']['sum_delta']:.3f} | per-target p_fano<.05 {n3['per_target']['n_p_fano_lt05']}")
        P(f"[{tag}] {nm}: null2 target-shuffle: median Fano null [{n2['null']['median_fano']['0.025']:.3f},{n2['null']['median_fano']['0.975']:.3f}] p {n2['p_two_sided']['median_fano']:.3f} | sd(mu_i) obs {n2['observed']['sd_mu_across_targets']:.4f} null [{n2['null']['sd_mu_across_targets']['0.025']:.4f},{n2['null']['sd_mu_across_targets']['0.975']:.4f}] p {n2['p_two_sided']['sd_mu_across_targets']:.3f}")
        for w, v in r["windows"].items():
            P(f"[{tag}] {nm}: window {w}: targets {v['n_targets']} mean mu {v['mean_mu']:.4f} median Fano {v['median_fano']:.3f} pooled Fano {v['pooled_fano']} P>=2 {v['pooled_ge2']:.4f} max {v['max']}")
        for lab, v in r["by_target_width"].items():
            P(f"[{tag}] {nm}: k_i in {lab}: target-tasks {v['n_target_tasks']} mean {v['mean']:.4f} Fano {v['fano'] if v['fano'] is None else round(v['fano'], 3)} P>=2 {v['ge2']:.4f} (Pois {v['poisson_ge2']:.4f}) P>=3 {v['ge3']:.5f} (Pois {v['poisson_ge3']:.5f}) max {v['max']}")
    cantP = cantelli_all(okP, NB) if NB else None; cantB = cantelli_all(okB, NB) if NB else None
    if cantP:
        for st in cantP["states"]: P(f"[{tag}] n_beta state T{st['T']}: values {st['n_values']} (inf {st['n_inf']}, <=0 {st['n_zero']}) min {st['min']:.2f} q10 {st['q10']:.2f} median {st['median']:.2f}")
        P(f"[{tag}] targets without any n_beta value (dead at all n_beta states): {cantP['targets_without_nbeta']}")
        for lab in ["nearest", "min", "min_fill_armmin"] + [k for k in cantP if k.endswith("_only") or k.endswith("_arm_min")] + ["ten", "min_F1", "nearest_F3", "min_F3", "nearest_F6.4", "min_F6.4"]:
            c = cantP[lab]; cc = c["cantelli"]
            P(f"[{tag}] Cantelli presence [{lab}] filled {c['n_targets_filled']}: n* min {c['nstar']['0.0']:.2f} med {c['nstar']['0.5']:.2f} | min(n*-mu) {c['nstar_minus_mu_min']:.2f} (u{c['nstar_minus_mu_argmin_target']}) | bound max {cc['max']:.3e} (u{cc['argmax_target']} mu {cc['argmax_mu']:.3f} var {cc['argmax_var']:.3f} n* {cc['argmax_nstar_min']:.2f}) median {cc['median']:.3e} sum {cc['sum_targets']:.3e} max over target-tasks {cc['max_over_target_tasks']:.3e} | >14% {cc['n_targets_gt_014']} >8% {cc['n_targets_gt_008']} | Poisson max {c['poisson_indep']['max']:.2e} | empirical N>=n* {c['empirical']['n_target_tasks_ge_nstar']} {c['empirical']['exceedances'][:6]}")
        for k, b in cantP["breakeven"].items():
            P(f"[{tag}] break-even n* (Cantelli = {b['c']:g}): max {b['max']:.2f} (u{b['argmax_target']} mu {b['argmax_mu']:.3f} sd {b['argmax_sd']:.3f} F {b['argmax_fano']}) median {b['median']:.2f} q90 {b['q90']:.2f} | F3 max {b['max_F3']:.2f} F6.4 max {b['max_F6.4']:.2f} | targets with n_beta_min < break-even {b['n_targets_nbeta_min_below_breakeven']}/{b['n_targets_compared']} (no n_beta {b['n_targets_without_nbeta']}, their max break-even {b['max_breakeven_targets_without_nbeta']}) | min margin n_beta_min - n*_be {b['min_margin_nbeta_min_minus_breakeven']:.2f} (u{b['argmin_margin_target']}) min ratio {b['min_ratio_nbeta_min_over_breakeven']:.2f}")
    # ---- M3 section-4 decomposition (M1 states) ----
    vd = decomp(PH2, ISB, ALIVE, KK, TIDX, THR, OALL, "M1 states T 11..100 (90)"); decs["m1_states"] = vd
    P(f"[{tag}] decomposition (T 11..100): n {vd['N_by_unit']['n']} mean {vd['N_by_unit']['mean']:.3f} Fano {vd['N_by_unit']['fano']:.2f} | unit share {vd['N_by_unit']['share_across']:.3f} (within Fano {vd['N_by_unit']['fano_within']:.2f}) | k-bin share {vd['N_by_kbin_fine']['share_across']:.3f} (within {vd['N_by_kbin_fine']['fano_within']:.2f}) | task share {vd['N_by_task']['share_across']:.3f} | C share {vd['N_by_pool']['share_across']:.3f} (within {vd['N_by_pool']['fano_within']:.2f}) | (C,k) share {vd['N_by_pool_x_kbin_fine']['share_across']:.3f} within Fano {vd['N_by_pool_x_kbin_fine']['fano_within']:.2f} | unit x k within {vd['N_by_unit_x_kbin_fine']['fano_within']:.2f}")
    pl = vd["pool"]; nu = vd["nullU_N"]
    P(f"[{tag}] pool C: mean {pl['C_mean']:.3f} Fano {pl['C_fano']} max {pl['C_max']} | null U: mean {pl['CU_mean']:.3f} total Fano {pl['CU_total_fano']} (within k-bin {pl['CU_fano_within_kbin']}) | C within k-bin Fano {vd['C_by_kbin_fine']['fano_within']} unit share {vd['C_by_unit']['share_across']} task share {vd['C_by_task']['share_across']} | C/CU {pl['ratio_C_CU']} pass N/C {pl['pass_rate_N_over_C']} | Spearman(N,C) {pl['sp_N_C']:+.3f} | lamU mean {nu['lamU_mean']:.4f} NU total Fano {nu['NU_total_fano']} obs/lamU {nu['ratio_obs_U_total']}")
    OUT.update(presence=resP, births=resB, cantelli_presence=cantP, cantelli_births=cantB, nbeta_files=[p for _, _, p in NB], R_null=R_NULL, seed_null=SEED, min_alive_tasks_per_target=MIN_ALIVE)
    OUT["decomposition"] = {k: {kk: vv for kk, vv in v.items() if kk != "arrays"} for k, v in decs.items()}; OUT["elapsed_s"] = time.time() - t0
    np.savez_compressed(os.path.join(RAW, f"counts_{tag}.npz"), N=np.where(ALIVE, ISB.sum(2), -1).astype(np.int16), B=np.where(ALIVE, BIRTH.sum(2), -1).astype(np.int16), alive=ALIVE, k=KK, isb=ISB, birth=BIRTH, thr_idx=TIDX,
                        C=np.where(ALIVE, (((THR >= 2) & ALIVE[:, None, :]) & ~np.eye(H, dtype=bool)[None]).sum(2), -1).astype(np.int16), T=np.arange(1, NT + 1),
                        dec_T=vd["arrays"]["T"], dec_i=vd["arrays"]["i"], dec_N=vd["arrays"]["N"], dec_C=vd["arrays"]["C"], dec_lamU=vd["arrays"]["lamU"], dec_CU=vd["arrays"]["CU"],
                        note=np.array(OUT["convention"] + " | arrays [T-1, i] ([T-1, i, j] for isb/birth: row target i, col candidate j); dead = -1; a=" + str(a)))
    json.dump(full(OUT), open(os.path.join(RAW, f"slopes_{tag}_full.json"), "w"), indent=1)
    P(f"[{tag}] wrote {RAW}/slopes_{tag}_full.json and counts_{tag}.npz ({time.time() - t0:.1f}s)")

def combine():
    ALL = dict(task="cobirth_seeds_1013 TASK B (tag 'slopes'): leaky slopes a = 0.03 and 0.2, seeds 0 and 1", convention="M1: T = 1..100 is the state end0[T-1]; Phase 2 = T 11..100. stoch/p159/p161/p172c use t = T+1.",
               script="analysis/cobirth_seeds_1013/slopes_counts.py", conditions={}, anchors={})
    for f in sorted(glob.glob(os.path.join(RAW, "slopes_*_full.json"))):
        d = json.load(open(f)); tag = d["tag"]
        if f.endswith("_anchor_full.json"):
            ALL["anchors"][tag] = dict(npz=d["npz"], a=d["a"], seed=d["seed"], repro=d["repro"], decomposition=d.get("decomposition"))
        else:
            for kind in ("presence", "births"):
                pt = d[kind].get("per_target", {})
                for key in ("p_fano", "p_max", "p_delta"):
                    for nm in ("shift", "perm", "strat"): d[kind]["null_candidate_series"][nm]["per_target"].pop(key, None)
                d[kind]["null_target_shuffle"]["per_target"].pop("p_fano", None); d[kind]["null_target_shuffle"]["per_target"].pop("p_max", None)
            for ck in ("cantelli_presence", "cantelli_births"):
                if d.get(ck):
                    for k, v in d[ck].items():
                        if isinstance(v, dict): v.pop("per_target", None)
            if d.get("cantelli_births"): d["cantelli_births"] = {k: d["cantelli_births"][k] for k in ("nearest", "min") if k in d["cantelli_births"]}
            d["births"].pop("per_target", None)
            ALL["conditions"][tag] = d
    for f in sorted(glob.glob(os.path.join(RAW, "p172c", "p172c_*_real.json"))):
        jd = json.load(open(f)); ALL.setdefault("p172c", {})[os.path.basename(f)] = dict(T_M1=jd.get("T_M1"), npz=jd.get("npz"), a=jd["a"], alive=jd["alive"], lam=jd["lam"],
                                                                                      n_b={str(r["i"]): r["n_b"] for r in jd["real"]}, A_le0=sum(1 for r in jd["real"] if r["A"] <= 0))
    an = os.path.join(RAW, "p172c_anchor.json")
    if os.path.exists(an): ALL["p172c_anchor"] = json.load(open(an))
    ex = os.path.join(RAW, "exceedance_check.json")   # p172c at the states of the cross-state exceedances + same-state N vs n_beta (written by the follow-up step in SLOPES_REPORT.md section 5)
    if os.path.exists(ex): ALL["exceedance_check"] = json.load(open(ex))
    # context: break-even n* for the published seed-0 arms a0.1/0.3/0.5 from cobirth_1013/m1_fano.json (same formula)
    m1 = os.path.join(WT, "results/cobirth_1013/m1_fano.json")
    if os.path.exists(m1):
        dm = json.load(open(m1)); ctx = {}
        for arm, r in dm["arms"].items():
            pt = r["presence"]["per_target"]; mu = np.array(pt["mu"], float); sd = np.sqrt(np.array(pt["var1"], float)); nb = np.array(r["presence"]["cantelli"]["per_unit_n_beta"]["per_target_nstar"], float)
            ctx[arm] = {f"c{c:g}": dict(max=float((mu + sd * math.sqrt((1 - c) / c)).max()), median=float(np.median(mu + sd * math.sqrt((1 - c) / c))), min_margin_nbeta=float((nb - (mu + sd * math.sqrt((1 - c) / c))).min())) for c in (0.14, 0.08)} | dict(nbeta_min=float(nb.min()))
        ALL["context_breakeven_seed0_m1"] = ctx
    json.dump(rnd(ALL), open(OUT_JSON, "w"), indent=0, separators=(",", ":")); print(f"wrote {OUT_JSON}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "anchor", "combine"]); ap.add_argument("--npz"); ap.add_argument("--a", type=float); ap.add_argument("--tag")
    ap.add_argument("--nbeta", nargs="*", default=[]); ap.add_argument("--rnull", type=int, default=1000); ap.add_argument("--decomp-m3", action="store_true")
    args = ap.parse_args()
    if args.cmd == "combine": combine()
    else: run(args, anchor_only=(args.cmd == "anchor"))
