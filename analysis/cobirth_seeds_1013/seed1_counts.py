"""seed1_counts.py (cobirth_seeds_1013, TASK A, agent tag 'seed1', 2026-10-14).

Generalized copy of analysis/cobirth_1013/m1_counts.py (M1: bridge counts, Fano, nulls 1 / 1b / 2, Cantelli) with
  - --npz / --a / --tag (any saved run of the box; the original was hard-wired to seed 0),
  - the M3 section-4 variance decomposition of analysis/cobirth_1013/m3_hubs.py (threshold pool C_i, width bins, null U),
  - Cantelli with per-target n_beta from several p172c states (nearest state per task; conservative = per-target minimum),
  - break-even n*: n*_p(i) = mu_i + sigma_i sqrt((1-p)/p), the smallest n* at which Cantelli reaches p (independent of n_beta),
  - the P177 seed-1 anchors (max loose bridges per target at the reference state, along training, lifetime-1 share t 10..60).
The bridge criterion bridges_at() and the M1 statistics / nulls are copied verbatim (no rewrite of the criterion).

Box: 784-100-10 MLP, 1200 fixed MNIST inputs (subset stored in the npz), CE, Adam 1e-3, batch 16, 4000 updates/task,
iid uniform 10-class labels per task, 100 tasks, hard leaky ReLU slope a. Saved task-end weights only (no training).
Task convention (M1): T = 1..100, state = end0[T-1]; phase 2 = T 11..100 (90 task ends).
  stoch_bridges / p161 / p170 / p172c / P159 use t = T + 1 (their 'task t' = end0[t-2]); converted explicitly below.
  M3 (m3_hubs.py) used phase 2 = t 10..101 = T 9..100 (92 states); its decomposition is also computed on that window
  ('M3win') so the seed-0 numbers of M3 table 4a can be reproduced.

usage: seed1_counts.py --npz PATH --a A --tag TAG [--ref KEY] [--nbeta T=path,T=path,...] [--nbeta-ref T] [--R 1000]
       seed1_counts.py combine TAG1 TAG2 ...      (merge RAW/<tag>.json into results/cobirth_seeds_1013/seed1.json)
  KEY selects the published anchors: a01s0 a03s0 a05s0 (stoch_counts / M1 / M3 / P159 / P161) or a01s1 a03s1 a05s1 (P177).
  --nbeta-ref T: the state whose n_beta is used for the 'single_state' Cantelli (the M1 layout).
CPU, 1 thread, float64."""
import os, sys, json, math, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/seed1"
WT = "/home/issan/Projects/claude/wt/cobirth_seeds_1013"
OUT_JSON = os.path.join(WT, "results/cobirth_seeds_1013/seed1.json")
H = 100; NT = 100; PH2 = list(range(11, 101)); WINDOWS = [(11, 40), (41, 70), (71, 100)]; SEED = 0; MIN_ALIVE = 30; MIN_ALIVE_WIN = 15
R_NULL = 1000                       # >= 500 required; 1000 gives a p-value floor 2/1001 (same as M1)
M3_WIN = list(range(9, 101))        # M3 phase 2: stoch t 10..101 = T 9..100
R_MC = 20                           # null U replicates per state (M3)
KBINS_FINE = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 1200)]
PCTS = (0.32, 0.26, 0.19, 0.14, 0.11, 0.08)   # rho_R thresholds of cumulative_push section 3 (seed 0): c = 1: 32/26/19 %, c = 2.8: 14/11/8 % (a0.1/a0.3/a0.5)
P = lambda *x: print(*x, flush=True)
nan = float("nan")

# published anchors -------------------------------------------------------------------------------------------------
PUB_STOCH = {"a01s0": dict(rows=1511, births=1399, tt=1078, nt_mean=1.38, nt_max=7, S_mean=0.088, S_max=3, B_mean=0.198, B_max=6, n1_mean=0.29, n1_max=7),
             "a03s0": dict(rows=2836, births=2557, tt=1840, nt_mean=1.53, nt_max=14, S_mean=0.128, S_max=3, B_mean=0.484, B_max=11, n1_mean=0.61, n1_max=11),
             "a05s0": dict(rows=7712, births=6649, tt=4685, nt_mean=1.64, nt_max=10, S_mean=0.211, S_max=4, B_mean=0.783, B_max=10, n1_mean=0.99, n1_max=10)}
# P159 (stage16_1008_seed1/logs/p177_table.log): max bridges per target at the reference state (stoch t 40 for a0.1, t 100 otherwise) and along training at t 2/10/20/40/100;
# P161: lifetime-1 share of loose-bridge runs over stoch t 10..60.
PUB_P177 = {"a01s0": dict(ref_t=40, ref_max=5, along=[54, 4, 3, 5, 2], life1=0.94), "a01s1": dict(ref_t=40, ref_max=6, along=[71, 3, 2, 6, 1], life1=0.95),
            "a03s0": dict(ref_t=100, ref_max=5, along=[60, 4, 5, 4, 5], life1=0.92), "a03s1": dict(ref_t=100, ref_max=4, along=[71, 5, 6, 2, 4], life1=0.93),
            "a05s0": dict(ref_t=100, ref_max=4, along=[70, 8, 4, 7, 4], life1=0.88), "a05s1": dict(ref_t=100, ref_max=5, along=[73, 6, 6, 4, 5], life1=0.89)}
ALONG_T_STOCH = [2, 10, 20, 40, 100]
# M1 (seed 0, presence): F_i median, pooled Fano, max N, null-1 shift interval of median F; M3 table 4a (seed 0, T 9..100): unit share, C Fano, (C,k) within Fano
PUB_M1 = {"a01s0": dict(F_med=1.385, pool_fano=1.631, maxN=7), "a03s0": dict(F_med=1.643, pool_fano=1.966, maxN=14), "a05s0": dict(F_med=1.329, pool_fano=1.390, maxN=10)}
PUB_M3 = {"a01s0": dict(unit_share=0.016, C_fano=1.47, Ck_within_fano=0.64, n=8255), "a03s0": dict(unit_share=0.020, C_fano=1.94, Ck_within_fano=0.91, n=9190), "a05s0": dict(unit_share=0.017, C_fano=1.79, Ck_within_fano=0.94, n=9200)}


def bridges_at(O, Dp, Kv):
    """verbatim from stoch_bridges.py bridges_at (rows = target i, cols = candidate j)."""
    alive = O.any(0); k = O.sum(0); need = (k + 1) // 2; Oi = O.astype(np.int32); ov = Oi.T @ Oi
    E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    isb = (thr >= 2) & (ov >= need[:, None]); isb[np.diag_indices(H)] = False; isb[~alive, :] = False; isb[:, ~alive] = False
    return isb, alive, k


def thr_and_T(O, Dp, Kv):
    """the threshold inputs T_i (top-3 of K_n dd/dz) and thr[i, j] = |O_j & T_i|, exactly as inside bridges_at (for the pool C_i)."""
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
    return float(var / (var + (nstar - mu) ** 2))


def ind_stats(M):
    """M: (n tasks, m candidates) 0/1 indicator matrix for one target. Row sums N_t. Returns dict incl. pairwise excess."""
    n = M.shape[0]; Nn = M.sum(1).astype(float); mu = float(Nn.mean()); var0 = float(Nn.var()); var1 = float(Nn.var(ddof=1)) if n > 1 else nan
    p = M.mean(0); sp2 = float((p ** 2).sum())
    G = (M.T.astype(float) @ M.astype(float)) / n - np.outer(p, p); delta = float(G.sum() - np.trace(G))
    return dict(n=n, mu=mu, var0=var0, var1=var1, fano=(var1 / mu if mu > 0 else nan), max=int(Nn.max()), ge2=float((Nn >= 2).mean()), ge3=float((Nn >= 3).mean()),
                sp2=sp2, delta=delta, delta_id=var0 - mu + sp2, var1_minus_mu=(var1 - mu) if n > 1 else nan, hist=np.bincount(Nn.astype(int)).tolist(), N=Nn)


def null_cols(M, R, rng, mode):
    """verbatim from m1_counts.py. mode 'shift': independent circular shift of each candidate's series; 'perm': independent permutation;
    'strat': independent permutation within width strata. Returns array (R, 5): fano, max, ge2, ge3, delta."""
    n, m = M.shape; ar = np.arange(n); p = M.mean(0); sp2 = float((p ** 2).sum()); mu = float(M.sum(1).mean()); out = np.empty((R, 5))
    if mode == "strat":
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


def var_decomp(vals, groups):
    """verbatim from m3_hubs.py: population variance of vals decomposed into across-group and within-group parts"""
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


# ---------------------------------------------------------------------------------------------------------------------
def analyze(kind, IND, aliveT, tag, rng, kmat, nbeta):
    """M1 analysis (verbatim order of rng calls so that seed-0 nulls reproduce M1) + generalized Cantelli + break-even n*.
    kind 'presence' or 'births'. IND: dict T -> (H,H) bool indicator (lifter / birth). aliveT: (NT, H) bool."""
    res = {}; per = []; Mats = {}
    cols_all = np.arange(H)
    for i in range(H):
        Ts = [T for T in PH2 if aliveT[T - 1, i]]
        if len(Ts) == 0: per.append(None); continue
        M = np.stack([IND[T][i, :] for T in Ts], 0).astype(np.int8); M = M[:, cols_all != i]
        Mats[i] = (Ts, M); st = ind_stats(M); st["i"] = i; st["Ts"] = Ts; per.append(st)
    ok = [st for st in per if st is not None and st["n"] >= MIN_ALIVE]
    res["n_targets_alive_any"] = len([s for s in per if s is not None]); res["n_targets_used"] = len(ok); res["min_alive_tasks"] = MIN_ALIVE
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
    # ---- null 1 / 1b (verbatim M1) ----
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
                                           n_max_above_null_q975=int((mx > np.quantile(arr[:, :, 1], 0.975, axis=1)).sum())))
    nulls["strat"]["note"] = f"independent permutation of each candidate's series within {NSTRATA} quantile strata of the target's width k_i(T) over its alive Phase-2 tasks"
    res["null_candidate_series"] = nulls
    # ---- pooled stats by target width (M1 4.4) ----
    kb_edges = [0, 3, 8, 30, 100, 300, 10 ** 9]; kb_lab = ["1-3", "4-8", "9-30", "31-100", "101-300", ">300"]; kt = {}
    kall = np.concatenate([[kmat[T - 1, s["i"]] for T in s["Ts"]] for s in ok]).astype(float)
    for lo, hi, lab in zip(kb_edges[:-1], kb_edges[1:], kb_lab):
        sel = (kall > lo) & (kall <= hi)
        if sel.sum() == 0: continue
        v = Nall[sel]; m_ = v.mean()
        kt[lab] = dict(n_target_tasks=int(sel.sum()), mean=float(m_), fano=float(v.var(ddof=1) / m_) if m_ > 0 and len(v) > 1 else None, ge2=float((v >= 2).mean()), ge3=float((v >= 3).mean()), max=int(v.max()), poisson_ge2=float(pois_tail(m_, 2)), poisson_ge3=float(pois_tail(m_, 3)))
    res["by_target_width"] = kt
    # ---- null 2 (verbatim M1) ----
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
                                      per_target=dict(p_fano=pf2, p_max=pm2, n_p_fano_lt05=int(np.nansum(pf2 < 0.05)), n_p_max_lt05=int(np.nansum(pm2 < 0.05)), n_fano_tested=int(np.isfinite(pf2).sum())))
    sdmu2 = np.zeros(R_NULL)
    for r in range(R_NULL):
        Nr = Nmat.copy()
        for row in range(len(PH2)):
            al = np.where(Nr[row] >= 0)[0]; Nr[row, al] = Nr[row, al[rng.permutation(len(al))]]
        mus = np.array([Nr[:, i][Nr[:, i] >= 0].mean() for i in used_idx]); sdmu2[r] = mus.std()
    res["null_target_shuffle"]["null"]["sd_mu_across_targets"] = q(sdmu2, (0.025, 0.5, 0.975)); res["null_target_shuffle"]["p_two_sided"]["sd_mu_across_targets"] = pval2(float(mu.std()), sdmu2)
    # ---- Cantelli (generalized: n*_i(T) from several p172c states) ----
    res["cantelli"] = cantelli_all(ok, mu, var1, nbeta, tag)
    # ---- break-even n*: smallest n* with Cantelli = p, per target ----
    sd1 = np.sqrt(np.where(np.isfinite(var1), var1, 0.0)); be = {}
    nbmin = nbeta_per_target_min(ok, nbeta) if nbeta else None
    for p in PCTS:
        ns = mu + sd1 * math.sqrt((1 - p) / p); arg = int(np.argmax(ns))
        d = dict(p=p, max=float(ns.max()), argmax_target=int(ok[arg]["i"]), mu_at_max=float(mu[arg]), var_at_max=float(var1[arg]), median=float(np.median(ns)), q90=float(np.quantile(ns, 0.9)))
        if nbmin is not None:
            have = np.isfinite(nbmin)
            d["n_beta_min_over_targets"] = float(np.nanmin(nbmin)) if have.any() else None
            d["margin_min_over_targets_nbeta_minus_breakeven"] = float(np.nanmin(nbmin - ns)) if have.any() else None
            d["n_targets_nbeta_le_breakeven"] = int(np.nansum(nbmin <= ns)); d["n_targets_with_nbeta"] = int(have.sum())
            d["ratio_min_nbeta_over_max_breakeven"] = float(np.nanmin(nbmin) / ns.max()) if have.any() else None
        be[f"{p:.2f}"] = d
    res["breakeven_nstar"] = be
    # ---- time windows (verbatim M1) ----
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
    return res, per


# ---------------------------------------------------------------------------------------------------------------------
def load_nbeta(spec):
    """spec: {T: path}. Returns {T: {i: n_b}} (p172c 'real' rows; p172c state = end0[t-2] with t = T + 1)."""
    out = {}
    for T, path in spec.items():
        jd = json.load(open(path)); out[int(T)] = {int(r["i"]): float(r["n_b"]) for r in jd["real"]}
        assert int(jd["task"]) == int(T) + 1, f"{path}: p172c task {jd['task']} != T+1 = {int(T) + 1}"
    return out


def nbeta_per_target_min(ok, nbeta):
    return np.array([min([nbeta[Ts][s["i"]] for Ts in nbeta if s["i"] in nbeta[Ts]], default=nan) for s in ok])


def nstar_nearest(i, T, nbeta, states):
    """n_beta of target i at the computed state nearest to T where i has a value (ties: earlier state). None if no state has i."""
    cand = sorted([Ts for Ts in states if i in nbeta[Ts]], key=lambda Ts: (abs(Ts - T), Ts))
    return (nbeta[cand[0]][i], cand[0]) if cand else (None, None)


def cantelli_all(ok, mu, var1, nbeta, tag):
    out = {}
    def block(nstar_rows, label, extra=None):
        """nstar_rows[a] = array of n* per alive phase-2 task of target a (aligned with ok[a]['N'])."""
        bt = []; pz = []; emp_n = 0; emp_frac = []; bmax_task = 0.0; pois_tt = []
        for a, s in enumerate(ok):
            ns = nstar_rows[a]; b = np.array([cantelli(mu[a], var1[a], x) for x in ns]); bt.append(b.mean()); bmax_task = max(bmax_task, float(b.max()))
            pzv = np.array([pois_tail(mu[a], x) for x in ns]); pz.append(pzv.mean())
            hit = (s["N"] >= ns); emp_n += int(hit.sum()); emp_frac.append(float(hit.mean()))
        bt = np.array(bt); pz = np.array(pz); allns = np.concatenate(nstar_rows); fin = allns[np.isfinite(allns)]
        d = dict(label=label, nstar=dict(median=float(np.median(fin)) if fin.size else None, min=float(fin.min()) if fin.size else None, n_inf=int((~np.isfinite(allns)).sum()), n_le_mu=int(sum(int((nstar_rows[a] <= mu[a]).sum()) for a in range(len(ok))))),
                 cantelli=dict(max=float(bt.max()), median=float(np.median(bt)), q90=float(np.quantile(bt, 0.9)), mean=float(bt.mean()), sum_over_targets=float(bt.sum()), max_single_target_task=bmax_task,
                               argmax_target=int(ok[int(np.argmax(bt))]["i"]), union90_max=float(min(1.0, 90 * bt.max()))),
                 poisson_indep=dict(max=float(pz.max()), median=float(np.median(pz)), mean=float(pz.mean())), empirical=dict(max=float(max(emp_frac)), n_target_tasks_ge_nstar=int(emp_n)),
                 per_target_cantelli=bt)
        if extra: d.update(extra)
        return d
    if nbeta:
        states = sorted(nbeta)
        # (a) nearest computed state per task
        rows = []; src_count = {Ts: 0 for Ts in states}; n_fill = 0; med_all = float(np.median([v for Ts in states for v in nbeta[Ts].values() if np.isfinite(v)]))
        for s in ok:
            r = []
            for T in s["Ts"]:
                v, Ts = nstar_nearest(s["i"], T, nbeta, states)
                if v is None: v = med_all; n_fill += 1
                else: src_count[Ts] += 1
                r.append(v)
            rows.append(np.array(r, float))
        out["nbeta_nearest_state"] = block(rows, f"per-target n_beta at the computed p172c state nearest to each task (states T {states}; ties -> earlier; targets without any value filled with the median {med_all:.2f})",
                                           dict(target_tasks_by_source_state={str(k): v for k, v in src_count.items()}, target_tasks_filled_with_median=n_fill))
        # (b) conservative: per-target minimum over computed states
        nbmin = nbeta_per_target_min(ok, nbeta); n_nov = int((~np.isfinite(nbmin)).sum()); nbmin_f = np.where(np.isfinite(nbmin), nbmin, np.nanmin(nbmin))
        out["nbeta_min_over_states"] = block([np.full(len(s["Ts"]), nbmin_f[a]) for a, s in enumerate(ok)], f"per-target minimum of n_beta over computed states {states} (targets with no value: {n_nov}, filled with the arm minimum)",
                                             dict(n_targets_without_value=n_nov, per_target_nstar=nbmin_f))
        # (c) single state (the M1 layout): n_beta at the reference state, missing filled with that state's median
        Tref = SINGLE_REF[0]
        if Tref in nbeta:
            v = np.array([nbeta[Tref].get(s["i"], nan) for s in ok]); have = np.isfinite(v) | np.isinf(v); fin = v[np.isfinite(v)]
            vf = np.where(have, v, np.median(fin))
            out["nbeta_single_state"] = block([np.full(len(s["Ts"]), vf[a]) for a, s in enumerate(ok)], f"per-target n_beta at the single state T {Tref} (M1 layout; {int(have.sum())}/{len(ok)} targets have a value, others filled with the median)",
                                              dict(state_T=Tref, n_targets_with_value=int(have.sum())))
        allv = np.array([v for Ts in states for v in nbeta[Ts].values()]); fin = allv[np.isfinite(allv)]
        out["nbeta_states_summary"] = {str(Ts): dict(n_units=len(nbeta[Ts]), min=float(min(v for v in nbeta[Ts].values() if np.isfinite(v))), median=float(np.median([v for v in nbeta[Ts].values() if np.isfinite(v)])),
                                                     n_inf=int(sum(1 for v in nbeta[Ts].values() if not np.isfinite(v))), n_le_0=int(sum(1 for v in nbeta[Ts].values() if v <= 0))) for Ts in states}
        consts = (("arm_min_all_states", float(fin.min())), ("arm_median_all_states", float(np.median(fin))), ("ten", 10.0))
    else:
        consts = (("ten", 10.0),)
    for lab, val in consts:
        out[lab] = block([np.full(len(s["Ts"]), val) for s in ok], f"constant n* = {val:.3f}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
def m3_decomp(ISB, BIRTH, ALIVE, KK, CC, CU_mean, CU_var, NU_mean, NU_var, NU_rep0, window, bwindow):
    """M3 section-4 decomposition on the (i, T) alive in `window` (T list). N = presence; B = births at T in `bwindow` (targets alive at T).
    M1 window: N and B on T 11..100. M3 window: N on T 9..100 (stoch t 10..101), B on T 10..100 (M3: t+1 for t in 10..100)."""
    recs = [(T, i) for T in window for i in np.where(ALIVE[T - 1])[0]]
    tt = np.array([r[0] for r in recs]); ii = np.array([r[1] for r in recs]); kk = np.array([KK[T - 1, i] for T, i in recs]); NN = np.array([ISB[T - 1][i].sum() for T, i in recs], float)
    C = np.array([CC[T - 1, i] for T, i in recs], float); CUm = np.array([CU_mean[T - 1, i] for T, i in recs]); CUv = np.array([CU_var[T - 1, i] for T, i in recs]); LU = np.array([NU_mean[T - 1, i] for T, i in recs]); LUv = np.array([NU_var[T - 1, i] for T, i in recs]); NU1 = np.array([NU_rep0[T - 1, i] for T, i in recs])
    kbf = np.array([kbin_of(k, KBINS_FINE) for k in kk])
    vd = dict(window=f"T {window[0]}..{window[-1]}", births_window=f"T {bwindow[0]}..{bwindow[-1]}", n=len(recs),
              N_by_unit=var_decomp(NN, ii), N_by_kbin_fine=var_decomp(NN, kbf), N_by_unit_x_kbin_fine=var_decomp(NN, ii * 100 + kbf), N_by_task=var_decomp(NN, tt),
              N_by_pool=var_decomp(NN, np.minimum(C, 30)), N_by_pool_x_kbin_fine=var_decomp(NN, kbf * 100 + np.minimum(C, 30)),
              C_by_unit=var_decomp(C, ii), C_by_kbin_fine=var_decomp(C, kbf), C_by_task=var_decomp(C, tt),
              NU_rep0_by_unit=var_decomp(NU1, ii), NU_rep0_by_kbin_fine=var_decomp(NU1, kbf))
    # per-target Fano of C_i over its alive tasks in the window (targets with >= MIN_ALIVE tasks)
    cf = []
    for i in range(H):
        s = ii == i
        if s.sum() >= MIN_ALIVE and C[s].mean() > 0: cf.append(C[s].var(ddof=1) / C[s].mean())
    vd["pool"] = dict(C_mean=float(C.mean()), C_var=float(C.var()), C_fano=float(C.var() / C.mean()) if C.mean() > 0 else None, C_max=int(C.max()),
                      C_fano_per_target=q(cf, (0.1, 0.5, 0.9)),
                      CU_mean=float(CUm.mean()), CU_hetero_var=float(CUm.var()), CU_within_var=float(CUv.mean()), CU_total_fano=float((CUm.var() + CUv.mean()) / CUm.mean()) if CUm.mean() > 0 else None,
                      ratio_C_CU=float(C.sum() / CUm.sum()) if CUm.sum() > 0 else None, pass_rate_N_over_C=float(NN.sum() / C.sum()) if C.sum() > 0 else None,
                      pass_rate_null=float(LU.sum() / CUm.sum()) if CUm.sum() > 0 else None, ratio_obs_over_lamU=float(NN.sum() / LU.sum()) if LU.sum() > 0 else None,
                      NU_total_fano=float((LU.var() + LUv.mean()) / LU.mean()) if LU.mean() > 0 else None)
    # births at T (targets alive at T; T >= 2)
    brecs = [(T, i) for T in bwindow if T >= 2 for i in np.where(ALIVE[T - 1])[0]]
    BB = np.array([BIRTH[T - 1][i].sum() for T, i in brecs], float); bi = np.array([r[1] for r in brecs]); bk = np.array([KK[T - 1, i] for T, i in brecs]); bkf = np.array([kbin_of(k, KBINS_FINE) for k in bk])
    bC = np.array([CC[T - 1, i] for T, i in brecs], float); bt_ = np.array([r[0] for r in brecs])
    vd["B_by_unit"] = var_decomp(BB, bi); vd["B_by_kbin_fine"] = var_decomp(BB, bkf); vd["B_by_task"] = var_decomp(BB, bt_); vd["B_by_pool_x_kbin_fine"] = var_decomp(BB, bkf * 100 + np.minimum(bC, 30))
    return vd


# ---------------------------------------------------------------------------------------------------------------------
SINGLE_REF = [None]


def run(npz, a, tag, ref, nbeta_spec, nbeta_ref):
    t0 = time.time(); dd = np.load(npz if os.path.isabs(npz) else os.path.join(ROOT, npz))
    X = load_X(dd["subset"]); N = X.shape[0]; Xt = np.concatenate([X, np.ones((N, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar
    ISB = np.zeros((NT, H, H), bool); ALIVE = np.zeros((NT, H), bool); KK = np.zeros((NT, H), np.int16); CC = np.zeros((NT, H), np.int16); OO = {}; TT = {}
    for T in range(1, NT + 1):
        W = np.concatenate([dd["end0"][T - 1].astype(np.float64), dd["end1"][T - 1].astype(np.float64)[:, None]], 1); Z = Xt @ W.T; tau = 0.1 * Z.std(0) + 1e-12
        s = 0.5 * (1 + np.tanh(0.5 * Z / tau)); O = Z > 0; Dp = (1 - a) * s * (1 - s) / tau
        isb, alive, k = bridges_at(O, Dp, Kv); ISB[T - 1] = isb; ALIVE[T - 1] = alive; KK[T - 1] = k
        Ti, thr = thr_and_T(O, Dp, Kv); pool = (thr >= 2) & alive[None, :]; pool[np.diag_indices(H)] = False; CC[T - 1] = np.where(alive, pool.sum(1), -1)   # C_i = #{alive j != i : |O_j & T_i| >= 2} (M3)
        if T >= M3_WIN[0]: OO[T] = O; TT[T] = Ti
    BIRTH = np.zeros((NT, H, H), bool); BIRTH[1:] = ISB[1:] & ~ISB[:-1]; BIRTH[0] = False
    Ncnt = np.where(ALIVE, ISB.sum(2), -1).astype(np.int16); Bcnt = np.where(ALIVE, BIRTH.sum(2), -1).astype(np.int16); Bcnt[0] = -1
    P(f"[{tag}] states + bridges done ({time.time() - t0:.1f}s); a={a} N={N}; alive units per T: T1 {ALIVE[0].sum()} T11 {ALIVE[10].sum()} T50 {ALIVE[49].sum()} T100 {ALIVE[99].sum()}")
    # ---- anchors in the stoch convention (t = T + 1) ----
    rows_stoch = int(sum(ISB[T - 1].sum() for T in range(9, 100)))                       # stoch rows t = 10..100 -> T = 9..99
    births_stoch = int(sum(BIRTH[T - 1][ALIVE[T - 1]].sum() for T in range(10, 100)))     # stoch births t = 11..100 -> T = 10..99
    recs = []
    for T in range(9, 99):
        for i in np.where(ALIVE[T - 1])[0]:
            n_t = int(ISB[T - 1][i].sum())
            if n_t >= 1:
                S = int((ISB[T - 1][i] & ISB[T][i]).sum()); n1 = int(ISB[T][i].sum()); recs.append((n_t, S, n1 - S, n1))
    rec = np.array(recs)
    stoch = dict(rows_t10_100=rows_stoch, births_t11_100=births_stoch, target_tasks_nt_ge1=len(recs), nt_mean=float(rec[:, 0].mean()), nt_max=int(rec[:, 0].max()), S_mean=float(rec[:, 1].mean()), S_max=int(rec[:, 1].max()),
                 B_mean=float(rec[:, 2].mean()), B_max=int(rec[:, 2].max()), n1_mean=float(rec[:, 3].mean()), n1_max=int(rec[:, 3].max()))
    if ref in PUB_STOCH:
        pub = PUB_STOCH[ref]; stoch["published"] = pub
        stoch["match"] = bool(rows_stoch == pub["rows"] and births_stoch == pub["births"] and len(recs) == pub["tt"] and rec[:, 0].max() == pub["nt_max"] and rec[:, 2].max() == pub["B_max"] and abs(rec[:, 2].mean() - pub["B_mean"]) < 6e-4 and abs(rec[:, 0].mean() - pub["nt_mean"]) < 6e-3)
    P(f"[{tag}] STOCH (t=T+1): rows t10..100 {rows_stoch} | births t11..100 {births_stoch} | target-tasks n_t>=1 {len(recs)} | n_t mean {rec[:,0].mean():.3f} max {rec[:,0].max()} | S mean {rec[:,1].mean():.3f} max {rec[:,1].max()} | B mean {rec[:,2].mean():.3f} max {rec[:,2].max()} | n1 mean {rec[:,3].mean():.3f} max {rec[:,3].max()}" + (f" | published {stoch['published']} MATCH {stoch['match']}" if "match" in stoch else ""))
    # P159 / P161 anchors: max bridges per target at stoch t (T = t - 1); lifetime-1 share of runs over stoch t 10..60 (T 9..59)
    along = [int(Ncnt[t - 2].max()) for t in ALONG_T_STOCH]
    pairs_by_T = {T: set(zip(*np.where(ISB[T - 1]))) for T in range(9, 60)}; alive_pairs = {}
    for T in range(9, 60):
        for pr in pairs_by_T[T]: alive_pairs.setdefault(pr, []).append(T)
    lifetimes = []
    for pr, ts in alive_pairs.items():
        ts = sorted(ts); run_ = 1
        for kx in range(1, len(ts)):
            if ts[kx] == ts[kx - 1] + 1: run_ += 1
            else: lifetimes.append(run_); run_ = 1
        lifetimes.append(run_)
    life1 = float(np.mean(np.array(lifetimes) == 1)); n_per_T = [len(pairs_by_T[T]) for T in range(9, 60)]
    p177 = dict(along_stoch_t=ALONG_T_STOCH, along_T=[t - 1 for t in ALONG_T_STOCH], max_bridges_per_target_along=along, lifetime1_share_t10_60=life1, lifetime_max_t10_60=int(max(lifetimes)), n_runs=len(lifetimes),
                count_per_task_median_t10_60=float(np.median(n_per_T)), max_bridges_per_target_phase2_T11_100=int(Ncnt[10:].max()), max_bridges_per_target_stoch_t10_101=int(Ncnt[8:].max()))
    if ref in PUB_P177:
        pub = PUB_P177[ref]; ref_max = int(Ncnt[pub["ref_t"] - 2].max()); p177.update(ref_t_stoch=pub["ref_t"], ref_T=pub["ref_t"] - 1, ref_max=ref_max, published=pub,
                                                                              match_ref_max=bool(ref_max == pub["ref_max"]), match_along=bool(along == pub["along"]), match_life1=bool(round(life1, 2) == pub["life1"]))
    P(f"[{tag}] P159/P161 anchors: max bridges/target at stoch t {ALONG_T_STOCH} = {along}; ref state {p177.get('ref_t_stoch')} max {p177.get('ref_max')} | lifetime-1 share t10..60 {life1:.4f} (runs {len(lifetimes)}, max {max(lifetimes)}) | max in phase 2 T11..100 {p177['max_bridges_per_target_phase2_T11_100']}"
      + (f" | published {PUB_P177[ref]} MATCH ref {p177['match_ref_max']} along {p177['match_along']} life1 {p177['match_life1']}" if ref in PUB_P177 else ""))
    # ---- null U for the threshold pool C (M3; separate rng with M3's seed and state order so seed 0 reproduces M3) ----
    rngU = np.random.default_rng(20261013); CU_mean = np.zeros((NT, H)); CU_var = np.zeros((NT, H)); NU_mean = np.zeros((NT, H)); NU_var = np.zeros((NT, H)); NU_rep0 = np.zeros((NT, H))
    for T in M3_WIN:
        O = OO[T]; Ti = TT[T]; k = KK[T - 1].astype(int); alive = ALIVE[T - 1]; need = (k + 1) // 2
        NU = np.zeros((R_MC, H)); CU = np.zeros((R_MC, H))
        for rep in range(R_MC):
            U = rngU.random((N, H)); ranks = np.argsort(np.argsort(U, axis=0), axis=0); On = ranks < k[None, :]
            thr_n = np.stack([On[Ti[:, i], :].sum(0) for i in range(H)], 0); ov_n = O.astype(np.int32).T @ On.astype(np.int32)
            isb_n = (thr_n >= 2) & (ov_n >= need[:, None]); isb_n[np.diag_indices(H)] = False; isb_n[~alive, :] = False; isb_n[:, ~alive] = False
            NU[rep] = isb_n.sum(1); pool_n = (thr_n >= 2) & alive[None, :]; pool_n[np.diag_indices(H)] = False; CU[rep] = pool_n.sum(1)
        CU_mean[T - 1] = CU.mean(0); CU_var[T - 1] = CU.var(0); NU_mean[T - 1] = NU.mean(0); NU_var[T - 1] = NU.var(0); NU_rep0[T - 1] = NU[0]
    m3 = dict(M1win=m3_decomp(ISB, BIRTH, ALIVE, KK, CC, CU_mean, CU_var, NU_mean, NU_var, NU_rep0, PH2, PH2), M3win=m3_decomp(ISB, BIRTH, ALIVE, KK, CC, CU_mean, CU_var, NU_mean, NU_var, NU_rep0, M3_WIN, list(range(10, 101))))
    if ref in PUB_M3:
        pub = PUB_M3[ref]; v = m3["M3win"]
        m3["anchor_M3"] = dict(published=pub, unit_share=v["N_by_unit"]["share_across"], C_fano=v["C_by_unit"]["fano"], Ck_within_fano=v["N_by_pool_x_kbin_fine"]["fano_within"], n=v["n"],
                               match=bool(v["n"] == pub["n"] and round(v["N_by_unit"]["share_across"], 3) == pub["unit_share"] and round(v["C_by_unit"]["fano"], 2) == pub["C_fano"] and round(v["N_by_pool_x_kbin_fine"]["fano_within"], 2) == pub["Ck_within_fano"]))
    for wn in ("M1win", "M3win"):
        v = m3[wn]; pl = v["pool"]
        P(f"[{tag}] M3 decomposition {v['window']} (n {v['n']}): N Fano {v['N_by_unit']['fano']:.3f} | unit share {v['N_by_unit']['share_across']:.4f} (within Fano {v['N_by_unit']['fano_within']:.3f}) | k-bin share {v['N_by_kbin_fine']['share_across']:.3f} within Fano {v['N_by_kbin_fine']['fano_within']:.3f} | task share {v['N_by_task']['share_across']:.3f} | C share {v['N_by_pool']['share_across']:.3f} within Fano {v['N_by_pool']['fano_within']:.3f} | (C,k) share {v['N_by_pool_x_kbin_fine']['share_across']:.3f} within Fano {v['N_by_pool_x_kbin_fine']['fano_within']:.3f}")
        P(f"[{tag}]   pool C: mean {pl['C_mean']:.3f} Fano {pl['C_fano']:.3f} max {pl['C_max']} | per-target C Fano q10/50/90 {pl['C_fano_per_target']['0.1']}/{pl['C_fano_per_target']['0.5']}/{pl['C_fano_per_target']['0.9']} | null U: C mean {pl['CU_mean']:.3f} total Fano {pl['CU_total_fano']:.3f} | C/C_U {pl['ratio_C_CU']:.3f} | pass N/C {pl['pass_rate_N_over_C']:.4f} null {pl['pass_rate_null']:.4f} | obs/lamU {pl['ratio_obs_over_lamU']:.2f} | C by k-bin within Fano {v['C_by_kbin_fine']['fano_within']:.3f}, C unit share {v['C_by_unit']['share_across']:.4f} | births: unit share {v['B_by_unit']['share_across']:.4f} (C,k) within Fano {v['B_by_pool_x_kbin_fine']['fano_within']:.3f}")
    if "anchor_M3" in m3: P(f"[{tag}] M3 anchor (T 9..100): {m3['anchor_M3']}")
    # ---- n_beta ----
    nbeta = load_nbeta(nbeta_spec) if nbeta_spec else {}
    SINGLE_REF[0] = nbeta_ref
    for Ts in sorted(nbeta): P(f"[{tag}] n_beta state T {Ts} (p172c t {Ts + 1}): units {len(nbeta[Ts])} min {min(nbeta[Ts].values()):.2f} median {np.median(list(nbeta[Ts].values())):.2f}")
    rng = np.random.default_rng(SEED)
    IND_P = {T: ISB[T - 1] for T in range(1, NT + 1)}; IND_B = {T: BIRTH[T - 1] for T in range(1, NT + 1)}
    resP, perP = analyze("presence", IND_P, ALIVE, tag, rng, KK, nbeta); P(f"[{tag}] presence analysis done ({time.time() - t0:.1f}s)")
    resB, perB = analyze("births", IND_B, ALIVE, tag, rng, KK, nbeta); P(f"[{tag}] births analysis done ({time.time() - t0:.1f}s)")
    if ref in PUB_M1:
        pub = PUB_M1[ref]; resP["anchor_M1"] = dict(published=pub, F_med=resP["fano_quantiles"]["0.5"], pool_fano=resP["pooled"]["fano"], maxN=resP["pooled"]["max"],
                                                    match=bool(round(resP["fano_quantiles"]["0.5"], 3) == pub["F_med"] and round(resP["pooled"]["fano"], 3) == pub["pool_fano"] and resP["pooled"]["max"] == pub["maxN"]))
        P(f"[{tag}] M1 anchor: {resP['anchor_M1']}")
    for nm, r in (("presence", resP), ("births", resB)):
        po = r["pooled"]; fq = r["fano_quantiles"]; n1 = r["null_candidate_series"]["shift"]; n3 = r["null_candidate_series"]["strat"]; n2 = r["null_target_shuffle"]
        P(f"[{tag}] {nm}: targets used {r['n_targets_used']} (mu=0: {r['n_targets_mu0']}) | mu_i median {r['mu_quantiles']['0.5']:.3f} [q10 {r['mu_quantiles']['0.1']:.3f}, q90 {r['mu_quantiles']['0.9']:.3f}] | pooled mean {po['mean']:.4f} Fano {po['fano']:.3f} (within {po['fano_within']:.3f} between {po['fano_between']:.3f}) max {po['max']} | P>=2 {po['ge2']:.4f} (Pois {po['poisson_ge2']:.4f}, mix {po['mixture_ge2']:.4f}) P>=3 {po['ge3']:.5f} (Pois {po['poisson_ge3']:.5f}, mix {po['mixture_ge3']:.5f}) P>=5 {po['ge5']:.5f} ({po['n_ge5']}) N>=10 {po['n_ge10']}")
        P(f"[{tag}] {nm}: per-target Fano q10 {fq['0.1']:.3f} q50 {fq['0.5']:.3f} q90 {fq['0.9']:.3f} max {fq['max']:.3f} share>1 {r['share_fano_gt1']:.3f} | sum delta {r['sum_delta']:.3f} share delta>0 {r['share_delta_gt0']:.3f} | identity err {r['identity_max_abs_err']:.1e}")
        for lab, nn in (("null1 shift", n1), ("null1b strat(k_i)", n3)):
            P(f"[{tag}] {nm}: {lab}: median F obs {nn['observed']['median_fano']:.3f} null [{nn['null']['median_fano']['0.025']:.3f},{nn['null']['median_fano']['0.975']:.3f}] p {nn['p_two_sided']['median_fano']:.3f} | max N obs {nn['observed']['max_N']} null [{nn['null']['max_N']['0.025']:.0f},{nn['null']['max_N']['0.975']:.0f}] p {nn['p_two_sided']['max_N']:.3f} | P>=3 null [{nn['null']['pooled_ge3']['0.025']:.4f},{nn['null']['pooled_ge3']['0.975']:.4f}] p {nn['p_two_sided']['pooled_ge3']:.3f} | sum delta null [{nn['null']['sum_delta']['0.025']:.2f},{nn['null']['sum_delta']['0.975']:.2f}] p {nn['p_two_sided']['sum_delta']:.3f} | per-target p_F<.05 {nn['per_target']['n_p_fano_lt05']}/{nn['per_target']['n_fano_tested']}")
        P(f"[{tag}] {nm}: null2 target-shuffle: median F null [{n2['null']['median_fano']['0.025']:.3f},{n2['null']['median_fano']['0.975']:.3f}] p {n2['p_two_sided']['median_fano']:.3f} | sd(mu_i) obs {n2['observed']['sd_mu_across_targets']:.4f} null [{n2['null']['sd_mu_across_targets']['0.025']:.4f},{n2['null']['sd_mu_across_targets']['0.975']:.4f}] p {n2['p_two_sided']['sd_mu_across_targets']:.3f}")
        for lab, c in r["cantelli"].items():
            if lab == "nbeta_states_summary": continue
            P(f"[{tag}] {nm}: Cantelli [{lab}] n* med {c['nstar']['median']} min {c['nstar']['min']}: bound max {c['cantelli']['max']:.3e} (target {c['cantelli']['argmax_target']}) median {c['cantelli']['median']:.3e} sum_i {c['cantelli']['sum_over_targets']:.3f} | Poisson max {c['poisson_indep']['max']:.3e} | empirical n(N>=n*) {c['empirical']['n_target_tasks_ge_nstar']} max freq {c['empirical']['max']:.4f}")
        for lab, b in r["breakeven_nstar"].items():
            P(f"[{tag}] {nm}: break-even n* (Cantelli = {b['p']:.2f}): max {b['max']:.3f} (target {b['argmax_target']}, mu {b['mu_at_max']:.3f} var {b['var_at_max']:.3f}) median {b['median']:.3f}" + (f" | n_beta min {b['n_beta_min_over_targets']:.2f}, min margin n_beta-n*_p {b['margin_min_over_targets_nbeta_minus_breakeven']:.2f}, targets n_beta<=n*_p {b['n_targets_nbeta_le_breakeven']}/{b['n_targets_with_nbeta']}" if "n_beta_min_over_targets" in b else ""))
        for w, v in r["windows"].items():
            P(f"[{tag}] {nm}: window {w}: targets {v['n_targets']} mean mu {v['mean_mu']:.4f} median Fano {v['median_fano']:.3f} pooled Fano {v['pooled_fano']:.3f} P>=2 {v['pooled_ge2']:.4f} max {v['max']}")
    np.savez_compressed(os.path.join(RAW, f"counts_{tag}.npz"), N=Ncnt, B=Bcnt, alive=ALIVE, k=KK, C=CC, isb=ISB, birth=BIRTH, CU_mean=CU_mean, CU_var=CU_var, NU_mean=NU_mean, T=np.arange(1, NT + 1),
                        note=np.array(f"M1 convention T = 1..100, state end0[T-1]; arrays [T-1, i] ([T-1, i, j] for isb/birth: row target, col candidate); dead target -> -1; C = threshold pool; CU/NU = null U (R {R_MC}) for T 9..100; npz {npz}; a {a}"))
    out = dict(tag=tag, npz=npz, a=a, ref=ref, R_null=R_NULL, R_MC=R_MC, stoch_anchor=stoch, p177_anchor=p177, m3_decomp=m3, nbeta_spec={str(k): v for k, v in (nbeta_spec or {}).items()}, nbeta_ref_T=nbeta_ref,
               presence=resP, births=resB, elapsed_s=time.time() - t0)
    json.dump(rnd(out), open(os.path.join(RAW, f"{tag}.json"), "w"), indent=1)
    P(f"[{tag}] saved {RAW}/{tag}.json and counts_{tag}.npz ({time.time() - t0:.1f}s)")


def combine(tags):
    ALL = dict(convention="M1: T = 1..100 is the state end0[T-1]; phase 2 = T 11..100. stoch/p161/p170/p172c/P159 use t = T + 1. M3 window = T 9..100.",
               criterion="verbatim bridges_at from stoch_bridges.py (P158/P161/P176 loose criterion)", conditions={})
    for t in tags:
        d = json.load(open(os.path.join(RAW, f"{t}.json")))
        for kind in ("presence", "births"):   # drop the long per-target arrays from the compact results copy
            for k in ("per_target",): d[kind].pop(k, None)
            for m in d[kind]["null_candidate_series"].values(): m.get("per_target", {}).pop("p_fano", None); m.get("per_target", {}).pop("p_max", None); m.get("per_target", {}).pop("p_delta", None)
            d[kind]["null_target_shuffle"]["per_target"].pop("p_fano", None); d[kind]["null_target_shuffle"]["per_target"].pop("p_max", None)
            for c in d[kind]["cantelli"].values():
                if isinstance(c, dict): c.pop("per_target_cantelli", None); c.pop("per_target_nstar", None)
            for kk in ("hist_poisson_same_mean", "hist_mixture_per_target_poisson"): d[kind]["pooled"][kk] = [round(x, 3) for x in d[kind]["pooled"][kk]]
        ALL["conditions"][t] = d
    for nm in ("p172c_anchor_check", "nbeta_pairing_supp"):   # written by seed1_nbeta_supp.py
        f = os.path.join(RAW, f"{nm}.json")
        if os.path.exists(f): ALL[nm] = json.load(open(f))
    json.dump(ALL, open(OUT_JSON, "w"), indent=0, separators=(",", ":"))
    P(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "combine": combine(sys.argv[2:]); sys.exit(0)
    kw = {}; argv = sys.argv[1:]; k_ = 0
    while k_ < len(argv): kw[argv[k_][2:]] = argv[k_ + 1]; k_ += 2
    if "R" in kw: R_NULL = int(kw["R"])
    spec = {int(x.split("=")[0]): x.split("=")[1] for x in kw["nbeta"].split(",")} if kw.get("nbeta") else None
    run(kw["npz"], float(kw["a"]), kw["tag"], kw.get("ref"), spec, int(kw["nbeta-ref"]) if "nbeta-ref" in kw else (sorted(spec)[0] if spec else None))
