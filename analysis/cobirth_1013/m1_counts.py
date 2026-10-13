"""M1 (cobirth_1013): simultaneous bridge counts N_i(T) and births B_i(T) per target per task-end, their second moments,
Fano factors, pooled histograms vs Poisson / per-target Poisson mixture, pairwise co-birth excess, two nulls (per-candidate
circular shift / permutation; within-task target shuffle), one-sided Chebyshev (Cantelli) bound with n*_i, and time windows.
Box: 1-hidden-layer MLP 784-100-10 on 1200 fixed MNIST inputs, hard leaky ReLU slope a in {0.1, 0.3, 0.5}, seed 0, saved task-end weights.
Task convention here: T = 1..100 with state end0[T-1]; Phase 2 = T 11..100 (90 task ends).
  stoch_bridges.py / p161 / p170 convention: t = T + 1 (t = 2..101, state end0[t-2]); their t >= 10 is T >= 9.
Bridge (lifter) criterion: verbatim copy of stoch_bridges.py bridges_at (P158/P161/P176 loose criterion):
  smooth gate d = a + (1-a) sigma(z/tau), tau = 0.1 sd_n z + 1e-12, e' = K_n * dd/dz, threshold inputs T_i = top-3 of e',
  open set O_i = {n: z_in > 0}, k_i = |O_i|; candidate j != i (alive) is a lifter of i iff |O_j cap T_i| >= 2 and |O_j cap O_i| >= (k_i+1)//2.
Births at T: pairs (i, j) that satisfy the criterion at T but not at T-1 (T >= 2); a target dead at T-1 has no lifters at T-1.
usage: m1_counts.py [a01 a03 a05]   (CPU, 1 thread, float64; no training)"""
import os, sys, json, math, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_1013/m1"
WT = "/home/issan/Projects/claude/wt/cobirth_1013"
OUT_JSON = os.path.join(WT, "results/cobirth_1013/m1_fano.json")
ARMS = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
NB_DIR = os.path.join(ROOT, "review/stage4_1002/block_1003/stage14_1007_h4/agent_bound")
NSTAR_FILES = {"a01": ("p172c_a01_t40_real.json", 39), "a03": ("p172c_a03_t100_real.json", 99), "a05": ("p172c_a05_t100_real.json", 99)}  # (file, T of that state: stoch t40 -> T 39, t100 -> T 99)
NSTAR_CONST = {"a01": dict(min=3.6, med=32.8), "a03": dict(min=12.8, med=35.3), "a05": dict(min=20.0, med=35.3)}  # H4_bound_report per-unit n_beta (arm min / median)
H = 100; NT = 100; PH2 = list(range(11, 101)); WINDOWS = [(11, 40), (41, 70), (71, 100)]; R_NULL = 1000; SEED = 0; MIN_ALIVE = 30; MIN_ALIVE_WIN = 15   # R_NULL: spec asks 200; 1000 gives p-value floor 2/1001
P = lambda *x: print(*x, flush=True)
nan = float("nan")

def bridges_at(O, Dp, Kv):
    """verbatim from stoch_bridges.py bridges_at (rows = target i, cols = candidate j)."""
    alive = O.any(0); k = O.sum(0); need = (k + 1) // 2; Oi = O.astype(np.int32); ov = Oi.T @ Oi
    E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    isb = (thr >= 2) & (ov >= need[:, None]); isb[np.diag_indices(H)] = False; isb[~alive, :] = False; isb[:, ~alive] = False
    return isb, alive, k

def pois_tail(lam, k):
    """P(Pois(lam) >= k), k real -> ceil."""
    k = int(math.ceil(k - 1e-12))
    if k <= 0: return 1.0
    if lam <= 0: return 0.0
    return float(1.0 - sum(math.exp(-lam + x * math.log(lam) - math.lgamma(x + 1)) for x in range(k)))

def pois_pmf(lam, v):
    if lam <= 0: return 1.0 if v == 0 else 0.0
    return math.exp(-lam + v * math.log(lam) - math.lgamma(v + 1))

def cantelli(mu, var, nstar):
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
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        if not np.isfinite(o): return None
        return float(f"{o:.6g}")
    return o

def analyze(kind, IND, aliveT, nstar_vec, tag, rng, kmat):
    """kind 'presence' or 'births'. IND: dict T -> (H,H) bool indicator (lifter / birth). aliveT: (NT, H) bool. Phase 2 per-target stats, nulls, Cantelli."""
    res = {}; per = []; Mats = {}
    cols_all = np.arange(H)
    for i in range(H):
        Ts = [T for T in PH2 if aliveT[T - 1, i]]
        if len(Ts) == 0: per.append(None); continue
        M = np.stack([IND[T][i, :] for T in Ts], 0).astype(np.int8); M = M[:, cols_all != i]  # (n_alive, 99)
        Mats[i] = (Ts, M); st = ind_stats(M); st["i"] = i; st["Ts"] = Ts; per.append(st)
    ok = [st for st in per if st is not None and st["n"] >= MIN_ALIVE]
    res["n_targets_alive_any"] = len([s for s in per if s is not None]); res["n_targets_used"] = len(ok); res["min_alive_tasks"] = MIN_ALIVE
    # identity check delta == var0 - mu + sum p^2
    res["identity_max_abs_err"] = float(max(abs(s["delta"] - s["delta_id"]) for s in ok)) if ok else None
    mu = np.array([s["mu"] for s in ok]); var1 = np.array([s["var1"] for s in ok]); var0 = np.array([s["var0"] for s in ok]); fano = np.array([s["fano"] for s in ok]); mx = np.array([s["max"] for s in ok]); n_i = np.array([s["n"] for s in ok], float)
    delta = np.array([s["delta"] for s in ok]); v1mm = np.array([s["var1_minus_mu"] for s in ok]); sp2 = np.array([s["sp2"] for s in ok])
    res["per_target"] = dict(i=[s["i"] for s in ok], n_alive=[s["n"] for s in ok], mu=mu, var1=var1, var0=var0, fano=fano, max=mx, ge2=[s["ge2"] for s in ok], ge3=[s["ge3"] for s in ok], delta=delta, var1_minus_mu=v1mm, sum_p2=sp2)
    res["fano_quantiles"] = q(fano); res["mu_quantiles"] = q(mu); res["share_fano_gt1"] = float(np.nanmean(fano > 1)) if np.isfinite(fano).any() else None
    res["delta_quantiles"] = q(delta); res["share_delta_gt0"] = float((delta > 0).mean()); res["sum_delta"] = float(delta.sum()); res["sum_var1_minus_mu"] = float(np.nansum(v1mm)); res["sum_sum_p2"] = float(sp2.sum())
    # pooled over target-tasks (all alive target-tasks of used targets)
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
    # ---- null 1: per-candidate circular shift / permutation (per target) ----
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
        # pooled per replicate
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
    # ---- pooled stats by target width k_i(T) at that task ----
    kb_edges = [0, 3, 8, 30, 100, 300, 10 ** 9]; kb_lab = ["1-3", "4-8", "9-30", "31-100", "101-300", ">300"]; kt = {}
    kall = np.concatenate([[kmat[T - 1, s["i"]] for T in s["Ts"]] for s in ok]).astype(float)
    for lo, hi, lab in zip(kb_edges[:-1], kb_edges[1:], kb_lab):
        sel = (kall > lo) & (kall <= hi)
        if sel.sum() == 0: continue
        v = Nall[sel]; m_ = v.mean()
        kt[lab] = dict(n_target_tasks=int(sel.sum()), mean=float(m_), fano=float(v.var(ddof=1) / m_) if m_ > 0 and len(v) > 1 else None, ge2=float((v >= 2).mean()), ge3=float((v >= 3).mean()), max=int(v.max()), poisson_ge2=float(pois_tail(m_, 2)), poisson_ge3=float(pois_tail(m_, 3)))
    res["by_target_width"] = kt
    # ---- null 2: within-task shuffle of targets (keeps same-task pairing & pooled histogram, destroys target identity) ----
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
    # sd of mu across targets under null 2 (target heterogeneity test)
    sdmu2 = np.zeros(R_NULL)
    for r in range(R_NULL):
        Nr = Nmat.copy()
        for row in range(len(PH2)):
            al = np.where(Nr[row] >= 0)[0]; Nr[row, al] = Nr[row, al[rng.permutation(len(al))]]
        mus = np.array([Nr[:, i][Nr[:, i] >= 0].mean() for i in used_idx]); sdmu2[r] = mus.std()
    res["null_target_shuffle"]["null"]["sd_mu_across_targets"] = q(sdmu2, (0.025, 0.5, 0.975)); res["null_target_shuffle"]["p_two_sided"]["sd_mu_across_targets"] = pval2(float(mu.std()), sdmu2)
    res["null_target_shuffle"]["note"] = "mu_i is not invariant under this null; sd(mu_i) across targets compares target heterogeneity with the cross-sectional draw."
    # ---- Cantelli bound per target ----
    cant = {}
    def cant_block(ns_vec, label):
        b = np.array([cantelli(mu[a], var1[a], ns_vec[a]) for a in range(len(ok))]); pz = np.array([pois_tail(mu[a], ns_vec[a]) for a in range(len(ok))])
        emp = np.array([float((ok[a]["N"] >= ns_vec[a]).mean()) for a in range(len(ok))]); n_emp = int(sum((ok[a]["N"] >= ns_vec[a]).sum() for a in range(len(ok))))
        return dict(label=label, nstar=q(ns_vec, (0.5,)), cantelli=dict(max=float(b.max()), median=float(np.median(b)), q90=float(np.quantile(b, .9)), mean=float(b.mean()), union90_max=float(min(1.0, 90 * b.max()))),
                    poisson_indep=dict(max=float(pz.max()), median=float(np.median(pz)), mean=float(pz.mean())), empirical=dict(max=float(emp.max()), mean=float(emp.mean()), n_target_tasks_ge_nstar=n_emp),
                    per_target_cantelli=b, per_target_nstar=ns_vec)
    if nstar_vec is not None:
        ns = np.array([nstar_vec.get(s["i"], nan) for s in ok]); have = np.isfinite(ns)
        if have.any():
            ns_f = np.where(have, ns, np.nanmedian(ns)); cant["per_unit_n_beta"] = cant_block(ns_f, f"per-unit n_beta from p172c (state T={NSTAR_FILES[tag][1]}; {int(have.sum())}/{len(ok)} targets have a value, others filled with median)")
            cant["per_unit_n_beta"]["n_targets_with_value"] = int(have.sum())
    for lab, val in (("arm_min", NSTAR_CONST[tag]["min"]), ("arm_median", NSTAR_CONST[tag]["med"]), ("ten", 10.0)):
        cant[lab] = cant_block(np.full(len(ok), val), f"constant n* = {val}")
    res["cantelli"] = cant
    # ---- time windows ----
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

def main(tags):
    ALL = {"convention": "T = 1..100 is the state end0[T-1]; Phase 2 = T 11..100 (90 task ends). stoch/p161/p170 use t = T+1. Births at T = pairs lifter at T and not at T-1. Dead target (k_i = 0) => N = -1 in npz, excluded from stats.",
           "criterion": "verbatim bridges_at from stoch_bridges.py (P158/P161/P176 loose criterion); smooth gate tau = 0.1 sd(z) + 1e-12; T_i = top-3 of K_n * dd/dz; need = (k_i+1)//2",
           "R_null": R_NULL, "seed": SEED, "min_alive_tasks_per_target": MIN_ALIVE, "arms": {}}
    for tag in tags:
        t0 = time.time(); src, a = ARMS[tag]; dd = np.load(os.path.join(ROOT, src))
        X = load_X(dd["subset"]); N = X.shape[0]; Xt = np.concatenate([X, np.ones((N, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar   # K_n = x_n.xbar + 1 (p161: Kv = Xt @ Xt.mean(0))
        ISB = np.zeros((NT, H, H), bool); ALIVE = np.zeros((NT, H), bool); KK = np.zeros((NT, H), np.int16)
        for T in range(1, NT + 1):
            W = np.concatenate([dd["end0"][T - 1].astype(np.float64), dd["end1"][T - 1].astype(np.float64)[:, None]], 1); Z = Xt @ W.T; tau = 0.1 * Z.std(0) + 1e-12
            s = 0.5 * (1 + np.tanh(0.5 * Z / tau)); O = Z > 0; Dp = (1 - a) * s * (1 - s) / tau
            isb, alive, k = bridges_at(O, Dp, Kv); ISB[T - 1] = isb; ALIVE[T - 1] = alive; KK[T - 1] = k
        BIRTH = np.zeros((NT, H, H), bool); BIRTH[1:] = ISB[1:] & ~ISB[:-1]; BIRTH[0] = False
        Ncnt = np.where(ALIVE, ISB.sum(2), -1).astype(np.int16); Bcnt = np.where(ALIVE, BIRTH.sum(2), -1).astype(np.int16); Bcnt[0] = -1
        Dcnt = np.zeros((NT, H), np.int16); Dcnt[1:] = (ISB[:-1] & ~ISB[1:]).sum(2); Dcnt = np.where(ALIVE, Dcnt, -1).astype(np.int16); Dcnt[0] = -1  # deaths of pairs (lifter at T-1, not at T), for the ledger
        P(f"[{tag}] states + bridges done ({time.time() - t0:.1f}s); alive units per T: T1 {ALIVE[0].sum()} T11 {ALIVE[10].sum()} T50 {ALIVE[49].sum()} T100 {ALIVE[99].sum()}")
        # ---- Step 1: reproduction in stoch convention (t = T+1) ----
        rows_stoch = int(sum(ISB[T - 1].sum() for T in range(9, 100)))                       # stoch rows: t = 10..100 -> T = 9..99
        births_stoch = int(sum(BIRTH[T - 1][ALIVE[T - 1]].sum() for T in range(10, 100)))     # stoch births t = 11..100 -> T = 10..99
        recs = []                                                                              # stoch (t,i) with n_t >= 1 and t+1 <= 100: t = 10..99 -> T = 9..98
        for T in range(9, 99):
            for i in np.where(ALIVE[T - 1])[0]:
                n_t = int(ISB[T - 1][i].sum())
                if n_t >= 1:
                    S = int((ISB[T - 1][i] & ISB[T][i]).sum()); n1 = int(ISB[T][i].sum()); recs.append((n_t, S, n1 - S, n1))
        rec = np.array(recs)
        repro = dict(rows_t10_100=rows_stoch, births_t11_100=births_stoch, target_tasks_nt_ge1=len(recs), nt_mean=float(rec[:, 0].mean()), nt_max=int(rec[:, 0].max()), S_mean=float(rec[:, 1].mean()), S_max=int(rec[:, 1].max()),
                     B_mean=float(rec[:, 2].mean()), B_max=int(rec[:, 2].max()), n1_mean=float(rec[:, 3].mean()), n1_max=int(rec[:, 3].max()),
                     published={"a01": dict(rows=1511, births=1399, tt=1078, nt_mean=1.38, nt_max=7, S_mean=0.088, S_max=3, B_mean=0.198, B_max=6, n1_mean=0.29, n1_max=7),
                                "a03": dict(rows=2836, births=2557, tt=1840, nt_mean=1.53, nt_max=14, S_mean=0.128, S_max=3, B_mean=0.484, B_max=11, n1_mean=0.61, n1_max=11),
                                "a05": dict(rows=7712, births=6649, tt=4685, nt_mean=1.64, nt_max=10, S_mean=0.211, S_max=4, B_mean=0.783, B_max=10, n1_mean=0.99, n1_max=10)}[tag])
        pub = repro["published"]; repro["match"] = bool(rows_stoch == pub["rows"] and births_stoch == pub["births"] and len(recs) == pub["tt"] and rec[:, 0].max() == pub["nt_max"] and rec[:, 2].max() == pub["B_max"] and abs(rec[:, 2].mean() - pub["B_mean"]) < 6e-4 and abs(rec[:, 0].mean() - pub["nt_mean"]) < 6e-3)
        P(f"[{tag}] REPRO (stoch convention t=T+1): rows t10..100 {rows_stoch} (pub {pub['rows']}) | births t11..100 {births_stoch} (pub {pub['births']}) | target-tasks n_t>=1 {len(recs)} (pub {pub['tt']}) | n_t mean {rec[:,0].mean():.3f} max {rec[:,0].max()} (pub {pub['nt_mean']}/{pub['nt_max']}) | S mean {rec[:,1].mean():.3f} max {rec[:,1].max()} | B mean {rec[:,2].mean():.3f} max {rec[:,2].max()} (pub {pub['B_mean']}/{pub['B_max']}) | n1 mean {rec[:,3].mean():.3f} max {rec[:,3].max()} | MATCH {repro['match']}")
        # ---- per-task ledger ----
        ledger = dict(T=list(range(1, NT + 1)), n_alive=ALIVE.sum(1).tolist(), pairs=[int(ISB[T - 1].sum()) for T in range(1, NT + 1)], births=[int(BIRTH[T - 1].sum()) if T > 1 else None for T in range(1, NT + 1)],
                      max_N=[int(Ncnt[T - 1].max()) for T in range(1, NT + 1)], k_median_alive=[float(np.median(KK[T - 1][ALIVE[T - 1]])) for T in range(1, NT + 1)])
        # ---- n* per unit ----
        nstar_vec = None
        try:
            jd = json.load(open(os.path.join(NB_DIR, NSTAR_FILES[tag][0]))); nstar_vec = {int(r["i"]): float(r["n_b"]) for r in jd["real"]}
        except Exception as e: P(f"[{tag}] n_beta file not read: {e}")
        rng = np.random.default_rng(SEED)
        IND_P = {T: ISB[T - 1] for T in range(1, NT + 1)}; IND_B = {T: BIRTH[T - 1] for T in range(1, NT + 1)}
        resP, perP = analyze("presence", IND_P, ALIVE, nstar_vec, tag, rng, KK); P(f"[{tag}] presence analysis done ({time.time() - t0:.1f}s)")
        resB, perB = analyze("births", IND_B, ALIVE, nstar_vec, tag, rng, KK); P(f"[{tag}] births analysis done ({time.time() - t0:.1f}s)")
        # Phase-2 headline print
        for nm, r in (("presence", resP), ("births", resB)):
            po = r["pooled"]; fq = r["fano_quantiles"]; n1 = r["null_candidate_series"]["shift"]; n2 = r["null_target_shuffle"]
            P(f"[{tag}] {nm}: targets used {r['n_targets_used']} | pooled mean {po['mean']:.4f} var {po['var']:.4f} Fano {po['fano']:.3f} (within {po['fano_within']:.3f} between {po['fano_between']:.3f}) max {po['max']} | P>=2 {po['ge2']:.4f} (Pois {po['poisson_ge2']:.4f}, mix {po['mixture_ge2']:.4f}) P>=3 {po['ge3']:.5f} (Pois {po['poisson_ge3']:.5f}, mix {po['mixture_ge3']:.5f})")
            P(f"[{tag}] {nm}: per-target Fano q10 {fq['0.1']:.3f} q50 {fq['0.5']:.3f} q90 {fq['0.9']:.3f} max {fq['max']:.3f} share>1 {r['share_fano_gt1']:.3f} | sum delta {r['sum_delta']:.3f} (sum var1-mu {r['sum_var1_minus_mu']:.3f}) share delta>0 {r['share_delta_gt0']:.3f} | identity err {r['identity_max_abs_err']:.1e}")
            P(f"[{tag}] {nm}: null1 shift: median Fano obs {n1['observed']['median_fano']:.3f} null [{n1['null']['median_fano']['0.025']:.3f},{n1['null']['median_fano']['0.975']:.3f}] p {n1['p_two_sided']['median_fano']:.3f} | max N obs {n1['observed']['max_N']} null [{n1['null']['max_N']['0.025']:.0f},{n1['null']['max_N']['0.975']:.0f}] p {n1['p_two_sided']['max_N']:.3f} | pooled P>=2 obs {n1['observed']['pooled_ge2']:.4f} null [{n1['null']['pooled_ge2']['0.025']:.4f},{n1['null']['pooled_ge2']['0.975']:.4f}] p {n1['p_two_sided']['pooled_ge2']:.3f} | per-target p_fano<.05 {n1['per_target']['n_p_fano_lt05']}/{n1['per_target']['n_fano_tested']} p_max<.05 {n1['per_target']['n_p_max_lt05']}")
            n3 = r["null_candidate_series"]["strat"]
            P(f"[{tag}] {nm}: null1b strat(k_i): median Fano obs {n3['observed']['median_fano']:.3f} null [{n3['null']['median_fano']['0.025']:.3f},{n3['null']['median_fano']['0.975']:.3f}] p {n3['p_two_sided']['median_fano']:.3f} | max N null [{n3['null']['max_N']['0.025']:.0f},{n3['null']['max_N']['0.975']:.0f}] p {n3['p_two_sided']['max_N']:.3f} | pooled P>=2 null [{n3['null']['pooled_ge2']['0.025']:.4f},{n3['null']['pooled_ge2']['0.975']:.4f}] p {n3['p_two_sided']['pooled_ge2']:.3f} | per-target p_fano<.05 {n3['per_target']['n_p_fano_lt05']}/{n3['per_target']['n_fano_tested']}")
            for lab, v in r["by_target_width"].items():
                P(f"[{tag}] {nm}: k_i in {lab}: target-tasks {v['n_target_tasks']} mean {v['mean']:.4f} Fano {v['fano'] if v['fano'] is None else round(v['fano'], 3)} P>=2 {v['ge2']:.4f} (Pois {v['poisson_ge2']:.4f}) P>=3 {v['ge3']:.5f} (Pois {v['poisson_ge3']:.5f}) max {v['max']}")
            P(f"[{tag}] {nm}: null2 target-shuffle: median Fano obs {n2['observed']['median_fano']:.3f} null [{n2['null']['median_fano']['0.025']:.3f},{n2['null']['median_fano']['0.975']:.3f}] p {n2['p_two_sided']['median_fano']:.3f} | sd(mu_i) obs {n2['observed']['sd_mu_across_targets']:.4f} null [{n2['null']['sd_mu_across_targets']['0.025']:.4f},{n2['null']['sd_mu_across_targets']['0.975']:.4f}] p {n2['p_two_sided']['sd_mu_across_targets']:.3f} | per-target p_fano<.05 {n2['per_target']['n_p_fano_lt05']}/{n2['per_target']['n_fano_tested']}")
            for lab, c in r["cantelli"].items():
                P(f"[{tag}] {nm}: Cantelli [{lab}] n* med {c['nstar']['0.5']:.1f}: bound max {c['cantelli']['max']:.3e} median {c['cantelli']['median']:.3e} (x90 union max {c['cantelli']['union90_max']:.3e}) | Poisson-indep max {c['poisson_indep']['max']:.3e} median {c['poisson_indep']['median']:.3e} | empirical freq max {c['empirical']['max']:.4f} n target-tasks >= n* {c['empirical']['n_target_tasks_ge_nstar']}")
            for w, v in r["windows"].items():
                P(f"[{tag}] {nm}: window {w}: targets {v['n_targets']} mean mu {v['mean_mu']:.4f} median Fano {v['median_fano']:.3f} pooled Fano {v['pooled_fano']:.3f} P>=2 {v['pooled_ge2']:.4f} max {v['max']}")
        # ---- save ----
        np.savez_compressed(os.path.join(RAW, f"counts_{tag}.npz"), N=Ncnt, B=Bcnt, D=Dcnt, alive=ALIVE, k=KK, isb=ISB, birth=BIRTH,
                            T=np.arange(1, NT + 1), note=np.array(ALL["convention"] + " | arrays [T-1, i] (and [T-1, i, j] for isb/birth: row i target, col j candidate); a=" + str(a)))
        ALL["arms"][tag] = dict(a=a, src=src, repro=repro, ledger=ledger, presence=resP, births=resB, n_beta_state_T=NSTAR_FILES[tag][1], elapsed_s=time.time() - t0)
        P(f"[{tag}] saved ({time.time() - t0:.1f}s)")
    json.dump(rnd(ALL), open(OUT_JSON, "w"), indent=0, separators=(",", ":"))
    json.dump(rnd(ALL), open(os.path.join(RAW, "m1_fano_full.json"), "w"), indent=1)
    P(f"wrote {OUT_JSON} and {RAW}/m1_fano_full.json")

if __name__ == "__main__":
    main(sys.argv[1:] or ["a01", "a03", "a05"])
