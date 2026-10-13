"""elu_counts.py (cobirth_seeds_1013, TASK C, agent tag 'elu'): generalization of cobirth_1013/m1_counts.py (bridge counts,
Fano, nulls 1/1b/2, Cantelli) + the cobirth_1013/m3_hubs.py section (4) variance decomposition (threshold pool C_i, width),
to any saved-weights npz and to a choice of gate convention for the 'threshold inputs'.

Box: 784-100-10 MLP, 1200 fixed MNIST inputs, CE, Adam 1e-3, batch 16, 4000 updates/task, iid uniform 10-class labels per task,
100 tasks.  Saved task-end weights only (no training).

Task convention (M1): T = 1..100, state = end0[T-1]; Phase 2 = T 11..100 (90 task ends).
  stoch_bridges.py / p161 / p170 / p172c use t = T + 1 (their 'task' t = end0[t-2]); the --repro check is done in that convention
  and converted explicitly (stoch rows t = 10..100 -> T = 9..99; births t = 11..100 -> T = 10..99).
  m3_hubs.py uses t = T + 1 with phase 2 = t >= 10 (T 9..100, 92 states); --decomp-phases 9-100 reproduces that.

Bridge (lifter) criterion: bridges_at() is the verbatim logic of stoch_bridges.py bridges_at (P158/P161/P176 loose criterion),
  returning in addition the threshold sets T and the counts thr (the M3 copy did the same; no logic change):
  e'_n = K_n * Dp_n (Dp = derivative of the gate), threshold inputs T_i = top-3 of e', open set O_i = {n: z_in > 0}, k_i = |O_i|;
  candidate j != i (both alive, k >= 1) is a lifter of i iff |O_j cap T_i| >= 2 and |O_j cap O_i| >= (k_i+1)//2.
Gate conventions for Dp (a CONVENTION, not a derivation):
  leaky : smooth gate d = a + (1-a) sigma(z/tau), tau = 0.1 sd_n z + 1e-12, Dp = (1-a) s(1-s)/tau   (verbatim, slope --a)
  E1    : the same smoothed step with a = 0, Dp = s(1-s)/tau.  [identity] the top-3 of K_n * Dp does not depend on a (a < 1),
          so E1 selects the same T_i as the leaky code would on these weights: inputs nearest z = 0 (either side) weighted by K_n.
  E2    : ELU gate derivative phi''(z) = e^z 1[z < 0]: Dp = exp(z - c_i) 1[z < 0] with c_i = max_{n: z_in<0} z_in (a per-unit constant
          factor e^{-c_i}, which leaves the per-unit ranking unchanged and avoids underflow).  T_i = closed inputs nearest 0 weighted by K_n.
Births at T: pairs lifter at T and not at T-1 (T >= 2); a target dead at T-1 has no lifters at T-1.
usage:
  elu_counts.py run --npz <path rel. to push_direction_proof_0930 or abs> --gate leaky|E1|E2 [--a 0.1] --tag NAME
                    [--R 1000] [--repro a01|a03|a05] [--nstar-file F --nstar-T T] [--decomp-phases 11-100[,9-100]]
  elu_counts.py compare --pairs ELU_s0_E1:ELU_s0_E2,ELU_s1_E1:ELU_s1_E2      (E1 vs E2 overlap of T_i, pairs, N)
  elu_counts.py post --tags t1,t2,...                                        (windowed break-even n*, no RNG)
  elu_counts.py combine --tags t1,t2,...                                     (-> results/cobirth_seeds_1013/elu.json)
CPU, 1 thread, float64."""
import os, sys, json, math, time, argparse
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/elu"
WT = "/home/issan/Projects/claude/wt/cobirth_seeds_1013"
OUT_JSON = os.path.join(WT, "results/cobirth_seeds_1013/elu.json")
NB_DIR = os.path.join(ROOT, "review/stage4_1002/block_1003/stage14_1007_h4/agent_bound")
H = 100; NT = 100; WINDOWS = [(11, 40), (41, 70), (71, 100)]; SEED = 0; MIN_ALIVE = 30; MIN_ALIVE_WIN = 15
NSTAR_GRID = [2.0, 3.0, 3.6, 4.0, 5.0, 6.0, 8.0, 10.0, 12.8, 15.0, 20.0, 35.3]   # 3.6 / 12.8 / 20.0 / 35.3 = leaky p172c n_beta arm min/median (NOT ELU)
BREAKEVEN_C = (0.14, 0.08, 0.32, 0.19)                                               # c = 2.8 thresholds 14/11/8 %, c = 1: 32/26/19 % (cumulative_push §3)
KBINS_FINE = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 1200)]  # m3_hubs.py
R_MC_U = 20   # m3_hubs.py null U replicates
os.makedirs(RAW, exist_ok=True)
nan = float("nan")
LOGF = None
def P(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True)
    if LOGF is not None: LOGF.write(s + "\n"); LOGF.flush()

PUBLISHED = {"a01": dict(rows=1511, births=1399, tt=1078, nt_mean=1.38, nt_max=7, S_mean=0.088, S_max=3, B_mean=0.198, B_max=6, n1_mean=0.29, n1_max=7),
             "a03": dict(rows=2836, births=2557, tt=1840, nt_mean=1.53, nt_max=14, S_mean=0.128, S_max=3, B_mean=0.484, B_max=11, n1_mean=0.61, n1_max=11),
             "a05": dict(rows=7712, births=6649, tt=4685, nt_mean=1.64, nt_max=10, S_mean=0.211, S_max=4, B_mean=0.783, B_max=10, n1_mean=0.99, n1_max=10)}

# ---------------------------------------------------------------------------------------------------------------- criterion
def bridges_at(O, Dp, Kv):
    """verbatim logic of stoch_bridges.py bridges_at (rows = target i, cols = candidate j); also returns T (3, H) and thr (H, H)."""
    alive = O.any(0); k = O.sum(0); need = (k + 1) // 2; Oi = O.astype(np.int32); ov = Oi.T @ Oi
    E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    isb = (thr >= 2) & (ov >= need[:, None]); isb[np.diag_indices(H)] = False; isb[~alive, :] = False; isb[:, ~alive] = False
    return isb, alive, k, T, thr, E

def gate_dp(Z, gate, a):
    if gate in ("leaky", "E1"):
        aa = a if gate == "leaky" else 0.0
        tau = 0.1 * Z.std(0) + 1e-12; s = 0.5 * (1 + np.tanh(0.5 * Z / tau)); return (1 - aa) * s * (1 - s) / tau
    if gate == "E2":
        neg = Z < 0; c = np.where(neg.any(0), np.where(neg, Z, -np.inf).max(0), 0.0)
        return np.where(neg, np.exp(np.minimum(Z - c[None, :], 0.0)), 0.0)
    raise ValueError(gate)

# ---------------------------------------------------------------------------------------------------------------- M1 helpers (copied)
def pois_tail(lam, k):
    """P(Pois(lam) >= k), k real -> ceil."""
    k = int(math.ceil(k - 1e-12))
    if k <= 0: return 1.0
    if lam <= 0: return 0.0
    return float(max(0.0, 1.0 - sum(math.exp(-lam + x * math.log(lam) - math.lgamma(x + 1)) for x in range(k))))

def pois_pmf(lam, v):
    if lam <= 0: return 1.0 if v == 0 else 0.0
    return math.exp(-lam + v * math.log(lam) - math.lgamma(v + 1))

def cantelli(mu, var, nstar):
    if not np.isfinite(var) or nstar <= mu: return 1.0
    return float(var / (var + (nstar - mu) ** 2))

def ind_stats(M):
    n = M.shape[0]; Nn = M.sum(1).astype(float); mu = float(Nn.mean()); var0 = float(Nn.var()); var1 = float(Nn.var(ddof=1)) if n > 1 else nan
    p = M.mean(0); sp2 = float((p ** 2).sum())
    G = (M.T.astype(float) @ M.astype(float)) / n - np.outer(p, p); delta = float(G.sum() - np.trace(G))
    return dict(n=n, mu=mu, var0=var0, var1=var1, fano=(var1 / mu if mu > 0 else nan), max=int(Nn.max()), ge2=float((Nn >= 2).mean()), ge3=float((Nn >= 3).mean()),
                sp2=sp2, delta=delta, delta_id=var0 - mu + sp2, var1_minus_mu=(var1 - mu) if n > 1 else nan, hist=np.bincount(Nn.astype(int)).tolist(), N=Nn)

def null_cols(M, R, rng, mode):
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
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (float, np.floating)):
        if not np.isfinite(o): return None
        return float(f"{o:.6g}")
    return o

def full(o):
    if isinstance(o, dict): return {str(k): full(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [full(v) for v in o]
    if isinstance(o, np.ndarray): return full(o.tolist())
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    return o

# m3 helpers (copied)
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

# ---------------------------------------------------------------------------------------------------------------- M1 analyze (copied; Cantelli generalized)
def analyze(kind, IND, aliveT, nstar_vec, nstar_T, rng, kmat, R_NULL, PH2):
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
        with np.errstate(all="ignore"):
            medF = np.nanmedian(arr[:, :, 0], axis=0); meanF = np.nanmean(arr[:, :, 0], axis=0)
        maxN = arr[:, :, 1].max(axis=0)
        ge2 = (arr[:, :, 2] * n_i[:, None]).sum(0) / n_i.sum(); ge3 = (arr[:, :, 3] * n_i[:, None]).sum(0) / n_i.sum(); sdel = arr[:, :, 4].sum(0)
        obs = dict(median_fano=float(np.nanmedian(fano)), mean_fano=float(np.nanmean(fano)), max_N=int(mx.max()), pooled_ge2=res["pooled"]["ge2"], pooled_ge3=res["pooled"]["ge3"], sum_delta=float(delta.sum()))
        with np.errstate(all="ignore"):
            nfa = int(np.nansum(fano > np.nanmedian(arr[:, :, 0], axis=1))); q025 = np.nanquantile(arr[:, :, 0], 0.025, axis=1); q975 = np.nanquantile(arr[:, :, 0], 0.975, axis=1)
        nulls[mode] = dict(R=R_NULL, observed=obs,
                           null=dict(median_fano=q(medF, (0.025, 0.5, 0.975)), mean_fano=q(meanF, (0.025, 0.5, 0.975)), max_N=q(maxN, (0.025, 0.5, 0.975)), pooled_ge2=q(ge2, (0.025, 0.5, 0.975)), pooled_ge3=q(ge3, (0.025, 0.5, 0.975)), sum_delta=q(sdel, (0.025, 0.5, 0.975))),
                           p_two_sided=dict(median_fano=pval2(obs["median_fano"], medF), mean_fano=pval2(obs["mean_fano"], meanF), max_N=pval2(obs["max_N"], maxN), pooled_ge2=pval2(obs["pooled_ge2"], ge2), pooled_ge3=pval2(obs["pooled_ge3"], ge3), sum_delta=pval2(obs["sum_delta"], sdel)),
                           per_target=dict(p_fano=pf, p_max=pm, p_delta=pd, n_p_fano_lt05=int(np.nansum(pf < 0.05)), n_p_max_lt05=int(np.nansum(pm < 0.05)), n_p_delta_lt05=int(np.nansum(pd < 0.05)),
                                           n_fano_above_null_median=nfa, n_fano_tested=int(np.isfinite(pf).sum()),
                                           null_fano_q025_med=q(q025, (0.5,)), null_fano_q975_med=q(q975, (0.5,)),
                                           n_max_above_null_q975=int((mx > np.quantile(arr[:, :, 1], 0.975, axis=1)).sum())))
    nulls["strat"]["note"] = f"independent permutation of each candidate's series within {NSTRATA} quantile strata of the target's width k_i(T) over its alive Phase-2 tasks"
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
    with np.errstate(all="ignore"):
        medF2 = np.nanmedian(fanoN, axis=1); sdF2 = np.nanstd(fanoN, axis=1)
    obs_sd = float(np.nanstd(fano))
    res["null_target_shuffle"] = dict(R=R_NULL, observed=dict(median_fano=float(np.nanmedian(fano)), sd_fano_across_targets=obs_sd, sd_mu_across_targets=float(mu.std()), max_of_per_target_max=int(mx.max())),
                                      null=dict(median_fano=q(medF2, (0.025, 0.5, 0.975)), sd_fano_across_targets=q(sdF2, (0.025, 0.5, 0.975))),
                                      p_two_sided=dict(median_fano=pval2(float(np.nanmedian(fano)), medF2), sd_fano_across_targets=pval2(obs_sd, sdF2)),
                                      per_target=dict(p_fano=pf2, p_max=pm2, n_p_fano_lt05=int(np.nansum(pf2 < 0.05)), n_p_max_lt05=int(np.nansum(pm2 < 0.05)), n_fano_tested=int(np.isfinite(pf2).sum())), note="")
    sdmu2 = np.zeros(R_NULL)
    for r in range(R_NULL):
        Nr = Nmat.copy()
        for row in range(len(PH2)):
            al = np.where(Nr[row] >= 0)[0]; Nr[row, al] = Nr[row, al[rng.permutation(len(al))]]
        mus = np.array([Nr[:, i][Nr[:, i] >= 0].mean() for i in used_idx]); sdmu2[r] = mus.std()
    res["null_target_shuffle"]["null"]["sd_mu_across_targets"] = q(sdmu2, (0.025, 0.5, 0.975)); res["null_target_shuffle"]["p_two_sided"]["sd_mu_across_targets"] = pval2(float(mu.std()), sdmu2)
    res["null_target_shuffle"]["note"] = "mu_i is not invariant under this null; sd(mu_i) across targets compares target heterogeneity with the cross-sectional draw."
    # ---- Cantelli: per-unit n_beta (if given), the n* grid, and the break-even n* ----
    cant = {}
    def cant_block(ns_vec, label):
        b = np.array([cantelli(mu[a], var1[a], ns_vec[a]) for a in range(len(ok))]); pz = np.array([pois_tail(mu[a], ns_vec[a]) for a in range(len(ok))])
        emp = np.array([float((ok[a]["N"] >= ns_vec[a]).mean()) for a in range(len(ok))]); n_emp = int(sum((ok[a]["N"] >= ns_vec[a]).sum() for a in range(len(ok))))
        return dict(label=label, nstar=q(ns_vec, (0.5,)), cantelli=dict(max=float(b.max()), median=float(np.median(b)), q90=float(np.quantile(b, .9)), mean=float(b.mean()), sum_over_targets=float(b.sum()), union90_max=float(min(1.0, 90 * b.max())), argmax_target=int(ok[int(np.argmax(b))]["i"])),
                    poisson_indep=dict(max=float(pz.max()), median=float(np.median(pz)), mean=float(pz.mean())), empirical=dict(max=float(emp.max()), mean=float(emp.mean()), n_target_tasks_ge_nstar=n_emp, n_targets_with_any=int((emp > 0).sum())),
                    per_target_cantelli=b, per_target_nstar=ns_vec)
    if nstar_vec is not None:
        ns = np.array([nstar_vec.get(s["i"], nan) for s in ok]); have = np.isfinite(ns)
        if have.any():
            ns_f = np.where(have, ns, np.nanmedian(ns)); cant["per_unit_n_beta"] = cant_block(ns_f, f"per-unit n_beta from p172c (state T={nstar_T}; {int(have.sum())}/{len(ok)} targets have a value, others filled with median)")
            cant["per_unit_n_beta"]["n_targets_with_value"] = int(have.sum())
    for val in NSTAR_GRID: cant[f"n{val:g}"] = cant_block(np.full(len(ok), val), f"constant n* = {val}")
    res["cantelli"] = cant
    be = {}
    sd1 = np.sqrt(np.maximum(var1, 0.0))
    for c in BREAKEVEN_C:
        nb = mu + sd1 * math.sqrt((1 - c) / c)     # Cantelli bound = c exactly at n* = mu + sigma sqrt((1-c)/c)  [algebra]
        a_m = int(np.argmax(nb))
        be[f"c{c:g}"] = dict(c=c, factor=math.sqrt((1 - c) / c), max=float(nb.max()), median=float(np.median(nb)), q90=float(np.quantile(nb, 0.9)), argmax_target=int(ok[a_m]["i"]),
                             argmax_mu=float(mu[a_m]), argmax_sd=float(sd1[a_m]), per_target=nb,
                             pooled_moment_version=float(lam + math.sqrt(float(Nall.var(ddof=1))) * math.sqrt((1 - c) / c)),
                             n_target_tasks_N_ge_breakeven=int(sum((ok[a]["N"] >= nb[a]).sum() for a in range(len(ok)))))
    res["breakeven_nstar"] = be
    win = {}
    for (lo, hi) in WINDOWS:
        rows = []
        for s in ok:
            sel = np.array([(lo <= T <= hi) for T in s["Ts"]])
            if sel.sum() < MIN_ALIVE_WIN: continue
            v = s["N"][sel]; m_ = v.mean(); rows.append((m_, v.var(ddof=1), v.var(ddof=1) / m_ if m_ > 0 else nan, v.max(), v))
        if not rows: continue
        allv = np.concatenate([r[4] for r in rows]); mus = np.array([r[0] for r in rows]); fs = np.array([r[2] for r in rows])
        win[f"T{lo}-{hi}"] = dict(n_targets=len(rows), mean_mu=float(mus.mean()), median_fano=float(np.nanmedian(fs)) if np.isfinite(fs).any() else None, mean_fano=float(np.nanmean(fs)) if np.isfinite(fs).any() else None,
                                  pooled_mean=float(allv.mean()), pooled_fano=float(allv.var(ddof=1) / allv.mean()) if allv.mean() > 0 else None, pooled_ge2=float((allv >= 2).mean()), pooled_ge3=float((allv >= 3).mean()), max=int(allv.max()), n_target_tasks=int(len(allv)))
    res["windows"] = win
    return res, per

# ---------------------------------------------------------------------------------------------------------------- M3 §4 decomposition (copied logic)
def decomposition(ISB, ALIVE, KK, TH, Ts_list, OO, rng_u, N_in):
    """Records = all alive (i, T) for T in Ts_list.  C_i = #{j != i alive: |O_j cap T_i| >= 2}.  var_decomp groups as m3_hubs.py (4).
    Null U: each candidate's open set -> uniform random subset of its size k_j (target's T_i, O_i actual), R_MC_U reps."""
    recs = []
    for T in Ts_list:
        isb = ISB[T - 1]; alive = ALIVE[T - 1]; k = KK[T - 1].astype(int); thr = TH[T - 1]; O = OO[T - 1]; Tset = TSETS[T - 1]; need = (k + 1) // 2
        Nobs = isb.sum(1); pool = (thr >= 2) & alive[None, :]; pool[np.diag_indices(H)] = False; Cobs = pool.sum(1)
        NU = np.zeros((R_MC_U, H)); CU = np.zeros((R_MC_U, H))
        for rep in range(R_MC_U):
            U = rng_u.random((N_in, H)); ranks = np.argsort(np.argsort(U, axis=0), axis=0); On = ranks < k[None, :]
            thr_n = np.stack([On[Tset[:, i], :].sum(0) for i in range(H)], 0); ov_n = O.astype(np.int32).T @ On.astype(np.int32)
            isb_n = (thr_n >= 2) & (ov_n >= need[:, None]); isb_n[np.diag_indices(H)] = False; isb_n[~alive, :] = False; isb_n[:, ~alive] = False
            NU[rep] = isb_n.sum(1); pool_n = (thr_n >= 2) & alive[None, :]; pool_n[np.diag_indices(H)] = False; CU[rep] = pool_n.sum(1)
        for i in np.where(alive)[0]:
            recs.append((T, int(i), int(k[i]), int(Nobs[i]), int(Cobs[i]), float(NU[:, i].mean()), float(NU[:, i].var()), float(CU[:, i].mean()), float(CU[:, i].var())))
    A = np.array(recs, float); tt = A[:, 0].astype(int); ii = A[:, 1].astype(int); kk = A[:, 2].astype(int); NN = A[:, 3]; CC = A[:, 4]; LU = A[:, 5]; LUv = A[:, 6]; CUm = A[:, 7]; CUv = A[:, 8]
    kbf = np.array([kbin_of(k, KBINS_FINE) for k in kk])
    vd = dict(n_records=len(recs), phase=f"T{Ts_list[0]}-{Ts_list[-1]}",
              N_by_unit=var_decomp(NN, ii), N_by_kbin_fine=var_decomp(NN, kbf), N_by_unit_x_kbin_fine=var_decomp(NN, ii * 100 + kbf), N_by_task=var_decomp(NN, tt),
              N_by_pool=var_decomp(NN, np.minimum(CC, 30)), N_by_pool_x_kbin_fine=var_decomp(NN, kbf * 100 + np.minimum(CC, 30)),
              C_by_unit=var_decomp(CC, ii), C_by_kbin_fine=var_decomp(CC, kbf), C_by_task=var_decomp(CC, tt))
    vd["pool"] = dict(C_mean=float(CC.mean()), C_var=float(CC.var()), C_fano=(float(CC.var() / CC.mean()) if CC.mean() > 0 else None), C_max=int(CC.max()),
                      CU_mean=float(CUm.mean()), CU_hetero_var=float(CUm.var()), CU_within_var=float(CUv.mean()), CU_total_fano=(float((CUm.var() + CUv.mean()) / CUm.mean()) if CUm.mean() > 0 else None),
                      ratio_C_CU=(float(CC.sum() / CUm.sum()) if CUm.sum() > 0 else None), pass_rate_N_over_C=(float(NN.sum() / CC.sum()) if CC.sum() > 0 else None),
                      pass_rate_null=(float(LU.sum() / CUm.sum()) if CUm.sum() > 0 else None), ratio_obs_lamU=(float(NN.sum() / LU.sum()) if LU.sum() > 0 else None),
                      sp_N_C=spearman(NN, CC), sp_C_k=spearman(CC, kk), sp_N_k=spearman(NN, kk), R_MC=R_MC_U,
                      NU_total_fano=(float((LU.var() + LUv.mean()) / LU.mean()) if LU.mean() > 0 else None), lamU_mean=float(LU.mean()))
    return vd

TSETS = None

# ---------------------------------------------------------------------------------------------------------------- run
def run(args):
    global LOGF, TSETS
    tag = args.tag; LOGF = open(os.path.join(RAW, f"{tag}.log"), "w"); t0 = time.time()
    src = args.npz if os.path.isabs(args.npz) else os.path.join(ROOT, args.npz); dd = np.load(src)
    a = float(args.a) if args.gate == "leaky" else 0.0
    PH2 = list(range(int(args.phase.split("-")[0]), int(args.phase.split("-")[1]) + 1))
    P(f"[{tag}] npz {src} act {str(dd['act']) if 'act' in dd.files else '?'} seed {int(dd['seed']) if 'seed' in dd.files else '?'} | gate {args.gate} a {a} | phase T{PH2[0]}-{PH2[-1]} | R_null {args.R}")
    X = load_X(dd["subset"]); N_in = X.shape[0]; Xt = np.concatenate([X, np.ones((N_in, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar
    P(f"[{tag}] K_n = x_n.xbar + 1: min {Kv.min():.4f} median {np.median(Kv):.4f} max {Kv.max():.4f} (all > 0: {bool((Kv > 0).all())})")
    ISB = np.zeros((NT, H, H), bool); ALIVE = np.zeros((NT, H), bool); KK = np.zeros((NT, H), np.int16); TS = np.zeros((NT, 3, H), np.int16); TH = np.zeros((NT, H, H), np.int16); OO = np.zeros((NT, N_in, H), bool)
    degenerate = np.zeros(NT, int); ties3 = np.zeros(NT, int); aT = np.zeros((NT, H), np.int8)
    for T in range(1, NT + 1):
        W = np.concatenate([dd["end0"][T - 1].astype(np.float64), dd["end1"][T - 1].astype(np.float64)[:, None]], 1); Z = Xt @ W.T
        Dp = gate_dp(Z, args.gate, a); O = Z > 0
        isb, alive, k, Tset, thr, E = bridges_at(O, Dp, Kv); ISB[T - 1] = isb; ALIVE[T - 1] = alive; KK[T - 1] = k; TS[T - 1] = Tset; TH[T - 1] = thr; OO[T - 1] = O
        Es = -np.sort(-E, axis=0)
        degenerate[T - 1] = int(((Es[2] <= 0) & alive).sum()); ties3[T - 1] = int(((Es[2] == Es[3]) & alive).sum())
        aT[T - 1] = np.array([O[Tset[:, i], i].sum() for i in range(H)])
    TSETS = TS
    BIRTH = np.zeros((NT, H, H), bool); BIRTH[1:] = ISB[1:] & ~ISB[:-1]; BIRTH[0] = False
    Ncnt = np.where(ALIVE, ISB.sum(2), -1).astype(np.int16); Bcnt = np.where(ALIVE, BIRTH.sum(2), -1).astype(np.int16); Bcnt[0] = -1
    Dcnt = np.zeros((NT, H), np.int16); Dcnt[1:] = (ISB[:-1] & ~ISB[1:]).sum(2); Dcnt = np.where(ALIVE, Dcnt, -1).astype(np.int16); Dcnt[0] = -1
    ph = np.array(PH2) - 1
    P(f"[{tag}] states + bridges done ({time.time() - t0:.1f}s); alive units per T: T1 {ALIVE[0].sum()} T11 {ALIVE[10].sum()} T50 {ALIVE[49].sum()} T100 {ALIVE[99].sum()} | phase-2 alive unit-tasks {int(ALIVE[ph].sum())} (dead k=0: {int((~ALIVE[ph]).sum())})")
    P(f"[{tag}] T_i diagnostics (phase 2, alive): 3rd-largest e' <= 0 (degenerate top-3) {int(degenerate[ph].sum())} | exact tie at 3rd/4th {int(ties3[ph].sum())} | overlap a_i = |T_i cap O_i| distribution {np.bincount(aT[ph][ALIVE[ph]].astype(int), minlength=4).tolist()}")
    kph = KK[ph][ALIVE[ph]].astype(float)
    P(f"[{tag}] width k_i (phase 2, alive): median {np.median(kph):.0f} q10 {np.quantile(kph, .1):.0f} q90 {np.quantile(kph, .9):.0f} max {kph.max():.0f} | open fraction median {np.median(kph) / N_in:.4f}")
    # ---- reproduction in stoch convention (t = T + 1) ----
    rows_stoch = int(sum(ISB[T - 1].sum() for T in range(9, 100)))
    births_stoch = int(sum(BIRTH[T - 1][ALIVE[T - 1]].sum() for T in range(10, 100)))
    recs = []
    for T in range(9, 99):
        for i in np.where(ALIVE[T - 1])[0]:
            n_t = int(ISB[T - 1][i].sum())
            if n_t >= 1:
                S = int((ISB[T - 1][i] & ISB[T][i]).sum()); n1 = int(ISB[T][i].sum()); recs.append((n_t, S, n1 - S, n1))
    rec = np.array(recs) if recs else np.zeros((1, 4))
    repro = dict(convention="stoch t = T+1: rows t=10..100 -> T=9..99; births t=11..100 -> T=10..99; (t,i) with n_t>=1, t=10..99 -> T=9..98",
                 rows_t10_100=rows_stoch, births_t11_100=births_stoch, target_tasks_nt_ge1=len(recs), nt_mean=float(rec[:, 0].mean()), nt_max=int(rec[:, 0].max()), S_mean=float(rec[:, 1].mean()), S_max=int(rec[:, 1].max()),
                 B_mean=float(rec[:, 2].mean()), B_max=int(rec[:, 2].max()), n1_mean=float(rec[:, 3].mean()), n1_max=int(rec[:, 3].max()))
    if args.repro:
        pub = PUBLISHED[args.repro]; repro["published"] = pub
        repro["match"] = bool(rows_stoch == pub["rows"] and births_stoch == pub["births"] and len(recs) == pub["tt"] and rec[:, 0].max() == pub["nt_max"] and rec[:, 2].max() == pub["B_max"] and abs(rec[:, 2].mean() - pub["B_mean"]) < 6e-4 and abs(rec[:, 0].mean() - pub["nt_mean"]) < 6e-3)
        P(f"[{tag}] ANCHOR (stoch convention t=T+1): rows t10..100 {rows_stoch} (pub {pub['rows']}) | births t11..100 {births_stoch} (pub {pub['births']}) | target-tasks n_t>=1 {len(recs)} (pub {pub['tt']}) | n_t mean {rec[:,0].mean():.3f} max {rec[:,0].max()} (pub {pub['nt_mean']}/{pub['nt_max']}) | B mean {rec[:,2].mean():.3f} max {rec[:,2].max()} (pub {pub['B_mean']}/{pub['B_max']}) | {'MATCH' if repro['match'] else 'MISMATCH'}")
    else:
        P(f"[{tag}] stoch-convention ledger (no published value): rows t10..100 {rows_stoch} | births t11..100 {births_stoch} | target-tasks n_t>=1 {len(recs)} | n_t mean {rec[:,0].mean():.3f} max {rec[:,0].max()} | B mean {rec[:,2].mean():.3f} max {rec[:,2].max()}")
    # ---- ledger and lifetimes (M1 convention) ----
    ledger = dict(T=list(range(1, NT + 1)), n_alive=ALIVE.sum(1).tolist(), pairs=[int(ISB[T - 1].sum()) for T in range(1, NT + 1)], births=[int(BIRTH[T - 1].sum()) if T > 1 else None for T in range(1, NT + 1)],
                  max_N=[int(Ncnt[T - 1].max()) for T in range(1, NT + 1)], k_median_alive=[float(np.median(KK[T - 1][ALIVE[T - 1]])) if ALIVE[T - 1].any() else None for T in range(1, NT + 1)],
                  degenerate_top3=degenerate.tolist(), ties_3rd=ties3.tolist())
    # pair-run lifetimes: runs of consecutive presence starting in PH2 (start T: present at T, absent at T-1), censored at T=100
    lens = []; cens = 0
    for i in range(H):
        for j in range(H):
            col = ISB[:, i, j]
            T = PH2[0]
            while T <= PH2[-1]:
                if col[T - 1] and not col[T - 2]:
                    L = 0
                    while T - 1 + L < NT and col[T - 1 + L]: L += 1
                    if T - 1 + L >= NT: cens += 1
                    else: lens.append(L)
                    T += L
                else: T += 1
    lens = np.array(lens)
    life = dict(n_runs_uncensored=int(len(lens)), n_censored=int(cens), share_len1=(float((lens == 1).mean()) if len(lens) else None), max_len=(int(lens.max()) if len(lens) else None),
                note="runs of a pair (i,j) being a lifter, started at T in phase 2 (absent at T-1), uncensored = ended before T=100; M1 convention (not P161's)")
    P(f"[{tag}] pair-run lifetimes (phase-2 starts): runs {life['n_runs_uncensored']} (censored {cens}) | share lifetime 1 task {life['share_len1']} | max {life['max_len']}")
    # ---- n* per unit (optional; leaky p172c only) ----
    nstar_vec = None
    if args.nstar_file:
        try:
            jd = json.load(open(os.path.join(NB_DIR, args.nstar_file))); nstar_vec = {int(r["i"]): float(r["n_b"]) for r in jd["real"]}
        except Exception as e: P(f"[{tag}] n_beta file not read: {e}")
    rng = np.random.default_rng(SEED)
    IND_P = {T: ISB[T - 1] for T in range(1, NT + 1)}; IND_B = {T: BIRTH[T - 1] for T in range(1, NT + 1)}
    resP, perP = analyze("presence", IND_P, ALIVE, nstar_vec, args.nstar_T, rng, KK, args.R, PH2); P(f"[{tag}] presence analysis done ({time.time() - t0:.1f}s)")
    resB, perB = analyze("births", IND_B, ALIVE, nstar_vec, args.nstar_T, rng, KK, args.R, PH2); P(f"[{tag}] births analysis done ({time.time() - t0:.1f}s)")
    for nm, r in (("presence", resP), ("births", resB)):
        po = r["pooled"]; fq = r["fano_quantiles"]; n1 = r["null_candidate_series"]["shift"]; n2 = r["null_target_shuffle"]; n3 = r["null_candidate_series"]["strat"]
        P(f"[{tag}] {nm}: targets used {r['n_targets_used']} (mu_i = 0: {r['n_targets_mu0']}) | pooled mean {po['mean']:.4f} var {po['var']:.4f} Fano {po['fano']} (within {po['fano_within']} between {po['fano_between']}) max {po['max']} | P>=2 {po['ge2']:.4f} (Pois {po['poisson_ge2']:.4f}, mix {po['mixture_ge2']:.4f}) P>=3 {po['ge3']:.5f} (Pois {po['poisson_ge3']:.5f}, mix {po['mixture_ge3']:.5f}) | n>=5 {po['n_ge5']} n>=10 {po['n_ge10']}")
        P(f"[{tag}] {nm}: per-target Fano q10 {fq['0.1']} q50 {fq['0.5']} q90 {fq['0.9']} max {fq['max']} (n finite {fq['n']}) share>1 {r['share_fano_gt1']} | mu q50 {r['mu_quantiles']['0.5']} [q10 {r['mu_quantiles']['0.1']}, q90 {r['mu_quantiles']['0.9']}] | sum delta {r['sum_delta']:.3f} share delta>0 {r['share_delta_gt0']:.3f} | identity err {r['identity_max_abs_err']:.1e}")
        for lab, nn in (("null1 shift", n1), ("null1 perm", r["null_candidate_series"]["perm"]), ("null1b strat(k_i)", n3)):
            P(f"[{tag}] {nm}: {lab}: median Fano obs {nn['observed']['median_fano']:.3f} null [{nn['null']['median_fano']['0.025']},{nn['null']['median_fano']['0.975']}] p {nn['p_two_sided']['median_fano']} | max N obs {nn['observed']['max_N']} null [{nn['null']['max_N']['0.025']},{nn['null']['max_N']['0.975']}] p {nn['p_two_sided']['max_N']} | P>=2 null [{nn['null']['pooled_ge2']['0.025']},{nn['null']['pooled_ge2']['0.975']}] p {nn['p_two_sided']['pooled_ge2']} | P>=3 null [{nn['null']['pooled_ge3']['0.025']},{nn['null']['pooled_ge3']['0.975']}] p {nn['p_two_sided']['pooled_ge3']} | sum delta null [{nn['null']['sum_delta']['0.025']},{nn['null']['sum_delta']['0.975']}] p {nn['p_two_sided']['sum_delta']} | per-target p_fano<.05 {nn['per_target']['n_p_fano_lt05']}/{nn['per_target']['n_fano_tested']} p_max<.05 {nn['per_target']['n_p_max_lt05']}")
        P(f"[{tag}] {nm}: null2 target-shuffle: median Fano obs {n2['observed']['median_fano']:.3f} null [{n2['null']['median_fano']['0.025']},{n2['null']['median_fano']['0.975']}] p {n2['p_two_sided']['median_fano']} | sd(mu_i) obs {n2['observed']['sd_mu_across_targets']:.4f} null [{n2['null']['sd_mu_across_targets']['0.025']},{n2['null']['sd_mu_across_targets']['0.975']}] p {n2['p_two_sided']['sd_mu_across_targets']}")
        for lab, v in r["by_target_width"].items():
            P(f"[{tag}] {nm}: k_i in {lab}: target-tasks {v['n_target_tasks']} mean {v['mean']:.4f} Fano {v['fano']} P>=2 {v['ge2']:.4f} (Pois {v['poisson_ge2']:.4f}) P>=3 {v['ge3']:.5f} (Pois {v['poisson_ge3']:.5f}) max {v['max']}")
        for lab, c in r["cantelli"].items():
            P(f"[{tag}] {nm}: Cantelli [{lab}] n* med {c['nstar']['0.5']:.2f}: bound max {c['cantelli']['max']:.3e} (target {c['cantelli']['argmax_target']}) median {c['cantelli']['median']:.3e} sum_i {c['cantelli']['sum_over_targets']:.3e} | Poisson-indep max {c['poisson_indep']['max']:.3e} | empirical max {c['empirical']['max']:.4f} n target-tasks >= n* {c['empirical']['n_target_tasks_ge_nstar']}")
        for lab, b in r["breakeven_nstar"].items():
            P(f"[{tag}] {nm}: break-even n* (Cantelli = {b['c']}): max {b['max']:.3f} (target {b['argmax_target']}, mu {b['argmax_mu']:.3f} sd {b['argmax_sd']:.3f}) median {b['median']:.3f} q90 {b['q90']:.3f} | pooled-moment version {b['pooled_moment_version']:.3f} | target-tasks with N >= own break-even {b['n_target_tasks_N_ge_breakeven']}")
        for w, v in r["windows"].items():
            P(f"[{tag}] {nm}: window {w}: targets {v['n_targets']} mean mu {v['mean_mu']:.4f} median Fano {v['median_fano']} pooled Fano {v['pooled_fano']} P>=2 {v['pooled_ge2']:.4f} max {v['max']}")
    # ---- M3 §4 decomposition ----
    rng_u = np.random.default_rng(20261013); decomp = {}
    for phs in args.decomp_phases.split(","):
        lo, hi = (int(x) for x in phs.split("-")); vd = decomposition(ISB, ALIVE, KK, TH, list(range(lo, hi + 1)), OO, rng_u, N_in); decomp[f"T{lo}-{hi}"] = vd
        pl = vd["pool"]
        P(f"[{tag}] (decomp {vd['phase']}, records {vd['n_records']}) N: mean {vd['N_by_unit']['mean']:.4f} Fano {vd['N_by_unit']['fano']} | by unit: share across {vd['N_by_unit']['share_across']} within Fano {vd['N_by_unit']['fano_within']} | by k-bin fine: share {vd['N_by_kbin_fine']['share_across']} within Fano {vd['N_by_kbin_fine']['fano_within']} | unit x k: within Fano {vd['N_by_unit_x_kbin_fine']['fano_within']} | by task: share {vd['N_by_task']['share_across']}")
        P(f"[{tag}] (decomp {vd['phase']}) N given pool: by C share {vd['N_by_pool']['share_across']} within Fano {vd['N_by_pool']['fano_within']} | by (C, k): share {vd['N_by_pool_x_kbin_fine']['share_across']} within Fano {vd['N_by_pool_x_kbin_fine']['fano_within']}")
        P(f"[{tag}] (decomp {vd['phase']}) pool C: mean {pl['C_mean']:.4f} var {pl['C_var']:.4f} Fano {pl['C_fano']} max {pl['C_max']} | null U (width-preserving uniform, R {R_MC_U}): mean {pl['CU_mean']:.4f} total Fano {pl['CU_total_fano']} | C/C_U {pl['ratio_C_CU']} | pass N/C {pl['pass_rate_N_over_C']} (null {pl['pass_rate_null']}) | obs/lamU {pl['ratio_obs_lamU']} | NU total Fano {pl['NU_total_fano']} | Spearman(N,C) {pl['sp_N_C']} (C,k) {pl['sp_C_k']} | C by unit share {vd['C_by_unit']['share_across']} by k within Fano {vd['C_by_kbin_fine']['fano_within']} by task share {vd['C_by_task']['share_across']}")
    # ---- save ----
    np.savez_compressed(os.path.join(RAW, f"counts_{tag}.npz"), N=Ncnt, B=Bcnt, D=Dcnt, alive=ALIVE, k=KK, isb=ISB, birth=BIRTH, Tsets=TS, aT=aT, T=np.arange(1, NT + 1),
                        note=np.array(f"M1 convention T = 1..100, state end0[T-1]; arrays [T-1, i] (isb/birth [T-1, i, j]: row target i, col candidate j; Tsets [T-1, 3, i]); dead = -1; gate {args.gate} a {a}; src {src}"))
    out = dict(tag=tag, src=src, gate=args.gate, a=a, phase=f"T{PH2[0]}-{PH2[-1]}", R_null=args.R, seed_null=SEED, min_alive_tasks=MIN_ALIVE,
               convention="T = 1..100 is the state end0[T-1]; Phase 2 = T 11..100. stoch/p161/p170/p172c use t = T+1. Births at T = lifter at T and not at T-1. Dead target (k_i = 0) excluded.",
               criterion="verbatim bridges_at (stoch_bridges.py); gate convention: " + {"leaky": "smooth leaky gate tau = 0.1 sd(z) + 1e-12, slope a", "E1": "E1 = smoothed step with a = 0 (same T_i as the leaky code for any a < 1)", "E2": "E2 = phi''(z) = e^z 1[z<0] (closed inputs nearest 0, weighted by K_n)"}[args.gate],
               diagnostics=dict(degenerate_top3_phase2=int(degenerate[ph].sum()), ties_3rd_phase2=int(ties3[ph].sum()), aT_hist_phase2=np.bincount(aT[ph][ALIVE[ph]].astype(int), minlength=4).tolist(),
                                k_phase2=q(kph, (0.1, 0.5, 0.9)), alive_unit_tasks_phase2=int(ALIVE[ph].sum()), dead_unit_tasks_phase2=int((~ALIVE[ph]).sum())),
               repro=repro, ledger=ledger, lifetimes=life, presence=resP, births=resB, decomposition=decomp, elapsed_s=time.time() - t0)
    json.dump(full(out), open(os.path.join(RAW, f"{tag}_full.json"), "w"), indent=1)
    P(f"[{tag}] saved {RAW}/counts_{tag}.npz and {RAW}/{tag}_full.json ({time.time() - t0:.1f}s)")

# ---------------------------------------------------------------------------------------------------------------- compare E1 vs E2
def compare(args):
    global LOGF
    LOGF = open(os.path.join(RAW, "compare.log"), "w"); res = {}
    for pr in args.pairs.split(","):
        t1, t2 = pr.split(":"); d1 = np.load(os.path.join(RAW, f"counts_{t1}.npz")); d2 = np.load(os.path.join(RAW, f"counts_{t2}.npz"))
        ph = np.arange(10, 100); al = d1["alive"][ph]; assert (al == d2["alive"][ph]).all()
        T1 = d1["Tsets"][ph]; T2 = d2["Tsets"][ph]; jac = []; same = 0; inter = []
        for a_ in range(len(ph)):
            for i in np.where(al[a_])[0]:
                s1 = set(T1[a_, :, i].tolist()); s2 = set(T2[a_, :, i].tolist()); jj = len(s1 & s2) / len(s1 | s2); jac.append(jj); same += int(s1 == s2); inter.append(len(s1 & s2))
        I1 = d1["isb"][ph]; I2 = d2["isb"][ph]; n1 = int(I1.sum()); n2 = int(I2.sum()); both = int((I1 & I2).sum())
        N1 = d1["N"][ph][al].astype(float); N2 = d2["N"][ph][al].astype(float)
        res[pr] = dict(target_tasks=int(al.sum()), T_jaccard_mean=float(np.mean(jac)), T_identical_share=float(same / len(jac)), T_intersection_hist=np.bincount(inter, minlength=4).tolist(),
                       pairs_1=n1, pairs_2=n2, pairs_both=both, share_of_1_in_2=(both / n1 if n1 else None), share_of_2_in_1=(both / n2 if n2 else None),
                       spearman_N=spearman(N1, N2), pearson_N=float(np.corrcoef(N1, N2)[0, 1]) if N1.std() > 0 and N2.std() > 0 else None, N_equal_share=float((N1 == N2).mean()))
        P(f"[compare {pr}] phase-2 alive target-tasks {al.sum()} | T_i Jaccard mean {np.mean(jac):.3f} identical {same / len(jac):.3f} |T1 cap T2| hist {np.bincount(inter, minlength=4).tolist()} | lifter pairs {t1} {n1} {t2} {n2} both {both} | Spearman N {res[pr]['spearman_N']} Pearson {res[pr]['pearson_N']} N equal {res[pr]['N_equal_share']:.3f}")
    json.dump(full(res), open(os.path.join(RAW, "compare.json"), "w"), indent=1)

# ---------------------------------------------------------------------------------------------------------------- post: windowed break-even
def post(args):
    """Per-target moments of presence N_i(T) inside 30-task windows (no RNG; from counts_<tag>.npz): break-even n* (Cantelli = c)
    and the Cantelli bound at fixed n*, to show how the 90-task numbers depend on the trend of mu_i(T)."""
    global LOGF
    LOGF = open(os.path.join(RAW, "post_windows.log"), "w"); res = {}
    for tag in args.tags.split(","):
        d = np.load(os.path.join(RAW, f"counts_{tag}.npz")); Nn = d["N"].astype(float); res[tag] = {}
        for (lo, hi) in [(11, 100)] + WINDOWS:
            mus, sds, mx = [], [], []
            for i in range(H):
                v = Nn[lo - 1:hi, i]; v = v[v >= 0]
                if len(v) < (MIN_ALIVE if (lo, hi) == (11, 100) else MIN_ALIVE_WIN): continue
                mus.append(v.mean()); sds.append(v.std(ddof=1)); mx.append(v.max())
            mus = np.array(mus); sds = np.array(sds); out = dict(n_targets=len(mus), mu_median=float(np.median(mus)), mu_max=float(mus.max()), fano_median=float(np.nanmedian(sds ** 2 / np.where(mus > 0, mus, np.nan))) if (mus > 0).any() else None, maxN=float(np.max(mx)))
            for c in (0.14, 0.08):
                nb = mus + sds * math.sqrt((1 - c) / c); out[f"breakeven_c{c:g}"] = dict(max=float(nb.max()), median=float(np.median(nb)))
            for ns in (10.0, 20.0, 35.3):
                b = np.array([cantelli(m_, s_ ** 2, ns) for m_, s_ in zip(mus, sds)]); out[f"cantelli_n{ns:g}"] = dict(max=float(b.max()), median=float(np.median(b)))
            res[tag][f"T{lo}-{hi}"] = out
            P(f"[post {tag} T{lo}-{hi}] targets {len(mus)} mu med {out['mu_median']:.3f} max {out['mu_max']:.3f} | F med {out['fano_median']} | max N {out['maxN']:.0f} | break-even 14%: max {out['breakeven_c0.14']['max']:.2f} med {out['breakeven_c0.14']['median']:.2f} | 8%: max {out['breakeven_c0.08']['max']:.2f} med {out['breakeven_c0.08']['median']:.2f} | Cantelli max at n*=10 {out['cantelli_n10']['max']:.3e} 20 {out['cantelli_n20']['max']:.3e} 35.3 {out['cantelli_n35.3']['max']:.3e}")
        # out-of-sample: per-target moments from T11-70, frequency of N >= break-even in T71-100 (in-sample frequencies cannot exceed the in-sample Cantelli bound [exact], so only this is a test)
        oos = {}
        for c in (0.14, 0.08):
            fr = []; ev = 0; tot = 0
            for i in range(H):
                v = Nn[10:70, i]; v = v[v >= 0]; w = Nn[70:100, i]; w = w[w >= 0]
                if len(v) < MIN_ALIVE or len(w) < MIN_ALIVE_WIN: continue
                nb = v.mean() + v.std(ddof=1) * math.sqrt((1 - c) / c); fr.append(float((w >= nb).mean())); ev += int((w >= nb).sum()); tot += len(w)
            fr = np.array(fr); oos[f"c{c:g}"] = dict(n_targets=len(fr), freq_max=float(fr.max()), freq_median=float(np.median(fr)), n_targets_freq_gt_c=int((fr > c).sum()), pooled_freq=ev / tot, events=ev, target_tasks=tot)
            P(f"[post {tag} OOS c={c}] break-even from T11-70 moments, frequency in T71-100: per-target max {fr.max():.3f} median {np.median(fr):.3f} | targets with freq > c: {int((fr > c).sum())}/{len(fr)} | pooled {ev}/{tot} = {ev / tot:.4f}")
        res[tag]["oos_T11_70_to_T71_100"] = oos
    json.dump(full(res), open(os.path.join(RAW, "post_windows.json"), "w"), indent=1)

# ---------------------------------------------------------------------------------------------------------------- combine
def combine(args):
    ALL = dict(task="cobirth_seeds_1013 TASK C (ELU): second moments of the simultaneous lifter count N_i(T), nulls, M3 §4 decomposition, Cantelli vs n* and break-even n*",
               script=os.path.abspath(__file__), raw_dir=RAW, conditions={})
    for tag in args.tags.split(","):
        d = json.load(open(os.path.join(RAW, f"{tag}_full.json")))
        for kind in ("presence", "births"):
            for key in ("per_target",): d[kind].pop(key, None)
            for mode in d[kind]["null_candidate_series"].values(): mode["per_target"].pop("p_fano", None); mode["per_target"].pop("p_max", None); mode["per_target"].pop("p_delta", None)
            d[kind]["null_target_shuffle"]["per_target"].pop("p_fano", None); d[kind]["null_target_shuffle"]["per_target"].pop("p_max", None)
            for c in d[kind]["cantelli"].values(): c.pop("per_target_cantelli", None); c.pop("per_target_nstar", None)
            for b in d[kind]["breakeven_nstar"].values(): b.pop("per_target", None)
        d["ledger"] = {k: d["ledger"][k] for k in ("n_alive", "pairs", "max_N", "k_median_alive")}
        ALL["conditions"][tag] = d
    cp = os.path.join(RAW, "compare.json")
    if os.path.exists(cp): ALL["E1_vs_E2"] = json.load(open(cp))
    pw = os.path.join(RAW, "post_windows.json")
    if os.path.exists(pw): ALL["windowed_breakeven"] = json.load(open(pw))
    json.dump(rnd(ALL), open(OUT_JSON, "w"), indent=0, separators=(",", ":"))
    print(f"wrote {OUT_JSON}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run"); r.add_argument("--npz", required=True); r.add_argument("--gate", choices=["leaky", "E1", "E2"], required=True); r.add_argument("--a", type=float, default=0.1)
    r.add_argument("--tag", required=True); r.add_argument("--R", type=int, default=1000); r.add_argument("--repro", default=None); r.add_argument("--nstar-file", dest="nstar_file", default=None)
    r.add_argument("--nstar-T", dest="nstar_T", type=int, default=None); r.add_argument("--decomp-phases", dest="decomp_phases", default="11-100"); r.add_argument("--phase", default="11-100")
    c = sub.add_parser("compare"); c.add_argument("--pairs", required=True)
    m = sub.add_parser("combine"); m.add_argument("--tags", required=True)
    pp = sub.add_parser("post"); pp.add_argument("--tags", required=True)
    args = ap.parse_args()
    {"run": run, "compare": compare, "combine": combine, "post": post}[args.cmd](args)
