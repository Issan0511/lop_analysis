"""nbeta_time (cobirth_seeds_1013, task D, 1013-14): n_beta over time and the state-matched Cantelli.

Convention (M1): task T = 1..100, state = end of task T = end0[T-1]; Phase 2 = T 11..100 (90 task ends).
The p172c scripts use t = T + 1 (their 'task' arg t maps to end0[t-2]); nbeta_time_p172c.py --T T runs t = T + 1 and stores T_M1 = T.

Inputs (read only):
  n_beta per state  : obsidian-research-data/cobirth_seeds_1013/nbeta_time/grid/p172c_<tag>_T<T>_real.json   (T = 10, 20, ..., 100; full records)
                      obsidian-research-data/cobirth_seeds_1013/nbeta_time/full/p172c_<tag>_T<T>_real_slim.json (all other T 11..99; slim, optional)
  single-state n_beta (the original M1 source): push_direction_proof_0930/.../agent_bound/p172c_{a01_t40,a03_t100,a05_t100}_real.json  (T 39 / 99 / 99)
  counts            : obsidian-research-data/cobirth_1013/m1/counts_<tag>.npz  (N[T-1, i] presence, -1 = dead target)
  M1 reference      : results/cobirth_1013/m1_fano.json (Fano fields, Cantelli with the single-state n_beta)

Cantelli per (target i, task T) [exact given mu_i, sigma_i^2 as population values; (N*) assumed]:
  c_i(T) = sigma_i^2 / (sigma_i^2 + (n_beta,i(T*) - mu_i)^2) if n_beta > mu_i else 1,
  mu_i, sigma_i^2 (ddof=1) from N_i over the alive Phase-2 tasks (as M1), T* = the computed state matched to T.
  Matching variants: 'grid_near_lo' nearest grid state, ties (T = 15, 25, ...) to the earlier one [primary];
                     'grid_near_hi' ties to the later one; 'exact' T* = T (needs the full set); 'single' the M1 single state (anchor).
  If target i is alive at T but dead at T*, n_beta is taken from the nearest computed state where i is alive (ties to the earlier); counted.
Break-even n*: smallest n* with Cantelli <= rho: n*_be = mu + sigma sqrt((1 - rho)/rho)   [algebra]; rho = 0.14 (schema) and the arm thresholds
  (c = 2.8: 14/11/8 %, c = 1: 32/26/19 %, cumulative_push_1013 section 3); uniform Fano Fbar in {3, 6.4}: sigma^2 = Fbar mu.
usage: nbeta_time_cantelli.py   (CPU, numpy only, seconds)"""
import os, json, glob, math
import numpy as np

RAW = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/nbeta_time"
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
NB_DIR = os.path.join(ROOT, "review/stage4_1002/block_1003/stage14_1007_h4/agent_bound")
CNT = "/home/issan/Projects/obsidian-research-data/cobirth_1013/m1"
WT = "/home/issan/Projects/claude/wt/cobirth_seeds_1013"
M1J = os.path.join(WT, "results/cobirth_1013/m1_fano.json")
OUT = os.path.join(WT, "results/cobirth_seeds_1013/nbeta_time.json")
ARMS = {"a01": dict(a=0.1, npz="LR_ref_s0", single=("p172c_a01_t40_real.json", 39), thr_c28=0.14, thr_c1=0.32, fano_pool_width_M3=0.64),
        "a03": dict(a=0.3, npz="LRa0.3_s0", single=("p172c_a03_t100_real.json", 99), thr_c28=0.11, thr_c1=0.26, fano_pool_width_M3=0.91),
        "a05": dict(a=0.5, npz="LRa0.5_s0", single=("p172c_a05_t100_real.json", 99), thr_c28=0.08, thr_c1=0.19, fano_pool_width_M3=0.94)}
GRID = list(range(10, 101, 10)); PH2 = list(range(11, 101)); H = 100; MIN_ALIVE = 30
nan = float("nan")


def qd(v, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0: return None
    return {("min" if x == 0 else "max" if x == 1 else f"q{int(round(100 * x))}"): float(np.quantile(v, x)) for x in qs} | {"n": int(len(v)), "mean": float(v.mean())}


def rnd(o):
    if isinstance(o, dict): return {str(k): rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v) for v in o]
    if isinstance(o, np.ndarray): return rnd(o.tolist())
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        if math.isinf(o): return "inf" if o > 0 else "-inf"
        if not np.isfinite(o): return None
        return float(f"{o:.6g}")
    return o


def load_state(path):
    d = json.load(open(path)); nb = np.full(H, nan); need = np.full(H, nan); mL = np.full(H, nan)
    for r in d["real"]: nb[r["i"]] = r["n_b"]; need[r["i"]] = r["need_b"]; mL[r["i"]] = r["max_L_b"]
    return d, nb, need, mL


def cantelli(mu, var, ns):
    if not np.isfinite(ns) or not np.isfinite(var): return nan
    if ns <= mu: return 1.0
    return float(var / (var + (ns - mu) ** 2))


def nearest(T, avail, tie="lo"):
    avail = np.asarray(sorted(avail)); d = np.abs(avail - T); m = d.min(); c = avail[d == m]
    return int(c.min() if tie == "lo" else c.max())


def main():
    m1 = json.load(open(M1J)); ALL = dict(convention="T = 1..100, state end0[T-1]; Phase 2 = T 11..100. p172c 'task' t = T + 1 (state end0[t-2]); grid T = 10..100 step 10 = p172c t = 11..101.",
                                         grid_T=GRID, arms={})
    for tag, A in ARMS.items():
        z = np.load(os.path.join(CNT, f"counts_{tag}.npz")); Ncnt = z["N"].astype(int)   # [T-1, i], -1 dead
        NB = {}; NEED = {}; ML = {}; ALIVE_NB = {}
        for T in GRID:
            d, nb, need, mL = load_state(os.path.join(RAW, f"grid/p172c_{tag}_T{T}_real.json")); assert d["T_M1"] == T and d["task"] == T + 1
            NB[T] = nb; NEED[T] = need; ML[T] = mL; ALIVE_NB[T] = np.isfinite(nb) | np.isinf(nb)
        FULL = {}
        for f in sorted(glob.glob(os.path.join(RAW, f"full/p172c_{tag}_T*_real_slim.json"))) if not os.environ.get("NO_FULL") else []:
            try: d, nb, need, mL = load_state(f)
            except json.JSONDecodeError: print("skip (being written):", f); continue
            FULL[d["T_M1"]] = nb
        for T in GRID: FULL[T] = NB[T]
        have_full = all(T in FULL for T in range(10, 101))
        d1, nb1, _, _ = load_state(os.path.join(NB_DIR, A["single"][0])); T1 = A["single"][1]; assert d1["task"] == T1 + 1
        R = dict(a=A["a"], npz=A["npz"], single_state_T=T1, n_full_states=len(FULL), have_full=have_full)
        # ---------- consistency: alive sets (p172c vs counts) ----------
        R["alive_set_match_grid"] = {T: bool(set(np.where(ALIVE_NB[T])[0]) == set(np.where(Ncnt[T - 1] >= 0)[0])) for T in GRID}
        R["alive_set_match_full"] = bool(all(set(np.where(np.isfinite(FULL[T]) | np.isinf(FULL[T]))[0]) == set(np.where(Ncnt[T - 1] >= 0)[0]) for T in FULL))
        # ---------- (1) distribution of n_beta over (target, state), grid ----------
        allnb = np.concatenate([NB[T][ALIVE_NB[T]] for T in GRID])
        per_state = {T: dict(n_alive=int(ALIVE_NB[T].sum()), min=float(np.nanmin(NB[T])), argmin_unit=int(np.nanargmin(NB[T])), q10=float(np.nanquantile(NB[T], 0.1)), median=float(np.nanmedian(NB[T])),
                             n_lt4=int((NB[T] < 4).sum()), n_lt10=int((NB[T] < 10).sum()), n_inf=int(np.isinf(NB[T]).sum()), n_zero=int((NB[T] == 0).sum()), n_need_le0=int((NEED[T] <= 0).sum())) for T in GRID}
        mins = np.array([per_state[T]["min"] for T in GRID]); meds = np.array([per_state[T]["median"] for T in GRID])
        def spearman(x, y):
            rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float); return float(np.corrcoef(rx, ry)[0, 1])
        R["grid"] = dict(n_target_states=int(len(allnb)), quantiles=qd(allnb), n_lt4=int((allnb < 4).sum()), n_lt10=int((allnb < 10).sum()), frac_lt4=float((allnb < 4).mean()), frac_lt10=float((allnb < 10).mean()),
                         n_inf=int(np.isinf(allnb).sum()), n_zero=int((allnb == 0).sum()), per_state=per_state,
                         min_over_states=dict(values=mins, range=[float(mins.min()), float(mins.max())], spearman_vs_T=spearman(np.array(GRID, float), mins), first_half_mean=float(mins[:5].mean()), second_half_mean=float(mins[5:].mean()),
                                              distinct_argmin_units=len(set(per_state[T]["argmin_unit"] for T in GRID))),
                         median_over_states=dict(values=meds, spearman_vs_T=spearman(np.array(GRID, float), meds)),
                         single_state_min=float(np.nanmin(nb1)), single_state_median=float(np.nanmedian(nb1)),
                         n_states_with_min_below_single_min=int((mins < np.nanmin(nb1)).sum()))
        # units below thresholds: which units, how many states
        low4 = sorted({(int(i), T) for T in GRID for i in np.where(NB[T] < 4)[0]}); low10 = sorted({(int(i), T) for T in GRID for i in np.where(NB[T] < 10)[0]})
        R["grid"]["lt4_unit_state"] = low4; R["grid"]["lt10_units"] = sorted({u for u, _ in low10}); R["grid"]["lt10_n_unit_state"] = len(low10)
        # per-unit variability over the grid states where alive
        cv = []; rng_ratio = []
        for i in range(H):
            v = np.array([NB[T][i] for T in GRID if ALIVE_NB[T][i]])
            if len(v) >= 3: cv.append(v.std() / v.mean()); rng_ratio.append(v.max() / v.min())
        rho_consec = [spearman(NB[T][ALIVE_NB[T] & ALIVE_NB[T + 10]], NB[T + 10][ALIVE_NB[T] & ALIVE_NB[T + 10]]) for T in GRID[:-1]]
        R["grid"]["per_unit_cv"] = qd(cv); R["grid"]["per_unit_max_over_min"] = qd(rng_ratio); R["grid"]["spearman_consecutive_grid_states"] = rho_consec
        # alive/missing across states
        alive_mat = np.stack([ALIVE_NB[T] for T in GRID], 0)   # (10, H)
        n_states_alive = alive_mat.sum(0)
        R["grid"]["units_alive_all_grid_states"] = int((n_states_alive == len(GRID)).sum()); R["grid"]["units_alive_some_not_all"] = int(((n_states_alive > 0) & (n_states_alive < len(GRID))).sum())
        R["grid"]["units_never_alive_on_grid"] = int((n_states_alive == 0).sum()); R["grid"]["n_states_alive_hist"] = np.bincount(n_states_alive, minlength=len(GRID) + 1)
        minG = np.array([np.nanmin([NB[T][i] for T in GRID if ALIVE_NB[T][i]]) if n_states_alive[i] > 0 else nan for i in range(H)]); okS = np.isfinite(nb1) & np.isfinite(minG)
        R["grid"]["per_target_min_over_grid"] = qd(minG); R["grid"]["single_state_nb"] = qd(nb1)
        R["grid"]["single_over_per_target_min_grid"] = qd(nb1[okS] / minG[okS]); R["grid"]["frac_single_above_min_grid"] = float((nb1[okS] > minG[okS]).mean())
        single_missing = [i for i in range(H) if not np.isfinite(nb1[i])]
        R["grid"]["single_state_missing_units"] = single_missing
        R["grid"]["single_state_missing_units_with_grid_value"] = int(sum(1 for i in single_missing if n_states_alive[i] > 0))
        if single_missing:
            filled_med = float(np.nanmedian(nb1)); own = [float(np.nanmin([NB[T][i] for T in GRID if ALIVE_NB[T][i]])) for i in single_missing if n_states_alive[i] > 0]
            R["grid"]["single_state_missing_units_min_grid_nb"] = qd(own); R["grid"]["single_state_median_fill"] = filled_med
            R["grid"]["single_state_missing_units_min_grid_nb_below_fill"] = int(sum(1 for x in own if x < filled_med))
        # full set distribution (if any)
        if len(FULL) > len(GRID):
            Ts = sorted(FULL); allf = np.concatenate([FULL[T][np.isfinite(FULL[T]) | np.isinf(FULL[T])] for T in Ts]); minsf = np.array([np.nanmin(FULL[T]) for T in Ts])
            R["full"] = dict(T_list=Ts, n_states=len(Ts), n_target_states=int(len(allf)), quantiles=qd(allf), n_lt4=int((allf < 4).sum()), n_lt10=int((allf < 10).sum()), frac_lt4=float((allf < 4).mean()), frac_lt10=float((allf < 10).mean()),
                             n_inf=int(np.isinf(allf).sum()), n_zero=int((allf == 0).sum()), min_per_state=dict(T=Ts, values=minsf, spearman_vs_T=spearman(np.array(Ts, float), minsf),
                             argmin_T=int(Ts[int(np.argmin(minsf))]), argmin_unit=int(np.nanargmin(FULL[Ts[int(np.argmin(minsf))]]))),
                             lt4_unit_state=sorted({(int(i), T) for T in Ts for i in np.where(FULL[T] < 4)[0]}),
                             lt10_n_unit_state=int(sum((FULL[T] < 10).sum() for T in Ts)), lt10_units=sorted({int(i) for T in Ts for i in np.where(FULL[T] < 10)[0]}),
                             per_target_min=qd(np.array([np.nanmin([FULL[T][i] for T in Ts]) if any(np.isfinite(FULL[T][i]) for T in Ts) else nan for i in range(H)])),
                             spearman_lag1=qd([spearman(FULL[T][np.isfinite(FULL[T]) & np.isfinite(FULL[T + 1])], FULL[T + 1][np.isfinite(FULL[T]) & np.isfinite(FULL[T + 1])]) for T in Ts if T + 1 in FULL]),
                             min_per_state_first_half_mean=float(minsf[:len(minsf) // 2].mean()), min_per_state_second_half_mean=float(minsf[len(minsf) // 2:].mean()))
        # ---------- (2) per-target mu, sigma^2 (M1) ----------
        used = []; MU = np.full(H, nan); VAR = np.full(H, nan); NALIVE = np.zeros(H, int)
        for i in range(H):
            v = np.array([Ncnt[T - 1, i] for T in PH2 if Ncnt[T - 1, i] >= 0], float); NALIVE[i] = len(v)
            if len(v) >= MIN_ALIVE: used.append(i); MU[i] = v.mean(); VAR[i] = v.var(ddof=1)
        used = np.array(used); F = VAR / MU
        m1p = m1["arms"][tag]["presence"]; m1mu = np.array(m1p["per_target"]["mu"]); m1i = np.array(m1p["per_target"]["i"])
        R["mu_sigma_repro"] = dict(n_used=len(used), max_abs_diff_mu_vs_m1=float(np.max(np.abs(MU[m1i] - m1mu))), fano_median=float(np.nanmedian(F[used])), m1_fano_median=float(m1p["fano_quantiles"]["0.5"]))
        # ---------- (3) Cantelli per (target, task) ----------
        def nb_lookup(i, Tstar, avail, tie, src):
            if np.isfinite(src[Tstar][i]) or np.isinf(src[Tstar][i]): return src[Tstar][i], Tstar, False
            al = [T for T in avail if np.isfinite(src[T][i]) or np.isinf(src[T][i])]
            if not al: return nan, None, True
            T2 = nearest(Tstar, al, tie); return src[T2][i], T2, True
        variants = {"grid_near_lo": (GRID, NB, "lo"), "grid_near_hi": (GRID, NB, "hi")}
        if have_full: variants["exact"] = (sorted(FULL), FULL, "lo")
        CAN = {}
        for vname, (avail, src, tie) in variants.items():
            rows = []   # (i, T, N, nb, Tstar_used, filled, c)
            for i in used:
                for T in PH2:
                    if Ncnt[T - 1, i] < 0: continue
                    Tstar = T if vname == "exact" else nearest(T, avail, tie)
                    nb, Tu, filled = nb_lookup(i, Tstar, avail, tie, src)
                    rows.append((i, T, Ncnt[T - 1, i], nb, Tu, filled, cantelli(MU[i], VAR[i], nb)))
            arr = np.array([(r[0], r[1], r[2], r[3], r[6], r[5]) for r in rows], float)
            c = arr[:, 4]; Nn = arr[:, 2]; nbv = arr[:, 3]; ii = arr[:, 0].astype(int); TT = arr[:, 1].astype(int)
            tm = np.array([c[ii == i].mean() for i in used])                     # time-mean per target = bound on rho_R(i)
            per_task_sum = np.array([c[TT == T].sum() for T in PH2])
            ge = Nn >= nbv
            CAN[vname] = dict(n_target_tasks=int(len(rows)), n_filled_from_other_state=int(arr[:, 5].sum()),
                              filled_units=sorted({int(r[0]) for r in rows if r[5]}),
                              cantelli_max=float(np.nanmax(c)), cantelli_median=float(np.nanmedian(c)), cantelli_q90=float(np.nanquantile(c, 0.9)),
                              argmax=dict(i=int(ii[np.nanargmax(c)]), T=int(TT[np.nanargmax(c)]), nb=float(nbv[np.nanargmax(c)]), mu=float(MU[ii[np.nanargmax(c)]]), var=float(VAR[ii[np.nanargmax(c)]])),
                              time_mean_per_target=dict(max=float(tm.max()), median=float(np.median(tm)), sum=float(tm.sum()), argmax_i=int(used[np.argmax(tm)])),
                              per_task_sum=dict(max=float(per_task_sum.max()), argmax_T=int(PH2[int(np.argmax(per_task_sum))]), mean=float(per_task_sum.mean())),
                              empirical_N_ge_nbeta=int(ge.sum()), max_N_over_nbeta=float(np.nanmax(Nn / nbv)), argmax_N_over_nbeta=dict(i=int(ii[np.nanargmax(Nn / nbv)]), T=int(TT[np.nanargmax(Nn / nbv)]), N=int(Nn[np.nanargmax(Nn / nbv)]), nb=float(nbv[np.nanargmax(Nn / nbv)])),
                              min_nbeta_minus_N=float(np.nanmin(nbv - Nn)), n_nbeta_le_mu=int((nbv <= MU[ii]).sum()),
                              threshold_c28=A["thr_c28"], threshold_c1=A["thr_c1"], n_targets_timemean_above_c28=int((tm > A["thr_c28"]).sum()), n_target_tasks_above_c28=int((c > A["thr_c28"]).sum()))
            # association of N_i(T) with the matched n_beta (the Cantelli above uses a time-constant mu_i, sigma_i^2)
            okp = np.isfinite(nbv); relN = Nn[okp] / MU[ii[okp]]
            lowq = np.array([nbv[k] <= np.nanquantile(nbv[ii == ii[k]], 0.1) for k in range(len(nbv))])
            CAN[vname]["assoc_N_nbeta"] = dict(spearman_N_vs_nbeta=spearman(Nn[okp], nbv[okp]),
                                               spearman_N_over_mu_vs_nbeta_over_target_median=spearman(relN, nbv[okp] / np.array([np.nanmedian(nbv[ii == i_]) for i_ in ii[okp]])),
                                               mean_N_over_mu_all=float(relN.mean()), mean_N_over_mu_nbeta_lt10=float((Nn / MU[ii])[nbv < 10].mean()) if (nbv < 10).any() else None, n_nbeta_lt10=int((nbv < 10).sum()),
                                               max_N_nbeta_lt10=int(Nn[nbv < 10].max()) if (nbv < 10).any() else None,
                                               mean_N_over_mu_target_lowest_decile_nbeta=float((Nn / MU[ii])[lowq].mean()), n_lowest_decile=int(lowq.sum()))
            # sensitivity: Cantelli with the pooled conditional moments of N on the low-n_beta subset (mean, var ddof=1 of N over the subset's (i,T) pairs)
            for lab_, sel in (("nbeta_lt10", nbv < 10), ("target_lowest_decile", lowq)):
                if sel.sum() >= 2:
                    mc, vc = float(Nn[sel].mean()), float(Nn[sel].var(ddof=1)); cc = np.array([cantelli(mc, vc, nb_) for nb_ in nbv[sel]])
                    CAN[vname]["assoc_N_nbeta"][f"cond_cantelli_{lab_}"] = dict(n=int(sel.sum()), mean_N=mc, var_N=vc, fano=vc / mc if mc > 0 else None, max=float(cc.max()), mean=float(cc.mean()),
                                                                               min_nbeta=float(nbv[sel].min()), max_N=int(Nn[sel].max()), n_N_ge_nbeta=int((Nn[sel] >= nbv[sel]).sum()),
                                                                               contribution_to_timemean_max=float(max(cc[ii[sel] == i_].sum() / (ii == i_).sum() for i_ in np.unique(ii[sel]))))
            # uniform Fbar variants on the same matched n_beta
            for Fb in (3.0, 6.4):
                cF = np.array([cantelli(MU[i_], Fb * MU[i_], nb_) for i_, nb_ in zip(ii, nbv)]); tmF = np.array([cF[ii == i].mean() for i in used])
                CAN[vname][f"uniform_F{Fb:g}"] = dict(max=float(np.nanmax(cF)), median=float(np.nanmedian(cF)), time_mean_max=float(tmF.max()), time_mean_sum=float(tmF.sum()), per_task_sum_max=float(max(cF[TT == T].sum() for T in PH2)))
        # single-state anchor (reproduces M1's per_unit_n_beta Cantelli incl. the median fill)
        ns1 = np.array([nb1[i] for i in used]); have1 = np.isfinite(ns1); ns1f = np.where(have1, ns1, np.nanmedian(ns1))
        b1 = np.array([cantelli(MU[i], VAR[i], n_) for i, n_ in zip(used, ns1f)]); m1c = m1p["cantelli"]["per_unit_n_beta"]["cantelli"]
        emp1 = int(sum(((np.array([Ncnt[T - 1, i] for T in PH2 if Ncnt[T - 1, i] >= 0]) >= n_).sum()) for i, n_ in zip(used, ns1f)))
        CAN["single_state_M1"] = dict(n_filled_median=int((~have1).sum()), cantelli_max=float(b1.max()), cantelli_median=float(np.median(b1)), sum=float(b1.sum()), empirical_N_ge_nbeta=emp1,
                                      m1_published=dict(max=m1c["max"], median=m1c["median"], sum=m1c["mean"] * len(used)),
                                      match=bool(abs(b1.max() - m1c["max"]) / m1c["max"] < 1e-5 and abs(np.median(b1) - m1c["median"]) / m1c["median"] < 1e-5))
        R["cantelli"] = CAN
        # ---------- (4) break-even n* vs observed n_beta ----------
        BE = {}
        sig = np.sqrt(VAR)
        for lab, rho in (("rho14", 0.14), ("c28_arm", A["thr_c28"]), ("c1_arm", A["thr_c1"])):
            k = math.sqrt((1 - rho) / rho); be = MU + sig * k
            beF = {Fb: MU + np.sqrt(Fb * MU) * k for Fb in (3.0, 6.4)}
            # compare with n_beta at every grid state where the target is alive, and with the target's min over the grid / full set
            pairs = [(i, T, NB[T][i]) for T in GRID for i in used if ALIVE_NB[T][i]]
            ratio = np.array([nb_ / be[i] for i, T, nb_ in pairs]); gap = np.array([nb_ - be[i] for i, T, nb_ in pairs])
            minnb_grid = np.array([np.nanmin([NB[T][i] for T in GRID if ALIVE_NB[T][i]]) if any(ALIVE_NB[T][i] for T in GRID) else nan for i in used])
            ent = dict(rho=rho, nstar_be_measuredF=qd(be[used]), nstar_be_max=float(np.nanmax(be[used])), argmax_i=int(used[np.nanargmax(be[used])]),
                       nstar_be_uniformF3=qd(beF[3.0][used]), nstar_be_uniformF6_4=qd(beF[6.4][used]),
                       grid_nbeta_over_be=qd(ratio), grid_min_gap=float(gap.min()), grid_n_nbeta_below_be=int((gap < 0).sum()), grid_n_pairs=len(pairs),
                       grid_n_nbeta_below_be_F6_4=int(sum(1 for i, T, nb_ in pairs if nb_ < beF[6.4][i])), grid_n_nbeta_below_be_F3=int(sum(1 for i, T, nb_ in pairs if nb_ < beF[3.0][i])),
                       per_target_min_grid_nbeta_over_be=qd(minnb_grid / be[used]), worst_target=dict(i=int(used[np.nanargmin(minnb_grid / be[used])]), min_grid_nbeta=float(np.nanmin(minnb_grid / be[used]) * be[used][np.nanargmin(minnb_grid / be[used])]), be=float(be[used][np.nanargmin(minnb_grid / be[used])])))
            if have_full:
                pf = [(i, T, FULL[T][i]) for T in sorted(FULL) if T >= 11 for i in used if np.isfinite(FULL[T][i])]
                gf = np.array([nb_ - be[i] for i, T, nb_ in pf]); ent["full_min_gap"] = float(gf.min()); ent["full_n_nbeta_below_be"] = int((gf < 0).sum()); ent["full_n_pairs"] = len(pf)
                ent["full_n_nbeta_below_be_F6_4"] = int(sum(1 for i, T, nb_ in pf if nb_ < beF[6.4][i]))
            BE[lab] = ent
        R["breakeven"] = BE
        # ---------- (5) M1 reference fields ----------
        po = m1p["pooled"]; fq = m1p["fano_quantiles"]; nc = m1p["null_candidate_series"]
        R["m1_reference"] = dict(mu_pooled=po["mean"], pool_fano=po["fano"], fano_median=fq["0.5"], fano_q90=fq["0.9"], fano_max=fq["max"], max_N=po["max"],
                                 null1_fano_interval=[nc["shift"]["null"]["median_fano"]["0.025"], nc["shift"]["null"]["median_fano"]["0.975"]],
                                 null1b_fano_interval=[nc["strat"]["null"]["median_fano"]["0.025"], nc["strat"]["null"]["median_fano"]["0.975"]],
                                 fano_given_pool_width_M3=A["fano_pool_width_M3"])
        ALL["arms"][tag] = R
        g = R["grid"]; cg = CAN["grid_near_lo"]
        print(f"[{tag}] grid n_beta over (target,state) n={g['n_target_states']}: min {g['quantiles']['min']:.2f} q10 {g['quantiles']['q10']:.2f} median {g['quantiles']['q50']:.2f} | <4 {g['n_lt4']} <10 {g['n_lt10']} | per-state min {np.round(mins, 2).tolist()} (single-state min {g['single_state_min']:.2f}) spearman(min,T) {g['min_over_states']['spearman_vs_T']:+.2f}")
        if "full" in R:
            fu = R["full"]; print(f"[{tag}] FULL n_beta over (target,state) states {fu['n_states']} n={fu['n_target_states']}: min {fu['quantiles']['min']:.2f} q10 {fu['quantiles']['q10']:.2f} median {fu['quantiles']['q50']:.2f} | <4 {fu['n_lt4']} {fu['lt4_unit_state']} <10 {fu['n_lt10']} | min-per-state spearman vs T {fu['min_per_state']['spearman_vs_T']:+.2f} argmin T{fu['min_per_state']['argmin_T']} u{fu['min_per_state']['argmin_unit']} | lag-1 rank corr median {fu['spearman_lag1']['q50']:.2f} | per-target min q10/50 {fu['per_target_min']['q10']:.2f}/{fu['per_target_min']['q50']:.2f}")
        print(f"[{tag}] per-target min over grid q10/50 {g['per_target_min_over_grid']['q10']:.2f}/{g['per_target_min_over_grid']['q50']:.2f} vs single-state q10/50 {g['single_state_nb']['q10']:.2f}/{g['single_state_nb']['q50']:.2f}; single/min-grid median {g['single_over_per_target_min_grid']['q50']:.2f}")
        print(f"[{tag}] units alive all grid states {g['units_alive_all_grid_states']} some-not-all {g['units_alive_some_not_all']} never {g['units_never_alive_on_grid']}; single-state missing {len(g['single_state_missing_units'])} (with a grid value {g['single_state_missing_units_with_grid_value']})")
        for vn, cv_ in CAN.items():
            if vn == "single_state_M1":
                print(f"[{tag}] Cantelli single-state (M1 anchor): max {cv_['cantelli_max']:.4g} median {cv_['cantelli_median']:.4g} sum {cv_['sum']:.4g} | M1 {cv_['m1_published']} MATCH {cv_['match']} | empirical {cv_['empirical_N_ge_nbeta']}")
            else:
                print(f"[{tag}] Cantelli {vn}: (i,T) max {cv_['cantelli_max']:.4g} median {cv_['cantelli_median']:.4g} | time-mean max {cv_['time_mean_per_target']['max']:.4g} sum {cv_['time_mean_per_target']['sum']:.4g} | per-task sum max {cv_['per_task_sum']['max']:.4g} | empirical N>=n_beta {cv_['empirical_N_ge_nbeta']} max N/n_beta {cv_['max_N_over_nbeta']:.3f} | filled {cv_['n_filled_from_other_state']} | F3 max {cv_['uniform_F3']['max']:.4g} F6.4 max {cv_['uniform_F6.4']['max']:.4g}")
        for lab, e in BE.items():
            print(f"[{tag}] break-even {lab} rho {e['rho']}: n*_be median {e['nstar_be_measuredF']['q50']:.2f} max {e['nstar_be_max']:.2f} (u{e['argmax_i']}) | grid n_beta/be min {e['grid_nbeta_over_be']['min']:.2f} below {e['grid_n_nbeta_below_be']}/{e['grid_n_pairs']} (F6.4 {e['grid_n_nbeta_below_be_F6_4']}) | worst target u{e['worst_target']['i']} min n_beta {e['worst_target']['min_grid_nbeta']:.2f} vs be {e['worst_target']['be']:.2f}" + (f" | full below {e['full_n_nbeta_below_be']}/{e['full_n_pairs']} min gap {e['full_min_gap']:.2f}" if have_full else ""))
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(rnd(ALL), open(OUT, "w"), indent=1); print("wrote", OUT)


if __name__ == "__main__":
    main()
