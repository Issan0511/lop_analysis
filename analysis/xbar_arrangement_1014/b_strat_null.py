"""xbar_arrangement_1014 agent B: brightness-stratified permutation null for the loose-bridge count N_i(T) and the threshold pool C_i(T).

Definitions copied from proj_004_drift/analysis/cobirth_seeds_1013/verify_counts.py (function `counts`, verbatim except that it also
returns the per-state gates O, top-3 indices T3, k and alive of phase 2):
  state T (M1, T = 1..100) = end0[T-1], end1[T-1];  Xt = [X, 1];  K_n = Xt_n . mean(Xt) (> 0);  z_in = w_i.x_n + b_i;
  O_i = {n : z_in > 0};  k_i = |O_i|;  alive = k_i > 0.
  T_i = top-3 of score:  leaky/E1: log K_n + log s(1-s) - log tau_i (tau_i = 0.1 sd_n(z_in) + 1e-12);  E2: log K_n + z_in on z_in < 0, -inf else.
  bridge j -> i: j != i both alive, |O_j cap T_i| >= 2 and |O_j cap O_i| >= ceil(k_i/2);  N_i = #bridges;  C_i = #{j != i alive : |O_j cap T_i| >= 2}.
  Phase 2 = T 11..100; targets = units with >= 30 alive phase-2 tasks.
Surrogates (new): at each phase-2 state, every unit's gate column o_j is permuted
  strat: independently within each of the 10 K-deciles (rank split of K over the 1200 inputs, 120 inputs per decile)
  flat : over all 1200 inputs.
  The target's own T_i, O_i, k_i stay actual; candidates j use permuted gates (one permutation per (unit, state, surrogate), shared across targets).
  R = 20 surrogates, rng = np.random.default_rng(1000 + r), one generator per (null, r), drawn state by state T = 11..100.
usage: b_strat_null.py <tag> <npz rel. to ROOT> <gate leaky|E1|E2> [R]"""
import os, sys, json, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT); from common import load_X
VER = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/verify"
WT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014"
RES = os.path.join(WT, "results/xbar_arrangement_1014")
BIG = "/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b"
PH = np.arange(10, 100)  # rows T-1 for T = 11..100


def counts(npz, gate):
    """verbatim copy of verify_counts.counts (cobirth_seeds_1013) + phase-2 storage of O, T3, k, alive, and K."""
    d = np.load(os.path.join(ROOT, npz)); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]; K = Xt @ Xt.mean(0)
    assert (K > 0).all(); logK = np.log(K)
    H = d["end0"].shape[1]; NT = d["end0"].shape[0]
    Nc = np.full((NT, H), -1, int); Cc = np.full((NT, H), -1, int); kk = np.zeros((NT, H), int)
    store = {}
    for T in range(1, NT + 1):
        W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64); Z = Xt[:, :-1] @ W.T + b[None, :]
        O = Z > 0; k = O.sum(0); alive = k > 0
        if gate in ("leaky", "E1"):
            tau = 0.1 * Z.std(0) + 1e-12; x = Z / tau[None, :]; ax = np.abs(x)
            score = logK[:, None] - ax - 2.0 * np.log1p(np.exp(-ax)) - np.log(tau)[None, :]
        else:
            score = np.where(Z < 0, logK[:, None] + Z, -np.inf)
        T3 = np.argsort(-score, axis=0, kind="stable")[:3]            # (3, H)
        Of = O.astype(np.int64)
        thr = Of[T3].sum(0)                                              # thr[i, j] = |T_i cap O_j|
        ov = Of.T @ Of                                                   # ov[i, j] = |O_i cap O_j|
        need = -(-k // 2)                                                # ceil(k/2)
        both = alive[:, None] & alive[None, :]; np.fill_diagonal(both, False)
        br = (thr >= 2) & (ov >= need[:, None]) & both
        kk[T - 1] = k
        Nc[T - 1] = np.where(alive, br.sum(1), -1); Cc[T - 1] = np.where(alive, ((thr >= 2) & both).sum(1), -1)
        if T >= 11: store[T] = (O.copy(), T3.copy(), k.copy(), alive.copy())
    return Nc, Cc, kk, store, K


def surr_counts(O, T3, k, alive, Operm):
    """N_i, C_i with the target's actual T_i, O_i, k_i and candidates' permuted gates Operm."""
    Pf = Operm.astype(np.float64)
    thr = Pf[T3].sum(0)                                                  # thr[i, j] = |T_i cap Operm_j|
    ov = O.astype(np.float64).T @ Pf                                     # ov[i, j] = |O_i cap Operm_j|  (exact in float64)
    need = -(-k // 2)
    both = alive[:, None] & alive[None, :]; np.fill_diagonal(both, False)
    c = (thr >= 2) & both
    br = c & (ov >= need[:, None])
    return np.where(alive, br.sum(1), -1), np.where(alive, c.sum(1), -1)


def permute(O, rng, blocks):
    """independent permutation of every column of O within each index block (blocks = list of index arrays)."""
    P = np.empty_like(O)
    H = O.shape[1]
    for idx in blocks:
        p = np.argsort(rng.random((len(idx), H)), axis=0)
        P[idx] = np.take_along_axis(O[idx], p, axis=0)
    return P


def series_stats(M, ids):
    """M: (90, H) phase-2 counts (-1 = dead). Per-target mu, var(ddof=1), F; pooled Fano (ddof=1) over the targets' alive tasks;
    pooled population Fano; max; # >= 10."""
    mus, vs, Fs, pool = [], [], [], []
    for i in ids:
        v = M[M[:, i] >= 0, i].astype(float); m = v.mean(); s2 = v.var(ddof=1)
        mus.append(m); vs.append(s2); Fs.append(s2 / m if m > 0 else np.nan); pool.append(v)
    allv = np.concatenate(pool); mus = np.array(mus); vs = np.array(vs); Fs = np.array(Fs)
    return dict(mu=mus, var=vs, F=Fs, F_median=float(np.nanmedian(Fs)), mu_median=float(np.median(mus)), mu_pooled=float(allv.mean()),
                pool_fano=float(allv.var(ddof=1) / allv.mean()) if allv.mean() > 0 else np.nan,
                pool_fano_pop=float(allv.var() / allv.mean()) if allv.mean() > 0 else np.nan,
                max=int(allv.max()), n_ge10=int((allv >= 10).sum()), n_tasks=int(len(allv)), n_F_nan=int(np.isnan(Fs).sum()))


def rankavg(a):
    a = np.asarray(a); s = np.argsort(a, kind="mergesort"); inv = np.empty(len(a), int); inv[s] = np.arange(len(a))
    a_s = a[s]; obs = np.r_[True, a_s[1:] != a_s[:-1]]; dense = obs.cumsum()[inv]; cnt = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def spearman(a, b):
    ra, rb = rankavg(a), rankavg(b)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


SCAL = ("F_median", "mu_median", "mu_pooled", "pool_fano", "pool_fano_pop", "max", "n_ge10")


def summarize_surr(stat_list):
    out = {}
    for key in SCAL:
        v = np.array([s[key] for s in stat_list], float)
        out[key] = dict(median=float(np.nanmedian(v)), q05=float(np.nanquantile(v, 0.05)), q95=float(np.nanquantile(v, 0.95)),
                        min=float(np.nanmin(v)), max=float(np.nanmax(v)))
    return out


def per_target_band(act, surr_mat):
    """act: (n_t,), surr_mat: (R, n_t). Fraction of targets inside [q05, q95] of the R surrogate values, and above q95."""
    q05 = np.nanquantile(surr_mat, 0.05, axis=0); q95 = np.nanquantile(surr_mat, 0.95, axis=0)
    ok = np.isfinite(act) & np.isfinite(q05) & np.isfinite(q95)
    inside = (act >= q05) & (act <= q95) & ok; above = (act > q95) & ok; below = (act < q05) & ok
    return dict(n_used=int(ok.sum()), frac_inside_90=float(inside.sum() / ok.sum()), frac_above_q95=float(above.sum() / ok.sum()),
                frac_below_q05=float(below.sum() / ok.sum()))


def common_cause(Mact, Msurr, ids):
    """Mact (90, H); Msurr (R, 90, H). Spearman over (i, T) of actual vs surrogate mean; per-target var ratios."""
    Mbar = Msurr.mean(0); Vr = Msurr.var(0, ddof=1)          # within-(T,i) surrogate variance across r
    R = Msurr.shape[0]
    xa, xb, ratio, ratio_c, sp_in, pe_in = [], [], [], [], [], []
    for i in ids:
        rows = Mact[:, i] >= 0
        a = Mact[rows, i].astype(float); m = Mbar[rows, i]
        xa.append(a); xb.append(m)
        va = a.var(ddof=1)
        if va > 0:
            vm = m.var(ddof=1); noise = Vr[rows, i].mean() / R
            ratio.append(vm / va); ratio_c.append((vm - noise) / va)
        if a.std() > 0 and m.std() > 0:
            sp_in.append(spearman(a, m)); pe_in.append(float(np.corrcoef(a, m)[0, 1]))
    xa = np.concatenate(xa); xb = np.concatenate(xb)
    return dict(spearman_iT=spearman(xa, xb), pearson_iT=float(np.corrcoef(xa, xb)[0, 1]),
                varratio_median=float(np.median(ratio)), varratio_q25=float(np.quantile(ratio, 0.25)), varratio_q75=float(np.quantile(ratio, 0.75)),
                varratio_noisecorr_median=float(np.median(ratio_c)), n_targets_var=len(ratio),
                within_target_spearman_median=float(np.nanmedian(sp_in)), within_target_pearson_median=float(np.nanmedian(pe_in)),
                n_targets_within=len(sp_in), surr_mean_of_mean=float(xb.mean()), act_mean=float(xa.mean()))


def main():
    tag, npz, gate = sys.argv[1:4]; R = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    t0 = time.time()
    Nc, Cc, kk, store, K = counts(npz, gate)
    res = dict(tag=tag, npz=npz, gate=gate, R=R, convention="M1: T = 1..100 state end0[T-1]; phase 2 = T 11..100; targets >= 30 alive phase-2 tasks")
    # ---- MATCH against the cobirth_seeds_1013 verifier
    v = np.load(os.path.join(VER, f"vcounts_{tag}.npz")); vs = json.load(open(os.path.join(VER, f"vstats_{tag}.json")))
    alive_all = Nc >= 0
    res["match"] = dict(N_equal_all=bool((v["N"] == Nc).all()), N_n_diff=int((v["N"] != Nc).sum()),
                        C_equal_alive=bool((v["C"][alive_all] == Cc[alive_all]).all()), k_equal=bool((v["k"] == kk).all()),
                        ref=dict(fano_median=vs["fano_median"], pool_fano=vs["pool_fano"], mu_pooled=vs["mu_pooled"], max_N=vs["max_N"],
                                 n_N_ge10=vs["n_N_ge10"], C_fano=vs["C_fano"], n_targets=vs["n_targets"]))
    # ---- deciles of K
    order = np.argsort(K, kind="stable"); nD = 10; n = len(K)
    blocks = [order[d * n // nD:(d + 1) * n // nD] for d in range(nD)]
    dec = np.empty(n, int)
    for d_, idx in enumerate(blocks): dec[idx] = d_
    res["K_decile_edges"] = [float(K[idx].min()) for idx in blocks] + [float(K.max())]
    H = Nc.shape[1]
    Na = Nc[PH]; Ca = Cc[PH]
    ids = [i for i in range(H) if (Na[:, i] >= 0).sum() >= 30]
    res["n_targets"] = len(ids)
    sN = series_stats(Na, ids); sC = series_stats(Ca, ids)
    res["match"]["own"] = dict(fano_median=sN["F_median"], pool_fano=sN["pool_fano"], mu_pooled=sN["mu_pooled"], max_N=sN["max"],
                               n_N_ge10=sN["n_ge10"], C_fano_pop_all_alive=float(Ca[Ca >= 0].var() / Ca[Ca >= 0].mean()))
    m_ok = (res["match"]["N_equal_all"] and res["match"]["C_equal_alive"]
            and abs(sN["F_median"] - vs["fano_median"]) < 1e-9 and abs(sN["pool_fano"] - vs["pool_fano"]) < 1e-9
            and abs(res["match"]["own"]["C_fano_pop_all_alive"] - vs["C_fano"]) < 1e-9 and sN["max"] == vs["max_N"] and sN["n_ge10"] == vs["n_N_ge10"])
    res["match"]["verdict"] = "MATCH" if m_ok else "MISMATCH"
    # ---- B5: actual occupancy by K-decile (alive-unit gates, phase 2 pooled), and where T_i falls
    open_d = np.zeros(nD); tot_d = np.zeros(nD); thr_d = np.zeros(nD)
    for T in range(11, 101):
        O, T3, k, alive = store[T]
        for d_, idx in enumerate(blocks):
            open_d[d_] += O[np.ix_(idx, alive)].sum(); tot_d[d_] += len(idx) * alive.sum()
        thr_d += np.bincount(dec[T3[:, alive].ravel()], minlength=nD)
    res["B5"] = dict(occupancy_by_decile=(open_d / tot_d).tolist(), occupancy_overall=float(open_d.sum() / tot_d.sum()),
                     Ti_share_by_decile=(thr_d / thr_d.sum()).tolist(), K_decile_mean=[float(K[idx].mean()) for idx in blocks])
    # ---- surrogates
    Ns = {"strat": np.full((R, 90, H), -1, np.int16), "flat": np.full((R, 90, H), -1, np.int16)}
    Cs = {"strat": np.full((R, 90, H), -1, np.int16), "flat": np.full((R, 90, H), -1, np.int16)}
    allidx = [np.arange(n)]
    for null, bl in (("strat", blocks), ("flat", allidx)):
        for r in range(R):
            rng = np.random.default_rng(1000 + r)
            for T in range(11, 101):
                O, T3, k, alive = store[T]
                P = permute(O, rng, bl)
                if r == 0 and T == 11:  # sanity: per-unit per-decile open counts preserved (strat) / total preserved (flat)
                    assert (P.sum(0) == O.sum(0)).all()
                    if null == "strat":
                        for idx in blocks: assert (P[idx].sum(0) == O[idx].sum(0)).all()
                n_, c_ = surr_counts(O, T3, k, alive, P)
                Ns[null][r, T - 11] = n_; Cs[null][r, T - 11] = c_
    np.savez_compressed(os.path.join(BIG, f"b_surr_{tag}.npz"), N_act=Na, C_act=Ca, Ns_strat=Ns["strat"], Cs_strat=Cs["strat"],
                        Ns_flat=Ns["flat"], Cs_flat=Cs["flat"], ids=np.array(ids), K=K, dec=dec)
    # ---- B1, B2, B3
    for qty, Mact, sA, S in (("N", Na, sN, Ns), ("C", Ca, sC, Cs)):
        out = dict(actual={k_: sA[k_] for k_ in SCAL} | dict(n_tasks=sA["n_tasks"], n_F_nan=sA["n_F_nan"]))
        out["actual"]["C_fano_pop_all_alive"] = float(Mact[Mact >= 0].var() / Mact[Mact >= 0].mean())
        for null in ("strat", "flat"):
            st = [series_stats(S[null][r], ids) for r in range(R)]
            Fm = np.array([s["F"] for s in st]); Mm = np.array([s["mu"] for s in st])
            out[null] = dict(summary=summarize_surr(st), n_F_nan_total=int(sum(s["n_F_nan"] for s in st)),
                             B2_F=per_target_band(sA["F"], Fm), B2_mu=per_target_band(sA["mu"], Mm),
                             B3=common_cause(Mact, S[null].astype(float), ids),
                             per_target_F_median_over_r=np.nanmedian(Fm, 0).tolist(), per_target_mu_median_over_r=np.median(Mm, 0).tolist())
        out["per_target"] = dict(i=ids, mu=sA["mu"].tolist(), F=sA["F"].tolist())
        res[qty] = out
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(os.path.join(RES, f"b_cond_{tag}.json"), "w"), indent=1)
    sm = lambda q, nl, key: res[q][nl]["summary"][key]["median"]
    print(f"[{tag}] {res['match']['verdict']} | N F_med act {sN['F_median']:.3f} strat {sm('N','strat','F_median'):.3f} flat {sm('N','flat','F_median'):.3f}"
          f" | C pool act {sC['pool_fano']:.3f} strat {sm('C','strat','pool_fano'):.3f} flat {sm('C','flat','pool_fano'):.3f}"
          f" | mu_med act {sN['mu_median']:.3f} strat {sm('N','strat','mu_median'):.3f} flat {sm('N','flat','mu_median'):.3f} | {res['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
