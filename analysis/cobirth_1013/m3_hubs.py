"""M3 (cobirth_1013): hubs, width heterogeneity, and where the clustering of lifters comes from.

Box: 1-hidden-layer MLP 784-100-10 on 1,200 fixed MNIST inputs, hard leaky ReLU slope a in {0.1, 0.3, 0.5}, CE, Adam,
batch 16, 4,000 updates/task, iid uniform labels redrawn each task, 100 tasks, seed 0.  Saved task-end weights only
(no training).  Lifter (loose bridge) criterion is copied verbatim from H4 `stoch_bridges.py: bridges_at` (= P161/P176):
  target i alive (k_i = |O_i| >= 1), thresholds T_i = top-3 of e'_in = K_n * dD_in/dz (smooth gate d = a + (1-a) sigma(z/tau),
  tau = 0.1 sd_n z_i + 1e-12), candidate j != i with k_j >= 1 is a lifter  <=>  |O_j & T_i| >= 2 and |O_j & O_i| >= (k_i+1)//2.
Task convention (same as stoch/p161/p170): "task t" = end0[t-2] (t = 2..101); phase 2 = t >= 10 (92 states);
transitions t -> t+1 for t = 10..100 (91 transitions, births land on t+1 = 11..101).  Reproduction targets (stoch, t = 10..100
rows / births at t+1 = 11..100): rows 1,511 / 2,836 / 7,712; births 1,399 / 2,557 / 6,649; N_t max 7 / 14 / 10; >= 10: 0 / 3 / 1.

Occupancy f_n: f^u_n = (# alive units open at n) / (# alive)  [unweighted];  f^g_n = sum_j g_j 1[z_jn > 0] / sum_j g_j with
g_j = |v_j^c|^2 (P161's strength-weighted version).  f^{-S}_n excludes the units in S from numerator and denominator
(removes the tautological contribution of the pair (i, j) itself, which is open at every qualifying input by definition).
Null U (P159 "uniform independent ranks"): each candidate j's open set replaced by a uniform random subset of size k_j,
target's T_i / O_i actual; MC R reps.  Analytic (no overlap) lambda_bin_i = sum_j q3(p_j) P(Bin(k_i, p_j) >= need_i),
q3(p) = 3p^2 - 2p^3, p_j = k_j/N; Chernoff form (BRIDGE_CONDITION B3): lambda_i <= sum_j q3(p_j) [4p_j(1-p_j)]^{k_i/2} (p_j < 1/2).

usage: m3_hubs.py <a01|a03|a05>   (one arm; writes RAW/m3_<tag>.{json,npz,log})
       m3_hubs.py combine         (merges the three arm JSONs into RES/m3_hubs.json)
CPU only, OMP 1.  Memory < 1 GB per arm.
"""
import os, sys, json, math, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_1013/m3"
RES = "/home/issan/Projects/claude/wt/cobirth_1013/results/cobirth_1013"
SRC = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
KBINS = [(1, 3), (4, 10), (11, 30), (31, 100), (101, 300), (301, 1200)]
KBINS_FINE = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 1200)]
R_MC = 20
HUB_FRAC = 0.05
os.makedirs(RAW, exist_ok=True); os.makedirs(RES, exist_ok=True)


def fl(x):
    if isinstance(x, (np.floating, float)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, np.ndarray): return [fl(v) for v in x.tolist()]
    if isinstance(x, dict): return {str(k): fl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [fl(v) for v in x]
    return x


def avg_rank(x):
    """average ranks (1-based) with ties averaged, no scipy"""
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


def pct_rank(v, i):
    """percentile (0..1) of v[i] within v (share of entries strictly below + half ties)"""
    return float(((v < v[i]).sum() + 0.5 * (v == v[i]).sum()) / len(v))


def bin_tail_vec(k, p, need):
    """P(Bin(k, p_j) >= need) for a vector p, scalar k, need; exact in log space"""
    p = np.asarray(p, float)
    if need <= 0: return np.ones_like(p)
    if need > k: return np.zeros_like(p)
    x = np.arange(0, k + 1); r = np.arange(1, k + 1)
    logC = np.concatenate([[0.0], np.cumsum(np.log(k - r + 1) - np.log(r))])
    pc = np.clip(p, 1e-300, 1 - 1e-16)
    lt = logC[need:, None] + x[need:, None] * np.log(pc)[None, :] + (k - x[need:])[:, None] * np.log1p(-pc)[None, :]
    return np.exp(lt).sum(0)


def var_decomp(vals, groups):
    """population variance of vals decomposed into across-group (variance of group means) and within-group parts"""
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


def run_arm(tag):
    t0 = time.time(); logf = open(os.path.join(RAW, f"m3_{tag}.log"), "w")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    src, a = SRC[tag]
    sys.path.insert(0, ROOT); from common import load_X
    dd = np.load(os.path.join(ROOT, src)); X = load_X(dd["subset"]); N = X.shape[0]; H = dd["end0"].shape[1]
    Xt = np.concatenate([X, np.ones((N, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar; Rn = np.linalg.norm(Xt - xbar[None, :], axis=1)
    nhub = int(round(HUB_FRAC * N)); hubR = np.zeros(N, bool); hubR[np.argsort(-Rn)[:nhub]] = True; hubK = np.zeros(N, bool); hubK[np.argsort(-Kv)[:nhub]] = True
    rng = np.random.default_rng(20261013)
    P(f"[{tag}] a={a} N={N} H={H} | K_n mean {Kv.mean():.3f} sd {Kv.std():.3f} | R_n mean {Rn.mean():.3f} sd {Rn.std():.3f} | Spearman(K_n, R_n) {spearman(Kv, Rn):+.3f} | hub sets: top {nhub} by R_n / by K_n, overlap {int((hubR & hubK).sum())}")

    # ---------------- states ----------------
    tasks = list(range(2, 102)); ST = {}
    def make_state(W1, b1, W2):
        W = np.concatenate([W1.astype(np.float64), b1.astype(np.float64)[:, None]], 1); Z = Xt @ W.T; tau = 0.1 * Z.std(0) + 1e-12
        s = 0.5 * (1 + np.tanh(0.5 * Z / tau)); Dp = (1 - a) * s * (1 - s) / tau; O = Z > 0; k = O.sum(0); alive = k > 0
        Vc = W2.T.astype(np.float64) - W2.T.astype(np.float64).mean(1, keepdims=True); g = (Vc * Vc).sum(1)
        m = W @ xbar; sd = Z.std(0); r = m / np.maximum(sd, 1e-12)
        return dict(Z=Z, tau=tau, Dp=Dp, O=O, k=k, alive=alive, g=g, m=m, sd=sd, r=r, Vc=Vc)
    for t in tasks: ST[t] = make_state(dd["end0"][t - 2], dd["end1"][t - 2], dd["end2"][t - 2])
    ST[1] = make_state(dd["init0"], dd["init1"], dd["init2"])  # initial weights (before task 1)
    P(f"[{tag}] states loaded ({time.time() - t0:.0f}s)")

    def bridges_at(st):
        """verbatim logic of stoch_bridges.py: bridges_at (H4) = P161/P176 loose criterion"""
        O, Dp = st["O"], st["Dp"]; alive = O.any(0); k = O.sum(0); need = (k + 1) // 2; Oi = O.astype(np.int32); ov = Oi.T @ Oi
        E = Kv[:, None] * Dp; T = np.argsort(-E, axis=0)[:3, :]; thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
        isb = (thr >= 2) & (ov >= need[:, None]); isb[np.diag_indices(H)] = False; isb[~alive, :] = False; isb[:, ~alive] = False
        return dict(T=T, thr=thr, ov=ov, isb=isb, need=need, k=k, alive=alive)
    B = {t: bridges_at(ST[t]) for t in [1] + tasks}
    pairs = {t: set(zip(*np.where(B[t]["isb"]))) for t in B}
    ph2 = [t for t in tasks if t >= 10]                      # 92 states, t = 10..101
    trans = [t for t in ph2 if t + 1 in ST]                 # 91 transitions t -> t+1

    # ---------------- reproduction check ----------------
    rows_1010 = sum(len(pairs[t]) for t in range(10, 101)); births_1110 = sum(len(pairs[t + 1] - pairs[t]) for t in range(10, 100))
    Ncounts = np.concatenate([B[t]["isb"].sum(1)[B[t]["alive"]] for t in ph2])
    repro = dict(rows_t10_100=int(rows_1010), births_t11_100=int(births_1110), N_max=int(Ncounts.max()), N_ge10=int((Ncounts >= 10).sum()), N_ge7=int((Ncounts >= 7).sum()), alive_unit_tasks=int(len(Ncounts)))
    P(f"[{tag}] REPRO rows(t=10..100) {rows_1010} | births(t+1=11..100) {births_1110} | N max {Ncounts.max()} | N>=10 {int((Ncounts >= 10).sum())} | N>=7 {int((Ncounts >= 7).sum())} | alive unit-tasks (t=10..101) {len(Ncounts)}")

    # ---------------- (1) occupancy ----------------
    def occ(st, excl=()):
        alive = st["alive"].copy()
        for u in excl: alive[u] = False
        na = int(alive.sum()); cnt = st["O"][:, alive].sum(1); return cnt / max(na, 1), cnt, na
    def occ_g(st):
        g = st["g"]; return (st["O"] * g[None, :]).sum(1) / g.sum()
    F_u = np.zeros((len(ph2), N)); F_g = np.zeros((len(ph2), N)); occ_rows = []
    hubF = {}
    def occ_stats(f, st):
        srt = np.sort(f)[::-1]; mass = f.sum(); top = srt[:nhub].sum() / mass if mass > 0 else float("nan")
        cum = np.cumsum(np.sort(f)); gini = 1 - 2 * (cum.sum() / (cum[-1] * len(f))) + 1.0 / len(f) if cum[-1] > 0 else float("nan")
        return dict(median=float(np.median(f)), q90=float(np.quantile(f, .9)), q99=float(np.quantile(f, .99)), max=float(f.max()), ge_half=float((f >= 0.5).mean()), zero=float((f == 0).mean()), mean=float(f.mean()), top5_share=float(top), gini=float(gini))
    for q, t in enumerate(ph2):
        st = ST[t]; fu, cnt, na = occ(st); fg = occ_g(st); F_u[q] = fu; F_g[q] = fg
        hf = np.zeros(N, bool); hf[np.argsort(-fu, kind="mergesort")[:nhub]] = True; hubF[t] = hf
        # per-unit enrichment on hub sets: P(j open | hub)/p_j
        O = st["O"][:, st["alive"]]; p = O.mean(0)
        enrR = O[hubR].mean(0) / p; enrK = O[hubK].mean(0) / p; enrF = O[hf].mean(0) / p
        kal = st["k"][st["alive"]]
        occ_rows.append(dict(t=t, n_alive=na, mean_p=float(p.mean()), median_k=float(np.median(kal)), u=occ_stats(fu, st), g=occ_stats(fg, st),
                             sp_fu_K=spearman(fu, Kv), sp_fu_R=spearman(fu, Rn), sp_fg_K=spearman(fg, Kv), sp_fg_R=spearman(fg, Rn),
                             enrich_R_med=float(np.median(enrR)), enrich_K_med=float(np.median(enrK)), enrich_F_med=float(np.median(enrF)),
                             gate_share_R=float(O[hubR].sum() / O.sum()), gate_share_K=float(O[hubK].sum() / O.sum()), gate_share_F=float(O[hf].sum() / O.sum()),
                             jacc_F_R=float((hf & hubR).sum() / (hf | hubR).sum()), jacc_F_K=float((hf & hubK).sum() / (hf | hubK).sum())))
    jac_FF = [float((hubF[t] & hubF[t + 1]).sum() / (hubF[t] | hubF[t + 1]).sum()) for t in trans]
    # phase-1 and init reference
    ref = {}
    for t in (1, 2, 3, 5, 9):
        st = ST[t]; fu, cnt, na = occ(st); fg = occ_g(st); ref[t] = dict(n_alive=na, u=occ_stats(fu, st), g=occ_stats(fg, st), sp_fu_K=spearman(fu, Kv), sp_fu_R=spearman(fu, Rn))
    def med(key, sub=None):
        vals = [(r[sub][key] if sub else r[key]) for r in occ_rows]; return float(np.median(vals))
    occ_summary = dict(n_states=len(ph2), n_alive_med=med("n_alive"), mean_p_med=med("mean_p"), median_k_med=med("median_k"),
                       u={k: med(k, "u") for k in ("median", "q90", "q99", "max", "ge_half", "zero", "mean", "top5_share", "gini")},
                       g={k: med(k, "g") for k in ("median", "q90", "q99", "max", "ge_half", "zero", "mean", "top5_share", "gini")},
                       u_pooled=occ_stats(F_u.ravel(), None), g_pooled=occ_stats(F_g.ravel(), None),
                       sp_fu_K_med=med("sp_fu_K"), sp_fu_R_med=med("sp_fu_R"), sp_fg_K_med=med("sp_fg_K"), sp_fg_R_med=med("sp_fg_R"),
                       sp_fu_K_range=[float(min(r["sp_fu_K"] for r in occ_rows)), float(max(r["sp_fu_K"] for r in occ_rows))],
                       sp_fu_R_range=[float(min(r["sp_fu_R"] for r in occ_rows)), float(max(r["sp_fu_R"] for r in occ_rows))],
                       enrich_R_med=med("enrich_R_med"), enrich_K_med=med("enrich_K_med"), enrich_F_med=med("enrich_F_med"),
                       gate_share_R_med=med("gate_share_R"), gate_share_K_med=med("gate_share_K"), gate_share_F_med=med("gate_share_F"), uniform_share=HUB_FRAC,
                       jacc_F_R_med=med("jacc_F_R"), jacc_F_K_med=med("jacc_F_K"), jacc_F_next_med=float(np.median(jac_FF)), jacc_F_next_min=float(np.min(jac_FF)),
                       jacc_F_first_last=float((hubF[ph2[0]] & hubF[ph2[-1]]).sum() / (hubF[ph2[0]] | hubF[ph2[-1]]).sum()),
                       ref={str(t): ref[t] for t in ref})
    # how many inputs are ever in the top-5% occupancy set across phase 2 (fixed hubs vs drifting)
    everF = np.zeros(N, int)
    for t in ph2: everF += hubF[t]
    occ_summary["hubF_distinct_inputs"] = int((everF > 0).sum()); occ_summary["hubF_inputs_in_ge_half_states"] = int((everF >= len(ph2) / 2).sum()); occ_summary["hubF_inputs_in_ge_90pct_states"] = int((everF >= 0.9 * len(ph2)).sum())
    P(f"[{tag}] (1) occupancy f^u (medians over 92 states): median {occ_summary['u']['median']:.4f} q90 {occ_summary['u']['q90']:.4f} q99 {occ_summary['u']['q99']:.4f} max {occ_summary['u']['max']:.3f} ge1/2 {occ_summary['u']['ge_half']:.3f} zero {occ_summary['u']['zero']:.3f} top5%share {occ_summary['u']['top5_share']:.3f} gini {occ_summary['u']['gini']:.3f} | f^g: median {occ_summary['g']['median']:.4f} max {occ_summary['g']['max']:.3f} ge1/2 {occ_summary['g']['ge_half']:.3f}")
    P(f"[{tag}] (1) Spearman(f^u, K_n) median {occ_summary['sp_fu_K_med']:+.3f} range {occ_summary['sp_fu_K_range']} | Spearman(f^u, R_n) median {occ_summary['sp_fu_R_med']:+.3f} range {occ_summary['sp_fu_R_range']} | enrichment P(open|hub)/p: R-hubs {occ_summary['enrich_R_med']:.2f} K-hubs {occ_summary['enrich_K_med']:.2f} | gate share on R/K/F hubs {occ_summary['gate_share_R_med']:.3f}/{occ_summary['gate_share_K_med']:.3f}/{occ_summary['gate_share_F_med']:.3f} (uniform 0.05)")
    P(f"[{tag}] (1) occupancy hubs (top 5% f^u): Jaccard with R-hubs {occ_summary['jacc_F_R_med']:.3f}, K-hubs {occ_summary['jacc_F_K_med']:.3f}, next task {occ_summary['jacc_F_next_med']:.3f} (min {occ_summary['jacc_F_next_min']:.3f}), t10 vs t101 {occ_summary['jacc_F_first_last']:.3f} | distinct inputs ever in top-5%: {occ_summary['hubF_distinct_inputs']} (in >=50% of states {occ_summary['hubF_inputs_in_ge_half_states']}, >=90% {occ_summary['hubF_inputs_in_ge_90pct_states']})")
    for t in (1, 2, 3, 5, 9): P(f"[{tag}] (1) ref t{t}: alive {ref[t]['n_alive']} f^u median {ref[t]['u']['median']:.3f} max {ref[t]['u']['max']:.3f} ge1/2 {ref[t]['u']['ge_half']:.3f} | f^g median {ref[t]['g']['median']:.3f} ge1/2 {ref[t]['g']['ge_half']:.3f} | Spearman(f^u,K) {ref[t]['sp_fu_K']:+.3f} (f^u,R) {ref[t]['sp_fu_R']:+.3f}")

    # ---------------- (2) births attributed to inputs ----------------
    def input_sets_for_pair(st, bt, i, j):
        Ti = bt["T"][:, i]; Qthr = [int(n) for n in Ti if st["O"][n, j]]; Qcov = np.where(st["O"][:, i] & st["O"][:, j])[0]; return Qthr, Qcov
    def collect(kind_pairs, state_t):
        """kind_pairs: list of (t_state, i, j) evaluated at state t_state"""
        rec = dict(thr=[], cov=[], thr_hubF=[], thr_hubR=[], thr_hubK=[], cov_hubF=[], cov_hubR=[], cov_hubK=[], thr_g=[], cov_g=[], within_thr=[], within_cov=[], n=0, thr_raw=[], thr_L=[], cov_L=[], within_thr_L=[], within_cov_L=[])
        for (t, i, j) in kind_pairs:
            st = ST[t]; bt = B[t]; Qthr, Qcov = input_sets_for_pair(st, bt, i, j)
            fex, cnt, na = occ(st, excl=(i, j)); fg = occ_g(st)
            if na < 1: continue
            Lset = tuple({int(i), int(j)} | set(int(x) for x in np.where(bt["isb"][i])[0])); fexL, _, naL = occ(st, excl=Lset)
            hf = hubF.get(t)
            rec["n"] += 1
            rec["thr"].append(float(np.mean(fex[Qthr]))); rec["cov"].append(float(np.mean(fex[Qcov])))
            rec["thr_L"].append(float(np.mean(fexL[Qthr]))); rec["cov_L"].append(float(np.mean(fexL[Qcov])))
            rec["within_thr_L"].append(float(np.mean(fexL[Qthr]) - np.mean(fexL[bt["T"][:, i]]))); rec["within_cov_L"].append(float(np.mean(fexL[Qcov]) - np.mean(fexL[np.where(st["O"][:, i])[0]])))
            rec["thr_raw"].append(float(np.mean(occ(st)[0][Qthr])))
            rec["thr_g"].append(float(np.mean(fg[Qthr]))); rec["cov_g"].append(float(np.mean(fg[Qcov])))
            if hf is not None: rec["thr_hubF"].append(float(hf[Qthr].mean())); rec["cov_hubF"].append(float(hf[Qcov].mean()))
            rec["thr_hubR"].append(float(hubR[Qthr].mean())); rec["thr_hubK"].append(float(hubK[Qthr].mean())); rec["cov_hubR"].append(float(hubR[Qcov].mean())); rec["cov_hubK"].append(float(hubK[Qcov].mean()))
            # within-target: the lifter's qualifying inputs vs the target's full T_i / O_i (occupancy excluding i and j)
            Ti = bt["T"][:, i]; Oi = np.where(st["O"][:, i])[0]
            rec["within_thr"].append(float(np.mean(fex[Qthr]) - np.mean(fex[Ti]))); rec["within_cov"].append(float(np.mean(fex[Qcov]) - np.mean(fex[Oi])))
        out = dict(n=rec["n"])
        for key in ("thr", "cov", "thr_raw", "thr_g", "cov_g", "thr_hubF", "thr_hubR", "thr_hubK", "cov_hubF", "cov_hubR", "cov_hubK", "within_thr", "within_cov", "thr_L", "cov_L", "within_thr_L", "within_cov_L"):
            v = np.array(rec[key], float)
            out[key] = dict(mean=float(v.mean()), median=float(np.median(v)), q90=float(np.quantile(v, .9))) if len(v) else None
        return out
    born = [(t + 1, i, j) for t in trans for (i, j) in (pairs[t + 1] - pairs[t])]
    died = [(t, i, j) for t in trans for (i, j) in (pairs[t] - pairs[t + 1])]
    pers = [(t, i, j) for t in trans for (i, j) in (pairs[t] & pairs[t + 1])]
    allp = [(t, i, j) for t in ph2 for (i, j) in pairs[t]]
    bs = collect(born, None); ds = collect(died, None); ps = collect(pers, None); aps = collect(allp, None)
    # baselines at the same states (t+1 for births): all threshold inputs of alive targets (excl. the target), all open gates (excl. the unit), uniform
    base = dict(thr=[], thr_hubF=[], thr_hubR=[], thr_hubK=[], thr_g=[], open=[], open_hubF=[], open_hubR=[], open_hubK=[], uni=[], uni_g=[], thr_L=[], open_L=[], thr_L_withlifters=[], thr_L_nolifters=[])
    for t in ph2:
        st = ST[t]; bt = B[t]; fu, cnt, na = occ(st); fg = occ_g(st); hf = hubF[t]
        for i in np.where(st["alive"])[0]:
            Ti = bt["T"][:, i]; fex_i = (cnt - st["O"][:, i]) / max(na - 1, 1)
            Lset = tuple({int(i)} | set(int(x) for x in np.where(bt["isb"][i])[0])); fexL = occ(st, excl=Lset)[0]
            base["thr_L"].append(float(fexL[Ti].mean())); base["open_L"].append(float(fexL[np.where(st["O"][:, i])[0]].mean()))
            (base["thr_L_withlifters"] if len(Lset) > 1 else base["thr_L_nolifters"]).append(float(fexL[Ti].mean()))
            base["thr"].append(float(fex_i[Ti].mean())); base["thr_hubF"].append(float(hf[Ti].mean())); base["thr_hubR"].append(float(hubR[Ti].mean())); base["thr_hubK"].append(float(hubK[Ti].mean())); base["thr_g"].append(float(fg[Ti].mean()))
            Oi = np.where(st["O"][:, i])[0]; base["open"].append(float(fex_i[Oi].mean())); base["open_hubF"].append(float(hf[Oi].mean())); base["open_hubR"].append(float(hubR[Oi].mean())); base["open_hubK"].append(float(hubK[Oi].mean()))
        base["uni"].append(float(fu.mean())); base["uni_g"].append(float(fg.mean()))
    base_s = {k: dict(mean=float(np.mean(v)), median=float(np.median(v))) for k, v in base.items() if len(v)}
    births = dict(born=bs, died=ds, persist=ps, all_pairs=aps, baseline=base_s, n_born=len(born), n_died=len(died), n_persist=len(pers))
    P(f"[{tag}] (2) births n {len(born)} | occupancy f^(-ij) of qualifying THRESHOLD inputs: born {bs['thr']['mean']:.4f} died {ds['thr']['mean']:.4f} persist {ps['thr']['mean']:.4f} | baseline all threshold inputs of alive targets f^(-i) {base_s['thr']['mean']:.4f} | all open gates f^(-i) {base_s['open']['mean']:.4f} | uniform {base_s['uni']['mean']:.4f}")
    P(f"[{tag}] (2) f^(-L) (target + all its lifters excluded): born thr {bs['thr_L']['mean']:.4f} died {ds['thr_L']['mean']:.4f} persist {ps['thr_L']['mean']:.4f} | all thr inputs f^(-L_i) {base_s['thr_L']['mean']:.4f} (targets with lifters {base_s['thr_L_withlifters']['mean']:.4f}, without {base_s['thr_L_nolifters']['mean']:.4f}) | born cov {bs['cov_L']['mean']:.4f} vs all open f^(-L_i) {base_s['open_L']['mean']:.4f} | within-target (f^-L): thr {bs['within_thr_L']['mean']:+.4f} cov {bs['within_cov_L']['mean']:+.4f}")
    P(f"[{tag}] (2) qualifying COVERAGE inputs: born {bs['cov']['mean']:.4f} died {ds['cov']['mean']:.4f} persist {ps['cov']['mean']:.4f} | raw f^u (incl. i,j) at born thr inputs {bs['thr_raw']['mean']:.4f} | strength-weighted f^g at born thr {bs['thr_g']['mean']:.4f} vs all thr {base_s['thr_g']['mean']:.4f}")
    P(f"[{tag}] (2) hub shares of born threshold inputs: F-hubs {bs['thr_hubF']['mean']:.3f} (all thr {base_s['thr_hubF']['mean']:.3f}) R-hubs {bs['thr_hubR']['mean']:.3f} (all thr {base_s['thr_hubR']['mean']:.3f}) K-hubs {bs['thr_hubK']['mean']:.3f} (all thr {base_s['thr_hubK']['mean']:.3f}) uniform 0.05 | within-target: born thr inputs minus target's T_i {bs['within_thr']['mean']:+.4f}, born cov inputs minus target's O_i {bs['within_cov']['mean']:+.4f} (died {ds['within_thr']['mean']:+.4f}/{ds['within_cov']['mean']:+.4f})")

    # co-born lifters at the same (i, t+1): similarity of their open sets vs baselines
    def jacc_mat(st):
        Oi = st["O"].astype(np.int32); ov = Oi.T @ Oi; k = st["k"]; un = k[:, None] + k[None, :] - ov; return np.where(un > 0, ov / np.maximum(un, 1), 0.0)
    cob_j = []; cob_T = []; cob_wide_both = []; base_j = []; cross_j = []
    byti = {}
    for (t, i, j) in born: byti.setdefault((t, i), []).append(j)
    for t in sorted(set(t for (t, i) in byti)):
        st = ST[t]; J = jacc_mat(st); alive = np.where(st["alive"])[0]
        iu = np.triu_indices(len(alive), 1); base_j.append(float(J[np.ix_(alive, alive)][iu].mean()))
        # lifters of different targets at the same state (any bridge pairs)
        lift_by_target = {}
        for (i, j) in pairs[t]: lift_by_target.setdefault(i, []).append(j)
        tg = list(lift_by_target)
        cr = []
        for x in range(len(tg)):
            for y in range(x + 1, len(tg)):
                for j1 in lift_by_target[tg[x]]:
                    for j2 in lift_by_target[tg[y]]:
                        if j1 != j2: cr.append(J[j1, j2])
        if cr: cross_j.append(float(np.mean(cr)))
        for (tt, i), js in byti.items():
            if tt != t or len(js) < 2: continue
            Ti = B[t]["T"][:, i]
            for x in range(len(js)):
                for y in range(x + 1, len(js)):
                    j1, j2 = js[x], js[y]; cob_j.append(float(J[j1, j2])); cob_T.append(int((st["O"][Ti, j1] & st["O"][Ti, j2]).sum()))
    cob = dict(n_pairs=len(cob_j), jaccard_mean=float(np.mean(cob_j)) if cob_j else None, jaccard_median=float(np.median(cob_j)) if cob_j else None,
               shared_T_dist={str(c): int(sum(1 for v in cob_T if v == c)) for c in (0, 1, 2, 3)}, baseline_all_alive_pairs_jaccard=float(np.mean(base_j)) if base_j else None,
               lifters_of_different_targets_jaccard=float(np.mean(cross_j)) if cross_j else None)
    births["coborn"] = cob
    P(f"[{tag}] (2) co-born lifter pairs (same target, same t+1) n {cob['n_pairs']}: Jaccard of open sets mean {cob['jaccard_mean']} | all alive pairs {cob['baseline_all_alive_pairs_jaccard']} | lifters of different targets {cob['lifters_of_different_targets_jaccard']} | shared threshold inputs 0/1/2/3: {cob['shared_T_dist']} (>=1 is forced by 2-of-3)")

    # ---------------- (3) target heterogeneity: N_i vs k_i, nulls ----------------
    recs = []  # per (t, i) alive in phase 2
    lamU_mc = {}; degU = {}
    for t in ph2:
        st = ST[t]; bt = B[t]; alive = st["alive"]; k = st["k"]; need = bt["need"]; T = bt["T"]; p = k / N
        Nobs = bt["isb"].sum(1); q3 = 3 * p ** 2 - 2 * p ** 3
        # MC null U
        NU = np.zeros((R_MC, H)); DU = np.zeros((R_MC, H)); CU = np.zeros((R_MC, H))
        for rep in range(R_MC):
            U = rng.random((N, H)); ranks = np.argsort(np.argsort(U, axis=0), axis=0); On = ranks < k[None, :]
            thr_n = np.stack([On[T[:, i], :].sum(0) for i in range(H)], 0); ov_n = st["O"].astype(np.int32).T @ On.astype(np.int32)
            isb_n = (thr_n >= 2) & (ov_n >= need[:, None]); isb_n[np.diag_indices(H)] = False; isb_n[~alive, :] = False; isb_n[:, ~alive] = False
            NU[rep] = isb_n.sum(1); DU[rep] = isb_n.sum(0)
            pool_n = (thr_n >= 2) & alive[None, :]; pool_n[np.diag_indices(H)] = False; CU[rep] = pool_n.sum(1)
        lamU_mc[t] = NU; degU[t] = DU
        pool = (bt["thr"] >= 2) & alive[None, :]; pool[np.diag_indices(H)] = False; Cobs = pool.sum(1)
        for i in np.where(alive)[0]:
            ki = int(k[i]); nd = int(need[i]); others = alive.copy(); others[i] = False
            lam_bin = float((q3[others] * bin_tail_vec(ki, p[others], nd)).sum())
            po = p[others]; ch = np.where(po < 0.5, (4 * po * (1 - po)) ** (ki / 2.0), 1.0); lam_ch = float((q3[others] * ch).sum())
            a_i = int(st["O"][T[:, i], i].sum())  # overlap |T_i & O_i|
            recs.append(dict(t=t, i=int(i), k=ki, need=nd, N=int(Nobs[i]), lamU=float(NU[:, i].mean()), lamU_var=float(NU[:, i].var()), lam_bin=lam_bin, lam_ch=lam_ch, a_i=a_i, r=float(st["r"][i]), p_mean_others=float(po.mean()), C=int(Cobs[i]), CU=float(CU[:, i].mean()), CU_var=float(CU[:, i].var())))
    tt = np.array([r["t"] for r in recs]); ii = np.array([r["i"] for r in recs]); kk = np.array([r["k"] for r in recs]); NN = np.array([r["N"] for r in recs], float)
    LU = np.array([r["lamU"] for r in recs]); LUv = np.array([r["lamU_var"] for r in recs]); LB = np.array([r["lam_bin"] for r in recs]); LC = np.array([r["lam_ch"] for r in recs]); AI = np.array([r["a_i"] for r in recs])
    CC = np.array([r["C"] for r in recs], float); CUm = np.array([r["CU"] for r in recs]); CUv = np.array([r["CU_var"] for r in recs])
    kb = np.array([kbin_of(k, KBINS) for k in kk]); kbf = np.array([kbin_of(k, KBINS_FINE) for k in kk])
    tab = []
    for b, (lo, hi) in enumerate(KBINS):
        s = kb == b
        if not s.any(): tab.append(dict(bin=f"{lo}-{hi}", n=0)); continue
        tab.append(dict(bin=f"{lo}-{hi}", n=int(s.sum()), share_targets=float(s.mean()), obs_mean=float(NN[s].mean()), obs_max=int(NN[s].max()), obs_ge3=int((NN[s] >= 3).sum()), obs_ge5=int((NN[s] >= 5).sum()), share_bridges=float(NN[s].sum() / NN.sum()),
                        lamU=float(LU[s].mean()), lam_bin=float(LB[s].mean()), lam_ch=float(LC[s].mean()), ratio_obs_U=(float(NN[s].mean() / LU[s].mean()) if LU[s].mean() > 0 else None), ratio_obs_bin=(float(NN[s].mean() / LB[s].mean()) if LB[s].mean() > 0 else None),
                        a_i_mean=float(AI[s].mean()), lamU_fano=(float(LUv[s].mean() / LU[s].mean()) if LU[s].mean() > 0 else None)))
    tabf = []
    for b, (lo, hi) in enumerate(KBINS_FINE):
        s = kbf == b
        if not s.any(): continue
        tabf.append(dict(bin=f"{lo}-{hi}", n=int(s.sum()), obs_mean=float(NN[s].mean()), obs_var=float(NN[s].var()), obs_max=int(NN[s].max()), lamU=float(LU[s].mean()), lam_bin=float(LB[s].mean()), lam_ch=float(LC[s].mean())))
    hetero = dict(n_target_tasks=len(recs), obs_total=float(NN.sum()), lamU_total=float(LU.sum()), lam_bin_total=float(LB.sum()), lam_ch_total=float(LC.sum()),
                  ratio_obs_U_total=float(NN.sum() / LU.sum()), ratio_obs_bin_total=float(NN.sum() / LB.sum()), sp_N_k=spearman(NN, kk), sp_lamU_k=spearman(LU, kk), sp_N_lamU=spearman(NN, LU), sp_N_lamch=spearman(NN, LC),
                  share_bridges_k_le10=float(NN[kk <= 10].sum() / NN.sum()), share_targets_k_le10=float((kk <= 10).mean()), share_bridges_k_le3=float(NN[kk <= 3].sum() / NN.sum()),
                  mean_p_alive=float(np.mean([r["p_mean_others"] for r in recs])), table=tab, table_fine=tabf, R_MC=R_MC,
                  pool=dict(C_mean=float(CC.mean()), C_var=float(CC.var()), C_fano=float(CC.var() / CC.mean()), C_max=int(CC.max()), CU_mean=float(CUm.mean()), CU_hetero_var=float(CUm.var()), CU_within_var=float(CUv.mean()), CU_total_fano=float((CUm.var() + CUv.mean()) / CUm.mean()),
                            ratio_C_CU=float(CC.sum() / CUm.sum()), pass_rate_N_over_C=float(NN.sum() / CC.sum()), sp_N_C=spearman(NN, CC), sp_C_k=spearman(CC, kk),
                            by_kbin=[dict(bin=f"{lo}-{hi}", n=int((kb == b).sum()), C_mean=float(CC[kb == b].mean()), CU_mean=float(CUm[kb == b].mean()), pass_rate=(float(NN[kb == b].sum() / CC[kb == b].sum()) if CC[kb == b].sum() > 0 else None), C_fano=(float(CC[kb == b].var() / CC[kb == b].mean()) if CC[kb == b].mean() > 0 else None)) for b, (lo, hi) in enumerate(KBINS) if (kb == b).any()]))
    P(f"[{tag}] (3) threshold pool C_i = #alive j!=i open on >=2 of T_i: mean {hetero['pool']['C_mean']:.3f} var {hetero['pool']['C_var']:.3f} Fano {hetero['pool']['C_fano']:.2f} max {hetero['pool']['C_max']} | null U: mean {hetero['pool']['CU_mean']:.3f} total Fano {hetero['pool']['CU_total_fano']:.2f} | C/CU {hetero['pool']['ratio_C_CU']:.2f} | pass rate N/C {hetero['pool']['pass_rate_N_over_C']:.3f} | Spearman(N,C) {hetero['pool']['sp_N_C']:+.3f} Spearman(C,k) {hetero['pool']['sp_C_k']:+.3f}")
    for r in hetero["pool"]["by_kbin"]: P(f"[{tag}] (3)   pool | k {r['bin']:>9s}: n {r['n']:5d} C mean {r['C_mean']:.3f} (null {r['CU_mean']:.3f}) pass rate N/C {r['pass_rate']} C Fano {r['C_fano']}")
    # decay shape: log(obs) slope vs log(chernoff) slope between consecutive non-empty coarse bins
    P(f"[{tag}] (3) N vs k: Spearman(N,k) {hetero['sp_N_k']:+.3f} | Spearman(lamU,k) {hetero['sp_lamU_k']:+.3f} | Spearman(N, lamU) {hetero['sp_N_lamU']:+.3f} | share of bridges on k<=10 targets {hetero['share_bridges_k_le10']:.3f} (targets k<=10 share {hetero['share_targets_k_le10']:.3f}) | obs/lamU total {hetero['ratio_obs_U_total']:.2f} | obs/lam_bin(no-overlap) {hetero['ratio_obs_bin_total']:.2f} | mean p (open fraction of candidates) {hetero['mean_p_alive']:.4f}")
    for r in tab:
        if r["n"]: P(f"[{tag}] (3)   k {r['bin']:>9s}: n {r['n']:5d} obs mean {r['obs_mean']:.3f} max {r['obs_max']:2d} (>=3: {r['obs_ge3']}, >=5: {r['obs_ge5']}) | lamU(MC) {r['lamU']:.4f} lam_bin {r['lam_bin']:.4f} chernoff {r['lam_ch']:.4f} | obs/lamU {r['ratio_obs_U']} | overlap a_i mean {r['a_i_mean']:.2f} | lamU Fano {r['lamU_fano']}")

    # ---------------- (4) variance decomposition ----------------
    vd = dict(N_by_unit=var_decomp(NN, ii), N_by_kbin=var_decomp(NN, kb), N_by_kbin_fine=var_decomp(NN, kbf), N_by_unit_x_kbin_fine=var_decomp(NN, ii * 100 + kbf), N_by_task=var_decomp(NN, tt),
              N_by_pool=var_decomp(NN, np.minimum(CC, 30)), N_by_pool_x_kbin_fine=var_decomp(NN, kbf * 100 + np.minimum(CC, 30)), C_by_unit=var_decomp(CC, ii), C_by_kbin_fine=var_decomp(CC, kbf), C_by_task=var_decomp(CC, tt),
              CU_mean_by_kbin_fine=var_decomp(CUm, kbf))
    # N given pool: under conditional independence of the coverage pass given (state), Var(N | C, k) <= E[N | C, k]; report residual Fano
    # null U: same decomposition on one MC replicate and on the mean (heterogeneity of lambda) + within-state var
    NU1 = np.array([lamU_mc[r["t"]][0, r["i"]] for r in recs]); vd["NU_rep0_by_unit"] = var_decomp(NU1, ii); vd["NU_rep0_by_kbin_fine"] = var_decomp(NU1, kbf)
    vd["lamU_hetero_var"] = float(LU.var()); vd["lamU_within_state_var_mean"] = float(LUv.mean()); vd["lamU_mean"] = float(LU.mean())
    vd["NU_total_fano"] = float((LU.var() + LUv.mean()) / LU.mean()) if LU.mean() > 0 else None
    # births B_{i,t+1}: targets alive at t+1
    brec = []
    for t in trans:
        st1 = ST[t + 1]; bt1 = B[t + 1]; cur = pairs[t + 1] - pairs[t]; cnt = {}
        for (i, j) in cur: cnt[i] = cnt.get(i, 0) + 1
        for i in np.where(st1["alive"])[0]:
            Nprev = int(B[t]["isb"][i].sum()) if B[t]["alive"][i] else 0
            brec.append(dict(t=t + 1, i=int(i), B=cnt.get(int(i), 0), k=int(st1["k"][i]), Nprev=Nprev, N=int(bt1["isb"][i].sum())))
    BB = np.array([r["B"] for r in brec], float); bi = np.array([r["i"] for r in brec]); bk = np.array([r["k"] for r in brec]); bkb = np.array([kbin_of(k, KBINS) for k in bk]); bkf = np.array([kbin_of(k, KBINS_FINE) for k in bk]); bNp = np.array([r["Nprev"] for r in brec]); bt_ = np.array([r["t"] for r in brec])
    vd["B_by_unit"] = var_decomp(BB, bi); vd["B_by_kbin"] = var_decomp(BB, bkb); vd["B_by_kbin_fine"] = var_decomp(BB, bkf); vd["B_by_unit_x_kbin_fine"] = var_decomp(BB, bi * 100 + bkf); vd["B_by_task"] = var_decomp(BB, bt_)
    vd["B_by_kbin_fine_x_Nprev"] = var_decomp(BB, bkf * 100 + np.minimum(bNp, 9))
    vd["B_dist"] = {str(c): int((BB == c).sum()) for c in range(0, int(BB.max()) + 1)}; vd["B_mean"] = float(BB.mean()); vd["B_var"] = float(BB.var()); vd["B_max"] = int(BB.max()); vd["n_B"] = int(len(BB))
    vd["B_given_Nprev_ge1"] = dict(n=int((bNp >= 1).sum()), mean=float(BB[bNp >= 1].mean()), var=float(BB[bNp >= 1].var()), max=int(BB[bNp >= 1].max()))
    # conditional Fano by fine k-bin (births)
    cf = []
    for b, (lo, hi) in enumerate(KBINS_FINE):
        s = bkf == b
        if s.sum() < 5: continue
        m_ = BB[s].mean(); v_ = BB[s].var(); cf.append(dict(bin=f"{lo}-{hi}", n=int(s.sum()), mean=float(m_), var=float(v_), fano=(float(v_ / m_) if m_ > 0 else None), max=int(BB[s].max()), P_ge2=float((BB[s] >= 2).mean()), P_ge2_poisson=float(1 - math.exp(-m_) * (1 + m_)), P_ge5=float((BB[s] >= 5).mean()), P_ge5_poisson=float(1 - sum(math.exp(-m_) * m_ ** x / math.factorial(x) for x in range(5)))))
    vd["B_cond_fano_by_kbin_fine"] = cf
    P(f"[{tag}] (4) N over (i,t): n {vd['N_by_unit']['n']} mean {vd['N_by_unit']['mean']:.3f} var {vd['N_by_unit']['var']:.3f} Fano {vd['N_by_unit']['fano']:.2f} | by unit: across {vd['N_by_unit']['share_across']:.3f} within {vd['N_by_unit']['share_within']:.3f} (within Fano {vd['N_by_unit']['fano_within']:.2f}) | by k-bin(fine): across {vd['N_by_kbin_fine']['share_across']:.3f} within {vd['N_by_kbin_fine']['share_within']:.3f} (within Fano {vd['N_by_kbin_fine']['fano_within']:.2f}) | by unit x k-bin: within Fano {vd['N_by_unit_x_kbin_fine']['fano_within']:.2f} | by task: across {vd['N_by_task']['share_across']:.3f}")
    P(f"[{tag}] (4) N given pool: by C: across {vd['N_by_pool']['share_across']:.3f} within Fano {vd['N_by_pool']['fano_within']:.2f} | by C x k-bin: across {vd['N_by_pool_x_kbin_fine']['share_across']:.3f} within Fano {vd['N_by_pool_x_kbin_fine']['fano_within']:.2f} | pool C itself: Fano {vd['C_by_unit']['fano']:.2f}, by unit across {vd['C_by_unit']['share_across']:.3f} within Fano {vd['C_by_unit']['fano_within']:.2f}, by k-bin within Fano {vd['C_by_kbin_fine']['fano_within']:.2f}, by task across {vd['C_by_task']['share_across']:.3f}")
    P(f"[{tag}] (4) null U: lamU mean {vd['lamU_mean']:.4f} | var of lamU across (i,t) {vd['lamU_hetero_var']:.4f} | within-state var (independent sum) {vd['lamU_within_state_var_mean']:.4f} | total Fano {vd['NU_total_fano']:.2f} | one replicate by unit: across {vd['NU_rep0_by_unit']['share_across']:.3f} within Fano {vd['NU_rep0_by_unit']['fano_within']:.2f} | by k-bin fine: within Fano {vd['NU_rep0_by_kbin_fine']['fano_within']:.2f}")
    P(f"[{tag}] (4) births B over (i,t+1): n {vd['n_B']} mean {vd['B_mean']:.3f} var {vd['B_var']:.3f} Fano {vd['B_by_unit']['fano']:.2f} max {vd['B_max']} | by unit: across {vd['B_by_unit']['share_across']:.3f} within Fano {vd['B_by_unit']['fano_within']:.2f} | by k-bin fine: across {vd['B_by_kbin_fine']['share_across']:.3f} within Fano {vd['B_by_kbin_fine']['fano_within']:.2f} | by unit x k-bin: within Fano {vd['B_by_unit_x_kbin_fine']['fano_within']:.2f} | by k-bin x Nprev: within Fano {vd['B_by_kbin_fine_x_Nprev']['fano_within']:.2f} | B dist {vd['B_dist']} | given Nprev>=1: n {vd['B_given_Nprev_ge1']['n']} mean {vd['B_given_Nprev_ge1']['mean']:.3f} max {vd['B_given_Nprev_ge1']['max']}")
    for r in cf: P(f"[{tag}] (4)   births | k {r['bin']:>8s}: n {r['n']:5d} mean {r['mean']:.3f} var {r['var']:.3f} Fano {r['fano']} max {r['max']} | P(B>=2) obs {r['P_ge2']:.4f} Poisson {r['P_ge2_poisson']:.4f} | P(B>=5) obs {r['P_ge5']:.5f} Poisson {r['P_ge5_poisson']:.2e}")

    # ---------------- (5) lifter width and lifter-side concentration ----------------
    def width_stats(plist):
        kj = np.array([ST[t]["k"][j] for (t, i, j) in plist], float); ki = np.array([ST[t]["k"][i] for (t, i, j) in plist], float)
        rj = np.array([ST[t]["r"][j] for (t, i, j) in plist]); ri = np.array([ST[t]["r"][i] for (t, i, j) in plist])
        if len(kj) == 0: return dict(n=0)
        return dict(n=int(len(kj)), frac_kj_gt_ki=float((kj > ki).mean()), frac_kj_ge_ki=float((kj >= ki).mean()), n_kj_le_ki=int((kj <= ki).sum()), ratio_median=float(np.median(kj / ki)), ratio_q10=float(np.quantile(kj / ki, .1)), kj_median=float(np.median(kj)), ki_median=float(np.median(ki)),
                    frac_rj_gt_ri=float((rj > ri).mean()), kj_mean=float(kj.mean()), ki_mean=float(ki.mean()))
    ws = dict(phase2_all=width_stats(allp), phase2_born=width_stats(born), phase2_persist=width_stats(pers))
    for t in (2, 3, 5, 9): ws[f"t{t}"] = width_stats([(t, i, j) for (i, j) in pairs[t]])
    ws["phase1_t2_9"] = width_stats([(t, i, j) for t in range(2, 10) for (i, j) in pairs[t]])
    # alive-unit width distribution for reference (phase 2): what fraction of alive pairs (i, j) have k_j > k_i?  exactly ~1/2 minus ties
    # lifter-side concentration
    conc = []
    for t in ph2:
        isb = B[t]["isb"]; deg = isb.sum(0); npairs = int(isb.sum())
        if npairs == 0: continue
        dsort = np.sort(deg)[::-1]; k = ST[t]["k"]; alive = ST[t]["alive"]
        lifters = np.where(deg > 0)[0]; kal = k[alive]; thr_top = np.sort(kal)[::-1][max(int(0.1 * len(kal)) - 1, 0)]
        DU = degU[t]; dU = DU.mean(0); dUs = np.sort(dU)[::-1]
        conc.append(dict(t=t, n_pairs=npairs, n_lifters=int(len(lifters)), top1_share=float(dsort[0] / npairs), top5_share=float(dsort[:5].sum() / npairs), sp_deg_k=spearman(deg[alive], k[alive]),
                         lifters_in_top10pct_width=float((k[lifters] >= thr_top).mean()), lifters_k_median=float(np.median(k[lifters])), alive_k_median=float(np.median(kal)),
                         null_top5_share=float(dUs[:5].sum() / max(dU.sum(), 1e-12)), null_sp_deg_k=spearman(dU[alive], k[alive]), null_n_lifters=float((DU > 0).sum(1).mean())))
    cs = {key: float(np.median([c[key] for c in conc])) for key in ("n_pairs", "n_lifters", "top1_share", "top5_share", "sp_deg_k", "lifters_in_top10pct_width", "lifters_k_median", "alive_k_median", "null_top5_share", "null_sp_deg_k", "null_n_lifters")}
    ws["lifter_concentration_median_over_states"] = cs
    P(f"[{tag}] (5) lifter width phase 2 (all pairs n {ws['phase2_all']['n']}): k_j > k_i {ws['phase2_all']['frac_kj_gt_ki']:.4f} (k_j <= k_i: {ws['phase2_all']['n_kj_le_ki']}) | k_j/k_i median {ws['phase2_all']['ratio_median']:.1f} q10 {ws['phase2_all']['ratio_q10']:.2f} | k_j median {ws['phase2_all']['kj_median']:.0f} k_i median {ws['phase2_all']['ki_median']:.0f} | r_j > r_i {ws['phase2_all']['frac_rj_gt_ri']:.3f} | born only: k_j > k_i {ws['phase2_born']['frac_kj_gt_ki']:.4f} | phase 1 (t2-9): {ws['phase1_t2_9']['frac_kj_gt_ki']:.3f} (t2 {ws['t2']['frac_kj_gt_ki']:.3f}, t3 {ws['t3']['frac_kj_gt_ki']:.3f}, t5 {ws['t5']['frac_kj_gt_ki']:.3f}, t9 {ws['t9']['frac_kj_gt_ki']:.3f})")
    P(f"[{tag}] (5) lifter-side concentration (medians over states): pairs {cs['n_pairs']:.0f} distinct lifters {cs['n_lifters']:.0f} (null U {cs['null_n_lifters']:.1f}) | top-1 lifter share {cs['top1_share']:.3f} top-5 share {cs['top5_share']:.3f} (null U top-5 {cs['null_top5_share']:.3f}) | Spearman(degree, k_j) {cs['sp_deg_k']:+.3f} (null {cs['null_sp_deg_k']:+.3f}) | lifters in top-10% width {cs['lifters_in_top10pct_width']:.3f} | lifter k median {cs['lifters_k_median']:.0f} vs alive {cs['alive_k_median']:.0f}")

    # ---------------- (6) heavy clusters (N >= 7) ----------------
    heavy = []
    for t in ph2:
        st = ST[t]; bt = B[t]; Nobs = bt["isb"].sum(1); fu, cnt, na = occ(st); fg = occ_g(st)
        for i in np.where(Nobs >= 7)[0]:
            i = int(i); js = [int(j) for j in np.where(bt["isb"][i])[0]]; Ti = [int(n) for n in bt["T"][:, i]]; Oi = [int(n) for n in np.where(st["O"][:, i])[0]]
            excl = set(js) | {i}; fex, cnt_ex, na_ex = occ(st, excl=tuple(excl))
            Tinfo = []
            for n in Ti:
                Tinfo.append(dict(n=n, in_Oi=bool(st["O"][n, i]), lifters_open=int(sum(st["O"][n, j] for j in js)), f_u=float(fu[n]), f_u_pct=pct_rank(fu, n), f_u_excl_cluster=float(fex[n]), f_g=float(fg[n]), K_pct=pct_rank(Kv, n), R_pct=pct_rank(Rn, n), hubF=bool(hubF[t][n]), hubR=bool(hubR[n]), hubK=bool(hubK[n])))
            Oinfo = [dict(n=n, lifters_open=int(sum(st["O"][n, j] for j in js)), f_u=float(fu[n]), f_u_pct=pct_rank(fu, n), f_u_excl_cluster=float(fex[n]), K_pct=pct_rank(Kv, n), R_pct=pct_rank(Rn, n)) for n in Oi]
            kal = st["k"][st["alive"]]; ksort = np.sort(kal)[::-1]; kthr = ksort[min(len(js) - 1, len(ksort) - 1)]
            Linfo = [dict(j=j, k_j=int(st["k"][j]), p_j=float(st["k"][j] / N), r_j=float(st["r"][j]), thr_open=int(st["O"][Ti, j].sum()), cov=int(bt["ov"][i, j]), degree=int(bt["isb"][:, j].sum()), in_top_width=bool(st["k"][j] >= kthr),
                          born_here=bool((i, j) not in pairs.get(t - 1, set())), survives=bool((i, j) in pairs.get(t + 1, set()))) for j in js]
            J = jacc_mat(st); jj = [J[j1, j2] for x, j1 in enumerate(js) for j2 in js[x + 1:]]
            # single-input dependence: the T_i input open on the most lifters; lifters that keep >=2 thresholds without it
            best = max(Tinfo, key=lambda d: d["lifters_open"]); nb = best["n"]
            keep_wo = int(sum(1 for j in js if st["O"][[n for n in Ti if n != nb], j].sum() >= 2))
            common_T = [n for n in Ti if all(st["O"][n, j] for j in js)]; common_O = [n for n in Oi if all(st["O"][n, j] for j in js)]
            lamU_i = float(lamU_mc[t][:, i].mean()); rec_i = next(r for r in recs if r["t"] == t and r["i"] == i)
            heavy.append(dict(t=t, i=i, N=int(Nobs[i]), k_i=int(st["k"][i]), need=int(bt["need"][i]), r_i=float(st["r"][i]), a_i=int(st["O"][Ti, i].sum()), T_i=Tinfo, O_i=Oinfo, lifters=Linfo,
                              lifters_k_median=float(np.median([l["k_j"] for l in Linfo])), alive_k_median=float(np.median(kal)), lifters_in_top_N_width=float(np.mean([l["in_top_width"] for l in Linfo])), lifters_jaccard_mean=float(np.mean(jj)) if jj else None,
                              all_alive_pairs_jaccard=float(J[np.ix_(np.where(st["alive"])[0], np.where(st["alive"])[0])][np.triu_indices(int(st["alive"].sum()), 1)].mean()),
                              most_shared_T_input=dict(n=nb, lifters_open=best["lifters_open"], f_u=best["f_u"], f_u_pct=best["f_u_pct"], f_u_excl_cluster=best["f_u_excl_cluster"], hubF=best["hubF"], hubR=best["hubR"], hubK=best["hubK"]),
                              lifters_keeping_2of3_without_it=keep_wo, common_T_inputs=common_T, common_O_inputs=common_O, lamU=lamU_i, lam_bin=rec_i["lam_bin"], lam_ch=rec_i["lam_ch"],
                              n_born_here=int(sum(l["born_here"] for l in Linfo)), n_survive=int(sum(l["survives"] for l in Linfo)), N_prev=int(B[t - 1]["isb"][i].sum()) if (t - 1) in B and B[t - 1]["alive"][i] else 0, N_next=int(B[t + 1]["isb"][i].sum()) if (t + 1) in B and B[t + 1]["alive"][i] else 0,
                              f_g_T_mean=float(np.mean([d["f_g"] for d in Tinfo])), f_u_T_mean=float(np.mean([d["f_u"] for d in Tinfo])), f_u_T_excl_mean=float(np.mean([d["f_u_excl_cluster"] for d in Tinfo]))))
    for h in heavy:
        P(f"[{tag}] (6) HEAVY t{h['t']} u{h['i']}: N {h['N']} (prev {h['N_prev']} next {h['N_next']}, born here {h['n_born_here']}, survive {h['n_survive']}) k_i {h['k_i']} need {h['need']} r_i {h['r_i']:+.2f} a_i {h['a_i']} | T_i " + "; ".join(f"n{d['n']}:{d['lifters_open']}/{h['N']} f_u {d['f_u']:.3f} (pct {d['f_u_pct']:.3f}, excl {d['f_u_excl_cluster']:.3f}) K% {d['K_pct']:.2f} R% {d['R_pct']:.2f} hubF/R/K {int(d['hubF'])}{int(d['hubR'])}{int(d['hubK'])}" for d in h["T_i"]) + f" | O_i lifters_open " + ",".join(f"n{d['n']}:{d['lifters_open']}" for d in h["O_i"][:12]) + (" ..." if len(h["O_i"]) > 12 else ""))
        P(f"[{tag}] (6)   lifters k_j {[l['k_j'] for l in h['lifters']]} (median {h['lifters_k_median']:.0f} vs alive median {h['alive_k_median']:.0f}; in top-N widest {h['lifters_in_top_N_width']:.2f}) degree {[l['degree'] for l in h['lifters']]} thr_open {[l['thr_open'] for l in h['lifters']]} | lifters' mutual Jaccard {h['lifters_jaccard_mean']:.3f} vs all alive pairs {h['all_alive_pairs_jaccard']:.3f} | most-shared T input n{h['most_shared_T_input']['n']} open on {h['most_shared_T_input']['lifters_open']}/{h['N']} (f_u {h['most_shared_T_input']['f_u']:.3f}, pct {h['most_shared_T_input']['f_u_pct']:.3f}); lifters keeping 2-of-3 without it {h['lifters_keeping_2of3_without_it']} | common T {h['common_T_inputs']} common O {h['common_O_inputs']} | lamU {h['lamU']:.3f} lam_bin {h['lam_bin']:.3f} chernoff {h['lam_ch']:.3f} | f_g(T_i) mean {h['f_g_T_mean']:.3f}")

    # ---------------- save ----------------
    Nmat = np.full((len(ph2), H), -1, int); Kmat = np.zeros((len(ph2), H), int); LUmat = np.full((len(ph2), H), np.nan); LBmat = np.full((len(ph2), H), np.nan); LCmat = np.full((len(ph2), H), np.nan); Tmat = np.zeros((len(ph2), 3, H), int); Rmat = np.zeros((len(ph2), H))
    for q, t in enumerate(ph2):
        Nmat[q, B[t]["alive"]] = B[t]["isb"].sum(1)[B[t]["alive"]]; Kmat[q] = ST[t]["k"]; Tmat[q] = B[t]["T"]; Rmat[q] = ST[t]["r"]
    for r in recs:
        q = ph2.index(r["t"]); LUmat[q, r["i"]] = r["lamU"]; LBmat[q, r["i"]] = r["lam_bin"]; LCmat[q, r["i"]] = r["lam_ch"]
    Bmat = np.full((len(trans), H), -1, int)
    for r in brec: Bmat[trans.index(r["t"] - 1), r["i"]] = r["B"]
    np.savez_compressed(os.path.join(RAW, f"m3_{tag}.npz"), tasks_ph2=np.array(ph2), N=Nmat, k=Kmat, lamU=LUmat, lam_bin=LBmat, lam_ch=LCmat, T=Tmat, r=Rmat, births=Bmat, f_u=F_u.astype(np.float32), f_g=F_g.astype(np.float32),
                        pairs_ph2=np.array(allp, int), born=np.array(born, int), died=np.array(died, int), Kv=Kv, Rn=Rn, hubR=hubR, hubK=hubK)
    out = dict(tag=tag, a=a, N=N, H=H, convention="task t = end0[t-2] (t=2..101); phase 2 = t>=10 (92 states); transitions t->t+1 for t=10..100; births land on t+1=11..101", repro=repro, occupancy=occ_summary, occupancy_per_state=occ_rows, births=births, hetero=hetero, variance=vd, width=ws, heavy=heavy, secs=time.time() - t0)
    json.dump(fl(out), open(os.path.join(RAW, f"m3_{tag}.json"), "w"), indent=1)
    P(f"[{tag}] done ({time.time() - t0:.0f}s) -> {RAW}/m3_{tag}.json / .npz / .log")
    logf.close()


def combine():
    arms = {}
    for tag in ("a01", "a03", "a05"):
        f = os.path.join(RAW, f"m3_{tag}.json")
        if os.path.exists(f): arms[tag] = json.load(open(f))
    summary = dict(task="M3 cobirth_1013: hubs, width heterogeneity, where clustering comes from", box="784-100-10 hard leaky a in {0.1,0.3,0.5}, 1200 MNIST inputs, Adam, batch 16, 4000 upd/task, iid labels, seed 0; saved task-end weights",
                   criterion="loose lifter = H4 stoch_bridges.bridges_at (P161/P176): |O_j & T_i| >= 2 and |O_j & O_i| >= (k_i+1)//2, T_i = top-3 of K_n dD/dz (smooth gate tau = 0.1 sd)",
                   convention=next(iter(arms.values()))["convention"] if arms else None, source_script="/home/issan/Projects/claude/wt/cobirth_1013/analysis/cobirth_1013/m3_hubs.py", raw_dir=RAW, arms={})
    for tag, d in arms.items():
        summary["arms"][tag] = dict(a=d["a"], repro=d["repro"], occupancy={k: v for k, v in d["occupancy"].items()}, births={k: v for k, v in d["births"].items()}, hetero=d["hetero"], variance=d["variance"], width=d["width"],
                                   heavy=[{k: v for k, v in h.items() if k not in ("O_i",)} for h in d["heavy"]], secs=d["secs"])
    json.dump(summary, open(os.path.join(RES, "m3_hubs.json"), "w"), indent=1)
    print(f"combined -> {RES}/m3_hubs.json ({len(arms)} arms)")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "combine": combine()
    else: run_arm(mode)
