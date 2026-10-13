"""TASK V (cobirth_seeds_1013, verifier, 1014): does the excess of the per-target Fano over null 1 survive a null that keeps each
candidate's slow rate changes (trend)?  null 1w = independent circular shift of each candidate's series WITHIN each of three
30-task windows (T 11-40, 41-70, 71-100; only the target's alive tasks), so each candidate keeps its per-window rate while the
same-task alignment of candidates inside a window is destroyed.  Also reports the observed per-target Fano around the per-window
mean of N_i (detrended F, ddof per window) and the trend share of Var N_i.
Input: verify/vcounts_<tag>.npz (N) and the agents' isb arrays (bit-identical to the verifier's own, checked in verify_counts.py).
usage: verify_trend_null.py [R]"""
import os, sys, json
import numpy as np
V = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013"
ISB = {"a01s0": f"{V}/../cobirth_1013/m1/counts_a01.npz", "a03s0": f"{V}/../cobirth_1013/m1/counts_a03.npz", "a05s0": f"{V}/../cobirth_1013/m1/counts_a05.npz",
       "a01s1": f"{V}/seed1/counts_a01s1.npz", "a03s1": f"{V}/seed1/counts_a03s1.npz", "a05s1": f"{V}/seed1/counts_a05s1.npz",
       "a003s0": f"{V}/slopes/counts_a0.03_s0.npz", "a003s1": f"{V}/slopes/counts_a0.03_s1.npz", "a02s0": f"{V}/slopes/counts_a0.2_s0.npz", "a02s1": f"{V}/slopes/counts_a0.2_s1.npz",
       "ELU_s0_E1": f"{V}/elu/counts_ELU_s0_E1.npz", "ELU_s0_E2": f"{V}/elu/counts_ELU_s0_E2.npz", "ELU_s1_E1": f"{V}/elu/counts_ELU_s1_E1.npz", "ELU_s1_E2": f"{V}/elu/counts_ELU_s1_E2.npz"}
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
WIN = [(11, 40), (41, 70), (71, 100)]
rng = np.random.default_rng(99)
out = {}
for tag, f in ISB.items():
    N = np.load(f"{V}/verify/vcounts_{tag}.npz")["N"]; isb = np.load(f)["isb"].astype(np.int8)
    mats = []; Fobs = []; Fdet = []; trend_share = []
    for i in range(N.shape[1]):
        rows = np.arange(10, 100); rows = rows[N[rows, i] >= 0]
        if len(rows) < 30: continue
        M = np.delete(isb[rows, i, :], i, axis=1); T = rows + 1; w = np.searchsorted([40, 70], T, side="left")   # window id 0/1/2
        v = M.sum(1).astype(float); mu = v.mean(); s2 = v.var(ddof=1)
        Fobs.append(s2 / mu if mu > 0 else np.nan)
        wm = np.array([v[w == g].mean() for g in range(3)])[w]; res = v - wm
        dof = len(v) - len(np.unique(w)); Fdet.append((res ** 2).sum() / dof / mu if mu > 0 else np.nan)
        trend_share.append(((wm - mu) ** 2).sum() / ((v - mu) ** 2).sum() if s2 > 0 else np.nan)
        mats.append((M, w))
    Fobs = np.array(Fobs); med_obs = float(np.nanmedian(Fobs)); null = np.empty(R)
    for r in range(R):
        F = np.empty(len(mats))
        for t, (M, w) in enumerate(mats):
            Mn = np.empty_like(M)
            for g in range(3):
                idx = np.where(w == g)[0]; n = len(idx)
                if n == 0: continue
                sub = M[idx]; off = rng.integers(0, n, size=sub.shape[1]); ii = (np.arange(n)[:, None] - off[None, :]) % n
                Mn[idx] = np.take_along_axis(sub, ii, axis=0)
            v = Mn.sum(1).astype(float); mu = v.mean(); F[t] = v.var(ddof=1) / mu if mu > 0 else np.nan
        null[r] = np.nanmedian(F)
    p = float(min(1.0, 2 * min((1 + (null >= med_obs).sum()) / (R + 1), (1 + (null <= med_obs).sum()) / (R + 1))))
    out[tag] = dict(median_F_obs=med_obs, null1w_q025_q975=[float(np.quantile(null, .025)), float(np.quantile(null, .975))], null1w_median=float(np.median(null)), p=p, R=R,
                    median_F_detrended=float(np.nanmedian(Fdet)), median_trend_share_of_var=float(np.nanmedian(trend_share)))
    print(f"{tag}: median F obs {med_obs:.3f} | null 1w [{out[tag]['null1w_q025_q975'][0]:.3f}, {out[tag]['null1w_q025_q975'][1]:.3f}] p {p:.3f} | detrended F median {out[tag]['median_F_detrended']:.3f} | trend share of Var N_i median {out[tag]['median_trend_share_of_var']:.3f}", flush=True)
json.dump(out, open(f"{V}/verify/verify_trend_null.json", "w"), indent=1)
