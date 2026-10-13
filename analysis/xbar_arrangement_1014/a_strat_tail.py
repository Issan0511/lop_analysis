"""xbar_arrangement_1014 / agent A, diagnostic for A1 (not in the task list): where does F-hat miss the K-profile?
Two relaxations of the exchangeable-pattern model, every phase-2 state:
  F_oct(n)   = mean_k Phat_{k,o(n)}(c_kn): empirical tail of u_k over the inputs of n's own K-octile only (exchangeable within octile)
  F_scale(n) = mean_k Phat^eps_k(c_kn / sig_k(o(n))): eps_kn = u_kn / sig_k(o(n)), sig_k(o) = sd of u_k within octile o,
               eps exchangeable over all inputs (only the per-octile scale of u is allowed to depend on K)
Also the within-octile dispersion index of f: D_o = var_{n in o}(f_n) / mean_{n in o}[sum_k Po_kn (1 - Po_kn) / H^2]
(= 1 + O(within-octile spread of c) if units are independent and u is exchangeable within the octile).
Outputs per file: octile ratios f/F (median over 90 states), Pearson(f, F) (median), Pearson(F, K) (median).
usage: a_strat_tail.py <tag>  -> results/xbar_arrangement_1014/a_strat_<tag>.json"""
import os, sys, json, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_occupancy import FILES, PHASE2, ROOT, load_X, pearson
OUTD = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
tag = sys.argv[1]; d = np.load(os.path.join(ROOT, FILES[tag])); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]
N = len(X); xbar = Xt.mean(0); K = Xt @ xbar; Kbar = float(xbar @ xbar)
edges = np.quantile(K, np.arange(1, 8) / 8.0); oc = np.searchsorted(edges, K, side="right"); groups = [np.flatnonzero(oc == o) for o in range(8)]
rows = dict(D_oct=[], oct_f=[], oct_Foct=[], oct_Fscale=[], r_oct=[], r_scale=[], p_f_Foct=[], p_f_Fscale=[], p_Foct_K=[], p_Fscale_K=[])
for T in PHASE2:
    W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64); Wt = np.c_[W, b]
    Z = Xt @ Wt.T; m = Wt @ xbar; U = Z - np.outer(K / Kbar, m); O = Z > 0; idx = np.flatnonzero(O.mean(0) > 0)
    f = O[:, idx].mean(1); C = -np.outer(K / Kbar, m[idx])
    Po = np.empty((N, len(idx))); Ps = np.empty((N, len(idx)))
    for j, k in enumerate(idx):
        u = U[:, k]; sig = np.empty(N)
        for g in groups:
            ug = np.sort(u[g]); Po[g, j] = (len(g) - np.searchsorted(ug, C[g, j], side="right")) / len(g); sig[g] = u[g].std()
        eps = np.sort(u / sig); Ps[:, j] = (N - np.searchsorted(eps, C[:, j] / sig, side="right")) / N
    Fo = Po.mean(1); Fs = Ps.mean(1); H = len(idx); bv = (Po * (1 - Po)).sum(1) / H ** 2
    rows["D_oct"].append(np.array([f[g].var() / bv[g].mean() if bv[g].mean() > 0 else np.nan for g in groups]))
    of = np.array([f[g].mean() for g in groups]); oo = np.array([Fo[g].mean() for g in groups]); os_ = np.array([Fs[g].mean() for g in groups])
    rows["oct_f"].append(of); rows["oct_Foct"].append(oo); rows["oct_Fscale"].append(os_)
    rows["r_oct"].append(of / oo); rows["r_scale"].append(of / os_)
    rows["p_f_Foct"].append(pearson(f, Fo)); rows["p_f_Fscale"].append(pearson(f, Fs)); rows["p_Foct_K"].append(pearson(Fo, K)); rows["p_Fscale_K"].append(pearson(Fs, K))
out = dict(tag=tag)
for k, v in rows.items():
    v = np.array(v, float)
    out[k] = dict(med=np.nanmedian(v, 0).tolist(), q10=np.nanquantile(v, 0.1, axis=0).tolist(), q90=np.nanquantile(v, 0.9, axis=0).tolist())
json.dump(out, open(os.path.join(OUTD, f"a_strat_{tag}.json"), "w"), indent=1)
print(tag, "r_oct", [round(x, 2) for x in out["r_oct"]["med"]], "r_scale", [round(x, 2) for x in out["r_scale"]["med"]],
      "D_oct", [round(x, 2) for x in out["D_oct"]["med"]], "p_f_Foct", round(out["p_f_Foct"]["med"], 3), "p_f_Fscale", round(out["p_f_Fscale"]["med"], 3), flush=True)
