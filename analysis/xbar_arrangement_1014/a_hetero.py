"""xbar_arrangement_1014 / agent A, extra check of (M1) beyond the linear correlation: does the spread of u-perp depend on K?
Exchangeability of (u_kn)_n with K implies that the distribution of u_kn within every K-octile is the same as overall.
Per alive unit (T = 11, 21, ..., 91, 100 and init): corr_n(|u_kn|, K_n); sd of u_kn within each K-octile / s_perp_k;
fraction of each octile with u_kn > 0 (sign balance).  Medians over units, pooled over the sampled states.
usage: a_hetero.py  -> results/xbar_arrangement_1014/a_hetero.json"""
import os, sys, json
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_occupancy import FILES, SAMPLED, ROOT, load_X, pearson
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014/a_hetero.json"
res = {}
for tag, rel in FILES.items():
    d = np.load(os.path.join(ROOT, rel)); X = load_X(d["subset"]); Xt = np.c_[X, np.ones(len(X))]
    xbar = Xt.mean(0); K = Xt @ xbar; Kbar = float(xbar @ xbar)
    edges = np.quantile(K, np.arange(1, 8) / 8.0); oc = np.searchsorted(edges, K, side="right")
    out = {}
    for label, Ts in (("init", [0]), ("phase2_sampled", SAMPLED)):
        cabs = []; sdoct = []; posoct = []
        for T in Ts:
            if T == 0: W = d["init0"].astype(np.float64); b = d["init1"].astype(np.float64)
            else: W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64)
            Wt = np.c_[W, b]; Z = Xt @ Wt.T; m = Wt @ xbar; U = Z - np.outer(K / Kbar, m)
            alive = (Z > 0).mean(0) > 0; sp = U.std(0)
            for k in np.flatnonzero(alive & (sp > 0)):
                u = U[:, k]; cabs.append(pearson(np.abs(u), K))
                sdoct.append([u[oc == o].std() / sp[k] for o in range(8)]); posoct.append([(u[oc == o] > 0).mean() for o in range(8)])
        cabs = np.array(cabs); sdoct = np.array(sdoct); posoct = np.array(posoct)
        out[label] = dict(n_units=int(len(cabs)), corr_absu_K_q=[float(np.quantile(cabs, p)) for p in (0.1, 0.5, 0.9)],
                          sd_oct_over_sp_med=np.median(sdoct, 0).tolist(), sd_oct_over_sp_q10=np.quantile(sdoct, 0.1, 0).tolist(), sd_oct_over_sp_q90=np.quantile(sdoct, 0.9, 0).tolist(),
                          frac_pos_oct_med=np.median(posoct, 0).tolist())
    res[tag] = out
    print(tag, json.dumps(out["phase2_sampled"]["corr_absu_K_q"]), [round(x, 2) for x in out["phase2_sampled"]["sd_oct_over_sp_med"]], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
