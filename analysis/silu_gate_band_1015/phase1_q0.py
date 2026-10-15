"""Phase-1 sign check for SiLU/GELU in the Gaussian (K-constant) model.
Q0 = E[phi' phi] (the self-shape = switching-push expectation, 段 0) and Q1 = Cov(phi', phi) (段 1)
for z ~ N(r sigma, sigma^2).  Scan r in [-1.5, 3] and sigma in [0.3, 100]; report (a) the region where Q0 < 0,
(b) the balance r0*(sigma) = the largest root of Q0 in r (shallow stable point) for small sigma, (c) sigma above which
Q0 > 0 for all r >= -1.5.  Uses the read-only float64 evaluator of stage17 (gm_core.f64_eval)."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
THEORY = ("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/"
          "block_1003/stage17_1014_silu_return/theory")
sys.path.insert(0, THEORY)
import math
import numpy as np
from gm_core import f64_eval
def val(v): return v.mant * math.exp(v.log)
OUT = "/home/issan/Projects/claude/proj_004_drift/results/silu_gate_band_1015/phase1_q0.json"
rs = np.round(np.arange(-1.5, 3.0001, 0.05), 3)
sigs = np.geomspace(0.3, 100, 60)
res = {"meta": {"r_grid": [float(rs[0]), float(rs[-1]), 0.05], "sigma_grid": [0.3, 100, 60], "evaluator": "gm_core.f64_eval"}}
for act in ("SiLU", "GELU"):
    neg0 = []; neg1 = []; bal = []
    for s in sigs:
        q0 = np.array([val(f64_eval(act, float(r), float(s))["Q0"]) for r in rs])
        q1 = np.array([val(f64_eval(act, float(r), float(s))["Q1"]) for r in rs])
        n0 = rs[q0 < 0]; n1 = rs[q1 < 0]
        neg0.append([float(s), (float(n0.min()), float(n0.max())) if n0.size else None])
        neg1.append([float(s), (float(n1.min()), float(n1.max())) if n1.size else None])
        # largest root of Q0 (sign change from - to + going up in r)
        idx = [i for i in range(len(rs) - 1) if q0[i] < 0 <= q0[i + 1]]
        bal.append([float(s), float(rs[idx[-1]]) if idx else None])
    s_all_pos0 = min((s for s, v in neg0 if v is None), default=None)
    s_all_pos1 = min((s for s, v in neg1 if v is None), default=None)
    rmax_neg0 = max((v[1] for s, v in neg0 if v), default=None)
    rmax_neg1 = max((v[1] for s, v in neg1 if v), default=None)
    res[act] = {"Q0_negative_r_range_by_sigma": neg0, "Q1_negative_r_range_by_sigma": neg1,
                "Q0_balance_r_by_sigma": bal,
                "sigma_above_which_Q0_positive_all_r": s_all_pos0, "sigma_above_which_Q1_positive_all_r": s_all_pos1,
                "max_r_with_Q0_negative_any_sigma": rmax_neg0, "max_r_with_Q1_negative_any_sigma": rmax_neg1}
    print(f"== {act}")
    print(f"  Q0 (段 0) negative somewhere only for r <= {rmax_neg0}; positive for all r in grid once sigma >= {s_all_pos0:.3f}")
    print(f"  Q1 (段 1) negative somewhere only for r <= {rmax_neg1}; positive for all r in grid once sigma >= {s_all_pos1:.3f}")
    for s, v in neg0[:14]:
        b = [bb for ss, bb in bal if ss == s][0]
        print(f"   sigma {s:6.3f}  Q0<0 for r in {v}   balance r0* = {b}")
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
