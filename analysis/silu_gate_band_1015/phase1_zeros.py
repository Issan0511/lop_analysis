"""Zeros of the self-shape Q0 = E[phi' phi] (段 0, Gaussian K-constant model, z ~ N(r sigma, sigma^2)) by bisection.
Fixes the 0.05-grid artifact of phase1_q0.py (which reported the last grid point with Q0 < 0 as the balance, e.g. -0.5
instead of the true root -0.464 for SiLU at sigma 0.3).  For each sigma: all sign changes of Q0 on r in [-8, 3]
(coarse scan step 0.01, then bisection to 1e-9), labelled upper (stable, r0*) / lower (unstable).  Also reports
sup over sigma >= 0.3 of the upper root (the r above which Q0 > 0 for every sigma >= 0.3) and the sigma above which
no root remains.  Same float64 evaluator as phase1_q0.py (stage17 gm_core.f64_eval, read-only)."""
import os, sys, json, math
os.environ["OMP_NUM_THREADS"] = "1"
THEORY = ("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/"
          "block_1003/stage17_1014_silu_return/theory")
sys.path.insert(0, THEORY)
import numpy as np
from gm_core import f64_eval
def q0(act, r, s): v = f64_eval(act, float(r), float(s))["Q0"]; return v.mant * math.exp(v.log)
def roots(act, s, lo=-8.0, hi=3.0, step=0.01):
    rs = np.arange(lo, hi + 1e-9, step); vals = np.array([q0(act, r, s) for r in rs]); out = []
    for i in range(len(rs) - 1):
        if vals[i] == 0: out.append((float(rs[i]), "exact")); continue
        if vals[i] * vals[i + 1] < 0:
            a, b, fa = rs[i], rs[i + 1], vals[i]
            for _ in range(60):
                m = 0.5 * (a + b); fm = q0(act, m, s)
                if fa * fm <= 0: b = m
                else: a, fa = m, fm
                if b - a < 1e-9: break
            out.append((0.5 * (a + b), "up" if vals[i] < 0 else "down"))   # "up": Q0 goes - -> + (stable upper root)
    return out
OUT = "/home/issan/Projects/claude/proj_004_drift/results/silu_gate_band_1015/phase1_zeros.json"
sigs = sorted(set([0.3, 0.33, 0.35, 0.4, 0.44, 0.45, 0.49, 0.5, 0.55, 0.6, 0.66, 0.7, 0.72, 0.73, 0.75, 0.8, 1.0, 1.35, 2.0, 5.0]))
res = {"meta": {"r_scan": [-8.0, 3.0, 0.01], "bisection_tol": 1e-9, "evaluator": "gm_core.f64_eval", "sigmas": sigs}}
for act in ("SiLU", "GELU"):
    rows = []
    for s in sigs:
        rt = roots(act, s)
        upper = max([r for r, k in rt if k == "up"], default=None)
        lower = min([r for r, k in rt if k == "down"], default=None)
        rows.append({"sigma": s, "roots": [[round(r, 6), k] for r, k in rt], "upper_root_r0star": upper, "lower_root": lower})
    ups = [row["upper_root_r0star"] for row in rows if row["upper_root_r0star"] is not None]
    res[act] = {"by_sigma": rows,
                "sup_upper_root_over_sigma_ge_0.3": max(ups) if ups else None,
                "smallest_sigma_with_no_root": min([row["sigma"] for row in rows if not row["roots"]], default=None)}
    print(f"== {act}: sup of upper root over sigma>=0.3 = {res[act]['sup_upper_root_over_sigma_ge_0.3']}; no root from sigma {res[act]['smallest_sigma_with_no_root']}")
    for row in rows:
        if row["sigma"] in (0.3, 0.4, 0.45, 0.49, 0.5, 0.66, 0.7, 0.73, 1.0, 2.0, 5.0):
            print(f"   sigma {row['sigma']:5.2f}  roots {row['roots']}")
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
