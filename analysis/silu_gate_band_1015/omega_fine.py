"""omega*(sigma) on a fine sigma grid near the readout thresholds (SiLU 0.940, GELU 0.568).

omega*(sigma) = min over the gate-2 band (r2 < r < rz) of G1/|G2|, Gaussian z ~ N(r sigma, sigma^2), K constant.
The gate part of theorem 6 is >= 0 at (r, sigma) iff 1 - Tbar <= G1/|G2| there, so the whole band is safe
iff 1 - Tbar <= omega*(sigma).  Re-uses the read-only evaluators of stage17_1014_silu_return/theory
(gm_core.mp_eval at mpmath dps 34, gm_band.band / band_depth); nothing is written there.
Also checks, on the same grid, that G1 >= 0 on r in [-12, 0) (sigma >= sigma_c of gate 1).
"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
THEORY = ("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/"
          "block_1003/stage17_1014_silu_return/theory")
sys.path.insert(0, THEORY)
import numpy as np
import mpmath as mp
from gm_core import mp_eval
from gm_band import band, band_depth

OUT = "/home/issan/Projects/claude/proj_004_drift/results/silu_gate_band_1015/omega_fine.json"
GRIDS = {"SiLU": [0.9404] + list(np.round(np.linspace(0.95, 1.40, 10), 4)),
         "GELU": [0.5680] + list(np.round(np.linspace(0.60, 1.40, 9), 4))}

res = {"meta": {"theory_dir": THEORY, "dps": int(mp.mp.dps),
                "def": "omega*(sigma)=min_{r in (r2,rz)} G1/|G2|; G1=Cov(phi'',phi'), G2=E[phi'']E[phi'] (Gaussian, K const)"}}
t0 = time.time()
for act, grid in GRIDS.items():
    rows = []
    for s in grid:
        b = band(act, float(s))
        d = band_depth(act, float(s), b["rz"], b["r2"], npts=41)
        g1min = min(float(mp_eval(act, mp.mpf(r), float(s))["G1"]) for r in np.linspace(-12, -0.05, 60))
        rows.append({"sigma": float(s), "band_r2": float(b["r2"]), "band_rz": float(b["rz"]),
                     "omega_star": d["omega_star"], "r_omega_star": d["r_omega_star"],
                     "min_G1_on_r_-12_0": g1min})
        print(f"{act} sigma {s:.4f} band ({float(b['r2']):.3f},{float(b['rz']):.3f}) omega* {d['omega_star']:.4g} at r {d['r_omega_star']:.3f}  minG1 {g1min:.3e}  [{time.time()-t0:.0f}s]", flush=True)
    res[act] = rows
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
