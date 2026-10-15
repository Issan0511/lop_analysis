"""Lower bound of the phase-1 self-shape push for SiLU/GELU (Gaussian, K constant).
Q0 = E[phi'(z) phi(z)], z ~ N(r sigma, sigma^2).  The expected switching push on the position is proportional to Q0,
so the change of the relative position r per switch is proportional to Q0/sigma.  Report min_{r>=0} Q0/sigma as a
function of sigma, and the overall minimum over sigma >= 1 (and sigma >= 0.5), with leaky a=0.1 for comparison."""
import os, sys, json, math
os.environ["OMP_NUM_THREADS"] = "1"
THEORY = ("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/"
          "block_1003/stage17_1014_silu_return/theory")
sys.path.insert(0, THEORY)
import numpy as np
from gm_core import f64_eval
def val(v): return v.mant * math.exp(v.log)
OUT = "/home/issan/Projects/claude/proj_004_drift/results/silu_gate_band_1015/phase1_lower_bound.json"
rs = np.round(np.arange(0.0, 3.0001, 0.05), 3)
sigs = np.concatenate([np.geomspace(0.3, 1.0, 13)[:-1], np.geomspace(1.0, 100, 40)])
# leaky a=0.1 reference: Q0/sigma = E[phi' phi]/sigma with phi=max(z,a z): closed form via Gaussian moments
from math import erf, sqrt, exp, pi
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2))); phi = lambda x: exp(-x * x / 2) / sqrt(2 * pi)
def leaky_q0_over_sigma(r, a=0.1):
    # E[z 1(z>0)] = sigma(r Phi(r) + phi(r)), E[z 1(z<0)] = sigma(r Phi(-r) - phi(r)); phi'phi = z (open), a^2 z (closed)
    return (r * Phi(r) + phi(r)) + a * a * (r * Phi(-r) - phi(r))
res = {"meta": {"r_grid": [0.0, 3.0, 0.05], "sigma_grid": [float(sigs[0]), float(sigs[-1]), len(sigs)]}}
for act in ("SiLU", "GELU"):
    rows = []
    for s in sigs:
        q = np.array([val(f64_eval(act, float(r), float(s))["Q0"]) / s for r in rs])
        i = int(q.argmin()); rows.append({"sigma": float(s), "min_Q0_over_sigma": float(q[i]), "argmin_r": float(rs[i]), "Q0_over_sigma_at_r0": float(q[0])})
    m05 = min((x for x in rows if x["sigma"] >= 0.5), key=lambda x: x["min_Q0_over_sigma"])
    m1 = min((x for x in rows if x["sigma"] >= 1.0 - 1e-12), key=lambda x: x["min_Q0_over_sigma"])
    res[act] = {"rows": rows, "min_over_sigma_ge_1": m1, "min_over_sigma_ge_0.5": m05}
    print(f"== {act}: min_(r>=0, sigma>=1) Q0/sigma = {m1['min_Q0_over_sigma']:.4f} at sigma {m1['sigma']:.3f}, r {m1['argmin_r']};  sigma>=0.5: {m05['min_Q0_over_sigma']:.4f} at sigma {m05['sigma']:.3f}")
    for x in rows:
        if x["sigma"] in (float(sigs[0]),) or abs(x["sigma"]-0.5)<0.03 or abs(x["sigma"]-1.0)<1e-9 or abs(x["sigma"]-2)<0.15 or abs(x["sigma"]-10)<0.6 or abs(x["sigma"]-100)<1e-6:
            print(f"   sigma {x['sigma']:7.3f}  min_r Q0/sigma {x['min_Q0_over_sigma']:.4f} (at r {x['argmin_r']})   at r=0: {x['Q0_over_sigma_at_r0']:.4f}")
res["leaky_a0.1_reference"] = {"Q0_over_sigma_at_r0": leaky_q0_over_sigma(0.0), "min_r_ge_0": min(leaky_q0_over_sigma(r) for r in rs)}
print("leaky a=0.1: Q0/sigma at r=0 =", round(leaky_q0_over_sigma(0.0), 4), " min over r>=0 =", round(res["leaky_a0.1_reference"]["min_r_ge_0"], 4))
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
