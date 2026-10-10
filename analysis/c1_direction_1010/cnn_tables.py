#!/usr/bin/env python3
"""c1_direction_1010 (spec sec. 2.1): tables of the CNN split, read straight off the per-state files.

    python analysis/c1_direction_1010/cnn_tables.py              # every arm found
    python analysis/c1_direction_1010/cnn_tables.py --interim-LR # also cnn_tables/interim_LR.md

Input:  results/c1_direction_1010/cnn/{arm}_s{seed}_t{t:02d}.npz  (src/c1_direction_1010.py measure)
Output: results/c1_direction_1010/cnn_tables/
    rates_seedwise.csv     one row per (arm, layer, t, seed, quantity): value, num, den
    rates_long.csv         per (arm, layer, t, quantity): seed mean, t-based 95% CI over seeds, pooled
    rates_wide_<arm>.csv   the seed means as one row per (layer, t)
    drift_share_by_seed_t.csv   per-seed median drift share across t (P4)
    channels_all.csv       channel level (every arm): G, G0, Gdrift, Grest, Pbar, That, S1, zbar ...
    scatter_LR_channels.csv     the LR rows of channels_all.csv (scatter-ready)

SIGN CONVENTION (as cnn_drive_verify_1009): G = <u, grad L>, u = gradient of the channel mean.
G > 0 = SINKING side (the SGD expected step lowers the channel mean), G < 0 = floating side.
Layer c1 is split by its downstream weight W2 = W2^0 + dW2^drift + dW2^rest (G = G0 + Gdrift + Grest);
layer c2 by W3.  The "carrier" quantities of a layer describe that weight: for c1 the 16 c2 channels
k (c_k = -<dW2[k], mu2^>, D2bar_k = sum_{n,q'} delta2_k), for c2 the 100 fc1 units u
(c_u = -<dW3[u], mu3^>, D3bar_u = sum_n delta3_u).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "cnn_drive_verify_1009"))
from statlite import spearman, t975                       # noqa: E402  (no scipy in the venv)

IN = ROOT / "results" / "c1_direction_1010" / "cnn"
OUT = ROOT / "results" / "c1_direction_1010" / "cnn_tables"
CH = 16
SNAKE_ARMS = ("SNA", "SNAc3", "CV06FC3", "CV3FC06")
SNAKE4 = "Snake4"          # the 4 Snake arms pooled per seed: 64 channels per seed (spec sec. 3)

# quantity name -> description (column labels say the sign convention explicitly)
Q = {
    "G>0 (sinking)": "rate of G > 0 over the layer's 16 channels",
    "Gdrift<0 | Pbar>0": "rate of Gdrift < 0 (drift term on the floating side) among channels with Pbar > 0",
    "Gdrift>0 | Pbar<=0": "rate of Gdrift > 0 (drift term on the sinking side) among channels with Pbar <= 0",
    "n Pbar<=0": "number of channels with Pbar <= 0 (per seed; pooled = total)",
    "Grest>0": "rate of Grest > 0 (rest term on the sinking side)",
    "sign(G0)=sign(G)": "rate of sign agreement of the init term with the full G",
    "|Gdrift|>|Grest|": "rate",
    "|Gdrift|>|G0|": "rate",
    "drift share (median)": "median over channels of |Gdrift| / (|G0| + |Gdrift| + |Grest|)",
    "Spearman(That,G)": "per-seed Spearman over the 16 channels of the DC transmission proxy That vs G",
    # extra context (same channels)
    "Gdrift<0 (all ch)": "rate of Gdrift < 0 over all 16 channels",
    "|Grest|>|Gdrift|": "rate (complement of |Gdrift|>|Grest| up to ties)",
    "G0>0": "rate of G0 > 0",
    "Grest>0 | Pbar>0": "rate among Pbar > 0",
    "S1>0": "rate of the centred self-form S1 > 0",
    "sign(S1)=sign(G)": "rate",
    # carrier (the downstream weight of this layer's split)
    "carrier c>0": "c1 rows: c_k > 0 over the 16 c2 channels k (W2 anti-parallel drift); "
                   "c2 rows: c_u > 0 over the 100 fc1 units u (W3)",
    "carrier Dbar>0": "c1 rows: D2bar_k > 0 over the 16 c2 channels (c2's own bias push sinking); "
                      "c2 rows: D3bar_u > 0 over the 100 fc1 units",
    "carrier mu-path share signed (median)": "c1 rows: median_k <dW2[k],mu2> / (<dW2[k],mu2> + db2_k) "
                                             "(decomposition of dzbar2_k at the current mu2); c2 rows: same for fc1",
    "carrier mu-path share abs (median)": "median of |<dW,mu>| / (|<dW,mu>| + |db|)",
    "carrier dzbar<0": "rate of <dW,mu> + db < 0 (the layer below the carrier moved down by its own parameters)",
}
RATE_LIKE = {k for k in Q if k not in ("n Pbar<=0", "drift share (median)", "Spearman(That,G)",
                                       "carrier mu-path share signed (median)",
                                       "carrier mu-path share abs (median)")}


def load_states(arms=None) -> list[dict]:
    rows = []
    for fn in sorted(IN.glob("*_s*_t*.npz")):
        if ".tmp" in fn.name:
            continue
        arm = fn.stem.split("_s")[0]
        if arms and arm not in arms:
            continue
        d = np.load(fn, allow_pickle=True)
        rows.append({k: d[k] for k in d.files} | {"_arm": arm, "_seed": int(d["seed"]), "_t": int(d["task"])})
    return rows


def layer_view(d: dict, layer: str) -> dict:
    """Per-channel arrays of one layer (c1 split by W2, c2 split by W3) and its carrier arrays."""
    if layer == "c1":
        sl = slice(0, CH)
        v = {"G": d["G_c1_full"], "G0": d["G_c1_init"], "Gd": d["G_c1_drift"], "Gr": d["G_c1_rest"],
             "Pbar": d["Pbar1"], "That": d["That_c1"], "gamma": d["gamma_bar_c1"],
             "c": d["c_k"], "Dbar": d["D2bar"], "dzW": d["dz2_W"], "dzb": d["dz2_b"]}
    else:
        sl = slice(CH, 2 * CH)
        v = {"G": d["G_c2_full"], "G0": d["G_c2_init"], "Gd": d["G_c2_drift"], "Gr": d["G_c2_rest"],
             "Pbar": d["Pbar2"], "That": d["That_c2"], "gamma": d["gamma_bar_c2"],
             "c": d["c_u"], "Dbar": d["D3bar"], "dzW": d["dz3_W"], "dzb": d["dz3_b"]}
    v["S1"] = d["S1"][sl] if "S1" in d else np.full(CH, np.nan)
    v["S0"] = d["S0"][sl] if "S0" in d else np.full(CH, np.nan)
    v["zbar"] = d["zbar"][sl]
    v["zsd"] = d["zsd"][sl]
    if "posthoc_version" in d:                       # post hoc A-F (coordinator, after the LR numbers)
        L = "c1" if layer == "c1" else "c2"
        v["G_DC"], v["G_AC"] = d[f"G_{L}_DC"], d[f"G_{L}_AC"]
        v["G_init_DC"], v["G_init_AC"] = d[f"G_{L}_init_DC"], d[f"G_{L}_init_AC"]
        if layer == "c1":
            ds, ppos = d["ds_kj"], d["Pbar1"] > 0                     # ds (k, j); mask by c1 channel j
            v["ds_pairs"] = ds.reshape(-1)
            v["ds_pairs_Ppos"] = np.broadcast_to(ppos[None, :], ds.shape).reshape(-1)
            v["A_pairs"] = d["Atil"].reshape(-1)                         # A-tilde (k, j)
            v["D_pairs"] = np.broadcast_to(d["D2bar"][:, None], ds.shape).reshape(-1)
            v["ThatD"] = d["ThatD_c1"]
            v["Z"], v["lead"], v["E_ac"] = d["Z_c1"], d["lead_c1"], d["E_ac_c1"]
            v["cD_sum"], v["cD_abs"] = np.atleast_1d(d["cD2_sum"]), np.atleast_1d(d["cD2_abs"])
            v["beta_noint"], v["beta_int"] = d["beta1_noint"], d["beta1_int"]
            v["R2_int"], v["R2_noint"] = d["R2_c1_int"], d["R2_c1_noint"]
            v["selfal"], v["perp"] = d["G_c1_selfal"], d["G_c1_perp"]
        else:
            ds, ppos = d["ds3_uk"], d["Pbar2"] > 0                     # ds (u, k); mask by c2 channel k
            v["ds_pairs"] = ds.reshape(-1)
            v["ds_pairs_Ppos"] = np.broadcast_to(ppos[None, :], ds.shape).reshape(-1)
            v["cD_sum"], v["cD_abs"] = np.atleast_1d(d["cD3_sum"]), np.atleast_1d(d["cD3_abs"])
            v["beta_noint"], v["beta_int"] = d["beta2_noint"], d["beta2_int"]
            v["R2_int"], v["R2_noint"] = d["R2_c2_int"], d["R2_c2_noint"]
            v["selfal"], v["perp"] = d["G_c2_selfal"], d["G_c2_perp"]
            v["own_selfal"], v["own_perp"] = d["G_c2_own_selfal"], d["G_c2_own_perp"]
            v["G_own"], v["G_up"] = d["G_c2_own_full"], d["G_c2_up_full"]
    return {k: np.asarray(x, dtype=float) if not k.endswith("_Ppos") else np.asarray(x, dtype=bool)
            for k, x in v.items()}


def seed_quantities(v: dict) -> dict:
    """quantity -> (value, num, den) for one (arm, layer, t, seed)."""
    G, G0, Gd, Gr, Pb = v["G"], v["G0"], v["Gd"], v["Gr"], v["Pbar"]
    pos, nonpos = Pb > 0, Pb <= 0

    def rate(mask, sel=None):
        if sel is None:
            sel = np.ones_like(mask, dtype=bool)
        den = int(sel.sum())
        num = int((mask & sel).sum())
        return (num / den if den else float("nan"), num, den)

    share = np.abs(Gd) / (np.abs(G0) + np.abs(Gd) + np.abs(Gr))
    dz = v["dzW"] + v["dzb"]
    sh_signed = v["dzW"] / dz
    sh_abs = np.abs(v["dzW"]) / (np.abs(v["dzW"]) + np.abs(v["dzb"]))
    out = {
        "G>0 (sinking)": rate(G > 0),
        "Gdrift<0 | Pbar>0": rate(Gd < 0, pos),
        "Gdrift>0 | Pbar<=0": rate(Gd > 0, nonpos),
        "n Pbar<=0": (int(nonpos.sum()), int(nonpos.sum()), len(Pb)),
        "Grest>0": rate(Gr > 0),
        "sign(G0)=sign(G)": rate(np.sign(G0) == np.sign(G)),
        "|Gdrift|>|Grest|": rate(np.abs(Gd) > np.abs(Gr)),
        "|Gdrift|>|G0|": rate(np.abs(Gd) > np.abs(G0)),
        "drift share (median)": (float(np.median(share)), np.nan, len(share)),
        "Spearman(That,G)": (spearman(v["That"], G), np.nan, len(G)),
        "Gdrift<0 (all ch)": rate(Gd < 0),
        "|Grest|>|Gdrift|": rate(np.abs(Gr) > np.abs(Gd)),
        "G0>0": rate(G0 > 0),
        "Grest>0 | Pbar>0": rate(Gr > 0, pos),
        "S1>0": rate(v["S1"] > 0),
        "sign(S1)=sign(G)": rate(np.sign(v["S1"]) == np.sign(G)),
        "carrier c>0": rate(v["c"] > 0),
        "carrier Dbar>0": rate(v["Dbar"] > 0),
        "carrier mu-path share signed (median)": (float(np.median(sh_signed)), np.nan, len(sh_signed)),
        "carrier mu-path share abs (median)": (float(np.median(sh_abs)), np.nan, len(sh_abs)),
        "carrier dzbar<0": rate(dz < 0),
    }
    if "G_DC" in v:
        out.update(posthoc_quantities(v, rate))
    return out


# post hoc quantities (coordinator's A-F, computed after the registered LR numbers were read).
# All labels start with "PH".  Signs: G > 0 sinking.  For c1 the pairs are (k, j) of W2 (256 per
# seed), for c2 the pairs are (u, k) of W3 (1600 per seed).
PH_RATES = ["PH G_DC<0", "PH G_AC<0", "PH G_DC>0", "PH G_AC>0", "PH sign(G_DC)=sign(G)",
            "PH |G_DC|>|G_AC|", "PH |G_DC|>|Gdrift|", "PH ds<0 (all pairs)", "PH ds<0 | Pbar>0",
            "PH ds<0 | Pbar<=0", "PH sum c*Dbar>0 (fraction of states)", "PH Z>0", "PH |E_ac|<|lead|",
            "PH sign(Atil_kj)=sign(Dbar_k)", "PH sign(ThatD)=sign(G_DC)", "PH beta_noint>0", "PH beta_int>0",
            "PH selfal>0", "PH sign(selfal)=sign(G)", "PH own selfal>0", "PH sign(own selfal)=sign(G_own)",
            "PH G_own>0", "PH G_up>0", "PH sign(G_own)=sign(G)", "PH sign(G_up)=sign(G)",
            "PH sign(G_init_DC)=sign(G_init)"]
PH_OTHER = ["PH margin ratio sum c*Dbar / sum |c*Dbar| (median)", "PH Spearman(Atil_kj,Dbar_k)",
            "PH Spearman(ThatD,G_DC)", "PH R2_int (median)", "PH R2_noint (median)",
            "PH selfal share (median)", "PH own selfal share (median)", "PH up share (median)",
            "PH DC share of the dW part (median)"]
RATE_LIKE.update(PH_RATES)


def posthoc_quantities(v: dict, rate) -> dict:
    G, Gd = v["G"], v["Gd"]
    o = {"PH G_DC<0": rate(v["G_DC"] < 0), "PH G_AC<0": rate(v["G_AC"] < 0),
         "PH G_DC>0": rate(v["G_DC"] > 0), "PH G_AC>0": rate(v["G_AC"] > 0),
         "PH sign(G_DC)=sign(G)": rate(np.sign(v["G_DC"]) == np.sign(G)),
         "PH |G_DC|>|G_AC|": rate(np.abs(v["G_DC"]) > np.abs(v["G_AC"])),
         "PH |G_DC|>|Gdrift|": rate(np.abs(v["G_DC"]) > np.abs(Gd)),
         "PH DC share of the dW part (median)": (float(np.median(np.abs(v["G_DC"]) /
                                                             (np.abs(v["G_DC"]) + np.abs(v["G_AC"])))), np.nan, len(G)),
         "PH sign(G_init_DC)=sign(G_init)": rate(np.sign(v["G_init_DC"]) == np.sign(v["G0"])),
         "PH ds<0 (all pairs)": rate(v["ds_pairs"] < 0),
         "PH ds<0 | Pbar>0": rate(v["ds_pairs"] < 0, v["ds_pairs_Ppos"]),
         "PH ds<0 | Pbar<=0": rate(v["ds_pairs"] < 0, ~v["ds_pairs_Ppos"]),
         "PH sum c*Dbar>0 (fraction of states)": rate(v["cD_sum"] > 0),
         "PH margin ratio sum c*Dbar / sum |c*Dbar| (median)": (float(np.median(v["cD_sum"] / v["cD_abs"])),
                                                                 np.nan, len(v["cD_sum"])),
         "PH beta_noint>0": rate(v["beta_noint"] > 0), "PH beta_int>0": rate(v["beta_int"] > 0),
         "PH R2_int (median)": (float(np.median(v["R2_int"])), np.nan, len(G)),
         "PH R2_noint (median)": (float(np.median(v["R2_noint"])), np.nan, len(G)),
         "PH selfal>0": rate(v["selfal"] > 0),
         "PH sign(selfal)=sign(G)": rate(np.sign(v["selfal"]) == np.sign(G)),
         "PH selfal share (median)": (float(np.median(np.abs(v["selfal"]) / (np.abs(v["selfal"]) + np.abs(v["perp"])))),
                                      np.nan, len(G))}
    if "Z" in v:                                       # c1 only
        o["PH Z>0"] = rate(v["Z"] > 0)
        o["PH |E_ac|<|lead|"] = rate(np.abs(v["E_ac"]) < np.abs(v["lead"]))
        o["PH sign(Atil_kj)=sign(Dbar_k)"] = rate(np.sign(v["A_pairs"]) == np.sign(v["D_pairs"]))
        o["PH Spearman(Atil_kj,Dbar_k)"] = (spearman(v["A_pairs"], v["D_pairs"]), np.nan, len(v["A_pairs"]))
        o["PH sign(ThatD)=sign(G_DC)"] = rate(np.sign(v["ThatD"]) == np.sign(v["G_DC"]))
        o["PH Spearman(ThatD,G_DC)"] = (spearman(v["ThatD"], v["G_DC"]), np.nan, len(G))
    if "G_own" in v:                                   # c2 only
        o["PH own selfal>0"] = rate(v["own_selfal"] > 0)
        o["PH sign(own selfal)=sign(G_own)"] = rate(np.sign(v["own_selfal"]) == np.sign(v["G_own"]))
        o["PH own selfal share (median)"] = (float(np.median(np.abs(v["own_selfal"]) /
                                                             (np.abs(v["own_selfal"]) + np.abs(v["own_perp"])))),
                                             np.nan, len(G))
        o["PH G_own>0"] = rate(v["G_own"] > 0)
        o["PH G_up>0"] = rate(v["G_up"] > 0)
        o["PH sign(G_own)=sign(G)"] = rate(np.sign(v["G_own"]) == np.sign(G))
        o["PH sign(G_up)=sign(G)"] = rate(np.sign(v["G_up"]) == np.sign(G))
        o["PH up share (median)"] = (float(np.median(np.abs(v["G_up"]) / (np.abs(v["G_own"]) + np.abs(v["G_up"])))),
                                     np.nan, len(G))
    return o


def summarize(seedwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, layer, t, q), g in seedwise.groupby(["arm", "layer", "t", "quantity"], sort=False):
        x = g["value"].to_numpy(float)
        x = x[np.isfinite(x)]
        n = len(x)
        m = float(x.mean()) if n else float("nan")
        if n >= 2:
            half = t975(n - 1) * float(x.std(ddof=1)) / math.sqrt(n)
        else:
            half = float("nan")
        r = {"arm": arm, "layer": layer, "t": t, "quantity": q, "seed_mean": m,
             "ci_lo": m - half, "ci_hi": m + half, "n_seeds": n,
             "seed_median": float(np.median(x)) if n else float("nan"),
             "posthoc": q.startswith("PH ")}
        if q in RATE_LIKE:
            num, den = g["num"].sum(), g["den"].sum()
            r.update(pooled=(num / den if den else float("nan")), pooled_num=int(num), pooled_den=int(den))
        elif q == "n Pbar<=0":
            r.update(pooled=float(g["num"].sum()), pooled_num=int(g["num"].sum()), pooled_den=int(g["den"].sum()))
        else:
            r.update(pooled=float("nan"), pooled_num=np.nan, pooled_den=np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def channel_table(states: list[dict]) -> pd.DataFrame:
    rows = []
    for d in states:
        for layer in ("c1", "c2"):
            v = layer_view(d, layer)
            share = np.abs(v["Gd"]) / (np.abs(v["G0"]) + np.abs(v["Gd"]) + np.abs(v["Gr"]))
            ph_cols = [c for c in ("G_DC", "G_AC", "G_init_DC", "G_init_AC", "ThatD", "Z", "lead", "E_ac",
                                   "beta_noint", "beta_int", "R2_int", "R2_noint", "selfal", "perp",
                                   "own_selfal", "own_perp", "G_own", "G_up") if c in v]
            for j in range(CH):
                r = {"arm": d["_arm"], "seed": d["_seed"], "t": d["_t"], "layer": layer, "ch": j,
                     "G": v["G"][j], "G0": v["G0"][j], "Gdrift": v["Gd"][j], "Grest": v["Gr"][j],
                     "drift_share": share[j], "Pbar": v["Pbar"][j], "That": v["That"][j],
                     "S0": v["S0"][j], "S1": v["S1"][j], "zbar": v["zbar"][j], "zsd": v["zsd"][j],
                     "gamma_bar": v["gamma"][j]}
                r.update({f"PH_{c}": v[c][j] for c in ph_cols})
                if layer == "c1" and "posthoc_version" in d:
                    r["PH_Pbar1_blk"] = float(d["Pbar1_blk"][j])
                rows.append(r)
    return pd.DataFrame(rows)


PH_C1_A = ["PH G_DC<0", "PH G_AC<0", "PH sign(G_DC)=sign(G)", "PH |G_DC|>|G_AC|", "PH |G_DC|>|Gdrift|",
           "PH DC share of the dW part (median)", "PH sign(G_init_DC)=sign(G_init)"]
PH_C1_PAIRS = ["PH ds<0 (all pairs)", "PH ds<0 | Pbar>0", "PH ds<0 | Pbar<=0", "PH sign(Atil_kj)=sign(Dbar_k)",
               "PH Spearman(Atil_kj,Dbar_k)", "PH sign(ThatD)=sign(G_DC)", "PH Spearman(ThatD,G_DC)"]
PH_C1_C = ["PH Z>0", "PH |E_ac|<|lead|", "Gdrift<0 | Pbar>0", "Gdrift>0 | Pbar<=0"]
PH_D = ["PH sum c*Dbar>0 (fraction of states)", "PH margin ratio sum c*Dbar / sum |c*Dbar| (median)"]
PH_E = ["PH beta_noint>0", "PH beta_int>0", "PH R2_int (median)", "PH R2_noint (median)", "PH selfal>0",
        "PH sign(selfal)=sign(G)", "PH selfal share (median)"]
PH_E_OWN = ["PH own selfal>0", "PH sign(own selfal)=sign(G_own)", "PH own selfal share (median)"]
PH_C2_B = ["PH G_DC>0", "PH G_DC<0", "PH G_AC>0", "PH sign(G_DC)=sign(G)", "PH |G_DC|>|G_AC|",
           "PH |G_DC|>|Gdrift|", "PH DC share of the dW part (median)", "PH sign(G_init_DC)=sign(G_init)"]
PH_C2_PAIRS = ["PH ds<0 (all pairs)", "PH ds<0 | Pbar>0", "PH ds<0 | Pbar<=0"]
PH_F = ["PH G_own>0", "PH G_up>0", "PH sign(G_own)=sign(G)", "PH sign(G_up)=sign(G)", "PH up share (median)"]


def ph_check_max(states, arm) -> list[str]:
    mx = {}
    for d in states:
        if d["_arm"] != arm and not (arm == SNAKE4 and d["_arm"] in SNAKE_ARMS):
            continue
        for k, x in d.items():
            if isinstance(k, str) and k.startswith("chk_PH_"):
                mx[k[4:]] = max(mx.get(k[4:], 0.0), float(x))
    return [f"`{k}` {v:.1e}" for k, v in sorted(mx.items())]


def posthoc_md_arm(longdf, arm, states) -> list[str]:
    ts = sorted(longdf[longdf.arm == arm].t.unique())
    L = [f"### {arm}: A — c1, ΔW2 = ΔW2^DC + ΔW2^AC per (k, j) block (G − G⁰ = G_DC + G_AC); init W2⁰ split the same way\n",
         md_table(longdf, arm, "c1", PH_C1_A, ts, pooled=False),
         f"\n### {arm}: A — pairs (k, j) of W2 (Δs_kj = Σ_off ΔW2[k,j]; Ã_kj gate-weighted, D̄_k plain); "
         "ThatD_j = Σ_k Δs_kj D̄_k vs the exact G_DC_j = Σ_k Δs_kj Ã_kj\n",
         md_table(longdf, arm, "c1", PH_C1_PAIRS, ts, pooled=False),
         f"\n### {arm}: C — c1 drift in Codex's form, G_drift_j = −(P̄1_blk_j/‖µ₂‖) Z_j − E_ac_j "
         "(registered drift sign rates repeated for reference)\n",
         md_table(longdf, arm, "c1", PH_C1_C, ts, pooled=False),
         f"\n### {arm}: D — coupling margins Σ c·D̄ (c1 rows: W2 over the c2 channels k; c2 rows: W3 over the fc1 units u)\n",
         "c1 (Σ_k c_k D̄_k):\n", md_table(longdf, arm, "c1", PH_D, ts, pooled=False),
         "\nc2 (Σ_u c_u D̄3_u):\n", md_table(longdf, arm, "c2", PH_D, ts, pooled=False),
         f"\n### {arm}: E — self-aligned regression of H on P per channel (G = β Σ R·P + Σ R·H⊥, no-intercept β)\n",
         "c1 (H1_j on P1_j, R = γ K1 at the winners):\n", md_table(longdf, arm, "c1", PH_E, ts, pooled=False),
         "\nc2 (H2_k on P2_k; `selfal` = β × self_shape S0 with the full tangent of P2_k along u_k; "
         "`own selfal` = β × Σ R2own P2 with the tangent along u_k's own W2[k], b2[k] part, vs G_own):\n",
         md_table(longdf, arm, "c2", PH_E + PH_E_OWN, ts, pooled=False),
         f"\n### {arm}: B — c2, ΔW3 = ΔW3^DC + ΔW3^AC per (u, k) over the 64 positions\n",
         md_table(longdf, arm, "c2", PH_C2_B + PH_C2_PAIRS, ts, pooled=False),
         f"\n### {arm}: F — c2 own / upstream split, G_c2 = G_own (u_k's W2[k], b2[k] part) + G_up (its W1, b1 part)\n",
         md_table(longdf, arm, "c2", PH_F, ts, pooled=False),
         f"\nPost hoc identity checks, max over the {arm} states (all thresholds 1e−10, U-row checks 1e−12): "
         + ", ".join(ph_check_max(states, arm)) + "\n"]
    return L


def posthoc_md(longdf, states, arms, title) -> str:
    L = [f"# {title}\n",
         "**Post hoc** (requested after the registered LR numbers were read; not registered predictions). "
         "Numbers as computed, no judgement. **G > 0 = sinking side.** Cells: seed mean over 10 seeds "
         "[t-based 95% CI]; for (median) quantities the per-seed value is the median over the seed's channels "
         "(or states for the margin ratio). Pairs: c1 = (k, j) of W2 (256 per seed), c2 = (u, k) of W3 "
         "(1600 per seed); `Pbar>0` conditions use the registered P̄ (pooled-feature mean of the layer's channel). "
         "In C, P̄1_blk_j = the mean over the 25 taps of µ₂'s j-block (zero padding included), which makes "
         "the identity exact with ‖µ₂‖ the full 400-vector norm (stored as `Pbar1_blk`).\n"]
    for arm in arms:
        if arm in set(longdf.arm):
            L += posthoc_md_arm(longdf, arm, states)
    return "\n".join(L) + "\n"


def fmt(m, lo, hi, nd=2):
    if not np.isfinite(m):
        return "—"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def md_name(q: str) -> str:
    """Pipe-free label for markdown tables: '|X|' -> 'abs(X)', ' | ' -> ' given '."""
    import re
    q = q.replace(" | ", " given ")
    return re.sub(r"\|([A-Za-z0-9_*]+)\|", r"abs(\1)", q)


def md_table(longdf: pd.DataFrame, arm: str, layer: str, quantities, ts, pooled=True) -> str:
    sub = longdf[(longdf.arm == arm) & (longdf.layer == layer)]
    head = "| t | " + " | ".join(md_name(q) for q in quantities) + " |"
    sep = "|---|" + "---|" * len(quantities)
    lines = [head, sep]
    for t in ts:
        cells = []
        for q in quantities:
            r = sub[(sub.t == t) & (sub.quantity == q)]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            if q == "n Pbar<=0":
                cells.append(f"{int(r.pooled_num)} / {int(r.pooled_den)}")
                continue
            s = fmt(r.seed_mean, r.ci_lo, r.ci_hi)
            if pooled and q in RATE_LIKE and np.isfinite(r.pooled):
                s += f" (pooled {r.pooled:.2f}, n={int(r.pooled_den)})"
            cells.append(s)
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def interim_lr(longdf, seedwise, chdf) -> str:
    ts = sorted(longdf[longdf.arm == "LR"].t.unique())
    L = []
    L.append("# c1_direction_1010 — interim LR (leaky 0.1) CNN split\n")
    L.append("Numbers only, as computed (no judgement against the predictions). "
             "**Sign convention: G > 0 = sinking side** (SGD expected step lowers the channel mean). "
             "Cells: seed mean over 10 seeds [t-based 95% CI] (pooled rate over all channels, n). "
             "c1 is split by W2 = W2⁰ + ΔW2^drift (µ₂ direction) + ΔW2^rest; c2 by W3 the same way (µ₃). "
             "Source: `results/c1_direction_1010/cnn/LR_s*_t*.npz`, script `analysis/c1_direction_1010/cnn_tables.py`.\n")
    L.append("## P1 (LR c1; registered t ∈ {5, 10, 20})\n")
    L.append(md_table(longdf, "LR", "c1", ["Gdrift<0 | Pbar>0", "sign(G0)=sign(G)", "|Gdrift|>|Grest|",
                                           "Grest>0", "G>0 (sinking)"], ts))
    L.append("\n## P2 (LR c2; registered t ∈ {5, 10, 20})\n")
    L.append(md_table(longdf, "LR", "c2", ["Grest>0", "|Grest|>|Gdrift|", "Gdrift<0 (all ch)",
                                           "Gdrift<0 | Pbar>0", "G>0 (sinking)"], ts))
    L.append("\n## P3 (anti-parallel drift and downstream bias push; registered t ≥ 5)\n")
    L.append("c1 rows = the W2 carrier over the 16 c2 channels k (c_k = −⟨ΔW2[k], µ̂₂⟩, D̄_k = Σ δ2_k); "
             "c2 rows = the W3 carrier over the 100 fc1 units u (c_u, D̄3_u).\n")
    L.append("W2 / c2 channels k:\n")
    L.append(md_table(longdf, "LR", "c1", ["carrier c>0", "carrier Dbar>0", "carrier dzbar<0",
                                           "carrier mu-path share abs (median)",
                                           "carrier mu-path share signed (median)"], ts))
    L.append("\nW3 / fc1 units u:\n")
    L.append(md_table(longdf, "LR", "c2", ["carrier c>0", "carrier Dbar>0", "carrier dzbar<0",
                                           "carrier mu-path share abs (median)",
                                           "carrier mu-path share signed (median)"], ts))
    L.append("\n## P4 (c1 drift share |G^drift|/(|G⁰|+|G^drift|+|G^rest|), per-seed median over the 16 channels)\n")
    sw = seedwise[(seedwise.arm == "LR") & (seedwise.layer == "c1") & (seedwise.quantity == "drift share (median)")]
    piv = sw.pivot(index="seed", columns="t", values="value").sort_index()
    L.append("| seed | " + " | ".join(f"t{t}" for t in piv.columns) + " | t20 > t1 | monotone t1→t20 |")
    L.append("|---|" + "---|" * (len(piv.columns) + 2))
    cnt = 0
    mono = 0
    for s, r in piv.iterrows():
        up = (r.get(20, np.nan) > r.get(1, np.nan))
        seq = [r.get(t, np.nan) for t in (1, 2, 5, 10, 20)]
        mon = all(b > a for a, b in zip(seq, seq[1:]))
        cnt += int(up); mono += int(mon)
        L.append(f"| {s} | " + " | ".join(f"{x:.3f}" for x in r.to_numpy()) + f" | {'yes' if up else 'no'} | "
                 f"{'yes' if mon else 'no'} |")
    L.append(f"| mean | " + " | ".join(f"{x:.3f}" for x in piv.mean().to_numpy()) + f" | {cnt}/{len(piv)} | {mono}/{len(piv)} |")
    L.append("\n## P5 (LR c1 at t30: channels with P̄1_j ≤ 0)\n")
    L.append(md_table(longdf, "LR", "c1", ["n Pbar<=0", "Gdrift>0 | Pbar<=0"], ts))
    L.append("\n## All quantities, LR c1 (split by W2)\n")
    allq = [q for q in Q if not q.startswith("carrier")]
    L.append(md_table(longdf, "LR", "c1", allq, ts, pooled=False))
    L.append("\n## All quantities, LR c2 (split by W3)\n")
    L.append(md_table(longdf, "LR", "c2", allq, ts, pooled=False))
    L.append("\n## Channel-level medians (LR), for scale\n")
    sub = chdf[chdf.arm == "LR"]
    med = sub.groupby(["layer", "t"])[["G", "G0", "Gdrift", "Grest", "Pbar", "That"]].median()
    L.append("| layer | t | median G | median G0 | median Gdrift | median Grest | median Pbar | median That |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (layer, t), r in med.iterrows():
        L.append(f"| {layer} | {t} | {r.G:.4g} | {r.G0:.4g} | {r.Gdrift:.4g} | {r.Grest:.4g} | {r.Pbar:.4g} | {r.That:.4g} |")
    return "\n".join(L) + "\n"


C1_COLS = ["G>0 (sinking)", "Gdrift<0 | Pbar>0", "sign(G0)=sign(G)", "|Gdrift|>|Grest|", "|Gdrift|>|G0|",
           "Grest>0", "drift share (median)", "Spearman(That,G)", "n Pbar<=0", "Gdrift>0 | Pbar<=0"]
C2_COLS = ["G>0 (sinking)", "Grest>0", "|Grest|>|Gdrift|", "Gdrift<0 (all ch)", "Gdrift<0 | Pbar>0",
           "sign(G0)=sign(G)", "|Gdrift|>|G0|", "drift share (median)", "Spearman(That,G)"]
CARRIER_COLS = ["carrier c>0", "carrier Dbar>0", "carrier dzbar<0", "carrier mu-path share abs (median)",
                "carrier mu-path share signed (median)"]


def final_md(longdf: pd.DataFrame) -> str:
    order = ["LR", "LRc"] + list(SNAKE_ARMS) + [SNAKE4]
    arms = [a for a in order if a in set(longdf.arm)] + sorted(set(longdf.arm) - set(order))
    L = ["# c1_direction_1010 — CNN split, all arms\n",
         "Numbers as computed, no judgement. **G > 0 = sinking side.** Cells: seed mean over the seeds "
         "[t-based 95% CI]; `n Pbar<=0` is the pooled count / total channels. c1 is split by W2 "
         "(G = G⁰ + G^drift + G^rest), c2 by W3. Snake4 = the 4 Snake arms pooled per seed (64 channels "
         "per seed; carrier 64 c2 channels / 400 fc1 units). Definitions: `quantities_legend.csv`.\n"]
    for arm in arms:
        ts = sorted(longdf[longdf.arm == arm].t.unique())
        L.append(f"## {arm}\n")
        L.append("c1 (split by W2):\n")
        L.append(md_table(longdf, arm, "c1", C1_COLS, ts, pooled=False))
        L.append("\nc2 (split by W3):\n")
        L.append(md_table(longdf, arm, "c2", C2_COLS, ts, pooled=False))
        L.append("\ncarrier of c1 = W2 over the c2 channels k:\n")
        L.append(md_table(longdf, arm, "c1", CARRIER_COLS, ts, pooled=False))
        L.append("\ncarrier of c2 = W3 over the fc1 units u:\n")
        L.append(md_table(longdf, arm, "c2", CARRIER_COLS, ts, pooled=False))
        L.append("")
        if any(longdf[(longdf.arm == arm)].quantity.str.startswith("PH ")):
            L.append(f"### {arm}: post hoc (A–F)\n")
            L += posthoc_md_arm(longdf, arm, STATES_CACHE)
            L.append("")
    return "\n".join(L) + "\n"


STATES_CACHE: list = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--interim-LR", action="store_true")
    ap.add_argument("--final", action="store_true", help="also cnn_tables/final_tables.md (every arm)")
    ap.add_argument("--posthoc-LR", action="store_true", help="also cnn_tables/posthoc_LR.md")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    states = load_states(arms or None)
    if not states:
        raise SystemExit(f"no states in {IN}")
    STATES_CACHE[:] = states
    OUT.mkdir(parents=True, exist_ok=True)
    srows = []
    for d in states:
        for layer in ("c1", "c2"):
            for q, (val, num, den) in seed_quantities(layer_view(d, layer)).items():
                srows.append({"arm": d["_arm"], "layer": layer, "t": d["_t"], "seed": d["_seed"],
                              "quantity": q, "value": val, "num": num, "den": den})
    # Snake4: the four Snake arms pooled per (seed, t) -- 64 channels per seed (carrier: 64 c2 ch / 400 fc1)
    by = {}
    for d in states:
        if d["_arm"] in SNAKE_ARMS:
            by.setdefault((d["_seed"], d["_t"]), {})[d["_arm"]] = d
    for (seed, t), dd in sorted(by.items()):
        if len(dd) != len(SNAKE_ARMS):
            continue
        for layer in ("c1", "c2"):
            views = [layer_view(dd[a], layer) for a in SNAKE_ARMS]
            common = set.intersection(*(set(w) for w in views))
            v = {k: np.concatenate([w[k] for w in views]) for k in views[0] if k in common}
            for q, (val, num, den) in seed_quantities(v).items():
                srows.append({"arm": SNAKE4, "layer": layer, "t": t, "seed": seed,
                              "quantity": q, "value": val, "num": num, "den": den})
    seedwise = pd.DataFrame(srows).sort_values(["arm", "layer", "t", "seed"], kind="stable")
    seedwise.to_csv(OUT / "rates_seedwise.csv", index=False)
    longdf = summarize(seedwise)
    longdf.to_csv(OUT / "rates_long.csv", index=False)
    for arm in sorted(longdf.arm.unique()):
        w = longdf[longdf.arm == arm].pivot_table(index=["layer", "t"], columns="quantity", values="seed_mean",
                                                  sort=False)
        w = w[[q for q in list(Q) + PH_RATES + PH_OTHER if q in w.columns]]
        w.to_csv(OUT / f"rates_wide_{arm}.csv")
    ds = seedwise[seedwise.quantity == "drift share (median)"].pivot_table(
        index=["arm", "layer", "seed"], columns="t", values="value")
    ds.columns = [f"t{c}" for c in ds.columns]
    ds.to_csv(OUT / "drift_share_by_seed_t.csv")
    chdf = channel_table(states)
    chdf.to_csv(OUT / "channels_all.csv", index=False)
    chdf[chdf.arm == "LR"].to_csv(OUT / "scatter_LR_channels.csv", index=False)
    pd.DataFrame([{"quantity": k, "definition": v} for k, v in Q.items()]).to_csv(OUT / "quantities_legend.csv",
                                                                                  index=False)
    if args.interim_LR:
        (OUT / "interim_LR.md").write_text(interim_lr(longdf, seedwise, chdf))
    if args.final:
        (OUT / "final_tables.md").write_text(final_md(longdf))
    if args.posthoc_LR:
        (OUT / "posthoc_LR.md").write_text(posthoc_md(longdf, states, ["LR"],
                                                      "c1_direction_1010 — post hoc A–F, LR (leaky 0.1)"))
    n = {a: sum(1 for d in states if d["_arm"] == a) for a in sorted({d["_arm"] for d in states})}
    print("states per arm:", n)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
