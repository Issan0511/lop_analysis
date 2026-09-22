#!/usr/bin/env python3
"""quiet_drift_cifar_0923 pass 2 -- score D1-D4 of PREDICTIONS_pass2.md and decompose the cost.

Window per slot: probes from hit999 + 2,500 to pass 1's pre-burst end (per_seed.csv `pre_end`).
Per slot, medians over the window's probes; per bundle, medians over seeds.

The norm cost of one e-fold of loss, first order:
    d||W1||^2 / d(-ln L) = 2 ||W1|| cos(-u1, W1) * share_W1 / ((||G1|| / L) cos(G1, u1))
with share_W1 = c_W1 / c_tot, c_l = <G_l, u_l> / L.  Written to pass2_per_seed.csv and
pass2_verdict.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "quiet_drift_cifar_0923"
BUNDLES = ("I_next", "I_C", "I_rev", "I_same", "A_own", "A_C", "A_same")
LAYERS = ("W1", "b1", "W2", "b2", "W3", "b3")


def main() -> None:
    ps = pd.read_csv(OUT / "per_seed.csv")
    rows = []
    for b in BUNDLES:
        with np.load(OUT / b / "diag.npz") as z:
            d = {k: z[k] for k in z.files}
        with np.load(OUT / b / "probe.npz") as z:
            p1 = {k: z[k] for k in ("correct", "n1", "ps_uu", "step")}
        # the diag replay is the pass-1 trajectory: same correct counts at every probe
        same = bool(np.array_equal(d["correct"], p1["correct"]))
        for s in range(10):
            r0 = ps[(ps.bundle == b) & (ps.seed == s)].iloc[0]
            if r0.status != "ok":
                continue
            k0 = (int(r0.hit999) + 2500) // 100
            k1 = int(r0.pre_end) // 100
            if k1 - k0 < 1:
                continue
            sel = slice(k0, k1 + 1)
            L = d["L"][sel, s]
            c = {nm: d[f"{nm}_Gu"][sel, s] / L for nm in LAYERS}
            ctot = sum(c.values())
            G1, u1 = d["W1_G"][sel, s], d["W1_u"][sel, s]
            cosGu = d["W1_Gu"][sel, s] / (G1 * u1)
            wn = d["W1_norm"][sel, s]
            out_u = -d["W1_cosuW"][sel, s]            # cos(-u1, W1): the step's outward part
            out_G = -d["W1_cosGW"][sel, s]            # cos(-G1, W1): descent's outward part
            share = c["W1"] / ctot
            cost_pred = 2 * wn * out_u * share / ((G1 / L) * cosGu)
            row = {"bundle": b, "seed": s, "n_probes": k1 - k0 + 1, "same_traj": same,
                   "c_tot": np.median(ctot), "share_W1": np.median(share),
                   "share_W2": np.median(c["W2"] / ctot), "share_W3": np.median(c["W3"] / ctot),
                   "share_b": np.median((c["b1"] + c["b2"] + c["b3"]) / ctot),
                   "GL_W1": np.median(G1 / L), "cosGu_W1": np.median(cosGu),
                   "u_W1": np.median(u1), "Cimg_W1": np.median(d["W1_Cimg"][sel, s]),
                   "imgsumL_W1": np.median(d["W1_img_sum"][sel, s] / (1200 * L)),
                   "top10share_W1": np.median(d["W1_img_top10share"][sel, s]),
                   "out_u_W1": np.median(out_u), "out_G_W1": np.median(out_G),
                   "wnorm": np.median(wn), "cost_pred": np.median(cost_pred),
                   "k_eff": np.median(d["k_eff"][sel, s])}
            for nm in ("W2", "W3"):
                row[f"GL_{nm}"] = np.median(d[f"{nm}_G"][sel, s] / L)
                row[f"u_{nm}"] = np.median(d[f"{nm}_u"][sel, s])
                row[f"cosGu_{nm}"] = np.median(d[f"{nm}_Gu"][sel, s] / (d[f"{nm}_G"][sel, s]
                                                                         * d[f"{nm}_u"][sel, s]))
                row[f"Cimg_{nm}"] = np.median(d[f"{nm}_Cimg"][sel, s])
                row[f"norm_{nm}"] = np.median(d[f"{nm}_norm"][sel, s])
            # the empirical cost per e-fold over the same window (pass 1's n1, float64 L here)
            n1 = p1["n1"][:, s]
            row["cost_emp"] = (n1[k1] - n1[k0]) / np.log(d["L"][k0, s] / d["L"][k1, s])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pass2_per_seed.csv", index=False, float_format="%.8g")
    med = df.groupby("bundle").median(numeric_only=True).reindex(BUNDLES)
    ic, ao, ac = med.loc["I_C"], med.loc["A_own"], med.loc["A_C"]
    terms = {"share_W1": float(np.log(ic.share_W1 / ao.share_W1)),
             "Cimg_W1": float(-np.log(ic.Cimg_W1 / ao.Cimg_W1)),
             "imgsumL_W1": float(-np.log(ic.imgsumL_W1 / ao.imgsumL_W1)),
             "cosGu_W1": float(-np.log(ic.cosGu_W1 / ao.cosGu_W1))}
    lhs = float(np.log(ic.u_W1 / ao.u_W1))
    rhs = float(np.log(ic.c_tot / ao.c_tot)) + sum(terms.values())
    v = {"D1": {"Cimg_I_C": float(ic.Cimg_W1), "Cimg_A_own": float(ao.Cimg_W1),
                "outcome": bool(ic.Cimg_W1 < ao.Cimg_W1)},
         "D2": {"log_terms_of_step_ratio": terms, "log_step_ratio": lhs,
                "log_c_tot_ratio": float(np.log(ic.c_tot / ao.c_tot)), "sum_check": rhs,
                "largest": max(terms, key=lambda k: abs(terms[k])),
                "outcome": max(terms, key=lambda k: abs(terms[k])) == "Cimg_W1"},
         "D3": {"c_tot_by_bundle": {b: float(med.loc[b, "c_tot"]) for b in BUNDLES},
                "n_probes_by_bundle": {b: float(med.loc[b, "n_probes"]) for b in BUNDLES},
                "outcome": bool(all(1 / 4000 < med.loc[b, "c_tot"] < 1 / 1000 for b in BUNDLES
                                    if med.loc[b, "n_probes"] >= 20))}}
    # D4 uses pass 1's rms W1 step over the 1,000-step windows of the pre-burst window
    r1 = ps.groupby("bundle").rms_u.median()
    dI = abs(np.log(r1["A_C"] / r1["I_C"]))
    dA = abs(np.log(r1["A_C"] / r1["A_own"]))
    v["D4"] = {"rms_u_pass1": {b: float(r1[b]) for b in ("I_C", "A_C", "A_own")},
               "log_dist_to_I_C": float(dI), "log_dist_to_A_own": float(dA),
               "outcome": bool(dI < dA)}
    pr = {"D1": 0.70, "D2": 0.35, "D3": 0.70, "D4": 0.60}
    v["scores"] = {k: {"p": pr[k], "outcome": v[k]["outcome"],
                       "brier": (pr[k] - float(v[k]["outcome"])) ** 2} for k in pr}
    v["bundle_medians"] = json.loads(med.to_json())
    v["same_trajectory_as_pass1"] = bool(df.same_traj.all())
    (OUT / "pass2_verdict.json").write_text(json.dumps(v, indent=1, default=float))
    cols = ["n_probes", "c_tot", "share_W1", "share_W2", "share_W3", "share_b", "GL_W1", "cosGu_W1",
            "u_W1", "Cimg_W1", "imgsumL_W1", "top10share_W1", "out_u_W1", "out_G_W1", "wnorm",
            "cost_pred", "cost_emp", "k_eff"]
    print(med[cols].to_string(float_format=lambda x: f"{x:.4g}"))
    for k in ("D1", "D2", "D3", "D4"):
        print(k, v[k]["outcome"], {kk: vv for kk, vv in v[k].items() if kk != "outcome"})
    print("same trajectory as pass 1:", v["same_trajectory_as_pass1"])


if __name__ == "__main__":
    main()
