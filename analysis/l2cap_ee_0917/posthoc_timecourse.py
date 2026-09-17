#!/usr/bin/env python3
"""l2cap_ee_0917 post hoc (NOT registered): per-arm time courses behind the verdict.

    python3 analysis/l2cap_ee_0917/posthoc_timecourse.py [--src results/l2cap_ee_0917/runs]
        -> results/l2cap_ee_0917/posthoc_timecourse.md

10-seed means at task ends of: online accuracy; the second layer's mean phi', ||mu2||, the wall constant
c2 = ||mu2|| / sd(e2' a1), d2, sigma2 and mean row norm of W2; the first layer's always-on units
(p+ = 1 on all 1200 images), their share of ||mu2||^2, their mean bias and q ||mu1||; sigma1, mean
phi'_1, the Adam step RMS and the share of coordinates with sqrt(v_hat) < eps.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARMS = ("ref", "cap1", "cap2", "cap12", "cap12_bfix")
T = (1, 2, 5, 8, 10, 14, 20, 30, 51, 75, 100)
KEYS = {"online": "online accuracy", "memo": "memo accuracy (trained net)", "dt2": "L2 training derivative (mean)",
        "dt0": "L2 share of (unit, image) with training derivative exactly 0", "zb2": "L2 zbar",
        "qmu2": "L2 q2 ||mu2||", "b2": "L2 b2", "cos2": "L2 cos(w2, e2)", "g2": "L2 mean phi' (analytic exp)", "mu2": "||mu2||", "c2": "c2 = ||mu2||/sd(e2'a1)",
        "d2": "L2 d = zbar/sigma", "sig2": "L2 sigma", "W2": "L2 mean ||w||", "on_frac": "L1 always-on share of units",
        "on_share": "always-on share of ||mu2||^2", "b1_on": "always-on units: mean b1",
        "qmu_on": "always-on units: mean q||mu1||", "sig1": "L1 sigma", "g1": "L1 mean phi'",
        "step1": "L1 Adam step RMS", "eps1": "L1 share sqrt(vhat) < eps"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / "l2cap_ee_0917" / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / "l2cap_ee_0917" / "posthoc_timecourse.md"))
    args = ap.parse_args()
    src = Path(args.src)
    data = {}
    for a in ARMS:
        acc = {}
        for s in range(10):
            rows = {int(r["task"]): r for r in csv.DictReader((src / f"{a}_s{s}" / "per_task.csv").open())}
            with np.load(src / f"{a}_s{s}" / "units.npz") as zz:
                z = dict(zz)
            p = f"s{s}_"
            end = {int(t): i for i, (t, st) in enumerate(zip(z[p + "task"], z[p + "step"])) if st == 6000}
            for k in T:
                i = end[k]
                on = z[p + "pplus_l1"][i] == 1.0
                m = z[p + "a1mean_l1"][i]
                w2n = z[p + "row_norm_l2"][i]
                v = {"online": float(rows[k]["online_acc"]), "memo": float(rows[k]["memo_acc"]),
                     "dt2": z[p + "dtrain_mean_l2"][i].mean(), "dt0": z[p + "dtrain_zero_l2"][i].mean(),
                     "zb2": z[p + "zbar_l2"][i].mean(), "qmu2": (z[p + "q_l2"][i] * z[p + "mu2_norm"][i][0]).mean(),
                     "b2": z[p + "b2"][i].mean(), "cos2": (z[p + "q_l2"][i] / w2n).mean(),
                     "g2": z[p + "gate_mean_l2"][i].mean(),
                     "mu2": z[p + "mu2_norm"][i][0], "c2": z[p + "mu2_norm"][i][0] / z[p + "mu2_proj_sd"][i][0],
                     "d2": np.nanmean(z[p + "d_l2"][i]), "sig2": z[p + "sigma_l2"][i].mean(),
                     "W2": z[p + "row_norm_l2"][i].mean(), "on_frac": on.mean(),
                     "on_share": (m[on] ** 2).sum() / (m ** 2).sum(),
                     "b1_on": z[p + "b1"][i][on].mean() if on.any() else np.nan,
                     # zbar = q ||mu1|| + b exactly, so q ||mu1|| = zbar - b (no ||mu1|| needed)
                     "qmu_on": (z[p + "zbar_l1"][i][on] - z[p + "b1"][i][on]).mean() if on.any() else np.nan,
                     "sig1": z[p + "sigma_l1"][i].mean(), "g1": z[p + "gate_mean_l1"][i].mean(),
                     "step1": z[p + "step_rms_l1"][i].mean(), "eps1": z[p + "eps_frac_l1"][i].mean()}
                for kk, x in v.items():
                    acc.setdefault((k, kk), []).append(float(x))
        data[a] = {k: (float(np.nanmean(x)) if np.isfinite(x).any() else float("nan"))
                   for k, x in ((k, np.array(x)) for k, x in acc.items())}
    L = ["# l2cap_ee_0917 — 事後の時間経過（未登録）", "",
         "> `analysis/l2cap_ee_0917/posthoc_timecourse.py`。10 seed 平均・タスク終端。always-on = 第1層で 1200 枚すべてが正側のユニット"
         "（その割合は seed ごとの 100 ユニット中の割合の平均、b1・q‖µ₁‖ はそのユニットがいる seed だけの平均）。", ""]
    for kk, name in KEYS.items():
        L += [f"### {name}", "", "| 腕 | " + " | ".join(f"t{k}" for k in T) + " |", "|---|" + "---|" * len(T)]
        for a in ARMS:
            L.append(f"| {a} | " + " | ".join(f"{data[a][(k, kk)]:.3g}" for k in T) + " |")
        L.append("")
    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
