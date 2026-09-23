#!/usr/bin/env python3
"""追補 1 §3: ABAB cycle center / half-difference of W1, and the verdicts Q9-Q11.

    python3 analysis/sgd_postfit_cifar_0923/abab_center.py [--root results/sgd_postfit_cifar_0923]

Cycle j = tasks (2j-1, 2j) = (A, B).  M_j = (W_{2j-1} + W_{2j}) / 2, D_j = (W_{2j} - W_{2j-1}) / 2,
from the task-end W1 snapshots.  g_early / g_late: per seed, the OLS slope of ||M_j||^2 on
x_j = 2j - 0.5 over j = 2..7 / 14..25 (per task), then the median over seeds.  The same numbers
are computed for Adam's LR_abab and LR_abab_stop (altlabels_cifar_0923, archive) as references.
Writes abab_center.json and abab_center.csv (arm, seed, j, M_sq, D_sq, dM_sq, radial).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARCH = Path.home() / "Projects/obsidian-research-data/altlabels_cifar_0923/results/altlabels_cifar_0923"
EARLY, LATE = range(2, 8), range(14, 26)


def snap(arm_dir: Path, s: int, t: int, per_seed_dir: bool) -> np.ndarray:
    d = arm_dir / f"seed{s}" if per_seed_dir else arm_dir
    return np.load(d / "snap" / f"LR_std_seed{s}" / f"t{t:02d}.npz")["W1"].astype(np.float64)


def arm_stats(arm_dir: Path, seeds, per_seed_dir=False, n_cycles=25) -> dict:
    per = {}
    rows = []
    for s in seeds:
        Ms, Ds = [], []
        try:
            for j in range(1, n_cycles + 1):
                a, b = snap(arm_dir, s, 2 * j - 1, per_seed_dir), snap(arm_dir, s, 2 * j, per_seed_dir)
                Ms.append((a + b) / 2)
                Ds.append((b - a) / 2)
        except FileNotFoundError:
            continue                                    # a dead slot writes no later snapshots
        msq = np.array([(m ** 2).sum() for m in Ms])
        dsq = np.array([(d ** 2).sum() for d in Ds])
        dm = [np.nan] + [float(((Ms[j] - Ms[j - 1]) ** 2).sum()) for j in range(1, len(Ms))]
        rad = [np.nan] + [float(2 * (Ms[j - 1] * (Ms[j] - Ms[j - 1])).sum()) for j in range(1, len(Ms))]
        x = np.array([2 * j - 0.5 for j in range(1, len(Ms) + 1)])
        fit = lambda js: float(np.polyfit(x[[j - 1 for j in js]], msq[[j - 1 for j in js]], 1)[0])
        per[s] = {"g_early": fit(EARLY), "g_late": fit(LATE),
                  "D_slope": float(np.polyfit(x[4:], dsq[4:], 1)[0]), "D_mean": float(dsq[4:].mean()),
                  "M_sq_last": float(msq[-1]), "D_sq_last": float(dsq[-1]),
                  "dM_late": float(np.nanmedian(dm[13:])), "radial_late": float(np.nanmedian(rad[13:]))}
        for j in range(len(Ms)):
            rows.append({"seed": s, "j": j + 1, "M_sq": msq[j], "D_sq": dsq[j], "dM_sq": dm[j],
                         "radial": rad[j]})
    med = {k: float(np.median([p[k] for p in per.values()])) for k in next(iter(per.values()))} \
        if per else {}
    return {"per_seed": per, "median": med, "n_seeds": len(per), "rows": rows}


def n1_slope(arm_dir: Path, seeds) -> float:
    v = []
    for s in seeds:
        try:
            ns = [float((snap(arm_dir, s, t, False) ** 2).sum()) for t in range(26, 51)]
        except FileNotFoundError:
            continue
        v.append(float(np.polyfit(range(26, 51), ns, 1)[0]))
    return float(np.median(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923"))
    a = ap.parse_args()
    root = Path(a.root)
    res = {}
    arms = {"SA_abab": (root / "SA_abab", range(10), False),
            "Adam_LR_abab": (ARCH / "LR_abab", range(10), False),
            "Adam_LR_abab_stop": (ARCH / "LR_abab_stop", range(5), True)}
    allrows = []
    for name, (d, seeds, psd) in arms.items():
        if not d.exists():
            continue
        st = arm_stats(d, seeds, psd)
        res[name] = {k: st[k] for k in ("median", "n_seeds", "per_seed")}
        allrows += [{"arm": name, **q} for q in st["rows"]]
        print(name, json.dumps(st["median"]), "n", st["n_seeds"])
    if "SA_abab" in res:
        m = res["SA_abab"]["median"]
        ge, gl = m["g_early"], m["g_late"]
        q9 = ("BOUNDED" if (gl <= 0.1 * ge and gl <= 15) else
              "GROWING" if gl >= 0.5 * ge else "SLOWING")
        q10 = "BOUNDED_DIFF" if m["D_slope"] <= 0.01 * m["D_mean"] else "DIFF_GROWS"
        res["verdict"] = {"Q9": q9, "Q10": q10}
        if (root / "SA_iid").exists():
            gi = n1_slope(root / "SA_iid", range(10))
            res["SA_iid_n1_slope_t26_50"] = gi
            res["Q11_ratio_center_to_iid"] = gl / gi if gi else float("nan")
        print("verdict", res["verdict"], "SA_iid slope", res.get("SA_iid_n1_slope_t26_50"))
    (root / "abab_center.json").write_text(json.dumps(res, indent=2) + "\n")
    with open(root / "abab_center.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "j", "M_sq", "D_sq", "dM_sq", "radial"])
        w.writeheader()
        w.writerows(allrows)


if __name__ == "__main__":
    main()
