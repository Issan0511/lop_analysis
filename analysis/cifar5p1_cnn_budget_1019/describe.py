#!/usr/bin/env python3
"""Post-hoc description of cifar5p1_cnn_budget_1019 (NOT registered; read after the verdict).

    python3 analysis/cifar5p1_cnn_budget_1019/describe.py --root results/cifar5p1_cnn_budget_1019

1. Where the new task's images land on the FC2 units: unit-median statistics on the task's own training images
   before the first update and after the last one, late hard tasks 21..29, seed mean (train_stats.csv).
2. FC2 unit distribution on the fixed probe at boundaries 2 and 30 (units/bNN.npz): quantiles of r, fractions of units
   below r = -1 / -2 / -3, fraction with p+ < 0.05, and the median weight-row norm of every layer.
3. SNAfrz1: frozen alpha and 2 alpha s on the probe at boundaries 1 and 30 (the sharp-gate quantity of sna_cnn_cause_1009).
Means are over seeds; per-seed arrays are kept in the JSON.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ("R", "ELU", "KKT1", "SNAfrz1")
BUDGETS = (780, 2340, 7800)
LATE = (21, 23, 25, 27, 29)
SITES = ("c1", "c2", "f1", "f2")
FIELDS = ("s_med", "m_med", "r_med", "dmean_med", "ppos_med", "dead_frac", "lowresp_frac", "zeroout_frac")


def read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def train_side(root, b, tasks):
    rows = read_csv(root / f"b{b}" / "train_stats.csv")
    out = {}
    for a in ARMS:
        for ph in ("start", "end"):
            q = [r for r in rows if r["arm"] == a and r["site"] == "f2" and r["phase"] == ph and int(r["task"]) in tasks]
            out[f"{a}_{ph}"] = {k: float(np.nanmean([f(r[k]) for r in q])) if q else None for k in FIELDS}
    return out


def unit_side(root, b, boundary):
    prov = json.loads((root / f"b{b}" / "provenance.json").read_text())
    slots = prov["meta"]["slots"]
    z = np.load(root / f"b{b}" / "units" / f"b{boundary:02d}.npz")
    out = {}
    for a in ARMS:
        rows = [i for i, (c, s) in enumerate(slots) if c.split("_")[0] == a]
        r = z["probe_f2_r"][rows].astype(np.float64)
        pp = z["probe_f2_ppos"][rows].astype(np.float64)
        d = {"r_q10": float(np.nanmean(np.nanquantile(r, 0.10, axis=1))),
             "r_q50": float(np.nanmean(np.nanquantile(r, 0.50, axis=1))),
             "r_q90": float(np.nanmean(np.nanquantile(r, 0.90, axis=1))),
             "frac_r_below_-1": float(np.mean(r < -1)), "frac_r_below_-2": float(np.mean(r < -2)),
             "frac_r_below_-3": float(np.mean(r < -3)), "frac_ppos_below_0.05": float(np.mean(pp < 0.05)),
             "per_seed_frac_r_below_-2": [float(x) for x in np.mean(r < -2, axis=1)]}
        for k in [k for k in z.files if k.startswith("wnorm_")]:
            d[f"{k}_median"] = float(np.mean(np.median(z[k][rows].astype(np.float64), axis=1)))
        out[a] = d
    return out


def snake_side(root, b):
    pr = read_csv(root / f"b{b}" / "probe_stats.csv")
    fa = json.loads((root / f"b{b}" / "frozen_alpha.json").read_text())["runs"]
    out = {"frozen_alpha_median": {s: float(np.mean([np.median(v[s]) for v in fa.values()])) for s in SITES} if fa else {}}
    for a in ("SNAfrz1", "KKT1"):
        for bd in (1, 30):
            for s in SITES:
                q = [f(r["two_alpha_s_med"]) for r in pr if r["arm"] == a and r["site"] == s and int(r["boundary"]) == bd]
                out[f"{a}_two_alpha_s_b{bd}_{s}"] = float(np.nanmean(q)) if q else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--tasks", default=",".join(map(str, LATE)))
    ap.add_argument("--last", type=int, default=30)
    a = ap.parse_args()
    tasks = tuple(int(x) for x in a.tasks.split(","))
    res = {"scope": "post-hoc description, not a registered comparison", "late_tasks": list(tasks), "budgets": {}}
    for b in BUDGETS:
        res["budgets"][str(b)] = {"train_side_f2": train_side(a.root, b, tasks),
                                  "units_f2_b2": unit_side(a.root, b, 2), f"units_f2_b{a.last}": unit_side(a.root, b, a.last),
                                  "snake": snake_side(a.root, b)}
    (a.root / "descriptive_posthoc.json").write_text(json.dumps(res, indent=1))
    print("1. FC2 on the task's own images (late hard tasks; start -> end)")
    for b in BUDGETS:
        t = res["budgets"][str(b)]["train_side_f2"]
        for arm in ARMS:
            s0, s1 = t[f"{arm}_start"], t[f"{arm}_end"]
            print(f"  {b:5d} {arm:8s} r {s0['r_med']:+.2f} -> {s1['r_med']:+.2f}  phi' {s0['dmean_med']:.2f} -> {s1['dmean_med']:.2f}"
                  f"  p+ {s0['ppos_med']:.2f} -> {s1['ppos_med']:.2f}  dead {s0['dead_frac']:.2f} -> {s1['dead_frac']:.2f}"
                  f"  low {s0['lowresp_frac']:.2f} -> {s1['lowresp_frac']:.2f}  s {s0['s_med']:.2f} -> {s1['s_med']:.2f}")
    print(f"2. FC2 units on the probe at boundary {a.last}")
    for b in BUDGETS:
        u = res["budgets"][str(b)][f"units_f2_b{a.last}"]
        for arm in ARMS:
            d = u[arm]
            print(f"  {b:5d} {arm:8s} r q10/50/90 {d['r_q10']:+.2f} {d['r_q50']:+.2f} {d['r_q90']:+.2f}  r<-1 {d['frac_r_below_-1']:.2f}"
                  f" r<-2 {d['frac_r_below_-2']:.2f} r<-3 {d['frac_r_below_-3']:.2f} p+<.05 {d['frac_ppos_below_0.05']:.2f}  |W| "
                  + " ".join(f"{k[6:8]} {d[f'{k}_median']:.2f}" for k in ("wnorm_c1", "wnorm_c2", "wnorm_f1", "wnorm_f2", "wnorm_f3")))
    print("3. Snake gates")
    for b in BUDGETS:
        print(f"  {b:5d}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['budgets'][str(b)]['snake'].items()
                                       if 'f2' in k or 'frozen' in k}))


if __name__ == "__main__":
    main()
