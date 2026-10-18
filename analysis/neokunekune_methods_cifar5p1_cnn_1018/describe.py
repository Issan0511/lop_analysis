#!/usr/bin/env python3
"""Post-hoc DESCRIPTIVE readout of the 12-arm main run (not a verdict, not registered).

    python3 analysis/neokunekune_methods_cifar5p1_cnn_1018/describe.py --main results/neokunekune_methods_cifar5p1_cnn_1018/main
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS12 = ("R", "SNA", "SNAfrz1", "NEO", "SN06", "SN3", "KKT1", "ELU", "LK", "RSL", "ReDo", "L2I")
SITES = ("c1", "c2", "f1", "f2")
TASKS = (2, 15, 29, 30)


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    a = ap.parse_args()
    rows, site, fresh = [], [], []
    for b in ("A", "B", "C"):
        d = a.main / b
        rows += list(csv.DictReader(open(d / "per_task.csv")))
        if (d / "site_stats.csv").exists():
            site += list(csv.DictReader(open(d / "site_stats.csv")))
        if (d / "fresh_control.csv").exists():
            fresh += list(csv.DictReader(open(d / "fresh_control.csv")))
    out = {"note": "post-hoc descriptive; seed means of per-run medians over units", "per_arm": {}}
    for arm in ARMS12:
        q = {"cfg": sorted({r["cfg"] for r in rows if r["arm"] == arm})}
        for s in SITES:
            for k in ("s_med", "dmean_med", "two_alpha_s_med", "r_med"):
                q[f"probe/{s}/{k}"] = {t: float(np.nanmean([f(r.get(k)) for r in site if r["arm"] == arm
                                                           and r["site"] == s and r["input"] == "probe"
                                                           and int(r["task"]) == t] or [np.nan])) for t in TASKS}
        q["switch_ce_t29"] = float(np.mean([f(r["switch_ce"]) for r in rows if r["arm"] == arm and int(r["task"]) == 29]))
        if arm == "ReDo":
            q["redo_per_site_total"] = {s: int(sum(f(r.get(f"n_redo_{s}") or 0) + f(r.get(f"n_redo_boundary_{s}") or 0)
                                                   for r in rows if r["arm"] == arm)) for s in SITES}
        q["fresh_gap_online"] = float(np.mean([f(r["fresh_gap_online"]) for r in fresh if r["arm"] == arm] or [np.nan]))
        out["per_arm"][arm] = q
    (a.main / "descriptive_posthoc.json").write_text(json.dumps(out, indent=2))
    for arm, q in out["per_arm"].items():
        print(arm.ljust(8), "s f2 t2->t30 %.2f->%.2f" % (q["probe/f2/s_med"][2], q["probe/f2/s_med"][30]),
              "phi' f2 t30 %.2f" % q["probe/f2/dmean_med"][30], "phi' c2 t30 %.2f" % q["probe/c2/dmean_med"][30],
              "switchCE t29 %.1f" % q["switch_ce_t29"], "fresh gap %+.3f" % q["fresh_gap_online"],
              q.get("redo_per_site_total", ""))


if __name__ == "__main__":
    main()
