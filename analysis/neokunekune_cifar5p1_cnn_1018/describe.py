#!/usr/bin/env python3
"""Post-hoc DESCRIPTIVE readout of neokunekune_cifar5p1_cnn_1018 (spec §6): what the gate did,
how alpha and the measured widths moved.  Not a verdict and not registered; written after the
registered verdict was computed.

    python3 analysis/neokunekune_cifar5p1_cnn_1018/describe.py --src results/neokunekune_cifar5p1_cnn_1018/main
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ("R", "SNA", "SNAfrz1", "NEO")
SITES = ("c1", "c2", "f1", "f2")
TASKS = (1, 2, 15, 29, 30)


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    a = ap.parse_args()
    site = list(csv.DictReader(open(a.src / "site_stats.csv")))
    blocks = list(csv.DictReader(open(a.src / "per_block.csv")))
    tasks = list(csv.DictReader(open(a.src / "per_task.csv")))
    alpha = list(csv.DictReader(open(a.src / "alpha_stats.csv")))
    out = {"note": "post-hoc descriptive; seed means of per-run medians over units/channels",
           "widths": {}, "alpha": {}, "gate": {}}
    for inp in ("train", "probe"):
        for s in SITES:
            for k in ("s_med", "two_alpha_s_med", "dmean_med", "r_med") + (("s_within_med", "s_between_med") if s[0] == "c" else ()):
                for arm in ARMS:
                    vals = {t: np.nanmean([f(r[k]) for r in site if r["arm"] == arm and r["site"] == s
                                           and r["input"] == inp and int(r["task"]) == t] or [np.nan])
                            for t in TASKS}
                    out["widths"][f"{inp}/{s}/{k}/{arm}"] = {t: float(v) for t, v in vals.items()}
    for s in SITES:
        for arm in ("SNA", "SNAfrz1", "NEO"):
            out["alpha"][f"{s}/{arm}"] = {t: float(np.mean([f(r["alpha_end_med"]) for r in alpha
                                                            if r["arm"] == arm and r["site"] == s
                                                            and int(r["task"]) == t])) for t in TASKS}
            out["alpha"][f"{s}/{arm}/at_hi_end_t30"] = float(np.mean([f(r["alpha_at_hi_end"]) for r in alpha
                                                                      if r["arm"] == arm and r["site"] == s
                                                                      and int(r["task"]) == 30]))
            out["alpha"][f"{s}/{arm}/at_lo_end_t30"] = float(np.mean([f(r["alpha_at_lo_end"]) for r in alpha
                                                                      if r["arm"] == arm and r["site"] == s
                                                                      and int(r["task"]) == 30]))
    for kind, sel in (("hard", lambda t: t % 2 == 1 and t > 1), ("easy", lambda t: t % 2 == 0)):
        prof = {}
        for r in blocks:
            if r["arm"] == "NEO" and sel(int(r["task"])):
                prof.setdefault(int(r["block"]), []).append(f(r["g_mean"]))
        out["gate"][f"neo_g_by_block_{kind}"] = {b: float(np.mean(v)) for b, v in sorted(prof.items())}
        out["gate"][f"neo_mult_mean_{kind}"] = float(np.mean([f(r["mult_mean"]) for r in tasks
                                                             if r["arm"] == "NEO" and sel(int(r["task"]))]))
        out["gate"][f"sna_shadow_g_mean_{kind}"] = float(np.mean([f(r["gate_mean"]) for r in tasks
                                                                 if r["arm"] == "SNA" and sel(int(r["task"]))]))
    out["gate"]["neo_first_block_below_0.5_hard"] = min(
        [b for b, v in out["gate"]["neo_g_by_block_hard"].items() if v >= 0.5] or [None])
    (a.src / "descriptive_posthoc.json").write_text(json.dumps(out, indent=2))
    g = out["gate"]
    print(f"NEO mult mean hard {g['neo_mult_mean_hard']:.3f} easy {g['neo_mult_mean_easy']:.3f}; "
          f"g block1 hard {g['neo_g_by_block_hard'][1]:.3f} block78 {g['neo_g_by_block_hard'][78]:.3f}")
    for s in SITES:
        print(s, "probe s t2/t30", {arm: (round(out["widths"][f"probe/{s}/s_med/{arm}"][2], 3),
                                          round(out["widths"][f"probe/{s}/s_med/{arm}"][30], 3)) for arm in ARMS},
              "2as t30", {arm: round(out["widths"][f"probe/{s}/two_alpha_s_med/{arm}"][30], 3) for arm in ARMS[1:]},
              "alpha t30", {arm: round(out["alpha"][f"{s}/{arm}"][30], 3) for arm in ARMS[1:]},
              "at_hi", {arm: out["alpha"][f"{s}/{arm}/at_hi_end_t30"] for arm in ARMS[1:]})


if __name__ == "__main__":
    main()
