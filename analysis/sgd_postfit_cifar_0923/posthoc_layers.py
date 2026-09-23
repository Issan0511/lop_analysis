#!/usr/bin/env python3
"""POST HOC (0923 evening, not registered). Per-layer ||W_l||^2 increments per task, t26-50, seed median:
post-fit window (switch_step, or hit999+500 for the pure-SGD arms, to the task end), the whole task, and the first 300 updates.
    python3 analysis/sgd_postfit_cifar_0923/posthoc_layers.py [ARM ...]"""
import csv, sys, numpy as np
from pathlib import Path
R = str(Path(__file__).resolve().parents[2] / "results/sgd_postfit_cifar_0923")
arms = sys.argv[1:] or ["A", "F", "S_lo", "S_mid", "S_hi", "A_ce", "SA_iid"]
TS = range(26, 51)
for arm in arms:
    rows = list(csv.DictReader(open(f"{R}/{arm}/per_task.csv")))
    sw = {(int(r["seed"]), int(r["task"])): int(r["switch_step"]) for r in rows
          if r.get("switch_step") not in ("", None)}
    hit = {(int(r["seed"]), int(r["task"])): int(r["hit999"]) for r in rows
           if r.get("hit999") not in ("", None)}
    per = {k: [] for k in ("post1", "post2", "post3", "task1", "task2", "task3", "shock1", "ce_e")}
    for s in range(10):
        z = np.load(f"{R}/{arm}/trace/LR_std_seed{s}.npz")
        acc = {k: [] for k in per}
        for t in TS:
            m = z["task"] == t
            if not m.any():
                continue
            st = z["step"][m]
            n = [z[f"n{l}"][m] for l in (1, 2, 3)]
            prev = z["task"] == t - 1
            n_prev_end = [z[f"n{l}"][prev][-1] for l in (1, 2, 3)]
            s0 = sw.get((s, t), hit.get((s, t), -1) + 500 if (s, t) in hit else None)
            if arm.startswith("SA_"):
                s0 = hit.get((s, t), -1) + 500
            if s0 is None or s0 < 0:
                continue
            i0 = np.searchsorted(st, s0)
            if i0 >= len(st):
                continue
            for l in range(3):
                acc[f"post{l+1}"].append(n[l][-1] - n[l][i0])
                acc[f"task{l+1}"].append(n[l][-1] - n_prev_end[l])
            # shock: first 300 steps of the task
            i3 = np.searchsorted(st, 300)
            acc["shock1"].append(n[0][min(i3, len(st) - 1)] - n_prev_end[0])
        for k in per:
            if acc[k]:
                per[k].append(np.mean(acc[k]))
    med = {k: np.median(v) if v else np.nan for k, v in per.items()}
    print(f"{arm:7s} post-fit dn1 {med['post1']:9.1f} dn2 {med['post2']:8.2f} dn3 {med['post3']:7.2f} | "
          f"whole task dn1 {med['task1']:8.1f} dn2 {med['task2']:7.2f} dn3 {med['task3']:6.2f} | first300 dn1 {med['shock1']:8.1f}")
