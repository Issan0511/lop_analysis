#!/usr/bin/env python3
"""POST HOC (0923 evening, not registered). Next-task hit999 vs the state at the previous task end
(||W1||, median margin, ce_st), per arm; arm A by task block and log-log slopes (t11-50, pooled seeds)."""
import csv, numpy as np
from pathlib import Path
R = str(Path(__file__).resolve().parents[2] / "results/sgd_postfit_cifar_0923")
out = {}
for arm in ["A", "F", "S_hi", "S_mid", "S_lo", "A_ce", "SA_iid"]:
    rows = list(csv.DictReader(open(f"{R}/{arm}/per_task.csv")))
    hit = {(int(r["seed"]), int(r["task"])): int(r["hit999"]) for r in rows if r["hit999"] not in ("", None)}
    X = []
    for s in range(10):
        z = np.load(f"{R}/{arm}/trace/LR_std_seed{s}.npz")
        for t in range(2, 51):
            m = z["task"] == t - 1
            if not m.any() or (s, t) not in hit or hit[(s, t)] < 0:
                continue
            X.append((t, np.sqrt(z["n1"][m][-1]), z["margin_med"][m][-1], z["ce_st"][m][-1], hit[(s, t)]))
    X = np.array(X)
    out[arm] = X
    late = X[X[:, 0] >= 26]
    print(f"{arm:6s} t26-50: |W1| med {np.median(late[:,1]):6.1f}  margin_med(prev end) {np.median(late[:,2]):6.2f}  "
          f"ce_st(prev end) {np.median(late[:,3]):.2e}  hit999 {np.median(late[:,4]):6.0f}")
A = out["A"]
print("\nArm A by task block (pooled seeds): |W1|, margin at prev end, next hit999")
for lo, hi in [(2, 6), (6, 11), (11, 21), (21, 31), (31, 41), (41, 51)]:
    b = A[(A[:, 0] >= lo) & (A[:, 0] < hi)]
    print(f"  t{lo:2d}-{hi-1:2d}: |W1| {np.median(b[:,1]):6.1f}  margin {np.median(b[:,2]):6.2f}  hit999 {np.median(b[:,4]):6.0f}")
b = A[A[:, 0] >= 11]
for j, name in [(1, "|W1|"), (2, "margin")]:
    sl = np.polyfit(np.log(b[:, j]), np.log(b[:, 4]), 1)[0]
    r = np.corrcoef(np.log(b[:, j]), np.log(b[:, 4]))[0, 1]
    print(f"  A t11-50 log-log slope of hit999 on {name}: {sl:.2f}  r={r:.2f}")
