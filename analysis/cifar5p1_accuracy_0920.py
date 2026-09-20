#!/usr/bin/env python3
"""The accuracy table for cifar5p1_mlp_0920 (spec §18).  Post-hoc, no new runs.

    python3 analysis/cifar5p1_accuracy_0920.py            # prints the markdown of §18

Three different accuracies get conflated when people say "accuracy", and in this box they
say different things, so they are printed side by side:

  total    the mean of `online_acc` over the 15 HARD tasks.  This is Kumar's "total
           average online accuracy" restricted to the scored tasks, i.e. the column of
           Lillo & Cheney's Table 2 that reads ReLU 4.76 / Rand. Smooth-Leaky 57.01.
  window   the same average over the LATE hard tasks (21-29): the spec's registered metric.
  train29  the fit on task 29's own 2500 images at the end of that task.
  test29   task 29's held-out test images (100 per class), 100-way argmax.
  test_all the same, averaged over all 15 hard tasks.

Chance is 1/100 = .01 because the head is 100-wide; a net that always names one of the
task's five classes gets .20.
"""

import json
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "results/cifar5p1_mlp_0920"
HARD = list(range(1, 31, 2))
LATE = HARD[10:]
# Lillo & Cheney v2, Table 2, the 5+1 CIFAR column (their CNN)
LC_TABLE2 = {"ReLU": 4.76, "GELU": 17.60, "CReLU": 20.56, "Tanh": 28.59, "Swish/SiLU": 35.31,
             "Rational": 40.41, "PReLU": 43.30, "eLU": 47.64, "Leaky-ReLU": 48.86,
             "SeLU": 49.07, "Smooth-Leaky": 49.87, "RReLU": 53.60, "CeLU": 54.23,
             "Rand. Smooth-Leaky": 57.01, "Deep Fourier": 72.29}


def cells(steps_hard=780):
    out = {}
    for d in sorted(SRC.iterdir()):
        if not (d / "provenance.json").exists() or "s10-19" in d.name:
            continue
        p = json.loads((d / "provenance.json").read_text())
        if (p["cond"], p["lr"], p["hidden"]) != ("std", 1e-4, 100):
            continue
        if p.get("steps_hard", 780) != steps_hard:
            continue
        name = p["arm"] + ("" if p.get("intervention", "none") == "none"
                           else "+" + p["intervention"])
        out[name] = pd.read_csv(d / "per_task.csv")
    return out


def row(t):
    h = t[t.task.isin(HARD)]
    return dict(total=h.groupby("seed").online_acc.mean().median(),
                t1=t[t.task == 1].online_acc.median(),
                window=t[t.task.isin(LATE)].groupby("seed").online_acc.mean().median(),
                train29=t[t.task == 29].train_acc.median(),
                test29=t[t.task == 29].test_acc.median(),
                test_all=h.groupby("seed").test_acc.mean().median())


def table(cs):
    r = {k: row(v) for k, v in cs.items()}
    order = sorted(r, key=lambda k: -r[k]["total"])
    lines = ["| 腕 | total | t1 | 窓(後期) | train末 | test末 | test平均 |",
             "|---|---|---|---|---|---|---|"]
    for k in order:
        q = r[k]
        lines.append(f"| `{k}` | **{q['total']:.3f}** | {q['t1']:.3f} | {q['window']:.3f} "
                     f"| {q['train29']:.3f} | {q['test29']:.3f} | {q['test_all']:.3f} |")
    return "\n".join(lines), r


def main():
    md780, r780 = table(cells(780))
    md7800, r7800 = table(cells(7800))
    print("### 780 更新（原典の予算・登録セル）\n")
    print(md780)
    print("\n### 7,800 更新（追補 5）\n")
    print(md7800)
    # the doors, CR and DF are separate failure modes; the "how much do the arms differ"
    # question is about the arms that are still doing the task at task 29
    working = [k for k in r780 if k not in ("CH", "CHB", "CHB0", "H", "C", "CR", "DF")]
    def span(col, keys):
        v = [r780[k][col] for k in keys]
        return max(v) - min(v)
    print(f"\nall arms   : total span {span('total', r780):.3f}   test_all span {span('test_all', r780):.3f}")
    print(f"working    : total span {span('total', working):.3f}   test_all span "
          f"{span('test_all', working):.3f}   (R sits at test_all {r780['R']['test_all']:.3f})")
    print(f"L&C CNN    : total span {(max(LC_TABLE2.values()) - min(LC_TABLE2.values())) / 100:.3f}")
    print(f"our ReLU total = {r780['R']['total']:.3f} vs L&C ReLU {LC_TABLE2['ReLU'] / 100:.4f}")


if __name__ == "__main__":
    main()
