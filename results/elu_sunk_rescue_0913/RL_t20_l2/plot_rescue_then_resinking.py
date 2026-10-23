#!/usr/bin/env python3
"""Descriptive visualization of registered primary-job measurements."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BRANCHES = ["A", "B20", "C20", "D20"]
STYLE = {
    "A": ("#3273a8", "-", "A: deep, train"),
    "B20": ("#3273a8", "--", "B20: deep, incident frozen"),
    "C20": ("#d66535", "-", "C20: once-lift, train"),
    "D20": ("#d66535", "--", "D20: once-lift, incident frozen"),
}

def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def main():
    performance = read_csv(HERE / "rows.csv")
    recovery = read_csv(HERE / "primary_unit_recovery.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.78, wspace=.24)

    ax = axes[0]
    tasks = np.arange(21, 26)
    for branch in BRANCHES:
        color, ls, label = STYLE[branch]
        curves = []
        for seed in range(3):
            rr = sorted((r for r in performance if int(r["seed"]) == seed and
                         r["branch"] == branch and int(r["step"]) == 6000),
                        key=lambda r: int(r["task"]))
            assert [int(r["task"]) for r in rr] == list(tasks)
            curves.append([100 * float(r["acc"]) for r in rr])
        curves = np.asarray(curves)
        for curve in curves:
            ax.plot(tasks, curve, color=color, ls=ls, lw=.8, alpha=.20)
        ax.plot(tasks, curves.mean(0), color=color, ls=ls, lw=2.4, marker="o",
                ms=4, label=label)
    ax.set(title="Fresh-task endpoint learning", xlabel="Future task",
           ylabel="Train accuracy at update 6000 (%)", xticks=tasks)
    ax.grid(alpha=.18)

    ax = axes[1]
    points = []
    for task in tasks:
        for step in (0, 75, 375, 1500, 3000, 6000):
            points.append((int(task), step, (int(task) - 21) * 6000 + step))
    for branch in BRANCHES:
        color, ls, label = STYLE[branch]
        curves = []
        for seed in range(3):
            curve = []
            for task, step, _ in points:
                rr = [r for r in recovery if int(r["seed"]) == seed and
                      r["branch"] == branch and int(r["task"]) == task and
                      int(r["step"]) == step]
                assert len(rr) == 20 and len({int(r["unit_id"]) for r in rr}) == 20
                curve.append(sum(int(r["sink_q_ge_0_95"]) for r in rr))
            curves.append(curve)
        curves = np.asarray(curves)
        for curve in curves:
            ax.plot([p[2] for p in points], curve, color=color, ls=ls,
                    lw=.7, alpha=.18)
        ax.plot([p[2] for p in points], curves.mean(0), color=color, ls=ls,
                lw=2.4, marker=".", ms=3.5, label=label)
    for x in range(6000, 30000, 6000):
        ax.axvline(x, color="#888888", lw=.6, alpha=.25)
    ax.annotate("once-lift at update 0\n(no ongoing clamp)", xy=(0, .3), xytext=(2200, 5.0),
                color="#9a4227", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#9a4227", lw=1.2))
    ax.annotate("task 22: rise occurs\nover the first 75 updates", xy=(6075, 19.4), xytext=(8500, 14.3),
                color="#7b442f", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#7b442f", lw=1.0))
    ax.set(title="The same 20 selected IDs re-sink", xlabel="Cumulative updates across tasks 21–25",
           ylabel="Count with q ≥ .95 (mean of 3 seeds)", xlim=(-300, 30300), ylim=(-.5, 20.8),
           yticks=np.arange(0, 21, 5))
    ax.grid(alpha=.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .985), ncol=4, frameon=False)
    fig.suptitle("One-shot ELU rescue: learning improves while selected units mostly re-sink", y=.875, fontsize=14)
    fig.text(.5, .025,
             "Registered measurements; descriptive visualization. Frozen incident parameters do not freeze upstream inputs.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.savefig(HERE / "rescue_then_resinking.png", dpi=190)
    plt.close(fig)

if __name__ == "__main__":
    main()
