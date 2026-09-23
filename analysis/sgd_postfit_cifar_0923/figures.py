#!/usr/bin/env python3
"""Figures for sgd_postfit_cifar_0923 (descriptive): per-task medians over seeds, per arm.

    python3 analysis/sgd_postfit_cifar_0923/figures.py [--root results/sgd_postfit_cifar_0923]

fig_curves.png : hit999 (capped, dead = 30,000) and ||W1||^2 at the task's end, per task
fig_speed_vs_norm.png : hit_t against ||W1||^2 at the start of task t, every (arm, seed, t)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARMS = ("A", "AR", "F", "S_lo", "S_mid", "S_hi", "A_ce")
COL = {"A": "#c0392b", "AR": "#e67e22", "F": "#2c3e50", "S_lo": "#27ae60", "S_mid": "#16a085",
       "S_hi": "#8e44ad", "A_ce": "#2980b9"}
STEPS = 30000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923"))
    a = ap.parse_args()
    root = Path(a.root)
    arms = [x for x in ARMS if (root / x / "per_task.csv").exists()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    pts = {}
    for x in arms:
        rows = list(csv.DictReader(open(root / x / "per_task.csv")))
        by = {}
        for q in rows:
            by[(int(q["seed"]), int(q["task"]))] = q
        tasks = sorted({t for _, t in by})
        hit, n1 = [], []
        for t in tasks:
            hs, ns = [], []
            for s in range(10):
                q = by.get((s, t))
                dead = q is None or q.get("memo_acc") in (None, "")
                hs.append(STEPS if dead or int(q["hit999"]) < 0 else int(q["hit999"]))
                if not dead:
                    z = np.load(root / x / "trace" / f"LR_std_seed{s}.npz")
                    m = (z["task"] == t)
                    ns.append(float(z["n1"][m][-1]))
                    n0 = float(z["n1"][m][0])
                    if int(q["hit999"]) >= 0:
                        pts.setdefault(x, []).append((n0, int(q["hit999"])))
            hit.append(np.median(hs))
            n1.append(np.median(ns) if ns else np.nan)
        ax[0].plot(tasks, hit, color=COL[x], label=x, lw=1.6)
        ax[1].plot(tasks, n1, color=COL[x], label=x, lw=1.6)
    ax[0].set_xlabel("task")
    ax[0].set_ylabel("hit999 (median over seeds; dead = 30,000)")
    ax[0].set_ylim(0, 6000)
    ax[1].set_xlabel("task")
    ax[1].set_ylabel("||W1||^2 at the task's end (median over live seeds)")
    ax[1].set_yscale("log")
    for q in ax:
        q.grid(alpha=0.3)
        q.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "fig_curves.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for x, p in pts.items():
        p = np.array(p)
        ax.scatter(p[:, 0], p[:, 1], s=4, alpha=0.35, color=COL[x], label=x)
    ax.set_xscale("log")
    ax.set_xlabel("||W1||^2 at the start of task t")
    ax.set_ylabel("hit999 of task t")
    ax.set_ylim(0, 8000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(root / "fig_speed_vs_norm.png", dpi=130)
    print("wrote", root / "fig_curves.png", root / "fig_speed_vs_norm.png")


if __name__ == "__main__":
    main()
