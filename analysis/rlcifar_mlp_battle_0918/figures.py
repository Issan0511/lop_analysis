#!/usr/bin/env python3
"""Figures for rlcifar_mlp_battle_0918 (run report.py first).

    python3 analysis/rlcifar_mlp_battle_0918/figures.py [--src DIR]

figure.png     (a) window per arm, raw vs std, one dot per seed
               (b) online_acc against task, median over seeds, raw
               (c) the same for std
phase.png      theta = 2 alpha_i z at t50 for the four snake arms, with each arm's gate on top
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "rlcifar_mlp_battle_0918"
ARMS = ("SNA", "KKA", "KKA23", "KKT1", "R", "LK001", "LR", "LK03", "SL", "RSL", "ELU", "SILU", "GELU")
SNAKE = ("SNA", "KKA", "KKA23", "KKT1")
PI = math.pi
COL = {"SNA": "#1f77b4", "KKA": "#ff7f0e", "KKA23": "#2ca02c", "KKT1": "#d62728", "R": "#7f7f7f",
       "LK001": "#9467bd", "LR": "#8c564b", "LK03": "#e377c2", "SL": "#17becf", "RSL": "#bcbd22",
       "ELU": "#aec7e8", "SILU": "#ffbb78", "GELU": "#98df8a"}
for f in ("Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break


def load(src: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(src / a / "per_task.csv", float_precision="round_trip")
                      for a in ARMS if (src / a / "per_task.csv").exists()], ignore_index=True)


def gate(arm, th):
    s = 1 + np.sin(th)
    if arm == "SNA":
        return s
    if arm == "KKT1":
        return np.where((th >= -2 * PI) & (th <= PI), s, 1.0)
    g = np.where((th >= -1.5 * PI) & (th <= 0.5 * PI), s, 2.0)
    return g * (2 / 3) if arm == "KKA23" else g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    a = ap.parse_args()
    src = Path(a.src) if a.src else OUT
    d = load(src)
    win = d[(d.task >= 31) & (d.task <= 50)].groupby(["arm", "cond", "seed"]).online_acc.mean().reset_index()
    order = [x for x in ARMS if x in set(win.arm)]

    fig = plt.figure(figsize=(12, 8.5))
    ax = fig.add_subplot(2, 1, 1)
    for i, arm in enumerate(order):
        for j, (cond, mk, dx) in enumerate((("raw", "o", -0.14), ("std", "^", 0.14))):
            v = win[(win.arm == arm) & (win.cond == cond)].online_acc.to_numpy()
            if not len(v):
                continue
            ax.scatter(np.full(len(v), i + dx), v, s=18, marker=mk, alpha=0.55,
                       color=COL[arm], edgecolor="none")
            ax.plot([i + dx - 0.1, i + dx + 0.1], [np.median(v)] * 2, color=COL[arm], lw=2.4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.set_ylabel("窓 t31–50 の online 正解率"); ax.set_ylim(0, 1.02)
    ax.set_title("RL-CIFAR × MLP・活性化バトル（丸 = raw、三角 = std、点 = seed、太線 = 中央値）")
    for k, cond in enumerate(("raw", "std")):
        ax = fig.add_subplot(2, 2, 3 + k)
        for arm in order:
            g = d[(d.arm == arm) & (d.cond == cond)].groupby("task").online_acc.median()
            ax.plot(g.index, g.values, color=COL[arm], lw=1.6, label=arm)
        ax.set_xlabel("タスク"); ax.set_ylabel("online 正解率（seed 中央値）")
        ax.set_title(f"入力 {cond}"); ax.set_ylim(0, 1.02)
        if k == 1:
            ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "figure.png", dpi=130)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    th = np.linspace(-3 * PI, 2.5 * PI, 2000)
    edges = np.linspace(-3 * PI, 2.5 * PI, 221)
    ctr = (edges[:-1] + edges[1:]) / 2
    for li, ax in enumerate(axes, start=1):
        for arm in SNAKE:
            hs = []
            for seed in range(10):
                f = src / arm / "hist" / f"{arm}_raw_seed{seed}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                if "th1" not in z.files or z[f"th{li}"].shape[0] < 50:
                    continue
                e = z["th_edges"]; c = (e[:-1] + e[1:]) / 2
                h, _ = np.histogram(c, bins=edges, weights=z[f"th{li}"][49].astype(float))
                hs.append(h / max(h.sum(), 1))
            if hs:
                m = np.mean(hs, 0)
                ax.fill_between(ctr, 0, m / max(m.max(), 1e-9) * 1.9, color=COL[arm], alpha=0.16, step="mid",
                                label=f"{arm} の前活性 (t50)")
            ax.plot(th, gate(arm, th), color=COL[arm], lw=1.6, ls="--" if arm != "SNA" else "-",
                    label=f"{arm} のゲート")
        for x in (-1.5 * PI, -PI / 2, 0, PI / 2):
            ax.axvline(x, color="0.8", lw=0.8, ls=":")
        ax.set_ylabel(f"第{li}層  φ′(θ)"); ax.set_ylim(-0.05, 2.3)
        if li == 1:
            ax.legend(fontsize=7, ncol=2, loc="upper left")
    axes[-1].set_xlabel("θ = 2α_i z（per-unit の α、raw、seed 平均）")
    axes[-1].set_xticks([-3 * PI, -2 * PI, -1.5 * PI, -PI, -PI / 2, 0, PI / 2, PI, 1.5 * PI, 2 * PI])
    axes[-1].set_xticklabels(["−3π", "−2π", "−3π/2", "−π", "−π/2", "0", "π/2", "π", "3π/2", "2π"])
    fig.tight_layout(); fig.savefig(OUT / "phase.png", dpi=130)
    print("wrote", OUT / "figure.png", OUT / "phase.png")


if __name__ == "__main__":
    main()
