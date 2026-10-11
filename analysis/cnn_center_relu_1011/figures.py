#!/usr/bin/env python3
"""Figures of cnn_center_relu_1011 (spec §9).

    python analysis/cnn_center_relu_1011/figures.py   # -> results/cnn_center_relu_1011/fig_online.png, fig_layers.png

Lines are the seed median, bands the seed range (min-max, not a confidence interval); Cr is the
single seed 10.  The numbers behind every panel are in per_task.csv / t0.csv and tables/.
Palette: the dataviz reference categorical slots 1-5 in fixed order (validated light mode;
slots 3-5 sit below 3:1 on the surface, so every line also carries a direct label and its own dash).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "cnn_center_relu_1011"
ARMS = ("ref", "C", "H", "CH")
COLOR = {"ref": "#2a78d6", "C": "#eb6834", "H": "#1baf7a", "CH": "#eda100", "Cr": "#e87ba4"}
DASH = {"ref": (0, ()), "C": (0, (6, 2)), "H": (0, (2, 1.5)), "CH": (0, ()), "Cr": (0, (1, 1.5))}
LABEL = {"ref": "ref（素の ReLU）", "C": "C（入力の中心化）", "H": "H（中間層の中心化）", "CH": "CH（両方）",
         "Cr": "Cr（乱数の定数・seed 10）"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LAYERS = ("c1", "c2", "f1", "f2")
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def load(arm: str) -> pd.DataFrame | None:
    p = OUT / arm / "per_task.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    t0 = OUT / arm / "t0.csv"
    if t0.exists():
        d = pd.concat([pd.read_csv(t0), d], ignore_index=True)
    return d


def band(ax, d: pd.DataFrame, col: str, arm: str, tmin: int = 1, clip=None):
    g = d[d.task >= tmin].groupby("task")[col]
    med, lo, hi = g.median(), g.min(), g.max()
    if clip is not None:
        med, lo, hi = med.clip(lower=clip), lo.clip(lower=clip), hi.clip(lower=clip)
    x = med.index.to_numpy()
    if d.seed.nunique() > 1:
        ax.fill_between(x, lo.to_numpy(), hi.to_numpy(), color=COLOR[arm], alpha=0.14, linewidth=0)
    ax.plot(x, med.to_numpy(), color=COLOR[arm], lw=2, linestyle=DASH[arm], label=LABEL[arm])
    return x[-1], float(med.iloc[-1])


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_online(frames: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.axvspan(20.5, 30.5, color="#f0efec", zorder=0)
    ax.text(25.5, 1.005, "主窓 t21–30", ha="center", va="bottom", color=MUTED, fontsize=8)
    ax.axhline(0.5, color=MUTED, lw=0.8, linestyle=(0, (2, 2)))
    ax.text(0.7, 0.51, "救命の線 0.5", color=MUTED, fontsize=8, va="bottom")
    ends = []
    for arm, d in frames.items():
        if d is None:
            continue
        xe, ye = band(ax, d, "online_acc", arm)
        ends.append((ye, arm, xe))
    # direct labels at the right end, nudged apart so they do not collide
    ends.sort()
    placed = []
    for ye, arm, xe in ends:
        y = ye
        for p in placed:
            if abs(y - p) < 0.045:
                y = p + 0.045
        placed.append(y)
        ax.text(xe + 0.4, y, arm, color=INK, fontsize=8, va="center")
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("課題")
    ax.set_ylabel("online 精度（更新前の当たり率の課題平均）")
    ax.set_title("ReLU の CNN（RL-CIFAR・Kumar らの CNN）: 腕ごとの online（線 = seed 中央値、帯 = seed の範囲）",
                 fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_online.png")
    plt.close(fig)


def fig_layers(frames: dict) -> None:
    rows = (("zbar", "位置 z̄（単位中央値）", "symlog"), ("mob", "開いた割合（単位中央値）", "log"),
            ("dead_frac", "死亡率（全画像で z ≤ 0 の単位）", "linear"))
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.6), dpi=160, sharex=True)
    for j, l in enumerate(LAYERS):
        for i, (q, name, scale) in enumerate(rows):
            ax = axes[i, j]
            for arm, d in frames.items():
                if d is None or arm == "Cr":
                    continue
                band(ax, d, f"{q}_{l}", arm, tmin=0, clip=1e-4 if scale == "log" else None)
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=1.0)
                ax.axhline(0, color=MUTED, lw=0.6)
            elif scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(8e-5, 1.2)
                ax.axhline(0.01, color=MUTED, lw=0.8, linestyle=(0, (2, 2)))
                if j == 0:
                    ax.text(0.5, 0.012, "1%（層の機能の低下）", color=MUTED, fontsize=7)
            else:
                ax.set_ylim(-0.03, 1.03)
            ax.axvspan(20.5, 30.5, color="#f0efec", zorder=0)
            style(ax)
            if i == 0:
                ax.set_title(l, color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            if i == 2:
                ax.set_xlabel("課題（0 = 初期値）")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("層ごとの位置・開いた割合・死亡率（線 = seed 中央値、帯 = seed の範囲、灰の帯 = 主窓）",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_layers.png")
    plt.close(fig)


def main() -> None:
    frames = {a: load(a) for a in ARMS + ("Cr",)}
    fig_online(frames)
    fig_layers(frames)
    print("wrote", OUT / "fig_online.png", OUT / "fig_layers.png")


if __name__ == "__main__":
    main()
