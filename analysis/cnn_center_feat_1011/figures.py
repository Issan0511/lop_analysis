#!/usr/bin/env python3
"""Figures of cnn_center_feat_1011 (spec §9).

    python analysis/cnn_center_feat_1011/figures.py   # -> results/cnn_center_feat_1011/fig_online.png, fig_layers.png, fig_posdc.png

Lines are the seed median, bands the seed range (min-max, not a confidence interval); Hs / CHs are the
single seed 10.  ref / C / H / CH are P181's runs (results/cnn_center_relu_1011, reused).  The numbers
behind every panel are in the per_task.csv / t0.csv files and tables/.
Palette: the dataviz reference categorical slots in fixed order, the P181 arms keeping P181's slots
(ref 1, C 2, H 3, CH 4); slots 3-5 sit below 3:1 on the surface, so every line also carries a direct
label and its own dash.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "cnn_center_feat_1011"
P181 = REPO / "results" / "cnn_center_relu_1011"
BACKUP = Path.home() / "Projects" / "obsidian-research-data"
REUSED = ("ref", "C", "H", "CH")
NEW = ("Hf", "CHf")
CTRL = ("Hs", "CHs")
COLOR = {"ref": "#2a78d6", "C": "#eb6834", "H": "#1baf7a", "CH": "#eda100", "Hs": "#e87ba4", "CHs": "#008300",
         "Hf": "#4a3aa7", "CHf": "#e34948"}
DASH = {"ref": (0, (1, 1.5)), "C": (0, (6, 2)), "H": (0, (2, 1.5)), "CH": (0, (4, 1.5, 1, 1.5)),
        "Hf": (0, ()), "CHf": (0, ()), "Hs": (0, (1, 1)), "CHs": (0, (3, 1))}
WIDTH = {"Hf": 2.4, "CHf": 2.4}
LABEL = {"ref": "ref（素の ReLU、P181）", "C": "C（入力の色、P181）", "H": "H（チャネル平均、P181）",
         "CH": "CH（C＋チャネル平均、P181）", "Hf": "H′（チャネル × 位置）", "CHf": "CH′（C＋チャネル × 位置）",
         "Hs": "Hs（位置をシャッフル・seed 10）", "CHs": "CHs（C＋位置をシャッフル・seed 10）"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LAYERS = ("c1", "c2", "f1", "f2")
NOISE_FLOOR = 0.0177                                  # spec §4: sqrt(beta/(2-beta))/sqrt(16)
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def base(arm: str) -> Path:
    return P181 if arm in REUSED else OUT


def load(arm: str) -> pd.DataFrame | None:
    p = base(arm) / arm / "per_task.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    t0 = base(arm) / arm / "t0.csv"
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
        ax.fill_between(x, lo.to_numpy(), hi.to_numpy(), color=COLOR[arm], alpha=0.12, linewidth=0)
    ax.plot(x, med.to_numpy(), color=COLOR[arm], lw=WIDTH.get(arm, 1.6), linestyle=DASH[arm], label=LABEL[arm])
    return x[-1], float(med.iloc[-1])


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def end_labels(ax, ends, gap):
    ends.sort()
    placed = []
    for ye, arm, xe in ends:
        y = ye
        for p in placed:
            if abs(y - p) < gap:
                y = p + gap
        placed.append(y)
        ax.text(xe + 0.4, y, {"Hf": "H′", "CHf": "CH′"}.get(arm, arm), color=INK, fontsize=8, va="center")


def fig_online(frames: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=160)
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
    end_labels(ax, ends, 0.045)
    ax.set_xlim(0.5, 33.0)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("課題")
    ax.set_ylabel("online 精度（更新前の当たり率の課題平均）")
    ax.set_title("ReLU の CNN（RL-CIFAR）: 中心化の扉ごとの online（線 = seed 中央値、帯 = seed の範囲）",
                 fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_online.png", bbox_inches="tight")
    plt.close(fig)


def fig_layers(frames: dict) -> None:
    rows = (("zbar", "位置 z̄（単位中央値）", "symlog"), ("mob", "開いた割合（単位中央値）", "log"),
            ("dead_frac", "死亡率（全画像で z ≤ 0 の単位）", "linear"))
    show = ("H", "CH", "Hf", "CHf")
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.6), dpi=160, sharex=True)
    for j, l in enumerate(LAYERS):
        for i, (q, name, scale) in enumerate(rows):
            ax = axes[i, j]
            for arm in show:
                d = frames.get(arm)
                if d is None:
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
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("層ごとの位置・開いた割合・死亡率: チャネル平均（P181）と チャネル × 位置（線 = seed 中央値、帯 = 範囲、灰 = 主窓）",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_layers.png", bbox_inches="tight")
    plt.close(fig)


def fig_posdc() -> None:
    """The per-position mean of f1's input after the door, over its rms deviation (tables/pos_dc.csv)."""
    p = OUT / "tables" / "pos_dc.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    t = t[t.layer == "f1"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ends = []
    for arm in REUSED + NEW + CTRL:
        d = t[t.arm == arm]
        if not len(d):
            continue
        g = d.groupby("task").pdc_post
        med, lo, hi = g.median(), g.min(), g.max()
        x = med.index.to_numpy()
        if d.seed.nunique() > 1:
            ax.fill_between(x, lo.clip(lower=1e-4), hi.clip(lower=1e-4), color=COLOR[arm], alpha=0.12, linewidth=0)
        ax.plot(x, med.clip(lower=1e-4), color=COLOR[arm], lw=WIDTH.get(arm, 1.6), linestyle=DASH[arm],
                marker="o", markersize=3, label=LABEL[arm])
        ends.append((float(np.log10(max(med.iloc[-1], 1e-4))), arm, x[-1]))
    ax.set_yscale("log")
    ax.axhline(NOISE_FLOOR, color=MUTED, lw=0.8, linestyle=(0, (2, 2)))
    ax.text(0.3, NOISE_FLOOR * 1.08, "EMA の揺れの見積もり 0.018（独立な標本を仮定、上界寄り）", color=MUTED, fontsize=7.5, va="bottom")
    ends.sort()
    placed = []
    for ye, arm, xe in ends:
        y = ye
        for q in placed:
            if abs(y - q) < 0.13:
                y = q + 0.13
        placed.append(y)
        ax.text(xe + 0.5, 10 ** y, {"Hf": "H′", "CHf": "CH′"}.get(arm, arm), color=INK, fontsize=8, va="center")
    ax.set_xlim(-0.5, 33)
    ax.set_xlabel("課題（0 = 初期値、課題末の 1,200 枚）")
    ax.set_ylabel("‖位置ごとの平均‖ / rms 偏差（扉の後）")
    ax.set_title("f1 の入力（c2 の pool 出力 16 × 8×8）に残る位置ごとの平均（線 = seed 中央値、帯 = 範囲）",
                 fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_posdc.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frames = {a: load(a) for a in REUSED + NEW + CTRL}
    fig_online(frames)
    fig_layers(frames)
    fig_posdc()
    print("wrote", OUT / "fig_online.png", OUT / "fig_layers.png", OUT / "fig_posdc.png")


if __name__ == "__main__":
    main()
