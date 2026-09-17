#!/usr/bin/env python3
"""avgpool_cnn_0917 — one figure, three panels, seed means over tasks:
online accuracy; the conv2 seat 2*alpha*zbar against the valley of phi' (-pi/2); and the
share of SNA's conv2 channels past the valley (registered call A at task 50).

Colour = activation (SNA blue, SN3 orange; validated pair), line = pool (solid avg,
dashed max = the rlcifar_cnn_0908 reference).  Reads the same complete box as report.py.

    python3 analysis/avgpool_cnn_0917/figures.py [--src DIR --allow-partial]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R  # noqa: E402

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"
COLOR = {"SNA": "#2a78d6", "SN3": "#eb6834"}
LINES = (("SNA_avg", "SNA", "-"), ("SNA", "SNA", "--"), ("SN3_avg", "SN3", "-"), ("SN3", "SN3", "--"))
NAME = {"SNA_avg": "SNA avg-pool", "SNA": "SNA max-pool", "SN3_avg": "SN3 avg-pool", "SN3": "SN3 max-pool"}


def font():
    for name in ("Noto Sans CJK JP", "Noto Sans CJK TC", "IPAexGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return name
    return None


def spread(items, gap):
    """[(y, text, color)] -> label y positions at least `gap` apart, order kept."""
    items = sorted(items)
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    return [(y2, t, c, y) for y2, (y, t, c) in zip(ys, items)]


def seat_curve(d, arm, host):
    g = d[(d.arm == arm) & d.online_acc.notna()]
    a = R.FIXED_ALPHA.get(arm)
    s = 2 * (a if a is not None else g["alpha_med_c2"]) * g["zbar_c2"]
    return s.groupby(g.task).mean()


def far_curve(d, hists, arm):
    per_seed = []
    for s in R.SEEDS:
        p = hists.get((arm, s))
        if p is None or not p.exists():
            continue
        m = np.load(p)["m_c2"].astype(np.float64)                    # (tasks, 16)
        g = d[(d.arm == arm) & (d.seed == s) & d.online_acc.notna()].sort_values("task")
        a = R.FIXED_ALPHA.get(arm)
        a = np.full(len(g), a) if a is not None else g["alpha_med_c2"].to_numpy()
        n = min(len(g), m.shape[0])
        per_seed.append((2 * a[:n, None] * m[:n] < R.VALLEY).mean(1))
    n = min(len(x) for x in per_seed)
    return np.arange(1, n + 1), np.mean([x[:n] for x in per_seed], axis=0)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("タスク", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlim(0.5, 50.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def end_labels(ax, curves, gap):
    items = [(float(c.iloc[-1]), NAME[a], COLOR[h]) for a, h, c in curves]
    for y2, text, col, y in spread(items, gap):
        ax.annotate(text, (50, y), xytext=(52, y2), color=INK2, fontsize=8.5, va="center",
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(R.OUT))
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    if a.allow_partial and src == R.OUT.resolve():
        raise SystemExit("--allow-partial is for a --src other than the main results")
    d, prov, missing, refs, hists = R.load(src, a.allow_partial)
    font()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.42))
    # online accuracy
    ax = axes[0]
    style(ax, "online 正解率（網掛け = 判定の窓 t31–50）")
    ax.axvspan(R.WIN[0] - 0.5, R.WIN[1] + 0.5, color=BAND, zorder=0, lw=0)
    curves = []
    for arm, host, ls in LINES:
        c = d[(d.arm == arm) & d.online_acc.notna()].groupby("task").online_acc.mean()
        ax.plot(c.index, c.values, color=COLOR[host], ls=ls, lw=2, label=NAME[arm], zorder=3)
        curves.append((arm, host, c))
    end_labels(ax, curves, 0.0028)
    ax.set_ylabel("seed 平均", color=INK2)
    # conv2 seat
    ax = axes[1]
    style(ax, "conv2 の座席 2α·z̄（点線 = φ′ の谷 −π/2）")
    ax.axhline(R.VALLEY, color=INK2, lw=1, ls=":", zorder=1)
    curves = []
    for arm, host, ls in LINES:
        c = seat_curve(d, arm, host)
        ax.plot(c.index, c.values, color=COLOR[host], ls=ls, lw=2, zorder=3)
        curves.append((arm, host, c))
    end_labels(ax, curves, 0.22)
    # far-side share, SNA only (call A)
    ax = axes[2]
    style(ax, "SNA の conv2 で谷を越えたチャネルの割合")
    ax.axhline(R.A_CUT, color=INK2, lw=1, ls=":", zorder=1)
    ax.annotate("判定 A の閾値 0.20", (1.5, R.A_CUT), xytext=(0, 4), textcoords="offset points",
                color=INK2, fontsize=8.5)
    curves = []
    for arm, ls in (("SNA_avg", "-"), ("SNA", "--")):
        t, v = far_curve(d, hists, arm)
        ax.plot(t, v, color=COLOR["SNA"], ls=ls, lw=2, zorder=3)
        curves.append((arm, "SNA", __import__("pandas").Series(v, index=t)))
    end_labels(ax, curves, 0.05)
    ax.set_ylim(-0.02, 1.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("avgpool_cnn_0917 — max-pool を avg-pool に替えた RL-CIFAR CNN（10 seed の平均）",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.subplots_adjust(left=0.05, right=0.93, top=0.83, bottom=0.12)
    tag = "" if src == R.OUT.resolve() else "_" + src.name
    out = R.OUT / f"figure{tag}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
