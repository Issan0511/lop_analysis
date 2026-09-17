#!/usr/bin/env python3
"""swish_battle_0917 — online accuracy per task, one figure per box.

Two panels on one y-axis: the Swish arms and the Snake arms, with LR and R as gray
references in both.  Every arm keeps one colour across panels and boxes (fixed order of
the validated categorical palette); lines are seed means; the registered window
(tasks 31-50) is shaded.  Reads the same complete-box data as report.py.

    python3 analysis/swish_battle_0917/figures.py --box mlp [--src DIR --allow-partial]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"SWA1": "#2a78d6", "SW1": "#eb6834", "SWA3": "#1baf7a", "SW3": "#eda100",
         "SNA": "#e87ba4", "SNAc3": "#008300", "SN06": "#4a3aa7", "SN3": "#e34948"}
GRAY = {"LR": "#8f8e89", "R": "#3f3e3b"}
COLOR.update({"SWA1u": COLOR["SWA1"], "SWA3u": COLOR["SWA3"]})   # same entity, floor lowered
DASHED = {"SWA1u", "SWA3u"}
PANELS = (("Swish（適応 SWA・固定 SW、破線 = α の下限を外した u）",
           ("SWA1", "SW1", "SWA3", "SW3", "SWA1u", "SWA3u")),
          ("Snake（適応 SNA・固定 SN）", ("SNA", "SNAc3", "SN06", "SN3")))
REFS = {"cnn": {"LR": "LR", "R": "R"}, "mlp": {"LR": "LR", "R": "R@gpu"}}


def font():
    for name in ("Noto Sans CJK JP", "Noto Sans CJK TC", "IPAexGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return name
    return None


def curve(d, arm):
    g = d[d.arm == arm]
    return g.groupby("task").online_acc.mean()


def spread_labels(items, lo, hi, gap):
    """items: [(y, text)] -> y positions at least `gap` apart, order kept."""
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - hi if ys and ys[-1] > hi else 0
    ys = [max(lo, y - over) for y in ys]
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - gap)
    return list(zip(ys, [t for _, t in items], [y for y, _ in items]))


def draw(box, d, out):
    font()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (title, arms) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axvspan(R.WIN[0] - 0.5, R.WIN[1] + 0.5, color="#f0efec", zorder=0, lw=0)
        ax.text((R.WIN[0] + R.WIN[1]) / 2, 1.005, "判定窓 t31–50", ha="center",
                va="bottom", color=INK2, fontsize=8.5)
        labels = []
        for ref, key in REFS[box].items():
            c = curve(d, key)
            if len(c):
                ax.plot(c.index, c.values, color=GRAY[ref], lw=1.4, zorder=2,
                        solid_capstyle="round", label=f"{ref}（参照）")
                labels.append((float(c.iloc[-1]), ref))
        for arm in arms:
            c = curve(d, arm)
            if not len(c):
                continue
            ax.plot(c.index, c.values, color=COLOR[arm], lw=2, zorder=3,
                    ls=(0, (4, 2)) if arm in DASHED else "-",
                    solid_capstyle="round", solid_joinstyle="round",
                    dash_capstyle="round", label=arm)
            ax.plot([c.index[-1]], [c.iloc[-1]], "o", ms=6.5, color=COLOR[arm],
                    mec=SURFACE, mew=2, zorder=4)
            labels.append((float(c.iloc[-1]), arm))
        for y, t, y0 in spread_labels(labels, 0.08, 1.0, 0.045):
            ax.annotate(t, xy=(R.N_TASKS, y0), xytext=(R.N_TASKS + 1.8, y),
                        color=INK2, fontsize=8.5, va="center",
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=46)
        ax.set_xlim(0.5, R.N_TASKS + 9)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xlabel("タスク", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.035), ncol=4, frameon=False,
                  fontsize=8.5, labelcolor=INK2, handlelength=1.6, borderaxespad=0)
    axes[0].set_ylabel("online 正解率（seed 平均）", color=INK2)
    name = {"mlp": "RL-MNIST・MLP（CPU、R のみ GPU 参照）",
            "cnn": "RL-CIFAR・CNN（GPU、SNA・SN06・SN3・LR・R は 0908 の参照）"}[box]
    fig.suptitle(f"swish_battle_0917 — {name}", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def draw_rank(box, d, out):
    """Window error rate (1 - online acc, t31-50) per arm on a log axis: one small dot
    per seed and a ring for the mean, arms sorted best first."""
    font()
    arms = [a for a in list(COLOR) + list(REFS[box].values()) if (d.arm == a).any()]
    rows = []
    for a in arms:
        v = R.per_seed(d, a, "online_acc", *R.WIN)
        if len(v):
            rows.append((a, 1 - v.values, float(1 - v.mean())))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (a, e, m) in enumerate(rows):
        base = a.split("@")[0]
        col = COLOR.get(base, GRAY.get(base, INK2))
        y = len(rows) - 1 - i
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        ax.scatter(e, [y] * len(e), s=18, color=col, alpha=0.55, lw=0, zorder=2)
        ax.scatter([m], [y], s=70, facecolor=SURFACE, edgecolor=col, lw=2, zorder=3,
                   marker="s" if base in DASHED else "o")
        ax.annotate(f"{1 - m:.3f}", xy=(m, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK2, fontsize=9)
    ax.set_xscale("log")
    lo = min(min(r[1].min() for r in rows), 0.02)
    ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if t >= lo / 1.5]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ticks[0] / 1.2, 1.1)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.set_xlabel("判定窓 t31–50 の誤り率 1 − online（対数軸・左ほど良い）。点 = seed、輪 = 平均、数字 = 平均の online 正解率",
                  color=INK2, fontsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0, axis="y")
    ax.set_ylim(-0.7, len(rows) - 0.2)
    name = {"mlp": "RL-MNIST・MLP", "cnn": "RL-CIFAR・CNN"}[box]
    ax.set_title(f"swish_battle_0917 — {name} の順位", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True, choices=("mlp", "cnn"))
    ap.add_argument("--src", default=str(R.OUT))
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    if a.allow_partial and src.resolve() == R.OUT.resolve():
        raise SystemExit("--allow-partial is for smoke directories only")
    d, _, _, _ = R.load(a.box, src, a.allow_partial)
    draw(a.box, d, src / f"online_{a.box}.png")
    draw_rank(a.box, d, src / f"rank_{a.box}.png")


if __name__ == "__main__":
    main()
