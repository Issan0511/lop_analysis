#!/usr/bin/env python3
"""Figure for resp_ee_0917 (report only; the verdict is verdict.py's).

    python3 analysis/resp_ee_0917/figure.py [--src results/resp_ee_0917] [--out <png>] [--epochs 80]

Three panels, one y-axis each, seed means with 95% t intervals over the valid seeds:
  (a) restore: the natural continuation N<t> and the t2-field arm R2_<t> against the branch task t
  (b) sink: the t2 net with the t'-field, fresh Adam (S2_<t'>r), against t'; references N2r and S2u30r
  (c) the uniform ladder: fit fraction F = (E - 0.1) / (E(N2r) - 0.1) against the shift, with the
      eps band (Delta* q0.1-q0.9) and the float32-zero band (Delta0 q0.1-q0.9), and the layer-1 rung
Palette: the dataviz reference instance, light mode, slots 1-3 (validated: all checks pass, aqua needs
relief -> every series is direct-labelled).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(os.environ.get("RESP_EE_REPO") or Path(__file__).resolve().parents[2])
_spec = importlib.util.spec_from_file_location("_verdict", REPO / "analysis" / "resp_ee_0917" / "verdict.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND = "#f0efec"
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})


def stats(vals):
    v = np.asarray(vals, float)
    n = len(v)
    m = float(v.mean())
    h = V.t_quantile(0.975, n - 1) * float(v.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    return m, h


def series(shards, valid, arms, k=1):
    out = []
    for a in arms:
        m, h = stats([V.E(shards[s]["arms"], a, k) for s in valid])
        out.append((m, h))
    return np.array(out)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def line(ax, x, st, color, label, dx=0.0, ls="-", dy=0):
    ax.errorbar(np.asarray(x) + dx, st[:, 0], yerr=st[:, 1], color=color, lw=2, ls=ls, marker="o", ms=6,
                mec=SURFACE, mew=1.5, capsize=0, elinewidth=1, label=label)
    ax.annotate(label, (x[-1] + dx, st[-1, 0]), xytext=(7, dy), textcoords="offset points",
                va="center", color=INK, fontsize=8.5)


def ref(ax, y, text, color, side="right", below=False, label=None):
    """A dashed reference level, labelled at one edge (above the line, or below it)."""
    ax.axhline(y, color=color, lw=1.2, ls=(0, (4, 3)), label=label)
    x, dx, ha = (1, -2, "right") if side == "right" else (0, 2, "left")
    ax.annotate(text, (x, y), xycoords=("axes fraction", "data"), xytext=(dx, -3 if below else 3),
                textcoords="offset points", ha=ha, va="top" if below else "bottom",
                color=INK2, fontsize=8)


def legend(ax, **kw):
    lg = ax.legend(frameon=True, fontsize=7.5, handlelength=2.2, borderpad=0.5, labelcolor=INK, **kw)
    lg.get_frame().set_facecolor(SURFACE)
    lg.get_frame().set_edgecolor(GRID)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / "resp_ee_0917"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args(argv)
    src = Path(args.src)
    shards, missing = V.load(src)
    txt = (REPO / "src" / "resp_ee_0917.py").read_text()
    prereg = next((ln.split('"')[1] for ln in txt.splitlines() if ln.startswith('PREREG_COMMIT = "')), None)
    res = V.analyze(shards, prereg if args.epochs == 80 else None, args.epochs)
    valid = res["valid_seeds"]
    if len(valid) < 2:
        raise SystemExit(f"need >= 2 valid seeds, have {valid}")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"wspace": 0.42})

    ts = [5, 7, 10, 15, 20]
    ax = axs[0]
    sn = series(shards, valid, [f"N{t}" for t in ts])
    sr = series(shards, valid, [f"R2_{t}" for t in ts])
    up = 1 if sr[-1, 0] >= sn[-1, 0] else -1
    line(ax, ts, sn, BLUE, "自然継続 N⟨t⟩", dx=-0.15, dy=9 if up > 0 else -9)
    line(ax, ts, sr, ORANGE, "t2 の場 R2_⟨t⟩", dx=0.15, dy=6 * up)
    n2, _ = stats([V.E(shards[s]["arms"], "N2") for s in valid])
    ref(ax, n2, "N2（t2 からの自然継続）", MUTED, side="left", below=True, label="N2（参照）")
    ref(ax, 0.1, "偶然水準", MUTED, side="left", below=True)
    legend(ax, loc="center right")
    ax.set_xticks(ts)
    ax.set_xlim(ts[0] - 1, ts[-1] + 6)
    ax.set_ylim(0, 1)
    style(ax, "(a) 崩壊した網に t2 の応答を戻す", "分岐したタスク t", "次のタスクの online 正解率")

    ax = axs[1]
    line(ax, ts, series(shards, valid, [f"S2_{t}r" for t in ts]), ORANGE, "t′ の場 S2_⟨t′⟩r")
    n2r, _ = stats([V.E(shards[s]["arms"], "N2r") for s in valid])
    ro, _ = stats([V.E(shards[s]["arms"], "S2u30r") for s in valid])
    ref(ax, n2r, "N2r（応答はそのまま）", BLUE, side="right", label="N2r（参照）")
    ref(ax, ro, "S2u30r（第2層の勾配が 0）", AQUA, side="left", below=True, label="S2u30r（参照）")
    ref(ax, 0.1, "偶然水準", MUTED, side="right", below=True)
    legend(ax, loc="center left", bbox_to_anchor=(0.42, 0.55))
    ax.set_xticks(ts)
    ax.set_xlim(ts[0] - 1, ts[-1] + 6)
    ax.set_ylim(0, 1)
    style(ax, "(b) t2 の網に後の時刻の応答を渡す", "移植した場の時刻 t′（Adam 初期化）", "次のタスクの online 正解率")

    ax = axs[2]
    lad = res["ladder"]
    ds = {q: float(np.median([shards[s]["dstar"]["dstar_q"][q] for s in valid])) for q in ("0.1", "0.9")}
    dz = {q: float(np.median([shards[s]["dstar"]["dzero_q"][q] for s in valid])) for q in ("0.1", "0.9")}
    ax.axvspan(ds["0.1"], ds["0.9"], color=BAND, lw=0)
    ax.axvspan(dz["0.1"], dz["0.9"], facecolor="none", edgecolor=MUTED, hatch="///", lw=0)
    ax.text(0.02, 0.03, "灰色の帯: ε を横切る Δ*（W2 座標の q0.1–q0.9）\n斜線の帯: 訓練の微分が全画像で 0 になる Δ₀（q0.1–q0.9）",
            transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=7.5, linespacing=1.4,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.3"), zorder=5)
    xs = [0] + list(V.LADDER)
    fs = []
    for d in xs:
        if d == 0:
            fs.append((1.0, 0.0))
            continue
        f = [(V.E(shards[s]["arms"], f"S2u{d}r") - 0.1) / (V.E(shards[s]["arms"], "N2r") - 0.1) for s in valid]
        fs.append(stats(f))
    line(ax, xs, np.array(fs), BLUE, "第2層を −Δ", dy=0)
    f1 = [(V.E(shards[s]["arms"], "S1u20r") - 0.1) / (V.E(shards[s]["arms"], "N2r") - 0.1) for s in valid]
    m1, h1 = stats(f1)
    ax.errorbar([20.6], [m1], yerr=[h1], color=AQUA, marker="D", ms=6, mec=SURFACE, mew=1.5, lw=0, elinewidth=1,
                label="第1層を −20")
    ax.annotate("第1層を −20", (20.6, m1), xytext=(-8, 0), textcoords="offset points", va="center", ha="right",
                color=INK, fontsize=8.5)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 0.13))
    ax.axhline(0.5, color=GRID, lw=1)
    ax.set_xticks(xs)
    ax.set_xlim(-1, 36)
    arr = np.array(fs)
    ax.set_ylim(min(-0.1, float((arr[:, 0] - arr[:, 1]).min()) - 0.05),
                max(1.2, float((arr[:, 0] + arr[:, 1]).max()) + 0.1))
    style(ax, "(c) 応答の大きさだけを下げる", "シフト Δ（t2・Adam 初期化）", "適合率 F（N2r = 1、偶然 = 0）")

    fig.text(0.01, -0.02, f"resp_ee_0917・RL ELU→ELU・有効 seed {len(valid)}・平均と 95% t 区間・"
             f"判定 {res.get('label')}（{res['ladder']['label'] if 'ladder' in res else ''}）"
             + ("・部分集計" if missing else ""), color=INK2, fontsize=8)
    out = Path(args.out) if args.out else src / "fig_resp_ee_0917.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
