#!/usr/bin/env python3
"""rlcifar_mlp_battle_0918 -- animate the preactivation distribution over the 50 tasks.

Reads only the per-task histograms the run already wrote (`<arm>/hist/<arm>_<cond>_seed<k>.npz`,
host `preact_hist`: z bins [-512, 256] width 0.1, and for the snake family the same mass in
theta = 2 alpha_i z_i on [-6pi, 6pi]). Nothing is recomputed from the snapshots.

Left column: z on a log density so the bulk and the runaway tails are both visible.
Right column: theta for the snake arms, with each arm's gate phi'(theta) drawn behind it.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "rlcifar_mlp_battle_0918"
PI = math.pi
SNAKE = ("SNA", "KKA", "KKA23", "KKT1")
# same arm, same colour as figures.py
COLORS = {"SNA": "#1f77b4", "KKA": "#ff7f0e", "KKA23": "#2ca02c", "KKT1": "#d62728", "R": "#7f7f7f",
          "LK001": "#9467bd", "LR": "#8c564b", "LK03": "#e377c2", "SL": "#17becf", "RSL": "#bcbd22",
          "ELU": "#aec7e8", "SILU": "#ffbb78", "GELU": "#98df8a"}
for _f in ("Noto Sans CJK JP", "Noto Sans CJK TC", "IPAGothic", "DejaVu Sans"):
    if any(_f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


def gate(arm: str, th: np.ndarray) -> np.ndarray:
    """phi'(theta) for the snake family, in theta = 2 alpha z coordinates."""
    g = 1.0 + np.sin(th)
    if arm == "SNA":
        return g
    if arm in ("KKA", "KKA23"):
        g = np.where((th < -1.5 * PI) | (th > 0.5 * PI), 2.0, g)
        return g * (2.0 / 3.0) if arm == "KKA23" else g
    return np.where((th < -2.0 * PI) | (th > PI), 1.0, g)      # KKT1


def load(arm: str, cond: str, seed: int, src: Path) -> dict:
    d = np.load(src / arm / "hist" / f"{arm}_{cond}_seed{seed}.npz")
    return {k: d[k] for k in d.files}


def density(h: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mass per unit of the coordinate, so bins of different width compare."""
    w = np.diff(edges)
    tot = h.sum()
    return h / (tot * w) if tot else h.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="raw", choices=("raw", "std"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="SNA,KKA,LR,SL,GELU,ELU")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=20, help="bins to merge for the z curves")
    ap.add_argument("--zlim", type=float, default=320.0, help="x range of the z panel")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = Path(a.src) if a.src else OUT
    arms = [s for s in a.arms.split(",") if s]
    snake = list(SNAKE)                       # the theta panel always shows all four

    D = {arm: load(arm, a.cond, a.seed, src) for arm in set(arms) | set(snake)}
    edges = D[arms[0]]["edges"]
    th_edges = D[snake[0]]["th_edges"]
    n_tasks = D[arms[0]]["h1"].shape[0]

    # merge z bins so 7680 thin bins read as a curve; keep the mass exact
    k = a.smooth
    ze = edges[::k]
    zc = (ze[:-1] + ze[1:]) / 2
    thc = (th_edges[:-1] + th_edges[1:]) / 2

    def zcurve(h):
        return density(h[: (len(h) // k) * k].reshape(-1, k).sum(1), ze)

    Z = {arm: {L: np.stack([zcurve(D[arm][f"h{L}"][t]) for t in range(n_tasks)]) for L in (1, 2)}
         for arm in arms}
    T = {arm: {L: np.stack([density(D[arm][f"th{L}"][t], th_edges) for t in range(n_tasks)])
               for L in (1, 2)} for arm in snake}

    fig, ax = plt.subplots(2, 2, figsize=(13.0, 7.4), dpi=96)
    fig.subplots_adjust(left=0.065, right=0.945, top=0.895, bottom=0.075, hspace=0.30, wspace=0.17)
    tlim = max(T[arm][L].max() for arm in snake for L in (1, 2)) * 1.05
    zfloor = 2e-5

    lines = {}
    for r, L in enumerate((1, 2)):
        axz, axt = ax[r, 0], ax[r, 1]
        axz.set_yscale("log")
        axz.set_xlim(-a.zlim, a.zlim / 2)
        axz.set_ylim(zfloor, 1.0)
        axz.axvline(0, color="0.8", lw=1, zorder=0)
        axz.set_ylabel(f"第 {L} 層  密度（対数）")
        axz.set_xlabel("前活性 z" if r == 1 else "")
        for arm in arms:
            lines[("z", arm, L)] = axz.plot([], [], lw=1.6, color=COLORS[arm], label=arm)[0]

        # gate behind the theta panel
        gx = np.linspace(th_edges[0], th_edges[-1], 2000)
        axg = axt.twinx()
        for arm in snake:
            axg.plot(gx, gate(arm, gx), lw=1.0, color=COLORS[arm], alpha=0.30, zorder=0)
        axg.set_ylim(-0.1, 2.3)
        axg.set_ylabel("ゲート φ′（薄線）", color="0.45", fontsize=9)
        axg.tick_params(axis="y", colors="0.45", labelsize=8)
        axt.set_zorder(axg.get_zorder() + 1)
        axt.patch.set_visible(False)
        axt.set_xlim(-2.2 * PI, 1.6 * PI)
        axt.set_ylim(0, tlim)
        axt.set_xticks([-2 * PI, -1.5 * PI, -PI, -0.5 * PI, 0, 0.5 * PI, PI])
        axt.set_xticklabels(["−2π", "−3π/2", "−π", "−π/2", "0", "π/2", "π"])
        axt.axvline(-0.5 * PI, color="0.75", lw=1, ls=":", zorder=0)   # the gate's valley
        for e in (-1.5 * PI, 0.5 * PI):                                # kunekune's band edges
            axt.axvline(e, color="0.85", lw=1, zorder=0)
        axt.set_xlabel("θ = 2αz" if r == 1 else "")
        for arm in snake:
            lines[("t", arm, L)] = axt.plot([], [], lw=1.7, color=COLORS[arm], label=arm)[0]
        if r == 0:
            axz.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
            axt.legend(loc="upper left", fontsize=8, framealpha=0.9)
            axz.set_title("前活性 z（腕を重ねる。y は対数密度）", fontsize=10)
            axt.set_title("θ = 2αz（snake 系。谷は θ = −π/2、灰線は kunekune の帯）", fontsize=10)

    band = ((thc < -1.5 * PI) | (thc > 0.5 * PI))
    wth = np.diff(th_edges)
    outside = {L: np.array([(T["KKA"][L][t][band] * wth[band]).sum() for t in range(n_tasks)])
               for L in (1, 2)}
    txt = {L: ax[r, 1].text(0.985, 0.90, "", transform=ax[r, 1].transAxes, ha="right",
                            fontsize=9, color="0.25")
           for r, L in enumerate((1, 2))}
    sup = fig.suptitle("", fontsize=12)

    def frame(t: int):
        for arm in arms:
            for L in (1, 2):
                y = Z[arm][L][t]
                m = y > 0
                lines[("z", arm, L)].set_data(zc[m], y[m])
        for arm in snake:
            for L in (1, 2):
                lines[("t", arm, L)].set_data(thc, T[arm][L][t])
        for L in (1, 2):
            txt[L].set_text(f"KKA の帯の外: {outside[L][t] * 100:.1f}%")
        acc = ", ".join(f"{arm} {D[arm]['acc'][t]:.2f}" for arm in arms[:3])
        sup.set_text(f"RL-CIFAR × MLP・前活性の分布（入力 {a.cond}・seed {a.seed}）　"
                     f"タスク {t + 1}/{n_tasks}　　memo: {acc}")
        return list(lines.values()) + list(txt.values()) + [sup]

    dst = Path(a.out) if a.out else OUT / f"anim_preact_{a.cond}_seed{a.seed}.gif"
    FuncAnimation(fig, frame, frames=n_tasks, blit=False).save(
        dst, writer=PillowWriter(fps=a.fps))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, {n_tasks} frames)")


if __name__ == "__main__":
    main()
