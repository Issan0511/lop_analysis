#!/usr/bin/env python3
"""relu_doors_0919 -- animate the preactivation distribution over the 50 tasks.

Reads only the per-task histograms the run wrote (`<arm>/hist/<arm>_raw_seed<k>.npz`,
the host's `preact_hist`: z bins [-512, 256], width 0.1).  Nothing is recomputed.

The point of the picture is WHERE the mass sits relative to 0 -- the ReLU kink:
  ref / C / CS  go negative  -> the gate closes, the units die
  LN            goes positive -> the gate never closes, ReLU becomes the identity
  CH / CHB      straddle 0    -> the fold survives and the net keeps learning
so the share of mass on each side is printed per arm, per frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path.home() / "Projects" / "obsidian-research-data" / "relu_doors_0919" / \
    "results" / "relu_doors_0919"
OUT = REPO / "results" / "relu_doors_0919"
COL = {"ref": "#7f7f7f", "C": "#9467bd", "CH": "#1f77b4", "CHB": "#2ca02c",
       "CHB0": "#17becf", "CS": "#d62728", "LN": "#ff7f0e"}
LABEL = {"ref": "ref（素の ReLU）", "C": "C（入力の中心化）", "CH": "CH（＋中間層の中心化）",
         "CHB": "CHB（＋bias）", "CHB0": "CHB0（bias 除去）",
         "CS": "CS（分散だけ）", "LN": "LN（素の LayerNorm）"}
for _f in ("Noto Sans CJK JP", "Noto Sans CJK TC", "IPAGothic", "DejaVu Sans"):
    if any(_f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


def src_dir(arm: str) -> Path:
    """The run's own hist/, wherever it lives (the raw data is archived after cleanup)."""
    for base in (OUT, ARCHIVE):
        if arm == "ref":
            p = base.parent / "rlcifar_mlp_battle_0918" / "R" / "hist"
            q = Path.home() / "Projects" / "obsidian-research-data" / \
                "rlcifar_mlp_battle_0918" / "results" / "rlcifar_mlp_battle_0918" / "R" / "hist"
            if p.exists():
                return p
            if q.exists():
                return q
            continue
        if (base / arm / "hist").exists():
            return base / arm / "hist"
    raise SystemExit(f"no hist/ for {arm}")


def load(arm: str, seed: int) -> dict:
    name = "R_raw" if arm == "ref" else f"{arm}_raw"
    d = np.load(src_dir(arm) / f"{name}_seed{seed}.npz")
    return {k: d[k] for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ref,C,CS,CH,LN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=20)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--zlim", type=float, default=260.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [s for s in a.arms.split(",") if s]
    D = {arm: load(arm, a.seed) for arm in arms}
    edges = D[arms[0]]["edges"]
    k = a.smooth
    ze = edges[::k]
    zc = (ze[:-1] + ze[1:]) / 2
    w = np.diff(ze)
    n_tasks = D[arms[0]]["h1"].shape[0]

    def curve(h):
        m = h[: (len(h) // k) * k].reshape(-1, k).sum(1)
        t = m.sum()
        return m / (t * w) if t else m.astype(float)

    Z = {arm: {L: np.stack([curve(D[arm][f"h{L}"][t]) for t in range(n_tasks)]) for L in (1, 2)}
         for arm in arms}
    neg = {arm: {L: np.array([D[arm][f"h{L}"][t][edges[:-1] < 0].sum() / D[arm][f"h{L}"][t].sum()
                              for t in range(n_tasks)]) for L in (1, 2)} for arm in arms}

    fig, ax = plt.subplots(2, 1, figsize=(11.5, 7.2), dpi=100)
    fig.subplots_adjust(left=0.085, right=0.79, top=0.88, bottom=0.08, hspace=0.26)
    lines, txt = {}, {}
    for r, L in enumerate((1, 2)):
        b = ax[r]
        b.set_yscale("log")
        b.set_xlim(-a.zlim, a.zlim / 2)
        b.set_ylim(2e-5, 1.0)
        b.axvline(0, color="0.25", lw=1.6, zorder=0)
        b.text(0.012, 0.03, "← z = 0（ReLU の折れ目）→", transform=b.transAxes,
               fontsize=8, color="0.3", va="bottom")
        b.set_ylabel(f"第 {L} 層  密度（対数）")
        b.set_xlabel("前活性 z" if r == 1 else "")
        b.grid(alpha=.2)
        for arm in arms:
            lines[(arm, L)] = b.plot([], [], lw=1.9, color=COL[arm], label=LABEL[arm])[0]
        for i, arm in enumerate(arms):
            txt[(arm, L)] = b.text(1.012, 0.93 - 0.13 * i, "", transform=b.transAxes,
                                   fontsize=8.5, color=COL[arm], va="top", family="monospace")
    ax[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    sup = fig.suptitle("", fontsize=12)

    def frame(t: int):
        for arm in arms:
            for L in (1, 2):
                y = Z[arm][L][t]
                m = y > 0
                lines[(arm, L)].set_data(zc[m], y[m])
                txt[(arm, L)].set_text(f"{arm:>5s}  z<0 {neg[arm][L][t] * 100:5.1f}%")
        acc = "  ".join(f"{arm} {D[arm]['acc'][t]:.2f}" for arm in arms)
        sup.set_text(f"relu_doors_0919・前活性の分布（入力 raw・seed {a.seed}）　"
                     f"タスク {t + 1}/{n_tasks}\nmemo: {acc}")
        return list(lines.values()) + list(txt.values()) + [sup]

    dst = Path(a.out) if a.out else OUT / f"anim_preact_seed{a.seed}.gif"
    dst.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, frame, frames=n_tasks, blit=False).save(dst, writer=PillowWriter(fps=a.fps))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, {n_tasks} frames)")


if __name__ == "__main__":
    main()
