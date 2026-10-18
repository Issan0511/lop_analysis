#!/usr/bin/env python3
"""Post-hoc figures of cifar5p1_cnn_budget_1019 (descriptive; not part of the registered verdict).

    python3 analysis/cifar5p1_cnn_budget_1019/figures.py --root results/cifar5p1_cnn_budget_1019

fig_fc2_trajectory.png (and fig_aux_{c1,c2,f1}_trajectory.png for the auxiliary layers): the width, relative
position, phi', positive occupancy and low-response fraction on the fixed probe at every boundary (unit
medians), seed mean with the 95% seed-bootstrap band, per arm and budget.
fig_online.png: per hard task, the first-780-update and whole-task pre-update online accuracy (seed mean),
with the fresh controls at tasks 9, 19, 29.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
ARMS = ("R", "ELU", "KKT1", "SNAfrz1")
TITLE = {"R": "ReLU", "ELU": "ELU α=3.6", "KKT1": "KKT1", "SNAfrz1": "SNAfrz1"}
BUDGETS = (780, 2340, 7800)
COL = {780: "#9ecae1", 2340: "#4292c6", 7800: "#08306b"}
SITE_NAME = {"c1": "conv1", "c2": "conv2", "f1": "FC1", "f2": "FC2"}
ROWS = (("s_med", "{L} の幅 s（中央値）", True), ("r_med", "{L} の相対位置 r = m/s（中央値）", False),
        ("dmean_med", "{L} の φ′ 入力平均（中央値）", False), ("ppos_med", "{L} の正側占有率 p⁺（中央値）", False),
        ("lowresp_frac", "{L} の低応答の割合（φ′ < 0.1）", False))


def fig_traj(res, out: Path, site: str = "f2"):
    tr = res["trajectories"]
    L = SITE_NAME[site]
    fig, ax = plt.subplots(len(ROWS), len(ARMS), figsize=(15, 15), sharex=True)
    x = np.arange(31)
    for j, a in enumerate(ARMS):
        for i, (k, lab, logy) in enumerate(ROWS):
            lab = lab.format(L=L)
            A = ax[i, j]
            for b in BUDGETS:
                q = tr[f"{a}_{b}_{site}_{k}"]
                m, lo, hi = (np.asarray(q[z], dtype=float) for z in ("mean", "lo", "hi"))
                A.plot(x, m, color=COL[b], lw=1.6, label=f"{b} 更新／課題")
                A.fill_between(x, lo, hi, color=COL[b], alpha=0.2, lw=0)
            if logy:
                A.set_yscale("log")
            if j == 0:
                A.set_ylabel(lab, fontsize=9)
            if i == 0:
                A.set_title(TITLE[a])
            if i == len(ROWS) - 1:
                A.set_xlabel("境界（課題 t の末）")
            A.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"固定 probe 上の {L}（unit の中央値の seed 平均と 95% 区間）" + ("" if site == "f2" else "。補助の層"), y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def fig_online(root: Path, out: Path):
    fig, ax = plt.subplots(2, len(ARMS), figsize=(15, 7), sharex=True, sharey="row")
    hard = list(range(1, 31, 2))
    for b in BUDGETS:
        rows = read_csv(root / f"b{b}" / "per_task.csv")
        fresh = read_csv(root / f"b{b}" / "fresh_control.csv")
        for j, a in enumerate(ARMS):
            for i, k in enumerate(("first780_online_acc", "online_acc")):
                y = [np.mean([float(r[k]) for r in rows if r["arm"] == a and int(r["task"]) == t]) for t in hard]
                ax[i, j].plot(hard, 100 * np.asarray(y), "-o", ms=3, color=COL[b], label=f"{b} 継続")
                ft = [9, 19, 29]
                fy = [np.mean([float(r[f"fresh_{k}"]) for r in fresh if r["arm"] == a and int(r["task"]) == t]) for t in ft]
                ax[i, j].plot(ft, 100 * np.asarray(fy), "x", ms=8, mew=2, color=COL[b], label=f"{b} fresh")
    for j, a in enumerate(ARMS):
        ax[0, j].set_title(TITLE[a])
        ax[1, j].set_xlabel("hard 課題 t")
        for i in range(2):
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel("最初の 780 更新の online (%)")
    ax[1, 0].set_ylabel("課題全体の online (%)")
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("hard 課題の更新前 online 精度（seed 平均）。× は初期状態から同じ batch 列で学ばせた fresh", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    a = ap.parse_args()
    res = json.loads((a.root / "verdict.json").read_text())
    outs = {"f2": "fig_fc2_trajectory.png", "c1": "fig_aux_c1_trajectory.png", "c2": "fig_aux_c2_trajectory.png",
            "f1": "fig_aux_f1_trajectory.png"}
    for site, name in outs.items():
        fig_traj(res, a.root / name, site)
    fig_online(a.root, a.root / "fig_online.png")
    print("wrote", *outs.values(), "fig_online.png")


if __name__ == "__main__":
    main()
