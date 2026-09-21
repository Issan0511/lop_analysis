#!/usr/bin/env python3
"""図 7 活性化 — 定義・13 腕のバトル・先行層（図表一覧 §2）。

(a) Snake・KKA・KKT1 の phi と phi'（定義図・alpha=1 に固定して形だけを見せる）
(b) 13 腕 × raw/std の窓（点と seed 範囲）
(c) 先行層: ELU・GELU・SILU の T* 分布（どちらの層が先に落ちるか）
元データ: results/rlcifar_mlp_battle_0918/<arm>/per_task.csv・results/cifar_ledger_0920/per_seed.csv
格: 登録（窓は tasks 31-50・符号検定）。集約は seed 中央値（§2.5-1）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import style as S

PI = np.pi
BATTLE = ("SNA", "KKA", "KKA23", "KKT1", "RSL", "LK03", "SL", "LR", "LK001", "GELU", "SILU", "ELU", "R")
WIN = (31, 50)


def phi(z, kind, a=1.0):
    snake = z + np.sin(a * z) ** 2 / a
    th = 2 * a * z
    if kind == "snake":
        return snake
    if kind == "kk":
        left = 2 * z + (0.75 * PI) / a + 0.5 / a
        right = 2 * z - (0.25 * PI) / a + 0.5 / a
        return np.where(th < -1.5 * PI, left, np.where(th > 0.5 * PI, right, snake))
    return np.where(th < -2 * PI, z, np.where(th > PI, z + 1 / a, snake))   # kkt1


def dphi(z, kind, a=1.0):
    th = 2 * a * z
    s = 1 + np.sin(th)
    if kind == "snake":
        return s
    if kind == "kk":
        return np.where((th >= -1.5 * PI) & (th <= 0.5 * PI), s, 2.0)
    return np.where((th >= -2 * PI) & (th <= PI), s, 1.0)


def battle_windows() -> pd.DataFrame:
    rows = []
    for arm in BATTLE:
        d = D.per_task(f"rlcifar_mlp_battle_0918/{arm}/per_task.csv")
        w = d[d.task.between(*WIN)]
        for (cond, seed), g in w.groupby(["cond", "seed"]):
            rows.append({"arm": arm, "cond": cond, "seed": seed, "window": g.online_acc.mean()})
    return pd.DataFrame(rows)


def build():
    fig = plt.figure(figsize=(13.0, 7.0), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.55, 1.0])
    ax_phi = fig.add_subplot(gs[0, 0])
    ax_dphi = fig.add_subplot(gs[1, 0])
    ax_win = fig.add_subplot(gs[:, 1])
    ax_t = fig.add_subplot(gs[:, 2])

    # (a) 定義図
    z = np.linspace(-6, 4, 1600)
    for kind, key, lab in (("snake", "SNA", "Snake"), ("kk", "KKA", "KKA"), ("kkt1", "KKT1", "KKT1")):
        ax_phi.plot(z, phi(z, kind), color=S.COLOR[key], lw=1.8, label=lab)
        ax_dphi.plot(z, dphi(z, kind), color=S.COLOR[key], lw=1.8)
    ax_phi.plot(z, np.maximum(z, 0), color=S.GREY, lw=1.0, ls="--", label="ReLU")
    ax_dphi.plot(z, (z > 0).astype(float), color=S.GREY, lw=1.0, ls="--")
    ax_phi.set_title("(a) $\\varphi$（$\\alpha=1$ に固定）")
    ax_dphi.set_title("(a) $\\varphi'$")
    ax_phi.legend(loc="upper left")
    for ax in (ax_phi, ax_dphi):
        ax.set_xlabel("$z$")
        ax.axhline(0, color="#bbbbbb", lw=0.7)
        ax.axvline(0, color="#bbbbbb", lw=0.7)
    S.grade(ax_phi, "column", "定義", loc="lower right")

    # (b) 13 腕 × raw/std
    w = battle_windows()
    order = w[w.cond == "std"].groupby("arm").window.median().sort_values().index.tolist()
    y = np.arange(len(order))
    for cond, dx, mk in (("raw", -0.17, "o"), ("std", 0.17, "s")):
        sub = w[w.cond == cond]
        med = sub.groupby("arm").window.median()
        lo, hi = sub.groupby("arm").window.min(), sub.groupby("arm").window.max()
        cols = [S.COLOR[a] for a in order]
        ax_win.hlines(y + dx, lo[order], hi[order], color=cols, lw=3, alpha=0.30)
        ax_win.scatter(med[order], y + dx, color=cols, s=34, marker=mk, zorder=3,
                       edgecolor="white", linewidth=0.5)
        # 8 腕が 0.85-0.99 に詰まって順位が読めないので、数値を添える（軸は 0-1 のまま）
        S.value_labels(ax_win, y + dx, med[order].to_numpy(), "{:.3f}", flip_at=0.55)
    ax_win.axvline(0.5, color="#999999", lw=0.9, ls=":")
    ax_win.annotate("崩壊の線 0.5", xy=(0.5, len(order) - 0.6), fontsize=7, color="#666666", ha="left")
    ax_win.set_yticks(y); ax_win.set_yticklabels(order)
    ax_win.set_xlim(0, 1.02); ax_win.set_xlabel("後期窓のオンライン精度（t31–50）")
    S.breathe(ax_win, "y", 0.03)
    ax_win.set_title("(b) 13 腕 × raw / std の窓")
    ax_win.grid(axis="y", alpha=0)
    ax_win.legend(handles=[plt.Line2D([], [], marker=m, ls="", color="#555555", label=c)
                           for c, m in (("raw", "o"), ("std", "s"))], loc="lower right")
    S.grade(ax_win, "registered", "A = SNA_BEATEN（両 cond）", loc="upper right")

    # (c) 先行層 — T* は 1 か 2 しか取らず、先に落ちる層は cond で決まり切っている。
    # 分布として描くと空になるので、腕 x cond の表として描く。
    ps = pd.read_csv(D.RES / "cifar_ledger_0920" / "per_seed.csv")
    arms = ["ELU", "GELU", "SILU", "R"]
    conds = ["raw", "std"]
    face = {"L1_FIRST": "#dce7f2", "L2_FIRST": "#f5e2d8", "SIMULTANEOUS": "#e9e9ea"}
    name = {"L1_FIRST": "第 1 層が先", "L2_FIRST": "第 2 層が先", "SIMULTANEOUS": "同時"}
    for i, arm in enumerate(arms):
        for j, cond in enumerate(conds):
            sub = ps[(ps.arm == arm) & (ps.cond == cond)]
            if len(sub) == 0:
                continue
            lab = sub.layer.mode().iloc[0]
            n = int((sub.layer == lab).sum())
            ts = sub.T_star.dropna()
            ax_t.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84,
                                         fc=face[lab], ec=S.COLOR[arm], lw=1.4, zorder=1))
            ax_t.text(j, i - 0.17, f"{name[lab]}  {n}/{len(sub)}", ha="center", va="center", fontsize=8.5)
            tstar = "—" if ts.empty else f"{int(ts.median())}"
            ax_t.text(j, i + 0.15, f"$T^*$ = {tstar}", ha="center", va="center", fontsize=8, color="#555555")
    ax_t.set_xlim(-0.6, 1.6); ax_t.set_ylim(-0.7, len(arms) - 0.3)
    ax_t.set_xticks(range(len(conds))); ax_t.set_xticklabels(conds)
    ax_t.set_yticks(range(len(arms))); ax_t.set_yticklabels(arms)
    ax_t.invert_yaxis()
    ax_t.grid(False)
    ax_t.set_title("(c) 先に落ちる層と崩壊の課題")
    S.grade(ax_t, "column", loc="lower right")

    fig.suptitle("図 7  活性化 — RL-CIFAR / MLP・50 課題・10 seed", fontsize=12)
    return fig


NOTE = """図 7. 活性化。(a) は定義図で、実装の alpha は unit ごとに c/W で動くが、ここでは形を見せるため
alpha=1 に固定している（測定値ではない）。
(b) 点は seed 中央値、線は seed の全範囲（信頼区間ではない）。窓は登録どおり t31-50、丸が raw、
四角が std。0.5 は登録の崩壊の線。登録判定は両 cond とも A = SNA_BEATEN。
腕が床（0.10-0.12）と上（0.85-0.99）の二つに分かれて位置から順位が読めないので、軸は 0-1 に
固定したまま（規約 §2.5-3）数値を点のそばに添えた。
(c) は cifar_ledger_0920 の per_seed。T* は 1 か 2 しか取らず、先に落ちる層は cond で決まり切って
いる（raw は第 1 層・std は第 2 層が 10/10・R は同時）ので、分布ではなく腕 x cond の表として描く。
元データ: results/rlcifar_mlp_battle_0918/<arm>/per_task.csv・results/cifar_ledger_0920/per_seed.csv。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig07_activation", NOTE))
