#!/usr/bin/env python3
"""付録 A  MNIST の移植の鎖 — resp_ee・respdyn・swap・escneff・neffdir の要約（図表一覧 §4 A）。

第二の箱（RL-MNIST ELU→ELU・100-100）で登録した 5 本。本文の図 9 は背骨の S4 だけで、
この 5 本はラベルを文で書く（決定 10b・0922）。大きさは箱をまたがない（BOX_SPECIFIC）。
元データ: results/{resp_ee_0917,respdyn_ee_0917}/runs/s*/arms.csv・verdict.csv
          results/{swap_ee_0917,escneff_ee_0917,neffdir_ee_0918}/{paired.csv,verdict.csv}
格: 5 本とも登録。集約は各走の登録判定どおり seed 平均（区間は平均 ± sd 由来・§2.5-1）。
    帯・線は seed の全範囲（信頼区間ではない・§2.5-2）。
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

C_NAT, C_RES, C_SINK, C_FLOOR = "#7a7a7a", "#2f6b8f", "#c05a3c", "#8a5aa8"


def arms_per_seed(run: str, k: int = 1) -> pd.DataFrame:
    """runs/s*/arms.csv を束ねて (seed, arm) -> 次課題 online（継続 k）。"""
    rows = []
    for p in sorted((D.RES / run / "runs").glob("s*/arms.csv")):
        d = pd.read_csv(p)
        d = d[d.k == k]
        rows.append(d[["seed", "arm", "online_acc"]])
    return pd.concat(rows).pivot_table(index="seed", columns="arm", values="online_acc")


def paired(run: str, k: int = 1) -> pd.DataFrame:
    p = pd.read_csv(D.RES / run / "paired.csv")
    return p[p.k == k].pivot_table(index="seed", columns="quantity", values="value")


def dots(ax, i, vals, color, label=None):
    vals = np.asarray(vals, float)
    ax.scatter(np.full(vals.shape, i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=16,
               color=color, alpha=0.7, zorder=3, label=label)
    ax.hlines(vals.mean(), i - 0.22, i + 0.22, color=color, lw=2.4, zorder=4)


# ---------------------------------------------------------------- (a) resp_ee
def panel_resp_ee(ax):
    E = arms_per_seed("resp_ee_0917")
    v = pd.read_csv(D.RES / "resp_ee_0917" / "verdict.csv").set_index("quantity")
    T = [2, 5, 7, 10, 15, 20]
    nat = np.column_stack([E[f"N{t}"] for t in T])
    S.band(ax, T, nat, C_NAT, "自然な継続 $N_t$", agg="mean")
    Tr = [5, 7, 10, 15, 20]
    res = np.column_stack([E[f"R2_{t}"] for t in Tr])
    S.band(ax, Tr, res, C_RES, "復元: $t$ の網に $t_2$ の場 $R_{2,t}$", agg="mean")
    snk = np.column_stack([E[f"S2_{t}r"] for t in Tr])
    S.band(ax, Tr, snk, C_SINK, "沈降: $t_2$ の網に $t$ の場 $S_{2,t}$", agg="mean")
    ax.axhline(E["S2u30r"].mean(), color=C_FLOOR, lw=1.0, ls=":", label="応答の床（一様 −30）")
    S.acc_axis(ax, label=False)
    ax.set_ylabel("次課題のオンライン精度")
    ax.set_xlabel("分岐した課題 $t$（移植元・移植先）")
    ax.set_xticks(T)
    ax.set_title("(a) 場の移植 resp_ee: P1 復元 +%.2f・P2 沈降 +%.2f" % (v.loc["P1", "mean"], v.loc["P2", "mean"]))
    ax.legend(loc="center right", fontsize=7)
    S.grade(ax, "registered", "RESPONSE_BOTH_WAYS", loc="lower left")


# ---------------------------------------------------------------- (b) respdyn
def panel_respdyn(ax):
    E = arms_per_seed("respdyn_ee_0917")
    arms = [("N2r", "健康 $t_2$", C_NAT), ("S2_10r", "固定した $t_{10}$ の場", C_SINK),
            ("S2dyn_10r", "動く $t_{10}$ の場", C_SINK), ("S2u30r", "応答の床", C_FLOOR),
            ("N10r", "崩壊 $t_{10}$", C_NAT)]
    for i, (a, lab, col) in enumerate(arms):
        dots(ax, i, E[a], col)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([l for _, l, _ in arms], fontsize=7.5, rotation=18, ha="right")
    S.acc_axis(ax, label=False)
    ax.set_ylabel("次課題のオンライン精度")
    m = E.mean()
    rho_fix = (m["N2r"] - m["S2_10r"]) / (m["N2r"] - m["N10r"])
    rho_dyn = (m["N2r"] - m["S2dyn_10r"]) / (m["N2r"] - m["N10r"])
    ax.set_title("(b) 動く場 respdyn: 介入差の比 %.2f → %.2f、床の上に %.2f" % (rho_fix, rho_dyn, m["S2dyn_10r"] - m["S2u30r"]))
    S.grade(ax, "registered", "PARTIAL", loc="upper right")


# ---------------------------------------------------------------- (c) swap
def panel_swap(ax):
    P = paired("swap_ee_0917")
    q = [("TE", "cap12 − ref\n（救命の総量）", "#2f4858"),
         ("L_S", "cap12 の網に\nref の場（失う）", C_SINK), ("L_S_fix", "同・固定した場", C_SINK),
         ("G_R", "ref の網に\ncap12 の場（得る）", C_RES), ("G_R_fix", "同・固定した場", C_RES)]
    for i, (k, lab, col) in enumerate(q):
        dots(ax, i, P[k], col)
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_xticks(range(len(q)))
    ax.set_xticklabels([l for _, l, _ in q], fontsize=7)
    ax.set_ylabel("次課題の online の差（seed 内）")
    m = P.mean()
    ax.set_title("(c) 場の入れ替え swap: 失う %.2f・得る %.2f（総量 %.2f 比）" % (m["L_S"] / m["TE"], m["G_R"] / m["TE"], 1.0))
    S.breathe(ax)
    S.grade(ax, "registered", "BOTH_WAYS", loc="lower right")


# ---------------------------------------------------------------- (d) escneff
def panel_escneff(ax):
    P = paired("escneff_ee_0917")
    q = [("R_none", "止めない"), ("R_c1", "第 1 層の\n成長を止める"), ("R_c2", "第 2 層"),
         ("R_c12", "両層"), ("R_c12b", "両層＋bias"), ("R_frz", "凍結")]
    cols = S.ladder(len(q), "plasma", 0.10, 0.85)
    for i, (k, lab) in enumerate(q):
        dots(ax, i, P[k], cols[i])
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_xticks(range(len(q)))
    ax.set_xticklabels([l for _, l in q], fontsize=7)
    ax.set_ylabel("応答の床からの残り $R$")
    r = np.corrcoef(P["dneff"], P["dR"])[0, 1]
    ax.set_title("(d) 逃げを塞ぐ escneff: 残り %.3f → %.3f、$r(\\Delta\\bar n_{\\rm eff},\\Delta R)$ = %+.2f" % (P["R_none"].mean(), P["R_c12"].mean(), r))
    S.breathe(ax)
    S.grade(ax, "registered", "NEFF_TRACKS（seed 10–19）", loc="upper right")


# ---------------------------------------------------------------- (e) neffdir
def panel_neffdir(ax):
    P = paired("neffdir_ee_0918")
    q = [("add_h", "担い手を足す\n（h 側）", True), ("add", "足す", True),
         ("drop_h", "落とす（h 側）", False), ("drop", "落とす", False), ("m25", "上位 25% を落とす", False)]
    for i, (k, lab, moved) in enumerate(q):
        col = C_RES if moved else C_NAT
        dots(ax, i, P[f"dR_{k}"], col)
        dn = P[f"dn_{k}"].mean()
        ax.annotate("$\\Delta\\bar n_{\\rm eff}$ %+.4f%s" % (dn, "" if moved else "（動かず）"),
                    xy=(i, P[f"dR_{k}"].max()), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.5, color="#555555")
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_xticks(range(len(q)))
    ax.set_xticklabels([l for _, l, _ in q], fontsize=7)
    ax.set_ylabel("学習能力の差 $\\Delta E$（seed 内）")
    ax.set_title("(e) 担い手を直接動かす neffdir: 足すと上がる %+.2f" % P["dR_add_h"].mean())
    S.breathe(ax, frac=0.12)
    S.grade(ax, "registered", "ADD_H_MOVES_E・落とす側は NOT_MANIPULATED", loc="lower right")


def panel_text(ax):
    ax.axis("off")
    lines = [
        "第二の箱: 乱数ラベル MNIST・784-100-100-10・両層 ELU・Adam 1e−3",
        "鎖の 5 本（すべて事前登録・10 seed）",
        "  resp_ee   固定した場を両方向に移植  RESPONSE_BOTH_WAYS",
        "  respdyn   動く場の移植             PARTIAL（介入差の報告比 0.47 → 0.72・残り 0.11）",
        "  swap      cap12 と ref の場を交換  BOTH_WAYS（対照差の 0.96 / 1.06 倍）",
        "  escneff   成長を止めて残りを測る    NEFF_TRACKS（未使用 seed 10–19）",
        "  neffdir   担い手 n̄_eff を直接動かす ADD_H_MOVES_E（落とす側は補償される）",
        "",
        "大きさは箱をまたがない（BOX_SPECIFIC）: 復元／沈降の非対称は",
        "  MNIST +76 / +26 pt、CIFAR の S4 は +47 / +68 pt で向きが逆。並べて比べない。",
        "本文の図 9 は背骨の S4 だけ（決定 10b）。",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=7.6,
            transform=ax.transAxes, linespacing=1.45)


def build():
    fig, axes = S.grid(2, 3, 13.0, 8.6)
    panel_resp_ee(axes[0]); panel_respdyn(axes[1]); panel_swap(axes[2])
    panel_escneff(axes[3]); panel_neffdir(axes[4]); panel_text(axes[5])
    fig.suptitle("付録 A  MNIST の移植の鎖 — RL-MNIST / ELU→ELU・10 seed", fontsize=12)
    return fig


NOTE = """付録 A. MNIST の移植の鎖（第二の箱・RL-MNIST ELU→ELU・10 seed）。5 本とも事前登録。
点は seed 1 つずつ、横線は seed 平均（各走の登録判定が平均と sd の区間なので平均を描く）。
(a) resp_ee: 帯は seed の全範囲（信頼区間ではない）。自然な継続 N_t、t の網に t2 の場を戻す
復元 R_{2,t}、t2 の網に t の場を渡す沈降 S_{2,t}。P1 = R_{2,10} − N_10 = +0.76、
P2 = N_2r − S_{2,10}r = +0.26（両比較の平均の 97.5% 区間の下端が正、対応差も各 10/10 で正）。登録判定 RESPONSE_BOTH_WAYS。
(b) respdyn: 移植元の場を移植元の網の継続に追随させると、固定した場より深く落ちる（0.46 → 0.32）。
人工介入差を自然対照差で割った報告の比は固定場 0.47・動く場 0.72。自然な喪失の厳密な媒介率ではない。応答の床（一様 −30）の上に 0.11 残る。PARTIAL。
(c) swap: cap12（両層の上限で救われた網）と ref の第 2 層の場を入れ替える。失う量 L_S と得る量 G_R
は自然対照差 TE の 0.96 倍と 1.06 倍（人工介入の効果の報告比）。BOTH_WAYS。
(d) escneff: t2 の網へ t10 の低応答の動く場を移植した状態で成長を止めると（c1・c2・c12・c12b）、応答の床からの残り R が減る。
減った量は n̄_eff の減りと seed 間で r = +0.92。未使用 seed 10–19 で登録。NEFF_TRACKS。
(e) neffdir: 課題全体では担い手 n̄_eff を足す介入だけが登録方向へ n̄_eff を動かし、学習能力が +0.13 上がる（比例予測を上回る）。
落とす介入は網が補償して n̄_eff が動かず、NOT_MANIPULATED。ADD_H_MOVES_E。
n̄_eff 単独の必要十分性・比例則は示していない。大きさは箱をまたがない（neff_pred BOX_SPECIFIC）。本文の図 9 は S4 だけ（決定 10b・0922）。
元データ: results/{resp_ee_0917,respdyn_ee_0917}/runs/s*/arms.csv、
results/{swap_ee_0917,escneff_ee_0917,neffdir_ee_0918}/paired.csv。
"""

if __name__ == "__main__":
    print(S.save(build(), "figA_mnist_chain", NOTE))
