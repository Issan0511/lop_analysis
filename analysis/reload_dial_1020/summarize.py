#!/usr/bin/env python3
"""reload_dial_1020 の表（summary.md）と図（dial.png）。rows.json・bound_30tasks.json を読む。"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path("results/reload_dial_1020")
rows = json.loads((OUT / "rows.json").read_text()); bound = json.loads((OUT / "bound_30tasks.json").read_text())
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]; NAME = {"LR": "leaky 0.1", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
CKS = ["init", "a20", "a50", "a80", "a95", "end"]
CKN = {"init": "初期値", "a20": "精度 20%", "a50": "精度 50%", "a80": "精度 80%", "a95": "精度 95%", "end": "課題 1 の末"}
SEEDS = [0, 1]

def r(act, s, ck):
    for x in rows:
        if x["act"] == act and x["seed"] == s and x["ck"] == ck: return x
def mean2(act, ck, f):
    v = [f(r(act, s, ck)) for s in SEEDS]; v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")
def rng2(act, ck, f):
    v = [f(r(act, s, ck)) for s in SEEDS]
    return min(v), max(v)
def P(x): return "—" if not math.isfinite(x) else f"{round(100 * x):d}%"
def G(x, n=3):
    if not math.isfinite(x): return "—"
    if x == 0: return "0"
    a = abs(x)
    if a >= 100: return f"{x:.0f}"
    if a >= 1: return f"{x:.2f}"
    if a >= 0.01: return f"{x:.3f}"
    return f"{x:.1e}"

L = ["# 装填のダイヤル reload_dial_1020（課題 1 の自信と切替の押し）\n",
     "1 隠れ層の乱数ラベル MNIST（784-100-10、Adam 1e-3・batch 16）。課題 1 の途中の 6 つの checkpoint から、新ラベル 5 通りへ切り替えた。"
     "値は seed 0・1 の平均。draw の値は 5 draw の平均をさらに 2 seed で平均した。全行は `rows.json`。\n"]

L.append("## 1 自信と κ と κ の下界\n")
L.append("| 活性化 | checkpoint | 更新 | 訓練精度 | 正解クラスの予測確率 | 損失の下がり幅 log K − L | κ | 下界 κ_lb | κ / κ_lb | 揃い A/N | ‖F‖ | ‖v^c‖ |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for act in ACTS:
    for ck in CKS:
        k = mean2(act, ck, lambda x: x["kappa"]); lb = mean2(act, ck, lambda x: x["kappa_lb"])
        ratio = mean2(act, ck, lambda x: x["kappa"] / x["kappa_lb"] if x["kappa_lb"] > 0 else float("nan"))
        L.append(f"| {NAME[act]} | {CKN[ck]} | {mean2(act, ck, lambda x: x['update']):.0f} | {mean2(act, ck, lambda x: x['acc']):.2f} | "
                 f"{mean2(act, ck, lambda x: x['conf_y']):.3f} | {G(mean2(act, ck, lambda x: x['gain']))} | {G(k)} | {G(lb)} | {G(ratio)} | "
                 f"{G(mean2(act, ck, lambda x: x['A'] / 1200))} | {G(mean2(act, ck, lambda x: x['F_norm']))} | {G(mean2(act, ck, lambda x: x['normV']))} |")
L.append("")

L.append("## 2 押しの大きさと向き\n")
L.append("期待の押しは新ラベルについて平均した全バッチの一歩（S/N、沈める側が正）。自己形は κ(1 − 1/K)Σ K_n φ′φ。実際の切替は draw ごとの値。生きた unit（開いた入力が 1 つ以上）で数えた。\n")
L.append("| 活性化 | checkpoint | 正解クラスの予測確率 | 期待の押し S/N（中央値） | 自己形（中央値） | 沈める側: 期待の押し | 自己形 | 実ラベル GD 一歩 | Adam 1 歩 | Adam 200 更新 | Adam 200 と期待の押しの符号一致 | 200 更新の沈み −Δm（中央値） |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for act in ACTS:
    for ck in CKS:
        d = lambda key: mean2(act, ck, lambda x: x["draws"][key][0])
        L.append(f"| {NAME[act]} | {CKN[ck]} | {mean2(act, ck, lambda x: x['conf_y']):.3f} | {G(mean2(act, ck, lambda x: x['S_exact_medN']))} | "
                 f"{G(mean2(act, ck, lambda x: x['S_self_medN']))} | {P(mean2(act, ck, lambda x: x['frac_sink_exact']))} | {P(mean2(act, ck, lambda x: x['frac_sink_self']))} | "
                 f"{P(d('frac_sink_gd'))} | {P(d('frac_sink_adam1'))} | {P(d('frac_sink_adam200'))} | {P(d('agree_exact_adam200'))} | {G(d('sink200_med'))} |")
L.append("")
L.append("draw の間のばらつき（Adam 200 更新で沈んだ unit の割合の標準偏差、5 draw、2 seed の大きい方）:\n")
L.append("| 活性化 | " + " | ".join(CKN[c] for c in CKS) + " |")
L.append("|---|" + "---|" * len(CKS))
for act in ACTS:
    L.append(f"| {NAME[act]} | " + " | ".join(P(max(r(act, s, ck)["draws"]["frac_sink_adam200"][1] for s in SEEDS)) for ck in CKS) + " |")
L.append("")

L.append("## 3 κ の下界の検査（checkpoint 60 点と、kappa_reload_1020 の 30 課題の課題末）\n")
allb = [x for x in rows] + bound
viol = [x for x in allb if x["kappa"] < x["kappa_lb"] - 1e-12]
worst = max(x["A_id_relerr"] for x in allb)
ratios = np.array([x["kappa"] / x["kappa_lb"] for x in bound if x["kappa_lb"] > 0])
neg_lb = sum(1 for x in bound if x["kappa_lb"] <= 0)
L.append(f"- 恒等式 A = Σ[(ℓ_y − ℓ̄) − b^c_y] の相対誤差の最大: {worst:.1e}（{len(allb)} 状態）")
L.append(f"- 下界 κ ≥ [N(log K − L) − B]/‖F‖² を破った状態: {len(viol)} / {len(allb)}")
L.append(f"- 30 課題の課題末（{len(bound)} 状態）で下界が正だった状態: {len(bound) - neg_lb}、κ/κ_lb の四分位: "
         + " / ".join(f"{q:.2f}" for q in np.percentile(ratios, [25, 50, 75])) if len(ratios) else "")
L.append("")
(OUT / "summary.md").write_text("\n".join(L) + "\n")
print("wrote", OUT / "summary.md")

# ---------- 図: 横軸 自信、上段 押しの大きさ（課題 1 の末 = 1）、下段 沈める側の割合
fam = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK KR", "Noto Sans CJK SC"):
    if cand in fam: plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "font.size": 9})
SER = [("期待の押し（新ラベルの平均）", "#2a78d6", lambda x: x["S_exact_medN"], lambda x: x["frac_sink_exact"]),
       ("自己形 κ(1−1/K)ΣKφ′φ", "#1baf7a", lambda x: x["S_self_medN"], lambda x: x["frac_sink_self"]),
       ("実際の Adam 200 更新", "#eb6834", lambda x: x["draws"]["sink200_med"][0], lambda x: x["draws"]["frac_sink_adam200"][0])]
fig, axes = plt.subplots(2, 5, figsize=(14, 6.2), sharex=True, constrained_layout=True)
for j, act in enumerate(ACTS):
    for name, col, fmag, ffrac in SER:
        for s, ls in zip(SEEDS, ["-", "--"]):
            xs = [r(act, s, ck)["conf_y"] for ck in CKS]
            end = fmag(r(act, s, "end"))
            mag = [fmag(r(act, s, ck)) / end if end else float("nan") for ck in CKS]
            frac = [ffrac(r(act, s, ck)) for ck in CKS]
            axes[0, j].plot(xs, mag, color=col, linestyle=ls, linewidth=1.8 if s == 0 else 1.2, marker="o", markersize=3.5,
                            label=name if (j == 0 and s == 0) else None)
            axes[1, j].plot(xs, frac, color=col, linestyle=ls, linewidth=1.8 if s == 0 else 1.2, marker="o", markersize=3.5)
    axes[0, j].set_title(NAME[act], fontsize=10, color="#0b0b0b", loc="left")
    axes[0, j].axhline(0, color="#8a8984", linewidth=0.8); axes[1, j].axhline(0.5, color="#8a8984", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1, j].set_ylim(0, 1.02); axes[1, j].set_xlabel("正解クラスの予測確率（課題 1 のラベル）")
axes[0, 0].set_ylabel("押しの大きさ（中央値、課題 1 の末 = 1）"); axes[1, 0].set_ylabel("沈める側の unit の割合")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels + [], loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("課題 1 を当てはめた自信と切替の押し（実線 seed 0、破線 seed 1。点は左から 初期値・精度 20/50/80/95%・課題 1 の末）",
             fontsize=10, color="#52514e", y=0.94)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "dial.png", dpi=150)
print("wrote", OUT / "dial.png")
