#!/usr/bin/env python3
"""REPORT_ONLY / 事後（結果を見た後に書いた集計・登録判定ではない）: 上限腕で自由に残した経路の推移。

    python3 analysis/wcap_rlmnist_0914/posthoc_state.py

読むもの: results/wcap_rlmnist_0914/{per_task.csv, layer_metrics.csv}（本走 80 run の集計済みファイル）
書くもの: results/wcap_rlmnist_0914/posthoc_state/report.md
値はすべて 10 seed の中央値（seed ごとのタスク末の値の中央値）。
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results" / "wcap_rlmnist_0914"
pt = pd.read_csv(R / "per_task.csv")
lm = pd.read_csv(R / "layer_metrics.csv")
TASKS = (1, 2, 3, 5, 10, 20, 30, 50)
ARMS = [("LR", "ref"), ("LR", "l2"), ("LR", "capT1"), ("LR", "cap2"), ("R", "ref"), ("R", "l2"), ("R", "cap2")]
COLS = [("online_acc", "online_acc ×100", 100), ("mob_l1", "mob1", 1), ("zbar_l1", "z̄1（中央値）", 1),
        ("dead_frac_l1", "dead1", 1), ("rowmean_abs_med_l1", "median|m_i|（W1 行平均）", 1),
        ("b_med_l1", "median b1", 1), ("mob_l2", "mob2", 1), ("zbar_l2", "z̄2", 1)]
L = ["# posthoc_state — 上限腕で自由に残した経路の推移（REPORT_ONLY・事後・登録判定ではない）", "",
     "自動生成: `analysis/wcap_rlmnist_0914/posthoc_state.py`。10 seed の中央値。", ""]
for col, name, k in COLS:
    L += [f"## {name}", "", "| 腕 | " + " | ".join(f"t{t}" for t in TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for act, arm in ARMS:
        g = pt[(pt.act == act) & (pt.arm == arm)].groupby("task")[col].median()
        L.append(f"| {act}_{arm} | " + " | ".join(f"{k * g.loc[t]:.3f}" for t in TASKS) + " |")
    L.append("")
L += ["## W3 のテンソルノルム ‖W3‖（上限腕でも自由）", "", "| 腕 | t0 | " + " | ".join(f"t{t}" for t in TASKS) + " |",
      "|---|---|" + "---|" * len(TASKS)]
for act, arm in ARMS:
    g = lm[(lm.act == act) & (lm.arm == arm) & (lm.tensor == "W3")].groupby("task").norm.median()
    L.append(f"| {act}_{arm} | {g.loc[0]:.2f} | " + " | ".join(f"{g.loc[t]:.2f}" for t in TASKS) + " |")
L += ["", "## b1 のテンソルノルム ‖b1‖", "", "| 腕 | t0 | " + " | ".join(f"t{t}" for t in TASKS) + " |",
      "|---|---|" + "---|" * len(TASKS)]
for act, arm in ARMS:
    g = lm[(lm.act == act) & (lm.arm == arm) & (lm.tensor == "b1")].groupby("task").norm.median()
    L.append(f"| {act}_{arm} | {g.loc[0]:.2f} | " + " | ".join(f"{g.loc[t]:.2f}" for t in TASKS) + " |")
(R / "posthoc_state" / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
