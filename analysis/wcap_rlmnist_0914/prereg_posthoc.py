#!/usr/bin/env python3
"""REPORT_ONLY / 事後 — spec_wcap_rlmnist_0914 §1 の動機になった既存データの再集計（登録判定ではない）。

    python3 analysis/wcap_rlmnist_0914/prereg_posthoc.py

読むもの（どちらも committed）:
  results/shell_l2_rlmnist_0913/per_task.csv        （CPU・R/SNA × none/l2/l2init/shell × seed 0–9）
  results/pmnist_rlmnist_0906/{LR,LR_l2,R_l2,R_l2init,SNA}/per_task.csv  （CUDA・参考）
書くもの: results/wcap_rlmnist_0914/prereg_posthoc/{per_seed.csv, report.md}

指標（seed ごとに計算してから seed 間で要約。bootstrap は seed の復元抽出 10,000 回・rng 0）:
  A(t)   = online_acc（タスク t の 30,000 更新前バッチ精度の平均）
  G      = A(1) − A(31–50)          fresh gap（task 1 は同じ手法を init から学ぶ走そのもの）
  D      = A(2–6) − A(31–50)        系列内の劣化（task 1 の過渡を含めない）
  P      = A(11–20) − A(41–50)      進行性の成分
  slope  = タスク 11–50 の OLS 傾き（pt / 10 タスク）
w_norm_l1 は宿主 evaluate の「非中心化」行ノルムの中央値であって、中心化ノルム ‖W̃_i‖ ではない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "wcap_rlmnist_0914" / "prereg_posthoc"
N_BOOT, RNG = 10_000, 0


def per_seed(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in d.groupby("seed"):
        A = g.set_index("task").online_acc.sort_index()
        assert list(A.index) == list(range(1, 51)), (s, len(A))
        h = A.loc[11:50]
        rows.append(dict(seed=s, A1=A[1], A2_6=A.loc[2:6].mean(), A31_50=A.loc[31:50].mean(),
                         A11_20=A.loc[11:20].mean(), A41_50=A.loc[41:50].mean(),
                         G=A[1] - A.loc[31:50].mean(), D=A.loc[2:6].mean() - A.loc[31:50].mean(),
                         P=A.loc[11:20].mean() - A.loc[41:50].mean(),
                         slope=np.polyfit(h.index, h.values, 1)[0] * 10,
                         wn1_t1=g[g.task == 1].w_norm_l1.item(), wn1_t50=g[g.task == 50].w_norm_l1.item(),
                         mob1_t1=g[g.task == 1].mob_l1.item(), mob1_t50=g[g.task == 50].mob_l1.item(),
                         dead1_t1=g[g.task == 1].dead_frac_l1.item(), dead1_t50=g[g.task == 50].dead_frac_l1.item()))
    return pd.DataFrame(rows)


def ci(x, rng) -> str:
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return f"{100 * x.mean():+.2f} [{100 * np.percentile(m, 2.5):+.2f}, {100 * np.percentile(m, 97.5):+.2f}]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)
    arms = {}
    sh = pd.read_csv(REPO / "results" / "shell_l2_rlmnist_0913" / "per_task.csv")
    for (act, reg), g in sh.groupby(["act", "reg"]):
        arms[f"{act}_{reg} (0913 CPU)"] = per_seed(g)
    for arm in ("LR", "LR_l2", "R_l2", "R_l2init", "SNA"):
        arms[f"{arm} (0906 CUDA)"] = per_seed(pd.read_csv(REPO / "results" / "pmnist_rlmnist_0906" / arm / "per_task.csv"))
    pd.concat([v.assign(arm=k) for k, v in arms.items()]).to_csv(OUT / "per_seed.csv", index=False)

    L = ["# prereg_posthoc — 水準と時間劣化の分離（REPORT_ONLY・事後・登録判定ではない）", "",
         "自動生成: `analysis/wcap_rlmnist_0914/prereg_posthoc.py`。単位は pt、[ ] は seed bootstrap 95% CI。", "",
         "| 腕 | n | A(1) | A(31–50) | G = A(1)−A(31–50) | G>0 の seed | D = A(2–6)−A(31–50) | P = A(11–20)−A(41–50) | 傾き 11–50 /10 タスク | ‖W1_i‖ 中央値（非中心化）t1→t50 | mob1 t1→t50 | dead1 t1→t50 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, q in arms.items():
        L.append(f"| {k} | {len(q)} | {100 * q.A1.mean():.2f} | {100 * q.A31_50.mean():.2f} | {ci(q.G, rng)} | "
                 f"{int((q.G > 0).sum())}/{len(q)} | {ci(q.D, rng)} | {ci(q.P, rng)} | {ci(q.slope, rng)} | "
                 f"{q.wn1_t1.median():.2f} → {q.wn1_t50.median():.2f} | {q.mob1_t1.median():.3f} → {q.mob1_t50.median():.3f} | "
                 f"{q.dead1_t1.median():.3f} → {q.dead1_t50.median():.3f} |")
    L += ["", "## 0913 R: l2init − l2 と shell − l2init の窓 31–50 の差を fresh 分と時間分に分ける（seed 対応）", "",
          "| 対比 | 窓 31–50 の差 | fresh 分（A(1) の差） | 時間分（G の差・符号は窓の差に足し合わせる向き） |", "|---|---|---|---|"]
    for act in ("R", "SNA"):
        a, b, c = (arms[f"{act}_{r} (0913 CPU)"].set_index("seed") for r in ("l2init", "l2", "shell"))
        L.append(f"| {act}: l2init − l2 | {ci(a.A31_50 - b.A31_50, rng)} | {ci(a.A1 - b.A1, rng)} | {ci(b.G - a.G, rng)} |")
        L.append(f"| {act}: shell − l2init | {ci(c.A31_50 - a.A31_50, rng)} | {ci(c.A1 - a.A1, rng)} | {ci(a.G - c.G, rng)} |")
    ref = arms["LR (0906 CUDA)"]
    L += ["", f"LR（0906）: mean P / mean G = {ref.P.mean() / ref.G.mean():.4f}（spec §5 の帯 0.1 の算術根拠）。", ""]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
