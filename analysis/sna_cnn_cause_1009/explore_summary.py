#!/usr/bin/env python3
"""Tables of the exploratory (unregistered) one-task forks of sna_cnn_cause_1009.

    python3 analysis/sna_cnn_cause_1009/explore_summary.py > results/sna_cnn_cause_1009/explore_summary.md

D<at>: all layers vs conv frozen; E<at>: all layers vs fc frozen (with per-layer moves);
Z10: alpha frozen during the task; T<at>: epoch-resolved traces.  Seed means (n = 10) and the
seed-paired difference with its SE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "sna_cnn_cause_1009"


def paired(d: pd.DataFrame, a: str, b: str, col: str = "online_acc") -> tuple[float, float, int]:
    x = d[d.fork == a].set_index("seed")[col]
    y = d[d.fork == b].set_index("seed")[col]
    diff = (x - y).dropna()
    return float(diff.mean()), float(diff.std() / np.sqrt(len(diff))), int((diff > 0).sum())


def fork_table(name: str, cols: list[str]) -> list[str]:
    f = ROOT / name / "per_task.csv"
    if not f.exists():
        return [f"({name}: not run)"]
    d = pd.read_csv(f)
    out = ["| fork | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for fk, g in d.groupby("fork", sort=False):
        out.append(f"| {fk} | {len(g)} | " + " | ".join(f"{g[c].mean():.4f}" if c == "online_acc" else f"{g[c].mean():.3f}"
                                                        for c in cols) + " |")
    return out


def main() -> None:
    L = ["# sna_cnn_cause_1009 — 登録外の探索（1 課題の fork）", "",
         "束 A のチェックポイント（seed 10–19）から 1 課題だけ続けた。online = その課題の online_acc。", ""]
    for at in (1, 10):
        L += [f"## 分解 D{at}: conv も学ぶか、conv を凍結するか（t{at} → t{at + 1}）", ""]
        L += fork_table(f"D{at}", ["online_acc", "ep_first99", "switch_ce"])
        d = pd.read_csv(ROOT / f"D{at}" / "per_task.csv") if (ROOT / f"D{at}" / "per_task.csv").exists() else None
        if d is not None:
            L += [""]
            for arm in ("SNA", "CV3FC06", "CV06FC3"):
                m, se, w = paired(d, f"{arm}>{arm}+fixconv", f"{arm}>{arm}")
                L.append(f"- {arm}: conv を凍結したときの得 = {m:+.4f} ± {se:.4f}（{w}/10 seed で正）")
        L += [""]
    for at in (1, 10):
        L += [f"## 分解 E{at}: fc を凍結して conv だけで学ぶ（t{at} → t{at + 1}）、各層の 1 課題の相対移動", ""]
        L += fork_table(f"E{at}", ["online_acc", "ep_first99", "move_c1", "move_c2", "move_f1", "move_f2", "move_f3"])
        L += [""]
    L += ["## Z10: 課題の中で α を固定（t10 → t11、沈んだ状態はそのまま）", ""]
    L += fork_table("Z10", ["online_acc", "ep_first99"])
    z = ROOT / "Z10" / "per_task.csv"
    if z.exists():
        d = pd.read_csv(z)
        L += [""]
        for arm in ("SNA", "CV3FC06", "CV06FC3"):
            m, se, w = paired(d, f"{arm}>{arm}@frz", f"{arm}>{arm}")
            L.append(f"- {arm}: α を固定したときの得 = {m:+.4f} ± {se:.4f}（{w}/10）")
    for at in (1, 10):
        f = ROOT / f"T{at}" / "trace.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        cols = ["acc", "ce", "gate_f1", "gate_f2", "seat_f2", "feat_drift", "move_c2", "move_f3"]
        L += ["", f"## T{at}: epoch ごとの記録（t{at} → t{at + 1}、probe 300 枚、seed 平均）", ""]
        for fk, g in d.groupby("fork", sort=False):
            m = g.groupby("epoch")[cols].mean()
            L += [f"### {fk}", "", "| epoch | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for e in (0, 1, 2, 3, 5, 10, 20, 50, 400):
                if e in m.index:
                    L.append(f"| {e} | " + " | ".join(f"{m.loc[e, c]:.3f}" for c in cols) + " |")
            L += [""]
    print("\n".join(L))


if __name__ == "__main__":
    main()
