#!/usr/bin/env python3
"""chargpt_seed10_1022 C の開始点と窓（spec §5 の表と C3 の段の値）と、それを元の layer_stats.csv から規則で出し直す関数。
規則（spec §5 C1）: R_k = 課題 11〜100 の課題末で L1 の死（u=10000・probe=uniform・layer=1・frac_p_lt005）が初めて 0.5 以下になった課題。
S_k = 保存済みの課題末（1〜10・25・39〜50・89〜100）のうち R_k − 1 以下で、L1 の死が 0.9 以上の最も遅い課題。E_k = min(R_k + 9, 100)。
"""
import csv, os
from pathlib import Path

CG = Path(os.environ.get("CHARGPT_SRC", "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019"))
M2 = CG / "results/chargpt_cl_1019/M2"
SAVED = set(range(1, 11)) | {25} | set(range(39, 51)) | set(range(89, 101))

# spec §5 C1 の表（seed: (R_k, S_k, E_k)）。seed 3 は R_k 無しで対象外
C1_TABLE = {0: (55, 50, 64), 1: (50, 43, 59), 2: (84, 50, 93), 4: (56, 50, 65), 5: (37, 25, 46),
            6: (68, 50, 77), 7: (44, 41, 53), 8: (65, 50, 74), 9: (57, 50, 66)}
# spec §5 C3 の段（S_k・R_k は spec の値。E_k = min(R_k + 9, 100) は「同じ規則」から）
C3_S = [25, 25, 45, 40, 25, 41, 40, 50, 42, 25]
C3_R = [41, 59, 50, 49, 51, 50, 51, 60, 50, 49]
C3_TABLE = {k: (C3_R[k], C3_S[k], min(C3_R[k] + 9, 100)) for k in range(10)}
C2_FROM, C2_TO = 25, 31


def l1_death(arm, seed, layer=1):
    d = {}
    with open(M2 / arm / f"seed{seed}" / "layer_stats.csv") as f:
        for r in csv.DictReader(f):
            if r["u"] == "10000" and r["probe"] == "uniform" and int(r["layer"]) == layer:
                d[int(r["task"])] = float(r["frac_p_lt005"])
    return d


def rule(arm, seed):
    d = l1_death(arm, seed)
    R = next((t for t in range(11, 101) if d[t] <= 0.5), None)
    if R is None:
        return dict(R=None, S=None, E=None, min_death=min(d[t] for t in range(11, 101)))
    S = max((t for t in SAVED if t <= R - 1 and d[t] >= 0.9), default=None)
    E = min(R + 9, 100)
    frac = sum(d[t] <= 0.5 for t in range(R, min(R + 10, 101))) / len(range(R, min(R + 10, 101)))
    return dict(R=R, S=S, E=E, death_S=d.get(S), frac_le05_10=frac)


def jobs(part):
    """(seed, from_task, to_task, model_arm, arms) の list。"""
    if part == "C1":
        return [(k, S, E, "C3", ["control", "lnex1", "nocommon"]) for k, (R, S, E) in sorted(C1_TABLE.items())]
    if part == "C2":
        return [(k, C2_FROM, C2_TO, "C3", ["control", "gamma03f", "gamma03"]) for k in range(10)]
    if part == "C3":
        return [(k, S, E, "ReLUcap", ["control", "lnex1"]) for k, (R, S, E) in sorted(C3_TABLE.items())]
    raise ValueError(part)


SRC_ARM = {"C1": "C3", "C2": "C3", "C3": "ReLUcap"}
