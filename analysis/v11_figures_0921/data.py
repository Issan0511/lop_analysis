#!/usr/bin/env python3
"""V11 の図が引く committed CSV の入口。走は一切起こさない（図表一覧 §5）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def per_task(path: str | Path) -> pd.DataFrame:
    p = RES / path if not str(path).startswith("/") else Path(path)
    d = pd.read_csv(p)
    return d


def matrix(d: pd.DataFrame, col: str, seed_col="seed", task_col="task", tasks=None) -> tuple[np.ndarray, np.ndarray]:
    """(seed, task) の行列にそろえる。欠けている (seed, task) は NaN。"""
    t = np.asarray(sorted(d[task_col].unique()) if tasks is None else tasks)
    s = np.asarray(sorted(d[seed_col].unique()))
    piv = d.pivot_table(index=seed_col, columns=task_col, values=col, aggfunc="mean")
    piv = piv.reindex(index=s, columns=t)
    return t, piv.to_numpy(float)


# --- 図 2・3・4 の扉 ------------------------------------------------------
DOOR_PATHS = {
    "ref": "relu_doors_h_ref_0920/ref/per_task.csv",
    "H":   "relu_doors_h_ref_0920/H/per_task.csv",
    "C":   "relu_doors_0919/C/per_task.csv",
    "CH":  "relu_doors_0919/CH/per_task.csv",
    "CHB": "relu_doors_0919/CHB/per_task.csv",
    "CS":  "relu_doors_0919/CS/per_task.csv",
    "LN":  "relu_doors_0919/LN/per_task.csv",
}
DOOR_ORDER = ("ref", "H", "C", "CH", "CHB", "CS", "LN")


def doors() -> dict[str, pd.DataFrame]:
    return {a: per_task(p) for a, p in DOOR_PATHS.items()}


def ch_chb_200() -> dict[str, pd.DataFrame]:
    return {a: per_task(f"ch_chb_200_0919/{a}/per_task.csv") for a in ("CH", "CHB")}


# --- 図 4・9 の 5+1 ------------------------------------------------------
P5 = "cifar5p1_mlp_0920"
ARMS_16 = ("SNA", "KKA", "KKT1", "ELU", "LK03", "RSL", "KKA23", "SL",
           "LR", "SILU", "GELU", "LK001", "LK07", "R", "DF", "CR")


def arm5p1(folder: str) -> pd.DataFrame:
    return per_task(f"{P5}/{folder}/per_task.csv")


def fresh5p1(folder: str) -> pd.DataFrame:
    return per_task(f"{P5}/{folder}/fresh_control.csv")


def window5p1(d: pd.DataFrame, tasks) -> pd.Series:
    """登録窓: hard 課題だけの online の平均（seed ごと）。"""
    return d[d.task.isin(tasks)].groupby("seed").online_acc.mean()
