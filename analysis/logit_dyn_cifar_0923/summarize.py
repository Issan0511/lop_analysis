#!/usr/bin/env python3
"""Tables for logit_dyn_cifar_0923 from task_ends.csv, switch.csv, abab_roundtrip.csv -> summary.txt"""
from pathlib import Path
import pandas as pd, numpy as np
OUT = Path(__file__).resolve().parents[2] / "results/logit_dyn_cifar_0923"
A = pd.read_csv(OUT / "task_ends.csv"); B = pd.read_csv(OUT / "switch.csv"); C = pd.read_csv(OUT / "abab_roundtrip.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); lines.append(s); print(s)
def block(df, lo, hi, cols):
    b = df[(df.t >= lo) & (df.t <= hi)].groupby(["arm", "seed"])[cols].mean().groupby("arm").median()
    return b
cols = ["spread", "zy_c", "m_med", "m_p05", "m_min", "M_mean", "ce", "shift", "n1", "n3", "rank2", "old_c", "acc_old", "correct"]
order = ["A", "A_ce", "F", "S_hi", "SA_iid", "LR_abab", "SA_abab"]
for lo, hi in ((1, 5), (5, 14), (41, 50)):
    P(f"\n== task ends t{lo}-{hi}: per-seed mean over tasks, median over seeds")
    P(block(A, lo, hi, cols).loc[order].round(4).to_string())
e, l = block(A, 5, 14, ["spread", "n1", "m_med"]), block(A, 41, 50, ["spread", "n1", "m_med"])
P("\n== ratio t41-50 / t5-14: spread, |W1| (sqrt n1), median margin")
P(pd.DataFrame({"spread": l.spread / e.spread, "normW1": np.sqrt(l.n1 / e.n1), "m_med": l.m_med / e.m_med}).loc[order].round(3).to_string())
P("\n== rows with correct < 1199 at a task end (arm: count / 500)")
P(A[A.correct < 1199].groupby("arm").size().to_string())
P("\n== switch (crossover fork, 50 cases): medians, and by fork task")
keys = ["push_cos", "push_scale", "push_resid_frac", "undo_logit", "cos_logit", "undo_W1", "keep_logit", "keep_logit_proj",
        "sw_t_m_med", "end_t_m_med", "sw_t_spread", "end_t_spread", "sw_t_M_mean", "end_t_M_mean", "sw_t_ce", "end_t_ce",
        "sw_new_M_mean", "end_new_M_mean", "sw_new_ce", "end_new_ce", "sw_new_m_med", "end_new_m_med", "sw_new_rank2", "end_new_rank2",
        "next_m_med", "next0_m_med", "next_spread", "next0_spread", "next_M_mean", "next0_M_mean", "next_rank2", "next0_rank2",
        "next_old_c", "next0_old_c", "next_acc_old", "next0_acc_old", "next_correct", "next0_correct"]
m = B.groupby("t")[keys].median().T; m["all"] = B[keys].median()
P(m.round(4).to_string())
P("\n== ABAB round trips (cycles 14-24): median")
P(C.groupby("arm")[["cos_logit", "cos_W1", "close_logit", "close_W1"]].median().round(3).to_string())
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
