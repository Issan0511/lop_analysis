#!/usr/bin/env python3
"""POST HOC.  Motion along the two exact symmetries of the network, at task ends and across the switch.
 (1) softmax translation: the class-mean row of W3 (w3bar) and b3bar move the logits' common mode, which neither the
     loss nor its gradient sees; plain SGD's gradient has zero class sum, so w3bar, b3bar are conserved exactly.
 (2) per-unit rescaling of a leaky unit i: (w_in_i, b_i) -> c (w_in_i, b_i), w_out_i -> w_out_i / c leaves the
     function unchanged.  Log gauge g_i = log|(w_in_i, b_i)| - log|w_out_i| (position on the orbit), log path gain
     p_i = log|(w_in_i, b_i)| + log|w_out_i|.  SGD conserves |w_in|^2 + b^2 - |w_out|^2 per unit to second order.
Outputs gauge_task_ends.csv, gauge_switch.csv, and prints tables (appended to symmetry.txt)."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
ARCH = Path.home() / "Projects/obsidian-research-data"
SP = ARCH / "sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923"; AL = ARCH / "altlabels_cifar_0923/results/altlabels_cifar_0923"
OUT = REPO / "results/logit_dyn_cifar_0923"
def load(p):
    d = np.load(p); return {k: d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
def unit_stats(P):
    win1 = np.sqrt((P["W1"] ** 2).sum(1) + P["b1"] ** 2); wout1 = np.sqrt((P["W2"] ** 2).sum(0))
    win2 = np.sqrt((P["W2"] ** 2).sum(1) + P["b2"] ** 2); wout2 = np.sqrt((P["W3"] ** 2).sum(0))
    w3bar = P["W3"].mean(0)
    return dict(g1=np.median(np.log(win1 / wout1)), p1=np.median(np.log(win1 * wout1)),
                g2=np.median(np.log(win2 / wout2)), p2=np.median(np.log(win2 * wout2)),
                C12=float((P["W1"] ** 2).sum() + (P["b1"] ** 2).sum() - (P["W2"] ** 2).sum()),
                C23=float((P["W2"] ** 2).sum() + (P["b2"] ** 2).sum() - (P["W3"] ** 2).sum()),
                n3=float((P["W3"] ** 2).sum()), n3_common=float(10 * (w3bar ** 2).sum()), b3bar=float(P["b3"].mean()),
                n1=float((P["W1"] ** 2).sum()), n2=float((P["W2"] ** 2).sum()))
rows = []
ARMS = {"A": SP / "A", "F": SP / "F", "S_hi": SP / "S_hi", "A_ce": SP / "A_ce", "SA_iid": SP / "SA_iid",
        "LR_abab": AL / "LR_abab", "SA_abab": SP / "SA_abab"}
for arm, d in ARMS.items():
    for s in range(10):
        for t in (0, 1, 5, 10, 25, 50):
            f = d / "snap" / f"LR_std_seed{s}" / f"t{t:02d}.npz"
            if f.exists():
                rows.append(dict(arm=arm, seed=s, t=t, **unit_stats(load(f))))
dT = pd.DataFrame(rows); dT.to_csv(OUT / "gauge_task_ends.csv", index=False)
rowsS = []
for t in (10, 20, 30, 40, 48):
    for s in range(10):
        sd = f"LR_std_seed{s}"
        st = {k: unit_stats(load(p)) for k, p in dict(sw=SP / f"xfork/t{t}_sw/snap/{sd}/t01.npz", end=SP / f"A/snap/{sd}/t{t:02d}.npz",
              nxt=SP / f"xfork/t{t}_Wend_Send/snap/{sd}/t01.npz", nxt0=SP / f"xfork/t{t}_Wsw_Ssw/snap/{sd}/t01.npz").items()}
        r = dict(t=t, seed=s)
        for q in ("g1", "p1", "g2", "p2", "C12", "C23", "n1", "n2", "n3", "n3_common"):
            r[f"push_d{q}"] = st["end"][q] - st["sw"][q]; r[f"ret_d{q}"] = st["nxt"][q] - st["end"][q]
            r[f"nopush_d{q}"] = st["nxt0"][q] - st["sw"][q]; r[f"keep_{q}"] = st["nxt"][q] - st["nxt0"][q]
        rowsS.append(r)
dS = pd.DataFrame(rowsS); dS.to_csv(OUT / "gauge_switch.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
lines = []
def Pr(x): lines.append(str(x)); print(x)
Pr("== task ends: median over seeds (g = log gauge, p = log path gain, per-unit medians; C = balance charges)")
tab = dT.groupby(["arm", "t"])[["g1", "p1", "g2", "p2", "C12", "C23", "n1", "n2", "n3", "n3_common", "b3bar"]].median()
Pr(tab.round(3).to_string())
Pr("\n== switch (50 cases, median): push = s_sw(t)->end(t), ret = end(t)->s_sw(t+1), nopush = s_sw(t)->s_sw(t+1) without push, keep = with - without")
cols = [c for c in dS.columns if c.split("_d")[-1] in ("g1", "p1", "g2", "p2", "C12", "n1", "n2", "n3", "n3_common") or c.startswith("keep_")]
Pr(dS[cols].median().round(4).to_string())
(OUT / "symmetry.txt").write_text("\n".join(lines) + "\n")
