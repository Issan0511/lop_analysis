#!/usr/bin/env python3
"""POST HOC.  Is layer 2 wild, and does layer 3 read only a small part of it?
z2 = W2 a1 + b2, a2 = leaky(z2, 0.1), logits z = W3 a2 + b3, recomputed in float64 from snapshots on each seed's 1,200 images.
Visible subspace of a2 = row space of the class-centered W3 (9 dims of 100; chance 9%).
Per consecutive task ends (t-1 -> t) and across the switch (crossover fork: push W_sw->W_end, refit W_end->W_next).
    python3 analysis/l2act_cifar_0923/l2act.py"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, torch
from src import rlcifar_mlp_battle_0918 as E
from src import pmnist_rlcifar_0907 as RC
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); DT = torch.float64
ARCH = Path.home() / "Projects/obsidian-research-data"
SP = ARCH / "sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923"; AL = ARCH / "altlabels_cifar_0923/results/altlabels_cifar_0923"
OUT = REPO / "results/l2act_cifar_0923"
cifar = RC.Cifar10(); X = {s: E.slot_inputs(cifar, s, "std", DEV).to(DT) for s in range(10)}
lk = lambda v: torch.nn.functional.leaky_relu(v, 0.1)
def load(p):
    d = np.load(p); return {k: torch.from_numpy(d[k]).to(DEV, DT) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
def fwd(P, s):
    a1 = lk(X[s] @ P["W1"].T + P["b1"]); z2 = a1 @ P["W2"].T + P["b2"]; a2 = lk(z2); z = a2 @ P["W3"].T + P["b3"]
    return z2, a2, z - z.mean(1, keepdim=True)
def vis_basis(W3):
    W3c = W3 - W3.mean(0, keepdim=True); U, S, Vt = torch.linalg.svd(W3c, full_matrices=False)
    return Vt[:9].T                                    # (100, 9)
nn = lambda a: float((a * a).sum()); ip = lambda a, b: float((a * b).sum())
def state(P, s):
    z2, a2, zc = fwd(P, s); zb = z2.mean(0, keepdim=True)
    return dict(z2=z2, a2=a2, zc=zc, W3=P["W3"], z2_rms=float(z2.pow(2).mean().sqrt()), frac_mean=nn(zb.expand_as(z2)) / nn(z2),
                neg_units=float((zb < 0).float().mean()), neg_entries=float((z2 < 0).float().mean()),
                a2_rms=float(a2.pow(2).mean().sqrt()), vis_a2=nn(a2 @ vis_basis(P["W3"])) / nn(a2),
                zc_rms=float(zc.pow(2).mean().sqrt()))
def change(S0, S1):
    da2 = S1["a2"] - S0["a2"]; Q1, Q0 = vis_basis(S1["W3"]), vis_basis(S0["W3"])
    dz_a = (da2 @ S1["W3"].T); dz_a = dz_a - dz_a.mean(1, keepdim=True)          # logit change from a2 moving
    dz_w = S0["a2"] @ (S1["W3"] - S0["W3"]).T; dz_w = dz_w - dz_w.mean(1, keepdim=True)  # from W3 moving
    dzc = S1["zc"] - S0["zc"]
    return dict(rel_a2=np.sqrt(nn(da2) / nn(S0["a2"])), rel_z2=np.sqrt(nn(S1["z2"] - S0["z2"]) / nn(S0["z2"])),
                rel_zc=np.sqrt(nn(dzc) / nn(S0["zc"])), vis_new=nn(da2 @ Q1) / nn(da2), vis_old=nn(da2 @ Q0) / nn(da2),
                dz_from_a2=np.sqrt(nn(dz_a)), dz_from_W3=np.sqrt(nn(dz_w)), dz_total=np.sqrt(nn(dzc)),
                cos_a2part_W3part=ip(dz_a, dz_w) / np.sqrt(nn(dz_a) * nn(dz_w)))
ARMS = {"A": SP / "A", "F": SP / "F", "S_hi": SP / "S_hi", "A_ce": SP / "A_ce", "SA_iid": SP / "SA_iid", "LR_abab": AL / "LR_abab", "SA_abab": SP / "SA_abab"}
rows = []
for arm, d in ARMS.items():
    for s in range(10):
        prev = None
        for t in range(1, 51):
            f = d / "snap" / f"LR_std_seed{s}" / f"t{t:02d}.npz"
            if not f.exists():
                prev = None; continue
            S1 = state(load(f), s)
            r = dict(arm=arm, seed=s, t=t, **{k: v for k, v in S1.items() if not isinstance(v, torch.Tensor)})
            if prev is not None:
                r.update(change(prev, S1))
            rows.append(r); prev = S1
dA = pd.DataFrame(rows); dA.to_csv(OUT / "task_ends.csv", index=False)
rowsB = []
for t in (10, 20, 30, 40, 48):
    for s in range(10):
        sd = f"LR_std_seed{s}"
        Ssw, Send = state(load(SP / f"xfork/t{t}_sw/snap/{sd}/t01.npz"), s), state(load(SP / f"A/snap/{sd}/t{t:02d}.npz"), s)
        Snx = state(load(SP / f"xfork/t{t}_Wend_Send/snap/{sd}/t01.npz"), s)
        r = dict(t=t, seed=s, z2_rms_sw=Ssw["z2_rms"], z2_rms_end=Send["z2_rms"], z2_rms_next=Snx["z2_rms"])
        r.update({f"push_{k}": v for k, v in change(Ssw, Send).items()}); r.update({f"ret_{k}": v for k, v in change(Send, Snx).items()})
        dp, dr = Send["a2"] - Ssw["a2"], Snx["a2"] - Send["a2"]
        r["undo_a2"] = -ip(dr, dp) / nn(dp)
        dp2, dr2 = Send["z2"] - Ssw["z2"], Snx["z2"] - Send["z2"]; r["undo_z2"] = -ip(dr2, dp2) / nn(dp2)
        rowsB.append(r)
dB = pd.DataFrame(rowsB); dB.to_csv(OUT / "switch.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
L = []
def P_(x): L.append(str(x)); print(x)
cols = ["z2_rms", "frac_mean", "neg_units", "neg_entries", "a2_rms", "vis_a2", "zc_rms", "rel_z2", "rel_a2", "rel_zc", "vis_new", "vis_old",
        "dz_from_a2", "dz_from_W3", "dz_total", "cos_a2part_W3part"]
order = ["A", "A_ce", "F", "S_hi", "SA_iid", "LR_abab", "SA_abab"]
for lo, hi in ((2, 5), (5, 14), (41, 50)):
    P_(f"\n== task ends t{lo}-{hi} (per-seed mean over tasks, median over seeds)")
    P_(dA[(dA.t >= lo) & (dA.t <= hi)].groupby(["arm", "seed"])[cols].mean().groupby("arm").median().loc[order].round(3).to_string())
P_("\n== switch, crossover fork (50 cases, median)")
P_(dB.drop(columns=["t", "seed"]).median().round(3).to_string())
(OUT / "summary.txt").write_text("\n".join(L) + "\n")
