#!/usr/bin/env python3
"""POST HOC.  How do the pre-softmax values of layer 3 (logits z = W3 a2 + b3) change?  Recomputed in float64
from the saved snapshots and each seed's 1,200 standardized images (the engine's slot_inputs / labels stream).

    python3 analysis/logit_dyn_cifar_0923/logits.py

Parts (all W, b from snapshots; zc = z - mean_k z; spread s = mean_i std_k z; margin m = z_y - max_{k!=y} z_k;
M = sum_k p_k (z_y - z_k); CE in float64):
  A  task ends, arms A (= LR_iid), F, S_hi, A_ce, SA_iid (iid labels) and SA_abab, LR_abab (abab labels.npz):
     spread, margins, M, CE, layer norms, and the previous labeling's residue in the logits (y_old = task t-1's
     labels; rank2 = share of images with y_old != y whose y_old logit is the largest after y; old_c = mean over those
     images of z_{y_old} minus the mean of the other non-current classes).
  B  the switch (crossover-fork checkpoints of LR_iid, t in {10,20,30,40,48}, seeds 0-9): W_sw(t), W_end(t),
     W_next = task t+1 at its s_sw from W_end, W_next0 = the same from W_sw (no push).  The push in logit space
     (cos / scale of zc_end vs zc_sw), the next task's start (M, CE, margin on y_{t+1}), the refit, and the logit-space
     undo fraction -<dz_ret, dz_push>/|dz_push|^2 next to the weight-space one.
  C  ABAB round trips in logit space vs weight space (LR_abab Adam, SA_abab SGD, cycles 14-24).
Checks: the regenerated labels are fitted (accuracy at task ends >= 0.999), and the recomputed median margin equals
the run's trace (last row of each task) to float32 round-off; a wrong-task label control must fail.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, torch
from src import rlcifar_mlp_battle_0918 as E
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); DT = torch.float64
ARCH = Path.home() / "Projects/obsidian-research-data"
SP = ARCH / "sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923"
AL = ARCH / "altlabels_cifar_0923/results/altlabels_cifar_0923"
OUT = REPO / "results/logit_dyn_cifar_0923"
SEEDS = range(10)
cifar = RC.Cifar10()
X = {s: E.slot_inputs(cifar, s, "std", DEV).to(DT) for s in SEEDS}
YI = {}
for s in SEEDS:                                   # iid labels, task t -> YI[s][t-1]
    g = H.stream("rlc_labels", s); YI[s] = [RC.task_labels(g).to(DEV) for _ in range(51)]

def load(p):
    d = np.load(p); return [torch.from_numpy(d[k]).to(DEV, DT) for k in ("W1", "b1", "W2", "b2", "W3", "b3")]

def logits(P, s):
    W1, b1, W2, b2, W3, b3 = P
    a1 = torch.nn.functional.leaky_relu(X[s] @ W1.T + b1, 0.1)
    a2 = torch.nn.functional.leaky_relu(a1 @ W2.T + b2, 0.1)
    return a2 @ W3.T + b3

def metrics(z, y, y_old=None):
    zc = z - z.mean(1, keepdim=True)
    zy = z.gather(1, y[:, None])
    oth = z.clone(); oth.scatter_(1, y[:, None], float("-inf"))
    marg = (zy[:, 0] - oth.amax(1))
    d = z - zy                                       # z_k - z_y
    p = torch.softmax(z, 1)
    M = (p * (-d)).sum(1)
    ce = torch.logsumexp(d, 1)                       # = CE, exact in float64
    out = dict(correct=int((z.argmax(1) == y).sum()), spread=float(z.std(1, unbiased=False).mean()),
               zy_c=float(zc.gather(1, y[:, None]).mean()), m_med=float(marg.median()), m_p05=float(marg.quantile(0.05)),
               m_min=float(marg.min()), M_mean=float(M.mean()), M_med=float(M.median()), ce=float(ce.mean()),
               shift=float(z.mean()), zc_norm=float(zc.norm()))
    if y_old is not None:
        sel = y_old != y
        o2 = z.clone(); o2.scatter_(1, y[:, None], float("-inf"))
        top_other = o2.argmax(1)
        out["rank2"] = float((top_other[sel] == y_old[sel]).float().mean())
        o3 = z.clone(); o3.scatter_(1, y[:, None], float("nan"))
        mean_others = torch.nanmean(o3, 1)
        zold = z.gather(1, y_old[:, None])[:, 0]
        # z_{y_old} minus the mean of the other 8 non-current classes: (9*mean9 - z_old)/8
        m8 = (9 * mean_others - zold) / 8
        out["old_c"] = float((zold - m8)[sel].mean())
        out["acc_old"] = float((z.argmax(1) == y_old).float().mean())
    return out

def norms(P):
    return dict(n1=float((P[0] ** 2).sum()), n2=float((P[2] ** 2).sum()), n3=float((P[4] ** 2).sum()),
                nb3=float((P[5] ** 2).sum()))

def labels_ab(path):
    lab = np.load(path); ls = list(lab["seeds"])
    return {s: {n: torch.from_numpy(lab[n][ls.index(s)].astype(np.int64)).to(DEV) for n in ("A", "B")} for s in SEEDS}

# ---------------- part A: task ends ----------------
ARMS = {"A": (SP / "A", "iid"), "F": (SP / "F", "iid"), "S_hi": (SP / "S_hi", "iid"), "A_ce": (SP / "A_ce", "iid"),
        "SA_iid": (SP / "SA_iid", "iid"), "SA_abab": (SP / "SA_abab", "abab"), "LR_abab": (AL / "LR_abab", "abab")}
rows, checks = [], []
for arm, (d, sch) in ARMS.items():
    lab = labels_ab(d / "labels.npz") if sch == "abab" else None
    for s in SEEDS:
        tr = np.load(d / "trace" / f"LR_std_seed{s}.npz")
        for t in range(1, 51):
            f = d / "snap" / f"LR_std_seed{s}" / f"t{t:02d}.npz"
            if not f.exists():
                continue
            P = load(f); z = logits(P, s)
            if sch == "iid":
                y, yo = YI[s][t - 1], (YI[s][t - 2] if t > 1 else None)
            else:
                y, yo = lab[s]["A" if t % 2 == 1 else "B"], lab[s]["B" if t % 2 == 1 else "A"]
            r = dict(arm=arm, seed=s, t=t, **metrics(z, y, yo), **norms(P))
            m = tr["task"] == t
            r["trace_m_med"] = float(tr["margin_med"][m][-1]); r["trace_correct"] = int(tr["correct"][m][-1])
            rows.append(r)
            if t in (10, 30, 50):                      # mutation control: the wrong task's labels
                yw = YI[s][t] if sch == "iid" else lab[s]["B" if t % 2 == 1 else "A"]
                checks.append(dict(arm=arm, seed=s, t=t, correct_wrong_labels=metrics(z, yw)["correct"]))
dA = pd.DataFrame(rows); dA.to_csv(OUT / "task_ends.csv", index=False)
dA["dm"] = (dA["m_med"] - dA["trace_m_med"]).abs()
chk = dict(min_correct=int(dA["correct"].min()), n_rows=len(dA), max_abs_margin_diff_vs_trace=float(dA["dm"].max()),
           correct_equals_trace=bool((dA["correct"] == dA["trace_correct"]).all()),
           wrong_label_control_max_correct=int(pd.DataFrame(checks)["correct_wrong_labels"].max()))
print("checks A:", chk)

# ---------------- part B: the switch (crossover fork) ----------------
def zc_of(P, s): z = logits(P, s); return z - z.mean(1, keepdim=True), z
rowsB = []
for t in (10, 20, 30, 40, 48):
    for s in SEEDS:
        sd = f"LR_std_seed{s}"
        Psw = load(SP / f"xfork/t{t}_sw/snap/{sd}/t01.npz"); Pend = load(SP / f"A/snap/{sd}/t{t:02d}.npz")
        Pnx = load(SP / f"xfork/t{t}_Wend_Send/snap/{sd}/t01.npz"); Pnx0 = load(SP / f"xfork/t{t}_Wsw_Ssw/snap/{sd}/t01.npz")
        yt, yn = YI[s][t - 1], YI[s][t]
        zsw_c, zsw = zc_of(Psw, s); zend_c, zend = zc_of(Pend, s); znx_c, znx = zc_of(Pnx, s); znx0_c, znx0 = zc_of(Pnx0, s)
        ip = lambda a, b: float((a * b).sum()); nn = lambda a: float((a * a).sum())
        dpush, dret = zend_c - zsw_c, znx_c - zend_c
        Wp, Wr = Pend[0] - Psw[0], Pnx[0] - Pend[0]
        r = dict(t=t, seed=s,
                 push_cos=ip(zend_c, zsw_c) / np.sqrt(nn(zend_c) * nn(zsw_c)), push_scale=ip(zend_c, zsw_c) / nn(zsw_c),
                 push_resid_frac=nn(zend_c - ip(zend_c, zsw_c) / nn(zsw_c) * zsw_c) / nn(zend_c - zsw_c),
                 undo_logit=-ip(dret, dpush) / nn(dpush), cos_logit=ip(dret, dpush) / np.sqrt(nn(dret) * nn(dpush)),
                 undo_W1=-ip(Wr, Wp) / nn(Wp),
                 keep_logit=np.sqrt(nn(znx_c - znx0_c) / nn(dpush)),
                 keep_logit_proj=ip(znx_c - znx0_c, dpush) / nn(dpush))
        for tag, z, y, yo in (("sw_t", zsw, yt, None), ("end_t", zend, yt, None), ("sw_new", zsw, yn, yt), ("end_new", zend, yn, yt),
                              ("next", znx, yn, yt), ("next0", znx0, yn, yt)):
            for k, v in metrics(z, y, yo).items():
                r[f"{tag}_{k}"] = v
        rowsB.append(r)
dB = pd.DataFrame(rowsB); dB.to_csv(OUT / "switch.csv", index=False)
chkB = dict(min_correct_next=int(dB["next_correct"].min()), min_correct_next0=int(dB["next0_correct"].min()),
            min_correct_end_t=int(dB["end_t_correct"].min()), min_correct_sw_t=int(dB["sw_t_correct"].min()))
print("checks B:", chkB)

# ---------------- part C: ABAB round trips ----------------
rowsC = []
for arm in ("LR_abab", "SA_abab"):
    d = ARMS[arm][0]
    for s in SEEDS:
        Z, Wt = {}, {}
        for t in range(1, 51):
            P = load(d / "snap" / f"LR_std_seed{s}" / f"t{t:02d}.npz"); zc, _ = zc_of(P, s); Z[t], Wt[t] = zc, P[0]
        for j in range(14, 25):
            a, b, c = 2 * j - 1, 2 * j, 2 * j + 1
            ip = lambda u, v: float((u * v).sum()); nn = lambda u: float((u * u).sum())
            zab, zba, wab, wba = Z[b] - Z[a], Z[c] - Z[b], Wt[b] - Wt[a], Wt[c] - Wt[b]
            rowsC.append(dict(arm=arm, seed=s, j=j, cos_logit=ip(zab, zba) / np.sqrt(nn(zab) * nn(zba)),
                              cos_W1=ip(wab, wba) / np.sqrt(nn(wab) * nn(wba)),
                              close_logit=np.sqrt(nn(Z[c] - Z[a]) / nn(zab)), close_W1=np.sqrt(nn(Wt[c] - Wt[a]) / nn(wab))))
dC = pd.DataFrame(rowsC); dC.to_csv(OUT / "abab_roundtrip.csv", index=False)
json.dump(dict(checks_A=chk, checks_B=chkB), open(OUT / "checks.json", "w"), indent=2)
print("done", len(dA), len(dB), len(dC))
