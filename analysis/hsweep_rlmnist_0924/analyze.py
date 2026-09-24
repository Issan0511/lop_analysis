#!/usr/bin/env python3
"""hsweep_rlmnist_0924 analysis: Σ-metric ledger, retention, read/unread split, P1/P2, spec verdict.

Definitions follow specs/spec_hsweep_rlmnist_0924.md.  Index convention: state_t is the
network after task t; D[s] = W[s+1] - W[s] is the displacement made by task s+1, and the
read map for that displacement is (W3·W2) of state s+1.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/hsweep_rlmnist_0924"
LATE = (41, 50)


def sigma_ip(A: np.ndarray, BS: np.ndarray) -> float:
    return float((A * BS).sum())


def read_projector(W2: np.ndarray, W3: np.ndarray) -> np.ndarray:
    M = W3 @ W2
    M = M - M.mean(0, keepdims=True)
    U, sv, _ = np.linalg.svd(M.T, full_matrices=False)
    U = U[:, sv > 1e-8 * max(sv[0], 1e-30)]
    return U @ U.T


def loglog_slope(t: np.ndarray, v: np.ndarray) -> float:
    return float(np.polyfit(np.log(t), np.log(v), 1)[0])


def analyze_series(d: Path, hidden: int, seed: int) -> dict:
    bank = np.load(d / "input_bank.npz")
    S = bank["covariance"].astype(np.float64)
    per_task = json.load(open(d / "per_task.json"))
    T = len(per_task)
    st = {t: np.load(d / f"state_{t:03d}.npz") for t in range(T + 1)}
    W = {t: st[t]["W1"].astype(np.float64) for t in range(T + 1)}
    WS = {t: W[t] @ S for t in range(T + 1)}
    V = {t: sigma_ip(W[t], WS[t]) for t in range(T + 1)}
    D = {s: W[s + 1] - W[s] for s in range(T)}
    DS = {s: D[s] @ S for s in range(T)}
    Q = {s: sigma_ip(D[s], DS[s]) for s in range(T)}
    X = {s: sigma_ip(W[s], DS[s]) for s in range(T)}
    rows = []
    for s in range(T):
        rows.append(dict(hidden=hidden, seed=seed, task=s + 1, V=V[s], V_next=V[s + 1], Q=Q[s], X=X[s],
                         c=X[s] / np.sqrt(V[s] * Q[s]), rho=np.sqrt(Q[s] / V[s]),
                         w1_sq=float((W[s] ** 2).sum()),
                         train_acc=per_task[s]["train_acc"], task_start_acc=per_task[s]["task_start_acc"]))
    tr = pd.DataFrame(rows)
    # late ledger and P1/P2 (transitions whose start task is in LATE; V_t is the width before task t+1)
    late = tr[(tr.task >= LATE[0]) & (tr.task <= LATE[1])]
    B = -2 * late.X.sum() / late.Q.sum()
    supply = late.Q.sum() / late.V.mean()
    neg = float((late.c < 0).mean())
    tt = late.task.to_numpy(float)
    slope = loglog_slope(tt, late.V.to_numpy())
    half = len(late) // 2
    halves = late.V.to_numpy()[half:].mean() / late.V.to_numpy()[:half].mean()
    P1 = bool(0.9 <= B <= 1.1 and supply >= 0.1 and neg >= 0.8)
    P2 = bool(P1 and abs(slope) <= 0.2 and 0.9 <= halves <= 1.1)
    # retention: primary s=20..39 (k<=10), secondary s=20..29 (k<=20)
    ret = {}
    for lo, hi, kmax in ((20, 39, 10), (20, 29, 20)):
        for k in range(kmax + 1):
            vals = [sigma_ip(W[s + 1 + k] - W[s], DS[s]) / Q[s] for s in range(lo, hi + 1) if s + 1 + k <= T]
            ret[f"f{k}_s{lo}_{hi}"] = float(np.mean(vals))
    # read/unread split, s=20..39, lags 1 and 10
    share, rf1, uf1, rf10, uf10 = [], [], [], [], []
    for s in range(20, 40):
        P = read_projector(st[s + 1]["W2"].astype(np.float64), st[s + 1]["W3"].astype(np.float64))
        Dr = P @ D[s]
        Du = D[s] - Dr
        qr, qu = sigma_ip(Dr, Dr @ S), sigma_ip(Du, Du @ S)
        share.append(qr / (qr + qu))
        for k, (rl, ul) in ((1, (rf1, uf1)), (10, (rf10, uf10))):
            if s + 1 + k <= T:
                dW = W[s + 1 + k] - W[s]
                rl.append(sigma_ip(dW, Dr @ S) / qr)
                ul.append(sigma_ip(dW, Du @ S) / qu if qu > 1e-9 * (qr + qu) else np.nan)   # h=8: no unread subspace
    out = dict(hidden=hidden, seed=seed, tasks=T,
               late_train_acc=float(late.train_acc.median()), late_task_start_acc=float(late.task_start_acc.median()),
               late_B=float(B), late_supply=float(supply), late_negc=neg, late_loglog=float(slope), late_halves=float(halves),
               P1=P1, P2=P2, late_dV=float((late.V_next - late.V).mean()), late_rho=float(late.rho.median()),
               late_c=float(late.c.median()), V_t50=V[T], V_t50_per_h=V[T] / hidden, w1_sq_t50=float((W[T] ** 2).sum()),
               read_share=float(np.mean(share)), read_f1=float(np.mean(rf1)), unread_f1=float(np.nanmean(uf1)),
               read_f10=float(np.mean(rf10)), unread_f10=float(np.nanmean(uf10)), **ret)
    return out, tr


def verdict(summary: pd.DataFrame) -> dict:
    g = summary.groupby("hidden").median(numeric_only=True).sort_index()
    fits = g[g.late_train_acc >= 0.99]
    f10 = fits["f10_s20_39"]
    result = dict(fitting_h=[int(h) for h in fits.index], nonfitting_h=[int(h) for h in g.index if h not in fits.index])
    if len(fits) < 2:
        result["verdict"] = "OTHER"
        result["reason"] = "fewer than two fitting arms"
        return result
    hmin = int(fits.index.min())
    ref = float(g.loc[100, "f10_s20_39"]) if 100 in g.index else float(fits["f10_s20_39"].iloc[-1])
    mono = bool(all(f10.iloc[i] <= f10.iloc[i + 1] + 1e-9 for i in range(len(f10) - 1)))   # f10 non-decreasing in h
    p2_counts = summary[summary.hidden.isin(fits.index)].groupby("hidden").P2.sum()
    p2_group = bool((p2_counts >= 4).any())
    fmin = float(f10.loc[hmin]); bmin = float(fits.loc[hmin, "late_B"])
    if mono and fmin <= 0.3 and (bmin >= 0.98 or p2_group):
        v = "CAPACITY"
    elif all(abs(f10 - ref) <= 0.1) and all((fits.late_B >= 0.88) & (fits.late_B <= 0.97)) and not p2_group:
        v = "LOSS_SET"
    elif (ref - fmin) >= 0.1 and fmin > 0.3 and not p2_group:
        v = "PARTIAL"
    else:
        v = "OTHER"
    result.update(verdict=v, h_min_fitting=hmin, f10_ref_h100=ref, f10_min_h=fmin, B_min_h=bmin,
                  monotone_f10=mono, any_group_P2=p2_group, p2_counts={int(k): int(x) for k, x in p2_counts.items()})
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    rows, trans = [], []
    for d in sorted(out.glob("h*_s*")):
        m = re.fullmatch(r"h(\d+)_s(\d+)", d.name)
        if not m:
            continue
        hidden, seed = int(m.group(1)), int(m.group(2))
        sd = d / f"seed_{seed:03d}"
        status = json.load(open(d / "status.json"))
        if status[str(seed)]["state"] != "complete":
            print(f"skip {d.name}: {status[str(seed)]['state']}")
            continue
        r, tr = analyze_series(sd, hidden, seed)
        rows.append(r); trans.append(tr)
    summary = pd.DataFrame(rows).sort_values(["hidden", "seed"])
    summary.to_csv(out / "per_seed.csv", index=False)
    pd.concat(trans).to_csv(out / "transitions.csv", index=False)
    g = summary.groupby("hidden").median(numeric_only=True).sort_index()
    cols = ["late_train_acc", "f1_s20_39", "f10_s20_39", "f20_s20_29", "late_B", "late_supply", "late_negc",
            "late_loglog", "late_halves", "late_dV", "late_rho", "late_c", "V_t50", "V_t50_per_h", "w1_sq_t50",
            "read_share", "read_f1", "unread_f1", "read_f10", "unread_f10"]
    counts = summary.groupby("hidden")[["P1", "P2"]].sum()
    lines = ["# hsweep_rlmnist_0924 summary (seed medians; P1/P2 = seeds passing of n)", "",
             g[cols].round(3).to_string(), "", counts.to_string(), ""]
    v = verdict(summary)
    lines += ["verdict: " + json.dumps(v, ensure_ascii=False)]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(v, open(out / "verdict.json", "w"), indent=2, ensure_ascii=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
