#!/usr/bin/env python3
"""eps_width_cifar_0923: effective (input-weighted) displacement |D|_S, c_S, sigma, balance and hit999 per arm, and the
scoring of the registered predictions (Claude, Codex).  Arms LL_e8/e6/e4/e3 (this run, std, 20 tasks) plus the earlier
A (eps 1e-8, battle engine, 50 tasks) and SA_iid (pure SGD) as anchors, same blocks.
    python3 analysis/eps_width_cifar_0923/analyze.py"""
import os, json, csv
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; OUT = REPO / "results/eps_width_cifar_0923"
SP = Path.home() / "Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923"
B = Path.home() / "Projects/obsidian-research-data/sgd_postfit_cifar_0923/basis_cache"
QL = {s: (lambda z: (z["Q"][:, :1199].astype(np.float64), z["lam"].astype(np.float64)))(np.load(B / f"basis_std_s{s}.npz")) for s in range(10)}
ARMS = {"LL_e8": (OUT / "LL_e8", "LL"), "LL_e6": (OUT / "LL_e6", "LL"), "LL_e4": (OUT / "LL_e4", "LL"), "LL_e3": (OUT / "LL_e3", "LL"),
        "A": (SP / "A", "LR"), "SA_iid": (SP / "SA_iid", "LR")}
rows = []
for arm, (d, pre) in ARMS.items():
    hit = {}
    pt = d / "per_task.csv"
    if pt.exists():
        for r in csv.DictReader(open(pt)):
            if r.get("hit999") not in ("", None):
                hit[(int(r["seed"]), int(r["task"]))] = int(r["hit999"])
    for s in range(10):
        Q, lam = QL[s]; prev = None
        for t in range(1, 21):
            f = d / "snap" / f"{pre}_std_seed{s}" / f"t{t:02d}.npz"
            if not f.exists():
                break
            W = np.load(f)["W1"].astype(np.float64); C = W @ Q
            if prev is not None:
                Cp = prev; CD = C - Cp
                V, dd, wd = float((lam * Cp * Cp).sum()), float((lam * CD * CD).sum()), float((lam * Cp * CD).sum())
                rows.append(dict(arm=arm, seed=s, t=t, W_S=np.sqrt(V), D_S=np.sqrt(dd), c_S=wd / np.sqrt(V * dd),
                                 sigma=float(np.median(np.sqrt((lam * C * C).sum(1)))), n1=float((W * W).sum()),
                                 hit999=hit.get((s, t), np.nan)))
            prev = C
df = pd.DataFrame(rows); df.to_csv(OUT / "per_task_eff.csv", index=False)
df["block"] = pd.cut(df.t, [1, 5, 10, 20], labels=["t2-5", "t6-10", "t11-20"])
g = df.groupby(["arm", "block", "seed"], observed=True)[["W_S", "D_S", "c_S", "sigma", "n1", "hit999"]].mean().groupby(["arm", "block"], observed=True).median()
g["W_S_star"] = g.D_S / (2 * g.c_S.abs()); g["obs/pred"] = g.W_S / g.W_S_star
pd.set_option("display.width", 220)
print(g.round(3).to_string())
L = g.xs("t11-20", level="block") if "t11-20" in g.index.get_level_values(1) else None
res = {}
if L is not None and all(a in L.index for a in ("LL_e8", "LL_e6", "LL_e4", "LL_e3")):
    r = {k: float(L.loc[k, "D_S"] / L.loc["LL_e8", "D_S"]) for k in ("LL_e6", "LL_e4", "LL_e3")}
    res = dict(Q1=bool(L.loc["LL_e8", "D_S"] > L.loc["LL_e6", "D_S"] > L.loc["LL_e4", "D_S"] > L.loc["LL_e3", "D_S"]),
               Q2=r["LL_e3"], Q2b_e6=r["LL_e6"], Q2b_e4=r["LL_e4"], Q3=float(L.loc["LL_e3", "sigma"] / L.loc["LL_e8", "sigma"]),
               Q4=bool(all(-0.35 <= L.loc[k, "c_S"] <= -0.18 for k in ("LL_e8", "LL_e6", "LL_e4", "LL_e3"))),
               Q5=bool(L.loc["LL_e3", "hit999"] <= 1.5 * L.loc["LL_e8", "hit999"]),
               Q6=bool(abs(L.loc["LL_e3", "sigma"] / L.loc["LL_e8", "sigma"] - r["LL_e3"]) <= 0.15),
               hit999={k: float(L.loc[k, "hit999"]) for k in L.index}, sigma={k: float(L.loc[k, "sigma"]) for k in L.index},
               D_S={k: float(L.loc[k, "D_S"]) for k in L.index}, c_S={k: float(L.loc[k, "c_S"]) for k in L.index})
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "verdict.json", "w"), indent=1)
