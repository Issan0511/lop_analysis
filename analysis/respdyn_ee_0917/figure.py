#!/usr/bin/env python3
"""Figure for respdyn_ee_0917 (after the run; report only).

    python3 analysis/respdyn_ee_0917/figure.py  -> results/respdyn_ee_0917/fig_respdyn_ee_0917.png

(a) E of the t2-branch arms (fresh Adam) and the t10-branch restore arms, seed means with 95% t intervals.
(b) The trained-with argument z2 + d over the first continuation task (unit x image mean; probes every
    750 updates), with its two parts for the moving arm: the net's own z2 and the field d.
(c) The exact-zero share of the second layer's training derivative over the same task.
(d) The side run: online accuracy per task of the float64 / F.elu natural trajectory against the float32
    prefix of the same seeds (resp_ee_0917).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "respdyn_ee_0917"
spec = importlib.util.spec_from_file_location("_respdyn_verdict", REPO / "analysis" / "respdyn_ee_0917" / "verdict.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

INK = "#1f2328"
MUTED = "#6e7781"
COL = {"N2r": "#4c78a8", "S2_10r": "#f58518", "S2dyn_10r": "#e45756", "S2u30r": "#72b7b2", "N10r": "#54a24b",
       "N10": "#9d9d9d", "S2ramp_r": "#b279a2", "S2u10r": "#bab0ac", "S2u20r": "#ff9da6",
       "N2": "#4c78a8", "R2_10": "#f58518", "R2dyn_10": "#e45756", "R2_10_noanchor": "#d6616b",
       "S2dyn_5r": "#f4a5a5", "S2dyn_20r": "#a33a3a"}


def main() -> None:
    shards, _ = V.load(SRC)
    seeds = sorted(shards)
    arms = pd.concat([shards[s]["arms"] for s in seeds])
    traj = pd.concat([pd.read_csv(SRC / "runs" / f"s{s}" / "traj.csv") for s in seeds])
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))

    # (a)
    a = ax[0, 0]
    groups = [["N2r", "S2dyn_5r", "S2_10r", "S2ramp_r", "S2dyn_10r", "S2dyn_20r", "S2u30r", "N10r"],
              ["N2", "R2_10", "R2dyn_10", "R2_10_noanchor", "N10"]]
    labels, xs, pos = [], [], 0
    for g in groups:
        for arm in g:
            e = arms[(arms["arm"] == arm) & (arms["k"] == 1)].groupby("seed")["online_acc"].first().to_numpy()
            ci = V.paired(e)
            c = COL.get(arm, "#8c8c8c")
            a.bar(pos, e.mean(), color=c, alpha=0.85, width=0.75, hatch="//" if arm == "R2_10_noanchor" else None)
            a.errorbar(pos, e.mean(), yerr=[[e.mean() - ci["lo"]], [ci["hi"] - e.mean()]], color=INK, capsize=3, lw=1)
            a.scatter(np.full(e.size, pos) + np.linspace(-0.22, 0.22, e.size), e, s=7, color=INK, alpha=0.5, zorder=3)
            labels.append(arm)
            xs.append(pos)
            pos += 1
        pos += 0.8
    a.set_xticks(xs)
    a.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    a.set_ylabel("E = online accuracy, continuation task 1")
    a.set_ylim(0, 1)
    a.axhline(0.1, color=MUTED, lw=0.8, ls=":")
    a.set_title("(a) t2 net with fresh Adam (left) / t10 net, Adam kept (right)", fontsize=10, loc="left")

    # (b)
    b = ax[0, 1]
    for arm in ("N2r", "S2_10r", "S2dyn_10r", "N10r", "N10", "S2u30r"):
        t = traj[(traj["arm"] == arm) & (traj["k"] == 1)].groupby("s")["effbar2"].mean()
        b.plot(t.index, t.to_numpy(), color=COL[arm], lw=2 if arm == "S2dyn_10r" else 1.4, label=arm)
    for part, ls in (("zarm2", "--"), ("dmean2", ":")):
        t = traj[(traj["arm"] == "S2dyn_10r") & (traj["k"] == 1)].groupby("s")[part].mean()
        base = t.iloc[0]
        b.plot(t.index, (t - base).to_numpy() + traj[(traj["arm"] == "S2dyn_10r") & (traj["k"] == 1) & (traj["s"] == 0)]["effbar2"].mean(),
               color=COL["S2dyn_10r"], lw=1, ls=ls, label=f"S2dyn_10r: start + change of {'own z2' if part == 'zarm2' else 'field d'}")
    b.set_xlabel("update in continuation task 1")
    b.set_ylabel("mean trained-with argument z2 + d")
    b.axhline(np.log(2.0 ** -24), color=MUTED, lw=0.8, ls=":")
    b.legend(fontsize=7, frameon=False)
    b.set_title("(b) where the second layer trains", fontsize=10, loc="left")

    # (c)
    c = ax[1, 0]
    for arm in ("N2r", "S2_10r", "S2dyn_10r", "N10r", "N10", "S2ramp_r", "S2u20r"):
        t = traj[(traj["arm"] == arm) & (traj["k"] == 1)].groupby("s")["zero2"].mean()
        c.plot(t.index, t.to_numpy(), color=COL[arm], lw=2 if arm == "S2dyn_10r" else 1.4, label=arm)
    c.set_xlabel("update in continuation task 1")
    c.set_ylabel("share of (unit, image) with training derivative exactly 0")
    c.set_ylim(-0.02, 1.02)
    c.legend(fontsize=7, frameon=False)
    c.set_title("(c) the dead share of layer 2", fontsize=10, loc="left")

    # (d)
    d = ax[1, 1]
    nat, _ = V.load_nat(SRC)
    ref = V.load_ref32(sorted({s for (_, s) in nat}))
    styles = {"f64_felu": ("#e45756", "-"), "f32_felu": ("#f58518", "--"), "f64_host": ("#72b7b2", "-.")}
    for v, (col, ls) in styles.items():
        rows = [nat[(vv, s)]["nat"].set_index("task")["online_acc"] for (vv, s) in nat if vv == v]
        if not rows:
            continue
        m = pd.concat(rows, axis=1).mean(axis=1)
        d.plot(m.index, m.to_numpy(), color=col, ls=ls, lw=1.8, label=f"{v} ({len(rows)} seeds)")
    seeds32 = sorted({s for (vv, s) in nat if vv == "f64_felu"})
    if seeds32:
        m32 = pd.concat([ref[s] for s in seeds32], axis=1).mean(axis=1)
        d.plot(m32.index, m32.to_numpy(), color=INK, lw=1.2, label="float32 prefix (resp_ee, same seeds)")
    d.set_xlabel("task")
    d.set_ylabel("online accuracy")
    d.set_ylim(0, 1)
    d.axhline(0.1, color=MUTED, lw=0.8, ls=":")
    d.legend(fontsize=7, frameon=False)
    d.set_title("(d) natural trajectory without exact zeros in the derivative", fontsize=10, loc="left")

    for axx in ax.flat:
        axx.spines[["top", "right"]].set_visible(False)
    fig.suptitle("respdyn_ee_0917 — moving second-layer field (RL, ELU→ELU, 6000 updates/task)", fontsize=11)
    fig.tight_layout()
    out = SRC / "fig_respdyn_ee_0917.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
