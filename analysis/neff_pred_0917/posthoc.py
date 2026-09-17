#!/usr/bin/env python3
"""neff_pred_0917 -- post hoc (report only, not registered).

    python3 analysis/neff_pred_0917/posthoc.py [--src results/neff_pred_0917]

Per box and seed, the trajectories behind the verdict: both layers' n_eff (|g|), mean |g|, the exact-zero
share, the negative-derivative share, mean z, and F, at the registered branch points and a few more.
Writes <src>/posthoc_trajectories.csv and <src>/fig_posthoc_neff_pred_0917.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "neff_pred_0917"))
import verdict as V                                                # noqa: E402

TS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 120, 149)


def box_curves(sh: dict) -> pd.DataFrame:
    out = []
    combos = [("ee", "RL", "ELU1", "ELU1")] + sorted(
        {(r.engine, r.env, r.act1, r.act2) for r in sh["gpu"][["engine", "env", "act1", "act2"]].drop_duplicates().itertuples()})
    for eng, env, a1, a2 in combos:
        pts = V.ee_points(sh["ee"]) if eng == "ee" else V.gpu_points(sh["gpu"], eng, env, a1, a2)
        pts = pts.assign(engine=eng, env=env, act1=a1, act2=a2)
        if eng != "ee":
            g = sh["gpu"]
            d = g[(g.engine == eng) & (g.env == env) & (g.act1 == a1) & (g.act2 == a2)][
                ["seed", "task", "start_zbar_l1", "start_zbar_l2"]].copy()
            d["t"] = d["task"] - 1
            pts = pts.merge(d.drop(columns="task"), on=["seed", "t"], how="left")
        out.append(pts)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / "neff_pred_0917"))
    args = ap.parse_args()
    src = Path(args.src)
    sh = V.load(src, None)
    cur = box_curves(sh)
    cur.to_csv(src / "posthoc_trajectories.csv", index=False)
    keep = ["neff_abs_l1", "neff_abs_l2", "gabs_l1", "gabs_l2", "zero_l1", "zero_l2", "neg_l2", "start_zbar_l1",
            "start_zbar_l2", "F", "E"]
    tab = (cur[cur.t.isin(TS)].groupby(["engine", "env", "act1", "act2", "t"])[keep].mean())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(tab.round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    boxes = [("ee", "RL", "ELU1", "ELU1"), ("B", "RL", "GELU", "GELU"), ("B", "PM", "GELU", "GELU"),
             ("L", "RL", "ELU1", "LR"), ("B", "RL", "SILU", "SILU"), ("B", "RL", "R", "R")]
    fig, axes = plt.subplots(2, len(boxes), figsize=(4 * len(boxes), 6.5), sharex="col")
    for j, b in enumerate(boxes):
        d = cur[(cur.engine == b[0]) & (cur.env == b[1]) & (cur.act1 == b[2]) & (cur.act2 == b[3])]
        for s, ds in d.groupby("seed"):
            ds = ds.sort_values("t")
            axes[0, j].plot(ds.t, ds.neff_abs_l1, color="tab:orange", lw=0.8, alpha=0.8)
            axes[0, j].plot(ds.t, ds.neff_abs_l2, color="tab:blue", lw=0.8, alpha=0.8)
            axes[1, j].plot(ds.t, ds.F, color="k", lw=0.8, alpha=0.8)
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylim(1e-5, 1.2)
        axes[0, j].set_title(f"{b[1]} {b[2]}->{b[3]} ({b[0]})")
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_xlabel("branch t")
    axes[0, 0].set_ylabel("n_eff (L1 orange, L2 blue)")
    axes[1, 0].set_ylabel("F")
    fig.tight_layout()
    fig.savefig(src / "fig_posthoc_neff_pred_0917.png", dpi=110)


if __name__ == "__main__":
    main()
