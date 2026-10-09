#!/usr/bin/env python3
"""(b2) frozen-parameter Adam reference: does Adam's cumulative response over the first S steps
of a task (labels reused, fresh batch orders, moments advancing, parameters frozen) have the
sign of SGD's expected push -G?

    python analysis/cnn_drive_verify_1009/frozen_agg.py --root results/cnn_drive_verify_1009
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "tables_b2"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in sorted((root / "frozen").glob("frozen_*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        L = r["L"]
        for s, v in r["rec"].items():
            A = v["adam"][1:]                 # random label draws (draw 0 = the real next task)
            Sg = v["sgd"][1:]
            for k in range(2 * CH):
                layer, j = divmod(k, CH)
                a = A[:, k]; g = Sg[:, k]
                rows.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "layer": f"c{layer + 1}",
                             "ch": j, "S": int(s), "G_full": float(r["G_full"][k]),
                             "adam_mean": a.mean(), "adam_se": a.std(ddof=1) / math.sqrt(L),
                             "sgd_mean": g.mean(), "sgd_se": g.std(ddof=1) / math.sqrt(L),
                             "adam_real": float(v["adam"][0, k]), "sgd_real": float(v["sgd"][0, k])})
        T = r["tail750_adam"][1:]
        for k in range(2 * CH):
            layer, j = divmod(k, CH)
            rows.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "layer": f"c{layer + 1}",
                         "ch": j, "S": -750, "G_full": float(r["G_full"][k]),
                         "adam_mean": T[:, k].mean(), "adam_se": T[:, k].std(ddof=1) / math.sqrt(L),
                         "sgd_mean": float("nan"), "sgd_se": float("nan"),
                         "adam_real": float(r["tail750_adam"][0, k]), "sgd_real": float("nan")})
    d = pd.DataFrame(rows)
    M = int((d.S == 1).sum())
    z = stats.norm.ppf(1 - 0.025 / max(M, 1))
    sgd_sign = -np.sign(d.G_full)
    d["rev_point"] = np.sign(d.adam_mean) != sgd_sign
    d["rev_sig"] = (d.adam_mean.abs() > z * d.adam_se) & (np.sign(d.adam_mean) != sgd_sign)
    d["same_sig"] = (d.adam_mean.abs() > z * d.adam_se) & (np.sign(d.adam_mean) == sgd_sign)
    d["real_rev"] = np.sign(d.adam_real) != sgd_sign
    d.to_csv(out / "frozen_channels.csv", index=False)
    g = d.groupby(["layer", "task", "S"]).agg(rev_point=("rev_point", "mean"), rev_sig=("rev_sig", "mean"),
                                             same_sig=("same_sig", "mean"), real_rev=("real_rev", "mean"),
                                             n=("rev_point", "size")).reset_index()
    g.to_csv(out / "frozen_rates.csv", index=False)
    pd.set_option("display.width", 220)
    print("z_bonf", z)
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
