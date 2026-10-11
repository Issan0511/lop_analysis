#!/usr/bin/env python3
"""Self shape S = <R_c, H_c> and Codex's Adam decomposition, joined to the main measurements.

    python analysis/cnn_drive_verify_1009/extra_agg.py --root results/cnn_drive_verify_1009
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import statlite as SL  # noqa: E402

CH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "tables_extra"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in sorted((root / "extra").glob("extra_*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        for k in range(2 * CH):
            layer, j = divmod(k, CH)
            rows.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "layer": f"c{layer + 1}", "ch": j,
                         "S0": float(r["S0"][k]), "S1": float(r["S1"][k]),
                         "dec_mom": float(r["dec_mom"][k]), "dec_meanscale": float(r["dec_meanscale"][k]),
                         "dec_cov": float(r["dec_cov"][k]), "dec_total": float(r["dec_total"][k])})
    X = pd.DataFrame(rows)
    ch = pd.read_csv(root / "tables" / "channels.csv")
    D = ch.merge(X, on=["arm", "seed", "task", "layer", "ch"])
    sgd = -np.sign(D.G_full)                                       # SGD's expected direction of the mean
    D["S0_sink"] = D.S0 > 0
    D["S1_sink"] = D.S1 > 0
    res = []
    for (layer, task), g in D.groupby(["layer", "task"]):
        res.append({"layer": layer, "task": task, "n": len(g),
                    "S0_pos": g.S0_sink.mean(), "S1_pos": g.S1_sink.mean(),
                    "agree_S0_full": (g.S0_sink == g.sink_full).mean(),
                    "agree_S1_full": (g.S1_sink == g.sink_full).mean(),
                    "agree_S0_self": (g.S0_sink == g.sink_self).mean() if "sink_self" in g else float("nan"),
                    "agree_S1_self": (g.S1_sink == g.sink_self).mean() if "sink_self" in g else float("nan"),
                    "spearman_S1_Gfull": SL.spearman(g.S1, g.G_full),
                    "spearman_S0_Gfull": SL.spearman(g.S0, g.G_full),
                    "spearman_S1_Gself": SL.spearman(g.S1, g.G_self) if "G_self" in g else float("nan")})
    T1 = pd.DataFrame(res)
    # Adam decomposition: which part carries the Adam direction, and the reversals
    s = np.sign
    D["ms_rev"] = s(D.dec_meanscale) != sgd            # coordinate scaling alone reverses
    D["adam_rev"] = s(D.dec_total) != sgd
    D["cov_flips"] = s(D.dec_meanscale + D.dec_cov) != s(D.dec_meanscale)
    D["mom_flips"] = s(D.dec_total) != s(D.dec_meanscale + D.dec_cov)
    D["cov_over_ms"] = D.dec_cov.abs() / D.dec_meanscale.abs()
    D["mom_over_ms"] = D.dec_mom.abs() / D.dec_meanscale.abs()
    D["consistency"] = (D.dec_total - D.fs_adam).abs() / D.fs_adam.abs()
    res = []
    for (layer, task), g in D.groupby(["layer", "task"]):
        rv = g[g.adam_rev]
        res.append({"layer": layer, "task": task, "n": len(g), "adam_rev": g.adam_rev.mean(),
                    "meanscale_rev": g.ms_rev.mean(), "cov_flips": g.cov_flips.mean(),
                    "mom_flips": g.mom_flips.mean(),
                    "rev_cases_with_meanscale_rev": rv.ms_rev.mean() if len(rv) else float("nan"),
                    "median_cov_over_meanscale": g.cov_over_ms.median(),
                    "median_mom_over_meanscale": g.mom_over_ms.median(),
                    "median_rel_dec_vs_fsadam": g.consistency.median()})
    T2 = pd.DataFrame(res)
    D.to_csv(out / "extra_channels.csv", index=False)
    T1.to_csv(out / "selfshape.csv", index=False)
    T2.to_csv(out / "adam_decomposition.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(T1.round(3).to_string(index=False))
    print(T2.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
