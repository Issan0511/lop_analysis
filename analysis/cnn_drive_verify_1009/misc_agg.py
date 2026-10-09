#!/usr/bin/env python3
"""c2exact (non-linear expected first Adam step) and capacity (full / literal-self NTK logdet slope).

    python analysis/cnn_drive_verify_1009/misc_agg.py --root results/cnn_drive_verify_1009
"""
from __future__ import annotations

import argparse
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
    out = root / "tables_misc"; out.mkdir(parents=True, exist_ok=True)
    ch = pd.read_csv(root / "tables" / "channels.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    # ---- c2 exact
    rows = []
    for fn in sorted((root / "c2exact").glob("c2exact_*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        for k in range(2 * CH):
            layer, j = divmod(k, CH)
            rows.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "layer": f"c{layer + 1}", "ch": j,
                         "exact": float(r["exact_mean"][k]), "exact_se": float(r["exact_se"][k]),
                         "linear": float(r["linear_mean"][k]), "K": r["K"]})
    if rows:
        X = pd.DataFrame(rows).merge(ch[["arm", "seed", "task", "layer", "ch", "G_full", "fs_adam", "fs_adam_se"]],
                                     on=["arm", "seed", "task", "layer", "ch"])
        M = len(X)
        z = SL.norm_ppf(1 - 0.025 / M)
        sgd = -np.sign(X.G_full)
        X["exact_rev_sig"] = (np.sign(X.exact) != sgd) & (X.exact.abs() > z * X.exact_se)
        X["exact_rev_point"] = np.sign(X.exact) != sgd
        X["lin_rev_point"] = np.sign(X.linear) != sgd
        X["exact_vs_lin_sign"] = np.sign(X.exact) == np.sign(X.linear)
        X["rel_nonlin"] = (X.exact - X.linear).abs() / X.linear.abs()
        T = X.groupby(["layer", "task"]).agg(exact_rev_sig=("exact_rev_sig", "mean"),
                                            exact_rev_point=("exact_rev_point", "mean"),
                                            lin_rev_point=("lin_rev_point", "mean"),
                                            exact_vs_lin_sign=("exact_vs_lin_sign", "mean"),
                                            median_rel_nonlin=("rel_nonlin", "median"),
                                            n=("exact", "size")).reset_index()
        T.to_csv(out / "c2exact.csv", index=False)
        X.to_csv(out / "c2exact_channels.csv", index=False)
        print("z_bonf", z); print(T.round(4).to_string(index=False))
    # ---- capacity
    rows = []
    for fn in sorted((root / "capacity").glob("capacity_*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        for row in r["rows"]:
            rows.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"], **row})
    if rows:
        Cp = pd.DataFrame(rows).merge(ch[["arm", "seed", "task", "layer", "ch", "G_full", "G_self"]],
                                      on=["arm", "seed", "task", "layer", "ch"], how="left")
        res = []
        for lam in ("0.01", "1", "100"):
            f = Cp[f"full_cap_slope_lam{lam}"]; sl = Cp[f"self_cap_slope_lam{lam}"]
            for layer, g in Cp.groupby("layer"):
                ff = f[g.index]; ss = sl[g.index]
                res.append({"lam": lam, "layer": layer, "n": len(g),
                            "full_pos": (ff > 0).mean(), "self_pos": (ss > 0).mean(),
                            "full_self_agree": (np.sign(ff) == np.sign(ss)).mean(),
                            "full_cap_vs_Gsub": (np.sign(ff) == np.sign(g.full_G_sub)).mean(),
                            "full_cap_vs_Gfull": (np.sign(ff) == np.sign(g.G_full)).mean(),
                            "self_cap_vs_Gself": (np.sign(ss) == np.sign(g.G_self)).mean(),
                            "full_Kd_psd": (g.full_Kd_eig_min >= 0).mean(),
                            "self_Kd_psd": (g.self_Kd_eig_min >= 0).mean()})
        T = pd.DataFrame(res)
        T.to_csv(out / "capacity.csv", index=False)
        Cp.to_csv(out / "capacity_channels.csv", index=False)
        print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
