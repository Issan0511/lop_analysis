#!/usr/bin/env python3
"""Phase 2: the leaky ReLU (LR) CNN next to the Snake CNN of phase 1, and the main verdict.

    python analysis/cnn_drive_verify_1009/compare_phase2.py --root results/cnn_drive_verify_1009

Main verdict (spec addendum 1.5): c1 full / literal-self sign agreement of LR, pooled over
t in {1,2,5,10,20,30}, rate per seed -> mean and t(9) 95% interval:
>= 0.75 SNAKE_SPECIFIC, < 0.65 CNN_GENERIC, otherwise PARTIAL.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import statlite as SL  # noqa: E402


def seed_ci(d: pd.DataFrame, col: str) -> dict:
    per = d.groupby("seed")[col].mean()
    n = len(per)
    m, sd = float(per.mean()), float(per.std(ddof=1)) if n > 1 else float("nan")
    hw = SL.t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"rate": m, "sd_seed": sd, "lo": m - hw, "hi": m + hw, "n_seed": n, "n": len(d)}


def verdict(r: float) -> str:
    return "SNAKE_SPECIFIC" if r >= 0.75 else ("CNN_GENERIC" if r < 0.65 else "PARTIAL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "LR" / "tables_compare"; out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(root / "tables" / "channels.csv"); S["box"] = "Snake"
    L = pd.read_csv(root / "LR" / "tables" / "channels.csv"); L["box"] = "LR"
    D = pd.concat([S, L], ignore_index=True)
    rows = []
    for (box, layer), g in D.groupby(["box", "layer"]):
        a = seed_ci(g, "agree"); sf = seed_ci(g, "sink_full"); ss = seed_ci(g, "sink_self")
        per = g.groupby("seed").agg(pf=("sink_full", "mean"), ps=("sink_self", "mean"), ag=("agree", "mean"))
        chance = float((per.pf * per.ps + (1 - per.pf) * (1 - per.ps)).mean())
        rows.append({"box": box, "layer": layer, "agree": a["rate"], "agree_lo": a["lo"], "agree_hi": a["hi"],
                     "agree_sd_seed": a["sd_seed"], "chance": chance, "sink_full": sf["rate"],
                     "sink_self": ss["rate"], "spearman_full_self": SL.spearman(g.G_full, g.G_self),
                     "full_float_self_sink": float(((~g.sink_full) & g.sink_self).mean()),
                     "full_sink_self_float": float((g.sink_full & ~g.sink_self).mean()), "n": len(g)})
    T_a = pd.DataFrame(rows)
    by_task = []
    for (box, layer, task), g in D.groupby(["box", "layer", "task"]):
        by_task.append({"box": box, "layer": layer, "task": task, "sink_full": g.sink_full.mean(),
                        "sink_self": g.sink_self.mean(), "agree": g.agree.mean(),
                        "adam_rev_sig": g.adam_rev_sig.mean(), "adam_rev_point": g.adam_rev_point.mean(),
                        "cond58_all": g[g.sink_full].img_cond58_all.mean() if g.sink_full.any() else float("nan"),
                        "img_cond5_median": g.img_cond5.median()})
    T_task = pd.DataFrame(by_task)
    lr_c1 = L[L.layer == "c1"]
    main_ci = seed_ci(lr_c1, "agree")
    rep = {"main": {"LR_c1_full_self_agreement": main_ci, "verdict_point": verdict(main_ci["rate"]),
                    "verdict_lo": verdict(main_ci["lo"]), "verdict_hi": verdict(main_ci["hi"])}}
    # extra (self shape, Adam decomposition)
    ext = []
    for box, p in (("Snake", root / "tables_extra" / "extra_channels.csv"),
                   ("LR", root / "LR" / "tables_extra" / "extra_channels.csv")):
        if p.exists():
            X = pd.read_csv(p)
            for layer, g in X.groupby("layer"):
                ext.append({"box": box, "layer": layer, "S0_pos": (g.S0 > 0).mean(), "S1_pos": (g.S1 > 0).mean(),
                            "agree_S1_full": ((g.S1 > 0) == g.sink_full).mean(),
                            "agree_S1_self": ((g.S1 > 0) == g.sink_self).mean(),
                            "spearman_S1_Gfull": SL.spearman(g.S1, g.G_full),
                            "spearman_S1_Gself": SL.spearman(g.S1, g.G_self),
                            "adam_rev": g.adam_rev.mean(), "meanscale_rev": g.ms_rev.mean(),
                            "cov_flips": g.cov_flips.mean(), "n": len(g)})
    T_ext = pd.DataFrame(ext)
    # frozen reference
    fz = []
    for box, p in (("Snake", root / "tables_b2" / "frozen_channels.csv"),
                   ("LR", root / "LR" / "tables_b2" / "frozen_channels.csv")):
        if p.exists():
            F = pd.read_csv(p)
            for (layer, Sn), g in F.groupby(["layer", "S"]):
                fz.append({"box": box, "layer": layer, "S": Sn, "rev_sig": g.rev_sig.mean(),
                           "same_sig": g.same_sig.mean(), "rev_point": g.rev_point.mean(), "n": len(g)})
    T_fz = pd.DataFrame(fz)
    # ledger
    lg = []
    for box, p in (("Snake", root / "tables_c" / "ledger_channels.csv"),
                   ("LR", root / "LR" / "tables_c" / "ledger_channels.csv")):
        if p.exists():
            G = pd.read_csv(p)
            G = G[G.task.isin([1, 10, 20])]
            for (layer, t), g in G.groupby(["layer", "task"]):
                lg.append({"box": box, "layer": layer, "task": t, "step10_down": (g.d10 < 0).mean(),
                           "push75_down": (g.d75 < 0).mean(), "net_down": (g.d30000 < 0).mean(),
                           "med_d10": g.d10.median(), "med_push": g.d75.median(), "med_net": g.d30000.median(),
                           "push_agree_negG": (np.sign(g.d75) == -np.sign(g.G_full)).mean(),
                           "ret_opposite": (np.sign(g.d30000 - g.d75) == -np.sign(g.d75)).mean(),
                           "rho_negG_net": SL.spearman(-g.G_full, g.d30000),
                           "rho_negGself_net": SL.spearman(-g.G_self, g.d30000), "n": len(g)})
    T_lg = pd.DataFrame(lg)
    # net sinking from init
    nz = []
    for box, p in (("Snake", root / "zbar" / "zbar_checkpoints.csv"),
                   ("LR", root / "LR" / "zbar" / "zbar_checkpoints.csv")):
        if p.exists():
            Z = pd.read_csv(p)
            piv = Z.pivot_table(index=["arm", "seed", "layer", "ch"], columns="task", values="zbar")
            for t in [c for c in piv.columns if c > 0]:
                d = (piv[t] - piv[0]).rename("d").reset_index()
                for layer, g in d.groupby("layer"):
                    per = g.groupby("seed").d.apply(lambda x: (x < 0).mean())
                    nz.append({"box": box, "layer": layer, "task": t, "down_from_init": per.mean(),
                               "sd_seed": per.std(), "median_d": g.d.median()})
    T_nz = pd.DataFrame(nz)
    for name, T in (("agree_sink", T_a), ("by_task", T_task), ("extra", T_ext), ("frozen", T_fz),
                    ("ledger", T_lg), ("net_from_init", T_nz)):
        T.to_csv(out / f"{name}.csv", index=False)
    (out / "main_verdict.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(json.dumps(rep, indent=1, default=float))
    for name, T in (("agree_sink", T_a), ("by_task", T_task), ("extra", T_ext), ("frozen", T_fz),
                    ("ledger", T_lg), ("net_from_init", T_nz)):
        print(f"\n## {name}\n{T.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
