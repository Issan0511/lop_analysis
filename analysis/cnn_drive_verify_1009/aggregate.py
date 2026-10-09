#!/usr/bin/env python3
"""Aggregate cnn_drive_verify_1009 measurements into tables (a) and (b1).

    python analysis/cnn_drive_verify_1009/aggregate.py --measure results/cnn_drive_verify_1009/measure \
        --out results/cnn_drive_verify_1009/tables

Bands follow spec sec. 6: G is exact up to the per-checkpoint numerical floor (max discrepancy of
the three exact formulas); Adam's MC expectation is judged with a Bonferroni z over all
channel-checkpoint cases; rates are given per seed, then mean, SD and a t(9) 95% interval.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CH = 16


def load(measure_dir: Path) -> pd.DataFrame:
    rows = []
    for fn in sorted(measure_dir.glob("measure_*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        floor = max(np.abs(r["G_full"] - r["G_full_R"]).max(), np.abs(r["G_full"] - r["G_full_J"]).max())
        for k in range(2 * CH):
            layer, j = divmod(k, CH)
            row = {"arm": r["arm"], "seed": r["seed"], "task": r["task"], "layer": f"c{layer + 1}",
                   "ch": j, "floor": floor, "fit_conf": r["fit_conf"], "L_u": r["L_u"]}
            for key in ("zbar", "zsd", "alpha", "G_full", "G_full_J", "G_full_R", "u_norm",
                        "real_adam", "real_sgd", "real_adam_exact"):
                row[key] = float(r[key][k])
            if "G_self" in r:
                row["G_self"] = float(r["G_self"][k])
            for key in ("g_pos", "cond3", "cond5", "cond8", "cond5_all", "cond8_all", "cond3_all",
                        "cond58_all", "gn_mean"):
                row["img_" + key] = float(r["img_" + key][k])
            for key in ("adam", "adam_se", "adam_m0", "adam_m0_se", "sgd", "sgd_se"):
                row["fs_" + key] = float(r["fs_" + key][k])
            row["fs_K"] = r["fs_K"]
            row["absA_within20"] = r["real_absA_within20"]
            row["absA_med"] = float(r["real_absA_q"][2])
            row["vold_ratio_med"] = float(r["vold_ratio_q"][1])
            rows.append(row)
    d = pd.DataFrame(rows)
    d["seat"] = 2 * d["alpha"] * d["zbar"]
    d["sink_full"] = d["G_full"] > 0
    d["det_full"] = d["G_full"].abs() > d["floor"]
    if "G_self" in d:
        d["sink_self"] = d["G_self"] > 0
        d["agree"] = d["sink_full"] == d["sink_self"]
    return d


def seed_rate(d: pd.DataFrame, col: str, by: list[str]) -> pd.DataFrame:
    """rate per seed, then mean, SD, t(9) 95% interval over seeds."""
    per = d.groupby(by + ["seed"])[col].mean().reset_index()
    out = []
    for key, g in per.groupby(by):
        x = g[col].to_numpy(float)
        n = len(x)
        m, sd = x.mean(), (x.std(ddof=1) if n > 1 else float("nan"))
        hw = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        key = key if isinstance(key, tuple) else (key,)
        out.append(dict(zip(by, key)) | {"rate": m, "sd_seed": sd, "lo": m - hw, "hi": m + hw,
                                        "n_seed": n, "n_cases": int(d.set_index(by).loc[key].shape[0])
                                        if len(by) else len(d)})
    return pd.DataFrame(out)


def adam_classes(d: pd.DataFrame) -> pd.DataFrame:
    M = len(d)
    z = stats.norm.ppf(1 - 0.025 / M)
    sgd_sign = -np.sign(d["G_full"])                    # sign of the SGD expected change of the mean
    a = d["fs_adam"]; se = d["fs_adam_se"]
    sig = a.abs() > z * se
    d = d.copy()
    d["adam_rev_point"] = np.sign(a) != sgd_sign
    d["adam_rev_sig"] = sig & (np.sign(a) != sgd_sign) & d["det_full"]
    d["adam_same_sig"] = sig & (np.sign(a) == sgd_sign) & d["det_full"]
    d["adam_undet"] = ~(d["adam_rev_sig"] | d["adam_same_sig"])
    a0 = d["fs_adam_m0"]; se0 = d["fs_adam_m0_se"]
    d["adam_m0_rev_sig"] = (a0.abs() > z * se0) & (np.sign(a0) != sgd_sign) & d["det_full"]
    d["real_adam_vs_sgd_exp"] = np.sign(d["real_adam"]) != sgd_sign
    d["real_vs_adam_exp"] = np.sign(d["real_adam"]) != np.sign(a)
    d.attrs["z_bonf"] = z
    return d


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = load(Path(args.measure))
    d = adam_classes(d)
    d.to_csv(out / "channels.csv", index=False)
    rep = {"n_checkpoints": int(d.groupby(["arm", "seed", "task"]).ngroups),
           "n_channel_cases": len(d), "z_bonf": d.attrs["z_bonf"],
           "floor_max": float(d["floor"].max()), "undetermined_G_full": int((~d["det_full"]).sum()),
           "max_rel_identity": float(((d["G_full"] - d["G_full_R"]).abs() / d["G_full"].abs()).max())}
    tabs = {}
    # (a) sinking fraction and full/self agreement
    tabs["sink_full_by_layer_task"] = seed_rate(d, "sink_full", ["layer", "task"])
    tabs["sink_full_by_layer_arm_task"] = seed_rate(d, "sink_full", ["layer", "arm", "task"])
    if "G_self" in d:
        tabs["sink_self_by_layer_task"] = seed_rate(d, "sink_self", ["layer", "task"])
        tabs["agree_by_layer_task"] = seed_rate(d, "agree", ["layer", "task"])
        tabs["agree_by_layer_arm"] = seed_rate(d, "agree", ["layer", "arm"])
        tabs["agree_by_layer"] = seed_rate(d, "agree", ["layer"])
        # chance baseline per seed (and layer): pf ps + (1-pf)(1-ps)
        per = d.groupby(["layer", "seed"]).agg(pf=("sink_full", "mean"), ps=("sink_self", "mean"),
                                              ag=("agree", "mean")).reset_index()
        per["chance"] = per.pf * per.ps + (1 - per.pf) * (1 - per.ps)
        per["excess"] = per.ag - per.chance
        ex = []
        for layer, g in per.groupby("layer"):
            x = g.excess.to_numpy()
            hw = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
            ex.append({"layer": layer, "agree": g.ag.mean(), "chance": g.chance.mean(),
                       "excess": x.mean(), "lo": x.mean() - hw, "hi": x.mean() + hw})
        tabs["agree_excess_over_chance"] = pd.DataFrame(ex)
        # rank / sign relation of the two
        rr = []
        for layer, g in d.groupby("layer"):
            rho = stats.spearmanr(g.G_full, g.G_self).statistic
            rr.append({"layer": layer, "spearman_full_self": rho})
        tabs["full_self_spearman"] = pd.DataFrame(rr)
    # seat relation
    srows = []
    for layer, g in d.groupby("layer"):
        above = g.seat > -math.pi / 2
        srows.append({"layer": layer, "n_above": int(above.sum()), "n_below": int((~above).sum()),
                      "sink_above": g.sink_full[above].mean(), "sink_below": g.sink_full[~above].mean(),
                      "auc_seat_for_sink": auc(g.seat.to_numpy(), g.sink_full.to_numpy())})
    tabs["seat_vs_sink"] = pd.DataFrame(srows)
    # Codex's per-image sufficient conditions
    crow = []
    for layer, g in d.groupby("layer"):
        gp = g[g.sink_full]
        crow.append({"layer": layer, "cases_sink": len(gp),
                     "frac_cond58_all_images": gp.img_cond58_all.mean() if len(gp) else float("nan"),
                     "frac_cond5_all_images": gp.img_cond5_all.mean() if len(gp) else float("nan"),
                     "median_img_frac_cond5": gp.img_cond5.median() if len(gp) else float("nan"),
                     "median_img_frac_cond8": gp.img_cond8.median() if len(gp) else float("nan"),
                     "median_img_frac_cond3_all": g.img_cond3.median(),
                     "mean_img_frac_cond3": g.img_cond3.mean(),
                     "median_img_frac_gpos": g.img_g_pos.median()})
    tabs["codex_conditions"] = pd.DataFrame(crow)
    # (b1) first Adam step
    tabs["adam_rev_sig_by_layer_task"] = seed_rate(d, "adam_rev_sig", ["layer", "task"])
    tabs["adam_rev_point_by_layer_task"] = seed_rate(d, "adam_rev_point", ["layer", "task"])
    tabs["adam_undet_by_layer_task"] = seed_rate(d, "adam_undet", ["layer", "task"])
    tabs["adam_rev_sig_by_layer_arm"] = seed_rate(d, "adam_rev_sig", ["layer", "arm"])
    tabs["adam_m0_rev_sig_by_layer_task"] = seed_rate(d, "adam_m0_rev_sig", ["layer", "task"])
    tabs["real_adam_vs_sgd_exp_by_layer_task"] = seed_rate(d, "real_adam_vs_sgd_exp", ["layer", "task"])
    tabs["real_vs_adam_exp_by_layer_task"] = seed_rate(d, "real_vs_adam_exp", ["layer", "task"])
    ck = d.groupby(["arm", "seed", "task"]).agg(absA_within20=("absA_within20", "first"),
                                               absA_med=("absA_med", "first"),
                                               vold_ratio_med=("vold_ratio_med", "first")).reset_index()
    tabs["signlike_first_step"] = ck.groupby("task")[["absA_within20", "absA_med", "vold_ratio_med"]].median().reset_index()
    # c2 non-linearity of the realized step
    c2 = d[d.layer == "c2"]
    rep["c2_real_step_exact_vs_linear_rel_median"] = float(((c2.real_adam_exact - c2.real_adam).abs()
                                                           / c2.real_adam.abs()).median())
    rep["c2_real_step_sign_exact_vs_linear_agree"] = float((np.sign(c2.real_adam_exact)
                                                            == np.sign(c2.real_adam)).mean())
    c1 = d[d.layer == "c1"]
    rep["c1_real_step_exact_vs_linear_rel_max"] = float(((c1.real_adam_exact - c1.real_adam).abs()
                                                        / c1.real_adam.abs()).max())
    for k, t in tabs.items():
        t.to_csv(out / f"{k}.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(json.dumps(rep, indent=1, default=float))
    for k, t in tabs.items():
        print(f"\n## {k}\n{t.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
