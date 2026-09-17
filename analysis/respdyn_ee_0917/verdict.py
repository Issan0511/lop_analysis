#!/usr/bin/env python3
"""Verdict for respdyn_ee_0917 (specs/spec_respdyn_ee_0917.md section 4).

    python3 analysis/respdyn_ee_0917/verdict.py                     # the registered run
    python3 analysis/respdyn_ee_0917/verdict.py --src results/_smoke_respdyn_ee_0917 --out /tmp/x --partial

Reads runs/s<seed>/{arms.csv, prefix.csv, provenance.json} and nat/<variant>_s<seed>/{nat.csv,
provenance.json} for the listed seeds only (no glob), builds every quantity per seed and only then
summarises.  The float32 reference of the side run is resp_ee_0917's committed prefix.  Refuses to read
the registered run directory before every job has a provenance.json unless --partial is given (and
then says so in the summary).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load_resp_ee_verdict():
    """resp_ee_0917's verdict (t quantile, paired interval, floor rule), by path under its own name."""
    spec = importlib.util.spec_from_file_location("_resp_ee_0917_verdict",
                                                  REPO / "analysis" / "resp_ee_0917" / "verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RV = _load_resp_ee_verdict()

RUN_ID = "respdyn_ee_0917"
MAIN_SRC = REPO / "results" / RUN_ID
RESP_EE_RUNS = REPO / "results" / "resp_ee_0917" / "runs"
SEEDS = tuple(range(10))
MIN_SEEDS = 8                                    # spec 4.2 (A)
LEVEL = 0.95                                     # the design's intervals (spec 4.1)
N_IMG = 1200
EPOCHS = 80
PREFIX_T = 22
NEW_ARMS = ["S2dyn_10r", "R2dyn_10", "S2ramp_r", "S2dyn_5r", "S2dyn_20r", "R2_10_noanchor"]
OLD_ARMS = (["S2_5r", "S2_7r", "S2_10r", "S2_15r", "S2_20r", "S2_10"]
            + [f"S2u{d}r" for d in (5, 10, 15, 20, 30)] + ["S1u20r"]
            + ["R2_5", "R2_7", "R2_10", "R2_15", "R2_20", "R2_10r", "N2r", "N10r"]
            + [f"N{t}" for t in (2, 5, 7, 10, 15, 20)])
ARM_NAMES = NEW_ARMS + OLD_ARMS
N_ARMS = [f"N{t}" for t in (2, 5, 7, 10, 15, 20)]
DYN_ARMS = ["S2dyn_10r", "R2dyn_10", "S2dyn_5r", "S2dyn_20r"]
UNANCHORED = ["R2_10_noanchor"]
MAIN, FIXED, FLOOR_ARM = "S2dyn_10r", "S2_10r", "S2u30r"
M_PAIR = (FIXED, MAIN)                           # movement part: E(fixed) - E(dyn)
R_PAIR = (MAIN, FLOOR_ARM)                       # residual above the response floor
A_PAIR = (MAIN, "N10r")                          # the design's literal reference
X_PAIR = ("R2dyn_10", "N2")                      # sub-judgment: is R2_10's excess the static field's?
EXCESS_PAIR = ("R2_10", "N2")
B_PAIRS = {"natural_gap": ("N2", "N10"), "fixed_sink": ("N2r", "S2_10r"), "floor_below_fixed": (FIXED, FLOOR_ARM)}
SECONDARY = [
    ("S2dyn_10r", "N10r", "dyn vs the collapsed net (fresh Adam): the design's literal reference"),
    ("S2u30r", "N10r", "forward-value part: frozen t2 features vs the collapsed net"),
    ("R2dyn_10", "R2_10", "restore: moving vs fixed t2 field"),
    ("R2_10", "N2", "restore excess (resp_ee)"),
    ("R2_10_noanchor", "R2_10", "restore without the fixed anchor"),
    ("R2_10_noanchor", "N2", "unanchored restore vs the natural t2 net"),
    ("S2_5r", "S2dyn_5r", "movement at t5 (fixed - dyn)"),
    ("S2_20r", "S2dyn_20r", "movement at t20 (fixed - dyn)"),
    ("S2dyn_20r", "S2u30r", "dyn t20 above the response floor"),
    ("S2u10r", "S2ramp_r", "ramp vs its starting rung (10)"),
    ("S2u20r", "S2ramp_r", "ramp vs the deepest rung it reaches in task 1 (20)"),
    ("N2r", "S2dyn_10r", "total sink with the moving field"),
    ("N2r", "S2_10r", "total sink with the fixed field (resp_ee P2)"),
    ("N2r", "N10r", "natural gap, both fresh Adam"),
]
NEFF_FLOOR = 1.0 / N_IMG                         # one image's worth: the resolution of n_eff
NEFF_EXCLUDE = ["S1u20r"]                        # a first-layer manipulation: layer 2's n_eff does not describe it
NAT_VARIANTS = {"f64_felu": tuple(range(10)), "f32_felu": (0, 1, 2), "f64_host": (0, 1, 2)}
NAT_REGISTERED = "f64_felu"
NAT_DTYPE = {"f64_felu": "torch.float64", "f32_felu": "torch.float32", "f64_host": "torch.float64"}
LATE = tuple(range(18, 23))
PRED = {
    "design": {"S2dyn_10r": 0.22, "label": "FULL_MEDIATION", "label_probs": {"FULL_MEDIATION": 0.55, "PARTIAL": 0.35, "STATIC_ONLY": 0.10},
               "R2dyn_10": 0.80, "sub": "EXCESS_IS_STATIC", "sub_prob": 0.50, "S2ramp_r": 0.45,
               "nat": "REPRODUCED", "nat_prob": 0.80},
    "impl": {"S2dyn_10r": 0.27, "label": "PARTIAL",
             "label_probs": {"PARTIAL": 0.55, "FULL_MEDIATION": 0.28, "STATIC_ONLY": 0.12, "UNRESOLVED": 0.04,
                             "MOVEMENT_HELPS": 0.01},
             "R2dyn_10": 0.88, "sub": "EXCESS_REMAINS", "sub_prob": 0.75, "S2ramp_r": 0.55,
             "nat": "REPRODUCED", "nat_prob": 0.85, "neff": "NEFF_PREDICTS", "neff_prob": 0.60,
             "R2_10_noanchor": 0.80, "S2dyn_5r": 0.70, "S2dyn_20r": 0.22},
}


# --------------------------------------------------------------------------

def load(src: Path, seeds=SEEDS) -> tuple[dict, list]:
    shards, missing = {}, []
    for s in seeds:
        d = src / "runs" / f"s{s}"
        need = [d / n for n in ("arms.csv", "prefix.csv", "provenance.json")]
        if not all(p.exists() for p in need):
            missing.append(s)
            continue
        shards[s] = {"arms": pd.read_csv(d / "arms.csv", float_precision="round_trip"),
                     "prefix": pd.read_csv(d / "prefix.csv", float_precision="round_trip"),
                     "prov": json.loads((d / "provenance.json").read_text())}
    return shards, missing


def load_nat(src: Path, variants=NAT_VARIANTS) -> tuple[dict, list]:
    out, missing = {}, []
    for v, seeds in variants.items():
        for s in seeds:
            d = src / "nat" / f"{v}_s{s}"
            if not ((d / "nat.csv").exists() and (d / "provenance.json").exists()):
                missing.append(f"{v}_s{s}")
                continue
            out[(v, s)] = {"nat": pd.read_csv(d / "nat.csv", float_precision="round_trip"),
                           "prov": json.loads((d / "provenance.json").read_text())}
    return out, missing


def load_ref32(seeds, root: Path = RESP_EE_RUNS) -> dict:
    out = {}
    for s in seeds:
        f = root / f"s{s}" / "prefix.csv"
        if f.exists():
            out[s] = pd.read_csv(f, float_precision="round_trip").set_index("task")["online_acc"]
    return out


_truth = RV._truth


def E(a: pd.DataFrame, arm: str, k: int = 1, col: str = "online_acc") -> float:
    r = a[(a["arm"] == arm) & (a["k"] == k)]
    return float(r[col].iloc[0])


def seed_validity(sh: dict, prereg: str | None, epochs: int) -> str | None:
    a, p, prov = sh["arms"], sh["prefix"], sh["prov"]
    if prereg is not None and prov.get("prereg_commit") != prereg:
        return f"prereg_commit {prov.get('prereg_commit')} != {prereg}"
    if not prov.get("flush_denormal"):
        return "flush_denormal off"
    if int(prov.get("epochs_per_task", -1)) != epochs:
        return f"epochs {prov.get('epochs_per_task')}"
    if len(p) != PREFIX_T or not np.isfinite(p["online_acc"]).all():
        return "prefix incomplete or non-finite"
    for arm in ARM_NAMES:
        r = a[a["arm"] == arm]
        if sorted(r["k"].tolist()) != [1, 2]:
            return f"{arm}: continuation tasks {sorted(r['k'].tolist())}"
        if not (np.isfinite(r["online_acc"]).all() and np.isfinite(r["online_ce"]).all()):
            return f"{arm}: non-finite"
        r1 = r[r["k"] == 1].iloc[0]
        if arm not in UNANCHORED and not (_truth(r1["logits_equal_full"]) and _truth(r1["logits_equal_mb"])):
            return f"{arm}: branch-point logits differ from the natural net"
        if arm in N_ARMS and not all(_truth(v) for v in r["hash_match_prefix"]):
            return f"{arm}: natural continuation does not reproduce the prefix"
        if arm in DYN_ARMS and not all(_truth(v) for v in r["shadow_hash_match_prefix"]):
            return f"{arm}: the shadow is not the natural continuation"
        if arm in DYN_ARMS + ["S2ramp_r"] and not _truth(r1["field0_equal_fixed"]):
            return f"{arm}: the field does not start at the fixed field"
    return None


def paired(x) -> dict:
    return RV.paired(x, LEVEL)


def main_label(m: dict, r: dict) -> str:
    if m["sign"] == "-":
        return "MOVEMENT_HELPS"
    if m["sign"] == "0":
        return "STATIC_ONLY" if r["sign"] == "+" else "UNRESOLVED"
    return "PARTIAL" if r["sign"] == "+" else "FULL_MEDIATION"


def literal_label(m: dict, a: dict) -> str:
    if m["sign"] == "-":
        return "MOVEMENT_HELPS"
    if m["sign"] == "0":
        return "STATIC_ONLY" if a["sign"] == "+" else "UNRESOLVED"
    return {"+": "PARTIAL", "0": "FULL_MEDIATION", "-": "BEYOND_NATURAL"}[a["sign"]]


def ols_pi(x, y, x0, level=0.95) -> dict:
    """Straight line y = b0 + b1 x by least squares, and the prediction interval for a new point at x0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    X = np.c_[np.ones(n), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s = math.sqrt(float(res @ res) / (n - 2))
    sxx = float(((x - x.mean()) ** 2).sum())
    tq = RV.t_quantile(0.5 + level / 2, n - 2)
    pred = float(b[0] + b[1] * x0)
    half = tq * s * math.sqrt(1 + 1 / n + (x0 - x.mean()) ** 2 / sxx)
    r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    return {"b0": float(b[0]), "b1": float(b[1]), "s": s, "n": n, "r2": r2, "pred": pred,
            "lo": pred - half, "hi": pred + half}


def lx(v: float) -> float:
    return math.log10(max(float(v), NEFF_FLOOR))


def neff_test(table: pd.DataFrame, col: str) -> dict:
    fit = table[table["arm"].isin([a for a in OLD_ARMS if a not in NEFF_EXCLUDE])]
    xs = [lx(v) for v in fit[col]]
    ys = fit["E_k1"].tolist()
    out = {"col": col, "n_fit": len(fit), "arms": {}}
    base = ols_pi(xs, ys, xs[0])
    out.update({k: base[k] for k in ("b0", "b1", "s", "r2")})
    for arm in NEW_ARMS:
        r = table[table["arm"] == arm].iloc[0]
        p = ols_pi(xs, ys, lx(r[col]))
        out["arms"][arm] = {"x": lx(r[col]), "E": float(r["E_k1"]), "pred": p["pred"], "lo": p["lo"],
                            "hi": p["hi"], "inside": bool(p["lo"] <= r["E_k1"] <= p["hi"])}
    loo = []
    for i in range(len(xs)):
        p = ols_pi(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:], xs[i])
        loo.append(p["lo"] <= ys[i] <= p["hi"])
    out["loo_coverage"] = float(np.mean(loo))
    return out


def nat_analysis(nat: dict, ref32: dict, prereg: str | None, epochs: int = EPOCHS) -> dict:
    out = {}
    for v, seeds in NAT_VARIANTS.items():
        rows, invalid = [], {}
        for s in seeds:
            if (v, s) not in nat:
                continue
            d, prov = nat[(v, s)]["nat"], nat[(v, s)]["prov"]
            why = None
            if prereg is not None and prov.get("prereg_commit") != prereg:
                why = f"prereg_commit {prov.get('prereg_commit')}"
            elif not prov.get("flush_denormal"):
                why = "flush off"
            elif int(prov.get("epochs_per_task", -1)) != epochs or len(d) != PREFIX_T:
                why = "length"
            elif prov.get("param_dtypes") != [NAT_DTYPE[v]]:
                why = f"dtype {prov.get('param_dtypes')}"
            elif not np.isfinite(d["online_acc"]).all():
                why = "non-finite"
            elif s not in ref32:
                why = "no float32 reference"
            if why:
                invalid[s] = why
                continue
            e = d.set_index("task")["online_acc"]
            r = ref32[s]
            mid = (r.loc[3] + r.loc[11]) / 2
            mid_late = (r.loc[3] + np.mean([r.loc[t] for t in LATE])) / 2
            e_late = float(np.mean([e.loc[t] for t in LATE]))
            t_half = next((t for t in range(3, PREFIX_T + 1) if e.loc[t] < mid), None)
            t_half32 = next((t for t in range(3, PREFIX_T + 1) if r.loc[t] < mid), None)
            di = d.set_index("task")
            rows.append({"variant": v, "seed": s, "E3": float(e.loc[3]), "E11": float(e.loc[11]), "E_late": e_late,
                         "E3_32": float(r.loc[3]), "E11_32": float(r.loc[11]),
                         "E_late_32": float(np.mean([r.loc[t] for t in LATE])),
                         "D_E": float(e.loc[11] - mid), "D_late": float(e_late - mid_late),
                         "C_E": bool(e.loc[11] < mid), "C_late": bool(e_late < mid_late),
                         "T_half": t_half, "T_half_32": t_half32,
                         "zbar2_t10": float(di.loc[10, "zbar2"]), "zbar2_t22": float(di.loc[22, "zbar2"]),
                         "zero2_t10": float(di.loc[10, "zero2"]), "zero2_t22": float(di.loc[22, "zero2"]),
                         "below_2p24_t10": float(di.loc[10, "below_2p24_2"]),
                         "log10_g2_median_t10": float(di.loc[10, "log10_g2_median"]),
                         "eps_w2_t10": float(di.loc[10, "eps_w2_end"]), "eps_w2_t22": float(di.loc[22, "eps_w2_end"])})
        res = {"rows": rows, "invalid": invalid, "n_valid": len(rows)}
        if len(rows) >= 2:
            dE = paired([r["D_E"] for r in rows])
            dL = paired([r["D_late"] for r in rows])
            res["D_E"], res["D_late"] = dE, dL
            if dE["hi"] < 0 and dL["hi"] < 0:
                lab = "REPRODUCED"
            elif dL["hi"] < 0:
                lab = "DELAYED"
            elif dL["lo"] > 0:
                lab = "NOT_REPRODUCED"
            else:
                lab = "UNRESOLVED"
            if v == NAT_REGISTERED and len(rows) < MIN_SEEDS:
                lab = "INAPPLICABLE"
            res["label"] = lab
        else:
            res["label"] = "INAPPLICABLE"
        out[v] = res
    return out


def analyze(shards: dict, nat: dict | None = None, ref32: dict | None = None, prereg: str | None = None,
            epochs: int = EPOCHS) -> dict:
    invalid = {s: why for s in sorted(shards) if (why := seed_validity(shards[s], prereg, epochs))}
    valid = [s for s in sorted(shards) if s not in invalid]
    res = {"valid_seeds": valid, "invalid_seeds": invalid, "n_valid": len(valid)}
    if nat is not None:
        res["nat"] = nat_analysis(nat, ref32 or {}, prereg, epochs)

    def diffs(x: str, y: str, k: int = 1) -> np.ndarray:
        return np.array([E(shards[s]["arms"], x, k) - E(shards[s]["arms"], y, k) for s in valid])

    if len(valid) < MIN_SEEDS:
        res["label"] = "INAPPLICABLE"
        res["why"] = f"{len(valid)} valid seeds < {MIN_SEEDS}"
        return res

    B = {k: paired(diffs(*p)) for k, p in B_PAIRS.items()}
    moved = {s: (E(shards[s]["arms"], MAIN, 1, "probe_dmean2_first"), E(shards[s]["arms"], MAIN, 1, "probe_dmean2_last"))
             for s in valid}
    B["field_moved"] = all(last < first for first, last in moved.values())
    B["field_mean_start_end"] = moved
    res["B"] = B
    m, r, a = paired(diffs(*M_PAIR)), paired(diffs(*R_PAIR)), paired(diffs(*A_PAIR))
    res["M"], res["R"], res["A"] = m, r, a
    if not all(B[k]["lo"] > 0 for k in B_PAIRS):
        res["label"] = res["literal_label"] = "NOT_REPRODUCED"
    elif not B["field_moved"]:
        res["label"] = res["literal_label"] = "FIELD_STATIC"
    else:
        res["label"] = main_label(m, r)
        res["literal_label"] = literal_label(m, a)

    tot = diffs("N2r", "N10r")
    res["rho"] = {"rho_dyn": float(np.mean(diffs("N2r", MAIN)) / np.mean(tot)),
                  "rho_fixed": float(np.mean(diffs("N2r", FIXED)) / np.mean(tot)),
                  "rho_star": float(np.mean(diffs("N2r", MAIN)) / np.mean(diffs("N2r", FLOOR_ARM))),
                  "gap_fresh": float(np.mean(tot))}

    ex = paired(diffs(*EXCESS_PAIR))
    x = paired(diffs(*X_PAIR))
    res["excess"], res["X"] = ex, x
    if ex["lo"] <= 0:
        res["sub_label"] = "NO_EXCESS"
    else:
        res["sub_label"] = {"0": "EXCESS_IS_STATIC", "+": "EXCESS_REMAINS", "-": "EXCESS_REVERSED"}[x["sign"]]

    sec = []
    for xx, yy, what in [(*M_PAIR, "M (registered)"), (*R_PAIR, "R (registered)"), (*X_PAIR, "X (registered)")] + SECONDARY:
        for k in (1, 2):
            sec.append({"contrast": f"{xx} - {yy}", "what": what, "k": k, **paired(diffs(xx, yy, k))})
    res["secondary"] = sec

    rows = []
    for arm in ARM_NAMES:
        r1 = [shards[s]["arms"].query("arm == @arm and k == 1").iloc[0] for s in valid]
        r2 = [shards[s]["arms"].query("arm == @arm and k == 2").iloc[0] for s in valid]
        e1 = np.array([q["online_acc"] for q in r1])
        ci = paired(e1)
        rows.append({
            "arm": arm, "E_k1": float(e1.mean()), "E_k1_lo": ci["lo"], "E_k1_hi": ci["hi"],
            "E_k2": float(np.mean([q["online_acc"] for q in r2])),
            "floor_k1": sum(float(q["online_acc"]) <= RV.floor_thr(float(q["major_frac"])) for q in r1),
            "nbar_neffT2_k1": float(np.mean([q["nbar_neffT2"] for q in r1])),
            "nbar_neffT2_k2": float(np.mean([q["nbar_neffT2"] for q in r2])),
            "neffT2_start": float(np.mean([q["neffT2_start"] for q in r1])),
            "zero2_start": float(np.mean([q["zero2_start"] for q in r1])),
            "probe_zero2_mean_k1": float(np.mean([q["probe_zero2_mean"] for q in r1])),
            "zero2_end_k1": float(np.mean([q["zero2_end"] for q in r1])),
            "effbar2_first_k1": float(np.mean([q["probe_effbar2_first"] for q in r1])),
            "effbar2_last_k1": float(np.mean([q["probe_effbar2_last"] for q in r1])),
            "dmean2_move_k1": float(np.mean([q["probe_dmean2_last"] - q["probe_dmean2_first"] for q in r1])),
            "zarm2_move_k1": float(np.mean([q["probe_zarm2_last"] - q["probe_zarm2_first"] for q in r1])),
            "step_w2_k1": float(np.mean([q["step_w2"] for q in r1])),
            "memo_k1": float(np.mean([q["memo_acc_end"] for q in r1])),
            "resp_ee_hash_match": (None if arm in NEW_ARMS else
                                   int(sum(_truth(q["hash_match_resp_ee"]) for q in r1 + r2))),
            "reset": arm.endswith("r")})
    table = pd.DataFrame(rows)
    res["arm_table"] = table
    res["neff"] = neff_test(table, "nbar_neffT2_k1")
    res["neff_start"] = neff_test(table, "neffT2_start")
    res["neff_label"] = "NEFF_PREDICTS" if res["neff"]["arms"][MAIN]["inside"] else "NEFF_MISSES"

    ci = {arm: (float(t["E_k1_lo"]), float(t["E_k1_hi"]), float(t["E_k1"])) for arm, t in table.set_index("arm").iterrows()}
    nat_lab = res.get("nat", {}).get(NAT_REGISTERED, {}).get("label")
    preds = {}
    for who, P in PRED.items():
        q = {}
        for arm in ("S2dyn_10r", "R2dyn_10", "S2ramp_r", "R2_10_noanchor", "S2dyn_5r", "S2dyn_20r"):
            if arm in P:
                lo, hi, mean = ci[arm]
                q[f"E_{arm}"] = {"pred": P[arm], "obs": mean, "lo": lo, "hi": hi, "hit": bool(lo <= P[arm] <= hi)}
        q["label"] = {"pred": P["label"], "obs": res["label"], "hit": P["label"] == res["label"],
                      "prob_of_obs": P["label_probs"].get(res["label"], 0.0)}
        q["literal_label"] = {"pred": P["label"], "obs": res["literal_label"], "hit": P["label"] == res["literal_label"],
                              "prob_of_obs": P["label_probs"].get(res["literal_label"], 0.0)}
        q["sub"] = {"pred": P["sub"], "prob": P["sub_prob"], "obs": res["sub_label"], "hit": P["sub"] == res["sub_label"]}
        q["nat"] = {"pred": P["nat"], "prob": P["nat_prob"], "obs": nat_lab, "hit": P["nat"] == nat_lab}
        if "neff" in P:
            q["neff"] = {"pred": P["neff"], "prob": P["neff_prob"], "obs": res["neff_label"],
                         "hit": P["neff"] == res["neff_label"]}
        preds[who] = q
    res["predictions"] = preds

    pref = pd.concat([shards[s]["prefix"] for s in valid])
    res["prefix"] = pref.groupby("task")[["online_acc", "memo_acc", "g2_end", "gtr2_end", "zero2_end",
                                          "neffT2_end", "zbar2_end", "mu2_end"]].mean()
    res["prefix_link"] = {s: int(sum(_truth(v) for v in shards[s]["prefix"]["hash_match_resp_ee"])) for s in valid}
    return res


# --------------------------------------------------------------------------

def _f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "nan" if not np.isfinite(x) else f"{x:+.{nd}f}"


def _ci(p: dict) -> str:
    return f"{_f(p['mean'])} [{_f(p['lo'])}, {_f(p['hi'])}] ({p['n_pos']}/{p['n']} +)"


def summary_md(res: dict, missing: list, partial: bool) -> str:
    L = [f"# {RUN_ID} — 判定", ""]
    if partial:
        L += [f"**--partial で集計（欠けているジョブ: {missing}）。登録判定ではない。**", ""]
    L += [f"有効 seed {res['n_valid']}（無効: {res['invalid_seeds'] or 'なし'}）", ""]
    nat = res.get("nat")
    if res["label"] == "INAPPLICABLE":
        L += [f"**INAPPLICABLE** — {res['why']}", ""]
    else:
        B = res["B"]
        L += ["## 適用条件 (B)", "",
              f"- 自然な学習能力の差 N2 − N10: {_ci(B['natural_gap'])}",
              f"- 固定した場の沈降 N2r − S2_10r: {_ci(B['fixed_sink'])}",
              f"- 応答の床が固定場より下 S2_10r − S2u30r: {_ci(B['floor_below_fixed'])}",
              f"- 主腕の場が動いた（課題 1 の場の平均が下がった seed）: "
              f"{sum(l < f for f, l in B['field_mean_start_end'].values())}/{res['n_valid']} → "
              f"{'OK' if B['field_moved'] else 'NG'}", "",
              "## 主判定（95%・seed 内の差）", "",
              "| 量 | 比較 | 平均 [95%] |", "|---|---|---|",
              f"| M（動きの寄与） | S2_10r − S2dyn_10r | {_ci(res['M'])} |",
              f"| R（応答の床からの残り） | S2dyn_10r − S2u30r | {_ci(res['R'])} |",
              f"| A（設計案の字義の参照） | S2dyn_10r − N10r | {_ci(res['A'])} |", "",
              f"**主ラベル: {res['label']}**（設計案の字義の規則では {res['literal_label']}）", "",
              f"媒介率（報告のみ）: ρ_dyn = (N2r − S2dyn_10r)/(N2r − N10r) = {_f(res['rho']['rho_dyn'], 3)}、"
              f"固定場の ρ = {_f(res['rho']['rho_fixed'], 3)}、応答の床に対する ρ* = (N2r − S2dyn_10r)/(N2r − S2u30r) = "
              f"{_f(res['rho']['rho_star'], 3)}（N2r − N10r = {_f(res['rho']['gap_fresh'])}）", "",
              "## 副判定: 復元の超過", "",
              f"- 超過 R2_10 − N2: {_ci(res['excess'])}",
              f"- X = R2dyn_10 − N2: {_ci(res['X'])}", "",
              f"**{res['sub_label']}**", ""]
        nf = res["neff"]
        L += ["## 機能的応答（n̄_eff）の予測", "",
              f"適合: 旧 {nf['n_fit']} 腕（S1u20r 除く）、E = {nf['b0']:+.3f} {nf['b1']:+.3f}·log10 n̄（n̄ は 1/1200 で下限）、"
              f"残差 SD {nf['s']:.3f}、R² {nf['r2']:.2f}、1 つ抜きの 95% 予測区間の被覆 {nf['loo_coverage']:.2f}", "",
              "| 新腕 | log10 n̄ | 予測 [95%] | 観測 E | 区間内 |", "|---|---|---|---|---|"]
        for arm, q in nf["arms"].items():
            L.append(f"| {arm} | {q['x']:+.2f} | {q['pred']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}] | {q['E']:.4f} | "
                     f"{'○' if q['inside'] else '×'} |")
        ns = res["neff_start"]
        L += ["", f"**{res['neff_label']}**（登録は S2dyn_10r の 1 点）。開始時の n_eff で同じことをすると（報告のみ）: "
              f"R² {ns['r2']:.2f}・S2dyn_10r の予測 {ns['arms'][MAIN]['pred']:.3f} "
              f"[{ns['arms'][MAIN]['lo']:.3f}, {ns['arms'][MAIN]['hi']:.3f}]", ""]
    if nat:
        L += ["## 併走: 訓練の微分が 0 にならない自然軌道", "",
              "| 変種 | 有効 seed | D_E = E₁₁ − 中点 | D_late = E₁₈₋₂₂ − 中点 | ラベル |", "|---|---|---|---|---|"]
        for v, q in nat.items():
            if "D_E" in q:
                L.append(f"| {v}{'（登録）' if v == NAT_REGISTERED else ''} | {q['n_valid']} | {_ci(q['D_E'])} | "
                         f"{_ci(q['D_late'])} | {q['label']} |")
            else:
                L.append(f"| {v} | {q['n_valid']} | — | — | {q['label']} |")
        L += ["", "| 変種 | seed | E3 | E11 | E 後半 | T½ | T½(f32) | z̄₂ t10 | 0 の割合 t10 | 2⁻²⁴ 未満 t10 | log10 φ′ 中央 t10 | ε 域 W2 t10 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v, q in nat.items():
            for r in q["rows"]:
                L.append(f"| {v} | {r['seed']} | {r['E3']:.3f} | {r['E11']:.3f} | {r['E_late']:.3f} | {r['T_half']} | "
                         f"{r['T_half_32']} | {r['zbar2_t10']:+.1f} | {r['zero2_t10']:.3f} | {r['below_2p24_t10']:.3f} | "
                         f"{r['log10_g2_median_t10']:+.1f} | {r['eps_w2_t10']:.3f} |")
        L.append("")
    if res["label"] == "INAPPLICABLE":
        return "\n".join(L) + "\n"
    L += ["## 予測との照合", "", "| 項目 | 設計案（Claude） | 実装時（Claude） | 観測 |", "|---|---|---|---|"]
    P = res["predictions"]

    def cell(who, key):
        q = P[who].get(key)
        if q is None:
            return "—"
        return f"{q['pred']} {'○' if q['hit'] else '×'}"
    for key, label in (("E_S2dyn_10r", "S2dyn_10r の E"), ("E_R2dyn_10", "R2dyn_10 の E"), ("E_S2ramp_r", "S2ramp_r の E"),
                       ("E_R2_10_noanchor", "R2_10_noanchor の E"), ("E_S2dyn_5r", "S2dyn_5r の E"),
                       ("E_S2dyn_20r", "S2dyn_20r の E")):
        q = P["impl"].get(key) or P["design"].get(key)
        L.append(f"| {label}（腕の 95% 区間に入れば ○） | {cell('design', key)} | {cell('impl', key)} | "
                 f"{q['obs']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}] |")
    for key, label in (("label", "主ラベル（本 spec の規則）"), ("literal_label", "主ラベル（設計案の字義の規則）"),
                       ("sub", "副判定"), ("nat", "併走 f64_felu"), ("neff", "n̄_eff の予測")):
        obs = P["impl"][key]["obs"] if key in P["impl"] else P["design"][key]["obs"]
        extra = ""
        if key in ("label", "literal_label"):
            extra = f"（観測ラベルに置いた確率: 設計案 {P['design'][key]['prob_of_obs']:.0%}・実装時 {P['impl'][key]['prob_of_obs']:.0%}）"
        L.append(f"| {label} | {cell('design', key)} | {cell('impl', key)} | {obs}{extra} |")
    L += ["", "## 副比較（95%・報告のみ）", "", "| 比較 | 内容 | k | 平均 [95%] |", "|---|---|---|---|"]
    L += [f"| {r['contrast']} | {r['what']} | {r['k']} | {_ci(r)} |" for r in res["secondary"]]
    t = res["arm_table"]
    L += ["", "## 腕ごと（seed 平均・k=1 は分岐後 1 タスク目）", "",
          "| 腕 | E k1 [95%] | E k2 | 床 | n̄_eff k1 | n_eff 開始 | 0 の割合 開始→平均→終了 | 実効引数 開始→終了 | 場の移動 | 網自身の移動 | 歩幅 W2 | resp_ee と bit 一致 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['arm']} | {r['E_k1']:.4f} [{r['E_k1_lo']:.3f}, {r['E_k1_hi']:.3f}] | {r['E_k2']:.4f} | {r['floor_k1']} | "
                 f"{r['nbar_neffT2_k1']:.4f} | {r['neffT2_start']:.4f} | {r['zero2_start']:.3f}→{r['probe_zero2_mean_k1']:.3f}→"
                 f"{r['zero2_end_k1']:.3f} | {r['effbar2_first_k1']:+.1f}→{r['effbar2_last_k1']:+.1f} | {r['dmean2_move_k1']:+.2f} | "
                 f"{r['zarm2_move_k1']:+.2f} | {r['step_w2_k1']:.3f} | "
                 f"{'—' if r['resp_ee_hash_match'] is None or (isinstance(r['resp_ee_hash_match'], float) and math.isnan(r['resp_ee_hash_match'])) else str(int(r['resp_ee_hash_match'])) + '/' + str(2 * res['n_valid'])} |")
    L += ["", f"resp_ee_0917 の接頭部との一致（状態 sha256・22 タスク中）: {res['prefix_link']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(MAIN_SRC))
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--prereg", default=None, help="required prereg_commit (default: the runner's)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args(argv)
    src = Path(args.src)
    out = Path(args.out) if args.out else src
    shards, missing = load(src)
    nat, nmiss = load_nat(src)
    miss = missing + nmiss
    if miss and not args.partial:
        raise SystemExit(f"jobs without provenance: {miss} (use --partial deliberately)")
    prereg = args.prereg
    if prereg is None:
        txt = (REPO / "src" / "respdyn_ee_0917.py").read_text()
        prereg = next((ln.split('"')[1] for ln in txt.splitlines() if ln.startswith("PREREG_COMMIT = \"")), None)
    ref32 = load_ref32(sorted({s for (_, s) in nat}))
    res = analyze(shards, nat, ref32, prereg, args.epochs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary_md(res, miss, bool(miss)))
    if res["label"] != "INAPPLICABLE":
        res["arm_table"].to_csv(out / "arm_table.csv", index=False)
        pd.DataFrame(res["secondary"]).to_csv(out / "secondary.csv", index=False)
        res["prefix"].to_csv(out / "prefix_mean.csv")
        pd.DataFrame([{"quantity": k, **res[k]} for k in ("M", "R", "A", "X", "excess")]
                     + [{"quantity": f"B_{k}", **res["B"][k]} for k in B_PAIRS]).to_csv(out / "verdict.csv", index=False)
        pd.DataFrame([{"version": "trajectory", "arm": a, **q} for a, q in res["neff"]["arms"].items()]
                     + [{"version": "start", "arm": a, **q} for a, q in res["neff_start"]["arms"].items()]
                     ).to_csv(out / "neff_test.csv", index=False)
    if res.get("nat"):
        pd.DataFrame([r for q in res["nat"].values() for r in q["rows"]]).to_csv(out / "nat_table.csv", index=False)
    slim = {k: v for k, v in res.items() if k not in ("arm_table", "prefix")}
    slim["missing"] = miss
    (out / "verdict.json").write_text(json.dumps(slim, indent=2, default=str))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
