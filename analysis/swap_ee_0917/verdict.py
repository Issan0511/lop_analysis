#!/usr/bin/env python3
"""Verdict for swap_ee_0917 (specs/spec_swap_ee_0917.md sections 4-5).

    python3 analysis/swap_ee_0917/verdict.py [--src results/swap_ee_0917/runs]

E(arm) = the online accuracy of the arm's first continuation task (resp_ee_0917's E).  Per seed:
  Q1  R_C = E(SC_dyn_r) - E(SC_u30_r)      the remainder above the floor in the host that cannot push
      R_R = E(NR_r)     - E(RR_u30_r)      the same in the host that pushes (its own collapsing field)
      D   = R_C - R_R
  Q2  TE  = E(NC_r) - E(NR_r);  L_S = E(NC_r) - E(SC_dyn_r);  G_R = E(RR_dyn_r) - E(NR_r)
  Q3  P_C = E(SC_dyn_r) - E(SC_dyn_free_r);  P_R = E(RR_dyn_cap_r) - E(RR_dyn_r)
      P_N = E(NR_cap_r) - E(NR_r);           P_F = E(NC_r) - E(NC_free_r)
  report R_T2 = E(S2dyn_10r) - E(S2u30r) (respdyn_ee_0917's R, re-run).
Paired t intervals across valid seeds; the Student-t machinery is mucap_el_0916's (imported).
analyze() is pure so that checks.py S8 can run it on synthetic seeds.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_mucap_el_verdict", REPO / "analysis" / "mucap_el_0916" / "verdict.py")
ELV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ELV)
paired, t_quantile = ELV.paired, ELV.t_quantile

RUN_ID = "swap_ee_0917"
SEEDS = tuple(range(10))
ARMS = ("NC_r", "NC_free_r", "SC_fix_r", "SC_dyn_r", "SC_dyn_free_r", "SC_u30_r",
        "NR_r", "NR_cap_r", "RR_fix_r", "RR_dyn_r", "RR_dyn_cap_r", "RR_u30_r", "S2dyn_10r", "S2u30r")
CAPPED = ("NC_r", "SC_fix_r", "SC_dyn_r", "SC_u30_r", "NR_cap_r", "RR_dyn_cap_r")   # caps must hold
BINDING = ("NC_r", "SC_dyn_r", "NR_cap_r", "RR_dyn_cap_r")                         # and must write rows
DYN_SWAP = ("SC_dyn_r", "SC_dyn_free_r", "RR_dyn_r", "RR_dyn_cap_r")
PREFIX_T = 12
MIN_SEEDS = 8                                   # spec 5.2 (A)
MAIN_LEVEL = 0.975                              # Q1: two quantities (Bonferroni)
COMP_LEVEL = 0.95
TOL_CAP = 1e-4                                  # spec 5.2 (C): 4 * eps32 * a radius of at most 200


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _true(v) -> bool:
    return v is True or v == "True"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(src: Path) -> dict:
    seeds = {}
    for s in SEEDS:
        d = src / f"s{s}"
        if not (d / "provenance.json").exists():
            continue
        with (d / "arms.csv").open() as fh:
            arms = list(csv.DictReader(fh))
        with (d / "prefix.csv").open() as fh:
            prefix = list(csv.DictReader(fh))
        seeds[s] = {"arms": arms, "prefix": prefix, "prov": json.loads((d / "provenance.json").read_text())}
    return seeds


def seed_validity(sd: dict) -> str | None:
    rows = {(r["arm"], int(r["k"])): r for r in sd["arms"]}
    for a in ARMS:
        for k in (1, 2):
            r = rows.get((a, k))
            if r is None:
                return f"{a} k{k} missing"
            if not math.isfinite(_f(r["online_acc"])):
                return f"{a} k{k} non-finite"
    pr = sd["prefix"]
    ref = [r for r in pr if r["net"] == "R"]
    cap = [r for r in pr if r["net"] == "C"]
    if len(ref) != PREFIX_T or not all(_true(r["hash_match_resp_ee"]) for r in ref):
        return "ref prefix is not resp_ee's"
    if len(cap) != PREFIX_T or not all(_true(r["match_l2cap_units"]) for r in cap):
        return "cap12 prefix is not l2cap_ee_0917's"
    for a in ARMS:
        r1 = rows[(a, 1)]
        if not _true(r1.get("logits_equal_full")):
            return f"{a}: branch-point logits differ"
    for a in DYN_SWAP:
        for k in (1, 2):
            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):
                return f"{a} k{k}: shadow is not the natural continuation"
        if not _true(rows[(a, 1)].get("field0_equal_fixed")):
            return f"{a}: d(0) is not the fixed field"
    for k in (1, 2):
        if not _true(rows[("S2dyn_10r", k)].get("shadow_hash_match_prefix")):
            return "S2dyn_10r shadow is not N10"
        for a, col in (("NR_r", "hash_match_resp_ee"), ("S2u30r", "hash_match_resp_ee"),
                       ("S2dyn_10r", "hash_match_respdyn"), ("S2u30r", "hash_match_respdyn")):
            if not _true(rows[(a, k)].get(col)):
                return f"{a} k{k}: {col} false"
    for a in CAPPED:
        for k in (1, 2):
            r = rows[(a, k)]
            if max(_f(r["exc_q"]), _f(r["exc_v"]), _f(r["exc_w2"])) > TOL_CAP:
                return f"{a} k{k}: above its caps"
    for a in BINDING:
        for k in (1, 2):
            r = rows[(a, k)]
            if _f(r["rows_par"]) + _f(r["rows_perp"]) + _f(r["rows_w2"]) <= 0:
                return f"{a} k{k}: caps never wrote"
    return None


# --------------------------------------------------------------------------

def analyze(seeds: dict) -> dict:
    invalid = {s: why for s in SEEDS if (why := (seed_validity(seeds[s]) if s in seeds else "missing"))}
    valid = [s for s in SEEDS if s not in invalid]
    E = {(s, a, k): _f(r["online_acc"]) for s in valid for r in seeds[s]["arms"]
         for a, k in [(r["arm"], int(r["k"]))]}

    def e(a, k=1):
        return np.array([E[(s, a, k)] for s in valid])

    q = {}
    for k in (1, 2):
        q[("R_C", k)] = e("SC_dyn_r", k) - e("SC_u30_r", k)
        q[("R_R", k)] = e("NR_r", k) - e("RR_u30_r", k)
        q[("D", k)] = q[("R_C", k)] - q[("R_R", k)]
        q[("R_T2", k)] = e("S2dyn_10r", k) - e("S2u30r", k)
        q[("R_C_minus_R_T2", k)] = q[("R_C", k)] - q[("R_T2", k)]
        q[("TE", k)] = e("NC_r", k) - e("NR_r", k)
        q[("L_S", k)] = e("NC_r", k) - e("SC_dyn_r", k)
        q[("L_S_fix", k)] = e("NC_r", k) - e("SC_fix_r", k)
        q[("G_R", k)] = e("RR_dyn_r", k) - e("NR_r", k)
        q[("G_R_fix", k)] = e("RR_fix_r", k) - e("NR_r", k)
        q[("M_C", k)] = e("SC_fix_r", k) - e("SC_dyn_r", k)
        q[("P_C", k)] = e("SC_dyn_r", k) - e("SC_dyn_free_r", k)
        q[("P_R", k)] = e("RR_dyn_cap_r", k) - e("RR_dyn_r", k)
        q[("P_N", k)] = e("NR_cap_r", k) - e("NR_r", k)
        q[("P_F", k)] = e("NC_r", k) - e("NC_free_r", k)
    st = {}
    for (name, k), v in q.items():
        for level in (MAIN_LEVEL, COMP_LEVEL):
            st[(name, k, level)] = paired(v, level)

    A_ok = len(valid) >= MIN_SEEDS
    B = {"TE": st[("TE", 1, 0.95)], "R_T2": st[("R_T2", 1, 0.95)]}
    B_ok = A_ok and all(b["lo"] > 0 for b in B.values())

    def lab_q1():
        if not A_ok:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        sc, sd = st[("R_C", 1, MAIN_LEVEL)]["sign"], st[("D", 1, MAIN_LEVEL)]["sign"]
        if sc != "+":
            return "NO_REMAINDER_IN_NONPUSHING_HOST"
        return {"+": "REMAINDER_FOLLOWS_HOST", "0": "REMAINDER_IN_BOTH",
                "-": "REMAINDER_LARGER_IN_PUSHING_HOST"}[sd]

    def lab_q2():
        if not A_ok:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        ls, gr = st[("L_S", 1, COMP_LEVEL)]["sign"] == "+", st[("G_R", 1, COMP_LEVEL)]["sign"] == "+"
        return {(True, True): "BOTH_WAYS", (True, False): "SINK_ONLY", (False, True): "RESTORE_ONLY",
                (False, False): "NEITHER"}[(ls, gr)]

    def lab_toggle(name):
        if not A_ok:
            return "INAPPLICABLE"
        return {"+": "GROWTH_COSTS", "-": "GROWTH_HELPS", "0": "NO_EFFECT"}[st[(name, 1, COMP_LEVEL)]["sign"]]

    labels = {"Q1": lab_q1(), "Q2": lab_q2(), **{f"Q3_{n}": lab_toggle(n) for n in ("P_C", "P_R", "P_N", "P_F")}}
    te = st[("TE", 1, 0.95)]["mean"]
    ratios = {"rho_sink_dyn": st[("L_S", 1, 0.95)]["mean"] / te if te else float("nan"),
              "rho_sink_fix": st[("L_S_fix", 1, 0.95)]["mean"] / te if te else float("nan"),
              "rho_restore_dyn": st[("G_R", 1, 0.95)]["mean"] / te if te else float("nan"),
              "rho_restore_fix": st[("G_R_fix", 1, 0.95)]["mean"] / te if te else float("nan")}
    arm_means = {a: {k: float(np.mean(e(a, k))) if valid else float("nan") for k in (1, 2)} for a in ARMS}
    report = {}
    for a in ARMS:
        for col in ("probe_zarm2_first", "probe_zarm2_750", "probe_zarm2_last", "probe_dmean2_first",
                    "probe_dmean2_750", "probe_dmean2_last", "probe_effbar2_first", "probe_effbar2_750",
                    "probe_effbar2_last", "nbar_neffT2", "probe_zero2_mean", "w2norm_end", "exc_w2_own",
                    "exc_v_own", "exc_q_own", "zero2_start", "zero2_end", "mu2_end", "memo_acc_end"):
            vals = [_f(r.get(col)) for s in valid for r in seeds[s]["arms"] if r["arm"] == a and r["k"] == "1"]
            if vals and any(math.isfinite(v) for v in vals):
                report[f"{a}|{col}"] = float(np.nanmean(vals))
    return {"valid_seeds": valid, "invalid_seeds": invalid, "A_ok": A_ok, "B": B, "B_ok": B_ok,
            "labels": labels, "stats": st, "per_seed": {f"{n}|{k}": v.tolist() for (n, k), v in q.items()},
            "ratios": ratios, "arm_means": arm_means, "report": report}


# --------------------------------------------------------------------------

def _fmt(x, nd=4):
    return "nan" if x is None or not math.isfinite(x) else f"{x:+.{nd}f}"


def summary_md(res: dict, env: dict) -> str:
    st = res["stats"]
    n = len(res["valid_seeds"])

    def line(name, k, level):
        x = st[(name, k, level)]
        return (f"| {name} | {k} | {level} | {_fmt(x['mean'])} | {_fmt(x['sd'])} | "
                f"[{_fmt(x['lo'])}, {_fmt(x['hi'])}] | {x['sign']} | "
                f"{sum(v > 0 for v in res['per_seed'][f'{name}|{k}'])}/{n} |")

    L = [f"# {RUN_ID} — 判定（cap12 の網と ref の網で第2層の場を入れ替える）", "",
         "> 自動生成: `analysis/swap_ee_0917/verdict.py`。spec: `specs/spec_swap_ee_0917.md`。", "",
         f"- HEAD `{env['head']}`・run の commit {env['run_commits']}・有効 seed {n}（無効: {res['invalid_seeds'] or 'なし'}）",
         f"- (A) 有効 seed ≥ {MIN_SEEDS}: **{res['A_ok']}**・(B) TE {_fmt(res['B']['TE']['mean'])} "
         f"[{_fmt(res['B']['TE']['lo'])}, {_fmt(res['B']['TE']['hi'])}]・R_T2 {_fmt(res['B']['R_T2']['mean'])} "
         f"[{_fmt(res['B']['R_T2']['lo'])}, {_fmt(res['B']['R_T2']['hi'])}] → **{res['B_ok']}**", "",
         "## 判定", "", "| 問い | ラベル |", "|---|---|"]
    for k, v in res["labels"].items():
        L.append(f"| {k} | **{v}** |")
    L += ["", "## 量（seed 内の差）", "", "| 量 | 継続 | 水準 | 平均 | SD | 区間 | 符号 | 正の seed |",
          "|---|---|---|---|---|---|---|---|"]
    for name in ("R_C", "D", "R_R", "R_T2", "R_C_minus_R_T2"):
        L.append(line(name, 1, MAIN_LEVEL))
    for name in ("R_C", "D", "R_R", "R_T2", "R_C_minus_R_T2", "TE", "L_S", "L_S_fix", "G_R", "G_R_fix", "M_C",
                 "P_C", "P_R", "P_N", "P_F"):
        L.append(line(name, 1, COMP_LEVEL))
    for name in ("R_C", "D", "TE", "L_S", "G_R", "P_C", "P_R", "P_N", "P_F"):
        L.append(line(name, 2, COMP_LEVEL))
    L += ["", "比（TE に対する・報告）: " + "、".join(f"{k} {v:.3f}" for k, v in res["ratios"].items()), "",
          "## 腕ごとの E（10 seed 平均）", "", "| 腕 | 継続 1 | 継続 2 |", "|---|---|---|"]
    for a, v in res["arm_means"].items():
        L.append(f"| {a} | {v[1]:.4f} | {v[2]:.4f} |")
    L += ["", "## 報告（継続 1 タスク目・seed 平均）", "", "| 腕 | 量 | 値 |", "|---|---|---|"]
    for k, v in res["report"].items():
        a, col = k.split("|")
        L.append(f"| {a} | {col} | {v:.4g} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / RUN_ID / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / RUN_ID))
    args = ap.parse_args()
    seeds = load(Path(args.src))
    res = analyze(seeds)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {"head": head, "run_commits": sorted({sd["prov"].get("git_hash") for sd in seeds.values()})}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "verdict.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "k", "level", "n", "mean", "sd", "lo", "hi", "sign"])
        for (name, k, level), x in res["stats"].items():
            w.writerow([name, k, level, x["n"], x["mean"], x["sd"], x["lo"], x["hi"], x["sign"]])
    with (out / "paired.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "k", "seed", "value"])
        for key, vals in res["per_seed"].items():
            name, k = key.split("|")
            for s, v in zip(res["valid_seeds"], vals):
                w.writerow([name, k, s, v])
    (out / "summary.md").write_text(summary_md(res, env))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
        "run_commits": env["run_commits"], "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"],
        "A_ok": res["A_ok"], "B_ok": res["B_ok"], "labels": res["labels"], "ratios": res["ratios"],
        "arm_means": res["arm_means"],
        "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "judgement": {"main_level": MAIN_LEVEL, "comp_level": COMP_LEVEL, "min_seeds": MIN_SEEDS, "tol_cap": TOL_CAP},
    }, indent=2, default=str))
    print(f"labels: {res['labels']}  valid {len(res['valid_seeds'])}  invalid {res['invalid_seeds']}")


if __name__ == "__main__":
    main()
