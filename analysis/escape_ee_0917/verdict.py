#!/usr/bin/env python3
"""Verdict for escape_ee_0917 (specs/spec_escape_ee_0917.md sections 4-5).

    python3 analysis/escape_ee_0917/verdict.py [--src results/escape_ee_0917/runs]

E(arm) = online accuracy of the first continuation task; climb(arm) = the host's own second-layer mean
preactivation at the end of that task minus at its start (probe).  Per seed:
  R_none = E(S2dyn_10r) - E(S2u30r)          respdyn's remainder (reproduced)
  R_c12  = E(S2dyn_c12) - E(S2u30r_c12)      the same host, room to grow removed
  dR     = R_none - R_c12                    (primary, with R_c12)
  dclimb = climb(S2dyn_10r) - climb(S2dyn_c12)   the manipulation check
  I      = E(N2r) - E(N2r_c12)               does the hold itself cost the healthy host?
  dR_fix = (E(S2_10r) - E(S2u30r)) - (E(S2_10r_c12) - E(S2u30r_c12))
  rho    = Spearman(climb, E) over the dyn arms none, c1, c2, c12, c12b
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

RUN_ID = "escape_ee_0917"
SEEDS = tuple(range(10))
ARMS = ("S2dyn_10r", "S2dyn_c1", "S2dyn_c2", "S2dyn_c12", "S2dyn_c12b", "S2dyn_frz", "S2u30r", "S2u30r_c12",
        "S2_10r", "S2_10r_c12", "N2r", "N2r_c12")
LADDER = ("S2dyn_10r", "S2dyn_c1", "S2dyn_c2", "S2dyn_c12", "S2dyn_c12b")     # rho is taken over these
DYN = ("S2dyn_10r", "S2dyn_c1", "S2dyn_c2", "S2dyn_c12", "S2dyn_c12b", "S2dyn_frz")
REPRO = {"S2dyn_10r": ("respdyn",), "S2u30r": ("respdyn", "resp_ee"), "S2_10r": ("respdyn", "resp_ee"),
         "N2r": ("respdyn", "resp_ee")}
HELD = {"S2dyn_c1": ("exc_q", "exc_v"), "S2dyn_c2": ("exc_w2",), "S2dyn_c12": ("exc_q", "exc_v", "exc_w2"),
        "S2dyn_c12b": ("exc_q", "exc_v", "exc_w2", "exc_b"), "S2dyn_frz": ("exc_frz",),
        "S2u30r_c12": ("exc_q", "exc_v", "exc_w2"), "S2_10r_c12": ("exc_q", "exc_v", "exc_w2"),
        "N2r_c12": ("exc_q", "exc_v", "exc_w2")}
BIND = ("S2dyn_c12",)                           # spec 5.2 (C): the primary hold must write in every task
PREFIX_T = 12
MIN_SEEDS = 8
MAIN_LEVEL = 0.975                              # dR and R_c12
COMP_LEVEL = 0.95
TOL_CAP = 1e-4                                  # 4 * eps32 * R, R <= 200 (as swap_ee_0917)
SIGN_ALPHA = 0.05


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _true(v) -> bool:
    return v is True or v == "True"


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
            if not (math.isfinite(_f(r["online_acc"])) and math.isfinite(_f(r["climb2"]))):
                return f"{a} k{k} non-finite"
        if not _true(rows[(a, 1)].get("logits_equal_full")):
            return f"{a}: branch-point logits differ"
    if len(sd["prefix"]) != PREFIX_T or not all(_true(r["hash_match_resp_ee"]) for r in sd["prefix"]):
        return "prefix is not resp_ee's"
    for k in (1, 2):
        for a, srcs in REPRO.items():
            for src in srcs:
                if not _true(rows[(a, k)].get(f"hash_match_{src}")):
                    return f"{a} k{k} is not {src}'s"
        for a in DYN:
            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):
                return f"{a} k{k}: shadow is not N10"
        for a, cols in HELD.items():
            r = rows[(a, k)]
            if max(_f(r.get(c)) for c in cols) > TOL_CAP or any(not math.isfinite(_f(r.get(c))) for c in cols):
                return f"{a} k{k}: outside its hold"
        for a in BIND:
            r = rows[(a, k)]
            if _f(r["rows_par"]) + _f(r["rows_perp"]) + _f(r["rows_w2"]) <= 0:
                return f"{a} k{k}: the hold never wrote"
    for a in DYN:
        if not _true(rows[(a, 1)].get("field0_equal_fixed")):
            return f"{a}: d(0) is not the fixed field"
    return None


def spearman(a, b) -> float:
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    sa, sb = ra.std(), rb.std()
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def sign_test(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = max(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def analyze(seeds: dict) -> dict:
    invalid = {s: why for s in SEEDS if (why := (seed_validity(seeds[s]) if s in seeds else "missing"))}
    valid = [s for s in SEEDS if s not in invalid]
    E = {(s, r["arm"], int(r["k"])): _f(r["online_acc"]) for s in valid for r in seeds[s]["arms"]}
    C = {(s, r["arm"], int(r["k"])): _f(r["climb2"]) for s in valid for r in seeds[s]["arms"]}

    def e(a, k=1):
        return np.array([E[(s, a, k)] for s in valid])

    def c(a, k=1):
        return np.array([C[(s, a, k)] for s in valid])

    q = {}
    for k in (1, 2):
        q[("R_none", k)] = e("S2dyn_10r", k) - e("S2u30r", k)
        q[("R_c12", k)] = e("S2dyn_c12", k) - e("S2u30r_c12", k)
        q[("dR", k)] = q[("R_none", k)] - q[("R_c12", k)]
        q[("R_c1", k)] = e("S2dyn_c1", k) - e("S2u30r", k)
        q[("R_c2", k)] = e("S2dyn_c2", k) - e("S2u30r", k)
        q[("R_c12b", k)] = e("S2dyn_c12b", k) - e("S2u30r", k)
        q[("R_frz", k)] = e("S2dyn_frz", k) - e("S2u30r", k)
        q[("dclimb", k)] = c("S2dyn_10r", k) - c("S2dyn_c12", k)
        q[("I", k)] = e("N2r", k) - e("N2r_c12", k)
        q[("dR_fix", k)] = (e("S2_10r", k) - e("S2u30r", k)) - (e("S2_10r_c12", k) - e("S2u30r_c12", k))
        q[("dR_b", k)] = q[("R_c12", k)] - q[("R_c12b", k)]
    st = {(n, k, lv): paired(v, lv) for (n, k), v in q.items() for lv in (MAIN_LEVEL, COMP_LEVEL)}
    rho = [spearman([C[(s, a, 1)] for a in LADDER], [E[(s, a, 1)] for a in LADDER]) for s in valid]
    pos = sum(r > 0 for r in rho if math.isfinite(r))
    neg = sum(r < 0 for r in rho if math.isfinite(r))
    p_rho = sign_test(max(pos, neg), pos + neg)

    A_ok = len(valid) >= MIN_SEEDS
    B_ok = A_ok and st[("R_none", 1, COMP_LEVEL)]["lo"] > 0
    C_ok = A_ok and st[("dclimb", 1, COMP_LEVEL)]["lo"] > 0

    def primary():
        if not A_ok:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        if not C_ok:
            return "ESCAPE_NOT_MANIPULATED"
        sd_, sc = st[("dR", 1, MAIN_LEVEL)]["sign"], st[("R_c12", 1, MAIN_LEVEL)]["sign"]
        if sd_ == "+":
            return "REMAINDER_REMOVED_BY_HOLD" if sc != "+" else "REMAINDER_REDUCED_BY_HOLD"
        return "HOLD_LEAVES_REMAINDER" if sd_ == "0" else "HOLD_RAISES_REMAINDER"

    Ist = st[("I", 1, COMP_LEVEL)]
    dRm = st[("dR", 1, COMP_LEVEL)]["mean"]
    impaired = bool(A_ok and Ist["sign"] == "+" and Ist["mean"] >= dRm)
    labels = {
        "primary": primary(),
        "climb_tracks": ("INAPPLICABLE" if not A_ok else
                         "CLIMB_TRACKS" if (pos > neg and p_rho < SIGN_ALPHA) else
                         "CLIMB_OPPOSES" if (neg > pos and p_rho < SIGN_ALPHA) else "CLIMB_UNRESOLVED"),
        "fixed_field": ("INAPPLICABLE" if not A_ok else
                        {"+": "FIX_REMAINDER_REDUCED", "0": "FIX_REMAINDER_UNCHANGED",
                         "-": "FIX_REMAINDER_RAISED"}[st[("dR_fix", 1, COMP_LEVEL)]["sign"]]),
        "impairment_flag": "HOLD_IMPAIRS" if impaired else "NO_IMPAIRMENT_FLAG",
    }
    arm_means = {a: {k: float(np.mean(e(a, k))) if valid else float("nan") for k in (1, 2)} for a in ARMS}
    climb_means = {a: {k: float(np.mean(c(a, k))) if valid else float("nan") for k in (1, 2)} for a in ARMS}
    report = {}
    for a in ARMS:
        for col in ("w1norm_end", "w2norm_end", "b2_mean_end", "probe_zero2_mean", "nbar_neffT2",
                    "probe_effbar2_750", "probe_effbar2_last", "probe_zarm2_750", "rows_w2", "rows_perp", "restored"):
            vals = [_f(r.get(col)) for s in valid for r in seeds[s]["arms"] if r["arm"] == a and r["k"] == "1"]
            if vals and any(math.isfinite(v) for v in vals):
                report[f"{a}|{col}"] = float(np.nanmean(vals))
    return {"valid_seeds": valid, "invalid_seeds": invalid, "A_ok": A_ok, "B_ok": B_ok, "C_ok": C_ok,
            "labels": labels, "stats": st, "per_seed": {f"{n}|{k}": v.tolist() for (n, k), v in q.items()},
            "rho": rho, "rho_pos": pos, "rho_neg": neg, "rho_p": p_rho, "arm_means": arm_means,
            "climb_means": climb_means, "report": report}


def _fmt(x, nd=4):
    return "nan" if x is None or not math.isfinite(x) else f"{x:+.{nd}f}"


def summary_md(res: dict, env: dict) -> str:
    st, n = res["stats"], len(res["valid_seeds"])

    def line(name, k, lv):
        x = st[(name, k, lv)]
        return (f"| {name} | {k} | {lv} | {_fmt(x['mean'])} | {_fmt(x['sd'])} | [{_fmt(x['lo'])}, {_fmt(x['hi'])}] | "
                f"{x['sign']} | {sum(v > 0 for v in res['per_seed'][f'{name}|{k}'])}/{n} |")

    L = [f"# {RUN_ID} — 判定（t2 の網の逃げ場を塞いで respdyn の主腕を回し直す）", "",
         "> 自動生成: `analysis/escape_ee_0917/verdict.py`。spec: `specs/spec_escape_ee_0917.md`。", "",
         f"- HEAD `{env['head']}`・run の commit {env['run_commits']}・有効 seed {n}（無効: {res['invalid_seeds'] or 'なし'}）",
         f"- (A) {res['A_ok']}・(B) R_none > 0: {res['B_ok']}・(C) 逃げが減った dclimb > 0: {res['C_ok']}", "",
         "## 判定", "", "| 問い | ラベル |", "|---|---|"]
    L += [f"| {k} | **{v}** |" for k, v in res["labels"].items()]
    L += ["", f"登りと E の順位相関（LADDER 5 腕・seed ごと）: 平均 {np.nanmean(res['rho']):+.3f}、正 {res['rho_pos']}・負 {res['rho_neg']}、"
          f"符号検定 p = {res['rho_p']:.3g}。seed 別: " + ", ".join(f"{r:+.2f}" for r in res["rho"]), "",
          "## 量（seed 内の差）", "", "| 量 | 継続 | 水準 | 平均 | SD | 区間 | 符号 | 正の seed |",
          "|---|---|---|---|---|---|---|---|"]
    for name in ("dR", "R_c12"):
        L.append(line(name, 1, MAIN_LEVEL))
    for name in ("R_none", "R_c12", "dR", "R_c1", "R_c2", "R_c12b", "R_frz", "dR_b", "dclimb", "I", "dR_fix"):
        L.append(line(name, 1, COMP_LEVEL))
    for name in ("R_none", "R_c12", "dR", "dclimb", "I", "dR_fix"):
        L.append(line(name, 2, COMP_LEVEL))
    L += ["", "## 腕ごとの E と網自身の登り（10 seed 平均）", "", "| 腕 | E 継続 1 | E 継続 2 | 登り 継続 1 | 登り 継続 2 |",
          "|---|---|---|---|---|"]
    for a in ARMS:
        L.append(f"| {a} | {res['arm_means'][a][1]:.4f} | {res['arm_means'][a][2]:.4f} | "
                 f"{res['climb_means'][a][1]:+.3f} | {res['climb_means'][a][2]:+.3f} |")
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
        for (name, k, lv), x in res["stats"].items():
            w.writerow([name, k, lv, x["n"], x["mean"], x["sd"], x["lo"], x["hi"], x["sign"]])
    with (out / "paired.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "k", "seed", "value"])
        for key, vals in res["per_seed"].items():
            name, k = key.split("|")
            for s, v in zip(res["valid_seeds"], vals):
                w.writerow([name, k, s, v])
        for s, r in zip(res["valid_seeds"], res["rho"]):
            w.writerow(["rho_climb_E", 1, s, r])
    (out / "summary.md").write_text(summary_md(res, env))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
        "run_commits": env["run_commits"], "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"],
        "A_ok": res["A_ok"], "B_ok": res["B_ok"], "C_ok": res["C_ok"], "labels": res["labels"],
        "rho": res["rho"], "rho_p": res["rho_p"], "arm_means": res["arm_means"], "climb_means": res["climb_means"],
        "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "judgement": {"main_level": MAIN_LEVEL, "comp_level": COMP_LEVEL, "min_seeds": MIN_SEEDS,
                      "tol_cap": TOL_CAP, "sign_alpha": SIGN_ALPHA, "ladder": LADDER},
    }, indent=2, default=str))
    print(f"labels: {res['labels']}  valid {len(res['valid_seeds'])}  invalid {res['invalid_seeds']}")


if __name__ == "__main__":
    main()
