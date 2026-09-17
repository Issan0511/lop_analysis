#!/usr/bin/env python3
"""Verdict for neffdir_ee_0918 (specs/spec_neffdir_ee_0918.md sections 4-5).

    python3 analysis/neffdir_ee_0918/verdict.py [--src results/neffdir_ee_0918/runs]

On seeds 30-39, first continuation task: E = online accuracy, F = E(S2u30r) (the response floor),
R(a) = E(a) - F, n(a) = nbar_neffS2(a) (the in-task effective number of images carrying the derivative the
arm trains with).  The early window uses early_acc (updates 0-1499) and nbar_neffS2_early (probes 0-1500).

  primary     add with growth held: dR_add_h = R(S2dyn_c12_a02) - R(S2dyn_c12) at 97.5%, checked by
              dn_add_h = n(c12_a02) - n(c12) > 0
  prop_*      the shortfall of an observed change against the proportional prediction
              R(base) * (kappa - 1), kappa = n(arm) / n(base) per seed, inside +-MARGIN
  secondaries the add on the free host; the drops (fixed masks) on the free and held hosts, whole task and
              early window; the compensation (raw carriers nbar_neffT2 up under a mask); the step mask
              (structure); F.elu (exact zeros); the within-seed ladder; the growth effect (escneff again)

Records: nothing may contradict resp_ee / respdyn / escape where they exist (no registered seed has one),
and every internal reproduction (shadows, branch logits, fields, holds, masks, lifts) must hold.
analyze() is pure.
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

RUN_ID = "neffdir_ee_0918"
SEEDS = tuple(range(30, 40))
ARMS = ("S2dyn_10r", "S2dyn_m50", "S2dyn_m25", "S2dyn_s50", "S2dyn_a02", "S2dyn_felu", "S2dyn_c12",
        "S2dyn_c12_m50", "S2dyn_c12_a02", "S2u30r", "N2r", "N2r_m50")
DYN = ("S2dyn_10r", "S2dyn_m50", "S2dyn_m25", "S2dyn_s50", "S2dyn_a02", "S2dyn_felu", "S2dyn_c12",
       "S2dyn_c12_m50", "S2dyn_c12_a02")
LADDER = ("S2dyn_m25", "S2dyn_m50", "S2dyn_10r", "S2dyn_a02")      # rho is taken over these
REPRO = {"S2dyn_10r": ("respdyn", "escape"), "S2dyn_c12": ("escape",), "S2u30r": ("resp_ee", "respdyn", "escape"),
         "N2r": ("resp_ee", "respdyn", "escape")}
HELD = {a: ("exc_q", "exc_v", "exc_w2") for a in ("S2dyn_c12", "S2dyn_c12_m50", "S2dyn_c12_a02")}
MASKED = {"S2dyn_m50": 0.5, "S2dyn_m25": 0.25, "S2dyn_c12_m50": 0.5, "N2r_m50": 0.5, "S2dyn_s50": 0.5}
TOP = {"S2dyn_a02": 0.02, "S2dyn_c12_a02": 0.02}
PREFIX_T = 12
MIN_SEEDS = 8
MAIN_LEVEL = 0.975                              # the primary (dR_add_h)
COMP_LEVEL = 0.95
TOL_CAP = 1e-4                                  # 4 * eps32 * R, R <= 200 (as swap / escape)
SIGN_ALPHA = 0.05
MARGIN = 0.03                                   # spec 5.3: half the marker model's shortfall at q = 0.5
KEEP_TOL = 0.01                                 # > 6 binomial sd of a share of 120000 pairs (q = 0.5: 0.0087)
LIFT_TOL = 0.003                                # > 6 binomial sd at r = 0.02 (0.0024)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _true(v) -> bool:
    return v is True or v == "True"


def _ok_record(v) -> bool:
    return v in ("", "None", "True", None, True)


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
            if not all(math.isfinite(_f(r.get(c))) for c in ("online_acc", "nbar_neffS2", "probe_zeroS2_mean",
                                                             "nbar_neffT2", "early_acc", "nbar_neffS2_early")):
                return f"{a} k{k} non-finite"
        if not (_true(rows[(a, 1)].get("logits_equal_full")) and _true(rows[(a, 1)].get("logits_equal_mb"))):
            return f"{a}: branch-point logits differ"
    if len(sd["prefix"]) != PREFIX_T:
        return "prefix incomplete"
    for r in sd["prefix"]:
        if not _ok_record(r.get("hash_match_resp_ee", "")):
            return f"prefix task {r['task']} contradicts resp_ee"
    for k in (1, 2):
        for a, srcs in REPRO.items():
            for src in srcs:
                if not _ok_record(rows[(a, k)].get(f"hash_match_{src}", "")):
                    return f"{a} k{k} contradicts {src}"
        for a in DYN:
            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):
                return f"{a} k{k}: shadow is not N10"
        for a, cols in HELD.items():
            r = rows[(a, k)]
            if any(not math.isfinite(_f(r.get(c))) for c in cols) or max(_f(r.get(c)) for c in cols) > TOL_CAP:
                return f"{a} k{k}: outside its hold"
            if _f(r["rows_par"]) + _f(r["rows_perp"]) + _f(r["rows_w2"]) <= 0:
                return f"{a} k{k}: the hold never wrote"
        for a, q in MASKED.items():
            if not abs(_f(rows[(a, k)].get("keep_share")) - q) <= KEEP_TOL:
                return f"{a} k{k}: keep share off"
    for a in DYN:
        if not _true(rows[(a, 1)].get("field0_equal_fixed")):
            return f"{a}: d(0) is not the fixed field"
    for a, r_ in TOP.items():
        r = rows[(a, 1)]
        if not (_true(r.get("field0_top_ok")) and abs(_f(r.get("lift_share")) - r_) <= LIFT_TOL):
            return f"{a}: the lifted field is not as built"
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


def proportion_label(x: dict) -> str:
    """x = paired() of the shortfall phi at COMP_LEVEL."""
    if not (math.isfinite(x["lo"]) and math.isfinite(x["hi"])):
        return "PROPORTION_UNRESOLVED"
    if -MARGIN < x["lo"] and x["hi"] < MARGIN:
        return "PROPORTIONAL"
    if x["lo"] > 0:
        return "SUBPROPORTIONAL"
    if x["hi"] < 0:
        return "SUPRAPROPORTIONAL"
    return "PROPORTION_UNRESOLVED"


def analyze(seeds: dict) -> dict:
    invalid = {s: why for s in SEEDS if (why := (seed_validity(seeds[s]) if s in seeds else "missing"))}
    valid = [s for s in SEEDS if s not in invalid]

    def col(name):
        return {(s, r["arm"], int(r["k"])): _f(r.get(name)) for s in valid for r in seeds[s]["arms"]}

    cols = {"E": col("online_acc"), "N": col("nbar_neffS2"), "T": col("nbar_neffT2"), "Z": col("probe_zeroS2_mean"),
            "Ee": col("early_acc"), "Ne": col("nbar_neffS2_early")}

    def v(c, a, k=1):
        return np.array([cols[c][(s, a, k)] for s in valid])

    def e(a, k=1):
        return v("E", a, k)

    def change(R, n, base, arm, sign):
        """(dR, dn, phi): sign +1 for an add (arm above base), -1 for a drop."""
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = n[arm] / n[base]
        dR = sign * (R[arm] - R[base])
        dn = sign * (n[arm] - n[base])
        phi = sign * (kap * R[base] - R[arm])            # shortfall of the observed change
        return dR, dn, phi, kap

    q = {}
    for k in (1, 2):
        F_ = e("S2u30r", k)
        R = {a: e(a, k) - F_ for a in DYN}
        n = {a: v("N", a, k) for a in DYN}
        q[("R_none", k)] = R["S2dyn_10r"]
        q[("R_c12", k)] = R["S2dyn_c12"]
        for tag, base, arm, sign in (("add_h", "S2dyn_c12", "S2dyn_c12_a02", 1), ("add", "S2dyn_10r", "S2dyn_a02", 1),
                                     ("drop_h", "S2dyn_c12", "S2dyn_c12_m50", -1),
                                     ("drop", "S2dyn_10r", "S2dyn_m50", -1), ("m25", "S2dyn_10r", "S2dyn_m25", -1)):
            dR, dn, phi, kap = change(R, n, base, arm, sign)
            q[(f"dR_{tag}", k)], q[(f"dn_{tag}", k)], q[(f"phi_{tag}", k)], q[(f"kappa_{tag}", k)] = dR, dn, phi, kap
        q[("comp", k)] = v("T", "S2dyn_m50", k) - v("T", "S2dyn_10r", k)
        q[("comp_h", k)] = v("T", "S2dyn_c12_m50", k) - v("T", "S2dyn_c12", k)
        q[("struct", k)] = e("S2dyn_s50", k) - e("S2dyn_m50", k)
        q[("dR_step", k)] = R["S2dyn_10r"] - R["S2dyn_s50"]
        q[("d_felu", k)] = e("S2dyn_felu", k) - e("S2dyn_10r", k)
        q[("dzero_felu", k)] = v("Z", "S2dyn_10r", k) - v("Z", "S2dyn_felu", k)
        q[("dn_felu", k)] = n["S2dyn_felu"] - n["S2dyn_10r"]
        q[("dR_growth", k)] = R["S2dyn_10r"] - R["S2dyn_c12"]
        q[("dn_growth", k)] = n["S2dyn_10r"] - n["S2dyn_c12"]
        q[("I_nat", k)] = e("N2r", k) - e("N2r_m50", k)
    # the early window (first continuation task only)
    Fe = v("Ee", "S2u30r")
    Re = {a: v("Ee", a) - Fe for a in DYN}
    ne = {a: v("Ne", a) for a in DYN}
    q[("R_none_e", 1)] = Re["S2dyn_10r"]
    for tag, base, arm, sign in (("drop_e", "S2dyn_10r", "S2dyn_m50", -1), ("drop_h_e", "S2dyn_c12", "S2dyn_c12_m50", -1),
                                 ("add_h_e", "S2dyn_c12", "S2dyn_c12_a02", 1)):
        dR, dn, phi, kap = change(Re, ne, base, arm, sign)
        q[(f"dR_{tag}", 1)], q[(f"dn_{tag}", 1)], q[(f"phi_{tag}", 1)], q[(f"kappa_{tag}", 1)] = dR, dn, phi, kap
    st = {(nm, k, lv): paired(x, lv) for (nm, k), x in q.items() for lv in (MAIN_LEVEL, COMP_LEVEL)}
    rho = [spearman([cols["N"][(s, a, 1)] for a in LADDER], [cols["E"][(s, a, 1)] for a in LADDER]) for s in valid]
    pos = sum(r > 0 for r in rho if math.isfinite(r))
    neg = sum(r < 0 for r in rho if math.isfinite(r))
    p_rho = sign_test(max(pos, neg), pos + neg)

    A_ok = len(valid) >= MIN_SEEDS
    B_ok = A_ok and st[("R_none", 1, COMP_LEVEL)]["lo"] > 0 and st[("R_c12", 1, COMP_LEVEL)]["lo"] > 0
    lo = lambda nm: st[(nm, 1, COMP_LEVEL)]["lo"]
    manip = {tag: A_ok and lo(f"dn_{tag}") > 0 for tag in ("add_h", "add", "drop_h", "drop", "m25", "drop_e", "drop_h_e",
                                                           "add_h_e")}
    manip["felu"] = A_ok and lo("dzero_felu") > 0

    def gate(ok):
        if not A_ok:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        return None if ok else "NOT_MANIPULATED"

    def direction(tag, level, word):
        g = gate(manip[tag])
        if g:
            return g
        return {"+": f"{word}_MOVES_E", "0": f"{word}_NO_EFFECT", "-": f"{word}_REVERSED"}[st[(f"dR_{tag}", 1, level)]["sign"]]

    def prop(tag):
        return gate(manip[tag]) or proportion_label(st[(f"phi_{tag}", 1, COMP_LEVEL)])

    def three(name, plus, zero, minus):
        return gate(True) or {"+": plus, "0": zero, "-": minus}[st[(name, 1, COMP_LEVEL)]["sign"]]

    felu = gate(manip["felu"])
    if felu is None:
        x = st[("d_felu", 1, COMP_LEVEL)]
        felu = ("FELU_SAME" if (-MARGIN < x["lo"] and x["hi"] < MARGIN) else
                "FELU_HIGHER" if x["lo"] > 0 else "FELU_LOWER" if x["hi"] < 0 else "FELU_UNRESOLVED")
    labels = {
        "primary": direction("add_h", MAIN_LEVEL, "ADD_H"),
        "prop_add_h": prop("add_h"),
        "add": direction("add", COMP_LEVEL, "ADD"),
        "prop_add": prop("add"),
        "drop_h": direction("drop_h", COMP_LEVEL, "DROP_H"),
        "prop_drop_h": prop("drop_h"),
        "drop": direction("drop", COMP_LEVEL, "DROP"),
        "prop_drop": prop("drop"),
        "m25": direction("m25", COMP_LEVEL, "M25"),
        "drop_early": direction("drop_e", COMP_LEVEL, "DROP_EARLY"),
        "prop_drop_early": prop("drop_e"),
        "drop_h_early": direction("drop_h_e", COMP_LEVEL, "DROP_H_EARLY"),
        "prop_drop_h_early": prop("drop_h_e"),
        "compensation": three("comp", "COMPENSATES", "NO_COMPENSATION", "RAW_FALLS"),
        "compensation_h": three("comp_h", "COMPENSATES_H", "NO_COMPENSATION_H", "RAW_FALLS_H"),
        "structure": three("struct", "STRUCTURE_MATTERS", "STRUCT_EQUAL", "STEP_WORSE"),
        "felu": felu,
        "ladder": ("INAPPLICABLE" if not A_ok else
                   "NEFF_TRACKS" if (pos > neg and p_rho < SIGN_ALPHA) else
                   "NEFF_OPPOSES" if (neg > pos and p_rho < SIGN_ALPHA) else "NEFF_UNRESOLVED"),
        "growth": three("dR_growth", "GROWTH_REMAINDER", "GROWTH_NO_REMAINDER", "GROWTH_REVERSED"),
    }
    arm_means, report = {}, {}
    for a in ARMS:
        arm_means[a] = {k: float(np.mean(e(a, k))) if valid else float("nan") for k in (1, 2)}
        for c in ("nbar_neffS2", "nbar_neffT2", "probe_zeroS2_mean", "probe_zero2_mean", "climb2", "early_acc",
                  "nbar_neffS2_early", "nbar_neffT2_early", "w1norm_end", "w2norm_end", "b2_mean_end", "keep_share",
                  "lift_share", "lift_units_alive", "neffS2_start", "neffT2_start", "rows_w2", "memo_acc_end"):
            for k in (1, 2):
                vals = [_f(r.get(c)) for s in valid for r in seeds[s]["arms"] if r["arm"] == a and r["k"] == str(k)]
                if vals and any(math.isfinite(x) for x in vals):
                    report[f"{a}|{c}|{k}"] = float(np.nanmean(vals))
    return {"valid_seeds": valid, "invalid_seeds": invalid, "A_ok": A_ok, "B_ok": B_ok, "manip": manip,
            "labels": labels, "stats": st, "per_seed": {f"{nm}|{k}": x.tolist() for (nm, k), x in q.items()},
            "rho": rho, "rho_pos": pos, "rho_neg": neg, "rho_p": p_rho,
            "arm_means": arm_means, "report": report}


def _fmt(x, nd=4):
    return "nan" if x is None or not math.isfinite(x) else f"{x:+.{nd}f}"


def summary_md(res: dict, env: dict) -> str:
    st, n = res["stats"], len(res["valid_seeds"])

    def line(name, k, lv):
        x = st[(name, k, lv)]
        vals = res["per_seed"][f"{name}|{k}"]
        return (f"| {name} | {k} | {lv} | {_fmt(x['mean'])} | {_fmt(x['sd'])} | [{_fmt(x['lo'])}, {_fmt(x['hi'])}] | "
                f"{x['sign']} | {sum(v > 0 for v in vals)}/{n} |")

    L = [f"# {RUN_ID} — 判定（担い手 n̄_eff を直接動かす）", "",
         "> 自動生成: `analysis/neffdir_ee_0918/verdict.py`。spec: `specs/spec_neffdir_ee_0918.md`。", "",
         f"- HEAD `{env['head']}`・run の commit {env['run_commits']}・有効 seed {n}（無効: {res['invalid_seeds'] or 'なし'}）",
         f"- (A) {res['A_ok']}・(B) R_none > 0 かつ R_c12 > 0: {res['B_ok']}",
         "- (C) 操作が n̄_eff を動かしたか（95% 下端 > 0）: " + "・".join(f"{k} {v}" for k, v in res["manip"].items()), "",
         "## 判定", "", "| 問い | ラベル |", "|---|---|"]
    L += [f"| {k} | **{v}** |" for k, v in res["labels"].items()]
    L += ["", f"n̄_eff と E の順位相関（{', '.join(LADDER)}・seed ごと）: 平均 {np.nanmean(res['rho']):+.3f}、"
          f"正 {res['rho_pos']}・負 {res['rho_neg']}、符号検定 p = {res['rho_p']:.3g}。seed 別: "
          + ", ".join(f"{r:+.2f}" for r in res["rho"]), "",
          "## 量（seed 内の差）", "", "| 量 | 継続 | 水準 | 平均 | SD | 区間 | 符号 | 正の seed |",
          "|---|---|---|---|---|---|---|---|"]
    L.append(line("dR_add_h", 1, MAIN_LEVEL))
    k1 = ["R_none", "R_c12"]
    for tag in ("add_h", "add", "drop_h", "drop", "m25"):
        k1 += [f"dR_{tag}", f"dn_{tag}", f"kappa_{tag}", f"phi_{tag}"]
    k1 += ["R_none_e"]
    for tag in ("drop_e", "drop_h_e", "add_h_e"):
        k1 += [f"dR_{tag}", f"dn_{tag}", f"kappa_{tag}", f"phi_{tag}"]
    k1 += ["comp", "comp_h", "struct", "dR_step", "d_felu", "dzero_felu", "dn_felu", "dR_growth", "dn_growth", "I_nat"]
    for name in k1:
        L.append(line(name, 1, COMP_LEVEL))
    for name in ("R_none", "R_c12", "dR_add_h", "dn_add_h", "phi_add_h", "dR_add", "dR_drop_h", "dn_drop_h",
                 "dR_drop", "dn_drop", "comp", "comp_h", "struct", "d_felu", "dR_growth"):
        L.append(line(name, 2, COMP_LEVEL))
    L += ["", "## 腕ごとの平均（有効 seed）", "",
          "| 腕 | E 継続 1 | E 継続 2 | E 前半 | n̄_eff(S) 1 | n̄_eff(T) 1 | n̄_eff(S) 前半 | 0 の割合(S) 1 | 登り 1 | ‖w1‖ 1 | ‖w2‖ 1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rp = res["report"]
    g = lambda a, c, k=1: rp.get(f"{a}|{c}|{k}", float("nan"))
    for a in ARMS:
        L.append(f"| {a} | {res['arm_means'][a][1]:.4f} | {res['arm_means'][a][2]:.4f} | {g(a, 'early_acc'):.4f} | "
                 f"{g(a, 'nbar_neffS2'):.5f} | {g(a, 'nbar_neffT2'):.5f} | {g(a, 'nbar_neffS2_early'):.5f} | "
                 f"{g(a, 'probe_zeroS2_mean'):.3f} | {g(a, 'climb2'):+.2f} | {g(a, 'w1norm_end'):.3f} | {g(a, 'w2norm_end'):.3f} |")
    L += ["", "## 報告（seed 平均）", "", "| 腕 | 量 | 継続 | 値 |", "|---|---|---|---|"]
    for key, v in rp.items():
        a, c, k = key.split("|")
        if c in ("keep_share", "lift_share", "lift_units_alive", "neffS2_start", "neffT2_start", "rows_w2",
                 "b2_mean_end", "memo_acc_end"):
            L.append(f"| {a} | {c} | {k} | {v:.5g} |")
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
            w.writerow(["rho_neff_E", 1, s, r])
    (out / "summary.md").write_text(summary_md(res, env))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
        "run_commits": env["run_commits"], "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"],
        "A_ok": res["A_ok"], "B_ok": res["B_ok"], "manip": res["manip"], "labels": res["labels"],
        "rho": res["rho"], "rho_p": res["rho_p"], "arm_means": res["arm_means"], "report": res["report"],
        "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "judgement": {"main_level": MAIN_LEVEL, "comp_level": COMP_LEVEL, "min_seeds": MIN_SEEDS,
                      "tol_cap": TOL_CAP, "sign_alpha": SIGN_ALPHA, "margin": MARGIN, "keep_tol": KEEP_TOL,
                      "lift_tol": LIFT_TOL, "ladder": LADDER, "early_updates": 1500},
    }, indent=2, default=str))
    print(f"labels: {res['labels']}  valid {len(res['valid_seeds'])}  invalid {res['invalid_seeds']}")


if __name__ == "__main__":
    main()
