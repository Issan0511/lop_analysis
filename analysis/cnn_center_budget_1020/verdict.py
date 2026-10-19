#!/usr/bin/env python3
"""Registered comparisons of cnn_center_budget_1020 (spec section 7, predictions section 9).  Fixed before
the production run.

    python3 analysis/cnn_center_budget_1020/verdict.py --root results/cnn_center_budget_1020

Main: D = centred arm - its baseline (RCB - R, ECB - ELU), per seed, of the first-780-update online
accuracy averaged over the late hard tasks 21, 23, 25, 27, 29, in pt, in four cells (ReLU / ELU x 780 /
7800).  Exact two-sided sign test (seeds with D = 0 dropped), Holm over the four cells; p_holm < .05 ->
CENTER_HIGHER (more positive seeds) / CENTER_LOWER (more negative), else TIE.  A diverged run removes its
seed from the cells it enters.  Secondary (sign tests, no correction): the interaction
(D_ELU,7800 - D_ELU,780) - (D_ReLU,7800 - D_ReLU,780) and the two budget differences; FC2 log width and
the FC2 sparse-detector fraction (open fraction in [.05, .3)) averaged over boundaries 21..29.
S-reuse: the R / ELU rows against cifar5p1_cnn_budget_1019 (per_task, probe_stats, fresh_control, the
previous run's columns except the bundle slot), bit for bit as written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("RCB", "ECB", "R", "ELU")
PAIRS = (("ReLU", "RCB", "R"), ("ELU", "ECB", "ELU"))
BUDGETS = (780, 7800)
SEEDS = tuple(range(2000, 2010))
LATE = (21, 23, 25, 27, 29)
WINDOW = tuple(range(21, 30))                      # boundaries 21..29 (ends of tasks 21..29)
N_TASKS = 30
FRESH = (9, 19, 29)
SITES = ("c1", "c2", "f1", "f2")
ALPHA = 0.05
N_BOOT = 20000
PREV = ("results", "cifar5p1_cnn_budget_1019")
REUSE_EXCLUDED = ("slot",)                         # bundle slot positions differ by construction
PPOS_BINS = ("p0", "lt05", "sparse", "mid", "hi", "ge95")
TRAJ = ("s_med", "m_med", "r_med", "dmean_med", "ppos_med", "dead_frac", "lowresp_frac", "eff_rank",
        "ppos_sparse_frac", "ppos_p0_frac", "abs_b_over_s_med", "a_over_s_med", "m_over_s_med")
FIRE = tuple(f"fire_rate_{t}" for t in SITES)
LAG_POINTS = (0, 1, 3, 10, 30, 100, 300, 780, 2340, 7800)


def f(x):
    return float(x) if x not in ("", None, "None") else float("nan")


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def read(root: Path) -> dict:
    data = {}
    for b in BUDGETS:
        d = root / f"b{b}"
        p = d / "provenance.json"
        status = json.loads(p.read_text()).get("status") if p.exists() else "MISSING"
        rows, probe, fresh, lag = (read_csv(d / n) for n in ("per_task.csv", "probe_stats.csv",
                                                              "fresh_control.csv", "lag.csv"))
        data[b] = {"status": status, "rows": rows, "probe": probe, "fresh": fresh, "lag": lag,
                   "I_rows": {(r["arm"], int(r["seed"]), int(r["task"])): r for r in rows},
                   "I_probe": {(r["arm"], int(r["seed"]), r["site"], int(r["boundary"])): r for r in probe},
                   "I_fresh": {(r["arm"], int(r["seed"]), int(r["task"])): r for r in fresh},
                   "dups": (len(rows) - len({(r["arm"], r["seed"], r["task"]) for r in rows}))
                           + (len(probe) - len({(r["arm"], r["seed"], r["site"], r["boundary"]) for r in probe}))
                           + (len(fresh) - len({(r["arm"], r["seed"], r["task"]) for r in fresh}))}
    return data


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def sign_test(d) -> dict:
    """Exact two-sided sign test; zeros (and NaN) dropped."""
    d = np.asarray([x for x in d if np.isfinite(x)], dtype=np.float64)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        p = 1.0
    else:
        k = min(pos, neg)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return {"n_nonzero": n, "positive": pos, "negative": neg, "zeros": int((d == 0).sum()), "p": p}


def holm(ps: dict) -> dict:
    """Holm-adjusted p-values (step-down, monotone)."""
    keys = sorted(ps, key=lambda k: (ps[k], k))
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def boot_ci(x, seed=1020, q=(0.025, 0.975)):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(x), size=(N_BOOT, len(x)))
    lo, hi = np.quantile(x[idx].mean(axis=1), q)
    return [float(lo), float(hi)]


def describe(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    fin = x[np.isfinite(x)]
    return {"n": int(len(fin)), "mean": float(fin.mean()) if len(fin) else None,
            "median": float(np.median(fin)) if len(fin) else None, "ci95_boot": boot_ci(fin),
            "per_seed": [None if not np.isfinite(v) else float(v) for v in x], **sign_test(fin)}


# --------------------------------------------------------------------------
# per-seed quantities
# --------------------------------------------------------------------------

def run_diverged(d, arm, seed) -> bool:
    rs = [d["I_rows"].get((arm, seed, t)) for t in range(1, N_TASKS + 1)]
    if any(r is None for r in rs):
        return True
    return any(r["diverged"] == "True" or int(r["bad_step"]) >= 0 for r in rs)


def late_mean(d, arm, field, seeds=SEEDS, tasks=LATE):
    v = []
    for s in seeds:
        xs = [f(d["I_rows"][(arm, s, t)][field]) for t in tasks if (arm, s, t) in d["I_rows"]]
        v.append(float(np.mean(xs)) if len(xs) == len(tasks) else float("nan"))
    return np.asarray(v)


def window_probe(d, arm, site, field, seeds=SEEDS, bounds=WINDOW, log=False):
    v = []
    for s in seeds:
        xs = [f(d["I_probe"][(arm, s, site, b)].get(field)) for b in bounds if (arm, s, site, b) in d["I_probe"]]
        xs = np.asarray(xs, dtype=np.float64)
        if log:
            xs = np.log(xs)
        v.append(float(xs.mean()) if len(xs) == len(bounds) else float("nan"))
    return np.asarray(v)


def fresh_gap(d, arm, task, field="gap_online_acc", seeds=SEEDS):
    return np.asarray([f(d["I_fresh"][(arm, s, task)][field]) if (arm, s, task) in d["I_fresh"] else float("nan")
                       for s in seeds])


def complete(data) -> dict:
    out = {}
    for b, d in data.items():
        for a in ARMS:
            want = {(a, s, t) for s in SEEDS for t in range(1, N_TASKS + 1)}
            ok = d["status"] == "COMPLETED" and d["dups"] == 0 and want <= set(d["I_rows"])
            ok = ok and all((a, s, site, bd) in d["I_probe"] for s in SEEDS for site in SITES
                            for bd in range(0, N_TASKS + 1))
            ok = ok and all((a, s, t) in d["I_fresh"] for s in SEEDS for t in FRESH)
            out[f"{a}|{b}"] = {"complete": bool(ok),
                               "diverged_seeds": [s for s in SEEDS if run_diverged(d, a, s)] if want <= set(d["I_rows"]) else None}
    return out


def paired(d, act, arm_c, arm_b, x_c, x_b):
    """Centred - baseline per seed, NaN where either run diverged."""
    out = np.asarray(x_c, dtype=np.float64) - np.asarray(x_b, dtype=np.float64)
    for i, s in enumerate(SEEDS):
        if run_diverged(d, arm_c, s) or run_diverged(d, arm_b, s):
            out[i] = float("nan")
    return out


# --------------------------------------------------------------------------
# S-reuse
# --------------------------------------------------------------------------

def reuse(root: Path, data, prev_root: Path | None = None) -> dict:
    prev_root = prev_root or (root.parent / PREV[1])
    res = {"previous_root": str(prev_root)}
    for b in BUDGETS:
        d = data[b]
        rb = {}
        for name, key_of, ours in (
                ("per_task", lambda r: (r["arm"], int(r["seed"]), int(r["task"])), d["I_rows"]),
                ("probe_stats", lambda r: (r["arm"], int(r["seed"]), r["site"], int(r["boundary"])), d["I_probe"]),
                ("fresh_control", lambda r: (r["arm"], int(r["seed"]), int(r["task"])), d["I_fresh"])):
            prev = [r for r in read_csv(prev_root / f"b{b}" / f"{name}.csv") if r["arm"] in ("R", "ELU")]
            if not prev:
                rb[name] = {"label": "NOT_AVAILABLE"}
                continue
            cols = [c for c in prev[0] if c not in REUSE_EXCLUDED]
            missing, mism, worst = 0, 0, {}
            for r in prev:
                q = ours.get(key_of(r))
                if q is None:
                    missing += 1
                    continue
                for c in cols:
                    if q.get(c) != r[c]:
                        mism += 1
                        try:
                            dv = abs(float(q.get(c)) - float(r[c]))
                        except (TypeError, ValueError):
                            dv = float("inf")
                        worst[c] = max(worst.get(c, 0.0), dv)
            rb[name] = {"label": "PASS" if (missing == 0 and mism == 0) else "FAIL", "rows": len(prev),
                        "columns": len(cols), "missing": missing, "mismatches": mism,
                        "max_abs_diff_by_column": worst}
        rb["label"] = "PASS" if all(v.get("label") == "PASS" for v in rb.values() if isinstance(v, dict)) else "FAIL"
        res[str(b)] = rb
    res["label"] = "PASS" if all(res[str(b)]["label"] == "PASS" for b in BUDGETS) else "FAIL"
    return res


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def verdict(root: Path, prev_root: Path | None = None) -> dict:
    data = read(root)
    comp = complete(data)
    res = {"spec": "specs/spec_cnn_center_budget_1020.md", "status": {str(b): data[b]["status"] for b in BUDGETS},
           "completeness": comp}
    # ---- main
    D, cells = {}, {}
    for act, ac, ab in PAIRS:
        for b in BUDGETS:
            d = data[b]
            key = f"{act}|{b}"
            if not (comp[f"{ac}|{b}"]["complete"] and comp[f"{ab}|{b}"]["complete"]):
                cells[key] = {"label": "INCOMPLETE"}
                D[key] = np.full(len(SEEDS), np.nan)
                continue
            x = 100.0 * paired(d, act, ac, ab, late_mean(d, ac, "first780_online_acc"),
                               late_mean(d, ab, "first780_online_acc"))
            D[key] = x
            cells[key] = describe(x)
    ps = {k: v["p"] for k, v in cells.items() if "p" in v}
    ph = holm(ps)
    for k, v in cells.items():
        if "p" not in v:
            continue
        v["p_holm"] = ph[k]
        v["label"] = ("CENTER_HIGHER" if v["positive"] > v["negative"] else "CENTER_LOWER") \
            if ph[k] < ALPHA and v["positive"] != v["negative"] else "TIE"
    res["main"] = {"quantity": "D = centred - baseline, first-780-update online, late hard tasks, pt",
                   "test": "exact two-sided sign test, Holm over 4 cells", "cells": cells}
    # ---- secondary
    sec = {}
    for act, ac, ab in PAIRS:
        sec[f"budget_diff_{act}"] = describe(D[f"{act}|7800"] - D[f"{act}|780"])
    sec["interaction"] = describe((D["ELU|7800"] - D["ELU|780"]) - (D["ReLU|7800"] - D["ReLU|780"]))
    W, SP = {}, {}
    for act, ac, ab in PAIRS:
        for b in BUDGETS:
            d = data[b]
            key = f"{act}|{b}"
            W[key] = paired(d, act, ac, ab, window_probe(d, ac, "f2", "s_med", log=True),
                            window_probe(d, ab, "f2", "s_med", log=True))
            SP[key] = paired(d, act, ac, ab, window_probe(d, ac, "f2", "ppos_sparse_frac"),
                             window_probe(d, ab, "f2", "ppos_sparse_frac"))
            sec[f"fc2_log_width_{key}"] = describe(W[key])
            sec[f"fc2_sparse_frac_{key}"] = describe(SP[key])
    res["secondary"] = sec
    # ---- predictions (spec section 9; Claude (Fable))
    def cnt(x, sign):
        x = np.asarray(x)
        return int((x > 0).sum()) if sign > 0 else int((x < 0).sum())
    pr = {}
    pr["Q1"] = {"rule": "ReLU 780: D > 0 in >= 9/10 seeds", "positive": cnt(D["ReLU|780"], 1),
                "holds": cnt(D["ReLU|780"], 1) >= 9}
    m780, m7800 = np.nanmean(D["ReLU|780"]), np.nanmean(D["ReLU|7800"])
    pr["Q2"] = {"rule": "ReLU 7800: D > 0 in >= 8/10 seeds and mean D_7800 < mean D_780",
                "positive": cnt(D["ReLU|7800"], 1), "mean_D780": float(m780), "mean_D7800": float(m7800),
                "holds": cnt(D["ReLU|7800"], 1) >= 8 and bool(m7800 < m780)}
    med = float(np.nanmedian(D["ELU|780"])) if np.isfinite(D["ELU|780"]).any() else float("nan")
    pr["Q3"] = {"rule": "ELU 780: |median D| < 1 pt", "median": med, "holds": bool(abs(med) < 1.0)}
    pr["Q4"] = {"rule": "ELU 7800: D < 0 in >= 8/10 seeds", "negative": cnt(D["ELU|7800"], -1),
                "holds": cnt(D["ELU|7800"], -1) >= 8}
    q5 = {}
    for act, ac, ab in PAIRS:
        wm = float(np.nanmedian(W[f"{act}|7800"])) if np.isfinite(W[f"{act}|7800"]).any() else float("nan")
        sm = float(np.nanmedian(SP[f"{act}|7800"])) if np.isfinite(SP[f"{act}|7800"]).any() else float("nan")
        q5[act] = {"median_dlog_width": wm, "median_dsparse": sm, "holds": bool(wm >= 0 and sm < 0)}
    n5 = sum(v["holds"] for v in q5.values())
    pr["Q5"] = {"rule": "7800, per activation: median dlog FC2 width >= 0 and median dsparse < 0 (boundaries 21..29)",
                **q5, "label": "TRUE" if n5 == 2 else "PARTIAL" if n5 == 1 else "FALSE"}
    res["predictions_claude_fable"] = pr
    # ---- descriptive
    desc = {}
    for b in BUDGETS:
        d = data[b]
        for a in ARMS:
            k = f"{a}|{b}"
            e = {fld: describe(100 * late_mean(d, a, fld)) for fld in
                 ("first100_online_acc", "first780_online_acc", "online_acc", "test_acc")}
            for t in FRESH:
                e[f"fresh_gap_t{t}"] = describe(100 * fresh_gap(d, a, t))
            for site in SITES:
                for fld in TRAJ:
                    e[f"{site}_{fld}_b30"] = describe(window_probe(d, a, site, fld, bounds=(30,)))
                    e[f"{site}_{fld}_win"] = describe(window_probe(d, a, site, fld))
                for k_ in PPOS_BINS:
                    e[f"{site}_ppos_{k_}_win"] = describe(window_probe(d, a, site, f"ppos_{k_}_frac"))
            if a in ("RCB", "ECB"):
                for fld in FIRE:
                    e[fld] = describe(late_mean(d, a, fld))
                for site in SITES:
                    for fld in ("lag_u0", "lag_end", "blag_first10", "blag_last100"):
                        e[f"{fld}_{site}"] = describe(late_mean(d, a, f"{fld}_{site}"))
                lagt = {}
                for u in LAG_POINTS:
                    if u > b:
                        continue
                    for site in SITES:
                        xs = [f(r[f"lag_{site}"]) for r in d["lag"] if r["arm"] == a and int(r["update"]) == u
                              and int(r["task"]) in LATE]
                        lagt[f"{site}|{u}"] = float(np.mean(xs)) if xs else None
                e["lag_late_hard_mean_by_update"] = lagt
            desc[k] = e
    res["descriptive"] = desc
    res["S_reuse"] = reuse(root, data, prev_root)
    return res


# --------------------------------------------------------------------------
# summary (Japanese)
# --------------------------------------------------------------------------

def fmt(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def summary(res: dict) -> str:
    L = ["# cnn_center_budget_1020 — CH′＋B の損益（活性化 × 予算）", ""]
    L.append("状態: " + "、".join(f"{b} 更新 `{s}`" for b, s in res["status"].items()) + "。")
    inc = [k for k, v in res["completeness"].items() if not v["complete"]]
    div = {k: v["diverged_seeds"] for k, v in res["completeness"].items() if v.get("diverged_seeds")}
    L.append(f"完走していない (腕, 予算): {', '.join(inc) if inc else 'なし'}。発散した run の seed: "
             f"{json.dumps(div, ensure_ascii=False) if div else 'なし'}。")
    r = res["S_reuse"]
    L.append(f"S-reuse（土台の R・ELU が前段とビット一致か）: `{r['label']}`"
             + "（" + "、".join(f"{b}: " + "・".join(f"{n} {r[str(b)][n]['label']}"
                                                   for n in ("per_task", "probe_stats", "fresh_control"))
                               for b in (780, 7800)) + "）。")
    L += ["", "## 主の判定", "",
          "D = 中心化の腕 − 土台の腕。後期 hard 課題（21, 23, 25, 27, 29）の最初の 780 更新の online の seed 平均（pt）。"
          "厳密な符号検定、Holm 4 比較。", "",
          "| セル | 平均 | 中央値 | 95% 区間 | 正 / 負 | p | p_holm | 判定 |", "|---|---:|---:|---|---:|---:|---:|---|"]
    for k, v in res["main"]["cells"].items():
        if v.get("label") == "INCOMPLETE":
            L.append(f"| {k} | — | — | — | — | — | — | `INCOMPLETE` |")
            continue
        ci = v["ci95_boot"]
        L.append(f"| {k} | {fmt(v['mean'])} | {fmt(v['median'])} | "
                 f"{'[' + fmt(ci[0]) + ', ' + fmt(ci[1]) + ']' if ci else '—'} | {v['positive']} / {v['negative']} | "
                 f"{v['p']:.4f} | {v['p_holm']:.4f} | `{v['label']}` |")
    L += ["", "## 副の比較（符号検定、補正なし）", "", "| 量 | 平均 | 中央値 | 正 / 負 | p |", "|---|---:|---:|---:|---:|"]
    for k, v in res["secondary"].items():
        nd = 3 if ("width" in k or "sparse" in k) else 2
        L.append(f"| {k} | {fmt(v['mean'], nd)} | {fmt(v['median'], nd)} | {v['positive']} / {v['negative']} | {v['p']:.4f} |")
    L += ["", "## 予測（Claude（Fable）、本走の前に固定）", ""]
    pr = res["predictions_claude_fable"]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        x = pr[q]
        extra = {k: v for k, v in x.items() if k not in ("rule", "holds")}
        L.append(f"- {q}: {'成立' if x['holds'] else '不成立'}。{x['rule']}。{json.dumps(extra, ensure_ascii=False)}")
    q5 = pr["Q5"]
    L.append(f"- Q5: `{q5['label']}`。ReLU {json.dumps(q5['ReLU'], ensure_ascii=False)}、ELU {json.dumps(q5['ELU'], ensure_ascii=False)}")
    L += ["", "## 記述（後期 hard 課題の seed 平均）", "",
          "| 腕・予算 | 最初 100 (%) | 最初 780 (%) | 課題全体 (%) | test (%) | fresh − 継続 t29 (pt) |",
          "|---|---:|---:|---:|---:|---:|"]
    for k, e in res["descriptive"].items():
        L.append(f"| {k} | {fmt(e['first100_online_acc']['mean'], sign=False)} | {fmt(e['first780_online_acc']['mean'], sign=False)} | "
                 f"{fmt(e['online_acc']['mean'], sign=False)} | {fmt(e['test_acc']['mean'], sign=False)} | {fmt(e['fresh_gap_t29']['mean'])} |")
    L += ["", "FC2（固定 probe、unit の中央値の seed 平均、境界 30）", "",
          "| 腕・予算 | 幅 s | r | φ′ | p⁺ | 死 | 低応答 | まばら [.05,.3) | p⁺ = 0 | \\|b\\|/s_q |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, e in res["descriptive"].items():
        g = lambda n: e[f"f2_{n}_b30"]["mean"]
        L.append(f"| {k} | {fmt(g('s_med'), sign=False)} | {fmt(g('r_med'))} | {fmt(g('dmean_med'), sign=False)} | "
                 f"{fmt(g('ppos_med'), sign=False)} | {fmt(g('dead_frac'), sign=False)} | {fmt(g('lowresp_frac'), sign=False)} | "
                 f"{fmt(g('ppos_sparse_frac'), sign=False)} | {fmt(g('ppos_p0_frac'), sign=False)} | {fmt(g('abs_b_over_s_med'), 3, sign=False)} |")
    L += ["", "射影の発火率（後期 hard 課題、unit × 更新あたり）と EMA の遅れ（‖x̄ − m‖/‖x̄‖、切替直後 u = 0 と課題末）", "",
          "| 腕・予算 | 発火 c1 | c2 | f1 | f2 | 遅れ u0 c1 | c2 | f1 | f2 | 遅れ 末 f2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, e in res["descriptive"].items():
        if not (k.startswith("RCB") or k.startswith("ECB")):
            continue
        L.append(f"| {k} | " + " | ".join(fmt(e[f'fire_rate_{t}']['mean'], 4, sign=False) for t in SITES) + " | "
                 + " | ".join(fmt(e[f'lag_u0_{t}']['mean'], 3, sign=False) for t in SITES)
                 + f" | {fmt(e['lag_end_f2']['mean'], 3, sign=False)} |")
    L += ["", "数値の全量は `verdict.json` にある。"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prev", default=None, help="cifar5p1_cnn_budget_1019 results (default: next to --root)")
    a = ap.parse_args()
    root = Path(a.root)
    res = verdict(root, Path(a.prev) if a.prev else None)
    (root / "verdict.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    (root / "summary.md").write_text(summary(res))
    print("main:", {k: v.get("label") for k, v in res["main"]["cells"].items()},
          "reuse:", res["S_reuse"]["label"], "Q5:", res["predictions_claude_fable"]["Q5"]["label"])


if __name__ == "__main__":
    main()
