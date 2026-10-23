#!/usr/bin/env python3
"""Registered verdict of neokunekune_methods_cifar5p1_cnn_1018 (spec §5-§6).  Fixed before calibration.

    python3 analysis/neokunekune_methods_cifar5p1_cnn_1018/verdict.py \
        --main results/neokunekune_methods_cifar5p1_cnn_1018/main \
        --selected results/neokunekune_methods_cifar5p1_cnn_1018/calib/selected.json

A(arm, seed) = mean pre-update online accuracy of hard tasks 21, 23, 25, 27, 29 (each arm at its
calibrated configuration).  Main family: D_X = A(NEO) - A(X); secondary family: A(SNAfrz1) - A(X);
X in {SN06, SN3, KKT1, ELU, LK, RSL, ReDo, L2I}.  Percentile bootstrap over seeds: one index matrix
default_rng(1018).integers(0, 10, size=(20000, 10)) for every comparison; Bonferroni over 8 per
family, two-sided quantiles 0.003125 / 0.996875.  Lower end > 0 -> BETTER_THAN_<X>; upper end < 0 ->
WORSE_THAN_<X>; else INCONCLUSIVE; an end equal to the threshold does not meet it.  Secondary label:
non-inferiority at 0.5 points (lower end > -0.005).  A comparison with an arm that is not complete
(10 seeds x 30 tasks, finite, no divergence) is INCOMPLETE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ARMS12 = ("R", "SNA", "SNAfrz1", "NEO", "SN06", "SN3", "KKT1", "ELU", "LK", "RSL", "ReDo", "L2I")
TARGETS = ("SN06", "SN3", "KKT1", "ELU", "LK", "RSL", "ReDo", "L2I")
FOCAL = {"main": "NEO", "secondary": "SNAfrz1"}
SEEDS = tuple(range(2000, 2010))
LATE = (21, 23, 25, 27, 29)
EARLY = (1, 3, 5, 7, 9)
HARD = tuple(range(1, 31, 2))
N_TASKS = 30
MARGIN = 0.005
N_BOOT = 20000
Q = (0.05 / 8 / 2, 1 - 0.05 / 8 / 2)                       # 0.003125, 0.996875
BUNDLES = ("A", "B", "C")
NOCHANGE_COLS_SKIP = ("arm", "slot", "classes")


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def read_rows(main: Path) -> tuple[list[dict], dict]:
    rows, status = [], {}
    for b in BUNDLES:
        d = main / b
        p = d / "provenance.json"
        status[b] = json.loads(p.read_text()).get("status") if p.exists() else "MISSING"
        if (d / "per_task.csv").exists():
            rows += list(csv.DictReader(open(d / "per_task.csv")))
    return rows, status


def complete_arms(rows, seeds=SEEDS) -> dict:
    out = {}
    for a in ARMS12:
        q = [r for r in rows if r["arm"] == a]
        keys = {(int(r["seed"]), int(r["task"])) for r in q}
        good = (len(q) == len(seeds) * N_TASKS and keys == {(s, t) for s in seeds for t in range(1, N_TASKS + 1)}
                and all(0 <= f(r["online_acc"]) <= 1 for r in q)
                and all(r["diverged"] == "False" and int(r["bad_step"]) < 0 for r in q))
        out[a] = good
    return out


def A_of(rows, arm, tasks=LATE, field="online_acc", seeds=SEEDS) -> np.ndarray:
    v = []
    for s in seeds:
        xs = [f(r[field]) for r in rows if r["arm"] == arm and int(r["seed"]) == s and int(r["task"]) in tasks]
        v.append(float(np.mean(xs)) if len(xs) == len(tasks) else float("nan"))
    return np.asarray(v)


def interval(d, idx, q=Q) -> dict:
    d = np.asarray(d, dtype=np.float64)
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, q)
    return {"mean": float(d.mean()), "interval": [float(lo), float(hi)], "quantiles": list(q),
            "median": float(np.median(d)), "positive": int((d > 0).sum()), "negative": int((d < 0).sum()),
            "per_seed": [float(x) for x in d]}


def label(iv, x) -> str:
    lo, hi = iv["interval"]
    return f"BETTER_THAN_{x}" if lo > 0 else f"WORSE_THAN_{x}" if hi < 0 else "INCONCLUSIVE"


def nochange_main(main: Path, prev_per_task: Path) -> dict:
    a = main / "A" / "per_task.csv"
    if not a.exists() or not prev_per_task.exists():
        return {"applicable": False, "reason": "missing files"}
    new = list(csv.DictReader(open(a)))
    if not all(r["cfg"].endswith("_lr0.0001") for r in new):
        return {"applicable": False, "reason": "bundle A did not select lr 1e-4 for all four arms"}
    old = list(csv.DictReader(open(prev_per_task)))
    cols = [c for c in old[0] if c in new[0] and c not in NOCHANGE_COLS_SKIP]
    idx = {(r["cfg"].split("_lr")[0], r["seed"], r["task"]): r for r in new}
    bad, missing = 0, 0
    for o in old:
        n = idx.get((o["arm"], o["seed"], o["task"]))
        if n is None:
            missing += 1
            continue
        bad += sum(o[c] != n[c] for c in cols)
    return {"applicable": True, "rows": len(old), "columns": len(cols), "mismatches": bad, "missing": missing,
            "label": "PASS" if bad == 0 and missing == 0 else "FAIL"}


def compute(main: Path, selected: dict | None, prev_per_task: Path | None, seeds=SEEDS) -> dict:
    rows, status = read_rows(main)
    if any(v == "RUNNING" for v in status.values()):
        raise SystemExit("a main bundle is still RUNNING; the verdict is computed once, after all end")
    comp = complete_arms(rows, seeds)
    idx = np.random.default_rng(1018).integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    A = {a: A_of(rows, a, seeds=seeds) for a in ARMS12}
    res = {"status": status, "complete": comp, "seeds": list(seeds), "late_window": list(LATE),
           "bootstrap": {"n": N_BOOT, "rng": "numpy default_rng(1018)", "index": "integers(0, n, size=(20000, n))",
                         "quantiles": list(Q), "family_size": len(TARGETS)},
           "families": {}}
    for fam, focal in FOCAL.items():
        out = {}
        for x in TARGETS:
            key = f"{focal}_minus_{x}"
            ok = comp.get(focal) and comp.get(x) and all(v == "COMPLETED" for v in status.values())
            if not ok or not (np.isfinite(A[focal]).all() and np.isfinite(A[x]).all()):
                out[key] = {"label": "INCOMPLETE"}
                continue
            iv = interval(A[focal] - A[x], idx)
            iv["label"] = label(iv, x)
            iv["noninferior_0.5pt"] = iv["interval"][0] > -MARGIN
            out[key] = iv
        res["families"][fam] = out
    aux = {"A_late": {a: list(A[a]) for a in ARMS12},
           "A_all_hard": {a: list(A_of(rows, a, HARD, seeds=seeds)) for a in ARMS12},
           "A_early": {a: list(A_of(rows, a, EARLY, seeds=seeds)) for a in ARMS12},
           "first100_late": {a: list(A_of(rows, a, LATE, "first100_online_acc", seeds)) for a in ARMS12},
           "train_late": {a: list(A_of(rows, a, LATE, "train_acc", seeds)) for a in ARMS12},
           "test_late": {a: list(A_of(rows, a, LATE, "test_acc", seeds)) for a in ARMS12}}
    aux["rank_by_mean_A"] = sorted(((a, float(np.mean(A[a]))) for a in ARMS12 if np.isfinite(A[a]).all()),
                                   key=lambda z: -z[1])
    aux["cfg"] = {a: sorted({r["cfg"] for r in rows if r["arm"] == a}) for a in ARMS12}
    aux["redo_resets_total"] = {s: int(sum(f(r.get(f"n_redo_{s}", 0)) + f(r.get(f"n_redo_boundary_{s}", "") or 0)
                                           for r in rows if r["arm"] == "ReDo")) for s in ("c1", "c2", "f1", "f2")}
    fresh = []
    for b in BUNDLES:
        p = main / b / "fresh_control.csv"
        if p.exists():
            fresh += list(csv.DictReader(open(p)))
    aux["fresh_gap_online"] = {a: [f(r["fresh_gap_online"]) for r in fresh if r["arm"] == a] for a in ARMS12}
    res["auxiliary"] = aux
    if selected is not None:
        res["selected"] = {a: {"cfg": v.get("cfg"), "EDGE": v.get("EDGE"), "A_mean_calib": v.get("A_mean")}
                           for a, v in selected["selected"].items()}
    if prev_per_task is not None:
        res["S_nochange_main"] = nochange_main(main, prev_per_task)
    return res


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.2f}"


def pp(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}"


def summary_md(res) -> str:
    L = ["# neokunekune_methods_cifar5p1_cnn_1018 — 判定", ""]
    L.append("束の状態: " + "、".join(f"{k} `{v}`" for k, v in res["status"].items()) + "。")
    inc = [a for a, v in res["complete"].items() if not v]
    L.append(f"完走していない腕: {'、'.join(inc) if inc else 'なし'}。")
    L += ["", "## 主量（後期 hard t=21,23,25,27,29 の online、seed 平均）と選ばれた設定", "",
          "| 順位 | 腕 | 主量 (%) | 全 hard (%) | 早期 − 後期 (pt) | 最初 100 更新 (%) | test 後期 (%) | 設定 |",
          "|---:|---|---:|---:|---:|---:|---:|---|"]
    aux = res["auxiliary"]
    for i, (a, m) in enumerate(aux["rank_by_mean_A"], 1):
        el = np.mean(np.subtract(aux["A_early"][a], aux["A_late"][a]))
        L.append(f"| {i} | {a} | {pct(m)} | {pct(np.mean(aux['A_all_hard'][a]))} | {pp(el)} | "
                 f"{pct(np.mean(aux['first100_late'][a]))} | {pct(np.mean(aux['test_late'][a]))} | "
                 f"`{', '.join(aux['cfg'][a])}` |")
    for fam, title in (("main", "主の族: NEO − X"), ("secondary", "副の族: SNAfrz1 − X")):
        L += ["", f"## {title}（Bonferroni 8、99.375% 区間）", "",
              "| 比較 | 平均 (pt) | 区間 (pt) | 正の seed | 判定 | 0.5 pt の非劣性 |", "|---|---:|---|---:|---|---|"]
        for k, iv in res["families"][fam].items():
            if iv["label"] == "INCOMPLETE":
                L.append(f"| {k} | — | — | — | `INCOMPLETE` | — |")
            else:
                L.append(f"| {k} | {pp(iv['mean'])} | [{pp(iv['interval'][0])}, {pp(iv['interval'][1])}] | "
                         f"{iv['positive']}/{len(iv['per_seed'])} | `{iv['label']}` | {'満たす' if iv['noninferior_0.5pt'] else '満たさない'} |")
    if "selected" in res:
        edges = {a: v["EDGE"] for a, v in res["selected"].items() if v.get("EDGE")}
        L += ["", "選んだ値が格子の端だった腕（`EDGE`）: " + ("、".join(f"{a}（{'・'.join(e)}）" for a, e in edges.items())
                                                       if edges else "なし") + "。格子は広げていない。"]
    if "S_nochange_main" in res:
        nc = res["S_nochange_main"]
        L.append("")
        L.append("S-nochange-main: " + (f"`{nc['label']}`（{nc['rows']} 行 × {nc['columns']} 列、不一致 {nc['mismatches']}）"
                                        if nc.get("applicable") else f"適用外（{nc['reason']}）") + "。")
    L += ["", "主量は online（更新前）精度。test 精度・fresh・幅は別の読みで、確認的判定ではない。数値の全量は `verdict.json`。"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, type=Path)
    ap.add_argument("--selected", type=Path, default=None)
    ap.add_argument("--prev", type=Path, default=Path(__file__).resolve().parents[2]
                    / "results" / "neokunekune_cifar5p1_cnn_1018" / "main" / "per_task.csv")
    a = ap.parse_args()
    sel = json.loads(a.selected.read_text()) if a.selected else None
    res = compute(a.main, sel, a.prev)
    (a.main / "verdict.json").write_text(json.dumps(res, indent=2, default=float))
    (a.main / "summary.md").write_text(summary_md(res))
    print(json.dumps({fam: {k: v["label"] for k, v in d.items()} for fam, d in res["families"].items()}, indent=2))


if __name__ == "__main__":
    main()
