#!/usr/bin/env python3
"""Registered verdict of neokunekune_cifar5p1_cnn_1018 (spec §5), plus the auxiliary and
descriptive readouts of §5.2 and §6.  Written before the production run.

    python3 analysis/neokunekune_cifar5p1_cnn_1018/verdict.py --src results/neokunekune_cifar5p1_cnn_1018/main

Main quantity A(arm, seed) = mean pre-update online accuracy of hard tasks 21, 23, 25, 27, 29.
D1 = A(NEO) - A(SNA): BETTER_THAN_SNA if the lower end of the two-sided 97.5% percentile
bootstrap interval > 0.  D2 = A(NEO) - A(SNAfrz1): NONINFERIOR_TO_FREEZE if the lower end
> -0.005.  20,000 seed resamples from numpy default_rng(1018), one index matrix
rng.integers(0, n, (20000, n)) shared by both differences, quantiles .0125 / .9875
(numpy's default linear rule).  An interval that crosses the threshold is INCONCLUSIVE; the
complementary determinate sides keep the RC run's names (WORSE_THAN_SNA: upper end < 0;
INFERIOR_TO_FREEZE: upper end < -0.005).  An end equal to a threshold does not meet it.
Any failed / missing run makes the registered verdict INCOMPLETE_OR_FAILED.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("R", "SNA", "SNAfrz1", "NEO")
SEEDS = tuple(range(2000, 2010))
N_TASKS = 30
LATE = (21, 23, 25, 27, 29)
EARLY = (1, 3, 5, 7, 9)
HARD = tuple(range(1, 31, 2))
FRESH_TASK = 29
MARGIN = 0.005
N_BOOT = 20000
Q_MAIN = (0.0125, 0.9875)
Q_AUX = (0.025, 0.975)


def boot_index(n: int) -> np.ndarray:
    return np.random.default_rng(1018).integers(0, n, size=(N_BOOT, n))


def interval(d: np.ndarray, idx: np.ndarray, q) -> dict:
    d = np.asarray(d, dtype=np.float64)
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, q)
    t9 = 2.2621571627409915                          # t(0.975, 9), descriptive only
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"mean": float(d.mean()), "interval": [float(lo), float(hi)], "quantiles": list(q),
            "median": float(np.median(d)), "positive": int((d > 0).sum()), "negative": int((d < 0).sum()),
            "n": int(len(d)), "per_seed": [float(x) for x in d],
            "t_interval_95_descriptive": [float(d.mean() - t9 * se), float(d.mean() + t9 * se)]}


def label_d1(iv) -> str:
    lo, hi = iv["interval"]
    return "BETTER_THAN_SNA" if lo > 0 else "WORSE_THAN_SNA" if hi < 0 else "INCONCLUSIVE"


def label_d2(iv) -> str:
    lo, hi = iv["interval"]
    return ("NONINFERIOR_TO_FREEZE" if lo > -MARGIN else "INFERIOR_TO_FREEZE" if hi < -MARGIN
            else "INCONCLUSIVE")


def read_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else float("nan")


def compute(src: Path, seeds=SEEDS, n_tasks=N_TASKS) -> dict:
    prov = json.loads((src / "provenance.json").read_text())
    status = prov.get("status")
    if status == "RUNNING":
        raise SystemExit("the run is still RUNNING; the verdict is computed once, after it ends")
    rows = read_csv(src / "per_task.csv")
    keys = [(r["arm"], int(r["seed"]), int(r["task"])) for r in rows]
    expected = {(a, s, t) for a in ARMS for s in seeds for t in range(1, n_tasks + 1)}
    problems = []
    if status != "COMPLETED":
        problems.append(f"status {status}")
    if len(keys) != len(set(keys)):
        problems.append("duplicate task records")
    missing = sorted(expected - set(keys))
    if missing:
        problems.append(f"{len(missing)} missing task records")
    extra = sorted(set(keys) - expected)
    if extra:
        problems.append(f"{len(extra)} unexpected task records")
    R = {k: r for k, r in zip(keys, rows)}
    bad = [k for k, r in R.items() if not (0 <= fnum(r["online_acc"]) <= 1) or r["diverged"] == "True"
           or int(r["bad_step"]) >= 0]
    if bad:
        problems.append(f"{len(bad)} non-finite / diverged records")
    on = {k: fnum(r["online_acc"]) for k, r in R.items()}
    # the three Snake arms share task 1 exactly
    t1_same = all(on.get((a, s, 1)) == on.get(("SNA", s, 1)) for a in ("SNAfrz1", "NEO") for s in seeds)
    if not t1_same:
        problems.append("Snake arms differ in task 1")
    complete = not problems

    def per_arm(field, tasks, agg=np.mean):
        out = {}
        for a in ARMS:
            vals = []
            for s in seeds:
                xs = [fnum(R[(a, s, t)][field]) for t in tasks if (a, s, t) in R]
                vals.append(float(agg(xs)) if len(xs) == len(tasks) else float("nan"))
            out[a] = vals
        return out

    A = per_arm("online_acc", LATE)
    idx = boot_index(len(seeds))
    res = {"status": status, "complete": complete, "problems": problems,
           "n_records": len(rows), "expected_records": len(expected),
           "snake_task1_identical": t1_same, "seeds": list(seeds),
           "late_window": list(LATE), "early_window": list(EARLY),
           "bootstrap": {"n": N_BOOT, "rng": "numpy default_rng(1018)",
                         "index": "rng.integers(0, n, size=(20000, n)), shared by every difference",
                         "main_quantiles": list(Q_MAIN), "aux_quantiles": list(Q_AUX)}}
    finite = all(np.isfinite(A[a]).all() for a in ("NEO", "SNA", "SNAfrz1"))
    if finite:
        d1 = np.subtract(A["NEO"], A["SNA"])
        d2 = np.subtract(A["NEO"], A["SNAfrz1"])
        iv1, iv2 = interval(d1, idx, Q_MAIN), interval(d2, idx, Q_MAIN)
        iv1["label"], iv2["label"] = label_d1(iv1), label_d2(iv2)
        iv2["margin"] = -MARGIN
        iv2["superior_to_freeze_same_interval"] = iv2["interval"][0] > 0
        res["D1_NEO_minus_SNA"] = iv1
        res["D2_NEO_minus_SNAfrz1"] = iv2
    if complete and finite:
        both = iv1["label"] == "BETTER_THAN_SNA" and iv2["label"] == "NONINFERIOR_TO_FREEZE"
        res["registered_verdict"] = {"D1": iv1["label"], "D2": iv2["label"], "both_met": both}
    else:
        res["registered_verdict"] = {"D1": None, "D2": None, "both_met": None,
                                     "label": "INCOMPLETE_OR_FAILED"}
    # ---- auxiliary (spec §5.2): not confirmatory
    aux = {"A_late": A, "A_all_hard": per_arm("online_acc", HARD), "A_early": per_arm("online_acc", EARLY),
           "first100_late": per_arm("first100_online_acc", LATE),
           "train_acc_late": per_arm("train_acc", LATE), "test_acc_late": per_arm("test_acc", LATE),
           "train_acc_all_hard": per_arm("train_acc", HARD), "test_acc_all_hard": per_arm("test_acc", HARD)}
    aux["early_minus_late"] = {a: list(np.subtract(aux["A_early"][a], A[a])) for a in ARMS}
    aux["vs_R_late"] = {}
    for a in ("NEO", "SNA", "SNAfrz1"):
        d = np.subtract(A[a], A["R"])
        if np.isfinite(d).all():
            aux["vs_R_late"][f"{a}_minus_R"] = interval(d, idx, Q_AUX)
    for name, field in (("test_late", "test_acc"), ("first100_late", "first100_online_acc")):
        X = per_arm(field, LATE)
        for a, b in (("NEO", "SNA"), ("NEO", "SNAfrz1")):
            d = np.subtract(X[a], X[b])
            if np.isfinite(d).all():
                aux.setdefault("other_paired", {})[f"{name}_{a}_minus_{b}"] = interval(d, idx, Q_AUX)
    aux["curves_hard_online_mean_over_seeds"] = {
        a: {t: float(np.mean([on[(a, s, t)] for s in seeds if (a, s, t) in on])) for t in HARD}
        for a in ARMS}
    res["auxiliary"] = aux
    # ---- fresh control (task 29)
    fpath = src / "fresh_control.csv"
    if fpath.exists():
        fr = read_csv(fpath)
        fc = {}
        for a in ARMS:
            q = [r for r in fr if r["arm"] == a]
            fc[a] = {k: [fnum(r[k]) for r in sorted(q, key=lambda r: int(r["seed"]))]
                     for k in ("fresh_online_acc", "continual_online_acc", "fresh_gap_online",
                               "fresh_test_acc", "continual_test_acc", "fresh_gap_test")}
            fc[a]["snake_twins_identical"] = [r["snake_fresh_identical_across_arms"] for r in q]
        res["fresh_control"] = {"task": int(fr[0]["task"]) if fr else None, "per_arm": fc}
    # ---- descriptive control and width readouts (spec §6): never a verdict
    desc = {}
    cal = json.loads((src / "calibration.json").read_text())["runs"] if (src / "calibration.json").exists() else {}
    desc["lref"] = {a: [cal.get(f"{a}_seed{s}", {}).get("lref") for s in seeds] for a in ("SNA", "SNAfrz1", "NEO")}
    desc["lref_floor_hits"] = sum(bool(v.get("floor_hit")) for v in cal.values())
    for f in ("gate_mean", "gate_first10", "gate_last10", "mult_mean", "sum_beta_eff", "switch_ce", "ce_step1"):
        desc[f] = {a: {t: float(np.nanmean([fnum(R[(a, s, t)][f]) for s in seeds if (a, s, t) in R]))
                       for t in range(1, n_tasks + 1)} for a in ARMS}
    for site in ("c1", "c2", "f1", "f2"):
        for f in (f"alpha_med_{site}", f"two_alpha_sqrtV_med_{site}", f"zsd_med_{site}", f"dmean_med_{site}"):
            desc[f] = {a: {t: float(np.nanmean([fnum(R[(a, s, t)][f]) for s in seeds if (a, s, t) in R]))
                           for t in range(1, n_tasks + 1)} for a in ARMS}
    bpath = src / "per_block.csv"
    if bpath.exists():
        prof = {}
        for r in read_csv(bpath):
            if r["arm"] != "NEO" or int(r["task"]) == 1:
                continue
            prof.setdefault(int(r["block"]), []).append(fnum(r["g_mean"]))
        desc["neo_g_by_block_tasks2to30"] = {b: float(np.mean(v)) for b, v in sorted(prof.items())}
    res["descriptive"] = desc
    return res


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.2f}"


def pp(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}"


def summary_md(res: dict) -> str:
    L = ["# neokunekune_cifar5p1_cnn_1018 — 判定", ""]
    rv = res["registered_verdict"]
    L.append(f"状態 `{res['status']}`、記録 {res['n_records']}/{res['expected_records']} 行。"
             + ("" if res["complete"] else f" 不備: {'; '.join(res['problems'])}。"))
    L.append("")
    if rv.get("label") == "INCOMPLETE_OR_FAILED":
        L.append("**登録判定: INCOMPLETE_OR_FAILED**（10 seed 完走が前提の確認的判定は出さない）。")
    else:
        L.append(f"**登録判定: D1 = {rv['D1']}、D2 = {rv['D2']}**"
                 + ("。両方を満たした。" if rv["both_met"] else "。"))
    L.append("")
    A = res["auxiliary"]["A_late"]
    L += ["## 主量（後期 hard t=21,23,25,27,29 の online 精度、seed 平均）", "",
          "| 腕 | 主量 A (%) | 全 hard (%) | 早期 (%) | 早期−後期 (pt) |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        L.append(f"| {a} | {pct(np.mean(A[a]))} | {pct(np.mean(res['auxiliary']['A_all_hard'][a]))} | "
                 f"{pct(np.mean(res['auxiliary']['A_early'][a]))} | "
                 f"{pp(np.mean(res['auxiliary']['early_minus_late'][a]))} |")
    L.append("")
    for key, name in (("D1_NEO_minus_SNA", "D1 = NEO − SNA"), ("D2_NEO_minus_SNAfrz1", "D2 = NEO − SNAfrz1")):
        if key in res:
            iv = res[key]
            L.append(f"- **{name}**: 平均 {pp(iv['mean'])} pt、97.5% 区間 [{pp(iv['interval'][0])}, "
                     f"{pp(iv['interval'][1])}] pt、正 {iv['positive']}/{iv['n']}、ラベル `{iv['label']}`。")
    L += ["", "## seed 別の主量 (%)", "", "| seed | " + " | ".join(ARMS) + " | D1 (pt) | D2 (pt) |",
          "|---|" + "---:|" * (len(ARMS) + 2)]
    for i, s in enumerate(res["seeds"]):
        d1 = res.get("D1_NEO_minus_SNA", {}).get("per_seed", [float("nan")] * len(res["seeds"]))[i]
        d2 = res.get("D2_NEO_minus_SNAfrz1", {}).get("per_seed", [float("nan")] * len(res["seeds"]))[i]
        L.append(f"| {s} | " + " | ".join(pct(A[a][i]) for a in ARMS) + f" | {pp(d1)} | {pp(d2)} |")
    L += ["", "## 補助量（確認的判定ではない）", ""]
    aux = res["auxiliary"]
    L += ["| 腕 | 最初100更新 後期 (%) | train 後期 (%) | test 後期 (%) | test 全 hard (%) |",
          "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        L.append(f"| {a} | {pct(np.mean(aux['first100_late'][a]))} | {pct(np.mean(aux['train_acc_late'][a]))} | "
                 f"{pct(np.mean(aux['test_acc_late'][a]))} | {pct(np.mean(aux['test_acc_all_hard'][a]))} |")
    L.append("")
    for k, iv in aux.get("vs_R_late", {}).items():
        L.append(f"- {k}: {pp(iv['mean'])} pt、95% 区間 [{pp(iv['interval'][0])}, {pp(iv['interval'][1])}]、"
                 f"正 {iv['positive']}/{iv['n']}")
    for k, iv in aux.get("other_paired", {}).items():
        L.append(f"- {k}: {pp(iv['mean'])} pt、95% 区間 [{pp(iv['interval'][0])}, {pp(iv['interval'][1])}]")
    if "fresh_control" in res:
        fc = res["fresh_control"]
        L += ["", f"## fresh 対照（課題 {fc['task']}）", "",
              "| 腕 | fresh online (%) | 継続 online (%) | gap (pt) | fresh test (%) | 継続 test (%) |",
              "|---|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            q = fc["per_arm"][a]
            L.append(f"| {a} | {pct(np.mean(q['fresh_online_acc']))} | {pct(np.mean(q['continual_online_acc']))} | "
                     f"{pp(np.mean(q['fresh_gap_online']))} | {pct(np.mean(q['fresh_test_acc']))} | "
                     f"{pct(np.mean(q['continual_test_acc']))} |")
    d = res["descriptive"]
    L += ["", "## 制御の記述（判定ではない）", "",
          f"- Lref（NEO、seed 平均）: {np.mean([x for x in d['lref']['NEO'] if x is not None]):.4f}、"
          f"床（1e-4）に当たった run: {d['lref_floor_hits']}。"]
    g = d["gate_mean"]["NEO"]
    L.append(f"- NEO の課題内平均 g（課題 2–30 の平均）: {np.mean([g[t] for t in range(2, 31)]):.3f}、"
             f"課題の最初 10 更新 {np.mean([d['gate_first10']['NEO'][t] for t in range(2, 31)]):.3f}、"
             f"最後 10 更新 {np.mean([d['gate_last10']['NEO'][t] for t in range(2, 31)]):.3f}。")
    L.append("")
    L.append("数値の全量は `verdict.json`。主量は online（更新前）精度で、test 精度・幅・制御は別の読み。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seeds", default=None, help="tests only: comma list")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
    res = compute(a.src, seeds)
    out = a.out or a.src
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "summary.md").write_text(summary_md(res))
    print(json.dumps(res["registered_verdict"], indent=2))


if __name__ == "__main__":
    main()
