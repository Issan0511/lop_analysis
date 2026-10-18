#!/usr/bin/env python3
"""Registered comparisons of cifar5p1_cnn_budget_1019 (spec §6).  Fixed before the production run.

    python3 analysis/cifar5p1_cnn_budget_1019/verdict.py --root results/cifar5p1_cnn_budget_1019

Per (arm, budget, seed), over the late hard tasks 21, 23, 25, 27, 29: F1 first-780-update online,
F2 whole-task online, F3 task-29 fresh - continual (whole-task online), F4 log FC2 width at boundary
30 (median over units of the probe SD).  Contrasts budget B - 780 for B in {2340, 7800} and the four
arms: 8 per family.  Percentile bootstrap over seeds from one index matrix
default_rng(1019).integers(0, 10, size=(20000, 10)); Bonferroni over 8, quantiles .003125 / .996875;
lower > 0 -> HIGHER, upper < 0 -> LOWER, else INCONCLUSIVE.  Descriptive quantities use 95%
(.025 / .975) from the same index matrix.  Reading rules of spec §6 are applied as written; the
2340 rows apply the same rules with 2340 in place of 7800 and are marked as that analogue.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ("R", "ELU", "KKT1", "SNAfrz1")
BUDGETS = (780, 2340, 7800)
LONG = (2340, 7800)
SEEDS = tuple(range(2000, 2010))
LATE = (21, 23, 25, 27, 29)
N_TASKS = 30
FRESH = (9, 19, 29)
N_BOOT = 20000
Q8 = (0.05 / 8 / 2, 1 - 0.05 / 8 / 2)
Q95 = (0.025, 0.975)
SITES = ("c1", "c2", "f1", "f2")
TRAJ_FIELDS = ("s_med", "m_med", "r_med", "dmean_med", "ppos_med", "dead_frac", "lowresp_frac", "zeroout_frac",
               "eff_rank")
FRESH_FIELDS = ("gap_online_acc", "gap_first780_online_acc", "gap_first100_online_acc", "gap_test_acc")
PREV = ("results", "neokunekune_methods_cifar5p1_cnn_1018", "main")
REUSE_EXCLUDED = ("slot",)          # bundle slot positions differ between the two runs by construction


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def read(root: Path):
    data = {}
    for b in BUDGETS:
        d = root / f"b{b}"
        p = d / "provenance.json"
        status = json.loads(p.read_text()).get("status") if p.exists() else "MISSING"
        rd = lambda n: read_csv(d / n) if (d / n).exists() else []
        fa = d / "frozen_alpha.json"
        rows, probe, fresh = rd("per_task.csv"), rd("probe_stats.csv"), rd("fresh_control.csv")
        data[b] = {"status": status, "rows": rows, "probe": probe, "fresh": fresh,
                   "frozen": json.loads(fa.read_text())["runs"] if fa.exists() else {},
                   "I_rows": {(r["arm"], int(r["seed"]), int(r["task"])): r for r in rows},
                   "I_probe": {(r["arm"], int(r["seed"]), r["site"], int(r["boundary"])): r for r in probe},
                   "I_fresh": {(r["arm"], int(r["seed"]), int(r["task"])): r for r in fresh},
                   "dups": (len(rows) - len({(r["arm"], r["seed"], r["task"]) for r in rows}))
                           + (len(probe) - len({(r["arm"], r["seed"], r["site"], r["boundary"]) for r in probe}))
                           + (len(fresh) - len({(r["arm"], r["seed"], r["task"]) for r in fresh}))}
    return data


def complete(data, seeds=SEEDS) -> dict:
    out = {}
    for b, d in data.items():
        for a in ARMS:
            want = {(a, s, t) for s in seeds for t in range(1, N_TASKS + 1)}
            have = {k for k in d["I_rows"] if k[0] == a}
            ok = d["status"] == "COMPLETED" and d["dups"] == 0 and want <= have
            ok = ok and all(d["I_rows"][k]["diverged"] == "False" and int(d["I_rows"][k]["bad_step"]) < 0 for k in want)
            ok = ok and all((a, s, site, bd) in d["I_probe"] for s in seeds for site in SITES
                            for bd in range(0, N_TASKS + 1))
            ok = ok and all((a, s, t) in d["I_fresh"] and int(d["I_fresh"][(a, s, t)]["fresh_bad_step"]) < 0
                            for s in seeds for t in FRESH)
            out[f"{a}|{b}"] = bool(ok)
    return out


def late_mean(d, arm, field, seeds=SEEDS, tasks=LATE):
    v = []
    for s in seeds:
        xs = [f(d["I_rows"][(arm, s, t)][field]) for t in tasks if (arm, s, t) in d["I_rows"]]
        v.append(float(np.mean(xs)) if len(xs) == len(tasks) else float("nan"))
    return np.asarray(v)


def probe_val(d, arm, site, boundary, field, seeds=SEEDS):
    return np.asarray([f(d["I_probe"][(arm, s, site, boundary)].get(field)) if (arm, s, site, boundary) in d["I_probe"]
                       else float("nan") for s in seeds])


def fresh_val(d, arm, task, field, seeds=SEEDS):
    return np.asarray([f(d["I_fresh"][(arm, s, task)][field]) if (arm, s, task) in d["I_fresh"] else float("nan")
                       for s in seeds])


def frozen_med(d, site, seeds=SEEDS):
    out = []
    for s in seeds:
        q = d["frozen"].get(f"SNAfrz1_lr0.0003|{s}")
        out.append(float(np.median(q[site])) if q else float("nan"))
    return np.asarray(out)


def iv(x, idx, q, keep_seeds=True) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        return {"label": "INCOMPLETE", "n_finite": int(np.isfinite(x).sum()),
                "mean_finite": float(np.nanmean(x)) if np.isfinite(x).any() else None}
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, q)
    out = {"mean": float(x.mean()), "interval": [float(lo), float(hi)], "quantiles": list(q),
           "positive": int((x > 0).sum()), "negative": int((x < 0).sum())}
    if keep_seeds:
        out["per_seed"] = [float(v) for v in x]
    return out


def label(x: dict) -> str:
    if x.get("label") == "INCOMPLETE":
        return "INCOMPLETE"
    lo, hi = x["interval"]
    return "HIGHER" if lo > 0 else "LOWER" if hi < 0 else "INCONCLUSIVE"


def below0(x: dict) -> bool:
    return "interval" in x and x["interval"][1] < 0


def above0(x: dict) -> bool:
    return "interval" in x and x["interval"][0] > 0


def reuse_780(root: Path, d780) -> dict:
    """S-reuse-780 (spec §6): the 780 budget against the 12-arm main per_task, shared columns, bitwise."""
    prev_dir = root.resolve().parents[1] / Path(*PREV)
    rows, header = [], None
    for b in ("A", "B"):
        p = prev_dir / b / "per_task.csv"
        if not p.exists():
            return {"applicable": False, "reason": f"missing {p}"}
        with open(p) as fh:
            rd = csv.DictReader(fh)
            header = rd.fieldnames if header is None else [c for c in header if c in rd.fieldnames]
            rows += list(rd)
    if not d780["rows"]:
        return {"applicable": False, "reason": "no 780 rows"}
    new_header = list(d780["rows"][0])
    cols = [c for c in header if c in new_header and c not in REUSE_EXCLUDED]
    prev = {(r["cfg"], r["seed"], r["task"]): r for r in rows}
    bad, missing, n, per_col = 0, 0, 0, {}
    for r in d780["rows"]:
        o = prev.get((r["cfg"], r["seed"], r["task"]))
        if o is None:
            missing += 1
            continue
        n += 1
        for c in cols:
            if o[c] != r[c]:
                bad += 1
                per_col[c] = per_col.get(c, 0) + 1
    return {"applicable": True, "rows": n, "columns": cols, "mismatches": bad, "mismatches_per_column": per_col,
            "missing": missing, "label": "PASS" if bad == 0 and missing == 0 and n == len(ARMS) * len(SEEDS) * N_TASKS
            else "FAIL"}


def compute(root: Path, seeds=SEEDS, prev_check=True) -> dict:
    data = read(root)
    if any(d["status"] == "RUNNING" for d in data.values()):
        raise SystemExit("a budget is still RUNNING; the verdict is computed once, after all end")
    comp = complete(data, seeds)
    n = len(seeds)
    idx = np.random.default_rng(1019).integers(0, n, size=(N_BOOT, n))
    Q = {}
    for b, d in data.items():
        for a in ARMS:
            Q[(a, b)] = {"first780": late_mean(d, a, "first780_online_acc", seeds),
                         "all": late_mean(d, a, "online_acc", seeds),
                         "first100": late_mean(d, a, "first100_online_acc", seeds),
                         "test": late_mean(d, a, "test_acc", seeds),
                         "fresh29": fresh_val(d, a, 29, "gap_online_acc", seeds)}
            for site in SITES:
                Q[(a, b)][f"logW30_{site}"] = np.log(probe_val(d, a, site, 30, "s_med", seeds))
                Q[(a, b)][f"logW2_{site}"] = np.log(probe_val(d, a, site, 2, "s_med", seeds))
            for fld in ("r_med", "dmean_med", "ppos_med", "dead_frac", "lowresp_frac", "zeroout_frac", "eff_rank"):
                Q[(a, b)][f"f2_{fld}_30"] = probe_val(d, a, "f2", 30, fld, seeds)
    fam_q = {"F1_first780": "first780", "F2_all": "all", "F3_fresh29": "fresh29", "F4_logW30": "logW30_f2"}
    fams = {}
    for fam, k in fam_q.items():
        out = {}
        for a in ARMS:
            for b in LONG:
                ok = comp[f"{a}|{b}"] and comp[f"{a}|780"]
                x = iv(Q[(a, b)][k] - Q[(a, 780)][k], idx, Q8) if ok else {"label": "INCOMPLETE"}
                x["label"] = label(x)
                out[f"{a}_{b}_minus_780"] = x
        fams[fam] = out
    desc = {"growth_logW30_minus_logW2": {}, "diff_B_minus_780": {}, "fresh_gaps": {}, "levels": {},
            "frozen_alpha_median": {}, "frozen_alpha_diff_B_minus_780": {}, "f2_dead_max_over_boundaries": {}}
    for a in ARMS:
        for b in BUDGETS:
            for site in SITES:
                g = Q[(a, b)][f"logW30_{site}"] - Q[(a, b)][f"logW2_{site}"]
                desc["growth_logW30_minus_logW2"][f"{a}_{b}_{site}"] = iv(g, idx, Q95)
            desc["levels"][f"{a}_{b}"] = {k: iv(v, idx, Q95, keep_seeds=False) for k, v in Q[(a, b)].items()}
            for t in FRESH:
                for fld in FRESH_FIELDS:
                    desc["fresh_gaps"][f"{a}_{b}_t{t}_{fld}"] = iv(fresh_val(data[b], a, t, fld, seeds), idx, Q95)
            dd = [probe_val(data[b], a, "f2", bd, "dead_frac", seeds) for bd in range(0, N_TASKS + 1)]
            desc["f2_dead_max_over_boundaries"][f"{a}_{b}"] = float(np.nanmax(dd)) if np.isfinite(dd).any() else None
        for b in LONG:
            for k in ("first100", "test", "f2_r_med_30", "f2_dmean_med_30", "f2_ppos_med_30", "f2_dead_frac_30",
                      "f2_lowresp_frac_30", "f2_eff_rank_30") + tuple(f"logW30_{s}" for s in SITES):
                desc["diff_B_minus_780"][f"{a}_{b}_{k}"] = iv(Q[(a, b)][k] - Q[(a, 780)][k], idx, Q95)
    for b in BUDGETS:
        for site in SITES:
            desc["frozen_alpha_median"][f"{b}_{site}"] = iv(frozen_med(data[b], site, seeds), idx, Q95)
            if b != 780:
                desc["frozen_alpha_diff_B_minus_780"][f"{b}_{site}"] = iv(
                    frozen_med(data[b], site, seeds) - frozen_med(data[780], site, seeds), idx, Q95)
    traj = {}
    for b, d in data.items():
        for a in ARMS:
            for site in SITES:
                for fld in TRAJ_FIELDS:
                    V = np.stack([probe_val(d, a, site, bd, fld, seeds) for bd in range(0, N_TASKS + 1)], axis=1)
                    mean = np.nanmean(V, axis=0) if np.isfinite(V).any() else np.full(V.shape[1], np.nan)
                    if np.isfinite(V).all():
                        lo, hi = np.quantile(V[idx].mean(axis=1), Q95, axis=0)
                    else:
                        lo = hi = np.full(V.shape[1], np.nan)
                    traj[f"{a}_{b}_{site}_{fld}"] = {"mean": [float(x) for x in mean], "lo": [float(x) for x in lo],
                                                     "hi": [float(x) for x in hi]}
    rules = {}
    for a in ARMS:
        for b in LONG:
            g = desc["growth_logW30_minus_logW2"][f"{a}_{b}_f2"]
            one = fams["F4_logW30"][f"{a}_{b}_minus_780"]["label"] == "HIGHER" and above0(g)
            two = below0(desc["diff_B_minus_780"][f"{a}_{b}_f2_r_med_30"]) and \
                below0(desc["diff_B_minus_780"][f"{a}_{b}_f2_dmean_med_30"])
            three = fams["F3_fresh29"][f"{a}_{b}_minus_780"]["label"]
            f1 = fams["F1_first780"][f"{a}_{b}_minus_780"]["label"]
            rules[f"{a}_{b}"] = {"form": "registered" if b == 7800 else "analogue (2340 in place of 7800)",
                                 "1_width_keeps_growing": bool(one), "2_negative_shift_and_response_drop": bool(two),
                                 "3_fresh_gap_label": three, "F1_label": f1,
                                 "next_intervention_candidate": bool(one and (f1 == "LOWER" or three == "HIGHER" or two))}
    elu = rules["ELU_7800"]
    eg = desc["fresh_gaps"]["ELU_7800_t29_gap_online_acc"]
    dead = Q[("ELU", 7800)]["f2_dead_frac_30"]
    elu_healthy = bool(elu["1_width_keeps_growing"] and elu["F1_label"] not in ("LOWER", "INCOMPLETE") and below0(eg)
                       and np.isfinite(dead).all() and float(np.max(dead)) == 0.0)
    res = {"status": {str(b): d["status"] for b, d in data.items()}, "complete": comp, "seeds": list(seeds),
           "late_window": list(LATE),
           "bootstrap": {"n": N_BOOT, "rng": "numpy default_rng(1019)", "index": "integers(0, n, size=(20000, n))",
                         "family_quantiles": list(Q8), "descriptive_quantiles": list(Q95)},
           "families": fams, "descriptive": desc, "trajectories": traj, "rules": rules,
           "ELU_healthy_after_growth": elu_healthy,
           "ELU_healthy_reading": "rule ① for ELU 7800, F1 ELU 7800-780 not LOWER, the 95% interval of the task-29 "
                                  "fresh - continual (7800) below 0, FC2 dead fraction 0 in every seed at boundary 30",
           "next_intervention_candidates": [k for k, v in rules.items() if v["next_intervention_candidate"]]}
    if prev_check:
        res["S_reuse_780"] = reuse_780(root, data[780])
    return res


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.2f}"


def pp(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}"


def lv(res, a, b, k):
    x = res["descriptive"]["levels"][f"{a}_{b}"][k]
    return x.get("mean", x.get("mean_finite"))


def summary_md(res) -> str:
    L = ["# cifar5p1_cnn_budget_1019 — 課題あたりの更新数の比較", ""]
    L.append("状態: " + "、".join(f"{b} 更新 `{s}`" for b, s in res["status"].items()) + "。")
    inc = [k for k, v in res["complete"].items() if not v]
    L.append(f"完走していない (腕, 予算): {'、'.join(inc) if inc else 'なし'}。")
    L += ["", "## 予算別の性能", "",
          "後期 hard 課題（21, 23, 25, 27, 29）の seed 平均。fresh − 継続は課題 29 の課題全体の online。", "",
          "| 腕 | 予算 | 最初 100 更新 (%) | 最初 780 更新 (%) | 課題全体 (%) | test (%) | fresh − 継続 (pt) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        for b in BUDGETS:
            L.append(f"| {a} | {b} | {pct(lv(res, a, b, 'first100'))} | {pct(lv(res, a, b, 'first780'))} | "
                     f"{pct(lv(res, a, b, 'all'))} | {pct(lv(res, a, b, 'test'))} | {pp(lv(res, a, b, 'fresh29'))} |")
    tr = res["trajectories"]
    L += ["", "## FC2 の幅・位置・応答", "",
          "固定 probe 上の unit の中央値を seed で平均した。境界 t は課題 t の末。", "",
          "| 腕 | 予算 | 幅 s（境界 2 → 30） | r（30） | φ′（30） | p⁺（30） | 死（30） | 低応答（30） |",
          "|---|---:|---|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        for b in BUDGETS:
            g = lambda k: tr[f"{a}_{b}_f2_{k}"]["mean"]
            s = g("s_med")
            L.append(f"| {a} | {b} | {s[2]:.2f} → {s[30]:.2f} | {g('r_med')[30]:+.2f} | {g('dmean_med')[30]:.2f} | "
                     f"{g('ppos_med')[30]:.2f} | {g('dead_frac')[30]:.2f} | {g('lowresp_frac')[30]:.2f} |")
    names = {"F1_first780": "F1 学び直し能力（最初の 780 更新の online）",
             "F2_all": "F2 長く学習した分（課題全体の online）",
             "F3_fresh29": "F3 課題 29 の fresh − 継続",
             "F4_logW30": "F4 境界 30 の FC2 幅（log）"}
    for fam, title in names.items():
        L += ["", f"## {title}", "", "予算 − 780 の seed 対応差。区間は Bonferroni 8 の 99.375%。", "",
              "| 腕・予算 | 平均 | 区間 | 正の seed | 判定 |", "|---|---:|---|---:|---|"]
        for k, x in res["families"][fam].items():
            if x["label"] == "INCOMPLETE":
                L.append(f"| {k} | — | — | — | `INCOMPLETE` |")
                continue
            fmt = (lambda v: f"{v:+.3f}") if fam == "F4_logW30" else pp
            L.append(f"| {k} | {fmt(x['mean'])} | [{fmt(x['interval'][0])}, {fmt(x['interval'][1])}] | "
                     f"{x['positive']}/{len(x['per_seed'])} | `{x['label']}` |")
    L += ["", "## 読みの規則（spec §6）", "",
          "7800 の行が登録した形。2340 の行は同じ規則を 2340 に当てた類比。", ""]
    for k, v in res["rules"].items():
        L.append(f"- {k}: ① 幅の継続成長 {'○' if v['1_width_keeps_growing'] else '×'}、② 負側への位置変化と応答低下 "
                 f"{'○' if v['2_negative_shift_and_response_drop'] else '×'}、③ fresh `{v['3_fresh_gap_label']}`、"
                 f"F1 `{v['F1_label']}`、次の介入の候補 {'○' if v['next_intervention_candidate'] else '×'}")
    L += ["", f"「ELU は幅の成長後も健全」は{'成り立つ' if res['ELU_healthy_after_growth'] else '成り立たない'}。"]
    if "S_reuse_780" in res:
        r = res["S_reuse_780"]
        L.append("S-reuse-780 は " + (f"`{r['label']}`（{r['rows']} 行 × {len(r['columns'])} 列、不一致 {r['mismatches']}、"
                                      f"欠け {r['missing']}）" if r.get("applicable") else f"適用外（{r.get('reason')}）") + "。")
    L += ["", "予算延長は幅だけを操作する介入ではない。この走だけで幅の因果効果が確立したとは扱わない。",
          "数値の全量は `verdict.json` にある。"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    a = ap.parse_args()
    res = compute(a.root)
    (a.root / "verdict.json").write_text(json.dumps(res, indent=1, default=float))
    (a.root / "summary.md").write_text(summary_md(res))
    print(json.dumps({fam: {k: v["label"] for k, v in d.items()} for fam, d in res["families"].items()}, indent=1))
    print("ELU healthy:", res["ELU_healthy_after_growth"], "candidates:", res["next_intervention_candidates"],
          "reuse:", res.get("S_reuse_780", {}).get("label"))


if __name__ == "__main__":
    main()
