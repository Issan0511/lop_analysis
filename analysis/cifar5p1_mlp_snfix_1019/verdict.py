#!/usr/bin/env python3
"""Registered comparisons of cifar5p1_mlp_snfix_1019 (spec §5).  Fixed before the production run.

    python3 analysis/cifar5p1_mlp_snfix_1019/verdict.py --root results/cifar5p1_mlp_snfix_1019

The per-run windows, the exact two-sided sign test (zeros dropped), Holm and the A_WINS / B_WINS / TIE
rule are imported from the 0920 report, not re-written.  Family N: the ten registered head-to-heads of
the new arms.  Family vs_best: the leader by median late window against the other 18 arms.  Q1-Q3 of the
new arms are read on seed medians.  S-reuse-0920 compares the 16 rerun cells' per_task.csv and
fresh_control.csv with the committed 0920 files byte for byte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import cifar5p1_mlp_0920_report as RPT    # noqa: E402

NEW = ("SN06", "SN3", "SNAfrz1")
OLD = ("SNA", "KKA", "R", "LR", "KKT1", "KKA23", "SL", "RSL", "LK001", "LK03", "ELU", "SILU", "GELU",
       "LK07", "CR", "DF")
ALL = OLD + NEW
SNAKE = ("SNA", "KKA", "KKA23", "KKT1") + NEW
FAMILY_N = (("SN06", "SNA"), ("SN3", "SNA"), ("SNAfrz1", "SNA"), ("SN06", "KKT1"), ("SN3", "KKT1"),
            ("SNAfrz1", "KKT1"), ("SN06", "ELU"), ("SN3", "ELU"), ("SNAfrz1", "ELU"), ("SN06", "SN3"))
COLLAPSE = RPT.COLLAPSE
HOST = Path("results") / "cifar5p1_mlp_0920"
TRACE_TASKS = (1, 2, 9, 10, 19, 20, 29, 30)


def cell(arm: str) -> str:
    return f"{arm}_std_lr0.0001"


def judge(tests: list[dict]) -> list[dict]:
    """Holm within the family, then the 0920 rule."""
    for t, q in zip(tests, RPT.holm([t["p"] for t in tests])):
        t["p_holm"] = q
        t["verdict"] = ("A_WINS" if q < 0.05 and t["median_diff"] > 0 else
                        "B_WINS" if q < 0.05 and t["median_diff"] < 0 else "TIE")
    return tests


def reuse_0920(root: Path, host: Path) -> dict:
    per, ok = {}, True
    for arm in OLD:
        same = {}
        for f in ("per_task.csv", "fresh_control.csv"):
            a, b = root / cell(arm) / f, host / cell(arm) / f
            same[f] = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        per[arm] = same
        ok &= all(same.values())
    return {"label": "PASS" if ok else "FAIL", "per_arm": per,
            "identical_files": sum(v for s in per.values() for v in s.values()), "files": 2 * len(OLD)}


def compute(root: Path, host: Path | None = None) -> dict:
    host = host or (root.resolve().parent / "cifar5p1_mlp_0920")     # results/cifar5p1_mlp_0920
    tasks, fresh, prov = RPT.load(root)                # refuses unfinished cells
    present = set(tasks.cell)
    missing = [cell(a) for a in ALL if cell(a) not in present]
    tasks = tasks[tasks.cell.isin([cell(a) for a in ALL])]
    fresh = fresh[fresh.cell.isin([cell(a) for a in ALL])]
    r = RPT.per_run(tasks)
    gap = fresh.set_index(["cell", "seed"]).fresh_gap
    r["fresh_gap"] = [gap.get((c, s), np.nan) for c, s in zip(r.cell, r.seed)]
    last = tasks[tasks.task == RPT.HARD[-1]].set_index(["cell", "seed"])
    r["mob_l2"] = [last.mob_l2.get((c, s), np.nan) for c, s in zip(r.cell, r.seed)]

    fam_n = []
    for a, b in FAMILY_N:
        if cell(a) in present and cell(b) in present:
            fam_n.append(RPT.pair(r, cell(a), cell(b)))
        else:
            fam_n.append({"a": cell(a), "b": cell(b), "col": "window", "n": 0, "a_wins": 0,
                          "median_diff": float("nan"), "p": 1.0, "incomplete": True})
    judge(fam_n)
    for t in fam_n:
        if t.get("incomplete"):
            t["verdict"] = "INCOMPLETE"
    cells = [cell(a) for a in ALL if cell(a) in present]
    ranked = sorted(cells, key=lambda c: -r[r.cell == c].window.median())
    best = ranked[0]
    vs_best = judge([RPT.pair(r, c, best) for c in ranked[1:]])

    table = []
    for c in ranked:
        g = r[r.cell == c]
        table.append({"cell": c, "arm": g.arm.iloc[0], "window": float(g.window.median()),
                      "early": float(g.early.median()), "drop": float(g["drop"].median()),
                      "fresh_gap": float(g.fresh_gap.median()), "gap_pos": int((g.fresh_gap > 0).sum()),
                      "n": int(len(g)), "dead_l2": float(g.dead_l2.median()), "zeroout_l2": float(g.zeroout_l2.median()),
                      "eff_rank_l2": float(g.eff_rank_l2.median()), "mob_l2": float(g.mob_l2.median()),
                      "w_norm_l1": float(g.w_norm_l1.median()), "train_end": float(g.train_end.median()),
                      "test_end": float(g.test_end.median()), "diverged": int(g.diverged.sum())})
    by_arm = {t["arm"]: t for t in table}
    q = {a: {"Q1_drop_pos": by_arm[a]["drop"] > 0, "Q2_gap_pos": by_arm[a]["fresh_gap"] > 0,
             "Q3_collapse": by_arm[a]["window"] < COLLAPSE} for a in NEW if a in by_arm}

    trace = {}
    for a in SNAKE:
        g = tasks[tasks.cell == cell(a)]
        if g.empty or "two_alpha_W_med_l1" not in g:
            continue
        trace[a] = {f"t{t}": {k: float(g[g.task == t][k].median()) for k in
                              ("two_alpha_W_med_l1", "two_alpha_W_med_l2", "alpha_med_l1", "alpha_med_l2",
                               "zsd_l1", "zsd_l2", "online_acc")}
                    for t in TRACE_TASKS if (g.task == t).any()}
    frozen = None
    fa = root / cell("SNAfrz1") / "frozen_alpha.json"
    if fa.exists():
        d = json.loads(fa.read_text())
        frozen = {"median_l1": float(np.median(d["l1"])), "median_l2": float(np.median(d["l2"])),
                  "unchanged_at_later_task_starts": d.get("unchanged_at_later_task_starts")}

    v = {f"{t['a'].split('_')[0]}_vs_{t['b'].split('_')[0]}": t["verdict"] for t in fam_n}
    reading = []
    if v.get("SN06_vs_SNA") == "TIE" and v.get("SNAfrz1_vs_SNA") == "TIE":
        reading.append("この箱では α を幅に合わせて動かしても止めても、後期窓は変わらない。")
    for a in NEW:
        if v.get(f"{a}_vs_SNA") == "A_WINS":
            reading.append(f"{a} は SNA に勝った。Snake 族の代表の候補として記録する。")
    reuse = reuse_0920(root, host)
    leader = str(r[r.cell == best].arm.iloc[0])
    pred = {"P1_SN3_vs_SNA_B_WINS": v.get("SN3_vs_SNA") == "B_WINS",
            "P2_SN06_vs_SNA_TIE": v.get("SN06_vs_SNA") == "TIE",
            "P3_SNAfrz1_vs_SNA_TIE": v.get("SNAfrz1_vs_SNA") == "TIE",
            "P4_new_arms_all_Q2": len(q) == 3 and all(x["Q2_gap_pos"] for x in q.values()),
            "P5_leader_is_snake": leader in SNAKE,
            "P6_reuse_pass": reuse["label"] == "PASS"}
    return {"missing_cells": missing, "family_N": fam_n, "vs_best": {"leader": leader, "tests": vs_best},
            "table": table, "Q_new_arms": q, "two_alpha_W_trace": trace, "SNAfrz1_frozen_alpha": frozen,
            "readings": reading, "S_reuse_0920": reuse, "predictions_claude": pred,
            "late_window": RPT.LATE, "early_window": RPT.EARLY}


def f(x, spec=".4f"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def summary_md(res: dict) -> str:
    L = ["# cifar5p1_mlp_snfix_1019 — 5+1 CIFAR の MLP に固定 Snake を足した 19 腕", ""]
    if res["missing_cells"]:
        L += [f"欠けている腕: {'、'.join(res['missing_cells'])}。", ""]
    L += ["## 19 腕の表", "",
          "後期窓は hard 課題 21・23・25・27・29 の更新前 online 精度の平均。値は seed の中央値。"
          "この表は対応なしで、優劣は下の対応符号検定で読む。", "",
          "| 順位 | 腕 | 後期窓 | 早期 | 低下 | fresh gap | gap > 0 | 第 2 層の死 | mob l2 | eff_rank l2 | test 末 | 発散 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, t in enumerate(res["table"], 1):
        star = " ★" if t["arm"] in NEW else ""
        L.append(f"| {i} | `{t['arm']}`{star} | {f(t['window'])} | {f(t['early'])} | {f(t['drop'], '+.4f')} | "
                 f"{f(t['fresh_gap'], '+.4f')} | {t['gap_pos']}/{t['n']} | {f(t['dead_l2'], '.3f')} | {f(t['mob_l2'], '.3f')} | "
                 f"{f(t['eff_rank_l2'], '.1f')} | {f(t['test_end'], '.3f')} | {t['diverged']} |")
    L += ["", "★ は今回足した腕。", "", "## 族 N（新しい腕の 10 比較、Holm）", "",
          "| a | b | a − b の中央値 | a が勝った seed | n | p | p_holm | 判定 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for t in res["family_N"]:
        L.append(f"| `{t['a'].split('_')[0]}` | `{t['b'].split('_')[0]}` | {f(t['median_diff'], '+.4f')} | {t['a_wins']} | "
                 f"{t['n']} | {f(t['p'])} | {f(t.get('p_holm'))} | `{t['verdict']}` |")
    vb = res["vs_best"]
    L += ["", f"## 首位 `{vb['leader']}` と残り 18 腕（Holm 18）", "",
          "| 腕 | 首位との差の中央値 | 勝った seed | n | p_holm | 判定 |", "|---|---:|---:|---:|---:|---|"]
    for t in vb["tests"]:
        L.append(f"| `{t['a'].split('_')[0]}` | {f(t['median_diff'], '+.4f')} | {t['a_wins']} | {t['n']} | "
                 f"{f(t['p_holm'])} | `{t['verdict']}` |")
    L += ["", "## 新しい 3 腕の読み", ""]
    for a, x in res["Q_new_arms"].items():
        L.append(f"- `{a}`: Q1 低下 > 0 は {'成り立つ' if x['Q1_drop_pos'] else '成り立たない'}。"
                 f"Q2 fresh gap > 0 は {'成り立つ' if x['Q2_gap_pos'] else '成り立たない'}。"
                 f"Q3 崩壊は {'起きた' if x['Q3_collapse'] else '起きていない'}。")
    for s in res["readings"]:
        L.append(f"- {s}")
    if res["SNAfrz1_frozen_alpha"]:
        fz = res["SNAfrz1_frozen_alpha"]
        L.append(f"- SNAfrz1 の固定時の α の中央値は、第 1 層 {fz['median_l1']:.4f}、第 2 層 {fz['median_l2']:.4f}。")
    L += ["", "## 2αW の中央値（記述、判定には使わない）", "", "| 腕 | 層 | " + " | ".join(f"課題 {t}" for t in TRACE_TASKS) + " |",
          "|---|---|" + "---:|" * len(TRACE_TASKS)]
    for a, tr in res["two_alpha_W_trace"].items():
        for l in (1, 2):
            L.append(f"| `{a}` | {l} | " + " | ".join(f(tr.get(f't{t}', {}).get(f'two_alpha_W_med_l{l}'), '.2f')
                                                     for t in TRACE_TASKS) + " |")
    ru = res["S_reuse_0920"]
    L += ["", f"S-reuse-0920 は `{ru['label']}`。16 腕の 32 ファイルのうち {ru['identical_files']} がバイト単位で一致した。", "",
          "## Claude の予測の採点", ""]
    for k, ok in res["predictions_claude"].items():
        L.append(f"- {k}: {'的中' if ok else '外れ'}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    a = ap.parse_args()
    res = compute(a.root)
    (a.root / "verdict.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (a.root / "summary.md").write_text(summary_md(res))
    pd.DataFrame([{**t, "family": "N"} for t in res["family_N"]]
                 + [{**t, "family": "vs_best"} for t in res["vs_best"]["tests"]]).to_csv(a.root / "paired_tests.csv", index=False)
    print({f"{t['a'].split('_')[0]}-{t['b'].split('_')[0]}": t["verdict"] for t in res["family_N"]})
    print("leader", res["vs_best"]["leader"], "reuse", res["S_reuse_0920"]["label"])


if __name__ == "__main__":
    main()
