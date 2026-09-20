#!/usr/bin/env python3
"""Report for cifar5p1_mlp_0920 (spec §3.3, §5, §10, addendum 1 §11.4).

    python3 src/cifar5p1_mlp_0920_report.py --src results/cifar5p1_mlp_0920

Reads every `<arm>_std_lr0.0001/` directory, scores the registered readings and writes
summary.md / paired_tests.csv / verdict.json next to them.

The window is the arm's mean online accuracy over the *hard* tasks only, which is the
paper's metric ("we measure agents' performance specifically on the hard tasks").  Arms
are compared by the **paired** difference over seeds, never by the difference of the two
medians in the table: the two can point opposite ways, which is what the ViT box's KKT1
turned out to be (memory `paired-vs-unpaired-ranking-trap`).
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cifar5p1_mlp_0920 as C

HARD = list(range(1, C.N_TASKS + 1, 2))
EARLY, LATE = HARD[:5], HARD[10:]
SEEDS = list(range(10))
COLLAPSE = 0.20                      # spec §3.3: the constant-prediction level on a hard task


def sign(diff) -> tuple[int, int, float]:
    """Exact two-sided sign test, zeros dropped (the 0918 rule)."""
    d = np.asarray(diff, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return pos, n, p


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni, returned in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, ps[i] * (len(ps) - rank)))
        out[i] = running
    return out


def load(src: Path, allow_partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`provenance.json` is written only when a cell finishes, so its absence means the
    cell is still running or died -- report that rather than quietly leaving the arm out
    of the table."""
    tasks, fresh, prov, unfinished = [], [], {}, []
    for d in sorted(src.iterdir()):
        f = d / "per_task.csv"
        if not f.exists() or d.name.startswith("_"):
            continue
        if not (d / "provenance.json").exists():
            unfinished.append(d.name)
            continue
        p = json.loads((d / "provenance.json").read_text())
        prov[d.name] = {k: p[k] for k in ("arm", "cond", "lr", "hidden", "n_params",
                                          "n_tasks", "steps_per_task", "divergences")}
        t = pd.read_csv(f)
        t["cell"] = d.name
        t["hidden"] = p["hidden"]
        tasks.append(t)
        if (d / "fresh_control.csv").exists():
            g = pd.read_csv(d / "fresh_control.csv")
            g["cell"] = d.name
            g["hidden"] = p["hidden"]
            fresh.append(g)
    if not tasks:
        raise SystemExit(f"no finished runs under {src} (unfinished: {unfinished})")
    if unfinished and not allow_partial:
        raise SystemExit(f"unfinished cells: {unfinished}  (pass --allow-partial to report anyway)")
    return pd.concat(tasks, ignore_index=True), pd.concat(fresh, ignore_index=True), prov


def per_run(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, seed): the registered windows and the end-of-task fit."""
    rows = []
    for (cell, seed), g in t.groupby(["cell", "seed"]):
        g = g.set_index("task")
        hard = g[g.index.isin(HARD)]
        have = lambda ts: [x for x in ts if x in g.index and not np.isnan(g.loc[x, "online_acc"])]
        early, late = have(EARLY), have(LATE)
        rows.append({
            "cell": cell, "arm": g["arm"].iloc[0], "hidden": int(g["hidden"].iloc[0]), "seed": seed,
            "window": float(g.loc[late, "online_acc"].mean()) if late else float("nan"),
            "early": float(g.loc[early, "online_acc"].mean()) if early else float("nan"),
            "diverged": len(hard) < len(HARD),
            "train_end": float(g.loc[HARD[-1], "train_acc"]) if HARD[-1] in g.index else float("nan"),
            "test_end": float(g.loc[HARD[-1], "test_acc"]) if HARD[-1] in g.index else float("nan"),
            "dead_l2": float(g.loc[HARD[-1], "dead_frac_l2"]) if HARD[-1] in g.index else float("nan"),
            "zeroout_l2": float(g.loc[HARD[-1], "zeroout_l2"]) if HARD[-1] in g.index else float("nan"),
            "eff_rank_l2": float(g.loc[HARD[-1], "eff_rank_l2"]) if HARD[-1] in g.index else float("nan"),
            "w_norm_l1": float(g.loc[HARD[-1], "w_norm_l1"]) if HARD[-1] in g.index else float("nan")})
    r = pd.DataFrame(rows)
    r["drop"] = r.early - r.window
    return r.sort_values(["cell", "seed"]).reset_index(drop=True)


def pair(r: pd.DataFrame, a: str, b: str, col: str = "window") -> dict:
    """a - b over the seeds where neither diverged."""
    x = r[r.cell == a].set_index("seed")
    y = r[r.cell == b].set_index("seed")
    seeds = [s for s in SEEDS if s in x.index and s in y.index
             and not x.loc[s, "diverged"] and not y.loc[s, "diverged"]]
    v = np.array([x.loc[s, col] - y.loc[s, col] for s in seeds], float)
    pos, n, p = sign(v)
    return {"a": a, "b": b, "col": col, "n": n, "a_wins": pos,
            "median_diff": float(np.median(v)) if len(v) else float("nan"), "p": p}


def f(v, spec=".4f"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/cifar5p1_mlp_0920")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report on the cells that finished (a look mid-run, not a verdict)")
    a = ap.parse_args()
    src = Path(a.src)
    tasks, fresh, prov = load(src, a.allow_partial)
    r = per_run(tasks)
    main_cells = sorted(r[r.hidden == C.HIDDEN].cell.unique(),
                        key=lambda c: -r[r.cell == c].window.median())
    arm_of = {c: r[r.cell == c].arm.iloc[0] for c in r.cell.unique()}

    gap = {c: fresh[fresh.cell == c].fresh_gap for c in sorted(fresh.cell.unique())}
    lines = ["# cifar5p1_mlp_0920 — 5+1 CIFAR × MLP, 16 arms",
             "",
             f"seeds 0–9, std, lr 1e−4, {C.N_TASKS} tasks × {C.STEPS_PER_TASK} updates. "
             f"Window = mean online accuracy over hard tasks {LATE} (late), "
             f"early = {EARLY}. Collapse = window < {COLLAPSE}.",
             "",
             "## arms (ordered by median late window — **this table is unpaired; "
             "read `paired_tests.csv` for who beats whom**)",
             "",
             "| arm | hidden | late window | early | drop | fresh gap | gap>0 | dead l2 | "
             "zeroout l2 | eff_rank l2 | train(end) | test(end) | div |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in main_cells:
        g = r[r.cell == c]
        gp = gap.get(c, pd.Series(dtype=float))
        lines.append(
            f"| `{arm_of[c]}` | {int(g.hidden.iloc[0])} | {f(g.window.median())}"
            f"{'*' if g.window.median() < COLLAPSE else ''} | {f(g.early.median())} "
            f"| {f(g['drop'].median(), '+.4f')} | {f(float(gp.median()) if len(gp) else float('nan'), '+.4f')} "
            f"| {int((gp > 0).sum())}/{len(gp)} | {f(g.dead_l2.median(), '.3f')} "
            f"| {f(g.zeroout_l2.median(), '.3f')} | {f(g.eff_rank_l2.median(), '.1f')} "
            f"| {f(g.train_end.median(), '.3f')} | {f(g.test_end.median(), '.3f')} "
            f"| {int(g.diverged.sum())} |")

    # every arm against the best one, and the registered head-to-heads
    best = main_cells[0]
    tests = [pair(r, c, best) for c in main_cells if c != best]
    named = []
    for x, y in (("SNA", "LR"), ("LK07", "LR"), ("LK07", "LK03"), ("KKA23", "KKA"),
                 ("DF", "RSL"), ("CR", "R"), ("SNA", "R")):
        cx, cy = f"{x}_std_lr0.0001", f"{y}_std_lr0.0001"
        if cx in set(r.cell) and cy in set(r.cell):
            named.append(pair(r, cx, cy))
    for group in (tests, named):
        for t, q in zip(group, holm([t["p"] for t in group])):
            t["p_holm"] = q
            t["verdict"] = ("A_WINS" if q < 0.05 and t["median_diff"] > 0 else
                            "B_WINS" if q < 0.05 and t["median_diff"] < 0 else "TIE")
    rows = [{**t, "family": fam} for fam, group in (("vs_best", tests), ("registered", named))
            for t in group]
    pd.DataFrame(rows).to_csv(src / "paired_tests.csv", index=False)

    lines += ["", f"## registered head-to-heads (paired sign test, Holm within this family)", "",
              "| a | b | median a−b | a wins | n | p | p_holm | verdict |", "|---|---|---|---|---|---|---|---|"]
    for t in named:
        lines.append(f"| `{arm_of[t['a']]}` | `{arm_of[t['b']]}` | {f(t['median_diff'], '+.4f')} "
                     f"| {t['a_wins']} | {t['n']} | {f(t['p'], '.4f')} | {f(t['p_holm'], '.4f')} "
                     f"| **{t['verdict']}** |")

    lines += ["", f"## every arm against the leader `{arm_of[best]}` (Holm over {len(tests)})", "",
              "| arm | median diff | wins | n | p_holm | verdict |", "|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| `{arm_of[t['a']]}` | {f(t['median_diff'], '+.4f')} | {t['a_wins']} "
                     f"| {t['n']} | {f(t['p_holm'], '.4f')} | {t['verdict']} |")

    ctrl = [c for c in r.cell.unique() if r[r.cell == c].hidden.iloc[0] != C.HIDDEN]
    if ctrl:
        lines += ["", "## equal-parameter control (hidden 94 vs 100; spec §11.1)", "",
                  "| arm | window h100 | window h94 | median paired diff | p |", "|---|---|---|---|---|"]
        for c in sorted(ctrl):
            base = c.replace("_h94", "")
            t = pair(r, base, c)
            lines.append(f"| `{arm_of[c]}` | {f(r[r.cell == base].window.median())} "
                         f"| {f(r[r.cell == c].window.median())} | {f(t['median_diff'], '+.4f')} "
                         f"| {f(t['p'], '.4f')} |")

    # registered readings, per arm
    verdict = {"window_median": {arm_of[c]: float(r[r.cell == c].window.median()) for c in main_cells},
               "order": [arm_of[c] for c in main_cells],
               "Q1_drop_positive": {arm_of[c]: bool(r[r.cell == c]["drop"].median() > 0) for c in main_cells},
               "Q2_gap_positive": {arm_of[c]: bool(gap[c].median() > 0) for c in main_cells if c in gap},
               "Q3_collapsed": {arm_of[c]: bool(r[r.cell == c].window.median() < COLLAPSE) for c in main_cells},
               "P1_R_last": arm_of[main_cells[-1]] == "R",
               "P2_SNA_beats_LR": next((t["verdict"] for t in named
                                        if t["a"].startswith("SNA_") and t["b"].startswith("LR_")), None),
               "P7_DF_first": arm_of[main_cells[0]] == "DF",
               "P8_CR_bottom4": arm_of[main_cells[0]] != "CR" and "CR" in [arm_of[c] for c in main_cells[-4:]],
               "P9_LK07_beats_both": all(
                   next((t["verdict"] for t in named if t["a"].startswith("LK07_")
                         and t["b"].startswith(b + "_")), None) == "A_WINS" for b in ("LR", "LK03")),
               "arms": len(main_cells), "provenance": prov}
    (src / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    (src / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {src}/summary.md, paired_tests.csv, verdict.json")


if __name__ == "__main__":
    main()
