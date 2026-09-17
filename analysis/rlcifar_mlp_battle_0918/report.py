#!/usr/bin/env python3
"""rlcifar_mlp_battle_0918 -- tables, the registered calls (spec §5) and the prediction score (§6).

    python3 analysis/rlcifar_mlp_battle_0918/report.py                 # after all 13 arms are complete
    python3 analysis/rlcifar_mlp_battle_0918/report.py --src DIR --allow-partial

Window = tasks 31-50 of online_acc, per run (arm, cond, seed).  Early = 1-10, late = 41-50,
drop = early - late.  Collapse = window < 0.5 (a run); an arm collapses when the median of its
10 seeds does.  Pairs: seed-paired exact two-sided sign test (zeros dropped); a win = at most one
seed against and p < .05.  Diverged runs are dropped from pairs (counted); collapsed runs are kept.
Seat = 2 alpha_med zbar; the valley of Snake's gate is at -pi/2 and the slope-2 peaks at -3pi/2, +pi/2.
"""
from __future__ import annotations

import argparse
import json
import math
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "rlcifar_mlp_battle_0918"
ARMS = ("SNA", "KKA", "KKA23", "KKT1", "R", "LK001", "LR", "LK03", "SL", "RSL", "ELU", "SILU", "GELU")
SNAKE = ("SNA", "KKA", "KKA23", "KKT1")
CONDS = ("raw", "std")
SEEDS = list(range(10))
WIN, EARLY, LATE = (31, 50), (1, 10), (41, 50)
N_TASKS = 50
PI = math.pi
PREDICTIONS = {                       # spec §6, Claude (2026-09-18 02:20, before implementation)
    "P1":  ("raw: A == SNA_TOP", 0.80),
    "P2":  ("raw: R collapses", 0.95),
    "P3":  ("raw: LK001 collapses", 0.75),
    "P4":  ("raw: LR collapses", 0.65),
    "P5":  ("raw: LK03 collapses", 0.50),
    "P6":  ("raw: SL collapses", 0.60),
    "P7":  ("raw: RSL collapses", 0.50),
    "P8":  ("raw: ELU collapses", 0.70),
    "P9":  ("raw: SILU collapses", 0.70),
    "P10": ("raw: GELU collapses", 0.70),
    "P11": ("std: A == SNA_TOP", 0.45),
    "P12": ("std: R collapses", 0.60),
    "P13": ("SNA_WINS_BOTH", 0.40),
    "P14": ("K1 == PERIOD_FREE in both conditions", 0.60),
    "P15": ("K2 == SCALE_FREE in both conditions", 0.60),
    "P16": ("K3 == TAIL_FREE in both conditions", 0.50),
    "P17": ("raw: median window order SNA ~ KKA ~ KKT1 > KKA23 > SL,RSL > LK03 > LR > LK001 > SILU,GELU > ELU > R "
            "(checked as: the four snake arms are the top four and R is last)", 0.40),
    "P18": ("every leaky arm (LK001, LR, LK03) is STD_HELPS", 0.50),
    "P19": ("SNA is TIE between raw and std", 0.45),
}


def sign(diff) -> tuple[int, int, float]:
    """Exact two-sided sign test, zeros dropped (the host verdicts' rule)."""
    d = np.asarray(diff, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return pos, n, p


def load(src: Path, allow_partial: bool):
    parts, prov, missing = [], {}, []
    for arm in ARMS:
        d = src / arm
        if not (d / "provenance.json").exists():
            missing.append(arm)
            if not (d / "per_task.csv").exists():
                continue
        else:
            prov[arm] = json.loads((d / "provenance.json").read_text())
        parts.append(pd.read_csv(d / "per_task.csv", float_precision="round_trip"))
    if missing and not allow_partial:
        raise SystemExit(f"incomplete box: {missing} have no provenance.json "
                         f"(pass --allow-partial --src DIR to read it anyway)")
    if not parts:
        raise SystemExit(f"no per_task.csv under {src}")
    return pd.concat(parts, ignore_index=True), prov, missing


def runs(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, cond, seed): window, early, late, drop, memo_min, tasks done, diverged."""
    out = []
    for (arm, cond, seed), g in d.groupby(["arm", "cond", "seed"]):
        ok = g[g.task.notna() & g.online_acc.notna()]
        win = ok[(ok.task >= WIN[0]) & (ok.task <= WIN[1])].online_acc
        ear = ok[(ok.task >= EARLY[0]) & (ok.task <= EARLY[1])].online_acc
        lat = ok[(ok.task >= LATE[0]) & (ok.task <= LATE[1])].online_acc
        div = bool(g.online_acc.isna().any())
        out.append({"arm": arm, "cond": cond, "seed": seed, "tasks": int(ok.task.max() or 0),
                    "diverged": div, "window": float(win.mean()) if len(win) else float("nan"),
                    "early": float(ear.mean()) if len(ear) else float("nan"),
                    "late": float(lat.mean()) if len(lat) else float("nan"),
                    "memo_min": float(ok.memo_acc.min()) if len(ok) else float("nan")})
    r = pd.DataFrame(out)
    r["drop"] = r.early - r.late
    r["collapsed"] = r.window < 0.5
    return r


def pair(r: pd.DataFrame, a: str, b: str, cond: str, col: str = "window") -> dict:
    """a - b over the seeds where neither diverged."""
    x = r[(r.arm == a) & (r.cond == cond)].set_index("seed")
    y = r[(r.arm == b) & (r.cond == cond)].set_index("seed")
    seeds = [s for s in SEEDS if s in x.index and s in y.index
             and not x.loc[s, "diverged"] and not y.loc[s, "diverged"]]
    v = np.array([x.loc[s, col] - y.loc[s, col] for s in seeds], float)
    pos, n, p = sign(v)
    win = (n - pos) <= 1 and p < 0.05 and pos > n - pos
    lose = pos <= 1 and p < 0.05 and pos < n - pos
    return {"a": a, "b": b, "cond": cond, "n": n, "wins": pos, "median": float(np.median(v)) if len(v) else float("nan"),
            "p": p, "verdict": "A_WINS" if win else ("B_WINS" if lose else "TIE"),
            "dropped_seeds": [s for s in SEEDS if s not in seeds]}


def label_A(r: pd.DataFrame, cond: str) -> dict:
    ps = {b: pair(r, "SNA", b, cond) for b in ARMS if b != "SNA"}
    beaten = [b for b, q in ps.items() if q["verdict"] == "B_WINS"]
    allwin = all(q["verdict"] == "A_WINS" for q in ps.values())
    return {"label": "SNA_BEATEN" if beaten else ("SNA_TOP" if allwin else "SNA_TIED_TOP"),
            "beaten_by": beaten, "pairs": ps}


def k_label(r: pd.DataFrame, a: str, b: str, cond: str, names: tuple[str, str, str]) -> dict:
    q = pair(r, a, b, cond)
    return {"label": names[0] if q["verdict"] == "A_WINS" else
            (names[1] if q["verdict"] == "B_WINS" else names[2]), **q}


def seat_course(d: pd.DataFrame, arm: str, cond: str, tasks=(1, 5, 10, 20, 30, 40, 50)) -> dict:
    g = d[(d.arm == arm) & (d.cond == cond)]
    out = {}
    for t in tasks:
        gt = g[g.task == t]
        if not len(gt):
            continue
        row = {}
        for l in (1, 2):
            if f"alpha_med_l{l}" in gt:
                row[f"seat_l{l}"] = float((2 * gt[f"alpha_med_l{l}"] * gt[f"zbar_l{l}"]).median())
                row[f"two_alpha_W_l{l}"] = float(gt[f"two_alpha_W_med_l{l}"].median())
            row[f"mob_l{l}"] = float(gt[f"mob_l{l}"].median())
            row[f"W_l{l}"] = float(gt[f"zsd_l{l}"].median())
            row[f"dead_l{l}"] = float(gt[f"dead_frac_l{l}"].median())
            row[f"eff_rank_l{l}"] = float(gt[f"eff_rank_l{l}"].median())
        row["online"] = float(gt.online_acc.median())
        row["memo"] = float(gt.memo_acc.median())
        out[t] = row
    return out


def band_mass(src: Path, arm: str, cond: str, tasks=(1, 10, 25, 50)) -> dict:
    """Share of theta = 2 alpha_i z outside the kunekune band [-3pi/2, pi/2], per layer and task
    (the adaptive arms' phase histogram; per-unit alpha, not the median)."""
    out = {}
    for seed in SEEDS:
        f = src / arm / "hist" / f"{arm}_{cond}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if "th1" not in d.files:
            return {}
        e = d["th_edges"]
        c = (e[:-1] + e[1:]) / 2
        for t in tasks:
            if t > d["th1"].shape[0]:
                continue
            for l in (1, 2):
                h = d[f"th{l}"][t - 1].astype(float)
                tot = h.sum() + d[f"thoob{l}"][t - 1]
                k = out.setdefault((t, l), [])
                k.append({"above": float(h[c > PI / 2].sum() / tot),
                          "below": float(h[c < -1.5 * PI].sum() / tot),
                          "median_theta": float(c[np.searchsorted(np.cumsum(h) / h.sum(), 0.5)])})
    return {f"t{t}_l{l}": {k: float(np.median([x[k] for x in v])) for k in ("above", "below", "median_theta")}
            for (t, l), v in sorted(out.items())}


def score(labels: dict) -> dict:
    L = labels
    got = {
        "P1": L["A"]["raw"]["label"] == "SNA_TOP",
        "P11": L["A"]["std"]["label"] == "SNA_TOP",
        "P13": L["main"] == "SNA_WINS_BOTH",
        "P14": all(L["K1"][c]["label"] == "PERIOD_FREE" for c in CONDS),
        "P15": all(L["K2"][c]["label"] == "SCALE_FREE" for c in CONDS),
        "P16": all(L["K3"][c]["label"] == "TAIL_FREE" for c in CONDS),
        "P17": (set(sorted(L["order"]["raw"], key=lambda a: -L["window_median"]["raw"][a])[:4]) == set(SNAKE)
                and L["order"]["raw"][-1] == "R"),
        "P18": all(L["std_vs_raw"][a]["label"] == "STD_HELPS" for a in ("LK001", "LR", "LK03")),
        "P19": L["std_vs_raw"]["SNA"]["label"] == "TIE",
    }
    for key, arm in (("P2", "R"), ("P3", "LK001"), ("P4", "LR"), ("P5", "LK03"), ("P6", "SL"),
                     ("P7", "RSL"), ("P8", "ELU"), ("P9", "SILU"), ("P10", "GELU")):
        got[key] = bool(L["collapsed"]["raw"].get(arm, False))
    got["P12"] = bool(L["collapsed"]["std"].get("R", False))
    out, brier = {}, []
    for k, (claim, p) in PREDICTIONS.items():
        hit = bool(got.get(k, False))
        out[k] = {"claim": claim, "p": p, "hit": hit}
        brier.append((p - (1 if hit else 0)) ** 2)
    out["_summary"] = {"n": len(brier), "hits": sum(v["hit"] for k, v in out.items() if k != "_summary"),
                       "brier": float(np.mean(brier))}
    return out


def f(v, spec=".4f"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    src = Path(a.src) if a.src else OUT
    if a.allow_partial and not a.src:
        raise SystemExit("--allow-partial only with --src")
    dst = src                                   # --src reads and writes one directory
    dst.mkdir(parents=True, exist_ok=True)
    d, prov, missing = load(src, a.allow_partial)
    r = runs(d)

    L = {"A": {}, "K1": {}, "K2": {}, "K3": {}, "collapsed": {}, "window_median": {},
         "order": {}, "std_vs_raw": {}, "diverged": {}}
    for cond in CONDS:
        L["A"][cond] = label_A(r, cond)
        L["K1"][cond] = k_label(r, "KKA", "SNA", cond, ("PERIOD_HURTS", "PERIOD_HELPS", "PERIOD_FREE"))
        L["K2"][cond] = k_label(r, "KKA23", "KKA", cond, ("SCALE_MATTERS_UP", "SCALE_MATTERS_DOWN", "SCALE_FREE"))
        L["K3"][cond] = k_label(r, "KKT1", "KKA", cond, ("TAIL1_BETTER", "TAIL2_BETTER", "TAIL_FREE"))
        w = {arm: float(r[(r.arm == arm) & (r.cond == cond)].window.median()) for arm in ARMS}
        L["window_median"][cond] = w
        L["order"][cond] = sorted(ARMS, key=lambda x: -w[x] if not math.isnan(w[x]) else 1e9)
        L["collapsed"][cond] = {arm: bool(w[arm] < 0.5) if not math.isnan(w[arm]) else None for arm in ARMS}
        L["diverged"][cond] = {arm: int(r[(r.arm == arm) & (r.cond == cond)].diverged.sum()) for arm in ARMS}
    L["main"] = "SNA_WINS_BOTH" if all(L["A"][c]["label"] == "SNA_TOP" for c in CONDS) else "NOT_BOTH"
    for arm in ARMS:
        x = r[(r.arm == arm) & (r.cond == "std")].set_index("seed")
        y = r[(r.arm == arm) & (r.cond == "raw")].set_index("seed")
        seeds = [s for s in SEEDS if s in x.index and s in y.index
                 and not x.loc[s, "diverged"] and not y.loc[s, "diverged"]]
        v = np.array([x.loc[s, "window"] - y.loc[s, "window"] for s in seeds], float)
        pos, n, p = sign(v)
        lab = "STD_HELPS" if (n - pos) <= 1 and p < 0.05 and pos > n - pos else (
            "STD_HURTS" if pos <= 1 and p < 0.05 and pos < n - pos else "TIE")
        L["std_vs_raw"][arm] = {"label": lab, "median": float(np.median(v)) if len(v) else float("nan"),
                                "n": n, "wins": pos, "p": p}
    S = score(L)

    lines = ["# rlcifar_mlp_battle_0918 -- RL-CIFAR x MLP activation battle", ""]
    if missing:
        lines += [f"**incomplete**: no provenance for {missing}", ""]
    lines += [f"box: {json.dumps({k: v for k, v in list(prov.values())[0].items() if k in ('n_tasks', 'epochs_per_task', 'steps_per_task', 'dims', 'lr', 'optimizer', 'engine', 'device')}) if prov else 'n/a'}", ""]
    lines += ["## windows (median over seeds; * = collapsed, d = diverged runs)", "",
              "| arm | raw | std | std-raw | raw drop | std drop | raw memo min | collapse raw/std |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rr = {c: r[(r.arm == arm) & (r.cond == c)] for c in CONDS}
        lines.append(f"| `{arm}` | {f(L['window_median']['raw'][arm])}"
                     f"{'*' if L['collapsed']['raw'][arm] else ''} | {f(L['window_median']['std'][arm])}"
                     f"{'*' if L['collapsed']['std'][arm] else ''} | {f(L['std_vs_raw'][arm]['median'], '+.4f')}"
                     f" ({L['std_vs_raw'][arm]['label']}) | {f(rr['raw']['drop'].median(), '+.4f')}"
                     f" | {f(rr['std']['drop'].median(), '+.4f')} | {f(rr['raw'].memo_min.median())}"
                     f" | {int(rr['raw'].collapsed.sum())}/{int(rr['std'].collapsed.sum())}"
                     f"{' d' + str(int(rr['raw'].diverged.sum()) + int(rr['std'].diverged.sum())) if (rr['raw'].diverged.sum() + rr['std'].diverged.sum()) else ''} |")
    lines += ["", "## registered calls", ""]
    for cond in CONDS:
        A = L["A"][cond]
        lines.append(f"- **A ({cond})**: `{A['label']}`" + (f" — beaten by {A['beaten_by']}" if A["beaten_by"] else ""))
        for b, q in A["pairs"].items():
            lines.append(f"    - SNA - {b}: median {f(q['median'], '+.4f')}, {q['wins']}/{q['n']} seeds, "
                         f"p={f(q['p'], '.4f')} → {q['verdict']}" +
                         (f" (dropped {q['dropped_seeds']})" if q["dropped_seeds"] else ""))
    lines.append(f"- **main**: `{L['main']}`")
    for k, nm in (("K1", "KKA - SNA"), ("K2", "KKA23 - KKA"), ("K3", "KKT1 - KKA")):
        for cond in CONDS:
            q = L[k][cond]
            lines.append(f"- **{k} ({cond})**: `{q['label']}` — {nm} median {f(q['median'], '+.4f')}, "
                         f"{q['wins']}/{q['n']} seeds, p={f(q['p'], '.4f')}")
    lines += ["", "## course of the snake arms (median over seeds)", ""]
    course = {}
    for arm in SNAKE:
        for cond in CONDS:
            course[f"{arm}_{cond}"] = seat_course(d, arm, cond)
    lines += ["| arm/cond | t | online | seat l1 | seat l2 | mob l1 | mob l2 | W l1 | W l2 | 2aW l1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key, cc in course.items():
        for t, row in cc.items():
            if t in (1, 10, 50):
                lines.append(f"| {key} | {t} | {f(row['online'])} | {f(row.get('seat_l1'), '+.2f')} | "
                             f"{f(row.get('seat_l2'), '+.2f')} | {f(row['mob_l1'], '.2f')} | {f(row['mob_l2'], '.2f')} | "
                             f"{f(row['W_l1'], '.1f')} | {f(row['W_l2'], '.1f')} | {f(row.get('two_alpha_W_l1'), '.2f')} |")
    bands = {f"{arm}_{cond}": band_mass(src, arm, cond) for arm in SNAKE for cond in CONDS}
    lines += ["", "## phase: share of 2 alpha_i z outside the kunekune band (median over seeds)", "",
              "| arm/cond | task/layer | above +pi/2 | below -3pi/2 | median theta |", "|---|---|---|---|---|"]
    for key, bm in bands.items():
        for tl, v in bm.items():
            lines.append(f"| {key} | {tl} | {f(v['above'], '.3f')} | {f(v['below'], '.3f')} | {f(v['median_theta'], '+.2f')} |")
    lines += ["", "## prediction score (spec §6, Claude)", "",
              f"hits {S['_summary']['hits']}/{S['_summary']['n']}, Brier {S['_summary']['brier']:.3f}", "",
              "| key | claim | p | hit |", "|---|---|---|---|"]
    for k, v in S.items():
        if k != "_summary":
            lines.append(f"| {k} | {v['claim']} | {v['p']:.2f} | {'yes' if v['hit'] else 'no'} |")
    (dst / "summary.md").write_text("\n".join(lines) + "\n")
    verdict = {"labels": {k: v for k, v in L.items()}, "score": S, "runs": r.to_dict("records"),
               "course": course, "bands": bands, "missing": missing,
               "provenance": {k: {kk: vv for kk, vv in v.items() if kk not in ("subset_sha256",)} for k, v in prov.items()}}
    (dst / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print("\n".join(lines[:60]))
    print(f"\nwrote {dst}/summary.md and verdict.json")


if __name__ == "__main__":
    main()
