#!/usr/bin/env python3
"""baselines_cifar5p1_1008 -- calibration selection (spec 3.2) and the registered verdict (spec 5-7).

    python analysis/baselines_cifar5p1_1008/verdict.py select [--src results/baselines_cifar5p1_1008/calib]
    python analysis/baselines_cifar5p1_1008/verdict.py judge  [--src results/baselines_cifar5p1_1008]

`select` reads every calib/<cell>/ directory (seeds 100-109), scores each cell by the mean over
the 10 seeds of the late window (hard tasks 21-29), drops cells with a diverged seed, picks the
best cell per method (exact ties -> the registered grid order) and writes calib_table.csv and
selected.json.  `judge` reads main/<method>_s0-9 and _s10-19 and the committed comparator records
of cifar5p1_mlp_0920, and writes verdict.json, paired.csv, per_seed.csv, secondary.csv and
predictions.csv.  Paired differences over seeds 0-19 only (never differences of medians).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import baselines_cifar5p1_1008 as E          # noqa: E402  (GRID, tags; imports torch)

RUN = E.EXPERIMENT
N_TASKS = 30
HARD = list(range(1, N_TASKS + 1, 2))
EARLY, LATE = HARD[:5], HARD[10:]                     # [1,3,5,7,9], [21,23,25,27,29]
T29 = 29
ALPHA = 0.05
BAND = 0.005
SEEDS = list(range(20))
METHODS = ("snp", "cbp", "redo")
NAME = {"snp": "SNP", "cbp": "CBP", "redo": "REDO", "none": "R"}
HOST = REPO / "results" / "cifar5p1_mlp_0920"
HOST_R = ("R_std_lr0.0001", "R_s10-19")
COMPARATORS = {"SNA": ("SNA_std_lr0.0001", "SNA_s10-19"),
               "KKT1": ("KKT1_std_lr0.0001", "KKT1_s10-19"),
               "L2I": ("R_std_lr0.0001_l2init1e-3", "R_s10-19l2init:1e3")}
DIAG = ("dead_frac_l2", "mob_l2", "eff_rank_l2", "zbar_l2", "zsd_l2", "w_norm_l1", "w_norm_l2", "w_norm_l3")
P2A_DEAD = 0.02                                       # spec 7.1
P4_REL = 0.3


# --------------------------------------------------------------------------
# statistics (spec 5.2)
# --------------------------------------------------------------------------

def sign_test(d) -> tuple[int, int, float]:
    """Exact two-sided sign test, zeros dropped (the host's rule).  Returns (positives, n, p).
    All differences zero (n = 0): p = 1 (no evidence of a difference; the host returns nan, which
    would poison Holm)."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else 1.0
    return pos, n, p


def order_interval(d) -> tuple[float, float]:
    """Distribution-free interval for the median: n = 20 -> [v(6), v(15)], n = 10 -> [v(2), v(9)]."""
    v = np.sort(np.asarray(d, float))
    if len(v) == 20:
        return float(v[5]), float(v[14])
    if len(v) == 10:
        return float(v[1]), float(v[8])
    raise ValueError(f"no registered interval for n = {len(v)}")


def label4(d) -> dict:
    """A_WINS / B_WINS / EQUIVALENT_WITHIN_0.005 / UNRESOLVED on paired differences (A - B)."""
    d = np.asarray(d, float)
    pos, n, p = sign_test(d)
    med = float(np.median(d))
    lo, hi = order_interval(d)
    if p < ALPHA and med > 0:
        lab = "A_WINS"
    elif p < ALPHA and med < 0:
        lab = "B_WINS"
    elif p >= ALPHA and -BAND <= lo and hi <= BAND:
        lab = "EQUIVALENT_WITHIN_0.005"
    else:
        lab = "UNRESOLVED"
    return {"n": len(d), "median": med, "lo": lo, "hi": hi, "a_pos": pos, "n_nonzero": n, "p": p,
            "label": lab}


def label_gap(d) -> dict:
    """GAP_REDUCED / GAP_INCREASED / GAP_NOT_SHOWN on paired gap differences (x - R)."""
    d = np.asarray(d, float)
    pos, n, p = sign_test(d)
    med = float(np.median(d))
    lo, hi = order_interval(d)
    lab = ("GAP_REDUCED" if p < ALPHA and med < 0 else
           "GAP_INCREASED" if p < ALPHA and med > 0 else "GAP_NOT_SHOWN")
    return {"n": len(d), "median": med, "lo": lo, "hi": hi, "a_pos": pos, "n_nonzero": n, "p": p,
            "label": lab}


def paired(a: pd.Series, b: pd.Series, seeds=SEEDS) -> np.ndarray:
    """Seed-by-seed differences a - b (both indexed by seed)."""
    return np.array([a.loc[s] - b.loc[s] for s in seeds], float)


def prop_p2a(dead_median: float) -> bool:
    return dead_median <= P2A_DEAD


def prop_p4(relpos_median: float) -> bool:
    return abs(relpos_median) < P4_REL


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, ps[i] * (len(ps) - rank)))
        out[i] = running
    return out


def relabel(entry: dict, p: float, gap: bool = False) -> str:
    med, lo, hi = entry["median"], entry["lo"], entry["hi"]
    if gap:
        return "GAP_REDUCED" if p < ALPHA and med < 0 else "GAP_INCREASED" if p < ALPHA and med > 0 \
            else "GAP_NOT_SHOWN"
    if p < ALPHA and med > 0:
        return "A_WINS"
    if p < ALPHA and med < 0:
        return "B_WINS"
    if p >= ALPHA and -BAND <= lo and hi <= BAND:
        return "EQUIVALENT_WITHIN_0.005"
    return "UNRESOLVED"


# --------------------------------------------------------------------------
# per-seed windows
# --------------------------------------------------------------------------

def windows(per_task: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per seed: late / early / t29 online, late-window means of the diagnostics."""
    out = []
    for seed, g in per_task.groupby("seed"):
        g = g.set_index("task")
        ok = all(t in g.index and np.isfinite(g.loc[t, "online_acc"]) for t in HARD)
        row = {"seed": int(seed), "complete": ok and len(g) == N_TASKS,
               "late": float(g.loc[LATE, "online_acc"].mean()) if ok else float("nan"),
               "early": float(g.loc[EARLY, "online_acc"].mean()) if ok else float("nan"),
               "t29": float(g.loc[T29, "online_acc"]) if ok else float("nan")}
        for k in DIAG:
            if k in g.columns and ok:
                row[f"{k}_late"] = float(g.loc[LATE, k].mean())
                row[f"{k}_t29"] = float(g.loc[T29, k])
        out.append(row)
    w = pd.DataFrame(out).set_index("seed").sort_index()
    if extra is not None and len(extra):
        for seed, g in extra.groupby("seed"):
            g = g.set_index("task")
            if not all(t in g.index for t in LATE):
                continue
            for k in ("relpos_l1", "relpos_l2", "zbar_over_zsd_l2", "dead_post_l2"):
                w.loc[int(seed), f"{k}_late"] = float(g.loc[LATE, k].mean())
            w.loc[int(seed), "relpos_l2_t29"] = float(g.loc[T29, "relpos_l2"])
            w.loc[int(seed), "resets_l1"] = float(g["n_reset_l1"].sum())
            w.loc[int(seed), "resets_l2"] = float(g["n_reset_l2"].sum())
            w.loc[int(seed), "resets_l2_late"] = float(g.loc[LATE, "n_reset_l2"].sum())
    return w


def read_run(d: Path) -> dict:
    out = {"per_task": pd.read_csv(d / "per_task.csv"), "dir": d}
    for name in ("fresh_control", "extra_task", "fresh_self", "redo_checks"):
        f = d / f"{name}.csv"
        out[name] = pd.read_csv(f) if f.exists() and f.stat().st_size else None
    p = d / "provenance.json"
    out["provenance"] = json.loads(p.read_text()) if p.exists() else None
    return out


# --------------------------------------------------------------------------
# calibration selection (spec 3.2)
# --------------------------------------------------------------------------

def select_from_table(table: list[dict]) -> dict:
    """table rows: {method, hyper: dict, seeds: list, late: list (per seed), diverged: bool}.
    Score = mean late window over seeds 100-109; diverged cells dropped; exact ties -> grid order."""
    sel = {}
    for method in METHODS:
        grid = E.GRID[method]
        rows = [r for r in table if r["method"] == method]
        for r in rows:
            if r["hyper"] not in grid:
                raise ValueError(f"unregistered cell {r['hyper']} for {method}")
            if sorted(r["seeds"]) != E.CALIB_SEEDS:
                raise ValueError(f"cell {r['hyper']} has seeds {r['seeds']}, not 100-109")
        have = [r["hyper"] for r in rows]
        missing = [h for h in grid if h not in have]
        if missing or len(have) != len(grid):
            raise ValueError(f"{method}: grid incomplete (missing {missing})")
        best, best_score = None, -math.inf
        for h in grid:                                  # grid order = tie-break order
            r = next(q for q in rows if q["hyper"] == h)
            if r["diverged"]:
                continue
            score = float(np.mean(np.asarray(r["late"], float)))
            if score > best_score:
                best, best_score = h, score
        if best is None:
            raise ValueError(f"{method}: every cell diverged")
        sel[method] = {"hyper": best, "score": best_score}
    return sel


def edge_of(method: str, hyper: dict) -> list[str]:
    edges = []
    for k in hyper:
        vals = sorted({float(h[k]) for h in E.GRID[method]})
        v = float(hyper[k])
        if len(vals) > 1 and v in (vals[0], vals[-1]):
            edges.append(f"{k}={hyper[k]} ({'low' if v == vals[0] else 'high'} end)")
    return edges


def cmd_select(src: Path) -> dict:
    table, csv_rows = [], []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        if not (d / "provenance.json").exists():
            raise SystemExit(f"unfinished calibration cell {d.name}")
        r = read_run(d)
        prov = r["provenance"]
        cfg = prov["method_cfg"]
        w = windows(r["per_task"], r["extra_task"])
        diverged = bool(prov["divergences"]) or not bool(w["complete"].all()) or len(w) != 10
        gaps = r["fresh_control"]["fresh_gap"] if r["fresh_control"] is not None else pd.Series(dtype=float)
        row = {"method": cfg["method"], "hyper": E.hyper(cfg), "seeds": prov["seeds"],
               "late": w["late"].tolist(), "diverged": diverged}
        if cfg["method"] != "none":
            table.append(row)
        csv_rows.append({"method": cfg["method"], "config": E.cfg_tag(cfg), **E.hyper(cfg),
                         "n_seeds": len(w), "diverged": diverged,
                         "score_mean_late": float(w["late"].mean()), "median_late": float(w["late"].median()),
                         "sd_late": float(w["late"].std()), "mean_early": float(w["early"].mean()),
                         "gap_median": float(gaps.median()) if len(gaps) else float("nan"),
                         "gap_pos": int((gaps > 0).sum()),
                         "dead_l2_late": float(w["dead_frac_l2_late"].mean()),
                         "mob_l2_late": float(w["mob_l2_late"].mean()),
                         "eff_rank_l2_late": float(w["eff_rank_l2_late"].mean()),
                         "relpos_l2_late": float(w["relpos_l2_late"].mean()),
                         "resets_l1": float(w["resets_l1"].sum()), "resets_l2": float(w["resets_l2"].sum())})
    sel = select_from_table(table)
    out = {"rule": "spec 3.2: max mean late window over seeds 100-109; diverged cells dropped; "
                   "exact ties -> registered grid order", "selected": {}, "scores": {}, "edges": {}}
    for m in METHODS:
        h = sel[m]["hyper"]
        out["selected"][m] = E.make_cfg(m, **{k: h.get(k) for k in ("eps", "sigma", "rho", "tau", "period")})
        out["scores"][m] = sel[m]["score"]
        out["edges"][m] = edge_of(m, h)
    for r in csv_rows:
        r["selected"] = r["method"] in sel and E.hyper(out["selected"][r["method"]]) == \
            {k: r[k] for k in ("eps", "sigma", "rho", "tau", "period") if k in r}
    pd.DataFrame(csv_rows).to_csv(src / "calib_table.csv", index=False)
    (src / "selected.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# the registered verdict (spec 5-7)
# --------------------------------------------------------------------------

def committed(cells: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.concat([pd.read_csv(HOST / c / "per_task.csv") for c in cells], ignore_index=True)
    f = pd.concat([pd.read_csv(HOST / c / "fresh_control.csv") for c in cells], ignore_index=True)
    return t, f


def fresh_column(path: Path) -> list[tuple[str, str]]:
    """(seed, fresh_online_acc) exactly as printed: the fresh network, shared by every arm."""
    rows = [l.split(",") for l in Path(path).read_text().splitlines()]
    i, j = rows[0].index("seed"), rows[0].index("fresh_online_acc")
    return [(r[i], r[j]) for r in rows[1:]]


def text_cols(path: Path) -> list[list[str]]:
    return [line.split(",") for line in path.read_text().splitlines()]


def same_but_eff_rank(a: Path, b: Path) -> tuple[bool, float]:
    """All columns but eff_rank_* byte-identical; returns (ok, max relative eff_rank deviation)."""
    A, Bt = text_cols(a), text_cols(b)
    if len(A) != len(Bt) or A[0] != Bt[0]:
        return False, float("nan")
    er = [i for i, c in enumerate(A[0]) if c.startswith("eff_rank")]
    ok, dev = True, 0.0
    for ra, rb in zip(A[1:], Bt[1:]):
        if len(ra) != len(rb):
            return False, float("nan")
        for i, (x, y) in enumerate(zip(ra, rb)):
            if i in er:
                if x != y:
                    fx, fy = float(x), float(y)
                    dev = max(dev, abs(fx - fy) / max(abs(fy), 1e-300))
            elif x != y:
                ok = False
    return ok, dev


def cmd_judge(src: Path) -> dict:
    main = src / "main"
    groups = ("s0-9", "s10-19")
    runs, applic, flags = {}, {}, {}
    missing = []
    for m in ("none",) + METHODS:
        for g in groups:
            d = main / f"{m}_{g}"
            if not (d / "provenance.json").exists():
                missing.append(d.name)
            else:
                runs[(m, g)] = read_run(d)
    if missing:
        return {"label": "INCOMPLETE", "missing": missing}

    # (A) completeness and identity
    A = {}
    win, gap, extra_w = {}, {}, {}
    for m in ("none",) + METHODS:
        pt = pd.concat([runs[(m, g)]["per_task"] for g in groups], ignore_index=True)
        ex = pd.concat([runs[(m, g)]["extra_task"] for g in groups], ignore_index=True)
        fr = pd.concat([runs[(m, g)]["fresh_control"] for g in groups], ignore_index=True)
        w = windows(pt, ex)
        div = sum(len(runs[(m, g)]["provenance"]["divergences"]) for g in groups)
        A[f"{m}_complete"] = (sorted(w.index) == SEEDS and bool(w["complete"].all())
                              and len(fr) == 20 and bool(np.isfinite(fr["fresh_gap"]).all()))
        A[f"{m}_divergences"] = div
        win[m] = w
        gap[m] = fr.set_index("seed")["fresh_gap"].sort_index()
    for g, cell in zip(groups, HOST_R):
        ok, dev = same_but_eff_rank(main / f"none_{g}" / "per_task.csv", HOST / cell / "per_task.csv")
        A[f"none_{g}_identical_but_eff_rank"] = ok
        A[f"none_{g}_eff_rank_max_rel_dev"] = dev
        A[f"none_{g}_fresh_identical"] = (main / f"none_{g}" / "fresh_control.csv").read_bytes() == \
            (HOST / cell / "fresh_control.csv").read_bytes()
        for m in METHODS:                                 # the fresh network is the same in every arm
            A[f"{m}_{g}_fresh_equals_none"] = fresh_column(main / f"{m}_{g}" / "fresh_control.csv") == \
                fresh_column(main / f"none_{g}" / "fresh_control.csv")
    comp = {}
    for k, cells in COMPARATORS.items():
        t, f = committed(cells)
        w = windows(t)
        A[f"{k}_records_20_seeds"] = sorted(w.index) == SEEDS and bool(w["complete"].all())
        comp[k] = w
    identity_keys = [k for k, v in A.items() if isinstance(v, bool)]
    check_failed = [k for k in identity_keys if not A[k] and not k.endswith("_complete")]
    incomplete = [m for m in ("none",) + METHODS if not A[f"{m}_complete"]]
    applic["A"] = {"pass": not check_failed and "none" not in incomplete,
                   "failed": check_failed, "incomplete_methods": incomplete, "detail": A}

    # (B) loss reproduced
    pos, n, p = sign_test(gap["none"].values)
    applic["B"] = {"pass": bool(p < ALPHA and float(np.median(gap["none"])) > 0), "gap_median":
                   float(np.median(gap["none"])), "gap_pos": pos, "n": n, "p": p}

    # (C) the method wrote
    C_ = {}
    for m in ("cbp", "redo"):
        w = win[m]
        C_[m] = {"resets_l1": float(w["resets_l1"].sum()), "resets_l2": float(w["resets_l2"].sum())}
    C_["cbp"]["active"] = C_["cbp"]["resets_l1"] > 0 and C_["cbp"]["resets_l2"] > 0
    C_["redo"]["active"] = C_["redo"]["resets_l2"] > 0
    applic["C"] = C_

    # registered judgments (spec 5.3)
    tests, paired_rows = [], []
    for m in METHODS:
        x = NAME[m]
        lab_ok = applic["A"]["pass"] and applic["B"]["pass"] and m not in incomplete \
            and A[f"{m}_divergences"] == 0
        for jid, ref, kind in (("J1", "none", "late"), ("J1g", "none", "gap"), ("J2", "SNA", "late"),
                               ("J3", "KKT1", "late"), ("J4", "L2I", "late")):
            if kind == "gap":
                d = paired(gap[m], gap["none"])
                e = label_gap(d)
            else:
                b = win["none"]["late"] if ref == "none" else comp[ref]["late"]
                d = paired(win[m]["late"], b)
                e = label4(d)
            e.update({"id": f"{jid}-{x}", "method": x, "ref": NAME.get(ref, ref), "kind": kind,
                      "issued": lab_ok})
            if not lab_ok:
                e["label"] = ("CHECK_FAILED" if not applic["A"]["pass"] else
                              "NOT_REPRODUCED" if not applic["B"]["pass"] else "INCOMPLETE")
            tests.append(e)
            for s, v in zip(SEEDS, d):
                paired_rows.append({"id": e["id"], "seed": s, "diff": v})
    ph = holm([t["p"] for t in tests])
    for t, q in zip(tests, ph):
        t["p_holm"] = q
        t["label_holm"] = relabel(t, q, gap=t["kind"] == "gap") if t["issued"] else t["label"]
        t["holm_changes_label"] = t["label_holm"] != t["label"]
    halves = []
    for t in tests:
        for name, idx in (("s0-9", list(range(10))), ("s10-19", list(range(10, 20)))):
            d = np.array([r["diff"] for r in paired_rows if r["id"] == t["id"] and r["seed"] in idx])
            pos, n, p = sign_test(d)
            lo, hi = order_interval(d)
            halves.append({"id": t["id"], "half": name, "median": float(np.median(d)), "lo": lo, "hi": hi,
                           "a_pos": pos, "n_nonzero": n, "p": p})

    # secondary (spec 6)
    sec = {}
    arms = {"R": win["none"], **{NAME[m]: win[m] for m in METHODS}, **comp}
    diag = []
    for name, w in arms.items():
        row = {"arm": name, "late_median": float(w["late"].median()), "late_mean": float(w["late"].mean()),
               "early_median": float(w["early"].median())}
        for k in DIAG:
            for suf in ("late", "t29"):
                c = f"{k}_{suf}"
                if c in w.columns:
                    row[c] = float(w[c].median())
        for c in ("relpos_l2_late", "relpos_l1_late", "zbar_over_zsd_l2_late", "dead_post_l2_late",
                  "resets_l1", "resets_l2", "resets_l2_late"):
            if c in w.columns:
                row[c] = float(w[c].median())
        if name in ("R",) or name in [NAME[m] for m in METHODS]:
            key = "none" if name == "R" else [m for m in METHODS if NAME[m] == name][0]
            row["gap_median"] = float(gap[key].median())
            row["gap_pos"] = int((gap[key] > 0).sum())
        diag.append(row)
    sec["diagnostics"] = diag
    # 6.2 S&P relative position
    d_rel = paired(win["snp"]["relpos_l2_late"], win["none"]["relpos_l2_late"])
    d_rat = paired(win["snp"]["zbar_over_zsd_l2_late"], win["none"]["zbar_over_zsd_l2_late"])
    sec["snp_relpos"] = {"relpos_l2": {**_desc(d_rel)}, "zbar_over_zsd_l2": {**_desc(d_rat)},
                         "R_relpos_l2_late_median": float(win["none"]["relpos_l2_late"].median()),
                         "SNP_relpos_l2_late_median": float(win["snp"]["relpos_l2_late"].median())}
    # 6.3 ReDo
    rd, rr = win["redo"], win["none"]
    d_late = paired(rd["late"], rr["late"])
    d_dead = paired(rd["dead_frac_l2_late"], rr["dead_frac_l2_late"])
    rec = {k: float(np.median(d_late) / np.median(paired(comp[k]["late"], rr["late"]))) for k in ("SNA", "L2I")}
    redo_dead_med = float(rd["dead_frac_l2_late"].median())
    j2 = next(t for t in tests if t["id"] == "J2-REDO")["label"]
    j4 = next(t for t in tests if t["id"] == "J4-REDO")["label"]
    cond_i = redo_dead_med <= P2A_DEAD
    if not cond_i:
        reading = "REDO_DID_NOT_REMOVE_DEATH"
    elif j2 in ("EQUIVALENT_WITHIN_0.005", "A_WINS") and j4 in ("EQUIVALENT_WITHIN_0.005", "A_WINS"):
        reading = "DEATH_REMOVAL_RESTORES"
    elif "B_WINS" in (j2, j4):
        reading = "DEATH_REMOVAL_NOT_ENOUGH"
    else:
        reading = "UNDECIDED"
    sec["redo"] = {"reading": reading, "dead_l2_late_median": redo_dead_med,
                   "dead_post_l2_late_median": float(rd["dead_post_l2_late"].median())
                   if "dead_post_l2_late" in rd.columns else float("nan"),
                   "R_dead_l2_late_median": float(rr["dead_frac_l2_late"].median()),
                   "resets_l1_median": float(rd["resets_l1"].median()),
                   "resets_l2_median": float(rd["resets_l2"].median()),
                   "resets_l2_late_median": float(rd["resets_l2_late"].median()),
                   "d_late_vs_R": _desc(d_late), "d_dead_vs_R": _desc(d_dead),
                   "recovery_fraction": rec,
                   "per_seed_corr_resets_l2_vs_late": float(np.corrcoef(rd["resets_l2"], rd["late"])[0, 1])
                   if rd["resets_l2"].std() > 0 else float("nan")}
    # 6.4 CBP, 6.5 fresh_self
    sec["cbp"] = {"resets_l1_total_median": float(win["cbp"]["resets_l1"].median()),
                  "resets_l2_total_median": float(win["cbp"]["resets_l2"].median())}
    fs = {}
    for m in METHODS:
        f = pd.concat([runs[(m, g)]["fresh_self"] for g in groups], ignore_index=True)
        fs[NAME[m]] = {"gap_self_median": float(f["gap_self"].median()), "gap_self_pos": int((f["gap_self"] > 0).sum()),
                       "fresh_self_median": float(f["fresh_self_online_acc"].median())}
    fs["R_fresh_median"] = float(pd.concat([runs[("none", g)]["fresh_control"] for g in groups])["fresh_online_acc"].median())
    sec["fresh_self"] = fs
    for m in METHODS:
        sec[f"gap_{NAME[m]}"] = {"median": float(gap[m].median()), "pos": int((gap[m] > 0).sum()),
                                 "p": sign_test(gap[m].values)[2]}

    verdict = {"run": RUN, "applicability": applic, "tests": tests, "halves": halves, "secondary": sec}
    verdict["predictions"] = score_predictions(verdict, win)
    pd.DataFrame(paired_rows).to_csv(src / "paired.csv", index=False)
    per_seed = []
    for name, w in arms.items():
        for s, r in w.iterrows():
            per_seed.append({"arm": name, "seed": s, **r.to_dict()})
    pd.DataFrame(per_seed).to_csv(src / "per_seed.csv", index=False)
    pd.DataFrame(diag).to_csv(src / "secondary.csv", index=False)
    pd.DataFrame(verdict["predictions"]).to_csv(src / "predictions.csv", index=False)
    (src / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json))
    return verdict


def _desc(d) -> dict:
    d = np.asarray(d, float)
    pos, n, p = sign_test(d)
    lo, hi = order_interval(d)
    return {"median": float(np.median(d)), "lo": lo, "hi": hi, "pos": pos, "n_nonzero": n, "p": p}


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------
# predictions (spec 7)
# --------------------------------------------------------------------------

PARENT = {"P1-SNP": .85, "P1-CBP": .85, "P1-REDO": .85, "P2a": .80, "P2b": .55, "P3": .15, "P4": .60, "P5": .25}
MINE = {"P1-SNP": .85, "P1-CBP": .93, "P1-REDO": .95, "P2a": .55, "P2b": .70, "P3": .05, "P4": .35, "P5": .10}
MINE_P6 = {"SNP": {"A_WINS": .03, "B_WINS": .85, "EQUIVALENT_WITHIN_0.005": .03, "UNRESOLVED": .09},
           "CBP": {"A_WINS": .05, "B_WINS": .75, "EQUIVALENT_WITHIN_0.005": .08, "UNRESOLVED": .12},
           "REDO": {"A_WINS": .05, "B_WINS": .70, "EQUIVALENT_WITHIN_0.005": .10, "UNRESOLVED": .15}}


def outcomes(verdict: dict, win: dict) -> dict:
    lab = {t["id"]: t["label"] for t in verdict["tests"]}
    issued = {t["id"]: t["issued"] for t in verdict["tests"]}
    o = {}
    for m in METHODS:
        x = NAME[m]
        o[f"P1-{x}"] = (lab[f"J1-{x}"] == "A_WINS") if issued[f"J1-{x}"] else None
    rd = win["redo"]
    o["P2a"] = bool(prop_p2a(float(rd["dead_frac_l2_late"].median()))) if issued["J1-REDO"] else None
    sna = verdict_comp_late(verdict)
    o["P2b"] = (sna["SNA_minus_REDO_median"] > BAND) if issued["J2-REDO"] else None
    o["P3"] = (lab["J2-CBP"] == "A_WINS") if issued["J2-CBP"] else None
    rel = verdict["secondary"]["snp_relpos"]["relpos_l2"]["median"]
    o["P4"] = bool(prop_p4(rel)) if issued["J1-SNP"] else None
    o["P5"] = any(lab[f"J2-{NAME[m]}"] == "A_WINS" for m in METHODS) \
        if all(issued[f"J2-{NAME[m]}"] for m in METHODS) else None
    for m in METHODS:
        x = NAME[m]
        o[f"P6-{x}"] = lab[f"J2-{x}"] if issued[f"J2-{x}"] else None
    return o


def verdict_comp_late(verdict: dict) -> dict:
    t = next(t for t in verdict["tests"] if t["id"] == "J2-REDO")
    return {"SNA_minus_REDO_median": -t["median"]}


def score_predictions(verdict: dict, win: dict) -> list[dict]:
    o = outcomes(verdict, win)
    rows = []
    for k in PARENT.keys() | MINE.keys():
        out = o.get(k)
        for who, table in (("parent", PARENT), ("mine", MINE)):
            if k not in table:
                continue
            p = table[k]
            rows.append({"prop": k, "who": who, "p": p, "outcome": out,
                         "brier": None if out is None else (p - float(out)) ** 2})
    for x, dist in MINE_P6.items():
        out = o.get(f"P6-{x}")
        rows.append({"prop": f"P6-{x}", "who": "mine", "p": json.dumps(dist), "outcome": out,
                     "brier": None if out is None else sum((q - float(lbl == out)) ** 2 for lbl, q in dist.items())})
    return sorted(rows, key=lambda r: (r["prop"], r["who"]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "judge"])
    ap.add_argument("--src", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        src = Path(a.src) if a.src else REPO / "results" / RUN / "calib"
        out = cmd_select(src)
        print(json.dumps(out, indent=2))
    else:
        src = Path(a.src) if a.src else REPO / "results" / RUN
        v = cmd_judge(src)
        print(json.dumps({k: v[k] for k in v if k in ("label", "missing", "applicability")}, indent=2,
                         default=_json))
        for t in v.get("tests", []):
            print(f"{t['id']:10s} {t['label']:24s} median {t['median']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}] "
                  f"{t['a_pos']}/{t['n_nonzero']} p {t['p']:.2g} holm {t['p_holm']:.2g}")


if __name__ == "__main__":
    main()
