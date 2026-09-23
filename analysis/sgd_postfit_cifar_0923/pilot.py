#!/usr/bin/env python3
"""The eta pilot of spec_sgd_postfit_cifar_0923 §2: stability per eta and the choice of eta_hi / eta_lo.

    python3 analysis/sgd_postfit_cifar_0923/pilot.py [--root results/sgd_postfit_cifar_0923/_pilot]

Selection legs `t12_eta<x>` (the S chain's tasks 1-2 from the init) decide; the descriptive legs
`t49_eta<x>` (task 49 from LR_iid's ckpts/t48.pt) are only reported.  Stable(eta) <=> in both
selection tasks every slot: (i) alive (all losses finite), (ii) switched, (iii) every eval after
s_sw has >= 1188 correct, (iv) every eval after s_sw has ce_st <= 2 ce_st(s_sw), (v) >= 1199
correct at step 30,000, (vi) ce_st(30,000) < ce_st(s_sw).  eta_hi = the largest grid value whose
own and every smaller grid value's legs are stable; eta_lo = eta_hi / 10.  A (Adam) reference
numbers for the same tasks come from LR_iid's trace (float32 CE only: LR_iid has no ce_st).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REF = Path.home() / "Projects/obsidian-research-data/altlabels_cifar_0923/results/altlabels_cifar_0923/LR_iid"
GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = range(10)
STEPS = 30000


def tag(eta: float) -> str:
    return f"{eta:g}"


def slot_stats(tr: dict, t: int, sw: int) -> dict:
    """One slot, one task: numbers at the switch and at the task's end (trace rows).  CE is
    ce_st where the run has it (this experiment's runs), the float32 CE otherwise (LR_iid)."""
    m = tr["task"] == t
    step, cor, n1 = tr["step"][m], tr["correct"][m], tr["n1"][m]
    ce = tr["ce_st"][m] if "ce_st" in tr else tr["ce"][m]
    end = np.nonzero(step == STEPS)[0]
    if sw < 0 or not len(end):
        return {"switched": sw >= 0, "complete": bool(len(end))}
    i_sw, i_end = int(np.nonzero(step == sw)[0][0]), int(end[0])
    ce_sw, ce_end = float(ce[i_sw]), float(ce[i_end])
    return {"switched": True, "complete": True, "switch_step": sw,
            "ce_col": "ce_st" if "ce_st" in tr else "ce",
            "ce_sw": ce_sw, "ce_end": ce_end,
            "ce_max_after_sw": float(ce[i_sw + 1:].max()) if i_end > i_sw else ce_sw,
            "efolds": (math.log(ce_sw / ce_end) if ce_end > 0 else math.inf),
            "efolds_max": (math.log(ce_sw / float(ce[i_sw:].min())) if ce[i_sw:].min() > 0
                           else math.inf),
            "dn1": float(n1[i_end] - n1[i_sw]), "n1_sw": float(n1[i_sw]),
            "correct_end": int(cor[i_end]),
            "min_correct_after_sw": int(cor[i_sw + 1:].min()) if i_end > i_sw else int(cor[i_sw])}


def ref_stats(t_ref: int) -> list[dict]:
    """A: LR_iid's own task t_ref, cut at its hit999 + 500 exactly as the engine would."""
    out = []
    rows = {(int(q["seed"]), int(q["task"])): q
            for q in csv.DictReader(open(REPO / "results/altlabels_cifar_0923/LR_iid/per_task.csv"))}
    for s in SEEDS:
        z = np.load(REF / "trace" / f"LR_std_seed{s}.npz")
        tr = {k: z[k] for k in z.files}
        h = int(rows[(s, t_ref)]["hit999"])
        sw = h + 500 if h >= 0 and h + 500 <= STEPS else -1
        out.append({"seed": s, **slot_stats(tr, t_ref, sw)})
    return out


def leg(d: Path, tasks: list[int]) -> dict:
    """Per task: per slot stats, from the run's per_task rows and traces."""
    if not (d / "provenance.json").exists():
        return {"missing": True}
    rows = list(csv.DictReader(open(d / "per_task.csv")))
    res = {}
    for t in tasks:
        per = []
        for s in SEEDS:
            q = [r for r in rows if int(r["seed"]) == s and int(r["task"]) == t]
            if not q or q[0].get("memo_acc") in (None, ""):
                per.append({"seed": s, "alive": False})
                continue
            z = np.load(d / "trace" / f"LR_std_seed{s}.npz")
            tr = {k: z[k] for k in z.files}
            per.append({"seed": s, "alive": True,
                        **slot_stats(tr, t, int(q[0]["switch_step"]))})
        res[t] = per
    return res


def stable(res: dict) -> tuple[bool, list]:
    why = []
    for t, per in res.items():
        for q in per:
            if not q.get("alive", True):
                why.append((t, q["seed"], "diverged"))
            elif not q.get("switched"):
                why.append((t, q["seed"], "no switch"))
            elif q["min_correct_after_sw"] < 1188:
                why.append((t, q["seed"], f"dip to {q['min_correct_after_sw']} correct"))
            elif q["ce_max_after_sw"] > 2 * q["ce_sw"]:
                why.append((t, q["seed"], f"ce rose to {q['ce_max_after_sw']:.3g} "
                                          f"(> 2 x {q['ce_sw']:.3g})"))
            elif q["correct_end"] < 1199:
                why.append((t, q["seed"], f"correct_end {q['correct_end']}"))
            elif not q["ce_end"] < q["ce_sw"]:
                why.append((t, q["seed"], f"ce_end {q['ce_end']:.3g} >= ce_sw {q['ce_sw']:.3g}"))
    return not why, why


def med(per: list[dict], k: str) -> float:
    v = [q[k] for q in per if q.get("switched") and k in q]
    return float(np.median(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923/_pilot"))
    ap.add_argument("--out", default=str(REPO / "results/sgd_postfit_cifar_0923/pilot.json"))
    a = ap.parse_args()
    root = Path(a.root)
    table, choice = [], {}
    for t in (1, 2, 49):
        table.append({"leg": "A (LR_iid)", "eta": None, "task": t, **summ(ref_stats(t))})
    stab = {}
    for eta in GRID:
        sel = leg(root / f"t12_eta{tag(eta)}", [1, 2])
        des = leg(root / f"t49_eta{tag(eta)}", [1])
        if sel.get("missing"):
            stab[tag(eta)] = {"stable": None, "why": ["missing"]}
        else:
            ok, why = stable(sel)
            stab[tag(eta)] = {"stable": ok, "why": why[:10]}
            for t, per in sel.items():
                table.append({"leg": "S selection", "eta": eta, "task": t, **summ(per)})
        if not des.get("missing"):
            ok49, why49 = stable(des)
            stab[tag(eta)]["t49_stable"] = ok49
            stab[tag(eta)]["t49_why"] = why49[:10]
            table.append({"leg": "S t49 (descriptive)", "eta": eta, "task": 49, **summ(des[1])})
    complete = all(stab[tag(e)]["stable"] is not None and "t49_stable" in stab[tag(e)]
                   for e in GRID)
    hi = lo49 = None
    for eta in GRID:                     # contiguous from the bottom of the grid
        if stab[tag(eta)]["stable"]:
            hi = eta
        else:
            break
    for eta in GRID:                     # v2.1: also stable on the Adam-inflated t49 net
        if stab[tag(eta)]["stable"] and stab[tag(eta)].get("t49_stable"):
            lo49 = eta
        else:
            break
    if complete:
        if hi is None:
            choice = {"eta_hi": None, "note": "no stable eta: no S arms"}
        else:
            lo = lo49 if (lo49 is not None and lo49 < hi) else round(hi / 10, 10)
            choice = {"eta_hi": hi, "eta_lo": lo,
                      "rule": "eta_hi: largest grid value stable (with every smaller one) on "
                              "tasks 1-2; eta_lo (v2.1): the same on tasks 1-2 and the t49 leg "
                              "if below eta_hi, else eta_hi / 10",
                      "eta_lo_source": ("t49-stable" if (lo49 is not None and lo49 < hi)
                                        else "eta_hi/10"),
                      "largest_t49_stable": lo49, "top_of_grid": hi == GRID[-1]}
    res = {"grid": GRID, "stability": stab, "choice": choice, "complete": complete,
           "table": table}
    Path(a.out).write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"{'leg':22s} {'eta':>6s} {'t':>3s} {'efolds':>7s} {'efmax':>6s} {'dn1':>9s} {'minC':>5s} "
          f"{'ce_sw':>9s} {'ce_end':>9s} {'dead':>4s}")
    for q in table:
        eta_s = "" if q["eta"] is None else f"{q['eta']:g}"
        print(f"{q['leg']:22s} {eta_s:>6s} "
              f"{q['task']:3d} {q['efolds_med']:7.2f} {q['efolds_max_med']:6.2f} {q['dn1_med']:9.1f} "
              f"{q['min_correct_after_sw_min']:5d} {q['ce_sw_med']:9.2e} {q['ce_end_med']:9.2e} "
              f"{q.get('n_dead', 0):4d}")
    for e in GRID:
        print(tag(e), stab[tag(e)])
    print("choice:", choice)


def summ(per: list[dict]) -> dict:
    sw = [q for q in per if q.get("switched")]
    return {"efolds_med": med(per, "efolds"), "efolds_max_med": med(per, "efolds_max"),
            "dn1_med": med(per, "dn1"),
            "min_correct_after_sw_min": min((q["min_correct_after_sw"] for q in sw), default=-1),
            "ce_sw_med": med(per, "ce_sw"), "ce_end_med": med(per, "ce_end"),
            "n_dead": sum(1 for q in per if not q.get("alive", True))}


if __name__ == "__main__":
    main()
