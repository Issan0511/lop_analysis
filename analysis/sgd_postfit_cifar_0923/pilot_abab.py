#!/usr/bin/env python3
"""追補 1 §2: the pure-SGD ABAB pilot -- pass/fail per eta and the choice of eta*.

    python3 analysis/sgd_postfit_cifar_0923/pilot_abab.py [--root results/sgd_postfit_cifar_0923/_pilot_abab]

Pass(eta) <=> in all 4 tasks every slot: alive (finite losses), hit999 within 30,000 steps, and
every trace eval after hit999 has >= 1188 correct.  eta* = the largest grid value that passes
together with every smaller one.  Fallback (spec): if no eta reaches hit999, the same with hit99.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GRID = (0.01, 0.03, 0.1, 0.3)
SEEDS = range(10)


def leg(d: Path, hitcol: str) -> dict:
    rows = list(csv.DictReader(open(d / "per_task.csv")))
    out = {"why": [], "hit": {}, "n1_end": {}, "min_after_hit": {}}
    for t in range(1, 5):
        hs, ns, mins = [], [], []
        for s in SEEDS:
            q = [r for r in rows if int(r["seed"]) == s and int(r["task"]) == t]
            if not q or q[0].get("memo_acc") in (None, ""):
                out["why"].append((t, s, "diverged"))
                continue
            h = int(q[0][hitcol])
            z = np.load(d / "trace" / f"LR_std_seed{s}.npz")
            m = z["task"] == t
            step, cor = z["step"][m], z["correct"][m]
            ns.append(float(z["n1"][m][-1]))
            if h < 0:
                out["why"].append((t, s, f"no {hitcol}"))
                continue
            hs.append(h)
            after = cor[step > h]
            mn = int(after.min()) if len(after) else int(cor[step == h][0])
            mins.append(mn)
            if mn < 1188:
                out["why"].append((t, s, f"dip to {mn} after {hitcol}"))
        out["hit"][t] = float(np.median(hs)) if hs else None
        out["n1_end"][t] = float(np.median(ns)) if ns else None
        out["min_after_hit"][t] = min(mins) if mins else None
    out["pass"] = not out["why"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923/_pilot_abab"))
    ap.add_argument("--out", default=str(REPO / "results/sgd_postfit_cifar_0923/pilot_abab.json"))
    a = ap.parse_args()
    root = Path(a.root)
    res = {"grid": GRID}
    for hitcol in ("hit999", "hit99"):
        legs = {f"{e:g}": leg(root / f"sa_eta{e:g}", hitcol) for e in GRID}
        best = None
        for e in GRID:
            if legs[f"{e:g}"]["pass"]:
                best = e
            else:
                break
        res[hitcol] = {"legs": legs, "eta_star": best}
        for e in GRID:
            L = legs[f"{e:g}"]
            print(f"[{hitcol}] eta {e:5g}: pass {L['pass']}  median hit t1-4 {L['hit']}  "
                  f"n1_end {L['n1_end']}  min after hit {L['min_after_hit']}  why {L['why'][:4]}")
        print(f"[{hitcol}] eta* = {best}")
        if best is not None:
            res["choice"] = {"eta_star": best, "hit_rule": hitcol}
            break
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print("choice:", res.get("choice"))


if __name__ == "__main__":
    main()
