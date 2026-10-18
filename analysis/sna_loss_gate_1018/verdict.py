#!/usr/bin/env python3
"""Registered CNN verdict; reject incomplete/invalid outputs before computing intervals."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ("SNA", "SNAfrz1", "SNA_lossgate")


def comparison(d, margin=0.):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(1018)
    means = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(d.mean()), ci95=[float(lo), float(hi)],
                positive_seeds=int((d > 0).sum()), margin=margin)


def compute(out):
    prov = json.loads((out / "provenance.json").read_text())
    if prov.get("status") != "COMPLETED":
        raise ValueError("run has not completed")
    if (prov.get("tasks"), prov.get("epochs"), prov.get("lr")) != (30, 400, .001):
        raise ValueError("not the registered production protocol")
    expected_slots = {(a, s) for a in ARMS for s in range(10, 20)}
    if set(map(tuple, prov.get("slots", []))) != expected_slots or len(prov["slots"]) != 30:
        raise ValueError("wrong slots")
    with (out / "per_task.csv").open() as f:
        rows = list(csv.DictReader(f))
    keys = [(r["arm"], int(r["seed"]), int(r["task"])) for r in rows]
    expected = {(a, s, t) for a, s in expected_slots for t in range(1, 31)}
    if len(rows) != 900 or set(keys) != expected:
        raise ValueError("incomplete or duplicate task records")
    vals = {k: float(r["online_acc"]) for k, r in zip(keys, rows)}
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in vals.values()):
        raise ValueError("invalid accuracy")
    if any(vals[(a, s, 1)] != vals[("SNA", s, 1)] for a, s in expected_slots):
        raise ValueError("task1 arms differ before intervention")
    means = {a: [np.mean([vals[a, s, t] for t in range(21, 31)]) for s in range(10, 20)]
             for a in ARMS}
    main = comparison(np.subtract(means["SNA_lossgate"], means["SNA"]))
    freeze = comparison(np.subtract(means["SNA_lossgate"], means["SNAfrz1"]), -.005)
    main["label"] = ("BETTER_THAN_SNA" if main["ci95"][0] > 0 else
                     "WORSE_THAN_SNA" if main["ci95"][1] < 0 else "UNRESOLVED_VS_SNA")
    freeze["label"] = ("NONINFERIOR_TO_FREEZE" if freeze["ci95"][0] > -.005 else
                       "INFERIOR_TO_FREEZE" if freeze["ci95"][1] < -.005 else "UNRESOLVED_VS_FREEZE")
    return dict(status="COMPLETED", label=main["label"], primary=main, freeze=freeze,
                window="tasks21-30", seed_means=means,
                switch_ce_over100={a: sum(float(r["switch_ce"]) > 100 for r in rows if r["arm"] == a)
                                   for a in ARMS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    out = p.parse_args().out
    try:
        result = compute(out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = dict(status="INCOMPLETE_OR_INVALID", label=None, error=str(exc))
    (out / "verdict.json").write_text(json.dumps(result, indent=2))
    (out / "summary.md").write_text("# Loss-gated Snake CNN production\n\n" +
        (f"Primary: **{result['label']}**\n\nPrimary and freeze comparisons: `verdict.json`.\n"
         if result["label"] else f"No scientific verdict: {result['status']} ({result['error']}).\n"))
    print(json.dumps(result, indent=2), flush=True)
    if result["label"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
