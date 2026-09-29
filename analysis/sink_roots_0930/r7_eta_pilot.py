#!/usr/bin/env python3
"""R7 (spec_postfit_elu_cifar_0924 §2): decide eta_S from the pilot (S, tasks 1-2, 10 seeds per grid point)."""
import csv, json, sys
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/r7/_pilot")
GRID = [0.001, 0.003, 0.01, 0.03, 0.1]
NEED99, NEED999 = 1188, 1199


def check_seed(d: Path):
    """Returns (stable, reasons) for one seed's 2-task pilot."""
    why = []
    if not (d / "provenance.json").exists():
        return None, ["missing"]
    rows = [r for r in csv.DictReader(open(d / "per_task.csv"))]
    tr = np.load(next((d / "trace").glob("*.npz")))
    if len(rows) < 2 or any(r.get("diverged") in ("True", "1") for r in rows):
        why.append("diverged or incomplete")
    if not np.isfinite(tr["ce"]).all() or not np.isfinite(tr["ce_st"]).all():
        why.append("non-finite loss")
    for r in rows:
        t = int(r["task"])
        sw = int(float(r["switch_step"])) if r.get("switch_step") not in (None, "", "-1") else -1
        if t == 1 and not (0 < sw <= 30000):
            why.append("task 1 did not switch")
        if not (0 < sw <= 30000):
            continue
        m = tr["task"] == t
        st, cor, ces = tr["step"][m], tr["correct"][m], tr["ce_st"][m]
        at = np.nonzero(st == sw)[0]
        if not len(at):
            why.append(f"t{t}: no eval at s_sw {sw}")
            continue
        c0 = ces[at[0]]
        after = st > sw
        if (cor[after] < NEED99).any():
            why.append(f"t{t}: correct < 1188 after s_sw")
        if (ces[after] > 2 * c0).any():
            why.append(f"t{t}: ce_st > 2 ce_st(s_sw)")
        end = np.nonzero(st == 30000)[0]
        if not len(end) or cor[end[0]] < NEED999:
            why.append(f"t{t}: correct(30000) < 1199")
        if not len(end) or not ces[end[0]] < c0:
            why.append(f"t{t}: ce_st(30000) >= ce_st(s_sw)")
    return (not why), why


def main():
    res = {}
    for eta in GRID:
        per = {}
        for s in range(10):
            ok, why = check_seed(RAW / f"eta{eta}" / f"s{s}")
            per[s] = {"stable": ok, "why": why}
        complete = all(v["stable"] is not None for v in per.values())
        res[eta] = {"complete": complete, "stable": complete and all(v["stable"] for v in per.values()),
                    "seeds": per}
        print(eta, "complete" if complete else "INCOMPLETE", "stable" if res[eta]["stable"] else "unstable",
              {s: v["why"] for s, v in per.items() if v["why"]})
    eta_s = None
    for eta in GRID:
        if not res[eta]["complete"]:
            break
        if res[eta]["stable"]:
            eta_s = eta
        else:
            break
    all_done = all(res[e]["complete"] for e in GRID)
    print("eta_S =", eta_s, "(all grid points complete)" if all_done else "(pilot not complete)")
    out = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930" / "R7_eta_pilot.json"
    out.write_text(json.dumps({"eta_S": eta_s, "complete": all_done, "grid": {str(k): v for k, v in res.items()}},
                              indent=1))


if __name__ == "__main__":
    main()
