#!/usr/bin/env python3
"""Synthetic tests of verdict.py (exact sign-test values, Holm, labels, divergence exclusion,
predictions, completeness, S-reuse).  python3 analysis/cnn_center_budget_1020/test_verdict.py"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict as V                                    # noqa: E402


def write_csv(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make(root: Path, D: dict, width: dict, sparse: dict, diverge=(), drop=()):
    """Baseline first780 = 0.5 everywhere; the centred arm = 0.5 + D[cell][seed] / 100."""
    for b in V.BUDGETS:
        rows, probe, fresh, lag = [], [], [], []
        for a in V.ARMS:
            act = "ReLU" if a in ("RCB", "R") else "ELU"
            cen = a in ("RCB", "ECB")
            for i, s in enumerate(V.SEEDS):
                for t in range(1, V.N_TASKS + 1):
                    if (a, b, s, t) in drop:
                        continue
                    x = float(0.5 + (D[f"{act}|{b}"][i] / 100 if cen else 0.0))
                    r = {"arm": a, "cfg": a, "seed": s, "slot": 0, "task": t, "first780_online_acc": repr(x),
                         "first100_online_acc": "0.1", "online_acc": repr(x), "test_acc": "0.4",
                         "diverged": "True" if (a, b, s) in diverge else "False", "bad_step": "-1"}
                    if cen:
                        for site in V.SITES:
                            for k in ("fire_rate", "lag_u0", "lag_end", "blag_first10", "blag_last100"):
                                r[f"{k}_{site}"] = "0.1"
                    rows.append(r)
                    if cen and t in V.LATE:
                        for u in V.LAG_POINTS:
                            if u <= b:
                                lag.append({"arm": a, "seed": s, "task": t, "update": u,
                                            **{f"lag_{site}": "0.5" for site in V.SITES}})
                for bd in range(0, V.N_TASKS + 1):
                    for site in V.SITES:
                        q = {"arm": a, "cfg": a, "seed": s, "site": site, "boundary": bd}
                        for k in V.TRAJ:
                            q[k] = "1.0"
                        for k in V.PPOS_BINS:
                            q[f"ppos_{k}_frac"] = "0.1"
                        if site == "f2":
                            q["s_med"] = repr(float(np.exp(width[f"{act}|{b}"][i])) if cen else 1.0)
                            q["ppos_sparse_frac"] = repr(float(0.2 + (sparse[f"{act}|{b}"][i] if cen else 0.0)))
                        probe.append(q)
                for t in V.FRESH:
                    fresh.append({"arm": a, "cfg": a, "seed": s, "task": t, "gap_online_acc": "0.0"})
        d = root / f"b{b}"
        write_csv(d / "per_task.csv", rows)
        write_csv(d / "probe_stats.csv", probe)
        write_csv(d / "fresh_control.csv", fresh)
        write_csv(d / "lag.csv", lag)
        (d / "provenance.json").write_text(json.dumps({"status": "COMPLETED"}))


def make_prev(prev: Path, root: Path):
    for b in V.BUDGETS:
        for n in ("per_task", "probe_stats", "fresh_control"):
            rows = [r for r in V.read_csv(root / f"b{b}" / f"{n}.csv") if r["arm"] in ("R", "ELU")]
            write_csv(prev / f"b{b}" / f"{n}.csv", rows)


def main():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    st = V.sign_test
    check("sign 10/0", st([1] * 10)["p"] == 2 / 1024)
    check("sign 9/1", st([1] * 9 + [-1])["p"] == 2 * 11 / 1024)
    check("sign 8/2", st([1] * 8 + [-1] * 2)["p"] == 2 * 56 / 1024)
    check("sign 5/5", st([1] * 5 + [-1] * 5)["p"] == 1.0)
    check("sign zeros dropped", st([1] * 9 + [0])["n_nonzero"] == 9 and st([1] * 9 + [0])["p"] == 2 / 512)
    check("sign all zero", st([0] * 10)["p"] == 1.0)
    h = V.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    check("holm", abs(h["a"] - 0.04) < 1e-15 and abs(h["c"] - 0.09) < 1e-15 and abs(h["b"] - 0.09) < 1e-15
          and h["d"] == 0.5)

    tmp = Path(tempfile.mkdtemp())
    try:
        z = np.zeros(10)
        D = {"ReLU|780": np.full(10, 2.0), "ReLU|7800": np.array([1.0] * 9 + [-1.0]),
             "ELU|780": np.array([0.5, -0.5] * 5), "ELU|7800": np.full(10, -1.0)}
        W = {"ReLU|780": z, "ReLU|7800": np.full(10, 0.3), "ELU|780": z, "ELU|7800": np.full(10, 0.2)}
        SP = {"ReLU|780": z, "ReLU|7800": np.full(10, -0.05), "ELU|780": z, "ELU|7800": np.full(10, 0.05)}
        root = tmp / "results" / "cnn_center_budget_1020"
        make(root, D, W, SP)
        make_prev(tmp / "results" / "cifar5p1_cnn_budget_1019", root)
        r = V.verdict(root)
        c = r["main"]["cells"]
        check("labels", c["ReLU|780"]["label"] == "CENTER_HIGHER" and c["ELU|7800"]["label"] == "CENTER_LOWER"
              and c["ReLU|7800"]["label"] == "CENTER_HIGHER" and c["ELU|780"]["label"] == "TIE")
        check("holm values", abs(c["ReLU|7800"]["p_holm"] - 2 * 22 / 1024) < 1e-12
              and abs(c["ReLU|780"]["p_holm"] - 4 * 2 / 1024) < 1e-12)
        check("D in pt", abs(c["ReLU|780"]["mean"] - 2.0) < 1e-9)
        pr = r["predictions_claude_fable"]
        check("Q1..Q4 hold", pr["Q1"]["holds"] and pr["Q2"]["holds"] and pr["Q3"]["holds"] and pr["Q4"]["holds"])
        check("Q5 partial (ReLU true, ELU false)", pr["Q5"]["label"] == "PARTIAL" and pr["Q5"]["ReLU"]["holds"]
              and not pr["Q5"]["ELU"]["holds"])
        check("interaction", abs(r["secondary"]["interaction"]["mean"] - ((-1 - 0) - (0.8 - 2.0))) < 1e-9)
        check("reuse pass", r["S_reuse"]["label"] == "PASS")
        # one mutated value in the previous run -> FAIL with one mismatch
        p = tmp / "results" / "cifar5p1_cnn_budget_1019" / "b7800" / "per_task.csv"
        rows = V.read_csv(p)
        rows[3]["test_acc"] = "0.41"
        write_csv(p, rows)
        r2 = V.verdict(root)
        check("reuse fail detected", r2["S_reuse"]["label"] == "FAIL"
              and r2["S_reuse"]["7800"]["per_task"]["mismatches"] == 1)
        # a diverged baseline run removes its seed from the ReLU cells of that budget
        shutil.rmtree(root)
        make(root, D, W, SP, diverge={("R", 780, 2003)})
        r3 = V.verdict(root)
        check("divergence excluded", r3["main"]["cells"]["ReLU|780"]["n"] == 9
              and r3["main"]["cells"]["ReLU|7800"]["n"] == 10)
        check("Q1 counts the excluded seed as not positive", r3["predictions_claude_fable"]["Q1"]["positive"] == 9)
        # a missing row -> INCOMPLETE for that cell only
        shutil.rmtree(root)
        make(root, D, W, SP, drop={("ECB", 7800, 2005, 17)})
        r4 = V.verdict(root)
        check("incomplete", r4["main"]["cells"]["ELU|7800"]["label"] == "INCOMPLETE"
              and r4["main"]["cells"]["ReLU|7800"]["label"] == "CENTER_HIGHER")
        s = V.summary(r)
        check("summary renders", "主の判定" in s and "Q5" in s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
