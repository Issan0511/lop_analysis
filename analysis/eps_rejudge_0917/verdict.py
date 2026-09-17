#!/usr/bin/env python3
"""eps_rejudge_0917 verdict: registered labels (spec 2-4) from kernels.json, l1_branch/s*.json and the
tables.py outputs.  Writes checks.json, verdict.csv, verdict.json, c2_table.csv, c2_start.csv and
c2_eps30full.csv (summary.md is written by hand from these files).

    python3 analysis/eps_rejudge_0917/verdict.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "eps_rejudge_0917"
GROUPS = {"layer1": ("W1", "b1"), "layer2": ("W2", "b2"), "readout": ("W3", "b3")}
MAIN_T = (15, 20)
KEYS = ("n", "M", "Z", "E", "Q", "Z_moved30", "still_had_grad", "still_m_only", "epsdom0", "M_epsdom0")


def read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def c2_label(sE):
    if sE is None:
        return "NOT_TESTABLE"
    return "ZERO_NOT_EPS" if sE < 0.1 else ("EPS_REAL" if sE >= 0.5 else "BOTH")


def main():
    kern = json.loads((OUT / "kernels.json").read_text())
    tab = json.loads((OUT / "tables_checks.json").read_text())
    seeds = [json.loads((OUT / "l1_branch" / f"s{s}.json").read_text()) for s in range(10)]
    rows = [r for d in seeds for r in d["rows"]]
    assert len(rows) == 60, len(rows)

    # ---------------- checks
    s1_bad = [(r["seed"], r["t"]) for r in rows if not r["S1_hash_match"]]
    s2a_bad = [(r["seed"], r["t"]) for r in rows
               if not r["S2a_replay_equal"] or any(v != 0 for v in r["classes_replay_E"].values())]
    s2b = [{"seed": r["seed"], "t": r["t"], "mutant_Z": r["S2b_mutant_Z"], "natural_Z": r["S2b_natural_Z"]}
           for r in rows if "S2b_mutant_Z" in r]
    s2b_pass = len(s2b) == 2 and all(x["mutant_Z"]["W2"] < x["natural_Z"]["W2"] for x in s2b)
    checks = {
        "S0": kern["S0"],
        "S1": {"n": len(rows), "hash_mismatch": s1_bad,
               "online_mismatch": [(r["seed"], r["t"]) for r in rows if not r["S1_online_match"]],
               "pass": not s1_bad},
        "S2a": {"bad": s2a_bad, "pass": not s2a_bad},
        "S2b": {"cases": s2b, "pass": s2b_pass},
        "S3": tab["S3"],
        "S4": tab["S4"],
        "provenance": [d["provenance"] for d in seeds],
    }
    checks["all_pass"] = all(checks[k]["pass"] for k in ("S0", "S1", "S2a", "S2b", "S3", "S4"))
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))

    # ---------------- C2
    ok = {(r["seed"], r["t"]) for r in rows if r["S1_hash_match"]}
    c2 = []
    labels = {}
    for t in (2, 5, 7, 10, 15, 20):
        rs = [r for r in rows if r["t"] == t and (r["seed"], r["t"]) in ok]
        for g, names in GROUPS.items():
            acc = {k: sum(r["classes"][n][k] for r in rs for n in names) for k in KEYS}
            still = acc["Z"] + acc["E"] + acc["Q"]
            sE = acc["E"] / still if still else None
            row = {"t": t, "group": g, "n_states": len(rs), **acc, "still": still,
                   "share_M": acc["M"] / acc["n"], "share_Z_of_still": (acc["Z"] / still if still else None),
                   "sE": sE, "label": c2_label(sE) if len(rs) == 10 else "NOT_TESTABLE"}
            for kind in ("rms_ep1", "rms_task"):
                for b in ("zero", "below_eps", "at_or_above_eps"):
                    row[f"{kind}_{b}"] = sum(r[kind][n][b] for r in rs for n in names) / acc["n"]
            row["nz_task_zero_share"] = sum(r["nz_task_zero"][n] for r in rs for n in names) / acc["n"]
            row["eps30_big_states"] = sum(1 for r in rs if r["eps30_ep1_big_update_at"] is not None)
            row["eps30_nonfinite_states"] = sum(1 for r in rs if not r["eps30_ep1_finite"])
            c2.append(row)
            if g == "layer2":
                labels[t] = row["label"]
    start = []
    for t in (2, 5, 7, 10, 15, 20):
        rs = [r for r in rows if r["t"] == t]
        start.append({"t": t, **{f"{k}_mean": sum(r["start"][k] for r in rs) / len(rs)
                                 for k in ("dead1", "pairzero1", "dead2", "pairzero2", "U2_margin_units")},
                      "U2_max_max": max(r["start"]["U2_max"] for r in rs),
                      "nat_online_mean": sum(r["nat_online"] for r in rs) / len(rs),
                      "nat_end_dead2_mean": sum(r["nat_end"]["dead2"] for r in rs) / len(rs)})
    full = []
    for r in rows:
        if "eps30full_online" in r:
            full.append({"seed": r["seed"], "t": r["t"], "nat_online": r["nat_online"],
                         "eps30_online": r["eps30full_online"], "diff": r["eps30full_online"] - r["nat_online"],
                         "nat_end_dead2": r["nat_end"]["dead2"], "nat_end_pairzero2": r["nat_end"]["pairzero2"],
                         "eps30_end_dead2": (r.get("eps30full_end") or {}).get("dead2"),
                         "eps30_end_pairzero2": (r.get("eps30full_end") or {}).get("pairzero2"),
                         "eps30_big_update_at": r["eps30full_big_update_at"], "eps30_finite": r["eps30full_finite"]})
    write_csv(OUT / "c2_table.csv", c2)
    write_csv(OUT / "c2_start.csv", start)
    write_csv(OUT / "c2_eps30full.csv", full)

    # ---------------- B
    bs = read_csv(OUT / "l3_seeds.csv")
    elu = {int(r["seed"]): r["class"] for r in bs if r["arm"] == "ELU1"}
    smi = {int(r["seed"]): r["class"] for r in bs if r["arm"] == "SMINH"}
    smi_arm = next(iter(set(smi.values()))) if len(set(smi.values())) == 1 else "SPLIT"

    # ---------------- C1 / C3
    cs = read_csv(OUT / "l1_seeds.csv")
    c1, c3 = {}, {}
    for arm in ("ref", "cap_perp", "cap_par", "cap_both"):
        rr = [r for r in cs if r["arm"] == arm]
        fad = sum(1 for r in rr if r["C1"] == "FREEZE_AT_DEATH")
        fwl = sum(1 for r in rr if "FROZEN_WHILE_LIVE" in r["C1"])
        nfd = sum(1 for r in rr if "NO_FULL_DEATH" in r["C1"])
        if fad >= 8:
            c1[arm] = "ZERO_NOT_EPS"
        elif fwl >= 3:
            c1[arm] = "EPS_REAL"
        elif nfd >= 6:
            c1[arm] = "NOT_REACHED"
        else:
            c1[arm] = "MIXED_C1"
        c1[arm + "_counts"] = f"FREEZE_AT_DEATH only {fad}/10, FROZEN_WHILE_LIVE {fwl}/10, NO_FULL_DEATH {nfd}/10"
        k3 = [r["C3"] for r in rr]
        c3[arm] = ("HISTORY_DECAY" if all(k == "HISTORY_DECAY" for k in k3)
                   else "NOT_DECAY" if "NOT_DECAY" in k3 else "PARTIAL")
        c3[arm + "_counts"] = ", ".join(f"{k} {k3.count(k)}" for k in sorted(set(k3)))

    d1 = tab["D1"]
    verdict = [
        {"part": "B", "target": "L3 ELU1 (description, seen before registration)", "label": " / ".join(f"s{s} {elu[s]}" for s in sorted(elu))},
        {"part": "B", "target": "L3 SMINH", "label": smi_arm, "detail": " / ".join(f"s{s} {smi[s]}" for s in sorted(smi))},
        *[{"part": "C2", "target": f"L1 layer 2, t{t}", "label": labels[t], "main": t in MAIN_T} for t in (2, 5, 7, 10, 15, 20)],
        *[{"part": "C1", "target": f"L1 {arm}", "label": c1[arm], "detail": c1[arm + "_counts"]} for arm in ("ref", "cap_perp", "cap_par", "cap_both")],
        *[{"part": "C3", "target": f"L1 {arm} layer 1", "label": c3[arm], "detail": c3[arm + "_counts"]} for arm in ("ref", "cap_perp", "cap_par", "cap_both")],
        {"part": "D1", "target": "L2 eps arms (l2_wall variant)", "label": d1},
    ]
    write_csv(OUT / "verdict.csv", verdict)
    preds = [
        ("P1", "C2 layer 2 t15 and t20 both ZERO_NOT_EPS", labels[15] == "ZERO_NOT_EPS" and labels[20] == "ZERO_NOT_EPS"),
        ("P1b", "C2 layer 2 t10 ZERO_NOT_EPS", labels[10] == "ZERO_NOT_EPS"),
        ("P2", "C1 ref ZERO_NOT_EPS", c1["ref"] == "ZERO_NOT_EPS"),
        ("P3", "C3 ref HISTORY_DECAY", c3["ref"] == "HISTORY_DECAY"),
        ("P4", "B SMINH EPS_REAL", smi_arm == "EPS_REAL"),
        ("P5", "D1 L2_NO_FLOOR", d1 == "L2_NO_FLOOR"),
    ]
    (OUT / "verdict.json").write_text(json.dumps({"verdict": verdict, "predictions": preds,
                                                  "checks_all_pass": checks["all_pass"]}, indent=2))
    for v in verdict:
        print(v)
    for p in preds:
        print(p)
    print("checks all_pass:", checks["all_pass"])


if __name__ == "__main__":
    main()
