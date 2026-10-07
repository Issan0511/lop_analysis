#!/usr/bin/env python3
"""cap_cifar5p1_1007 -- registered readouts, labels, applicability and prediction scores.

    python analysis/cap_cifar5p1_1007/verdict.py            # reads results/cap_cifar5p1_1007/<arm>/

Spec: specs/spec_cap_cifar5p1_1007.md sections 3-6 and 8.  Everything is a within-seed paired
difference over seeds 0-9.  cap12 (main) uses 97.5 % two-sided t intervals (two endpoints,
Bonferroni), cap1 / cap2 use 95 %.  Sign: lower > 0 -> '+', upper < 0 -> '-', otherwise '0'
(an endpoint exactly 0 is '0').  SD = 0 gives the one-point interval at the mean (no division).
The Student-t quantile is the registered continued-fraction one of resp_cifar_ee_0920, imported
unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.resp_cifar_ee_0920.stats import t_quantile      # noqa: E402  (registered, unchanged)

RUN = "cap_cifar5p1_1007"
SRC = REPO / "results" / RUN
HOST = REPO / "results" / "cifar5p1_mlp_0920" / "R_std_lr0.0001"
ARMS = ("ref", "cap1", "cap2", "cap12")
CAPPED = {"cap1": (1,), "cap2": (2,), "cap12": (1, 2)}         # layers whose rows are capped
MAIN_ARM = "cap12"
SEEDS = tuple(range(10))
N_TASKS = 30
EARLY = (1, 3, 5, 7, 9)                                       # spec 3: E0, hard 1-9
LATE = (21, 23, 25, 27, 29)                                   # spec 3: E1, hard 21-29
FRESH_TASK = 29
CAP_TASKS = tuple(range(2, N_TASKS + 1))                      # spec 5 (C)
MAIN_LEVEL = 0.975                                            # spec 4.1
COMP_LEVEL = 0.95
LABELS = ("RESCUED", "LEVEL_ONLY", "LEVEL_UP_GAP_UP", "GAP_ONLY", "NO_EFFECT", "GAP_UP", "WORSE")
UNIT_COLS = ("dead_frac_l2", "mob_l2", "eff_rank_l2", "zbar_l2")


# --------------------------------------------------------------------------
# statistics (spec 4)
# --------------------------------------------------------------------------

def sign(lo: float, hi: float) -> str:
    return "+" if lo > 0 else "-" if hi < 0 else "0"


def interval(x, level: float) -> dict:
    """Paired two-sided t interval of the mean of x (one value per seed)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) != len(SEEDS) or not np.isfinite(x).all():
        raise ValueError(f"interval needs {len(SEEDS)} finite values, got {x}")
    n, mean = len(x), float(x.mean())
    sd = 0.0 if np.all(x == x[0]) else float(x.std(ddof=1))     # identical values: exactly 0, no rounding
    if sd == 0.0:
        lo = hi = mean
    else:
        w = t_quantile(0.5 + level / 2.0, n - 1) * sd / math.sqrt(n)
        lo, hi = mean - w, mean + w
    return {"n": n, "mean": mean, "sd": sd, "lo": lo, "hi": hi, "level": level,
            "degenerate_sd": sd == 0.0, "sign": sign(lo, hi)}


def label(s1: str, s2: str) -> str:
    """spec 4.2: delta1 sign x delta2 sign -> label (delta2 < 0 means less loss)."""
    if s1 == "-":
        return "WORSE"
    return {("+", "-"): "RESCUED", ("+", "0"): "LEVEL_ONLY", ("+", "+"): "LEVEL_UP_GAP_UP",
            ("0", "-"): "GAP_ONLY", ("0", "0"): "NO_EFFECT", ("0", "+"): "GAP_UP"}[(s1, s2)]


def endpoints(per_task: list[dict], fresh: list[dict]) -> dict:
    """Per-seed E0, E1 (online means over the registered hard tasks) and E2 (fresh gap), seed order."""
    on = {(int(r["seed"]), int(r["task"])): float(r["online_acc"]) for r in per_task}
    gap = {int(r["seed"]): float(r["fresh_gap"]) for r in fresh if int(r["task"]) == FRESH_TASK}
    return {"E0": np.array([np.mean([on[s, t] for t in EARLY]) for s in SEEDS]),
            "E1": np.array([np.mean([on[s, t] for t in LATE]) for s in SEEDS]),
            "E2": np.array([gap[s] for s in SEEDS])}


def judge(ep: dict, rows_written: dict, integrity: dict) -> dict:
    """The registered decision (spec 4-5).

    ep[arm] = endpoints(); rows_written[arm][layer] = array over CAP_TASKS of the rows the cap
    wrote, summed over seeds; integrity = {"complete": bool, "consistent": bool, "detail": ...}.
    """
    out = {"applicability": {}, "arms": {}, "interaction": {}}
    A = bool(integrity.get("complete")) and bool(integrity.get("consistent"))
    out["applicability"]["A"] = {"pass": A, **{k: v for k, v in integrity.items() if k != "detail"}}
    if not integrity.get("complete"):
        out["status"] = "INCOMPLETE"
        return out
    if not integrity.get("consistent"):
        out["status"] = "CHECK_FAILED"
        return out
    gap_ref = interval(ep["ref"]["E2"], COMP_LEVEL)
    B = gap_ref["lo"] > 0
    out["applicability"]["B"] = {"pass": B, "ref_gap": gap_ref}
    C = {}
    for arm, layers in CAPPED.items():
        zero = {f"W{l}": [t for t, n in zip(CAP_TASKS, rows_written[arm][l]) if not n > 0]
                for l in layers}
        C[arm] = {"pass": not any(zero.values()), "zero_tasks": zero}
    out["applicability"]["C"] = C
    d = {arm: {k: ep[arm][k] - ep["ref"][k] for k in ("E0", "E1", "E2")} for arm in CAPPED}
    for arm in CAPPED:
        level = MAIN_LEVEL if arm == MAIN_ARM else COMP_LEVEL
        a = {"level": level,
             "d0": interval(d[arm]["E0"], COMP_LEVEL),
             "d1": interval(d[arm]["E1"], level), "d2": interval(d[arm]["E2"], level),
             "d1_95": interval(d[arm]["E1"], COMP_LEVEL), "d2_95": interval(d[arm]["E2"], COMP_LEVEL)}
        a["impaired"] = a["d0"]["hi"] < 0
        if not B:
            a["label"] = "NOT_REPRODUCED"
        elif not C[arm]["pass"]:
            a["label"] = "INAPPLICABLE"
        else:
            a["label"] = label(a["d1"]["sign"], a["d2"]["sign"])
        out["arms"][arm] = a
    for k in ("E1", "E2"):
        out["interaction"][k] = interval(d["cap12"][k] - d["cap1"][k] - d["cap2"][k], COMP_LEVEL)
    out["status"] = "JUDGED" if B else "NOT_REPRODUCED"
    out["main_label"] = out["arms"][MAIN_ARM]["label"]
    out["deltas"] = {arm: {k: v.tolist() for k, v in dd.items()} for arm, dd in d.items()}
    return out


# --------------------------------------------------------------------------
# prediction scoring (spec 8)
# --------------------------------------------------------------------------

PARENT_LABEL_P = {"RESCUED": .45, "LEVEL_ONLY": .15, "GAP_ONLY": .15, "NO_EFFECT": .15, "WORSE": .10}
MINE_LABEL_P = {"RESCUED": .35, "LEVEL_ONLY": .12, "GAP_ONLY": .05, "NO_EFFECT": .25, "WORSE": .20,
                "LEVEL_UP_GAP_UP": .01, "GAP_UP": .02}


def label_score(dist: dict, realized: str) -> dict:
    if realized not in LABELS:
        return {"scored": False, "reason": realized}
    return {"scored": True, "p_realized": dist.get(realized, 0.0),
            "brier": sum((dist.get(k, 0.0) - (k == realized)) ** 2 for k in LABELS)}


def binary(p: float, outcome) -> dict:
    if outcome is None:
        return {"p": p, "scored": False}
    o = bool(outcome)
    return {"p": p, "outcome": o, "scored": True, "brier": (p - o) ** 2}


# --------------------------------------------------------------------------
# reading the run
# --------------------------------------------------------------------------

def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(src: Path) -> dict:
    data = {}
    for arm in ARMS:
        d = src / arm
        data[arm] = {"per_task": read_csv(d / "per_task.csv"), "fresh": read_csv(d / "fresh_control.csv"),
                     "cap": read_csv(d / "cap_task.csv"),
                     "prov": json.loads((d / "provenance.json").read_text())}
    return data


def check_integrity(data: dict, src: Path, host: Path = HOST) -> dict:
    """spec 5 (A): complete, finite, ref = host (via the S-nochange record), task 1 common, fresh
    common, no monitor violation."""
    det, complete, consistent = {}, True, True
    for arm in ARMS:
        pt, fr = data[arm]["per_task"], data[arm]["fresh"]
        keys = {(int(r["seed"]), int(r["task"])) for r in pt if r.get("online_acc", "") != ""}
        ok = keys == {(s, t) for s in SEEDS for t in range(1, N_TASKS + 1)}
        ok &= all(math.isfinite(float(r["online_acc"])) for r in pt if r.get("online_acc", "") != "")
        ok &= sorted(int(r["seed"]) for r in fr) == list(SEEDS)
        ok &= all(math.isfinite(float(r["fresh_gap"])) for r in fr)
        ok &= not data[arm]["prov"].get("divergences")
        det[f"complete_{arm}"] = bool(ok)
        complete &= bool(ok)
    if not complete:
        return {"complete": False, "consistent": False, "detail": det}
    ck = json.loads((src / "checks.json").read_text())
    nc = ck.get("S-nochange", {})
    det["checks_all_pass"] = bool(ck.get("all_pass"))
    det["ref_per_task_sha256"] = sha(src / "ref" / "per_task.csv")
    det["ref_equals_checked_ref"] = det["ref_per_task_sha256"] == nc.get("ref_per_task_sha256")
    det["ref_fresh_equals_committed"] = sha(src / "ref" / "fresh_control.csv") == sha(host / "fresh_control.csv")
    committed = read_csv(host / "per_task.csv")
    mine = data["ref"]["per_task"]
    non_er = [k for k in committed[0] if not k.startswith("eff_rank")]
    det["ref_non_eff_rank_columns_identical"] = (len(committed) == len(mine) and all(
        all(a[k] == b[k] for k in non_er) for a, b in zip(mine, committed)))
    t1 = {arm: [r for r in data[arm]["per_task"] if r["task"] == "1"] for arm in ARMS}
    det["task1_rows_common"] = all(t1[arm] == t1["ref"] for arm in ARMS)
    fresh = {arm: [r["fresh_online_acc"] for r in data[arm]["fresh"]] for arm in ARMS}
    host_fresh = [r["fresh_online_acc"] for r in read_csv(host / "fresh_control.csv")]
    det["fresh_common_and_host"] = all(fresh[arm] == host_fresh for arm in ARMS)
    det["violations"] = {arm: sum(int(r["viol_w1"]) + int(r["viol_w2"]) for r in data[arm]["cap"])
                         for arm in ARMS}
    det["fresh_rows_written"] = {arm: data[arm]["prov"].get("fresh_rows_written") for arm in ARMS}
    consistent = (det["checks_all_pass"] and det["ref_equals_checked_ref"]
                  and det["ref_fresh_equals_committed"] and det["ref_non_eff_rank_columns_identical"]
                  and det["task1_rows_common"] and det["fresh_common_and_host"]
                  and not any(det["violations"].values())
                  and all(v == [0, 0] for v in det["fresh_rows_written"].values()))
    return {"complete": True, "consistent": bool(consistent), "detail": det}


def rows_written(data: dict) -> dict:
    out = {}
    for arm, layers in CAPPED.items():
        out[arm] = {l: np.array([sum(int(r[f"rows_w{l}"]) for r in data[arm]["cap"] if int(r["task"]) == t)
                                 for t in CAP_TASKS]) for l in layers}
    return out


def per_seed_cols(rows: list[dict], col: str) -> dict:
    return {(int(r["seed"]), int(r["task"])): float(r[col]) for r in rows}


def secondary(data: dict) -> dict:
    """spec 6 (i)-(v), REPORT_ONLY."""
    sec = {}
    ref = data["ref"]["per_task"]
    for col in ("w_norm_l2", "w_norm_l1"):
        v = per_seed_cols(ref, col)
        sec[f"i_ref_{col}_up_t1_t29"] = int(sum(v[s, 29] > v[s, 1] for s in SEEDS))
    z = per_seed_cols(ref, "zbar_l2")
    sec["ii_ref_zbar_l2_down_t1_t29"] = int(sum(z[s, 29] < z[s, 1] for s in SEEDS))
    sec["iii"], sec["iv"], sec["v"] = {}, {}, {}
    for arm in ARMS:
        pt, cap = data[arm]["per_task"], data[arm]["cap"]
        row = {}
        for col in UNIT_COLS + ("w_norm_l1", "w_norm_l2", "w_norm_l3", "online_acc"):
            v = per_seed_cols(pt, col)
            row[f"{col}_late_med"] = float(np.median([np.mean([v[s, t] for t in LATE]) for s in SEEDS]))
            row[f"{col}_t29_med"] = float(np.median([v[s, 29] for s in SEEDS]))
            row[f"{col}_t1_med"] = float(np.median([v[s, 1] for s in SEEDS]))
        sec["iii"][arm] = {f"{col}_{w}_med": row[f"{col}_{w}_med"]
                           for col in UNIT_COLS for w in ("late", "t29", "t1")}
        mu = per_seed_cols(cap, "mu2_norm")
        sec["iv"][arm] = {"mu2_norm_t1_med": float(np.median([mu[s, 1] for s in SEEDS])),
                          "mu2_norm_t29_med": float(np.median([mu[s, 29] for s in SEEDS])),
                          "mu2_norm_late_med": float(np.median([np.mean([mu[s, t] for t in LATE])
                                                                for s in SEEDS]))}
        b1, b2 = per_seed_cols(cap, "b1_mean"), per_seed_cols(cap, "b2_mean")
        rmax1, rmax2 = per_seed_cols(cap, "ratio_max_l1"), per_seed_cols(cap, "ratio_max_l2")
        rmed1, rmed2 = per_seed_cols(cap, "ratio_med_l1"), per_seed_cols(cap, "ratio_med_l2")
        sec["v"][arm] = {
            "w_norm_l3_t29_med": row["w_norm_l3_t29_med"], "w_norm_l1_t29_med": row["w_norm_l1_t29_med"],
            "w_norm_l2_t29_med": row["w_norm_l2_t29_med"],
            "rows_w1_total": int(sum(int(r["rows_w1"]) for r in cap)),
            "rows_w2_total": int(sum(int(r["rows_w2"]) for r in cap)),
            "rem_w1_total": float(sum(float(r["rem_w1"]) for r in cap)),
            "rem_w2_total": float(sum(float(r["rem_w2"]) for r in cap)),
            "ratio_max_l1_t29_med": float(np.median([rmax1[s, 29] for s in SEEDS])),
            "ratio_med_l1_t29_med": float(np.median([rmed1[s, 29] for s in SEEDS])),
            "ratio_max_l2_t29_med": float(np.median([rmax2[s, 29] for s in SEEDS])),
            "ratio_med_l2_t29_med": float(np.median([rmed2[s, 29] for s in SEEDS])),
            "b1_mean_t1_med": float(np.median([b1[s, 1] for s in SEEDS])),
            "b1_mean_t29_med": float(np.median([b1[s, 29] for s in SEEDS])),
            "b2_mean_t1_med": float(np.median([b2[s, 1] for s in SEEDS])),
            "b2_mean_t29_med": float(np.median([b2[s, 29] for s in SEEDS])),
            "excess_max_eps32": {f"W{l}": max((float(r[f"excess_w{l}"]) for r in cap), default=float("-inf"))
                                 for l in (1, 2)},
            "online_late_med": row["online_acc_late_med"]}
        zero_by_seed = {}
        if arm in CAPPED:
            for l in CAPPED[arm]:
                w = per_seed_cols(cap, f"rows_w{l}")
                zero_by_seed[f"W{l}"] = {s: int(sum(w[s, t] == 0 for t in CAP_TASKS)) for s in SEEDS}
        sec["v"][arm]["tasks_without_writes_by_seed"] = zero_by_seed
    return sec


def score_predictions(v: dict, sec: dict) -> list[dict]:
    arms = v.get("arms", {})
    main = arms.get(MAIN_ARM, {}).get("label", v.get("status", "?"))
    judged = v.get("status") == "JUDGED"
    d1 = {a: arms[a]["d1"]["mean"] for a in arms} if judged else {}
    preds = []

    def add(who, prop, res):
        preds.append({"who": who, "proposition": prop, **res})

    add("parent", "cap12 label distribution", label_score(PARENT_LABEL_P, main))
    add("claude", "cap12 label distribution", label_score(MINE_LABEL_P, main))
    p2 = d1["cap2"] > d1["cap1"] if judged else None
    add("parent", "mean d1(cap2) > mean d1(cap1)", binary(.60, p2))
    add("claude", "mean d1(cap2) > mean d1(cap1)", binary(.35, p2))
    add("parent", "ref ||W2|| up t1->t29 in 10/10", binary(.80, sec["i_ref_w_norm_l2_up_t1_t29"] == 10))
    imp = arms[MAIN_ARM]["impaired"] if judged else None
    add("parent", "IMPAIRED on cap12", binary(.30, imp))
    add("claude", "IMPAIRED on cap12", binary(.15, imp))
    add("claude", "cap1 d1 sign + (95%)", binary(.45, arms["cap1"]["d1"]["sign"] == "+" if judged else None))
    add("claude", "cap2 d1 sign + (95%)", binary(.30, arms["cap2"]["d1"]["sign"] == "+" if judged else None))
    cpass = (all(c["pass"] for c in v["applicability"]["C"].values())
             if "C" in v.get("applicability", {}) else None)
    add("claude", "(C) holds for cap1, cap2, cap12", binary(.85, cpass))
    add("claude", "cap12 dead_frac_l2 late median < ref",
        binary(.60, sec["iii"]["cap12"]["dead_frac_l2_late_med"] < sec["iii"]["ref"]["dead_frac_l2_late_med"]))
    add("claude", "cap12 w_norm_l3 t29 median > ref",
        binary(.70, sec["v"]["cap12"]["w_norm_l3_t29_med"] > sec["v"]["ref"]["w_norm_l3_t29_med"]))
    add("claude", "E1 interaction 95% interval contains 0",
        binary(.55, v["interaction"]["E1"]["sign"] == "0" if judged else None))
    return preds


def write_outputs(src: Path, v: dict, sec: dict, preds: list[dict], ep: dict) -> None:
    (src / "verdict.json").write_text(json.dumps({"verdict": v, "secondary": sec, "predictions": preds},
                                                 indent=2, default=float))
    with open(src / "paired.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "endpoint", "level", "n", "mean", "sd", "lo", "hi", "sign", "degenerate_sd"])
        for arm, a in v.get("arms", {}).items():
            for k in ("d0", "d1", "d2", "d1_95", "d2_95"):
                q = a[k]
                w.writerow([arm, k, q["level"], q["n"], f"{q['mean']:.10g}", f"{q['sd']:.10g}",
                            f"{q['lo']:.10g}", f"{q['hi']:.10g}", q["sign"], q["degenerate_sd"]])
        for k, q in v.get("interaction", {}).items():
            w.writerow(["interaction", k, q["level"], q["n"], f"{q['mean']:.10g}", f"{q['sd']:.10g}",
                        f"{q['lo']:.10g}", f"{q['hi']:.10g}", q["sign"], q["degenerate_sd"]])
    with open(src / "per_seed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "arm", "E0", "E1", "E2", "d0", "d1", "d2"])
        for arm in ARMS:
            for i, s in enumerate(SEEDS):
                dd = [ep[arm][k][i] - ep["ref"][k][i] for k in ("E0", "E1", "E2")]
                w.writerow([s, arm] + [f"{ep[arm][k][i]:.10g}" for k in ("E0", "E1", "E2")]
                           + [f"{x:.10g}" for x in dd])
    with open(src / "predictions.csv", "w", newline="") as fh:
        keys = ["who", "proposition", "p", "outcome", "scored", "brier", "p_realized", "reason"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for p in preds:
            w.writerow(p)
    with open(src / "secondary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "quantity", "value"])
        for part in ("iii", "iv", "v"):
            for arm, d in sec[part].items():
                for k, val in d.items():
                    w.writerow([arm, f"{part}:{k}", json.dumps(val) if isinstance(val, dict) else val])
        for k in ("i_ref_w_norm_l2_up_t1_t29", "i_ref_w_norm_l1_up_t1_t29", "ii_ref_zbar_l2_down_t1_t29"):
            w.writerow(["ref", k, sec[k]])


def main(src: Path = SRC) -> dict:
    data = load(src)
    integ = check_integrity(data, src)
    ep = {arm: endpoints(data[arm]["per_task"], data[arm]["fresh"]) for arm in ARMS} if integ["complete"] else {}
    v = judge(ep, rows_written(data), integ) if integ["complete"] else judge({}, {}, integ)
    v["integrity_detail"] = integ.get("detail")
    sec = secondary(data) if integ["complete"] else {}
    # spec 8: nothing is scored on an incomplete or inconsistent run
    preds = score_predictions(v, sec) if v.get("status") in ("JUDGED", "NOT_REPRODUCED") else []
    write_outputs(src, v, sec, preds, ep)
    print(json.dumps({"status": v.get("status"), "main_label": v.get("main_label"),
                      "labels": {a: x["label"] for a, x in v.get("arms", {}).items()},
                      "impaired": {a: x["impaired"] for a, x in v.get("arms", {}).items()}}, indent=1))
    return v


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SRC)
