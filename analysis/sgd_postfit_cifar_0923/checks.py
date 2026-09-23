#!/usr/bin/env python3
"""Checks C1-C5 of spec_sgd_postfit_cifar_0923 §5, with their mutation controls.

    python3 analysis/sgd_postfit_cifar_0923/checks.py --run DIR:MODE [--run DIR:MODE ...] \
        [--ref <altlabels LR_iid dir>] [--pair DIR_A:DIR_B:TASKS ...] [--out checks.json]

MODE is none | idle | adam | adam_restore | freeze | sgd | adam_ce (what DIR was run with;
idle = freeze with a switch past the task's end).  Against the reference LR_iid run (same
R = 10 layout, same streams):
  C1  none:          every trace row of every task DIR has == the reference's (bit for bit)
  C7  idle:          the same (the masked path with no slot switched is the parent's step)
  C2  adam:          the same, and the per_task rows' shared columns too
  C3  freeze/sgd:    task 1, each slot: rows at step <= s_sw == the reference; the rows after
                     s_sw must NOT all equal it (mutation control: the comparison can fail)
      adam_restore:  task 1 whole == the reference (the reset acts after the task's last step)
      adam_ce:       task 1, each slot: rows at step <= pin_step == the reference, later not
  C4  per_task flags, every task x switched slot:
        freeze: pf_P_same = 1 and pf_mv_same = 1, n1..n3 constant over [s_sw, 30000], and
                n1 moved between step 0 and s_sw (the slot learned: the mask was reset)
        adam_ce: the same from pin_step (pinned slots), pin_step > switch_step
        sgd:    pf_mv_same = 1 and pf_P_same = 0
        adam_restore: pf_mv_restored = 1 and pf_mv_same = 0 and pf_P_same = 0
        adam:   pf_P_same = 0 and pf_mv_same = 0 (mutation control: the flags can fail)
  C5/C6 --pair A:B:1-2  trace rows of those tasks equal between two runs (pilot vs main;
                     resumed vs continuous)
  C9  --xfork DIR    each fork point's (end, end) cell == LR_iid's task t+1 up to the bundle's
                     end; the (sw, sw) cell must differ (mutation control)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REF = Path.home() / "Projects/obsidian-research-data/altlabels_cifar_0923/results/altlabels_cifar_0923/LR_iid"
COLS = ("correct", "ce", "margin_med", "n1", "n2", "n3", "sig_med")
SEEDS = range(10)


def trace(d: Path, s: int) -> dict:
    z = np.load(d / "trace" / f"LR_std_seed{s}.npz")
    return {k: z[k] for k in z.files}


def rows_of(tr: dict, t: int, lo=None, hi=None) -> dict:
    m = tr["task"] == t
    if lo is not None:
        m &= tr["step"] > lo
    if hi is not None:
        m &= tr["step"] <= hi
    return {k: tr[k][m] for k in ("step",) + COLS}


def same(a: dict, b: dict) -> bool:
    """Bit-for-bit equality of every column (NaN == NaN counts as equal: same bits)."""
    if len(a["step"]) != len(b["step"]) or not np.array_equal(a["step"], b["step"]):
        return False
    return all(np.array_equal(a[k].view(np.uint64) if a[k].dtype == np.float64 else a[k],
                              b[k].view(np.uint64) if b[k].dtype == np.float64 else b[k])
               for k in COLS)


REPO = Path(__file__).resolve().parents[2]
REF_ROWS = REPO / "results/altlabels_cifar_0923/LR_iid/per_task.csv"    # committed; not archived


def per_task(d: Path) -> list[dict]:
    f = d / "per_task.csv"
    return list(csv.DictReader(open(f if f.exists() or d != REF else REF_ROWS)))


def check_run(d: Path, mode: str, ref: Path) -> dict:
    out = {"dir": str(d), "mode": mode}
    rows = per_task(d)
    dead = [(q["seed"], q["task"]) for q in rows if q.get("memo_acc") in (None, "")]
    out["dead_rows"] = dead
    rows = [q for q in rows if q.get("memo_acc") not in (None, "")]
    tasks = sorted({int(q["task"]) for q in rows})
    out["tasks"] = [tasks[0], tasks[-1]] if tasks else []
    if mode in ("none", "adam", "idle"):
        bad = []
        for s in SEEDS:
            a, b = trace(d, s), trace(ref, s)
            for t in tasks:
                if not same(rows_of(a, t), rows_of(b, t)):
                    bad.append((s, t))
        key = {"none": "C1", "idle": "C7", "adam": "C2_trace"}[mode]
        out[key] = {"ok": not bad, "mismatch": bad[:20], "n_checked": len(tasks) * len(SEEDS)}
        if mode == "adam":
            rref = {(q["seed"], q["task"]): q for q in per_task(ref)}
            shared = [k for k in rows[0] if k in next(iter(rref.values())) and k not in ("lr",)]
            badr = [(q["seed"], q["task"], k) for q in rows for k in shared
                    if (q["seed"], q["task"]) in rref and q[k] != rref[(q["seed"], q["task"])][k]]
            out["C2_rows"] = {"ok": not badr, "mismatch": badr[:20], "cols": len(shared)}
    if mode in ("freeze", "sgd", "adam_restore", "adam_ce"):
        col = "pin_step" if mode == "adam_ce" else "switch_step"
        sw = {int(q["seed"]): int(q[col]) for q in rows if q["task"] == "1"}
        pre_bad, post_same = [], []
        for s in SEEDS:
            if s not in sw or sw[s] < 0:
                continue
            a, b = trace(d, s), trace(ref, s)
            if mode == "adam_restore":
                if not same(rows_of(a, 1), rows_of(b, 1)):
                    pre_bad.append(s)
                continue
            if not same(rows_of(a, 1, hi=sw[s]), rows_of(b, 1, hi=sw[s])):
                pre_bad.append(s)
            if same(rows_of(a, 1, lo=sw[s]), rows_of(b, 1, lo=sw[s])):
                post_same.append(s)            # the mutation control failed: nothing changed
        out["C3"] = {"ok": not pre_bad and not post_same, "pre_switch_mismatch": pre_bad,
                     "post_switch_identical": post_same, "switch_task1": sw}
    # C4: flags
    need = {"freeze": {"pf_P_same": "1", "pf_mv_same": "1"},
            "adam_ce": {"pf_P_same": "1", "pf_mv_same": "1"},
            "sgd": {"pf_P_same": "0", "pf_mv_same": "1"},
            "adam_restore": {"pf_P_same": "0", "pf_mv_same": "0", "pf_mv_restored": "1"},
            "adam": {"pf_P_same": "0", "pf_mv_same": "0"}}.get(mode)
    if need:
        bad, n_sw, n_nosw = [], 0, 0
        col = "pin_step" if mode == "adam_ce" else "switch_step"
        for q in rows:
            if q.get(col) in (None, "", "-1"):
                n_nosw += 1
                continue
            n_sw += 1
            for k, v in need.items():
                if q[k] != v:
                    bad.append((q["seed"], q["task"], k, q[k]))
            if mode == "adam_ce" and not int(q["pin_step"]) > int(q["switch_step"]):
                bad.append((q["seed"], q["task"], "pin_step<=switch_step", q["pin_step"]))
        const_bad, no_learn = [], []
        if mode in ("freeze", "adam_ce"):
            swm = {(int(q["seed"]), int(q["task"])): int(q[col]) for q in rows}
            for s in SEEDS:
                a = trace(d, s)
                for t in tasks:
                    k = swm.get((s, t), -1)
                    if k < 0:
                        continue
                    r = rows_of(a, t, lo=k - 1)          # the pin step itself and everything after
                    for c in ("n1", "n2", "n3"):
                        if len(np.unique(r[c])) != 1:
                            const_bad.append((s, t, c))
                    r0 = rows_of(a, t, hi=k)
                    if r0["n1"][0] == r0["n1"][-1]:
                        no_learn.append((s, t))
        out["C4"] = {"ok": not bad and not const_bad and not no_learn, "flag_mismatch": bad[:20],
                     "not_constant": const_bad[:20], "no_learning_before_pin": no_learn[:20],
                     "switched": n_sw, "not_switched": n_nosw}
    return out


def check_pair(a: Path, b: Path, tasks: list[int]) -> dict:
    bad = [(s, t) for s in SEEDS for t in tasks
           if not same(rows_of(trace(a, s), t), rows_of(trace(b, s), t))]
    return {"a": str(a), "b": str(b), "tasks": tasks, "ok": not bad, "mismatch": bad[:20]}


def check_xfork(d: Path, ref: Path) -> dict:
    """C9: the (end, end) cell of fork point t is LR_iid's own task t+1 until the bundle ends."""
    bad, ctrl_same = [], []
    for sub in sorted(d.glob("t??_Wend_Send")):
        t = int(sub.name[1:3])
        for s in SEEDS:
            a, b = trace(sub, s), trace(ref, s)
            ra = rows_of(a, 1)
            rb = {k: v[:len(ra["step"])] for k, v in rows_of(b, t + 1).items()}
            if not same(ra, rb):
                bad.append((t, s))
            c = rows_of(trace(d / f"t{t:02d}_Wsw_Ssw", s), 1)
            n = min(len(c["step"]), len(ra["step"]))
            if same({k: v[:n] for k, v in c.items()}, {k: v[:n] for k, v in ra.items()}):
                ctrl_same.append((t, s))
    return {"ok": not bad and not ctrl_same, "end_end_mismatch": bad[:20],
            "sw_sw_identical_to_end_end": ctrl_same[:20]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xfork", default=None)
    ap.add_argument("--run", action="append", default=[], help="DIR:MODE")
    ap.add_argument("--pair", action="append", default=[], help="DIR_A:DIR_B:T0-T1")
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"ref": a.ref, "runs": [], "pairs": []}
    for x in a.run:
        d, mode = x.rsplit(":", 1)
        res["runs"].append(check_run(Path(d), mode, Path(a.ref)))
    for x in a.pair:
        da, db, tt = x.split(":")
        t0, t1 = (int(v) for v in tt.split("-"))
        res["pairs"].append(check_pair(Path(da), Path(db), list(range(t0, t1 + 1))))
    if a.xfork:
        res["C9"] = check_xfork(Path(a.xfork), Path(a.ref))
    txt = json.dumps(res, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
