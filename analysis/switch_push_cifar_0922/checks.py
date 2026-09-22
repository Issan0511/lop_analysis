#!/usr/bin/env python3
"""Checks for switch_push_cifar_0922 (spec §6).

    python3 analysis/switch_push_cifar_0922/checks.py cpu     # S2a, S2b, S3, S4, S5
    python3 analysis/switch_push_cifar_0922/checks.py soff    # S1, S6 (read the GPU check runs)

Results accumulate in results/switch_push_cifar_0922/checks.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import layer_chimera_cifar_0921 as C          # noqa: E402
from src import pmnist_rlcifar_0907 as RC               # noqa: E402

OUT = REPO / "results" / "switch_push_cifar_0922"
SA = REPO / "results" / "layer_chimera_cifar_0921"
SA_RAW = Path("/home/issan/Projects/obsidian-research-data/layer_chimera_cifar_0921/results/layer_chimera_cifar_0921")
RES: dict[str, dict] = {}


def record(name, ok, detail):
    RES[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:360]}", flush=True)
    return bool(ok)


def s2a() -> bool:
    """The masked expression with frz = 1 is the parent's own update, bit for bit."""
    g = torch.Generator().manual_seed(7)
    b1, b2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
    one, zero = torch.ones(()), torch.zeros(())
    d1, d2 = torch.full((), b1), torch.full((), b2)
    bad_same, froze = {}, {}
    for scale in (1e-6, 1e-3, 1.0, 1e3):
        gr = torch.randn(4096, generator=g) * scale
        m0 = torch.randn(4096, generator=g) * scale
        v0 = (torch.randn(4096, generator=g) * scale).abs()
        p0 = torch.randn(4096, generator=g)
        mA, vA, pA = m0.clone(), v0.clone(), p0.clone()
        mA.mul_(b1).add_(gr, alpha=1 - b1)                                  # the parent's three lines
        vA.mul_(b2).addcmul_(gr, gr, value=1 - b2)
        pA.sub_(lr * (mA * one) / ((vA * one).sqrt() + eps))
        mB, vB, pB = m0.clone(), v0.clone(), p0.clone()
        mB.mul_(d1).add_(gr * one, alpha=1 - b1)                            # masked, frz = 1
        vB.mul_(d2).addcmul_(gr * one, gr, value=1 - b2)
        pB.sub_(one * (lr * (mB * one) / ((vB * one).sqrt() + eps)))
        bad_same[scale] = {w: int((x != y).sum()) for x, y, w in
                           ((mA, mB, "m"), (vA, vB, "v"), (pA, pB, "p"))}
        mC, vC, pC = m0.clone(), v0.clone(), p0.clone()
        mC.mul_(one).add_(gr * zero, alpha=1 - b1)                          # masked, frz = 0
        vC.mul_(one).addcmul_(gr * zero, gr, value=1 - b2)
        pC.sub_(zero * (lr * (mC * one) / ((vC * one).sqrt() + eps)))
        froze[scale] = {w: bool(torch.equal(x, y)) for x, y, w in
                        ((m0, mC, "m"), (v0, vC, "v"), (p0, pC, "p"))}
    ok = all(v == 0 for d in bad_same.values() for v in d.values()) and \
         all(v for d in froze.values() for v in d.values())
    return record("S2a masked update: frz=1 is the parent bit for bit, frz=0 changes nothing", ok,
                  {"elements_differing_at_frz_1": bad_same, "unchanged_at_frz_0": froze})


def _short(out: Path, cell, tasks, epochs, cifar, **kw):
    shutil.rmtree(out, ignore_errors=True)
    d = {}
    C.run(cell, [0, 1], ["raw"], tasks, epochs, torch.device("cpu"), out, cifar=cifar, graph=False,
          debug=d, progress=lambda m: None, **kw)
    return d


def s2b_s3_s4(cifar) -> bool:
    """frz = 0 really freezes (S2b); the window sits where the spec says (S3); the budget matches (S4)."""
    tmp = OUT / "_checks"
    spt = C.STEPS_PER_EPOCH * 2                                   # 150 updates per task
    K = 40
    ok_all = True
    sched = {}
    for where in ("switch", "mid"):
        out = tmp / f"win_{where}"
        d = _short(out, "LE", 1, 2, cifar, freeze_steps=K, freeze_where=where)
        frz = np.array(d["frz"])
        first = 0 if where == "switch" else (spt - K) // 2
        want = np.ones(spt)
        want[first:first + K] = 0.0
        sched[where] = {"frozen": int((frz == 0).sum()), "first_frozen": int(np.argmin(frz)),
                        "schedule_matches": bool(np.array_equal(frz, want)), "window_first": first}
        ok_all &= sched[where]["schedule_matches"] and sched[where]["frozen"] == K
    record("S3 the freeze window sits where the spec says", ok_all, sched)
    record("S4 both positions freeze the same number of updates", 
           sched["switch"]["frozen"] == sched["mid"]["frozen"] == K,
           {"K": K, "switch": sched["switch"]["frozen"], "mid": sched["mid"]["frozen"]})

    out = tmp / "frozen_all"                                       # freeze the whole task
    _short(out, "LE", 1, 1, cifar, freeze_steps=C.STEPS_PER_EPOCH, freeze_where="switch")
    a = np.load(C.snapshot_path(out, "LE", "raw", 0, 0))
    b = np.load(C.snapshot_path(out, "LE", "raw", 0, 1))
    same = {k: bool(np.array_equal(a[k], b[k])) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    ok = all(same[k] for k in ("W1", "b1", "W2", "b2")) and not any(same[k] for k in ("W3", "b3"))
    return record("S2b frz=0 freezes W1,b1,W2,b2 bit for bit while W3,b3 train (mutation control)",
                  ok, same) and ok_all


def s5(cifar) -> bool:
    out = OUT / "_checks" / "resume"
    shutil.rmtree(out, ignore_errors=True)
    kw = dict(cifar=cifar, graph=False, checkpoint=True, snapshots=False, progress=lambda m: None)
    C.run("LE", [0], ["raw"], 1, 1, torch.device("cpu"), out, freeze_steps=40, freeze_where="switch", **kw)
    refused = False
    try:
        C.run("LE", [0], ["raw"], 2, 1, torch.device("cpu"), out, freeze_steps=40, freeze_where="mid",
              resume=True, **kw)
    except SystemExit as e:
        refused = "another configuration" in str(e)
    C.run("LE", [0], ["raw"], 2, 1, torch.device("cpu"), out, freeze_steps=40, freeze_where="switch",
          resume=True, **kw)
    prov = json.loads((out / "provenance.json").read_text())
    return record("S5 resume refuses another freeze configuration", refused and prov["resumed_at_task"] == [2],
                  {"refused": refused, "resumed_at_task": prov["resumed_at_task"], "freeze": prov["freeze"]})


def rows(path: Path, tasks: int, drop=()):
    with open(path) as fh:
        return [{k: v for k, v in q.items() if k not in drop}
                for q in csv.DictReader(fh) if int(q["task"]) <= tasks]


def soff(tasks: int) -> bool:
    off = OUT / "_checks" / "LE_off"
    a, b = rows(off / "per_task.csv", tasks), rows(SA / "LE" / "per_task.csv", tasks)
    record("S1 --freeze-steps 0 reproduces S-A LE (rows)", a == b, {"rows": len(a), "rows_ref": len(b)})
    n, snaps_ok = 0, True
    for s in range(10):
        for t in range(tasks + 1):
            p = np.load(C.snapshot_path(off, "LE", "raw", s, t))
            q = np.load(C.snapshot_path(SA_RAW / "LE", "LE", "raw", s, t))
            n += 1
            if not all(np.array_equal(p[k], q[k]) for k in p.files):
                snaps_ok = False
    record("S1 --freeze-steps 0 reproduces S-A LE (snapshots bit-identical)", snaps_ok, {"snapshots": n})
    sw = rows(OUT / "_checks" / "LE_sw750_t1" / "per_task.csv", 1)
    md = rows(OUT / "_checks" / "LE_md750_t1" / "per_task.csv", 1)
    ref = [q for q in b if int(q["task"]) == 1]
    keys = [k for k in ref[0] if k not in ("arm", "cond", "seed", "slot", "lr", "task")]
    d_sw = sum(1 for x, y in zip(sw, ref) if any(x[k] != y[k] for k in keys))
    d_md = sum(1 for x, y in zip(md, ref) if any(x[k] != y[k] for k in keys))
    d_pair = sum(1 for x, y in zip(sw, md) if any(x[k] != y[k] for k in keys))
    return record("S6 switch and mid differ from the parent and from each other at t1",
                  d_sw == d_md == d_pair == 10, {"switch_vs_ref": d_sw, "mid_vs_ref": d_md,
                                                 "switch_vs_mid": d_pair, "slots": len(ref)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cpu", "soff"])
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "checks.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    if a.stage == "cpu":
        cifar = RC.Cifar10()
        s2a(); s2b_s3_s4(cifar); s5(cifar)
    else:
        soff(a.tasks)
    prev.update(RES)
    prev["_git"] = C.git_state()
    prev["_note"] = ("the freeze is exact while gradients are finite; a non-finite gradient would "
                     "propagate through 0 * inf, and the engine marks such slots diverged per task")
    f.write_text(json.dumps(prev, indent=2, default=str))
    print("all pass:", all(v["pass"] for k, v in prev.items() if not k.startswith("_")))


if __name__ == "__main__":
    main()
