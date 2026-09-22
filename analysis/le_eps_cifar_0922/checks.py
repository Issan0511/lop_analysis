#!/usr/bin/env python3
"""Checks for le_eps_cifar_0922 (spec §6) that do not need the GPU runs:

  S2b  the engine's Adam line with eps=1e-3 equals an independent torch.optim.Adam(eps=1e-3)
       replica on the same 75 batches (R=1, seed 0, eager, cpu); the replica with eps=1e-8 differs.
  S3   a checkpoint written with one eps refuses to resume under another.
  S4   provenance.json carries adam_eps.

S1 (bit identity of --eps 1e-8 with S-A) and S2a (t1 rows differ under --eps 1e-3) read the two
GPU check runs; see soff() below.  Everything is recorded in results/le_eps_cifar_0922/checks.json.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import layer_chimera_cifar_0921 as C          # noqa: E402
from src import pmnist_rlcifar_0907 as RC               # noqa: E402

OUT = REPO / "results" / "le_eps_cifar_0922"
SA = REPO / "results" / "layer_chimera_cifar_0921"
SA_RAW = Path("/home/issan/Projects/obsidian-research-data/layer_chimera_cifar_0921/results/layer_chimera_cifar_0921")
RES: dict[str, dict] = {}


def record(name, ok, detail):
    RES[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:400]}", flush=True)
    return bool(ok)


def replica(init, X, labels, orders, eps, lr=1e-3, steps=75):
    act = C.make_act("LE")
    act.init_state(1, torch.device("cpu"), key="replica")
    P = [q.clone().requires_grad_(True) for q in init]
    opt = torch.optim.Adam(P, lr=lr, betas=(0.9, 0.999), eps=eps)
    for j in range(steps):
        idx = orders[0, j * C.BATCH:(j + 1) * C.BATCH]
        xb, yb = X[:, idx], labels[:, idx]
        *_, z3 = C.forward(P, xb, act, train=True)
        loss = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return [q.detach() for q in P]


def s2b(tmp: Path, cifar) -> bool:
    dev = torch.device("cpu")
    eps = 1e-3
    d = {}
    out = tmp / "s2b_engine"
    shutil.rmtree(out, ignore_errors=True)
    C.run("LE", [0], ["raw"], 1, 1, dev, out, eps=eps, debug=d, graph=False, cifar=cifar,
          progress=lambda m: None)
    snap = np.load(C.snapshot_path(out, "LE", "raw", 0, 1))
    eng = [torch.from_numpy(snap[k]) for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
    X = C.slot_inputs(cifar, 0, "raw", dev)[None]
    init = [q.clone() for q in d["init"]]
    same = replica(init, X, d["labels"][0], d["orders"][0], eps)
    other = replica(init, X, d["labels"][0], d["orders"][0], 1e-8)
    err_same = max(float((a[0] - b).abs().max()) for a, b in zip(same, eng))
    err_other = max(float((a[0] - b).abs().max()) for a, b in zip(other, eng))
    moved = max(float((a - b[0]).abs().max()) for a, b in zip(eng, init))
    # arithmetic: 75 float32 Adam steps of size <= lr = 1e-3 accumulate round-off of order
    # 75 * 2^-24 * |p| ~ 1e-5 at |p| ~ 1; a different eps changes whole steps (order lr)
    ok = err_same < 1e-5 and err_other > 1e-4 and err_other > 10 * err_same
    return record("S2b eps reaches the update (torch.optim.Adam replica)", ok,
                  {"eps": eps, "max_abs_err_same_eps": err_same, "max_abs_err_eps_1e-8": err_other,
                   "max_abs_move": moved, "rule": "same < 1e-5 and other > 1e-4 and other > 10*same"})


def s3(tmp: Path, cifar) -> bool:
    dev = torch.device("cpu")
    out = tmp / "s3_ckpt"
    shutil.rmtree(out, ignore_errors=True)
    C.run("LE", [0], ["raw"], 1, 1, dev, out, eps=1e-3, graph=False, cifar=cifar, checkpoint=True,
          snapshots=False, progress=lambda m: None)
    refused = False
    try:
        C.run("LE", [0], ["raw"], 2, 1, dev, out, eps=1e-4, graph=False, cifar=cifar, checkpoint=True,
              snapshots=False, resume=True, progress=lambda m: None)
    except SystemExit as e:
        refused = "another configuration" in str(e)
    resumed = json.loads((out / "provenance.json").read_text())
    C.run("LE", [0], ["raw"], 2, 1, dev, out, eps=1e-3, graph=False, cifar=cifar, checkpoint=True,
          snapshots=False, resume=True, progress=lambda m: None)
    prov = json.loads((out / "provenance.json").read_text())
    ok = refused and prov["resumed_at_task"] == [2] and prov["adam_eps"] == 1e-3
    record("S4 provenance carries adam_eps", prov.get("adam_eps") == 1e-3,
           {"adam_eps": prov.get("adam_eps"), "adam_betas": prov.get("adam_betas")})
    return record("S3 resume refuses another eps", ok,
                  {"refused": refused, "resumed_at_task": prov["resumed_at_task"],
                   "first_prov_tasks": resumed["n_tasks"]})


def rows(path: Path, tasks: int, drop=()):
    with open(path) as fh:
        return [{k: v for k, v in q.items() if k not in drop}
                for q in csv.DictReader(fh) if int(q["task"]) <= tasks]


def soff(tasks: int) -> bool:
    """S1: --eps 1e-8 run == S-A LE (rows, snapshots); S2a: --eps 1e-3 t1 rows differ."""
    e8 = OUT / "_checks" / "LE_e8"
    a, b = rows(e8 / "per_task.csv", tasks), rows(SA / "LE" / "per_task.csv", tasks)
    same_rows = a == b
    snaps_ok, n = True, 0
    for s in range(10):
        for t in range(tasks + 1):
            p = np.load(C.snapshot_path(e8, "LE", "raw", s, t))
            q = np.load(C.snapshot_path(SA_RAW / "LE", "LE", "raw", s, t))
            n += 1
            if set(p.files) != set(q.files) or not all(np.array_equal(p[k], q[k]) for k in p.files):
                snaps_ok = False
    record("S1 --eps 1e-8 reproduces S-A LE (rows)", same_rows, {"rows": len(a), "rows_ref": len(b)})
    record("S1 --eps 1e-8 reproduces S-A LE (snapshots bit-identical)", snaps_ok, {"snapshots": n})
    e3 = rows(OUT / "_checks" / "LE_e3_t1" / "per_task.csv", 1)
    b1 = [q for q in b if int(q["task"]) == 1]
    e31 = [q for q in e3 if int(q["task"]) == 1]
    diff = sum(1 for x, y in zip(e31, b1) if x["online_acc"] != y["online_acc"]) if "online_acc" in b1[0] else None
    keys = [k for k in b1[0] if k not in ("arm", "cond", "seed", "slot", "lr", "task")]
    ndiff = sum(1 for x, y in zip(e31, b1) if any(x[k] != y[k] for k in keys))
    return record("S2a --eps 1e-3 changes every slot's t1 row", ndiff == len(b1) == 10,
                  {"slots_differing": ndiff, "slots": len(b1), "columns_compared": len(keys)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cpu", "soff"])
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tmp = OUT / "_checks"
    tmp.mkdir(parents=True, exist_ok=True)
    f = OUT / "checks.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    if a.stage == "cpu":
        cifar = RC.Cifar10()
        s2b(tmp, cifar)
        s3(tmp, cifar)
    else:
        soff(a.tasks)
    prev.update(RES)
    prev["_git"] = C.git_state()
    f.write_text(json.dumps(prev, indent=2, default=str))
    print("all pass:", all(v["pass"] for k, v in prev.items() if not k.startswith("_")))


if __name__ == "__main__":
    main()
