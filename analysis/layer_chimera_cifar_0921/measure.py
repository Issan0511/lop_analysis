#!/usr/bin/env python3
"""Per-state layer quantities for layer_chimera_cifar_0921 (spec §4.2), rebuilt from snapshots.

    python3 analysis/layer_chimera_cifar_0921/measure.py --cells LL,EL,GL,LE,EE,GE,GG
    python3 analysis/layer_chimera_cifar_0921/measure.py --cells LL --tasks 3 --out DIR

Each state is replayed with the run's own slot order on the same device, which reproduces the
forward `evaluate` ran bit for bit; check S4 asserts that against the saved float16 z and the
registered columns of per_task.csv.  Nothing here trains.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import layer_chimera_cifar_0921 as C
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC

COLS = ["cell", "act1", "act2", "cond", "seed", "task", "layer", "p_pos", "d_signed", "D_abs",
        "Q", "Q_diag", "zero_disagree", "undefined_ratio", "dead_frac", "mu_norm", "w_row_med",
        "cos_med", "zbar_med", "zsd_med", "a_rms"]


def states(out: Path, cell: str, slots, task: int, device, cifar):
    """Stacked (P, X) for one task over the run's slots, in the run's order."""
    ds = [np.load(C.snapshot_path(out, cell, cd, s, task)) for s, cd in slots]
    P = [torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device)
         for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
    X = torch.stack([C.slot_inputs(cifar, s, cd, device) for s, cd in slots])
    z16 = [np.stack([d[k] for d in ds]) if "z1" in ds[0].files else None for k in ("z1", "z2")]
    return P, X, z16


def labels(slots, task, device, shift: int = 0):
    Y = []
    for s, cd in slots:
        g = H.stream("rlc_labels", s)
        y = None
        for _ in range(task + shift):
            y = RC.task_labels(g)
        Y.append(y)
    return torch.stack(Y).to(device) if task else None


@torch.no_grad()
def _layer_rows(z, a, act, layer, h_in):
    """The spec §4.2 columns for one layer, per slot."""
    R = z.shape[0]
    out = []
    zd = z.double()
    m, sd = zd.mean(1), zd.std(1, unbiased=False)               # (R, H), ddof=0
    mu = h_in.double().mean(1)                                  # (R, n_in)
    with torch.enable_grad():
        zz = z.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(act.phi(zz, layer, True).sum(), zz)[0].detach()
    gd = act.dphi(z, layer)
    for r in range(R):
        ok = sd[r] > 0
        ratio = torch.where(ok, m[r] / torch.where(ok, sd[r], torch.ones_like(sd[r])),
                            torch.zeros_like(m[r]))
        out.append({
            "p_pos": float((z[r] > 0).double().mean()),
            "d_signed": float(ratio[ok].mean()) if ok.any() else float("nan"),
            "D_abs": float(ratio[ok].abs().mean()) if ok.any() else float("nan"),
            "Q": float((g[r].abs() < C.DEAD_TOL).double().mean()),
            "Q_diag": float((gd[r].abs() < C.DEAD_TOL).double().mean()),
            "zero_disagree": int(((g[r] == 0) != (gd[r] == 0)).sum()),
            "undefined_ratio": int((~ok).sum()),
            "dead_frac": float((gd[r].abs().amax(0) < C.DEAD_TOL).double().mean()),
            "mu_norm": float(mu[r].norm()),
            "zbar_med": float(z[r].mean(0).median()),
            "zsd_med": float(z[r].std(0).median()),
            "a_rms": float(a[r].double().pow(2).mean().sqrt()),
        })
    return out


def run_cell(out: Path, cell: str, tasks: int, device, cifar, check: bool, shift: int = 0):
    prov = json.loads((out / "provenance.json").read_text())
    slots = [(q["seed"], q["cond"]) for q in prov["slots"]]
    act = C.make_act(cell)
    a1n, a2n = C.CELLS[cell]
    rows, probs = [], []
    for t in range(tasks + 1):
        P, X, z16 = states(out, cell, slots, t, device, cifar)
        z1, a1, z2, a2, logits = C.forward(P, X, act)
        for li, (z, a, h) in enumerate(((z1, a1, X), (z2, a2, a1))):
            W = P[2 * li]                                  # li = 0 -> W1, li = 1 -> W2
            mu = h.double().mean(1)
            for r, d in enumerate(_layer_rows(z, a, act, li, h)):
                wr = W[r].double()
                cos = (wr @ mu[r]) / (wr.norm(dim=1) * mu[r].norm()).clamp(min=1e-300)
                d.update({"cell": cell, "act1": a1n, "act2": a2n, "cond": slots[r][1],
                          "seed": slots[r][0], "task": t, "layer": li + 1,
                          "w_row_med": float(wr.norm(dim=1).median()), "cos_med": float(cos.median())})
                rows.append({k: d[k] for k in COLS})
        if check and t > 0:                                     # S4 / S6
            Y = labels(slots, t, device, shift)
            acc = (logits.argmax(-1) == Y).float().mean(1)      # the engine's own expression/dtype
            for r, (s, cd) in enumerate(slots):
                probs.append({"task": t, "seed": s, "cond": cd,
                              "z16_exact": bool((z1[r].cpu().numpy().astype(np.float16) == z16[0][r]).all()
                                                and (z2[r].cpu().numpy().astype(np.float16) == z16[1][r]).all()),
                              "memo_acc": f"{float(acc[r]):.10g}"})   # H.write_csv's format
    return rows, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.CELL_ORDER))
    ap.add_argument("--src", default=str(C.OUT_ROOT))
    ap.add_argument("--out", default=str(C.OUT_ROOT))
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="S4/S6: replay equals the saved z and memo_acc")
    ap.add_argument("--label-shift", type=int, default=0, help="mutation control: use another task's labels")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    cifar = RC.Cifar10()
    src, out = Path(a.src), Path(a.out)
    rows, checks, t0 = [], [], time.time()
    for cell in a.cells.split(","):
        r, pr = run_cell(src / cell, cell, a.tasks, device, cifar, a.check, a.label_shift)
        rows += r
        print(f"{cell}: {len(r)} rows ({time.time() - t0:.0f} s)", flush=True)
        if a.check:
            import csv as _csv
            reg = {(int(q["seed"]), q["cond"], int(q["task"])): q
                   for q in _csv.DictReader(open(src / cell / "per_task.csv"))
                   if q.get("memo_acc")}
            for q in pr:
                k = (q["seed"], q["cond"], q["task"])
                q["cell"] = cell
                # the engine writes f"{x:.10g}" of a float32-derived value; 10 significant digits
                # round-trip a float32 exactly, so the registered column must match character for
                # character -- no tolerance is needed or allowed here
                q["memo_matches"] = k in reg and reg[k]["memo_acc"] == q["memo_acc"]
            checks += pr
    H.write_csv(out / "measure.csv", rows)
    prov = {"run_id": C.EXPERIMENT, "stage": "measure", **C.git_state(), "cells": a.cells,
            "tasks": a.tasks, "device": str(device), "torch": torch.__version__,
            "wall_s": time.time() - t0, "rows": len(rows)}
    (out / "measure_provenance.json").write_text(json.dumps(prov, indent=2))
    if a.check:
        bad = [q for q in checks if not (q["z16_exact"] and q["memo_matches"])]
        res = {"check": "S4 S-replay / S6 S-floor", "pass": not bad, "states": len(checks),
               "label_shift": a.label_shift, "failures": bad[:20],
               "mutation": "run again with --label-shift 1: memo_matches must then fail "
                           "(the labels are what S6's floor is built from)"}
        (out / "checks_measure.json").write_text(json.dumps(res, indent=2))
        print(("PASS" if not bad else f"FAIL ({len(bad)})"), "S4/S6 over", len(checks), "states")
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
