#!/usr/bin/env python3
"""Checks for layer_chimera_cifar_0921 (spec §7).  Every check names what it would catch.

    python3 src/layer_chimera_cifar_0921_checks.py fast            # S2, S3, S5, S8, S9
    python3 src/layer_chimera_cifar_0921_checks.py soff            # S1, after the reference runs
    python3 src/layer_chimera_cifar_0921_checks.py cost            # S7

`fast` needs no run on disk; it trains 2 two-epoch tasks on check seeds (100+), never 0-9.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import layer_chimera_cifar_0921 as C
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC

OUT = C.OUT_ROOT
RESULTS = []


def record(name: str, ok: bool, detail: dict) -> bool:
    RESULTS.append({"check": name, "pass": bool(ok), **detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail, default=float)}", flush=True)
    return ok


def toy(R=3, B=8, seed=0, device=None):
    """A small stacked state: parameters from the host's init, inputs from the host's CIFAR."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = [torch.stack([H.init_params(s, device, C.DIMS)[i].detach() for s in range(100, 100 + R)])
         for i in range(6)]
    X = torch.randn(R, B, C.DIMS[0], generator=g).to(device) * 0.5
    Y = torch.randint(0, C.N_CLASSES, (R, B), generator=g).to(device)
    return [q.contiguous() for q in P], X, Y


# --------------------------------------------------------------------------
# S2: the layer dispatch is the one the cell names
# --------------------------------------------------------------------------
def s_layer(device) -> bool:
    P, X, _ = toy(device=device)
    ok, worst = True, {}
    for cell in C.CELL_ORDER:
        act = C.make_act(cell)
        a1n, a2n = C.CELLS[cell]
        r1, r2 = C.make_arm(a1n), C.make_arm(a2n)
        with torch.no_grad():
            z1, a1, z2, a2, log = C.forward(P, X, act)
            # independent rebuild: the host's own affine maps, one layer at a time
            W1, b1, W2, b2, W3, b3 = P
            rz1 = torch.baddbmm(b1[:, None, :], X, W1.transpose(1, 2))
            ra1 = r1.phi(rz1, 0, False)
            rz2 = torch.baddbmm(b2[:, None, :], ra1, W2.transpose(1, 2))
            ra2 = r2.phi(rz2, 1, False)
            rlog = torch.baddbmm(b3[:, None, :], ra2, W3.transpose(1, 2))
            same = all(torch.equal(u, v) for u, v in ((z1, rz1), (a1, ra1), (z2, rz2),
                                                      (a2, ra2), (log, rlog)))
            # mutation: swap the two layers' activations
            swapped = C.Chimera(cell, C.make_arm(a2n), C.make_arm(a1n))
            sz1, sa1, sz2, sa2, slog = C.forward(P, X, swapped)
            differs = (a1n == a2n) or not torch.equal(slog, log)
        ok &= same and differs
        worst[cell] = {"identical_to_reference": bool(same), "swap_differs": bool(differs)}
    return record("S2 S-layer", ok, {"cells": worst,
                                     "mutation": "act1<->act2 swapped must change the logits"})


# --------------------------------------------------------------------------
# S3: the training derivative of layer l is that layer's phi'
# --------------------------------------------------------------------------
def s_grad(device) -> bool:
    P, X, Y = toy(device=device)
    ok, det = True, {}
    for cell in C.CELL_ORDER:
        act = C.make_act(cell)
        a1n, a2n = C.CELLS[cell]
        d1, d2 = C.make_arm(a1n).dphi, C.make_arm(a2n).dphi
        Pg = [q.clone().requires_grad_(True) for q in P]
        W1, b1, W2, b2, W3, b3 = Pg
        z1 = torch.baddbmm(b1[:, None, :], X, W1.transpose(1, 2)); z1.retain_grad()
        a1 = act.phi(z1, 0, True); a1.retain_grad()
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2)); z2.retain_grad()
        a2 = act.phi(z2, 1, True); a2.retain_grad()
        z3 = torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))
        loss = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), Y.reshape(-1),
                               reduction="none").view(X.shape[0], -1).mean(1).sum()
        loss.backward()
        # chain rule with each layer's own analytic phi'
        p2 = (a2.grad * d2(z2.detach(), 1) - z2.grad).abs().max()
        p1 = (a1.grad * d1(z1.detach(), 0) - z1.grad).abs().max()
        s2 = z2.grad.abs().max().clamp(min=1e-30)
        s1 = z1.grad.abs().max().clamp(min=1e-30)
        r2v, r1v = float(p2 / s2), float(p1 / s1)
        # mutation: use layer 1's phi' on layer 2
        m2 = float((a2.grad * d1(z2.detach(), 1) - z2.grad).abs().max() / s2)
        good = r1v <= 1e-6 and r2v <= 1e-6
        caught = (a1n == a2n) or m2 > 1e-6
        ok &= good and caught
        det[cell] = {"rel_l1": r1v, "rel_l2": r2v, "mutation_rel_l2": m2, "caught": bool(caught)}
    return record("S3 S-grad", ok, {"cells": det, "tol": 1e-6,
                                    "mutation": "layer-1 phi' applied to layer 2"})


# --------------------------------------------------------------------------
# S5: what |phi'| < DEAD_TOL means per activation (spec §4.2)
# --------------------------------------------------------------------------
def s_gate(device) -> bool:
    z = torch.linspace(-120, 40, 1_600_001, device=device, dtype=torch.float32)
    det, ok = {}, True
    for arm in ("LR", "ELU", "GELU"):
        a = C.make_arm(arm)
        q = a.dphi(z, 0)
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(a.phi(zz, 0, True).sum(), zz)[0]
        agree = float((g - q).abs().max())
        low = q.abs() < C.DEAD_TOL
        zmax = float(z[low].max()) if low.any() else float("nan")
        frac_signed = float(((q < C.DEAD_TOL) & ~low).float().mean())   # what dropping abs() adds
        det[arm] = {"autograd_vs_dphi_max_abs_diff": agree, "any_low": bool(low.any()),
                    "largest_z_with_low_gate": zmax,
                    "extra_fraction_if_abs_dropped": frac_signed}
        if arm == "LR":
            ok &= not bool(low.any())                       # spec §1.3-1: identically zero
        else:
            ok &= bool(low.any()) and agree <= 1e-7
    # the spec's arithmetic: ELU crosses at ln(1e-6), GELU near -5.394
    ok &= abs(det["ELU"]["largest_z_with_low_gate"] - math.log(1e-6)) < 1e-3
    ok &= abs(det["GELU"]["largest_z_with_low_gate"] - (-5.3941)) < 1e-2
    return record("S5 S-gate", ok, {"arms": det, "dead_tol": C.DEAD_TOL,
                                    "mutation": "dropping abs() adds the negative-gate band "
                                                "(extra_fraction_if_abs_dropped > 0 for GELU)"})


# --------------------------------------------------------------------------
# S8 / S9: resume is bit-exact; a diverged slot does not touch the others
# --------------------------------------------------------------------------
def _short(cell, out, device, tasks=2, epochs=2, seeds=(100, 101, 102), **kw):
    return C.run(cell, list(seeds), ["raw"], tasks, epochs, device, Path(out), **kw)


def s_resume(device, tmp: Path) -> bool:
    a, b = tmp / "res_straight", tmp / "res_resume"
    _short("EL", a, device, checkpoint=False)
    _short("EL", b, device, tasks=1, checkpoint=True)
    _short("EL", b, device, tasks=2, checkpoint=True, resume=True)
    rows = [(a / "per_task.csv").read_text(), (b / "per_task.csv").read_text()]
    same = rows[0] == rows[1]
    snap = all(_npz_equal(C.snapshot_path(a, "EL", "raw", s, t), C.snapshot_path(b, "EL", "raw", s, t))
               for s in (100, 101, 102) for t in (0, 1, 2))
    return record("S8 S-resume", same and snap,
                  {"per_task_identical": same, "snapshots_identical": snap,
                   "mutation": "a ckpt without the Adam moments would change task 2"})


def s_diverge(device, tmp: Path) -> bool:
    a, b = tmp / "div_full", tmp / "div_clean"
    _short("EE", a, device, checkpoint=False, nan_slot=1)
    _short("EE", b, device, checkpoint=False, seeds=(100, 102))
    ra = [r for r in (a / "per_task.csv").read_text().splitlines()[1:] if ",100," in r or ",102," in r]
    rb = (b / "per_task.csv").read_text().splitlines()[1:]
    def strip(rows):
        out = []
        for r in rows:
            f = r.split(",")
            del f[3]                                   # slot index differs between the two runs
            out.append(",".join(f))
        return sorted(out)
    same = strip(ra) == strip(rb)
    prov = json.loads((a / "provenance.json").read_text())
    poisoned = len(prov["divergences"]) == 1 and prov["divergences"][0]["seed"] == 101
    return record("S9 S-diverge", same and poisoned,
                  {"survivors_identical": same, "only_slot_1_diverged": poisoned,
                   "mutation": "a NaN that leaked across slots would change seeds 100/102"})


def _npz_equal(p: Path, q: Path) -> bool:
    if not (p.exists() and q.exists()):
        return False
    a, b = np.load(p), np.load(q)
    return sorted(a.files) == sorted(b.files) and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and
        (a[k] == b[k]).all() for k in a.files)


# --------------------------------------------------------------------------
# S1: the fork equals the parent engine in the configuration we actually run
# --------------------------------------------------------------------------
def s_off(ref_root: Path, run_root: Path, tasks: int) -> bool:
    from src import rlcifar_mlp_battle_0918 as B
    pairs = {"LL": "LR", "EE": "ELU", "GG": "GELU"}
    det, ok = {}, True
    for cell, arm in pairs.items():
        ref, mine = ref_root / arm, run_root / cell
        rr = [l.split(",") for l in (ref / "per_task.csv").read_text().splitlines()]
        mm = [l.split(",") for l in (mine / "per_task.csv").read_text().splitlines()]
        col, tcol = rr[0].index("arm"), rr[0].index("task")
        def rows(t):                                         # drop the arm column, keep t <= tasks
            return [",".join(f[:col] + f[col + 1:]) for f in t[1:] if int(f[tcol]) <= tasks]
        csv_same = rr[0] == mm[0] and rows(rr) == rows(mm)
        snaps = all(_npz_equal(B.snapshot_path(ref, arm, "raw", s, t),
                               C.snapshot_path(mine, cell, "raw", s, t))
                    for s in range(10) for t in range(tasks + 1))
        hz = all(_hist_equal(ref / "hist" / f"{arm}_raw_seed{s}.npz",
                             mine / "hist" / f"{cell}_raw_seed{s}.npz", tasks) for s in range(10))
        det[cell] = {"reference_arm": arm, "per_task_identical": csv_same,
                     "snapshots_identical": snaps, "hist_identical": hz}
        ok &= csv_same and snaps and hz
    # mutation control: EL must match neither reference
    el = run_root / "EL"
    if el.exists():
        clash = [arm for arm in ("LR", "ELU")
                 if _npz_equal(ref_root / arm / "snap" / f"{arm}_raw_seed0" / "t01.npz",
                               C.snapshot_path(el, "EL", "raw", 0, 1))]
        det["EL_mutation"] = {"matches": clash, "expected": []}
        ok &= not clash
    return record("S1 S-off", ok, {"cells": det, "tasks_compared": tasks,
                                   "mutation": "EL must match neither LR nor ELU at t01"})


def _hist_equal(p: Path, q: Path, tasks: int) -> bool:
    if not (p.exists() and q.exists()):
        return False
    a, b = np.load(p), np.load(q)
    if sorted(a.files) != sorted(b.files):
        return False
    for k in a.files:
        u, v = a[k], b[k]
        if k in ("edges", "th_edges"):
            if not (u == v).all():
                return False
        elif not (u[:tasks] == v[:tasks]).all():
            return False
    return True


def s_cost(device, tmp: Path) -> bool:
    t0 = time.time()
    prov = C.run("EE", list(range(100, 110)), ["raw"], 1, 20, device, tmp / "cost", checkpoint=False)
    ms = prov["step_ms_last_task"]
    est = ms * 30_000 * 50 / 3.6e6
    return record("S7 S-cost", True, {"ms_per_step_R10_alone": ms, "hours_per_cell_estimate": est,
                                      "wall_s": time.time() - t0, "note": "single job, no contention"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fast", "soff", "cost"])
    ap.add_argument("--ref", default=str(OUT / "_soff"))
    ap.add_argument("--run", default=str(OUT))
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--tmp", default=str(OUT / "_checks"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    if a.stage == "fast":
        ok = all([s_layer(device), s_grad(device), s_gate(device),
                  s_resume(device, tmp), s_diverge(device, tmp)])
    elif a.stage == "cost":
        ok = s_cost(device, tmp)
    else:
        ok = s_off(Path(a.ref), Path(a.run), a.tasks)
    out = OUT / f"checks_{a.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stage": a.stage, "all_pass": ok, "device": str(device),
                               "torch": torch.__version__, "git": C.git_state(),
                               "checks": RESULTS}, indent=2, default=float))
    print(("ALL PASS " if ok else "FAILED ") + str(out))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
