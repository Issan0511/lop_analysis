#!/usr/bin/env python3
"""Checks for rlcifar_mlp_battle_0918 (spec §4 and addendum 1).  All must pass before the main run.

    python3 src/rlcifar_mlp_battle_0918_checks.py --only S-act,S-std,S-stack,S-eval,S-rsl-mode,S-diverge,S-snap
    python3 src/rlcifar_mlp_battle_0918_checks.py --only S-reuse                 # ~15 min on the gpu
    python3 src/rlcifar_mlp_battle_0918_checks.py --only S-cost --arm SNA        # one per arm, run side by side
    python3 src/rlcifar_mlp_battle_0918_checks.py --collect                      # parts -> checks.json

Seeds are 100 and up (0-9 are the registered seeds and are never touched here).
Thresholds are derived in each check's docstring, not chosen.

S-act      every arm's gate (dphi, what the metrics read) equals autograd's derivative of the
           phi training uses; kunekune joins are continuous in phi and phi'.  Mutations fail.
S-std      the std transform: plane order R,G,B and the constants' residuals; raw != std.
S-stack    the engine's per-slot streams (labels, batch orders, init) are the host's bit for bit;
           step-0 logits are bit-identical to H.forward; gradients agree to 1e-6; SNA's alpha
           statistic after one update agrees to 1e-6.  Power: slot 0 vs seed 101.
S-eval     the batched metrics and histograms equal the host's evaluate_rl / preact_hist on the
           same weights, for 10 activations.
S-rsl-mode RSL draws r in [l, u] per element per call in training; eval is fixed r and
           deterministic; two runs with the same key are identical.  Mutation: eval with train=True.
S-diverge  a NaN slot is recorded as diverged and leaves the other slots bit-identical.
S-snap     the snapshot's float16 z1/z2 and the replayed float32 z reproduce the run's own
           recorded metrics; the wrong condition does not.
S-reuse    (400 epochs, 3 tasks) the engine's realizations of seed 100 (8 slots, W1 x (1+k 1e-7))
           bracket the 0917 host probe rows (SNA, LR) within 4 sd; mutations (SNA with beta=0,
           LR with slope 0) do not.
S-cost     R=20 (seeds 100-109 x raw,std), 1 task x 20 epochs: ms/step, RSS, GPU memory.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC
from src import pmnist_rlmnist_0906 as RL
from src import rlcifar_mlp_battle_0918 as E

OUTDIR = H.REPO / "results" / "_checks_rlcifar_mlp_battle_0918"
PROBE = Path("/tmp/claude-1000/-home-issan-Projects-claude/a83c5c0e-cc56-415e-8b64-308b2a003a6e/"
             "scratchpad/mlpcifar_probe")
# the 0917 host probe rows (seed 100, cpu) as printed then; used if the scratchpad is gone
PROBE_ROWS = {"SNA": {"online": [0.851496, 0.931502, 0.940740], "memo": [1.0, 1.0, 1.0]},
              "LR": {"online": [0.857496, 0.860054, 0.803471], "memo": [1.0, 1.0, 0.865833]}}
F64_TOL = 1e-9
SEEDS = [100, 101, 102]
quiet = lambda *_: None


def part(name: str, res: dict) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"part_{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(name, "PASS" if res.get("pass") else "FAIL", flush=True)
    return res


# --------------------------------------------------------------------------
def s_act() -> dict:
    """float64, cpu: |dphi - autograd| <= 1e-9.  Noise is ~1e-16 x (1 + |theta|) with |theta| <= 2*2*60,
    i.e. <= 1e-13; the smallest wiring error a mutation makes is 0.2 (a leaky slope) or a moved band
    edge (O(1) gate error over a whole period).  Joins: phi and phi' are compared at z_b +- 1e-9;
    |phi(z_b+d) - phi(z_b-d)| <= 2 d max|phi'| = 4e-9, so the bound is 1e-7; phi' changes by at most
    2 * 2 alpha * d <= 8e-9 across the pair, bound 1e-6.  The mutated join jumps by 1/(2 alpha) >= 0.25.
    ELU is also compared in float32 on the device the run uses (bit-equality on cuda)."""
    dev = torch.device("cpu")
    grid = torch.linspace(-60, 60, 24100, dtype=torch.float64).reshape(1, 241, 100)
    z = torch.cat([grid, 0.37 * grid.flip(1)], 0)
    g = torch.Generator().manual_seed(7)
    alpha = torch.empty(2, 100, dtype=torch.float64).uniform_(0.01, 2.0, generator=g)

    def prep(act):
        act.init_state(2, dev, "S-act")
        if act.adaptive:
            V = (act.c / alpha) ** 2
            act.V = [V.clone(), V.clone()]
        return act

    def auto(act, zz):
        zz = zz.clone().requires_grad_(True)
        y = act.phi(zz, 0, train=False)
        (gr,) = torch.autograd.grad(y.sum(), zz)
        return gr

    res, ok = {"arms": {}}, True
    for arm in E.ARM_ORDER:
        act = prep(E.make_act(arm))
        err = float((auto(act, z) - act.dphi(z, 0)).abs().max())
        res["arms"][arm] = {"max_abs_err_f64": err}
        ok &= err <= F64_TOL

    def joins(act, thetas):
        a = act.alpha(0)[:, None, :]
        out = []
        for th in thetas:
            zb = th / (2 * a)
            lo, hi = zb - 1e-9, zb + 1e-9
            out.append({"theta": th, "phi_jump": float((act.phi(hi, 0) - act.phi(lo, 0)).abs().max()),
                        "gate_jump": float((act.dphi(hi, 0) - act.dphi(lo, 0)).abs().max())})
        return out
    J = {"KKA": (-1.5 * math.pi, 0.5 * math.pi), "KKA23": (-1.5 * math.pi, 0.5 * math.pi),
         "KKT1": (-2.0 * math.pi, math.pi)}
    res["joins"] = {}
    for arm, ths in J.items():
        jj = joins(prep(E.make_act(arm)), ths)
        res["joins"][arm] = jj
        ok &= all(j["phi_jump"] <= 1e-7 and j["gate_jump"] <= 1e-6 for j in jj)

    # mutations: each must be caught by the criterion above
    class LeakyMut(E.Leaky):
        def dphi(self, zz, layer=0):
            return torch.where(zz > 0, torch.ones_like(zz), torch.full_like(zz, 0.3))

    class KKT1Mut(E.SnakeFamily):          # gate band of KKA on the KKT1 function
        def dphi(self, zz, layer=0):
            a = self.alpha(layer)[:, None, :]; th = 2.0 * a * zz
            inside = (th >= -1.5 * math.pi) & (th <= 0.5 * math.pi)
            return torch.where(inside, 1.0 + torch.sin(th), torch.ones_like(zz))

    class KKAMut(E.SnakeFamily):           # right line shifted by 1/(2 alpha)
        def phi(self, zz, layer=0, train=False):
            out = super().phi(zz, layer, train)
            a = self.alpha(layer)[:, None, :]
            return torch.where(2.0 * a * zz > 0.5 * math.pi, out + 0.5 / a, out)

    m1 = LeakyMut("LRmut", 0.1); m1.init_state(2, dev, "m")
    m2 = prep(KKT1Mut("KKT1mut", "kkt1"))
    m3 = prep(KKAMut("KKAmut", "kk"))
    res["mutations"] = {
        "leaky_dphi_slope": float((auto(m1, z) - m1.dphi(z)).abs().max()),
        "kkt1_gate_band": float((auto(m2, z) - m2.dphi(z, 0)).abs().max()),
        "kka_right_shift_phi_jump": joins(m3, (0.5 * math.pi,))[0]["phi_jump"],
    }
    mut_caught = (res["mutations"]["leaky_dphi_slope"] > F64_TOL
                  and res["mutations"]["kkt1_gate_band"] > F64_TOL
                  and res["mutations"]["kka_right_shift_phi_jump"] > 1e-7)
    res["mutations"]["all_caught"] = mut_caught

    # ELU in float32: the gate is autograd's derivative on the run's device, bit for bit
    elu = E.ELU()
    res["elu_f32"] = {}
    for d in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        zz = torch.linspace(-120, 5, 1250001, device=d, dtype=torch.float32).requires_grad_(True)
        y = F.elu(zz, 1.0)
        (gr,) = torch.autograd.grad(y, zz, torch.ones_like(y))
        dd = elu.dphi(zz.detach())
        res["elu_f32"][d] = {"bit_equal": bool(torch.equal(gr, dd)),
                             "max_abs_diff": float((gr - dd).abs().max()),
                             "autograd_zero_above": float(zz.detach()[gr == 0].max()),
                             "elu_plus_1_zero_above": float(zz.detach()[(F.elu(zz.detach()) + 1) == 0].max())}
    run_dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok &= res["elu_f32"][run_dev]["bit_equal"]
    res["pass"] = bool(ok and mut_caught)
    return res


# --------------------------------------------------------------------------
def s_std(cifar) -> dict:
    """The L&C constants are the dataset's per-plane mean/std rounded to 4 digits, so after the
    transform |mean| <= 5e-5/0.2435 = 2.1e-4 and |std - 1| <= 2.1e-4 plus the constants' own
    rounding; bound 1e-3.  A plane-order error moves a mean by >= (0.4914-0.4822)/0.2470 = 0.037."""
    x = cifar.train_u8.to(torch.float64) / 255.0
    planes = x.view(-1, 3, 1024)
    raw_mean = planes.mean((0, 2)).tolist()
    xs = E.standardize(x.to(torch.float32)).to(torch.float64).view(-1, 3, 1024)
    std_mean = xs.mean((0, 2)).tolist()
    std_std = xs.std((0, 2)).tolist()
    res = {"raw_plane_mean": raw_mean, "constants_mean": E.STD_MEAN,
           "std_plane_mean": std_mean, "std_plane_std": std_std}
    ok = all(abs(a - b) <= 1e-3 for a, b in zip(raw_mean, E.STD_MEAN))
    ok &= all(abs(m) <= 1e-3 for m in std_mean) and all(abs(s - 1) <= 1e-3 for s in std_std)
    # mutation: plane order G,R,B
    mut = ((x.view(-1, 3, 1024)[:, [1, 0, 2]] - torch.tensor(E.STD_MEAN, dtype=torch.float64)[None, :, None])
           / torch.tensor(E.STD_STD, dtype=torch.float64)[None, :, None]).mean((0, 2)).tolist()
    res["mutation_plane_order_mean"] = mut
    res["mutation_caught"] = any(abs(m) > 1e-3 for m in mut)
    # raw and std inputs differ for the same seed
    xr = E.slot_inputs(cifar, 100, "raw", torch.device("cpu"))
    xd = E.slot_inputs(cifar, 100, "std", torch.device("cpu"))
    res["raw_vs_std_max_abs_diff"] = float((xr - xd).abs().max())
    res["pass"] = bool(ok and res["mutation_caught"] and res["raw_vs_std_max_abs_diff"] > 0.1)
    return res


# --------------------------------------------------------------------------
def host_act(arm, dev, V=None, r=None):
    if arm in ("LR", "R"):
        return H.ARMS[arm]
    if arm == "SNA":
        a = H.AdaptiveSnake(0.6, 0.01, dev, lo=0.005, hi=3.0, widths=(100, 100))
        if V is not None:
            a.V = [V[0][r].clone(), V[1][r].clone()]
        return a
    raise KeyError(arm)


def s_stack(dev, cifar) -> dict:
    """Streams and init: bit equality (they are the same generator calls).  Step-0 logits: bit
    equality (measured 2026-09-18 02:00).  Gradients: bmm's backward sums in another order; float32
    round-off is ~1e-7 relative per op (measured 1.6e-7), bound 1e-6.  V after one update: var over
    a different axis layout, same bound."""
    res, ok = {}, True
    dbg = {}
    with tempfile.TemporaryDirectory() as d:
        E.run("LR", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, debug=dbg, progress=quiet,
              snapshots=False)
    lab_ok = ord_ok = init_ok = True
    for r, s in enumerate(SEEDS):
        gl, gb = H.stream("rlc_labels", s), H.stream("rlc_batch", s)
        for t in range(2):
            lab_ok &= torch.equal(RC.task_labels(gl), dbg["labels"][t][r])
            for e in range(2):
                ord_ok &= torch.equal(torch.randperm(RC.N_IMAGES, generator=gb), dbg["orders"][2 * t + e][r])
        ip = H.init_params(s, torch.device("cpu"), RC.DIMS)
        init_ok &= all(torch.equal(ip[i].detach(), dbg["init"][i][r]) for i in range(6))
    power = not torch.equal(RC.task_labels(H.stream("rlc_labels", 101)), dbg["labels"][0][0])
    res["streams"] = {"labels": lab_ok, "orders": ord_ok, "init": init_ok, "power_slot0_vs_seed101": power}
    ok &= lab_ok and ord_ok and init_ok and power

    xs = [cifar.images(RC.subset_idx(s), dev) for s in SEEDS]
    ys = [RC.task_labels(H.stream("rlc_labels", s)).to(dev) for s in SEEDS]
    ords = [torch.randperm(RC.N_IMAGES, generator=H.stream("rlc_batch", s)).to(dev) for s in SEEDS]
    xb = torch.stack([xs[r][ords[r][:16]] for r in range(3)])
    yb = torch.stack([ys[r][ords[r][:16]] for r in range(3)])
    res["step0"] = {}
    for arm in ("LR", "R", "SNA"):
        act = E.make_act(arm); act.init_state(3, dev, "S-stack")
        P = [torch.stack([H.init_params(s, dev, RC.DIMS)[i].detach() for s in SEEDS]).requires_grad_(True)
             for i in range(6)]
        z1, a1, z2, a2, z3 = E.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, 10), yb.reshape(-1), reduction="none").view(3, 16).mean(1)
        G = torch.autograd.grad(lossv.sum(), P)
        rows = []
        for r, s in enumerate(SEEDS):
            hp = [q.detach().clone().requires_grad_(True) for q in H.init_params(s, dev, RC.DIMS)]
            ha = host_act(arm, dev)
            out = H.forward(hp, xb[r], ha)
            hg = torch.autograd.grad(F.cross_entropy(out[4], yb[r]), hp)
            row = {"seed": s, "logits_bit_equal": bool(torch.equal(out[4].detach(), z3[r].detach())),
                   "grad_rel": max(float((G[i][r] - hg[i]).abs().max() / hg[i].abs().max().clamp_min(1e-30))
                                   for i in range(6))}
            if arm == "SNA":
                ha.update(out[0].detach(), out[2].detach())
                row["_V"] = ha.V
            rows.append(row)
        if arm == "SNA":
            act.update(z1.detach(), z2.detach())
            for r, row in enumerate(rows):
                row["V_rel"] = max(float(((act.V[l][r] - row["_V"][l]).abs() / row["_V"][l]).max()) for l in (0, 1))
        for row in rows:
            row.pop("_V", None)
        power = not torch.equal(z3[0].detach(), H.forward(
            [q.detach() for q in H.init_params(101, dev, RC.DIMS)], xb[0], host_act(arm, dev))[4])
        res["step0"][arm] = {"rows": rows, "power_slot0_vs_seed101_params": power}
        ok &= all(r_["logits_bit_equal"] and r_["grad_rel"] <= 1e-6 and r_.get("V_rel", 0) <= 1e-6
                  for r_ in rows) and power
    res["pass"] = bool(ok)
    return res


# --------------------------------------------------------------------------
def s_eval(dev, cifar) -> dict:
    """Same weights, same inputs: the engine's per-slot metric expressions are the host's, so the
    columns agree to float32 round-off of the forward (bound 1e-6 relative, 1e-9 absolute near 0).
    Histogram counts can move only for values within the forward's max deviation of a bin edge:
    allowed L1 difference = 2 x that count (0 when the forward is bit-identical)."""
    X = torch.stack([cifar.images(RC.subset_idx(s), dev) for s in SEEDS])
    Y = torch.stack([RC.task_labels(H.stream("rlc_labels", s)) for s in SEEDS]).to(dev)
    P = [torch.stack([H.init_params(s, dev, RC.DIMS)[i].detach() for s in SEEDS]) for i in range(6)]
    P[0] = P[0] * 25.0            # push units into the tails: dead/zero/saturation columns get exercised
    res, ok = {"arms": {}}, True
    for arm in ("LR", "R", "SNA", "LK001", "LK03", "SL", "RSL", "ELU", "SILU", "GELU"):
        act = E.make_act(arm); act.init_state(3, dev, "S-eval")
        if act.adaptive:
            gv = torch.Generator().manual_seed(3)
            act.V = [torch.empty(3, 100).uniform_(1.0, 400.0, generator=gv).to(dev) for _ in range(2)]
        rows, hists, zs = E.evaluate(P, X, Y, act)
        with torch.no_grad():
            zz = E.forward(P, X, act)                      # the R=3 forward evaluate used
        worst, hist_ok, zbit = 0.0, True, True
        for r, s in enumerate(SEEDS):
            hp = [P[i][r].contiguous() for i in range(6)]
            ha = host_act(arm, dev, act.V if act.adaptive else None, r) if arm in ("LR", "R", "SNA") else act
            ref = RL.evaluate_rl(hp, X[r], Y[r], ha)
            ref_h = RC.preact_hist(hp, X[r], ha)
            for k, v in ref.items():
                err = abs(rows[r][k] - v) / max(abs(v), 1e-3)
                worst = max(worst, err)
            hz = H.forward(hp, X[r], ha)
            for k, zref in ((1, hz[0]), (2, hz[2])):
                zref = zref.detach().cpu().numpy()
                dev_max = float(np.abs(zref - zz[2 * k - 2][r].cpu().numpy()).max())
                zbit &= dev_max == 0.0
                v = zref.reshape(-1)
                near = int((np.abs(v - np.round(v / 0.1) * 0.1) <= dev_max).sum()) if dev_max > 0 else 0
                l1 = int(np.abs(ref_h[f"h{k}"].astype(np.int64) - hists[r][f"h{k}"]).sum())
                hist_ok &= l1 <= 2 * near and ref_h[f"oob{k}"] == hists[r][f"oob{k}"]
                hist_ok &= bool(np.allclose(ref_h[f"m{k}"], hists[r][f"m{k}"], rtol=1e-6, atol=1e-6))
                z16 = zref.astype(np.float16)
                hist_ok &= bool((np.abs(zs[k - 1][r].astype(np.float32) - z16.astype(np.float32))
                                 <= 2 * np.spacing(np.abs(z16)).astype(np.float32)).all())
        res["arms"][arm] = {"worst_rel": worst, "hist_ok": bool(hist_ok), "z_bit_equal_R1_vs_host": bool(zbit)}
        ok &= worst <= 1e-6 and hist_ok
    res["pass"] = bool(ok)
    return res


# --------------------------------------------------------------------------
def s_rsl_mode(dev, cifar) -> dict:
    act = E.RandSmoothLeaky(); act.init_state(2, dev, "S-rsl")
    z = torch.randn(2, 16, 100, device=dev, generator=torch.Generator(device=dev).manual_seed(5)) * 3
    r1, r2 = act.draw(z), act.draw(z)
    res = {"r_min": float(r1.min()), "r_max": float(r1.max()),
           "two_draws_differ": bool(not torch.equal(r1, r2)),
           "train_calls_differ": bool(not torch.equal(act.phi(z, train=True), act.phi(z, train=True)))}
    e1, e2 = act.phi(z, train=False), act.phi(z, train=False)
    res["eval_deterministic"] = bool(torch.equal(e1, e2))
    res["eval_equals_SL_at_mid"] = bool(torch.equal(e1, E.SmoothLeaky(a=(0.125 + 0.333) / 2).phi(z)))
    res["mutation_eval_with_train_differs"] = bool(not torch.equal(act.phi(z, train=True), act.phi(z, train=True)))
    rows = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            E.run("RSL", [100], ["raw"], 1, 1, dev, Path(d), cifar=cifar, progress=quiet, snapshots=False)
            rows.append(pd.read_csv(f"{d}/per_task.csv"))
    res["same_key_runs_identical"] = bool(rows[0].equals(rows[1]))
    res["pass"] = bool(res["r_min"] >= 0.125 and res["r_max"] <= 0.333 and res["two_draws_differ"]
                       and res["train_calls_differ"] and res["eval_deterministic"]
                       and res["eval_equals_SL_at_mid"] and res["mutation_eval_with_train_differs"]
                       and res["same_key_runs_identical"])
    return res


# --------------------------------------------------------------------------
def s_diverge(dev, cifar) -> dict:
    out = {}
    for tag, ns in (("clean", None), ("nan0", 0)):
        with tempfile.TemporaryDirectory() as d:
            prov = E.run("LR", SEEDS, ["raw"], 2, 1, dev, Path(d), cifar=cifar, progress=quiet,
                         nan_slot=ns, snapshots=False)
            out[tag] = (pd.read_csv(f"{d}/per_task.csv", float_precision="round_trip"), prov["divergences"])
    a, b = out["clean"][0], out["nan0"][0]
    cols = [c for c in a.columns if c not in ("slot",)]
    same = all(a[a.seed == s][cols].reset_index(drop=True).equals(b[b.seed == s][cols].reset_index(drop=True))
               for s in (101, 102))
    div = out["nan0"][1]
    b0 = b[b.seed == 100]
    res = {"others_bit_identical": bool(same), "divergences": div,
           "nan_slot_rows": int(len(b0)), "nan_slot_acc_is_nan": bool(b0["acc"].isna().all()),
           "power_101_vs_102_differ": bool(not a[a.seed == 101]["online_acc"].reset_index(drop=True).equals(
               a[a.seed == 102]["online_acc"].reset_index(drop=True)))}
    res["pass"] = bool(same and len(div) == 1 and div[0]["seed"] == 100 and div[0]["task"] == 1
                       and res["nan_slot_rows"] == 1 and res["nan_slot_acc_is_nan"]
                       and res["power_101_vs_102_differ"])
    return res


# --------------------------------------------------------------------------
def s_snap(dev, cifar) -> dict:
    """replay() is the run's own forward at R=1.  memo/zbar from the replayed z must equal the
    recorded columns (to the forward's R=20 vs R=1 round-off, bound 1e-6); the float16 z in the
    snapshot equals the replayed z cast to float16 up to that same round-off (|dz| <= 1e-6 * |z|
    moves a float16 value by at most one step, 0.01 at |z| <= 256)."""
    res, ok = {}, True
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        for arm in ("SNA", "KKA"):
            E.run(arm, [100, 101], ["raw", "std"], 2, 1, dev, out / arm, cifar=cifar, progress=quiet)
            rec = pd.read_csv(out / arm / "per_task.csv", float_precision="round_trip")
            r_ = {}
            for cond in ("raw", "std"):
                row = rec[(rec.seed == 100) & (rec.cond == cond) & (rec.task == 2)].iloc[0]
                z1, a1, z2, a2, logits, y = E.replay(out / arm, arm, cond, 100, 2, dev, cifar)
                snap = np.load(E.snapshot_path(out / arm, arm, cond, 100, 2))
                memo = float((logits.argmax(1) == y).float().mean())
                zbar = float(z1.mean(0).median())
                r_[cond] = {"memo_equal": memo == row["memo_acc"],
                            "zbar_rel": abs(zbar - row["zbar_l1"]) / max(abs(row["zbar_l1"]), 1e-6),
                            "z16_max_abs": float(np.abs(snap["z1"].astype(np.float32) - z1.cpu().numpy()).max()),
                            "snap_keys": sorted(snap.files)}
                zr32 = z1.cpu().numpy()
                r_[cond]["z16_within_2_steps"] = bool((np.abs(snap["z1"].astype(np.float32) - zr32)
                                                      <= 2 * np.spacing(np.abs(zr32).astype(np.float16)).astype(np.float32) + 1e-6 * np.abs(zr32)).all())
                ok &= r_[cond]["memo_equal"] and r_[cond]["zbar_rel"] <= 1e-6 and r_[cond]["z16_within_2_steps"]
            # mutation: replay the std snapshot on raw inputs
            snap_std = np.load(E.snapshot_path(out / arm, arm, "std", 100, 2))
            zr = E.replay(out / arm, arm, "raw", 100, 2, dev, cifar)[0]
            r_["mutation_wrong_cond_z_diff"] = float(np.abs(snap_std["z1"].astype(np.float32) - zr.cpu().numpy()).max())
            ok &= r_["mutation_wrong_cond_z_diff"] > 0.01
            res[arm] = r_
        res["t00_exists"] = E.snapshot_path(out / "SNA", "SNA", "raw", 100, 0).exists()
        ok &= res["t00_exists"]
    res["pass"] = bool(ok)
    return res


# --------------------------------------------------------------------------
def s_reuse(dev, cifar) -> dict:
    """The trajectory is chaotic (a 1e-7 change of W1 spreads to 2e-3 within 75 steps, the same as
    the engine-vs-host difference), so the probe row is one realization of the same run.  The engine
    draws 8 realizations (W1 x (1 + k 1e-7)); the probe must lie within 4 sd * sqrt(1 + 1/8) of their
    mean at every task (a 4-sigma prediction interval; with sd = 0 the probe must equal the value).
    Mutations must fall outside: SNA with beta = 0 (alpha frozen at c) and LR with slope 0 (R);
    LR with slope 0.3 is reported as descriptive power."""
    pert = [k * 1e-7 for k in range(8)]
    probe = {}
    for arm, sub in (("SNA", "snake"), ("LR", "relu")):
        f = PROBE / sub / "per_task.csv"
        if f.exists():
            d = pd.read_csv(f); d = d[d.arm == arm].sort_values("task")
            probe[arm] = {"online": d.online_acc.tolist(), "memo": d.memo_acc.tolist(), "source": str(f)}
        else:
            probe[arm] = {**PROBE_ROWS[arm], "source": "spec copy"}

    def realize(arm, n_tasks, beta=0.01):
        with tempfile.TemporaryDirectory() as d:
            E.run(arm, [100], ["raw"] * 8, n_tasks, 400, dev, Path(d), cifar=cifar, progress=print,
                  perturb=pert, snapshots=False, beta=beta)
            return pd.read_csv(f"{d}/per_task.csv")

    def inside(rec, pr, n_tasks):
        out = []
        for t in range(1, n_tasks + 1):
            for col, key in (("online_acc", "online"), ("memo_acc", "memo")):
                v = rec[rec.task == t][col].to_numpy()
                m, sd = float(v.mean()), float(v.std(ddof=1))
                p = pr[key][t - 1]
                half = 4 * sd * math.sqrt(1 + 1 / len(v))
                out.append({"task": t, "col": col, "probe": p, "mean": m, "sd": sd,
                            "inside": bool(abs(p - m) <= half) if sd > 0 else bool(p == m)})
        return out

    res = {"probe": probe}
    res["SNA"] = inside(realize("SNA", 3), probe["SNA"], 3)
    res["LR"] = inside(realize("LR", 3), probe["LR"], 3)
    res["mut_SNA_beta0"] = inside(realize("SNA", 1, beta=0.0), probe["SNA"], 1)
    res["mut_LR_as_R"] = inside(realize("R", 1), probe["LR"], 1)
    res["descriptive_LR_as_LK03"] = inside(realize("LK03", 1), probe["LR"], 1)   # power, not required
    ok = all(x["inside"] for x in res["SNA"] + res["LR"])
    caught = (not all(x["inside"] for x in res["mut_SNA_beta0"] if x["col"] == "online_acc")
              and not all(x["inside"] for x in res["mut_LR_as_R"] if x["col"] == "online_acc"))
    res["mutations_caught"] = caught
    res["pass"] = bool(ok and caught)
    return res


# --------------------------------------------------------------------------
def s_cost(dev, cifar, arm) -> dict:
    seeds = list(range(100, 110))
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time()
        prov = E.run(arm, seeds, ["raw", "std"], 1, 20, dev, Path(d), cifar=cifar, progress=quiet)
        wall = time.time() - t0
        snap_mb = sum(f.stat().st_size for f in Path(d).rglob("*.npz")) / 2 ** 20
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {"arm": arm, "R": 20, "ms_per_step": prov["step_ms_last_task"],
            "hours_per_process_50_tasks": prov["step_ms_last_task"] * 30000 * 50 / 3.6e6,
            "wall_s_1task_20ep": wall, "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "torch_max_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None,
            "nvidia_smi_self_mb": next((int(l.split(",")[1]) for l in gpu.splitlines()
                                        if l.split(",")[0].strip() == str(__import__("os").getpid())), None),
            "npz_mb_for_1_task_incl_t00": snap_mb, "pass": True}


# --------------------------------------------------------------------------
def collect() -> None:
    parts = {p.stem[len("part_"):]: json.loads(p.read_text()) for p in sorted(OUTDIR.glob("part_*.json"))}
    need = ("S-act", "S-std", "S-stack", "S-eval", "S-rsl-mode", "S-diverge", "S-snap", "S-reuse")
    res = {k: parts.get(k, {"pass": False, "missing": True}) for k in need}
    res["S-cost"] = {k[len("S-cost_"):]: v for k, v in parts.items() if k.startswith("S-cost_")}
    res["S-cost"]["pass"] = bool(res["S-cost"])
    res["git_hash"] = H.git_hash()
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("S-"))
    (OUTDIR / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    print("all_pass", res["all_pass"], {k: v["pass"] for k, v in res.items() if k.startswith("S-")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--arm", default="SNA")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        return collect()
    torch.set_num_threads(4)
    dev = H.setup(a.device)
    cifar = RC.Cifar10()
    for name in a.only.split(","):
        t0 = time.time()
        if name == "S-act":
            r = s_act()
        elif name == "S-std":
            r = s_std(cifar)
        elif name == "S-stack":
            r = s_stack(dev, cifar)
        elif name == "S-eval":
            r = s_eval(dev, cifar)
        elif name == "S-rsl-mode":
            r = s_rsl_mode(dev, cifar)
        elif name == "S-diverge":
            r = s_diverge(dev, cifar)
        elif name == "S-snap":
            r = s_snap(dev, cifar)
        elif name == "S-reuse":
            r = s_reuse(dev, cifar)
        elif name == "S-cost":
            r = s_cost(dev, cifar, a.arm)
            name = f"S-cost_{a.arm}"
        else:
            raise SystemExit(f"unknown check {name!r}")
        r["seconds"] = round(time.time() - t0, 1)
        part(name, r)


if __name__ == "__main__":
    main()
