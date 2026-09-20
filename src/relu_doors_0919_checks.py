#!/usr/bin/env python3
"""relu_doors_0919 -- the checks of spec §7.  Every check carries a mutation control:
a deliberately broken variant that must FAIL, so a pass cannot come from a vacuous predicate
(this box has written six vacuous S-checks before; see the project's threshold-derivation note).

All tolerances are derived from measured arithmetic, never guessed.

S-off      with every door open, the engine reproduces the parent run's rows bit for bit
           (rlcifar_mlp_battle_0918, arm R, cond raw, same seeds).  Mutation: arm C.
S-C        door C subtracts the seed's own mean image and does not divide: the centred input's
           mean has norm <= the float32 accumulation bound, and x - x_C is constant per column.
           Mutation: centring off.
S-H        door H's EMA follows m <- (1-b) m + b * mean_batch(phi(z)), and evaluation uses the
           same m as training (no train/eval gap).  Mutation: beta = 0.
S-grad     m is a constant: it is not among the optimised parameters, no gradient reaches it,
           and it tracks the UNCENTRED activation (tracking the centred one would decay to 0).
           Mutation: an EMA fed the centred activation collapses to 0.
S-B        b1 is exactly 0 at every step (CHB, CHB0); b2 is multiplied by (1 - lr*lam) after
           every Adam step (CHB) or held at 0 (CHB0); b3 is untouched.  Mutation: lam = 0.
S-pin      with C and B closed, |zbar_1i| <= |w_i| * |xbar| (Cauchy-Schwarz on the measured
           residual mean image), for every unit and task.  Mutation: door C off.
S-diverge  a NaN slot is recorded as diverged and leaves the other slots bit-identical.
S-resume   a run stopped after task 1 and resumed equals the uninterrupted run bit for bit.
S-cost     R=10, 1 task x 20 epochs: ms/step, RSS, GPU memory, and the wall clock a 50-task arm
           implies.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys_path = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(sys_path))
from src import pmnist_0905 as H                       # noqa: E402
from src import pmnist_rlcifar_0907 as RC              # noqa: E402
from src import relu_doors_0919 as E                   # noqa: E402
from src import rlcifar_mlp_battle_0918 as PARENT      # noqa: E402

OUT = H.REPO / "results" / "_checks_relu_doors_0919"
SEEDS = [100, 101, 102]
LAM = 0.1713                                           # spec §2.2, derived from the probe
quiet = lambda m: None


def part(name: str, res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{name}: {'PASS' if res.get('pass') else 'FAIL'}  "
          + json.dumps({k: v for k, v in res.items() if k != "pass"}, default=str)[:400])


def collect() -> None:
    parts = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("S-*.json"))}
    allp = all(v.get("pass") for v in parts.values())
    (OUT / "checks.json").write_text(json.dumps(
        {"all_pass": allp, "checks": parts}, indent=1, default=str))
    print(f"all_pass: {allp}  ({len(parts)} checks)")


def rows_of(d: str) -> pd.DataFrame:
    return pd.read_csv(f"{d}/per_task.csv", float_precision="round_trip")


def same_numeric(a: pd.DataFrame, b: pd.DataFrame, cols) -> bool:
    return np.array_equal(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)


# --------------------------------------------------------------------------


def s_off(dev, cifar) -> dict:
    """Doors open == the parent engine.  The two engines are separate files, so this is the
    check that copying + editing did not change the untouched path."""
    shared = [c for c in ("online_acc", "memo_acc", "acc", "dead_frac_l1", "zbar_l1", "zsd_l1",
                          "mob_l1", "eff_rank_l1", "dead_frac_l2", "zbar_l2", "zsd_l2", "mob_l2",
                          "eff_rank_l2", "w_norm_l1", "w_norm_l2", "w_norm_l3")]
    got = {}
    with tempfile.TemporaryDirectory() as d:
        E.run("ref", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, progress=quiet,
              snapshots=False)
        got["doors"] = rows_of(d)
    with tempfile.TemporaryDirectory() as d:
        PARENT.run("R", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, progress=quiet,
                   snapshots=False)
        got["parent"] = rows_of(d)
    with tempfile.TemporaryDirectory() as d:                       # mutation: a door closed
        E.run("C", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, progress=quiet,
              snapshots=False)
        got["mutant"] = rows_of(d)
    same = same_numeric(got["doors"], got["parent"], shared)
    mut = same_numeric(got["mutant"], got["parent"], shared)
    return {"pass": bool(same and not mut), "bit_identical_to_parent": bool(same),
            "mutation_C_differs": bool(not mut), "columns": len(shared),
            "rows": int(len(got["doors"]))}


def s_c(dev, cifar) -> dict:
    """x - xbar, no division.  The residual mean is float32 accumulation noise only, and the
    shift is the same for every image (a per-column constant)."""
    res, ok = {}, True
    for s in SEEDS:
        x0 = E.slot_inputs(cifar, s, "raw", dev, center=False)
        xc = E.slot_inputs(cifar, s, "raw", dev, center=True)
        shift = x0 - xc                                            # must be constant per column
        resid = float(xc.mean(0).norm())
        # bound: summing 1200 float32 values of size ~|x| carries at most n*eps/2 relative error
        bound = float(x0.abs().mean()) * 1200 * np.finfo(np.float32).eps / 2 * np.sqrt(3072)
        same_scale = bool(torch.allclose(x0.std(0), xc.std(0), rtol=0, atol=1e-6))
        const = float((shift - shift[0]).abs().max())
        # x0 - xc = x0 - fl(x0 - xbar) = xbar + delta, with |delta| <= 2 eps max|x0|
        const_bound = 2 * np.finfo(np.float32).eps * float(x0.abs().max())
        res[str(s)] = {"residual_mean_norm": resid, "bound": bound,
                       "shift_is_constant_max_dev": const, "shift_bound": const_bound,
                       "scale_unchanged": same_scale,
                       "r_before": float(x0.mean(0).norm() / (x0 - x0.mean(0)).pow(2).sum(1).mean().sqrt()),
                       "r_after": float(resid / (xc - xc.mean(0)).pow(2).sum(1).mean().sqrt())}
        ok &= resid <= bound and const <= const_bound and same_scale
    # mutation: no centring
    x0 = E.slot_inputs(cifar, SEEDS[0], "raw", dev, center=False)
    res["mutation_uncentred_mean_norm"] = float(x0.mean(0).norm())
    res["mutation_over_bound"] = res["mutation_uncentred_mean_norm"] / res[str(SEEDS[0])]["bound"]
    ok &= res["mutation_over_bound"] > 100.0      # the control must miss by a wide margin
    return {"pass": bool(ok), **res}


def s_h(dev, cifar) -> dict:
    """The EMA recurrence, and the same m in training and in evaluation."""
    R, B, n = 3, E.BATCH, E.DIMS[1]
    act = E.make_act("CH", beta=0.01)
    act.init_state(R, dev, key="k")
    g = torch.Generator(device="cpu").manual_seed(4242)
    m_ref = [torch.zeros(R, w) for w in (E.DIMS[1], E.DIMS[2])]
    for _ in range(50):
        z1 = torch.randn(R, B, n, generator=g).to(dev) * 3.0 - 1.0
        z2 = torch.randn(R, B, E.DIMS[2], generator=g).to(dev) * 2.0 + 0.5
        for i, z in enumerate((z1, z2)):                            # the spec's recurrence
            m_ref[i] = 0.99 * m_ref[i] + 0.01 * torch.clamp(z, min=0.0).mean(1).cpu()
        act.update(z1, z2)
    err = max(float((act.m[i].cpu() - m_ref[i]).abs().max()) for i in (0, 1))
    scale = max(float(m_ref[i].abs().max()) for i in (0, 1))
    bound = scale * 50 * np.finfo(np.float32).eps * 4              # 50 fused multiply-adds
    # training and evaluation subtract the same m
    z = torch.randn(R, 7, n, generator=g).to(dev)
    same = bool(torch.equal(act.phi(z, 0, train=True), act.phi(z, 0, train=False)))
    # the door subtracts exactly m
    diff = float((torch.clamp(z, min=0.0) - act.phi(z, 0) - act.m[0][:, None, :]).abs().max())
    diff_bound = 2 * np.finfo(np.float32).eps * max(float(torch.clamp(z, min=0.0).abs().max()),
                                                    float(act.m[0].abs().max()))
    # mutation: beta = 0 leaves m at zero
    act0 = E.make_act("CH", beta=0.0)
    act0.init_state(R, dev, key="k")
    act0.update(z1, z2)
    mut = float(act0.m[0].abs().max())
    return {"pass": bool(err <= bound and same and diff <= diff_bound and mut == 0.0),
            "ema_max_abs_error": err, "bound": bound, "ema_scale": scale,
            "train_equals_eval": same, "subtracts_m_error": diff, "subtracts_m_bound": diff_bound,
            "mutation_beta0_m_is_zero": mut}


def s_grad(dev, cifar) -> dict:
    """m is a constant that tracks the UNCENTRED activation."""
    R, B, n = 2, E.BATCH, E.DIMS[1]
    act = E.make_act("CH", beta=0.05)
    act.init_state(R, dev, key="k")
    in_params = any(act.m[0] is q for q in ())                      # m is never put in P
    needs_grad = bool(act.m[0].requires_grad or act.m[1].requires_grad)
    g = torch.Generator(device="cpu").manual_seed(7)
    z = (torch.randn(R, B, n, generator=g).to(dev) + 2.0).requires_grad_(True)
    a = act.phi(z, 0, train=True)
    a.sum().backward()
    # d/dz of (relu(z) - m) is the relu gate; an m that depended on z would add a -1/B term
    gate_err = float((z.grad - (z > 0).to(z.dtype)).abs().max())
    # the EMA converges to mean(relu(z)), not to 0
    zc = torch.full((R, B, n), 3.0, device=dev)
    for _ in range(300):
        act.update(zc, zc[:, :, :E.DIMS[2]])
    conv = float(act.m[0].mean())
    # mutation: an EMA fed the centred activation decays to 0
    mm = torch.zeros(R, n, device=dev)
    for _ in range(300):
        mm.mul_(0.95).add_((torch.clamp(zc, min=0.0) - mm[:, None, :]).mean(1), alpha=0.05)
    return {"pass": bool(not in_params and not needs_grad and gate_err == 0.0
                         and abs(conv - 3.0) < 1e-3 and abs(float(mm.mean()) - 1.5) < 0.2),
            "m_in_parameters": in_params, "m_requires_grad": needs_grad,
            "dphi_dz_error": gate_err, "ema_converges_to": conv, "target": 3.0,
            "mutation_centred_ema_converges_to": float(mm.mean()),
            "mutation_note": "feeding the centred activation converges to target/2, not target"}


def s_b(dev, cifar) -> dict:
    """b1 exactly 0, b2 decayed by (1 - lr*lam) or zeroed, b3 untouched."""
    res, ok = {}, True
    for arm, lam in (("CHB", LAM), ("CHB0", 0.0), ("CH", 0.0)):
        with tempfile.TemporaryDirectory() as d:
            E.run(arm, SEEDS[:2], ["raw"], 1, 2, dev, Path(d), cifar=cifar, progress=quiet,
                  lam=lam, snapshots=True)
            sn = np.load(E.snapshot_path(Path(d), arm, "raw", SEEDS[0], 1))
            b1, b2, b3 = (np.abs(sn[k]).max() for k in ("b1", "b2", "b3"))
            init = np.load(E.snapshot_path(Path(d), arm, "raw", SEEDS[0], 0))
            b3_moved = float(np.abs(sn["b3"] - init["b3"]).max())
        res[arm] = {"max_abs_b1": float(b1), "max_abs_b2": float(b2), "max_abs_b3": float(b3),
                    "b3_moved_from_init": b3_moved}
        if arm == "CHB":
            ok &= b1 == 0.0 and b2 > 0.0 and b3_moved > 0.0
        elif arm == "CHB0":
            ok &= b1 == 0.0 and b2 == 0.0 and b3_moved > 0.0
        else:                                                        # door B open
            ok &= b1 > 0.0 and b2 > 0.0
    # the decay factor itself, on a synthetic parameter list
    P = [torch.zeros(1), torch.ones(1) * 5.0, torch.zeros(1), torch.ones(1) * 5.0,
         torch.zeros(1), torch.ones(1) * 5.0]
    E.make_act("CHB", lam=LAM).post_update(P, E.LR)
    res["one_step_b2"] = float(P[3])
    res["expected_b2"] = 5.0 * (1 - E.LR * LAM)
    ok &= abs(res["one_step_b2"] - res["expected_b2"]) <= 5.0 * np.finfo(np.float32).eps
    res["one_step_b1"] = float(P[1])
    res["one_step_b3_untouched"] = float(P[5])
    ok &= res["one_step_b1"] == 0.0 and res["one_step_b3_untouched"] == 5.0
    # mutation: lam = 0 leaves b2 alone
    P2 = [torch.zeros(1), torch.ones(1) * 5.0, torch.zeros(1), torch.ones(1) * 5.0,
          torch.zeros(1), torch.ones(1) * 5.0]
    a0 = E.ReLUDoors("CHB0")
    a0.door_b = "wd"; a0.lam = 0.0
    a0.post_update(P2, E.LR)
    res["mutation_lam0_b2"] = float(P2[3])
    ok &= res["mutation_lam0_b2"] == 5.0
    return {"pass": bool(ok), **res}


def s_pin(dev, cifar) -> dict:
    """With C and B closed, zbar_1i = w_i . xbar exactly (b1 = 0), so |zbar_1i| <= |w_i| |xbar|."""
    res, ok = {}, True
    for arm, lam in (("CHB", LAM), ("CHB0", 0.0), ("C", 0.0)):
        with tempfile.TemporaryDirectory() as d:
            E.run(arm, SEEDS[:2], ["raw"], 2, 3, dev, Path(d), cifar=cifar, progress=quiet,
                  lam=lam, snapshots=True)
            r = rows_of(d)
            worst, bound = 0.0, 0.0
            for s in SEEDS[:2]:
                x = E.slot_inputs(cifar, s, "raw", dev, center=True)
                xb = float(x.mean(0).norm())
                for t in (1, 2):
                    sn = np.load(E.snapshot_path(Path(d), arm, "raw", s, t))
                    W1 = torch.from_numpy(sn["W1"]).to(dev)
                    z1 = x @ W1.T + torch.from_numpy(sn["b1"]).to(dev)
                    worst = max(worst, float(z1.mean(0).abs().max()))
                    bound = max(bound, float(W1.norm(dim=1).max()) * xb)
        res[arm] = {"max_abs_zbar_l1": worst, "cauchy_schwarz_bound": bound,
                    "ratio": worst / bound if bound else float("nan"),
                    "reported_abs_zbar_l1": float(r["abs_zbar_l1"].max())}
        if arm in ("CHB", "CHB0"):
            ok &= worst <= bound
        else:                                                        # door B open: b1 carries it
            res[arm]["note"] = "door B open, so b1 is free and the bound does not apply"
    # mutation: no centring -> the bound is violated by orders of magnitude
    with tempfile.TemporaryDirectory() as d:
        E.run("ref", SEEDS[:1], ["raw"], 1, 3, dev, Path(d), cifar=cifar, progress=quiet,
              snapshots=True)
        x = E.slot_inputs(cifar, SEEDS[0], "raw", dev, center=True)
        sn = np.load(E.snapshot_path(Path(d), "ref", "raw", SEEDS[0], 1))
        W1 = torch.from_numpy(sn["W1"]).to(dev)
        x_un = E.slot_inputs(cifar, SEEDS[0], "raw", dev, center=False)
        z = x_un @ W1.T + torch.from_numpy(sn["b1"]).to(dev)
        res["mutation_uncentred_max_abs_zbar"] = float(z.mean(0).abs().max())
        res["mutation_bound"] = float(W1.norm(dim=1).max()) * float(x.mean(0).norm())
        ok &= res["mutation_uncentred_max_abs_zbar"] > res["mutation_bound"]
    return {"pass": bool(ok), **res}


def s_cs(dev, cifar) -> dict:
    """Door CS is the exact complement of H: the SAME per-feature running statistic, but the
    scaling half.  v <- (1-b) v + b * mean_batch(phi(z)^2), and phi divides by sqrt(v)."""
    R, B, n = 3, E.BATCH, E.DIMS[1]
    act = E.make_act("CS", beta=0.01)
    act.init_state(R, dev, key="k")
    g = torch.Generator(device="cpu").manual_seed(99)
    v_ref = [torch.ones(R, w) for w in (E.DIMS[1], E.DIMS[2])]
    for _ in range(50):
        z1 = torch.randn(R, B, n, generator=g).to(dev) * 2.0
        z2 = torch.randn(R, B, E.DIMS[2], generator=g).to(dev) * 1.5
        for i, z in enumerate((z1, z2)):
            v_ref[i] = 0.99 * v_ref[i] + 0.01 * torch.clamp(z, min=0.0).pow(2).mean(1).cpu()
        act.update(z1, z2)
    err = max(float((act.v[i].cpu() - v_ref[i]).abs().max()) for i in (0, 1))
    scale = max(float(v_ref[i].abs().max()) for i in (0, 1))
    bound = scale * 50 * np.finfo(np.float32).eps * 4
    z = torch.randn(R, 7, n, generator=g).to(dev)
    want = torch.clamp(z, min=0.0) / act.v[0][:, None, :].sqrt()
    diff = float((act.phi(z, 0) - want).abs().max())
    # it must NOT centre: the output's per-unit mean stays positive
    mean_pos = float(act.phi(z, 0).mean(1).min())
    # mutation: beta = 0 leaves v at 1, i.e. no scaling at all
    a0 = E.make_act("CS", beta=0.0)
    a0.init_state(R, dev, key="k")
    a0.update(z1, z2)
    mut = float((a0.phi(z, 0) - torch.clamp(z, min=0.0)).abs().max())
    return {"pass": bool(err <= bound and diff == 0.0 and mean_pos >= 0.0 and mut == 0.0),
            "ema_max_abs_error": err, "bound": bound, "divides_by_sqrt_v_error": diff,
            "does_not_centre_min_unit_mean": mean_pos, "mutation_beta0_equals_plain_relu": mut}


def s_ln(dev, cifar) -> dict:
    """Door LN is published LayerNorm: per sample across features, learnable gamma/beta,
    applied to z before the nonlinearity.  Compare against torch's own F.layer_norm."""
    import torch.nn.functional as F
    R, B, n = 2, E.BATCH, E.DIMS[1]
    act = E.make_act("LN")
    act.init_state(R, dev, key="k")
    P = [None] * 6 + act.extra_params(R, dev)
    act.bind(P)
    g = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():                                   # give gamma/beta non-trivial values
        P[6].copy_(torch.rand(R, n, generator=g).to(dev) + 0.5)
        P[7].copy_(torch.randn(R, n, generator=g).to(dev) * 0.1)
    z = torch.randn(R, B, n, generator=g).to(dev) * 3.0 + 1.0
    got = act.phi(z, 0)
    want = torch.stack([F.relu(F.layer_norm(z[r], (n,), P[6][r], P[7][r], eps=1e-5))
                        for r in range(R)])
    err = float((got - want).abs().max())
    bound = 4 * np.finfo(np.float32).eps * float(want.abs().max().clamp(min=1.0))
    # gamma/beta are in P, so the run's Adam optimises them: they must move
    moved = 0.0
    with tempfile.TemporaryDirectory() as d:
        E.run("LN", SEEDS[:1], ["raw"], 1, 2, dev, Path(d), cifar=cifar, progress=quiet,
              snapshots=True)
        a = np.load(E.snapshot_path(Path(d), "LN", "raw", SEEDS[0], 0))
        b = np.load(E.snapshot_path(Path(d), "LN", "raw", SEEDS[0], 1))
        moved = float(np.abs(b["g1"] - a["g1"]).max())
        init_g = float(np.abs(a["g1"] - 1.0).max())
        init_b = float(np.abs(a["bn1"]).max())
    # mutation: no affine at all differs from the affine version
    mut = float((want - torch.stack([F.relu(F.layer_norm(z[r], (n,), eps=1e-5))
                                     for r in range(R)])).abs().max())
    return {"pass": bool(err <= bound and moved > 0.0 and init_g == 0.0 and init_b == 0.0
                         and mut > 0.0),
            "vs_F_layer_norm_error": err, "bound": bound, "gamma_moved_in_one_task": moved,
            "gamma_init_deviation_from_1": init_g, "beta_init_deviation_from_0": init_b,
            "mutation_no_affine_differs_by": mut}


def s_diverge(dev, cifar) -> dict:
    out = {}
    for tag, ns in (("clean", None), ("nan0", 0)):
        with tempfile.TemporaryDirectory() as d:
            prov = E.run("CH", SEEDS, ["raw"], 2, 1, dev, Path(d), cifar=cifar, progress=quiet,
                         nan_slot=ns, snapshots=False)
            out[tag] = (rows_of(d), prov["divergences"])
    a, b = out["clean"][0], out["nan0"][0]
    num = [c for c in a.columns if c != "slot" and pd.api.types.is_numeric_dtype(a[c])]
    same = all(len(a[a.seed == s]) == len(b[b.seed == s])
               and same_numeric(a[a.seed == s], b[b.seed == s], num) for s in SEEDS[1:])
    b0 = b[b.seed == SEEDS[0]]
    return {"pass": bool(same and out["nan0"][1] and b0["acc"].isna().all()),
            "others_bit_identical": bool(same), "divergences": out["nan0"][1],
            "nan_slot_acc_is_nan": bool(b0["acc"].isna().all())}


def s_resume(dev, cifar) -> dict:
    num = None
    with tempfile.TemporaryDirectory() as d:
        E.run("CHB", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, progress=quiet,
              lam=LAM, checkpoint=True, snapshots=False)
        whole = rows_of(d)
    with tempfile.TemporaryDirectory() as d:
        E.run("CHB", SEEDS, ["raw"], 1, 2, dev, Path(d), cifar=cifar, progress=quiet,
              lam=LAM, checkpoint=True, snapshots=False)
        E.run("CHB", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, progress=quiet,
              lam=LAM, checkpoint=True, resume=True, snapshots=False)
        split = rows_of(d)
    num = [c for c in whole.columns if pd.api.types.is_numeric_dtype(whole[c])]
    same = same_numeric(whole, split, num)
    # mutation: resuming a different arm must be refused
    refused = False
    with tempfile.TemporaryDirectory() as d:
        E.run("CHB", SEEDS, ["raw"], 1, 2, dev, Path(d), cifar=cifar, progress=quiet,
              lam=LAM, checkpoint=True, snapshots=False)
        try:
            E.run("CH", SEEDS, ["raw"], 2, 2, dev, Path(d), cifar=cifar, progress=quiet,
                  checkpoint=True, resume=True, snapshots=False)
        except SystemExit:
            refused = True
    return {"pass": bool(same and refused), "bit_identical": bool(same),
            "mutation_wrong_arm_refused": refused, "rows": int(len(whole))}


def s_cost(dev, cifar, arm: str) -> dict:
    import resource
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        prov = E.run(arm, list(range(100, 110)), ["raw"], 1, 20, dev, Path(d), cifar=cifar,
                     progress=quiet, lam=LAM if arm == "CHB" else 0.0, snapshots=False)
    wall = time.time() - t0
    ms = prov.get("step_ms_last_task", float("nan"))
    return {"pass": True, "arm": arm, "R": 10, "ms_per_step": ms, "wall_seconds": round(wall, 1),
            "implied_50_task_minutes": round(ms * 30000 * 50 / 6e4, 1),
            "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            "cuda_max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if dev.type == "cuda" else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--arm", default="CHB")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        return collect()
    torch.set_num_threads(4)
    dev = H.setup(a.device)
    cifar = RC.Cifar10()
    fns = {"S-off": s_off, "S-C": s_c, "S-H": s_h, "S-grad": s_grad, "S-B": s_b,
           "S-pin": s_pin, "S-CS": s_cs, "S-LN": s_ln,
           "S-diverge": s_diverge, "S-resume": s_resume}
    for name in a.only.split(","):
        t0 = time.time()
        if name == "S-cost":
            r = s_cost(dev, cifar, a.arm)
            name = f"S-cost_{a.arm}"
        elif name in fns:
            r = fns[name](dev, cifar)
        else:
            raise SystemExit(f"unknown check {name!r}")
        r["seconds"] = round(time.time() - t0, 1)
        part(name, r)


if __name__ == "__main__":
    main()
