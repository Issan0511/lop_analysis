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
S-snap     replaying the snapshots in the run's slot layout reproduces the recorded metrics and
           the stored float16 z exactly; the wrong condition does not.
S-graph    the captured-CUDA-graph step equals the eager step bit for bit, all 13 arms.
S-resume   a run stopped after task 1 and resumed equals the uninterrupted run bit for bit.
S-reuse    (400 epochs) the engine's realizations of seed 100 and the unmodified host's own
           realizations agree in mean at task 1, within 4 host sd; mutations (SNA with beta=0,
           LR with slope 0) do not.  The 0917 probe row is reported beside them.
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
    """Same weights, same inputs: the engine's per-slot metric expressions are the host's; only the
    forward differs (batched vs plain BLAS on the 1200-image batch), by at most D = max|z_eng - z_host|
    per layer, measured here.  Bounds that follow from D:
      zbar, zbar_min, zsd (means, medians, sd of values moved by <= D): |diff| <= D + 1e-6 |v|
      mob (mean of phi', Lipschitz L = sup|phi''|; 1.5 bounds every non-snake arm's smooth part and a
          kink moves the gate only for values within D of 0, a fraction <= that count / 1200):
          |diff| <= L D + (#|z| <= D)/1200 + 1e-6 |v|, L = 2 alpha_max for the snake family
      eff_rank (float64 from activations moved by <= L' D): relative 1e-6 + L' D / median|a|
      acc, dead_frac, zeroout, w_norm, alpha stats: exact unless a value sits within D of a threshold
          (reported; acc must be exact)
      histogram: counts move only for values within D of a bin edge: L1 <= 2 x that count
      per-unit means m: |diff| <= D + 1e-6 |m|;  float16 z: within 2 float16 steps + D."""
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
        L = 2.0 * float(act.alpha(0).max().clamp_min(act.alpha(1).max())) if act.adaptive else 1.5
        viol, worst_ratio, hist_ok, info = [], 0.0, True, {}
        for r, s in enumerate(SEEDS):
            hp = [P[i][r].contiguous() for i in range(6)]
            ha = host_act(arm, dev, act.V if act.adaptive else None, r) if arm in ("LR", "R", "SNA") else act
            ref = RL.evaluate_rl(hp, X[r], Y[r], ha)
            ref_h = RC.preact_hist(hp, X[r], ha)
            hz = H.forward(hp, X[r], ha)
            D, near0 = {}, {}
            for k, zref_t in ((1, hz[0]), (2, hz[2])):
                zref = zref_t.detach().cpu().numpy()
                D[k] = float(np.abs(zref - zz[2 * k - 2][r].cpu().numpy()).max())
                near0[k] = int((np.abs(zref) <= D[k]).sum())
                v = zref.reshape(-1)
                near = int((np.abs(v - np.round(v / 0.1) * 0.1) <= D[k]).sum())
                l1 = int(np.abs(ref_h[f"h{k}"].astype(np.int64) - hists[r][f"h{k}"]).sum())
                ok_h = l1 <= 2 * near and ref_h[f"oob{k}"] == hists[r][f"oob{k}"]
                ok_m = bool((np.abs(ref_h[f"m{k}"] - hists[r][f"m{k}"]) <= D[k] + 1e-6 * np.abs(ref_h[f"m{k}"])).all())
                z16 = zref.astype(np.float16)
                ok_z = bool((np.abs(zs[k - 1][r].astype(np.float32) - z16.astype(np.float32))
                             <= 2 * np.spacing(np.abs(z16)).astype(np.float32) + D[k]).all())
                hist_ok &= ok_h and ok_m and ok_z
                info[f"s{s}_l{k}"] = {"D": D[k], "hist_L1": l1, "hist_bound": 2 * near, "m_ok": ok_m, "z16_ok": ok_z}
            amed = {1: float(hz[1].abs().median()), 2: float(hz[3].abs().median())}
            for key, v in ref.items():
                diff = abs(rows[r][key] - v)
                lk = 2 if key.endswith("_l2") else 1
                if key.startswith(("zbar", "zsd")):
                    bound = D[lk] + 1e-6 * abs(v)
                elif key.startswith("mob"):
                    bound = L * D[lk] + near0[lk] / 1200 + 1e-6 * abs(v)
                elif key.startswith("eff_rank"):
                    bound = abs(v) * (1e-6 + L * D[lk] / max(amed[lk], 1e-12))
                elif key == "acc":
                    bound = 0.0
                else:                                   # dead/zeroout/w_norm/alpha: exact expected
                    bound = 0.0 if not key.startswith(("dead", "zeroout")) else near0[lk] / 100
                if diff > bound:
                    viol.append({"seed": s, "col": key, "host": v, "engine": rows[r][key], "bound": bound})
                if bound > 0:
                    worst_ratio = max(worst_ratio, diff / bound)
        res["arms"][arm] = {"violations": viol, "worst_diff_over_bound": worst_ratio,
                            "hist_ok": bool(hist_ok), "per_layer": info}
        ok &= not viol and hist_ok
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
    num = [c for c in cols if pd.api.types.is_numeric_dtype(a[c])]
    txt = [c for c in cols if c not in num]

    def same_rows(s):          # values, not dtypes: an all-zero column reads as int next to a NaN row's float
        x, y = a[a.seed == s], b[b.seed == s]
        return (len(x) == len(y)
                and np.array_equal(x[num].to_numpy(float), y[num].to_numpy(float), equal_nan=True)
                and (x[txt].to_numpy() == y[txt].to_numpy()).all())
    same = all(same_rows(s) for s in (101, 102))
    div = out["nan0"][1]
    b0 = b[b.seed == 100]
    res = {"others_bit_identical": bool(same), "divergences": div,
           "nan_slot_rows": int(len(b0)), "nan_slot_acc_is_nan": bool(b0["acc"].isna().all()),
           "power_101_vs_102_differ": bool(not np.array_equal(a[a.seed == 101][num].to_numpy(float),
                                                              a[a.seed == 102][num].to_numpy(float)))}
    res["pass"] = bool(same and len(div) == 1 and div[0]["seed"] == 100 and div[0]["task"] == 1
                       and res["nan_slot_rows"] == 1 and res["nan_slot_acc_is_nan"]
                       and res["power_101_vs_102_differ"])
    return res


# --------------------------------------------------------------------------
def s_snap(dev, cifar) -> dict:
    """replay_stack() with the run's own slot list is the forward evaluate ran, so every recorded
    column it re-derives (memo_acc, zbar_l1, zbar_l2, mob_l1 at 10 digits) and the snapshot's
    float16 z must match exactly.  The single-slot replay() differs by batched-BLAS round-off only:
    reported, and its memo must still match.  Mutation: the std snapshot on raw inputs."""
    res, ok = {}, True
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        for arm in ("SNA", "KKA", "ELU"):
            prov = E.run(arm, [100, 101], ["raw", "std"], 2, 1, dev, out / arm, cifar=cifar, progress=quiet)
            slots = [(q["seed"], q["cond"]) for q in prov["slots"]]
            rec = pd.read_csv(out / arm / "per_task.csv", float_precision="round_trip")
            z1, a1, z2, a2, logits, Y = E.replay_stack(out / arm, arm, slots, 2, dev, cifar)
            act = E.make_act(arm); act.init_state(len(slots), dev, "snap")
            if act.adaptive:
                ds = [np.load(E.snapshot_path(out / arm, arm, cd, s, 2)) for s, cd in slots]
                act.V = [torch.stack([torch.from_numpy(x[k]) for x in ds]).to(dev) for k in ("V1", "V2")]
            r_ = {}
            for r, (s, cd) in enumerate(slots):
                row = rec[(rec.slot == r) & (rec.task == 2)].iloc[0]
                snap = np.load(E.snapshot_path(out / arm, arm, cd, s, 2))
                g = lambda x: f"{float(x):.10g}"
                got = {"memo_acc": g((logits[r].argmax(1) == Y[r]).float().mean()),
                       "zbar_l1": g(z1[r].mean(0).median()), "zbar_l2": g(z2[r].mean(0).median()),
                       "mob_l1": g(act.dphi(z1, 0)[r].mean(0).median())}
                exact = all(got[k] == g(row[k]) for k in got)
                z16 = bool(np.array_equal(snap["z1"], z1[r].cpu().numpy().astype(np.float16))
                           and np.array_equal(snap["z2"], z2[r].cpu().numpy().astype(np.float16)))
                one = E.replay(out / arm, arm, cd, s, 2, dev, cifar)
                r_[f"{s}_{cd}"] = {"stack_exact": exact, "z16_exact": z16,
                                   "single_slot_memo_equal": g((one[4].argmax(1) == one[5]).float().mean()) == g(row["memo_acc"]),
                                   "single_slot_max_abs_dz1": float((one[0] - z1[r]).abs().max())}
                ok &= exact and z16 and r_[f"{s}_{cd}"]["single_slot_memo_equal"]
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
def s_graph(dev, cifar) -> dict:
    """The main run replays one captured CUDA graph per step; the other checks drive the same
    step eagerly.  Same kernels on the same tensors: rows (round-trip floats), histograms and
    the t02 snapshots must be identical for every arm (R=4, 2 tasks x 2 epochs).  The provenance
    engine string shows which path ran, so a pass cannot come from running eager twice."""
    res, ok = {}, True
    for arm in E.ARM_ORDER:
        out = {}
        for tag, g in (("eager", False), ("graph", True)):
            with tempfile.TemporaryDirectory() as d:
                prov = E.run(arm, [100, 101], ["raw", "std"], 2, 2, dev, Path(d), cifar=cifar,
                             progress=quiet, graph=g)
                out[tag] = (pd.read_csv(f"{d}/per_task.csv", float_precision="round_trip"),
                            {f.name: dict(np.load(f)) for f in (Path(d) / "hist").glob("*.npz")},
                            {f.parent.name: dict(np.load(f)) for f in Path(d).glob("snap/*/t02.npz")},
                            prov["engine"])
        a, b = out["eager"], out["graph"]
        same = (a[0].equals(b[0])
                and all(np.array_equal(a[1][f][k], b[1][f][k]) for f in a[1] for k in a[1][f])
                and all(np.array_equal(a[2][f][k], b[2][f][k]) for f in a[2] for k in a[2][f]))
        paths = ("eager" in a[3] and "CUDA graph" in b[3])
        res[arm] = {"identical": bool(same), "paths": [a[3], b[3]]}
        ok &= same and paths
    res["pass"] = bool(ok)
    return res


# --------------------------------------------------------------------------
def s_resume(dev, cifar) -> dict:
    """Run 3 tasks straight; run 1 task, stop, resume to 3.  Rows (round-trip floats), histogram
    files and the t03 snapshots must be identical, and provenance must show the resume happened
    (resumed_at_task == [2]) -- otherwise a pass could come from never taking the resume path.
    RSL exercises the cuda generator state, SNA the alpha statistics."""
    res, ok = {}, True
    for arm in ("RSL", "SNA"):
        with tempfile.TemporaryDirectory() as d:
            a_dir, b_dir = Path(d) / "a", Path(d) / "b"
            E.run(arm, [100, 101], ["raw", "std"], 3, 1, dev, a_dir, cifar=cifar, progress=quiet,
                  checkpoint=True, resume=True)
            E.run(arm, [100, 101], ["raw", "std"], 1, 1, dev, b_dir, cifar=cifar, progress=quiet,
                  checkpoint=True, resume=True)
            prov = E.run(arm, [100, 101], ["raw", "std"], 3, 1, dev, b_dir, cifar=cifar, progress=quiet,
                         checkpoint=True, resume=True)
            ra = pd.read_csv(a_dir / "per_task.csv", float_precision="round_trip")
            rb = pd.read_csv(b_dir / "per_task.csv", float_precision="round_trip")
            same_rows = ra.equals(rb)
            same_hist = all(
                all(np.array_equal(np.load(f)[k], np.load(b_dir / "hist" / f.name)[k]) for k in np.load(f).files)
                for f in sorted((a_dir / "hist").glob("*.npz")))
            same_snap = all(
                all(np.array_equal(np.load(f)[k], np.load(b_dir / f.relative_to(a_dir))[k]) for k in np.load(f).files)
                for f in sorted(a_dir.glob("snap/*/t03.npz")))
            res[arm] = {"rows_identical": bool(same_rows), "hist_identical": bool(same_hist),
                        "t03_snapshots_identical": bool(same_snap), "resumed_at_task": prov["resumed_at_task"],
                        "n_rows": int(len(rb))}
            ok &= same_rows and same_hist and same_snap and prov["resumed_at_task"] == [2] and len(rb) == 12
    res["pass"] = bool(ok)
    return res


# --------------------------------------------------------------------------
def s_reuse(dev, cifar) -> dict:
    """Does the stacked engine train the host's box?  The trajectory is chaotic (a 1e-7 change of
    W1 spreads to 2e-3 within 75 steps), so a run is one realization and the comparison has to be
    between distributions.  Measured 2026-09-18 03:30: the host's own realizations of LR task 1
    (cuda, W1 x (1 + k 1e-7), k = 0..4) have sd 0.0030, while eight realizations inside one stacked
    process have sd 0.0012 -- slots of one process stay correlated, so the host's spread is the
    one that calibrates the test:
        |mean_engine - mean_host| <= 4 sd_host sqrt(1/n_e + 1/n_h)   (equality if sd_host = 0)
    Mutations (SNA with beta = 0, LR with slope 0) must fall outside.  The 0917 probe row is
    reported next to both (descriptive: it was made on the cpu by another session)."""
    pert = [k * 1e-7 for k in range(8)]
    n_h = 4
    probe = {}
    for arm, sub in (("SNA", "snake"), ("LR", "relu")):
        f = PROBE / sub / "per_task.csv"
        if f.exists():
            d = pd.read_csv(f); d = d[d.arm == arm].sort_values("task")
            probe[arm] = {"online": d.online_acc.tolist(), "memo": d.memo_acc.tolist(), "source": str(f)}
        else:
            probe[arm] = {**PROBE_ROWS[arm], "source": "spec copy"}

    def engine(arm, n_tasks, beta=0.01):
        with tempfile.TemporaryDirectory() as d:
            E.run(arm, [100], ["raw"] * 8, n_tasks, 400, dev, Path(d), cifar=cifar,
                  progress=lambda m: print(m, flush=True), perturb=pert, snapshots=False, beta=beta)
            return pd.read_csv(f"{d}/per_task.csv")

    def host(arm):
        """The unmodified host, same device, W1 x (1 + k 1e-7): its own realizations of task 1."""
        out = []
        orig = H.init_params
        for k in range(n_h):
            def pert_init(seed, device, dims=None, k=k):
                ps = orig(seed, device, dims)
                with torch.no_grad():
                    ps[0].mul_(1 + k * 1e-7)
                return ps
            H.init_params = pert_init
            try:
                rows, _ = RC.run_one(arm, 100, 1e-3, 1, cifar, dev, epochs=400)
            finally:
                H.init_params = orig
            out.append({"online_acc": rows[0]["online_acc"], "memo_acc": rows[0]["memo_acc"]})
            print(f"host {arm} k={k} online {out[-1]['online_acc']:.4f}", flush=True)
        return pd.DataFrame(out)

    def compare(rec, hrec, task=1):
        out = []
        for col in ("online_acc", "memo_acc"):
            e = rec[rec.task == task][col].to_numpy()
            h = hrec[col].to_numpy()
            me, mh = float(e.mean()), float(h.mean())
            sdh, sde = float(h.std(ddof=1)), float(e.std(ddof=1))
            half = 4 * sdh * math.sqrt(1 / len(e) + 1 / len(h))
            out.append({"col": col, "task": task, "engine_mean": me, "engine_sd": sde,
                        "host_mean": mh, "host_sd": sdh, "half_width": half,
                        "inside": bool(abs(me - mh) <= half) if sdh > 0 else bool(me == mh)})
        return out

    res, ok = {"probe": probe, "n_host": n_h, "n_engine": 8}, True
    for arm in ("SNA", "LR"):
        hrec = host(arm)
        rec = engine(arm, 3)
        res[arm] = compare(rec, hrec)
        res[f"{arm}_descriptive"] = {
            "host_task1_online": hrec.online_acc.tolist(),
            "engine_online_by_task": {t: rec[rec.task == t].online_acc.tolist() for t in (1, 2, 3)},
            "probe_online": probe[arm]["online"], "probe_source": probe[arm]["source"]}
        ok &= all(x["inside"] for x in res[arm])
        res[f"mut_{arm}"] = compare(engine("SNA", 1, beta=0.0) if arm == "SNA" else engine("R", 1), hrec)
    caught = all(not x["inside"] for x in (res["mut_SNA"] + res["mut_LR"]) if x["col"] == "online_acc")
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
    need = ("S-act", "S-std", "S-stack", "S-eval", "S-rsl-mode", "S-diverge", "S-snap", "S-graph",
            "S-resume", "S-reuse")
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
        elif name == "S-graph":
            r = s_graph(dev, cifar)
        elif name == "S-resume":
            r = s_resume(dev, cifar)
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
