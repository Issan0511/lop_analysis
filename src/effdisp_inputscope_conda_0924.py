"""CondA input/teacher context factorial, preregistered in effdisp_inputscope_0924.

The latent SCR input is shared across all 16 arms. X controls the learner's
persistent-bit context and Y controls which context the fixed LTU teacher sees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import effdisp_conda_0924 as E

M, R, F, H = 20, 5, 15, 100
SEEDS = tuple(range(200, 205))
ARMS = tuple((f"{label}_k{k}_X{x}Y{y}", k, "adam", 1e-3, 1e-8, R, act)
             for label, act in (("LR", "leaky"), ("SNA06", "sna06"))
             for k in (1, 7) for x in (0, 1) for y in (0, 1))


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def arm_design(arm):
    name, k, _, _, _, r, act = arm
    tail = name.rsplit("_", 1)[-1]
    return {"name": name, "k": k, "m": M, "r": r, "f": M-r,
            "activation": act, "X": int(tail[1]), "Y": int(tail[3]),
            "optimizer": "Adam", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8}


def context_inputs(flip, initial_bits, random_bits, x_flags, y_flags):
    """Return paired learner and teacher inputs from one latent random draw."""
    raw = torch.cat((flip[:, :F], random_bits), dim=1)
    fixed = torch.cat((initial_bits[:, :F], random_bits), dim=1)
    learner = torch.where(x_flags[:, None], raw, fixed)
    teacher = torch.where(y_flags[:, None], raw, fixed)
    return learner, teacher


def support_readout(P, target, flip, initial_bits, arms, seeds, adapt_ema):
    """Exact 32-point conditional MSE and teacher targets for each arm/seed."""
    dev = flip.device
    n = len(seeds)
    bits = ((torch.arange(32, device=dev)[:, None] >>
             torch.arange(R, device=dev)) & 1).float()
    mse, y32, dz_zero, dz_abs, active = [], [], [], [], []
    for j, arm in enumerate(arms):
        sl = slice(j*n, (j+1)*n)
        design = arm_design(arm)
        fp = flip[sl, :F] if design["X"] else initial_bits[sl, :F]
        tp = flip[sl, :F] if design["Y"] else initial_bits[sl, :F]
        xi = torch.cat((fp[None].expand(32, -1, -1), bits[:, None].expand(-1, n, -1)), -1)
        xt = torch.cat((tp[None].expand(32, -1, -1), bits[:, None].expand(-1, n, -1)), -1)
        yt = torch.einsum("nhm,snm->snh", target["W"][sl], xt) + target["b"][sl][None]
        yt = ((yt >= target["tau"][sl][None]).float() * target["v"][sl][None]).sum(-1) + target["cout"][sl][None]
        pre = torch.einsum("nhm,snm->snh", P["W"][sl], xi) + P["b"][sl][None]
        alpha = E.alpha_from_ema(adapt_ema[sl], arm[6])[None] if arm[6] in E.SNA_C else None
        a, derivative = E._activation(pre, arm[6], alpha)
        yp = (a * P["v"][sl][None]).sum(-1) + P["c"][sl][None]
        mse.append(((yp - yt)**2).mean(0))
        y32.append(yt.transpose(0, 1))
        dz_zero.append((derivative == 0).float().mean((0, 2)))
        dz_abs.append(derivative.abs().mean((0, 2)))
        active.append((derivative != 0).any(0).float().mean(1))
    return (torch.cat(mse), torch.cat(y32), torch.cat(dz_zero),
            torch.cat(dz_abs), torch.cat(active))


def snapshot(out, task, period, arms, seeds, P, opt, target, flip, initial_bits,
             adapt_ema, start_mse=None, excluded=frozenset()):
    mse, y32, zero, absmean, active = support_readout(
        P, target, flip, initial_bits, arms, seeds, adapt_ema)
    n = len(seeds)
    for j, arm in enumerate(arms):
        design = arm_design(arm)
        for si, seed in enumerate(seeds):
            key = (arm[0], seed)
            if key in excluded:
                continue
            q = j*n + si
            mu_original = torch.cat((flip[q, :F], torch.full((R,), .5, device=flip.device)))
            mu_initial = torch.cat((initial_bits[q, :F], torch.full((R,), .5, device=flip.device)))
            mu_input = mu_original if design["X"] else mu_initial
            arrays = {name: value[q].detach().cpu().numpy() for name, value in P.items()}
            arrays.update({"adam_m_"+name: value[q].detach().cpu().numpy()
                           for name, value in opt.m.items()})
            arrays.update({"adam_v_"+name: value[q].detach().cpu().numpy()
                           for name, value in opt.v.items()})
            arrays.update(task=np.int64(task), step=np.int64(task*period),
                          optimizer_step=np.int64(opt.step_num.item()),
                          flip_state=flip[q, :F].detach().cpu().numpy(),
                          mu_original=mu_original.detach().cpu().numpy(),
                          mu_initial=mu_initial.detach().cpu().numpy(),
                          mu_input=mu_input.detach().cpu().numpy(),
                          teacher32targets=y32[q].detach().cpu().numpy(),
                          mse=np.float64(mse[q].item()),
                          task_start_mse=np.float64((mse if start_mse is None else start_mse)[q].item()),
                          derivative_exactzero_frac=np.float64(zero[q].item()),
                          derivative_absmean=np.float64(absmean[q].item()),
                          activeunit_frac=np.float64(active[q].item()),
                          w_sigma2=np.float64(.25 * P["W"][q, :, F:].square().sum().item()))
            if arm[6] in E.SNA_C:
                arrays["VEMA"] = adapt_ema[q].detach().cpu().numpy()
                arrays["alpha"] = E.alpha_from_ema(adapt_ema[q], arm[6]).detach().cpu().numpy()
            _atomic_npz(out / arm[0] / f"seed{seed}" / f"t{task:03d}.npz", **arrays)


def run(out, seeds=SEEDS, tasks=400, period=10000, device="cuda", graph=True,
        arms=ARMS, inject_nonfinite=frozenset()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    P, target, flip = E.initialize(arms, seeds, device, M)
    initial_bits = flip.clone()
    adapt_ema = torch.ones((len(arms)*len(seeds), H), device=device)
    opt = E.Optimizer(P, arms, len(seeds))
    n = len(seeds)
    designs = [arm_design(a) for a in arms]
    xflags = torch.tensor([d["X"] for d in designs for _ in seeds],
                          dtype=torch.bool, device=device)
    yflags = torch.tensor([d["Y"] for d in designs for _ in seeds],
                          dtype=torch.bool, device=device)
    input_gens = [E._generator(seed, 4, device) for seed in seeds]
    statuses = {(a[0], seed): {"status": "RUNNING", "last_saved_task": 0}
                for a in arms for seed in seeds}
    for (name, seed), value in statuses.items():
        _atomic_json(out / name / f"seed{seed}" / "status.json", value)
    for si, seed in enumerate(seeds):
        payload = {name: tensor[si].detach().cpu().numpy() for name, tensor in target.items()}
        payload["initial_bits"] = initial_bits[si].detach().cpu().numpy()
        _atomic_npz(out / "teacher" / f"seed{seed}.npz", **payload)

    def train_step():
        random_bits = torch.stack([torch.randint(0, 2, (R,), device=device,
                                               generator=g).float() for g in input_gens])
        random_bits = random_bits.repeat(len(arms), 1)
        xi, xt = context_inputs(flip, initial_bits, random_bits, xflags, yflags)
        y = E.teacher_output(xt, target)
        prevar = P["W"][:, :, F:].square().sum(-1) * .25
        opt.step(E.gradients(P, xi, y, arms, n, adapt_ema))
        for j, arm in enumerate(arms):
            if arm[6] in E.SNA_C:
                adapt_ema[j*n:(j+1)*n].mul_(.99).add_(prevar[j*n:(j+1)*n], alpha=.01)

    gobj = None
    if graph and str(device).startswith("cuda"):
        initial_P = {name: value.clone() for name, value in P.items()}
        initial_rng = [g.get_state() for g in input_gens]
        for _ in range(3):
            train_step()
        torch.cuda.synchronize()
        gobj = torch.cuda.CUDAGraph()
        for g in input_gens:
            gobj.register_generator_state(g)
        with torch.cuda.graph(gobj):
            train_step()
        torch.cuda.synchronize()
        for name in P:
            P[name].copy_(initial_P[name])
            opt.m[name].zero_()
            opt.v[name].zero_()
        opt.step_num.zero_()
        adapt_ema.fill_(1.0)
        for g, state in zip(input_gens, initial_rng):
            g.set_state(state)

    snapshot(out, 0, period, arms, seeds, P, opt, target, flip, initial_bits, adapt_ema)
    start = time.monotonic()
    for task in range(1, tasks+1):
        if task > 1:
            E.apply_flip(flip, arms, seeds, task-1, M)
        start_mse = support_readout(P, target, flip, initial_bits, arms, seeds, adapt_ema)[0]
        for _ in range(period):
            (train_step() if gobj is None else gobj.replay())
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        for j, arm in enumerate(arms):
            for si, seed in enumerate(seeds):
                if (arm[0], seed, task) in inject_nonfinite:
                    P["W"][j*n+si, 0, 0] = float("nan")
        state_tensors = (*P.values(), *opt.m.values(), *opt.v.values(), adapt_ema)
        finite = torch.stack([torch.isfinite(p).reshape(p.shape[0], -1).all(1)
                              for p in state_tensors]).all(0).cpu().tolist()
        excluded = set()
        for j, arm in enumerate(arms):
            for si, seed in enumerate(seeds):
                key = (arm[0], seed)
                status = statuses[key]
                if status["status"] == "DIVERGED":
                    excluded.add(key)
                elif not finite[j*n+si]:
                    status.update(status="DIVERGED", first_bad_task=task,
                                  first_bad_step=task*period)
                    excluded.add(key)
                    _atomic_json(out / arm[0] / f"seed{seed}" / "status.json", status)
        snapshot(out, task, period, arms, seeds, P, opt, target, flip, initial_bits,
                 adapt_ema, start_mse, excluded)
        for arm in arms:
            for seed in seeds:
                status = statuses[(arm[0], seed)]
                if status["status"] == "RUNNING":
                    status["last_saved_task"] = task
                    _atomic_json(out / arm[0] / f"seed{seed}" / "status.json", status)
    for (name, seed), status in statuses.items():
        if status["status"] == "RUNNING":
            status["status"] = "COMPLETED"
            _atomic_json(out / name / f"seed{seed}" / "status.json", status)
    return time.monotonic() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--period", type=int, default=10000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--source-git-hash", default=None)
    args = ap.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        ap.error("seeds must be nonempty and unique")
    if args.tasks < 1 or args.period < 1:
        ap.error("tasks and period must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA unavailable")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"output directory is nonempty: {args.out}")
    actual_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.source_git_hash is not None and args.source_git_hash != actual_hash:
        ap.error(f"source git hash mismatch: requested {args.source_git_hash}, HEAD {actual_hash}")
    root = Path(__file__).resolve().parents[1]
    specs = ("specs/spec_effdisp_inputscope_0924.md",
             "specs/spec_effdisp_inputscope_0924_addendum_c06.md")
    metadata = {"git_hash": actual_hash,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "spec_sha256": {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in specs},
                "torch_version": torch.__version__, "args": dict(vars(args)),
                "m": M, "r": R, "f": F, "hidden": H, "teacher_hidden": H,
                "beta": .7, "arms": [arm_design(a) for a in ARMS],
                "sigma_diag": [0.0]*F+[.25]*R,
                "status_rule": "analyze COMPLETED only; DIVERGED/RUNNING excluded",
                "snapshot": "t000 initialization; tNNN after task N and before next flip"}
    metadata["args"]["out"] = str(args.out)
    _atomic_json(args.out / "metadata.json", metadata)
    elapsed = run(args.out, tuple(args.seeds), args.tasks, args.period,
                  args.device, not args.no_graph)
    _atomic_json(args.out / "runtime.json", {"elapsed_s": elapsed})
    print(json.dumps({"elapsed_s": elapsed}))


if __name__ == "__main__":
    main()
