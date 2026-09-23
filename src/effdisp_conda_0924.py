"""Paired CondA task-end trajectories for the effective-dispersion validation.

This is an explicit extension of the established SCR environment: exactly k
*distinct* persistent bits change at each boundary.  The teacher and network
formulas are those of LTUTarget and VecMLP; the optimizer can be Adam or SGD.
No result interpretation lives in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .envs import LTUTarget
from .nets import VecMLP

M, F, H, TEACHER_H = 20, 15, 100, 100
SEEDS = tuple(range(100, 105))
PRIMARY = (("k0", 0, "adam", 1e-3, 1e-8, 5, "leaky"),
           ("k1", 1, "adam", 1e-3, 1e-8, 5, "leaky"),
           ("k3", 3, "adam", 1e-3, 1e-8, 5, "leaky"),
           ("k7", 7, "adam", 1e-3, 1e-8, 5, "leaky"),
           ("k1_eps1e3", 1, "adam", 1e-3, 1e-3, 5, "leaky"),
           ("k1_sgd", 1, "sgd", 1e-2, 0.0, 5, "leaky"))
EXTRA = (("k1_r2", 1, "adam", 1e-3, 1e-8, 2, "leaky"),
         ("k1_r10", 1, "adam", 1e-3, 1e-8, 10, "leaky"),
         ("k1_relu", 1, "adam", 1e-3, 1e-8, 5, "relu"),
         ("k1_elu", 1, "adam", 1e-3, 1e-8, 5, "elu"),
         ("k1_gelu", 1, "adam", 1e-3, 1e-8, 5, "gelu"),
         ("k1_silu", 1, "adam", 1e-3, 1e-8, 5, "silu"),
         ("k1_snake", 1, "adam", 1e-3, 1e-8, 5, "snake"),
         ("k1_sna03", 1, "adam", 1e-3, 1e-8, 5, "sna03"),
         ("k1_sna06", 1, "adam", 1e-3, 1e-8, 5, "sna06"),
         ("k1_sna1", 1, "adam", 1e-3, 1e-8, 5, "sna1"))
FREQUENCY = (("k1_T1000", 1, "adam", 1e-3, 1e-8, 5, "leaky"),)
M40 = (("m40_r5_k1", 1, "adam", 1e-3, 1e-8, 5, "leaky"),
       ("m40_r10_k1", 1, "adam", 1e-3, 1e-8, 10, "leaky"),
       ("m40_r10_k2", 2, "adam", 1e-3, 1e-8, 10, "leaky"),
       ("m40_r10_k1_sna06", 1, "adam", 1e-3, 1e-8, 10, "sna06"),
       ("m40_r10_k2_sna06", 2, "adam", 1e-3, 1e-8, 10, "sna06"))
SNA_C = {"sna03": 0.3, "sna06": 0.6, "sna1": 1.0}


def _generator(seed: int, role: int, device: str) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(92400000 + 1000 * seed + role)
    return g


def initialize(arms, seeds=SEEDS, device="cpu", m=M):
    """One independently seeded realization per seed, repeated across arms."""
    device = torch.device(device)
    n = len(seeds)
    teachers, nets, initial = [], [], []
    for seed in seeds:
        teacher = LTUTarget(1, m, TEACHER_H, 0.7,
                            _generator(seed, 1, device.type), device)
        net = VecMLP(1, H, m, _generator(seed, 2, device.type), device,
                     act_alpha=0.1)
        teachers.append(teacher.state_dict())
        nets.append(net.state_dict())
        initial.append(torch.randint(0, 2, (m,), generator=_generator(seed, 3, "cpu"),
                                     dtype=torch.int64))
    # arm major, seed minor
    def stack_repeated(items, key):
        one = torch.cat([v[key] for v in items], dim=0)
        return one.repeat(len(arms), *([1] * (one.ndim - 1))).clone()
    P = {key: stack_repeated(nets, key) for key in ("W", "b", "v", "c")}
    target = {key: stack_repeated(teachers, key)
              for key in ("W", "b", "v", "cout", "tau")}
    flip = torch.stack(initial).to(device=device, dtype=torch.float32).repeat(len(arms), 1)
    return P, target, flip


def flip_indices(seeds, task, f, device="cpu", m=M):
    """Task-specific permutation; k arms use nested prefixes, including k=0."""
    # Draw a full 20-bit priority order, then restrict to eligible persistent
    # coordinates. This couples r=2/5/10 variants without changing input RNG.
    rows = []
    for seed in seeds:
        order = torch.randperm(m, generator=_generator(seed, 100 + task, "cpu"))
        rows.append(order[order < f])
    return torch.stack(rows).to(device)


def apply_flip(flip, arms, seeds, task, m=M):
    n = len(seeds)
    for j, (_, k, _, _, _, r, _) in enumerate(arms):
        f = m - r
        perm = flip_indices(seeds, task, f, flip.device, m)
        if k:
            rows = torch.arange(j * n, (j + 1) * n, device=flip.device)[:, None]
            cols = perm[:, :k]
            flip[rows, cols] = 1.0 - flip[rows, cols]


def teacher_output(x, target):
    pre = torch.bmm(target["W"], x.unsqueeze(-1)).squeeze(-1) + target["b"]
    active = (pre >= target["tau"]).to(x.dtype)
    return (active * target["v"]).sum(-1) + target["cout"]


def _activation(pre, kind, alpha=None):
    if kind == "leaky":
        return torch.where(pre > 0, pre, 0.1 * pre), torch.where(
            pre > 0, torch.ones_like(pre), torch.full_like(pre, 0.1))
    if kind == "relu":
        return torch.relu(pre), (pre > 0).to(pre.dtype)
    if kind == "elu":
        neg = torch.clamp(pre, max=0.0)
        return torch.nn.functional.elu(pre, alpha=1.0, inplace=False), torch.where(
            pre > 0, torch.ones_like(pre), torch.exp(neg))
    if kind == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(pre / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * pre.square()) / math.sqrt(2.0 * math.pi)
        return pre * cdf, cdf + pre * pdf
    if kind == "silu":
        sig = torch.sigmoid(pre)
        return pre * sig, sig + pre * sig * (1.0 - sig)
    if kind == "snake":
        return pre + torch.sin(pre).square(), 1.0 + torch.sin(2.0 * pre)
    if kind in SNA_C:
        if alpha is None:
            raise ValueError("adaptive Snake requires frozen alpha")
        phase = alpha * pre
        return pre + torch.sin(phase).square() / alpha, 1.0 + torch.sin(2.0 * phase)
    raise ValueError(kind)


def alpha_from_ema(ema, kind):
    return torch.clamp(SNA_C[kind] / torch.sqrt(ema), min=0.05, max=3.0)


def gradients(P, x, y, arms=None, nseeds=None, adapt_ema=None):
    pre = torch.bmm(P["W"], x.unsqueeze(-1)).squeeze(-1) + P["b"]
    if arms is None:
        a, gate = _activation(pre, "leaky")
    else:
        av, gv = [], []
        for j, arm in enumerate(arms):
            sl = slice(j * nseeds, (j + 1) * nseeds)
            alpha = alpha_from_ema(adapt_ema[sl], arm[6]) if arm[6] in SNA_C else None
            aj, gj = _activation(pre[sl], arm[6], alpha)
            av.append(aj)
            gv.append(gj)
        a, gate = torch.cat(av), torch.cat(gv)
    delta = (a * P["v"]).sum(-1) + P["c"] - y
    gb = (2.0 * delta)[:, None] * P["v"] * gate
    return {"W": gb[:, :, None] * x[:, None, :], "b": gb,
            "v": (2.0 * delta)[:, None] * a, "c": 2.0 * delta}


class Optimizer:
    def __init__(self, P, arms, nseeds):
        self.P = P
        self.lr = torch.tensor([a[3] for a in arms for _ in range(nseeds)],
                               dtype=torch.float32, device=P["W"].device)
        self.eps = torch.tensor([a[4] for a in arms for _ in range(nseeds)],
                                dtype=torch.float32, device=P["W"].device)
        self.adam = torch.tensor([a[2] == "adam" for a in arms for _ in range(nseeds)],
                                 dtype=torch.bool, device=P["W"].device)
        self.m = {k: torch.zeros_like(v) for k, v in P.items()}
        self.v = {k: torch.zeros_like(v) for k, v in P.items()}
        self.step_num = torch.zeros((), dtype=torch.int64, device=P["W"].device)
        self.beta1 = torch.tensor(0.9, device=P["W"].device)
        self.beta2 = torch.tensor(0.999, device=P["W"].device)

    def step(self, grad):
        self.step_num.add_(1)
        # Both branches are computed to keep one static CUDA graph. SGD slots use
        # the raw gradient, with zero weight decay and no optimizer-state reset.
        t = self.step_num.to(torch.float32)
        bc1 = 1.0 - torch.pow(self.beta1, t)
        bc2 = 1.0 - torch.pow(self.beta2, t)
        for key, p in self.P.items():
            g = grad[key]
            m, v = self.m[key], self.v[key]
            m.mul_(0.9).add_(g, alpha=0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            nd = p.ndim - 1
            lr = self.lr.view(-1, *([1] * nd))
            eps = self.eps.view(-1, *([1] * nd))
            adam = self.adam.view(-1, *([1] * nd))
            direction = torch.where(adam, (m / bc1) / ((v / bc2).sqrt() + eps), g)
            p.sub_(lr * direction)


def exact_support_metrics(P, target, flip, arms, nseeds, m=M, adapt_ema=None):
    """Exact conditional support MSE and within-task covariance width."""
    device = flip.device
    mse, within, mu, dz_zero, dz_abs, active = [], [], [], [], [], []
    for j, (_, _, _, _, _, r, act) in enumerate(arms):
        f = m - r
        sl = slice(j * nseeds, (j + 1) * nseeds)
        bits = ((torch.arange(2 ** r, device=device)[:, None]
                 >> torch.arange(r, device=device)) & 1).float()
        x = torch.cat((flip[sl, :f][None].expand(2 ** r, -1, -1),
                       bits[:, None].expand(-1, nseeds, -1)), dim=-1)
        yt = torch.einsum("nhm,snm->snh", target["W"][sl], x) + target["b"][sl][None]
        yt = ((yt >= target["tau"][sl][None]).float() * target["v"][sl][None]).sum(-1) + target["cout"][sl][None]
        pre = torch.einsum("nhm,snm->snh", P["W"][sl], x) + P["b"][sl][None]
        alpha = alpha_from_ema(adapt_ema[sl], act)[None] if act in SNA_C else None
        a, derivative = _activation(pre, act, alpha)
        yp = (a * P["v"][sl][None]).sum(-1) + P["c"][sl][None]
        mse.append(((yp - yt) ** 2).mean(0))
        within.append((P["W"][sl, :, f:] ** 2).sum((-1, -2)) * 0.25)
        mu.append(torch.cat((flip[sl, :f], torch.full((nseeds, r), 0.5, device=device)), dim=1))
        dz_zero.append((derivative == 0).float().mean((0, 2)))
        dz_abs.append(derivative.abs().mean((0, 2)))
        active.append((derivative != 0).any(0).float().mean(1))
    return (torch.cat(mse), torch.cat(within), torch.cat(mu),
            torch.cat(dz_zero), torch.cat(dz_abs), torch.cat(active))


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(temporary, path)


def save_task(out, task, arms, seeds, P, target, flip, period, step,
              excluded=frozenset(), task_start_mse=None, m=M, adapt_ema=None):
    mse, within, mu, dz_zero, dz_abs, active = exact_support_metrics(
        P, target, flip, arms, len(seeds), m, adapt_ema)
    for j, (name, k, method, lr, eps, rbits, act) in enumerate(arms):
        for si, seed in enumerate(seeds):
            r = j * len(seeds) + si
            if (name, seed) in excluded:
                continue
            dest = out / name / f"seed{seed}" / f"t{task:03d}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = {key: P[key][r].detach().cpu().numpy() for key in P}
            payload.update(flip_state=flip[r, :m-rbits].detach().cpu().numpy(),
                           mu=mu[r].detach().cpu().numpy(),
                           mse=np.float64(mse[r].item()),
                           task_start_mse=np.float64((mse if task_start_mse is None
                                                      else task_start_mse)[r].item()),
                           w_sigma2=np.float64(within[r].item()),
                           derivative_exactzero_frac=np.float64(dz_zero[r].item()),
                           derivative_absmean=np.float64(dz_abs[r].item()),
                           activeunit_frac=np.float64(active[r].item()),
                           task=np.int64(task), step=np.int64(step), period=np.int64(period),
                           m=np.int64(m), r=np.int64(rbits))
            if act in SNA_C:
                payload["VEMA"] = adapt_ema[r].detach().cpu().numpy()
                payload["alpha"] = alpha_from_ema(adapt_ema[r], act).detach().cpu().numpy()
            temporary = dest.with_suffix(".npz.tmp")
            with temporary.open("wb") as fh:
                np.savez(fh, **payload)
            os.replace(temporary, dest)


def run_group(out, arms, seeds, period, tasks, device, graph=True,
              inject_nonfinite=frozenset(), m=M):
    P, target, flip = initialize(arms, seeds, device, m)
    adapt_ema = torch.ones((len(arms) * len(seeds), H), device=device)
    for si, seed in enumerate(seeds):
        dest = out / f"teacher_m{m}" / f"seed{seed}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value[si].detach().cpu().numpy() for key, value in target.items()}
        payload["initial_bits"] = flip[si].detach().cpu().numpy()
        temporary = dest.with_suffix(".npz.tmp")
        with temporary.open("wb") as fh:
            np.savez(fh, **payload)
        os.replace(temporary, dest)
    statuses = {(arm[0], seed): {"status": "RUNNING", "last_saved_task": 0}
                for arm in arms for seed in seeds}
    for (name, seed), value in statuses.items():
        _write_json(out / name / f"seed{seed}" / "status.json", value)
    opt = Optimizer(P, arms, len(seeds))
    n = len(seeds)
    input_gens = [_generator(seed, 4, device) for seed in seeds]
    persist_mask = torch.tensor([[i < m - arm[5] for i in range(m)]
                                 for arm in arms for _ in seeds], device=device)

    def train_step():
        # Draw one independent random-bit vector per seed, then broadcast to
        # every arm. Flip priorities use a different CPU stream.
        rnd = torch.stack([torch.randint(0, 2, (m,), generator=g,
                                          device=device).float() for g in input_gens])
        rnd = rnd.repeat(len(arms), 1)
        x = torch.where(persist_mask, flip, rnd)
        y = teacher_output(x, target)
        # Conditional input variance is exact for Bernoulli(1/2) free bits;
        # measure it before the optimizer changes W, then update EMA afterward.
        variance = torch.stack([(P["W"][j*n:(j+1)*n, :, m-arm[5]:].square().sum(-1) * 0.25)
                                for j, arm in enumerate(arms)])
        opt.step(gradients(P, x, y, arms, n, adapt_ema))
        for j, arm in enumerate(arms):
            if arm[6] in SNA_C:
                adapt_ema[j*n:(j+1)*n].mul_(0.99).add_(variance[j], alpha=0.01)

    graph_obj = None
    if graph and str(device).startswith("cuda"):
        # Capture must not count as an update. Restore all state and RNG streams
        # after warmup/capture; CUDA graph RNG state advances on replay.
        start_P = {k: v.clone() for k, v in P.items()}
        start_ema = adapt_ema.clone()
        start_rng = [g.get_state() for g in input_gens]
        for _ in range(3):
            train_step()
        torch.cuda.synchronize()
        graph_obj = torch.cuda.CUDAGraph()
        for g in input_gens:
            graph_obj.register_generator_state(g)
        with torch.cuda.graph(graph_obj):
            train_step()
        torch.cuda.synchronize()
        for k in P:
            P[k].copy_(start_P[k])
            opt.m[k].zero_()
            opt.v[k].zero_()
        opt.step_num.zero_()
        adapt_ema.copy_(start_ema)
        for g, state in zip(input_gens, start_rng):
            g.set_state(state)

    save_task(out, 0, arms, seeds, P, target, flip, period, 0,
              m=m, adapt_ema=adapt_ema)
    start = time.monotonic()
    for task in range(1, tasks + 1):
        if task > 1:
            apply_flip(flip, arms, seeds, task - 1, m)
        task_start_mse = exact_support_metrics(P, target, flip, arms, n, m, adapt_ema)[0]
        for _ in range(period):
            if graph_obj is None:
                train_step()
            else:
                graph_obj.replay()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        # Detect failure per realization. The R dimension has independent
        # parameter/optimizer slots, so one NaN never invalidates other runs.
        for j, arm in enumerate(arms):
            for si, seed in enumerate(seeds):
                if (arm[0], seed, task) in inject_nonfinite:
                    P["W"][j * n + si, 0, 0] = float("nan")
        finite = torch.stack([torch.isfinite(p).reshape(p.shape[0], -1).all(1)
                              for p in P.values()]).all(0).cpu().tolist()
        excluded = set()
        for j, arm in enumerate(arms):
            for si, seed in enumerate(seeds):
                key = (arm[0], seed)
                state = statuses[key]
                if state["status"] == "DIVERGED":
                    excluded.add(key)
                elif not finite[j * n + si]:
                    state.update(status="DIVERGED", first_bad_task=task,
                                 first_bad_step=task * period)
                    excluded.add(key)
                    _write_json(out / arm[0] / f"seed{seed}" / "status.json", state)
        save_task(out, task, arms, seeds, P, target, flip, period,
                  task * period, excluded, task_start_mse, m, adapt_ema)
        for j, arm in enumerate(arms):
            for seed in seeds:
                state = statuses[(arm[0], seed)]
                if state["status"] == "RUNNING":
                    state["last_saved_task"] = task
    for (name, seed), state in statuses.items():
        if state["status"] == "RUNNING":
            state["status"] = "COMPLETED"
            _write_json(out / name / f"seed{seed}" / "status.json", state)
    return time.monotonic() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--period", type=int, default=10000)
    ap.add_argument("--frequency-period", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--group", choices=("primary", "frequency", "m40", "both"), default="both")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    if args.tasks < 1 or args.period < 1 or args.frequency_period < 1:
        ap.error("tasks and periods must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        ap.error("seeds must be nonempty and unique")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    for group_m, group in ((M, PRIMARY + EXTRA + FREQUENCY), (40, M40)):
        for name, k, _, _, _, r, _ in group:
            if not (1 <= r < group_m and 0 <= k <= group_m-r):
                raise ValueError(f"invalid CondA bit counts in {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"m_values": [20, 40], "hidden": H, "teacher_hidden": TEACHER_H,
                "beta": 0.7, "batch": 1,
                "elu_rule": "torch.nn.functional.elu(alpha=1,inplace=False); derivative exp(z) on z<=0",
                "snake_rule": "fixed alpha=1: phi(z)=z+sin(z)^2; derivative=1+sin(2z)",
                "adaptive_rule": "VEMA init 1; after each optimizer step V=.99*V+.01*(.25*sum(preupdate W_free^2)); alpha=clip(c/sqrt(V),.05,3), detached for gradients; phi=z+sin(alpha*z)^2/alpha",
                "weight_decay": 0.0, "seeds": args.seeds, "tasks": args.tasks,
                "primary_period": args.period, "frequency_period": args.frequency_period,
                "primary_arms": PRIMARY, "extra_arms": EXTRA, "frequency_arms": FREQUENCY,
                "m40_arms": M40,
                "spec_arm_map": {"A_k0": "k0", "A_k1": "k1", "A_k3": "k3", "A_k7": "k7",
                                 "E_k1": "k1_eps1e3", "S_k1": "k1_sgd", "A_fast": "k1_T1000",
                                 "A_r2": "k1_r2", "A_r10": "k1_r10", "A_ReLU": "k1_relu",
                                 "A_ELU": "k1_elu", "A_GELU": "k1_gelu", "A_SiLU": "k1_silu",
                                 "A_Snake": "k1_snake", "A_SNA03": "k1_sna03",
                                 "A_SNA06": "k1_sna06", "A_SNA1": "k1_sna1",
                                 "A40_r5_k1": "m40_r5_k1", "A40_r10_k1": "m40_r10_k1",
                                 "A40_r10_k2": "m40_r10_k2",
                                 "A40_r10_k1_SNA06": "m40_r10_k1_sna06",
                                 "A40_r10_k2_SNA06": "m40_r10_k2_sna06"},
                "arm_dimensions": {a[0]: {"m": m, "r": a[5], "f": m-a[5]}
                                   for m, group in ((20, PRIMARY + EXTRA + FREQUENCY), (40, M40))
                                   for a in group},
                "within_sigma_diag": "per arm: [0]*(m-r)+[0.25]*r",
                "flip_rule": "distinct k-bit nested prefix of per-seed task permutation",
                "snapshot": "t000 init; tNNN after task N, before next flip",
                "device": args.device, "cuda_graph": not args.no_graph,
                "torch_version": torch.__version__,
                "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "status_rule": "only COMPLETED cases are analyzable; DIVERGED and RUNNING are excluded"}
    root = Path(__file__).resolve().parents[1]
    specs = ("specs/spec_effdisp_validation_0924.md",
             "specs/spec_effdisp_validation_0924_addendum_bits_acts.md",
             "specs/spec_effdisp_validation_0924_addendum_snake_censoring.md",
             "specs/spec_effdisp_validation_0924_addendum_adaptive_dimension.md")
    metadata["spec_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in specs}
    _write_json(args.out / "metadata.json", metadata)
    durations = {}
    if args.group in ("primary", "both"):
        durations["primary_s"] = run_group(args.out, PRIMARY + EXTRA, args.seeds, args.period,
                                            args.tasks, args.device, not args.no_graph)
    if args.group in ("frequency", "both"):
        durations["frequency_s"] = run_group(args.out, FREQUENCY, args.seeds,
                                              args.frequency_period, args.tasks,
                                              args.device, not args.no_graph)
    if args.group in ("m40", "both"):
        durations["m40_s"] = run_group(args.out, M40, args.seeds, args.period,
                                       args.tasks, args.device, not args.no_graph, m=40)
    (args.out / "runtime.json").write_text(json.dumps(durations, indent=2) + "\n")
    print(json.dumps(durations))


if __name__ == "__main__":
    main()
