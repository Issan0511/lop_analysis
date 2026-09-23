#!/usr/bin/env python3
"""Fresh task-boundary weight traces for RL-MNIST and PermutedMNIST.

Production defaults use Adam 1e-3 on both data constructions. A CUDA graph
contains one RL epoch (75 steps) or one PM chunk
(25 steps); all sampling uses the host's CPU RNG streams.  Every saved state
is *after* the final update for that task, without an intervening update.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H
from src import pmnist_rlmnist_0906 as RL

PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
ADAPTIVE_C = {"SNA03": 0.3, "SNA06": 0.6, "SNA1": 1.0}


def make_act(name: str, device: torch.device):
    return H.AdaptiveSnake(ADAPTIVE_C[name], 0.01, device) if name in ADAPTIVE_C else name


def activation(z: torch.Tensor, name, layer: int = 0) -> torch.Tensor:
    if isinstance(name, H.AdaptiveSnake):
        return name.phi(z, layer)
    if name == "R":
        return torch.relu(z)
    if name == "LR":
        return torch.where(z > 0, z, 0.1 * z)
    if name == "ELU":
        return torch.nn.functional.elu(z, alpha=1.0)
    if name == "GELU":
        return torch.nn.functional.gelu(z, approximate="none")
    if name == "SiLU":
        return torch.nn.functional.silu(z)
    if name == "SN1":
        return z + torch.sin(z) ** 2
    raise ValueError(name)


def derivative(z: torch.Tensor, name, layer: int = 0) -> torch.Tensor:
    if isinstance(name, H.AdaptiveSnake):
        return name.dphi(z, layer)
    if name == "R":
        return (z > 0).to(z.dtype)
    if name == "LR":
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
    if name == "ELU":
        # Matches F.elu's non-inplace backward in torch 2.13, including its
        # nonzero negative tail where elu(z)+1 has already rounded to zero.
        return torch.where(z > 0, torch.ones_like(z), torch.exp(z))
    if name == "GELU":
        return 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    if name == "SiLU":
        sig = torch.sigmoid(z)
        return sig + z * sig * (1 - sig)
    if name == "SN1":
        return 1 + torch.sin(2 * z)
    raise ValueError(name)


def forward(params: list[torch.Tensor], x: torch.Tensor, name):
    w1, b1, w2, b2, w3, b3 = params
    z1 = x @ w1.T + b1
    a1 = activation(z1, name, 0)
    z2 = a1 @ w2.T + b2
    a2 = activation(z2, name, 1)
    logits = a2 @ w3.T + b3
    return z1, a1, z2, a2, logits


def manual_grads(params: list[torch.Tensor], x: torch.Tensor,
                 y: torch.Tensor, name):
    """Mean CE gradient; its tensor operations are safe in a CUDA graph."""
    z1, a1, z2, a2, logits = forward(params, x, name)
    prob = torch.softmax(logits, dim=1)
    dz3 = (prob - torch.nn.functional.one_hot(y, 10)) / x.shape[0]
    gw3, gb3 = dz3.T @ a2, dz3.sum(0)
    dz2 = (dz3 @ params[4]) * derivative(z2, name, 1)
    gw2, gb2 = dz2.T @ a1, dz2.sum(0)
    dz1 = (dz2 @ params[2]) * derivative(z1, name, 0)
    gw1, gb1 = dz1.T @ x, dz1.sum(0)
    hit = (logits.argmax(1) == y).float().mean()
    return (gw1, gb1, gw2, gb2, gw3, gb3), hit, logits, z1, z2


def covariance(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Population covariance of the exact task inputs, stored in float64."""
    xd = x.double()
    mean = xd.mean(0)
    xc = xd - mean
    cov = xc.T @ xc / x.shape[0]
    return mean.cpu().numpy(), cov.cpu().numpy()


def cpu_array(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().copy()


def save_state(out: Path, task: int, params: list[torch.Tensor],
               optimizer: str, moments: list[list[torch.Tensor]] | None,
               step: int, act=None) -> None:
    state = {name: cpu_array(value) for name, value in zip(PARAM_NAMES, params)}
    state["task"] = np.int64(task)
    state["step"] = np.int64(step)
    if optimizer == "adam" and moments is not None:
        for label, group in zip(("m", "v"), moments):
            state.update({f"{label}_{name}": cpu_array(value)
                          for name, value in zip(PARAM_NAMES, group)})
    if isinstance(act, H.AdaptiveSnake):
        for layer in (0, 1):
            state[f"adaptive_V{layer+1}"] = cpu_array(act.V[layer])
            state[f"adaptive_alpha{layer+1}"] = cpu_array(act.alpha(layer))
    np.savez(out / f"state_{task:03d}.npz", **state)


@torch.no_grad()
def accuracy(params, x, y, act, batch: int = 4096) -> float:
    hit = torch.zeros((), dtype=torch.long, device=x.device)
    for i in range(0, len(x), batch):
        hit += (forward(params, x[i:i + batch], act)[-1].argmax(1)
                == y[i:i + batch]).sum()
    return float(hit) / len(x)


@torch.no_grad()
def gradient_probe(params, x, y, act) -> tuple[float, float]:
    """W1/W2 CE gradient norms on the deterministic first 16 task examples."""
    grads, *_ = manual_grads(params, x[:H.BATCH], y[:H.BATCH], act)
    return float(grads[0].norm()), float(grads[2].norm())


class Engine:
    def __init__(self, params, act, optimizer: str, lr: float,
                 device: torch.device, max_steps: int, graph_steps: int):
        self.params, self.act, self.optimizer, self.lr = params, act, optimizer, lr
        self.device, self.graph_steps = device, graph_steps
        self.moments = ([[torch.zeros_like(p) for p in params] for _ in range(2)]
                        if optimizer == "adam" else None)
        self.x = torch.zeros((graph_steps * H.BATCH, 784), device=device)
        self.y = torch.zeros((graph_steps * H.BATCH,), dtype=torch.long, device=device)
        self.acc = torch.zeros((), device=device)
        self.bad = torch.zeros((), dtype=torch.bool, device=device)
        self.step_t = torch.zeros((), dtype=torch.long, device=device)
        # Host Adam uses Python-double bias powers, cast to float32 by tensor ops.
        # Indexing keeps correction continuous across task boundaries.
        if optimizer == "adam":
            ticks = np.arange(1, max_steps + graph_steps + 2, dtype=np.float64)
            self.c1 = torch.from_numpy((1 - np.power(0.9, ticks)).astype(np.float32)).to(device)
            self.c2 = torch.from_numpy((1 - np.power(0.999, ticks)).astype(np.float32)).to(device)
        self.graph = None
        if device.type == "cuda":
            self._capture()

    def _one(self, i: int) -> None:
        a, b = i * H.BATCH, (i + 1) * H.BATCH
        grads, hit, logits, z1, z2 = manual_grads(self.params, self.x[a:b], self.y[a:b], self.act)
        with torch.no_grad():
            self.acc.add_(hit)
            self.bad.logical_or_(~torch.isfinite(logits).all())
            if self.optimizer == "sgd":
                for p, g in zip(self.params, grads):
                    p.sub_(self.lr * g)
            else:
                assert self.moments is not None
                ix = self.step_t.view(1)
                c1 = torch.gather(self.c1, 0, ix)[0]
                c2 = torch.gather(self.c2, 0, ix)[0]
                for p, g, m, v in zip(self.params, grads, *self.moments):
                    m.mul_(0.9).add_(g, alpha=0.1)
                    v.mul_(0.999).addcmul_(g, g, value=0.001)
                    p.sub_(self.lr * (m / c1) / ((v / c2).sqrt() + 1e-8))
            self.step_t.add_(1)
            if isinstance(self.act, H.AdaptiveSnake):
                self.act.update(z1, z2)

    def _chunk(self) -> None:
        for i in range(self.graph_steps):
            self._one(i)

    def _capture(self) -> None:
        mutable = [*self.params, self.acc, self.bad, self.step_t]
        if self.moments is not None:
            mutable += self.moments[0] + self.moments[1]
        if isinstance(self.act, H.AdaptiveSnake):
            mutable += self.act.V
        keep = [q.clone() for q in mutable]
        # Warmup on a side stream, as required for graph capture. Restore in place.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._chunk()
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():
            for q, old in zip(mutable, keep):
                q.copy_(old)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._chunk()
        with torch.no_grad():
            for q, old in zip(mutable, keep):
                q.copy_(old)

    def run_chunk(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert len(x) == len(self.x) and len(y) == len(self.y)
        self.x.copy_(x)
        self.y.copy_(y)
        if self.graph is None:
            self._chunk()
        else:
            self.graph.replay()

    def read_acc(self) -> float:
        return float(self.acc)

    def reset_acc(self) -> None:
        self.acc.zero_()
        self.bad.zero_()

    def check(self) -> None:
        if bool(self.bad) or not all(bool(torch.isfinite(p).all()) for p in self.params):
            raise FloatingPointError(f"nonfinite run after step {int(self.step_t)}")


def _data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    here = H.DATA_DIR
    if here.exists():
        return here
    return Path.home() / "Projects/claude/proj_004_drift/data/mnist"


def run(args: argparse.Namespace) -> None:
    if args.tasks < 1 or args.epochs < 1 or args.seeds.strip() == "":
        raise ValueError("positive tasks/epochs and at least one seed required")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("--lr must be finite and positive")
    if args.benchmark and args.tasks > 2:
        raise ValueError("benchmark is restricted to at most two tasks")
    if args.mode == "rl" and args.epochs != 400 and not args.smoke:
        raise ValueError("noncanonical RL epochs require --smoke")
    if args.mode == "pm" and args.epochs != 1 and not args.smoke:
        raise ValueError("noncanonical PM epochs require --smoke")
    if args.mode == "rl" and args.optimizer != "adam" and not args.smoke:
        raise ValueError("production RL uses Adam")
    if args.mode == "pm" and args.optimizer == "sgd" and args.act != "LR" and not args.smoke:
        raise ValueError("production PM SGD reference is LR only")
    device = H.setup(args.device)
    H.DATA_DIR = _data_dir(args.data_dir)
    mnist = H.Mnist(device)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mode, act_name = args.mode, args.act
    spt = (RL.STEPS_PER_EPOCH if mode == "rl" else H.STEPS_PER_TASK) * args.epochs
    optimizer = args.optimizer
    lr = args.lr if args.lr is not None else (0.001 if optimizer == "adam" else 0.01)
    metadata = {
        "mode": mode, "activation": act_name, "smoke": bool(args.smoke),
        "tasks": args.tasks, "seeds": seeds, "epochs": args.epochs,
        "steps_per_task": spt, "optimizer": optimizer, "lr": lr,
        "batch": H.BATCH, "dims": list(H.DIMS), "data_sha256": mnist.sha256,
        "data_dir": str(H.DATA_DIR), "torch": torch.__version__,
        "device": str(device), "git_hash": H.git_hash(),
        "source": "pmnist_rlmnist_0906" if mode == "rl" else "pmnist_0905",
        "rng_roles": (["init", "rl_subset", "rl_labels", "rl_batch"]
                      if mode == "rl" else ["init", "perm", "data", "batch"]),
        "snapshot": "state_000 is initialization; state_t is exactly after task t",
        "covariance": "population centered covariance in W1 input-column coordinates; PM x[:, permutation]",
        "reference_bank": "seed's first task actual permuted 10000 training images" if mode == "pm" else "same fixed 1200 images per seed",
        "elu_implementation": "torch.nn.functional.elu(alpha=1, inplace=False); negative backward exp(z), including float32 tail",
        "sn1_implementation": "fixed alpha=1: phi(z)=z+sin(z)^2, derivative=1+sin(2z)",
        "adaptive_implementation": "alpha=clip(c/sqrt(V),0.05,3), V initialized 1; EMA beta=0.01 of pre-update batch z variance (population), applied after optimizer; V/alpha saved at task ends",
        "adaptive_c": ADAPTIVE_C.get(act_name),
        "gradient_probe": "mean-CE manual W1/W2 gradient norms on first 16 unshuffled current-task training examples at task start/end",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    start = time.time()
    status = {str(seed): {"state": "pending", "completed_tasks": 0} for seed in seeds}

    def write_status() -> None:
        (out / "status.json").write_text(json.dumps(status, indent=2))

    write_status()
    for seed in seeds:
        status[str(seed)]["state"] = "running"
        write_status()
        try:
            sd = out / f"seed_{seed:03d}"
            sd.mkdir(exist_ok=True)
            params = [p.detach() for p in H.init_params(seed, device)]
            act = make_act(act_name, device)
            graph_steps = 75 if mode == "rl" else 25
            eng = Engine(params, act, optimizer, lr, device, args.tasks * spt, graph_steps)
            save_state(sd, 0, params, optimizer, eng.moments, 0, act)
            if mode == "rl":
                subset = RL.subset_idx(seed)
                x = mnist.train_x[subset.to(device)]
                mu, cov = covariance(x)
                np.savez(sd / "input_bank.npz", subset_idx=subset.numpy(), mean=mu,
                         covariance=cov, dataset_size=RL.N_IMAGES)
                lab_gen, batch_gen = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
            else:
                perm_gen, data_gen, batch_gen = (H.stream(role, seed)
                                                  for role in ("perm", "data", "batch"))
            rows = []
            for t in range(1, args.tasks + 1):
                task_start = time.time()
                eng.reset_acc()
                if mode == "rl":
                    y_cpu = RL.task_labels(lab_gen)
                    y = y_cpu.to(device)
                    np.savez(sd / f"task_{t:03d}_input.npz", labels=y_cpu.numpy())
                    task_start_acc = accuracy(params, x, y, act)
                    grad_w1_start, grad_w2_start = gradient_probe(params, x, y, act)
                    for _ in range(args.epochs):
                        order = torch.randperm(RL.N_IMAGES, generator=batch_gen).to(device)
                        eng.run_chunk(x[order], y[order])
                    train_acc = accuracy(params, x, y, act)
                    test_acc = None
                else:
                    perm_cpu = torch.randperm(784, generator=perm_gen)
                    idx_cpu = H.stratified_draw(mnist, data_gen)
                    order = torch.randperm(H.TASK_EXAMPLES, generator=batch_gen).to(device)
                    perm = perm_cpu.to(device)
                    x = mnist.train_x[idx_cpu.to(device)][:, perm]
                    y = mnist.train_y[idx_cpu.to(device)]
                    mu, cov = covariance(x)
                    if t == 1:
                        reference_cov = torch.from_numpy(cov).to(device)
                        np.savez(sd / "reference_bank.npz", mean=mu, covariance=cov,
                                 permutation=perm_cpu.numpy(), train_idx=idx_cpu.numpy(),
                                 bank="first_task_actual_permuted_training_images")
                    np.savez(sd / f"task_{t:03d}_input.npz", permutation=perm_cpu.numpy(),
                             train_idx=idx_cpu.numpy(), mean=mu, covariance=cov)
                    task_start_acc = accuracy(params, x, y, act)
                    grad_w1_start, grad_w2_start = gradient_probe(params, x, y, act)
                    xs, ys = x[order], y[order]
                    for _ in range(args.epochs):
                        for k in range(0, H.STEPS_PER_TASK, graph_steps):
                            eng.run_chunk(xs[k * H.BATCH:(k + graph_steps) * H.BATCH],
                                          ys[k * H.BATCH:(k + graph_steps) * H.BATCH])
                    train_acc = accuracy(params, x, y, act)
                    test_acc = accuracy(params, mnist.test_x[:, perm], mnist.test_y, act)
                eng.check()
                grad_w1_end, grad_w2_end = gradient_probe(params, x, y, act)
                save_state(sd, t, params, optimizer, eng.moments, int(eng.step_t), act)
                with torch.no_grad():
                    z1, a1, z2, a2, _ = forward(params, x, act)
                    diag = {}
                    for layer, z, a in ((1, z1, a1), (2, z2, a2)):
                        d = derivative(z, act, layer - 1)
                        diag[f"sigma_z{layer}_median"] = float(z.std(0, unbiased=False).median())
                        diag[f"mean_z{layer}_median"] = float(z.mean(0).median())
                        diag[f"activation_sigma_l{layer}_median"] = float(a.std(0, unbiased=False).median())
                        diag[f"derivative_zero_frac_l{layer}"] = float((d == 0).float().mean())
                        diag[f"derivative_abs_mean_l{layer}"] = float(d.abs().mean())
                        diag[f"active_unit_frac_l{layer}"] = float((d.abs().amax(0) >= H.DEAD_TOL).float().mean())
                        if isinstance(act, H.AdaptiveSnake):
                            alpha = act.alpha(layer - 1)
                            raw_alpha = act.c / act.V[layer - 1].sqrt()
                            diag[f"adaptive_alpha_median_l{layer}"] = float(alpha.median())
                            diag[f"adaptive_alpha_clip_frac_l{layer}"] = float(((raw_alpha < act.lo) | (raw_alpha > act.hi)).float().mean())
                    if mode == "pm":
                        w = params[0].double()
                        ref_variance = ((w @ reference_cov) * w).sum(1).clamp_min(0)
                        reference_sigma = float(ref_variance.sqrt().median())
                    else:
                        reference_sigma = diag["sigma_z1_median"]
                row = {"seed": seed, "task": t, "global_step": int(eng.step_t),
                       "task_start_acc": task_start_acc,
                       "grad_w1_start_norm": grad_w1_start,
                       "grad_w2_start_norm": grad_w2_start,
                       "grad_w1_end_norm": grad_w1_end,
                       "grad_w2_end_norm": grad_w2_end,
                       "online_acc": eng.read_acc() / spt, "train_acc": train_acc,
                       "test_acc": test_acc, **diag,
                       "sigma_z1_reference_median": reference_sigma,
                       "wall_s": time.time() - task_start}
                rows.append(row)
                (sd / "per_task.json").write_text(json.dumps(rows, indent=2))
                status[str(seed)]["completed_tasks"] = t
                write_status()
                print(json.dumps(row), flush=True)
        except Exception as exc:
            status[str(seed)]["state"] = "failed"
            status[str(seed)]["error"] = repr(exc)
            status[str(seed)]["traceback"] = traceback.format_exc()
            write_status()
            print(f"seed={seed} FAILED: {exc!r}", file=sys.stderr, flush=True)
            continue
        status[str(seed)]["state"] = "complete"
        write_status()
    metadata["wall_s"] = time.time() - start
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if any(item["state"] == "failed" for item in status.values()):
        raise RuntimeError(f"one or more seeds failed; see {out / 'status.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("rl", "pm"), required=True)
    p.add_argument("--act", choices=("LR", "R", "ELU", "GELU", "SiLU", "SN1",
                                     "SNA03", "SNA06", "SNA1"), default="LR")
    p.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--seeds", default="100,101,102,103,104")
    p.add_argument("--epochs", type=int, default=None,
                   help="RL=400, PM=1; reductions require --smoke")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 400 if args.mode == "rl" else 1
    run(args)


if __name__ == "__main__":
    main()
