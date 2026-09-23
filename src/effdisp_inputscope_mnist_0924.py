#!/usr/bin/env python3
"""Paired input/label factorial on the fixed 1200-image RL-MNIST bank.

All cells use the original RL-MNIST 400x75-update task schedule.  The image
subset, initial weights, first-task identity permutation, first random label
vector, and per-epoch batch orders are shared by seed across cells and arms.
This is a controlled small-bank experiment, not canonical PermutedMNIST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import effdisp_mnist_0924 as M
from src import pmnist_0905 as H
from src import pmnist_rlmnist_0906 as RL

REPO = Path(__file__).resolve().parents[1]
SEEDS = (200, 201, 202, 203, 204)
CELLS = ("X0Y0", "X0Y1", "X1Y0", "X1Y1")
ACTS = ("LR", "SNA06", "SN05", "SNA03")
EPOCHS = 400
STEPS_PER_TASK = EPOCHS * RL.STEPS_PER_EPOCH
SPEC_FILES = ("specs/spec_effdisp_inputscope_0924.md",
              "specs/spec_effdisp_inputscope_0924_addendum_c06.md")


def make_act(name: str, device: torch.device):
    if name == "SN05":
        # H.AdaptiveSnake(beta=0) is exactly fixed alpha=.05. It uses the
        # established Snake forward and derivative, and retains V=1 forever.
        return H.AdaptiveSnake(0.05, 0.0, device)
    return M.make_act(name, device)


def schedules(seed: int, tasks: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Independent streams; first task is common identity input/random labels."""
    g_perm = H.stream("effdisp_inputscope_perm", seed)
    g_labels = H.stream("effdisp_inputscope_labels", seed)
    perms = [torch.arange(784)]
    labels = [RL.task_labels(g_labels)]
    for _ in range(1, tasks):
        perms.append(torch.randperm(784, generator=g_perm))
        labels.append(RL.task_labels(g_labels))
    return perms, labels


def actual_task(perms: list[torch.Tensor], labels: list[torch.Tensor],
                cell: str, task: int) -> tuple[torch.Tensor, torch.Tensor]:
    i = task - 1
    return (perms[i if cell[1] == "1" else 0],
            labels[i if cell[3] == "1" else 0])


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def diagnostics(params, x: torch.Tensor, act) -> dict[str, float]:
    with torch.no_grad():
        z1, a1, z2, a2, _ = M.forward(params, x, act)
        result = {}
        for layer, z, a in ((1, z1, a1), (2, z2, a2)):
            d = M.derivative(z, act, layer - 1)
            result[f"sigma_z{layer}_median"] = float(z.std(0, unbiased=False).median())
            result[f"mean_z{layer}_median"] = float(z.mean(0).median())
            result[f"activation_sigma_l{layer}_median"] = float(a.std(0, unbiased=False).median())
            result[f"derivative_zero_frac_l{layer}"] = float((d == 0).float().mean())
            result[f"derivative_abs_mean_l{layer}"] = float(d.abs().mean())
            result[f"active_unit_frac_l{layer}"] = float((d.abs().amax(0) >= H.DEAD_TOL).float().mean())
            if isinstance(act, H.AdaptiveSnake):
                alpha = act.alpha(layer - 1)
                raw = act.c / act.V[layer - 1].sqrt()
                result[f"adaptive_alpha_median_l{layer}"] = float(alpha.median())
                result[f"adaptive_alpha_clip_frac_l{layer}"] = float(((raw < act.lo) | (raw > act.hi)).float().mean())
        return result


def run(args: argparse.Namespace) -> None:
    if args.tasks < 1 or args.epochs < 1 or not args.seeds:
        raise ValueError("tasks, epochs, and seed list must be nonempty and positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds")
    if args.act in ("SN05", "SNA03") and args.cell != "X0Y1":
        raise ValueError("SN05/SNA03 are registered only for X0Y1")
    registered_tasks = 150 if args.cell == "X0Y1" and args.act in ("LR", "SNA06") else 50
    if not args.smoke and (args.tasks != registered_tasks or args.epochs != EPOCHS
                           or tuple(args.seeds) != SEEDS):
        raise ValueError("nonregistered task/epoch/seed settings require --smoke")
    if args.smoke and args.tasks > 2:
        raise ValueError("smoke is limited to two tasks")
    source_hash = H.git_hash()
    if args.source_git_hash and source_hash != args.source_git_hash:
        raise ValueError(f"checkout changed: expected {args.source_git_hash}, found {source_hash}")
    device = H.setup(args.device)
    H.DATA_DIR = M._data_dir(args.data_dir)
    mnist = H.Mnist(device)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    spt = args.epochs * RL.STEPS_PER_EPOCH
    spec_hash = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                 for name in SPEC_FILES}
    metadata = {
        "run_id": "effdisp_inputscope_0924", "mode": "inputscope_mnist",
        "activation": args.act, "cell": args.cell, "input_permutation": "moving" if args.cell[1] == "1" else "fixed",
        "random_labels": "moving" if args.cell[3] == "1" else "fixed",
        "tasks": args.tasks, "seeds": list(args.seeds), "epochs": args.epochs,
        "steps_per_task": spt, "batch": H.BATCH, "n_images": RL.N_IMAGES,
        "optimizer": "adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0.0, "dims": list(H.DIMS), "smoke": bool(args.smoke),
        "git_hash": source_hash, "spec_sha256": spec_hash,
        "data_sha256": mnist.sha256, "data_dir": str(H.DATA_DIR),
        "torch": torch.__version__, "device": str(device),
        "subset_role": "rl_subset", "perm_role": "effdisp_inputscope_perm",
        "labels_role": "effdisp_inputscope_labels", "batch_role": "effdisp_inputscope_batch",
        "initial_task": "identity permutation and same random label vector in every cell/arm for a seed",
        "input_bank": "seed_N/input_bank.npz has normalized float32 images[1200,784], raw-coordinate population covariance/mean/subset_idx; task p reconstructs x=images[:,p], mean=base_mean[p], covariance=base_covariance[p][:,p]",
        "snapshot": "state_000 initialization; state_t exactly after task t, before next task's input/label change",
        "gradient_probe": "W1/W2 mean-CE gradient norms on first 16 unshuffled current-task images at task start/end",
        "snake": {"SN05": "fixed alpha=.05 using beta=0, Vinit=1",
                  "SNA03": "c=.3, EMA beta=.01", "SNA06": "c=.6, EMA beta=.01"}.get(args.act),
        "snake_c": {"SN05": 0.05, "SNA03": 0.3, "SNA06": 0.6}.get(args.act),
        "snake_ema_beta": (0.0 if args.act == "SN05" else 0.01
                           if args.act in ("SNA03", "SNA06") else None),
        "snake_alpha_clip": [0.05, 3.0] if args.act != "LR" else None,
    }
    atomic_json(out / "metadata.json", metadata)
    status = {str(seed): {"state": "pending", "completed_tasks": 0} for seed in args.seeds}

    def write_status() -> None:
        atomic_json(out / "status.json", status)

    write_status()
    start = time.time()
    for seed in args.seeds:
        status[str(seed)]["state"] = "running"
        write_status()
        try:
            sd = out / f"seed_{seed:03d}"
            sd.mkdir()
            params = [p.detach() for p in H.init_params(seed, device)]
            act = make_act(args.act, device)
            engine = M.Engine(params, act, "adam", 0.001, device, args.tasks * spt, 75)
            M.save_state(sd, 0, params, "adam", engine.moments, 0, act)
            idx = RL.subset_idx(seed)
            x_base = mnist.train_x[idx.to(device)]
            base_mean, base_covariance = M.covariance(x_base)
            np.savez_compressed(sd / "input_bank.npz", subset_idx=idx.numpy(),
                                images=M.cpu_array(x_base), mean=base_mean,
                                covariance=base_covariance, dataset_size=RL.N_IMAGES)
            reference_cov = torch.from_numpy(base_covariance).to(device)
            perms, labels = schedules(seed, args.tasks)
            batch_gen = H.stream("effdisp_inputscope_batch", seed)
            rows = []
            for task in range(1, args.tasks + 1):
                task_start = time.time()
                engine.reset_acc()
                p_cpu, y_cpu = actual_task(perms, labels, args.cell, task)
                np.savez(sd / f"task_{task:03d}_input.npz", permutation=p_cpu.numpy(),
                         labels=y_cpu.numpy())
                x, y = x_base[:, p_cpu.to(device)], y_cpu.to(device)
                task_start_acc = M.accuracy(params, x, y, act)
                grad_w1_start, grad_w2_start = M.gradient_probe(params, x, y, act)
                for _ in range(args.epochs):
                    order = torch.randperm(RL.N_IMAGES, generator=batch_gen).to(device)
                    engine.run_chunk(x[order], y[order])
                engine.check()
                if engine.moments is None or not all(
                    bool(torch.isfinite(moment).all())
                    for group in engine.moments for moment in group
                ):
                    raise FloatingPointError(f"nonfinite Adam moments after task {task}")
                if isinstance(act, H.AdaptiveSnake) and not all(
                    bool(torch.isfinite(v).all()) for v in act.V
                ):
                    raise FloatingPointError(f"nonfinite adaptive V after task {task}")
                train_acc = M.accuracy(params, x, y, act)
                grad_w1_end, grad_w2_end = M.gradient_probe(params, x, y, act)
                M.save_state(sd, task, params, "adam", engine.moments, int(engine.step_t), act)
                diag = diagnostics(params, x, act)
                with torch.no_grad():
                    w = params[0].double()
                    ref_variance = ((w @ reference_cov) * w).sum(1).clamp_min(0)
                    reference_sigma = float(ref_variance.sqrt().median())
                row = {"seed": seed, "task": task, "global_step": int(engine.step_t),
                       "task_start_acc": task_start_acc, "train_acc": train_acc,
                       "online_acc": engine.read_acc() / spt, "test_acc": None,
                       "grad_w1_start_norm": grad_w1_start, "grad_w2_start_norm": grad_w2_start,
                       "grad_w1_end_norm": grad_w1_end, "grad_w2_end_norm": grad_w2_end,
                       "sigma_z1_reference_median": reference_sigma, **diag,
                       "wall_s": time.time() - task_start}
                rows.append(row)
                atomic_json(sd / "per_task.json", rows)
                status[str(seed)]["completed_tasks"] = task
                write_status()
                print(json.dumps(row), flush=True)
        except Exception as exc:
            status[str(seed)].update(state="failed", error=repr(exc), traceback=traceback.format_exc())
            write_status()
            print(f"seed={seed} FAILED: {exc!r}", file=sys.stderr, flush=True)
            continue
        status[str(seed)]["state"] = "complete"
        write_status()
    metadata["wall_s"] = time.time() - start
    atomic_json(out / "metadata.json", metadata)
    if any(item["state"] == "failed" for item in status.values()):
        raise RuntimeError(f"one or more seeds failed; see {out / 'status.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act", choices=ACTS, required=True)
    ap.add_argument("--cell", choices=CELLS, required=True)
    ap.add_argument("--tasks", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--source-git-hash", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
