#!/usr/bin/env python3
"""Hidden-width sweep on the fixed 1200-image RL-MNIST bank (cell X0Y1, leaky .1, Adam 1e-3).

Engine and schedule are copied from codex/effdisp_inputscope_0924
src/effdisp_inputscope_mnist_0924.py (0277e4d) with one change: the hidden
width h of the 784-h-h-10 MLP is a parameter.  With h=100 the image subset,
initial weights, random-label streams and batch orders are identical to that
run's LR_X0Y1 arm (seeds 200-204); only the device differs (cpu here).
Every saved state is *after* the final update of a task.
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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import effdisp_mnist_0924 as M
from src import pmnist_0905 as H
from src import pmnist_rlmnist_0906 as RL

REPO = Path(__file__).resolve().parents[1]
SEEDS = (200, 201, 202, 203, 204)
HIDDENS = (8, 16, 32, 64, 100, 300)
EPOCHS = 400
TASKS = 50
SPEC_FILES = ("specs/spec_hsweep_rlmnist_0924.md",)


def schedules(seed: int, tasks: int) -> list[torch.Tensor]:
    """Same label stream as effdisp_inputscope_0924 (first task shared by seed)."""
    g_labels = H.stream("effdisp_inputscope_labels", seed)
    return [RL.task_labels(g_labels) for _ in range(tasks)]


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
        return result


def run(args: argparse.Namespace) -> None:
    if args.tasks < 1 or args.epochs < 1 or not args.seeds:
        raise ValueError("tasks, epochs, and seed list must be nonempty and positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds")
    if not args.smoke and (args.tasks != TASKS or args.epochs != EPOCHS
                           or not set(args.seeds) <= set(SEEDS) or args.hidden not in HIDDENS):
        raise ValueError("nonregistered task/epoch/seed/hidden settings require --smoke")
    if args.smoke and args.tasks > 2:
        raise ValueError("smoke is limited to two tasks")
    source_hash = H.git_hash()
    if args.source_git_hash and source_hash != args.source_git_hash:
        raise ValueError(f"checkout changed: expected {args.source_git_hash}, found {source_hash}")
    torch.set_num_threads(args.threads)
    device = H.setup(args.device)
    H.DATA_DIR = M._data_dir(args.data_dir)
    mnist = H.Mnist(device)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    spt = args.epochs * RL.STEPS_PER_EPOCH
    dims = (784, args.hidden, args.hidden, 10)
    act = "LR"
    spec_hash = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                 for name in SPEC_FILES}
    metadata = {
        "run_id": "hsweep_rlmnist_0924", "mode": "hsweep_mnist_x0y1",
        "activation": act, "cell": "X0Y1", "input_permutation": "fixed", "random_labels": "moving",
        "hidden": args.hidden, "dims": list(dims),
        "tasks": args.tasks, "seeds": list(args.seeds), "epochs": args.epochs,
        "steps_per_task": spt, "batch": H.BATCH, "n_images": RL.N_IMAGES,
        "optimizer": "adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0.0, "smoke": bool(args.smoke), "threads": args.threads,
        "git_hash": source_hash, "spec_sha256": spec_hash,
        "data_sha256": mnist.sha256, "data_dir": str(H.DATA_DIR),
        "torch": torch.__version__, "device": str(device),
        "engine_origin": "codex/effdisp_inputscope_0924 src/effdisp_inputscope_mnist_0924.py (0277e4d), hidden width added",
        "subset_role": "rl_subset", "labels_role": "effdisp_inputscope_labels",
        "batch_role": "effdisp_inputscope_batch",
        "input_bank": "seed_N/input_bank.npz has normalized float32 images[1200,784], population covariance/mean/subset_idx; identity permutation in every task",
        "snapshot": "state_000 initialization; state_t exactly after task t, before the next task's label change",
        "gradient_probe": "W1/W2 mean-CE gradient norms on first 16 unshuffled current-task images at task start/end",
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
            params = [p.detach() for p in H.init_params(seed, device, dims)]
            engine = M.Engine(params, act, "adam", 0.001, device, args.tasks * spt, 75)
            M.save_state(sd, 0, params, "adam", engine.moments, 0, act)
            idx = RL.subset_idx(seed)
            x = mnist.train_x[idx.to(device)]
            base_mean, base_covariance = M.covariance(x)
            np.savez_compressed(sd / "input_bank.npz", subset_idx=idx.numpy(),
                                images=M.cpu_array(x), mean=base_mean,
                                covariance=base_covariance, dataset_size=RL.N_IMAGES)
            reference_cov = torch.from_numpy(base_covariance).to(device)
            labels = schedules(seed, args.tasks)
            batch_gen = H.stream("effdisp_inputscope_batch", seed)
            rows = []
            for task in range(1, args.tasks + 1):
                task_start = time.time()
                engine.reset_acc()
                y_cpu = labels[task - 1]
                np.savez(sd / f"task_{task:03d}_input.npz", permutation=np.arange(784),
                         labels=y_cpu.numpy())
                y = y_cpu.to(device)
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
                train_acc = M.accuracy(params, x, y, act)
                grad_w1_end, grad_w2_end = M.gradient_probe(params, x, y, act)
                M.save_state(sd, task, params, "adam", engine.moments, int(engine.step_t), act)
                diag = diagnostics(params, x, act)
                with torch.no_grad():
                    w = params[0].double()
                    ref_variance = ((w @ reference_cov) * w).sum(1).clamp_min(0)
                    reference_sigma = float(ref_variance.sqrt().median())
                    v_sigma = float(ref_variance.sum())
                row = {"seed": seed, "task": task, "global_step": int(engine.step_t),
                       "task_start_acc": task_start_acc, "train_acc": train_acc,
                       "online_acc": engine.read_acc() / spt, "test_acc": None,
                       "grad_w1_start_norm": grad_w1_start, "grad_w2_start_norm": grad_w2_start,
                       "grad_w1_end_norm": grad_w1_end, "grad_w2_end_norm": grad_w2_end,
                       "sigma_z1_reference_median": reference_sigma, "V_sigma": v_sigma,
                       "w1_sq": float((params[0].double() ** 2).sum()), **diag,
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
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--tasks", type=int, default=TASKS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--source-git-hash", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
