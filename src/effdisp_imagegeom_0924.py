#!/usr/bin/env python3
"""Pre-registered CIFAR image geometry experiment (spec_effdisp_imagegeom_0924).

The 50 trajectories are executed in four homogeneous input-width groups.  Each
slot has its own seed, geometry, optimizer, labels and batch-order stream; slots
with the same seed deliberately consume identical label and batch-order draws.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC

EXPERIMENT = "effdisp_imagegeom_0924"
SOURCE = Path(__file__).resolve()
SPEC = SOURCE.parents[1] / "specs" / "spec_effdisp_imagegeom_0924.md"
DEFAULT_ARCHIVE = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar10/cifar-10-python.tar.gz")
DEFAULT_OUT = Path("/home/issan/Projects/obsidian-research-data/effdisp_imagegeom_0924/raw")
GEOMETRIES = {"rgb32": 3072, "dup64": 12288, "dup64_half": 12288,
              "gray32": 1024, "avg16": 768}
GROUPS = (("rgb32",), ("dup64", "dup64_half"), ("gray32",), ("avg16",))
OPTIMIZERS = ("adam", "sgd")
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
N_IMAGES, BATCH, EPOCHS, TASKS = 1200, 16, 400, 50
ADAM_LR, SGD_LR, BETA1, BETA2, EPS = 0.001, 0.01, 0.9, 0.999, 1e-8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE.parents[1], text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    keys = ("geometry", "optimizer", "seed", "task", "start_acc", "end_acc",
            "online_acc", "start_ce", "end_ce", "mean_abs_dphi_l1",
            "mean_abs_dphi_l2", "activeunit_frac_l1", "activeunit_frac_l2",
            "positive_gate_frac_l1", "positive_gate_frac_l2", "status")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def refuse_nonempty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing existing nonempty output: {path}")


def transform(x: torch.Tensor, geometry: str) -> torch.Tensor:
    """Return CHW flattened float32 bank; input is raw01 in CIFAR CHW order."""
    if geometry not in GEOMETRIES:
        raise ValueError(geometry)
    z = x.reshape(-1, 3, 32, 32)
    if geometry.startswith("dup64"):
        z = z.repeat_interleave(2, -2).repeat_interleave(2, -1)
        if geometry == "dup64_half":
            z = z * 0.5
    elif geometry == "gray32":
        z = (z[:, 0:1] * 0.299 + z[:, 1:2] * 0.587 + z[:, 2:3] * 0.114)
    elif geometry == "avg16":
        z = F.avg_pool2d(z, 2, 2)
    return z.reshape(x.shape[0], -1).contiguous()


def mapped_first_weight(w: torch.Tensor, geometry: str) -> torch.Tensor:
    if geometry not in ("dup64", "dup64_half"):
        raise ValueError(geometry)
    z = w.reshape(w.shape[0], 3, 32, 32).repeat_interleave(2, -2).repeat_interleave(2, -1)
    return (z / (4.0 if geometry == "dup64" else 2.0)).reshape(w.shape[0], -1).contiguous()


def initial_params(seed: int, geometry: str, device: torch.device) -> list[torch.Tensor]:
    # Biases and downstream layers come from the same host baseline, by seed.
    base = H.init_params(seed, torch.device("cpu"), (3072, 100, 100, 10))
    out = [q.detach().clone() for q in base]
    if geometry in ("dup64", "dup64_half"):
        out[0] = mapped_first_weight(out[0], geometry)
    elif geometry in ("gray32", "avg16"):
        d = GEOMETRIES[geometry]
        g = H.stream("init_native_geometry_" + geometry, seed)
        out[0] = ((torch.rand((100, d), generator=g) * 2 - 1) / math.sqrt(d))
    return [q.to(device) for q in out]


def forward(P: list[torch.Tensor], X: torch.Tensor):
    w1, b1, w2, b2, w3, b3 = P
    z1 = torch.baddbmm(b1[:, None, :], X, w1.transpose(1, 2))
    a1 = torch.where(z1 > 0, z1, 0.1 * z1)
    z2 = torch.baddbmm(b2[:, None, :], a1, w2.transpose(1, 2))
    a2 = torch.where(z2 > 0, z2, 0.1 * z2)
    logits = torch.baddbmm(b3[:, None, :], a2, w3.transpose(1, 2))
    return z1, z2, logits


def evaluate(P: list[torch.Tensor], X: torch.Tensor, Y: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        z1, z2, logits = forward(P, X)
        r = X.shape[0]
        ce = F.cross_entropy(logits.reshape(-1, 10), Y.reshape(-1), reduction="none").view(r, -1).mean(1)
        acc = (logits.argmax(-1) == Y).float().mean(1)
        out = {"acc": acc, "ce": ce}
        for li, z in enumerate((z1, z2), 1):
            gate = torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
            out[f"mean_abs_dphi_l{li}"] = gate.abs().mean((1, 2))
            out[f"activeunit_frac_l{li}"] = (gate.abs() > H.DEAD_TOL).any(1).float().mean(1)
            out[f"positive_gate_frac_l{li}"] = (z > 0).float().mean((1, 2))
        return out


def update_in_place(P, M, V, grads, adam_mask, live_mask, step: torch.Tensor):
    """Independent Adam/SGD slots; graph-friendly in-place update."""
    with torch.no_grad():
        step.add_(1)
        t = step.to(torch.float32)
        inv1 = (1.0 - torch.pow(BETA1, t)).reciprocal()
        inv2 = (1.0 - torch.pow(BETA2, t)).reciprocal()
        for p, m, v, g in zip(P, M, V, grads):
            mask = adam_mask.view((-1,) + (1,) * (p.ndim - 1))
            m_new = m * BETA1 + g * (1 - BETA1)
            v_new = v * BETA2 + g.square() * (1 - BETA2)
            live = live_mask.view((-1,) + (1,) * (p.ndim - 1))
            m.copy_(torch.where(mask & live, m_new, m))
            v.copy_(torch.where(mask & live, v_new, v))
            adam_delta = ADAM_LR * (m * inv1) / ((v * inv2).sqrt() + EPS)
            p.sub_(torch.where(live, torch.where(mask, adam_delta, SGD_LR * g),
                               torch.zeros_like(p)))


def case_finite(P, M, V) -> torch.Tensor:
    ok = torch.ones(P[0].shape[0], dtype=torch.bool, device=P[0].device)
    for q in (*P, *M, *V):
        ok &= torch.isfinite(q).flatten(1).all(1)
    return ok


def save_snapshot(path: Path, P, M, V, r: int, task: int, step: int, optimizer: str) -> None:
    payload = {name: p[r].detach().cpu().numpy() for name, p in zip(PARAM_NAMES, P)}
    # SGD has no optimizer moments; explicitly store scalar zero state instead of
    # repeatedly writing full-size zero W1 tensors.
    for kind, state in (("adam_m", M), ("adam_v", V)):
        for name, q in zip(PARAM_NAMES, state):
            payload[f"{kind}_{name}"] = (q[r].detach().cpu().numpy() if optimizer == "adam"
                                          else np.array(0.0, dtype=np.float32))
    payload.update(task=np.int32(task), step=np.int64(step), optimizer_step=np.int64(step if optimizer == "adam" else 0))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def make_data(archive: Path, seeds: tuple[int, ...], tasks: int, out: Path):
    # RC.Cifar10 fixes the archive path relative to the clone. Temporarily set
    # its module constant so the parser/validation are exactly the host's.
    old_dir, old_name = RC.DATA_DIR, RC.ARCHIVE
    RC.DATA_DIR, RC.ARCHIVE = archive.parent, archive.name
    try:
        data = RC.Cifar10()
    finally:
        RC.DATA_DIR, RC.ARCHIVE = old_dir, old_name
    labels = {}
    indices = {}
    base = {}
    for seed in seeds:
        idx = RC.subset_idx(seed)
        indices[seed] = idx.numpy()
        base[seed] = data.images(idx, torch.device("cpu"))
        g = H.stream("rlc_labels", seed)
        labels[seed] = torch.stack([RC.task_labels(g) for _ in range(tasks)])
        (out / "labels").mkdir(exist_ok=True)
        np.savez_compressed(out / "labels" / f"seed{seed}.npz", labels=labels[seed].numpy())
    for geometry in GEOMETRIES:
        d = out / "banks" / geometry
        d.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            x = transform(base[seed], geometry)
            np.savez_compressed(d / f"seed{seed}.npz", x=x.numpy(), indices=indices[seed])
    return labels, base, indices, data.sha256[archive.name]


def group_cases(group: tuple[str, ...], seeds: tuple[int, ...]):
    return [(geometry, optimizer, seed) for geometry in group for optimizer in OPTIMIZERS for seed in seeds]


def run_group(group: tuple[str, ...], seeds: tuple[int, ...], tasks: int, epochs: int,
              out: Path, labels: dict[int, torch.Tensor], base: dict[int, torch.Tensor],
              device: torch.device, all_rows: list[dict], graph: bool) -> None:
    cases = group_cases(group, seeds)
    r = len(cases)
    X = torch.stack([transform(base[seed], geometry) for geometry, _, seed in cases]).to(device)
    P = [torch.stack([initial_params(seed, geometry, device)[i] for geometry, _, seed in cases]).detach().requires_grad_(True)
         for i in range(6)]
    M, V = [torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P]
    adam_mask = torch.tensor([optimizer == "adam" for _, optimizer, _ in cases], dtype=torch.bool, device=device)
    live_mask = torch.ones(r, dtype=torch.bool, device=device)
    static_idx = torch.zeros(r, BATCH, dtype=torch.long, device=device)
    Y = torch.zeros(r, N_IMAGES, dtype=torch.long, device=device)
    step = torch.zeros((), dtype=torch.int64, device=device)
    correct_sum = torch.zeros(r, device=device)
    arange = torch.arange(r, device=device)[:, None]
    dirs = []
    statuses = []
    for geometry, optimizer, seed in cases:
        d = out / geometry / optimizer / f"seed{seed}"
        refuse_nonempty(d)
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
        statuses.append({"geometry": geometry, "optimizer": optimizer, "seed": seed,
                         "status": "RUNNING", "last_saved_task": 0, "first_nonfinite_task": None})
    for i, d in enumerate(dirs):
        save_snapshot(d / "t000.npz", P, M, V, i, 0, 0, cases[i][1])
        atomic_json(d / "status.json", statuses[i])

    def train_step():
        xb, yb = X[arange, static_idx], Y[arange, static_idx]
        _, _, logits = forward(P, xb)
        loss = F.cross_entropy(logits.reshape(-1, 10), yb.reshape(-1), reduction="none").view(r, BATCH).mean(1)
        correct_sum.add_((logits.detach().argmax(-1) == yb).float().mean(1))
        grads = torch.autograd.grad(torch.where(live_mask, loss, torch.zeros_like(loss)).sum(), P)
        update_in_place(P, M, V, grads, adam_mask, live_mask, step)

    use_graph = graph and device.type == "cuda"
    cg = None
    if use_graph:
        # Warm-up and capture each execute a step. Restore every touched tensor;
        # batch indices/labels and CPU generators have not yet been drawn.
        state = [q.detach().clone() for q in (*P, *M, *V, step, correct_sum)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                train_step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            train_step()
        with torch.no_grad():
            for q, old in zip((*P, *M, *V, step, correct_sum), state):
                q.copy_(old)
        del state
    batch_g = {seed: H.stream("rlc_batch", seed) for seed in seeds}
    start_time = time.monotonic()
    for task in range(1, tasks + 1):
        Y.copy_(torch.stack([labels[seed][task - 1] for _, _, seed in cases]).to(device))
        start = evaluate(P, X, Y)
        correct_sum.zero_()
        for _ in range(epochs):
            perms = {seed: torch.randperm(N_IMAGES, generator=batch_g[seed]) for seed in seeds}
            orders = torch.stack([perms[seed] for _, _, seed in cases]).to(device)
            for bi in range(N_IMAGES // BATCH):
                static_idx.copy_(orders[:, bi * BATCH:(bi + 1) * BATCH])
                if cg is None:
                    train_step()
                else:
                    cg.replay()
        end = evaluate(P, X, Y)
        finite = case_finite(P, M, V)
        finite &= torch.isfinite(end["ce"])
        for i, (geometry, optimizer, seed) in enumerate(cases):
            if statuses[i]["status"] == "DIVERGED":
                continue
            ok = bool(finite[i].item())
            row = {"geometry": geometry, "optimizer": optimizer, "seed": seed, "task": task,
                   "start_acc": float(start["acc"][i]), "end_acc": float(end["acc"][i]) if ok else math.nan,
                   "online_acc": float(correct_sum[i] / (epochs * (N_IMAGES // BATCH))) if ok else math.nan,
                   "start_ce": float(start["ce"][i]), "end_ce": float(end["ce"][i]) if ok else math.nan,
                   "status": "OK" if ok else "DIVERGED"}
            for key in ("mean_abs_dphi_l1", "mean_abs_dphi_l2", "activeunit_frac_l1",
                        "activeunit_frac_l2", "positive_gate_frac_l1", "positive_gate_frac_l2"):
                row[key] = float(end[key][i]) if ok else math.nan
            all_rows.append(row)
            if ok:
                save_snapshot(dirs[i] / f"t{task:03d}.npz", P, M, V, i, task,
                              task * epochs * (N_IMAGES // BATCH), optimizer)
                statuses[i]["last_saved_task"] = task
                if task == tasks:
                    statuses[i]["status"] = "COMPLETED"
            else:
                statuses[i]["status"] = "DIVERGED"
                statuses[i]["first_nonfinite_task"] = task
                # From the next task, this slot is inert and finite. Its missing
                # snapshots and explicit DIVERGED status preserve the failed path.
                live_mask[i] = False
                with torch.no_grad():
                    for q in (*P, *M, *V):
                        q[i].zero_()
            atomic_json(dirs[i] / "status.json", statuses[i])
        atomic_csv(out / "per_task.csv", all_rows)
        print(f"{group} task {task}/{tasks}, {time.monotonic() - start_time:.1f}s, "
              f"finite {int(finite.sum())}/{r}", flush=True)


def run(out: Path, archive: Path, seeds: tuple[int, ...], tasks: int, epochs: int,
        device: torch.device, graph: bool, source_git_hash: str | None = None,
        groups: tuple[tuple[str, ...], ...] = GROUPS) -> None:
    refuse_nonempty(out)
    if source_git_hash is not None and git_head() != source_git_hash:
        raise RuntimeError(f"HEAD {git_head()} != requested source hash {source_git_hash}")
    if not archive.exists():
        raise FileNotFoundError(archive)
    started_at = utc_now()
    out.mkdir(parents=True, exist_ok=True)
    labels, base, _, archive_hash = make_data(archive, seeds, tasks, out)
    metadata = {"experiment": EXPERIMENT, "git_hash": source_git_hash or git_head(),
                "started_at_utc": started_at, "finished_at_utc": None,
                "source_sha256": sha256(SOURCE), "spec_sha256": sha256(SPEC),
                "archive": str(archive), "archive_sha256": archive_hash,
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device": str(device), "graph": bool(graph and device.type == "cuda"),
                "seeds": list(seeds), "tasks": tasks, "epochs": epochs,
                "batch": BATCH, "images_per_seed": N_IMAGES,
                "steps_per_task": epochs * (N_IMAGES // BATCH),
                "geometry_dims": GEOMETRIES, "geometries": list(GEOMETRIES),
                "optimizers": list(OPTIMIZERS),
                "args": {"seeds": list(seeds), "tasks": tasks, "epochs": epochs,
                         "batch": BATCH, "group_names": [list(g) for g in groups]},
                "cases": [{"geometry": g, "optimizer": o, "seed": s,
                           "dims": [GEOMETRIES[g], 100, 100, 10]}
                          for group in groups for g, o, s in group_cases(group, seeds)],
                "geometry_definitions": {"rgb32": "raw uint8/255 CHW",
                                         "dup64": "2x2 spatial repeat of rgb32",
                                         "dup64_half": "dup64 / 2",
                                         "gray32": "0.299R+0.587G+0.114B",
                                         "avg16": "nonoverlapping 2x2 spatial average"},
                "optimizer_settings": {"adam": {"lr": ADAM_LR, "betas": [BETA1, BETA2], "eps": EPS},
                                       "sgd": {"lr": SGD_LR, "momentum": 0}},
                "rng_streams": {"subset": "H.stream('rlc_subset', seed) via RC.subset_idx",
                                "labels": "H.stream('rlc_labels', seed), one RC.task_labels per task",
                                "batch": "H.stream('rlc_batch', seed), one randperm(1200) per epoch",
                                "init": "H.init_params(seed, CPU, (3072,100,100,10)); native first W from H.stream('init_native_geometry_'+geometry,seed)"},
                "bank_layout": "banks/{geometry}/seed{seed}.npz x float32 [1200,d] and indices int64 [1200]",
                "label_layout": "labels/seed{seed}.npz labels int64 [tasks,1200]",
                "snapshot_layout": "{geometry}/{optimizer}/seed{seed}/t000..tNNN.npz; W1,b1,W2,b2,W3,b3; adam_m_*,adam_v_*; task,step,optimizer_step",
                "sgd_moment_encoding": "scalar float32 zero for every adam_m_* and adam_v_* key",
                "metric_definitions": {"start_acc/start_ce": "full fixed bank, new task labels, before first update",
                                       "online_acc": "mean pre-update batch accuracy",
                                       "end_acc/end_ce": "full fixed bank after final update",
                                       "activeunit_frac": "fraction of units with any |phi'|>1e-6 on full bank",
                                       "positive_gate_frac": "fraction of image-unit preactivations >0"}}
    atomic_json(out / "metadata.json", metadata)
    rows: list[dict] = []
    atomic_csv(out / "per_task.csv", rows)
    for group in groups:
        run_group(group, seeds, tasks, epochs, out, labels, base, device, rows, graph)
    statuses = []
    for group in groups:
        for geometry, optimizer, seed in group_cases(group, seeds):
            f = out / geometry / optimizer / f"seed{seed}" / "status.json"
            statuses.append(json.loads(f.read_text()))
    failed = [s for s in statuses if s["status"] != "COMPLETED"]
    finished_at = utc_now()
    metadata["finished_at_utc"] = finished_at
    atomic_json(out / "metadata.json", metadata)
    atomic_json(out / "DONE.json", {"status": "SUCCESS" if not failed else "FAILED",
                                   "git_hash": metadata["git_hash"],
                                   "source_sha256": metadata["source_sha256"],
                                   "source_hash": metadata["git_hash"],
                                   "started_at_utc": started_at, "finished_at_utc": finished_at,
                                   "expected_cases": len(statuses),
                                   "completed_cases": len(statuses) - len(failed),
                                   "failed_cases": failed})
    if failed:
        raise SystemExit(f"{len(failed)} of {len(statuses)} series diverged or did not complete")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 305)))
    p.add_argument("--tasks", type=int, default=TASKS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--source-git-hash")
    p.add_argument("--group", choices=("all", "rgb32", "dup64", "gray32", "avg16"), default="all")
    a = p.parse_args()
    if not a.seeds or len(set(a.seeds)) != len(a.seeds) or a.tasks < 1 or a.epochs < 1:
        p.error("seeds unique and nonempty; tasks and epochs positive")
    groups = GROUPS if a.group == "all" else (next(g for g in GROUPS if g[0] == a.group),)
    run(a.out, a.archive, tuple(a.seeds), a.tasks, a.epochs, torch.device(a.device),
        not a.no_graph, a.source_git_hash, groups)


if __name__ == "__main__":
    main()
