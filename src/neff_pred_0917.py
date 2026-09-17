#!/usr/bin/env python3
"""Does the functional response n_eff predict relearning outside resp_ee_0917's box?

    python3 -m src.neff_pred_0917 gpu --out results/neff_pred_0917/gpu            # GELU / SiLU / ReLU / chimeras
    OMP_NUM_THREADS=1 python3 -m src.neff_pred_0917 ee --seed 10 --out results/neff_pred_0917/ee/s10

specs/spec_neff_pred_0917.md (registered at PREREG_COMMIT).  No intervention: every box runs its natural
trajectory, and the branch at task t is the net that starts task t+1 -- with Adam carried on, "branch and
learn one new task" is the natural task t+1 itself, so E(t) is task t+1's online accuracy and the
predictor is read at task t+1's start, on task t+1's inputs (in RL the images never change, so this is
also task t's end; in PM the new permutation is already in place).

gpu: the two batched engines that made the records, imported unchanged and run side by side on the same
     per-seed draws, 150 tasks:
       B = relu_gelu_silu_rl_0914.Engine (its 30 models: RL/PM x LR, ELU1, R, GELU, SILU x seeds 0-2;
           tasks 1-150 are relu_gelu_silu_rl_ext150_0914's rows bit for bit, check S0a)
       L = layer_chimera_rl_0914.Engine (its 24 models: RL/PM x EE, EL, LE, LL x seeds 0-2;
           tasks 1-50 are that run's rows bit for bit; 51-150 are new)
     The derivative each engine's backward multiplies by is its own gate(); the statistics below read that
     tensor (for ELU in these engines it is exp(z), not expm1 + 1).
ee:  resp_ee_0917's natural prefix (the calibration box: RL, ELU->ELU, expm1 derivative, CPU) for seeds the
     calibration never saw, tasks 1-51, through its own run_prefix (checks S0b).

Per unit i of a layer, g_i(x) the derivative training uses on image x (N images):
    neff_abs_i = (sum_x |g_i|)^2 / (N sum_x g_i^2),   neff_sgn_i = (sum_x g_i)^2 / (N sum_x g_i^2),
0 for a unit with g = 0 on every image; a layer's value is the unit mean.  For ELU, ReLU and leaky g >= 0
and the two agree (resp_ee's neffT is this quantity); GELU and SiLU have a negative lobe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import relu_gelu_silu_rl_0914 as B      # engine, activations, streams, data; not touched
from src import layer_chimera_rl_0914 as L       # engine for the per-layer chimeras; not touched

EXPERIMENT = "neff_pred_0917"
SPEC = "specs/spec_neff_pred_0917.md"
PREREG_COMMIT = "f829fc2dfe3805ce3fa92f64a5f40da63724e141"   # specs/spec_neff_pred_0917.md, pushed before the main run
ROOT = Path(__file__).resolve().parents[1]
TASKS = 150
EE_TASKS = 51
N_IMG = B.N
STEPS = B.STEPS
EXT150_COMMIT = "fca473d"                         # relu_gelu_silu_rl_ext150_0914 results (rows.csv)
CHIMERA_COMMIT = "58c1819"                        # layer_chimera_rl_0914 results (rows.csv)
STAT_KEYS = ("neff_abs", "neff_sgn", "gabs", "gsgn", "zero", "tiny", "neg", "dead")
UNIT_KEYS = ("neff_abs", "neff_sgn", "gabs", "zero")          # saved per unit at each task start
TINY = 1e-8                                                    # Adam's eps (report only)


def model_key(m: dict) -> tuple[str, str]:
    """(act1, act2) for either engine's model dict."""
    return (m["act"], m["act"]) if "act" in m else (m["act1"], m["act2"])


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------

@torch.no_grad()
def response_stats(g: torch.Tensor) -> dict[str, torch.Tensor]:
    """g: [M, N, U] float32, the derivative the engine's backward uses.  Returns [M, U] float64 arrays."""
    n = g.shape[1]
    g64 = g.double()
    ga = g64.abs()
    s1a, s1s, s2 = ga.sum(1), g64.sum(1), g64.square().sum(1)
    live = s2 > 0
    den = (n * s2).clamp_min(1e-300)
    zero = torch.zeros_like(s2)
    return {"neff_abs": torch.where(live, s1a.square() / den, zero),
            "neff_sgn": torch.where(live, s1s.square() / den, zero),
            "gabs": ga.mean(1), "gsgn": g64.mean(1),
            "zero": (g == 0).double().mean(1), "tiny": (g.abs() < TINY).double().mean(1),
            "neg": (g < 0).double().mean(1), "dead": (~live).double()}


class Box:
    """One engine, its forward and its gate, and the models it carries."""

    def __init__(self, name: str, mod, models: list[dict], device: str):
        self.name, self.mod, self.models = name, mod, models
        self.e = mod.Engine(models=models, device=device)

    def fields(self):
        e = self.e
        if self.mod is B:
            z1, a1, z2, a2, lg = B.forward(e.p, e.cx, e.A)
            return (z1, B.gate(z1, e.A)), (z2, B.gate(z2, e.A))
        z1, a1, z2, a2, lg = L.forward(e.p, e.cx, e.elu1, e.elu2)
        return (z1, L.gate(z1, e.elu1)), (z2, L.gate(z2, e.elu2))

    @torch.no_grad()
    def stats(self) -> tuple[dict, dict]:
        """(unit means per model: {f'{k}_l{l}': np[M]}, per-unit arrays {f'{k}_l{l}': np[M, U]})."""
        means, units = {}, {}
        for li, (z, g) in enumerate(self.fields(), start=1):
            st = response_stats(g)
            for k, v in st.items():
                means[f"{k}_l{li}"] = v.mean(1).cpu().numpy()
                if k in UNIT_KEYS:
                    units[f"{k}_l{li}"] = v.cpu().numpy().astype(np.float32)
            means[f"zbar_l{li}"] = z.double().mean(dim=(1, 2)).cpu().numpy()
        return means, units


# --------------------------------------------------------------------------
# gpu: the task loop of relu_gelu_silu_rl_ext150_0914.run / layer_chimera_rl_0914.run, both engines
# --------------------------------------------------------------------------

def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", f"analysis/{EXPERIMENT}", SPEC],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_hash": head, "git_dirty_code": dirty}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_gpu(out: Path, tasks: int = TASKS, device: str = "cuda", boxes=("B", "L"), progress=True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    start_state = git_state()                      # read before the first update (pitfall: commits mid-run)
    t_start = time.time()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    bx = [Box(n, {"B": B, "L": L}[n], {"B": B.MODELS, "L": L.MODELS}[n], device) for n in boxes]
    dev = bx[0].e.device
    seeds = sorted({m["seed"] for b in bx for m in b.models})
    gperm = {s: B.stream("env_perm_0913", s) for s in seeds}
    glabel = {s: B.stream("env_labels_0913", s) for s in seeds}
    gbatch = {s: B.stream("env_batch_0913", s) for s in seeds}
    ax = B.read_idx(B.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    ay = B.read_idx(B.DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
    subset = {s: torch.randperm(len(ax), generator=B.stream("rl_subset", s))[:N_IMG].numpy() for s in seeds}
    xs = {s: torch.tensor(ax[subset[s]], device=dev) for s in seeds}
    ys_cpu = {s: torch.tensor(ay[subset[s]]) for s in seeds}
    ys = {s: ys_cpu[s].to(dev) for s in seeds}
    rows: list[dict] = []
    units: dict[str, list] = {}
    nonfinite = {b.name: [None] * len(b.models) for b in bx}
    for b in bx:
        b.e.capture()
    t0 = time.monotonic()
    for task in range(1, tasks + 1):
        epochs_needed = -(-(STEPS * B.BATCH) // N_IMG)
        orders = {}
        for s in seeds:
            flat = torch.stack([torch.randperm(N_IMG, generator=gbatch[s]) for _ in range(epochs_needed)]
                               ).reshape(-1)[:STEPS * B.BATCH]
            orders[s] = flat.reshape(STEPS, B.BATCH)
        perm_cpu = {s: torch.randperm(784, generator=gperm[s]) for s in seeds}
        perm = {s: perm_cpu[s].to(dev) for s in seeds}
        labels_cpu = {s: torch.randint(10, (N_IMG,), generator=glabel[s]) for s in seeds}
        labels = {s: labels_cpu[s].to(dev) for s in seeds}
        floors = {}
        for s in seeds:
            floors[s, "PM"] = float(torch.bincount(ys_cpu[s], minlength=10).max()) / N_IMG
            floors[s, "RL"] = float(torch.bincount(labels_cpu[s], minlength=10).max()) / N_IMG
        for b in bx:
            e = b.e
            orderstack = torch.stack([orders[m["seed"]] for m in b.models], 1).to(dev)
            for j, m in enumerate(b.models):
                s = m["seed"]
                e.cx[j].copy_(xs[s][:, perm[s]] if m["env"] == "PM" else xs[s])
                yy = ys[s] if m["env"] == "PM" else labels[s]
                e.cy[j].copy_(torch.nn.functional.one_hot(yy, 10))
            st0, un0 = b.stats()                   # task start: this task's inputs, weights not yet moved
            e.acc.zero_()
            e.ce.zero_()
            for step in range(0, STEPS, B.BLOCK):
                e.indices.copy_(orderstack[step:step + B.BLOCK])
                e.replay_block()
            met, _ = e.evaluate()
            st1, _ = b.stats()                     # task end
            online_acc = (e.acc / STEPS).cpu().numpy()
            online_ce = (e.ce / STEPS).cpu().numpy()
            ok = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in e.p]).all(0).cpu().numpy()
            fin = [bool(ok[j]) and bool(np.isfinite(online_ce[j])) for j in range(len(b.models))]
            for j, m in enumerate(b.models):
                if not fin[j] and nonfinite[b.name][j] is None:
                    nonfinite[b.name][j] = task
                a1, a2 = model_key(m)
                rows.append({"engine": b.name, "seed": m["seed"], "env": m["env"], "act1": a1, "act2": a2,
                             "task": task, "floor_acc": floors[m["seed"], m["env"]],
                             "online_acc": float(online_acc[j]), "online_ce": float(online_ce[j]),
                             "finite": fin[j],
                             **{k: float(v[j]) for k, v in met.items()},
                             **{f"start_{k}": float(v[j]) for k, v in st0.items()},
                             **{f"end_{k}": float(v[j]) for k, v in st1.items()}})
            for k, v in un0.items():
                units.setdefault(f"{b.name}_start_{k}", []).append(v)
        if task % 10 == 0 or task == tasks:
            write_csv(out / "rows.csv", rows)
        if progress:
            smp = {f"{r['env']}-{r['act1']}/{r['act2']}": r["online_acc"] for r in rows
                   if r["task"] == task and r["seed"] == 0 and r["engine"] == "B" and r["act1"] == "GELU"}
            print(f"TASK {task}/{tasks} elapsed={time.monotonic() - t0:.1f}s s0 "
                  + " ".join(f"{k}={v:.3f}" for k, v in smp.items()), flush=True)
    write_csv(out / "rows.csv", rows)
    np.savez_compressed(out / "units.npz", **{k: np.stack(v) for k, v in units.items()})
    dname = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
    data_paths = [B.DATA / n for n in B.DATA_FILES]
    prov = {"experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
            "spec_sha256": sha(ROOT / SPEC) if (ROOT / SPEC).exists() else None,
            **start_state, "git_state_read": "at start",
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": dname,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tasks": tasks, "steps_per_task": STEPS, "images": N_IMG, "batch": B.BATCH,
            "engines": {b.name: {"module": b.mod.__name__, "models": b.models} for b in bx},
            "nonfinite_first_task": nonfinite,
            "code_sha256": {"src/neff_pred_0917.py": sha(Path(__file__)),
                            "src/relu_gelu_silu_rl_0914.py": sha(Path(B.__file__)),
                            "src/layer_chimera_rl_0914.py": sha(Path(L.__file__))},
            "data_sha256": {p.name: sha(p) for p in data_paths},
            "subset_sha256": {s: B.arrsha(v) for s, v in subset.items()},
            "wall_seconds": time.time() - t_start,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out} ({len(rows)} rows, {time.time() - t_start:.1f}s)", flush=True)
    return prov


# --------------------------------------------------------------------------
# ee: resp_ee_0917's natural prefix for unused seeds
# --------------------------------------------------------------------------

def run_ee(seed: int, out: Path, tasks: int = EE_TASKS, progress=True) -> dict:
    from src import pmnist_0905 as H
    from src import resp_ee_0917 as RE
    out.mkdir(parents=True, exist_ok=True)
    start_state = git_state()
    t_start = time.time()
    mnist = H.Mnist(torch.device("cpu"))
    rows, units, _, _, info = RE.run_prefix(seed, mnist, RE.EPOCHS, tasks, save_at=(), progress=progress)
    write_csv(out / "prefix.csv", rows)
    np.savez_compressed(out / "units.npz", **units)
    prov = {"experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
            "spec_sha256": sha(ROOT / SPEC) if (ROOT / SPEC).exists() else None,
            **start_state, "git_state_read": "at start",
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch": torch.__version__, "python": sys.version.split()[0],
            "threads": torch.get_num_threads(), "flush_denormal": RE._flush_is_on(),
            "seed": seed, "tasks": tasks, "epochs_per_task": RE.EPOCHS, "box": "resp_ee_0917 natural prefix",
            "code_sha256": {"src/neff_pred_0917.py": sha(Path(__file__)),
                            "src/resp_ee_0917.py": sha(Path(RE.__file__)),
                            "src/mucap_el_run_0916.py": sha(ROOT / "src" / "mucap_el_run_0916.py"),
                            "src/pmnist_0905.py": sha(ROOT / "src" / "pmnist_0905.py"),
                            "src/pmnist_rlmnist_0906.py": sha(ROOT / "src" / "pmnist_rlmnist_0906.py"),
                            "src/elu_growth_0909.py": sha(ROOT / "src" / "elu_growth_0909.py")},
            "data_sha256": mnist.sha256, "prefix_info": info,
            "seconds_total": time.time() - t_start,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if progress:
        print(f"wrote {out} ({len(rows)} rows, {time.time() - t_start:.1f}s)", flush=True)
    return prov


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpu")
    g.add_argument("--out", required=True)
    g.add_argument("--tasks", type=int, default=TASKS)
    g.add_argument("--device", default="cuda")
    e = sub.add_parser("ee")
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tasks", type=int, default=EE_TASKS)
    e.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        run_gpu(Path(args.out), args.tasks, args.device)
    else:
        from src import pmnist_0905 as H
        from src import resp_ee_0917  # noqa: F401  its import chain (elu_growth_0909) sets torch threads to 1,
        torch.set_num_threads(args.threads)   # so it is imported before --threads is applied
        torch.set_flush_denormal(True)
        H.setup("cpu")
        run_ee(args.seed, Path(args.out), args.tasks)


if __name__ == "__main__":
    main()
