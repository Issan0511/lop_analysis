#!/usr/bin/env python3
"""adamw_dose_1010 CLI: AdamW (decoupled weight decay, all six tensors) on two boxes.

    # box 1 (CPU, one thread): Random Label MNIST, ELU -> ELU, 100 tasks x 80 epochs
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
        -m src.adamw_dose_1010_run mnist --arm wd3e-2 --seeds 0 --out results/adamw_dose_1010/mnist/runs/wd3e-2_s0
    # box 2 (GPU, R = 10 stacked, two threads): 5+1 CIFAR x MLP x ReLU, 30 tasks
    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
        -m src.adamw_dose_1010_run c51 --arm adamw10 --seeds 0-9 --out results/adamw_dose_1010/c51/adamw10

specs/spec_adamw_dose_1010.md (registered at PREREG_COMMIT).  Box 1 arms (spec 2.2): ref (lambda 0), wd1e-3,
wd1e-2, wd3e-2, wd1e-1 (one lambda on W1 b1 W2 b2 W3 b3).  Box 2 arms (spec 2.3): adamw5, adamw10, and the
lambda = 0 path `nochange` the reuse checks run.  The engines are src/adamw_dose_1010.py and
src/adamw_dose_1010_c51.py.

An output under results/adamw_dose_1010/ (the registered place) is refused unless checks.json says all_pass for
every required check on these very sources, the tree is clean (src, analysis, specs), the configuration is the
registered one and the arm and seeds are registered for that place (spec 6 S9).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from src import pmnist_0905 as H                 # host; not touched
from src import shell_l2_rlmnist_0913 as SH      # file_sha256, git_dirty; not touched
from src import adamw_dose_1010 as E1            # box 1 engine
from src import adamw_dose_1010_c51 as E2        # box 2 engine
from src import rlcifar_mlp_battle_0918 as B     # parse_ints; not touched

EXPERIMENT = "adamw_dose_1010"
SPEC = "specs/spec_adamw_dose_1010.md"
PREREG_COMMIT = E1.PREREG_COMMIT
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "results" / EXPERIMENT
CHECKS_JSON = OUT_ROOT / "checks.json"
SOURCES = ("src/adamw_dose_1010.py", "src/adamw_dose_1010_c51.py", "src/adamw_dose_1010_run.py",
           "analysis/adamw_dose_1010/checks.py")
REQUIRED_CHECKS = ("S0", "S-wd", "S-torch", "S-deadpix", "S5", "S-ctrl", "S7", "S-nochange", "S-host-repro",
                   "S-graph", "S-fresh", "S8", "S9")
# box 1: arm -> lambda (all six tensors); box 2: arm -> lambda
ARM_MAP = {"ref": 0.0, "wd1e-3": 1e-3, "wd1e-2": 1e-2, "wd3e-2": 3e-2, "wd1e-1": 1e-1}
ARMS = tuple(ARM_MAP)
C51_ARMS = tuple(E2.ARM_LAMBDA)                   # adamw5, adamw10 and the lambda = 0 path; lambdas in E2
TASKS = 100
EPOCHS = E1.EPOCHS
MNIST_REGISTERED = dict(tasks=TASKS, epochs=EPOCHS, lr=E1.LR, threads=1)
C51_SEEDS = {"adamw5": [list(range(10))], "adamw10": [list(range(10)), list(range(100, 110))],
             "nochange": [list(range(10))]}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def peak_rss_kb() -> int:
    """VmHWM of this address space (getrusage's ru_maxrss survives exec and so carries a forking parent's peak)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def admit(box: str, arm: str, seeds: list[int], run_cfg: dict, registered: dict, checks_json: Path = CHECKS_JSON,
          repo: Path = REPO) -> dict:
    """Refuse a registered output unless the checks passed on these sources, the tree is clean and the run is a
    registered one (spec 6 S9)."""
    bad = [k for k, v in registered.items() if run_cfg.get(k) != v]
    if bad:
        raise SystemExit(f"ABORT: not the registered configuration: {bad}")
    if box == "mnist":
        if arm not in ARM_MAP or not all(0 <= s <= 9 for s in seeds):
            raise SystemExit(f"ABORT: box 1 registers arms {ARMS} on seeds 0-9, not {arm} {seeds}")
    else:
        if arm not in C51_SEEDS or seeds not in C51_SEEDS[arm]:
            raise SystemExit(f"ABORT: box 2 registers {C51_SEEDS}, not {arm} {seeds}")
    if not checks_json.exists():
        raise SystemExit(f"ABORT: {checks_json} missing")
    ck = json.loads(checks_json.read_text())
    missing = [k for k in REQUIRED_CHECKS if k not in ck]
    if missing or not ck.get("all_pass"):
        raise SystemExit(f"ABORT: checks not all_pass (missing {missing})")
    now = {s: sha256_file(repo / s) for s in SOURCES}
    if ck.get("source_sha256") != now:
        raise SystemExit("ABORT: sources changed since the checks ran")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src", "analysis", "specs"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted code or spec:\n{dirty}")
    return {"checks_sha256": sha256_file(checks_json), "source_sha256": now}


def _registered_place(out: Path) -> bool:
    return out.resolve().is_relative_to(OUT_ROOT.resolve())


# --------------------------------------------------------------------------
# box 1
# --------------------------------------------------------------------------

def main_mnist(args) -> None:
    torch.set_num_threads(args.threads)               # after every import (elu_growth's chain sets 1)
    torch.set_flush_denormal(True)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    run_cfg = dict(tasks=args.tasks, epochs=args.epochs, lr=args.lr, threads=torch.get_num_threads())
    admission = admit("mnist", args.arm, seeds, run_cfg, MNIST_REGISTERED) if _registered_place(out) else None
    device = H.setup(args.device)
    mnist = H.Mnist(device)
    out.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    t0 = time.time()
    all_rows, all_wd, all_arrays, infos = [], [], {}, {}
    lam = ARM_MAP[args.arm]
    lams = (lam,) * 6
    for seed in seeds:
        rows, arrays, wd_rows, info = E1.run_adamw(seed, args.lr, args.tasks, mnist, device, lams=lams,
                                                   arm=args.arm, epochs=args.epochs, progress=True)
        for r in rows:
            r["arm"] = args.arm
        for r in wd_rows:
            r["arm"] = args.arm
        info["divergence"]["arm"] = args.arm
        all_rows += rows
        all_wd += wd_rows
        infos[str(seed)] = info
        for k, v in arrays.items():
            all_arrays[f"s{seed}_{k}"] = v
    for name, rows_ in (("per_task.csv", all_rows), ("wd_task.csv", all_wd)):
        keys = sorted({k for r in rows_ for k in r})
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "seed", "task"] +
                               [k for k in keys if k not in ("arm", "seed", "task")])
            w.writeheader()
            w.writerows(rows_)
    np.savez_compressed(out / "units.npz", **all_arrays)
    root = me.parents[1]
    (out / "provenance.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "box": "mnist", "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(root / SPEC) if (root / SPEC).exists() else None,
        "git_hash": H.git_hash(),
        "git_dirty_code": SH.git_dirty(["src", "analysis/adamw_dose_1010", SPEC]),
        "admission": admission,
        "checks_sha256": sha256_file(CHECKS_JSON) if CHECKS_JSON.exists() else None,
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "python": sys.version.split()[0], "device": args.device,
        "threads": args.threads, "torch_threads": torch.get_num_threads(),
        "arm": args.arm, "lam": lam, "lams": list(lams), "c_hex": float(E1.c32(args.lr, lam)).hex(),
        "seeds": seeds, "lr": args.lr,
        "n_tasks": args.tasks, "epochs_per_task": args.epochs, "batch": E1.BATCH,
        "steps_per_task": E1.STEPS_PER_EPOCH * args.epochs, "n_images": E1.N_IMAGES,
        "act1": "ELU1", "act2": "ELU1", "ledger": False, "flush_denormal": True,
        "code_sha256": {f"src/{n}": SH.file_sha256(me.parent / n) for n in
                        ("adamw_dose_1010_run.py", "adamw_dose_1010.py", "mucap_el_run_0916.py",
                         "mucap_el_0916.py", "l2cap_ee_0917.py", "pmnist_0905.py", "pmnist_rlmnist_0906.py",
                         "elu_growth_0909.py", "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "per_seed": infos,
        "wall_clock_s": time.time() - t0,
        "peak_rss_kb": peak_rss_kb(), "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2, default=str))
    print(f"wrote {out}  ({len(all_rows)} task rows, {time.time() - t0:.1f}s)")


# --------------------------------------------------------------------------
# box 2
# --------------------------------------------------------------------------

def main_c51(args) -> None:
    torch.set_num_threads(args.threads)
    seeds = B.parse_ints(args.seeds)
    out = Path(args.out)
    run_cfg = dict(tasks=args.tasks, lr=args.lr, steps_hard=args.steps_hard, steps_easy=args.steps_easy,
                   fresh=not args.no_fresh, graph=not args.no_graph, threads=torch.get_num_threads())
    admission = admit("c51", args.arm, seeds, run_cfg, E2.REGISTERED) if _registered_place(out) else None
    device = H.setup(args.device)
    prov = E2.run(args.arm, seeds, run_cfg["tasks"], device, out, lr=run_cfg["lr"], fresh=run_cfg["fresh"],
                  graph=run_cfg["graph"], steps_hard=run_cfg["steps_hard"], steps_easy=run_cfg["steps_easy"])
    prov |= {"admission": admission, "spec_sha256": sha256_file(REPO / SPEC),
             "checks_sha256": sha256_file(CHECKS_JSON) if CHECKS_JSON.exists() else None,
             "cli_sha256": sha256_file(Path(__file__)),
             "peak_rss_kb": peak_rss_kb(), "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="box", required=True)
    m = sub.add_parser("mnist")
    m.add_argument("--arm", required=True, choices=ARMS)
    m.add_argument("--seeds", default="0")
    m.add_argument("--lr", type=float, default=E1.LR)
    m.add_argument("--tasks", type=int, default=TASKS)
    m.add_argument("--epochs", type=int, default=EPOCHS)
    m.add_argument("--threads", type=int, default=1)
    m.add_argument("--out", required=True)
    m.add_argument("--device", default="cpu")
    c = sub.add_parser("c51")
    c.add_argument("--arm", required=True, choices=C51_ARMS)
    c.add_argument("--seeds", default="0-9")
    c.add_argument("--tasks", type=int, default=E2.C.N_TASKS)
    c.add_argument("--lr", type=float, default=E2.LR)
    c.add_argument("--steps-hard", type=int, default=E2.C.STEPS_PER_TASK)
    c.add_argument("--steps-easy", type=int, default=E2.C.STEPS_PER_TASK)
    c.add_argument("--no-fresh", action="store_true")
    c.add_argument("--no-graph", action="store_true", help="eager steps (checks)")
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="auto")
    c.add_argument("--threads", type=int, default=E2.THREADS)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.box == "mnist":
        main_mnist(args)
    else:
        main_c51(args)


if __name__ == "__main__":
    main()
