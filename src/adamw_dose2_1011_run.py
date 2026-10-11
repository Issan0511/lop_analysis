#!/usr/bin/env python3
"""adamw_dose2_1011 CLI.

    # A1 (GPU, R = 10 stacked, two threads): RL-CIFAR x ELU->ELU / std, AdamW from cap_cifar_ee_0920's task-1 prefix
    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.adamw_dose2_1011_run rlcifar --arm wd1e0 \
        --out results/adamw_dose2_1011/rlcifar/wd1e0
    # A2 (GPU, R = 10 stacked, two threads): 5+1 CIFAR x MLP x ReLU, coupled L2 toward 0
    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.adamw_dose2_1011_run c51 --arm l2c1e-2 \
        --seeds 0-9 --out results/adamw_dose2_1011/c51/l2c1e-2

specs/spec_adamw_dose2_1011.md (registered at PREREG_COMMIT).  A1 arms (spec 2.2): wd3e-2, wd1e-1, wd1e0 (one lambda
on W1 b1 W2 b2 W3 b3, tasks 2-50); check-only: ref0 (lambda 0) and ro1 (W3, b3 only).  A2 arms (spec 2.4): l2c1e-2,
l2c5e-3; check-only: nochange, mut_after1e-2, dec2e-2.

An output under results/adamw_dose2_1011/ (the registered place) is refused unless checks.json says all_pass for every
required check on these very sources, the tree is clean (src, analysis, specs), the configuration is the registered
one and the arm and seeds are registered for that place (spec 6 S-cli).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src import adamw_dose2_1011 as A1             # A1 engine
from src import adamw_dose2_1011_c51 as A2         # A2 engine
from src import pmnist_0905 as H                   # setup; not touched
from src import rlcifar_mlp_battle_0918 as B       # parse_ints; not touched

EXPERIMENT = "adamw_dose2_1011"
SPEC = "specs/spec_adamw_dose2_1011.md"
PREREG_COMMIT = A1.PREREG_COMMIT
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "results" / EXPERIMENT
CHECKS_JSON = OUT_ROOT / "checks.json"
SOURCES = ("src/adamw_dose2_1011.py", "src/adamw_dose2_1011_c51.py", "src/adamw_dose2_1011_run.py",
           "analysis/adamw_dose2_1011/checks.py", "analysis/adamw_dose2_1011/rule.py",
           "analysis/adamw_dose2_1011/lines.py")
REQUIRED_CHECKS = ("S0", "S-wd", "S-torch", "S-branch", "S-ctrl", "S-graph", "S-resume", "S-nochange",
                   "S-host-repro", "S-coupled-host", "S-mut", "S-graph-c51", "S-fresh-c51", "S-verdict", "S-cli",
                   "S-cost", "S-line")
A1_REGISTERED = dict(tasks=A1.TASKS, epochs=A1.EPOCHS, graph=True, threads=2, seeds=list(A1.SEEDS))
A1_ARMS = A1.REGISTERED_ARMS
C51_SEEDS = {"l2c1e-2": [list(range(10))], "l2c5e-3": [list(range(10))]}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def peak_rss_kb() -> int:
    """VmHWM of this address space (ru_maxrss survives exec and so carries a forking parent's peak)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def admit(box: str, arm: str, seeds: list[int], run_cfg: dict, registered: dict, checks_json: Path = CHECKS_JSON,
          repo: Path = REPO) -> dict:
    """Refuse a registered output unless the checks passed on these sources, the tree is clean and the run is a
    registered one (spec 6 S-cli)."""
    bad = [k for k, v in registered.items() if run_cfg.get(k) != v]
    if bad:
        raise SystemExit(f"ABORT: not the registered configuration: {bad}")
    if box == "rlcifar":
        if arm not in A1_ARMS or seeds != list(A1.SEEDS):
            raise SystemExit(f"ABORT: A1 registers arms {A1_ARMS} on seeds 0-9, not {arm} {seeds}")
    else:
        if arm not in C51_SEEDS or seeds not in C51_SEEDS[arm]:
            raise SystemExit(f"ABORT: A2 registers {C51_SEEDS}, not {arm} {seeds}")
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


def main_rlcifar(args) -> None:
    seeds = B.parse_ints(args.seeds)
    out = Path(args.out)
    torch.set_num_threads(args.threads)
    run_cfg = dict(tasks=args.tasks, epochs=args.epochs, graph=not args.no_graph, threads=torch.get_num_threads(),
                   seeds=seeds)
    if args.prefix is not None and _registered_place(out):
        raise SystemExit("ABORT: --prefix is for check runs; the registered arms branch from cap_cifar_ee_0920's prefix")
    admission = admit("rlcifar", args.arm, seeds, run_cfg, A1_REGISTERED) if _registered_place(out) else None
    prefix = None if args.prefix is None else torch.load(args.prefix, map_location="cpu", weights_only=False)
    t0 = time.time()
    done = A1.run_arm(args.arm, out, seeds=seeds, tasks=args.tasks, epochs=args.epochs, graph=not args.no_graph,
                      admission=admission, prefix=prefix)
    extra = {"cli_sha256": sha256_file(Path(__file__)), "spec_sha256": sha256_file(REPO / SPEC),
             "checks_sha256": sha256_file(CHECKS_JSON) if CHECKS_JSON.exists() else None,
             "admission": admission, "completed": bool(done), "wall_clock_this_invocation_s": time.time() - t0,
             "peak_rss_kb": peak_rss_kb(), "torch_threads": torch.get_num_threads(),
             "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved()}
    (out / f"cli_{int(time.time())}.json").write_text(json.dumps(extra, indent=2))
    if not done:
        print("STOP acknowledged; checkpoint saved.", flush=True)


def main_c51(args) -> None:
    torch.set_num_threads(args.threads)
    seeds = B.parse_ints(args.seeds)
    out = Path(args.out)
    run_cfg = dict(tasks=args.tasks, lr=args.lr, steps_hard=args.steps_hard, steps_easy=args.steps_easy,
                   fresh=not args.no_fresh, graph=not args.no_graph, threads=torch.get_num_threads())
    admission = admit("c51", args.arm, seeds, run_cfg, A2.REGISTERED) if _registered_place(out) else None
    device = H.setup(args.device)
    prov = A2.run(args.arm, seeds, run_cfg["tasks"], device, out, lr=run_cfg["lr"], fresh=run_cfg["fresh"],
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
    r = sub.add_parser("rlcifar")
    r.add_argument("--arm", required=True, choices=tuple(A1.ARM_LAMBDA))
    r.add_argument("--seeds", default="0-9")
    r.add_argument("--tasks", type=int, default=A1.TASKS)
    r.add_argument("--epochs", type=int, default=A1.EPOCHS)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--no-graph", action="store_true")
    r.add_argument("--prefix", default=None, help="check runs only: a task-1 state saved by torch.save")
    r.add_argument("--out", required=True)
    c = sub.add_parser("c51")
    c.add_argument("--arm", required=True, choices=tuple(A2.ARM_SPEC))
    c.add_argument("--seeds", default="0-9")
    c.add_argument("--tasks", type=int, default=A2.C.N_TASKS)
    c.add_argument("--lr", type=float, default=A2.LR)
    c.add_argument("--steps-hard", type=int, default=A2.C.STEPS_PER_TASK)
    c.add_argument("--steps-easy", type=int, default=A2.C.STEPS_PER_TASK)
    c.add_argument("--no-fresh", action="store_true")
    c.add_argument("--no-graph", action="store_true", help="eager steps (checks)")
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="auto")
    c.add_argument("--threads", type=int, default=A2.THREADS)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.box == "rlcifar":
        main_rlcifar(args)
    else:
        main_c51(args)


if __name__ == "__main__":
    main()
