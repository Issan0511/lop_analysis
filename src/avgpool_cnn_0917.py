#!/usr/bin/env python3
"""avgpool_cnn_0917 -- Kumar's Random-Label CIFAR CNN with its two max-pools replaced by
avg-pools (spec: specs/spec_avgpool_cnn_0917.md).

    python3 src/avgpool_cnn_0917.py run --arm SNA_avg --seed 0 --device cuda

The host `src/rlcifar_cnn_0908.py` is imported unchanged and driven through its own
`run_one`.  Only its module-level `forward_cnn` is replaced at run time, by
`forward_pool`: the host function line for line, with the two
`F.max_pool2d(., POOL, POOL)` taking the pool as an argument.  The host's training loop,
`evaluate_cnn` and `preact_hist` all call `forward_cnn` by module name, so the swap
reaches every path (S-wiring).  With pool = max the swapped forward reproduces the
stored reference rows bit for bit (S-reuse), so an avg arm differs from its host arm
in the pool function only.

Arms (host activation, pool):
  SNA_avg   adaptive Snake, c = 0.6, beta = 0.01, clip [0.005, 3], per channel   avg
  SN3_avg   fixed Snake alpha = 3                                                 avg
  SNA_max / SN3_max   the same with max-pool; checks only (they are the reference arms)

`evaluate_cnn` still reports `mob_pool_*`, the gate at the max-pool winner.  In the avg
arms that is not the gate the weights learn through (every position counts, `mob_*`);
it is kept for comparison with the reference.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H              # host; not touched
from src import pmnist_rlcifar_0907 as RC     # CIFAR data layer; not touched
from src import rlcifar_cnn_0908 as CN        # the CNN box; not touched

EXPERIMENT = "avgpool_cnn_0917"
OUT = H.REPO / "results" / EXPERIMENT
TASKS, EPOCHS, LR = 50, 400, 1e-3
POOLS = {"max": F.max_pool2d, "avg": F.avg_pool2d}
ARMS = {"SNA_avg": ("SNA", "avg"), "SN3_avg": ("SN3", "avg"),
        "SNA_max": ("SNA", "max"), "SN3_max": ("SN3", "max")}
MAIN_ARMS = ("SNA_avg", "SN3_avg")
HOST_FILE = H.REPO / "src" / "rlcifar_cnn_0908.py"
_RUN_ONE = inspect.signature(CN.run_one).parameters
SNA_DEFAULTS = {k: _RUN_ONE[k].default for k in ("c", "beta", "alpha_lo", "alpha_hi")}


def forward_pool(params, x, act, pool):
    """rlcifar_cnn_0908.forward_cnn line for line; `pool` replaces F.max_pool2d."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = params
    ada = isinstance(act, CN.ChannelSnake)
    z1 = F.conv2d(x, Wc1, bc1, padding=CN.PAD)
    a1 = act.phi(z1, 0) if ada else act.phi(z1)
    z2 = F.conv2d(pool(a1, CN.POOL, CN.POOL), Wc2, bc2, padding=CN.PAD)
    a2 = act.phi(z2, 1) if ada else act.phi(z2)
    h = pool(a2, CN.POOL, CN.POOL).flatten(1)
    z3 = h @ Wf1.T + bf1
    a3 = act.phi(z3, 2) if ada else act.phi(z3)
    z4 = a3 @ Wf2.T + bf2
    a4 = act.phi(z4, 3) if ada else act.phi(z4)
    return z1, a1, z2, a2, z3, a3, z4, a4, a4 @ Wf3.T + bf3


def use_pool(name: str):
    """Point the host's `forward_cnn` at `forward_pool` with this pool."""
    pool = POOLS[name]

    def forward_cnn(params, x, act):
        return forward_pool(params, x, act, pool)

    forward_cnn.pool = name
    CN.forward_cnn = forward_cnn
    return forward_cnn


def run_arm(arm: str, seed: int, data, device, tasks: int = TASKS, epochs: int = EPOCHS,
            hist_dir: Path | None = None, debug: dict | None = None):
    """One (arm, seed) through the host's run_one.  Rows keep the host arm name."""
    host_arm, pool = ARMS[arm]
    use_pool(pool)
    # c, beta and the alpha clip are the host defaults (0.6, 0.01, [0.005, 3]), the
    # values the reference SNA ran with
    return CN.run_one(host_arm, seed, LR, tasks, data, device, epochs=epochs,
                      hist_dir=hist_dir, debug=debug)


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(H.REPO), *a], capture_output=True,
                              text=True).stdout.strip()
    return {"git_hash": g("rev-parse", "HEAD"),
            "git_dirty_code": bool(g("status", "--porcelain", "--", "src", "analysis"))}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_run(a) -> None:
    gs = git_state()                  # the code this run starts with, not HEAD at its end
    device = H.setup(a.device)
    torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else OUT / a.arm / f"seed{a.seed}"
    if (out / "provenance.json").exists() and not a.force:
        raise SystemExit(f"{out} already complete (provenance.json present); --force to redo")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = RC.Cifar10()
    rows, div = run_arm(a.arm, a.seed, data, device, tasks=a.tasks, epochs=a.epochs,
                        hist_dir=out / "hist")
    host_arm, pool = ARMS[a.arm]
    for r in rows:
        r["arm"] = a.arm
        r["pool"] = pool
    H.write_csv(out / "per_task.csv", rows)
    done = len([r for r in rows if "memo_acc" in r and r["memo_acc"] == r["memo_acc"]])
    prov = {"run_id": EXPERIMENT, "arm": a.arm, "host_arm": host_arm, "pool": pool,
            "seed": a.seed, **gs,
            "host": "src/rlcifar_cnn_0908.py", "host_sha256": sha256(HOST_FILE),
            "forward": "forward_pool (host forward_cnn with the pool as an argument)",
            "pool_call": {"max": "F.max_pool2d(a, 2, 2)", "avg": "F.avg_pool2d(a, 2, 2)"}[pool],
            "lr": LR, "optimizer": "adam", "tasks": a.tasks, "epochs_per_task": a.epochs,
            "sna": SNA_DEFAULTS if host_arm == "SNA" else None,
            "tasks_completed": done, "divergence": div, "data_sha256": data.sha256,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(), "torch": torch.__version__,
            "hostname": socket.gethostname(),
            "cuda_max_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                if device.type == "cuda" else None),
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "wall_clock_s": time.time() - t0}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"[{time.strftime('%F %T')}] {a.arm} seed={a.seed} {a.device}: "
          f"{done}/{a.tasks} tasks in {time.time() - t0:.0f}s"
          f"{'  DIVERGED@task ' + str(div['task']) if div['diverged'] else ''}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one (arm, seed)")
    r.add_argument("--arm", required=True, choices=list(ARMS))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--tasks", type=int, default=TASKS)
    r.add_argument("--epochs", type=int, default=EPOCHS)
    r.add_argument("--out", default=None)
    r.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)


if __name__ == "__main__":
    main()
