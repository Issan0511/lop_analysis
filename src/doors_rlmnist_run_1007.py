#!/usr/bin/env python3
"""Random Label MNIST, ELU -> ELU: the centering doors C / H / CH (doors_rlmnist_1007).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
        -m src.doors_rlmnist_run_1007 --arm H --seeds 0 --out results/doors_rlmnist_1007/runs/H_s0

specs/spec_doors_rlmnist_1007.md (registered at PREREG_COMMIT).  Four arms, both doors on from the first
update of task 1:

    ref  -            (l2cap_ee_0917's ref, bit for bit: check S0)
    C    input centred: x - x.mean(0) over the seed's 1200 images
    H    both hidden outputs centred: phi(z_l) - m_l, m_l the beta = 0.01 EMA of the uncentred batch mean
    CH   both

Everything else is src/doors_rlmnist_1007.py's run_doors (the host loop written out).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import shell_l2_rlmnist_0913 as SH      # file_sha256, git_dirty; not touched
from src import doors_rlmnist_1007 as D          # the doors and the loop

EXPERIMENT = "doors_rlmnist_1007"
SPEC = "specs/spec_doors_rlmnist_1007.md"
PREREG_COMMIT = D.PREREG_COMMIT
# arm -> (door C, door H)
ARM_MAP = {"ref": (False, False), "C": (True, False), "H": (False, True), "CH": (True, True)}
ARMS = tuple(ARM_MAP)
TASKS = 100
EPOCHS = D.EPOCHS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", type=int, default=TASKS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)               # after every import (elu_growth's chain sets 1)
    torch.set_flush_denormal(True)
    device = H.setup(args.device)
    mnist = H.Mnist(device)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    t0 = time.time()
    all_rows, all_arrays, infos = [], {}, {}
    door_c, door_h = ARM_MAP[args.arm]
    for seed in seeds:
        rows, arrays, info = D.run_doors(seed, args.lr, args.tasks, mnist, device, door_c=door_c,
                                         door_h=door_h, epochs=args.epochs, progress=True)
        for r in rows:
            r["arm"] = args.arm
        info["divergence"]["arm"] = args.arm
        all_rows += rows
        infos[str(seed)] = info
        for k, v in arrays.items():
            all_arrays[f"s{seed}_{k}"] = v
    keys = sorted({k for r in all_rows for k in r})
    with (out / "per_task.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "task"] +
                           [k for k in keys if k not in ("arm", "seed", "task")])
        w.writeheader()
        w.writerows(all_rows)
    np.savez_compressed(out / "units.npz", **all_arrays)
    root = me.parents[1]
    (out / "provenance.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(root / SPEC) if (root / SPEC).exists() else None,
        "git_hash": H.git_hash(),
        "git_dirty_code": SH.git_dirty(["src", "analysis/doors_rlmnist_1007", SPEC]),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "python": sys.version.split()[0], "device": args.device,
        "threads": args.threads, "torch_threads": torch.get_num_threads(),
        "arm": args.arm, "seeds": seeds, "lr": args.lr,
        "n_tasks": args.tasks, "epochs_per_task": args.epochs, "batch": D.BATCH,
        "steps_per_task": D.STEPS_PER_EPOCH * args.epochs, "n_images": D.N_IMAGES,
        "act1": "ELU1", "act2": "ELU1", "ledger": False,
        "doors": {"C": door_c, "H": door_h, "beta": D.BETA if door_h else None},
        "flush_denormal": True,
        "code_sha256": {f"src/{n}": SH.file_sha256(me.parent / n) for n in
                        ("doors_rlmnist_run_1007.py", "doors_rlmnist_1007.py", "mucap_el_run_0916.py",
                         "mucap_el_0916.py", "l2cap_ee_0917.py", "pmnist_0905.py", "pmnist_rlmnist_0906.py",
                         "elu_growth_0909.py", "shell_l2_rlmnist_0913.py")},
        "data_sha256": mnist.sha256, "per_seed": infos,
        "wall_clock_s": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2, default=str))
    print(f"wrote {out}  ({len(all_rows)} task rows, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
