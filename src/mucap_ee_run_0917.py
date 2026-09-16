#!/usr/bin/env python3
"""Random Label MNIST, ELU -> ELU: the same four first-layer caps as mucap_el_0916, second layer ELU.

    OMP_NUM_THREADS=1 python3 src/mucap_ee_run_0917.py --arm cap_perp --seeds 0 \
        --out results/mucap_ee_0917/runs/cap_perp_s0

specs/spec_mucap_ee_0917.md (registered at PREREG_COMMIT).  The question: the EE box's second layer
dies by task 10 (layer_chimera_rl_0914); in the EL box the caps that stopped the first layer's
perpendicular growth also kept ||mu2|| (the second layer's input mean) small, and the one that did not
(cap_par) doubled it.  Does the first-layer cap alone decide whether an ELU second layer collapses?

Everything but the second hidden layer's activation is src/mucap_el_run_0916.py's run_one, called with
act2_name="ELU1": the box, the arms, the caps (first layer only, from task 2, radius = the run's own
task-1-end row), the diagnostics and the per-update ledger.  This file only fixes that choice and
writes the shard with this experiment's provenance.

Two choices differ from the 0916 EL grid (spec 2.1): 100 tasks instead of 150, and the first-layer
ledger off by default -- the EE net slows down as its second layer sinks (exp of underflowing
preactivations), and the question here is the second layer's.  torch.set_flush_denormal(True): the
same ref trajectory with and without it was bit-identical on every reported quantity for 40 tasks
(pilot, seed 0), and without it the per-task time grew fourfold by task 40.
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
from src import shell_l2_rlmnist_0913 as SH      # 0913 runner; not touched
from src import mucap_el_run_0916 as EL          # the box, the arms and the diagnostics

EXPERIMENT = "mucap_ee_0917"
SPEC = "specs/spec_mucap_ee_0917.md"
PREREG_COMMIT = "21038e6d699677000f34c57587a82cc5a8d5e296"   # specs/spec_mucap_ee_0917.md, pushed before any EE cap arm ran
ACT2_NAME = "ELU1"
ARMS = EL.ARMS
TASKS = 100
EPOCHS = EL.EPOCHS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", type=int, default=TASKS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    device = H.setup(args.device)
    mnist = H.Mnist(device)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    t0 = time.time()
    all_rows, all_arrays, all_led, infos = [], {}, {}, {}
    for seed in seeds:
        rows, arrays, led, info = EL.run_one(args.arm, seed, args.lr, args.tasks, mnist, device,
                                             epochs=args.epochs, ledger=args.ledger,
                                             progress=True, act2_name=ACT2_NAME)
        all_rows += rows
        infos[str(seed)] = info
        for k, v in arrays.items():
            all_arrays[f"s{seed}_{k}"] = v
        for k, v in led.items():
            all_led[f"s{seed}_{k}"] = v
    keys = sorted({k for r in all_rows for k in r})
    with (out / "per_task.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "task"] +
                           [k for k in keys if k not in ("arm", "seed", "task")])
        w.writeheader()
        w.writerows(all_rows)
    np.savez_compressed(out / "units.npz", **all_arrays)
    if all_led:
        np.savez_compressed(out / "ledger.npz", **all_led)
    root = me.parents[1]
    (out / "provenance.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(root / SPEC) if (root / SPEC).exists() else None,
        "git_hash": H.git_hash(),
        "git_dirty_code": SH.git_dirty(["src", "analysis/mucap_ee_0917"]),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "python": sys.version.split()[0], "device": args.device,
        "threads": args.threads, "arm": args.arm, "seeds": seeds, "lr": args.lr,
        "n_tasks": args.tasks, "epochs_per_task": args.epochs, "batch": EL.BATCH,
        "steps_per_task": EL.STEPS_PER_EPOCH * args.epochs, "n_images": EL.N_IMAGES,
        "act1": "ELU1", "act2": ACT2_NAME, "cap_layer": EL.CAP_LAYER, "ledger": args.ledger,
        "flush_denormal": True,
        "code_sha256": {f"src/{n}": SH.file_sha256(me.parent / n) for n in
                        ("mucap_ee_run_0917.py", "mucap_el_run_0916.py", "mucap_el_0916.py",
                         "pmnist_0905.py", "pmnist_rlmnist_0906.py", "elu_growth_0909.py")},
        "data_sha256": mnist.sha256, "per_seed": infos,
        "wall_clock_s": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2, default=str))
    print(f"wrote {out}  ({len(all_rows)} task rows, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
