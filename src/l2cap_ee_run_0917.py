#!/usr/bin/env python3
"""Random Label MNIST, ELU -> ELU: caps on the second layer's own weight growth (l2cap_ee_0917).

    OMP_NUM_THREADS=1 python3 src/l2cap_ee_run_0917.py --arm cap12 --seeds 0 \
        --out results/l2cap_ee_0917/runs/cap12_s0

specs/spec_l2cap_ee_0917.md (registered at PREREG_COMMIT).  mucap_ee_0917 capped the first layer only
and every arm still collapsed; its cap_both kept ||mu2|| small while the second layer's rows grew the
most.  Five arms, a 2x2 of (first-layer caps) x (second-layer row-norm cap) plus one with both hidden
biases held:

    ref        -                                   cap2        second-layer ||w2_i|| <= task-1 end
    cap1       mucap_ee's cap_both (|q|, ||v||)    cap12       cap1 + cap2
    cap12_bfix cap12 + b1, b2 put back to their task-1-end values after every update

Everything else is src/mucap_el_run_0916.py's run_one with act2_name="ELU1" (mucap_ee_0917's box,
flush_denormal on, first-layer ledger off, 100 tasks); cap1 and ref are that experiment's cap_both and
ref, bit for bit (check S0b).
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

EXPERIMENT = "l2cap_ee_0917"
SPEC = "specs/spec_l2cap_ee_0917.md"
PREREG_COMMIT = "b83d59335fd72e0ae4787f139752e253e1b63df1"   # specs/spec_l2cap_ee_0917.md, pushed before any arm with a second-layer cap ran
ACT2_NAME = "ELU1"
# arm -> (first-layer arm of the shared runner, second-layer row-norm cap, hidden biases held)
ARM_MAP = {"ref": ("ref", False, False), "cap1": ("cap_both", False, False), "cap2": ("ref", True, False),
           "cap12": ("cap_both", True, False), "cap12_bfix": ("cap_both", True, True)}
ARMS = tuple(ARM_MAP)
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
        a1, w2, bf = ARM_MAP[args.arm]
        rows, arrays, led, info = EL.run_one(a1, seed, args.lr, args.tasks, mnist, device,
                                             epochs=args.epochs, ledger=args.ledger,
                                             progress=True, act2_name=ACT2_NAME, w2cap=w2, bias_fix=bf)
        for r in rows:
            r["arm"] = args.arm
        info["divergence"]["arm"] = args.arm
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
        "arm_map": dict(zip(("layer1_arm", "w2cap", "bias_fix"), ARM_MAP[args.arm])),
        "flush_denormal": True,
        "code_sha256": {f"src/{n}": SH.file_sha256(me.parent / n) for n in
                        ("l2cap_ee_run_0917.py", "l2cap_ee_0917.py", "mucap_el_run_0916.py", "mucap_el_0916.py",
                         "pmnist_0905.py", "pmnist_rlmnist_0906.py", "elu_growth_0909.py")},
        "data_sha256": mnist.sha256, "per_seed": infos,
        "wall_clock_s": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2, default=str))
    print(f"wrote {out}  ({len(all_rows)} task rows, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
