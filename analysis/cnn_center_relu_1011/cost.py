#!/usr/bin/env python3
"""Seconds per epoch of one arm's bundle (K-cost).  No evaluation, no accuracy printed.

    python analysis/cnn_center_relu_1011/cost.py --arm CH --seeds 100-109 --epochs 40 [--wait-file F]

Seeds 100-109 are not registered seeds.  With --wait-file the timed loop starts only once the
file exists (to start several processes together); the first epoch (CUDA graph capture) is
timed separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                          # noqa: E402

from src import pmnist_0905 as H                      # noqa: E402
from src import pmnist_rlcifar_0907 as RC             # noqa: E402
from src import sna_cnn_cause_1009 as E               # noqa: E402
from src import cnn_center_relu_1011 as M             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="CH")
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mem-gb", type=float, default=2.4)
    ap.add_argument("--wait-file", default=None)
    args = ap.parse_args()
    dev = H.setup("cuda")
    di = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(di).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb / total), di)
    cifar = RC.Cifar10()
    B = M.Bundle([(args.arm, s) for s in E.parse_seeds(args.seeds)], cifar, dev, graph=True)
    M.diagnostics(B.P, B.X, B.Y, B.act)                 # the evaluation's peak, as in `run`
    B.new_labels()
    t0 = time.time()
    B.run_epoch()                                        # capture + first epoch
    torch.cuda.synchronize()
    t_first = time.time() - t0
    if args.wait_file:
        while not Path(args.wait_file).exists():
            time.sleep(0.05)
    t0 = time.time()
    for _ in range(args.epochs):
        B.run_epoch()
    torch.cuda.synchronize()
    dt = time.time() - t0
    t1 = time.time()
    M.diagnostics(B.P, B.X, B.Y, B.act)
    torch.cuda.synchronize()
    t_diag = time.time() - t1
    print(json.dumps({"arm": args.arm, "R": B.R, "epochs": args.epochs, "s_per_epoch": dt / args.epochs,
                      "first_epoch_s": t_first, "diagnostics_s": t_diag, "pid": os.getpid(),
                      "max_mem_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "max_mem_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20}))


if __name__ == "__main__":
    main()
