#!/usr/bin/env python3
"""postfit_elu_cifar_0924 -- stop or replace the post-fit Adam updates in the ELU/std box
(spec: specs/spec_postfit_elu_cifar_0924.md).

The engine is `src/rlcifar_mlp_battle_0918.py` run() with its `postfit` hook, unchanged; this file
is only the CLI.  sgd_postfit_cifar_0923.py ran the same hook on leaky 0.1 ("LR"); here the
activation is "ELU" (S5 cap_cifar_ee_0920's box: seeds 0-9, R = 10, std, 50 tasks x 30,000 steps,
Adam 1e-3), with a trace every 100 steps, end-of-task snapshots and kept checkpoints:

    python3 src/postfit_elu_cifar_0924.py --arm A   --out results/postfit_elu_cifar_0924/A
    python3 src/postfit_elu_cifar_0924.py --arm F   --out results/postfit_elu_cifar_0924/F
    python3 src/postfit_elu_cifar_0924.py --arm F99 --out results/postfit_elu_cifar_0924/F99
    python3 src/postfit_elu_cifar_0924.py --arm S --eta <eta_S> --out results/postfit_elu_cifar_0924/S

    # the eta pilot (spec §2): tasks 1-2 of S at one grid point
    python3 src/postfit_elu_cifar_0924.py --arm S --eta 0.01 --tasks 2 --pilot \
        --out results/postfit_elu_cifar_0924/_pilot/eta0.01

    # check K1 before the main run: A's tasks 1-2 (known outcome, S5 ref)
    python3 src/postfit_elu_cifar_0924.py --arm A --tasks 2 --check \
        --out results/postfit_elu_cifar_0924/_checks/K1_A_t2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rlcifar_mlp_battle_0918 as B           # noqa: E402  the engine
from src import pmnist_0905 as H                       # noqa: E402  device setup

EXPERIMENT = "postfit_elu_cifar_0924"
ACT = "ELU"
SEEDS = list(range(10))
TASKS = 50
EPOCHS = 400
ETA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
# arm -> (postfit mode, hit threshold as accuracy); need_correct(0.999) = 1199, (0.99) = 1188
ARMS = {"A": ("adam", 0.999), "F": ("freeze", 0.999), "S": ("sgd", 0.999),
        "F99": ("freeze", 0.99)}
EXTRA = 500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--eta", type=float, default=None, help="S only: the SGD step after the fit")
    ap.add_argument("--tasks", type=int, default=TASKS)
    ap.add_argument("--pilot", action="store_true", help="eta pilot: S, tasks 1-2, a grid eta")
    ap.add_argument("--check", action="store_true", help="pre-launch check: A, tasks 1-2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="sink_roots_0930 (CPU): a subset of the registered seeds, one per process")
    ap.add_argument("--no-keep-ckpts", action="store_true",
                    help="sink_roots_0930: skip ckpts/t<NN>.pt (the registered analyses read snapshots only)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    mode, acc = ARMS[a.arm]
    if (a.arm == "S") != (a.eta is not None):
        raise SystemExit("--eta goes with --arm S, and only with it")
    if a.pilot and a.check:
        raise SystemExit("--pilot and --check are separate stages")
    # registered settings (spec §1-§2); the short stages are the only exceptions
    if a.pilot:
        if a.arm != "S" or a.tasks != 2 or a.eta not in ETA_GRID:
            raise SystemExit(f"the pilot is S, --tasks 2, eta in {ETA_GRID}")
    elif a.check:
        if a.arm != "A" or a.tasks != 2:
            raise SystemExit("the pre-launch check is A, --tasks 2")
    elif a.tasks != TASKS:
        raise SystemExit(f"a main run has {TASKS} tasks")
    if a.eta is not None and not a.eta > 0:
        raise SystemExit(f"eta must be > 0; got {a.eta}")

    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    seeds = SEEDS if a.seeds is None else a.seeds
    if not set(seeds) <= set(SEEDS):
        raise SystemExit(f"seeds must be a subset of {SEEDS}")
    postfit = {"mode": mode, "acc": acc, "extra": EXTRA, "eta": a.eta, "x": None}
    stage = "pilot" if a.pilot else ("check" if a.check else "main")
    B.run(ACT, seeds, ["std"], a.tasks, EPOCHS, device, Path(a.out), checkpoint=True,
          resume=not a.no_resume, graph=device.type == "cuda", schedule="iid", hit_every=100,
          keep_ckpts=not a.no_keep_ckpts, run_id=EXPERIMENT, postfit=postfit,
          extra_prov={"postfit_elu_cifar_0924": {"arm": a.arm, "stage": stage, "seeds": seeds,
                                                  "spec": "specs/spec_postfit_elu_cifar_0924.md",
                                                  "run_under": "sink_roots_0930 R7 (CPU, one seed per process)",
                                                  "keep_ckpts": not a.no_keep_ckpts}})


if __name__ == "__main__":
    main()
