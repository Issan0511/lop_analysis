#!/usr/bin/env python3
"""sgd_postfit_cifar_0923 -- switch the optimizer after the fit (spec_sgd_postfit_cifar_0923.md).

The engine is `src/rlcifar_mlp_battle_0918.py` run() with its `postfit` hook; this file is only
the CLI.  Every arm is altlabels_cifar_0923's LR_iid (LR, std, seeds 0-9, R = 10, iid labels,
50 tasks x 30,000 steps, trace every 100 steps, one CUDA graph) except for what the slot does
after its own switch step s_sw = hit999 + 500:

    python3 src/sgd_postfit_cifar_0923.py run --mode adam          --out results/sgd_postfit_cifar_0923/A
    python3 src/sgd_postfit_cifar_0923.py run --mode adam_restore  --out results/sgd_postfit_cifar_0923/AR
    python3 src/sgd_postfit_cifar_0923.py run --mode freeze        --out results/sgd_postfit_cifar_0923/F
    python3 src/sgd_postfit_cifar_0923.py run --mode sgd --eta 0.1 --out results/sgd_postfit_cifar_0923/S_hi
    python3 src/sgd_postfit_cifar_0923.py run --mode adam_ce --x 1.3 --out results/sgd_postfit_cifar_0923/A_ce

    # check C1: the engine without the hook (the parent's path)
    python3 src/sgd_postfit_cifar_0923.py run --mode none --tasks 1 --out .../_checks/C1

    # the eta pilot's descriptive leg: task 49 from LR_iid's checkpoint after task 48
    python3 src/sgd_postfit_cifar_0923.py run --mode sgd --eta 0.1 --tasks 1 \
        --restore <altlabels LR_iid>/ckpts/t48.pt --out .../_pilot/t49_eta0.1

    # the 2x2 crossover (spec §2b): weights {s_sw, task end} x Adam state {s_sw, task end} of
    # LR_iid's task t, then task t+1 under Adam with the stop rule
    python3 src/sgd_postfit_cifar_0923.py xfork --t 10,20,30,40,48 --out results/sgd_postfit_cifar_0923/xfork
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rlcifar_mlp_battle_0918 as B           # noqa: E402  the engine
from src import pmnist_0905 as H                       # noqa: E402  device setup, csv

EXPERIMENT = "sgd_postfit_cifar_0923"
MODES = ("none",) + B.POSTFIT_MODES
LR_IID = (Path.home() / "Projects/obsidian-research-data/altlabels_cifar_0923/results/"
          "altlabels_cifar_0923/LR_iid")
XCELLS = (("sw", "sw"), ("sw", "end"), ("end", "sw"), ("end", "end"))   # (weights, Adam state)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def do_xfork(a, device) -> None:
    """Spec §2b.  For each fork point t of LR_iid:
    1. `t<t>_sw`: task t again from ckpts/t<t-1>.pt in freeze mode -> its end-of-task checkpoint
       holds LR_iid's own (W, m, v) right after s_sw (the steps up to s_sw are LR_iid's, C3), with
       the same streams and tc as LR_iid's ckpts/t<t>.pt (every task draws the same batches).
    2. four checkpoints: weights from {sw, end} x (m, v) from {sw, end}, everything else from
       LR_iid's ckpts/t<t>.pt.
    3. task t+1 from each under the engine's Adam, R = 10 lockstep, stop 0.999,500 (the bundle
       ends when every slot is 500 steps past its hit999, or at 30,000)."""
    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.time()
    for t in B.parse_ints(a.t):
        ck_prev, ck_end = src / "ckpts" / f"t{t - 1:02d}.pt", src / "ckpts" / f"t{t:02d}.pt"
        o_sw = out / f"t{t:02d}_sw"
        if not (o_sw / "provenance.json").exists():
            B.run("LR", list(range(10)), ["std"], 1, a.epochs, device, o_sw, checkpoint=True,
                  resume=True, graph=True, schedule="iid", hit_every=100, keep_ckpts=True,
                  restore={"path": ck_prev}, run_id=EXPERIMENT,
                  postfit={"mode": "freeze", "acc": 0.999, "extra": 500, "eta": None},
                  extra_prov={"xfork": {"t": t, "leg": "sw", "src": str(src)}})
        sw = torch.load(o_sw / "ckpts" / "t01.pt", map_location="cpu", weights_only=False)
        end = torch.load(ck_end, map_location="cpu", weights_only=False)
        # the sw leg must end on LR_iid's own streams and clock, or the cells differ in more
        # than (W, m, v)
        for s in sw["g_batch"]:
            assert torch.equal(sw["g_batch"][s], end["g_batch"][s]), f"g_batch seed {s} t {t}"
            assert torch.equal(sw["g_lab"][s], end["g_lab"][s]), f"g_lab seed {s} t {t}"
        assert sw["tc"] == end["tc"], (sw["tc"], end["tc"])
        sw_rows = {int(q["seed"]): q for q in csv.DictReader(open(o_sw / "per_task.csv"))}
        for wp, ws in XCELLS:
            name = f"t{t:02d}_W{wp}_S{ws}"
            o = out / name
            if not (o / "provenance.json").exists():
                st = dict(end)
                st["P"] = (sw if wp == "sw" else end)["P"]
                st["m"] = (sw if ws == "sw" else end)["m"]
                st["v"] = (sw if ws == "sw" else end)["v"]
                ck = out / "_ckpts" / f"{name}.pt"
                ck.parent.mkdir(parents=True, exist_ok=True)
                B.save_atomic(st, ck)
                B.run("LR", list(range(10)), ["std"], 1, a.epochs, device, o, checkpoint=False,
                      resume=False, graph=True, schedule="iid", hit_every=100,
                      stop=(0.999, 500), restore={"path": ck}, run_id=EXPERIMENT,
                      extra_prov={"xfork": {"t": t, "weights": wp, "state": ws,
                                            "src": str(src), "ckpt_end": str(ck_end),
                                            "ckpt_end_sha256": sha256_file(ck_end),
                                            "sw_leg": str(o_sw)}})
            for q in csv.DictReader(open(o / "per_task.csv")):
                s = int(q["seed"])
                rows.append({"t": t, "weights": wp, "state": ws, "seed": s,
                             "hit99": q.get("hit99", ""), "hit999": q.get("hit999", ""),
                             "steps": q.get("steps", ""), "memo_acc": q.get("memo_acc", ""),
                             "switch_step_t": sw_rows[s]["switch_step"],
                             "status": "alive" if q.get("memo_acc") not in (None, "") else "diverged"})
            H.write_csv(out / "xfork.csv", rows)
            print(f"[{time.strftime('%T')}] xfork {name} done "
                  f"({(time.time() - t_start) / 60:.1f} min)", flush=True)
    (out / "provenance.json").write_text(json.dumps(
        {"run_id": EXPERIMENT, "stage": "xfork", **B.git_state(), "src": str(src),
         "t": B.parse_ints(a.t), "cells": XCELLS, "n_rows": len(rows),
         "device": str(device), "torch": torch.__version__,
         "wall_clock_s": time.time() - t_start}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "xfork"])
    ap.add_argument("--mode", default=None, choices=MODES,
                    help="none = no hook (the parent's path); else the post-fit mode")
    ap.add_argument("--eta", type=float, default=None, help="sgd / sgd_all: the SGD learning rate")
    ap.add_argument("--schedule", default="iid", choices=["iid", "abab"],
                    help="label schedule (追補 1 runs abab with sgd_all)")
    ap.add_argument("--x", type=float, default=None, help="adam_ce: the CE drop in e-folds")
    ap.add_argument("--acc", type=float, default=0.999, help="the hit that starts the clock")
    ap.add_argument("--extra", type=int, default=500, help="s_sw = hit + extra")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--conds", default="std")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hit-every", type=int, default=100)
    ap.add_argument("--keep-ckpts", action="store_true")
    ap.add_argument("--restore", default=None, help="start from this ckpts/t<NN>.pt")
    ap.add_argument("--t", default="10,20,30,40,48", help="xfork: LR_iid's fork points")
    ap.add_argument("--src", default=str(LR_IID), help="xfork: LR_iid's run directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    if a.stage == "xfork":
        return do_xfork(a, device)
    if a.mode is None:
        raise SystemExit("run needs --mode")
    if (a.mode in ("sgd", "sgd_all")) != (a.eta is not None):
        raise SystemExit("--eta goes with --mode sgd / sgd_all, and only with them")
    if (a.mode == "adam_ce") != (a.x is not None):
        raise SystemExit("--x goes with --mode adam_ce, and only with it")
    postfit = (None if a.mode == "none" else
               {"mode": a.mode, "acc": a.acc, "extra": a.extra, "eta": a.eta, "x": a.x})
    B.run("LR", B.parse_ints(a.seeds), a.conds.split(","), a.tasks, a.epochs, device,
          Path(a.out), checkpoint=True, resume=not a.no_resume, graph=not a.no_graph,
          schedule=a.schedule, hit_every=a.hit_every, keep_ckpts=a.keep_ckpts,
          restore={"path": a.restore} if a.restore else None, run_id=EXPERIMENT,
          postfit=postfit)


if __name__ == "__main__":
    main()
