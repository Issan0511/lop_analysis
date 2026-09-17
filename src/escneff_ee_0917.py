#!/usr/bin/env python3
"""escape_ee_0917 again, on unused seeds 10-19, with n_eff as the manipulation check.

    OMP_NUM_THREADS=1 python3 src/escneff_ee_0917.py --seed 10 --out results/escneff_ee_0917/runs/s10

specs/spec_escneff_ee_0917.md (registered at PREREG_COMMIT).  escape_ee_0917 held the t2 host's growth
and respdyn's remainder fell from 0.112 to 0.066, but its registered manipulation check (the host's mean
second-layer climb) failed, so the question went unanswered; post hoc, the five hold arms' E followed the
host's in-task n_eff (the effective number of images that carry the second layer's training derivative),
not the mean climb.  This file runs escape_ee_0917's twelve arms unchanged -- the same runner, imported --
on seeds none of those arms has seen, and adds only what those seeds need:

  * the prefix (the natural ref trajectory, tasks 1-12) is compared with neff_pred_0917's CPU record,
    which exists for seeds 10-12 (its natural trajectories were seen; no branch was ever run there);
    resp_ee / respdyn have no record for these seeds, so those comparisons are empty (None);
  * provenance names this experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import shell_l2_rlmnist_0913 as SH      # file_sha256, git_dirty; not touched
from src import swap_ee_0917 as SW               # _csv_rows; not touched
from src import escape_ee_0917 as ES             # the twelve arms, the prefix, the per-seed run; not touched

EXPERIMENT = "escneff_ee_0917"
SPEC = "specs/spec_escneff_ee_0917.md"
PREREG_COMMIT = "unregistered"                   # set to the spec's registration commit before the run
ROOT = Path(__file__).resolve().parents[1]
NEFF_PRED_EE = ROOT / "results" / "neff_pred_0917" / "ee"
SEEDS = tuple(range(10, 20))
EPOCHS = ES.EPOCHS
GIT_AT_START: dict = {}


def _git_state() -> dict:
    return {"hash": H.git_hash(), "dirty": SH.git_dirty(["src", "analysis/escneff_ee_0917"])}


def prefix(seed: int, mnist: H.Mnist, epochs: int = EPOCHS, progress: bool = True, es=ES) -> dict:
    """escape_ee_0917.prefix, plus the comparison with neff_pred_0917's record of the same trajectory."""
    pre = es.prefix(seed, mnist, epochs, progress)
    rec = {int(r["task"]): r["state_sha256"] for r in SW._csv_rows(NEFF_PRED_EE / f"s{seed}" / "prefix.csv")}
    for r in pre["rows"]:
        r["hash_match_neff_pred"] = None if r["task"] not in rec else rec[r["task"]] == r["state_sha256"]
    pre["record_tasks_neff_pred"] = sum(r["task"] in rec for r in pre["rows"])
    return pre


def run_seed(seed: int, out: Path, mnist: H.Mnist, epochs: int = EPOCHS, arms=None, progress: bool = True,
             pre: dict | None = None, es=ES) -> dict:
    git0 = dict(GIT_AT_START) or _git_state()
    es.GIT_AT_START.clear()
    es.GIT_AT_START.update(git0)
    if pre is None:
        pre = prefix(seed, mnist, epochs, progress, es)
    prov = es.run_seed(seed, out, mnist, epochs, arms, progress, pre)
    prov.update({
        "experiment": EXPERIMENT, "spec": SPEC, "prereg_commit": PREREG_COMMIT,
        "spec_sha256": SH.file_sha256(ROOT / SPEC) if (ROOT / SPEC).exists() else None,
        "runner_experiment": es.EXPERIMENT, "runner_prereg_commit": es.PREREG_COMMIT,
        "code_sha256": {**prov["code_sha256"],
                        "src/escneff_ee_0917.py": SH.file_sha256(Path(__file__).resolve())},
        "prefix_record_tasks_neff_pred": pre["record_tasks_neff_pred"],
        "prefix_hash_match_neff_pred": sum(bool(r["hash_match_neff_pred"]) for r in pre["rows"]),
    })
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=None, help="comma list (checks and S-cost only)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    GIT_AT_START.update(_git_state())
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    arms = args.arms.split(",") if args.arms else None
    run_seed(args.seed, Path(args.out), mnist, args.epochs, arms)


if __name__ == "__main__":
    main()
