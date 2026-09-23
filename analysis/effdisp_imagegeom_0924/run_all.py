#!/usr/bin/env python3
"""Run all four width groups for the registered 50-series CIFAR geometry study.

Run only after the concurrent inputscope GPU job has ended. The runner refuses
an existing nonempty output root, so a partial study is never overwritten.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import effdisp_imagegeom_0924 as E

PREREG_COMMIT = "6fb5151"


def require_preregistered_source(source_hash: str) -> None:
    repo = E.SOURCE.parents[1]
    result = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, source_hash],
                            cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"preregistration {PREREG_COMMIT} is not an ancestor of {source_hash}: "
                           f"{result.stderr.strip()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=E.DEFAULT_OUT)
    p.add_argument("--archive", type=Path, default=E.DEFAULT_ARCHIVE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-git-hash", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 305)))
    p.add_argument("--tasks", type=int, default=E.TASKS)
    p.add_argument("--epochs", type=int, default=E.EPOCHS)
    p.add_argument("--no-graph", action="store_true")
    a = p.parse_args()
    if not a.seeds or len(set(a.seeds)) != len(a.seeds) or a.tasks < 1 or a.epochs < 1:
        p.error("seeds must be unique/nonempty and tasks/epochs positive")
    require_preregistered_source(a.source_git_hash)
    E.run(a.out, a.archive, tuple(a.seeds), a.tasks, a.epochs,
          torch.device(a.device), not a.no_graph, a.source_git_hash, E.GROUPS)


if __name__ == "__main__":
    main()
