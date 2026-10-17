"""Frozen registration and complete-data validation for run 3."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = "chb_decompose_200_1018"
OUT = ROOT / "results" / RUN
REFERENCE = ROOT / "results/ch_chb_performance_1011"
SPEC = ROOT / "specs" / f"spec_{RUN}.md"
COMMON_SPEC = ROOT / "specs/spec_six_runs_common_1018.md"
REGISTERED = "5c741be8"
SEEDS = list(range(10, 20))
ARMS = ("CH", "CHB", "CH_bfreeze", "CH_wWD", "CH_bWD")
LR, LAM, BETA, EPOCHS, TASKS = .001, .1713, .01, 400, 200
PYTHON = "/home/issan/Projects/claude/proj_004_drift/.venv/bin/python"
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar10")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def put(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def registration():
    commit = git("rev-parse", REGISTERED + "^{commit}")
    git("merge-base", "--is-ancestor", commit, "HEAD")
    for path in (SPEC, COMMON_SPEC):
        original = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path.relative_to(ROOT)}"],
                                  check=True, capture_output=True).stdout
        if original != path.read_bytes():
            raise ValueError("registered spec changed: " + str(path))
    return dict(commit=commit, spec_sha256=sha(SPEC), common_spec_sha256=sha(COMMON_SPEC),
                prediction="BIAS_STOP_SUFFICES", prediction_basis="common spec §8; Issa execution instruction")


def sources():
    files = [SPEC, COMMON_SPEC, ROOT / "src/chb_decompose_200_1018.py",
             *(ROOT / "src" / f for f in ("relu_doors_0919.py", "pmnist_0905.py",
                                          "pmnist_rlcifar_0907.py", "pmnist_rlmnist_0906.py")),
             *sorted(Path(__file__).parent.glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def config():
    return dict(run_id=RUN, seeds=SEEDS, arms=list(ARMS), tasks=TASKS, epochs=EPOCHS,
                steps_per_task=30000, dims=[3072, 100, 100, 10], batch=16, R=10,
                conds=["raw"], device="cuda", lr=LR, lam=LAM, beta=BETA,
                dtype="float32", tf32=False, threads=2, graph=True,
                window=[181, 200], early_window=[31, 50], m=7, epsilon_eq=.005)


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def integer(v):
    x = float(v)
    if isinstance(v, bool) or not math.isfinite(x) or not x.is_integer():
        raise ValueError("invalid integer")
    return int(x)


def validate_rows(rows, arm, tasks=TASKS):
    if arm not in ARMS or not 0 <= tasks <= TASKS:
        raise ValueError("invalid arm/tasks")
    wanted = {(s, t) for s in SEEDS for t in range(1, tasks + 1)}
    if len(rows) != len(wanted):
        raise ValueError("INCOMPLETE: wrong row count")
    out = {}
    for row in rows:
        seed, task = integer(row["seed"]), integer(row["task"])
        key = (seed, task)
        if key not in wanted or key in out:
            raise ValueError("INCOMPLETE: unexpected/duplicate key")
        if row["arm"] != arm or row["cond"] != "raw" or row["iv"] != "none":
            raise ValueError("CHECK_FAILED: arm/condition")
        if integer(row["slot"]) != SEEDS.index(seed) or float(row["lr"]) != LR:
            raise ValueError("CHECK_FAILED: slot/lr")
        for name in ("online_acc", "memo_acc"):
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError("DIVERGED: nonfinite performance")
            if not 0 <= value <= 1:
                raise ValueError("CHECK_FAILED: invalid accuracy")
        out[key] = row
    return out
