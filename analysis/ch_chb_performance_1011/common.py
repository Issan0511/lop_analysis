"""Frozen settings and CPU-only validation for the 1011 performance replication."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = "ch_chb_performance_1011"
OUT = ROOT / "results" / RUN
SPEC = ROOT / "specs" / f"spec_{RUN}.md"
PYTHON = Path("/home/issan/Projects/claude/proj_004_drift/.venv/bin/python")
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar10")
ARCHIVE = "cifar-10-python.tar.gz"
SEEDS = list(range(10, 20))
ARMS = ("CH", "CHB")
TASKS = 200
LAM = {"CH": 0.0, "CHB": 0.1713}
EPOCHS, BETA, LR, THREADS = 400, 0.01, 0.001, 2
DOORS = {"CH": dict(c=True, h=True, b="none"), "CHB": dict(c=True, h=True, b="wd")}
DIAGNOSTICS = tuple(
    f"{name}_l{layer}" for layer in (1, 2)
    for name in ("zbar", "zsd", "bias_over_sd", "w_norm", "dead_frac", "gate_zero_frac")
)
REQUIRED_METRICS = ("online_acc", "memo_acc", *DIAGNOSTICS)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def sources():
    paths = [SPEC, *(ROOT / "src" / name for name in (
        "relu_doors_0919.py", "pmnist_0905.py", "pmnist_rlmnist_0906.py", "pmnist_rlcifar_0907.py")),
        *sorted(Path(__file__).parent.glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def config():
    return dict(run_id=RUN, arms=list(ARMS), seeds=SEEDS, tasks=TASKS,
                epochs=EPOCHS, conds=["raw"], beta=BETA, lr=LR, lam=LAM,
                threads=THREADS, device="cuda", graph=True, snapshots=True,
                primary_window=[181, 200], early_window=[31, 50])


def expected_meta(arm, prereg_commit):
    return dict(arm=arm, seeds=SEEDS, conds=["raw"], epochs=EPOCHS, lr=LR,
                lam=LAM[arm], beta=BETA, doors=DOORS[arm], perturb=None,
                nan_slot=None, run_id=RUN, prereg_commit=prereg_commit)


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label}: boolean is not an integer")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid integer {value!r}") from exc
    if not math.isfinite(x) or not x.is_integer():
        raise ValueError(f"{label}: invalid integer {value!r}")
    return int(x)


def validate_rows(rows, arm, tasks=TASKS):
    """No dropping/imputation: require the entire specified seed-by-task rectangle."""
    if arm not in ARMS or not 0 <= tasks <= TASKS:
        raise ValueError("invalid arm/task count")
    wanted = {(s, t) for s in SEEDS for t in range(1, tasks + 1)}
    if len(rows) != len(wanted):
        raise ValueError(f"{arm}: expected {len(wanted)} rows, found {len(rows)}")
    out = {}
    for raw in rows:
        s, t = integer(raw.get("seed"), "seed"), integer(raw.get("task"), "task")
        key = (s, t)
        if key not in wanted or key in out:
            raise ValueError(f"{arm}: unexpected or duplicate seed/task {key}")
        if raw.get("arm") != arm or raw.get("cond") != "raw" or raw.get("iv") != "none":
            raise ValueError(f"{arm} {key}: wrong arm/condition/intervention")
        if integer(raw.get("slot"), "slot") != SEEDS.index(s) or float(raw.get("lr", -1)) != LR:
            raise ValueError(f"{arm} {key}: wrong slot or learning rate")
        row = dict(raw, seed=s, task=t)
        for name in REQUIRED_METRICS:
            try:
                v = float(raw[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{arm} {key}: missing/invalid {name}") from exc
            if not math.isfinite(v):
                raise ValueError(f"{arm} {key}: non-finite {name}")
            if name in ("online_acc", "memo_acc") or "frac" in name:
                if not 0 <= v <= 1:
                    raise ValueError(f"{arm} {key}: out-of-range {name}={v}")
            elif name.startswith(("zsd", "bias_over_sd", "w_norm")) and v < 0:
                raise ValueError(f"{arm} {key}: negative {name}={v}")
            row[name] = v
        out[key] = row
    if set(out) != wanted:
        raise ValueError(f"{arm}: missing seed/task rows")
    return out


def validate_checkpoint(st, arm, prereg_commit):
    if st.get("meta") != expected_meta(arm, prereg_commit):
        raise ValueError(f"{arm}: checkpoint configuration differs")
    t = integer(st.get("t"), "checkpoint task")
    if not 1 <= t <= TASKS or integer(st.get("tc"), "Adam step") != t * EPOCHS * 75:
        raise ValueError(f"{arm}: invalid task or optimizer step")
    if st.get("diverged") or not bool(st["alive"].all()) or len(st["alive"]) != len(SEEDS):
        raise ValueError(f"{arm}: failed/diverged seed")
    validate_rows(st["rows"], arm, t)
    for key in ("g_lab", "g_batch"):
        if set(st[key]) != set(SEEDS):
            raise ValueError(f"{arm}: wrong {key} seeds")
    if len(st.get("act_state", {}).get("m", [])) != 2:
        raise ValueError(f"{arm}: missing hidden EMA state")
    # Imported only for checking actual checkpoints; plan/report arithmetic stays CPU-only.
    import torch
    shapes = [(10, 100, 3072), (10, 100), (10, 100, 100), (10, 100), (10, 10, 100), (10, 10)]
    for key in ("P", "m", "v"):
        if len(st[key]) != len(shapes):
            raise ValueError(f"{arm}: incomplete {key}")
        for value, shape in zip(st[key], shapes):
            if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{arm}: invalid {key} tensor")
    for value in st["act_state"]["m"]:
        if tuple(value.shape) != (10, 100) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"{arm}: invalid EMA tensor")
    return t
