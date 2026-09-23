#!/usr/bin/env python3
"""Run the preregistered effective-dispersion suites sequentially.

No analysis is performed here. Each subprocess owns a separate output directory
and log pair. The top-level status and DONE files are written atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/home/issan/Projects/obsidian-research-data/effdisp_validation_0924")
SEEDS = (100, 101, 102, 103, 104)
ACTS = ("LR", "R", "ELU", "GELU", "SiLU", "SN1", "SNA03", "SNA06", "SNA1")
CONDA_EXPECTED_ARM_COUNT = 22
PREREG_COMMITS = ("290c4f8", "2ed7321", "89b8789", "5502cc1")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def suites(raw: Path) -> list[tuple[str, list[str], Path]]:
    result = []
    for act in ACTS:
        dest = raw / f"pm_{act}"
        result.append((f"pm_{act}", [sys.executable, "-u", "src/effdisp_mnist_0924.py",
                       "--mode", "pm", "--act", act, "--tasks", "50", "--epochs", "1",
                       "--optimizer", "adam", "--lr", "0.001",
                       "--seeds", ",".join(map(str, SEEDS)), "--device", "cuda",
                       "--out", str(dest)], dest))
    dest = raw / "pm_sgd" / "LR"
    result.append(("pm_sgd_LR", [sys.executable, "-u", "src/effdisp_mnist_0924.py",
                   "--mode", "pm", "--act", "LR", "--tasks", "50", "--epochs", "1",
                   "--optimizer", "sgd", "--lr", "0.01",
                   "--seeds", ",".join(map(str, SEEDS)), "--device", "cuda",
                   "--out", str(dest)], dest))
    dest = raw / "conda"
    result.append(("conda", [sys.executable, "-u", "-m", "src.effdisp_conda_0924",
                   "--group", "both", "--tasks", "100", "--period", "10000",
                   "--frequency-period", "1000", "--seeds", *map(str, SEEDS),
                   "--device", "cuda", "--out", str(dest)], dest))
    for act in ACTS:
        dest = raw / f"rl_{act}"
        result.append((f"rl_{act}", [sys.executable, "-u", "src/effdisp_mnist_0924.py",
                       "--mode", "rl", "--act", act, "--tasks", "50", "--epochs", "400",
                       "--optimizer", "adam", "--lr", "0.001",
                       "--seeds", ",".join(map(str, SEEDS)), "--device", "cuda",
                       "--out", str(dest)], dest))
    return result


def confirmed_complete(name: str, dest: Path, git_hash: str) -> bool:
    """A resume skip needs complete metadata, statuses, and terminal snapshots."""
    try:
        meta = json.loads((dest / "metadata.json").read_text())
        if meta["git_hash"] != git_hash or meta["tasks"] != (100 if name == "conda" else 50):
            return False
        if tuple(meta["seeds"]) != SEEDS:
            return False
        if name == "conda":
            arms = {arm[0] for group in ("primary_arms", "extra_arms", "frequency_arms", "m40_arms")
                    for arm in meta[group]}
            if len(arms) != CONDA_EXPECTED_ARM_COUNT or set(meta["spec_arm_map"].values()) != arms:
                return False
            for arm in arms:
                for seed in SEEDS:
                    sd = dest / arm / f"seed{seed}"
                    state = json.loads((sd / "status.json").read_text())
                    if state["status"] != "COMPLETED" or state["last_saved_task"] != 100:
                        return False
                    if not (sd / "t100.npz").is_file():
                        return False
        else:
            if name == "pm_sgd_LR":
                mode, act, optimizer, lr = "pm", "LR", "sgd", 0.01
            else:
                mode, act = name.split("_", 1)
                optimizer, lr = "adam", 0.001
            if meta["mode"] != mode or meta["activation"] != act or meta["smoke"]:
                return False
            if meta["optimizer"] != optimizer or meta["lr"] != lr:
                return False
            if meta["epochs"] != (400 if mode == "rl" else 1):
                return False
            statuses = json.loads((dest / "status.json").read_text())
            if set(statuses) != {str(s) for s in SEEDS}:
                return False
            for seed in SEEDS:
                st = statuses[str(seed)]
                if st["state"] != "complete" or st["completed_tasks"] != 50:
                    return False
                if not (dest / f"seed_{seed:03d}" / "state_050.npz").is_file():
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--resume", action="store_true",
                    help="skip confirmed complete suites; refuse incomplete existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="print commands; write nothing")
    args = ap.parse_args()
    archive = args.archive.expanduser().resolve()
    raw, logs = archive / "raw", archive / "logs"
    jobs = suites(raw)
    if args.dry_run:
        for name, cmd, dest in jobs:
            print(json.dumps({"suite": name, "cmd": cmd, "cwd": str(REPO),
                              "out": str(dest), "stdout": str(logs / f"{name}.stdout.log"),
                              "stderr": str(logs / f"{name}.stderr.log")}))
        return 0
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    for prereg in PREREG_COMMITS:
        if subprocess.run(["git", "merge-base", "--is-ancestor", prereg, git_hash],
                          cwd=REPO, check=False).returncode:
            ap.error(f"preregistered commit {prereg} is not an ancestor of HEAD")
    prior_file = archive / "run_status.json"
    if not args.resume and (raw.exists() or logs.exists() or prior_file.exists()):
        ap.error(f"archive already has raw/log/status outputs: {archive}; use --resume")
    if args.resume and raw.exists() and not prior_file.is_file():
        ap.error("--resume requires existing run_status.json when raw/ exists")
    if args.resume and prior_file.is_file():
        status = json.loads(prior_file.read_text())
        if status.get("git_hash") != git_hash or status.get("prereg_commits") != list(PREREG_COMMITS):
            ap.error("existing run_status.json does not match this git/prereg provenance")
    else:
        status = {"started_at": now(), "git_hash": git_hash,
                  "prereg_commits": list(PREREG_COMMITS), "cwd": str(REPO),
                  "python": sys.executable, "archive": str(archive), "suites": {}}
    # Preflight the entire resume before launching even one suite.
    for name, cmd, dest in jobs:
        if dest.exists() and not (args.resume and confirmed_complete(name, dest, git_hash)):
            ap.error(f"existing suite is not confirmed complete; refusing overwrite: {dest}")
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    atomic_json(prior_file, status)
    for name, cmd, dest in jobs:
        record = {"command": cmd, "cwd": str(REPO), "output": str(dest),
                  "stdout": str(logs / f"{name}.stdout.log"),
                  "stderr": str(logs / f"{name}.stderr.log")}
        if dest.exists():
            record.update(state="skipped_complete", finished_at=now(), returncode=0)
            status["suites"][name] = record
            atomic_json(prior_file, status)
            continue
        record.update(state="running", started_at=now(), returncode=None)
        status["suites"][name] = record
        atomic_json(prior_file, status)
        print(f"[{record['started_at']}] starting {name}", flush=True)
        try:
            with (logs / f"{name}.stdout.log").open("x") as stdout, \
                 (logs / f"{name}.stderr.log").open("x") as stderr:
                code = subprocess.run(cmd, cwd=REPO, stdout=stdout, stderr=stderr,
                                      check=False).returncode
        except Exception as exc:
            code = -1
            record["launch_error"] = repr(exc)
        record.update(state="done" if code == 0 and confirmed_complete(name, dest, git_hash)
                      else "failed", finished_at=now(), returncode=code)
        status["suites"][name] = record
        atomic_json(prior_file, status)
        print(f"[{record['finished_at']}] {name}: {record['state']} (exit {code})", flush=True)
    failures = [name for name, item in status["suites"].items() if item["state"] == "failed"]
    atomic_json(archive / "DONE.json", {"finished_at": now(), "git_hash": git_hash,
                                       "prereg_commits": list(PREREG_COMMITS),
                                       "success": not failures, "failures": failures,
                                       "suite_states": {k: v["state"] for k, v in status["suites"].items()}})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
