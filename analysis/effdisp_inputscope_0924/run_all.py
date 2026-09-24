#!/usr/bin/env python3
"""Run registered input-scope suites sequentially on one GPU.

The launcher records a single source commit at startup and passes it to every
trainer. It never reads results or runs analysis. An existing incomplete raw
suite is refused rather than overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/home/issan/Projects/obsidian-research-data/effdisp_inputscope_0924")
SEEDS = (200, 201, 202, 203, 204)
CELLS = ("X0Y0", "X0Y1", "X1Y0", "X1Y1")
PREREG = ("2e9f878", "cee50cc")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def jobs(raw: Path, source_hash: str) -> list[tuple[str, list[str], Path]]:
    conda = raw / "conda"
    result = [("conda", [sys.executable, "-u", "-m", "src.effdisp_inputscope_conda_0924",
                         "--out", str(conda), "--tasks", "400", "--period", "10000",
                         "--seeds", *map(str, SEEDS), "--device", "cuda",
                         "--source-git-hash", source_hash], conda)]
    for act in ("LR", "SNA06"):
        for cell in CELLS:
            dest = raw / "mnist" / f"{act}_{cell}"
            tasks = 150 if cell == "X0Y1" else 50
            result.append((f"{act}_{cell}",
                           [sys.executable, "-u", "src/effdisp_inputscope_mnist_0924.py",
                            "--act", act, "--cell", cell, "--tasks", str(tasks),
                            "--epochs", "400", "--seeds", *map(str, SEEDS),
                            "--device", "cuda", "--source-git-hash", source_hash,
                            "--out", str(dest)], dest))
    for act in ("SN05", "SNA03"):
        cell, dest = "X0Y1", raw / "mnist" / f"{act}_X0Y1"
        result.append((f"{act}_{cell}",
                       [sys.executable, "-u", "src/effdisp_inputscope_mnist_0924.py",
                        "--act", act, "--cell", cell, "--tasks", "50",
                        "--epochs", "400", "--seeds", *map(str, SEEDS),
                        "--device", "cuda", "--source-git-hash", source_hash,
                        "--out", str(dest)], dest))
    return result


def confirmed(name: str, dest: Path, source_hash: str) -> bool:
    try:
        meta = json.loads((dest / "metadata.json").read_text())
        if meta["git_hash"] != source_hash:
            return False
        if name == "conda":
            args = meta["args"]
            if args["tasks"] != 400 or args["period"] != 10000 or tuple(args["seeds"]) != SEEDS:
                return False
            arms = [a["name"] for a in meta["arms"]]
            if len(arms) != 16 or len(set(arms)) != 16:
                return False
            for arm in arms:
                for seed in SEEDS:
                    sd = dest / arm / f"seed{seed}"
                    status = json.loads((sd / "status.json").read_text())
                    if status["status"] != "COMPLETED" or status["last_saved_task"] != 400:
                        return False
                    if not (sd / "t000.npz").is_file() or not (sd / "t400.npz").is_file():
                        return False
            return True
        act, cell = name.split("_", 1)
        tasks = 150 if cell == "X0Y1" and act in ("LR", "SNA06") else 50
        if (meta["activation"] != act or meta["cell"] != cell or meta["tasks"] != tasks
                or tuple(meta["seeds"]) != SEEDS or meta["epochs"] != 400
                or meta["steps_per_task"] != 30000 or meta["optimizer"] != "adam"
                or meta["lr"] != 0.001 or meta["smoke"]):
            return False
        statuses = json.loads((dest / "status.json").read_text())
        if set(statuses) != {str(seed) for seed in SEEDS}:
            return False
        for seed in SEEDS:
            sd = dest / f"seed_{seed:03d}"
            item = statuses[str(seed)]
            if item["state"] != "complete" or item["completed_tasks"] != tasks:
                return False
            if not all((sd / filename).is_file() for filename in
                       ("input_bank.npz", "state_000.npz", f"state_{tasks:03d}.npz",
                        "task_001_input.npz", f"task_{tasks:03d}_input.npz", "per_task.json")):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--dry-run", action="store_true", help="print commands without writing or training")
    ap.add_argument("--resume", action="store_true", help="skip only confirmed complete suites")
    args = ap.parse_args()
    archive = args.archive.expanduser().resolve()
    raw, logs = archive / "raw", archive / "logs"
    source_hash = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=REPO, text=True).strip()
    queue = jobs(raw, source_hash)
    if args.dry_run:
        for name, command, dest in queue:
            print(json.dumps({"suite": name, "command": command, "cwd": str(REPO),
                              "output": str(dest)}))
        return 0
    for commit in PREREG:
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, source_hash],
                          cwd=REPO, check=False).returncode:
            ap.error(f"preregistered commit {commit} is not an ancestor of HEAD")
    status_path = archive / "run_status.json"
    if not args.resume and (raw.exists() or logs.exists() or status_path.exists()):
        ap.error(f"existing raw/log/status at {archive}; use --resume")
    if args.resume and raw.exists() and not status_path.is_file():
        ap.error("--resume requires run_status.json alongside existing raw/")
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("git_hash") != source_hash or status.get("prereg_commits") != list(PREREG):
            ap.error("existing run_status.json has different source/prereg commits")
    else:
        code_files = ("src/effdisp_inputscope_conda_0924.py",
                      "src/effdisp_inputscope_mnist_0924.py",
                      "analysis/effdisp_inputscope_0924/run_all.py")
        status = {"started_at": utc_now(), "git_hash": source_hash,
                  "prereg_commits": list(PREREG), "cwd": str(REPO),
                  "python": sys.executable,
                  "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                                    for name in code_files},
                  "archive": str(archive), "suites": {}}
    # Validate the entire resume before any subprocess starts.
    for name, _, dest in queue:
        if dest.exists() and not (args.resume and confirmed(name, dest, source_hash)):
            ap.error(f"existing suite is incomplete or mismatched: {dest}")
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    atomic_json(status_path, status)
    for name, command, dest in queue:
        record = {"command": command, "cwd": str(REPO), "output": str(dest),
                  "stdout": str(logs / f"{name}.stdout.log"),
                  "stderr": str(logs / f"{name}.stderr.log")}
        if dest.exists():
            record.update(state="skipped_complete", finished_at=utc_now(), returncode=0)
        else:
            record.update(state="running", started_at=utc_now(), returncode=None)
            status["suites"][name] = record
            atomic_json(status_path, status)
            print(f"[{record['started_at']}] starting {name}", flush=True)
            try:
                with (logs / f"{name}.stdout.log").open("x") as stdout, \
                     (logs / f"{name}.stderr.log").open("x") as stderr:
                    code = subprocess.run(command, cwd=REPO, stdout=stdout, stderr=stderr,
                                          check=False).returncode
            except Exception as exc:
                code = -1
                record["launch_error"] = repr(exc)
            record.update(state="done" if code == 0 and confirmed(name, dest, source_hash)
                          else "failed", finished_at=utc_now(), returncode=code)
            print(f"[{record['finished_at']}] {name}: {record['state']} (exit {code})", flush=True)
        status["suites"][name] = record
        atomic_json(status_path, status)
    failures = [name for name, item in status["suites"].items() if item["state"] == "failed"]
    atomic_json(archive / "DONE.json", {"finished_at": utc_now(), "git_hash": source_hash,
                                       "prereg_commits": list(PREREG),
                                       "success": not failures, "failures": failures,
                                       "suite_states": {name: item["state"]
                                                        for name, item in status["suites"].items()}})
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
