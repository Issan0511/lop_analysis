#!/usr/bin/env python3
"""Run the nine preregistered jobs; wait on process exit, without status polling."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[2]
JOBS = [(a, "base") for a in ("R", "GELU", "ELU", "LR", "LK001", "SNA")]
JOBS += [("GELU", "early"), ("GELU", "late"), ("ELU", "early")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2, choices=(1, 2))
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "results/erosion_race_0919")
    args = ap.parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "src", "analysis", "specs"], cwd=REPO, text=True)
    if dirty.strip():
        raise SystemExit("Commit source, analysis, and preregistration before main runs.\n" + dirty)
    dest = args.out.resolve()
    launch = dest / "_launch"
    launch.mkdir(parents=True, exist_ok=True)
    if (launch / "plan.json").exists():
        raise SystemExit("Existing launch plan: refusing to overwrite/repeat a main run.")
    for arm, mode in JOBS:
        if (dest / f"{arm}_{mode}").exists():
            raise SystemExit(f"Existing output for {arm}/{mode}; inspect before rerunning.")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    spec = REPO / "specs/spec_erosion_race_0919.md"
    plan = {"git_hash": sha, "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
            "seeds": list(range(1001, 1006)), "tasks": 2, "epochs": 400,
            "jobs": JOBS, "workers": args.workers, "python": sys.executable,
            "data_dir": str(args.data_dir.resolve())}
    (launch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    lock = threading.Lock()

    def event(**kw):
        row = {"utc": datetime.now(timezone.utc).isoformat(), **kw}
        line = json.dumps(row)
        with lock:
            with (launch / "events.jsonl").open("a") as f:
                f.write(line + "\n")
            print(line, flush=True)

    def run_job(job):
        arm, mode = job
        name = f"{arm}_{mode}"
        if (launch / "STOP").exists():
            event(event="not_started", job=name, reason="STOP file")
            return 1
        cmd = [sys.executable, "-m", "src.erosion_race_0919", "--arm", arm,
               "--mode", mode, "--seeds", "1001,1002,1003,1004,1005",
               "--tasks", "2", "--epochs", "400", "--out", str(dest / name),
               "--data-dir", str(args.data_dir.resolve())]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        event(event="start", job=name, command=cmd)
        with (launch / f"{name}.log").open("w") as f:
            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        event(event="done" if result.returncode == 0 else "failed", job=name,
              returncode=result.returncode)
        if result.returncode:
            (launch / "STOP").touch()
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = [f.result() for f in as_completed([pool.submit(run_job, j) for j in JOBS])]
    event(event="launcher_exit", successful=sum(c == 0 for c in codes), planned=len(JOBS))
    raise SystemExit(int(any(codes)))


if __name__ == "__main__":
    main()
