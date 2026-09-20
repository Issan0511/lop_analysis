#!/usr/bin/env python3
"""relu_doors_0919 -- launch the four arms (spec §8).

Four processes, one arm each (R = 10, seeds 0-9, input raw).  Each writes ckpt.pt every task,
so a killed job is restarted from where it stopped (up to `retries`).  A STOP file in
`_launch/` stops new launches without touching what is already running -- use it before
editing src/ or analysis/ during a run (the provenance git state is taken at each job's start).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "relu_doors_0919"
LAUNCH = OUT / "_launch"
ARMS = ("C", "CH", "CHB", "CHB0")
LAM = 0.1713                      # spec §2.2, derived from the probe; do not change


def ev(rec: dict) -> None:
    LAUNCH.mkdir(parents=True, exist_ok=True)
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with open(LAUNCH / "events.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    arms = [s for s in a.arms.split(",") if s]
    LAUNCH.mkdir(parents=True, exist_ok=True)
    (LAUNCH / "logs").mkdir(exist_ok=True)
    (LAUNCH / "plan.json").write_text(json.dumps(
        {"arms": arms, "seeds": a.seeds, "tasks": a.tasks, "lam": LAM,
         "note": "1 process per arm, R=10, input raw"}, indent=1))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    procs, tries = {}, {arm: 0 for arm in arms}

    def start(arm: str) -> None:
        if (LAUNCH / "STOP").exists():
            ev({"ev": "stopped_by_file", "job": arm})
            return
        log = open(LAUNCH / "logs" / f"{arm}.log", "a")
        cmd = ["python3", str(REPO / "src" / "relu_doors_0919.py"), "run", "--arm", arm,
               "--seeds", a.seeds, "--tasks", str(a.tasks), "--out", str(OUT / arm)]
        if arm in ("CHB",):
            cmd += ["--lam", str(LAM)]
        procs[arm] = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        ev({"ev": "start", "job": arm, "try": tries[arm], "pid": procs[arm].pid})

    for arm in arms:
        start(arm)
    t0 = time.time()
    while procs:
        time.sleep(20)
        for arm, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[arm]
            mins = (time.time() - t0) / 60
            if rc == 0:
                ev({"ev": "done", "job": arm, "rc": 0, "minutes": round(mins, 1)})
            elif tries[arm] < a.retries:
                tries[arm] += 1
                ev({"ev": "retry", "job": arm, "rc": rc, "try": tries[arm]})
                start(arm)                      # resumes from ckpt.pt
            else:
                ev({"ev": "gave_up", "job": arm, "rc": rc})
    ev({"ev": "launcher_exit"})


if __name__ == "__main__":
    main()
