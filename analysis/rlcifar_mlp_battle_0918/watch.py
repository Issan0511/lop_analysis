#!/usr/bin/env python3
"""One-screen progress of rlcifar_mlp_battle_0918.

Reads only how many tasks each slot has finished (row counts and file times), never a result
column: the box is judged when it is complete.
"""
import csv
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "rlcifar_mlp_battle_0918"
L = OUT / "_launch"
plan = json.loads((L / "plan.json").read_text())
N_TASKS, R = 50, 20
now = time.time()
done = 0
for j in plan["jobs"]:
    arm = j["arm"]
    d = OUT / arm
    f = d / "per_task.csv"
    if (d / "provenance.json").exists():
        done += 1
        print(f"{arm:6s} done")
        continue
    if not f.exists():
        print(f"{arm:6s} {'started' if (L / 'logs' / (arm + '.log')).exists() else 'pending'}")
        continue
    with f.open() as fh:
        tasks = [int(row["task"]) for row in csv.DictReader(fh)]
    per_slot = len(tasks) / R
    started = (L / "logs" / f"{arm}.log").stat().st_mtime if (L / "logs" / f"{arm}.log").exists() else None
    t0 = min(started or now, (d / "ckpt.pt").stat().st_mtime if (d / "ckpt.pt").exists() else now)
    el = (now - (started or now)) / 60
    rate = el / max(per_slot, 1e-9)
    print(f"{arm:6s} task {per_slot:4.1f}/{N_TASKS}  {el:5.1f} min elapsed, "
          f"{rate:4.1f} min/task, ETA {rate * (N_TASKS - per_slot):5.0f} min")
print(f"done {done}/{len(plan['jobs'])}", "STOP" if (L / "STOP").exists() else "")
