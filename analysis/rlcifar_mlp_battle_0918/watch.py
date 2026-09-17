#!/usr/bin/env python3
"""One-screen progress of rlcifar_mlp_battle_0918: per arm, the last task line and ETA.
Reads only the runner logs (not per_task.csv), so it shows no per-arm results beyond the
running mean the runner itself prints."""
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "rlcifar_mlp_battle_0918"
L = OUT / "_launch"
plan = json.loads((L / "plan.json").read_text())
done = 0
for j in plan["jobs"]:
    arm = j["arm"]
    if (OUT / arm / "provenance.json").exists():
        done += 1
        print(f"{arm:6s} done")
        continue
    f = L / "logs" / f"{arm}.log"
    if not f.exists():
        print(f"{arm:6s} pending")
        continue
    lines = [l for l in f.read_text().splitlines() if " task " in l]
    if not lines:
        print(f"{arm:6s} started, no task finished yet  ({time.ctime(f.stat().st_mtime)})")
        continue
    m = re.search(r"task\s+(\d+)/(\d+).*?([\d.]+) ms/step.*ETA (\d+) min", lines[-1])
    print(f"{arm:6s} task {m.group(1)}/{m.group(2)}  {m.group(3)} ms/step  ETA {m.group(4)} min"
          if m else f"{arm:6s} {lines[-1]}")
print(f"done {done}/{len(plan['jobs'])}", "STOP" if (L / "STOP").exists() else "")
