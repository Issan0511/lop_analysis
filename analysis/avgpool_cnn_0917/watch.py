#!/usr/bin/env python3
"""Emit one line per event worth acting on (for a Monitor): a FAILED run, all runs done,
the launcher exiting or dying, MemAvailable under 4 GB.  Polls every 60 s."""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "avgpool_cnn_0917"
EV = OUT / "_launch" / "events.jsonl"
PLAN = OUT / "_launch" / "plan.json"


def counts():
    jobs = json.loads(PLAN.read_text())["jobs"]
    done = sum((OUT / j["arm"] / f"seed{j['seed']}" / "provenance.json").exists() for j in jobs)
    return done, len(jobs)


def mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20


def launcher_alive():
    try:
        os.kill(int((OUT / "_launch" / "launcher.pid").read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


seen = EV.stat().st_size if EV.exists() else 0
done, total = counts()
reported, low, dead = done >= total, False, False
print(f"watch start {time.strftime('%T')}: {done}/{total} mem {mem_gb():.1f}G", flush=True)
while True:
    if EV.exists() and EV.stat().st_size > seen:
        with open(EV) as fh:
            fh.seek(seen)
            chunk = fh.read()
        seen += len(chunk.encode())
        for line in chunk.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("ev") in ("FAILED", "launcher_exit"):
                print(f"{e['t']} {e['ev']} {e.get('job', '')} {e.get('failed', '')}", flush=True)
    done, total = counts()
    if not reported and done >= total:
        reported = True
        print(f"{time.strftime('%T')} ALL_DONE {done}/{total}", flush=True)
    m = mem_gb()
    if m < 4 and not low:
        low = True
        print(f"{time.strftime('%T')} LOW_MEMORY {m:.1f}G", flush=True)
    elif m > 6:
        low = False
    if not launcher_alive() and not dead and not reported:
        dead = True
        print(f"{time.strftime('%T')} LAUNCHER_GONE {done}/{total}", flush=True)
    time.sleep(60)
