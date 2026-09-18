#!/usr/bin/env python3
"""Emit one line per event worth acting on (for a Monitor): a FAILED run, a box (or the cnn
SNAc3 arm, addendum 3) finishing,
the launcher exiting or dying, MemAvailable under 4 GB, and the other session's mucap
jobs ending (room to raise the caps).  Polls every 60 s."""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "swish_battle_0917"
EV = OUT / "_launch" / "events.jsonl"
PLAN = OUT / "_launch" / "plan.json"
# groups whose completion is worth a message: whole boxes, and cnn SNAc3 (label A, addendum 3)
GROUPS = {"mlp": ("mlp", None), "cnn": ("cnn", None), "cnn_SNAc3": ("cnn", "SNAc3")}


def planned(group):
    box, arm = GROUPS[group]
    jobs = json.loads(PLAN.read_text())["jobs"]
    return [j for j in jobs if j["box"] == box and (arm is None or j["arm"] == arm)]


def n_done(group):
    return sum((OUT / j["box"] / j["arm"] / f"seed{j['seed']}" / "provenance.json").exists()
               for j in planned(group))


def total(group):
    return len(planned(group))


def mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20


def other_jobs():
    n = 0
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                if b"mucap_ee_run_0917" in Path(f"/proc/{p}/cmdline").read_bytes():
                    n += 1
            except OSError:
                pass
    return n


def launcher_alive():
    try:
        pid = int((OUT / "_launch" / "launcher.pid").read_text())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


seen = EV.stat().st_size if EV.exists() else 0
reported = {g: n_done(g) >= total(g) for g in GROUPS}
low = False
others = other_jobs()
dead_reported = False
print(f"watch start {time.strftime('%T')}: " + " ".join(f"{g} {n_done(g)}/{total(g)}" for g in GROUPS) + " "
      f"mem {mem_gb():.1f}G other_jobs {others}", flush=True)
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
    for g in GROUPS:
        if not reported[g] and n_done(g) >= total(g):     # totals follow the live plan
            reported[g] = True
            print(f"{time.strftime('%T')} GROUP_DONE {g} {n_done(g)}/{total(g)}", flush=True)
    m = mem_gb()
    if m < 4 and not low:
        low = True
        print(f"{time.strftime('%T')} LOW_MEMORY {m:.1f}G", flush=True)
    elif m > 6:
        low = False
    o = other_jobs()
    if others > 0 and o == 0:
        print(f"{time.strftime('%T')} OTHER_JOBS_ENDED mem {m:.1f}G", flush=True)
    others = o
    if not launcher_alive() and not dead_reported and not all(reported.values()):
        dead_reported = True
        print(f"{time.strftime('%T')} LAUNCHER_GONE " + " ".join(f"{g} {n_done(g)}/{total(g)}" for g in GROUPS), flush=True)
    time.sleep(60)
