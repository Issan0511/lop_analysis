#!/usr/bin/env python3
"""Emit one line per event worth acting on (for a Monitor): a FAILED run, a box finishing,
the launcher exiting or dying, MemAvailable under 4 GB, and the other session's mucap
jobs ending (room to raise the caps).  Polls every 60 s."""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "swish_battle_0917"
EV = OUT / "_launch" / "events.jsonl"
TOTAL = {"mlp": 90, "cnn": 50}


def n_done(box):
    return len(list((OUT / box).glob("*/seed*/provenance.json"))) if (OUT / box).exists() else 0


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
reported = {b: n_done(b) >= TOTAL[b] for b in TOTAL}
low = False
others = other_jobs()
dead_reported = False
print(f"watch start {time.strftime('%T')}: mlp {n_done('mlp')}/90 cnn {n_done('cnn')}/50 "
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
    for b in TOTAL:
        if not reported[b] and n_done(b) >= TOTAL[b]:
            reported[b] = True
            print(f"{time.strftime('%T')} BOX_DONE {b} {n_done(b)}/{TOTAL[b]}", flush=True)
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
        print(f"{time.strftime('%T')} LAUNCHER_GONE mlp {n_done('mlp')}/90 cnn {n_done('cnn')}/50", flush=True)
    time.sleep(60)
