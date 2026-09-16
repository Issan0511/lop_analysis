#!/usr/bin/env python3
"""Job scheduler for swish_battle_0917.  Start it detached:

    setsid nohup python3 analysis/swish_battle_0917/launch.py --plan PLAN.json > LOG 2>&1 &

PLAN.json: {"gpu_max": int, "cpu_threads_max": int, "reserve_gb": float,
            "jobs": [{"box", "arm", "seed", "device", "threads", "rss_gb"}, ...]}
Jobs start in list order within each device.  A job starts only when its device has
room (gpu: processes < gpu_max; cpu: threads in use + its threads <= cpu_threads_max)
and MemAvailable - rss_gb >= reserve_gb.  Starts are serialized with a 25 s wait so each
new process's RSS shows up in MemAvailable before the next decision (memory note on
shared machines).  A job whose provenance.json exists is skipped, so a restart resumes.
The plan file is re-read every cycle: caps can be changed while it runs.  Touch
<out>/_launch/STOP to stop starting new jobs (running ones finish).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "swish_battle_0917"
L = OUT / "_launch"
RUNNER = REPO / "src" / "swish_battle_0917.py"


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def job_dir(j: dict) -> Path:
    return OUT / j["box"] / j["arm"] / f"seed{j['seed']}"


def job_name(j: dict) -> str:
    return f"{j['box']}_{j['arm']}_seed{j['seed']}_{j['device']}"


def log(ev: dict) -> None:
    ev = {"t": time.strftime("%F %T"), **ev}
    with open(L / "events.jsonl", "a") as fh:
        fh.write(json.dumps(ev) + "\n")
    print(json.dumps(ev), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    (L / "logs").mkdir(parents=True, exist_ok=True)
    (L / "launcher.pid").write_text(str(__import__("os").getpid()))
    running: dict[str, tuple[subprocess.Popen, dict, float]] = {}
    failed: set[str] = set()
    log({"ev": "launcher_start", "plan": a.plan})
    while True:
        plan = json.loads(Path(a.plan).read_text())
        # reap
        for name, (p, j, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[name]
                ok = rc == 0 and (job_dir(j) / "provenance.json").exists()
                if not ok:
                    failed.add(name)
                log({"ev": "done" if ok else "FAILED", "job": name, "rc": rc,
                     "minutes": round((time.time() - t0) / 60, 1)})
        pending = [j for j in plan["jobs"]
                   if job_name(j) not in running and job_name(j) not in failed
                   and not (job_dir(j) / "provenance.json").exists()]
        if not pending and not running:
            log({"ev": "launcher_exit", "failed": sorted(failed)})
            return
        started = False
        if not (L / "STOP").exists():
            gpu_n = sum(1 for _, j, _ in running.values() if j["device"] == "cuda")
            cpu_t = sum(j["threads"] for _, j, _ in running.values() if j["device"] == "cpu")
            for j in pending:
                if j["device"] == "cuda" and gpu_n >= plan["gpu_max"]:
                    continue
                if j["device"] == "cpu" and cpu_t + j["threads"] > plan["cpu_threads_max"]:
                    continue
                avail = mem_available_gb()
                if avail - j["rss_gb"] < plan["reserve_gb"]:
                    break                              # memory, not slots: wait
                name = job_name(j)
                cmd = [sys.executable, str(RUNNER), "run", "--box", j["box"], "--arm", j["arm"],
                       "--seed", str(j["seed"]), "--device", j["device"],
                       "--threads", str(j["threads"])]
                fh = open(L / "logs" / f"{name}.log", "a")
                p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                running[name] = (p, j, time.time())
                log({"ev": "start", "job": name, "pid": p.pid, "mem_avail_gb": round(avail, 2),
                     "gpu_n": gpu_n + (j["device"] == "cuda"),
                     "cpu_threads": cpu_t + (j["threads"] if j["device"] == "cpu" else 0)})
                started = True
                break
        time.sleep(25 if started else 20)


if __name__ == "__main__":
    main()
