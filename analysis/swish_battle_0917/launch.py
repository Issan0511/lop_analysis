#!/usr/bin/env python3
"""Job scheduler for swish_battle_0917.  Start it detached:

    setsid nohup python3 analysis/swish_battle_0917/launch.py --plan PLAN.json > LOG 2>&1 &

PLAN.json: {"gpu_max": int, "cpu_threads_max": int, "reserve_gb": float,
            "env": {"cuda": {...}, "cpu": {...}},            # optional, per device
            "jobs": [{"box", "arm", "seed", "device", "threads", "rss_gb"}, ...]}
Jobs start in list order.  A job starts only when its device has room (cuda: processes
< gpu_max; cpu: threads in use + its threads <= cpu_threads_max) and
MemAvailable - rss_gb >= reserve_gb.  Starts are serialized with a 25 s wait so each new
process's RSS shows up in MemAvailable before the next decision (memory note on shared
machines).  A job whose provenance.json exists is skipped, so a restart resumes; plan
jobs already running (found in /proc, e.g. started by a previous launcher) are adopted
and counted against the caps, so a restart never duplicates a run.  The plan file is
re-read every cycle: caps and env can be changed while it runs.  Touch
<out>/_launch/STOP to stop starting new jobs (running ones finish).
"""
from __future__ import annotations

import argparse
import json
import os
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


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a zombie child of ours still answers kill(0); poll() handles those
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def scan_runs() -> dict[str, int]:
    """name -> pid of every plan-shaped run of RUNNER in /proc.  Runs with --out are
    smoke runs writing elsewhere and are ignored."""
    found = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            argv = Path(f"/proc/{p}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [x.decode(errors="replace") for x in argv if x]
        if len(argv) < 3 or not argv[1].endswith("swish_battle_0917.py") or argv[2] != "run":
            continue
        if "--out" in argv:
            continue
        kv = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith("--")}
        try:
            name = f"{kv['--box']}_{kv['--arm']}_seed{kv['--seed']}_{kv['--device']}"
        except KeyError:
            continue
        found[name] = int(p)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    (L / "logs").mkdir(parents=True, exist_ok=True)
    (L / "launcher.pid").write_text(str(os.getpid()))
    running: dict[str, dict] = {}          # name -> {"job", "t0", "proc" or "pid"}
    failed: set[str] = set()
    log({"ev": "launcher_start", "plan": a.plan, "pid": os.getpid()})
    while True:
        plan = json.loads(Path(a.plan).read_text())
        jobs = {job_name(j): j for j in plan["jobs"]}
        # adopt plan runs that are alive but not ours
        for name, pid in scan_runs().items():
            if name in jobs and name not in running:
                running[name] = {"job": jobs[name], "t0": time.time(), "pid": pid}
                log({"ev": "adopt", "job": name, "pid": pid})
        # reap
        for name, r in list(running.items()):
            if "proc" in r:
                rc = r["proc"].poll()
                if rc is None:
                    continue
            else:
                if alive(r["pid"]):
                    continue
                rc = None
            del running[name]
            ok = (rc in (0, None)) and (job_dir(r["job"]) / "provenance.json").exists()
            if not ok:
                failed.add(name)
            log({"ev": "done" if ok else "FAILED", "job": name, "rc": rc,
                 "minutes": round((time.time() - r["t0"]) / 60, 1)})
        pending = [j for n, j in jobs.items()
                   if n not in running and n not in failed
                   and not (job_dir(j) / "provenance.json").exists()]
        if not pending and not running:
            log({"ev": "launcher_exit", "failed": sorted(failed)})
            return
        started = False
        if not (L / "STOP").exists():
            gpu_n = sum(1 for r in running.values() if r["job"]["device"] == "cuda")
            cpu_t = sum(r["job"]["threads"] for r in running.values() if r["job"]["device"] == "cpu")
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
                env = {**os.environ, **plan.get("env", {}).get(j["device"], {})}
                fh = open(L / "logs" / f"{name}.log", "a")
                p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
                running[name] = {"job": j, "t0": time.time(), "proc": p}
                log({"ev": "start", "job": name, "pid": p.pid, "mem_avail_gb": round(avail, 2),
                     "gpu_n": gpu_n + (j["device"] == "cuda"),
                     "cpu_threads": cpu_t + (j["threads"] if j["device"] == "cpu" else 0),
                     "env": plan.get("env", {}).get(j["device"], {})})
                started = True
                break
        time.sleep(25 if started else 20)


if __name__ == "__main__":
    main()
