#!/usr/bin/env python3
"""Job scheduler for layer_chimera_cifar_0921 (spec §8).  Start it detached:

    setsid nohup python3 analysis/layer_chimera_cifar_0921/launch.py --plan PLAN.json > LOG 2>&1 &

PLAN.json: {"gpu_max": int, "own_max": int, "reserve_gb": float, "min_gpu_mb": int, "retries": int,
            "jobs": [{"cell", "rss_gb", "threads"}, ...]}
One job = one cell = one process of src/layer_chimera_cifar_0921.py (R = 10: seeds 0-9 x raw).
The GPU may be shared with another launcher, so the cap counts every CUDA compute process
holding at least `min_gpu_mb` (the desktop's small CUDA clients sit near 100 MB) together with this
launcher's own runs; a job starts only while that count is below gpu_max, this launcher runs fewer
than own_max jobs, and MemAvailable - rss_gb >= reserve_gb.  Jobs start in list order, 25 s apart.
A job whose provenance.json exists is done; plan runs already alive are adopted; a run that dies is
restarted up to `retries` times and resumes from its per-task checkpoint (the runner resumes by
default).  The plan is re-read every cycle; touch <out>/_launch/STOP to stop starting new jobs.
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
OUT = REPO / "results" / "layer_chimera_cifar_0921"
L = OUT / "_launch"
RUNNER = REPO / "src" / "layer_chimera_cifar_0921.py"


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def gpu_pids(min_mb: int) -> set[int]:
    try:
        txt = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    out = set()
    for line in txt.splitlines():
        try:
            pid, mb = (int(x) for x in line.split(","))
        except ValueError:
            continue
        if mb >= min_mb:
            out.add(pid)
    return out


def job_dir(j: dict) -> Path:
    return OUT / j["cell"]


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
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def scan_runs() -> dict[str, int]:
    """arm -> pid of every plan-shaped run of RUNNER in /proc (runs with --out ignored)."""
    found = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            argv = Path(f"/proc/{p}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [x.decode(errors="replace") for x in argv if x]
        if len(argv) < 3 or not argv[1].endswith("layer_chimera_cifar_0921.py") or argv[2] != "run":
            continue
        if "--out" in argv or "--cell" not in argv:
            continue
        found[argv[argv.index("--cell") + 1]] = int(p)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    (L / "logs").mkdir(parents=True, exist_ok=True)
    (L / "launcher.pid").write_text(str(os.getpid()))
    running: dict[str, dict] = {}
    tries: dict[str, int] = {}
    failed: set[str] = set()
    log({"ev": "launcher_start", "plan": a.plan, "pid": os.getpid()})
    while True:
        plan = json.loads(Path(a.plan).read_text())
        jobs = {j["cell"]: j for j in plan["jobs"]}
        for name, pid in scan_runs().items():
            if name in jobs and name not in running:
                running[name] = {"job": jobs[name], "t0": time.time(), "pid": pid}
                log({"ev": "adopt", "job": name, "pid": pid})
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
                tries[name] = tries.get(name, 0) + 1
                if tries[name] > plan.get("retries", 2):
                    failed.add(name)
            log({"ev": "done" if ok else "FAILED", "job": name, "rc": rc, "tries": tries.get(name, 0),
                 "minutes": round((time.time() - r["t0"]) / 60, 1)})
        pending = [j for n, j in jobs.items()
                   if n not in running and n not in failed
                   and not (job_dir(j) / "provenance.json").exists()]
        if not pending and not running:
            log({"ev": "launcher_exit", "failed": sorted(failed)})
            return
        started = False
        if not (L / "STOP").exists() and pending and len(running) < plan["own_max"]:
            own = {r["proc"].pid if "proc" in r else r["pid"] for r in running.values()}
            gpu_n = len(gpu_pids(plan.get("min_gpu_mb", 500)) | own)
            j = pending[0]
            avail = mem_available_gb()
            if gpu_n < plan["gpu_max"] and avail - j["rss_gb"] >= plan["reserve_gb"]:
                name = j["cell"]
                cmd = [sys.executable, str(RUNNER), "run", "--cell", name,
                       "--threads", str(j.get("threads", 2))]
                fh = open(L / "logs" / f"{name}.log", "a")
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, env=env)
                running[name] = {"job": j, "t0": time.time(), "proc": p}
                log({"ev": "start", "job": name, "pid": p.pid, "mem_avail_gb": round(avail, 2),
                     "gpu_n": gpu_n + 1, "own": len(running)})
                started = True
        time.sleep(25 if started else 20)


if __name__ == "__main__":
    main()
