#!/usr/bin/env python3
"""Job scheduler for altlabels_cifar_0923 (spec §2, §7.4), forked from le_eps_cifar_0922/launch.py.
Start it detached:

    setsid nohup python3 analysis/altlabels_cifar_0923/launch.py --plan analysis/altlabels_cifar_0923/plan.json \
        > results/altlabels_cifar_0923/_launch/launcher.log 2>&1 &

PLAN.json: {"gpu_max": int, "own_max": int, "reserve_gb": float, "min_gpu_mb": int, "retries": int,
            "jobs": [{"arm", "activation", "schedule", "seeds", "conds", "tasks", "device",
                      "threads", "rss_gb", "hit_every", "keep_ckpts", "stop"}, ...]}
One job = one process of src/altlabels_cifar_0923.py run --out OUT/<arm>.  `arm` is the job's
directory name (LR_abab, ...), `activation` the battle arm the engine trains (LR, SNA).
The GPU may be shared with another launcher, so the cap counts every CUDA compute process holding
at least `min_gpu_mb` (the desktop's small CUDA clients sit near 100 MB) together with this
launcher's own runs; a job starts only while that count is below gpu_max, this launcher runs fewer
than own_max jobs, and MemAvailable - rss_gb >= reserve_gb.  Jobs start in list order, 25 s apart.
A job counts as done only when its provenance.json agrees with the planned configuration (see
done()); a mismatch is refused rather than adopted.  Plan runs already alive are adopted by their
--out path; a run that dies is restarted up to `retries` times and resumes from its per-task
ckpt.pt (the runner resumes by default, and refuses a checkpoint from another setting).  The
runner also takes an exclusive lock on its output directory, so two launchers cannot both write
one.  The plan is re-read every cycle; touch <out>/_launch/STOP to stop starting new jobs.

A `"device": "cpu"` job does not count against gpu_max.  plan_stop.json's two stop chains run on
cuda (R = 1, eager, one seed after another in one process): check S5b shows the cpu reproduces
neither the reference chain nor itself once the stop rule feeds task lengths forward.
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
OUT = REPO / "results" / "altlabels_cifar_0923"
L = OUT / "_launch"
RUNNER = REPO / "src" / "altlabels_cifar_0923.py"


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
    return OUT / j["arm"]


# what a finished job's provenance must agree with before it counts as done.  Without this a
# plan edited between launcher runs would adopt results produced under the old settings.
JOB_KEYS = {"run": (("arm", "activation"), ("schedule", "schedule"), ("n_tasks", "tasks"),
                    ("hit_every", "hit_every"), ("keep_ckpts", "keep_ckpts")),
            "chain": (("arm", "activation"), ("schedule", "schedule"), ("n_tasks", "tasks"),
                      ("hit_every", "hit_every"))}


def done(j: dict) -> bool:
    """True when job_dir(j) holds a finished run of THIS configuration; refuse a mismatch."""
    p = job_dir(j) / "provenance.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    stage = j.get("stage", "run")
    bad = {pk: (d.get(pk), j.get(jk)) for pk, jk in JOB_KEYS[stage]
           if jk in j and d.get(pk) != j[jk]}
    want_stop = [float(x) if i == 0 else int(x)
                 for i, x in enumerate(str(j["stop"]).split(","))] if j.get("stop") else None
    if (d.get("stop") or None) != want_stop:
        bad["stop"] = (d.get("stop"), want_stop)
    if (d.get("device") or "").split(":")[0] != ("cuda" if j.get("device", "auto") != "cpu"
                                                 else "cpu"):
        bad["device"] = (d.get("device"), j.get("device"))
    if bad:
        raise SystemExit(f"{job_dir(j)} was produced with another configuration {bad} "
                         f"(found, planned); move it aside or rename the job")
    return True


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
    """resolved --out path -> pid, for every run of RUNNER in /proc under OUT.

    `arm` may name a subdirectory (the stop chains are one process per seed,
    LR_iid_stop/seed3), so runs are matched on the whole --out path, not on its basename.
    """
    found = {}
    root = str(OUT.resolve())
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            argv = Path(f"/proc/{p}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [x.decode(errors="replace") for x in argv if x]
        # the script may sit anywhere in argv (python -u script.py run ...), and the stage
        # token is whatever follows it
        ix = next((i for i, x in enumerate(argv) if x.endswith("altlabels_cifar_0923.py")), None)
        if ix is None or ix + 1 >= len(argv) or argv[ix + 1] not in ("run", "chain"):
            continue
        if "--out" not in argv:
            continue
        o = Path(argv[argv.index("--out") + 1])
        if not o.is_absolute():                 # relative to THAT process's cwd, not ours
            try:
                o = Path(os.readlink(f"/proc/{p}/cwd")) / o
            except OSError:
                continue
        o = str(o.resolve())
        if o.startswith(root + os.sep):
            found[o] = int(p)
    return found


def cmd_of(j: dict) -> list[str]:
    c = [sys.executable, str(RUNNER), j.get("stage", "run"),
         "--arm", j["activation"], "--schedule", j["schedule"],
         "--seeds", str(j.get("seeds", "0-9")), "--conds", str(j.get("conds", "std")),
         "--tasks", str(j.get("tasks", 50)), "--device", j.get("device", "auto"),
         "--threads", str(j.get("threads", 2)), "--out", str(OUT / j["arm"])]
    if j.get("hit_every"):
        c += ["--hit-every", str(j["hit_every"])]
    if j.get("keep_ckpts"):
        c += ["--keep-ckpts"]
    if j.get("stop"):
        c += ["--stop", str(j["stop"])]
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    (L / "logs").mkdir(parents=True, exist_ok=True)
    (L / f"launcher_{Path(a.plan).stem}.pid").write_text(str(os.getpid()))
    running: dict[str, dict] = {}
    tries: dict[str, int] = {}
    failed: set[str] = set()
    log({"ev": "launcher_start", "plan": a.plan, "pid": os.getpid()})
    while True:
        plan = json.loads(Path(a.plan).read_text())
        jobs = {j["arm"]: j for j in plan["jobs"]}
        by_out = {str(job_dir(j).resolve()): j["arm"] for j in plan["jobs"]}
        for path, pid in scan_runs().items():
            name = by_out.get(path)
            if name is not None and name not in running:
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
            ok = (rc in (0, None)) and done(r["job"])
            if not ok:
                tries[name] = tries.get(name, 0) + 1
                if tries[name] > plan.get("retries", 2):
                    failed.add(name)
            log({"ev": "done" if ok else "FAILED", "job": name, "rc": rc, "tries": tries.get(name, 0),
                 "minutes": round((time.time() - r["t0"]) / 60, 1)})
        pending = [j for n, j in jobs.items()
                   if n not in running and n not in failed and not done(j)]
        if not pending and not running:
            log({"ev": "launcher_exit", "failed": sorted(failed)})
            return
        started = False
        if not (L / "STOP").exists() and pending and len(running) < plan["own_max"]:
            own = {r["proc"].pid if "proc" in r else r["pid"] for r in running.values()
                   if r["job"].get("device", "auto") != "cpu"}
            gpu_n = len(gpu_pids(plan.get("min_gpu_mb", 500)) | own)
            j = pending[0]
            avail = mem_available_gb()
            on_gpu = j.get("device", "auto") != "cpu"
            if (gpu_n < plan["gpu_max"] or not on_gpu) and avail - j["rss_gb"] >= plan["reserve_gb"]:
                name = j["arm"]
                # a job name may contain a '/' (LR_iid_stop/seed3): never in a log file name
                fh = open(L / "logs" / (name.replace("/", "_") + ".log"), "a")
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                p = subprocess.Popen(cmd_of(j), cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, env=env)
                running[name] = {"job": j, "t0": time.time(), "proc": p}
                log({"ev": "start", "job": name, "pid": p.pid, "mem_avail_gb": round(avail, 2),
                     "gpu_n": gpu_n + int(on_gpu), "own": len(running), "cmd": cmd_of(j)})
                started = True
        time.sleep(25 if started else 20)


if __name__ == "__main__":
    main()
