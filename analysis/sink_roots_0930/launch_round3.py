#!/usr/bin/env python3
"""sink_roots_0930 round 3 (spec_sink_roots_0930_round3.md): R1p, R2p, R3p in one queue.
A job starts when its dependencies' `.done` markers exist and the number of my experiment processes on the machine
(python processes running a script under this worktree's src/, i.e. R7 and every round; paused ones not counted) is below the budget
`total` of results/sink_roots_0930/caps_round2.json (read every loop).  `.done` is written on rc 0.
results/sink_roots_0930/STOP_round3 stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R2R = RAW / "round3"
RES = ROOT / "results" / "sink_roots_0930"
LOG = RES / "launch_round3_log.jsonl"
STOP = RES / "STOP_round3"
CAPS = RES / "caps_round2.json"
PY = sys.executable
SRC = ROOT / "src"
S3 = SRC / "sink_roots_round3"
ENG = [PY, str(SRC / "sink_roots_mnist_0930.py")]
MARK = str(SRC) + "/"


def budget():
    try:
        return int(json.loads(CAPS.read_text())["total"])
    except Exception:
        return 12


def my_procs():
    n = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if args and b"python" in args[0] and any(MARK.encode() in x for x in args[1:3]):
            try:
                state = (p / "stat").read_text().rsplit(")", 1)[1].split()[0]
            except OSError:
                continue
            if state not in ("T", "t"):              # a job paused with SIGSTOP does not use a core
                n += 1
    return n


def job(name, cmd, deps=(), cwd=None, env=None, out=None):
    out = Path(out) if out else R2R / "jobs" / name
    return {"name": name, "cmd": [str(c) for c in cmd], "deps": [Path(d) for d in deps], "cwd": cwd or ROOT,
            "env": env or {}, "out": out}


def jobs():
    J = []
    for s in range(5):                                                       # R1p
        J.append(job(f"R1p_s{s}", [PY, S3 / "cifar_path_ledger.py", "--seed", s, "--tasks", 3, "--every", 100,
                                   "--out", R2R / "R1p"]))
    for act in ("ELU", "GELU", "SILU", "LR"):                                # R2p
        for s in (0, 1, 2):
            J.append(job(f"R2p_{act}_s{s}", [PY, S3 / "adam" / "branch_probe.py", "--act", act, "--seed", s,
                                             "--switches", "3,10,20", "--out", R2R / "R2p" / "branches"], cwd=S3 / "adam"))
    for kind, tgts in (("pold", ("0.26", "0.43", "0.57")), ("lsd", ("1", "2.4", "7"))):   # R3p
        for tgt in tgts:
            for act in ("ELU", "SILU", "LR"):
                for s in (0, 1, 2):
                    n = f"R3p_{kind}{tgt}_{act}_s{s}"
                    J.append(job(n, ENG + ["--name", n, "--out", RAW / "mnist" / n, "--act", act, "--seed", s, "--tasks", 30,
                                           "--T", 16000, "--stop", kind, "--stop-at", tgt], out=RAW / "mnist" / n))
    return J


def main():
    J = [j for j in jobs() if not (j["out"] / ".done").exists()]
    print(f"{len(J)} round-3 jobs", flush=True)
    (R2R / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while J or running:
        for name, (p, j, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                fh.close()
                if rc == 0:
                    (j["out"] / ".done").write_text(f"{time.time()}\n")
                with LOG.open("a") as f:
                    f.write(json.dumps({"name": name, "rc": rc, "start": t0, "end": time.time(), "cmd": j["cmd"]}) + "\n")
                del running[name]
        if not STOP.exists():
            for j in list(J):
                if my_procs() >= budget():
                    break
                if not all(d.exists() for d in j["deps"]):
                    continue
                j["out"].mkdir(parents=True, exist_ok=True)
                fh = open(R2R / "logs" / f"{j['name']}.log", "w")
                env = dict(os.environ, **j["env"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=j["cwd"], env=env)
                running[j["name"]] = (p, j, time.time(), fh)
                J.remove(j)
                time.sleep(0.5)                       # let the new process appear in /proc before counting again
        elif not running:
            break
        time.sleep(5)
    print("round-3 queue finished" if not J else f"stopped with {len(J)} left", flush=True)


if __name__ == "__main__":
    main()
