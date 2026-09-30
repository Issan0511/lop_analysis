#!/usr/bin/env python3
"""sink_roots_0930 round 4 (spec_sink_roots_0930_round4.md): priorities 1 and 2 of the round-2 / ratio requests.
A job starts when its dependencies' `.done` markers exist and the number of my experiment processes on the machine
(python processes running a script under this worktree's src/, i.e. R7 and every round; paused ones not counted) is below the budget
`total` of results/sink_roots_0930/caps_round2.json (read every loop).  `.done` is written on rc 0.
results/sink_roots_0930/STOP_round4 stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R2R = RAW / "round4"
RES = ROOT / "results" / "sink_roots_0930"
LOG = RES / "launch_round4_log.jsonl"
STOP = RES / "STOP_round4"
CAPS = RES / "caps_round2.json"
PY = sys.executable
SRC = ROOT / "src"
S4 = SRC / "sink_roots_round4"
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
    def eng(n, args):
        return job(n, ENG + ["--name", n, "--out", RAW / "mnist" / n] + [str(x) for x in args], out=RAW / "mnist" / n)
    S = (0, 1, 2)
    # priority 1: G1 (b)(c) first (the long ones), then RR1, G1 (a), cap_long_GELU
    for mode in ("row", "perp"):
        for s in S:
            J.append(eng(f"G1b_cap50{mode}_ELU_s{s}", ["--act", "ELU", "--seed", s, "--tasks", 100, "--wcap-from", 50,
                                                       "--wcap-mode", mode, "--snap-before", 5, 50, 55, 60, 80, 100]))
    for s in S:
        J.append(eng(f"G1c_rescale50_ELU_s{s}", ["--act", "ELU", "--seed", s, "--tasks", 80, "--rescale-at", 50, "--rescale-ref", 5,
                                                 "--wcap-from", 50, "--snap-before", 5, 50, 55, 60, 80]))
    for arm in ("A", "B"):
        for s in S:
            J.append(eng(f"RR1_{arm}_ELU_s{s}", ["--act", "ELU", "--seed", s, "--tasks", 30, "--rr1", arm, "--rr1-from", 5,
                                                 "--snap-before", 5, 10, 20, 30]))
    for sc, nt in (("0.75", 30), ("1.5", 30), ("3", 60)):
        for s in S:
            snaps = [5, 10, 20, 30] + ([50, 55, 60] if nt == 60 else [])
            J.append(eng(f"G1a_scale{sc}_ELU_s{s}", ["--act", "ELU", "--seed", s, "--tasks", nt, "--wcap-from", 5,
                                                     "--wcap-scale", sc, "--snap-before"] + snaps))
    for frm in (5, 2):
        for s in S:
            J.append(eng(f"capL_from{frm}_GELU_s{s}", ["--act", "GELU", "--seed", s, "--tasks", 200, "--wcap-from", frm]))
    # priority 2
    for act, tag in (("ELUT", "G3tail"), ("LRF", "RR3bfloor")):
        for s in S:
            J.append(eng(f"{tag}_{act}_s{s}", ["--act", act, "--seed", s, "--tasks", 30, "--snap-before", 5, 10, 20, 30]))
    for s in S:
        J.append(eng(f"G2same_ELU_s{s}", ["--act", "ELU", "--seed", s, "--tasks", 30, "--same-labels", "--snap-before", 5, 10, 20, 30]))
    for sig, tag in (("0.002", "2eta"), ("0.005", "5eta")):
        for s in S:
            J.append(eng(f"RR2noise{tag}_ELU_s{s}", ["--act", "ELU", "--seed", s, "--tasks", 30, "--noise-sigma", sig,
                                                      "--noise-from-task", 5, "--noise-from-step", 200]))
    for act in ("GELU", "SILU"):
        for s in (0, 1):
            J.append(job(f"Reps0_{act}_s{s}", [PY, S4 / "valley" / "alpha_ladder_1layer.py", "--act", act, "--seeds", s, "--eps1", "1e-14",
                                                "--eps_diag", "--tasks", 200, "--out", R2R / "R_eps_zero_v2", "--tag", f"{act}_eps1e-14"],
                         cwd=S4 / "valley"))
    return J


def main():
    J = [j for j in jobs() if not (j["out"] / ".done").exists()]
    print(f"{len(J)} round-4 jobs", flush=True)
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
    print("round-4 queue finished" if not J else f"stopped with {len(J)} left", flush=True)


if __name__ == "__main__":
    main()
