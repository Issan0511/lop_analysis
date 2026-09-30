#!/usr/bin/env python3
"""sink_roots_0930 round 6 (spec_sink_roots_0930_round6.md): the two-layer replays, G6, K_sweep and RR4.
A job starts when its dependencies' `.done` markers exist and the number of my experiment processes on the machine
(python processes running a script under this worktree's src/, i.e. R7 and every round; paused ones not counted) is below the budget
`total` of results/sink_roots_0930/caps_round2.json (read every loop).  `.done` is written on rc 0.
results/sink_roots_0930/STOP_round6 stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R2R = RAW / "round6"
RES = ROOT / "results" / "sink_roots_0930"
LOG = RES / "launch_round6_log.jsonl"
STOP = RES / "STOP_round6"
CAPS = RES / "caps_round2.json"
PY = sys.executable
SRC = ROOT / "src"
S6 = SRC / "sink_roots_round6"
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
    ST = RAW / "round1" / "sign_states"
    TP = [PY, S6 / "twolayer_probe.py"]
    for f in ("L2_ELU_K10_s0_t002", "L2_ELU_K10_s0_t003", "L2_ELU_K10_s1_t002", "L2_ELU_K10_s1_t003", "L2_LR_K10_s0_t002", "L2_SILU_K10_s0_t002"):
        J.append(job(f"RR1_{f}", TP + ["rr1", ST / f"{f}.npz", "--out", R2R / "RR1"], cwd=S6))
    for f in ("L2_ELU_K10_s0_t002", "L2_ELU_K10_s0_t003", "L2_ELU_K10_s1_t002", "L2_ELU_K10_s1_t003"):
        J.append(job(f"RR2_{f}", TP + ["rr2", ST / f"{f}.npz", "--out", R2R / "RR2"], cwd=S6))
    for act in ("ELU", "LR", "SILU"):
        for s in (0, 1):
            fs = [ST / f"L2_{act}_K10_s{s}_t{t:03d}.npz" for t in (2, 3, 4, 6)]
            J.append(job(f"RR3_{act}_s{s}", TP + ["rr3"] + fs + ["--out", R2R / "RR3"], cwd=S6))
    for act in ("ELU", "LR", "SILU"):
        for t in (2, 3, 6):
            f = f"L2_{act}_K10_s0_t{t:03d}"
            J.append(job(f"RR4_{f}", TP + ["rr4", ST / f"{f}.npz", "--out", R2R / "RR4"], cwd=S6))
    for s in (0, 1, 2):
        J.append(job(f"G6_ELU_s{s}", [PY, S6 / "growth" / "replay_geom.py", "--act", "ELU", "--seed", s, "--arm", "wcap", "--t0", 5,
                                      "--t1", 30, "--out", R2R / "G6" / f"ELU_s{s}"], cwd=S6 / "growth"))
    for K in (2, 5, 10, 20):
        for s in (0, 1):
            J.append(job(f"KS_GELU_K{K}_s{s}", [PY, S6 / "valley" / "alpha_ladder_1layer.py", "--act", "GELU", "--seeds", s, "--K", K,
                                                 "--tasks", 100, "--out", R2R / "K_sweep", "--tag", f"GELU_K{K}"], cwd=S6 / "valley"))
    def eng(n, args):
        return job(n, ENG + ["--name", n, "--out", RAW / "mnist" / n] + [str(x) for x in args], out=RAW / "mnist" / n)
    for arm, extra in (("L1cap", ["--wcap-from", 5]), ("L2cap", ["--wcap2-from", 5]), ("L12cap", ["--wcap-from", 5, "--wcap2-from", 5])):
        for s in (0, 1, 2):
            J.append(eng(f"RR4tl_{arm}_ELU_s{s}", ["--act", "ELU", "--seed", s, "--layers", 2, "--T", 6000, "--tasks", 20]
                         + extra + ["--snap-before"] + list(range(5, 21))))
    return J


def main():
    J = [j for j in jobs() if not (j["out"] / ".done").exists()]
    print(f"{len(J)} round-6 jobs", flush=True)
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
    print("round-6 queue finished" if not J else f"stopped with {len(J)} left", flush=True)


if __name__ == "__main__":
    main()
