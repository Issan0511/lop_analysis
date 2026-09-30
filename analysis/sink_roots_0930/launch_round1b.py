#!/usr/bin/env python3
"""sink_roots_0930 round 1b (spec_sink_roots_0930_round1b.md): the W1-cap follow-ups, at most MAX jobs at once.  A job is done when its `.done` marker exists (written on rc 0).
`results/sink_roots_0930/STOP_round1` stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R1 = RAW / "round1"
RES = ROOT / "results" / "sink_roots_0930"
LOG = RES / "launch_round1b_log.jsonl"
STOP = RES / "STOP_round1b"
MAX = 2
PY = sys.executable
SR = ROOT / "src" / "sink_roots_round1"
ENG = [PY, str(ROOT / "src" / "sink_roots_mnist_0930.py")]


def job(name, cmd, out, env=None):
    return {"name": name, "cmd": cmd, "out": Path(out), "env": env or {}}


def jobs():
    J = []
    def eng(n, args):
        return job(n, ENG + ["--name", n, "--out", str(RAW / "mnist" / n)] + args, RAW / "mnist" / n)
    for act in ("GELU", "LR"):                                             # 1
        for s in (0, 1, 2):
            J.append(eng(f"R1b1_wcap_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30", "--wcap-from", "5"]))
    for mode in ("par", "perp"):                                           # 3
        for act in ("ELU", "SILU"):
            for s in (0, 1, 2):
                J.append(eng(f"R1b3_{mode}_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30", "--wcap-from", "5",
                                                         "--wcap-mode", mode]))
    for sc in ("0.5", "2", "4"):                                           # 4
        for s in (0, 1, 2):
            J.append(eng(f"R1b4_scale{sc}_ELU_s{s}", ["--act", "ELU", "--seed", str(s), "--tasks", "30", "--wcap-from", "5",
                                                      "--wcap-scale", sc]))
    for act in ("ELU", "SILU"):                                            # 2 (long; last so the short ones report first)
        for s in (0, 1, 2):
            J.append(eng(f"R1b2_long_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "200", "--wcap-from", "5"]))
    return J


def main():
    J = [j for j in jobs() if not (j["out"] / ".done").exists()]
    print(f"{len(J)} round-1 jobs", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
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
            while J and len(running) < MAX:
                j = J.pop(0)
                if j["name"] == "F1_s0_p23" and not (R1 / "F1_dry" / "job" / ".done").exists():
                    J.insert(0, j)                                        # the dry run gates the F1 mains
                    break
                j["out"].mkdir(parents=True, exist_ok=True)
                fh = open(j["out"] / "stdout.log", "w")
                env = dict(os.environ, **j["env"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=ROOT, env=env)
                running[j["name"]] = (p, j, time.time(), fh)
        elif not running:
            break
        time.sleep(5)
    print("round-1 queue finished" if not J else f"stopped with {len(J)} left", flush=True)


if __name__ == "__main__":
    main()
