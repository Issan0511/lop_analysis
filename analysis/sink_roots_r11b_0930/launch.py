#!/usr/bin/env python3
"""sink_roots_r11b_0930 main runs (spec_sink_roots_r11b_0930.md §2): 6 arms x seeds 0-2 x 30 tasks, one process per run.
At most `total` of results/sink_roots_r11b_0930/caps.json run at once (read every loop); `.done` is written on rc 0;
results/sink_roots_r11b_0930/STOP stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_r11b_0930")
RES = ROOT / "results" / "sink_roots_r11b_0930"
ARMS = ("ELU", "GELU", "SILU", "LR", "R", "KKT1")


def budget():
    try:
        return int(json.loads((RES / "caps.json").read_text())["total"])
    except Exception:
        return 4


def main():
    jobs = [(a, s) for s in (0, 1, 2) for a in ARMS]
    jobs = [j for j in jobs if not (RAW / f"{j[0]}_s{j[1]}" / ".done").exists()]
    (RAW / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while jobs or running:
        for key, (p, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                fh.close()
                if rc == 0:
                    (RAW / f"{key[0]}_s{key[1]}" / ".done").write_text(f"{time.time()}\n")
                with (RES / "launch_log.jsonl").open("a") as f:
                    f.write(json.dumps({"arm": key[0], "seed": key[1], "rc": rc, "start": t0, "end": time.time()}) + "\n")
                del running[key]
        while jobs and len(running) < budget() and not (RES / "STOP").exists():
            a, s = jobs.pop(0)
            out = RAW / f"{a}_s{s}"
            out.mkdir(parents=True, exist_ok=True)
            fh = open(RAW / "logs" / f"{a}_s{s}.log", "w")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.Popen([sys.executable, str(ROOT / "src" / "sink_roots_c5p1_0930.py"), "--arm", a, "--seed", str(s),
                                  "--out", str(out)], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT, env=env)
            running[(a, s)] = (p, time.time(), fh)
        if (RES / "STOP").exists() and not running:
            break
        time.sleep(5)
    print("queue finished" if not jobs else f"stopped with {len(jobs)} left", flush=True)


if __name__ == "__main__":
    main()
