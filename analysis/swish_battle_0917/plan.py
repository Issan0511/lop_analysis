#!/usr/bin/env python3
"""Write the job plan for launch.py (spec §3).  The live copy under _launch/ is re-read by
the launcher every cycle, so the caps can be edited while it runs.

cnn (cuda): SNAc3, SWA1, SW1, SWA3 interleaved by seed, then SW3 (lowest priority).
mlp (cpu, 1 thread): all nine arms interleaved by seed.
rss_gb is the measured peak (cnn cuda 2.03 GB, mlp cpu 1.05 GB) rounded up.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CNN_CORE = ("SNAc3", "SWA1", "SW1", "SWA3")
CNN_LAST = ("SW3",)
MLP = ("SWA1", "SW1", "SWA3", "SW3", "SNA", "LR", "SNAc3", "SN3", "SN06")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-max", type=int, default=3)
    ap.add_argument("--cpu-threads-max", type=int, default=3)
    ap.add_argument("--reserve-gb", type=float, default=6.0)
    ap.add_argument("--out", default=str(REPO / "results/swish_battle_0917/_launch/plan.json"))
    a = ap.parse_args()
    cnn = [dict(box="cnn", arm=x, seed=s, device="cuda", threads=1, rss_gb=2.1)
           for s in range(10) for x in CNN_CORE]
    cnn += [dict(box="cnn", arm=x, seed=s, device="cuda", threads=1, rss_gb=2.1)
            for x in CNN_LAST for s in range(10)]
    mlp = [dict(box="mlp", arm=x, seed=s, device="cpu", threads=1, rss_gb=1.1)
           for s in range(10) for x in MLP]
    plan = {"gpu_max": a.gpu_max, "cpu_threads_max": a.cpu_threads_max,
            "reserve_gb": a.reserve_gb, "jobs": cnn + mlp}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"{len(cnn)} cnn + {len(mlp)} mlp jobs -> {a.out}")


if __name__ == "__main__":
    main()
