#!/usr/bin/env python3
"""Write the job plan for launch.py (spec §3).  The live copy under _launch/ is re-read by
the launcher every cycle, so the caps can be edited while it runs.

cnn (cuda): SNAc3 first (addendum 3: label A needs only it), then SWA1, SW1, SWA3,
            SWA1u, SWA3u, SW3 interleaved by seed.
mlp (cpu, 1 thread): all nine arms interleaved by seed.
rss_gb is the measured peak (cnn cuda 2.2 GB over a full run, mlp cpu 1.05 GB) rounded up.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CNN_FIRST = ("SNAc3",)                                  # addendum 3: label A first
CNN_SWISH = ("SWA1", "SW1", "SWA3", "SWA1u", "SWA3u", "SW3")
MLP = ("SWA1", "SW1", "SWA3", "SW3", "SNA", "LR", "SNAc3", "SN3", "SN06")
MLP_U = ("SWA1u", "SWA3u")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-max", type=int, default=3)
    ap.add_argument("--cpu-threads-max", type=int, default=3)
    ap.add_argument("--reserve-gb", type=float, default=6.0)
    ap.add_argument("--out", default=str(REPO / "results/swish_battle_0917/_launch/plan.json"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cnn = [dict(box="cnn", arm=x, seed=s, device="cuda", threads=1, rss_gb=2.2)
           for x in CNN_FIRST for s in range(10)]
    cnn += [dict(box="cnn", arm=x, seed=s, device="cuda", threads=1, rss_gb=2.2)
            for s in range(10) for x in CNN_SWISH]
    mlp = [dict(box="mlp", arm=x, seed=s, device="cpu", threads=1, rss_gb=1.1)
           for s in range(10) for x in MLP]
    # spec addendum 2: floor-lowered adaptive Swish, after the registered mlp arms
    mlp += [dict(box="mlp", arm=x, seed=s, device="cpu", threads=1, rss_gb=1.1)
            for s in range(10) for x in MLP_U]
    plan = {"gpu_max": a.gpu_max, "cpu_threads_max": a.cpu_threads_max,
            "reserve_gb": a.reserve_gb, "jobs": cnn + mlp}
    if Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists (the live plan); --force to overwrite")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"{len(cnn)} cnn + {len(mlp)} mlp jobs -> {a.out}")


if __name__ == "__main__":
    main()
